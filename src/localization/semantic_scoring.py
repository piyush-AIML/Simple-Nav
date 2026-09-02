"""Semantic evidence scoring (Stage 23, planner v2 §6) — combine detector +
VLM evidence to break visual ties between appearance-similar places.

    semantic_similarity = 0.4*scene_match + 0.4*landmark_jaccard + 0.2*object_jaccard

Rule-4-aware asymmetries (unknown is better than confidently wrong):

- QUERY-side missing/unknown evidence is NEUTRAL (0.5): a tagging failure on
  the live frame must never actively penalize a place (Rule 4).
- PLACE-side absence with a non-empty query is real evidence of absence
  (0.0): the place has genuinely never shown those landmarks/objects.
- Both sides empty -> 0.5 (empty-vs-empty is neutral, not 0 and not 1).
- Disjoint non-empty sets floor at 0.1 — the score is never exactly 0.
- Stub-tagger mode (no tag info on either side) degrades to object-overlap
  only, so vlm_enabled: false keeps working without double-counting through
  the visual term.
- No query evidence at all -> exactly 0.5 (neutral).

Never raises: bad input yields the neutral path, never a crash.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.perception.scene_tagger import normalize_scene_type


def _scene(tags: Any) -> str:
    """scene_type of one tag record (SceneTags, dict, or None), normalized
    onto the current vocabulary (legacy 'corridor_junction' == 'junction')."""
    if tags is None:
        return "unknown"
    s = tags.get("scene_type") if isinstance(tags, dict) else getattr(tags, "scene_type", None)
    return normalize_scene_type(str(s)) if s else "unknown"


def _landmarks(tags: Any) -> set[str]:
    """Landmarks of one tag record, lower-cased and stripped (planner v2 §6.2)."""
    if tags is None:
        return set()
    lms = tags.get("landmarks", []) if isinstance(tags, dict) else getattr(tags, "landmarks", None)
    if not lms:
        return set()
    return {str(l).strip().lower() for l in lms if l}


def _has_tag_info(tags: Any) -> bool:
    """True when a tag record carries real information (known scene or any
    landmark). Stub-tagger output (unknown scene + empty landmarks) is 'no
    information' — it must not be treated as evidence."""
    if _scene(tags) != "unknown":
        return True
    return bool(_landmarks(tags))


def _object_classes(objects: Any) -> set[str]:
    """COCO class names of DetectedObjects / dicts / plain strings."""
    out: set[str] = set()
    for o in (objects or []):
        if isinstance(o, dict):
            c = o.get("class") or o.get("class_name")
        elif isinstance(o, str):
            c = o
        else:
            c = getattr(o, "class_name", None)
        if c:
            out.add(str(c).strip().lower())
    return out


def _dominant_scene(place_tags: list[Any]) -> str:
    """Mode of scene types across the place's stored exemplar tags."""
    if not place_tags:
        return "unknown"
    return Counter(_scene(t) for t in place_tags).most_common(1)[0][0]


def _scene_match(query_scene: str, place_scene: str) -> float:
    """1.0 exact match; 0.5 if either side is unknown; else 0.0.

    Planner v3 §9 re-tune: the old 0.3 penalized the PLACE side for
    'unknown', but on this dataset 46% of observations have scene
    'unknown' because tagging failed — absence of stored evidence is not
    evidence of absence (the same Rule 4 that protects the query side).
    Unknown now abstains (0.5) instead of voting against."""
    if query_scene == place_scene and query_scene != "unknown":
        return 1.0
    if query_scene == "unknown" or place_scene == "unknown":
        return 0.5
    return 0.0


def _jaccard_rule4(query: set[str], place: set[str]) -> float:
    """Jaccard with the Rule-4 asymmetries documented at module top."""
    if not query:
        return 0.5
    if not place:
        return 0.0
    j = len(query & place) / len(query | place)
    return max(j, 0.1)


def semantic_similarity(
    query_tags: Any = None,
    query_objects: list | None = None,
    place_tags: Any = None,
    place_object_classes: list | None = None,
) -> float:
    """Returns a score in [0, 1]. Never raises; missing/unknown tags -> 0.5
    (neutral — never actively penalizes a place just because tagging failed,
    per Rule 4)."""
    try:
        if place_tags is None:
            place_tags_list: list[Any] = []
        elif isinstance(place_tags, (list, tuple)):
            place_tags_list = list(place_tags)
        else:
            place_tags_list = [place_tags]

        q_objects = _object_classes(query_objects)
        p_objects = _object_classes(place_object_classes)

        if not _has_tag_info(query_tags) and not q_objects:
            return 0.5  # no query evidence at all — neutral (Rule 4)

        place_has_info = any(_has_tag_info(t) for t in place_tags_list)
        if not _has_tag_info(query_tags) and not place_has_info:
            # stub-tagger mode on both sides: object overlap only
            return _jaccard_rule4(q_objects, p_objects)

        p_lms: set[str] = set()
        for t in place_tags_list:
            p_lms |= _landmarks(t)

        scene_match = _scene_match(_scene(query_tags), _dominant_scene(place_tags_list))
        lm_term = _jaccard_rule4(_landmarks(query_tags), p_lms)
        obj_term = _jaccard_rule4(q_objects, p_objects)
        return 0.4 * scene_match + 0.4 * lm_term + 0.2 * obj_term
    except Exception:  # never raises — degrade to neutral
        return 0.5
