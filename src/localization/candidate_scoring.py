"""Candidate scoring (§19 / Stage 16): combine multiple evidence types into
one interpretable score per candidate.

    total = w_visual*visual + w_semantic*semantic + w_temporal*temporal
            + w_graph*graph

All terms normalized to [0, 1]. Missing metadata yields neutral terms (0.5),
never a hard zero — absence of evidence is not evidence of absence.

Stage 23 (planner v2 §6): the w_semantic slot is filled by
src.localization.semantic_scoring.semantic_similarity (scene + landmark +
object evidence) — previously a placeholder scene/landmark-only term.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.localization.retrieval import Candidate, RetrievalResult
from src.localization.semantic_scoring import semantic_similarity
from src.mapping.place_builder import Place
from src.utils import setup_logger

logger = setup_logger("candidate_scoring")


@dataclass
class ScoredCandidate(Candidate):
    visual_term: float = 0.0
    semantic_term: float = 0.0
    temporal_term: float = 0.0
    graph_term: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "visual_term": round(float(self.visual_term), 4),
            "semantic_term": round(float(self.semantic_term), 4),
            "temporal_term": round(float(self.temporal_term), 4),
            "graph_term": round(float(self.graph_term), 4),
            "total": round(float(self.total), 4),
        })
        return d


def landmark_jaccard(list_a: list[str], list_b: list[str]) -> float:
    if not list_a and not list_b:
        return 0.0
    if not list_a or not list_b:
        return 0.0
    set_a, set_b = set(list_a), set(list_b)
    return len(set_a & set_b) / len(set_a | set_b)


def semantic_term(query_scene: str | None, query_landmarks: list[str],
                  place_scene: str, place_landmarks: list[str]) -> float:
    """Legacy Stage-16 placeholder (scene + landmarks only). Superseded by
    semantic_similarity (Stage 23) in the scoring path; kept for its tests
    and as a reference for the original term shape. 0.5 = neutral when the
    query carries no semantic evidence."""
    if query_scene is None or query_scene == "unknown":
        return 0.5
    scene = 1.0 if query_scene == place_scene else (0.5 if place_scene == "unknown" else 0.0)
    lm = landmark_jaccard(query_landmarks or [], place_landmarks or [])
    return 0.5 * scene + 0.5 * lm


def score_candidates(
    result: RetrievalResult,
    query_tags: dict | None,
    query_objects: list[dict] | None,
    previous_place_id: int | None,
    graph_terms: dict[int, float] | None,
    places: dict[int, Place] | None,
    config: dict,
) -> list[ScoredCandidate]:
    """graph_terms: place_id -> graph term in [0,1] (computed by
    src.localization.graph_constraints, Stage 17). None -> neutral 0.5.
    places: place_id -> Place records (for semantic evidence); None -> neutral."""
    w = {
        "w_visual": float(config.get("w_visual", 0.5)),
        "w_semantic": float(config.get("w_semantic", 0.25)),
        "w_temporal": float(config.get("w_temporal", 0.15)),
        "w_graph": float(config.get("w_graph", 0.1)),
    }
    total_w = sum(w.values())
    if total_w <= 0:
        w = {k: 0.25 for k in w}

    scored: list[ScoredCandidate] = []
    for cand in result.candidates:
        place = graph_terms and graph_terms.get(cand.place_id)
        gt = place if place is not None else 0.5
        place_rec = places.get(cand.place_id) if places else None
        if place_rec is not None:
            place_scene = (
                place_rec.scene_types.most_common(1)[0][0]
                if place_rec.scene_types else "unknown"
            )
            place_tags = {"scene_type": place_scene, "landmarks": place_rec.landmarks}
            place_objects = getattr(place_rec, "object_classes", None)
        else:
            place_tags = None
            place_objects = None
        # Stage 23: real semantic evidence (scene + landmarks + objects),
        # neutral 0.5 when the query carries nothing (Rule 4)
        s = semantic_similarity(query_tags, query_objects, place_tags, place_objects)
        temporal = 1.0 if previous_place_id is not None and cand.place_id == previous_place_id else 0.5
        total = (
            w["w_visual"] * cand.visual_score
            + w["w_semantic"] * s
            + w["w_temporal"] * temporal
            + w["w_graph"] * gt
        ) / sum(w.values())
        scored.append(
            ScoredCandidate(
                place_id=cand.place_id, visual_score=cand.visual_score,
                best_exemplar_id=cand.best_exemplar_id,
                supporting_exemplar_count=cand.supporting_exemplar_count,
                margin=cand.margin,
                visual_term=cand.visual_score,
                semantic_term=s, temporal_term=temporal, graph_term=gt,
                total=total,
            )
        )
    scored.sort(key=lambda c: c.total, reverse=True)
    return scored
