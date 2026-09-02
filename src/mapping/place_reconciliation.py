"""Place reconciliation (§12): merge duplicate places (revisits) while
keeping visually similar but physically distinct places separate.

Merge requires MULTIPLE signals: visual similarity + landmark similarity +
scene compatibility + temporal/context compatibility (§12 evidence for
merging). Any of several hard conflicts rejects the merge (§12 evidence
against merging). The process is a deterministic fixed-point iteration and
logs every merge to reconciliation_log.json.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.mapping.observations import Observation
from src.mapping.place_builder import DISCRIMINATIVE_LANDMARK_TYPES, Place
from src.perception.scene_tagger import normalize_scene_type
from src.utils import setup_logger

logger = setup_logger("reconciliation")

# Scenes whose places form long, visually continuous regions. Generic
# closed-vocab landmarks (door, direction_sign, ...) repeat in every such
# place, so corridor merging is gated on DISCRIMINATING evidence only.
CORRIDOR_SCENES = {"corridor", "junction"}


def _dominant_scene(place: Place) -> str:
    return (
        normalize_scene_type(place.scene_types.most_common(1)[0][0])
        if place.scene_types else "unknown"
    )


def _discriminating_evidence(place: Place) -> set[str]:
    """Identity-bearing merge evidence: discriminative landmark tokens
    (reception/desk/elevator/stairs/entrance/room_sign) plus the readable
    sign text seen in the place (room numbers etc.). Generic wall fixtures
    (door, direction_sign, fire_equipment, ...) never count."""
    tokens = {
        lm for lm in (place.landmarks or []) if lm in DISCRIMINATIVE_LANDMARK_TYPES
    }
    return tokens | set(place.sign_texts or [])


def _set_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cap_exemplars(exemplar_ids: list[str], max_n: int) -> list[str]:
    """Deterministic temporal spread when a merged place exceeds the exemplar
    cap — prevents best-of-any-exemplar similarity from inflating as places
    grow (unbounded exemplars let one distant view bridge distinct regions)."""
    if max_n <= 0 or len(exemplar_ids) <= max_n:
        return exemplar_ids
    idx = np.linspace(0, len(exemplar_ids) - 1, max_n).round().astype(int)
    return [exemplar_ids[i] for i in idx]


@dataclass
class MergeDecision:
    a: str
    b: str
    visual_similarity: float
    landmark_jaccard: float
    scene_compatible: bool
    context_compatible: bool
    merged: bool
    reasons: list[str] = field(default_factory=list)


def _place_exemplar_embeddings(place: Place, by_id: dict[str, Observation]) -> np.ndarray | None:
    vecs = [by_id[eid].embedding for eid in place.exemplar_ids if eid in by_id and by_id[eid].embedding is not None]
    return np.stack(vecs) if vecs else None


def _visual_similarity(a: Place, b: Place, by_id: dict[str, Observation]) -> float:
    """Best pairwise exemplar cosine similarity between two places."""
    va = _place_exemplar_embeddings(a, by_id)
    vb = _place_exemplar_embeddings(b, by_id)
    if va is None or vb is None:
        return 0.0
    sims = va @ vb.T
    return float(sims.max())


def _landmark_jaccard(a: Place, b: Place) -> float:
    if not a.landmarks and not b.landmarks:
        return 0.0  # no evidence either way — neutral, not a merge signal
    if not a.landmarks or not b.landmarks:
        return 0.0
    set_a, set_b = set(a.landmarks), set(b.landmarks)
    return len(set_a & set_b) / len(set_a | set_b)


def _scene_compatible(a: Place, b: Place) -> bool:
    """Most-common scene types are compatible (equal, or one side unknown)."""
    sa = normalize_scene_type(a.scene_types.most_common(1)[0][0]) if a.scene_types else "unknown"
    sb = normalize_scene_type(b.scene_types.most_common(1)[0][0]) if b.scene_types else "unknown"
    if sa == "unknown" or sb == "unknown":
        return True
    if sa == sb:
        return True
    # corridor <-> junction is compatible (a junction IS a corridor)
    return {sa, sb} == {"corridor", "junction"}


def _context_compatible(a: Place, b: Place, places: list[Place]) -> bool:
    """Revisit pattern: place b sits between observations of place a in the
    walkthrough (a's segments occur both before and after b's). Requires the
    place order in the source walkthrough, which the segment ids encode
    lexicographically (seg_000 < seg_001 < ...)."""
    seg_order = [p.segment_ids[0] for p in places]  # places built in walkthrough order
    if len(seg_order) < 3 or a.place_id == b.place_id:
        return False
    idx_a = seg_order.index(a.segment_ids[0])
    idx_b = seg_order.index(b.segment_ids[0])
    # b is "between" two visits of a? That would need a's segment both before
    # and after b. With one segment per place (pre-reconciliation), detect the
    # revisit signal differently: a and b share NO temporal adjacency, i.e.
    # they are separated by other places, and are not direct neighbors.
    return abs(idx_a - idx_b) > 1


def decide_merge(
    a: Place,
    b: Place,
    places: list[Place],
    by_id: dict[str, Observation],
    config: dict,
) -> MergeDecision:
    vis = _visual_similarity(a, b, by_id)
    lm = _landmark_jaccard(a, b)
    scene_ok = _scene_compatible(a, b)
    ctx_ok = _context_compatible(a, b, places)

    v_thr = float(config.get("merge_visual_threshold", 0.75))
    extra_thr = float(config.get("merge_visual_extra_threshold", 0.92))
    lm_thr = float(config.get("merge_landmark_threshold", 0.5))
    scene_required = bool(config.get("merge_scene_required", True))

    reasons: list[str] = []
    visual_strong = vis >= v_thr
    scene_ok_required = (not scene_required) or scene_ok

    # -- corridor pair: scene-aware merge guard (scene tagger v2 schema) --
    # Generic closed-vocab landmarks (door, direction_sign, ...) appear in
    # every corridor, so corridor↔corridor merging requires the calibrated
    # near-identical visual floor AND shared DISCRIMINATING evidence
    # (reception/desk/elevator/stairs/entrance/room_sign tokens or readable
    # sign text). Disjoint discriminators = different places (identity
    # conflict, v1-style). This keeps a chain of visually similar but
    # distinct corridor regions from collapsing into one mega-place.
    corridor_pair = _dominant_scene(a) in CORRIDOR_SCENES and _dominant_scene(b) in CORRIDOR_SCENES
    if corridor_pair:
        da, db = _discriminating_evidence(a), _discriminating_evidence(b)
        disc_j = _set_jaccard(da, db)
        scene_conflict = scene_required and not scene_ok and visual_strong
        # Corridor merge evidence comes in two forms:
        #  - shared identity evidence on a genuinely-same-view floor
        #    (>= extra_thr — the calibrated "same view" threshold), OR
        #  - a near-exact same view with no identity evidence on either side
        #    (>= 0.95: a same-pose revisit of a signless stretch — the
        #    corridor-flicker split case, cos=1.0 on identical frames). Below
        #    0.95 is a different view of the corridor and needs real identity
        #    evidence, so visually-similar-but-distinct corridor regions can
        #    never chain into one mega-place.
        no_disc = not da and not db
        disc_strong = disc_j >= lm_thr and vis >= extra_thr
        exact_view = no_disc and vis >= 0.95
        corridor_evidence = disc_strong or exact_view
        landmark_conflict = bool(da) and bool(db) and not (da & db) and vis >= v_thr - 0.15

        if disc_strong:
            reasons.append("landmarks")
        if exact_view:
            reasons.append("visual_near_identical")
        if visual_strong:
            reasons.append("visual")
        if ctx_ok and visual_strong:
            reasons.append("context")
        if scene_ok:
            reasons.append("scene")
        if landmark_conflict:
            reasons.append("landmark_conflict")
        if scene_conflict:
            reasons.append("scene_conflict")

        merged = (
            corridor_evidence
            and scene_ok_required
            and not landmark_conflict
            and not scene_conflict
        )
        return MergeDecision(a.place_id, b.place_id, vis, lm, scene_ok, ctx_ok, merged, reasons)

    # -- room / special / mixed pairs: original evidence rules unchanged --
    landmark_strong = lm >= lm_thr
    landmarks_both_empty = not a.landmarks and not b.landmarks

    if visual_strong:
        reasons.append("visual")
    if landmark_strong:
        reasons.append("landmarks")
    if landmarks_both_empty and visual_strong:
        reasons.append("landmarks_neutral")
    if ctx_ok and visual_strong:
        reasons.append("context")
    if scene_ok:
        reasons.append("scene")

    landmark_conflict = bool(a.landmarks) and bool(b.landmarks) and lm < 1e-9 and vis >= v_thr - 0.15
    scene_conflict = scene_required and not scene_ok and visual_strong
    if landmark_conflict:
        reasons.append("landmark_conflict")
    if scene_conflict:
        reasons.append("scene_conflict")

    # merge = strong visual + scene compatible + at least one EXTRA signal
    # (shared landmarks, or near-identical visual when neither place has
    # landmark evidence — 0.92: typical cross-area ResNet18 similarity in the
    # same building is ~0.87, so 0.92 means genuinely the same view) — and no
    # hard conflict. Revisit context alone is never sufficient (weak signal).
    extra_signal = landmark_strong or (landmarks_both_empty and vis >= extra_thr)
    merged = (
        visual_strong
        and scene_ok_required
        and not landmark_conflict
        and not scene_conflict
        and extra_signal
    )
    return MergeDecision(a.place_id, b.place_id, vis, lm, scene_ok, ctx_ok, merged, reasons)


def _merge_places(a: Place, b: Place, max_exemplars: int | None = None) -> Place:
    merged = Place(
        place_id=min(a.place_id, b.place_id),
        segment_ids=sorted(a.segment_ids + b.segment_ids),
        observation_ids=sorted(a.observation_ids + b.observation_ids),
        exemplar_ids=a.exemplar_ids + b.exemplar_ids,
        scene_types=a.scene_types + b.scene_types,
        landmarks=_top_landmarks(a.landmarks + b.landmarks),
        sign_texts=_top_landmarks(a.sign_texts + b.sign_texts),
        walkable_directions=a.walkable_directions + b.walkable_directions,
        object_classes=_top_landmarks(a.object_classes + b.object_classes),
        visual_stats={**a.visual_stats, **b.visual_stats},
    )
    if max_exemplars is not None:
        merged.exemplar_ids = _cap_exemplars(merged.exemplar_ids, max_exemplars)
    return merged


def _top_landmarks(landmarks: list[str], limit: int = 8) -> list[str]:
    counts = Counter(landmarks)
    return [lm for lm, _ in counts.most_common(limit)]


def reconcile_places(
    places: list[Place],
    observations: list[Observation],
    config: dict,
    log_path=None,
) -> list[Place]:
    """Merge duplicate places to a fixed point. Deterministic; logs every
    merge decision to log_path (JSONL) when provided."""
    by_id = {o.id: o for o in observations}
    log_rows: list[dict] = []

    current = list(places)
    changed = True
    iteration = 0
    while changed and iteration < 10:
        changed = False
        iteration += 1
        i = 0
        while i < len(current):
            j = i + 1
            while j < len(current):
                decision = decide_merge(current[i], current[j], current, by_id, config)
                log_rows.append(
                    {
                        "a": decision.a,
                        "b": decision.b,
                        "visual_similarity": round(decision.visual_similarity, 4),
                        "landmark_jaccard": round(decision.landmark_jaccard, 4),
                        "scene_compatible": decision.scene_compatible,
                        "context_compatible": decision.context_compatible,
                        "merged": decision.merged,
                        "reasons": decision.reasons,
                    }
                )
                if decision.merged:
                    merged_place = _merge_places(
                        current[i], current[j],
                        max_exemplars=int(config.get("max_exemplars_per_place", 3)),
                    )
                    current = [current[k] for k in range(len(current)) if k not in (i, j)]
                    current.insert(i, merged_place)
                    logger.info(
                        f"Merged {decision.a} + {decision.b} -> {merged_place.place_id} "
                        f"(visual={decision.visual_similarity:.2f}, landmarks={decision.landmark_jaccard:.2f})"
                    )
                    changed = True
                    # restart the sweep at the merged place
                    j = i + 1
                else:
                    j += 1
            i += 1

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for row in log_rows:
                f.write(json.dumps(row) + "\n")

    logger.info(
        f"Reconciliation: {len(places)} -> {len(current)} places "
        f"({len([r for r in log_rows if r['merged']])} merges in {iteration} passes)"
    )
    return current
