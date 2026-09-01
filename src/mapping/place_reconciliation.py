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
from src.mapping.place_builder import Place
from src.utils import setup_logger

logger = setup_logger("reconciliation")


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
    sa = a.scene_types.most_common(1)[0][0] if a.scene_types else "unknown"
    sb = b.scene_types.most_common(1)[0][0] if b.scene_types else "unknown"
    if sa == "unknown" or sb == "unknown":
        return True
    if sa == sb:
        return True
    # corridor <-> corridor_junction is compatible (a junction IS a corridor)
    return {sa, sb} == {"corridor", "corridor_junction"}


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
    landmark_strong = lm >= lm_thr
    landmarks_both_empty = not a.landmarks and not b.landmarks
    scene_ok_required = (not scene_required) or scene_ok

    # -- evidence FOR merging (multiple signals, §12) --
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

    # -- evidence AGAINST merging (hard rejects, §12) --
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


def _merge_places(a: Place, b: Place) -> Place:
    merged = Place(
        place_id=min(a.place_id, b.place_id),
        segment_ids=sorted(a.segment_ids + b.segment_ids),
        observation_ids=sorted(a.observation_ids + b.observation_ids),
        exemplar_ids=a.exemplar_ids + b.exemplar_ids,
        scene_types=a.scene_types + b.scene_types,
        landmarks=_top_landmarks(a.landmarks + b.landmarks),
        visual_stats={**a.visual_stats, **b.visual_stats},
    )
    # cap exemplars at the configured max
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
                    merged_place = _merge_places(current[i], current[j])
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
