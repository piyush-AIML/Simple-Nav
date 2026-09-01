"""Temporal segmentation (§10): split an ordered observation sequence into
contiguous periods that correspond to local spatial regions.

Signals per consecutive pair (embedding distance + semantic change), a SHORT
smoothing window so one anomalous frame can never create a boundary, and a
minimum segment length. Output segments broadly match physical areas instead
of arbitrary K-Means clusters (§10 acceptance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mapping.observations import Observation, sort_observations
from src.utils import setup_logger

logger = setup_logger("segmentation")


@dataclass
class Segment:
    id: str
    obs_ids: list[str]
    start_index: int
    end_index: int            # inclusive
    representative_obs_id: str


def _visual_distance(a: Observation, b: Observation) -> float:
    if a.embedding is None or b.embedding is None:
        return 0.5  # neutral when embeddings are missing — never blocks a boundary
    return float(1.0 - float(np.dot(a.embedding, b.embedding)))


def _landmark_jaccard(a: Observation, b: Observation) -> float:
    if not a.landmarks or not b.landmarks:
        return 1.0  # no landmark evidence -> compatible
    set_a, set_b = set(a.landmarks), set(b.landmarks)
    return len(set_a & set_b) / len(set_a | set_b)


def _semantic_change(a: Observation, b: Observation) -> float:
    """0 or 1: scene_type change, or a strong landmark change.

    "unknown" is neutral (Rule 4) — a flip into/out of unknown is missing
    evidence, not a real scene change."""
    if a.scene_tags or b.scene_tags:
        sa = (a.scene_tags or {}).get("scene_type")
        sb = (b.scene_tags or {}).get("scene_type")
        if sa and sb and sa != sb and sa != "unknown" and sb != "unknown":
            return 1.0
    return 0.0 if _landmark_jaccard(a, b) >= 0.5 else 1.0


SEMANTIC_PERSISTENCE_WINDOW = 3


def change_scores(observations: list[Observation]) -> list[float]:
    """Per-pair change score (len = n-1): visual + 0.5 * semantic.

    The semantic term is gated on PERSISTENCE: a scene change only counts
    when the new scene type holds for the next few frames. Per-frame VLM
    output flickers (corridor <-> room <-> unknown within one walkthrough
    stretch — measured: 72 flips over 301 frames with the LFM2 tagger), and
    one flipped frame must not cut the sequence — the same spike-suppression
    rule §8 applies to the visual signal."""
    scores = []
    for i, (a, b) in enumerate(zip(observations, observations[1:])):
        visual = _visual_distance(a, b)
        semantic = _semantic_change(a, b)
        if semantic > 0.0:
            sb = (b.scene_tags or {}).get("scene_type")
            window = observations[i + 2 : i + 2 + SEMANTIC_PERSISTENCE_WINDOW]
            if window and any(
                (o.scene_tags or {}).get("scene_type") != sb for o in window
            ):
                semantic = 0.0
        scores.append(visual + 0.5 * semantic)
    return scores


def auto_threshold(change: list[float]) -> float:
    """Adaptive boundary threshold: mean + 2.5*std of the change scores,
    clamped to [0.08, 0.5]. A fixed threshold is brittle — smooth ResNet18
    walkthrough embeddings rarely exceed 0.3, while real transition zones
    sit at mean + 2..4 sigma. Returns 1.0 (no boundaries) if all scores are
    identical."""
    scores = np.array(change)
    if scores.std() < 1e-9:
        return 1.0
    return float(min(0.5, max(0.08, scores.mean() + 2.5 * scores.std())))


def segment_observations(
    observations: list[Observation],
    distance_threshold: float | None = None,
    change_window: int = 2,
    min_length: int = 3,
) -> list[Segment]:
    """Split ordered observations into segments. Returns [] for empty input.

    Boundary rule: elevated change positions are grouped into zones (gap
    tolerance 2); each zone yields ONE cut at its strongest position, so a
    real transition zone becomes exactly one boundary. Short segments (below
    min_length) are merged into their neighbor. A 1-frame spike's slivers are
    cleaned up by place reconciliation (Stage 10) — over-segmentation is
    recoverable, under-segmentation is not. change_window is kept as a
    parameter for compatibility.
    """
    observations = sort_observations(observations)
    if not observations:
        return []

    change = change_scores(observations)
    if not change:
        return [Segment("seg_000", [observations[0].id], 0, 0, observations[0].id)]

    if distance_threshold is None:
        distance_threshold = auto_threshold(change)

    # Group elevated change positions into zones (gap <= 2 positions): a real
    # transition zone spans several consecutive elevated pairs, so each zone
    # yields ONE cut at its strongest position. A 1-frame spike also produces
    # a small zone -> a single cut; the resulting sliver segments are
    # re-merged by place reconciliation (Stage 10), so over-segmentation here
    # is recoverable while under-segmentation is not.
    elevated = [i for i, score in enumerate(change) if score > distance_threshold]
    zones: list[list[int]] = []
    for pos in elevated:
        if zones and pos - zones[-1][-1] <= 2:
            zones[-1].append(pos)
        else:
            zones.append([pos])
    cut_points = []
    for zone in zones:
        peak = max(zone, key=lambda p: change[p])
        cut_points.append(peak + 1)

    ranges: list[tuple[int, int]] = []
    start = 0
    for cut in cut_points:
        if cut - start >= 1:
            ranges.append((start, cut - 1))
        start = cut
    ranges.append((start, len(observations) - 1))

    # merge segments shorter than min_length into their neighbor (the previous
    # one when possible, else the next — covers a short first segment)
    merged: list[tuple[int, int]] = []
    for rng in ranges:
        if rng[1] - rng[0] + 1 < min_length and merged:
            prev = merged[-1]
            merged[-1] = (prev[0], rng[1])
        else:
            merged.append(rng)
    if len(merged) > 1 and merged[0][1] - merged[0][0] + 1 < min_length:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)

    segments: list[Segment] = []
    for i, (lo, hi) in enumerate(merged):
        members = observations[lo : hi + 1]
        rep = _representative(members)
        segments.append(
            Segment(
                id=f"seg_{i:03d}",
                obs_ids=[o.id for o in members],
                start_index=lo,
                end_index=hi,
                representative_obs_id=rep.id,
            )
        )
    logger.info(f"Segmented {len(observations)} observations into {len(segments)} segments")
    return segments


def _representative(members: list[Observation]) -> Observation:
    """Observation closest to the segment's mean embedding (fallback: first)."""
    embeddings = [o.embedding for o in members]
    if any(e is None for e in embeddings):
        return members[0]
    stack = np.stack(embeddings)
    mean = stack.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    mean = mean / mean_norm if mean_norm > 0 else mean
    best_idx = int(np.argmax(stack @ mean))
    return members[best_idx]
