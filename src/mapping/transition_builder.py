"""Transition extraction (§13): turn the ordered place sequence into
movement evidence.

Debouncing comes FIRST (§13): a place must persist >= transition_persistence
observations before it counts as a real visit, so A->B->A noise collapses
into a single A run and can never create an edge. Then consecutive distinct
places in the debounced sequence produce directional transition statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils import setup_logger

logger = setup_logger("transition_builder")


@dataclass
class TransitionStats:
    a: str
    b: str
    forward_count: int = 0            # A -> B observations
    reverse_count: int = 0            # B -> A observations
    transition_duration: float = 0.0  # mean duration of A->B movements (seconds)
    supporting_observations: int = 0
    visual_transition_strength: float = 0.0  # mean boundary change score at crossings
    confidence: float = 0.0

    def total(self) -> int:
        return self.forward_count + self.reverse_count


def debounce(ordered_place_ids: list[str], min_persistence: int) -> list[str]:
    """Remove visits shorter than min_persistence (spike noise)."""
    if not ordered_place_ids:
        return []
    result: list[str] = []
    run_start = 0
    i = 0
    while i < len(ordered_place_ids):
        j = i
        while j < len(ordered_place_ids) and ordered_place_ids[j] == ordered_place_ids[i]:
            j += 1
        if j - i >= min_persistence:
            result.extend([ordered_place_ids[i]] * (j - i))
        i = j
    return result


def extract_transitions(
    ordered_place_ids: list[str],
    min_persistence: int = 3,
    minimum_edge_support: int = 3,
    timestamps: list[float] | None = None,
    transition_strengths: list[float] | None = None,
) -> list[TransitionStats]:
    """Build TransitionStats for each ordered pair of distinct places in the
    DEBOUNCED sequence. Stats are per direction:
      - forward_count: A->B crossings (debounced)
      - reverse_count: B->A crossings
      - transition_duration: mean time spent crossing A->B (if timestamps)
      - supporting_observations: total observations of the two places
      - visual_transition_strength: mean boundary score at crossings (optional)
      - confidence: min(1, total / minimum_edge_support) * persistence factor
    """
    sequence = debounce(ordered_place_ids, min_persistence)
    if len(sequence) < 2:
        return []

    stats: dict[tuple[str, str], TransitionStats] = {}
    order: list[tuple[str, str]] = []

    def get_stat(a: str, b: str) -> TransitionStats:
        key = (a, b)
        if key not in stats:
            stats[key] = TransitionStats(a=a, b=b)
            order.append(key)
        return stats[key]

    prev_place, prev_idx = sequence[0], 0
    for i in range(1, len(sequence)):
        if sequence[i] == prev_place:
            continue
        stat = get_stat(prev_place, sequence[i])
        if timestamps is not None:
            stat.transition_duration += (timestamps[i] - timestamps[prev_idx])
        stat.forward_count += 1
        if transition_strengths is not None and prev_idx < len(transition_strengths):
            stat.visual_transition_strength += transition_strengths[prev_idx]
        prev_place, prev_idx = sequence[i], i

    results: list[TransitionStats] = []
    for key in order:
        stat = stats[key]
        if timestamps is not None and stat.forward_count:
            stat.transition_duration /= stat.forward_count
        if transition_strengths is not None and stat.forward_count:
            stat.visual_transition_strength /= stat.forward_count
        support = ordered_place_ids.count(stat.a) + ordered_place_ids.count(stat.b)
        stat.supporting_observations = support
        stat.confidence = round(min(1.0, stat.total() / minimum_edge_support) * 0.9 + 0.1, 3)
        results.append(stat)

    logger.info(
        f"Extracted {len(results)} transition(s) from {len(sequence)} debounced "
        f"place observations (persistence={min_persistence})"
    )
    return results
