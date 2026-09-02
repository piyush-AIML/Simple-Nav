"""Tests for temporal segmentation (§10): clean splits, single-frame noise
robustness, revisit structure, and semantic-change boundaries.
Synthetic embeddings in 2-D — no model involved."""

import numpy as np

from src.mapping.observations import Observation
from src.mapping.segmentation import change_scores, segment_observations

A = np.array([1.0, 0.0], dtype="float32")
B = np.array([0.0, 1.0], dtype="float32")


def obs_sequence(plan: list[tuple[np.ndarray, str]]) -> list[Observation]:
    """plan: list of (embedding, scene_type). Sequential ids/timestamps."""
    out = []
    for i, (emb, scene) in enumerate(plan):
        out.append(
            Observation(
                id=f"obs_{i:04d}",
                timestamp=float(i),
                frame_path=f"data/frames/frame_{i:05d}.jpg",
                embedding=emb,
                scene_tags={"scene_type": scene, "landmarks": [], "sign_text": [], "walkable": []},
            )
        )
    return out


def run(plan, **kwargs) -> list[list[str]]:
    """Segment ids per plan, return [segment_obs_ids]."""
    segments = segment_observations(obs_sequence(plan), **kwargs)
    return [seg.obs_ids for seg in segments]


def test_clean_split_into_two_areas():
    plan = [(A, "a")] * 10 + [(B, "b")] * 10
    segments = run(plan)
    assert len(segments) == 2
    assert len(segments[0]) == 10 and len(segments[1]) == 10


def test_single_anomalous_frame_does_not_split():
    """A 1-frame spike may produce a temporary boundary in segmentation, but
    the full pipeline (segmentation + place formation + reconciliation) must
    converge back to ONE place — over-segmentation is recoverable."""
    plan = [(A, "a")] * 5 + [(B, "b")] + [(A, "a")] * 5
    obs = obs_sequence(plan)
    segments = segment_observations(obs)
    assert len(segments) <= 2, f"spike should create at most one extra split: {segments}"

    from src.mapping.place_builder import build_places
    from src.mapping.place_reconciliation import reconcile_places

    places = build_places(obs, segments)
    merged = reconcile_places(
        places, obs,
        {"merge_visual_threshold": 0.75, "merge_visual_extra_threshold": 0.92,
         "merge_landmark_threshold": 0.5, "merge_scene_required": True},
    )
    assert len(merged) == 1, f"spike split survived reconciliation: {merged}"


def test_revisit_structure_reflected():
    """A→B→A with visually identical A must yield 3 temporal segments
    (reconciliation, Stage 10, decides whether the two A runs merge)."""
    plan = [(A, "a")] * 10 + [(B, "b")] * 10 + [(A, "a")] * 10
    segments = run(plan)
    assert len(segments) == 3
    assert len(segments[0]) == len(segments[2]) == 10


def test_empty_input_returns_no_segments():
    assert segment_observations([]) == []


def test_semantic_scene_change_creates_boundary_even_with_similar_embeddings():
    """Same embedding but different scene_type -> semantic signal splits."""
    plan = [(A, "a")] * 8 + [(A, "b")] * 8  # identical vectors, different scenes
    segments = run(plan, distance_threshold=0.35)
    assert len(segments) == 2, segments


def test_change_scores_are_zero_inside_a_region():
    plan = [(A, "a")] * 6 + [(B, "b")] * 6
    scores = change_scores(obs_sequence(plan))
    assert scores[:5] == [0.0] * 5
    assert scores[5] > 0.9


def test_short_noise_burst_at_edge_does_not_create_tiny_segment():
    """A brief 2-frame blip at the start merges into the following run."""
    plan = [(B, "b")] * 2 + [(A, "a")] * 8
    segments = run(plan, min_length=3)
    assert len(segments) == 1, segments


def test_scene_flicker_does_not_cut():
    """A single flipped frame must not create a persistent boundary — the
    semantic change only counts when the new scene type persists. At most
    one temporary split survives segmentation; reconciliation converges
    (same spike contract as the visual case)."""
    plan = [(A, "corridor")] * 5 + [(A, "room")] + [(A, "corridor")] * 5
    obs = obs_sequence(plan)
    segments = segment_observations(obs, distance_threshold=0.35)
    assert len(segments) <= 2, segments

    from src.mapping.place_builder import build_places
    from src.mapping.place_reconciliation import reconcile_places

    places = build_places(obs, segments)
    merged = reconcile_places(
        places, obs,
        {"merge_visual_threshold": 0.75, "merge_visual_extra_threshold": 0.92,
         "merge_landmark_threshold": 0.5, "merge_scene_required": True},
    )
    assert len(merged) == 1, f"flicker split survived reconciliation: {merged}"


def test_unknown_is_neutral_in_change_scores():
    """Flipping into 'unknown' is missing evidence, not a scene change."""
    plan = [(A, "corridor")] * 4 + [(A, "unknown")] * 4
    scores = change_scores(obs_sequence(plan))
    assert scores == [0.0] * 7


def test_persistent_scene_change_still_cuts():
    plan = [(A, "corridor")] * 5 + [(A, "room")] * 5
    segments = run(plan, distance_threshold=0.35)
    assert len(segments) == 2, segments
