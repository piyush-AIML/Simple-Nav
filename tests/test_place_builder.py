"""Tests for place formation (§11): segments -> places, exemplar selection
methods, and place statistics."""

import numpy as np

from src.mapping.observations import Observation
from src.mapping.place_builder import build_places, select_exemplars
from src.mapping.segmentation import Segment

A = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
B = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
C = np.array([0.0, 0.0, 1.0, 0.0], dtype="float32")


def make_obs(i: int, emb: np.ndarray, scene: str = "corridor", landmarks=None) -> Observation:
    return Observation(
        id=f"obs_{i:04d}",
        timestamp=float(i),
        frame_path=f"data/frames/frame_{i:05d}.jpg",
        embedding=emb,
        scene_tags={"scene_type": scene, "landmarks": landmarks or []},
        landmarks=landmarks or [],
    )


def test_build_places_one_per_segment():
    obs = [
        make_obs(0, A), make_obs(1, A), make_obs(2, A),
        make_obs(3, B), make_obs(4, B), make_obs(5, B),
    ]
    segments = [
        Segment("seg_000", ["obs_0000", "obs_0001", "obs_0002"], 0, 2, "obs_0001"),
        Segment("seg_001", ["obs_0003", "obs_0004", "obs_0005"], 3, 5, "obs_0004"),
    ]
    places = build_places(obs, segments)
    assert len(places) == 2
    assert places[0].observation_ids == ["obs_0000", "obs_0001", "obs_0002"]
    assert places[0].segment_ids == ["seg_000"]


def test_place_carries_scene_and_landmark_stats():
    obs = [
        make_obs(0, A, scene="room", landmarks=["desk"]),
        make_obs(1, A, scene="room", landmarks=["desk"]),
        make_obs(2, A, scene="corridor", landmarks=["blue sign"]),
    ]
    segments = [Segment("seg_000", ["obs_0000", "obs_0001", "obs_0002"], 0, 2, "obs_0001")]
    place = build_places(obs, segments)[0]
    assert place.scene_types["room"] == 2
    assert place.scene_types["corridor"] == 1
    assert place.landmarks == ["desk", "blue sign"]
    assert place.visual_stats["observation_count"] == 3
    assert place.visual_stats["mean_similarity"] > 0.9  # identical vectors


def test_missing_observations_are_skipped():
    obs = [make_obs(0, A)]
    segments = [Segment("seg_000", ["obs_0000", "obs_9999"], 0, 1, "obs_0000")]
    places = build_places(obs, segments)
    assert len(places) == 1
    assert places[0].observation_ids == ["obs_0000"]


def test_exemplars_temporal_diversity_spans_extent():
    obs = [make_obs(i, A if i % 2 == 0 else B) for i in range(10)]
    ids = select_exemplars(obs, method="temporal_diversity", max_exemplars=3)
    assert len(ids) == 3
    assert ids[0] == "obs_0000"       # first
    assert ids[1] == "obs_0009"       # last
    assert ids[2] == "obs_0005"       # middle (index 5 of 10)


def test_exemplars_diversity_picks_distinct_vectors():
    obs = [make_obs(i, [A, B, C][i % 3]) for i in range(6)]
    ids = select_exemplars(obs, method="diversity", max_exemplars=3)
    embeddings = {o.id: o.embedding for o in obs}
    distinct = {tuple(embeddings[oid].tolist()) for oid in ids}
    assert len(distinct) == 3  # one representative per distinct vector


def test_exemplars_kmeans_matches_legacy_behavior():
    obs = [make_obs(i, [A, B][i % 2]) for i in range(6)]
    ids = select_exemplars(obs, method="kmeans", max_exemplars=2)
    assert len(ids) == 2


def test_exemplars_unknown_method_raises():
    obs = [make_obs(0, A)]
    try:
        select_exemplars(obs, method="bogus")
        assert False, "should raise"
    except ValueError as e:
        assert "bogus" in str(e)


def test_empty_inputs():
    assert build_places([], []) == []
    assert select_exemplars([], method="temporal_diversity", max_exemplars=3) == []
