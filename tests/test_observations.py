"""Tests for the Observation model and its JSONL serialization (§4)."""

import numpy as np

from src.mapping.observations import (
    Observation,
    load_observations_jsonl,
    save_observations_jsonl,
    sort_observations,
)


def make_obs(i: int, timestamp: float, **overrides) -> Observation:
    base = dict(
        id=f"obs_{i:06d}",
        timestamp=timestamp,
        frame_path=f"data/frames/frame_{i:05d}.jpg",
        embedding=np.array([float(i), 1.0, 0.0, 0.0], dtype="float32"),
        quality_score=0.9,
    )
    base.update(overrides)
    return Observation(**base)


def test_round_trip_preserves_metadata(tmp_path):
    obs = make_obs(1, 12.5, scene_tags={"scene_type": "corridor"}, landmarks=["blue sign"])
    path = tmp_path / "obs.jsonl"
    save_observations_jsonl([obs], path)
    loaded = load_observations_jsonl(path)
    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.id == "obs_000001"
    assert restored.timestamp == 12.5
    assert restored.frame_path == "data/frames/frame_00001.jpg"
    assert restored.scene_tags == {"scene_type": "corridor"}
    assert restored.landmarks == ["blue sign"]
    assert restored.embedding is None  # never serialized to JSONL


def test_round_trip_with_embeddings_aligned(tmp_path):
    obs_list = [make_obs(0, 0.0), make_obs(1, 1.0)]
    path = tmp_path / "obs.jsonl"
    save_observations_jsonl(obs_list, path)
    embeddings = np.stack([o.embedding for o in obs_list])
    loaded = load_observations_jsonl(path, embeddings=embeddings)
    for o, orig in zip(loaded, obs_list):
        assert o.embedding is not None
        np.testing.assert_allclose(o.embedding, orig.embedding)


def test_metadata_may_be_absent():
    """Per §4 instruction 6: perception metadata absent must not break anything."""
    obs = make_obs(2, 5.0, scene_tags=None, landmarks=[])
    d = obs.to_dict()
    assert d["scene_tags"] is None
    assert d["landmarks"] == []
    restored = Observation(
        id=d["id"], timestamp=d["timestamp"], frame_path=d["frame_path"],
        scene_tags=d.get("scene_tags"), landmarks=d.get("landmarks") or [],
    )
    assert restored.scene_tags is None
    assert restored.landmarks == []


def test_sort_observations_by_timestamp_then_id():
    a = make_obs(5, 3.0)
    b = make_obs(1, 3.0)  # same timestamp, lower id first
    c = make_obs(2, 1.0)
    ordered = sort_observations([a, b, c])
    assert [o.id for o in ordered] == ["obs_000002", "obs_000001", "obs_000005"]


def test_to_dict_excludes_embedding():
    obs = make_obs(0, 0.0)
    d = obs.to_dict()
    assert "embedding" not in d
