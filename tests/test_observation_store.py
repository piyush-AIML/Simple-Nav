"""Tests for the ObservationStore (§9): save/load round trip, id -> all
metadata + frame path recovery, FAISS search, append behavior."""

import json

import numpy as np
import pytest

from src.mapping.observation_store import ObservationStore, ObservationStoreError
from src.mapping.observations import Observation


def make_obs(i: int, vec: np.ndarray | None = None, **overrides) -> Observation:
    base = dict(
        id=f"obs_{i:04d}",
        timestamp=float(i),
        frame_path=f"data/frames/frame_{i:05d}.jpg",
        embedding=np.array([float(i), 1.0, 0.0, 0.0], dtype="float32") if vec is None else vec,
        quality_score=0.9,
        scene_tags={"scene_type": "corridor"},
        landmarks=["blue sign"],
    )
    base.update(overrides)
    return Observation(**base)


def test_save_load_round_trip_recovers_everything(tmp_path):
    obs_list = [
        make_obs(0, np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")),
        make_obs(1, np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")),
    ]
    store = ObservationStore(tmp_path)
    store.add(obs_list)
    store.save(encoder_name="resnet18")

    fresh = ObservationStore.load(tmp_path)
    assert len(fresh) == 2

    restored = fresh.get("obs_0001")
    assert restored.frame_path == "data/frames/frame_00001.jpg"
    assert restored.scene_tags == {"scene_type": "corridor"}
    assert restored.landmarks == ["blue sign"]
    np.testing.assert_allclose(restored.embedding, obs_list[1].embedding)


def test_search_returns_nearest(tmp_path):
    obs_list = [
        make_obs(0, np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")),
        make_obs(1, np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")),
        make_obs(2, np.array([0.0, 0.0, 1.0, 0.0], dtype="float32")),
    ]
    store = ObservationStore(tmp_path)
    store.add(obs_list)

    query = np.array([0.9, 0.4, 0.0, 0.0], dtype="float32")  # closest to obs_0000
    results = store.search(query, top_k=2)
    assert [obs.id for obs, _ in results] == ["obs_0000", "obs_0001"]
    assert results[0][1] > results[1][1]


def test_append_preserves_ids(tmp_path):
    store = ObservationStore(tmp_path)
    store.add([make_obs(0, np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))])
    store.add([make_obs(1, np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"))])
    store.save(encoder_name="resnet18")

    fresh = ObservationStore.load(tmp_path)
    assert [o.id for o in fresh.all()] == ["obs_0000", "obs_0001"]
    assert fresh.get("obs_0000").id == "obs_0000"


def test_duplicate_id_rejected():
    store = ObservationStore("unused")
    store.add([make_obs(0)])
    with pytest.raises(ObservationStoreError, match="Duplicate"):
        store.add([make_obs(0, np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"))])


def test_missing_embedding_rejected():
    store = ObservationStore("unused")
    obs = make_obs(0, vec=None)
    obs.embedding = None
    with pytest.raises(ObservationStoreError, match="no embedding"):
        store.add([obs])


def test_save_writes_the_encoder_name_it_was_given(tmp_path):
    """Planner v3 §6: encoder.json must record the encoder that actually
    produced the vectors — a non-default name, not a hardcoded literal."""
    store = ObservationStore(tmp_path)
    store.add([make_obs(0)])
    store.save(encoder_name="dinov2_registers_small")

    with open(tmp_path / "encoder.json") as f:
        meta = json.load(f)
    assert meta["model"] == "dinov2_registers_small"
    assert meta["dimension"] == 4


def test_load_missing_files_raises(tmp_path):
    with pytest.raises(ObservationStoreError, match="missing files"):
        ObservationStore.load(tmp_path)


def test_unknown_id_raises(tmp_path):
    store = ObservationStore(tmp_path)
    store.add([make_obs(0)])
    with pytest.raises(KeyError):
        store.get("obs_9999")
