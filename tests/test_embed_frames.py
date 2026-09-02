"""Tests for embed_frames.py helpers (Stage 24 batching config + scene-type
debounce) — no models involved."""

from src.embed_frames import _batch_size, _smooth_scene_types
from src.mapping.observations import Observation


def obs(i: int, scene: str) -> Observation:
    return Observation(
        id=f"obs_{i:04d}",
        timestamp=float(i),
        frame_path=f"data/frames/frame_{i:05d}.jpg",
        scene_tags={"scene_type": scene, "landmarks": []},
    )


def test_scene_type_debounce_smooths_flicker():
    observations = [
        obs(0, "corridor"), obs(1, "corridor"), obs(2, "room"),
        obs(3, "corridor"), obs(4, "corridor"), obs(5, "unknown"),
        obs(6, "corridor"),
    ]
    smoothed = _smooth_scene_types(observations)
    # the lone 'room' and 'unknown' flips fall to the corridor consensus
    assert [o.scene_tags["scene_type"] for o in smoothed] == [
        "corridor"] * 7


def test_scene_type_debounce_keeps_persistent_changes():
    observations = [obs(i, "corridor") for i in range(6)] + [
        obs(i, "room") for i in range(6, 12)
    ]
    smoothed = _smooth_scene_types(observations)
    scenes = [o.scene_tags["scene_type"] for o in smoothed]
    assert scenes[:5] == ["corridor"] * 5
    assert scenes[7:] == ["room"] * 5


def test_scene_type_debounce_tolerates_missing_tags():
    observations = [obs(0, "corridor")]
    observations.append(Observation(id="obs_0001", timestamp=1.0,
                                    frame_path="x.jpg", scene_tags=None))
    smoothed = _smooth_scene_types(observations)  # must not raise
    assert smoothed[0].scene_tags["scene_type"] == "corridor"


def test_batch_size_from_runtime_config():
    cfg = {"runtime": {"vlm_batch_size": 12, "detector_batch_size": 16,
                       "encoder_batch_size": 16}}
    assert _batch_size(cfg, "vlm_batch_size", 12) == 12
    assert _batch_size(cfg, "detector_batch_size", 16) == 16
    assert _batch_size(cfg, "encoder_batch_size", 16) == 16  # planner v3 §6
    assert _batch_size({}, "vlm_batch_size", 12) == 12  # defaults
    assert _batch_size({"runtime": {"vlm_batch_size": 0}}, "vlm_batch_size", 12) == 1


def test_save_observations_dir_threads_encoder_name(tmp_path):
    """Planner v3 §6: the name written to encoder.json is the one handed to
    save_observations_dir — not a hardcoded literal (dead-parameter fix)."""
    import json

    import numpy as np

    from src.embed_frames import save_observations_dir
    from src.mapping.observations import Observation

    obs = [
        Observation(id="obs_0000", timestamp=0.0, frame_path="data/frames/f0.jpg",
                    embedding=np.array([1.0, 0.0, 0.0], dtype="float32")),
        Observation(id="obs_0001", timestamp=1.0, frame_path="data/frames/f1.jpg",
                    embedding=np.array([0.0, 1.0, 0.0], dtype="float32")),
    ]
    save_observations_dir(obs, tmp_path, encoder_name="dinov2_registers_small")
    with open(tmp_path / "encoder.json") as f:
        meta = json.load(f)
    assert meta["model"] == "dinov2_registers_small"
    assert meta["dimension"] == 3
