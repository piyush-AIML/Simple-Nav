"""Stage 2: turn every extracted frame into a feature vector.

Output:
  - embeddings.npy      shape (N, 512), one row per frame, in temporal order
  - frame_names.json    list of N filenames, same order as embeddings rows
  - data/observations/  (Stage 02+): observations.jsonl + embeddings.npy +
    encoder.json — the Observation records that the mapping pipeline consumes.
    Legacy .npy/json outputs remain, unchanged, for the baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.mapping.observation_store import ObservationStore
from src.mapping.observations import Observation
from src.perception import backend_banner
from src.utils import load_config, project_root, resolve_path, setup_logger

logger = setup_logger("embed_frames")


def embed_frames(frames_dir: Path, config: dict) -> tuple[np.ndarray, list[str]]:
    """Encode every extracted frame (planner v3 §6).

    The encoder comes from config.embedding.model via the registry in
    src/embeddings/encoder.py — the registry is the single source of truth
    for which encoder produced the stored vectors (Finding A: this used to
    be hardcoded to ResNet18 and the config knob was decorative). Batched
    like the detector/VLM offline passes (Stage 24 precedent)."""
    from src.embeddings.encoder import get_encoder

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}. Run extract_frames.py first.")

    encoder = get_encoder(config)
    batch_size = _batch_size(config, "encoder_batch_size", 16)
    embeddings = []
    for start in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[start:start + batch_size]
        embeddings.append(encoder.batch_encode([str(p) for p in chunk]))
        logger.info(f"Embedded {min(start + batch_size, len(frame_paths))}/{len(frame_paths)} frames")

    return np.concatenate(embeddings).astype("float32"), [p.name for p in frame_paths]


def build_observations(
    frame_paths: list[Path], embeddings: np.ndarray
) -> list[Observation]:
    """Turn aligned frame/embedding lists into Observation records.

    timestamp is the frame index (the provisional dataset has no known fps;
    with a real video this becomes seconds = index / fps).
    frame_path is stored relative to the project root for portability.
    """
    root = project_root()
    observations = []
    for i, (path, emb) in enumerate(zip(frame_paths, embeddings)):
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        observations.append(
            Observation(
                id=f"obs_{i:06d}",
                timestamp=float(i),
                frame_path=rel,
                embedding=emb.astype("float32"),
            )
        )
    return observations


def save_observations_dir(
    observations: list[Observation], observations_dir: Path, encoder_name: str
) -> None:
    """Write a full ObservationStore (Stage 05): JSONL metadata + embeddings
    + FAISS index + id order, so data/observations/ is loadable as-is.
    encoder_name must be the encoder that produced the embeddings — the
    store records it in encoder.json (planner v3 §6); no function in this
    chain may write a name it wasn't handed."""
    store = ObservationStore(observations_dir)
    store.add(observations)
    store.save(encoder_name=encoder_name)
    logger.info(f"Saved {len(observations)} observations to {observations_dir}")


def _smooth_scene_types(observations: list[Observation],
                        window: int = 2) -> list[Observation]:
    """Temporal debounce of per-frame VLM scene types (window ±2 = 5 frames).

    Per-frame VLM output flickers — measured 72 scene-type flips over the
    301-frame pass, with 138 'unknown'. A place must be described by the
    consensus scene of its neighborhood, not by whichever single frame the
    tagger happened to see (same debounce philosophy as Stage 11's
    transitions). Only scene_type is smoothed; landmarks are untouched."""
    from collections import Counter

    scenes = [
        (obs.scene_tags or {}).get("scene_type", "unknown") for obs in observations
    ]
    for i, obs in enumerate(observations):
        lo, hi = max(0, i - window), min(len(scenes), i + window + 1)
        consensus = Counter(scenes[lo:hi]).most_common(1)[0][0]
        if obs.scene_tags is not None and consensus != obs.scene_tags.get("scene_type"):
            obs.scene_tags = {**obs.scene_tags, "scene_type": consensus}
    return observations


def _batch_size(config: dict, key: str, default: int) -> int:
    """Offline mapping-pass batch size (Stage 24); live loop stays 1-at-a-time."""
    return max(1, int((config.get("runtime", {}) or {}).get(key, default)))


def attach_detections(observations: list[Observation], config: dict) -> list[Observation]:
    """Fill Observation.objects from the configured detector (Stage 06).
    Detector failures never break the pipeline — objects stay [] (§6).
    Batched for the offline pass (Stage 24)."""
    perception = config.get("perception", {})
    if not perception.get("detector_enabled", True):
        return observations
    from src.perception.detector import get_detector

    detector = get_detector(config)
    if detector.name == "stub":
        logger.warning("Detector unavailable — observations keep objects=[]")
        return observations
    batch_size = _batch_size(config, "detector_batch_size", 16)
    paths = [obs.frame_path for obs in observations]
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        results = detector.detect_batch(chunk)
        for obs, objs in zip(observations[start:start + batch_size], results):
            obs.objects = [o.to_dict() for o in objs]
    logger.info(f"Attached detections to {len(observations)} observations "
                f"({detector.name}, batch={batch_size})")
    return observations


def attach_scene_tags(observations: list[Observation], config: dict) -> list[Observation]:
    """Fill Observation.scene_tags + landmarks from the VLM (Stage 07).
    Failures degrade to unknown/[] — never crash (§7).
    Batched for the offline pass (Stage 24)."""
    perception = config.get("perception", {})
    if not perception.get("vlm_enabled", True):
        return observations
    from src.perception.scene_tagger import get_scene_tagger

    tagger = get_scene_tagger(config)
    if tagger.name() == "stub":
        logger.warning("VLM unavailable — observations keep scene_tags=None")
        return observations
    batch_size = _batch_size(config, "vlm_batch_size", 12)
    paths = [obs.frame_path for obs in observations]
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        objs_lists = [obs.objects for obs in observations[start:start + batch_size]]
        tags = tagger.tag_batch(chunk, objs_lists)
        for obs, tags_i in zip(observations[start:start + batch_size], tags):
            obs.scene_tags = tags_i.to_dict()
            obs.landmarks = tags_i.landmarks
    logger.info(f"Attached scene tags to {len(observations)} observations "
                f"({tagger.name()}, batch={batch_size})")
    return _smooth_scene_types(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed extracted frames with a pretrained CNN.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    logger.info(backend_banner(config))
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    observations_dir = resolve_path(config["paths"]["observations_dir"])

    from src.embeddings.encoder import get_encoder

    encoder = get_encoder(config)
    embeddings, names = embed_frames(frames_dir, config)

    # Legacy parallel outputs (embeddings.npy + frame_names.json) are only
    # written when their config keys still exist — the v2 pipeline consumes
    # ObservationStore, not these files.
    legacy_keys = ("embeddings_file", "frame_names_file")
    if all(k in config["paths"] for k in legacy_keys):
        embeddings_file = resolve_path(config["paths"]["embeddings_file"])
        frame_names_file = resolve_path(config["paths"]["frame_names_file"])
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_file, embeddings)
        with open(frame_names_file, "w") as f:
            json.dump(names, f, indent=2)
        logger.info(f"Saved {embeddings.shape} embeddings to {embeddings_file}")

    # Observation records (Stage 02) — detections + scene tags attached with
    # the batched offline pass (Stage 24).
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    observations = build_observations(frame_paths, embeddings)
    observations = attach_detections(observations, config)
    observations = attach_scene_tags(observations, config)
    save_observations_dir(observations, observations_dir, encoder_name=encoder.name)


if __name__ == "__main__":
    main()
