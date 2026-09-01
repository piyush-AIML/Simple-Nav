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

from src.embedder import embed_image
from src.mapping.observation_store import ObservationStore
from src.mapping.observations import Observation
from src.utils import load_config, project_root, resolve_path, setup_logger

logger = setup_logger("embed_frames")


def embed_frames(frames_dir: Path) -> tuple[np.ndarray, list[str]]:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}. Run extract_frames.py first.")

    embeddings = []
    names = []
    for i, path in enumerate(frame_paths):
        vec = embed_image(str(path))
        embeddings.append(vec)
        names.append(path.name)
        if (i + 1) % 25 == 0 or (i + 1) == len(frame_paths):
            logger.info(f"Embedded {i + 1}/{len(frame_paths)} frames")

    return np.stack(embeddings).astype("float32"), names


def build_observations(
    frame_paths: list[Path], embeddings: np.ndarray, encoder_name: str = "resnet18"
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
    observations: list[Observation], observations_dir: Path, encoder_name: str = "resnet18"
) -> None:
    """Write a full ObservationStore (Stage 05): JSONL metadata + embeddings
    + FAISS index + id order, so data/observations/ is loadable as-is."""
    store = ObservationStore(observations_dir)
    store.add(observations)
    store.save()
    logger.info(f"Saved {len(observations)} observations to {observations_dir}")


def attach_detections(observations: list[Observation], config: dict) -> list[Observation]:
    """Fill Observation.objects from the configured detector (Stage 06).
    Detector failures never break the pipeline — objects stay [] (§6)."""
    perception = config.get("perception", {})
    if not perception.get("detector_enabled", True):
        return observations
    from src.perception.detector import get_detector

    detector = get_detector(config)
    if detector.name == "stub":
        logger.warning("Detector unavailable — observations keep objects=[]")
        return observations
    for obs in observations:
        obs.objects = [o.to_dict() for o in detector.detect(obs.frame_path)]
    logger.info(f"Attached detections to {len(observations)} observations ({detector.name})")
    return observations


def attach_scene_tags(observations: list[Observation], config: dict) -> list[Observation]:
    """Fill Observation.scene_tags + landmarks from the VLM (Stage 07).
    Failures degrade to unknown/[] — never crash (§7)."""
    perception = config.get("perception", {})
    if not perception.get("vlm_enabled", True):
        return observations
    from src.perception.scene_tagger import get_scene_tagger

    tagger = get_scene_tagger(config)
    if tagger.name() == "stub":
        logger.warning("VLM unavailable — observations keep scene_tags=None")
        return observations
    for obs in observations:
        tags = tagger.tag(obs.frame_path, obs.objects)
        obs.scene_tags = tags.to_dict()
        obs.landmarks = tags.landmarks
    logger.info(f"Attached scene tags to {len(observations)} observations ({tagger.name()})")
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed extracted frames with a pretrained CNN.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    embeddings_file = resolve_path(config["paths"]["embeddings_file"])
    frame_names_file = resolve_path(config["paths"]["frame_names_file"])
    observations_dir = resolve_path(config["paths"]["observations_dir"])

    embeddings, names = embed_frames(frames_dir)

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    with open(frame_names_file, "w") as f:
        json.dump(names, f, indent=2)
    logger.info(f"Saved {embeddings.shape} embeddings to {embeddings_file}")

    # Observation records (Stage 02) — legacy outputs above remain untouched.
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    observations = build_observations(frame_paths, embeddings)
    observations = attach_detections(observations, config)
    observations = attach_scene_tags(observations, config)
    save_observations_dir(observations, observations_dir)


if __name__ == "__main__":
    main()
