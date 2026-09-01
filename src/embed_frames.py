"""Stage 2: turn every extracted frame into a feature vector.

Output:
  - embeddings.npy      shape (N, 512), one row per frame, in temporal order
  - frame_names.json    list of N filenames, same order as embeddings rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.embedder import embed_image
from src.utils import load_config, resolve_path, setup_logger

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed extracted frames with a pretrained CNN.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    embeddings_file = resolve_path(config["paths"]["embeddings_file"])
    frame_names_file = resolve_path(config["paths"]["frame_names_file"])

    embeddings, names = embed_frames(frames_dir)

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    with open(frame_names_file, "w") as f:
        json.dump(names, f, indent=2)

    logger.info(f"Saved {embeddings.shape} embeddings to {embeddings_file}")


if __name__ == "__main__":
    main()
