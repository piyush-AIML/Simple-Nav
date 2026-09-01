"""Stage 1: extract frames from a walkthrough video at a fixed interval.

Simple by design: no blur checking, no adaptive sampling — every Nth frame
is saved, in order, with a zero-padded filename so sorting by name preserves
temporal order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("extract_frames")


def extract_frames(video_path: Path, frames_dir: Path, sample_every_n: int) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_n == 0:
            out_path = frames_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    logger.info(f"Read {frame_idx} frames, saved {saved_count} to {frames_dir}")
    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a walkthrough video.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    video_path = resolve_path(config["paths"]["video"])
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    sample_every_n = config["extraction"]["sample_every_n_frames"]

    extract_frames(video_path, frames_dir, sample_every_n)


if __name__ == "__main__":
    main()
