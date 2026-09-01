"""Stage 1: extract frames from a walkthrough video.

Default: SMART sampling (Stage 03) — quality gate + visual novelty +
transition-aware retention + temporal gap; see src/extraction/frames.py.

Legacy fixed-interval sampling (every Nth frame) remains available with
--baseline, byte-for-byte the old behavior.

Kept frames are saved in order with a zero-padded filename so sorting by name
preserves temporal order; a sampling_log.json records the decision for every
source frame (great for validating the sampler).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.extraction.frames import SmartFrameSampler
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("extract_frames")


def extract_frames(video_path: Path, frames_dir: Path, sample_every_n: int) -> int:
    """Legacy baseline: save 1 out of every N frames."""
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


def extract_frames_smart(video_path: Path, frames_dir: Path, config: dict) -> int:
    """Default: smart sampling (§5). Also writes sampling_log.json."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    sampler = SmartFrameSampler(config)
    kept = sampler.process(video_path)

    for saved_count, sampled in enumerate(kept):
        out_path = frames_dir / f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(str(out_path), sampled.frame)

    log_path = frames_dir / "sampling_log.json"
    sampler.save_log(log_path)
    logger.info(f"Smart sampling kept {len(kept)} frames -> {frames_dir} (log: {log_path})")
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a walkthrough video.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use legacy fixed-interval sampling (every Nth frame) instead of smart sampling",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    video_path = resolve_path(config["paths"]["video"])
    frames_dir = resolve_path(config["paths"]["frames_dir"])

    if args.baseline:
        sample_every_n = config["extraction"]["sample_every_n_frames"]
        extract_frames(video_path, frames_dir, sample_every_n)
    else:
        extract_frames_smart(video_path, frames_dir, config)


if __name__ == "__main__":
    main()
