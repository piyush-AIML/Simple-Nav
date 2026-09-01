"""Stage 24 acceptance probe (planner v2 §7): measure offline map-building
wall-clock with batching enabled vs disabled on the real 301-frame set.
Reports both numbers for STATE.md.

Run:
    conda run -n ML python scripts/time_batch_vs_sequential.py [--n 36]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("time_batch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=36, help="frames to time")
    args = parser.parse_args()

    config = load_config()
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    paths = sorted(frames_dir.glob("*.jpg"))[: args.n]

    # detector: batched vs sequential
    from src.perception.detector import get_detector

    detector = get_detector(config)
    t0 = time.perf_counter()
    for p in paths:
        detector.detect(str(p))
    det_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    detector.detect_batch([str(p) for p in paths])
    det_batch = time.perf_counter() - t0

    # VLM: batched vs sequential (one model load, both timings)
    from src.perception.scene_tagger import get_scene_tagger

    tagger = get_scene_tagger(config)
    t0 = time.perf_counter()
    tagger.tag_batch([str(p) for p in paths])
    vlm_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    for p in paths:
        tagger.tag(str(p))
    vlm_seq = time.perf_counter() - t0

    print(f"frames_timed: {len(paths)}")
    print(f"detector_sequential_s: {det_seq:.2f}  detector_batched_s: {det_batch:.2f}")
    print(f"vlm_sequential_s: {vlm_seq:.2f}  vlm_batched_s: {vlm_batch:.2f}")
    print(f"vlm_speedup: {vlm_seq / max(vlm_batch, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
