"""Failure analysis inspector (Stage 27, planner v2 §10).

Filters the tracker decision log (data/evaluation/decision_log.jsonl, written
by evaluate_suite.py) for the failure cases the planner calls out:
    - LOST events
    - low-confidence ARRIVED confirmations
    - frames where top-1 (and top-3) disagree with the hand-labeled
      data/test_labels.json

For each, dumps the frame + its top-3 candidates + semantic tags side by
side into data/evaluation/failure_log.jsonl — the "here's what the system
got wrong and why" evidence, stronger than a single accuracy number.

Run:
    conda run -n ML python scripts/failure_inspector.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("failure_inspector")


def load_labels(eval_dir: Path) -> dict:
    labels_file = eval_dir / "test_labels.json"
    if not labels_file.exists():
        logger.warning(f"No {labels_file} — label-disagreement checks are skipped")
        return {}
    with open(labels_file) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 27 failure analysis inspector.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    eval_dir = resolve_path(config["paths"]["evaluation_dir"])
    decision_log = eval_dir / "decision_log.jsonl"
    if not decision_log.exists():
        raise SystemExit(
            f"No decision log at {decision_log} — run evaluate_suite.py first."
        )

    labels = load_labels(eval_dir)
    failures: list[dict] = []
    with open(decision_log) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        kind = None
        if row.get("state") == "LOST":
            kind = "lost"
        elif row.get("state") == "ARRIVED" and row.get("confidence_level") == "LOW":
            kind = "low_confidence_arrived"
        # label disagreement: real label for this frame disagrees with the
        # tracker (top-1 vs top-3 split out)
        label = labels.get(Path(row.get("frame_path", "")).name)
        if label is not None and kind is None:
            try:
                label_id = int(label)
            except (TypeError, ValueError):
                label_id = label
            top_cands = [c.get("place_id") for c in row.get("candidates", [])[:3]]
            if top_cands and label_id not in top_cands:
                kind = "top3_misses_label"
            elif top_cands and label_id != top_cands[0]:
                kind = "top1_misses_label"
        if kind is None:
            continue
        failures.append(
            {
                "kind": kind,
                "frame": row.get("frame_path"),
                "timestamp": row.get("timestamp"),
                "label": labels.get(Path(row.get("frame_path", "")).name),
                "reported": row.get("reported"),
                "state": row.get("state"),
                "confidence_level": row.get("confidence_level"),
                "top_candidates": row.get("candidates", [])[:3],
                "scene_tags": row.get("scene_tags"),
            }
        )

    out_path = eval_dir / "failure_log.jsonl"
    with open(out_path, "w") as f:
        for failure in failures:
            f.write(json.dumps(failure, default=str) + "\n")
    logger.info(f"Wrote {len(failures)} failure cases to {out_path}")

    for failure in failures[:5]:
        print(json.dumps(failure, indent=2))


if __name__ == "__main__":
    main()
