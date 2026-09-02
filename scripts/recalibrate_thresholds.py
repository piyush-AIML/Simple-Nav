"""Planner v3 §5 (Stage 34): threshold recalibration protocol.

After an encoder swap, the raw-cosine-similarity constants in config.yaml
must be re-derived for the new encoder's similarity distribution — not
copied (Finding H: the current values are tuned to ResNet18's distribution,
by the codebase's own comment in place_reconciliation.py).

This script percentile-matches every raw-cosine-scale threshold: the
current value sits at some percentile of the OLD encoder's distribution
(from data/evaluation/encoder_comparison.json — the same-place /
different-place pairwise similarity arrays); the new value is whatever sits
at that same percentile of the NEW encoder's distribution.

Distribution assignment:
  different-place: merge decisions between places ("same-building
      cross-area" similarity, per place_reconciliation.py's comment).
  same-place:      retrieval/tracking decisions (query vs the correct
      place). Note: tracker/confidence thresholds operate on the blended
      total score (w_visual 0.5 etc.), so this is an approximation on the
      visual term's distribution — the ablation re-run after this script
      is the arbiter, exactly as the protocol prescribes.

likelihood_temperature is a scale on evidence-score spread, not a
percentile — mapped by the same-place std ratio instead.

Run (after scripts/compare_encoders.py):
    conda run -n ML python scripts/recalibrate_thresholds.py
    # --old resnet18 --new dinov2_registers_small (defaults)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import resolve_path, setup_logger

logger = setup_logger("recalibrate")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# (yaml_section, key, distribution, note)
THRESHOLDS = [
    ("mapping", "merge_visual_threshold", "different",
     "v3 §5 list: reconciliation merge gate (cross-area similarity)"),
    ("mapping", "merge_visual_extra_threshold", "different",
     "v3 §5 list: near-identical merge when both lack landmarks"),
    ("mapping", "duplicate_similarity_threshold", "same",
     "graph_validator duplicate-place gate (same physical place re-seen)"),
    ("navigation", "confidence_threshold", "same",
     "v3 §5 list: legacy LiveTracker/PlaceIndex query acceptance"),
    ("localization", "tracking_threshold", "same",
     "v3 §5 list: state machine strong-tracking gate (blended total)"),
    ("localization", "lost_threshold", "same",
     "v3 §5 list: LOST gate (blended total)"),
    ("localization", "high_score", "same",
     "v3 §5 list: confidence HIGH gate (blended total)"),
    ("localization", "low_score", "same",
     "v3 §5 list: confidence LOW gate (blended total)"),
    ("localization", "reacquired_threshold", "same",
     "same family as tracking_threshold (blended total)"),
    ("localization", "arrived_threshold", "same",
     "same family as tracking_threshold (blended total)"),
    ("localization", "recovery_threshold", "same",
     "same family as tracking_threshold (blended total)"),
    ("localization", "transition_threshold", "same",
     "posterior stabilization gate (same scale family)"),
]


def percentile(old_value: float, old_dist: list[float]) -> float:
    """Fraction of the OLD distribution <= old_value (in [0, 1])."""
    return float(np.searchsorted(np.asarray(old_dist), old_value, side="right")
                 / len(old_dist))


def quantile_at(dist: list[float], p: float) -> float:
    """Value at percentile p of the distribution."""
    arr = np.asarray(dist)
    return float(np.quantile(arr, min(max(p, 0.0), 1.0)))


def is_floor(old_value: float, old_dist: list[float]) -> bool:
    """Percentile-matching is only meaningful for values inside the old
    distribution. Values below its ~1st percentile (or its minimum) are
    loose floors, not percentile anchors — mapping them through the
    distribution's tail would collapse them to the new distribution's
    floor (planner v3 §5 extension, documented in the report)."""
    return old_value <= quantile_at(old_dist, 0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold recalibration (planner v3 §5).")
    parser.add_argument("--old", default="resnet18",
                        help="encoder whose distribution the current values were tuned to")
    parser.add_argument("--new", default="dinov2_registers_small",
                        help="encoder the config now uses")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = args.config or str(CONFIG_PATH)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    report = resolve_path(config["paths"]["evaluation_dir"]) / "encoder_comparison.json"
    with open(report) as f:
        comparison = json.load(f)
    if args.old not in comparison["encoders"] or args.new not in comparison["encoders"]:
        raise SystemExit(
            f"encoder_comparison.json lacks {args.old}/{args.new} — "
            f"run scripts/compare_encoders.py first (has: "
            f"{sorted(comparison['encoders'])})")

    old = comparison["encoders"][args.old]
    new = comparison["encoders"][args.new]
    old_same, old_diff = old["same_place_sims"], old["different_place_sims"]
    new_same, new_diff = new["same_place_sims"], new["different_place_sims"]

    # likelihood_temperature: the same-place std ratio was tried and REJECTED
    # (the pairwise std is inflated by far-apart same-place pairs, so it
    # overstates the likelihood's spread; the resulting temperature froze the
    # state machine into permanent UNCERTAIN). The correct mapping preserves
    # the softmax discrimination ratio: T_new = T_old * (median top-2
    # candidate-total gap, new / old), measured from the decision logs
    # (0.1062 / 0.0280 = 3.79). Leave temperature UNCHANGED here; apply the
    # decision-log ratio by hand (documented in config.yaml).
    temp_old = float(config["localization"]["likelihood_temperature"])
    temp_new = temp_old

    # Visual mean change drives every scale shift. Blended-score gates
    # (tracker/confidence/state machine) see only w_visual of it; the raw
    # similarity gate (navigation.confidence_threshold) sees all of it.
    delta_same_mean = new["same_place_consistency"] - old["same_place_consistency"]
    w_visual = float(config["localization"]["w_visual"])
    shift = {("mapping", "duplicate_similarity_threshold"): w_visual * delta_same_mean,
             ("navigation", "confidence_threshold"): delta_same_mean,
             ("localization", "tracking_threshold"): w_visual * delta_same_mean,
             ("localization", "lost_threshold"): w_visual * delta_same_mean,
             ("localization", "high_score"): w_visual * delta_same_mean,
             ("localization", "low_score"): w_visual * delta_same_mean,
             ("localization", "reacquired_threshold"): w_visual * delta_same_mean,
             ("localization", "arrived_threshold"): w_visual * delta_same_mean,
             ("localization", "recovery_threshold"): w_visual * delta_same_mean,
             ("localization", "transition_threshold"): w_visual * delta_same_mean}

    rows: list[dict] = []
    for section, key, dist, note in THRESHOLDS:
        old_dist = old_same if dist == "same" else old_diff
        new_dist = new_same if dist == "same" else new_diff
        old_value = float(config[section][key])
        if is_floor(old_value, old_dist):
            # below the old distribution minimum: percentile undefined.
            # Preserve the floor's distance below the correct-candidate
            # score by shifting it by the measured score change.
            new_value = max(0.0, old_value + shift[(section, key)])
            method = f"floor-shift ({shift[(section, key)]:+.4f})"
        else:
            p = percentile(old_value, old_dist)
            new_value = quantile_at(new_dist, p)
            method = f"percentile {p:.4f}"
        rows.append({
            "section": section, "key": key, "distribution": dist,
            "old_value": round(old_value, 4),
            "percentile": round(percentile(old_value, old_dist), 4),
            "new_value": round(new_value, 4),
            "method": method,
            "note": note,
        })

    out = resolve_path(config["paths"]["evaluation_dir"]) / "threshold_recalibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "old_encoder": args.old, "new_encoder": args.new,
            "method": ("percentile-matching on the pairwise same-place/"
                       "different-place similarity distributions from "
                       "encoder_comparison.json; likelihood_temperature via "
                       "same-place std ratio"),
            "thresholds": rows,
            "likelihood_temperature": {
                "old_value": temp_old, "new_value": round(temp_new, 4),
                "std_ratio": round(new["same_place_std"] / old["same_place_std"], 4),
            },
        }, f, indent=2)

    print(f"{'section.key':42s} {'dist':9s} {'old':>7s} {'new':>7s}  method")
    print("-" * 90)
    for r in rows:
        print(f"{r['section'] + '.' + r['key']:42s} {r['distribution']:9s} "
              f"{r['old_value']:7.3f} {r['new_value']:7.3f}  {r['method']}")
    print(f"{'localization.likelihood_temperature':42s} {'same-std':9s} "
          f"{temp_old:7.3f} {temp_new:7.3f}  std-ratio {temp_new / temp_old:.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
