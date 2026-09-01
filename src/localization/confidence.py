"""Confidence calibration (§25 / Stage 22): map estimator output to
HIGH / MEDIUM / LOW / UNKNOWN — deliberately NOT percentages, since a raw
similarity value is not a probability (§25).

Inputs: posterior best score, score margin, posterior entropy, unknown mass.
"""

from __future__ import annotations

from src.localization.state_estimator import StateEstimate


def estimate_confidence(
    estimate: StateEstimate,
    score_margin: float,
    config: dict | None = None,
) -> str:
    cfg = config or {}
    high_score = float(cfg.get("high_score", 0.6))
    high_margin = float(cfg.get("high_margin", 0.15))
    high_entropy = float(cfg.get("high_entropy", 0.4))
    low_score = float(cfg.get("low_score", 0.3))
    low_margin = float(cfg.get("low_margin", 0.02))
    unknown_high_mass = float(cfg.get("unknown_high_mass", 0.4))

    if estimate.unknown_mass > unknown_high_mass:
        return "UNKNOWN"
    if estimate.best_score > high_score and score_margin > high_margin and estimate.entropy < high_entropy:
        return "HIGH"
    if estimate.best_score < low_score or score_margin < low_margin:
        return "LOW"
    return "MEDIUM"
