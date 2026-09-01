"""Tests for confidence calibration (§25/Stage 22): HIGH/MEDIUM/LOW/UNKNOWN
mapping and its boundary behavior."""

from src.localization.confidence import estimate_confidence
from src.localization.state_estimator import StateEstimate

CFG = {
    "high_score": 0.6, "high_margin": 0.15, "high_entropy": 0.4,
    "low_score": 0.3, "low_margin": 0.02, "unknown_high_mass": 0.4,
}


def est(best_score: float, unknown_mass: float = 0.0, entropy: float = 0.1,
        best_place_id: int = 1) -> StateEstimate:
    return StateEstimate(
        posterior={best_place_id: best_score},
        unknown_mass=unknown_mass, best_place_id=best_place_id,
        best_score=best_score, entropy=entropy,
    )


def test_high_requires_score_margin_and_entropy():
    assert estimate_confidence(est(0.8), 0.3, CFG) == "HIGH"
    assert estimate_confidence(est(0.8), 0.05, CFG) != "HIGH"   # small margin
    assert estimate_confidence(est(0.8), 0.3, CFG, ) == "HIGH"


def test_high_entropy_blocks_high():
    assert estimate_confidence(est(0.8, entropy=0.9), 0.3, CFG) != "HIGH"


def test_low_on_weak_score_or_tiny_margin():
    assert estimate_confidence(est(0.2), 0.3, CFG) == "LOW"
    assert estimate_confidence(est(0.5), 0.01, CFG) == "LOW"


def test_unknown_when_unknown_mass_high():
    assert estimate_confidence(est(0.5, unknown_mass=0.6), 0.3, CFG) == "UNKNOWN"


def test_medium_in_between():
    assert estimate_confidence(est(0.5), 0.1, CFG) == "MEDIUM"


def test_no_belief_is_unknown():
    empty = StateEstimate(posterior={}, unknown_mass=1.0, best_place_id=None,
                          best_score=0.0, entropy=1.0)
    assert estimate_confidence(empty, 0.0, CFG) == "UNKNOWN"
