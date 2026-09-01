"""Tests for smart frame sampling (§5): quality gate, novelty gate,
transition-aware retention, temporal gap. All synthetic frames — no video
file needed.
"""

import cv2
import numpy as np
import pytest

from src.extraction.frames import SmartFrameSampler
from src.extraction.quality import blur_variance, brightness, frame_is_valid, reject_reasons

DEFAULT_CFG = {"sampling": {}}


def sharp_frame(value: int = 128) -> np.ndarray:
    """A sharp, mid-brightness synthetic frame (noise gives Laplacian variance)."""
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    img = img + rng.integers(0, 40, size=img.shape, dtype=np.uint8)
    return img


def dark_frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def blurred_frame() -> np.ndarray:
    return cv2.GaussianBlur(sharp_frame(128), (21, 21), 10)


def stripes(horizontal: bool) -> np.ndarray:
    """Sharp 8px black/white stripes. Vertical vs horizontal stripes stay very
    different even after the 32x32 descriptor downscale (mean brightness ~127,
    Laplacian variance high, so they pass the quality gate)."""
    stripe = np.kron(np.array([[0, 1]] * 8, dtype=np.uint8) * 255, np.ones((8, 8), dtype=np.uint8))
    img = np.stack([stripe] * 3, axis=-1)
    return img.transpose(1, 0, 2) if horizontal else img  # swap spatial axes


# ---------- quality gates ----------

def test_quality_rejects_invalid_dark_blurry():
    cfg = {"blur_threshold": 40.0, "dark_threshold": 20.0}
    assert reject_reasons(np.zeros((0, 0, 3), dtype=np.uint8), cfg) == ["invalid_image"]
    assert "too_dark" in reject_reasons(dark_frame(), cfg)
    assert "blurry" in reject_reasons(blurred_frame(), cfg)
    assert reject_reasons(sharp_frame(), cfg) == []


def test_quality_measurements_sane():
    bv = blur_variance(blurred_frame())
    br = brightness(dark_frame())
    assert bv < blur_variance(sharp_frame())
    assert br < 20.0
    assert frame_is_valid(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not frame_is_valid(np.zeros((4, 4), dtype=np.uint8))


# ---------- sampler behavior ----------

def test_identical_frames_are_redundant():
    sampler = SmartFrameSampler(DEFAULT_CFG)
    frame = sharp_frame()
    keep0, _ = sampler.decide(frame, 0, 0.0)
    assert keep0
    for i in range(1, 20):
        keep, reasons = sampler.decide(frame.copy(), i, i / 30.0)
        assert not keep, f"frame {i} should be redundant"
        assert "redundant" in reasons


def test_abrupt_change_is_kept():
    sampler = SmartFrameSampler({"sampling": {"min_interval_seconds": 0.0}})
    a = stripes(horizontal=False)
    b = stripes(horizontal=True)
    assert sampler.decide(a, 0, 0.0)[0]
    keep, reasons = sampler.decide(b, 1, 1.0)
    assert keep, reasons


def noise_pattern(seed: int, mean: int = 127) -> np.ndarray:
    """Sharp random noise around a mean luminance. Two patterns with very
    different means (40 vs 220) have near-opposite 32x32 descriptors."""
    rng = np.random.default_rng(seed)
    lo, hi = max(0, mean - 40), min(255, mean + 40)
    return rng.integers(lo, hi, size=(64, 64, 3), dtype=np.uint8)


def test_single_spike_is_not_a_transition():
    """One anomalous frame amid a run must not trigger the sustained-change
    force-keeping (§5: 'Do not treat one spike as a transition'). The spike
    itself is new information and is kept by novelty; the sustained-change
    machinery must stay quiet."""
    cfg = {"sampling": {"transition_window": 3, "transition_keep_after": 2, "novelty_threshold": 0.05, "min_interval_seconds": 0.0}}
    sampler = SmartFrameSampler(cfg)
    a = stripes(horizontal=False)
    b = stripes(horizontal=True)
    assert sampler.decide(a, 0, 0.0)[0]
    keep1, reasons = sampler.decide(b, 1, 1.0)  # novel -> kept
    assert keep1
    assert "sustained_change" not in reasons
    # back to a-frames: alternate keep/redundant is fine, but the sustained
    # window [0.75, 0.75, 0, 0, ...] must never fire
    for i in range(2, 12):
        _, reasons = sampler.decide(a, i, float(i))
        assert "sustained_change" not in reasons, f"spike aftermath frame {i}: {reasons}"


def test_sustained_change_triggers_retention():
    """A sustained change (>= window frames far away) force-keeps the change
    plus transition_keep_after following frames."""
    cfg = {"sampling": {"transition_window": 3, "transition_keep_after": 2, "novelty_threshold": 0.05, "min_interval_seconds": 0.0}}
    sampler = SmartFrameSampler(cfg)
    # alternate dark-ish and bright noise: every consecutive pair is far in
    # descriptor space, so the distance window stays full
    a = noise_pattern(seed=0, mean=40)
    assert sampler.decide(a, 0, 0.0)[0]
    keep1, reasons = sampler.decide(noise_pattern(1, mean=220), 1, 1.0)
    assert keep1 and "sustained_change" not in reasons  # window not full yet
    keep2, reasons = sampler.decide(noise_pattern(2, mean=40), 2, 2.0)
    assert keep2 and "sustained_change" not in reasons
    # third consecutive far frame completes the window -> force-kept + trigger
    b = noise_pattern(3, mean=220)
    keep3, reasons = sampler.decide(b, 3, 3.0)
    assert keep3 and "sustained_change" in reasons
    # next transition_keep_after frames are redundant but force-kept
    for i in (4, 5):
        keep, reasons = sampler.decide(b, i, float(i))
        assert keep, f"force-kept frame {i}: {reasons}"
    # afterwards, redundancy wins again
    keep, reasons = sampler.decide(b, 6, 6.0)
    assert not keep and "redundant" in reasons


def test_temporal_gap_limits_frequency():
    cfg = {"sampling": {"min_interval_seconds": 1.0, "novelty_threshold": 0.0}}
    sampler = SmartFrameSampler(cfg)
    frames = [sharp_frame(50 + i * 10) for i in range(5)]  # all mutually novel
    keep0, _ = sampler.decide(frames[0], 0, 0.0)
    assert keep0
    keep1, reasons = sampler.decide(frames[1], 1, 0.2)
    assert not keep1 and "too_frequent" in reasons
    keep2, _ = sampler.decide(frames[2], 2, 1.2)
    assert keep2


def test_dark_frames_never_accepted():
    sampler = SmartFrameSampler(DEFAULT_CFG)
    keep, reasons = sampler.decide(dark_frame(), 0, 0.0)
    assert not keep and "too_dark" in reasons
