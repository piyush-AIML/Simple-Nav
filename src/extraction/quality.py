"""Image quality gates for smart frame sampling (§5).

Cheap, CPU-only measurements — deliberately no neural networks here, so the
sampler can run on raw frames as fast as the video stream arrives.
"""

from __future__ import annotations

import cv2
import numpy as np


def frame_is_valid(frame: np.ndarray) -> bool:
    """Non-empty, 3-channel image."""
    return frame is not None and frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0


def blur_variance(frame: np.ndarray) -> float:
    """Variance of the Laplacian — the standard blur/sharpness proxy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(frame: np.ndarray) -> float:
    """Mean grayscale intensity in [0, 255]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def compute_quality(frame: np.ndarray) -> dict:
    """Return quality measurements + a normalized quality_score in [0, 1].

    The score favors sharp frames with mid-range brightness: blur variance is
    capped at 500 (beyond that is already very sharp), brightness penalty is
    distance from the 128 midpoint, scaled.
    """
    if not frame_is_valid(frame):
        return {"valid": False, "blur_variance": 0.0, "brightness": 0.0, "quality_score": 0.0}
    bv = blur_variance(frame)
    br = brightness(frame)
    sharpness = min(1.0, bv / 500.0)
    exposure = 1.0 - abs(br - 128.0) / 128.0
    score = 0.5 * sharpness + 0.5 * exposure
    return {
        "valid": True,
        "blur_variance": bv,
        "brightness": br,
        "quality_score": float(score),
    }


def reject_reasons(frame: np.ndarray, config: dict) -> list[str]:
    """Why this frame fails the quality gate; empty list = acceptable."""
    reasons: list[str] = []
    if not frame_is_valid(frame):
        return ["invalid_image"]
    if blur_variance(frame) < config.get("blur_threshold", 40.0):
        reasons.append("blurry")
    if brightness(frame) < config.get("dark_threshold", 20.0):
        reasons.append("too_dark")
    return reasons
