"""Smart frame sampling (§5): keep frames that add spatial information.

Decision logic, in order:
    1. quality gate      — invalid / blurry / too dark  -> reject
    2. novelty gate      — nearly identical to the last ACCEPTED frame -> reject
    3. transition-aware  — a SUSTAINED visual change (>= transition_window
                           consecutive frames above transition_threshold)
                           force-keeps the change and the next few frames
    4. temporal gap      — never accept closer than min_interval_seconds

A single spike never counts as a transition (§5: "Do not treat one spike as
a transition"). The cheap descriptor is a 32x32 grayscale L2-normalized
downscale — no neural network involved.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.extraction.quality import reject_reasons
from src.utils import setup_logger

logger = setup_logger("smart_sampler")

DESCRIPTOR_SIZE = 32


def _descriptor(frame: np.ndarray) -> np.ndarray:
    """Cheap visual descriptor: mean-subtracted 32x32 grayscale structure +
    a small luminance tail (mean, std). Mean-subtraction matters: the raw
    grayscale vector is dominated by mean luminance (L2-normalized images
    with different brightness are ~98% similar), which would make the novelty
    gate blind to pattern changes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
    small = cv2.resize(gray, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE))
    mean = float(small.mean())
    structure = (small - mean).flatten()
    struct_norm = np.linalg.norm(structure)
    structure = structure / struct_norm if struct_norm > 0 else structure
    luminance = np.array([mean / 255.0, float(small.std()) / 128.0], dtype="float32")
    descriptor = np.concatenate([structure, luminance])
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm > 0 else descriptor


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - float(np.dot(a, b)))


@dataclass
class SampledFrame:
    frame: np.ndarray
    frame_idx: int
    timestamp: float


@dataclass
class SmartFrameSampler:
    config: dict[str, Any]
    log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        s = self.config.get("sampling", {})
        self.min_interval_seconds = float(s.get("min_interval_seconds", 0.5))
        self.novelty_threshold = float(s.get("novelty_threshold", 0.15))
        self.transition_threshold = float(s.get("transition_threshold", 0.4))
        self.transition_window = int(s.get("transition_window", 5))
        self.transition_keep_after = int(s.get("transition_keep_after", 2))
        self.blur_threshold = float(s.get("blur_threshold", 40.0))
        self.dark_threshold = float(s.get("dark_threshold", 20.0))
        self.reset()

    def reset(self) -> None:
        self._last_accepted_descriptor: np.ndarray | None = None
        self._last_accepted_time: float | None = None
        self._recent_distances: deque[float] = deque(maxlen=self.transition_window)
        self._transition_active = 0  # frames still force-kept after a sustained change
        self.log = []

    def decide(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> tuple[bool, list[str]]:
        """Return (keep, reasons). Reasons are empty when kept."""
        reasons: list[str] = []

        # 1. quality gate
        reasons.extend(reject_reasons(frame, self.config))
        if reasons:
            return False, reasons

        descriptor = _descriptor(frame)

        # 3. transition-aware retention (checked before novelty so a sustained
        #    change can force-keep even a temporarily redundant frame)
        keep_for_transition = False
        if self._transition_active > 0:
            keep_for_transition = True
            self._transition_active -= 1

        distance: float | None = None
        if self._last_accepted_descriptor is not None:
            distance = _cosine_distance(self._last_accepted_descriptor, descriptor)
            self._recent_distances.append(distance)
            sustained = (
                len(self._recent_distances) >= self.transition_window
                and all(d > self.transition_threshold for d in self._recent_distances)
            )
            if sustained and not keep_for_transition:
                keep_for_transition = True
                self._transition_active = self.transition_keep_after
                reasons.append("sustained_change")

        # 2. novelty gate
        if distance is not None and distance < self.novelty_threshold and not keep_for_transition:
            reasons.append("redundant")
            return False, reasons

        # 4. temporal gap
        if (
            self._last_accepted_time is not None
            and timestamp - self._last_accepted_time < self.min_interval_seconds
            and not keep_for_transition
        ):
            reasons.append("too_frequent")
            return False, reasons

        # accept
        self._last_accepted_descriptor = descriptor
        self._last_accepted_time = timestamp
        self.log.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "decision": "keep",
                "descriptor_distance": distance,
                "reasons": reasons,
            }
        )
        return True, reasons

    def process(self, video_path: Path) -> list[SampledFrame]:
        """Sample a whole video file. Logs every decision to self.log and
        returns the kept frames with their source frame index and timestamp.
        """
        self.reset()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        kept: list[SampledFrame] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps
            keep, reasons = self.decide(frame, frame_idx, timestamp)
            if not keep:
                self.log.append(
                    {"frame_idx": frame_idx, "timestamp": timestamp, "decision": "reject", "reasons": reasons}
                )
            else:
                kept.append(SampledFrame(frame=frame, frame_idx=frame_idx, timestamp=timestamp))
            frame_idx += 1
        cap.release()

        kept_count = len(kept)
        rejected = len(self.log) - kept_count
        logger.info(
            f"Sampled {kept_count}/{frame_idx} frames ({rejected} rejected: "
            f"{'blurry/dark/redundant/frequent per log'}) from {video_path}"
        )
        return kept

    def save_log(self, path: Path) -> None:
        path.write_text(json.dumps(self.log, indent=2))
