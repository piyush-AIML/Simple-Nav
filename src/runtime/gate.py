"""Runtime gate (Stage 24, planner v2 §7): the cheap pre-filter that runs
BEFORE detector/VLM/encoder in the live localization loop.

    camera frame
       -> should_process() (near-zero cost)
       -> "redundant"          -> reuse last state, skip everything below
       -> "novel"|"forced_interval" -> full expensive pipeline

Deliberately reuses the Stage 03 mean-subtracted 32x32 descriptor + novelty
threshold from src/extraction/frames.py — the planner forbids building a
second novelty metric. The one addition is a forced interval
(runtime.max_stale_seconds): even if nothing looks novel, the expensive path
re-runs at least every N seconds so a slow, gradual scene change (walking
slowly down a long uniform corridor) can't leave the tracker on stale state.

Failure bias: a corrupt/unreadable frame returns run_expensive=True — fail
toward doing the expensive check, never toward silently skipping, since
skipping on a decode error could freeze tracking on stale state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.extraction.frames import _cosine_distance, _descriptor  # Stage 03 descriptor — reuse, don't rebuild
from src.extraction.quality import frame_is_valid


@dataclass
class GateDecision:
    run_expensive: bool
    reason: str  # "novel" | "redundant" | "forced_interval"


def should_process(
    frame,
    last_processed_embedding_lowres,
    last_processed_ts,
    config: dict,
    now: float | None = None,
) -> GateDecision:
    """Cheap pre-filter BEFORE detector/VLM/encoder run.

    `last_processed_embedding_lowres`: the Stage 03 descriptor of the last
    frame the expensive path actually processed. `last_processed_ts`: when
    that happened (time.monotonic() seconds in the live loop). `now` is an
    injectable clock for tests; defaults to time.monotonic().

    Never raises — bad input yields run_expensive=True ("novel"), so the
    expensive path sees it and degrades safely on its own terms.
    """
    rt = config.get("runtime", {}) or {}
    max_stale = float(rt.get("max_stale_seconds", 5.0))
    novelty_threshold = float((config.get("sampling", {}) or {}).get("novelty_threshold", 0.15))
    now = time.monotonic() if now is None else now

    try:
        valid = frame_is_valid(frame)
    except Exception:
        valid = False  # non-array garbage input
    if not valid:
        # corrupt/unreadable frame -> let the expensive path handle it
        return GateDecision(run_expensive=True, reason="novel")

    if last_processed_ts is not None and now - last_processed_ts > max_stale:
        return GateDecision(run_expensive=True, reason="forced_interval")

    if last_processed_embedding_lowres is None:
        return GateDecision(run_expensive=True, reason="novel")

    distance = _cosine_distance(last_processed_embedding_lowres, _descriptor(frame))
    if distance < novelty_threshold:
        return GateDecision(run_expensive=False, reason="redundant")
    return GateDecision(run_expensive=True, reason="novel")
