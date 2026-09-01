"""Tests for the Stage 24 runtime gate (planner v2 §7): cheap pre-filter
before detector/VLM/encoder, reusing the Stage 03 novelty descriptor."""

import numpy as np

from src.extraction.frames import _descriptor
from src.runtime.gate import GateDecision, should_process

CFG = {"runtime": {"max_stale_seconds": 5.0}, "sampling": {"novelty_threshold": 0.15}}


def frame(color: int = 100) -> np.ndarray:
    return np.full((64, 64, 3), color, dtype=np.uint8)


def test_first_frame_always_expensive():
    d = should_process(frame(), None, None, CFG)
    assert d.run_expensive and d.reason == "novel"


def test_identical_consecutive_frames_redundant_every_time():
    f = frame()
    desc = _descriptor(f)
    first = should_process(f, None, 0.0, CFG, now=0.0)
    assert first.reason == "novel"
    for t in range(1, 6):
        d = should_process(f, desc, float(t - 1), CFG, now=float(t))
        assert d == GateDecision(run_expensive=False, reason="redundant")


def test_novel_frame_passes():
    # frame(0)'s descriptor is the zero vector; frame(255)'s is a unit vector
    # -> distance 1.0, far above the novelty threshold
    d = should_process(frame(255), _descriptor(frame(0)), 0.0, CFG, now=0.1)
    assert d.run_expensive and d.reason == "novel"


def test_stale_state_forces_run_anyway():
    """Frames past max_stale_seconds with no novelty -> forced run (slow
    gradual changes must not stick on stale state)."""
    f = frame()
    d = should_process(f, _descriptor(f), 0.0, CFG, now=10.0)
    assert d == GateDecision(run_expensive=True, reason="forced_interval")


def test_corrupt_frame_fails_toward_expensive():
    """Never silently skip: a decode error must not freeze tracking."""
    assert should_process(None, _descriptor(frame()), 0.0, CFG, now=0.1).run_expensive
    assert should_process(np.zeros((64, 64), dtype=np.uint8), _descriptor(frame()),
                          0.0, CFG, now=0.1).run_expensive
    assert should_process("not an image", None, None, CFG).run_expensive


def test_gate_never_raises_on_garbage_config():
    # empty config -> defaults applied; novel frame -> expensive, no crash
    d = should_process(frame(255), _descriptor(frame(0)), 0.0, {}, now=0.0)
    assert d.run_expensive
