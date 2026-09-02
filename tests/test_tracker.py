"""Tests for the LocalizationTracker state machine (§22-§24): tracking,
arrival, uncertain holds, lost -> reacquisition, stabilization, and the
backward-compatible status dict."""

from tests.conftest import V0, V1, V2, WEAK, make_toy_bundle

from src.localization.tracker import (
    ARRIVED,
    LOST,
    REACQUIRING,
    TRACKING,
    UNCERTAIN,
    LocalizationTracker,
)

LOC_CFG = {
    "localization": {
        "top_k": 10, "exemplar_search_factor": 4,
        "w_visual": 0.5, "w_semantic": 0.25, "w_temporal": 0.15, "w_graph": 0.1,
        "graph_radius": 2, "graph_penalty_strength": 0.5,
        "self_transition_prior": 0.7, "far_transition_prior": 0.05, "likelihood_temperature": 0.1,
        "tracking_threshold": 0.5, "tracking_entropy": 0.6, "uncertain_floor": 0.2,
        "lost_after": 3, "lost_threshold": 0.2, "lost_unknown_mass": 0.6,
        "reacquired_threshold": 0.5,
        "arrived_threshold": 0.6, "arrived_confirmations": 2,
        "transition_confirmation_count": 3, "transition_threshold": 0.65,
        "global_reacquisition_after": 3, "recovery_threshold": 0.45,
        "high_score": 0.6, "high_margin": 0.15, "high_entropy": 0.4,
        "low_score": 0.3, "low_margin": 0.02, "unknown_high_mass": 0.4,
    },
    "routing": {"weighted": True},
}


def make_tracker(tmp_path, destination=None) -> LocalizationTracker:
    tracker = LocalizationTracker(make_toy_bundle(tmp_path), LOC_CFG, destination_id=destination)
    return tracker


def test_tracking_converges_and_keeps_status_contract(tmp_path):
    tracker = make_tracker(tmp_path, destination=2)
    status = None
    for _ in range(4):
        status = tracker.process_frame(V0)
    assert status["state"] == TRACKING
    assert status["place_id"] == 0
    # backward-compatible keys all present
    for key in ("place_id", "place_name", "confidence", "changed", "low_confidence",
                "arrived", "route", "directions"):
        assert key in status
    assert status["route"] == [0, 1, 2]
    assert status["directions"] == "Lobby -> Corridor -> Room101"


def test_term_breakdown_reports_all_four_terms(tmp_path):
    """Planner v3 §8: the status dict carries the per-candidate evidence
    breakdown the demo app displays — visual/semantic/temporal/graph terms
    for the top-3 scored candidates."""
    tracker = make_tracker(tmp_path)
    status = tracker.process_frame(V0)
    breakdown = status["term_breakdown"]
    assert len(breakdown) == 3  # top-3 candidates
    best = breakdown[0]
    assert best["place_id"] == 0
    assert best["place_name"] == "Lobby"
    assert best["visual_term"] >= best["semantic_term"]  # exact exemplar hit
    for key in ("visual_term", "semantic_term", "temporal_term", "graph_term", "total"):
        assert 0.0 <= best[key] <= 1.0


def test_arrival_is_detected_after_confirmations(tmp_path):
    tracker = make_tracker(tmp_path, destination=2)
    for _ in range(4):
        tracker.process_frame(V0)
    status = None
    for _ in range(4):
        status = tracker.process_frame(V2)
    assert status["arrived"] is True
    assert status["state"] == ARRIVED


def test_uncertain_holds_last_position(tmp_path):
    tracker = make_tracker(tmp_path, destination=2)
    for _ in range(4):
        tracker.process_frame(V0)
    status = tracker.process_frame(WEAK)
    assert status["state"] == UNCERTAIN
    assert status["low_confidence"] is True
    assert status["place_id"] == 0  # held, no jump


def test_lost_then_reacquisition(tmp_path):
    tracker = make_tracker(tmp_path, destination=2)
    for _ in range(4):
        tracker.process_frame(V0)
    # weak frames accumulate unknown mass; once it clears the LOST bar the
    # tracker drops to LOST (mode -> GLOBAL for reacquisition)
    status = None
    for _ in range(3):
        status = tracker.process_frame(WEAK)
    assert status["state"] == LOST
    assert status["mode"] == "GLOBAL"
    # further weak evidence keeps REACQUIRING
    status = tracker.process_frame(WEAK)
    assert status["state"] == REACQUIRING
    # strong new evidence at place 2 -> posterior clears the reacquisition
    # threshold after a frame or two -> TRACKING at 2
    status = tracker.process_frame(V2)
    assert status["state"] in (REACQUIRING, TRACKING)
    for _ in range(3):
        status = tracker.process_frame(V2)
    # destination IS place 2 — arriving there correctly triggers ARRIVED
    assert status["state"] in (TRACKING, ARRIVED)
    assert status["place_id"] == 2


def test_high_posterior_confirms_immediately(tmp_path):
    """With the default transition_threshold (0.65), the corridor posterior
    crosses the threshold on the second V1 frame -> immediate confirmation
    (the K-consecutive rule is the fallback for weaker evidence)."""
    tracker = make_tracker(tmp_path, destination=2)
    for _ in range(4):
        tracker.process_frame(V0)
    changed_flags = []
    for _ in range(2):
        changed_flags.append(tracker.process_frame(V1)["changed"])
    assert changed_flags == [False, True]
    assert tracker._current_place_id == 1


def test_transition_needs_k_consecutive_frames(tmp_path):
    """With a high transition_threshold, only the K-consecutive rule can
    confirm the move — and it needs exactly K frames."""
    cfg = dict(LOC_CFG)
    cfg["localization"] = {**cfg["localization"], "transition_threshold": 0.95}
    tracker = LocalizationTracker(make_toy_bundle(tmp_path), cfg, destination_id=2)
    for _ in range(4):
        tracker.process_frame(V0)
    changed_flags = []
    for _ in range(2):
        changed_flags.append(tracker.process_frame(V1)["changed"])
    assert changed_flags == [False, False]
    assert tracker._current_place_id == 0
    # third consecutive corridor frame confirms (K = transition_confirmation_count)
    status = tracker.process_frame(V1)
    assert status["changed"] is True
    assert status["place_id"] == 1


def test_no_flicker_on_alternating_evidence(tmp_path):
    tracker = make_tracker(tmp_path, destination=2)
    for _ in range(4):
        tracker.process_frame(V0)
    for i in range(6):
        status = tracker.process_frame(V1 if i % 2 == 0 else V0)
        assert status["changed"] is False, f"flicker at frame {i}"
    assert tracker._current_place_id == 0


def test_first_frame_uses_global_mode(tmp_path):
    tracker = make_tracker(tmp_path, destination=2)
    status = tracker.process_frame(V0)
    assert status["mode"] == "GLOBAL"  # no previous place yet
    assert status["state"] in (UNCERTAIN, TRACKING)
    assert status["place_id"] == 0


def test_decision_log_populated(tmp_path):
    tracker = make_tracker(tmp_path)
    tracker.process_frame(V0)
    tracker.process_frame(V1)
    assert len(tracker.decision_log) == 2
    assert "posterior" in tracker.decision_log[0]
    assert "state" in tracker.decision_log[0]
