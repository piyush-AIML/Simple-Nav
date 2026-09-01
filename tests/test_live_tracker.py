"""Tests for the live-tracking/re-routing state machine, using small
synthetic exemplar vectors instead of real images or a real CNN — this
keeps the test fast and focused purely on the tracking LOGIC (did it detect
a place change, did it recompute the route, did it hold position on a weak
frame), independent of embedding quality.
"""

import networkx as nx
import numpy as np

from src.live_tracker import LiveTracker
from src.localize import PlaceIndex

PLACE_NAMES = {"0": "Lobby", "1": "Corridor", "2": "Room101"}


def make_toy_index() -> PlaceIndex:
    # one clean, orthogonal exemplar per place
    exemplars = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )
    exemplar_place_ids = np.array([0, 1, 2], dtype="int64")
    return PlaceIndex(exemplars, exemplar_place_ids, PLACE_NAMES)


def make_toy_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1, weight=5)
    g.add_edge(1, 2, weight=5)
    return g


def make_tracker(threshold: float = 0.5) -> LiveTracker:
    return LiveTracker(
        place_index=make_toy_index(),
        graph=make_toy_graph(),
        place_names=PLACE_NAMES,
        confidence_threshold=threshold,
        smoothing_window=1,  # no smoothing lag, easier to assert on
    )


def test_first_frame_sets_position_and_route():
    tracker = make_tracker()
    tracker.set_destination(2)
    status = tracker.process_frame(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    assert status["place_id"] == 0
    assert status["changed"] is True
    assert status["directions"] == "Lobby -> Corridor -> Room101"


def test_staying_in_place_does_not_report_change():
    tracker = make_tracker()
    tracker.set_destination(2)
    tracker.process_frame(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    status = tracker.process_frame(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    assert status["changed"] is False


def test_moving_to_new_place_triggers_reroute():
    tracker = make_tracker()
    tracker.set_destination(2)
    tracker.process_frame(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))  # at Lobby
    status = tracker.process_frame(np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"))  # moved to Corridor
    assert status["place_id"] == 1
    assert status["changed"] is True
    assert status["directions"] == "Corridor -> Room101"


def test_arrival_is_detected():
    tracker = make_tracker()
    tracker.set_destination(2)
    tracker.process_frame(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    tracker.process_frame(np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"))
    status = tracker.process_frame(np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"))
    assert status["arrived"] is True
    assert status["directions"] is None


def test_low_confidence_frame_holds_last_position():
    tracker = make_tracker(threshold=0.5)
    tracker.set_destination(2)
    tracker.process_frame(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))  # confident: at Lobby
    # a weak/ambiguous vector, similarity to everything is low
    weak = np.array([0.1, 0.1, 0.1, 0.97], dtype="float32")
    status = tracker.process_frame(weak)
    assert status["low_confidence"] is True
    assert status["changed"] is False
    assert status["place_id"] == 0  # held last known position, didn't jump
