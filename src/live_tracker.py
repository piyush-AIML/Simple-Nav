"""Shared live-tracking + re-routing logic.

This is deliberately factored out of both live_navigate.py (webcam CLI) and
app.py's Live Mode tab, so "did the tracked place change, do we need to
recompute the route" exists in exactly one place and can be unit-tested
without needing an actual camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from src.localize import PlaceIndex, LiveLocalizer
from src.navigate import format_directions, get_route


@dataclass
class LiveTracker:
    place_index: PlaceIndex
    graph: nx.Graph
    place_names: dict[str, str]
    confidence_threshold: float = 0.40
    smoothing_window: int = 5
    use_weighted_routing: bool = True
    destination_id: Optional[int] = None

    _localizer: LiveLocalizer = field(init=False, repr=False)
    _current_place_id: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._localizer = LiveLocalizer(self.place_index, window=self.smoothing_window)

    def set_destination(self, destination_id: int) -> None:
        self.destination_id = destination_id

    def process_frame(self, embedding: np.ndarray) -> dict:
        """Feed in one embedded frame. Returns a status dict describing what
        happened, so callers (CLI or UI) can react without duplicating logic:

            place_id       int | None   currently tracked place
            place_name     str
            confidence     float        best raw similarity score this frame
            changed        bool         did the tracked place change this update
            low_confidence bool         was this frame too weak to trust
            arrived        bool         reached the destination
            route          list[int] | None
            directions     str | None
        """
        _, _, score = self.place_index.query(embedding, top_k=1)[0]
        low_confidence = score < self.confidence_threshold

        if low_confidence:
            # Hold the last known position rather than jumping on a weak frame.
            place_id = self._current_place_id
            changed = False
        else:
            place_id, _ = self._localizer.update(embedding)
            changed = place_id != self._current_place_id
            self._current_place_id = place_id

        place_name = (
            self.place_names.get(str(place_id), f"Place_{place_id}")
            if place_id is not None
            else "Unrecognized"
        )

        result = {
            "place_id": place_id,
            "place_name": place_name,
            "confidence": score,
            "changed": changed,
            "low_confidence": low_confidence,
            "arrived": False,
            "route": None,
            "directions": None,
        }

        if place_id is not None and self.destination_id is not None:
            if place_id == self.destination_id:
                result["arrived"] = True
            else:
                route = get_route(self.graph, place_id, self.destination_id, self.use_weighted_routing)
                result["route"] = route
                result["directions"] = format_directions(route, self.place_names) if route else None

        return result
