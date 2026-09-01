"""LocalizationTracker — the runtime state machine (§22-§24, Stages 19-22).

States: TRACKING / UNCERTAIN / LOST / REACQUIRING / ARRIVED (§22).

Per frame: retrieval (LOCAL or GLOBAL mode) -> graph terms -> candidate
scoring -> Bayes filter update -> stabilization (K-consecutive confirmation)
-> state transition -> confidence level -> route/directions.

The status dict stays BACKWARD-COMPATIBLE with the legacy LiveTracker
(place_id, place_name, confidence, changed, low_confidence, arrived, route,
directions) plus new keys (state, confidence_level, unknown_mass, mode),
so app.py / live_navigate.py can switch without UI rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.localization.candidate_scoring import score_candidates
from src.localization.confidence import estimate_confidence
from src.localization.graph_constraints import graph_terms_map
from src.localization.retrieval import CandidateRetriever
from src.localization.state_estimator import StateEstimate, StateEstimator
from src.mapping.map_artifact import MapBundle
from src.navigate import format_directions, get_route
from src.utils import setup_logger

logger = setup_logger("localization_tracker")

TRACKING = "TRACKING"
UNCERTAIN = "UNCERTAIN"
LOST = "LOST"
REACQUIRING = "REACQUIRING"
ARRIVED = "ARRIVED"

MODE_LOCAL = "LOCAL"
MODE_GLOBAL = "GLOBAL"


@dataclass
class LocalizationTracker:
    bundle: MapBundle
    config: dict
    destination_id: Optional[int] = None
    decision_log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        loc = self.config.get("localization", {})
        self._retriever = CandidateRetriever(
            self.bundle,
            top_k=int(loc.get("top_k", 10)),
            search_factor=int(loc.get("exemplar_search_factor", 4)),
        )
        self._places = {p.place_id: p for p in self.bundle.places}
        self._estimator = StateEstimator(self.bundle.graph, [], loc)
        self._state = UNCERTAIN
        self._mode = MODE_GLOBAL  # no previous place yet -> global first
        self._current_place_id: Optional[int] = None
        self._pending_place_id: Optional[int] = None
        self._pending_count = 0
        self._weak_streak = 0
        self._lost_streak = 0
        self._arrived_streak = 0
        self._confirmed_place_id: Optional[int] = None

    # ---------- configuration shortcuts ----------
    def _cfg(self, key: str, default):
        return self.config.get("localization", {}).get(key, default)

    def set_destination(self, destination_id: int) -> None:
        self.destination_id = destination_id

    # ---------- main entry point ----------
    def process_frame(
        self,
        embedding: np.ndarray,
        query_tags: dict | None = None,
        query_objects: list[dict] | None = None,
        timestamp: float | None = None,
    ) -> dict:
        loc = self.config.get("localization", {})

        # ---- retrieval (LOCAL vs GLOBAL per §24/Stage 21) ----
        if self._mode == MODE_GLOBAL:
            retrieval = self._retriever.retrieve(embedding)
        else:
            from src.localization.graph_constraints import local_candidate_set

            local = local_candidate_set(self._current_place_id, self.bundle.graph,
                                        int(loc.get("graph_radius", 2)))
            retrieval = self._retriever.retrieve(embedding)
            if local is not None:
                retrieval.candidates = [c for c in retrieval.candidates if c.place_id in local]

        # ---- scoring + Bayes update ----
        terms = graph_terms_map(retrieval.candidates, self._current_place_id,
                                self.bundle.graph, loc)
        scored = score_candidates(retrieval, query_tags, query_objects,
                                  self._current_place_id, terms, self._places, loc)
        # observation likelihood uses VISUAL+SEMANTIC evidence only — the
        # temporal/graph influence enters exclusively through the transition
        # prior (§Rule 5: retrieval proposes, temporal/graph decides)
        w_v = float(loc.get("w_visual", 0.5))
        w_s = float(loc.get("w_semantic", 0.25))
        evidence = {
            c.place_id: (w_v * c.visual_term + w_s * c.semantic_term) / (w_v + w_s)
            for c in scored
        }
        estimate = self._estimator.update(scored, evidence=evidence, timestamp=timestamp)

        # ---- stabilization (§23/Stage 20): K-consecutive confirmation ----
        confirmed = self._confirm_place(estimate, loc)

        # ---- state machine (§22/Stage 19) ----
        state, mode = self._advance_state(estimate, confirmed, loc)

        # ---- confidence (§25/Stage 22) ----
        confidence_level = estimate_confidence(estimate, retrieval.score_margin, loc)

        # ---- outputs ----
        # report the best place during initial acquisition (nothing confirmed
        # yet); afterwards the confirmed place is the one that matters
        place_id = self._current_place_id
        if place_id is None and estimate.best_place_id is not None:
            place_id = estimate.best_place_id
        arrived = False
        route, directions = None, None
        if place_id is not None and self.destination_id is not None:
            if place_id == self.destination_id and state in (TRACKING, ARRIVED):
                if state == TRACKING:
                    self._arrived_streak += 1
                    if self._arrived_streak >= int(loc.get("arrived_confirmations", 2)):
                        state = ARRIVED
                arrived = state == ARRIVED
            else:
                self._arrived_streak = 0
                route = get_route(self.bundle.graph, place_id, self.destination_id,
                                  self.config.get("routing", {}).get("weighted", True))
                directions = format_directions(route, self.bundle.place_names) if route else None

        place_name = (
            self.bundle.place_names.get(str(place_id), f"Place_{place_id}")
            if place_id is not None else "Unrecognized"
        )
        result = {
            "place_id": place_id,
            "place_name": place_name,
            "confidence": float(estimate.best_score),
            "changed": confirmed,
            "low_confidence": state in (UNCERTAIN, LOST),
            "arrived": arrived,
            "route": route,
            "directions": directions,
            "state": state,
            "confidence_level": confidence_level,
            "unknown_mass": float(estimate.unknown_mass),
            "mode": self._mode,
        }
        self._log_decision(result, estimate, retrieval, loc)
        return result

    # ---------- stabilization (Stage 20) ----------
    def _confirm_place(self, estimate: StateEstimate, loc: dict) -> bool:
        """A place change is confirmed only after K consecutive supporting
        observations, or immediately when the posterior clears a high
        threshold (§23)."""
        k = int(loc.get("transition_confirmation_count", 3))
        threshold = float(loc.get("transition_threshold", 0.65))

        best = estimate.best_place_id
        if best is None:
            self._pending_count = 0
            return False
        if best == self._current_place_id:
            self._pending_place_id = None
            self._pending_count = 0
            return False

        if estimate.best_score > threshold:
            self._confirmed_place_id = best
            self._pending_place_id = None
            self._pending_count = 0
            return self._set_current(best)

        if self._pending_place_id == best:
            self._pending_count += 1
        else:
            self._pending_place_id = best
            self._pending_count = 1
        if self._pending_count >= k:
            self._confirmed_place_id = best
            self._pending_place_id = None
            self._pending_count = 0
            return self._set_current(best)
        return False

    def _set_current(self, place_id: int) -> bool:
        changed = place_id != self._current_place_id
        self._current_place_id = place_id
        self._weak_streak = 0
        self._lost_streak = 0
        return changed

    # ---------- state machine (Stage 19) ----------
    def _advance_state(self, estimate: StateEstimate, confirmed: bool, loc: dict) -> tuple[str, str]:
        best = estimate.best_place_id
        best_score = estimate.best_score
        tracking_thr = float(loc.get("tracking_threshold", 0.5))
        tracking_entropy = float(loc.get("tracking_entropy", 0.6))
        uncertain_floor = float(loc.get("uncertain_floor", 0.2))
        lost_thr = float(loc.get("lost_threshold", 0.2))
        lost_unknown = float(loc.get("lost_unknown_mass", 0.6))
        lost_after = int(loc.get("lost_after", 3))
        reacquired_thr = float(loc.get("reacquired_threshold", 0.5))
        global_after = int(loc.get("global_reacquisition_after", 3))
        recovery_thr = float(loc.get("recovery_threshold", 0.45))

        strong = best_score > tracking_thr and estimate.entropy < tracking_entropy

        if self._state == LOST:
            self._lost_streak += 1
            if best_score > reacquired_thr:
                self._state = TRACKING
                self._mode = MODE_LOCAL
                self._lost_streak = 0
            else:
                self._state = REACQUIRING
            return self._state, self._mode

        if self._state == REACQUIRING:
            if best_score > reacquired_thr:
                self._state = TRACKING
                self._mode = MODE_LOCAL
            else:
                self._state = REACQUIRING
                self._mode = MODE_GLOBAL
            return self._state, self._mode

        # TRACKING / UNCERTAIN / ARRIVED.
        # `stable`: no pending place switch — TRACKING must survive stable
        # frames (confirmed only fires on CHANGES), and must be held during
        # an in-progress switch until the K-consecutive rule confirms it.
        stable = self._pending_place_id is None or self._pending_place_id == best
        if strong and (stable or self._current_place_id is None):
            self._state = TRACKING
            self._weak_streak = 0
            if self._current_place_id is not None:
                self._mode = MODE_LOCAL
        else:
            self._weak_streak += 1
            self._state = UNCERTAIN
            # LOST: posterior collapsed OR the belief mass is mostly unknown
            # (§Rule 4 — 'unknown' beats a confident wrong guess)
            if self._weak_streak >= lost_after and (
                best_score < lost_thr or estimate.unknown_mass > lost_unknown
            ):
                self._state = LOST
                self._mode = MODE_GLOBAL
                self._lost_streak = 0
                self._estimator.reset()
        return self._state, self._mode

    # ---------- decision log (Stage 27) ----------
    def _log_decision(self, result: dict, estimate: StateEstimate, retrieval, loc: dict) -> None:
        self.decision_log.append(
            {
                "state": result["state"],
                "confidence_level": result["confidence_level"],
                "unknown_mass": result["unknown_mass"],
                "mode": result["mode"],
                "place_id": result["place_id"],
                "best_score": result["confidence"],
                "candidates": [c.to_dict() for c in retrieval.candidates[:5]],
                "posterior": estimate.to_dict()["posterior"],
            }
        )
        if len(self.decision_log) > 20000:
            self.decision_log = self.decision_log[-10000:]
