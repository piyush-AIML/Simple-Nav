"""Candidate retrieval (§18 / Stage 15): FAISS becomes a candidate GENERATOR,
not the localization authority (§Rule 5).

Top-K exemplar rows are searched, aggregated by place, and returned as
Candidate records with margins — the downstream state estimator decides.
The legacy PlaceIndex.query stays untouched for baseline consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from src.mapping.place_builder import Place
from src.utils import setup_logger

logger = setup_logger("retrieval")


@dataclass
class Candidate:
    place_id: int
    visual_score: float              # best exemplar similarity (0..1)
    best_exemplar_id: str
    supporting_exemplar_count: int   # exemplars of this place inside top-K
    margin: float                    # best_score - second_best_score

    def to_dict(self) -> dict:
        return {
            "place_id": self.place_id,
            "visual_score": round(float(self.visual_score), 4),
            "best_exemplar_id": self.best_exemplar_id,
            "supporting_exemplar_count": self.supporting_exemplar_count,
            "margin": round(float(self.margin), 4),
        }


@dataclass
class RetrievalResult:
    candidates: list[Candidate]      # sorted by visual_score desc
    best_score: float
    second_best_score: float
    score_margin: float

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "best_score": round(float(self.best_score), 4),
            "second_best_score": round(float(self.second_best_score), 4),
            "score_margin": round(float(self.score_margin), 4),
        }


class CandidateRetriever:
    def __init__(self, bundle, top_k: int = 10, search_factor: int = 4):
        """bundle: MapBundle (Stage 14) — exemplars, exemplar_place_ids, store."""
        self.places = {p.place_id: p for p in bundle.places}
        self.top_k = top_k
        self.search_factor = search_factor
        self._store = bundle.store
        self._index = faiss.IndexFlatIP(bundle.exemplars.shape[1])
        self._index.add(bundle.exemplars)
        self._exemplar_place_ids = bundle.exemplar_place_ids
        self._exemplar_ids = bundle.exemplar_ids  # row -> observation id

    def retrieve(self, embedding: np.ndarray) -> RetrievalResult:
        if self._index.ntotal == 0:
            return RetrievalResult(candidates=[], best_score=0.0, second_best_score=0.0, score_margin=0.0)
        vec = embedding.reshape(1, -1).astype("float32")
        search_k = min(self._index.ntotal, max(self.top_k * self.search_factor, 10))
        scores, indices = self._index.search(vec, search_k)

        best_per_place: dict[int, tuple[float, str, int]] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            place_id = int(self._exemplar_place_ids[idx])
            exemplar_id = self._exemplar_ids[int(idx)]
            cur = best_per_place.get(place_id)
            if cur is None:
                best_per_place[place_id] = (float(score), exemplar_id, 1)
            else:
                best_per_place[place_id] = (
                    max(cur[0], float(score)),
                    exemplar_id if float(score) > cur[0] else cur[1],
                    cur[2] + 1,
                )

        ranked = sorted(best_per_place.items(), key=lambda kv: kv[1][0], reverse=True)[: self.top_k]
        candidates = [
            Candidate(place_id=pid, visual_score=sc, best_exemplar_id=eid,
                      supporting_exemplar_count=n, margin=0.0)
            for pid, (sc, eid, n) in ranked
        ]
        # margins: difference to the next-best place's score
        for i, cand in enumerate(candidates):
            second = candidates[i + 1].visual_score if i + 1 < len(candidates) else 0.0
            cand.margin = cand.visual_score - second

        best = candidates[0].visual_score if candidates else 0.0
        second = candidates[1].visual_score if len(candidates) > 1 else 0.0
        return RetrievalResult(
            candidates=candidates,
            best_score=best,
            second_best_score=second,
            score_margin=best - second,
        )
