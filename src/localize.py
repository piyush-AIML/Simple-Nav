"""Stage 5 (runtime): "where am I" via simple vector retrieval.

This is the whole localization method: embed the query image, and do a
nearest-neighbor lookup against stored exemplar vectors using a flat
(brute-force) FAISS index over cosine similarity — no probabilistic
filtering, no graph-aware reasoning. A place can have multiple exemplar
vectors (see build_places.py), so we search all exemplars and then collapse
results down to unique places, keeping each place's best-scoring exemplar.

Adds two small but meaningful behaviors on top of plain nearest-neighbor:
  - a confidence threshold: if the best match is too weak, report "not
    confidently recognized" instead of forcing a guess
  - majority-vote smoothing across a sliding window of recent predictions,
    for use with a sequence of frames (see LiveLocalizer)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import faiss
import numpy as np

from src.embedder import embed_image
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("localize")

NOT_RECOGNIZED = "Unrecognized"


class PlaceIndex:
    """Wraps a flat FAISS index over (possibly multiple-per-place) exemplar vectors."""

    def __init__(self, exemplars: np.ndarray, exemplar_place_ids: np.ndarray, place_names: dict[str, str]):
        self.exemplar_place_ids = exemplar_place_ids
        self.place_names = place_names
        self.index = faiss.IndexFlatIP(exemplars.shape[1])  # inner product == cosine (vectors are unit-norm)
        self.index.add(exemplars)

    @classmethod
    def load(cls, exemplars_file: Path, exemplar_ids_file: Path, names_file: Path) -> "PlaceIndex":
        exemplars = np.load(exemplars_file)
        exemplar_place_ids = np.load(exemplar_ids_file)
        with open(names_file, "r") as f:
            place_names = json.load(f)
        return cls(exemplars, exemplar_place_ids, place_names)

    def query(self, embedding: np.ndarray, top_k: int = 3) -> list[tuple[int, str, float]]:
        """Return up to top_k distinct PLACES as (place_id, place_name, score),
        best score first. Internally searches more exemplar rows than top_k
        since several rows can belong to the same place.
        """
        vec = embedding.reshape(1, -1).astype("float32")
        search_k = min(self.index.ntotal, max(top_k * 4, 10))
        scores, indices = self.index.search(vec, search_k)

        best_score_per_place: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            place_id = int(self.exemplar_place_ids[idx])
            if place_id not in best_score_per_place or score > best_score_per_place[place_id]:
                best_score_per_place[place_id] = float(score)

        ranked = sorted(best_score_per_place.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(pid, self.place_names.get(str(pid), f"Place_{pid}"), score) for pid, score in ranked]


class LiveLocalizer:
    """Majority-vote smoothing over a sliding window of recent predictions.

    Use this when localizing from a sequence of frames (e.g. a short video
    or a live camera loop) rather than a single uploaded photo — it reduces
    flicker between visually-similar neighboring places.
    """

    def __init__(self, place_index: PlaceIndex, window: int = 5):
        self.place_index = place_index
        self.history: deque[int] = deque(maxlen=window)

    def update(self, embedding: np.ndarray) -> tuple[int, str]:
        top = self.place_index.query(embedding, top_k=1)[0]
        place_id = top[0]
        self.history.append(place_id)
        winner_id, _ = Counter(self.history).most_common(1)[0]
        winner_name = self.place_index.place_names.get(str(winner_id), f"Place_{winner_id}")
        return winner_id, winner_name


def localize_image(
    image_path: str, place_index: PlaceIndex, confidence_threshold: float = 0.0
) -> tuple[int | None, str, float]:
    """Localize a single image. If the best score is below
    confidence_threshold, returns (None, NOT_RECOGNIZED, score) instead of
    forcing a guess.
    """
    embedding = embed_image(image_path)
    place_id, name, score = place_index.query(embedding, top_k=1)[0]
    if score < confidence_threshold:
        return None, NOT_RECOGNIZED, score
    return place_id, name, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize a single image against the map.")
    parser.add_argument("image", help="Path to a query image")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    exemplars_file = resolve_path(config["paths"]["place_exemplars_file"])
    exemplar_ids_file = resolve_path(config["paths"]["exemplar_place_ids_file"])
    names_file = resolve_path(config["paths"]["place_names_file"])
    threshold = config["navigation"]["confidence_threshold"]

    place_index = PlaceIndex.load(exemplars_file, exemplar_ids_file, names_file)
    place_id, name, score = localize_image(args.image, place_index, threshold)

    if place_id is None:
        logger.warning(f"Location not confidently recognized (best similarity={score:.4f} < {threshold})")
    else:
        logger.info(f"Predicted place: {name} (id={place_id}, similarity={score:.4f})")


if __name__ == "__main__":
    main()
