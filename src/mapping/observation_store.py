"""ObservationStore (§9): the shared memory used by both graph formation and
localization — a vector index for similarity search plus a metadata store,
deliberately NOT forcing every metadata field into the vector database.

On-disk layout (under data/observations/):
    observations.jsonl    metadata rows (no embeddings)
    embeddings.npy        (N, D) aligned with file order
    encoder.json          {"model", "version", "dimension"}
    id_order.json         observation ids in index row order
    index.faiss           FAISS IndexFlatIP (cosine, vectors are unit-norm)

Given an observation id the store recovers all metadata AND the original
frame path (§9 acceptance).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.mapping.observations import (
    Observation,
    load_observations_jsonl,
    save_observations_jsonl,
    sort_observations,
)
from src.utils import setup_logger

logger = setup_logger("observation_store")

FILES = ("observations.jsonl", "embeddings.npy", "encoder.json", "id_order.json", "index.faiss")


class ObservationStoreError(RuntimeError):
    pass


class ObservationStore:
    def __init__(self, obs_dir: Path):
        self.obs_dir = Path(obs_dir)
        self.observations: list[Observation] = []
        self._index: faiss.Index | None = None
        self._id_order: list[str] = []

    # ---------- construction ----------

    def add(self, observations: list[Observation]) -> None:
        """Append observations (rebuilding the index). Ids must be unique."""
        existing = {o.id for o in self.observations}
        for obs in observations:
            if obs.id in existing:
                raise ObservationStoreError(f"Duplicate observation id: {obs.id}")
            if obs.embedding is None:
                raise ObservationStoreError(f"Observation {obs.id} has no embedding")
            existing.add(obs.id)
        self.observations.extend(observations)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        ordered = sort_observations(self.observations)
        self._id_order = [o.id for o in ordered]
        embeddings = np.stack([o.embedding for o in ordered]).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

    # ---------- queries ----------

    def get(self, obs_id: str) -> Observation:
        """All metadata + embedding + original frame path for one id."""
        for obs in self.observations:
            if obs.id == obs_id:
                return obs
        raise KeyError(f"Unknown observation id: {obs_id}")

    def all(self) -> list[Observation]:
        return sort_observations(self.observations)

    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[Observation, float]]:
        """Nearest observations by cosine similarity (inner product)."""
        if self._index is None:
            raise ObservationStoreError("Store is empty — nothing to search")
        vec = embedding.reshape(1, -1).astype("float32")
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
        by_id = {o.id: o for o in self.observations}
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            obs = by_id[self._id_order[int(idx)]]
            results.append((obs, float(score)))
        return results

    def __len__(self) -> int:
        return len(self.observations)

    # ---------- persistence ----------

    def save(self, encoder_name: str) -> None:
        """Persist the store. encoder_name is recorded in encoder.json and
        must be the encoder that produced the stored embeddings (planner v3
        §6) — callers pass the real value from get_encoder(config), never a
        literal, so the file can't silently lie about the vectors."""
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        save_observations_jsonl(self.observations, self.obs_dir / "observations.jsonl")
        embeddings = np.stack([o.embedding for o in self.observations if o.embedding is not None]).astype("float32")
        np.save(self.obs_dir / "embeddings.npy", embeddings)
        with open(self.obs_dir / "encoder.json", "w") as f:
            json.dump(
                {"model": encoder_name, "dimension": int(embeddings.shape[1])},
                f,
                indent=2,
            )
        with open(self.obs_dir / "id_order.json", "w") as f:
            json.dump(self._id_order, f, indent=2)
        if self._index is not None:
            faiss.write_index(self._index, str(self.obs_dir / "index.faiss"))

    @classmethod
    def load(cls, obs_dir: Path) -> "ObservationStore":
        obs_dir = Path(obs_dir)
        missing = [name for name in FILES if not (obs_dir / name).exists()]
        if missing:
            raise ObservationStoreError(
                f"ObservationStore at {obs_dir} is missing files: {missing}"
            )
        store = cls(obs_dir)
        embeddings = np.load(obs_dir / "embeddings.npy")
        store.observations = load_observations_jsonl(obs_dir / "observations.jsonl", embeddings)
        with open(obs_dir / "id_order.json") as f:
            store._id_order = json.load(f)
        store._index = faiss.read_index(str(obs_dir / "index.faiss"))
        logger.info(f"Loaded ObservationStore: {len(store.observations)} observations from {obs_dir}")
        return store
