"""Observation: the single object representing visual evidence at a known
point in time (§4 of the planner).

Every accepted frame is traceable: video -> timestamp -> image -> embedding ->
metadata. Observation is the common currency passed between frame extraction,
mapping, and localization; loose parallel arrays (embeddings.npy + frame_names
.json + place_assignments.json) are gradually replaced by it, while the legacy
files keep being produced for the baseline.

Embeddings are intentionally NOT serialized to JSONL (they live in a parallel
.npy array aligned by file order — see ObservationStore, Stage 05).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Observation:
    id: str
    timestamp: float              # seconds into the source video
    frame_path: str               # path to the saved frame image
    embedding: np.ndarray | None = None          # never serialized to JSONL
    quality_score: float | None = None
    objects: list[dict] = field(default_factory=list)
    # [{"class": ..., "confidence": ..., "bbox": [x1, y1, x2, y2]}]
    scene_tags: dict | None = None
    # {"scene_type": ..., "landmarks": [...], "navigation_relevance": [...], "description": ...}
    landmarks: list[str] = field(default_factory=list)
    segment_id: str | None = None
    place_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Metadata dict for JSONL serialization (embedding excluded)."""
        d = asdict(self)
        d.pop("embedding", None)
        return d


def observation_from_dict(d: dict[str, Any], embedding: np.ndarray | None = None) -> Observation:
    return Observation(
        id=d["id"],
        timestamp=float(d["timestamp"]),
        frame_path=d["frame_path"],
        embedding=embedding,
        quality_score=d.get("quality_score"),
        objects=d.get("objects") or [],
        scene_tags=d.get("scene_tags"),
        landmarks=d.get("landmarks") or [],
        segment_id=d.get("segment_id"),
        place_id=d.get("place_id"),
    )


def save_observations_jsonl(observations: list[Observation], path: Path) -> None:
    """Save metadata rows (no embeddings) as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for obs in observations:
            f.write(json.dumps(obs.to_dict()) + "\n")


def load_observations_jsonl(
    path: Path, embeddings: np.ndarray | None = None
) -> list[Observation]:
    """Load JSONL metadata; optionally attach embeddings row-aligned."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    observations = []
    for i, row in enumerate(rows):
        emb = embeddings[i] if embeddings is not None and i < len(embeddings) else None
        observations.append(observation_from_dict(row, emb))
    return observations


def sort_observations(observations: list[Observation]) -> list[Observation]:
    """Order by timestamp, then id — preserves explicit frame ordering (§4)."""
    return sorted(observations, key=lambda o: (o.timestamp, o.id))
