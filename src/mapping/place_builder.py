"""Place formation (§11): convert temporal segments into persistent physical
places. A place is NOT a cluster of images — it is a spatially coherent region
with multiple visual observations, a persistent identity, multi-exemplar
representation, and semantic statistics.

Exemplar selection methods (§11):
  - "kmeans":            secondary K-Means within the place (legacy approach)
  - "diversity":         greedy max-min over embeddings
  - "temporal_diversity" (default): spread exemplars across the temporal
                         extent, then max-min diversify remaining slots
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.mapping.observations import Observation
from src.mapping.segmentation import Segment
from src.perception.scene_tagger import normalize_scene_type
from src.utils import setup_logger

logger = setup_logger("place_builder")

# Landmark classes for MERGE evidence (scene tagger v2 closed vocabulary).
# v2 tokens carry no identity: "door" appears in 283/301 observations and
# "direction_sign" in 235 — wall fixtures shared by every corridor. Merge
# evidence must come from tokens that mark identity/function, plus the actual
# readable sign text (room numbers etc.) the VLM reports in sign_text.
GENERIC_LANDMARK_TYPES = frozenset(
    {"door", "direction_sign", "corridor_opening", "fire_equipment", "junction", "other"}
)
DISCRIMINATIVE_LANDMARK_TYPES = frozenset(
    {"room_sign", "stairs", "elevator", "entrance", "reception", "desk"}
)

# sign_text values that repeat the closed vocabulary or object names the VLM
# over-reports as signage ("Door", "Direction Sign", "unknown" on a wall) —
# not readable place identity. Everything else kept (e.g. "room 101").
SIGN_TEXT_JUNK = (
    GENERIC_LANDMARK_TYPES | DISCRIMINATIVE_LANDMARK_TYPES
    | {"direction sign", "fire equipment", "unknown", "window", "wall", "doorway"}
)


def aggregate_sign_texts(observations: list[Observation], limit: int = 8) -> list[str]:
    """Readable sign text seen in this place, most frequent first. Junk
    (generic vocabulary over-reports) filtered out — sign text is the place's
    identity evidence (§12; scene tagger v2 schema)."""
    counts: Counter[str] = Counter()
    for o in observations:
        for t in ((o.scene_tags or {}).get("sign_text") or []):
            token = str(t).strip().lower()
            if token and token not in SIGN_TEXT_JUNK:
                counts[token] += 1
    return [t for t, _ in counts.most_common(limit)]


@dataclass
class Place:
    place_id: str
    segment_ids: list[str] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)
    exemplar_ids: list[str] = field(default_factory=list)
    scene_types: Counter[str] = field(default_factory=Counter)
    landmarks: list[str] = field(default_factory=list)
    sign_texts: list[str] = field(default_factory=list)  # readable sign text, top-8 (identity evidence, §12)
    walkable_directions: Counter[str] = field(default_factory=Counter)  # per-direction vote counts (§15 junction evidence)
    object_classes: list[str] = field(default_factory=list)  # Stage 23: historical COCO classes
    visual_stats: dict = field(default_factory=dict)  # {"mean_similarity", "std_similarity"}

    def to_dict(self) -> dict:
        return {
            "place_id": self.place_id,
            "segment_ids": self.segment_ids,
            "observation_ids": self.observation_ids,
            "exemplar_ids": self.exemplar_ids,
            "scene_types": dict(self.scene_types),
            "landmarks": self.landmarks,
            "sign_texts": self.sign_texts,
            "walkable_directions": dict(self.walkable_directions),
            "object_classes": self.object_classes,
            "visual_stats": self.visual_stats,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Place":
        return cls(
            place_id=d["place_id"],
            segment_ids=list(d.get("segment_ids", [])),
            observation_ids=list(d.get("observation_ids", [])),
            exemplar_ids=list(d.get("exemplar_ids", [])),
            scene_types=Counter(d.get("scene_types", {})),
            landmarks=list(d.get("landmarks", [])),
            sign_texts=list(d.get("sign_texts", [])),
            walkable_directions=Counter(d.get("walkable_directions", {})),
            object_classes=list(d.get("object_classes", [])),
            visual_stats=dict(d.get("visual_stats", {})),
        )


def select_exemplars(
    observations: list[Observation],
    method: str = "temporal_diversity",
    max_exemplars: int = 3,
) -> list[str]:
    """Pick up to `max_exemplars` observation ids to represent this place."""
    if not observations:
        return []
    observations = sorted(observations, key=lambda o: o.timestamp)
    max_exemplars = min(max_exemplars, len(observations))

    if method == "kmeans":
        from sklearn.cluster import KMeans

        embeddings = np.stack([o.embedding for o in observations])
        k = min(max_exemplars, len(observations))
        if k <= 1:
            return [observations[len(observations) // 2].id]
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(embeddings)
        chosen = []
        for label in range(k):
            members = [o for o, l in zip(observations, labels) if l == label]
            if members:
                chosen.append(_nearest_to_mean(members).id)
        return chosen

    if method == "diversity":
        return [o.id for o in _greedy_maxmin(observations, max_exemplars)]

    if method == "temporal_diversity":
        chosen: list[Observation] = []
        if max_exemplars >= 1:
            chosen.append(observations[0])
        if max_exemplars >= 2:
            chosen.append(observations[-1])
        if max_exemplars >= 3:
            chosen.append(observations[len(observations) // 2])
        if len(chosen) < max_exemplars:
            candidates = [o for o in observations if o not in chosen]
            chosen.extend(_greedy_maxmin(candidates, max_exemplars - len(chosen)))
        return [o.id for o in chosen[:max_exemplars]]

    raise ValueError(f"Unknown exemplar method: {method!r}")


def _nearest_to_mean(members: list[Observation]) -> Observation:
    embeddings = np.stack([o.embedding for o in members])
    mean = embeddings.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    mean = mean / mean_norm if mean_norm > 0 else mean
    return members[int(np.argmax(embeddings @ mean))]


def _greedy_maxmin(observations: list[Observation], count: int) -> list[Observation]:
    """Greedy farthest-point sampling on cosine distance."""
    if count <= 0 or not observations:
        return []
    embeddings = np.stack([o.embedding for o in observations])
    chosen = [observations[0]]
    chosen_vecs = [embeddings[0]]
    while len(chosen) < count:
        sims = np.max(np.stack([embeddings @ v for v in chosen_vecs]), axis=0)
        idx = int(np.argmin(sims))
        chosen.append(observations[idx])
        chosen_vecs.append(embeddings[idx])
    return chosen


def _landmark_union(observations: list[Observation]) -> list[str]:
    counts: Counter[str] = Counter()
    for o in observations:
        counts.update(o.landmarks or [])
    return [lm for lm, _ in counts.most_common()]


def _object_union(observations: list[Observation]) -> list[str]:
    """Historical COCO class names observed in this place (Stage 23 evidence)."""
    counts: Counter[str] = Counter()
    for o in observations:
        for obj in (o.objects or []):
            cls = obj.get("class") if isinstance(obj, dict) else getattr(obj, "class_name", None)
            if cls:
                counts[str(cls)] += 1
    return [c for c, _ in counts.most_common()]


def _visual_stats(observations: list[Observation]) -> dict:
    embeddings = np.stack([o.embedding for o in observations])
    mean = embeddings.mean(axis=0)
    mean_norm = np.linalg.norm(mean)
    mean = mean / mean_norm if mean_norm > 0 else mean
    sims = embeddings @ mean
    return {
        "mean_similarity": round(float(sims.mean()), 4),
        "std_similarity": round(float(sims.std()), 4),
        "observation_count": len(observations),
    }


def build_places(
    store_observations: list[Observation],
    segments: list[Segment],
    exemplar_method: str = "temporal_diversity",
    max_exemplars: int = 3,
) -> list[Place]:
    """One Place per segment (reconciliation, Stage 10, merges them)."""
    by_id = {o.id: o for o in store_observations}
    places: list[Place] = []
    for i, segment in enumerate(segments):
        members = [by_id[oid] for oid in segment.obs_ids if oid in by_id]
        if not members:
            continue
        exemplars = select_exemplars(members, method=exemplar_method, max_exemplars=max_exemplars)
        places.append(
            Place(
                place_id=i,  # int ids keep the legacy PlaceIndex bridge seamless
                segment_ids=[segment.id],
                observation_ids=[o.id for o in members],
                exemplar_ids=exemplars,
                scene_types=Counter(
                    normalize_scene_type((o.scene_tags or {}).get("scene_type", "unknown"))
                    for o in members
                ),
                landmarks=_landmark_union(members),
                sign_texts=aggregate_sign_texts(members),
                walkable_directions=Counter(
                    d for o in members for d in ((o.scene_tags or {}).get("walkable") or [])
                ),
                object_classes=_object_union(members),
                visual_stats=_visual_stats(members),
            )
        )
    logger.info(f"Built {len(places)} places from {len(segments)} segments")
    return places
