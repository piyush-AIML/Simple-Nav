"""Shared test fixtures: a toy MapBundle over a Lobby-Corridor-Room101 chain
with orthogonal exemplars (one per place, plus a second exemplar for the
corridor). Localization tests build on this."""

from collections import Counter

import networkx as nx
import numpy as np
import pytest

from src.localization.retrieval import Candidate
from src.mapping.map_artifact import MapBundle
from src.mapping.observation_store import ObservationStore
from src.mapping.observations import Observation
from src.mapping.place_builder import Place

# orthogonal unit vectors: place 0 = lobby, 1 = corridor, 2 = room101
V0 = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
V1 = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
V1B = np.array([0.0, 0.9, 0.44, 0.0], dtype="float32")  # second corridor exemplar
V2 = np.array([0.0, 0.0, 1.0, 0.0], dtype="float32")
WEAK = np.array([0.1, 0.1, 0.1, 0.97], dtype="float32")  # matches nothing well


def make_toy_bundle(tmp_path) -> MapBundle:
    store = ObservationStore(tmp_path / "obs")
    obs_list = [
        Observation(id="obs_0000", timestamp=0.0, frame_path="f0.jpg", embedding=V0),
        Observation(id="obs_0001", timestamp=1.0, frame_path="f1.jpg", embedding=V1),
        Observation(id="obs_0002", timestamp=2.0, frame_path="f2.jpg", embedding=V1B),
        Observation(id="obs_0003", timestamp=3.0, frame_path="f3.jpg", embedding=V2),
    ]
    store.add(obs_list)
    places = [
        Place(place_id=0, observation_ids=["obs_0000"], exemplar_ids=["obs_0000"],
              scene_types=Counter({"room": 1}), landmarks=[], visual_stats={}),
        Place(place_id=1, observation_ids=["obs_0001", "obs_0002"],
              exemplar_ids=["obs_0001", "obs_0002"],
              scene_types=Counter({"corridor": 2}), landmarks=[], visual_stats={}),
        Place(place_id=2, observation_ids=["obs_0003"], exemplar_ids=["obs_0003"],
              scene_types=Counter({"room": 1}), landmarks=[], visual_stats={}),
    ]
    graph = nx.Graph()
    graph.add_edge(0, 1, confidence=0.9, forward_count=5, reverse_count=2,
                   supporting_observations=20, mean_visual_strength=0.7)
    graph.add_edge(1, 2, confidence=0.9, forward_count=4, reverse_count=1,
                   supporting_observations=15, mean_visual_strength=0.7)
    return MapBundle(
        manifest={"map_id": "toy"}, places=places, graph=graph,
        exemplars=np.stack([V0, V1, V1B, V2]),
        exemplar_place_ids=np.array([0, 1, 1, 2], dtype="int64"),
        exemplar_ids=[o.id for o in obs_list],
        place_names={"0": "Lobby", "1": "Corridor", "2": "Room101"},
        store=store, map_dir=tmp_path,
    )


@pytest.fixture
def toy_bundle(tmp_path) -> MapBundle:
    return make_toy_bundle(tmp_path)


def make_scored(place_id: int, total: float) -> Candidate:
    return Candidate(place_id, total, f"e{place_id}", 1, 0.0)
