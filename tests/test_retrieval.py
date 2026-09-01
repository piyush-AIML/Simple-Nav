"""Tests for candidate retrieval (§18): top-K aggregation, margins,
supporting exemplar counts. Uses a small synthetic MapBundle."""

from collections import Counter

import networkx as nx
import numpy as np

from src.localization.retrieval import CandidateRetriever, RetrievalResult
from src.mapping.map_artifact import MapBundle
from src.mapping.observation_store import ObservationStore
from src.mapping.observations import Observation
from src.mapping.place_builder import Place


def make_bundle(tmp_path) -> MapBundle:
    """3 places; place 1 has 2 exemplars, the rest 1. Orthogonal vectors."""
    store = ObservationStore(tmp_path / "obs")
    obs_list = [
        Observation(id=f"obs_{i:04d}", timestamp=float(i), frame_path=f"f{i}.jpg",
                    embedding=v) for i, v in enumerate([
                        np.array([1, 0, 0, 0], dtype="float32"),   # place 0
                        np.array([0, 1, 0, 0], dtype="float32"),   # place 1
                        np.array([0, 0, 1, 0], dtype="float32"),   # place 1
                        np.array([0, 0, 0, 1], dtype="float32"),   # place 2
                    ])
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
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    bundle = MapBundle(
        manifest={"map_id": "toy"}, places=places, graph=graph,
        exemplars=np.stack([o.embedding for o in obs_list]),
        exemplar_place_ids=np.array([0, 1, 1, 2], dtype="int64"),
        exemplar_ids=[o.id for o in obs_list],
        place_names={"0": "P0", "1": "P1", "2": "P2"},
        store=store, map_dir=tmp_path,
    )
    return bundle


def test_retrieve_ranks_and_aggregates(tmp_path):
    retriever = CandidateRetriever(make_bundle(tmp_path), top_k=3, search_factor=4)
    result = retriever.retrieve(np.array([0.9, 0.3, 0.1, 0.0], dtype="float32"))
    assert result.candidates[0].place_id == 0
    assert result.best_score > result.second_best_score
    assert result.score_margin == pytest_approx(result.best_score - result.second_best_score)


def test_supporting_exemplar_count(tmp_path):
    retriever = CandidateRetriever(make_bundle(tmp_path), top_k=4, search_factor=4)
    result = retriever.retrieve(np.array([0.1, 0.9, 0.8, 0.0], dtype="float32"))
    place1 = [c for c in result.candidates if c.place_id == 1][0]
    assert place1.supporting_exemplar_count == 2
    assert place1.best_exemplar_id == "obs_0001"


def test_margin_computed_between_places(tmp_path):
    retriever = CandidateRetriever(make_bundle(tmp_path), top_k=2)
    result = retriever.retrieve(np.array([0.6, 0.5, 0.1, 0.0], dtype="float32"))
    assert len(result.candidates) == 2
    assert result.candidates[0].margin == pytest_approx(result.candidates[0].visual_score - result.candidates[1].visual_score)
    # last candidate's margin is its score - 0
    assert result.candidates[1].margin == pytest_approx(result.candidates[1].visual_score)


def test_empty_index_returns_empty(tmp_path):
    store = ObservationStore(tmp_path / "obs2")
    empty_bundle = MapBundle(
        manifest={}, places=[], graph=nx.Graph(),
        exemplars=np.zeros((0, 4), dtype="float32"),
        exemplar_place_ids=np.array([], dtype="int64"),
        exemplar_ids=[], place_names={}, store=store, map_dir=tmp_path,
    )
    retriever = CandidateRetriever(empty_bundle, top_k=3)
    result = retriever.retrieve(np.zeros(4, dtype="float32"))
    assert result.candidates == []
    assert result.best_score == 0.0


def pytest_approx(x):
    import pytest
    return pytest.approx(x, rel=1e-6)
