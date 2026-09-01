"""Tests for the versioned map artifact (§17): write -> reload without
build-time objects, manifest fields, and the legacy PlaceIndex bridge."""

import json
from collections import Counter

import networkx as nx
import numpy as np

from src.mapping.map_artifact import MapBundle, write_map
from src.mapping.observation_store import ObservationStore
from src.mapping.observations import Observation
from src.mapping.place_builder import Place


def make_store(tmp_path) -> ObservationStore:
    store = ObservationStore(tmp_path / "obs")
    obs_list = [
        Observation(id="obs_0000", timestamp=0.0, frame_path="data/frames/a.jpg",
                    embedding=np.array([1.0, 0.0, 0.0], dtype="float32")),
        Observation(id="obs_0001", timestamp=1.0, frame_path="data/frames/b.jpg",
                    embedding=np.array([0.0, 1.0, 0.0], dtype="float32")),
        Observation(id="obs_0002", timestamp=2.0, frame_path="data/frames/c.jpg",
                    embedding=np.array([0.0, 0.0, 1.0], dtype="float32")),
    ]
    store.add(obs_list)
    return store


def make_places() -> list[Place]:
    return [
        Place(place_id=0, segment_ids=["seg_000"],
              observation_ids=["obs_0000", "obs_0001"], exemplar_ids=["obs_0000", "obs_0001"],
              scene_types=Counter({"corridor": 2}), landmarks=["blue sign"],
              visual_stats={"mean_similarity": 0.9, "std_similarity": 0.05}),
        Place(place_id=1, segment_ids=["seg_001"],
              observation_ids=["obs_0002"], exemplar_ids=["obs_0002"],
              scene_types=Counter({"room": 1}), landmarks=[], visual_stats={}),
    ]


def make_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node(0, node_type="corridor", observation_count=2)
    g.add_node(1, node_type="room", observation_count=1)
    g.add_edge(0, 1, confidence=0.9, forward_count=3, reverse_count=1,
               supporting_observations=5, mean_visual_strength=0.7)
    return g


def test_write_then_load_round_trip(tmp_path):
    store = make_store(tmp_path)
    target = write_map(
        tmp_path / "maps",
        map_id="testmap",
        building_id="b", floor_id="f",
        encoder_name="resnet18", encoder_version="v1", embedding_dimension=3,
        store=store, places=make_places(), graph=make_graph(),
    )
    assert target.name == "testmap"

    # reload from disk in a fresh bundle (no build-time objects)
    bundle = MapBundle.load(target)
    assert bundle.manifest["map_id"] == "testmap"
    assert bundle.manifest["place_count"] == 2
    assert bundle.manifest["edge_count"] == 1
    assert bundle.manifest["embedding_dimension"] == 3
    assert len(bundle.places) == 2
    assert bundle.places[0].scene_types["corridor"] == 2  # Counter round-trips
    assert set(bundle.graph.edges()) == {(0, 1)}
    assert bundle.graph[0][1]["forward_count"] == 3
    assert bundle.exemplars.shape == (3, 3)
    assert len(bundle.store) == 3


def test_manifest_hash_and_fields(tmp_path):
    store = make_store(tmp_path)
    target = write_map(
        tmp_path / "maps", map_id="m2", building_id="college", floor_id="floor_1",
        encoder_name="resnet18", encoder_version="v1", embedding_dimension=3,
        store=store, places=make_places(), graph=make_graph(),
    )
    with open(target / "manifest.json") as f:
        manifest = json.load(f)
    for key in ("map_id", "building_id", "floor_id", "encoder", "encoder_version",
                "embedding_dimension", "observation_count", "place_count",
                "edge_count", "created_at", "hash"):
        assert key in manifest, key
    assert manifest["observation_count"] == 3


def test_to_place_index_bridge(tmp_path):
    store = make_store(tmp_path)
    target = write_map(
        tmp_path / "maps", map_id="m3", building_id="b", floor_id="f",
        encoder_name="resnet18", encoder_version="v1", embedding_dimension=3,
        store=store, places=make_places(), graph=make_graph(),
    )
    bundle = MapBundle.load(target)
    index = bundle.to_place_index()
    # query with an exemplar-like vector resolves to the right place
    top = index.query(np.array([1.0, 0.0, 0.0], dtype="float32"), top_k=1)
    assert top[0][0] == 0  # place_00 is the first exemplar place in the index
