"""Versioned map artifact (§17): a self-contained, reloadable map directory,
independent of build-time Python objects. Another process can load it from
disk without the exact classes that wrote it.

Layout (data/map/<map_id>/):
    manifest.json              map metadata + counts + content hash
    places.json                place records (id, name, obs ids, exemplars, ...)
    graph.json                 portable graph export (nodes with node_type, edges)
    exemplars.npy              (M, D) exemplar vectors
    exemplar_place_ids.npy     (M,) place id per exemplar row
    observation_metadata.json  observation records referenced by the map
    vector_index/              index.faiss + id_order.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from src.mapping.graph_builder import export_graph_json
from src.mapping.observation_store import ObservationStore
from src.mapping.place_builder import Place
from src.utils import setup_logger

logger = setup_logger("map_artifact")

MANIFEST_KEYS = (
    "map_id", "building_id", "floor_id", "encoder", "encoder_version",
    "embedding_dimension", "observation_count", "place_count", "edge_count",
    "created_at", "hash",
)


@dataclass
class MapBundle:
    manifest: dict
    places: list[Place]
    graph: nx.Graph
    exemplars: np.ndarray
    exemplar_place_ids: np.ndarray
    exemplar_ids: list[str]          # row -> observation id
    place_names: dict[str, str]
    store: ObservationStore
    map_dir: Path

    @classmethod
    def load(cls, map_dir: Path) -> "MapBundle":
        map_dir = Path(map_dir)
        with open(map_dir / "manifest.json") as f:
            manifest = json.load(f)
        with open(map_dir / "places.json") as f:
            places = [Place.from_dict(p) for p in json.load(f)]
        with open(map_dir / "graph.json") as f:
            graph_data = json.load(f)
        graph = nx.Graph()
        for node in graph_data["nodes"]:
            graph.add_node(node["id"], node_type=node.get("node_type", "unknown"),
                           observation_count=node.get("observation_count", 0))
        for edge in graph_data["edges"]:
            graph.add_edge(edge["source"], edge["target"],
                           confidence=edge.get("confidence", 0.0),
                           forward_count=edge.get("forward_count", 0),
                           reverse_count=edge.get("reverse_count", 0),
                           supporting_observations=edge.get("supporting_observations", 0),
                           mean_visual_strength=edge.get("mean_visual_strength", 0.0))
        exemplars = np.load(map_dir / "exemplars.npy")
        exemplar_place_ids = np.load(map_dir / "exemplar_place_ids.npy")
        with open(map_dir / "exemplar_ids.json") as f:
            exemplar_ids = json.load(f)
        # legacy-style names: str key -> display name; overridable via the
        # place_names.json shipped in the map (or edited by hand later)
        place_names = {str(p.place_id): f"Place_{p.place_id}" for p in places}
        store = ObservationStore.load(map_dir / "vector_index")
        bundle = cls(manifest, places, graph, exemplars, exemplar_place_ids,
                     exemplar_ids, place_names, store, map_dir)
        # allow callers to override names afterwards via bundle.place_names
        bundle._names_file = map_dir / "place_names.json"
        if bundle._names_file.exists():
            with open(bundle._names_file) as f:
                bundle.place_names.update(json.load(f))
        logger.info(f"Loaded map {manifest.get('map_id')} "
                    f"({len(places)} places, {graph.number_of_edges()} edges)")
        return bundle

    def to_place_index(self):
        """Bridge to the legacy retrieval API (baseline consumers keep
        working unchanged)."""
        from src.localize import PlaceIndex

        return PlaceIndex(self.exemplars, self.exemplar_place_ids, self.place_names)


def _content_hash(places: list[Place], graph: nx.Graph) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([p.to_dict() for p in places], sort_keys=True).encode())
    for u, v in sorted(graph.edges()):
        h.update(f"{u}{v}".encode())
    return h.hexdigest()[:16]


def write_map(
    map_dir: Path,
    *,
    map_id: str,
    building_id: str,
    floor_id: str,
    encoder_name: str,
    encoder_version: str,
    embedding_dimension: int,
    store: ObservationStore,
    places: list[Place],
    graph: nx.Graph,
    place_names: dict[str, str] | None = None,
) -> Path:
    """Write the full map artifact; returns the map directory path."""
    target = Path(map_dir) / map_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    # places.json (place records)
    with open(target / "places.json", "w") as f:
        json.dump([p.to_dict() for p in places], f, indent=2)

    # graph.json
    export_graph_json(graph, target / "graph.json")

    # exemplars (rows in place order — exemplar_ids.json records the row->obs
    # id mapping so loaders never have to guess the order)
    exemplars: list[np.ndarray] = []
    exemplar_place_ids: list[str] = []
    exemplar_ids: list[str] = []
    by_id = {o.id: o for o in store.all()}
    for place in places:
        for eid in place.exemplar_ids:
            obs = by_id.get(eid)
            if obs is not None and obs.embedding is not None:
                exemplars.append(obs.embedding)
                exemplar_place_ids.append(place.place_id)
                exemplar_ids.append(eid)
    if exemplars:
        np.save(target / "exemplars.npy", np.stack(exemplars).astype("float32"))
        np.save(target / "exemplar_place_ids.npy", np.array(exemplar_place_ids, dtype="int64"))
    else:
        np.save(target / "exemplars.npy", np.zeros((0, embedding_dimension), dtype="float32"))
        np.save(target / "exemplar_place_ids.npy", np.array([], dtype="int64"))
    with open(target / "exemplar_ids.json", "w") as f:
        json.dump(exemplar_ids, f, indent=2)

    # observation metadata (only the ids referenced by places)
    referenced = {oid for p in places for oid in p.observation_ids}
    meta = [o.to_dict() for o in store.all() if o.id in referenced]
    with open(target / "observation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # names (default = Place_<id>; rename by hand or via label tooling)
    default_names = {str(p.place_id): f"Place_{p.place_id}" for p in places}
    with open(target / "place_names.json", "w") as f:
        json.dump(place_names or default_names, f, indent=2)

    # manifest
    manifest = {
        "map_id": map_id,
        "building_id": building_id,
        "floor_id": floor_id,
        "encoder": encoder_name,
        "encoder_version": encoder_version,
        "embedding_dimension": embedding_dimension,
        "observation_count": len(referenced),
        "place_count": len(places),
        "edge_count": int(graph.number_of_edges()),
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "hash": _content_hash(places, graph),
    }
    with open(target / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # vector index (copy of the store index + ids, plus store metadata)
    vec_dir = target / "vector_index"
    vec_dir.mkdir()
    store.save()
    shutil.copy(store.obs_dir / "index.faiss", vec_dir / "index.faiss")
    shutil.copy(store.obs_dir / "id_order.json", vec_dir / "id_order.json")
    shutil.copy(store.obs_dir / "observations.jsonl", vec_dir / "observations.jsonl")
    shutil.copy(store.obs_dir / "embeddings.npy", vec_dir / "embeddings.npy")
    shutil.copy(store.obs_dir / "encoder.json", vec_dir / "encoder.json")

    logger.info(f"Wrote versioned map -> {target} "
                f"({len(places)} places, {manifest['edge_count']} edges, "
                f"{manifest['observation_count']} observations)")
    return target
