"""Graph construction (§14) + junction detection (§15).

Node = place, edge = physical connection with measurable confidence.
Edges are created only from transitions that pass the support/confidence
gates (§14: frequency + persistence + semantic compatibility + repeat
observations). Directional evidence is stored per edge (A->B vs B->A counts);
the routing graph itself is undirected.

Junction detection (§15) is SOFT metadata only — node_type is stored on each
node but never used to hard-restrict routing.
"""

from __future__ import annotations

import json
from collections import Counter

import networkx as nx

from src.mapping.place_builder import Place
from src.mapping.transition_builder import TransitionStats
from src.perception.scene_tagger import normalize_scene_type
from src.utils import setup_logger

logger = setup_logger("graph_builder")


def build_graph(
    places: list[Place],
    transitions: list[TransitionStats],
    edge_confidence_threshold: float = 0.6,
    minimum_edge_support: int = 3,
) -> nx.Graph:
    """Node = place id; edges only from transitions passing the gates."""
    graph = nx.Graph()
    for place in places:
        graph.add_node(
            place.place_id,
            node_type="unknown",
            observation_count=len(place.observation_ids),
        )

    for stat in transitions:
        if stat.a not in graph or stat.b not in graph:
            logger.warning(f"Transition {stat.a}->{stat.b} references unknown place — skipped")
            continue
        if stat.total() < minimum_edge_support or stat.confidence < edge_confidence_threshold:
            logger.info(
                f"Skipped weak transition {stat.a}->{stat.b} "
                f"(total={stat.total()}, confidence={stat.confidence})"
            )
            continue
        if graph.has_edge(stat.a, stat.b):
            edge = graph[stat.a][stat.b]
            edge["confidence"] = max(edge["confidence"], stat.confidence)
            edge["forward_count"] += stat.forward_count
            edge["reverse_count"] += stat.reverse_count
            edge["supporting_observations"] += stat.supporting_observations
            edge["mean_visual_strength"] = max(edge["mean_visual_strength"], stat.visual_transition_strength)
        else:
            graph.add_edge(
                stat.a,
                stat.b,
                confidence=stat.confidence,
                forward_count=stat.forward_count,
                reverse_count=stat.reverse_count,
                supporting_observations=stat.supporting_observations,
                mean_visual_strength=stat.visual_transition_strength,
            )

    logger.info(
        f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
        f"({len(transitions)} candidate transitions)"
    )
    return graph


def _walkable_junction_evidence(walkable: Counter, n_obs: int) -> bool:
    """Branching evidence from VLM walkable votes (§15): at least two distinct
    walkable directions, each reported in at least half the place's
    observations. Majority support keeps one anomalous frame from voting."""
    if n_obs <= 0:
        return False
    required = max(1, (n_obs + 1) // 2)  # ceil(n_obs / 2)
    strong = [d for d, c in walkable.items() if c >= required]
    return len(strong) >= 2


def detect_junctions(
    graph: nx.Graph,
    places: list[Place],
    junction_min_degree: int = 3,
    junction_semantic_evidence: bool = True,
) -> None:
    """Set node_type per place (soft metadata — §15). Mutates node attrs.

    Geometry stays authoritative: degree >= junction_min_degree is always a
    junction. Below that gate, a VLM 'junction' scene with branching walkable
    evidence (two directions, majority support) is treated as a junction —
    evidence, never a hard topology claim. 'junction' without that evidence
    degrades to a corridor (v1 parity)."""
    place_by_id = {p.place_id: p for p in places}
    for node, attrs in graph.nodes(data=True):
        degree = graph.degree(node)
        place = place_by_id.get(node)
        scene_counter = place.scene_types if place is not None else Counter()
        dominant_scene = normalize_scene_type(
            scene_counter.most_common(1)[0][0] if scene_counter else "unknown"
        )

        if degree >= junction_min_degree:
            attrs["node_type"] = "junction"
        elif dominant_scene in ("stairs", "elevator", "entrance"):
            attrs["node_type"] = dominant_scene
        elif dominant_scene == "junction":
            evidence = (
                junction_semantic_evidence
                and place is not None
                and _walkable_junction_evidence(place.walkable_directions, len(place.observation_ids))
            )
            attrs["node_type"] = "junction" if evidence else "corridor"
        elif dominant_scene == "corridor":
            attrs["node_type"] = "corridor"
        elif dominant_scene in ("room", "lobby"):
            attrs["node_type"] = "room"
        else:
            attrs["node_type"] = "unknown"


def export_graph_json(graph: nx.Graph, path) -> None:
    """Portable graph export (consumed by the versioned map, Stage 14)."""
    data = {
        "nodes": [
            {"id": node, "node_type": attrs.get("node_type", "unknown"),
             "observation_count": attrs.get("observation_count", 0)}
            for node, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "confidence": round(float(attrs.get("confidence", 0.0)), 4),
                "forward_count": int(attrs.get("forward_count", 0)),
                "reverse_count": int(attrs.get("reverse_count", 0)),
                "supporting_observations": int(attrs.get("supporting_observations", 0)),
                "mean_visual_strength": round(float(attrs.get("mean_visual_strength", 0.0)), 4),
            }
            for u, v, attrs in graph.edges(data=True)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
