"""Graph-constrained candidate filtering (§20 / Stage 17).

Physically implausible candidates are PENALIZED, never hard-rejected (§20:
"Do not hard-reject every non-neighbor candidate permanently"). Local
tracking searches previous place + neighbors + small graph radius; global
candidates are used when local confidence is poor — and a user may move
several places between observations, so distance is a soft penalty.
"""

from __future__ import annotations

import networkx as nx


def local_candidate_set(previous_place_id: int | None, graph: nx.Graph, radius: int = 2) -> set[int] | None:
    """Places within `radius` hops of the previous place (including itself).
    None when there is no previous place (-> global search)."""
    if previous_place_id is None:
        return None
    if previous_place_id not in graph:
        return None
    try:
        return set(nx.single_source_shortest_path_length(graph, previous_place_id, cutoff=radius))
    except nx.NetworkXError:
        return None


def graph_penalty(candidate_place_id: int, previous_place_id: int | None,
                  graph: nx.Graph, strength: float = 0.5) -> float:
    """Soft penalty in [0, 1]: 0 for the previous place itself, growing with
    shortest-path distance; full strength for unreachable places."""
    if previous_place_id is None or candidate_place_id == previous_place_id:
        return 0.0
    if candidate_place_id not in graph or previous_place_id not in graph:
        return strength
    try:
        distance = nx.shortest_path_length(graph, previous_place_id, candidate_place_id)
    except nx.NetworkXError:
        return strength
    return strength * (1.0 - float(__import__("math").exp(-distance / 2.0)))


def apply_graph_constraints(
    scored,
    previous_place_id: int | None,
    graph: nx.Graph,
    config: dict,
) -> list:
    """Attach a graph_term to each ScoredCandidate: 1.0 for the previous
    place, decaying by graph distance; keeps the total comparable."""
    strength = float(config.get("graph_penalty_strength", 0.5))
    for cand in scored:
        cand.graph_term = 1.0 - graph_penalty(cand.place_id, previous_place_id, graph, strength)
    return scored


def graph_terms_map(scored, previous_place_id: int | None, graph: nx.Graph,
                    config: dict) -> dict[int, float]:
    """Convenience: place_id -> graph term for score_candidates."""
    terms = {}
    strength = float(config.get("graph_penalty_strength", 0.5))
    for cand in scored:
        terms[cand.place_id] = 1.0 - graph_penalty(cand.place_id, previous_place_id, graph, strength)
    return terms
