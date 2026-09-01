"""Basic tests for the routing logic, using a small hand-built toy graph.

These don't touch the embedding model or FAISS at all — pure graph-logic
tests, fast and dependency-light.
"""

import networkx as nx

from src.navigate import format_directions, get_route, name_to_id_map


def make_toy_graph() -> nx.Graph:
    # 0 - 1 - 2 - 3   (a simple corridor)
    #     |
    #     4            (a branch off place 1)
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 4)])
    return g


TOY_NAMES = {"0": "Lobby", "1": "CorridorA", "2": "Room101", "3": "Room102", "4": "Stairwell"}


def test_shortest_path_simple_corridor():
    graph = make_toy_graph()
    route = get_route(graph, source_id=0, dest_id=3)
    assert route == [0, 1, 2, 3]


def test_shortest_path_via_branch():
    graph = make_toy_graph()
    route = get_route(graph, source_id=4, dest_id=3)
    assert route == [4, 1, 2, 3]


def test_no_path_returns_none():
    graph = make_toy_graph()
    graph.add_node(99)  # isolated, unreachable node
    route = get_route(graph, source_id=0, dest_id=99)
    assert route is None


def test_format_directions_produces_readable_string():
    route = [0, 1, 2, 3]
    text = format_directions(route, TOY_NAMES)
    assert text == "Lobby -> CorridorA -> Room101 -> Room102"


def test_name_to_id_map_round_trips():
    mapping = name_to_id_map(TOY_NAMES)
    assert mapping["Lobby"] == 0
    assert mapping["Stairwell"] == 4
