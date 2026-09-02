"""Tests for graph construction (§14) and junction detection (§15)."""

from collections import Counter

from src.mapping.graph_builder import build_graph, detect_junctions, export_graph_json
from src.mapping.place_builder import Place
from src.mapping.transition_builder import TransitionStats


def place(pid: str, scene: str = "room", obs_count: int = 5,
          walkable: Counter | None = None) -> Place:
    return Place(
        place_id=pid,
        observation_ids=[f"obs_{pid}_{i}" for i in range(obs_count)],
        scene_types=Counter({scene: obs_count}),
        walkable_directions=walkable or Counter(),
    )


def transition(a: str, b: str, total: int, conf: float = 0.9) -> TransitionStats:
    fwd, rev = (total, 0) if total > 0 else (0, -total)
    return TransitionStats(
        a=a, b=b, forward_count=fwd, reverse_count=rev,
        supporting_observations=10, confidence=conf,
    )


def test_corridor_triangle_recovered():
    places = [place("lobby", "room"), place("corridor", "corridor"), place("room101", "room")]
    transitions = [
        transition("lobby", "corridor", 4),
        transition("corridor", "room101", 3),
    ]
    graph = build_graph(places, transitions, minimum_edge_support=3)
    assert set(graph.nodes) == {"lobby", "corridor", "room101"}
    assert set(graph.edges()) == {("lobby", "corridor"), ("corridor", "room101")}
    assert graph["lobby"]["corridor"]["forward_count"] == 4
    assert graph["lobby"]["corridor"]["confidence"] == 0.9


def test_weak_transitions_do_not_create_edges():
    places = [place("a"), place("b"), place("c")]
    weak = transition("a", "b", 1, conf=0.4)          # below both gates
    strong = transition("b", "c", 5, conf=0.9)
    graph = build_graph(places, [weak, strong], edge_confidence_threshold=0.6, minimum_edge_support=3)
    assert ("a", "b") not in graph.edges()
    assert ("b", "c") in graph.edges()


def test_unknown_place_transitions_skipped():
    places = [place("a"), place("b")]
    stray = transition("a", "ghost", 5)
    graph = build_graph(places, [stray], minimum_edge_support=1)
    assert "ghost" not in graph.nodes
    assert graph.number_of_edges() == 0


def test_directional_counts_stored_separately():
    """§14: A->B and B->A stored separately; routing graph stays undirected."""
    places = [place("a"), place("b")]
    ab = TransitionStats(a="a", b="b", forward_count=5, reverse_count=2,
                         supporting_observations=20, confidence=0.9)
    graph = build_graph(places, [ab], minimum_edge_support=1)
    assert graph.has_edge("a", "b")
    assert graph["a"]["b"]["forward_count"] == 5
    assert graph["a"]["b"]["reverse_count"] == 2


def test_junction_detection_by_degree_and_scene():
    places = [
        place("junc", "corridor"),
        place("n1", "room"), place("n2", "room"), place("n3", "room"),
        place("stairs", "stairs"),
    ]
    transitions = [
        transition("junc", "n1", 5), transition("junc", "n2", 5),
        transition("junc", "n3", 5), transition("n1", "stairs", 5),
    ]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places, junction_min_degree=3)
    assert graph.nodes["junc"]["node_type"] == "junction"
    assert graph.nodes["stairs"]["node_type"] == "stairs"
    assert graph.nodes["n2"]["node_type"] == "room"
    assert graph.nodes["n3"]["node_type"] == "room"


def test_junction_scene_without_walkable_evidence_is_corridor():
    """A 'junction' scene alone never forces node_type — v1 parity: without
    branching walkable votes it degrades to corridor (soft metadata, §15)."""
    places = [
        place("twoway", "junction", obs_count=6),   # no walkable votes
        place("n1", "room"), place("n2", "room"),
    ]
    transitions = [transition("twoway", "n1", 5), transition("twoway", "n2", 5)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places)
    assert graph.nodes["twoway"]["node_type"] == "corridor"


def test_junction_scene_with_walkable_evidence_is_junction():
    """Branching walkable votes (two directions in a majority of the place's
    observations) corroborate the VLM 'junction' scene below the degree gate."""
    walkable = Counter({"forward": 6, "left": 6})  # obs_count=6: both >= ceil(6/2)
    places = [
        place("split", "junction", obs_count=6, walkable=walkable),
        place("n1", "room"), place("n2", "room"),
    ]
    transitions = [transition("split", "n1", 5), transition("split", "n2", 5)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places)
    assert graph.nodes["split"]["node_type"] == "junction"


def test_junction_evidence_requires_majority_support():
    """One isolated frame must not vote a place into junction — each direction
    needs support in at least half the observations."""
    walkable = Counter({"forward": 5, "left": 1})  # 6 obs: left seen once only
    places = [
        place("twoway", "junction", obs_count=6, walkable=walkable),
        place("n1", "room"), place("n2", "room"),
    ]
    transitions = [transition("twoway", "n1", 5), transition("twoway", "n2", 5)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places)
    assert graph.nodes["twoway"]["node_type"] == "corridor"


def test_junction_evidence_can_be_disabled():
    walkable = Counter({"forward": 6, "left": 6})
    places = [
        place("split", "junction", obs_count=6, walkable=walkable),
        place("n1", "room"), place("n2", "room"),
    ]
    transitions = [transition("split", "n1", 5), transition("split", "n2", 5)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places, junction_semantic_evidence=False)
    assert graph.nodes["split"]["node_type"] == "corridor"


def test_lobby_scene_maps_to_room():
    places = [
        place("lobby", "lobby"),
        place("n1", "room"),
    ]
    transitions = [transition("lobby", "n1", 5)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places)
    assert graph.nodes["lobby"]["node_type"] == "room"


def test_legacy_corridor_junction_scene_maps_to_corridor():
    """Stored v1 scene counts normalize on read: corridor_junction is treated
    exactly like junction-without-walkable (v1 node_type parity)."""
    from collections import Counter as C

    p = Place(
        place_id="oldjunc",
        observation_ids=[f"obs_{i}" for i in range(5)],
        scene_types=C({"corridor_junction": 5}),
    )
    places = [p, place("n1", "room"), place("n2", "room")]
    transitions = [transition("oldjunc", "n1", 5), transition("oldjunc", "n2", 5)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places)
    assert graph.nodes["oldjunc"]["node_type"] == "corridor"


def test_export_graph_json_round_trip():
    import json

    places = [place("a"), place("b")]
    transitions = [transition("a", "b", 4)]
    graph = build_graph(places, transitions, minimum_edge_support=1)
    detect_junctions(graph, places)
    path = "data/map/_test_graph.json"
    export_graph_json(graph, path)
    with open(path) as f:
        data = json.load(f)
    assert {n["id"] for n in data["nodes"]} == {"a", "b"}
    assert data["edges"][0]["source"] == "a"
    assert data["edges"][0]["confidence"] == 0.9
