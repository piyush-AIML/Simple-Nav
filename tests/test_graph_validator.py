"""Tests for graph validation (§16): every check produces the right warning
and a clean graph produces none."""

from collections import Counter

import networkx as nx
import numpy as np

from src.mapping.graph_builder import build_graph, detect_junctions
from src.mapping.graph_validator import validate_graph, write_validation_report
from src.mapping.observations import Observation
from src.mapping.place_builder import Place
from src.mapping.transition_builder import TransitionStats

CFG = {
    "weak_edge_threshold": 0.4,
    "min_observations_per_place": 3,
    "high_variance_threshold": 0.35,
    "duplicate_similarity_threshold": 0.9,
}


def make_place(pid: str, scene: str = "room", obs_count: int = 5,
               std_sim: float = 0.1, landmarks=None) -> Place:
    return Place(
        place_id=pid,
        observation_ids=[f"obs_{pid}_{i}" for i in range(obs_count)],
        exemplar_ids=[f"obs_{pid}_0"],
        scene_types=Counter({scene: obs_count}),
        landmarks=landmarks or [],
        visual_stats={"mean_similarity": 0.8, "std_similarity": std_sim, "observation_count": obs_count},
    )


def make_transition(a: str, b: str, total: int, conf: float) -> TransitionStats:
    return TransitionStats(a=a, b=b, forward_count=total, reverse_count=0,
                           supporting_observations=10, confidence=conf)


def obs_for(place: Place, embedding: np.ndarray) -> Observation:
    return Observation(
        id=place.exemplar_ids[0], timestamp=0.0, frame_path="x.jpg", embedding=embedding
    )


def test_clean_graph_no_warnings():
    places = [make_place("a"), make_place("b")]
    graph = build_graph(places, [make_transition("a", "b", 5, 0.9)], minimum_edge_support=1)
    assert validate_graph(graph, places, CFG) == []


def test_isolated_node_warns():
    places = [make_place("a"), make_place("b"), make_place("lonely")]
    graph = build_graph(places, [make_transition("a", "b", 5, 0.9)], minimum_edge_support=1)
    warnings = validate_graph(graph, places, CFG)
    isolated = [w for w in warnings if w.check == "isolated_node"]
    assert len(isolated) == 1
    assert "lonely" in isolated[0].message


def test_disconnected_component_warns():
    places = [make_place("a"), make_place("b"), make_place("c"), make_place("d")]
    graph = build_graph(places, [
        make_transition("a", "b", 5, 0.9),
        make_transition("c", "d", 5, 0.9),
    ], minimum_edge_support=1)
    warnings = [w for w in validate_graph(graph, places, CFG) if w.check == "disconnected_component"]
    assert len(warnings) == 1


def test_weak_edge_warns():
    """An edge that passes a LOW construction gate but sits under the
    validation threshold is flagged (construction gate 0.6 default would have
    filtered it entirely — validation catches the 0.4-0.6 zone)."""
    places = [make_place("a"), make_place("b")]
    graph = build_graph(
        places, [make_transition("a", "b", 1, 0.3)],
        edge_confidence_threshold=0.2, minimum_edge_support=1,
    )
    warnings = [w for w in validate_graph(graph, places, CFG) if w.check == "weak_edge"]
    assert len(warnings) == 1


def test_low_observation_count_warns():
    places = [make_place("tiny", obs_count=1), make_place("b")]
    graph = build_graph(places, [make_transition("tiny", "b", 5, 0.9)], minimum_edge_support=1)
    warnings = [w for w in validate_graph(graph, places, CFG) if w.check == "suspicious_node"]
    assert any("only 1 observations" in w.message for w in warnings)


def test_high_variance_warns():
    places = [make_place("noisy", std_sim=0.6), make_place("b")]
    graph = build_graph(places, [make_transition("noisy", "b", 5, 0.9)], minimum_edge_support=1)
    warnings = [w for w in validate_graph(graph, places, CFG) if w.check == "suspicious_node"]
    assert any("variance" in w.message for w in warnings)


def test_duplicate_candidates_warn():
    a = make_place("a")
    b = make_place("b")
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    observations = [obs_for(a, emb), obs_for(b, emb.copy())]  # identical exemplars
    places = [a, b]
    graph = build_graph(places, [make_transition("a", "b", 5, 0.9)], minimum_edge_support=1)
    warnings = [w for w in validate_graph(graph, places, CFG, observations=observations)
                if w.check == "duplicate_candidate"]
    assert len(warnings) == 1
    assert "may be duplicates" in warnings[0].message


def test_report_written(tmp_path):
    warnings = validate_graph(
        build_graph([make_place("a")], [], minimum_edge_support=1),
        [make_place("a")], CFG,
    )
    md = tmp_path / "graph_validation.md"
    js = tmp_path / "graph_validation.json"
    write_validation_report(warnings, md, js)
    assert md.exists() and js.exists()
    assert "isolated" in md.read_text()
