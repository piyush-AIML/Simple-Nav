"""Tests for candidate scoring (§19) and graph constraints (§20)."""

from collections import Counter

import networkx as nx

from src.localization.candidate_scoring import landmark_jaccard, score_candidates, semantic_term
from src.localization.graph_constraints import (
    apply_graph_constraints,
    graph_penalty,
    graph_terms_map,
    local_candidate_set,
)
from src.localization.retrieval import Candidate, RetrievalResult
from src.mapping.place_builder import Place

CFG = {"w_visual": 0.5, "w_semantic": 0.25, "w_temporal": 0.15, "w_graph": 0.1}


def make_result(*scores: float) -> RetrievalResult:
    candidates = [
        Candidate(place_id=i, visual_score=s, best_exemplar_id=f"e{i}",
                  supporting_exemplar_count=1, margin=0.0)
        for i, s in enumerate(scores)
    ]
    if candidates:
        for i, c in enumerate(candidates):
            c.margin = c.visual_score - (candidates[i + 1].visual_score if i + 1 < len(candidates) else 0.0)
    return RetrievalResult(candidates, scores[0] if scores else 0.0,
                           scores[1] if len(scores) > 1 else 0.0,
                           (scores[0] - scores[1]) if len(scores) > 1 else 0.0)


def place(pid: int, scene: str = "room", landmarks=None) -> Place:
    return Place(place_id=pid, observation_ids=[], exemplar_ids=[],
                 scene_types=Counter({scene: 1}), landmarks=landmarks or [], visual_stats={})


def test_semantic_wins_over_visual_ambiguity():
    """Two visually equal candidates; one matches the query scene+landmarks."""
    places = {0: place(0, "corridor", ["blue sign"]), 1: place(1, "room", ["desk"])}
    result = make_result(0.8, 0.8)
    tags = {"scene_type": "corridor", "landmarks": ["blue sign"]}
    scored = score_candidates(result, tags, None, None, None, places, CFG)
    assert scored[0].place_id == 0
    assert scored[0].semantic_term > scored[1].semantic_term


def test_missing_query_tags_neutral():
    places = {0: place(0, "corridor"), 1: place(1, "room")}
    scored = score_candidates(make_result(0.8, 0.7), None, None, None, None, places, CFG)
    for c in scored:
        assert c.semantic_term == 0.5  # neutral, no crash


def test_temporal_term_prefers_previous_place():
    places = {0: place(0), 1: place(1)}
    scored = score_candidates(make_result(0.9, 0.8), None, None, 1, None, places, CFG)
    # place 1 is visually second but temporally current
    assert scored[0].place_id == 1
    assert scored[0].temporal_term == 1.0
    assert scored[1].temporal_term == 0.5


def test_graph_term_applied():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    scored = score_candidates(
        make_result(0.9, 0.8, 0.7), None, None, 0,
        graph_terms_map([Candidate(0, 0.9, "e0", 1, 0), Candidate(1, 0.8, "e1", 1, 0),
                         Candidate(2, 0.7, "e2", 1, 0)], 0, graph, CFG),
        {0: place(0), 1: place(1), 2: place(2)}, CFG,
    )
    by_id = {c.place_id: c for c in scored}
    assert by_id[0].graph_term == 1.0                 # previous place itself
    assert by_id[1].graph_term > by_id[2].graph_term  # neighbor beats 2 hops


def test_landmark_jaccard():
    assert landmark_jaccard(["a", "b"], ["a", "b", "c"]) == 2 / 3
    assert landmark_jaccard([], []) == 0.0
    assert landmark_jaccard(["a"], []) == 0.0


def test_semantic_term_unknown_query_is_neutral():
    assert semantic_term(None, [], "room", []) == 0.5
    assert semantic_term("unknown", [], "room", []) == 0.5  # half weight scene match


def test_local_candidate_set_radius():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 4)])
    assert local_candidate_set(1, graph, radius=1) == {0, 1, 2, 4}
    assert local_candidate_set(1, graph, radius=2) == {0, 1, 2, 3, 4}
    assert local_candidate_set(None, graph) is None
    assert local_candidate_set(99, graph) is None  # unknown previous place


def test_graph_penalty_soft_not_hard():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    assert graph_penalty(0, 0, graph) == 0.0           # the previous place itself
    p_near = graph_penalty(1, 0, graph)                # neighbor
    p_far = graph_penalty(3, 0, graph)
    assert 0.0 < p_near < p_far < 0.5                  # penalized but not excluded
    assert graph_penalty(99, 0, graph) == 0.5          # unreachable: full strength
    assert graph_penalty(0, None, graph) == 0.0        # no previous -> no penalty


def test_apply_graph_constraints_sets_terms():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1)])
    scored = [Candidate(0, 0.9, "e0", 1, 0), Candidate(1, 0.8, "e1", 1, 0)]
    out = apply_graph_constraints(scored, 0, graph, CFG)
    assert out[0].graph_term == 1.0
    assert out[1].graph_term == 1.0 - graph_penalty(1, 0, graph)
