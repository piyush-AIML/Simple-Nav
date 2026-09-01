"""Tests for the Bayes-filter state estimator (§21): concentration, unknown
mass, graph priors, and reset behavior."""

import networkx as nx
import numpy as np
import pytest

from src.localization.retrieval import Candidate
from src.localization.state_estimator import (
    StateEstimate,
    StateEstimator,
    softmax_likelihood,
    transition_prior,
)

CFG = {"self_transition_prior": 0.7, "far_transition_prior": 0.05, "likelihood_temperature": 0.1}


def make_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge(0, 1, confidence=0.9)
    g.add_edge(1, 2, confidence=0.9)
    return g


def scored_for(*pairs) -> list:
    """pairs: (place_id, total)."""
    return [Candidate(pid, score, f"e{pid}", 1, 0.0) for pid, score in pairs]


def test_strong_evidence_concentrates():
    """Belief builds over frames — a few strong observations at place 1
    concentrate the posterior there."""
    est = StateEstimator(make_graph(), [], CFG)
    for _ in range(3):
        est.update(scored_for((1, 0.9), (0, 0.5), (2, 0.5)))
    estimate = est.update(scored_for((1, 0.9), (0, 0.5), (2, 0.5)))
    assert estimate.best_place_id == 1
    assert estimate.best_score > 0.5
    assert estimate.unknown_mass < 0.35
    assert estimate.entropy < 0.5


def test_weak_evidence_raises_unknown_mass():
    est = StateEstimator(make_graph(), [], CFG)
    estimate = est.update(scored_for((1, 0.11), (0, 0.10), (2, 0.09)))
    assert estimate.unknown_mass > 0.5


def test_graph_prior_resolves_tie():
    """After settling at place 1, an ambiguous frame keeps 1 ahead of its
    symmetric neighbors (self-transition mass)."""
    est = StateEstimator(make_graph(), [], CFG)
    for _ in range(3):
        est.update(scored_for((1, 0.8), (0, 0.1), (2, 0.1)))  # settle at 1
    estimate = est.update(scored_for((1, 0.5), (2, 0.5), (0, 0.5)))
    assert estimate.best_place_id == 1


def test_temporal_carry_moves_along_edges():
    est = StateEstimator(make_graph(), [], CFG)
    for _ in range(2):
        est.update(scored_for((0, 0.9), (1, 0.4), (2, 0.1)))  # at 0
    for _ in range(2):
        est.update(scored_for((1, 0.9), (0, 0.3), (2, 0.3)))  # moved to 1
    assert est.update(scored_for((1, 0.8), (2, 0.4), (0, 0.2))).best_place_id == 1


def test_reset_returns_uniform():
    est = StateEstimator(make_graph(), [], CFG)
    for _ in range(3):
        est.update(scored_for((0, 0.9), (1, 0.1), (2, 0.1)))  # settle at 0
    est.reset()
    estimate = est.update(scored_for((1, 0.6), (2, 0.6), (0, 0.6)))
    # after reset the prior is uniform and equal likelihoods leave only the
    # topology prior: the degree-2 junction (place 1) carries the most mass
    assert estimate.posterior[1] > estimate.posterior[0]
    assert estimate.posterior[0] == pytest.approx(estimate.posterior[2], abs=1e-9)
    assert estimate.unknown_mass > 0.5  # flat evidence stays uncertain


def test_empty_scored_returns_unknown():
    est = StateEstimator(make_graph(), [], CFG)
    estimate = est.update([])
    assert estimate.best_place_id is None
    assert estimate.unknown_mass == 1.0


def test_softmax_likelihood_properties():
    lik = softmax_likelihood(scored_for((0, 0.9), (1, 0.5)), temperature=0.1)
    total = sum(lik.values())
    assert abs(total - 1.0) < 1e-6
    assert lik[0] > lik[1]


def test_transition_prior_structure():
    priors = transition_prior(make_graph(), [])
    row = priors[1]
    assert abs(row[1] - 0.7) < 1e-9          # self-transition mass
    assert abs(sum(row.values()) - 1.0) < 1e-9
    assert row[0] == row[2]                   # symmetric neighbors


def test_normalized_entropy_bounds():
    est = StateEstimator(make_graph(), [], CFG)
    for _ in range(3):
        est.update(scored_for((1, 0.99), (0, 0.01), (2, 0.01)))  # concentrated
    single = est.update(scored_for((1, 0.99), (0, 0.01), (2, 0.01)))
    est.reset()
    spread = est.update(scored_for((0, 0.4), (1, 0.4), (2, 0.4)))  # uniform-ish
    assert 0.0 <= single.entropy < spread.entropy <= 1.0
