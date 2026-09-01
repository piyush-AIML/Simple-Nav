"""Probabilistic state estimator (§21 / Stage 18): a Bayes filter over places.

    P(S_t | O_1..t)  ∝  L_t(S) · Σ_{S'} P(S | S') · prior(S')

- L_t(S): observation likelihood from scored candidates (softmax over totals)
- P(S | S'): transition prior from graph connectivity + transition stats
- residual probability mass is explicit UNKNOWN mass (§Rule 4: unknown is
  better than confidently wrong)

No neural sequence model — a simple probability update (§21).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.mapping.transition_builder import TransitionStats
from src.utils import setup_logger

logger = setup_logger("state_estimator")


@dataclass
class StateEstimate:
    posterior: dict[int, float]
    unknown_mass: float
    best_place_id: int | None
    best_score: float
    entropy: float

    def to_dict(self) -> dict:
        return {
            "posterior": {str(k): round(float(v), 4) for k, v in self.posterior.items()},
            "unknown_mass": round(float(self.unknown_mass), 4),
            "best_place_id": self.best_place_id,
            "best_score": round(float(self.best_score), 4),
            "entropy": round(float(self.entropy), 4),
        }


def softmax_likelihood(scored, temperature: float = 0.1,
                       evidence: dict[int, float] | None = None) -> dict[int, float]:
    """Candidate evidence -> normalized observation likelihoods via softmax.

    evidence: place_id -> observation-driven evidence in [0, 1] (visual +
    semantic only). Default: the candidate's visual score. Temporal/graph
    terms MUST NOT enter the likelihood — they belong to the transition prior,
    and double-counting them makes weak evidence look strong (the belief then
    never goes LOST, violating §Rule 4).
    """
    if not scored:
        return {}
    if evidence is None:
        values = [float(c.visual_score) for c in scored]
    else:
        values = [float(evidence.get(c.place_id, c.visual_score)) for c in scored]
    values = np.array(values, dtype="float64")
    if temperature <= 0:
        temperature = 1e-6
    exp = np.exp((values - values.max()) / temperature)
    probs = exp / exp.sum()
    return {c.place_id: float(p) for c, p in zip(scored, probs)}


def transition_prior(
    graph: nx.Graph,
    transitions: list[TransitionStats],
    self_transition: float = 0.7,
    far_transition: float = 0.05,
) -> dict[int, dict[int, float]]:
    """P(S_t | S_{t-1}): proper row-stochastic transition matrix. Self gets
    self_transition; each neighbor gets a share of the remaining mass by edge
    confidence; every non-neighbor gets far_transition (a small chance of
    moving several places between observations — §20 soft penalty).
    Isolated nodes keep the remaining mass on self."""
    priors: dict[int, dict[int, float]] = {}
    nodes = list(graph.nodes)
    for node in nodes:
        row: dict[int, float] = {int(node): self_transition}
        neighbors = list(graph.neighbors(node))
        far_nodes = [n for n in nodes if n != node and n not in neighbors]
        far_mass = far_transition * len(far_nodes)
        remaining = max(0.0, 1.0 - self_transition - far_mass)
        if neighbors:
            confidences = [float(graph[node][nbr].get("confidence", 1.0)) for nbr in neighbors]
            total_conf = sum(confidences) or 1.0
            for nbr, conf in zip(neighbors, confidences):
                row[int(nbr)] = remaining * (conf / total_conf)
        else:
            row[int(node)] += remaining  # isolated: keep the mass on self
        for far in far_nodes:
            row[int(far)] = far_transition
        priors[int(node)] = row
    return priors


class StateEstimator:
    def __init__(self, graph: nx.Graph, transitions: list[TransitionStats],
                 config: dict | None = None):
        cfg = config or {}
        self.graph = graph
        self.self_transition = float(cfg.get("self_transition_prior", 0.7))
        self.far_transition = float(cfg.get("far_transition_prior", 0.05))
        self.likelihood_temperature = float(cfg.get("likelihood_temperature", 0.1))
        self._transition_prior = transition_prior(
            graph, transitions, self.self_transition, self.far_transition
        )
        self._prior: dict[int, float] | None = None

    def reset(self, prior: dict[int, float] | None = None) -> None:
        """Clear the belief (LOST/reacquisition). None -> uniform over places."""
        self._prior = prior

    def _init_prior(self) -> dict[int, float]:
        nodes = list(self.graph.nodes)
        if not nodes:
            return {}
        mass = 1.0 / len(nodes)
        return {int(n): mass for n in nodes}

    def update(self, scored, evidence: dict[int, float] | None = None,
               timestamp: float | None = None) -> StateEstimate:
        if not scored:
            return StateEstimate(posterior={}, unknown_mass=1.0,
                                 best_place_id=None, best_score=0.0, entropy=1.0)
        if self._prior is None:
            self._prior = self._init_prior()

        likelihood = softmax_likelihood(scored, self.likelihood_temperature, evidence)

        # predicted prior: Σ_{S'} P(S | S') · prior(S'), then NORMALIZED —
        # the posterior stays raw (sub-probability) so its residual is the
        # explicit unknown mass
        predicted: dict[int, float] = {}
        for s_prime, prob in self._prior.items():
            row = self._transition_prior.get(s_prime, {s_prime: 1.0})
            for s, p in row.items():
                predicted[s] = predicted.get(s, 0.0) + prob * p
        pred_total = sum(predicted.values())
        predicted_norm = {k: v / pred_total for k, v in predicted.items()} if pred_total > 0 else predicted

        # filter update: posterior ∝ likelihood × normalized prediction
        posterior: dict[int, float] = {}
        for place_id, pred in predicted_norm.items():
            mass = likelihood.get(place_id, 0.0) * pred
            if mass > 0:
                posterior[place_id] = mass

        self._prior = posterior

        best_id = max(posterior, key=posterior.get) if posterior else None
        best_score = posterior[best_id] if best_id is not None else 0.0
        unknown_mass = max(0.0, 1.0 - sum(posterior.values()))
        entropy = _normalized_entropy(posterior, len(self.graph.nodes))
        return StateEstimate(posterior, unknown_mass, best_id, best_score, entropy)


def _normalized_entropy(posterior: dict[int, float], n_places: int) -> float:
    if not posterior or n_places <= 1:
        return 0.0 if posterior else 1.0
    probs = np.array(list(posterior.values()), dtype="float64")
    probs = probs / probs.sum()
    h = -float((probs * np.log(probs + 1e-12)).sum())
    return h / np.log(n_places)  # normalized to [0, 1]
