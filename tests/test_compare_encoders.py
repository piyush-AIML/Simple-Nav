"""Fast (no-model) tests for scripts/compare_encoders.py (planner v3 §7):
the metric math is pure numpy, so the acceptance-critical parts are
testable without re-embedding anything."""

import numpy as np

from scripts.compare_encoders import compute_metrics, place_assignments


def test_compute_metrics_separates_same_from_different():
    # two tight clusters: place 0 around e0, place 1 around e1
    rng = np.random.default_rng(3)
    e0 = np.array([1.0, 0.0, 0.0], dtype="float32")
    e1 = np.array([0.0, 1.0, 0.0], dtype="float32")
    embeddings = np.concatenate([
        e0 + 0.05 * rng.standard_normal((4, 3)),
        e1 + 0.05 * rng.standard_normal((4, 3)),
    ]).astype("float32")
    place_ids = ["0", "0", "0", "0", "1", "1", "1", "1"]

    m = compute_metrics(embeddings, place_ids)
    assert m["same_place_consistency"] > 0.9
    assert m["different_place_separability"] < 0.3
    assert m["separation_margin"] > 0.6
    assert m["n_observations"] == 8


def test_compute_metrics_excludes_self_pairs():
    # two same-place observations with sim 0.5: all-pairs mean would be 0.75
    # (diagonal included), off-diagonal mean is 0.5 — the metric must be 0.5
    emb = np.array([[1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype="float32")
    m = compute_metrics(emb, ["0", "0"])
    assert abs(m["same_place_consistency"] - 0.5) < 1e-6
    assert m["n_observations"] == 2


def test_place_assignments_uses_map_membership():
    from collections import Counter

    from src.mapping.place_builder import Place

    places = [
        Place(place_id=0, observation_ids=["a", "b"], exemplar_ids=["a"],
              scene_types=Counter({"room": 2}), landmarks=[], visual_stats={}),
        Place(place_id=1, observation_ids=["c"], exemplar_ids=["c"],
              scene_types=Counter({"corridor": 1}), landmarks=[], visual_stats={}),
    ]
    assert place_assignments(type("B", (), {"places": places})()) == {
        "a": "0", "b": "0", "c": "1",
    }
