"""Stage 27 evaluation suite (evaluate_suite.py) on the toy bundle: metric
shapes, bounds, and the ablation progression — full correctness is verified
by the real-map run in the executor flow."""

from evaluate_suite import run_ablation, run_graph_metrics, run_localization_metrics

LOC_CFG = {"localization": {}, "routing": {}}


def test_localization_metrics_shapes(toy_bundle):
    m = run_localization_metrics(toy_bundle, LOC_CFG)
    for key in ("top1_accuracy", "top3_accuracy", "segment_accuracy",
                "transition_recall", "transition_precision", "false_jump_rate"):
        assert 0.0 <= m[key] <= 1.0
    assert m["label_source"] == "pseudo"
    # the toy bundle has orthogonal exemplars — an unambiguous map must
    # not score poorly
    assert m["top1_accuracy"] >= 0.5


def test_localization_metrics_with_real_labels(toy_bundle):
    labels = {"obs_0000": 0, "obs_0001": 1, "obs_0002": 1, "obs_0003": 2}
    m = run_localization_metrics(toy_bundle, LOC_CFG, labels=labels)
    assert m["label_source"] == "real"


def test_ablation_four_variants_in_order(toy_bundle):
    a = run_ablation(toy_bundle, LOC_CFG)
    assert [v["variant"] for v in a["variants"]] == ["visual", "semantic", "temporal", "graph"]
    # graph variant enables everything, visual variant only the visual signal
    assert a["variants"][0]["signals"] == {"visual": True, "semantic": False,
                                           "temporal": False, "graph": False}
    assert a["variants"][-1]["signals"]["graph"] is True


def test_graph_metrics_on_chain(toy_bundle):
    g = run_graph_metrics(toy_bundle)
    assert 0.0 <= g["node_purity"] <= 1.0
    assert g["edge_recall"] == 1.0   # walkthrough transitions are all in the graph
    assert g["edge_precision"] == 1.0
    assert g["fragmentation_components"] == 0  # single connected chain
