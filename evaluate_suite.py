"""Stage 27 research evaluation suite (planner v2 §10).

Metric definitions are the original planner's §30-31, implemented here:
localization (top-1, top-3, segment accuracy, transition accuracy, false
jump rate, reacquisition time, unknown precision), graph (node purity,
duplicate rate, merge error, edge precision/recall, fragmentation), and the
incremental ablation (visual -> +semantic -> +temporal -> +graph, §32).

CRITICAL CAVEAT (Rule 6 / planner v2 §10): on the current provisional
301-frame set every number is SELF-CONSISTENCY against pseudo-labels, NOT
accuracy against ground truth. Reports label themselves accordingly. When
real labels exist (data/test_labels.json mapping frame names -> place ids,
per evaluate.py), the same functions compute real accuracy instead.

Run:
    conda run -n ML python evaluate_suite.py
Outputs: data/evaluation/decision_log.jsonl + ablation_report.json
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import networkx as nx
import numpy as np

from src.localization.state_estimator import StateEstimator
from src.localization.tracker import LOST, LocalizationTracker, TRACKING, UNCERTAIN
from src.mapping.map_artifact import MapBundle
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("evaluate_suite")

PSEUDO_NOTE = (
    "SELF-CONSISTENCY on pseudo-labels (Rule 6): the ground truth below is the "
    "map's own place assignment, NOT physical accuracy. Drop real labels into "
    "data/test_labels.json and re-run to get accuracy against ground truth."
)


def find_bundle_dir(config: dict) -> Path:
    map_dir = resolve_path(config["paths"]["map_dir"])
    candidates = sorted(
        (p for p in map_dir.iterdir() if (p / "manifest.json").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No map bundle found under {map_dir} — run build_map.py first")
    return candidates[-1]


def gt_place_of_obs(bundle: MapBundle, labels: dict | None) -> dict[str, int]:
    """obs id -> ground-truth place id. `labels`: {obs_id: place_id} (real
    labels). None -> map-membership pseudo-labels."""
    if labels:
        return {oid: int(pid) for oid, pid in labels.items()}
    gt: dict[str, int] = {}
    for place in bundle.places:
        for oid in place.observation_ids:
            gt[oid] = int(place.place_id)
    return gt


def _run_pass(bundle: MapBundle, config: dict, variant: str = "graph",
              gt: dict[str, int] | None = None) -> tuple[list[dict], list[int]]:
    """Feed the bundle's observations through the tracker in timestamp order.
    Returns (per-frame rows, ground-truth place ids aligned by frame)."""
    cfg = copy.deepcopy(config)
    loc = cfg.setdefault("localization", {})
    if variant == "visual":
        loc["w_semantic"] = 0.0  # evidence = visual only
    tracker = LocalizationTracker(bundle, cfg)

    n = bundle.graph.number_of_nodes() or 1
    if variant in ("visual", "semantic", "temporal"):
        # replace the transition prior: uniform (visual/semantic) or
        # self-biased without graph structure (temporal). A complete graph
        # makes every state a neighbor, so the existing transition_prior()
        # produces exactly those matrices — no estimator changes needed.
        complete = nx.complete_graph(list(bundle.graph.nodes))
        if variant == "temporal":
            params = {"self_transition_prior": 0.7, "far_transition_prior": 0.0}
        else:
            params = {"self_transition_prior": 1.0 / n, "far_transition_prior": 1.0 / n}
        tracker._estimator = StateEstimator(complete, [], params)

    rows: list[dict] = []
    gt_ids: list[int] = []
    for obs in tracker.bundle.store.all():
        result = tracker.process_frame(
            obs.embedding, query_tags=obs.scene_tags,
            query_objects=obs.objects, timestamp=obs.timestamp,
        )
        cands = tracker.decision_log[-1]["candidates"] if tracker.decision_log else []
        rows.append(
            {
                "obs_id": obs.id,
                "frame_path": obs.frame_path,
                "timestamp": obs.timestamp,
                "gt": (gt or {}).get(obs.id),
                "reported": result["place_id"],
                "state": result["state"],
                "confidence_level": result["confidence_level"],
                "confidence": result["confidence"],
                "candidates": cands,
                "scene_tags": obs.scene_tags,
            }
        )
        gt_ids.append((gt or {}).get(obs.id, -1))
    return rows, gt_ids


def _graph_distance(graph: nx.Graph, a, b) -> int | None:
    if a is None or b is None:
        return None
    try:
        return nx.shortest_path_length(graph, int(a), int(b))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def run_localization_metrics(bundle: MapBundle, config: dict,
                             labels: dict | None = None) -> dict:
    gt = gt_place_of_obs(bundle, labels)
    rows, _ = _run_pass(bundle, config, "graph", gt)
    n = len(rows)

    # top-1 / top-3
    top1 = sum(1 for r in rows if r["gt"] is not None and r["reported"] == r["gt"])
    denom = sum(1 for r in rows if r["gt"] is not None)
    top3 = 0
    for r in rows:
        if r["gt"] is None:
            continue
        top3 += int(any(int(c.get("place_id", -1)) == r["gt"] for c in r["candidates"][:3]))
    top1 = top1 / denom if denom else 0.0
    top3 = top3 / denom if denom else 0.0

    # segment accuracy: per maximal run of one gt place, fraction correct
    run_accs: list[float] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1]["gt"] == rows[i]["gt"]:
            j += 1
        run_rows = rows[i:j + 1]
        if any(r["gt"] is not None for r in run_rows):
            correct = sum(1 for r in run_rows if r["gt"] is not None and r["reported"] == r["gt"])
            run_accs.append(correct / len(run_rows))
        i = j + 1
    segment_accuracy = float(np.mean(run_accs)) if run_accs else 0.0

    # transition accuracy: unique consecutive gt pairs vs unique consecutive
    # reported pairs
    gt_pairs = {(rows[i]["gt"], rows[i + 1]["gt"]) for i in range(n - 1)
                if rows[i]["gt"] is not None and rows[i + 1]["gt"] is not None
                and rows[i]["gt"] != rows[i + 1]["gt"]}
    rep_pairs = {(rows[i]["reported"], rows[i + 1]["reported"]) for i in range(n - 1)
                 if rows[i]["reported"] is not None and rows[i + 1]["reported"] is not None
                 and rows[i]["reported"] != rows[i + 1]["reported"]}
    trans_recall = len(gt_pairs & rep_pairs) / len(gt_pairs) if gt_pairs else 1.0
    trans_precision = len(gt_pairs & rep_pairs) / len(rep_pairs) if rep_pairs else 0.0

    # false jump rate: reported != gt AND the tracker jumped to a
    # non-adjacent place in one step
    jumps = 0
    for i in range(1, n):
        prev, cur = rows[i - 1]["reported"], rows[i]["reported"]
        if prev is None or cur is None or prev == cur:
            continue
        d = _graph_distance(bundle.graph, prev, cur)
        if d is not None and d > 1:
            jumps += 1
    false_jump_rate = jumps / max(1, n - 1)

    # reacquisition time: mean frames from LOST to TRACKING
    lost_starts, reacq, episodes = 0, 0.0, 0
    in_lost = False
    for r in rows:
        if r["state"] == LOST and not in_lost:
            in_lost, lost_starts = True, 1
        elif in_lost and r["state"] == TRACKING:
            reacq += lost_starts
            episodes += 1
            in_lost = False
        elif in_lost:
            lost_starts += 1
    reacquisition_frames = reacq / episodes if episodes else None

    # unknown precision: when the system says UNCERTAIN/LOST, how often is
    # its reported place actually wrong?
    flagged = [r for r in rows if r["state"] in (UNCERTAIN, LOST)]
    unknown_precision = (
        sum(1 for r in flagged if r["gt"] is not None and r["reported"] != r["gt"]) / len(flagged)
        if flagged else None
    )

    return {
        "frames": n,
        "label_source": "real" if labels else "pseudo",
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "segment_accuracy": round(segment_accuracy, 4),
        "transition_recall": round(trans_recall, 4),
        "transition_precision": round(trans_precision, 4),
        "false_jump_rate": round(false_jump_rate, 4),
        "reacquisition_time_frames": (round(reacquisition_frames, 2)
                                      if reacquisition_frames is not None else None),
        "unknown_precision": (round(unknown_precision, 4)
                              if unknown_precision is not None else None),
        "note": PSEUDO_NOTE if not labels else "Accuracy against real labels.",
    }


def run_graph_metrics(bundle: MapBundle, labels: dict | None = None) -> dict:
    # node purity: mean intra-place similarity vs mean cross-place exemplar
    # similarity
    intra = [p.visual_stats["mean_similarity"] for p in bundle.places
             if "mean_similarity" in p.visual_stats]
    mean_intra = float(np.mean(intra)) if intra else 0.0
    ex = bundle.exemplars
    if ex.ndim == 1:
        ex = ex.reshape(1, -1)
    sims = ex @ ex.T
    pid = bundle.exemplar_place_ids
    cross = [sims[i, j] for i in range(len(pid)) for j in range(i + 1, len(pid))
             if pid[i] != pid[j]]
    mean_cross = float(np.mean(cross)) if cross else 0.0
    purity = mean_intra / (mean_intra + mean_cross) if (mean_intra + mean_cross) > 0 else 0.0

    # duplicate rate: distinct place pairs above the reconciliation threshold
    # that remain separate nodes
    merge_thr = 0.92
    pairs = total = 0
    for i in range(len(pid)):
        for j in range(i + 1, len(pid)):
            if pid[i] == pid[j]:
                continue
            total += 1
            pairs += int(sims[i, j] >= merge_thr)
    duplicate_rate = pairs / total if total else 0.0

    # merge error: needs real labels; None otherwise (honest, not guessed)
    merge_error = None
    if labels:
        bad = 0
        for place in bundle.places:
            distinct = {labels.get(oid) for oid in place.observation_ids
                        if oid in labels}
            bad += int(len(distinct) > 1)
        merge_error = bad / len(bundle.places) if bundle.places else 0.0

    # edge precision/recall vs the walkthrough's true consecutive-place pairs
    gt_edges = set()
    seq: list[int] = []
    for obs in bundle.store.all():
        for place in bundle.places:
            if obs.id in place.observation_ids:
                seq.append(int(place.place_id))
                break
    for a, b in zip(seq, seq[1:]):
        if a != b:
            gt_edges.add(tuple(sorted((a, b))))
    graph_edges = {tuple(sorted((int(u), int(v)))) for u, v in bundle.graph.edges()}
    edge_precision = len(gt_edges & graph_edges) / len(graph_edges) if graph_edges else 0.0
    edge_recall = len(gt_edges & graph_edges) / len(gt_edges) if gt_edges else 1.0

    fragmentation = max(0, nx.number_connected_components(bundle.graph) - 1)

    return {
        "node_purity": round(purity, 4),
        "intra_place_mean_similarity": round(mean_intra, 4),
        "cross_place_mean_similarity": round(mean_cross, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "merge_error": round(merge_error, 4) if merge_error is not None else None,
        "edge_precision": round(edge_precision, 4),
        "edge_recall": round(edge_recall, 4),
        "fragmentation_components": fragmentation,
        "note": PSEUDO_NOTE,
    }


ABLATION_VARIANTS = ["visual", "semantic", "temporal", "graph"]


def run_ablation(bundle: MapBundle, config: dict, labels: dict | None = None) -> dict:
    """Incremental localization: visual -> +semantic -> +temporal -> +graph
    (original planner §32 progression). Each variant re-runs the full
    tracker with that signal set."""
    gt = gt_place_of_obs(bundle, labels)
    variants = []
    prev_top1 = -1.0
    for variant in ABLATION_VARIANTS:
        rows, _ = _run_pass(bundle, config, variant, gt)
        denom = sum(1 for r in rows if r["gt"] is not None)
        top1 = sum(1 for r in rows if r["gt"] is not None and r["reported"] == r["gt"])
        top1 = top1 / denom if denom else 0.0
        variants.append({
            "variant": variant,
            "signals": {
                "visual": True,
                "semantic": variant in ("semantic", "temporal", "graph"),
                "temporal": variant in ("temporal", "graph"),
                "graph": variant == "graph",
            },
            "top1_accuracy": round(top1, 4),
        })
        prev_top1 = top1
    semantic_active = any(
        obs.scene_tags and obs.scene_tags.get("scene_type", "unknown") != "unknown"
        for obs in bundle.store.all()
    )
    tops = [v["top1_accuracy"] for v in variants]
    return {
        "variants": variants,
        "semantic_active": semantic_active,
        "monotonic_or_explained": all(a <= b + 1e-9 for a, b in zip(tops, tops[1:])),
        "note": PSEUDO_NOTE,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 27 research evaluation suite.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    bundle = MapBundle.load(find_bundle_dir(config))
    eval_dir = resolve_path(config["paths"]["evaluation_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    labels: dict | None = None
    labels_file = eval_dir / "test_labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)
        logger.info(f"Using real labels from {labels_file}")
    else:
        logger.warning("No test_labels.json — metrics are pseudo-label self-consistency (Rule 6)")

    loc_metrics = run_localization_metrics(bundle, config, labels)
    graph_metrics = run_graph_metrics(bundle, labels)
    ablation = run_ablation(bundle, config, labels)

    # decision log for the failure inspector
    rows, _ = _run_pass(bundle, config, "graph", gt_place_of_obs(bundle, labels))
    log_path = eval_dir / "decision_log.jsonl"
    with open(log_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Decision log written: {log_path}")

    ablation_path = eval_dir / "ablation_report.json"
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "map": str(find_bundle_dir(config)),
        "localization_metrics": loc_metrics,
        "graph_metrics": graph_metrics,
        "ablation": ablation,
    }
    with open(ablation_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Ablation report written: {ablation_path}")

    print(json.dumps(loc_metrics, indent=2))
    print(json.dumps(graph_metrics, indent=2))
    print(json.dumps(ablation, indent=2))


if __name__ == "__main__":
    main()
