"""Stage 01 benchmark: freeze and measure the CURRENT baseline pipeline.

This is the PERMANENT comparison harness. Run it now (provisional dataset,
301 frames) and again after every backbone change; record each run in
data/evaluation/reports/.

Baseline pipeline measured here (unchanged legacy code):
    ResNet18 -> K-Means (k by silhouette sweep) -> transition-count graph
    -> FAISS exemplar retrieval -> majority-vote smoothing

Usage:
    conda run -n ML python scripts/benchmark_baseline.py

Ground truth required:
    data/evaluation/heldout_labels.json      {"frame_00241.jpg": "area", ...}
    data/evaluation/exemplar_labels.json     {"place_id": "area", ...}

Both are created by the executor (Claude) by viewing frames; see the executor
document Stage 01. If they are missing the script writes only the structural
part of the report (graph counts, latency) and clearly marks metrics as absent.

Benchmark builds its own map under data/evaluation/baseline_map/ — the real
map artifacts in data/map/ are never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# scripts/ runs from the project root; make `src` importable when invoked as
# `python scripts/<name>.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.build_graph import build_graph
from src.build_places import build_places
from src.embedder import embed_image
from src.localize import PlaceIndex, LiveLocalizer
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("benchmark_baseline")

SPLIT_FRACTION_TRAIN = 0.8  # temporal held-out split of the provisional frames


def frame_paths(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def temporal_split(paths: list[Path], fraction_train: float) -> tuple[list[Path], list[Path]]:
    split_at = int(len(paths) * fraction_train)
    return paths[:split_at], paths[split_at:]


def run() -> dict:
    config = load_config()
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    eval_dir = resolve_path(config["paths"]["evaluation_dir"])
    k_min = config["clustering"]["k_min"]
    k_max = config["clustering"]["k_max"]
    max_exemplars = config["clustering"]["max_exemplars_per_place"]
    smoothing_window = config["navigation"]["smoothing_window"]

    labels_file = eval_dir / "heldout_labels.json"
    exemplar_labels_file = eval_dir / "exemplar_labels.json"
    baseline_map_dir = eval_dir / "baseline_map"
    baseline_map_dir.mkdir(parents=True, exist_ok=True)

    all_paths = frame_paths(frames_dir)
    train_paths, heldout_paths = temporal_split(all_paths, SPLIT_FRACTION_TRAIN)
    logger.info(f"Split: {len(train_paths)} train, {len(heldout_paths)} held-out")

    # ---------- build the baseline map on the train split only ----------
    t0 = time.perf_counter()
    train_embeddings = np.stack([embed_image(str(p)) for p in train_paths]).astype("float32")
    t_build = time.perf_counter() - t0
    logger.info(f"Embedded {len(train_paths)} train frames in {t_build:.1f}s")

    assignments, exemplars, exemplar_place_ids = build_places(
        train_embeddings, k_min, k_max, max_exemplars
    )
    graph = build_graph(assignments)
    place_names = {str(i): f"Place_{i}" for i in sorted(set(assignments))}
    place_index = PlaceIndex(exemplars, exemplar_place_ids, place_names)

    # representative train frame per exemplar (for the executor's labeling step)
    exemplar_frames: dict[str, list[str]] = {pid: [] for pid in place_names}
    for row, place_id in enumerate(exemplar_place_ids):
        emb = exemplars[row]
        dists = np.linalg.norm(train_embeddings - emb, axis=1)
        nearest = int(np.argmin(dists))
        exemplar_frames[str(place_id)].append(train_paths[nearest].name)
    with open(eval_dir / "exemplar_frames.json", "w") as f:
        json.dump(exemplar_frames, f, indent=2)

    report: dict = {
        "dataset": "provisional",
        "provisional_note": (
            "Built from 301 pre-extracted frames (source video missing). "
            "Without human area labels the metrics default to full-data KMeans "
            "pseudo-labels (self-consistency/generalization, NOT physical "
            "accuracy) — drop real labels into heldout_labels.json + "
            "exemplar_labels.json to get physical-area metrics. Re-run with "
            "real mapping+evaluation walkthroughs before research conclusions."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split": {
            "total_frames": len(all_paths),
            "train_frames": len(train_paths),
            "heldout_frames": len(heldout_paths),
            "fraction_train": SPLIT_FRACTION_TRAIN,
        },
        "params": {
            "embedding": "resnet18_512d",
            "k_min": k_min,
            "k_max": k_max,
            "max_exemplars_per_place": max_exemplars,
            "smoothing_window": smoothing_window,
            "place_index": "faiss_flat_inner_product",
            "graph": "transition_count_undirected",
        },
        "graph": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "chosen_k": len(place_names),
            "exemplar_count": int(exemplars.shape[0]),
        },
        "metrics": {},
    }
    latencies: list[float] = []

    # ---------- held-out localization metrics (require labels) ----------
    real_labels = labels_file.exists() and exemplar_labels_file.exists()

    if real_labels:
        with open(labels_file) as f:
            heldout_labels: dict[str, str] = json.load(f)
        with open(exemplar_labels_file) as f:
            place_to_area: dict[str, str] = json.load(f)

        y_pred_top1: list[str] = []
        y_pred_top3: list[list[str]] = []
        true_areas: list[str] = []
        latencies: list[float] = []

        # per-frame localization: raw FAISS top-3 (single-frame behavior)
        # sequence localization: majority-vote LiveLocalizer (live-tracking behavior)
        localizer = LiveLocalizer(place_index, window=smoothing_window)
        seq_pred_places: list[int] = []

        for p in heldout_paths:
            t0 = time.perf_counter()
            emb = embed_image(str(p))
            top3 = place_index.query(emb, top_k=3)
            latency = time.perf_counter() - t0
            latencies.append(latency)

            pred_place = top3[0][0]
            y_pred_top1.append(place_to_area.get(str(pred_place), "UNKNOWN_AREA"))
            y_pred_top3.append([place_to_area.get(str(pid), "UNKNOWN_AREA") for pid, _, _ in top3])
            true_areas.append(heldout_labels[p.name])

            # majority-vote over the sliding window of raw top-1 places
            localizer.update(emb)
            seq_pred_places.append(localizer.history[-1] if localizer.history else pred_place)

        top1 = sum(a == b for a, b in zip(y_pred_top1, true_areas)) / len(true_areas)
        top3 = sum(
            true in preds for true, preds in zip(true_areas, y_pred_top3)
        ) / len(true_areas)

        # false jump rate: predicted place changes while the true area is unchanged
        jumps = 0
        frames_with_unchanged_area = 0
        for i in range(1, len(seq_pred_places)):
            if true_areas[i] == true_areas[i - 1]:
                frames_with_unchanged_area += 1
                if seq_pred_places[i] != seq_pred_places[i - 1]:
                    jumps += 1
        false_jump_rate = jumps / frames_with_unchanged_area if frames_with_unchanged_area else None

        # transition detection: true area changes vs predicted place changes within ±3 frames
        true_transitions = [
            i for i in range(1, len(true_areas)) if true_areas[i] != true_areas[i - 1]
        ]
        detected_transitions = [
            i for i in range(1, len(seq_pred_places)) if seq_pred_places[i] != seq_pred_places[i - 1]
        ]
        detected_true = sum(
            any(abs(d - t) <= 3 for d in detected_transitions) for t in true_transitions
        )
        transition_recall = detected_true / len(true_transitions) if true_transitions else None
        transition_precision = (
            sum(any(abs(d - t) <= 3 for t in true_transitions) for d in detected_transitions)
            / len(detected_transitions)
            if detected_transitions
            else None
        )

        report["metrics"].update(
            {
                "top1_accuracy": round(top1, 4),
                "top3_accuracy": round(top3, 4),
                "false_jump_rate": round(false_jump_rate, 4) if false_jump_rate is not None else None,
                "transition_detection_recall": round(transition_recall, 4) if transition_recall is not None else None,
                "transition_detection_precision": round(transition_precision, 4) if transition_precision is not None else None,
                "unknown_place_predictions": int(sum(1 for a in y_pred_top1 if a == "UNKNOWN_AREA")),
            }
        )
    else:
        # No human labels available (executor model has no vision in this
        # session) -> pseudo-labels: cluster ALL frames with the same KMeans
        # pipeline and measure whether the train-built map reproduces the
        # full-data cluster structure on the UNSEEN held-out frames. This is
        # a self-consistency / generalization measure, NOT physical accuracy.
        # Replace by dropping real labels into heldout_labels.json +
        # exemplar_labels.json (see data/evaluation/*.template.json).
        logger.warning(
            "No ground truth labels found — using full-data KMeans pseudo-labels. "
            "Metrics are labeled self-consistency and are NOT physical accuracy. "
            "See data/evaluation/*.template.json to add real area labels."
        )
        heldout_embeddings = np.stack([embed_image(str(p)) for p in heldout_paths]).astype("float32")
        all_embeddings = np.concatenate([train_embeddings, heldout_embeddings], axis=0)

        from sklearn.cluster import KMeans as _KMeans
        from src.build_places import choose_best_k as _choose_best_k

        k_full = _choose_best_k(all_embeddings, k_min, k_max)
        pseudo_clusters = _KMeans(n_clusters=k_full, n_init=10, random_state=42).fit_predict(all_embeddings)
        train_clusters = pseudo_clusters[: len(train_embeddings)]
        heldout_clusters = pseudo_clusters[len(train_embeddings):]

        # map each train place -> majority pseudo-cluster of its train frames
        place_to_cluster: dict[str, int] = {}
        for pid in sorted(set(assignments)):
            members = [c for a, c in zip(assignments, train_clusters) if a == pid]
            place_to_cluster[str(pid)] = int(max(set(members), key=members.count))

        y_pred_top1, y_pred_top3, true_areas, latencies = [], [], [], []
        localizer = LiveLocalizer(place_index, window=smoothing_window)
        seq_pred_places: list[int] = []

        for i, p in enumerate(heldout_paths):
            t0 = time.perf_counter()
            emb = embed_image(str(p))
            top3 = place_index.query(emb, top_k=3)
            latencies.append(time.perf_counter() - t0)

            pred_place = top3[0][0]
            y_pred_top1.append(place_to_cluster.get(str(pred_place), -1))
            y_pred_top3.append([place_to_cluster.get(str(pid), -1) for pid, _, _ in top3])
            true_areas.append(int(heldout_clusters[i]))

            localizer.update(emb)
            seq_pred_places.append(localizer.history[-1] if localizer.history else pred_place)

        top1 = sum(a == b for a, b in zip(y_pred_top1, true_areas)) / len(true_areas)
        top3 = sum(
            (b in preds) for b, preds in zip(true_areas, y_pred_top3)
        ) / len(true_areas)

        jumps = 0
        frames_with_unchanged_area = 0
        for i in range(1, len(seq_pred_places)):
            if true_areas[i] == true_areas[i - 1]:
                frames_with_unchanged_area += 1
                if seq_pred_places[i] != seq_pred_places[i - 1]:
                    jumps += 1
        false_jump_rate = jumps / frames_with_unchanged_area if frames_with_unchanged_area else None

        true_transitions = [i for i in range(1, len(true_areas)) if true_areas[i] != true_areas[i - 1]]
        detected_transitions = [
            i for i in range(1, len(seq_pred_places)) if seq_pred_places[i] != seq_pred_places[i - 1]
        ]
        detected_true = sum(
            any(abs(d - t) <= 3 for d in detected_transitions) for t in true_transitions
        )
        transition_recall = detected_true / len(true_transitions) if true_transitions else None
        transition_precision = (
            sum(any(abs(d - t) <= 3 for t in true_transitions) for d in detected_transitions)
            / len(detected_transitions)
            if detected_transitions
            else None
        )

        report["pseudo_label_method"] = (
            f"full-data KMeans (k={k_full}, silhouette-selected) as pseudo ground truth; "
            "train places mapped to pseudo-clusters by majority vote; metrics measure "
            "map generalization to unseen frames, not physical accuracy"
        )
        report["metrics"].update(
            {
                "top1_accuracy_self_consistency": round(top1, 4),
                "top3_accuracy_self_consistency": round(top3, 4),
                "false_jump_rate_self_consistency": round(false_jump_rate, 4) if false_jump_rate is not None else None,
                "transition_detection_recall_self_consistency": round(transition_recall, 4) if transition_recall is not None else None,
                "transition_detection_precision_self_consistency": round(transition_precision, 4) if transition_precision is not None else None,
                "pseudo_cluster_count": int(k_full),
            }
        )

    # ---------- latency ----------
    if latencies:
        report["metrics"]["latency_mean_ms"] = round(float(np.mean(latencies)) * 1000, 2)
        report["metrics"]["latency_p95_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 2)

    # ---------- write report ----------
    eval_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = eval_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "baseline_report.json"
    md_path = reports_dir / "baseline_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_md(report))
    logger.info(f"Report written: {json_path}")

    # also a dated copy
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (reports_dir / f"baseline_report_{stamp}.json").write_text(json.dumps(report, indent=2))
    return report


def render_md(report: dict) -> str:
    m = report.get("metrics", {})
    g = report.get("graph", {})
    lines = [
        "# Baseline Report (provisional)",
        "",
        f"*Generated {report.get('timestamp')}* — dataset: `{report.get('dataset')}`",
        "",
        f"> {report.get('provisional_note', '')}",
        "",
    ]
    if report.get("pseudo_label_method"):
        lines.append(f"> **Labels:** {report['pseudo_label_method']}")
        lines.append("")
    lines += [
        "## Graph",
        f"- nodes: {g.get('node_count')} · edges: {g.get('edge_count')} · k: {g.get('chosen_k')} · exemplars: {g.get('exemplar_count')}",
        "",
        "## Metrics (held-out)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in m.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Split")
    lines.append(f"- total {report['split']['total_frames']} · train {report['split']['train_frames']} · held-out {report['split']['heldout_frames']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the baseline pipeline (permanent harness).")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    run()


if __name__ == "__main__":
    main()
