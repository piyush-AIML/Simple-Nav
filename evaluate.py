"""Evaluate localization accuracy against a small, hand-labeled test set.

Expects data/test_labels.json in the form:
    {"frame_0123.jpg": "Lobby", "frame_0456.jpg": "Room 101", ...}

For a meaningful number, these should ideally be frames NOT used when the
map's place exemplars were built (e.g. held out, or photos taken separately
from the mapping walkthrough) — otherwise you're just measuring whether
K-Means fit its own training data.

Also reports a naive baseline (plain color-histogram matching) alongside the
CNN-based approach, so the report has a real point of comparison rather than
just one number in isolation.

Run with:  python evaluate.py
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.localize import PlaceIndex, localize_image
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("evaluate")

HIST_BINS = (8, 8, 8)  # per-channel bins in HSV space


def histogram_embed(image_path) -> np.ndarray:
    """A deliberately simple baseline feature: a normalized HSV color
    histogram. No deep learning at all — this exists purely as a point of
    comparison for the CNN-based approach.
    """
    img = cv2.imread(str(image_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype("float32")
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def build_histogram_baseline(frames_dir, frame_names: list[str], assignments: list[int]) -> dict[int, np.ndarray]:
    """Average histogram per place, computed from the same frames used to
    build the real map (this baseline isn't meant to be rigorous — it's a
    simple reference point, clearly labeled as such in the output).
    """
    per_place: dict[int, list[np.ndarray]] = {}
    for name, place_id in zip(frame_names, assignments):
        per_place.setdefault(place_id, []).append(histogram_embed(frames_dir / name))

    centroids = {}
    for place_id, hists in per_place.items():
        mean_hist = np.mean(hists, axis=0)
        norm = np.linalg.norm(mean_hist)
        centroids[place_id] = mean_hist / norm if norm > 0 else mean_hist
    return centroids


def baseline_predict(image_path, centroids: dict[int, np.ndarray], place_names: dict[str, str]) -> str:
    vec = histogram_embed(image_path)
    best_id, best_score = None, -1.0
    for place_id, centroid in centroids.items():
        score = float(vec @ centroid)
        if score > best_score:
            best_id, best_score = place_id, score
    return place_names.get(str(best_id), f"Place_{best_id}")


def report(name: str, y_true: list[str], y_pred: list[str]) -> None:
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy on {len(y_true)} test frames: {acc:.2%}")
    print(classification_report(y_true, y_pred, zero_division=0))
    labels = sorted(set(y_true) | set(y_pred))
    print("Confusion matrix (rows=true, cols=predicted):")
    print("labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))


def main() -> None:
    config = load_config()
    exemplars_file = resolve_path(config["paths"]["place_exemplars_file"])
    exemplar_ids_file = resolve_path(config["paths"]["exemplar_place_ids_file"])
    names_file = resolve_path(config["paths"]["place_names_file"])
    test_labels_file = resolve_path(config["paths"]["test_labels_file"])
    frames_dir = resolve_path(config["paths"]["frames_dir"])
    assignments_file = resolve_path(config["paths"]["place_assignments_file"])
    frame_names_file = resolve_path(config["paths"]["frame_names_file"])
    threshold = config["navigation"]["confidence_threshold"]

    if not test_labels_file.exists():
        logger.warning(
            f"No test set found at {test_labels_file}. Create it by hand-labeling "
            f"a handful of frames from {frames_dir}, e.g.:\n"
            '  { "frame_00010.jpg": "Lobby", "frame_00050.jpg": "Room 101" }'
        )
        return

    with open(test_labels_file, "r") as f:
        test_labels = json.load(f)
    with open(assignments_file, "r") as f:
        assignments = json.load(f)
    with open(frame_names_file, "r") as f:
        frame_names = json.load(f)
    with open(names_file, "r") as f:
        place_names = json.load(f)

    place_index = PlaceIndex.load(exemplars_file, exemplar_ids_file, names_file)
    hist_centroids = build_histogram_baseline(frames_dir, frame_names, assignments)

    y_true, y_pred_cnn, y_pred_baseline = [], [], []
    for frame_name, true_place in test_labels.items():
        image_path = frames_dir / frame_name
        if not image_path.exists():
            logger.warning(f"Skipping missing frame: {image_path}")
            continue

        _, predicted_place, score = localize_image(str(image_path), place_index, confidence_threshold=threshold)
        baseline_place = baseline_predict(image_path, hist_centroids, place_names)

        y_true.append(true_place)
        y_pred_cnn.append(predicted_place)
        y_pred_baseline.append(baseline_place)

    if not y_true:
        logger.warning("No valid test entries found.")
        return

    report("CNN-based retrieval (this project's method)", y_true, y_pred_cnn)
    report("Naive baseline: color-histogram matching", y_true, y_pred_baseline)


if __name__ == "__main__":
    main()
