"""Stage 3: group frames into "places" with K-Means.

k is not hand-picked: we sweep a small range and keep whichever k gives the
best silhouette score. This is still a single, one-shot clustering pass —
no temporal reasoning, no splitting/merging refinement.

Each place is then represented by one or more EXEMPLAR vectors rather than a
single averaged centroid: a small secondary KMeans runs within that place's
own frames (up to `max_exemplars_per_place`), so a place that looks
different from two angles or lighting conditions gets more than one
"reference look" instead of being flattened into a single blurry average.

Output:
  - place_assignments.json     list of N place ids, aligned with frame_names order
  - place_exemplars.npy        shape (M, 512): M >= num_places exemplar vectors
  - exemplar_place_ids.npy     shape (M,): which place each exemplar row belongs to
  - place_names.json           {"0": "Place_0", ...} template — rename these by
                                hand (or via label_places.py) to real names.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("build_places")


def choose_best_k(embeddings: np.ndarray, k_min: int, k_max: int) -> int:
    n_samples = embeddings.shape[0]
    k_max = min(k_max, n_samples - 1)  # silhouette needs at least 2 clusters, < n_samples
    if k_max < k_min:
        return max(2, min(k_min, n_samples - 1))

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        logger.info(f"k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score

    logger.info(f"Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_exemplars(
    embeddings: np.ndarray, assignments: list[int], max_exemplars_per_place: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each place, find up to `max_exemplars_per_place` representative
    vectors via a small secondary KMeans on that place's own members.

    Falls back to a single mean vector when a place has too few members to
    usefully sub-cluster.
    """
    exemplars: list[np.ndarray] = []
    exemplar_place_ids: list[int] = []

    for place_id in sorted(set(assignments)):
        member_indices = [i for i, a in enumerate(assignments) if a == place_id]
        members = embeddings[member_indices]
        k = min(max_exemplars_per_place, len(members))

        if k <= 1:
            exemplars.append(_normalize(members.mean(axis=0)))
            exemplar_place_ids.append(place_id)
            continue

        sub_labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(members)
        for sub_id in range(k):
            sub_members = members[sub_labels == sub_id]
            if len(sub_members) == 0:
                continue
            exemplars.append(_normalize(sub_members.mean(axis=0)))
            exemplar_place_ids.append(place_id)

    return np.stack(exemplars).astype("float32"), np.array(exemplar_place_ids, dtype="int64")


def build_places(
    embeddings: np.ndarray, k_min: int, k_max: int, max_exemplars_per_place: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    best_k = choose_best_k(embeddings, k_min, k_max)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    assignments = kmeans.fit_predict(embeddings).tolist()

    exemplars, exemplar_place_ids = build_exemplars(embeddings, assignments, max_exemplars_per_place)
    return assignments, exemplars, exemplar_place_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster embedded frames into places.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    embeddings_file = resolve_path(config["paths"]["embeddings_file"])
    assignments_file = resolve_path(config["paths"]["place_assignments_file"])
    exemplars_file = resolve_path(config["paths"]["place_exemplars_file"])
    exemplar_ids_file = resolve_path(config["paths"]["exemplar_place_ids_file"])
    names_file = resolve_path(config["paths"]["place_names_file"])
    k_min = config["clustering"]["k_min"]
    k_max = config["clustering"]["k_max"]
    max_exemplars = config["clustering"]["max_exemplars_per_place"]

    embeddings = np.load(embeddings_file)
    assignments, exemplars, exemplar_place_ids = build_places(embeddings, k_min, k_max, max_exemplars)

    with open(assignments_file, "w") as f:
        json.dump(assignments, f, indent=2)
    np.save(exemplars_file, exemplars)
    np.save(exemplar_ids_file, exemplar_place_ids)

    num_places = len(set(assignments))
    # Only write a fresh template if place_names.json doesn't already exist,
    # so a human's manual renaming isn't overwritten by re-running this stage.
    if not names_file.exists():
        default_names = {str(i): f"Place_{i}" for i in range(num_places)}
        with open(names_file, "w") as f:
            json.dump(default_names, f, indent=2)
        logger.info(
            f"Wrote default names to {names_file} — rename these (or run "
            f"src/label_places.py) after reviewing data/frames."
        )
    else:
        logger.info(f"{names_file} already exists — leaving your custom names alone.")

    logger.info(
        f"Built {num_places} places ({exemplars.shape[0]} exemplar vectors total) "
        f"from {len(assignments)} frames"
    )


if __name__ == "__main__":
    main()
