"""Interactive helper for naming places, instead of hand-editing JSON.

Shows a few representative frames for each place (the ones closest to that
place's own exemplar vectors — i.e. its most "typical" looks) and lets you
type a name for it. Press Enter to keep the current/default name.

Run with:  python -m src.label_places
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("label_places")


def pick_example_frame_indices(
    embeddings: np.ndarray, assignments: list[int], place_id: int, exemplars: np.ndarray, exemplar_place_ids: np.ndarray
) -> list[int]:
    """For each exemplar belonging to this place, find the actual frame
    closest to it — these become the "representative photos" shown to the
    person doing the labeling.
    """
    member_indices = [i for i, a in enumerate(assignments) if a == place_id]
    place_exemplar_rows = [i for i, pid in enumerate(exemplar_place_ids) if pid == place_id]

    chosen = []
    for ex_row in place_exemplar_rows:
        exemplar_vec = exemplars[ex_row]
        sims = embeddings[member_indices] @ exemplar_vec
        best_local_idx = member_indices[int(np.argmax(sims))]
        if best_local_idx not in chosen:
            chosen.append(best_local_idx)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively name each discovered place.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    embeddings = np.load(resolve_path(config["paths"]["embeddings_file"]))
    exemplars = np.load(resolve_path(config["paths"]["place_exemplars_file"]))
    exemplar_place_ids = np.load(resolve_path(config["paths"]["exemplar_place_ids_file"]))
    with open(resolve_path(config["paths"]["place_assignments_file"])) as f:
        assignments = json.load(f)
    with open(resolve_path(config["paths"]["frame_names_file"])) as f:
        frame_names = json.load(f)
    names_file = resolve_path(config["paths"]["place_names_file"])
    with open(names_file) as f:
        place_names = json.load(f)

    frames_dir = resolve_path(config["paths"]["frames_dir"])

    for place_id in sorted(set(assignments)):
        current_name = place_names.get(str(place_id), f"Place_{place_id}")
        example_indices = pick_example_frame_indices(embeddings, assignments, place_id, exemplars, exemplar_place_ids)

        fig, axes = plt.subplots(1, len(example_indices), figsize=(4 * len(example_indices), 4))
        if len(example_indices) == 1:
            axes = [axes]
        for ax, idx in zip(axes, example_indices):
            img = Image.open(frames_dir / frame_names[idx])
            ax.imshow(img)
            ax.axis("off")
        fig.suptitle(f"Place {place_id} — currently named {current_name!r}")
        plt.show(block=False)
        plt.pause(0.1)

        new_name = input(f"Name for place {place_id} [{current_name}]: ").strip()
        plt.close(fig)
        if new_name:
            place_names[str(place_id)] = new_name

    with open(names_file, "w") as f:
        json.dump(place_names, f, indent=2)
    logger.info(f"Saved place names to {names_file}")


if __name__ == "__main__":
    main()
