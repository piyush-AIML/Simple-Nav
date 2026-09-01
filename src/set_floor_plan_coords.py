"""One-time interactive tool: click on a floor plan image to record where
each mapped place sits, so routes can later be drawn on the actual floor
plan instead of an abstract graph layout.

Usage:
  1. Put a floor plan image at data/map/floor_plan.png (or update config.yaml)
  2. Run:  python -m src.set_floor_plan_coords
  3. Click on the image once for each place, in the printed order.
     Close the window (or wait for the last click) when done.

This is entirely optional — if you skip it, the app just shows the abstract
graph view instead of a floor-plan overlay.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("set_floor_plan_coords")


def main() -> None:
    parser = argparse.ArgumentParser(description="Click to place each mapped place on a floor plan image.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    floor_plan_image = resolve_path(config["paths"]["floor_plan_image"])
    coords_file = resolve_path(config["paths"]["floor_plan_coords_file"])
    names_file = resolve_path(config["paths"]["place_names_file"])

    if not floor_plan_image.exists():
        raise SystemExit(
            f"No floor plan image found at {floor_plan_image}. "
            "Add one (any PNG/JPG of your building's layout) and try again."
        )

    with open(names_file, "r") as f:
        place_names = json.load(f)

    place_ids = sorted(place_names.keys(), key=int)
    logger.info("Click on the floor plan once for each place, in this order:")
    for pid in place_ids:
        logger.info(f"  {pid}: {place_names[pid]}")

    img = plt.imread(floor_plan_image)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img)
    ax.set_title(f"Click for: {place_names[place_ids[0]]} (1 of {len(place_ids)})")

    points = plt.ginput(n=len(place_ids), timeout=0)
    plt.close(fig)

    if len(points) != len(place_ids):
        raise SystemExit("Not enough clicks registered — run again and click once per place.")

    coords = {pid: {"x": float(x), "y": float(y)} for pid, (x, y) in zip(place_ids, points)}
    with open(coords_file, "w") as f:
        json.dump(coords, f, indent=2)

    logger.info(f"Saved floor plan coordinates for {len(coords)} places to {coords_file}")


if __name__ == "__main__":
    main()
