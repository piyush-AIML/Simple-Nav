"""Optional floor-plan overlay for routes.

If a floor plan image and hand-placed place coordinates exist, draws the
route as a line over the actual floor plan instead of (or alongside) the
abstract graph layout. Entirely optional: if either file is missing, callers
should fall back to the abstract graph view — see `is_available()`.

Coordinates are set once via `python -m src.set_floor_plan_coords`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def is_available(floor_plan_image: Path, floor_plan_coords_file: Path) -> bool:
    return floor_plan_image.exists() and floor_plan_coords_file.exists()


def load_coords(floor_plan_coords_file: Path) -> dict[str, dict[str, float]]:
    with open(floor_plan_coords_file, "r") as f:
        return json.load(f)


def draw_route_on_floor_plan(
    floor_plan_image: Path,
    coords: dict[str, dict[str, float]],
    place_names: dict[str, str],
    route: Optional[list[int]] = None,
) -> Figure:
    """Draw every known place as a marker on the floor plan, with the given
    route (a list of place ids) highlighted as a connected path if provided.
    """
    img = plt.imread(floor_plan_image)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    for pid_str, xy in coords.items():
        ax.scatter([xy["x"]], [xy["y"]], c="#4c78a8", s=80, zorder=3)
        ax.annotate(
            place_names.get(pid_str, f"Place_{pid_str}"),
            (xy["x"], xy["y"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color="black",
        )

    if route and len(route) > 1:
        xs = [coords[str(pid)]["x"] for pid in route if str(pid) in coords]
        ys = [coords[str(pid)]["y"] for pid in route if str(pid) in coords]
        if len(xs) == len(route):
            ax.plot(xs, ys, c="#d95f02", linewidth=3, zorder=2)
            ax.scatter(xs[0], ys[0], c="#2ca02c", s=140, zorder=4, label="Start")
            ax.scatter(xs[-1], ys[-1], c="#d62728", s=140, zorder=4, label="Destination")
            ax.legend(loc="upper right")

    ax.axis("off")
    return fig
