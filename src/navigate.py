"""Stage 6 (runtime): "how do I get there" via shortest path.

Loads the saved place graph and computes a shortest path between two named
places using NetworkX. By default this favors STRONGLY-OBSERVED connections
(edges the pipeline saw many transitions across) rather than plain
fewest-hops: edge weight in the graph is a transition COUNT, so higher is
"more reliable," and we route by minimizing the inverse of that — i.e. the
routing cost of a well-traveled edge is low, and a rarely-seen edge is
treated as more expensive to cross. Set use_weighted=False for plain
unweighted shortest path.

Still simple: no confidence-weighted rerouting, no lost-tracking detection —
if a path exists, it's returned as a plain ordered list of place names.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("navigate")


def load_graph(graph_file: Path) -> nx.Graph:
    with open(graph_file, "rb") as f:
        return pickle.load(f)


def load_place_names(names_file: Path) -> dict[str, str]:
    with open(names_file, "r") as f:
        return json.load(f)


def name_to_id_map(place_names: dict[str, str]) -> dict[str, int]:
    return {name: int(pid) for pid, name in place_names.items()}


def _inverse_count_weight(u, v, edge_attrs: dict) -> float:
    """Higher observed transition count -> lower routing cost."""
    count = edge_attrs.get("weight", 1)
    return 1.0 / max(count, 1)


def get_route(
    graph: nx.Graph, source_id: int, dest_id: int, use_weighted: bool = True
) -> Optional[list[int]]:
    """Return the list of place ids on the shortest path, or None if unreachable."""
    weight = _inverse_count_weight if use_weighted else None
    try:
        return nx.shortest_path(graph, source=source_id, target=dest_id, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def format_directions(route: list[int], place_names: dict[str, str]) -> str:
    names = [place_names.get(str(pid), f"Place_{pid}") for pid in route]
    return " -> ".join(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get directions between two places.")
    parser.add_argument("--source", required=True, help="Name of the starting place")
    parser.add_argument("--destination", required=True, help="Name of the destination place")
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use plain fewest-hops routing instead of favoring strongly-observed edges",
    )
    parser.add_argument("--speak", action="store_true", help="Speak the directions aloud (needs pyttsx3 + audio)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    graph_file = resolve_path(config["paths"]["graph_file"])
    names_file = resolve_path(config["paths"]["place_names_file"])
    use_weighted = config["navigation"].get("use_weighted_routing", True) and not args.unweighted

    graph = load_graph(graph_file)
    place_names = load_place_names(names_file)
    name_to_id = name_to_id_map(place_names)

    if args.source not in name_to_id:
        raise SystemExit(f"Unknown source place: {args.source!r}. Known: {list(name_to_id)}")
    if args.destination not in name_to_id:
        raise SystemExit(f"Unknown destination place: {args.destination!r}. Known: {list(name_to_id)}")

    route = get_route(graph, name_to_id[args.source], name_to_id[args.destination], use_weighted)
    if route is None:
        logger.warning(f"No path found between {args.source!r} and {args.destination!r}")
        return

    directions = format_directions(route, place_names)
    logger.info("Route: " + directions)

    if args.speak:
        from src.speak import speak

        speak(f"Route: {directions.replace('->', 'then')}")


if __name__ == "__main__":
    main()
