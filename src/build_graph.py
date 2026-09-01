"""Stage 4: build a graph of which places connect to which.

Rule is deliberately simple: whenever the place assigned to consecutive
frames changes from A to B, that's evidence of an A-B connection. Edge
weight = how many times that transition was observed (not used for routing
by default — see navigate.py — but kept for anyone who wants to experiment
with weighted shortest paths later).
"""

from __future__ import annotations

import argparse
import json
import pickle

import networkx as nx

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("build_graph")


def build_graph(assignments: list[int]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(set(assignments))

    for prev_place, curr_place in zip(assignments, assignments[1:]):
        if prev_place == curr_place:
            continue
        if graph.has_edge(prev_place, curr_place):
            graph[prev_place][curr_place]["weight"] += 1
        else:
            graph.add_edge(prev_place, curr_place, weight=1)

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the place-connectivity graph.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    assignments_file = resolve_path(config["paths"]["place_assignments_file"])
    graph_file = resolve_path(config["paths"]["graph_file"])

    with open(assignments_file, "r") as f:
        assignments = json.load(f)

    graph = build_graph(assignments)

    with open(graph_file, "wb") as f:
        pickle.dump(graph, f)

    logger.info(
        f"Built graph with {graph.number_of_nodes()} nodes and "
        f"{graph.number_of_edges()} edges -> {graph_file}"
    )


if __name__ == "__main__":
    main()
