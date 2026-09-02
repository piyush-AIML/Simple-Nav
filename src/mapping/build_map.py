"""build_map: end-to-end mapping orchestrator (Stages 02-14).

Chains the new pipeline: ObservationStore -> temporal segmentation -> place
formation -> reconciliation -> transitions -> graph + junctions -> validation
-> versioned map artifact. The legacy KMeans pipeline (build_places.py +
build_graph.py + localize.py) is untouched and remains the baseline.

Run:
    conda run -n ML python -m src.mapping.build_map
"""

from __future__ import annotations

import argparse

from src.embeddings.encoder import get_encoder
from src.mapping.graph_builder import build_graph, detect_junctions
from src.mapping.graph_validator import validate_graph, write_validation_report
from src.mapping.map_artifact import write_map
from src.mapping.observation_store import ObservationStore
from src.mapping.place_builder import build_places
from src.mapping.place_reconciliation import reconcile_places
from src.mapping.segmentation import segment_observations
from src.mapping.transition_builder import extract_transitions
from src.perception import backend_banner
from src.utils import (
    load_config,
    print_config_issues,
    resolve_path,
    setup_logger,
    validate_config,
)

logger = setup_logger("build_map")


def place_sequence(store: ObservationStore, places) -> tuple[list[str], list[float]]:
    """Ordered place ids + timestamps for the observation stream."""
    place_of_obs: dict[str, str] = {}
    for place in places:
        for oid in place.observation_ids:
            place_of_obs[oid] = place.place_id
    ordered = store.all()
    seq = [place_of_obs.get(o.id) for o in ordered if o.id in place_of_obs]
    ts = [o.timestamp for o in ordered if o.id in place_of_obs]
    return seq, ts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the versioned map (Stages 02-14).")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    # Stage 30 (planner v2 §13): config validation is the FIRST line —
    # hard errors abort, warnings print
    issues = validate_config(config)
    print_config_issues(issues, logger)
    if any(level == "error" for level, _ in issues):
        raise SystemExit("config.yaml failed validation — fix the errors above")
    logger.info(backend_banner(config))

    mapping_cfg = config.get("mapping", {})
    obs_dir = resolve_path(config["paths"]["observations_dir"])
    map_dir = resolve_path(config["paths"]["map_dir"])
    map_identity = config.get("map", {})

    # 02-05: observations
    store = ObservationStore.load(obs_dir)
    observations = store.all()
    logger.info(f"Loaded {len(observations)} observations")

    # 08: temporal segmentation
    segments = segment_observations(
        observations,
        distance_threshold=mapping_cfg.get("segment_distance_threshold"),  # None -> adaptive
        change_window=mapping_cfg.get("segment_change_window", 2),
        min_length=mapping_cfg.get("segment_min_length", 3),
    )

    # 09: place formation
    places = build_places(
        observations,
        segments,
        exemplar_method=mapping_cfg.get("exemplar_method", "temporal_diversity"),
        max_exemplars=mapping_cfg.get("max_exemplars_per_place", 3),
    )

    # 10: reconciliation
    log_path = resolve_path(config["paths"]["evaluation_dir"]) / "reconciliation_log.jsonl"
    places = reconcile_places(places, observations, mapping_cfg, log_path=log_path)

    # 11: transitions
    seq, ts = place_sequence(store, places)
    transitions = extract_transitions(
        seq,
        min_persistence=mapping_cfg.get("transition_persistence", 3),
        minimum_edge_support=mapping_cfg.get("minimum_edge_support", 3),
        timestamps=ts,
    )

    # 12: graph + junctions
    graph = build_graph(
        places,
        transitions,
        edge_confidence_threshold=mapping_cfg.get("edge_confidence_threshold", 0.6),
        minimum_edge_support=mapping_cfg.get("minimum_edge_support", 3),
    )
    detect_junctions(
        graph,
        places,
        junction_min_degree=mapping_cfg.get("junction_min_degree", 3),
        junction_semantic_evidence=mapping_cfg.get("junction_semantic_evidence", True),
    )

    # 13: validation
    warnings = validate_graph(graph, places, mapping_cfg, observations=observations)
    write_validation_report(
        warnings,
        resolve_path(config["paths"]["map_dir"]) / "graph_validation.md",
        resolve_path(config["paths"]["map_dir"]) / "graph_validation.json",
    )

    # 14: versioned artifact
    encoder = get_encoder(config)
    target = write_map(
        map_dir,
        map_id=map_identity.get("map_id", "map_v1"),
        building_id=map_identity.get("building_id", "unknown"),
        floor_id=map_identity.get("floor_id", "unknown"),
        encoder_name=encoder.name,
        encoder_version=encoder.version,
        embedding_dimension=encoder.dimension,
        store=store,
        places=places,
        graph=graph,
    )
    logger.info(f"Map complete: {target}")


if __name__ == "__main__":
    main()
