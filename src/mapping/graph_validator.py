"""Graph validation (§16): detect construction errors automatically, before
runtime. Checks: connectivity, isolated nodes, weak edges, suspicious nodes
(low observation count / high variance / incompatible semantics), and
duplicate-candidate pairs. Output is a human-readable + machine-readable
report — map quality is inspectable without opening every frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import networkx as nx

from src.mapping.place_builder import Place
from src.utils import setup_logger

logger = setup_logger("graph_validator")


@dataclass
class ValidationWarning:
    level: str     # "WARN" | "ERROR"
    check: str     # disconnected_component | isolated_node | weak_edge | suspicious_node | duplicate_candidate
    message: str

    def to_dict(self) -> dict:
        return {"level": self.level, "check": self.check, "message": self.message}


def validate_graph(
    graph: nx.Graph,
    places: list[Place],
    config: dict | None = None,
    observations: list | None = None,
) -> list[ValidationWarning]:
    cfg = config or {}
    warnings: list[ValidationWarning] = []
    place_by_id = {p.place_id: p for p in places}
    by_id = {o.id: o for o in (observations or [])}

    weak_edge_thr = float(cfg.get("weak_edge_threshold", 0.4))
    min_obs = int(cfg.get("min_observations_per_place", 3))
    high_var = float(cfg.get("high_variance_threshold", 0.35))
    dup_sim = float(cfg.get("duplicate_similarity_threshold", 0.9))

    # -- connectivity --
    components = list(nx.connected_components(graph))
    if len(components) > 1:
        for comp in components[1:]:
            names = ", ".join(sorted(comp))
            warnings.append(
                ValidationWarning("WARN", "disconnected_component",
                                  f"Unexpected disconnected component: {{{names}}}")
            )

    # -- isolated / weak nodes --
    for node, attrs in graph.nodes(data=True):
        if graph.degree(node) == 0:
            warnings.append(
                ValidationWarning("WARN", "isolated_node",
                                  f"Place {node} has no physical connections (degree 0)")
            )
        place = place_by_id.get(node)
        if place is not None and len(place.observation_ids) < min_obs:
            warnings.append(
                ValidationWarning("WARN", "suspicious_node",
                                  f"Place {node} has only {len(place.observation_ids)} observations (< {min_obs})")
            )
        var = (place.visual_stats.get("std_similarity") if place and place.visual_stats else None)
        if var is not None and var > high_var:
            warnings.append(
                ValidationWarning("WARN", "suspicious_node",
                                  f"Place {node} has high internal visual variance (std={var:.2f})")
            )
        if place is not None and place.scene_types:
            # "unknown" is missing evidence, not a conflicting type (Rule 4),
            # and a small minority is per-frame VLM flicker — only flag a
            # genuine mix of KNOWN types with real support
            known = {s: c for s, c in place.scene_types.items() if s != "unknown"}
            total = sum(known.values())
            if total > 0 and len(known) > 1:
                top = sorted(known.values(), reverse=True)
                minority_fraction = (total - top[0]) / total
                if minority_fraction >= 0.2:
                    warnings.append(
                        ValidationWarning("WARN", "suspicious_node",
                                          f"Place {node} mixes scene types: {known}")
                    )

    # -- weak edges --
    for u, v, attrs in graph.edges(data=True):
        conf = float(attrs.get("confidence", 0.0))
        support = int(attrs.get("supporting_observations", 0))
        if conf < weak_edge_thr or support < min_obs:
            warnings.append(
                ValidationWarning(
                    "WARN", "weak_edge",
                    f"Edge {u}-{v} is weak (confidence={conf:.2f}, support={support})"
                )
            )

    # -- duplicate candidates --
    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            a, b = places[i], places[j]
            if a.place_id not in graph or b.place_id not in graph:
                continue
            if a.place_id == b.place_id:
                continue
            sim = _place_similarity(a, b, by_id)
            if sim is not None and sim >= dup_sim:
                warnings.append(
                    ValidationWarning(
                        "WARN", "duplicate_candidate",
                        f"Place {a.place_id} and {b.place_id} may be duplicates "
                        f"(exemplar similarity {sim:.2f})"
                    )
                )

    logger.info(f"Graph validation: {len(warnings)} warning(s)")
    return warnings


def _place_similarity(a: Place, b: Place, by_id: dict) -> float | None:
    """Mean pairwise exemplar cosine similarity; None if no usable embeddings."""
    import numpy as np

    vecs_a = [by_id[e].embedding for e in a.exemplar_ids if e in by_id and by_id[e].embedding is not None]
    vecs_b = [by_id[e].embedding for e in b.exemplar_ids if e in by_id and by_id[e].embedding is not None]
    if not vecs_a or not vecs_b:
        return None
    sims = np.stack(vecs_a) @ np.stack(vecs_b).T
    return float(sims.mean())


def write_validation_report(warnings: list[ValidationWarning], md_path, json_path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump([w.to_dict() for w in warnings], f, indent=2)

    lines = ["# Graph Validation Report", ""]
    if not warnings:
        lines.append("No warnings — the graph looks clean.")
    else:
        for w in warnings:
            lines.append(f"- **[{w.level}]** `{w.check}` — {w.message}")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Validation report written: {json_path} and {md_path}")
