"""Compare registered VisualEncoders on the metrics that actually matter for
this project's task (place retrieval under viewpoint/lighting change), not
generic benchmark numbers. Planner v3 §7 (Stage 33) — the tool Finding F
found missing, without which §4's own acceptance rule ("gate the swap on
measurement, don't swap because it's generally better") is unenforceable.

  same_place_consistency:       mean pairwise cosine similarity between
                                observations already assigned to the same
                                place (higher = better — the encoder agrees
                                these are the same place from different
                                angles).
  different_place_separability: mean pairwise cosine similarity between
                                observations in different places (lower =
                                better — the encoder doesn't confuse
                                distinct places).
  separation_margin:            same_place_consistency -
                                different_place_separability — the actual
                                quantity worth comparing across encoders; a
                                big single number can hide a bad
                                different-place score.

Reuses the existing observation set (config.paths.observations_dir) and its
place assignments from the last successful build_map.py run — no new
capture needed. Writes data/evaluation/encoder_comparison.json.

Run:
    conda run -n ML python scripts/compare_encoders.py
    conda run -n ML python scripts/compare_encoders.py --encoders resnet18 dinov2_registers_small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_suite import find_bundle_dir
from src.mapping.map_artifact import MapBundle
from src.mapping.observation_store import ObservationStore
from src.utils import load_config, project_root, resolve_path, setup_logger

logger = setup_logger("compare_encoders")

DEFAULT_ENCODERS = ("resnet18", "dinov2_registers_small")


def place_assignments(bundle: MapBundle) -> dict[str, str]:
    """obs id -> place id from the last built map (unassigned obs omitted)."""
    mapping: dict[str, str] = {}
    for place in bundle.places:
        for oid in place.observation_ids:
            mapping[oid] = str(place.place_id)
    return mapping


def compute_metrics(embeddings: np.ndarray, place_ids: list[str]) -> dict:
    """The three Stage 33 metrics over an (N, D) embedding matrix whose rows
    align with place_ids. Pure numpy — unit-testable without any model.
    The raw pairwise similarity arrays are included so Stage 34's
    threshold recalibration can percentile-match against the real
    distributions (mean/std alone can't reproduce a percentile)."""
    sims = embeddings.astype("float32") @ embeddings.astype("float32").T
    n = sims.shape[0]
    pids = np.array(place_ids, dtype=object)
    same_mask = pids[:, None] == pids[None, :]
    np.fill_diagonal(same_mask, False)  # no self-similarity
    diff_mask = ~same_mask
    same_vals = sims[same_mask]
    diff_vals = sims[diff_mask]
    return {
        "n_observations": n,
        "same_place_consistency": float(same_vals.mean()),
        "same_place_std": float(same_vals.std()),
        "different_place_separability": float(diff_vals.mean()),
        "different_place_std": float(diff_vals.std()),
        "separation_margin": float(same_vals.mean() - diff_vals.mean()),
        "same_place_sims": [float(v) for v in np.sort(same_vals)],
        "different_place_sims": [float(v) for v in np.sort(diff_vals)],
    }


def compare(
    encoder_names: list[str],
    observations,
    place_of_obs: dict[str, str],
) -> dict[str, dict]:
    """Re-embed every observation with each encoder and compute the metrics.
    Never raises on a missing/ungated model — skip it with a logged warning
    (Rule 4 applies to tooling too), so one gated DINOv3 access failure
    doesn't block comparing the other two."""
    from src.embeddings.encoder import _REGISTRY, get_encoder

    root = project_root()
    results: dict[str, dict] = {}
    for name in encoder_names:
        if name not in _REGISTRY:
            logger.warning(f"Skipping unknown encoder {name!r} "
                           f"(registered: {sorted(_REGISTRY)})")
            continue
        try:
            encoder = get_encoder({"embedding": {"model": name}})
            paths, ids = [], []
            for obs in observations:
                p = root / obs.frame_path
                if not p.exists():
                    logger.warning(f"Skipping {obs.id}: frame missing {p}")
                    continue
                paths.append(str(p))
                ids.append(obs.id)
            if not paths:
                logger.warning(f"Encoder {name}: no usable frames — skipping")
                continue
            embeddings = encoder.batch_encode(paths)
            rows, pids = [], []
            for emb, oid in zip(embeddings, ids):
                pid = place_of_obs.get(oid)
                if pid is not None:
                    rows.append(emb)
                    pids.append(pid)
            metrics = compute_metrics(np.stack(rows), pids)
            metrics.update({"encoder": name, "version": encoder.version,
                            "dimension": encoder.dimension})
            results[name] = metrics
            logger.info(f"{name}: same={metrics['same_place_consistency']:.4f} "
                        f"diff={metrics['different_place_separability']:.4f} "
                        f"margin={metrics['separation_margin']:.4f}")
        except Exception as exc:  # gated model, missing deps, HF outage...
            logger.warning(f"Skipping encoder {name!r}: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VisualEncoders (planner v3 §7 / Stage 33).")
    parser.add_argument("--encoders", nargs="+", default=list(DEFAULT_ENCODERS),
                        help=f"registered encoder names (default: {list(DEFAULT_ENCODERS)})")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    store = ObservationStore.load(resolve_path(config["paths"]["observations_dir"]))
    bundle = MapBundle.load(find_bundle_dir(config))
    place_of_obs = place_assignments(bundle)

    results = compare(args.encoders, store.all(), place_of_obs)
    if not results:
        raise SystemExit("No encoder produced results — nothing to compare")

    out = resolve_path(config["paths"]["evaluation_dir"]) / "encoder_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "dataset": bundle.manifest.get("map_id"),
            "place_assignments_from": bundle.manifest.get("map_id"),
            "encoders": results,
            "best_separation_margin": max(
                results, key=lambda n: results[n]["separation_margin"]),
        }, f, indent=2)
    logger.info(f"Wrote {out}")
    for name, m in results.items():
        print(f"{name:24s} same={m['same_place_consistency']:.4f}  "
              f"diff={m['different_place_separability']:.4f}  "
              f"margin={m['separation_margin']:.4f}")


if __name__ == "__main__":
    main()
