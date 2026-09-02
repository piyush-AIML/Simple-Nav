"""Re-tag stored observations with the CURRENT VLM prompt/schema without
re-embedding (scene tagger v2 upgrade, 2026-09-02).

Loads the ObservationStore under config.paths.observations_dir, re-runs the
configured VLM over each stored frame (reusing the stored detector objects),
replaces only scene_tags + landmarks, applies the same ±2 temporal scene
smoothing the offline mapping pass uses, and persists the store back. The
embeddings and the FAISS index are untouched — the JSONL rows change, the
vectors do not. Refuses to run when the VLM is unavailable (stub), because
silently keeping v1 tags would corrupt the v1->v2 migration.

Usage:
    python scripts/retag_observations.py [--config config.yaml] [--batch-size N]
    python -m src.mapping.build_map      # next: rebuild the map bundle
    python evaluate_suite.py             # then: re-measure
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root on path

from src.embed_frames import _smooth_scene_types, _batch_size
from src.mapping.observation_store import ObservationStore
from src.perception.scene_tagger import SCENE_TYPES, get_scene_tagger
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("retag_observations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    obs_dir = resolve_path(config["paths"]["observations_dir"])
    store = ObservationStore.load(obs_dir)
    observations = store.observations
    logger.info(f"Loaded {len(observations)} observations from {obs_dir}")

    tagger = get_scene_tagger(config)
    if tagger.name() == "stub":
        logger.error("VLM unavailable — refusing to keep stale v1 tags. "
                     "Fix perception.vlm_model / GPU / network, then re-run.")
        sys.exit(1)
    logger.info(f"Tagging with {tagger.name()}")

    batch_size = args.batch_size or _batch_size(config, "vlm_batch_size", 12)
    paths = [obs.frame_path for obs in observations]
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        objs_lists = [obs.objects for obs in observations[start:start + batch_size]]
        tags = tagger.tag_batch(chunk, objs_lists)
        for obs, tags_i in zip(observations[start:start + batch_size], tags):
            obs.scene_tags = tags_i.to_dict()
            obs.landmarks = tags_i.landmarks
        logger.info(f"Re-tagged {min(start + batch_size, len(paths))}/{len(paths)} frames")

    _smooth_scene_types(observations)

    with open(obs_dir / "encoder.json") as f:
        encoder_name = json.load(f)["model"]
    store.save(encoder_name)  # jsonl updated; embeddings/index byte-identical

    before_v2 = sum(
        1 for o in observations
        if "junction" in ((o.scene_tags or {}).get("scene_type") or "unknown")
        or "lobby" in ((o.scene_tags or {}).get("scene_type") or "unknown")
    )
    counts = Counter((o.scene_tags or {}).get("scene_type", "unknown") for o in observations)
    logger.info(f"Re-tagged {len(observations)} observations -> scene histogram: {dict(counts)}")
    if before_v2:
        logger.info(f"v2-only scene values seen (junction/lobby): {before_v2}")

    non_unknown = sum(n for k, n in counts.items() if k in SCENE_TYPES and k != "unknown")
    logger.info(f"Known-scene rate: {non_unknown}/{len(observations)} "
                f"({non_unknown / len(observations):.1%})")
    print("Next: python -m src.mapping.build_map  (rebuild the map bundle)")


if __name__ == "__main__":
    main()
