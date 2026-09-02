"""Continuous camera-based localization with automatic re-routing.

Opens your webcam, periodically grabs a frame, localizes it against the map,
and whenever your tracked place changes, recomputes and reprints directions
to your chosen destination. The Bayes tracker's K-consecutive confirmation
avoids flicker between visually similar places, and it holds your last known
position on a low-confidence frame instead of jumping around.

Planner v3 §8 (Stage 35): runs the REAL pipeline — MapBundle +
LocalizationTracker (visual + semantic + temporal + graph evidence, state
machine, confidence calibration) — and the Stage 24 runtime gate
(src/runtime/gate.py::should_process) skips redundant frames before the
expensive encoder/detector/VLM path (closing Finding D: the gate had zero
production callers).

This is snapshot-at-an-interval, not true frame-by-frame video tracking —
simple by design, same spirit as the rest of the project.

Run:
    python live_navigate.py --destination "Room 101"

Press 'q' in the video window to quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
from PIL import Image

from evaluate_suite import find_bundle_dir
from src.embeddings.encoder import get_encoder
from src.extraction.frames import _descriptor  # Stage 03 descriptor — same reuse the gate itself does
from src.localization.tracker import LocalizationTracker
from src.mapping.map_artifact import MapBundle
from src.navigate import name_to_id_map
from src.runtime.gate import should_process
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("live_navigate")


def get_perception_models(config: dict) -> tuple:
    """One detector + tagger pair, built once (model loads are expensive)."""
    detector = tagger = None
    perception = config.get("perception", {})
    if perception.get("detector_enabled", True):
        from src.perception.detector import get_detector

        detector = get_detector(config)
    if perception.get("vlm_enabled", True):
        from src.perception.scene_tagger import get_scene_tagger

        tagger = get_scene_tagger(config)
    return detector, tagger


def query_evidence(image: Image.Image, config: dict, detector, tagger) -> tuple[dict | None, list | None]:
    """Detector + VLM on one frame; failures degrade to None (Rule 4)."""
    objects = None
    if detector is not None and detector.name != "stub":  # property, unlike tagger.name()
        try:
            objects = [o.to_dict() for o in detector.detect(image)]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Detector failed on frame: {exc}")
    tags = None
    if tagger is not None and tagger.name() != "stub":
        try:
            tags = tagger.tag(image, objects).to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"VLM failed on frame: {exc}")
    return tags, objects


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam localization + re-routing.")
    parser.add_argument("--destination", required=True, help="Name of the destination place")
    parser.add_argument("--camera-index", type=int, default=None, help="Overrides config.yaml live.camera_index")
    parser.add_argument(
        "--interval-seconds", type=float, default=None,
        help="Overrides the runtime gate's forced re-run interval (runtime.max_stale_seconds)",
    )
    parser.add_argument("--speak", action="store_true", help="Speak updated directions aloud (needs pyttsx3 + audio)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    bundle = MapBundle.load(find_bundle_dir(config))
    name_to_id = name_to_id_map(bundle.place_names)

    if args.destination not in name_to_id:
        raise SystemExit(f"Unknown destination: {args.destination!r}. Known places: {list(name_to_id)}")

    if args.interval_seconds is not None:
        config.setdefault("runtime", {})["max_stale_seconds"] = args.interval_seconds

    camera_index = args.camera_index if args.camera_index is not None else config["live"]["camera_index"]

    tracker = LocalizationTracker(bundle, config)
    tracker.set_destination(name_to_id[args.destination])

    detector, tagger = get_perception_models(config)
    encoder = get_encoder(config)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Check that a webcam is "
            "connected and not in use by another application."
        )

    logger.info(f"Live navigation started — heading to {args.destination!r}. Press 'q' to quit.")
    last_descriptor = None          # Stage 03 descriptor of the last processed frame
    last_processed_ts = None        # time.monotonic() of that processing

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Could not read from camera — stopping.")
                break

            cv2.imshow("SimpleNav — live camera (press q to quit)", frame)

            # Stage 24 runtime gate: skip redundant frames BEFORE the
            # expensive encoder/detector/VLM path (planner v3 §8 / Finding D)
            decision = should_process(frame, last_descriptor, last_processed_ts, config)
            logger.debug(f"gate: {decision.reason}")
            if not decision.run_expensive:
                continue

            last_descriptor = _descriptor(frame)
            last_processed_ts = time.monotonic()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            embedding = encoder.encode(image)
            tags, objects = query_evidence(image, config, detector, tagger)
            status = tracker.process_frame(embedding, tags, objects)

            if status["low_confidence"]:
                logger.info(f"Location unclear (confidence={status['confidence']:.2f}) — holding position")
            elif status["changed"]:
                logger.info(f"Now at: {status['place_name']} (state={status['state']})")
                if status["arrived"]:
                    logger.info("You have arrived at your destination!")
                    if args.speak:
                        from src.speak import speak

                        speak("You have arrived at your destination.")
                elif status["directions"]:
                    logger.info("Updated route: " + status["directions"])
                    if args.speak:
                        from src.speak import speak

                        speak(f"New route: {status['directions'].replace('->', 'then')}")
                else:
                    logger.warning("No known path from here to the destination.")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
