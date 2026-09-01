"""Continuous camera-based localization with automatic re-routing.

Opens your webcam, periodically grabs a frame, localizes it against the map,
and whenever your tracked place changes, recomputes and reprints directions
to your chosen destination. Uses majority-vote smoothing to avoid flicker
between visually similar places, and holds your last known position on a
low-confidence frame instead of jumping around.

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

from src.embedder import embed_image
from src.live_tracker import LiveTracker
from src.localize import PlaceIndex
from src.navigate import load_graph, load_place_names, name_to_id_map
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("live_navigate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam localization + re-routing.")
    parser.add_argument("--destination", required=True, help="Name of the destination place")
    parser.add_argument("--camera-index", type=int, default=None, help="Overrides config.yaml live.camera_index")
    parser.add_argument(
        "--interval-seconds", type=float, default=None, help="Overrides config.yaml live.capture_interval_seconds"
    )
    parser.add_argument("--speak", action="store_true", help="Speak updated directions aloud (needs pyttsx3 + audio)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    exemplars_file = resolve_path(config["paths"]["place_exemplars_file"])
    exemplar_ids_file = resolve_path(config["paths"]["exemplar_place_ids_file"])
    names_file = resolve_path(config["paths"]["place_names_file"])
    graph_file = resolve_path(config["paths"]["graph_file"])

    camera_index = args.camera_index if args.camera_index is not None else config["live"]["camera_index"]
    interval = args.interval_seconds if args.interval_seconds is not None else config["live"]["capture_interval_seconds"]

    place_index = PlaceIndex.load(exemplars_file, exemplar_ids_file, names_file)
    graph = load_graph(graph_file)
    place_names = load_place_names(names_file)
    name_to_id = name_to_id_map(place_names)

    if args.destination not in name_to_id:
        raise SystemExit(f"Unknown destination: {args.destination!r}. Known places: {list(name_to_id)}")

    tracker = LiveTracker(
        place_index=place_index,
        graph=graph,
        place_names=place_names,
        confidence_threshold=config["navigation"]["confidence_threshold"],
        smoothing_window=config["navigation"]["smoothing_window"],
        use_weighted_routing=config["navigation"].get("use_weighted_routing", True),
    )
    tracker.set_destination(name_to_id[args.destination])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Check that a webcam is "
            "connected and not in use by another application."
        )

    logger.info(f"Live navigation started — heading to {args.destination!r}. Press 'q' to quit.")
    last_capture_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Could not read from camera — stopping.")
                break

            cv2.imshow("SimpleNav — live camera (press q to quit)", frame)

            now = time.time()
            if now - last_capture_time >= interval:
                last_capture_time = now
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embedding = embed_image(rgb_frame)
                status = tracker.process_frame(embedding)

                if status["low_confidence"]:
                    logger.info(f"Location unclear (confidence={status['confidence']:.2f}) — holding position")
                elif status["changed"]:
                    logger.info(f"Now at: {status['place_name']}")
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
