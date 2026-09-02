"""SimpleNav demo app.

Run with:  streamlit run app.py

Two modes:
  - Single Photo: upload one image, get your predicted location + a route
  - Live Mode: take repeated snapshots as you move; the route updates
    automatically whenever your detected location changes (re-routing)

Planner v3 §8 (Stage 35): the app runs the REAL pipeline — MapBundle +
LocalizationTracker (visual + semantic + temporal + graph evidence, Bayes
filter, state machine, confidence calibration) — not the legacy
PlaceIndex/LiveTracker stack. Uploaded/live photos are tagged by the same
detector + VLM the mapping pass uses, so the semantic term is real evidence
(Rule 7: the product a user runs is the product that was tested).

Note on "live": browser camera input in Streamlit gives one photo per click,
not a continuous video stream. Live Mode here is snapshot-driven — take a
new photo each time you've moved, and the app re-localizes and re-routes.
For genuinely continuous webcam tracking on your own machine, use
`python live_navigate.py --destination "..."` instead (see README).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.embeddings.encoder import get_encoder
from src.floor_plan import draw_route_on_floor_plan, is_available as floor_plan_available, load_coords
from src.localization.tracker import LocalizationTracker
from src.mapping.map_artifact import MapBundle
from src.navigate import name_to_id_map
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger("app")

rear_camera = components.declare_component(
    "rear_camera",
    path=str(Path(__file__).parent / "components" / "rear_camera"),
)


def find_bundle_dir(config: dict) -> Path:
    """Newest map bundle under config.paths.map_dir (same convention as
    evaluate_suite.py)."""
    map_dir = resolve_path(config["paths"]["map_dir"])
    candidates = sorted(
        (p for p in map_dir.iterdir() if (p / "manifest.json").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No map bundle under {map_dir} — run build_map.py first")
    return candidates[-1]


def load_bundle(config: dict) -> MapBundle:
    """Versioned map artifact -> the real localization bundle. Small enough
    (301 obs) to load per interaction — no caching needed."""
    return MapBundle.load(find_bundle_dir(config))


def get_perception_models(config: dict) -> tuple:
    """One detector + tagger pair, built once (model loads are expensive —
    LFM2.5-VL takes seconds to load; per-frame calls reuse the instances)."""
    if "perception_models" not in st.session_state:
        detector = tagger = None
        perception = config.get("perception", {})
        if perception.get("detector_enabled", True):
            from src.perception.detector import get_detector

            detector = get_detector(config)
        if perception.get("vlm_enabled", True):
            from src.perception.scene_tagger import get_scene_tagger

            tagger = get_scene_tagger(config)
        st.session_state.perception_models = (detector, tagger)
    return st.session_state.perception_models


def query_evidence(image: Image.Image, config: dict) -> tuple[dict | None, list | None]:
    """Detector + VLM on one query image (same models as the mapping pass).
    Failures degrade to None -> the semantic term goes neutral 0.5 (Rule 4);
    they never crash the demo."""
    detector, tagger = get_perception_models(config)
    objects = None
    if detector is not None and detector.name != "stub":  # property, unlike tagger.name()
        try:
            objects = [o.to_dict() for o in detector.detect(image)]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Detector failed on query image: {exc}")
    tags = None
    if tagger is not None and tagger.name() != "stub":
        try:
            tags = tagger.tag(image, objects).to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"VLM failed on query image: {exc}")
    return tags, objects


def encode_image(image: Image.Image, config: dict):
    """The configured encoder (planner v3 §4 — same one that built the map)."""
    return get_encoder(config).encode(image)


def draw_graph_view(graph: nx.Graph, place_names: dict[str, str], route: list[int] | None):
    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(graph, seed=42)
    labels = {n: place_names.get(str(n), f"Place_{n}") for n in graph.nodes}

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="lightgray")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#b7d4f0", node_size=900)

    if route and len(route) > 1:
        route_edges = list(zip(route, route[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=route_edges, ax=ax, edge_color="#d95f02", width=3)
        nx.draw_networkx_nodes(graph, pos, nodelist=route, ax=ax, node_color="#fdae61", node_size=950)

    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8)
    ax.axis("off")
    return fig


def draw_map_view(config, graph, place_names, route):
    """Floor plan overlay if one's been set up, otherwise the abstract graph.
    The floor-plan keys are optional in config (planner v2 config rewrite
    dropped them) — their absence must fall back to the graph view, not
    crash the app."""
    paths_cfg = config.get("paths", {})
    floor_plan_image = resolve_path(paths_cfg["floor_plan_image"]) if "floor_plan_image" in paths_cfg else None
    floor_plan_coords_file = (
        resolve_path(paths_cfg["floor_plan_coords_file"])
        if "floor_plan_coords_file" in paths_cfg else None
    )
    if (floor_plan_image is not None and floor_plan_coords_file is not None
            and floor_plan_available(floor_plan_image, floor_plan_coords_file)):
        coords = load_coords(floor_plan_coords_file)
        return draw_route_on_floor_plan(floor_plan_image, coords, place_names, route)
    return draw_graph_view(graph, place_names, route)


def show_term_breakdown(status: dict) -> None:
    """Per-candidate visual/semantic/temporal/graph terms (planner v3 §8 —
    the data always existed on each scored candidate; now it's visible)."""
    rows = status.get("term_breakdown") or []
    if not rows:
        return
    st.caption("Top-3 candidates — evidence breakdown")
    st.dataframe(
        [
            {
                "Place": r["place_name"],
                "total": r["total"],
                "visual": r["visual_term"],
                "semantic": r["semantic_term"],
                "temporal": r["temporal_term"],
                "graph": r["graph_term"],
            }
            for r in rows
        ],
        use_container_width=True,
    )


def show_tracker_state(status: dict) -> None:
    """The state machine + calibrated confidence, visible (planner v3 §8)."""
    st.caption(
        f"Tracker state: **{status['state']}** "
        f"· confidence: **{status['confidence_level']}** "
        f"· unknown mass: {status['unknown_mass']:.2f}"
    )


def speak_button(text: str, key: str) -> None:
    if st.button("🔊 Speak directions", key=key):
        from src.speak import speak

        ok = speak(text.replace("->", "then"))
        if not ok:
            st.info("Text-to-speech isn't available in this environment — see the terminal log for details.")


def decode_camera_photo(photo_data) -> Image.Image | None:
    """Decode camera component values from Streamlit's JSON or string form."""
    if isinstance(photo_data, dict):
        photo_data = photo_data.get("value")
        if isinstance(photo_data, dict):
            photo_data = photo_data.get("value")

    if not isinstance(photo_data, str) or "," not in photo_data:
        return None

    _, encoded_image = photo_data.split(",", 1)
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    except Exception:
        return None


def single_photo_tab(config, bundle):
    col1, col2 = st.columns(2)
    current_place_name = None
    route = None

    with col1:
        st.subheader("1. Where am I?")
        uploaded = st.file_uploader("Upload a photo of your current view", type=["jpg", "jpeg", "png"])

        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Query image", width=300)
            with st.spinner("Localizing with visual + semantic + temporal + graph evidence..."):
                embedding = encode_image(image, config)
                tags, objects = query_evidence(image, config)
                tracker = LocalizationTracker(bundle, config)
                status = tracker.process_frame(embedding, tags, objects)

            # keep the raw evidence so "Get directions" can re-route without
            # re-encoding or re-tagging the photo
            st.session_state.single_status = status
            st.session_state.single_embedding = embedding
            st.session_state.single_tags = tags
            st.session_state.single_objects = objects
            best_name = status["place_name"]
            best_score = status["confidence"]

            # The tracker's calibrated confidence_level is the designed
            # recognition gate (the posterior mass alone is a sub-probability
            # on a different scale than any similarity threshold — planner v3
            # §8; legacy navigation.confidence_threshold does not apply here)
            if status["confidence_level"] == "UNKNOWN" or status["state"] in ("LOST", "REACQUIRING"):
                st.warning(
                    f"Location not confidently recognized "
                    f"(state={status['state']}, level={status['confidence_level']})."
                )
            else:
                current_place_name = best_name
                st.success(
                    f"Predicted location: **{best_name}** "
                    f"(posterior {best_score:.2f}, state {status['state']})."
                )

            show_tracker_state(status)
            show_term_breakdown(status)

    with col2:
        st.subheader("2. Where do I want to go?")
        all_names = [bundle.place_names[str(pid)] for pid in sorted(bundle.graph.nodes)]
        destination_name = st.selectbox("Destination", all_names, key="single_dest")

        if st.button("Get directions", key="single_go"):
            status = st.session_state.get("single_status")
            if status is None or status.get("place_id") is None:
                st.write("Upload an image on the left first.")
            else:
                tracker = LocalizationTracker(bundle, config)
                tracker.set_destination(name_to_id_map(bundle.place_names)[destination_name])
                routed = tracker.process_frame(
                    st.session_state.get("single_embedding"),
                    st.session_state.get("single_tags"),
                    st.session_state.get("single_objects"),
                )
                route = routed["route"]
                if route is None:
                    st.warning("No known path between these places yet.")
                else:
                    directions = routed["directions"]
                    st.info("Route: " + directions)
                    speak_button(directions, key="single_speak")

    st.pyplot(draw_map_view(config, bundle.graph, bundle.place_names, route))


def live_mode_tab(config, bundle):
    st.caption(
        "Take a new snapshot each time you've moved. The app re-localizes and, "
        "if you've reached a new place, automatically recomputes the route. "
        "(Browser cameras give one photo per click, not continuous video — "
        "for real continuous tracking on your own machine, use "
        "`python live_navigate.py --destination \"...\"` instead.)"
    )
    st.info("Live Mode uses the rear camera. Allow camera access when your browser asks.")

    all_names = [bundle.place_names[str(pid)] for pid in sorted(bundle.graph.nodes)]
    destination_name = st.selectbox("Destination", all_names, key="live_dest")

    needs_new_tracker = (
        "live_tracker" not in st.session_state or st.session_state.get("live_dest_name") != destination_name
    )
    if needs_new_tracker:
        tracker = LocalizationTracker(bundle, config)
        tracker.set_destination(name_to_id_map(bundle.place_names)[destination_name])
        st.session_state.live_tracker = tracker
        st.session_state.live_dest_name = destination_name

    photo_data = rear_camera(key="live_rear_camera", default=None)

    route = None
    if photo_data:
        image = decode_camera_photo(photo_data)
        if image is None:
            st.error("The camera returned an invalid image. Please reload the page and try again.")
        else:
            embedding = encode_image(image, config)
            tags, objects = query_evidence(image, config)
            status = st.session_state.live_tracker.process_frame(embedding, tags, objects)
            route = status["route"]

            show_tracker_state(status)
            if status["low_confidence"]:
                st.warning(f"Location unclear (confidence {status['confidence']:.2f}) — holding last known position.")
            else:
                st.success(f"You are at: **{status['place_name']}** (confidence {status['confidence']:.2f})")

            if status["arrived"]:
                st.balloons()
                st.success("You have arrived at your destination!")
            elif status["directions"]:
                label = "Position updated — new route:" if status["changed"] else "Route:"
                st.info(f"{label} {status['directions']}")
                speak_button(status["directions"], key="live_speak")
            elif status["place_id"] is not None:
                st.warning("No known path to the destination from here yet.")

    st.pyplot(draw_map_view(config, bundle.graph, bundle.place_names, route))


def main() -> None:
    st.set_page_config(page_title="SimpleNav Demo", layout="wide")
    st.title("SimpleNav — Indoor Navigation Demo")
    st.caption(
        "Camera-only localization combining appearance + semantic landmarks + "
        "temporal continuity + graph topology, over a place graph built "
        "autonomously from a walkthrough video."
    )

    try:
        config = load_config()
        bundle = load_bundle(config)
    except (FileNotFoundError, KeyError) as e:
        st.error(
            "Map data not found. Run the pipeline first: extract_frames.py -> "
            "embed_frames.py -> build_map.py.\n\n"
            f"Details: {e}"
        )
        return

    tab1, tab2 = st.tabs(["📷 Single Photo", "🎥 Live Mode"])
    with tab1:
        single_photo_tab(config, bundle)
    with tab2:
        live_mode_tab(config, bundle)


if __name__ == "__main__":
    main()
