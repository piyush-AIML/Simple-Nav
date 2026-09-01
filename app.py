"""SimpleNav demo app.

Run with:  streamlit run app.py

Two modes:
  - Single Photo: upload one image, get your predicted location + a route
  - Live Mode: take repeated snapshots as you move; the route updates
    automatically whenever your detected location changes (re-routing)

Note on "live": browser camera input in Streamlit gives one photo per click,
not a continuous video stream. Live Mode here is snapshot-driven — take a
new photo each time you've moved, and the app re-localizes and re-routes.
For genuinely continuous webcam tracking on your own machine, use
`python live_navigate.py --destination "..."` instead (see README).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
import io
from pathlib import Path

from src.embedder import embed_image
from src.floor_plan import draw_route_on_floor_plan, is_available as floor_plan_available, load_coords
from src.live_tracker import LiveTracker
from src.localize import PlaceIndex
from src.navigate import format_directions, get_route, load_graph, load_place_names, name_to_id_map
from src.utils import load_config, resolve_path


rear_camera = components.declare_component(
    "rear_camera",
    path=str(Path(__file__).parent / "components" / "rear_camera"),
)


@st.cache_resource
def load_map():
    config = load_config()
    place_index = PlaceIndex.load(
        resolve_path(config["paths"]["place_exemplars_file"]),
        resolve_path(config["paths"]["exemplar_place_ids_file"]),
        resolve_path(config["paths"]["place_names_file"]),
    )
    graph = load_graph(resolve_path(config["paths"]["graph_file"]))
    place_names = load_place_names(resolve_path(config["paths"]["place_names_file"]))
    return config, place_index, graph, place_names


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
    """Floor plan overlay if one's been set up, otherwise the abstract graph."""
    floor_plan_image = resolve_path(config["paths"]["floor_plan_image"])
    floor_plan_coords_file = resolve_path(config["paths"]["floor_plan_coords_file"])
    if floor_plan_available(floor_plan_image, floor_plan_coords_file):
        coords = load_coords(floor_plan_coords_file)
        return draw_route_on_floor_plan(floor_plan_image, coords, place_names, route)
    return draw_graph_view(graph, place_names, route)


def show_top_matches(matches: list[tuple[int, str, float]]) -> None:
    if not matches:
        return
    st.caption("Top matches")
    st.bar_chart({name: score for _, name, score in matches})


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


def single_photo_tab(config, place_index, graph, place_names):
    threshold = config["navigation"]["confidence_threshold"]
    use_weighted = config["navigation"].get("use_weighted_routing", True)

    col1, col2 = st.columns(2)
    current_place_id = None
    current_place_name = None
    route = None

    with col1:
        st.subheader("1. Where am I?")
        uploaded = st.file_uploader("Upload a photo of your current view", type=["jpg", "jpeg", "png"])

        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Query image", width=300)
            embedding = embed_image(image)
            matches = place_index.query(embedding, top_k=3)
            best_id, best_name, best_score = matches[0]

            if best_score < threshold:
                st.warning(f"Location not confidently recognized (best similarity {best_score:.2f}).")
            else:
                current_place_id, current_place_name = best_id, best_name
                st.success(f"Predicted location: **{best_name}** (similarity: {best_score:.2f})")

            show_top_matches(matches)

    with col2:
        st.subheader("2. Where do I want to go?")
        all_names = [place_names[str(pid)] for pid in sorted(graph.nodes)]
        destination_name = st.selectbox("Destination", all_names, key="single_dest")

        if current_place_name is not None and st.button("Get directions", key="single_go"):
            name_to_id = name_to_id_map(place_names)
            route = get_route(graph, current_place_id, name_to_id[destination_name], use_weighted)
            if route is None:
                st.warning("No known path between these places yet.")
            else:
                directions = format_directions(route, place_names)
                st.info("Route: " + directions)
                speak_button(directions, key="single_speak")
        elif current_place_name is None:
            st.write("Upload an image on the left first.")

    st.pyplot(draw_map_view(config, graph, place_names, route))


def live_mode_tab(config, place_index, graph, place_names):
    st.caption(
        "Take a new snapshot each time you've moved. The app re-localizes and, "
        "if you've reached a new place, automatically recomputes the route. "
        "(Browser cameras give one photo per click, not continuous video — "
        "for real continuous tracking on your own machine, use "
        "`python live_navigate.py --destination \"...\"` instead.)"
    )
    st.info("Live Mode uses the rear camera. Allow camera access when your browser asks.")

    all_names = [place_names[str(pid)] for pid in sorted(graph.nodes)]
    destination_name = st.selectbox("Destination", all_names, key="live_dest")
    name_to_id = name_to_id_map(place_names)

    needs_new_tracker = (
        "live_tracker" not in st.session_state or st.session_state.get("live_dest_name") != destination_name
    )
    if needs_new_tracker:
        tracker = LiveTracker(
            place_index=place_index,
            graph=graph,
            place_names=place_names,
            confidence_threshold=config["navigation"]["confidence_threshold"],
            smoothing_window=config["navigation"]["smoothing_window"],
            use_weighted_routing=config["navigation"].get("use_weighted_routing", True),
        )
        tracker.set_destination(name_to_id[destination_name])
        st.session_state.live_tracker = tracker
        st.session_state.live_dest_name = destination_name

    photo_data = rear_camera(key="live_rear_camera", default=None)

    route = None
    if photo_data:
        image = decode_camera_photo(photo_data)
        if image is None:
            st.error("The camera returned an invalid image. Please reload the page and try again.")
        else:
            embedding = embed_image(image)
            status = st.session_state.live_tracker.process_frame(embedding)
            route = status["route"]

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

    st.pyplot(draw_map_view(config, graph, place_names, route))


def main() -> None:
    st.set_page_config(page_title="SimpleNav Demo", layout="wide")
    st.title("SimpleNav — Indoor Navigation Demo")
    st.caption(
        "Vector-retrieval localization (FAISS flat index) + shortest-path routing "
        "over a place graph built from a walkthrough video."
    )

    try:
        config, place_index, graph, place_names = load_map()
    except FileNotFoundError as e:
        st.error(
            "Map data not found. Run the pipeline first: extract_frames.py -> "
            "embed_frames.py -> build_places.py -> build_graph.py.\n\n"
            f"Details: {e}"
        )
        return

    tab1, tab2 = st.tabs(["📷 Single Photo", "🎥 Live Mode"])
    with tab1:
        single_photo_tab(config, place_index, graph, place_names)
    with tab2:
        live_mode_tab(config, place_index, graph, place_names)


if __name__ == "__main__":
    main()
