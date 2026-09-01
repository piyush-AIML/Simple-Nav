# SimpleNav — Vision-Based Indoor Navigation (Prototype)

A smartphone-camera-only indoor navigation prototype: record a walkthrough
video of a building once, and the system can then tell you which mapped
area a new photo (or live camera feed) was taken in, and give you —
and automatically update — directions to another area. No GPS, no BLE
beacons, no WiFi fingerprinting, no manual floor-plan digitization.

## Problem statement

Indoor spaces (colleges, hospitals, offices) generally don't have reliable
GPS coverage, and infrastructure-based indoor positioning (BLE beacons,
WiFi RTT) is expensive to install and maintain. This project explores
whether a purely vision-based approach — using only a phone/webcam camera
and publicly available pretrained models — can support basic "where am I" /
"how do I get there" functionality within a single building.

## Architecture

```
                     OFFLINE: build the map (run once per building)
 ┌────────────┐   ┌───────────────┐   ┌───────────────────┐   ┌───────────────┐
 │  Walkthrough│──▶│ extract_frames │──▶│  embed_frames      │──▶│ build_places  │
 │  video      │   │ (every Nth     │   │  (pretrained       │   │ (K-Means, k   │
 │             │   │  frame)        │   │  ResNet18 features)│   │  by silhouette,│
 └────────────┘   └───────────────┘   └───────────────────┘   │  multiple      │
                                                                 │  exemplars per │
                                                                 │  place)        │
                                                                 └───────┬───────┘
                                                                         │
                                                                 ┌───────▼───────┐
                                                                 │ build_graph   │
                                                                 │ (place        │
                                                                 │  transitions, │
                                                                 │  weighted by  │
                                                                 │  count)       │
                                                                 └───────┬───────┘
                                                                         │
                                        map = exemplar vectors + place names + graph
                                                                         │
      RUNTIME: use the map                                              │
 ┌────────────┐   ┌───────────────┐   ┌───────────────────┐             │
 │ Photo / live│──▶│ embed (same   │──▶│ localize.py:        │◀────────────┘
 │ camera frame│   │ ResNet18)     │   │ FAISS flat vector    │
 └────────────┘   └───────────────┘   │ retrieval, confidence │
                                       │ threshold gate        │
                                       └──────────┬────────────┘
                                                  │
                                       ┌──────────▼────────────┐
                                       │ navigate.py:           │
                                       │ weighted shortest path │
                                       │ → turn-by-turn names   │
                                       └──────────┬─────────────┘
                                                  │
                          ┌───────────────────────┴───────────────────────┐
                          │                                               │
                 ┌────────▼────────┐                            ┌─────────▼────────┐
                 │ live_tracker.py  │                            │ app.py (Streamlit)│
                 │ re-routes when   │◀── used by both ──────────▶│ single photo +    │
                 │ your place       │                            │ live camera tabs, │
                 │ changes          │                            │ floor plan, TTS   │
                 └──────────────────┘                            └───────────────────┘
```

## Tech stack

| Component | Choice |
|---|---|
| Frame extraction | OpenCV, fixed sampling interval |
| Feature extraction | Pretrained ResNet18 (ImageNet weights), classification head removed |
| Place discovery | K-Means, k chosen by sweeping a range and picking the best silhouette score |
| Place representation | Multiple exemplar vectors per place (small secondary KMeans within each place), not one blurry average |
| Map structure | NetworkX graph, edges weighted by observed transition counts |
| Localization | FAISS `IndexFlatIP` (flat vector retrieval) + confidence threshold + majority-vote smoothing |
| Routing | NetworkX shortest path, weighted to favor strongly-observed connections |
| Live tracking | Snapshot-based camera input with automatic re-routing on position change |
| Demo UI | Streamlit (single-photo + live camera tabs) |
| Optional extras | Floor-plan route overlay, text-to-speech directions |
| Config | Single `config.yaml`, no hardcoded paths |

## Setup

```bash
pip install -r requirements.txt
```

Put a walkthrough video at `data/raw_video.mp4` (or update `config.yaml`).
The first run needs internet access once, to download the pretrained
ResNet18 weights (standard PyTorch model hub — should work on any normal
connection).

## Building the map

Run these in order:

```bash
python -m src.extract_frames
python -m src.embed_frames
python -m src.build_places
python -m src.build_graph
```

Then name your places. Two ways:

- **Interactive (recommended):** `python -m src.label_places` — shows a few
  representative photos per discovered place and lets you type a name.
- **Manual:** open `data/map/place_names.json` and rename the auto-generated
  `"Place_0"`, `"Place_1"`, ... entries yourself.

Re-running `build_places.py` won't overwrite a `place_names.json` that
already has custom names.

## Using it

**Localize a single photo:**
```bash
python -m src.localize path/to/photo.jpg
```
If the best match is too weak (below `navigation.confidence_threshold` in
`config.yaml`), it reports "not confidently recognized" instead of forcing
a guess.

**Get directions:**
```bash
python -m src.navigate --source "Lobby" --destination "Room 101"
```
Routes by default favor connections the pipeline observed more often
(`--unweighted` for plain fewest-hops). Add `--speak` to hear the directions
aloud (needs `pyttsx3` + a working system audio/TTS backend).

**Continuous live navigation from your webcam**, with automatic re-routing
as you move:
```bash
python live_navigate.py --destination "Room 101"
```
Press `q` in the video window to quit. This periodically grabs a frame
(interval configurable in `config.yaml` under `live:`), localizes it, and
whenever your tracked place changes, recomputes and reprints the route.
Low-confidence frames hold your last known position instead of causing the
route to jump around.

The desktop live mode defaults to camera index `1`, which is commonly the rear
camera when both front and rear cameras are available. Use
`--camera-index 0` (or another index) if your system assigns cameras
differently. In the Streamlit browser Live Mode, use the camera-switch button
in the preview to select the rear camera.

**Interactive demo (browser):**
```bash
streamlit run app.py
```
Has two tabs:
- **Single Photo** — upload one image, see your predicted place (with
  top-3 candidate scores), pick a destination, get a route.
- **Live Mode** — take repeated snapshots as you move; the app re-localizes
  and automatically recomputes the route whenever your detected place
  changes. (Browser cameras give one photo per click, not continuous video —
  for genuinely continuous tracking, use `live_navigate.py` above instead.)

The Streamlit Live Mode requests the rear camera directly. Grant camera
permission when prompted; browser camera access requires `localhost` or HTTPS.

## Optional: floor-plan route overlay

By default, routes are drawn on an abstract graph layout. If you'd rather
see the route on your actual floor plan:

1. Save a floor plan image to `data/map/floor_plan.png`
2. Run `python -m src.set_floor_plan_coords` and click once on the image
   for each place, in the order it lists
3. `app.py` will automatically switch to drawing routes on the floor plan
   instead of the abstract graph

This is entirely optional — skip it and everything still works with the
graph view.

## Evaluating accuracy

Hand-label a small set of test frames in `data/test_labels.json`:

```json
{ "frame_00010.jpg": "Lobby", "frame_00050.jpg": "Room 101" }
```

Then run:

```bash
python evaluate.py
```

Reports accuracy, a per-place classification report, and a confusion
matrix — for both this project's CNN-based retrieval AND a naive
color-histogram baseline, so there's a real point of comparison rather than
one number in isolation. For a meaningful number, use frames that weren't
part of building the map (a short second walkthrough, or photos taken on a
different day).

## Testing

```bash
pytest tests/
```

Covers routing (shortest path, unreachable-node handling, direction
formatting) and the live-tracking/re-routing state machine (place-change
detection, arrival detection, holding position on low-confidence frames) —
all with small hand-built toy graphs and synthetic vectors, no ML
dependencies or camera required.

## Limitations

- Single-pass K-Means clustering — no correction if a cluster spans two
  visually similar but actually-different rooms.
- Live tracking is snapshot/interval-based, not true frame-by-frame video
  tracking.
- The confidence threshold is a single fixed number, not a calibrated
  probability — tune it against your own building's data.
- No live camera/AR pose estimation — position is "which mapped place do
  you most resemble right now," not a precise coordinate.

## Future work

- Explore temporal continuity when building places, instead of clustering
  frames purely by appearance (revisits of a similar-looking spot later in
  the video currently get lumped into the same cluster).
- Replace single-frame nearest-neighbor localization with a probabilistic
  filter that carries belief over time for more robust tracking.
- Add a lightweight way to detect and correct bad place merges/splits
  automatically instead of relying on manual review.
- Investigate on-device deployment (model quantization, mobile inference)
  for a true continuous live-camera experience on a phone.
