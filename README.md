# SimpleNav — Vision-Based Indoor Navigation (Prototype)

A smartphone-camera-only indoor navigation prototype: record a walkthrough
video of a building once, and the system autonomously builds a semantic
connectivity graph from it, then tells you which mapped area a new photo (or
live camera feed) was taken in — and gives you spoken turn-by-turn
directions to another area. No GPS, no BLE beacons, no WiFi fingerprinting,
no manual floor-plan digitization.

## Problem statement

Indoor spaces (colleges, hospitals, offices) generally don't have reliable
GPS coverage, and infrastructure-based indoor positioning (BLE beacons,
WiFi RTT) is expensive to install and maintain. This project explores
whether a purely vision-based approach — using only a phone/webcam camera
and publicly available pretrained models — can support basic "where am I" /
"how do I get there" functionality within a single building.

## Architecture

Two pipelines. The offline one runs once per building; the runtime one runs
continuously against the camera.

```
                       OFFLINE MAPPING                        RUNTIME LOCALIZATION
                     walkthrough video                          camera stream
                            │                                        │
                            ▼                                        ▼
                     smart sampling                           runtime gate (§24)
               (quality + novelty + sustained                 (cheap novelty + forced
                     change retention)                        interval; skip redundant
                            │                                  frames, reuse state)
                            ▼                                        │
                observation creation                                  ▼
                            │                                ┌───────┴───────┐
               ┌────────────┴────────────┐                   ▼               ▼
               ▼                         ▼             YOLO26n           embedding
          YOLO26n                  LFM2.5-VL-450M     (objects)        (ResNet18)
         (objects)                  (scene tags)           │               │
               │                         │                 ▼               ▼
               └────────────┬────────────┘          semantic scoring     FAISS top-K
                            ▼                          (§23: scene +       │
                     visual embedding              landmarks + objects)   ▼
                            │                                │       candidate scoring
                            ▼                                │               │
                   temporal segmentation                    │        +──────┴───────+
                     (adaptive threshold)                   │        ▼              ▼
                            │                                │  temporal state    graph
                            ▼                                │  model (Bayes)  constraints
                     place formation +                       │        │              │
                     reconciliation,                         └────────┴──────┬───────┘
                     transitions                                       ▼
                            │                                   stable location
                            ▼                                           │
                    graph + validation                                 ▼
                            │                                       destination
                            ▼                                           │
                     versioned map                                    ▼
                     (college_env_v1)                            shortest path
                                                                        │
                                                                        ▼
                                                              simple instruction + TTS
```

Everything is evidence, nothing is ground truth: the detector, VLM, and
retrieval propose; the Bayesian state estimator + graph topology decide.
Unknown beats confidently wrong — residual probability mass is explicit.

## Tech stack

| Component | Choice |
|---|---|
| Frame sampling | OpenCV smart sampler: quality gate (blur/dark) + mean-subtracted 32×32 descriptor novelty + sustained-change retention |
| Object detection | YOLO26n (`ultralytics`, COCO classes) — evidence for object-overlap scoring |
| Scene/landmark tagging | `LiquidAI/LFM2.5-VL-450M` at bf16 via transformers (fixed JSON schema, validated; fallback chain LFM2 → SmolVLM2-500M → deterministic stub) |
| Feature extraction | Pretrained ResNet18 (ImageNet weights), classification head removed, behind a `VisualEncoder` registry |
| Temporal segmentation | Adaptive threshold (mean + 2.5σ of embedding change scores) |
| Place formation | One place per temporal segment; ≤3 exemplars via temporal diversity; multi-signal reconciliation (visual + landmarks + scene + context) |
| Map structure | NetworkX graph from debounced transitions, junction detection, validation report (connectivity/isolated/weak-edge/duplicates) |
| Map artifact | Versioned bundle under `data/map/<map_id>/`: manifest, places, graph, exemplars, FAISS index |
| Localization | FAISS top-K retrieval → candidate scoring (visual + semantic + temporal + graph terms) → Bayes state estimator (likelihood = visual+semantic only; graph/temporal enter via the transition prior) |
| State machine | TRACKING / UNCERTAIN / LOST / REACQUIRING / ARRIVED, K-consecutive stabilization, LOCAL/GLOBAL reacquisition, HIGH/MEDIUM/LOW/UNKNOWN confidence |
| Routing | NetworkX shortest path, weighted to favor strongly-observed connections |
| Live loop | Runtime gate reusing the Stage 03 novelty descriptor + forced interval (`runtime.max_stale_seconds`) |
| Demo UI | Streamlit (single-photo + live camera tabs) |
| Optional extras | Floor-plan route overlay, text-to-speech directions (`pyttsx3`) |
| Config | Single `config.yaml`, no hardcoded paths |

## Setup

```bash
pip install -r requirements.txt
```

The first run needs internet access once, to download the pretrained model
weights (ResNet18 from the PyTorch hub; YOLO26n and LFM2.5-VL-450M from
Hugging Face / Ultralytics). The VLM runs at bf16 and fits a 6 GB GPU
without quantization — `bitsandbytes` is not required (and must NOT be used
to quantize LFM2 to 4-bit; it degrades sharply).

## Building the map

Put a walkthrough video at `data/College_env.mp4` (or update
`config.yaml` → `paths.video`), then:

```bash
python -m src.extract_frames      # smart frame sampling
python -m src.embed_frames        # embeddings + YOLO26n objects + VLM scene tags
python -m src.mapping.build_map   # segmentation → places → graph → versioned map
```

Then name your places by editing `data/map/<map_id>/place_names.json`.

The legacy Milestone-A pipeline (`scripts/benchmark_baseline.py`) remains as
the permanent before/after comparison harness — it is never deleted and its
outputs stay reproducible.

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
Press `q` in the video window to quit. The live loop runs the full
localization pipeline (detector → VLM → embedding → retrieval → semantic
scoring → Bayes update) only on novel frames; redundant frames reuse the
last state (runtime gate, `runtime.max_stale_seconds` forces a periodic
re-run).

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

Hand-label a small set of test frames in `data/evaluation/test_labels.json`:

```json
{ "frame_00010.jpg": 0, "frame_00050.jpg": 3 }
```

Then run the research evaluation suite:

```bash
python evaluate_suite.py                 # localization + graph metrics + ablation
python scripts/failure_inspector.py      # failure_log.jsonl from the decision log
python evaluate.py                       # legacy single-run report + histogram baseline
```

**Important honesty note:** without real labels, every number the suite
reports is *self-consistency against pseudo-labels*, not physical accuracy.
Record a real walkthrough + a second independent evaluation walkthrough and
label them before drawing research conclusions; the suite labels its output
accordingly. For a meaningful number, use frames that weren't part of
building the map.

## Testing

```bash
pytest tests/
```

130+ tests covering the full pipeline — sampling, observations, segmentation,
place building/reconciliation, graph construction/validation, retrieval,
scoring, semantic scoring, the Bayes state estimator, the localization state
machine, confidence calibration, the runtime gate, and the evaluation suite.
Slow tests (real YOLO/VLM/encoder smoke) are marked `slow` and skippable;
fast tests use small toy graphs and synthetic vectors only.

## Limitations

- The current dataset is provisional: 301 pre-extracted frames from a
  walkthrough video that is no longer in the repo. Metrics on it are
  self-consistency, not accuracy (see above).
- Place reconciliation can still merge two visually near-identical but
  physically distinct rooms; semantic evidence (scene tags + landmarks +
  objects) is the current mitigation.
- Live tracking is interval-based, not true frame-by-frame video tracking.
- No live camera/AR pose estimation — position is "which mapped place do
  you most resemble right now," not a precise coordinate.

## Future work

- Run the registered-but-untested encoder comparison (DINOv2 vs ResNet18)
  behind the `VisualEncoder` interface, gated on a measured win.
- Consider YOLOE-26n (open-vocabulary detection) for door/stairs/sign
  bounding boxes — an explicit, separately-reviewed extension.
- Investigate on-device deployment (LFM2 already runs <250 ms/frame on a
  Jetson Orin at 512×512) for a true continuous live-camera experience on a
  phone.
