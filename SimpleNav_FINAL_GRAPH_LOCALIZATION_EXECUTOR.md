# SimpleNav — Stage-by-Stage Implementation Executor

> **Companion to:** `SimpleNav_FINAL_GRAPH_LOCALIZATION_PLANNER.md` (the spec — cross-references below use `§` + section number).
>
> **Purpose:** this is the runbook the executor (Claude) follows, **one stage at a time, in the exact order below**, to turn the current SimpleNav prototype into the research-grade camera-only indoor navigation system. Every stage is independently committable and the legacy baseline stays runnable at every step.
>
> **Status: ⬜ All 30 stages pending — implementation starts at Stage 00 on the next turn.**

---

## 0. How to execute this document

1. **Read both files first**: this executor + the planner md. The planner holds the *why* and the acceptance criteria; this file holds the *what*, *where*, and *how*.
2. **One stage per turn (or per commit).** Start at the lowest-numbered unchecked stage. Never skip ahead: stages have dependencies (Appendix G).
3. **At the end of each stage:**
   - run every `Verify` command listed, from the project root;
   - tick the stage's `Status:` checkbox to `✅ Done` in the master table and in the stage header;
   - report to the user and **suggest a commit message** — the user commits (they do their own git).
4. **Environment** (verified 2026-09-02): conda env `ML` at `/home/lenovo/miniconda3/envs/ML`, Python 3.11.14, torch 2.10.0+cu126, GPU NVIDIA RTX 3050 **6 GB Laptop**. There is **no `python` on PATH** — always invoke as:
   ```bash
   conda run -n ML python -m pytest tests/ -q          # tests
   conda run -n ML python -m src.<module> ...           # any pipeline CLI
   conda run -n ML python scripts/<script>.py ...       # scripts
   ```
5. **Missing packages to install (into `ML`) at the stage that needs them**: `ultralytics` (Stage 05), `bitsandbytes` (Stage 06), `pyttsx3` (Stage 29). **Ask the user before installing anything.**
6. **Baseline preservation is mandatory** (§1 Rules 1–6): legacy files are never deleted, backbone-changing CLIs get a `--baseline` flag, and the Stage 01 benchmark is re-run after every backbone change for comparison.

---

## 1. Current state snapshot (what we start from)

| Path | Role | Notes |
|---|---|---|
| `src/extract_frames.py` | fixed every-Nth frame extraction | to be upgraded by Stage 03 |
| `src/embedder.py` | frozen ResNet18, 512-d L2-normalized embeddings | to be wrapped by Stage 04 |
| `src/embed_frames.py` | frames → `embeddings.npy` + `frame_names.json` | to emit Observations by Stage 02/05 |
| `src/build_places.py` | KMeans (k by silhouette sweep) + ≤3 exemplars/place | stays as `--baseline` from Stage 09 |
| `src/build_graph.py` | transition-count NetworkX graph | stays as baseline path |
| `src/localize.py` | `PlaceIndex` (FAISS IndexFlatIP) + `LiveLocalizer` majority vote | stays as baseline; new retrieval from Stage 15 |
| `src/live_tracker.py` | `LiveTracker.process_frame(embedding) → status dict` | status-dict shape must remain compatible |
| `src/navigate.py` | weighted shortest path + `format_directions` | kept unchanged (§1 Rule 1) |
| `src/floor_plan.py`, `src/label_places.py`, `src/speak.py`, `src/utils.py` | support components | kept; utils used by everything |
| `app.py` / `live_navigate.py` | Streamlit demo / webcam CLI | consume new tracker at Stage 28 |
| `evaluate.py` | test_labels.json accuracy + histogram baseline | superseded by Stage 25 suite (kept) |
| `tests/test_live_tracker.py`, `tests/test_navigate.py` | 10 passing tests | must keep passing |
| `config.yaml` | paths + extraction/embedding/clustering/navigation/live | grows per stage (Appendix F) |
| `data/frames/` | **301 extracted frames (untracked, provisional)** | the only dataset right now |
| `data/map/` | tracked legacy artifacts: `embeddings.npy`, `place_exemplars.npy`, `exemplar_place_ids.npy`, `place_assignments.json`, `frame_names.json`, `place_names.json` (11 unnamed places), `graph.pkl` | legacy format; versioned map from Stage 14 |

**⚠ Blockers known today:**
- `data/College_env.mp4` (referenced in `config.yaml`) does **not exist**. The 301 frames are the **provisional** dataset. Real mapping + evaluation walkthroughs are a user task (Appendix B) — the design works with the provisional set until they arrive.
- No `data/test_labels.json`, no ground-truth place labels. Stage 01 creates provisional labels by viewing held-out frames.

---

## 2. Master stage table

| # | Stage | Milestone | Deliverable | Status |
|---|---|---|---|---|
| 00 | Environment & provisional dataset | A — Baseline | working env, data dirs, frame manifest | ⬜ |
| 01 | Freeze & benchmark baseline | A — Baseline | `baseline_report.json/.md` + permanent harness | ✅ |
| 02 | Observation model | B — Mapping | `Observation` dataclass + JSONL I/O | ✅ |
| 03 | Smart sampling | B — Mapping | quality + novelty sampling | ✅ |
| 04 | Encoder abstraction | B — Mapping | `VisualEncoder` + registry | ✅ |
| 05 | Observation store | B — Mapping | `ObservationStore` (FAISS + JSONL) | ⬜ |
| 06 | Object detector (YOLOv8n) | B — Mapping | `Detector` interface + YOLO impl | ⬜ |
| 07 | VLM scene/landmark tagging | B — Mapping | `SceneTagger` (Qwen2.5-VL 3B 4-bit) | ⏸ deferred — stub active; Qwen 4-bit output incoherent on this GPU (parked 2026-09-02); re-enable `perception.vlm_enabled` after debugging |
| 08 | Temporal segmentation | B — Mapping | segment boundaries from ordered obs | ✅ |
| 09 | Place formation | B — Mapping | places + exemplars from segments | ✅ |
| 10 | Place reconciliation | B — Mapping | revisit merging, duplicate prevention | ✅ |
| 11 | Transition extraction | B — Mapping | debounced transition statistics | ✅ |
| 12 | Graph builder + junction detection | B — Mapping | confidence-weighted graph + `graph.json` | ✅ |
| 13 | Graph validation | B — Mapping | `graph_validation.json/.md` | ✅ |
| 14 | Versioned map artifact | B — Mapping | `MapBundle` manifest + loader | ✅ — Milestone B achieved: 7 places / 6 edges from 301 obs |
| 15 | Candidate retrieval | C — Localization | top-K candidates + margins | ✅ |
| 16 | Candidate scoring | C — Localization | visual+semantic+temporal+graph score | ✅ |
| 17 | Graph-constrained filtering | C — Localization | soft graph penalties, local-first policy | ✅ |
| 18 | Probabilistic state estimator | C — Localization | Bayes filter posterior | ✅ |
| 19 | State machine | C — Localization | TRACKING/UNCERTAIN/LOST/REACQUIRING/ARRIVED | ✅ |
| 20 | Transition stabilization | C — Localization | K-consecutive confirmation | ✅ |
| 21 | Global reacquisition | C — Localization | LOCAL vs GLOBAL retrieval modes | ✅ |
| 22 | Confidence calibration | C — Localization | HIGH/MEDIUM/LOW/UNKNOWN | ✅ |
| 23 | Semantic localization | C — Localization | detector/VLM evidence in the posterior | ⬜ — scoring wired; runtime evidence path pending (VLM parked) |
| 24 | Runtime optimization | D — Navigation | cheap gate before expensive models | ⬜ |
| 25 | Evaluation suite | E — Research | map + localization metric runners | ⬜ |
| 26 | Ablations | E — Research | progression A–G / A–F reports | ⬜ |
| 27 | Failure analysis | E — Research | decision log + inspector | ⬜ |
| 28 | UI integration | D — Navigation | app.py + live_navigate.py on new tracker | ⬜ |
| 29 | Final cleanup & docs | E — Research | README, requirements, full test pass | ⬜ |

**Milestones** (§40): A = known-performance baseline · B = autonomous indoor graph · C = stable current place · D = full navigation loop · E = research validation.

---

## 3. Stage runbooks

---

### Stage 00 — Environment & Provisional Dataset

**Planner ref:** §3 (dataset prerequisite) · **Milestone:** A

**Objective:** get a verified, reproducible environment and make the 301-frame set a first-class (provisional) dataset.

**Actions**

- [ ] 1. Confirm environment facts (already verified; re-check only if something fails):
  ```bash
  conda run -n ML python -c "import torch, faiss; print(torch.__version__, faiss.__version__, torch.cuda.is_available())"
  conda run -n ML python -m pytest tests/ -q     # expect 10 passed
  ```
- [ ] 2. Create data directories per target structure (§2): `data/raw/`, `data/observations/`, `data/evaluation/` (add `.gitkeep`).
- [ ] 3. Create `data/frames/manifest.json` — one entry per frame with `frame`, `index`, `source_video` (`"College_env.mp4 (missing — provisional dataset)"`), `fps_unknown`. Simple loop over sorted `frame_*.jpg`.
- [ ] 4. Update `config.yaml`:
  - [ ] `paths.video` → comment that `College_env.mp4` is absent; keep value for when the user provides a video.
  - [ ] add `paths.observations_dir: "data/observations"`, `paths.evaluation_dir: "data/evaluation"`.
- [ ] 5. Run the legacy pipeline end-to-end as a smoke test (it will reuse existing outputs where present):
  ```bash
  conda run -n ML python -m src.embed_frames     # may re-embed 301 frames
  conda run -n ML python -m src.build_places
  conda run -n ML python -m src.build_graph
  conda run -n ML python -m src.navigate --source "Place_0" --destination "Place_10"
  ```
  Fix anything that breaks before moving on (embedding re-run is fine; no new files expected).

**Definition of done**

- [ ] `conda run -n ML python -m pytest tests/ -q` → 10 passed
- [ ] `data/{raw,observations,evaluation}` exist; `data/frames/manifest.json` lists 301 frames
- [ ] legacy pipeline CLI chain runs end-to-end without errors

**Suggested commit message:** `Stage-00: env verification and provisional dataset layout`

---

### Stage 01 — Freeze & Benchmark the Baseline

**Planner ref:** §3 (Stage 0), §30 (metrics), §37-01 · **Milestone:** A

**Objective:** an objective, reproducible, machine-readable baseline before any backbone change. **This harness is permanent** — re-run it after every backbone stage (02–24) and record results in `data/evaluation/reports/`.

**Actions**

- [ ] 1. Create `scripts/benchmark_baseline.py` (new `scripts/` dir). It must be runnable with `conda run -n ML python scripts/benchmark_baseline.py` and accept `--provisional` (default on while no real video exists).
  - **Split:** temporal held-out 80/20 of the 301 frames (first 80% train map, last 20% held out — never used to build the map, per §Rule 6). Write split to `data/evaluation/baseline_split.json`.
  - **Build:** on the train 80% run the current pipeline pieces via their library functions (reuse `src.embedder.embed_image`, `src.build_places.build_places`, `src.build_graph.build_graph`, `src.localize.PlaceIndex`) — no CLI shell-outs.
  - **Ground truth (two paths):**
    - *If the executor session has vision:* view the train exemplar frames + held-out frames (contact sheets via `scripts/make_contact_sheet.py`) and label coarse physical areas into `data/evaluation/heldout_labels.json` + `data/evaluation/exemplar_labels.json` (templates: `*.template.json`). The script prefers these.
    - *Vision-less fallback (automatic):* if those files are missing, the script computes **full-data KMeans pseudo-labels** (k silhouette-selected over all 301 embeddings) and reports **self-consistency** metrics (does the train-built map reproduce the full-data cluster structure on unseen frames?) — clearly labeled, NOT physical accuracy. Offer the user the contact sheets so they can fill real area labels when they want physical metrics.
  - **Metrics** (§30) on the held-out 60 frames, sequence-ordered:
    - top-1 and top-3 accuracy (match to the nearest labeled area, mapping KMeans place → dominant label);
    - false jump rate (fraction of frames where the majority-vote tracker changes place without a true area change);
    - transition detection accuracy (true area-boundary crossings vs detected place changes);
    - graph node count, edge count;
    - localization latency (mean seconds per frame, measured with `time.perf_counter` around `embed_image` + `PlaceIndex.query`).
  - **Output:** `data/evaluation/baseline_report.json` (all numbers + params + `"dataset": "provisional"` + timestamp) and `data/evaluation/baseline_report.md` (readable summary). `data/evaluation/reports/` keeps dated copies.
- [ ] 2. Save the split + report; nothing in `data/map/` legacy artifacts is modified.

**Definition of done**

- [ ] `baseline_report.json` + `.md` exist and are machine-readable
- [ ] report explicitly labeled `provisional` (no real held-out walkthrough yet)
- [ ] report includes: top-1, top-3, false jump rate, transition accuracy, node/edge counts, latency

**Verify:** `conda run -n ML python scripts/benchmark_baseline.py` runs clean; re-run produces identical numbers for the same split (deterministic seeds).

**Suggested commit message:** `Stage-01: baseline benchmark harness + provisional report`

---

### Stage 02 — Observation Model

**Planner ref:** §4 (Stage 1) · **Milestone:** B

**Objective:** one common object for "visual evidence at a known point in time"; stop passing loose parallel arrays (§4 Why).

**Create**

- `src/mapping/__init__.py` (empty; starts the `mapping` package)
- `src/mapping/observations.py`:

```python
@dataclass
class Observation:
    id: str                      # e.g. "obs_000123" — stable, orderable
    timestamp: float             # seconds into source video
    frame_path: str              # path to the saved frame image
    embedding: np.ndarray | None = None      # never serialized to JSONL
    quality_score: float | None = None
    objects: list[dict] = field(default_factory=list)   # {"class","confidence","bbox":[x1,y1,x2,y2]}
    scene_tags: dict | None = None             # VLM output, fixed schema (§7)
    landmarks: list[str] = field(default_factory=list)
    segment_id: str | None = None
    place_id: str | None = None
```

Functions:
- `observation_to_dict(obs) -> dict` (excludes `embedding`),
- `observation_from_dict(d, embedding=None) -> Observation`,
- `save_observations_jsonl(obs_list, path)`, `load_observations_jsonl(path, embeddings=None) -> list[Observation]`,
- `sort_observations(obs_list) -> list[Observation]` (by `timestamp`, then `id`).

**Modify**

- [ ] `src/embed_frames.py`: after writing legacy `.npy`/json outputs, also build `Observation` records (id `obs_{i:06d}`, timestamp = i × `sample_every_n / fps` — fps unknown for provisional data, use `timestamp = float(i)` with a note, frame_path relative to project root) and save `data/observations/observations.jsonl` + `data/observations/embeddings.npy` + `data/observations/encoder.json` (`{"model": "resnet18", "dimension": 512}`). Legacy outputs remain untouched (baseline).

**Config:** no new keys (paths.observations_dir added in Stage 00).

**Tests — `tests/test_observations.py`**

- round-trip: save → load preserves all fields except embedding (embedding restored from array),
- ordering: `sort_observations` orders by timestamp,
- metadata may be absent (objects=[], scene_tags=None) without breaking (per §4 instruction 6).

**Definition of done**

- [ ] every accepted frame is traceable: video → timestamp → image → embedding → metadata (§4 acceptance)
- [ ] legacy `embeddings.npy`/`frame_names.json` still produced unchanged

**Verify:** `conda run -n ML python -m pytest tests/test_observations.py -q` and re-run `python -m src.embed_frames` → legacy files byte-identical plus new `data/observations/*`.

**Suggested commit message:** `Stage-02: Observation model + JSONL store`

---

### Stage 03 — Smart Sampling

**Planner ref:** §5 (Stage 2) · **Milestone:** B

**Objective:** keep frames that add spatial information; drop redundant/blurry/dark frames; retain transitions (§5 algorithm).

**Create**

- `src/extraction/__init__.py`
- `src/extraction/quality.py`:
  - `frame_is_valid(frame) -> bool` (non-empty, 3-channel),
  - `blur_variance(frame) -> float` (variance of Laplacian — §5 blur gate),
  - `brightness(frame) -> float` (mean of grayscale),
  - `compute_quality(frame) -> dict` returning `{"valid", "blur_variance", "brightness", "quality_score"}` where `quality_score` in [0,1] (low blur + mid-range brightness → high score),
  - `reject_reasons(frame) -> list[str]` — why it failed (invalid/dark/blurry).
- `src/extraction/frames.py`:
  - `SmartFrameSampler(config)` with `process(video_path) -> list[dict]` (each dict: `frame` (BGR ndarray), `frame_idx`, `timestamp`).
  - Decision logic (exact order, §5 suggested logic):
    1. `quality`: reject invalid, `blur_variance < blur_threshold`, `brightness < dark_threshold` → **reject**;
    2. `novelty`: cosine distance of a cheap descriptor vs last **accepted** frame. Cheap descriptor: 32×32 grayscale resize, flattened, L2-normalized (no neural net — keeps extraction fast). If `distance < novelty_threshold` → **reject** (nearly identical);
    3. `transition-aware retention`: maintain a short running window (length `transition_window`) of descriptor distances. If a **sustained** rise (≥ `transition_window` consecutive frames above `transition_threshold`) occurs, force-keep the first frame at the change and a couple after (`transition_keep_after`) — a single spike does not count (§5 "Do not treat one spike as a transition");
    4. `temporal gap`: if `timestamp - last_accepted_timestamp < min_interval_seconds` → **reject**, else **keep** and update last-accepted descriptor/embedding.

**Modify**

- [ ] `src/extract_frames.py`: default path = smart sampling; `--baseline` flag restores fixed every-Nth behavior exactly as today. Keep writing zero-padded `frame_%05d.jpg` and an accompanying `sampling_log.json` (per-frame accept/reject reason — great for Stage 13 validation).

**Config — new `sampling:` section (§35):**

```yaml
sampling:
  min_interval_seconds: 0.5
  novelty_threshold: 0.15        # descriptor distance — tune on provisional data
  transition_threshold: 0.4
  transition_window: 5
  transition_keep_after: 2
  blur_threshold: 40.0
  dark_threshold: 20.0
```

**Tests — `tests/test_extraction.py`** (synthetic frames, no video file):

- all-black frame → rejected (dark); heavily blurred frame (Gaussian blur) → rejected; sharp frame → accepted;
- 20 nearly identical frames → 1 kept;
- abrupt change between two different synthetic patterns → kept, plus neighbors kept (transition retention);
- single spike among identical frames → no extra keeps.

**Definition of done**

- [ ] on a real walkthrough, frame count drops substantially vs fixed-N (§5 acceptance)
- [ ] `--baseline` produces identical behavior to today's extractor

**Verify:** `conda run -n ML python -m pytest tests/test_extraction.py -q`. (Real-video reduction check requires a video — deferred until the user provides one; provisional check runs sampling on the 301 frames via a tiny harness comparing counts.)

**Suggested commit message:** `Stage-03: smart frame sampling with quality/novelty gate`

---

### Stage 04 — Encoder Abstraction

**Planner ref:** §8 (Stage 5) · **Milestone:** B

**Objective:** detach the system from ResNet18; swap encoders by configuration only (§8 acceptance).

**Create**

- `src/embeddings/__init__.py`
- `src/embeddings/encoder.py`:

```python
class VisualEncoder(ABC):
    name: str                    # "resnet18", "clip_vit_b32", ...
    version: str                 # model weights/version string
    dimension: int               # 512
    def encode(self, image) -> np.ndarray: ...       # HxWx3 or PIL or path -> L2-normalized float32
    def batch_encode(self, images) -> np.ndarray: ...  # (N, dimension)

def get_encoder(config) -> VisualEncoder   # registry keyed by config["embedding"]["model"]
```

- `ResNet18Encoder(VisualEncoder)`: wraps the existing logic in `src/embedder.py` (reuse `load_model`, `TRANSFORM`; do not duplicate). `name="resnet18"`, `version="torchvision_imagenet"`, `dimension=512`.
- Keep `src/embedder.py` as a thin compat wrapper: `embed_image` delegates to `get_encoder({"embedding": {"model": "resnet18"}})` so every legacy caller (`app.py`, `localize.py`, `evaluate.py`) works unchanged.

**Config — extend `embedding:`**

```yaml
embedding:
  model: "resnet18"
  image_size: 224
```

**Tests — `tests/test_encoder.py`**

- `get_encoder` returns ResNet18Encoder for `"resnet18"`, raises for unknown model;
- output shape (512,), L2 norm ≈ 1, dtype float32;
- `encode` accepts ndarray, PIL Image, and file path identically.

**Definition of done**

- [ ] encoder swappable via config with zero changes in mapping/localization logic (§8 acceptance)
- [ ] legacy `embed_image` callers behave identically

**Verify:** `conda run -n ML python -m pytest tests/test_encoder.py -q`; re-run `python -m src.localize <one held-out frame>` → same result as before this stage.

**Suggested commit message:** `Stage-04: VisualEncoder abstraction with ResNet18 baseline`

---

### Stage 05 — Observation Store

**Planner ref:** §9 (Stage 6) · **Milestone:** B

**Objective:** the shared memory used by both graph formation and localization: vector index + metadata store (§9 architecture — never force metadata into the vector DB).

**Create** — `src/mapping/observation_store.py`:

```python
class ObservationStore:
    def __init__(self, obs_dir: Path): ...
    def add(self, observations: list[Observation]) -> None        # appends, rebuilds index
    def get(self, obs_id: str) -> Observation                     # metadata + embedding + frame path
    def all(self) -> list[Observation]
    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[Observation, float]]
    def save(self) / @classmethod load(cls, obs_dir) -> "ObservationStore"
```

- On-disk layout (all under `data/observations/`):
  - `observations.jsonl` — metadata rows (no embeddings),
  - `embeddings.npy` — (N, D) aligned with file order,
  - `encoder.json` — `{"model", "version", "dimension"}`,
  - `index.faiss` — FAISS `IndexFlatIP` (cosine == dot since vectors are unit-norm), rebuilt on save,
  - `id_order.json` — obs ids in index row order (guarantees id→row mapping survives reordering).
- `get(obs_id)` must recover **all** metadata **and** the original frame path (§9 acceptance); store frame paths relative to project root.
- Failure mode: missing index/metadata raises a clear `ObservationStoreError` listing what's missing.

**Modify**

- [ ] `src/embed_frames.py`: additionally write an `ObservationStore` at `data/observations/` (build Observations as Stage 02, save store). Legacy `.npy`/json outputs stay.

**Tests — `tests/test_observation_store.py`**

- add N observations → save → fresh load → `get` returns identical metadata + embedding + existing frame path;
- `search` returns nearest by cosine similarity;
- `id_order.json` round-trips after append (new obs appended, ids stable).

**Definition of done**

- [ ] given an observation id, all metadata + original frame are recoverable (§9 acceptance)
- [ ] mapping and localization both read from this store from Stage 08 onward

**Verify:** `conda run -n ML python -m pytest tests/test_observation_store.py -q`; re-run `embed_frames` → `data/observations/` populated, legacy files unchanged.

**Suggested commit message:** `Stage-05: ObservationStore (FAISS + JSONL metadata)`

---

### Stage 06 — Object Detector (YOLOv8n)

**Planner ref:** §6 (Stage 3) · **Milestone:** B

**Objective:** structured visual evidence without letting detection control the map (§Rule 3).

**Install (ask user first):** `conda run -n ML pip install ultralytics`

**Create** — `src/perception/__init__.py`, `src/perception/detector.py`:

```python
@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]   # normalized [x1, y1, x2, y2] in [0, 1]

class Detector(ABC):
    def detect(self, image) -> list[DetectedObject]: ...   # may return [] — never raises
    def classes(self) -> list[str]: ...

class YoloDetector(Detector):   # ultralytics YOLOv8n; filters by confidence; maps COCO names
class StubDetector(Detector):   # deterministic, no model — for tests and offline runs
def get_detector(config) -> Detector
```

- [ ] Normalized bboxes (§6 instruction 2), confidence included (§6 instruction 3), threshold applied (§6 instruction 4), failure → `[]` not crash (§6 instruction 5).
- [ ] `StubDetector` deterministically returns a couple of objects for a synthetic "scene" (configurable) — used by tests and by the mapping pipeline when `detector_enabled: false`.
- [ ] COCO-class reality: YOLO gives `person/couch/chair/refrigerator/fire hydrant/...`. Door/stairs/sign coverage comes from the VLM layer (Stage 07) — document this in the module docstring.

**Config — new `perception:` section:**

```yaml
perception:
  detector_enabled: true
  detector_model: "yolov8n.pt"
  detector_confidence: 0.35
```

**Modify**

- [ ] `src/embed_frames.py` (or a new `src/build_observations.py` orchestrator introduced here): when `detector_enabled`, populate `Observation.objects`; when off, leave `[]`.

**Tests — `tests/test_detector.py`**

- stub returns expected classes/bboxes and never raises on a corrupted "image";
- YOLO (if weights download works) runs on one real frame, returns objects with bboxes in [0,1] — mark `@pytest.mark.slow` to skip in CI;
- `get_detector` honors config (`detector_enabled: false` → StubDetector).

**Definition of done**

- [ ] frames carry stable object metadata; swapping the detector touches no graph code (§6 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_detector.py -q`

**Suggested commit message:** `Stage-06: object detector abstraction + YOLOv8n`

---

### Stage 07 — VLM Scene & Landmark Tagging

**Planner ref:** §7 (Stage 4) · **Milestone:** B

**Objective:** deterministic-enough semantic evidence from a local model that fits 6 GB VRAM. VLM **must not** produce coordinates/topology/distances/location (§7 "must NOT produce").

**Install (ask user first):** `conda run -n ML pip install bitsandbytes`

**Create** — `src/perception/scene_tagger.py`:

```python
@dataclass
class SceneTags:
    scene_type: str               # controlled vocabulary: room/corridor/corridor_junction/stairs/elevator/entrance/unknown
    landmarks: list[str]          # "blue room-number sign", "staircase on left" — short noun phrases
    navigation_relevance: list[str]  # subset of: junction, stairs, elevator, entrance, sign, door, ...
    description: str              # ≤ 1 short sentence, stored but not used for matching

SCENE_TYPES = {...}               # closed vocabulary the prompt must choose from
VALID_NAV_TAGS = {...}

class SceneTagger(ABC):
    def tag(self, image, objects: list[DetectedObject] | None = None) -> SceneTags: ...
    def name(self) -> str: ...

class QwenVLTagger(SceneTagger):
    # Qwen/Qwen2.5-VL-3B-Instruct, 4-bit (BitsAndBytesConfig), device_map="auto",
    # max_new_tokens=128, temperature=0 (deterministic-ish)
class SmolVLMTagger(SceneTagger):
    # HuggingFaceTB/SmolVLM2-2.2B-Instruct — lighter fallback if 3B is slow
class StubTagger(SceneTagger):
    # deterministic: derives scene_type from detector objects (e.g. door+corridor),
    # landmarks=[] — used when vlm_enabled=false and in tests
def get_scene_tagger(config) -> SceneTagger
```

- [ ] **Fixed prompt** with the exact JSON schema embedded; ask for JSON only. `_parse_and_validate(text) -> SceneTags`: json.loads → schema check (required keys, scene_type in vocabulary, lists of short strings, description ≤ 160 chars) → on any failure return `scene_type="unknown"`, `landmarks=[]` (never crash, never trust malformed output) (§7 instruction 3).
- [ ] Store only navigation-relevant fields (§7 instruction 5): `description` kept for humans, excluded from matching.

**Config — extend `perception:`**

```yaml
perception:
  vlm_enabled: true
  vlm_model: "Qwen/Qwen2.5-VL-3B-Instruct"
  vlm_quantization: "4bit"
  vlm_max_tokens: 128
  vlm_cache_dir: null          # optional HF cache override
```

**Modify**

- [ ] observation build: when `vlm_enabled`, populate `Observation.scene_tags` + `landmarks`; when off, `None`/`[]` (pipeline must not break, §4 instruction 6).

**Tests — `tests/test_scene_tagger.py`**

- stub is deterministic (same image → same tags);
- `_parse_and_validate`: valid JSON passes; malformed JSON, wrong scene_type, missing keys → `unknown`/`[]` fallback;
- QwenVLTagger marked `@pytest.mark.slow`: one real tag on a frame (smoke; skip if no GPU/HF access).

**Definition of done**

- [ ] VLM output comparable across observations and improves place discrimination (§7 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_scene_tagger.py -q`

**Suggested commit message:** `Stage-07: VLM scene/landmark tagging (Qwen2.5-VL 4-bit)`

---

### Stage 08 — Temporal Segmentation

**Planner ref:** §10 (Stage 7) · **Milestone:** B

**Objective:** contiguous periods of the walkthrough ≈ local spatial regions; no single-frame noise boundaries (§10).

**Create** — `src/mapping/segmentation.py`:

```python
@dataclass
class Segment:
    id: str                     # "seg_000"
    obs_ids: list[str]          # ordered observation ids
    start_index: int            # into the ordered observation list
    end_index: int
    representative_obs_id: str  # obs closest to segment mean embedding

def segment_observations(observations: list[Observation], config) -> list[Segment]
```

Algorithm (concrete):

1. Order observations by timestamp (Stage 02 helper).
2. Per consecutive pair compute change signals:
   - `visual_d_t` = 1 − cosine(emb_t, emb_{t+1}) ∈ [0, 1] (need embeddings; if missing, fall back to 0.5 — a neutral signal, never blocks a boundary);
   - `semantic_d_t` ∈ {0, 1}: 1 if scene_type changes or landmark set changes materially (landmark Jaccard < 0.5 when both non-empty, else 0) — §10 "scene change / landmark change / object change";
   - `change_t` = `visual_d_t + 0.5 * semantic_d_t`.
3. **Smoothing window** (short, per §10 "Use a short temporal window"): boundary score `b_t = max(change over [t−W, t+W])` with W = `segment_change_window` (default 2) — one anomalous frame can't create a boundary (it must be high for W+1 consecutive frames).
4. Boundary at t where `b_t > segment_distance_threshold`.
5. Merge segments shorter than `segment_min_length` (default 3) into their larger neighbor.
6. Return segments with contiguous obs ids and representative = argmin of distance to segment mean embedding.

**Config — new `mapping:` section (grows through Stage 13):**

```yaml
mapping:
  segment_distance_threshold: 0.35
  segment_change_window: 2
  segment_min_length: 3
```

**Tests — `tests/test_segmentation.py`** (synthetic embeddings in 2-D, hand-placed):

- A…A (10) then B…B (10) → exactly 2 segments;
- A…A with one anomalous frame in the middle → still 1 segment (window kills the spike);
- A→B→A sequences where A is visually identical → segments reflect the temporal structure (this is *why* reconciliation exists in Stage 10).

**Definition of done**

- [ ] segments correspond to physical areas, not arbitrary K-Means clusters (§10 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_segmentation.py -q`

**Suggested commit message:** `Stage-08: temporal segmentation of observations`

---

### Stage 09 — Place Formation

**Planner ref:** §11 (Stage 8) · **Milestone:** B

**Objective:** convert segments into persistent physical places (§11 "A place is not a cluster of images").

**Create** — `src/mapping/place_builder.py`:

```python
@dataclass
class Place:
    place_id: str
    segment_ids: list[str]
    observation_ids: list[str]
    exemplar_ids: list[str]      # chosen observation ids serving as exemplars
    scene_types: Counter[str]
    landmarks: list[str]         # union, most common first
    visual_stats: dict           # mean/std of within-place embedding similarities

def build_places(store: ObservationStore, segments: list[Segment], config) -> list[Place]
def select_exemplars(obs_list: list[Observation], method: str, max_exemplars: int) -> list[str]
```

- [ ] One place per segment initially (reconciliation, Stage 10, merges them).
- [ ] Exemplar methods (§11 Methods A/B/C), chosen by `mapping.exemplar_method`:
  - `"kmeans"` — secondary K-Means within the segment (like today's `build_exemplars`);
  - `"diversity"` — greedy max-min: pick the obs farthest from already-chosen exemplars, up to `max_exemplars_per_place`;
  - `"temporal_diversity"` (default) — pick exemplars spread across the segment's temporal extent (first, last, middle) then diversify with max-min if more slots remain.
- [ ] `visual_stats`: within-place mean cosine similarity and std (feeds Stage 13 "high internal variance" check and Stage 16 scoring).

**Modify**

- [ ] `src/build_places.py`: add `--baseline` flag (current KMeans path); default path calls `build_places()` + `select_exemplars`. Legacy output files written from the new places too (place_assignments.json etc. — but do not overwrite `place_names.json` with custom names, keep Stage 9 of current behavior).

**Config — extend `mapping:`**

```yaml
mapping:
  exemplar_method: "temporal_diversity"
  max_exemplars_per_place: 3
```

**Tests — `tests/test_place_builder.py`**

- segments → places: ids, observation membership, scene_types counter;
- exemplar methods: temporal_diversity picks spread-out ids; diversity picks max-min; kmeans matches today's behavior on a toy set.

**Definition of done**

- [ ] places tolerate viewpoint/lighting/direction changes via multi-exemplar representation (§11 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_place_builder.py -q`

**Suggested commit message:** `Stage-09: place formation + exemplar selection`

---

### Stage 10 — Place Reconciliation

**Planner ref:** §12 (Stage 9) · **Milestone:** B

**Objective:** `Lobby → Corridor → Lobby` merges into one Lobby place; visually similar distinct rooms stay separate (§12).

**Create** — `src/mapping/place_reconciliation.py`:

```python
def reconcile_places(places: list[Place], store: ObservationStore, config) -> list[Place]
```

- [ ] Merge evidence (require **multiple** signals, §12 "Require multiple signals"):
  - visual: mean exemplar similarity between places > `merge_visual_threshold`;
  - landmarks: landmark Jaccard > `merge_landmark_threshold` **or** both have empty landmarks (then visual must be strong);
  - scene: scene types compatible (`merge_scene_required` gates this);
  - temporal/context: the two segments occurred in a revisit pattern (place A observed before and after the candidate's segment — from the walkthrough order) **or** repeated traversal evidence.
- [ ] Reject-merging rules (§12 "Evidence against merging"): landmark conflicts strongly (Jaccard ≈ 0 with ≥ 1 confident landmark each), scene types differ substantially, graph context conflicts (to be refined in Stage 12+), observations repeatedly more similar to a third place.
- [ ] Deterministic, idempotent (merging is a fixed-point iteration; merged place takes the lower place_id; log every merge to a `reconciliation_log.json`).

**Config — extend `mapping:`**

```yaml
mapping:
  merge_visual_threshold: 0.75
  merge_landmark_threshold: 0.5
  merge_scene_required: true
```

**Tests — `tests/test_place_reconciliation.py`**

- LobbyA / Corridor / LobbyB (synthetic embeddings: LobbyA ≈ LobbyB, both ≠ Corridor) → 2 places after reconciliation;
- RoomX ≈ RoomY visually, differing landmarks, different scene context → stay separate;
- idempotent: reconcile(reconcile(P)) == reconcile(P).

**Definition of done**

- [ ] revisits become one place; similar-but-distinct places stay separate (§12 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_place_reconciliation.py -q`

**Suggested commit message:** `Stage-10: place reconciliation (multi-signal merge)`

---

### Stage 11 — Transition Extraction

**Planner ref:** §13 (Stage 10) · **Milestone:** B

**Objective:** turn the place sequence into debounced movement evidence (§13).

**Create** — `src/mapping/transition_builder.py`:

```python
@dataclass
class TransitionStats:
    a: str; b: str
    forward_count: int          # A→B observations
    reverse_count: int          # B→A
    transition_duration: float  # mean duration of the movement A→B in seconds
    supporting_observations: int
    visual_transition_strength: float   # mean boundary change score (Stage 08) at crossings
    confidence: float           # computed, see below

def extract_transitions(ordered_place_ids: list[str], observations, segments, config) -> list[TransitionStats]
def _debounce(ordered_place_ids: list[str], min_persistence: int) -> list[str]
```

- [ ] **Debounce first** (§13 debouncing): filter the raw place sequence so a place must persist ≥ `transition_persistence` observations before it counts as a real visit; `A→B→A` within the persistence window is treated as noise and removed (stay in A).
- [ ] From the debounced sequence, count transitions between consecutive distinct places; aggregate directional counts, durations, supporting observations.
- [ ] `confidence = min(1.0, forward_count / minimum_edge_support) * persistence_factor` — tuned so a single noisy crossing yields confidence < 0.5.

**Config — extend `mapping:`**

```yaml
mapping:
  transition_persistence: 3
  minimum_edge_support: 3
```

**Tests — `tests/test_transition_builder.py`**

- `A A A A B B B B C C` → transitions A→B, B→C, forward counts 1 each;
- `A B A B A B` (noise, persistence 3) → debounced stays `A A A A A` → no transitions;
- `A A A B B A A A` with persistence 3 → A→B counts 1, B→A counts 1 (real traversal both ways).

**Definition of done**

- [ ] false one-frame transitions never create persistent edges (§13 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_transition_builder.py -q`

**Suggested commit message:** `Stage-11: debounced transition extraction`

---

### Stage 12 — Graph Builder + Junction Detection

**Planner ref:** §14 (Stage 11), §15 (Stage 12) · **Milestone:** B

**Objective:** the physical connectivity graph: node = place, edge = physical connection with measurable confidence (§14). Junction detection as soft metadata only (§15).

**Create** — `src/mapping/graph_builder.py`:

```python
def build_graph(places: list[Place], transitions: list[TransitionStats], store: ObservationStore, config) -> nx.Graph
def detect_junctions(graph: nx.Graph, places: list[Place], store: ObservationStore, config) -> dict[str, str]
def export_graph_json(graph: nx.Graph, path: Path) -> None
```

- [ ] Edges only from transitions with `confidence >= edge_confidence_threshold` **and** `forward_count + reverse_count >= minimum_edge_support` (§14 edge confidence: frequency + persistence + semantic compatibility + repeat observations). Edge attrs: `confidence`, `forward_count`, `reverse_count`, `supporting_observations`, `mean_visual_strength`.
- [ ] Directional evidence stored separately (§14 "Store A→B, B→A separately") as edge attrs; the routing graph itself remains undirected (NetworkX `Graph`).
- [ ] `detect_junctions` (§15 signals: multiple outgoing paths + scene-type changes + visual discontinuity + repeated traversal): `node_type` per place — `room / corridor / junction / stairs / elevator / entrance / unknown` — stored as node attribute, **never** used to hard-restrict routing (§15 "soft metadata").

**Config — extend `mapping:`**

```yaml
mapping:
  edge_confidence_threshold: 0.6
  junction_min_degree: 3
```

**Tests — `tests/test_graph_builder.py`**

- toy walkthrough Lobby–Corridor–Room101 → exact triangle path recovered;
- noisy sequence (Stage 11 noise fixture) → no spurious edge;
- junction detection: degree-3 node marked `junction`, corridor node marked `corridor`.

**Definition of done**

- [ ] graph correctly represents main accessible paths (§14 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_graph_builder.py -q`

**Suggested commit message:** `Stage-12: graph builder + junction detection`

---

### Stage 13 — Graph Validation

**Planner ref:** §16 (Stage 13) · **Milestone:** B

**Objective:** detect construction errors automatically before runtime (§16 required checks).

**Create** — `src/mapping/graph_validator.py`:

```python
@dataclass
class ValidationWarning:
    level: str            # "WARN" | "ERROR"
    check: str            # "disconnected_component" | "isolated_node" | "weak_edge" | "suspicious_node" | "duplicate_candidate"
    message: str

def validate_graph(graph: nx.Graph, places: list[Place], store: ObservationStore, config) -> list[ValidationWarning]
def write_validation_report(warnings, md_path, json_path) -> None
```

Checks (each maps to §16):

- **Connectivity:** number of connected components > 1 → WARN per extra component;
- **Isolated nodes:** degree-0 places with no physical evidence (no transitions, `supporting_observations == 0`);
- **Weak edges:** `confidence < weak_edge_threshold` or total support < `minimum_edge_support` → WARN listing edge;
- **Suspicious nodes:** observation count < `min_observations_per_place`; within-place std similarity > `high_variance_threshold`; scene types incompatible within the place;
- **Duplicate candidates:** pairs of places with mean exemplar similarity > `duplicate_similarity_threshold` and distinct identities → WARN "may be duplicates" (feeds Stage 10 tuning).

**Config — extend `mapping:`**

```yaml
mapping:
  weak_edge_threshold: 0.4
  min_observations_per_place: 3
  high_variance_threshold: 0.35
  duplicate_similarity_threshold: 0.9
```

**Modify**

- [ ] a `build_map.py` orchestrator script that chains Stages 02–13 into one CLI: `conda run -n ML python -m src.mapping.build_map` → observations → segments → places → reconcile → transitions → graph → validation → writes `data/map/graph_validation.json` + `.md` (example messages per §16: `WARN: Place 3 and Place 9 may be duplicates.`).

**Tests — `tests/test_graph_validator.py`**

- graph with an isolated node → warning;
- edge with support 1 → weak-edge warning;
- two near-identical places with different ids → duplicate warning;
- clean toy graph → no warnings.

**Definition of done**

- [ ] map quality inspectable without opening every frame (§16 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_graph_validator.py -q`; run `build_map` on provisional data → validation report exists.

**Suggested commit message:** `Stage-13: graph validation + build_map orchestrator`

---

### Stage 14 — Versioned Map Artifact

**Planner ref:** §17 (Stage 14) · **Milestone:** B

**Objective:** reproducible, reloadable map independent of build-time Python objects (§17).

**Create** — `src/mapping/map_artifact.py`:

```python
def write_map(map_dir: Path, *, map_id: str, building_id: str, floor_id: str,
              store: ObservationStore, places: list[Place], graph: nx.Graph,
              encoder: VisualEncoder) -> Path      # returns <map_dir>/<map_id>/
class MapBundle:
    @classmethod
    def load(cls, map_dir: Path) -> "MapBundle"
    # exposes: manifest, places, graph (networkx), exemplars (np.ndarray),
    # exemplar_place_ids (np.ndarray), place_names (dict), store (ObservationStore)
    def to_place_index(self) -> PlaceIndex          # bridge to legacy retrieval (Stage 15+)
```

Layout (§17):

```text
data/map/<map_id>/
    manifest.json        # map_id, building_id, floor_id, encoder, encoder_version,
                         # embedding_dimension, observation_count, place_count, edge_count,
                         # created_at, hash (sha256 of places.json+graph.json)
    places.json          # place records (id, name, segment_ids, obs ids, exemplars, scene, landmarks, stats)
    graph.json           # nodes with node_type + edges with confidence/directional counts
    exemplars.npy        # (M, D) aligned with exemplar_place_ids.npy
    exemplar_place_ids.npy
    observation_metadata.json
    vector_index/        # index.faiss + id_order.json
```

- [ ] `MapBundle.to_place_index()` builds a legacy `PlaceIndex` from `exemplars.npy` — the bridge that keeps `app.py`/`live_navigate.py`/baseline tests working while the new localization stages land.
- [ ] Legacy pickle `graph.pkl` continues to be written by the baseline path only.

**Tests — `tests/test_map_artifact.py`**

- write → fresh load: manifest fields correct; graph topology identical; exemplars identical;
- `to_place_index()` gives same query results as the legacy files they were built from.

**Definition of done**

- [ ] another process loads the map with no build-time objects (§17 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_map_artifact.py -q`; `build_map` writes a versioned map dir.

**Suggested commit message:** `Stage-14: versioned map artifact + MapBundle loader`

**⛔ Milestone checkpoints:** after Stage 14, run the Stage 01 benchmark with the *new* mapping (reuse `--provisional` labels) and record `reports/map_v2_report.json`. **Milestone B achieved** when places/graph compare favorably to the KMeans baseline (node purity, edge recall — see Stage 25 metric defs).

---

### Stage 15 — Candidate Retrieval

**Planner ref:** §18 (Stage 15) · **Milestone:** C

**Objective:** FAISS becomes a candidate **generator**, not the authority (§Rule 5).

**Create** — `src/localization/__init__.py`, `src/localization/retrieval.py`:

```python
@dataclass
class Candidate:
    place_id: str
    visual_score: float              # best exemplar similarity (0..1)
    best_exemplar_id: str
    supporting_exemplar_count: int   # exemplars of this place within top-K
    margin: float                    # best_score - second_best_score

@dataclass
class RetrievalResult:
    candidates: list[Candidate]      # sorted by visual_score desc
    best_score: float
    second_best_score: float
    score_margin: float

class CandidateRetriever:
    def __init__(self, store: ObservationStore, places: list[Place], top_k: int = 10): ...
    def retrieve(self, embedding: np.ndarray) -> RetrievalResult
```

- [ ] Search `top_k * 4` (min 10) exemplar rows (per-place exemplars from Stage 09), aggregate by place keeping best exemplar + supporting count; compute margin metrics (§18 important metrics).
- [ ] Legacy `PlaceIndex.query` stays untouched for baseline consumers.

**Config — new `localization:` section (grows through Stage 23):**

```yaml
localization:
  top_k: 10
  exemplar_search_factor: 4
```

**Tests — `tests/test_retrieval.py`**

- orthogonal exemplars → correct candidate ranks and margin > 0;
- place with 2 supporting exemplars in top-K → `supporting_exemplar_count == 2`;
- margin = best − second_best computed exactly.

**Definition of done**

- [ ] localization receives a meaningful candidate set, not one guess (§18 acceptance)

**Verify:** `conda run -n ML python -m pytest tests/test_retrieval.py -q`

**Suggested commit message:** `Stage-15: candidate retrieval (top-K per place)`

---

### Stage 16 — Candidate Scoring

**Planner ref:** §19 (Stage 16) · **Milestone:** C

**Objective:** combine multiple evidence types; start interpretable, don't overfit (§19).

**Create** — `src/localization/candidate_scoring.py`:

```python
@dataclass
class ScoredCandidate(Candidate):
    visual_term: float
    semantic_term: float
    temporal_term: float
    graph_term: float
    total: float

def score_candidates(result: RetrievalResult, query_tags: SceneTags | None,
                     query_objects: list[DetectedObject] | None,
                     previous_place_id: str | None, graph: nx.Graph,
                     config) -> list[ScoredCandidate]
```

- `total = w_visual·visual_term + w_semantic·semantic_term + w_temporal·temporal_term + w_graph·graph_term` with weights from config, all terms normalized to [0,1]:
  - `visual_term` = candidate.visual_score (multi-exemplar already — §19 "compare against several place exemplars");
  - `semantic_term` = scene match (1 if scene_type equal, 0.5 compatible, else 0) + landmark Jaccard, combined `0.5·scene + 0.5·landmark_jaccard`; when query tags absent → 0.5 (neutral, §19 "interpretable normalized terms");
  - `temporal_term` = 1 if place == previous_place_id else decayed by elapsed time/similarity to neighbors (placeholders wired in Stage 17/18);
  - `graph_term` = 1 if neighbor of previous place else soft decay (filled by Stage 17).

**Config — extend `localization:`**

```yaml
localization:
  w_visual: 0.5
  w_semantic: 0.25
  w_temporal: 0.15
  w_graph: 0.1
```

**Tests — `tests/test_candidate_scoring.py`**

- ambiguous visual (two equal scores), strong semantic for one → semantic winner (§19 acceptance);
- missing query tags → neutral terms, no crash.

**Verify:** `conda run -n ML python -m pytest tests/test_candidate_scoring.py -q`

**Suggested commit message:** `Stage-16: interpretable candidate scoring`

---

### Stage 17 — Graph-Constrained Candidate Filtering

**Planner ref:** §20 (Stage 17) · **Milestone:** C

**Objective:** physically implausible candidates penalized, never hard-rejected (§20 "Do not hard-reject").

**Create** — `src/localization/graph_constraints.py`:

```python
def local_candidate_set(previous_place_id: str | None, graph: nx.Graph, radius: int) -> set[str] | None
def graph_penalty(candidate_place_id: str, previous_place_id: str | None,
                  graph: nx.Graph, strength: float) -> float   # in [0, 1]
def apply_graph_constraints(scored: list[ScoredCandidate], previous_place_id, graph, config) -> list[ScoredCandidate]
```

- [ ] Local candidate policy (§20): search set = previous place + its neighbors + graph radius ≤ `graph_radius` (default 2). Global candidates only when `local_confidence_poor` (max local score < `local_weak_threshold`).
- [ ] Soft penalty: `penalty = strength * (1 - exp(-shortest_graph_distance / 2))` — a place 3 hops away is penalized, not eliminated; a place unreachable keeps a full penalty.
- [ ] `graph_penalty` multiplies the candidate total (documented as a score shaper, not a filter).

**Config — extend `localization:`**

```yaml
localization:
  graph_radius: 2
  graph_penalty_strength: 0.5
  local_weak_threshold: 0.35
```

**Tests — `tests/test_graph_constraints.py`**

- previous=A, neighbors {B,C}, candidate Z 4 hops away → penalized but still present;
- local set for A includes A,B,C and radius-2 nodes; far nodes excluded from the *local* set but retrievable globally;
- unreachable candidate gets max penalty.

**Verify:** `conda run -n ML python -m pytest tests/test_graph_constraints.py -q`

**Suggested commit message:** `Stage-17: graph-constrained candidate filtering`

---

### Stage 18 — Probabilistic State Estimator

**Planner ref:** §21 (Stage 18) · **Milestone:** C

**Objective:** Bayesian belief over places; graph/temporal priors; explicit unknown mass (§21, §Rule 4). **No neural sequence model.**

**Create** — `src/localization/state_estimator.py`:

```python
class StateEstimator:
    def __init__(self, places, graph, transition_stats, config): ...
    def reset(self, prior: dict[str, float] | None = None): ...
    def update(self, scored: list[ScoredCandidate], query_tags, query_objects,
               timestamp: float) -> StateEstimate   # posterior + unknown_mass

@dataclass
class StateEstimate:
    posterior: dict[str, float]      # place_id -> P(S_t | O_1..t); may sum < 1
    unknown_mass: float              # 1 - sum(posterior)
    best_place_id: str | None
    best_score: float
    entropy: float                   # normalized Shannon entropy of posterior
```

Update (concrete math, §21):

- **Observation likelihood** `L_t(S)` from `ScoredCandidate.total` via softmax over candidate totals with temperature `likelihood_temperature` (default 0.1):
  `L_t(S) = exp(total_S / τ) / Σ_candidates exp(total_C / τ)`; candidates not in the top-K get `L = 0`.
- **Transition prior** `P(S_t | S_{t-1})`: from graph adjacency + `TransitionStats.confidence` — neighbors weighted by edge confidence normalized, self-transition `self_transition_prior` (default 0.7), non-neighbors `far_transition_prior` (default 0.05). If no graph: uniform over all places.
- **Filter update**: `posterior(S) ∝ L_t(S) · Σ_{S'} P(S|S') · prior(S')`, then `unknown_mass = max(0, 1 − Σ posterior)` (residual mass is genuine "I don't know", §Rule 4).
- `entropy`: normalized Shannon entropy of the posterior over places (feeds Stage 22).
- Prior carried across updates; `reset()` on LOST/REACQUIRING.

**Tests — `tests/test_state_estimator.py`**

- constant strong evidence for A → posterior concentrates on A, entropy ↓;
- weak uniform evidence → unknown_mass high;
- graph prior: equal-likelihood A and Z (Z far away) after previous=A → A wins;
- after reset, prior uniform.

**Verify:** `conda run -n ML python -m pytest tests/test_state_estimator.py -q`

**Suggested commit message:** `Stage-18: probabilistic state estimator (Bayes filter)`

---

### Stage 19 — State Machine

**Planner ref:** §22 (Stage 19) · **Milestone:** C

**Objective:** five explicit states; no long-lived false localization (§22 acceptance).

**Create** — `src/localization/tracker.py` (new `LocalizationTracker`; do NOT modify `src/live_tracker.py` — that stays as the baseline):

```python
STATE_TRACKING, STATE_UNCERTAIN, STATE_LOST, STATE_REACQUIRING, STATE_ARRIVED = "TRACKING", "UNCERTAIN", "LOST", "REACQUIRING", "ARRIVED"

class LocalizationTracker:
    def __init__(self, bundle: MapBundle, config, destination_id: str | None = None): ...
    def process_frame(self, observation: Observation | None, embedding: np.ndarray,
                      timestamp: float) -> dict     # status dict, see below
```

- [ ] **Status dict must stay backward-compatible** with what `app.py` / `live_navigate.py` consume: `place_id, place_name, confidence, changed, low_confidence, arrived, route, directions` — plus new keys `state`, `confidence_level`, `unknown_mass`, `decision` (Stage 27 log).
- [ ] State rules (concrete thresholds all from config):
  - `TRACKING`: best posterior > `tracking_threshold` (0.5) and entropy < `tracking_entropy` (0.6);
  - `UNCERTAIN`: evidence weak (best < tracking_threshold) but current place still plausible (its posterior ≥ `uncertain_floor` 0.2) — keep last place, mark low_confidence;
  - `LOST`: ≥ `lost_after` (3) consecutive observations with best posterior < `lost_threshold` (0.2) and no agreement → state LOST, `place_id=None`, route cleared;
  - `REACQUIRING`: from LOST, run global retrieval (Stage 21); when best posterior > `reacquired_threshold` (0.5) → TRACKING;
  - `ARRIVED`: destination place has best posterior > `arrived_threshold` (0.6) for `arrived_confirmations` (2) consecutive observations → ARRIVED (kept until a strong move signal, then TRACKING).
- [ ] `changed` only on **confirmed** place transitions (Stage 20 wiring).

**Tests — `tests/test_state_machine.py`** (synthetic 4-D orthogonal exemplars, toy graph Lobby–Corridor–Room101):

- strong sequence Lobby→Corridor→Room101 → TRACKING the whole way, ARRIVED at destination;
- weak garbage frames after TRACKING → UNCERTAIN → LOST within `lost_after` frames;
- recovery sequence → REACQUIRING → TRACKING.

**Verify:** `conda run -n ML python -m pytest tests/test_state_machine.py -q`

**Suggested commit message:** `Stage-19: localization state machine`

---

### Stage 20 — Transition Stabilization

**Planner ref:** §23 (Stage 20) · **Milestone:** C

**Objective:** no `A→B→A→B` flicker from weak evidence (§23).

**Create** — extend `src/localization/tracker.py` (or `src/localization/stabilization.py`):

- [ ] A place change is **confirmed** only when either:
  - the new place's best posterior > `transition_threshold` (0.65), **or**
  - the new place wins `transition_confirmation_count` (3) **consecutive** observations (posterior max for each).
- [ ] Until confirmed, keep previous place; keep the raw evidence in `pending_transition` state (visible in the decision log, Stage 27).
- [ ] Majority-vote `LiveLocalizer` remains the baseline (untouched).

**Config — extend `localization:`**

```yaml
localization:
  transition_confirmation_count: 3
  transition_threshold: 0.65
```

**Tests — `tests/test_stabilization.py`**

- `A A A B B B` → B confirmed exactly at the 3rd consecutive B (posterior below threshold);
- `A B A B A B` weak evidence → stays A, never confirms B;
- strong single B spike (posterior > 0.65) → immediate confirm.

**Verify:** `conda run -n ML python -m pytest tests/test_stabilization.py -q`

**Suggested commit message:** `Stage-20: transition stabilization (K-consecutive)`

---

### Stage 21 — Global Reacquisition

**Planner ref:** §24 (Stage 21) · **Milestone:** C

**Objective:** recover from wrong belief via two retrieval modes (§24).

**Create** — extend `src/localization/tracker.py` with modes:

- [ ] `LOCAL TRACKING` (default): retrieval restricted to local candidate set (Stage 17) — previous place + neighbors + radius;
- [ ] `GLOBAL REACQUISITION`: triggered when state is LOST **or** when best posterior < `global_reacquisition_after` (3) consecutive observations → whole-map retrieval (`local_candidate_set = None`); return to LOCAL once best posterior > `recovery_threshold` (0.45);
- [ ] mode transitions logged in the decision log; global retrieval must also accept that the user may have moved several places between observations (§20/§24 note).

**Config — extend `localization:`**

```yaml
localization:
  global_reacquisition_after: 3
  recovery_threshold: 0.45
```

**Tests — `tests/test_reacquisition.py`**

- deliberately inject a wrong belief (reset posterior to a wrong place) → feed frames from a different area → recovers within N observations;
- global retrieval finds a far-away place when local search would never (local search excludes it, global includes it).

**Verify:** `conda run -n ML python -m pytest tests/test_reacquisition.py -q`

**Suggested commit message:** `Stage-21: global reacquisition (LOCAL/GLOBAL modes)`

---

### Stage 22 — Confidence Calibration

**Planner ref:** §25 (Stage 22) · **Milestone:** C

**Objective:** confidence that correlates with correctness; not fake percentages (§25).

**Create** — `src/localization/confidence.py`:

```python
def estimate_confidence(estimate: StateEstimate, retrieval: RetrievalResult,
                        previous_place_id: str | None, graph: nx.Graph,
                        config) -> str   # "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
```

Inputs (§25 new confidence inputs): visual score, score margin, temporal agreement (same place as previous), graph agreement (neighbor), posterior entropy. Rules (all config thresholds):

- `UNKNOWN` if unknown_mass > `unknown_high_mass` (0.4);
- `HIGH` if best_score > `high_score` (0.6) **and** margin > `high_margin` (0.15) **and** entropy < `high_entropy` (0.4);
- `LOW` if best_score < `low_score` (0.3) or margin < `low_margin` (0.02);
- else `MEDIUM`.

**Config — extend `localization:`**

```yaml
localization:
  high_score: 0.6
  high_margin: 0.15
  high_entropy: 0.4
  low_score: 0.3
  low_margin: 0.02
  unknown_high_mass: 0.4
```

**Modify**

- [ ] `LocalizationTracker` status dict gains `confidence_level`.

**Tests — `tests/test_confidence.py`**

- high-score+high-margin → HIGH; low-margin → LOW; high unknown mass → UNKNOWN; boundaries respected.

**Verify:** `conda run -n ML python -m pytest tests/test_confidence.py -q`

**Suggested commit message:** `Stage-22: confidence calibration (HIGH/MEDIUM/LOW/UNKNOWN)`

---

### Stage 23 — Semantic Localization

**Planner ref:** §26 (Stage 23) · **Milestone:** C

**Objective:** object/VLM evidence shifts probability where appearance alone is ambiguous — as a secondary signal (§26).

**Actions**

- [ ] Wire detector + VLM outputs into `score_candidates` (Stage 16 semantic term) at runtime: query `SceneTags` + objects compared against place stored landmarks/scene_types.
- [ ] Robustness to detector/VLM errors (§26 "must be robust"): semantic term saturates (capped at `semantic_cap` 0.6), weights modest (`w_semantic ≤ 0.3`), absent metadata → neutral 0.5 (never blocks a candidate).
- [ ] Example test scenario from §26: two visually near-identical corridors; one has stored landmark "staircase on left" + "blue room sign", query matches → candidate with matching landmarks wins.

**Tests — `tests/test_semantic_localization.py`**

- identical visual scores, one matching landmark set → semantic winner;
- wrong/missing landmarks (simulated detector error) → visual-only behavior, no crash, no wild swing.

**Verify:** `conda run -n ML python -m pytest tests/test_semantic_localization.py -q`

**Suggested commit message:** `Stage-23: semantic evidence in localization`

**⛔ Milestone checkpoints:** after Stage 23, run the Stage 01 benchmark *through the new tracker* (sequence metrics on the 60 held-out frames): top-1/top-3, false jump rate, transition accuracy, reacquisition time, unknown precision. Record `reports/localization_v2_report.json`. **Milestone C achieved** when the new tracker beats baseline majority-vote on false-jump rate with comparable top-1.

---

### Stage 24 — Runtime Optimization

**Planner ref:** §27 (Stage 24) · **Milestone:** D

**Objective:** continuous navigation stays practical on a laptop/phone-class GPU (§27).

**Create** — `src/runtime/gate.py` (new `src/runtime/` package) or `src/localization/runtime_gate.py`:

```python
class RuntimeGate:
    # cheap quality/novelty gate: only when the scene meaningfully changed do
    # expensive models (detector/VLM/encoder) run
    def should_process(self, frame, last_processed_at: float, last_descriptor) -> tuple[bool, dict]
```

- [ ] Per-frame cheap gate: brightness/blur sanity (Stage 03 quality fn reused) + descriptor novelty vs last processed frame + `observation_interval` (min seconds between expensive evaluations). Expensive pipeline skipped when gate says no (status dict: `skipped=True`, carry last posterior).
- [ ] Config — new `runtime:` section (§35):

```yaml
runtime:
  observation_interval: 1.0
  quality_gate_enabled: true
  global_reacquisition_after: 3
```

**Modify**

- [ ] `live_navigate.py`: use `LocalizationTracker` + `RuntimeGate`; keep `--speak`, `--camera-index`, `--interval-seconds` flags working.
- [ ] Latency measurement: report p50/p95 of per-frame localization latency in the status/log.

**Tests — `tests/test_runtime_gate.py`**

- unchanged frames skip expensive path; changed frame processes; interval respected.

**Verify:** `conda run -n ML python -m pytest tests/test_runtime_gate.py -q`; `live_navigate.py --destination ... --camera-index N` boots (needs webcam — user can verify interactively).

**Suggested commit message:** `Stage-24: runtime gate + live pipeline optimization`

---

### Stage 25 — Evaluation Suite

**Planner ref:** §30 (localization metrics), §31 (graph metrics) · **Milestone:** E

**Objective:** metric runners that produce the research numbers.

**Create**

- `scripts/evaluate_map.py` — graph metrics (§31) given `heldout_labels.json` + the built map:
  - node purity (within-place obs label coherence), node duplicate rate, node merge error;
  - edge precision (% generated edges physically real — requires true area transitions from the walkthrough order), edge recall (% true transitions discovered);
  - graph fragmentation (unexpected disconnected components);
  - outputs `data/evaluation/graph_report.json/.md`.
- `scripts/evaluate_localization.py` — localization metrics (§30) on held-out **sequences**:
  - top-1/top-3 accuracy, segment accuracy (fraction of frames correctly localized), transition accuracy (A→B detected correctly), false jump rate, reacquisition time (observations to recover after forced LOST), unknown precision (when UNKNOWN, how often genuinely uncertain);
  - outputs `data/evaluation/localization_report.json/.md`.
- [ ] Both scripts accept `--mode baseline|v2` so every milestone comparison reuses the same runner.

**Tests — `tests/test_metrics.py`** (each metric unit-tested on tiny hand-built prediction/truth lists).

**Definition of done**

- [ ] metrics computable on unseen walkthroughs when the user provides them (§Rule 6)

**Verify:** `conda run -n ML python -m pytest tests/test_metrics.py -q`; run both scripts in `--mode baseline` on provisional labels → reports exist.

**Suggested commit message:** `Stage-25: evaluation suite (map + localization metrics)`

---

### Stage 26 — Ablations

**Planner ref:** §32 · **Milestone:** E

**Objective:** show exactly which complexity produces improvement (§32).

**Create** — `scripts/run_ablations.py`:

- [ ] Localization progression (§32): A nearest-neighbour → B +threshold → C +majority vote → D +semantic evidence → E +graph constraint → F +probabilistic temporal state → G +lost/reacquisition. Each level = a config preset (`localization.ablation_level`) toggling pipeline pieces — implemented with a single parametrized runner, no code duplication.
- [ ] Graph progression: A KMeans+transitions → B temporal segmentation → C +reconciliation → D +transition validation → E +semantic evidence → F +graph validation.
- [ ] Output `data/evaluation/ablation_report.json/.md` with per-level metric tables (reuse Stage 25 runners).

**Verify:** `conda run -n ML python scripts/run_ablations.py --mode v2` produces the report on provisional data.

**Suggested commit message:** `Stage-26: ablation runner + report`

---

### Stage 27 — Failure Analysis

**Planner ref:** §33 · **Milestone:** E

**Objective:** every failed sequence inspectable (§33).

**Create**

- [ ] `LocalizationTracker` writes a **decision log** `data/evaluation/decision_log.jsonl` — one record per processed observation with the full §33 schema: `timestamp, query_frame, top-K candidates, visual scores, semantic scores, previous_posterior, graph_neighbors, graph_penalties, final_posterior, state, confidence, unknown_mass, mode (local/global)`.
- [ ] `scripts/inspect_failures.py`: loads the log + held-out labels, finds frames where final place ≠ true area, renders a readable failure report (frame path, candidates with all scores, posterior — exactly the §33 example shape).

**Verify:** run a short tracked sequence on the 60 held-out frames → `decision_log.jsonl` non-empty; `inspect_failures.py` prints at least one readable failure/success record.

**Suggested commit message:** `Stage-27: failure analysis + decision log`

---

### Stage 28 — UI Integration

**Planner ref:** §36 (app.py / live_tracker.py entries) · **Milestone:** D

**Objective:** the demo surfaces the new localization with minimal churn (§Rule 1 — keep UI simple).

**Actions**

- [ ] `app.py`: `load_map()` loads a `MapBundle` (Stage 14); Live Mode + Single Photo tabs construct `LocalizationTracker` instead of `LiveTracker`; new status keys (`state`, `confidence_level`) rendered as a small status line; a `localization.mode: baseline|v2` config toggle switches between `LiveTracker` (legacy) and `LocalizationTracker`.
- [ ] `live_navigate.py`: same toggle; `--speak` still works.
- [ ] Keep: Streamlit tabs, floor-plan overlay, graph view, TTS, top-matches bar chart.
- [ ] Existing `tests/test_live_tracker.py` and `tests/test_navigate.py` still pass (legacy code untouched).

**Verify:** `conda run -n ML python -m pytest tests/ -q` (all green); `streamlit run app.py` boots (user checks UI).

**Suggested commit message:** `Stage-28: UI + live CLI on new localization tracker`

---

### Stage 29 — Final Cleanup & Docs

**Planner ref:** §2 (target structure), §36, §38/§39 (DoD) · **Milestone:** E

**Actions**

- [ ] `README.md`: rewrite pipeline description for the new architecture (keep the old pipeline section under "Legacy baseline").
- [ ] `requirements.txt`: final list (opencv-python, numpy, scikit-learn, torch, torchvision, faiss-cpu, networkx, matplotlib, streamlit, PyYAML, pyttsx3, ultralytics, bitsandbytes, transformers, accelerate).
- [ ] `config.yaml`: full §35 structure with comments; remove nothing the baseline needs.
- [ ] Move legacy modules into `src/baseline/` (imports updated in `app.py`/`live_navigate.py` only for the `mode: baseline` path) — or keep them in place if that's less churn; **no dead code left unreferenced**.
- [ ] Full pass: `pytest tests/ -q` all green; end-to-end smoke (`build_map` → benchmark `--mode v2` → `inspect_failures`).
- [ ] Final comparison report `data/evaluation/final_report.md`: baseline vs v2, all metrics, all ablation levels.

**Definition of done (final, §38/§39):** check both lists from the planner — graph subsystem and localization subsystem checklists must all be tickable.

**Suggested commit message:** `Stage-29: cleanup, docs, final report`

---

## 4. Baseline preservation rules (bind every stage)

1. Legacy files (`src/localize.py`, `src/live_tracker.py`, `src/build_places.py`, `src/build_graph.py`, `src/extract_frames.py` fixed-N path, pickle graph) are **never deleted**; backbone-changing CLIs get `--baseline`.
2. Every backbone change re-runs the Stage 01 benchmark in both modes and records the delta in `data/evaluation/reports/`.
3. New thresholds live in `config.yaml` only (Appendix F), never hardcoded.
4. Detector/VLM output is evidence, never ground truth for coordinates/topology/final localization (§Rule 3).
5. Unknown > confidently wrong: every stage must support the UNKNOWN/UNCERTAIN path (§Rule 4).
6. Evaluation only on observations not used to build the map (§Rule 6).

## 5. Appendix A — Existing file → planned role map (§36)

| Existing file | Role after migration |
|---|---|
| `src/extract_frames.py` | default = smart sampling (Stage 03); `--baseline` = fixed-N |
| `src/embedder.py` | compat wrapper over `VisualEncoder` (Stage 04) |
| `src/embed_frames.py` | also builds Observations + ObservationStore (Stages 02/05) |
| `src/build_places.py` | `--baseline` KMeans; default → place_builder (Stage 09) |
| `src/build_graph.py` | `--baseline` transition-count; default → graph_builder (Stage 12) |
| `src/localize.py` | baseline `PlaceIndex`/`LiveLocalizer`; bridged via `MapBundle.to_place_index` (Stage 14) |
| `src/live_tracker.py` | baseline tracker; status-dict contract preserved by `LocalizationTracker` (Stage 19) |
| `src/navigate.py` | unchanged (§Rule 1) |
| `src/floor_plan.py`, `src/label_places.py`, `src/speak.py`, `src/utils.py` | unchanged |
| `app.py`, `live_navigate.py` | `mode: baseline\|v2` toggle (Stage 28) |
| `evaluate.py` | kept; superseded by Stage 25 runners |

## 6. Appendix B — Dataset plan

- **Now (provisional):** 301 untracked frames in `data/frames/`, no source video. Stage 01 labels 60 held-out frames by viewing them.
- **When the user provides videos:** place at `data/raw/<name>.mp4`; update `config.yaml paths.video`; re-run the mapping pipeline for `building_id`/`floor_id`; record **two** videos — mapping walkthrough + independent evaluation walkthrough (§3 instruction 2/3); re-label and re-benchmark (reports become non-provisional).
- Real-video stage 03 check: compare smart-sampled frame count vs fixed-N on the same walkthrough.

## 7. Appendix C — Metric definitions (§30/§31)

Localization: top-1 / top-3 accuracy · segment accuracy (frames of a true area correctly localized) · transition accuracy (true A→B movement detected) · false jump rate (place change without true area change) · reacquisition time (observations to recover after forced LOST) · unknown precision (when UNKNOWN/UNCERTAIN, fraction genuinely uncertain).

Graph: node purity · node duplicate rate (one physical place → many nodes) · node merge error (two places merged) · edge precision · edge recall · graph fragmentation (unexpected disconnected components).

## 8. Appendix D — Ablation matrix (§32)

Localization: A nearest-neighbour → B +threshold → C +majority vote → D +semantic → E +graph → F +probabilistic temporal → G +lost/reacquisition.
Graph: A KMeans+transitions → B temporal segmentation → C +reconciliation → D +transition validation → E +semantic → F +graph validation.

## 9. Appendix E — Decision-log schema (§33)

`{"timestamp", "query_frame", "mode", "state", "confidence_level", "unknown_mass", "candidates": [{"place_id", "visual", "semantic", "graph_penalty", "total", "posterior"}], "previous_posterior", "graph_neighbors", "final_posterior", "arrived", "route"}` — one JSON object per line in `data/evaluation/decision_log.jsonl`.

## 10. Appendix F — Final config schema (§35)

```yaml
paths:      # + observations_dir, evaluation_dir, map_id
sampling:   # min_interval_seconds, novelty_threshold, transition_threshold, transition_window,
            # transition_keep_after, blur_threshold, dark_threshold
perception: # detector_enabled/model/confidence, vlm_enabled/model/quantization/max_tokens
embedding:  # model, image_size
mapping:    # segment_*, merge_*, transition_persistence, minimum_edge_support,
            # edge_confidence_threshold, junction_min_degree, weak_edge_threshold,
            # min_observations_per_place, high_variance_threshold, duplicate_similarity_threshold,
            # exemplar_method, max_exemplars_per_place
localization: # top_k, exemplar_search_factor, w_*, graph_radius, graph_penalty_strength,
            # local_weak_threshold, tracking_threshold, tracking_entropy, uncertain_floor,
            # lost_after, lost_threshold, reacquired_threshold, arrived_threshold,
            # arrived_confirmations, transition_confirmation_count, transition_threshold,
            # global_reacquisition_after, recovery_threshold, confidence thresholds,
            # semantic_cap, mode (baseline|v2), ablation_level
runtime:    # observation_interval, quality_gate_enabled
routing:    # weighted (existing)
live:       # existing
navigation: # existing (legacy baseline settings)
```

## 11. Appendix G — Stage dependency graph

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14
                                                                    │
14 → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 27
                                                │     │     │
                                                └─► 28 ◄──┘
                                                │
                                                29
```

Parallelizable pairs (after both deps exist): (06, 07) · (25, 26, 27). Stages 06/07 need the Observation pipeline (02–05) to be wired, not to be perfect.

---

*End of executor document. Next turn: start at the lowest-numbered unchecked stage (Stage 00) and work through in order, ticking the table as each completes.*
