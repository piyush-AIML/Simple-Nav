# SimpleNav — Final Planner (v3): Embedding Upgrade + Product-Completeness Closure

> **This is a full replacement for `LATEST-SimpleNav_FINAL_GRAPH_LOCALIZATION_PLANNER_v2.md`.**
> v2's Stages 0–28 (Milestones A–C + the perception-model fix) are done and
> verified — this document does not repeat their spec text. What v3 adds,
> based on a fresh file-by-file audit of the actual code (not the docs
> describing it):
>
> 1. **Eight confirmed bugs/gaps that v2 did not know about**, found by
>    tracing every embedding call site end to end (§2). The most important:
>    the encoder is not actually swappable today, and the demo app does not
>    run the pipeline that Stages 15–24 were built for.
> 2. **A concrete, evidence-based encoder upgrade** — DINOv2-with-Registers
>    (default) with a documented, gated DINOv3 stretch option — instead of
>    the "optional, someday" note in v2 §5.4 (§4).
> 3. **A recalibration protocol**, because the project's own code comments
>    admit its similarity thresholds are tuned to ResNet18's distribution
>    (§5).
> 4. **A product-unification stage** closing the gap between what the
>    tracker can do and what a user actually sees (§6).
> 5. **A validation/risk register** for every change in this document (§9) —
>    the "multi-loop" methodology used to produce it is in §1.
> 6. An updated Definition of Done (§10) that supersedes v2 §16.
>
> Rules 1–6 from v2 are unchanged and still govern every stage below. One
> rule is added (§0).

---

# 0. Rule 7 — New

**Rule 7 — The product a user runs must be the product that was tested.**
A feature is not "done" because a stage's unit tests pass; it is done when
the entry point a person actually launches (`app.py`, `live_navigate.py`)
exercises it. This audit found three Stage 15–24 features (Bayes tracker,
semantic scoring, runtime gate) that are fully implemented, fully tested,
and completely unreachable from the running application. Rule 7 exists so
that gap can't recur.

---

# 1. Methodology — How This Audit Was Done (the "multi-loop" pass)

Three passes over the real repository (cloned and read file-by-file, not
inferred from README/STATE.md prose), each validating the previous:

**Loop 1 — Inventory.** Read every file under `src/`, `scripts/`, root-level
entry points, `config.yaml`, `tests/`, and all five planning documents.
Built the module map in §2's table from primary evidence (grep + full
reads), not from STATE.md's own claims about itself.

**Loop 2 — Cross-check.** For every "done" claim in `STATE.md` / planner v2,
traced the actual call graph to confirm it end-to-end. This is what caught
the gap between claim and reality: e.g. `STATE.md` §2.4 lists "DINOv2
encoder comparison" as merely "optional, low priority" — Loop 2 found that
the config knob it would use (`embedding.model`) doesn't actually reach the
embeddings that get stored, which promotes this from "nice to have" to
"prerequisite bug" (§2.1). Every finding in §2 below cites the exact
file/line evidence, not a paraphrase of a doc.

**Loop 3 — Solution validation.** Every fix proposed in §4–§8 is checked
against three constraints before being accepted into this plan: (a) Rule 1
(does it touch frozen Stage 0–22 code unnecessarily?), (b) the 6 GB VRAM
budget, (c) does it introduce a new unverifiable claim the way the ones in
§2 were introduced? §9 is the artifact of this loop — a risk row per
change, each with its own falsifiable check, so this plan doesn't repeat
the mistake it's fixing.

---

# 2. Confirmed Findings — File-by-File Audit

## 2.1 Module state table

| Module | State | Evidence |
|---|---|---|
| `src/mapping/*`, `src/localization/*` (Stages 8–22) | ✅ solid | Clean interfaces, graceful degradation (Rule 4) throughout, well-tested |
| `src/perception/detector.py`, `scene_tagger.py` (Stages 6–7, v2 §5) | ✅ solid | `LFM2VLTagger` fully implemented incl. true batching; fallback chain correct |
| `src/localization/semantic_scoring.py` (Stage 23) | ✅ implemented, ⚠️ under-validated | Well-designed Rule-4 asymmetries; but see §2.7 — the acceptance metric (visually-ambiguous subset) was never finished |
| `src/runtime/gate.py` (Stage 24) | ✅ implemented, ❌ **not wired in** | See §2.4 |
| `src/embeddings/encoder.py` | ✅ clean registry, ❌ **the only encoder ever registered is ResNet18, and the registry is bypassed by the actual embedding step** | See §2.1(a) below |
| `src/embed_frames.py` | ❌ **hardcoded to ResNet18, ignores `config.embedding.model`** | §2.1(a) |
| `src/mapping/observation_store.py` | ❌ **hardcodes `"model": "resnet18"` in `encoder.json`** | §2.2 |
| `src/mapping/build_map.py` | ✅ correctly config-driven **for metadata only** | §2.1(a) |
| `app.py`, `live_navigate.py`, `src/localize.py`, `src/live_tracker.py` | ⚠️ working, but **on an entirely different, older pipeline** than Stages 15–24 | §2.3 |
| `scripts/benchmark_baseline.py` | ✅ correct — intentionally frozen (Rule 2) | — |
| `scripts/compare_encoders.py` | ❌ **does not exist**, despite being referenced by `encoder.py`'s own docstring and required by v2 §5.4's own acceptance rule | §2.6 |
| `src/utils.py::validate_config` | ⚠️ checks detector/VLM model ids and loss weights, **not** `embedding.model` | §2.5 |
| `config.yaml` similarity thresholds (`mapping.merge_visual_threshold`, `merge_visual_extra_threshold`, `localization.*_threshold`, `likelihood_temperature`) | ⚠️ explicitly calibrated to ResNet18's similarity distribution, by the codebase's own comment | §2.8 |
| `DEMO_RUN_OF_SHOW.md` | ❌ **describes a demo `app.py` cannot perform**, and cites a stale place/edge count | §2.3, §2.9 |
| `STATE.md` §2.1 (non-monotonic ablation) | ⚠️ open finding, diagnostic plan written but not executed | §2.7 |
| Test suite (156 passing, 7 slow) | ✅ real and current | Re-confirmed by reading `tests/` directly |

## 2.2 Finding A (critical) — The encoder is not actually swappable

`src/embed_frames.py::embed_frames()` calls `src.embedder.embed_image()`
for every frame (line 36). `embed_image()` **hardcodes**
`ResNet18Encoder()` (`src/embedder.py:92-94`) — it does not read
`config["embedding"]["model"]` at all. This is the function that produces
the vectors written into `data/observations/embeddings.npy`, which is what
`ObservationStore` (and therefore `build_map.py`, `evaluate_suite.py`, and
the versioned map) actually consumes.

Meanwhile, `build_map.py` line 121 calls `get_encoder(config)` — the
config-driven registry — but **only to read `.name`/`.version`/`.dimension`
for the map manifest's metadata fields.** It never re-embeds anything.

**Consequence, demonstrated, not hypothesized:** today, if you change
`config.yaml`'s `embedding.model` from `resnet18` to anything else, nothing
happens to the actual vectors. Worse, if a second encoder were registered
(as v2 §5.4 suggested "optionally" doing), the map manifest would silently
**lie** about which model produced the stored embeddings — `manifest.json`
currently reads `"encoder": "resnet18"` (confirmed by reading
`data/map/college_env_v1/manifest.json`) purely because the config also
happens to say `resnet18` right now, not because anything enforces the two
staying in sync.

This makes the encoder registry (`src/embeddings/encoder.py`) currently
**decorative** — a clean interface with nothing wired to it. Fixing this is
a prerequisite for any encoder swap, including the one this document
proposes in §4. It is Stage 31 below.

## 2.3 Finding B — `ObservationStore.save()` independently hardcodes the encoder name

`src/mapping/observation_store.py:107-112` writes
`{"model": "resnet18", "dimension": ...}` to `encoder.json` unconditionally.
The `encoder_name` parameter threaded through
`embed_frames.build_observations()` and `save_observations_dir()` is
accepted but **never used** in either function body — dead parameters that
currently give false confidence they do something. Two independent
hardcodes of the same fact (here and in Finding A) means fixing one without
the other leaves a silent mismatch.

## 2.4 Finding C (critical) — The demo app and the live loop run a different pipeline than Stages 15–24 were built for

`app.py` imports `PlaceIndex` (`src/localize.py`) and `LiveTracker`
(`src/live_tracker.py`) — the Milestone-A legacy stack — and calls
`embed_image()` directly (lines 28, 132, 202). `live_navigate.py` does the
same (lines 25, 95). Neither imports `MapBundle`
(`src/mapping/map_artifact.py`) or `LocalizationTracker`
(`src/localization/tracker.py`) — the stack that Stages 15–22 (candidate
retrieval, semantic scoring, the Bayes state estimator, the
TRACKING/UNCERTAIN/LOST/REACQUIRING/ARRIVED state machine, confidence
calibration) were actually built for and which `evaluate_suite.py` is the
**only** consumer of.

**Practical effect:** semantic evidence (Stage 23), the probabilistic state
estimator (Stage 18), the state machine (Stage 19), and confidence levels
(Stage 22) — four stages this project spent real engineering effort on —
never run when a person opens the app or points a webcam at the building.
They only run inside an offline evaluation script. `STATE.md` §2.4 calls
this "noted, not a gap" — this audit disagrees: for a system whose whole
value proposition is "appearance + semantic + temporal + graph evidence
combined," shipping a demo that only uses appearance is not a cosmetic
omission.

**This is corroborated independently by `DEMO_RUN_OF_SHOW.md` itself**
(§2.9) — it instructs the presenter to demonstrate things `app.py` cannot
do.

## 2.5 Finding D — the runtime gate (Stage 24) is built, tested, and never called

`src/runtime/gate.py::should_process()` has a real, passing unit test
(`tests/test_runtime_gate.py`) but is imported nowhere outside its own test
file (confirmed by repo-wide grep). `live_navigate.py`'s only throttling is
a flat `time.time() - last_capture_time >= interval` check
(`live_navigate.py`, main loop) — it never touches `should_process()`.
`DEMO_RUN_OF_SHOW.md`'s closing section claims "the gate skips redundant
frames" as a live-demo talking point; no runnable code path does that
today.

## 2.6 Finding E — `validate_config()` has a blind spot exactly where this plan needs it not to

`src/utils.py::validate_config()` checks `perception.detector_model` and
`perception.vlm_model` against known-model tuples, and sanity-checks the
four localization weights. It does **not** check `embedding.model` against
the encoder registry at all. Today this is low-stakes (there's only one
registered encoder). Once §4 adds a second one, an unregistered/misspelled
value would pass "config valid" and only fail deep inside `build_map.py`
after segmentation, place formation, and reconciliation have already run —
wasted compute and a worse failure mode than the fast, first-line check
`validate_config()` already gives the detector/VLM choices.

## 2.7 Finding F — `scripts/compare_encoders.py` doesn't exist

`src/embeddings/encoder.py`'s own module docstring says new encoders
"can be added behind this interface and compared with
`scripts/compare_encoders.py`-style metrics." Planner v2 §5.4 repeats this:
"run the comparison script Stage 5 already specified... gate any swap on
that measurement, don't swap because it's generally better." **The script
does not exist anywhere in the repository.** The project's own rule for
approving an encoder swap has never been enforceable. §7 below builds it.

## 2.8 Finding G — the open ablation finding (STATE.md §2.1) is unresolved, and matters more after an encoder swap

`STATE.md` already documents that semantic evidence currently *hurts*
top-1 (0.761 visual-only → 0.684 with semantic added) and flags this as the
main open item, with a specific next measurement planned (the
"visually-ambiguous subset" top-1 comparison) but not executed. Two root
causes are plausible per STATE.md's own hypothesis list: circular
pseudo-labels, and sparse/flickery landmark yield (18/301 observations with
non-empty landmarks, 72 scene-type flips pre-debounce). This audit adds a
third consideration relevant to §4: **if the visual signal's own quality
changes (better embeddings), the semantic term's relative contribution and
the weights in `config.yaml` (`w_visual: 0.5`, `w_semantic: 0.25`) need
re-examination together, not in isolation** — improving one term changes
where the other's problems become visible. This is folded into Stage 36
rather than treated as a separate, disconnected task.

## 2.9 Finding H — the config-encoded evidence that thresholds are ResNet18-specific

`src/mapping/place_reconciliation.py` line ~136 contains this comment,
verified verbatim in the file:

> `# both places lack landmarks -> merge only when near-identical (0.92;`
> `# same-building cross-area ResNet18 similarity is ~0.87)`

This is the project's own documentation that `merge_visual_extra_threshold:
0.92` (and by the same logic, `merge_visual_threshold: 0.75`,
`navigation.confidence_threshold: 0.40`, and every `localization.*`
threshold expressed on a raw cosine-similarity scale) is tuned to ResNet18's
specific similarity distribution — not a universal constant. A different
encoder architecture (self-supervised ViT vs. supervised CNN) produces a
different similarity distribution over the same building. Carrying these
numbers over unchanged after an encoder swap is a **correctness risk**, not
a tuning nicety. §5 makes this an explicit, gated stage.

## 2.10 Finding I — `DEMO_RUN_OF_SHOW.md` is stale against the current map artifact

The document says "7 places, 6 edges, 0 validation warnings." The current
`data/map/college_env_v1/manifest.json` (rebuilt with the real VLM stack,
per `STATE.md` §1) shows **5 places / 4 edges** with **1 validation
warning**. This is a small thing next to Findings A–D, but it means the
demo script itself would embarrass a presenter reading it verbatim, and it
is fixed in the same hygiene pass as Finding C (Stage 35).

## 2.11 Finding J — minor tech debt (fold into Stage 31/32, not separate work)

- `LFM2VLTagger._do_load()` uses `torch_dtype=torch.bfloat16`; the
  planner v2 §5.2 reference snippet itself already uses the newer `dtype=`
  kwarg. Harmless today, a deprecation warning waiting to happen on a
  `transformers` bump. Fix opportunistically when touching this file for
  any other reason — not worth its own stage.
- Camera index default (`live.camera_index: 1`) is an OS-dependent
  assumption, already flagged in the README; unchanged priority.

---

# 3. Why This Changes the Priority Order

v2 treated the encoder question as optional, last-priority polish (§5.4:
"Low priority; do this only after Stages 23–30 are done"). Finding A (§2.2)
changes that: **the encoder pipeline has a latent correctness bug today,
independent of whether anyone ever swaps the model.** The `encoder_name`
dead parameters and the hardcoded `"resnet18"` string in
`observation_store.py` are landmines for future-you (or a teammate) the
first time a second encoder is registered without reading this document.
Stage 31 (fixing the wiring) is therefore promoted ahead of Stage 32
(picking the new model) — fix the pipe before changing what flows through
it.

---

# 4. Stage 32 — Encoder Selection v3: DINOv2-with-Registers (default)

## 4.1 Candidates considered, and why

| | ResNet18 (current) | DINOv2-small | **DINOv2-with-Registers-Small (recommended default)** | DINOv3 ViT-S+/16 (documented stretch option) |
|---|---|---|---|---|
| Params | 11.7M | 21M | 21M | 29M |
| Embedding dim | 512 | 384 | 384 | 384 |
| Training objective | Supervised, ImageNet-1k classification | Self-supervised (DINO + iBOT), 142M curated images | Same as DINOv2-small + register tokens (removes attention-map artifacts) | Self-supervised, 1.7B images, RoPE + SwiGLU, Gram-anchored dense features |
| License | Public (torchvision) | **Apache 2.0, ungated** | **Apache 2.0, ungated** | Meta's DINOv3 license — gated, manual approval (confirmed: can take **"up to a few days"**, per Meta's own FAQ, and approval is not guaranteed) |
| Viewpoint/lighting invariance (what this project needs — same place, different angle) | Weak — trained for object categories, not instance/place retrieval | Strong | Strong, with cleaner attention (fewer background artifacts than plain DINOv2) | Strongest — Meta reports +10.8 pts (Met) / +7.6 pts (AmsterTime) over DINOv2 on landmark/place retrieval benchmarks specifically |
| Fits 6 GB VRAM alongside YOLO26n + LFM2.5-VL-450M | Yes (already does) | Yes, trivially (~85 MB fp32) | Yes, trivially | Yes, trivially (~115 MB fp32) |
| Setup friction | None | None | None | **HF gated repo — needs account, license acceptance, manual Meta approval before first download** |

## 4.2 Decision

**Default: `facebook/dinov2-with-registers-small`.** It is a strict
upgrade over plain `facebook/dinov2-small` at zero extra parameter cost
(same 21M/384-d) — the register tokens fix the attention-map artifacts the
original DINOv2 paper's own follow-up work (*Vision Transformers Need
Registers*, Darcet et al.) identified, which otherwise show up as spurious
high-norm patches that can distort pooled/CLS-token similarity in exactly
the way this project measures place similarity. It is ungated,
Apache-2.0-licensed, and downloads immediately — consistent with Stage 30's
"zero manual patching" fresh-clone requirement (§8.1).

**Documented, optional stretch option: DINOv3 ViT-S+/16**
(`facebook/dinov3-vits16plus-pretrain-lvd1689m`). Meta's own reported gains
are specifically on **landmark/place-retrieval-style benchmarks** — the
closest published proxy to "is this the same room from a different angle,"
which is this project's exact task. This is a real, evidence-backed reason
to prefer it over DINOv2 *if* the gated access is available in time — but
the license terms (manual approval, "up to a few days," not guaranteed) are
incompatible with a submission on a fixed deadline being the *default*
path. This is exactly the situation Rule 4 and the project's own §5.4
selection criterion exist for: **do not make an uncertain, externally-gated
dependency the required path.** Register it behind the same interface
(§4.3), gate its adoption on `scripts/compare_encoders.py` (§7) if/when
access is granted, per the same "measure, don't assume" rule the project
already applies to itself.

Do **not** delete `ResNet18Encoder` — it remains the Milestone-A baseline
per Rule 2, registered and reachable, never the default once §4 lands.

## 4.3 Implementation

Add to `src/embeddings/encoder.py` (same file, same registry pattern —
`ResNet18Encoder` is untouched):

```python
class Dinov2RegistersEncoder(VisualEncoder):
    """DINOv2 with register tokens — self-supervised ViT-S/14 features.
    Default encoder as of planner v3 §4. Strong viewpoint/lighting
    invariance for the same-place-different-angle retrieval task this
    project actually does (unlike ResNet18's ImageNet-classification
    objective). Ungated, Apache-2.0 (see planner v3 §4.1 for why this
    is preferred over the gated DINOv3 option)."""

    name = "dinov2_registers_small"
    version = "facebook/dinov2-with-registers-small"
    dimension = 384

    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            cls._processor = AutoImageProcessor.from_pretrained(cls.version)
            cls._model = AutoModel.from_pretrained(cls.version)
            cls._model.eval()
            if torch.cuda.is_available():
                cls._model = cls._model.to("cuda")
            logger.info(f"DINOv2-with-Registers loaded: {cls.version}")
        return cls._processor, cls._model

    def _forward(self, pil_images: list[Image.Image]) -> np.ndarray:
        import torch

        processor, model = self._load()
        inputs = processor(images=pil_images, return_tensors="pt")
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS-token embedding via the model's pooler — this is the usage
        # HuggingFace's own DINOv2/DINOv3 model cards recommend for
        # whole-image retrieval (as opposed to the per-patch tokens used
        # for dense/segmentation tasks). Validate this choice empirically
        # with scripts/compare_encoders.py (Stage 33) rather than assuming
        # it — if pooler_output ever underperforms raw
        # last_hidden_state[:, 0, :] on the same-place-consistency metric,
        # switch to that instead; both are one-line changes here.
        vecs = outputs.pooler_output.cpu().numpy().astype("float32")
        return vecs

    def encode(self, image: ImageInput) -> np.ndarray:
        vec = self._forward([_to_pil(image)])[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def batch_encode(self, images: list[ImageInput]) -> np.ndarray:
        vecs = self._forward([_to_pil(img) for img in images])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class Dinov3Encoder(VisualEncoder):
    """DINOv3 ViT-S+/16 — documented, opt-in stretch option (planner v3
    §4.1). Requires a Hugging Face account with the gated DINOv3 license
    accepted AND approved for facebook/dinov3-vits16plus-pretrain-lvd1689m
    (approval is manual and not instant — see planner v3 §4.1). NOT the
    default: do not select this in config.yaml unless access is already
    confirmed working, or build_map.py's fail-fast fallback (unchanged
    pattern from get_detector/get_scene_tagger) will silently drop to
    whatever encoder is next in the registry."""

    name = "dinov3_vits16plus"
    version = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    dimension = 384
    # implementation mirrors Dinov2RegistersEncoder exactly (same
    # AutoImageProcessor/AutoModel/pooler_output pattern) — omitted here
    # to avoid duplicating ~25 lines; factor the shared body into a
    # private _HFViTEncoder base class shared by both, the same way
    # scene_tagger.py already shares _HFVisionTagger across VLM backends.


_REGISTRY: dict[str, type[VisualEncoder]] = {
    "resnet18": ResNet18Encoder,
    "dinov2_registers_small": Dinov2RegistersEncoder,
    "dinov3_vits16plus": Dinov3Encoder,  # opt-in only, see class docstring
}
```

**Preprocessing note (do not skip):** `TRANSFORM` in `src/embedder.py` is
ResNet18/torchvision-specific (224×224 resize, ImageNet mean/std). The new
encoders **must** use their own `AutoImageProcessor` (loaded from the same
`version` string, shown above) — that processor already encodes the
correct resize/crop/normalization for that model. Do not reuse `TRANSFORM`
for anything except `ResNet18Encoder`.

## 4.4 Config diff

```yaml
embedding:
  model: "dinov2_registers_small"   # was: "resnet18"
  # Rollback: set back to "resnet18" — no other change needed, both
  # encoders stay registered (Rule 2-style: don't delete the baseline).
  # Optional, gated stretch (see planner v3 §4.1 before enabling):
  # model: "dinov3_vits16plus"
```

## 4.5 Acceptance criteria

- [ ] `Dinov2RegistersEncoder` registered, passes the same test shape as
  `test_encoder.py`'s existing `ResNet18Encoder` tests (dimension 384,
  L2-normalized, all three input types accepted, batch == sequential).
- [ ] `scripts/compare_encoders.py` (Stage 33, §7) run on the current
  301-observation set; DINOv2-with-Registers shows equal-or-better
  same-place viewpoint consistency and different-place separability than
  ResNet18 on that measurement — this is the actual gate, not "DINOv2 is
  generally better."
- [ ] Map rebuilt end-to-end with the new encoder (requires Stage 31 fixed
  first — §6); manifest's `encoder`/`encoder_version`/`embedding_dimension`
  match what was actually used, verified by re-reading `manifest.json`
  after the build, not assumed.

---

# 5. Stage 34 — Threshold Recalibration Protocol

**Goal:** don't let Finding G (§2.9) become a silent regression after §4
lands. This is a repeatable *protocol*, not a one-time hand-tune — every
future encoder swap re-runs it.

1. After rebuilding the map with the new encoder (§6), run
   `scripts/compare_encoders.py` (§7) to get the new encoder's own
   same-place / different-place similarity distributions (mean + std for
   each).
2. For every raw-cosine-similarity-scale constant in `config.yaml`
   (`mapping.merge_visual_threshold`, `merge_visual_extra_threshold`,
   `navigation.confidence_threshold`, `localization.tracking_threshold`,
   `lost_threshold`, `high_score`, `low_score`, `likelihood_temperature`),
   recompute it as the same *percentile* of the new distribution that the
   current value represents in ResNet18's distribution, rather than copying
   the number. (E.g. if `0.92` currently sits at the 97th percentile of
   ResNet18 same-building cross-area similarity, the DINOv2 replacement is
   whatever value sits at the 97th percentile of DINOv2's same-building
   cross-area similarity — not literally `0.92`.)
3. Re-run `evaluate_suite.py`'s ablation (Stage 27, already built) with the
   recalibrated thresholds; compare top-1/top-3/segment-accuracy against
   the ResNet18 baseline **on the same 301-observation pseudo-label set**
   so the comparison isn't confounded by also changing the dataset.
4. Write the before/after numbers into `STATE.md`, explicitly labeled
   "encoder swap, pseudo-label self-consistency" (Rule 6 — never present
   this as real-world accuracy either).

**Acceptance criteria:**
- [ ] A recalibrated threshold table exists in `STATE.md`, each value
  traceable to a specific percentile measurement, not hand-picked.
- [ ] Ablation re-run shows no unexplained regression vs. the ResNet18
  baseline; any regression that does appear is written up the same way
  Finding G was (hypothesis + planned next measurement), not silently
  accepted.

---

# 6. Stage 31 — Fix the Encoder Pipeline (prerequisite for Stage 32)

This must land **before** §4's new encoder class is used for anything real
— otherwise Finding A just gets a second, better-hidden instance.

1. **`src/embed_frames.py::embed_frames()`**: replace the hardcoded
   `from src.embedder import embed_image` / per-frame loop with
   `get_encoder(config)` and its `batch_encode()` — this also finally gives
   the offline embedding pass the same batching treatment Stage 24 already
   gave the detector and VLM (`src/embed_frames.py` currently embeds one
   frame at a time even though `attach_detections`/`attach_scene_tags`
   next to it are already batched — an inconsistency worth closing in the
   same pass).

    ```python
    def embed_frames(frames_dir: Path, config: dict) -> tuple[np.ndarray, list[str]]:
        from src.embeddings.encoder import get_encoder

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frames found in {frames_dir}. Run extract_frames.py first.")
        encoder = get_encoder(config)
        batch_size = _batch_size(config, "encoder_batch_size", 16)
        embeddings = []
        for start in range(0, len(frame_paths), batch_size):
            chunk = frame_paths[start:start + batch_size]
            embeddings.append(encoder.batch_encode([str(p) for p in chunk]))
            logger.info(f"Embedded {min(start + batch_size, len(frame_paths))}/{len(frame_paths)} frames")
        return np.concatenate(embeddings).astype("float32"), [p.name for p in frame_paths]
    ```

   `main()` updates to pass `config` through; add `encoder_batch_size: 16`
   to the `runtime:` config section alongside the existing
   `vlm_batch_size`/`detector_batch_size` (Stage 24 precedent).

2. **`build_observations()` / `save_observations_dir()` /
   `ObservationStore.save()`**: thread the real `encoder.name` (from the
   same `get_encoder(config)` call, not a string literal) all the way to
   `encoder.json`. Delete the dead unused `encoder_name: str = "resnet18"`
   default parameters and replace with the actual value — no function in
   this chain should be able to write an encoder name it wasn't handed.

3. **`src/utils.py::validate_config()`**: add the same pattern already used
   for `detector_model`/`vlm_model`:

    ```python
    from src.embeddings.encoder import _REGISTRY as KNOWN_ENCODERS  # or a
    # public accessor if exposing the private registry directly is
    # undesirable — either way, validate_config must not maintain its own
    # separate, driftable copy of the encoder name list the way
    # KNOWN_DETECTOR_MODELS/KNOWN_VLM_MODELS currently do for their layers.

    emb = (config.get("embedding") or {}).get("model")
    if emb not in KNOWN_ENCODERS:
        issues.append(("error", f"embedding.model: unknown model {emb!r} "
                                f"(known: {sorted(KNOWN_ENCODERS)})"))
    ```

4. **Rebuild the map** (`data/map/college_env_v1`) once §6.1–6.3 land, even
   before switching the default encoder — this proves the fixed pipeline
   reproduces the *same* ResNet18 map it produces today (a regression
   check on the fix itself, independent of the encoder swap).

**Acceptance criteria:**
- [ ] `embed_frames.py` calls `get_encoder(config).batch_encode(...)`;
  `grep -r "from src.embedder import embed_image" src/embed_frames.py`
  returns nothing.
- [ ] `manifest.json` and `vector_index/encoder.json` always agree with
  each other and with `config.yaml`'s `embedding.model`, verified by a new
  assertion in `tests/test_map_artifact.py` (or `test_observation_store.py`)
  that rebuilds a tiny synthetic store with a non-default encoder name and
  checks both files reflect it — not just eyeballing the manifest once.
- [ ] Re-running `build_map.py` with `embedding.model: resnet18` unchanged
  reproduces the current 5-places/4-edges map bit-for-bit (or documents
  exactly why not, e.g. nondeterminism already present in KMeans/exemplar
  selection) — this is the "did the refactor break anything" check.

---

# 7. Stage 33 — Build `scripts/compare_encoders.py`

This is the tool Finding F (§2.7) found missing. Without it, Stage 32's own
acceptance criterion ("gate the swap on measurement") cannot be satisfied.

**Spec:**

```python
"""Compare registered VisualEncoders on two metrics that actually matter
for this project's task (place retrieval under viewpoint/lighting change),
not generic benchmark numbers:

  same_place_consistency: mean pairwise cosine similarity between
      observations already assigned to the same place (higher = better —
      the encoder agrees these are the same place from different angles).
  different_place_separability: mean pairwise cosine similarity between
      observations in different places (lower = better — the encoder
      doesn't confuse distinct places).
  separation_margin: same_place_consistency - different_place_separability
      (the actual quantity worth comparing across encoders; a big single
      number can hide a bad different-place score).

Run against the existing 301-observation set (or whatever
config.paths.observations_dir currently holds) and its place assignments
from the last successful build_map.py run — this reuses data that already
exists, no new capture needed.
"""

def compare(encoder_names: list[str], observations, place_assignments, config) -> dict:
    ...  # for each encoder: re-embed every observation's frame_path,
    ...  # compute the three metrics above, return a dict keyed by encoder
    ...  # name; never raises on a missing/ungated model — skip it with a
    ...  # logged warning (Rule 4 applies to tooling too), so one gated
    ...  # DINOv3 access failure doesn't block comparing the other two.
```

Write `data/evaluation/encoder_comparison.json` (same directory convention
as Stage 27's other reports).

**Acceptance criteria:**
- [ ] Runs end-to-end comparing `resnet18` and `dinov2_registers_small` at
  minimum (DINOv3 included only if gated access is already available —
  never a hard requirement of this stage passing).
- [ ] Output is human-readable enough to paste directly into `STATE.md` as
  the justification for whichever encoder ends up default.

---

# 8. Stage 35 — Unify the Product (closes Finding C, D, I)

**Goal:** make `app.py` and `live_navigate.py` run the same pipeline
`evaluate_suite.py` already validates, and make `DEMO_RUN_OF_SHOW.md`
describe only things that are actually true of the running app. This is
the single highest-leverage stage in this document for "guaranteed final
complete product," because it's the difference between a system that was
*tested* to combine four evidence signals and a system that a person
*sees* combine four evidence signals.

1. **`app.py`**: replace `PlaceIndex.load(...)` / `embed_image` with
   `MapBundle.load(map_dir)` (already built, `src/mapping/map_artifact.py`)
   and `LocalizationTracker` (already built,
   `src/localization/tracker.py`) — both already return the
   backward-compatible status dict `tracker.py`'s own docstring promises
   ("stays BACKWARD-COMPATIBLE with the legacy LiveTracker... so app.py /
   live_navigate.py can switch without UI rewrites"). This means the
   `single_photo_tab`/`live_mode_tab` Streamlit rendering code barely
   changes — only the object construction and the embedding call
   (`get_encoder(config).encode(image)` instead of `embed_image(image)`)
   change. Add the term breakdown `DEMO_RUN_OF_SHOW.md` already promises
   (visual/semantic/temporal/graph contribution per candidate) as a new
   `st.dataframe` under the top-matches chart — the data already exists on
   each scored candidate, it's just never displayed today.
2. **`live_navigate.py`**: same swap, plus actually call
   `src.runtime.gate.should_process()` before the expensive path (closing
   Finding D) instead of the flat elapsed-time check — this also makes the
   `runtime.max_stale_seconds` config value finally do something outside
   `evaluate_suite.py`.
3. **`src/live_tracker.py` and `src/localize.py`**: keep them in the
   codebase (Rule 2-style — don't delete), but their only remaining caller
   after step 1–2 is `scripts/benchmark_baseline.py`, which is correct —
   that script is the intentionally-frozen legacy baseline and should keep
   using the legacy stack it's benchmarking.
4. **`DEMO_RUN_OF_SHOW.md`**: update the place/edge/warning counts to match
   the current manifest (5/4/1, not 7/6/0), and remove the "gate skips
   redundant frames" and "state machine"/"term breakdown" claims from being
   aspirational — after steps 1–2, they're literally true of `app.py`, so
   the fix here is confirming the doc against the *now-accurate* app, not
   just editing prose.

**Acceptance criteria:**
- [ ] `app.py`'s Single-Photo tab displays a visual/semantic/temporal/graph
  term breakdown for its top-3 matches (matches `DEMO_RUN_OF_SHOW.md` §3.1
  verbatim, because now it's true).
- [ ] `app.py`'s Live Mode tab shows the tracker's `state` field
  (TRACKING/UNCERTAIN/LOST/REACQUIRING/ARRIVED) somewhere in the UI.
- [ ] `live_navigate.py` logs `GateDecision.reason` per frame at debug
  level, so a demo run's terminal output visibly shows "redundant" frames
  being skipped.
- [ ] `grep -rn "from src.embedder import embed_image" app.py
  live_navigate.py` returns nothing.
- [ ] Existing app.py/live_navigate.py manual smoke test (open the app,
  upload a photo, get a route) still works — this is a refactor of what
  powers the UI, not a UI rewrite, so the acceptance bar is "looks the
  same, now backed by the real pipeline," not "looks different."

---

# 9. Stage 36 — Finish the Ablation Investigation (STATE.md §2.1)

Picks up exactly where `STATE.md` left off — this is not new work, it's
closing out work the project already scoped and stopped mid-way.

1. Compute the "visually-ambiguous subset" (frames where top-2 visual
   margin < 0.05) — this was "mid-implementation at stop" per `STATE.md`.
2. On that subset only, compare visual-only vs. +semantic top-1. This is
   the *actual* acceptance criterion Stage 23 always specified
   ("improve on the subset tagged visually ambiguous"); overall top-1 was
   never required to rise, so report both numbers with that framing.
3. If the subset shows a real benefit: done, document it, no config change
   needed.
4. If it doesn't: re-tune per `STATE.md`'s own next-step list (scene-type-
   only semantic scoring when landmarks are empty; down-weight
   landmark/object terms when they're single generic words) — gated on the
   subset metric, never on overall top-1, exactly as already decided.
5. Do this **after** Stage 32/34 land, not before — if the visual encoder
   changes, the set of "visually ambiguous" frames changes too, and running
   this analysis on the old ResNet18 embeddings would produce a subset
   definition that's stale by the time anyone reads the report.

**Acceptance criteria:**
- [ ] `data/evaluation/ablation_report.json` includes the
  visually-ambiguous-subset breakdown alongside the existing full-set
  numbers.
- [ ] `STATE.md` §2.1 is updated from "open finding" to either "resolved:
  semantic evidence helps on the ambiguous subset by X" or "resolved:
  re-tuned semantic weighting, see config diff" — not left open twice.

---

# 10. Definition of Done — v3 (supersedes v2 §16)

Every line from v2 §16 still applies (Stages 0–30 as originally specified)
**plus**:

- [ ] Stage 31: `embedding.model` actually controls the embeddings that get
  stored — verified by the new test in §6's acceptance criteria, not by
  reading the config file and assuming.
- [ ] Stage 32: `Dinov2RegistersEncoder` is the default; `ResNet18Encoder`
  remains registered and reachable; DINOv3 option is documented and
  behind an explicit opt-in, never silently attempted.
- [ ] Stage 33: `scripts/compare_encoders.py` exists and has been run at
  least once; its output is what justifies Stage 32's default, not
  intuition.
- [ ] Stage 34: every raw-similarity threshold in `config.yaml` has been
  re-derived for the new encoder's distribution (not copied), with the
  percentile-matching method documented in `STATE.md`.
- [ ] Stage 35: `app.py` and `live_navigate.py` run `MapBundle` +
  `LocalizationTracker`; the runtime gate is wired into the live loop;
  `DEMO_RUN_OF_SHOW.md` matches the running app exactly, re-verified by
  actually walking through it once, not just re-reading it.
- [ ] Stage 36: the visually-ambiguous-subset ablation measurement exists
  and `STATE.md` §2.1 is closed one way or the other.
- [ ] Stage 29 (v2 §12, unchanged, still blocked on the user): real
  walkthrough data recorded and labeled.
- [ ] Full test suite re-run after all of the above:
  `conda run -n ML python -m pytest tests/ -q -m "not slow"` — count must
  exceed the current 156, zero unexplained failures, and the new
  `test_encoder.py` / `test_map_artifact.py` / `test_runtime_gate.py`
  additions from §6/§8 are in that count, not just the pre-existing ones.

**Only when every line above is checked — on top of, not instead of, v2's
own checklist — does this stop being "a working prototype with known gaps"
and become the final, submission-ready system.** This is a stricter bar
than v2's, deliberately: v2's own checklist could be fully checked while
Findings A–I were all still true, which is exactly the gap this revision
closes.

---

# 11. Risk Register / Validation Loop

Every change above, with how it's verified and how it's undone if wrong.
This table *is* the "validation" pass this document is built on — nothing
above landed in this plan without a row here.

| Change | Risk | Falsifiable check | Rollback |
|---|---|---|---|
| Stage 31: route `embed_frames.py` through `get_encoder(config)` | Batched `batch_encode()` produces numerically different vectors than the old per-frame loop, invalidating the "reproduces the current map" check | §6's bit-for-bit (or documented-diff) rebuild check with `embedding.model: resnet18` unchanged | Revert `embed_frames.py` to the direct `embed_image()` call; `ResNet18Encoder.batch_encode` already exists and is tested independently, so the batching itself isn't new risk, only its new call site is |
| Stage 32: DINOv2-with-Registers as default | Similarity distribution shift breaks downstream thresholds silently | Stage 34's recalibration protocol + Stage 27's ablation re-run, compared against the ResNet18 baseline on the same dataset | `embedding.model: resnet18` in config.yaml — one line, both encoders stay registered |
| Stage 32 (optional): DINOv3 stretch | Gated HF access denied or delayed past the deadline | Registry lookup fails closed the same way `get_detector`/`get_scene_tagger` already fail closed on a bad model id — falls back to the configured default, never silently to whatever loaded last | Never make it the default in the first place (§4.2) — this is a config value nobody sets unless they've already confirmed access |
| Stage 34: recalibrated thresholds | New thresholds overfit to the 301-observation pseudo-label set the same way the current ones might be | Same caveat this project already applies everywhere (Rule 6): label the result "pseudo-label self-consistency," re-validate once Stage 29's real data lands | Thresholds are config values; keep the old ResNet18-tuned values in a comment alongside the new ones for a quick diff-and-revert |
| Stage 35: app.py/live_navigate.py onto `MapBundle`+`LocalizationTracker` | Larger blast radius than other stages — this touches the actual demo surface right before a presentation | `tracker.py`'s own documented backward-compatible status dict (§8 point 1) means the Streamlit rendering code changes minimally; test manually well before any real demo, not the night before | `src/live_tracker.py`/`src/localize.py` are kept, not deleted (Rule 2) — a one-line import revert restores the exact current behavior |
| Stage 36: re-tuning semantic weights on the ambiguous subset | Could overfit a rule to 301 frames' worth of noise | Explicitly gated on the subset metric only, per the project's own existing rule — never gated on overall top-1, which is the metric most likely to reward overfitting here | Semantic weight change is a config diff; StubTagger path already provides a clean "semantic off" fallback if the re-tune misbehaves |

---

# 12. Recommended Final Architecture (v3)

Unchanged from v2 §15 with two additions: the encoder box now says
"DINOv2-w/-Registers (default) / ResNet18 (baseline)," and both the offline
and runtime paths now point at the **same** `MapBundle` /
`LocalizationTracker` objects — collapsing what were two divergent
pipelines (legacy vs. v2) into one, per Rule 7.

```text
                         SIMPLE-NAV (v3)
                             |
              +--------------+--------------+
              |                             |
              v                             v
       OFFLINE MAPPING                RUNTIME LOCALIZATION
       (build_map.py)                 (app.py / live_navigate.py —
              |                        NOW THE SAME STACK AS BELOW)
              v                             v
      walkthrough video               camera stream
              |                             |
              v                             v
       smart sampling                runtime gate (Stage 24 — NOW WIRED IN)
              |                             |
              v                             v
   DINOv2-w/-Registers encoder    [skip if redundant] -> reuse state
   (config: embedding.model,             |
    ResNet18 kept as baseline)           v
       YOLO26n + LFM2.5-VL-450M   DINOv2-w/-Registers encoder (same
              |                    get_encoder(config) call as mapping)
              v                          |
      temporal segmentation       detector + VLM -> semantic scoring
              |                    (Stage 23) -> FAISS retrieval
              v                          |
       place formation +          candidate scoring (visual+semantic
       reconciliation              +temporal+graph) -> Bayes state
              |                    estimator -> state machine
              v                    (TRACKING/UNCERTAIN/LOST/
       graph construction +        REACQUIRING/ARRIVED) -> confidence
       validation                        |
              |                          v
              v                    route + spoken instruction
       versioned map artifact      (rendered in app.py: term
       (manifest now honestly            breakdown + state, per
        records the real encoder)        Stage 35)
```

---

# 13. Final Research Thesis (updated)

Unchanged core claim from v1/v2 — this revision doesn't need a new thesis,
it needs the existing one to be *demonstrably* what the running product
does:

> A camera-only indoor navigation system that autonomously constructs a
> semantic connectivity graph from a single-floor walkthrough and performs
> continuous visual localization by combining appearance, semantic
> landmarks, temporal continuity, and graph topology.

What v3 adds to the evidence for that thesis, on top of v2's ablation table
and pseudo-vs-real accuracy comparison: **an encoder ablation
(`encoder_comparison.json`, Stage 33) showing the appearance term itself
was measured and chosen, not assumed** — and, critically, a demo where a
person watching it can actually see all four signals working together
(Stage 35), rather than being told they do in a document and shown a
product that only uses one of them.

---

# Appendix A — `requirements.txt` delta (on top of v2's Appendix A)

No new packages required — `transformers>=5.1` (already pinned per v2
Appendix A) already covers `AutoImageProcessor`/`AutoModel` for the DINOv2
family. Add a comment only:

```text
transformers>=5.1   # floor per LFM2.5-VL-450M's model card; also covers
                    # AutoModel/AutoImageProcessor for DINOv2-with-Registers
                    # (planner v3 §4) and, if gated access is available,
                    # DINOv3 — no separate package needed for either.
```

# Appendix B — Config schema delta (full, on top of v2 §14)

```yaml
embedding:
  model: "dinov2_registers_small"   # was: "resnet18" (planner v3 §4)

runtime:
  max_stale_seconds: 5.0
  vlm_batch_size: 12
  detector_batch_size: 16
  encoder_batch_size: 16            # NEW — planner v3 §6, closes the
                                     # embed_frames.py batching gap

# mapping.* / localization.* thresholds: values unchanged in this document
# on purpose — they are outputs of Stage 34's recalibration protocol
# (§5), not hand-edited here. Do not copy new numbers into this schema
# until that measurement has actually run; that's the entire point of
# making it a protocol instead of a one-time edit.
```
