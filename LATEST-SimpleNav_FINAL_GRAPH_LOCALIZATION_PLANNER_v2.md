# SimpleNav — Final Graph + Localization Upgrade Planner (v2)

> **This is a full replacement for `SimpleNav_FINAL_GRAPH_LOCALIZATION_PLANNER.md`.**
> Stages 0–22 (Milestones A–C) are DONE and verified — see `STATE.md` and the
> original planner/executor for their full historical spec text, which this
> document does not repeat. What changed here:
> 1. A root-cause fix for the Stage 07 VLM failure (§4).
> 2. Updated, web-verified model choices for the perception stack (§5).
> 3. Stages 23–29, which were one-paragraph goals before, are now fully
>    specified: algorithm, config, tests, acceptance criteria — the same
>    level of detail Stages 0–22 already had in the executor doc.
> 4. Two new stages (§12, §13) closing gaps this audit found: the
>    provisional-dataset problem, and a pre-submission hardening pass.
> 5. A single Definition-of-Done checklist (§16) that gates "complete."
>
> Nothing here contradicts Rule 1–6 below. The system stays boring on
> purpose; only the perception backends and the unfinished stages change.

---

# 0. Final System Definition

A camera-only indoor navigation prototype that autonomously builds a
semantic connectivity graph from one walkthrough video, then localizes a
live camera stream against that graph using appearance + semantic evidence
+ temporal continuity + graph topology — and turns the result into a
spoken turn-by-turn instruction. Unchanged from v1.

# 1. Engineering Rules

These are unchanged and still govern every stage below.

## Rule 1 — Do not rewrite working components unnecessarily
Stages 0–22 are frozen. This update touches perception model *choices*
(config + one new tagger class), not the mapping/localization/state-machine
code that already passes its test suite.

## Rule 2 — Keep a baseline implementation
The legacy Milestone-A path (`benchmark_baseline.py`) stays as the
permanent before/after harness. Do not delete it.

## Rule 3 — Never allow a support model to silently become ground truth
Still true, and more relevant than ever: the new VLM adds bounding-box
grounding capability (see §5) — that is still *evidence*, never a
coordinate, never a topology edge.

## Rule 4 — Unknown is better than confidently wrong
Unchanged. Applies to the new VLM backend's fallback chain (§5.2) exactly
as it applied to the old one.

## Rule 5 — Retrieval proposes; temporal/graph reasoning decides
Unchanged.

## Rule 6 — Evaluation must use unseen observations
Unchanged — and directly relevant to §12: the current 0.885 top-1 number
is self-consistency on provisional pseudo-labels, not accuracy against
ground truth. Do not present it as the latter (see §12).

# 2. Target Repository Structure — Delta

Only new/changed paths vs. the current repo:

```text
src/perception/
    detector.py            # unchanged interface; new default model id
    scene_tagger.py         # + LFM2VLTagger class (new primary backend)
config.yaml                 # perception.* defaults updated (§5.3)
requirements.txt             # corrected — see Appendix A
README.md                    # needs a rewrite pass — see Stage 28 (§11)
src/localization/
    semantic_scoring.py      # NEW — Stage 23 (§6)
src/runtime/
    gate.py                  # NEW — Stage 24 (§7)
data/evaluation/
    ablation_report.json     # NEW — Stage 27 (§10)
    failure_log.jsonl        # NEW — Stage 27 (§10)
data/raw/real_walkthrough_*.mp4   # NEW — Stage 29 (§12)
```

# 3. Status Summary — Stages 0–22 (Milestones A–C)

| Stage | Name | Status | Verified |
|---|---|---|---|
| 00–01 | Baseline + benchmark harness | ✅ | top-1 self-consistency 0.885, top-3 0.918, ~22ms/frame |
| 02 | Observation model | ✅ | `Observation` dataclass + JSONL I/O |
| 03 | Smart sampling | ✅ | quality gate + novelty + sustained-change retention |
| 04 | Object detection (YOLO) | ✅ (model swap pending — §5.1) | `YoloDetector`/`StubDetector`, COCO only |
| 05 | Encoder abstraction | ✅ interface / ⚠️ never benchmarked against alternatives | only `ResNet18Encoder` registered — see §5.4 |
| 06 | Observation store | ✅ | FAISS + JSONL, 301 observations populated |
| 07 | VLM scene/landmark tagging | ⚠️ **BLOCKED → fixed here** | interface/schema/stub all ✅; Qwen2.5-VL 4-bit backend broken — root cause in §4, fix in §5.2 |
| 08 | Temporal segmentation | ✅ | adaptive threshold (mean + 2.5σ) |
| 09 | Place formation + exemplars | ✅ | `temporal_diversity` method, ≤3 exemplars/place |
| 10 | Place reconciliation | ✅ | multi-signal merge, 0.92 visual-only threshold |
| 11 | Transition extraction | ✅ | debounced, `minimum_edge_support: 1` |
| 12 | Graph construction | ✅ | 7 places, 6 edges, 0 validation warnings on the 301-obs set |
| 13 | Graph validation | ✅ | connectivity/isolated/weak-edge/duplicate checks |
| 14 | Versioned map artifact | ✅ | `data/map/college_env_v1/` |
| 15 | Candidate retrieval | ✅ | top-K + margins |
| 16 | Candidate scoring | ✅ | visual/semantic/temporal/graph weighted terms |
| 17 | Graph-constrained filtering | ✅ | soft penalties, local-first |
| 18 | Bayes state estimator | ✅ | likelihood = visual+semantic only; transition prior handles graph/temporal |
| 19 | State machine | ✅ | TRACKING/UNCERTAIN/LOST/REACQUIRING/ARRIVED |
| 20 | Transition stabilization | ✅ | K-consecutive or high-posterior confirmation |
| 21 | Global reacquisition | ✅ | LOCAL/GLOBAL modes |
| 22 | Confidence calibration | ✅ | HIGH/MEDIUM/LOW/UNKNOWN |

130/136 tests pass (6 slow/skippable). This is real, working engineering —
the remaining work is Stages 23–29 plus the perception-model fix, not a
rebuild.

---

# 4. Incident Report — Stage 07 VLM Failure

**Symptom (from STATE.md):** `Qwen/Qwen2.5-VL-3B-Instruct`, loaded 4-bit via
`BitsAndBytesConfig`, produces incoherent text — prompt echo, garbled
tokens, malformed JSON like `{ ( and corridor, Junction: unknown }` — even
after `_strip_prompt_echo` and capping `max_new_tokens`.

**Root cause (confirmed via web research, not guesswork):** this is a
known, widely-reported ecosystem bug, not a bug in `scene_tagger.py`.
Qwen2.5-VL models quantized through bitsandbytes (and separately, AWQ)
have repeatedly broken across `transformers` releases from ~4.50.0 onward
— GitHub issues on both `transformers` and `vllm` document the *exact*
symptom class (garbled/gibberish output, sometimes literal repeated
punctuation, regardless of prompt, image, or Flash Attention on/off),
traced to how those releases changed generation/rotary-embedding handling
for the Qwen2.5-VL architecture specifically. Your environment runs
`transformers 5.3.0` — far past the versions where this was last confirmed
broken, so there's no guarantee it was ever fixed for this
quantization path; it may just have evolved into a different-looking
failure. Separately, 4-bit NF4 quantization of a repurposed causal-LM
checkpoint bolted onto a vision tower is inherently more fragile than
quantizing a model that was designed and validated for 4-bit from the
start.

**Two ways to close this out:**

- **Cheap diagnostic (5 minutes, if you want confirmation):** in a
  disposable venv, `pip install "transformers<4.50"` and re-run the
  Stage-07 slow test. If output becomes coherent, that confirms the
  regression window. Do **not** ship on a pinned-old `transformers`,
  though — Milestone C's other stages may depend on newer transformers
  behavior, and you'd be trading one fragility for another.
- **Actual fix (recommended, done in §5.2):** stop using a 3B model
  quantized to 4-bit through bitsandbytes for this task at all. Move to a
  VLM that is small enough to run at bf16/8-bit — precisions that don't
  have this failure mode — while matching or beating Qwen2.5-VL-3B on the
  benchmarks that matter for structured JSON tagging.

# 5. Updated Perception Stack — Model Selection v2

## 5.1 Object detector: YOLOv8n → YOLO26n

Your `ultralytics` (8.4.137, per STATE.md) already ships YOLO26. This is a
one-line config change, no code change — `Detector`/`YoloDetector` already
treat the model id as a config string.

| | YOLOv8n (current) | **YOLO26n (recommended)** |
|---|---|---|
| Params | ~3.2M | **2.4M** |
| FLOPs | ~8.7B | **5.4B** |
| COCO mAP50-95 | ~37.3 | **40.9** |
| Head | NMS-based | **NMS-free (lower, more consistent per-frame latency)** |
| CPU inference | baseline | **up to ~43% faster** (matters if this ever runs on a phone CPU, per Stage 24) |

Why this matters for "many frames": NMS-free end-to-end decoding removes a
variable-cost post-processing step that scales with how cluttered a frame
is — on a 301-frame mapping pass (and worse, on a continuous live stream)
that variability compounds. YOLO26n is smaller, more accurate, and more
predictably fast than what you have. Change:

```yaml
perception:
  detector_model: "yolo26n.pt"   # was: "yolov8n.pt"
```

Keep `YoloDetector` exactly as-is; only the weights file changes. If a
pinned `ultralytics` version in some environment doesn't resolve
`yolo26n.pt`, `get_detector()`'s existing try/except already falls back
to `StubDetector` safely — but add a warning-level log there so a silent
fallback during map-building doesn't go unnoticed (`logger.warning` is
already there; just also assert on it in a startup smoke check, see §13).

**Optional, not required — closing the door/stairs/sign gap:** Ultralytics
also ships `YOLOE-26n`, an open-vocabulary variant that detects arbitrary
text-prompted classes (e.g. `"door"`, `"staircase"`, `"exit sign"`,
`"reception desk"`) without any fine-tuning, at 3.9M params / 6.1 GFLOPs —
still nano-class. This would let those specific navigation-relevant
classes get real bounding boxes instead of living only in VLM free text.
This is genuinely useful but changes the Rule-3 boundary (a second
detector head producing nav-relevant boxes) — treat it as an explicit,
separately-reviewed addition (a "Stage 23b") rather than folding it into
the required path. The VLM alone (§5.2) is sufficient to unblock Stage 07
and hit Stage 23's acceptance criteria.

## 5.2 VLM scene tagger: Qwen2.5-VL-3B (4-bit) → LFM2.5-VL-450M (bf16/8-bit)

**Primary: `LiquidAI/LFM2.5-VL-450M`.**

| Benchmark | Qwen2.5-VL-3B (4-bit, broken) | SmolVLM2-500M | **LFM2.5-VL-450M** |
|---|---|---|---|
| Params | 3B (quantized, unstable) | 500M | **450M** |
| MM-IFEval (instruction-following / structured-output steerability) | n/a — output invalid | 11.27 | **45.00** |
| MMBench (dev en) | — | 52.32 | **60.91** |
| POPE (hallucination robustness) | — | 82.67 | **86.93** |
| RefCOCO-M (bounding-box grounding) | n/a | not supported | **81.28** |
| Function calling (BFCLv4) | n/a | not supported | **21.08** |
| Edge latency | OOM/unstable at fp16 on 6GB alongside other models | ~ok | **<250ms per 512×512 frame on a Jetson Orin** — an RTX 3050 desktop GPU will comfortably beat that |

Why this is the right swap, not just a smaller model:

- **MM-IFEval is the metric that predicts your exact failure mode** — it
  measures whether a model actually follows explicit output-format
  instructions. LFM2.5-VL-450M's 45.00 vs. SmolVLM2's 11.27 is a 4x gap;
  this is the single most relevant number for "will this model reliably
  emit the fixed JSON schema Stage 07 requires."
- It has native structured-output support: bounding-box grounding
  (RefCOCO-M) and function-calling were added in this exact release,
  which means the maintainers tested and tuned for the JSON-schema use
  case you need — this isn't a repurposed chat model.
- **At 450M params it does not need 4-bit quantization to fit a 6GB
  card.** Run it at bf16 (~1GB) or 8-bit (~0.5GB) — both are stable
  precisions for this model. Avoid `bitsandbytes` 4-bit for this model
  specifically: independent int4 quantization tests of LFM2.5-VL-450M
  show it degrading sharply at int4 (unlike its 3B sibling, which
  tolerates int4 fine) — so int4 would reintroduce the exact class of
  problem you're escaping. This also means you can drop the
  `bitsandbytes` dependency from the default path entirely; it becomes
  optional.
- It's actively maintained by Liquid AI with first-class `transformers`,
  `llama.cpp`/GGUF, and MLX support — three independent runtimes, which is
  a good sign for not hitting another quantization-integration bug six
  months from now.

**Implementation — add one class, following the exact pattern already in
`scene_tagger.py`:**

```python
class LFM2VLTagger(_HFVisionTagger):
    """LiquidAI LFM2.5-VL-450M — default backend (bf16, fits 6 GB VRAM
    without quantization; do not use int4 — see planner v2 §5.2)."""

    def __init__(self, model_id: str = "LiquidAI/LFM2.5-VL-450M", max_new_tokens: int = 256):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

    def _load(self) -> None:
        # override base _load: skip the 4-bit BitsAndBytesConfig entirely
        if self._loaded:
            return
        self._do_load()
        self._loaded = True

    def _do_load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto"
        )
        logger.info(f"LFM2.5-VL loaded: {self.model_id} (bf16)")

    def _generate(self, image, prompt: str) -> str:
        from PIL import Image
        if isinstance(image, str):
            image = Image.open(image)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt_text, images=[image], return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        output = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _strip_prompt_echo(output)

    def name(self) -> str:
        return f"lfm2.5vl-450m:{self.model_id}"
```

Pin `transformers` to whatever minimum version the
`LiquidAI/LFM2.5-VL-450M` model card specifies at integration time (check
the card — LFM2 support was added to `transformers` after Qwen2.5-VL's,
so confirm the floor rather than assuming your current 5.3.0 is new
enough or reusing the Qwen floor).

**Fallback chain (Rule 4 — unchanged shape, updated members):**

1. `LFM2VLTagger` (primary)
2. `SmolVLMTagger`, model id updated from `SmolVLM2-2.2B-Instruct` to
   `SmolVLM2-500M-Instruct` — the 2.2B pick was never actually the
   "lighter" option once you have a working 450M model; keep it as a
   fallback only, sized to actually be lighter than the primary.
3. `StubTagger` — unchanged, zero risk, already correct.

`get_scene_tagger()`'s dispatch logic (currently `if "Qwen" in model_id`)
needs a third branch:

```python
tagger_cls = (
    LFM2VLTagger if "LFM2" in model_id else
    QwenVLTagger if "Qwen" in model_id else
    SmolVLMTagger
)
```

Keep `QwenVLTagger` in the codebase (don't delete it) behind an explicit
opt-in — it's a valid future comparison point once/if the transformers
ecosystem stabilizes for it, but it is no longer the default.

## 5.3 Config diff

```yaml
perception:
  detector_enabled: true
  detector_model: "yolo26n.pt"          # was yolov8n.pt
  detector_confidence: 0.35
  vlm_enabled: true                      # was false — unblocked
  vlm_model: "LiquidAI/LFM2.5-VL-450M"   # was Qwen/Qwen2.5-VL-3B-Instruct
  vlm_quantization: "bf16"               # was "4bit" — do not use 4bit with this model
  vlm_max_tokens: 256
```

## 5.4 Encoder — optional, not required for "complete"

Stage 5 built the `VisualEncoder` registry but only ever registered
`ResNet18Encoder`; the planned CLIP-like/DINO-like comparison never ran.
This is not broken — 0.885 top-1 self-consistency is a reasonable number
for a college-level demo — but if you want a measured (not assumed)
answer: register `facebook/dinov2-small` (21M params, strong
viewpoint-invariant retrieval features) behind the same interface and run
the comparison script Stage 5 already specified (same-place consistency
vs. different-place separability). Gate any swap on that measurement, per
Rule/§8's own selection criterion — don't swap because it's "generally
better." Low priority; do this only after Stages 23–30 are done.

---

# 6. Stage 23 — Integrate Semantic Evidence Correctly

**Goal (unchanged):** use detector + VLM evidence to break visual ties
between appearance-similar places.

**This was previously one paragraph. Concrete spec:**

Create `src/localization/semantic_scoring.py`:

```python
def semantic_similarity(query_tags: SceneTags, query_objects: list[DetectedObject],
                         place_tags: list[SceneTags]) -> float:
    """Returns a score in [0, 1]. Never raises; missing/unknown tags -> 0.5
    (neutral — never actively penalizes a place just because tagging failed,
    per Rule 4)."""
```

Algorithm:

1. **Scene-type match**: 1.0 if `query_tags.scene_type == place's dominant
   scene_type` (mode across the place's stored exemplar tags), 0.3 if
   either is `"unknown"`, else 0.0.
2. **Landmark overlap**: Jaccard similarity between `query_tags.landmarks`
   (lower-cased, stripped) and the union of landmark sets stored across
   the place's exemplars. Empty-vs-empty → neutral 0.5, not 0 and not 1.
3. **Object overlap**: Jaccard similarity between detected COCO class
   names for the query vs. the place's historically observed class sets.
4. **Combine**: `0.4 * scene_match + 0.4 * landmark_jaccard + 0.2 * object_jaccard`.
5. Feed this into the existing `w_semantic` term in candidate scoring
   (`config.yaml` already has `localization.w_semantic: 0.25` reserved —
   Stage 16 built the slot, Stage 23 is what actually fills it with real
   numbers instead of a placeholder/neutral constant).

**Important (unchanged, now testable):** semantic evidence must be able to
*fail to help* without breaking localization. Concretely: if `vlm_enabled:
false` (StubTagger active), `semantic_similarity` must degrade gracefully
to using object-overlap only, and the effective weight on semantic
evidence in that mode should not silently double-count through the visual
term.

**Tests — `tests/test_semantic_scoring.py`:**
- identical tags → 1.0; completely disjoint non-empty landmark sets → low but non-zero score (never exactly 0, per Rule 4);
- one side `"unknown"` → capped contribution, never dominates the combined score;
- stub-tagger inputs (empty landmarks always) → falls back to object-overlap only, doesn't crash;
- synthetic case from the planner's own example: two visually-near-identical corridors, one with `["stairs on left", "blue room sign"]`, the other with `[]` → semantic score must separate them by ≥ 0.4.

**Acceptance criteria:**
- [ ] On the 301-obs set, re-run `evaluate.py`: top-1 self-consistency must not regress vs. the 0.885 baseline, and should improve on the subset of frames the ablation (§10) tags as "visually ambiguous."
- [ ] `test_semantic_scoring.py` passes, including the disjoint-landmarks separation case above.

---

# 7. Stage 24 — Optimize the Runtime Pipeline

**Goal (unchanged):** continuous navigation must stay practical, on a
laptop or eventually a phone.

**Concrete spec.** Create `src/runtime/gate.py`:

```python
@dataclass
class GateDecision:
    run_expensive: bool
    reason: str  # "novel" | "redundant" | "forced_interval"

def should_process(frame, last_processed_embedding_lowres, last_processed_ts,
                    config) -> GateDecision:
    """Cheap pre-filter BEFORE detector/VLM/encoder run. Reuses the same
    32x32 mean-subtracted descriptor + novelty threshold already built in
    Stage 02/03 (src/extraction/quality.py) — do not build a second
    novelty metric, reuse the existing one via config.sampling.novelty_threshold.
    """
```

Runtime loop (as specified, now with the gate wired in explicitly):

```text
camera frame
   -> cheap quality/novelty gate (reuses Stage 03's descriptor — near-zero cost)
   -> [gate says "redundant"] -> reuse last state, skip everything below
   -> [gate says "novel" or forced_interval elapsed] ->
        detector (YOLO26n) -> VLM (LFM2.5-VL-450M) -> embedding (ResNet18)
        -> candidate retrieval -> semantic scoring (Stage 23) -> state update
```

**Important optimization, made concrete:**
- `forced_interval`: even if nothing looks novel, re-run the expensive
  path at least every N seconds (config: `runtime.max_stale_seconds`,
  default 5) so a genuinely slow, gradual scene change (walking slowly
  down a long uniform corridor) doesn't get stuck on stale state forever.
- **Batch the offline map-building pass.** This is the actual "large
  number of frames" throughput win, separate from the live-runtime gate:
  `YoloDetector.detect()` and `LFM2VLTagger.tag()` currently process one
  image at a time. For `build_map.py`'s one-time pass over all sampled
  observations, add batched variants — Ultralytics' `.predict()` already
  accepts a list of image paths/arrays and batches internally; for the
  VLM, batch `self._processor(...)` calls in groups of 8–16 images
  (bounded by testing actual VRAM headroom, since LFM2.5-VL-450M is small
  enough that batching won't be VRAM-bound the way Qwen-3B was). This
  turns O(N) sequential model-load-adjacent overhead into O(N/batch_size).

**Config — new `runtime:` section:**

```yaml
runtime:
  max_stale_seconds: 5.0
  vlm_batch_size: 12          # offline mapping pass only; live loop stays 1-at-a-time
  detector_batch_size: 16
```

**Tests — `tests/test_runtime_gate.py`:**
- identical consecutive frames -> gate says "redundant" every time after the first;
- frames past `max_stale_seconds` with no novelty -> gate forces a run anyway;
- gate never raises on a corrupt/unreadable frame (returns `run_expensive=True` — fail toward doing the expensive check, not toward silently skipping, since skipping on a decode error could freeze tracking on stale state).

**Acceptance criteria:**
- [ ] On a synthetic 5-minute static-camera "test" stream (same room repeated), the expensive path runs at most a handful of times, not once per captured frame.
- [ ] Offline map-building wall-clock time on the 301-frame set drops measurably with batching enabled vs. disabled (report both numbers in `STATE.md`'s next update).

---

# 8. Stage 25 — Keep Routing Simple

Unchanged from v1: `current place + destination -> shortest path`, with
`use_weighted_routing` (already implemented) as the only refinement.
**Do not touch this stage.** It's done and it's supposed to stay boring.

# 9. Stage 26 — Keep LLM/TTS at the Edge

Unchanged goal: turn a place-name list into one natural-language sentence,
then TTS it. The LLM must never decide location, edges, or graph
correctness.

**One consolidation worth considering, not required:** `LiquidAI/LFM2.5-VL-450M`'s
text-only backbone (its language model, used with no image input) is
itself a usable ~350M instruction-following text model. If Stage 26 needs
an LLM for the instruction-composition step, reusing the *same already-loaded*
model in text-only mode avoids adding a second model+dependency purely for
one sentence of templating. Weigh this against Rule 2 — a plain Python
f-string template (`"Continue through {a}, follow {b}, then take {c}."`)
may genuinely be simpler and sufficient, and the planner's own instruction
("do not make this the centerpiece") leans toward the template. Try the
template first; only reach for the LLM if the templated phrasing is
noticeably worse in the demo.

---

# 10. Stage 27 — Build the Research Evaluation Suite

**Previously a metric-name list with no implementation plan. Concrete spec.**

Create `evaluate_suite.py` (separate from the existing `evaluate.py`,
which stays as the simple single-run accuracy report):

```python
def run_localization_metrics(...) -> dict:
    # top-1, top-3, segment accuracy, transition accuracy,
    # false-jump rate, reacquisition time, unknown precision
    # — definitions exactly as already named in the original planner §30.

def run_graph_metrics(...) -> dict:
    # node purity, duplicate rate, merge error, edge precision/recall,
    # fragmentation — exactly as named in the original planner §31.

def run_ablation(...) -> dict:
    # re-run localization with each of: {visual only}, {+semantic},
    # {+temporal}, {+graph} added incrementally, per original §32's
    # progression tables. Write data/evaluation/ablation_report.json.
```

**Failure analysis workflow (was a header with no content):**
1. Every `LocalizationTracker` decision already writes a decision log
   (Stage 19 built this). Add a script, `scripts/failure_inspector.py`,
   that filters that log for: LOST events, low-confidence ARRIVED
   confirmations, and any frame where top-1 and top-3 disagree with the
   hand-labeled `test_labels.json`.
2. For each, dump the frame + its top-3 candidates + semantic tags side
   by side into `data/evaluation/failure_log.jsonl` — this is what you
   actually show in a report/demo as "here's what the system got wrong
   and why," which is stronger evidence of understanding than a single
   accuracy number.

**Critical caveat, carried from Rule 6 / §12 below:** every number this
stage produces on the current 301-frame set is *self-consistency against
pseudo-labels*, not accuracy against ground truth. Label evaluation
reports and any submission material accordingly until Stage 29 (§12)
provides a real walkthrough + real labels. Do not let "top-1: 0.885" stand
unqualified in front of an evaluator.

**Acceptance criteria:**
- [ ] `evaluate_suite.py` runs end-to-end and writes both JSON reports.
- [ ] `failure_inspector.py` produces at least one human-readable failure case per LOST event in the current dataset.
- [ ] The ablation table shows a monotonic (or explained-if-not) improvement as visual → +semantic → +temporal → +graph are added — this is the evidence for the "final research thesis" in §17.

---

# 11. Stage 28 — Migration, Hygiene, and Documentation

Most of the original migration table (`extract_frames.py`,
`embedder.py`, etc. → their planned roles) is complete — it happened
across Stages 0–22. What's left:

- [ ] **Fix `requirements.txt`** — see Appendix A. It is currently missing
  `ultralytics`, `transformers`, `bitsandbytes` (now optional, not
  default), `qwen-vl-utils` (now optional), and `pytest`, and lists
  `faiss-cpu` where the working env actually uses `faiss`. A fresh clone
  cannot currently run the perception stage from `pip install -r
  requirements.txt` alone.
- [ ] **Rewrite `README.md`'s Architecture/Tech-stack sections.** They
  currently describe only the Milestone-A baseline (fixed sampling,
  single-pass KMeans, flat FAISS). Someone reading only the README has no
  idea Milestones B and C (smart sampling, temporal segmentation, the
  Bayes filter, the perception layer) exist. At minimum, update the
  architecture diagram and tech-stack table to match `STATE.md`, and move
  the "Future work" section's already-completed items (probabilistic
  filter, temporal continuity) out of future work.
- [ ] Update `README.md`'s model names once §5 lands (`yolo26n.pt`,
  `LFM2.5-VL-450M`) so setup instructions don't reference dead config.

**Acceptance criteria:**
- [ ] `python -m venv fresh && pip install -r requirements.txt` followed by `pytest tests/` succeeds with no `ModuleNotFoundError`, in a clean environment with no pre-existing conda env.
- [ ] README's architecture diagram and tech-stack table match the actual current pipeline (spot-check against `STATE.md`).

---

# 12. Stage 29 — Real-World Dataset & Validation (NEW)

**Goal:** replace the provisional, pseudo-labeled 301-frame dataset with
something that actually validates the system, closing the gap Rule 6
already warned about.

1. Record a real walkthrough video of the target building/floor —
   config already has a `paths.video` slot waiting for this
   (`data/College_env.mp4`, currently missing per the config.yaml
   comment).
2. Record a **second**, independent walkthrough on a different day/lighting
   condition for held-out evaluation — Milestone A's own benchmark
   methodology (80/20 temporal split of one video) is a weaker test than a
   genuinely separate walkthrough, which is what `evaluate.py`'s docstring
   already recommends ("use frames that weren't part of building the
   map").
3. Hand-label a real `data/test_labels.json` and `heldout_labels.json` /
   `exemplar_labels.json` using `scripts/make_contact_sheet.py` (already
   built, unused so far because there was no real footage to label).
4. Re-run `benchmark_baseline.py` and `evaluate_suite.py` (§10) against
   real ground truth. Report both the old pseudo-label self-consistency
   number and the new real-accuracy number side by side — the delta
   between them is itself a useful, honest result.

**Acceptance criteria:**
- [ ] At least one real walkthrough video + one independent real evaluation walkthrough exist in `data/raw/`.
- [ ] `data/test_labels.json` contains real, hand-verified labels (not pseudo-labels).
- [ ] A report exists comparing pseudo-label self-consistency vs. real-label accuracy.

---

# 13. Stage 30 — Final Hardening & Submission Readiness (NEW)

This is the stage that turns "all the pieces work" into "guaranteed
complete product." Checklist:

- [ ] **Dependency smoke test**: fresh clone, fresh env, `pip install -r requirements.txt`, full `pytest tests/`, then a full `build_map.py` run on real data (§12), then `app.py` end to end — all with zero manual patching.
- [ ] **Startup assertions, not silent fallbacks**: `get_detector()` and `get_scene_tagger()` already log a warning on fallback (good) — add a one-line startup banner in `app.py`/`live_navigate.py`/`build_map.py` that prints which detector/tagger backend actually loaded, so a silent StubTagger fallback during a live demo is visible immediately, not discovered after the fact.
- [ ] **Config validation**: a `validate_config.py` that checks every path in `config.yaml` resolves, every referenced model id is one of the known/registered ones, and every weight (`w_visual` + `w_semantic` + `w_temporal` + `w_graph`) sums sanely — run this as the first line of `build_map.py`.
- [ ] **Demo script**: a short, written run-of-show for the Streamlit demo — which building, which two destinations to route between, what to say about the ablation numbers (§10) and the honest pseudo-vs-real accuracy comparison (§12) — so the "final research thesis" (§17) is what gets presented, not an ad hoc walkthrough.
- [ ] **Full test count check**: re-run `pytest tests/ -q` after all of §6–§12 land; the new tests (`test_semantic_scoring.py`, `test_runtime_gate.py`) should push the count past the current 130, with 0 unexplained failures.

**This stage is the gate.** Nothing is "the final complete product" until every box above is checked — that's what makes the guarantee real rather than aspirational.

---

# 14. Updated Configuration Schema — Full Delta

```yaml
perception:
  detector_model: "yolo26n.pt"            # was yolov8n.pt
  vlm_enabled: true                        # was false
  vlm_model: "LiquidAI/LFM2.5-VL-450M"     # was Qwen/Qwen2.5-VL-3B-Instruct
  vlm_quantization: "bf16"                 # was 4bit

runtime:                                    # NEW section (Stage 24)
  max_stale_seconds: 5.0
  vlm_batch_size: 12
  detector_batch_size: 16
```

Everything else in `config.yaml` (mapping/localization/navigation/live
sections) is unchanged.

# 15. Recommended Final Architecture (v2)

```text
                         SIMPLE-NAV
                             |
              +--------------+--------------+
              |                             |
              v                             v
       OFFLINE MAPPING                RUNTIME LOCALIZATION
              |                             |
              v                             v
      walkthrough video               camera stream
              |                             |
              v                             v
       smart sampling                runtime gate (Stage 24, NEW)
              |                             |
              v                             v
      observation creation          [skip if redundant] -> reuse state
              |                             |
       +------+------+                      v
       v             v               observation
 YOLO26n         LFM2.5-VL-450M            |
 (detector)      (bf16, scene tags)  +-----+-----+
       |             |               v           v
       +------+------+          detector      embedding
              v                  + VLM       (ResNet18)
      semantic metadata               |           |
              |                        v           v
              v                  semantic     FAISS Top-K
       visual embedding          scoring           |
              |                  (Stage 23,        v
              v                   NEW)      candidate scoring
      temporal segmentation           |            |
              |                        +-----+------+
              v                              v
       place formation              temporal state model
              |                              |
              v                              v
       place reconciliation          graph constraints
              |                              |
              v                              v
       transition discovery           stable location
              |                              |
              v                              v
       graph construction               destination
              |                              |
              v                              v
       graph validation               shortest path
              |                              |
              v                              v
       versioned map                 simple instruction
                                              |
                                              v
                                             TTS
```

# 16. Definition of Done — Final Product Checklist

A single list, gating "complete." Everything here must be true:

- [ ] Stages 0–22: unchanged, still passing (130+ tests).
- [ ] Stage 07 incident: root cause documented (§4), `LFM2VLTagger` implemented and passing its smoke test on real hardware (not just the stub).
- [ ] `config.yaml` defaults to `yolo26n.pt` + `LFM2.5-VL-450M` + `vlm_enabled: true`.
- [ ] Stage 23 (semantic integration) implemented and tested — semantic evidence measurably separates the "two similar corridors" case.
- [ ] Stage 24 (runtime gate + batching) implemented — offline batch time improves, live loop doesn't reprocess redundant frames.
- [ ] Stage 27 (evaluation suite + ablation + failure log) implemented and run at least once.
- [ ] Stage 28 hygiene: `requirements.txt` installs clean, README matches reality.
- [ ] Stage 29: real walkthrough data exists, real accuracy is reported alongside (not instead of) the pseudo-label self-consistency number.
- [ ] Stage 30: every box in §13 checked.

Only when every line above is checked does this stop being "a working
prototype with known gaps" and become the final, submission-ready system.

# 17. Final Research Thesis

Unchanged from v1 — this is still the correct framing and doesn't need a
model swap to remain true:

> A camera-only indoor navigation system that autonomously constructs a
> semantic connectivity graph from a single-floor walkthrough and performs
> continuous visual localization by combining appearance, semantic
> landmarks, temporal continuity, and graph topology.

What v2 adds to the evidence for that thesis: the ablation table (§10)
showing each signal's incremental contribution, and — once §12 lands — a
real-world accuracy number standing next to the self-consistency number,
so the thesis is demonstrated against ground truth, not just internally
consistent.

---

# Appendix A — Corrected `requirements.txt`

```text
opencv-python
numpy
scikit-learn
torch
torchvision
faiss-cpu          # or `faiss` if running the pinned conda env from STATE.md
networkx
matplotlib
streamlit
PyYAML
pyttsx3
ultralytics>=8.4          # YOLO26n support
transformers               # pin exact floor per LFM2.5-VL-450M's model card
accelerate
pillow
pytest
# Optional — only needed if experimenting with the deprecated Qwen2.5-VL backend:
# bitsandbytes
# qwen-vl-utils
```
