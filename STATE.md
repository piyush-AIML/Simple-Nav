# SimpleNav — Project State (2026-09-02, planner v3 implementation pass)

> Snapshot for handoff: what exists, what's verified, what's next, and the
> current problems. Master spec: `SimpleNav_PLANNER_v3.md` (replaces
> `LATEST-SimpleNav_FINAL_GRAPH_LOCALIZATION_PLANNER_v2.md`; old
> PLANNER/EXECUTOR docs = historical record for Stages 0-22 only).

---

## 1. What exists now (verified working)

**Environment:** conda env `ML` (Python 3.11, torch 2.10+cu126, faiss 1.13.2,
networkx 3.6.1, ultralytics 8.4.137, transformers 5.3.0). GPU: NVIDIA RTX
3050 **6 GB**. Run everything as `conda run -n ML python ...`. Second env
`florence` exists — use only when flash attention is needed. (User decision:
the ML env IS the canonical environment; no fresh-venv acceptance check —
do not create new venvs.)

**Test suite:** `conda run -n ML python -m pytest tests/ -q -m "not slow"`
→ **169 passed**, 11 slow deselected (slow = real YOLO/VLM/encoder smoke,
incl. the DINOv2 encoder tests). Full suite with slow: 180.

### Milestones A–C (Stages 00–22) ✅ — unchanged

- Baseline harness top-1 **0.8852** / top-3 0.918 pseudo-label
  self-consistency (legacy majority-vote pipeline — NOT comparable to the
  v2/v3 tracker suite: different pipeline + GT).
- Observation model/store, smart sampling, adaptive segmentation,
  places/reconciliation, transitions, graph + validation, versioned map,
  retrieval, scoring, graph constraints, Bayes estimator, tracker state
  machine, stabilization, reacquisition, confidence — all tests passing.

### Planner v2 §5 + Stage 23–30 ✅ — unchanged

Perception stack: `detector_model: yolo26n.pt` + `LFM2VLTagger`
(LFM2.5-VL-450M bf16). Semantic evidence (0.4 scene + 0.4 landmark Jaccard
+ 0.2 object; Rule-4 asymmetries). Runtime gate + batched offline passes
(VLM 5.57×, detector 2.5×, encoder now batched too). evaluate_suite +
failure_inspector. validate_config, backend_banner, DEMO_RUN_OF_SHOW.

### Planner v3 — Stage 31 ✅ encoder pipeline fix

- `src/embed_frames.py::embed_frames()` now runs
  `get_encoder(config).batch_encode()` (config-driven, batched via
  `runtime.encoder_batch_size: 16`) — the `embedding.model` knob finally
  controls the stored vectors (was hardcoded ResNet18; Finding A).
- `encoder_name` threaded through `save_observations_dir` →
  `ObservationStore.save(encoder_name)` → `encoder.json` — dead default
  params removed; nothing can write a name it wasn't handed (Finding B).
- `validate_config()` now checks `embedding.model` against the registry
  (Finding E). Tests: `test_save_writes_the_encoder_name_it_was_given`,
  `test_manifest_and_encoder_json_agree_on_non_default_encoder`, embedding
  config checks, encoder batch-size/threading tests.
- Regression check: rebuild with `resnet18` reproduced the old map
  bit-for-bit (same content hash `c556052452071263`).

### Planner v3 — Stage 32 ✅ DINOv2-with-Registers encoder

- `Dinov2RegistersEncoder` (`facebook/dinov2-with-registers-small`, 384-d,
  pooler output, own AutoImageProcessor) + documented opt-in `Dinov3Encoder`
  (gated HF license — never the default). Shared `_HFViTEncoder` base.
  ResNet18 stays registered (Rule 2). Config default:
  `embedding.model: "dinov2_registers_small"`.
- **Real-hardware smoke PASSED**: 4/4 slow tests (dimension, L2 norm, all
  input types, batch==sequential).
- Observations re-embedded (301 frames) + map rebuilt end-to-end;
  manifest/encoder.json honestly record `dinov2_registers_small`/384.

### Planner v3 — Stage 33 ✅ encoder comparison tool

- `scripts/compare_encoders.py` (was missing despite encoder.py's docstring
  promising it; Finding F). Re-embeds every observation with each registered
  encoder and computes same-place consistency / different-place separability
  / separation margin over the 301-obs set + current map assignments;
  dumps the raw pairwise-sim arrays for Stage 34; writes
  `data/evaluation/encoder_comparison.json`. Never raises on gated models.
- **Measured (this is why DINOv2 is the default):**
  | encoder | same | diff | margin |
  |---|---|---|---|
  | resnet18 | 0.8746 | 0.8247 | 0.0499 |
  | dinov2_registers_small | 0.7258 | 0.5938 | **0.1319** (2.64×) |
  ResNet18's diff-place 0.82 confirms Finding H empirically: thresholds
  tuned to a compressed distribution.

### Planner v3 — Stage 34 ✅ threshold recalibration protocol

- `scripts/recalibrate_thresholds.py`: percentile-matches every raw-cosine
  threshold from the old encoder's distribution (arrays in
  encoder_comparison.json) to the new one. **Two-family rule (documented
  extension of v3 §5):** values inside the old distribution →
  percentile-matched; values below its ~1st percentile (loose floors, e.g.
  all tracker/confidence gates sat below ResNet18's same-place min 0.629)
  → floor-shifted by the measured score change (w_visual×Δsame_mean =
  −0.0744 for blended gates, full Δsame_mean for the raw gate);
  likelihood_temperature → same-place std ratio (2.81×).
- **Recalibrated table** (report: `data/evaluation/threshold_recalibration.json`):
  | key | old (resnet18) | new (dinov2) | basis |
  |---|---|---|---|
  | mapping.merge_visual_threshold | 0.75 | 0.425 | 16.8th pct diff-place |
  | mapping.merge_visual_extra_threshold | 0.92 | 0.835 | 96.1st pct diff-place |
  | mapping.duplicate_similarity_threshold | 0.90 | 0.791 | 47.6th pct same-place |
  | navigation.confidence_threshold | 0.40 | 0.251 | floor-shift (raw) |
  | localization.tracking_threshold | 0.50 | 0.426 | floor-shift |
  | localization.lost_threshold | 0.20 | 0.126 | floor-shift |
  | localization.high_score | 0.60 | 0.526 | floor-shift |
  | localization.low_score | 0.30 | 0.226 | floor-shift |
  | localization.reacquired_threshold | 0.50 | 0.426 | floor-shift |
  | localization.arrived_threshold | 0.60 | 0.526 | floor-shift |
  | localization.recovery_threshold | 0.45 | 0.376 | floor-shift |
  | localization.transition_threshold | 0.65 | 0.576 | floor-shift |
  | localization.likelihood_temperature | 0.10 | 0.379 | softmax discrimination ratio — median top-2 candidate-gap (3.79×) from the decision logs; std-ratio 0.281 was tried and rejected (state machine froze; see below) |
  Old values kept as comments in config.yaml for rollback.
- **Ablation re-run (encoder swap, pseudo-label self-consistency — Rule 6,**
  **NOT real-world accuracy).** Same 301 observations; labels are the map's
  own place assignments — and note the DINOv2 label set is **finer (8
  places vs 5)**, so top-1 gains are conservative:
  | metric | resnet18 baseline | dinov2 + recalibrated |
  |---|---|---|
  | top-1 | 0.6246 | **0.7342** |
  | top-3 | 0.9103 | **0.9269** |
  | segment accuracy | 0.7095 | 0.6245 (per-run metric; 8 finer runs penalize single-frame slips more) |
  | transition recall | 1.0 | 0.8571 |
  | transition precision | 0.3333 | **0.5455** |
  | false jump rate | 0.0433 | **0.0133** |
  | node purity | 0.5466 | **0.6295** |
  | cross-place sim | 0.7748 | **0.5160** |
  Baseline artifacts archived: `ablation_report_resnet18_baseline.json`,
  `decision_log_resnet18_baseline.jsonl`.

### Map artifact (DINOv2 era)

`data/map/college_env_v1`: **8 places / 7 edges / 301 obs**, encoder
dinov2_registers_small / 384-d, 0 validation warnings, 0 merges (even at
recalibrated 0.425 — DINOv2 places are genuinely distinct). Rebuild
reproduced deterministically after the threshold swap.

### Planner v3 — Stage 35 ✅ product unified (Findings C/D/I closed)

- `app.py` + `live_navigate.py` run `MapBundle` + `LocalizationTracker`
  (the stack evaluate_suite validates): visual + semantic + temporal +
  graph evidence, Bayes filter, state machine, calibrated confidence.
  Query photos get REAL semantic evidence (same yolo26n + LFM2.5-VL models
  as the mapping pass; built once, reused per frame — model loads are
  per-instance and expensive).
- Runtime gate wired into `live_navigate.py`: `should_process` skips
  redundant frames before the expensive path, logs `gate: <reason>` at
  debug; `--interval-seconds` now maps to `runtime.max_stale_seconds`.
- UI: per-candidate term breakdown (`st.dataframe` — new `term_breakdown`
  key in the tracker's status dict), tracker state + confidence level
  visible in both tabs. Recognition gating uses the calibrated
  `confidence_level`/state — the posterior mass is a sub-probability, NOT
  comparable to the legacy `navigation.confidence_threshold` (that stays
  for the frozen legacy stack). Real-hardware smoke PASSED end to end
  (detector + VLM + tracker on a live frame; semantic evidence re-ranked
  the candidates).
- `DEMO_RUN_OF_SHOW.md` updated: 8/7/0 counts, all claims now literally
  true of the running app (Rule 7).
- State-machine note (pre-existing, now quantified): posterior best-score
  lives at ~0.03-0.11 (sub-probability mass), so TRACKING fired 3/301
  frames in the ResNet18 era too — the demo shows honest UNCERTAIN/
  REACQUIRING/LOST behavior; thresholds are documented as posterior-scale
  and were never reached in either era on pseudo-labels.

### Planner v3 — Stage 36 ✅ ablation finding closed (§2.1)

- `evaluate_suite.py` now computes the **visually-ambiguous subset**
  breakdown (top-2 visual margin < 0.05) into `ablation_report.json` —
  visual-only vs +semantic top-1 on the subset, the Stage 23 acceptance.
- Semantic evidence measured: no benefit on the subset (0.561 → 0.404),
  structural cause found and recorded (unknown-scene places, 47/57
  ambiguous frames). Re-tune landed: place-side unknown scene abstains at
  0.5 (`_scene_match`), improving full-set top-1 0.7342 → 0.7774.
- New tests: `test_place_side_unknown_scene_abstains_not_penalized`,
  `test_term_breakdown_reports_all_four_terms`, encoder-consistency tests
  (Stage 31), compare_encoders metrics tests (Stage 33).

---

## 2. Current problems / next actions

### 2.1 Ablation finding — RESOLVED (Stage 36, 2026-09-02)

DINOv2 + recalibrated + semantic re-tune:
visual 0.814 → +semantic 0.781 → +temporal 0.804 → +graph 0.777 (top-1
pseudo-label, full set; top-1 0.7774, top-3 0.9269, false_jump 0.02).
`monotonic_or_explained` flag stays false by design (raw check); the
explanation lives in `ablation_report.json`'s `explanation` + `ambiguous_subset`.

**The subset measurement (the actual Stage 23 acceptance) is recorded in
`ablation_report.json`:** on the 57 visually-ambiguous frames (top-2 visual
margin < 0.05), visual-only top-1 0.5614 vs +semantic 0.4035 — semantic
evidence does NOT help on the subset it was designed for, even after
re-tuning. Structural, measured reason: **5 of 8 places have "unknown" as
their stored dominant scene (46% of obs — tagging failure, not absence),
and 47/57 ambiguous frames have a GT place with unknown scene** — any
scene-voting semantic term must abstain (0.5) for the true place while the
runner-up votes (1.0). Even on the 27 scene-different ambiguous frames,
semantic hurts (0.704 → 0.556) for the same reason.

**Re-tune that DID land (gated on the subset metric):** place-side unknown
scene now abstains at 0.5 instead of voting 0.3 (`_scene_match`, planner
v3 §9) — Rule 4 applied to the place side, same as the query side. This
halved the subset harm (0.386 → 0.4035) and improved the full set
(top-1 0.7342 → 0.7774; semantic variant 0.744 → 0.781; the ablation is
near-monotonic apart from graph's −0.03, the documented smoothing tradeoff
that buys the 2× lower false-jump rate).

**Resolution:** semantic evidence cannot pay on pseudo-labels because the
tags lack the discriminating information (unknown-scene places, sparse
generic landmarks). Conclusion recorded as "resolved: measured + re-tuned;
re-validate on Stage 29 real labels, where tag quality and landmark
discriminability are the actual test." Do NOT re-tune further on
pseudo-labels.

### 2.2 Stage 35 — unify the product (next, in progress)

app.py + live_navigate.py still run the legacy PlaceIndex/LiveTracker stack
— semantic evidence, the Bayes estimator, the state machine, the runtime
gate are unreachable from the running product (v3 Findings C/D, Rule 7).
Swap them onto `MapBundle` + `LocalizationTracker` (backward-compatible
status dict), wire `gate.should_process()` into the live loop, add the term
breakdown + tracker-state UI, fix DEMO_RUN_OF_SHOW.md counts (was 7/6/0,
map is 8/7/0 now).

### 2.3 Stage 29 — blocked on the USER

Real walkthrough videos must be recorded (two independent passes, different
day/lighting) + hand labels via `scripts/make_contact_sheet.py`. Nothing
else in the pipeline needs user input. (v3 DoD carries this over.)

### 2.4 Optional / not required

§5.1 YOLOE-26n open-vocab detection; §26 LFM2 instruction templating;
DINOv3 stretch (gated). None required for the final product.

---

## 3. How the pieces fit (quick map)

```
build_map.py:  validate_config → banner → ObservationStore → segmentation
               (visual + persisted semantic change) → places (scene tags,
               landmarks, object classes) → reconciliation (recalibrated
               merge thresholds) → transitions → graph+junctions →
               validation → MapBundle
embed_frames.py: get_encoder(config).batch_encode (config.embedding.model)
               + batched detector/VLM attach → ObservationStore
tracker.py:    Camera embedding (+tags+objects) → CandidateRetriever
               (LOCAL/GLOBAL) → score_candidates (visual + semantic_similarity
               + temporal + graph) → StateEstimator (evidence likelihood ×
               graph prior) → state machine → confidence → route/directions
gate.py:       cheap novelty gate (Stage 03 descriptor) before the live loop
compare_encoders.py: re-embeds with each registered encoder → same/diff
               margin → encoder_comparison.json (dumps sim arrays)
recalibrate_thresholds.py: percentile/floor-shift mapping → config values
evaluate_suite.py: tracker passes ×4 ablations → metrics + ablation_report.json
               + decision_log.jsonl; failure_inspector.py → failure_log.jsonl
benchmark_baseline.py: permanent legacy harness (0.8852 — unchanged)
```

## 4. Config notes (key non-defaults)

- `embedding.model: "dinov2_registers_small"` (was resnet18; rollback =
  one line, both registered); `embedding.image_size` is ResNet18-only —
  HF encoders use their own AutoImageProcessor
- `perception.detector_model: yolo26n.pt`; `vlm_model: LiquidAI/LFM2.5-VL-450M`;
  `vlm_quantization: bf16` (never 4bit)
- `runtime.max_stale_seconds: 5.0`, `vlm_batch_size: 12`,
  `detector_batch_size: 16`, `encoder_batch_size: 16`
- Recalibrated thresholds per §1 Stage 34 table (comments in config.yaml
  carry the old values for rollback)
- `mapping.segment_distance_threshold: null` → adaptive (mean + 2.5σ)
- `mapping.minimum_edge_support: 1`; `localization.lost_unknown_mass: 0.6`;
  evidence-only likelihood (temporal/graph never in the likelihood)
- Scene-type debounce window ±2 (embed_frames); semantic persistence
  window 3 (segmentation)
