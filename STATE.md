# SimpleNav — Project State (2026-09-02, planner v2 implementation pass)

> Snapshot for handoff: what exists, what's verified, what's next, and the
> current problems. Master spec: `LATEST-SimpleNav_FINAL_GRAPH_LOCALIZATION_PLANNER_v2.md`
> (replaces the old planner; old PLANNER/EXECUTOR docs = historical record
> for Stages 0-22 only).

---

## 1. What exists now (verified working)

**Environment:** conda env `ML` (Python 3.11, torch 2.10+cu126, faiss 1.13.2,
networkx 3.6.1, ultralytics 8.4.137, transformers 5.3.0). GPU: NVIDIA RTX
3050 **6 GB**. Run everything as `conda run -n ML python ...`. Second env
`florence` exists — use only when flash attention is needed. (User decision:
the ML env IS the canonical environment; no fresh-venv acceptance check —
do not create new venvs.)

**Test suite:** `conda run -n ML python -m pytest tests/ -q -m "not slow"`
→ **156 passed**, 7 slow deselected (slow = real YOLO/VLM/encoder smoke).

### Milestones A–C (Stages 00–22) ✅ — unchanged

- Baseline harness top-1 **0.8852** / top-3 0.918 pseudo-label
  self-consistency (re-verified 2026-09-02, no regression). Legacy majority-
  vote pipeline — NOT the same metric/GT as the v2 tracker suite below.
- Observation model/store, smart sampling, adaptive segmentation,
  places/reconciliation, transitions, graph + validation, versioned map,
  retrieval, scoring, graph constraints, Bayes estimator, tracker state
  machine, stabilization, reacquisition, confidence — 130 original tests.

### Planner v2 §5 — Perception swap ✅ (DONE this session)

- `detector_model: yolo26n.pt` (config-only swap; `YoloDetector.name` now
  derives from the weights file).
- `LFM2VLTagger` (`LiquidAI/LFM2.5-VL-450M`, bf16, no bitsandbytes) added;
  dispatch branch LFM2 → Qwen(opt-in) → SmolVLM2-500M; SmolVLM id updated.
- **Real-hardware smoke PASSED**: coherent, schema-valid tags
  (e.g. `corridor`, `"A long corridor with windows and a bench."`). Stage 07
  incident closed — the Qwen 4-bit garbage was a documented transformers
  ≥4.50 regression (planner v2 §4).
- 301 observations re-tagged (batched, 3.5 min): scene distribution
  corridor 123 / unknown 138 / room 31 / elevator 6 / entrance 3; landmarks
  sparse (18/301 non-empty); 72 raw scene flips per pass → flicker.

### Stage 23 — Semantic evidence ✅ (DONE)

- `src/localization/semantic_scoring.py`: 0.4 scene + 0.4 landmark Jaccard +
  0.2 object Jaccard; Rule-4 asymmetries (query-side missing → neutral 0.5,
  place-side absence → 0.0, disjoint → floor 0.1, no evidence → 0.5, stub
  mode → object-only). Never raises.
- Wired into `score_candidates` (replaces the legacy placeholder term; old
  `semantic_term` kept for its tests). `Place.object_classes` added
  (populated in `build_places`, merged in reconciliation, serialized).
- `tests/test_semantic_scoring.py` incl. the two-corridors ≥0.4 separation
  case. **OPEN FINDING — see §2.1:** on pseudo-labels, semantic evidence
  currently HURTS top-1 (0.761 → 0.684).

### Stage 24 — Runtime gate + batching ✅ (DONE)

- `src/runtime/gate.py`: reuses Stage 03 32×32 descriptor + novelty
  threshold; `forced_interval` via `runtime.max_stale_seconds: 5`; fails
  toward expensive on corrupt frames. `tests/test_runtime_gate.py` (6).
- Batched offline pass: `YoloDetector.detect_batch`, `LFM2VLTagger.tag_batch`
  (single vision-tower pass, per-item fallback), wired in
  `attach_detections`/`attach_scene_tags` via `runtime.{vlm,detector}_batch_size`.
- **Measured on 36 real frames: VLM 90.8s → 16.3s (5.57×); detector 1.85s →
  0.74s** (`scripts/time_batch_vs_sequential.py`).

### Stage 27 — Evaluation suite ✅ runtime / ⚠️ ablation explanation PENDING

- `evaluate_suite.py` runs end-to-end: localization metrics (top-1/3,
  segment acc, transition rec/prec, false-jump, reacquisition, unknown
  precision), graph metrics, incremental ablation; writes
  `data/evaluation/ablation_report.json` + `decision_log.jsonl`.
- `scripts/failure_inspector.py` → `data/evaluation/failure_log.jsonl`
  (2 LOST cases dumped with frame + top-3 + tags, verified).
- Current numbers (map-built pseudo-labels, graph variant):
  top-1 0.6246, top-3 (see report), node purity 0.547, edge prec/recall
  1.0, fragmentation 0. **Ablation: visual 0.761 → +semantic 0.684 →
  +temporal 0.628 → +graph 0.625 — non-monotonic, explanation required**
  (planner §10: "monotonic or explained-if-not"). See §2.1.

### Stage 28 — Hygiene ✅ (DONE)

- `requirements.txt` corrected per planner v2 Appendix A.
- README fully rewritten: v2 architecture diagram, tech-stack table, model
  names, honesty note, future-work updated.
- `pytest.ini` registers the `slow` mark.
- Fresh-venv acceptance: **waived by user** (ML env is canonical).

### Stage 30 — Hardening (DONE except items needing Stage 29)

- `validate_config()` in src/utils + `scripts/validate_config.py`: paths
  resolve, model ids registered, LFM2-int4 forbidden, weights sane; runs as
  the first line of `build_map.py` (hard errors abort).
- `backend_banner()` in src/perception — printed at startup in
  embed_frames/build_map; factories still log + warn on stub fallback.
- `DEMO_RUN_OF_SHOW.md` written.
- app.py/live_navigate.py remain on the LEGACY stack — planner v2 does not
  require swapping them (noted, not a gap).

### Map artifact (rebuilt with the new perception stack)

`data/map/college_env_v1`: **5 places / 4 edges / 301 obs** (was 7/6 in the
stub era). 1 validation warning: Place 0 (152 obs) mixes corridor 37 /
room 20 — genuine, kept. Revisit merge no longer fires (see §2.2).
Places now carry real scene tags, landmarks, object classes.

---

## 2. Current problems / next actions

### 2.1 Ablation is non-monotonic — the main open finding

visual 0.761 → +semantic 0.684 → +temporal 0.628 → +graph 0.625 (top-1,
pseudo-label self-consistency). Candidate explanations to verify next
session:
1. **Pseudo-label GT is circular**: GT = the map's own place assignment;
   per-frame semantic tags are noisy (72 flips/301; generic landmarks like
   "white wall"/"door"), so the semantic term adds noise around an already
   decisive visual signal (w_visual 0.5 at temperature 0.1).
2. **Temporal/graph drop is the classic smoothing tradeoff**: the Bayes
   filter lags at transition frames (K-consecutive + self-prior 0.7) —
   expected; should show up as a lower false-jump rate, not lower top-1.
   Check false_jump_rate + segment_accuracy in the report before judging.
3. **Planned measurement** (was mid-implementation at stop): the
   visually-ambiguous subset — frames where top-2 visual margin < 0.05 —
   comparing visual-only vs +semantic top-1 ON THAT SUBSET. That is the
   planner's actual acceptance for semantic evidence ("improve on the
   subset tagged visually ambiguous"); overall top-1 need not rise.
4. If the subset analysis shows no benefit: re-tune the semantic term
   (e.g. scene-type-only when landmarks are empty; weight landmark/object
   terms down when they're single generic words) — gate any change on the
   subset metric, not on overall top-1.

Note for honesty: the v2 tracker's 0.62-0.76 top-1 is NOT comparable to the
legacy 0.885 (different pipeline — Bayes tracker vs majority-vote — and
different GT). Never present them as a regression without that context.

### 2.2 Map structure changed vs the verified stub-era baseline

8 segments → 7 places (1 revisit merge) became 5 segments → 5 places (0
merges). Causes, both direct consequences of real VLM tags:
- Scene-type flicker was over-segmenting (35 segments) — fixed with the
  persistence gate in `change_scores` (scene change counts only if the new
  type holds 3 frames; unknown is neutral) + ±2-frame scene-type debounce
  in `attach_scene_tags`. Result: 5 clean segments.
- The revisit merge (both-empty-landmarks + vis ≥0.92) no longer applies
  because places now HAVE landmarks; real VLM landmarks are generic and
  disjoint across revisits so `landmark_strong (≥0.5)` rarely fires →
  revisits stay split. **Calibration decision deferred to Stage 29 real
  labels** — do not re-tune merge thresholds on pseudo-labels.
- Validator `suspicious_node` now excludes "unknown" (missing evidence,
  Rule 4) and requires ≥20% known-type minority before warning.

### 2.3 Stage 29 — blocked on the USER

Real walkthrough videos must be recorded (two independent passes, different
day/lighting) + hand labels via `scripts/make_contact_sheet.py`. Nothing
else in the pipeline needs user input.

### 2.4 Optional / not required (planner v2)

§5.4 DINOv2 encoder comparison; §5.1 YOLOE-26n open-vocab detection
("Stage 23b"); §26 LFM2 text-mode instruction templating (f-string template
first — not started, only needed if the demo phrasing feels robotic).

---

## 3. How the pieces fit (quick map)

```
build_map.py:  validate_config → banner → ObservationStore → segmentation
               (visual + persisted semantic change) → places (scene tags,
               landmarks, object classes) → reconciliation → transitions →
               graph+junctions → validation → MapBundle
tracker.py:    Camera embedding (+tags+objects) → CandidateRetriever
               (LOCAL/GLOBAL) → score_candidates (visual + semantic_similarity
               + temporal + graph) → StateEstimator (evidence likelihood ×
               graph prior) → state machine → confidence → route/directions
gate.py:       cheap novelty gate (Stage 03 descriptor) before the live loop
evaluate_suite.py: tracker passes ×4 ablations → metrics + ablation_report.json
               + decision_log.jsonl; failure_inspector.py → failure_log.jsonl
benchmark_baseline.py: permanent legacy harness (0.8852 — unchanged)
```

## 4. Config notes (key non-defaults)

- `perception.detector_model: yolo26n.pt`; `vlm_enabled: true`;
  `vlm_model: LiquidAI/LFM2.5-VL-450M`; `vlm_quantization: bf16` (never 4bit)
- `runtime.max_stale_seconds: 5.0`, `vlm_batch_size: 12`,
  `detector_batch_size: 16`
- `mapping.segment_distance_threshold: null` → adaptive (mean + 2.5σ)
- `mapping.minimum_edge_support: 1`; `merge_visual_extra_threshold: 0.92`
- `localization.lost_unknown_mass: 0.6`; evidence-only likelihood
  (temporal/graph never in the likelihood)
- Scene-type debounce window ±2 (embed_frames); semantic persistence
  window 3 (segmentation)
