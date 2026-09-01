# SimpleNav — Project State (2026-09-02)

> Snapshot for handoff: what exists, what's verified, what's next, and the
> current problem. Companion docs: `SimpleNav_FINAL_GRAPH_LOCALIZATION_PLANNER.md`
> (spec) and `SimpleNav_FINAL_GRAPH_LOCALIZATION_EXECUTOR.md` (runbook).

---

## 1. What exists now (verified working)

**Environment:** conda env `ML` (Python 3.11, torch 2.10+cu126, faiss 1.13.2,
networkx 3.6.1, ultralytics 8.4.137, bitsandbytes 0.50.2, transformers 5.3.0).
GPU: NVIDIA RTX 3050 **6 GB**. Run everything as `conda run -n ML python ...`
(no `python` on PATH).

**Test suite:** `conda run -n ML python -m pytest tests/ -q` → **130 passed**
(6 slow deselected; slow = YOLO + encoder + VLM smoke).

### Milestone A — Baseline (Stages 00-01) ✅
- `scripts/benchmark_baseline.py` — permanent harness: 80/20 temporal split of
  the 301 provisional frames, legacy pipeline (ResNet18 → KMeans → transition
  graph → FAISS → majority vote) on train 80%, metrics on unseen 20%.
- Provisional labels: full-data KMeans pseudo-labels (vision-less fallback —
  the executor model has no image input). Results in
  `data/evaluation/reports/baseline_report.json`:
  top-1 self-consistency **0.885**, top-3 **0.918**, false-jump **0.170**,
  transition recall **1.0**, precision **0.857**, latency ~22 ms.
- Real area labels can replace pseudo-labels via `heldout_labels.json` +
  `exemplar_labels.json` (templates shipped; contact-sheet tool
  `scripts/make_contact_sheet.py`).

### Milestone B — Robust Mapping (Stages 02-14) ✅
- `Observation` dataclass + JSONL I/O (`src/mapping/observations.py`)
- Smart sampling: quality gate (blur/dark) + mean-subtracted 32×32 descriptor
  novelty + sustained-change retention (`src/extraction/quality.py`,
  `src/extraction/frames.py`); `extract_frames.py --baseline` keeps fixed-N
- `VisualEncoder` abstraction + registry (`src/embeddings/encoder.py`),
  ResNet18 baseline; `embedder.py` = compat wrapper
- `ObservationStore` (FAISS + JSONL + id_order; `src/mapping/observation_store.py`),
  `data/observations/` populated with 301 observations + YOLO detections
- Temporal segmentation with **adaptive threshold** (mean + 2.5σ of change
  scores; fixed 0.35 found nothing) and zone-based cuts
  (`src/mapping/segmentation.py`)
- Place formation + exemplars (kmeans/diversity/temporal_diversity),
  reconciliation with multi-signal merge + `merge_visual_extra_threshold: 0.92`
  (`src/mapping/place_builder.py`, `place_reconciliation.py`)
- Debounced transition extraction (`transition_builder.py`), confidence-
  gated graph + junction detection (`graph_builder.py`), validation report
  (`graph_validator.py`)
- **Versioned map artifact** (`map_artifact.py`): `data/map/college_env_v1/`
  with manifest, places.json, graph.json, exemplars, vector_index; `MapBundle`
  loader; `src/mapping/build_map.py` orchestrator chains 02→14.
- **Real result on 301 obs:** 8 segments → 7 places (1 revisit merged) → 6
  edges, 0 validation warnings. Linear corridor chain 0-1-2-3-4-6-7.

### Milestone C — Robust Localization (Stages 15-22) ✅
- `CandidateRetriever` (top-K per place + margins; `src/localization/retrieval.py`)
- `score_candidates` (visual/semantic/temporal/graph weighted terms;
  `candidate_scoring.py`)
- Graph constraints: soft distance penalties, local-first policy
  (`graph_constraints.py`)
- `StateEstimator` Bayes filter (`state_estimator.py`):
  posterior ∝ L·P(s|s′)·prior; **likelihood = visual+semantic evidence only**
  (temporal/graph enter ONLY via the transition prior — double-counting made
  weak evidence look strong and LOST unreachable); explicit unknown mass;
  normalized transition matrix (far mass accounted)
- `LocalizationTracker` (`src/localization/tracker.py`): TRACKING/UNCERTAIN/
  LOST/REACQUIRING/ARRIVED state machine, K-consecutive + high-posterior
  confirmation, LOCAL/GLOBAL modes, decision log; status dict backward-
  compatible with legacy `LiveTracker`
- `estimate_confidence` → HIGH/MEDIUM/LOW/UNKNOWN (`confidence.py`)
- Tests: `test_tracker.py` (10), `test_state_estimator.py` (9),
  `test_confidence.py` (6), `test_retrieval.py` (4), `test_candidate_scoring.py` (9)

---

## 2. Current problem / pending items

1. **Stage 07 VLM — PARKED (deferred by user):** `Qwen2.5-VL-3B-Instruct`
   4-bit loads and runs on the 3050 but produces **incoherent text** (echoes
   the prompt, emits garbage JSON like `{ ( and corridor, Junction: unknown }`).
   `_strip_prompt_echo` + max_new_tokens=256 were added but output was still
   garbage. Suspected: 4-bit quantization incompatibility on this GPU, or a
   transformers 5.3 chat-template mismatch. `perception.vlm_enabled: false`
   in config.yaml — pipeline runs on the deterministic `StubTagger`. Re-enable
   after debugging (candidate: try float16, or SmolVLM2, or transformers pin).

2. **Stage 23 (semantic localization at runtime):** the scoring path supports
   query tags/objects, but the tracker's runtime evidence comes from visual
   only until the VLM works. Unit-level tests exist for the scoring logic.

3. **Stage 24 (runtime gate):** not started — cheap quality/novelty gate for
   the live loop, latency p50/p95 reporting.

4. **Stages 25-29:** evaluation suite (map + localization metrics), ablations,
   failure analysis inspector, UI integration (`app.py`/`live_navigate.py` on
   `LocalizationTracker` with `mode: baseline|v2` toggle), final cleanup.

5. **Dataset:** provisional (301 frames, no source video). Real walkthrough
   videos + real area labels would replace pseudo-labels and validate
   physically (not just self-consistency).

---

## 3. How the pieces fit (quick map)

```
build_map.py:  ObservationStore → segmentation → places → reconciliation
               → transitions → graph+junctions → validation → MapBundle
tracker.py:    Camera embedding → CandidateRetriever (LOCAL/GLOBAL) → scoring
               → StateEstimator (evidence likelihood × graph prior) → state
               machine → confidence → route/directions (navigate.py, legacy)
benchmark_baseline.py: permanent before/after harness (provisional labels)
```

## 4. Config notes (key non-defaults)

- `mapping.segment_distance_threshold: null` → adaptive (mean + 2.5σ)
- `mapping.minimum_edge_support: 1` → single walkthrough = 1 crossing per edge
- `mapping.merge_visual_extra_threshold: 0.92` → same-building ResNet18
  cross-area similarity ≈ 0.87; 0.85 merged everything
- `localization.lost_unknown_mass: 0.6` → LOST fires on unknown mass (weak
  evidence never drops posterior below lost_threshold otherwise)
- `perception.vlm_enabled: false` → stub tagger active (VLM parked)
