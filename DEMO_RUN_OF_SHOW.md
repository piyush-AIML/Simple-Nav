# SimpleNav — Demo Run-of-Show (Stage 35, planner v3 §8)

What to say and do when demonstrating the final system. Adjust building/destination
names once Stage 29's real walkthrough exists (currently: the provisional
college-corridor dataset, `data/map/college_env_v1`).

Everything here is literally true of the running app: since Stage 35 the app
runs the same MapBundle + LocalizationTracker pipeline the evaluation suite
validates (visual + semantic + temporal + graph, Bayes filter, state machine,
runtime gate) — not a legacy stack that only *looks* the same.

## 0. Pre-flight (2 minutes, before the audience arrives)

```bash
conda run -n ML python scripts/validate_config.py      # must print "config valid"
conda run -n ML python -m pytest tests/ -q -m "not slow"   # must be green
conda run -n ML python evaluate_suite.py               # refresh reports
```

Load nothing else on the GPU. Start the Streamlit app in a terminal, ready to show:
`streamlit run app.py`.

## 1. The one-sentence thesis

> A camera-only indoor navigation system that autonomously constructs a
> semantic connectivity graph from a single-floor walkthrough and performs
> continuous visual localization by combining appearance, semantic
> landmarks, temporal continuity, and graph topology.

Say this first, then everything below is evidence for it.

## 2. Walk the pipeline (map view)

1. Show the versioned map artifact: `data/map/college_env_v1/` — manifest,
   places.json, graph.json, exemplars, vector index. One folder, versioned,
   reproducible.
2. Show the graph: 8 places, 7 edges, 0 validation warnings — built
   autonomously from one walkthrough (301 frames), no floor plan, no GPS
   (DINOv2-with-Registers encoder, planner v3 §4).
3. Point out the evidence chain on one place record: scene types, landmarks,
   object classes, exemplar stats — the semantic graph.

## 3. Live localization (the demo's centerpiece)

1. Single-Photo tab: pick a frame from `data/frames/` and show the top-3
   candidates with their visual + semantic + temporal + graph term breakdown
   (the app's evidence table — real semantic evidence from the same
   LFM2.5-VL-450M model that built the map, not a placeholder). The tracker
   state + calibrated confidence level are on screen. If you have a
   visually-ambiguous pair (two similar corridors), that is the frame to
   show — say: *"appearance alone can't separate these; semantic evidence
   breaks the tie"* (Stage 23's exact acceptance case).
2. Live Mode: walk (or pan the camera) and let the tracker follow; call out
   the state machine when it shows UNCERTAIN on a blank wall and reacquires
   at the next landmark — *"unknown beats confidently wrong."*
3. Route demo: pick two destinations, show the weighted shortest path + the
   spoken instruction (TTS).
4. `live_navigate.py` terminal: debug-level log lines show the runtime gate
   skipping redundant frames (`gate: redundant`) before the expensive
   encoder/detector/VLM path — the Stage 24 gate doing its job in the live
   product.

## 4. The numbers — and what they do NOT mean (say this verbatim)

- The ablation table (`data/evaluation/ablation_report.json`): top-1 as
  visual → +semantic → +temporal → +graph are added incrementally. Each row
  is a full tracker re-run. Be honest about the shape: on pseudo-labels the
  progression is NOT monotonic (0.814 visual-only vs 0.734 full) — semantic
  evidence is measured against the visually-ambiguous subset (Stage 36's
  analysis), not against overall top-1, and the temporal/graph smoothing's
  payoff shows up as a 2.6× lower false-jump rate, not higher top-1.
- The encoder comparison (`data/evaluation/encoder_comparison.json`): the
  DINOv2-with-Registers swap was gated on a 2.64× better separation margin
  (0.132 vs 0.050) — appearance was measured and chosen, not assumed.
- **Honesty slide:** every number on the provisional dataset is
  *self-consistency against pseudo-labels*, not physical accuracy — 0.885 is
  the baseline harness's top-1 self-consistency, not 88.5% true accuracy.
  Show `pseudo vs real accuracy` side by side once Stage 29's real
  walkthrough + hand labels land; the delta between the two columns is
  itself the honest result.
- The failure log (`data/evaluation/failure_log.jsonl`) is the third slide:
  one LOST event with its frame, top-3 candidates, and tags — *"here's what
  the system got wrong and why"* is stronger than a single number.

## 5. Close

- What was hard: the VLM quantization incident (planner v2 §4) — one slide,
  root cause + the bf16 LFM2.5-VL-450M fix.
- Runtime: the gate skips redundant frames in the live loop (and the offline
  pass batches the VLM 5.57× — measured, `scripts/time_batch_vs_sequential.py`).
- What's next: real walkthrough dataset (Stage 29 — the one number that will
  replace pseudo-label self-consistency), the ambiguous-subset semantic
  analysis (Stage 36), YOLOE-26n open-vocabulary doors/stairs.
