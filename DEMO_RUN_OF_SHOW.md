# SimpleNav — Demo Run-of-Show (Stage 30, planner v2 §13)

What to say and do when demonstrating the final system. Adjust building/destination
names once Stage 29's real walkthrough exists (currently: the provisional
college-corridor dataset, `data/map/college_env_v1`).

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
2. Show the graph: 7 places, 6 edges, 0 validation warnings — built
   autonomously from one walkthrough (301 frames), no floor plan, no GPS.
3. Point out the evidence chain on one place record: scene types, landmarks,
   object classes, exemplar stats — the semantic graph.

## 3. Live localization (the demo's centerpiece)

1. Single-Photo tab: pick a frame from `data/frames/` and show the top-3
   candidates with their visual + semantic + temporal + graph term breakdown.
   If you have a visually-ambiguous pair (two similar corridors), that is the
   frame to show — say: *"appearance alone can't separate these; semantic
   evidence breaks the tie"* (Stage 23's exact acceptance case).
2. Live Mode: walk (or pan the camera) and let the tracker follow; call out
   the state machine when it shows UNCERTAIN on a blank wall and reacquires
   at the next landmark — *"unknown beats confidently wrong."*
3. Route demo: pick two destinations, show the weighted shortest path + the
   spoken instruction (TTS).

## 4. The numbers — and what they do NOT mean (say this verbatim)

- The ablation table (`data/evaluation/ablation_report.json`): top-1 as
  visual → +semantic → +temporal → +graph are added incrementally. Each row
  is a full tracker re-run; the progression (expected monotonic) is the
  evidence for *which* signal buys what.
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
- Runtime: the gate skips redundant frames (5.57× batched VLM on the offline
  pass — measured, `scripts/time_batch_vs_sequential.py`).
- What's next: real walkthrough dataset (Stage 29), optional DINOv2 encoder
  comparison, YOLOE-26n open-vocabulary doors/stairs.
