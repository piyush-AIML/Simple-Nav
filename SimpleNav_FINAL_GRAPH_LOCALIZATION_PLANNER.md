# SimpleNav — Final Graph + Localization Upgrade Planner

> **Purpose:** turn the current SimpleNav prototype into a robust, research-grade, camera-only indoor navigation system without introducing external hardware or unnecessary complexity.
>
> **Scope:** the backbone is limited to **(1) autonomous indoor graph formation** and **(2) continuous graph-aware localization**. Perception, UI, routing, and audio remain replaceable/simple support components.

---

# 0. Final System Definition

The system accepts a **complete walkthrough video of a single floor** and builds a reusable visual-spatial map.

At runtime, a phone/webcam camera continuously observes the environment and estimates where the user is in that map. The graph constrains localization so visually similar but physically impossible locations are rejected.

The complete system is:

```text
                    OFFLINE MAP BUILDING

single-floor walkthrough video
             │
             ▼
      smart frame selection
             │
             ▼
      scene/object analysis
       ┌─────┴─────────┐
       ▼               ▼
 object detector       VLM
 boxes/classes         scene + landmarks
       └─────┬─────────┘
             ▼
      visual embeddings
             │
             ▼
      observation store
             │
             ├───────────────┐
             ▼               ▼
   temporal segmentation   visual grouping
             └──────┬────────┘
                    ▼
             persistent places
                    │
                    ▼
          transition discovery
                    │
                    ▼
          graph reconciliation
                    │
                    ▼
             graph validation
                    │
                    ▼
             VERSIONED MAP


                       RUNTIME

camera stream
     │
     ▼
smart observation sampling
     │
     ▼
object/VLM metadata
     │
     ▼
visual embedding
     │
     ▼
vector retrieval → Top-K candidates
     │
     ▼
temporal state estimator
     │
     ▼
graph constraints
     │
     ▼
stable current place
     │
     ├───────────────► destination
     │                      │
     │                      ▼
     │                 shortest path
     │                      │
     │                      ▼
     │                simple instruction
     │                      │
     │                      ▼
     │                     TTS
     │
     └────────── repeat continuously
```

---

# 1. Engineering Rules

These rules apply throughout the entire implementation.

## Rule 1 — Do not rewrite working components unnecessarily

Keep the current:

- Streamlit demo
- CLI tools
- NetworkX routing
- FAISS retrieval
- TTS
- floor-plan visualization

unless a backbone change requires an interface update.

The current repository already separates extraction, embedding, place discovery, graph construction, localization, live tracking, and UI. Preserve that general structure while replacing weak algorithms behind clean interfaces.

## Rule 2 — Keep a baseline implementation

The current pipeline must remain runnable as a baseline:

```text
ResNet18
→ K-Means
→ transition-count graph
→ FAISS
→ majority smoothing
```

Every new research component must be compared with it.

## Rule 3 — Never allow a support model to silently become ground truth

The detector and VLM provide evidence.

They do not directly define:

- coordinates,
- topology,
- graph edges,
- final localization.

## Rule 4 — Unknown is better than confidently wrong

The system must support:

```text
TRACKING
UNCERTAIN
LOST
REACQUIRING
ARRIVED
```

## Rule 5 — Retrieval proposes; temporal/graph reasoning decides

FAISS must become:

```text
candidate generator
```

not:

```text
final localization authority
```

## Rule 6 — Evaluation must use unseen observations

Never report final performance only on frames used to build the map.

---

# 2. Target Repository Structure

Recommended final structure:

```text
Simple-Nav/
│
├── app.py
├── live_navigate.py
├── evaluate.py
├── config.yaml
├── requirements.txt
│
├── components/
│   └── rear_camera/
│       └── index.html
│
├── src/
│   ├── __init__.py
│   │
│   ├── extraction/
│   │   ├── frames.py
│   │   └── quality.py
│   │
│   ├── perception/
│   │   ├── detector.py
│   │   └── scene_tagger.py
│   │
│   ├── embeddings/
│   │   └── encoder.py
│   │
│   ├── mapping/
│   │   ├── observations.py
│   │   ├── segmentation.py
│   │   ├── place_builder.py
│   │   ├── place_reconciliation.py
│   │   ├── transition_builder.py
│   │   ├── graph_builder.py
│   │   └── graph_validator.py
│   │
│   ├── localization/
│   │   ├── retrieval.py
│   │   ├── candidate_scoring.py
│   │   ├── state_estimator.py
│   │   └── tracker.py
│   │
│   ├── navigate.py
│   ├── floor_plan.py
│   ├── speak.py
│   └── utils.py
│
├── tests/
│   ├── test_extraction.py
│   ├── test_segmentation.py
│   ├── test_place_builder.py
│   ├── test_graph_builder.py
│   ├── test_graph_validator.py
│   ├── test_retrieval.py
│   ├── test_state_estimator.py
│   ├── test_tracker.py
│   └── test_navigate.py
│
└── data/
    ├── raw/
    ├── frames/
    ├── observations/
    ├── map/
    └── evaluation/
```

A gradual migration is preferable. Do not perform a giant restructuring commit.

---

# 3. Stage 0 — Freeze and Benchmark the Existing Prototype

## Goal

Create an objective baseline before changing the backbone.

## Current baseline to preserve

```text
extract_frames.py
→ embed_frames.py
→ build_places.py
→ build_graph.py
→ localize.py
→ live_tracker.py
```

The current system uses fixed-interval frame sampling, frozen ResNet18 embeddings, K-Means place discovery, multiple place exemplars, a transition-count NetworkX graph, FAISS inner-product retrieval, confidence thresholding, and majority-vote smoothing.

## Instructions

1. Create a small but representative building dataset.
2. Record at least:
   - one mapping walkthrough,
   - one independent evaluation walkthrough.
3. Keep the evaluation walkthrough visually different enough to avoid leakage.
4. Run the current pipeline.
5. Save:
   - localization accuracy,
   - top-3 accuracy,
   - false jump rate,
   - transition detection accuracy,
   - graph node count,
   - graph edge count,
   - localization latency.

## Required experiment sets

```text
A: normal lighting
B: different walking speed
C: reverse direction
D: changed viewpoint
E: partial occlusion
F: lower light
```

## Acceptance criteria

The baseline is reproducible from a clean environment and produces a machine-readable evaluation report.

## Deliverable

```text
baseline_report.json
baseline_report.md
```

---

# 4. Stage 1 — Introduce a Clean Observation Model

## Goal

Separate raw frames from the concept of a spatial observation.

## Create

```python
Observation
```

with fields conceptually equivalent to:

```python
Observation(
    id,
    timestamp,
    frame_path,
    embedding,
    quality_score,
    objects,
    scene_tags,
    landmarks,
    segment_id=None,
    place_id=None,
)
```

## Instructions

1. Stop passing loose parallel arrays everywhere.
2. Preserve frame ordering explicitly.
3. Record source video timestamp/frame index.
4. Store embedding model name/version.
5. Store image quality measurements.
6. Allow perception metadata to be absent without breaking the pipeline.

## Why

The rest of the system needs one common object representing:

```text
visual evidence at a known point in time
```

## Acceptance criteria

Every accepted frame can be traced back to:

```text
video → timestamp → image → embedding → metadata
```

---

# 5. Stage 2 — Replace Fixed Frame Sampling With Smart Sampling

## Current problem

The current extractor saves every Nth frame. This creates redundant data and can miss meaningful transitions.

## Goal

Keep frames that add new spatial information.

## Algorithm

For each incoming frame, calculate:

```text
quality
+
temporal distance
+
visual novelty
```

### Quality

Reject:

- extreme blur,
- unusably dark frames,
- invalid images.

### Visual novelty

Compare the current embedding or a cheap visual descriptor with the last accepted observation.

If nearly identical:

```text
discard
```

If sufficiently different:

```text
keep
```

### Transition-aware retention

Bias retention around strong sustained changes:

```text
large change
→ keep a few observations around the transition
```

Do not treat one spike as a transition.

## Suggested logic

```text
candidate frame
   │
   ├─ bad quality? → reject
   │
   ├─ nearly identical? → reject
   │
   ├─ strong sustained change? → keep
   │
   └─ enough temporal gap + useful novelty? → keep
```

## Acceptance criteria

The same walkthrough should contain substantially fewer redundant frames without losing:

- room entrances,
- corridor turns,
- junctions,
- stair/elevator areas,
- major visual landmarks.

---

# 6. Stage 3 — Add Lightweight Object Detection

## Goal

Add structured visual information without letting object detection control the map.

## Output

For each selected frame:

```json
{
  "objects": [
    {
      "class": "door",
      "confidence": 0.93,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

## Important navigation objects

Prioritize:

- doors,
- stairs,
- elevators,
- corridor signs,
- room signs,
- desks,
- reception areas,
- distinctive fixtures,
- fire equipment,
- entrances,
- major landmarks.

Do not attempt to detect every object.

## Instructions

1. Make detector interface independent of the specific model.
2. Return normalized bounding boxes.
3. Include confidence.
4. Apply a confidence threshold.
5. Allow detector failure without crashing mapping.

## Acceptance criteria

Frames contain stable object metadata and the detector can be swapped without touching graph code.

---

# 7. Stage 4 — Add VLM Scene and Landmark Tagging

## Goal

Turn detector output and the image into useful semantic evidence.

## VLM responsibilities

The VLM may produce:

```json
{
  "scene_type": "corridor_junction",
  "landmarks": [
    "blue room-number sign",
    "staircase on left"
  ],
  "description": "T-shaped corridor intersection with stairs on the left.",
  "navigation_relevance": [
    "junction",
    "stairs"
  ]
}
```

## The VLM must NOT produce

Do not trust it directly for:

```text
x/y coordinates
graph topology
exact distances
current location
```

## Instructions

1. Keep the prompt schema fixed.
2. Require structured JSON output.
3. Validate output before storage.
4. Remove irrelevant natural-language detail.
5. Store only navigation-relevant fields.

## Acceptance criteria

VLM output is deterministic enough to compare observations and useful enough to improve place discrimination.

---

# 8. Stage 5 — Build a Visual Encoder Abstraction

## Goal

Detach the system from ResNet18.

Create:

```python
class VisualEncoder:
    def encode(image) -> np.ndarray:
        ...
```

## Baseline implementation

Keep:

```text
ResNet18 pooled feature
```

as the baseline encoder.

## Research candidates

Evaluate compact pretrained retrieval-oriented encoders one at a time.

Candidate families:

- CLIP-like models,
- DINO-like self-supervised encoders,
- other image retrieval encoders.

## Selection criterion

Do not select the model because it is generally "better."

Measure:

```text
same-place viewpoint consistency
vs
different-place separability
```

## Acceptance criteria

The encoder can be swapped by configuration without changing graph/localization logic.

---

# 9. Stage 6 — Build the Observation Store

## Goal

Create the memory used by both graph formation and localization.

For each observation store:

```text
observation_id
timestamp
frame_path
embedding
objects
scene_type
landmarks
quality
```

Optional derived fields:

```text
segment_id
place_id
```

## Recommended architecture

Use:

```text
vector index
+
metadata store
```

The vector index handles similarity search.

The metadata store handles structured information.

Do not force every metadata field into the vector database itself.

## Acceptance criteria

Given an observation ID, the system can recover all metadata and the original frame.

---

# 10. Stage 7 — Temporal Segmentation

## Goal

Identify contiguous periods of the walkthrough corresponding to local spatial regions.

## Input

Ordered observations:

```text
O1, O2, O3, ... On
```

## Compute

For consecutive observations:

```text
visual_distance(Ot, Ot+1)
```

Also include, when available:

```text
scene change
landmark change
object change
```

## Detect stable regions

A segment is a period where:

```text
visual appearance remains reasonably stable
+
semantic scene remains compatible
```

Example:

```text
O1 O2 O3 O4 O5
      PLACE A

O6 O7
      TRANSITION

O8 O9 O10 O11
      PLACE B
```

## Important

Use a short temporal window.

Do not make a segment boundary because of one anomalous frame.

## Acceptance criteria

A simple walkthrough produces segments that broadly correspond to physical areas rather than arbitrary K-Means clusters.

---

# 11. Stage 8 — Persistent Place Formation

## Goal

Convert temporal segments into persistent physical places.

## Key concept

A place is not a cluster of images.

A place is:

```text
a spatially coherent region
with multiple visual observations
and persistent identity
```

## Place representation

Each place stores:

```text
place_id
observations
exemplars
scene types
landmarks
visual statistics
```

## Exemplar selection

Compare:

### Method A

Secondary K-Means.

### Method B

Diversity-based selection.

### Method C

Temporal-boundary + diversity selection.

The default should become whichever performs best on held-out localization.

## Acceptance criteria

Each physical place can tolerate reasonable:

- viewpoint change,
- lighting change,
- distance change,
- direction-of-travel change.

---

# 12. Stage 9 — Place Reconciliation

## Goal

Prevent duplicate places and accidental merges.

## Example

A walkthrough can produce:

```text
Segment 1 = Lobby
Segment 2 = Corridor
Segment 3 = Lobby
```

The system must be able to reconcile:

```text
Segment 1 == Segment 3
```

without merging visually similar but distinct rooms.

## Evidence for merging

Require multiple signals:

```text
visual similarity
+
landmark similarity
+
scene compatibility
+
temporal/context compatibility
```

## Evidence against merging

Reject merges when:

- landmarks conflict strongly,
- graph context conflicts,
- scene types differ substantially,
- observations are repeatedly more similar to another place.

## Acceptance criteria

The map can represent revisited physical locations as a single place while preserving distinct visually similar places.

---

# 13. Stage 10 — Transition Extraction

## Goal

Turn the sequence of places into movement evidence.

For:

```text
A A A A B B B C C
```

derive:

```text
A → B
B → C
```

## Store transition statistics

For each pair:

```text
forward_count
reverse_count
transition_duration
supporting_observations
visual_transition_strength
confidence
```

## Debouncing

Never create an edge from:

```text
A → B → A
```

based on a single noisy prediction.

Require stable persistence.

## Acceptance criteria

False one-frame transitions do not create persistent graph edges.

---

# 14. Stage 11 — Autonomous Graph Construction

## Goal

Build the physical connectivity graph.

## Graph model

```text
Node = place
Edge = physical connection
```

Do not make individual frames graph nodes in the final representation.

## Edge confidence

Use evidence from:

```text
transition frequency
+
transition persistence
+
semantic compatibility
+
repeat observations
```

## Directional evidence

Store:

```text
A → B
B → A
```

separately during construction.

Do not interpret unequal counts as physical one-way movement automatically.

The final routing graph may remain undirected.

## Acceptance criteria

The graph correctly represents the main accessible paths in the floor.

---

# 15. Stage 12 — Junction Detection

## Goal

Identify high-value graph locations.

A junction is useful because it is both:

- structurally important,
- highly informative for localization.

## Detection signals

Use:

```text
multiple outgoing transition paths
+
scene-type changes
+
visual discontinuity
+
repeated traversal
```

## Optional node metadata

```text
node_type:
    room
    corridor
    junction
    stairs
    elevator
    entrance
    unknown
```

Classification should remain soft metadata, not a hard requirement for routing.

---

# 16. Stage 13 — Graph Validation

## Goal

Detect construction errors before runtime.

## Required checks

### Connectivity

Find unexpected disconnected components.

### Isolated nodes

Find nodes with no meaningful physical evidence.

### Weak edges

Flag edges supported by too little data.

### Suspicious nodes

Flag:

- very low observation count,
- extremely high internal variance,
- incompatible semantic groups.

### Duplicate candidates

Flag pairs with:

```text
very high visual similarity
+
different graph identities
```

## Output

```text
graph_validation.json
graph_validation.md
```

Example:

```text
WARN: Place 3 and Place 9 may be duplicates.
WARN: Edge 4-7 has only one transition.
WARN: Place 6 has high internal visual variance.
```

## Acceptance criteria

Map quality can be inspected without manually opening every frame.

---

# 17. Stage 14 — Create a Versioned Map Artifact

## Goal

Make the generated map reproducible.

## Store

```text
map/
    manifest.json
    places.json
    graph.json
    exemplars.npy
    observation_metadata.json
    vector_index/
```

## Manifest fields

```json
{
  "map_id": "...",
  "building_id": "...",
  "floor_id": "...",
  "encoder": "...",
  "encoder_version": "...",
  "embedding_dimension": 512,
  "observation_count": 0,
  "place_count": 0,
  "edge_count": 0
}
```

## Acceptance criteria

Another process can load the map without depending on the exact build-time Python objects.

---

# 18. Stage 15 — Refactor FAISS Into Candidate Retrieval

## Goal

Turn nearest-neighbour search into candidate generation.

## Current behavior

```text
best exemplar
→ final place
```

## New behavior

```text
query
→ top-K exemplars
→ aggregate by place
→ candidate places
```

## Candidate record

```python
Candidate(
    place_id,
    visual_score,
    best_exemplar_id,
    supporting_exemplar_count,
    margin,
)
```

## Important metrics

For each query calculate:

```text
best_score
second_best_score
score_margin
```

## Acceptance criteria

Localization receives a meaningful candidate set rather than only one guess.

---

# 19. Stage 16 — Add Place-Level Candidate Scoring

## Goal

Combine multiple pieces of evidence.

Initial score:

```text
score =
    w_visual * visual_similarity
  + w_semantic * semantic_similarity
  + w_temporal * temporal_consistency
  + w_graph * graph_compatibility
```

Do not overfit weights immediately.

Start with interpretable normalized terms.

## Visual similarity

Compare the query against several place exemplars rather than only one.

## Semantic similarity

Compare:

```text
scene type
landmarks
important objects
```

## Acceptance criteria

A visually ambiguous query can be resolved correctly when semantic/contextual evidence is stronger.

---

# 20. Stage 17 — Introduce Graph-Constrained Candidate Filtering

## Goal

Prevent physically implausible localization.

Suppose:

```text
previous place = A
```

and graph neighbors are:

```text
A → B
A → C
```

A candidate:

```text
Z
```

far away in the graph should be heavily penalized during local tracking.

## Local candidate policy

Search:

```text
previous place
+
neighbors
+
small graph radius
```

first.

Use global candidates only when:

```text
local confidence is poor
```

## Important

Do not hard-reject every non-neighbor candidate permanently.

A user may move several places between observations.

Use graph distance as a soft penalty.

---

# 21. Stage 18 — Replace Majority Vote With Probabilistic State Estimation

## Current behavior

The current `LiveLocalizer` uses a sliding majority vote.

Keep it as a baseline.

## New model

Use an HMM-like or Bayesian state estimator.

Let:

```text
S_t = place at time t
O_t = observation at time t
```

Use:

```text
P(S_t | O_1...O_t)
∝
P(O_t | S_t)
×
P(S_t | S_{t-1})
```

Where:

```text
P(O_t | S_t)
```

comes from visual/semantic evidence.

And:

```text
P(S_t | S_{t-1})
```

comes from graph connectivity and transition statistics.

## Important

This does not need a neural sequence model.

A simple probability update is sufficient for the first robust version.

---

# 22. Stage 19 — Build the Localization State Machine

## States

```text
TRACKING
UNCERTAIN
LOST
REACQUIRING
ARRIVED
```

## TRACKING

Strong posterior and consistent graph trajectory.

## UNCERTAIN

Evidence is weak but current state remains plausible.

## LOST

Repeated observations disagree with all plausible states.

## REACQUIRING

Run broader/global retrieval.

## ARRIVED

Destination is confidently observed and remains stable.

## Acceptance criteria

The system does not produce long-lived false localization after a few bad frames.

---

# 23. Stage 20 — Stabilize State Transitions

## Goal

Prevent location flicker.

Do not allow:

```text
A → B → A → B
```

from weak evidence.

A new state should require either:

```text
K consecutive supporting observations
```

or:

```text
posterior confidence above a transition threshold
```

## Transition example

```text
A A A A B B B
        ↑
   confirmed B
```

The exact K should be configured and evaluated.

---

# 24. Stage 21 — Add Global Reacquisition

## Problem

Graph constraints can become harmful if the tracker is already wrong.

## Solution

Use two localization modes.

### LOCAL TRACKING

Strong previous state:

```text
current region
→ nearby graph candidates
```

### GLOBAL REACQUISITION

Low confidence for repeated observations:

```text
entire map retrieval
```

Then return to local tracking once confidence recovers.

## Acceptance criteria

The system can recover from deliberate localization errors or visually ambiguous sequences.

---

# 25. Stage 22 — Calibrate Confidence

## Current problem

A value such as:

```text
similarity > 0.40
```

is not a probability.

## New confidence inputs

Use:

```text
visual score
score margin
temporal agreement
graph agreement
posterior entropy
```

## Confidence states

```text
HIGH
MEDIUM
LOW
UNKNOWN
```

Do not call them percentages unless experimentally calibrated.

## Acceptance criteria

Confidence correlates meaningfully with correctness.

---

# 26. Stage 23 — Integrate Semantic Evidence Correctly

## Goal

Use the object/VLM layer to strengthen localization where appearance alone is ambiguous.

Example:

Two corridors may look nearly identical.

But one contains:

```text
stairs on left
blue room sign
fire extinguisher
```

and the other does not.

The semantic evidence can shift candidate probability.

## Important

Semantic matching must be robust to detector/VLM errors.

Use it as a secondary signal, not an absolute condition.

---

# 27. Stage 24 — Optimize the Runtime Pipeline

## Goal

Continuous navigation must remain practical.

## Runtime loop

```text
camera frame
   ↓
cheap quality/novelty gate
   ↓
only selected observations run through expensive AI
   ↓
embedding + metadata
   ↓
candidate retrieval
   ↓
state update
```

## Important optimization

Do not run every expensive model on every raw frame if the environment has not meaningfully changed.

This is especially important on phones or ordinary laptops.

---

# 28. Stage 25 — Keep Routing Simple

## Goal

Do not over-engineer routing.

Once localization is strong:

```text
current place
+
destination
→ shortest path
```

is enough.

The graph may store edge reliability, but the routing layer should remain easy to understand.

## Optional

Prefer stronger edges through a simple cost such as:

```text
cost = inverse(reliability)
```

but do not make routing the research centerpiece.

---

# 29. Stage 26 — Keep LLM/TTS at the Edge

## Goal

Use an LLM only for natural language.

Input:

```text
[
  "Lobby",
  "Corridor A",
  "Junction",
  "Stairs"
]
```

Output:

```text
"Continue through the lobby, follow Corridor A,
then take the stairs at the junction."
```

The LLM must never determine:

- where the user is,
- whether an edge exists,
- whether the graph is correct.

TTS simply speaks the instruction.

---

# 30. Stage 27 — Build the Research Evaluation Suite

## Localization metrics

### Top-1 accuracy

Correct current place.

### Top-3 accuracy

Correct place contained in candidate set.

### Segment accuracy

Percentage of a full movement sequence correctly localized.

### Transition accuracy

Correctly identify movement A → B.

### False jump rate

How often does the tracker jump to an implausible place?

### Reacquisition time

Number of observations needed to recover after losing localization.

### Unknown precision

When the system says UNKNOWN, how often is it genuinely uncertain?

---

# 31. Graph Evaluation

Measure:

### Node purity

Are observations within a place spatially coherent?

### Node duplicate rate

How often is one physical place represented by multiple nodes?

### Node merge error

How often are two distinct places merged?

### Edge precision

Percentage of generated edges that are physically real.

### Edge recall

Percentage of real accessible transitions discovered.

### Graph fragmentation

Number of unexpected disconnected components.

---

# 32. Ablation Study

The final research report should contain ablations.

## Localization progression

```text
A: nearest neighbour
B: nearest neighbour + threshold
C: + majority vote
D: + semantic evidence
E: + graph constraint
F: + probabilistic temporal state
G: + lost/reacquisition
```

## Graph progression

```text
A: K-Means + transitions
B: temporal segmentation
C: + place reconciliation
D: + transition validation
E: + semantic evidence
F: + graph validation
```

This will show exactly which complexity actually produces improvement.

---

# 33. Failure Analysis Workflow

Every failed sequence should be inspectable.

For a localization decision record:

```text
timestamp
query frame
top-K candidates
visual scores
semantic scores
previous posterior
graph neighbors
graph penalties
final posterior
state
confidence
```

Example:

```text
Frame: 1842

Candidate A:
visual = 0.84
semantic = 0.71
graph = 0.96

Candidate B:
visual = 0.87
semantic = 0.42
graph = 0.18

Final:
A = 0.64
B = 0.22
unknown = 0.14
```

This is essential for research debugging.

---

# 34. Test Strategy

Testing should exist at three levels.

## Unit tests

Test:

- frame novelty logic,
- segment boundaries,
- place merging,
- transition debouncing,
- graph construction,
- graph validation,
- candidate aggregation,
- Bayesian update,
- state transitions.

## Synthetic integration tests

Use tiny artificial graphs and synthetic vectors.

Example:

```text
Lobby
  |
Corridor
  |
Room 101
```

Test that:

```text
Lobby → Corridor → Room101
```

is recovered.

## Real-world benchmark tests

Use actual recorded walkthroughs.

---

# 35. Configuration Structure

Replace scattered thresholds with explicit configuration.

Suggested structure:

```yaml
sampling:
  min_interval_seconds: 0.5
  novelty_threshold: ...
  blur_threshold: ...

perception:
  detector_enabled: true
  vlm_enabled: true
  detector_confidence: ...

embedding:
  model: ...
  image_size: 224

mapping:
  segment_distance_threshold: ...
  merge_threshold: ...
  transition_persistence: ...
  minimum_edge_support: ...

localization:
  top_k: 10
  temporal_weight: ...
  graph_weight: ...
  semantic_weight: ...
  transition_confirmation_count: ...
  uncertain_threshold: ...
  lost_threshold: ...

runtime:
  observation_interval: ...
  global_reacquisition_after: ...

routing:
  weighted: true
```

Every experimental parameter must be externally configurable.

---

# 36. Stage 28 — Migration Strategy From Current Code

Do not rewrite the whole project.

## Existing file → planned role

### `extract_frames.py`

Keep interface, replace internals with smart sampling.

### `embedder.py`

Refactor into encoder abstraction.

### `embed_frames.py`

Build Observation records instead of only `.npy`.

### `build_places.py`

Replace pure K-Means place generation with:

```text
segmentation
→ candidate places
→ reconciliation
→ exemplar selection
```

K-Means remains as baseline.

### `build_graph.py`

Replace simple consecutive-cluster edge creation with validated transition evidence.

### `localize.py`

Split into:

```text
retrieval
candidate scoring
state estimation
```

### `live_tracker.py`

Turn into the runtime state machine.

### `navigate.py`

Keep mostly unchanged.

### `app.py`

Keep UI simple and call the new localization API.

### `evaluate.py`

Expand into the benchmark runner.

---

# 37. Implementation Order — Exact Sequence

Follow this order.

```text
01. Freeze baseline
02. Observation model
03. Smart sampling
04. Encoder abstraction
05. Observation store
06. Detector
07. VLM metadata
08. Temporal segmentation
09. Place formation
10. Place reconciliation
11. Transition extraction
12. Graph builder
13. Graph validator
14. Versioned map
15. Candidate retrieval
16. Candidate scoring
17. Graph constraints
18. Probabilistic state estimator
19. State machine
20. Transition stabilization
21. Global reacquisition
22. Confidence calibration
23. Semantic localization
24. Runtime optimization
25. Evaluation suite
26. Ablations
27. Failure analysis
28. UI integration
29. Final cleanup
```

Do not skip ahead to fancy models before the graph and state estimator work.

---

# 38. Definition of Done for the Graph

The graph subsystem is considered finished when:

```text
✓ A single walkthrough automatically produces places.
✓ Revisited places can be recognized as the same place.
✓ Similar-looking distinct places can remain separate.
✓ Transient recognition noise does not create fake edges.
✓ Main physical transitions are discovered.
✓ Graph confidence is measurable.
✓ Suspicious nodes/edges are automatically flagged.
✓ The map is versioned and reloadable.
✓ Graph generation works without manual node placement.
```

Manual labeling may still be used for human-readable names.

It should not be required to construct topology.

---

# 39. Definition of Done for Localization

The localization subsystem is considered finished when:

```text
✓ Camera observations are continuously processed.
✓ Retrieval produces multiple candidates.
✓ Visual similarity alone is not the final decision.
✓ Graph connectivity influences candidate probability.
✓ Temporal continuity stabilizes movement.
✓ The system supports UNKNOWN/UNCERTAIN.
✓ The system supports LOST and global reacquisition.
✓ False location jumps are measured.
✓ Localization can recover after a failure.
✓ Confidence is meaningfully calibrated.
✓ The system works on unseen walkthrough data.
```

---

# 40. Research Milestones

## Milestone A — Baseline

```text
existing SimpleNav
```

Output:

```text
known performance
known failure cases
```

## Milestone B — Robust Mapping

```text
smart observations
+
temporal place formation
+
validated graph
```

Output:

```text
autonomous indoor graph
```

## Milestone C — Robust Localization

```text
top-K retrieval
+
temporal state estimation
+
graph constraints
```

Output:

```text
stable current place
```

## Milestone D — Full Navigation

```text
localization
→ destination
→ graph route
→ natural-language instruction
→ audio
```

## Milestone E — Research Validation

```text
ablation
+
robustness testing
+
failure analysis
```

---

# 41. Recommended Final Architecture

```text
                         SIMPLE-NAV
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
       OFFLINE MAPPING                RUNTIME LOCALIZATION
              │                             │
              ▼                             ▼
      walkthrough video                 camera stream
              │                             │
              ▼                             ▼
       smart sampling                  smart sampling
              │                             │
              ▼                             ▼
      observation creation              observation
              │                             │
       ┌──────┴──────┐                      │
       ▼             ▼                      ▼
 detector           VLM                 embedding
       │             │                      │
       └──────┬──────┘                      ▼
              ▼                         FAISS Top-K
      semantic metadata                     │
              │                             ▼
              ▼                      candidate scoring
       visual embedding                     │
              │                             ▼
              ▼                     temporal state model
      temporal segmentation                 │
              │                             ▼
              ▼                     graph constraints
       place formation                      │
              │                             ▼
              ▼                       stable location
       place reconciliation                 │
              │                             ▼
              ▼                         destination
       transition discovery                 │
              │                             ▼
              ▼                         shortest path
       graph construction                   │
              │                             ▼
              ▼                        simple instruction
       graph validation                     │
              │                             ▼
              ▼                            TTS
       versioned map
```

---

# 42. Final Research Thesis

The final system should not be presented as:

> "An image classifier that tells the user which room they are in."

The stronger formulation is:

> **A camera-only indoor navigation system that autonomously constructs a semantic connectivity graph from a single-floor walkthrough and performs continuous visual localization by combining appearance, semantic landmarks, temporal continuity, and graph topology.**

The key technical idea is the separation of responsibilities:

```text
Perception
    ↓
"What do I see?"

Retrieval
    ↓
"What mapped places look like this?"

Temporal model
    ↓
"What location is consistent with recent observations?"

Graph
    ↓
"What locations are physically plausible from here?"

State estimator
    ↓
"What should I believe right now?"
```

That is the backbone the project should optimize.

The rest of the system should remain deliberately boring.
