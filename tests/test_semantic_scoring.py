"""Tests for Stage 23 semantic evidence scoring (planner v2 §6):

semantic_similarity = 0.4*scene + 0.4*landmark Jaccard + 0.2*object Jaccard,
with Rule-4 asymmetries: query-side missing evidence is neutral, place-side
absence is real evidence, disjoint non-empty sets are never exactly 0, and
stub-tagger inputs degrade to object-overlap only.
"""

from src.localization.semantic_scoring import semantic_similarity
from src.perception.detector import DetectedObject
from src.perception.scene_tagger import SceneTags


def chair() -> DetectedObject:
    return DetectedObject(class_name="chair", confidence=0.9, bbox=(0, 0, 1, 1))


def test_identical_tags_score_one():
    q = SceneTags(scene_type="corridor", landmarks=["blue sign", "stairs on left"])
    p = [SceneTags(scene_type="corridor", landmarks=["blue sign", "stairs on left"])]
    assert semantic_similarity(q, [chair()], p, ["chair"]) == 1.0


def test_disjoint_landmarks_low_but_nonzero():
    """Completely disjoint non-empty landmark sets -> low but non-zero score
    (never exactly 0, per Rule 4)."""
    q = SceneTags(scene_type="corridor", landmarks=["blue sign"])
    p = [SceneTags(scene_type="room", landmarks=["desk"])]
    score = semantic_similarity(q, [], p)
    assert 0.0 < score < 0.3


def test_unknown_scene_capped_never_dominates():
    """One side 'unknown' -> capped contribution, never dominates the score."""
    q = SceneTags(scene_type="unknown", landmarks=["blue sign"])
    p = [SceneTags(scene_type="room", landmarks=["desk"])]
    score = semantic_similarity(q, [], p)
    assert score < 0.5


def test_stub_tagger_falls_back_to_object_overlap_only():
    """Stub-tagger inputs (empty landmarks always, unknown scene) -> object
    overlap only, no crash."""
    stub_q = SceneTags()
    stub_p = [SceneTags()]
    assert semantic_similarity(stub_q, [chair()], stub_p, ["chair"]) == 1.0
    # disjoint object sets: exactly the floored object term, nothing else
    assert semantic_similarity(stub_q, [chair()], stub_p, ["tv"]) == 0.1
    # no evidence at all -> neutral
    assert semantic_similarity(stub_q, [], stub_p) == 0.5


def test_no_query_evidence_neutral():
    assert semantic_similarity(None, None, [SceneTags(scene_type="corridor")]) == 0.5
    assert semantic_similarity(None, [], None, ["chair"]) == 0.5


def test_similar_corridors_separated_by_landmarks():
    """The planner's synthetic case: two visually-near-identical corridors,
    one with ['stairs on left', 'blue room sign'], the other with [] —
    semantic score must separate them by >= 0.4."""
    q = SceneTags(scene_type="corridor", landmarks=["stairs on left", "blue room sign"])
    corridor_a = [SceneTags(scene_type="corridor", landmarks=["stairs on left", "blue room sign"])]
    corridor_b = [SceneTags(scene_type="corridor", landmarks=[])]
    score_a = semantic_similarity(q, [], corridor_a)
    score_b = semantic_similarity(q, [], corridor_b)
    assert score_a - score_b >= 0.4


def test_dict_inputs_tolerated():
    q = {"scene_type": "room", "landmarks": ["desk"]}
    objs = [{"class": "chair", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]
    p = [{"scene_type": "room", "landmarks": ["desk"]}]
    assert semantic_similarity(q, objs, p, ["chair"]) == 1.0


def test_never_raises_on_garbage():
    assert semantic_similarity("garbage", 123, object(), None) == 0.5
    assert semantic_similarity(None, None, None, None) == 0.5
