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


def test_place_side_unknown_scene_abstains_not_penalized():
    """Planner v3 §9 re-tune: a PLACE whose stored scene is 'unknown'
    (tagging failed, not scene absent) must abstain at 0.5 — the same Rule 4
    that protects the query side — not vote at 0.3. 46% of this dataset's
    observations are unknown-scene; the old 0.3 made the semantic term
    systematically vote against exactly those places."""
    q = SceneTags(scene_type="corridor", landmarks=["blue sign"])
    p_unknown = [SceneTags(scene_type="unknown", landmarks=["blue sign"])]
    p_mismatch = [SceneTags(scene_type="room", landmarks=["blue sign"])]
    score_unknown = semantic_similarity(q, [], p_unknown)
    score_mismatch = semantic_similarity(q, [], p_mismatch)
    assert score_unknown > score_mismatch  # abstaining beats voting against


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


def test_legacy_corridor_junction_matches_junction():
    """v1 stored scenes ('corridor_junction') normalize onto v2 'junction' —
    pre-upgrade map artifacts must score identically against fresh tags."""
    q = {"scene_type": "junction", "landmarks": ["door"]}
    p_legacy = [{"scene_type": "corridor_junction", "landmarks": ["door"]}]
    p_new = [{"scene_type": "junction", "landmarks": ["door"]}]
    score_legacy = semantic_similarity(q, [], p_legacy)
    score_new = semantic_similarity(q, [], p_new)
    assert score_legacy == score_new == 0.9  # scene+landmark full match, object-neutral


def test_dict_inputs_tolerated():
    q = {"scene_type": "room", "landmarks": ["desk"]}
    objs = [{"class": "chair", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]
    p = [{"scene_type": "room", "landmarks": ["desk"]}]
    assert semantic_similarity(q, objs, p, ["chair"]) == 1.0


def test_never_raises_on_garbage():
    assert semantic_similarity("garbage", 123, object(), None) == 0.5
    assert semantic_similarity(None, None, None, None) == 0.5
