"""Tests for place reconciliation (§12): revisits merge, visually similar
distinct rooms stay separate, scene conflicts block merges, idempotence."""

import numpy as np

from src.mapping.observations import Observation
from src.mapping.place_builder import Place, build_places
from src.mapping.place_reconciliation import decide_merge, reconcile_places
from src.mapping.segmentation import Segment

LOBBY = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
CORRIDOR = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
ROOM_X = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")   # visually close to lobby
ROOM_Y = np.array([0.85, 0.15, 0.0, 0.0], dtype="float32")  # ...and to each other

CONFIG = {"merge_visual_threshold": 0.75, "merge_visual_extra_threshold": 0.92,
          "merge_landmark_threshold": 0.5, "merge_scene_required": True}


def make_obs(i: int, emb, scene: str, landmarks=None) -> Observation:
    return Observation(
        id=f"obs_{i:04d}", timestamp=float(i),
        frame_path=f"data/frames/frame_{i:05d}.jpg",
        embedding=emb, scene_tags={"scene_type": scene, "landmarks": landmarks or []},
        landmarks=landmarks or [],
    )


def make_places(segments: list[tuple[list[Observation], str]]) -> tuple[list[Place], list[Observation]]:
    """segments: list of (obs_list, scene_landmarks_label). Builds one Place per
    segment with exemplars = first obs."""
    all_obs: list[Observation] = []
    places: list[Place] = []
    for si, (obs_list, _) in enumerate(segments):
        seg = Segment(f"seg_{si:03d}", [o.id for o in obs_list], 0, len(obs_list) - 1, obs_list[0].id)
        place = build_places(obs_list, [seg], exemplar_method="temporal_diversity", max_exemplars=2)[0]
        place.place_id = f"place_{si:02d}"
        places.append(place)
        all_obs.extend(obs_list)
    return places, all_obs


def test_lobby_corridor_lobby_merges_revisits():
    """LobbyA / Corridor / LobbyB: the two lobby segments merge into one place."""
    lobby_a = [make_obs(0, LOBBY, "room", ["reception desk"])] * 3
    corridor = [make_obs(3, CORRIDOR, "corridor", ["blue sign"])] * 3
    lobby_b = [make_obs(6, LOBBY, "room", ["reception desk"])] * 3
    places, obs = make_places([(lobby_a, "a"), (corridor, "b"), (lobby_b, "c")])

    merged = reconcile_places(places, obs, CONFIG)
    assert len(merged) == 2, [p.place_id for p in merged]
    ids = [p.place_id for p in merged]
    assert "place_00" in ids  # merged place keeps the lower id
    lobby = [p for p in merged if p.place_id == "place_00"][0]
    assert len(lobby.observation_ids) == 6  # both lobby segments joined


def test_visually_similar_distinct_rooms_stay_separate():
    """ROOM_X and ROOM_Y look similar but have conflicting landmarks -> no merge."""
    room_x = [make_obs(0, ROOM_X, "room", ["green door"])] * 3
    room_y = [make_obs(3, ROOM_Y, "room", ["red door"])] * 3
    places, obs = make_places([(room_x, "x"), (room_y, "y")])

    decision = decide_merge(places[0], places[1], places, {o.id: o for o in obs}, CONFIG)
    assert not decision.merged
    assert "landmark_conflict" in decision.reasons

    merged = reconcile_places(places, obs, CONFIG)
    assert len(merged) == 2


def test_scene_conflict_blocks_merge():
    """Visually identical but different scene types -> no merge."""
    room = [make_obs(0, LOBBY, "room", [])] * 3
    elevator = [make_obs(3, LOBBY, "elevator", [])] * 3
    places, obs = make_places([(room, "r"), (elevator, "e")])

    decision = decide_merge(places[0], places[1], places, {o.id: o for o in obs}, CONFIG)
    assert not decision.merged
    assert "scene_conflict" in decision.reasons


def test_reconciliation_is_idempotent():
    lobby_a = [make_obs(0, LOBBY, "room", ["desk"])] * 3
    corridor = [make_obs(3, CORRIDOR, "corridor", [])] * 3
    lobby_b = [make_obs(6, LOBBY, "room", ["desk"])] * 3
    places, obs = make_places([(lobby_a, "a"), (corridor, "b"), (lobby_b, "c")])

    once = reconcile_places(places, obs, CONFIG)
    twice = reconcile_places(once, obs, CONFIG)
    assert [p.place_id for p in once] == [p.place_id for p in twice]
    assert [sorted(p.observation_ids) for p in once] == [sorted(p.observation_ids) for p in twice]


def test_no_merge_without_extra_signal():
    """Strong-but-not-extra-strong visual with NO landmark/context evidence
    -> no merge (extra signal required; §12 multiple signals)."""
    near = np.array([0.78, 0.63, 0.0, 0.0], dtype="float32")  # cos(LOBBY, near) ≈ 0.78
    a = [make_obs(0, LOBBY, "room", [])] * 3
    b = [make_obs(3, near, "room", [])] * 3  # adjacent, no landmarks
    places, obs = make_places([(a, "a"), (b, "b")])
    decision = decide_merge(places[0], places[1], places, {o.id: o for o in obs}, CONFIG)
    assert 0.75 <= decision.visual_similarity < 0.85  # strong but not extra-strong
    assert not decision.merged  # no shared landmarks, no revisit context
