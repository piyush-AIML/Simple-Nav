"""Tests for the VLM scene tagger (§7): stub determinism, strict schema
validation, config-driven fallback, and (slow, skippable) real-model smoke."""

import json

import numpy as np
import pytest

from src.perception.detector import DetectedObject
from src.perception.scene_tagger import (
    PROMPT,
    SCENE_TYPES,
    SceneTags,
    StubTagger,
    _strip_prompt_echo,
    get_scene_tagger,
    parse_and_validate,
)


def valid_json() -> str:
    return json.dumps(
        {
            "scene_type": "corridor_junction",
            "landmarks": ["blue room-number sign", "staircase on left"],
            "navigation_relevance": ["junction", "stairs"],
            "description": "T-shaped corridor intersection with stairs on the left.",
        }
    )


def test_stub_tagger_deterministic():
    t = StubTagger()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    a = t.tag(img, [])
    b = t.tag(img, [])
    assert a.to_dict() == b.to_dict()
    assert t.name() == "stub"


def test_stub_derives_scene_from_objects():
    t = StubTagger()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    tags = t.tag(
        img,
        [DetectedObject(class_name="couch", confidence=0.9, bbox=(0, 0, 1, 1))],
    )
    assert tags.scene_type == "room"
    tags2 = t.tag(img, [DetectedObject(class_name="fire hydrant", confidence=0.9, bbox=(0, 0, 1, 1))])
    assert "fire_equipment" in tags2.navigation_relevance


def test_parse_valid_json():
    tags = parse_and_validate(valid_json())
    assert tags.scene_type == "corridor_junction"
    assert tags.landmarks == ["blue room-number sign", "staircase on left"]
    assert tags.navigation_relevance == ["junction", "stairs"]


def test_parse_rejects_malformed():
    assert parse_and_validate("") == SceneTags()
    assert parse_and_validate("not json at all") == SceneTags()
    assert parse_and_validate("{broken json") == SceneTags()


def test_parse_rejects_bad_scene_type():
    out = json.dumps({"scene_type": "moon crater", "landmarks": [], "navigation_relevance": []})
    assert parse_and_validate(out).scene_type == "unknown"


def test_parse_filters_unknown_nav_tags():
    out = json.dumps({"scene_type": "room", "landmarks": ["desk"],
                      "navigation_relevance": ["junction", "teleport"]})
    tags = parse_and_validate(out)
    assert tags.navigation_relevance == ["junction"]


def test_strip_prompt_echo():
    echoed = "system\nYou are a helpful assistant.\nuser\n...prompt...\nassistant\n{\"scene_type\": \"room\"}"
    stripped = _strip_prompt_echo(echoed)
    assert stripped == '{"scene_type": "room"}'
    # no marker -> output unchanged
    assert _strip_prompt_echo("plain text") == "plain text"


def test_parse_handles_markdown_fences():
    out = f"```json\n{valid_json()}\n```"
    tags = parse_and_validate(out)
    assert tags.scene_type == "corridor_junction"


def test_parse_never_emits_coordinates_or_topology():
    """The schema has no fields for coordinates/topology — anything extra in
    the model output is dropped by validation (§7 must-NOT list)."""
    sneaky = json.dumps(
        {
            "scene_type": "room",
            "landmarks": [],
            "navigation_relevance": [],
            "description": "x: 0.5, y: 0.3",
            "coords": [1.0, 2.0],
        }
    )
    tags = parse_and_validate(sneaky)
    assert not hasattr(tags, "coords")
    assert tags.scene_type == "room"  # description is human-only, not used for matching


def test_prompt_contains_schema_and_closed_vocab():
    assert "JSON ONLY" in PROMPT
    assert "corridor_junction" in PROMPT
    assert all(t in PROMPT for t in SCENE_TYPES)


def test_get_scene_tagger_disabled_returns_stub():
    t = get_scene_tagger({"perception": {"vlm_enabled": False}})
    assert t.name() == "stub"


def test_get_scene_tagger_bad_model_falls_back_to_stub():
    t = get_scene_tagger(
        {"perception": {"vlm_enabled": True, "vlm_model": "nonexistent/model-zzz"}}
    )
    assert t.name() == "stub"


@pytest.mark.slow
def test_qwen_tagger_real_smoke():
    """One real tag on one frame. Skips (rather than fails) when the model
    can't be loaded — no network/no GPU environments shouldn't break CI."""
    try:
        from src.perception.scene_tagger import QwenVLTagger

        t = QwenVLTagger(max_new_tokens=64)
        t._load()
    except Exception as e:
        pytest.skip(f"Qwen2.5-VL unavailable: {e}")
    tags = t.tag("data/frames/frame_00010.jpg")
    assert tags.scene_type in SCENE_TYPES


@pytest.mark.slow
def test_lfm2_tagger_real_smoke():
    """One real tag on one frame with the default backend (planner v2 §5.2).
    Skips (rather than fails) when the model can't be loaded."""
    try:
        from src.perception.scene_tagger import LFM2VLTagger

        t = LFM2VLTagger(max_new_tokens=64)
        t._load()
    except Exception as e:
        pytest.skip(f"LFM2.5-VL unavailable: {e}")
    tags = t.tag("data/frames/frame_00010.jpg")
    assert tags.scene_type in SCENE_TYPES


def test_get_scene_tagger_dispatch_lfm2(monkeypatch):
    """LFM2 model id -> LFM2VLTagger (planner v2 §5.2 dispatch branch),
    verified without touching the network."""
    from src.perception import scene_tagger as st

    monkeypatch.setattr(st.LFM2VLTagger, "_load", lambda self: None)
    t = get_scene_tagger({"perception": {"vlm_enabled": True,
                                         "vlm_model": "LiquidAI/LFM2.5-VL-450M"}})
    assert t.name().startswith("lfm2.5vl-450m")


def test_get_scene_tagger_dispatch_smolvlm(monkeypatch):
    from src.perception import scene_tagger as st

    monkeypatch.setattr(st.SmolVLMTagger, "_load", lambda self: None)
    t = get_scene_tagger({"perception": {"vlm_enabled": True,
                                         "vlm_model": "HuggingFaceTB/SmolVLM2-500M-Instruct"}})
    assert t.name().startswith("smolvlm2")
