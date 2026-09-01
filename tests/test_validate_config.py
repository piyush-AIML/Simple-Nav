"""Tests for the Stage 30 config validation (planner v2 §13): paths resolve,
model ids are registered, weights sum sanely, and LFM2 int4 is forbidden."""

from pathlib import Path

import pytest

from src.utils import validate_config


def good_config(tmp_path) -> dict:
    return {
        "paths": {
            "video": "data/College_env.mp4",
            "frames_dir": "data/frames",
            "map_dir": "data/map",
            "observations_dir": "data/observations",
            "evaluation_dir": "data/evaluation",
        },
        "perception": {
            "detector_model": "yolo26n.pt",
            "vlm_model": "LiquidAI/LFM2.5-VL-450M",
            "vlm_quantization": "bf16",
            "vlm_enabled": True,
        },
        "localization": {
            "w_visual": 0.5, "w_semantic": 0.25,
            "w_temporal": 0.15, "w_graph": 0.1,
        },
    }


def test_good_config_has_no_errors(monkeypatch):
    from src import utils

    monkeypatch.setattr(utils, "project_root", lambda: Path("/mnt/c/PS/projects/simplenav"))
    issues = validate_config(good_config(None))
    assert not [i for i in issues if i[0] == "error"]


def test_unknown_detector_model_is_error(monkeypatch):
    from src import utils

    monkeypatch.setattr(utils, "project_root", lambda: Path("/mnt/c/PS/projects/simplenav"))
    cfg = good_config(None)
    cfg["perception"]["detector_model"] = "yolov5s.pt"
    issues = validate_config(cfg)
    assert any("detector_model" in m and lvl == "error" for lvl, m in issues)


def test_lfm2_int4_is_forbidden(monkeypatch):
    from src import utils

    monkeypatch.setattr(utils, "project_root", lambda: Path("/mnt/c/PS/projects/simplenav"))
    cfg = good_config(None)
    cfg["perception"]["vlm_quantization"] = "4bit"
    issues = validate_config(cfg)
    assert any("4bit" in m and lvl == "error" for lvl, m in issues)


def test_bad_weights_are_errors(monkeypatch):
    from src import utils

    monkeypatch.setattr(utils, "project_root", lambda: Path("/mnt/c/PS/projects/simplenav"))
    cfg = good_config(None)
    cfg["localization"]["w_visual"] = 1.5
    cfg["localization"]["w_semantic"] = 0.0
    cfg["localization"]["w_temporal"] = 0.0
    cfg["localization"]["w_graph"] = 0.0
    issues = validate_config(cfg)
    assert any("w_visual" in m and lvl == "error" for lvl, m in issues)
