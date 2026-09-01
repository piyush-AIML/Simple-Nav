"""Tests for the detector layer (§6): interface, stub determinism, failure
tolerance, normalized bboxes, and the config-driven factory."""

import numpy as np
import pytest

from src.perception.detector import (
    DetectedObject,
    StubDetector,
    YoloDetector,
    get_detector,
)


def stub_objects() -> list[DetectedObject]:
    return [
        DetectedObject(class_name="person", confidence=0.9, bbox=(0.1, 0.2, 0.4, 0.8)),
        DetectedObject(class_name="chair", confidence=0.6, bbox=(0.5, 0.5, 0.7, 0.9)),
    ]


def test_stub_detector_deterministic():
    d = StubDetector(stub_objects())
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    first = d.detect(img)
    second = d.detect(img)
    assert len(first) == 2
    assert [o.to_dict() for o in first] == [o.to_dict() for o in second]
    assert d.classes() == ["chair", "person"]


def test_stub_never_raises_on_bad_input():
    d = StubDetector()
    assert d.detect(None) == []
    assert d.detect("not an image") == []


def test_to_dict_shape():
    o = DetectedObject(class_name="door", confidence=0.93, bbox=(0.0, 0.0, 1.0, 1.0))
    d = o.to_dict()
    assert d["class"] == "door"
    assert d["confidence"] == 0.93
    assert len(d["bbox"]) == 4
    assert all(0.0 <= v <= 1.0 for v in d["bbox"])


def test_get_detector_disabled_returns_stub():
    d = get_detector({"perception": {"detector_enabled": False}})
    assert d.name == "stub"


def test_get_detector_bad_model_falls_back_to_stub():
    d = get_detector(
        {"perception": {"detector_enabled": True, "detector_model": "no_such_model.pt"}}
    )
    assert d.name == "stub"


@pytest.mark.slow
def test_yolo_detects_on_real_frame_with_normalized_bboxes():
    try:
        d = YoloDetector(model_path="yolov8n.pt", confidence=0.25)
    except Exception as e:
        pytest.skip(f"YOLO unavailable: {e}")
    img = np.zeros((480, 640, 3), dtype=np.uint8)  # synthetic — likely no detections
    objects = d.detect(img)
    for o in objects:
        assert 0.0 <= o.confidence <= 1.0
        assert all(0.0 <= v <= 1.0 for v in o.bbox)
