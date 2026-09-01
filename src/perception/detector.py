"""Object detection layer (§6) — structured visual evidence WITHOUT letting
detection control the map (§Rule 3: the detector provides evidence, it never
defines coordinates, topology, edges, or the final localization).

Design:
- Detector interface is model-independent; swapping YOLO for anything else
  touches no graph/localization code.
- bboxes are normalized [x1, y1, x2, y2] in [0, 1].
- Detector failure NEVER crashes the pipeline: detect() returns [].
- COCO reality check: YOLOv8n detects COCO classes (person, chair, sofa,
  fire hydrant, ...) but NOT doors/stairs/signs — those navigation-relevant
  classes come from the VLM layer (src/perception/scene_tagger.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils import setup_logger

logger = setup_logger("detector")


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # normalized [x1, y1, x2, y2] in [0, 1]

    def to_dict(self) -> dict:
        return {"class": self.class_name, "confidence": round(float(self.confidence), 4),
                "bbox": [round(float(v), 4) for v in self.bbox]}


class Detector(ABC):
    name: str = "abstract"

    @abstractmethod
    def detect(self, image: Any) -> list[DetectedObject]:
        """Detect objects. May return [] — must never raise on bad input."""

    def classes(self) -> list[str]:
        return []


class YoloDetector(Detector):
    """Ultralytics YOLOv8n on COCO classes (default model for Stage 06)."""

    name = "yolov8n"

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35):
        self.model_path = model_path
        self.confidence = confidence
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed — run: conda run -n ML pip install ultralytics"
            ) from e
        self._model = YOLO(model_path)
        logger.info(f"YOLO model loaded: {model_path}")

    def detect(self, image: Any) -> list[DetectedObject]:
        try:
            results = self._model.predict(
                source=image, conf=self.confidence, verbose=False
            )
        except Exception as e:
            logger.warning(f"YOLO detection failed ({e}) — returning no objects")
            return []
        objects: list[DetectedObject] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                h, w = result.orig_shape
                objects.append(
                    DetectedObject(
                        class_name=names.get(cls_id, f"cls_{cls_id}"),
                        confidence=conf,
                        bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
                    )
                )
        return objects

    def classes(self) -> list[str]:
        return list(self._model.names.values())


class StubDetector(Detector):
    """Deterministic detector for tests and offline runs — no model involved.
    Returns a configurable fixed set of objects (or an empty list)."""

    name = "stub"

    def __init__(self, objects: list[DetectedObject] | None = None):
        self._objects = objects or []

    def detect(self, image: Any) -> list[DetectedObject]:
        try:
            np.asarray(image)  # touch the input so bad inputs still don't crash
        except Exception:
            return []
        return list(self._objects)

    def classes(self) -> list[str]:
        return sorted({o.class_name for o in self._objects})


def get_detector(config: dict) -> Detector:
    """Build a detector from the perception config section. Never raises:
    if the detector is disabled or fails to load, returns a StubDetector."""
    perception = config.get("perception", {})
    if not perception.get("detector_enabled", True):
        return StubDetector()
    model_path = perception.get("detector_model", "yolov8n.pt")
    confidence = float(perception.get("detector_confidence", 0.35))
    try:
        return YoloDetector(model_path=model_path, confidence=confidence)
    except Exception as e:
        logger.warning(f"Could not load {model_path}: {e} — using StubDetector")
        return StubDetector()
