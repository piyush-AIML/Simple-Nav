"""Visual encoder abstraction (§8): the system talks to a VisualEncoder, not
to ResNet18 directly. Swapping the model is a config change.

Baseline implementation: frozen ResNet18 pooled features (the same model the
project has always used) — see ResNet18Encoder, which wraps the shared logic
in src/embedder.py rather than duplicating it.

Research candidates (CLIP-like, DINO-like, retrieval encoders) can be added
behind this interface and compared with scripts/compare_encoders.py-style
metrics (same-place viewpoint consistency vs different-place separability)
without touching any mapping/localization code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PIL import Image

from src.embedder import TRANSFORM, load_model
from src.utils import setup_logger

logger = setup_logger("encoder")

ImageInput = Union[str, "Image.Image", np.ndarray]  # path | PIL Image | HxWx3 ndarray


def _to_pil(image: ImageInput) -> Image.Image:
    """Normalize any accepted input to an RGB PIL image."""
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


class VisualEncoder(ABC):
    name: str = "abstract"
    version: str = ""
    dimension: int = 0

    @abstractmethod
    def encode(self, image: ImageInput) -> np.ndarray:
        """One image -> L2-normalized float32 vector of `dimension`."""

    def batch_encode(self, images: list[ImageInput]) -> np.ndarray:
        """Many images -> (N, dimension). Default: sequential encode."""
        return np.stack([self.encode(img) for img in images]).astype("float32")

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "dimension": self.dimension}


class ResNet18Encoder(VisualEncoder):
    """Frozen ResNet18 pooled features — the project baseline."""

    name = "resnet18"
    version = "torchvision_imagenet"
    dimension = 512

    def encode(self, image: ImageInput) -> np.ndarray:
        import torch

        model = load_model()
        tensor = TRANSFORM(_to_pil(image)).unsqueeze(0)
        if next(model.parameters()).is_cuda:
            tensor = tensor.cuda()
        with torch.no_grad():
            features = model(tensor)
        vec = features.squeeze(0).cpu().numpy().astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def batch_encode(self, images: list[ImageInput]) -> np.ndarray:
        """Small batched pass — faster than N single calls for >= 8 images."""
        import torch

        model = load_model()
        batch = torch.stack([TRANSFORM(_to_pil(img)) for img in images])
        if next(model.parameters()).is_cuda:
            batch = batch.cuda()
        with torch.no_grad():
            features = model(batch).cpu().numpy().astype("float32")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms


_REGISTRY: dict[str, type[VisualEncoder]] = {"resnet18": ResNet18Encoder}


def register_encoder(name: str, encoder_cls: type[VisualEncoder]) -> None:
    _REGISTRY[name] = encoder_cls


def get_encoder(config: dict) -> VisualEncoder:
    """Build an encoder from config.embedding.model."""
    embedding_cfg = config.get("embedding", {})
    model_name = embedding_cfg.get("model", "resnet18")
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding model {model_name!r}. Registered: {sorted(_REGISTRY)}"
        )
    encoder = _REGISTRY[model_name]()
    logger.info(f"Encoder: {encoder.describe()}")
    return encoder
