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


class _HFViTEncoder(VisualEncoder):
    """Shared body for Hugging Face ViT-family encoders (planner v3 §4).

    Each subclass sets `version` to the HF model id; the image processor is
    loaded from the same id, so preprocessing always matches the model —
    never the ResNet18-specific TRANSFORM in src/embedder.py (planner v3
    §4.3 note). Pooler (CLS) output is used for whole-image retrieval, per
    HuggingFace's DINOv2/DINOv3 model-card recommendation; if
    scripts/compare_encoders.py (Stage 33) ever shows raw
    last_hidden_state[:, 0, :] beating pooler_output on same-place
    consistency, swap it here — both are one-line changes."""

    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            cls._processor = AutoImageProcessor.from_pretrained(cls.version)
            cls._model = AutoModel.from_pretrained(cls.version)
            cls._model.eval()
            if torch.cuda.is_available():
                cls._model = cls._model.to("cuda")
            logger.info(f"{cls.__name__} loaded: {cls.version}")
        return cls._processor, cls._model

    def _forward(self, pil_images: list[Image.Image]) -> np.ndarray:
        import torch

        processor, model = self._load()
        inputs = processor(images=pil_images, return_tensors="pt")
        if next(model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.cpu().numpy().astype("float32")

    def encode(self, image: ImageInput) -> np.ndarray:
        vec = self._forward([_to_pil(image)])[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def batch_encode(self, images: list[ImageInput]) -> np.ndarray:
        vecs = self._forward([_to_pil(img) for img in images])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class Dinov2RegistersEncoder(_HFViTEncoder):
    """DINOv2-with-Registers small (ViT-S/14, 384-d) — the default encoder
    as of planner v3 §4. Strong viewpoint/lighting invariance for the
    same-place-different-angle retrieval task this project actually does
    (unlike ResNet18's ImageNet-classification objective). Register tokens
    remove the attention-map artifacts plain DINOv2 shows as spurious
    high-norm patches. Ungated, Apache-2.0 — the reason it beats the gated
    DINOv3 option as the default (§4.1)."""

    name = "dinov2_registers_small"
    version = "facebook/dinov2-with-registers-small"
    dimension = 384


class Dinov3Encoder(_HFViTEncoder):
    """DINOv3 ViT-S+/16 — documented, opt-in stretch option (planner v3
    §4.1). Requires a Hugging Face account with the gated DINOv3 license
    accepted AND approved (manual approval, "up to a few days", not
    guaranteed — Meta's own FAQ). NOT the default: do not select this in
    config.yaml unless access is already confirmed working."""

    name = "dinov3_vits16plus"
    version = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    dimension = 384


_REGISTRY: dict[str, type[VisualEncoder]] = {
    "resnet18": ResNet18Encoder,
    "dinov2_registers_small": Dinov2RegistersEncoder,
    "dinov3_vits16plus": Dinov3Encoder,  # opt-in only — gated HF license, see class docstring
}


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
