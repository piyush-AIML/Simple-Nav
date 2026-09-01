"""Shared image embedding utility — COMPATIBILITY WRAPPER.

Every stage that needs to turn an image into a vector (map building,
localization, evaluation, the demo app) goes through this one module so
there's exactly one definition of "how we embed an image."

Since Stage 04, the actual encoding lives behind the VisualEncoder interface
(src/embeddings/encoder.py); this module keeps the legacy names
(load_model, TRANSFORM, IMAGE_SIZE, embed_image) so existing callers work
unchanged. New code should use `get_encoder(config)` instead.

Uses a pretrained ResNet18 as a frozen feature extractor: we drop the final
classification layer and keep the 512-d pooled features. No fine-tuning —
this is an off-the-shelf ImageNet backbone, used purely for its general
visual features.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.utils import setup_logger

logger = setup_logger("embedder")

_MODEL: nn.Module | None = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FALLBACK_SEED = 42  # keeps the untrained fallback model identical across separate runs

IMAGE_SIZE = 224

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model() -> nn.Module:
    """Load (once) a pretrained ResNet18 with the classification head removed."""
    global _MODEL
    if _MODEL is None:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        except Exception as e:
            # This should not happen with a normal internet connection —
            # torchvision downloads these weights from a standard public
            # URL. If it does happen (blocked network, no internet), we
            # deliberately do NOT fail silently: an untrained random model
            # would otherwise produce embeddings that look fine but are
            # meaningless, and worse, a *different* random model every
            # time this process restarts would make embeddings computed by
            # different pipeline stages incompatible with each other. We
            # fix the random seed so the fallback is at least internally
            # consistent across runs, and we make the degraded mode loud.
            logger.warning(
                f"Could not download pretrained ImageNet weights ({e}). "
                "Falling back to an UNTRAINED ResNet18 with a fixed random "
                "seed so results stay self-consistent across script runs, "
                "but feature quality will be much worse than with real "
                "pretrained weights. Check your internet connection and "
                "re-run once it succeeds — this fallback is a safety net, "
                "not a substitute for the real thing."
            )
            torch.manual_seed(_FALLBACK_SEED)
            backbone = models.resnet18(weights=None)
        # Drop the final fc layer -> output is the 512-d pooled feature vector.
        backbone.fc = nn.Identity()
        backbone.eval()
        backbone.to(_DEVICE)
        _MODEL = backbone
    return _MODEL


def embed_image(image: Union[str, "Image.Image", np.ndarray]) -> np.ndarray:
    """Embed a single image (file path, PIL Image, or HxWx3 numpy array).

    Returns an L2-normalized 512-d float32 vector, so cosine similarity is
    just a dot product. Delegates to the VisualEncoder abstraction (Stage 04).
    """
    from src.embeddings.encoder import ResNet18Encoder

    return ResNet18Encoder().encode(image)
