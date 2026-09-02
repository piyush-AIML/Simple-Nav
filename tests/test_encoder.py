"""Tests for the VisualEncoder abstraction (§8): registry, dimensions,
input flexibility, and L2 normalization. Uses tiny synthetic images — the
model itself is loaded lazily on first encode.
"""

import numpy as np
import pytest
from PIL import Image

from src.embeddings.encoder import (
    Dinov2RegistersEncoder,
    ResNet18Encoder,
    get_encoder,
)


def tiny_rgb_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


@pytest.mark.slow
def test_get_encoder_registry():
    encoder = get_encoder({"embedding": {"model": "resnet18"}})
    assert isinstance(encoder, ResNet18Encoder)
    assert encoder.name == "resnet18"
    assert encoder.dimension == 512
    assert encoder.version

    with pytest.raises(ValueError):
        get_encoder({"embedding": {"model": "does_not_exist"}})

    # default when no config key present
    assert isinstance(get_encoder({}), ResNet18Encoder)


@pytest.mark.slow
def test_encode_output_properties():
    encoder = ResNet18Encoder()
    vec = encoder.encode(tiny_rgb_image())
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


@pytest.mark.slow
def test_encode_accepts_all_input_types(tmp_path):
    encoder = ResNet18Encoder()
    arr = tiny_rgb_image()
    img = Image.fromarray(arr)
    path = tmp_path / "img.png"
    img.save(path)

    v_arr = encoder.encode(arr)
    v_img = encoder.encode(img)
    v_path = encoder.encode(str(path))
    for v in (v_arr, v_img, v_path):
        np.testing.assert_allclose(v, v_arr, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_batch_encode_matches_sequential():
    encoder = ResNet18Encoder()
    images = [tiny_rgb_image(), tiny_rgb_image() + 5]
    batch = encoder.batch_encode(images)
    assert batch.shape == (2, 512)
    # tolerance accommodates GPU reduction-order nondeterminism (~1e-5 abs);
    # cosine similarity is affected at the 1e-9 level — irrelevant in practice
    for row, img in zip(batch, images):
        single = encoder.encode(img)
        np.testing.assert_allclose(row, single, rtol=1e-3, atol=1e-4)


# ---------- planner v3 §4: DINOv2-with-Registers (new default) ----------


def test_dinov2_registers_registered_not_default_change_get_encoder_fallback():
    """Registry exposure + dimension, without loading the model (slow)."""
    from src.embeddings.encoder import _REGISTRY

    assert "dinov2_registers_small" in _REGISTRY
    assert "dinov3_vits16plus" in _REGISTRY  # documented opt-in, never default
    assert get_encoder({}).name == "resnet18"  # dict-level fallback unchanged


@pytest.mark.slow
def test_dinov2_get_encoder_registry():
    encoder = get_encoder({"embedding": {"model": "dinov2_registers_small"}})
    assert isinstance(encoder, Dinov2RegistersEncoder)
    assert encoder.name == "dinov2_registers_small"
    assert encoder.dimension == 384
    assert encoder.version == "facebook/dinov2-with-registers-small"


@pytest.mark.slow
def test_dinov2_encode_output_properties():
    encoder = Dinov2RegistersEncoder()
    vec = encoder.encode(tiny_rgb_image())
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


@pytest.mark.slow
def test_dinov2_accepts_all_input_types(tmp_path):
    encoder = Dinov2RegistersEncoder()
    arr = tiny_rgb_image()
    img = Image.fromarray(arr)
    path = tmp_path / "img.png"
    img.save(path)

    v_arr = encoder.encode(arr)
    v_img = encoder.encode(img)
    v_path = encoder.encode(str(path))
    for v in (v_arr, v_img, v_path):
        np.testing.assert_allclose(v, v_arr, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
def test_dinov2_batch_encode_matches_sequential():
    encoder = Dinov2RegistersEncoder()
    images = [tiny_rgb_image(), tiny_rgb_image() + 5]
    batch = encoder.batch_encode(images)
    assert batch.shape == (2, 384)
    for row, img in zip(batch, images):
        single = encoder.encode(img)
        np.testing.assert_allclose(row, single, rtol=1e-3, atol=1e-4)
