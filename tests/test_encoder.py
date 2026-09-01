"""Tests for the VisualEncoder abstraction (§8): registry, dimensions,
input flexibility, and L2 normalization. Uses tiny synthetic images — the
model itself is loaded lazily on first encode.
"""

import numpy as np
import pytest
from PIL import Image

from src.embeddings.encoder import ResNet18Encoder, get_encoder


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
