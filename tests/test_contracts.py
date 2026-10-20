import numpy as np
import pytest

from cv_project import CLASS_NAMES, build_cnn, preprocess_images


def test_image_preprocessing_shape_and_range():
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    images[1] = 127
    images[2] = 255
    x = preprocess_images(images)
    assert x.shape == (3, 28, 28, 1)
    assert x.dtype == np.float32
    assert 0 <= x.min() <= x.max() <= 1
    assert np.isclose(x[1].mean(), 127 / 255)


def test_dataset_has_ten_named_classes():
    assert len(CLASS_NAMES) == 10
    assert len(set(CLASS_NAMES)) == 10


def test_model_output_contract():
    tf = pytest.importorskip("tensorflow")
    model = build_cnn()
    probabilities = model(np.zeros((2, 28, 28, 1), dtype=np.float32), training=False).numpy()
    assert probabilities.shape == (2, 10)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
