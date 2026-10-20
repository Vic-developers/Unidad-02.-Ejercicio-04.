"""Reusable, deterministic Fashion-MNIST utilities."""
from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass


def load_and_split(seed: int = SEED):
    """Load public Keras data; split official train stratified 90/10."""
    import tensorflow as tf
    (x_all, y_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    idx_train, idx_val = train_test_split(
        np.arange(len(y_all)), test_size=0.10, random_state=seed,
        stratify=y_all,
    )
    splits = {
        "train": (x_all[idx_train], y_all[idx_train]),
        "validation": (x_all[idx_val], y_all[idx_val]),
        "test": (x_test, y_test),
    }
    normalized = {}
    for name, (x, y) in splits.items():
        normalized[name] = (preprocess_images(x), y.astype(np.int64))
    return normalized


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Convert uint8 grayscale images to float32 [0,1] with a channel axis."""
    x = np.asarray(images, dtype=np.float32) / 255.0
    if x.ndim == 3:
        x = x[..., np.newaxis]
    if x.ndim != 4 or x.shape[1:] != (28, 28, 1):
        raise ValueError(f"Expected (N,28,28) or (N,28,28,1), got {x.shape}")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Image values must be finite and within [0,1]")
    return x


def audit_splits(splits: dict, seed: int = SEED) -> dict:
    audit = {
        "seed": seed,
        "class_names": CLASS_NAMES,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "splits": {},
    }
    try:
        import tensorflow as tf
        audit["tensorflow"] = tf.__version__
        audit["visible_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        audit["tensorflow"] = None
        audit["visible_gpus"] = []
    for name, (x, y) in splits.items():
        audit["splits"][name] = {
            "n": int(len(y)), "image_shape": list(x.shape[1:]),
            "labels_shape": list(y.shape), "pixel_min": float(x.min()),
            "pixel_max": float(x.max()),
            "class_counts": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(10)},
            "images_sha256": sha256_array(x), "labels_sha256": sha256_array(y),
        }
    return audit


def build_dense(input_shape=(28, 28, 1), n_classes: int = 10):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="dense_baseline")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape=(28, 28, 1), n_classes: int = 10):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="small_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
