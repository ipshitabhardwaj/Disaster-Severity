"""Inference utilities for the disaster severity classifier.

Single source of truth for how to load a trained model and how to prepare an
image for it. `app/app.py` (Streamlit UI) and `utils/gradcam.py` (visual
explanations) both import from here so preprocessing stays in lock-step with
training. The training pipeline (`notebooks/02_model_training.ipynb`) rescales
pixels to [0, 1]; this module matches that exactly.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

# Class ordering must match the order used at training time
# (`class_names=CLASS_NAMES` in `image_dataset_from_directory`). Reordering
# here would silently mislabel predictions.
CLASS_NAMES: Tuple[str, ...] = ("Earthquake", "Fire", "Flood", "Normal")

# Below this top-class probability we surface a warning to the caller.
LOW_CONFIDENCE_THRESHOLD: float = 0.60


def load_model(model_path: str) -> tf.keras.Model:
    """Load a Keras model saved as `.h5` (or any format Keras accepts).

    Args:
        model_path: Filesystem path to the saved model file.

    Returns:
        A Keras model ready for `predict()`.
    """
    return tf.keras.models.load_model(model_path)


def preprocess_image(
    img_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Load an image from disk and normalize it for inference.

    Steps:
        1. Open with PIL and convert to RGB (drops alpha / handles grayscale).
        2. Resize to the model's expected input size.
        3. Rescale pixel values from [0, 255] to [0, 1] — matches training.
        4. Add a leading batch dimension.

    Args:
        img_path: Filesystem path to the input image.
        target_size: (height, width) the model expects. Default (224, 224).

    Returns:
        A float32 array of shape (1, H, W, 3), pixel values in [0, 1].
    """
    with Image.open(img_path) as raw:
        img = raw.convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(
    model: tf.keras.Model,
    img_array: np.ndarray,
) -> Tuple[str, Dict[str, Union[float, str]]]:
    """Run inference and return (class_label, confidence_dict).

    The confidence dictionary holds the four class probabilities. When the
    top probability is below `LOW_CONFIDENCE_THRESHOLD`, an extra `'warning'`
    key with a human-readable message is added so downstream callers can
    branch on its presence without recomputing the max.

    Args:
        model: Loaded Keras classifier (output shape (1, 4), softmax).
        img_array: Preprocessed input from `preprocess_image()` —
            shape (1, H, W, 3), pixel values in [0, 1]. A bare (H, W, 3)
            array is also accepted and promoted to a batch of one.

    Returns:
        Tuple of:
            - class_label: one of "Earthquake", "Fire", "Flood", "Normal".
            - confidence_dict: {class_name: probability, ...} plus an
              optional 'warning' key for low-confidence predictions.
    """
    # Accept a single un-batched image for convenience.
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    class_label = CLASS_NAMES[top_idx]

    confidence_dict: Dict[str, Union[float, str]] = {
        name: float(p) for name, p in zip(CLASS_NAMES, probs)
    }
    if top_prob < LOW_CONFIDENCE_THRESHOLD:
        confidence_dict["warning"] = (
            f"Low confidence ({top_prob:.1%}) — top class '{class_label}' is "
            f"below the {LOW_CONFIDENCE_THRESHOLD:.0%} threshold; "
            "consider manual review."
        )

    return class_label, confidence_dict
