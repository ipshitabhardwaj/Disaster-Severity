"""Grad-CAM / Grad-CAM++ explainability utilities for the disaster classifier.

Works with any EfficientNet variant (B0/B1/B3) produced by
`notebooks/02_model_training.ipynb` — the last Conv2D layer is auto-detected
rather than hardcoded, since each backbone has a different architecture.

`app/app.py` (Streamlit UI) and `notebooks/03_gradcam.ipynb` /
`notebooks/04_dashboard_dev.ipynb` both import from here, so heatmap
generation stays in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def find_project_root(marker: str = "data", max_up: int = 5) -> Path:
    """Walk up from this file until a directory containing `marker/` is found."""
    cur = Path(__file__).resolve().parent
    for _ in range(max_up + 1):
        if (cur / marker).is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        f"Could not locate project root containing '{marker}/' "
        f"starting from {Path(__file__).resolve()}"
    )


PROJECT_ROOT = find_project_root("data")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.predict import CLASS_NAMES, load_model, predict, preprocess_image  # noqa: E402


# ── Layer detection ──────────────────────────────────────────────────────


def _find_last_conv_path(layer) -> Optional[list]:
    """Depth-first, reverse-order search for the last Conv2D layer.

    Returns the path (list of layer names, outermost first) from `layer`
    down to the Conv2D layer, or None if none is found under `layer`.
    Handles both flat models (EfficientNet built with `input_tensor=`, whose
    internal layers land directly in `model.layers`) and nested models
    (a backbone stored as a single sub-model layer).
    """
    sublayers = getattr(layer, "layers", None)
    if sublayers:
        for sub in reversed(sublayers):
            found = _find_last_conv_path(sub)
            if found is not None:
                return [sub.name] + found if found != [sub.name] else found
        return None
    if isinstance(layer, tf.keras.layers.Conv2D):
        return [layer.name]
    return None


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Auto-detect the name of the last Conv2D layer in `model`.

    Searches `model.layers` from the end backward (recursing into any
    nested sub-models) so it works unmodified across EfficientNetB0/B1/B3
    or any other convolutional backbone.

    Args:
        model: A compiled/loaded Keras model.

    Returns:
        The layer name to pass to `model.get_layer(...)`.

    Raises:
        ValueError: If no Conv2D layer is found anywhere in the model.
    """
    path = _find_last_conv_path(model)
    if not path:
        raise ValueError(
            "No Conv2D layer found in model — cannot auto-detect a "
            "Grad-CAM target layer. Pass `layer_name` explicitly."
        )
    return path[-1]


def _build_grad_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
    try:
        conv_layer = model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Layer '{layer_name}' not found directly on `model`. If your "
            "backbone is nested as a sub-model, pass the sub-model's own "
            "layer name or flatten the architecture (see "
            "notebooks/02_model_training.ipynb's `build_model`, which uses "
            "`input_tensor=` to keep the graph flat)."
        ) from exc
    return tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])


def _ensure_batched(img_array: np.ndarray) -> np.ndarray:
    if img_array.ndim == 3:
        return np.expand_dims(img_array, axis=0)
    return img_array


# ── Grad-CAM ──────────────────────────────────────────────────────────────


def get_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Standard Grad-CAM heatmap via `tf.GradientTape`.

    Args:
        model: Loaded Keras classifier.
        img_array: Preprocessed image, shape (1, H, W, 3) or (H, W, 3),
            EfficientNet-preprocessed (see `utils/predict.preprocess_image`).
        class_idx: Index into `CLASS_NAMES` to explain.
        layer_name: Conv layer to target. Auto-detected via
            `get_last_conv_layer_name` when None.

    Returns:
        Heatmap as a float32 numpy array, shape (H, W), normalized to [0, 1].
    """
    if layer_name is None:
        layer_name = get_last_conv_layer_name(model)
    grad_model = _build_grad_model(model, layer_name)
    img_array = _ensure_batched(img_array)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy().astype(np.float32)


def get_gradcam_plus_plus_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Grad-CAM++ heatmap using second- and third-order gradients.

    Grad-CAM++ weights each spatial location's gradient contribution with
    coefficients (alphas) derived from higher-order derivatives, which
    localizes better than vanilla Grad-CAM when multiple instances of the
    same class appear in one image.

    Args:
        model: Loaded Keras classifier.
        img_array: Preprocessed image, shape (1, H, W, 3) or (H, W, 3),
            EfficientNet-preprocessed.
        class_idx: Index into `CLASS_NAMES` to explain.
        layer_name: Conv layer to target. Auto-detected via
            `get_last_conv_layer_name` when None.

    Returns:
        Heatmap as a float32 numpy array, shape (H, W), normalized to [0, 1].
    """
    if layer_name is None:
        layer_name = get_last_conv_layer_name(model)
    grad_model = _build_grad_model(model, layer_name)
    img_array = _ensure_batched(img_array)

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_output, predictions = grad_model(img_array)
                output = predictions[:, class_idx]
            conv_first_grad = tape3.gradient(output, conv_output)
        conv_second_grad = tape2.gradient(conv_first_grad, conv_output)
    conv_third_grad = tape1.gradient(conv_second_grad, conv_output)

    conv_output = conv_output[0].numpy()
    conv_first_grad = conv_first_grad[0].numpy()
    conv_second_grad = conv_second_grad[0].numpy()
    conv_third_grad = conv_third_grad[0].numpy()

    global_sum = np.sum(conv_output, axis=(0, 1))

    alpha_num = conv_second_grad
    alpha_denom = 2.0 * conv_second_grad + conv_third_grad * global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom

    alpha_norm = np.sum(alphas, axis=(0, 1))
    alpha_norm = np.where(alpha_norm != 0.0, alpha_norm, np.ones_like(alpha_norm))
    alphas = alphas / alpha_norm

    weights = np.maximum(conv_first_grad, 0.0)
    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))

    heatmap = np.sum(deep_linearization_weights * conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap.astype(np.float32)


# ── Overlay ───────────────────────────────────────────────────────────────


def _to_uint8_rgb(original_img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(original_img, Image.Image):
        return np.array(original_img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(original_img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def overlay_heatmap(
    heatmap: np.ndarray,
    original_img: Union[Image.Image, np.ndarray],
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend a Grad-CAM heatmap onto the original image.

    Args:
        heatmap: 2D array in [0, 1] (output of `get_gradcam_heatmap` or
            `get_gradcam_plus_plus_heatmap`).
        original_img: PIL Image or numpy array (RGB), any size. Float
            arrays are assumed to be in [0, 1]; ints are assumed [0, 255].
        alpha: Heatmap opacity in the blend (0 = original only, 1 = heatmap only).
        colormap: OpenCV colormap applied to the heatmap.

    Returns:
        Tuple of (overlay_bgr, overlay_rgb) — same H×W as `original_img`,
        uint8. `overlay_bgr` is ready for `cv2.imwrite`/`cv2.imshow`;
        `overlay_rgb` is ready for `matplotlib`/`streamlit`.
    """
    original_rgb = _to_uint8_rgb(original_img)
    h, w = original_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)

    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(heatmap_color_bgr, alpha, original_bgr, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_bgr, overlay_rgb


# ── Quality metric ────────────────────────────────────────────────────────


def compute_average_drop(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int,
    heatmap: np.ndarray,
) -> float:
    """Average Drop % — how much the model's confidence falls when the
    heatmap's top-20%-activation region is masked out of the input.

    A larger drop means the masked region was actually important to the
    model's decision, i.e. the heatmap is correctly localizing evidence.

    Args:
        model: Loaded Keras classifier.
        img_array: Preprocessed image, shape (1, H, W, 3) or (H, W, 3),
            EfficientNet-preprocessed.
        class_idx: Class index the heatmap was computed for.
        heatmap: 2D array in [0, 1], any size (will be resized to match
            `img_array`).

    Returns:
        Percentage drop in confidence: max(0, (Y_c - O_c) / Y_c) * 100.
    """
    img_array = _ensure_batched(img_array)
    h, w = img_array.shape[1:3]

    original_probs = model.predict(img_array, verbose=0)[0]
    y_c = float(original_probs[class_idx])

    heatmap_resized = cv2.resize(heatmap, (w, h))
    threshold = np.percentile(heatmap_resized, 80)  # top 20% of activation
    mask = (heatmap_resized >= threshold).astype(np.float32)[..., None]

    masked_img = img_array.copy()
    masked_img[0] = masked_img[0] * (1 - mask)

    masked_probs = model.predict(masked_img, verbose=0)[0]
    o_c = float(masked_probs[class_idx])

    return max(0.0, (y_c - o_c) / (y_c + 1e-8)) * 100.0


# ── Convenience wrapper (used by app/app.py) ────────────────────────────


def predict_and_explain(model: tf.keras.Model, image_path: str) -> Dict[str, object]:
    """Run prediction + both Grad-CAM variants for a single image on disk.

    Convenience wrapper combining `utils.predict` and this module's Grad-CAM
    functions — this is what `app/app.py` calls per uploaded image.

    Returns a dict with keys: class, confidence, all_probs, warning
    (optional), original (uint8 RGB array), heatmap, heatmap_plus_plus,
    overlay (Grad-CAM, RGB), overlay_plus_plus (Grad-CAM++, RGB),
    average_drop, average_drop_plus_plus.
    """
    target_size = model.input_shape[1:3]
    img_array = preprocess_image(image_path, target_size=target_size)
    class_label, confidence_dict = predict(model, img_array)
    class_idx = CLASS_NAMES.index(class_label)

    original_uint8 = np.uint8(np.clip(img_array[0], 0, 255))

    heatmap = get_gradcam_heatmap(model, img_array, class_idx)
    heatmap_pp = get_gradcam_plus_plus_heatmap(model, img_array, class_idx)

    _, overlay_rgb = overlay_heatmap(heatmap, original_uint8)
    _, overlay_pp_rgb = overlay_heatmap(heatmap_pp, original_uint8)

    avg_drop = compute_average_drop(model, img_array, class_idx, heatmap)
    avg_drop_pp = compute_average_drop(model, img_array, class_idx, heatmap_pp)

    result: Dict[str, object] = {
        "class": class_label,
        "confidence": confidence_dict[class_label],
        "all_probs": {k: v for k, v in confidence_dict.items() if k != "warning"},
        "original": original_uint8,
        "heatmap": heatmap,
        "heatmap_plus_plus": heatmap_pp,
        "overlay": overlay_rgb,
        "overlay_plus_plus": overlay_pp_rgb,
        "average_drop": avg_drop,
        "average_drop_plus_plus": avg_drop_pp,
    }
    if "warning" in confidence_dict:
        result["warning"] = confidence_dict["warning"]
    return result


# ── Gallery generation ────────────────────────────────────────────────────


def generate_gradcam_gallery(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names,
    output_dir: str,
    n_per_class: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Generate a Grad-CAM / Grad-CAM++ comparison gallery for a test set.

    Picks up to `n_per_class` images per class from `test_ds` (a batched,
    EfficientNet-preprocessed, categorical-label dataset — see
    `notebooks/02_model_training.ipynb`'s `make_dataset`), renders a
    side-by-side (Original | Grad-CAM | Grad-CAM++) figure for each, and
    saves them to `output_dir/gradcam_gallery/`.

    Args:
        model: Loaded Keras classifier.
        test_ds: Batched `tf.data.Dataset` yielding (images, one-hot labels).
        class_names: Ordered list of class names matching label indices.
        output_dir: Directory under which `gradcam_gallery/` is created.
        n_per_class: Number of images to sample per class.

    Returns:
        Dict keyed by class name with per-class average Grad-CAM /
        Grad-CAM++ Average Drop % and image count, e.g.
        {"Earthquake": {"gradcam_avg_drop": 42.1,
                         "gradcam_pp_avg_drop": 51.3, "n_images": 5}, ...}
    """
    import matplotlib.pyplot as plt

    gallery_dir = Path(output_dir) / "gradcam_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    per_class_images = {c: [] for c in class_names}
    for imgs, lbls in test_ds.unbatch():
        cls_name = class_names[int(np.argmax(lbls.numpy()))]
        if len(per_class_images[cls_name]) < n_per_class:
            per_class_images[cls_name].append(imgs.numpy())
        if all(len(v) >= n_per_class for v in per_class_images.values()):
            break

    summary: Dict[str, Dict[str, float]] = {}
    for cls_name, imgs_list in per_class_images.items():
        cls_idx = class_names.index(cls_name)
        drops, drops_pp = [], []

        for i, img in enumerate(imgs_list):
            img_batch = np.expand_dims(img, axis=0)
            heatmap = get_gradcam_heatmap(model, img_batch, cls_idx)
            heatmap_pp = get_gradcam_plus_plus_heatmap(model, img_batch, cls_idx)

            original_uint8 = np.uint8(np.clip(img, 0, 255))
            _, overlay_rgb = overlay_heatmap(heatmap, original_uint8)
            _, overlay_pp_rgb = overlay_heatmap(heatmap_pp, original_uint8)

            drop = compute_average_drop(model, img_batch, cls_idx, heatmap)
            drop_pp = compute_average_drop(model, img_batch, cls_idx, heatmap_pp)
            drops.append(drop)
            drops_pp.append(drop_pp)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(original_uint8)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(overlay_rgb)
            axes[1].set_title(f"Grad-CAM ({drop:.1f}% drop)")
            axes[1].axis("off")
            axes[2].imshow(overlay_pp_rgb)
            axes[2].set_title(f"Grad-CAM++ ({drop_pp:.1f}% drop)")
            axes[2].axis("off")
            plt.suptitle(f"{cls_name} — sample {i + 1}")
            plt.tight_layout()
            fig.savefig(gallery_dir / f"{cls_name}_{i + 1}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        summary[cls_name] = {
            "gradcam_avg_drop": float(np.mean(drops)) if drops else 0.0,
            "gradcam_pp_avg_drop": float(np.mean(drops_pp)) if drops_pp else 0.0,
            "n_images": len(imgs_list),
        }

    return summary
