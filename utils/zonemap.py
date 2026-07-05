"""Zone Damage Map — tile a large aerial image into a grid, classify each
tile with the trained EfficientNet, and paint a color-coded damage overlay.

Used by `app/app.py`'s "Zone Analysis" tab. The idea: a single wide-area
aerial photo often contains a mix of intact and damaged zones, so instead of
one label for the whole frame we slide a 224x224 window (stride 224, no
overlap) across it, run inference per tile, and tint each tile by its
predicted class. This reveals *where* in the scene the model sees damage.

Preprocessing matches training exactly (EfficientNet `preprocess_input` on
raw [0, 255] pixels — see `utils/predict.py`), so per-tile predictions are
consistent with the main classifier.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.applications.efficientnet import preprocess_input

from utils.predict import CLASS_NAMES

# Consistent, natural palette used across the whole app (badges, zone map, PDF).
CLASS_COLORS: Dict[str, str] = {
    "Earthquake": "#E24B4A",  # red
    "Fire":       "#EF9F27",  # orange
    "Flood":      "#378ADD",  # blue
    "Normal":     "#1D9E75",  # green
}

TILE = 224  # sliding-window size == EfficientNetB0 input size


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def zone_damage_map(
    model,
    pil_image: Image.Image,
    alpha: float = 0.45,
) -> Tuple[Image.Image, Dict[str, object]]:
    """Tile `pil_image`, classify each tile, and return a damage overlay.

    A sliding window of `TILE`x`TILE` with stride `TILE` (no overlap) walks the
    image. Edge tiles smaller than a full window are still classified (the crop
    is resized up to `TILE` before inference). Each tile region is tinted with
    its predicted class color and outlined; the class label + confidence are
    drawn on tiles when there is room.

    Works on any image size, including ones smaller than a single tile — those
    resolve to a single whole-image classification.

    Args:
        model: Loaded Keras classifier (expects [0, 255] EfficientNet inputs).
        pil_image: Input image (any mode; converted to RGB internally).
        alpha: Overlay opacity for the class tint (0 = none, 1 = opaque).

    Returns:
        Tuple of:
            - overlay: RGB `PIL.Image` — the original with the color-coded grid.
            - stats: dict with keys:
                'n_tiles', 'n_damage' (non-Normal tiles),
                'damage_pct' (float 0-100),
                'class_counts' ({class_name: int}),
                'tiles' (list of per-tile dicts: row, col, box, class, conf).
    """
    img = pil_image.convert("RGB")
    w, h = img.size

    # At least one tile in each direction, even for tiny images.
    xs = list(range(0, max(1, w), TILE)) or [0]
    ys = list(range(0, max(1, h), TILE)) or [0]

    overlay = img.copy()
    tint_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    tint_draw = ImageDraw.Draw(tint_layer)
    line_draw = ImageDraw.Draw(overlay)

    # Collect every tile's crop first, then classify in one batched predict().
    boxes: List[Tuple[int, int, int, int]] = []
    grid: List[Tuple[int, int]] = []
    batch = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            box = (x, y, min(x + TILE, w), min(y + TILE, h))
            crop = img.crop(box).resize((TILE, TILE))
            batch.append(preprocess_input(np.asarray(crop, dtype=np.float32)))
            boxes.append(box)
            grid.append((r, c))

    probs = model.predict(np.stack(batch, axis=0), verbose=0)

    tiles: List[Dict[str, object]] = []
    class_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    a = int(round(alpha * 255))
    for (r, c), box, p in zip(grid, boxes, probs):
        idx = int(np.argmax(p))
        cls = CLASS_NAMES[idx]
        conf = float(p[idx])
        class_counts[cls] += 1

        rgb = _hex_to_rgb(CLASS_COLORS[cls])
        tint_draw.rectangle(box, fill=(*rgb, a))
        line_draw.rectangle(box, outline=rgb, width=3)
        tiles.append({"row": r, "col": c, "box": box, "class": cls, "conf": conf})

    overlay = Image.alpha_composite(overlay.convert("RGBA"), tint_layer).convert("RGB")

    n_tiles = len(tiles)
    n_damage = n_tiles - class_counts["Normal"]
    stats: Dict[str, object] = {
        "n_tiles": n_tiles,
        "n_damage": n_damage,
        "damage_pct": (100.0 * n_damage / n_tiles) if n_tiles else 0.0,
        "class_counts": class_counts,
        "tiles": tiles,
        "grid_shape": (len(ys), len(xs)),
    }
    return overlay, stats
