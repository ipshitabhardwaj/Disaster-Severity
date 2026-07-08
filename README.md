# Project 43 — Explainable Disaster Severity Assessment
[▶ Watch the demo](/demo_video.mp4)

An aerial-image disaster classifier built on EfficientNet transfer learning,
paired with Grad-CAM / Grad-CAM++ visual explanations so predictions aren't
a black box. Given an aerial photo, the model classifies it into one of four
categories — **Earthquake**, **Fire**, **Flood**, or **Normal** — and a
Streamlit dashboard shows exactly which pixels drove that decision, with a
confidence-based warning when a prediction needs human review.

> C-DAC Mohali | ML to Generative AI & LLMs

## Team

| Member | Role |
|---|---|
| Ipshita | ML Engineer — data pipeline, model training (EfficientNetB0/B1/B3) |
| Anuksha | Explainability Lead — Grad-CAM / Grad-CAM++, evaluation metrics |
| Rishika | App Developer — Streamlit dashboard |

## Dataset

**AIDERv2** (Aerial Image Dataset for Emergency Response), via Zenodo:
https://zenodo.org/records/10891054

| Split | Earthquake | Fire | Flood | Normal | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,927 | 3,509 | 4,063 | 3,900 | 13,399 |
| Val | 239 | 439 | 505 | 223 | 1,406 |
| Test | 239 | 436 | 502 | 477 | 1,654 |

Earthquake is the minority class across every split — training uses
inverse-frequency class weights to compensate (see notebook 02, section 4).

## Tech stack

| Component | Tool |
|---|---|
| Backbone | EfficientNetB0 / B1 / B3 (ImageNet-pretrained, transfer learning) |
| Framework | TensorFlow / Keras 2.21 |
| Explainability | Grad-CAM, Grad-CAM++ (custom `tf.GradientTape` implementation) |
| Data pipeline | `tf.data` (`image_dataset_from_directory`, on-the-fly augmentation) |
| Dashboard | Streamlit (dark theme) + Plotly |
| Zone damage map | Sliding-window tiling + per-tile inference (Pillow) |
| PDF export | fpdf2 |
| Image processing | OpenCV, Pillow |
| Metrics | scikit-learn (confusion matrix, classification report) |

## Setup

```bash
pip install tensorflow opencv-python pillow scikit-learn matplotlib seaborn plotly streamlit pandas numpy fpdf2
```

Download AIDERv2 from the [Zenodo record](https://zenodo.org/records/10891054)
and arrange it into the folder structure below (already the case if you
cloned this repo with `data/` populated):

```
Project43-DisasterSeverity/
├── data/
│   ├── Train/{Earthquake,Fire,Flood,Normal}/
│   ├── Val/{Earthquake,Fire,Flood,Normal}/
│   └── Test/{Earthquake,Fire,Flood,Normal}/
├── models/              # trained .h5 checkpoints land here
├── outputs/             # plots, reports, Grad-CAM gallery land here
├── notebooks/
│   ├── 01_pipeline.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_gradcam.ipynb
│   └── 04_dashboard_dev.ipynb
├── utils/
│   ├── predict.py      # model load + preprocessing (single source of truth)
│   ├── gradcam.py      # Grad-CAM / Grad-CAM++ + gallery
│   ├── zonemap.py      # tiled zone damage map
│   └── report.py       # PDF report export (fpdf2)
├── app/
│   └── app.py
└── .streamlit/
    └── config.toml     # dark theme
```

All notebooks and modules auto-detect the project root by walking up from
their own location until they find a folder containing `data/` — no
hardcoded absolute paths, so it doesn't matter where Jupyter is launched
from.

## Running the notebooks (in order)

1. **`01_pipeline.ipynb`** — inspects the dataset, computes class weights,
   verifies the `tf.data` pipeline and augmentation. Sanity checks only, no
   training.
2. **`02_model_training.ipynb`** — trains EfficientNetB0 (and optionally
   B1/B3) in two phases: frozen-base feature extraction, then fine-tuning
   the top layers. Its final cell promotes the best variant to
   `models/best_model.h5`.
3. **`03_gradcam.ipynb`** — loads `models/best_model.h5`, generates
   single-image and full-gallery Grad-CAM / Grad-CAM++ explanations,
   computes the Average Drop % quality metric, and re-runs evaluation
   (confusion matrix, F1) on the test set.
4. **`04_dashboard_dev.ipynb`** — exercises `utils/predict.py` and
   `utils/gradcam.py` exactly as `app/app.py` calls them, including
   confidence-threshold bucketing, before testing the actual dashboard.

Each notebook fails gracefully (prints a clear message, doesn't crash) if
`models/best_model.h5` doesn't exist yet — run notebook 02 first.

## Running the app

From the project root:

```bash
python -m streamlit run app/app.py
```

Opens the dashboard at `http://localhost:8501`. The app ships a dark theme by
default (`.streamlit/config.toml`) and fails gracefully with a clear message
if `models/best_model.h5` is missing.

The dashboard has three tabs:

- **🔎 Analyze** — upload an aerial image (or click a sidebar sample) to get
  the predicted class, an animated confidence meter, a color-coded
  probability chart, and the Original / Grad-CAM / Grad-CAM++ triptych. Export
  the whole analysis as a **PDF report** with one click.
- **🗺️ Zone Analysis** — tiles a large image into 224×224 patches, classifies
  each, and paints a color-coded damage map next to the original with a
  "X% of zones show damage" summary.
- **📊 Model Performance** — metric cards, training curves, confusion matrix,
  the per-class classification report, and the model-comparison table.

## Screenshots

_Placeholder — add real screenshots before the demo._

| View | Screenshot |
|---|---|
| Analyze tab (prediction + Grad-CAM) | `docs/screenshots/analyze.png` _(to add)_ |
| Zone Analysis (damage map) | `docs/screenshots/zone.png` _(to add)_ |
| Model Performance dashboard | `docs/screenshots/performance.png` _(to add)_ |
| PDF report export | `docs/screenshots/report.png` _(to add)_ |

## Expected outputs

| File | Generated by |
|---|---|
| `outputs/class_distribution.png`, `sample_images.png`, `augmentation_preview.png` | Notebook 01 |
| `models/{b0,b1,b3}_phase{1,2}.h5`, `models/best_model_{b0,b1,b3}.h5`, `models/best_model.h5` | Notebook 02 |
| `outputs/{B0,B1,B3}_curves.png`, `{B0,B1,B3}_confusion.png`, `{B0,B1,B3}_report.txt`, `model_comparison.csv` | Notebook 02 |
| `outputs/gradcam_demo_per_class.png`, `gradcam_vs_plusplus_earthquake.png` | Notebook 03 |
| `outputs/gradcam_gallery/` (20+ side-by-side comparison images), `gradcam_gallery_summary.csv` | Notebook 03 |
| `outputs/gradcam_eval_confusion.png`, `gradcam_eval_report.txt` | Notebook 03 |

## Model performance

**EfficientNetB0 — 96.94% best validation accuracy, 97% test accuracy,
0.96 macro F1.** Authoritative numbers live in `outputs/model_comparison.csv`,
`outputs/B0_report.txt`, and `outputs/B0_curves.png`.

B1 and B3 were **skipped on purpose** — training ran on CPU only, and B0
already reaches 97% test accuracy, so the extra wall-clock cost of the larger
backbones wasn't justified. The code supports them (just swap the backbone in
notebook 02); the comparison rows are left as `N/A (skipped)`.

| Metric | EfficientNetB0 | EfficientNetB1 | EfficientNetB3 |
|---|---|---|---|
| Params | 5.3M | 7.8M | 12M |
| Input size | 224×224 | 240×240 | 300×300 |
| Best val accuracy | **0.9694** | N/A (skipped) | N/A (skipped) |
| Test accuracy | **0.97** | N/A (skipped) | N/A (skipped) |
| Macro F1 | **0.96** | N/A (skipped) | N/A (skipped) |

Per-class results on the 1,654-image test set (`outputs/B0_report.txt`):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Earthquake | 0.94 | 0.95 | 0.94 | 239 |
| Fire | 0.98 | 0.99 | 0.99 | 436 |
| Flood | 0.95 | 0.99 | 0.97 | 502 |
| Normal | 0.98 | 0.94 | 0.96 | 477 |
| **Macro avg** | **0.96** | **0.96** | **0.96** | **1,654** |

Training was two-phase: **15 epochs** of frozen-base feature extraction
(Phase 1) followed by **8 epochs** of fine-tuning the top layers (Phase 2).
Grad-CAM / Grad-CAM++ explanation quality is quantified per class in
`outputs/gradcam_gallery_summary.csv` via the **Average Drop %** metric.

## Creative features

1. **Grad-CAM + Grad-CAM++ explainability** — both heatmaps are computed and
   shown side by side with the original, so every prediction comes with a
   visual "why". Grad-CAM++ localizes multiple disjoint regions of evidence
   (e.g. scattered earthquake damage) better than vanilla Grad-CAM.
2. **Zone damage map** — the *Zone Analysis* tab tiles a large aerial image
   into 224×224 patches (sliding window, stride 224), classifies each patch,
   and paints a color-coded damage overlay
   (🔴 Earthquake · 🟠 Fire · 🔵 Flood · 🟢 Normal) beside the original, with
   a "X% of zones show damage" summary and legend. Works on normal-sized
   images too (one zone).
3. **PDF report export** — one click turns any analysis into a clean,
   single-page **PDF** (via `fpdf2`) with the images, prediction, full
   probability table, Average Drop % metric, and model info — ready to hand
   off to a responder.
4. **Confidence thresholding with a visual badge** — a large, class-colored
   severity badge plus an **animated confidence meter** (green/yellow/red),
   with a sidebar threshold slider that flags low-confidence predictions for
   human review instead of presenting them as fact.
5. **Model performance dashboard tab** — metric cards, training curves,
   confusion matrix, per-class classification report, and the model-comparison
   table, all inside the app.
6. **Sample image quick-load buttons** — sidebar buttons load a real test
   image per class directly from `data/Test/`, so the dashboard can be demoed
   without sourcing an external image.
7. **Quantified explanation quality** — the **Average Drop %** metric measures
   whether the heatmap's highlighted region is actually causally important to
   the model's decision, not just visually plausible.
8. **Architecture-agnostic Grad-CAM** — `get_last_conv_layer_name()`
   auto-detects the target conv layer, so the same code works unmodified
   across EfficientNetB0/B1/B3 without hardcoding layer names.
9. **Graceful degradation everywhere** — every module (`predict.py`,
   `gradcam.py`, `zonemap.py`, `report.py`, `app.py`, notebooks 03/04) handles
   a missing `models/best_model.h5` or a bad image with a clear message
   instead of a stack trace.
