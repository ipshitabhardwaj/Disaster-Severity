"""Project 43 — Explainable Disaster Severity Assessment (Streamlit dashboard).

An aerial-image classifier (EfficientNetB0) with Grad-CAM / Grad-CAM++
explanations, a tile-based Zone Damage Map, a PDF report export, and a model
performance dashboard. Paths are anchored via `find_project_root()` (from
`utils/gradcam.py`) so nothing is hardcoded.

UI: "recon console" theme — dark navy void, teal signal accent, viewfinder
corner brackets on panels, mono telemetry type. Semantic class colors
(Earthquake / Fire / Flood / Normal) are shared with utils/zonemap.py and are
intentionally left untouched so badges, charts, zone tiles, and the PDF all
agree.
"""

import io
import os
import sys
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Bootstrap project root onto sys.path, then use the shared root finder ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from utils.gradcam import find_project_root, predict_and_explain
from utils.predict import CLASS_NAMES as PREDICT_CLASS_NAMES
from utils.zonemap import CLASS_COLORS, zone_damage_map
from utils.report import build_report_pdf

PROJECT_ROOT = str(find_project_root("data"))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.h5")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "Test")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

CLASS_NAMES = list(PREDICT_CLASS_NAMES)
CLASS_EMOJI = {"Earthquake": "🏚️", "Fire": "🔥", "Flood": "🌊", "Normal": "✅"}

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Severity Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — "recon console" design system ──────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  /* Semantic class colors — shared with utils/zonemap.py, do not drift */
  --eq: #E24B4A; --fire: #EF9F27; --flood: #378ADD; --normal: #1D9E75;

  /* Console surface tokens */
  --void:    #0a0e16;
  --void-2:  #0d1220;
  --panel:   #11182a;
  --panel-2: #161f36;
  --line:    #232e46;
  --line-hi: #324062;
  --text-hi: #eef2f9;
  --text:    #a7b2c6;
  --text-dim:#5b6780;

  /* Single signature accent — teal signal, not generic blue */
  --signal:   #48d1c4;
  --signal-2: #2fb8ac;
  --signal-dim: rgba(72,209,196,0.14);
}

html, body, [class*="css"], .stMarkdown, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Contour-line texture — faint topographic rings from the top-right corner,
   a nod to the aerial/geospatial subject rather than a generic dot-grid. */
.stApp {
  background:
    repeating-radial-gradient(circle at 100% 0%,
      rgba(72,209,196,0.05) 0px, rgba(72,209,196,0.05) 1px,
      transparent 1px, transparent 46px),
    radial-gradient(1000px 500px at 100% -10%, #0f1626 0%, var(--void) 60%);
}
.block-container { padding-top: 2.8rem; max-width: 1340px; }

.mono { font-family: 'IBM Plex Mono', monospace; }
.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.14em; text-transform: uppercase; font-weight: 600;
  color: var(--signal); margin: 0 0 6px 0;
}

/* ── Viewfinder corner brackets — the one signature device, reused
   everywhere a panel needs framing (console bar, cards, telemetry). ── */
.bracket { position: relative; }
.bracket::before, .bracket::after {
  content: ""; position: absolute; width: 12px; height: 12px;
  opacity: 0.65; pointer-events: none;
}
.bracket::before { top: -1px; left: -1px; border-top: 2px solid var(--signal); border-left: 2px solid var(--signal); }
.bracket::after  { bottom: -1px; right: -1px; border-bottom: 2px solid var(--signal); border-right: 2px solid var(--signal); }

/* ── Console bar (replaces the hero card) ─────────────────────────── */
.console-bar {
  position: relative; overflow: hidden;
  display: flex; justify-content: space-between; align-items: center;
  gap: 20px; flex-wrap: wrap;
  background: linear-gradient(90deg, var(--void-2) 0%, var(--panel) 55%, var(--void-2) 100%);
  border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 22px; margin-bottom: 20px; margin-top: 6px;
}
.console-bar::after {
  content: ""; position: absolute; top: 0; left: -30%; width: 22%; height: 100%;
  background: linear-gradient(100deg, transparent, rgba(72,209,196,0.10), transparent);
  transform: skewX(-20deg); animation: scan 7s ease-in-out infinite;
}
@keyframes scan { 0% { left: -30%; } 55%, 100% { left: 130%; } }

.console-left { display: flex; align-items: center; gap: 14px; position: relative; z-index: 1; }
.dot-pulse {
  width: 9px; height: 9px; border-radius: 50%; background: var(--signal);
  box-shadow: 0 0 0 0 rgba(72,209,196,0.6); flex-shrink: 0;
  animation: pulse 2.2s ease-out infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(72,209,196,0.55); }
  70%  { box-shadow: 0 0 0 8px rgba(72,209,196,0); }
  100% { box-shadow: 0 0 0 0 rgba(72,209,196,0); }
}
.console-title h1 {
  color: var(--text-hi); margin: 0; font-size: 1.55rem; font-weight: 800;
  letter-spacing: -0.01em; line-height: 1.15;
}
.console-title .eyebrow { margin-bottom: 4px; }
.console-right { display: flex; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }
.tag {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.03em; color: var(--text); background: rgba(255,255,255,0.03);
  border: 1px solid var(--line-hi); border-radius: 6px; padding: 6px 10px;
}
.tag.signal { color: #072522; background: var(--signal); border-color: var(--signal); font-weight: 700; }

/* ── Card containers ───────────────────────────────────────────────── */
.card {
  background: linear-gradient(180deg, rgba(22,31,54,0.65) 0%, rgba(17,24,42,0.65) 100%);
  border: 1px solid var(--line); border-radius: 12px;
  padding: 16px 20px; margin-bottom: 14px;
}
.card h3, .card h4 { margin-top: 0; }

/* ── Severity / prediction badge ───────────────────────────────────── */
.sev-badge {
  position: relative; display: flex; align-items: center; gap: 16px;
  border-radius: 12px; padding: 18px 22px; margin: 4px 0 16px 0;
  color: #fff; font-weight: 800;
  border: 1px solid rgba(255,255,255,0.14);
}
.sev-badge .emoji { font-size: 2.2rem; }
.sev-badge .label {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; line-height: 1.15;
  letter-spacing: 0.01em; text-transform: uppercase;
}
.sev-badge .sub { font-size: 0.84rem; font-weight: 500; opacity: 0.9; margin-top: 2px; }

/* ── Signal-strength confidence meter (with tick marks) ────────────── */
.meter-wrap {
  position: relative; background: var(--void); border: 1px solid var(--line);
  border-radius: 8px; height: 26px; width: 100%; overflow: hidden; margin: 6px 0 2px 0;
}
.meter-fill {
  height: 100%; text-align: right; color: #fff; font-family: 'IBM Plex Mono', monospace;
  font-weight: 600; font-size: 0.78rem; line-height: 26px; padding-right: 10px;
  white-space: nowrap;
}
.meter-ticks { position: absolute; inset: 0; display: flex; pointer-events: none; }
.meter-ticks span { flex: 1; border-right: 1px solid rgba(255,255,255,0.10); }
.meter-ticks span:last-child { border-right: none; }

/* ── Telemetry (metric) cards ──────────────────────────────────────── */
.telemetry {
  background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--signal);
  border-radius: 8px; padding: 14px 16px; height: 100%;
}
.telemetry .val {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: var(--text-hi);
  line-height: 1.2;
}
.telemetry .lbl {
  font-size: 0.72rem; color: var(--text-dim); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;
}

/* ── Sidebar polish ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--void-2); border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] div.stButton > button {
  border-radius: 8px; border: 1px solid var(--line-hi);
  background: var(--panel-2); font-weight: 600; font-family: 'Inter', sans-serif;
  transition: all 0.18s ease; padding: 10px 6px;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
  border-color: var(--signal); background: var(--signal-dim); color: var(--text-hi);
}

/* Primary action buttons in main area */
div[data-testid="stMainBlockContainer"] div.stButton > button,
.stDownloadButton > button {
  border-radius: 8px; transition: all 0.18s ease;
  font-family: 'Inter', sans-serif; font-weight: 600;
}
div[data-testid="stMainBlockContainer"] div.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--signal); box-shadow: 0 0 0 1px var(--signal);
}

/* Tabs restyled as a mono pipeline-stage selector */
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Mono', monospace; font-weight: 600;
  letter-spacing: 0.03em; font-size: 0.85rem; text-transform: uppercase;
}
.stTabs [aria-selected="true"] { color: var(--signal) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--signal) !important; }

/* Misc */
.legend-row { display:flex; align-items:center; gap:10px; margin: 5px 0; }
.legend-swatch { width:16px; height:16px; border-radius:3px; display:inline-block; }
.explain { color: var(--text-dim); font-size:0.85rem; line-height: 1.55; }

.kv { display:flex; justify-content:space-between; align-items:center;
  padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.84rem; }
.kv:last-child { border-bottom: none; }
.kv b { color: var(--text-hi); font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
.kv span { color: var(--text-dim); }

.stSpinner > div { border-top-color: var(--signal) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def confidence_color(prob: float) -> str:
    """Green >80%, yellow 60-80%, red <60%."""
    if prob > 0.80:
        return "#1D9E75"
    if prob >= 0.60:
        return "#EAB308"
    return "#E24B4A"


def severity_badge_html(pred_class: str, confidence: float) -> str:
    color = CLASS_COLORS.get(pred_class, "#5B8DEF")
    emoji = CLASS_EMOJI.get(pred_class, "❓")
    return f"""
    <div class="sev-badge bracket" style="background: linear-gradient(120deg, {color} 0%, {color}cc 100%);">
        <div class="emoji">{emoji}</div>
        <div>
            <div class="label">{pred_class}</div>
            <div class="sub">Predicted disaster class &middot; {confidence:.1%} confidence</div>
        </div>
    </div>"""


def confidence_meter_html(confidence: float) -> str:
    color = confidence_color(confidence)
    pct = max(4.0, confidence * 100.0)  # keep a sliver visible even at ~0
    return f"""
    <div class="meter-wrap">
        <div class="meter-fill" style="width:{pct:.1f}%; background:{color};">
            {confidence:.1%}
        </div>
        <div class="meter-ticks"><span></span><span></span><span></span><span></span></div>
    </div>"""


@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def get_model_variant(model):
    h, w = model.input_shape[1:3]
    size_map = {224: "B0", 240: "B1", 300: "B3"}
    return size_map.get(h, f"Custom ({h}x{w})"), (h, w)


@st.cache_data(show_spinner=False)
def get_sample_bytes(class_name):
    """Bytes of the first available test image for `class_name`, or None."""
    import glob
    files = (glob.glob(os.path.join(TEST_DIR, class_name, "*.jpg")) +
             glob.glob(os.path.join(TEST_DIR, class_name, "*.jpeg")) +
             glob.glob(os.path.join(TEST_DIR, class_name, "*.png")))
    if not files:
        return None
    with open(files[0], "rb") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def analyze_bytes(image_bytes: bytes):
    """Run predict + Grad-CAM for raw image bytes (cached on the bytes)."""
    model = load_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        return predict_and_explain(model, tmp_path)
    finally:
        os.unlink(tmp_path)


# ── Console bar (header) ─────────────────────────────────────────────────
st.markdown(
    """
    <div class="console-bar bracket">
        <div class="console-left">
            <span class="dot-pulse"></span>
            <div class="console-title">
                <div class="eyebrow">Aerial Damage Recon &middot; C-DAC Mohali</div>
                <h1>Disaster Severity Console</h1>
            </div>
        </div>
        <div class="console-right">
            <span class="tag">AIDERv2</span>
            <span class="tag">EfficientNetB0</span>
            <span class="tag">Grad-CAM + Grad-CAM++</span>
            <span class="tag signal">97% test accuracy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Fail gracefully if the model is missing ─────────────────────────────
if not os.path.isfile(MODEL_PATH):
    st.error(
        f"Trained model not found at `{MODEL_PATH}`.\n\n"
        "Run **notebooks/02_model_training.ipynb** to completion — its final "
        "cell promotes the best variant to `models/best_model.h5`, which this "
        "app loads."
    )
    st.stop()

model = load_model()
variant, input_size = get_model_variant(model)
num_params = model.count_params()

# ── Session state ───────────────────────────────────────────────────────
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
    st.session_state.image_name = None

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="eyebrow">Control Panel</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="card bracket" style="border-left:3px solid var(--signal);">
            <div class="eyebrow" style="margin-bottom:10px;">Model Info</div>
            <div class="kv"><span>Variant</span><b>EfficientNet{variant}</b></div>
            <div class="kv"><span>Params</span><b>{num_params:,}</b></div>
            <div class="kv"><span>Input</span><b>{input_size[0]}&times;{input_size[1]}</b></div>
            <div class="kv"><span>Val accuracy</span>
                 <b style="color:var(--normal);">96.94%</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Try a sample image**")
    sample_cols = st.columns(2)
    for i, cls in enumerate(CLASS_NAMES):
        col = sample_cols[i % 2]
        if col.button(f"{CLASS_EMOJI[cls]} {cls}", key=f"sample_{cls}", width="stretch"):
            b = get_sample_bytes(cls)
            if b:
                st.session_state.image_bytes = b
                st.session_state.image_name = f"sample_{cls}.jpg"
            else:
                st.warning(f"No sample image found for {cls}.")

    st.markdown("---")
    st.markdown("**Review threshold**")
    confidence_threshold = st.slider(
        "Warn below this confidence", min_value=0.40, max_value=0.90,
        value=0.60, step=0.05, format="%.0f%%",
    )
    thr_color = confidence_color(confidence_threshold)
    st.markdown(
        f'<div class="explain">Predictions under '
        f'<b style="color:{thr_color};">{confidence_threshold:.0%}</b> are flagged '
        "for human review.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "This console classifies aerial imagery into **Earthquake, Fire, "
            "Flood, or Normal** using an EfficientNetB0 transfer-learning model "
            "trained on the **AIDERv2** dataset.\n\n"
            "Predictions are made explainable with **Grad-CAM** and "
            "**Grad-CAM++** heatmaps, quantified by an **Average Drop %** "
            "faithfulness metric. The **Zone Map** tab tiles large images "
            "to localize damage, and every analysis can be exported to a "
            "**PDF report**.\n\n"
            "_C-DAC Mohali — ML to Generative AI & LLMs._"
        )

# ── Uploader (shared across tabs) ───────────────────────────────────────
st.markdown('<div class="eyebrow">Input Feed</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload an aerial image (PNG/JPG) — or pick a sample from the sidebar",
    type=["png", "jpg", "jpeg"],
)
if uploaded_file is not None:
    st.session_state.image_bytes = uploaded_file.getvalue()
    st.session_state.image_name = uploaded_file.name

image_bytes = st.session_state.image_bytes

tab_analyze, tab_zone, tab_perf = st.tabs(
    ["01 · Analyze", "02 · Zone Map", "03 · Performance"]
)

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ════════════════════════════════════════════════════════════════════════
with tab_analyze:
    if image_bytes is None:
        st.info("👆 Upload an aerial disaster image, or pick a sample from the sidebar, to get started.")
    else:
        try:
            with st.spinner("Analyzing image…"):
                result = analyze_bytes(image_bytes)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Could not analyze this image: {exc}")
            result = None

        if result is not None:
            pred_class = result["class"]
            confidence = result["confidence"]

            left, right = st.columns([1, 1.15])

            with left:
                st.markdown('<div class="eyebrow">Input</div>', unsafe_allow_html=True)
                st.image(result["original"], width="stretch",
                         caption=st.session_state.image_name or "uploaded image")

            with right:
                st.markdown('<div class="eyebrow">Classification</div>', unsafe_allow_html=True)
                st.markdown(severity_badge_html(pred_class, confidence), unsafe_allow_html=True)
                st.markdown("**Confidence**")
                st.markdown(confidence_meter_html(confidence), unsafe_allow_html=True)

                if confidence < confidence_threshold:
                    st.warning("⚠️ Low confidence — human review recommended.")
                else:
                    st.success("✅ Confident prediction.")

                probs = result["all_probs"]
                bar_colors = [CLASS_COLORS.get(k, "#5B8DEF") for k in probs.keys()]
                fig = go.Figure(go.Bar(
                    x=list(probs.values()), y=list(probs.keys()), orientation="h",
                    marker=dict(
                        color=bar_colors, cornerradius=6,
                        line=dict(color="rgba(255,255,255,0.08)", width=1),
                    ),
                    text=[f"{v:.1%}" for v in probs.values()], textposition="outside",
                    textfont=dict(size=13, color="#e2e8f0", family="IBM Plex Mono, monospace"),
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                ))
                fig.update_layout(
                    template="plotly_dark", xaxis=dict(range=[0, 1.08], tickformat=".0%"),
                    height=240, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#a7b2c6"),
                    bargap=0.35,
                )
                fig.update_yaxes(showgrid=False)
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig, width="stretch")

            st.markdown("---")
            st.markdown('<div class="eyebrow">Explainability</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="explain">Grad-CAM highlights the pixels that most '
                "drove the prediction (red = high influence). Grad-CAM++ refines "
                "this when evidence is spread across several regions of the frame.</div>",
                unsafe_allow_html=True,
            )
            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown("**Original**")
                st.image(result["original"], width="stretch")
            with g2:
                st.markdown(f"**Grad-CAM** · {result['average_drop']:.1f}% avg drop")
                st.image(result["overlay"], width="stretch")
            with g3:
                st.markdown(f"**Grad-CAM++** · {result['average_drop_plus_plus']:.1f}% avg drop")
                st.image(result["overlay_plus_plus"], width="stretch")

            # ── PDF report export ──────────────────────────────────────
            st.markdown("---")
            try:
                pdf_bytes = build_report_pdf(
                    original_img=result["original"],
                    gradcam_img=result["overlay"],
                    pred_class=pred_class,
                    confidence=confidence,
                    all_probs=result["all_probs"],
                    average_drop=result["average_drop"],
                    model_info=f"EfficientNet{variant} ({num_params:,} params, "
                               f"{input_size[0]}x{input_size[1]})  |  96.94% val accuracy",
                )
                st.download_button(
                    "📄 Download PDF Report", data=pdf_bytes,
                    file_name="disaster_assessment_report.pdf", mime="application/pdf",
                    width="stretch",
                )
            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.info(f"PDF export unavailable: {exc}")

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — ZONE ANALYSIS
# ════════════════════════════════════════════════════════════════════════
with tab_zone:
    st.markdown('<div class="eyebrow">Zone Damage Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="explain">Large aerial images are tiled into 224×224 patches '
        "(sliding window, stride 224). Each patch is classified independently and "
        "tinted by its predicted class, so you can see <b>where</b> damage is "
        "concentrated. Normal-sized images resolve to a single zone.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if image_bytes is None:
        st.info("Pick or upload an image first (sidebar sample or the uploader above).")
    else:
        run_zone = st.button("▶️ Run zone analysis", width="stretch")
        if run_zone:
            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                with st.spinner("Tiling and classifying zones…"):
                    overlay, stats = zone_damage_map(model, pil_img)

                zc1, zc2 = st.columns(2)
                with zc1:
                    st.markdown("**Original**")
                    st.image(pil_img, width="stretch")
                with zc2:
                    st.markdown("**Damage map**")
                    st.image(overlay, width="stretch")

                rows, cols = stats["grid_shape"]
                dmg = stats["damage_pct"]
                dmg_color = "#1D9E75" if dmg == 0 else ("#EAB308" if dmg < 50 else "#E24B4A")
                st.markdown(
                    f"""
                    <div class="card bracket">
                      <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:700;">
                        <span style="color:{dmg_color};">{dmg:.0f}% of zones show damage</span>
                        &nbsp;·&nbsp; {stats['n_damage']} / {stats['n_tiles']} tiles
                        &nbsp;·&nbsp; grid {rows}×{cols}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Legend + per-class tile counts
                legend_html = '<div class="card bracket"><div class="eyebrow" style="margin-bottom:10px;">Legend &amp; Zone Counts</div>'
                for cls in CLASS_NAMES:
                    cnt = stats["class_counts"][cls]
                    legend_html += (
                        f'<div class="legend-row">'
                        f'<span class="legend-swatch" style="background:{CLASS_COLORS[cls]};"></span>'
                        f'<span>{CLASS_EMOJI[cls]} <b>{cls}</b> — {cnt} zone(s)</span></div>'
                    )
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)

            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.error(f"Zone analysis failed for this image: {exc}")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="eyebrow">Model Performance</div>', unsafe_allow_html=True)

    metrics = [
        ("97%", "Overall Accuracy"),
        ("96%", "Macro F1"),
        ("96.94%", "Best Val Accuracy"),
        ("15 + 8", "Epochs (P1 + P2)"),
        ("16,723", "Images · 4 classes"),
    ]
    mcols = st.columns(len(metrics))
    for col, (val, lbl) in zip(mcols, metrics):
        col.markdown(
            f'<div class="telemetry"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Training curves**")
        curves = os.path.join(OUT_DIR, "B0_curves.png")
        if os.path.isfile(curves):
            st.image(curves, width="stretch")
        else:
            st.info("`outputs/B0_curves.png` not found — run notebook 02.")
    with pc2:
        st.markdown("**Confusion matrix (test set)**")
        cm = os.path.join(OUT_DIR, "B0_confusion.png")
        if os.path.isfile(cm):
            st.image(cm, width="stretch")
        else:
            st.info("`outputs/B0_confusion.png` not found — run notebook 02.")

    st.markdown("**Classification report (test set)**")
    report_path = os.path.join(OUT_DIR, "B0_report.txt")
    if os.path.isfile(report_path):
        try:
            import pandas as pd
            with open(report_path) as f:
                report_txt = f.read()
            rows = []
            for line in report_txt.strip().splitlines():
                parts = line.split()
                if not parts or parts[0] == "precision":
                    continue
                if parts[0] == "accuracy":
                    rows.append(["accuracy", "", "", parts[1], parts[2]])
                elif parts[0] in ("macro", "weighted"):
                    rows.append([f"{parts[0]} avg", parts[2], parts[3], parts[4], parts[5]])
                else:
                    rows.append([parts[0], parts[1], parts[2], parts[3], parts[4]])
            df_report = pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"])
            st.dataframe(df_report, width="stretch", hide_index=True)
        except Exception as exc:
            st.info(f"Could not parse classification report: {exc}")
            st.text(report_txt)
    else:
        st.info("`outputs/B0_report.txt` not found — run notebook 02.")

    st.markdown("**Model comparison**")
    comp_path = os.path.join(OUT_DIR, "model_comparison.csv")
    if os.path.isfile(comp_path):
        try:
            import pandas as pd
            df_comp = pd.read_csv(comp_path)
            st.dataframe(df_comp, width="stretch", hide_index=True)
        except Exception as exc:
            st.info(f"Could not read model comparison: {exc}")
    else:
        st.info("`outputs/model_comparison.csv` not found — run notebook 02.")
    st.caption(
        "ℹ️ EfficientNetB1 and B3 were skipped due to CPU-only training "
        "constraints — B0 already reaches 97% test accuracy, so the extra "
        "wall-clock cost of the larger backbones wasn't justified for this project."
    )