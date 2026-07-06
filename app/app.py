"""Project 43 — Explainable Disaster Severity Assessment (Streamlit dashboard).

An aerial-image classifier (EfficientNetB0) with Grad-CAM / Grad-CAM++
explanations, a tile-based Zone Damage Map, a PDF report export, and a model
performance dashboard. Paths are anchored via `find_project_root()` (from
`utils/gradcam.py`) so nothing is hardcoded.
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
    page_title="Disaster Severity Assessment",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  /* Class colors — mirror utils/zonemap.CLASS_COLORS so the whole app
     (badge, chart, zone legend, and overlay tiles) stays consistent. */
  --eq: #E24B4A; --fire: #EF9F27; --flood: #378ADD; --normal: #1D9E75;
  /* Sophisticated slate palette */
  --bg-0: #0f172a; --bg-1: #131c31; --bg-2: #1a2438;
  --border: #26324a; --border-hi: #33415c;
  --text-hi: #f1f5f9; --text: #cbd5e1; --text-mut: #94a3b8;
  --accent: #60a5fa; --accent-2: #3b82f6;
}

/* Global typography */
html, body, [class*="css"], .stMarkdown, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #16223c 0%, var(--bg-0) 55%); }
.block-container { padding-top: 1.4rem; max-width: 1320px; }

h1, h2, h3, h4 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
.stMarkdown h4 { font-weight: 700; color: var(--text-hi); margin-top: 0.2rem; }

/* ── Hero header ───────────────────────────────────────────────────── */
.hero {
  position: relative; overflow: hidden;
  background:
    linear-gradient(125deg, #0b1220 0%, #14264a 48%, #1e3a63 100%);
  border-radius: 20px; padding: 30px 34px; margin-bottom: 22px;
  border: 1px solid rgba(96,165,250,0.16);
  box-shadow: 0 12px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
  animation: heroIn 0.5s cubic-bezier(.22,.61,.36,1) both;
}
/* Subtle dot-grid texture overlay for depth */
.hero::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
  background-size: 22px 22px; opacity: 0.6;
  -webkit-mask-image: linear-gradient(105deg, #000 0%, transparent 70%);
          mask-image: linear-gradient(105deg, #000 0%, transparent 70%);
}
/* Soft glow accent in the corner */
.hero::after {
  content: ""; position: absolute; top: -60px; right: -40px;
  width: 260px; height: 260px; border-radius: 50%; pointer-events: none;
  background: radial-gradient(circle, rgba(96,165,250,0.28) 0%, transparent 70%);
  filter: blur(8px);
}
.hero h1 { color: #fff; margin: 0; font-size: 2.1rem; font-weight: 800;
  letter-spacing: -0.02em; position: relative; z-index: 1; }
.hero p { color: #b9c6da; margin: 6px 0 0 0; font-size: 0.97rem;
  position: relative; z-index: 1; }
.hero .badges { margin-top: 16px; position: relative; z-index: 1; }
.hero .pill {
  display: inline-block; background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.16); color: #eaf1fb;
  padding: 6px 14px; border-radius: 999px; font-size: 0.8rem;
  margin: 0 8px 8px 0; font-weight: 600; letter-spacing: 0.01em;
  backdrop-filter: blur(6px); transition: all 0.25s ease;
}
.hero .pill:hover { border-color: rgba(96,165,250,0.5);
  background: rgba(96,165,250,0.12); }
/* Accuracy badge — accent gradient with animated shimmer + glow */
.hero .pill.accent {
  position: relative; overflow: hidden; color: #fff; border: none;
  background: linear-gradient(120deg, var(--normal) 0%, #0ea371 100%);
  box-shadow: 0 0 0 1px rgba(16,185,129,0.4), 0 0 18px rgba(16,185,129,0.35);
  animation: pulseGlow 2.6s ease-in-out infinite;
}
.hero .pill.accent::after {
  content: ""; position: absolute; top: 0; left: -60%; width: 50%; height: 100%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,0.45), transparent);
  transform: skewX(-20deg); animation: shimmer 3.2s ease-in-out infinite;
}
@keyframes heroIn { from { opacity: 0; transform: translateY(-10px); } }
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 1px rgba(16,185,129,0.4), 0 0 14px rgba(16,185,129,0.30); }
  50% { box-shadow: 0 0 0 1px rgba(16,185,129,0.55), 0 0 24px rgba(16,185,129,0.5); }
}
@keyframes shimmer {
  0% { left: -60%; } 55%, 100% { left: 130%; }
}

/* ── Card containers (glassmorphism) ───────────────────────────────── */
.card {
  background: linear-gradient(180deg, rgba(26,36,56,0.72) 0%, rgba(19,28,49,0.72) 100%);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--border); border-radius: 16px;
  padding: 18px 22px; box-shadow: 0 6px 22px rgba(0,0,0,0.30);
  margin-bottom: 16px; transition: transform 0.25s ease, box-shadow 0.25s ease,
    border-color 0.25s ease;
}
.card:hover {
  transform: translateY(-3px); border-color: var(--border-hi);
  box-shadow: 0 12px 32px rgba(0,0,0,0.42);
}
.card h3 { margin-top: 0; }

/* ── Severity / prediction badge ───────────────────────────────────── */
.sev-badge {
  position: relative; overflow: hidden;
  display: flex; align-items: center; gap: 16px;
  border-radius: 16px; padding: 20px 24px; margin: 4px 0 16px 0;
  color: #fff; font-weight: 800;
  box-shadow: 0 8px 26px rgba(0,0,0,0.35);
  animation: fadeUp 0.4s cubic-bezier(.22,.61,.36,1) both;
}
.sev-badge::after {
  content: ""; position: absolute; top: -40px; right: -20px;
  width: 150px; height: 150px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, transparent 70%);
  pointer-events: none;
}
.sev-badge .emoji { font-size: 2.5rem; position: relative; z-index: 1;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
.sev-badge .label { font-size: 1.85rem; line-height: 1.1; position: relative;
  z-index: 1; letter-spacing: -0.01em; }
.sev-badge .sub { font-size: 0.86rem; font-weight: 500; opacity: 0.92;
  position: relative; z-index: 1; }

/* ── Animated confidence meter ─────────────────────────────────────── */
.meter-wrap { background: var(--bg-0); border: 1px solid var(--border);
  border-radius: 999px; height: 28px; width: 100%; overflow: hidden;
  margin: 6px 0 2px 0; box-shadow: inset 0 1px 3px rgba(0,0,0,0.4); }
.meter-fill { height: 100%; border-radius: 999px; text-align: right;
  color: #fff; font-weight: 700; font-size: 0.8rem; line-height: 28px;
  padding-right: 12px; white-space: nowrap;
  box-shadow: 0 0 12px rgba(255,255,255,0.12);
  animation: growbar 1.0s cubic-bezier(.22,.61,.36,1) forwards; }
@keyframes growbar { from { width: 0% !important; } }

/* ── Metric cards ──────────────────────────────────────────────────── */
.metric-card {
  position: relative; overflow: hidden;
  background: linear-gradient(180deg, rgba(26,36,56,0.72) 0%, rgba(19,28,49,0.72) 100%);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--border); border-radius: 16px;
  padding: 18px 16px; text-align: center; height: 100%;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
  animation: fadeUp 0.5s cubic-bezier(.22,.61,.36,1) both;
}
.metric-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--accent-2), var(--accent));
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--border-hi);
  box-shadow: 0 12px 30px rgba(0,0,0,0.42); }
.metric-card .val { font-size: 1.9rem; font-weight: 800;
  background: linear-gradient(120deg, #eaf1fb, #9dc2ff);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; line-height: 1.15; }
.metric-card .lbl { font-size: 0.76rem; color: var(--text-mut); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }

/* ── Sidebar polish ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] div.stButton > button {
  border-radius: 12px; border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(30,41,66,0.9), rgba(21,30,52,0.9));
  font-weight: 600; transition: all 0.22s cubic-bezier(.22,.61,.36,1);
  padding: 11px 6px;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
  transform: translateY(-2px); border-color: var(--accent);
  box-shadow: 0 8px 18px rgba(96,165,250,0.28);
  background: linear-gradient(180deg, rgba(37,52,84,0.95), rgba(26,37,64,0.95));
}
section[data-testid="stSidebar"] div.stButton > button:active { transform: translateY(0); }

/* Primary action buttons in main area */
div[data-testid="stMainBlockContainer"] div.stButton > button,
.stDownloadButton > button {
  border-radius: 12px; transition: all 0.22s cubic-bezier(.22,.61,.36,1);
}
div[data-testid="stMainBlockContainer"] div.stButton > button:hover,
.stDownloadButton > button:hover {
  transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59,130,246,0.28);
}

/* Tabs */
button[data-testid="stTab"], .stTabs [data-baseweb="tab"] {
  font-weight: 600; letter-spacing: 0.01em;
}

/* Fade-in for images / results */
div[data-testid="stImage"] { animation: fadeUp 0.5s ease both; }

/* ── Shared bits ───────────────────────────────────────────────────── */
.legend-row { display:flex; align-items:center; gap:10px; margin: 5px 0; }
.legend-swatch { width:18px; height:18px; border-radius:5px; display:inline-block;
  box-shadow: 0 0 8px rgba(0,0,0,0.35); }
.explain { color: var(--text-mut); font-size:0.86rem; line-height: 1.55; }

/* Key/value rows in sidebar model card */
.kv { display:flex; justify-content:space-between; align-items:center;
  padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.86rem; }
.kv:last-child { border-bottom: none; }
.kv b { color: var(--text-hi); font-weight: 700; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } }

/* Custom loading spinner tint */
.stSpinner > div { border-top-color: var(--accent) !important; }
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
    <div class="sev-badge" style="background: linear-gradient(120deg, {color} 0%, {color}cc 100%);">
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


# ── Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🛰️ Disaster Severity Assessment</h1>
        <p>Explainable aerial-image disaster classification &mdash; Team Anuksha &middot; Rishika &middot; Ipshita &middot; C-DAC Mohali</p>
        <div class="badges">
            <span class="pill accent">✦ 97% Test Accuracy</span>
            <span class="pill">AIDERv2</span>
            <span class="pill">EfficientNetB0</span>
            <span class="pill">Grad-CAM + Grad-CAM++</span>
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
    st.markdown("### 🛰️ Control Panel")

    st.markdown(
        f"""
        <div class="card" style="padding:16px 18px; border-left:3px solid var(--accent-2);">
            <div style="font-weight:700; margin-bottom:10px; color:var(--text-hi);
                 text-transform:uppercase; letter-spacing:0.06em; font-size:0.78rem;">
                 Model Info</div>
            <div class="kv"><span class="explain">Variant</span><b>EfficientNet{variant}</b></div>
            <div class="kv"><span class="explain">Params</span><b>{num_params:,}</b></div>
            <div class="kv"><span class="explain">Input</span><b>{input_size[0]}×{input_size[1]}</b></div>
            <div class="kv"><span class="explain">Val accuracy</span>
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
            "This dashboard classifies aerial imagery into **Earthquake, Fire, "
            "Flood, or Normal** using an EfficientNetB0 transfer-learning model "
            "trained on the **AIDERv2** dataset.\n\n"
            "Predictions are made explainable with **Grad-CAM** and "
            "**Grad-CAM++** heatmaps, quantified by an **Average Drop %** "
            "faithfulness metric. The **Zone Analysis** tab tiles large images "
            "to localize damage, and every analysis can be exported to a "
            "**PDF report**.\n\n"
            "_C-DAC Mohali — ML to Generative AI & LLMs._"
        )

# ── Uploader (shared across tabs) ───────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an aerial image (PNG/JPG) — or pick a sample from the sidebar",
    type=["png", "jpg", "jpeg"],
)
if uploaded_file is not None:
    st.session_state.image_bytes = uploaded_file.getvalue()
    st.session_state.image_name = uploaded_file.name

image_bytes = st.session_state.image_bytes

tab_analyze, tab_zone, tab_perf = st.tabs(
    ["🔎 Analyze", "🗺️ Zone Analysis", "📊 Model Performance"]
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
                st.markdown("#### 📷 Input")
                st.image(result["original"], width="stretch",
                         caption=st.session_state.image_name or "uploaded image")

            with right:
                st.markdown("#### 🧭 Prediction")
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
                        color=bar_colors, cornerradius=8,
                        line=dict(color="rgba(255,255,255,0.08)", width=1),
                    ),
                    text=[f"{v:.1%}" for v in probs.values()], textposition="outside",
                    textfont=dict(size=13, color="#e2e8f0",
                                  family="Inter, sans-serif"),
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                ))
                fig.update_layout(
                    template="plotly_dark", xaxis=dict(range=[0, 1.08], tickformat=".0%"),
                    height=240, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                    bargap=0.35,
                )
                fig.update_yaxes(showgrid=False)
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig, width="stretch")

            st.markdown("---")
            st.markdown("#### 🔥 Explainability — where the model looked")
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
    st.markdown("#### 🗺️ Zone Damage Map")
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
                    <div class="card">
                      <div style="font-size:1.15rem; font-weight:700;">
                        <span style="color:{dmg_color};">{dmg:.0f}% of zones show damage</span>
                        &nbsp;·&nbsp; {stats['n_damage']} / {stats['n_tiles']} tiles
                        &nbsp;·&nbsp; grid {rows}×{cols}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Legend + per-class tile counts
                legend_html = '<div class="card"><b>Legend &amp; zone counts</b><br/>'
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
    st.markdown("#### 📊 Model Performance")

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
            f'<div class="metric-card"><div class="val">{val}</div>'
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
