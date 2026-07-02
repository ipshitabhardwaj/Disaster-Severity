
import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import glob
import tempfile
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from utils.gradcam import predict_and_explain
from utils.predict import CLASS_NAMES as PREDICT_CLASS_NAMES

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.h5')
TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'Test')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Severity Assessment",
    page_icon="🛰️",
    layout="wide"
)

CLASS_NAMES = list(PREDICT_CLASS_NAMES)
CLASS_COLORS = {
    'Earthquake': '#e74c3c',
    'Fire':       '#e67e22',
    'Flood':      '#3498db',
    'Normal':     '#2ecc71'
}
CLASS_EMOJI = {
    'Earthquake': '🏚️',
    'Fire':       '🔥',
    'Flood':      '🌊',
    'Normal':     '✅'
}


def confidence_color(prob):
    """Green >80%, yellow 60-80%, red <60%."""
    if prob > 0.80:
        return '#2ecc71'
    if prob >= 0.60:
        return '#f1c40f'
    return '#e74c3c'


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def get_model_variant(model):
    """Guess EfficientNet variant from input resolution."""
    h, w = model.input_shape[1:3]
    size_map = {224: 'B0', 240: 'B1', 300: 'B3'}
    variant = size_map.get(h, f'Custom ({h}x{w})')
    return variant, (h, w)


@st.cache_data
def get_sample_image(class_name):
    """First available test image for `class_name`, or None."""
    files = (glob.glob(os.path.join(TEST_DIR, class_name, '*.jpg')) +
             glob.glob(os.path.join(TEST_DIR, class_name, '*.jpeg')) +
             glob.glob(os.path.join(TEST_DIR, class_name, '*.png')))
    return files[0] if files else None


st.sidebar.title("🛰️ Disaster Severity Assessment")
st.sidebar.markdown("Upload an aerial image to classify the disaster type and visualize model attention.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Classes:**")
for cls, emoji in CLASS_EMOJI.items():
    st.sidebar.markdown(f"{emoji} {cls}")

st.title("🛰️ Disaster Severity Assessment")
st.markdown("Powered by **EfficientNet** + **Grad-CAM** | AIDERv2 Dataset")
st.markdown("---")

if not os.path.isfile(MODEL_PATH):
    st.error(
        f"Trained model not found at `{MODEL_PATH}`.\n\n"
        "Train and save the model first by running "
        "**notebooks/02_model_training.ipynb** to completion — "
        "its final cell promotes the best variant to "
        "`models/best_model.h5`, which this app loads."
    )
    st.stop()

model = load_model()
variant, input_size = get_model_variant(model)
num_params = model.count_params()

# ── Sidebar: model info ─────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Model info:**")
st.sidebar.markdown(f"- Variant: **EfficientNet{variant}**")
st.sidebar.markdown(f"- Parameters: **{num_params:,}**")
st.sidebar.markdown(f"- Input size: **{input_size[0]}×{input_size[1]}**")

# ── Sidebar: confidence threshold ───────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Review threshold:**")
confidence_threshold = st.sidebar.slider(
    "Warn below this confidence",
    min_value=0.40, max_value=0.80, value=0.60, step=0.05,
    format="%.0f%%",
)

# ── Sidebar: sample images ──────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Try a sample image:**")

if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = None

sample_cols = st.sidebar.columns(2)
for i, cls in enumerate(CLASS_NAMES):
    col = sample_cols[i % 2]
    if col.button(f"{CLASS_EMOJI[cls]} {cls}", key=f"sample_{cls}", use_container_width=True):
        sample_path = get_sample_image(cls)
        if sample_path:
            st.session_state.selected_image_path = sample_path
        else:
            st.sidebar.warning(f"No sample image found for {cls}.")

uploaded_file = st.file_uploader(
    "Upload an aerial image (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)

image_path = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name
elif st.session_state.selected_image_path:
    image_path = st.session_state.selected_image_path
    st.caption(f"Showing sample image: `{os.path.basename(image_path)}`")

if image_path is not None:
    with st.spinner("Analyzing image..."):
        result = predict_and_explain(model, image_path)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(result['original'], use_container_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Overlay")
        st.image(result['overlay'], use_container_width=True)

    with col3:
        st.subheader("🔀 Grad-CAM++ Overlay")
        st.image(result['overlay_plus_plus'], use_container_width=True)

    st.markdown("---")

    pred_class = result['class']
    confidence = result['confidence']
    color = CLASS_COLORS[pred_class]
    emoji = CLASS_EMOJI[pred_class]

    st.markdown(f"""
    <div style='background-color:{color}22; border-left: 5px solid {color};
    padding: 16px; border-radius: 8px;'>
        <h2 style='color:{color}'>{emoji} Predicted: {pred_class}</h2>
        <h3>Confidence: {confidence:.2%}</h3>
    </div>
    """, unsafe_allow_html=True)

    if confidence < confidence_threshold:
        st.warning("⚠️ Low confidence — human review recommended")

    st.markdown("---")

    st.subheader("📊 Class Probabilities")
    probs = result['all_probs']
    bar_colors = [confidence_color(v) for v in probs.values()]
    fig = go.Figure(go.Bar(
        x=list(probs.values()),
        y=list(probs.keys()),
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.2%}" for v in probs.values()],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 >80% confident · 🟡 60–80% · 🔴 <60%")

    if uploaded_file is not None:
        os.unlink(image_path)

else:
    st.info("👆 Upload an aerial disaster image, or pick a sample from the sidebar, to get started.")
