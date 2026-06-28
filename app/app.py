
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gradcam import predict_and_explain

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disaster Severity Assessment",
    page_icon="🛰️",
    layout="wide"
)

CLASS_NAMES = ['Earthquake', 'Fire', 'Flood', 'Normal']
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

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'models', 'efficientnet_phase1.h5'
    )
    return tf.keras.models.load_model(model_path)

st.sidebar.title("🛰️ Disaster Severity Assessment")
st.sidebar.markdown("Upload an aerial image to classify the disaster type and visualize model attention.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Classes:**")
for cls, emoji in CLASS_EMOJI.items():
    st.sidebar.markdown(f"{emoji} {cls}")

st.title("🛰️ Disaster Severity Assessment")
st.markdown("Powered by **EfficientNetB0** + **Grad-CAM** | AIDERv2 Dataset")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an aerial image (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        model = load_model()
    except Exception:
        st.warning("⚠️ Model not found — using placeholder for UI demo.")
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(4, activation='softmax')(x)
        model = models.Model(base.input, out)

    with st.spinner("Analyzing image..."):
        result = predict_and_explain(model, tmp_path)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(result['original'], use_column_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        fig, ax = plt.subplots()
        ax.imshow(result['heatmap'], cmap='jet')
        ax.axis('off')
        st.pyplot(fig)

    with col3:
        st.subheader("🔀 Overlay")
        st.image(result['overlay'], use_column_width=True)

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

    st.markdown("---")

    st.subheader("📊 Class Probabilities")
    probs = result['all_probs']
    fig = go.Figure(go.Bar(
        x=list(probs.keys()),
        y=list(probs.values()),
        marker_color=[CLASS_COLORS[c] for c in probs.keys()],
        text=[f"{v:.2%}" for v in probs.values()],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=350,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    os.unlink(tmp_path)

else:
    st.info("👆 Upload an aerial disaster image to get started.")
