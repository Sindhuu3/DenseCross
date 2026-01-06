import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

from gradcam import grad_cam, detect_orientation
from utils import preprocess_image

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)

st.set_page_config(page_title="DFU Grade Prediction", layout="centered")
st.title("🦶 Diabetic Foot Ulcer Grade Prediction")
st.caption("AI-powered DFU severity assessment with Grad-CAM")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "dfu_densenet_model.h5",  # you can change to .keras later
        compile=False
    )
    with open("class_map.json") as f:
        class_map = json.load(f)

    return model, class_map

model, class_map = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload wound image", type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_arr = preprocess_image(image, IMG_SIZE)

    # Prediction
    preds = model.predict(img_arr)[0]
    grade_idx = int(np.argmax(preds))
    grade = grade_idx + 1
    confidence = preds[grade_idx] * 100

    # Grad-CAM
    heatmap = grad_cam(model, img_arr)
    orientation = detect_orientation(heatmap)

    # -------- Heatmap Visualization (NO cv2) --------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.resize(IMG_SIZE))
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")

    # ---------------- RESULTS ----------------
    st.subheader(f"🩺 Predicted Grade: Grade {grade}")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
    st.write(f"📍 Spread Direction: **{orientation}**")

    st.pyplot(fig)

    st.subheader("🔍 Grade Probabilities")
    for i, p in enumerate(preds):
        st.write(f"Grade {i+1}: {p * 100:.2f}%")
