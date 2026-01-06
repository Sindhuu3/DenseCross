import streamlit as st
import tensorflow as tf #app.py
import numpy as np
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Safe OpenCV import for Streamlit Cloud
try:
    import cv2
except ImportError:
    cv2 = None
    st.warning("⚠️ OpenCV (cv2) not available. Grad-CAM features may be limited.")

import json
from PIL import Image
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
        "dfu_densenet_model.h5",
        compile=False
    )
    with open("class_map.json") as f:
        class_map = json.load(f)
    inv_map = {v: k for k, v in class_map.items()}
    return model, inv_map

model, class_map = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload wound image", type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_arr = preprocess_image(image, IMG_SIZE)

    preds = model.predict(img_arr)[0]
    grade_idx = np.argmax(preds)
    grade = grade_idx + 1
    confidence = preds[grade_idx] * 100

    heatmap = grad_cam(model, img_arr)
    orientation = detect_orientation(heatmap)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap_col = cv2.applyColorMap(
        np.uint8(255 * heatmap), cv2.COLORMAP_JET
    )

    overlay = heatmap_col * 0.4 + np.array(image.resize(IMG_SIZE))

    st.subheader(f"🩺 Predicted Grade: Grade {grade}")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
    st.write(f"📍 Spread Direction: **{orientation}**")

    st.image(overlay.astype("uint8"), caption="Grad-CAM Explanation")

    st.subheader("🔍 Grade Probabilities")
    for i, p in enumerate(preds):
        st.write(f"Grade {i+1}: {p*100:.2f}%")
