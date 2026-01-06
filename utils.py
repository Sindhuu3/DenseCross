import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for DenseNet / CNN inference
    Streamlit Cloud safe (NO OpenCV)
    """

    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")

    # Resize using PIL
    image = image.resize(target_size)

    # Convert to numpy
    img = np.array(image).astype("float32")

    # Normalize
    img /= 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


