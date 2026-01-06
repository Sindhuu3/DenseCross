import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for DenseNet:
    - Resize
    - Normalize
    - Expand dims
    """

    if isinstance(image, Image.Image):
        img = image.resize(target_size)
        img = np.array(img)
    else:
        raise ValueError("Input must be a PIL Image")

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

