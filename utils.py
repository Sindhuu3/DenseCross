import numpy as np #utils.py
import tensorflow as tf

def preprocess_image(image, img_size):
    img = image.resize(img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0
    return img_arr
