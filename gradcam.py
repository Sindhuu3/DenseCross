import tensorflow as tf
import numpy as np


def grad_cam(model, img_array, layer_name="conv5_block16_concat"):
    """
    Generate Grad-CAM heatmap for a given model and image
    Works with DenseNet / ResNet / CNNs
    """

    # Build a model that maps input -> (conv output, predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ],
    )

    # Forward + backward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_max(predictions)

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolution outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)

    heatmap /= max_val

    return heatmap.numpy()


def detect_orientation(heatmap):
    """
    Detect dominant activation region in the heatmap
    """

    h, w = heatmap.shape

    scores = {
        "Left": heatmap[:, :w // 2].sum(),
        "Right": heatmap[:, w // 2:].sum(),
        "Top": heatmap[:h // 2, :].sum(),
        "Bottom": heatmap[h // 2:, :].sum(),
    }

    return max(scores, key=scores.get)
