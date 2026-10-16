"""
predict.py
==========
Inference for a single image + Grad-CAM heatmap (AI explainability).

Grad-CAM shows WHERE the network "looked" when it made its decision.
In medical AI this is not a gimmick - radiologists need to know the model is
looking at the lung opacity, not at a corner artifact, before they trust it.

Outputs
-------
outputs/predictions/<image>_gradcam.png
dict: {"prediction", "confidence", "class_index", "gradcam_path"}
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src import config
from src.model import load_trained_model
from src.preprocessing import load_and_preprocess


def _find_last_conv_layer(model):
    """Return the last convolutional layer (works for CNN and MobileNetV2)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam(image_batch: np.ndarray, model, class_index: int = 0):
    """
    Compute a Grad-CAM heatmap for one preprocessed image batch.

    Returns
    -------
    heatmap : np.ndarray (H, W) in [0, 1]
    prob    : float - predicted PNEUMONIA probability
    """
    last_conv = _find_last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs,
                                [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        loss = predictions[:, 0] if class_index == 1 else 1.0 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)                 # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (c,)
    conv_outputs = conv_outputs[0]                            # (h, w, c)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]    # (h, w, 1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    return heatmap.numpy(), float(predictions.numpy().ravel()[0])


def overlay_gradcam(original_gray: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend the Grad-CAM heatmap onto the grayscale X-ray (BGR for saving)."""
    h, w = original_gray.shape[:2]
    # Smooth + upsample the coarse conv heatmap (e.g. 16x16) to image size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (0, 0), sigmaX=3)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_LINEAR)
    image_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)
    return overlay


def predict_image(image_path: str | Path,
                  model_path=None,
                  save_gradcam: bool = True) -> dict:
    """
    Full single-image inference pipeline.

    Returns a dictionary with the diagnosis, confidence and heatmap path -
    the same structure the Flask API returns as JSON.
    """
    model = load_trained_model(model_path)

    # Preprocess exactly like training ------------------------------------
    batch = load_and_preprocess(image_path)                  # (1, H, W, 1) in [0,1]
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    raw_resized = cv2.resize(raw, (config.IMG_SIZE, config.IMG_SIZE))

    # Prediction ----------------------------------------------------------
    prob = float(model.predict(batch, verbose=0).ravel()[0])
    class_index = int(prob >= 0.5)
    label = config.CLASS_NAMES[class_index]
    confidence = prob if class_index == 1 else 1.0 - prob

    result = {
        "image_path": str(image_path),
        "prediction": label,
        "class_index": class_index,
        "pneumonia_probability": round(prob, 4),
        "confidence": round(float(confidence), 4),
    }

    # Grad-CAM explanation ----------------------------------------------
    try:
        heatmap, _ = make_gradcam(batch, model)
        if save_gradcam:
            config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            out_name = Path(image_path).stem + "_gradcam.png"
            out_path = config.PREDICTIONS_DIR / out_name
            overlay = overlay_gradcam(raw_resized, heatmap)
            cv2.imwrite(str(out_path), overlay)
            result["gradcam_path"] = str(out_path)
    except Exception as exc:      # Grad-CAM is best-effort; never block inference
        result["gradcam_error"] = str(exc)

    return result


def predict_sample_grid(n: int = 8, model_path=None, seed: int = config.SEED) -> list:
    """
    Pull `n` random test images (balanced across classes), predict all of them,
    save a nice side-by-side grid figure.  This becomes a README highlight.
    """
    import random
    from src import visualize

    model = load_trained_model(model_path)
    rng = random.Random(seed)
    samples = []
    for cls in config.CLASS_NAMES:
        cls_dir = config.DATA_DIR / "test" / cls
        files = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpeg"))
        samples += [(f, cls) for f in rng.sample(files, min(n // 2, len(files)))]

    records, batches, heats = [], [], []
    for fpath, true_label in samples:
        batch = load_and_preprocess(fpath)
        prob = float(model.predict(batch, verbose=0).ravel()[0])
        pred_label = config.CLASS_NAMES[int(prob >= 0.5)]
        heatmap, _ = make_gradcam(batch, model)
        raw = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        raw = cv2.resize(raw, (config.IMG_SIZE, config.IMG_SIZE))
        batches.append(raw)
        heats.append(overlay_gradcam(raw, heatmap))
        records.append({
            "image_path": str(fpath),
            "true_label": true_label,
            "prediction": pred_label,
            "pneumonia_probability": round(prob, 4),
        })

    visualize.plot_prediction_grid(batches, records, heats)
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict one X-ray image")
    parser.add_argument("--image", type=str, required=True, help="path to X-ray image")
    args = parser.parse_args()
    print(predict_image(args.image))
