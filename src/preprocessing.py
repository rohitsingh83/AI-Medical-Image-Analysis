"""
preprocessing.py
================
Image-level preprocessing utilities used for:
  * the data-exploration notebook
  * the "preprocessing demo" figure in outputs/figures
  * the Flask upload path (consistency with training is critical!)

Pipeline (classic medical-imaging style)
----------------------------------------
  1. Load grayscale
  2. Resize to IMG_SIZE x IMG_SIZE (network input)
  3. CLAHE contrast enhancement (brings out faint lung opacities)
  4. Normalize to [0, 1]
  5. Add channel dimension -> (H, W, 1)

NOTE: the Keras training pipeline feeds raw [0-255] float pixels and the model
contains its own `Rescaling(1/255)` layer.  `preprocess_for_model()` therefore
returns [0-255] float32 - EXACTLY like training.  (Normalizing here too would
divide by 255 twice and silently break inference - a classic deployment bug!)
"""

from pathlib import Path

import cv2
import numpy as np

from src import config


def resize(image: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize with area interpolation (best for shrinking medical images)."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization - standard in radiology CV."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale uint8 [0,255] -> float32 [0,1]."""
    return image.astype(np.float32) / 255.0


def preprocess_for_model(image: np.ndarray,
                         size: int = config.IMG_SIZE,
                         use_clahe: bool = False) -> np.ndarray:
    """
    Full preprocessing for one image, ready for model.predict().

    Parameters
    ----------
    image : np.ndarray  - grayscale uint8 image of any size
    use_clahe : bool    - enable CLAHE enhancement (off by default to match training)

    Returns
    -------
    np.ndarray of shape (1, size, size, 1), float32 in [0, 255]
    (the model's embedded Rescaling layer converts to [0, 1])
    """
    if image.ndim == 3:                      # drop colour channels if present
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(image, size)
    if use_clahe:
        image = apply_clahe(image)
    image = image.astype(np.float32)         # keep [0,255] to match training!
    return image.reshape(1, size, size, 1)


def load_and_preprocess(image_path: str | Path, use_clahe: bool = False) -> np.ndarray:
    """Convenience wrapper: file path -> model-ready tensor."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return preprocess_for_model(image, use_clahe=use_clahe)
