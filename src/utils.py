"""
utils.py
========
Small shared helpers: reproducibility, logging and image I/O.
"""

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config


def set_global_seed(seed: int = config.SEED) -> None:
    """
    Fix every random seed we can touch.

    Why? So that your GitHub results are reproducible for anyone who runs the code
    (interviewers LOVE reproducibility).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[utils] Global seed set to {seed}")


def load_image_gray(image_path: str | Path) -> np.ndarray:
    """Load an image as a grayscale NumPy array (uint8)."""
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    return image.numpy().squeeze().astype(np.uint8)


def save_json(data: dict, path: str | Path) -> None:
    """Save a dictionary as pretty JSON (used for history + metrics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[utils] Saved JSON -> {path}")


def load_json(path: str | Path) -> dict:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
