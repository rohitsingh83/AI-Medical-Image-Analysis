"""
config.py
=========
Central configuration file — every path and hyperparameter lives here.

Why a config file?
------------------
In industry projects, hard-coded values are scattered bugs waiting to happen.
One config module = one place to tune the whole system.
"""

import os
from pathlib import Path

# Disable oneDNN CPU feature probing on Windows to prevent import stalls
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so the repo works anywhere)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data root:
#   - "data/synthetic/chest_xray"  -> virtual simulation dataset (default, auto-generated)
#   - "data/raw/chest_xray"        -> real Kaggle "Chest X-Ray Images (Pneumonia)" dataset
#     (same folder layout: train/, val or test/ with NORMAL/ and PNEUMONIA/ subfolders)
DATA_DIR = PROJECT_ROOT / "data" / "synthetic" / "chest_xray"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
UPLOAD_DIR = PROJECT_ROOT / "uploads"          # Flask demo temporary uploads

# ----------------------------------------------------------------------------
# Image settings
# ----------------------------------------------------------------------------
IMG_SIZE = 128            # 128 for fast CPU training; use 256 for the real dataset
CHANNELS = 1              # grayscale X-rays
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # index 0 = NORMAL, index 1 = PNEUMONIA

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 8                # enough for the synthetic demo; 15-25 for real data
LEARNING_RATE = 1e-3
SEED = 42
VAL_SPLIT = 0.15          # used only when no explicit val/ folder exists (real Kaggle data)

# ----------------------------------------------------------------------------
# Artifact names
# ----------------------------------------------------------------------------
MODEL_NAME = "pneumonia_cnn.keras"          # Keras v3 native format (recommended)
MODEL_NAME_MOBILENET = "pneumonia_mobilenetv2.keras"
HISTORY_FILE = "history.json"
TRAIN_LOG_FILE = "training_log.txt"


def ensure_dirs() -> None:
    """Create every output folder the project needs (safe to call repeatedly)."""
    for directory in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR,
                      UPLOAD_DIR, RAW_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Project root :", PROJECT_ROOT)
    print("Data dir     :", DATA_DIR)
    print("Model dir    :", MODELS_DIR)
