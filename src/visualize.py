"""
visualize.py
============
Every plot the project produces lives here (single responsibility).

Figures written to outputs/figures/
-----------------------------------
samples_grid.png          sample virtual X-rays (NORMAL vs PNEUMONIA)
preprocessing_demo.png    original -> resize -> CLAHE -> normalized
class_distribution.png    image counts per split and class
training_curves.png       accuracy + loss vs epoch
confusion_matrix.png      annotated heatmap
roc_curve.png             ROC with AUC score
prediction_grid.png       true vs predicted + Grad-CAM heatmaps
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")            # headless backend (works on servers / CI)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from src import config

# Consistent visual style for every figure (looks professional on GitHub)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 110,
})


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved figure -> {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Dataset exploration
# ---------------------------------------------------------------------------
def plot_sample_grid(data_dir: Path | str = config.DATA_DIR,
                     n_per_class: int = 4) -> Path:
    """Show a grid of sample X-rays: one row per class."""
    data_dir = Path(data_dir)
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6.2))
    for row, cls in enumerate(config.CLASS_NAMES):
        files = sorted((data_dir / "train" / cls).glob("*.png"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(files):
                img = cv2.imread(str(files[col]), cv2.IMREAD_GRAYSCALE)
                ax.imshow(img, cmap="gray")
            ax.set_title(f"{cls}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Virtual Simulation Dataset - Sample Chest X-Rays", fontsize=13)
    fig.tight_layout()
    return _save(fig, "samples_grid.png")


def plot_preprocessing_demo(image_path: Path | str = None) -> Path:
    """4-step preprocessing visual: original -> resize -> CLAHE -> normalized."""
    from src import preprocessing
    if image_path is None:
        files = sorted((config.DATA_DIR / "train" / "PNEUMONIA").glob("*.png"))
        image_path = files[0]
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    resized = preprocessing.resize(original)
    clahe = preprocessing.apply_clahe(resized)
    normalized = (preprocessing.normalize(clahe) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, img, title in zip(
            axes,
            [original, resized, clahe, normalized],
            ["1. Original", f"2. Resize ({config.IMG_SIZE}px)", "3. CLAHE contrast", "4. Normalized"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Medical Image Preprocessing Pipeline")
    fig.tight_layout()
    return _save(fig, "preprocessing_demo.png")


def plot_class_distribution(data_dir: Path | str = config.DATA_DIR) -> Path:
    """Bar chart: how many images per split and per class."""
    data_dir = Path(data_dir)
    splits = ["train", "val", "test"]
    counts = {cls: [] for cls in config.CLASS_NAMES}
    for split in splits:
        for cls in config.CLASS_NAMES:
            folder = data_dir / split / cls
            n = len(list(folder.glob("*.png"))) + len(list(folder.glob("*.jpeg")))
            counts[cls].append(n)

    x = np.arange(len(splits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, counts["NORMAL"], width, label="NORMAL", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, counts["PNEUMONIA"], width, label="PNEUMONIA", color="#C44E52")
    ax.bar_label(bars1, fontsize=8)
    ax.bar_label(bars2, fontsize=8)
    ax.set_xticks(x, splits)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset Class Distribution")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "class_distribution.png")


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict) -> Path:
    """Accuracy and loss curves (train vs validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(history["accuracy"], marker="o", label="Train Accuracy")
    axes[0].plot(history["val_accuracy"], marker="o", label="Val Accuracy")
    axes[0].set_title("Model Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["loss"], marker="o", label="Train Loss", color="#C44E52")
    axes[1].plot(history["val_loss"], marker="o", label="Val Loss", color="#DD8452")
    axes[1].set_title("Model Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].legend()

    fig.suptitle("Training History - Pneumonia Detection CNN")
    fig.tight_layout()
    return _save(fig, "training_curves.png")


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> Path:
    """Annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True Label", xlabel="Predicted Label",
           title="Confusion Matrix (Test Set)")
    # Write counts inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)
    fig.tight_layout()
    return _save(fig, "confusion_matrix.png")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Path:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(fpr, tpr, color="#C44E52", lw=2.2,
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve - Pneumonia Detector")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "roc_curve.png")


# ---------------------------------------------------------------------------
# 4. Prediction showcase
# ---------------------------------------------------------------------------
def plot_prediction_grid(images: list, records: list, overlays: list = None) -> Path:
    """
    Grid of predictions: X-ray + true/pred labels (optionally with Grad-CAM).
    This is the money screenshot for the README.
    """
    n = len(images)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    has_cam = overlays is not None
    fig, axes = plt.subplots(rows, cols * (2 if has_cam else 1),
                             figsize=(3.6 * cols * (2 if has_cam else 1), 3.4 * rows),
                             squeeze=False)

    for i, (img, rec) in enumerate(zip(images, records)):
        r, c = divmod(i, cols)
        base_col = c * (2 if has_cam else 1)
        ax = axes[r][base_col]
        ax.imshow(img, cmap="gray")
        ok = rec["true_label"] == rec["prediction"]
        color = "#2ca02c" if ok else "#d62728"
        ax.set_title(f"True: {rec['true_label']}\nPred: {rec['prediction']} "
                     f"({rec['pneumonia_probability']:.2f})",
                     fontsize=9, color=color)
        ax.axis("off")
        if has_cam:
            ax2 = axes[r][base_col + 1]
            ax2.imshow(cv2.cvtColor(overlays[i], cv2.COLOR_BGR2RGB))
            ax2.set_title("Grad-CAM heatmap", fontsize=8)
            ax2.axis("off")

    # Hide unused axes
    total_slots = rows * cols * (2 if has_cam else 1)
    for i in range(n * (2 if has_cam else 1), total_slots):
        r, c = divmod(i, cols * (2 if has_cam else 1))
        axes[r][c].axis("off")

    fig.suptitle("AI Predictions vs Ground Truth (Test Set)", fontsize=13)
    fig.tight_layout()
    return _save(fig, "prediction_grid.png")
