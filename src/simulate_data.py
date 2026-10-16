"""
simulate_data.py
================
VIRTUAL SIMULATION DATASET GENERATOR
------------------------------------
We cannot legally redistribute real hospital X-rays inside a GitHub repo, and a
student laptop should not need a 2 GB download just to run the demo.  So this
module generates *synthetic chest X-ray phantoms* that mimic the structure of
real radiographs:

    NORMAL     : clear dark lung fields, visible ribs, spine, heart shadow
    PNEUMONIA  : same anatomy + cloudy bright opacities inside the lung fields
                 (exactly what consolidation looks like on a real X-ray)

The output folder structure is IDENTICAL to the famous Kaggle dataset
"Chest X-Ray Images (Pneumonia)" (Kermany et al., 2018):

    data/synthetic/chest_xray/
        train/NORMAL/       train/PNEUMONIA/
        val/NORMAL/         val/PNEUMONIA/
        test/NORMAL/        test/PNEUMONIA/

=> The SAME training code runs on this virtual dataset OR on the real Kaggle
   data.  You only change one path in src/config.py.  That is exactly how
   industry teams swap datasets behind a stable pipeline.

Usage
-----
    python -m src.simulate_data                  # default sizes
    python -m src.simulate_data --train 500      # more images per class
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src import config


# ---------------------------------------------------------------------------
# Anatomy helpers
# ---------------------------------------------------------------------------
def _torso_mask(size: int, cx: float, cy: float) -> np.ndarray:
    """Elliptical body silhouette."""
    yy, xx = np.mgrid[0:size, 0:size]
    return (((xx - cx) / (0.42 * size)) ** 2 + ((yy - cy) / (0.47 * size)) ** 2) <= 1.0


def _ellipse_mask(size: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (((xx - cx) / (rx * size)) ** 2 + ((yy - cy) / (ry * size)) ** 2) <= 1.0


def _add_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 6.0) -> np.ndarray:
    return img + rng.normal(0.0, sigma, img.shape)


def generate_normal(size: int, rng: np.random.Generator) -> tuple:
    """Generate one synthetic NORMAL chest X-ray phantom -> (image, lung_masks)."""
    img = np.full((size, size), 25.0)                       # dark background (air)
    cx, cy = size / 2, size / 2 + rng.uniform(-2, 2)

    # Soft tissue / torso
    torso = _torso_mask(size, cx, cy)
    img[torso] = 118.0

    # Lungs: two darker elliptical fields
    lung_masks = []
    for side in (-1, 1):
        lx = cx + side * 0.155 * size + rng.uniform(-2, 2)
        lung = _ellipse_mask(size, lx, cy - 0.03 * size, 0.125, 0.27)
        lung_masks.append(lung)
        img[lung] = 52.0

    # Spine: bright vertical column
    spine = _ellipse_mask(size, cx, cy + 0.02 * size, 0.035, 0.32)
    img[spine] = 175.0

    # Heart shadow (patient-left = image-right of center)
    heart = _ellipse_mask(size, cx + 0.05 * size, cy + 0.13 * size, 0.10, 0.14)
    img[heart] = 95.0

    # Ribs: curved dark arcs
    for i in range(7):
        thickness = rng.integers(2, 4)
        for side in (-1, 1):
            centre = (int(cx + side * 0.16 * size), int(cy - 0.22 * size + i * 0.075 * size))
            axes = (int(0.16 * size), int(0.10 * size))
            cv2.ellipse(img, centre, axes, angle=0 if side > 0 else 180,
                        startAngle=200 if side > 0 else 20, endAngle=340 if side > 0 else 160,
                        color=70.0, thickness=int(thickness))

    # Diaphragm domes at the bottom of the chest
    for side in (-1, 1):
        dome = _ellipse_mask(size, cx + side * 0.16 * size, cy + 0.34 * size, 0.16, 0.10)
        img[dome & torso] = 130.0

    img = _add_noise(img, rng, sigma=5.0)
    return img, lung_masks


def add_pneumonia_opacities(img: np.ndarray,
                            lung_masks: list,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Add cloudy bright consolidations inside the lung fields.
    This mimics pneumonia / infection opacities on a chest X-ray.
    """
    img = img.copy()
    size = img.shape[0]
    n_blobs = rng.integers(2, 5)                      # 2-4 opacities
    for _ in range(n_blobs):
        side = int(rng.integers(0, 2))                # pick a lung
        ys, xs = np.nonzero(lung_masks[side])
        if len(xs) == 0:
            continue
        idx = rng.integers(0, len(xs))
        cx_b, cy_b = xs[idx], ys[idx]
        # Bias opacities toward the lower lung zones (classic pneumonia pattern)
        cy_b = min(size - 3, cy_b + int(0.05 * size))
        radius = int(rng.uniform(0.035, 0.085) * size)
        cv2.circle(img, (int(cx_b), int(cy_b)), radius,
                   color=float(rng.uniform(135, 185)), thickness=-1)

    # Blur so the opacities look like soft consolidation, not hard circles
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=size * 0.012)
    return img


def generate_pneumonia(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one synthetic PNEUMONIA chest X-ray phantom."""
    img, lung_masks = generate_normal(size, rng)
    img = add_pneumonia_opacities(img, lung_masks, rng)
    return img


def augment_final(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild acquisition-style variation: rotation, shift, brightness."""
    size = img.shape[0]
    angle = rng.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    M[0, 2] += rng.uniform(-4, 4)
    M[1, 2] += rng.uniform(-4, 4)
    img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    img = img * rng.uniform(0.92, 1.08) + rng.uniform(-6, 6)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_split(split_dir: Path, label: str, count: int,
                   size: int, rng: np.random.Generator) -> None:
    """Generate `count` images for one class inside one split folder."""
    out = split_dir / label
    out.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        if label == "NORMAL":
            img, _ = generate_normal(size, rng)
        else:
            img = generate_pneumonia(size, rng)
        img = augment_final(img, rng)
        cv2.imwrite(str(out / f"{label.lower()}_{i:04d}.png"), img)
    print(f"  [ok] {split_dir.name}/{label}: {count} images ({size}x{size})")


def generate_dataset(data_root: Path, size: int,
                     n_train: int, n_val: int, n_test: int, seed: int) -> Path:
    """Generate the complete virtual dataset and return its root path."""
    rng = np.random.default_rng(seed)
    splits = {"train": n_train, "val": n_val, "test": n_test}
    print("=" * 60)
    print("VIRTUAL SIMULATION - generating synthetic chest X-rays")
    print("=" * 60)
    for split, count in splits.items():
        for label in config.CLASS_NAMES:
            generate_split(data_root / split, label, count, size, rng)
    print("=" * 60)
    print(f"Dataset ready at: {data_root}")
    return data_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the virtual X-ray dataset")
    parser.add_argument("--train", type=int, default=400, help="train images per class")
    parser.add_argument("--val", type=int, default=80, help="val images per class")
    parser.add_argument("--test", type=int, default=100, help="test images per class")
    parser.add_argument("--size", type=int, default=config.IMG_SIZE, help="image size px")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    generate_dataset(config.DATA_DIR, args.size,
                     args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
