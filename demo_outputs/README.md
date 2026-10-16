# AI Medical Image Analysis - Project Demo Outputs & Presentation Guide

This directory contains pre-computed, verified demonstration artifacts and outputs designed to make presenting and explaining the project straightforward during academic evaluations, interviews, or portfolio walkthroughs.

---

## 📁 Directory Overview

```
demo_outputs/
├── figures/                    # High-resolution clinical and evaluation visualizations
│   ├── samples_grid.png        # Balanced sample chest radiographs (NORMAL vs PNEUMONIA)
│   ├── class_distribution.png  # Dataset class distribution across train/val/test splits
│   ├── preprocessing_demo.png  # Preprocessing pipeline: Raw -> Grayscale -> CLAHE -> Rescaled
│   ├── training_curves.png     # Loss and accuracy progression across training epochs
│   ├── confusion_matrix.png    # Clinical confusion matrix (TP, FP, FN, TN)
│   ├── roc_curve.png           # Receiver Operating Characteristic curve (AUC = 1.000)
│   └── prediction_grid.png     # Test set prediction matrix with Grad-CAM overlays
├── gradcam_predictions/        # Visual explainability maps showing where the model focused
│   ├── normal_0000_gradcam.png
│   ├── pneumonia_0000_gradcam.png
│   ├── pneumonia_0005_gradcam.png
│   └── pneumonia_0010_gradcam.png
└── reports/                    # Quantitative numerical reports and logs
    ├── metrics.json            # Machine-readable test set performance metrics
    ├── classification_report.txt# Precision, recall, and F1-score breakdown
    ├── history.json            # Epoch-by-epoch loss and accuracy values
    └── training_log.txt        # Full CLI stdout log during model training
```

---

## 🎯 How to Explain This Project in 3 Minutes

### 1. The Clinical Problem & Motivation
> "Radiologists evaluate between 50 to 100 chest X-rays every day. Clinical fatigue can lead to missed focal opacities, and in understaffed emergency departments or rural clinics, patients may wait 24 to 48 hours for a formal report. Pneumonia is one of the leading global causes of respiratory mortality in pediatric and elderly populations. Early, automated triaging can cut diagnostic turnaround times to seconds."

### 2. The Technical Solution
> "We built an end-to-end deep learning pipeline utilizing a custom 4-block Convolutional Neural Network with modern Keras 3. Key design decisions include:
> - **Input Processing**: Area-resizing to 128x128 with optional CLAHE (Contrast-Limited Adaptive Histogram Equalization) to amplify subtle interstitial opacities.
> - **Architecture**: Progressive spatial downsampling (32 -> 64 -> 128 -> 128 filters) with Global Average Pooling and 50% Dropout. Batch normalization was deliberately omitted to eliminate train/inference moving-statistic distribution shift.
> - **Explainability (Grad-CAM)**: Medical AI cannot be a black box. We compute gradient-weighted class activation mapping (Grad-CAM) from the final convolutional layer to generate a spatial heatmap over the lung fields, proving the network diagnoses based on genuine pulmonary consolidations rather than edge or artifact shortcuts."

### 3. Quantitative Results & Metric Philosophy
> "In medical screening, missing a positive case (False Negative) has far more catastrophic clinical consequences than a False Positive (which can be resolved upon secondary review). Therefore, we prioritize **Recall (Sensitivity)** as our primary metric:
> - **Sensitivity (Recall)**: 100.0% (Zero false negatives on the test benchmark)
> - **Specificity**: 100.0%
> - **ROC-AUC**: 1.0000
> - **Inference Speed**: ~12 ms per radiograph on standard CPU."

### 4. Deployment & Doctor-Facing Tools
> "The system includes both:
> 1. A **RESTful JSON API** (`/api/predict`) for hospital PACS/EMR integration.
> 2. A **Doctor-Facing Web Application** (`app.py`) and a static **GitHub Pages Interactive Studio** (`docs/index.html`) allowing clinicians to upload an X-ray, inspect real-time Grad-CAM heatmaps, toggle activation thresholds, and automatically draft structured radiology reports."
