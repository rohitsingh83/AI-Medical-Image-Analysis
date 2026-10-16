# 2. Tech Stack Options — Choose the Best Beginner-Friendly Stack

---

## Option A — Easiest (Beginner, CPU-only, 1–2 days)

| Aspect | Choice |
|---|---|
| Language | Python 3.10+ |
| Libraries | TensorFlow/Keras, OpenCV, NumPy, Matplotlib, scikit-learn |
| Model type | Simple custom CNN (Conv → Pool → Dense) |
| Dataset | Virtual simulation dataset (auto-generated) or small public subset |
| Difficulty | ★☆☆☆☆ |
| GPU required | No — CPU training in minutes |
| Expected output | Binary classifier (~90%+ on demo data), accuracy/loss plots, confusion matrix |

**Best for:** first end-to-end ML project, fast GitHub proof of work.

## Option B — Intermediate (Balanced, 3–5 days) ✅ SELECTED

| Aspect | Choice |
|---|---|
| Language | Python 3.10+ |
| Libraries | TensorFlow/Keras 3, OpenCV (CLAHE), NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Flask |
| Model type | Custom CNN **+** MobileNetV2 transfer learning option |
| Dataset | Virtual simulation generator + drop-in Kaggle "Chest X-Ray Images (Pneumonia)" (Kermany 2018) |
| Difficulty | ★★☆☆☆ |
| GPU required | No (CPU works); Google Colab GPU speeds up real-data training |
| Expected output | Full pipeline: training logs, 7 publication-style figures, metrics.json, Grad-CAM explainability, Flask web app + JSON API |

**Best for:** students who want **easy execution + strong GitHub proof + practical understanding**. This is exactly what this repository implements.

## Option C — Advanced (Portfolio/Research grade, 1–3 weeks)

| Aspect | Choice |
|---|---|
| Language | Python + Docker + cloud |
| Libraries | PyTorch Lightning / TF, MONAI (medical imaging framework), pydicom, Albumentations, MLflow/W&B, FastAPI, Docker |
| Model type | ResNet-50 / DenseNet / EfficientNet + U-Net segmentation; ensemble; test-time augmentation |
| Dataset | Real Kaggle/NIH/MIMIC-CXR with DICOM metadata, 5-class diseases |
| Difficulty | ★★★★☆ |
| GPU required | Yes (Colab T4 / local RTX) |
| Expected output | Multi-class detection + segmentation, experiment tracking, containerised API, CI/CD |

**Best for:** ML research internships, medical-AI specialised roles.

---

## Final Selection: Option B (Intermediate)

**Why Option B wins for a student:**

1. **Easy execution** — runs on any laptop CPU in minutes (no GPU gatekeeping)
2. **Strong GitHub proof** — logs, figures, metrics, web app demo = screenshot-rich README
3. **Practical understanding** — you touch every stage of the real industry pipeline
4. **Realistic** — same code trains on the real Kaggle dataset with one config change
5. **Interview-ready** — transfer learning AND custom CNN both available to discuss

### Selected Stack Summary

```
Python 3.10+  ·  TensorFlow / Keras 3  ·  OpenCV  ·  NumPy  ·  Pandas
scikit-learn  ·  Matplotlib / Seaborn  ·  Flask
Models: Custom CNN (default) | MobileNetV2 transfer learning (optional)
Data:   Virtual simulation generator | Kaggle Chest X-Ray (drop-in)
```

---

*Previous: [1. Project Explanation](01_project_explanation.md) · Next: [3. Architecture](03_architecture.md)*
