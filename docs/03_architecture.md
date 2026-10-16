# 3. Complete Project Architecture

---

## 3.1 System Block Diagram (Text)

```
                         ┌─────────────────────────────────────────────┐
                         │              INPUT: Medical Images          │
                         │   Chest X-Ray (PNG/JPG) - NORMAL / SICK     │
                         │   Sources: Kaggle dataset | virtual sim gen │
                         └──────────────────────┬──────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING MODULE (src/preprocessing.py, src/data_loader.py)              │
│   • Grayscale conversion        • Resize 128x128 (256x256 for real data)     │
│   • CLAHE contrast enhancement  • Normalization to [0,1]                     │
│   • Augmentation: flip / rotate / zoom / shift (training only)               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │  tf.data.Dataset (batched, prefetched)
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION + MODEL (src/model.py)                                    │
│                                                                              │
│   [Augmentation] → [Rescaling 1/255] → Conv(32) → Pool                      │
│                  → Conv(64) → Pool → Conv(128) → Pool                        │
│                  → Conv(128, "last_conv") → Pool                             │
│   (OR: MobileNetV2 pretrained backbone, frozen)                              │
│                                                                              │
│   → GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)     │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │  P(pneumonia) ∈ [0,1]
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PREDICTION & EXPLANATION (src/predict.py)                                    │
│   • threshold 0.5 → NORMAL / PNEUMONIA                                       │
│   • confidence score                                                          │
│   • Grad-CAM heatmap (last conv layer → weighted activation map)              │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────────┐
│ EVALUATION (src/evaluate.py)│     │ OUTPUT VISUALIZATION (src/visualize.py) │
│  • Accuracy / Precision     │     │  • training_curves.png                  │
│  • Recall / F1 / ROC-AUC    │     │  • confusion_matrix.png                 │
│  • Confusion matrix         │     │  • roc_curve.png                        │
│  • classification_report.txt│     │  • prediction_grid.png (+Grad-CAM)      │
│  • metrics.json             │     │  • samples_grid.png, preprocessing...   │
└─────────────────────────────┘     └─────────────────┬───────────────────────┘
                                                      ▼
                                    ┌─────────────────────────────────────────┐
                                    │ DEPLOYMENT (app.py, Flask)              │
                                    │  • Web UI: upload X-ray → result        │
                                    │  • JSON API: POST /api/predict          │
                                    │  • Health check: GET /health            │
                                    └─────────────────────────────────────────┘
```

## 3.2 Module Explanation

| Module | Responsibility | Why it exists |
|---|---|---|
| `src/config.py` | Single source of truth for paths & hyperparameters | No magic numbers scattered in code |
| `src/simulate_data.py` | Generate virtual X-ray phantoms | Reproducible demo without 2 GB download / privacy issues |
| `src/data_loader.py` | Folder structure → `tf.data` pipelines | Same contract as real Kaggle data |
| `src/preprocessing.py` | Resize, CLAHE, normalize | Consistency between training and deployment |
| `src/model.py` | CNN / MobileNetV2 builders | Swap architectures via one argument |
| `src/train.py` | Training loop, checkpoints, logging | Reproducible, resumable training |
| `src/evaluate.py` | Metrics + report files | The "proof" artifacts |
| `src/predict.py` | Single-image inference + Grad-CAM | Explainable predictions |
| `src/visualize.py` | All figures | README-quality visuals |
| `main.py` | CLI over everything | One-command demo |
| `app.py` | Flask web app | "Deployed product" story |

## 3.3 Data Flow (step by step)

1. **Collection** — `simulate_data.py` writes `train|val|test / NORMAL|PNEUMONIA` folders (or user drops the real Kaggle data into `data/raw/chest_xray/`)
2. **Loading** — `data_loader.py` builds batched `tf.data.Dataset`s; labels cast to float for sigmoid output
3. **Preprocessing** — embedded in the model (`Rescaling`, augmentation layers) so training and inference can never diverge
4. **Training** — Adam + binary cross-entropy; ModelCheckpoint saves the best `val_accuracy`; EarlyStopping prevents overfitting
5. **Prediction** — probability ≥ 0.5 → PNEUMONIA; Grad-CAM highlights decision regions
6. **Evaluation** — metrics.json + classification_report.txt + confusion matrix + ROC
7. **Visualization** — 7 PNG figures into `outputs/figures/`
8. **Deployment** — Flask loads the same `.keras` file; uploads run through identical preprocessing

## 3.4 Key Design Decisions (interview talking points)

1. **Preprocessing inside the model** (Rescaling/augmentation layers) — the saved `.keras` file carries its own preprocessing; deployment cannot mis-scale pixels.
2. **Recall-first evaluation** — in pneumonia screening a false negative (missed case) is worse than a false alarm; we report recall prominently.
3. **GlobalAveragePooling instead of Flatten** — fewer parameters, less overfitting, works at other input sizes.
4. **No BatchNormalization in the small CNN** — BN moving statistics are fragile with few batches; Conv+Pool+Dropout is the robust recipe at this scale (we diagnosed this empirically during development — a great debugging story).
5. **Grad-CAM explainability** — clinicians must see *where* the model looked before trusting it.
6. **Virtual simulation with real-data compatibility** — folder contract identical to Kaggle's dataset, so the pipeline is dataset-agnostic.

---

*Previous: [2. Tech Stack](02_tech_stack.md) · Next: [4. Implementation Plan](04_implementation_plan.md)*
