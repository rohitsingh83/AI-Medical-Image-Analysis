# 7. How to Run the Project & Expected Outputs

---

## 7.1 One-Command Demo

```bash
python main.py demo
```
Runs: simulate → explore → train → evaluate → predict. Everything below is produced automatically.

## 7.2 Stage-by-Stage Commands

```bash
# 1. Generate the virtual dataset (or place Kaggle data in data/raw/chest_xray/)
python main.py simulate

# 2. Dataset exploration figures
python main.py explore

# 3. Train the CNN (default 8 epochs; use --model mobilenetv2 for transfer learning)
python main.py train --epochs 8

# 4. Evaluate on the held-out test set
python main.py evaluate

# 5. Predict a balanced sample grid + Grad-CAM
python main.py predict

# 6. Predict ONE image
python main.py predict --image data/synthetic/chest_xray/test/PNEUMONIA/pneumonia_0000.png

# 7. Launch the web app
python app.py    # → http://127.0.0.1:5000
```

## 7.3 Expected Console Output (abbreviated)

```
============================================================
VIRTUAL SIMULATION - generating synthetic chest X-rays
============================================================
  [ok] train/NORMAL: 400 images (128x128)
  [ok] train/PNEUMONIA: 400 images (128x128)
  ...
Dataset ready at: .../data/synthetic/chest_xray

============================================================
TRAINING STARTED | model=cnn | epochs=8
============================================================
Epoch 1/8 ... val_accuracy: 0.9...
Epoch 2/8 ... val_accuracy: 1.0...
============================================================
TRAINING COMPLETE | best val accuracy: 100.00%
Model saved to   : .../models/pneumonia_cnn.keras

=============================================================
EVALUATION RESULTS (test set)
=============================================================
Accuracy      : 9x.xx%
Precision     : 0.9x
Recall        : 0.9x
F1 score      : 0.9x
ROC-AUC       : 0.9x+
```

## 7.4 Expected Artifacts After a Full Run

```
models/pneumonia_cnn.keras            trained model
outputs/reports/history.json          training curves data
outputs/reports/training_log.txt      human-readable log
outputs/reports/metrics.json          test metrics
outputs/reports/classification_report.txt
outputs/figures/*.png                 7 figures
outputs/predictions/*_gradcam.png     heatmaps
```

## 7.5 Expected Results — and How to Interpret Them

- On the **virtual dataset**, test accuracy is typically **high (95–100%)** because synthetic patterns are cleaner than real clinical data. This is expected — say so honestly in interviews.
- On the **real Kaggle dataset** (same code!), the same architecture typically reaches **~85–92% accuracy**, with recall slightly higher than precision when you weight the loss toward catching positives.
- If accuracy is ~50%: check the training log — the classic causes are (a) labels not loaded, (b) train/val preprocessing mismatch, (c) BatchNorm train/inference mismatch (we hit and fixed this! see `docs/03_architecture.md` §3.4).
- If train acc ≈ 100% but val/test is low: overfitting → more augmentation, fewer epochs, or more data.

## 7.6 Web API

```bash
# Health check
curl http://127.0.0.1:5000/health

# Predict
curl -F "image=@data/synthetic/chest_xray/test/NORMAL/normal_0000.png" \
     http://127.0.0.1:5000/api/predict
```
Response:
```json
{
  "prediction": "NORMAL",
  "class_index": 0,
  "pneumonia_probability": 0.0213,
  "confidence": 0.9787,
  "gradcam_path": "outputs/predictions/normal_0000_gradcam.png"
}
```

---

*Previous: [6. Virtual Simulation](06_virtual_simulation.md) · Next: [8. GitHub Strategy](08_github_strategy.md)*
