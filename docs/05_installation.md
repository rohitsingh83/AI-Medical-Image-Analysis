# 5. Installation & Environment Setup

---

## 5.1 Requirements

| Item | Version |
|---|---|
| Python | 3.10 – 3.13 (64-bit) |
| pip | latest |
| RAM | 4 GB minimum (8 GB recommended) |
| Disk | ~2 GB (TensorFlow) + dataset space |
| GPU | **Not required** (CPU training works; Colab GPU optional) |

## 5.2 Windows (PowerShell / CMD)

```powershell
# 1. Get the code
git clone https://github.com/<your-username>/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate          # PowerShell/CMD
# If PowerShell blocks scripts:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify
python main.py
```

## 5.3 macOS / Linux

```bash
git clone https://github.com/<your-username>/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

## 5.4 Google Colab (zero-install option)

```python
!git clone https://github.com/<your-username>/AI-Medical-Image-Analysis.git
%cd AI-Medical-Image-Analysis
!pip install -r requirements.txt
!python main.py demo
```
Enable **GPU runtime** (Runtime → Change runtime type → T4 GPU) for faster real-data training.

## 5.5 requirements.txt (what each package is for)

| Package | Purpose |
|---|---|
| tensorflow | CNN training + Keras 3 API |
| opencv-python | image I/O, CLAHE, Grad-CAM overlays |
| numpy / pandas | array + table operations |
| scikit-learn | metrics (confusion matrix, ROC, classification report) |
| matplotlib / seaborn | all figures |
| flask | web deployment |
| jupyter / ipykernel | notebooks |

## 5.6 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: tensorflow` | Activate the venv; `pip install -r requirements.txt` |
| TF wheel download fails (slow network) | `pip install --resume-retries 10 tensorflow` or download the wheel manually and `pip install <wheel>.whl` |
| `Could not find a version that satisfies tensorflow` | Check `python --version` — use 3.10–3.13 |
| Training very slow | Reduce `IMG_SIZE` to 128 and `EPOCHS` to 5 in `src/config.py`; close other apps |
| Flask "model missing" | Run `python main.py train` first |
| Out-of-memory | Lower `BATCH_SIZE` in `src/config.py` |
| Predictions identical for every image | Train/inference preprocessing mismatch (e.g. double normalization — the model already contains a `Rescaling(1/255)` layer, so inputs must be [0–255] floats like training). We hit exactly this bug and fixed it in `src/preprocessing.py` |
| Train accuracy 100% but predict is constant | BatchNorm moving-stats mismatch on tiny models — compare `model(x, training=True)` vs `model.predict(x)`; see `docs/03_architecture.md` §3.4 |

---

*Previous: [4. Implementation Plan](04_implementation_plan.md) · Next: [6. Virtual Simulation](06_virtual_simulation.md)*
