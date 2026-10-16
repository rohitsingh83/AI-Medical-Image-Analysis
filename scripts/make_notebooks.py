"""
Generate the 3 project notebooks (nbformat v4 JSON) without requiring jupyter.
Run:  python scripts/make_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def cell_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def cell_code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
nb1 = notebook([
    cell_md("""# 01 - Data Exploration
## AI-Powered Medical Image Analysis

This notebook explores the chest X-ray dataset: sample images, class balance,
and the preprocessing pipeline.

**Run order:** from the project root, first generate data if needed:
`python main.py simulate`"""),
    cell_code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # make `src` importable

from src import config, visualize
from src.preprocessing import resize, apply_clahe, normalize

config.ensure_dirs()
print("Data dir :", config.DATA_DIR)
print("Classes  :", config.CLASS_NAMES)"""),
    cell_md("## 1. Sample chest X-rays (virtual simulation dataset)"),
    cell_code("""from IPython.display import Image, display

path = visualize.plot_sample_grid()
display(Image(filename=str(path)))"""),
    cell_md("## 2. Class distribution across splits"),
    cell_code("""path = visualize.plot_class_distribution()
display(Image(filename=str(path)))"""),
    cell_md("## 3. Preprocessing pipeline: original -> resize -> CLAHE -> normalize"),
    cell_code("""path = visualize.plot_preprocessing_demo()
display(Image(filename=str(path)))"""),
    cell_md("""## What we see
- **NORMAL** images: clear dark lung fields, visible ribs/spine/heart shadow
- **PNEUMONIA** images: cloudy bright opacities inside the lung fields (consolidation)
- Augmentation-style variation (rotation, shift, brightness) mimics real acquisition differences

Next: `02_model_training.ipynb`"""),
])

# ---------------------------------------------------------------------------
nb2 = notebook([
    cell_md("""# 02 - Model Training
## CNN for Pneumonia Detection (TensorFlow / Keras)

Builds the CNN, trains it on the X-ray dataset, and plots the training curves.

> Tip: for a quick demo use `EPOCHS = 5`. For the real Kaggle dataset use 15-25
> and consider training on Google Colab with a GPU."""),
    cell_code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from src import config
from src.utils import set_global_seed
config.ensure_dirs()
set_global_seed()"""),
    cell_md("## 1. Load datasets"),
    cell_code("""from src import data_loader

train_ds, val_ds, test_ds = data_loader.load_datasets()"""),
    cell_md("## 2. Build the CNN"),
    cell_code("""from src import model as model_lib

model = model_lib.build_cnn()
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)"""),
    cell_md("## 3. Train (equivalent to `python main.py train`)"),
    cell_code("""from src.train import train_model

history = train_model(model_type="cnn", epochs=8)"""),
    cell_md("## 4. Training curves"),
    cell_code("""from IPython.display import Image, display
from src import visualize

path = visualize.plot_training_curves(history)
display(Image(filename=str(path)))"""),
    cell_md("""## Notes
- The model embeds its own preprocessing (`Rescaling` + augmentation layers),
  so the saved `.keras` file cannot be mis-scaled at deployment time.
- `ModelCheckpoint` keeps the best `val_accuracy` weights.
- We deliberately avoid BatchNormalization in this small CNN (train/inference
  mismatch risk) - see `docs/03_architecture.md`.

Next: `03_evaluation_and_deployment.ipynb`"""),
])

# ---------------------------------------------------------------------------
nb3 = notebook([
    cell_md("""# 03 - Evaluation, Explainability & Deployment
Clinical metrics, confusion matrix, ROC curve, Grad-CAM, and the Flask API snippet."""),
    cell_code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from src import config
config.ensure_dirs()"""),
    cell_md("## 1. Full evaluation on the test set"),
    cell_code("""from src.evaluate import evaluate_model

metrics = evaluate_model()
metrics"""),
    cell_md("## 2. Confusion matrix & ROC curve"),
    cell_code("""from IPython.display import Image, display

display(Image(filename=str(config.FIGURES_DIR / "confusion_matrix.png")))
display(Image(filename=str(config.FIGURES_DIR / "roc_curve.png")))"""),
    cell_md("## 3. Prediction + Grad-CAM on one image"),
    cell_code("""from src.predict import predict_image
from IPython.display import Image, display

sample = sorted((config.DATA_DIR / "test" / "PNEUMONIA").glob("*.png"))[0]
result = predict_image(sample)
print(result)
display(Image(filename=result["gradcam_path"]))"""),
    cell_md("## 4. Balanced prediction grid"),
    cell_code("""from src.predict import predict_sample_grid

records = predict_sample_grid(n=8)
display(Image(filename=str(config.FIGURES_DIR / "prediction_grid.png")))"""),
    cell_md("""## 5. Flask deployment (run in a terminal)

```bash
python app.py
# open http://127.0.0.1:5000
```

JSON API from Python:"""),
    cell_code("""# Uncomment to call the running Flask API:
# import requests
# r = requests.post("http://127.0.0.1:5000/api/predict",
#                   files={"image": open(sample, "rb")})
# print(r.json())"""),
    cell_md("""## Interpretation checklist
- Accuracy alone is NOT enough in medicine - read **recall (sensitivity)** first.
- Inspect the confusion matrix: where are the false negatives?
- Grad-CAM should light up the **lung opacities**, not image corners.
- On real clinical data expect lower numbers than this clean synthetic demo -
  state that honestly in interviews."""),
])

for name, nb in [("01_data_exploration.ipynb", nb1),
                 ("02_model_training.ipynb", nb2),
                 ("03_evaluation_and_deployment.ipynb", nb3)]:
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1))
    print("wrote", path)
