# 4. Full Implementation Plan — 10 Phases

Each phase lists: **What → Why → Expected output → Common mistakes → How to verify**.

---

## Phase 1 — Setup
**What:** Install Python 3.10+, create a virtual environment, `pip install -r requirements.txt`, clone the repo.
**Why:** Reproducible environment; avoids "works on my machine" bugs.
**Expected output:** `python -c "import tensorflow"` runs without error.
**Common mistakes:** Installing packages globally; Python 3.13 with very old TF versions; forgetting to activate the venv.
**Verify:** `python main.py` prints the help menu.

## Phase 2 — Dataset Collection
**What:** Run `python main.py simulate` (virtual dataset) **or** download the Kaggle Chest X-Ray dataset into `data/raw/chest_xray/`.
**Why:** Models learn from labelled images; the folder contract mirrors real hospital archives.
**Expected output:** `data/synthetic/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}` filled with PNGs + `samples_grid.png`.
**Common mistakes:** Wrong folder nesting (labels must be subfolders of split folders); mixing train/test images; class imbalance ignored on real data.
**Verify:** `python main.py explore` → check class counts in `class_distribution.png`.

## Phase 3 — Image Preprocessing
**What:** Implement resize, grayscale, CLAHE, normalization, augmentation (`src/preprocessing.py` + augmentation layers).
**Why:** Neural nets need fixed-size, normalised inputs; augmentation fights overfitting; CLAHE enhances faint opacities.
**Expected output:** `preprocessing_demo.png` showing the 4-step pipeline.
**Common mistakes:** Forgetting to normalize at inference; augmenting the validation set; resizing with wrong interpolation.
**Verify:** Print `batch.shape` = (32, 128, 128, 1) and pixel range [0, 255] before the model's Rescaling layer.

## Phase 4 — Model Building
**What:** Build the CNN in `src/model.py` (Conv blocks → GAP → Dense → sigmoid). Optionally `build_mobilenetv2()`.
**Why:** CNNs are the standard architecture for image classification; transfer learning boosts performance on small datasets.
**Expected output:** `model.summary()` showing ~257K parameters.
**Common mistakes:** Using softmax+2 outputs with binary_crossentropy; wrong input shape; BatchNorm misuse in tiny models (see architecture doc).
**Verify:** `model.output_shape == (None, 1)` and a dummy predict returns a value in [0, 1].

## Phase 5 — Training
**What:** `python main.py train --epochs 8` — fit with validation, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
**Why:** The model learns patterns; checkpoints keep the best weights; callbacks prevent overfitting.
**Expected output:** `models/pneumonia_cnn.keras`, `history.json`, `training_log.txt`, `training_curves.png`.
**Common mistakes:** Not shuffling train data; shuffling val data; training too many epochs (overfit); monitoring train loss instead of val metrics.
**Verify:** `val_accuracy` rises across epochs and `training_curves.png` shows train/val moving together.

## Phase 6 — Evaluation
**What:** `python main.py evaluate` — accuracy, precision, recall, F1, ROC-AUC, confusion matrix on the untouched test set.
**Why:** Honest performance measurement; medical AI lives or dies on false negatives.
**Expected output:** `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`.
**Common mistakes:** Evaluating on training data; reporting accuracy only; ignoring class imbalance; tuning the threshold without saying so.
**Verify:** Check the confusion matrix cells manually — where are the false negatives?

## Phase 7 — Prediction
**What:** `python main.py predict --image <path>` for one image; `python main.py predict` for the balanced test grid.
**Why:** This is the product moment — what the doctor actually sees.
**Expected output:** Console result + `prediction_grid.png` + Grad-CAM overlays in `outputs/predictions/`.
**Common mistakes:** Different preprocessing at inference than training; loading the wrong model file.
**Verify:** Known test images classify correctly and Grad-CAM highlights lung regions (not image corners!).

## Phase 8 — Visualization
**What:** Generate all figures (`src/visualize.py` produces them automatically at each stage).
**Why:** Humans trust what they can see; README with charts gets interviewers interested.
**Expected output:** 7 figures in `outputs/figures/`.
**Common mistakes:** Non-reproducible random samples (fix the seed); unreadable plots; missing axis labels.
**Verify:** Every figure is legible at GitHub preview size and referenced in the README.

## Phase 9 — GitHub Publishing
**What:** Create the repo, push in meaningful commits, polish the README (see docs 8–9).
**Why:** Proof of work compounds — commit history shows the *process*, not just the result.
**Expected output:** Professional public repo with README badges, figures, docs, tags.
**Common mistakes:** One giant "final code" commit; committing datasets/weights; empty README; no license.
**Verify:** Fresh clone → `pip install -r requirements.txt` → `python main.py demo` works.

## Phase 10 — Final Output
**What:** Demo video/GIF of the Flask app, final metrics table, shareable LinkedIn post, interview story prepared (docs 10–11).
**Why:** The project must be *presentable*, not just functional.
**Expected output:** Repo ready to link on your resume; demo assets in `images/`.
**Common mistakes:** Over-claiming ("AI that diagnoses patients!"); no disclaimer; untested README commands.
**Verify:** Follow the README exactly as a stranger would — every command must work.

---

*Previous: [3. Architecture](03_architecture.md) · Next: [5. Installation](05_installation.md)*
