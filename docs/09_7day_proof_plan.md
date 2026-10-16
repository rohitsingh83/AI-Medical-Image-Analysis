# 9. Step-by-Step GitHub Proof Plan (7 Days)

Goal: a **credible commit history** that shows real engineering work over a week — because interviewers check the commit graph, not just the final code.

---

## Day 1 — Environment + Project Skeleton
**Do:**
- Create the GitHub repo (name, description, topics, MIT license)
- Set up venv + `requirements.txt`
- Push folder structure + `config.py` + `.gitignore`

**Commits:**
```
chore: initialise repository with project structure
chore: add requirements.txt and .gitignore
feat: add central configuration module
```
**Proof:** repo home screenshot; `git log` shows 3 commits.

## Day 2 — Dataset
**Do:**
- Implement `src/simulate_data.py` (or download Kaggle data)
- Generate the virtual dataset
- Exploration notebook/figures

**Commits:**
```
feat: add virtual X-ray phantom generator
feat: generate virtual dataset with train/val/test splits
feat: add dataset exploration figures
```
**Proof:** `samples_grid.png`, `class_distribution.png` committed.

## Day 3 — Preprocessing
**Do:**
- `src/preprocessing.py` (resize, CLAHE, normalize)
- `src/data_loader.py` (tf.data pipelines)
- `preprocessing_demo.png`

**Commits:**
```
feat: add preprocessing pipeline with CLAHE enhancement
feat: add tf.data loaders with train/val/test contract
docs: document preprocessing workflow
```
**Proof:** `preprocessing_demo.png` + console output of dataset shapes.

## Day 4 — Model
**Do:**
- `src/model.py` (CNN + MobileNetV2)
- Unit sanity check (dummy predict in [0,1])

**Commits:**
```
feat: add CNN classifier with in-model augmentation
feat: add MobileNetV2 transfer learning option
```
**Proof:** `model.summary()` screenshot in docs.

## Day 5 — Training
**Do:**
- `src/train.py` + callbacks
- Run training; save model + logs + curves

**Commits:**
```
feat: add training pipeline with checkpoints and early stopping
feat: train pneumonia CNN - best val accuracy XX.X%
```
**Proof:** `training_log.txt`, `training_curves.png`, `history.json`.

## Day 6 — Evaluation + Prediction + Explainability
**Do:**
- `src/evaluate.py`, `src/predict.py`, `src/visualize.py`
- Generate all evaluation figures + Grad-CAM grid

**Commits:**
```
feat: add evaluation module with clinical metrics
feat: add Grad-CAM explainability for predictions
feat: add visualization module - confusion matrix, ROC, prediction grid
```
**Proof:** `confusion_matrix.png`, `roc_curve.png`, `prediction_grid.png`, `metrics.json`.

## Day 7 — Deployment + README + Publish
**Do:**
- Flask `app.py` + templates; screenshot the UI
- Final README (results tables, figures, install/usage)
- Tag release `v1.0` with model weights as release asset

**Commits:**
```
feat: add Flask web application with upload and JSON API
docs: complete README with results and demo screenshots
chore: release v1.0 with trained model weights
```
**Proof:** app screenshots, `/api/predict` curl output, release page, final repo view.

---

## Ongoing Proof Habits
- Never rewrite history — mistakes + fixes (like our BatchNorm debugging) are *good* signals
- Keep a `notes.md` or PR-style descriptions of what you learned each day
- Screenshot everything as you go (Day 7 is too late to reconstruct proof)

---

*Previous: [8. GitHub Strategy](08_github_strategy.md) · Next: [10. Interview Q&A](10_interview_qa.md)*
