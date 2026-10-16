# 8. GitHub Upload Strategy

---

## 8.1 Repository Creation

**Best repo name:** `AI-Medical-Image-Analysis` (clear, searchable, keyword-rich)

Alternatives: `pneumonia-detection-cnn`, `medical-image-analysis-ai`, `chest-xray-ai-deep-learning`

**Description (copy-paste):**
> AI-powered chest X-ray analysis system that detects pneumonia using a CNN (TensorFlow/Keras) with Grad-CAM explainability, clinical-grade evaluation, and a Flask web app. Full ML pipeline with virtual simulation + real Kaggle dataset support.

**Topics/tags (add all):**
`machine-learning` `deep-learning` `computer-vision` `medical-imaging` `healthcare-ai` `tensorflow` `keras` `cnn` `pneumonia-detection` `chest-xray` `flask` `python` `grad-cam` `data-science` `portfolio-project`

**Settings:**
- Public repository
- Add README, `.gitignore` (already provided), MIT License
- Add repo topics, description, and a screenshot in the README (social preview)
- Pin the repo on your GitHub profile

## 8.2 Git Commands (first push)

```bash
cd AI-Medical-Image-Analysis
git init
git add .
git commit -m "feat: initial project structure - config, data pipeline, model modules"
git branch -M main
git remote add origin https://github.com/<your-username>/AI-Medical-Image-Analysis.git
git push -u origin main
```

## 8.3 Commit Strategy (the professional habit)

**Golden rule: one logical change per commit, present-tense message, why + what.**

```
feat: add virtual X-ray dataset generator
feat: implement preprocessing pipeline with CLAHE
feat: add CNN model builder with augmentation layers
feat: add training pipeline with checkpoints and logging
feat: add evaluation - confusion matrix, ROC, classification report
feat: add Grad-CAM prediction and visualization modules
feat: add Flask web application with upload and JSON API
docs: add README with architecture, results and demo screenshots
fix: remove BatchNormalization to fix train/inference mismatch
chore: add .gitignore and requirements.txt
```

What NOT to do:
- ❌ `final code`, `update`, `asdf`, `commit1`
- ❌ one giant commit of everything
- ❌ committing `data/`, `*.keras`, `venv/` (the .gitignore handles this)

## 8.4 Images / Videos / Large Files

- **Figures** (`outputs/figures/*.png`, `images/*.png`): commit directly — README needs them.
- **Model weights** (`*.keras`, `*.h5`): GitHub Releases → attach `pneumonia_cnn.keras` as a release asset (v1.0). Mention the link in the README.
- **Dataset**: never commit. Document the download/generate command instead.
- **Demo video**: record a 1–2 min screen capture (OBS Studio), upload to YouTube/LinkedIn, embed the link in the README. Optionally add a compressed GIF (< 10 MB) to `images/`.

## 8.5 Making the Repo Professional (checklist)

- [ ] Polished README with badges, table of contents, architecture diagram, figures
- [ ] `docs/` with deep-dive material
- [ ] `requirements.txt` + exact run commands that work on a fresh clone
- [ ] `.gitignore` clean (no `__pycache__`, no data, no venv)
- [ ] LICENSE (MIT)
- [ ] Consistent code style + docstrings
- [ ] Honest results section (including limitations & disclaimer)
- [ ] Release with model weights attached
- [ ] Screenshots/GIF of the web app

## 8.6 After Pushing — Amplify

1. LinkedIn post: "Built an AI system that analyses chest X-rays..." + repo link + 3 learnings
2. Add the project to your resume under *Projects* with 2–3 bullet outcomes
3. Pin the repo; add it to your GitHub profile README
4. Prepare the interview story (see `docs/10_interview_qa.md`)

---

*Previous: [7. Run & Outputs](07_run_and_outputs.md) · Next: [9. 7-Day Proof Plan](09_7day_proof_plan.md)*
