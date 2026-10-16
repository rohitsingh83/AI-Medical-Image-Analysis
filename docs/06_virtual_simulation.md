# 6. Virtual Simulation Implementation

---

## 6.1 Why Virtual Simulation?

1. **Legal/ethical** — real patient images (DICOM, hospital archives) cannot be redistributed on GitHub; datasets like Kaggle's still require an account + license acceptance.
2. **Practical** — a student laptop should run the full pipeline in minutes, not after a 2 GB download.
3. **Industry-paralleled** — medical device companies validate on phantoms and simulated data before clinical trials; imaging research routinely uses digital phantoms (XCAT, simulated CT/MRI noise models).

The virtual dataset is a **faithful stand-in for the engineering workflow** — and the identical code trains on real public data (one config line changes).

## 6.2 How the Dataset Simulates a Real Medical Environment

| Real hospital pipeline | This project's simulation |
|---|---|
| PACS archive with study folders | `train/ val/ test/` split folders |
| Radiologist labels each study | Folder name = ground-truth label (NORMAL/PNEUMONIA) |
| X-ray acquisition variation (positioning, exposure) | Random rotation, translation, brightness jitter per image |
| Pneumonia appears as cloudy consolidations in lung fields | Gaussian opacity blobs drawn inside lung regions, blurred |
| Anatomy: torso, lungs, ribs, spine, heart shadow | Procedural phantom: elliptical torso, dark lung ellipses, rib arcs, spine column, heart shadow, diaphragm domes |

## 6.3 How Images Represent Real Diagnoses

- **NORMAL**: dark, clear lung fields (air is radiolucent), visible rib cage, mediastinum/heart shadow — the classic healthy chest radiograph pattern.
- **PNEUMONIA**: same anatomy + bright cloudy opacities (consolidation) biased toward the lower lung zones — pneumonia classically presents as lower-lobe airspace opacities.
- Noise, blur and acquisition-style augmentation make the classification non-trivial — the model must look at *texture inside the lungs*, not just global brightness.

## 6.4 How Model Predictions Simulate Doctor Assistance

- The model outputs **NORMAL / PNEUMONIA + probability** — exactly the "second opinion" interface real CAD products use.
- The **Grad-CAM heatmap** simulates pointing at the region of interest: "the model is flagging this area of the right lower lobe."
- The **Flask app** simulates the clinic workflow: upload study → get prioritised result.
- **We never claim to diagnose patients** — the system *assists*, mirroring real regulatory language ("adjunctive" / "triage" tools).

## 6.5 Step-by-Step Workflow

```
Step 1  Run:  python main.py simulate
        → generates 1,160 synthetic X-rays (train 800 / val 160 / test 200)

Step 2  Run:  python main.py explore
        → samples_grid.png, class_distribution.png, preprocessing_demo.png

Step 3  Run:  python main.py train
        → CNN learns lung patterns; checkpoints best weights

Step 4  Run:  python main.py evaluate
        → clinical metrics on held-out test set

Step 5  Run:  python main.py predict
        → prediction grid + Grad-CAM heatmaps

Step 6  Run:  python app.py
        → web demo: upload → prediction → heatmap (screenshots for README)

Step 7  Capture proof artifacts (list below) and push to GitHub
```

## 6.6 Outputs to Generate / Graphs to Show

| Output | File | Purpose |
|---|---|---|
| Sample X-rays | `outputs/figures/samples_grid.png` | dataset realism |
| Class distribution | `outputs/figures/class_distribution.png` | data hygiene |
| Preprocessing steps | `outputs/figures/preprocessing_demo.png` | pipeline visibility |
| Training curves | `outputs/figures/training_curves.png` | learning happened |
| Confusion matrix | `outputs/figures/confusion_matrix.png` | error analysis |
| ROC curve | `outputs/figures/roc_curve.png` | discrimination ability |
| Predictions + heatmaps | `outputs/figures/prediction_grid.png` | explainability |
| Training log | `outputs/reports/training_log.txt` | reproducibility |
| Metrics | `outputs/reports/metrics.json` | machine-readable results |
| Web app result page | `images/screenshot_*.png` | deployment proof |

## 6.7 Proof Students Should Capture (checklist)

- [ ] Terminal screenshot of `python main.py demo` completing
- [ ] Training log with rising accuracy
- [ ] All 7 figures
- [ ] Grad-CAM output on a single prediction
- [ ] Flask home page + result page (before/after upload)
- [ ] JSON API response (`curl -F ... /api/predict`)
- [ ] GitHub repo home with README figures
- [ ] Optional: 2-minute screen recording (upload → result) for LinkedIn

---

*Previous: [5. Installation](05_installation.md) · Next: [7. How to Run & Expected Outputs](07_run_and_outputs.md)*
