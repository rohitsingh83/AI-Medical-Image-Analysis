# 10. Interview Q&A — Project Explanation & Predicted Questions

---

## Tell me about this project (2-minute pitch)

> This project is an AI-powered system that analyses chest X-rays to detect pneumonia. The idea is that AI can learn from thousands of labelled medical images and then assist doctors by flagging disease signs faster and more consistently than manual review alone.
>
> The pipeline starts with medical image data — I support both a virtual simulation dataset I generate procedurally and the real public Kaggle chest X-ray dataset. Images are preprocessed — resized, contrast-enhanced with CLAHE, normalized, and augmented. A CNN is trained to classify each X-ray as NORMAL or PNEUMONIA. I evaluate it with clinically meaningful metrics — especially recall, because a missed pneumonia case is worse than a false alarm — and I visualise decisions with Grad-CAM heatmaps so doctors can see where the model looked. Finally I deployed it as a Flask web app where a clinician uploads an X-ray and instantly gets a prediction with confidence and a heatmap.
>
> In industry, the same architecture powers products from companies like Qure.ai, Annalise.ai, Siemens Healthineers, and Aidoc. The big learning for me was not just building the model — it was debugging real engineering issues like a BatchNorm train/inference mismatch, and building the whole pipeline to be reproducible and deployable.

---

## Predicted Questions & Answers

**Q1. What was the main objective?**
A: Build an end-to-end AI system that analyses chest X-rays and detects pneumonia — making diagnosis faster and giving doctors a reliable second opinion.

**Q2. Why is AI important in medical imaging?**
A: It reduces human error from fatigue, speeds up turnaround (hours → seconds), enables screening at scale in underserved areas, and catches subtle early signs that are easy to miss visually.

**Q3. Which model did you use and why?**
A: A custom CNN with Conv→Pool blocks, GlobalAveragePooling, and Dropout. CNNs are designed for image data — they learn hierarchical features (edges → textures → opacities) and are lightweight enough to train on a CPU. I also implemented a MobileNetV2 transfer-learning option to show I understand pretrained backbones.

**Q4. What data did you use?**
A: A virtual simulation dataset of synthetic chest X-ray phantoms that I generate procedurally — because real patient data can't be redistributed — structured identically to the public Kaggle "Chest X-Ray Images (Pneumonia)" dataset (Kermany et al., 2018), which the same pipeline trains on unchanged.

**Q5. How did you preprocess the images?**
A: Resize to 128×128, grayscale, CLAHE contrast enhancement to bring out faint opacities, normalization to [0,1], and augmentation — flips, small rotations, zooms, shifts — during training only. Critically, the Rescaling and augmentation live *inside* the model, so deployment can never mis-scale an image.

**Q6. How did you evaluate the model?**
A: On a held-out test set with accuracy, precision, recall (sensitivity), F1, ROC-AUC, and a confusion matrix. I highlight recall because in pneumonia screening a false negative can be fatal — we want to catch every sick patient even at the cost of some false alarms.

**Q7. How did you deploy it?**
A: A Flask web app with an upload form, a result page showing the prediction, confidence and Grad-CAM heatmap, plus a JSON API (`/api/predict`) and a health-check endpoint — the same interface pattern used by production medical imaging services.

**Q8. What is Grad-CAM and why does it matter?**
A: Gradient-weighted Class Activation Mapping — it highlights the image regions most responsible for the prediction. In healthcare, explainability is mandatory for trust: a radiologist needs to verify the model is looking at the actual lung opacity, not a spurious artifact.

**Q9. What was the hardest part?**
A: Two real debugging stories. First, a BatchNorm train/inference mismatch — training hit 100% but predictions were constant; I compared `training=True` vs `training=False` forwards, confirmed bad moving statistics at this small scale, removed BN, and retrained. Second, a deployment bug where my Flask path normalized pixels to [0,1] while the model already embedded its own Rescaling layer — double normalization silently made every upload near-black and constant. Both taught me that in ML engineering, train/serve consistency matters as much as model architecture.

**Q10. What results did you get?**
A: On the virtual dataset, high 90s accuracy (synthetic data is cleaner — I say this openly). On real Kaggle data this architecture typically reaches ~85–92%. The honest framing: the engineering pipeline is production-shaped; the numbers depend on the dataset.

**Q11. What are the limitations?**
A: Trained on synthetic/limited data, not clinically validated, single disease class, pediatric-biased public dataset, not a medical device. In interviews, stating limitations clearly builds credibility — real medical AI teams care about this more than headline accuracy.

**Q12. How would you extend this?**
A: Multi-class (COVID/TB), class-weighted loss for imbalance, DICOM input via pydicom, segmentation (U-Net) for lesion outlines, calibration analysis, Docker + cloud deployment, and experiment tracking with MLflow/W&B.

**Q13. Which companies work on this?**
A: Qure.ai (TB/COVID X-ray AI), Annalise.ai (multi-findings), Aidoc (triage), Rad AI (report drafting), Enlitic, Harrison.ai, Arterys (now Tempus), Siemens Healthineers, GE Healthcare, Philips, Canon Medical, Google Health/Medicine.

**Q14. What metrics would a hospital care about?**
A: Sensitivity (recall) first — missed cases are dangerous; then specificity, PPV/NPV at real prevalence, AUC, calibration, inference latency, and workflow integration. Accuracy alone is meaningless in screening.

---

*Previous: [9. 7-Day Proof Plan](09_7day_proof_plan.md) · Next: [11. Complete Guide (A–M)](11_complete_guide.md)*
