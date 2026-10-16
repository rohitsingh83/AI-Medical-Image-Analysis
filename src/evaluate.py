"""
evaluate.py
===========
Rigorous evaluation on the held-out TEST set - the metrics that go on your
README and that interviewers will grill you on.

Produces
--------
outputs/reports/metrics.json          accuracy, precision, recall, f1, roc-auc
outputs/reports/classification_report.txt
outputs/figures/confusion_matrix.png
outputs/figures/roc_curve.png

Medical-imaging note
--------------------
Accuracy alone is NOT enough in healthcare.  A missed pneumonia case (false
negative) is far worse than a false alarm.  That is why we report RECALL
(sensitivity) prominently and inspect the confusion matrix cell by cell.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src import config, data_loader, model as model_lib, visualize
from src.utils import save_json, set_global_seed


def collect_predictions(model, dataset):
    """Run the model over a tf.data.Dataset -> (y_true, y_prob)."""
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).ravel()
        y_prob.append(probs)
        y_true.append(labels.numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate_model(model_path=None) -> dict:
    """Full evaluation pipeline on the test split."""
    config.ensure_dirs()
    set_global_seed()

    # 1) Load model + test data -------------------------------------------
    model = model_lib.load_trained_model(model_path)
    _, _, test_ds = data_loader.load_datasets()

    # 2) Predictions ------------------------------------------------------
    y_true, y_prob = collect_predictions(model, test_ds)
    y_pred = (y_prob >= 0.5).astype(int)

    # 3) Metrics ----------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": config.CLASS_NAMES,
        "n_test_samples": int(len(y_true)),
        "threshold": 0.5,
    }
    save_json(metrics, config.REPORTS_DIR / "metrics.json")

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
    header = (
        "AI-Medical-Image-Analysis - CLASSIFICATION REPORT\n"
        "=" * 55 + "\n"
        f"Test samples : {len(y_true)}\n"
        f"Accuracy     : {acc*100:.2f}%\n"
        f"Precision    : {prec:.4f}\n"
        f"Recall (Sens): {rec:.4f}   <- most important in pneumonia screening\n"
        f"F1 score     : {f1:.4f}\n"
        f"ROC-AUC      : {auc:.4f}\n"
        f"Confusion Matrix:\n{cm}\n"
        + "=" * 55 + "\n\n"
    )
    report_path = config.REPORTS_DIR / "classification_report.txt"
    report_path.write_text(header + report, encoding="utf-8")

    # 4) Figures ----------------------------------------------------------
    visualize.plot_confusion_matrix(cm, config.CLASS_NAMES)
    visualize.plot_roc_curve(y_true, y_prob)

    # 5) Console summary --------------------------------------------------
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS (test set)")
    print("=" * 55)
    print(f"Accuracy      : {acc*100:.2f}%")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1 score      : {f1:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print("=" * 55)
    print(f"Saved -> {report_path}")
    print(f"Saved -> {config.REPORTS_DIR / 'metrics.json'}")
    print(f"Saved -> {config.FIGURES_DIR / 'confusion_matrix.png'}")
    print(f"Saved -> {config.FIGURES_DIR / 'roc_curve.png'}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
