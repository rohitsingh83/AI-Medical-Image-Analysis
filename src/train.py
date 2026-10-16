"""
train.py
========
Model training pipeline.

Responsibilities
----------------
* load datasets
* build the CNN
* train with validation + checkpoints + early stopping
* save: models/pneumonia_cnn.keras, outputs/reports/history.json,
        outputs/reports/training_log.txt
* draw: outputs/figures/training_curves.png
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src import config, data_loader, model as model_lib, visualize
from src.utils import save_json, set_global_seed


def train_model(model_type: str = "cnn", epochs: int = config.EPOCHS) -> dict:
    """
    Train the classifier end-to-end.

    Parameters
    ----------
    model_type : "cnn" (default) or "mobilenetv2" (transfer learning)
    epochs     : number of training epochs

    Returns
    -------
    history : dict of training curves (loss/accuracy/...)
    """
    config.ensure_dirs()
    set_global_seed()

    # 1) Data ---------------------------------------------------------------
    train_ds, val_ds, _ = data_loader.load_datasets()

    # 2) Model --------------------------------------------------------------
    if model_type == "mobilenetv2":
        model = model_lib.build_mobilenetv2()
        model_path = config.MODELS_DIR / config.MODEL_NAME_MOBILENET
    else:
        model = model_lib.build_cnn()
        model_path = config.MODELS_DIR / config.MODEL_NAME

    # 3) Callbacks ---------------------------------------------------------
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    # 4) Train -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"TRAINING STARTED | model={model_type} | epochs={epochs}")
    print("=" * 60 + "\n")
    start = time.time()
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start

    history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}

    # 5) Persist artefacts -------------------------------------------------
    model.save(model_path)          # final weights (best checkpoint already saved)
    save_json(history, config.REPORTS_DIR / config.HISTORY_FILE)

    best_val_acc = max(history.get("val_accuracy", [0]))
    log_lines = [
        "AI-Medical-Image-Analysis - TRAINING LOG",
        "=" * 50,
        f"Model type      : {model_type}",
        f"Image size      : {config.IMG_SIZE}x{config.IMG_SIZE}",
        f"Batch size      : {config.BATCH_SIZE}",
        f"Epochs run      : {len(history['loss'])}",
        f"Training time   : {elapsed:.1f} s ({elapsed/60:.1f} min)",
        f"Best val acc    : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)",
        f"Final train acc : {history['accuracy'][-1]:.4f}",
        f"Final val acc   : {history['val_accuracy'][-1]:.4f}",
        f"Final val loss  : {history['val_loss'][-1]:.4f}",
        "",
        "Per-epoch metrics:",
        "-" * 50,
    ]
    for i in range(len(history["loss"])):
        log_lines.append(
            f"Epoch {i+1:02d} | "
            f"loss={history['loss'][i]:.4f} acc={history['accuracy'][i]:.4f} "
            f"precision={history.get('precision', [0]*99)[i]:.4f} "
            f"recall={history.get('recall', [0]*99)[i]:.4f} | "
            f"val_loss={history['val_loss'][i]:.4f} val_acc={history['val_accuracy'][i]:.4f}"
        )
    log_path = config.REPORTS_DIR / config.TRAIN_LOG_FILE
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"[train] Training log -> {log_path}")

    # 6) Curves figure -----------------------------------------------------
    visualize.plot_training_curves(history)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE | best val accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to   : {model_path}")
    print("=" * 60)
    return history


if __name__ == "__main__":
    train_model()
