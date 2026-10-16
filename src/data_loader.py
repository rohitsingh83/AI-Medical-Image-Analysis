"""
data_loader.py
==============
Turn the on-disk folder structure into ready-to-train TensorFlow datasets.

Folder contract (identical to the Kaggle chest X-ray dataset):
    <DATA_DIR>/train/NORMAL/*.png
    <DATA_DIR>/train/PNEUMONIA/*.png
    <DATA_DIR>/val/...      (optional - else we split from train)
    <DATA_DIR>/test/...

Returns
-------
train_ds, val_ds, test_ds : tf.data.Dataset of (image, label)
                            image: float32 (H, W, 1) in [0, 255]
                            label: float32 scalar, 0.0 = NORMAL, 1.0 = PNEUMONIA

Note: pixel scaling to [0,1] happens INSIDE the model (Rescaling layer), so the
saved model file embeds its own preprocessing - deployment cannot get it wrong.
"""

from pathlib import Path

import tensorflow as tf

from src import config


def _prepare(ds: tf.data.Dataset, shuffle: bool = False) -> tf.data.Dataset:
    """Cast labels to float32 (binary sigmoid target) and prefetch for speed."""
    ds = ds.map(lambda x, y: (x, tf.cast(y, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=config.SEED, reshuffle_each_iteration=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_datasets(data_dir: Path | str = config.DATA_DIR,
                  img_size: int = config.IMG_SIZE,
                  batch_size: int = config.BATCH_SIZE):
    """
    Build train / val / test datasets from `data_dir`.

    If a `val/` folder exists we use it; otherwise we carve a validation
    split out of `train/` (needed for the real Kaggle dataset).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}.\n"
            f"Run  python -m src.simulate_data  to create the virtual dataset,\n"
            f"or download the real Kaggle dataset into data/raw/chest_xray/"
        )

    img_size_tuple = (img_size, img_size)
    common = dict(image_size=img_size_tuple, batch_size=batch_size,
                  color_mode="grayscale", label_mode="int", seed=config.SEED)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    raw_train = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True,
        validation_split=config.VAL_SPLIT if not val_dir.exists() else None,
        subset="training" if not val_dir.exists() else None, **common)

    if val_dir.exists():
        raw_val = tf.keras.utils.image_dataset_from_directory(val_dir, shuffle=False, **common)
    else:
        raw_val = tf.keras.utils.image_dataset_from_directory(
            train_dir, shuffle=False, validation_split=config.VAL_SPLIT,
            subset="validation", **common)

    raw_test = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **common)

    train_ds = _prepare(raw_train, shuffle=True)
    val_ds = _prepare(raw_val)
    test_ds = _prepare(raw_test)

    print(f"[data_loader] Classes        : {raw_train.class_names}")
    print(f"[data_loader] Train batches  : {len(train_ds)}")
    print(f"[data_loader] Val batches    : {len(val_ds)}")
    print(f"[data_loader] Test batches   : {len(test_ds)}")
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    load_datasets()
