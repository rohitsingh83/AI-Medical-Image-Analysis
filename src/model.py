"""
model.py
========
Model builders.

1. build_cnn()         - custom CNN (beginner-friendly, trains in minutes on CPU)
2. build_mobilenetv2() - transfer learning with MobileNetV2 (industry-style option)

The custom CNN is the default for this project: tiny, fast, fully explainable
in an interview, and accurate enough on the demo dataset.

Both models end with a single sigmoid unit:
    output < 0.5  ->  NORMAL
    output >= 0.5 ->  PNEUMONIA
(the saved model embeds its Rescaling layer, so raw [0-255] floats go in)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src import config


def _augmentation_layers() -> keras.Sequential:
    """
    On-the-fly data augmentation as model layers.

    Why layers instead of ImageDataGenerator?
    They only run during training (training=True), they are saved inside the
    .keras file, and they are the modern Keras-3 recommended approach.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),          # ~+/- 5.4 degrees
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.03, 0.03),
    ], name="augmentation")


def build_cnn(input_shape: tuple = config.INPUT_SHAPE,
              learning_rate: float = config.LEARNING_RATE,
              with_augmentation: bool = True) -> keras.Model:
    """
    Simple 4-block CNN for binary pneumonia classification.

    Architecture
    ------------
        Input (128,128,1)
        -> Augmentation + Rescaling (training only / embedded preprocessing)
        -> [Conv32 -> Conv64 -> Conv128 -> Conv128] each with MaxPool
        -> GlobalAveragePooling -> Dense128 -> Dropout0.5 -> Dense1 (sigmoid)

    Total params: ~260K - small enough for a student laptop CPU.

    Design note: we deliberately avoid BatchNormalization in this small CNN.
    BN's moving statistics are fragile with few batches per epoch and cause a
    train/inference mismatch (perfect train metrics, constant outputs at
    predict time).  Conv+Pool+Dropout is the robust, beginner-safe recipe here.
    (Transfer-learning models keep the frozen backbone's own BN layers.)
    """
    inputs = keras.Input(shape=input_shape, name="xray_image")
    x = inputs
    if with_augmentation:
        x = _augmentation_layers()(x)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

    # Final conv block - named so Grad-CAM can find it
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="pneumonia_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),   # recall matters most in medicine!
        ],
    )
    model.summary()
    return model


def build_mobilenetv2(input_shape: tuple = config.INPUT_SHAPE,
                      learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    Transfer-learning classifier: MobileNetV2 backbone pre-trained on ImageNet.

    Grayscale X-rays are repeated to 3 channels so they fit the RGB backbone.
    The backbone is frozen at first (its edge/texture filters transfer well to
    X-rays); optionally fine-tune the top layers in a second training pass.

    Use this when you want a stronger model / a "transfer learning" story
    for your resume.  Needs more RAM + time on CPU.
    """
    inputs = keras.Input(shape=input_shape, name="xray_image")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Concatenate()([x, x, x])                       # 1ch -> 3ch
    x = layers.Resizing(160, 160)(x)                          # MobileNetV2 input size

    base = keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), include_top=False, weights="imagenet")
    base.trainable = False                                    # freeze backbone
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs, outputs, name="pneumonia_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    model.summary()
    return model


def load_trained_model(model_path=None) -> keras.Model:
    """Load a saved .keras model (used by evaluate / predict / Flask)."""
    from pathlib import Path
    if model_path is None:
        model_path = config.MODELS_DIR / config.MODEL_NAME
    else:
        model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python main.py train")
    return keras.models.load_model(model_path)
