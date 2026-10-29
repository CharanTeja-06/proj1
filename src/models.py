"""
MobileNetV2 Model Definition and Fine-Tuning Utilities for ISL Alphabet Recognition.
"""

import tensorflow as tf
from data import NUM_CLASSES

IMG_SIZE = 224


def build_model(fine_tune_from: int = 100):
    """
    Constructs MobileNetV2 with an improved classification head:
    - Pre-trained MobileNetV2 base (initially frozen)
    - GlobalAveragePooling2D
    - BatchNormalization for feature scaling stability
    - Dense(512, relu) + Dropout(0.4) for high-capacity non-linear feature separation
    - Dense(256, relu) + Dropout(0.3) for fine-grained disambiguation (e.g., S vs Z)
    - Dense(26, softmax) final classification output
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # Frozen during phase 1

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="hand_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="isl_alphabet_output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ISL_MobileNetV2")
    return model


def unfreeze_model(model, fine_tune_from: int = 100):
    """
    Unfreezes MobileNetV2 base layers from index `fine_tune_from` to the end.
    Allows domain-specific fine-tuning on subtle hand gesture contours (e.g. S vs Z).
    """
    base = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower() or (hasattr(layer, "layers") and len(layer.layers) > 10):
            base = layer
            break

    if base is None:
        base = model.layers[1]

    base.trainable = True
    total_base_layers = len(base.layers)
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"[unfreeze_model] Fine-tuning enabled from layer {fine_tune_from}/{total_base_layers} "
          f"({trainable_count} base layers trainable).")
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
