"""
Two-phase training script for improved ISL recognition.

Phase 1 — Feature extraction (base frozen, head trained)
Phase 2 — Fine-tuning (top layers of MobileNetV2 unfrozen, lower LR)

Run from the `src/` directory:
    python train.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from pathlib import Path
from data import make_dataset
from models import build_model, unfreeze_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "isl_model.keras"

tf.keras.utils.set_random_seed(42)

# ─── Datasets ────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = make_dataset("Training", training=True)
val_ds   = make_dataset("Validation", training=False)

# ─── Phase 1: Train head with frozen base ────────────────────────────────────
print("\n=== Phase 1: Training classification head (base frozen) ===")
model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

phase1_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,           # was 2 — more patience to find real plateau
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,           # more aggressive reduction
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,                # allow more epochs; early stopping handles it
    callbacks=phase1_callbacks
)

# ─── Phase 2: Fine-tune top MobileNetV2 layers ───────────────────────────────
print("\n=== Phase 2: Fine-tuning top MobileNetV2 layers ===")
model = unfreeze_model(model, fine_tune_from=100)

# Lower LR is critical for fine-tuning — avoids destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

phase2_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=phase2_callbacks
)

model.save(MODEL_PATH)
print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
