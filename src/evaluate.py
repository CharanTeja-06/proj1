"""
Comprehensive evaluation: overall accuracy, per-letter accuracy,
confusion matrix saved as PNG, and a classification report.

Run from the `src/` directory:
    python evaluate.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "mpl")

import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import make_dataset, CLASSES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print("Building test dataset...")
test_ds = make_dataset("Testing", training=False, batch_size=64)

# ─── Collect all predictions ─────────────────────────────────────────────────
print("Evaluating on test set...")
start_time = time.time()
y_true, y_pred = [], []
batch_count = 0

for images, labels in test_ds:
    # Direct model invocation avoids per-batch predict() graph rebuild overhead
    outputs = model(images, training=False)
    preds = tf.argmax(outputs, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds.numpy().tolist())
    batch_count += 1
    if batch_count % 10 == 0:
        print(f"  Processed {len(y_true)} test samples...")

elapsed = time.time() - start_time
print(f"Inference completed for {len(y_true)} samples in {elapsed:.1f}s ({len(y_true)/elapsed:.1f} samples/sec).")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ─── Overall accuracy ────────────────────────────────────────────────────────
overall = accuracy_score(y_true, y_pred)
print(f"\n==========================================")
print(f"   OVERALL TEST ACCURACY: {overall * 100:.2f}%")
print(f"==========================================")

# ─── Per-letter accuracy ─────────────────────────────────────────────────────
print("\nPer-Letter Accuracy Breakdown:")
per_letter = {}
for i, letter in enumerate(CLASSES):
    mask = (y_true == i)
    if mask.sum() == 0:
        continue
    correct = (y_pred[mask] == i).sum()
    total = mask.sum()
    acc = correct / total
    per_letter[letter] = acc
    bar = "#" * int(acc * 20) + "-" * (20 - int(acc * 20))
    flag = "  <-- LOW ACCURACY" if acc < 0.75 else ""
    if letter in ("S", "Z"):
        flag += "  [CRITICAL EVALUATION CASE]"
    print(f"  {letter}: [{bar}] {acc * 100:5.1f}% ({correct:3d}/{total:3d}){flag}")

# ─── S and Z Focus Analysis ──────────────────────────────────────────────────
print("\nFocus Analysis for Letters S & Z:")
for char in ["S", "Z"]:
    idx = CLASSES.index(char)
    mask = (y_true == idx)
    total = mask.sum()
    correct = (y_pred[mask] == idx).sum()
    acc = correct / total if total > 0 else 0
    # Misclassifications
    wrong_preds = y_pred[mask & (y_pred != idx)]
    print(f"  Letter '{char}': Accuracy = {acc*100:.2f}% ({correct}/{total})")
    if len(wrong_preds) > 0:
        vals, counts = np.unique(wrong_preds, return_counts=True)
        top_errs = sorted(zip(vals, counts), key=lambda x: -x[1])[:3]
        err_str = ", ".join(f"'{CLASSES[k]}' ({c} times)" for k, c in top_errs)
        print(f"    Most common misclassifications: {err_str}")

# ─── Classification report ───────────────────────────────────────────────────
report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
print(f"\nClassification Report:\n{report}")
report_path = REPORTS_DIR / "classification_report.txt"
report_path.write_text(report, encoding="utf-8")
print(f"Saved classification report to: {report_path}")

# ─── Confusion matrix using pure Matplotlib (no pandas/seaborn DLL issue) ─────
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(16, 14), dpi=120)
cax = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, fontsize=10)
ax.set_yticklabels(CLASSES, fontsize=10)
ax.set_xlabel("Predicted Letter", fontsize=12, fontweight="bold")
ax.set_ylabel("True Letter", fontsize=12, fontweight="bold")
ax.set_title(f"ISL Alphabet Normalised Confusion Matrix (Overall Acc = {overall*100:.2f}%)",
             fontsize=14, fontweight="bold", pad=12)

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        val = cm_norm[i, j]
        if val >= 0.01:
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

plt.tight_layout()
cm_path = REPORTS_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=120)
plt.close(fig)
print(f"Saved confusion matrix image to: {cm_path}")
