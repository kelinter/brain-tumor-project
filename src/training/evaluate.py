"""
Phase 4: Load the best‐weights model, evaluate on the validation set,
and produce metrics (loss, accuracy, AUC, confusion matrix, ROC curve)
plus plots for the report.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc as compute_auc
)

# add project root for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.sys.path.insert(0, root)

from src.training.train import build_model
from src.data.data_loader import prepare_datasets

# Paths
MODEL_DIR      = os.path.join(root, "models")
LOG_DIR        = os.path.join(root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

BEST_WEIGHTS   = os.path.join(MODEL_DIR, "best_weights.h5")
HISTORY_FILE   = os.path.join(MODEL_DIR, "history.json")
REPORT_TXT     = os.path.join(LOG_DIR, "classification_report.txt")
CONF_MATRIX_IMG= os.path.join(LOG_DIR, "confusion_matrix.png")
ROC_CURVE_IMG  = os.path.join(LOG_DIR, "roc_curve.png")
LOSS_ACC_IMG   = os.path.join(LOG_DIR, "loss_accuracy.png")

# 1) Rebuild and load weights
model = build_model()
model.load_weights(BEST_WEIGHTS)

# NEEDED: compile before evaluate
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",                         # optimizer choice doesn’t matter for evaluation
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# 2) Reload datasets
#    We only need the validation split here
_, val_ds = prepare_datasets()

# 3) Evaluate built-in metrics
loss, acc, auc_metric = model.evaluate(val_ds, verbose=0)
print(f"Validation  loss={loss:.4f},  acc={acc:.4f},  AUC={auc_metric:.4f}")

# 4) Gather all predictions & truths
y_true_probs = []
y_true       = []
for x_batch, y_batch in val_ds:
    y_true.extend(y_batch.numpy().flatten())
    y_true_probs.extend(model.predict(x_batch).flatten())

y_true       = np.array(y_true, dtype=int)
y_pred_probs = np.array(y_true_probs)
y_pred       = (y_pred_probs >= 0.5).astype(int)

# 5) Classification report & confusion matrix
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["no", "yes"],
    zero_division=0
)
print(report)
with open(REPORT_TXT, "w") as f:
    f.write(report)
print(f"Classification report saved to {REPORT_TXT}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# Plot confusion matrix
plt.figure()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["no","yes"])
plt.yticks([0,1], ["no","yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
plt.tight_layout()
plt.savefig(CONF_MATRIX_IMG)
print(f"Confusion matrix image saved to {CONF_MATRIX_IMG}")

# 6) ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc      = compute_auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_CURVE_IMG)
print(f"ROC curve saved to {ROC_CURVE_IMG}")

# 7) Plot training vs. validation loss & accuracy
with open(HISTORY_FILE) as f:
    history = json.load(f)

plt.figure()
epochs = range(1, len(history["loss"])+1)
plt.plot(epochs, history["loss"],   label="train_loss")
plt.plot(epochs, history["val_loss"],   label="val_loss")
plt.plot(epochs, history["accuracy"],   label="train_acc")
plt.plot(epochs, history["val_accuracy"], label="val_acc")
plt.title("Training vs. Validation")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_ACC_IMG)
print(f"Loss/accuracy plot saved to {LOSS_ACC_IMG}")
