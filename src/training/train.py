"""
Phase 3: Define and train a transfer-learning–based CNN classifier for brain tumor detection.
This script:
 1. Loads train/validation datasets via prepare_datasets()
 2. Builds a pretrained backbone (MobileNetV2)
 3. Adds classification head (GlobalAveragePooling, Dropout, Dense)
 4. Compiles and fits the model, saving the best weights
"""
import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import json



# get project root (two levels up from src/training)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

# Import our data loader
from src.data.data_loader import prepare_datasets

# Build the model
def build_model(image_size=(224,224), dropout_rate=0.5):
    """
    Constructs and returns a MobileNetV2-based model:
      - frozen ImageNet backbone
      - GlobalAveragePooling
      - Dropout
      - single-unit Dense(sigmoid) head
    """
    base = MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet"
    )
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(dropout_rate, name="dropout")(x)
    output = Dense(1, activation="sigmoid", name="classifier")(x)

    return Model(inputs=base.input, outputs=output)


# Configuration
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 10
PATIENCE    = 3

# Paths to save model artifacts
ROOT_DIR  = root
MODEL_DIR = os.path.join(ROOT_DIR, "models")
CHECKPOINT = os.path.join(MODEL_DIR, "best_weights.h5")
SUMMARY_FILE = os.path.join(MODEL_DIR, "model_summary.txt")
HISTORY_FILE = os.path.join(MODEL_DIR, "history.json")
FINAL_MODEL = os.path.join(MODEL_DIR, "final_model.h5")


# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load datasets
train_ds, val_ds = prepare_datasets()


# 2) Build the model
model = build_model(image_size=IMAGE_SIZE, dropout_rate=0.5)

# Print & save model summary
print("\n" + "="*40 + " MODEL SUMMARY " + "="*40 + "\n")
model.summary()
print("\n" + "="*100 + "\n")

with open(SUMMARY_FILE, "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))
print(f"Model summary saved to {SUMMARY_FILE}")

# 3) Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)


# grab all of the binary labels out of train_ds so we can compute a weight for each class
all_labels = np.concatenate([
    y.numpy().flatten() 
    for _, y in train_ds
], axis=0)

# compute “balanced” class weights for classes [0,1]
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1]),
    y=all_labels
)
# turn it into the format {class_index: weight}
class_weights = {i: w for i, w in enumerate(class_weights_array)}

print("using class weights:", class_weights)



# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=CHECKPOINT,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# and now pass it into fit():
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,      # ← here!
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# 5) Save training history for Phase 4
with open(HISTORY_FILE, "w") as f:
    json.dump(history.history, f)
print(f"Training history saved to {HISTORY_FILE}")

# 6) Save final model
model.save(FINAL_MODEL)
print(f"✅ Training complete. Best weights at: {CHECKPOINT}")
