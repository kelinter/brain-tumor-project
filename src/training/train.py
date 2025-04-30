"""
Phase 3: Define and train a transfer-learning–based CNN classifier for brain tumor detection.
This script:
 1. Loads train/validation datasets via prepare_datasets()
 2. Builds a pretrained backbone (MobileNetV2)
 3. Adds classification head (GlobalAveragePooling, Dropout, Dense)
 4. Compiles and fits the model, saving the best weights
"""
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import os, sys
# get project root (two levels up from src/training)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

# Import our data loader
from src.data.data_loader import prepare_datasets

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SPEED = 42

# Paths to save model artifacts
ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR    = os.path.join(ROOT_DIR, "models")
CHECKPOINT   = os.path.join(MODEL_DIR, "best_weights.h5")


# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load datasets
train_ds, val_ds = prepare_datasets()

# 2) Build the model
base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze backbone layers
for layer in base_model.layers:
    layer.trainable = False
    
# Attach classification head
x = GlobalAveragePooling2D(name="gap")(base_model.output)
x = Dropout(0.5, name="dropout")(x)
output = Dense(1, activation="sigmoid", name="classifier")(x)

model = Model(inputs=base_model.input, outputs=output)

# Print Model Summary
print("\n" + "="*40 + " MODEL SUMMARY " + "="*40 + "\n")
model.summary()  # prints to stdout
print("\n" + "="*100 + "\n")

# Save it to a text file in the models folder:
summary_file = os.path.join(MODEL_DIR, "model_summary.txt")
with open(summary_file, "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))
print(f"Model summary saved to {summary_file}")



# 3) Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# Callbacks

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
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

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# Save the model
# ModelCheckpoint writes the best weights (best_weights.h5) whenever validation loss improves
model.save(os.path.join(MODEL_DIR, "final_model.h5"))
print(f"✅ Training complete. Best weights at: {CHECKPOINT}")