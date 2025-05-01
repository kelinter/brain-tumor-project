import tensorflow as tf
import os

# Data loader for processed MRI images
# -------------------------------------
# This script prepares TensorFlow Dataset objects for training and validation.
# It loads images from the directory structure, applies augmentation to the training set,
# normalizes pixel values, and optimizes input pipelines with prefetching and shuffling.

# Paths and parameters
# ---------------------
# Root directory containing processed images organized into subfolders 'yes' and 'no'

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/processed")
)

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/processed")
)
# Target size for all images (must match preprocessing output)
IMAGE_SIZE = (224, 224)
# Number of examples per batch
BATCH_SIZE = 32 
# Fraction of data reserved for validation (match example)
VALIDATION_SPLIT = 0.1
# Random seed for reproducibility of splits and shuffling (match example)
SEED = 42

# 1) Data Augmentation Pipeline
# ------------------------------
# Implements on-the-fly transformations to increase dataset diversity:
# - RandomFlip: horizontal flips
# - RandomRotation: rotations up to ±15°
# - RandomZoom: zoom in/out ±10%
rotation_factor = 15.0 / 360.0  # Convert degrees to [0,1] for RandomRotation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", name="rand_flip"),
    tf.keras.layers.RandomRotation(
        factor=rotation_factor,
        fill_mode="nearest",
        name="rand_rotate"
    ),
    tf.keras.layers.RandomZoom(
        height_factor=0.1,
        width_factor=0.1,
        fill_mode="nearest",
        name="rand_zoom"
    ),
], name="data_augmentation")

# 2) Normalization Layer
# ----------------------
# Scales pixel values from [0,255] to [0,1]
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255, name="normalization")

def prepare_datasets():
    """
    Builds and returns training and validation tf.data.Dataset objects.

    Workflow:
      a) Load images and binary labels ('yes', 'no') from directory structure
      b) Split into training/validation sets using VALIDATION_SPLIT
      c) Apply augmentation + normalization to training data
      d) Apply normalization to validation data
      e) Shuffle and prefetch to optimize performance

    Returns:
        train_ds: tf.data.Dataset of (image_batch, label_batch)
        val_ds:   tf.data.Dataset of (image_batch, label_batch)
    """
    # create dataset from directory and split
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",        # infer labels from folder names
        label_mode="binary",      # binary classification
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="binary",
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    
    # b) Apply augmentation and normalization to training dataset
    train_ds = (
        train_ds
        .map(lambda imgs, lbls: (data_augmentation(imgs, training=True), lbls),
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda imgs, lbls: (normalization_layer(imgs), lbls),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000, seed=SEED)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # c) Apply only normalization to validation dataset
    val_ds = (
        val_ds
        .map(lambda imgs, lbls: (normalization_layer(imgs), lbls),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


if __name__ == "__main__":
    # Sanity check: load one batch from each and print shapes
    train_dataset, val_dataset = prepare_datasets()
    for images, labels in train_dataset.take(1):
        print(f"Train batch shape: {images.shape}, labels shape: {labels.shape}")
    for images, labels in val_dataset.take(1):
        print(f"Val batch shape: {images.shape}, labels shape: {labels.shape}")