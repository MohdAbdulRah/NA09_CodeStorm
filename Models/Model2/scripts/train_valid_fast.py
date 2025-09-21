"""
Ultra-fast training version using MobileNetV2 (alpha=0.35) with:
✔ Smaller image size (128x128)
✔ Stronger data augmentation
✔ Class weights to fix imbalance
✔ Mixed precision for faster GPU training
✔ Dataset caching and prefetching
✔ Saves model as models/best_agri_model_fast.h5
"""

import os, json
from glob import glob
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster GPU training
mixed_precision.set_global_policy('mixed_float16')

# Config
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")
MODELS_DIR = "models"
BATCH_SIZE = 64
IMG_SIZE = (128, 128)
EPOCHS = 15
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs(MODELS_DIR, exist_ok=True)

# --- Helper functions ---
def collect_paths_and_labels(base_dir):
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    paths, labels = [], []
    for cls in classes:
        files = glob(os.path.join(base_dir, cls, "*"))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths += files
        labels += [cls] * len(files)
    return paths, labels, classes

def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.05)
    crop_size = int(IMG_SIZE[0] * 0.9)
    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
    img = tf.image.resize(img, IMG_SIZE)
    return img

def make_dataset(paths, labels, class_to_idx, training=True):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices([class_to_idx[l] for l in labels])
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    if training:
        ds = ds.shuffle(buffer_size=1000, seed=SEED)
    def _load(path, label):
        img = preprocess_image(path)
        if training:
            img = augment_image(img)
        return img, label
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)  # Cache + prefetch for speed
    return ds

def build_model(num_classes):
    base = MobileNetV2(weights="imagenet", include_top=False, alpha=0.35, input_shape=IMG_SIZE+(3,))
    base.trainable = False  # Freeze base
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)  # Force float32 output
    model = models.Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# --- Main ---
def main():
    train_paths, train_labels, train_classes = collect_paths_and_labels(TRAIN_DIR)
    val_paths, val_labels, val_classes = collect_paths_and_labels(VAL_DIR)

    classes = sorted(list(set(train_classes) | set(val_classes)))
    print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images across {len(classes)} classes")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(MODELS_DIR, "class_indices.json"), "w") as f:
        json.dump(class_to_idx, f)

    y_train_idx = [class_to_idx[l] for l in train_labels]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(classes)), y=y_train_idx)
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    train_ds = make_dataset(train_paths, train_labels, class_to_idx, training=True)
    val_ds = make_dataset(val_paths, val_labels, class_to_idx, training=False)

    model = build_model(len(classes))
    ckpt_path = os.path.join(MODELS_DIR, "best_agri_model_valid_fast.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS, callbacks=callbacks,
              class_weight=class_weights,
              verbose=1)

    print("Training finished. Model saved at:", ckpt_path)

if __name__ == "__main__":
    main()
