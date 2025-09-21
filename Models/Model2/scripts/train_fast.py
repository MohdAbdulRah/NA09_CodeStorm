# scripts/train_fast.py
"""
Fast training version using MobileNetV2 with:
✔ Stronger data augmentation
✔ Class weights (to fix imbalance)
Saves model as models/best_agri_model_fast.h5
"""

import os, json
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Config
DATA_DIR = "data"
MODELS_DIR = "models"
BATCH_SIZE = 64
IMG_SIZE = (160, 160)
EPOCHS = 20   # zyada epochs for better convergence
VALIDATION_SPLIT = 0.2
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs(MODELS_DIR, exist_ok=True)

def collect_paths_and_labels(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    paths, labels = [], []
    for cls in classes:
        files = glob(os.path.join(data_dir, cls, "*"))
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
    # Stronger augmentations to generalize better
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.05)
    # Random crop + resize back
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
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def build_model(num_classes):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE+(3,))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    paths, labels, classes = collect_paths_and_labels(DATA_DIR)
    if not paths:
        raise SystemExit("No images found in data/ folder.")

    print(f"Found {len(paths)} images across {len(classes)} classes")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=VALIDATION_SPLIT, stratify=labels, random_state=SEED
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(MODELS_DIR, "class_indices.json"), "w") as f:
        json.dump(class_to_idx, f)

    # Compute class weights to fix imbalance
    y_train_idx = [class_to_idx[l] for l in train_labels]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=y_train_idx
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    train_ds = make_dataset(train_paths, train_labels, class_to_idx, training=True)
    val_ds = make_dataset(val_paths, val_labels, class_to_idx, training=False)

    model = build_model(len(classes))
    ckpt_path = os.path.join(MODELS_DIR, "best_agri_model_fast.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS, callbacks=callbacks,
              class_weight=class_weights,  # ✅ important
              verbose=1)

    print("Training finished. Fast model with class weights at:", ckpt_path)

if __name__ == "__main__":
    main()
