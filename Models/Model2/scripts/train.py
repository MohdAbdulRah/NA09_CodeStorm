# scripts/train.py
"""
Train a universal plant disease model using EfficientNetB0 transfer learning.
Saves best weights to models/best_agri_model.h5 and class indices to models/class_indices.json
"""

import os
import json
from glob import glob
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Config
DATA_DIR = "data"
MODELS_DIR = "models"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 15
VALIDATION_SPLIT = 0.2
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

os.makedirs(MODELS_DIR, exist_ok=True)

def collect_paths_and_labels(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    paths = []
    labels = []
    for cls in classes:
        files = glob(os.path.join(data_dir, cls, "*"))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths += files
        labels += [cls] * len(files)
    return paths, labels, classes

def preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    # Decode as 3 channels (RGB) to match EfficientNetB0 pretrained weights
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
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
    # Clear any previous models from memory
    tf.keras.backend.clear_session()
    
    # Method 1: Try direct EfficientNetB0 with explicit shape
    try:
        base = EfficientNetB0(
            weights="imagenet", 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        print("✓ Successfully loaded EfficientNetB0 with ImageNet weights")
    except Exception as e:
        print(f"Method 1 failed: {e}")
        print("Trying alternative approach...")
        
        # Method 2: Load without weights first, then load weights manually
        try:
            base = EfficientNetB0(
                weights=None, 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
            # Try to load ImageNet weights manually
            base.load_weights(tf.keras.utils.get_file(
                'efficientnetb0_notop.h5',
                'https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnetb0_notop.h5',
                cache_subdir='models'
            ))
            print("✓ Successfully loaded EfficientNetB0 weights manually")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            print("Using EfficientNetB0 without pretrained weights...")
            base = EfficientNetB0(
                weights=None, 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
    
    base.trainable = False
    
    # Build the model using the functional API
    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base

def main():
    paths, labels, classes = collect_paths_and_labels(DATA_DIR)
    if len(paths) == 0:
        raise SystemExit("No images found in data/ - make sure dataset is placed correctly.")

    print(f"Found {len(paths)} images across {len(classes)} classes")
    print(f"Classes: {classes}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=VALIDATION_SPLIT, stratify=labels, random_state=SEED
    )

    class_to_idx = {c: i for i, c in enumerate(classes)}
    with open(os.path.join(MODELS_DIR, "class_indices.json"), "w") as f:
        json.dump(class_to_idx, f)
    print("Classes saved to models/class_indices.json")

    train_ds = make_dataset(train_paths, train_labels, class_to_idx, training=True)
    val_ds = make_dataset(val_paths, val_labels, class_to_idx, training=False)

    model, base = build_model(len(classes))
    print(f"Model built successfully with {len(classes)} classes")
    
    ckpt_path = os.path.join(MODELS_DIR, "best_agri_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, 
            save_best_only=True, 
            monitor="val_accuracy", 
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=6, 
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting initial training...")
    # Initial training
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        verbose=1
    )

    print("Starting fine-tuning...")
    # Fine-tuning last layers
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=5, 
        callbacks=callbacks,
        verbose=1
    )

    print("Training finished. Best model at:", ckpt_path)

if __name__ == "__main__":
    main()


