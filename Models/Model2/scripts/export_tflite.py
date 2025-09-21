# scripts/export_tflite.py
"""
Convert saved Keras model to TFLite (with optional quantization using a small representative set).
Produces models/agri_model.tflite
"""
import os
import tensorflow as tf
from glob import glob

MODEL_H5 = "models/best_agri_model.h5"
TFLITE_OUT = "models/agri_model.tflite"
DATA_DIR = "data"

if not os.path.exists(MODEL_H5):
    raise SystemExit("Model not found. Train first.")

model = tf.keras.models.load_model(MODEL_H5)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset generator (small)
def representative_dataset_gen():
    files = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        files.extend(glob(os.path.join(DATA_DIR, "**", ext), recursive=True)[:200])
    import numpy as np
    from PIL import Image
    for f in files:
        img = Image.open(f).convert("RGB").resize((224,224))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, 0)
        yield [arr]

try:
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
except Exception as e:
    print("Quantization setup failed (optional). Continuing without full int8 quantization.", e)

tflite_model = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)
print("TFLite model written to", TFLITE_OUT)
