# scripts/inference.py
"""
Single-image inference + multilingual text + optional TTS (gTTS).
Shows top-3 predictions with confidence scores.
"""
import os
import json
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from gtts import gTTS

MODEL_PATH = "models/best_agri_model_fast.h5"
CLASS_INDICES = "models/class_indices.json"
LABEL_MAP = "translations/label_map.json"
IMG_SIZE = (160, 160)  # Fixed: match model input
CONF_THRESHOLD = 0.1   # relaxed threshold

def load_model_and_maps():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found at {MODEL_PATH}. Train first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    with open(LABEL_MAP, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return model, idx_to_class, label_map

def preprocess(path):
    """
    Load image, resize to model input, normalize, and add batch dimension.
    """
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr.astype(np.float32), 0)

def speak_text(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        out = "tmp_tts_out.mp3"
        tts.save(out)
        if os.name == "nt":
            os.system(f"start {out}")
        elif os.uname().sysname == "Darwin":
            os.system(f"afplay {out}")
        else:
            os.system(f"xdg-open {out} &")
    except Exception as e:
        print("TTS failed:", e)

def predict(image_path, lang="en", speak=False):
    model, idx_to_class, label_map = load_model_and_maps()
    x = preprocess(image_path)
    preds = model.predict(x)[0]

    # Top-3 predictions
    top_indices = preds.argsort()[-3:][::-1]
    results = []
    for idx in top_indices:
        class_name = idx_to_class[idx]
        conf = float(preds[idx])
        translations = label_map.get(class_name, {"en": class_name})
        out_text = translations.get(lang, translations.get("en", class_name))
        precaution = translations.get("precaution", {}).get(lang, "")
        results.append({"class": class_name, "confidence": conf, "text": out_text,"precaution":precaution})

    top_pred = results[0]

    # Uncertain case
    if top_pred["confidence"] < CONF_THRESHOLD:
        response = {
            "status": "uncertain",
            "confidence": top_pred["confidence"],
            "text": {
                "en": "Uncertain result. Please retake photo or consult an expert.",
                "hi": "परिणाम अनिश्चित है। कृपया फोटो फिर से लें या विशेषज्ञ से संपर्क करें।",
                "te": "ఫలితం స్పష్టంగా లేదు. దయచేసి ఫోటో మళ్లీ తీసుకోండి లేదా నిపుణుడిని సంప్రదించండి."
            }.get(lang, "Uncertain"),
            "top3": results
        }
        if speak:
            speak_text(response["text"], lang_code=lang if lang in ["en", "hi", "te"] else "en")
        return response

    # Otherwise return top-1 + top-3
    if speak:
        tcode = lang if lang in ["en", "hi", "te"] else "en"
        speak_text(top_pred["text"], lang_code=tcode)

    return {"status": "ok", "prediction": top_pred}  #, "top3": results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--lang", default="en", help="en/hi/te")
    parser.add_argument("--speak", action="store_true", help="Enable speech (gTTS)")
    args = parser.parse_args()

    out = predict(args.image, lang=args.lang, speak=args.speak)
    print(out)

