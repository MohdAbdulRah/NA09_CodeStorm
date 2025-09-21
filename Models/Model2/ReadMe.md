# AgriVision AI - Universal Plant Disease Detector

## Setup
1. Create venv & install:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt

2. Put your Kaggle dataset (already organized by class folders) inside `data/`.

3. Edit translations/label_map.json to add translations for all classes.

## Run training
python scripts/train.py

Trained model & class indices will be saved to `models/best_agri_model.h5` and `models/class_indices.json`.

## Test inference (text + TTS)
python scripts/inference.py --image test_images/tomato_leaf.jpg --lang te --speak

## Export TFLite
python scripts/export_tflite.py

Output: `models/agri_model.tflite`

## Run REST API
uvicorn scripts.fastapi_server:app --reload --host 0.0.0.0 --port 8000
POST /predict/ (multipart file field `file`, optional `lang` param)

