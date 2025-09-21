# scripts/fastapi_server.py
"""
FastAPI server that accepts an uploaded image and returns prediction (no TTS).
POST /predict/ with form-data: file (image), lang (optional)
"""
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
import json
from scripts.inference import predict  # calls models and label map

app = FastAPI()

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...), lang: str = Form("en")):
    tmp_path = f"tmp_upload_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        # predict from our inference function
        res = predict(tmp_path, lang=lang, speak=False)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# run with: uvicorn scripts.fastapi_server:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    uvicorn.run("scripts.fastapi_server:app", host="0.0.0.0", port=8000, reload=True)
