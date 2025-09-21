# scripts/prepare_dataset.py
"""
Scan `data/` and create a label_map template file at translations/label_map.json
Run this once after placing your dataset under data/
"""
import os
import json

DATA_DIR = "data"
OUT = "translations/label_map.json"

def scan_classes(data_dir):
    if not os.path.exists(data_dir):
        raise SystemExit("No data/ directory found. Put your class folders inside data/")
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return classes

def create_template(classes, out_file=OUT):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    template = {}
    for c in classes:
        template[c] = {
            "en": c.replace("_", " "),
            "hi": "",
            "te": ""
        }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"Label map template written: {out_file}. Fill hi/te translations and save.")

if __name__ == "__main__":
    cls = scan_classes(DATA_DIR)
    print("Detected classes:", cls)
    create_template(cls)
