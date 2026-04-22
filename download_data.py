"""
Download and prepare the uTHCD Tamil dataset from Kaggle.

Usage:
  1. Place your kaggle.json API key in ~/.kaggle/kaggle.json
     (or set KAGGLE_USERNAME / KAGGLE_KEY env vars)
  2. Run: python download_data.py
"""

import os
import zipfile

DATASET = "faizalhajamohideen/uthcdtamil-h"
OUT_DIR = "data/uthcd_tamil"

os.makedirs("data", exist_ok=True)

print("Downloading dataset from Kaggle...")
os.system(f"kaggle datasets download -d {DATASET} -p data/")

zip_path = f"data/{DATASET.split('/')[-1]}.zip"
if not os.path.exists(zip_path):
    # Kaggle sometimes names the zip differently
    for f in os.listdir("data"):
        if f.endswith(".zip"):
            zip_path = os.path.join("data", f)
            break

print(f"Extracting {zip_path} ...")
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("data/raw")

print("Done. Now check data/raw/ and set DATA_DIR in train.py accordingly.")
print("The dataset root must have one subfolder per class (ImageFolder format).")
