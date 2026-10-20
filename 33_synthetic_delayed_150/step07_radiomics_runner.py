import os
import cv2
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from radiomics import featureextractor

ROOT = Path(".") / "33_synthetic_delayed_150"
IMAGE_DIR = ROOT / "02_preprocessed"
MASK_DIR = ROOT / "04_refined_masks"
OUTPUT_DIR = ROOT / "07_radiomics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "radiomic_features.csv"
LOG_FILE = OUTPUT_DIR / "radiomics.log"
PARAM_FILE = OUTPUT_DIR / "params.yaml"

# Reuse the exact param configuration from the existing pipeline
PARAM_TEXT = """
imageType:
  Original: {}

setting:
  normalize: true
  normalizeScale: 100
  binWidth: 25
  label: 1
  force2D: true
  force2Ddimension: 0

featureClass:
  firstorder:
  shape2D:
  glcm:
"""

with open(PARAM_FILE, "w", encoding="utf-8") as f:
    f.write(PARAM_TEXT)

# Set radiomics logger to WARNING to avoid console spam
radiomics_logger = logging.getLogger("radiomics")
radiomics_logger.setLevel(logging.ERROR)

print("=" * 70)
print("STAGE 7 — RADIOMIC FEATURE EXTRACTION ON VALID CANDIDATE MASKS")
print("=" * 70)

extractor = featureextractor.RadiomicsFeatureExtractor(str(PARAM_FILE))
extractor.enableAllImageTypes()
extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("shape2D")
extractor.enableFeatureClassByName("glcm")

mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])

rows = []
skipped_empty = 0
success = 0
failed = 0

for idx, file in enumerate(mask_files, start=1):
    mask_path = MASK_DIR / file
    image_path = IMAGE_DIR / file

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or np.count_nonzero(mask > 0) == 0:
        skipped_empty += 1
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        failed += 1
        continue

    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    try:
        image_itk = sitk.GetImageFromArray(image.astype(np.float32))
        image_itk.SetSpacing((1.0, 1.0))
        mask_itk = sitk.GetImageFromArray(mask_bin.astype(np.float32))
        mask_itk.SetSpacing((1.0, 1.0))

        feature_vector = extractor.execute(image_itk, mask_itk)
        features = {"Subject": file}

        for key, value in feature_vector.items():
            if key.startswith("diagnostics"):
                continue
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = float(value)
                else:
                    continue
            try:
                value = float(value)
            except Exception:
                continue

            cleaned_key = key.replace("original_", "")
            features[cleaned_key] = value

        rows.append(features)
        success += 1

        if success % 25 == 0 or success == 133:
            print(f"Extracted {success} valid samples...")

    except Exception as e:
        print(f"FAILED {file} -> {e}")
        failed += 1

df = pd.DataFrame(rows)
if "Subject" in df.columns:
    cols = ["Subject"] + [c for c in df.columns if c != "Subject"]
    df = df[cols]

df.to_csv(CSV_PATH, index=False)

print("=" * 70)
print("STAGE 7 COMPLETE")
print(f"Total masks checked   : {len(mask_files)}")
print(f"Skipped empty masks   : {skipped_empty}")
print(f"Successfully extracted: {success}")
print(f"Failed                : {failed}")
print(f"Radiomics CSV         : {CSV_PATH} (rows: {len(df)}, columns: {len(df.columns)})")
print("=" * 70)
