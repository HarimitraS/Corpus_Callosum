import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

ROOT = Path(".") / "33_synthetic_delayed_150"
ROI_DIR = ROOT / "02_preprocessed"
MASK_DIR = ROOT / "04_refined_masks"
OUTPUT_DIR = ROOT / "06_texture"

GLCM_DIR = OUTPUT_DIR / "glcm_images"
LBP_DIR = OUTPUT_DIR / "lbp_images"
ENTROPY_DIR = OUTPUT_DIR / "entropy_maps"
CSV_PATH = OUTPUT_DIR / "texture_features.csv"

for d in [OUTPUT_DIR, GLCM_DIR, LBP_DIR, ENTROPY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"

def compute_glcm_features(image):
    image = image.astype(np.uint8)
    glcm = graycomatrix(
        image, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        symmetric=True, normed=True
    )
    return {
        "contrast": np.mean(graycoprops(glcm, "contrast")),
        "correlation": np.mean(graycoprops(glcm, "correlation")),
        "energy": np.mean(graycoprops(glcm, "energy")),
        "homogeneity": np.mean(graycoprops(glcm, "homogeneity")),
        "asm": np.mean(graycoprops(glcm, "ASM")),
        "dissimilarity": np.mean(graycoprops(glcm, "dissimilarity")),
    }

def compute_lbp_features(image):
    image = image.astype(np.uint8)
    lbp = local_binary_pattern(
        image, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD
    )
    return {
        "mean": float(np.mean(lbp)),
        "std": float(np.std(lbp)),
        "lbp": lbp
    }

def compute_entropy(image):
    image = image.astype(np.uint8)
    hist, _ = np.histogram(image, bins=256, range=[0, 256])
    hist = hist.flatten()
    s = np.sum(hist)
    if s == 0:
        return 0.0
    hist = hist / s
    return float(entropy(hist + 1e-12, base=2))

mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])
print("=" * 70)
print(f"STAGE 6 — TEXTURE ANALYSIS ON {len(mask_files)} CANDIDATES")
print("=" * 70)

records = []
skipped_empty = 0
success = 0
failed = 0

for file in mask_files:
    try:
        mask = cv2.imread(str(MASK_DIR / file), cv2.IMREAD_GRAYSCALE)
        roi = cv2.imread(str(ROI_DIR / file), cv2.IMREAD_GRAYSCALE)
        if mask is None or roi is None:
            failed += 1
            continue

        if np.count_nonzero(mask > 0) == 0:
            skipped_empty += 1
            continue

        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        glcm = compute_glcm_features(masked_roi)
        lbp = compute_lbp_features(masked_roi)
        entropy_val = compute_entropy(masked_roi)

        records.append({
            "Subject": file,
            "Contrast": glcm["contrast"],
            "Correlation": glcm["correlation"],
            "Energy": glcm["energy"],
            "Homogeneity": glcm["homogeneity"],
            "ASM": glcm["asm"],
            "Dissimilarity": glcm["dissimilarity"],
            "Entropy": entropy_val,
            "LBP_Mean": lbp["mean"],
            "LBP_STD": lbp["std"]
        })
        success += 1
    except Exception as e:
        print(f"FAILED: {file} -> {e}")
        failed += 1

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

print("=" * 70)
print("STAGE 6 COMPLETE")
print(f"Total input masks : {len(mask_files)}")
print(f"Skipped empty     : {skipped_empty}")
print(f"Success           : {success}")
print(f"Failed            : {failed}")
print(f"Texture CSV       : {CSV_PATH} (rows: {len(df)})")
print("=" * 70)
