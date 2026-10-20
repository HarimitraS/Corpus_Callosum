import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

ROOT = Path(".") / "33_synthetic_delayed_150"
MASK_DIR = ROOT / "04_refined_masks"
ROI_DIR = ROOT / "02_preprocessed"
OUTPUT_DIR = ROOT / "05_morphology"

SKELETON_DIR = OUTPUT_DIR / "skeletons"
THICKNESS_DIR = OUTPUT_DIR / "thickness_maps"
MEASURE_DIR = OUTPUT_DIR / "measurements"

for folder in [OUTPUT_DIR, SKELETON_DIR, THICKNESS_DIR, MEASURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def load_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(path)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def create_skeleton(mask):
    skeleton = skeletonize(mask > 0)
    return (skeleton.astype(np.uint8)) * 255

def create_distance_map(mask):
    return distance_transform_edt(mask > 0)

def create_thickness_map(distance):
    return distance * 2.0

def contour_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return {
        "area": area,
        "perimeter": perimeter,
        "bbox_w": w,
        "bbox_h": h,
        "cx": cx,
        "cy": cy
    }

def skeleton_length(skeleton):
    return int(np.count_nonzero(skeleton))

def thickness_statistics(thickness, skeleton):
    values = thickness[skeleton > 0]
    if len(values) == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "std": float(np.std(values))
    }

mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])
print("=" * 70)
print(f"STAGE 5 — MORPHOLOGICAL ANALYSIS ON {len(mask_files)} MASKS")
print("=" * 70)

results = []
skipped_empty = 0
success = 0
failed = 0

for file in mask_files:
    try:
        mask = load_binary_mask(MASK_DIR / file)
        fg = np.count_nonzero(mask > 0)
        if fg == 0:
            skipped_empty += 1
            continue

        skeleton = create_skeleton(mask)
        distance = create_distance_map(mask)
        thickness = create_thickness_map(distance)

        cv2.imwrite(str(SKELETON_DIR / file), skeleton)

        norm = cv2.normalize(thickness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imwrite(str(THICKNESS_DIR / file), colored)

        feature = contour_features(mask)
        if feature is None or feature["area"] == 0:
            skipped_empty += 1
            continue

        stats = thickness_statistics(thickness, skeleton)

        results.append({
            "Subject": file,
            "Area": feature["area"],
            "Perimeter": feature["perimeter"],
            "BoundingBoxWidth": feature["bbox_w"],
            "BoundingBoxHeight": feature["bbox_h"],
            "CentroidX": feature["cx"],
            "CentroidY": feature["cy"],
            "SkeletonLength": skeleton_length(skeleton),
            "MeanThickness": stats["mean"],
            "MaxThickness": stats["max"],
            "MinThickness": stats["min"],
            "StdThickness": stats["std"]
        })
        success += 1
    except Exception as e:
        print(f"FAILED: {file} -> {e}")
        failed += 1

df = pd.DataFrame(results)
csv_out = MEASURE_DIR / "features.csv"
df.to_csv(csv_out, index=False)

print("=" * 70)
print("STAGE 5 COMPLETE")
print(f"Total input masks : {len(mask_files)}")
print(f"Skipped empty     : {skipped_empty}")
print(f"Successfully measured: {success}")
print(f"Failed            : {failed}")
print(f"Morphology CSV    : {csv_out} (rows: {len(df)})")
print("=" * 70)
