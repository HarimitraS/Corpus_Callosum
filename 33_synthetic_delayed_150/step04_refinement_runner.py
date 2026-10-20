import os
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from pathlib import Path
import csv

ROOT = Path(".") / "33_synthetic_delayed_150"
INPUT_MASK_DIR = ROOT / "03b_unet_localization" / "masks"
INPUT_IMG_DIR = ROOT / "02_preprocessed"
OUTPUT_DIR = ROOT / "04_refined_masks"
OVERLAY_DIR = ROOT / "04_refined_overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

files = sorted([f for f in os.listdir(INPUT_MASK_DIR) if f.lower().endswith(".png")])

print("=" * 70)
print(f"STAGE 4 — MASK REFINEMENT ON {len(files)} CANDIDATES")
print("=" * 70)

results = []
valid_count = 0
empty_initial = 0
empty_post_refine = 0

for file in files:
    mask_path = INPUT_MASK_DIR / file
    img_path = INPUT_IMG_DIR / file

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if mask is None or img is None:
        results.append([file, 0, 0, "FAILED", "Read error"])
        continue

    initial_fg = int(np.count_nonzero(mask > 0))

    if initial_fg == 0:
        empty_initial += 1
        cv2.imwrite(str(OUTPUT_DIR / file), mask)
        results.append([file, 0, 0, "EMPTY_INITIAL", "Zero U-Net foreground"])
        continue

    # Morphological Opening
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    # Morphological Closing
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)

    # Connected Components - Keep largest
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = np.zeros_like(closed)
        largest_mask[labels == largest] = 255
        refined = largest_mask
    else:
        refined = np.zeros_like(closed)

    # Fill holes
    filled = binary_fill_holes(refined > 0)
    refined = (filled * 255).astype(np.uint8)

    # Contour smoothing
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(refined)
    for cnt in contours:
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(final, [approx], -1, 255, -1)

    final_fg = int(np.count_nonzero(final > 0))

    cv2.imwrite(str(OUTPUT_DIR / file), final)

    # Create overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[final > 0] = [0, 255, 0]
    cv2.imwrite(str(OVERLAY_DIR / file), overlay)

    if final_fg == 0:
        empty_post_refine += 1
        results.append([file, initial_fg, 0, "EMPTY_POST_REFINE", "Removed during morphological refinement"])
    else:
        valid_count += 1
        results.append([file, initial_fg, final_fg, "VALID", "Refined mask produced"])

csv_path = OUTPUT_DIR / "refinement_summary.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["File", "Initial_FG_Pixels", "Final_FG_Pixels", "Status", "Reason"])
    writer.writerows(results)

print("=" * 70)
print("STAGE 4 COMPLETE")
print(f"Total processed           : {len(files)}")
print(f"Empty initially (U-Net)   : {empty_initial}")
print(f"Empty post-refine         : {empty_post_refine}")
print(f"Total empty refined masks : {empty_initial + empty_post_refine} ({(empty_initial + empty_post_refine)/len(files)*100:.1f}%)")
print(f"Valid non-empty masks     : {valid_count} ({valid_count/len(files)*100:.1f}%)")
print(f"Output masks              : {OUTPUT_DIR}")
print("=" * 70)
