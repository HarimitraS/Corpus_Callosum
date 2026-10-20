import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import csv

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(".") / "33_synthetic_delayed_150"
INPUT_DIR = ROOT / "02_preprocessed"
OUTPUT_ROOT = ROOT / "03b_unet_localization"

PROB_DIR = OUTPUT_ROOT / "probability"
MASK_DIR = OUTPUT_ROOT / "masks"
OVERLAY_DIR = OUTPUT_ROOT / "overlays"
ROI_DIR = OUTPUT_ROOT / "roi"

for directory in [OUTPUT_ROOT, PROB_DIR, MASK_DIR, OVERLAY_DIR, ROI_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================================
# EXISTING U-NET
# ==========================================================

sys.path.insert(0, str(Path(".").resolve() / "unet_cc"))
from model import get_model

MODEL_PATH = Path(".") / "unet_cc" / "best_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {DEVICE}")

model = get_model()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
)
model.to(DEVICE)
model.eval()

files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png")])

print("=" * 70)
print(f"STAGE 3B — FULL-IMAGE U-NET LOCALIZATION ON {len(files)} IMAGES")
print("=" * 70)

THRESHOLD = 0.5
PADDING_RATIO = 0.25
MIN_MASK_PIXELS = 1

success = 0
empty = 0
failed = 0
results = []

with torch.no_grad():
    for index, file in enumerate(files, start=1):
        try:
            image_path = INPUT_DIR / file
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Could not read image")

            original_h, original_w = img.shape

            img_resized = cv2.resize(
                img, (256, 256), interpolation=cv2.INTER_LINEAR
            )
            img_resized = img_resized.astype(np.float32) / 255.0

            tensor = torch.tensor(
                img_resized, dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0).to(DEVICE)

            prediction = model(tensor)
            probability = torch.sigmoid(prediction).cpu().numpy()[0, 0]

            probability_original = cv2.resize(
                probability, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )
            probability_uint8 = (np.clip(probability_original, 0.0, 1.0) * 255).astype(np.uint8)

            cv2.imwrite(str(PROB_DIR / file), probability_uint8)

            # Standard threshold >= 0.5
            mask = (probability_original >= THRESHOLD).astype(np.uint8)
            mask_pixels = int(np.count_nonzero(mask))
            mask_uint8 = (mask * 255).astype(np.uint8)

            cv2.imwrite(str(MASK_DIR / file), mask_uint8)

            if mask_pixels < MIN_MASK_PIXELS:
                empty += 1
                results.append([
                    file, original_w, original_h, mask_pixels,
                    "", "", "", "", "EMPTY", "U-Net produced no foreground at 0.5"
                ])
                continue

            ys, xs = np.where(mask > 0)
            x1 = int(xs.min())
            x2 = int(xs.max())
            y1 = int(ys.min())
            y2 = int(ys.max())

            width = x2 - x1
            height = y2 - y1

            pad_x = max(1, int(PADDING_RATIO * width))
            pad_y = max(1, int(PADDING_RATIO * height))

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(original_w, x2 + pad_x + 1)
            y2 = min(original_h, y2 + pad_y + 1)

            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                failed += 1
                results.append([
                    file, original_w, original_h, mask_pixels,
                    x1, y1, x2, y2, "FAILED", "Empty ROI"
                ])
                continue

            cv2.imwrite(str(ROI_DIR / file), roi)

            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay[mask > 0] = [0, 255, 0]
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)
            cv2.imwrite(str(OVERLAY_DIR / file), overlay)

            success += 1
            results.append([
                file, original_w, original_h, mask_pixels,
                x1, y1, x2, y2, "SUCCESS", "Full-image U-Net localization"
            ])

            if index % 25 == 0 or index == len(files):
                print(f"[{index:03d}/{len(files):03d}] Valid: {success} | Empty: {empty} | Failed: {failed}")

        except Exception as e:
            failed += 1
            results.append([file, "", "", "", "", "", "", "", "FAILED", str(e)])

csv_path = OUTPUT_ROOT / "unet_localization_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "File", "Original_Width", "Original_Height", "Mask_Pixels",
        "X1", "Y1", "X2", "Y2", "Status", "Reason"
    ])
    writer.writerows(results)

print("=" * 70)
print("STAGE 3B COMPLETE")
print(f"Total images : {len(files)}")
print(f"Valid masks  : {success}")
print(f"Empty masks  : {empty}")
print(f"Failed ROIs  : {failed}")
print(f"Results CSV  : {csv_path}")
print("=" * 70)
