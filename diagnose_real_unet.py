import cv2
import torch
import numpy as np
import sys
from pathlib import Path

ROOT = Path(".")

sys.path.insert(0, str(ROOT / "unet_cc"))

from model import get_model

IMAGE_DIR = ROOT / "dataset" / "images"
MASK_DIR = ROOT / "dataset" / "masks"
MODEL_PATH = ROOT / "unet_cc" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

files = sorted([
    f for f in IMAGE_DIR.iterdir()
    if f.suffix.lower() == ".png"
])

print("=" * 70)
print("REAL DATASET U-NET DIAGNOSTIC")
print("=" * 70)
print("Device :", DEVICE)
print("Images :", len(files))
print()

empty_predictions = 0
prediction_areas = []
ground_truth_areas = []

with torch.no_grad():

    for file in files:

        img = cv2.imread(
            str(file),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(MASK_DIR / file.name),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None or mask is None:
            continue

        img = cv2.resize(
            img,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        x = (
            torch.from_numpy(
                img.astype(np.float32) / 255.0
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        output = model(x)

        probability = torch.sigmoid(output)

        pred = (
            probability.cpu().numpy()[0, 0] > 0.5
        )

        gt = mask > 127

        pred_area = int(pred.sum())
        gt_area = int(gt.sum())

        prediction_areas.append(pred_area)
        ground_truth_areas.append(gt_area)

        if pred_area == 0:
            empty_predictions += 1

print("==============================================")
print("RESULTS")
print("==============================================")

print("Total images              :", len(prediction_areas))
print("Empty predictions        :", empty_predictions)
print(
    "Non-empty predictions    :",
    len(prediction_areas) - empty_predictions
)

print()
print(
    "Prediction area - min    :",
    min(prediction_areas)
)

print(
    "Prediction area - median :",
    int(np.median(prediction_areas))
)

print(
    "Prediction area - max    :",
    max(prediction_areas)
)

print()
print(
    "Ground-truth area - min  :",
    min(ground_truth_areas)
)

print(
    "Ground-truth area median :",
    int(np.median(ground_truth_areas))
)

print(
    "Ground-truth area - max  :",
    max(ground_truth_areas)
)

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
