import os
import cv2
import torch
import numpy as np

from pathlib import Path
import sys

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

ROI_DIR = ROOT / "03_cc_extraction" / "roi"

OUTPUT_DIR = ROOT / "04_unet_masks"

MODEL_PATH = (
    ROOT.parent /
    "unet_cc" /
    "best_model.pth"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# IMPORT EXISTING MODEL
# ==========================================================

sys.path.insert(
    0,
    str(ROOT.parent / "unet_cc")
)

from model import get_model

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using Device : {DEVICE}")

# ==========================================================
# LOAD MODEL
# ==========================================================

model = get_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device(DEVICE)
    )
)

model.to(DEVICE)
model.eval()

print("Model Loaded Successfully")

# ==========================================================
# IMAGE LIST
# ==========================================================

files = sorted([
    f
    for f in os.listdir(ROI_DIR)
    if f.lower().endswith(".png")
])

print(f"\nFound {len(files)} ROI Images\n")

print("================================")
print("GAN DELAYED U-NET SEGMENTATION")
print("================================")

# ==========================================================
# PREDICTION
# ==========================================================

success = 0
failed = 0

with torch.no_grad():

    for file in files:

        print(f"Processing : {file}")

        try:

            img_path = ROI_DIR / file

            img = cv2.imread(
                str(img_path),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                raise RuntimeError(
                    f"Could not read {file}"
                )

            original_h, original_w = img.shape

            # ------------------------------------------------
            # Resize to model input
            # ------------------------------------------------

            img_resized = cv2.resize(
                img,
                (256, 256),
                interpolation=cv2.INTER_LINEAR
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            img_resized = (
                img_resized.astype(
                    np.float32
                ) / 255.0
            )

            # ------------------------------------------------
            # NumPy -> Tensor
            # ------------------------------------------------

            tensor = torch.tensor(
                img_resized,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)

            tensor = tensor.to(DEVICE)

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            pred = model(tensor)

            pred = torch.sigmoid(pred)

            pred = (
                pred
                .cpu()
                .numpy()[0, 0]
            )

            # ------------------------------------------------
            # Binary mask
            # ------------------------------------------------

            pred = (
                pred > 0.5
            ).astype(
                np.uint8
            ) * 255

            # ------------------------------------------------
            # Restore original ROI size
            # ------------------------------------------------

            pred = cv2.resize(
                pred,
                (
                    original_w,
                    original_h
                ),
                interpolation=cv2.INTER_NEAREST
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            save_path = (
                OUTPUT_DIR / file
            )

            cv2.imwrite(
                str(save_path),
                pred
            )

            print(
                f"Saved : {file}"
            )

            success += 1

        except Exception as e:

            print(
                f"FAILED : {file} -> {e}"
            )

            failed += 1


# ==========================================================
# SUMMARY
# ==========================================================

print("\n================================")
print("GAN DELAYED U-NET SEGMENTATION COMPLETE")
print("================================")
print(f"Total Images : {len(files)}")
print(f"Successful   : {success}")
print(f"Failed       : {failed}")
print(f"Masks Saved  : {OUTPUT_DIR}")
print("================================")