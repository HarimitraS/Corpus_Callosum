import os
import cv2
import torch
import numpy as np

from model import get_model

# ==========================================
# PATHS
# ==========================================

ROI_DIR = r"E:\Corpus_Callosum\25_delayed_cc_extraction\roi"

OUTPUT_DIR = r"E:\Corpus_Callosum\26_delayed_unet_masks"

MODEL_PATH = r"E:\Corpus_Callosum\unet_cc\best_model.pth"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# DEVICE
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device : {DEVICE}")

# ==========================================
# LOAD MODEL
# ==========================================

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

# ==========================================
# IMAGE LIST
# ==========================================

files = sorted([
    f
    for f in os.listdir(ROI_DIR)
    if f.endswith(".png")
])

print(f"\nFound {len(files)} ROI Images\n")

print("================================")
print("DELAYED U-NET SEGMENTATION")
print("================================")
# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    for file in files:

        print(f"Processing : {file}")

        img_path = os.path.join(
            ROI_DIR,
            file
        )

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print(f"Could not read : {file}")
            continue

        original_h, original_w = img.shape

        img_resized = cv2.resize(
            img,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        img_resized = img_resized.astype(np.float32) / 255.0

        tensor = torch.tensor(
            img_resized
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(DEVICE)

        pred = model(tensor)

        pred = torch.sigmoid(pred)

        pred = pred.cpu().numpy()[0, 0]

        pred = (pred > 0.5).astype(np.uint8) * 255

        pred = cv2.resize(
            pred,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            file
        )

        cv2.imwrite(
            save_path,
            pred
        )

        print(f"Saved : {file}")

print("\n================================")
print("DELAYED U-NET SEGMENTATION COMPLETE")
print("================================")
print(f"Total Images : {len(files)}")
print(f"Masks Saved  : {OUTPUT_DIR}")
print("================================")
