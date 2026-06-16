import os
import cv2
import torch
import numpy as np

from model import get_model

# ==========================
# PATHS
# ==========================

ROI_DIR = r"../8_roi_2"

OUTPUT_DIR = r"../11_unet_masks"

MODEL_PATH = "best_model.pth"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================
# LOAD MODEL
# ==========================

DEVICE = "cuda"

model = get_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

# ==========================
# PREDICT
# ==========================

files = sorted([
    f for f in os.listdir(ROI_DIR)
    if f.endswith(".png")
])

print(
    f"Processing {len(files)} images..."
)

with torch.no_grad():

    for file in files:

        img_path = os.path.join(
            ROI_DIR,
            file
        )

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        original_h, original_w = img.shape

        img_resized = cv2.resize(
            img,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        img_resized = (
            img_resized.astype(np.float32)
            / 255.0
        )

        tensor = torch.tensor(
            img_resized
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(DEVICE)

        pred = model(tensor)

        pred = torch.sigmoid(pred)

        pred = pred.cpu().numpy()[0, 0]

        pred = (
            pred > 0.5
        ).astype(np.uint8) * 255

        pred = cv2.resize(
            pred,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                file
            ),
            pred
        )

print("DONE")