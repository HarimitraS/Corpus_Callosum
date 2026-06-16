import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry

# ==================================================
# PATHS
# ==================================================

ROI_DIR = r"C:\Users\kanishk\OneDrive\Desktop\infant\Corpus_Callosum\8_roi_2"

OUTPUT_DIR = r"C:\Users\kanishk\OneDrive\Desktop\infant\Corpus_Callosum\10_medsam_masks"

CHECKPOINT = r"C:\Users\kanishk\OneDrive\Desktop\infant\MedSAM\work_dir\MedSAM\medsam_vit_b.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD MODEL
# ==================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

medsam_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

medsam_model.to(device)
medsam_model.eval()

# ==================================================
# MEDSAM FUNCTION
# ==================================================

@torch.no_grad()
def segment_roi(img):

    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_rgb = cv2.resize(
        img_rgb,
        (1024, 1024),
        interpolation=cv2.INTER_LINEAR
    )

    img_tensor = (
        torch.tensor(img_rgb)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    img_tensor = img_tensor / 255.0

    image_embedding = medsam_model.image_encoder(
        img_tensor
    )

    # ----------------------------------
    # AUTO BOX FOR CC ROI
    # ----------------------------------

    x1 = int(0.10 * w)
    y1 = int(0.10 * h)

    x2 = int(0.90 * w)
    y2 = int(0.90 * h)

    box = np.array(
        [[x1, y1, x2, y2]]
    )

    box_1024 = box.copy().astype(np.float32)

    box_1024[:, [0, 2]] = (
        box_1024[:, [0, 2]]
        * 1024
        / w
    )

    box_1024[:, [1, 3]] = (
        box_1024[:, [1, 3]]
        * 1024
        / h
    )

    box_torch = (
        torch.tensor(
            box_1024,
            dtype=torch.float,
            device=device
        )
    )

    sparse_embeddings, dense_embeddings = (
        medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
    )

    low_res_logits, _ = (
        medsam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    )

    mask = torch.sigmoid(
        low_res_logits
    )

    mask = (
        mask[0, 0]
        .cpu()
        .numpy()
    )

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    mask = (
        mask > 0.5
    ).astype(np.uint8)

    return mask * 255


# ==================================================
# PROCESS ALL ROI IMAGES
# ==================================================

files = sorted([
    f for f in os.listdir(ROI_DIR)
    if f.lower().endswith(".png")
])

print("\nProcessing", len(files), "ROIs\n")

for file in tqdm(files):

    path = os.path.join(
        ROI_DIR,
        file
    )

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        continue

    mask = segment_roi(img)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            file
        ),
        mask
    )

print("\nDone.")