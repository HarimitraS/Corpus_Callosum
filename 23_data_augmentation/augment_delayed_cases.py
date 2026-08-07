import os
import cv2
import pandas as pd

from albumentations import (
    Compose,
    Rotate,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    ElasticTransform,
    GaussianBlur
)

# =====================================================
# PATHS
# =====================================================

ROOT = r"E:\Corpus_Callosum"

IMAGE_DIR = os.path.join(ROOT, "8_roi")
MASK_DIR = os.path.join(ROOT, "12_refined_masks")

OUT_IMAGE_DIR = os.path.join(
    ROOT,
    "23_data_augmentation",
    "augmented_images"
)

OUT_MASK_DIR = os.path.join(
    ROOT,
    "23_data_augmentation",
    "augmented_masks"
)

LOG_FILE = os.path.join(
    ROOT,
    "23_data_augmentation",
    "augmentation_log.csv"
)

META_FILE = os.path.join(
    ROOT,
    "meta.csv"
)

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =====================================================
# LOAD METADATA
# =====================================================

print("=" * 60)
print("Loading metadata...")
print("=" * 60)

meta = pd.read_csv(
    META_FILE,
    sep=";"
)

meta.columns = (
    meta.columns
    .str.strip()
    .str.lower()
)

IMAGE_COLUMN = "image_id"
LABEL_COLUMN = "myelinisation"

meta[IMAGE_COLUMN] = (
    meta[IMAGE_COLUMN]
    .astype(str)
    .str.strip()
)

meta[LABEL_COLUMN] = (
    meta[LABEL_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

delayed_ids = meta.loc[
    meta[LABEL_COLUMN] == "delayed",
    IMAGE_COLUMN
].tolist()

print("\nDelayed Subjects Found:")
print(delayed_ids)

# =====================================================
# AUGMENTATION PIPELINE
# =====================================================

transform = Compose([

    Rotate(
        limit=8,
        border_mode=cv2.BORDER_REFLECT,
        p=0.8
    ),

    ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.05,
        rotate_limit=0,
        border_mode=cv2.BORDER_REFLECT,
        p=0.8
    ),

    ElasticTransform(
        alpha=20,
        sigma=4,
        p=0.4
    ),

    RandomBrightnessContrast(
        brightness_limit=0.08,
        contrast_limit=0.08,
        p=0.5
    ),

    GaussianBlur(
        blur_limit=(3,3),
        p=0.2
    )

])

# =====================================================
# SETTINGS
# =====================================================

N_AUG = 50

log = []

generated = 0
processed = 0
skipped = 0

print("\nStarting augmentation...\n")


# =====================================================
# AUGMENTATION LOOP
# =====================================================

for sid in delayed_ids:

    img_path = os.path.join(IMAGE_DIR, sid + ".png")
    mask_path = os.path.join(MASK_DIR, sid + ".png")

    if not os.path.exists(img_path):
        print(f"[SKIPPED] ROI image not found: {sid}")
        skipped += 1
        continue

    if not os.path.exists(mask_path):
        print(f"[SKIPPED] Refined mask not found: {sid}")
        skipped += 1
        continue

    image = cv2.imread(img_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[ERROR] Could not read ROI image: {sid}")
        skipped += 1
        continue

    if mask is None:
        print(f"[ERROR] Could not read mask: {sid}")
        skipped += 1
        continue

    # -------------------------------------------------
    # IMPORTANT FIX
    # Resize mask to ROI size
    # -------------------------------------------------

    if image.shape[:2] != mask.shape[:2]:

        print(f"Resizing mask for {sid}")

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    processed += 1

    print(f"Processing {sid}")

    for i in range(N_AUG):

        augmented = transform(
            image=image,
            mask=mask
        )

        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        filename = f"{sid}_aug_{i:03d}.png"

        cv2.imwrite(
            os.path.join(
                OUT_IMAGE_DIR,
                filename
            ),
            aug_image
        )

        cv2.imwrite(
            os.path.join(
                OUT_MASK_DIR,
                filename
            ),
            aug_mask
        )

        log.append({

            "original_subject": sid,

            "augmented_subject": filename[:-4],

            "augmentation_number": i + 1

        })

        generated += 1


# =====================================================
# SAVE LOG
# =====================================================

log_df = pd.DataFrame(log)

log_df.to_csv(
    LOG_FILE,
    index=False
)

print("\n")
print("=" * 60)
print("AUGMENTATION COMPLETED")
print("=" * 60)

print(f"Processed Subjects : {processed}")
print(f"Skipped Subjects   : {skipped}")
print(f"Generated Images   : {generated}")

print("\nSaved Images To:")
print(OUT_IMAGE_DIR)

print("\nSaved Masks To:")
print(OUT_MASK_DIR)

print("\nSaved Log:")
print(LOG_FILE)

print("=" * 60)