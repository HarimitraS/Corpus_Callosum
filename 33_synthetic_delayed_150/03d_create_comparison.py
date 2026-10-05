import cv2
import glob
import os
import math
import numpy as np


# ==========================================================
# PATHS
# ==========================================================

TRAIN_IMAGES = r".\dataset\images"
TRAIN_MASKS = r".\dataset\masks"

GAN_IMAGES = (
    r".\33_synthetic_delayed_150\candidates"
)

GAN_MASKS = (
    r".\33_synthetic_delayed_150\03b_unet_localization\masks"
)

OUTPUT = (
    r".\33_synthetic_delayed_150\validation"
)


os.makedirs(
    OUTPUT,
    exist_ok=True
)


# ==========================================================
# IMAGE NORMALIZATION
# ==========================================================

def normalize_display(img):

    if img is None:
        return None

    img = img.astype(
        np.float32
    )

    mn = img.min()
    mx = img.max()

    if mx <= mn:
        return np.zeros_like(
            img,
            dtype=np.uint8
        )

    img = (
        (img - mn) /
        (mx - mn) *
        255
    )

    return img.astype(
        np.uint8
    )


# ==========================================================
# MAKE PANEL
# ==========================================================

def make_panel(
    image,
    mask=None,
    size=(256, 256)
):

    image = normalize_display(
        image
    )

    image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    if mask is not None:

        mask = cv2.resize(
            mask,
            size,
            interpolation=cv2.INTER_NEAREST
        )

        binary = mask > 0

        # Draw mask boundary
        contours, _ = cv2.findContours(
            binary.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            image,
            contours,
            -1,
            (0, 255, 0),
            2
        )

    return image


# ==========================================================
# TRAINING CONTACT SHEET
# ==========================================================

train_files = sorted(
    glob.glob(
        os.path.join(
            TRAIN_IMAGES,
            "*.png"
        )
    )
)


print(
    f"Training images: {len(train_files)}"
)


tiles = []

for f in train_files:

    name = os.path.basename(f)

    image = cv2.imread(
        f,
        cv2.IMREAD_GRAYSCALE
    )

    mask_path = os.path.join(
        TRAIN_MASKS,
        name
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    panel = make_panel(
        image,
        mask
    )

    # Add filename
    cv2.putText(
        panel,
        name,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    tiles.append(panel)


cols = 5
rows = math.ceil(
    len(tiles) / cols
)

sheet = np.zeros(
    (
        rows * 256,
        cols * 256,
        3
    ),
    dtype=np.uint8
)


for i, tile in enumerate(tiles):

    r = i // cols
    c = i % cols

    sheet[
        r * 256:(r + 1) * 256,
        c * 256:(c + 1) * 256
    ] = tile


train_output = os.path.join(
    OUTPUT,
    "training_images_and_gt_masks.png"
)

cv2.imwrite(
    train_output,
    sheet
)


# ==========================================================
# GAN CONTACT SHEET
# ==========================================================

gan_files = sorted(
    glob.glob(
        os.path.join(
            GAN_IMAGES,
            "*.png"
        )
    )
)


print(
    f"GAN images: {len(gan_files)}"
)


# Select 50 evenly distributed candidates
if len(gan_files) > 50:

    indices = np.linspace(
        0,
        len(gan_files) - 1,
        50
    ).astype(int)

    gan_files = [
        gan_files[i]
        for i in indices
    ]


tiles = []


for f in gan_files:

    name = os.path.basename(f)

    image = cv2.imread(
        f,
        cv2.IMREAD_GRAYSCALE
    )

    mask_path = os.path.join(
        GAN_MASKS,
        name
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    panel = make_panel(
        image,
        mask
    )

    cv2.putText(
        panel,
        name,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    tiles.append(panel)


cols = 5
rows = math.ceil(
    len(tiles) / cols
)

sheet = np.zeros(
    (
        rows * 256,
        cols * 256,
        3
    ),
    dtype=np.uint8
)


for i, tile in enumerate(tiles):

    r = i // cols
    c = i % cols

    sheet[
        r * 256:(r + 1) * 256,
        c * 256:(c + 1) * 256
    ] = tile


gan_output = os.path.join(
    OUTPUT,
    "gan_candidates_with_unet_masks.png"
)

cv2.imwrite(
    gan_output,
    sheet
)


# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 70)
print("CONTACT SHEETS CREATED")
print("=" * 70)

print(
    f"Training : {train_output}"
)

print(
    f"GAN      : {gan_output}"
)

print("=" * 70)