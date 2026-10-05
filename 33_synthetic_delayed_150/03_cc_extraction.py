import os
import cv2
import numpy as np
import pandas as pd

from skimage.filters import frangi
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    disk
)
from skimage.measure import (
    label,
    regionprops
)


# =====================================================
# PATHS
# =====================================================

ROOT = r"E:\Corpus_Callosum\33_synthetic_delayed_150"

INPUT_DIR = os.path.join(
    ROOT,
    "02_preprocessed"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "03_cc_extraction"
)

MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

ROI_DIR = os.path.join(
    OUTPUT_DIR,
    "roi"
)


for d in [
    OUTPUT_DIR,
    MASK_DIR,
    OVERLAY_DIR,
    ROI_DIR
]:
    os.makedirs(d, exist_ok=True)


results = []


# =====================================================
# PREPROCESS
# SAME AS EXISTING PIPELINE
# =====================================================

def preprocess(img):

    img = img.astype(np.float32)

    img = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-8
    )

    img_uint8 = (
        img * 255
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img_uint8 = clahe.apply(
        img_uint8
    )

    img = (
        img_uint8.astype(np.float32)
        / 255.0
    )

    return img


# =====================================================
# CORPUS CALLOSUM CANDIDATE
# SAME EXISTING ALGORITHM
# =====================================================

def cc_candidate(img):

    vessel = frangi(
        img,
        sigmas=range(1, 6),
        black_ridges=False
    )

    thresh = np.percentile(
        vessel,
        92
    )

    mask = vessel > thresh

    mask = binary_closing(
        mask,
        footprint=disk(3)
    )

    mask = remove_small_objects(
        mask,
        min_size=150
    )

    lbl = label(mask)

    props = regionprops(lbl)

    if len(props) == 0:

        return np.zeros_like(
            mask,
            dtype=np.uint8
        )

    h, w = img.shape

    candidates = []

    for p in props:

        y, x = p.centroid

        area = p.area

        ecc = p.eccentricity

        central_score = (
            1.0
            - abs(x - w / 2)
            / (w / 2)
        )

        score = (
            area * 0.4
            + ecc * 200
            + central_score * 100
        )

        candidates.append(
            (score, p.label)
        )

    best_label = max(
        candidates
    )[1]

    final_mask = (
        lbl == best_label
    )

    return final_mask.astype(
        np.uint8
    )


# =====================================================
# ROI FROM MASK
# SAME EXISTING ALGORITHM
# =====================================================

def roi_from_mask(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max()

    y1 = ys.min()
    y2 = ys.max()

    pad_x = int(
        0.25 * (x2 - x1)
    )

    pad_y = int(
        0.25 * (y2 - y1)
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        mask.shape[1],
        x2 + pad_x
    )

    y2 = min(
        mask.shape[0],
        y2 + pad_y
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# =====================================================
# PROCESSING
# =====================================================

print()
print("=" * 60)
print("STAGE 3 — GAN DELAYED CC EXTRACTION")
print("=" * 60)
print()

files = sorted(
    os.listdir(INPUT_DIR)
)

image_files = [
    f for f in files
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    )
]

print(
    "Input images:",
    len(image_files)
)

success = 0
failed = 0


for file in image_files:

    print(
        f"[{success + failed + 1:03d}/{len(image_files):03d}]",
        file
    )

    path = os.path.join(
        INPUT_DIR,
        file
    )

    try:

        img = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:

            raise RuntimeError(
                "Could not read image"
            )

        img_norm = preprocess(
            img
        )

        mask = cc_candidate(
            img_norm
        )

        bbox = roi_from_mask(
            mask
        )

        if bbox is None:

            print(
                "   REJECT: No ROI detected"
            )

            failed += 1

            continue

        x1, y1, x2, y2 = bbox

        roi = img[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:

            print(
                "   REJECT: Empty ROI"
            )

            failed += 1

            continue

        # -------------------------------------------------
        # Save ROI
        # -------------------------------------------------

        cv2.imwrite(
            os.path.join(
                ROI_DIR,
                file
            ),
            roi
        )

        # -------------------------------------------------
        # Save mask
        # -------------------------------------------------

        mask_img = (
            mask * 255
        ).astype(
            np.uint8
        )

        cv2.imwrite(
            os.path.join(
                MASK_DIR,
                file
            ),
            mask_img
        )

        # -------------------------------------------------
        # Save overlay
        # -------------------------------------------------

        overlay = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

        overlay[
            mask > 0
        ] = [0, 255, 0]

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.imwrite(
            os.path.join(
                OVERLAY_DIR,
                file
            ),
            overlay
        )

        # -------------------------------------------------
        # Record
        # -------------------------------------------------

        component_area = int(
            np.sum(mask > 0)
        )

        roi_height, roi_width = roi.shape

        results.append([
            file,
            x1,
            y1,
            x2,
            y2,
            component_area,
            roi_width,
            roi_height
        ])

        success += 1

        print(
            f"   OK | ROI={roi_width}x{roi_height}"
            f" | CC pixels={component_area}"
        )

    except Exception as e:

        print(
            f"   FAILED: {e}"
        )

        failed += 1


# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(
    results,
    columns=[
        "File",
        "X1",
        "Y1",
        "X2",
        "Y2",
        "CC_Area_Pixels",
        "ROI_Width",
        "ROI_Height"
    ]
)

csv_path = os.path.join(
    OUTPUT_DIR,
    "roi_coordinates.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# =====================================================
# SUMMARY
# =====================================================

print()
print("=" * 60)
print("STAGE 3 COMPLETE")
print("=" * 60)

print(
    "Input images :",
    len(image_files)
)

print(
    "Successful   :",
    success
)

print(
    "Failed       :",
    failed
)

print(
    "ROI folder   :",
    ROI_DIR
)

print(
    "Mask folder  :",
    MASK_DIR
)

print(
    "Overlay      :",
    OVERLAY_DIR
)

print(
    "CSV          :",
    csv_path
)

print("=" * 60)