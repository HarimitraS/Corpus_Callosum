import os
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# =====================================================
# PATHS
# =====================================================

MASK_DIR = r"E:\Corpus_Callosum\26_delayed_unet_masks"

ROI_DIR = r"E:\Corpus_Callosum\25_delayed_cc_extraction\roi"

OUTPUT_MASK = r"E:\Corpus_Callosum\27_delayed_refined_masks"

OUTPUT_OVERLAY = r"E:\Corpus_Callosum\27_delayed_overlay_refined"

os.makedirs(OUTPUT_MASK, exist_ok=True)
os.makedirs(OUTPUT_OVERLAY, exist_ok=True)

# =====================================================
# MORPHOLOGY KERNELS
# =====================================================

kernel_small = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

kernel_large = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

files = sorted([
    f
    for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])

print("\n================================")
print("REFINING SEGMENTATION MASKS")
print("================================")
print(f"Found {len(files)} masks\n")
for file in files:

    print(f"Processing : {file}")

    mask = cv2.imread(
        os.path.join(MASK_DIR, file),
        cv2.IMREAD_GRAYSCALE
    )

    roi = cv2.imread(
        os.path.join(ROI_DIR, file),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None or roi is None:
        print(f"Skipping {file}")
        continue

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_large
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        largest_mask = np.zeros_like(mask)

        largest_mask[labels == largest] = 255

        mask = largest_mask

    filled = binary_fill_holes(mask > 0)

    mask = (filled * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final = np.zeros_like(mask)

    for cnt in contours:

        epsilon = 0.003 * cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            True
        )

        cv2.drawContours(
            final,
            [approx],
            -1,
            255,
            -1
        )

    cv2.imwrite(
        os.path.join(
            OUTPUT_MASK,
            file
        ),
        final
    )

    overlay = cv2.cvtColor(
        roi,
        cv2.COLOR_GRAY2BGR
    )

    overlay[final > 0] = (0, 255, 0)

    cv2.imwrite(
        os.path.join(
            OUTPUT_OVERLAY,
            file
        ),
        overlay
    )

    print(f"Saved : {file}")

print("\n================================")
print("REFINEMENT COMPLETE")
print("================================")
print(f"Masks    : {OUTPUT_MASK}")
print(f"Overlay  : {OUTPUT_OVERLAY}")
print("================================")