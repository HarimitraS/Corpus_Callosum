import os
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASK_DIR = os.path.join(BASE_DIR, "11_unet_masks")
ROI_DIR = os.path.join(BASE_DIR, "8_roi_2")

OUTPUT_MASK = os.path.join(BASE_DIR, "12_refined_masks")
OUTPUT_OVERLAY = os.path.join(BASE_DIR, "13_overlay_refined")

os.makedirs(OUTPUT_MASK, exist_ok=True)
os.makedirs(OUTPUT_OVERLAY, exist_ok=True)

# =====================================================
# KERNELS
# =====================================================

kernel_small = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3,3)
)

kernel_large = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5,5)
)

files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])

print(f"\nProcessing {len(files)} masks...\n")

for file in files:

    mask = cv2.imread(
        os.path.join(MASK_DIR,file),
        cv2.IMREAD_GRAYSCALE
    )

    roi = cv2.imread(
        os.path.join(ROI_DIR,file),
        cv2.IMREAD_GRAYSCALE
    )

    # -----------------------------------
    # Threshold
    # -----------------------------------

    _,mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------
    # Opening
    # -----------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    # -----------------------------------
    # Closing
    # -----------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_large
    )

    # -----------------------------------
    # Largest Component
    # -----------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:

        largest = 1 + np.argmax(stats[1:,cv2.CC_STAT_AREA])

        mask = np.zeros_like(mask)

        mask[labels==largest]=255

    # -----------------------------------
    # Fill Holes
    # -----------------------------------

    filled = binary_fill_holes(mask>0)

    mask = (filled*255).astype(np.uint8)

    # -----------------------------------
    # Smooth Contour
    # -----------------------------------

    contours,_ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final = np.zeros_like(mask)

    for cnt in contours:

        epsilon = 0.003*cv2.arcLength(cnt,True)

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

    # -----------------------------------
    # Save Mask
    # -----------------------------------

    cv2.imwrite(
        os.path.join(
            OUTPUT_MASK,
            file
        ),
        final
    )

    # -----------------------------------
    # Overlay
    # -----------------------------------

    overlay = cv2.cvtColor(
        roi,
        cv2.COLOR_GRAY2BGR
    )

    overlay[final>0]=(0,255,0)

    cv2.imwrite(
        os.path.join(
            OUTPUT_OVERLAY,
            file
        ),
        overlay
    )

print("\nFinished Successfully.")