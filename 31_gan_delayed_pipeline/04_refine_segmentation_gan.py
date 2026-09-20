import os
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

# =====================================================
# PATHS
# =====================================================

ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MASK_DIR = os.path.join(
    ROOT,
    "04_unet_masks"
)

ROI_DIR = os.path.join(
    ROOT,
    "03_cc_extraction",
    "roi"
)

OUTPUT_MASK = os.path.join(
    ROOT,
    "05_refined_masks"
)

OUTPUT_OVERLAY = os.path.join(
    ROOT,
    "05_refined_overlays"
)

os.makedirs(
    OUTPUT_MASK,
    exist_ok=True
)

os.makedirs(
    OUTPUT_OVERLAY,
    exist_ok=True
)

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

# =====================================================
# IMAGE LIST
# =====================================================

files = sorted([
    f
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
])

print("\n================================")
print("GAN MASK REFINEMENT")
print("================================")
print(f"Found {len(files)} masks\n")

success = 0
failed = 0

# =====================================================
# PROCESS
# =====================================================

for file in files:

    print(f"Processing : {file}")

    try:

        mask = cv2.imread(
            os.path.join(
                MASK_DIR,
                file
            ),
            cv2.IMREAD_GRAYSCALE
        )

        roi = cv2.imread(
            os.path.join(
                ROI_DIR,
                file
            ),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {file}"
            )

        if roi is None:
            raise RuntimeError(
                f"Could not read ROI: {file}"
            )

        # -------------------------------------------------
        # Binary threshold
        # -------------------------------------------------

        _, mask = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

        # -------------------------------------------------
        # Morphological opening
        # -------------------------------------------------

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel_small
        )

        # -------------------------------------------------
        # Morphological closing
        # -------------------------------------------------

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel_large
        )

        # -------------------------------------------------
        # Keep largest connected component
        # -------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask
            )
        )

        if num_labels > 1:

            largest = (
                1
                + np.argmax(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA
                    ]
                )
            )

            largest_mask = np.zeros_like(
                mask
            )

            largest_mask[
                labels == largest
            ] = 255

            mask = largest_mask

        # -------------------------------------------------
        # Fill holes
        # -------------------------------------------------

        filled = binary_fill_holes(
            mask > 0
        )

        mask = (
            filled * 255
        ).astype(
            np.uint8
        )

        # -------------------------------------------------
        # Smooth contour
        # -------------------------------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        final = np.zeros_like(
            mask
        )

        for cnt in contours:

            epsilon = (
                0.003
                * cv2.arcLength(
                    cnt,
                    True
                )
            )

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

        # -------------------------------------------------
        # Save refined mask
        # -------------------------------------------------

        cv2.imwrite(
            os.path.join(
                OUTPUT_MASK,
                file
            ),
            final
        )

        # -------------------------------------------------
        # Save overlay
        # -------------------------------------------------

        overlay = cv2.cvtColor(
            roi,
            cv2.COLOR_GRAY2BGR
        )

        overlay[
            final > 0
        ] = (0, 255, 0)

        cv2.imwrite(
            os.path.join(
                OUTPUT_OVERLAY,
                file
            ),
            overlay
        )

        print(f"Saved : {file}")

        success += 1

    except Exception as e:

        print(
            f"FAILED : {file} -> {e}"
        )

        failed += 1


# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("REFINEMENT COMPLETE")
print("================================")
print(f"Total    : {len(files)}")
print(f"Success  : {success}")
print(f"Failed   : {failed}")
print(f"Masks    : {OUTPUT_MASK}")
print(f"Overlays : {OUTPUT_OVERLAY}")
print("================================")