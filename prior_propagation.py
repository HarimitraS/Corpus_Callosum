import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ======================================
# PATHS
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "4_sagittal_slices")
ATLAS_DIR = os.path.join(BASE_DIR, "atlas")

OUTPUT_DIR = os.path.join(BASE_DIR, "7_prior_masks_2")
ROI_DIR = os.path.join(BASE_DIR, "8_roi_2")
OVERLAY_DIR = os.path.join(BASE_DIR, "9_overlays_2")

for d in [OUTPUT_DIR, ROI_DIR, OVERLAY_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================================
# LOAD ATLAS
# ======================================

atlas = cv2.imread(
    os.path.join(
        ATLAS_DIR,
        "average_atlas.png"
    ),
    cv2.IMREAD_GRAYSCALE
)

if atlas is None:
    raise Exception("Could not load average_atlas.png")

prior = cv2.imread(
    os.path.join(
        ATLAS_DIR,
        "cc_prior.png"
    ),
    cv2.IMREAD_GRAYSCALE
)

if prior is None:
    raise Exception("Could not load cc_prior.png")

atlas = atlas.astype(np.float32)

h, w = atlas.shape

# Force prior to atlas size
prior = cv2.resize(
    prior,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ======================================
# REGISTRATION SETTINGS
# ======================================

warp_mode = cv2.MOTION_AFFINE

criteria = (
    cv2.TERM_CRITERIA_EPS |
    cv2.TERM_CRITERIA_COUNT,
    300,
    1e-7
)

results = []

# ======================================
# PROCESS SUBJECTS
# ======================================

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".png")
])

print("\nStarting Prior Propagation...\n")

for file in tqdm(files):

    path = os.path.join(
        INPUT_DIR,
        file
    )

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print("Skipping:", file)
        continue

    # Force subject image to atlas size
    img = cv2.resize(
        img,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    img_f = img.astype(np.float32)

    try:

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        cc, warp_matrix = cv2.findTransformECC(
            atlas,
            img_f,
            warp_matrix,
            warp_mode,
            criteria
        )

        warped_prior = cv2.warpAffine(
            prior,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST +
                  cv2.WARP_INVERSE_MAP
        )

    except Exception as e:

        print(
            f"Registration failed for {file}"
        )

        warped_prior = prior.copy()

    # ==================================
    # CLEAN MASK
    # ==================================

    warped_prior = (
        warped_prior > 127
    ).astype(np.uint8) * 255

    if warped_prior.shape != img.shape:

        warped_prior = cv2.resize(
            warped_prior,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Debug
    print(
        file,
        "IMG:",
        img.shape,
        "MASK:",
        warped_prior.shape
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            file
        ),
        warped_prior
    )

    # ==================================
    # ROI
    # ==================================

    ys, xs = np.where(
        warped_prior > 0
    )

    if len(xs) == 0:
        continue

    x1 = int(xs.min())
    x2 = int(xs.max())

    y1 = int(ys.min())
    y2 = int(ys.max())

    # Tight ROI padding

    pad_x = 0

    pad_y = 0

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        w,
        x2 + pad_x
    )

    y2 = min(
        h,
        y2 + pad_y
    )

    roi = img[
        y1:y2,
        x1:x2
    ]

    cv2.imwrite(
        os.path.join(
            ROI_DIR,
            file
        ),
        roi
    )

    # ==================================
    # OVERLAY
    # ==================================

    overlay = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    mask_idx = warped_prior > 0

    overlay[mask_idx] = [0, 255, 0]

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

    results.append([
        file,
        x1,
        y1,
        x2,
        y2
    ])

# ======================================
# SAVE CSV
# ======================================

df = pd.DataFrame(
    results,
    columns=[
        "File",
        "X1",
        "Y1",
        "X2",
        "Y2"
    ]
)

df.to_csv(
    os.path.join(
        BASE_DIR,
        "roi_coordinates.csv"
    ),
    index=False
)

print("\nDone.\n")

