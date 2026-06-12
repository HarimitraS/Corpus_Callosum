import os
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "4_sagittal_slices"
)

ATLAS_DIR = os.path.join(
    BASE_DIR,
    "atlas"
)

os.makedirs(ATLAS_DIR, exist_ok=True)

# ==========================================
# LOAD FILES
# ==========================================

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".png")
])

if len(files) == 0:
    raise Exception("No PNG files found")

print(f"\nFound {len(files)} images\n")

# ==========================================
# REFERENCE IMAGE
# ==========================================

ref = cv2.imread(
    os.path.join(INPUT_DIR, files[0]),
    cv2.IMREAD_GRAYSCALE
)

ref = ref.astype(np.float32)

h, w = ref.shape

atlas_sum = np.zeros(
    (h, w),
    dtype=np.float32
)

# ==========================================
# ECC REGISTRATION
# ==========================================

warp_mode = cv2.MOTION_AFFINE

criteria = (
    cv2.TERM_CRITERIA_EPS |
    cv2.TERM_CRITERIA_COUNT,
    100,
    1e-6
)

# ==========================================
# BUILD ATLAS
# ==========================================

for file in tqdm(files):

    img = cv2.imread(
        os.path.join(INPUT_DIR, file),
        cv2.IMREAD_GRAYSCALE
    )

    img = img.astype(np.float32)

    try:

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        cc, warp_matrix = cv2.findTransformECC(
            ref,
            img,
            warp_matrix,
            warp_mode,
            criteria
        )

        aligned = cv2.warpAffine(
            img,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR +
                  cv2.WARP_INVERSE_MAP
        )

    except:

        aligned = img

    atlas_sum += aligned

# ==========================================
# AVERAGE
# ==========================================

atlas = atlas_sum / len(files)

atlas = cv2.normalize(
    atlas,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

atlas = atlas.astype(np.uint8)

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    os.path.join(
        ATLAS_DIR,
        "average_atlas.png"
    ),
    atlas
)

print("\nAtlas saved.\n")
