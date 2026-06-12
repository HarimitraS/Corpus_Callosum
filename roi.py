import os
import cv2
import numpy as np
import pandas as pd

from scipy.ndimage import (
    binary_fill_holes,
    binary_closing
)

from skimage.measure import (
    label,
    regionprops
)

from skimage.morphology import (
    remove_small_objects
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(
    BASE_DIR,
    "4_sagittal_slices"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "5_roi"
)

ROI_IMAGES = os.path.join(
    OUTPUT_FOLDER,
    "roi_images"
)

ROI_PREVIEW = os.path.join(
    OUTPUT_FOLDER,
    "preview"
)

os.makedirs(ROI_IMAGES, exist_ok=True)
os.makedirs(ROI_PREVIEW, exist_ok=True)

metadata = []

# ==================================================
# BRAIN MASK
# ==================================================

def brain_mask(img):

    threshold = np.percentile(img, 20)

    mask = img > threshold

    mask = binary_closing(mask)

    mask = binary_fill_holes(mask)

    mask = remove_small_objects(
        mask,
        min_size=500
    )

    return mask.astype(np.uint8)

# ==================================================
# MIDLINE DETECTION
# ==================================================

def find_midline(img):

    h, w = img.shape

    best_score = -1
    best_x = w // 2

    for x in range(
        int(w * 0.3),
        int(w * 0.7)
    ):

        left = img[:, :x]
        right = img[:, x:]

        width = min(
            left.shape[1],
            right.shape[1]
        )

        if width < 20:
            continue

        left = left[:, -width:]
        right = right[:, :width]

        right = np.fliplr(right)

        score = np.corrcoef(
            left.flatten(),
            right.flatten()
        )[0, 1]

        if np.isnan(score):
            continue

        if score > best_score:
            best_score = score
            best_x = x

    return best_x

# ==================================================
# ROI DETECTION
# ==================================================

def detect_roi(img):

    h, w = img.shape

    search_y1 = int(h * 0.15)
    search_y2 = int(h * 0.65)

    search_x1 = int(w * 0.25)
    search_x2 = int(w * 0.75)

    search = img[
        search_y1:search_y2,
        search_x1:search_x2
    ]

    threshold = np.percentile(
        search,
        80
    )

    candidate = search > threshold

    candidate = remove_small_objects(
        candidate,
        min_size=100
    )

    lbl = label(candidate)

    props = regionprops(lbl)

    if len(props) == 0:

        return (
            search_x1,
            search_y1,
            search_x2,
            search_y2
        )

    best = max(
        props,
        key=lambda p:
        p.area * max(
            p.eccentricity,
            0.1
        )
    )

    minr, minc, maxr, maxc = best.bbox

    minr += search_y1
    maxr += search_y1

    minc += search_x1
    maxc += search_x1

    pad_x = int(
        0.30 * (maxc - minc)
    )

    pad_y = int(
        0.30 * (maxr - minr)
    )

    minc = max(0, minc - pad_x)
    maxc = min(w, maxc + pad_x)

    minr = max(0, minr - pad_y)
    maxr = min(h, maxr + pad_y)

    return (
        minc,
        minr,
        maxc,
        maxr
    )

# ==================================================
# PROCESS
# ==================================================

print("\n================================")
print("ROI LOCALIZATION STARTED")
print("================================\n")

for file in sorted(os.listdir(INPUT_FOLDER)):

    if not file.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".tif")
    ):
        continue

    print("Processing:", file)

    path = os.path.join(
        INPUT_FOLDER,
        file
    )

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print("Could not read:", file)
        continue

    img = img.astype(
        np.float32
    )

    img = (
        img - img.min()
    ) / (
        img.max()
        - img.min()
        + 1e-8
    )

    mask = brain_mask(img)

    midline = find_midline(
        img * mask
    )

    x1, y1, x2, y2 = detect_roi(
        img
    )

    roi = img[
        y1:y2,
        x1:x2
    ]

    roi_uint8 = (
        roi * 255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            ROI_IMAGES,
            file
        ),
        roi_uint8
    )

    preview = (
        img * 255
    ).astype(np.uint8)

    preview = cv2.cvtColor(
        preview,
        cv2.COLOR_GRAY2BGR
    )

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    cv2.imwrite(
        os.path.join(
            ROI_PREVIEW,
            file
        ),
        preview
    )

    metadata.append([
        file,
        midline,
        x1,
        y1,
        x2,
        y2,
        roi.shape[1],
        roi.shape[0]
    ])

    print("Saved ROI")

# ==================================================
# CSV
# ==================================================

df = pd.DataFrame(
    metadata,
    columns=[
        "File",
        "Midline",
        "X1",
        "Y1",
        "X2",
        "Y2",
        "ROI_Width",
        "ROI_Height"
    ]
)

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "roi_metadata.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\n================================")
print("ROI LOCALIZATION COMPLETE")
print("================================")
print("Images Saved :", ROI_IMAGES)
print("Preview Saved:", ROI_PREVIEW)
print("CSV Saved    :", csv_path)