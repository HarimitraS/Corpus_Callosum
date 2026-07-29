import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASK_DIR = os.path.join(BASE_DIR, "12_refined_masks")
ROI_DIR = os.path.join(BASE_DIR, "8_roi_2")

SKELETON_DIR = os.path.join(BASE_DIR, "14_skeletons")
THICKNESS_DIR = os.path.join(BASE_DIR, "15_thickness_maps")
MEASURE_DIR = os.path.join(BASE_DIR, "16_measurements")
VIS_DIR = os.path.join(BASE_DIR, "17_visualizations")

for folder in [
    SKELETON_DIR,
    THICKNESS_DIR,
    MEASURE_DIR,
    VIS_DIR
]:
    os.makedirs(folder, exist_ok=True)

# ==========================================================
# FEATURE STORAGE
# ==========================================================

results = []

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def load_binary_mask(path):
    """
    Load segmentation mask and convert to binary.
    """

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read {path}")

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask


def create_skeleton(mask):
    """
    Skeletonize binary mask.
    """

    binary = mask > 0

    skeleton = skeletonize(binary)

    skeleton = (
        skeleton.astype(np.uint8)
    ) * 255

    return skeleton


def create_distance_map(mask):
    """
    Euclidean distance transform.
    """

    binary = mask > 0

    dist = distance_transform_edt(binary)

    return dist


def create_thickness_map(distance_map):
    """
    Thickness = 2 × radius
    """

    thickness = distance_map * 2.0

    return thickness


def contour_features(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    x, y, w, h = cv2.boundingRect(cnt)

    M = cv2.moments(cnt)

    if M["m00"] != 0:

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    else:

        cx = 0
        cy = 0

    return {

        "area": area,

        "perimeter": perimeter,

        "bbox_w": w,

        "bbox_h": h,

        "cx": cx,

        "cy": cy
    }


def skeleton_length(skeleton):

    return np.count_nonzero(skeleton)


def thickness_statistics(thickness, skeleton):

    values = thickness[skeleton > 0]

    if len(values) == 0:

        return {

            "mean": 0,

            "max": 0,

            "min": 0,

            "std": 0
        }

    return {

        "mean": float(np.mean(values)),

        "max": float(np.max(values)),

        "min": float(np.min(values)),

        "std": float(np.std(values))
    }


def save_skeleton(name, skeleton):

    cv2.imwrite(

        os.path.join(
            SKELETON_DIR,
            name
        ),

        skeleton
    )


def save_thickness_map(name, thickness):
    normalized = cv2.normalize(
        thickness,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    cv2.imwrite(
        os.path.join(
            THICKNESS_DIR,
            name
        ),
        colored
    )


# ==========================================================
# VISUALIZATION
# ==========================================================


def save_visualization(
    name,
    roi,
    mask,
    skeleton,
    thickness
):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    skeleton_rgb[skeleton > 0] = (0, 255, 0)

    thickness_norm = cv2.normalize(
        thickness,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    thickness_color = cv2.applyColorMap(
        thickness_norm,
        cv2.COLORMAP_JET
    )

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].imshow(roi, cmap="gray")
    ax[0, 0].set_title("ROI")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(mask, cmap="gray")
    ax[0, 1].set_title("Refined Mask")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(skeleton_rgb)
    ax[1, 0].set_title("Skeleton")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(cv2.cvtColor(thickness_color, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("Thickness Map")
    ax[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, name), dpi=300)
    plt.close()


# ==========================================================
# PROCESS ALL SUBJECTS
# ==========================================================

mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])

print("\nStarting Morphological Analysis...\n")

for file in tqdm(mask_files):
    try:
        mask = load_binary_mask(os.path.join(MASK_DIR, file))
        roi = cv2.imread(os.path.join(ROI_DIR, file), cv2.IMREAD_GRAYSCALE)

        if roi is None:
            roi = np.zeros_like(mask)

        if roi.shape != mask.shape:
            roi = cv2.resize(roi, (mask.shape[1], mask.shape[0]))

        skeleton = create_skeleton(mask)
        distance_map = create_distance_map(mask)
        thickness_map = create_thickness_map(distance_map)

        save_skeleton(file, skeleton)
        save_thickness_map(file, thickness_map)

        features = contour_features(mask)
        if features is None:
            continue

        stats = thickness_statistics(thickness_map, skeleton)
        save_visualization(file, roi, mask, skeleton, thickness_map)

        results.append({
            "Subject": file,
            "Area": features["area"],
            "Perimeter": features["perimeter"],
            "BoundingBoxWidth": features["bbox_w"],
            "BoundingBoxHeight": features["bbox_h"],
            "CentroidX": features["cx"],
            "CentroidY": features["cy"],
            "SkeletonLength": skeleton_length(skeleton),
            "MeanThickness": stats["mean"],
            "MaxThickness": stats["max"],
            "MinThickness": stats["min"],
            "StdThickness": stats["std"]
        })

    except Exception as e:
        print(f"Skipped {file}: {e}")


# ==========================================================
# SAVE FEATURES
# ==========================================================

df = pd.DataFrame(results)
csv_path = os.path.join(MEASURE_DIR, "features.csv")
df.to_csv(csv_path, index=False)

print("\n===================================")
print("Morphological Analysis Complete")
print("===================================")
print(f"Subjects Processed : {len(df)}")
print(f"CSV Saved          : {csv_path}")
print(f"Skeletons          : {SKELETON_DIR}")
print(f"Thickness Maps     : {THICKNESS_DIR}")
print(f"Visualizations     : {VIS_DIR}")
print("===================================\n")