import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# ==========================================================
# PATHS
# ==========================================================

MASK_DIR = r"E:\Corpus_Callosum\27_delayed_refined_masks"

ROI_DIR = r"E:\Corpus_Callosum\25_delayed_cc_extraction\roi"

OUTPUT_DIR = r"E:\Corpus_Callosum\28_delayed_morphological_analysis"

SKELETON_DIR = os.path.join(
    OUTPUT_DIR,
    "skeletons"
)

THICKNESS_DIR = os.path.join(
    OUTPUT_DIR,
    "thickness_maps"
)

MEASURE_DIR = os.path.join(
    OUTPUT_DIR,
    "measurements"
)

VIS_DIR = os.path.join(
    OUTPUT_DIR,
    "visualizations"
)

for folder in [
    OUTPUT_DIR,
    SKELETON_DIR,
    THICKNESS_DIR,
    MEASURE_DIR,
    VIS_DIR
]:
    os.makedirs(folder, exist_ok=True)

results = []

# ==========================================================
# FUNCTIONS
# ==========================================================

def load_binary_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(path)

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask


def create_skeleton(mask):

    skeleton = skeletonize(mask > 0)

    return (
        skeleton.astype(np.uint8)
    ) * 255


def create_distance_map(mask):

    return distance_transform_edt(mask > 0)


def create_thickness_map(distance):

    return distance * 2.0


def contour_features(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    x, y, w, h = cv2.boundingRect(cnt)

    M = cv2.moments(cnt)

    if M["m00"] != 0:

        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])

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


def thickness_statistics(
    thickness,
    skeleton
):

    values = thickness[
        skeleton > 0
    ]

    if len(values) == 0:

        return {
            "mean":0,
            "max":0,
            "min":0,
            "std":0
        }

    return {
        "mean":float(np.mean(values)),
        "max":float(np.max(values)),
        "min":float(np.min(values)),
        "std":float(np.std(values))
    }


def save_skeleton(name, skeleton):

    cv2.imwrite(
        os.path.join(
            SKELETON_DIR,
            name
        ),
        skeleton
    )


def save_thickness_map(
    name,
    thickness
):

    norm = cv2.normalize(
        thickness,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        norm,
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

    skeleton_rgb = cv2.cvtColor(
        skeleton,
        cv2.COLOR_GRAY2BGR
    )

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

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(10,10)
    )

    ax[0,0].imshow(roi, cmap="gray")
    ax[0,0].set_title("ROI")
    ax[0,0].axis("off")

    ax[0,1].imshow(mask, cmap="gray")
    ax[0,1].set_title("Refined Mask")
    ax[0,1].axis("off")

    ax[1,0].imshow(skeleton_rgb)
    ax[1,0].set_title("Skeleton")
    ax[1,0].axis("off")

    ax[1,1].imshow(
        cv2.cvtColor(
            thickness_color,
            cv2.COLOR_BGR2RGB
        )
    )

    ax[1,1].set_title("Thickness Map")
    ax[1,1].axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            VIS_DIR,
            name
        ),
        dpi=300
    )

    plt.close()


# ==========================================================
# PROCESS
# ==========================================================

mask_files = sorted([
    f
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
])

print("\n===================================")
print("Delayed Morphological Analysis")
print("===================================\n")

for file in tqdm(mask_files):

    try:

        mask = load_binary_mask(
            os.path.join(
                MASK_DIR,
                file
            )
        )

        roi = cv2.imread(
            os.path.join(
                ROI_DIR,
                file
            ),
            cv2.IMREAD_GRAYSCALE
        )

        if roi is None:
            roi = np.zeros_like(mask)

        if roi.shape != mask.shape:

            roi = cv2.resize(
                roi,
                (
                    mask.shape[1],
                    mask.shape[0]
                )
            )

        skeleton = create_skeleton(mask)

        distance = create_distance_map(mask)

        thickness = create_thickness_map(distance)

        save_skeleton(
            file,
            skeleton
        )

        save_thickness_map(
            file,
            thickness
        )

        feature = contour_features(mask)

        if feature is None:
            continue

        stats = thickness_statistics(
            thickness,
            skeleton
        )

        save_visualization(
            file,
            roi,
            mask,
            skeleton,
            thickness
        )

        results.append({

            "Subject": file,

            "Area": feature["area"],

            "Perimeter": feature["perimeter"],

            "BoundingBoxWidth": feature["bbox_w"],

            "BoundingBoxHeight": feature["bbox_h"],

            "CentroidX": feature["cx"],

            "CentroidY": feature["cy"],

            "SkeletonLength": skeleton_length(skeleton),

            "MeanThickness": stats["mean"],

            "MaxThickness": stats["max"],

            "MinThickness": stats["min"],

            "StdThickness": stats["std"]

        })

    except Exception as e:

        print(f"Skipped {file}")

        print(e)


# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(results)

csv_path = os.path.join(
    MEASURE_DIR,
    "features.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\n===================================")
print("DELAYED MORPHOLOGICAL ANALYSIS COMPLETE")
print("===================================")
print(f"Subjects Processed : {len(df)}")
print(f"CSV Saved          : {csv_path}")
print(f"Skeletons          : {SKELETON_DIR}")
print(f"Thickness Maps     : {THICKNESS_DIR}")
print(f"Visualizations     : {VIS_DIR}")
print("===================================")