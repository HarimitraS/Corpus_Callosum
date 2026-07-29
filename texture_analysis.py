import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.stats import entropy

from skimage.feature import (
    graycomatrix,
    graycoprops,
    local_binary_pattern
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROI_DIR = os.path.join(
    BASE_DIR,
    "8_roi_2"
)

MASK_DIR = os.path.join(
    BASE_DIR,
    "12_refined_masks"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "20_texture_analysis"
)

GLCM_DIR = os.path.join(
    OUTPUT_DIR,
    "glcm_images"
)

LBP_DIR = os.path.join(
    OUTPUT_DIR,
    "lbp_images"
)

ENTROPY_DIR = os.path.join(
    OUTPUT_DIR,
    "entropy_maps"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "texture_features.csv"
)

os.makedirs(GLCM_DIR, exist_ok=True)
os.makedirs(LBP_DIR, exist_ok=True)
os.makedirs(ENTROPY_DIR, exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

GLCM_DISTANCES = [1]

GLCM_ANGLES = [
    0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4
]

LBP_RADIUS = 3

LBP_POINTS = 8 * LBP_RADIUS

# ============================================================
# FEATURE STORAGE
# ============================================================

features = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_roi(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise Exception(f"Cannot read ROI : {path}")

    return img


def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise Exception(f"Cannot read Mask : {path}")

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mask


def apply_mask(image, mask):

    if image.shape != mask.shape:

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    result = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return result


# ============================================================
# GLCM FEATURES
# ============================================================

def compute_glcm_features(image):

    image = image.astype(np.uint8)

    glcm = graycomatrix(

        image,

        distances=GLCM_DISTANCES,

        angles=GLCM_ANGLES,

        symmetric=True,

        normed=True

    )

    contrast = np.mean(
        graycoprops(glcm, "contrast")
    )

    correlation = np.mean(
        graycoprops(glcm, "correlation")
    )

    energy = np.mean(
        graycoprops(glcm, "energy")
    )

    homogeneity = np.mean(
        graycoprops(glcm, "homogeneity")
    )

    asm = np.mean(
        graycoprops(glcm, "ASM")
    )

    dissimilarity = np.mean(
        graycoprops(glcm, "dissimilarity")
    )

    return {

        "contrast": contrast,

        "correlation": correlation,

        "energy": energy,

        "homogeneity": homogeneity,

        "asm": asm,

        "dissimilarity": dissimilarity,

        "glcm": glcm

    }


# ============================================================
# SAVE GLCM
# ============================================================

def save_glcm(glcm, filename):

    matrix = glcm[:, :, 0, 0]

    plt.figure(figsize=(5,5))

    plt.imshow(
        matrix,
        cmap="viridis"
    )

    plt.colorbar()

    plt.title("GLCM")

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            GLCM_DIR,
            filename
        ),

        dpi=300

    )

    plt.close()
# ============================================================
# LBP FEATURES
# ============================================================

def compute_lbp_features(image):

    lbp = local_binary_pattern(

        image,

        P=LBP_POINTS,

        R=LBP_RADIUS,

        method="uniform"

    )

    hist, _ = np.histogram(

        lbp.ravel(),

        bins=np.arange(0, LBP_POINTS + 3),

        range=(0, LBP_POINTS + 2),

        density=True

    )

    lbp_mean = float(np.mean(lbp))

    lbp_std = float(np.std(lbp))

    return {

        "lbp": lbp,

        "hist": hist,

        "mean": lbp_mean,

        "std": lbp_std

    }


# ============================================================
# SAVE LBP IMAGE
# ============================================================

def save_lbp(lbp, filename):

    lbp_norm = cv2.normalize(

        lbp,

        None,

        0,

        255,

        cv2.NORM_MINMAX

    ).astype(np.uint8)

    cv2.imwrite(

        os.path.join(
            LBP_DIR,
            filename
        ),

        lbp_norm

    )


# ============================================================
# ENTROPY
# ============================================================

def compute_entropy(image):

    hist = cv2.calcHist(

        [image],

        [0],

        None,

        [256],

        [0,256]

    )

    hist = hist.flatten()

    hist = hist / np.sum(hist)

    value = entropy(

        hist + 1e-12,

        base=2

    )

    return float(value)


# ============================================================
# ENTROPY MAP
# ============================================================

def save_entropy_map(image, filename):

    image = image.astype(np.float32)

    image = cv2.normalize(

        image,

        None,

        0,

        255,

        cv2.NORM_MINMAX

    )

    image = image.astype(np.uint8)

    entropy_map = cv2.Laplacian(

        image,

        cv2.CV_64F

    )

    entropy_map = np.abs(entropy_map)

    entropy_map = cv2.normalize(

        entropy_map,

        None,

        0,

        255,

        cv2.NORM_MINMAX

    )

    entropy_map = entropy_map.astype(np.uint8)

    colored = cv2.applyColorMap(

        entropy_map,

        cv2.COLORMAP_JET

    )

    cv2.imwrite(

        os.path.join(
            ENTROPY_DIR,
            filename
        ),

        colored

    )


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(filename):

    roi_path = os.path.join(

        ROI_DIR,

        filename

    )

    mask_path = os.path.join(

        MASK_DIR,

        filename

    )

    roi = load_roi(roi_path)

    mask = load_mask(mask_path)

    masked_roi = apply_mask(

        roi,

        mask

    )

    glcm = compute_glcm_features(

        masked_roi

    )

    save_glcm(

        glcm["glcm"],

        filename

    )

    lbp = compute_lbp_features(

        masked_roi

    )

    save_lbp(

        lbp["lbp"],

        filename

    )

    entropy_value = compute_entropy(

        masked_roi

    )

    save_entropy_map(

        masked_roi,

        filename

    )

    row = {

        "Subject": filename,

        "Contrast": glcm["contrast"],

        "Correlation": glcm["correlation"],

        "Energy": glcm["energy"],

        "Homogeneity": glcm["homogeneity"],

        "ASM": glcm["asm"],

        "Dissimilarity": glcm["dissimilarity"],

        "Entropy": entropy_value,

        "LBP_Mean": lbp["mean"],

        "LBP_STD": lbp["std"]

    }

    return row
# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("Corpus Callosum Texture Analysis")
    print("========================================\n")

    if not os.path.exists(ROI_DIR):
        raise FileNotFoundError(f"ROI folder not found:\n{ROI_DIR}")

    if not os.path.exists(MASK_DIR):
        raise FileNotFoundError(f"Mask folder not found:\n{MASK_DIR}")

    files = sorted([

        f

        for f in os.listdir(MASK_DIR)

        if f.lower().endswith(".png")

    ])

    print(f"Found {len(files)} masks.\n")

    success = 0

    failed = 0

    for file in tqdm(files):

        try:

            row = process_subject(file)

            features.append(row)

            success += 1

        except Exception as e:

            print(f"\nSkipped {file}")

            print(e)

            failed += 1

    df = pd.DataFrame(features)

    df = df.sort_values(

        by="Subject"

    )

    df.to_csv(

        CSV_PATH,

        index=False

    )

    print("\n========================================")
    print("Texture Analysis Complete")
    print("========================================")

    print(f"Subjects Processed : {success}")

    print(f"Failed             : {failed}")

    print(f"CSV Saved          : {CSV_PATH}")

    print(f"GLCM Images        : {GLCM_DIR}")

    print(f"LBP Images         : {LBP_DIR}")

    print(f"Entropy Maps       : {ENTROPY_DIR}")

    print("========================================\n")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()