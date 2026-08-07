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

ROI_DIR = r"E:\Corpus_Callosum\25_delayed_cc_extraction\roi"

MASK_DIR = r"E:\Corpus_Callosum\27_delayed_refined_masks"

OUTPUT_DIR = r"E:\Corpus_Callosum\29_delayed_texture_analysis"

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GLCM_DIR, exist_ok=True)
os.makedirs(LBP_DIR, exist_ok=True)
os.makedirs(ENTROPY_DIR, exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

GLCM_DISTANCES = [1]

GLCM_ANGLES = [
    0,
    np.pi/4,
    np.pi/2,
    3*np.pi/4
]

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

features = []

# ============================================================
# LOAD ROI
# ============================================================

def load_roi(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise Exception(f"Cannot read ROI : {path}")

    return img


# ============================================================
# LOAD MASK
# ============================================================

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


# ============================================================
# APPLY MASK
# ============================================================

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

    return cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


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

    return {
        "contrast": np.mean(graycoprops(glcm,"contrast")),
        "correlation": np.mean(graycoprops(glcm,"correlation")),
        "energy": np.mean(graycoprops(glcm,"energy")),
        "homogeneity": np.mean(graycoprops(glcm,"homogeneity")),
        "asm": np.mean(graycoprops(glcm,"ASM")),
        "dissimilarity": np.mean(graycoprops(glcm,"dissimilarity")),
        "glcm": glcm
    }
# ============================================================
# SAVE GLCM
# ============================================================

def save_glcm(glcm, filename):

    matrix = glcm[:, :, 0, 0]

    plt.figure(figsize=(5,5))

    plt.imshow(matrix, cmap="viridis")

    plt.colorbar()

    plt.title("GLCM")

    plt.tight_layout()

    plt.savefig(
        os.path.join(GLCM_DIR, filename),
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

    return {
        "lbp": lbp,
        "hist": hist,
        "mean": float(np.mean(lbp)),
        "std": float(np.std(lbp))
    }


# ============================================================
# SAVE LBP
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
        os.path.join(LBP_DIR, filename),
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

    return float(
        entropy(hist + 1e-12, base=2)
    )


# ============================================================
# SAVE ENTROPY MAP
# ============================================================

def save_entropy_map(image, filename):

    image = image.astype(np.float32)

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

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
    ).astype(np.uint8)

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
# PROCESS SUBJECT
# ============================================================

def process_subject(filename):

    roi = load_roi(
        os.path.join(ROI_DIR, filename)
    )

    mask = load_mask(
        os.path.join(MASK_DIR, filename)
    )

    masked_roi = apply_mask(
        roi,
        mask
    )

    glcm = compute_glcm_features(masked_roi)

    save_glcm(glcm["glcm"], filename)

    lbp = compute_lbp_features(masked_roi)

    save_lbp(lbp["lbp"], filename)

    entropy_value = compute_entropy(masked_roi)

    save_entropy_map(masked_roi, filename)

    return {
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("DELAYED TEXTURE ANALYSIS")
    print("========================================\n")

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

            features.append(
                process_subject(file)
            )

            success += 1

        except Exception as e:

            print(f"\nSkipped {file}")
            print(e)

            failed += 1

    df = pd.DataFrame(features)

    df = df.sort_values("Subject")

    df.to_csv(
        CSV_PATH,
        index=False
    )

    print("\n========================================")
    print("DELAYED TEXTURE ANALYSIS COMPLETE")
    print("========================================")
    print(f"Processed : {success}")
    print(f"Failed    : {failed}")
    print(f"CSV       : {CSV_PATH}")
    print(f"GLCM      : {GLCM_DIR}")
    print(f"LBP       : {LBP_DIR}")
    print(f"Entropy   : {ENTROPY_DIR}")
    print("========================================")


if __name__ == "__main__":
    main()
    