"""
=============================================================
Radiomic Feature Extraction
Corpus Callosum Biomarker Discovery

Author : Kanishk Upadhyay

Uses:
    - 4_sagittal_slices
    - 12_refined_masks
    - PyRadiomics

Output:
    radiomic_features.csv
=============================================================
"""

import os
import cv2
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm
from radiomics import featureextractor

# ==========================================================
# PROJECT PATHS
# ==========================================================

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folders
IMAGE_DIR = os.path.join(CURRENT_DIR, "4_sagittal_slices")
MASK_DIR = os.path.join(CURRENT_DIR, "12_refined_masks")

# Output folder
OUTPUT_DIR = os.path.join(CURRENT_DIR, "21_radiomic_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output files
CSV_PATH = os.path.join(OUTPUT_DIR, "radiomic_features.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "radiomics.log")

# PyRadiomics parameter file
# (params.yaml is in the project root beside this script)
PARAM_FILE = os.path.join(CURRENT_DIR, "params.yaml")

# ==========================================================
# LOGGER
# ==========================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ==========================================================
# PARAMETER FILE
# ==========================================================

PARAMETER_TEXT = """
imageType:

  Original: {}

setting:

  normalize: true

  normalizeScale: 100

  binWidth: 25

  label: 1

  force2D: true

  force2Ddimension: 0

featureClass:

  firstorder:

  shape2D:

  glcm:
"""

# ==========================================================
# CREATE PARAMETER FILE
# ==========================================================

def create_parameter_file():

    if os.path.exists(PARAM_FILE):
        return

    with open(PARAM_FILE, "w") as f:
        f.write(PARAMETER_TEXT)

# ==========================================================
# INITIALIZE PYRADIOMICS
# ==========================================================
print("PARAM_FILE:", PARAM_FILE)
print("Exists:", os.path.exists(PARAM_FILE))
def initialize_extractor():

    create_parameter_file()

    extractor = featureextractor.RadiomicsFeatureExtractor(
        PARAM_FILE
    )

    extractor.enableAllImageTypes()

    extractor.enableFeatureClassByName(
        "firstorder"
    )

    extractor.enableFeatureClassByName(
        "shape2D"
    )

    extractor.enableFeatureClassByName(
        "glcm"
    )

    return extractor

# ==========================================================
# LOAD IMAGE
# ==========================================================

def load_image(subject):

    image_path = os.path.join(
        IMAGE_DIR,
        subject + ".png"
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(image_path)

    return image

# ==========================================================
# LOAD MASK
# ==========================================================

def load_mask(subject):

    mask_path = os.path.join(
        MASK_DIR,
        subject + ".png"
    )

    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(mask_path)

    _, mask = cv2.threshold(
        mask,
        127,
        1,
        cv2.THRESH_BINARY
    )

    return mask

# ==========================================================
# NUMPY -> SIMPLEITK
# ==========================================================

def convert_to_sitk(array):

    img = sitk.GetImageFromArray(
        array.astype(np.float32)
    )

    img.SetSpacing((1.0,1.0))

    return img

# ==========================================================
# PREPARE IMAGE
# ==========================================================

def prepare(subject):

    image = load_image(subject)

    mask = load_mask(subject)

    if image.shape != mask.shape:

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    image_itk = convert_to_sitk(image)

    mask_itk = convert_to_sitk(mask)

    return image_itk, mask_itk
# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(extractor, image_itk, mask_itk):

    feature_vector = extractor.execute(
        image_itk,
        mask_itk
    )

    features = {}

    for key, value in feature_vector.items():

        # Skip diagnostic information
        if key.startswith("diagnostics"):
            continue

        # Convert numpy values to Python scalars
        if isinstance(value, np.ndarray):

            if value.size == 1:
                value = float(value)

            else:
                continue

        try:

            value = float(value)

        except Exception:
            continue

        # Remove the long PyRadiomics prefix
        key = key.replace(
            "original_",
            ""
        )

        features[key] = value

    return features


# ==========================================================
# PROCESS ONE SUBJECT
# ==========================================================

def process_subject(subject, extractor):

    image_itk, mask_itk = prepare(subject)

    feature_dict = extract_features(

        extractor,

        image_itk,

        mask_itk

    )

    feature_dict["Subject"] = subject

    return feature_dict


# ==========================================================
# SUBJECT LIST
# ==========================================================

def get_subjects():

    subjects = []

    for file in os.listdir(MASK_DIR):

        if file.lower().endswith(".png"):

            subject = os.path.splitext(file)[0]

            image_path = os.path.join(

                IMAGE_DIR,

                subject + ".png"

            )

            if os.path.exists(image_path):

                subjects.append(subject)

    subjects.sort()

    return subjects


# ==========================================================
# SAVE CSV
# ==========================================================

def save_dataframe(rows):

    df = pd.DataFrame(rows)

    if "Subject" in df.columns:

        cols = ["Subject"]

        cols.extend(

            [

                c

                for c in df.columns

                if c != "Subject"

            ]

        )

        df = df[cols]

    df.to_csv(

        CSV_PATH,

        index=False

    )

    return df
# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    print("\n==========================================")
    print(" Corpus Callosum Radiomic Feature Extraction")
    print("==========================================\n")

    extractor = initialize_extractor()

    subjects = get_subjects()

    print(f"Found {len(subjects)} subjects.\n")

    rows = []

    success = 0
    failed = 0

    for subject in tqdm(subjects):

        try:

            feature_row = process_subject(

                subject,

                extractor

            )

            rows.append(feature_row)

            success += 1

        except Exception as e:

            logging.exception(subject)

            print(f"\nFailed : {subject}")

            print(e)

            failed += 1

    if len(rows) == 0:

        print("No radiomic features extracted.")

        return

    df = save_dataframe(rows)

    print("\n==========================================")
    print("Radiomic Feature Extraction Complete")
    print("==========================================")

    print(f"Subjects Processed : {success}")
    print(f"Failed Subjects    : {failed}")
    print(f"Features Extracted : {len(df.columns)-1}")
    print(f"CSV Saved          : {CSV_PATH}")

    print("==========================================\n")


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":

    main()
