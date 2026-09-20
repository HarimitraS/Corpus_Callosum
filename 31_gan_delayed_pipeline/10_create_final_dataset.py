from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = (
    ROOT
    / "22_statistical_analysis"
    / "cleaned_master_dataset.csv"
)

GAN_PATH = (
    ROOT
    / "31_gan_delayed_pipeline"
    / "09_merged_features"
    / "gan_delayed_master_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "31_gan_delayed_pipeline"
    / "10_final_dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_dataset_with_gan.csv"
)

# ==========================================================
# LOAD
# ==========================================================

original = pd.read_csv(
    ORIGINAL_PATH
)

gan = pd.read_csv(
    GAN_PATH
)

# ==========================================================
# STANDARDIZE SUBJECT
# ==========================================================

def clean_subject(x):

    x = str(x).strip()

    extensions = [
        ".nii.gz",
        ".nii",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    for ext in extensions:

        if x.lower().endswith(ext):

            x = x[:-len(ext)]

            break

    return x


original["Subject"] = (
    original["Subject"]
    .apply(clean_subject)
)

gan["Subject"] = (
    gan["Subject"]
    .apply(clean_subject)
)

# ==========================================================
# REMOVE DUPLICATE SUBJECTS
# ==========================================================

original = original.drop_duplicates(
    subset="Subject"
)

gan = gan.drop_duplicates(
    subset="Subject"
)

# ==========================================================
# LABEL GAN DATA
# ==========================================================

gan["myelinisation"] = "delayed"

# GAN images do not have real patient metadata.
gan["age"] = pd.NA
gan["age_corrected"] = pd.NA
gan["doctor_predicted_age"] = pd.NA
gan["diagnosis"] = pd.NA
gan["group"] = pd.NA

# ==========================================================
# ALIGN COLUMNS
# ==========================================================

all_columns = list(
    dict.fromkeys(
        list(original.columns)
        + list(gan.columns)
    )
)

original = original.reindex(
    columns=all_columns
)

gan = gan.reindex(
    columns=all_columns
)

# ==========================================================
# COMBINE
# ==========================================================

final = pd.concat(
    [
        original,
        gan
    ],
    ignore_index=True
)

# ==========================================================
# REMOVE DUPLICATE SUBJECTS
# ==========================================================

final = final.drop_duplicates(
    subset="Subject",
    keep="first"
)

# ==========================================================
# SAVE
# ==========================================================

final.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("FINAL DATASET CREATED")
print("==========================================")

print(
    f"Original rows       : {len(original)}"
)

print(
    f"GAN delayed rows    : {len(gan)}"
)

print(
    f"Final rows          : {len(final)}"
)

print(
    f"Final columns       : {len(final.columns)}"
)

print("\nClass distribution:")

print(
    final["myelinisation"]
    .value_counts(
        dropna=False
    )
)

print(
    f"\nSaved to : {OUTPUT_PATH}"
)

print("==========================================")