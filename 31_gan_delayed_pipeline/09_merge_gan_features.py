from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

MORPH_PATH = (
    ROOT
    / "06_morphology"
    / "measurements"
    / "features.csv"
)

TEXTURE_PATH = (
    ROOT
    / "07_texture_analysis"
    / "texture_features.csv"
)

RADIOMIC_PATH = (
    ROOT
    / "08_radiomics"
    / "radiomic_features.csv"
)

OUTPUT_DIR = ROOT / "09_merged_features"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "gan_delayed_master_features.csv"
)

# ==========================================================
# LOAD
# ==========================================================

morph = pd.read_csv(
    MORPH_PATH
)

texture = pd.read_csv(
    TEXTURE_PATH
)

radiomic = pd.read_csv(
    RADIOMIC_PATH
)

# ==========================================================
# STANDARDIZE SUBJECT IDs
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


for df in [
    morph,
    texture,
    radiomic
]:

    if "Subject" not in df.columns:

        raise ValueError(
            "Subject column missing."
        )

    df["Subject"] = (
        df["Subject"]
        .apply(clean_subject)
    )

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

morph = morph.drop_duplicates(
    subset="Subject"
)

texture = texture.drop_duplicates(
    subset="Subject"
)

radiomic = radiomic.drop_duplicates(
    subset="Subject"
)

# ==========================================================
# MERGE
# ==========================================================

master = morph.merge(
    texture,
    on="Subject",
    how="inner"
)

master = master.merge(
    radiomic,
    on="Subject",
    how="inner"
)

# ==========================================================
# REMOVE DUPLICATE COLUMNS
# ==========================================================

master = master.loc[
    :,
    ~master.columns.duplicated()
]

# ==========================================================
# SAVE
# ==========================================================

master.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("GAN DELAYED FEATURE MERGE COMPLETE")
print("==========================================")

print(
    f"Morphology subjects : {len(morph)}"
)

print(
    f"Texture subjects    : {len(texture)}"
)

print(
    f"Radiomic subjects   : {len(radiomic)}"
)

print(
    f"Merged subjects     : {len(master)}"
)

print(
    f"Total features      : {master.shape[1] - 1}"
)

print(
    f"Saved to            : {OUTPUT_PATH}"
)

print("==========================================")