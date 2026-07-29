from pathlib import Path
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MORPH_PATH = PROJECT_ROOT / "16_measurements" / "features.csv"
TEXTURE_PATH = PROJECT_ROOT / "20_texture_analysis" / "texture_features.csv"
RADIOMIC_PATH = PROJECT_ROOT / "21_radiomic_features" / "radiomic_features.csv"
META_PATH = PROJECT_ROOT / "meta.csv"

OUTPUT_DIR = PROJECT_ROOT / "22_statistical_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "master_dataset.csv"

# ==========================================================
# Load CSVs
# ==========================================================

morph = pd.read_csv(MORPH_PATH)
texture = pd.read_csv(TEXTURE_PATH)
radiomic = pd.read_csv(RADIOMIC_PATH)

meta = pd.read_csv(
    META_PATH,
    sep=";",
    engine="python"
)

# ==========================================================
# Detect Subject ID column
# ==========================================================

def find_subject_column(df):
    candidates = [
        "Subject",
        "subject",
        "Subject_ID",
        "subject_id",
        "image_id",
        "ID",
        "id",
        "Filename",
        "filename",
        "Image",
        "image"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find subject identifier.\nAvailable columns:\n{list(df.columns)}"
    )


morph_id = find_subject_column(morph)
texture_id = find_subject_column(texture)
radiomic_id = find_subject_column(radiomic)
meta_id = find_subject_column(meta)

# ==========================================================
# Rename ID columns
# ==========================================================

morph = morph.rename(columns={morph_id: "Subject"})
texture = texture.rename(columns={texture_id: "Subject"})
radiomic = radiomic.rename(columns={radiomic_id: "Subject"})
meta = meta.rename(columns={meta_id: "Subject"})

# ==========================================================
# Standardize Subject IDs
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


for df in [morph, texture, radiomic, meta]:
    df["Subject"] = df["Subject"].apply(clean_subject)

# ==========================================================
# Remove duplicate subjects
# ==========================================================

morph = morph.drop_duplicates(subset="Subject")
texture = texture.drop_duplicates(subset="Subject")
radiomic = radiomic.drop_duplicates(subset="Subject")
meta = meta.drop_duplicates(subset="Subject")

# ==========================================================
# Merge datasets
# ==========================================================

master = morph.merge(texture, on="Subject", how="inner")

master = master.merge(
    radiomic,
    on="Subject",
    how="inner"
)

master = master.merge(
    meta,
    on="Subject",
    how="inner"
)

# ==========================================================
# Summary
# ==========================================================

print("\nRows in each dataset")
print("------------------------------")
print("Morphology :", len(morph))
print("Texture    :", len(texture))
print("Radiomics  :", len(radiomic))
print("Metadata   :", len(meta))
print("Merged     :", len(master))
print("------------------------------")

# ==========================================================
# Save
# ==========================================================

master.to_csv(OUTPUT_PATH, index=False)

print("\n============================================================")
print("Master dataset created successfully.")
print(f"Subjects : {len(master)}")
print(f"Features : {master.shape[1] - 1}")
print(f"Saved to : {OUTPUT_PATH}")
print("============================================================")