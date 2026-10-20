from pathlib import Path
import pandas as pd

ROOT = Path(".") / "33_synthetic_delayed_150"

MORPH_PATH = ROOT / "05_morphology" / "measurements" / "features.csv"
TEXTURE_PATH = ROOT / "06_texture" / "texture_features.csv"
RADIOMIC_PATH = ROOT / "07_radiomics" / "radiomic_features.csv"

OUTPUT_DIR = ROOT / "08_merged_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "synthetic_delayed_150_master_features.csv"

morph = pd.read_csv(MORPH_PATH)
texture = pd.read_csv(TEXTURE_PATH)
radiomic = pd.read_csv(RADIOMIC_PATH)

def clean_subject(x):
    x = str(x).strip()
    for ext in [".nii.gz", ".nii", ".png", ".jpg", ".jpeg"]:
        if x.lower().endswith(ext):
            x = x[:-len(ext)]
            break
    return x

for df in [morph, texture, radiomic]:
    df["Subject"] = df["Subject"].apply(clean_subject)

morph = morph.drop_duplicates(subset="Subject")
texture = texture.drop_duplicates(subset="Subject")
radiomic = radiomic.drop_duplicates(subset="Subject")

master = morph.merge(texture, on="Subject", how="inner")
master = master.merge(radiomic, on="Subject", how="inner")
master = master.loc[:, ~master.columns.duplicated()]

master.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("STAGE 8 — FEATURE MERGING COMPLETE")
print("=" * 70)
print(f"Morphology subjects : {len(morph)}")
print(f"Texture subjects    : {len(texture)}")
print(f"Radiomics subjects  : {len(radiomic)}")
print(f"Merged subjects     : {len(master)}")
print(f"Total columns       : {master.shape[1]}")
print(f"Total features      : {master.shape[1] - 1}")
print(f"Missing values      : {master.isna().sum().sum()}")
print(f"Output saved to     : {OUTPUT_PATH}")
print("=" * 70)
