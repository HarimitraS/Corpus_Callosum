from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "22_statistical_analysis" / "master_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "22_statistical_analysis" / "cleaned_master_dataset.csv"

# ==========================================================
# Load
# ==========================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("Original Dataset")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])
print("=" * 60)

# ==========================================================
# Remove duplicate columns
# ==========================================================

df = df.loc[:, ~df.columns.duplicated()]

# ==========================================================
# Replace infinite values
# ==========================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================================
# Remove constant features
# ==========================================================

exclude = [
    "Subject",
    "myelinisation",
    "diagnosis",
    "group"
]

constant_cols = []

for col in df.columns:
    if col in exclude:
        continue

    if df[col].nunique(dropna=False) <= 1:
        constant_cols.append(col)

df.drop(columns=constant_cols, inplace=True)

# ==========================================================
# Remove all-null columns
# ==========================================================

df.dropna(axis=1, how="all", inplace=True)

# ==========================================================
# Remove columns with >30% missing values
# ==========================================================

missing_ratio = df.isnull().mean()

drop_cols = missing_ratio[missing_ratio > 0.30].index.tolist()

drop_cols = [c for c in drop_cols if c not in exclude]

df.drop(columns=drop_cols, inplace=True)

# ==========================================================
# Fill remaining missing values
# ==========================================================

for col in df.columns:

    if col in exclude:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================================
# Save
# ==========================================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaning Summary")
print("----------------------------")
print("Constant columns removed :", len(constant_cols))
print("High-missing columns removed :", len(drop_cols))
print("----------------------------")
print("Final Rows    :", df.shape[0])
print("Final Columns :", df.shape[1])
print("\nSaved to")
print(OUTPUT_PATH)