from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "31_gan_delayed_pipeline"
    / "10_final_dataset"
    / "final_dataset_with_gan.csv"
)

OUTPUT_DIR = (
    ROOT
    / "22_statistical_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_statistical_dataset.csv"
)

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(
    INPUT_PATH
)

# ==========================================================
# KEEP ONLY NORMAL / DELAYED
# ==========================================================

df = df[
    df["myelinisation"].isin(
        ["normal", "delayed"]
    )
].copy()

# ==========================================================
# REMOVE NON-FEATURE METADATA
# ==========================================================

metadata = [
    "Subject",
    "myelinisation",
    "diagnosis",
    "group"
]

# Keep age-related columns out of statistical feature testing
# because GAN-generated subjects do not have real patient age.
metadata.extend([
    "age",
    "age_corrected",
    "doctor_predicted_age"
])

feature_columns = [
    col
    for col in df.columns
    if col not in metadata
]

# ==========================================================
# KEEP NUMERIC FEATURES ONLY
# ==========================================================

numeric_features = [
    col
    for col in feature_columns
    if pd.api.types.is_numeric_dtype(
        df[col]
    )
]

keep_columns = (
    metadata
    + numeric_features
)

df = df[
    [
        col
        for col in keep_columns
        if col in df.columns
    ]
]

# ==========================================================
# REMOVE COMPLETELY EMPTY FEATURES
# ==========================================================

numeric_cols = [
    col
    for col in numeric_features
    if col in df.columns
]

empty_features = [
    col
    for col in numeric_cols
    if df[col].notna().sum() == 0
]

df = df.drop(
    columns=empty_features
)

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("STATISTICAL DATASET PREPARED")
print("==========================================")

print(
    f"Rows              : {len(df)}"
)

print(
    f"Normal            : "
    f"{(df['myelinisation'] == 'normal').sum()}"
)

print(
    f"Delayed           : "
    f"{(df['myelinisation'] == 'delayed').sum()}"
)

print(
    f"Numeric features  : "
    f"{len([c for c in df.columns if c not in metadata])}"
)

print(
    f"Empty features removed : "
    f"{len(empty_features)}"
)

print(
    f"\nSaved to : {OUTPUT_PATH}"
)

print("==========================================")