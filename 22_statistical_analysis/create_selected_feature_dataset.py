from pathlib import Path
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    ROOT / "final_statistical_dataset.csv"
)

FEATURE_PATH = (
    ROOT
    / "feature_selection"
    / "final_selected_features.csv"
)

OUTPUT_PATH = (
    ROOT
    / "feature_selection"
    / "selected_feature_dataset.csv"
)

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(
    DATASET_PATH
)

selected = pd.read_csv(
    FEATURE_PATH
)

features = selected[
    "Feature"
].tolist()

# ==========================================================
# KEEP LABEL + SELECTED FEATURES
# ==========================================================

columns = [
    "Subject",
    "myelinisation"
] + features

columns = [
    c for c in columns
    if c in df.columns
]

final_df = df[
    columns
].copy()

# ==========================================================
# SAVE
# ==========================================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("SELECTED FEATURE DATASET CREATED")
print("==========================================")

print(
    "Rows     :",
    len(final_df)
)

print(
    "Features :",
    len(features)
)

print(
    "\nClass distribution:"
)

print(
    final_df[
        "myelinisation"
    ].value_counts()
)

print(
    "\nSelected features:"
)

for i, feature in enumerate(
    features,
    1
):

    print(
        f"{i:2d}. {feature}"
    )

print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)

print("==========================================")