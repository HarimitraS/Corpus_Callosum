from pathlib import Path
import pandas as pd
import numpy as np


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

RESULTS_PATH = (
    ROOT / "statistical_results.csv"
)

DATASET_PATH = (
    ROOT / "final_statistical_dataset.csv"
)

OUTPUT_DIR = (
    ROOT / "feature_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_selected_features.csv"
)


# ==========================================================
# PARAMETERS
# ==========================================================

FDR_THRESHOLD = 0.05

CORRELATION_THRESHOLD = 0.90

MAX_FEATURES = 20


# ==========================================================
# LOAD
# ==========================================================

results = pd.read_csv(
    RESULTS_PATH
)

df = pd.read_csv(
    DATASET_PATH
)


# ==========================================================
# SIGNIFICANT FEATURES
# ==========================================================

significant = results[
    results["Adjusted_P"] < FDR_THRESHOLD
].copy()

significant[
    "Absolute_Effect_Size"
] = significant[
    "Effect_Size"
].abs()


# ==========================================================
# SORT BY EFFECT SIZE
# ==========================================================

significant = significant.sort_values(
    "Absolute_Effect_Size",
    ascending=False
)


# ==========================================================
# AVAILABLE FEATURES
# ==========================================================

available = [
    feature
    for feature in significant["Feature"]
    if feature in df.columns
]


feature_data = df[
    available
].apply(
    pd.to_numeric,
    errors="coerce"
)


# ==========================================================
# CORRELATION MATRIX
# ==========================================================

correlation = feature_data.corr(
    method="spearman"
).abs()


# ==========================================================
# GREEDY FEATURE SELECTION
# ==========================================================

selected = []

for feature in significant["Feature"]:

    if feature not in correlation.columns:
        continue

    if len(selected) == 0:

        selected.append(
            feature
        )

        continue

    too_correlated = False

    for chosen in selected:

        corr_value = correlation.loc[
            feature,
            chosen
        ]

        if pd.notna(corr_value):

            if corr_value >= CORRELATION_THRESHOLD:

                too_correlated = True

                break

    if not too_correlated:

        selected.append(
            feature
        )

    if len(selected) >= MAX_FEATURES:

        break


# ==========================================================
# FINAL TABLE
# ==========================================================

final = significant[
    significant["Feature"].isin(
        selected
    )
].copy()

final["Absolute_Effect_Size"] = (
    final["Effect_Size"].abs()
)

final = final.sort_values(
    "Absolute_Effect_Size",
    ascending=False
)

final.insert(
    0,
    "Rank",
    range(
        1,
        len(final) + 1
    )
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
print("FINAL FEATURE REDUCTION")
print("==========================================")

print(
    "Significant candidates :",
    len(significant)
)

print(
    "Correlation threshold  :",
    CORRELATION_THRESHOLD
)

print(
    "Final selected         :",
    len(final)
)

print("\nSelected features:\n")

print(
    final[
        [
            "Rank",
            "Feature",
            "P_Value",
            "Adjusted_P",
            "Effect_Size",
            "Absolute_Effect_Size"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)

print("==========================================")