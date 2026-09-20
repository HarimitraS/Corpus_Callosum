from pathlib import Path

import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "22_statistical_analysis"
    / "final_statistical_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "22_statistical_analysis"
)

ALL_RESULTS = (
    OUTPUT_DIR
    / "statistical_results.csv"
)

SIGNIFICANT_RESULTS = (
    OUTPUT_DIR
    / "significant_features.csv"
)


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(INPUT_PATH)


# ==========================================================
# BINARY LABELS
# ==========================================================

label_column = "myelinisation"

df = df[
    df[label_column].isin(
        ["normal", "delayed"]
    )
].copy()

normal = df[
    df[label_column] == "normal"
]

delayed = df[
    df[label_column] == "delayed"
]


# ==========================================================
# FEATURES TO ANALYSE
# ==========================================================

exclude = [
    "Subject",
    "myelinisation",
    "diagnosis",
    "group",
    "age",
    "age_corrected",
    "doctor_predicted_age"
]

features = []

for col in df.columns:

    if col in exclude:
        continue

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):
        features.append(col)


# ==========================================================
# STATISTICAL ANALYSIS
# ==========================================================

results = []

for feature in features:

    x = normal[feature].dropna()
    y = delayed[feature].dropna()

    if len(x) < 3 or len(y) < 3:
        continue

    try:

        u_stat, p = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        n1 = len(x)
        n2 = len(y)

        effect = (
            (2 * u_stat)
            / (n1 * n2)
        ) - 1

        results.append({

            "Feature": feature,

            "Normal_N": n1,

            "Delayed_N": n2,

            "Normal_Mean": x.mean(),

            "Delayed_Mean": y.mean(),

            "U": u_stat,

            "P_Value": p,

            "Effect_Size": effect

        })

    except Exception:
        continue


results = pd.DataFrame(
    results
)


# ==========================================================
# CHECK
# ==========================================================

if results.empty:

    print(
        "============================================================"
    )

    print(
        "No valid features available for statistical analysis."
    )

    print(
        "============================================================"
    )

    exit()


# ==========================================================
# BENJAMINI-HOCHBERG FDR
# ==========================================================

reject, p_adj, _, _ = multipletests(
    results["P_Value"],
    alpha=0.05,
    method="fdr_bh"
)

results["Adjusted_P"] = p_adj

results["Significant"] = reject


# ==========================================================
# SORT
# ==========================================================

results = results.sort_values(
    "Adjusted_P"
)

significant = results[
    results["Significant"] == True
]


# ==========================================================
# SAVE
# ==========================================================

results.to_csv(
    ALL_RESULTS,
    index=False
)

significant.to_csv(
    SIGNIFICANT_RESULTS,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print(
    "============================================================"
)

print(
    "Statistical Analysis Complete"
)

print(
    "============================================================"
)

print(
    "Total Features       :",
    len(features)
)

print(
    "Analysed Features    :",
    len(results)
)

print(
    "Significant Features :",
    len(significant)
)

print(
    "\nGroup Sizes:"
)

print(
    "Normal               :",
    len(normal)
)

print(
    "Delayed              :",
    len(delayed)
)

print(
    "\nSaved:"
)

print(
    ALL_RESULTS
)

print(
    SIGNIFICANT_RESULTS
)

print(
    "============================================================"
)