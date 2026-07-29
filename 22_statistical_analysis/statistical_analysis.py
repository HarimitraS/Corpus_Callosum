from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "22_statistical_analysis" / "cleaned_master_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "22_statistical_analysis"

ALL_RESULTS = OUTPUT_DIR / "statistical_results.csv"
SIGNIFICANT_RESULTS = OUTPUT_DIR / "significant_features.csv"

# ==========================================================
# Load
# ==========================================================

df = pd.read_csv(INPUT_PATH)

# ==========================================================
# Binary labels
# ==========================================================

label_column = "myelinisation"

df = df[df[label_column].isin(["normal", "delayed"])].copy()

normal = df[df[label_column] == "normal"]
delayed = df[df[label_column] == "delayed"]

# ==========================================================
# Features to analyse
# ==========================================================

exclude = [
    "Subject",
    "myelinisation",
    "diagnosis",
    "group"
]

features = []

for col in df.columns:
    if col in exclude:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        features.append(col)

# ==========================================================
# Statistical Analysis
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

        effect = (2 * u_stat) / (n1 * n2) - 1

        results.append({

            "Feature": feature,

            "Normal_Mean": x.mean(),
            "Delayed_Mean": y.mean(),

            "U": u_stat,
            "P_Value": p,
            "Effect_Size": effect

        })

    except Exception:
        continue

results = pd.DataFrame(results)

# ==========================================================
# Benjamini-Hochberg FDR
# ==========================================================

reject, p_adj, _, _ = multipletests(
    results["P_Value"],
    alpha=0.05,
    method="fdr_bh"
)

results["Adjusted_P"] = p_adj
results["Significant"] = reject

# ==========================================================
# Sort
# ==========================================================

results = results.sort_values(
    "Adjusted_P"
)

significant = results[
    results["Significant"] == True
]

# ==========================================================
# Save
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
# Summary
# ==========================================================

print("=" * 60)
print("Statistical Analysis Complete")
print("=" * 60)

print("Total Features      :", len(features))
print("Analysed Features   :", len(results))
print("Significant Features:", len(significant))

print("\nSaved:")
print(ALL_RESULTS)
print(SIGNIFICANT_RESULTS)