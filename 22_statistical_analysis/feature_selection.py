from pathlib import Path
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    ROOT / "statistical_results.csv"
)

OUTPUT_DIR = ROOT / "feature_selection"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(
    INPUT_PATH
)


# ==========================================================
# SIGNIFICANT FEATURES
# ==========================================================

significant = df[
    df["Significant"] == True
].copy()


# ==========================================================
# EFFECT SIZE
# ==========================================================

significant["Absolute_Effect_Size"] = (
    significant["Effect_Size"]
    .abs()
)


# ==========================================================
# RANK BY ADJUSTED P
# ==========================================================

by_p = significant.sort_values(
    "Adjusted_P"
)

by_p.to_csv(
    OUTPUT_DIR / "significant_ranked_by_p.csv",
    index=False
)


# ==========================================================
# RANK BY EFFECT SIZE
# ==========================================================

by_effect = significant.sort_values(
    "Absolute_Effect_Size",
    ascending=False
)

by_effect.to_csv(
    OUTPUT_DIR / "significant_ranked_by_effect.csv",
    index=False
)


# ==========================================================
# TOP 20
# ==========================================================

top20 = by_effect.head(20)

top20.to_csv(
    OUTPUT_DIR / "top_20_features.csv",
    index=False
)


# ==========================================================
# TOP 50
# ==========================================================

top50 = by_effect.head(50)

top50.to_csv(
    OUTPUT_DIR / "top_50_features.csv",
    index=False
)


# ==========================================================
# TOP 100
# ==========================================================

top100 = by_effect.head(100)

top100.to_csv(
    OUTPUT_DIR / "top_100_features.csv",
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n==========================================")
print("FEATURE SELECTION COMPLETE")
print("==========================================")

print(
    "Total tested       :",
    len(df)
)

print(
    "Significant        :",
    len(significant)
)

print(
    "Top 20             :",
    len(top20)
)

print(
    "Top 50             :",
    len(top50)
)

print(
    "Top 100            :",
    len(top100)
)

print("\nTop 20 by absolute effect size:\n")

print(
    top20[
        [
            "Feature",
            "P_Value",
            "Adjusted_P",
            "Effect_Size",
            "Absolute_Effect_Size"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")

print(
    OUTPUT_DIR
)

print("==========================================")