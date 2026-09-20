from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# GAN SAMPLE VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FINAL_DATASET = (
    BASE_DIR
    / "feature_selection"
    / "selected_feature_dataset.csv"
)

GAN_FEATURES = (
    PROJECT_DIR
    / "31_gan_delayed_pipeline"
    / "09_merged_features"
    / "gan_delayed_master_features.csv"
)

OUTPUT_DIR = BASE_DIR / "gan_sample_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "gan_sample_summary.csv"
COMPARISON_FILE = OUTPUT_DIR / "gan_feature_comparison.csv"


print("=" * 70)
print("GAN SAMPLE VALIDATION")
print("=" * 70)


# ============================================================
# 1. LOAD FINAL DATASET
# ============================================================

print("\nLoading final selected-feature dataset...")

final_df = pd.read_csv(FINAL_DATASET)

print("Final dataset shape:", final_df.shape)


# ============================================================
# 2. LOAD GAN MASTER FEATURES
# ============================================================

print("\nLoading GAN feature dataset...")

gan_df = pd.read_csv(GAN_FEATURES)

print("GAN feature dataset shape:", gan_df.shape)


# ============================================================
# 3. DISPLAY INFORMATION
# ============================================================

print("\nFinal dataset columns:")
for c in final_df.columns:
    print(" ", c)

print("\nGAN dataset columns:")
for c in gan_df.columns:
    print(" ", c)


# ============================================================
# 4. FIND POSSIBLE SOURCE COLUMN
# ============================================================

source_candidates = [
    "Source",
    "source",
    "Dataset",
    "dataset",
    "Data_Source",
    "data_source",
    "Sample_Type",
    "sample_type",
    "Type",
    "type",
    "Origin",
    "origin"
]

source_col = None

for c in source_candidates:
    if c in final_df.columns:
        source_col = c
        break

if source_col:
    print("\nDetected source column:", source_col)
    print(final_df[source_col].value_counts(dropna=False))
else:
    print("\nNo explicit source column detected.")


# ============================================================
# 5. FIND POSSIBLE GAN IDENTIFIER
# ============================================================

gan_mask = pd.Series(False, index=final_df.index)

for c in final_df.columns:

    if final_df[c].dtype == object:

        values = (
            final_df[c]
            .astype(str)
            .str.lower()
        )

        mask = (
            values.str.contains("gan", na=False)
            | values.str.contains("synthetic", na=False)
            | values.str.contains("synthetic_mri", na=False)
        )

        if mask.any():
            print(
                f"\nPossible GAN identifier found in column: {c}"
            )

            print(
                final_df.loc[mask, c].head(20).to_string(index=False)
            )

            gan_mask = gan_mask | mask


# ============================================================
# 6. IF NO GAN IDENTIFIER, CHECK GAN FEATURE DATASET
# ============================================================

if gan_mask.sum() == 0:

    print(
        "\nGAN rows could not be identified automatically "
        "from the final dataset."
    )

    print(
        "The GAN master dataset contains:",
        len(gan_df),
        "rows."
    )

    print(
        "\nThis is expected if the final selected dataset "
        "does not preserve the source identifier."
    )


# ============================================================
# 7. SUMMARY
# ============================================================

summary = {
    "Final_Dataset_Rows": len(final_df),
    "Final_Dataset_Columns": len(final_df.columns),
    "GAN_Master_Rows": len(gan_df),
    "GAN_Rows_Detected_In_Final": int(gan_mask.sum())
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 8. NUMERIC FEATURE SUMMARY
# ============================================================

final_numeric = final_df.select_dtypes(
    include=np.number
)

gan_numeric = gan_df.select_dtypes(
    include=np.number
)

print("\nFinal numeric columns:", len(final_numeric.columns))
print("GAN numeric columns:", len(gan_numeric.columns))


# ============================================================
# 9. FEATURE-LEVEL COMPARISON
# ============================================================

common_features = sorted(
    set(final_numeric.columns)
    & set(gan_numeric.columns)
)

print(
    "\nCommon numeric features:",
    len(common_features)
)


comparison_rows = []


for feature in common_features:

    final_values = pd.to_numeric(
        final_numeric[feature],
        errors="coerce"
    ).dropna()

    gan_values = pd.to_numeric(
        gan_numeric[feature],
        errors="coerce"
    ).dropna()

    if len(final_values) == 0 or len(gan_values) == 0:
        continue

    final_mean = final_values.mean()
    gan_mean = gan_values.mean()

    final_std = final_values.std()
    gan_std = gan_values.std()

    mean_difference = (
        gan_mean - final_mean
    )

    relative_difference = (
        abs(mean_difference)
        / (abs(final_mean) + 1e-12)
    )

    comparison_rows.append({
        "Feature": feature,
        "Final_Mean": final_mean,
        "Final_Std": final_std,
        "GAN_Mean": gan_mean,
        "GAN_Std": gan_std,
        "Mean_Difference": mean_difference,
        "Relative_Mean_Difference": relative_difference
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df = comparison_df.sort_values(
    "Relative_Mean_Difference"
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GAN VALIDATION SUMMARY")
print("=" * 70)

print(
    "\nFinal dataset rows:",
    len(final_df)
)

print(
    "GAN master rows:",
    len(gan_df)
)

print(
    "Common numeric features:",
    len(common_features)
)

print(
    "\nValidation files:"
)

print(
    SUMMARY_FILE
)

print(
    COMPARISON_FILE
)

print("\nDONE.")