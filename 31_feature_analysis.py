# ============================================================
# 31_feature_analysis.py
# Corpus Callosum Feature Analysis
# Normal vs Delayed Myelinisation
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = r"E:\Corpus_Callosum"

MASTER_PATH = os.path.join(
    ROOT,
    "22_statistical_analysis",
    "master_dataset.csv"
)

MORPH_PATH = os.path.join(
    ROOT,
    "16_measurements",
    "features.csv"
)

TEXTURE_PATH = os.path.join(
    ROOT,
    "20_texture_analysis",
    "texture_features.csv"
)

RADIOMIC_PATH = os.path.join(
    ROOT,
    "21_radiomic_features",
    "radiomic_features.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "31_feature_analysis_results"
)

PLOTS_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CORPUS CALLOSUM FEATURE ANALYSIS")
print("NORMAL vs DELAYED MYELINISATION")
print("=" * 70)

print("\nPurpose:")
print("Identify features that differ between Normal and Delayed")
print("myelinisation subjects.")

print("\nIMPORTANT:")
print("Normal subjects : 190")
print("Delayed subjects: 2")

print("\nThis is exploratory analysis.")
print("It is NOT a statistically powered clinical analysis.")

print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

master = pd.read_csv(
    MASTER_PATH
)

morph = pd.read_csv(
    MORPH_PATH
)

texture = pd.read_csv(
    TEXTURE_PATH
)

radiomic = pd.read_csv(
    RADIOMIC_PATH
)


print("\nDataset sizes")
print("-" * 70)

print("Master     :", master.shape)
print("Morphology :", morph.shape)
print("Texture    :", texture.shape)
print("Radiomics  :", radiomic.shape)


# ============================================================
# NORMALIZE SUBJECT IDs
# ============================================================

def normalize_subject(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            ".png",
            "",
            regex=False
        )
        .str.replace(
            ".nii.gz",
            "",
            regex=False
        )
    )


master["Subject"] = normalize_subject(
    master["Subject"]
)

morph["Subject"] = normalize_subject(
    morph["Subject"]
)

texture["Subject"] = normalize_subject(
    texture["Subject"]
)

radiomic["Subject"] = normalize_subject(
    radiomic["Subject"]
)


# ============================================================
# PREPARE LABELS
# ============================================================

labels = master[
    [
        "Subject",
        "myelinisation"
    ]
].copy()


labels["myelinisation"] = (
    labels["myelinisation"]
    .astype(str)
    .str.strip()
    .str.lower()
)


labels = labels[
    labels["myelinisation"].isin(
        [
            "normal",
            "delayed"
        ]
    )
].copy()


labels["Target"] = (
    labels["myelinisation"]
    .map(
        {
            "normal": 0,
            "delayed": 1
        }
    )
)


# ============================================================
# PREPARE MORPHOLOGY
# ============================================================

morph_features = morph.drop(
    columns=["Subject"],
    errors="ignore"
).copy()


morph_features = morph_features.add_prefix(
    "MORPH_"
)


morph_features.insert(
    0,
    "Subject",
    morph["Subject"]
)


# ============================================================
# PREPARE TEXTURE
# ============================================================

texture_features = texture.drop(
    columns=["Subject"],
    errors="ignore"
).copy()


texture_features = texture_features.add_prefix(
    "TEXTURE_"
)


texture_features.insert(
    0,
    "Subject",
    texture["Subject"]
)


# ============================================================
# PREPARE RADIOMICS
# ============================================================

radiomic_features = radiomic.drop(
    columns=["Subject"],
    errors="ignore"
).copy()


radiomic_features = radiomic_features.add_prefix(
    "RAD_"
)


radiomic_features.insert(
    0,
    "Subject",
    radiomic["Subject"]
)


# ============================================================
# MERGE
# ============================================================

print("\nMerging feature datasets...")

dataset = labels.merge(
    morph_features,
    on="Subject",
    how="inner"
)


dataset = dataset.merge(
    texture_features,
    on="Subject",
    how="inner"
)


dataset = dataset.merge(
    radiomic_features,
    on="Subject",
    how="inner"
)


print(
    "\nMerged dataset:",
    dataset.shape
)


# ============================================================
# VERIFY LABELS
# ============================================================

print("\nClass distribution")
print("-" * 70)

print(
    dataset["myelinisation"]
    .value_counts()
)


print("\nDelayed subjects")
print("-" * 70)

print(
    dataset[
        dataset["Target"] == 1
    ][
        [
            "Subject",
            "myelinisation"
        ]
    ].to_string(index=False)
)


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

DROP_COLUMNS = [
    "Subject",
    "myelinisation",
    "Target"
]


X = dataset.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)


X = X.select_dtypes(
    include=[np.number]
)


# Replace infinite values

X = X.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# ============================================================
# REMOVE EMPTY FEATURES
# ============================================================

empty_columns = X.columns[
    X.isna().all()
]


if len(empty_columns) > 0:

    print(
        "\nRemoving all-NaN features:",
        len(empty_columns)
    )

    X = X.drop(
        columns=empty_columns
    )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = X.columns[
    X.nunique(
        dropna=True
    ) <= 1
]


print(
    "\nConstant features:",
    len(constant_columns)
)


if len(constant_columns) > 0:

    X = X.drop(
        columns=constant_columns
    )


print(
    "Usable features:",
    X.shape[1]
)


# ============================================================
# SAVE MERGED DATASET
# ============================================================

merged_path = os.path.join(
    OUTPUT_DIR,
    "merged_feature_dataset.csv"
)


dataset.to_csv(
    merged_path,
    index=False
)


# ============================================================
# BASIC FEATURE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)


normal_mask = (
    dataset["Target"] == 0
)

delayed_mask = (
    dataset["Target"] == 1
)


normal_data = X.loc[
    normal_mask
]

delayed_data = X.loc[
    delayed_mask
]


statistics = []


for feature in X.columns:

    normal_values = (
        pd.to_numeric(
            normal_data[feature],
            errors="coerce"
        )
        .dropna()
    )


    delayed_values = (
        pd.to_numeric(
            delayed_data[feature],
            errors="coerce"
        )
        .dropna()
    )


    if len(normal_values) == 0:
        continue

    if len(delayed_values) == 0:
        continue


    normal_mean = (
        normal_values.mean()
    )

    delayed_mean = (
        delayed_values.mean()
    )


    normal_median = (
        normal_values.median()
    )

    delayed_median = (
        delayed_values.median()
    )


    normal_std = (
        normal_values.std()
    )

    delayed_std = (
        delayed_values.std()
    )


    # Difference in means
    mean_difference = (
        delayed_mean -
        normal_mean
    )


    # Relative difference
    denominator = (
        abs(normal_mean)
        + 1e-10
    )


    relative_difference = (
        mean_difference /
        denominator
    )


    # Standardized effect-like score
    pooled_std = np.sqrt(
        (
            normal_values.var()
            +
            delayed_values.var()
        ) / 2
    )


    if pooled_std > 0:

        standardized_difference = (
            abs(mean_difference)
            / pooled_std
        )

    else:

        standardized_difference = np.inf


    statistics.append({

        "Feature":
            feature,

        "Normal_Mean":
            normal_mean,

        "Delayed_Mean":
            delayed_mean,

        "Normal_Median":
            normal_median,

        "Delayed_Median":
            delayed_median,

        "Normal_STD":
            normal_std,

        "Delayed_STD":
            delayed_std,

        "Mean_Difference":
            mean_difference,

        "Relative_Difference":
            relative_difference,

        "Standardized_Difference":
            standardized_difference
    })


statistics_df = pd.DataFrame(
    statistics
)


# ============================================================
# SORT FEATURES
# ============================================================

statistics_df = (
    statistics_df
    .sort_values(
        "Standardized_Difference",
        ascending=False
    )
)


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics_path = os.path.join(
    OUTPUT_DIR,
    "feature_statistics.csv"
)


statistics_df.to_csv(
    statistics_path,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\nTop 30 features by separation")
print("-" * 70)


print(
    statistics_df[
        [
            "Feature",
            "Normal_Mean",
            "Delayed_Mean",
            "Standardized_Difference"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# TOP FEATURE LIST
# ============================================================

top_features = (
    statistics_df[
        "Feature"
    ]
    .head(20)
    .tolist()
)


top_features_path = os.path.join(
    OUTPUT_DIR,
    "top_20_features.txt"
)


with open(
    top_features_path,
    "w",
    encoding="utf-8"
) as f:

    for feature in top_features:

        f.write(
            feature + "\n"
        )


# ============================================================
# BOX PLOTS
# ============================================================

print("\nGenerating feature plots...")


for feature in top_features:

    normal_values = (
        pd.to_numeric(
            normal_data[feature],
            errors="coerce"
        )
        .dropna()
    )


    delayed_values = (
        pd.to_numeric(
            delayed_data[feature],
            errors="coerce"
        )
        .dropna()
    )


    if (
        len(normal_values) == 0
        or len(delayed_values) == 0
    ):
        continue


    plt.figure(
        figsize=(7, 5)
    )


    plt.boxplot(
        [
            normal_values,
            delayed_values
        ],
        labels=[
            "Normal",
            "Delayed"
        ]
    )


    # Show individual delayed points
    delayed_x = np.ones(
        len(delayed_values)
    ) * 2


    plt.scatter(
        delayed_x,
        delayed_values,
        alpha=0.8
    )


    plt.title(
        f"Normal vs Delayed: {feature}"
    )

    plt.ylabel(
        feature
    )

    plt.tight_layout()


    safe_name = (
        feature
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
        .replace(
            ":",
            "_"
        )
    )


    plot_path = os.path.join(
        PLOTS_DIR,
        f"{safe_name}.png"
    )


    plt.savefig(
        plot_path,
        dpi=200
    )


    plt.close()


# ============================================================
# PCA
# ============================================================

print("\nRunning PCA...")


# Median imputation

X_pca = X.copy()


X_pca = X_pca.fillna(
    X_pca.median()
)


# Remove columns that still contain NaN

X_pca = X_pca.dropna(
    axis=1
)


# Standardize

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X_pca
)


# PCA

pca = PCA(
    n_components=2,
    random_state=42
)


X_pca_2d = pca.fit_transform(
    X_scaled
)


pca_df = pd.DataFrame(
    {
        "PC1": X_pca_2d[:, 0],
        "PC2": X_pca_2d[:, 1],
        "Subject": dataset["Subject"].values,
        "Class": dataset["myelinisation"].values
    }
)


# ============================================================
# PCA PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


normal_points = (
    pca_df["Class"] == "normal"
)


delayed_points = (
    pca_df["Class"] == "delayed"
)


plt.scatter(
    pca_df.loc[
        normal_points,
        "PC1"
    ],
    pca_df.loc[
        normal_points,
        "PC2"
    ],
    label="Normal",
    alpha=0.6
)


plt.scatter(
    pca_df.loc[
        delayed_points,
        "PC1"
    ],
    pca_df.loc[
        delayed_points,
        "PC2"
    ],
    label="Delayed",
    s=100
)


# Label delayed subjects

for _, row in pca_df[
    delayed_points
].iterrows():

    plt.annotate(
        row["Subject"],
        (
            row["PC1"],
            row["PC2"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )


plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)"
)


plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)"
)


plt.title(
    "PCA: Normal vs Delayed Myelinisation"
)


plt.legend()

plt.tight_layout()


pca_path = os.path.join(
    PLOTS_DIR,
    "PCA_normal_vs_delayed.png"
)


plt.savefig(
    pca_path,
    dpi=250
)


plt.close()


# ============================================================
# SAVE PCA DATA
# ============================================================

pca_csv_path = os.path.join(
    OUTPUT_DIR,
    "pca_coordinates.csv"
)


pca_df.to_csv(
    pca_csv_path,
    index=False
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\nGenerating correlation analysis...")


# Use top features only

correlation_features = top_features[
    :15
]


correlation_data = dataset[
    correlation_features
].copy()


correlation_matrix = (
    correlation_data
    .corr()
)


correlation_path = os.path.join(
    OUTPUT_DIR,
    "top_feature_correlations.csv"
)


correlation_matrix.to_csv(
    correlation_path
)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(12, 10)
)


plt.imshow(
    correlation_matrix,
    aspect="auto"
)


plt.colorbar(
    label="Correlation"
)


plt.xticks(
    range(
        len(correlation_features)
    ),
    correlation_features,
    rotation=90,
    fontsize=7
)


plt.yticks(
    range(
        len(correlation_features)
    ),
    correlation_features,
    fontsize=7
)


plt.title(
    "Correlation of Top Features"
)


plt.tight_layout()


heatmap_path = os.path.join(
    PLOTS_DIR,
    "top_feature_correlation_heatmap.png"
)


plt.savefig(
    heatmap_path,
    dpi=250
)


plt.close()


# ============================================================
# DELAYED SUBJECT FEATURE PROFILE
# ============================================================

print("\nCreating delayed subject profiles...")


delayed_profiles = dataset[
    dataset["Target"] == 1
].copy()


profile_columns = [
    "Subject",
    "myelinisation"
] + top_features


profile_columns = [
    col
    for col in profile_columns
    if col in delayed_profiles.columns
]


delayed_profile_path = os.path.join(
    OUTPUT_DIR,
    "delayed_subject_feature_profiles.csv"
)


delayed_profiles[
    profile_columns
].to_csv(
    delayed_profile_path,
    index=False
)


# ============================================================
# FEATURE GROUP SUMMARY
# ============================================================

print("\nFeature group summary")
print("-" * 70)


morph_count = sum(
    feature.startswith("MORPH_")
    for feature in X.columns
)


texture_count = sum(
    feature.startswith("TEXTURE_")
    for feature in X.columns
)


radiomic_count = sum(
    feature.startswith("RAD_")
    for feature in X.columns
)


print(
    "Morphology features:",
    morph_count
)

print(
    "Texture features   :",
    texture_count
)

print(
    "Radiomic features  :",
    radiomic_count
)


# ============================================================
# TOP FEATURES BY GROUP
# ============================================================

print("\nTop features by group")
print("-" * 70)


for group_name, prefix in [
    ("MORPHOLOGY", "MORPH_"),
    ("TEXTURE", "TEXTURE_"),
    ("RADIOMICS", "RAD_")
]:

    group_df = statistics_df[
        statistics_df["Feature"]
        .str.startswith(prefix)
    ]


    print(
        f"\n{group_name}"
    )


    print(
        group_df[
            [
                "Feature",
                "Normal_Mean",
                "Delayed_Mean",
                "Standardized_Difference"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "Total_Subjects":
        len(dataset),

    "Normal_Subjects":
        int(normal_mask.sum()),

    "Delayed_Subjects":
        int(delayed_mask.sum()),

    "Total_Usable_Features":
        X.shape[1],

    "Morphology_Features":
        morph_count,

    "Texture_Features":
        texture_count,

    "Radiomic_Features":
        radiomic_count
}


summary_df = pd.DataFrame(
    [summary]
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "feature_analysis_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(
    OUTPUT_DIR
)

print("\nImportant files:")

print(
    "Feature statistics:"
)

print(
    statistics_path
)

print(
    "\nTop features:"
)

print(
    top_features_path
)

print(
    "\nPCA plot:"
)

print(
    pca_path
)

print(
    "\nCorrelation heatmap:"
)

print(
    heatmap_path
)

print(
    "\nDelayed subject profiles:"
)

print(
    delayed_profile_path
)

print("\nPlots folder:")
print(
    PLOTS_DIR
)

print("=" * 70)