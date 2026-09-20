from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL RESULTS + PLOTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = (
    BASE_DIR
    / "feature_selection"
    / "selected_feature_dataset.csv"
)

MODEL_RESULTS_FILE = (
    BASE_DIR
    / "final_model_validation"
    / "final_model_validation_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "final_results"
)

PLOTS_DIR = (
    OUTPUT_DIR
    / "plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("FINAL RESULTS AND PLOTS")
print("=" * 80)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nFinal dataset not found:\n{DATASET_FILE}"
    )


if not MODEL_RESULTS_FILE.exists():

    raise FileNotFoundError(
        f"\nModel validation results not found:\n"
        f"{MODEL_RESULTS_FILE}"
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATASET_FILE
)

results = pd.read_csv(
    MODEL_RESULTS_FILE
)


print("\nDataset shape:")
print(df.shape)

print("\nModel results shape:")
print(results.shape)


# ============================================================
# 3. FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "myelinisation",
    "myelination",
    "myelination_status",
    "Myelination_Status",
    "Myelination",
    "status",
    "Status",
    "class",
    "Class",
    "label",
    "Label",
    "target",
    "Target"
]


target_col = None


for col in target_candidates:

    if col in df.columns:

        target_col = col

        break


if target_col is None:

    print("\nAvailable columns:")

    for col in df.columns:
        print(" -", col)

    raise ValueError(
        "\nTarget column could not be detected."
    )


print(
    "\nTarget column:",
    target_col
)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

class_counts = (
    df[target_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .value_counts()
)


normal_count = int(
    class_counts.get(
        "normal",
        0
    )
)


delayed_count = int(
    class_counts.get(
        "delayed",
        0
    )
)


total_samples = len(df)


# ============================================================
# 5. FEATURE COUNT
# ============================================================

feature_df = df.drop(
    columns=[target_col]
)


if "Subject" in feature_df.columns:

    feature_df = feature_df.drop(
        columns=["Subject"]
    )


if "subject" in feature_df.columns:

    feature_df = feature_df.drop(
        columns=["subject"]
    )


numeric_features = feature_df.select_dtypes(
    include=np.number
)


feature_count = len(
    numeric_features.columns
)


# ============================================================
# 6. MISSING VALUES
# ============================================================

missing_values = int(
    numeric_features
    .isna()
    .sum()
    .sum()
)


# ============================================================
# 7. DATASET SUMMARY
# ============================================================

dataset_summary = pd.DataFrame({

    "Metric": [

        "Total Samples",

        "Normal Samples",

        "Delayed Samples",

        "Normal Percentage",

        "Delayed Percentage",

        "Selected Features",

        "Missing Feature Values"

    ],

    "Value": [

        total_samples,

        normal_count,

        delayed_count,

        round(
            normal_count / total_samples * 100,
            2
        ),

        round(
            delayed_count / total_samples * 100,
            2
        ),

        feature_count,

        missing_values
    ]
})


dataset_summary_file = (
    OUTPUT_DIR
    / "dataset_summary.csv"
)


dataset_summary.to_csv(
    dataset_summary_file,
    index=False
)


# ============================================================
# 8. CLASS DISTRIBUTION PLOT
# ============================================================

labels = [
    "Normal",
    "Delayed"
]

values = [
    normal_count,
    delayed_count
]


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    labels,
    values
)


plt.xlabel(
    "Myelination Status"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Distribution of Myelination Status"
)


for i, value in enumerate(values):

    plt.text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )


plt.tight_layout()


class_plot_file = (
    PLOTS_DIR
    / "class_distribution.png"
)


plt.savefig(
    class_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 9. PREPARE MODEL RESULTS
# ============================================================

required_columns = [

    "Model",

    "Accuracy_Mean",
    "Accuracy_STD",

    "Precision_Mean",
    "Precision_STD",

    "Recall_Mean",
    "Recall_STD",

    "F1_Mean",
    "F1_STD",

    "ROC_AUC_Mean",
    "ROC_AUC_STD"
]


missing_result_columns = [

    col

    for col in required_columns

    if col not in results.columns
]


if missing_result_columns:

    raise ValueError(
        "\nMissing model-result columns:\n"
        + str(missing_result_columns)
    )


# ============================================================
# 10. SAVE FINAL MODEL RESULTS
# ============================================================

final_model_results = results[
    required_columns
].copy()


final_model_results = (
    final_model_results
    .sort_values(
        "F1_Mean",
        ascending=False
    )
    .reset_index(drop=True)
)


final_model_results_file = (
    OUTPUT_DIR
    / "final_model_results.csv"
)


final_model_results.to_csv(
    final_model_results_file,
    index=False
)


# ============================================================
# 11. MODEL PERFORMANCE PLOT
# ============================================================

model_names = (
    final_model_results["Model"]
    .tolist()
)


x = np.arange(
    len(model_names)
)


width = 0.18


plt.figure(
    figsize=(12, 6)
)


plt.bar(
    x - 2 * width,
    final_model_results["Accuracy_Mean"],
    width,
    label="Accuracy"
)


plt.bar(
    x - width,
    final_model_results["Precision_Mean"],
    width,
    label="Precision"
)


plt.bar(
    x,
    final_model_results["Recall_Mean"],
    width,
    label="Recall"
)


plt.bar(
    x + width,
    final_model_results["F1_Mean"],
    width,
    label="F1"
)


plt.bar(
    x + 2 * width,
    final_model_results["ROC_AUC_Mean"],
    width,
    label="ROC-AUC"
)


plt.xticks(
    x,
    model_names,
    rotation=20,
    ha="right"
)


plt.ylabel(
    "Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Final Model Performance Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.legend()


plt.tight_layout()


model_comparison_file = (
    PLOTS_DIR
    / "model_performance_comparison.png"
)


plt.savefig(
    model_comparison_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 12. F1 SCORE PLOT
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.bar(
    model_names,
    final_model_results["F1_Mean"]
)


plt.ylabel(
    "F1 Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "F1 Score Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.xticks(
    rotation=20,
    ha="right"
)


for i, value in enumerate(
    final_model_results["F1_Mean"]
):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


f1_plot_file = (
    PLOTS_DIR
    / "f1_score_comparison.png"
)


plt.savefig(
    f1_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 13. ROC-AUC PLOT
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.bar(
    model_names,
    final_model_results["ROC_AUC_Mean"]
)


plt.ylabel(
    "ROC-AUC"
)

plt.xlabel(
    "Model"
)

plt.title(
    "ROC-AUC Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.xticks(
    rotation=20,
    ha="right"
)


for i, value in enumerate(
    final_model_results["ROC_AUC_Mean"]
):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


roc_plot_file = (
    PLOTS_DIR
    / "roc_auc_comparison.png"
)


plt.savefig(
    roc_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 14. ACCURACY PLOT
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.bar(
    model_names,
    final_model_results["Accuracy_Mean"]
)


plt.ylabel(
    "Accuracy"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Accuracy Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.xticks(
    rotation=20,
    ha="right"
)


for i, value in enumerate(
    final_model_results["Accuracy_Mean"]
):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )


plt.tight_layout()


accuracy_plot_file = (
    PLOTS_DIR
    / "accuracy_comparison.png"
)


plt.savefig(
    accuracy_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 15. ERROR / STABILITY PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.errorbar(
    model_names,
    final_model_results["F1_Mean"],
    yerr=final_model_results["F1_STD"],
    fmt="o",
    capsize=5
)


plt.ylabel(
    "F1 Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "F1 Score Stability Across 5-Fold Validation"
)


plt.ylim(
    0,
    1.05
)


plt.xticks(
    rotation=20,
    ha="right"
)


plt.tight_layout()


stability_plot_file = (
    PLOTS_DIR
    / "f1_stability.png"
)


plt.savefig(
    stability_plot_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 16. BEST MODEL
# ============================================================

best_model = (
    final_model_results
    .sort_values(
        "F1_Mean",
        ascending=False
    )
    .iloc[0]
)


best_model_summary = pd.DataFrame({

    "Metric": [

        "Best Model",

        "Accuracy Mean",

        "Accuracy STD",

        "Precision Mean",

        "Precision STD",

        "Recall Mean",

        "Recall STD",

        "F1 Mean",

        "F1 STD",

        "ROC-AUC Mean",

        "ROC-AUC STD"

    ],

    "Value": [

        best_model["Model"],

        best_model["Accuracy_Mean"],

        best_model["Accuracy_STD"],

        best_model["Precision_Mean"],

        best_model["Precision_STD"],

        best_model["Recall_Mean"],

        best_model["Recall_STD"],

        best_model["F1_Mean"],

        best_model["F1_STD"],

        best_model["ROC_AUC_Mean"],

        best_model["ROC_AUC_STD"]
    ]
})


best_model_file = (
    OUTPUT_DIR
    / "best_model_summary.csv"
)


best_model_summary.to_csv(
    best_model_file,
    index=False
)


# ============================================================
# 17. PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(
    "\nTotal samples:",
    total_samples
)

print(
    "Normal samples:",
    normal_count
)

print(
    "Delayed samples:",
    delayed_count
)

print(
    "Selected features:",
    feature_count
)

print(
    "Missing values:",
    missing_values
)


print("\n")
print("=" * 80)
print("FINAL MODEL RESULTS")
print("=" * 80)

print(
    final_model_results.to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    "\nModel:",
    best_model["Model"]
)

print(
    "Accuracy:",
    f"{best_model['Accuracy_Mean']:.4f}"
)

print(
    "Precision:",
    f"{best_model['Precision_Mean']:.4f}"
)

print(
    "Recall:",
    f"{best_model['Recall_Mean']:.4f}"
)

print(
    "F1:",
    f"{best_model['F1_Mean']:.4f}"
)

print(
    "ROC-AUC:",
    f"{best_model['ROC_AUC_Mean']:.4f}"
)


# ============================================================
# 18. PRINT CREATED FILES
# ============================================================

print("\n")
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(
    "\nDataset summary:"
)

print(
    dataset_summary_file
)


print(
    "\nFinal model results:"
)

print(
    final_model_results_file
)


print(
    "\nBest model summary:"
)

print(
    best_model_file
)


print(
    "\nPlots:"
)

print(
    class_plot_file
)

print(
    model_comparison_file
)

print(
    f1_plot_file
)

print(
    roc_plot_file
)

print(
    accuracy_plot_file
)

print(
    stability_plot_file
)


print("\n")
print("=" * 80)
print("FINAL RESULTS AND PLOTS COMPLETE")
print("=" * 80)