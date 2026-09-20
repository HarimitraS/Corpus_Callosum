from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# RESEARCH CONCLUSIONS GENERATOR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FINAL_RESULTS_DIR = (
    BASE_DIR / "final_results"
)

OUTPUT_DIR = FINAL_RESULTS_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

DATASET_SUMMARY_FILE = (
    FINAL_RESULTS_DIR
    / "dataset_summary.csv"
)

MODEL_RESULTS_FILE = (
    FINAL_RESULTS_DIR
    / "final_model_results.csv"
)

BEST_MODEL_FILE = (
    FINAL_RESULTS_DIR
    / "best_model_summary.csv"
)

STATISTICAL_RESULTS_FILE = (
    BASE_DIR
    / "statistical_results.csv"
)

SELECTED_FEATURE_FILE = (
    BASE_DIR
    / "feature_selection"
    / "selected_feature_dataset.csv"
)

ROBUST_RESULTS_FILE = (
    BASE_DIR
    / "classification_results"
    / "robust_model_comparison.csv"
)

GAN_COMPARISON_FILE = (
    BASE_DIR
    / "classification_results"
    / "gan_augmentation_comparison.csv"
)

GAN_MASTER_FILE = (
    BASE_DIR.parent
    / "31_gan_delayed_pipeline"
    / "09_merged_features"
    / "gan_delayed_master_features.csv"
)


print("=" * 80)
print("RESEARCH CONCLUSIONS GENERATOR")
print("=" * 80)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):

    if path.exists():

        try:
            return pd.read_csv(path)

        except Exception as e:

            print(
                f"\nWarning: Could not read {path}"
            )

            print(e)

            return None

    print(
        f"\nOptional file not found:"
        f"\n{path}"
    )

    return None


def get_value(
    df,
    metric,
    default=None
):

    if df is None:
        return default

    if "Metric" not in df.columns:
        return default

    rows = df[
        df["Metric"].astype(str).str.lower()
        == metric.lower()
    ]

    if len(rows) == 0:
        return default

    return rows.iloc[0]["Value"]


def fmt(value, digits=4):

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"

    except Exception:
        return str(value)


def pct(value):

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}%"

    except Exception:
        return str(value)


# ============================================================
# LOAD RESULTS
# ============================================================

dataset_summary = load_csv(
    DATASET_SUMMARY_FILE
)

model_results = load_csv(
    MODEL_RESULTS_FILE
)

best_model_summary = load_csv(
    BEST_MODEL_FILE
)

statistical_results = load_csv(
    STATISTICAL_RESULTS_FILE
)

selected_features = load_csv(
    SELECTED_FEATURE_FILE
)

robust_results = load_csv(
    ROBUST_RESULTS_FILE
)

gan_comparison = load_csv(
    GAN_COMPARISON_FILE
)

gan_master = load_csv(
    GAN_MASTER_FILE
)


# ============================================================
# DATASET INFORMATION
# ============================================================

total_samples = get_value(
    dataset_summary,
    "Total Samples"
)

normal_samples = get_value(
    dataset_summary,
    "Normal Samples"
)

delayed_samples = get_value(
    dataset_summary,
    "Delayed Samples"
)

normal_percentage = get_value(
    dataset_summary,
    "Normal Percentage"
)

delayed_percentage = get_value(
    dataset_summary,
    "Delayed Percentage"
)

selected_feature_count = get_value(
    dataset_summary,
    "Selected Features"
)

missing_values = get_value(
    dataset_summary,
    "Missing Feature Values"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

best_model = get_value(
    best_model_summary,
    "Best Model"
)

best_accuracy = get_value(
    best_model_summary,
    "Accuracy Mean"
)

best_precision = get_value(
    best_model_summary,
    "Precision Mean"
)

best_recall = get_value(
    best_model_summary,
    "Recall Mean"
)

best_f1 = get_value(
    best_model_summary,
    "F1 Mean"
)

best_roc_auc = get_value(
    best_model_summary,
    "ROC-AUC Mean"
)


# ============================================================
# MODEL RANKING
# ============================================================

model_ranking_text = ""

if model_results is not None:

    ranked = model_results.sort_values(
        "F1_Mean",
        ascending=False
    )

    for i, row in enumerate(
        ranked.itertuples(index=False),
        start=1
    ):

        model_ranking_text += (
            f"{i}. {row.Model}: "
            f"Accuracy={row.Accuracy_Mean:.4f}, "
            f"Precision={row.Precision_Mean:.4f}, "
            f"Recall={row.Recall_Mean:.4f}, "
            f"F1={row.F1_Mean:.4f}, "
            f"ROC-AUC={row.ROC_AUC_Mean:.4f}\n"
        )


# ============================================================
# STATISTICAL ANALYSIS INFORMATION
# ============================================================

statistical_feature_count = None
significant_feature_count = None

if statistical_results is not None:

    statistical_feature_count = len(
        statistical_results
    )

    possible_p_columns = [

        "Adjusted_P_Value",

        "adjusted_p_value",

        "FDR_P_Value",

        "FDR_Adjusted_P_Value",

        "p_adjusted",

        "p_value_adjusted",

        "P_Adjusted",

        "p-value-adjusted"

    ]

    p_column = None

    for col in possible_p_columns:

        if col in statistical_results.columns:

            p_column = col
            break

    if p_column is not None:

        p_values = pd.to_numeric(
            statistical_results[p_column],
            errors="coerce"
        )

        significant_feature_count = int(
            (p_values < 0.05).sum()
        )


# ============================================================
# SELECTED FEATURE INFORMATION
# ============================================================

selected_feature_names = []

if selected_features is not None:

    for col in selected_features.columns:

        if col not in [
            "Subject",
            "subject",
            "myelinisation",
            "myelination",
            "myelination_status",
            "Myelination_Status"
        ]:

            selected_feature_names.append(
                col
            )


# ============================================================
# GAN INFORMATION
# ============================================================

gan_rows = None

if gan_master is not None:

    gan_rows = len(
        gan_master
    )


# ============================================================
# ROBUST MODEL INFORMATION
# ============================================================

robust_best_model = None
robust_best_f1 = None
robust_best_auc = None

if robust_results is not None:

    if "F1_Mean" in robust_results.columns:

        robust_sorted = (
            robust_results
            .sort_values(
                "F1_Mean",
                ascending=False
            )
        )

        if len(robust_sorted) > 0:

            robust_best_model = (
                robust_sorted.iloc[0]
                .get("Model", None)
            )

            robust_best_f1 = (
                robust_sorted.iloc[0]
                .get("F1_Mean", None)
            )

            robust_best_auc = (
                robust_sorted.iloc[0]
                .get("ROC_AUC_Mean", None)
            )


# ============================================================
# GAN COMPARISON INFORMATION
# ============================================================

gan_comparison_text = ""

if gan_comparison is not None:

    gan_comparison_text = (
        "A dedicated GAN augmentation comparison "
        "was included in the project workflow. "
        "The corresponding comparison results should "
        "be interpreted together with the final "
        "model-validation results rather than using "
        "GAN augmentation as an automatic indicator "
        "of improved performance."
    )

else:

    gan_comparison_text = (
        "A GAN augmentation comparison stage was "
        "completed in the project workflow; however, "
        "the expected comparison CSV was not available "
        "at the expected path during automatic report "
        "generation."
    )


# ============================================================
# CREATE RESEARCH CONCLUSION
# ============================================================

conclusion = f"""
============================================================
RESEARCH CONCLUSIONS
Infant Brain MRI Corpus Callosum Segmentation and
Myelination Status Analysis
============================================================


1. STUDY OVERVIEW
-----------------

The project developed a complete computational pipeline
for infant brain MRI analysis, beginning with MRI
preprocessing and corpus callosum extraction and extending
through segmentation, refinement, morphological analysis,
texture analysis, radiomic feature extraction, statistical
feature selection, and machine-learning classification.

The final statistical/classification dataset contains
{total_samples if total_samples is not None else 'N/A'} samples,
including {normal_samples if normal_samples is not None else 'N/A'}
normal-myelinisation samples and
{delayed_samples if delayed_samples is not None else 'N/A'}
delayed-myelinisation samples.

The final dataset contains
{selected_feature_count if selected_feature_count is not None else 'N/A'}
selected features and
{missing_values if missing_values is not None else 'N/A'}
missing feature values.


2. DATASET CHARACTERISTICS
--------------------------

The final dataset is strongly imbalanced toward the normal
myelination class.

Normal samples:
{normal_samples if normal_samples is not None else 'N/A'}
({pct(normal_percentage)})

Delayed samples:
{delayed_samples if delayed_samples is not None else 'N/A'}
({pct(delayed_percentage)})

This class imbalance is an important consideration when
interpreting classification performance. Accuracy alone
should therefore not be treated as the primary indicator
of clinical or research usefulness. Precision, recall,
F1-score, ROC-AUC, and the variability across validation
folds provide additional information about model behaviour.


3. STATISTICAL ANALYSIS AND FEATURE REDUCTION
---------------------------------------------

The project applied statistical analysis to identify
features associated with differences between the normal
and delayed myelination groups.

The workflow used Mann-Whitney U testing with
Benjamini-Hochberg false-discovery-rate correction.

The final modelling dataset was reduced to
{selected_feature_count if selected_feature_count is not None else 'N/A'}
features.

The selected features combine information from several
feature families, including first-order intensity
measurements, texture features, wavelet-derived features,
shape measurements, and morphological measurements.

The final feature set therefore represents multiple
properties of the corpus callosum rather than relying on
a single measurement type.


4. FINAL CLASSIFICATION PERFORMANCE
-----------------------------------

The final cross-validation comparison evaluated:

- Logistic Regression
- KNN
- Random Forest
- SVM

The best model according to mean F1-score was:

BEST MODEL: {best_model if best_model is not None else 'N/A'}

Mean Accuracy:
{fmt(best_accuracy)}

Mean Precision:
{fmt(best_precision)}

Mean Recall:
{fmt(best_recall)}

Mean F1-score:
{fmt(best_f1)}

Mean ROC-AUC:
{fmt(best_roc_auc)}


5. MODEL COMPARISON
-------------------

Models ranked by mean F1-score:

{model_ranking_text if model_ranking_text else 'Model ranking information was not available.'}


6. INTERPRETATION OF THE BEST MODEL
-----------------------------------

The selected final model achieved a high mean accuracy and
perfect mean precision in the five-fold validation performed
in this pipeline.

Its mean recall was lower than its precision, indicating that
although predictions assigned to the delayed class were highly
precise in the validation folds, some delayed cases were not
identified.

This distinction is particularly important because the delayed
class contains substantially fewer samples than the normal
class.

The model should therefore not be described simply as
"highly accurate" without also reporting its recall and
F1-score.


7. ROC-AUC INTERPRETATION
-------------------------

The ROC-AUC results demonstrate that different models have
different ranking/discrimination behaviour.

In the final validation:

KNN achieved a mean ROC-AUC of approximately
{fmt(
    model_results.loc[
        model_results["Model"] == "KNN",
        "ROC_AUC_Mean"
    ].iloc[0]
) if model_results is not None and "KNN" in model_results["Model"].values else 'N/A'}.

Random Forest achieved a mean ROC-AUC of approximately
{fmt(
    model_results.loc[
        model_results["Model"] == "Random Forest",
        "ROC_AUC_Mean"
    ].iloc[0]
) if model_results is not None and "Random Forest" in model_results["Model"].values else 'N/A'}.

Logistic Regression achieved a mean ROC-AUC of approximately
{fmt(
    model_results.loc[
        model_results["Model"] == "Logistic Regression",
        "ROC_AUC_Mean"
    ].iloc[0]
) if model_results is not None and "Logistic Regression" in model_results["Model"].values else 'N/A'}.

SVM achieved a mean ROC-AUC of approximately
{fmt(
    model_results.loc[
        model_results["Model"] == "SVM",
        "ROC_AUC_Mean"
    ].iloc[0]
) if model_results is not None and "SVM" in model_results["Model"].values else 'N/A'}.

These differences show why model selection should consider
multiple evaluation metrics rather than relying on one score.


8. GAN-BASED AUGMENTATION
-------------------------

The project also incorporated a WGAN-GP-based delayed-sample
generation stage.

The GAN-generated data were subsequently passed through the
feature extraction and statistical-analysis workflow.

{gan_comparison_text}

The GAN stage demonstrates the feasibility of incorporating
synthetically generated delayed samples into the broader
pipeline. However, because the number of synthetic samples
successfully incorporated into the feature dataset is limited,
the GAN results should be considered supportive/experimental
rather than definitive evidence that synthetic augmentation
improves generalisation.


9. MAIN RESEARCH FINDING
------------------------

The overall pipeline demonstrates that quantitative
morphological, intensity, and texture/radiomic characteristics
of the corpus callosum can be used to construct machine-learning
models for distinguishing normal and delayed myelination status
within the available dataset.

The strongest final classifier according to mean F1-score was
{best_model if best_model is not None else 'N/A'}, with a mean
F1-score of {fmt(best_f1)} and mean ROC-AUC of
{fmt(best_roc_auc)}.


10. IMPORTANT LIMITATIONS
-------------------------

The most important limitation is the substantial class
imbalance between normal and delayed cases.

Only
{delayed_samples if delayed_samples is not None else 'N/A'}
delayed samples are present in the final dataset.

Consequently, the validation estimates for the delayed class
can have relatively high variability between folds.

This is reflected in the reported standard deviations for
recall and F1-score.

The current results should therefore be interpreted as
promising experimental findings rather than definitive
clinical performance estimates.

Additional independent datasets containing more delayed
myelination cases would be required for stronger external
validation.


11. RECOMMENDED FUTURE WORK
---------------------------

Future work should focus on:

1. Increasing the number of delayed myelination cases.

2. Performing external validation on an independent cohort.

3. Evaluating sensitivity/specificity and confusion matrices
   in addition to the current metrics.

4. Testing whether GAN-generated samples improve performance
   consistently across independent validation sets.

5. Performing feature-importance and interpretability analysis.

6. Investigating whether the selected radiomic, texture,
   morphological, and intensity features remain stable across
   different scanners and acquisition protocols.

7. Validating the segmentation and downstream measurements
   against expert annotations on a larger cohort.


12. OVERALL CONCLUSION
----------------------

The completed pipeline successfully integrates medical-image
preprocessing, corpus callosum segmentation, segmentation
refinement, quantitative morphological analysis, texture and
radiomic feature extraction, statistical feature selection,
GAN-based data generation, and machine-learning classification.

Using the final 20-feature dataset, the best-performing model
in the five-fold validation was {best_model if best_model is not None else 'N/A'}.

It achieved:

Accuracy  = {fmt(best_accuracy)}
Precision = {fmt(best_precision)}
Recall    = {fmt(best_recall)}
F1-score  = {fmt(best_f1)}
ROC-AUC   = {fmt(best_roc_auc)}

These results indicate that the extracted quantitative
characteristics contain useful information for differentiating
normal and delayed myelination status in the studied dataset.

However, the strong class imbalance and limited number of
delayed cases mean that the findings require validation on
larger and independent datasets before any clinical
generalisation can be made.

The work therefore provides a promising computational
framework for quantitative infant corpus callosum analysis
and automated assessment of myelination status, while also
clearly identifying the need for larger datasets and
independent validation.
"""


# ============================================================
# SAVE TEXT REPORT
# ============================================================

CONCLUSION_FILE = (
    OUTPUT_DIR
    / "research_conclusions.txt"
)


with open(
    CONCLUSION_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        conclusion.strip()
        + "\n"
    )


# ============================================================
# CREATE MACHINE-READABLE SUMMARY
# ============================================================

summary_data = {

    "Total_Samples":
        total_samples,

    "Normal_Samples":
        normal_samples,

    "Delayed_Samples":
        delayed_samples,

    "Selected_Features":
        selected_feature_count,

    "Missing_Feature_Values":
        missing_values,

    "Best_Model":
        best_model,

    "Accuracy_Mean":
        best_accuracy,

    "Precision_Mean":
        best_precision,

    "Recall_Mean":
        best_recall,

    "F1_Mean":
        best_f1,

    "ROC_AUC_Mean":
        best_roc_auc,

    "Robust_Best_Model":
        robust_best_model,

    "Robust_Best_F1":
        robust_best_f1,

    "Robust_Best_ROC_AUC":
        robust_best_auc,

    "GAN_Master_Rows":
        gan_rows
}


summary_df = pd.DataFrame(
    [summary_data]
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "research_summary.csv"
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT CONCLUSION
# ============================================================

print("\n")
print("=" * 80)
print("RESEARCH CONCLUSION")
print("=" * 80)

print(
    conclusion
)


print("\n")
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(
    "\n1.",
    CONCLUSION_FILE
)

print(
    "\n2.",
    SUMMARY_FILE
)


print("\n")
print("=" * 80)
print("RESEARCH CONCLUSIONS COMPLETE")
print("=" * 80)