from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ============================================================
# FINAL MODEL VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "feature_selection"
    / "selected_feature_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "final_model_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = (
    OUTPUT_DIR
    / "final_model_validation_results.csv"
)


print("=" * 80)
print("FINAL MODEL VALIDATION")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. FIND TARGET COLUMN
# ============================================================

# IMPORTANT:
# The actual target column in this project is:
# myelinisation

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


print("\nTarget column:", target_col)


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    df[target_col]
    .value_counts(dropna=False)
)


# ============================================================
# 4. ENCODE TARGET
# ============================================================

target = (
    df[target_col]
    .astype(str)
    .str.strip()
    .str.lower()
)


print("\nUnique target values:")

print(
    target.value_counts()
)


# The project uses:
# Normal
# Delayed

mapping = {
    "normal": 0,
    "delayed": 1
}


y = target.map(mapping)


if y.isna().any():

    unknown_values = (
        target[y.isna()]
        .unique()
        .tolist()
    )

    raise ValueError(
        "\nUnknown target values found:\n"
        + str(unknown_values)
        + "\n\nExpected values are: Normal and Delayed"
    )


y = y.astype(int)


# ============================================================
# 5. CREATE FEATURE MATRIX
# ============================================================

X = df.drop(
    columns=[target_col]
)


# Subject is an identifier and must NOT be used
# as a predictive feature.

if "Subject" in X.columns:

    X = X.drop(
        columns=["Subject"]
    )

elif "subject" in X.columns:

    X = X.drop(
        columns=["subject"]
    )


# Keep only numeric feature columns
X = X.select_dtypes(
    include=np.number
)


# ============================================================
# 6. REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    col
    for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]


if constant_columns:

    print("\nConstant features removed:")

    for col in constant_columns:
        print(" -", col)

    X = X.drop(
        columns=constant_columns
    )


print(
    "\nNumber of features used:",
    X.shape[1]
)


print("\nFeatures used:")

for col in X.columns:
    print(" -", col)


# ============================================================
# 7. CHECK DATA
# ============================================================

print(
    "\nTotal missing feature values:",
    int(X.isna().sum().sum())
)

print(
    "Total samples:",
    len(X)
)

print(
    "Normal samples:",
    int((y == 0).sum())
)

print(
    "Delayed samples:",
    int((y == 1).sum())
)


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            )
        )
    ]),


    "KNN": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    "Random Forest": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]),


    "SVM": Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        )
    ])
}


# ============================================================
# 10. SCORING
# ============================================================

scoring = {

    "accuracy": "accuracy",

    "precision": "precision",

    "recall": "recall",

    "f1": "f1",

    "roc_auc": "roc_auc"
}


# ============================================================
# 11. RUN VALIDATION
# ============================================================

results = []


for model_name, model in models.items():

    print("\n")
    print("=" * 80)
    print(model_name)
    print("=" * 80)


    scores = cross_validate(

        estimator=model,

        X=X,

        y=y,

        cv=cv,

        scoring=scoring,

        return_train_score=False,

        n_jobs=-1
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy_mean = (
        scores["test_accuracy"].mean()
    )

    accuracy_std = (
        scores["test_accuracy"].std()
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_mean = (
        scores["test_precision"].mean()
    )

    precision_std = (
        scores["test_precision"].std()
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_mean = (
        scores["test_recall"].mean()
    )

    recall_std = (
        scores["test_recall"].std()
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1_mean = (
        scores["test_f1"].mean()
    )

    f1_std = (
        scores["test_f1"].std()
    )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc_mean = (
        scores["test_roc_auc"].mean()
    )

    roc_auc_std = (
        scores["test_roc_auc"].std()
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Accuracy_Mean":
            accuracy_mean,

        "Accuracy_STD":
            accuracy_std,

        "Precision_Mean":
            precision_mean,

        "Precision_STD":
            precision_std,

        "Recall_Mean":
            recall_mean,

        "Recall_STD":
            recall_std,

        "F1_Mean":
            f1_mean,

        "F1_STD":
            f1_std,

        "ROC_AUC_Mean":
            roc_auc_mean,

        "ROC_AUC_STD":
            roc_auc_std
    })


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Accuracy : "
        f"{accuracy_mean:.4f} "
        f"+/- {accuracy_std:.4f}"
    )

    print(
        f"Precision: "
        f"{precision_mean:.4f} "
        f"+/- {precision_std:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_mean:.4f} "
        f"+/- {recall_std:.4f}"
    )

    print(
        f"F1       : "
        f"{f1_mean:.4f} "
        f"+/- {f1_std:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc_mean:.4f} "
        f"+/- {roc_auc_std:.4f}"
    )


# ============================================================
# 12. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# Sort by F1
results_df = (
    results_df
    .sort_values(
        by="F1_Mean",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 14. DISPLAY FINAL TABLE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL VALIDATION RESULTS")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 15. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]


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
# 16. SAVE BEST MODEL SUMMARY
# ============================================================

best_model_summary = pd.DataFrame({

    "Metric": [
        "Best Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ],

    "Value": [
        best_model["Model"],
        best_model["Accuracy_Mean"],
        best_model["Precision_Mean"],
        best_model["Recall_Mean"],
        best_model["F1_Mean"],
        best_model["ROC_AUC_Mean"]
    ]
})


best_model_summary.to_csv(
    OUTPUT_DIR
    / "best_model_summary.csv",
    index=False
)


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(
    "\n1.",
    RESULT_FILE
)

print(
    "\n2.",
    OUTPUT_DIR / "best_model_summary.csv"
)

print("\n")
print("=" * 80)
print("FINAL MODEL VALIDATION COMPLETE")
print("=" * 80)