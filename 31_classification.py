# ============================================================
# 31_classification.py
# Corpus Callosum Normal vs Delayed Myelinisation
# BASELINE CLASSIFICATION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

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
    "31_classification_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Small feature count because we only have 2 delayed subjects
TOP_K_FEATURES = 15


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CORPUS CALLOSUM CLASSIFICATION")
print("NORMAL vs DELAYED MYELINISATION")
print("=" * 70)

print("\nTarget:")
print("    myelinisation")

print("\nClasses:")
print("    Normal  = 0")
print("    Delayed = 1")

print("\nIMPORTANT:")
print("There are only 2 real delayed subjects.")
print("Results are exploratory and should not be interpreted")
print("as a clinically validated classifier.")

print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

master = pd.read_csv(MASTER_PATH)
morph = pd.read_csv(MORPH_PATH)
texture = pd.read_csv(TEXTURE_PATH)
radiomic = pd.read_csv(RADIOMIC_PATH)

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
        .str.replace(".png", "", regex=False)
        .str.replace(".nii.gz", "", regex=False)
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
# REMOVE DUPLICATE SUBJECTS
# ============================================================

master = master.drop_duplicates(
    subset="Subject"
)

morph = morph.drop_duplicates(
    subset="Subject"
)

texture = texture.drop_duplicates(
    subset="Subject"
)

radiomic = radiomic.drop_duplicates(
    subset="Subject"
)


# ============================================================
# CREATE TARGET
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
        ["normal", "delayed"]
    )
].copy()


labels["Target"] = labels[
    "myelinisation"
].map(
    {
        "normal": 0,
        "delayed": 1
    }
)


print("\nMyelinisation distribution")
print("-" * 70)

print(
    labels["myelinisation"].value_counts()
)


# ============================================================
# SHOW DELAYED SUBJECTS
# ============================================================

print("\nDelayed subjects")
print("-" * 70)

print(
    labels[
        labels["Target"] == 1
    ][
        ["Subject", "myelinisation"]
    ].to_string(index=False)
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
# MERGE ALL FEATURES BY SUBJECT
# ============================================================

print("\nMerging datasets by Subject...")

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
# VERIFY DELAYED SUBJECTS AFTER MERGE
# ============================================================

print("\nDelayed subjects after feature merge")
print("-" * 70)

delayed_after_merge = dataset[
    dataset["Target"] == 1
]

print(
    delayed_after_merge[
        ["Subject", "myelinisation"]
    ].to_string(index=False)
)


# ============================================================
# CLASS COUNTS
# ============================================================

normal_count = (
    dataset["Target"] == 0
).sum()

delayed_count = (
    dataset["Target"] == 1
).sum()


print("\nFinal class distribution")
print("-" * 70)

print("Normal :", normal_count)
print("Delayed:", delayed_count)
print("Total  :", len(dataset))


if delayed_count < 2:

    raise ValueError(
        "\nERROR: Fewer than 2 delayed subjects survived "
        "the feature merge."
    )


# ============================================================
# BUILD FEATURE MATRIX
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


y = dataset["Target"]


# ============================================================
# NUMERIC FEATURES ONLY
# ============================================================

X = X.select_dtypes(
    include=[np.number]
)


# ============================================================
# REPLACE INFINITY
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# REMOVE ALL-NaN FEATURES
# ============================================================

all_nan_columns = X.columns[
    X.isna().all()
]


if len(all_nan_columns) > 0:

    print(
        "\nRemoving all-NaN features:",
        len(all_nan_columns)
    )

    X = X.drop(
        columns=all_nan_columns
    )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = X.columns[
    X.nunique(dropna=True) <= 1
]


print(
    "\nConstant features found:",
    len(constant_columns)
)


if len(constant_columns) > 0:

    X = X.drop(
        columns=constant_columns
    )


print(
    "Features after constant removal:",
    X.shape[1]
)


# ============================================================
# REMOVE DUPLICATE FEATURE COLUMNS
# ============================================================

duplicate_feature_mask = X.T.duplicated()

duplicate_feature_count = (
    duplicate_feature_mask.sum()
)


print(
    "Duplicate feature columns:",
    duplicate_feature_count
)


if duplicate_feature_count > 0:

    X = X.loc[
        :,
        ~duplicate_feature_mask
    ]


print(
    "Final feature count:",
    X.shape[1]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE
        )
}


# ============================================================
# PIPELINE
# ============================================================

def create_pipeline(model):

    k = min(
        TOP_K_FEATURES,
        X.shape[1]
    )

    return Pipeline([

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
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=k
            )
        ),

        (
            "classifier",
            model
        )
    ])


# ============================================================
# LEAVE-ONE-OUT
# ============================================================

print("\n")
print("=" * 70)
print("LEAVE-ONE-OUT EVALUATION")
print("=" * 70)

print(
    "\nSamples:",
    len(X)
)

print(
    "Delayed:",
    delayed_count
)

print(
    "Feature selection:",
    TOP_K_FEATURES,
    "features"
)

print(
    "\nThis may take some time because every subject"
)

print(
    "is evaluated separately."
)


loo = LeaveOneOut()


# ============================================================
# CUSTOM SAFE LOOCV
# ============================================================

def evaluate_model(
    model_name,
    model
):

    print(
        f"\nRunning {model_name}..."
    )

    predictions = np.zeros(
        len(y),
        dtype=int
    )

    probabilities = np.zeros(
        len(y),
        dtype=float
    )


    for fold_number, (
        train_idx,
        test_idx
    ) in enumerate(
        loo.split(X),
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]


        pipeline = create_pipeline(
            model
        )


        pipeline.fit(
            X_train,
            y_train
        )


        prediction = pipeline.predict(
            X_test
        )


        predictions[
            test_idx[0]
        ] = prediction[0]


        try:

            probability = (
                pipeline.predict_proba(
                    X_test
                )[0][1]
            )

        except Exception:

            probability = float(
                pipeline.decision_function(
                    X_test
                )[0]
            )


        probabilities[
            test_idx[0]
        ] = probability


        if fold_number % 25 == 0:

            print(
                f"  Fold {fold_number}/{len(X)}"
            )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )


    try:

        roc_auc = roc_auc_score(
            y,
            probabilities
        )

    except Exception:

        roc_auc = np.nan


    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )


    tn, fp, fn, tp = cm.ravel()


    print(
        f"\n{model_name}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


    return {

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1,

        "ROC-AUC":
            roc_auc,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    }


# ============================================================
# RUN ALL MODELS
# ============================================================

results = []


for model_name, model in models.items():

    result = evaluate_model(
        model_name,
        model
    )

    results.append(
        result
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "baseline_classification_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE MERGED DATASET
# ============================================================

merged_path = os.path.join(
    OUTPUT_DIR,
    "merged_classification_dataset.csv"
)


dataset.to_csv(
    merged_path,
    index=False
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_path = os.path.join(
    OUTPUT_DIR,
    "classification_features.txt"
)


with open(
    feature_path,
    "w",
    encoding="utf-8"
) as f:

    for feature in X.columns:

        f.write(
            str(feature) + "\n"
        )


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 100)
print("BASELINE CLASSIFICATION RESULTS")
print("=" * 100)


display_df = results_df.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_idx = results_df[
    "F1 Score"
].idxmax()


best_model = results_df.loc[
    best_idx
]


print("\n")
print("=" * 70)
print("BEST BASELINE MODEL")
print("=" * 70)

print(
    "Model:",
    best_model["Model"]
)

print(
    "F1:",
    f"{best_model['F1 Score'] * 100:.2f}%"
)

print(
    "Recall:",
    f"{best_model['Recall'] * 100:.2f}%"
)

print(
    "ROC-AUC:",
    f"{best_model['ROC-AUC'] * 100:.2f}%"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved:"
)

print(
    results_path
)

print(
    "\nMerged dataset:"
)

print(
    merged_path
)

print(
    "\nFeature list:"
)

print(
    feature_path
)

print("\n")
print("Next stage:")
print("GAN-augmented classification")

print("=" * 70)