import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# PATHS
# ==========================================================

ORIGINAL_PATH = r"22_statistical_analysis\cleaned_master_dataset.csv"

GAN_PATH = (
    r"22_statistical_analysis"
    r"\feature_selection"
    r"\selected_feature_dataset.csv"
)

OUTPUT_DIR = r"22_statistical_analysis\classification_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "gan_augmentation_comparison.csv"
)

IMPROVEMENT_PATH = os.path.join(
    OUTPUT_DIR,
    "gan_augmentation_improvement.csv"
)

# ==========================================================
# FINAL 20 FEATURES
# ==========================================================

FEATURES = [
    "square_glcm_Idmn",
    "shape2D_PerimeterSurfaceRatio",
    "CentroidY",
    "squareroot_firstorder_TotalEnergy",
    "wavelet-L_firstorder_Mean",
    "exponential_firstorder_Skewness",
    "shape2D_MajorAxisLength",
    "square_glcm_Correlation",
    "square_glcm_Idn",
    "firstorder_10Percentile",
    "squareroot_firstorder_Minimum",
    "square_firstorder_Skewness",
    "StdThickness",
    "square_glcm_ClusterShade",
    "wavelet-L_firstorder_Range",
    "glcm_Imc2",
    "wavelet-L_glcm_SumEntropy",
    "squareroot_glcm_Idmn",
    "squareroot_glcm_Correlation",
    "BoundingBoxWidth"
]

# ==========================================================
# MODELS
# ==========================================================

MODELS = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ==========================================================
# LOAD DATA
# ==========================================================

original = pd.read_csv(ORIGINAL_PATH)
gan = pd.read_csv(GAN_PATH)

# ==========================================================
# CHECK FEATURES
# ==========================================================

missing_original = [
    f for f in FEATURES
    if f not in original.columns
]

missing_gan = [
    f for f in FEATURES
    if f not in gan.columns
]

if missing_original:
    print("Missing features in ORIGINAL dataset:")
    print(missing_original)
    raise ValueError("Required original features are missing.")

if missing_gan:
    print("Missing features in GAN dataset:")
    print(missing_gan)
    raise ValueError("Required GAN features are missing.")

# ==========================================================
# PREPARE ORIGINAL DATA
# ==========================================================

original = original[
    FEATURES + ["myelinisation"]
].copy()

original = original.dropna(
    subset=FEATURES + ["myelinisation"]
)

# ==========================================================
# PREPARE GAN DATA
# ==========================================================

gan = gan[
    FEATURES + ["myelinisation"]
].copy()

gan = gan.dropna(
    subset=FEATURES + ["myelinisation"]
)

# ==========================================================
# DATASETS
# ==========================================================

datasets = {
    "Original": original,
    "GAN Augmented": gan
}

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv = RepeatedStratifiedKFold(
    n_splits=2,
    n_repeats=10,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []

# ==========================================================
# HEADER
# ==========================================================

print("\n==========================================")
print("GAN AUGMENTATION COMPARISON")
print("==========================================")

# ==========================================================
# EVALUATION
# ==========================================================

for dataset_name, data in datasets.items():

    print("\n==========================================")
    print(dataset_name)
    print("==========================================")

    X = data[FEATURES]

    y = data["myelinisation"].map({
        "normal": 0,
        "delayed": 1
    })

    print(f"Samples : {len(data)}")

    print("\nClasses:")
    print(data["myelinisation"].value_counts())

    for model_name, model in MODELS.items():

        print("\n------------------------------------------")
        print(model_name)
        print("------------------------------------------")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        row = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Samples": len(data),
            "Normal": int((y == 0).sum()),
            "Delayed": int((y == 1).sum()),

            "Accuracy_Mean":
                scores["test_accuracy"].mean(),

            "Accuracy_STD":
                scores["test_accuracy"].std(),

            "Precision_Mean":
                scores["test_precision"].mean(),

            "Precision_STD":
                scores["test_precision"].std(),

            "Recall_Mean":
                scores["test_recall"].mean(),

            "Recall_STD":
                scores["test_recall"].std(),

            "F1_Mean":
                scores["test_f1"].mean(),

            "F1_STD":
                scores["test_f1"].std(),

            "ROC_AUC_Mean":
                scores["test_roc_auc"].mean(),

            "ROC_AUC_STD":
                scores["test_roc_auc"].std()
        }

        results.append(row)

        print(
            f"Accuracy : "
            f"{row['Accuracy_Mean']:.4f} ± "
            f"{row['Accuracy_STD']:.4f}"
        )

        print(
            f"Precision: "
            f"{row['Precision_Mean']:.4f} ± "
            f"{row['Precision_STD']:.4f}"
        )

        print(
            f"Recall   : "
            f"{row['Recall_Mean']:.4f} ± "
            f"{row['Recall_STD']:.4f}"
        )

        print(
            f"F1       : "
            f"{row['F1_Mean']:.4f} ± "
            f"{row['F1_STD']:.4f}"
        )

        print(
            f"ROC-AUC  : "
            f"{row['ROC_AUC_Mean']:.4f} ± "
            f"{row['ROC_AUC_STD']:.4f}"
        )

# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================================
# IMPROVEMENT
# ==========================================================

original_results = results_df[
    results_df["Dataset"] == "Original"
].set_index("Model")

gan_results = results_df[
    results_df["Dataset"] == "GAN Augmented"
].set_index("Model")

improvement = []

for model_name in MODELS.keys():

    improvement.append({

        "Model": model_name,

        "Accuracy_Change":
            gan_results.loc[
                model_name,
                "Accuracy_Mean"
            ]
            -
            original_results.loc[
                model_name,
                "Accuracy_Mean"
            ],

        "Precision_Change":
            gan_results.loc[
                model_name,
                "Precision_Mean"
            ]
            -
            original_results.loc[
                model_name,
                "Precision_Mean"
            ],

        "Recall_Change":
            gan_results.loc[
                model_name,
                "Recall_Mean"
            ]
            -
            original_results.loc[
                model_name,
                "Recall_Mean"
            ],

        "F1_Change":
            gan_results.loc[
                model_name,
                "F1_Mean"
            ]
            -
            original_results.loc[
                model_name,
                "F1_Mean"
            ],

        "ROC_AUC_Change":
            gan_results.loc[
                model_name,
                "ROC_AUC_Mean"
            ]
            -
            original_results.loc[
                model_name,
                "ROC_AUC_Mean"
            ]
    })

improvement_df = pd.DataFrame(improvement)

improvement_df.to_csv(
    IMPROVEMENT_PATH,
    index=False
)

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n==========================================")
print("COMPARISON COMPLETE")
print("==========================================")

print("\nFull Results:")

print(
    results_df.to_string(index=False)
)

print("\n==========================================")
print("GAN IMPROVEMENT")
print("==========================================")

print(
    improvement_df.to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_PATH)
print(IMPROVEMENT_PATH)

print("==========================================")