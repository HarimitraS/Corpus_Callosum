import os
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# PATHS
# ==========================================================

DATA_PATH = (
    r"22_statistical_analysis"
    r"\feature_selection"
    r"\selected_feature_dataset.csv"
)

OUTPUT_DIR = (
    r"22_statistical_analysis"
    r"\classification_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "robust_model_comparison.csv"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_PATH)

TARGET = "myelinisation"

X = df.drop(
    columns=[
        TARGET,
        "Subject"
    ]
)

y = df[TARGET].map({
    "normal": 0,
    "delayed": 1
})

print("\n==========================================")
print("ROBUST MODEL EVALUATION")
print("==========================================")

print(f"Dataset shape : {df.shape}")
print(f"Features      : {X.shape[1]}")

print("\nClass distribution:")
print(df[TARGET].value_counts())

# ==========================================================
# CHECK DATA
# ==========================================================

if X.isna().sum().sum() > 0:
    raise ValueError("Missing values detected.")

if not all(
    np.issubdtype(dtype, np.number)
    for dtype in X.dtypes
):
    raise ValueError("Non-numeric feature detected.")

# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=42
            )
        )
    ]),

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
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
# REPEATED STRATIFIED CROSS VALIDATION
# ==========================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
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
# EVALUATE
# ==========================================================

for name, model in models.items():

    print("\n------------------------------------------")
    print(name)
    print("------------------------------------------")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()

    precision_mean = scores["test_precision"].mean()
    precision_std = scores["test_precision"].std()

    recall_mean = scores["test_recall"].mean()
    recall_std = scores["test_recall"].std()

    f1_mean = scores["test_f1"].mean()
    f1_std = scores["test_f1"].std()

    auc_mean = scores["test_roc_auc"].mean()
    auc_std = scores["test_roc_auc"].std()

    print(
        f"Accuracy  : "
        f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
    )

    print(
        f"Precision : "
        f"{precision_mean:.4f} ± {precision_std:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_mean:.4f} ± {recall_std:.4f}"
    )

    print(
        f"F1        : "
        f"{f1_mean:.4f} ± {f1_std:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{auc_mean:.4f} ± {auc_std:.4f}"
    )

    results.append({

        "Model": name,

        "Accuracy_Mean": accuracy_mean,
        "Accuracy_STD": accuracy_std,

        "Precision_Mean": precision_mean,
        "Precision_STD": precision_std,

        "Recall_Mean": recall_mean,
        "Recall_STD": recall_std,

        "F1_Mean": f1_mean,
        "F1_STD": f1_std,

        "ROC_AUC_Mean": auc_mean,
        "ROC_AUC_STD": auc_std

    })

# ==========================================================
# RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "F1_Mean",
    ascending=False
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n==========================================")
print("ROBUST EVALUATION COMPLETE")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest model by mean F1:")

print(
    results_df.iloc[0]["Model"]
)

print("\nSaved to:")

print(
    RESULTS_PATH
)

print("==========================================")