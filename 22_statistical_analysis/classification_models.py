import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================================
# PATHS
# ==========================================================

DATA_PATH = r"22_statistical_analysis\feature_selection\selected_feature_dataset.csv"

OUTPUT_DIR = r"22_statistical_analysis\classification_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_PATH)

TARGET = "myelinisation"

# IMPORTANT:
# Subject is an identifier, NOT a feature.
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
print("CLASSIFICATION")
print("==========================================")

print(f"Dataset shape : {df.shape}")
print(f"Features      : {X.shape[1]}")

print("\nClass distribution:")
print(df[TARGET].value_counts())

# ==========================================================
# CHECK DATA
# ==========================================================

if X.isna().sum().sum() > 0:

    print("\nERROR: Missing values detected.")

    print(
        X.isna().sum()[
            X.isna().sum() > 0
        ]
    )

    raise ValueError(
        "Dataset contains missing values."
    )

if not all(
    np.issubdtype(
        dtype,
        np.number
    )
    for dtype in X.dtypes
):

    print("\nERROR: Non-numeric feature detected.")

    print(X.dtypes)

    raise ValueError(
        "All features must be numeric."
    )

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples :", len(X_train))
print("Testing samples  :", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

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
                probability=True,
                class_weight="balanced",
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
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# ==========================================================
# TRAIN + EVALUATE
# ==========================================================

for name, model in models.items():

    print("\n------------------------------------------")
    print(name)
    print("------------------------------------------")

    # ------------------------------------------
    # TRAIN
    # ------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # ------------------------------------------
    # TEST PREDICTION
    # ------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # ------------------------------------------
    # PROBABILITY
    # ------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_prob = None

    # ------------------------------------------
    # METRICS
    # ------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    if y_prob is not None:

        auc = roc_auc_score(
            y_test,
            y_prob
        )

    else:

        auc = np.nan

    # ------------------------------------------
    # CROSS-VALIDATION
    # ------------------------------------------

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )

    cv_f1_mean = cv_scores.mean()
    cv_f1_std = cv_scores.std()

    # ------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # ------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print(
        f"ROC-AUC        : {auc:.4f}"
    )

    print(
        f"CV F1          : "
        f"{cv_f1_mean:.4f} ± {cv_f1_std:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    # ------------------------------------------
    # STORE RESULTS
    # ------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": auc,

        "CV_F1_Mean": cv_f1_mean,

        "CV_F1_STD": cv_f1_std

    })

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "F1_Score",
    ascending=False
)

# ==========================================================
# SAVE
# ==========================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n==========================================")
print("CLASSIFICATION COMPLETE")
print("==========================================")

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved to:")

print(
    RESULTS_PATH
)

print("==========================================")