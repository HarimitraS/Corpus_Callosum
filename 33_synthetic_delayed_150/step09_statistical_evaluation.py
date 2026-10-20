import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

ROOT = Path(".")
OUTPUT_DIR = ROOT / "33_synthetic_delayed_150" / "09_statistical_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_PATH = ROOT / "22_statistical_analysis" / "cleaned_master_dataset.csv"
SYNTH_PATH = ROOT / "33_synthetic_delayed_150" / "08_merged_features" / "synthetic_delayed_150_master_features.csv"

# The validated 20 features
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

orig_df = pd.read_csv(ORIGINAL_PATH)
synth_df = pd.read_csv(SYNTH_PATH)

# Build isolated synthetic delayed dataset
synth_delayed = synth_df[FEATURES].copy()
synth_delayed["Subject"] = synth_df["Subject"]
synth_delayed["myelinisation"] = "delayed"
synth_delayed["data_type"] = "synthetic"

# Prepare original clinical dataset
orig_clean = orig_df[FEATURES + ["Subject", "myelinisation"]].copy()
orig_clean["data_type"] = "real"

# Encode labels: normal=0, delayed=1
y_real = (orig_clean["myelinisation"] == "delayed").astype(int).values
X_real = orig_clean[FEATURES].values

# We evaluate TWO paradigms:
# Paradigm A (Rigorous Clinical Evaluation):
# 2-fold stratified CV on REAL clinical subjects (190 normal, 2 delayed).
# On training folds, optionally add synthetic delayed samples to balance classes.
# EVALUATION IS EXCLUSIVELY ON HELD-OUT REAL SUBJECTS! (Zero synthetic leakage into test set).

# Paradigm B (Standard CV on Mixed Dataset):
# Mixed dataset with synthetic samples partitioned across folds (demonstrates apparent metric change).

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
}

print("=" * 80)
print("STATISTICAL EVALUATION: RIGOROUS REAL HELD-OUT TEST")
print(f"Real subjects: {len(orig_clean)} (Normal: {sum(y_real==0)}, Delayed: {sum(y_real==1)})")
print(f"Synthetic available: {len(synth_delayed)}")
print("=" * 80)

# 10 Repeats of 2-Fold Stratified CV on Real
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

results_summary = []

for model_name, model_inst in models.items():
    # 1. Baseline: Train on Real, Test on Real
    acc_b, prec_b, rec_b, f1_b, auc_b = [], [], [], [], []
    # 2. Augmented: Train on Real + Synthetic, Test on Real
    acc_a, prec_a, rec_a, f1_a, auc_a = [], [], [], [], []

    for train_idx, test_idx in skf.split(X_real, y_real):
        X_tr_real, y_tr_real = X_real[train_idx], y_real[train_idx]
        X_te_real, y_te_real = X_real[test_idx], y_real[test_idx]

        # Pipeline with scaler
        scaler = StandardScaler()
        X_tr_real_s = scaler.fit_transform(X_tr_real)
        X_te_real_s = scaler.transform(X_te_real)

        # Fit Baseline
        from sklearn.base import clone
        clf_base = clone(model_inst)
        clf_base.fit(X_tr_real_s, y_tr_real)
        preds_b = clf_base.predict(X_te_real_s)
        probs_b = clf_base.predict_proba(X_te_real_s)[:, 1] if hasattr(clf_base, "predict_proba") else preds_b

        acc_b.append(accuracy_score(y_te_real, preds_b))
        prec_b.append(precision_score(y_te_real, preds_b, zero_division=0))
        rec_b.append(recall_score(y_te_real, preds_b, zero_division=0))
        f1_b.append(f1_score(y_te_real, preds_b, zero_division=0))
        try:
            auc_b.append(roc_auc_score(y_te_real, probs_b))
        except Exception:
            auc_b.append(0.5)

        # Fit Augmented (Add synthetic delayed to training fold only!)
        # Take 50 synthetic delayed cases to augment the training set
        synth_X_tr = synth_delayed[FEATURES].values
        synth_y_tr = np.ones(len(synth_X_tr), dtype=int)

        X_tr_aug = np.vstack([X_tr_real, synth_X_tr])
        y_tr_aug = np.concatenate([y_tr_real, synth_y_tr])

        scaler_aug = StandardScaler()
        X_tr_aug_s = scaler_aug.fit_transform(X_tr_aug)
        X_te_aug_s = scaler_aug.transform(X_te_real)

        clf_aug = clone(model_inst)
        clf_aug.fit(X_tr_aug_s, y_tr_aug)
        preds_a = clf_aug.predict(X_te_aug_s)
        probs_a = clf_aug.predict_proba(X_te_aug_s)[:, 1] if hasattr(clf_aug, "predict_proba") else preds_a

        acc_a.append(accuracy_score(y_te_real, preds_a))
        prec_a.append(precision_score(y_te_real, preds_a, zero_division=0))
        rec_a.append(recall_score(y_te_real, preds_a, zero_division=0))
        f1_a.append(f1_score(y_te_real, preds_a, zero_division=0))
        try:
            auc_a.append(roc_auc_score(y_te_real, probs_a))
        except Exception:
            auc_a.append(0.5)

    results_summary.append({
        "Model": model_name,
        "Setting": "Baseline (Train Real, Test Real)",
        "Accuracy": np.mean(acc_b),
        "Precision": np.mean(prec_b),
        "Recall": np.mean(rec_b),
        "F1": np.mean(f1_b),
        "ROC_AUC": np.mean(auc_b)
    })
    results_summary.append({
        "Model": model_name,
        "Setting": "Augmented (Train Real+Synth, Test Real)",
        "Accuracy": np.mean(acc_a),
        "Precision": np.mean(prec_a),
        "Recall": np.mean(rec_a),
        "F1": np.mean(f1_a),
        "ROC_AUC": np.mean(auc_a)
    })

eval_df = pd.DataFrame(results_summary)
eval_df.to_csv(OUTPUT_DIR / "rigorous_clinical_evaluation.csv", index=False)

print("\n" + eval_df.to_string(index=False))
print("\nResults saved to:", OUTPUT_DIR / "rigorous_clinical_evaluation.csv")
