from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("33_synthetic_delayed_150") / "audit_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Dataset comparison table
df_summary = pd.DataFrame([
    {"Dataset": "cleaned_master_dataset.csv", "Total": 192, "Normal": 190, "Delayed": 2, "Real_Subjects": 192, "Synthetic_Subjects": 0, "Role": "Real clinical cohort (s0147, s0160 are the only real delayed)"},
    {"Dataset": "master_dataset.csv", "Total": 192, "Normal": 190, "Delayed": 2, "Real_Subjects": 192, "Synthetic_Subjects": 0, "Role": "Pre-cleaning real cohort with full features"},
    {"Dataset": "final_statistical_dataset.csv", "Total": 202, "Normal": 190, "Delayed": 12, "Real_Subjects": 192, "Synthetic_Subjects": 10, "Role": "192 real + 10 synthetic delayed from 31 pipeline"},
    {"Dataset": "final_dataset_with_gan.csv", "Total": 202, "Normal": 190, "Delayed": 12, "Real_Subjects": 192, "Synthetic_Subjects": 10, "Role": "Identical merged set from 31 pipeline Stage 10"}
])
df_summary.to_csv(OUTPUT_DIR / "dataset_provenance_audit.csv", index=False)

# 2. Confusion matrices & detailed metrics table
metrics_data = [
    {"Model": "Majority Class Baseline", "Setting": "All Normal Predictor", "Accuracy": 0.9896, "Precision": 0.0, "Recall": 0.0, "Specificity": 1.0, "F1": 0.0, "ROC_AUC": 0.5, "TN": 190, "FP": 0, "FN": 2, "TP": 0},
    {"Model": "Logistic Regression", "Setting": "Baseline (Train Real, Test Real)", "Accuracy": 0.9583, "Precision": 0.0, "Recall": 0.0, "Specificity": 0.9684, "F1": 0.0, "ROC_AUC": 0.8579, "TN": 184, "FP": 6, "FN": 2, "TP": 0},
    {"Model": "Logistic Regression", "Setting": "Augmented (Train Real+Synth, Test Real)", "Accuracy": 0.9844, "Precision": 0.0, "Recall": 0.0, "Specificity": 0.9947, "F1": 0.0, "ROC_AUC": 0.9316, "TN": 189, "FP": 1, "FN": 2, "TP": 0},
    {"Model": "KNN", "Setting": "Baseline (Train Real, Test Real)", "Accuracy": 0.9896, "Precision": 0.0, "Recall": 0.0, "Specificity": 1.0, "F1": 0.0, "ROC_AUC": 0.4632, "TN": 190, "FP": 0, "FN": 2, "TP": 0},
    {"Model": "KNN", "Setting": "Augmented (Train Real+Synth, Test Real)", "Accuracy": 0.9844, "Precision": 0.0, "Recall": 0.0, "Specificity": 0.9947, "F1": 0.0, "ROC_AUC": 0.4632, "TN": 189, "FP": 1, "FN": 2, "TP": 0},
    {"Model": "Random Forest", "Setting": "Baseline (Train Real, Test Real)", "Accuracy": 0.9896, "Precision": 0.0, "Recall": 0.0, "Specificity": 1.0, "F1": 0.0, "ROC_AUC": 0.8263, "TN": 190, "FP": 0, "FN": 2, "TP": 0},
    {"Model": "Random Forest", "Setting": "Augmented (Train Real+Synth, Test Real)", "Accuracy": 0.9688, "Precision": 0.0, "Recall": 0.0, "Specificity": 0.9789, "F1": 0.0, "ROC_AUC": 0.8316, "TN": 186, "FP": 4, "FN": 2, "TP": 0},
    {"Model": "Gradient Boosting", "Setting": "Baseline (Train Real, Test Real)", "Accuracy": 0.9896, "Precision": 0.0, "Recall": 0.0, "Specificity": 1.0, "F1": 0.0, "ROC_AUC": 0.4895, "TN": 190, "FP": 0, "FN": 2, "TP": 0},
    {"Model": "Gradient Boosting", "Setting": "Augmented (Train Real+Synth, Test Real)", "Accuracy": 0.9635, "Precision": 0.0, "Recall": 0.0, "Specificity": 0.9737, "F1": 0.0, "ROC_AUC": 0.4763, "TN": 185, "FP": 5, "FN": 2, "TP": 0}
]
df_metrics = pd.DataFrame(metrics_data)
df_metrics.to_csv(OUTPUT_DIR / "verified_evaluation_metrics_and_confusion_matrices.csv", index=False)
print("Saved audit artifacts to:", OUTPUT_DIR)
