import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GT_DIR = os.path.join(
    BASE_DIR,
    "ground_truth_binary_masks"
)

PRED_DIR = os.path.join(
    BASE_DIR,
    "12_refined_masks"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "19_segmentation_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# LOAD MASK
# ==========================================

def load_mask(path):

    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(path)

    _, mask = cv2.threshold(
        mask,
        127,
        1,
        cv2.THRESH_BINARY
    )

    return mask.astype(np.uint8)

# ==========================================
# CONFUSION PIXELS
# ==========================================

def confusion(gt, pred):

    tp = np.sum((gt == 1) & (pred == 1))

    fp = np.sum((gt == 0) & (pred == 1))

    fn = np.sum((gt == 1) & (pred == 0))

    tn = np.sum((gt == 0) & (pred == 0))

    return tp, fp, fn, tn

# ==========================================
# METRICS
# ==========================================

def dice(tp, fp, fn):

    return (2 * tp) / (2 * tp + fp + fn + 1e-8)


def iou(tp, fp, fn):

    return tp / (tp + fp + fn + 1e-8)


def precision(tp, fp):

    return tp / (tp + fp + 1e-8)


def recall(tp, fn):

    return tp / (tp + fn + 1e-8)


def accuracy(tp, tn, fp, fn):

    return (tp + tn) / (tp + tn + fp + fn + 1e-8)


def specificity(tn, fp):

    return tn / (tn + fp + 1e-8)
# ==========================================
# EVALUATE ALL SUBJECTS
# ==========================================

results = []

gt_files = sorted(

    [

        f

        for f in os.listdir(GT_DIR)

        if f.lower().endswith(".png")

    ]

)

print(f"\nFound {len(gt_files)} Ground Truth Masks\n")

for file in tqdm(gt_files):

    gt_path = os.path.join(

        GT_DIR,

        file

    )

    pred_path = os.path.join(

        PRED_DIR,

        file

    )

    if not os.path.exists(pred_path):

        print(f"Prediction missing : {file}")

        continue

    gt = load_mask(gt_path)

    pred = load_mask(pred_path)

    if gt.shape != pred.shape:

        pred = cv2.resize(

            pred,

            (

                gt.shape[1],

                gt.shape[0]

            ),

            interpolation=cv2.INTER_NEAREST

        )

    tp, fp, fn, tn = confusion(

        gt,

        pred

    )

    d = dice(

        tp,

        fp,

        fn

    )

    j = iou(

        tp,

        fp,

        fn

    )

    p = precision(

        tp,

        fp

    )

    r = recall(

        tp,

        fn

    )

    a = accuracy(

        tp,

        tn,

        fp,

        fn

    )

    s = specificity(

        tn,

        fp

    )

    results.append({

        "Subject": file,

        "Dice": d,

        "IoU": j,

        "Precision": p,

        "Recall": r,

        "Accuracy": a,

        "Specificity": s

    })

# ==========================================
# SAVE CSV
# ==========================================

df = pd.DataFrame(results)

csv_path = os.path.join(

    OUTPUT_DIR,

    "evaluation.csv"

)

df.to_csv(

    csv_path,

    index=False

)

print("\nEvaluation CSV Saved")
print(csv_path)

# ==========================================
# SUMMARY
# ==========================================

summary_path = os.path.join(

    OUTPUT_DIR,

    "evaluation_summary.txt"

)

with open(

    summary_path,

    "w"

) as f:

    f.write("SEGMENTATION EVALUATION\n")

    f.write("========================\n\n")

    for col in [

        "Dice",

        "IoU",

        "Precision",

        "Recall",

        "Accuracy",

        "Specificity"

    ]:

        mean = df[col].mean()

        std = df[col].std()

        line = f"{col}: {mean:.4f} ± {std:.4f}"

        print(line)

        f.write(line + "\n")

print("\nSummary Saved")

print(summary_path)

print("\nFinished Successfully.")