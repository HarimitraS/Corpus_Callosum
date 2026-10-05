import cv2
import glob
import os
import numpy as np


PROB_DIR = (
    r".\33_synthetic_delayed_150"
    r"\03b_unet_localization\probability"
)


files = sorted(
    glob.glob(
        os.path.join(
            PROB_DIR,
            "*.png"
        )
    )
)


thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]


results = []


print()
print("=" * 80)
print("U-NET RAW PROBABILITY DIAGNOSTIC")
print("=" * 80)

print(
    f"Probability maps: {len(files)}"
)


for f in files:

    name = os.path.basename(f)

    prob_img = cv2.imread(
        f,
        cv2.IMREAD_GRAYSCALE
    )

    if prob_img is None:
        continue


    # Saved probability maps are 0-255
    prob = (
        prob_img.astype(
            np.float32
        ) / 255.0
    )


    maximum = float(
        np.max(prob)
    )

    mean = float(
        np.mean(prob)
    )

    p90 = float(
        np.percentile(
            prob,
            90
        )
    )

    p95 = float(
        np.percentile(
            prob,
            95
        )
    )

    p99 = float(
        np.percentile(
            prob,
            99
        )
    )


    threshold_counts = []

    for t in thresholds:

        count = int(
            np.count_nonzero(
                prob >= t
            )
        )

        threshold_counts.append(
            count
        )


    results.append(
        (
            name,
            maximum,
            mean,
            p90,
            p95,
            p99,
            *threshold_counts
        )
    )


# ==========================================================
# SUMMARY
# ==========================================================

max_values = [
    r[1]
    for r in results
]

mean_values = [
    r[2]
    for r in results
]


print()
print("PROBABILITY STATISTICS")
print("-" * 80)

for label, values in [
    ("Maximum", max_values),
    ("Mean", mean_values)
]:

    p = np.percentile(
        values,
        [0, 10, 25, 50, 75, 90, 95, 99, 100]
    )

    print()
    print(label)

    for percentile, value in zip(
        [0, 10, 25, 50, 75, 90, 95, 99, 100],
        p
    ):

        print(
            f"  {percentile:>3}% : {value:.4f}"
        )


# ==========================================================
# AREA STATISTICS AT EACH THRESHOLD
# ==========================================================

print()
print()
print("MASK AREA DISTRIBUTION AT DIFFERENT THRESHOLDS")
print("-" * 80)

for i, t in enumerate(thresholds):

    index = 6 + i

    values = [
        r[index]
        for r in results
    ]

    p = np.percentile(
        values,
        [0, 10, 25, 50, 75, 90, 95, 99, 100]
    )

    print()
    print(
        f"THRESHOLD = {t:.2f}"
    )

    for percentile, value in zip(
        [0, 10, 25, 50, 75, 90, 95, 99, 100],
        p
    ):

        print(
            f"  {percentile:>3}% : {value:.1f}"
        )


# ==========================================================
# HOW MANY HAVE MEANINGFUL AREA?
# ==========================================================

print()
print()
print("COUNTS ABOVE THRESHOLDS")
print("-" * 80)

for i, t in enumerate(thresholds):

    index = 6 + i

    values = [
        r[index]
        for r in results
    ]

    print()
    print(
        f"Probability threshold {t:.2f}"
    )

    for minimum_area in [
        10,
        25,
        50,
        100,
        250,
        500
    ]:

        count = sum(
            x >= minimum_area
            for x in values
        )

        print(
            f"  area >= {minimum_area:4d} : "
            f"{count:3d}/{len(values)}"
        )


# ==========================================================
# TOP 20 BY MAXIMUM PROBABILITY
# ==========================================================

print()
print()
print("TOP 20 BY MAXIMUM PROBABILITY")
print("-" * 80)

for row in sorted(
    results,
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(
        f"{row[0]:30s} "
        f"max={row[1]:.3f} "
        f"mean={row[2]:.3f} "
        f"p95={row[4]:.3f} "
        f"p99={row[5]:.3f} "
        f">=.5={row[10]}"
    )


print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)