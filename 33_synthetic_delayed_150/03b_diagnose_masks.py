import cv2
import glob
import numpy as np
import os

MASK_DIR = r".\33_synthetic_delayed_150\03b_unet_localization\masks"

files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))

areas = []
widths = []
heights = []
records = []

for f in files:

    mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    binary = mask > 0

    area = int(np.count_nonzero(binary))

    if area > 0:
        ys, xs = np.where(binary)

        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
    else:
        width = 0
        height = 0

    name = os.path.basename(f)

    areas.append(area)
    widths.append(width)
    heights.append(height)

    records.append(
        (area, width, height, name)
    )


print()
print("=" * 70)
print("U-NET MASK DIAGNOSTIC")
print("=" * 70)

print(f"Total masks: {len(records)}")

print()
print("MASK AREA PERCENTILES")
print("-" * 40)

if areas:
    p = np.percentile(
        areas,
        [0, 10, 25, 50, 75, 90, 95, 99, 100]
    )

    for percentile, value in zip(
        [0, 10, 25, 50, 75, 90, 95, 99, 100],
        p
    ):
        print(
            f"{percentile:>3}% : {value:.2f}"
        )


print()
print("MASK WIDTH PERCENTILES")
print("-" * 40)

if widths:
    p = np.percentile(
        widths,
        [0, 10, 25, 50, 75, 90, 95, 99, 100]
    )

    for percentile, value in zip(
        [0, 10, 25, 50, 75, 90, 95, 99, 100],
        p
    ):
        print(
            f"{percentile:>3}% : {value:.2f}"
        )


print()
print("MASK HEIGHT PERCENTILES")
print("-" * 40)

if heights:
    p = np.percentile(
        heights,
        [0, 10, 25, 50, 75, 90, 95, 99, 100]
    )

    for percentile, value in zip(
        [0, 10, 25, 50, 75, 90, 95, 99, 100],
        p
    ):
        print(
            f"{percentile:>3}% : {value:.2f}"
        )


print()
print("LARGEST 20 MASKS")
print("-" * 70)

largest = sorted(
    records,
    key=lambda x: x[0],
    reverse=True
)[:20]

for area, width, height, name in largest:

    print(
        f"{name:35s} "
        f"area={area:5d} "
        f"width={width:3d} "
        f"height={height:3d}"
    )


print()
print("SMALLEST 20 NON-EMPTY MASKS")
print("-" * 70)

non_empty = [
    r for r in records
    if r[0] > 0
]

smallest = sorted(
    non_empty,
    key=lambda x: x[0]
)[:20]

for area, width, height, name in smallest:

    print(
        f"{name:35s} "
        f"area={area:5d} "
        f"width={width:3d} "
        f"height={height:3d}"
    )


print()
print("=" * 70)