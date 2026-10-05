import cv2
import glob
import os
import numpy as np


IMAGE_DIR = r".\dataset\images"
MASK_DIR = r".\dataset\masks"


def percentile_report(values):

    values = np.array(values, dtype=float)

    return {
        "min": np.min(values),
        "p10": np.percentile(values, 10),
        "p25": np.percentile(values, 25),
        "median": np.percentile(values, 50),
        "p75": np.percentile(values, 75),
        "p90": np.percentile(values, 90),
        "max": np.max(values),
    }


image_files = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.png"
        )
    )
)

mask_files = sorted(
    glob.glob(
        os.path.join(
            MASK_DIR,
            "*.png"
        )
    )
)


print()
print("=" * 75)
print("U-NET TRAINING DATA INSPECTION")
print("=" * 75)

print(
    f"Training images : {len(image_files)}"
)

print(
    f"Training masks  : {len(mask_files)}"
)


image_shapes = []
mask_shapes = []

mask_areas = []
mask_widths = []
mask_heights = []

image_stds = []

paired = []


for image_path in image_files:

    name = os.path.basename(image_path)

    mask_path = os.path.join(
        MASK_DIR,
        name
    )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(
            f"WARNING: Cannot read {name}"
        )
        continue

    if mask is None:
        print(
            f"WARNING: Missing mask for {name}"
        )
        continue


    h, w = image.shape

    mh, mw = mask.shape

    image_shapes.append(
        (w, h)
    )

    mask_shapes.append(
        (mw, mh)
    )


    image_stds.append(
        float(np.std(image))
    )


    binary = mask > 127

    area = int(
        np.count_nonzero(binary)
    )

    mask_areas.append(
        area
    )


    if area > 0:

        ys, xs = np.where(binary)

        width = int(
            xs.max() -
            xs.min() +
            1
        )

        height = int(
            ys.max() -
            ys.min() +
            1
        )

    else:

        width = 0
        height = 0


    mask_widths.append(
        width
    )

    mask_heights.append(
        height
    )


    paired.append(
        (
            name,
            w,
            h,
            area,
            width,
            height,
            float(np.std(image))
        )
    )


print()
print("IMAGE DIMENSIONS")
print("-" * 50)

print(
    sorted(
        set(image_shapes)
    )
)


print()
print("MASK DIMENSIONS")
print("-" * 50)

print(
    sorted(
        set(mask_shapes)
    )
)


print()
print("IMAGE INTENSITY STD")
print("-" * 50)

print(
    percentile_report(
        image_stds
    )
)


print()
print("GROUND-TRUTH MASK AREA")
print("-" * 50)

print(
    percentile_report(
        mask_areas
    )
)


print()
print("GROUND-TRUTH MASK WIDTH")
print("-" * 50)

print(
    percentile_report(
        mask_widths
    )
)


print()
print("GROUND-TRUTH MASK HEIGHT")
print("-" * 50)

print(
    percentile_report(
        mask_heights
    )
)


print()
print("TRAINING SAMPLE DETAILS")
print("-" * 75)

for row in paired:

    print(
        f"{row[0]:35s} "
        f"image={row[1]}x{row[2]} "
        f"mask_area={row[3]:5d} "
        f"mask_bbox={row[4]}x{row[5]} "
        f"image_std={row[6]:.2f}"
    )


print()
print("=" * 75)
print("INSPECTION COMPLETE")
print("=" * 75)