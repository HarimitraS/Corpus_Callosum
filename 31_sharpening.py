# ============================================================
# 31_sharpening.py
# WGAN-GP MRI Image Sharpening
# ============================================================

import os
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"E:\Corpus_Callosum\24_gan_generation\generated_images_wgan_gp"

OUTPUT_DIR = r"E:\Corpus_Callosum\24_gan_generation\sharpened_wgan_gp"


# ============================================================
# SETTINGS
# ============================================================

# Mild sharpening parameters
GAUSSIAN_SIGMA = 1.0
SHARPEN_AMOUNT = 1.2


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SHARPEN FUNCTION
# ============================================================

def sharpen_image(image):

    # Convert to float for safe processing
    image_float = image.astype(
        np.float32
    )

    # Slight Gaussian blur
    blurred = cv2.GaussianBlur(
        image_float,
        (0, 0),
        GAUSSIAN_SIGMA
    )

    # Unsharp masking
    sharpened = (
        image_float
        + SHARPEN_AMOUNT
        * (image_float - blurred)
    )

    # Keep pixel values valid
    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(
        np.uint8
    )


# ============================================================
# FIND ALL PNG IMAGES
# ============================================================

image_paths = []


for root, dirs, files in os.walk(
    INPUT_DIR
):

    for file in files:

        if file.lower().endswith(
            ".png"
        ):

            image_paths.append(
                os.path.join(
                    root,
                    file
                )
            )


image_paths.sort()


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WGAN-GP MRI SHARPENING")
print("=" * 70)

print(
    "Input :",
    INPUT_DIR
)

print(
    "Output:",
    OUTPUT_DIR
)

print(
    "Images:",
    len(image_paths)
)

print("=" * 70)


# ============================================================
# PROCESS IMAGES
# ============================================================

processed = 0
failed = 0


for image_path in image_paths:

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            print(
                "Could not read:",
                image_path
            )

            failed += 1

            continue


        # ----------------------------------------------------
        # Sharpen
        # ----------------------------------------------------

        sharpened = sharpen_image(
            image
        )


        # ----------------------------------------------------
        # Preserve epoch folder
        # ----------------------------------------------------

        relative_path = os.path.relpath(
            image_path,
            INPUT_DIR
        )


        output_path = os.path.join(
            OUTPUT_DIR,
            relative_path
        )


        output_folder = os.path.dirname(
            output_path
        )


        os.makedirs(
            output_folder,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        success = cv2.imwrite(
            output_path,
            sharpened
        )


        if success:

            processed += 1

        else:

            failed += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed % 10 == 0:

            print(
                f"Processed: {processed}/{len(image_paths)}"
            )


    except Exception as e:

        print(
            "ERROR:",
            image_path
        )

        print(
            e
        )

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SHARPENING COMPLETE")
print("=" * 70)

print(
    "Total images found :",
    len(image_paths)
)

print(
    "Successfully saved :",
    processed
)

print(
    "Failed             :",
    failed
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)