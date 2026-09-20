import os
import cv2


# ==========================================================
# SETTINGS
# ==========================================================

INPUT_DIR = r"E:\Corpus_Callosum\24_gan_generation\generated_images_wgan_gp\individual"

ADDITIONAL_DIR = r"E:\Corpus_Callosum\24_gan_generation\generated_images_wgan_gp\additional_22"

OUTPUT_DIR = r"E:\Corpus_Callosum\24_gan_generation\sharpened_wgan_gp"


# Mild sharpening
GAUSSIAN_SIGMA = 1.0
SHARPEN_AMOUNT = 1.2


# ==========================================================
# SHARPEN FUNCTION
# ==========================================================

def sharpen_image(image):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        GAUSSIAN_SIGMA
    )

    sharpened = cv2.addWeighted(
        image,
        1.0 + SHARPEN_AMOUNT,
        blurred,
        -SHARPEN_AMOUNT,
        0
    )

    return sharpened


# ==========================================================
# PROCESS IMAGES
# ==========================================================

def process_directory(input_dir, output_base):

    count = 0

    for root, dirs, files in os.walk(input_dir):

        for file in files:

            if not file.lower().endswith(".png"):
                continue

            input_path = os.path.join(
                root,
                file
            )

            # Preserve subfolder structure
            relative_path = os.path.relpath(
                root,
                input_dir
            )

            output_dir = os.path.join(
                output_base,
                relative_path
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            output_path = os.path.join(
                output_dir,
                file
            )

            # Read grayscale
            image = cv2.imread(
                input_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                print(
                    "WARNING: Could not read:",
                    input_path
                )
                continue

            # Sharpen
            sharpened = sharpen_image(
                image
            )

            # Save
            cv2.imwrite(
                output_path,
                sharpened
            )

            count += 1

            print(
                f"[{count}] {file}"
            )

    return count


# ==========================================================
# START
# ==========================================================

print()
print("=" * 60)
print("WGAN-GP IMAGE SHARPENING")
print("=" * 60)

print("Original images :", INPUT_DIR)
print("Additional      :", ADDITIONAL_DIR)
print("Output          :", OUTPUT_DIR)


# ==========================================================
# SHARPEN ORIGINAL 128
# ==========================================================

print()
print("-" * 60)
print("SHARPENING EXISTING 128 IMAGES")
print("-" * 60)

count_original = process_directory(
    INPUT_DIR,
    os.path.join(
        OUTPUT_DIR,
        "existing_128"
    )
)


# ==========================================================
# SHARPEN NEW 22
# ==========================================================

print()
print("-" * 60)
print("SHARPENING ADDITIONAL 22 IMAGES")
print("-" * 60)

count_additional = process_directory(
    ADDITIONAL_DIR,
    os.path.join(
        OUTPUT_DIR,
        "additional_22"
    )
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

total = count_original + count_additional

print()
print("=" * 60)
print("SHARPENING COMPLETE")
print("=" * 60)

print(
    "Existing images sharpened :",
    count_original
)

print(
    "Additional images sharpened:",
    count_additional
)

print(
    "Total sharpened images     :",
    total
)

print(
    "Output directory           :",
    OUTPUT_DIR
)

print("=" * 60)

if total == 150:

    print()
    print("SUCCESS!")
    print("150 synthetic delayed images are sharpened.")
    print()

else:

    print()
    print("WARNING!")
    print(
        "Expected 150 images but processed",
        total
    )
    print()