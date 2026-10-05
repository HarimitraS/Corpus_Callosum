from pathlib import Path

import cv2
import numpy as np


# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(r"E:\Corpus_Callosum\33_synthetic_delayed_150")

INPUT_DIR = ROOT / "candidates"
OUTPUT_DIR = ROOT / "02_preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# EXISTING PREPROCESSING ALGORITHM
# ==========================================================

def preprocess_image(image):

    # 1. Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # 2. Intensity normalization
    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    image = image.astype(np.uint8)

    # 3. Non-Local Means denoising
    image = cv2.fastNlMeansDenoising(
        image,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 4. CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    image = clahe.apply(image)

    # 5. Edge-preserving enhancement
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0
    )

    enhanced = cv2.addWeighted(
        image,
        1.3,
        blurred,
        -0.3,
        0
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    return enhanced


# ==========================================================
# MAIN
# ==========================================================

def main():

    images = sorted(
        INPUT_DIR.glob("*.png")
    )

    print("=" * 60)
    print("STAGE 2 — GAN SYNTHETIC MRI PREPROCESSING")
    print("=" * 60)

    print(f"Input images : {len(images)}")

    if not images:
        print("No PNG images found.")
        return

    success = 0
    failed = 0

    for image_path in images:

        try:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                raise RuntimeError(
                    f"Could not read {image_path}"
                )

            processed = preprocess_image(
                image
            )

            output_path = (
                OUTPUT_DIR /
                image_path.name
            )

            if not cv2.imwrite(
                str(output_path),
                processed
            ):
                raise RuntimeError(
                    "cv2.imwrite failed"
                )

            success += 1

        except Exception as e:

            print(
                f"FAILED: {image_path.name} -> {e}"
            )

            failed += 1

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Successful : {success}")
    print(f"Failed     : {failed}")
    print(f"Output     : {OUTPUT_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()