import os
import cv2
import nibabel as nib
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

ROOT = r"E:\Corpus_Callosum"

MRI_ROOT = r"E:\infant_mri_dataset"

OUTPUT_DIR = os.path.join(
    ROOT,
    "24_gan_generation",
    "delayed_images"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# DELAYED SUBJECTS
# ==========================================================

DELAYED_SUBJECTS = [

    "s0147",
    "s0160",
    "s0341",
    "s0412"

]

# ==========================================================
# PROCESS
# ==========================================================

print("="*60)
print("Extracting Delayed MRI Images")
print("="*60)

saved = 0

for subject in DELAYED_SUBJECTS:

    nii_path = os.path.join(

        MRI_ROOT,

        subject,

        "t1.nii.gz"

    )

    if not os.path.exists(nii_path):

        print(f"Missing : {subject}")

        continue

    img = nib.load(nii_path)

    volume = img.get_fdata()

    sagittal_index = volume.shape[0] // 2

    slice_img = volume[sagittal_index, :, :]

    slice_img = np.rot90(slice_img)

    slice_img = cv2.normalize(

        slice_img,

        None,

        0,

        255,

        cv2.NORM_MINMAX

    )

    slice_img = slice_img.astype(np.uint8)

    out_path = os.path.join(

        OUTPUT_DIR,

        subject + ".png"

    )

    cv2.imwrite(

        out_path,

        slice_img

    )

    print(f"Saved : {subject}")

    saved += 1

print("="*60)
print(f"Total Saved : {saved}")
print("="*60)