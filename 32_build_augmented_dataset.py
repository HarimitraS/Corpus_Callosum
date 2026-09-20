import os
import shutil
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE = r"E:\Corpus_Callosum"

REAL_SAGITTAL = os.path.join(
    BASE, "5_sagittal_slices_png"
)

SHARPENED_GAN = os.path.join(
    BASE,
    "24_gan_generation",
    "sharpened_wgan_gp"
)

OUTPUT = os.path.join(
    BASE,
    "32_augmented_dataset"
)

NORMAL_OUT = os.path.join(
    OUTPUT, "normal"
)

DELAYED_REAL_OUT = os.path.join(
    OUTPUT, "delayed_real"
)

DELAYED_SYNTH_OUT = os.path.join(
    OUTPUT, "delayed_synthetic"
)

os.makedirs(NORMAL_OUT, exist_ok=True)
os.makedirs(DELAYED_REAL_OUT, exist_ok=True)
os.makedirs(DELAYED_SYNTH_OUT, exist_ok=True)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

MASTER = os.path.join(
    BASE,
    "22_statistical_analysis",
    "master_dataset.csv"
)

df = pd.read_csv(MASTER)

df["myelinisation"] = (
    df["myelinisation"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nMaster dataset:")
print(df["myelinisation"].value_counts())


# ============================================================
# IDENTIFY REAL SUBJECTS
# ============================================================

normal_subjects = set(
    df.loc[
        df["myelinisation"] == "normal",
        "Subject"
    ].astype(str)
)

delayed_subjects = set(
    df.loc[
        df["myelinisation"] == "delayed",
        "Subject"
    ].astype(str)
)

print("\nReal normal subjects:", len(normal_subjects))
print("Real delayed subjects:", len(delayed_subjects))
print("Delayed:", sorted(delayed_subjects))


records = []


# ============================================================
# COPY REAL NORMAL IMAGES
# ============================================================

normal_count = 0

for subject in sorted(normal_subjects):

    filename = f"{subject}.png"

    source = os.path.join(
        REAL_SAGITTAL,
        filename
    )

    if not os.path.isfile(source):
        print(
            "WARNING - missing normal image:",
            source
        )
        continue

    destination = os.path.join(
        NORMAL_OUT,
        filename
    )

    shutil.copy2(
        source,
        destination
    )

    records.append({
        "Image": filename,
        "Path": destination,
        "Class": "normal",
        "Label": 0,
        "Source": "Real",
        "Subject": subject
    })

    normal_count += 1


# ============================================================
# COPY REAL DELAYED IMAGES
# ============================================================

delayed_real_count = 0

for subject in sorted(delayed_subjects):

    filename = f"{subject}.png"

    source = os.path.join(
        REAL_SAGITTAL,
        filename
    )

    if not os.path.isfile(source):
        print(
            "WARNING - missing delayed image:",
            source
        )
        continue

    destination = os.path.join(
        DELAYED_REAL_OUT,
        filename
    )

    shutil.copy2(
        source,
        destination
    )

    records.append({
        "Image": filename,
        "Path": destination,
        "Class": "delayed",
        "Label": 1,
        "Source": "Real",
        "Subject": subject
    })

    delayed_real_count += 1


# ============================================================
# FIND SYNTHETIC GAN IMAGES
# ============================================================

gan_images = []

for root, dirs, files in os.walk(SHARPENED_GAN):

    for file in files:

        if not file.lower().endswith(".png"):
            continue

        # Skip grid preview images
        if file.lower().startswith("epoch_"):
            continue

        full_path = os.path.join(
            root,
            file
        )

        gan_images.append(full_path)


gan_images.sort()

print(
    "\nAll individual sharpened GAN images found:",
    len(gan_images)
)


# ============================================================
# COPY ALL SYNTHETIC IMAGES
# ============================================================

synthetic_count = 0

for i, source in enumerate(
    gan_images,
    start=1
):

    destination_filename = (
        f"synthetic_delayed_{i:03d}.png"
    )

    destination = os.path.join(
        DELAYED_SYNTH_OUT,
        destination_filename
    )

    shutil.copy2(
        source,
        destination
    )

    records.append({
        "Image": destination_filename,
        "Path": destination,
        "Class": "delayed",
        "Label": 1,
        "Source": "Synthetic_GAN",
        "Subject": f"SYNTHETIC_{i:03d}"
    })

    synthetic_count += 1


# ============================================================
# CREATE DATASET CSV
# ============================================================

dataset_df = pd.DataFrame(records)

csv_path = os.path.join(
    OUTPUT,
    "dataset.csv"
)

dataset_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("AUGMENTED DATASET CREATED")
print("=" * 60)

print(
    "Real Normal       :",
    normal_count
)

print(
    "Real Delayed      :",
    delayed_real_count
)

print(
    "Synthetic Delayed :",
    synthetic_count
)

print("-" * 60)

print(
    "Total Normal      :",
    (dataset_df["Label"] == 0).sum()
)

print(
    "Total Delayed     :",
    (dataset_df["Label"] == 1).sum()
)

print(
    "Total Images      :",
    len(dataset_df)
)

print("-" * 60)

print("\nSource distribution:")
print(
    dataset_df["Source"].value_counts()
)

print("\nClass distribution:")
print(
    dataset_df["Class"].value_counts()
)

print("\nCSV saved to:")
print(csv_path)

print("\nDataset folder:")
print(OUTPUT)

print("=" * 60)