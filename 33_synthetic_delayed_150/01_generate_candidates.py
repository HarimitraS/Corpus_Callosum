import os
import csv
import hashlib
from datetime import datetime, timezone

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = r"E:\Corpus_Callosum"

EXPERIMENT_DIR = os.path.join(
    ROOT,
    "33_synthetic_delayed_150"
)

CANDIDATE_DIR = os.path.join(
    EXPERIMENT_DIR,
    "candidates"
)

METADATA_DIR = os.path.join(
    EXPERIMENT_DIR,
    "metadata"
)

METADATA_CSV = os.path.join(
    METADATA_DIR,
    "generation_metadata.csv"
)

MODEL_PATH = os.path.join(
    ROOT,
    "24_gan_generation",
    "models_wgan_gp",
    "generator_wgan_gp_128.pth"
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

LATENT_DIM = 100
IMAGE_SIZE = 128
CHANNELS = 1

NUM_CANDIDATES = 200

# Fixed starting seed.
# Every candidate gets its own deterministic seed.
BASE_SEED = 202610050001


# ============================================================
# EXISTING SYNTHETIC DATA
# ============================================================

EXISTING_SYNTHETIC_DIRS = [
    os.path.join(
        ROOT,
        "31_gan_delayed_pipeline",
        "01_input"
    ),

    os.path.join(
        ROOT,
        "24_gan_generation"
    )
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(CANDIDATE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# ============================================================
# SHA256
# ============================================================

def sha256_file(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# GENERATOR
# EXACT WGAN-GP ARCHITECTURE
# ============================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            # 1 -> 4
            nn.ConvTranspose2d(
                LATENT_DIM,
                512,
                4,
                1,
                0,
                bias=False
            ),

            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 4 -> 8
            nn.ConvTranspose2d(
                512,
                256,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8 -> 16
            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16 -> 32
            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32 -> 64
            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1,
                bias=False
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 64 -> 128
            nn.ConvTranspose2d(
                32,
                CHANNELS,
                4,
                2,
                1,
                bias=False
            ),

            nn.Tanh()
        )


    def forward(self, x):

        return self.net(x)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("WGAN-GP — FRESH SYNTHETIC DELAYED MRI GENERATION")
print("=" * 70)

print("Device              :", DEVICE)
print("Generator checkpoint:", MODEL_PATH)
print("Candidate count     :", NUM_CANDIDATES)
print("Latent dimension    :", LATENT_DIM)
print("Image size          :", f"{IMAGE_SIZE} x {IMAGE_SIZE}")
print("Base seed           :", BASE_SEED)
print("Output directory    :", CANDIDATE_DIR)
print()


# ============================================================
# SAFETY CHECKS
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        f"Generator checkpoint not found:\n{MODEL_PATH}"
    )


# Do not accidentally overwrite a previous experiment.

existing_candidates = [
    f
    for f in os.listdir(CANDIDATE_DIR)
    if f.lower().endswith(".png")
]

if existing_candidates:

    raise RuntimeError(
        "\nCandidate directory is not empty.\n"
        "Existing candidate PNG files were detected.\n"
        "This script will NOT overwrite them.\n\n"
        f"Directory:\n{CANDIDATE_DIR}\n\n"
        "If you intentionally want to continue an existing "
        "generation experiment, handle that separately."
    )


# ============================================================
# CHECKPOINT HASH
# ============================================================

checkpoint_hash = sha256_file(MODEL_PATH)

print("Checkpoint SHA256:")
print(checkpoint_hash)
print()


# ============================================================
# INDEX EXISTING SYNTHETIC IMAGES
# ============================================================

print("Indexing existing synthetic PNG files...")

existing_hashes = {}

existing_file_count = 0

for directory in EXISTING_SYNTHETIC_DIRS:

    if not os.path.isdir(directory):
        continue

    for root_dir, _, files in os.walk(directory):

        for filename in files:

            if not filename.lower().endswith(".png"):
                continue

            full_path = os.path.join(
                root_dir,
                filename
            )

            try:

                file_hash = sha256_file(full_path)

                existing_hashes[file_hash] = full_path

                existing_file_count += 1

            except Exception as e:

                print(
                    "WARNING: Could not hash:",
                    full_path,
                    "|",
                    e
                )


print(
    "Existing PNG files indexed:",
    existing_file_count
)

print(
    "Unique existing hashes    :",
    len(existing_hashes)
)

print()


# ============================================================
# LOAD GENERATOR
# ============================================================

print("Loading WGAN-GP generator...")

netG = Generator().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

netG.load_state_dict(state_dict)

netG.eval()

print("Generator loaded successfully.")
print()


# ============================================================
# METADATA
# ============================================================

generation_timestamp = datetime.now(
    timezone.utc
).isoformat()


metadata_rows = []

duplicate_count = 0
invalid_count = 0
generated_count = 0


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("=" * 70)
print("GENERATING CANDIDATES")
print("=" * 70)
print()


with torch.no_grad():

    for i in range(NUM_CANDIDATES):

        candidate_id = i + 1

        latent_seed = BASE_SEED + i

        # Deterministic CPU generator.
        rng = torch.Generator(
            device="cpu"
        )

        rng.manual_seed(latent_seed)

        noise = torch.randn(
            1,
            LATENT_DIM,
            1,
            1,
            generator=rng,
            device=DEVICE
        )

        fake = netG(noise)

        # Convert [-1, 1] -> [0, 255]

        image_tensor = (
            (fake + 1.0) / 2.0
        ) * 255.0

        image_tensor = image_tensor.clamp(
            0,
            255
        )

        image_array = (
            image_tensor
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # Basic generation validity

        if image_array.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE
        ):

            invalid_count += 1

            metadata_rows.append({

                "candidate_id":
                    f"candidate_{candidate_id:06d}",

                "latent_seed":
                    latent_seed,

                "generator_checkpoint":
                    MODEL_PATH,

                "checkpoint_sha256":
                    checkpoint_hash,

                "generation_timestamp":
                    generation_timestamp,

                "filename":
                    "",

                "sha256":
                    "",

                "width":
                    image_array.shape[1]
                    if len(image_array.shape) >= 2
                    else "",

                "height":
                    image_array.shape[0]
                    if len(image_array.shape) >= 2
                    else "",

                "min_intensity":
                    "",

                "max_intensity":
                    "",

                "mean_intensity":
                    "",

                "std_intensity":
                    "",

                "validation_status":
                    "generation_invalid",

                "validation_reason":
                    "incorrect_dimensions"

            })

            continue


        # Save as grayscale PNG

        image = Image.fromarray(
            image_array,
            mode="L"
        )

        filename = (
            f"candidate_{candidate_id:06d}.png"
        )

        output_path = os.path.join(
            CANDIDATE_DIR,
            filename
        )

        image.save(
            output_path,
            format="PNG"
        )


        # Hash AFTER PNG encoding

        file_hash = sha256_file(
            output_path
        )


        # Exact duplicate check

        if file_hash in existing_hashes:

            duplicate_count += 1

            validation_status = "duplicate_existing"

            validation_reason = (
                "SHA256 matches an existing "
                "synthetic PNG"
            )

        else:

            generated_count += 1

            validation_status = "generated"

            validation_reason = (
                "Fresh candidate generated "
                "from deterministic latent seed"
            )


        metadata_rows.append({

            "candidate_id":
                f"candidate_{candidate_id:06d}",

            "latent_seed":
                latent_seed,

            "generator_checkpoint":
                MODEL_PATH,

            "checkpoint_sha256":
                checkpoint_hash,

            "generation_timestamp":
                generation_timestamp,

            "filename":
                filename,

            "sha256":
                file_hash,

            "width":
                image_array.shape[1],

            "height":
                image_array.shape[0],

            "min_intensity":
                int(image_array.min()),

            "max_intensity":
                int(image_array.max()),

            "mean_intensity":
                float(image_array.mean()),

            "std_intensity":
                float(image_array.std()),

            "validation_status":
                validation_status,

            "validation_reason":
                validation_reason

        })


        print(
            f"[{candidate_id:03d}/{NUM_CANDIDATES}] "
            f"seed={latent_seed} "
            f"mean={image_array.mean():.2f} "
            f"std={image_array.std():.2f} "
            f"status={validation_status}"
        )


# ============================================================
# SAVE METADATA
# ============================================================

fieldnames = [

    "candidate_id",
    "latent_seed",
    "generator_checkpoint",
    "checkpoint_sha256",
    "generation_timestamp",
    "filename",
    "sha256",
    "width",
    "height",
    "min_intensity",
    "max_intensity",
    "mean_intensity",
    "std_intensity",
    "validation_status",
    "validation_reason"

]


with open(
    METADATA_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        metadata_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)

print(
    "Requested candidates       :",
    NUM_CANDIDATES
)

print(
    "Successfully generated     :",
    generated_count
)

print(
    "Generation-invalid         :",
    invalid_count
)

print(
    "Exact duplicates detected  :",
    duplicate_count
)

print(
    "Candidate PNG files        :",
    len([
        f
        for f in os.listdir(CANDIDATE_DIR)
        if f.lower().endswith(".png")
    ])
)

print()
print(
    "Metadata CSV:",
    METADATA_CSV
)

print()
print(
    "IMPORTANT:"
)

print(
    "These images have NOT yet been declared valid."
)

print(
    "They must now pass the existing MRI processing "
    "and feature-extraction pipeline."
)

print("=" * 70)