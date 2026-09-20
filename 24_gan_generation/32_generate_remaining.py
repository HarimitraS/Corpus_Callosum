import os
import torch
import torch.nn as nn
from PIL import Image


# ==========================================================
# SETTINGS
# ==========================================================

DEVICE = torch.device("cpu")

LATENT_DIM = 100
IMAGE_SIZE = 128
CHANNELS = 1

NUM_IMAGES = 22


# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT = r"E:\Corpus_Callosum\24_gan_generation"

MODEL_PATH = os.path.join(
    ROOT,
    "models_wgan_gp",
    "generator_wgan_gp_128.pth"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "generated_images_wgan_gp",
    "additional_22"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# GENERATOR
# SAME ARCHITECTURE USED FOR TRAINING
# ==========================================================

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


# ==========================================================
# CHECK MODEL
# ==========================================================

print()
print("=" * 60)
print("GENERATING REMAINING WGAN-GP IMAGES")
print("=" * 60)

print("Device       :", DEVICE)
print("Model        :", MODEL_PATH)
print("Output       :", OUTPUT_DIR)
print("Images       :", NUM_IMAGES)


# ==========================================================
# LOAD GENERATOR
# ==========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Generator model not found:\n{MODEL_PATH}"
    )


netG = Generator().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

netG.load_state_dict(state_dict)

netG.eval()

print()
print("Generator loaded successfully.")


# ==========================================================
# GENERATE IMAGES
# ==========================================================

with torch.no_grad():

    for i in range(NUM_IMAGES):

        # Generate random latent vector
        noise = torch.randn(
            1,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        # Generate image
        fake = netG(noise)

        # [-1, 1] -> [0, 255]
        image = (
            (fake + 1.0) / 2.0
        ) * 255.0

        image = image.clamp(
            0,
            255
        )

        # Remove channel dimension
        image = image.squeeze(
            0
        ).squeeze(
            0
        ).cpu().numpy()

        # Convert to uint8 image
        image = Image.fromarray(
            image.astype("uint8"),
            mode="L"
        )

        # Save
        output_path = os.path.join(
            OUTPUT_DIR,
            f"synthetic_mri_additional_{i + 1:03d}.png"
        )

        image.save(
            output_path
        )

        print(
            f"[{i + 1:02d}/{NUM_IMAGES}] Saved:",
            output_path
        )


# ==========================================================
# VERIFY OUTPUT
# ==========================================================

generated_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith(".png")
]

print()
print("=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)

print(
    "New images generated:",
    len(generated_files)
)

print(
    "Output directory:",
    OUTPUT_DIR
)

print("=" * 60)


# ==========================================================
# FINAL CHECK
# ==========================================================

if len(generated_files) == NUM_IMAGES:

    print()
    print("SUCCESS: All 22 additional images were generated.")
    print("Existing individual images : 128")
    print("New images                  :", len(generated_files))
    print("Total synthetic images      :", 128 + len(generated_files))
    print()

else:

    print()
    print("WARNING: Expected 22 images,")
    print("but found", len(generated_files))
    print()