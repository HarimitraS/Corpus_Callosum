import os
import torch
import torch.nn as nn
from PIL import Image


# ==========================================================
# SETTINGS
# ==========================================================

DEVICE = torch.device("cpu")

LATENT_DIM = 100

NUM_IMAGES = 16

CHANNELS = 1


# ==========================================================
# PATHS
# ==========================================================

ROOT = r"E:\Corpus_Callosum\24_gan_generation"

MODEL_PATH = os.path.join(
    ROOT,
    "models_128",
    "generator_128_v2.pth"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "individual_generated_images"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# GENERATOR
# ==========================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            # 1 x 1 -> 4 x 4
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

            # 4 x 4 -> 8 x 8
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

            # 8 x 8 -> 16 x 16
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

            # 16 x 16 -> 32 x 32
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

            # 32 x 32 -> 64 x 64
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

            # 64 x 64 -> 128 x 128
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

print("=" * 60)
print("GENERATING INDIVIDUAL GAN IMAGES")
print("=" * 60)

print("Model:", MODEL_PATH)
print()


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Generator model not found:\n{MODEL_PATH}"
    )


# ==========================================================
# LOAD TRAINED GENERATOR
# ==========================================================

netG = Generator().to(DEVICE)

netG.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

netG.eval()

print("Generator loaded successfully.")
print()


# ==========================================================
# GENERATE IMAGES
# ==========================================================

with torch.no_grad():

    noise = torch.randn(
        NUM_IMAGES,
        LATENT_DIM,
        1,
        1,
        device=DEVICE
    )

    generated_images = netG(noise).cpu()


# ==========================================================
# SAVE INDIVIDUAL IMAGES
# ==========================================================

print("Saving individual images...")
print()


for i in range(NUM_IMAGES):

    image = generated_images[i]

    # Convert [-1, 1] to [0, 255]
    image = (
        (image + 1.0) / 2.0
    ) * 255.0

    image = image.clamp(
        0,
        255
    )

    image = image.squeeze(0).numpy()

    image = Image.fromarray(
        image.astype("uint8")
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"synthetic_mri_{i + 1:03d}.png"
    )

    image.save(output_path)

    print(
        f"[{i + 1:02d}/{NUM_IMAGES}] Saved: "
        f"{output_path}"
    )


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 60)
print("INDIVIDUAL IMAGE GENERATION COMPLETE")
print("=" * 60)

print("Total Images:", NUM_IMAGES)

print("Output Folder:", OUTPUT_DIR)

print("=" * 60)