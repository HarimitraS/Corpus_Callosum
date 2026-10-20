import os
import hashlib
from datetime import datetime, timezone
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pandas as pd

# Paths relative to project root
EXPERIMENT_DIR = os.path.join(".", "33_synthetic_delayed_150")
CANDIDATE_DIR = os.path.join(EXPERIMENT_DIR, "candidates")
METADATA_CSV = os.path.join(EXPERIMENT_DIR, "metadata", "generation_metadata.csv")
MODEL_PATH = os.path.join(".", "24_gan_generation", "models_wgan_gp", "generator_wgan_gp_128.pth")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
IMAGE_SIZE = 128
CHANNELS = 1
NUM_CANDIDATES = 200
BASE_SEED = 202610050001

os.makedirs(CANDIDATE_DIR, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

print("=" * 70)
print("STAGE 1: GENERATING 200 DETERMINISTIC CANDIDATES")
print("=" * 70)

metadata_df = pd.read_csv(METADATA_CSV)
expected_shas = dict(zip(metadata_df["filename"], metadata_df["sha256"]))

netG = Generator().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
netG.load_state_dict(state_dict)
netG.eval()

mismatches = 0
matches = 0

with torch.no_grad():
    for i in range(NUM_CANDIDATES):
        candidate_id = i + 1
        latent_seed = BASE_SEED + i
        filename = f"candidate_{candidate_id:06d}.png"
        output_path = os.path.join(CANDIDATE_DIR, filename)

        rng = torch.Generator(device="cpu")
        rng.manual_seed(latent_seed)

        noise = torch.randn(
            1, LATENT_DIM, 1, 1,
            generator=rng,
            device=DEVICE
        )

        fake = netG(noise)
        image_tensor = ((fake + 1.0) / 2.0) * 255.0
        image_tensor = image_tensor.clamp(0, 255)
        image_array = image_tensor.squeeze().cpu().numpy().astype(np.uint8)

        image = Image.fromarray(image_array, mode="L")
        image.save(output_path, format="PNG")

        actual_sha = sha256_file(output_path)
        expected_sha = expected_shas.get(filename)

        if actual_sha == expected_sha:
            matches += 1
        else:
            mismatches += 1
            print(f"MISMATCH for {filename}: got {actual_sha}, expected {expected_sha}")

        if (candidate_id % 25 == 0) or candidate_id == NUM_CANDIDATES:
            print(f"Generated {candidate_id}/{NUM_CANDIDATES} | Matches: {matches} | Mismatches: {mismatches}")

print("=" * 70)
print(f"GENERATION COMPLETE. Total: {NUM_CANDIDATES}, Matches: {matches}, Mismatches: {mismatches}")
print("=" * 70)
if mismatches > 0:
    raise RuntimeError(f"Hash verification failed for {mismatches} candidates!")
