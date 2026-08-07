import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from dataset_loader import get_dataloader

# ==========================================================
# SETTINGS
# ==========================================================

DEVICE = torch.device("cpu")

LATENT_DIM = 100

IMAGE_SIZE = 512

CHANNELS = 1

FEATURES_GEN = 64

FEATURES_DISC = 64

EPOCHS = 500

LR = 0.0002

BETA1 = 0.5

ROOT = r"E:\Corpus_Callosum\24_gan_generation"

MODEL_DIR = os.path.join(ROOT, "models")

GENERATED_DIR = os.path.join(ROOT, "generated_images")

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(GENERATED_DIR, exist_ok=True)

# ==========================================================
# DATALOADER
# ==========================================================

loader = get_dataloader(batch_size=2)

# ==========================================================
# GENERATOR
# ==========================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.ConvTranspose2d(LATENT_DIM,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,32,4,2,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32,16,4,2,1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16,8,4,2,1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8,1,4,2,1,bias=False),
            nn.Tanh()

        )

    def forward(self,x):

        return self.net(x)

# ==========================================================
# DISCRIMINATOR
# ==========================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1,8,4,2,1,bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8,16,4,2,1,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16,32,4,2,1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512,1,4,1,0,bias=False),

            nn.Sigmoid()

        )

    def forward(self,x):

        return self.net(x).view(-1)

# ==========================================================
# INITIALIZE
# ==========================================================

netG = Generator().to(DEVICE)

netD = Discriminator().to(DEVICE)

criterion = nn.BCELoss()

optimizerD = optim.Adam(
    netD.parameters(),
    lr=LR,
    betas=(BETA1,0.999)
)

optimizerG = optim.Adam(
    netG.parameters(),
    lr=LR,
    betas=(BETA1,0.999)
)

fixed_noise = torch.randn(
    16,
    LATENT_DIM,
    1,
    1,
    device=DEVICE
)
# ==========================================================
# TRAINING LOOP
# ==========================================================

print("=" * 60)
print("Starting DCGAN Training")
print("=" * 60)

real_label = 1.
fake_label = 0.

for epoch in range(EPOCHS):

    for i, real_images in enumerate(loader):

        # -----------------------------
        # Train Discriminator
        # -----------------------------

        netD.zero_grad()

        real_images = real_images.to(DEVICE)

        batch_size = real_images.size(0)

        labels = torch.full(
            (batch_size,),
            real_label,
            dtype=torch.float,
            device=DEVICE
        )

        output = netD(real_images)

        loss_real = criterion(output, labels)

        loss_real.backward()

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        fake_images = netG(noise)

        labels.fill_(fake_label)

        output = netD(fake_images.detach())

        loss_fake = criterion(output, labels)

        loss_fake.backward()

        lossD = loss_real + loss_fake

        optimizerD.step()

        # -----------------------------
        # Train Generator
        # -----------------------------

        netG.zero_grad()

        labels.fill_(real_label)

        output = netD(fake_images)

        lossG = criterion(output, labels)

        lossG.backward()

        optimizerG.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss_D={lossD.item():.4f} "
        f"Loss_G={lossG.item():.4f}"
    )
    # ==========================================================
# SAVE GENERATED IMAGES
# ==========================================================

    if (epoch + 1) % 25 == 0:

        with torch.no_grad():

            fake = netG(fixed_noise).detach().cpu()

        image_path = os.path.join(

            GENERATED_DIR,

            f"epoch_{epoch+1:03d}.png"

        )

        vutils.save_image(

            fake,

            image_path,

            normalize=True,

            nrow=4

        )

        print(f"Saved Sample Image : {image_path}")

# ==========================================================
# SAVE MODELS
# ==========================================================

torch.save(

    netG.state_dict(),

    os.path.join(

        MODEL_DIR,

        "generator.pth"

    )

)

torch.save(

    netD.state_dict(),

    os.path.join(

        MODEL_DIR,

        "discriminator.pth"

    )

)

print("\n" + "=" * 60)
print("DCGAN Training Complete")
print("=" * 60)
print("Generator Saved     :", os.path.join(MODEL_DIR, "generator.pth"))
print("Discriminator Saved :", os.path.join(MODEL_DIR, "discriminator.pth"))
print("Generated Images    :", GENERATED_DIR)
print("=" * 60)