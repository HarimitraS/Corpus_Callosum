import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from PIL import Image

from dataset_loader import get_dataloader


# ==========================================================
# SETTINGS
# ==========================================================

DEVICE = torch.device("cpu")

LATENT_DIM = 100

IMAGE_SIZE = 128

CHANNELS = 1

EPOCHS = 200

BATCH_SIZE = 2

LR = 0.0001

BETA1 = 0.0

BETA2 = 0.9

CRITIC_ITERATIONS = 5

LAMBDA_GP = 10


# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT = r"E:\Corpus_Callosum\24_gan_generation"

MODEL_DIR = os.path.join(
    ROOT,
    "models_wgan_gp"
)

GRID_DIR = os.path.join(
    ROOT,
    "generated_images_wgan_gp",
    "grids"
)

INDIVIDUAL_DIR = os.path.join(
    ROOT,
    "generated_images_wgan_gp",
    "individual"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    GRID_DIR,
    exist_ok=True
)

os.makedirs(
    INDIVIDUAL_DIR,
    exist_ok=True
)


# ==========================================================
# DATASET
# ==========================================================

loader = get_dataloader(
    batch_size=BATCH_SIZE
)


# ==========================================================
# GENERATOR
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
# CRITIC
# ==========================================================

class Critic(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(
                CHANNELS,
                32,
                4,
                2,
                1,
                bias=False
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 64 -> 32
            nn.Conv2d(
                32,
                64,
                4,
                2,
                1,
                bias=False
            ),
            nn.InstanceNorm2d(
                64,
                affine=True
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 32 -> 16
            nn.Conv2d(
                64,
                128,
                4,
                2,
                1,
                bias=False
            ),
            nn.InstanceNorm2d(
                128,
                affine=True
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 16 -> 8
            nn.Conv2d(
                128,
                256,
                4,
                2,
                1,
                bias=False
            ),
            nn.InstanceNorm2d(
                256,
                affine=True
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 8 -> 4
            nn.Conv2d(
                256,
                512,
                4,
                2,
                1,
                bias=False
            ),
            nn.InstanceNorm2d(
                512,
                affine=True
            ),
            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            # 4 -> 1
            nn.Conv2d(
                512,
                1,
                4,
                1,
                0,
                bias=False
            )
        )


    def forward(self, x):

        return self.net(x).view(-1)


# ==========================================================
# GRADIENT PENALTY
# ==========================================================

def gradient_penalty(
    critic,
    real,
    fake
):

    batch_size = real.size(0)

    alpha = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=DEVICE
    )

    alpha = alpha.expand_as(real)

    interpolated = (
        alpha * real
        +
        (1 - alpha) * fake
    )

    interpolated.requires_grad_(True)

    mixed_scores = critic(
        interpolated
    )

    gradients = torch.autograd.grad(

        outputs=mixed_scores,

        inputs=interpolated,

        grad_outputs=torch.ones_like(
            mixed_scores
        ),

        create_graph=True,

        retain_graph=True

    )[0]

    gradients = gradients.view(
        batch_size,
        -1
    )

    gradient_norm = gradients.norm(
        2,
        dim=1
    )

    penalty = torch.mean(
        (gradient_norm - 1) ** 2
    )

    return penalty


# ==========================================================
# INITIALIZE MODELS
# ==========================================================

netG = Generator().to(DEVICE)

netC = Critic().to(DEVICE)


# ==========================================================
# OPTIMIZERS
# ==========================================================

optimizerG = optim.Adam(
    netG.parameters(),
    lr=LR,
    betas=(BETA1, BETA2)
)

optimizerC = optim.Adam(
    netC.parameters(),
    lr=LR,
    betas=(BETA1, BETA2)
)


# ==========================================================
# FIXED NOISE
# ==========================================================

fixed_noise = torch.randn(
    16,
    LATENT_DIM,
    1,
    1,
    device=DEVICE
)


# ==========================================================
# START TRAINING
# ==========================================================

print()
print("=" * 60)
print("STARTING 128x128 WGAN-GP TRAINING")
print("=" * 60)

print("Device          :", DEVICE)
print("Images          :", len(loader.dataset))
print("Image Size      :", IMAGE_SIZE, "x", IMAGE_SIZE)
print("Epochs          :", EPOCHS)
print("Batch Size      :", BATCH_SIZE)
print("Critic Steps    :", CRITIC_ITERATIONS)
print("Gradient Lambda :", LAMBDA_GP)

print("=" * 60)


# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(EPOCHS):

    for batch_idx, real_images in enumerate(loader):

        real_images = real_images.to(DEVICE)

        batch_size = real_images.size(0)


        # ==================================================
        # TRAIN CRITIC
        # ==================================================

        for _ in range(CRITIC_ITERATIONS):

            noise = torch.randn(
                batch_size,
                LATENT_DIM,
                1,
                1,
                device=DEVICE
            )

            fake_images = netG(noise)


            critic_real = netC(
                real_images
            )

            critic_fake = netC(
                fake_images.detach()
            )


            gp = gradient_penalty(
                netC,
                real_images,
                fake_images.detach()
            )


            lossC = (
                -torch.mean(critic_real)
                +
                torch.mean(critic_fake)
                +
                LAMBDA_GP * gp
            )


            optimizerC.zero_grad()

            lossC.backward()

            optimizerC.step()


        # ==================================================
        # TRAIN GENERATOR
        # ==================================================

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            1,
            1,
            device=DEVICE
        )

        fake_images = netG(noise)

        critic_fake = netC(
            fake_images
        )


        lossG = -torch.mean(
            critic_fake
        )


        optimizerG.zero_grad()

        lossG.backward()

        optimizerG.step()


    # ======================================================
    # PRINT EPOCH
    # ======================================================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss_C={lossC.item():.4f} "
        f"Loss_G={lossG.item():.4f}"
    )


    # ======================================================
    # SAVE EVERY 25 EPOCHS
    # ======================================================

    if (epoch + 1) % 25 == 0:

        with torch.no_grad():

            fake = netG(
                fixed_noise
            ).cpu()


        # ==================================================
        # SAVE COMBINED GRID
        # ==================================================

        grid_path = os.path.join(
            GRID_DIR,
            f"epoch_{epoch + 1:03d}.png"
        )

        vutils.save_image(
            fake,
            grid_path,
            normalize=True,
            nrow=4
        )

        print(
            "Saved Grid:",
            grid_path
        )


        # ==================================================
        # SAVE INDIVIDUAL IMAGES
        # ==================================================

        epoch_dir = os.path.join(
            INDIVIDUAL_DIR,
            f"epoch_{epoch + 1:03d}"
        )

        os.makedirs(
            epoch_dir,
            exist_ok=True
        )


        for i in range(
            fake.size(0)
        ):

            image = fake[i]

            # [-1, 1] -> [0, 255]

            image = (
                (image + 1.0) / 2.0
            ) * 255.0

            image = image.clamp(
                0,
                255
            )

            image = image.squeeze(
                0
            ).numpy()


            image = Image.fromarray(
                image.astype("uint8")
            )


            individual_path = os.path.join(
                epoch_dir,
                f"synthetic_mri_{i + 1:03d}.png"
            )


            image.save(
                individual_path
            )


        print(
            "Saved 16 individual images:",
            epoch_dir
        )


# ==========================================================
# SAVE GENERATOR
# ==========================================================

generator_path = os.path.join(
    MODEL_DIR,
    "generator_wgan_gp_128.pth"
)

torch.save(
    netG.state_dict(),
    generator_path
)


# ==========================================================
# SAVE CRITIC
# ==========================================================

critic_path = os.path.join(
    MODEL_DIR,
    "critic_wgan_gp_128.pth"
)

torch.save(
    netC.state_dict(),
    critic_path
)


# ==========================================================
# COMPLETE
# ==========================================================

print()
print("=" * 60)
print("WGAN-GP TRAINING COMPLETE")
print("=" * 60)

print(
    "Generator:",
    generator_path
)

print(
    "Critic:",
    critic_path
)

print(
    "Grid Images:",
    GRID_DIR
)

print(
    "Individual Images:",
    INDIVIDUAL_DIR
)

print("=" * 60)