import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

# ==========================================================
# PATHS
# ==========================================================

ROOT = r"E:\Corpus_Callosum"

IMAGE_DIR = os.path.join(
    ROOT,
    "24_gan_generation",
    "delayed_images"
)

# ==========================================================
# IMAGE TRANSFORMS
# ==========================================================

transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((512, 512)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )

])

# ==========================================================
# DATASET
# ==========================================================

class MRIDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir

        self.transform = transform

        self.images = sorted([

            f

            for f in os.listdir(image_dir)

            if f.lower().endswith(".png")

        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        filename = self.images[index]

        path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(path).convert("L")

        if self.transform:

            image = self.transform(image)

        return image

# ==========================================================
# DATALOADER
# ==========================================================

def get_dataloader(batch_size=2):

    dataset = MRIDataset(

        IMAGE_DIR,

        transform=transform

    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=True,

        drop_last=True

    )

    return loader

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    loader = get_dataloader()

    print("=" * 60)
    print("MRI Dataset Loader")
    print("=" * 60)

    print(f"Images Found : {len(loader.dataset)}")

    for batch in loader:

        print("Batch Shape :", batch.shape)

        break

    print("=" * 60)