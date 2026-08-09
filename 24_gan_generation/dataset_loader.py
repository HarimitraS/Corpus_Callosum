import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


# ==========================================================
# PATH
# ==========================================================

IMAGE_DIR = r"E:\Corpus_Callosum\4_sagittal_slices"


# ==========================================================
# IMAGE TRANSFORMS
# ==========================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (128, 128)
    ),

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

    def __init__(
        self,
        image_dir,
        transform=None
    ):

        self.image_dir = image_dir

        self.transform = transform

        # Automatically find ALL PNG images
        self.images = sorted([

            filename

            for filename in os.listdir(image_dir)

            if filename.lower().endswith(".png")

        ])

        print("\n" + "=" * 60)
        print("GAN MRI DATASET")
        print("=" * 60)

        print(
            "Dataset Path:",
            self.image_dir
        )

        print(
            "Images Found:",
            len(self.images)
        )

        print("=" * 60)


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        filename = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(
            image_path
        ).convert("L")


        if self.transform:

            image = self.transform(
                image
            )


        return image


# ==========================================================
# DATALOADER
# ==========================================================

def get_dataloader(
    batch_size=2
):

    dataset = MRIDataset(
        IMAGE_DIR,
        transform=transform
    )


    if len(dataset) == 0:

        raise RuntimeError(
            "No PNG images found in:\n"
            + IMAGE_DIR
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

    loader = get_dataloader(
        batch_size=2
    )


    print("\n" + "=" * 60)
    print("MRI DATASET LOADER TEST")
    print("=" * 60)


    print(
        "Images Found:",
        len(loader.dataset)
    )


    for batch in loader:

        print(
            "Batch Shape:",
            batch.shape
        )

        break


    print("=" * 60)