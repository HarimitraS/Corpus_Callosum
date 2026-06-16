import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class CCDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        img = cv2.imread(
            os.path.join(self.image_dir, file),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            os.path.join(self.mask_dir, file),
            cv2.IMREAD_GRAYSCALE
        )

        # FORCE SAME SIZE

        img = cv2.resize(
            img,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        img = img.astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )