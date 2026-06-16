import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import CCDataset
from model import get_model

DEVICE = "cuda"

IMAGE_DIR = r"../dataset/images"
MASK_DIR = r"../dataset/masks"

dataset = CCDataset(
    IMAGE_DIR,
    MASK_DIR
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=4
)

model = get_model().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.BCEWithLogitsLoss()

best_loss = 999

for epoch in range(50):

    model.train()

    train_loss = 0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        loss = loss_fn(
            pred,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    print(
        f"Epoch {epoch+1} Loss {train_loss:.4f}"
    )

    if train_loss < best_loss:

        best_loss = train_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

print("Training Finished")