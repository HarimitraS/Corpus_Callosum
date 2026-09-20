import torch
import sys
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("31_gan_delayed_pipeline")

sys.path.insert(0, str(ROOT.parent / "unet_cc"))

from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()

model.load_state_dict(
    torch.load(
        ROOT.parent / "unet_cc" / "best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

roi_dir = ROOT / "03_cc_extraction" / "roi"

ids = [
    "synthetic_delayed_001.png",
    "synthetic_delayed_002.png",
    "synthetic_delayed_003.png",
    "synthetic_delayed_004.png",
    "synthetic_delayed_017.png"
]

print("=" * 60)
print("U-NET RAW PROBABILITY DIAGNOSTIC")
print("=" * 60)
print("Device:", device)
print()

for filename in ids:

    path = roi_dir / filename

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(filename, "-> ROI NOT FOUND")
        continue

    resized = cv2.resize(image, (256, 256))

    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output)

    p = probability.squeeze().cpu().numpy()

    print(filename)
    print("  Input min/max :", round(float(resized.min()), 4), round(float(resized.max()), 4))
    print("  Prob min      :", round(float(p.min()), 6))
    print("  Prob max      :", round(float(p.max()), 6))
    print("  Prob mean     :", round(float(p.mean()), 6))
    print("  > 0.10 pixels :", int((p > 0.10).sum()))
    print("  > 0.20 pixels :", int((p > 0.20).sum()))
    print("  > 0.30 pixels :", int((p > 0.30).sum()))
    print("  > 0.40 pixels :", int((p > 0.40).sum()))
    print("  > 0.50 pixels :", int((p > 0.50).sum()))
    print()

print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
