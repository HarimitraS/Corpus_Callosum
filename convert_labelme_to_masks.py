import os
import json
import cv2
import numpy as np

SOURCE_DIR = r"C:\Users\kanishk\OneDrive\Desktop\infant\Corpus_Callosum\8_roi_2"

IMAGES_DIR = r"C:\Users\kanishk\OneDrive\Desktop\infant\Corpus_Callosum\dataset\images"
MASKS_DIR = r"C:\Users\kanishk\OneDrive\Desktop\infant\Corpus_Callosum\dataset\masks"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

json_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".json")]

for json_file in json_files:

    json_path = os.path.join(SOURCE_DIR, json_file)

    with open(json_path, "r") as f:
        data = json.load(f)

    image_name = data["imagePath"]

    img_path = os.path.join(SOURCE_DIR, image_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:

        pts = np.array(shape["points"], dtype=np.int32)

        cv2.fillPoly(mask, [pts], 255)

    cv2.imwrite(
        os.path.join(MASKS_DIR, image_name),
        mask
    )

    cv2.imwrite(
        os.path.join(IMAGES_DIR, image_name),
        img
    )

print("DONE")