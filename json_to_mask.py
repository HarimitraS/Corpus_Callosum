import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_DIR = os.path.join(BASE_DIR, "ground_truth_masks")

OUTPUT_DIR = os.path.join(BASE_DIR, "ground_truth_binary_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# PROCESS
# ==========================================

json_files = sorted(

    [

        f

        for f in os.listdir(JSON_DIR)

        if f.endswith(".json")

    ]

)

print(f"\nFound {len(json_files)} JSON files\n")

for file in tqdm(json_files):

    json_path = os.path.join(JSON_DIR, file)

    with open(json_path, "r") as f:

        data = json.load(f)

    h = data["imageHeight"]

    w = data["imageWidth"]

    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:

        if shape["shape_type"] != "polygon":
            continue

        pts = np.array(

            shape["points"],

            dtype=np.int32

        )

        cv2.fillPoly(

            mask,

            [pts],

            255

        )

    out_name = file.replace(".json", ".png")

    cv2.imwrite(

        os.path.join(

            OUTPUT_DIR,

            out_name

        ),

        mask

    )

print("\n==============================")
print("Finished!")
print(f"Masks saved to:\n{OUTPUT_DIR}")
print("==============================")