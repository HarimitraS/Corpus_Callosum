from pathlib import Path

subjects = ["s0341", "s0412"]

folders = {
    "Sagittal Slice": "4_sagittal_slices",
    "ROI": "5_roi",
    "Refined Mask": "12_refined_masks",
}

for subject in subjects:
    print("\n" + "="*60)
    print(subject)
    print("="*60)

    for stage, folder in folders.items():

        path = Path(folder)

        found = list(path.glob(f"{subject}*"))

        if found:
            print(f"{stage:20} : FOUND")
            for f in found:
                print("   ", f.name)
        else:
            print(f"{stage:20} : NOT FOUND")