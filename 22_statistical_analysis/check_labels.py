import pandas as pd

meta = pd.read_csv("meta.csv", sep=";")
master = pd.read_csv("22_statistical_analysis/master_dataset.csv")

delayed_meta = meta[meta["myelinisation"]=="delayed"]["image_id"]

print("Delayed in metadata:")
print(delayed_meta.tolist())

print()

print("Delayed in master:")
print(master[master["myelinisation"]=="delayed"]["Subject"].tolist())