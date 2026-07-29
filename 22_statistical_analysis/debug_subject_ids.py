import pandas as pd

morph = pd.read_csv("16_measurements/features.csv")
texture = pd.read_csv("20_texture_analysis/texture_features.csv")
radiomic = pd.read_csv("21_radiomic_features/radiomic_features.csv")
meta = pd.read_csv("meta.csv", sep=";", engine="python")

print("\n=== Morphology ===")
print(morph.columns.tolist())
print(morph.iloc[:5, 0].tolist())

print("\n=== Texture ===")
print(texture.columns.tolist())
print(texture.iloc[:5, 0].tolist())

print("\n=== Radiomics ===")
print(radiomic.columns.tolist())
print(radiomic.iloc[:5, 0].tolist())

print("\n=== Metadata ===")
print(meta.columns.tolist())
print(meta["image_id"].head().tolist())