import pandas as pd

with open("meta.csv", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 80:
            print(f"{i+1}: {line.rstrip()}")

print("\nTrying pandas...")

try:
    df = pd.read_csv("meta.csv")
    print(df.head())
    print(df.columns.tolist())
except Exception as e:
    print(e)