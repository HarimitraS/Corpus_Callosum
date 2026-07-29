from pathlib import Path

root = Path(".")

print("CSV files found:\n")

for f in sorted(root.rglob("*.csv")):
    print(f)