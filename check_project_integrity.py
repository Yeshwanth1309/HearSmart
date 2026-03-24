import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.getcwd()

print("\n===== PROJECT INTEGRITY CHECK =====\n")

# 1. Check required directories
required_dirs = [
    "src",
    "features",
    "features/mfcc",
    "features/mfcc/train",
    "features/mfcc/val",
    "features/mfcc/test",
    "data",
    "models",
    "results"
]

print("Checking required directories...")
for d in required_dirs:
    path = os.path.join(PROJECT_ROOT, d)
    if not os.path.isdir(path):
        print(f"[FAIL] Missing directory: {d}")
        sys.exit(1)
    else:
        print(f"[OK] {d}")

# 2. Check MFCC file counts
print("\nChecking MFCC file counts...")

train_count = len(os.listdir("features/mfcc/train"))
val_count = len(os.listdir("features/mfcc/val"))
test_count = len(os.listdir("features/mfcc/test"))

print(f"Train MFCC files: {train_count}")
print(f"Val MFCC files:   {val_count}")
print(f"Test MFCC files:  {test_count}")

assert train_count == 6112, "Train MFCC count mismatch!"
assert val_count == 1310, "Val MFCC count mismatch!"
assert test_count == 1310, "Test MFCC count mismatch!"

print("[OK] MFCC file counts correct")

# 3. Check sample MFCC file shape
print("\nChecking MFCC file shape...")

sample_file = os.listdir("features/mfcc/train")[0]
sample_path = os.path.join("features/mfcc/train", sample_file)

sample = np.load(sample_path)

print("Sample shape:", sample.shape)
print("Sample dtype:", sample.dtype)

assert sample.shape == (80,), "MFCC shape mismatch!"
print("[OK] MFCC shape correct")

# 4. Check splits exist
print("\nChecking split CSV files...")

required_csvs = [
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv"
]

for csv in required_csvs:
    if not os.path.isfile(csv):
        print(f"[FAIL] Missing CSV: {csv}")
        sys.exit(1)
    else:
        print(f"[OK] {csv}")

print("\nChecking train CSV sample...")
df = pd.read_csv("data/splits/train.csv")
print("Train CSV rows:", len(df))
assert len(df) == 6112, "Train CSV row mismatch!"
print("[OK] Train CSV valid")

# 5. Check models folder exists
print("\nChecking models folder...")
if not os.path.isdir("models"):
    print("[FAIL] models folder missing")
    sys.exit(1)
else:
    print("[OK] models folder present")

print("\n===== ALL CHECKS PASSED =====")
print("Project is ready for Colab upload.\n")
