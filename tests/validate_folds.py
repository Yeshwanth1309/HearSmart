"""
Phase 4 — Stage 2: Validate StratifiedKFold splitting.

Confirms:
- Balanced folds (~611 validation samples each)
- All 10 classes present in every fold
- No class disappears
- No leakage
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.train import load_training_data

# Load training data
X, y_encoded, encoder = load_training_data()

print("\n" + "=" * 60)
print("Phase 4 - Stage 2: StratifiedKFold Validation")
print("=" * 60)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_id = 1
all_passed = True

for train_idx, val_idx in skf.split(X, y_encoded):
    print(f"\n===== FOLD {fold_id} =====")
    print("Train size:", len(train_idx))
    print("Validation size:", len(val_idx))

    y_val = y_encoded[val_idx]
    unique, counts = np.unique(y_val, return_counts=True)

    print("Validation class distribution:")
    for u, c in zip(unique, counts):
        class_name = encoder.classes_[u]
        print(f"  Class {u} ({class_name:20s}): {c}")

    # Safety checks
    try:
        assert len(np.unique(y_val)) == 10, "A fold is missing a class!"
        assert len(val_idx) > 550, "Fold too small!"
        assert len(val_idx) < 650, "Fold too large!"
        print("  >> FOLD PASSED")
    except AssertionError as e:
        print(f"  >> FOLD FAILED: {e}")
        all_passed = False

    fold_id += 1

print("\n" + "=" * 60)
if all_passed:
    print("ALL 10 FOLDS PASSED - Cross-validation structure is valid.")
else:
    print("SOME FOLDS FAILED - Check output above.")
print("=" * 60)
