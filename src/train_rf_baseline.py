"""
Phase 4 — Stage 3: Random Forest Baseline (No Tuning).

Trains a simple Random Forest with 10-fold stratified CV to validate
that MFCC features work and the entire pipeline is correct.

No tuning, no saving, no scaling — just a controlled baseline.
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from src.train import load_training_data

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
X, y_encoded, encoder = load_training_data()

print("\n" + "=" * 60)
print("Phase 4 - Stage 3: Random Forest Baseline (No Tuning)")
print("=" * 60)

# ------------------------------------------------------------------
# 2. Initialize CV
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------------------------------------------------
# 3. Train fold-by-fold
# ------------------------------------------------------------------
fold_id = 1
macro_f1_scores = []
accuracy_scores = []

for train_idx, val_idx in skf.split(X, y_encoded):

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro")
    acc = accuracy_score(y_val, y_pred)

    macro_f1_scores.append(macro_f1)
    accuracy_scores.append(acc)

    print(f"\n===== FOLD {fold_id} =====")
    print("Macro F1:", round(macro_f1, 4))
    print("Accuracy:", round(acc, 4))

    fold_id += 1

# ------------------------------------------------------------------
# 4. Final summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("===== BASELINE RESULTS =====")
print("=" * 60)
print("Mean Macro F1:", round(np.mean(macro_f1_scores), 4))
print("Std Macro F1: ", round(np.std(macro_f1_scores), 4))
print("Mean Accuracy:", round(np.mean(accuracy_scores), 4))
print("=" * 60)
