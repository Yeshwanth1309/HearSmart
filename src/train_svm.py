"""
Phase 4 — Stage 5: SVM With Proper Scaling.

Trains SVM using StandardScaler + Pipeline with 10-fold stratified CV
and GridSearchCV. Saves the best model.
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.train import load_training_data

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
X, y_encoded, encoder = load_training_data()

print("\n" + "=" * 60)
print("Phase 4 - Stage 5: SVM With Proper Scaling")
print("=" * 60)

# ------------------------------------------------------------------
# 2. CV setup
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------------------------------------------------
# 3. Define pipeline
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])

# ------------------------------------------------------------------
# 4. Parameter grid (controlled)
# ------------------------------------------------------------------
param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["rbf", "linear"],
    "svm__gamma": ["scale"],
}

total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"Grid: {total_combos} combinations x 10 folds = {total_combos * 10} fits")

# ------------------------------------------------------------------
# 5. GridSearch
# ------------------------------------------------------------------
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=skf,
    scoring="f1_macro",
    n_jobs=2,
    verbose=2,
)

# ------------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------------
print("\nStarting grid search...")
grid.fit(X, y_encoded)

# ------------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("===== SVM RESULTS =====")
print("=" * 60)
print("Best Macro F1:", round(grid.best_score_, 4))
print("Best Params:  ", grid.best_params_)
print("=" * 60)

# ------------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(grid.best_estimator_, "models/svm.pkl")
print("Model saved to models/svm.pkl")
