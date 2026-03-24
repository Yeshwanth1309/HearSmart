"""
Phase 4 — Stage 4: Random Forest Hyperparameter Tuning.

Uses GridSearchCV with 10-fold stratified CV and macro F1 scoring
to find the best RF configuration. Saves the best model.
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

from src.train import load_training_data

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
X, y_encoded, encoder = load_training_data()

print("\n" + "=" * 60)
print("Phase 4 - Stage 4: Random Forest Hyperparameter Tuning")
print("=" * 60)

# ------------------------------------------------------------------
# 2. Define CV
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------------------------------------------------
# 3. Define parameter grid
# ------------------------------------------------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}

total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"Grid: {total_combos} combinations x 10 folds = {total_combos * 10} fits")

# ------------------------------------------------------------------
# 4. Setup GridSearch
# ------------------------------------------------------------------
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=skf,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2,
)

# ------------------------------------------------------------------
# 5. Fit
# ------------------------------------------------------------------
print("\nStarting grid search...")
grid.fit(X, y_encoded)

# ------------------------------------------------------------------
# 6. Print results
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("===== TUNED RANDOM FOREST =====")
print("=" * 60)
print("Best Macro F1:", round(grid.best_score_, 4))
print("Best Params:  ", grid.best_params_)
print("=" * 60)

# ------------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(grid.best_estimator_, "models/random_forest.pkl")
print("Model saved to models/random_forest.pkl")
