"""
Phase 4 — Stage 6: XGBoost (Efficient & Thermal-Safe).

Trains XGBoost with 10-fold stratified CV, GridSearchCV,
macro F1 scoring, and limited CPU usage. Saves the best model.
"""

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.train import load_training_data

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
X, y_encoded, encoder = load_training_data()

print("\n" + "=" * 60)
print("Phase 4 - Stage 6: XGBoost (Efficient & Thermal-Safe)")
print("=" * 60)

# ------------------------------------------------------------------
# 2. Stratified CV
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------------------------------------------------
# 3. Define model (thermal-safe version)
# ------------------------------------------------------------------
xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=10,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=2,
    tree_method="hist",
)

# ------------------------------------------------------------------
# 4. Controlled parameter grid
# ------------------------------------------------------------------
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
}

total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"Grid: {total_combos} combinations x 10 folds = {total_combos * 10} fits")

# ------------------------------------------------------------------
# 5. GridSearch setup
# ------------------------------------------------------------------
grid = GridSearchCV(
    estimator=xgb,
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
print("===== XGBOOST RESULTS =====")
print("=" * 60)
print("Best Macro F1:", round(grid.best_score_, 4))
print("Best Params:  ", grid.best_params_)
print("=" * 60)

# ------------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
grid.best_estimator_.save_model("models/xgboost.json")
print("Model saved to models/xgboost.json")
