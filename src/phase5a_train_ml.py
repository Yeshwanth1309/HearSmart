"""
Phase 5a — Train RF, SVM, XGBoost on 8-class MFCC features.
"""
import os, sys, json, logging
sys.path.insert(0, ".")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import joblib

from src.data.label_map_v2 import CLASS_NAMES_V2
from src.utils import setup_logging, set_seed

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)

X_tr = np.load("features/mfcc_v2/train/X.npy")
y_tr = np.load("features/mfcc_v2/train/y.npy")
X_va = np.load("features/mfcc_v2/val/X.npy")
y_va = np.load("features/mfcc_v2/val/y.npy")
X_te = np.load("features/mfcc_v2/test/X.npy")
y_te = np.load("features/mfcc_v2/test/y.npy")

print(f"  Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

results = {}

# ── 1. Random Forest ────────────────────────────────────────────────────────
print("\n  [RF] Training ...")
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                            class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_te)
rf_acc  = accuracy_score(y_te, rf_pred)
rf_f1   = f1_score(y_te, rf_pred, average="macro", zero_division=0)
print(f"  [RF] Test acc={rf_acc:.4f}  macro-F1={rf_f1:.4f}")
print(classification_report(y_te, rf_pred, target_names=CLASS_NAMES_V2, zero_division=0))
joblib.dump(rf, "models/rf_v2.pkl")
results["rf"] = {"accuracy": float(rf_acc), "macro_f1": float(rf_f1)}

# ── 2. SVM ──────────────────────────────────────────────────────────────────
print("\n  [SVM] Training ...")
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                   class_weight="balanced", random_state=42)),
])
svm.fit(X_tr, y_tr)
svm_pred = svm.predict(X_te)
svm_acc  = accuracy_score(y_te, svm_pred)
svm_f1   = f1_score(y_te, svm_pred, average="macro", zero_division=0)
print(f"  [SVM] Test acc={svm_acc:.4f}  macro-F1={svm_f1:.4f}")
print(classification_report(y_te, svm_pred, target_names=CLASS_NAMES_V2, zero_division=0))
joblib.dump(svm, "models/svm_v2.pkl")
results["svm"] = {"accuracy": float(svm_acc), "macro_f1": float(svm_f1)}

# ── 3. XGBoost ──────────────────────────────────────────────────────────────
print("\n  [XGB] Training ...")
from collections import Counter
cnt = Counter(y_tr.tolist())
scale_pos = {i: len(y_tr) / (8 * cnt.get(i, 1)) for i in range(8)}
xgb = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="mlogloss",
                    random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
xgb_pred = xgb.predict(X_te)
xgb_acc  = accuracy_score(y_te, xgb_pred)
xgb_f1   = f1_score(y_te, xgb_pred, average="macro", zero_division=0)
print(f"  [XGB] Test acc={xgb_acc:.4f}  macro-F1={xgb_f1:.4f}")
print(classification_report(y_te, xgb_pred, target_names=CLASS_NAMES_V2, zero_division=0))
xgb.save_model("models/xgb_v2.json")
results["xgb"] = {"accuracy": float(xgb_acc), "macro_f1": float(xgb_f1)}

import json
with open("results/metrics_v2_ml.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n  Saved -> results/metrics_v2_ml.json")
print("  Phase 5a complete.")
