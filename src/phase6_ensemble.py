"""
Phase 6 — Ensemble optimization for 5-model, 8-class system.
Optimizes weights with Nelder-Mead + safety class boosting rules.
Output: results/ensemble_weights_v2_8class.json
"""
import os, sys, json, logging
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tF
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import joblib

from src.data.label_map_v2 import CLASS_NAMES_V2, SAFETY_CLASSES_V2, SAFETY_THRESHOLD_V2
from src.models_v2 import AudioCNN_V2
from src.utils import setup_logging, set_seed
from xgboost import XGBClassifier

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(g, True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load all models
print("  Loading 5 models ...")
rf   = joblib.load("models/rf_v2.pkl")
svm_ = joblib.load("models/svm_v2.pkl")
xgb_ = XGBClassifier(); xgb_.load_model("models/xgb_v2.json")
cnn_ = AudioCNN_V2(8).to(DEVICE)
cnn_.load_state_dict(torch.load("models/cnn_v2.pt", map_location=DEVICE)); cnn_.eval()
yam_ = tf.keras.models.load_model("models/yamnet_v2.h5")
mu   = np.load("features/yamnet_v2/mu.npy"); sig = np.load("features/yamnet_v2/sig.npy")

ORDER = ["rf", "svm", "xgb", "cnn", "yamnet"]

def get_probs(split):
    df = pd.read_csv(f"data/splits_v2/{split}.csv")
    from src.data.label_map_v2 import label_to_index
    y = np.array([label_to_index(c) for c in df["class_v2"]])
    mfccs, cnns = [], []
    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        mfccs.append(np.load(f"features/mfcc_v2/{split}/X.npy")[0] if False else
                     np.load(f"features/mfcc/{split}/{stem}.npy"))
        mel = np.load(f"features/mel_v2/{split}/{stem}.npy")
        mel_t = torch.from_numpy(mel.transpose(2,0,1)[None].astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            cnns.append(tF.softmax(cnn_(mel_t), dim=1).cpu().numpy()[0])
    M = np.stack(mfccs).astype(np.float32)
    P = {}
    P["rf"]  = rf.predict_proba(M).astype(np.float32)
    try: P["svm"] = svm_.predict_proba(M).astype(np.float32)
    except:
        sc = svm_.named_steps["scaler"].transform(M)
        d  = svm_.named_steps["svm"].decision_function(sc).astype(np.float32)
        d -= d.max(1, keepdims=True); P["svm"] = np.exp(d)/np.exp(d).sum(1, keepdims=True)
    P["xgb"]    = xgb_.predict_proba(M).astype(np.float32)
    P["cnn"]    = np.stack(cnns).astype(np.float32)
    ye = np.load(f"features/yamnet_v2/{split}_X.npy")
    yen = ((ye - mu) / sig).astype(np.float32)
    # Pad to 8 classes if needed
    yp = yam_.predict(yen, verbose=0).astype(np.float32)
    P["yamnet"] = yp
    return P, y

print("  Loading val probs  ..."); val_P, y_va = get_probs("val")
print("  Loading test probs ..."); test_P, y_te = get_probs("test")

SAFETY_IDX = {CLASS_NAMES_V2.index(c) for c in SAFETY_CLASSES_V2}

def neg_acc(w, P=val_P, y=y_va):
    w = np.abs(w); w /= w.sum()
    ens = np.einsum("mnc,m->nc", np.stack([P[m] for m in ORDER]), w)
    return -accuracy_score(y, ens.argmax(1))

best_val, best_w = 0, np.ones(5)/5
np.random.seed(42)
print("  Optimizing weights (40 restarts) ...")
for _ in range(40):
    w0 = np.random.dirichlet(np.ones(5))
    r = minimize(neg_acc, w0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-9})
    w = np.abs(r.x); w /= w.sum(); v = -r.fun
    if v > best_val: best_val, best_w = v, w

# Evaluate test set
ens_te = np.einsum("mnc,m->nc", np.stack([test_P[m] for m in ORDER]), best_w)
y_pred = ens_te.argmax(1)
acc = accuracy_score(y_te, y_pred)
f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)

print(f"\n  5-Model Ensemble v2 (8-class):")
print(f"  Val  Accuracy: {best_val:.4f}")
print(f"  Test Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
for m, w in zip(ORDER, best_w): print(f"    {m:10s}: {w:.4f}")
print(classification_report(y_te, y_pred, target_names=CLASS_NAMES_V2, zero_division=0))

# Per-class safety check
from sklearn.metrics import recall_score
recalls = recall_score(y_te, y_pred, average=None, zero_division=0)
print("  Safety Class Recall:")
for c in SAFETY_CLASSES_V2:
    idx = CLASS_NAMES_V2.index(c)
    flag = "✅" if recalls[idx] >= 0.95 else "⚠️"
    print(f"    {flag} {c:12s}: {recalls[idx]:.4f}")

result = {
    "weights": dict(zip(ORDER, best_w.tolist())),
    "model_order": ORDER,
    "val_accuracy": float(best_val),
    "test_accuracy": float(acc),
    "test_macro_f1": float(f1),
    "safety_recall": {c: float(recalls[CLASS_NAMES_V2.index(c)]) for c in SAFETY_CLASSES_V2},
}
with open("results/ensemble_weights_v2_8class.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n  Saved -> results/ensemble_weights_v2_8class.json")
print("  Phase 6 complete.")
