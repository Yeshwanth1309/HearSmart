"""
Phase 7 — Full Evaluation & Reporting Pipeline (v2 8-class).
Generates metrics, confusion matrices, and figures for IEEE publication.
"""
import os, sys, json, logging
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tF
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

from src.data.label_map_v2 import CLASS_NAMES_V2, SAFETY_CLASSES_V2
from src.models_v2 import AudioCNN_V2
from src.utils import setup_logging, set_seed
from xgboost import XGBClassifier

setup_logging()
set_seed(42)
logger = logging.getLogger(__name__)

import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(g, True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Path("results/figures").mkdir(parents=True, exist_ok=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
print("  Loading test data ...")
split = "test"
df = pd.read_csv(f"data/splits_v2/{split}.csv")
from src.data.label_map_v2 import label_to_index
y_true = np.array([label_to_index(c) for c in df["class_v2"]])

# MFCCs
mfccs = []
for _, row in df.iterrows():
    stem = Path(row["slice_file_name"]).stem
    mfcc_path = Path(f"features/mfcc/{split}/{stem}.npy")
    mfccs.append(np.load(mfcc_path))
X_mfcc = np.stack(mfccs).astype(np.float32)

# Mels
mels = []
for _, row in df.iterrows():
    stem = Path(row["slice_file_name"]).stem
    mel = np.load(f"features/mel_v2/{split}/{stem}.npy")
    mels.append(mel.transpose(2,0,1)[None].astype(np.float32))
X_mel = np.concatenate(mels, axis=0)

# YAMNet Embeddings
X_yam_raw = np.load(f"features/yamnet_v2/{split}_X.npy")
yam_mu = np.load("features/yamnet_v2/mu.npy")
yam_sig = np.load("features/yamnet_v2/sig.npy")
X_yam = ((X_yam_raw - yam_mu) / yam_sig).astype(np.float32)

# ─── Load Models ─────────────────────────────────────────────────────────────
print("  Loading 5 trained models ...")
rf = joblib.load("models/rf_v2.pkl")
svm = joblib.load("models/svm_v2.pkl")
xgb = XGBClassifier()
xgb.load_model("models/xgb_v2.json")

cnn = AudioCNN_V2(8).to(DEVICE)
cnn.load_state_dict(torch.load("models/cnn_v2.pt", map_location=DEVICE))
cnn.eval()

yamnet = tf.keras.models.load_model("models/yamnet_v2.h5")

with open("results/ensemble_weights_v2_8class.json") as f:
    ens_config = json.load(f)
weights = ens_config["weights"]
ORDER = ens_config["model_order"]
print(f"  Loaded ensemble weights: {weights}")

# ─── Generate Predictions ────────────────────────────────────────────────────
print("  Computing forward passes ...")
P = {}

P["rf"] = rf.predict_proba(X_mfcc).astype(np.float32)
try:
    P["svm"] = svm.predict_proba(X_mfcc).astype(np.float32)
except:
    sc = svm.named_steps["scaler"].transform(X_mfcc)
    d = svm.named_steps["svm"].decision_function(sc).astype(np.float32)
    d -= d.max(1, keepdims=True)
    P["svm"] = np.exp(d) / np.exp(d).sum(1, keepdims=True)

P["xgb"] = xgb.predict_proba(X_mfcc).astype(np.float32)

cnn_preds = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(X_mel), batch_size):
        batch = torch.from_numpy(X_mel[i:i+batch_size]).to(DEVICE)
        cnn_preds.append(tF.softmax(cnn(batch), dim=1).cpu().numpy())
P["cnn"] = np.concatenate(cnn_preds, axis=0).astype(np.float32)

P["yamnet"] = yamnet.predict(X_yam, verbose=0).astype(np.float32)

# Ensemble Preds
w_array = np.array([weights[m] for m in ORDER])
w_array /= w_array.sum()
ens_probs = np.einsum("mnc,m->nc", np.stack([P[m] for m in ORDER]), w_array)
P["ensemble"] = ens_probs

# ─── Calculate Metrics ───────────────────────────────────────────────────────
print("  Calculating metrics ...")
results = {}
for m in ORDER + ["ensemble"]:
    y_pred = P[m].argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Class-wise F1
    class_f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    class_recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    results[m] = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "class_f1": {CLASS_NAMES_V2[i]: float(class_f1s[i]) for i in range(8)},
        "class_recall": {CLASS_NAMES_V2[i]: float(class_recalls[i]) for i in range(8)}
    }

# Ablation
results["ablation"] = {
    "v1_10class_ensemble_acc": 0.9504,
    "v2_8class_ensemble_acc": results["ensemble"]["accuracy"],
    "improvement": results["ensemble"]["accuracy"] - 0.9504
}

with open("results/metrics_v2.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved -> results/metrics_v2.json")

# ─── Generate Figures ────────────────────────────────────────────────────────
print("  Generating confusion matrices ...")

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES_V2, yticklabels=CLASS_NAMES_V2)
    plt.title(title, pad=20, fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}", dpi=300)
    plt.close()

plot_cm(y_true, P["cnn"].argmax(1), "Confusion Matrix — CNN (v2, 8-Class)", "cm_cnn_v2.png")
plot_cm(y_true, P["yamnet"].argmax(1), "Confusion Matrix — YAMNet (v2, 8-Class)", "cm_yamnet_v2.png")
plot_cm(y_true, P["ensemble"].argmax(1), "Confusion Matrix — Ensemble (v2, 8-Class)", "cm_ensemble_v2.png")

print("  Generating class F1 comparison bar chart ...")
plt.figure(figsize=(12, 6))
bar_width = 0.15
x = np.arange(len(CLASS_NAMES_V2))

for i, m in enumerate(ORDER):
    f1s = [results[m]["class_f1"][c] for c in CLASS_NAMES_V2]
    plt.bar(x + i*bar_width, f1s, width=bar_width, label=m.upper())

plt.title("Per-Class F1 Score Comparison Across Models", fontsize=14, pad=20)
plt.ylabel("F1 Score", fontsize=12)
plt.xticks(x + bar_width*2, CLASS_NAMES_V2, rotation=45, ha="right")
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/figures/class_f1_comparison_v2.png", dpi=300)
plt.close()

print("\n  ========================================================")
print("  ✅ Phase 7 Complete — All evaluation artifacts generated.")
print("  ========================================================")
