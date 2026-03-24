"""
Phase 11 – Full ESC-50 Retrain.
Trains:
  1. ESC50_CNN  → models/cnn_esc50.pth
  2. YAMNet head → models/yamnet_esc50.h5

Then optimises a 7-model ensemble (5 US8K + 2 ESC50 specialists) on
the US8K validation set and saves to results/ensemble_weights_v3.json.

Run from project root:
    python src/train_esc50_full.py
"""
import os, sys, json, time, logging
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from scipy.optimize import minimize
import joblib

from src.data.esc50_loader import (ESC50Dataset, load_esc50_metadata,
                                   US8K_CLASSES, class_weights_from_df)
from src.models import AudioCNN
from src.utils import set_seed, setup_logging

# ── housekeeping ──────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)

AUDIO_DIR = "data/ESC50/audio/"
META_CSV  = "data/ESC50/meta/esc50.csv"
N_CLASSES = 10

# ─────────────────────────────────────────────────────────────────────────────
# 1.  ESC-50 CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────
class ESC50_CNN(nn.Module):
    """3-block CNN with Global Average Pooling (as specified)."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        # Head
        self.fc1  = nn.Linear(128, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2  = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNN Training
# ─────────────────────────────────────────────────────────────────────────────
def train_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  TRAINING ESC-50 CNN on {device}")
    print(f"{'='*60}")

    train_df, test_df = load_esc50_metadata(META_CSV)

    # ── class-balanced sampler ────────────────────────────────────────────
    cw = class_weights_from_df(train_df)
    sample_weights = np.array([
        cw[US8K_CLASSES.index(lbl)] for lbl in train_df["us8k_label"]
    ], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    train_ds = ESC50Dataset(train_df, AUDIO_DIR, augment=True)
    test_ds  = ESC50Dataset(test_df,  AUDIO_DIR, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,
                              num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False,
                              num_workers=0)

    # ── model ────────────────────────────────────────────────────────────
    model   = ESC50_CNN(N_CLASSES).to(device)
    weights  = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_acc, best_f1 = 0.0, 0.0
    for epoch in range(1, 81):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # eval
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                p = model(x.to(device)).argmax(1).cpu()
                preds.extend(p.numpy())
                labels.extend(y.numpy())
        acc = accuracy_score(labels, preds)
        mf1 = f1_score(labels, preds, average="macro", zero_division=0)
        if acc > best_acc:
            best_acc = acc; best_f1 = mf1
            torch.save(model.state_dict(), "models/cnn_esc50.pth")
        if epoch % 10 == 0:
            print(f"  Ep {epoch:3d} | loss={total_loss/len(train_loader):.4f} "
                  f"| acc={acc:.4f} | f1={mf1:.4f}  (best={best_acc:.4f})")

    print(f"\n  CNN done. Best acc={best_acc:.4f}  f1={best_f1:.4f}")
    return best_acc, best_f1

# ─────────────────────────────────────────────────────────────────────────────
# 3.  YAMNet Fine-tune
# ─────────────────────────────────────────────────────────────────────────────
def train_yamnet():
    import tensorflow as tf, tensorflow_hub as hub
    import librosa

    print(f"\n{'='*60}")
    print(f"  FINE-TUNING YAMNet on ESC-50 (full dataset)")
    print(f"{'='*60}")

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)

    y_base = hub.load("https://tfhub.dev/google/yamnet/1")
    train_df, test_df = load_esc50_metadata(META_CSV)

    def extract(df):
        embs, lbls = [], []
        for _, row in df.iterrows():
            fp = os.path.join(AUDIO_DIR, row["filename"])
            try:
                wav, _ = librosa.load(fp, sr=16000, duration=5.0)
                # use full 5-second context for ESC-50
                wav = wav[:80000] if len(wav) > 80000 else np.pad(wav, (0, 80000 - len(wav)))
                _, emb, _ = y_base(wav.astype(np.float32))
                embs.append(tf.reduce_mean(emb, 0).numpy())
                lbls.append(US8K_CLASSES.index(ESC50_TO_US8K_MAP_local(row["category"])))
            except Exception as e:
                logger.warning(f"skip {fp}: {e}")
        return np.array(embs, dtype=np.float32), np.array(lbls)

    # local map import
    from src.data.esc50_loader import ESC50_TO_US8K_MAP as ESC50_TO_US8K_MAP_local

    print("  Extracting train embeddings …")
    X_tr, y_tr = extract(train_df)
    print("  Extracting test  embeddings …")
    X_te, y_te = extract(test_df)

    # normalise with US8K stats
    try:
        E_us = np.load("features/yamnet_embeddings/train.npy")
        mu  = E_us.mean(0, keepdims=True)
        sig = E_us.std(0,  keepdims=True) + 1e-8
    except FileNotFoundError:
        mu  = X_tr.mean(0, keepdims=True)
        sig = X_tr.std(0,  keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / sig
    X_te = (X_te - mu) / sig

    # class weights from train_df
    cw = class_weights_from_df(train_df)
    cw_dict = {i: float(cw[i]) for i in range(10)}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb = tf.keras.callbacks.ModelCheckpoint(
        "models/yamnet_esc50.h5", monitor="val_accuracy",
        save_best_only=True, verbose=0,
    )
    model.fit(X_tr, y_tr,
              validation_data=(X_te, y_te),
              epochs=80, batch_size=16,
              class_weight=cw_dict,
              callbacks=[cb], verbose=1)

    # restore best
    model = tf.keras.models.load_model("models/yamnet_esc50.h5")
    y_pred = model.predict(X_te, verbose=0).argmax(1)
    acc = accuracy_score(y_te, y_pred)
    mf1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\n  YAMNet-ESC50: acc={acc:.4f}  macro-f1={mf1:.4f}")
    print(classification_report(y_te, y_pred, target_names=US8K_CLASSES,
                                zero_division=0))
    return acc, mf1

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Ensemble optimisation (7-model)
# ─────────────────────────────────────────────────────────────────────────────
def optimise_ensemble():
    import tensorflow as tf, tensorflow_hub as hub
    from xgboost import XGBClassifier

    print(f"\n{'='*60}")
    print(f"  OPTIMISING 7-MODEL ENSEMBLE ON US8K VAL SET")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load all models
    rf  = joblib.load("models/random_forest.pkl")
    svm_m = joblib.load("models/svm.pkl")
    xgb_m = XGBClassifier(); xgb_m.load_model("models/xgboost.json")

    cnn_us8k = AudioCNN(num_classes=10).to(device)
    cnn_us8k.load_state_dict(torch.load("models/cnn_best.pt", map_location=device))
    cnn_us8k.eval()

    cnn_esc = ESC50_CNN(N_CLASSES).to(device)
    cnn_esc.load_state_dict(torch.load("models/cnn_esc50.pth", map_location=device))
    cnn_esc.eval()

    y_head_us8k = tf.keras.models.load_model("models/yamnet_head.h5")
    y_head_esc  = tf.keras.models.load_model("models/yamnet_esc50.h5")

    E_train = np.load("features/yamnet_embeddings/train.npy")
    emb_mu  = E_train.mean(0, keepdims=True)
    emb_sig = E_train.std(0,  keepdims=True) + 1e-8

    # get val probs
    val_df = pd.read_csv("data/splits/val.csv")
    y_val  = np.array([US8K_CLASSES.index(c) for c in val_df["class"]])

    def get_probs(split_df, split_name):
        from pathlib import Path as P
        import torch.nn.functional as tF
        mfcc_list, cnn_list, cnn_esc_list = [], [], []
        for _, row in split_df.iterrows():
            stem = P(row["slice_file_name"]).stem
            mfcc_list.append(np.load(f"features/mfcc/{split_name}/{stem}.npy"))
            mel = np.load(f"features/mel/{split_name}/{stem}.npy")
            mel_t = torch.from_numpy(mel.transpose(2,0,1)[None].astype(np.float32)).to(device)
            with torch.no_grad():
                cnn_list.append(tF.softmax(cnn_us8k(mel_t),1).cpu().numpy()[0])
                cnn_esc_list.append(tF.softmax(cnn_esc(mel_t),1).cpu().numpy()[0])
        mfcc = np.stack(mfcc_list)
        probs = {}
        probs["rf"]  = rf.predict_proba(mfcc).astype(np.float32)
        try: probs["svm"] = svm_m.predict_proba(mfcc).astype(np.float32)
        except:
            sc = svm_m.named_steps["scaler"].transform(mfcc)
            d  = svm_m.named_steps["svm"].decision_function(sc).astype(np.float32)
            d -= d.max(1, keepdims=True)
            probs["svm"] = np.exp(d) / np.exp(d).sum(1, keepdims=True)
        probs["xgb"]       = xgb_m.predict_proba(mfcc).astype(np.float32)
        probs["cnn"]       = np.stack(cnn_list).astype(np.float32)
        probs["cnn_esc"]   = np.stack(cnn_esc_list).astype(np.float32)
        y_emb = np.load(f"features/yamnet_embeddings/{split_name}.npy")
        y_emb_n = ((y_emb - emb_mu) / emb_sig).astype(np.float32)
        probs["yamnet"]     = y_head_us8k.predict(y_emb_n, verbose=0).astype(np.float32)
        probs["yamnet_esc"] = y_head_esc.predict(y_emb_n, verbose=0).astype(np.float32)
        return probs

    print("  Loading val probs …")
    val_probs = get_probs(val_df, "val")
    print("  Loading test probs …")
    test_df   = pd.read_csv("data/splits/test.csv")
    test_probs = get_probs(test_df, "test")
    y_test = np.array([US8K_CLASSES.index(c) for c in test_df["class"]])

    order = ["rf", "svm", "xgb", "cnn", "cnn_esc", "yamnet", "yamnet_esc"]

    def neg_acc(w, probs=val_probs, y=y_val):
        w = np.abs(w); w /= w.sum()
        P = np.stack([probs[m] for m in order])
        ens = np.einsum("mnc,m->nc", P, w)
        return -accuracy_score(y, ens.argmax(1))

    best_val, best_w = 0, np.ones(len(order)) / len(order)
    print("  Optimising weights (30 restarts) …")
    for _ in range(30):
        w0 = np.random.dirichlet(np.ones(len(order)))
        res = minimize(neg_acc, w0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-8})
        w = np.abs(res.x); w /= w.sum()
        va = -res.fun
        if va > best_val: best_val, best_w = va, w

    P_test = np.stack([test_probs[m] for m in order])
    ens_test = np.einsum("mnc,m->nc", P_test, best_w)
    test_preds = ens_test.argmax(1)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1  = f1_score(y_test, test_preds, average="macro")

    print(f"\n  7-Model Ensemble:")
    print(f"  Val  Accuracy: {best_val:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}  Macro F1: {test_f1:.4f}")
    for m, w in zip(order, best_w):
        print(f"    {m:12s}: {w:.4f}")
    print(classification_report(y_test, test_preds, target_names=US8K_CLASSES))

    result = {
        "weights": dict(zip(order, best_w.tolist())),
        "model_order": order,
        "val_accuracy":  float(best_val),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
    }
    with open("results/ensemble_weights_v3.json", "w") as f:
        json.dump(result, f, indent=2)
    print("  Saved → results/ensemble_weights_v3.json")
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    cnn_acc, cnn_f1     = train_cnn()
    yam_acc, yam_f1     = train_yamnet()
    ensemble_result     = optimise_ensemble()

    print(f"\n{'='*60}")
    print(f"  PHASE 11 COMPLETE  ({(time.time()-t0)/60:.1f} min)")
    print(f"  CNN  (ESC-50): acc={cnn_acc:.4f}  f1={cnn_f1:.4f}")
    print(f"  YAM  (ESC-50): acc={yam_acc:.4f}  f1={yam_f1:.4f}")
    print(f"  7-Model:       acc={ensemble_result['test_accuracy']:.4f}  "
          f"f1={ensemble_result['test_macro_f1']:.4f}")
    print(f"{'='*60}")
