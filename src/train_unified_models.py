"""
Unified Model Training (UrbanSound8K + FULL ESC-50).

Combines the datasets to train a single robust CNN and a single robust YAMNet head.
Removes Random Forest from the final ensemble.
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.optimize import minimize
import joblib

from src.models import AudioCNN
from src.utils import set_seed, setup_logging
from src.data.esc50_loader import (ESC50Dataset, load_esc50_metadata,
                                   US8K_CLASSES, ESC50_TO_US8K_MAP)

setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)

AUDIO_DIR_ESC  = "data/ESC50/audio/"
META_CSV_ESC   = "data/ESC50/meta/esc50.csv"
N_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Unified Dataset for CNN (Loads pre-extracted US8K + raw ESC50)
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedCNNDataset(Dataset):
    def __init__(self, us8k_split, esc_df, augment=False):
        self.augment = augment
        
        # 1. Setup US8K paths
        self.us8k_df = pd.read_csv(f"data/splits/{us8k_split}.csv")
        self.us8k_dir = f"features/mel/{us8k_split}/"
        
        # 2. Setup ESC50 loader
        self.esc_ds = ESC50Dataset(esc_df, AUDIO_DIR_ESC, augment=augment)
        
        self.n_us8k = len(self.us8k_df)
        self.n_esc  = len(esc_df)
        self.total  = self.n_us8k + self.n_esc

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < self.n_us8k:
            # Load US8K pre-extracted
            row = self.us8k_df.iloc[idx]
            stem = os.path.splitext(row["slice_file_name"])[0]
            mel = np.load(os.path.join(self.us8k_dir, f"{stem}.npy"))
            # US8K shape is (128, 128, 1). Torch wants (1, 128, 128)
            mel_t = torch.from_numpy(mel.transpose(2,0,1).astype(np.float32))
            label = US8K_CLASSES.index(row["class"])
            return mel_t, torch.tensor(label, dtype=torch.long)
        else:
            # Load from ESC50 Dataset
            # ESC50Dataset returns (1, 128, 128) tensor already
            esc_idx = idx - self.n_us8k
            return self.esc_ds[esc_idx]


def get_unified_class_weights(us8k_df, esc_df):
    counts = {c: 0 for c in US8K_CLASSES}
    # US8K counts
    for c in us8k_df["class"]: counts[c] += 1
    # ESC50 counts
    for c in esc_df["us8k_label"]: counts[c] += 1
    
    total = sum(counts.values())
    weights = np.zeros(10, dtype=np.float32)
    for i, c in enumerate(US8K_CLASSES):
        weights[i] = total / (10 * max(1, counts[c]))
    return weights

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train Unified CNN
# ─────────────────────────────────────────────────────────────────────────────
def train_unified_cnn():
    print(f"\n{'='*60}\n  TRAINING UNIFIED CNN (US8K + ESC50) on {DEVICE}\n{'='*60}")
    
    esc_train_df, esc_test_df = load_esc50_metadata(META_CSV_ESC)
    us8k_train_df = pd.read_csv("data/splits/train.csv")
    
    train_ds = UnifiedCNNDataset("train", esc_train_df, augment=True)
    val_ds   = UnifiedCNNDataset("val",   esc_test_df,  augment=False) # Evaluated on combined val set
    
    # Class weights for loss
    cw = get_unified_class_weights(us8k_train_df, esc_train_df)
    weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
    
    # We use the robust architecture that performed well on ESC50
    # or the standard US8K CNN. Using the AudioCNN from src.models.
    model = AudioCNN(num_classes=N_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    best_acc = 0.0
    for epoch in range(1, 51):
        model.train()
        tot_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()
        
        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
                acts.extend(y.numpy())
        acc = accuracy_score(acts, preds)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/cnn_unified.pt")
            
        if epoch % 5 == 0:
            print(f"  Ep {epoch:2d} | loss={tot_loss/len(train_loader):.4f} | val_acc={acc:.4f} (best={best_acc:.4f})")
            
    print(f"  Unified CNN Done. Best Val Acc: {best_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Train Unified YAMNet
# ─────────────────────────────────────────────────────────────────────────────
def train_unified_yamnet():
    import tensorflow as tf, tensorflow_hub as hub
    import librosa
    
    print(f"\n{'='*60}\n  TRAINING UNIFIED YAMNet (US8K + ESC50)\n{'='*60}")
    for g in tf.config.list_physical_devices("GPU"): 
        tf.config.experimental.set_memory_growth(g, True)
        
    y_base = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # -- 1. Load US8K Embeddings
    X_us8k_tr = np.load("features/yamnet_embeddings/train.npy")
    y_us8k_tr = np.array([US8K_CLASSES.index(c) for c in pd.read_csv("data/splits/train.csv")["class"]])
    
    X_us8k_val = np.load("features/yamnet_embeddings/val.npy")
    y_us8k_val = np.array([US8K_CLASSES.index(c) for c in pd.read_csv("data/splits/val.csv")["class"]])
    
    # -- 2. Extract ESC50 Embeddings (cached if possible, but fast enough to re-extract)
    esc_train_df, esc_test_df = load_esc50_metadata(META_CSV_ESC)
    
    def extract_esc(df):
        embs, lbls = [], []
        from src.data.esc50_loader import ESC50_TO_US8K_MAP as E_MAP
        for _, row in df.iterrows():
            try:
                wav, _ = librosa.load(os.path.join(AUDIO_DIR_ESC, row["filename"]), sr=16000, duration=5.0)
                wav = wav[:80000] if len(wav) > 80000 else np.pad(wav, (0, max(0, 80000-len(wav))))
                embs.append(tf.reduce_mean(y_base(wav.astype(np.float32))[1], 0).numpy())
                lbls.append(US8K_CLASSES.index(E_MAP[row["category"]]))
            except: pass
        return np.array(embs, dtype=np.float32), np.array(lbls)

    print("  Extracting ESC50 train...")
    X_esc_tr, y_esc_tr = extract_esc(esc_train_df)
    print("  Extracting ESC50 val...")
    X_esc_val, y_esc_val = extract_esc(esc_test_df)
    
    # -- 3. Combine
    X_tr = np.vstack([X_us8k_tr, X_esc_tr])
    y_tr = np.concatenate([y_us8k_tr, y_esc_tr])
    X_val = np.vstack([X_us8k_val, X_esc_val])
    y_val = np.concatenate([y_us8k_val, y_esc_val])
    
    # -- 4. Normalize globally
    mu  = X_tr.mean(0, keepdims=True)
    sig = X_tr.std(0, keepdims=True) + 1e-8
    
    np.save("models/yamnet_unified_mu.npy", mu)
    np.save("models/yamnet_unified_sig.npy", sig)
    
    X_tr_n = (X_tr - mu) / sig
    X_val_n = (X_val - mu) / sig
    
    # -- Class weights
    from collections import Counter
    cnt = Counter(y_tr)
    cw = {k: len(y_tr)/(10*v) for k,v in cnt.items()}
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    cb = tf.keras.callbacks.ModelCheckpoint("models/yamnet_unified.h5", 
                                            monitor="val_accuracy", save_best_only=True, verbose=0)
    
    model.fit(X_tr_n, y_tr, validation_data=(X_val_n, y_val),
              epochs=60, batch_size=32, class_weight=cw, callbacks=[cb], verbose=0)
              
    y_p = tf.keras.models.load_model("models/yamnet_unified.h5").predict(X_val_n, verbose=0).argmax(1)
    acc = accuracy_score(y_val, y_p)
    print(f"  Unified YAMNet Done. Best Val Acc: {acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Ensemble Optimization (4 Models)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_unified_ensemble():
    import tensorflow as tf
    from xgboost import XGBClassifier
    print(f"\n{'='*60}\n  OPTIMIZING UNIFIED ENSEMBLE (4 MODELS)\n{'='*60}")
    
    # Clean 4 models: SVM, XGBoost, CNN(Unified), YAMNet(Unified)
    # Removing Random Forest entirely as requested to reduce clutter/overhead.
    svm_m = joblib.load("models/svm.pkl")
    xgb_m = XGBClassifier(); xgb_m.load_model("models/xgboost.json")
    
    cnn_u = AudioCNN(num_classes=10).to(DEVICE)
    cnn_u.load_state_dict(torch.load("models/cnn_unified.pt", map_location=DEVICE))
    cnn_u.eval()
    
    yam_u = tf.keras.models.load_model("models/yamnet_unified.h5")
    ymu = np.load("models/yamnet_unified_mu.npy")
    ysig = np.load("models/yamnet_unified_sig.npy")
    
    def get_probs(split_name):
        df = pd.read_csv(f"data/splits/{split_name}.csv")
        y = np.array([US8K_CLASSES.index(c) for c in df["class"]])
        
        mfccs, cnn_ps = [], []
        import torch.nn.functional as tF
        from pathlib import Path
        for _, row in df.iterrows():
            stem = Path(row["slice_file_name"]).stem
            mfccs.append(np.load(f"features/mfcc/{split_name}/{stem}.npy"))
            mel = np.load(f"features/mel/{split_name}/{stem}.npy")
            mel_t = torch.from_numpy(mel.transpose(2,0,1)[None].astype(np.float32)).to(DEVICE)
            with torch.no_grad():
                cnn_ps.append(tF.softmax(cnn_u(mel_t),1).cpu().numpy()[0])
                
        M = np.stack(mfccs)
        P = {}
        # SVM logic
        try: P["svm"] = svm_m.predict_proba(M).astype(np.float32)
        except:
            sc = svm_m.named_steps["scaler"].transform(M)
            d  = svm_m.named_steps["svm"].decision_function(sc).astype(np.float32)
            d -= d.max(1,keepdims=True)
            P["svm"] = np.exp(d)/np.exp(d).sum(1,keepdims=True)
            
        P["xgb"] = xgb_m.predict_proba(M).astype(np.float32)
        P["cnn"] = np.stack(cnn_ps).astype(np.float32)
        
        yemb = np.load(f"features/yamnet_embeddings/{split_name}.npy")
        yemb_n = ((yemb - ymu) / ysig).astype(np.float32)
        P["yamnet"] = yam_u.predict(yemb_n, verbose=0).astype(np.float32)
        
        return P, y

    print("  Loading Validation Data...")
    val_P, y_val = get_probs("val")
    print("  Loading Test Data...")
    test_P, y_test = get_probs("test")
    
    ORDER = ["svm", "xgb", "cnn", "yamnet"]
    def neg_acc(w):
        w = np.abs(w); w /= w.sum()
        ens = np.einsum("mnc,m->nc", np.stack([val_P[m] for m in ORDER]), w)
        return -accuracy_score(y_val, ens.argmax(1))
        
    best_val, best_w = 0, np.ones(4)/4
    for _ in range(30):
        w0 = np.random.dirichlet(np.ones(4))
        r = minimize(neg_acc, w0, method="Nelder-Mead", options={"maxiter":2000})
        w = np.abs(r.x); w /= w.sum()
        v = -r.fun
        if v > best_val: best_val, best_w = v, w
        
    ens_te = np.einsum("mnc,m->nc", np.stack([test_P[m] for m in ORDER]), best_w)
    te_p = ens_te.argmax(1)
    acc, mf1 = accuracy_score(y_test, te_p), f1_score(y_test, te_p, average="macro")
    
    print(f"\n  Final 4-Model Unified Ensemble:")
    print(f"  Test Accuracy: {acc:.4f}  Macro F1: {mf1:.4f}")
    for m, w in zip(ORDER, best_w): print(f"    {m:10s}: {w:.4f}")
    
    result = {
        "weights": dict(zip(ORDER, best_w.tolist())), "model_order": ORDER,
        "test_accuracy": float(acc), "test_macro_f1": float(mf1)
    }
    with open("results/ensemble_weights_unified.json","w") as f: json.dump(result, f, indent=2)

if __name__ == "__main__":
    t0 = time.time()
    train_unified_cnn()
    train_unified_yamnet()
    optimize_unified_ensemble()
    print(f"\n  ALL UNIFIED TRAINING COMPLETE in {(time.time()-t0)/60:.1f} mins.")
