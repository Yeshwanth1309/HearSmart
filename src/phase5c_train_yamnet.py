"""
Phase 5c — Fine-tune YAMNet on 8-class (US8K + ESC-50 embeddings combined).
Architecture: Dense(512,relu) → Drop(0.5) → Dense(256,relu) → Drop(0.4) → Dense(8,softmax)
Output: models/yamnet_v2.h5
"""
import os, sys, logging
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.label_map_v2 import CLASS_NAMES_V2, label_to_index, ESC50_TO_8CLASS
from src.utils import setup_logging, set_seed

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)

import tensorflow as tf, tensorflow_hub as hub, librosa
for g in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(g, True)

print("\n  Loading ESC-50 embeddings ...")
from src.data.esc50_loader import load_esc50_metadata
esc_train_df, esc_val_df = load_esc50_metadata("data/ESC50/meta/esc50.csv")
y_base = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_esc_embeddings(df, split_name):
    embs, lbls = [], []
    for _, row in df.iterrows():
        fp = f"data/ESC50/audio/{row['filename']}"
        try:
            wav, _ = librosa.load(fp, sr=16000, duration=5.0)
            wav = wav[:80000] if len(wav) > 80000 else np.pad(wav, (0, max(0, 80000-len(wav))))
            emb = tf.reduce_mean(y_base(wav.astype(np.float32))[1], 0).numpy()
            embs.append(emb)
            lbls.append(label_to_index(ESC50_TO_8CLASS.get(row["category"], "background_noise")))
        except Exception as e:
            pass
    arr = np.array(embs, dtype=np.float32)
    logger.info(f"  ESC-50 {split_name}: {arr.shape}")
    return arr, np.array(lbls, dtype=np.int64)

X_esc_tr, y_esc_tr = extract_esc_embeddings(esc_train_df, "train")
X_esc_va, y_esc_va = extract_esc_embeddings(esc_val_df,   "val")

# Load US8K embeddings
X_us_tr = np.load("features/yamnet_v2/train_X.npy")
y_us_tr = np.load("features/yamnet_v2/train_y.npy")
X_us_va = np.load("features/yamnet_v2/val_X.npy")
y_us_va = np.load("features/yamnet_v2/val_y.npy")
X_us_te = np.load("features/yamnet_v2/test_X.npy")
y_us_te = np.load("features/yamnet_v2/test_y.npy")

# Combine
X_tr = np.vstack([X_us_tr, X_esc_tr]); y_tr = np.concatenate([y_us_tr, y_esc_tr])
X_va = np.vstack([X_us_va, X_esc_va]); y_va = np.concatenate([y_us_va, y_esc_va])

# Normalize with global stats from Phase 4
mu  = np.load("features/yamnet_v2/mu.npy")
sig = np.load("features/yamnet_v2/sig.npy")
X_tr_n = ((X_tr - mu) / sig).astype(np.float32)
X_va_n = ((X_va - mu) / sig).astype(np.float32)
X_te_n = ((X_us_te - mu) / sig).astype(np.float32)

print(f"  Combined Train: {X_tr_n.shape}  Val: {X_va_n.shape}  Test: {X_te_n.shape}")

# Class weights
cnt = Counter(y_tr.tolist())
cw = {k: len(y_tr)/(8*max(1,v)) for k,v in cnt.items()}
print("  Class weights:", {CLASS_NAMES_V2[k]: round(v,2) for k,v in cw.items()})

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(8, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cbs = [
    tf.keras.callbacks.ModelCheckpoint("models/yamnet_v2.h5", monitor="val_accuracy",
                                       save_best_only=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                          patience=8, min_lr=1e-5, verbose=0),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                      restore_best_weights=True, verbose=0),
]

print("  Training YAMNet v2 (80 epochs max) ...")
model.fit(X_tr_n, y_tr, validation_data=(X_va_n, y_va),
          epochs=80, batch_size=32, class_weight=cw, callbacks=cbs, verbose=0)

model = tf.keras.models.load_model("models/yamnet_v2.h5")
y_pred = model.predict(X_te_n, verbose=0).argmax(1)
acc = accuracy_score(y_us_te, y_pred)
f1  = f1_score(y_us_te, y_pred, average="macro", zero_division=0)
print(f"\n  YAMNet v2 Test: acc={acc:.4f}  macro-F1={f1:.4f}")
print(classification_report(y_us_te, y_pred, target_names=CLASS_NAMES_V2, zero_division=0))
print("  Phase 5c complete.")
