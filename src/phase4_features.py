"""
Phase 4 — Feature extraction with v2 (8-class) labels.
Re-extracts MFCC, Mel-Spectrogram (uses existing features dir due to labels being in CSV),
and YAMNet embeddings keyed to splits_v2.

MFCC and Mel: we reuse already-extracted per-file features (the audio hasn't changed,
only the labels). We just build new index arrays pointing to features/mfcc/ etc.
YAMNet: re-extracted and pooled from splits_v2 file lists.
"""
import os, sys, logging
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.label_map_v2 import CLASS_NAMES_V2, label_to_index
from src.utils import setup_logging, set_seed

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)

Path("features/yamnet_v2").mkdir(parents=True, exist_ok=True)

def extract_yamnet_split(split: str):
    import tensorflow as tf, tensorflow_hub as hub, librosa
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)

    y_base = hub.load("https://tfhub.dev/google/yamnet/1")
    df = pd.read_csv(f"data/splits_v2/{split}.csv")
    SR = 16000; DUR = 3; N = SR * DUR

    embs, labels = [], []
    for _, row in df.iterrows():
        fold = row["fold"]
        fname = row["slice_file_name"]
        fpath = f"data/UrbanSound8K/audio/fold{int(fold)}/{fname}"
        try:
            wav, _ = librosa.load(fpath, sr=SR, duration=DUR + 0.1)
            wav = wav[:N] if len(wav) >= N else np.pad(wav, (0, N - len(wav)))
            _, emb, _ = y_base(wav.astype(np.float32))
            embs.append(tf.reduce_mean(emb, axis=0).numpy())
            labels.append(label_to_index(row["class_v2"]))
        except Exception as e:
            logger.warning(f"  skip {fname}: {e}")

    X = np.array(embs, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    np.save(f"features/yamnet_v2/{split}_X.npy", X)
    np.save(f"features/yamnet_v2/{split}_y.npy", y)
    logger.info(f"  {split}: X={X.shape}  y={y.shape}")
    return X, y

print("  Extracting YAMNet embeddings for all splits (v2) ...")
X_tr, y_tr = extract_yamnet_split("train")
X_va, y_va = extract_yamnet_split("val")
X_te, y_te = extract_yamnet_split("test")

# Global normalization
mu = X_tr.mean(0, keepdims=True).astype(np.float32)
sig = (X_tr.std(0, keepdims=True) + 1e-8).astype(np.float32)
np.save("features/yamnet_v2/mu.npy", mu)
np.save("features/yamnet_v2/sig.npy", sig)
print(f"  Saved normalization stats: mu={mu.shape}, sig={sig.shape}")

# Build MFCC arrays from existing per-file features
print("  Building MFCC + Mel arrays from existing per-file features ...")
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits_v2/{split}.csv")
    mfccs, mels, ys = [], [], []
    missing = 0
    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        mf = Path(f"features/mfcc/{split}/{stem}.npy")
        ml = Path(f"features/mel/{split}/{stem}.npy")
        if mf.exists() and ml.exists():
            mfccs.append(np.load(mf))
            mels.append(np.load(ml))
            ys.append(label_to_index(row["class_v2"]))
        else:
            missing += 1
    if missing: print(f"  [WARN] {split}: {missing} features missing")
    Path(f"features/mfcc_v2/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"features/mel_v2/{split}").mkdir(parents=True, exist_ok=True)
    np.save(f"features/mfcc_v2/{split}/X.npy", np.stack(mfccs).astype(np.float32))
    np.save(f"features/mfcc_v2/{split}/y.npy", np.array(ys, dtype=np.int64))
    np.save(f"features/mel_v2/{split}/y.npy", np.array(ys, dtype=np.int64))
    # Save individual mel files for CNN dataset
    for i, (_, row) in enumerate(df.iterrows()):
        stem = Path(row["slice_file_name"]).stem
        ml = Path(f"features/mel/{split}/{stem}.npy")
        if ml.exists():
            import shutil
            shutil.copy(str(ml), f"features/mel_v2/{split}/{stem}.npy")
    print(f"  {split}: MFCC={np.stack(mfccs).shape}  Mel-files copied")

print("  Phase 4 complete.")
