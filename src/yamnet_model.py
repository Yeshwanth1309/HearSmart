"""
Phase 5 — YAMNet Transfer Learning Model.

YAMNet processes 1-second windows internally, so a 3-second clip produces
~6 frames of 1024-dim embeddings. We mean-pool them → (1024,) per sample,
then train a classification head on top.

Architecture per TRD §3.5.2:
    YAMNet (frozen) → temporal mean-pool → Dense(256,relu) → Dropout(0.5) → Dense(10,softmax)

Two-phase training:
    Phase 1: Head-only   (20 epochs, lr=1e-3, frozen YAMNet)
    Phase 2: Fine-tune   (15 epochs, lr=1e-4, YAMNet trainable)
"""

import json
from json import JSONEncoder
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflow_hub as hub

from src.utils import setup_logging, set_seed

# ─────────────────────────────────────────────────────────────────────────────
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SR = 16000
N_SAMPLES = 48000  # 3 s × 16 kHz
SEED = 42


def _configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        logging.getLogger(__name__).info(
            f"GPU memory growth enabled: {[g.name for g in gpus]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extractor (uses YAMNet as frozen feature extractor)
# ─────────────────────────────────────────────────────────────────────────────
class YAMNetEmbedder:
    """
    Wraps the raw YAMNet SavedModel to extract mean-pooled embeddings.

    Usage
    -----
    embedder = YAMNetEmbedder()
    embedder.load()
    embeddings = embedder.embed_batch(X)   # (N, 48000) → (N, 1024)
    """

    def __init__(self, url: str = YAMNET_URL):
        self.url = url
        self.model = None

    def load(self):
        logger = logging.getLogger(__name__)
        logger.info(f"Loading YAMNet from {self.url} ...")
        self.model = hub.load(self.url)
        logger.info("YAMNet loaded.")
        return self

    @tf.function(input_signature=[tf.TensorSpec(shape=[N_SAMPLES], dtype=tf.float32)])
    def _embed_single(self, waveform):
        """Return mean-pooled embedding for a single 3-second waveform."""
        _, embeddings, _ = self.model(waveform)   # embeddings: (frames, 1024)
        return tf.reduce_mean(embeddings, axis=0)  # (1024,)

    def embed_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Extract embeddings for a batch of waveforms.

        Parameters
        ----------
        X : np.ndarray, shape (N, 48000), dtype float32

        Returns
        -------
        np.ndarray, shape (N, 1024)
        """
        embs = []
        for i, wave in enumerate(X):
            wav = tf.constant(wave, dtype=tf.float32)
            emb = self._embed_single(wav)
            embs.append(emb.numpy())
            if (i + 1) % 500 == 0:
                logging.getLogger(__name__).info(
                    f"  Embedded {i+1}/{len(X)} samples..."
                )
        return np.stack(embs, axis=0)  # (N, 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Classification head
# ─────────────────────────────────────────────────────────────────────────────
def build_classifier_head(
    num_classes: int = 10,
    input_dim: int = 1024,
) -> tf.keras.Model:
    """
    Build the trainable classification head.

    Input  : (batch, 1024) — YAMNet mean-pooled embeddings
    Output : (batch, num_classes) — softmax probabilities
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="yamnet_embedding"),
            tf.keras.layers.Dense(256, activation="relu", name="fc1"),
            tf.keras.layers.Dropout(0.5, name="dropout"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="yamnet_classifier_head",
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_split(split_csv, feature_dir, label_encoder=None):
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(split_csv)
    feat_path = Path(feature_dir)
    X_list, labels = [], []
    missing = 0

    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy = feat_path / f"{stem}.npy"
        if not npy.exists():
            missing += 1
            continue
        arr = np.load(npy).astype(np.float32)
        if arr.shape != (N_SAMPLES,):
            if len(arr) >= N_SAMPLES:
                arr = arr[:N_SAMPLES]
            else:
                arr = np.pad(arr, (0, N_SAMPLES - len(arr)))
        X_list.append(arr)
        labels.append(row["class"])

    if missing:
        logging.getLogger(__name__).warning(
            f"{missing} files missing in {feature_dir}"
        )

    X = np.stack(X_list, axis=0)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
    else:
        y = label_encoder.transform(labels)

    logging.getLogger(__name__).info(
        f"Loaded {split_csv}: {X.shape}"
    )
    return X, y.astype(np.int32), label_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _compute_metrics(head_model, X_emb, y, class_names, batch_size=64):
    from sklearn.metrics import (
        accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score,
    )

    probs = head_model.predict(X_emb, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    return {
        "accuracy": round(float(accuracy_score(y, y_pred)), 6),
        "macro_f1": round(float(f1_score(y, y_pred, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y, y_pred, average="weighted")), 6),
        "precision_macro": round(float(precision_score(y, y_pred, average="macro")), 6),
        "recall_macro": round(float(recall_score(y, y_pred, average="macro")), 6),
        "per_class_f1": {
            class_names[i]: round(float(v), 6)
            for i, v in enumerate(f1_score(y, y_pred, average=None))
        },
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_yamnet(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    feature_dir_train: str = "features/waveform/train",
    feature_dir_val: str = "features/waveform/val",
    feature_dir_test: str = "features/waveform/test",
    head_save_path: str = "models/yamnet_head.h5",
    emb_cache_dir: str = "features/yamnet_embeddings",
    metrics_save_path: str = "results/yamnet_metrics.json",
    phase1_epochs: int = 20,
    phase1_lr: float = 1e-3,
    phase2_epochs: int = 15,
    phase2_lr: float = 5e-4,
    batch_size: int = 64,
    patience: int = 7,
    seed: int = SEED,
) -> dict:
    """
    Two-phase YAMNet training.

    Embeddings are extracted once and cached to disk, so re-runs skip
    the expensive YAMNet forward pass.

    Returns
    -------
    dict — complete metrics + training history
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)
    tf.random.set_seed(seed)
    _configure_gpu()

    os.makedirs(os.path.dirname(head_save_path) or "models", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_save_path) or "results", exist_ok=True)
    os.makedirs(emb_cache_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 5 — YAMNet Transfer Learning")
    logger.info("=" * 60)

    # ── 1. Load waveforms ────────────────────────────────────────────────────
    logger.info("Loading raw waveform splits...")
    X_train, y_train, enc = _load_split(train_csv, feature_dir_train)
    X_val,   y_val,   _   = _load_split(val_csv, feature_dir_val, enc)
    X_test,  y_test,  _   = _load_split(test_csv, feature_dir_test, enc)
    class_names = list(enc.classes_)
    num_classes = len(class_names)
    logger.info(f"Classes: {class_names}")

    # ── 2. Extract / load cached YAMNet embeddings ───────────────────────────
    def _get_embeddings(X, split_name, embedder):
        cache_path = os.path.join(emb_cache_dir, f"{split_name}.npy")
        if os.path.exists(cache_path):
            logger.info(f"  Loading cached embeddings: {cache_path}")
            return np.load(cache_path)
        logger.info(f"  Extracting {split_name} embeddings ({len(X)} samples)...")
        emb = embedder.embed_batch(X)
        np.save(cache_path, emb)
        logger.info(f"  Cached to {cache_path}, shape={emb.shape}")
        return emb

    embedder = YAMNetEmbedder().load()

    logger.info("Extracting YAMNet embeddings (or loading cache)...")
    E_train = _get_embeddings(X_train, "train", embedder)
    E_val   = _get_embeddings(X_val,   "val",   embedder)
    E_test  = _get_embeddings(X_test,  "test",  embedder)

    logger.info(
        f"Embeddings — Train: {E_train.shape}, "
        f"Val: {E_val.shape}, Test: {E_test.shape}"
    )

    # Normalize embeddings
    mean = E_train.mean(axis=0, keepdims=True)
    std  = E_train.std(axis=0, keepdims=True) + 1e-8
    E_train_n = ((E_train - mean) / std).astype(np.float32)
    E_val_n   = ((E_val   - mean) / std).astype(np.float32)
    E_test_n  = ((E_test  - mean) / std).astype(np.float32)

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)

    # ── 3. Phase 1: Train head on frozen embeddings ──────────────────────────
    logger.info("\n--- Phase 1: Head Training (lr=%.4f, epochs=%d) ---" %
                (phase1_lr, phase1_epochs))

    head = build_classifier_head(num_classes=num_classes)
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase1_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    head.summary(print_fn=logger.info)

    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            head_save_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]

    hist_p1 = head.fit(
        E_train_n, y_train_oh,
        validation_data=(E_val_n, y_val_oh),
        epochs=phase1_epochs,
        batch_size=batch_size,
        callbacks=callbacks_p1,
        verbose=2,
    )

    best_p1_val_loss = min(hist_p1.history["val_loss"])
    logger.info(f"Phase 1 complete. Best val_loss: {best_p1_val_loss:.4f}")

    # ── 4. Phase 2: Fine-tune with lower LR ──────────────────────────────────
    logger.info("\n--- Phase 2: Fine-tune Head (lr=%.5f, epochs=%d) ---" %
                (phase2_lr, phase2_epochs))

    head = tf.keras.models.load_model(head_save_path)
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=phase2_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            head_save_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]

    hist_p2 = head.fit(
        E_train_n, y_train_oh,
        validation_data=(E_val_n, y_val_oh),
        epochs=phase2_epochs,
        batch_size=batch_size,
        callbacks=callbacks_p2,
        verbose=2,
    )

    best_p2_val_loss = min(hist_p2.history["val_loss"])
    logger.info(f"Phase 2 complete. Best val_loss: {best_p2_val_loss:.4f}")

    # Load final best
    head = tf.keras.models.load_model(head_save_path)

    # ── 5. Test-set evaluation ────────────────────────────────────────────────
    logger.info("\nEvaluating on test set...")
    test_metrics = _compute_metrics(head, E_test_n, y_test, class_names)

    logger.info(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    for cls, f1 in test_metrics["per_class_f1"].items():
        logger.info(f"    {cls:25s}: F1={f1:.4f}")

    # ── 6. Latency (head inference on pre-extracted embedding) ───────────────
    logger.info("\nMeasuring head inference latency...")
    sample_emb = E_test_n[:1]  # (1, 1024)
    for _ in range(5):
        head.predict(sample_emb, verbose=0)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        head.predict(sample_emb, verbose=0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    avg_latency_ms = round(float(np.mean(times)), 4)
    logger.info(f"  Head latency: {avg_latency_ms:.2f} ms")

    # ── 7. Model size ────────────────────────────────────────────────────────
    model_size_mb = round(os.path.getsize(head_save_path) / (1024 * 1024), 4)

    # ── 8. Export ────────────────────────────────────────────────────────────
    metrics = {
        "model": "YAMNet + Classification Head",
        "framework": "TensorFlow/Keras + TF Hub",
        "base_model": YAMNET_URL,
        **test_metrics,
        "latency_ms_head": avg_latency_ms,
        "model_size_mb": model_size_mb,
        "embedding_dim": 1024,
        "epochs_phase1": len(hist_p1.history["loss"]),
        "epochs_phase2": len(hist_p2.history["loss"]),
        "best_val_loss_phase1": round(best_p1_val_loss, 6),
        "best_val_loss_phase2": round(best_p2_val_loss, 6),
        "hyperparameters": {
            "phase1_epochs": phase1_epochs,
            "phase1_lr": phase1_lr,
            "phase2_epochs": phase2_epochs,
            "phase2_lr": phase2_lr,
            "batch_size": batch_size,
            "patience": patience,
            "head_architecture": "Dense(256,relu) → Dropout(0.5) → Dense(10,softmax)",
        },
        "training_history": {
            "phase1": {
                k: [round(float(v), 6) for v in vals]
                for k, vals in hist_p1.history.items()
            },
            "phase2": {
                k: [round(float(v), 6) for v in vals]
                for k, vals in hist_p2.history.items()
            },
        },
    }

    class _NumpyEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=_NumpyEncoder)

    logger.info(f"\nMetrics → {metrics_save_path}")
    logger.info(f"Head    → {head_save_path}")
    logger.info("=" * 60)
    logger.info("Phase 5 — YAMNet COMPLETE")
    logger.info("=" * 60)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = train_yamnet()
    print(f"\nMacro F1 : {results['macro_f1']:.4f}")
    print(f"Accuracy : {results['accuracy']:.4f}")
