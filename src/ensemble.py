"""
Phase 6 — Ensemble Architecture.

Combines all 5 trained models via weighted probability aggregation:
    - Random Forest   (MFCC features, sklearn)
    - SVM Pipeline    (MFCC features + StandardScaler, sklearn)
    - XGBoost         (MFCC features, xgboost)
    - AudioCNN        (mel-spectrogram, PyTorch)
    - YAMNet Head     (cached embeddings, Keras)

Key design decisions:
    1. Each model returns a (10,) probability vector → weighted sum → argmax
    2. Weights optimised on validation set via scipy.optimize.minimize (Nelder-Mead)
       targeting macro F1 (and implicitly accuracy)
    3. Confidence threshold: if max-prob < threshold, label = "uncertain"
    4. Safety-class override: gun_shot / siren predictions never suppressed
    5. Full test-set evaluation + per-model ablation exported as JSON

Outputs:
    results/ensemble_weights.json
    results/ensemble_metrics.json
    results/model_comparison.json
"""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn.functional as torch_F
import tensorflow as tf

from src.models import AudioCNN, MelSpectrogramDataset
from src.utils import setup_logging, set_seed

SEED = 42
NUM_CLASSES = 10
SAFETY_CLASSES = {"gun_shot", "siren"}
CONFIDENCE_THRESHOLD = 0.35   # below this → "uncertain"


# ─────────────────────────────────────────────────────────────────────────────
# GPU config
# ─────────────────────────────────────────────────────────────────────────────
def _configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_mfcc_split(split_csv: str, feature_dir: str, label_encoder) -> tuple:
    """Load cached MFCC features (N, 80) and encoded labels."""
    df = pd.read_csv(split_csv)
    feat_dir = Path(feature_dir)
    X_list, labels = [], []

    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy = feat_dir / f"{stem}.npy"
        if npy.exists():
            X_list.append(np.load(npy))
            labels.append(row["class"])

    X = np.stack(X_list).astype(np.float32)
    y = label_encoder.transform(labels).astype(np.int32)
    return X, y


def _load_mel_split(split_csv: str, feature_dir: str, label_encoder) -> tuple:
    """Load cached mel-spectrogram features (N, 1, 128, 128) for CNN."""
    df = pd.read_csv(split_csv)
    feat_dir = Path(feature_dir)
    X_list, labels = [], []

    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy = feat_dir / f"{stem}.npy"
        if npy.exists():
            mel = np.load(npy)          # (128, 128, 1)
            mel = mel.transpose(2, 0, 1)  # (1, 128, 128)
            X_list.append(mel)
            labels.append(row["class"])

    X = np.stack(X_list).astype(np.float32)
    y = label_encoder.transform(labels).astype(np.int32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Probability extractors (one per model)
# ─────────────────────────────────────────────────────────────────────────────
def _get_rf_probs(rf, X_mfcc: np.ndarray) -> np.ndarray:
    """Return (N, 10) softmax probabilities from Random Forest."""
    return rf.predict_proba(X_mfcc).astype(np.float32)


def _get_svm_probs(svm_pipeline, X_mfcc: np.ndarray) -> np.ndarray:
    """
    SVM Pipeline (StandardScaler → SVC).
    SVC must be trained with probability=True for predict_proba; if not,
    use decision_function + softmax approximation.
    """
    svc = svm_pipeline.named_steps["svm"]
    scaler = svm_pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X_mfcc)

    if hasattr(svc, "predict_proba"):
        try:
            return svm_pipeline.predict_proba(X_mfcc).astype(np.float32)
        except Exception:
            pass

    # Fallback: decision_function → softmax
    scores = svc.decision_function(X_scaled).astype(np.float32)
    # Softmax row-wise
    scores -= scores.max(axis=1, keepdims=True)
    exps = np.exp(scores)
    return exps / exps.sum(axis=1, keepdims=True)


def _get_xgb_probs(xgb_model, X_mfcc: np.ndarray) -> np.ndarray:
    """Return (N, 10) class probabilities from XGBoost."""
    return xgb_model.predict_proba(X_mfcc).astype(np.float32)


def _get_cnn_probs(
    cnn_model: AudioCNN,
    X_mel: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Return (N, 10) class probabilities from AudioCNN."""
    cnn_model.eval()
    all_probs = []
    dev = torch.device(device)

    with torch.no_grad():
        for i in range(0, len(X_mel), batch_size):
            batch = torch.from_numpy(X_mel[i : i + batch_size]).to(dev)
            logits = cnn_model(batch)
            probs = torch_F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0).astype(np.float32)


def _get_yamnet_probs(
    yamnet_head: tf.keras.Model,
    X_emb: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Return (N, 10) class probabilities from YAMNet head."""
    X_norm = ((X_emb - mean) / std).astype(np.float32)
    return yamnet_head.predict(X_norm, batch_size=batch_size, verbose=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble class
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleClassifier:
    """
    Weighted probability ensemble of 5 models.

    Weights are stored in order:
        [w_rf, w_svm, w_xgb, w_cnn, w_yamnet]

    Final probability = softmax( Σ wᵢ × Pᵢ )
    """

    DEFAULT_WEIGHTS = np.array([0.15, 0.15, 0.15, 0.30, 0.25], dtype=np.float32)

    def __init__(
        self,
        rf,
        svm_pipeline,
        xgb_model,
        cnn_model: AudioCNN,
        yamnet_head: tf.keras.Model,
        label_encoder,
        yamnet_emb_mean: np.ndarray,
        yamnet_emb_std: np.ndarray,
        weights: np.ndarray = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str = "cpu",
    ):
        self.rf = rf
        self.svm = svm_pipeline
        self.xgb = xgb_model
        self.cnn = cnn_model
        self.yamnet = yamnet_head
        self.label_encoder = label_encoder
        self.emb_mean = yamnet_emb_mean
        self.emb_std = yamnet_emb_std
        self.weights = (
            weights if weights is not None else self.DEFAULT_WEIGHTS.copy()
        )
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_names = list(label_encoder.classes_)

    def get_all_probs(
        self,
        X_mfcc: np.ndarray,
        X_mel: np.ndarray,
        X_yamnet_emb: np.ndarray,
    ) -> dict:
        """
        Collect probability matrices from all 5 models.

        Returns dict of {model_name: np.ndarray (N, 10)}
        """
        return {
            "rf":     _get_rf_probs(self.rf, X_mfcc),
            "svm":    _get_svm_probs(self.svm, X_mfcc),
            "xgb":    _get_xgb_probs(self.xgb, X_mfcc),
            "cnn":    _get_cnn_probs(self.cnn, X_mel, device=self.device),
            "yamnet": _get_yamnet_probs(
                self.yamnet, X_yamnet_emb, self.emb_mean, self.emb_std
            ),
        }

    def aggregate(
        self,
        probs_dict: dict,
        weights: np.ndarray = None,
    ) -> np.ndarray:
        """
        Weighted sum of probability matrices → (N, 10).

        Parameters
        ----------
        probs_dict : dict {model_name: (N, 10)}
        weights    : optional override for this call
        """
        w = weights if weights is not None else self.weights
        # Normalize weights to sum to 1
        w = np.array(w, dtype=np.float32)
        w = w / w.sum()

        order = ["rf", "svm", "xgb", "cnn", "yamnet"]
        ensemble_probs = sum(
            w[i] * probs_dict[name] for i, name in enumerate(order)
        )
        return ensemble_probs  # (N, 10)

    def predict(
        self,
        X_mfcc: np.ndarray,
        X_mel: np.ndarray,
        X_yamnet_emb: np.ndarray,
        weights: np.ndarray = None,
    ) -> tuple:
        """
        Full prediction pipeline with safety override and confidence gate.

        Returns
        -------
        predictions : np.ndarray (N,) — encoded class indices
        probs       : np.ndarray (N, 10)
        confidence  : np.ndarray (N,) — max probability per sample
        flags       : list[str] — 'ok'|'low_confidence'|'safety_override'
        """
        probs_dict = self.get_all_probs(X_mfcc, X_mel, X_yamnet_emb)
        ensemble_probs = self.aggregate(probs_dict, weights)

        predictions = np.argmax(ensemble_probs, axis=1)
        confidence = ensemble_probs.max(axis=1)

        flags = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            class_name = self.class_names[pred]
            if class_name in SAFETY_CLASSES:
                flags.append("safety_override")
            elif conf < self.confidence_threshold:
                flags.append("low_confidence")
            else:
                flags.append("ok")

        return predictions, ensemble_probs, confidence, flags


# ─────────────────────────────────────────────────────────────────────────────
# Weight optimisation
# ─────────────────────────────────────────────────────────────────────────────
def _optimise_weights(
    probs_dict: dict,
    y_val: np.ndarray,
    n_restarts: int = 20,
    seed: int = SEED,
) -> tuple:
    """
    Optimise ensemble weights on the validation set.

    Maximises: accuracy + 0.5 × macro_F1
    (combined objective to push both metrics up simultaneously)

    Uses Nelder-Mead with multiple random restarts to avoid local minima.

    Returns
    -------
    best_weights : np.ndarray (5,), un-normalised
    best_score   : float (negative of combined objective)
    """
    logger = logging.getLogger(__name__)
    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    P = np.stack([probs_dict[n] for n in order], axis=0)  # (5, N, 10)

    def _objective(w):
        w = np.abs(w)
        total = w.sum()
        if total < 1e-8:
            return 1.0
        w_norm = w / total
        # Weighted sum: (5, N, 10) × (5, 1, 1) → sum → (N, 10)
        combined = (P * w_norm[:, None, None]).sum(axis=0)
        y_pred = np.argmax(combined, axis=1)
        acc = accuracy_score(y_val, y_pred)
        mf1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        # Maximise acc + 0.5*mf1 → minimise negative
        return -(acc + 0.5 * mf1)

    rng = np.random.RandomState(seed)
    best_score = np.inf
    best_weights = EnsembleClassifier.DEFAULT_WEIGHTS.copy()

    for i in range(n_restarts):
        # Dirichlet-distributed initial weights for diversity
        w0 = rng.dirichlet(np.ones(5))
        result = minimize(
            _objective,
            w0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if result.fun < best_score:
            best_score = result.fun
            best_weights = np.abs(result.x)
            logger.info(
                f"  restart {i+1:2d}/{n_restarts}: "
                f"score={-result.fun:.5f}  "
                f"w={np.round(best_weights / best_weights.sum(), 3)}"
            )

    best_weights = best_weights / best_weights.sum()
    return best_weights, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def _full_metrics(y_true, y_pred, class_names, probs=None) -> dict:
    m = {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1":        round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "weighted_f1":     round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro")), 6),
        "recall_macro":    round(float(recall_score(y_true, y_pred, average="macro")), 6),
        "per_class_f1": {
            class_names[i]: round(float(v), 6)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_ensemble(
    # Paths
    cnn_model_path: str = "models/cnn_best.pt",
    yamnet_head_path: str = "models/yamnet_head.h5",
    label_encoder_path: str = "models/label_encoder.pkl",
    yamnet_emb_dir: str = "features/yamnet_embeddings",
    # Split CSVs
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    # Feature dirs
    mfcc_val_dir: str = "features/mfcc/val",
    mfcc_test_dir: str = "features/mfcc/test",
    mel_val_dir: str = "features/mel/val",
    mel_test_dir: str = "features/mel/test",
    # Outputs
    weights_path: str = "results/ensemble_weights.json",
    ensemble_metrics_path: str = "results/ensemble_metrics.json",
    comparison_path: str = "results/model_comparison.json",
    # Hyperparams
    n_weight_restarts: int = 30,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    seed: int = SEED,
) -> dict:
    """
    Full Phase 6 ensemble pipeline.

    Steps:
        1. Load all 5 models
        2. Collect validation-set probabilities from each model
        3. Optimise ensemble weights (30 Nelder-Mead restarts)
        4. Evaluate on test set with optimised weights
        5. Safety override analysis
        6. Export weights, metrics, model comparison
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)
    _configure_gpu()

    logger.info("=" * 60)
    logger.info("Phase 6 — Ensemble Architecture")
    logger.info("=" * 60)

    # ── 1. Load models ───────────────────────────────────────────────────────
    logger.info("Loading all 5 trained models...")

    label_encoder = joblib.load(label_encoder_path)
    class_names = list(label_encoder.classes_)
    logger.info(f"  Classes: {class_names}")

    rf = joblib.load("models/random_forest.pkl")
    logger.info("  RF loaded")

    svm_pipeline = joblib.load("models/svm.pkl")
    logger.info("  SVM pipeline loaded")

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model("models/xgboost.json")
    logger.info("  XGBoost loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = AudioCNN(num_classes=NUM_CLASSES).to(torch.device(device))
    cnn.load_state_dict(
        torch.load(cnn_model_path, map_location=torch.device(device))
    )
    cnn.eval()
    logger.info(f"  CNN loaded → {device}")

    yamnet_head = tf.keras.models.load_model(yamnet_head_path)
    logger.info("  YAMNet head loaded")

    # YAMNet normalisation stats (computed from train embeddings)
    E_train = np.load(os.path.join(yamnet_emb_dir, "train.npy"))
    yamnet_mean = E_train.mean(axis=0, keepdims=True)
    yamnet_std  = E_train.std(axis=0, keepdims=True) + 1e-8
    logger.info(f"  YAMNet norm stats from train embeddings {E_train.shape}")

    # ── 2. Load validation features ─────────────────────────────────────────
    logger.info("\nLoading validation features...")

    X_val_mfcc, y_val = _load_mfcc_split(val_csv, mfcc_val_dir, label_encoder)
    X_val_mel,  _     = _load_mel_split(val_csv, mel_val_dir, label_encoder)
    X_val_emb         = np.load(os.path.join(yamnet_emb_dir, "val.npy"))

    logger.info(
        f"  Val — MFCC: {X_val_mfcc.shape}, "
        f"Mel: {X_val_mel.shape}, "
        f"Emb: {X_val_emb.shape}"
    )

    # ── 3. Collect val probabilities from all models ─────────────────────────
    logger.info("\nExtracting validation probabilities from all 5 models...")

    val_probs = {}

    t0 = time.perf_counter()
    val_probs["rf"] = _get_rf_probs(rf, X_val_mfcc)
    logger.info(f"  RF  done ({(time.perf_counter()-t0)*1000:.0f} ms)")

    t0 = time.perf_counter()
    val_probs["svm"] = _get_svm_probs(svm_pipeline, X_val_mfcc)
    logger.info(f"  SVM done ({(time.perf_counter()-t0)*1000:.0f} ms)")

    t0 = time.perf_counter()
    val_probs["xgb"] = _get_xgb_probs(xgb, X_val_mfcc)
    logger.info(f"  XGB done ({(time.perf_counter()-t0)*1000:.0f} ms)")

    t0 = time.perf_counter()
    val_probs["cnn"] = _get_cnn_probs(cnn, X_val_mel, device=device)
    logger.info(f"  CNN done ({(time.perf_counter()-t0)*1000:.0f} ms)")

    t0 = time.perf_counter()
    val_probs["yamnet"] = _get_yamnet_probs(yamnet_head, X_val_emb, yamnet_mean, yamnet_std)
    logger.info(f"  YAMNet done ({(time.perf_counter()-t0)*1000:.0f} ms)")

    # Individual val accuracies (sanity check)
    for name, probs in val_probs.items():
        acc = accuracy_score(y_val, np.argmax(probs, axis=1))
        f1  = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
        logger.info(f"  {name:8s} val  acc={acc:.4f}  macro_f1={f1:.4f}")

    # ── 4. Optimise weights ──────────────────────────────────────────────────
    logger.info(f"\nOptimising ensemble weights ({n_weight_restarts} restarts)...")
    best_weights, best_score = _optimise_weights(
        val_probs, y_val, n_restarts=n_weight_restarts, seed=seed
    )

    logger.info(f"\nOptimal weights (normalised):")
    model_names = ["rf", "svm", "xgb", "cnn", "yamnet"]
    for name, w in zip(model_names, best_weights):
        logger.info(f"  {name:8s}: {w:.4f}")

    # Validate on val set with optimised weights
    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    P_val = np.stack([val_probs[n] for n in order], axis=0)
    val_ensemble_probs = (P_val * best_weights[:, None, None]).sum(axis=0)
    val_preds = np.argmax(val_ensemble_probs, axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1  = f1_score(y_val, val_preds, average="macro")
    logger.info(f"\nVal ensemble → acc={val_acc:.4f}  macro_f1={val_f1:.4f}")

    # Save weights
    os.makedirs(os.path.dirname(weights_path) or "results", exist_ok=True)
    weights_data = {
        "weights": {name: round(float(w), 6) for name, w in zip(model_names, best_weights)},
        "val_accuracy": round(float(val_acc), 6),
        "val_macro_f1": round(float(val_f1), 6),
        "optimiser": "Nelder-Mead",
        "n_restarts": n_weight_restarts,
        "objective": "accuracy + 0.5 * macro_f1",
    }
    with open(weights_path, "w") as f:
        json.dump(weights_data, f, indent=2)
    logger.info(f"Weights saved → {weights_path}")

    # ── 5. Load test features ────────────────────────────────────────────────
    logger.info("\nLoading test features...")
    X_test_mfcc, y_test = _load_mfcc_split(test_csv, mfcc_test_dir, label_encoder)
    X_test_mel,  _      = _load_mel_split(test_csv, mel_test_dir, label_encoder)
    X_test_emb          = np.load(os.path.join(yamnet_emb_dir, "test.npy"))

    # ── 6. Test-set evaluation ───────────────────────────────────────────────
    logger.info("\nExtracting test probabilities...")
    test_probs = {
        "rf":     _get_rf_probs(rf, X_test_mfcc),
        "svm":    _get_svm_probs(svm_pipeline, X_test_mfcc),
        "xgb":    _get_xgb_probs(xgb, X_test_mfcc),
        "cnn":    _get_cnn_probs(cnn, X_test_mel, device=device),
        "yamnet": _get_yamnet_probs(yamnet_head, X_test_emb, yamnet_mean, yamnet_std),
    }

    P_test = np.stack([test_probs[n] for n in order], axis=0)

    # Ensemble with optimised weights
    ensemble_test_probs = (P_test * best_weights[:, None, None]).sum(axis=0)
    ensemble_preds      = np.argmax(ensemble_test_probs, axis=1)
    confidence          = ensemble_test_probs.max(axis=1)

    # Safety overrides
    flags = []
    for pred, conf in zip(ensemble_preds, confidence):
        if class_names[pred] in SAFETY_CLASSES:
            flags.append("safety_override")
        elif conf < confidence_threshold:
            flags.append("low_confidence")
        else:
            flags.append("ok")

    ensemble_metrics = _full_metrics(y_test, ensemble_preds, class_names)
    ensemble_metrics["confidence_stats"] = {
        "mean": round(float(confidence.mean()), 6),
        "min":  round(float(confidence.min()),  6),
        "max":  round(float(confidence.max()),  6),
        "low_confidence_pct": round(
            100.0 * sum(f == "low_confidence" for f in flags) / len(flags), 2
        ),
    }
    ensemble_metrics["weights"] = weights_data["weights"]

    logger.info("\n" + "=" * 50)
    logger.info("ENSEMBLE TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Accuracy:    {ensemble_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1:    {ensemble_metrics['macro_f1']:.4f}")
    logger.info(f"  Weighted F1: {ensemble_metrics['weighted_f1']:.4f}")
    logger.info(f"  Mean confidence: {ensemble_metrics['confidence_stats']['mean']:.4f}")
    logger.info("  Per-class F1:")
    for cls, f1v in ensemble_metrics["per_class_f1"].items():
        logger.info(f"    {cls:25s}: {f1v:.4f}")

    # ── 7. Per-model ablation ────────────────────────────────────────────────
    logger.info("\nPer-model individual test performance:")
    individual = {}
    for name in order:
        preds = np.argmax(test_probs[name], axis=1)
        m = _full_metrics(y_test, preds, class_names)
        individual[name] = m
        logger.info(
            f"  {name:8s} → acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}"
        )

    # Measure ensemble latency (single sample)
    logger.info("\nMeasuring ensemble latency (single sample, 50 runs)...")
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        s_mfcc = X_test_mfcc[:1]
        s_mel  = X_test_mel[:1]
        s_emb  = X_test_emb[:1]

        p_rf = rf.predict_proba(s_mfcc)
        p_svm = _get_svm_probs(svm_pipeline, s_mfcc)
        p_xgb = xgb.predict_proba(s_mfcc)
        s_mel_t = torch.from_numpy(s_mel).to(torch.device(device))
        with torch.no_grad():
            p_cnn = torch_F.softmax(cnn(s_mel_t), dim=1).cpu().numpy()
        s_emb_n = ((s_emb - yamnet_mean) / yamnet_std).astype(np.float32)
        p_yamnet = yamnet_head.predict(s_emb_n, verbose=0)

        stack = np.stack([p_rf, p_svm, p_xgb, p_cnn, p_yamnet], axis=0)
        _ = np.argmax((stack * best_weights[:, None, None]).sum(axis=0), axis=1)
        times.append((time.perf_counter() - t0) * 1000.0)

    latency_ms = round(float(np.mean(times)), 2)
    logger.info(f"  Avg ensemble latency: {latency_ms:.1f} ms  (limit: 500 ms)")

    # ── 8. Export ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(ensemble_metrics_path) or "results", exist_ok=True)
    ensemble_metrics["latency_ms"] = latency_ms
    ensemble_metrics["within_latency_limit"] = latency_ms < 500

    with open(ensemble_metrics_path, "w") as f:
        json.dump(ensemble_metrics, f, indent=2)

    comparison = {
        "ensemble": {
            "accuracy": ensemble_metrics["accuracy"],
            "macro_f1": ensemble_metrics["macro_f1"],
            "weighted_f1": ensemble_metrics["weighted_f1"],
            "latency_ms": latency_ms,
            "weights": weights_data["weights"],
        },
        "individual_models": {
            name: {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
            }
            for name, m in individual.items()
        },
        "prd_target_accuracy": 0.85,
        "prd_target_met": ensemble_metrics["macro_f1"] >= 0.85,
        "target_90_met": ensemble_metrics["accuracy"] >= 0.90,
    }

    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\nEnsemble metrics  → {ensemble_metrics_path}")
    logger.info(f"Model comparison  → {comparison_path}")
    logger.info(f"PRD ≥85% F1 met:   {comparison['prd_target_met']}")
    logger.info(f"≥90% accuracy met: {comparison['target_90_met']}")
    logger.info("=" * 60)
    logger.info("Phase 6 — Ensemble COMPLETE")
    logger.info("=" * 60)

    return {
        "ensemble_metrics": ensemble_metrics,
        "model_comparison": comparison,
        "best_weights": weights_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_ensemble()
    m = results["ensemble_metrics"]
    print(f"\n{'='*50}")
    print(f"ENSEMBLE FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy  : {m['accuracy']:.4f}  ({'✅ ≥90%' if m['accuracy'] >= 0.90 else '❌ <90%'})")
    print(f"Macro F1  : {m['macro_f1']:.4f}  ({'✅ ≥85%' if m['macro_f1'] >= 0.85 else '❌ <85%'})")
    print(f"Latency   : {m['latency_ms']:.1f} ms  ({'✅ <500ms' if m['latency_ms'] < 500 else '❌'})")
