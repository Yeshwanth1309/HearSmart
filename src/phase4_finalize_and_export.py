"""
Phase 4 — Finalize and Export.

Re-trains Random Forest, SVM, and XGBoost with their best tuned
hyperparameters, evaluates on the held-out test set, measures inference
latency and model sizes, then exports every metric to a single JSON file.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work in Colab
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import setup_logging, set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# Random seeds (numpy + sklearn share the same numpy seed)
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
set_seed(SEED)


# ======================================================================
# Helper — load a split (train / val / test) from CSV + .npy features
# ======================================================================
def _load_split(
    split_csv: str,
    feature_dir: str,
) -> tuple:
    """Return (X, labels) for a given split.

    Parameters
    ----------
    split_csv : str
        Path to the CSV that lists ``slice_file_name`` and ``class``.
    feature_dir : str
        Directory containing the cached ``.npy`` MFCC feature files.

    Returns
    -------
    X : np.ndarray, shape (N, 80)
    labels : np.ndarray of str, shape (N,)
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(split_csv)
    logger.info(f"Loaded {split_csv} ({len(df)} rows)")

    feature_dir_path = Path(feature_dir)
    features, labels = [], []
    missing = 0

    for _, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy_path = feature_dir_path / f"{stem}.npy"
        if not npy_path.exists():
            logger.error(f"Missing feature file: {npy_path}")
            missing += 1
            continue
        features.append(np.load(npy_path))
        labels.append(row["class"])

    if missing:
        raise RuntimeError(
            f"STOP: {missing} feature files missing in {feature_dir}. "
            "Cannot proceed with incomplete data."
        )

    X = np.stack(features, axis=0)
    assert X.shape[1] == 80, (
        f"Feature dimension mismatch — expected 80, got {X.shape[1]}"
    )
    return X, np.array(labels)


# ======================================================================
# Helper — evaluate a trained model on the test set
# ======================================================================
def _evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    *,
    preprocess_fn=None,
) -> dict:
    """Compute macro/weighted F1, per-class F1, and confusion matrix.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : arrays
    class_names : list[str]
        Human-readable label for each encoded class index.
    preprocess_fn : callable or None
        Optional transform applied to X_test before prediction.
    """
    X = preprocess_fn(X_test) if preprocess_fn else X_test
    y_pred = model.predict(X)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    per_class_f1_arr = f1_score(y_test, y_pred, average=None)
    per_class_f1 = {
        class_names[i]: float(per_class_f1_arr[i])
        for i in range(len(class_names))
    }

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "per_class_f1": {k: round(v, 6) for k, v in per_class_f1.items()},
        "confusion_matrix": cm,
    }


# ======================================================================
# Helper — measure inference latency (100 single-sample predictions)
# ======================================================================
def _measure_latency(
    model,
    X_test: np.ndarray,
    *,
    n_runs: int = 100,
    preprocess_fn=None,
) -> float:
    """Return average single-sample latency in **milliseconds**."""
    rng = np.random.RandomState(SEED)
    indices = rng.randint(0, len(X_test), size=n_runs)

    times = []
    for idx in indices:
        sample = X_test[idx : idx + 1]
        if preprocess_fn:
            sample = preprocess_fn(sample)
        t0 = time.perf_counter()
        model.predict(sample)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = float(np.mean(times)) * 1000.0
    return round(avg_ms, 4)


# ======================================================================
# Helper — compute the serialised model size in MB
# ======================================================================
def _model_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return round(os.path.getsize(path) / (1024 * 1024), 4)


# ======================================================================
# Main pipeline
# ======================================================================
def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Phase 4 — Finalize and Export")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load all three splits
    # ------------------------------------------------------------------
    logger.info("Loading MFCC features for train / val / test …")

    X_train, y_train_labels = _load_split(
        split_csv="data/splits/train.csv",
        feature_dir="features/mfcc/train",
    )
    X_val, y_val_labels = _load_split(
        split_csv="data/splits/val.csv",
        feature_dir="features/mfcc/val",
    )
    X_test, y_test_labels = _load_split(
        split_csv="data/splits/test.csv",
        feature_dir="features/mfcc/test",
    )

    # Combine train + val for final re-training
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full_labels = np.concatenate([y_train_labels, y_val_labels], axis=0)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_full = label_encoder.fit_transform(y_train_full_labels)
    y_test = label_encoder.transform(y_test_labels)

    class_names = list(label_encoder.classes_)
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(
        f"Train+Val: {X_train_full.shape[0]}, Test: {X_test.shape[0]}"
    )

    # ------------------------------------------------------------------
    # 2. Ensure output directories exist
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Train & evaluate — Random Forest
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Training Random Forest (tuned) …")

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_full, y_train_full)

    rf_path = "models/random_forest.pkl"
    joblib.dump(rf, rf_path)
    logger.info(f"Saved Random Forest → {rf_path}")

    rf_metrics = _evaluate(rf, X_test, y_test, class_names)
    rf_metrics["latency_ms"] = _measure_latency(rf, X_test)
    rf_metrics["model_size_mb"] = _model_size_mb(rf_path)

    logger.info(f"  Macro F1:  {rf_metrics['macro_f1']}")
    logger.info(f"  Weighted F1: {rf_metrics['weighted_f1']}")
    logger.info(f"  Latency:   {rf_metrics['latency_ms']} ms")
    logger.info(f"  Size:      {rf_metrics['model_size_mb']} MB")

    # ------------------------------------------------------------------
    # 4. Train & evaluate — SVM (with StandardScaler)
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Training SVM (tuned, with StandardScaler) …")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)

    svm = SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        random_state=SEED,
    )
    svm.fit(X_train_scaled, y_train_full)

    svm_model_path = "models/svm_model.pkl"
    svm_scaler_path = "models/svm_scaler.pkl"
    joblib.dump(svm, svm_model_path)
    joblib.dump(scaler, svm_scaler_path)
    logger.info(f"Saved SVM model  → {svm_model_path}")
    logger.info(f"Saved SVM scaler → {svm_scaler_path}")

    svm_metrics = _evaluate(
        svm,
        X_test,
        y_test,
        class_names,
        preprocess_fn=scaler.transform,
    )
    svm_metrics["latency_ms"] = _measure_latency(
        svm, X_test, preprocess_fn=scaler.transform,
    )
    svm_metrics["model_size_mb"] = round(
        _model_size_mb(svm_model_path) + _model_size_mb(svm_scaler_path), 4
    )

    logger.info(f"  Macro F1:  {svm_metrics['macro_f1']}")
    logger.info(f"  Weighted F1: {svm_metrics['weighted_f1']}")
    logger.info(f"  Latency:   {svm_metrics['latency_ms']} ms")
    logger.info(f"  Size:      {svm_metrics['model_size_mb']} MB")

    # ------------------------------------------------------------------
    # 5. Train & evaluate — XGBoost
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Training XGBoost (tuned) …")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(class_names),
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=2,
        tree_method="hist",
    )
    xgb.fit(X_train_full, y_train_full)

    xgb_path = "models/xgboost.pkl"
    joblib.dump(xgb, xgb_path)
    logger.info(f"Saved XGBoost → {xgb_path}")

    xgb_metrics = _evaluate(xgb, X_test, y_test, class_names)
    xgb_metrics["latency_ms"] = _measure_latency(xgb, X_test)
    xgb_metrics["model_size_mb"] = _model_size_mb(xgb_path)

    logger.info(f"  Macro F1:  {xgb_metrics['macro_f1']}")
    logger.info(f"  Weighted F1: {xgb_metrics['weighted_f1']}")
    logger.info(f"  Latency:   {xgb_metrics['latency_ms']} ms")
    logger.info(f"  Size:      {xgb_metrics['model_size_mb']} MB")

    # ------------------------------------------------------------------
    # 6. Export all metrics to JSON
    # ------------------------------------------------------------------
    # Remove confusion_matrix from the top-level entries (kept per the
    # user's JSON structure specification — added as extra key if needed)
    output = {
        "random_forest": {
            "macro_f1": rf_metrics["macro_f1"],
            "weighted_f1": rf_metrics["weighted_f1"],
            "per_class_f1": rf_metrics["per_class_f1"],
            "confusion_matrix": rf_metrics["confusion_matrix"],
            "latency_ms": rf_metrics["latency_ms"],
            "model_size_mb": rf_metrics["model_size_mb"],
        },
        "svm": {
            "macro_f1": svm_metrics["macro_f1"],
            "weighted_f1": svm_metrics["weighted_f1"],
            "per_class_f1": svm_metrics["per_class_f1"],
            "confusion_matrix": svm_metrics["confusion_matrix"],
            "latency_ms": svm_metrics["latency_ms"],
            "model_size_mb": svm_metrics["model_size_mb"],
        },
        "xgboost": {
            "macro_f1": xgb_metrics["macro_f1"],
            "weighted_f1": xgb_metrics["weighted_f1"],
            "per_class_f1": xgb_metrics["per_class_f1"],
            "confusion_matrix": xgb_metrics["confusion_matrix"],
            "latency_ms": xgb_metrics["latency_ms"],
            "model_size_mb": xgb_metrics["model_size_mb"],
        },
    }

    json_path = "results/traditional_ml_metrics.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("-" * 60)
    logger.info(f"Metrics exported → {json_path}")

    print("\nPhase 4 artifacts exported successfully")


# ======================================================================
# CLI entry point
# ======================================================================
if __name__ == "__main__":
    main()
