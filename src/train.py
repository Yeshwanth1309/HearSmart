"""
Phase 4 — Traditional ML Model Training.

Step 1: Load cached MFCC training features, encode labels, and validate.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

from src.utils import setup_logging, set_seed


def load_training_data(
    split_csv: str = "data/splits/train.csv",
    feature_dir: str = "features/mfcc/train",
    encoder_path: str = "models/label_encoder.pkl",
) -> tuple:
    """
    Load cached MFCC features and labels for the training split.

    Reads the training CSV, loads the corresponding .npy feature file
    for each row, stacks them into a feature matrix, encodes string
    labels with a LabelEncoder, and saves the encoder to disk.

    Args:
        split_csv: Path to the training split CSV.
        feature_dir: Directory containing cached MFCC .npy files.
        encoder_path: Path to save the fitted LabelEncoder.

    Returns:
        Tuple of (X, y_encoded, label_encoder) where:
            X: np.ndarray of shape (N, 80), dtype float32
            y_encoded: np.ndarray of shape (N,), dtype int
            label_encoder: fitted sklearn LabelEncoder
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(42)

    logger.info("=" * 60)
    logger.info("Phase 4 — Step 1: Loading MFCC Training Data")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load split CSV
    # ------------------------------------------------------------------
    df = pd.read_csv(split_csv)
    logger.info(f"Loaded split CSV: {split_csv} ({len(df)} rows)")

    # ------------------------------------------------------------------
    # 2. Load .npy feature files
    # ------------------------------------------------------------------
    feature_dir_path = Path(feature_dir)
    features_list = []
    labels_list = []
    missing_count = 0

    for idx, row in df.iterrows():
        stem = Path(row["slice_file_name"]).stem
        npy_path = feature_dir_path / f"{stem}.npy"

        if not npy_path.exists():
            logger.error(f"Missing feature file: {npy_path}")
            missing_count += 1
            continue

        feat = np.load(npy_path)
        features_list.append(feat)
        labels_list.append(row["class"])

    if missing_count > 0:
        raise RuntimeError(
            f"STOP: {missing_count} feature files are missing. "
            "Cannot proceed with incomplete data."
        )

    # ------------------------------------------------------------------
    # 3. Stack into arrays
    # ------------------------------------------------------------------
    X = np.stack(features_list, axis=0)  # (N, 80)
    y_labels = np.array(labels_list)     # (N,) string labels

    # ------------------------------------------------------------------
    # 4. Shape assertion
    # ------------------------------------------------------------------
    assert X.shape[1] == 80, (
        f"STOP: Feature dimension mismatch — expected 80, got {X.shape[1]}. "
        "Cached MFCC files may be corrupted."
    )
    assert X.shape[0] == len(df), (
        f"STOP: Row count mismatch — expected {len(df)}, got {X.shape[0]}."
    )

    # ------------------------------------------------------------------
    # 5. Encode labels
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    # ------------------------------------------------------------------
    # 6. Save encoder
    # ------------------------------------------------------------------
    encoder_dir = Path(encoder_path).parent
    encoder_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"Saved LabelEncoder to: {encoder_path}")

    # ------------------------------------------------------------------
    # 7. Print diagnostics
    # ------------------------------------------------------------------
    unique_classes = label_encoder.classes_
    num_classes = len(unique_classes)

    logger.info("")
    logger.info("--- Data Summary ---")
    logger.info(f"  X shape        : {X.shape}")
    logger.info(f"  y shape        : {y_encoded.shape}")
    logger.info(f"  X dtype        : {X.dtype}")
    logger.info(f"  Num classes    : {num_classes}")
    logger.info(f"  Missing files  : {missing_count}")
    logger.info("")
    logger.info("--- Class Distribution ---")

    for cls_name in unique_classes:
        count = int(np.sum(y_labels == cls_name))
        pct = count / len(y_labels) * 100
        logger.info(f"  {cls_name:20s} : {count:5d}  ({pct:.1f}%)")

    logger.info("")
    logger.info("Step 1 complete. Data is ready for training.")

    return X, y_encoded, label_encoder


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    X, y, encoder = load_training_data()
    print()
    print(f"X shape       : {X.shape}")
    print(f"y shape       : {y.shape}")
    print(f"Num classes   : {len(encoder.classes_)}")
    print(f"Classes       : {list(encoder.classes_)}")
