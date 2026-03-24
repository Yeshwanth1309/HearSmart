"""
Phase 5 — Deep Learning Models.

Contains:
    AudioCNN        — PyTorch CNN for mel-spectrogram classification (128x128x1)
    MelSpectrogramDataset — PyTorch Dataset for cached mel-spectrogram .npy files
    train_cnn       — Full training loop with early stopping, LR scheduling,
                      checkpointing, and training-curve logging
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils import setup_logging, set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 10
SEED = 42


# ======================================================================
# Dataset
# ======================================================================
class MelSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for cached mel-spectrogram .npy files.

    Reads a split CSV (train/val/test) and loads the corresponding
    cached mel-spectrogram from ``feature_dir/{stem}.npy``.

    Each sample is returned as:
        x : torch.Tensor, shape (1, 128, 128)   — channel-first
        y : int                                   — encoded class label
    """

    def __init__(
        self,
        split_csv: str,
        feature_dir: str,
        label_encoder=None,
        transform=None,
    ):
        """
        Parameters
        ----------
        split_csv : str
            Path to the split CSV with columns ``slice_file_name`` and ``class``.
        feature_dir : str
            Directory containing cached ``.npy`` mel-spectrogram files.
        label_encoder : sklearn.preprocessing.LabelEncoder or None
            Pre-fitted encoder. If None, a new one is fitted on the labels.
        transform : callable or None
            Optional transform applied to the numpy array before conversion.
        """
        self.logger = logging.getLogger(__name__)
        self.feature_dir = Path(feature_dir)
        self.transform = transform

        df = pd.read_csv(split_csv)
        self.logger.info(
            f"MelSpectrogramDataset: loaded {split_csv} ({len(df)} rows)"
        )

        # Build file list and labels
        self.samples = []
        self.labels = []
        missing = 0

        for _, row in df.iterrows():
            stem = Path(row["slice_file_name"]).stem
            npy_path = self.feature_dir / f"{stem}.npy"
            if not npy_path.exists():
                missing += 1
                continue
            self.samples.append(npy_path)
            self.labels.append(row["class"])

        if missing > 0:
            self.logger.warning(
                f"MelSpectrogramDataset: {missing} feature files missing "
                f"in {feature_dir}"
            )

        # Encode labels
        if label_encoder is not None:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(self.labels)
        else:
            from sklearn.preprocessing import LabelEncoder

            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        self.num_classes = len(self.label_encoder.classes_)
        self.logger.info(
            f"MelSpectrogramDataset: {len(self.samples)} samples, "
            f"{self.num_classes} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load (128, 128, 1) float32
        mel = np.load(self.samples[idx])

        if self.transform is not None:
            mel = self.transform(mel)

        # Convert (H, W, C) → (C, H, W) for PyTorch
        mel = mel.transpose(2, 0, 1)  # (1, 128, 128)
        x = torch.from_numpy(mel).float()
        y = int(self.encoded_labels[idx])
        return x, y


# ======================================================================
# CNN Architecture
# ======================================================================
class AudioCNN(nn.Module):
    """
    CNN for mel-spectrogram classification.

    Architecture per TRD §3.5.2:
        Conv2D stack → BatchNorm → ReLU → MaxPool → Dropout → FC

    Input:  (batch, 1, 128, 128)
    Output: (batch, 10) logits

    Design choices:
        - 4 convolutional blocks with increasing filter depth (32→64→128→256)
        - BatchNorm after each conv for training stability
        - MaxPool2d(2) after each block for spatial reduction
        - Dropout (0.3 conv, 0.5 FC) for regularisation
        - Global Average Pooling before FC to reduce parameters
        - L2 regularisation applied via optimizer weight_decay
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        # Block 1: 1 → 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # Block 2: 32 → 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # Block 3: 64 → 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # Block 4: 128 → 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        # Global Average Pooling → FC
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, 128, 128)

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        x = self.classifier(x)  # (batch, num_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (for ensemble compatibility)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ======================================================================
# Training loop
# ======================================================================
def train_cnn(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    feature_dir_train: str = "features/mel/train",
    feature_dir_val: str = "features/mel/val",
    feature_dir_test: str = "features/mel/test",
    model_save_path: str = "models/cnn_best.pt",
    metrics_save_path: str = "results/cnn_metrics.json",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = SEED,
) -> dict:
    """
    Train CNN with early stopping, LR scheduling, and checkpointing.

    Parameters
    ----------
    train_csv, val_csv, test_csv : str
        Paths to split CSVs.
    feature_dir_train, feature_dir_val, feature_dir_test : str
        Directories with cached mel-spectrogram .npy files.
    model_save_path : str
        Where to save the best model state_dict.
    metrics_save_path : str
        Where to export final metrics JSON.
    num_epochs : int
        Maximum training epochs (default: 50).
    batch_size : int
        Mini-batch size (default: 32).
    learning_rate : float
        Initial learning rate for Adam (default: 1e-3).
    weight_decay : float
        L2 regularisation (default: 1e-4).
    patience : int
        Early stopping patience (default: 10).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Training results including metrics and training curves.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("Phase 5 — CNN Training")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    logger.info("Loading mel-spectrogram datasets...")

    train_dataset = MelSpectrogramDataset(train_csv, feature_dir_train)
    label_encoder = train_dataset.label_encoder

    val_dataset = MelSpectrogramDataset(
        val_csv, feature_dir_val, label_encoder=label_encoder
    )
    test_dataset = MelSpectrogramDataset(
        test_csv, feature_dir_test, label_encoder=label_encoder
    )

    class_names = list(label_encoder.classes_)
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(
        f"Samples — Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 2. Initialize model, optimizer, scheduler, loss
    # ------------------------------------------------------------------
    model = AudioCNN(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    logger.info(f"\nTraining for up to {num_epochs} epochs (patience={patience})")
    logger.info("-" * 60)

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Compute validation macro F1
        from sklearn.metrics import f1_score as sklearn_f1

        val_f1 = sklearn_f1(all_val_labels, all_val_preds, average="macro")

        # Step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["val_f1"].append(round(val_f1, 6))
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # --- Checkpointing (best val loss) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(
                f"  → Saved best model (val_loss={val_loss:.4f}, "
                f"val_f1={val_f1:.4f})"
            )
        else:
            epochs_without_improvement += 1

        # --- Early stopping ---
        if epochs_without_improvement >= patience:
            logger.info(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    logger.info("-" * 60)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best val F1: {best_val_f1:.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("\nEvaluating on test set...")

    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_test_preds = []
    all_test_labels = []
    all_test_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(batch_y.numpy())
            all_test_probs.extend(probs.cpu().numpy())

    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)

    # Compute metrics
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score as sk_f1,
        precision_score,
        recall_score,
    )

    test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_macro_f1 = sk_f1(all_test_labels, all_test_preds, average="macro")
    test_weighted_f1 = sk_f1(all_test_labels, all_test_preds, average="weighted")
    test_precision = precision_score(
        all_test_labels, all_test_preds, average="macro"
    )
    test_recall = recall_score(
        all_test_labels, all_test_preds, average="macro"
    )
    per_class_f1 = sk_f1(all_test_labels, all_test_preds, average=None)
    cm = confusion_matrix(all_test_labels, all_test_preds)

    logger.info(f"\n--- Test Results ---")
    logger.info(f"  Accuracy:    {test_acc:.4f}")
    logger.info(f"  Macro F1:    {test_macro_f1:.4f}")
    logger.info(f"  Weighted F1: {test_weighted_f1:.4f}")
    logger.info(f"  Precision:   {test_precision:.4f}")
    logger.info(f"  Recall:      {test_recall:.4f}")

    for i, cls_name in enumerate(class_names):
        logger.info(f"    {cls_name:20s}: F1={per_class_f1[i]:.4f}")

    # ------------------------------------------------------------------
    # 5. Measure inference latency
    # ------------------------------------------------------------------
    logger.info("\nMeasuring inference latency...")

    latency_times = []
    sample_x, _ = test_dataset[0]
    sample_x = sample_x.unsqueeze(0).to(device)  # (1, 1, 128, 128)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(sample_x)

    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(sample_x)
        t1 = time.perf_counter()
        latency_times.append((t1 - t0) * 1000.0)

    avg_latency_ms = float(np.mean(latency_times))
    logger.info(f"  Avg inference latency: {avg_latency_ms:.2f} ms (CPU)")

    # ------------------------------------------------------------------
    # 6. Measure model size
    # ------------------------------------------------------------------
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    logger.info(f"  Model size: {model_size_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 7. Export metrics
    # ------------------------------------------------------------------
    metrics = {
        "model": "AudioCNN",
        "framework": "PyTorch",
        "accuracy": round(test_acc, 6),
        "macro_f1": round(test_macro_f1, 6),
        "weighted_f1": round(test_weighted_f1, 6),
        "precision_macro": round(test_precision, 6),
        "recall_macro": round(test_recall, 6),
        "per_class_f1": {
            class_names[i]: round(float(per_class_f1[i]), 6)
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "latency_ms": round(avg_latency_ms, 4),
        "model_size_mb": round(model_size_mb, 4),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_f1": round(best_val_f1, 6),
        "epochs_trained": len(history["train_loss"]),
        "training_history": history,
        "hyperparameters": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
            "criterion": "CrossEntropyLoss",
        },
    }

    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nMetrics exported → {metrics_save_path}")
    logger.info(f"Model saved → {model_save_path}")
    logger.info("=" * 60)
    logger.info("Phase 5 — CNN Training COMPLETE")
    logger.info("=" * 60)

    return metrics


# ======================================================================
# CLI entry point
# ======================================================================
if __name__ == "__main__":
    results = train_cnn()
    print(f"\nFinal Macro F1: {results['macro_f1']:.4f}")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
