"""
Train the Safety Binary Classifier (V3 Feature 2).

Uses existing YAMNet embeddings from features/yamnet_v2/ to train
a lightweight binary classifier: Safety (siren/horn) vs Non-Safety.

Output: models/safety_classifier.pt
"""
import os, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data.label_map_v2 import CLASS_NAMES_V2
from src.models.safety_model import SafetyClassifier, SAFETY_POSITIVE_CLASSES
from src.utils import setup_logging, set_seed

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SAFETY_CLASS_WEIGHT = 3.0


def load_data(split: str):
    """Load YAMNet embeddings and create binary safety labels."""
    import pandas as pd

    df = pd.read_csv(f"data/splits_v2/{split}.csv")
    labels_8class = df["class_v2"].values

    # Create binary labels: 1 = safety (siren/horn), 0 = non-safety
    binary_labels = np.array([
        1 if c in SAFETY_POSITIVE_CLASSES else 0
        for c in labels_8class
    ], dtype=np.float32)

    # Load embeddings — try existing yamnet_v2 arrays
    emb_path = Path(f"features/yamnet_v2/{split}_X.npy")
    if emb_path.exists():
        embeddings = np.load(str(emb_path)).astype(np.float32)
    else:
        # Fallback: extract on the fly
        logger.warning(f"Embeddings not found at {emb_path}. "
                      "Please run feature extraction first.")
        raise FileNotFoundError(f"Missing {emb_path}")

    # Normalize with mu/sig
    mu = np.load("features/yamnet_v2/mu.npy")
    sig = np.load("features/yamnet_v2/sig.npy")
    embeddings = ((embeddings - mu) / sig).astype(np.float32)

    logger.info(f"  {split}: {len(embeddings)} samples, "
               f"{int(binary_labels.sum())} safety, "
               f"{int(len(binary_labels) - binary_labels.sum())} non-safety")

    return embeddings, binary_labels


def train():
    print("=" * 60)
    print("  Training Safety Binary Classifier (V3)")
    print("=" * 60)

    # Load data
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")

    # Create datasets
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = SafetyClassifier(input_dim=1024).to(DEVICE)
    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Device: {DEVICE}")

    # Weighted loss (safety = 3.0x)
    n_safety = int(y_train.sum())
    n_non = len(y_train) - n_safety
    pos_weight = torch.tensor([SAFETY_CLASS_WEIGHT * n_non / max(n_safety, 1)],
                              dtype=torch.float32).to(DEVICE)
    # Cap weight to avoid instability
    pos_weight = torch.clamp(pos_weight, max=10.0)
    print(f"  Pos weight: {pos_weight.item():.2f}")

    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_acc = 0
    best_state = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss_per_sample = criterion(pred, yb)
            # Apply class weighting
            weights = torch.where(yb == 1, pos_weight, torch.ones_like(yb))
            loss = (loss_per_sample * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = (model(xb) >= 0.5).float()
                correct += (pred == yb).sum().item()
                total += yb.shape[0]
        val_acc = correct / total
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
                  f"Loss: {total_loss/len(train_loader):.4f}  "
                  f"Val Acc: {val_acc:.4f}  "
                  f"Best: {best_val_acc:.4f}")

    # Save best
    if best_state:
        model.load_state_dict(best_state)
    out_path = "models/safety_classifier.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\n  ✅ Saved → {out_path} (Val Acc: {best_val_acc:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    train()
