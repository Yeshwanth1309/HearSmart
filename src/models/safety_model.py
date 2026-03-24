"""
Safety Binary Classifier — Feature 2 (V3).

A lightweight binary classifier that operates on YAMNet 1024-dim embeddings
to determine if the current audio environment is safety-critical
(Siren or Horn) with minimal latency.

Architecture:
    YAMNet embedding (1024-dim)
    → Dense(256) → ReLU → Dropout(0.3)
    → Dense(64) → ReLU
    → Dense(1) → Sigmoid

Training labels:
    Siren → 1 (Safety)
    Horn  → 1 (Safety)
    All other 6 classes → 0 (Non-Safety)

Class weight: Safety = 3.0 (minority oversampling)
"""
import torch
import torch.nn as nn


class SafetyClassifier(nn.Module):
    """Binary classifier: Safety (siren/horn) vs Non-Safety."""

    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1024) YAMNet embedding vector
        Returns:
            (batch, 1) probability of Safety class
        """
        return self.net(x)

    def predict(self, embedding: "np.ndarray", threshold: float = 0.5) -> bool:
        """
        Convenience method for single-sample inference.

        Args:
            embedding: (1024,) or (1, 1024) numpy array
            threshold: decision boundary (default 0.5)

        Returns:
            True if Safety, False otherwise
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            x = torch.from_numpy(embedding.astype(np.float32))
            prob = self.net(x).item()
        return prob >= threshold

    def predict_proba(self, embedding: "np.ndarray") -> float:
        """Return raw safety probability."""
        import numpy as np
        self.eval()
        with torch.no_grad():
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            x = torch.from_numpy(embedding.astype(np.float32))
            return self.net(x).item()


# ─── Safety label mapping ────────────────────────────────────────────────────
SAFETY_POSITIVE_CLASSES = {"siren", "horn"}


def make_safety_labels(class_names: list) -> dict:
    """Return mapping: class_name → 1 (safety) or 0 (non-safety)."""
    return {c: 1 if c in SAFETY_POSITIVE_CLASSES else 0 for c in class_names}
