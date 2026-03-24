"""
Shared model definitions for v2 (8-class) architecture.
Import from here to avoid re-running training code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN_V2(nn.Module):
    """3-block CNN with BatchNorm and GlobalAvgPool for 8-class classification."""
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32); self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64); self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128); self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 256); self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))
