"""
Phase 5b — Train Unified CNN on 8-class (US8K + ESC-50 combined).
Output: models/cnn_v2.pt
"""
import os, sys, logging
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
from collections import Counter

from src.data.label_map_v2 import CLASS_NAMES_V2, label_to_index, ESC50_TO_8CLASS
from src.utils import setup_logging, set_seed

setup_logging(); set_seed(42)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 8

class AudioCNN_V2(nn.Module):
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

class US8K_MelDataset(Dataset):
    def __init__(self, split):
        self.df = pd.read_csv(f"data/splits_v2/{split}.csv")
        self.split = split
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        stem = Path(row["slice_file_name"]).stem
        mel = np.load(f"features/mel_v2/{self.split}/{stem}.npy")
        mel_t = torch.from_numpy(mel.transpose(2,0,1).astype(np.float32))
        return mel_t, torch.tensor(label_to_index(row["class_v2"]), dtype=torch.long)

class ESC50_MelDataset_V2(Dataset):
    """ESC-50 dataset mapped to 8-class schema."""
    def __init__(self, df, audio_dir, augment=False):
        import librosa
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.augment = augment
        self.sr = 22050; self.dur = 3; self.n_mels = 128
        self.samples = self.sr * self.dur

    def _load(self, path):
        import librosa
        wav, _ = librosa.load(path, sr=self.sr, duration=self.dur+0.1)
        if len(wav) >= self.samples: wav = wav[:self.samples]
        else: wav = np.pad(wav, (0, self.samples - len(wav)))
        if self.augment:
            if np.random.rand() < 0.4:
                start = np.random.randint(0, max(1, len(wav)//4))
                wav = wav[start:]; wav = np.pad(wav, (0, self.samples - len(wav)))
            wav = wav * (0.8 + 0.4 * np.random.rand())
        wav = wav / (np.abs(wav).max() + 1e-8)
        return wav

    def _to_mel(self, wav):
        import librosa
        hop = self.samples // 127
        mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=128, n_fft=2048, hop_length=hop, center=False)
        if mel.shape[1] < 128: mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))
        mel = mel[:, :128]
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return torch.from_numpy(mel_db[None])

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = self._load(os.path.join(self.audio_dir, row["filename"]))
        mel = self._to_mel(wav)
        lbl = label_to_index(ESC50_TO_8CLASS.get(row["category"], "background_noise"))
        return mel, torch.tensor(lbl, dtype=torch.long)

from torch.utils.data import ConcatDataset
from src.data.esc50_loader import load_esc50_metadata

ESC_AUDIO = "data/ESC50/audio/"; ESC_META = "data/ESC50/meta/esc50.csv"
esc_train_df, esc_val_df = load_esc50_metadata(ESC_META)

us8k_tr = US8K_MelDataset("train"); us8k_va = US8K_MelDataset("val")
esc_tr  = ESC50_MelDataset_V2(esc_train_df, ESC_AUDIO, augment=True)
esc_va  = ESC50_MelDataset_V2(esc_val_df,   ESC_AUDIO, augment=False)

train_ds = ConcatDataset([us8k_tr, esc_tr])
val_ds   = ConcatDataset([us8k_va, esc_va])

# Class weights from US8K split (primary)
us8k_df = pd.read_csv("data/splits_v2/train.csv")
cnt = Counter(us8k_df["class_v2"].tolist())
total = sum(cnt.values())
cw = [total / (8 * cnt.get(c, 1)) for c in CLASS_NAMES_V2]
weights_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

# WeightedRandomSampler
sample_weights = []
for i in range(len(us8k_tr)):
    _, y = us8k_tr[i]
    sample_weights.append(cw[y.item()])
for i in range(len(esc_tr)):
    _, y = esc_tr[i]
    sample_weights.append(cw[y.item()])
sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.float32), len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,  num_workers=0)

model = AudioCNN_V2(N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_t)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

best_acc = 0.0
print(f"\n  Training Unified CNN v2 on {DEVICE} (60 epochs) ...")
for epoch in range(1, 61):
    model.train(); tot = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward(); optimizer.step(); tot += loss.item()
    scheduler.step()

    model.eval(); preds, acts = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            acts.extend(y.numpy())
    acc = accuracy_score(acts, preds)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/cnn_v2.pt")
    if epoch % 10 == 0:
        print(f"  Ep{epoch:3d} | loss={tot/len(train_loader):.4f} | val_acc={acc:.4f} (best={best_acc:.4f})")

# Final test evaluation
te_ds = US8K_MelDataset("test")
te_loader = DataLoader(te_ds, batch_size=32, shuffle=False)
model.load_state_dict(torch.load("models/cnn_v2.pt", map_location=DEVICE))
model.eval(); preds, acts = [], []
with torch.no_grad():
    for x, y in te_loader:
        preds.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
        acts.extend(y.numpy())
acc = accuracy_score(acts, preds); f1 = f1_score(acts, preds, average="macro", zero_division=0)
print(f"\n  CNN v2 Test: acc={acc:.4f}  macro-F1={f1:.4f}")
print(classification_report(acts, preds, target_names=CLASS_NAMES_V2, zero_division=0))
print("  Phase 5b complete.")
