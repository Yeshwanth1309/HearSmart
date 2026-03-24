"""
ESC-50 Dataset Loader — Phase 11 (Full Dataset).

Maps ALL 50 ESC-50 categories to the 10 UrbanSound8K labels using acoustic similarity.
Audio files are 5-second WAV clips at 44.1 kHz.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ─── Full ESC-50 → US8K label mapping ────────────────────────────────────────
# Maps all 50 ESC-50 categories to the 10 US8K classes using acoustic similarity.
ESC50_TO_US8K_MAP = {
    # Animals
    "dog":              "dog_bark",
    "rooster":          "dog_bark",       # animal call -> dog_bark
    "pig":              "dog_bark",
    "cow":              "dog_bark",
    "frog":             "dog_bark",
    "cat":              "dog_bark",
    "hen":              "dog_bark",
    "insects":          "dog_bark",
    "sheep":            "dog_bark",
    "crow":             "dog_bark",

    # Nature & Weather
    "rain":             "air_conditioner",   # steady background
    "sea_waves":        "air_conditioner",
    "crackling_fire":   "air_conditioner",
    "crickets":         "air_conditioner",
    "chirping_birds":   "street_music",      # pleasant outdoor ambient
    "water_drops":      "air_conditioner",
    "wind":             "air_conditioner",
    "pouring_water":    "air_conditioner",
    "toilet_flush":     "air_conditioner",
    "thunderstorm":     "jackhammer",        # impulsive loud

    # Human non-speech
    "crying_baby":      "children_playing",
    "sneezing":         "children_playing",
    "clapping":         "children_playing",
    "breathing":        "children_playing",
    "coughing":         "children_playing",
    "footsteps":        "street_music",
    "laughing":         "children_playing",
    "brushing_teeth":   "air_conditioner",
    "snoring":          "air_conditioner",
    "drinking_sipping": "air_conditioner",

    # Interior/Domestic
    "door_wood_knock":  "street_music",
    "mouse_click":      "air_conditioner",
    "keyboard_typing":  "air_conditioner",
    "door_wood_creaks": "air_conditioner",
    "can_opening":      "air_conditioner",
    "washing_machine":  "engine_idling",
    "vacuum_cleaner":   "engine_idling",
    "clock_alarm":      "car_horn",          # alert → car_horn
    "clock_tick":       "air_conditioner",
    "glass_breaking":   "gun_shot",          # sharp impulsive → gun_shot

    # Exterior / Urban
    "helicopter":       "engine_idling",
    "chainsaw":         "drilling",
    "siren":            "siren",
    "car_horn":         "car_horn",
    "engine":           "engine_idling",
    "train":            "engine_idling",
    "church_bells":     "street_music",
    "airplane":         "engine_idling",
    "fireworks":        "gun_shot",          # impulsive → gun_shot
    "hand_saw":         "drilling",          # mechanical cutting → drilling
}

# UrbanSound8K class list (fixed order, indices 0-9)
US8K_CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]

# ─── Dataset class ────────────────────────────────────────────────────────────
class ESC50Dataset(Dataset):
    """PyTorch Dataset for all ESC-50 samples mapped to US8K labels."""

    def __init__(self, df, audio_dir, target_sr=22050, duration=3.0, n_mels=128,
                 augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        self.target_samples = int(target_sr * duration)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_and_preprocess(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.target_sr, mono=True)

        # Pad/Trim to 3 s
        if len(audio) > self.target_samples:
            # random crop during training for augmentation
            if self.augment:
                start = np.random.randint(0, len(audio) - self.target_samples)
                audio = audio[start: start + self.target_samples]
            else:
                audio = audio[:self.target_samples]
        else:
            audio = np.pad(audio, (0, self.target_samples - len(audio)))

        # Amplitude normalize
        max_val = np.abs(audio).max()
        if max_val > 1e-6:
            audio = audio / max_val

        # Optional augmentation: time shift + small gain jitter
        if self.augment:
            shift = np.random.randint(-1000, 1000)
            audio = np.roll(audio, shift)
            audio *= np.random.uniform(0.85, 1.15)
            audio = np.clip(audio, -1.0, 1.0)

        return audio

    def _audio_to_mel(self, audio):
        hop_length = self.target_samples // (128 - 1)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_mels=self.n_mels,
            n_fft=2048, hop_length=hop_length, center=False,
        )
        if mel.shape[1] < 128:
            mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])), mode='constant')
        mel = mel[:, :128]  # ensure exactly 128 frames
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return np.expand_dims(mel_db, 0)  # (1, 128, 128)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self.audio_dir / row["filename"]
        audio = self._load_and_preprocess(file_path)
        mel = self._audio_to_mel(audio)
        label = US8K_CLASSES.index(ESC50_TO_US8K_MAP[row["category"]])
        return torch.from_numpy(mel), torch.tensor(label, dtype=torch.long)


# ─── Loader helper ────────────────────────────────────────────────────────────
def load_esc50_metadata(csv_path):
    """
    Load ESC-50 CSV and split by standard 5-fold.
    Returns train_df (folds 1-4), test_df (fold 5).
    All 50 classes mapped via ESC50_TO_US8K_MAP.
    """
    df = pd.read_csv(csv_path)
    # All 50 classes are in the map
    df = df[df["category"].isin(ESC50_TO_US8K_MAP.keys())].copy()
    df["us8k_label"] = df["category"].map(ESC50_TO_US8K_MAP)

    train_df = df[df["fold"] != 5].reset_index(drop=True)
    test_df  = df[df["fold"] == 5].reset_index(drop=True)

    logger.info(f"ESC-50 total: {len(df)} | train: {len(train_df)} | test: {len(test_df)}")
    return train_df, test_df


def class_weights_from_df(df):
    """Compute inverse-frequency class weights for balanced training."""
    counts = df["us8k_label"].value_counts()
    total  = len(df)
    weights = np.zeros(len(US8K_CLASSES), dtype=np.float32)
    for i, cls in enumerate(US8K_CLASSES):
        n = counts.get(cls, 1)
        weights[i] = total / (len(US8K_CLASSES) * n)
    return weights


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    train_df, test_df = load_esc50_metadata("data/ESC50/meta/esc50.csv")
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print("\nTrain label distribution:")
    print(train_df["us8k_label"].value_counts().to_string())
    w = class_weights_from_df(train_df)
    print("\nClass weights:", dict(zip(US8K_CLASSES, w.tolist())))
