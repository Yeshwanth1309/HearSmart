"""
Feature extraction module for the hearing aid research project.

Provides extractors for three feature types:
- **MFCC**: 40 coefficients aggregated to (80,) via mean+std — for traditional ML.
- **Mel-spectrogram**: (128, 128, 1) — for CNN models.
- **Raw waveform**: 1-D float32 array (48 000 samples @ 16 kHz × 3 s) — for YAMNet.

Also provides:
- Audio augmentation (time stretch, pitch shift, Gaussian noise, gain, time shift).
- SpecAugment (time and frequency masking on mel-spectrograms).
- Feature caching to disk.

All operations are deterministic when seeded.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import pandas as pd

from src.utils import setup_logging, set_seed
from src.data.preprocessor import AudioPreprocessor


# ======================================================================
# Module-level preprocessor singleton (lazy-initialized)
# ======================================================================

_preprocessor: AudioPreprocessor | None = None


def _get_preprocessor(**kwargs) -> AudioPreprocessor:
    """Return (and optionally create) the module-level AudioPreprocessor."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor(**kwargs)
    return _preprocessor


# ======================================================================
# MFCC extractor
# ======================================================================


def extract_mfcc(
    file_path: str,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extract MFCC features from an audio file for traditional ML models.

    Computes MFCCs and aggregates them across time via mean and standard
    deviation to produce a fixed-length feature vector.

    Args:
        file_path: Path to audio file.
        sr: Sample rate in Hz (default: 16000).
        n_mfcc: Number of MFCCs to extract (default: 40).
        n_fft: FFT window size (default: 2048).
        hop_length: Hop length in samples (default: 512).

    Returns:
        1-D numpy array of shape ``(n_mfcc * 2,)`` = ``(80,)`` containing
        concatenated mean and std of the MFCCs.

    Raises:
        RuntimeError: If extraction or quality checks fail.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(42)

    logger.info(f"Extracting MFCC features from: {file_path}")

    try:
        preprocessor = _get_preprocessor(target_sr=sr)
        audio, _ = preprocessor.process(file_path)

        # Compute MFCCs — shape (n_mfcc, time_frames)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )

        # Aggregate across time
        mfcc_mean = np.mean(mfccs, axis=1)   # (n_mfcc,)
        mfcc_std = np.std(mfccs, axis=1)     # (n_mfcc,)
        features = np.concatenate([mfcc_mean, mfcc_std])  # (n_mfcc*2,)

        # Quality checks
        expected_shape = (n_mfcc * 2,)
        assert features.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {features.shape}"
        )
        if np.any(np.isnan(features)):
            raise RuntimeError("MFCC features contain NaN values")
        if np.any(np.isinf(features)):
            raise RuntimeError("MFCC features contain Inf values")

        logger.info(f"MFCC features extracted: shape {features.shape}")
        return features.astype(np.float32)

    except RuntimeError:
        raise
    except Exception as e:
        msg = f"MFCC extraction failed for {file_path}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


# ======================================================================
# Mel-spectrogram extractor
# ======================================================================


def extract_mel_spectrogram(
    file_path: str,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    target_height: int = 128,
    target_width: int = 128,
) -> np.ndarray:
    """
    Extract a Mel-spectrogram from an audio file for CNN models.

    Computes the mel-spectrogram, converts to log-dB scale, normalizes
    per-sample, resizes to ``(target_height, target_width)``, and adds a
    trailing channel dimension.

    Args:
        file_path: Path to audio file.
        sr: Sample rate in Hz (default: 16000).
        n_fft: FFT window size (default: 1024).
        hop_length: Hop length in samples (default: 512).
        n_mels: Number of mel bands (default: 128).
        target_height: Target spectrogram height (default: 128).
        target_width: Target spectrogram width (default: 128).

    Returns:
        3-D numpy array of shape ``(128, 128, 1)`` (float32).

    Raises:
        RuntimeError: If extraction or quality checks fail.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(42)

    logger.info(f"Extracting Mel-spectrogram from: {file_path}")

    try:
        preprocessor = _get_preprocessor(target_sr=sr)
        audio, _ = preprocessor.process(file_path)

        # Compute power mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=sr // 2,
            power=2.0,
            center=False,
        )

        logger.debug(f"Raw mel shape: {mel.shape}")

        # Convert to dB scale
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Per-sample z-normalization
        eps = 1e-8
        mean = np.mean(mel_db)
        std = np.std(mel_db)
        mel_norm = (mel_db - mean) / (std + eps)

        # ----- Resize to (target_height, target_width) -----
        # mel_norm shape is (n_mels, time_frames)
        # We need exactly (128, 128).
        current_h, current_w = mel_norm.shape

        # Height: should already be n_mels=128; handle mismatch
        if current_h != target_height:
            # Use linear interpolation along freq axis
            indices = np.linspace(0, current_h - 1, target_height)
            mel_norm = np.array(
                [np.interp(indices, np.arange(current_h), mel_norm[:, t])
                 for t in range(current_w)]
            ).T  # -> (target_height, current_w)

        # Width: resize time axis
        if mel_norm.shape[1] != target_width:
            current_w = mel_norm.shape[1]
            indices = np.linspace(0, current_w - 1, target_width)
            mel_norm = np.array(
                [np.interp(indices, np.arange(current_w), mel_norm[f, :])
                 for f in range(target_height)]
            )  # -> (target_height, target_width)

        # Add channel dimension → (128, 128, 1)
        mel_final = np.expand_dims(mel_norm, axis=-1).astype(np.float32)

        # Quality checks
        assert mel_final.shape == (target_height, target_width, 1), (
            f"Expected ({target_height}, {target_width}, 1), "
            f"got {mel_final.shape}"
        )
        if np.any(np.isnan(mel_final)):
            raise RuntimeError("Mel-spectrogram contains NaN values")
        if np.any(np.isinf(mel_final)):
            raise RuntimeError("Mel-spectrogram contains Inf values")

        logger.info(f"Mel-spectrogram extracted: shape {mel_final.shape}")
        return mel_final

    except RuntimeError:
        raise
    except Exception as e:
        msg = f"Mel-spectrogram extraction failed for {file_path}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


# ======================================================================
# Raw waveform extractor
# ======================================================================


def extract_waveform(
    file_path: str,
    sr: int = 16000,
    duration: float = 3.0,
) -> np.ndarray:
    """
    Extract a raw waveform from an audio file for YAMNet models.

    Args:
        file_path: Path to audio file.
        sr: Sample rate in Hz (default: 16000).
        duration: Target duration in seconds (default: 3.0).

    Returns:
        1-D numpy array of shape ``(sr * duration,)`` = ``(48000,)``
        with dtype float32.

    Raises:
        RuntimeError: If extraction or quality checks fail.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(42)

    logger.info(f"Extracting raw waveform from: {file_path}")

    try:
        preprocessor = _get_preprocessor(target_sr=sr, duration=duration)
        waveform, _ = preprocessor.process(file_path)
        waveform = waveform.astype(np.float32)

        expected_length = int(sr * duration)
        assert waveform.ndim == 1, (
            f"Expected 1-D array, got {waveform.ndim}-D"
        )
        assert len(waveform) == expected_length, (
            f"Expected length {expected_length}, got {len(waveform)}"
        )
        if np.any(np.isnan(waveform)):
            raise RuntimeError("Waveform contains NaN values")
        if np.any(np.isinf(waveform)):
            raise RuntimeError("Waveform contains Inf values")

        logger.info(
            f"Waveform extracted: shape {waveform.shape}, "
            f"dtype {waveform.dtype}"
        )
        return waveform

    except RuntimeError:
        raise
    except Exception as e:
        msg = f"Waveform extraction failed for {file_path}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


# ======================================================================
# Audio augmentation
# ======================================================================


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply probabilistic audio augmentations for training data.

    Augmentations (applied in order when triggered):
    1. Time shift (p=0.5, ±10 % of length)
    2. Additive Gaussian noise (p=0.5, std=0.005 × peak)
    3. Gain scaling (p=0.3, range 0.8–1.2)
    4. Time stretch (p=0.4, rate 0.8–1.2)
    5. Pitch shift (p=0.4, ±2 semitones)

    Args:
        audio: Input audio array (float32).
        sr: Sample rate in Hz (default: 16000).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Augmented audio array with the **same shape** as input, dtype float32.

    Raises:
        RuntimeError: If augmentation fails.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)

    logger.debug(f"augment_audio input shape: {audio.shape}")

    try:
        augmented = audio.copy().astype(np.float32)
        original_length = len(augmented)
        applied = []

        # 1. Time shift (p=0.5)
        if np.random.random() < 0.5:
            max_shift = int(0.1 * original_length)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                augmented = np.pad(augmented, (shift, 0), mode="constant")[
                    :original_length
                ]
            elif shift < 0:
                augmented = np.pad(augmented, (0, -shift), mode="constant")[
                    -shift:
                ]
            applied.append(f"time_shift({shift})")

        # 2. Additive Gaussian noise (p=0.5)
        if np.random.random() < 0.5:
            noise_std = 0.005 * np.max(np.abs(augmented))
            noise = np.random.normal(0, noise_std, augmented.shape).astype(
                np.float32
            )
            augmented = augmented + noise
            applied.append(f"gaussian_noise(std={noise_std:.6f})")

        # 3. Gain scaling (p=0.3)
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            augmented = augmented * gain
            applied.append(f"gain({gain:.3f})")

        # 4. Time stretch (p=0.4, rate 0.8–1.2)
        if np.random.random() < 0.4:
            rate = np.random.uniform(0.8, 1.2)
            stretched = librosa.effects.time_stretch(augmented, rate=rate)
            # Restore original length
            if len(stretched) > original_length:
                stretched = stretched[:original_length]
            elif len(stretched) < original_length:
                stretched = np.pad(
                    stretched,
                    (0, original_length - len(stretched)),
                    mode="constant",
                )
            augmented = stretched.astype(np.float32)
            applied.append(f"time_stretch({rate:.3f})")

        # 5. Pitch shift (p=0.4, ±2 semitones)
        if np.random.random() < 0.4:
            n_steps = np.random.uniform(-2.0, 2.0)
            shifted = librosa.effects.pitch_shift(
                augmented, sr=sr, n_steps=n_steps
            )
            augmented = shifted.astype(np.float32)
            applied.append(f"pitch_shift({n_steps:.2f})")

        # Clip to [-1, 1]
        augmented = np.clip(augmented, -1.0, 1.0)

        # Ensure original shape preserved
        assert augmented.shape == audio.shape, (
            f"Shape mismatch: expected {audio.shape}, got {augmented.shape}"
        )
        if np.any(np.isnan(augmented)):
            raise RuntimeError("Augmented audio contains NaN values")
        if np.any(np.isinf(augmented)):
            raise RuntimeError("Augmented audio contains Inf values")

        if applied:
            logger.debug(f"Augmentations applied: {', '.join(applied)}")
        else:
            logger.debug("No augmentations applied (all skipped by RNG)")

        return augmented

    except RuntimeError:
        raise
    except Exception as e:
        msg = f"Audio augmentation failed: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


# ======================================================================
# SpecAugment (mel-spectrogram only)
# ======================================================================


def spec_augment(
    mel_spec: np.ndarray,
    num_freq_masks: int = 2,
    freq_mask_width: int = 15,
    num_time_masks: int = 2,
    time_mask_width: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply SpecAugment masks to a mel-spectrogram.

    Applies *frequency masking* and *time masking* as described in
    Park et al. (2019). Should only be used for training data.

    Args:
        mel_spec: Input mel-spectrogram of shape ``(H, W, 1)`` (float32).
        num_freq_masks: Number of frequency masks to apply (default: 2).
        freq_mask_width: Max width of each frequency mask (default: 15).
        num_time_masks: Number of time masks to apply (default: 2).
        time_mask_width: Max width of each time mask (default: 15).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Augmented mel-spectrogram with the **same shape** as input (float32).

    Raises:
        RuntimeError: If SpecAugment fails.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(seed)

    logger.debug(f"SpecAugment input shape: {mel_spec.shape}")

    try:
        assert mel_spec.ndim == 3 and mel_spec.shape[2] == 1, (
            f"Expected shape (H, W, 1), got {mel_spec.shape}"
        )

        augmented = mel_spec.copy().astype(np.float32)
        height, width, _ = augmented.shape

        # Frequency masking
        for i in range(num_freq_masks):
            f = np.random.randint(0, min(freq_mask_width, height) + 1)
            f0 = np.random.randint(0, max(height - f, 1))
            augmented[f0 : f0 + f, :, :] = 0.0
            logger.debug(
                f"Freq mask {i + 1}: rows [{f0}:{f0 + f}] (width={f})"
            )

        # Time masking
        for i in range(num_time_masks):
            t = np.random.randint(0, min(time_mask_width, width) + 1)
            t0 = np.random.randint(0, max(width - t, 1))
            augmented[:, t0 : t0 + t, :] = 0.0
            logger.debug(
                f"Time mask {i + 1}: cols [{t0}:{t0 + t}] (width={t})"
            )

        # Quality checks — shape unchanged, no NaN
        assert augmented.shape == mel_spec.shape, (
            f"Shape changed: {mel_spec.shape} → {augmented.shape}"
        )
        if np.any(np.isnan(augmented)):
            raise RuntimeError("SpecAugment introduced NaN values")

        logger.debug("SpecAugment applied successfully")
        return augmented

    except RuntimeError:
        raise
    except Exception as e:
        msg = f"SpecAugment failed: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


# ======================================================================
# Feature caching
# ======================================================================


def cache_features(
    split_csv: str,
    split_name: str,
    feature_root: str = "features",
    apply_augmentation: bool = False,
) -> None:
    """
    Cache extracted features for a data split to disk.

    Extracts MFCC, Mel-spectrogram, and waveform features for every file
    listed in the split CSV and saves them as ``.npy`` files. Optionally
    applies audio augmentation and SpecAugment for training data.

    Args:
        split_csv: Path to split CSV file (e.g. ``data/splits/train.csv``).
        split_name: Name of the split (``train``, ``val``, ``test``).
        feature_root: Root directory for cached features (default: ``features``).
        apply_augmentation: Whether to apply augmentation (default: False).
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(42)

    logger.info(f"Feature caching — split: {split_name}")
    logger.info(f"  CSV: {split_csv}")
    logger.info(f"  Augmentation: {'ON' if apply_augmentation else 'OFF'}")

    try:
        df = pd.read_csv(split_csv)
        logger.info(f"  Files: {len(df)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load split CSV: {e}") from e

    root = Path(feature_root)
    mfcc_dir = root / "mfcc" / split_name
    mel_dir = root / "mel" / split_name
    wav_dir = root / "waveform" / split_name

    for d in (mfcc_dir, mel_dir, wav_dir):
        d.mkdir(parents=True, exist_ok=True)

    cached = skipped = failed = 0
    total = len(df)

    preprocessor = _get_preprocessor()

    for idx, row in df.iterrows():
        fold = row["fold"]
        fname = row["slice_file_name"]
        audio_path = (
            Path("data") / "UrbanSound8K" / "audio" / f"fold{fold}" / fname
        )
        stem = Path(fname).stem
        mfcc_f = mfcc_dir / f"{stem}.npy"
        mel_f = mel_dir / f"{stem}.npy"
        wav_f = wav_dir / f"{stem}.npy"

        if mfcc_f.exists() and mel_f.exists() and wav_f.exists():
            skipped += 1
            continue

        try:
            audio_str = str(audio_path)

            if apply_augmentation:
                # Preprocess then augment the raw audio
                audio, _ = preprocessor.process(audio_str)
                aug_audio = augment_audio(audio, seed=42 + idx)

                # MFCC from augmented audio
                mfccs = librosa.feature.mfcc(
                    y=aug_audio, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512
                )
                mfcc_feat = np.concatenate(
                    [np.mean(mfccs, axis=1), np.std(mfccs, axis=1)]
                ).astype(np.float32)

                # Mel-spectrogram from augmented audio
                mel_feat = _mel_from_audio(aug_audio)

                # Apply SpecAugment on mel
                mel_feat = spec_augment(mel_feat, seed=42 + idx)

                # Waveform is the augmented audio itself
                wav_feat = aug_audio.astype(np.float32)
            else:
                mfcc_feat = extract_mfcc(audio_str)
                mel_feat = extract_mel_spectrogram(audio_str)
                wav_feat = extract_waveform(audio_str)

            if not mfcc_f.exists():
                np.save(mfcc_f, mfcc_feat)
            if not mel_f.exists():
                np.save(mel_f, mel_feat)
            if not wav_f.exists():
                np.save(wav_f, wav_feat)

            cached += 1

            if (idx + 1) % 100 == 0:
                logger.info(f"  Progress: {idx + 1}/{total}")

        except Exception as e:
            failed += 1
            logger.error(f"  Failed {fname}: {e}")
            continue

    logger.info(
        f"Caching complete — cached: {cached}, skipped: {skipped}, "
        f"failed: {failed}, total: {total}"
    )


def _mel_from_audio(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    target_height: int = 128,
    target_width: int = 128,
) -> np.ndarray:
    """
    Internal helper: compute a mel-spectrogram directly from an audio array.

    Returns shape ``(128, 128, 1)`` as float32.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=sr // 2,
        power=2.0,
        center=False,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    eps = 1e-8
    mean = np.mean(mel_db)
    std = np.std(mel_db)
    mel_norm = (mel_db - mean) / (std + eps)

    current_h, current_w = mel_norm.shape

    if current_h != target_height:
        indices = np.linspace(0, current_h - 1, target_height)
        mel_norm = np.array(
            [np.interp(indices, np.arange(current_h), mel_norm[:, t])
             for t in range(current_w)]
        ).T

    if mel_norm.shape[1] != target_width:
        current_w = mel_norm.shape[1]
        indices = np.linspace(0, current_w - 1, target_width)
        mel_norm = np.array(
            [np.interp(indices, np.arange(current_w), mel_norm[f, :])
             for f in range(target_height)]
        )

    return np.expand_dims(mel_norm, axis=-1).astype(np.float32)
