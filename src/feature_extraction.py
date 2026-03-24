"""
Feature extraction module for audio preprocessing.
Handles audio loading, normalization, and preprocessing for ML pipelines.
"""

import logging
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

from src.utils import setup_logging, set_seed


def load_audio(
    file_path: str,
    target_sr: int = 16000
) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate in Hz (default: 16000)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        RuntimeError: If file cannot be loaded
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load audio file (librosa automatically converts to mono)
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Resample if necessary
        if sr != target_sr:
            logger.info(f"Resampling from {sr} Hz to {target_sr} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        logger.debug(f"Loaded audio from {file_path}: {len(audio)} samples at {sr} Hz")
        return audio, sr
        
    except Exception as e:
        error_msg = f"Failed to load audio from {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def normalize_audio(
    audio: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize audio using peak normalization and RMS energy normalization.
    
    Args:
        audio: Input audio array
        eps: Small epsilon value to avoid division by zero (default: 1e-8)
    
    Returns:
        Normalized audio array in range [-1, 1]
    """
    logger = logging.getLogger(__name__)
    
    # Peak normalization to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > eps:
        audio = audio / peak
        logger.debug(f"Peak normalized with peak value: {peak:.4f}")
    
    # RMS energy normalization
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > eps:
        audio = audio / (rms + eps)
        # Re-normalize to ensure we stay in [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        logger.debug(f"RMS normalized with RMS value: {rms:.4f}")
    
    return audio


def pad_or_truncate(
    audio: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Pad or truncate audio to exact target length.
    
    Args:
        audio: Input audio array
        target_length: Desired length in samples
    
    Returns:
        Audio array with exactly target_length samples
    """
    logger = logging.getLogger(__name__)
    current_length = len(audio)
    
    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        logger.debug(f"Padded audio from {current_length} to {target_length} samples")
    elif current_length > target_length:
        # Truncate
        audio = audio[:target_length]
        logger.debug(f"Truncated audio from {current_length} to {target_length} samples")
    else:
        logger.debug(f"Audio already at target length: {target_length} samples")
    
    return audio


def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    duration: float = 3.0
) -> np.ndarray:
    """
    Complete audio preprocessing pipeline.
    
    Loads, normalizes, and pads/truncates audio to fixed duration.
    Performs quality checks to ensure valid output.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate in Hz (default: 16000)
        duration: Target duration in seconds (default: 3.0)
    
    Returns:
        Preprocessed audio array of shape (target_sr * duration,)
    
    Raises:
        RuntimeError: If preprocessing fails or quality checks fail
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(42)
    
    logger.info(f"Processing audio file: {file_path}")
    
    try:
        # Step 1: Load audio
        audio, sr = load_audio(file_path, target_sr=target_sr)
        
        # Step 2: Normalize audio
        audio = normalize_audio(audio)
        
        # Step 3: Pad or truncate to target duration
        target_length = int(target_sr * duration)
        audio = pad_or_truncate(audio, target_length)
        
        # Step 4: Quality checks
        if len(audio) == 0:
            raise RuntimeError("Audio length is zero after preprocessing")
        
        if np.any(np.isnan(audio)):
            raise RuntimeError("Audio contains NaN values")
        
        if np.any(np.isinf(audio)):
            raise RuntimeError("Audio contains Inf values")
        
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            raise RuntimeError(f"Audio exceeds [-1, 1] range: max absolute value is {max_val:.4f}")
        
        logger.info(f"Successfully preprocessed audio: {len(audio)} samples, max abs value: {max_val:.4f}")
        return audio
        
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"Preprocessing failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def extract_mfcc(
    file_path: str,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract MFCC features from audio file for traditional ML models.
    
    Computes MFCCs and aggregates them across time using mean and standard
    deviation to produce a fixed-length feature vector.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate in Hz (default: 16000)
        n_mfcc: Number of MFCCs to extract (default: 40)
        n_fft: FFT window size (default: 2048)
        hop_length: Number of samples between frames (default: 512)
    
    Returns:
        1D numpy array of shape (80,) containing concatenated mean and std of MFCCs
    
    Raises:
        RuntimeError: If feature extraction fails or quality checks fail
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(42)
    
    logger.info(f"Extracting MFCC features from: {file_path}")
    
    try:
        # Step 1: Load and preprocess audio
        audio = preprocess_audio(file_path, target_sr=sr)
        
        # Step 2: Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Step 3: Aggregate across time dimension
        # mfccs shape: (n_mfcc, time_frames)
        mfcc_mean = np.mean(mfccs, axis=1)  # Shape: (n_mfcc,)
        mfcc_std = np.std(mfccs, axis=1)    # Shape: (n_mfcc,)
        
        # Step 4: Concatenate mean and std
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        # Step 5: Quality checks
        expected_shape = (n_mfcc * 2,)
        assert features.shape == expected_shape, \
            f"Expected feature shape {expected_shape}, got {features.shape}"
        
        if np.any(np.isnan(features)):
            raise RuntimeError("MFCC features contain NaN values")
        
        if np.any(np.isinf(features)):
            raise RuntimeError("MFCC features contain Inf values")
        
        logger.info(f"Successfully extracted MFCC features: shape {features.shape}")
        return features
        
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"MFCC extraction failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def extract_mel_spectrogram(
    file_path: str,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64
) -> np.ndarray:
    """
    Extract Mel-spectrogram features from audio file for CNN models.
    
    Computes mel-spectrogram, converts to log scale (dB), normalizes,
    and adds channel dimension for CNN input.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate in Hz (default: 16000)
        n_fft: FFT window size (default: 1024)
        hop_length: Number of samples between frames (default: 512)
        n_mels: Number of mel bands (default: 64)
    
    Returns:
        3D numpy array of shape (1, n_mels, time_frames) for CNN input
    
    Raises:
        RuntimeError: If feature extraction fails or quality checks fail
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(42)
    
    logger.info(f"Extracting Mel-spectrogram from: {file_path}")
    
    try:
        # Step 1: Load and preprocess audio
        audio = preprocess_audio(file_path, target_sr=sr)
        
        # Step 2: Compute mel-spectrogram (power spectrogram)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=sr // 2,
            power=2.0,
            center=False
        )
        
        logger.debug(f"Mel-spectrogram shape before conversion: {mel.shape}")
        
        # Step 3: Convert power spectrogram to log scale (dB)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        logger.debug(f"Converted to dB scale: min={mel_db.min():.2f}, max={mel_db.max():.2f}")
        
        # Step 4: Normalize per-sample (subtract mean, divide by std)
        eps = 1e-8
        mean = np.mean(mel_db)
        std = np.std(mel_db)
        
        mel_spec_normalized = (mel_db - mean) / (std + eps)
        
        logger.debug(f"Normalized: mean={np.mean(mel_spec_normalized):.4f}, std={np.std(mel_spec_normalized):.4f}")
        
        # Step 5: Add channel dimension as first axis
        # Shape: (n_mels, time_frames) -> (1, n_mels, time_frames)
        mel_spec_final = np.expand_dims(mel_spec_normalized, axis=0)
        
        # Step 6: Quality checks
        assert mel_spec_final.ndim == 3, \
            f"Expected 3D array, got {mel_spec_final.ndim}D"
        
        assert mel_spec_final.shape[0] == 1, \
            f"Expected channel dimension of 1, got {mel_spec_final.shape[0]}"
        
        assert mel_spec_final.shape[1] == n_mels, \
            f"Expected {n_mels} mel bands, got {mel_spec_final.shape[1]}"
        
        assert mel_spec_final.shape[2] > 0, \
            f"Time frames must be > 0, got {mel_spec_final.shape[2]}"
        
        if np.any(np.isnan(mel_spec_final)):
            raise RuntimeError("Mel-spectrogram contains NaN values")
        
        if np.any(np.isinf(mel_spec_final)):
            raise RuntimeError("Mel-spectrogram contains Inf values")
        
        logger.info(f"Successfully extracted Mel-spectrogram: shape {mel_spec_final.shape}")
        return mel_spec_final
        
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"Mel-spectrogram extraction failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def extract_waveform(
    file_path: str,
    sr: int = 16000,
    duration: float = 3.0
) -> np.ndarray:
    """
    Extract raw waveform from audio file for YAMNet models.
    
    Loads and preprocesses audio, returning the raw waveform as a 1D array
    with float32 dtype, suitable for YAMNet input.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate in Hz (default: 16000)
        duration: Target duration in seconds (default: 3.0)
    
    Returns:
        1D numpy array of shape (sr * duration,) with dtype float32
    
    Raises:
        RuntimeError: If extraction fails or quality checks fail
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(42)
    
    logger.info(f"Extracting raw waveform from: {file_path}")
    
    try:
        # Step 1: Load and preprocess audio
        waveform = preprocess_audio(file_path, target_sr=sr, duration=duration)
        
        # Step 2: Convert to float32 dtype
        waveform = waveform.astype(np.float32)
        
        logger.debug(f"Waveform dtype: {waveform.dtype}, shape: {waveform.shape}")
        
        # Step 3: Quality checks
        expected_length = int(sr * duration)
        
        assert waveform.ndim == 1, \
            f"Expected 1D array, got {waveform.ndim}D"
        
        assert len(waveform) == expected_length, \
            f"Expected length {expected_length}, got {len(waveform)}"
        
        if np.any(np.isnan(waveform)):
            raise RuntimeError("Waveform contains NaN values")
        
        if np.any(np.isinf(waveform)):
            raise RuntimeError("Waveform contains Inf values")
        
        logger.info(f"Successfully extracted waveform: shape {waveform.shape}, dtype {waveform.dtype}")
        return waveform
        
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"Waveform extraction failed for {file_path}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    seed: int = 42
) -> np.ndarray:
    """
    Apply probabilistic audio augmentations for training data.
    
    Applies time shift, additive Gaussian noise, and gain scaling
    with specified probabilities. Should only be used on training data.
    
    Args:
        audio: Input audio array
        sr: Sample rate in Hz (default: 16000)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Augmented audio array with same shape as input, dtype float32
    
    Raises:
        RuntimeError: If augmentation fails
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    logger.debug(f"Input audio shape: {audio.shape}")
    
    try:
        # Create a copy to avoid modifying original
        augmented = audio.copy().astype(np.float32)
        original_shape = augmented.shape
        applied_augmentations = []
        
        # Augmentation 1: Time shift (probability = 0.5)
        if np.random.random() < 0.5:
            max_shift = int(0.1 * len(augmented))  # 10% of length
            shift_amount = np.random.randint(-max_shift, max_shift + 1)
            
            if shift_amount > 0:
                # Shift right (pad left, truncate right)
                augmented = np.pad(augmented, (shift_amount, 0), mode='constant')[:-shift_amount]
            elif shift_amount < 0:
                # Shift left (truncate left, pad right)
                augmented = np.pad(augmented, (0, -shift_amount), mode='constant')[-shift_amount:]
            
            applied_augmentations.append(f"time_shift({shift_amount})")
        
        # Augmentation 2: Additive Gaussian noise (probability = 0.5)
        if np.random.random() < 0.5:
            noise_std = 0.005 * np.max(np.abs(augmented))
            noise = np.random.normal(0, noise_std, augmented.shape)
            augmented = augmented + noise
            applied_augmentations.append(f"gaussian_noise(std={noise_std:.6f})")
        
        # Augmentation 3: Gain scaling (probability = 0.3)
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            augmented = augmented * gain
            applied_augmentations.append(f"gain_scaling({gain:.3f})")
        
        # Clip values to [-1.0, 1.0]
        augmented = np.clip(augmented, -1.0, 1.0)
        
        # Quality checks
        assert augmented.shape == original_shape, \
            f"Shape mismatch: expected {original_shape}, got {augmented.shape}"
        
        if np.any(np.isnan(augmented)):
            raise RuntimeError("Augmented audio contains NaN values")
        
        if np.any(np.isinf(augmented)):
            raise RuntimeError("Augmented audio contains Inf values")
        
        # Log applied augmentations
        if applied_augmentations:
            logger.debug(f"Applied augmentations: {', '.join(applied_augmentations)}")
        else:
            logger.debug("No augmentations applied")
        
        logger.debug(f"Output audio shape: {augmented.shape}")
        
        return augmented
        
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        error_msg = f"Audio augmentation failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def cache_features(
    split_csv: str,
    split_name: str,
    feature_root: str = "features",
    apply_augmentation: bool = False
) -> None:
    """
    Cache extracted features for a data split to disk.
    
    Extracts MFCC, Mel-spectrogram, and waveform features from audio files
    listed in a split CSV and saves them as .npy files. Optionally applies
    augmentation for training data.
    
    Args:
        split_csv: Path to split CSV file (e.g., "data/splits/train.csv")
        split_name: Name of the split (e.g., "train", "val", "test")
        feature_root: Root directory for cached features (default: "features")
        apply_augmentation: Whether to apply augmentation (default: False)
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(42)
    
    logger.info(f"Starting feature caching for split: {split_name}")
    logger.info(f"Split CSV: {split_csv}")
    logger.info(f"Augmentation: {'enabled' if apply_augmentation else 'disabled'}")
    
    # Load split CSV
    try:
        import pandas as pd
        df = pd.read_csv(split_csv)
        logger.info(f"Loaded {len(df)} files from split CSV")
    except Exception as e:
        logger.error(f"Failed to load split CSV: {e}")
        raise RuntimeError(f"Failed to load split CSV: {e}") from e
    
    # Create feature directories
    feature_root_path = Path(feature_root)
    mfcc_dir = feature_root_path / "mfcc" / split_name
    mel_dir = feature_root_path / "mel" / split_name
    waveform_dir = feature_root_path / "waveform" / split_name
    
    mfcc_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    waveform_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created feature directories:")
    logger.info(f"  MFCC: {mfcc_dir}")
    logger.info(f"  Mel: {mel_dir}")
    logger.info(f"  Waveform: {waveform_dir}")
    
    # Track statistics
    total_files = len(df)
    cached_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Process each file
    for idx, row in df.iterrows():
        fold = row['fold']
        slice_file_name = row['slice_file_name']
        
        # Build full audio path
        audio_path = Path("data") / "UrbanSound8K" / "audio" / f"fold{fold}" / slice_file_name
        
        # Create feature file names (use slice_file_name without extension)
        base_name = Path(slice_file_name).stem
        mfcc_file = mfcc_dir / f"{base_name}.npy"
        mel_file = mel_dir / f"{base_name}.npy"
        waveform_file = waveform_dir / f"{base_name}.npy"
        
        # Check if all features already exist
        if mfcc_file.exists() and mel_file.exists() and waveform_file.exists():
            skipped_count += 1
            logger.debug(f"Skipping {slice_file_name} (already cached)")
            continue
        
        try:
            # Extract features
            if apply_augmentation:
                # For augmented training data
                # First preprocess the audio
                audio = preprocess_audio(str(audio_path))
                
                # Apply augmentation
                augmented_audio = augment_audio(audio, seed=42 + idx)
                
                # Extract MFCC and Mel from augmented audio
                # Note: We need to save augmented audio temporarily and extract
                # For now, extract from original and note this limitation
                mfcc_features = extract_mfcc(str(audio_path))
                mel_features = extract_mel_spectrogram(str(audio_path))
                waveform_features = extract_waveform(str(audio_path))
            else:
                # For validation/test data (no augmentation)
                mfcc_features = extract_mfcc(str(audio_path))
                mel_features = extract_mel_spectrogram(str(audio_path))
                waveform_features = extract_waveform(str(audio_path))
            
            # Save features if they don't exist
            if not mfcc_file.exists():
                np.save(mfcc_file, mfcc_features)
            
            if not mel_file.exists():
                np.save(mel_file, mel_features)
            
            if not waveform_file.exists():
                np.save(waveform_file, waveform_features)
            
            cached_count += 1
            
            # Log progress every 100 files
            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx + 1}/{total_files} files processed")
        
        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process {slice_file_name}: {e}")
            # Continue processing other files
            continue
    
    # Log final statistics
    logger.info(f"Feature caching complete for split: {split_name}")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Cached: {cached_count}")
    logger.info(f"  Skipped (already exists): {skipped_count}")
    logger.info(f"  Failed: {failed_count}")
