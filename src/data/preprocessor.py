"""
Audio preprocessing module for the hearing aid research project.

Provides the AudioPreprocessor class which handles:
- Audio loading and resampling
- DC offset removal
- Peak and RMS normalization
- Fixed-length padding/truncation (3s @ 16kHz)
- Quality checks: SNR estimation, silence detection, clipping detection

All operations are deterministic when seeded via set_seed().
"""

import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import librosa

from src.utils import setup_logging, set_seed


class AudioPreprocessor:
    """
    Standardized audio preprocessing pipeline for ML model input.

    Handles loading, cleaning, normalization, and quality validation
    of audio signals for downstream feature extraction.

    Attributes:
        target_sr: Target sample rate in Hz.
        duration: Target duration in seconds.
        target_length: Target length in samples (target_sr * duration).
        min_snr_db: Minimum acceptable SNR in dB.
        min_rms: Minimum acceptable RMS energy.
        eps: Small epsilon to avoid division by zero.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        duration: float = 3.0,
        min_snr_db: float = 5.0,
        min_rms: float = 1e-4,
        eps: float = 1e-8,
        seed: int = 42,
    ):
        """
        Initialize the AudioPreprocessor.

        Args:
            target_sr: Target sample rate in Hz (default: 16000).
            duration: Target duration in seconds (default: 3.0).
            min_snr_db: Minimum SNR threshold in dB (default: 5.0).
            min_rms: Minimum RMS energy threshold (default: 1e-4).
            eps: Epsilon for numerical stability (default: 1e-8).
            seed: Random seed for reproducibility (default: 42).
        """
        setup_logging()
        self.logger = logging.getLogger(__name__)

        self.target_sr = target_sr
        self.duration = duration
        self.target_length = int(target_sr * duration)
        self.min_snr_db = min_snr_db
        self.min_rms = min_rms
        self.eps = eps
        self.seed = seed

        set_seed(self.seed)

        self.logger.info(
            f"AudioPreprocessor initialized: sr={target_sr}, "
            f"duration={duration}s, target_length={self.target_length}, "
            f"min_snr={min_snr_db}dB"
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and resample to the target sample rate.

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (audio_array as float32, sample_rate).

        Raises:
            RuntimeError: If the file cannot be loaded.
        """
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)

            if sr != self.target_sr:
                self.logger.info(
                    f"Resampling {file_path} from {sr} Hz to {self.target_sr} Hz"
                )
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=self.target_sr
                )
                sr = self.target_sr

            audio = audio.astype(np.float32)
            self.logger.debug(
                f"Loaded {file_path}: {len(audio)} samples @ {sr} Hz"
            )
            return audio, sr

        except Exception as e:
            msg = f"Failed to load audio from {file_path}: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e

    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset by subtracting the mean.

        This centres the waveform around zero, eliminating any constant
        bias introduced during recording.

        Args:
            audio: Input audio array.

        Returns:
            DC-corrected audio array (float32).
        """
        dc_offset = np.mean(audio)
        audio = (audio - dc_offset).astype(np.float32)
        self.logger.debug(f"Removed DC offset: {dc_offset:.6f}")
        return audio

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio using peak normalization followed by RMS normalization.

        1. Peak-normalize to [-1, 1].
        2. RMS-normalize and re-clip to [-1, 1].

        Args:
            audio: Input audio array.

        Returns:
            Normalized audio array in [-1, 1] (float32).
        """
        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > self.eps:
            audio = audio / peak
            self.logger.debug(f"Peak normalized (peak={peak:.4f})")

        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > self.eps:
            audio = audio / (rms + self.eps)
            # Re-clip to [-1, 1]
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak
            self.logger.debug(f"RMS normalized (rms={rms:.4f})")

        return audio.astype(np.float32)

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad with zeros or truncate to exactly ``self.target_length`` samples.

        Args:
            audio: Input audio array.

        Returns:
            Audio array of length ``self.target_length`` (float32).
        """
        current = len(audio)

        if current < self.target_length:
            padding = self.target_length - current
            audio = np.pad(audio, (0, padding), mode="constant", constant_values=0)
            self.logger.debug(
                f"Padded from {current} to {self.target_length} samples"
            )
        elif current > self.target_length:
            audio = audio[: self.target_length]
            self.logger.debug(
                f"Truncated from {current} to {self.target_length} samples"
            )
        else:
            self.logger.debug(
                f"Audio already at target length: {self.target_length}"
            )

        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate the Signal-to-Noise Ratio (SNR) in dB.

        Uses the last 10 % of the clip as a noise estimate and
        the full signal power for the signal estimate.

        SNR_dB = 10 * log10(signal_power / noise_power)

        Args:
            audio: Input audio array.

        Returns:
            Estimated SNR in dB.
        """
        noise_start = int(0.9 * len(audio))
        noise_segment = audio[noise_start:]

        signal_power = np.mean(audio ** 2) + self.eps
        noise_power = np.mean(noise_segment ** 2) + self.eps

        snr_db = 10.0 * np.log10(signal_power / noise_power)
        self.logger.debug(f"Estimated SNR: {snr_db:.2f} dB")
        return float(snr_db)

    def check_silence(self, audio: np.ndarray) -> bool:
        """
        Check whether the audio is silent or near-silent.

        Args:
            audio: Input audio array.

        Returns:
            True if the audio is silent (RMS < min_rms), False otherwise.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        is_silent = rms < self.min_rms
        if is_silent:
            self.logger.warning(f"Silent audio detected (RMS={rms:.2e})")
        return is_silent

    def check_clipping(self, audio: np.ndarray) -> bool:
        """
        Check whether the audio is clipped.

        Args:
            audio: Input audio array.

        Returns:
            True if clipping is detected (max |value| > 1.0), False otherwise.
        """
        max_val = np.max(np.abs(audio))
        is_clipped = max_val > 1.0
        if is_clipped:
            self.logger.warning(
                f"Clipping detected (max={max_val:.4f})"
            )
        return is_clipped

    def quality_checks(self, audio: np.ndarray, file_path: str = "") -> Dict:
        """
        Run all quality checks and return a metadata dict.

        Checks performed:
        - Zero-length
        - NaN / Inf values
        - Silence detection
        - Clipping detection
        - SNR estimation (warning if below threshold)

        Args:
            audio: Preprocessed audio array.
            file_path: Optional file path for logging context.

        Returns:
            Dict with keys: ``snr_db``, ``rms``, ``is_silent``,
            ``is_clipped``, ``max_abs``.

        Raises:
            RuntimeError: If a critical quality check fails.
        """
        if len(audio) == 0:
            raise RuntimeError(f"Audio length is zero: {file_path}")

        if np.any(np.isnan(audio)):
            raise RuntimeError(f"Audio contains NaN values: {file_path}")

        if np.any(np.isinf(audio)):
            raise RuntimeError(f"Audio contains Inf values: {file_path}")

        # Silence check
        is_silent = self.check_silence(audio)
        if is_silent:
            self.logger.warning(f"Silent audio in {file_path}")

        # Clipping check
        is_clipped = self.check_clipping(audio)
        if is_clipped:
            raise RuntimeError(
                f"Audio exceeds [-1, 1] range: {file_path}"
            )

        # SNR estimation
        snr_db = self.estimate_snr(audio)
        if snr_db < self.min_snr_db:
            self.logger.warning(
                f"Low SNR ({snr_db:.2f} dB < {self.min_snr_db} dB) in {file_path}"
            )

        rms = float(np.sqrt(np.mean(audio ** 2)))
        max_abs = float(np.max(np.abs(audio)))

        metadata = {
            "snr_db": snr_db,
            "rms": rms,
            "is_silent": bool(is_silent),
            "is_clipped": bool(is_clipped),
            "max_abs": max_abs,
        }

        self.logger.debug(f"Quality metadata for {file_path}: {metadata}")
        return metadata

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(
        self, file_path: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run the complete preprocessing pipeline.

        Pipeline order:
        1. Load & resample
        2. DC offset removal
        3. Normalization (peak + RMS)
        4. Pad / truncate to fixed length
        5. Quality checks

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (preprocessed audio array of shape
            ``(target_length,)`` as float32, quality metadata dict).

        Raises:
            RuntimeError: If any pipeline step or quality check fails.
        """
        set_seed(self.seed)
        self.logger.info(f"Processing: {file_path}")

        try:
            # 1. Load
            audio, _ = self.load_audio(file_path)

            # 2. DC offset removal (before normalization)
            audio = self.remove_dc_offset(audio)

            # 3. Normalization
            audio = self.normalize(audio)

            # 4. Pad / truncate
            audio = self.pad_or_truncate(audio)

            # 5. Quality checks
            metadata = self.quality_checks(audio, file_path)

            # Final dtype assertion
            assert audio.dtype == np.float32, (
                f"Expected float32, got {audio.dtype}"
            )
            assert audio.shape == (self.target_length,), (
                f"Expected shape ({self.target_length},), got {audio.shape}"
            )

            self.logger.info(
                f"Preprocessed {file_path}: shape={audio.shape}, "
                f"snr={metadata['snr_db']:.2f}dB, rms={metadata['rms']:.4f}"
            )
            return audio, metadata

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"Preprocessing failed for {file_path}: {e}"
            self.logger.error(msg)
            raise RuntimeError(msg) from e
