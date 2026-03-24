"""
Feature validation tests for Phase 3 completion.

Validates:
- DC offset removal
- SNR estimation
- Feature shapes (MFCC, Mel-spectrogram, Waveform)
- Deterministic reproducibility
- No NaN / Inf values
- SpecAugment shape preservation
- Augmentation shape preservation
"""

import sys
import os
import traceback
import tempfile

# Suppress TF warnings before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import soundfile as sf

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import AudioPreprocessor
from src.features.extractor import (
    extract_mfcc,
    extract_mel_spectrogram,
    extract_waveform,
    augment_audio,
    spec_augment,
)
from src.utils import set_seed


def _create_test_wav(path: str, sr: int = 16000, duration: float = 3.0):
    """Create a simple test WAV file with a 440 Hz sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, audio, sr)
    return path


def test_dc_offset_removal():
    """DC offset removal should centre the waveform around zero."""
    preprocessor = AudioPreprocessor()
    # Audio with a large DC offset
    audio = np.ones(16000, dtype=np.float32) * 0.5
    audio += 0.3 * np.sin(np.linspace(0, 2 * np.pi * 100, 16000))
    corrected = preprocessor.remove_dc_offset(audio)

    assert corrected.dtype == np.float32, "Expected float32"
    assert abs(np.mean(corrected)) < 1e-6, (
        f"DC offset not removed: mean={np.mean(corrected):.8f}"
    )
    print("  [PASS] DC offset removal")


def test_snr_estimation():
    """SNR estimation should return a finite dB value."""
    preprocessor = AudioPreprocessor()
    t = np.linspace(0, 3.0, 48000, endpoint=False)
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    snr = preprocessor.estimate_snr(audio)
    assert np.isfinite(snr), f"SNR is not finite: {snr}"
    assert isinstance(snr, float), "Expected float"
    print(f"  [PASS] SNR estimation (snr={snr:.2f} dB)")


def test_quality_checks():
    """quality_checks should return metadata dict with expected keys."""
    preprocessor = AudioPreprocessor()
    t = np.linspace(0, 3.0, 48000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    meta = preprocessor.quality_checks(audio, file_path="test_file")
    assert "snr_db" in meta
    assert "rms" in meta
    assert "is_silent" in meta
    assert "is_clipped" in meta
    assert "max_abs" in meta
    assert meta["is_clipped"] == False
    snr_val = meta["snr_db"]
    rms_val = meta["rms"]
    print("  [PASS] Quality checks (snr=%.2f, rms=%.4f)" % (snr_val, rms_val))


def test_preprocessor_pipeline():
    """Full pipeline should return (48000,) float32 array + metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = _create_test_wav(os.path.join(tmpdir, "test.wav"))
        preprocessor = AudioPreprocessor(target_sr=16000, duration=3.0)
        audio, meta = preprocessor.process(wav_path)

        assert audio.shape == (48000,), f"Expected (48000,), got {audio.shape}"
        assert audio.dtype == np.float32, f"Expected float32, got {audio.dtype}"
        assert not np.any(np.isnan(audio)), "NaN detected"
        assert not np.any(np.isinf(audio)), "Inf detected"
        assert "snr_db" in meta
        print(f"  [PASS] Preprocessor pipeline (shape={audio.shape}, snr={meta['snr_db']:.2f})")


def test_mfcc_shape():
    """MFCC should be (80,) float32."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = _create_test_wav(os.path.join(tmpdir, "test.wav"))
        features = extract_mfcc(wav_path)

        assert features.shape == (80,), f"Expected (80,), got {features.shape}"
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        print(f"  [PASS] MFCC shape = {features.shape}")


def test_mel_spectrogram_shape():
    """Mel-spectrogram should be (128, 128, 1) float32."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = _create_test_wav(os.path.join(tmpdir, "test.wav"))
        features = extract_mel_spectrogram(wav_path)

        assert features.shape == (128, 128, 1), (
            f"Expected (128, 128, 1), got {features.shape}"
        )
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        print(f"  [PASS] Mel-spectrogram shape = {features.shape}")


def test_waveform_shape():
    """Raw waveform should be (48000,) float32."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = _create_test_wav(os.path.join(tmpdir, "test.wav"))
        features = extract_waveform(wav_path)

        assert features.shape == (48000,), f"Expected (48000,), got {features.shape}"
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        print(f"  [PASS] Waveform shape = {features.shape}")


def test_deterministic_extraction():
    """Two identical extractions should produce identical features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = _create_test_wav(os.path.join(tmpdir, "test.wav"))

        set_seed(42)
        mfcc1 = extract_mfcc(wav_path)
        mel1 = extract_mel_spectrogram(wav_path)
        wav1 = extract_waveform(wav_path)

        set_seed(42)
        mfcc2 = extract_mfcc(wav_path)
        mel2 = extract_mel_spectrogram(wav_path)
        wav2 = extract_waveform(wav_path)

        assert np.allclose(mfcc1, mfcc2), "MFCC not deterministic"
        assert np.allclose(mel1, mel2), "Mel-spectrogram not deterministic"
        assert np.allclose(wav1, wav2), "Waveform not deterministic"
        print("  [PASS] Deterministic extraction verified")


def test_augment_audio_shape():
    """Augmented audio should preserve shape and be float32."""
    t = np.linspace(0, 3.0, 48000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    augmented = augment_audio(audio, sr=16000, seed=42)

    assert augmented.shape == audio.shape, (
        f"Shape mismatch: {audio.shape} vs {augmented.shape}"
    )
    assert augmented.dtype == np.float32
    assert not np.any(np.isnan(augmented))
    assert not np.any(np.isinf(augmented))
    assert np.max(np.abs(augmented)) <= 1.0, "Augmented audio exceeds [-1, 1]"
    print(f"  [PASS] Augmentation shape preserved = {augmented.shape}")


def test_spec_augment_shape():
    """SpecAugment should preserve shape and introduce no NaN."""
    mel = np.random.randn(128, 128, 1).astype(np.float32)
    augmented = spec_augment(mel, seed=42)

    assert augmented.shape == mel.shape, (
        f"Shape mismatch: {mel.shape} vs {augmented.shape}"
    )
    assert augmented.dtype == np.float32
    assert not np.any(np.isnan(augmented))
    print(f"  [PASS] SpecAugment shape preserved = {augmented.shape}")


def test_spec_augment_masks_applied():
    """SpecAugment should zero out some values (masks applied)."""
    mel = np.ones((128, 128, 1), dtype=np.float32)
    augmented = spec_augment(mel, seed=42)

    # Some values should be zeroed
    num_zeros = np.sum(augmented == 0.0)
    assert num_zeros > 0, "No masking was applied"
    print(f"  [PASS] SpecAugment masks applied ({num_zeros} zeros)")


# ======================================================================
# Runner
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Feature Validation Tests")
    print("=" * 60)

    tests = [
        ("DC Offset Removal", test_dc_offset_removal),
        ("SNR Estimation", test_snr_estimation),
        ("Quality Checks", test_quality_checks),
        ("Preprocessor Pipeline", test_preprocessor_pipeline),
        ("MFCC Shape", test_mfcc_shape),
        ("Mel-spectrogram Shape", test_mel_spectrogram_shape),
        ("Waveform Shape", test_waveform_shape),
        ("Deterministic Extraction", test_deterministic_extraction),
        ("Augment Audio Shape", test_augment_audio_shape),
        ("SpecAugment Shape", test_spec_augment_shape),
        ("SpecAugment Masks", test_spec_augment_masks_applied),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED [OK]")
