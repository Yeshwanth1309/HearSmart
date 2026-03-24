"""
Phase 9 — Integration Test: audio file → inference pipeline → recommendation.

Tests the full end-to-end pipeline WITHOUT starting the HTTP server:
    1. Load all 5 models
    2. Run inference on a real audio file from the test split
    3. Verify prediction shape, confidence bounds, recommendation validity
    4. Verify latency < 500ms constraint

Run: python -m pytest tests/test_integration.py -v
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture — load once for all tests
# ─────────────────────────────────────────────────────────────────────────────
_MODELS = None
_WEIGHTS = None
_TEST_AUDIO_PATH = None


def _load_fixture():
    global _MODELS, _WEIGHTS, _TEST_AUDIO_PATH

    if _MODELS is not None:
        return

    import joblib
    import tensorflow as tf
    import tensorflow_hub as hub
    import torch

    from src.models import AudioCNN

    _MODELS = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    _MODELS["rf"]  = joblib.load("models/random_forest.pkl")
    _MODELS["svm"] = joblib.load("models/svm.pkl")

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model("models/xgboost.json")
    _MODELS["xgb"] = xgb

    cnn = AudioCNN(num_classes=10).to(torch.device(device))
    cnn.load_state_dict(torch.load("models/cnn_best.pt", map_location=device))
    cnn.eval()
    _MODELS["cnn"] = cnn
    _MODELS["_device"] = device

    _MODELS["yamnet_base"] = hub.load("https://tfhub.dev/google/yamnet/1")
    _MODELS["yamnet_head"] = tf.keras.models.load_model("models/yamnet_head.h5")

    E_train = np.load("features/yamnet_embeddings/train.npy")
    _MODELS["yamnet_mean"] = E_train.mean(axis=0, keepdims=True)
    _MODELS["yamnet_std"]  = E_train.std(axis=0, keepdims=True) + 1e-8

    with open("results/ensemble_weights.json") as f:
        _WEIGHTS = json.load(f)["weights"]

    # Pick a real audio file from the test split
    df = pd.read_csv("data/splits/test.csv")
    row = df.iloc[0]
    fold = int(row["fold"])
    fname = row["slice_file_name"]
    _TEST_AUDIO_PATH = str(
        Path("data/UrbanSound8K/audio") / f"fold{fold}" / fname
    )
    assert Path(_TEST_AUDIO_PATH).exists(), f"Test audio not found: {_TEST_AUDIO_PATH}"


def _run_full_inference(audio_path: str) -> dict:
    """Mirror of demo/api.py inference logic for testing."""
    import tensorflow as tf
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import zoom

    from src.feature_extraction import (
        extract_mel_spectrogram, extract_mfcc, extract_waveform,
        load_audio, preprocess_audio,
    )

    t0 = time.perf_counter()

    # All feature functions take a file path
    mfcc = extract_mfcc(audio_path).reshape(1, -1)              # (1, 80)
    wave = extract_waveform(audio_path)                         # (48000,)

    # Mel-spec: extract_mel_spectrogram returns (1, n_mels, T)
    mel_raw = extract_mel_spectrogram(audio_path)               # (1, n_mels, T)
    # CNN expects (1, 1, 128, 128) — resize spatial dims with scipy zoom
    mel_2d = mel_raw[0]                                         # (n_mels, T)
    if mel_2d.shape != (128, 128):
        zy = 128 / mel_2d.shape[0]
        zx = 128 / mel_2d.shape[1]
        mel_2d = zoom(mel_2d, (zy, zx), order=1)
    mel = mel_2d[np.newaxis, np.newaxis]                        # (1, 1, 128, 128)

    probs = {}
    probs["rf"]  = _MODELS["rf"].predict_proba(mfcc).astype(np.float32)
    try:
        probs["svm"] = _MODELS["svm"].predict_proba(mfcc).astype(np.float32)
    except Exception:
        sc  = _MODELS["svm"].named_steps["scaler"].transform(mfcc)
        d   = _MODELS["svm"].named_steps["svm"].decision_function(sc).astype(np.float32)
        d  -= d.max(1, keepdims=True)
        probs["svm"] = np.exp(d) / np.exp(d).sum(1, keepdims=True)
    probs["xgb"] = _MODELS["xgb"].predict_proba(mfcc).astype(np.float32)

    device = _MODELS["_device"]
    with torch.no_grad():
        mel_t = torch.from_numpy(mel).to(torch.device(device))
        probs["cnn"] = F.softmax(_MODELS["cnn"](mel_t), dim=1).cpu().numpy().astype(np.float32)

    wav_tf   = tf.constant(wave.flatten(), dtype=tf.float32)
    _, emb, _ = _MODELS["yamnet_base"](wav_tf)
    ep       = tf.reduce_mean(emb, axis=0, keepdims=True).numpy()
    en       = ((ep - _MODELS["yamnet_mean"]) / _MODELS["yamnet_std"]).astype(np.float32)
    probs["yamnet"] = _MODELS["yamnet_head"].predict(en, verbose=0).astype(np.float32)

    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    w = np.array([_WEIGHTS[n] for n in order], dtype=np.float32)
    P = np.stack([probs[n][0] for n in order], axis=0)
    ens = (P * w[:, None]).sum(0)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "class_name":    CLASS_NAMES[int(np.argmax(ens))],
        "confidence":    float(ens.max()),
        "probs":         ens,
        "model_probs":   probs,
        "latency_ms":    latency_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
class TestInferencePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _load_fixture()

    def test_models_loaded(self):
        """All 5 models + yamnet base must be in the model dict."""
        for key in ["rf", "svm", "xgb", "cnn", "yamnet_base", "yamnet_head"]:
            self.assertIn(key, _MODELS)

    def test_weights_sum_to_one(self):
        total = sum(_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_weights_positive(self):
        for name, w in _WEIGHTS.items():
            self.assertGreater(w, 0, f"{name} weight must be > 0")

    def test_inference_returns_valid_class(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        self.assertIn(result["class_name"], CLASS_NAMES)

    def test_confidence_in_0_1(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_probs_sum_to_one(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        self.assertAlmostEqual(float(result["probs"].sum()), 1.0, places=4)

    def test_probs_length_is_10(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        self.assertEqual(len(result["probs"]), 10)

    def test_all_probs_nonnegative(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        self.assertTrue(np.all(result["probs"] >= 0))

    def test_latency_under_500ms(self):
        """PRD constraint: inference latency < 500ms."""
        result = _run_full_inference(_TEST_AUDIO_PATH)
        self.assertLess(result["latency_ms"], 500,
                        f"Latency {result['latency_ms']:.1f}ms exceeds 500ms limit")

    def test_each_model_outputs_10_probs(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        for model_name, p in result["model_probs"].items():
            self.assertEqual(p.shape[1], 10, f"{model_name}: expected 10 classes")

    def test_each_model_probs_sum_to_one(self):
        result = _run_full_inference(_TEST_AUDIO_PATH)
        for model_name, p in result["model_probs"].items():
            self.assertAlmostEqual(float(p[0].sum()), 1.0, places=4,
                                   msg=f"{model_name} probs don't sum to 1")


class TestEndToEndWithRecommendation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _load_fixture()
        cls.result = _run_full_inference(_TEST_AUDIO_PATH)

    def test_recommendation_generated(self):
        from src.recommendations import recommend_from_probs
        rec = recommend_from_probs(self.result["probs"], CLASS_NAMES)
        self.assertIsNotNone(rec)

    def test_recommendation_volume_in_range(self):
        from src.recommendations import recommend_from_probs
        rec = recommend_from_probs(self.result["probs"], CLASS_NAMES)
        self.assertGreaterEqual(rec.volume, 1)
        self.assertLessEqual(rec.volume, 10)

    def test_recommendation_class_matches_prediction(self):
        from src.recommendations import recommend_from_probs
        rec = recommend_from_probs(self.result["probs"], CLASS_NAMES)
        self.assertEqual(rec.environment, self.result["class_name"])

    def test_recommendation_json_serialisable(self):
        from src.recommendations import recommend_from_probs
        import json as _json
        rec = recommend_from_probs(self.result["probs"], CLASS_NAMES)
        try:
            _json.dumps(rec.to_dict())
        except TypeError as e:
            self.fail(f"Recommendation not JSON serialisable: {e}")

    def test_full_pipeline_with_user_profile(self):
        from src.recommendations import recommend_from_probs
        rec = recommend_from_probs(
            self.result["probs"], CLASS_NAMES,
            acoustic_context={"snr_db": 12.0, "speech_prob": 0.7},
            user_profile={"hearing_loss": "moderate", "age_group": "elderly"},
        )
        self.assertGreaterEqual(len(rec.tiers_applied), 1)
        self.assertLessEqual(rec.volume, 10)


class TestEnsembleWeights(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _load_fixture()

    def test_weights_file_exists(self):
        self.assertTrue(Path("results/ensemble_weights.json").exists())

    def test_ensemble_metrics_file_exists(self):
        self.assertTrue(Path("results/ensemble_metrics.json").exists())

    def test_ensemble_accuracy_above_90(self):
        with open("results/ensemble_metrics.json") as f:
            m = json.load(f)
        self.assertGreaterEqual(m["accuracy"], 0.90,
                                f"Ensemble accuracy {m['accuracy']:.4f} < 90%")

    def test_ensemble_macro_f1_above_85(self):
        with open("results/ensemble_metrics.json") as f:
            m = json.load(f)
        self.assertGreaterEqual(m["macro_f1"], 0.85,
                                f"Ensemble F1 {m['macro_f1']:.4f} < 85%")

    def test_recommendation_rules_exported(self):
        self.assertTrue(Path("results/recommendation_rules.json").exists())

    def test_evaluation_report_exists(self):
        self.assertTrue(Path("results/evaluation_report.json").exists())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
