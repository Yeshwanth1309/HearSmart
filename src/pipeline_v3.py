"""
V3 Inference Pipeline — Full HearSmart Engine.

Implements all 6 V3 features in a single, optimized inference call:
  Feature 1: Clean 2-Model Ensemble (YAMNet head + AudioCNN only)
  Feature 2: Dedicated Safety Binary Classifier
  Feature 3: Triple Safety Fusion Logic (3-condition OR gate)
  Feature 4: Temporal Consistency Check (sliding window of last 3)
  Feature 5: Risk Scoring System
  Feature 6: Environment Transition Smoothing

Single YAMNet call per inference — embeddings reused everywhere.
"""
import csv
import io
import json
import logging
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as torch_F

from src.data.label_map_v2 import (
    CLASS_NAMES_V2,
    CLASS_EMOJIS_V2,
    RECOMMENDATION_TABLE_V2,
    get_recommendation,
)
from src.features.extractor import extract_mel_spectrogram, extract_waveform
from src.models_v2 import AudioCNN_V2
from src.models.safety_model import SafetyClassifier

logger = logging.getLogger(__name__)

# ─── YAMNET Raw Class Indices (AudioSet 521 classes) ─────────────────────────
# These are the indices in YAMNet's 521-class output for safety-relevant sounds.
# We resolve them lazily from the class map CSV on first use.
_YAMNET_SAFETY_INDICES: dict = {}  # e.g. {"Siren": [394], "Car horn": [381], ...}

_YAMNET_CLASSES_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "master/research/audioset/yamnet/yamnet_class_map.csv"
)

_YAMNET_SAFETY_NAMES = [
    "Siren", "Car horn, honking", "Ambulance (siren)",
    "Police car (siren)", "Fire engine, fire truck (siren)",
]


def _resolve_yamnet_safety_indices():
    """Load YAMNet class map and find indices for safety-relevant AudioSet classes."""
    global _YAMNET_SAFETY_INDICES
    if _YAMNET_SAFETY_INDICES:
        return

    try:
        resp = urllib.request.urlopen(_YAMNET_CLASSES_URL, timeout=10)
        text = resp.read().decode("utf-8")
    except Exception:
        # Fallback: hardcoded indices from AudioSet ontology
        _YAMNET_SAFETY_INDICES = {
            "Siren": [394],
            "Car horn, honking": [381],
            "Ambulance (siren)": [395],
            "Police car (siren)": [396],
            "Fire engine, fire truck (siren)": [397],
        }
        logger.warning("Using hardcoded YAMNet safety indices (offline mode)")
        return

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        display_name = row.get("display_name", "")
        idx = int(row.get("index", -1))
        if display_name in _YAMNET_SAFETY_NAMES:
            _YAMNET_SAFETY_INDICES.setdefault(display_name, []).append(idx)

    logger.info(f"Resolved YAMNet safety indices: {_YAMNET_SAFETY_INDICES}")


# ─── Result Dataclass ────────────────────────────────────────────────────────

@dataclass
class V3InferenceResult:
    """Complete V3 inference output."""
    # Classification
    environment: str
    confidence: float
    ensemble_probs: np.ndarray

    # Safety
    safety_verdict: bool         # True = safety triggered
    safety_probability: float    # raw safety model probability
    safety_source: str           # "none" | "safety_model" | "yamnet_raw" | "cnn" | "multi"

    # Temporal
    temporal_status: str         # "Confirmed" | "Uncertain"

    # Risk
    risk_score: float            # 0.0 – 1.0
    risk_action: str             # "apply" | "conservative" | "hold"

    # Transition
    transition_approved: bool
    settings_status: str         # "Applied" | "Held" | "Safety Override" | "Conservative"

    # Recommendation
    volume: int
    noise_reduction: str
    directionality: str
    speech_enhancement: bool
    reasoning: str

    # Per-model detail
    per_model: dict = field(default_factory=dict)

    # Performance
    latency_ms: float = 0.0


# ─── Singleton Engine ────────────────────────────────────────────────────────

class HearSmartV3Engine:
    """
    Stateful inference engine with temporal buffers and transition smoothing.
    Loads models lazily on first call.
    """

    def __init__(self):
        self._models: dict = {}
        self._loaded = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Feature 4: Temporal consistency buffer
        self._prediction_buffer: deque = deque(maxlen=3)

        # Feature 6: Transition smoothing state
        self._transition_counter: dict = {}
        self._current_environment: str = "background_noise"

        # Ensemble weights (2-model: cnn, yamnet)
        self._weights = {"cnn": 0.30, "yamnet": 0.70}

    # ── Model Loading ────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return

        import tensorflow as tf
        import tensorflow_hub as hub

        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)

        logger.info("Loading V3 2-model ensemble + safety classifier ...")

        # YAMNet base (TF Hub) — loaded ONCE, reused everywhere
        self._models["yamnet_base"] = hub.load("https://tfhub.dev/google/yamnet/1")

        # YAMNet classification head (fine-tuned on 8-class)
        self._models["yamnet_head"] = tf.keras.models.load_model("models/yamnet_v2.h5")
        self._models["yamnet_mu"] = np.load("features/yamnet_v2/mu.npy")
        self._models["yamnet_sig"] = np.load("features/yamnet_v2/sig.npy")

        # AudioCNN (PyTorch)
        cnn = AudioCNN_V2(8).to(torch.device(self._device))
        cnn.load_state_dict(torch.load("models/cnn_v2.pt", map_location=self._device))
        cnn.eval()
        self._models["cnn"] = cnn

        # Safety binary classifier
        safety_path = Path("models/safety_classifier.pt")
        safety_model = SafetyClassifier(input_dim=1024)
        if safety_path.exists():
            safety_model.load_state_dict(
                torch.load(str(safety_path), map_location=self._device)
            )
            logger.info("Safety classifier loaded from models/safety_classifier.pt")
        else:
            logger.warning("Safety classifier not found — using untrained weights")
        safety_model.eval()
        self._models["safety"] = safety_model

        # Load optimized ensemble weights if available
        wfile = Path("results/ensemble_weights_v3_2model.json")
        if wfile.exists():
            with open(wfile) as f:
                data = json.load(f)
                self._weights = data.get("weights", self._weights)
        logger.info(f"Ensemble weights: {self._weights}")

        # Resolve YAMNet raw safety indices
        _resolve_yamnet_safety_indices()

        self._loaded = True
        logger.info("V3 engine ready.")

    # ── Feature 3: Triple Safety Fusion ──────────────────────────────────

    def _triple_safety_check(
        self,
        safety_verdict: bool,
        yamnet_raw_scores: np.ndarray,   # (521,) mean pooled
        cnn_probs: np.ndarray,           # (8,) softmax
    ) -> tuple:
        """
        3-condition OR gate for safety detection.

        Returns:
            (is_safety: bool, source: str)
        """
        sources = []

        # Condition 1: Dedicated safety binary classifier
        if safety_verdict:
            sources.append("safety_model")

        # Condition 2: YAMNet raw 521-class scores for siren/horn
        siren_indices = []
        horn_indices = []
        for name, idxs in _YAMNET_SAFETY_INDICES.items():
            if "horn" in name.lower():
                horn_indices.extend(idxs)
            else:
                siren_indices.extend(idxs)

        if siren_indices:
            siren_score = float(max(yamnet_raw_scores[i] for i in siren_indices
                                    if i < len(yamnet_raw_scores)))
            if siren_score > 0.2:
                sources.append("yamnet_raw")

        if horn_indices:
            horn_score = float(max(yamnet_raw_scores[i] for i in horn_indices
                                   if i < len(yamnet_raw_scores)))
            if horn_score > 0.2:
                sources.append("yamnet_raw")

        # Condition 3: AudioCNN 8-class probs for siren/horn
        siren_idx_8 = CLASS_NAMES_V2.index("siren")
        horn_idx_8 = CLASS_NAMES_V2.index("horn")
        if cnn_probs[siren_idx_8] > 0.5 or cnn_probs[horn_idx_8] > 0.5:
            sources.append("cnn")

        is_safety = len(sources) > 0
        if len(sources) > 1:
            source = "multi"
        elif len(sources) == 1:
            source = sources[0]
        else:
            source = "none"

        return is_safety, source

    # ── Feature 4: Temporal Consistency ───────────────────────────────────

    def _check_temporal_consistency(self, new_prediction: str) -> str:
        """
        Sliding window check over last 3 predictions.
        Safety classes bypass — always Confirmed.
        """
        self._prediction_buffer.append(new_prediction)

        # Safety bypass
        if new_prediction in ("siren", "horn"):
            return "Confirmed"

        # Need at least 2/3 agreement
        if self._prediction_buffer.count(new_prediction) >= 2:
            return "Confirmed"

        return "Uncertain"

    # ── Feature 5: Risk Scoring ──────────────────────────────────────────

    def _calculate_risk_score(
        self,
        ensemble_confidence: float,
        top_class: str,
        yamnet_top: str,
        cnn_top: str,
    ) -> tuple:
        """
        Calculate risk score and determine action.

        Returns:
            (risk_score: float, action: str)
        """
        CLASS_RISK_WEIGHTS = {
            "siren": 1.0,
            "horn": 1.0,
            "speech": 0.5,
            "traffic": 0.6,
            "construction": 0.6,
            "music": 0.3,
            "dog_bark": 0.4,
            "background_noise": 0.3,
        }

        class_weight = CLASS_RISK_WEIGHTS.get(top_class, 0.3)
        model_agreement = 1.0 if yamnet_top == cnn_top else 0.5

        risk_score = ensemble_confidence * class_weight * model_agreement
        risk_score = min(1.0, max(0.0, risk_score))

        if risk_score > 0.7:
            action = "apply"
        elif risk_score >= 0.4:
            action = "conservative"
        else:
            action = "hold"

        return risk_score, action

    # ── Feature 6: Transition Smoothing ──────────────────────────────────

    def _smooth_transition(self, confirmed_class: str) -> bool:
        """
        Require 2 consecutive confirmations before changing environment.
        Safety classes bypass — immediate transition.
        """
        if confirmed_class in ("siren", "horn"):
            self._transition_counter.clear()
            return True

        if confirmed_class != self._current_environment:
            self._transition_counter[confirmed_class] = \
                self._transition_counter.get(confirmed_class, 0) + 1

            if self._transition_counter.get(confirmed_class, 0) >= 2:
                self._transition_counter.clear()
                return True
            return False

        # Same as current — already transitioned
        return True

    # ── Main Inference ───────────────────────────────────────────────────

    def infer(self, audio_path: str) -> V3InferenceResult:
        """
        Full V3 inference pipeline.

        Steps:
          1. Single YAMNet call — reuse embeddings everywhere
          2. Parallel model inference (YAMNet head + CNN + Safety)
          3. Triple safety fusion check
          4. Weighted 2-model ensemble
          5. Temporal consistency check
          6. Risk scoring
          7. Transition smoothing
          8. Generate recommendation
        """
        import tensorflow as tf

        self._ensure_loaded()
        t0 = time.perf_counter()

        # ── Step 1: Feature extraction ───────────────────────────────────
        wave = extract_waveform(audio_path)
        mel = extract_mel_spectrogram(audio_path)

        # Single YAMNet call — reuse everywhere
        wav_tf = tf.constant(wave.flatten(), dtype=tf.float32)
        scores_tf, embeddings_tf, _ = self._models["yamnet_base"](wav_tf)

        # Mean-pooled embedding vector (1024-d) — reused by head + safety
        embedding_vector = tf.reduce_mean(embeddings_tf, axis=0).numpy()  # (1024,)

        # YAMNet raw 521-class scores (for triple safety check)
        yamnet_raw_scores = tf.reduce_mean(scores_tf, axis=0).numpy()  # (521,)

        # ── Step 2: Parallel model inference ─────────────────────────────

        # YAMNet classification head (8-class)
        mu = self._models["yamnet_mu"]
        sig = self._models["yamnet_sig"]
        emb_norm = ((embedding_vector.reshape(1, -1) - mu) / sig).astype(np.float32)
        yamnet_probs = self._models["yamnet_head"].predict(
            emb_norm, verbose=0
        )[0].astype(np.float32)  # (8,)

        # AudioCNN (8-class)
        mel_t = torch.from_numpy(
            mel.transpose(2, 0, 1)[None].astype(np.float32)
        ).to(self._device)
        with torch.no_grad():
            cnn_probs = torch_F.softmax(
                self._models["cnn"](mel_t), dim=1
            ).cpu().numpy()[0].astype(np.float32)  # (8,)

        # Safety binary classifier (reuses same embedding_vector)
        safety_prob = self._models["safety"].predict_proba(embedding_vector)
        safety_verdict = safety_prob >= 0.5

        # ── Step 3: Triple safety fusion ─────────────────────────────────
        is_safety, safety_source = self._triple_safety_check(
            safety_verdict, yamnet_raw_scores, cnn_probs
        )

        if is_safety:
            # Determine which safety class (siren or horn)
            siren_score = yamnet_probs[CLASS_NAMES_V2.index("siren")]
            horn_score = yamnet_probs[CLASS_NAMES_V2.index("horn")]
            safety_class = "siren" if siren_score >= horn_score else "horn"

            self._current_environment = safety_class
            self._prediction_buffer.append(safety_class)

            rec = RECOMMENDATION_TABLE_V2.get(safety_class, RECOMMENDATION_TABLE_V2["siren"])
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            return V3InferenceResult(
                environment=safety_class,
                confidence=float(max(siren_score, horn_score)),
                ensemble_probs=yamnet_probs,
                safety_verdict=True,
                safety_probability=safety_prob,
                safety_source=safety_source,
                temporal_status="Confirmed",
                risk_score=1.0,
                risk_action="apply",
                transition_approved=True,
                settings_status="Safety Override",
                volume=9,
                noise_reduction="Off",
                directionality="Omnidirectional",
                speech_enhancement=False,
                reasoning=f"⚠️ SAFETY OVERRIDE — {safety_class.title()} detected "
                          f"via {safety_source}. Maximum awareness settings applied.",
                per_model={
                    "YAMNet": (CLASS_NAMES_V2[int(yamnet_probs.argmax())],
                               float(yamnet_probs.max()), self._weights.get("yamnet", 0.7)),
                    "CNN": (CLASS_NAMES_V2[int(cnn_probs.argmax())],
                            float(cnn_probs.max()), self._weights.get("cnn", 0.3)),
                },
                latency_ms=latency_ms,
            )

        # ── Step 4: Weighted 2-model ensemble ────────────────────────────
        w_cnn = self._weights.get("cnn", 0.30)
        w_yam = self._weights.get("yamnet", 0.70)
        total_w = w_cnn + w_yam
        w_cnn /= total_w
        w_yam /= total_w

        ensemble_probs = (w_yam * yamnet_probs + w_cnn * cnn_probs)
        top_idx = int(ensemble_probs.argmax())
        top_class = CLASS_NAMES_V2[top_idx]
        ensemble_confidence = float(ensemble_probs[top_idx])

        yamnet_top = CLASS_NAMES_V2[int(yamnet_probs.argmax())]
        cnn_top = CLASS_NAMES_V2[int(cnn_probs.argmax())]

        # ── Step 5: Temporal consistency ─────────────────────────────────
        temporal_status = self._check_temporal_consistency(top_class)

        # ── Step 6: Risk scoring ─────────────────────────────────────────
        risk_score, risk_action = self._calculate_risk_score(
            ensemble_confidence, top_class, yamnet_top, cnn_top
        )

        # ── Step 7: Transition smoothing ─────────────────────────────────
        transition_ok = self._smooth_transition(top_class)

        # ── Step 8: Generate recommendation ──────────────────────────────
        rec = get_recommendation(top_class)

        if temporal_status == "Uncertain":
            settings_status = "Held"
            reasoning = (f"Temporal check: {top_class.replace('_', ' ').title()} "
                        f"not yet confirmed (need 2/3 agreement). Holding current settings.")
            # Hold current settings
            current_rec = get_recommendation(self._current_environment)
            volume = current_rec["volume"]
            noise_reduction = current_rec["noise_reduction"]
            directionality = current_rec["directionality"]
            speech_enhancement = current_rec["speech_enhancement"]
        elif risk_action == "hold":
            settings_status = "Held"
            reasoning = (f"Risk score {risk_score:.2f} too low — "
                        f"holding current settings for stability.")
            current_rec = get_recommendation(self._current_environment)
            volume = current_rec["volume"]
            noise_reduction = current_rec["noise_reduction"]
            directionality = current_rec["directionality"]
            speech_enhancement = current_rec["speech_enhancement"]
        elif not transition_ok:
            settings_status = "Held"
            reasoning = (f"Transition to {top_class.replace('_', ' ').title()} "
                        f"pending — need 2 consecutive confirmations.")
            current_rec = get_recommendation(self._current_environment)
            volume = current_rec["volume"]
            noise_reduction = current_rec["noise_reduction"]
            directionality = current_rec["directionality"]
            speech_enhancement = current_rec["speech_enhancement"]
        elif risk_action == "conservative":
            settings_status = "Conservative"
            reasoning = (f"Risk score {risk_score:.2f} — "
                        f"conservative settings applied (volume changes halved).")
            # Apply with halved volume delta
            default_vol = 5
            full_vol = rec["volume"]
            volume = default_vol + (full_vol - default_vol) // 2
            noise_reduction = rec["noise_reduction"]
            directionality = rec["directionality"]
            speech_enhancement = rec["speech_enhancement"]
            self._current_environment = top_class
        else:
            settings_status = "Applied"
            reasoning = rec["reasoning"]
            volume = rec["volume"]
            noise_reduction = rec["noise_reduction"]
            directionality = rec["directionality"]
            speech_enhancement = rec["speech_enhancement"]
            self._current_environment = top_class

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return V3InferenceResult(
            environment=top_class,
            confidence=ensemble_confidence,
            ensemble_probs=ensemble_probs,
            safety_verdict=False,
            safety_probability=safety_prob,
            safety_source="none",
            temporal_status=temporal_status,
            risk_score=risk_score,
            risk_action=risk_action,
            transition_approved=transition_ok,
            settings_status=settings_status,
            volume=volume,
            noise_reduction=noise_reduction,
            directionality=directionality,
            speech_enhancement=speech_enhancement,
            reasoning=reasoning,
            per_model={
                "YAMNet": (yamnet_top, float(yamnet_probs.max()), w_yam),
                "CNN": (cnn_top, float(cnn_probs.max()), w_cnn),
            },
            latency_ms=latency_ms,
        )

    def reset_buffers(self):
        """Reset temporal and transition state (e.g. for testing)."""
        self._prediction_buffer.clear()
        self._transition_counter.clear()
        self._current_environment = "background_noise"


# ─── Module-level singleton ──────────────────────────────────────────────────
_engine: Optional[HearSmartV3Engine] = None


def get_engine() -> HearSmartV3Engine:
    """Return the global V3 engine singleton (lazy init)."""
    global _engine
    if _engine is None:
        _engine = HearSmartV3Engine()
    return _engine
