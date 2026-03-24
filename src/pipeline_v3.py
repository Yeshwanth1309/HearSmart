"""
V3 Inference Pipeline — HearSmart Engine (Hardened).

All 6 V3 features + 5 safety hardening measures:
  H1: Calibrated YAMNet class index mapping (SIREN/HORN/TRAFFIC constants)
  H2: Negative class guard — traffic indices suppress siren/horn scores
  H3: Minimum duration filter — SAFETY_FRAMES_REQUIRED consecutive frames
  H4: Per-class confidence floors before settings are applied
  H5: Dog Bark vs Horn disambiguation using raw YAMNet scores

Single YAMNet call per inference — embeddings reused everywhere.
"""
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as torch_F

from src.data.label_map_v2 import (
    CLASS_NAMES_V2,
    RECOMMENDATION_TABLE_V2,
    get_recommendation,
)
from src.features.extractor import extract_mel_spectrogram, extract_waveform
from src.models_v2 import AudioCNN_V2
from src.models.safety_model import SafetyClassifier
from src.safety_config import (
    SIREN_YAMNET_INDICES,
    HORN_YAMNET_INDICES,
    TRAFFIC_YAMNET_INDICES,
    DOG_YAMNET_INDICES,
    YAMNET_SIREN_THRESHOLD,
    YAMNET_HORN_THRESHOLD,
    CNN_SIREN_THRESHOLD,
    CNN_HORN_THRESHOLD,
    SAFETY_CLASSIFIER_THRESHOLD,
    SAFETY_CLASSIFIER_HIGH_CONF,
    SAFETY_FRAMES_REQUIRED,
    TRAFFIC_NEGATIVE_GUARD_REDUCTION,
    CONFIDENCE_FLOORS,
)

logger = logging.getLogger(__name__)

# ─── Result Dataclass ────────────────────────────────────────────────────────

@dataclass
class V3InferenceResult:
    """Complete V3 inference output."""
    environment: str
    confidence: float
    ensemble_probs: np.ndarray

    safety_verdict: bool
    safety_probability: float
    safety_source: str

    temporal_status: str
    risk_score: float
    risk_action: str

    transition_approved: bool
    settings_status: str

    volume: int
    noise_reduction: str
    directionality: str
    speech_enhancement: bool
    reasoning: str

    # Debug info
    yamnet_raw_top3: list = field(default_factory=list)
    cnn_top3: list = field(default_factory=list)
    safety_frame_count: int = 0
    confidence_floor_blocked: bool = False

    per_model: dict = field(default_factory=dict)
    latency_ms: float = 0.0


# ─── Engine ──────────────────────────────────────────────────────────────────

class HearSmartV3Engine:
    """
    Stateful hardened inference engine.
    Loads models lazily on first call.
    """

    def __init__(self):
        self._models: dict = {}
        self._loaded = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Feature 4: Temporal consistency buffer (last 3 predictions)
        self._prediction_buffer: deque = deque(maxlen=3)

        # Feature 6: Transition smoothing
        self._transition_counter: dict = {}
        self._current_environment: str = "background_noise"

        # Ensemble weights
        self._weights = {"cnn": 0.30, "yamnet": 0.70}

        # H3: Safety frame counter
        self._safety_frame_counter: int = 0

    # ── Model Loading ─────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return

        import tensorflow as tf
        import tensorflow_hub as hub

        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)

        logger.info("Loading V3 hardened engine ...")

        self._models["yamnet_base"] = hub.load("https://tfhub.dev/google/yamnet/1")
        self._models["yamnet_head"] = tf.keras.models.load_model("models/yamnet_v2.h5")
        self._models["yamnet_mu"]   = np.load("features/yamnet_v2/mu.npy")
        self._models["yamnet_sig"]  = np.load("features/yamnet_v2/sig.npy")

        cnn = AudioCNN_V2(8).to(torch.device(self._device))
        cnn.load_state_dict(
            torch.load("models/cnn_v2.pt", map_location=self._device,
                       weights_only=False)
        )
        cnn.eval()
        self._models["cnn"] = cnn

        safety_path = Path("models/safety_classifier.pt")
        safety_model = SafetyClassifier(input_dim=1024)
        if safety_path.exists():
            safety_model.load_state_dict(
                torch.load(str(safety_path), map_location=self._device,
                           weights_only=False)
            )
            logger.info("Safety classifier loaded.")
        else:
            logger.warning("Safety classifier not found — using untrained weights.")
        safety_model.eval()
        self._models["safety"] = safety_model

        wfile = Path("results/ensemble_weights_v3_2model.json")
        if wfile.exists():
            with open(wfile) as f:
                self._weights = json.load(f).get("weights", self._weights)
        logger.info(f"Weights: {self._weights}")

        self._loaded = True
        logger.info("V3 hardened engine ready.")

    # ── H1+H2: Calibrated Safety Score Extraction ────────────────────────

    def _extract_safety_scores(self, yamnet_raw: np.ndarray) -> dict:
        """
        Extract siren/horn scores from YAMNet 521-class output.
        Applies H2 traffic negative guard.
        """
        n = len(yamnet_raw)

        # Raw siren and horn scores
        siren_score = float(max(
            (yamnet_raw[i] for i in SIREN_YAMNET_INDICES if i < n), default=0.0
        ))
        horn_score = float(max(
            (yamnet_raw[i] for i in HORN_YAMNET_INDICES if i < n), default=0.0
        ))

        # H2: Traffic negative guard
        # If YAMNet's top raw class is a traffic class, downweight safety scores
        top_raw_idx = int(np.argmax(yamnet_raw))
        traffic_guard_active = top_raw_idx in TRAFFIC_YAMNET_INDICES
        if traffic_guard_active:
            siren_score *= (1.0 - TRAFFIC_NEGATIVE_GUARD_REDUCTION)
            horn_score  *= (1.0 - TRAFFIC_NEGATIVE_GUARD_REDUCTION)
            logger.debug(f"Traffic guard active (idx={top_raw_idx}), "
                        f"safety scores reduced by {TRAFFIC_NEGATIVE_GUARD_REDUCTION:.0%}")

        # Dog score for disambiguation
        dog_score = float(max(
            (yamnet_raw[i] for i in DOG_YAMNET_INDICES if i < n), default=0.0
        ))

        return {
            "siren": siren_score,
            "horn":  horn_score,
            "dog":   dog_score,
            "traffic_guard": traffic_guard_active,
            "top_raw_idx": top_raw_idx,
        }

    # ── H3: Safety Duration Filter ────────────────────────────────────────

    def _safety_duration_check(
        self,
        safety_classifier_prob: float,
        safety_scores: dict,
        cnn_probs: np.ndarray,
        yamnet_probs: np.ndarray,       # NEW: fine-tuned 8-class head output
    ) -> tuple:
        """
        Require SAFETY_FRAMES_REQUIRED consecutive frames of safety detection.
        Exception: immediate trigger if safety_classifier_prob > HIGH_CONF threshold.

        Condition 4 (real-audio fix): YAMNet fine-tuned 8-class head confidently
        predicting siren/horn is a strong signal — include it in fusion.
        Real sirens score 72-98% on the fine-tuned head even when raw 521-class
        scores are ambiguous.

        Returns:
            (trigger_now: bool, source: str, raw_detected: bool)
        """
        siren_idx_8 = CLASS_NAMES_V2.index("siren")
        horn_idx_8  = CLASS_NAMES_V2.index("horn")

        # Determine raw detection (pre-filter)
        sources = []

        # Condition 1: safety binary classifier
        if safety_classifier_prob >= SAFETY_CLASSIFIER_THRESHOLD:
            sources.append("safety_model")

        # Condition 2: YAMNet raw 521-class scores with calibrated thresholds
        if safety_scores["siren"] > YAMNET_SIREN_THRESHOLD:
            sources.append("yamnet_siren")
        if safety_scores["horn"] > YAMNET_HORN_THRESHOLD:
            sources.append("yamnet_horn")

        # Condition 3: CNN 8-class probs (raised to 0.65 post-audit to stop
        # church-bells false alarms — CNN fires on bell harmonics at 0.55)
        if cnn_probs[siren_idx_8] > CNN_SIREN_THRESHOLD:
            sources.append("cnn_siren")
        if cnn_probs[horn_idx_8] > CNN_HORN_THRESHOLD:
            sources.append("cnn_horn")

        # Condition 4: YAMNet fine-tuned 8-class HEAD (real-audio fix)
        # The fine-tuned head was trained on UrbanSound8K + ESC-50 real audio.
        # When it predicts siren/horn with >= 50% confidence, that's a reliable
        # signal — use a LOWER threshold than CNN (0.50 vs 0.65) because the
        # YAMNet head specialises on our exact 8-class taxonomy.
        YAMNET_HEAD_SAFETY_THRESHOLD = 0.50
        if yamnet_probs[siren_idx_8] >= YAMNET_HEAD_SAFETY_THRESHOLD:
            sources.append("yamnet_head_siren")
        if yamnet_probs[horn_idx_8] >= YAMNET_HEAD_SAFETY_THRESHOLD:
            sources.append("yamnet_head_horn")

        raw_detected = len(sources) > 0

        # Update frame counter
        if raw_detected:
            self._safety_frame_counter += 1
        else:
            self._safety_frame_counter = max(0, self._safety_frame_counter - 1)

        source_str = "multi" if len(sources) > 1 else (sources[0] if sources else "none")

        # H3 immediate bypass: very high safety classifier confidence
        if safety_classifier_prob >= SAFETY_CLASSIFIER_HIGH_CONF:
            logger.debug(f"Safety high-conf bypass (prob={safety_classifier_prob:.3f})")
            return True, "safety_model_highconf", raw_detected

        # Normal frame-count gate
        if self._safety_frame_counter >= SAFETY_FRAMES_REQUIRED:
            return True, source_str, raw_detected

        return False, source_str, raw_detected

    # ── H5: Dog Bark vs Horn Disambiguation ──────────────────────────────

    def _disambiguate_bark_vs_horn(
        self,
        ensemble_class: str,
        ensemble_confidence: float,
        safety_scores: dict,
        cnn_probs: np.ndarray,
    ) -> str:
        """
        Prevent dangerous Dog Bark ↔ Horn confusion.
        - If ensemble says Dog Bark but horn YAMNet score is high → reclassify Horn
        - If ensemble says Horn but dog YAMNet score is high and confidence is low
          → reclassify Dog Bark
        """
        siren_idx_8 = CLASS_NAMES_V2.index("siren")
        horn_idx_8  = CLASS_NAMES_V2.index("horn")

        if ensemble_class == "dog_bark":
            horn_yamnet = safety_scores["horn"]
            if horn_yamnet > 0.25:
                logger.debug(f"Dog→Horn reclassify: horn_yamnet={horn_yamnet:.3f}")
                return "horn"

        if ensemble_class == "horn":
            dog_yamnet = safety_scores["dog"]
            if dog_yamnet > 0.40 and ensemble_confidence < 0.60:
                logger.debug(f"Horn→Dog reclassify: dog_yamnet={dog_yamnet:.3f}, "
                            f"conf={ensemble_confidence:.3f}")
                return "dog_bark"

        return ensemble_class

    # ── Feature 4: Temporal Consistency ──────────────────────────────────

    def _check_temporal_consistency(self, new_prediction: str) -> str:
        self._prediction_buffer.append(new_prediction)
        if new_prediction in ("siren", "horn"):
            return "Confirmed"
        if self._prediction_buffer.count(new_prediction) >= 2:
            return "Confirmed"
        return "Uncertain"

    # ── Feature 5: Risk Scoring ───────────────────────────────────────────

    def _calculate_risk_score(
        self,
        ensemble_confidence: float,
        top_class: str,
        yamnet_top: str,
        cnn_top: str,
    ) -> tuple:
        CLASS_RISK_WEIGHTS = {
            "siren": 1.0, "horn": 1.0,
            "speech": 0.5, "traffic": 0.6,
            "construction": 0.6, "music": 0.3,
            "dog_bark": 0.4, "background_noise": 0.3,
        }
        class_weight = CLASS_RISK_WEIGHTS.get(top_class, 0.3)
        model_agreement = 1.0 if yamnet_top == cnn_top else 0.5
        risk_score = min(1.0, max(0.0,
            ensemble_confidence * class_weight * model_agreement
        ))
        action = "apply" if risk_score > 0.7 else ("conservative" if risk_score >= 0.4 else "hold")
        return risk_score, action

    # ── Feature 6: Transition Smoothing ──────────────────────────────────

    def _smooth_transition(self, confirmed_class: str) -> bool:
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
        return True

    # ── Main Inference ────────────────────────────────────────────────────

    def infer(self, audio_path: str) -> V3InferenceResult:
        """Full hardened V3 inference pipeline."""
        import tensorflow as tf

        self._ensure_loaded()
        t0 = time.perf_counter()

        # Step 1: Feature extraction
        wave = extract_waveform(audio_path)
        mel  = extract_mel_spectrogram(audio_path)

        wav_tf = tf.constant(wave.flatten(), dtype=tf.float32)
        scores_tf, embeddings_tf, _ = self._models["yamnet_base"](wav_tf)

        embedding_vector  = tf.reduce_mean(embeddings_tf, axis=0).numpy()   # (1024,)
        yamnet_raw_scores = tf.reduce_mean(scores_tf, axis=0).numpy()        # (521,)

        # Step 2: Parallel model inference
        mu, sig = self._models["yamnet_mu"], self._models["yamnet_sig"]
        emb_norm = ((embedding_vector.reshape(1, -1) - mu) / sig).astype(np.float32)
        yamnet_probs = self._models["yamnet_head"].predict(
            emb_norm, verbose=0
        )[0].astype(np.float32)

        mel_t = torch.from_numpy(
            mel.transpose(2, 0, 1)[None].astype(np.float32)
        ).to(self._device)
        with torch.no_grad():
            cnn_probs = torch_F.softmax(
                self._models["cnn"](mel_t), dim=1
            ).cpu().numpy()[0].astype(np.float32)

        safety_prob = self._models["safety"].predict_proba(embedding_vector)

        # H1+H2: Extract calibrated safety scores with traffic guard
        safety_scores = self._extract_safety_scores(yamnet_raw_scores)

        # Step 3: H3 Safety duration check with all 4 conditions
        trigger_safety, safety_source, raw_safety = self._safety_duration_check(
            safety_prob, safety_scores, cnn_probs, yamnet_probs
        )

        # Debug: YAMNet raw top-3 and CNN top-3
        raw_top3_idx = np.argsort(yamnet_raw_scores)[::-1][:3]
        yamnet_raw_top3 = [(int(i), float(yamnet_raw_scores[i])) for i in raw_top3_idx]
        cnn_top3_idx = np.argsort(cnn_probs)[::-1][:3]
        cnn_top3 = [(CLASS_NAMES_V2[i], float(cnn_probs[i])) for i in cnn_top3_idx]

        if trigger_safety:
            siren_score_8 = yamnet_probs[CLASS_NAMES_V2.index("siren")]
            horn_score_8  = yamnet_probs[CLASS_NAMES_V2.index("horn")]
            safety_class  = "siren" if siren_score_8 >= horn_score_8 else "horn"

            # Also check raw scores for disambiguation
            if safety_scores["siren"] > safety_scores["horn"]:
                safety_class = "siren"
            else:
                safety_class = "horn"

            self._current_environment = safety_class
            self._prediction_buffer.append(safety_class)

            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return V3InferenceResult(
                environment=safety_class,
                confidence=float(max(siren_score_8, horn_score_8)),
                ensemble_probs=yamnet_probs,
                safety_verdict=True,
                safety_probability=safety_prob,
                safety_source=safety_source,
                temporal_status="Confirmed",
                risk_score=1.0,
                risk_action="apply",
                transition_approved=True,
                settings_status="Safety Override",
                volume=9, noise_reduction="Off",
                directionality="Omnidirectional",
                speech_enhancement=False,
                reasoning=(f"SAFETY OVERRIDE [{safety_class.upper()}] "
                          f"via {safety_source}. "
                          f"siren_raw={safety_scores['siren']:.3f} "
                          f"horn_raw={safety_scores['horn']:.3f} "
                          f"classifier={safety_prob:.3f}"),
                yamnet_raw_top3=yamnet_raw_top3,
                cnn_top3=cnn_top3,
                safety_frame_count=self._safety_frame_counter,
                per_model={
                    "YAMNet": (CLASS_NAMES_V2[int(yamnet_probs.argmax())],
                               float(yamnet_probs.max()),
                               self._weights.get("yamnet", 0.7)),
                    "CNN":    (CLASS_NAMES_V2[int(cnn_probs.argmax())],
                               float(cnn_probs.max()),
                               self._weights.get("cnn", 0.3)),
                },
                latency_ms=latency_ms,
            )

        # Step 4: 2-model weighted ensemble
        w_cnn = self._weights.get("cnn", 0.30)
        w_yam = self._weights.get("yamnet", 0.70)
        total_w = w_cnn + w_yam
        w_cnn /= total_w; w_yam /= total_w

        ensemble_probs    = w_yam * yamnet_probs + w_cnn * cnn_probs
        top_idx           = int(ensemble_probs.argmax())
        top_class         = CLASS_NAMES_V2[top_idx]
        ensemble_conf     = float(ensemble_probs[top_idx])
        yamnet_top        = CLASS_NAMES_V2[int(yamnet_probs.argmax())]
        cnn_top_name      = CLASS_NAMES_V2[int(cnn_probs.argmax())]

        # H5: Dog Bark vs Horn disambiguation (BEFORE temporal check)
        top_class = self._disambiguate_bark_vs_horn(
            top_class, ensemble_conf, safety_scores, cnn_probs
        )

        # H4: Confidence floor check
        floor = CONFIDENCE_FLOORS.get(top_class, 0.40)
        conf_floor_blocked = ensemble_conf < floor

        # Step 5: Temporal consistency
        temporal_status = self._check_temporal_consistency(top_class)

        # Step 6: Risk scoring
        risk_score, risk_action = self._calculate_risk_score(
            ensemble_conf, top_class, yamnet_top, cnn_top_name
        )

        # Step 7: Transition smoothing
        transition_ok = self._smooth_transition(top_class)

        # Step 8: Recommendation logic
        rec = get_recommendation(top_class)

        def _held_settings():
            cur = get_recommendation(self._current_environment)
            return cur["volume"], cur["noise_reduction"], \
                   cur["directionality"], cur["speech_enhancement"]

        if conf_floor_blocked:
            settings_status = "Held"
            reasoning = (f"Confidence {ensemble_conf:.1%} below floor "
                        f"{floor:.0%} for {top_class} — holding settings.")
            volume, noise_reduction, directionality, speech_enhancement = _held_settings()

        elif temporal_status == "Uncertain":
            settings_status = "Held"
            reasoning = (f"Temporal: {top_class} not yet confirmed "
                        f"(need 2/3 agreement).")
            volume, noise_reduction, directionality, speech_enhancement = _held_settings()

        elif risk_action == "hold":
            settings_status = "Held"
            reasoning = f"Risk {risk_score:.2f} too low — holding."
            volume, noise_reduction, directionality, speech_enhancement = _held_settings()

        elif not transition_ok:
            settings_status = "Held"
            reasoning = (f"Transition to {top_class} pending "
                        f"(need 2 consecutive confirmations).")
            volume, noise_reduction, directionality, speech_enhancement = _held_settings()

        elif risk_action == "conservative":
            settings_status = "Conservative"
            reasoning = f"Risk {risk_score:.2f} — conservative (halved delta)."
            default_vol = 5
            volume = default_vol + (rec["volume"] - default_vol) // 2
            noise_reduction  = rec["noise_reduction"]
            directionality   = rec["directionality"]
            speech_enhancement = rec["speech_enhancement"]
            self._current_environment = top_class

        else:
            settings_status = "Applied"
            reasoning = rec["reasoning"]
            volume      = rec["volume"]
            noise_reduction  = rec["noise_reduction"]
            directionality   = rec["directionality"]
            speech_enhancement = rec["speech_enhancement"]
            self._current_environment = top_class

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return V3InferenceResult(
            environment=top_class,
            confidence=ensemble_conf,
            ensemble_probs=ensemble_probs,
            safety_verdict=False,
            safety_probability=safety_prob,
            safety_source="none",
            temporal_status=temporal_status,
            risk_score=risk_score,
            risk_action=risk_action,
            transition_approved=transition_ok,
            settings_status=settings_status,
            confidence_floor_blocked=conf_floor_blocked,
            volume=volume,
            noise_reduction=noise_reduction,
            directionality=directionality,
            speech_enhancement=speech_enhancement,
            reasoning=reasoning,
            yamnet_raw_top3=yamnet_raw_top3,
            cnn_top3=cnn_top3,
            safety_frame_count=self._safety_frame_counter,
            per_model={
                "YAMNet": (yamnet_top, float(yamnet_probs.max()), w_yam),
                "CNN":    (cnn_top_name, float(cnn_probs.max()), w_cnn),
            },
            latency_ms=latency_ms,
        )

    def reset_buffers(self):
        """Reset all stateful buffers."""
        self._prediction_buffer.clear()
        self._transition_counter.clear()
        self._current_environment = "background_noise"
        self._safety_frame_counter = 0


# ─── Singleton ───────────────────────────────────────────────────────────────
_engine: Optional[HearSmartV3Engine] = None

def get_engine() -> HearSmartV3Engine:
    global _engine
    if _engine is None:
        _engine = HearSmartV3Engine()
    return _engine
