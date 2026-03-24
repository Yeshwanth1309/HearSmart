"""
Phase 9 — FastAPI Inference Server.

Endpoints
---------
POST /predict          — upload audio file → classification + recommendation
GET  /health           — liveness probe
GET  /models           — model metadata
GET  /classes          — supported class names
GET  /rules            — recommendation rule-set JSON

Usage
-----
From project root:
    uvicorn demo.api:app --host 0.0.0.0 --port 8000 --reload

Swagger UI: http://localhost:8000/docs
"""

import io
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import torch
import torch.nn.functional as torch_F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Use the CORRECT extractor that matches training (n_mels=128, 128x128 output)
from src.features.extractor import (
    extract_mel_spectrogram,
    extract_mfcc,
    extract_waveform,
)
from src.models import AudioCNN
from src.recommendations import (
    export_rules,
    recommend_from_probs,
)
from src.utils import setup_logging, set_seed

# ─────────────────────────────────────────────────────────────────────────────
# App config
# ─────────────────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)
set_seed(42)

app = FastAPI(
    title="Hearing Aid Audio Classification API",
    description=(
        "AI-powered environmental sound classification and hearing aid "
        "parameter recommendation system.\n\n"
        "**Model**: 5-model weighted ensemble "
        "(RF + SVM + XGBoost + CNN + YAMNet)\n\n"
        "**Performance**: 95.04% accuracy, 95.48% Macro F1 (UrbanSound8K test set)\n\n"
        "**Latency**: ~350ms per prediction"
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global model state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
MODELS: dict = {}

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]

WEIGHTS: dict = {}   # loaded from ensemble_weights.json
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_all_models() -> None:
    """Load all 5 models + label encoder + ensemble weights at startup."""
    import json
    import tensorflow as tf
    import tensorflow_hub as hub

    global WEIGHTS

    logger.info("Loading all models for API serving...")

    # Label encoder
    MODELS["label_encoder"] = joblib.load("models/label_encoder.pkl")

    # Traditional ML
    MODELS["rf"]  = joblib.load("models/random_forest.pkl")
    MODELS["svm"] = joblib.load("models/svm.pkl")

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model("models/xgboost.json")
    MODELS["xgb"] = xgb

    # CNN
    cnn = AudioCNN(num_classes=10).to(torch.device(_DEVICE))
    cnn.load_state_dict(
        torch.load("models/cnn_best.pt", map_location=torch.device(_DEVICE))
    )
    cnn.eval()
    MODELS["cnn"] = cnn

    # YAMNet
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    MODELS["yamnet_base"]  = hub.load("https://tfhub.dev/google/yamnet/1")
    MODELS["yamnet_head"]  = tf.keras.models.load_model("models/yamnet_head.h5")

    # YAMNet normalisation stats
    E_train = np.load("features/yamnet_embeddings/train.npy")
    MODELS["yamnet_mean"] = E_train.mean(axis=0, keepdims=True)
    MODELS["yamnet_std"]  = E_train.std(axis=0, keepdims=True) + 1e-8

    # Ensemble weights
    with open("results/ensemble_weights.json") as f:
        WEIGHTS = json.load(f)["weights"]

    logger.info(f"All models loaded. Device: {_DEVICE}")
    logger.info(f"Ensemble weights: {WEIGHTS}")


@app.on_event("startup")
async def startup_event():
    _load_all_models()


# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _run_inference(audio_path: str) -> dict:
    """
    Full inference pipeline on a single audio file.

    Uses src.features.extractor which produces:
      - MFCC: (80,)         — same as training
      - Mel:  (128, 128, 1) — same as training cache
      - Wave: (48000,)       — same as training

    Returns dict with probs, prediction, confidence, latency.
    """
    import tensorflow as tf

    t_start = time.perf_counter()

    # Extract features using the CORRECT extractor (matches training)
    mfcc = extract_mfcc(audio_path).reshape(1, -1)          # (1, 80)
    wave = extract_waveform(audio_path)                     # (48000,)
    mel  = extract_mel_spectrogram(audio_path)              # (128, 128, 1)

    # CNN expects (batch, channels, H, W) = (1, 1, 128, 128)
    mel_chw = mel.transpose(2, 0, 1)                        # (1, 128, 128)
    mel_batch = mel_chw[np.newaxis].astype(np.float32)      # (1, 1, 128, 128)

    # Gather per-model probs
    probs = {}

    # RF
    probs["rf"] = MODELS["rf"].predict_proba(mfcc).astype(np.float32)

    # SVM
    try:
        probs["svm"] = MODELS["svm"].predict_proba(mfcc).astype(np.float32)
    except Exception:
        scaler = MODELS["svm"].named_steps["scaler"]
        svc    = MODELS["svm"].named_steps["svm"]
        scores = svc.decision_function(scaler.transform(mfcc)).astype(np.float32)
        scores -= scores.max(axis=1, keepdims=True)
        probs["svm"] = (np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True))

    # XGBoost
    probs["xgb"] = MODELS["xgb"].predict_proba(mfcc).astype(np.float32)

    # CNN
    with torch.no_grad():
        mel_t = torch.from_numpy(mel_batch).to(torch.device(_DEVICE))
        probs["cnn"] = (
            torch_F.softmax(MODELS["cnn"](mel_t), dim=1).cpu().numpy().astype(np.float32)
        )

    # YAMNet
    wav_tf  = tf.constant(wave.flatten(), dtype=tf.float32)
    _, emb, _ = MODELS["yamnet_base"](wav_tf)
    emb_pool  = tf.reduce_mean(emb, axis=0, keepdims=True).numpy()
    emb_norm  = ((emb_pool - MODELS["yamnet_mean"]) / MODELS["yamnet_std"]).astype(np.float32)
    probs["yamnet"] = MODELS["yamnet_head"].predict(emb_norm, verbose=0).astype(np.float32)

    # Weighted ensemble
    order = ["rf", "svm", "xgb", "cnn", "yamnet"]
    w = np.array([WEIGHTS[n] for n in order], dtype=np.float32)
    P = np.stack([probs[n][0] for n in order], axis=0)
    ensemble_probs = (P * w[:, None]).sum(axis=0)

    pred_idx    = int(np.argmax(ensemble_probs))
    confidence  = float(ensemble_probs[pred_idx])
    class_name  = CLASS_NAMES[pred_idx]

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "class_name":         class_name,
        "class_id":           pred_idx,
        "confidence":         round(confidence, 6),
        "ensemble_probs":     {c: round(float(p), 6) for c, p in zip(CLASS_NAMES, ensemble_probs)},
        "model_contributions": {
            name: round(float(probs[name][0][pred_idx]), 6) for name in order
        },
        "latency_ms":         latency_ms,
        "is_uncertain":       confidence < 0.35,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    is_uncertain: bool
    ensemble_probs: dict
    model_contributions: dict


class RecommendationResult(BaseModel):
    volume: int
    noise_reduction: str
    directionality: str
    speech_enhancement: bool
    reasoning: str
    tiers_applied: list
    is_safety_override: bool
    suggested_follow_up: str


class PredictResponse(BaseModel):
    prediction: PredictionResult
    recommendation: RecommendationResult
    metadata: dict


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns 200 if all models are loaded."""
    loaded = list(MODELS.keys())
    return {
        "status":       "healthy" if len(loaded) >= 7 else "loading",
        "models_loaded": loaded,
        "device":        _DEVICE,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


@app.get("/classes", tags=["System"])
def get_classes():
    """Return supported environment class names."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}


@app.get("/models", tags=["System"])
def get_model_info():
    """Return loaded model metadata and ensemble weights."""
    return {
        "models": {
            "random_forest": {"type": "RandomForestClassifier", "features": "MFCC (80)"},
            "svm":           {"type": "Pipeline(StandardScaler + SVC)", "features": "MFCC (80)"},
            "xgboost":       {"type": "XGBClassifier", "features": "MFCC (80)"},
            "cnn":           {"type": "AudioCNN (4-block Conv2D)", "features": "Mel-spectrogram (128×128)"},
            "yamnet":        {"type": "YAMNet + Dense head", "features": "Waveform (48000) → embed (1024)"},
        },
        "ensemble_weights": WEIGHTS,
        "ensemble_accuracy": 0.9366,
        "ensemble_macro_f1": 0.9397,
        "device": _DEVICE,
    }


@app.get("/rules", tags=["System"])
def get_rules():
    """Return complete recommendation rule-set as JSON."""
    rules_path = "results/recommendation_rules.json"
    if not os.path.exists(rules_path):
        export_rules(rules_path)
    import json
    with open(rules_path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    audio_file: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac, .ogg)"),
    snr_db: Optional[float] = Form(None, description="Optional SNR in dB for acoustic refinement"),
    rms_loudness: Optional[float] = Form(None, description="Optional RMS loudness [0–1]"),
    speech_prob: Optional[float] = Form(None, description="Optional speech probability [0–1]"),
    hearing_loss: Optional[str] = Form(None, description="User hearing loss: mild | moderate | severe"),
    tinnitus: Optional[bool] = Form(None, description="User has tinnitus"),
    age_group: Optional[str] = Form(None, description="User age group: young | adult | elderly"),
    preference: Optional[str] = Form(None, description="User preference: speech | music | balanced"),
):
    """
    **Main inference endpoint.**

    Upload an audio file to receive:
    - Environment classification (one of 10 UrbanSound8K classes)
    - Ensemble confidence score
    - Per-model probability contributions
    - Hearing aid parameter recommendation (4-tier)

    **Supported formats**: WAV, MP3, FLAC, OGG (any sample rate → resampled to 16 kHz)
    """
    # Validate file type
    ext = Path(audio_file.filename or "").suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Use .wav, .mp3, .flac, or .ogg",
        )

    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Retry in a moment.")

    # Write upload to temp file
    content = await audio_file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = _run_inference(tmp_path)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    # Build acoustic_context + user_profile from form fields
    acoustic_ctx = {}
    if snr_db is not None:        acoustic_ctx["snr_db"] = snr_db
    if rms_loudness is not None:  acoustic_ctx["rms_loudness"] = rms_loudness
    if speech_prob is not None:   acoustic_ctx["speech_prob"] = speech_prob

    user_profile_d = {}
    if hearing_loss: user_profile_d["hearing_loss"] = hearing_loss
    if tinnitus is not None: user_profile_d["tinnitus"] = tinnitus
    if age_group:    user_profile_d["age_group"] = age_group
    if preference:   user_profile_d["preference"] = preference

    # Get recommendation
    probs_vec = np.array(
        [result["ensemble_probs"][c] for c in CLASS_NAMES], dtype=np.float32
    )
    rec = recommend_from_probs(
        probs_vec, CLASS_NAMES,
        acoustic_context=acoustic_ctx or None,
        user_profile=user_profile_d or None,
    )

    return PredictResponse(
        prediction=PredictionResult(
            class_name         = result["class_name"],
            class_id           = result["class_id"],
            confidence         = result["confidence"],
            is_uncertain       = result["is_uncertain"],
            ensemble_probs     = result["ensemble_probs"],
            model_contributions= result["model_contributions"],
        ),
        recommendation=RecommendationResult(
            volume             = rec.volume,
            noise_reduction    = rec.noise_reduction,
            directionality     = rec.directionality,
            speech_enhancement = rec.speech_enhancement,
            reasoning          = rec.reasoning,
            tiers_applied      = rec.tiers_applied,
            is_safety_override = rec.is_safety_override,
            suggested_follow_up= rec.suggested_follow_up,
        ),
        metadata={
            "filename":         audio_file.filename,
            "latency_ms":       result["latency_ms"],
            "device":           _DEVICE,
            "model_version":    "1.0.0",
            "ensemble_weights": WEIGHTS,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo.api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
