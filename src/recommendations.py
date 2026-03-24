"""
Phase 8 — Recommendation Engine.

Converts classified environment + confidence scores into optimal hearing aid
settings using a 4-tier decision architecture per TRD §3.7.

Tiers:
    Tier 1 — Rule-based class → base parameters (volume, noise reduction,
              directionality, speech enhancement)
    Tier 2 — Confidence adjustment: low-confidence → conservative defaults
    Tier 3 — Acoustic refinement:  SNR, loudness, speech probability → fine tuning
    Tier 4 — User personalization: hearing loss severity, tinnitus, age, preference

Public API
----------
    recommend(class_name, confidence, acoustic_context=None, user_profile=None)
        → Recommendation

    recommend_from_probs(probs, class_names, acoustic_context=None, user_profile=None)
        → Recommendation   (confidence-weighted blend over top-k classes)

    export_rules(path)  → writes rules JSON for API/Flutter consumption

Outputs
-------
    results/recommendation_rules.json   — complete rule-set as JSON
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants & types
# ─────────────────────────────────────────────────────────────────────────────
NOISE_LEVELS  = ("Low", "Medium", "High")
DIRECTIONAL   = ("Omnidirectional", "Directional", "Adaptive")

LOW_CONFIDENCE_THRESHOLD = 0.35   # below this → Tier 2 conservative mode
HIGH_CONFIDENCE_THRESHOLD = 0.70  # above this → full rule weight
SAFETY_CLASSES = {"car_horn", "gun_shot", "siren"}
SAFETY_THRESHOLD = 0.45           # per-model threshold for safety override

# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Recommendation:
    """Complete hearing aid parameter recommendation."""
    # Core settings
    volume: int                    # 1–10
    noise_reduction: str           # "Low" | "Medium" | "High"
    directionality: str            # "Omnidirectional" | "Directional" | "Adaptive"
    speech_enhancement: bool       # True | False

    # Meta
    environment: str               # predicted class name
    confidence: float              # 0.0–1.0
    is_uncertain: bool             # True if confidence < threshold
    is_safety_override: bool       # True if safety class was forced
    tiers_applied: list            # which tiers fired [1,2,3,4]
    reasoning: str                 # human-readable explanation

    # Optional advanced
    volume_range: tuple = field(default_factory=lambda: (1, 10))
    suggested_follow_up: str = ""  # e.g. "Monitor for speech"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["volume_range"] = list(d["volume_range"])
        return d

    def summary(self) -> str:
        lines = [
            f"Environment : {self.environment}  (confidence: {self.confidence:.1%})",
            f"Volume      : {self.volume}/10",
            f"Noise Red.  : {self.noise_reduction}",
            f"Direction.  : {self.directionality}",
            f"Speech Enh. : {'On' if self.speech_enhancement else 'Off'}",
        ]
        if self.is_uncertain:
            lines.append("⚠  Low confidence — conservative defaults applied")
        if self.is_safety_override:
            lines.append("🚨 Safety override active")
        lines.append(f"Reasoning   : {self.reasoning}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Rule-based base mapping  (TRD §3.7, Table)
# ─────────────────────────────────────────────────────────────────────────────
_TIER1_RULES: dict[str, dict] = {
    "air_conditioner": {
        "volume": 5,
        "noise_reduction": "High",
        "directionality": "Omnidirectional",
        "speech_enhancement": True,
        "reasoning": (
            "Continuous low-frequency background noise detected. "
            "High noise reduction filters the hum; speech enhancement "
            "preserves conversation clarity."
        ),
    },
    "car_horn": {
        "volume": 3,
        "noise_reduction": "High",
        "directionality": "Adaptive",
        "speech_enhancement": False,
        "reasoning": (
            "Safety-critical alert sound (car horn). Volume reduced to prevent "
            "discomfort; adaptive microphone steers toward the source; "
            "noise reduction high to suppress background traffic."
        ),
    },
    "children_playing": {
        "volume": 6,
        "noise_reduction": "Medium",
        "directionality": "Omnidirectional",
        "speech_enhancement": True,
        "reasoning": (
            "Lively, multi-directional play environment. Omnidirectional "
            "mode captures all voices; medium noise reduction preserves "
            "natural ambiance while reducing harshness."
        ),
    },
    "dog_bark": {
        "volume": 5,
        "noise_reduction": "Medium",
        "directionality": "Adaptive",
        "speech_enhancement": False,
        "reasoning": (
            "Intermittent animal sound. Adaptive microphone localises the "
            "source; medium noise reduction balances bark attenuation "
            "with environmental awareness."
        ),
    },
    "drilling": {
        "volume": 3,
        "noise_reduction": "High",
        "directionality": "Directional",
        "speech_enhancement": True,
        "reasoning": (
            "High-intensity construction noise. Volume reduced to protect "
            "hearing; directional mode focuses on speech in front; "
            "maximum noise reduction suppresses drill harmonics."
        ),
    },
    "engine_idling": {
        "volume": 5,
        "noise_reduction": "High",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": (
            "Steady engine background noise. High noise reduction on omnidirectional "
            "mode effectively suppresses the constant low-frequency engine rumble."
        ),
    },
    "gun_shot": {
        "volume": 2,
        "noise_reduction": "High",
        "directionality": "Adaptive",
        "speech_enhancement": False,
        "reasoning": (
            "⚠ SAFETY: Gunshot detected. Volume minimised to prevent acoustic "
            "trauma; adaptive microphone enables spatial awareness; "
            "high noise reduction protects from follow-on impulse noise."
        ),
    },
    "jackhammer": {
        "volume": 3,
        "noise_reduction": "High",
        "directionality": "Directional",
        "speech_enhancement": True,
        "reasoning": (
            "Intense impulsive construction noise. Same profile as drilling — "
            "directional focus for speech, maximum noise reduction, "
            "reduced volume to prevent fatigue."
        ),
    },
    "siren": {
        "volume": 4,
        "noise_reduction": "High",
        "directionality": "Adaptive",
        "speech_enhancement": False,
        "reasoning": (
            "⚠ SAFETY: Emergency siren detected. Volume moderated to allow "
            "natural awareness without discomfort; adaptive mode tracks "
            "the siren direction for situational awareness."
        ),
    },
    "street_music": {
        "volume": 7,
        "noise_reduction": "Low",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": (
            "Outdoor entertainment environment. Higher volume and low noise "
            "reduction preserves the full musical quality; omnidirectional "
            "captures the natural soundscape."
        ),
    },
}

# Conservative defaults (Tier 2 fallback)
_CONSERVATIVE_DEFAULTS = {
    "volume": 5,
    "noise_reduction": "Medium",
    "directionality": "Omnidirectional",
    "speech_enhancement": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Acoustic refinement helpers
# ─────────────────────────────────────────────────────────────────────────────
def _noise_level_index(level: str) -> int:
    return NOISE_LEVELS.index(level)

def _noise_level_from_index(idx: int) -> str:
    return NOISE_LEVELS[max(0, min(2, idx))]

def _apply_tier3(
    settings: dict,
    acoustic_context: dict,
    tiers: list,
    notes: list,
) -> None:
    """
    In-place Tier 3 adjustments based on acoustic_context.

    Expected keys (all optional):
        snr_db        : float   — signal-to-noise ratio in dB
        rms_loudness  : float   — RMS energy 0.0–1.0
        speech_prob   : float   — probability 0.0–1.0 that speech is present
    """
    snr        = acoustic_context.get("snr_db", None)
    loudness   = acoustic_context.get("rms_loudness", None)
    speech_p   = acoustic_context.get("speech_prob", None)

    fired = False

    if snr is not None:
        if snr > 20:  # excellent SNR — reduce noise suppression
            ni = _noise_level_index(settings["noise_reduction"])
            if ni > 0:
                settings["noise_reduction"] = _noise_level_from_index(ni - 1)
                notes.append(f"High SNR ({snr:.1f} dB) → noise reduction relaxed")
                fired = True
        elif snr < 5:  # very noisy
            ni = _noise_level_index(settings["noise_reduction"])
            settings["noise_reduction"] = _noise_level_from_index(ni + 1)
            settings["speech_enhancement"] = True
            notes.append(f"Low SNR ({snr:.1f} dB) → noise reduction raised + speech enhancement on")
            fired = True

    if loudness is not None:
        if loudness > 0.8:  # loud environment
            settings["volume"] = max(1, settings["volume"] - 2)
            notes.append(f"High loudness ({loudness:.2f}) → volume reduced by 2")
            fired = True
        elif loudness < 0.1:  # very quiet
            settings["volume"] = min(10, settings["volume"] + 2)
            notes.append(f"Low loudness ({loudness:.2f}) → volume increased by 2")
            fired = True

    if speech_p is not None and speech_p > 0.65:
        settings["directionality"] = "Directional"
        settings["speech_enhancement"] = True
        notes.append(f"High speech probability ({speech_p:.2f}) → directional + speech enhancement")
        fired = True

    if fired:
        tiers.append(3)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 — User personalization
# ─────────────────────────────────────────────────────────────────────────────
def _apply_tier4(
    settings: dict,
    user_profile: dict,
    tiers: list,
    notes: list,
) -> None:
    """
    In-place Tier 4 adjustments based on user_profile.

    Expected keys (all optional):
        hearing_loss   : "mild" | "moderate" | "severe"
        tinnitus       : bool
        age_group      : "young" | "adult" | "elderly"
        preference     : "speech" | "music" | "balanced"
    """
    fired = False

    loss = user_profile.get("hearing_loss", None)
    if loss == "severe":
        settings["volume"] = min(10, settings["volume"] + 2)
        settings["speech_enhancement"] = True
        notes.append("Severe hearing loss → +2 volume, speech enhancement on")
        fired = True
    elif loss == "moderate":
        settings["volume"] = min(10, settings["volume"] + 1)
        notes.append("Moderate hearing loss → +1 volume")
        fired = True

    if user_profile.get("tinnitus", False):
        ni = _noise_level_index(settings["noise_reduction"])
        settings["noise_reduction"] = _noise_level_from_index(ni + 1)
        notes.append("Tinnitus → noise reduction raised by one level")
        fired = True

    age = user_profile.get("age_group", None)
    if age == "elderly":
        settings["speech_enhancement"] = True
        notes.append("Elderly user profile → speech enhancement enabled")
        fired = True

    pref = user_profile.get("preference", None)
    if pref == "music":
        ni = _noise_level_index(settings["noise_reduction"])
        settings["noise_reduction"] = _noise_level_from_index(ni - 1)
        notes.append("Music preference → noise reduction relaxed for natural sound")
        fired = True
    elif pref == "speech":
        settings["speech_enhancement"] = True
        settings["directionality"] = "Directional"
        notes.append("Speech preference → directional + speech enhancement")
        fired = True

    if fired:
        tiers.append(4)


# ─────────────────────────────────────────────────────────────────────────────
# Core recommendation function
# ─────────────────────────────────────────────────────────────────────────────
def recommend(
    class_name: str,
    confidence: float,
    acoustic_context: Optional[dict] = None,
    user_profile: Optional[dict] = None,
    is_safety_override: bool = False,
) -> Recommendation:
    """
    Generate a 4-tier hearing aid recommendation.

    Parameters
    ----------
    class_name         : str    — predicted environment class
    confidence         : float  — model ensemble confidence (0–1)
    acoustic_context   : dict   — optional {snr_db, rms_loudness, speech_prob}
    user_profile       : dict   — optional {hearing_loss, tinnitus, age_group, preference}
    is_safety_override : bool   — True if a safety class was force-predicted

    Returns
    -------
    Recommendation dataclass
    """
    logger = logging.getLogger(__name__)
    tiers_applied = [1]
    notes = []

    # ── Tier 1: Rule-based base ───────────────────────────────────────────
    if class_name not in _TIER1_RULES:
        logger.warning(f"Unknown class '{class_name}'; using conservative defaults.")
        settings = _CONSERVATIVE_DEFAULTS.copy()
        base_reasoning = f"Unknown environment class '{class_name}'. Conservative defaults applied."
        is_uncertain = True
    else:
        settings = dict(_TIER1_RULES[class_name])
        base_reasoning = settings.pop("reasoning")
        is_uncertain = confidence < LOW_CONFIDENCE_THRESHOLD

    # ── Tier 2: Confidence adjustment ────────────────────────────────────
    if is_uncertain and not is_safety_override:
        tiers_applied.append(2)
        settings["volume"]           = _CONSERVATIVE_DEFAULTS["volume"]
        settings["noise_reduction"]  = _CONSERVATIVE_DEFAULTS["noise_reduction"]
        settings["directionality"]   = _CONSERVATIVE_DEFAULTS["directionality"]
        settings["speech_enhancement"] = _CONSERVATIVE_DEFAULTS["speech_enhancement"]
        notes.append(f"Low confidence ({confidence:.1%}) → conservative defaults (Tier 2)")

    # ── Tier 3: Acoustic refinement ──────────────────────────────────────
    if acoustic_context:
        _apply_tier3(settings, acoustic_context, tiers_applied, notes)

    # ── Tier 4: Personalisation ───────────────────────────────────────────
    if user_profile:
        _apply_tier4(settings, user_profile, tiers_applied, notes)

    # ── Build reasoning string ────────────────────────────────────────────
    full_reasoning = base_reasoning
    if notes:
        full_reasoning += " | " + "; ".join(notes)

    # ── Build follow-up suggestion ────────────────────────────────────────
    follow_up = ""
    if class_name in SAFETY_CLASSES:
        follow_up = "Monitor audio continuously. Alert user if sound persists."
    elif confidence < HIGH_CONFIDENCE_THRESHOLD:
        follow_up = "Re-evaluate in 3 seconds if environment changes."

    return Recommendation(
        volume            = int(np.clip(settings["volume"], 1, 10)),
        noise_reduction   = settings["noise_reduction"],
        directionality    = settings["directionality"],
        speech_enhancement= bool(settings["speech_enhancement"]),
        environment       = class_name,
        confidence        = round(float(confidence), 6),
        is_uncertain      = is_uncertain,
        is_safety_override= is_safety_override,
        tiers_applied     = sorted(set(tiers_applied)),
        reasoning         = full_reasoning,
        suggested_follow_up = follow_up,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Confidence-weighted blend for top-k classes
# ─────────────────────────────────────────────────────────────────────────────
def recommend_from_probs(
    probs: np.ndarray,
    class_names: list,
    acoustic_context: Optional[dict] = None,
    user_profile: Optional[dict] = None,
    top_k: int = 3,
    blend_temperature: float = 2.0,
) -> Recommendation:
    """
    Generate a recommendation from raw probability vector with
    confidence-weighted blending across the top-k classes.

    For ambiguous predictions (e.g. 35% dog bark, 30% children playing),
    this blends the settings from both classes proportionally, rather than
    hard-committing to the top-1 prediction.

    Parameters
    ----------
    probs              : np.ndarray (num_classes,) — ensemble softmax probs
    class_names        : list[str] — ordered class names matching probs
    acoustic_context   : dict — optional acoustic features
    user_profile       : dict — optional user profile
    top_k              : int — number of classes to blend (default: 3)
    blend_temperature  : float — sharpening factor (higher = more top-1 weight)

    Returns
    -------
    Recommendation — with blended settings and primary class name
    """
    probs = np.array(probs, dtype=np.float64)
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_idx]
    top_names = [class_names[i] for i in top_idx]

    # Primary prediction
    primary_class = top_names[0]
    primary_conf  = float(top_probs[0])

    # Safety override check
    is_safety = any(
        n in SAFETY_CLASSES and float(probs[class_names.index(n)]) >= SAFETY_THRESHOLD
        for n in class_names
        if n in SAFETY_CLASSES
    )
    if is_safety:
        safety_scores = {
            n: float(probs[class_names.index(n)])
            for n in SAFETY_CLASSES
            if n in class_names
        }
        primary_class = max(safety_scores, key=safety_scores.get)
        primary_conf  = safety_scores[primary_class]
        return recommend(
            primary_class, primary_conf,
            acoustic_context, user_profile,
            is_safety_override=True,
        )

    # If top-1 confident enough, skip blending
    if primary_conf >= HIGH_CONFIDENCE_THRESHOLD:
        return recommend(primary_class, primary_conf, acoustic_context, user_profile)

    # Blend: sharpened softmax over top-k probabilities
    blend_weights = np.exp(blend_temperature * top_probs)
    blend_weights = blend_weights / blend_weights.sum()

    # Aggregate numerical settings (volume is blendable; strings take top-1)
    blended_volume = 0.0
    for name, w in zip(top_names, blend_weights):
        if name in _TIER1_RULES:
            blended_volume += w * _TIER1_RULES[name]["volume"]

    # Get primary rec then override volume with blend
    rec = recommend(primary_class, primary_conf, acoustic_context, user_profile)

    blended_vol_int = int(round(np.clip(blended_volume, 1, 10)))
    if blended_vol_int != rec.volume:
        rec.volume = blended_vol_int
        blend_str = ", ".join(
            f"{n} ({w:.1%})" for n, w in zip(top_names, blend_weights)
        )
        rec.reasoning += f" | Volume blended from top-{top_k}: [{blend_str}]"
        if 3 not in rec.tiers_applied:
            rec.tiers_applied.append(3)
            rec.tiers_applied.sort()

    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Rules export
# ─────────────────────────────────────────────────────────────────────────────
def export_rules(path: str = "results/recommendation_rules.json") -> None:
    """
    Export complete rule-set as JSON for consumption by API/Flutter app.

    Includes:
      - Tier 1 rules for all 10 classes
      - Conservative defaults
      - Tier 2/3/4 descriptions
      - Safety class configuration
      - Threshold values
    """
    rules = {
        "version": "1.0.0",
        "description": "Hearing Aid Recommendation Engine — 4-Tier Rule Set",
        "thresholds": {
            "low_confidence":  LOW_CONFIDENCE_THRESHOLD,
            "high_confidence": HIGH_CONFIDENCE_THRESHOLD,
            "safety_override": SAFETY_THRESHOLD,
        },
        "safety_classes": list(SAFETY_CLASSES),
        "conservative_defaults": _CONSERVATIVE_DEFAULTS,
        "tier1_rules": {
            cls: {
                "volume":            r["volume"],
                "noise_reduction":   r["noise_reduction"],
                "directionality":    r["directionality"],
                "speech_enhancement":r["speech_enhancement"],
                "reasoning":         r["reasoning"],
            }
            for cls, r in _TIER1_RULES.items()
        },
        "tier2_description": (
            "If confidence < low_confidence_threshold and not a safety override, "
            "apply conservative defaults: volume=5, noise_reduction=Medium, "
            "directionality=Omnidirectional, speech_enhancement=True."
        ),
        "tier3_rules": {
            "high_snr": {
                "condition": "snr_db > 20",
                "action": "Reduce noise_reduction by one level",
            },
            "low_snr": {
                "condition": "snr_db < 5",
                "action": "Increase noise_reduction by one level + speech_enhancement=True",
            },
            "high_loudness": {
                "condition": "rms_loudness > 0.8",
                "action": "volume -= 2",
            },
            "low_loudness": {
                "condition": "rms_loudness < 0.1",
                "action": "volume += 2",
            },
            "speech_present": {
                "condition": "speech_prob > 0.65",
                "action": "directionality=Directional, speech_enhancement=True",
            },
        },
        "tier4_rules": {
            "severe_hearing_loss": {
                "condition": "hearing_loss == 'severe'",
                "action": "volume += 2, speech_enhancement=True",
            },
            "moderate_hearing_loss": {
                "condition": "hearing_loss == 'moderate'",
                "action": "volume += 1",
            },
            "tinnitus": {
                "condition": "tinnitus == True",
                "action": "noise_reduction raised by one level",
            },
            "elderly": {
                "condition": "age_group == 'elderly'",
                "action": "speech_enhancement=True",
            },
            "music_preference": {
                "condition": "preference == 'music'",
                "action": "noise_reduction relaxed by one level",
            },
            "speech_preference": {
                "condition": "preference == 'speech'",
                "action": "directionality=Directional, speech_enhancement=True",
            },
        },
        "volume_clamp": {"min": 1, "max": 10},
        "valid_noise_levels": list(NOISE_LEVELS),
        "valid_directionality": list(DIRECTIONAL),
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)
    logging.getLogger(__name__).info(f"Rules exported → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation (for test suite)
# ─────────────────────────────────────────────────────────────────────────────
def batch_recommend(
    predictions: list[tuple[str, float]],
    acoustic_contexts: Optional[list[dict]] = None,
    user_profile: Optional[dict] = None,
) -> list[Recommendation]:
    """
    Run recommend() for a batch of (class_name, confidence) pairs.

    Parameters
    ----------
    predictions       : list[(class_name, confidence)]
    acoustic_contexts : list[dict] | None — one dict per prediction
    user_profile      : dict | None — shared profile for all

    Returns
    -------
    list[Recommendation]
    """
    results = []
    for i, (cls, conf) in enumerate(predictions):
        ctx = acoustic_contexts[i] if acoustic_contexts else None
        results.append(recommend(cls, conf, ctx, user_profile))
    return results
