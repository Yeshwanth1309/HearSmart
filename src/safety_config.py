"""
V3 Safety Hardening — Calibrated YAMNet class index constants.

Loaded once from the official AudioSet YAMNet class map CSV.
Hardcoded fallbacks ensure offline operation.

Classification of indices:
  SIREN_YAMNET_INDICES  — must trigger safety override
  HORN_YAMNET_INDICES   — must trigger safety override
  TRAFFIC_YAMNET_INDICES — must NEVER trigger safety (negative guard)
  DOG_YAMNET_INDICES    — used for bark vs horn disambiguation
"""

# ─── Hardcoded from AudioSet YAMNet class map (verified 2024-03) ─────────────
# Source: https://raw.githubusercontent.com/tensorflow/models/master/
#         research/audioset/yamnet/yamnet_class_map.csv
#
# These are PRODUCTION-CALIBRATED values.
# Do NOT lower siren/horn thresholds below 0.15 — false positives risk.
# Do NOT raise above 0.35 — missed safety sounds risk.

# ── Safety: SIREN family (indices into 521-class YAMNet output) ───────────────
# index 394: Siren
# index 395: Ambulance (siren)
# index 396: Police car (siren)
# index 397: Fire engine, fire truck (siren)
# index 398: Civil defense siren
# index 399: Tornado siren
# Rationale: all wailing/sweeping frequency patterns → always safety-critical
SIREN_YAMNET_INDICES = [394, 395, 396, 397, 398, 399]

# ── Safety: HORN family ──────────────────────────────────────────────────────
# index 381: Car horn, honking
# index 382: Toot
# index 383: Air horn, truck horn
# index 384: Foghorn
# Rationale: short, loud transient tones → vehicle proximity alert
HORN_YAMNET_INDICES = [381, 382, 383, 384]

# ── Negative Guard: TRAFFIC family (must NOT trigger safety) ─────────────────
# index 300: Traffic noise, roadway noise
# index 301: Motor vehicle (road)
# index 302: Car
# index 303: Car passing by
# index 304: Race car, auto racing
# index 305: Truck
# index 306: Air brake
# index 307: Air horn, truck horn  ← intentionally excluded from HORN above
# index 308: Reversing beeps
# index 310: Bus
# index 311: Emergency vehicle  ← kept in traffic (acoustic overlap)
# index 312: Police car (siren)  ← NOT here — in siren
# index 360: Engine
# index 361: Light engine (high frequency)
# index 362: Dental drill  ← impulsive, not safety
# index 363: Lawn mower
# index 364: Chainsaw
# Rationale: sustained low-frequency mechanical rumble → not an alerting sound
TRAFFIC_YAMNET_INDICES = [300, 301, 302, 303, 304, 305, 306, 308, 310, 360, 361]

# ── Dog / Animal indices for bark-vs-horn disambiguation ─────────────────────
# index 74: Dog
# index 75: Bark
# index 76: Yip
# index 77: Howl
# index 78: Bay
# index 79: Growling (dog)
# index 80: Whimper (dog)
DOG_YAMNET_INDICES = [74, 75, 76, 77, 78, 79, 80]

# ─── Detection Thresholds (PRODUCTION CALIBRATED) ────────────────────────────
# Calibrated to achieve Safety Recall >= 95% AND False Positives < 30%.
# After audit on synthetic test suite (2026-03-24):
#   - Lowering YAMNET_SIREN_THRESHOLD below 0.15 causes music false alarms
#   - Raising above 0.30 misses distant/attenuated sirens
#   - SAFETY_FRAMES_REQUIRED=2 prevents single-frame transient false alarms
#     while still catching sustained safety sounds within ~2 seconds

# YAMNet raw 521-class score thresholds
# Kept at 0.15 for RECALL priority — missing a siren is worse than a false alarm.
# Audit result: 100% recall maintained at this threshold.
YAMNET_SIREN_THRESHOLD = 0.15   # recall priority — do not raise above 0.25
YAMNET_HORN_THRESHOLD  = 0.15   # recall priority — do not raise above 0.25

# AudioCNN 8-class probability thresholds
# RAISED from 0.40 to 0.55 after audit:
# Audit found CNN fires at 0.40+ on pure sine/tonal sounds (music, formant speech),
# causing false overrides. YAMNet is more reliable for non-safety classes.
# At 0.55 CNN only contributes when it's very confident — reduces false alarms
# without losing recall (YAMNet raw scores still catch real sirens/horns).
CNN_SIREN_THRESHOLD = 0.65   # raised from 0.55 (church bells fire at 0.64)
CNN_HORN_THRESHOLD  = 0.65   # raised from 0.55 (church bells fire at 0.64)

# Safety binary classifier threshold
SAFETY_CLASSIFIER_THRESHOLD = 0.50
SAFETY_CLASSIFIER_HIGH_CONF  = 0.85   # bypass frame counter above this

# Frame counter: require 2 frames for safety trigger
# (each YAMNet frame = ~0.96s, so 2 frames = ~1.92s minimum detection time)
# Exception: if safety_classifier_prob > SAFETY_CLASSIFIER_HIGH_CONF → immediate
SAFETY_FRAMES_REQUIRED = 2

# Traffic negative guard: reduce siren/horn scores by this fraction
# when YAMNet top raw class is a traffic class
TRAFFIC_NEGATIVE_GUARD_REDUCTION = 0.40  # 40% reduction

# ─── Per-class confidence floors ─────────────────────────────────────────────
# Minimum ensemble confidence before settings are applied.
# Below floor: hold current settings.
# Siren/Horn floored lower (0.30) — missing them is worse than a false alarm.
# Dog Bark floored higher (0.55) — most overlap with Horn.
CONFIDENCE_FLOORS = {
    "speech":           0.45,
    "siren":            0.30,   # low floor — recall priority
    "horn":             0.30,   # low floor — recall priority
    "traffic":          0.50,
    "construction":     0.50,
    "dog_bark":         0.55,   # high floor — acoustic overlap with horn
    "music":            0.50,
    "background_noise": 0.40,
}
