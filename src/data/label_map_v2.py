"""
Label mapping v2 — 8-Class Acoustically Distinct Schema.

Supersedes the original 10-class UrbanSound8K and 50-class ESC-50 taxonomies.
All models trained from Phase 3 onwards must use these mappings.

Class IDs:
  0 = speech
  1 = siren
  2 = horn
  3 = traffic
  4 = construction
  5 = dog_bark
  6 = music
  7 = background_noise
"""

# ─────────────────────────────────────────────────────────────────────────────
# v2 — 8-Class Target Schema
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES_V2 = [
    "speech",           # 0 — Safety P1
    "siren",            # 1 — Safety P1
    "horn",             # 2 — Safety P1
    "traffic",          # 3 — High P2
    "construction",     # 4 — High P2
    "dog_bark",         # 5 — Medium P3
    "music",            # 6 — Medium P3
    "background_noise", # 7 — Low P4
]

NUM_CLASSES_V2 = len(CLASS_NAMES_V2)

# Safety classes with lowered confidence thresholds
SAFETY_CLASSES_V2 = {"speech", "siren", "horn"}
SAFETY_THRESHOLD_V2 = {
    "speech": 0.25,
    "siren":  0.20,
    "horn":   0.25,
}
DEFAULT_THRESHOLD_V2 = 0.50

CLASS_EMOJIS_V2 = {
    "speech":           "🗣️",
    "siren":            "🚑",
    "horn":             "📯",
    "traffic":          "🚦",
    "construction":     "🏗️",
    "dog_bark":         "🐕",
    "music":            "🎵",
    "background_noise": "🔇",
}

# ─────────────────────────────────────────────────────────────────────────────
# UrbanSound8K (10 classes) → 8-Class V2
# ─────────────────────────────────────────────────────────────────────────────

US8K_TO_8CLASS: dict[str, str] = {
    "air_conditioner":  "background_noise",  # Diffuse, low-freq hum
    "engine_idling":    "traffic",           # Motor vehicle rumble
    "drilling":         "construction",      # Impulsive broadband
    "jackhammer":       "construction",      # Impulsive broadband (same cluster as drilling)
    "gun_shot":         "construction",      # Short impulse burst
    "car_horn":         "horn",              # Short harmonic transient
    "children_playing": "speech",            # Voiced, social soundscape
    "dog_bark":         "dog_bark",          # Perceptually distinct
    "siren":            "siren",             # Safety critical
    "street_music":     "music",             # Harmonic, periodic
}

# ─────────────────────────────────────────────────────────────────────────────
# ESC-50 (50 classes) → 8-Class V2
# ─────────────────────────────────────────────────────────────────────────────

ESC50_TO_8CLASS: dict[str, str] = {
    # Animals
    "dog":                  "dog_bark",
    "rooster":              "background_noise",
    "pig":                  "background_noise",
    "cow":                  "background_noise",
    "frog":                 "background_noise",
    "cat":                  "background_noise",
    "hen":                  "background_noise",
    "insects":              "background_noise",
    "sheep":                "background_noise",
    "crow":                 "background_noise",
    # Natural soundscapes
    "rain":                 "background_noise",
    "sea_waves":            "background_noise",
    "crackling_fire":       "background_noise",
    "crickets":             "background_noise",
    "chirping_birds":       "background_noise",
    "water_drops":          "background_noise",
    "wind":                 "background_noise",
    "pouring_water":        "background_noise",
    "toilet_flush":         "background_noise",
    "thunderstorm":         "background_noise",
    # Human non-speech
    "crying_baby":          "speech",
    "sneezing":             "speech",
    "clapping":             "speech",
    "breathing":            "speech",
    "coughing":             "speech",
    "footsteps":            "speech",
    "laughing":             "speech",
    "brushing_teeth":       "background_noise",
    "snoring":              "background_noise",
    "drinking_sipping":     "background_noise",
    # Interior domestic
    "door_wood_knock":      "construction",
    "mouse_click":          "background_noise",
    "keyboard_typing":      "background_noise",
    "door_wood_creaks":     "background_noise",
    "can_opening":          "background_noise",
    "washing_machine":      "background_noise",
    "vacuum_cleaner":       "background_noise",
    "clock_alarm":          "siren",
    "clock_tick":           "background_noise",
    "glass_breaking":       "construction",
    # Exterior urban
    "helicopter":           "traffic",
    "chainsaw":             "construction",
    "siren":                "siren",
    "car_horn":             "horn",
    "engine":               "traffic",
    "train":                "traffic",
    "church_bells":         "music",
    "airplane":             "traffic",
    "fireworks":            "construction",
    "hand_saw":             "construction",
}

# ─────────────────────────────────────────────────────────────────────────────
# Hearing Aid Recommendation Table (8-Class)
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDATION_TABLE_V2: dict[str, dict] = {
    "speech": {
        "volume": 7,
        "noise_reduction": "Low",
        "directionality": "Directional",
        "speech_enhancement": True,
        "reasoning": "Speech detected — enhance clarity, focus microphone forward.",
    },
    "siren": {
        "volume": 9,
        "noise_reduction": "Off",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": "⚠️ Emergency siren detected — maximize volume, remove all filters.",
    },
    "horn": {
        "volume": 8,
        "noise_reduction": "Off",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": "⚠️ Horn detected — vehicle alert, maximize awareness.",
    },
    "traffic": {
        "volume": 5,
        "noise_reduction": "High",
        "directionality": "Directional",
        "speech_enhancement": True,
        "reasoning": "Traffic noise — suppress rumble, maintain speech clarity.",
    },
    "construction": {
        "volume": 4,
        "noise_reduction": "High",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": "Construction noise — reduce exposure, protect from impulse sounds.",
    },
    "dog_bark": {
        "volume": 5,
        "noise_reduction": "Medium",
        "directionality": "Adaptive",
        "speech_enhancement": False,
        "reasoning": "Dog bark detected — moderate adaptation, no speech enhancement needed.",
    },
    "music": {
        "volume": 6,
        "noise_reduction": "Low",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": "Music environment — preserve dynamics, omnidirectional capture.",
    },
    "background_noise": {
        "volume": 5,
        "noise_reduction": "Medium",
        "directionality": "Omnidirectional",
        "speech_enhancement": False,
        "reasoning": "Background noise — balanced default settings.",
    },
}


def remap_us8k_label(original_label: str) -> str:
    """Map a US8K label string to the v2 8-class schema."""
    mapped = US8K_TO_8CLASS.get(original_label)
    if mapped is None:
        raise ValueError(f"Unknown US8K label: '{original_label}'. "
                         f"Valid labels: {list(US8K_TO_8CLASS.keys())}")
    return mapped


def remap_esc50_label(esc50_category: str) -> str:
    """Map an ESC-50 category string to the v2 8-class schema."""
    mapped = ESC50_TO_8CLASS.get(esc50_category)
    if mapped is None:
        # Fallback to background noise for unmapped categories
        return "background_noise"
    return mapped


def label_to_index(label: str) -> int:
    """Convert a v2 class name to its integer index."""
    return CLASS_NAMES_V2.index(label)


def index_to_label(idx: int) -> str:
    """Convert a v2 integer index to its class name."""
    return CLASS_NAMES_V2[idx]


def get_recommendation(class_name: str) -> dict:
    """Return hearing aid settings dict for a given v2 class name."""
    return RECOMMENDATION_TABLE_V2.get(class_name, RECOMMENDATION_TABLE_V2["background_noise"])


def get_safety_threshold(class_name: str) -> float:
    """Return the confidence threshold for a given v2 class name."""
    return SAFETY_THRESHOLD_V2.get(class_name, DEFAULT_THRESHOLD_V2)


if __name__ == "__main__":
    print("── 8-Class V2 Schema ─────────────────────────────────────")
    for i, c in enumerate(CLASS_NAMES_V2):
        r = get_recommendation(c)
        threshold = get_safety_threshold(c)
        print(f"  [{i}] {CLASS_EMOJIS_V2[c]} {c:20s} | "
              f"Vol={r['volume']} NR={r['noise_reduction']:6s} | "
              f"Threshold={threshold:.2f}")
    print()
    print("── US8K Remap Test ───────────────────────────────────────")
    for orig, new in US8K_TO_8CLASS.items():
        print(f"  {orig:20s} → {new}")
