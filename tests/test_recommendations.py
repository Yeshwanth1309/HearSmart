"""
Phase 8 — Unit Tests: Recommendation Engine.

Tests all 4 tiers, all 10 environment classes, safety overrides,
confidence-weighted blending, and edge cases.

Run: python -m pytest tests/test_recommendations.py -v
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.recommendations import (
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    SAFETY_THRESHOLD,
    SAFETY_CLASSES,
    Recommendation,
    batch_recommend,
    export_rules,
    recommend,
    recommend_from_probs,
    _TIER1_RULES,
    _CONSERVATIVE_DEFAULTS,
)

ALL_CLASSES = list(_TIER1_RULES.keys())
NUM_CLASSES = 10


# ─────────────────────────────────────────────────────────────────────────────
class TestTier1RuleMapping(unittest.TestCase):
    """Tier 1: each of the 10 classes produces valid base settings."""

    def test_all_10_classes_have_rules(self):
        """Every UrbanSound8K class must have a rule."""
        expected = {
            "air_conditioner", "car_horn", "children_playing", "dog_bark",
            "drilling", "engine_idling", "gun_shot", "jackhammer",
            "siren", "street_music",
        }
        self.assertEqual(set(_TIER1_RULES.keys()), expected)

    def test_volume_in_valid_range(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.85)
            self.assertGreaterEqual(rec.volume, 1, f"{cls}: volume < 1")
            self.assertLessEqual(rec.volume, 10,  f"{cls}: volume > 10")

    def test_noise_reduction_valid(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.85)
            self.assertIn(rec.noise_reduction, ("Low", "Medium", "High"),
                          f"{cls}: invalid noise_reduction")

    def test_directionality_valid(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.85)
            self.assertIn(rec.directionality,
                          ("Omnidirectional", "Directional", "Adaptive"),
                          f"{cls}: invalid directionality")

    def test_speech_enhancement_is_bool(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.85)
            self.assertIsInstance(rec.speech_enhancement, bool, f"{cls}: speech_enhancement not bool")

    def test_return_type_is_recommendation(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.75)
            self.assertIsInstance(rec, Recommendation)

    def test_environment_field_matches_input(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.75)
            self.assertEqual(rec.environment, cls)

    def test_tier1_always_applied(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.85)
            self.assertIn(1, rec.tiers_applied)

    def test_known_rule_values(self):
        """Spot-check specific known values from TRD §3.7 Table."""
        cases = [
            ("air_conditioner", 5,  "High",   "Omnidirectional", True),
            ("street_music",    7,  "Low",    "Omnidirectional", False),
            ("gun_shot",        2,  "High",   "Adaptive",        False),
            ("drilling",        3,  "High",   "Directional",     True),
            ("siren",           4,  "High",   "Adaptive",        False),
            ("children_playing",6,  "Medium", "Omnidirectional", True),
        ]
        for cls, vol, nr, dirn, se in cases:
            rec = recommend(cls, 0.85)
            with self.subTest(cls=cls):
                self.assertEqual(rec.volume,             vol,  f"{cls} volume mismatch")
                self.assertEqual(rec.noise_reduction,    nr,   f"{cls} noise_reduction mismatch")
                self.assertEqual(rec.directionality,     dirn, f"{cls} directionality mismatch")
                self.assertEqual(rec.speech_enhancement, se,   f"{cls} speech_enhancement mismatch")

    def test_reasoning_not_empty(self):
        for cls in ALL_CLASSES:
            rec = recommend(cls, 0.80)
            self.assertTrue(len(rec.reasoning) > 10, f"{cls}: reasoning too short")


# ─────────────────────────────────────────────────────────────────────────────
class TestTier2ConfidenceAdjustment(unittest.TestCase):
    """Tier 2: low-confidence predictions fall back to conservative defaults."""

    def test_low_confidence_triggers_tier2(self):
        rec = recommend("street_music", 0.20)
        self.assertIn(2, rec.tiers_applied)
        self.assertTrue(rec.is_uncertain)

    def test_low_confidence_volume_is_conservative(self):
        rec = recommend("street_music", 0.20)  # Tier 1 would give vol=7
        self.assertEqual(rec.volume, _CONSERVATIVE_DEFAULTS["volume"])

    def test_low_confidence_noise_reduction_is_conservative(self):
        rec = recommend("street_music", 0.20)
        self.assertEqual(rec.noise_reduction, _CONSERVATIVE_DEFAULTS["noise_reduction"])

    def test_low_confidence_directionality_is_conservative(self):
        rec = recommend("street_music", 0.20)
        self.assertEqual(rec.directionality, _CONSERVATIVE_DEFAULTS["directionality"])

    def test_low_confidence_speech_enhancement_on(self):
        rec = recommend("street_music", 0.20)
        self.assertTrue(rec.speech_enhancement)

    def test_high_confidence_does_not_trigger_tier2(self):
        rec = recommend("street_music", 0.90)
        self.assertNotIn(2, rec.tiers_applied)
        self.assertFalse(rec.is_uncertain)

    def test_boundary_exactly_at_threshold_is_uncertain(self):
        rec = recommend("dog_bark", LOW_CONFIDENCE_THRESHOLD - 0.001)
        self.assertTrue(rec.is_uncertain)

    def test_boundary_at_threshold_is_not_uncertain(self):
        rec = recommend("dog_bark", LOW_CONFIDENCE_THRESHOLD)
        self.assertFalse(rec.is_uncertain)

    def test_safety_class_high_confidence_skips_tier2(self):
        """Even at low confidence, if is_safety_override=True, don't apply Tier 2."""
        rec = recommend("gun_shot", 0.10, is_safety_override=True)
        self.assertNotIn(2, rec.tiers_applied)
        self.assertEqual(rec.volume, 2)  # Tier 1 gun_shot value


# ─────────────────────────────────────────────────────────────────────────────
class TestTier3AcousticRefinement(unittest.TestCase):
    """Tier 3: acoustic context adjusts settings."""

    def test_high_snr_relaxes_noise_reduction(self):
        """High SNR (clean signal) → noise reduction drops."""
        rec_base = recommend("air_conditioner", 0.80)
        rec_snr  = recommend("air_conditioner", 0.80, acoustic_context={"snr_db": 25.0})
        self.assertIn(3, rec_snr.tiers_applied)
        # air_conditioner has "High" → should drop to "Medium"
        self.assertLess(
            ("Low", "Medium", "High").index(rec_snr.noise_reduction),
            ("Low", "Medium", "High").index(rec_base.noise_reduction),
        )

    def test_low_snr_raises_noise_reduction(self):
        rec = recommend("children_playing", 0.80, acoustic_context={"snr_db": 2.0})
        self.assertIn(3, rec.tiers_applied)
        self.assertTrue(rec.speech_enhancement)

    def test_high_loudness_reduces_volume(self):
        rec_base = recommend("street_music", 0.80)                # vol=7
        rec_loud = recommend("street_music", 0.80, acoustic_context={"rms_loudness": 0.9})
        self.assertEqual(rec_loud.volume, rec_base.volume - 2)

    def test_low_loudness_increases_volume(self):
        rec_base  = recommend("engine_idling", 0.80)              # vol=5
        rec_quiet = recommend("engine_idling", 0.80, acoustic_context={"rms_loudness": 0.05})
        self.assertEqual(rec_quiet.volume, rec_base.volume + 2)

    def test_volume_never_exceeds_10(self):
        rec = recommend("street_music", 0.80, acoustic_context={"rms_loudness": 0.001})
        self.assertLessEqual(rec.volume, 10)

    def test_volume_never_below_1(self):
        rec = recommend("gun_shot", 0.80, acoustic_context={"rms_loudness": 0.99})
        self.assertGreaterEqual(rec.volume, 1)

    def test_high_speech_prob_sets_directional(self):
        rec = recommend("air_conditioner", 0.80, acoustic_context={"speech_prob": 0.85})
        self.assertEqual(rec.directionality, "Directional")
        self.assertTrue(rec.speech_enhancement)

    def test_no_acoustic_context_skips_tier3(self):
        rec = recommend("dog_bark", 0.80)
        self.assertNotIn(3, rec.tiers_applied)

    def test_multiple_acoustic_factors_all_applied(self):
        rec = recommend("engine_idling", 0.80, acoustic_context={
            "snr_db": 2.0, "rms_loudness": 0.05, "speech_prob": 0.85
        })
        self.assertIn(3, rec.tiers_applied)


# ─────────────────────────────────────────────────────────────────────────────
class TestTier4Personalization(unittest.TestCase):
    """Tier 4: user profile modifies settings."""

    def test_severe_hearing_loss_increases_volume(self):
        rec_base = recommend("dog_bark", 0.85)
        rec_srv  = recommend("dog_bark", 0.85, user_profile={"hearing_loss": "severe"})
        self.assertEqual(rec_srv.volume, min(10, rec_base.volume + 2))
        self.assertIn(4, rec_srv.tiers_applied)

    def test_moderate_hearing_loss_increases_volume(self):
        rec_base = recommend("engine_idling", 0.85)
        rec_mod  = recommend("engine_idling", 0.85, user_profile={"hearing_loss": "moderate"})
        self.assertEqual(rec_mod.volume, min(10, rec_base.volume + 1))

    def test_tinnitus_raises_noise_reduction(self):
        rec = recommend("street_music", 0.85, user_profile={"tinnitus": True})
        # street_music base = "Low" → tinnitus shifts to "Medium"
        self.assertGreater(
            ("Low", "Medium", "High").index(rec.noise_reduction), 0
        )
        self.assertIn(4, rec.tiers_applied)

    def test_elderly_enables_speech_enhancement(self):
        rec = recommend("engine_idling", 0.85, user_profile={"age_group": "elderly"})
        self.assertTrue(rec.speech_enhancement)
        self.assertIn(4, rec.tiers_applied)

    def test_music_preference_relaxes_noise_reduction(self):
        rec_base = recommend("air_conditioner", 0.85)   # noise_reduction="High"
        rec_music = recommend("air_conditioner", 0.85, user_profile={"preference": "music"})
        self.assertLess(
            ("Low", "Medium", "High").index(rec_music.noise_reduction),
            ("Low", "Medium", "High").index(rec_base.noise_reduction),
        )

    def test_speech_preference_sets_directional(self):
        rec = recommend("children_playing", 0.85, user_profile={"preference": "speech"})
        self.assertEqual(rec.directionality, "Directional")
        self.assertTrue(rec.speech_enhancement)

    def test_mild_hearing_loss_no_change(self):
        rec_base = recommend("siren", 0.85)
        rec_mild = recommend("siren", 0.85, user_profile={"hearing_loss": "mild"})
        # mild not explicitly handled → no volume change
        self.assertEqual(rec_base.volume, rec_mild.volume)

    def test_multiple_profile_fields(self):
        rec = recommend("jackhammer", 0.85, user_profile={
            "hearing_loss": "severe", "tinnitus": True, "age_group": "elderly"
        })
        self.assertIn(4, rec.tiers_applied)
        self.assertLessEqual(rec.volume, 10)


# ─────────────────────────────────────────────────────────────────────────────
class TestSafetyOverride(unittest.TestCase):
    """Safety classes must never be suppressed."""

    def test_gun_shot_is_safety_class(self):
        self.assertIn("gun_shot", SAFETY_CLASSES)

    def test_siren_is_safety_class(self):
        self.assertIn("siren", SAFETY_CLASSES)

    def test_car_horn_is_safety_class(self):
        self.assertIn("car_horn", SAFETY_CLASSES)

    def test_safety_override_flag_set(self):
        rec = recommend("gun_shot", 0.75, is_safety_override=True)
        self.assertTrue(rec.is_safety_override)

    def test_safety_override_skips_tier2(self):
        """Even at very low confidence, safety settings must apply."""
        rec = recommend("siren", 0.05, is_safety_override=True)
        self.assertEqual(rec.volume, 4)          # siren Tier 1 volume
        self.assertEqual(rec.noise_reduction, "High")
        self.assertNotIn(2, rec.tiers_applied)   # Tier 2 NOT applied

    def test_gun_shot_volume_is_2(self):
        """Gun shot must always set volume=2 (hearing protection)."""
        rec = recommend("gun_shot", 0.95)
        self.assertEqual(rec.volume, 2)

    def test_safety_class_detected_in_probs(self):
        """recommend_from_probs detects safety class above threshold."""
        probs = np.full(NUM_CLASSES, 0.05)
        class_names = list(_TIER1_RULES.keys())
        gun_idx = class_names.index("gun_shot")
        probs[gun_idx] = 0.55   # above SAFETY_THRESHOLD=0.45
        probs = probs / probs.sum()
        rec = recommend_from_probs(probs, class_names)
        self.assertTrue(rec.is_safety_override)
        self.assertEqual(rec.environment, "gun_shot")

    def test_safety_class_below_threshold_not_overridden(self):
        """Safety class with prob < SAFETY_THRESHOLD does not trigger override."""
        probs = np.zeros(NUM_CLASSES)
        class_names = list(_TIER1_RULES.keys())
        gun_idx = class_names.index("gun_shot")
        top_idx = class_names.index("street_music")
        probs[gun_idx] = 0.30   # below SAFETY_THRESHOLD
        probs[top_idx] = 0.60
        probs = probs / probs.sum()
        rec = recommend_from_probs(probs, class_names)
        self.assertFalse(rec.is_safety_override)


# ─────────────────────────────────────────────────────────────────────────────
class TestProbabilityWeightedBlend(unittest.TestCase):
    """recommend_from_probs: confidence-weighted blending."""

    def test_high_confidence_uses_top1(self):
        probs = np.zeros(NUM_CLASSES)
        class_names = list(_TIER1_RULES.keys())
        probs[class_names.index("street_music")] = 0.92
        probs = probs / probs.sum()
        rec = recommend_from_probs(probs, class_names)
        self.assertEqual(rec.environment, "street_music")
        self.assertFalse(rec.is_uncertain)

    def test_low_confidence_triggers_uncertain(self):
        probs = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)  # uniform
        class_names = list(_TIER1_RULES.keys())
        rec = recommend_from_probs(probs, class_names)
        self.assertTrue(rec.is_uncertain)

    def test_returns_recommendation_instance(self):
        probs = np.ones(NUM_CLASSES) / NUM_CLASSES
        rec = recommend_from_probs(probs, list(_TIER1_RULES.keys()))
        self.assertIsInstance(rec, Recommendation)

    def test_probs_dont_need_to_be_normalised(self):
        """Function should handle un-normalised probs gracefully."""
        probs = np.array([0.5, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        rec = recommend_from_probs(probs, list(_TIER1_RULES.keys()))
        self.assertIsInstance(rec, Recommendation)

    def test_blended_volume_within_range(self):
        probs = np.array([0.35, 0.30, 0.10, 0.05, 0.05, 0.05, 0.02, 0.03, 0.02, 0.03])
        rec = recommend_from_probs(probs, list(_TIER1_RULES.keys()))
        self.assertGreaterEqual(rec.volume, 1)
        self.assertLessEqual(rec.volume, 10)


# ─────────────────────────────────────────────────────────────────────────────
class TestBatchRecommend(unittest.TestCase):
    """batch_recommend: applies to multiple predictions."""

    def test_batch_length_matches_input(self):
        preds = [(cls, 0.80) for cls in ALL_CLASSES]
        recs = batch_recommend(preds)
        self.assertEqual(len(recs), NUM_CLASSES)

    def test_batch_all_valid_recommendations(self):
        preds = [(cls, 0.75) for cls in ALL_CLASSES]
        recs = batch_recommend(preds)
        for rec in recs:
            self.assertIsInstance(rec, Recommendation)
            self.assertGreaterEqual(rec.volume, 1)
            self.assertLessEqual(rec.volume, 10)

    def test_batch_with_acoustic_contexts(self):
        preds = [(cls, 0.80) for cls in ALL_CLASSES]
        ctxs  = [{"snr_db": 15.0}] * NUM_CLASSES
        recs  = batch_recommend(preds, acoustic_contexts=ctxs)
        self.assertEqual(len(recs), NUM_CLASSES)

    def test_batch_with_user_profile(self):
        preds = [(cls, 0.80) for cls in ALL_CLASSES]
        recs  = batch_recommend(preds, user_profile={"hearing_loss": "severe"})
        for rec in recs:
            self.assertLessEqual(rec.volume, 10)


# ─────────────────────────────────────────────────────────────────────────────
class TestRecommendationToDict(unittest.TestCase):
    """Recommendation.to_dict() — JSON serialisability."""

    def test_to_dict_returns_dict(self):
        rec = recommend("siren", 0.88)
        d = rec.to_dict()
        self.assertIsInstance(d, dict)

    def test_to_dict_contains_required_keys(self):
        rec = recommend("siren", 0.88)
        d = rec.to_dict()
        required = {"volume", "noise_reduction", "directionality",
                    "speech_enhancement", "environment", "confidence",
                    "is_uncertain", "is_safety_override", "tiers_applied", "reasoning"}
        self.assertTrue(required.issubset(d.keys()))

    def test_to_dict_is_json_serialisable(self):
        rec = recommend("drilling", 0.78)
        try:
            json.dumps(rec.to_dict())
        except TypeError as e:
            self.fail(f"to_dict() is not JSON serialisable: {e}")

    def test_summary_returns_string(self):
        rec = recommend("gun_shot", 0.92)
        s = rec.summary()
        self.assertIsInstance(s, str)
        self.assertIn("gun_shot", s)


# ─────────────────────────────────────────────────────────────────────────────
class TestExportRules(unittest.TestCase):
    """export_rules() — produces valid JSON with expected structure."""

    def test_export_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        export_rules(path)
        self.assertTrue(os.path.exists(path))
        os.unlink(path)

    def test_export_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        export_rules(path)
        with open(path) as f:
            data = json.load(f)
        os.unlink(path)
        self.assertIsInstance(data, dict)

    def test_export_has_all_10_classes(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        export_rules(path)
        with open(path) as f:
            data = json.load(f)
        os.unlink(path)
        self.assertEqual(len(data["tier1_rules"]), 10)

    def test_export_has_version_field(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        export_rules(path)
        with open(path) as f:
            data = json.load(f)
        os.unlink(path)
        self.assertIn("version", data)
        self.assertIn("thresholds", data)
        self.assertIn("safety_classes", data)


# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases(unittest.TestCase):
    """Edge cases: unknown class, zero confidence, boundary values."""

    def test_unknown_class_falls_back_gracefully(self):
        rec = recommend("unknown_sound", 0.80)
        self.assertEqual(rec.volume, _CONSERVATIVE_DEFAULTS["volume"])
        self.assertTrue(rec.is_uncertain)

    def test_confidence_exactly_zero(self):
        rec = recommend("siren", 0.0)
        self.assertTrue(rec.is_uncertain)
        self.assertGreaterEqual(rec.volume, 1)

    def test_confidence_exactly_one(self):
        rec = recommend("siren", 1.0)
        self.assertFalse(rec.is_uncertain)
        self.assertEqual(rec.volume, 4)  # siren Tier 1

    def test_all_tiers_can_fire_together(self):
        rec = recommend(
            "air_conditioner", 0.80,
            acoustic_context={"snr_db": 2.0, "speech_prob": 0.80},
            user_profile={"hearing_loss": "severe", "tinnitus": True},
        )
        self.assertIn(1, rec.tiers_applied)
        self.assertIn(3, rec.tiers_applied)
        self.assertIn(4, rec.tiers_applied)

    def test_volume_clamp_after_all_tiers(self):
        """Multiple +2 boosts from severe hearing loss + quiet env → still ≤ 10."""
        rec = recommend(
            "street_music", 0.80,   # base vol=7
            acoustic_context={"rms_loudness": 0.001},  # +2
            user_profile={"hearing_loss": "severe"},    # +2
        )
        self.assertLessEqual(rec.volume, 10)
        self.assertGreaterEqual(rec.volume, 1)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
