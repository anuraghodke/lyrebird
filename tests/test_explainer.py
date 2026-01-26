"""
Tests for Step 5: Reasoning Generator

Verifies:
- Feature interpretation
- Melody comparison explanations
- Rhythm comparison explanations
- Combined feature explanations
"""

import numpy as np
import pytest

from src.explainer import (
    HPCP_SIZE,
    KEY_NAMES,
    MELODY_DIMS,
    RHYTHM_DIMS,
    _compare_melody,
    _compare_rhythm,
    _interpret_melody_features,
    _interpret_rhythm_features,
    generate_reasoning,
)


class TestConstants:
    """Test that constants are properly defined."""

    def test_hpcp_size(self):
        """Verify HPCP size matches audio_analyzer."""
        assert HPCP_SIZE == 12

    def test_melody_dims(self):
        """Verify melody dimensions match audio_analyzer."""
        assert MELODY_DIMS == 26

    def test_rhythm_dims(self):
        """Verify rhythm dimensions match audio_analyzer."""
        assert RHYTHM_DIMS == 16

    def test_key_names(self):
        """Verify all 12 key names are defined."""
        assert len(KEY_NAMES) == 12
        assert "C" in KEY_NAMES
        assert "A" in KEY_NAMES


class TestInterpretMelodyFeatures:
    """Test _interpret_melody_features function."""

    def test_returns_dict_with_expected_keys(self):
        """Verify returned dict has all expected keys."""
        features = np.random.rand(MELODY_DIMS).astype(np.float32)
        result = _interpret_melody_features(features)

        assert "key" in result
        assert "mode" in result
        assert "key_strength" in result
        assert "harmonic_complexity" in result
        assert "hpcp_mean" in result

    def test_detects_major_mode(self):
        """Verify major mode is detected when mode value > 0.5."""
        features = np.zeros(MELODY_DIMS, dtype=np.float32)
        features[25] = 0.8  # mode value
        result = _interpret_melody_features(features)

        assert result["mode"] == "major"

    def test_detects_minor_mode(self):
        """Verify minor mode is detected when mode value <= 0.5."""
        features = np.zeros(MELODY_DIMS, dtype=np.float32)
        features[25] = 0.3  # mode value
        result = _interpret_melody_features(features)

        assert result["mode"] == "minor"

    def test_detects_dominant_key(self):
        """Verify dominant pitch class is detected correctly."""
        features = np.zeros(MELODY_DIMS, dtype=np.float32)
        # Set A (index 9) as dominant pitch
        features[9] = 1.0
        result = _interpret_melody_features(features)

        assert result["key"] == "A"
        assert result["dominant_pitch"] == 9


class TestInterpretRhythmFeatures:
    """Test _interpret_rhythm_features function."""

    def test_returns_dict_with_expected_keys(self):
        """Verify returned dict has all expected keys."""
        features = np.random.rand(RHYTHM_DIMS).astype(np.float32)
        result = _interpret_rhythm_features(features)

        assert "bpm" in result
        assert "tempo_category" in result
        assert "beat_regularity" in result
        assert "onset_rate" in result

    def test_denormalizes_bpm_correctly(self):
        """Verify BPM is denormalized correctly."""
        features = np.zeros(RHYTHM_DIMS, dtype=np.float32)

        # normalized = 0 -> bpm = 60
        features[0] = 0.0
        result = _interpret_rhythm_features(features)
        assert result["bpm"] == pytest.approx(60.0)

        # normalized = 0.5 -> bpm = 120
        features[0] = 0.5
        result = _interpret_rhythm_features(features)
        assert result["bpm"] == pytest.approx(120.0)

        # normalized = 1.0 -> bpm = 180
        features[0] = 1.0
        result = _interpret_rhythm_features(features)
        assert result["bpm"] == pytest.approx(180.0)

    def test_categorizes_tempo_slow(self):
        """Verify slow tempo category."""
        features = np.zeros(RHYTHM_DIMS, dtype=np.float32)
        features[0] = 0.1  # ~72 BPM
        result = _interpret_rhythm_features(features)

        assert result["tempo_category"] == "slow"

    def test_categorizes_tempo_moderate(self):
        """Verify moderate tempo category."""
        features = np.zeros(RHYTHM_DIMS, dtype=np.float32)
        features[0] = 0.4  # ~108 BPM
        result = _interpret_rhythm_features(features)

        assert result["tempo_category"] == "moderate"

    def test_categorizes_tempo_upbeat(self):
        """Verify upbeat tempo category."""
        features = np.zeros(RHYTHM_DIMS, dtype=np.float32)
        features[0] = 0.6  # ~132 BPM
        result = _interpret_rhythm_features(features)

        assert result["tempo_category"] == "upbeat"

    def test_categorizes_tempo_fast(self):
        """Verify fast tempo category."""
        features = np.zeros(RHYTHM_DIMS, dtype=np.float32)
        features[0] = 0.9  # ~168 BPM
        result = _interpret_rhythm_features(features)

        assert result["tempo_category"] == "fast"


class TestCompareMelody:
    """Test _compare_melody function."""

    def test_same_key_and_mode(self):
        """Verify explanation for same key and mode."""
        ref = {
            "key": "C",
            "mode": "major",
            "harmonic_complexity": 0.1,
            "hpcp_mean": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }
        candidate = {
            "key": "C",
            "mode": "major",
            "harmonic_complexity": 0.1,
            "hpcp_mean": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }

        explanations = _compare_melody(ref, candidate)

        assert any("C major" in e for e in explanations)

    def test_same_mode_different_key(self):
        """Verify explanation for same mode but different key."""
        ref = {
            "key": "C",
            "mode": "minor",
            "harmonic_complexity": 0.2,
            "hpcp_mean": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }
        candidate = {
            "key": "A",
            "mode": "minor",
            "harmonic_complexity": 0.3,
            "hpcp_mean": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        }

        explanations = _compare_melody(ref, candidate)

        assert any("minor" in e for e in explanations)

    def test_similar_hpcp_generates_explanation(self):
        """Verify explanation for similar pitch patterns."""
        hpcp = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01, 0, 0, 0, 0])
        ref = {
            "key": "C",
            "mode": "major",
            "harmonic_complexity": 0.1,
            "hpcp_mean": hpcp,
        }
        candidate = {
            "key": "C",
            "mode": "major",
            "harmonic_complexity": 0.1,
            "hpcp_mean": hpcp * 0.95,  # Very similar
        }

        explanations = _compare_melody(ref, candidate)

        assert any("pitch" in e.lower() or "melodic" in e.lower() for e in explanations)


class TestCompareRhythm:
    """Test _compare_rhythm function."""

    def test_identical_tempo(self):
        """Verify explanation for identical tempo."""
        ref = {
            "bpm": 120.0,
            "tempo_category": "moderate",
            "beat_regularity": 0.8,
            "onset_rate": 0.5,
        }
        candidate = {
            "bpm": 121.0,
            "tempo_category": "moderate",
            "beat_regularity": 0.8,
            "onset_rate": 0.5,
        }

        explanations = _compare_rhythm(ref, candidate)

        assert any("120" in e or "tempo" in e.lower() for e in explanations)

    def test_similar_tempo(self):
        """Verify explanation for similar tempo."""
        ref = {
            "bpm": 120.0,
            "tempo_category": "moderate",
            "beat_regularity": 0.5,
            "onset_rate": 0.3,
        }
        candidate = {
            "bpm": 130.0,
            "tempo_category": "upbeat",
            "beat_regularity": 0.5,
            "onset_rate": 0.3,
        }

        explanations = _compare_rhythm(ref, candidate)

        assert any("tempo" in e.lower() or "bpm" in e.lower() for e in explanations)

    def test_steady_beat(self):
        """Verify explanation for steady beat."""
        ref = {
            "bpm": 100.0,
            "tempo_category": "moderate",
            "beat_regularity": 0.9,
            "onset_rate": 0.4,
        }
        candidate = {
            "bpm": 140.0,
            "tempo_category": "upbeat",
            "beat_regularity": 0.85,
            "onset_rate": 0.4,
        }

        explanations = _compare_rhythm(ref, candidate)

        assert any("steady" in e.lower() or "consistent" in e.lower() for e in explanations)


class TestGenerateReasoning:
    """Test generate_reasoning function."""

    def test_returns_list_of_strings(self):
        """Verify returns a list of strings."""
        ref = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)
        cand = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)

        result = generate_reasoning(ref, cand, feature_type="both")

        assert isinstance(result, list)
        assert all(isinstance(e, str) for e in result)

    def test_returns_max_4_explanations(self):
        """Verify returns at most 4 explanations."""
        ref = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)
        cand = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)

        result = generate_reasoning(ref, cand, feature_type="both")

        assert len(result) <= 4

    def test_returns_at_least_1_explanation(self):
        """Verify returns at least 1 explanation."""
        ref = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)
        cand = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)

        result = generate_reasoning(ref, cand, feature_type="both")

        assert len(result) >= 1

    def test_melody_feature_type(self):
        """Verify melody-only analysis works."""
        ref = np.random.rand(MELODY_DIMS).astype(np.float32)
        cand = np.random.rand(MELODY_DIMS).astype(np.float32)

        result = generate_reasoning(ref, cand, feature_type="melody")

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_rhythm_feature_type(self):
        """Verify rhythm-only analysis works."""
        ref = np.random.rand(RHYTHM_DIMS).astype(np.float32)
        cand = np.random.rand(RHYTHM_DIMS).astype(np.float32)

        result = generate_reasoning(ref, cand, feature_type="rhythm")

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_invalid_feature_type_raises(self):
        """Verify invalid feature_type raises ValueError."""
        ref = np.random.rand(MELODY_DIMS).astype(np.float32)
        cand = np.random.rand(MELODY_DIMS).astype(np.float32)

        with pytest.raises(ValueError) as exc_info:
            generate_reasoning(ref, cand, feature_type="invalid")

        assert "Invalid feature_type" in str(exc_info.value)

    def test_accepts_dict_with_features_key(self):
        """Verify accepts dict with 'features' key."""
        ref = {"features": np.random.rand(MELODY_DIMS + RHYTHM_DIMS)}
        cand = {"features": np.random.rand(MELODY_DIMS + RHYTHM_DIMS)}

        result = generate_reasoning(ref, cand, feature_type="both")

        assert isinstance(result, list)


class TestIntegration:
    """Integration tests for explainer workflow."""

    def test_similar_tracks_get_explanations(self):
        """Test that similar tracks get meaningful explanations."""
        # Create similar features
        base = np.random.rand(MELODY_DIMS + RHYTHM_DIMS).astype(np.float32)
        similar = base + np.random.rand(MELODY_DIMS + RHYTHM_DIMS) * 0.1

        result = generate_reasoning(base, similar, feature_type="both")

        assert len(result) >= 1
        # At least one explanation should mention something specific
        all_text = " ".join(result).lower()
        keywords = ["tempo", "bpm", "key", "major", "minor", "harmonic", "rhythm", "beat"]
        assert any(kw in all_text for kw in keywords)

    def test_with_real_feature_dimensions(self):
        """Test with actual audio feature dimensions."""
        # Melody features (26 dims)
        melody_ref = np.zeros(MELODY_DIMS, dtype=np.float32)
        melody_ref[:12] = [0.3, 0.1, 0.05, 0.2, 0.1, 0.05, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01]
        melody_ref[25] = 1.0  # Major mode

        melody_cand = melody_ref.copy()

        result = generate_reasoning(melody_ref, melody_cand, feature_type="melody")

        assert any("major" in e.lower() for e in result)

    def test_different_tempos_noted(self):
        """Test that different tempos are noted appropriately."""
        # Rhythm features (16 dims)
        rhythm_ref = np.zeros(RHYTHM_DIMS, dtype=np.float32)
        rhythm_ref[0] = 0.5  # 120 BPM

        rhythm_cand = np.zeros(RHYTHM_DIMS, dtype=np.float32)
        rhythm_cand[0] = 0.55  # ~126 BPM

        result = generate_reasoning(rhythm_ref, rhythm_cand, feature_type="rhythm")

        # Should mention tempo similarity
        assert any("tempo" in e.lower() or "bpm" in e.lower() for e in result)
