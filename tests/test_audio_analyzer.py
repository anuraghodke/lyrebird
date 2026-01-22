"""
Tests for Step 3: Audio Analysis with Essentia

Verifies:
- Audio loading from bytes
- Melody feature extraction (26 dimensions)
- Rhythm feature extraction (16 dimensions)
- Combined feature extraction (42 dimensions)
- Error handling for invalid inputs
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio_analyzer import (
    BOTH_DIMS,
    HPCP_SIZE,
    MELODY_DIMS,
    RHYTHM_DIMS,
    SAMPLE_RATE,
    AudioAnalysisError,
    _ensure_essentia,
    _load_audio_from_bytes,
    analyze_melody,
    analyze_rhythm,
    analyze_track,
)


# Check if Essentia is available for integration tests
def essentia_installed():
    """Check if Essentia is installed."""
    try:
        import essentia  # noqa: F401
        import essentia.standard  # noqa: F401

        return True
    except ImportError:
        return False


ESSENTIA_AVAILABLE = essentia_installed()


class TestConstants:
    """Test that constants are properly defined."""

    def test_sample_rate(self):
        """Verify sample rate is standard 44100 Hz."""
        assert SAMPLE_RATE == 44100

    def test_hpcp_size(self):
        """Verify HPCP size is 12 (chromatic bins)."""
        assert HPCP_SIZE == 12

    def test_melody_dims(self):
        """Verify melody feature dimensions: HPCP mean (12) + std (12) + key (2)."""
        assert MELODY_DIMS == 26

    def test_rhythm_dims(self):
        """Verify rhythm feature dimensions."""
        assert RHYTHM_DIMS == 16

    def test_both_dims(self):
        """Verify combined dimensions = melody + rhythm."""
        assert BOTH_DIMS == MELODY_DIMS + RHYTHM_DIMS
        assert BOTH_DIMS == 42


class TestAudioAnalysisError:
    """Test the AudioAnalysisError exception."""

    def test_is_exception(self):
        """Verify AudioAnalysisError is an Exception."""
        assert issubclass(AudioAnalysisError, Exception)

    def test_can_raise(self):
        """Verify AudioAnalysisError can be raised and caught."""
        with pytest.raises(AudioAnalysisError) as exc_info:
            raise AudioAnalysisError("Test error")
        assert "Test error" in str(exc_info.value)


class TestEnsureEssentia:
    """Test _ensure_essentia function."""

    def test_raises_import_error_when_not_installed(self):
        """Verify ImportError is raised when Essentia is not installed."""
        with (
            patch.dict("sys.modules", {"essentia": None, "essentia.standard": None}),
            patch("builtins.__import__", side_effect=ImportError("No module")),
            pytest.raises(ImportError) as exc_info,
        ):
            _ensure_essentia()
        assert "Essentia is required" in str(exc_info.value)


class TestLoadAudioFromBytes:
    """Test _load_audio_from_bytes function."""

    def test_empty_bytes_raises_value_error(self):
        """Verify empty bytes raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _load_audio_from_bytes(b"")
        assert "cannot be empty" in str(exc_info.value)

    def test_none_bytes_raises_value_error(self):
        """Verify None raises ValueError."""
        with pytest.raises(ValueError):
            _load_audio_from_bytes(None)

    def test_loads_audio_via_temp_file(self):
        """Verify audio is loaded through a temporary file."""
        # Create mock audio data (2 seconds at 44100 Hz)
        mock_audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        mock_loader_instance = MagicMock(return_value=mock_audio)
        mock_loader_class = MagicMock(return_value=mock_loader_instance)

        # Create mock essentia module
        mock_essentia = MagicMock()
        mock_essentia_standard = MagicMock()
        mock_essentia_standard.MonoLoader = mock_loader_class

        with patch.dict(
            sys.modules,
            {"essentia": mock_essentia, "essentia.standard": mock_essentia_standard},
        ):
            result = _load_audio_from_bytes(b"fake mp3 data")
            assert len(result) == SAMPLE_RATE * 2
            mock_loader_class.assert_called_once()

    def test_short_audio_raises_error(self):
        """Verify audio shorter than minimum duration raises error."""
        # Return audio that's too short (0.5 seconds)
        short_audio = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        mock_loader_instance = MagicMock(return_value=short_audio)
        mock_loader_class = MagicMock(return_value=mock_loader_instance)

        # Create mock essentia module
        mock_essentia = MagicMock()
        mock_essentia_standard = MagicMock()
        mock_essentia_standard.MonoLoader = mock_loader_class

        with patch.dict(
            sys.modules,
            {"essentia": mock_essentia, "essentia.standard": mock_essentia_standard},
        ):
            with pytest.raises(AudioAnalysisError) as exc_info:
                _load_audio_from_bytes(b"fake mp3 data")
            assert "too short" in str(exc_info.value)

    def test_decode_error_raises_audio_analysis_error(self):
        """Verify decode failure raises AudioAnalysisError."""
        mock_loader_instance = MagicMock(side_effect=RuntimeError("Failed to decode"))
        mock_loader_class = MagicMock(return_value=mock_loader_instance)

        # Create mock essentia module
        mock_essentia = MagicMock()
        mock_essentia_standard = MagicMock()
        mock_essentia_standard.MonoLoader = mock_loader_class

        with patch.dict(
            sys.modules,
            {"essentia": mock_essentia, "essentia.standard": mock_essentia_standard},
        ):
            with pytest.raises(AudioAnalysisError) as exc_info:
                _load_audio_from_bytes(b"invalid audio data")
            assert "Failed to decode" in str(exc_info.value)


class TestAnalyzeMelody:
    """Test analyze_melody function."""

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_hpcp_features")
    @patch("src.audio_analyzer._extract_key_features")
    def test_returns_26_dimensions(
        self, mock_key, mock_hpcp, mock_load, mock_ensure
    ):
        """Verify analyze_melody returns 26-dimensional vector."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_hpcp.return_value = np.random.rand(24)
        mock_key.return_value = np.array([0.8, 1.0])

        features = analyze_melody(b"test audio")

        assert features.shape == (MELODY_DIMS,)
        mock_load.assert_called_once_with(b"test audio")
        mock_hpcp.assert_called_once()
        mock_key.assert_called_once()

    @patch("src.audio_analyzer._ensure_essentia")
    def test_empty_bytes_raises_value_error(self, mock_ensure):
        """Verify empty bytes raises ValueError."""
        with pytest.raises(ValueError):
            analyze_melody(b"")


class TestAnalyzeRhythm:
    """Test analyze_rhythm function."""

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_rhythm_features")
    def test_returns_16_dimensions(self, mock_rhythm, mock_load, mock_ensure):
        """Verify analyze_rhythm returns 16-dimensional vector."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_rhythm.return_value = np.random.rand(RHYTHM_DIMS)

        features = analyze_rhythm(b"test audio")

        assert features.shape == (RHYTHM_DIMS,)
        mock_load.assert_called_once_with(b"test audio")
        mock_rhythm.assert_called_once()

    @patch("src.audio_analyzer._ensure_essentia")
    def test_empty_bytes_raises_value_error(self, mock_ensure):
        """Verify empty bytes raises ValueError."""
        with pytest.raises(ValueError):
            analyze_rhythm(b"")


class TestAnalyzeTrack:
    """Test analyze_track function."""

    def test_invalid_feature_type_raises_value_error(self):
        """Verify invalid feature_type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            analyze_track(b"test", feature_type="invalid")
        assert "Invalid feature_type" in str(exc_info.value)

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_hpcp_features")
    @patch("src.audio_analyzer._extract_key_features")
    def test_melody_type_returns_26_dims(
        self, mock_key, mock_hpcp, mock_load, mock_ensure
    ):
        """Verify feature_type='melody' returns 26 dimensions."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_hpcp.return_value = np.random.rand(24)
        mock_key.return_value = np.array([0.8, 1.0])

        features = analyze_track(b"test", feature_type="melody")

        assert features.shape == (MELODY_DIMS,)

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_rhythm_features")
    def test_rhythm_type_returns_16_dims(self, mock_rhythm, mock_load, mock_ensure):
        """Verify feature_type='rhythm' returns 16 dimensions."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_rhythm.return_value = np.random.rand(RHYTHM_DIMS)

        features = analyze_track(b"test", feature_type="rhythm")

        assert features.shape == (RHYTHM_DIMS,)

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_hpcp_features")
    @patch("src.audio_analyzer._extract_key_features")
    @patch("src.audio_analyzer._extract_rhythm_features")
    def test_both_type_returns_42_dims(
        self, mock_rhythm, mock_key, mock_hpcp, mock_load, mock_ensure
    ):
        """Verify feature_type='both' returns 42 dimensions."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_hpcp.return_value = np.random.rand(24)
        mock_key.return_value = np.array([0.8, 1.0])
        mock_rhythm.return_value = np.random.rand(RHYTHM_DIMS)

        features = analyze_track(b"test", feature_type="both")

        assert features.shape == (BOTH_DIMS,)

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_hpcp_features")
    @patch("src.audio_analyzer._extract_key_features")
    @patch("src.audio_analyzer._extract_rhythm_features")
    def test_default_feature_type_is_both(
        self, mock_rhythm, mock_key, mock_hpcp, mock_load, mock_ensure
    ):
        """Verify default feature_type is 'both'."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_hpcp.return_value = np.random.rand(24)
        mock_key.return_value = np.array([0.8, 1.0])
        mock_rhythm.return_value = np.random.rand(RHYTHM_DIMS)

        features = analyze_track(b"test")

        assert features.shape == (BOTH_DIMS,)


class TestFeatureVectorProperties:
    """Test properties of extracted feature vectors."""

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_hpcp_features")
    @patch("src.audio_analyzer._extract_key_features")
    @patch("src.audio_analyzer._extract_rhythm_features")
    def test_features_are_float_type(
        self, mock_rhythm, mock_key, mock_hpcp, mock_load, mock_ensure
    ):
        """Verify features are numpy float arrays."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_hpcp.return_value = np.random.rand(24).astype(np.float64)
        mock_key.return_value = np.array([0.8, 1.0])
        mock_rhythm.return_value = np.random.rand(RHYTHM_DIMS)

        features = analyze_track(b"test", feature_type="both")

        assert np.issubdtype(features.dtype, np.floating)

    @patch("src.audio_analyzer._ensure_essentia")
    @patch("src.audio_analyzer._load_audio_from_bytes")
    @patch("src.audio_analyzer._extract_hpcp_features")
    @patch("src.audio_analyzer._extract_key_features")
    @patch("src.audio_analyzer._extract_rhythm_features")
    def test_features_are_finite(
        self, mock_rhythm, mock_key, mock_hpcp, mock_load, mock_ensure
    ):
        """Verify features contain no NaN or Inf values."""
        mock_load.return_value = np.zeros(SAMPLE_RATE * 2)
        mock_hpcp.return_value = np.random.rand(24)
        mock_key.return_value = np.array([0.8, 1.0])
        mock_rhythm.return_value = np.random.rand(RHYTHM_DIMS)

        features = analyze_track(b"test", feature_type="both")

        assert np.all(np.isfinite(features))


@pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not installed")
class TestEssentiaIntegration:
    """Integration tests that require Essentia to be installed.

    These tests are skipped if Essentia is not available.
    """

    @pytest.fixture
    def sample_audio(self):
        """Generate a simple synthetic audio signal for testing."""
        # Generate a 2-second sine wave at 440 Hz
        duration = 2.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
        # Mix of frequencies for more interesting features
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t)
            + 0.3 * np.sin(2 * np.pi * 880 * t)
            + 0.2 * np.sin(2 * np.pi * 1320 * t)
        ).astype(np.float32)
        return audio

    def test_hpcp_features_with_real_essentia(self, sample_audio):
        """Test HPCP feature extraction with real Essentia."""
        from src.audio_analyzer import _extract_hpcp_features

        features = _extract_hpcp_features(sample_audio)

        assert features.shape == (24,)
        # HPCP values should be in [0, 1] range
        assert np.all(features[:12] >= 0)
        assert np.all(features[:12] <= 1)

    def test_key_features_with_real_essentia(self, sample_audio):
        """Test key feature extraction with real Essentia."""
        from src.audio_analyzer import _extract_key_features

        features = _extract_key_features(sample_audio)

        assert features.shape == (2,)
        # Key strength should be in [0, 1]
        assert 0 <= features[0] <= 1
        # Mode should be 0 or 1
        assert features[1] in [0.0, 1.0]

    def test_rhythm_features_with_real_essentia(self, sample_audio):
        """Test rhythm feature extraction with real Essentia."""
        from src.audio_analyzer import _extract_rhythm_features

        features = _extract_rhythm_features(sample_audio)

        assert features.shape == (RHYTHM_DIMS,)
        # All values should be in reasonable range
        assert np.all(np.isfinite(features))


class TestBPMNormalization:
    """Test BPM normalization logic."""

    def test_bpm_60_normalizes_to_0(self):
        """Verify BPM=60 normalizes to 0."""
        # (60 - 60) / 120 = 0
        bpm = 60
        normalized = np.clip((bpm - 60) / 120, 0, 1)
        assert normalized == pytest.approx(0.0)

    def test_bpm_120_normalizes_to_half(self):
        """Verify BPM=120 normalizes to 0.5."""
        # (120 - 60) / 120 = 0.5
        bpm = 120
        normalized = np.clip((bpm - 60) / 120, 0, 1)
        assert normalized == pytest.approx(0.5)

    def test_bpm_180_normalizes_to_1(self):
        """Verify BPM=180 normalizes to 1.0."""
        # (180 - 60) / 120 = 1.0
        bpm = 180
        normalized = np.clip((bpm - 60) / 120, 0, 1)
        assert normalized == pytest.approx(1.0)

    def test_bpm_below_60_clips_to_0(self):
        """Verify BPM below 60 clips to 0."""
        bpm = 40
        normalized = np.clip((bpm - 60) / 120, 0, 1)
        assert normalized == pytest.approx(0.0)

    def test_bpm_above_180_clips_to_1(self):
        """Verify BPM above 180 clips to 1."""
        bpm = 200
        normalized = np.clip((bpm - 60) / 120, 0, 1)
        assert normalized == pytest.approx(1.0)
