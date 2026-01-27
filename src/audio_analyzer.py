"""
Essentia-based audio feature extraction for melody and rhythm analysis.
"""

import tempfile
from pathlib import Path

import numpy as np

# Constants
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_SIZE = 1024
MIN_AUDIO_DURATION = 1.0  # Minimum audio duration in seconds

# Feature dimensions
HPCP_SIZE = 12
MELODY_DIMS = 26  # HPCP mean (12) + HPCP std (12) + key strength (1) + mode (1)
RHYTHM_DIMS = 16  # BPM (1) + confidence (1) + beat regularity (1) + onset rate (1) + onset stats (4) + beat histogram (8)
BOTH_DIMS = MELODY_DIMS + RHYTHM_DIMS  # 42


class AudioAnalysisError(Exception):
    """Raised when audio analysis fails."""

    pass


def _ensure_essentia():
    """
    Check that Essentia is installed and suppress noisy warnings.

    Raises:
        ImportError: If Essentia is not installed.
    """
    try:
        import essentia
        import essentia.standard  # noqa: F401

        # Suppress informational warnings (like HPCP bandPreset warning)
        essentia.log.warningActive = False
    except ImportError as e:
        raise ImportError(
            "Essentia is required for audio analysis. "
            "Install it with: pip install 'lyrebird[audio]'"
        ) from e


def _load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Load audio from bytes into a numpy array.

    Essentia doesn't support loading from bytes directly, so we write
    to a temporary file and load from there.

    Args:
        audio_bytes: Raw audio data as bytes (MP3 format)

    Returns:
        Mono audio signal as numpy array

    Raises:
        ValueError: If audio_bytes is empty
        AudioAnalysisError: If audio cannot be decoded or is too short
    """
    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty")

    _ensure_essentia()
    from essentia.standard import MonoLoader

    # Write bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        # Load audio using Essentia
        loader = MonoLoader(filename=tmp_path, sampleRate=SAMPLE_RATE)
        audio = loader()

        # Check minimum duration
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_DURATION:
            raise AudioAnalysisError(
                f"Audio too short: {duration:.2f}s (minimum: {MIN_AUDIO_DURATION}s)"
            )

        return audio

    except RuntimeError as e:
        raise AudioAnalysisError(f"Failed to decode audio: {e}") from e
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


def _extract_hpcp_features(audio: np.ndarray) -> np.ndarray:
    """
    Extract HPCP (Harmonic Pitch Class Profile) features.

    Args:
        audio: Mono audio signal as numpy array

    Returns:
        24-dimensional vector: HPCP mean (12) + HPCP std (12)
    """
    from essentia.standard import (
        HPCP,
        FrameGenerator,
        SpectralPeaks,
        Spectrum,
        Windowing,
    )

    # Initialize algorithms
    windowing = Windowing(type="hann")
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(
        sampleRate=SAMPLE_RATE,
        magnitudeThreshold=0.0,
        maxPeaks=100,
        minFrequency=20.0,
        maxFrequency=3500.0,
        orderBy="magnitude",
    )
    hpcp = HPCP(
        sampleRate=SAMPLE_RATE,
        size=HPCP_SIZE,
        referenceFrequency=440.0,
        harmonics=4,
        bandPreset=True,
        minFrequency=40.0,
        maxFrequency=5000.0,
        nonLinear=False,
        normalized="unitSum",
    )

    # Collect HPCP vectors for each frame
    hpcp_frames = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        peaks_freq, peaks_mag = spectral_peaks(spec)
        hpcp_vector = hpcp(peaks_freq, peaks_mag)
        hpcp_frames.append(hpcp_vector)

    hpcp_frames = np.array(hpcp_frames)

    # Compute mean and std across frames
    hpcp_mean = np.mean(hpcp_frames, axis=0)
    hpcp_std = np.std(hpcp_frames, axis=0)

    return np.concatenate([hpcp_mean, hpcp_std])


def _extract_key_features(audio: np.ndarray) -> np.ndarray:
    """
    Extract key/mode features.

    Args:
        audio: Mono audio signal as numpy array

    Returns:
        2-dimensional vector: key strength (1) + mode (1, 0=minor, 1=major)
    """
    from essentia.standard import (
        HPCP,
        FrameGenerator,
        Key,
        SpectralPeaks,
        Spectrum,
        Windowing,
    )

    # Initialize algorithms
    windowing = Windowing(type="hann")
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(
        sampleRate=SAMPLE_RATE,
        magnitudeThreshold=0.0,
        maxPeaks=100,
        minFrequency=20.0,
        maxFrequency=3500.0,
        orderBy="magnitude",
    )
    hpcp = HPCP(
        sampleRate=SAMPLE_RATE,
        size=36,  # Higher resolution for key detection
        referenceFrequency=440.0,
        harmonics=4,
        bandPreset=True,
        minFrequency=40.0,
        maxFrequency=5000.0,
        nonLinear=False,
        normalized="unitSum",
    )
    key = Key(profileType="shaath")

    # Collect HPCP vectors
    hpcp_frames = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        peaks_freq, peaks_mag = spectral_peaks(spec)
        hpcp_vector = hpcp(peaks_freq, peaks_mag)
        hpcp_frames.append(hpcp_vector)

    # Compute average HPCP for key detection
    avg_hpcp = np.mean(np.array(hpcp_frames), axis=0)

    # Detect key
    key_name, scale, strength, _ = key(avg_hpcp)

    # Encode mode as binary (0=minor, 1=major)
    mode_value = 1.0 if scale == "major" else 0.0

    return np.array([strength, mode_value])


def _extract_rhythm_features(audio: np.ndarray) -> np.ndarray:
    """
    Extract rhythm features.

    Args:
        audio: Mono audio signal as numpy array

    Returns:
        16-dimensional vector:
        - BPM normalized (1)
        - BPM confidence (1)
        - Beat regularity (1)
        - Onset rate (1)
        - Onset stats: mean, std, min, max intervals (4)
        - Beat histogram (8 bins)
    """
    from essentia.standard import (
        FrameGenerator,
        OnsetDetection,
        Onsets,
        RhythmExtractor2013,
        Windowing,
    )

    # Extract BPM and beats
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, confidence, _, beat_intervals = rhythm_extractor(audio)

    # Normalize BPM to [0, 1] range: (bpm - 60) / 120
    # This maps 60-180 BPM to 0-1 range
    bpm_normalized = np.clip((bpm - 60) / 120, 0, 1)

    # Beat regularity (std of beat intervals, inverted and normalized)
    if len(beat_intervals) > 1:
        beat_std = np.std(beat_intervals)
        # Lower std = higher regularity
        beat_regularity = 1.0 / (1.0 + beat_std)
    else:
        beat_regularity = 0.0

    # Extract onsets
    windowing = Windowing(type="hann")
    onset_detection = OnsetDetection(method="hfc", sampleRate=SAMPLE_RATE)
    onsets = Onsets()

    onset_features = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
        windowed = windowing(frame)
        onset_value = onset_detection(windowed, windowed)
        onset_features.append(onset_value)

    onset_features = np.array(onset_features).reshape(1, -1)
    onset_times = onsets(onset_features, [1.0])

    # Onset rate (onsets per second)
    duration = len(audio) / SAMPLE_RATE
    onset_rate = len(onset_times) / duration if duration > 0 else 0.0
    # Normalize onset rate (typical range 0-10 onsets/sec)
    onset_rate_normalized = np.clip(onset_rate / 10.0, 0, 1)

    # Onset interval statistics
    if len(onset_times) > 1:
        onset_intervals = np.diff(onset_times)
        onset_mean = np.mean(onset_intervals)
        onset_std = np.std(onset_intervals)
        onset_min = np.min(onset_intervals)
        onset_max = np.max(onset_intervals)
        # Normalize intervals (typical range 0-2 seconds)
        onset_stats = np.array([onset_mean, onset_std, onset_min, onset_max]) / 2.0
        onset_stats = np.clip(onset_stats, 0, 1)
    else:
        onset_stats = np.zeros(4)

    # Beat histogram (8 bins based on beat intervals)
    beat_histogram = np.zeros(8)
    if len(beat_intervals) > 0:
        # Bin beat intervals into 8 bins (0.2s to 1.0s range)
        bin_edges = np.linspace(0.2, 1.0, 9)
        for interval in beat_intervals:
            bin_idx = np.searchsorted(bin_edges[1:], interval)
            if bin_idx < 8:
                beat_histogram[bin_idx] += 1
        # Normalize histogram
        if np.sum(beat_histogram) > 0:
            beat_histogram = beat_histogram / np.sum(beat_histogram)

    return np.concatenate(
        [
            [bpm_normalized],
            [confidence],
            [beat_regularity],
            [onset_rate_normalized],
            onset_stats,
            beat_histogram,
        ]
    )


def analyze_melody(audio_bytes: bytes) -> np.ndarray:
    """
    Extract melodic features from audio using Essentia.

    Features extracted (26 dimensions):
    - HPCP mean (12-bin harmonic pitch class profile)
    - HPCP std (12 bins)
    - Key strength (1)
    - Mode (1, 0=minor, 1=major)

    Args:
        audio_bytes: Raw audio data as bytes (MP3 format)

    Returns:
        Numpy array of melodic feature vector (26 dimensions)

    Raises:
        ImportError: If Essentia is not installed
        ValueError: If audio_bytes is empty
        AudioAnalysisError: If audio cannot be decoded or is too short
    """
    _ensure_essentia()

    # Load audio from bytes
    audio = _load_audio_from_bytes(audio_bytes)

    # Extract HPCP features (24 dims)
    hpcp_features = _extract_hpcp_features(audio)

    # Extract key features (2 dims)
    key_features = _extract_key_features(audio)

    # Combine into melody feature vector (26 dims)
    return np.concatenate([hpcp_features, key_features])


def analyze_rhythm(audio_bytes: bytes) -> np.ndarray:
    """
    Extract rhythmic features from audio using Essentia.

    Features extracted (16 dimensions):
    - BPM normalized (1)
    - BPM confidence (1)
    - Beat regularity (1)
    - Onset rate (1)
    - Onset interval stats: mean, std, min, max (4)
    - Beat histogram (8 bins)

    Args:
        audio_bytes: Raw audio data as bytes (MP3 format)

    Returns:
        Numpy array of rhythmic feature vector (16 dimensions)

    Raises:
        ImportError: If Essentia is not installed
        ValueError: If audio_bytes is empty
        AudioAnalysisError: If audio cannot be decoded or is too short
    """
    _ensure_essentia()

    # Load audio from bytes
    audio = _load_audio_from_bytes(audio_bytes)

    # Extract rhythm features (16 dims)
    return _extract_rhythm_features(audio)


def analyze_track(audio_bytes: bytes, feature_type: str = "both") -> np.ndarray:
    """
    Extract features from audio based on specified type.

    Args:
        audio_bytes: Raw audio data as bytes (MP3 format)
        feature_type: One of "melody", "rhythm", or "both"

    Returns:
        Numpy array of feature vector:
        - "melody": 26 dimensions
        - "rhythm": 16 dimensions
        - "both": 42 dimensions

    Raises:
        ImportError: If Essentia is not installed
        ValueError: If audio_bytes is empty or feature_type is invalid
        AudioAnalysisError: If audio cannot be decoded or is too short
    """
    valid_types = ("melody", "rhythm", "both")
    if feature_type not in valid_types:
        raise ValueError(
            f"Invalid feature_type: {feature_type}. Must be one of {valid_types}"
        )

    _ensure_essentia()

    # Load audio from bytes once
    audio = _load_audio_from_bytes(audio_bytes)

    if feature_type == "melody":
        hpcp_features = _extract_hpcp_features(audio)
        key_features = _extract_key_features(audio)
        return np.concatenate([hpcp_features, key_features])

    elif feature_type == "rhythm":
        return _extract_rhythm_features(audio)

    else:  # both
        hpcp_features = _extract_hpcp_features(audio)
        key_features = _extract_key_features(audio)
        rhythm_features = _extract_rhythm_features(audio)
        return np.concatenate([hpcp_features, key_features, rhythm_features])
