"""
Generate human-readable explanations for why tracks are similar.

Uses rule-based interpretation of audio features to produce explanations.
"""

import numpy as np

# Feature dimension constants (must match audio_analyzer.py)
HPCP_SIZE = 12
MELODY_DIMS = 26
RHYTHM_DIMS = 16

# Musical key names
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _interpret_melody_features(features: np.ndarray) -> dict:
    """
    Interpret melody feature vector into human-readable properties.

    Args:
        features: 26-dim melody feature vector
            - HPCP mean (12)
            - HPCP std (12)
            - key strength (1)
            - mode (1): 0=minor, 1=major

    Returns:
        Dictionary with interpreted properties
    """
    features = np.atleast_1d(features).astype(np.float32)

    hpcp_mean = features[:HPCP_SIZE]
    hpcp_std = features[HPCP_SIZE : HPCP_SIZE * 2]
    key_strength = features[24] if len(features) > 24 else 0.0
    mode_value = features[25] if len(features) > 25 else 0.5

    # Determine dominant pitch class
    dominant_pitch = int(np.argmax(hpcp_mean))
    key_name = KEY_NAMES[dominant_pitch]

    # Determine mode
    mode = "major" if mode_value > 0.5 else "minor"

    # Calculate harmonic complexity (based on HPCP std)
    harmonic_complexity = float(np.mean(hpcp_std))

    return {
        "key": key_name,
        "mode": mode,
        "key_strength": float(key_strength),
        "dominant_pitch": dominant_pitch,
        "harmonic_complexity": harmonic_complexity,
        "hpcp_mean": hpcp_mean,
    }


def _interpret_rhythm_features(features: np.ndarray) -> dict:
    """
    Interpret rhythm feature vector into human-readable properties.

    Args:
        features: 16-dim rhythm feature vector
            - BPM normalized (1)
            - BPM confidence (1)
            - Beat regularity (1)
            - Onset rate (1)
            - Onset stats (4)
            - Beat histogram (8)

    Returns:
        Dictionary with interpreted properties
    """
    features = np.atleast_1d(features).astype(np.float32)

    # Denormalize BPM: normalized = (bpm - 60) / 120, so bpm = normalized * 120 + 60
    bpm_normalized = features[0] if len(features) > 0 else 0.5
    bpm = bpm_normalized * 120 + 60

    confidence = features[1] if len(features) > 1 else 0.0
    beat_regularity = features[2] if len(features) > 2 else 0.0
    onset_rate = features[3] if len(features) > 3 else 0.0

    # Onset stats
    onset_std = features[5] if len(features) > 5 else 0.0

    # Beat histogram (8 bins)
    beat_histogram = features[8:16] if len(features) >= 16 else np.zeros(8)

    # Determine tempo category
    if bpm < 80:
        tempo_category = "slow"
    elif bpm < 120:
        tempo_category = "moderate"
    elif bpm < 150:
        tempo_category = "upbeat"
    else:
        tempo_category = "fast"

    # Determine rhythmic feel based on beat histogram
    dominant_beat_bin = int(np.argmax(beat_histogram)) if len(beat_histogram) > 0 else 0

    return {
        "bpm": float(bpm),
        "bpm_normalized": float(bpm_normalized),
        "confidence": float(confidence),
        "beat_regularity": float(beat_regularity),
        "onset_rate": float(onset_rate),
        "onset_variability": float(onset_std),
        "tempo_category": tempo_category,
        "dominant_beat_bin": dominant_beat_bin,
    }


def _compare_melody(ref: dict, candidate: dict) -> list[str]:
    """
    Generate melody comparison explanations.

    Args:
        ref: Interpreted reference melody features
        candidate: Interpreted candidate melody features

    Returns:
        List of explanation strings
    """
    explanations = []

    # Key comparison
    if ref["key"] == candidate["key"] and ref["mode"] == candidate["mode"]:
        explanations.append(
            f"Both tracks are in {ref['key']} {ref['mode']}"
        )
    elif ref["mode"] == candidate["mode"]:
        explanations.append(
            f"Both tracks have a {ref['mode']} tonality"
        )

    # Harmonic complexity comparison
    ref_complexity = ref["harmonic_complexity"]
    cand_complexity = candidate["harmonic_complexity"]
    complexity_diff = abs(ref_complexity - cand_complexity)

    if complexity_diff < 0.1:
        if ref_complexity > 0.15:
            explanations.append("Both tracks have rich harmonic content")
        else:
            explanations.append("Both tracks have simple, clean harmonies")

    # HPCP similarity (pitch class profile)
    hpcp_similarity = float(
        np.dot(ref["hpcp_mean"], candidate["hpcp_mean"])
        / (np.linalg.norm(ref["hpcp_mean"]) * np.linalg.norm(candidate["hpcp_mean"]) + 1e-8)
    )
    if hpcp_similarity > 0.8:
        explanations.append("Very similar melodic pitch patterns")
    elif hpcp_similarity > 0.6:
        explanations.append("Similar harmonic structure and chord progressions")

    return explanations


def _compare_rhythm(ref: dict, candidate: dict) -> list[str]:
    """
    Generate rhythm comparison explanations.

    Args:
        ref: Interpreted reference rhythm features
        candidate: Interpreted candidate rhythm features

    Returns:
        List of explanation strings
    """
    explanations = []

    # BPM comparison
    bpm_diff = abs(ref["bpm"] - candidate["bpm"])
    if bpm_diff < 5:
        explanations.append(f"Nearly identical tempo around {ref['bpm']:.0f} BPM")
    elif bpm_diff < 15:
        explanations.append(
            f"Similar tempo ({ref['bpm']:.0f} vs {candidate['bpm']:.0f} BPM)"
        )
    elif ref["tempo_category"] == candidate["tempo_category"]:
        explanations.append(f"Both tracks have a {ref['tempo_category']} tempo")

    # Beat regularity comparison
    ref_regularity = ref["beat_regularity"]
    cand_regularity = candidate["beat_regularity"]
    if abs(ref_regularity - cand_regularity) < 0.2:
        if ref_regularity > 0.7:
            explanations.append("Both tracks have a steady, consistent beat")
        elif ref_regularity < 0.4:
            explanations.append("Both tracks have a loose, varied rhythmic feel")

    # Onset rate comparison (rhythmic density)
    onset_diff = abs(ref["onset_rate"] - candidate["onset_rate"])
    if onset_diff < 0.15:
        if ref["onset_rate"] > 0.5:
            explanations.append("Similar rhythmic density with frequent accents")
        else:
            explanations.append("Both tracks have a sparse, spacious rhythm")

    return explanations


def generate_reasoning(
    ref_features: dict | np.ndarray,
    candidate_features: dict | np.ndarray,
    feature_type: str = "both",
) -> list[str]:
    """
    Generate 3-4 bullet points explaining why tracks are similar.

    Args:
        ref_features: Feature vector or dict from reference track
        candidate_features: Feature vector or dict from candidate track
        feature_type: One of "melody", "rhythm", or "both"

    Returns:
        List of 3-4 explanation strings

    Raises:
        ValueError: If feature_type is invalid
    """
    valid_types = ("melody", "rhythm", "both")
    if feature_type not in valid_types:
        raise ValueError(
            f"Invalid feature_type: {feature_type}. Must be one of {valid_types}"
        )

    # Convert to numpy arrays if dicts with 'features' key
    if isinstance(ref_features, dict) and "features" in ref_features:
        ref_features = ref_features["features"]
    if isinstance(candidate_features, dict) and "features" in candidate_features:
        candidate_features = candidate_features["features"]

    ref_arr = np.atleast_1d(ref_features).astype(np.float32)
    cand_arr = np.atleast_1d(candidate_features).astype(np.float32)

    explanations = []

    if feature_type == "melody":
        ref_melody = _interpret_melody_features(ref_arr)
        cand_melody = _interpret_melody_features(cand_arr)
        explanations.extend(_compare_melody(ref_melody, cand_melody))

    elif feature_type == "rhythm":
        ref_rhythm = _interpret_rhythm_features(ref_arr)
        cand_rhythm = _interpret_rhythm_features(cand_arr)
        explanations.extend(_compare_rhythm(ref_rhythm, cand_rhythm))

    else:  # both
        # Split combined features
        ref_melody_features = ref_arr[:MELODY_DIMS]
        ref_rhythm_features = ref_arr[MELODY_DIMS:MELODY_DIMS + RHYTHM_DIMS]
        cand_melody_features = cand_arr[:MELODY_DIMS]
        cand_rhythm_features = cand_arr[MELODY_DIMS:MELODY_DIMS + RHYTHM_DIMS]

        ref_melody = _interpret_melody_features(ref_melody_features)
        cand_melody = _interpret_melody_features(cand_melody_features)
        ref_rhythm = _interpret_rhythm_features(ref_rhythm_features)
        cand_rhythm = _interpret_rhythm_features(cand_rhythm_features)

        explanations.extend(_compare_melody(ref_melody, cand_melody))
        explanations.extend(_compare_rhythm(ref_rhythm, cand_rhythm))

    # Ensure we return 3-4 explanations
    if len(explanations) == 0:
        explanations.append("Tracks share similar overall audio characteristics")

    # Limit to 4 explanations
    return explanations[:4]
