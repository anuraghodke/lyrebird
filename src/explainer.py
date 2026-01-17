"""
Generate human-readable explanations for why tracks are similar.
"""


def generate_reasoning(
    ref_features: dict,
    candidate_features: dict,
    feature_type: str = "both",
) -> list[str]:
    """
    Generate 3-4 bullet points explaining why tracks are similar.

    Args:
        ref_features: Extracted features from reference track
        candidate_features: Extracted features from candidate track
        feature_type: One of "melody", "rhythm", or "both"

    Returns:
        List of 3-4 explanation strings, e.g.:
        - "Similar melodic contour in the chorus"
        - "Matching harmonic progression (both in C major)"
        - "Comparable tempo at 120 BPM"
        - "Similar rhythmic syncopation patterns"
    """
    # Placeholder - will be implemented in Step 5
    raise NotImplementedError("Will be implemented in Step 5: Reasoning Generator")
