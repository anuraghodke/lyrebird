"""Auto-generated comparison functions for lyrebird."""

import numpy as np

from src.audio_analyzer import (
    extract_hpcp_features,
    extract_key_features,
    extract_rhythm_features,
)

# Generated: 2026-01-30 22:07:35
# Prompt: "same intervals between notes, pitch agnostic"
def melodic_interval_similarity(ref_audio, candidate_audio):
    import essentia.standard as es
    
    # Extract pitch sequences using predominant melody
    melody_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=512)
    ref_pitch, ref_confidence = melody_extractor(ref_audio)
    cand_pitch, cand_confidence = melody_extractor(candidate_audio)
    
    # Filter out silent/unvoiced segments (pitch = 0 or low confidence)
    ref_valid = (ref_pitch > 0) & (ref_confidence > 0.5)
    cand_valid = (cand_pitch > 0) & (cand_confidence > 0.5)
    
    ref_pitch_filtered = ref_pitch[ref_valid]
    cand_pitch_filtered = cand_pitch[cand_valid]
    
    if len(ref_pitch_filtered) < 2 or len(cand_pitch_filtered) < 2:
        return 0.0
    
    # Convert to semitone differences (intervals)
    ref_intervals = 12 * np.log2(ref_pitch_filtered[1:] / ref_pitch_filtered[:-1])
    cand_intervals = 12 * np.log2(cand_pitch_filtered[1:] / cand_pitch_filtered[:-1])
    
    # Remove extreme jumps (likely octave errors or noise)
    ref_intervals = ref_intervals[np.abs(ref_intervals) < 24]
    cand_intervals = cand_intervals[np.abs(cand_intervals) < 24]
    
    if len(ref_intervals) == 0 or len(cand_intervals) == 0:
        return 0.0
    
    # Create interval histograms (quantized to semitones)
    bins = np.arange(-24, 25)  # -24 to +24 semitones
    ref_hist, _ = np.histogram(ref_intervals, bins=bins, density=True)
    cand_hist, _ = np.histogram(cand_intervals, bins=bins, density=True)
    
    # Compute histogram correlation
    correlation = np.corrcoef(ref_hist, cand_hist)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # Convert correlation to 0-1 similarity score
    similarity = (correlation + 1.0) / 2.0
    
    return float(np.clip(similarity, 0.0, 1.0))
