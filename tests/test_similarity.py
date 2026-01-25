"""
Tests for Step 4: Faiss Similarity Search

Verifies:
- Vector normalization
- Faiss index building
- Similar track finding
- Cosine similarity computation
"""

import numpy as np
import pytest

from src.similarity import (
    build_faiss_index,
    compute_similarity,
    find_similar,
    normalize_vectors,
)


class TestNormalizeVectors:
    """Test normalize_vectors function."""

    def test_normalizes_single_vector(self):
        """Verify single vector is L2 normalized."""
        vec = np.array([3.0, 4.0])
        normalized = normalize_vectors(vec)

        # L2 norm should be 1
        assert np.linalg.norm(normalized) == pytest.approx(1.0)
        # Direction should be preserved
        assert normalized[0, 0] == pytest.approx(0.6)
        assert normalized[0, 1] == pytest.approx(0.8)

    def test_normalizes_multiple_vectors(self):
        """Verify multiple vectors are each L2 normalized."""
        vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
        normalized = normalize_vectors(vectors)

        # Each row should have L2 norm of 1
        for i in range(len(vectors)):
            assert np.linalg.norm(normalized[i]) == pytest.approx(1.0)

    def test_handles_zero_vector(self):
        """Verify zero vector doesn't cause division by zero."""
        vec = np.array([0.0, 0.0])
        normalized = normalize_vectors(vec)

        # Should not raise and should return zeros
        assert np.allclose(normalized, [[0.0, 0.0]])

    def test_returns_float32(self):
        """Verify output is float32."""
        vec = np.array([1.0, 2.0], dtype=np.float64)
        normalized = normalize_vectors(vec)

        assert normalized.dtype == np.float32


class TestBuildFaissIndex:
    """Test build_faiss_index function."""

    def test_builds_index_from_vectors(self):
        """Verify index is built from feature vectors."""
        vectors = np.random.rand(10, 42).astype(np.float32)
        index = build_faiss_index(vectors)

        assert index.ntotal == 10
        assert index.d == 42

    def test_raises_on_empty_vectors(self):
        """Verify ValueError is raised for empty input."""
        with pytest.raises(ValueError) as exc_info:
            build_faiss_index(np.array([]))
        assert "cannot be empty" in str(exc_info.value)

    def test_raises_on_none(self):
        """Verify ValueError is raised for None input."""
        with pytest.raises(ValueError):
            build_faiss_index(None)

    def test_handles_single_vector(self):
        """Verify single vector creates valid index."""
        vec = np.random.rand(1, 26).astype(np.float32)
        index = build_faiss_index(vec)

        assert index.ntotal == 1
        assert index.d == 26

    def test_normalizes_vectors(self):
        """Verify vectors are normalized in the index."""
        # Create vectors with known norms
        vectors = np.array([[3.0, 4.0], [5.0, 12.0]])
        index = build_faiss_index(vectors)

        # Reconstruct and check normalization
        reconstructed = np.zeros((2, 2), dtype=np.float32)
        index.reconstruct_n(0, 2, reconstructed)

        for i in range(2):
            norm = np.linalg.norm(reconstructed[i])
            assert norm == pytest.approx(1.0, abs=1e-5)


class TestFindSimilar:
    """Test find_similar function."""

    def test_finds_identical_vector(self):
        """Verify identical vector has similarity 1.0."""
        ref = np.array([1.0, 0.0, 0.0])
        candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        results = find_similar(ref, candidates, k=3)

        # First result should be the identical vector
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_finds_k_most_similar(self):
        """Verify returns k most similar tracks."""
        ref = np.array([1.0, 0.0])
        candidates = np.array(
            [
                [1.0, 0.0],  # Most similar
                [0.9, 0.1],  # Very similar
                [0.5, 0.5],  # Somewhat similar
                [0.0, 1.0],  # Orthogonal
                [-1.0, 0.0],  # Opposite
            ]
        )

        results = find_similar(ref, candidates, k=3)

        assert len(results) == 3
        # Should be sorted by similarity descending
        assert results[0][1] >= results[1][1] >= results[2][1]

    def test_respects_threshold(self):
        """Verify threshold filters low similarity results."""
        ref = np.array([1.0, 0.0])
        candidates = np.array(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-1.0, 0.0]]  # sim~1  # sim~0.7
        )  # sim~0  # sim~-1

        results = find_similar(ref, candidates, k=10, threshold=0.5)

        # Only vectors with similarity >= 0.5 should be returned
        for _idx, sim in results:
            assert sim >= 0.5

    def test_limits_k_to_candidates(self):
        """Verify k is limited to number of candidates."""
        ref = np.array([1.0, 0.0])
        candidates = np.array([[1.0, 0.0], [0.5, 0.5]])

        results = find_similar(ref, candidates, k=100)

        assert len(results) <= 2

    def test_raises_on_empty_ref(self):
        """Verify ValueError for empty reference vector."""
        with pytest.raises(ValueError) as exc_info:
            find_similar(np.array([]), np.array([[1, 2, 3]]))
        assert "ref_vector cannot be empty" in str(exc_info.value)

    def test_raises_on_empty_candidates(self):
        """Verify ValueError for empty candidates."""
        with pytest.raises(ValueError) as exc_info:
            find_similar(np.array([1, 2, 3]), np.array([]))
        assert "candidate_vectors cannot be empty" in str(exc_info.value)

    def test_raises_on_dimension_mismatch(self):
        """Verify ValueError for mismatched dimensions."""
        ref = np.array([1.0, 2.0, 3.0])
        candidates = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError) as exc_info:
            find_similar(ref, candidates)
        assert "Dimension mismatch" in str(exc_info.value)

    def test_returns_correct_indices(self):
        """Verify returned indices map to correct candidates."""
        ref = np.array([1.0, 0.0, 0.0])
        candidates = np.array(
            [
                [0.0, 1.0, 0.0],  # idx 0
                [1.0, 0.0, 0.0],  # idx 1 - most similar
                [0.0, 0.0, 1.0],  # idx 2
            ]
        )

        results = find_similar(ref, candidates, k=1)

        assert results[0][0] == 1  # Index of most similar


class TestComputeSimilarity:
    """Test compute_similarity function."""

    def test_identical_vectors_return_1(self):
        """Verify identical vectors have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = compute_similarity(vec, vec)

        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors_return_0(self):
        """Verify orthogonal vectors have similarity 0.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        sim = compute_similarity(vec1, vec2)

        assert sim == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_1(self):
        """Verify opposite vectors have similarity -1.0."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        sim = compute_similarity(vec1, vec2)

        assert sim == pytest.approx(-1.0)

    def test_raises_on_dimension_mismatch(self):
        """Verify ValueError for mismatched dimensions."""
        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError) as exc_info:
            compute_similarity(vec1, vec2)
        assert "Dimension mismatch" in str(exc_info.value)

    def test_handles_zero_vector(self):
        """Verify zero vector returns 0 similarity."""
        vec1 = np.array([0.0, 0.0])
        vec2 = np.array([1.0, 2.0])
        sim = compute_similarity(vec1, vec2)

        assert sim == 0.0

    def test_is_symmetric(self):
        """Verify similarity is symmetric."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        sim1 = compute_similarity(vec1, vec2)
        sim2 = compute_similarity(vec2, vec1)

        assert sim1 == pytest.approx(sim2)


class TestIntegration:
    """Integration tests for similarity search workflow."""

    def test_find_similar_tracks_workflow(self):
        """Test typical workflow of finding similar tracks."""
        # Simulate feature vectors for 5 tracks
        np.random.seed(42)
        track_features = np.random.rand(5, 26).astype(np.float32)

        # Create a reference that's similar to track 0
        ref_features = track_features[0] + np.random.rand(26) * 0.1

        # Find similar tracks
        results = find_similar(ref_features, track_features, k=3)

        # Track 0 should be most similar
        assert len(results) == 3
        assert results[0][0] == 0
        assert results[0][1] > 0.9  # Should be very similar

    def test_with_audio_feature_dimensions(self):
        """Test with actual audio feature dimensions (26, 16, 42)."""
        for dim in [26, 16, 42]:
            ref = np.random.rand(dim).astype(np.float32)
            candidates = np.random.rand(10, dim).astype(np.float32)

            results = find_similar(ref, candidates, k=5)

            assert len(results) == 5
            for idx, sim in results:
                assert 0 <= idx < 10
                assert -1.0 <= sim <= 1.0
