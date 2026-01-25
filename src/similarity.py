"""
Faiss-based vector similarity search for finding similar tracks.
"""

import faiss
import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2 normalize vectors for cosine similarity via inner product.

    Args:
        vectors: 2D numpy array of shape (n, dim) or 1D array of shape (dim,)

    Returns:
        L2 normalized vectors with same shape as input
    """
    vectors = np.atleast_2d(vectors).astype(np.float32)

    # Compute L2 norms
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)

    return vectors / norms


def build_faiss_index(feature_vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a Faiss index from feature vectors.

    Uses inner product (IP) index with L2 normalized vectors,
    which is equivalent to cosine similarity.

    Args:
        feature_vectors: 2D numpy array of shape (n_tracks, feature_dim)

    Returns:
        Faiss IndexFlatIP for similarity search

    Raises:
        ValueError: If feature_vectors is empty or has wrong shape
    """
    if feature_vectors is None or len(feature_vectors) == 0:
        raise ValueError("feature_vectors cannot be empty")

    vectors = np.atleast_2d(feature_vectors).astype(np.float32)

    if vectors.ndim != 2:
        raise ValueError(
            f"feature_vectors must be 2D, got shape {feature_vectors.shape}"
        )

    # L2 normalize for cosine similarity
    normalized = normalize_vectors(vectors)

    # Create inner product index (cosine similarity with normalized vectors)
    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Add vectors to index
    index.add(normalized)

    return index


def find_similar(
    ref_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    k: int = 10,
    threshold: float = 0.0,
) -> list[tuple[int, float]]:
    """
    Find k most similar tracks to reference using Faiss.

    Uses cosine similarity (via L2 normalized vectors + inner product).
    Similarity scores range from -1 to 1, where 1 is identical.

    Args:
        ref_vector: Feature vector of reference track (1D array)
        candidate_vectors: Feature vectors of candidate tracks (2D array)
        k: Number of similar tracks to return
        threshold: Minimum similarity threshold (-1.0 to 1.0, default 0.0)

    Returns:
        List of (track_index, similarity_score) tuples, sorted by similarity descending

    Raises:
        ValueError: If inputs are invalid
    """
    if ref_vector is None or len(ref_vector) == 0:
        raise ValueError("ref_vector cannot be empty")

    if candidate_vectors is None or len(candidate_vectors) == 0:
        raise ValueError("candidate_vectors cannot be empty")

    ref = np.atleast_2d(ref_vector).astype(np.float32)
    candidates = np.atleast_2d(candidate_vectors).astype(np.float32)

    if ref.shape[1] != candidates.shape[1]:
        raise ValueError(
            f"Dimension mismatch: ref_vector has {ref.shape[1]} dims, "
            f"candidate_vectors has {candidates.shape[1]} dims"
        )

    # Limit k to number of candidates
    k = min(k, len(candidates))

    # Build index from candidates
    index = build_faiss_index(candidates)

    # Normalize query vector
    ref_normalized = normalize_vectors(ref)

    # Search for k nearest neighbors
    similarities, indices = index.search(ref_normalized, k)

    # Convert to list of tuples, filtering by threshold
    results = []
    for idx, sim in zip(indices[0], similarities[0], strict=True):
        if idx >= 0 and sim >= threshold:  # idx=-1 means not found
            results.append((int(idx), float(sim)))

    return results


def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First feature vector
        vec2: Second feature vector

    Returns:
        Cosine similarity score (-1 to 1)

    Raises:
        ValueError: If vectors have different dimensions
    """
    v1 = np.atleast_1d(vec1).astype(np.float32)
    v2 = np.atleast_1d(vec2).astype(np.float32)

    if v1.shape != v2.shape:
        raise ValueError(
            f"Dimension mismatch: vec1 has shape {v1.shape}, vec2 has shape {v2.shape}"
        )

    # L2 normalize
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))
