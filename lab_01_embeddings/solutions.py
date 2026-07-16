"""
============================================================
  Lab 01: Solutions
  Complete implementations for all TODO sections
============================================================
"""

import numpy as np


# ============================================================
# SOLUTION: compute_similarity_matrix (Step 3)
# ============================================================

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all embeddings.

    Uses sklearn for efficiency, but the manual numpy version
    is shown below for educational purposes.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(embeddings)


def compute_similarity_matrix_manual(embeddings: np.ndarray) -> np.ndarray:
    """
    Manual numpy implementation of cosine similarity.
    This is slower but helps understand the math.

    cos(A, B) = (A · B) / (||A|| × ||B||)
    """
    # Normalize each embedding to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Dot product of normalized vectors = cosine similarity
    similarity_matrix = np.dot(normalized, normalized.T)

    return similarity_matrix


# ============================================================
# SOLUTION: find_top_similar_pairs (Step 5)
# ============================================================

def find_top_similar_pairs_solution(
    similarity_matrix: np.ndarray,
    terms: list[dict],
    top_k: int = 10,
) -> list[tuple]:
    """
    Find the top-K most similar term pairs.

    Returns list of (score, term1_it, term2_it, cat1, cat2)
    """
    pairs = []

    # Iterate over upper triangle (i < j) to avoid duplicates
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            score = similarity_matrix[i][j]
            pairs.append((
                score,
                terms[i]["it"],
                terms[j]["it"],
                terms[i]["category"],
                terms[j]["category"],
            ))

    # Sort by similarity score, highest first
    pairs.sort(key=lambda x: x[0], reverse=True)

    return pairs[:top_k]
