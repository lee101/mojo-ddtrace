"""Cosine similarity over embeddings, mirroring
`ddtrace.llmobs._evaluators.semantic._cosine_similarity`.

Upstream's fallback branch walks the two vectors once in Python, and its
preferred NumPy branch walks them four times (`norm`, `norm`, `dot`). The
kernel does one pass with three accumulators, and it can score a whole batch
of candidate embeddings against a query in a single call.

Mojo emits FMA, so the dot product and the magnitudes differ from a plain
Python loop in the last bits; results agree to about 1e-15 relative, and
asserted tolerances in the tests are set from that measurement.
"""

import numpy as np

from ._lib import cosine_similarity_pairs, cosine_similarity_parts

__all__ = ["cosine_similarity", "cosine_similarity_pairs"]


def cosine_similarity(vec1, vec2) -> float:
    """Cosine similarity of two equal-length vectors, in [-1, 1].

    A zero vector scores 0.0, which is upstream's rule.
    """
    return float(cosine_similarity_parts(vec1, vec2)[3])


def cosine_similarity_many(vec1, vec2, n: int) -> list:
    """Cosine similarity of `n`-dimensional vectors laid out back to back.

    Accepts a (pairs, n) array or a flat buffer of pairs * n values."""
    a = np.ascontiguousarray(vec1, dtype=np.float64).ravel()
    b = np.ascontiguousarray(vec2, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(
            f"Vectors must have same length: {a.size} != {b.size}"
        )
    if n <= 0 or a.size % n:
        raise ValueError(
            "%d values is not a whole number of %d-dimensional vectors"
            % (a.size, n)
        )
    scores = cosine_similarity_pairs(a, b, n, a.size // n)
    return [float(row[3]) for row in scores]
