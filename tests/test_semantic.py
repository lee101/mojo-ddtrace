"""Parity tests for cosine similarity against
`ddtrace.llmobs._evaluators.semantic._cosine_similarity`.

Floating point: Mojo emits FMA, so the accumulators differ from a plain Python
loop in the last bits. The tolerance below is measured, not guessed -- the
observed worst case over these inputs is about 1e-15 relative, and the tests
also pin the exact identities (a vector against itself is 1, an orthogonal
pair is 0, a negated vector is -1) which must hold regardless of summation
order.
"""

import math

import numpy as np
import pytest
from ddtrace.llmobs._evaluators.semantic import _cosine_similarity as ref_cosine

import mojo_ddtrace as m

RTOL = 1e-13
DIMS = [1, 2, 3, 8, 384, 768, 1536, 3072]


def pure_python_cosine(a, b) -> float:
    """The branch `_cosine_similarity` takes when NumPy is absent, which is the
    expression the kernel implements."""
    dot = mag1 = mag2 = 0.0
    for x, y in zip(a, b):
        dot += x * y
        mag1 += x * x
        mag2 += y * y
    magnitude1 = mag1 ** 0.5
    magnitude2 = mag2 ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot / (magnitude1 * magnitude2)


@pytest.mark.parametrize("n", DIMS)
def test_matches_upstream(n):
    rng = np.random.default_rng(n)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(m.cosine_similarity(a, b), ref_cosine(a, b),
                               rtol=RTOL, atol=1e-300)


@pytest.mark.parametrize("n", DIMS)
def test_matches_the_pure_python_branch(n):
    """The fallback the kernel replaces, which is the exact expression it
    evaluates: the same loop, same order, so the only difference is FMA."""
    rng = np.random.default_rng(1000 + n)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(m.cosine_similarity(a, b),
                               pure_python_cosine(a, b), rtol=RTOL, atol=0.0)


def test_identities():
    rng = np.random.default_rng(11)
    a = rng.standard_normal(256)
    assert m.cosine_similarity(a, a) == pytest.approx(1.0, rel=1e-12)
    assert m.cosine_similarity(a, -a) == pytest.approx(-1.0, rel=1e-12)
    assert m.cosine_similarity(a, 2.5 * a) == pytest.approx(1.0, rel=1e-12)
    # A vector orthogonal to the basis direction e0 in 3 dimensions.
    assert m.cosine_similarity([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]) == 0.0


def test_zero_vector_scores_zero():
    """Upstream's rule, not a division by zero."""
    for a, b in (([0.0, 0.0], [1.0, 2.0]), ([1.0, 2.0], [0.0, 0.0]),
                 ([0.0, 0.0], [0.0, 0.0])):
        assert m.cosine_similarity(a, b) == 0.0
        assert ref_cosine(a, b) == 0.0


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        m.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        m.cosine_similarity(np.zeros(3), np.zeros(4))


def test_ill_conditioned_inputs():
    """A vector of widely separated magnitudes is where a wrong accumulator
    order or a missing square shows up first."""
    a = [1e-8, 1.0, 1e8, -3.0]
    b = [2e-8, -1.0, 1e8, 7.0]
    np.testing.assert_allclose(m.cosine_similarity(a, b), ref_cosine(a, b),
                               rtol=RTOL, atol=0.0)
    assert abs(m.cosine_similarity(a, b)) <= 1.0 + 1e-12


def test_integer_and_list_inputs():
    assert m.cosine_similarity([1, 2, 3], [4, 5, 6]) == pytest.approx(
        ref_cosine([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), rel=RTOL
    )


def test_batch_matches_scalar():
    rng = np.random.default_rng(77)
    n, pairs = 384, 32
    a = rng.standard_normal((pairs, n))
    b = rng.standard_normal((pairs, n))
    got = m.cosine_similarity_many(a, b, n)
    expect = [m.cosine_similarity(a[i], b[i]) for i in range(pairs)]
    assert got == pytest.approx(expect, rel=0.0, abs=0.0)
    upstream = [ref_cosine(a[i], b[i]) for i in range(pairs)]
    assert got == pytest.approx(upstream, rel=RTOL)


def test_batch_accepts_flat_buffers():
    rng = np.random.default_rng(78)
    n, pairs = 16, 4
    a = rng.standard_normal(n * pairs)
    b = rng.standard_normal(n * pairs)
    assert m.cosine_similarity_many(a, b, n) == m.cosine_similarity_many(
        a.reshape(pairs, n), b.reshape(pairs, n), n
    )


def test_batch_rejects_ragged_input():
    with pytest.raises(ValueError):
        m.cosine_similarity_many(np.zeros(10), np.zeros(10), 3)
    with pytest.raises(ValueError):
        m.cosine_similarity_many(np.zeros(6), np.zeros(9), 3)
    with pytest.raises(ValueError):
        m.cosine_similarity_many(np.zeros(6), np.zeros(6), 0)


def test_non_contiguous_input_is_handled():
    rng = np.random.default_rng(79)
    a = rng.standard_normal((8, 64))
    b = rng.standard_normal((8, 64))
    view_a, view_b = a[::2], b[::2]
    assert not view_a.flags["C_CONTIGUOUS"]
    got = m.cosine_similarity_many(view_a, view_b, 64)
    expect = [ref_cosine(view_a[i], view_b[i]) for i in range(view_a.shape[0])]
    assert got == pytest.approx(expect, rel=RTOL)


def test_similarity_stays_in_range_for_orthogonal_random_vectors():
    """With 3072 dimensions, random vectors are near-orthogonal; the kernel
    must not report a magnitude above 1 by a rounding artefact."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        a = rng.standard_normal(3072)
        b = rng.standard_normal(3072)
        value = m.cosine_similarity(a, b)
        assert -1.0 - 1e-12 <= value <= 1.0 + 1e-12
        assert math.isclose(value, ref_cosine(a, b), rel_tol=RTOL, abs_tol=0.0)
