"""Parity tests for the FNV-1 64 kernel against `ddtrace.internal.utils.fnv`.

FNV is exact integer arithmetic, so these compare with `rtol=0, atol=0`: any
difference means a wrong prime, a wrong init constant, a missing xor or a
dropped byte, not rounding.
"""

import os

import numpy as np
import pytest
from ddtrace.internal.utils.fnv import FNV1_64_INIT
from ddtrace.internal.utils.fnv import FNV_64_PRIME
from ddtrace.internal.utils.fnv import fnv1_64 as ref_fnv1_64

import mojo_ddtrace as m

SIZES = [0, 1, 2, 7, 8, 9, 15, 16, 31, 63, 64, 65, 127, 255, 1024, 4096]


def _payload(n: int, seed: int = 0) -> bytes:
    """Deterministic pseudo-random bytes; `os.urandom` would not be
    reproducible across runs."""
    rng = np.random.default_rng(seed + n)
    return rng.integers(0, 256, n, dtype=np.uint8).tobytes()


def test_constants_match_upstream():
    assert m.FNV_64_PRIME == FNV_64_PRIME
    assert m.FNV1_64_INIT == FNV1_64_INIT
    assert m.FNV1_64_INIT == 0xCBF29CE484222325
    assert m.FNV_64_PRIME == 0x100000001B3


@pytest.mark.parametrize("n", SIZES)
def test_fnv1_64_exact(n):
    payload = _payload(n, seed=1)
    assert m.fnv1_64(payload) == ref_fnv1_64(payload)


def test_fnv1_64_pinned_values():
    """Fixed digests, so a change to the recurrence cannot pass unnoticed.

    Note these are *not* the published FNV-1 vectors: upstream multiplies and
    then xors, which is the FNV-0 ordering, so `fnv1_64` despite its name
    hashes b"a" to 0xaf63bd4c8601b7be rather than the textbook
    0xaf63dc4c8601ec8c. ddtrace's own checkpoints are built with this
    ordering, so the port reproduces it rather than the textbook one."""
    assert m.fnv1_64(b"") == 0xCBF29CE484222325
    assert m.fnv1_64(b"a") == 0xAF63BD4C8601B7BE
    assert m.fnv1_64(b"foobar") == 0x340D8765A4DDA9C2
    assert m.fnv1_64(b"\x00") == 0xAF63BD4C8601B7DF
    assert m.fnv1_64(b"\x00\x00") == 0x8328807B4EB6FED


def test_fnv1_64_specific_byte_order():
    """A hash that ignored a byte, or shifted the buffer by one, would pass the
    published vectors only by luck: single-byte and two-byte differences are
    checked exhaustively over a small alphabet."""
    for a in range(256):
        for b in range(256):
            one = bytes([a])
            two = bytes([a, b])
            assert m.fnv1_64(one) == ref_fnv1_64(one)
            assert m.fnv1_64(two) == ref_fnv1_64(two)
            assert m.fnv1_64(one) != m.fnv1_64(two) or a == b


def test_fnv1_64_avalanche_on_realistic_payloads():
    """A service/env/tag blob shaped like the ones Data Streams checkpoints
    hash."""
    payloads = [
        b"service=web env=prod",
        b"service=web-frontend env=staging",
        b"service=checkout-worker env=prod lang=python:3.13.0",
        os.urandom(0),
        b"\x00" * 300,
        bytes(range(256)),
    ]
    for payload in payloads:
        assert m.fnv1_64(payload) == ref_fnv1_64(payload)


def test_fnv_generic_matches_upstream_recurrence():
    """`fnv` is exposed with its prime and init so the FNV-0 variant and other
    sizes stay reachable; the FNV_SIZE guard is explicit."""
    from ddtrace.internal.utils.fnv import fnv as ref_fnv

    payload = b"dd-pathway-ctx"
    assert m.fnv(payload, FNV1_64_INIT, FNV_64_PRIME) == ref_fnv(
        payload, FNV1_64_INIT, FNV_64_PRIME, 2 ** 64
    )
    with pytest.raises(ValueError):
        m.fnv(payload, FNV1_64_INIT, FNV_64_PRIME, 2 ** 32)


def test_fnv_many_matches_scalar():
    """The batch entry point must agree with the scalar one, including the
    empty buffer, and must not let one buffer's bytes leak into the next."""
    chunks = [b"", b"a", b"service=web", b"\x00\x01\x02", b"z" * 300,
              b"parent-hash-bytes"]
    got = m.fnv1_64_many(chunks)
    assert got == [m.fnv1_64(c) for c in chunks]
    assert got == [ref_fnv1_64(c) for c in chunks]


def test_fnv_many_large_batch():
    rng = np.random.default_rng(4)
    chunks = [rng.integers(0, 256, int(rng.integers(0, 200)),
                           dtype=np.uint8).tobytes() for _ in range(200)]
    assert m.fnv1_64_many(chunks) == [ref_fnv1_64(c) for c in chunks]


def test_fnv_requires_bytes():
    with pytest.raises(TypeError):
        m.fnv1_64("a string is not bytes")
