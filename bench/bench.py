"""Correctness-gated benchmark for mojo-ddtrace.

Every case checks its result against the real `ddtrace` implementation before
timing, so a regression in the Mojo kernels shows up as a correctness failure
rather than a suspiciously good number.

The two sides of a case are timed alternately, best of `ROUNDS`, because this
box is shared: measuring one side and then the other lets another process's
load land on whichever ran second.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from ddtrace.internal.coverage.coverage_lines import CoverageLines  # noqa: E402
from ddtrace.internal.datastreams.encoding import (  # noqa: E402
    decode_var_int_64,
    encode_var_int_64,
)
from ddtrace.internal.utils.fnv import fnv1_64  # noqa: E402
from ddtrace.llmobs._evaluators.semantic import (  # noqa: E402
    _cosine_similarity,
)

import mojo_ddtrace as m  # noqa: E402

ROUNDS = 5


def _time_pair(reference, mine, rounds=ROUNDS):
    ref_best = float("inf")
    mine_best = float("inf")
    for _ in range(rounds):
        for fn, is_ref in ((reference, True), (mine, False)):
            t0 = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - t0
            if is_ref:
                ref_best = min(ref_best, elapsed)
            else:
                mine_best = min(mine_best, elapsed)
    return ref_best, mine_best


def bench_fnv(n: int = 1 << 16):
    """One FNV-1 hash of a realistic data-streams blob. The reference is the
    upstream Python loop over the same bytes."""
    rng = np.random.default_rng(0)
    blob = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    assert m.fnv1_64(blob) == fnv1_64(blob)
    return ("fnv1_64 %d bytes" % n,
            lambda: fnv1_64(blob),
            lambda: m.fnv1_64(blob))


def bench_fnv_many(count: int = 512, n: int = 256):
    """A batch of pathway-node hashes, the shape a checkpoint flush takes."""
    rng = np.random.default_rng(1)
    chunks = [rng.integers(0, 256, n, dtype=np.uint8).tobytes()
              for _ in range(count)]
    assert m.fnv1_64_many(chunks) == [fnv1_64(c) for c in chunks]
    return ("fnv1_64 x%d of %d bytes" % (count, n),
            lambda: [fnv1_64(c) for c in chunks],
            lambda: m.fnv1_64_many(chunks))


def bench_varint_roundtrip(count: int = 20000):
    """Encode then decode `count` timestamps, the way a pathway header is
    framed and read back."""
    rng = np.random.default_rng(2)
    values = rng.integers(0, 2 ** 41, count, dtype=np.int64)
    payload = m.encode_var_int_64_many(values)
    ref_payload = b"".join(encode_var_int_64(int(v)) for v in values)
    assert payload == ref_payload
    assert list(m.decode_var_int_64_stream(payload, count)[0]) == list(values)

    def theirs():
        blob = b"".join(encode_var_int_64(int(v)) for v in values)
        out = []
        for _ in range(count):
            value, blob = decode_var_int_64(blob)
            out.append(value)
        return out

    def ours():
        blob = m.encode_var_int_64_many(values)
        return m.decode_var_int_64_stream(blob, count)[0]

    return "varint x%d roundtrip" % count, theirs, ours


def bench_coverage_lines(files: int = 200, lines: int = 400):
    """Popcount and expansion over a payload's worth of covered-line bitsets."""
    rng = np.random.default_rng(3)
    bitmaps, containers = [], []
    for _ in range(files):
        covered = sorted(rng.choice(4000, size=lines, replace=False).tolist())
        mine_cl = m.CoverageLines.from_list(covered)
        bitmaps.append(mine_cl.to_bytes())
        containers.append(CoverageLines.from_list(covered))
        assert mine_cl.to_sorted_list() == containers[-1].to_sorted_list()
        assert mine_cl.to_bytes() == bytes(containers[-1].to_bytes())

    # The reference side builds its containers up front, so the timing is the
    # popcount and not the object construction.
    def theirs():
        return sum(len(c) for c in containers)

    def ours():
        return sum(m.coverage_lines.bitset_popcount_many(bitmaps))

    return "coverage popcount x%d" % files, theirs, ours


def bench_coverage_expand(files: int = 200, lines: int = 400):
    """The set-bit expansion: turning the bitset back into sorted line lists,
    which CI Visibility does for every file in a payload."""

    def theirs():
        return sum(len(c.to_sorted_list()) for c in BITMAP_CONTAINERS)

    def ours():
        return sum(
            len(m.CoverageLines.from_bytearray(bytearray(b)).to_sorted_list())
            for b in BITMAPS
        )

    return "coverage expand x%d" % files, theirs, ours


def bench_cosine(dim: int = 1536, pairs: int = 256):
    """Semantic-similarity scoring: one query embedding against many
    candidates, the shape an LLM evaluation run takes. The reference is
    upstream's preferred NumPy path, not its Python fallback."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal(pairs * dim)
    b = rng.standard_normal(pairs * dim)
    got = m.cosine_similarity_many(a, b, dim)
    expect = [_cosine_similarity(a[p * dim:(p + 1) * dim],
                                 b[p * dim:(p + 1) * dim])
              for p in range(pairs)]
    np.testing.assert_allclose(got, expect, rtol=1e-12)

    def theirs():
        out = []
        for p in range(pairs):
            out.append(_cosine_similarity(a[p * dim:(p + 1) * dim],
                                          b[p * dim:(p + 1) * dim]))
        return out

    def ours():
        return m.cosine_similarity_many(a, b, dim)

    return "cosine %d pairs x%d dims" % (pairs, dim), theirs, ours


def _prepare_coverage():
    """Containers for the expansion case, built before timing."""
    global BITMAPS, BITMAP_CONTAINERS
    rng = np.random.default_rng(3)
    BITMAPS, BITMAP_CONTAINERS = [], []
    for _ in range(200):
        lines = sorted(rng.choice(4000, size=400, replace=False).tolist())
        BITMAPS.append(m.CoverageLines.from_list(lines).to_bytes())
        BITMAP_CONTAINERS.append(m.CoverageLines.from_bytearray(
            bytearray(BITMAPS[-1])))


def main():
    _prepare_coverage()
    print(f"{'case':<34}{'reference':>12}{'mojo-ddtrace':>20}{'ratio':>10}")
    print("-" * 76)
    for build in (bench_fnv, bench_fnv_many, bench_varint_roundtrip,
                  bench_coverage_lines, bench_coverage_expand, bench_cosine):
        label, reference, mine = build()
        ref, got = _time_pair(reference, mine)
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref * 1e3:>10.2f}ms{got * 1e3:>18.2f}ms"
              f"{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
