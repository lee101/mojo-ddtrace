"""Parity tests for the covered-line bitset against
`ddtrace.internal.coverage.coverage_lines`.

The container is a byte-level data structure: the bit order inside each byte
is part of the CI Visibility wire format, so these compare the exact bytes and
the exact line lists against the upstream implementation.
"""

import copy

import numpy as np
import pytest
from ddtrace.internal.coverage.coverage_lines import CoverageLines as RefLines

import mojo_ddtrace as m

RANGES = [
    [],
    [0],
    [1],
    [7],
    [8],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 9, 80],
    [31, 32, 33],
    list(range(0, 256)),
    [255, 256, 511],
    [1000, 2000, 3000, 12345],
]


@pytest.mark.parametrize("lines", RANGES)
def test_from_list_parity(lines):
    mine, theirs = m.CoverageLines.from_list(lines), RefLines.from_list(lines)
    assert mine.to_sorted_list() == theirs.to_sorted_list()
    assert mine.to_bytes() == bytes(theirs.to_bytes())
    assert len(mine) == len(theirs)
    assert bool(mine) == bool(theirs)
    assert repr(mine) == repr(theirs)


def test_every_byte_pattern():
    """Exhaustive over the 256 possible byte values, so a reversed bit order or
    a shifted base cannot slip through."""
    for pattern in range(256):
        raw = bytes([pattern] * 5)
        mine = m.CoverageLines.from_bytearray(bytearray(raw))
        theirs = RefLines.from_bytearray(bytearray(raw))
        assert mine.to_sorted_list() == theirs.to_sorted_list(), pattern
        assert len(mine) == len(theirs), pattern


def test_bit_order_is_big_endian_within_the_byte():
    """Line 0 is bit 0x80 of byte 0. This is what the CI Visibility backend
    expects, and inverting it would still round-trip through the container."""
    lines = m.CoverageLines.from_list([0, 1, 2])
    assert lines.to_bytes()[0] == 0b1110_0000
    assert m.CoverageLines.from_list([7]).to_bytes()[0] == 0b0000_0001
    assert m.CoverageLines.from_list([8]).to_bytes()[1] == 0b1000_0000


def test_random_line_sets_match():
    rng = np.random.default_rng(20240917)
    for _ in range(50):
        size = int(rng.integers(0, 300))
        lines = sorted(rng.choice(4000, size=size, replace=False).tolist())
        mine, theirs = m.CoverageLines.from_list(lines), RefLines.from_list(lines)
        assert mine.to_sorted_list() == theirs.to_sorted_list()
        assert mine.to_bytes() == bytes(theirs.to_bytes())
        assert len(mine) == len(theirs) == len(set(lines))


def test_duplicate_adds_count_once():
    lines = m.CoverageLines.from_list([5, 5, 5, 6])
    assert lines.to_sorted_list() == [5, 6]
    assert len(lines) == 2


def test_update_merges_and_grows():
    mine = m.CoverageLines.from_list([1, 2, 3])
    theirs = RefLines.from_list([1, 2, 3])
    mine.update(m.CoverageLines.from_list([2, 3, 100]))
    theirs.update(RefLines.from_list([2, 3, 100]))
    assert mine.to_sorted_list() == theirs.to_sorted_list()
    assert mine.to_bytes() == bytes(theirs.to_bytes())

    # update into a longer source must extend the destination, upstream's
    # `if len(other) > len(self)` branch.
    small, small_ref = m.CoverageLines(1), RefLines(1)
    big = m.CoverageLines.from_list([7, 8, 9])
    big_ref = RefLines.from_list([7, 8, 9])
    small.update(big)
    small_ref.update(big_ref)
    assert small.to_sorted_list() == small_ref.to_sorted_list()
    assert small.to_bytes() == bytes(small_ref.to_bytes())


def test_update_with_empty_source_is_a_noop():
    mine, theirs = m.CoverageLines.from_list([1, 2]), RefLines.from_list([1, 2])
    mine.update(m.CoverageLines())
    theirs.update(RefLines())
    assert mine.to_sorted_list() == theirs.to_sorted_list()
    assert mine.to_bytes() == bytes(theirs.to_bytes())


def test_add_grows_the_buffer_like_upstream():
    mine, theirs = m.CoverageLines(), RefLines()
    for line in (0, 40, 900, 5):
        mine.add(line)
        theirs.add(line)
    assert mine.to_bytes() == bytes(theirs.to_bytes())
    assert mine.to_sorted_list() == theirs.to_sorted_list()
    assert len(mine) == len(theirs)


def test_equality_copy_and_repr():
    a, b = m.CoverageLines.from_list([1, 2, 3]), m.CoverageLines.from_list([1, 2])
    assert a == m.CoverageLines.from_list([1, 2, 3])
    assert a != b
    assert (a == "not a CoverageLines") is False

    clone = copy.copy(a)
    clone.add(1000)
    assert clone.to_sorted_list() == [1, 2, 3, 1000]
    assert a.to_sorted_list() == [1, 2, 3]


def test_from_bytearray_keeps_the_bytes():
    # 0xc0 -> lines 0, 1;  0x40 -> line 8 + 1 = 9;  0x00 -> nothing;
    # 0x80 in byte 3 -> line 3 * 8 + 0 = 24.
    raw = bytearray(b"\xc0\x40\x00\x80")
    lines = m.CoverageLines.from_bytearray(raw)
    assert lines.to_sorted_list() == [0, 1, 9, 24]
    assert lines.to_bytes() == bytes(raw)


def test_empty_container():
    empty = m.CoverageLines()
    assert len(empty) == 0
    assert not empty
    assert empty.to_sorted_list() == []
    assert empty.to_bytes() == bytearray(32)
    assert empty.to_bytes() == bytes(RefLines().to_bytes())


def test_popcount_many_matches_scalar():
    rng = np.random.default_rng(3)
    sets = []
    for _ in range(64):
        size = int(rng.integers(0, 200))
        lines = sorted(rng.choice(3000, size=size, replace=False).tolist())
        sets.append(m.CoverageLines.from_list(lines))
    got = m.coverage_lines.bitset_popcount_many([s.to_bytes() for s in sets])
    assert got == [len(s) for s in sets]
    assert got == [len(RefLines.from_bytearray(bytearray(s.to_bytes())))
                   for s in sets]


def test_popcount_many_empty_input():
    assert list(m.coverage_lines.bitset_popcount_many([])) == []
