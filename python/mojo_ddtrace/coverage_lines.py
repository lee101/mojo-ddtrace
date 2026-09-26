"""Covered-line bitset, mirroring `ddtrace.internal.coverage.coverage_lines`.

The container keeps line numbers big-endian within each byte, because that is
what the CI Visibility backend expects on the wire: line `L` lives in byte
`L // 8` at bit `0b1000_0000 >> (L % 8)`. Every pass over the bits -- the
popcount, the expansion to a sorted list, the OR merge, the single-bit set --
is a kernel; only the buffer growth stays in Python, as upstream.
"""

from ._lib import bitset_add, bitset_expand, bitset_or, bitset_popcount

__all__ = ["CoverageLines", "bitset_popcount_many"]


def bitset_popcount_many(buffers) -> list:
    """Number of covered lines in each of several bitsets, in one call."""
    from ._lib import bitset_popcount_many as _many

    return [int(n) for n in _many(buffers)]


class CoverageLines:
    def __init__(self, initial_size: int = 32):
        # Upstream picks 32 from the p50 source length of its own code base.
        self._lines = bytearray(initial_size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoverageLines):
            return NotImplemented
        return self._lines == other._lines

    def __hash__(self):
        return hash(bytes(self._lines))

    def __len__(self) -> int:
        return self._num_lines()

    def __bool__(self) -> bool:
        return self._num_lines() > 0

    def __copy__(self):
        new = CoverageLines()
        new._lines = bytearray(self._lines)
        return new

    def __repr__(self) -> str:
        return f"CoverageLines(num_lines={self._num_lines()})"

    def _num_lines(self) -> int:
        return bitset_popcount(self._lines)

    def add(self, line_number: int) -> None:
        lines_byte = line_number // 8
        if lines_byte >= len(self._lines):
            self._lines.extend(bytearray(lines_byte - len(self._lines) + 1))
        bitset_add(self._lines, line_number)

    def to_sorted_list(self) -> list:
        """Covered line numbers, ascending."""
        return [int(line) for line in bitset_expand(self._lines)]

    def update(self, other: "CoverageLines") -> None:
        if len(other._lines) > len(self._lines):
            self._lines.extend(bytearray(len(other._lines) - len(self._lines)))
        bitset_or(self._lines, other._lines)

    def to_bytes(self) -> bytes:
        return bytes(self._lines)

    @classmethod
    def from_list(cls, lines) -> "CoverageLines":
        coverage = cls()
        for line in lines:
            coverage.add(int(line))
        return coverage

    @classmethod
    def from_bytearray(cls, lines) -> "CoverageLines":
        coverage = cls()
        coverage._lines = bytearray(lines)
        return coverage
