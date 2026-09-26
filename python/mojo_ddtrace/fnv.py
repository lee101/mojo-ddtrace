"""FNV-1 64 over byte buffers, mirroring `ddtrace.internal.utils.fnv`.

Upstream is a Python loop doing a big-integer multiply, a modulo and an xor
per byte. The kernel does the same recurrence in `UInt64`, where the modulo is
just the wrap-around of the multiply, so the hashes are bit-identical.
"""

from ._lib import fnv, fnv_many

FNV_64_PRIME = 0x100000001B3
FNV1_64_INIT = 0xCBF29CE484222325

__all__ = ["FNV1_64_INIT", "FNV_64_PRIME", "fnv", "fnv1_64", "fnv1_64_many"]


def fnv1_64(data) -> int:
    """64-bit FNV-1 hash of `data`."""
    return fnv(bytes(data), FNV1_64_INIT, FNV_64_PRIME)


def fnv1_64_many(chunks) -> list:
    """FNV-1 of several buffers in one kernel call."""
    return [int(h) for h in fnv_many([bytes(c) for c in chunks], FNV1_64_INIT,
                                     FNV_64_PRIME)]
