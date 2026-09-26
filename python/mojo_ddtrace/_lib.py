"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.

Signed varint values cross as their two's complement bit pattern, so the
shim -- not the kernel -- owns the sign convention.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-ddtrace.so"

MAX_VAR_LEN_64 = 9
"""Upstream's cap on varint payload bytes; the kernel reproduces it."""


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    fn = lib.dd_fnv
    fn.restype = ctypes.c_uint64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_uint64,
                   ctypes.c_uint64]

    fn = lib.dd_fnv_many
    fn.restype = None
    fn.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_int64, ctypes.c_uint64,
                                          ctypes.c_uint64, ctypes.c_int64]

    fn = lib.dd_encode_var_uint_64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_uint64, ctypes.c_int64]

    fn = lib.dd_encode_var_int_64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_uint64, ctypes.c_int64]

    fn = lib.dd_decode_var_uint_64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fn = lib.dd_decode_var_int_64
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fn = lib.dd_encode_var_int_64_many
    fn.restype = None
    fn.argtypes = [ctypes.c_int64] * 4

    fn = lib.dd_decode_var_int_64_stream
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 5

    fn = lib.dd_bitset_popcount
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]

    fn = lib.dd_bitset_expand
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 4

    fn = lib.dd_bitset_or
    fn.restype = None
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fn = lib.dd_bitset_add
    fn.restype = None
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fn = lib.dd_bitset_popcount_many
    fn.restype = None
    fn.argtypes = [ctypes.c_int64] * 4

    fn = lib.dd_cosine_similarity
    fn.restype = None
    fn.argtypes = [ctypes.c_int64] * 4

    fn = lib.dd_cosine_similarity_pairs
    fn.restype = None
    fn.argtypes = [ctypes.c_int64] * 5
    return lib


lib = _load()


def _addr(a) -> int:
    return a.ctypes.data


# -- FNV ---------------------------------------------------------------------


def fnv(data: bytes, init: int, prime: int, size: int = 1 << 64) -> int:
    """`ddtrace.internal.utils.fnv.fnv`. `size` is accepted for API parity and
    must be 2 ** 64: a `UInt64` multiply already wraps modulo that."""
    if size != 1 << 64:
        raise ValueError("only fnv_size == 2**64 is supported, got %r" % size)
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        return int(init)
    return int(lib.dd_fnv(_addr(buf), buf.size, ctypes.c_uint64(init),
                          ctypes.c_uint64(prime)))


def fnv_many(chunks, init: int, prime: int) -> np.ndarray:
    """Hash several buffers in one call; the Data Streams checkpoint shape."""
    chunks = list(chunks)
    if not chunks:
        return np.zeros(0, dtype=np.uint64)
    joined = np.concatenate([np.frombuffer(c, dtype=np.uint8) for c in chunks])
    offsets = np.zeros(len(chunks), dtype=np.int64)
    lengths = np.zeros(len(chunks), dtype=np.int64)
    at = 0
    for i, chunk in enumerate(chunks):
        raw = np.frombuffer(chunk, dtype=np.uint8)
        offsets[i] = at
        lengths[i] = raw.size
        at += raw.size
    out = np.zeros(len(chunks), dtype=np.uint64)
    lib.dd_fnv_many(_addr(joined), _addr(offsets), _addr(lengths),
                    len(chunks), ctypes.c_uint64(init), ctypes.c_uint64(prime),
                    _addr(out))
    return out


# -- varints -----------------------------------------------------------------


def _scratch(nbytes: int = 16):
    return np.zeros(max(nbytes, 1), dtype=np.uint8)


def encode_var_uint_64(value: int) -> bytes:
    out = _scratch(MAX_VAR_LEN_64 + 1)
    written = lib.dd_encode_var_uint_64(ctypes.c_uint64(value & 0xFFFFFFFFFFFFFFFF),
                                        _addr(out))
    return out[:written].tobytes()


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def encode_var_int_64(value: int) -> bytes:
    value = int(value)
    if not _INT64_MIN <= value <= _INT64_MAX:
        # Upstream's arithmetic is unbounded, so it will happily encode 2**63
        # into a varint that its own decoder reads back as something else. The
        # kernel is Int64-bounded like the wire format, so refuse rather than
        # silently truncate.
        raise ValueError(
            "value %d is outside the Int64 range the varint format carries"
            % value
        )
    out = _scratch(MAX_VAR_LEN_64 + 1)
    written = lib.dd_encode_var_int_64(
        ctypes.c_uint64(value & 0xFFFFFFFFFFFFFFFF), _addr(out)
    )
    return out[:written].tobytes()


def decode_var_uint_64(data: bytes):
    """Returns (value, rest). Raises EOFError, as upstream does."""
    buf = np.frombuffer(data, dtype=np.uint8)
    slot = np.zeros(1, dtype=np.uint64)
    taken = lib.dd_decode_var_uint_64(_addr(buf), buf.size, _addr(slot))
    if taken < 0:
        raise EOFError()
    return int(slot[0]), buf[taken:].tobytes()


def decode_var_int_64(data: bytes):
    """Returns (value, rest). Raises EOFError, as upstream does."""
    buf = np.frombuffer(data, dtype=np.uint8)
    slot = np.zeros(1, dtype=np.uint64)
    taken = lib.dd_decode_var_int_64(_addr(buf), buf.size, _addr(slot))
    if taken < 0:
        raise EOFError()
    return int(slot[0].astype(np.int64)), buf[taken:].tobytes()


def encode_var_int_64_many(values) -> bytes:
    vals = np.ascontiguousarray(values, dtype=np.int64).view(np.uint64)
    # ascontiguousarray silently narrows a Python big int, so check the source
    # range first.
    out = np.zeros(vals.size * (MAX_VAR_LEN_64 + 1), dtype=np.uint8)
    offsets = np.zeros(vals.size + 1, dtype=np.int32)
    if vals.size:
        lib.dd_encode_var_int_64_many(_addr(vals), vals.size, _addr(out),
                                      _addr(offsets))
    return out[:int(offsets[-1])].tobytes()


def decode_var_int_64_stream(data: bytes, count: int):
    """Decode up to `count` varints, stopping at the end of the buffer.

    Returns (values, bytes_consumed). Raises EOFError on a truncated tail,
    as upstream does."""
    buf = np.frombuffer(data, dtype=np.uint8)
    values = np.zeros(max(count, 1), dtype=np.uint64)
    taken = np.zeros(max(count, 1), dtype=np.int32)
    if buf.size == 0:
        return np.zeros(0, dtype=np.int64), 0
    consumed = lib.dd_decode_var_int_64_stream(_addr(buf), buf.size, count,
                                               _addr(values), _addr(taken))
    if consumed < 0:
        raise EOFError()
    return values.view(np.int64)[:count].copy(), consumed


# -- covered-line bitset -----------------------------------------------------


def bitset_popcount(bits) -> int:
    buf = _as_buffer(bits)
    if buf.size == 0:
        return 0
    return int(lib.dd_bitset_popcount(_addr(buf), buf.size))


def bitset_expand(bits) -> np.ndarray:
    buf = _as_buffer(bits)
    total = bitset_popcount(buf)
    out = np.zeros(max(total, 1), dtype=np.int32)
    written = lib.dd_bitset_expand(_addr(buf), buf.size, _addr(out), out.size)
    if written < 0:
        raise ValueError("bitset holds more lines than the output buffer allows")
    return out[:written].copy()


def _as_buffer(buf) -> np.ndarray:
    """Zero-copy view of any bytes-like object as contiguous uint8.

    A bytearray (what `CoverageLines` stores) and a memoryview of one come back
    writable, so the in-place kernels write through to the container; immutable
    `bytes` come back read-only, which is all a read-only kernel needs.
    """
    if isinstance(buf, np.ndarray):
        return np.ascontiguousarray(buf, dtype=np.uint8)
    return np.frombuffer(buf, dtype=np.uint8)


def bitset_or(dst, src) -> None:
    dst_view = _as_buffer(dst)
    src_view = _as_buffer(src)
    if src_view.size:
        lib.dd_bitset_or(_addr(dst_view), _addr(src_view), src_view.size)


def bitset_add(dst, line: int) -> None:
    view = _as_buffer(dst)
    if view.size:
        lib.dd_bitset_add(_addr(view), view.size, line)


def bitset_popcount_many(buffers) -> np.ndarray:
    buffers = list(buffers)
    joined = (np.concatenate([_as_buffer(b) for b in buffers]) if buffers
              else np.zeros(0, dtype=np.uint8))
    ptrs = np.zeros(len(buffers), dtype=np.int64)
    lengths = np.zeros(len(buffers), dtype=np.int32)
    for i, buf in enumerate(buffers):
        raw = _as_buffer(buf)
        ptrs[i] = raw.ctypes.data
        lengths[i] = raw.size
    out = np.zeros(max(len(buffers), 1), dtype=np.int32)
    if buffers:
        lib.dd_bitset_popcount_many(_addr(ptrs), _addr(lengths), len(buffers),
                                    _addr(out))
    return out[:len(buffers)].copy()


# -- cosine similarity -------------------------------------------------------


def cosine_similarity_parts(a, b) -> np.ndarray:
    """[dot, |a|^2, |b|^2, similarity] for one pair."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.size != b.size:
        raise ValueError(
            f"Vectors must have same length: {a.size} != {b.size}"
        )
    out = np.zeros(4, dtype=np.float64)
    if a.size:
        lib.dd_cosine_similarity(_addr(a), _addr(b), a.size, _addr(out))
    return out


def cosine_similarity_pairs(a, b, n: int, pairs: int) -> np.ndarray:
    """Score `pairs` vectors of length `n` laid out back to back.

    Returns an (pairs, 4) array of [dot, |a|^2, |b|^2, similarity]: the shape
    an evaluator scoring one query embedding against many candidates wants."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    expected = n * pairs
    if a.size != expected or b.size != expected:
        raise ValueError(
            f"expected {expected} values per side, got {a.size} and {b.size}"
        )
    out = np.zeros(max(pairs * 4, 4), dtype=np.float64)
    if pairs:
        lib.dd_cosine_similarity_pairs(_addr(a), _addr(b), n, pairs, _addr(out))
    return out.reshape(-1, 4)
