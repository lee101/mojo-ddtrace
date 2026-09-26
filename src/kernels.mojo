"""Kernels for the pure-Python numeric inner loops of ddtrace.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

Four upstream surfaces are here, each mirrored exactly:

* `ddtrace.internal.utils.fnv` -- FNV-1 64 over a byte buffer. Python pays a
  big-int multiply, a modulo and an xor per byte; here the modulo is the
  natural wrap-around of a `UInt64`.
* `ddtrace.internal.datastreams.encoding` -- the ULEB128 / zig-zag varint
  codec, whose Python encoder reallocates its buffer once per byte.
* `ddtrace.internal.coverage.coverage_lines` -- the covered-line bitset:
  popcount, set-bit expansion, byte-wise OR merge.
* `ddtrace.llmobs._evaluators.semantic` -- cosine similarity over embeddings,
  three accumulators in one pass.

Semantics follow upstream exactly, including its cap of nine varint bytes and
the fact that the ninth byte is taken as final whatever its continuation bit
says.
"""

from std.math import sqrt

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def i32p(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# FNV-1 64
# ---------------------------------------------------------------------------
#
# ddtrace.internal.utils.fnv.fnv does, per byte,
#     hval = (hval * fnv_prime) % fnv_size
#     hval = hval ^ byte
# with fnv_size = 2 ** 64. A `UInt64` multiply already wraps modulo 2 ** 64, so
# the kernel is the same recurrence with the Python bigint arithmetic gone.


@export("dd_fnv")
def dd_fnv(data_addr: Int, n: Int, init: UInt64, prime: UInt64) abi("C") -> UInt64:
    """FNV over `n` bytes, starting from `init`. Returns 0 for an empty buffer,
    matching upstream's loop that never runs."""
    var data = bp(data_addr)
    var h = init
    for i in range(n):
        h = (h * prime) ^ UInt64(data[unsafe_offset=i])
    return h


@export("dd_fnv_many")
def dd_fnv_many(data_addr: Int, offsets_addr: Int, lengths_addr: Int, n: Int,
                init: UInt64, prime: UInt64, dst_addr: Int) abi("C"):
    """Hash `n` separate buffers in one call.

    This is the shape ddtrace's Data Streams checkpoint needs: every pathway
    node hashes a service/env/tag blob, then hashes the 16-byte struct-packed
    pair of that hash and its parent's. Doing the whole batch in one crossing
    keeps the per-call cost off the critical path.
    """
    var data = bp(data_addr)
    var offsets = i64p(offsets_addr)
    var lengths = i64p(lengths_addr)
    var dst = u64p(dst_addr)
    for t in range(n):
        var h = init
        var start = offsets[unsafe_offset=t]
        var length = lengths[unsafe_offset=t]
        for i in range(length):
            h = (h * prime) ^ UInt64(data[unsafe_offset=start + i])
        dst[unsafe_offset=t] = h


# ---------------------------------------------------------------------------
# ULEB128 / zig-zag varints
# ---------------------------------------------------------------------------
#
# ddtrace.internal.datastreams.encoding caps a varint at nine payload bytes and
# a final byte, and the decoder returns the ninth byte as final regardless of
# its continuation bit. Both quirks are reproduced here, because the
# checkpoints ddtrace exchanges are hashed and compared byte for byte.

comptime MAX_VAR_LEN_64 = 9


def _encode_var_uint_64(v: UInt64, dst_addr: Int) -> Int:
    """Write the ULEB128 encoding of `v` and return the byte count (1..10)."""
    var dst = bp(dst_addr)
    var value = v
    var written = 0
    for _ in range(MAX_VAR_LEN_64):
        if value < 0x80:
            break
        dst[unsafe_offset=written] = UInt8((value & 0xFF) | 0x80)
        value = value >> 7
        written += 1
    dst[unsafe_offset=written] = UInt8(value & 0xFF)
    return written + 1


@export("dd_encode_var_uint_64")
def dd_encode_var_uint_64(v: UInt64, dst_addr: Int) abi("C") -> Int:
    return _encode_var_uint_64(v, dst_addr)


@export("dd_decode_var_uint_64")
def dd_decode_var_uint_64(src_addr: Int, avail: Int, value_addr: Int) abi("C") -> Int:
    return _decode_var_uint_64(src_addr, avail, value_addr)


@export("dd_encode_var_int_64")
def dd_encode_var_int_64(v_bits: UInt64, dst_addr: Int) abi("C") -> Int:
    """Zig-zag then ULEB128, as upstream's `v >> 63 ^ (v << 1)`.

    `v_bits` is the two's complement bit pattern of the signed value, and the
    decoded value comes back the same way, so the kernel never has to
    reinterpret a signed integer and the shim owns the sign convention.
    """
    var zigzag = (v_bits << 1) ^ (UInt64(0) - (v_bits >> 63))
    return _encode_var_uint_64(zigzag, dst_addr)


def _decode_var_uint_64(src_addr: Int, avail: Int, value_addr: Int) -> Int:
    """Decode one ULEB128 from a buffer with `avail` readable bytes.

    Returns the number of bytes consumed, or -1 when the buffer runs out, which
    is upstream's `EOFError`.
    """
    var src = bp(src_addr)
    var dst = u64p(value_addr)
    var x = UInt64(0)
    var shift = UInt64(0)
    for i in range(MAX_VAR_LEN_64):
        if avail <= i:
            return -1
        var n = UInt64(src[unsafe_offset=i])
        if n < 0x80 or i == MAX_VAR_LEN_64 - 1:
            dst[unsafe_offset=0] = x | (n << UInt64(shift))
            return i + 1
        x = x | ((n & 0x7F) << UInt64(shift))
        shift = shift + UInt64(7)
    return -1


@export("dd_decode_var_int_64")
def dd_decode_var_int_64(src_addr: Int, avail: Int, value_addr: Int) abi("C") -> Int:
    """Zig-zag decode of one varint; see `dd_encode_var_int_64`."""
    var dst = u64p(value_addr)
    var taken = _decode_var_uint_64(src_addr, avail, value_addr)
    if taken < 0:
        return -1
    var u = dst[unsafe_offset=0]
    dst[unsafe_offset=0] = (u >> 1) ^ (UInt64(0) - (u & 1))
    return taken


@export("dd_encode_var_int_64_many")
def dd_encode_var_int_64_many(vals_addr: Int, n: Int, dst_addr: Int,
                              offsets_addr: Int) abi("C"):
    """Encode `n` zig-zag varints back to back, recording where each one
    started. This is a checkpoint body: upstream concatenates the encoded
    values with `+`, one reallocating bytes object per value."""
    var vals = u64p(vals_addr)
    var dst = bp(dst_addr)
    var offsets = i32p(offsets_addr)
    var at = 0
    for t in range(n):
        offsets[unsafe_offset=t] = Int32(at)
        at += dd_encode_var_int_64(vals[unsafe_offset=t], dst_addr + at)
    offsets[unsafe_offset=n] = Int32(at)


@export("dd_decode_var_int_64_stream")
def dd_decode_var_int_64_stream(src_addr: Int, nbytes: Int, count: Int,
                                dst_addr: Int, taken_addr: Int) abi("C") -> Int:
    """Decode up to `count` varints from a stream, stopping at the end of the
    buffer or on a truncated tail.

    Returns -1 if a trailing partial varint was found (upstream's EOFError),
    otherwise the number of bytes consumed.
    """
    var dst = u64p(dst_addr)
    var taken = i32p(taken_addr)
    var at = 0
    for t in range(count):
        if at >= nbytes:
            return at
        var n = dd_decode_var_int_64(src_addr + at, nbytes - at, dst_addr + t * 8)
        if n < 0:
            return -1
        taken[unsafe_offset=t] = Int32(n)
        at += n
    return at


# ---------------------------------------------------------------------------
# Covered-line bitset
# ---------------------------------------------------------------------------
#
# ddtrace.internal.coverage.coverage_lines stores line numbers big-endian within
# each byte: line L lives in byte L // 8 at bit 0b1000_0000 >> (L % 8), which is
# what the CI Visibility backend expects on the wire.


@export("dd_bitset_popcount")
def dd_bitset_popcount(src_addr: Int, n: Int) abi("C") -> Int:
    """Total number of set bits, as `CoverageLines._num_lines`."""
    var src = bp(src_addr)
    var total = 0
    for i in range(n):
        var b = src[unsafe_offset=i]
        for bit in range(8):
            if (b & UInt8(0x80 >> UInt8(bit))) != UInt8(0):
                total += 1
    return total


@export("dd_bitset_expand")
def dd_bitset_expand(src_addr: Int, n: Int, dst_addr: Int, cap: Int) abi("C") -> Int:
    """The covered line numbers in ascending order, as
    `CoverageLines.to_sorted_list`. Returns the count, or -1 if `cap` is too
    small, so the caller never overruns its buffer."""
    var src = bp(src_addr)
    var dst = i32p(dst_addr)
    var written = 0
    for i in range(n):
        var base = i * 8
        var b = src[unsafe_offset=i]
        var bit = 0
        while bit < 8 and b != UInt8(0):
            if (b & UInt8(0x80)) != UInt8(0):
                if written >= cap:
                    return -1
                dst[unsafe_offset=written] = Int32(base + bit)
                written += 1
            b = b << 1
            bit += 1
    return written


@export("dd_bitset_or")
def dd_bitset_or(dst_addr: Int, src_addr: Int, n: Int) abi("C"):
    """`CoverageLines.update`: dst |= src over `n` bytes."""
    var dst = bp(dst_addr)
    var src = bp(src_addr)
    for i in range(n):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] | src[unsafe_offset=i]


@export("dd_bitset_add")
def dd_bitset_add(dst_addr: Int, n: Int, line: Int) abi("C"):
    """`CoverageLines.add` for a line already inside the current range. The
    caller grows the buffer first, exactly as upstream does."""
    var dst = bp(dst_addr)
    var byte_index = line // 8
    if byte_index < n and line >= 0:
        dst[unsafe_offset=byte_index] = (
            dst[unsafe_offset=byte_index] | UInt8(0x80 >> UInt8(line % 8))
        )


@export("dd_bitset_popcount_many")
def dd_bitset_popcount_many(ptrs_addr: Int, lengths_addr: Int, n: Int,
                            dst_addr: Int) abi("C"):
    """Popcount `n` separate bitsets, for a payload carrying many files."""
    var ptrs = i64p(ptrs_addr)
    var lengths = i32p(lengths_addr)
    var dst = i32p(dst_addr)
    for t in range(n):
        dst[unsafe_offset=t] = Int32(dd_bitset_popcount(
            Int(ptrs[unsafe_offset=t]), Int(lengths[unsafe_offset=t])
        ))


# ---------------------------------------------------------------------------
# Cosine similarity over embeddings
# ---------------------------------------------------------------------------
#
# ddtrace.llmobs._evaluators.semantic._cosine_similarity accumulates the dot
# product and both magnitudes in one pass and then divides. The numpy path it
# prefers walks the vectors four times; this does it once. The zero-magnitude
# rule is upstream's: a zero vector scores 0.0 rather than raising.


@export("dd_cosine_similarity")
def dd_cosine_similarity(a_addr: Int, b_addr: Int, n: Int, dst_addr: Int) abi("C"):
    """Fill dst with [dot, |a|^2, |b|^2, similarity]."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    var dot = Float64(0.0)
    var mag1 = Float64(0.0)
    var mag2 = Float64(0.0)
    for i in range(n):
        var x = a[unsafe_offset=i]
        var y = b[unsafe_offset=i]
        dot += x * y
        mag1 += x * x
        mag2 += y * y
    dst[unsafe_offset=0] = dot
    dst[unsafe_offset=1] = mag1
    dst[unsafe_offset=2] = mag2
    dst[unsafe_offset=3] = _ratio(dot, mag1, mag2)


@export("dd_cosine_similarity_pairs")
def dd_cosine_similarity_pairs(a_addr: Int, b_addr: Int, n: Int, pairs: Int,
                               dst_addr: Int) abi("C"):
    """Score `pairs` vector pairs laid out contiguously, one after another."""
    var a = fp(a_addr)
    var b = fp(b_addr)
    var dst = fp(dst_addr)
    for p in range(pairs):
        var base = p * n
        var dot = Float64(0.0)
        var mag1 = Float64(0.0)
        var mag2 = Float64(0.0)
        for i in range(n):
            var x = a[unsafe_offset=base + i]
            var y = b[unsafe_offset=base + i]
            dot += x * y
            mag1 += x * x
            mag2 += y * y
        var at = p * 4
        dst[unsafe_offset=at] = dot
        dst[unsafe_offset=at + 1] = mag1
        dst[unsafe_offset=at + 2] = mag2
        dst[unsafe_offset=at + 3] = _ratio(dot, mag1, mag2)


def _ratio(dot: Float64, mag1: Float64, mag2: Float64) -> Float64:
    var m1 = sqrt(mag1)
    var m2 = sqrt(mag2)
    if m1 == 0.0 or m2 == 0.0:
        return Float64(0.0)
    return dot / (m1 * m2)
