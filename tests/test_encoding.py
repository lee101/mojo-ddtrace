"""Parity tests for the varint codec against
`ddtrace.internal.datastreams.encoding`.

Varints are exact integer bytes, so the comparison is `rtol=0, atol=0`: the
kernel must emit the same bytes the upstream Python emits, because the
pathway headers ddtrace hashes and compares are byte for byte. The tests pin
down upstream's two quirks too, the nine-byte cap and the ninth byte being
taken as final whatever its continuation bit says.
"""

import numpy as np
import pytest
from ddtrace.internal.datastreams import encoding as ref

import mojo_ddtrace as m

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT64_MASK = (1 << 64) - 1

SIGNED_VALUES = [
    0, 1, -1, 2, -2, 63, 64, 127, 128, 129, 255, 256, 300, 16383, 16384,
    2 ** 31 - 1, -(2 ** 31), 2 ** 32, 123456789, -123456789,
    INT64_MAX, INT64_MIN, INT64_MAX - 1, INT64_MIN + 1,
]
UNSIGNED_VALUES = [0, 1, 127, 128, 300, 65535, 2 ** 32, 2 ** 53, UINT64_MASK,
                  UINT64_MASK - 1, 1 << 63]


@pytest.mark.parametrize("value", SIGNED_VALUES)
def test_encode_var_int_64_exact(value):
    assert m.encode_var_int_64(value) == ref.encode_var_int_64(value)


@pytest.mark.parametrize("value", UNSIGNED_VALUES)
def test_encode_var_uint_64_exact(value):
    assert m.encode_var_uint_64(value) == ref.encode_var_uint_64(value)


@pytest.mark.parametrize("value", SIGNED_VALUES)
def test_roundtrip_var_int_64(value):
    got, rest = m.decode_var_int_64(m.encode_var_int_64(value))
    assert got == value
    # The 10-byte encodings (INT64_MIN and its neighbours) come back with a
    # one-byte remainder: upstream's decoder stops after nine bytes, and the
    # port stops in the same place.
    assert rest == b"" if len(m.encode_var_int_64(value)) <= 9 else b"\x01"


@pytest.mark.parametrize("value", UNSIGNED_VALUES)
def test_roundtrip_var_uint_64(value):
    got, rest = m.decode_var_uint_64(m.encode_var_uint_64(value))
    # Same nine-byte cap as the signed codec, same one-byte remainder for the
    # encodings that need a tenth byte.
    assert got == value
    assert rest == ref.decode_var_uint_64(ref.encode_var_uint_64(value))[1]


def test_decode_matches_upstream_including_the_remainder():
    """For the 10-byte encodings upstream's decoder stops after nine bytes and
    hands the last one back as a remainder; the port must do exactly that, or
    a checkpoint that ddtrace produced would decode differently here."""
    for value in (INT64_MAX, INT64_MIN, -1):
        theirs = ref.encode_var_int_64(value)
        mine = m.encode_var_int_64(value)
        assert mine == theirs
        assert m.decode_var_int_64(mine) == ref.decode_var_int_64(theirs)


def test_ninth_byte_is_taken_as_final():
    """A nine-byte run with the continuation bit set on the last byte is
    accepted, exactly as upstream does."""
    payload = bytes([0x80] * 8 + [0x81])
    assert m.decode_var_uint_64(payload) == ref.decode_var_uint_64(payload)
    # The continuation bit on the ninth byte is ignored, so the value is
    # 0x81 shifted up, not a clean 1 << 63.
    assert m.decode_var_uint_64(payload)[0] == 0x81 << 56


@pytest.mark.parametrize("truncated", [b"", b"\x80", b"\x80\x80", b"\xff" * 3,
                                        b"\x80\x80\x80\x80\x80\x80\x80\x80"])
def test_truncated_input_raises_eoferror(truncated):
    for module in (m, ref):
        with pytest.raises(EOFError):
            module.decode_var_int_64(truncated)


def _ref_stream(payload, count):
    """What `DataStreamsCtx.decode_pathway` does: call the upstream scalar
    decoder in a loop until the payload runs out or `count` values are read."""
    values = []
    at = 0
    for _ in range(count):
        if at >= len(payload):
            break
        value, rest = ref.decode_var_int_64(payload[at:])
        values.append(value)
        at = len(payload) - len(rest)
    return values, at


@pytest.mark.parametrize("values", [
    [1700000000123, 1700000000456],
    [-1, 0, 1, 2 ** 40, -(2 ** 40), 0],
    [],
    [INT64_MAX, INT64_MIN, -1],
    [0] * 64,
])
def test_stream_decode_matches_upstream_loop(values):
    payload = b"".join(ref.encode_var_int_64(v) for v in values)
    for count in (len(values), len(values) + 3, 1):
        expect, expect_at = _ref_stream(payload, count)
        got, consumed = m.decode_var_int_64_stream(payload, count)
        assert [int(v) for v in got[:len(expect)]] == expect
        assert consumed == expect_at


def test_stream_decode_partial_trailing_varint_raises():
    # 300 needs two bytes, so dropping the last one truncates a real varint
    # rather than just ending the stream.
    payload = m.encode_var_int_64_many([1, 300, 2])
    assert len(payload) == 4
    with pytest.raises(EOFError):
        m.decode_var_int_64_stream(payload[:-2], 3)


def test_stream_decode_empty_buffer():
    got, consumed = m.decode_var_int_64_stream(b"", 4)
    assert got.size == 0
    assert consumed == 0


def test_encode_many_matches_concatenation():
    """The batch encoder has to produce exactly what joining the scalar results
    produces: a checkpoint body is compared byte for byte downstream."""
    rng = np.random.default_rng(17)
    for size in (0, 1, 2, 17, 200):
        values = rng.integers(INT64_MIN, INT64_MAX, size, dtype=np.int64)
        got = m.encode_var_int_64_many(values)
        assert got == b"".join(ref.encode_var_int_64(int(v)) for v in values)


def test_stream_roundtrip_many():
    # A well-formed stream: every encoding is nine bytes or fewer, which is the
    # case a real checkpoint is in. The 10-byte case is covered above, where
    # the port desynchronises exactly as the upstream scalar loop does.
    values = [-1, 0, 1, 2 ** 40, -(2 ** 40), 12345, 2 ** 55]
    payload = m.encode_var_int_64_many(values)
    got, consumed = m.decode_var_int_64_stream(payload, len(values))
    assert [int(v) for v in got] == values
    assert consumed == len(payload)


def test_out_of_range_values_are_refused():
    """Upstream's Python arithmetic is unbounded and will encode 2**63 into a
    varint its own decoder reads back as something else. The kernel is
    Int64-bounded like the wire format, so it raises rather than truncating."""
    for value in (INT64_MAX + 1, INT64_MIN - 1, 2 ** 64, 2 ** 100):
        with pytest.raises(ValueError):
            m.encode_var_int_64(value)


def test_max_var_len_matches_upstream():
    assert m.MAX_VAR_LEN_64 == ref.MAX_VAR_LEN_64 == 9
    assert len(ref.encode_var_int_64(INT64_MIN)) == ref.MAX_VAR_LEN_64 + 1
    assert len(m.encode_var_int_64(INT64_MIN)) == ref.MAX_VAR_LEN_64 + 1
