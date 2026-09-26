"""ULEB128 / zig-zag varint codec, mirroring `ddtrace.internal.datastreams.encoding`.

Upstream builds the encoded bytes with `b += struct.pack(...)` inside the
loop, which reallocates once per byte. The kernel writes into one buffer and
returns the length. The decoder keeps upstream's two quirks exactly, because
the pathway headers ddtrace hashes and compares are byte for byte:

* a varint is at most `MAX_VAR_LEN_64` payload bytes plus a final byte, and
* the ninth payload byte is taken as final whatever its continuation bit says.

Running out of buffer raises `EOFError`, as upstream does.
"""

from ._lib import (
    MAX_VAR_LEN_64,
    decode_var_int_64,
    decode_var_int_64_stream,
    decode_var_uint_64,
    encode_var_int_64,
    encode_var_int_64_many,
    encode_var_uint_64,
)

__all__ = [
    "MAX_VAR_LEN_64",
    "decode_var_int_64",
    "decode_var_int_64_stream",
    "decode_var_uint_64",
    "encode_var_int_64",
    "encode_var_int_64_many",
    "encode_var_uint_64",
]
