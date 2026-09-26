"""mojo-ddtrace: the numeric inner loops of ddtrace, in Mojo.

Installable alongside the real `ddtrace` package, which it is tested against
for parity. ddtrace is overwhelmingly IO, configuration and string plumbing;
what is here is the handful of places where it actually loops over bytes or
floats in Python:

* FNV-1 64 hashing of data-streams checkpoints and schema definitions,
* the ULEB128 / zig-zag varint codec those checkpoints are framed with,
* the covered-line bitset behind CI Visibility,
* cosine similarity over LLMObs embeddings.
"""

from .coverage_lines import CoverageLines
from .encoding import (
    MAX_VAR_LEN_64,
    decode_var_int_64,
    decode_var_int_64_stream,
    decode_var_uint_64,
    encode_var_int_64,
    encode_var_int_64_many,
    encode_var_uint_64,
)
from .fnv import FNV1_64_INIT, FNV_64_PRIME, fnv, fnv1_64, fnv1_64_many
from .semantic import cosine_similarity, cosine_similarity_many

__all__ = [
    "CoverageLines",
    "FNV1_64_INIT",
    "FNV_64_PRIME",
    "MAX_VAR_LEN_64",
    "cosine_similarity",
    "cosine_similarity_many",
    "decode_var_int_64",
    "decode_var_int_64_stream",
    "decode_var_uint_64",
    "encode_var_int_64",
    "encode_var_int_64_many",
    "encode_var_uint_64",
    "fnv",
    "fnv1_64",
    "fnv1_64_many",
]
__version__ = "0.1.0"
