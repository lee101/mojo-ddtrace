# mojo-ddtrace

`mojo-ddtrace` is the compute-oriented subset of
[ddtrace](https://github.com/DataDog/dd-trace-py): the places where the tracer
actually loops over bytes or floats in Python, moved into Mojo. It installs
alongside the real `ddtrace` package, and the tests compare it against that
package directly.

```python
import mojo_ddtrace as mdd

mdd.fnv1_64(b"service=web env=prod")     # the Data Streams pathway hash
mdd.encode_var_int_64(1700000000123)     # b'\xd0\x89\x89\x18'
mdd.CoverageLines.from_list([1, 9, 80]).to_sorted_list()   # [1, 9, 80]
mdd.cosine_similarity(query, candidate)  # one pass, three accumulators
```

## Why this package, and why it is small

ddtrace 4.15.2 is an APM tracer: almost all of it is IO, configuration,
string formatting and dict bookkeeping. Everything numerically heavy is
already native -- Data Streams statistics are Rust DDSketch
(`ddtrace.internal.native`), the msgpack/JSON encoders and tagset codec are
the `_encoding.so` / `_tagset.so` extensions, process metrics come from
`native.process_metrics()`, and the entire profiler lives in
`ddtrace.profiling._profiler`. `ddtrace/sourcecode/` is a docstring. The
Python half of `ddtrace/profiling/` is import and thread wiring.

What is left, and what this port covers, is four small kernels that upstream
writes as Python loops:

| upstream surface | what it loops over | why it is worth compiling |
| --- | --- | --- |
| `ddtrace.internal.utils.fnv` | every byte of a buffer: a big-int multiply, a modulo and an xor | runs on every Data Streams checkpoint and every schema build; the modulo is just `UInt64` wrap-around |
| `ddtrace.internal.datastreams.encoding` | up to nine varint bytes, with `b += struct.pack(...)` and a buffer realloc per byte | frames every pathway header ddtrace sends and parses |
| `ddtrace.internal.coverage.coverage_lines` | every bit of a covered-line bitset | one popcount and one bit expansion per file in a CI Visibility payload |
| `ddtrace.llmobs._evaluators.semantic` | every element of two embedding vectors, three accumulators | 384 to 3072 dimensions per comparison, once per candidate an evaluator scores |

The same varint kernel is duplicated three more times upstream
(`errortracking/_handled_exceptions/bytecode_injector.py`,
`internal/bytecode_injection/core.py`, and the two
`internal/coverage/instrumentation_py3_*` modules). Those call sites are not
ported; the codec is.

## Covered subset

| area | implemented API | kernel |
| --- | --- | --- |
| FNV | `fnv`, `fnv1_64`, `fnv1_64_many` | `dd_fnv`, `dd_fnv_many` |
| Varints | `encode_var_int_64`, `decode_var_int_64`, `encode_var_uint_64`, `decode_var_uint_64`, `encode_var_int_64_many`, `decode_var_int_64_stream`, `MAX_VAR_LEN_64` | `dd_encode_var_int_64`, `dd_decode_var_int_64`, `dd_encode_var_uint_64`, `dd_decode_var_uint_64`, `dd_encode_var_int_64_many`, `dd_decode_var_int_64_stream` |
| Coverage lines | `CoverageLines` with `add`, `update`, `to_sorted_list`, `to_bytes`, `from_list`, `from_bytearray`, `__len__`, `__bool__`, `__eq__`, `__copy__`, `__repr__`; `bitset_popcount_many` | `dd_bitset_add`, `dd_bitset_or`, `dd_bitset_popcount`, `dd_bitset_expand`, `dd_bitset_popcount_many` |
| Semantic similarity | `cosine_similarity`, `cosine_similarity_many` | `dd_cosine_similarity`, `dd_cosine_similarity_pairs` |

Sixteen exported kernels, all in one `src/kernels.mojo`.

### Not implemented, and why

- **The tracer, the writer, the processors, the agent client, config, sampling
  rules, telemetry, and every `contrib/` integration.** These are IO, state
  machines and string plumbing. There is no numeric core to move.
- **The Rust and C extensions** (`internal/native` DDSketch, `_encoding`,
  `_tagset`, `profiling/_profiler`, `libdatadog`). The compute is already
  compiled; a second implementation would be a downgrade, not a port.
- **Varint call sites in the bytecode injectors and errortracking
  instrumentation.** Same kernel, but the surrounding linetable rewriting is
  intricate interpreter surgery whose value does not justify the risk here.
- **The W3C tracestate and baggage byte-budget loops** in
  `internal/utils/http.py` and `propagation/http.py`. They loop over at most
  32 members, and the expensive part (`str.encode`) is already C.
- **`RateLimiter`** in `internal/rate_limiter.py`. Real arithmetic, but it is
  a scalar token-bucket state machine with no loop over data -- a kernel would
  be a fake one.
- **The Knuth priority-sampling hash** in `_trace/sampling_rule.py`. It is one
  `UInt64` multiply per span with no array to amortise the call over, so
  crossing the FFI would cost more than the arithmetic.

### Deliberate divergences

- `encode_var_int_64` raises `ValueError` outside the `Int64` range. Upstream's
  Python arithmetic is unbounded and will encode `2**63` into a varint that its
  own decoder reads back as something else; the kernel is `Int64`-bounded like
  the wire format, so it refuses rather than truncating silently.
- `fnv` accepts `fnv_size` for API parity and rejects anything but `2**64`
  rather than pretending to honour a smaller modulus.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-ddtrace.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task. Against the shared toolchain:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Tests

161 tests, all checked against the real `ddtrace` package:

| file | what it pins down |
| --- | --- |
| `tests/test_fnv.py` | exact digests against `ddtrace.internal.utils.fnv` for 16 buffer sizes, the empty buffer, fixed pinned digests, exhaustive single- and two-byte vectors, realistic service/env blobs, the batch entry point against the scalar one, and the `fnv_size` guard |
| `tests/test_encoding.py` | exact bytes against `ddtrace.internal.datastreams.encoding` for 24 signed and 11 unsigned values, round trips, the nine-byte cap and the ninth byte taken as final, `EOFError` on eight truncated inputs, the batch encoder against concatenation, and the stream decoder against a loop of the upstream scalar decoder including the case where the two disagree about a tenth byte |
| `tests/test_coverage_lines.py` | line lists, byte images, lengths and `repr` against `ddtrace.internal.coverage.coverage_lines`, exhaustively over all 256 byte patterns, the bit order, growth, merge, duplicate adds, equality, copy independence, and batch popcount |
| `tests/test_semantic.py` | scores against `_cosine_similarity` for eight vector lengths up to 3072, against the pure-Python branch it replaces, plus the exact identities (self is 1, negation is -1, orthogonal is 0, a zero vector is 0), ill-conditioned magnitudes, batch versus scalar, and non-contiguous input |

FNV and the varint codec are exact integer arithmetic and are asserted with
`rtol=0, atol=0`; only the cosine similarity carries a tolerance, `rtol=1e-13`
against a measured worst case of about 1e-15.

## Performance

Best-of-five wall clock, alternating the two candidates, same process, on the
shared 36-core box. Every case verifies numerical agreement with the reference
before timing. The reference for each case is the real `ddtrace` function, with
containers built before the clock starts where the upstream API would have
built them per call.

| case | reference | mojo-ddtrace | result |
| --- | ---: | ---: | ---: |
| fnv1_64, 65536 bytes | 35.92 ms | 0.22 ms | 164.96x |
| fnv1_64, 512 buffers of 256 bytes | 70.19 ms | 2.59 ms | 27.06x |
| varint, 20000 values, encode and decode | 257.62 ms | 1.06 ms | 242.42x |
| coverage popcount, 200 files | 11.48 ms | 3.48 ms | 3.30x |
| coverage expand, 200 files x 400 lines | 26.16 ms | 26.83 ms | 0.98x |
| cosine, 256 pairs x 1536 dims | 6.12 ms | 1.15 ms | 5.33x |

The one case at parity is the bit expansion, and the reason is worth stating:
upstream's `to_sorted_list` spends its time appending 80 000 Python ints to a
list, and this port's `to_sorted_list` spends exactly the same time building
the same Python list. The loop over bits is no longer the cost. A port that
returned a NumPy array instead of a list would win that case, and would not be
a drop-in for the CI Visibility payload builder, which needs a list.

The large multipliers are not the kernels being clever; they are Python
per-byte and per-call overhead disappearing. FNV-1 is one multiply and one xor
per byte, and the upstream loop pays big-integer arithmetic for each of them.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-ddtrace.so`.

The `python/mojo_ddtrace` layer owns every array. Buffers cross the C ABI as
64-bit addresses and are reconstructed in Mojo as
`Pointer[UInt8, AnyOrigin[mut=True]]` and friends, which keeps the exported
symbols non-parametric.

Two ABI decisions are worth knowing about:

- Signed varint values cross as their two's complement bit pattern, in and
  out, so the kernel never reinterprets a signed integer and the shim owns the
  sign convention. `encode_var_int_64_many` therefore takes a `uint64` view of
  an `int64` array.
- `CoverageLines` stores a `bytearray`, and the shim gets a zero-copy writable
  `uint8` view of it with `np.frombuffer`, so the in-place OR and set-bit
  kernels write through to the container rather than into a copy.

Upstream's varint decoder stops after nine bytes and hands the tenth back as a
remainder, and takes the ninth byte as final whatever its continuation bit
says. Both quirks are reproduced, because ddtrace's pathway headers are hashed
and compared byte for byte: a "correct" ten-byte decoder here would decode
ddtrace's own checkpoints differently. The tests pin both.

## Numerics

FNV and the varint codec are exact, and the tests assert equality. Cosine
similarity is not: Mojo emits FMA, so `dot += x * y` is fused where the Python
loop is not. Measured spread against the pure-Python branch is about 1e-15
relative, against upstream's NumPy path about 5e-16, and the identities
(self-similarity 1, negated -1, orthogonal 0) hold exactly. The tests use
`rtol=1e-13`, a factor of a hundred over the observed error.

Note that `fnv1_64` is not textbook FNV-1: upstream multiplies and *then*
xors, which is the FNV-0 ordering, so `fnv1_64(b"a")` is `0xaf63bd4c8601b7be`
rather than the published `0xaf63dc4c8601ec8c`. The port reproduces what
ddtrace does, since that is what its checkpoints were built with.

## License

MIT
