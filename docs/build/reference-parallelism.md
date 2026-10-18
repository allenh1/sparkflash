# Reference parallelism: what it parallelises, and why it changes no value

Status: implementation on `phase2/reference-parallel` (based on main 13d880d), measured on the dev
host against a released-shape synthetic slice; the node golden fixture has not yet been re-run.
This note is the provenance record for the change and, deliberately, its scope limit.

## Why

The first node golden-fixture attempt (2026-10-18, merged main 13d880d, binary 4b774051…, the
`sequence_fixture` real-checkpoint path) was aborted after 245 s by the owner's window rule. At the
abort the process had read 14,723,496,820 B (disk read_bytes 14,851,194,880 B), had used 219.89 s of
CPU at ≈0.92 of one core, peak RSS 3,975,484 KiB - i.e. the reference pass was **compute-bound and
single-threaded**, at about 4.8 % of one non-Engram weight sweep. The per-layer cost is dominated by
decoding and scale-applying the routed experts' FP4 weights (each routed expert body is three
11,796,480-element tensors; a 130-token prefill routes roughly most of a layer's 384 experts, so a
layer decodes and scales ≈11.8 G elements), then by the FP8 attention weights' decode and the dense
and expert matmuls. The generation is deterministic and per-tensor; nothing in it needs a specific
thread order.

## What is parallelised (and what is not)

Splitting is always over **independent output work**:

| routine | split over | why independent |
|:--|:--|:--|
| `quant::decode*` | element/index chunks | one element is a pure function of its index |
| `linalg::apply_scales` | row chunks | each element is multiplied by its own grid entry |
| `linalg::rms_norm` | row chunks | each row's mean and scaling are per-row |
| `linalg::matmul` | output rows/batches | each output element is a fixed-order `sum_col x*w` |
| `moe_forward` routed path | contiguous expert groups | each choice's expert output is independent; the accumulator is combined serially in the same ascending-expert order |
| `moe_forward` shared expert | token rows | each token's row is independent |
| `sparse_attention` (unchanged) | - | per-query softmax/dot reductions stay serial |

The reduction order of every per-output sum is untouched: each output element's `acc += x[col]*w[col]`
loop is the same loop with the same `col` order, and the MoE routed accumulation is still one `+=`
per choice in ascending expert order over the same per-choice vectors. No helper reassociates a sum,
changes a clamp, or introduces a fused operation. The parallel pass is therefore **bit-identical** to
the serial pass by construction, not by comparison.

## Controls and proof

* `parallel::with_reference_threads(n, body)` pins the thread count for tests (1 = serial).
* `SPARKFLASH_REFERENCE_THREADS` sets the process default; otherwise available parallelism.
* Nested calls (inside a worker) stay serial, so no oversubscription.

Tests added by the change:

* the committed CI fixture (`tests/fixtures/sequence-130-tiny.json`, 29,508 B, sha256
  `b9e209239be17e4edcd594156131b56d149af247c53f9c827b8e14cbecf2a094`) regenerates **byte-identically**
  under 1, 2, 4 and 8 reference threads, and every count equals the committed bytes. This is the
  end-to-end byte check; at the tiny configuration's sizes it reaches the MoE routed-expert and
  shared-expert threaded arms, not every per-routine arm;
* the per-routine threaded arms are covered bit-for-bit by unit tests: `quant` decoding for every
  format the decoder accepts (BF16, F32, F8E4M3, F8E8M0, F4E2M1 - F16 is deliberately unsupported
  and a test asserts it stays refused), `matmul` (output-row split), `apply_scales` for both
  `Block32` and `PerRow32` grids, and `rms_norm`.

The committed CI fixture bytes are unchanged by this change: the byte-identical regeneration test
still passes and the sha256 above is still the file's.

## Measured speedup (released-shape slice, dev host)

The measurement uses `ModelConfig::v41_flash()` restricted to the first 1-2 backbone layers (no MTP
stages, no Engram tables, no KV-source layers; those are device-window work), with a synthetic
weight source that returns exactly the plan's dtypes and shapes so every read, decode, scale pass
and matmul has the released cost. Both timed runs follow a warm run of the same source, so every
tensor read in the timed runs hits the source's cache: the numbers time the reference pass, not the
generator. Thread count 20 (the node has 20 cores); 130-token prefill and one decode step.

Measurement binary: `examples/reference_parallel_bench` (committed), 5,444,320 B, sha256
`c94ad6aa49439b2a0fdc9403ea3fde3f6db03e9e055a43e39249a00809c40886`, built from this MR's tree on
base 51cd0c52bdcea88aa0a41481800a022361efda3c. Dev host: 88 cores, load average 15-21 during the
runs (shared runner host), so the threaded arm is a **lower bound**; the node is dedicated.

| configuration | serial (1 thread) | threaded (20 threads) | speedup |
|:--|--:|--:|--:|
| 1 layer, prefill | 146.78 s | 22.38 s | 6.56× |
| 2 layers, prefill | 271.15 s | 41.37 s | 6.55× |
| 1 layer, 1 decode | 15.26 s | 5.07 s | 3.01× |
| 2 layers, 1 decode | 16.26 s | 5.62 s | 2.90× |

Differencing the two layer counts: one layer's prefill is 124.37 s serial and 18.99 s threaded
(6.55×), while the fixed prefill part (embedding, head, state) is 22.41 s serial and 3.39 s
threaded. The one-vs-two-layer decode difference is within run-to-run noise (≈0.5-1 s per layer):
the fixed decode part dominates - 15.26 s serial and 5.07 s threaded for one layer's step - because
each step still decodes the 661,913,600-element embedding and head tables on the host. The threaded
MoE pass is limited to the number of distinct experts in the chunk: a decode step reads six experts
per layer, so it gets six-way expert parallelism, while a prefill chunk routes most of a layer's 384
experts; that is why the prefill ratio (6.55×) is much better than the decode ratio (2.9-3.0×).

**Projected golden wall time.** The node's aborted serial run extrapolates to 85.3 min of prefill
(245 s for 14,723,496,820 B read of ≈307.6 GB). At the measured 6.55× prefill ratio the projected
node prefill is 13.0 min of compute; the ≈300 GB sweep overlaps that compute, and the node's
6.1 GB/s sequential NVMe needs ≈50 s if it does not, so the projected golden-fixture prefill is
**≈13 min (range 13-15)** on the 20-core node. That meets the "under ~15 min" bar only if the node
scales at least as well as this loaded dev host did; it is a projection, and the node window must
confirm it, so the window should be 45 min rather than tight. For the decode steps, the full model
extrapolated from the slice is ≈54 s serial per step (40 layers × ≈1.0 s + 14.26 s fixed) before the
MTP stages and Engram paging the slice does not contain; at the measured 2.9× ratio one step is
≈19 s, so the 8 decode steps project to ≈3-6 min (allowing for the unmeasured MTP/Engram share) and
the whole golden run to **≈20 min (range 18-25 min)** versus ≈2 h serial. A 45 min window is
recommended.

## Honest limits

* This changes **no reference value** and therefore no golden-fixture number, but it is still a real
  engine change: the `WeightSource` trait now requires `Send + Sync` so the MoE pass can read
  independent experts concurrently (sources are immutable from the caller's side; the safetensors
  source already used a mutex only for its header cache).
* The measured ratio is from the dev host with a synthetic source; the node window must confirm the
  golden wall time. No device number is claimed or changed by this work.
* Engram paging, the HSI/indexer layers and the MTP stages are not in the sliced measurement; the
  `moe_forward` and matmul paths they use are the same parallel paths, but their share of the node
  wall time still has to be measured on the node.
* The sparse-attention dot/softmax reductions are deliberately left serial, so prefill rows with
  long windows keep a serial cost; that is the honest trade-off for not touching a reduction order.

