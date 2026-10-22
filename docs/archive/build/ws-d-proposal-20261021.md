# WS-D proposal: the M=1 dense tile waste (decision paper, no code)

Status: proposal only for the user's WS-D un-park decision. No kernel, tiling, cubin, manifest or
allow-list change has been made; nothing here is a measurement of a fixed kernel. The numbers are the
current probe row plus a stated basis for what a re-tile would recover, not a promise.

Evidence: lean row provenance `4e4cea50` (599.94 ms, tokens + logits sha identical 4/4) and the
post-WS-A2 seam-basis probe row (14.935 ms/layer drained). 40 layers, one token per step, TP4 per-rank
geometry from `crates/sparkflash-engine/config/deepseek_v41_flash.runtime.json`.

## The M=1 dense launches at decode (per token, per rank)

| launch | block | kernel (pinned) | grid at M=1 | blocks | MAC/layer | kernel ms/layer | GMAC/s | thread slots doing work |
|---|---|---|---|---|---|---|---|---|
| wq_a [1280x5120] + wkv [512x5120] | attention_projections | b12x_mxfp8_dense_sm121 (x2) | (1, cols/64) | 20 + 8 = 28 | 9,175,040 | 4.771 | 1.92 | 16/256 per block (6.25%) |
| wq_b [8192x1280] per rank | query_path | b12x_mxfp8_dense_sm121 | (1, 128) | 128 | 10,485,760 | 0.620 | 16.9 | 16/256 per block (6.25%) |
| wo_a [2048x4096] per rank | attention_output | b12x_bf16a_mxfp8w_wo_a_sm121 | (2, 4) | 8 | 8,388,608 | pair 1.33 with wo_b | n/a | 256/256 (full) |
| wo_b [5120x256] per rank | attention_output | b12x_mxfp8_dense_sm121 | (1, 80) | 80 | 1,310,720 | pair 1.33 with wo_a (not separable in this row) | - | 16/256 per block |
| routed gate/up 6 slots x 640 x 5120 | moe_gate_up | routed_gate_up_sm121 | (6, 3) | 18 | 19,660,800 (17.7M real) | 0.825 | 23.8 | 256/256 (full) |

The tile geometry is the whole story. `b12x_mxfp8_dense_sm121` is a 64-row x 64-column SIMT tile,
256 threads, each thread owning a 4-row x 4-column sub-tile; the row base is
`blockIdx.x*64 + (lane/16)*4`. At M=1 only lane group 0 (lanes 0-15) has a row inside the tile, so
16 of 256 thread slots do any arithmetic and one warp of eight has any active lane. The block count is
then `ceil(cols/64)` and the arithmetic per output is a continuous f32 sum over K in 32-value blocks
ascending and ascending inside each block, with one bf16 round at the end.

The same kernel on the same M=1 shape already runs 8.8x faster when the grid is larger: wq_b at 8192
columns is 128 blocks and reaches 16.9 GMAC/s, while wq_a+wkv at 28 blocks reaches 1.92 GMAC/s. The
extra 100 blocks buy latency hiding across the device's SMs; no arithmetic differs.

## Expected recovery (basis, not a promise)

| change | kernel ms/layer now | expected kernel ms/layer | recovery ms/step | basis |
|---|---|---|---|---|
| re-tile the dense kernel for M=1 (64x64 -> columns-only: 1 row x 256 columns, one output per thread, K contract untouched) used by wq_a + wkv | 4.771 | 0.75-1.50 | 130-160 | the same kernel family measures 16.9 GMAC/s at M=1 on a 128-block grid (query_path). A columns-only mapping removes the row sub-tile and puts one output per thread: the projections would go from 448 working threads (28 half-warps) to 1792 (56 warps), i.e. ~4x the memory-level parallelism, which is the measured gap's main cause. The anchor differs in inner too (1280 vs 5120), so the projected rate is taken at 6-12 GMAC/s, not 16.9 |
| same variant for wo_b | inside the 1.33 pair | 0.10-0.25 | 10-20 | 80 blocks, 16/256 slots, 1.31M MAC; re-tile removes the idle row slots; the wo_a/wo_b split needs one probe row to price exactly |
| same variant for wq_b | 0.620 | 0.30-0.40 | 7-11 | it is the best-utilized dense launch today; the same columns-only mapping still merges its 4-column-per-thread sub-tile into one output per thread, so the headroom is smaller |
| routed gate/up | 0.825 | 0.825 | 0 | not an M=1 tile-waste case: no M tile, 256/256 threads, 18 blocks at 23.8 GMAC/s. Its 33 kernel ms/step is the pinned FP4 shape (640 padded columns vs 576 real), not tile waste |

Aggregate expectation on the lean row: **-145 to -190 ms/step** if the re-tile lands in the upper half
of the basis (599.94 -> ~410-455 ms/step), **-110 to -130** at the conservative end (599.94 ->
~470-490). The biggest single item is attention_projections: 199 kernel ms/step, 9.17M MACs per layer
run at 1.92 GMAC/s because only 16 of 256 thread slots per block do arithmetic and there are 28 blocks
in total. This is the largest lever left that the arithmetic allows without changing a value.

## Bit-identity risk

The change is admissible because of what the kernel's arithmetic contract already says: the tiling
decides which thread owns which output and never reorders a sum, so moving tile boundaries changes
scheduling only. Concretely:

- Allowed: thread re-mapping (who owns which output column), block decomposition, N-tile width,
  register/shared staging, and launch count. Every output keeps its continuous ascending-K f32
  running sum and its single bf16 round, so the bytes are expected to be identical by construction.
- Excluded: split-K. The dense kernel's contract is a running f32 sum, so a split reduction changes
  the rounding sequence. (routed_gate_up_sm121's 8-way K split is legitimate only because its
  reference arithmetic is block-partial-based and its reducer re-adds the 160 block partials in
  ascending order; copying that pattern here would change values, not just the schedule.)
- Excluded: tensor-core MXFP8 MMA (different accumulation order/dtype), pre-decoded f32 weight
  buffers (exact but 4x storage and a new B load path; unnecessary for this win).
- Acceptance is the gate we already run: tokens + logits sha identical 4/4 plus an identical layer
  chain digest. A tiling mistake shows up as a gate failure, not as silent drift, because the gate is
  bit-exact. The pinned CPU twin (mxfp8_ref::dense_gemm) is the pre-gate oracle for the variant.

## Device-hours one measurement row needs

| step | node time |
|---|---|
| kernel-vs-pinned-oracle bit-compare on released layer bytes (existing example-harness pattern) | ~0.5 h |
| freeze + build + lean row + chain row + identity gate + one speed-log line | ~1 h |
| one tiling iteration if the first variant misses | +1.5-2 h |

So **1.5-2 device-hours** for the first decision-grade row, 3-4 if it needs one iteration. Host-side
implementation of the variant (new cubin, baseline regen, D49 manifest entry, ABI-split allow-list)
is about a day, no node time, and is exactly the set of files the WS-A freeze forbids - which is why
this needs the WS-D un-park and not a WS-A slot.

## What WS-D would own if un-parked

1. One M=1 dense kernel variant (`b12x_mxfp8_dense_sm121` re-tiled to 1x16 with one column per
   thread) and its cubin baseline entry; the routed gate/up kernel and wo_a stay as they are.
2. The launch-side tile choice, so prefill rows>1 keep the 64x64 tile and decode rows=1 take the new
   one; no arithmetic branch, only the load of the second cubin.
3. New rows: kernel-vs-oracle, lean row, evidence-ON row, identity gate, speed-log line.

Not in scope of this proposal: any attention/mHC/MoE arithmetic change, any dtype change, any change
to the pinned shapes that the ABI and the manifests record.
