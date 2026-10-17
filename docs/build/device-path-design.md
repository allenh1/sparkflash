# The device path: design before code

This is task-31's first deliverable. It fixes the device-side layouts, the upload path, the device KV
layout and the per-layer dispatch *before* any code exists, because two facts measured against the
released checkpoint decide most of the shape:

* **`REFERENCE_TENSOR_LIMIT` is 32 Mi elements** (`sparkflash-engine/src/linalg.rs:76`). `embed.weight`
  and `head.weight` are 661,913,600 elements each, so a whole-tensor decode of either is refused by
  design - the reader's own comment says a real checkpoint is "refused rather than silently paged
  in". Every large tensor must reach the device through a **row or block reader**, never through a
  materialised decode.
* **Every Engram table is 98.3 G elements.** At 32 Mi elements per decode that is 2,900 excess
  decodes, i.e. it can never be materialised at all. Engram rows must be fetched through the M2
  store (`sparkflash-engram`: `cache.get(layer, row, out)`, `layout.rs::row_offset`,
  `prefetch.rs`'s io_uring worker) and uploaded a row at a time.

Neither fact is a limitation to work around quietly; they are the reason the device path is a
streaming design rather than an "upload the model" design.

## 1. How the CUDA code gets built and executed

Compilation happens on the dev box inside the pinned image; only execution happens on the node.

* amd64 `nvcc` from `nvidia/cuda@sha256:7d2f6a8c2071...` compiles the `.cu` files to cubin/PTX for
  `sm_121a`. Device code is host-arch independent, so this is the same sm_121a binary an aarch64
  `nvcc` would emit.
* The host side is cross-compiled for `aarch64-unknown-linux-gnu` by the zig toolchain pinned in
  [scripts/build-aarch64.sh](../scripts/build-aarch64.sh).
* **Nothing links a CUDA library.** cudarc stays on `fallback-dynamic-loading`, so `libcuda.so.1` is
  dlopen'd at run time on the node. This is a build invariant, not an observation: enabling
  `dynamic-linking` would put an x86_64-shaped link line under an aarch64 target. See
  [aarch64-pipeline.md](aarch64-pipeline.md) section 5 for the evidence chain.

## 2. The stored forms, exactly

Two different scale grids are in play, and confusing them is the classic way to get silently wrong
numbers, so both are stated with their source:

| form | shape | scales | source |
|---|---|---|---|
| expert (FP4) weight | `[out, in/2]` packed E2M1, in K-major pairs | `[out, ceil(in/32)]` ue8m0 - **one row per output row**, no blocking along `out` | `dtype.rs::expert_scale_shape`, model.py:219-224 |
| dense (FP8) weight | `[out, in]` E4M3 | `[ceil(out/32), ceil(in/32)]` ue8m0 - the true 2-D grid | `dtype.rs::dense_scale_shape`, kernel.py:293-298 |
| MoE activation | `[tokens*6, 640]` E4M3, one byte per value | `[tokens*6, 20]` ue8m0 | `mxfp4_ref.rs` (`MOE_ACT_BYTES_PER_ROW`), decision D14 |
| KV row | `[positions, 256]` packed E2M1 (512 latent / 2) | `[positions, 32]` E4M3 (512 / 16) | `mla_ref.rs::ROW_PACKED_BYTES`, `KV_SCALE_CHANNELS = 16` |

### Verified against the released checkpoint, not inferred

Read from the safetensors header of `model-00003-of-00048.safetensors` in snapshot
`fb2764a5cf321eaa5070ca8f9e892818f477c16d` on the node, read-only. Layer 0 carries **2310** expert
tensors - 385 entries x 3 projections x (weight + scale), so the experts are 385 named entries, not
384 - and each projection is its own tensor rather than one stacked tensor:

| tensor | dtype | shape | what the shape says |
|---|---|---|---|
| `experts.0.w1.weight` | I8 | `[2304, 2560]` | 2304 = 576 per rank x 4 ranks, the global intermediate width; 2560 = 5120/2, the hidden halved because FP4 packs two values per byte |
| `experts.0.w1.scale` | F8_E8M0 | `[2304, 160]` | 160 = 5120/32: the per-row grid of section 2 - 160, not `ceil(2304/32)` |
| `experts.0.w2.weight` | I8 | `[5120, 1152]` | out is the 5120 hidden; 1152 = 2304/2 |
| `experts.0.w2.scale` | F8_E8M0 | `[5120, 72]` | 72 = 2304/32 |

Two consequences the upload path must implement, and both are visible in those numbers:

1. **The stored tensors are global-width, so a rank reads a column slice.** Rank 0's share of an
   expert is 576 of the 2304 columns - 288 packed bytes of the 1152 - and its scale row is 18 bytes of
   the 72. The reference already warns about exactly this slice: a loader that produced "the
   reference's whole-expert 72-wide or 576-wide per-rank 18-wide scale rows would misindex every row
   after the first", and `check_moe_layout` is the assertion that catches it.
2. **The kernel's stride is padded and the checkpoint's is not.** The kernel wants 320 packed bytes and
   20 scale bytes per row (`WIDTH_PADDED = 640`, because 576 packs to 288 and 576/32 = 18), so the
   upload must widen every row from 288 to 320 and from 18 to 20 with a zero-filled tail. The tensor's
   own strides cannot be passed through.

The activation operand is FP8 E4M3, not packed FP4: `moe_combine` asserts this, and its message calls
the wrong operand "the packed E2M1 operand this kernel must not accept". The routing weight is part of
the activation operand (it is multiplied in before the bf16 cast the quantiser reads), *not* folded
into the combine GEMM.

## 3. Device layouts

`SPARK_M4_HIDDEN = 5120`, `MOE_WIDTH_PADDED = 640`, `MOE_QUANT_BLOCK = 32`, `FP4_PER_BYTE = 2`,
`MOE_TOP_K = 6`, from `kernels/m4_common_sm121.cuh` and the kernel's own header.

**MoE combine (`b12x_mxfp4_moe_sm121`).** The kernel takes five flat buffers, so the device layout is
exactly their concatenation, in the pre-padded strides the reference fixes (`pre_padded_moe_layout`):

| buffer | device layout | bytes |
|---|---|---|
| `hidden` | `[tokens][5120]` bf16 | `tokens * 10240` |
| `act_fp8` | `[tokens*6][640]` E4M3 | `tokens * 3840` |
| `act_scales` | `[tokens*6][20]` ue8m0 | `tokens * 120` |
| `w_fp4` | `[experts][5120][320]` packed E2M1 | `experts * 1,638,400` |
| `w_scales` | `[experts][5120][20]` ue8m0 | `experts * 102,400` |
| `expert_ids` | `[tokens][6]` u32 | `tokens * 24` |
| `combined` | `[tokens][5120]` bf16 | `tokens * 10240` |

### Sizes, with their config and their scope stated

Every size below says which of three things it measures, because conflating them is how a slice
becomes four times too big: **global checkpoint** (the tensor as stored, all four ranks) versus
**rank-local device** (the slice rank 0 uploads, after the column slice and the stride padding), and
**which kernel** (the combine kernel takes one weight tensor, the down projection `w2`, not all six).

Released dimensions: hidden 5120, global intermediate 2304 = 576 per rank padded to 640
(`WIDTH_PADDED`), `SCALE_BLOCK` 32, and **384** expert entries - the index's maximum index is 383,
not 384, and each entry carries **six** tensors (`w1/w2/w3` x weight+scale), which is 2304 tensors in
layer 0.

| per expert | global checkpoint | rank-0 device slice |
|---|---|---|
| `w2` packed | `[5120, 1152]` = 5,898,240 B | `[5120, 288]` padded to `[5120, 320]` = 1,638,400 B |
| `w2.scale` | `[5120, 72]` = 368,640 B | `[5120, 18]` padded to `[5120, 20]` = 102,400 B |
| **combine subtotal** | **6,266,880 B** | **1,740,800 B (1.66 MiB)** |
| all six tensors | 18,800,640 B (17.93 MiB) | 4,874,240 B (4.65 MiB) |

A whole layer at released dimensions is therefore **7.22 GB globally** (384 x 18,800,640 B, six
tensors) and **1.87 GB for one rank's six tensors** - both past the 2 GB budget and between a fifth
and two thirds of the node's 3,115 MB free. The combine kernel needs only `w2`, which would be 668 MB
for all 384 experts, and the slice does not take even that: **the expert bound is 64 experts - 111 MB
for the combine weights, 312 MB if all six tensors are ever uploaded** - and the report states the
bytes actually allocated.

**Device KV for `flash_mla_csa2_sm121`.** The kernel reads the latent cache as two parallel flat
arrays, so the device side is a structure-of-arrays, not an array of structs:

* `kv_fp4`: position `p` at byte offset `p * 256`, next position at `+256` - no per-block padding,
  because the kernel indexes positions directly and the tile shapes are the host's business.
* `kv_scales`: position `p` at byte offset `p * 32`.
* `selected`: the candidate position ids, `[selected_count]` u32, at most `MAX_SELECTED = 512`.
* `index_key`: `[2048][8][128]` bf16 (HSI blocks x 8 tokens x index dim), `index_query`:
  `[64][128]` bf16, `block_scores`/`block_best`: `[2048]`.
* `out`: `[64][512]` bf16.

A paged allocator is therefore *not* needed for the slice: a flat cache of `positions` rows is 288
bytes per position, so 4096 positions is 1.2 MB. Paging belongs to the serving layer, not to this
kernel, and the design should not invent it yet.

## 4. Upload path: loader to device

The upload path is deliberately dumb and auditable:

1. **Locate** the tensor by name in the checkpoint index (`safetensors.rs::tensor` gives the header
   without decoding). Shard file, offset, dtype and shape all come from the header.
2. **Bound** the transfer by elements before any allocation, from the expert bound above. This is
   where the 32 Mi limit is respected rather than tripped: the slice never asks for a whole tensor
   that cannot fit.
3. **Stage** the bytes in page-aligned host memory with `mmap`-style reads of the shard range, in
   expert-sized or row-sized chunks, so peak host memory is one chunk, not the tensor.
4. **Copy** each chunk with one asynchronous `cuMemcpyHtoDAsync` into its pre-computed device offset,
   then release the staging buffer.
5. **Verify** the transfer cheaply: a sha256 of the staged bytes is compared against the same range
   read a second time through a different chunk size. This catches short reads and stride mistakes
   without decoding anything.

The device offsets are computed once by a Rust-side mirror of the layouts in section 3, and that
mirror is what the slice's host code hands to the kernel. The reference's `check_moe_layout` exists
precisely because an off-by-one in these strides misindexes every row after the first; the device
mirror must be checked against it, not trusted.

Engram rows take the same path with a different source: a row is fetched into the M2 cache by
`cache.get_or_fill_rows`, read out of the cache, and uploaded as one row. Row addressing is by
`(layer, row)` through `layout.rs::row_offset`, never by element offset into a materialised table.

## 5. The six kernels and their per-layer dispatch

| kernel | stage | launch |
|---|---|---|
| `mega_mhc_sm121` | fused Mega-mHC: the two hyper-connection sweeps, the A/B/C shifts and the statistics in one pass over the hidden dimension, one block per mHC tile | `SPARK_M4_MHC_THREADS = 128` |
| `b12x_mxfp8_wo_a_sm121` | the grouped output projection `wo_a`: 8 groups of rank 1024, each reducing 8 attention heads' concatenated 512-wide outputs | dense B12x grid |
| `b12x_mxfp8_dense_sm121` | the dense MXFP8 GEMM that every dense projection uses: attention projections and the MLP up/gate halves | dense 2-D scale grid, `[ceil(out/32), ceil(in/32)]` |
| `flash_mla_csa2_sm121` | attention plus HSI indexer scoring | `SPARK_M4_MLA_THREADS = 256` |
| `b12x_mxfp4_moe_sm121` | routed expert **combine**; the up/gate halves are the same MXFP4 GEMM and are deliberately not duplicated in this file | `grid(tokens, 20)`, block 256, one output element per thread |
| `cert_lm_head_sm121` | certified LM head: 32,320 logits per rank against a 5,120-wide hidden, cheap upper bound first, exact dot product only when the bound can win | row-blocked over the vocabulary |

Dispatch is **data-driven, not a match on a layer index**: each layer descriptor states which stages
it runs, and every launch carries `layer_id` and `step_seq` because the kernels write into the
`GpuTraceRing`. The two trace-argument slots are part of the ABI, not optional, so a launcher that
forgets them changes the arithmetic's operand layout - the ring pointer sits between the shape
arguments and the trace metadata in the MoE signature.

`cert_lm_head_sm121` is the one stage that cannot be deferred: `head.weight` is refused by the 32 Mi
limit, so it must run row-blocked over the vocabulary, with the block size chosen so the logits
buffer stays small. That is dispatch design, not an optimisation.

## 6. The vertical slice

One layer, real weights, one kernel, one diff:

* **Kernel under test**: `b12x_mxfp4_moe_sm121`, the combine GEMM. Real expert weights from the
  checkpoint into `w_fp4`/`w_scales`.
* **Inputs**: a small token count (`MOE_TOP_K = 6` slots each) with expert ids chosen from the real
  routing rather than invented, activations built by the CPU reference's own
  `routed_activation_operand` so the A operand is the reference's arithmetic, not a hand-rolled copy.
* **Reference**: `mxfp4_ref::moe_combine` on the same bytes. It returns bf16 bits, the same form the
  kernel writes, so the comparison is bit-level, not a re-derived float comparison.
* **The padding is asserted, not assumed**: a wrong column slice still produces a finite, wrong
  number, so the slice checks the layout explicitly. The rank-0 slice is bytes 0..288 of each
  1152-byte global row and scales 0..18 of each 72, and both are **derived** from the reference's own
  `MOE_WIDTH`, `FP4_PER_BYTE` and `QUANT_BLOCK` rather than typed as literals, with a
  `const` assertion pinning them to 288 and 18. The destination pad 288..320 and 18..20 stays zero.
  The stored dtype and shape are asserted against the released `[5120, 1152]` and `[5120, 72]`, and
  the header's byte span is asserted to be exactly `rows x src_row`.
  **The general rule: only the destination pad can be asserted. The source tail cannot**, and what
  proves the source slice is the AGREEMENT between kernel and reference consuming the same bytes - a
  differing slice or stride convention then surfaces as a disagreement rather than as a plausible
  number.
  **Bytes 288..1152 of a source row are ranks 1..3's real data, not padding.** An earlier version of
  the slice asserted that tail to be zero; it fired on the very first real tensor, which is recorded
  here because the mistake is easy to repeat. What proves the slice is therefore not the tail but the
  agreement: a slice or stride convention that disagrees with the reference shows up as a
  disagreement, because both sides consume exactly the same bytes. A failed assertion stops the run;
  it does not degrade into a plausible result.
* **Budget**: I will state the exact device bytes allocated before the launch and the actual peak
  `MemAvailable` on the node before and after. The expert bound is chosen so the total stays under
  roughly 700 MB, a fifth of the node's free memory at worst, and I will not allocate the full layer
  if the pre-launch check says the node is tighter than that.
* **Deliverable**: a measured agreement number (max absolute delta and the fraction of elements
  exactly equal, in bf16 bits) and the diff, not a pass/fail test. The point of the first execution is
  to learn what the hardware does, so a disagreement is a result.

## 7. What the slice does NOT prove

* It does not run a model: no attention, no routing, no sampling, no token generation.
* It does not prove the other five kernels, and it does not exercise `flash_mla_csa2` at all.
* It does not prove the KV cache layout: section 3's KV design is written but the slice does not use it.
* It does not prove multi-rank anything: no RoCEnante, no sharding, no all-reduce.
* It measures one kernel's launch time (section 11) but makes no performance claim about the
  layer or the model; the device was empty when it was measured, which section 11.2 states.
* It does not prove the Engram path: row retrieval is designed but not exercised, because no store
  exists on the node yet.
* It does not prove the upload path for the tensors the 32 Mi limit refuses; it proves the path for one
  expert-weight tensor that fits.

## 8. Open items to settle in code, not prose

1. `SPARK_M4_MOE_TILE_D` is 256 and `static_assert(SPARK_M4_HIDDEN % 256 == 0)` holds for 5120, so the
   grid is `(tokens, 20)`. Confirm against the real launch rather than trusting the header.
2. Whether the checkpoint's routed expert tensors are stored as one tensor per expert or one stacked
   tensor per layer - the upload path differs, and the slice must read the index to find out.
3. Whether `nvrtc` at run time or a build-time cubin is the right way to get the kernel into the
   process. Build-time is more auditable and matches "compile here, execute there"; run-time would
   need nvrtc dlopen'd on the node. The design leans build-time.
4. The exact per-launch trace-ring arguments, which are part of the ABI and must be threaded through
   the launcher. The slice builds the cubin without `-DSPARKFLASH_ENABLE_GPU_TRACE`, so the trace
   macros compile to `(void)(ring)` and a null ring is safe; with the trace enabled the same argument
   is dereferenced and the launcher would have to supply a real `GpuTraceRing`.

## 9. The first execution attempt, and what the device said

The host half ran end to end: the released checkpoint was read, 64 experts of layer 0's `w2` were
sliced from 1152-byte global rows into the kernel's 320-byte padded stride, and the CPU reference
computed its result without a single assertion firing. The device half stopped at context creation:

    device 0: NVIDIA GB10 of 1 (total 124610 MiB)
    cuDevicePrimaryCtxRetain failed: 2 (OUT_OF_MEMORY)
    cuCtxCreate_v2            failed: 2 (OUT_OF_MEMORY)

The calls that succeed and the calls that fail, exactly:

| step | result |
|---|---|
| `cuInit` | 0 (SUCCESS) |
| `cuDeviceGetCount` / `cuDeviceGet` | 0 (SUCCESS), one device |
| `cuDeviceGetName` | 0 (SUCCESS), "NVIDIA GB10" |
| `cuDeviceTotalMem_v2` | 0 (SUCCESS), 124,610 MiB |
| `cuDevicePrimaryCtxSetFlags` | called with `CU_CTX_LMEM_RESIZE_TO_MAX`, then |
| `cuDevicePrimaryCtxRetain` | **2 (OUT_OF_MEMORY)** |
| `cuCtxCreate_v2` | **2 (OUT_OF_MEMORY)**, tried once, not retried |

So the driver is reachable and the device is present; obtaining a context is what fails.
MemAvailable across the same attempt: **3,768,676 kB before, 4,409,276 kB after** - it rose, because
no device memory was allocated and the movement is page-cache noise. Nothing was allocated, no
context is left retained, and nothing is retried. The first attempt returned
CUDA_ERROR_INVALID_CONTEXT (201) because only `cuInit` had been called - that was a launcher bug and
it is fixed. The surviving failure is not: both routes were tried, the primary context was asked not
to pre-reserve maximum local memory (`CU_CTX_LMEM_RESIZE_TO_MAX`), and the node had 3.7-4.4 GB of
system memory available against a device reporting 124,610 MiB total. vLLM was up throughout and was
not touched, stopped or reconfigured.

This is a capacity statement about this node in this state, in the same family as `doctor --local`
exiting 1: a second context does not fit beside the running inference. It is not evidence that the
kernel or the launcher is wrong, and it is not a reason to stop the inference to find out.

Section 11 records what happened when the device was freed: it ran, twice.

Contributed from the Engram work (task-33), supplied by engram-store; the body is theirs, with only
the heading level adjusted to this file's numbered-section convention.

## 10. Engram read path: what the I/O budget actually requires

Measured, not inferred: M2 store measurements plus the M1 trace ring, all local, zero checkpoint I/O apart
from eight released rows read read-only to prove the decoder - layer 1 ids 0, 1, 4095, 12345678 and the last
row 384006167, layer 14 ids 0, 7 and the last row 384016681, 2,112 bytes of weight and scale bytes in total.
This section is the budget, with scopes stated.

### 10.1 The row consumer, per row

One row is 264 bytes: 256 F8_E4M3 weights and 8 F8_E8M0 scales, one per 32 columns (the PerRow32 rule of
dtype.rs; layout.rs::ENGRAM_ROW_BYTES = 264). At the real 1 GiB cache budget (uma 256 MiB + dram 768 MiB -
the default 8 GiB DRAM tier is unreachable on the serving node) a cache hit costs p50 282-303 ns and the
decode costs p50 5,089-5,417 ns, so 5.6-6.5 us per row. Scope: per row and per LOOKUP, on every rank -
every rank performs all 24 lookups per layer, so this figure is not divided by world size.

The decode line is the reference in crates/sparkflash-kernels/src/engram_head_ref.rs, written to follow the
tree literally, and quant.rs's f8e4m3_to_f32 calls pow2() per element (quant.rs:109-128). A 256-entry table
decodes the same row in 824 ns, 6.5x faster: a serving path wants the table, not the literal decoder. Both
figures are under the 18 us that would be needed to keep 32 reads in flight by compute alone (measured,
io_uring at depth 32, on this box's storage), so the consumer cannot buy the concurrency: the issue side
must lead.

### 10.2 Requirement: queue_depth >= 48

At batch 1 a token offers 24 lookups per layer, below the depth that measured 18.1 us/row. queue_depth must
be at least 48, with both layers' lookups issued together, or the next token's ids prefetched. Otherwise the
token sits in the depth-8-to-24 regime at 1.2-3.2 ms/token. This is an I/O-side requirement and it does not
depend on where the head runs.

### 10.3 Requirement: the head must run on the device

The head is not a small sigmoid mix. It is a wkv projection of the 24 concatenated rows, [25600 x 6144] at
the released shape (dim 5120, hc 4, 24 columns = orders x heads, forward/engram.rs:63-70 and 589-592), which
is 157,286,400 multiply-accumulates per token per layer, then hc = 4 normalised dots of 5120 with the signed
square root (forward/engram.rs:648-664), then a 4 x 5120 write.

Measured on a CPU core: 152-184 ms per token per layer, about 0.35 s per token for both layers, against
0.87 ms of I/O at depth 32 (scope: per token, global - both layers are on every rank). The head is therefore
viable only on the device, which is what the shipped path does: the trace scope is ENGRAM_GPU_GATE. The
number to check against the 0.87 ms is 0.629 GFLOP per token for both layers, i.e. the device must sustain
about 0.72 TFLOP/s to hide the head inside the I/O. This is not "I/O-bound, compute is free": the I/O is
hideable ONLY IF the head is on the device. The 0.72 TFLOP/s is a REQUIREMENT derived from the operation
count, not a measurement - no kernel was run to produce it.

### 10.4 Sizing finding: the decoded wkv alone is the node's whole budget

head_ref needs wkv decoded to f32: 600 MiB per layer, 1.2 GB for both, because read_dense
(forward/engram.rs:486-503) decodes fp8 to f32. On the serving node with 3,086 MB available that alone is
the whole budget, before any KV cache. Like CacheConfig::default()'s 8 GiB DRAM tier, code that takes the
default sizing there is wrong by construction.

### 10.5 What this does not prove

- The device-level numbers behind 10.1 are this box's filesystem, not the store: ZFS over SATA SSDs, no NVMe.
  The store's own request is 16.46x the logical row (rchar; the straddle model predicted 4,358 B against
  4,346 B measured, 0.3%). The 240x device figure is ZFS reading whole records and must not travel as if it
  were the store's.
- The CPU head figure is a scalar reference. It bounds nothing about the device except the operation count.
- The 18 us threshold comes from the depth-32 io_uring measurement on this box's storage. On the node's NVMe
  it must be re-measured, not assumed.
- The row path was measured on synthetic rows. The released rows were used only to prove the decoder bit for
  bit against an independent decoder, on those eight released rows - including the last row of each table
  (384006167 on layer 1, 384016681 on layer 14), which is what proves the tail is data and not padding.
- Withdrawn: an earlier "8.8x slack" figure is withdrawn rather than adjusted. It compared a reconstructed
  CPU consumer against the token's I/O, and with the real head the ratio inverts by about 400x. It is
  recorded here so the correction stays visible instead of being silently applied.

## 11. It ran, and what it measures

Both correctness runs were made on the GB10 after the operator stopped the resident inference, and the
throughput numbers below were taken in the same window. Same cubin throughout (sha256 183ed3c7...). The
launcher gained a `--repeat` flag and a CUDA-event timing region for the throughput runs; its no-flag
output was diffed on the node against the launcher that made the correctness runs and is identical, apart
from the device line, whose MemAvailable changes with what else is on the box.

### 11.1 Correctness, on the device

- `--experts 64 --tokens 8`: 40,960 output elements compared, 40,960 exact, 100.0000%, max abs delta 0e0.
- `--experts 384 --tokens 32`, every routed expert of layer 0: 163,840 elements compared, 163,840 exact,
  100.0000%, max abs delta 0e0.
- Both: `rank-1..3 tail zero and pad tail zero in every uploaded row: true`.
- Device bytes allocated, from the launcher's own tally, reconciled to the byte against its seven
  `cuMemAlloc` requests: hidden `tokens x 5120 x 2`, activations `tokens x 6 x 640`, activation scales
  `tokens x 6 x 20`, weights `experts x 5120 x 320`, weight scales `experts x 5120 x 20`, expert ids
  `tokens x 6 x 4`, output `tokens x 5120 x 2`. That is 111,606,912 B (106 MiB) and 669,250,048 B
  (638 MiB) for the two runs, and the same tally reproduces 114,542,592 B and 671,598,592 B for the same
  expert counts at 128 tokens.

Bit-exact agreement between kernel and reference on the same uploaded bytes proves the kernel's arithmetic
on the rank-0 column slice. It does not by itself prove those bytes are the interesting ones - that is what
the shape assertions and the destination-pad check of section 2 cover, and what section 6's rule about
source tails leaves to agreement rather than assertion.

### 11.2 Throughput: one kernel, empty device

Fifty timed launches per configuration after one untimed warm-up. Uploads, the comparison and the CPU
reference are all outside the timed region. Each launch gets its own CUDA event pair for the distribution;
the whole batch gets one event pair for the per-launch figure, which is the one to quote because per-launch
synchronisation would otherwise be part of every sample. `ms/launch` is the batched figure; the range is
min - p50 - max of the fifty samples. `tokens/s` is tokens per launch divided by the batched time: a
kernel-level rate, not a model rate.

| experts | tokens | ms/launch | per-launch min - p50 - max | tokens/s |
|---|---|---|---|---|
| 64 | 8 | 1.2074 | 1.1543 - 1.1636 - 1.6147 | 6,626 |
| 64 | 128 | 32.8389 | 32.67 - 32.81 - 33.42 | 3,898 |
| 384 | 32 | 8.5165 | 8.19 - 8.46 - 9.14 | 3,757 |
| 384 | 128 | 33.1731 | 32.93 - 33.05 - 34.02 | 3,859 |

At 128 tokens the two expert counts land within 1% of each other: the per-launch time is set by the token
count and not by how many experts were uploaded, which is the shape expected of a grid of `(tokens, 20)`.
The 8-token row is overhead-dominated - 1.2 ms for 160 blocks.

TWO BOUNDARIES that must travel with any number above:

- The device was EMPTY. vLLM was stopped by the operator to make room (MemAvailable 120,267 MiB on a
  device reporting 124,610 MiB). This is not a serving-condition figure. The resident model is exactly what
  refused a second context in section 9, and nothing here says what the same kernel does beside it.
- This measures ONE KERNEL of the six, on one layer's combine weights, rank-0 slice, 8 or 128 tokens, with
  no routing, no attention, no sampling and no KV cache. It is a kernel launch time, not a layer time and
  not a token time for the model.
