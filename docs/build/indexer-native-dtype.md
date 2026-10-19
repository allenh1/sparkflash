# The indexer's native dtype (task-8 / F3)

The HSI indexer was the last attention projection whose arithmetic was f32 where the released
model is not. docs/build/native-arithmetic-audit.md rows 19-21 describe it: the reference's wq_b is
an fp8 W8A8 ColumnParallelLinear, and its score chain is bf16, while the chain's host indexer used
an f32 matmul over dequantised weights with no activation quantisation and kept every score value
in f32. Batch 2 implemented the native rule. Batch 8 (task-8) only moved the projection behind
the device seam: same rule, same numbers, new owner for the launch. See "F3 status" below.

## The native rule

- model.py:519 fixes wq_b at the model dtype, so act_quant(q_norm_out, 32, ue8m0) runs on the bf16
  q_norm output, the fp8 GEMM accumulates in f32, and its output is bf16 (model.py:187-199).
- model.py:553-557 is a bf16 score path: einsum(q, index_k) hands back bf16, relu(score) * weights
  is a bf16 product, and the head sum accumulates in f32 and narrows to bf16 once.
- model.py:555 multiplies the bf16 weights_proj(x) output by a Python-float factor, which narrows
  to bf16.
- The two-level top-k and the reachability mask run on the bf16 scores (model.py:560-611); tie
  order and the shift into the shared index space are unchanged.

## What changed

- indexer_activation_native (attention.rs, batch 2; named indexer_query_activation then): the q
  latent is narrowed to bf16 (the q_norm output is a bf16 tensor in the reference) and quantised to
  E4M3 over 32-value blocks with the power-of-two ceil rule and the activation amax floor. It is the
  one implementation of that step and the public host rule `indexer_wq_b_native` is built on it.
- indexer_wq_b_native / index_source_indices (attention.rs, batch 2): the wq_b GEMM consumes that
  activation, accumulates in f32 over the dequantised fp8 weights and narrows the output to bf16;
  the rope and the fp4 q round trip are unchanged (their dtype is F5, task-10).
- AttentionProjectionBackend::indexer_wq_b (batch 8): the interface the device arm implements. The
  default `None` keeps the host rule; `AttentionDeviceBackend` returns rows from the already-pinned
  `b12x_mxfp8_dense_sm121` kernel, projecting in row chunks of
  `indexer_chunk_rows(cols, MAX_OUT_BYTES)` (8 rows at the released 4,096 columns) and quantising
  the activation per chunk. GEMM rows are independent, so the chunking is bit-transparent; no new
  kernel or pin was added.
- crates/sparkflash-cli/tests/indexer_seam.rs (batch 8, no GPU): the host rule and the kernel
  twin `sparkflash_kernels::mxfp8_ref::dense_gemm`, fed the same B12x-packed weight bytes and the
  same activation codes, agree bit for bit at [rows 5, cols 64, inner 64], and the chunked path is
  bit-identical to one projection of all rows.
- index_scores (attention.rs): the einsum result narrows to bf16, each rectified product narrows to
  bf16, the per-head sum accumulates in f32 and narrows once.
- index_weights (attention.rs): the bf16 layer input is narrowed, the weights_proj result narrows
  to bf16, and the factor product narrows to bf16.
- quant.rs gains f32_to_bf16_bits and bf16_round (ties to even); loader.rs's existing bf16 cast now
  delegates to the same function, so there is one cast implementation.
- The device chain prints one line at startup naming the indexer arithmetic, so a window log shows
  which arithmetic the host seam ran:

    indexer arithmetic: native dtype (bf16 q_norm output quantised to E4M3 over 32-value blocks,
    fp8 W8A8 wq_b with a bf16 output, bf16 weights and scores, discrete two-level top-k; the wq_b
    projection is the same device B12x MXFP8 GEMM in row chunks, while indexer scoring, weights and
    the two-level top-k stay host seams, and at position 0 the compressed set is empty so no
    projection launch runs)

Host tests (crates/sparkflash-engine, lib): the query activation is narrowed then quantised and is
a fixed point of a second round trip; the scores equal the bf16-rounded dot/product/sum and differ
from the f32 score; the per-query weights are bf16 values; the chunk size is derived from the
output buffer and a chunked projection is bit-identical to one projection
(`chunk_rows_are_derived_from_the_output_buffer`,
`chunked_projection_is_bit_identical_to_one_projection`).

## The discrete risk is not reachable at the current window lengths

The audit's MEDIUM rating rests on selection being discrete, not on the score delta. With the
released configuration this cannot fire at the lengths the b68/b71 windows run:

- index_topk = 512; the longest compared position is 137, and the compressed position count at the
  schema's largest position is end_pos / ratio <= 137 (ratio is 1 or 2), so
  topk = min(index_topk, n_pos) = n_pos: every reachable compressed position is selected.
- candidate_topk_blocks = 2048 x candidate_block_size = 8 = 16,384 candidate positions, far above
  n_pos, so the level-one mask keeps every position as well.

The selected list is therefore the reachability mask, independent of the scores, and the two
arithmetics produce identical selections until a run has more than 512 compressed positions (long
context with ratio 1 above 512 tokens, or ratio 2 above 1,024). A window at the planned lengths can
record the dtype line and the unchanged selection, but it cannot measure a flip.

## Measured effect

The committed CI sequence fixture (tests/fixtures/sequence-130-tiny.json) is generated by
generate_fixture, so it was regenerated in the same change and is the one host-visible numeric
comparison available without a checkpoint:

- sha256 before  b9e209239be17e4edcd594156131b56d149af247c53f9c827b8e14cbecf2a094
- sha256 after   77735f64cef45374d99cc0789d1c8eb2a1d78d96e7c600c79c42d1fe8ba5247d
- tokens changed: 0 of 9 steps
- max |logit delta| over all recorded pairs: 1.2e-8 (the tiny configuration's logits are small)

The tiny configuration has index_topk = 4 with roughly 65 compressed positions, so its selection is
not degenerate - a flip is possible in principle there, and none of the nine steps' tokens changed.

These are the batch-2 numbers. Batch 8 moved the projection to the device seam without touching
the host rule, so the fixture was not regenerated: it does not exercise a device build. No window
ran in either change.

## F3 status (2026-10-19, batch 8)

- The arithmetic is native and was accepted in batch 2; batch 8 is a seam change, not a second
  arithmetic decision. Both the host arm and the device arm run the same rule, and the no-GPU twin
  test pins that they agree on identical bytes.
- The device chain can now reach the indexer's wq_b projection on the already-pinned B12x MXFP8
  dense kernel (path A, Lead decision). No new kernel, no new pin, no fixture change, no window.
  The chunked launches are arithmetically transparent and their chunk size is derived from the
  shared output buffer, not hardcoded.
- Still host seams, deliberately: indexer scoring (`index_scores`), the per-query weights, the
  two-level top-k and the reachability mask. F3 does not need them on the device; the indexer's
  discrete selection is the measurement gap, and that stays on task-12 (N9).
- At position 0 the compressed set is empty (`n_pos == 0`), so a 40-layer position-0 chain run makes
  no indexer projection launch; the seam is exercised by the twin test, not by that run.

## What is owed

- A long-context measurement: a window with more than 512 compressed positions, comparing the
  selected top-k lists before and after this change (or against a released selection if one is ever
  recorded; vLLM does not record topk_idxs). Until then the dtype rule is verified by host tests and
  the selection claim is an arithmetic argument, not a measurement.
- F5 (task-10) still owns the bf16 rounds at the seams around this path (the fp4 q/k round trips
  see an f32 rope output), so the indexer is not fully native until that lands; this change is the
  F3 part only.

## What this does not claim

No device run happened with this change and no released-model measurement exists for it: the N9
stage re-run will record the selection and logits at the planned positions, which the argument above
says must be unchanged.

## Recorded follow-up (Lead decision 2026-10-19)

- The accepted evidence for F3 is the native rule, its host tests and the arithmetic argument
  above. No window leg is run for show: at the planned lengths a selection flip cannot happen.
- The long-context flip measurement is a named item on task-12 (N9): one long-context run with more
  than 512 compressed positions (ratio 1 above 512 tokens, ratio 2 above 1,024), comparing our own
  pre-change and post-change selected top-k lists layer by layer. vLLM records no topk_idxs, so
  there is no released selection to compare against.
- That leg must cover every discrete selection stage in the same run - the indexer top-k and any
  other top-k stage - not one leg per stage.
- The host test the_bf16_score_path_can_erase_a_margin_the_f32_path_keeps is a property demo of the
  mechanism only (a sub-step margin disappears and the tie rule decides); it is not evidence about
  the released model.
## Kernel half (task-45): covered by the existing pinned GEMM, zero ABI change

The Lead's decision on 2026-10-19 is that F3's kernel half is the **minimal** option. The native
arithmetic above already exists as a pinned, measured device kernel - b12x_mxfp8_dense_sm121 - and
its rule is exactly the one this change implements on the host:

- E4M3 activations with one ue8m0 scale per 32-value block (per row), the checkpoint's own E8M0
  scale grid;
- E4M3 weights in the B12x 32x32 micro-tile order with one ue8m0 scale per 32x32 block;
- f32 accumulation, K ascending in 32-value blocks and ascending inside a block;
- one bf16 round at the end.

No kernel, KernelId, catalog entry, baseline manifest line or ABI declaration is added for the
indexer. The typed launch is the existing pinned Mxfp8DenseParams
(crates/sparkflash-kernels/src/mxfp8_dense_launch.rs; KERNEL_ID = KernelId::Mxfp8Dense, ARG_WORDS =
11, ROWS_INDEX/COLS_INDEX/INNER_INDEX = 5/6/7, grid (ceil(rows/64), ceil(cols/64), 1), 256 threads,
64x64 tile) and the declaration stays the frozen copy in m4_abi::MXFP8_DENSE. The earlier b60
measurement already compared that kernel with mxfp8_ref on exact packed bytes
(bf16_mismatches=0/32768, max_abs_diff=0.0), so a second symbol would duplicate proven arithmetic
and add an ABI we would have to defend. A dedicated indexer_wq_b entry was considered and rejected
for that reason; it is only worth revisiting if the projection needs its own ownership boundary.

### The seam contract

Projection on device only: the engine keeps rope, the fp4 q round trip (F5/task-10), the bf16 score
chain, the per-query weights and the two-level top-k on the host exactly as they are today, and only
the wq_b GEMM moves. The operands that cross the seam are:

- activations: the bf16 q_norm latent, quantised by the engine (act_quant: one absolute maximum per
  32-value block, 1e-4 amax floor, ue8m0 round-up-to-power-of-two) to [seqlen][q_lora_rank] E4M3
  codes plus [seqlen][q_lora_rank/32] ue8m0 scales - the kernel does not quantise;
- weights: the checkpoint's [cols][inner] E4M3 bytes plus its [cols/32][inner/32] ue8m0 grid,
  tiled into B12x order by attention_device's read_packed (b12x_weight_index); no host
  re-quantisation of weights;
- output: [seqlen][cols] bf16 values (read back as bf16-representable f32 rows), with
  cols = index_n_heads * index_head_dim and inner = q_lora_rank.

Released geometry: cols = 4096, inner = 1280, 40 K blocks, weight scales [128][40], activation
scales 40 per row. The tiny configuration (cols = 16, inner = 32) is the same kernel with a
half-empty B12x N tile; the upper 16 columns are zero padding the kernel never reads.

### Evidence (no GPU run)

crates/sparkflash-kernels/tests/indexer_wq_b_twin.rs pins the contract at the indexer geometry
without a device:

- the_kernel_arithmetic_matches_the_op_by_op_reference_on_exact_packed_bytes: an independent scalar
  reference that starts from the seam inputs (bf16-representable latent -> act_quant_fp8_row -> B12x
  pack of checkpoint-order bytes) is bit-identical to mxfp8_ref::dense_gemm at
  [1, 4096, 1280], i.e. two implementations of the declared f32 order on the same bytes;
- rows_are_independent_so_a_chunked_launch_is_the_same_bytes: per-row launches concatenate to the
  same bf16 bytes as one multi-row launch, which is what makes the chunking below bit-transparent;
- the_tiny_config_geometry_is_accepted_with_zero_padded_weight_tiles: the 16x32 tiny geometry is a
  bijective B12x pack with a zero tail, so the host-side test configuration exercises the same path;
- an_accumulation_grouping_change_flips_the_bf16_result_when_the_sum_cancels: a property demo of the
  mechanism only. On ordinary data the per-element order and a block-dot order differ in f32 by
  about 1e-5 relative and the single bf16 round hides it; on a fixture whose real sum cancels (the
  last 20 blocks are the sign-negations of the first 20 with paired activation scales) the residual
  survives the bf16 round and the two outputs differ. This is why the twin mirrors the kernel's
  declared order rather than a "mathematically equivalent" one. Like the engine's bf16-margin demo,
  it demonstrates the mechanism; it is not evidence about the released model's scores.
- the_indexer_shape_is_whole_32_value_blocks_and_the_launch_geometry_is_the_pinned_one: byte counts
  (b12x_buffer_bytes and b12x_scale_bytes), grid for 1/8/137 rows, ARG_WORDS, KERNEL_ID and the
  pinned declaration (PARAMS shape words, declaration_matches) all hold at these dims.

### Trap 1: the shared output staging buffer is eight rows wide

attention_device.rs sizes MAX_OUT_BYTES at 32,768 bf16 values for the five attention projections.
Indexer wq_b emits rows * 4096 values, so eight released-geometry rows fit and nine do not. The
seam therefore chunks rows (chunk = MAX_OUT_BYTES / (cols * 2)), one D2H per chunk, rather than
growing the shared constant: the GEMM's rows are independent, and the test above proves chunking is
bit-transparent. Growing the buffer instead is an explicit decision (it changes a staging constant
whose other users are the attention projections), not a silent edit made to fit this seam.

### Trap 2: the indexer stays a host seam beyond the projection

Scores, per-query weights and both top-k levels are host and already native (the sections above).
A selection flip is only reachable with more than 512 compressed positions (index_topk = 512; ratio
1 above 512 tokens, ratio 2 above 1,024), which is the named long-context leg on task-12 (N9). A b70
window at the planned lengths cannot measure a flip and must not be used to re-open F3.

### Error-budget implication of the hidden f32 gap

The hazard measurements give F3's projection an explicit error budget, not just a test note:

- The seam's output contract is bf16, so the projection's own last round quantises every result at
  about 2^-9 (roughly 2e-3) relative. The measured f32 gap between two accumulation orders is about
  3.2e-5 relative on ordinary data, some sixty times smaller. For ordinary scores the accumulation
  order is therefore not a term in the projection's error budget: a reference whose f32 sum differs
  from the declared order by that much still produces identical bf16 rows.
- That also bounds what a bf16 comparison can prove. The b60 kernel-vs-mxfp8_ref result
  (bf16_mismatches=0/32768) is consistent with the same f32 order but would not by itself exclude a
  near-equivalent one; the pin on the order comes from the kernel source and the twin, not from a
  bf16 mismatch count. Any future device-vs-host claim should say which level it compares.
- The ordering can surface only where the bf16 margin collapses: a score near zero (the relu
  boundary), a tie, or a cancellation regime like the fixture. That is exactly the regime where the
  discrete top-k is most sensitive, so the residual risk is a selection risk, not a score error
  budget risk. At the current windows it cannot fire: index_topk = 512 is above the largest
  reachable compressed position count (137), so selection is the reachability mask. The named
  evidence for the residual is the task-12 long-context measurement with more than 512 compressed
  positions; if such a run records a bf16 margin below about 3e-5 relative or a selection at the
  threshold, the comparison then has to move to the f32 accumulation order, which the twin defines.

### What this does not claim

No device run happened for task-45: the kernel is the already-pinned one, so its b60 measurement is
the device evidence and this task adds the indexer-geometry contract on top. The device comparison
belongs to the window that first runs the projection on device, and it must be named there as its
own line rather than folded into the existing attention-projection claims.