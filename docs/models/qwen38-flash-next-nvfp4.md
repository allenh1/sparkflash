# Qwen3.8-Flash-Next (NVFP4)

Status: **served at TP=1 on an RTX Spark (GB10), with a tensor-core prefill and the DFlash 2
block drafter.** The pack is `sparkflash-model-qwen4-exp`, registered as
`nvidia/Qwen3.8-Flash-Next-NVFP4` (weight selector `qwen3.8-flash-next-nvfp4`). There is no host
forward arm: the model runs on the device or not at all. The current rates are in
[Where it stands](#where-it-stands); the sections after it record how the arm got there.

## What the checkpoint is

`nvidia/Qwen3.8-Flash-Next-NVFP4` is NVIDIA's ModelOpt 0.46 post-training quantisation of
`Qwen/Qwen3.8-Flash-Next`, a multimodal `Qwen4ExpForConditionalGeneration`. The released index has
**299,545 tensor names and 132,639,846,394 bytes**; the pack's plan derives all of them from the
configuration and the tests pin that total, so a shape this pack believes cannot silently drift
from the checkpoint.

| Property | Value |
| :-- | :-- |
| Served id | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Family | `qwen4-exp` |
| Decoder layers | 48: 12 x (3 linear-attention + 1 sparse-attention) |
| Hidden width | 2,560, widened to 4 streams (10,240) by the hyper-connections |
| Gated DeltaNet | 16 key heads, 48 value heads, 128 dims, conv width 4, sigmoid output gate |
| Qwen Sparse Attention | 24 query heads, 2 KV heads, head 256, rotary 64, QSA indexer 4Q/1KV x 128 |
| Indexer budget | 2,048 tokens in 4-token blocks (512 blocks) |
| Routed experts | 512 per layer, 10 active, intermediate 640, NVFP4 W4A4 |
| Shared expert | 640 wide, gated by a sigmoid scalar |
| PLE | layer 2 (index 1): 16 hashed heads (8 bigram + 8 trigram), 160 dims each, 320,001,536 rows |
| Vocabulary | 248,320 |
| Context | 262,144 native |
| MTP | 1 layer, the speculative drafter when no DFlash 2 drafter is named (72% of its drafts accepted) |
| DFlash 2 drafter | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), loaded separately: 5 layers, block 8, reads target layers 5/15/25/35/45 |
| Vision | 27-layer tower in the checkpoint; host front end and reference tower in the pack, device prefill not wired |

## Storage, and what is device-resident

| Class | Bytes |
| :-- | ---: |
| Text language model + PLE table | 129,043,957,786 |
| PLE n-gram table (128 FP8 shards + constants) | 51,200,246,042 |
| **Device-resident total** (text model, every non-PLE tensor) | **77,843,711,744** |
| MTP (served, speculative) | 2,698,026,496 |
| Vision tower (planned, not served) | 897,862,112 |

The 51.2 GiB n-gram table stays on the host. A token indexes 16 rows of 160 bytes, so the host
side of the PLE is 2.5 KiB per token: at 10 tok/s that is 25 KiB/s of unique reads. Keeping it on
the device would cost 51.2 GiB of HBM for no throughput, which is exactly the trade the Engram
store makes for V4.1.

Quantisation rules (verified against the shard headers):

* **Routed experts are NVFP4**: logical `[out, in]` stored as `U8 [out, in/2]` (two E2M1 nibbles,
  low nibble first), `F8_E4M3 [out, in/16]` block scales, plus per-tensor `F32` `weight_scale_2`
  and static `input_scale`. `sparkflash_weights::nvfp4` decodes exactly this for every ModelOpt
  checkpoint, and its tests pin the nibble order and the 16-column scale group (the E2M1 codebook
  is pinned with the shared decoders in `sparkflash_weights::quant`).
* **The PLE table is FP8**: 128 shards of `F8_E4M3 [2,500,012, 160]` with one per-tensor `BF16`
  scale. `split_ngram_parts = 128` divides the padded row count exactly.
* Everything else (attention, shared experts, router, hyper-connections, embeddings, LM head, MTP
  non-expert) is BF16, and the MTP routed experts are FP8 with 128x128 block scales.

## Where it stands

Measured on an RTX Spark (GB10, 48 SMs, `sm_121a`, CUDA 13.4), greedy, with a desktop session
open on the node; prefill is the range over four runs:

| | rate |
| :-- | ---: |
| prefill, 535-token prompt | 446-476 tok/s |
| prefill, 2,251 tokens | 476-527 tok/s |
| prefill, 8,983 tokens | 408-446 tok/s |
| decode, no drafter (`SPARKFLASH_QWEN4_SPEC=0`) | 21.5 tok/s |
| decode, the MTP head (one draft a step) | 28.3 tok/s |
| decode, DFlash 2 at the adaptive depth | 46.0 tok/s |

Decode is the aggregate over eight prompts (six chats and two raw completions, at most 256 tokens
each, 1,461 in all), and each drafter's text is identical to the no-drafter run's, character for
character: a draft changes how many positions a forward covers, never what they hold. The MTP
head has 72% of its one draft a step accepted (612 of 850); DFlash 2 has 73% of its drafts
accepted (1,085 of 1,477) and covers 3.9 tokens a verify step, from 2.0 on the prompt it drafts
worst to 7.4 on the one it drafts best. Prefill was 96-174 tok/s on
the token kernels before the tensor-core image.

### Prefill on the tensor cores

`kernels/qwen4_prefill_sm121.cu` is a second image that takes a chunk of `GEMM_MIN_BATCH` (64)
tokens or more. A prompt is cut into layer-major chunks of 4,096 tokens, the pieces the server
already hands the model: the routed experts are 1.26 GB of NVFP4 a layer and a chunk reads each
expert it routes to once per 32 of its rows, so the chunk width is what amortises them (at 128
tokens the weight traffic alone caps prefill near 580 tok/s, at 512 near 2,300). Inside a chunk:

* the dense projections are a BF16 `mma.sync` GEMM on the shared tile body
  (`kernels/opt_mma_common_sm121.cuh`, `kernels/opt_mma_bf16_sm121.cuh`) that the DeepSeek kernels
  also include, split over K when a matrix has too few tiles for the 48 SMs. The split is chosen
  per 512-row piece (`DENSE_ROWS`), as it was when chunks were 512 tokens, and a tail under 64
  rows still runs on its own, so every row computes to the same bits at any chunk width;
* the routed experts are grouped by expert (`q4p_moe_plan`) into items of 32 (token, slot) rows,
  and each item runs on the tensor cores against a panel of its expert's NVFP4 weights, decoded
  to BF16 in registers (exact: an E2M1 times E4M3 product fits BF16) with `weight_scale_2` applied
  in f32;
* the Gated DeltaNet convolution runs over the chunk's tokens in parallel, and the recurrence
  keeps its state in shared memory across the chunk;
* the sparse-attention indexer, selection and attention take 128 rows at a time
  (`ATTENTION_CHUNK`), since a row's block scores are a context's worth of floats.

| variable | effect |
| :-- | :-- |
| `SPARKFLASH_QWEN4_PREFILL_MMA=0` | every chunk on the token kernels, at their 128-token chunk |
| `SPARKFLASH_QWEN4_PREFILL_CHUNK=N` | tokens per chunk, clamped to 16..4,096 and, past 512, rounded down to a multiple of 512; the scratch is about 0.85 MiB a token |
| `SPARKFLASH_QWEN4_PREFILL_PARTS=dense,experts,gdn` | which of the image's three paths a chunk may take, for bisecting a difference against the token kernels |

### Speculative decode with DFlash 2

`SPARKFLASH_QWEN4_DFLASH2` loads the drafter beside the target and replaces the MTP head. It is
optional: unset, the canonical `incoai/Qwen3.8-Flash-Next-DFlash2` cache snapshot is used when
present, so `hf download incoai/Qwen3.8-Flash-Next-DFlash2` is the normal setup. Set the variable
to a path or a repo id only to override that choice. DFlash drafts a whole block at once: the token
the target just chose
and seven mask positions run through the drafter's five non-causal layers, whose keys and values
for the committed context come from the target's own hidden states at layers 5, 15, 25, 35 and 45.
DFlash 2 adds a grouped causal convolution across the block and a low-rank selector that walks
the target head's top candidates for each row, so the drafted tokens agree with one another. The
drafter's projections, norms, rotary and SwiGLU are the token image's kernels on its weights
(`kernels/dflash2_sm121.cu` adds the attention, the convolution and the selector), and its
embedding table is the target's. Its head is an E4M3 copy of the target's, and the block's
projections are E4M3 too (one f32 scale a row, quantised at load, read by the tiled GEMV with
sixteen codes to a lane's chunk): a draft reads half the bytes, and since a draft is only a
proposal the verify checks against the BF16 model, the output does not change. The context path
keeps BF16, because a prefill's rows run through it on the GEMM.

The five taps are the checkpoint's own definition, not an inference: the drafter's snapshot ships
`patches/qwen38-dflash2-capture.patch`, whose comment reads "DFlash2 was trained on completed,
zero-based layer outputs, averaged over the four hyper-connection streams", and whose assert pins
`layer_ids == [5, 15, 25, 35, 45]`. `capture_features` is called with the layer loop's zero-based
index after that layer's MLP has been added, and the capture kernel means the widened residual
over `hc_count`, which is the same tensor. So the acceptance this drafter measures - 73% of its
drafts over the eight-prompt set - is the drafter's own, not a mis-fed one; and the patch also
confirms the taps are a TP1/CP1 capture, which is what this arm serves.

A step verifies the chain as one forward of the target over up to eight rows. Every row computes
exactly as the one-token decode step it stands in for - the batched GEMV, the tiled wide GEMVs and
the hyper passes are held to the row-alone bits by `tests/verify_rows.rs` - so the greedy text with
a drafter is the text without one. The verify leaves each Gated DeltaNet layer's recurrent state
as it found it, writes its last row's state into a spare, and keeps the rows' convolved inputs and
gates. A step that keeps every row swaps the spare in; one that keeps fewer first replays just
those rows into the spare through the same recurrence kernel, which is bit for bit the state they
left. That replaces a 3 MiB per-layer snapshot after every row, and one spare state per layer
replaces seven snapshots. The short convolution and PLE windows still snapshot each row, and a
rejected draft rolls them back to the last kept row.

How many drafts a step verifies is the policy V4.1's MTP stages use,
`sparkflash_model::speculation::DepthPolicy`: the drafter's confidence in each draft, calibrated by
the acceptances the sequence has seen, against a modelled step cost (59 ms for the step and the
drafter's block, 8.4 ms a verified draft on an RTX Spark).

| variable | effect |
| :-- | :-- |
| `SPARKFLASH_QWEN4_DFLASH2=DIR\|REPO_ID` | override the drafter; unset uses the canonical cache snapshot, and no cache entry serves the MTP head |
| `SPARKFLASH_QWEN4_DFLASH2_DEPTH=adaptive\|fixed` | adaptive (the default) or always all seven drafts |
| `SPARKFLASH_QWEN4_DFLASH2_MIN_DEPTH=N` | the fewest drafts an adaptive step verifies |
| `SPARKFLASH_QWEN4_SPEC=0` | no drafter at all |
| `SPARKFLASH_QWEN4_SPEC_DEPTH=2` | two chained MTP proposals instead of one (MTP only) |
| `SPARKFLASH_QWEN4_SPEC_TRACE=1` | log each draft's confidence, its rank in the target's own row, and whether it was kept |

### Decode since the drafter

The last three changes to the step itself, measured over the eight prompts above with text
identical to the no-drafter run (the last two were measured together):

| change | no drafter | DFlash 2 |
| :-- | ---: | ---: |
| before | 20.68 tok/s | 41.73 tok/s |
| the wide GEMVs and hyper passes tiled, so a verify's rows share each activation read | 21.14 | 44.71 |
| the PLE staging slab fenced by an event instead of a full synchronize, and the expert down pass reading four rows a warp with an L2 prefetch of the gate/up panel | 21.52 | 45.97 |

### Where the decode step's floor is

[scripts/decode_floor.py](../../scripts/decode_floor.py) tallies a step's bytes from the
checkpoint's own safetensors headers and names each group after the profiler mark that times it -
the first rule of the [optimisation strategy](../optimization-strategy.md#0-the-rules-everything-else-follows)
applied to this arm. One row reads 9.46 GiB: 8.18 GiB of dense stack once, plus the 1.28 GiB that
its ten experts a layer name. At 270 GB/s, the rate `lm_head` reaches on this node, that is
37.6 ms, and one `SPARKFLASH_QWEN4_PROF=1` run at one row measures:

| stage | GiB | floor ms | measured ms | GB/s | of floor |
| :-- | ---: | ---: | ---: | ---: | ---: |
| GDN input projections | 2.829 | 11.25 | 11.40 | 266 | 99% |
| hyper connections (both passes) | 1.230 | 4.89 | 7.11 | 186 | 69% |
| `lm_head` | 1.184 | 4.71 | 4.65 | 273 | 101% |
| GDN output projection | 1.055 | 4.19 | 5.46 | 207 | 77% |
| sparse projections | 0.865 | 3.44 | 3.32 | 279 | 103% |
| shared expert | 0.449 | 1.78 | 2.21 | 218 | 81% |
| attention output projection | 0.381 | 1.51 | 1.91 | 214 | 79% |
| router | 0.120 | 0.48 | 1.17 | 110 | 41% |
| routed experts (10 of 512) | 1.282 | 5.10 | 6.77 | 203 | 75% |
| one row, all stages | 9.458 | 37.61 | 55.61 | 183 | 68% |

Three of the four largest stages read at 266-279 GB/s and have nothing left in them; the fourth
is the hyper pass. The rest of the gap is 2-30% in six stages of a few tenths to 2.2 ms, 6.7 ms
together - and a quarter of the profiled numbers is the marks, so about 5 ms of the real step,
which is what the wall clock says: the same one-token step takes 44.0 ms with the drafter off, so it moves its
9.46 GiB at 231 GB/s, **85% of the floor**, and the profile's total overstates it by a quarter
(the marks serialise the stream and cost about 4.5 us each, of which a step has about 1,400).

Measured by wall clock on this node instead (greedy, 256 tokens a request, three interleaved
rounds of one prompt; the profiled steps are not comparable with these):

| step | ms | tokens | tok/s |
| :-- | ---: | ---: | ---: |
| one row, no drafter (`SPARKFLASH_QWEN4_SPEC=0`) | 44.0 | 1.00 | 22.7 |
| the adaptive policy's step (60% of drafts accepted) | 71.2 | 2.45 | 35.8 |
| a full seven-draft verify (`SPARKFLASH_QWEN4_DFLASH2_DEPTH=fixed`) | 89.8 | 2.53 | 28.2 |

A verified row therefore costs **6.5 ms** (1.38 GB of experts, near their floor) and a step's
fixed part **44 ms**: the seven rows of a full block cost as much as the whole dense stack, and
they buy 0.08 more tokens a step than the policy's median depth. That is why `fixed` measures
28.2 tok/s against the adaptive 35.8, and why the depth policy - not a kernel - is now the
instrument that sets single-stream decode.

The policy's cost model (`DFLASH2_STEP_COST`: 59 ms a step, 8.4 ms a row) overprices both parts
of today's step by nearly the same factor (1.34 and 1.29), which turns out not to matter: the
policy compares a draft's chained probability against `rate * row_ms`, and a uniform overpricing
leaves that threshold where it was. Recalibrating it to 44.0 and 6.5 moves the depth chosen on
these prompts by at most one draft. What sets acceptance is the drafter, and every draft past the
third is worth less than its row: the fixed-depth run keeps 22% of its drafts against the
adaptive run's 60%.

## The architecture, as the reference implements it

1. **Embedding then widening.** The token embedding is tiled `hc_count = 4` times into the widened
   residual stream.
2. **Gated residual (hyper-connection).** Each branch read is
   `norm -> down (silu, /4) -> up (sigmoid) -> mean over streams`; the branch writes back
   `hyper + branch (x) injection`, with `injection = 2 * sigmoid(inject(normed) / 4)`. The final
   mixer folds the four streams into one with no injection. There is deliberately **no final
   RMSNorm**: the released model has none, which is why the plan has no `model.norm` tensor.
3. **Gated DeltaNet.** Depthwise causal convolution (width 4, SiLU), then the delta rule with
   `g = -exp(A_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)`, Q/K L2-normalised, 3 value
   heads per key head, gated RMSNorm output gated by **`sigmoid(z)`**.

   The gate activation is `output_gate_type`, not `hidden_act`: the reference builds this norm as
   `Qwen4ExpTextRMSNormGated(head_v_dim, eps, activation=config.output_gate_type or
   config.hidden_act)`, and the released configuration names `"sigmoid"` while `hidden_act` is
   `"silu"` (which governs the convolutions and the experts). Both arms originally applied
   `silu(z)` here, so every one of the 36 linear-attention layers ran its output gate at the wrong
   scale and sign - and device/host parity could not see it, because the two arms shared the
   mistake. On the released checkpoint the symptom was a flat logit row and token soup; with the
   gate corrected the model answers "The capital of France is Paris."
4. **Qwen Sparse Attention.** GQA with a sigmoid output gate. The QSA indexer pools complete
   4-token blocks, normalises and ropes each pooled key at the block start, scores
   `sum_h relu(q_h . k_h) / sqrt(128)`, keeps the best 512 blocks plus the incomplete tail, and
   applies that selection as an additive mask on top of the causal mask. Queries and keys are
   rope'd over the first 64 of 256 dims (base 10,000,000).
5. **MoE.** Router softmax (f32), top-10, renormalised (`norm_topk_prob`), routed bodies plus the
   sigmoid-gated shared expert.
6. **PLE.** `key_proj` (10,240 wide), `value_proj` (2,560), query/key norms grouped per stream,
   the scalar gate `sign(g) * sqrt(|g|)` then sigmoid, and a dilated depthwise convolution
   (kernel 4, dilation 3) added to the gated value.

## Reusing the Engram tricks

`sparkflash-engram` already owns the arithmetic that makes a 51B-parameter n-gram table servable.
The PLE is the same construction with Qwen's constants, and the pack reuses it rather than
reinventing it:

* **Hash multipliers**: the same `splitmix64`, the same golden-ratio increment and mixing
  constants, the same bound `(i64::MAX / vocab) / 2`, the same `2 * draw + 1` odd multiplier rule,
  and the same per-layer seed `seed + 10007 * ple_layer_index` (Qwen's `layer_multipliers`).
* **Prime bucket layout**: each of the 16 heads owns the next prime above
  `ngram_vocab_size_base = 20,000,000`; offsets are the running sum. The released 16 primes run
  from 20,000,003 to 20,000,171, for 320,001,446 rows, padded to 320,001,536 so 128 equal shards
  hold 2,500,012 rows each. `PleLayout` re-derives all of it, and the tests re-verify primality
  and the shard division.
* **Rolling multiplicative-XOR**: `token0 * m0` then XOR of `token_s * m_s`, reduced per head with
  `rem_euclid`; a prefill and a split decode produce byte-identical rows.
* **Serving shape**: a bounded row cache, whole-row batched reads per shard, and a prefetch entry
  point, the same shape as the Engram `EngramRowCache`/`PrefetchEngine` pair. The device arm calls
  it once per token at the PLE layer.

Where Qwen differs, the pack follows Qwen: a shift is valid only inside the current EOS-delimited
segment (an n-gram never crosses a turn boundary), an invalid shift contributes the EOS id, and
padding positions are rewritten to EOS before hashing. The tests pin five reference rows computed
with a Python transcription of the reference, including the row after an EOS.

## The device arm (TP=1)

`Qwen4ExpDeviceBackend::open` uploads every non-PLE tensor (77.8 GiB), keeps the PLE table on the
host, and refuses without a cubin, with a TP world other than 1, or with a context limit outside
`1..=262,144`.

### Converted rank file

`sparkflash convert --checkpoint "$CKPT" --output-dir "$CONV"` writes one `rank-0-of-1.bin`
(80,565,841,920 bytes, 151,721 slots, 73,728 NVFP4 scale pairs); `--inspect FILE --verify-weights`
re-reads every slot. The file holds the exact bytes the walk uploads, plus the two per-tensor
NVFP4 scales in walk order; the routed experts' pointer tables are rebuilt at load because they
hold device addresses.

`SPARKFLASH_QWEN_CONVERTED_DIR=$CONV sparkflash serve ...` reads the file into one allocation with
the parallel direct-I/O path the V4.1 and GLM arms use. Measured on an RTX Spark: **15.7 s from
launch to `/health`**, against 132 s from the released checkpoint; both include the DFlash 2
drafter. Conversion is one pass and took 341 s on the node.

The checkpoint directory is still required: its `config.json`, `tokenizer.json` and 51.2 GiB of
host-paged PLE shards are the source for everything but the resident weights. The converted file
replaces the 77.8 GiB per-tensor upload; it does not remove the checkpoint.

`tests/device_parity.rs::the_converted_rank_file_matches_the_checkpoint_path` runs the whole TP=1
loop over a file the converter wrote and requires the checkpoint path's logits; it agrees to
~2e-7 relative on the tiny fixture. `DeviceSequence` allocates per-session KV/indexer caches (f32), GDN convolution and
recurrent state, PLE convolution state and one workspace, then runs the layer order above one
token at a time through `kernels/qwen4_exp_sm121.cu`.

Kernels: embed, BF16 GEMV, grouped zero-centred RMSNorm, hyper mix/inject, gated residual
injection, SwiGLU, router softmax+top-k, NVFP4 expert body, MoE combine, GDN convolution and
recurrence, partial RoPE, QSA block selection, sparse GQA attention with the output gate, PLE
gate/convolution and the query/gate deinterleave. `build.rs` compiles them with nvcc when available
(it selects a host g++ nvcc 12.9 accepts, and honours `SPARKFLASH_NVCC_HOST`) and falls back to a
clang syntax probe on a CPU-only host; the cubin cfg is set only on a real compile.

Three properties of this cubin are deliberate and worth stating, because each was found by the
parity gate below rather than by reading the code:

* **No dynamic shared memory.** The shared launch path passes a dynamic shared-memory size of zero,
  so every buffer is a fixed array with a bound (`Q4E_MAX_LOWRANK`, `Q4E_MAX_INTERMEDIATE`,
  `Q4E_MAX_SELECTED`, `Q4E_MAX_BUDGET`) and the host refuses a geometry past it. The first version
  declared `extern __shared__` buffers, which faulted with an illegal address.
* **The indexer selection is bounded by the budget, not by the sequence.** A 262,144-token position
  has 65,536 complete blocks; a score per block was a quarter of a megabyte of per-thread local
  memory and overflowed anywhere past 4,096 blocks. Scores now fold into a top-budget set as they
  are produced and are emitted in ascending block order, matching the reference's list.
* **The router keeps its selected weights in a local array.** The softmax array doubles as the
  output buffer, so writing slot `i` back into it overwrote expert `i`'s probability before the
  remaining slots were ranked, and the fixture's two-slot selection picked the wrong second expert.
* **The indexer query's layernorm weight is applied once.** It is applied by the query norm before
  RoPE, and the selection kernel applied it again, scaling every block score by up to 1.1x. That
  changed which blocks a position attended to whenever the budget truncated the set, which is
  every position past the budget - and nothing before that, because an untruncated selection does
  not care about the scores.
* **E4M3 has no infinity, and the block scales use its exponent-15 codes.** The decoder returned
  `+inf`/`-inf` for `0x78..0x7E`, which are finite values (256 to 448) in the format; only
  `0x7F`/`0xFF` are NaN. The released experts carry 420 such codes in layer 0's first nine scale
  tensors, so the very first MoE layer of the very first token summed `+inf` and `-inf` into one
  row and turned the whole expert accumulator into NaN - which the router then reported as a
  `-1` index, two hundred lines away from the cause. The host decoder had it right all along, and
  the tiny fixture never caught it because it encoded every block scale as `1.0`. The fixture now
  spreads its scales across the codebook, and `tests/quant_kernel.rs` pins all 256 codes against
  the host decoder through a test-seam kernel (`q4e_decode_f8e4m3`).
* **The PLE hasher is seeded by the layer's index, not its position among PLE layers.** The two
  coincide only when the PLE sits on layer 1, which is what the tiny fixture does and what the
  released checkpoint does not (it sits on layer 2), so the released model was hashing a different
  table row for every token. Each PLE layer also owns its convolution state now; one shared buffer
  would interleave the histories of two PLE layers.

**Verified here:** `tests/device_parity.rs` runs the whole TP=1 loop - conformance, upload, host
PLE pages, prefill and three decode steps - against the host oracle on the tiny checkpoint, and the
logits agree to ~2e-7 relative on an RTX 5080 (sm_120). A second sequence carries an end-of-text id
in the middle of the prompt, so the PLE segment rule (which restarts the n-gram window) is compared
across the two hashers rather than assumed. A second configuration is deliberately asymmetric -
three residual streams, a five-row bottleneck, six query heads, a three-way top-k, a two-token
n-gram, PLE on a layer that is not the first - because the tiny fixture is powers of two everywhere
and a kernel that only works on those passes it. The same run on a build whose cubin targets
another architecture skips with a message rather than failing.

**Verified on the released checkpoint** on a 128 GiB GB10 (aarch64, `sm_121a`): it uploads,
prefills and decodes, and returns `The capital of France is Paris.` Its throughput and the
bandwidth model behind it are in "Decode throughput" above. Because the tiny fixture is powers of
two everywhere, the released run is also the only exercise of the widths themselves - 2,560-wide
rows, 10,240-wide hyper-connections, 512 experts - rather than a different code path.

**Not verified:** parity against vLLM's quantised path on the released checkpoint. The device
arithmetic has the one documented difference above, so the gate is a tolerance, and the tiny
fixture is what pins it.

Two diagnostics exist for a deployed run, both off unless asked for and both silent on a healthy
token: `SPARKFLASH_QWEN4_PROBE=1` copies back and reports the first buffer that holds a non-finite
value, naming its layer, position and phase (`embed`, `after_ple`, `gdn_conv_out`, `mixed`,
`router_logits`, `expert_accumulator`, `moe_combine_out`, `after_mlp`), and
`SPARKFLASH_QWEN4_TRACE=1` synchronises after every launch and names the kernel. The probe is what
localised the E4M3 bug to layer 0's expert accumulator without a 77.8 GiB bisect. A router selection
outside the expert table is a typed error that prints every index and weight, not an
index-out-of-bounds panic.

The device arithmetic has one documented numeric difference: NVFP4 experts use f32 activations, so
`input_scale` is carried but not applied. That is more accurate than W4A4 but not bit-identical to
vLLM's quantised path, which is why the gate is a tolerance.

## Decode throughput

A decode step reads the whole dense stack and a tenth of the expert stack, so the
bandwidth floor is set by what the checkpoint actually stores, not by the activated parameter
count. Tallied from the safetensors headers (`/tmp/q38_dtypes.py` in the session):

| class | on disk | dtype | read per token |
| :-- | ---: | :-- | ---: |
| Routed experts, 512 x 48 | 70.47 GB | NVFP4 (U8 nibbles + E4M3 scales) | 1.38 GB (10 of 512) |
| Attention stack | 6.87 GB | **BF16** | 6.87 GB |
| `lm_head` + `embed_tokens` | 2.54 GB | BF16 | 1.27 GB (`lm_head` only) |
| Shared expert, router, hyper-connections | 1.38 GB | BF16 | 1.38 GB |
| PLE n-gram table | 51.20 GB | FP8 | 2.5 KiB (stays on the host) |

That is **10.9 GB per token, 63% of it the BF16 attention stack**. At the 286 GB/s that
`lm_head` reaches on this node (1.27 GB in 4.44 ms), one token per weight pass floors at
**26 tok/s**, and two tokens per pass - a batch of two, where both rows share one weight read -
at **52 tok/s**. A fully 4-bit model would floor near 91 tok/s, but this artifact only quantises
the routed experts: `attention` is 6.87 GB of BF16 and dominates the read. Reaching 80-100 tok/s
at c=2 needs ~5.7-7.2 GB per token, which is quantising the attention stack (FP8: 38 tok/s,
NVFP4: 48 tok/s single stream) or verifying several tokens per pass (MTP, below).

**Measured, on the GB10 node, greedy 32 tokens:**

| | decode | TTFT |
| :-- | ---: | ---: |
| Start of the throughput work | 0.20 tok/s | 21,700 ms |
| Whole layer on device, scale-block hoisting, bf162 loads | 6.4 tok/s | 783 ms |
| Warp-per-row GEMV, parallel top-k, staged activations | 9.9 tok/s | 524 ms |
| Parallel hyper-injection | 15.5 tok/s | 345 ms |
| No host readback per layer, parallel indexer scoring | 15.9 tok/s | 352 ms |
| Async driver dispatch, pooled session buffers, GPU-side state reset | 18.1 tok/s | 560 ms |
| Block-shared E2M1 decode table in the routed experts | 19.2 tok/s | 503 ms |
| Chunk-wide sparse attention (warp-per-slot, no per-key barrier) | 21.0 tok/s | 472 ms |
| 128-token chunks through a shared-memory GEMM | **19.7 tok/s** | - |

The last row is the sparse-attention rewrite. It is one change with two effects: decode
gains ~10% because the attention kernel no longer spends nine barriers on every selected key, and
prefill stops degrading with prompt length, which is the larger win:

| prompt | prefill, before | prefill, after | TTFT, before | TTFT, after |
| ---: | ---: | ---: | ---: | ---: |
| 35 tokens | 67 tok/s | **74-76 tok/s** | 503 ms | **465-475 ms** |
| 343 tokens | ~64 tok/s | **71-72 tok/s** | ~4,000 ms | **4,766-4,796 ms** |
| 679 tokens | 52 tok/s | **71 tok/s** | 12,965 ms | **9,520-9,550 ms** |

At 679 tokens that is **+37% prefill and 26% less TTFT**, and the flat 71-78 tok/s from 35 to 679
tokens is the point: the old curve fell 22% from the short prompt to the long one because the cost
was per selected key, and it is now per chunk. Decode on the same sweep is 19.5-21.0 tok/s.

The 128-token chunk then took prefill to **93-94 tok/s** from 175 tokens up, with TTFT at 679
tokens at **7,192 ms** against 12,965 ms at the start of this work. Decode is unchanged at
18.7-19.7 tok/s: the chunk is a prefill property, and the GEMM hands back to the GEMV below
sixty-four tokens.

**The profile that found them** (`SPARKFLASH_QWEN4_PROF=1`, which stamps a stream drain and
charges the elapsed time to a stage name; it costs ~6 ms across ~1,400 marks, so a benchmark runs
without it). At 64.61 ms/token:

| stage | ms | share | what it was |
| :-- | ---: | ---: | :-- |
| `attn_inject` | 29.21 | 45.2% | attention projections. 143 MB a layer in 608 us is 235 GB/s, i.e. 93% of the loop's best observed rate. `q4e_qsa_select` is *not* it at this context: at 32 tokens there are 8 blocks to score, so parallelising it measured neutral, and its win is at long context |
| `experts` | 7.79 | 12.1% | 172 GB/s; the E2M1 decode is ~4 ALU ops per weight, which is what block-scaled tensor-core MMA removes |
| `hyper_down` + `hyper_up` | 7.49 | 11.6% | 178 GB/s over 97 launches - the down projection is only 320 rows, so it has 320 warps carrying the whole 10,240-wide row |
| `lm_head` | 4.75 | 7.4% | 267 GB/s, at the memory system's limit |
| `shared` | 3.36 | 5.2% | 139 GB/s |
| `router` | 1.38 | 2.1% | 512 rows at 91 GB/s |

### The E2M1 decode, and why the table is in shared memory

`experts` was 12.1% of the loop at 172 GB/s while `lm_head`, which reads the same kind of
DRAM stream, reached 267 GB/s - so the routed experts were not waiting on the memory system.
The per-weight cost was the scalar decoder: `spark_fp4_e2m1_decode` builds its value with a
shift, a mask, a 32-bit table shift, an int-to-float, a `0.5f` scale and a sign-bit insert,
about thirteen issue slots for one multiply. A block touch 4.9M packed weights per
(layer, slot), so that arithmetic, not bytes, set the stage.

The sixteen E2M1 codes are exactly representable, so a table of the decoded values is
lossless. `q4e_e2m1_table_fill` builds it once per block into shared memory, next to the
activation staging and folded into the `__syncthreads()` already there, and each nibble then
costs a shift, a mask, one shared load and the multiply. Decode 17.9 -> 19.2 tok/s (52 -> 56 ms/token), prefill
59 -> 67 tok/s and TTFT 576 -> 503 ms at a 34-token prompt, 53 -> 62-66 tok/s and TTFT
4803 -> 3909-4117 ms at 256 tokens, 46 -> 52-53 tok/s and TTFT 14573 -> 12965 ms at 672.
The two-prompt spread is run-to-run variance on the node, not two builds.

The table is in **shared memory, not a warp-shuffle register table**, and that is not a
preference. A register table indexed by `__shfl_sync(mask, value, nibble)` is one instruction
too, and it was tried first: it faults. The row-dot loops start each lane at its own chunk
offset (`for (int chunk = lane; chunk < chunks; chunk += 32)`), so on a row that is not a
whole number of 32-chunk strides - 640 and 2,560 are not - the lanes leave the loop at
different counts. A lane-indexed shuffle then names lanes that are no longer participating,
`__shfl_sync` with a full mask over a partial warp is undefined, and the first request after
the change died with `cuMemAlloc failed: 700` because the context was already poisoned. Every
parity test passed on the workstation, because the reduced geometries all round to a uniform
trip count; the released 640-wide expert rows are what expose it. Shared memory has no
convergence requirement, so the same saving is safe.

Batching the prefill chunk wider was measured too. `Q4E_MAX_BATCH` 8 -> 16 is neutral
(prefill 65.5 -> 65.7 tok/s at 256 tokens, 52.6 -> 51.5 at 672) and it costs 44 more
registers in `q4e_hyper_down` (54 -> 98, i.e. half the occupancy) with no spills anywhere,
so the chunk stays at eight. The neutrality is the informative part: doubling the tokens in a
chunk doubles its wall clock, which says the chunk is *not* bound by the one weight read it
amortises but by per-token work - the same work decode pays.

Programmatic dependent launch was measured and is *not* one of these. The driver already
supports it (`cuLaunchKernelEx` with programmatic stream serialization, `opt_`-gated) and
`kernels/opt_pdl_sm121.cuh` is the opt-in, so all 23 qwen kernels were wired to it and the
gate widened to `q4e_`. It came out slightly *worse* - `attn_inject` 29.21 -> 29.90 ms,
`experts` 7.79 -> 7.91 - because the host already runs ~1,400 launches ahead of the GPU, so
there is no launch latency left to overlap; only the two `griddepcontrol` instructions were
added. Reverted - so the loop sits at 174 GB/s rather than `lm_head`'s 267 for some reason
other than launch gaps between kernels.

More warps is not it either. `q4e_hyper_down` is the strongest candidate for that theory -
320 rows against a 10,240-wide row, 146 GB/s, and `lm_head` reaches 267 with 248,320 rows -
so it was rewritten to split each row across eight warps with the slices summed in shared
memory. Wall clock 16.23 against 16.36 tok/s: neutral, and reverted rather than carried as
complexity. The remaining explanation is that a BF16 row-dot reads *eight activation floats
per sixteen bytes of weights*, so a third of what the kernel moves is the activation, and the
ratio is set by the dtype rather than the shape. That one is untested.

Both experiments share a methodological lesson worth recording: a profiled run is not
comparable to another one. The same binary produced 64.61 and 82.99 ms of profile total, with
every stage 15-45% worse and `ple_host` at 11.39 ms, while the wall-clock benchmark in the
same script read 16.1 and 16.2 tok/s on both. Compare *wall clock* across runs, and use the
profile only for the shape within a single run.

The *total* is the least reliable number in this table: `ple_host` moves between 0.95 and 7.05 ms
from one run to the next, because it is a `pread` of sixteen random rows out of the 51.2 GB
n-gram table and the page-cache state decides it. Compare stages, not totals.

Five bugs accounted for most of the 78x between 0.20 and 15.9 tok/s, and none of them was
arithmetic. Three of the five are the same defect - a kernel shaped so that almost none of the
machine can work on it - and two are the same defect on the host side:

* **The GEMV was one 256-thread block per row with a block-wide tree reduction.** At vocabulary
  width that is 248,320 blocks and it streams at 286 GB/s, but a 512-row projection is five load
  iterations per thread around eight `__syncthreads`, and it measured **2.6 GB/s** - a hundred
  times off the same kernel's steady state. One warp per row with 16-byte loads and a shuffle
  reduction fixes the shape, and the 40-chunk row also keeps four loads in flight.
* **Two `copy_dtoh` calls per layer, each draining the stream.** The shared expert's gate and
  the indexer's selected-token count are one scalar each, and the host read both back mid-step:
  48 + 12 host round trips per token, each of which waits for everything queued. `q4e_moe_combine`
  and `q4e_attn` now take the device pointer and read it themselves. `shared` 4.62 -> 3.39 ms,
  `attn_inject` 32.65 -> 30.29 ms, 68.44 -> 62.46 ms of profile total.
* **The router's top-k was serial on thread 0**, scanning all 512 experts and re-scanning the
  slots already taken for each of ten slots: 28,000 dependent shared-memory loads a layer, ~1 ms,
  32% of a token. It is now ten rounds of a block-wide best-of that keeps the reference's order
  (highest probability, lowest index on a tie) as a total order, with a picked expert struck out
  at -1.0 instead of being rescanned. 48.09 ms -> 1.41 ms.
* **The hyper-connection injection used four lanes of 256.** `for (stream = threadIdx.x; ...)`
  walked all 10,240 columns serially inside each of four threads, twice a layer: ~400 us a call,
  for 80 KiB of 2-byte loads. One warp per stream with 16-byte chunks: 44 ms -> 2.55 ms.
* **Every expert read its activation from global memory, two scalars per packed byte**, with
  adjacent lanes a 64-column stride apart - a 32-way gather per load. Staging the activation in
  shared memory is what let the weight traffic matter instead. 16.13 ms -> 9.91 ms.

The MTP head is now served as a speculative drafter (`device.rs`, no separate module: the
checkpoint's one MTP layer is a sparse-attention + MoE block with its own hyper-connections, so
it runs through the same kernels). A step runs the main model over `[token, draft]`, emits the
model's own sample at the first position, and answers the following step from the second row's
logits when the caller feeds the token the model chose. A rejected draft leaves the recurrence
where the step found it, as the DFlash 2 verify does.

Measured: drafts agree with the model's own sample 38% of the time (85 of 222 on the benchmark
prompts), decode 19.8-22.5 tok/s against 18.9-20.2 without it. The pair's second row costs about
0.67 of a standalone row, so two tokens for 1.34 rows is nearly a wash at this agreement rate -
the head is correct and neutral, and it is the agreement that has to improve before a step that
verifies more than one token pays. The traps are 128x128 FP8 block scales, the whole-row pre-fc
norm, and the missing `mtp.norm`/`mtp.lm_head`.

On the eight chat and completion prompts in [Where it stands](#where-it-stands) the same head has 72% of its drafts
accepted and decodes at 28.3 tok/s against 21.5 without it; the 38% above was measured on the
raw-completion benchmark prompts.

## Serving

The pack is served through `sparkflash serve`'s generic host path
(`sparkflash_serve::HostDriver` over `ModelBackend`), not through the V4.1 engine:

    sparkflash serve --model nvidia/Qwen3.8-Flash-Next-NVFP4 \
        --context 262144 --bind 127.0.0.1:8000

With no `--checkpoint`, the server resolves the canonical Hugging Face cache snapshot for the
model id; with no `--tokenizer`, it uses that snapshot's `tokenizer.json`. `--checkpoint` and
`--tokenizer` still override the cache.

`--model` takes either the catalogue id or the weights id (`qwen3.8-flash-next-nvfp4`). The
V4.1-only flags (`--engine`, `--engram-dir`, `--continuous-batching`, the `--tp4-*` group and the
device-token window) are refused alongside it rather than silently ignored. `--synthetic` writes
the pack's own tiny fixture to a temporary directory and serves it with a byte codec: that is a
runnable demo of the same path - router, queue, streaming, sampling - on a workstation with no
132.6 GB checkpoint, and it says so at startup.

`--concurrency N` runs N host-driver sessions. The default is 1, the historical serial path. Each
session owns one non-blocking CUDA stream and its own KV, recurrent and scratch buffers; the
77.8 GiB of weights and the kernel handles are shared read-only, so the sessions' kernels are
scheduled independently.

Concurrency alone does not add throughput. A decode step reads about 10.9 GB of weights for one
row, so two sessions on their own streams read it twice: measured on this node with four 256-token
prompts, 29.9 tok/s aggregate at one session with the DFlash 2 drafter, 32.8 at two and 32.4 at
four (1.10x), and 1.03x on 32-token greedy requests. The DRAM is the ceiling and the streams share
no bytes.

`--fuse-batch` (requires `--concurrency` above 1, `--model` only) steps the sessions as one batch
instead. Every member's row goes through one pass over the weights: the row-independent stages -
the embedding, the hyper connections, the projections, the router, the shared expert, the head -
run once over all the rows, and only the per-sequence stages run once per row on that row's slice:
the n-gram windows and PLE convolution, the KV append, the indexer and the attention, the Gated
DeltaNet convolution and recurrence. Each session keeps its own caches, recurrent state and
sampler; only the per-row activation scratch is shared.

| concurrency | a session a worker | `--fuse-batch` |
| :-- | --: | --: |
| 1 | 29.9 tok/s | 20.4 tok/s |
| 2 | 32.8 | 35.4 |
| 4 | 32.4 | 42.5 |

The aggregate across the concurrent requests, 256 generated tokens each, four prompts; the
four-session fused figure is the median of five runs (41.1 to 45.1). The fused column at one member
is lower than the unbatched one because a fused session does not speculate: the 29.9 above is the DFlash 2 drafter, and a plain one-row step measures 20.4. Batching
recovers speculating throughput at two rows and reaches 1.30x the best unbatched number at four,
where the dense stack is read once for four rows and each row adds only its routed experts. A batch
is capped by the engine's capacity, which is `--concurrency`, and at four members by this node's
memory at a 16k context: `--concurrency 8` was killed during start-up with the checkpoint resident.

A fused member can verify its drafts inside the batch: its token plus the drafts it proposed are
that member's rows of the same pass, so a verify is not a second read of the weights.
`SPARKFLASH_QWEN4_FUSE_SPEC=1` turns it on and it is exact - the batch gate runs it, at one, two and
four members, and requires every member's tokens to match a lone member's.

It also took two bugs out of the way, both of which made the drafter look useless rather than
broken. The depth policy would not verify anything until it had seen an acceptance, and it only
sees acceptances for drafts it verifies, so a fresh policy sat at zero drafts forever
(`DepthPolicy::depth` now verifies one draft until a bin has been sampled). And the fused engine
answered every `decode` with a forward, ignoring the answers the last verify had already queued for
the positions it accepted, so a batch paid eight rows to recompute a token it had - that alone was
14.6 tok/s against 38.1 once the queue is drained.

One concurrent session, 256-token prompts: 39.3 tok/s with the drafter and no batching, 38.1 with
the batch and the in-batch verify - the batch costs a lone verifying session nothing, which is what
a batch of one should cost. Two members verifying do not measure better: 43 to 63 tok/s aggregate
against 65 for the single member.

The profile says why, and it is the batch's occupancy rather than the drafter. A step of two rows
costs 60-68 ms and a step of six 87 ms, so a row past the first costs about 6 ms and the rest is the
weights read once; and under two verifying members the profiled steps are two and three rows, not
sixteen. Queued answers keep the members out of phase - a member whose drafts were accepted is
answered for free while the other steps alone - so the batch rarely holds both of them, and a
one-member step verifying one draft pays the whole weight read for a couple of tokens. The policy
reacts correctly by verifying less, and the pair ends up at one stream's rate.

That is a property of speculative decoding, not of the batch: with drafts, two streams desynchronise
and the occasions when their verifies align are few. It is why a fused *decode* batch gains 1.7x
while a fused *verify* batch does not. Making it pay would mean keeping the members in step, which
means not spending the free answers the drafter just earned - a trade that is not worth its tokens.

A prefill has a thread of its own - one a member, up to the session count. Its prompt is one
sequence's chunks, not a row of a batch, so the sequence leaves the pool for as long as the prompt
runs and the other members keep decoding: against a 2,333-token prompt that took 3.28 s to prefill,
a 120-token completion that took 5.99 s alone took 8.08 s beside it. It lost two thirds of the
prompt's duration to bandwidth contention, where a blocked engine would have lost all of it.

Prefills no longer queue behind each other either. A 60-token prompt submitted behind that
2,333-token one took 2.06 s against 0.57 s alone, where a single prefill worker would have made it
wait the prompt's whole 3.2 s first; and two equal 1,193-token prompts prefill in 2.78 s together
against 1.45 s each alone, which is the shape of the trade - the second prompt starts at once, and
both then share the memory system.

The gate is exactness, not closeness: `a_fused_batch_emits_what_its_members_emit_alone` runs the
tiny fixture three ways - a plain session, a fused engine with one member stepping, and a fused
engine with two members stepping from two threads - and requires the prefill logits row to match
(bit-identical on this build) and every generated token to match, member by member. The startup
guard computes the session caches from `--context` and refuses a worker count over the KV budget -
10 GiB, the slice this laptop can hold beside the weights and the paging n-gram table, which admits
one full 262,144-token window and needs a shorter `--context` for a second session.
`--kv-budget-gib` raises it on a node with more room. The tiny-fixture
test `two_concurrent_requests_are_both_served` drives two requests through the real router and checks
that both complete; `SPARKFLASH_QWEN4_BATCH_TRACE` names each fused step's members on stderr, which is
how a run proves its requests were actually fused.

### The stream a launch runs on

Moving each session onto its own CUDA stream also moved the ordering rules: work on a session's
stream is not ordered against the legacy default stream, so anything a forward queues there races
the layers that read it. Two such sites were found and fixed after the stream change:

* the forward's embedding launches ran on the default stream while every other launch of the same
  forward ran on the session's. Under greedy decoding three identical requests produced three
  different answers, and at a low rate an immediate end-of-generation token with no completion at
  all, because the layers occasionally read a row before its embedding had been written.
  `forward_batch` now launches the embedding through the session's `launch`, and the same three
  requests are byte-identical again;
* the DFlash 2 drafter read its drafted block back with a plain
  `cuMemcpyDtoH_v2`, which is ordered against the default stream only, so the host read the
  previous step's block while the selector was still queued. Every draft was stale and the target
  rejected almost all of them: acceptance 1 of 200 drafts (0.5%) and 16.8 tok/s of decode. With
  the readback on the session's stream the same prompt accepts 52-67% of its drafts and decodes at
  34-42 tok/s (36.4 tok/s over four prompts of 256 tokens), against 25.5 tok/s for the MTP head on
  the same four. The probe path had the same shape and uses the stream-scoped copy too, and loading
  now synchronises after the E4M3 weight quantisation, which runs on the default stream before any
  session exists.

Three pieces make that work:

* [codec] is the ChatML prompt seam: the released tool catalogue and its `<tool_call>` XML call
  format, assistant `reasoning_content`/`tool_calls` continuation, the merged `<tool_response>`
  turn, the `<think>` generation prompt and `enable_thinking: false` closing the block. It
  validates that the tokenizer's `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<|image_pad|>`,
  `<|vision_start|>` and `<|vision_end|>` sit at the ids the configuration names. An image part
  renders as the released template's one `<|image_pad|>` placeholder, which the vision front end
  expands into the packed span (vision start, the merged rows, vision end).
* [sampling] applies repetition/presence/frequency penalties, temperature, top-k, top-p and the
  draw on the host, over the logits row the step already copied back. The order is vLLM's, greedy
  still applies penalties, and a greedy request consumes no random number. As in vLLM, the
  repetition penalty covers the prompt and the generated tokens, while the presence and frequency
  penalties count only the generated tokens. The arm therefore serves `temperature > 0` honestly
  instead of always answering greedily.
* the session stops at an end-of-generation id (`248044` or `248046`), returning no next token,
  which is what the host driver reads as a finished answer.

`sparkflash-api`'s `QwenResponseEngine` splits the raw decode stream back into the OpenAI
channels: text before `</think>` becomes `reasoning_content`, `<tool_call>` XML becomes
`tool_calls` with an OpenAI `arguments` JSON object, and a streamed answer finishes with
`finish_reason: "tool_calls"` when it carried a call. The adapter is installed by the Qwen serving
arm only.

The host arm has no prefix cache: every request prefills its whole prompt, so the OpenAI usage
block reports `cached_tokens: 0`, `prompt_cache_hit_tokens: 0`, and
`prompt_cache_miss_tokens: prompt_tokens`, and the extended KV stats report the whole prompt as
the uncached suffix. That is the honest answer rather than a zeroed hit rate.

Decode used to fall with context because `q4e_qsa_pick` scanned all 512 kept blocks for every
block past the indexer budget. The kept set is a min-heap now, so each later block costs
`O(log budget)`; greedy decode on this node measured 23.9 tok/s at a 3.6k-token prompt, 22.9 at
8.0k, and 19.7 at 28.5k, against roughly 3-5 tok/s before the change at those lengths.

What the served arm does **not** claim yet: images. `GET /v1/models` reports `tools: true`,
`thinking: true`, `multimodal: false`, because the device prefill does not consume the vision
rows. The pack now carries a Qwen-style preparation (`smart_resize` to a multiple of 32,
normalise with mean/std 0.5, patchify in 2x2-merge-block order), the packed image span, and a
host f32 reference tower with axial 2-D RoPE and bilinear learned-position resampling
(`crates/sparkflash-model-qwen4-exp/src/vision.rs`). It is not yet the released processor bit for
bit: alpha is dropped as Pillow does, but resampling is the shared OpenCV-style bicubic rather
than torchvision's PIL bicubic. The reference tower also caps an image at 576 merged tokens (the
48 x 48 learned-position grid) because its score matrix is `patches x patches`; the released
16-Mi-pixel ceiling would need a 17 GiB matrix. Its unit tests pin the resize plan, the endpoint
taps, the packed span length, the token cap and the tower's row count; the released checkpoint's
weights load through `Qwen4Vision::load`. The remaining step is the device prefill: writing the
merged rows over the placeholder embeddings and using the image's 3-D M-RoPE positions in the
sparse-attention and indexer rotations. Until that lands, advertising vision would be a promise
the device arm cannot keep.

## Memory

At TP=1 and the default 262,144-token window the QSA K/V caches are
`12 x 262,144 x 2 x 256 x 4 = 6.4 GiB`, the indexer keys `12 x 262,144 x 128 x 4 = 1.6 GiB`, and
the GDN recurrent state `36 x 48 x 128 x 128 x 4 = 113 MiB`. The steady state is about 80 GiB beside a
128 GiB node. Each extra `--concurrency` worker allocates another cache set, which is why the
serve path refuses a worker count whose caches exceed the 10 GiB slice; at the full window that is
one session, and two fit the same slice at half the window. Upload peaks at one tensor at a time
because the loader reads it, copies it and drops the host copy; a bulk streaming loader is
throughput work, not a correctness question.

## Verification

    cargo test -p sparkflash-model-qwen4-exp     # 45 unit tests, no GPU needed
    cargo test -p sparkflash-model-qwen4-exp --test checkpoint_roundtrip  # no GPU needed
    SPARKFLASH_CUDA_ARCH=sm_120 cargo test -p sparkflash-model-qwen4-exp --test device_parity
    SPARKFLASH_CUDA_ARCH=sm_120 cargo test -p sparkflash-model-qwen4-exp --test serve_tiny
    cargo run -p sparkflash-model-qwen4-exp --example device_self_test   # needs nvcc + a GPU
    cargo test --release -p sparkflash-model-qwen4-exp --test verify_rows    # needs a GPU
    cargo test --release -p sparkflash-model-qwen4-exp --test dflash_kernel  # needs a GPU

`verify_rows` holds every kernel a verify forward runs - the batched BF16 GEMV, the tiled wide
GEMVs, the hyper passes, the snapshot writers, the recurrence replay - to the bits the same row
gets alone, which is what keeps a drafter's greedy text equal to the plain decode's, and checks the
drafter's E4M3 GEMV against the codes and scales its quantiser writes. `dflash_kernel` checks the
drafter's vocabulary top-k, candidate selector, block convolution and ring attention against
host references. Three `#[ignore]` benches
time kernels at the released shapes without a checkpoint: `decode_bench` (the decode step's
kernels at one token and a verify's eight), `expert_bench` and `gemm_bench`
(`-- --ignored --nocapture`).

The tests cover the released config and its invariants, the full 299,545-name plan and its byte
identity (132,639,846,394 total; 129,043,957,786 text+PLE; 51,200,246,042 PLE table;
77,843,711,744 device-resident), the NVFP4 and FP8 decoders, the Engram-derived hash layout and
five pinned reference rows, prefill/decode hash identity, the paged table's cache and sharding,
and the tiny reference forward (finite logits, prefix causality, determinism, PLE contribution,
context and vocabulary refusals).

The checkpoint round trip is the gate that runs without a GPU: `write_synthetic_checkpoint` writes
a complete shard-shaped checkpoint for a configuration (every served tensor, every released dtype,
all 128 PLE shards, config and index), and the test reopens it, checks the plan's conformance
against what was written, loads it through the host loader, runs the reference forward, and asserts
the logits are finite, non-degenerate and deterministic. `write_synthetic_checkpoint_sharded`
splits the same plan across several files, which is the shape the released checkpoint actually has:
the test pins that the multi-shard index, its per-file headers and the same seed load to a
byte-identical model. The same suite pins the released configuration against every kernel bound
(the bottleneck, the expert width, the block budget, the selected-token buffer) and checks that a
configuration past one is refused at open rather than silently producing a short answer. It exercises BF16, FP8 E4M3 and NVFP4 with
both per-tensor scales, so no decoding path is untested.

That fixture is also what makes the device gate cheap enough to run on a workstation: the tiny
checkpoint uploads in well under a second, and the parity test compares prefill and decode logits
against the same bytes the host read.

`tests/serve_tiny.rs` is the serving gate: the pack's device backend behind the generic host
driver and the real OpenAI router, driven by ordinary `/v1/models`, `/v1/completions`,
`/v1/chat/completions` and streaming requests. It found one bug that only a multi-threaded
serving process can hit - the CUDA context is current on the thread that opened the checkpoint,
not on the driver's worker thread, so every session failed with CUDA_ERROR_INVALID_CONTEXT until
the backend bound the context to the thread that opens it.

## Deploying on a GB10 node

The reference target is a single NVIDIA GB10 ("RTX Spark N1X", compute capability 12.1) on
aarch64 Gentoo with the 615.78.08 driver, whose user-mode component reports CUDA 13.4. The whole
deployment is three steps: a toolkit that can emit `sm_121a`, a checkpoint copy, and the binary.

The toolkit comes from Portage, because the node has no CUDA at all:

    emerge --oneshot dev-util/nvidia-cuda-toolkit   # 13.4.1; accepts the NVIDIA-CUDA licence

`dev-util/nvidia-cuda-toolkit` is licence-masked, so `/etc/portage/package.license` needs
`dev-util/nvidia-cuda-toolkit NVIDIA-CUDA` before the merge. The resulting `nvcc` lives at
`/opt/cuda/bin/nvcc`, which is not on `PATH` by default.

CUDA 13.4 refuses the gcc 16 this distribution ships as a host compiler, and the node has no
older gcc. `build.rs` already passes `-allow-unsupported-compiler` (and `-ccbin` when it finds
g++-14/13/12), which is enough here: the kernel is plain CUDA C, so only the host-compiler version
gate stands in the way.

    export SPARKFLASH_NVCC=/opt/cuda/bin/nvcc
    export SPARKFLASH_CUDA_ARCH=sm_121a
    cargo build --release --bin sparkflash

`cargo` does not track "nvcc appeared" as an input, so a binary built before the toolkit landed
keeps its cubin-less `build.rs` result. Touch `crates/sparkflash-model-qwen4-exp/build.rs` (or set
`SPARKFLASH_NVCC` from the start) to force the kernel image into the next build.

The checkpoint is 132.6 GB over eleven files - ten text shards plus `model-fp8-mtp-ple.safetensors`,
which carries the 128 PLE table shards - and is fetched straight from the Hub into the canonical
Hugging Face cache, since it is a NVIDIA-published repository:

    hf download nvidia/Qwen3.8-Flash-Next-NVFP4

The cache entry is `models--nvidia--Qwen3.8-Flash-Next-NVFP4`; `snapshots/<refs/main>` is the
checkpoint directory every tool should use. Do not copy it into a home-directory path: the web
console and the serve command resolve the canonical snapshot themselves.

Then, on the node:

    ./target/release/sparkflash serve \
        --model nvidia/Qwen3.8-Flash-Next-NVFP4 \
        --context 262144 --bind 0.0.0.0:8000

With no `--checkpoint`, the server uses that cache snapshot; with no `--tokenizer`, it uses the
snapshot's `tokenizer.json`. Both flags still override the cache for a custom copy.

`--context 64` (or no `--context` with `--synthetic`) is what the tiny fixture serves, because the
fixture's own configuration has a 64-token window: passing the released 262,144 to it is refused at
open, which is the context guard doing its job rather than a fixture bug.

The startup log reports the device, the resolved pack and the window:

    device 0: NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU) of 1 (total 118815 MiB, MemAvailable 112858 MiB)
    sparkflash serve: nvidia/Qwen3.8-Flash-Next-NVFP4 (qwen3.8-flash-next-nvfp4) at TP=1 on ~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/<refs/main>, context 262144

One deployment note belongs to the network rather than the model: the node's address is a DHCP
lease, and moving it between interfaces changes the lease. Serving on `0.0.0.0` keeps the endpoint
valid across that change; the client just needs the new address.

**The node does not survive a full-core compile while serving.** The 77.8 GiB device allocation is
`shmem` on unified memory, not HBM behind a separate pool, so it competes with the page cache and
with every gcc on the box. A twenty-way `cc1plus` build alongside the server took MemAvailable to
2 GiB of 116 and the kernel killed the server:

    NVRM: GPU0 nvCheckOkFailedNoLog: Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal
    oom-kill: task=sparkflash ... shmem-rss:87746432kB

`vm.min_free_kbytes` (raised to 2 GiB before the upload, which is what keeps the node reachable
while 77.8 GiB goes in) is not protection: the ``OOM`` killer scores by anonymous and shmem
resident size, and the server is the largest such task by construction. Two consequences for
measurement as well as for serving: any compile, build or page-cache-heavy job moves decode
throughput by 30-40% (15.5 tok/s idle against 8.3 contended) because both the PLE `pread` and the
memory system are shared, and a benchmark taken during one is not comparable to one taken idle.

