# The Phase 1 per-layer boundary table: format, host capture, and what the perturbation test proves

Phase 1's first two exit criteria are a per-layer boundary table and a perturbation test that makes
the table able to fail ([roadmap-to-serving.md](roadmap-to-serving.md) section 4.5, items 1 and 2).
This document is the format the two halves have to agree on: one table per step, eighteen named
tensors per layer row, two dtypes, one byte order, so a device run and the host reference can be
compared tensor by tensor without a translation layer. It also states each tensor's **comparison
class**, records the host-side perturbation evidence, and reports one finding about the criterion as
literally written.

The host half is committed: `crates/sparkflash-engine/src/forward/boundary.rs` builds and compares
the tables, `forward::InferenceEngine::prefill_observed`/`decode_observed` fill one per step, and
`crates/sparkflash-engine/tests/boundary.rs` is the perturbation test. The host dumper and reader of
the device emission format are committed too: `BoundaryTable::write_v1`/`read_v1`, with
`examples/boundary_dump.rs` and `examples/boundary_diff.rs`, and section 11 records the first
released-checkpoint dump. The device half — the engine
emitting the same table and the bars for device-vs-reference — is the other half of Phase 1 and is
not claimed here. Every measurement in this document is host-only; no device number appears in it.

## 1. What one table is

One step of the reference forward is one table. A table carries the step's header and one row per
layer that ran:

| header field | meaning | source |
| :--- | :--- | :--- |
| start position | the absolute position of the step's first token | `BoundaryTable::start_position` |
| consumed prefix | the token ids the sequence had already consumed before this step, in order | `BoundaryTable::consumed_prefix` |
| tokens | the token ids this step consumed, in order | `BoundaryTable::tokens` |
| MTP anchor | the backbone-sampled token the MTP stages consume as their first draft token, or `None` when no MTP row ran | `BoundaryTable::mtp_anchor` |
| rows | one per layer that ran, in ascending layer id | `BoundaryTable::rows` |

The header identifies the **history** the step ran over, not only its position. A decode table
depends on the consumed prefix through the window ring and, for compressor layers, the compressed
KV and indexer state those tokens built; an MTP row additionally depends on the anchor, which is
not a boundary tensor and cannot be recovered from the row. The prefix's length is the start
position for a sequence, but the two are compared independently so a capture bug in either field is
caught; the anchor is a separate field because it cannot be recovered from the rows. A row belongs
to one layer: a backbone layer `0..n_layers-1`, or an MTP stage `n_layers + stage`.

Rows are only produced when a capture was asked for: `prefill` and `decode`
call the same `run` with no capture and behave exactly as before, while `prefill_observed` and
`decode_observed` return `ObservedStep { output, boundary }` and the default entry points'
`logits`, `next_token` and `consumed` are unchanged (asserted by
`the_observed_step_is_the_plain_step_and_two_runs_capture_identically`). The default pass allocates
no boundary buffer and the MTP stage buffers exist only when a capture does
(`forward/mod.rs` `run`, `forward/mtp.rs` `dspark_forward_observed`).

**Backbone rows.** Exactly `n_layers` rows, layer ids `0` to `n_layers - 1`, in that order. For the
released configuration that is the forty rows of criterion 1.

**MTP rows.** One row per DSpark stage that this engine actually ran, at layer id `n_layers + stage`.
A first step has backbone rows only because the host engine skips the MTP pass entirely at
`start_pos == 0` (`forward/mod.rs` `run`, the `main_hidden.is_empty() || start_pos == 0` branch).
That is the **engine/capture contract**, not a statement that the released reference does no MTP work
on a first step: the reference differs in exactly that respect. `forward_spec` returns no draft at
`start_pos == 0` (model.py:1123-1125, 1274-1282), but each stage's attention still runs its prefill
branch there to seed the stage's window KV cache
([../engine/01-forward-order.md](../engine/01-forward-order.md) sections 3.8-3.9: "attention seeds the
cache, returns `x`; `forward_spec` returns `None`"; [../architecture/05-dspark.md](../architecture/05-dspark.md):86-97).
The engine also **skips that start-0 stage attention seeding entirely**, so its MTP stage window
rings hold nothing from a first step; the reference's rings hold the seeded main rows. That is a
history difference, not a row-count one: a device harness that reproduces the reference's start-0
seeding and then diffs its MTP rows on the following decode step against this engine's rows is
comparing two different cache histories. It must either seed the engine's way or compare those MTP
rows only after the histories are reconciled; the row counts below say nothing about cache contents.

So the counts below are what a capture of this engine emits. A first step's table has exactly
`n_layers` rows (40 for the released configuration) and a decode step after a prefill has
`n_layers + n_mtp_layers` rows (43 for 40 + 3), at layer ids `n_layers + stage`. A device harness
that also reproduces the reference's KV-seeding stage passes on a first step must decide explicitly
whether those passes emit rows; a run that captures a decode step must emit all
`n_layers + n_mtp_layers` rows or its row count is wrong. The stage row has the same eighteen
tensors with `positions` equal to the stage's draft block length (`config.dspark_block_size`); those
rows after the backbone are not part of criterion 1's "forty rows".

The table is **one step**, not a sequence. Two tables compare only if all four step-identity
fields agree - start position, consumed prefix, chunk tokens and MTP anchor - and the row/layer
layout agrees. The host diff refuses each mismatch by name rather than comparing different steps
(`BoundaryTable::diff`). The same-token/different-start and same-start/different-token cases are
pinned by `a_diff_of_the_same_token_at_a_different_start_is_refused` and
`a_diff_of_a_different_token_at_the_same_start_is_refused`; the different-history case (same start
and token, different consumed prefix) by
`a_diff_of_the_same_decode_header_over_a_different_prefix_is_refused`; the anchor case by
`a_diff_of_the_same_header_over_a_different_mtp_anchor_is_refused`; the row-count, layer-id,
tensor-count, tensor-name/shape and element-format refusals have unit tests in
`forward/boundary.rs`. The older composed refusal test
`a_diff_between_different_steps_or_layer_sets_is_refused` still covers the row-count and
different-token paths at table level.

## 2. The eighteen tensors of one row

The row is the block of [../engine/01-forward-order.md](../engine/01-forward-order.md) section 3.2
(`Block.forward`, model.py:968-994) cut at the points a device replay can also record, in the order
the forward computes them. `L` is the row's layer id; `positions` is the number of tokens in the
step; `dim`, `hc_mult` are the configuration's; `dtype` and `shape` are what the row stores.

| # | tensor name | layer id | dtype | shape | what it is | capture point |
| ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `residual_in` | L | f32 | `[positions, hc_mult, dim]` | the stream the block starts from | block entry, before `hc_mixes` (model.py:968) |
| 2 | `hc_in_pre` | L | f32 | `[positions, hc_mult]` | the input mix the block starts from: the previous block's `ffn_pre` (identity at layer 0) | block entry (model.py:983, 1258-1260) |
| 3 | `hc_attn_pre` | L | f32 | `[positions, hc_mult]` | attention predictor's pre set | after `hc_mixes` (model.py:982) |
| 4 | `hc_attn_post` | L | f32 | `[positions, hc_mult]` | attention predictor's post set | after `hc_mixes` (model.py:982) |
| 5 | `hc_attn_comb` | L | f32 | `[positions, hc_mult, hc_mult]` | attention predictor's comb set, `[source copy, destination copy]` | after `hc_mixes` (model.py:982) |
| 6 | `pre_attention_hidden` | L | f32 | `[positions, dim]` | the row's attention block f32 input: `attn_norm(hc_pre(...))`; for an MTP row this is the draft block's own input, and the stage additionally consumes `main_x` and the header's anchor (not recorded in this tensor) | between model.py:984 and 985 |
| 7 | `pre_attention_xhat` | L | u8 | `[positions, dim]` | E4M3 codes of #6, the quantised block input | the activation quantiser of model.py:187-205 |
| 8 | `pre_attention_scales` | L | u8 | `[positions, dim / 32]` | E8M0 scale per 32-element block of #6 | same |
| 9 | `attention_out` | L | f32 | `[positions, dim]` | `Attention.forward`'s output, `wo_b(wo_a(...))`, before `hc_post` | after model.py:985 |
| 10 | `post_attention_hidden` | L | f32 | `[positions, hc_mult, dim]` | the stream after the attention `hc_post` | after model.py:986 |
| 11 | `hc_ffn_pre` | L | f32 | `[positions, hc_mult]` | FFN predictor's pre set | after `hc_mixes` (model.py:989) |
| 12 | `hc_ffn_post` | L | f32 | `[positions, hc_mult]` | FFN predictor's post set | after `hc_mixes` (model.py:989) |
| 13 | `hc_ffn_comb` | L | f32 | `[positions, hc_mult, hc_mult]` | FFN predictor's comb set, `[source copy, destination copy]` | after `hc_mixes` (model.py:989) |
| 14 | `pre_ffn_hidden` | L | f32 | `[positions, dim]` | the FFN block's f32 input: `ffn_norm(hc_pre(attn_pre, ...))` | between model.py:991 and 992 |
| 15 | `pre_ffn_xhat` | L | u8 | `[positions, dim]` | E4M3 codes of #14, the quantised block input | the activation quantiser of model.py:187-205 |
| 16 | `pre_ffn_scales` | L | u8 | `[positions, dim / 32]` | E8M0 scale per 32-element block of #14 | same |
| 17 | `ffn_out` | L | f32 | `[positions, dim]` | the MoE output, before `hc_post` | after model.py:992 |
| 18 | `residual_out` | L | f32 | `[positions, hc_mult, dim]` | the stream after the FFN `hc_post` = the next layer's `residual_in` | after model.py:993 |

The names are constants in `forward/boundary.rs` (`RESIDUAL_IN` ... `RESIDUAL_OUT`) and
`boundary::TENSOR_ORDER` is exactly this list; the tests assert a row carries all eighteen in this
order, so a device emitter and the host reference cannot silently disagree about what row 7 is.

Four details of the capture are deliberate:

* **The pre-attention and pre-FFN rows are the block input, not the norm output alone.** For a
  backbone row, #6 is the tensor the attention GEMM actually consumes (after `attn_norm`); #14 is
  the FFN's counterpart. The FFN's `hc_pre` reads **this block's attention `pre`**, not its own
  (model.py:990) — the shift that makes Single-Pass mHC observable — so `pre_ffn_hidden` and
  `hc_ffn_pre` are not computed from the same coefficient.
* **#7/#8 and #15/#16 are diagnostic for the device path.** They are the reference activation
  quantiser's E4M3 codes and E8M0 scales of #6/#14, recorded so the device FP8 GEMM can be compared
  code for code. The host attention and FFN consume the f32 #6/#14, not these bytes. Their
  exact-codes comparison class (section 5) is still meaningful — same f32 input must give the same
  codes — but a host attention result is not computed from them, and agreement of the codes does not
  by itself say the f32 stream agreed.
* **An MTP row's attention has inputs outside the eighteen tensors.** The stage's attention consumes
  `main_x` - `main_norm(main_proj(cat(main_hiddens)))` over the backbone taps (model.py:1130,
  1122-1126) - as well as the draft block's own #6. `main_x` is **not recorded** by this format (it
  is a per-step input derived from the backbone taps, not a boundary of the stage's block). The
  anchor *is* recorded, in the table header's `mtp_anchor`. A replay of an MTP row therefore needs
  both the header and `main_x` from somewhere else; section 7.3 scopes this explicitly.
* **`residual_out` is the row's last tensor and the next row's first.** The test
  `a_one_code_change_in_layer_17s_wq_b_flags_layer_17_and_no_other_layer` asserts
  `residual_out(L) == residual_in(L+1)` byte for byte for every consecutive backbone pair, which
  pins the table's row-to-row wiring.

## 3. Layout rules

**Positions are the outer dimension.** Every tensor is token-major: `[positions, ...]`, with one
token's data contiguous. A stream is `[positions, hc_mult, dim]` with the `hc_mult` copies of one
token's `dim` values contiguous and in copy order 0..hc_mult-1, which is the array the reference
carries as `[b, s, n, d]` (model.py:1258; [../architecture/02-mhc.md](../architecture/02-mhc.md)
section 1). A coefficient set is `[positions, hc_mult]`; a comb set is
`[positions, hc_mult, hc_mult]` with the **row** the source copy `j` and the **column** the
destination copy `k` - the released `comb[j][k]` orientation
([../architecture/02-mhc.md](../architecture/02-mhc.md) sections 3-4, model.py:962-966). The host
applies it exactly that way: `hc_post[k] = post[k] * y + sum_j comb[j][k] * residual[j]`
(`forward/mhc.rs` `hc_post`), so a destination column is summed over its sources and each
destination column sums to 1 up to the Sinkhorn eps. This is stated explicitly because it is easy to
get backwards: an earlier host revision applied the comb transposed, and the row-17 numbers in an
earlier capture were computed from that application. The numbers in section 6 were measured again
after the correction and supersede them.

**All indices are element indices, not byte indices** in the shapes above. The host table validates
that a tensor's element count equals the product of its shape when it is pushed
(`LayerBoundary::push`), so a writer cannot emit a shape that disagrees with its data.

**Byte order is little-endian for every multi-byte element**: f32 and u32 (the token ids in a
manifest) are IEEE-754 binary32 / unsigned 32-bit in little-endian order, which is the order
safetensors uses for the checkpoint and the order every host in this project is (`quant.rs`'s
`f32_from_le_bytes`, "raw binary32 in little-endian order ... byte assembly rather than
conversion"). The two byte tensors are one byte per element and have no order.

**E4M3 and E8M0 are the reference's formats, not a choice.**
`quantize_activation_fp8` is kernel.py:41-95 with block size 32 and `scale_dtype = ue8m0`
(kernel.py:98-124): the scale is `2^ceil(log2(amax / 448))` with `amax` floored at `1e-4`, each
element is clamped to +/-448 after division and rounded to the nearest E4M3 code (ties to even), and
the scale is stored as the E8M0 byte `c` meaning `2^(c - 127)` (`quant.rs` `e8m0_encode_ceil`,
`f8e8m0_to_f32`). E4M3 code `0x7f`/`0xff` (NaN) and E8M0 code `0xff` (NaN) never appear for a
finite block; the tests assert that. The device kernels implement this same ceil rule in
`kernels/m4_activation_quant_sm121.cuh`'s `spark_act_quant_scale_code` (D23): the checkpoint's
weight-grid rule `floor(log2(amax)) - element_emax` is a different rule, lives in a differently
named helper in `m4_common_sm121.cuh`, and is never used for an activation. The block is 32
**contiguous elements along `dim`**, one block per 32 channels of one token.

**`dim` must be a multiple of 32** for the quantised pair to exist at all: the reference asserts
`N % block_size == 0` rather than quantising a short block (kernel.py:108), and the observed entry
points refuse a configuration whose `dim` does not divide into blocks with
`EngineError::Shape`, naming the requirement, before any weight is read. The released `dim` is
5120, which divides; a configuration of, say, 100 would have no valid quantised block input and no
boundary table.

## 4. The device emission format

The host table is compared in memory; the device run must emit the same bytes in a file. This is the
contract, stated in the terms the device can satisfy and a script can parse without knowing the
model. Version 1:

    # phase1-boundary v1
    step <start_position> <n_tokens> <token-id>...
    prefix <n_consumed> <token-id>...
    mtp-anchor <token-id|none>
    backbone <n_layers>
    row <layer-id>
    tensor <layer-id> <tensor-name> <dtype> <shape> <byte-offset> <byte-length>
    tensor ...
    row ...
    end

Rules:

* The `# phase1-boundary v1` line is the first line; a reader that does not recognise the version
  refuses the file. Blank lines and lines starting with `#` after it are ignored.
* `step` gives the step's absolute start position, its chunk token count, then that many token ids
  as decimal u32 in order.
* `prefix` gives the consumed prefix of section 1: how many token ids the sequence had already
  consumed, then those ids in order. A first step emits `prefix 0`. The start position and the
  prefix length agree for a sequence, and a reader refuses a manifest where they disagree.
* `mtp-anchor` gives the sampled anchor as a decimal u32 when the table has at least one MTP row,
  and `none` when no MTP row ran. The anchor is the first draft token the stages consume
  (model.py:1131-1132), not one of the row's eighteen tensors.
* `backbone` gives the model's `n_layers` (40 for the released configuration). It makes the anchor
  rule checkable: with `mtp-anchor none` the table has exactly `backbone` rows, layers `0` to
  `backbone-1`; with an anchor it has those rows plus at least one stage row, layer ids `backbone`
  upward. A reader must reject a manifest whose anchor presence disagrees with its row set: a
  `none` anchor alongside a stage row has lost an input, and an anchor alongside no stage row claims
  a stage that did not run. It also requires the rows to be exactly `0, 1, 2, ...` with no gap, so a
  missing row is a refused table rather than a shorter one.
* The four header lines (`step`, `prefix`, `mtp-anchor`, `backbone`) and the row/layer layout are
  the step's identity. `BoundaryTable::diff` compares the step fields (start position, chunk tokens,
  consumed prefix, MTP anchor) and the row set before any tensor; comparing the same token at the
  same position over a different prefix, or under a different anchor, is comparing different steps.
  The `backbone` line is a layout declaration beside that identity: the table does not carry it, so
  the pair tool (`boundary::diff_v1`, what `examples/boundary_diff.rs` runs) reads both manifests
  and refuses a pair whose backbone counts disagree before it diffs the rows - two anchored decode
  tables with identical rows but different backbone declarations would otherwise compare as the
  same step. Each header line must appear exactly once; a duplicate is refused rather than
  last-wins.
* `row` opens one layer's row and gives its layer id (backbone `0..n_layers-1`, MTP
  `n_layers + stage`). Rows are exactly `0, 1, 2, ...` and within a row the eighteen `tensor` lines
  are in the order of section 2, exactly once each. A row line carries no other field.
* `dtype` is one of three tokens, matching section 2: `f32` (4 bytes, little-endian), `u8-e4m3`
  (1 byte, an E4M3 code), `u8-e8m0` (1 byte, an E8M0 scale code). The token must be the one the
  tensor's name requires; a reader rejects any other token rather than guessing. A line with a
  trailing field is refused, not ignored.
* `shape` is the comma-separated dimension list of section 2, e.g. `1,4,5120`; the element count is
  the product of the dimensions, and the byte length must equal it times the dtype's width.
* `byte-offset` is the offset from the start of the binary blob named beside the manifest; lengths
  and offsets are decimal bytes. Tensors may be packed in any order in the blob, but a reader
  refuses two tensor lines whose byte ranges overlap, and a range that leaves the blob. The natural
  order — rows ascending, tensor order of section 2, no padding — is what `BoundaryTable::write_v1`
  writes, so two dumps of the same step from this writer are byte-identical; the reader accepts
  either the natural order or any non-overlapping packing.
* A tensor the run **cannot** emit must be an explicit line `missing <layer-id> <tensor-name>
  <reason>`, never an omitted `tensor` line. An omitted line changes the row's tensor count and the
  diff refuses the table (`BoundaryTable::diff`); an explicit `missing` line makes the same failure
  name the tensor, which is what criterion 1's "the run fails if any row misses its bar" needs.
* The blob is one file per step; the manifest may repeat for each step of a run, since a run's steps
  are separate tables. For a decode step, `n_tokens` is 1.

For sizing: at the released `dim = 5120`, `hc_mult = 4`, a one-token step is 338,448 bytes per row
(three streams of 81,920 B, four 20,480-B dense tensors, two code tensors of 5,120 B, two scale
tensors of 160 B, five 16-B coefficient sets and two 64-B comb sets), so the forty backbone rows are
13,537,920 bytes (12.9 MiB) and a decode step with three MTP stages is 43 rows. All of it is a
per-step diagnostic buffer, not a resident cache.

## 5. Comparison classes

Criterion 1 says every row is compared "each with its comparison class and bar". A boundary tensor
is one of two kinds, and each has one class:

| class | tensors | host-vs-host bar (this half) | device-vs-reference bar (Phase 1 device half) |
| :--- | :--- | :--- | :--- |
| **exact-bits** | `residual_in`, `hc_in_pre`, `hc_attn_pre/post/comb`, `pre_attention_hidden`, `attention_out`, `post_attention_hidden`, `hc_ffn_pre/post/comb`, `pre_ffn_hidden`, `ffn_out`, `residual_out` | 0 differing elements, 0 byte delta: two host runs with the same weights are bit-identical, and the perturbation test's "no other layer" assertion is exactly this bar | a tolerance bar, not bits: the device path carries its hidden buffers as bf16 ([device-path-design.md](device-path-design.md) section 3) and its reduction orders are not the host's, so the bar is measured by the run itself as the single-layer delta compounded over the layers it traversed, per criterion 3's decision. It is **not** a constant chosen to pass |
| **exact-codes** | `pre_attention_xhat`, `pre_attention_scales`, `pre_ffn_xhat`, `pre_ffn_scales` | 0 differing codes: same f32 input, same quantiser, same codes | 0 differing codes when the upstream f32 input is exact; when the upstream f32 differs, an element within rounding distance of a code boundary may move by one code (and a block whose `amax` sits on a power-of-two boundary may move its whole scale by one). The run reports the count and names the block; a code diff is never allowed to be silently explained away |

The per-tensor `max_abs_delta` is a value delta only for an exact-bits tensor. For an exact-codes
tensor it is the absolute integer distance between the two codes (for example 2 for `0x10` against
`0x12`), a unitless count rather than a numeric difference; `BoundaryRowDiff::max_abs_delta` is the
maximum over both kinds, so it is meaningful within a class and the per-tensor value is what a bar
checks.

Three consequences for the device half, so the classes are not just labels:

1. **A residual-stream diff is a propagated diff.** The row's `residual_in` is the previous row's
   `residual_out`; a difference anywhere upstream shows at every later row. The device comparison
   must attribute by the **first differing row** (`BoundaryDiff::first_differing`), whose inputs are
   the reference's, and testing a later row's own arithmetic needs the input-substitution replay of
   section 7.3. A comparison that reports "row 30 mismatched" without rows 0-29 being clean has
   attributed nothing.
2. **The exact-codes class is stricter than a tolerance and depends on the exact input.** It is the
   honest class for a quantiser whose rounding is specified: if the device quantises the same f32 it
   must produce the same code, and if it produces a different code the input differed. Comparing
   codes with a tolerance would hide exactly the arithmetic difference the table exists to find.
3. **Bars are per tensor, not per row.** A row can be exact in its streams and off by one code in a
   scale block; reporting the worst number for the row loses that. The host diff reports per tensor
   (`BoundaryTensorDiff::differing`, `max_abs_delta`), and the device report must too.

The host diff implements exact-bits as `f32::to_bits` equality — so `-0.0` and `+0.0` are
differences — and code equality for the byte tensors; a differing NaN pair reports an infinite
`max_abs_delta` so a numeric bar cannot pass by looking small (`compare_data`). The reason bits
rather than a tolerance are the host-vs-host bar: the reference is deterministic, so any bit change
is a real change, and the perturbation test needs a bar that cannot absorb its own signal.

## 6. The perturbation test

`crates/sparkflash-engine/tests/boundary.rs`, run by
`cargo test -p sparkflash-engine --test boundary`. The config is small but real: eighteen backbone
layers, `dim = 64`, `hc_mult = 2`, two heads of `head_dim = 16`, four routed experts, all
`compress_ratios` zero so every layer is sliding-window only, and one MTP stage targeting layer 17.
Not the released shapes and not `ModelConfig::tiny()` (two backbone layers): eighteen is the
smallest depth in which layer 17 is a real backbone layer, and the released layer-heterogeneity
modes are covered by other tests, so this one isolates the capture and the perturbation.

**The injection.** The weight source serves the plan's own names and shapes as deterministic finite
f32 values (a per-element hash of name, element and seed; the source is a test fixture, not a
checkpoint). The perturbed run differs in exactly one code: take element 0 of
`layers.17.attn.wq_b.weight`, re-encode the clean f32 value to E4M3, flip one bit of the code
(`code ^ 1`), decode it back to the f32 that code stands for, and serve that. The stored checkpoint
carries `wq_b` as FP8 E4M3 ([device-path-design.md](device-path-design.md) section 2), but this
source is a fixture: its clean values are f32, so the result is the f32 the changed E4M3 code
stands for, not a byte reproduction of the stored tensor's decoded value. It is a code-space
stand-in for a checkpoint edit, not an arbitrary f32 nudge. The source values sit at `0.05..0.15`
rather than the `0.01`-scale source of `tests/forward.rs` as a test-design choice: the larger source
kept the one-code change from being lost below the fixture's own rounding in this test. The
committed assertions are the row's measured diff counts and deltas below, not a score or ulp bar.

**The result.** The two runs' tables have 18 rows, one per backbone layer, each the full eighteen
tensors in the documented order and shapes; the unperturbed run's `residual_out(L)` equals
`residual_in(L+1)` byte for byte for every consecutive backbone pair; the perturbed run's logits
differ. The diff reports:

* `differing_layers() == [17]` — layer 17's row and no other;
* every other row `is_identical()`, `max_abs_delta() == 0.0`, 0 differing elements;
* inside row 17, the eight tensors computed before `wq_b` is read (`residual_in`, `hc_in_pre`,
  `hc_attn_pre/post/comb`, `pre_attention_hidden`, `pre_attention_xhat/scales`) are bit-identical;
* the row has **161 differing values in six tensors**: `attention_out` (64, max abs delta
  1.3969839e-9), `post_attention_hidden` (17, 1.4901161e-8), `hc_ffn_comb` (3, 5.9604645e-8),
  `pre_ffn_hidden` (8, 1.4901161e-8), `ffn_out` (12, 2.3283064e-10) and `residual_out` (57,
  2.9802322e-8). `hc_ffn_pre`, `hc_ffn_post` and the four quantised tensors
  (`pre_attention_xhat/scales`, `pre_ffn_xhat/scales`) are bit-identical in this run.

The row's `max_abs_delta` of 5.9604645e-8 is not a residual-stream delta: it is the largest delta of
the three differing `hc_ffn_comb` coefficients (destination-column entries of the FFN's Sinkhorn
matrix). `residual_out`'s own maximum delta is 2.9802322e-8. The counts are the measured numbers of
the committed test configuration and seed
(`a_one_code_change_in_layer_17s_wq_b_flags_layer_17_and_no_other_layer`); that test asserts all six
per-tensor counts and deltas and the 161 total, comparing the deltas by bits, so this paragraph
cannot drift from the code. A one-code weight change is small, and whether a particular tensor of
the row absorbs it or rounds it away is what gives each boundary its own comparison class (section
5) instead of one bar for the whole row. The quantised block inputs round back to the same codes
here, which is why section 5 permits codes to agree while the f32 stream they were quantised from
differs; it is a fact about the perturbation's size, not about the capture. These numbers were
re-measured after the comb-orientation correction described in section 3. An earlier revision
reported 155 differing values and called 5.96e-8 "exactly one f32 ulp at the residual's magnitude";
that attribution was wrong - the row maximum is an `hc_ffn_comb` coefficient delta, not a
residual-stream delta - and the transposed comb application produced those numbers. They do not
describe this code.

## 7. What a middle-layer change does, and the replay protocol

### 7.1 The finding

Criterion 2 asks that a one-code change in layer 17's `wq_b` be flagged at layer 17 "and no other
layer". A raw end-to-end diff of two full forward passes **cannot** satisfy that reading for a
middle layer, for a structural reason: layer L's `residual_out` is layer L+1's `residual_in`, so a
difference at layer L is carried by every later row. The test
`a_middle_layer_change_propagates_and_the_first_divergence_is_its_layer` pins it: with the same
one-code injection at layer 5, the diff's `first_differing()` is `Some(5)`, rows 0-4 are identical
and rows 6-17 are not. That is arithmetic, not a capture defect.

The literal criterion is satisfiable in exactly one arrangement: the perturbed layer is the **last**
row the table holds. The test therefore uses a config whose depth makes layer 17 the last backbone
layer — 18 layers — and every row before it is bit-identical while no row after it exists. That is
the reading committed here, and it is what the criterion tests: the table can fail, at the right
layer, and not elsewhere.

The reading has a cost the roadmap should record: as written, criterion 2 only exercises
attribution for a fault at the top of the stack. The general **default attribution rule** is
first-differing-row attribution: the rows before the first difference are bit-identical, so that
layer received the reference's recorded inputs and its row is the first place a difference can enter
- it is the layer the end-to-end diff attributes first. That is a default rule and a heuristic about
where to look, **not a proof**: it attributes at the granularity of a row and its eighteen recorded
tensors, and everything after the first difference is contaminated by the diverged stream - those
layers ran on different inputs, so their rows cannot say whether their own arithmetic is right.
Localising a fault to a particular operation, or testing a later layer's own arithmetic in
isolation, needs the input-substitution replay of section 7.3; it is a property of the device half's
localisation tool, not of the table format.

### 7.2 Why the test can fail (mutation evidence)

Two mutations were applied while developing this half and reverted; both make the acceptance test
fail, which is the evidence the test is not decoration:

* **Capture silently returns nothing**: change `run_observed` to call `self.run(state, tokens, None)`.
  Every test that asserts a row or a header fails (the first is `clean.boundary.len() ==
  config.n_layers`, 0 against 18, and the MTP tests fail the same way). A capture that returned
  nothing cannot pass the row and tensor assertions.
* **The injection silently does nothing**: make the source ignore its one-code change. The
  perturbation test fails at "the one-code change must change the perturbed run's logits", and the
  propagation test fails at `first_differing() == Some(5)` (it gets `None`). A no-op perturbation
  cannot pass the localisation assertion.

### 7.3 The input-substitution replay, for the device half

The end-to-end diff localises the **first** diverging row without help: rows before it are identical,
so its layer received identical inputs. A layer after that row is a different question - it ran on
the diverged stream, so a matching row would be surprising and a differing row proves nothing about
its own arithmetic. To test layer L's arithmetic in isolation, substitute its input.

**Scope: only a first-step backbone row is self-contained.** Row L's `residual_in` and `hc_in_pre`
fully determine the block's arithmetic when the attention has no prior cache to read, which is the
`start_pos == 0` case: the prefill attention attends over the chunk itself and `positions` is the
whole sequence so far, so no earlier window/compressed/index state exists. For any row after a
first step the row is **not** the layer's complete input:

* a **decode backbone row** (start position > 0) has its attention read the prior window ring built
  by the consumed prefix, and for a compressor layer the compressed KV and indexer state that
  prefix published. Those caches are engine state, not row tensors; `residual_in` and `hc_in_pre`
  alone do not determine the row.
* an **MTP row** has its stage attention consume `main_x` (the normalized projection of the
  backbone taps, model.py:1130, 1122-1126) and the sampled anchor in addition to the block's own
  input. The anchor is in the table header (`mtp_anchor`); `main_x` is **not recorded**. The stage's
  window ring is prior state as well.

The protocol is therefore:

1. Run the reference forward once for the step and keep the host table (it is the same table this
   document describes). For a first-step backbone row L, `residual_in` and `hc_in_pre` are the
   exact inputs and no cache is needed.
2. For a later-step backbone row, first seed the device's window/compressed/index caches to the
   engine's state after the consumed prefix (re-run the prefix on the device, or copy the engine's
   caches); for an MTP row, additionally obtain `main_x` and the header's anchor, and seed the
   stage's window ring. Then run layer L from row L's reference entry stream and input mix, with the
   same weights the compared run used, and write the device row for L.
3. Compare row L only. Its step inputs are now identical on both sides by construction, so every
   difference in the row is L's own arithmetic - and a clean row says L is right even if the
   end-to-end run was wrong above it.
4. Repeat for each layer whose arithmetic is in question; the cost is one layer's work per replay,
   not a full re-run of the stack.

For one token the reference forward is seconds on the host ([roadmap-to-serving.md](roadmap-to-serving.md)
section 4.6), the reference side of a replay is a lookup in the kept table, and the device side is a
single block once its caches are seeded. The host half supplies the reference row values; it does
not record the caches or `main_x`, so the device harness must supply those from the engine or replay
the prefix. This document does not implement the replay; it states the protocol the device half's
localisation tool must follow.

## 8. Code map

| piece | location | notes |
| :--- | :--- | :--- |
| tensor names, order, data and diff types | `crates/sparkflash-engine/src/forward/boundary.rs` | `TENSOR_ORDER`, `LayerBoundary`, `BoundaryTable`, `BoundaryDiff`; `push_stream`/`push_mixes`/`push_comb`/`push_block_input` validate shapes and refuse a dim that does not divide into blocks. `push_comb` always emits `[positions, hc_mult, hc_mult]`, so `hc_mult == 1` cannot collapse the comb to 2-D |
| observed entry points | `crates/sparkflash-engine/src/forward/mod.rs` (`prefill_observed`, `decode_observed`, `run_observed`) | build the table, run the shared pass, return `ObservedStep` |
| backbone capture sites | `crates/sparkflash-engine/src/forward/mod.rs` `fn block` | the nine points named in section 2; the row is pushed after the block returns |
| MTP capture | `crates/sparkflash-engine/src/forward/mtp.rs` `dspark_forward_observed`, `struct StageBoundary` | per-position buffers collected into one row per stage; buffers allocated only when a capture was asked for |
| the perturbation test | `crates/sparkflash-engine/tests/boundary.rs` | 15 integration tests: perturbation localisation, determinism/plain-path equality for prefill and decode, MTP rows on the tiny and on a 40-layer/3-stage config, middle-layer propagation, header/prefix/anchor capture, the exact diff refusals, the `dim % 32` refusal, and the `hc_mult == 1` comb shape in backbone and MTP rows |
| the diff-refusal unit tests | `crates/sparkflash-engine/src/forward/boundary.rs` `mod tests` | one test per header field (start, token, prefix, anchor), row count and layer id, tensor count/name/shape/format, duplicate rows, bit equality (`-0.0` vs `+0.0`), the two `max_abs_delta` units, and `push_block_input`'s 32-block refusal |
| the reference forward order | [../engine/01-forward-order.md](../engine/01-forward-order.md) section 3.2 | `Block.forward`, model.py:968-994 |
| mHC coefficient math | [../architecture/02-mhc.md](../architecture/02-mhc.md) sections 3-5 | `hc_split_sinkhorn`, `hc_pre`, `hc_post`; the comb is `[source][destination]` and `hc_post` sums a destination column over its sources (`forward/mhc.rs` `hc_post`) |

## 9. Reproduction

    cargo test -p sparkflash-engine --test boundary
    cargo test -p sparkflash-engine
    cargo clippy -p sparkflash-engine --all-targets -- -D warnings
    cargo fmt --check

The 15 boundary tests and what each pins:

| test | pins |
| :--- | :--- |
| `a_one_code_change_in_layer_17s_wq_b_flags_layer_17_and_no_other_layer` | the acceptance criterion: full rows, exact-bits elsewhere, layer 17 flagged, all six differing tensors with their counts and bit-pinned max deltas (161 total), input-side tensors unchanged, row-to-row stream wiring by bits |
| `the_observed_step_is_the_plain_step_and_two_runs_capture_identically` | the capture only reads (same logits as `prefill`), determinism, and that a different seed moves the table |
| `a_plain_decode_is_the_observed_decode` | the plain and observed decode paths return the same logits, token, draft, confidence and consumption when the MTP stages run |
| `the_header_records_the_consumed_prefix_and_the_mtp_anchor` | the header's prefix is empty on a first step, the decode's prefix is the consumed tokens, and the MTP anchor is the sampled next token |
| `the_mtp_stages_record_their_own_rows_when_they_run` | no MTP rows or anchor on a first step, one per stage after it, layer ids `n_layers + stage`, block-length shapes, non-zero rows |
| `the_released_row_count_rules_hold_at_forty_layers_and_three_stages` | the roadmap's row-count rule at the released layer counts: 40 rows on a first step, 43 after a prefill (40 backbone + 3 stages), stage ids `n_layers + stage` |
| `one_mhc_copy_still_records_a_comb_matrix` | `hc_mult == 1` still records a `[positions, 1, 1]` comb in a backbone row, not a 2-D tensor |
| `an_mtp_row_at_hc_mult_one_records_a_comb_matrix` | the same for an MTP row |
| `a_middle_layer_change_propagates_and_the_first_divergence_is_its_layer` | section 7.1's finding: first-differing-row attribution, and that later rows cannot be zero |
| `a_diff_between_different_steps_or_layer_sets_is_refused` | a table is non-empty by construction; the diff refuses different row counts, different tokens and different start positions with exact named messages rather than comparing different steps |
| `a_diff_of_the_same_decode_header_over_a_different_prefix_is_refused` | the review's repro: same start and token, 19 rows each, different consumed prefix - refused by the prefix message instead of comparing 18 differing backbone rows |
| `a_diff_of_the_same_header_over_a_different_mtp_anchor_is_refused` | same weights and header, different sampled anchor - refused by the anchor message |
| `a_diff_of_a_different_token_at_the_same_start_is_refused` | same start/prefix, different same-length chunk token - exact token message |
| `a_diff_of_the_same_token_at_a_different_start_is_refused` | same token, different start position - exact position message (the B4 gap; the prefix also differs in this engine-generated pair, so the unit test in `forward/boundary.rs` holds it equal) |
| `a_dim_that_does_not_divide_into_32_element_blocks_is_refused` | the observed entry point refuses `dim = 48` before running the step, with the exact shape message |

## 10. What this does not prove

* **No device numbers.** This document and the host tests state the format and the host-vs-host
  bars only. Device-vs-reference bars are the device half's, measured as criterion 3 describes.
* **Not the released shapes.** The perturbation test runs eighteen layers of `dim = 64`; it proves
  the capture, the row wiring and the localisation, not the released model's numerics. The
  40-row table on released shapes is the Phase 1 device run's deliverable.
* **One token, and only the window path.** The test's config has `compress_ratios` all zero, so no
  compressor, indexer, Engram or HSI path is exercised by the boundary test; those are covered by
  the engine's forward tests and are the heterogeneity criterion of the device half. A decode step
  after a prefill is exercised on the tiny config (one stage) and at the released layer counts on
  small shapes (40 backbone layers and three stages, which pins the 43-row rule).
* **No replay implementation, and a scoped replay.** Section 7.3 states the input-substitution
  protocol; the device half implements it. Only a first-step backbone row is self-contained. A decode
  row needs the engine's window/compressed/index caches for the consumed prefix, and an MTP row needs
  those caches plus `main_x` and the header anchor; the table records the anchor, not the caches or
  `main_x`.
* **Attribution is first-row, not causal for later rows.** The first differing row's layer received
  the same inputs as the reference, so the table attributes that one layer's arithmetic; every later
  row is explained by the divergence above it and is not evidence for or against its own layer until
  the replay of section 7.3 substitutes the input.

## 11. The released-checkpoint host dump (measured)

The host side of the format is committed and measured. `BoundaryTable::write_v1` and `read_v1` in
`forward/boundary.rs` write and read section 4, `boundary::diff_v1` reads two pairs as a pair
(requiring both to declare the same backbone count) and diffs them, `examples/boundary_dump.rs`
writes a pair from one reference step, and `examples/boundary_diff.rs` runs that pair gate, prints
the per-layer/per-tensor differences and exits 1 when anything differs. The first released-checkpoint dump was produced on
2026-10-17 on `node0` (aarch64) with the release build of the example committed in !44
(`9d4e78d`); this MR adds the head dump flags, against the released snapshot in the node's Hugging
Face cache:

    boundary_dump --checkpoint <snapshot> --out phase1-host-token0

It wrote 40 rows of 18 tensors, a 34,767-byte manifest and a 13,537,920-byte blob, which is exactly
the section 4 size table (40 x 338,448 bytes); `boundary_diff` of the pair against itself reports all
40 rows bit-identical. A later run of the same deterministic example (18:20, the table above is 17:34)
with `--head-input-out` and `--logits-out` wrote criterion 3's two sides as raw little-endian
f32: 5,120 head-input values (all finite, min -2.0506, max 1.41963,
L2 norm 18.157493) and 129,280 logits with argmax 5 and top-5 [5: 13.230665207, 201: 12.492528915,
372: 11.897726059, 30: 11.392500877, 1897: 11.015835762]. The device head harness reads those two
files and will compare its own logits against them. A CPU precheck of that comparison
(`m7_layer0 --stage 6`, run on `node0` on 2026-10-17 while vLLM held the node: 50 s, no CUDA
context) recomputed all 129,280 logits from the reference's f32 head input and from the bf16 codes
the kernel ABI is fed, in the kernel's ascending order. Both recomputations reproduce the reference
argmax 5 and top-5 exactly; the f32 recomputation matches the reference logits exactly (max abs
delta 0), so the host's accumulation order is the engine's, and the bf16 conversion the kernel cannot
avoid moves a logit by at most 2.4320781e-2. That number is why the stage-5 comparison uses a bf16-matched
reference rather than the f32-hidden logits: otherwise the conversion would be charged to the head.
The measured device-vs-matched delta was 0 (the device reproduced the matched host's values); the bar
remains the provisional 2e-2 ceiling rather than folding a 2.4e-2 conversion into a bar about the
head's arithmetic.

The device run then happened, in a full window on `node0` with vLLM stopped (2026-10-17,
118 GiB free; the timed log is `/home/user/phase1-evidence/stage5-device-run-timed.log`, and the
rebuilt-binary logs that match the current PASS lines are `stage5-device-final.log` and
`stage6-cpu-final.log`). `m7_layer0 --stage 5` launched `cert_lm_head_sm121` once per rank over the four
32,320-row shards of `head.weight` and printed:

* device argmax **5**, value **13.237006187**, equal to the bf16-matched host reference (and to the
  f32-hidden reference argmax 5);
* **all 129,280 entries exact, 0 certified**: the screen's bound (per-128-block hidden sums times the
  row's max |w|) never fell below the running maximum on this hidden state, so it pruned nothing and
  the whole vocabulary took the exact path - the run is the exact BF16 GEMM, not a pruned one. The
  harness prints no bound margin, so this records what happened, not how far above the maximum the
  bound sat;
* **max |device - bf16-matched host| = 0** over every entry: the device values equal the host
  recomputation in the kernel's order, so the head's arithmetic is reproduced exactly (as a value
  comparison, not a NaN-aware bit comparison) and the only difference from the reference logits is
  the measured 2.4320781e-2 bf16 conversion of the hidden vector;
* the certificate check was therefore not exercised and says so rather than claiming it passed; the
  kernel's 32-value block reduction was cross-checked against the per-entry scan on every rank;
* the run took 1.56 s including the 1.32 GB upload, and a second run printed the identical stage-5
  result lines.

This is criterion 3's device half for the argmax and the exact logits. The top-5 set is not claimed:
a certified entry stores a bound, and the kernel's contract is the argmax; with 0 certified entries on
this token the top-5 happens to be recoverable, but the harness does not claim a contract it cannot
honour in general.

As a precondition for the chained 40-layer token harness, the per-stage harness was run with --layer
on other layers: stage 1 (mHC) PASS at layers 1 and 4 (2 tokens, every boundary), stage 2 (attention)
PASS at layer 1 (2 positions, exact 99.9878%, max abs delta 7.6293945e-6 against its 1.6e-2 bar) and
stage 3 (MoE, 8 experts) PASS at layer 1 (combine exact 100%, max ULP 0 against its 2-code bar). A
later run (log /home/user/phase1-evidence/layer-dispatch-recon-engram-last.log) adds stage 1 PASS at
layers 14 and 39 (2 tokens, every boundary), stage 2 PASS at layer 14 (2 positions) and stage 3 PASS at
layer 14 (8 experts, 2 tokens). The stages read the layers.N. tensor names and take --layer, so the
kernels are not layer-0-specific; what remains is chaining them, which the stages deliberately do not do
(each builds its own inputs and compares one stage's outputs). Log of the first run:
/home/user/phase1-evidence/layer-dispatch-recon.log. The one-token report on the same snapshot
printed:

* 95,819 tensors verified, 4 Engram tensors paged (the two released Engram tables, weight and scale);
* 40 rows, all host-vs-host bit-exact, two runs byte-identical over the boundary tables and the
  129,280 logits;
* every logit finite, argmax 5, top-5 [5: 13.230665207, 201: 12.492528915, 372: 11.897726059,
  30: 11.392500877, 1897: 11.015835762];
* audit plan=95,819 read=2,643 unread=93,176 partitioned as [unselected-expert 90,720,
  DSpark-stage 2,401, text-path 40, indexer 15];
* wall clock 2:20.25 and peak RSS 6,689,384 KiB (6.4 GiB) for the whole run.

The artifacts stay on the node: `/home/user/hostrun/phase1-host-token0.manifest`, `.blob`,
`.head.f32` and `.logits.f32`, and `/home/user/hostrun/host_report_token0.txt`; they are not
committed here (the blob alone is 13 MB). This is the host column of criterion 1 and the reference half of criterion 3. It is not the
device column: no device table has been captured, so nothing here says the device path matches this
table, and the bars of section 5 remain to be measured against it.


## 12. The chained device path, layer 0 (measured 2026-10-18)

This is the device column for the layer-0 boundary: the M7 harness stages that feed each other's
inputs, run on the node in one window (vLLM stopped; pinned nvcc V13.0.88 cubins, binary sha256
`56f82fb7...`, mega_mhc cubin sha256 `e21293e3...`). Stage 7 (B2) is the chained replay and ran in
the next windows on the same node; its final run is binary sha256 `2b518a35...`, the cross-build of
the pass-5 code commit `1845f2d` (the evidence binary is that commit's own build; the docs head's
rebuild reproduces it), and the log is
`/home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log` (23,897 B,
local copy sha256 `c2de544c1789743e73ccc5d5277dc3165d98296d0f5cc867cc819d9160754ac9`; 20 device
launches: 16 for the chain, 4 for the forced-routing control), UTC 04:41:22Z-04:41:26Z. The pass-5
log's metric and content lines are bit-identical to the pass-4 log's (only the label/argv/binary-sha/
UTC header lines and the device/node MemAvailable lines - header before and after, process start,
per-launch and staged-weight - differ), so the tie plumbing did not move any measured number. The pass-4 log (`...rereview4-20261018T041707Z.log`, binary `f99f14bf...`,
pre-plumbing) remains on disk and is no longer cited. Every log names its argv, binary
sha256, UTC
start/end and `MemAvailable` before and after; a log of a run that loads kernels names the cubin
sha256s too, with one exception: the two B1 probe attempts
(`phase1-b1-stage8-probes-20261018T011320Z.log`, `...011404Z.log`) record the binary only, so they
are cited for their probe numbers and not as cubin provenance. The logs are
`/home/user/phase1-evidence/phase1-d8-*` and the earlier B0 runs `phase1-b0-*`. **Log
regeneration pending:** the stage-1 and stage-2 logs behind the table below were produced before the
M2 fix of their printed comparison-class strings and still print the retired stream-mean fold as the
predictor. Their numbers are cited here, but they are not current evidence until regenerated in a
device window; the printed class lines in this file are the ones the fixed harness emits.

| stage | what it checks | measured result | log (UTC) |
|---|---|---|---|
| 0 | released `[24, n*d]` projection against the engine's captured row-0 coefficients, plus the captured activation quantiser recomputed with the ceil rule | projection worst abs delta 5.96e-8 against a 1e-5 bar; ceil-rule codes 5,120/5,120 and scales 160/160 exact for both block inputs; retired floor rule would give 4,288/5,120 + 134/160 (attn) and 4,096/5,120 + 128/160 (ffn). PASS | 01:25:33Z-01:25:34Z |
| 1, layers 0/1/4 | `mega_mhc_fused_bf16_sm121` against `mega_mhc_ref::mega_mhc_layer_with_scale` | PASS 4/4 tokens each. Layer 0: token 0 matches **5,119/5,120 codes** (one one-code difference at index 615) with 160/160 scales bit-exact; tokens 1-3 match 5,120/5,120 codes and 160/160 scales. Layer 1: 5,120/5,120 codes and 160/160 scales at every token. Layer 4: token 0 matches **5,118/5,120 codes** (two one-code differences, scales exact) and its other tokens match everything. Where a code differs the cause is the known fixed-order vs sequential `1/rms` reduction delta, inside the stage-1 bar. | 01:26:13Z-01:26:26Z |
| 8 | dense MXFP8 GEMM, grouped wo_a, SwiGLU activation quantiser against the in-crate references on identical bytes | 4/4 probes bit-exact: dense 16,384/16,384 and wo_a 8,192/8,192 (max ULP 0); swiglu 4x640 codes 2,560/2,560 + scales 80/80; swiglu 2x2304 codes 4,608/4,608 + scales 144/144 | 01:26:36Z-01:26:37Z |
| 2, layer 0 | `flash_mla_csa2_sm121` against `mla_ref::attention_with_sink` | PASS 3/3 positions: 32,768/32,768 bf16 exact at positions 0 and 1; position 3 32,765/32,768 (99.9908% vs 99.5% bar), max ULP 1, max abs delta 1.19e-7 (bar 1.6e-2) | 01:26:49Z-01:26:50Z |
| 3, layer 0 | `b12x_mxfp4_moe_sm121` down/combine against `mxfp4_ref::moe_combine` | PASS: combine 20,480/20,480 bf16 exact, max ULP 0; the printed quantisation gap (quantised vs f32 activation) is 100% of outputs, unchanged, because this stage's device boundary is the combine only | 01:26:51Z-01:27:03Z |
| 7 (B2), layer 0 | the chained layer-0 replay: K1 mHC, K2/K3/K4 dense, K5 flash_mla, K6 wo_a, K7 dense, K8 mHC, K9-K12 routed combine, K13/K14/K15 shared dense, K16 final hc_post; all 18 boundary tensors of row 0 captured and diffed against the host v1 row by class; plus a forced-routing control that re-runs the routed MoE with the host row's own top-6 ids and weights | PASS 18/18 captured and diffed, no bar applied (this is the single-layer measurement, not the 40-layer chain). Every kernel is bit-exact against its in-crate reference on the same bytes: K2 1,280/1,280, K3 32,768/32,768, K4 512/512, K5 32,768/32,768 (100% vs `mla_ref` with the sink), K6 8,192/8,192, K7 5,120/5,120, K9-K12 5,120/5,120 max ULP 0 (forced control also 5,120/5,120 max ULP 0), K13/K14 2,304/2,304, K15 5,120/5,120. Row-0 deltas: upstream `pre_attention_hidden` max 2.76e-7 (mean 4.28e-8), `pre_attention_xhat` 5,119/5,120 codes (the one difference is distance 1) and `pre_attention_scales` 160/160 exact; `attention_out` max 2.57e-1 (mean 5.72e-2), `post_attention_hidden` bf16(host)-exact 48.18%, `pre_ffn_hidden` max 3.16e-2, `pre_ffn_xhat` 1,596/5,120 codes (histogram 1: 1,694, 2-3: 898, 4-15: 751, 16+: 181; mean distance 7.33, max 177), `pre_ffn_scales` 154/160 (6 differ, all distance 1), `ffn_out` max 2.15e-1, `residual_out` bf16(host)-exact 5.96% (max vs bf16 1.875e-1). The forced-routing control measures the discrete flip's own contribution pointwise: max abs |`ffn_out` free - `ffn_out` forced| = 3.750000e-1 at index 5071 (signed -3.750000e-1), 4,853/5,120 points differ, and the two host deltas are reported separately (free 2.150760e-1 at index 1233, forced 2.160320e-1 at index 5071; their difference is a change in max error, not a max-abs contribution). The flip is a large pointwise perturbation, and the measured relation at the max host error is: at index 1233 the flip contributes 2.500000e-1, 116.24% of the 2.150760e-1 host error there, so that max-error point is flip-dominated; forcing the host row's routing does not reduce the max host error | 04:41:22Z-04:41:26Z |

Rule change (D23): the old floor-rule behaviour is the mislabelled log above (4,287/5,120 codes
and 134/160 scales on token 0, but with the new binary and new cubin hashes in its header). The
pre-D8 floor-rule cubin is retained at
`/home/user/phase1-evidence/b0-out/mega_mhc_fused_bf16_sm121.cubin` (70,152 B sha256
`6487f6ff...`); a properly labelled
`OLD-VS-NEW` run (old cubin + new ceil-rule reference, both binary and cubin hashes in the log)
is a **device-window item** and is not cited as evidence until it exists.

This B2 run is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure, and the report must say
so: the device path uses FP4 main/window KV (D4) and bf16/quantised arithmetic where the host
reference is f32 with no GEMM activation quantiser and no bf16 writes
(`crates/sparkflash-engine/src/forward/attention.rs` module docs, "What is deliberately not
reproduced", items 1-2). The column therefore measures attribution and gross mismatch, layer by
layer; task-3 has since landed the FP8 window-KV device path and measured its engine-format column
(163,834/163,840 exact, max abs 9.765625e-4, max ULP 1 against the declared 99.5% / 1.6e-2 class;
section 13.7), which is a format-fidelity class separate from this FP4 health-check column, and the
release-faithful arithmetic that would close exactness is still Phase 2. Criterion 1 is not closed
on this run.

The row-0 deltas have two named sources, separated by the B2 attribution check in the same log:
(1) the window-KV format of D4 - the same f32 latent re-encoded in the device's FP4 E2M1 format and
the host's FP8 E4M3 format predicts `attention_out` with a max abs difference of 3.13e-1 on the same
latent and the same tail, which is the format's own contribution and the valid D4 number; the
host-format prediction is 1.05e-1 from the host row against the measured 2.57e-1, but it reuses the
device's q and attention-tail operands, so it also carries the upstream device-side differences and
is reported as a diagnostic, not attributed to the format; (2) the host reference's own
non-reproduced arithmetic, which the residual belongs to. The prediction's control - the same
arithmetic in the device's format - reproduces the run's `attention_out` exactly. The one discrete
event measured on this row is the router's top-6: the ordered lists differ in four slots and the
expert sets differ by one member (device only 246, host only 236). Its effect is measured, not
assumed: the forced-routing control re-derives the host row's routing from the host row, builds the
forced operands from it and verifies them against the checkpoint before any launch - every rank's w2
weight and scale slices, and every slot's w1/w3-derived activation codes and scales at every rank,
bit-for-bit for the requested ids/weights; it also recomputes the forced `ffn_out` from the forced
routed-combine bits and the K15 shared output through the free path's combination, requires it
bit-identical to the tensor it diffs, and reads that returned recomputation for every reported
forced number (so a copy after the assertion cannot change a metric), and its verdict requires
identical ids with unchanged weights
to reproduce the routed bytes exactly (a difference there is nondeterminism and aborts) while a flip
must produce a nonzero difference unless the two independently computed references are bit-identical
(then the flip cannot move the bytes and is reported, not aborted; Stage C, section 13.5, found such
a layer) - then reports the free-vs-forced difference pointwise - max abs
3.750000e-1 at index 5071 (signed -3.750000e-1), 4,853/5,120 points differ - separately from the two
host deltas (free 2.150760e-1 at index 1233, forced 2.160320e-1 at index 5071), whose difference is a
change in max error and not the flip's max-abs contribution. The flip is a large pointwise
perturbation, and the measured relation to the max host error is the opposite of a small one: at the
free max-error index 1233 the flip contributes 2.500000e-1, 116.24% of the 2.150760e-1 host error
there, so that max-error point is flip-dominated; forcing the host row's routing leaves the max host
error essentially unchanged. The residual consists of the device path's quantised routed arithmetic
plus the propagated upstream deltas, and the combine kernel is excluded (K9-K12 exact on the same
bytes, free and forced). No device `v1` table file is written yet
(the stage diffs row 0 in-process and prints the deltas), and the 40-layer token chain is a separate
program whose bars are the B2 single-layer deltas compounded over the layers a tensor traversed,
printed before the numbers they judge. The readers, the released router and the free-vs-forced
metric behind this section are compiled and exercised on the CPU by
`crates/sparkflash-kernels/tests/b2_harness_offline.rs` (14 tests: the readers, the released router,
the free-vs-forced metric, the forced-output tie against the forced routed bits - including that the
metric consumes the tie's returned tensor, so a post-assert copy cannot change it - the
identical-routing determinism rules, and the full pre-launch operand verifier exercised end to end on
synthetic tensors including a top-k-6 fixture that corrupts slot 3's scale and slot 5's w1), so the
checkpoint/manifest
parsing, the released router, the free-vs-forced metric, the determinism verdict (positive and negative cases: ids equal with
changed weights may round to equal bytes, ids equal with unchanged weights must reproduce the bytes,
a flip must change them) and the operand-byte provenance verifier are not validated only by a device
run; the manifest offset + length overflow branch is exercised too.

The 40-layer token chain that consumes these bar inputs is stage 9; it writes a device `v1` pair,
and its measured column - determinism, the `boundary_diff` verdicts, the compounded bars and the
audit - is section 13.

## 13. The 40-layer chained device column (Stage C, measured 2026-10-18)

Stage C is the device half's 40-layer column: `m7_layer0 --stage 9` runs one released-checkpoint
first step (token id 0, position 0, backbone rows only - a first step has no MTP rows) as a single
chained device forward of 40 rows, records the same 18 boundary tensors per row that stage 7
records, and writes its own device `v1` pair so `boundary_diff` runs unchanged.

The code is rebased onto main `2214fe6` (the squashed B2 !51) and reuses the B2 harness instead of
a second reader. The duplicate C safetensors reader is deleted; the shared
`crates/sparkflash-kernels/tests/support/b2_harness.rs` module supplies the snapshot reader and the
host boundary table (the module gains the read audit - `read_names`/`read_bytes`/`audit` - and a
`manifest_text` accessor so the stage-9 audit and manifest comparison read the shared structures).
Both the stage-7 replay and `stage9_ffn_block` use the four B2 helpers: `combine_bf16_routed_with_shared`
for `ffn_out` and for the forced path's free combination, `verify_forced_ffn_out` (whose RETURNED
recomputation feeds both host deltas and the pointwise flip metric), `forced_control_verdict` for
the ids/weights/routed-bits verdict, and `verify_forced_operand_provenance` with
`FORCED_OPERAND_GEOMETRY` before every forced launch. The frozen bars still come from the B2
evidence of record (section 13.3), and the older Stage-C run is retired (section 13.1).

Per row the chain runs the 15 launches of decision D5 (K1..K15) and four forced-routing launches
that re-run K9-K12 with the host row's own top-6 ids and weights on the same bytes, so every row
carries both a free-running and a forced `ffn_out` measurement. A dedicated K16 launch applies the
last `hc_post` for the row before an Engram layer and for the last row, because the next layer's
K1 exists only when the chain continues (D5): 763 launches per chain run =
40 x (15 + 4) + 1 final K16 + 2 K16 before the Engram layers. The two Engram writes (layers 1 and
14) are the engine's released `forward::engram::engram_forward` with `ngram_hashes` over a paged
`SafeTensorsSource` - the released code, not a second copy of the arithmetic - applied to the
bf16 device stream (the seam statement prints before the numbers). The chain carries the previous
row's FFN output and its kernel-order B/C/A mHC coefficient triples into the next row's K1, so the
next K1's step 1 performs the previous row's last `hc_post`.

### 13.1 The evidence, and what makes it a run rather than a claim

- log `/home/user/phase1-evidence/phase1-c-stage9-40rows-harness5b-20261018T050416Z.log`
  (572,420 B, sha256 `bd0f908ef2e3519512a20bd53eb86303fcfad65550f7b145a6c81c7a2c973a65`):
  `--stage 9 --checkpoint <released snapshot> --host-row /home/user/hostrun/phase1-host-token0
  --stage9-out /home/user/phase1-evidence/stage9-out-r5b --stage9-layers 40 --stage9-runs 2
  --cubin-dir /home/user/hostrun/cubins`, 40 rows x 2 chain runs, UTC 05:04:16Z-05:09:14Z,
  wall 298.6 s, exit 0. The wrapper records argv, binary sha256, every cubin sha256, UTC start/end
  and node MemAvailable before/after; the harness prints its own wall, v1 blob sha256s and
  memory-floor checks. Node MemAvailable: 123,438,768 kB before, 123,194,824 kB after, minimum
  during the run 122,285,404 kB (floor 32 GiB). The wall is 67.7 s longer than the retired run's
  230.9 s because every layer's forced operands are now verified against the checkpoint before the
  forced launches.
- binary sha256 `c1708abcfa19bf37fe4fa74c7f0057d1689df46ffada196b3cdd84ccb355f211` (the C head's
  cross-build; the C branch is rebased onto main `2214fe6`). Cubins:
  `mega_mhc_fused_bf16_sm121 e21293e3...`, `swiglu_mxfp8_quant_sm121 4dbef63b...`,
  `b12x_mxfp8_dense_sm121 03e4c422...`, `b12x_mxfp8_wo_a_sm121 84833781...`,
  `b12x_mxfp4_moe_sm121 36a15a4f...`, `flash_mla_csa2_sm121 822df7f8...`,
  `cert_lm_head_sm121 a833a336...`.
- device pair `/home/user/phase1-evidence/stage9-out-r5b/device.manifest` (34,767 B, sha256
  `6093c74b...`) and `device.blob` (13,537,920 B, sha256 `fecd813b...`). Both are
  bit-identical to the retired Stage-C run's pair, and all 1,520 printed row lines are bit-identical
  to that run's: the helper adaptation changed verification and reporting, not the device column.
- diff log `phase1-c-stage9-diff-20261018T050941Z.log` (sha256
  `b86be96879239a82a46559d0df2bd2e050ccc2c0f66986fe76a044ed676b5eb4`): the control and device runs
  of `boundary_diff`, binary sha256 `5ffb11f9...`, host manifest `6093c74b...`, host blob
  `68970cc8...`, device manifest `6093c74b...`, device blob `fecd813b...`. The control exits
  0 with all 40 rows bit-identical and the device column exits 1 with all 40 rows differing; its
  body is line-for-line the retired run's diff body because the pair it reads is the same pair.
- Retired, kept on disk as historical evidence and no longer cited: the earlier Stage-C run
  (`phase1-c-stage9-40rows-final-20261018T023529Z.log`, binary `f5430527...`), its smoke log
  (`phase1-c-stage9-smoke-20261018T023940Z.log`) and its diff log
  (`phase1-c-stage9-diff-20261018T024000Z.log`, binary `a7822cab...`). A pre-refinement attempt at
  this column (`phase1-c-stage9-40rows-harness5-20261018T045752Z.log`, 259,368 B, sha256
  `5b8da864...`, exit 1, no pair written) aborted on the L36 zero-effect flip before the verdict
  refinement; it is the historical record of that case, not column evidence.

### 13.2 Determinism, the v1 pair, and the diff verdict

- **Determinism: PASS.** The two chain runs are bit-identical on all 18 tensors of all 40 rows;
  both v1 blobs hash to `fecd813b...` (manifest `6093c74b...`). The two runs' printed row
  reports are identical line for line.
- **The v1 writer is proven, not asserted.** The 40-row device manifest is byte-identical to the
  host pair's manifest (same 34,767 B, same `6093c74b...`), so `boundary_diff` reads both
  halves through the same reader; the device blob is 13,537,920 B, the same layout the host blob
  uses.
- **`boundary_diff` verdicts.** The control run (host pair against itself) prints "diff: all
  40 rows bit-identical" and exits 0. The device column (host pair against the device pair) prints
  "diff: differs in 40 rows" and exits 1. Both statuses are the engine tool's documented contract
  (0 = identical, 1 = differs). Its per-row line "row N: X of 18 tensors differ" reports the number
  of differing values across that row, and its "diff: differs in N rows" is its own summary
  wording: that is the engine tool's verdict, not a pass/fail label this document applies. The
  claims made here are the pair's identity as a v1 pair, the manifest equality and the all-rows
  differ result.

### 13.3 The bars, the row-0 circularity, and the totals

Bars are pre-registered in design section 8 and printed before the numbers they judge:
`bar(row L, T) = single_layer_delta_B2(T) * (L + 1)` for the three stream-carrying tensors
(`residual_in`, `post_attention_hidden`, `residual_out`) and
`single_layer_delta_B2(T)` for the per-layer tensors. `single_layer_delta_B2` is the frozen
Stage-B measurement of record in
`/home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log` lines 149-166
(binary `2b518a35...`; 23,897 B, local copy sha256 `c2de544c...`); this run does not re-measure it. Two layer-0 exact seeds
inherit the measured delta of the tensor that produces them, which the log's bar-input table prints:
`residual_in` uses `residual_out`'s delta (row L residual_in is row L-1 residual_out) and
`hc_in_pre` uses `hc_ffn_pre`'s. No bar comes from this run's numbers.

**Row 0 is circular, and is labelled so.** Row 0 is the Stage-B replay: it runs the same layer-0
launches stage 7 ran, and its bars ARE its own Stage-B measured deltas by construction. It landed
at 18/18 at or under (worst measured/bar 1.000), which proves the chain reproduces Stage B bit for
bit on that row - and is not independent evidence about the 40-layer column. **Only rows 1-39 are
independent evidence; no row is used to judge itself, and there is no terminal-row exemption:
row 39's stream-tensor bars are the frozen B2 delta times 40, a number fixed before this run, not
row 39's own measured value.**

Five row-0 comparisons land inside the frozen log's 7-significant-digit print precision, so the
harness names them `AT-BAR-PRECISION` instead of claiming either a pass with no headroom or a
fail caused by rounding: `hc_attn_post` (3.278255e-7), `hc_ffn_post` (7.206082e-3),
`hc_ffn_comb` (1.175366e-3), `pre_ffn_hidden` (3.163002e-2) and `pre_ffn_xhat` (codes
3,524/5,120, bar 6.882812e-1 = exactly that fraction). The class is 1e-6 relative on the printed
constant; a zero bar stays exact.

Totals over the 40 x 18 = 720 tensor-row comparisons: **306 at or under the bar, 414 over it.**
**The first over-bar row is row 1** (the first post-Engram row). The run never stops early on a bar
miss: the column runs to row 39 because its job is attribution. Per tensor, over-bar counts of the
40 rows: `hc_attn_pre`, `hc_attn_post`, `hc_attn_comb`, `pre_attention_hidden`,
`pre_attention_xhat`, `post_attention_hidden` and `residual_out` 39 each; `residual_in` 38;
`ffn_out` 34; `attention_out` 24; `pre_attention_scales` 23; `pre_ffn_hidden` 16;
`hc_ffn_post` 4; `hc_ffn_comb` 2. Four tensors never miss: `hc_in_pre`, `hc_ffn_pre`,
`pre_ffn_xhat` and `pre_ffn_scales`.

### 13.4 What the row-by-row column measures

Max-abs deltas at selected rows (all values from the log's row lines; rows 1-39 are the independent
ones):

| tensor | row 1 | row 13 | row 14 | row 20 | row 30 | row 39 |
|---|---|---|---|---|---|---|
| `residual_in` | 1.815126e-1 | 3.628571e0 | 5.634033e0 | 7.340234e1 | 2.822906e3 | 2.608062e3 |
| `residual_out` | 1.411274e0 | 3.851440e0 | 7.421875e0 | 3.731406e2 | 3.681188e3 | 3.031875e3 |
| `post_attention_hidden` | 1.791198e-1 | 3.535789e0 | 6.494629e0 | 6.760352e1 | 2.825281e3 | 2.813406e3 |
| `ffn_out` | 7.049999e-1 | 3.282175e-1 | 1.269871e0 | 1.435820e2 | 1.250229e2 | 4.921960e1 |
| `attention_out` | 1.524859e-1 | 2.435673e-1 | 4.817200e-1 | 3.712856e0 | 7.084037e0 | 3.376434e1 |

The reading: the per-layer attention output stays inside its flat 2.565311e-1 bar through row 13
(first over-bar row 14; 24 of 40 rows over), while the stream tensors compound: residual_out goes
from 1.41e0 at row 1 to 3.03e3 at row 39, residual_in from 1.82e-1 to 2.61e3 and
post_attention_hidden from 1.79e-1 to 2.81e3; ffn_out is 1.44e2 at row 20 (the column maximum is
7.957832e2 at row 24). That is the
gross-mismatch growth the health-check class predicts; it is reported, not smoothed, and the
discrete routing events are kept out of the bars.

The Engram rows: row 1 (after the layer-1 write) is the first over-bar row (9 of 18 over); row 14
(after the layer-14 write) has 12 of 18 over. The writes themselves are identical across the two
runs (layer 1: 20,479/20,480 values changed, max abs write 1.194677e0; layer 14: 20,480/20,480,
max abs 1.369427e1), and the host pair's wiring check confirms the seam is the only thing that
breaks residual_out(L) = residual_in(L+1): the host pair differs at rows [0, 13] and nothing else,
which is exactly the rows before the Engram layers [1, 14].

### 13.5 The discrete routing events

Each layer prints its router comparison (ordered top-6 lists, differing slots as (slot, device,
host), set difference) and one forced-control line that separates four measurements: the
ids/weights/routed-bytes verdict together with whether the two independently computed references are
bit-identical, the pointwise |`ffn_out` free - forced| maximum with its index and the number of
differing points, the two host max-abs deltas separately (free-running and forced, each with its
index), and the routed-combine exactness. A layer whose lists or sets differ is printed as
`DISCRETE EXPERT FLIP: a divergence event to report per layer, not a smooth bar`. **17 of 40
layers flip: [0, 1, 2, 4, 5, 8, 10, 11, 15, 16, 17, 19, 20, 23, 24, 35, 36].**

The attribution is pointwise and says nothing it cannot measure. Row 0 (the Stage-B replay) carries
the B2 numbers: the flip's pointwise max abs is 3.750000e-1 at index 5071 (signed -3.750000e-1) with
4,853/5,120 points differing; the two host deltas are free 2.150760e-1 (max at index 1233) and forced
2.160320e-1 (max at index 5071). The difference of those two maxima is a change in max error, not the
flip's max-abs contribution; the pointwise measurement is. At the free max-error index 1233 the flip
contributes 2.500000e-1, 116.24% of the 2.150760e-1 host error there, so that row-0 max-error point
is flip-dominated, and forcing the host routing does not reduce the max host error. The earlier
reading of the flip as the difference of the two host-error maxima (free minus forced) was wrong -
it subtracted two maxima taken at different indices - and is withdrawn.

Across the column the free-running `ffn_out` max-abs reaches 7.957832e2 and the forced reaches
5.397832e2; neither maximum is decomposed into a flip contribution, and the flips are never folded
into a tolerance. **The L36 worked example (zero-effect flip).** Layer 36's ordered ids differ in
slot 5 (device 186, host 150), but the two independently computed routed-combine references are
bit-identical, so the flip cannot move the bf16 output by construction: the routed bytes are
identical, the pointwise flip is 0.000000e0 (0/5,120 points differ), and both host deltas are
1.932627e2 at index 2455. The verdict prints `references bit-identical true` and allows the
outcome; the pre-refinement attempt (the retired `...harness5-...T045752Z.log`) aborted here because
the old rule treated ids-differ + bytes-identical as a copy. The refined rule keeps that abort for a
flip whose references differ - the two references are the harness's own checked 100%-exact
comparisons, so a forced launch that ignored the host routing cannot pass - and adds a separate abort
when the references are bit-identical but the routed bytes differ (a launch that does not reproduce
its own reference; it is broken or nondeterministic).

### 13.6 Exactness inside the harness, the audit, and the seams

- Every kernel against its in-crate reference on the same bytes: 80/80 K5 checks 100% exact
  against `mla_ref`, 640/640 per-layer dense/WKV/wo_a/shared reference checks exact, 160/160
  routed-combine checks exact (free and forced), zero WARNING lines.
- Audit: `plan=95819 read=2673 unread=93146`, and the unread set partitions exactly:
  unselected-expert 90,660 + DSpark-stage 2,401 + text-path 40 + indexer 32 + compressor 11 +
  head-not-claimed 2 = 93,146, with 0 unclassified. The compressor/indexer counts are larger than
  the host report's because the S7 seam is degenerate at position 0 - the window holds only
  position 0, so this chain never reads those tensors while the host reference does; the log says
  this is a property of the health-check column, not an unread-tensor bug. Bytes through the
  harness's own reader: 2,661 tensor reads / 20,338,614,720 B; the two Engram tables are read
  through the engine's paged source and are outside that counter. The byte total is larger than the
  earlier Stage-C run's 15,826,461,120 B for the same 2,661 distinct tensors because the shared B2
  reader counts every read it performs, and the forced-operand provenance verifier re-reads the
  host-routing operands before each forced launch; the tensor count is unchanged.
- Seam labels: the log prints the seam lines exactly once (router; gate/up GEMM; SwiGLU and the
  activation quantiser; HSI candidate merge; attention operand preparation; attention side channel;
  Engram write; routed/shared elementwise sum; head and sampling) plus the sink line, all before
  the numbers; the audit line follows the numbers. The head seam is labelled NOT EXERCISED
  (criterion 1 stops at the boundary table; stage 5 owns the head).

### 13.7 What this closes, and what it does not

This is the Phase-1 health-check column: the device path uses FP4 main/window KV (D4) and
bf16/quantised arithmetic where the host reference is f32 with no GEMM activation quantiser and no
bf16 writes, so the column measures attribution and gross mismatch. **Criterion 1 is NOT closed on
this run.** The format-faithful exactness run is the later one after Phase 2 (release-faithful
arithmetic), and its bars may not be invented to make a row pass. Task-3 has since landed the first
half: the window-KV device path now accepts the engine's released FP8 E4M3 + E8M0 window format
(`--stage2-kv-format engine`), reproduces the engine's bf16 storage round on both sides, and measured
163,834/163,840 exact (99.9963%), max abs 9.765625e-4, max ULP 1 at positions 0/1/127/128/129 (P0,
P1, P127 and P128 all 32,768/32,768 exact; P129 32,762/32,768), against the pre-declared 99.5% /
1.6e-2 class (final logs `phase1-task3-abi-stage2-engine-a/-b`, 14,127 B
each, sha256 `3abfdd7d...`/`1577daa2...`). That engine-format column is a format-fidelity tolerance class, not the criterion-1
closure: the engine reference forward still keeps f32 dequant, so its bf16 arithmetic is not
reproduced. The FP4 default path was re-measured on the merged chain at 163,838/163,840 exact, max
abs 2.4414063e-4, max ULP 1, with every per-position line identical between the merged-head
old-kernel binary and the task-3 binary; the equality line is re-run on the final binary in
`phase1-task3-abi-stage2-fp4-a/-b` (11,647 B each, sha256
`4613aac5...`/`b455cc57...`, PASS), and the earlier 9.765625e-4 is the pre-D8 chain and is
labelled historical. The round-2 review fixed the stage-7 and stage-9 K5 sites that still used the
retired 15-parameter launch array: every flash_mla launch now goes through one typed 19-parameter
builder (`FlashMlaParams`) and two launch wrappers, pinned by the
`m4_abi::every_flash_mla_launch_site_uses_the_typed_parameter_builder` source-text test with its
mutation. The ABI-fixed binary `14e0266f...` (the build of head `c520c93`, command
`./scripts/build-aarch64.sh --example=m7_layer0` run in `/home/user/wt/task3-window-kv`; the
aarch64 hash is path-dependent, and the node copy of the evidence binary hashes to the same
`14e0266f...`) re-ran the stage-7 layer-0 replay (the 18-row measurement table and all 18
`single_layer_delta` lines are bit-identical to the merged B2 log, headers aside) and the
stage-9 smoke (2 rows, 1 run: PASS framing, 0 FAIL; row 1 `residual_out` max-abs 1.411274 equals
the merged Stage-C row-1 number) in logs
`phase1-task3-abi-stage7-b2compare` (24,005 B, sha256 `fb87a5e4...`) and
`phase1-task3-abi-stage9-smoke` (26,155 B, sha256 `b14a6036...`). The
discrete expert flips above are divergence events, not bugs, and they are reported per layer rather
than folded into a smooth bar. What this run does close is the device half's 40-layer column as a
deterministic, self-auditing, format-compatible measurement: the same token the host reference
captured, run end to end through the released checkpoint with the released seams labelled, twice,
bit-identically, with its own v1 pair proving the writer.

One behavioural refinement reaches the shared B2 code: the forced-control verdict now takes the two
independently computed routed-combine references as evidence (section 13.5). B2's own row-0 case is
unchanged - its references differ, so the copy mutation still aborts - and the CPU suite
(`crates/sparkflash-kernels/tests/b2_harness_offline.rs`, 14 tests) covers the allowed zero-effect
flip, the copy abort, the broken-launch abort (references identical but bytes differ) and the
identical-ids rules, so the refinement is exercised without a device window. The operand-provenance
verifier is unchanged and still runs before every forced launch.
