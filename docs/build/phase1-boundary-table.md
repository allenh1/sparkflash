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
finite block; the tests assert that. The block is 32 **contiguous elements along `dim`**, one block
per 32 channels of one token.

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
2026-10-17 on `node0` (aarch64) with the release build of the example from this MR's branch
(the example is introduced by this MR; `main` at `e0c1502` does not have it), against the released
snapshot in the node's Hugging Face cache:

    boundary_dump --checkpoint <snapshot> --out phase1-host-token0

It wrote 40 rows of 18 tensors, a 34,767-byte manifest and a 13,537,920-byte blob, which is exactly
the section 4 size table (40 x 338,448 bytes); `boundary_diff` of the pair against itself reports all
40 rows bit-identical. The one-token report on the same snapshot printed:

* 95,819 tensors verified, 4 Engram tensors paged (the two released Engram tables, weight and scale);
* 40 rows, all host-vs-host bit-exact, two runs byte-identical over the boundary tables and the
  129,280 logits;
* every logit finite, argmax 5, top-5 [5: 13.230665207, 201: 12.492528915, 372: 11.897726059,
  30: 11.392500877, 1897: 11.015835762];
* audit plan=95,819 read=2,643 unread=93,176 partitioned as [unselected-expert 90,720,
  DSpark-stage 2,401, text-path 40, indexer 15];
* wall clock 2:20.25 and peak RSS 6,689,384 KiB (6.4 GiB) for the whole run.

The artifacts stay on the node: `/home/user/hostrun/phase1-host-token0.manifest` and `.blob`,
and `/home/user/hostrun/host_report_token0.txt`; they are not committed here (the blob alone is
13 MB). This is the host column of criterion 1 and the reference half of criterion 3. It is not the
device column: no device table has been captured, so nothing here says the device path matches this
table, and the bars of section 5 remain to be measured against it.

