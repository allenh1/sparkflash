# Layer-0 MoE stage: kernel sequence, exactness bars, and the evidence contract

Owner: kernel-lead (task-11). Consumers: device-lead's four-rank forward loop (filters wiring bugs from
kernel mismatches), Lead/QA window adjudication. Status: contract pinned here and implemented by `crates/sparkflash-kernels/examples/m26_moe_stage.rs`
(device run plus `--twin-only` host twin; landed in this branch). The host twin starts from the loop's
H3 `pre_ffn_hidden` bytes, so the engine host seams stay engine code; the twin validates the H4 rule
bit-exactly and the kernel stages 1-6. No new kernel and no new baseline entry: every launch below is an existing
twenty-entry-baseline cubin at its current pin.

## 1. The sequence the four-rank loop runs, per layer, per rank

Host seams (engine host code, exact f32; no cubin launch):

| # | seam | call | output |
| --- | --- | --- | --- |
| H1 | hc_ffn mixes | `mhc::hc_mixes(config, post_attention_hidden, layers.N.hc_ffn_fn [24,20480] F32, .hc_ffn_scale [3] F32, .hc_ffn_base [24] F32, &mut ffn_mixes)` | `ffn_mixes.pre/post/comb` |
| H2 | ffn pre-mix | `mhc::hc_pre(config, attn_mixes.pre, post_attention_hidden, &mut ffn_in)` | `ffn_in` f32 [tokens][5120] |
| H3 | ffn norm | `linalg::rms_norm(ffn_in, layers.N.ffn_norm.weight, 5120, norm_eps)` | `pre_ffn_hidden` f32 [tokens][5120] |
| H3b | router input round | BF16 RNE of `pre_ffn_hidden` (`m4_quant::bf16_bits` per value; the engine device chain stages the same `x_bf16`) | router x BF16 [tokens][5120] |
| H4 | activation quant | `quant::quantize_activation_fp8(pre_ffn_hidden row)` per token (D8, f32 in) | E4M3 codes [tokens][5120], E8M0 scales [tokens][160] |

H2 reuses the attention predictor's pre mix (`attn_mixes.pre`) already computed by the mHC entry
stage; it is NOT `ffn_mixes.pre`. H4's operand is the gate/up A operand. `m4_quant::act_quant_fp8_row`
is the kernel-crate twin of H4's rule (same scale code, same floor, same codes).

Device launches (per rank, per layer; 5 launches):

| # | kernel (cubin) | pin | launches/rank | grid, block | args |
| --- | --- | --- | --- | --- | --- |
| K1 | `moe_router_sm121` | 25,432 B / `42f6bbdc...b688cc` | 1 | `router_launch::launch` (module-owned) | `RouterLaunchParams`: x(H3b BF16 bytes), gate(layers.N.ffn.gate.weight BF16 [384,5120]), bias(F32 [384]), ids u32[tokens*6], weights f32[tokens*6], scores f32[tokens*384], tokens, trace_ring=0, layer_id=N, step_seq |
| K2 | `routed_gate_up_sm121` | 15,672 B / `12ddc097...5aac3d` | 2 (gate=w1, up=w3) | (tokens*6, 20, 1), 256 threads | `GateUpLaunchParams`: act_fp8(H4), act_scales(H4), w_fp4 = this rank's 576-row slice rows 576r..576r+575 of released w1/w3 I8 [2304,2560], w_scales = same rows of [2304,160], expert_ids = router ids, out BF16 [tokens*6][640], tokens, trace_ring=0, layer_id=N, step_seq=r |
| K3 | `swiglu_mxfp8_quant_weighted_sm121` | 13,000 B / `0e602daf...15c0fe` | 1 | ((tokens*6*640/32).div_ceil(8), 1, 1), 256 threads | 11 words: gate K2 out, up K2 out, router weights buffer IN PLACE, codes out [tokens*6][640], scales out [tokens*6][20], rows=tokens*6, cols=640, swiglu_limit bits (10.0f32.to_bits), trace_ring=0, layer_id=N, step_seq=r |
| K4 | `b12x_mxfp4_moe_stored_sm121` | 11,544 B / `c452a9aa...a35bb7` | 1 | (tokens, 160, 1), 256 threads | 10 words: act_fp8 K3 codes (640/row), act_scales K3 scales (20/row), w_fp4 = rank's stored w2 codes base (288 B/row), w_scales = rank's stored w2 scale base (18 B/row), expert_ids, slot_acc f32 [tokens*6][5120] caller-zeroed ONCE, tokens, trace_ring=0, layer_id=N, step_seq=r |

K4 weight bytes: released `layers.N.ffn.experts.e.w2.weight` I8 [5120,1152] and `.w2.scale` F8_E8M0
[5120,72]. Rank r's stored slice is the per-row prefix at byte offset r*288 (codes) and r*18 (scales),
i.e. two separate tensors [selected_experts][5120][288] and [selected_experts][5120][18]. Never compose
the 306-byte interleaved row; never pass one base pointer for both. Grid (tokens,160,1), block
(256,1,1), 10 argument words is confirmed (the Lead's note is correct).

Host reduction (no kernel):

| # | step | call | output |
| --- | --- | --- | --- |
| R1 | rank finalise | `mxfp4_ref::moe_combine_routed_f32(slot_acc, ids, tokens, 5120)` | f32 [tokens][5120] (per-slot bf16 round, ascending expert-id add; routing weight already folded by K3) |
| R2 | canonical reduce | payload = bf16 RNE of each rank's R1 f32; `sparkflash_comm::reduce::canonical_all_reduce_into` | routed delta BF16 [tokens][5120] (f32 ascending-rank sum seeded 0.0, one bf16 RNE round) |
| H5 | shared expert | `moe::add_shared_expert(config, prefix, pre_ffn_hidden, src, limit, &mut scratch, &mut ffn_out)` on top of the routed delta (host, replicated; `ffn_out` already holds the routed delta) | `ffn_out` f32 [tokens][5120] |
| H6 | ffn hc_post | `mhc::hc_post(config, ffn_out, post_attention_hidden, ffn_mixes.post, ffn_mixes.comb, &mut residual_out)` | `residual_out` f32 [tokens][hc_mult][5120] |

The brief's arrows name K1..K4 plus R2 and H6; H5 (shared expert) is the engine's arithmetic between
R2 and H6 and must not be dropped (the reference adds it before the MoE bf16 boundary). If the loop
stages it separately, record it as its own seam with the same field names.

## 2. Exactness bars

Bit-exact (a mismatch is a finding, never a tolerance):

| stage | bar |
| --- | --- |
| H1-H4, H5, H6 | same engine host code, same input bytes -> byte-identical fields; H6 is exact vs the twin given the same `ffn_out` bytes |
| H3b | BF16 RNE of `pre_ffn_hidden`; the device kernel and the twin consume those bits |
| K1 | ids EXACT vs `router_ref`; two launches byte-identical |
| K2 | device gate and up BF16 BIT-IDENTICAL (ULP 0) vs `routed_gate_up_ref` on every tokens*6*640 value of both projections; padding columns 576..639 exactly zero; two launches byte-identical |
| K3 | E4M3 codes and ue8m0 scales BIT-EXACT vs `swiglu_ref::quantise_row_weighted` over the same 640-wide rows and weights; real 576-wide operand BIT-EXACT vs `mxfp4_ref::routed_activation_operand`; two launches byte-identical |
| K4 | rank f32 slot_acc BIT-EXACT vs `mxfp4_ref::moe_combine_rank_stored` on the same operands (tokens*6*5120 values); second launch from a freshly zeroed accumulator byte-identical |
| R1 | rank f32 routed BIT-EXACT vs the twin R1 |
| R2 | BF16 routed delta BIT-EXACT vs `canonical_all_reduce_into` over the four twin payloads |

Documented tolerances (only where the reference itself is approximate):

| stage | tolerance | why |
| --- | --- | --- |
| K1 scores | max abs <= 1e-6 and <= 8 ULP vs twin | device CUDA expf/log1pf vs host libm (m17a b33 class) |
| K1 weights | max abs <= 1e-6 and <= 16 ULP vs twin | same, after the selected-weight normalisation |
| R2 vs native full-K | routed delta relative L2 <= 5.0e-3 (engine-dev routed seam bar; host path measured 2.75e-3) | the distributed arm rounds each rank's partial to bf16 and sums four payloads; the native reference sums all 2304 K columns in f32 before the per-expert bf16 round. Structural to TP4's one-reduce-per-layer plan, not a kernel defect. |

Certification status of the four kernels (window evidence):
- `moe_router_sm121`: device-certified (m17a window b33: PASS 35 checks at tokens 8 and 130; ids exact, scores/weights inside the bars above).
- `routed_gate_up_sm121`: device-certified at the split-K pin (m1w window m10: BIT-IDENTICAL; m17a b33 at the pre-split-K pin).
- `swiglu_mxfp8_quant_weighted_sm121`: device-certified (m17a window b33; pin unchanged by the split-K rewrite).
- `b12x_mxfp4_moe_stored_sm121`: device-certified (m1w window m25: PASS at tokens 1 and 8, 11544 B / c452a9aa...).
- R2 and H1-H6 are host/transport code (canonical reduce already exercised by the layer-0 attention all-reduce), not kernel launches.

## 3. Input bundle (loop -> twin/harness)

The loop dumps, at layer 0, the two run-specific host-seam inputs (raw little-endian f32) and a
manifest; everything else is read from the released checkpoint:

```
moe-stage-input.json          # sparkflash-moe-stage-input/1
{
  "schema": "sparkflash-moe-stage-input/1",
  "layer": 0, "tokens": T, "hc_mult": H, "dim": 5120, "position": P,
  "post_attention_hidden": {"path": "post_attention_hidden.f32", "dtype": "f32-le", "bytes": T*H*5120*4, "sha256": "..."},
  "attn_pre_mix":          {"path": "attn_pre_mix.f32",          "dtype": "f32-le", "bytes": T*H*4,      "sha256": "..."}
}
```

The expanded bundle the harness consumes is the loop's host-seam output, dumped next to the manifest
(raw little-endian):

```
pre_ffn_hidden.f32  (T*5120 f32: H3's output; the host twin's starting point)
act_fp8.bin         (T*5120 E4M3: optional H4 codes; the harness recomputes them from pre_ffn_hidden
act_scales.bin      (T*160 E8M0:  and requires byte equality, so the H4 rule is pinned bit-exactly)
```

The raw inputs (`post_attention_hidden`, `attn_pre_mix`) stay in the record for audit; the engine host
seams H1/H2/H3/H5/H6 are engine code the loop runs, so the harness does not re-implement them. The
harness derives the router's H3b BF16 rows from `pre_ffn_hidden` itself and records the expected
record (mode `twin`) or the measured one (mode `device`); the loop's recorded `pre_ffn_hidden` digest
is the join key.


## 4. Evidence record (loop records it; twin/harness emit the expected record)

Schema `sparkflash-moe-stage-evidence/1`, one record per rank, field names fixed:

```
schema, stage="layer0-moe", rank, layer, tokens, ranks=4, experts=384, top_k=6, hidden=5120,
inter_per_rank=576, inter_padded=640, hc_mult, selected_rows=tokens*6, launches_per_rank=5,
launches_stage=20, all_reduce_calls_per_rank=1, weight_h2d_bytes_per_token=0, hc_mult=4 (width 20480),
cubin_pins: {moe_router_sm121, routed_gate_up_sm121, swiglu_mxfp8_quant_weighted_sm121, b12x_mxfp4_moe_stored_sm121} (bytes, sha256)
input: {post_attention_hidden_sha256, attn_pre_mix_sha256, ffn_in_sha256, pre_ffn_hidden_sha256,
        act_fp8_sha256, act_scales_sha256, post_attention_hidden_bytes, ffn_in_bytes,
        pre_ffn_hidden_bytes, act_fp8_bytes, act_scales_bytes}
router: {ids:[tokens*6], ids_sha256, weights_sha256, scores_sha256, ids_exact,
         scores_max_abs_delta, scores_max_ulp, weights_max_abs_delta, weights_max_ulp,
         second_launch_byte_identical}
gate_up: {gate_sha256, up_sha256, gate_up_mismatches_vs_twin, padding_nonzero,
          second_launch_byte_identical, launches=2}
swiglu: {codes_sha256, scales_sha256, code_mismatches_vs_twin, scale_mismatches_vs_twin,
         real_width_operand_mismatches, second_launch_byte_identical, launches=1}
combine: {arg_words=10, grid:[tokens,160,1], block:[256,1,1], slot_acc_f32_values=tokens*6*5120,
          slot_acc_sha256, mismatches_vs_twin, second_launch_byte_identical, launches=1}
reduce: {rank_routed_f32_sha256, payload_sha256, payload_sha256_by_rank:[4], routed_delta_sha256,
         rule="bf16 payload, ascending-rank f32 sum from 0.0, one bf16 RNE round",
         mismatches_vs_canonical_twin, native_relative_l2, native_relative_l2_bar=5.0e-3}
epilogue: {shared_expert_added, shared_expert_sha256, ffn_out_sha256, ffn_out_bytes,
           hc_post_output_sha256, hc_post_bytes=tokens*hc_mult*5120, hc_post_mismatches_vs_engine}
verdict: "PASS"|"FAIL", failed_checks:[...], pass_lines:[...]
```

All sha256 values are over the little-endian payload bytes (f32 arrays as f32 bytes, BF16 output as
packed u16 LE, code/scale operands as raw bytes). The ring/slot buffers are f32 before R1/R2.

## 5. m26 harness (device run and `--twin-only` host twin)

```
m26_moe_stage --checkpoint <snapshot dir> --cubin-dir <staged cubins> [--input <bundle dir>]
              [--tokens T] [--layer 0] [--twin-only] [--evidence <out.json>]
              [--dump routed_delta.bf16] [--repeat 0]
```

With `--input` it runs K1..K4 from the loop's expanded bundle and compares every stage to the same
kernel-crate twins the window uses; without `--input` it takes released `embed.weight` rows as the
declared pre-FFN stand-in (m17a's convention). `--twin-only` runs the CPU twin (no driver) and
records the EXPECTED evidence record; the device run records the measured one. Four rank shards run
in one process (the loop runs one shard per rank; the arithmetic is identical). Expected PASS lines:

```
m26 stage 1 router: PASS (<n> checks)
m26 stage 2 gate/up rank <r>: PASS (<n> checks)
m26 stage 3 swiglu rank <r>: PASS (<n> checks)
m26 stage 4 stored combine rank <r>: PASS (<n> checks)
m26 stage 5 canonical reduce: PASS (<n> checks)
m26 stage 6 native reference: PASS (<n> checks)
m26_moe_stage: PASS (<n> checks; layer 0; tokens <T>; 4 rank shards; 5 launches per rank; four pinned cubins)
m26_moe_stage: evidence <path>
```

Exit 0 on the final PASS line, 2 on any FAIL/no-go; the failure line names every failed check with
its measured detail. The per-check lines are the evidence: router ids exact / scores <= 1e-6 and
<= 8 ULP / weights <= 1e-6 and <= 16 ULP / two launches byte-identical; gate/up bit-identical, padding
zero, two launches byte-identical; swiglu codes+scales bit-exact, real-width operand bit-exact, two
launches byte-identical; stored combine slot accumulator bit-exact, second chain from a zeroed
accumulator byte-identical; payloads and reduced delta bit-exact; native relative L2 <= 5.0e-3. A
check that fails prints FAIL with its count and first mismatch, and the process exits 2 before the
final PASS line.
## 6. Adjudicating the wired loop against the twin (window recipe)

1. The loop dumps `moe-stage-input.json` + raw host-seam files and its own §4 record at layer 0.
2. Host twin: `m26_moe_stage --twin-only --checkpoint ... --input <expanded bundle> --evidence <expected.json>`.
   If the twin's `input.pre_ffn_hidden_sha256` or the H4 act bytes disagree with the loop's record,
   the seam that produced them is the bug (a wiring/host-seam bug), not a kernel mismatch.
3. Device: `m26_moe_stage --checkpoint ... --input <expanded bundle> --evidence <measured.json>`.
   Both runs are deterministic, so a loop field that differs from the measured record names the
   stage (router -> K1, gate_up -> K2, swiglu -> K3, combine -> K4, reduce -> R1/R2); the
   exact/tolerance classes above say whether the difference is a hard finding. m26 PASS proves the
   kernels on those bytes; the measured record proves the loop's staging.
4. Refusal distinction: a loop that refuses before K1 is a wiring refusal (no digests); a loop that
   launches K1..K4 but fails a bit-exact bar is a kernel mismatch and must be reported as a finding,
   never relaxed into a tolerance.
