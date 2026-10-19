# Native-arithmetic audit: the device forward chain against the released forward

Status: **docs-only audit; no code, kernel, ABI or binary changed, and nothing merged.** Branch
`task/native-audit` (parent `4f599b7`, the `--dump-logits` binary staged on the node).
The released forward was read read-only at `/home/user/dsv41-research/inference/`; the chain is
this branch's tree, which is byte-identical to `task/logits-dump` `4f599b7` for every file cited
here. No GPU window was held, nothing was built, and no measurement was re-taken: the numbers
quoted are the ones already in the record, cited where they live. This note edits no Lead-owned
document (`DECISIONS.md`, `STATUS-2026-10-19.md`, `docs/build/device-forward-chain.md`).

**Why this note exists.** D39 fixed the target: the chain must implement the checkpoint's *native*
arithmetic (the arithmetic the released forward actually executes, fp8/fp4 as trained), not a
higher-precision simulation of it, and acceptance is a numeric/distributional comparison with vLLM
as the behavioural oracle. This note answers the two questions that decision left open, stage by
stage: **what does the released forward compute, and what does the chain compute at the same
stage?** It also answers, for every seam the chain's own ledger *declares* as a host stand-in,
whether its dtype and rounding order match `model.py`.

Every claim below is labelled one of three ways:

- **SOURCE** - read from a cited line of the released forward or of this tree.
- **MEASURED** - an existing logged number, cited with the run/log it comes from.
- **INFERRED** - a consequence of the two above that no measurement isolates yet; the inference is
  spelled out and never reported as a measurement.

**Verdict legend** (used in section 3):

| verdict | meaning |
|:--|:--|
| `native` | the chain performs the same dtype, the same rounding points and the same quantiser rule at this stage; only f32 accumulation *association* differs, which native itself does not reproduce in software (the MMA decides it) |
| `native, minus a round` | same rule, but one bf16 rounding point the released forward performs is absent, so the chain is *more precise* here than native |
| `native, minus a quant` | a native activation quantisation is absent, so the chain is *more precise* than native |
| `native, plus a quant` | the chain quantises a tensor native leaves in bf16/f32, so the chain is *less precise* than native |
| `differs` | the arithmetic class or the rounding *structure* differs and neither "more" nor "less precise" summarises it |
| `order only` | same values up to f32 addition order and the f32 libm/rsqrt last bits |

Severity is about consequence, not about size alone: **HIGH** = a native quantisation replaced by a
different one (a different model); **MEDIUM** = a native quantisation or bf16 round omitted on a path
whose result is consumed *discretely* (selection, codes that are later compared) or an entire block
run in f32; **LOW** = one missing or extra bf16 rounding point, or an accumulation-order/association
difference.

---

## 1. Scope and method

In scope: the 40-layer backbone's `Transformer.forward` (`model.py:1245-1290`) and every arithmetic
stage under it - the mHC entry/post chain, the attention sublayer (five projections, two norms,
RoPE, window KV, compressor, indexer, format-2 attention body, inverse RoPE), the MoE (router, six
routed experts, shared expert), the two Engram layers, and the tail (final collapse, final norm,
head). Out of scope by construction: the vision tower, the MTP/DSpark heads (the chain refuses
them), the tokenizer and the sampling loop (`sample` is greedy at temperature 0, which is what the
comparison runs use).

Method: the released forward's arithmetic was read from `model.py` and `kernel.py` literally, with
the module dtype of each tensor determined from the `dtype=` argument of its constructor (and, where
a constructor passes none, from `default_dtype`, `model.py:26,1191`, which the released
`inference/config.json:49-50` sets to fp8/fp4). The chain side was read from the exact call path the
`hybrid_sequence` device run takes, not from the engine's f32 reference: the same tree also contains
that reference, and the two must not be confused (that confusion is exactly what criterion 1
measured, section 5).

A note on the released snapshot. The HF snapshot the runs read
(`models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10...`) is the inference-format
release: its `config.json` declares `quant_method: fp8`, `weight_block_size: [32, 32]`,
`scale_fmt: ue8m0`, `expert_dtype: fp4`, and its weight map carries `layers.N.attn.wo_a.scale`
beside `layers.N.attn.wo_a.weight` (verified in the local index: 96,085 entries). The released *code*
does not read that fp8 `wo_a` directly: `convert.py` dequantises `wo_a.weight` to bf16 and drops its
scale (`convert.py:157-173`), and `model.py` declares `wo_a` a bf16 `ColumnParallelLinear`
(`model.py:645-649`). The chain reads the fp8 bytes plus the scale (section 3.2). Because a ue8m0
scale is a power of two, the two weight representations hold **identical values** - that equality is
sourced, not measured, and it is why `wo_a` is the one projection where the chain's *weight* is
native.

---

## 2. The native arithmetic in one page

### 2.1 The quantisers and the formats

- **One dynamic activation rule.** `act_quant(x, 32, "ue8m0", e8m0fnu)` (`model.py:27-33,187,197`;
  `kernel.py:41-124`): per 32 contiguous elements, `amax` floored at `1e-4`, then
  `scale = 2^ceil(log2(amax/448))` (`fast_round_scale`, `kernel.py:36-38,78`), clamp to `+/-448`,
  round-nearest-even to E4M3. With `inplace=True` it writes the **dequantised** values back into the
  tensor (`out_dtype = in_dtype`, `kernel.py:52,81-95`), i.e. it returns bf16 values that are exactly
  `code * scale`.
- **fp8 weights** are E4M3 with one ue8m0 scale per 32x32 block (`kernel.py:208-300`).
- **fp4 routed weights** are E2M1 with one ue8m0 scale per 32 values along K (`kernel.py:478-560`);
  the activation operand for them is still **fp8**, block 32, the same rule as above
  (`model.py:184-195`).
- **fp4 for the compressed KV** uses **16-element groups and an E4M3 scale**, `s = e4m3(amax/6)` with
  the `6*2^-9` amax floor (`kernel.py:128-181`; called at `model.py:760` as
  `fp4_act_quant(latent, 16, True, scale_dtype=float8_e4m3fn)`), in place, so the cache holds bf16.
- **fp4 for the indexer q/k** uses 32-element groups and ue8m0, in place (`model.py:545-552`).

### 2.2 The GEMM arithmetic (what "native" means that software cannot reproduce)

Both GEMM kernels (`kernel.py:208-309` for fp8, `kernel.py:478-583` for fp4) do the same three
things: a hardware MMA accumulates one 32-element K block of **stored codes** into an f32 fragment
(`T.gemm`); then, per K block, `C_accum += C_local * scale_a * scale_b` (`kernel.py:269,553`); then
one f32-to-bf16 cast of the finished accumulator (`kernel.py:271,556`, output dtype from
`torch.get_default_dtype()` = bf16, `kernel.py:299,583`).

Two consequences matter for every verdict in this note:

1. **The scale multiplies are exact.** A ue8m0 scale is a power of two and every stored code has at
   most four significant bits, so both the code product and its scaling are exact in f32. The only
   native roundings in a GEMM are the f32 accumulator additions and the final bf16 cast.
2. **The accumulation order belongs to the hardware.** `T.gemm` lowers to an MMA whose reduction
   order inside the 32-value block is not a specification. No software kernel can be bit-identical
   to it, and the released forward is not bit-identical to itself across GPU architectures. This is
   why "token equality" could not have been an acceptance criterion (D39) and why this audit's
   verdicts are about *rule and dtype*, with accumulation association reported separately
   (`order only`, section 4 F8).

### 2.3 Where the released forward rounds to bf16, and where it stays f32

**Rounds to bf16** (every one of these is a rounding point the chain can miss):

| where | anchor |
|:--|:--|
| `RMSNorm.forward` output (input dtype) | `model.py:288-293` |
| `apply_rotary_emb` writes the f32 rotation back into the bf16 tensor | `model.py:392-409` |
| every quantised GEMM output (bf16 accumulator cast) | `kernel.py:271,556,299,583` |
| every `inplace=True` quantiser output (bf16 dequantised values) | `kernel.py:52,81-95,136,167-181` |
| attention output, and **each softmax probability before the P.V product** | `kernel.py:339,377-380,385-387` |
| `hc_pre` / `hc_post` outputs (`.to(x.dtype)`, `.type_as(x)`) | `model.py:958-966` |
| `MoE.forward` output (`y.type_as(x)`), each expert's own output implicit in its bf16 GEMM | `model.py:846-852,899-904` |
| Engram embedding row after dequantisation, and the Engram write | `model.py:318-319,364-365` |
| indexer scores and per-query weights (bf16 tensors, bf16 einsum) | `model.py:553-557` |

**Stays f32** (a chain that quantised these would be wrong the other way):

| where | anchor |
|:--|:--|
| `hc_mixes` projection, `rsqrt`, Sinkhorn | `model.py:946-956` |
| compressor pooling **and its f32 `wkv`/`wgate` Linears when ratio > 1** | `model.py:446-449,462-486` |
| gate scores, softplus-sqrt, bias (selection only), weights, `/sum`, `*route_scale` | `model.py:792-828` |
| SwiGLU clamps and `F.silu(gate)*up`, and the routing-weight multiply | `model.py:844-850` |
| the routed combine accumulator `y` (f32, experts added into it) | `model.py:895-904` |
| Engram key/value/gate | `model.py:355-365` |
| **the head**: f32 weight parameter and `F.linear(x.float(), self.weight)` | `model.py:1006-1012` |

### 2.4 Which module has which dtype (the native arithmetic's skeleton)

| module | dtype / rule | anchor |
|:--|:--|:--|
| `embed` | bf16 table, bf16 row | `model.py:152-181` (bf16 default; the snapshot stores bf16) |
| `attn_norm`, `ffn_norm`, `q_norm`, `kv_norm`, `norm` | bf16 weight, f32 compute, bf16 out | `model.py:281-294` |
| `wq_a`, `wq_b`, `wkv`, `wo_b` | fp8 W8A8, block 32, ue8m0 both sides | `model.py:636-643,650` + `181-205` |
| **`wo_a`** | **bf16 grouped einsum** (`ColumnParallelLinear(dtype=bfloat16)`) | `model.py:645-649,784-787` |
| `attn_sink` | f32 | `model.py:639` |
| `compressor.wkv`/`wgate` | **f32** when ratio > 1, **bf16** when ratio == 1; `norm` bf16 | `model.py:446-449,462-486` |
| `indexer.wq_b` | **fp8 W8A8** | `model.py:519` |
| `indexer.wk`, `weights_proj` | **bf16** Linear (plain `F.linear`) | `model.py:522,520` |
| `gate.weight` | bf16 (bf16 default; the snapshot stores bf16 - the chain's reader asserts 2 bytes/entry and b68 read it) | `model.py:797-800` |
| `gate.bias` | f32 | `model.py:800` |
| routed experts `w1/w2/w3` | **fp4** E2M1, ue8m0 per 32 along K | `model.py:873-882,184-195` |
| shared expert `w1/w2/w3` | **fp8** E4M3, 32x32 ue8m0 (default dtype) | `model.py:887` |
| Engram `embed` | fp8 table + ue8m0 block-32 row scales, dequantised per lookup to **bf16** | `model.py:296-326` |
| Engram `wkv` | **fp8** Linear (default dtype) | `model.py:345` |
| `head` | f32 parameter, f32 GEMM, f32 logits | `model.py:1006-1012` |
| `hc_*_fn/base/scale` | f32 | `model.py:939-943` |

The declaration is `inference/config.json:49-50` (`dtype: fp8`, `expert_dtype: fp4`) feeding
`default_dtype = torch.float8_e4m3fn` at `model.py:1191`; the bf16/f32 exceptions above are the
explicit `dtype=` arguments and `set_dtype(torch.float32)` blocks.

---

## 3. Stage map: native arithmetic against the chain, stage by stage

The chain column cites the code the **device run** executes. Where a stage is device, the engine's
f32 reference is *not* cited: it is a different implementation, and criterion 1 measured it, not the
device path.

### 3.1 Residual stream and the mHC chain

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | embedding gather | bf16 row (`model.py:174-181`) | host reads the bf16 row and widens to f32 (embed-gather seam; values are bf16-exact) | `native` | NONE | SOURCE; declared host seam |
| 2 | `hc_mixes`: projection, `rsqrt`, f32 Sinkhorn | f32 GEMM over the bf16 stream widened to f32; f32 `rsqrt`; f32 Sinkhorn (`model.py:946-956`; `kernel.py:407-475`) | device `mega_mhc_sm121` step 0: f32 scalar projection, f32 rsqrt, f32 Sinkhorn | `native` | NONE | MEASURED: projection 5.96e-8 worst abs against the engine twin and against the released projection (D23 stage 0, B0 record) |
| 3 | `hc_pre`: collapse the hc copies | f32 sum then **`.to(x.dtype)` bf16** (`model.py:958-961`) | device entry keeps the collapse in f32 registers (`kernels/mega_mhc_sm121.cu:333-348`) | `native, minus a round` | LOW | SOURCE; F5 |
| 4 | `attn_norm` | RMSNorm: bf16 in, f32 compute, **bf16 out** (`281-294`) | device: f32 norm folded into the entry capture `xhat_normed` (`mega_mhc_sm121.cu:337,340-348`); the f32 vector is what the chain consumes | `native, minus a round` | LOW | SOURCE; the capture also feeds the host quantisers (F5) |
| 5 | `hc_post`: expand and mix the residual | f32 compute then `.type_as(x)` bf16 (`model.py:963-966`) | device `mega_mhc_sm121` step 1: f32 compute, bf16 write | `native` | NONE | MEASURED: entry twin capture exact (`capture_exact`, `device_chain.rs:1416,1605`); the post-attention delta is measured against the f32 reference (section 5) |
| 6 | FFN output staged into the post | `y.type_as(x)` bf16, then `hc_post` uses it | host FFN output BF16-staged before the device post (`write_layer_y_bf16`, call site `device_chain.rs:1198`) | `native` | NONE | SOURCE + ledger line "deferred Y, BF16-staged" |
| 7 | B/C/A coefficients between the deferred epilogue and the next entry | f32 coefficients carried in registers | read back f32 and re-uploaded f32 (`device_chain.rs:1184-1196`) | `native` | NONE | SOURCE: declared coefficient round-trip, f32 both sides |

### 3.2 Attention sublayer

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 8 | `wq_a` | fp8 W8A8 on the **bf16** attn_norm output, bf16 out (`model.py:636,187-205`) | device dense B12x MXFP8: engine ceil-rule quantiser over the **f32** capture (`attention_device.rs:1052`), 32x32 ue8m0 weight grid, f32 block accumulation, bf16 out (`kernels/b12x_mxfp8_dense_sm121.cu:93-113`) | `native` rule; the quantiser's input has one bf16 round missing (F5) | NONE + LOW(F5) | MEASURED: stage 8 dense 16,384/16,384 bit-exact against the in-crate `mxfp8_ref` on the same bytes; 5,120/5,120 codes and 160/160 scales at stage 0 (D23) |
| 9 | `q_norm` | RMSNorm, bf16 out (`model.py:970`) | host f32 `query_path_with` (`engine/forward/attention.rs:1658-1720`), no bf16 round | `native` (task-10 narrows the norm output before the device wq_b staging) | NONE | SOURCE; test `the_host_q_norm_output_is_a_bf16_tensor_before_the_wq_b_quantiser` |
| 10 | `wq_b` | fp8 W8A8 on the **bf16** q_norm output (`model.py:971`) | device dense MXFP8; activation quantised from the f32 q_norm vector (`attention_device.rs:1083`) | `native` rule + F5 | NONE + LOW(F5) | MEASURED: `wq_b` reproduces `mxfp8_ref` 0/32,768, max_abs 0.0 (STATUS-2026-10-19 section 3) |
| 11 | q RoPE | f32 rotation from the bf16 value, written back into the bf16 tensor (`392-409`) | host f32 `rotate_rows` on bf16-exact values; bf16 when uploaded to the body (`attention_device.rs:1289-1292`) | `native` (one rounding, at the same point) | NONE | SOURCE; task-11 aligned the table arithmetic to the reference f32 YaRN (F8); the cos/sin last bit is the platform libm, declared |
| 12 | `wkv` | fp8 W8A8 on the bf16 attn_norm output | device dense MXFP8 from the f32 capture (`attention_device.rs:1110`) | `native` + F5 | NONE + LOW(F5) | MEASURED as #8 |
| 13 | `kv_norm` then window RoPE | RMSNorm bf16 out; rope in place bf16; then `act_quant(kv, 32, ue8m0, inplace=True)` (`model.py:700-707`) | host f32 `window_kv_with` (`attention.rs:1723-1760`): f32 norm, f32 rope, fp8 round trip with the ceil rule (`round_trip_rows`, `attention.rs:181-233`) | `native` (task-10 added the two rounds) | LOW | MEASURED before the rounds: stage 1 reproduced 5,119/5,120 codes and 160/160 scales (layer 0 token 0), 5,118/5,120 at layer 4 token 0 (D23) - the 1-2 code gap the rounds close; device re-measure is a window leg |
| 14 | window row construction (host) | the quantiser sees a **bf16** row (the in-place result) | `engine_window_row` bf16-rounds the row before quantising and requires the two quantisers to agree (`cuda_harness.rs:220-241`) | `native` (the quantiser input is bf16, exactly as native) | NONE | SOURCE; the row it rounds is the chain's f32 rope output, see #13 |
| 15 | compressor (ratio > 1) | f32 `wkv`/`wgate`, f32 softmax pooling, then `norm(kv.to(dtype))` = **bf16 latent** (`model.py:446-486`) | host `compressor_step`: f32 matmuls (`attention.rs:929-934`), the same pooling (`pool_group`), f32 rms_norm (`attention.rs:990`) | `native` (task-10: narrowed before and after the norm) | NONE | SOURCE; the f32 pooling matches native's f32 pooling; test `the_compressor_latent_is_a_bf16_tensor_before_the_fp4_quantiser` |
| 16 | compressor (ratio == 1) | **bf16** `wkv` Linear then norm (`model.py:446-449,461-463`) | host f32 matmul then f32 norm (`attention.rs:913-921`) | `native` (task-10: the wkv output and the norm are narrowed) | NONE | SOURCE; same test as #15 |
| 17 | compressed KV: RoPE then fp4 (block 16, E4M3 scale) in place | bf16 latent -> rope bf16 -> `fp4_act_quant(..., 16, True, e4m3)` -> bf16 cache (`model.py:757-761`) | host `write_compressed_cache`: f32 rotate then `round_trip_rows(Element::Fp4, ScaleRule::E4M3Six)` with the `6*2^-9` floor and block 16 (`attention.rs:1869-1930,207-211`) | `native` (task-10 narrows the rope writeback; the latent was already bf16 from #15) | NONE | SOURCE: block size, scale dtype and floor all match `kernel.py:128-181` |
| 18 | indexer k: `wk` (bf16 GEMM), `k_norm`, rope, fp4 block 32 ue8m0 in place | bf16 Linear -> bf16 norm -> bf16 rope -> fp4 (`model.py:540-546`) | host `write_index_keys`: f32 matmul over the bf16 weights widened to f32, f32 norm, f32 rope, fp4 round trip (`attention.rs:1799-1861`) | `native` (task-10 narrows the k_norm output and the rope writeback before the fp4 round trip) | NONE | SOURCE; test `the_quantiser_seams_narrow_the_host_value_to_bf16_first` |
| 19 | indexer q: `wq_b` **fp8 W8A8**, rope, fp4 block 32 | `ColumnParallelLinear` fp8: `act_quant` on the bf16 q_norm output, fp8 GEMM, bf16 out, then rope, then fp4 (`model.py:519,550-552`) | host `index_source_indices`: **f32 matmul over dequantised fp8 weights with no activation quantisation**, f32 rope, fp4 round trip (`attention.rs:1045-1080`) | `native, minus a quant` (task-10 narrows the rope writeback; the absent W8A8 activation quant is the remaining gap) | MEDIUM (F3) | SOURCE; unmeasured |
| 20 | indexer weights and scores | `weights_proj` bf16 GEMM x float -> bf16; `einsum` bf16 x bf16 -> bf16; `relu`, `* weights` -> bf16, `sum(head)` -> bf16 (`model.py:553-557`) | host f32 throughout: f32 `index_weights` (`attention.rs:1771-1788`), f32 dot and f32 head sum (`attention.rs:1102-1119`) | `native, minus rounds` | LOW (F3) | SOURCE; unmeasured |
| 21 | two-level top-k and reachability | on the **bf16** scores (`model.py:560-580,583-611`) | host on f32 scores; the two levels, the shift into the shared index space and the tie rule are reproduced (`attention.rs:1120-1205,1214+`) | `native` structure + F3 score dtype | LOW-MEDIUM (F3) | SOURCE; a flip needs a near-tie |

### 3.3 Attention body (format 2)

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 22 | q, kv operands | q bf16, kv bf16 (window rows dequantised from fp8, compressed rows from fp4), sink f32 (`kernel.py:311-338`) | q bf16; window row `bf16_rn(code*scale)` (`kernels/flash_mla_csa2_seq_sm121.cu:134-141`); compressed `fp4_code * e4m3_scale` in f32, which is bf16-exact (`kernel.py:167-181` produces the same values); sink f32 | `native` | NONE | SOURCE + MEASURED: the body twin feeds the host body the same staged bytes: 0 mismatches on 66/120 layer-position pairs, worst <= 7.811069e-3 on the rest (measured before the task-7 F2 round; a post-F2 body twin now reports that round by design) |
| 23 | score dot, f32 softmax, sink in the denominator after the max | f32 (`kernel.py:361-387`; sink at `383`) | f32 score, f32 max, f32 exp, sink added to the sum after the max (`flash_mla_csa2_seq_sm121.cu:319-365`) | `native` | NONE | SOURCE; MEASURED twin as #22 |
| 24 | **each softmax probability rounded to bf16 before P.V** | `acc_s_cast = BF16`; `T.copy(acc_s, acc_s_cast)`; `T.gemm(acc_s_cast, kv_shared, acc_o)` (`kernel.py:339,377-380`) | engine reference body now casts the probability to bf16 before the value product, denominator stays f32 (task-7, `sparse_attention`); the device body still accumulates f32 (F2 variant pending) | `native` in the host twin, device `native, minus a round` | MEDIUM (F2) for the device | test `the_body_probabilities_are_bf16_rounded_before_the_value_product` (fails without the cast); the kernels-crate union twin (`mla_ref.rs`) carries the same rule, so `format_two_union_equals_the_engine_reference_on_the_attended_multiset` stays bit-for-bit and the twin's own union tests moved with it |
| 25 | output | f32 divide, bf16 out (`kernel.py:385-390`) | `__float2bfloat16_rn(acc / s_sum)` (`flash_mla_csa2_seq_sm121.cu:393`) | `native` | NONE | SOURCE |
| 26 | inverse RoPE on the attention output | in place on the bf16 output (`model.py:775`) | host f32 `rotate_rows(..., true)` (`attention.rs:2308-2318`); the native wo_a arm narrows its activation to bf16 (task-6, F1) | `native` for the native arm; the MXFP8 comparison arm stays f32 by design | NONE (F1) | SOURCE; task-6 `wo_a path` ledger |

### 3.4 Output projections

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 27 | **`wo_a`** | **bf16 grouped einsum over the bf16 attention output** (`model.py:645-649,784-787`) | device `b12x_mxfp8_wo_a_sm121`: the activation is **quantised to fp8** (block 32, ue8m0 ceil) against fp8 weights with 32x32 ue8m0 (`attention_device.rs:1145-1207`; `kernels/b12x_mxfp8_wo_a_sm121.cu:53-110`), bf16 out | **`native, plus a quant`** | **HIGH (F1)** | SOURCE; the weight values are identical to native's bf16 (section 1); unmeasured |
| 28 | `wo_b` | fp8 W8A8 on the bf16 wo_a output (`model.py:650,260-279`) | device dense MXFP8; the activation is the device wo_a's bf16 output read back (`attention_device.rs:1208-1231`), i.e. bf16-exact, as native | `native` | NONE | SOURCE; MEASURED as #8 |

### 3.5 MoE

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 29 | router | bf16 gate weight, x widened to f32, f32 GEMM, f32 sqrt-softplus, f32 bias for selection only, f32 weights, `/ (sum + 1e-20)`, `* 1.5` (`model.py:792-828`) | device `moe_router_sm121`: bf16 x staged (the f32 capture is rounded at staging, `moe_device.rs:553-556`), bf16 gate weight (asserted 2 bytes/entry, `moe_device.rs:540-545`), f32 accumulate/scores/weights | `native` | NONE | MEASURED: router twin `equal=true` at all 120 printed layer-position pairs (3 positions x 40 layers), scores_max_abs <= 5.96e-8, weights_max_abs <= 2.98e-8 - against the engine twin, i.e. self-consistency, not native |
| 30 | routed gate/up | fp4 weights, **fp8 activation on the bf16 ffn_norm output** (`model.py:846-847,184-195`) | device: activation quantised host-side from the f32 ffn capture (`moe_device.rs:557`), then the MXFP4 gate/up GEMM | `native` rule + F5 round | NONE + LOW(F5) | MEASURED: gate/up twin (D23 stage 8: swiglu 2,560/2,560 codes, 80/80 scales) |
| 31 | SwiGLU clamps, raw and weighted | f32 clamps (asymmetric: gate upper only), `F.silu(gate)*up` f32; with weights: `weights * x` then **`.to(bf16)`** before the w2 quantiser (`model.py:845-852`) | routed: `swiglu_mxfp8_quant_weighted_sm121.cu:101-110` - same clamp, f32 silu, **bf16 round of the product**, then the ceil-rule quantiser: native to the letter. Shared: `swiglu_mxfp8_quant_sm121.cu:133-140` - same clamp and silu but **no bf16 round before its quantiser** | routed `native`; shared `native, minus a round` | routed NONE; shared LOW (F6) | routed MEASURED (stage 8); shared SOURCE, unmeasured |
| 32 | routed w2 + combine | each expert's fp8 GEMM returns **bf16**, added into an f32 `y` in ascending expert id (`model.py:846-852,895-903`) | device rank chain: per rank `combined = bf16(previous_rank + sum over that rank's slots)` (`kernels/b12x_mxfp4_moe_sm121.cu:150-197`; host serialisation `moe_device.rs:870-943`) | `differs` | LOW (F7) | MEASURED: device chain == its own chained-bf16 twin 20,480/20,480 (D23 stage 3); the twin is the chain's rule, not native's |
| 33 | shared expert | fp8 W8A8; activation = **bf16** ffn_norm output; `F.silu*up` -> `.to(bf16)` -> w2 | device fp8 weights; activation quantised from the f32 capture (`moe_device.rs:1194`); w2 dense MXFP8; host f32 add into the routed row | `native` + F5 + F6 | NONE + F5 LOW + F6 LOW | SOURCE |
| 34 | MoE output rounding and residual add | `y.type_as(x)` bf16 after the shared add (`model.py:903-904`) | host f32 add, then BF16-staged into the device post (#6) | `native` | NONE | SOURCE + ledger line |

### 3.6 Engram (layers 1 and 14)

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 35 | table lookup and dequantisation | fp8 row x ue8m0 block-32 scale, then **`.to(torch.bfloat16)`** (`model.py:312-322`) | host `read_table` / `read_paged_table` dequantise and cast to bf16 (`engram.rs:476-487,520-522`) | `native` (the cast is written and is exact for the format) | NONE (was MEDIUM F4) | PINNED (task-9): twin `the_row_dequant_is_the_f32_reference_rounded_to_bf16` moves 0 of 256 values on all eight captured rows; E4M3 x 2^k is bf16-exact |
| 36 | `wkv` | **fp8 W8A8** on the bf16 embed row (`model.py:345,184-195`) | host `wkv_native` reads the raw fp8 codes and 32x32 E8M0 grid and applies the block scale to the block dot in the device kernel's reduction order (`engram.rs:591-660`) | `native` | NONE (was MEDIUM F4) | PINNED (task-9): kernels `engram_embed_sm121` / `engram_wkv_sm121` in `baseline/kernel_baseline.txt`; twins `the_wkv_block_dot_is_the_kernels_shuffle_tree`, `the_wkv_projection_is_the_released_w8a8_block_rule`; CI fixture moved 2.21e-5 |
| 37 | key/value/gate and the write | key/value/gate f32 from the bf16 `wkv` output; `(h + gate*value).to(x.dtype)` bf16 (`model.py:355-365`) | gate over the bf16-rounded stream with bf16-exact key/value, one bf16 cast in `engram_write` (`engram.rs:749-780,806-830`) | `native` | NONE (was NONE + MEDIUM F4) | PINNED (task-9): `engram_gate_write_sm121` in the baseline; twin measures 5.883e-5 gate / 3.3e-3 relative write against the old f32 seam |

### 3.7 Tail

| # | stage | native | chain | verdict | severity | evidence |
|:--|:--|:--|:--|:--|:--|:--|
| 38 | final collapse | `hc_pre` -> bf16 (`model.py:1268`) | host f32 `mhc::hc_pre` (`engine/forward/mod.rs:1499`) then **BF16-staged** before the device norm (`moe_device.rs:1379-1382`) | `native` | NONE | SOURCE; the same bf16 cast native performs |
| 39 | final norm | RMSNorm bf16 in, bf16 out (`model.py:1269`) | device `rmsnorm_sm121`, bf16 out (`moe_device.rs:1363-1420`) | `native` | NONE | MEASURED: `final norm diag: rows=1 cols=5120 bf16_mismatches=0` at every position of the b68 run (against the host twin) |
| 40 | head | f32 weight (up-cast from the bf16 checkpoint), `F.linear(x.float(), weight)`, f32 logits, argmax ties to the first entry (`model.py:1006-1012,1289`) | device certified head: bf16 weight, f32 dot, f32 output (`kernels/cert_lm_head_sm121.cu:1-45`; `crates/sparkflash-kernels/src/lm_head_ref.rs:99-107`), ties to the lowest entry | `native` (f32 GEMM over bf16-valued operands) | NONE | MEASURED: the device reduce equals the host four-shard assembly at every position (STATUS-2026-10-19 section 3); the pruning certificate is proved in `lm_head_ref.rs` |

Reading the table by verdict: **no stage is wrong in its native *rule*** - every quantiser's block
size, scale dtype, ceil rule, clamp and floor matches, and so does the f32-vs-bf16 placement of every
module in section 2.4. The differences are in rounding points and in one quantiser: row 27 (`wo_a`,
F1), row 24 (the P.V probability round, F2), rows 19-21 (the indexer, F3), rows 35-37 (Engram, F4),
the seams listed under F5, row 31's shared half (F6), row 32's rounding structure (F7) and the F8
association class, which is not a deviation class at all. Everything else is native in rule *and*
rounding point - and of those, the ones marked MEASURED carry a same-operand or released-record
comparison, not just a source reading.

---

## 4. The deviation register

Each entry: what native does, what the chain does, the direction, why it matters downstream, what is
measured, and what would isolate it.

### F1 - `wo_a` is MXFP8 in the chain and bf16 in native. **HIGH.**

Native: `wo_a` is a `ColumnParallelLinear(..., dtype=torch.bfloat16)` (`model.py:645-649`), applied as
a grouped `einsum` (`model.py:784-787`) whose comment states the reason: "convert.py dequantizes it to
bf16; an fp8 grouped GEMM would halve the memory" (`model.py:784`). No activation quantisation
happens anywhere on that path: the operand is the bf16 inverse-RoPE'd attention output, the product
is an f32 MMA over bf16 values, the result is bf16.

Chain: `AttentionDeviceBackend::wo_a` first **quantises the activation to fp8** with the engine ceil
rule (`attention_device.rs:1145`) and runs a W8A8 grouped kernel against the fp8 `wo_a` weights
(`kernels/b12x_mxfp8_wo_a_sm121.cu:53-110`). The *weights* are the same numbers (section 1), so the
deviation is exactly one extra activation quantisation: E4M3 with a 32-block power-of-two scale, i.e.
a relative perturbation of up to 2^-4 per element, applied where native applies none.

Why it matters: it is the only projection where the chain's arithmetic is *coarser* than native, and
it sits on the residual path of every layer. It is the deviation most likely to make the chain's
logits differ from a native stack systematically rather than stochastically.

Measured: **not isolated.** The stage twin's layer-0 post-attention delta (9,627/20,480 bf16 codes,
max_abs 3.120759e-2) is the *device path against the f32 reference*, and the four native-matching
projections account for most of it by construction; no measurement separates `wo_a`'s share.

Isolation (proposal, not actioned): run the engine's *host* `wo_a` projection **for that one
projection only** while every other stage stays device. That is not the D37 mode (which made all
five host-f32) and it is native arithmetic (bf16 einsum), so it would measure F1 directly: compare
the device `wo_a` output against an f32 einsum over bf16 operands on the same captured attention
output. One harness flag, one device window run.

### F2 - the attention body does not round each softmax probability to bf16 before P.V. **MEDIUM.**

Native: `acc_s_cast` is a bf16 shared tile (`kernel.py:339`); `T.copy(acc_s, acc_s_cast)` rounds
every probability (`kernel.py:377`) and the P.V GEMM consumes the bf16 probabilities
(`kernel.py:380`). The chain keeps the probabilities in f32 through the value accumulation:
`acc += s_scores[slot] * value` (`kernels/flash_mla_csa2_seq_sm121.cu:375-389,475-492`).

Direction: the chain is *more precise* here, by up to 2^-9 relative per probability. It is invisible
to every existing twin, because all of them are the engine's reference body, which also keeps f32
probabilities: the "body matches on identical staged operands" result (0 mismatches on 66/120 pairs,
<= 7.8e-3 elsewhere) measures the kernel against the reference, not against native. This is the one
*body* stage where the chain and its reference share an omission native does not have.

**Status (task-7).** The engine's reference body now rounds: `sparse_attention` casts every
softmax probability to bf16 (`quant::bf16_round`) before the P*V accumulation and keeps the
denominator on the f32 probability, exactly the two values the reference kernel uses
(`kernel.py:339-341` for the cast, `342-353` for the f32 sum). The pinned test
`the_body_probabilities_are_bf16_rounded_before_the_value_product` spells the arithmetic out and
fails when the cast is removed (verified), so the rule is proven from the reference line rather
than read off a window. The kernels-crate union twin (`mla_ref.rs`) carries the same
round, which keeps `format_two_union_equals_the_engine_reference_on_the_attended_multiset` bit-for-bit
between the two host references; its expected mixtures, its window-vs-union premise and its slow
f64 reference all moved to the rounded probability with it. The device body still keeps f32 
(`kernels/flash_mla_csa2_seq_sm121.cu:375-389,475-492`): the host twin is now native and the
kernel variant is the remaining leg of this task (format-2 body variant, twin comparison once the
kernel casts too, cubin pin, window). Until that variant lands the body twin will report the
difference on a device window - deliberately, because the omission is now visible to it instead
of shared.

The isolation proposal above is superseded by the pinned host test: the twin difference is a
one-line cast, so a window is only needed to confirm the kernel variant did not regress anything,
not to establish the rule.

### F3 - the indexer query projection and the score dtype. **MEDIUM.**

Native indexer: `wq_b` is a `ColumnParallelLinear` at the default dtype, i.e. **fp8 W8A8**
(`model.py:519`): `act_quant(qr, 32, ue8m0)` on the bf16 q_norm output, then an fp8 GEMM with a bf16
output. `weights_proj` and `wk` are bf16 Linears, and the score path (`model.py:553-557`) is bf16
tensors: bf16 einsum, bf16 `relu * weights`, f32-accumulated sum to bf16.

Chain: the host indexer (`attention.rs:1045-1080`) computes `wq_b` as an **f32 matmul over the
dequantised fp8 weights with no activation quantisation**, then rope, then the fp4 q round trip; the
scores, the per-head weights and the head sum are all f32 (`attention.rs:1102-1119`). The k path
(`write_index_keys`) and the fp4 round trips themselves match native's format and rules (block 32,
ue8m0, power-of-two ceil with the fp4 amax floor, `attention.rs:1799-1861,207-211`).

Why MEDIUM rather than LOW: the indexer output is consumed **discretely** (the two top-k levels), so
a small score deviation does not stay small - it either keeps the selected set or changes it. The q
deviation is first-order (an absent fp8 quantisation propagating through the fp4 q codes), and the
score's f32-vs-bf16 dtype is second-order. No measurement isolates either: the body twin is fed the
chain's own `combined` index list, so it cannot see a selection difference against native.

Isolation: compare the chain's index scores and selected sets against the released forward's recorded
`topk_idxs` (the fixture class already records per-stage codes, so this is a fixture extension, not a
new model).

### F4 - Engram runs entirely in f32 on the host. **MEDIUM.**

Native Engram (layers 1 and 14, `model.py:1245-1280`): the fp8 embedding row is dequantised and
**cast to bf16** (`model.py:318-319`); `wkv` is an fp8 W8A8 Linear over that bf16 row
(`model.py:345`), whose activation quantisation is a native quantisation; key/value/gate are f32
from its bf16 output; the write is `(h + gate*value).to(x.dtype)` bf16.

Chain: `engram_forward` (`engine/forward/engram.rs:602-785`) dequantises the table into f32 without
the bf16 cast (`:476-487`), dequantises the fp8 `wkv` weights into f32 and runs a plain f32 matmul
with **no activation quantisation** (`:558-570`), and computes the gate and the write in f32 before
the bf16 staging into the device mixer.

Direction: chain more precise than native (an absent fp8 activation quantisation and two absent bf16
rounds). Magnitude: the fp8 quantisation effect on a 6,144-wide reduction is small in expectation
(the per-element relative error is ~2^-4, the dot's is ~2^-4/sqrt(K) for a random-sign sum), but the
Engram term is an additive write to the residual stream on two of forty layers, and the chain's f32
`value` also differs from native's bf16-rounded one. It is the largest *fully* f32 block left in the
chain and it is a declared host seam, not a device stage. Unmeasured.

**Status (task-9).** The host seam now implements the released rule op for op, and the three device
kernels that will be the window's comparison are written and pinned host-side:

* `kernels/engram_embed_sm121.cu` reads a 264-byte row (256 E4M3 codes, 8 E8M0 scales), dequantises
  and casts to bf16, then applies the reference act_quant rule (block 32, ue8m0 scale, 1e-4 amax
  floor, ceil-power-of-two scale, +-448 clamp, RNE).
* `kernels/engram_wkv_sm121.cu` is the fp8 W8A8 projection over the checkpoint's raw `[out][in]`
  codes and `[out/32][in/32]` grid: the 32-value code dot in the five-step `__shfl_down_sync` tree,
  scaled per block and accumulated in ascending blocks, output bf16.
* `kernels/engram_gate_write_sm121.cu` is the f32 gate over bf16 key/value/stream and one bf16 cast
  of `h + gate * value`.

All three are in `baseline/kernel_baseline.txt` (25376 / 8088 / 18744 bytes, 0 PTX membar lines,
`V13.0.88`, `cuda_13.0.r13.0/compiler.36424714_0`), and
`crates/sparkflash-kernels/src/engram_native_ref.rs` reproduces each op for op
(`tests/engram_native_twin.rs`).

The two "absent rounds" this section used to name are not equally weighted, and the twin measures
which one carries F4:

* the row's bf16 cast is **exact**: an E4M3 value has three mantissa bits and an E8M0 scale is a
  power of two, so all 256 values of all eight captured rows are unchanged by it (0 moved);
* the activation quantisation of a fetched row built from table values is also exact for the same
  reason (a power-of-two scale keeps the four significant bits), so it is not the deviation either;
* the discriminating rounds are the **bf16 wkv output** and the **bf16 write**. On a synthetic head
  over the released rows the native rule moves the gate by 5.883e-5 and the write by 0.9295 on a
  285.3 magnitude (3.3e-3 relative) - the bf16 output round is what the gate sees.

The CI fixture regenerates on this tree (task-9 alone, from 304ff7c): sha256
`17030896f4cb5b35266bf97003918489e0551010835b817438a143360d678780`, 0/9 tokens changed, max
|logit delta| 2.2097999999998175e-05. The boundary measurement does not move: layer 0 has no Engram
write, so `docs/build/phase1-boundary-table.md` section 6 keeps task-10's numbers.

Open, and stated rather than claimed: the numeric device-vs-twin comparison is a window measurement
(no GPU run was made; the nodes are frozen by the operator), and the on-disk read path has not yet
been compared with the released vLLM Engram patch (that comparison needs `docker exec` on a node).

Isolation: the Engram fixture stage (`engram_golden.json`) already exists; extend it to compare the
chain's `value` and `gate` against the released forward's tensors for the two layers.

### F5 - missing bf16 rounding points at host seams that feed dynamic quantisers. **LOW each, systematic.**

The chain's activation quantisers take f32 vectors (`quantise` at `attention_device.rs:618-624`,
`quantise_activation` at `moe_device.rs:2030-2033`), and the vectors they receive are host or kernel
captures in f32, whereas native quantises a tensor whose dtype is already bf16 (the preceding
module's output). The missing rounds are:

| seam | native rounds | chain input | consequence |
|:--|:--|:--|:--|
| `attn_norm` -> `wq_a`/`wkv` quantisers | RMSNorm output bf16 (`model.py:292-293`) | f32 `xhat_normed` capture (`mega_mhc_sm121.cu:340-348`) | a few % of E4M3 codes can differ by one, and rarely a block scale |
| `q_norm` -> `wq_b` quantiser | bf16 (`970-971`) | f32 `query_path_with` | as above |
| `kv_norm` + window RoPE -> window quantiser | bf16 twice (`703-707`) | f32 `window_kv_with` | measured: 1-2 codes in 5,120 (D23 stage 1) |
| compressor `norm` -> compressed-fp4 quantiser | bf16 (`484-486,757-760`) | f32 `compressor_step` | a few % of FP4 codes |
| `ffn_norm` -> routed/shared activation quantisers | bf16 (`977-981,846-847`) | f32 `ffn_in` capture | as above |
| inverse RoPE -> `wo_a` quantiser | bf16 (`775`) | f32 rotation | subsumed by F1 |
| `hc_pre` collapse -> norms | bf16 (`958-961`) | f32 registers in the entry kernel | a small perturbation of every normed vector |

**Status (task-10).** The host-side seams now round where native does, and each round is pinned by a
test that fails without it:

* `wq_a`/`wkv`/`wq_b`/`wo_b` activation quantisers: the device staging narrows the f32 
  activation to bf16 before `quantize_activation_fp8` (`quantise_native_activation`, 
  `attention_device.rs`); test `the_w8a8_activation_quantiser_reads_the_bf16_value`. q_norm's host
  output is also narrowed in `query_path_with` before the device wq_b staging, pinned by
  `the_host_q_norm_output_is_a_bf16_tensor_before_the_wq_b_quantiser`.
* `kv_norm` + window RoPE -> window quantiser: `window_kv_with` narrows the norm output and the
  rope writeback before the fp8 round trip; test `the_quantiser_seams_narrow_the_host_value_to_bf16_first`.
* compressor `norm` -> compressed-fp4 quantiser: `compressor_step` narrows before and after the
  norm (`kv.to(dtype)`) and `write_compressed_cache` narrows the rope writeback; test
  `the_compressor_latent_is_a_bf16_tensor_before_the_fp4_quantiser`.
* indexer k and q: the rope writeback is narrowed before the fp4 round trip (task-8 did the
  q/dtype path, task-10 adds the writeback and the k path).
* inverse RoPE -> `wo_a` quantiser: covered by F1: task-6's native `wo_a` arm rounds its activation
  to bf16 (`WoAMode::Native`, measured 3.60/3.85/4.39% exact at layers 21/30/39). The MXFP8
  comparison arm is deliberately left f32: it is the non-native baseline of the F1 ledger, not a
  chain path.
* `ffn_norm` -> routed/shared activation quantisers and the `hc_pre` collapse stay OPEN: both rounds
  live in `moe_device.rs` and the entry kernel, which task-4/5 own this cycle. Reported, not
  edited here.

The host reference fixture was regenerated on the branch that carries both host-side changes
(task-10 F5 and task-7 F2): sha256 `c9aa2cd5bb093c579cdfcb4fcc9f8f100ea22789d9841600830b0f020ae7927d`, 0/9 tokens
changed, max |logit delta| 1.4999999999043556e-8 against the base 4bf96e9 fixture and
5.000000000837668e-9 against the task-10-only fixture that this branch supersedes. Task-9's native
Engram seam regenerates it once more, from that file: sha256
`17030896f4cb5b35266bf97003918489e0551010835b817438a143360d678780`, 0/9 tokens changed, max
|logit delta| 2.2097999999998175e-05 - the bf16 output round of the Engram projection on layers 1
and 14, which is the measured size of F4. The pinned
boundary measurement in docs/build/phase1-boundary-table.md section 6 moves with it (87
differing values in five tensors, was 161 in six) and was re-checked on the stacked tree.
Regenerating a fixture by copying either base's file is not valid: only a run on the tree that
carries the arithmetic produces the numbers above.

The router is *not* in this list: its staging rounds the f32 capture to bf16 before the kernel
(`moe_device.rs:553-556`), so the kernel sees what native's bf16 ffn_norm output would be - and the
router twin measured the two agree to 5.96e-8. Neither is the tail: `moe_device.rs:1379-1382`
bf16-stages the collapsed tail before the final norm, so that round is present.

Measured: the window row was the only measured seam before task-10 (D23 stage 1: 5,119/5,120 codes
and 160/160 scales for layer 0 token 0; 5,118/5,120 for layer 4 token 0; 5,120/5,120 for the
other tokens and layer 1). That 1-2 code gap is exactly what the two missing rounds cost, so the
host seam now carries both; re-measuring the device window codes in a window is a follow-up leg,
not a host change. The remaining seams are INFERRED to be the same class and order of magnitude.
The activation-quantiser rounds are structural for every block, not just the measured window row:
the CLI test's deterministic scan finds a block where the f32 bits change an E4M3 code, which is
why the round is a real arithmetic change and not a no-op.

Note that D8's stage-0 result (the ceil rule recomputed from the captured f32 vector reproduces
5,120/5,120 codes and 160/160 scales) does **not** cover this: it compares two quantisers on the same
f32 input, not the chain's input against native's bf16 input.

### F6 - the shared expert's SwiGLU activation is not bf16-rounded before its w2 quantiser. **LOW.**

Native: `Expert.forward` computes `x = F.silu(gate) * up` in f32, multiplies by the routing weights
when present, and calls `self.w2(x.to(dtype))` (`model.py:846-852`) - the `.to(dtype)` is a bf16
round **before** `w2`'s `act_quant`. The routed path's weighted kernel reproduces this exactly
(`swiglu_mxfp8_quant_weighted_sm121.cu:101-110`, and its header says so). The shared path's
unweighted kernel quantises the unrounded f32 activation (`swiglu_mxfp8_quant_sm121.cu:133-140`).

Direction: chain more precise. Consequence: a subset of the 2,304-wide intermediate's E4M3 codes can
differ by one from native's, which the w2 GEMM then averages. Low, but it is a single missing round
in a device kernel rather than a host seam, so it is cheap to fix if the Lead wants native exactness
there.

### F7 - the routed combine rounds a bf16 rank chain instead of per-expert outputs. **LOW.**

Native: each expert returns bf16 (its w2 GEMM's output cast), and `y[idx] += expert(...)` adds those
bf16 values into an f32 accumulator (`model.py:846-852,899-903`) - six bf16 roundings per token, of
individual expert outputs.

Chain: one f32 accumulation per rank over that rank's slots, then **one bf16 cast per rank**, with the
next rank seeded from the previous rank's bf16 value (`b12x_mxfp4_moe_sm121.cu:150-197`,
`moe_device.rs:870-943`): four bf16 roundings per token, of partial sums. Measured: the device chain
is bit-exact against its own chained-bf16 twin (20,480/20,480, D23 stage 3) - which is the chain's
rule, not native's.

Direction: neither more nor less precise; a different rounding *structure* at bf16 scale. Expected
difference from native: of order a bf16 ulp on the FFN output with a random sign - it can flip the
occasional bf16 code but does not bias the result.

### F8 - accumulation association and f32 last-bit differences (no rule difference)

**NONE: the YaRN table arithmetic is aligned to the reference; the libm/rsqrt last bits are declared, not guessed.**

All GEMMs: native accumulates code products per 32-block in an MMA and applies both scales to the
block partial (`kernel.py:269,553`); the chain's dense/wo_a/mxfp4 kernels dequantise each product and
accumulate ascending K in f32 (`b12x_mxfp8_dense_sm121.cu:93-113`; `b12x_mxfp8_wo_a_sm121.cu:88-110`;
`b12x_mxfp4_moe_sm121.cu:168-196`). Because the scale multiplies are exact (section 2.2), the
operands are the same f32 values in both; only the association of the f32 additions differs. This is
**not** a discrepancy class, and it is why zero-mismatch twins are attainable at all (the mHC
projection 5.96e-8, `wq_b` 0/32,768 max 0.0, the body 66/120 exact).

Two f32 items inside this class are not rule differences. Task-11 aligned the one that is a host
arithmetic question and declared the one that is the reference's own platform behaviour:

- **`torch.rsqrt` against the host's `1/sqrt` - declared, not aligned.** Native calls
  `torch.rsqrt` in RMSNorm (`model.py:292`), the Engram gate's `rstd` (`model.py:359`) and
  `hc_mixes` (`model.py:953`). Every host site computes the correctly rounded reciprocal of the f32
  square root: `linalg.rs:338` (`1.0 / (mean + eps).sqrt()`), `mhc.rs:207`,
  `engram.rs:769-770` (and `engram.rs:715` for the `dim**-0.5` constant, measured equal to the
  reference's narrowed Python value at dims 8, 16, 32, 64, 128, 512 and 5,120). The test
  `the_norm_rsqrt_is_the_correctly_rounded_reciprocal_sqrt` pins the bits
  (`crates/sparkflash-engine/src/forward/mod.rs:1806`). CPU torch's `rsqrt` is that same `1/sqrt`, so
  a CPU-generated reference would match this host only if that backend also computes the plain
  reciprocal; PyTorch does not document that as bit-identical, and its CUDA lowering is a device
  intrinsic accurate to about 2 ulp, so on the released platform a device normaliser and this host
  can differ in the last bits. This is the ambiguous seam inside F8: no host arithmetic can be shown
  bit-equal to native's `torch.rsqrt` on every backend, so it is declared rather than guessed at,
  with the bound 2 ulp relative (about 2.4e-7 on the normaliser; a bf16 ulp at 1.0 is 3.9e-3) and the
  mHC twin's 5.96e-8 as the measured downstream bound. A bit-level device-vs-host normaliser
  comparison must expect the last bits to differ; it is not a defect.
- **The RoPE tables - aligned to the reference's f32 arithmetic.** Native builds them with
  `torch.polar` over f32 tensors (`model.py:369-390`): a f32 `arange` for the frequencies, then
  YaRN's ramp `((arange(f32) - low) / max(high - low, 1e-3)).clamp(0, 1)` and the blend
  `freqs / factor * (1 - smooth) + freqs * smooth` with `smooth = 1 - ramp` (`model.py:384-386`).
  The host computed the ramp in f64 and reused `ramp` in place of the reference's `(1 - smooth)`,
  which re-derives the value from the rounded `smooth` and is not always the same f32 value. At the
  released geometry (`rope_head_dim` 64, base 160000, `original_seq_len` 16,384, factor 16, beta
  32/1) that changed pair 14 by one ulp; task-11 reproduces the reference's f32 ramp and blend
  (`attention.rs:290-351`). Measured: over positions 0..=4096 the released table's 32 pairs move
  3,862 of 131,104 `cos` entries and 3,926 of 131,104 `sin` entries (first at position 47). The
  pinned test `the_yarn_ramp_is_the_reference_f32_arithmetic` (`attention.rs:2935`) spells the
  reference out independently and fails on the f64 variant. The tiny CI fixture is unaffected
  because its ramp values are exactly 0 and 1; it was regenerated byte-identical (sha256
  `c9aa2cd5bb093c579cdfcb4fcc9f8f100ea22789d9841600830b0f020ae7927d`). The remaining ambiguity is
  the libm, and it is declared: `cos`/`sin` and the base `pow` are the platform's f32 functions.
  Measured on this host, glibc's f32 `cos`/`sin` are within 1 ulp of the correctly rounded
  double-precision value across the released table's 131,104 angles; the native platform is CUDA,
  whose `sinf`/`cosf` are libdevice approximations (up to 2 ulp). The table's last bit is
  platform-defined on both sides and cannot be aligned host-side; the stage-1 class-C agreement
  (5,119/5,120 codes, D23) bounds what it does downstream.


---

## 5. Measured evidence: what each number is actually comparing

Every existing "chain" number was produced by one of three comparison classes, and **only one class
compares against native**:

| class | what is compared | evidence items |
|:--|:--|:--|
| **A. chain vs the f32 simulation reference** | the device path against the engine's f32 reference (the code path that reproduces the fixture's nine tokens) | `stage twin` post-attention 9,627/20,480 bf16 codes, max_abs 3.120759e-2 (position 0 layer 0); `stage twin` host FFN Y 4,936/5,120, max_abs 1.989301e-2; b68 position 129: 128,528/129,280 logits outside the 0.02 bar, max 15.009581, argmax 8060; b68 0/9 tokens; the router/expert-set findings of STATUS section 3 |
| **B. chain vs a device twin on the same staged operands** | one arithmetic implementation against another within the chain (same bytes in) | attention body twin: 66/120 layer-position pairs 0 mismatches, worst <= 7.811069e-3; router twin: 120/120 `equal=true`, scores <= 5.96e-8, weights <= 2.98e-8; final norm `bf16_mismatches=0` at every position; entry `capture_exact`; MoE combine 20,480/20,480 bit-exact; `wq_b` vs `mxfp8_ref` 0/32,768; stage-8 dense 16,384/16,384 and wo_a 8,192/8,192 ULP 0 (D23) |
| **C. chain vs a released-forward record** | the chain's arithmetic against the released forward's own recorded tensors | **stage 1 only**: the window row / kv_norm / RoPE / quantiser chain reproduces the released codes 5,119/5,120 and scales 160/160 (layer 0 token 0), 5,118/5,120 (layer 4 token 0), exact elsewhere (D23) |

What follows from the classification:

- **The b68 criterion-1 failure is a class-A number.** 128,528 of 129,280 logits outside the bar,
  0/9 tokens, the layer-3 expert-set flip - all of it is the device chain against the f32 simulation.
  Class A says "the chain does not compute the f32 reference". It does not say the chain computes a
  different model from the checkpoint, which is what D39 made the target. Read as a native verdict,
  class A is not evidence either way.
- **Class B cannot catch a deviation the chain and its twin share.** Every host seam the twin also
  uses is invisible to it: F5's missing rounds are in both, F2's probability handling is in both,
  F4's Engram is not in the twin at all, F3's indexer is not in the twin at all. A class-B PASS is a
  wiring/property statement, not a native statement.
- **Class C is the only direct native evidence, and it covers one stage of one layer.** It bounds F5
  at the window row (1-2 codes in 5,120) and says the *quantiser rules* (block size, scale dtype,
  ceil rule, amax floor) are native to that stage. It says nothing about F1-F4, F6, F7.

So the honest statement of where the chain stands against native is: **the rules are audited
(section 3), one seam is measured against native (class C), and six deviations are sourced but
unmeasured.** That is the gap a vLLM comparison is meant to close behaviourally, and it is also why
the first-order deviations (F1, F3, F4) are worth isolating before the comparison - a numeric
comparison will tell the Lead whether they matter, but not which one it was.

---

## 6. Consequences for the numeric acceptance criterion

1. **Bit equality with native is unattainable by construction**, before any of the deviations above:
   native's f32 accumulation order is the MMA's (section 2.2), and the chain's is a written loop.
   D39's move to a numeric, distributional criterion is therefore the only criterion that can be
   met; this audit's contribution is that this is a property of the arithmetic, not a defect of the
   chain.
2. **Against native, the chain is closer than the f32 fixture is at most of the stages where the
   two differ.** For the four fp8 attention projections, the router, the routed experts, the shared
   expert's weights, the attention formats, the window/compressed/indexer quantiser rules and the
   head, the chain *is* native and the f32 reference is the simulation. Only `wo_a` (F1) is a
   projection where the chain substitutes a coarser arithmetic than native.
3. **The vLLM comparison should expect distribution agreement with three specific signatures**: a
   possible systematic logit offset from F1 (one extra E4M3 activation quantisation on every layer's
   residual path), discrete jumps whenever F3's indexer selection differs from vLLM's, and
   bf16-scale stochastic differences from F2/F5/F6/F7. A comparison that shows *stochastic*
   bf16-scale differences only is consistent with everything in this audit; a large systematic
   offset points at F1 or at a stage this audit could not measure.
4. **What D37 measured, read correctly.** The host-f32 projection mode made the four native W8A8
   projections compute f32 over dequantised weights - *higher* precision than native - and `wo_a`
   bf16 einsum - *native*. Its premise ("the projections' inputs are bf16-representable, so f32 is
   the point") is true only for `wo_a`. Recorded here so the decision record can be read later
   without reconstructing it; D39 stands.

---

## 7. The declared host seams, one at a time

The chain's own ledger prints the seam list; this table takes it and verdicts each seam's dtype and
rounding order against `model.py`. "Declared at" is the code the device run executes at that seam.

| seam (as the ledger names it) | declared at | native dtype and rounding order | chain dtype and rounding order | verdict |
|:--|:--|:--|:--|:--|
| `q_norm` | `attention.rs:1658-1720` (host) | bf16 in, f32 compute, **bf16 out** (`model.py:292-293`) | f32 in/out | **mismatch**: one bf16 round missing (F5) |
| `kv_norm` | `attention.rs:1723-1760` | bf16 out, then rope bf16, then in-place quantiser | f32 out, f32 rope, then a host fp8 round trip and a bf16 re-quantise in `engine_window_row` | **mismatch** upstream of the quantiser (F5); the quantiser's own input *is* bf16, so seam #14 is native |
| `RoPE` (q, kv, compressed latent, inverse) | `attention.rs:338-420,1066,1744,1833,1888,2309` | f32 rotation of a bf16 value, written back to bf16 once (`model.py:392-409`) | f32 rotation; bf16 only where the result is staged for the body | q/body **match** (one round at the same point); kv/compressed/inverse **mismatch** (the round is missing; inverse feeds F1) |
| window row construction | `cuda_harness.rs:220-241` | quantiser input is bf16 (`model.py:706-707`) | bf16 by construction, and the two quantisers must agree | **match** |
| compressor / cache | `attention.rs:893-994,1869-1930` | f32 pooling and f32 GEMMs when ratio > 1, then `norm(...).to(bf16)`; **bf16** GEMM when ratio == 1; fp4-16-E4M3 in place | f32 throughout; fp4 round trip with the same block/dtype/floor | **rule match, dtype mismatch**: the bf16 rounds are missing (F5); the ratio-1/ratio>1 dtype split is a native subtlety the chain collapses to f32, which is *more* precise for ratio 1 and equal for ratio > 1 apart from the latent's round |
| selected-list assembly | `attention.rs:1084-1205` | top-k on bf16 scores, shift by the window length, unreachable -> -1, lower-position tie | same structure on f32 scores | **structure match, score dtype mismatch** (F3) |
| indexer scoring | `attention.rs:1045-1119` | fp8 q projection, bf16 k projection, bf16 scores/weights | f32 projections, f32 scores/weights | **mismatch** (F3) |
| `D8 quantisers` | `attention_device.rs:618-624`, `moe_device.rs:2030-2033` | ceil rule on a bf16 tensor | ceil rule on an f32 vector | **rule match** (measured 5,120/5,120 codes at stage 0), **input dtype mismatch** (F5) |
| router selected-id readback / local mapping | `moe_device.rs:576-600,646-701` | ids are int32, mapped to per-rank experts | ids read back as u32, mapped by expert index; no arithmetic | **match** |
| selected-expert staging | `moe_device.rs:482-1134,1852-1896` | the weight bytes and the router's ids | same bytes, staged per distinct expert; the GEMMs are device-native | **match** |
| shared add (routed + shared) | `moe_device.rs` host add; BF16-staged at `device_chain.rs:1198` | `y += shared(x)` f32, then `y.type_as(x)` bf16 (`model.py:903-904`) | f32 add, then one bf16 stage | **match** |
| head-weight read/upload | `moe_device.rs:1438-1471` | bf16 checkpoint values up-cast to f32 in a f32 GEMM | bf16 bytes uploaded, f32 dot | **match** (f32 GEMM over bf16-valued operands) |
| 4-byte token + status readback | `device_chain.rs` tail | greedy argmax of f32 logits | readback only | **match** (no arithmetic) |
| four-shard block-max/argmax assembly | `moe_device.rs:2041-2076` (+ `WINNER_CERTIFIED` fallback) | torch argmax: max value, first index wins ties (`model.py:1285-1289`) | lower global entry index wins ties | **match** |
| embedding gather | `device_chain.rs` embed seam (`model.py:174-181`) | bf16 row | bf16 row widened to f32 exactly | **match** |
| B/C/A coefficient round-trip | `device_chain.rs:1184-1196` | f32 coefficients | f32 readback and re-upload | **match** |
| **Engram** | `engram.rs:602-785` (host, layers 1 and 14) | fp8 table -> dequant -> bf16; fp8 W8A8 `wkv`; f32 key/value/gate; bf16 write | f32 table, f32 matmul, no quantiser, f32 gate, bf16 staging | **mismatch** (F4) |

---

## 8. What this note does not claim

- It does not re-open D39, and it does not propose a precision-first path. F1 is written as a
  deviation *from native*, and the fix, if any, is to make `wo_a` native (bf16 einsum) - not to make
  anything f32.
- It does not claim any of F1-F7 has a measured magnitude against native. Only the window-row seam
  has one (class C, section 5).
- It does not audit the MTP/DSpark heads, the vision tower, batching, or any decode-time state other
  than what the b68/b67 runs exercised.
- It does not claim the chain's *rules* are wrong where the verdict is `native`: those rules are
  sourced line by line in sections 2 and 3, and several are independently measured in class B.

## 9. If the Lead wants the deviations isolated

Ordered by cost-to-information, each a small harness addition plus a device window run (none of them
re-opens the precision decision):

1. **F1, `wo_a` isolation** - one flag to run the engine's bf16 host `wo_a` for that projection only,
   compare its f32 output against the device kernel's on the same captured attention output. This is
   the only first-order deviation on the residual path.
2. **F2, probability rounding** - a compile-time variant of the body kernel that casts `s_scores` to
   bf16 before the value loop; compare against the native-recorded attention output. Cheap because
   the kernel already has the slot loop in shared memory.
3. **F3, indexer selection** - extend the fixture class with the released `topk_idxs` for the eight
   index-source layers at positions 0..2 and compare the chain's lists (not just its scores).
4. **F4, Engram** - extend `engram_golden.json` to the `value`/`gate` tensors of the two Engram layers
   and compare the host seam's f32 tensors against them.
5. **F6** - one bf16 round in the unweighted SwiGLU epilogue before its quantiser; a kernel test
   against a reference that pins the round, then a window re-run of stage 8's shared probe.

The `--dump-logits` binary already staged on the node (`4f599b7`, sha256 `7e0ce4bf...`) is what turns
any of these into the *vLLM* comparison that is the class this audit cannot provide on its own.
