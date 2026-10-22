# Moving the three seam blocks onto the device: a code-level plan

Status: **plan only** — no kernel source, cubin, pin-set edit, wiring or MR. Every statement below is
traced to the host code that computes it today and the device shape that reproduces its bits. Base
`268f2ab1` (published 249.25 ms/step, device 71.2 ms/step). Probe budgets: `ref_seam` 0.501,
`ffn_seams` 0.521, `mhc_entry` 0.318 ms/layer = 20.0 / 20.8 / 12.7 ms/step.

## 1. The identity rule, and the cost that is not hidden

| requirement | host does | device must do | guard |
| :-- | :-- | :-- | :-- |
| one chain per output | `acc += x[k]*w[k]`, k ascending from 0.0, one round per op | one thread owns the output, ascending k, `__fadd_rn(acc, __fmul_rn(x,w))` | `-fmad=false`; disassembly test: no FFMA/fma in the moved chains (the `ptx_membar_lines` precedent, `cuda_driver.rs:3559`) |
| no split-K, no reassociation | row split only (`linalg.rs:365`) | grid may put each output in its own thread; never reduce across threads | thread count == output count, checked by the wrapper |
| serial sums stay serial | `mhc.rs:216` square sum 20480, `linalg.rs:1135` mean 5120 | one thread, ascending | no warp/block reduction in the chain |
| exact where host is exact | `bf16_round` RNE, ue8m0 exponent, e4m3 RNE, `1.0/sqrtf` | same bit ops; `sqrtf` then `/` (correctly rounded), never `rsqrtf` | differential twin test, bits not bars |
| transcendentals stay host | `sigmoid` (`linalg.rs:1173`), `softmax`, `split_sinkhorn` `exp` (`mhc.rs:336`), `pool_group` `exp` (`attention.rs:985`) | not moved: CUDA `expf` is not host `expf` bit for bit | residue is small: 24 logits + a 4×4 sinkhorn per mHC seam |

The cost that does not disappear: device time is hidden behind host work only until the host reads the
result. A seam that launches dots and immediately `d2h`s the raws stalls for kernel + transfer
(measured 6-13 µs/call). Every row is priced as a trade: **host_statement vs (device_time + stall)**.

## 2. `ref_seam` — 0.501 ms/layer, 40 calls/step (`attention.rs:1980`)

| statement (calls/step) | today | device shape | kernel | bytes/call | device µs | residue |
| :-- | :-- | :-- | :-- | --: | --: | :-- |
| compressor `wkv`/`wgate` [512×5120] (7: layers 2,8,14 ×2, 20 ×1) | `matmul_hot_rows`, ascending 5120, mul+add, then `narrow_bf16` (`attention.rs:1035`) | one lane per output column, ascending K (existing tiling) | **new instantiation** `b12x_mxfp8_gemv_m1_nf_f32a_f32w_sm121` (+ `bf16w`): same body and 11-word ABI, weight read as f32/bf16 instead of F8-decoded (`mxfp8_gemv_launch.rs:74`); the compressor weights are bf16 at ratio 1 and f32 above it (`attention.rs:1029`, `model.py:446-448`), so pinned `nf_f32a` cannot read them | 10.49 MB f32 / 5.24 MB bf16 | 53.6 / 26.8 | `pool_group` `exp` host: 2×512 f32 d2h |
| `pool_group` (3 layers) | max over ratio, `exp`, normalise, weighted sum | **host** | — | — | — | 1 d2h/call |
| `write_index_keys` `wk` [128×512] (2.5 amortised) + rms 128 + rope + fp4 | ascending 512, `narrow_bf16`, `narrow_and_round_trip Fp4` (`attention.rs:2623`) | one lane per output column | pinned `b12x_mxfp8_gemv_m1_nf_f32a_sm121` **unchanged** (F8 weights, f32 activation, bf16 store) | 66 KB | 0.4 | rms/rope/fp4 host (128 values) |
| `index_weights` [32×5120] (8) | bf16-round input, ascending 5120 (`attention.rs:2584`) | one lane per output column | pinned `nf_f32a` **unchanged** (F8 weights; confirm the arena dtype before wiring) | 164+5 KB | 0.9 | out d2h 128 B/call |
| indexer `wq_b` [4096×1280] (8) | activation bf16-round + fp8 round trip (exact), ascending 1280, bf16 store | 16 blocks × 1 warp | pinned `nf_f32a` **unchanged** (f8 × f8 products are exact, so rounded mul+add equals the host chain) | 5.24 MB + 164 KB | 27.6 | rope+fp4 host or device |
| `index_scores` (8): 32 heads × n_pos dots of 128, rectified ascending head sum with bf16 rounds (`attention.rs:1418`) | host serial per position | **one thread per position**, 32 heads serially ascending, dot serial 128, same rounds | **new** `index_scores_sm121` (no pinned symbol does the rectified head reduce) | keys n_pos×128 f32 (32 KB at the probe window) | 10 | scores d2h for host top-k |
| top-k + tie rules + candidate select (9 layers) | sort over n_pos | **host**: data-dependent ordering with tie-to-lower-position; µs-scale at n_pos ≤ 129 | — | — | — | ids stay host |
| `fetch_compressed` (38 layers) | host copy of n_pos×512 f32 | **host for now**: `stage_attention` turns the rows into device operand bytes; moving rows without staging only relocates work | — | ~4 MB/step | — | — |
| seam digests | `evidence_f32_sha256` gated off in the lean row (`tp4_forward.rs:1446`) | host, evidence only | — | — | — | the identity row's oracle |

`n_pos` is `position/ratio`: the probe row's window is 129 positions (5160 layer-positions / 40 layers,
`ws-a-host-path-20261021.md` § 2), so n_pos ≤ 64 at ratio 2 and ≤ 129 at ratio 1. `index_scores` bytes
and `fetch_compressed` copies scale linearly with n_pos at longer context; nothing else in § 2 does.

Device added `0.67 ms/step` (0.51 if bf16 weights), stalls `0.18-0.39` (≈30 calls/step) →
**net −11 to −15 ms/step** at a 60-80 % movable host share. The compressor GEMV is the one row where
host (30-150 µs est.) may not beat device+stall (62 µs), so it moves last.

## 3. `ffn_seams` — 0.521 ms/layer, 40 calls/step (`tp4_forward.rs:3901`)

| statement | today | device shape | kernel | bytes | device µs | residue |
| :-- | :-- | :-- | :-- | --: | --: | :-- |
| `hc_mixes`: 24 dots over 20480 + square sum (`mhc.rs:133`) | 24 independent ascending chains, square sum ascending, `mean/width`, `1/sqrt(mean+eps)` | 25 threads, one per coefficient + square sum, ascending 20480 (a chain cannot be split: 20480 dependent 4-cycle adds ≈ 82 k cycles ≈ 51 µs at 1.6 GHz) | **new** `mhc_hostarm_dots_sm121`. Pinned `mega_mhc_sm121` does **not** qualify: native arm — FMA-contracted update (`layer_zero.rs:613`), tile-reduction RMS (bar 3e-4, `:578`), its own sinkhorn; its report is bars, not bits | 1.97 MB + 80 KB | 51 | 25 f32 d2h, 24 coeff h2d |
| coefficients: 8 sigmoid, comb affine, `split_sinkhorn` 20 iters (`mhc.rs:256`) | host | **host** (`exp`) | — | — | 2-6 | — |
| `hc_pre` 5120×4 ascending copies | mul+add 4-long chains | 5120 threads ×4 (phase B) | `mhc_hostarm_pre_sm121` | 80 KB | 2 | `attn_in` 20 KB d2h (host `hc_post` needs it) |
| `rms_norm` 5120 (`linalg.rs:1116`) | serial mean ascending, `1/sqrt`, elementwise | serial 5120 thread + parallel scale | same kernel | 20 KB | 13 | — |
| `quantize_activation_fp8` 5120 (`quant.rs:677`) | amax per 32 (order-free), ue8m0 code, exact `/`, e4m3 RNE | 5120 threads / 160 blocks | same kernel or `act_quant_fp8_sm121` | 20 KB → 5.3 KB | 2 | — |
| 6 seam digests | gated off in the lean row | host | — | — | — | — |

**Phase A (first): move only `hc_mixes`** — coefficients untouched, one statement, trade is
`hc_mixes_host` vs 51 + 8-13 µs. **Phase B** folds `hc_pre`/`rms_norm`/quantiser into a second
kernel; it pays only once the streams stop crossing the bus (§ 5).

## 4. `mhc_entry` — 0.318 ms/layer, 40 calls/step (`tp4_forward.rs:876`)

Same statements as § 3, different data only:

| | `ffn_seams` | `mhc_entry` |
| :-- | :-- | :-- |
| predictor / pre-mix | `ffn_fn` [24×20480] f32 / previous layer's `ffn_pre` | `hc_attn_fn` [24×20480] f32 / previous layer's `ffn_pre` |
| device-visible output | `pre_ffn_hidden` + 5.3 KB act codes | `normed` 20 KB, h2d today by `run_layer_attention_projections` |
| net | § 3 | −2 to −6 at phase A; `normed`'s h2d disappears entirely at phase B |

## 5. Expected step, and what 150 actually needs

| slice | host removed | device added | stalls | net | step after |
| :-- | --: | --: | --: | --: | --: |
| `ref_seam` | 12-16 | 0.67 | 0.18-0.39 | **−11 to −15** | 234-238 |
| mHC phase A (both seams, 80 calls) | 8-20 (`hc_mixes` 100-250 µs est./call) | 4.08 | 1.4-3.1 | **−5 to −18** | 216-233 |
| mHC phase B (+ `hc_post` on device) | 3-6 | 1.4 | −1 to −2 (drops `normed`/`pre_ffn` h2d) | **−3 to −7** | 209-230 |
| central, all three | | | | **≈ −24 → 222-232** | |

The three budgets sum to 53.6 ms/step, so removing *all* of them leaves 196: **150 is not reachable
from these slices.** The bus is now the binding constraint — the loop already pays 12 d2h and ~23 h2d
per layer-position (385 KB h2d, `ws-a-host-path-20261021.md` § 2), so each added d2h must be beaten
by a shorter kernel than the host statement it replaces. The remaining blocks
(`attention_output` ~22 ms/step, `moe_reduce` 11.8, `stage_attention` 6.6, the `hc_post` and
attention-residual epilogues) must move *with* their transfers; `hc_post` on device is the natural
next slice because it removes the 80 KB stream h2d per seam and makes phase B free of stream traffic.

## 6. Identity proof plan (no tolerance anywhere)

| statement | oracle today | new pin, captured **before** the change | end check |
| :-- | :-- | :-- | :-- |
| `hc_mixes` dots + square sum | captured-boundary tests at base (`the_mixes_are_the_rms_scaled_projection_with_norm_eps`, `the_captured_hc_pre_native_rounds_the_collapse`, `mhc.rs:640/1305`) + the evidence-row seam digests | 25 raw f32 + square sum at decode shape, two fixtures (m11 layer-0 tensors, synthetic seed), as exact u32 arrays + sha256, committed before any kernel exists; run against the unmodified host reference to prove the pin is pre-change (the exact-bit pin form proven on branch `task/host-ilp`, `the_interleaved_seams_reproduce_the_reference_bits`) | kernel Rust twin == pin, device == twin |
| compressor / `wk` / `weights_proj` / `wq_b` | `tests/mxfp8_gemv_oracle.rs` (host twin vs every `Mode`), existing `nf_f32a` twin | served-shape output bits on the boundary-replay capture (real layers 2/8/14/20, the loop's own `normed` row and q latent) | new `f32w` twin vs `matmul_hot_rows`; reused symbols' twin vs the seam's host function |
| `index_scores` | none | twin of `index_scores` vs the host function on the capture's q/keys/weights (n_pos×32 bits) | twin == pin, device == twin |
| whole step | evidence-ON identity row: every per-layer seam digest, token ids, logits sha byte-compared (`scripts/identity-gate.sh`, WS-A pattern) | — | digest-for-digest equal; one changed `act_fp8`/`normed` sha fails the slice |

Per-slice order: pin (mergeable alone) → twin vs pin → cubin + `kernel_baseline.txt` entry (bytes,
sha, new `ptx_fma_lines=0`) → wiring behind the existing arm switch → identity row. No
`native_rounding` involvement.

## 7. Recommendation and open questions

1. **mHC phase A** (`mhc_hostarm_dots_sm121`) as the first slice — 80 calls/step, one statement, the
   coefficient chain untouched, and the cleanest oracle. This inverts the value order (ref_seam is the
   larger budget) on risk, not on value; if the Lead prefers value order, slice 2 needs no new
   arithmetic at all and is the safe first move.
2. **`wq_b` + `wk` + `weights_proj` on pinned `nf_f32a`**: zero new arithmetic, removes three
   launch-bound host matmuls per index-source layer.
3. **Rest of `ref_seam`** (`index_scores` kernel; compressor instantiation last — byte-bound, 54 µs
   for f32 weights, wins only on the removed spawn wave).

New pinned symbols this plan adds (one cubin each; no pinned cubin is mutated):
`mhc_hostarm_dots_sm121`, `mhc_hostarm_pre_sm121`, `index_scores_sm121`,
`b12x_mxfp8_gemv_m1_nf_f32a_f32w_sm121` (+ `bf16w`), `act_quant_fp8_sm121`; the pin set grows from 22.

Open, answered by driver-fixer's probe-nosync row before slice 1: host µs per seam statement
(`hc_mixes`, `wq_b`, compressor GEMV, `index_scores`, `fetch_compressed`) and whether the
43-thread spawn wave is on the critical path. If `hc_mixes` is under ~65 µs/call, phase A does not
pay and slice 2 goes first.