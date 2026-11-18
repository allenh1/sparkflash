# SenseNova-Loop-9B (FP8 / BF16)

Status: **host reference on every target, and a TP=1 device arm served on the RTX 5080.** The FP8
mirror boots at its **160K** default with an **8-bit K/V cache**, keeps only the embedding on the
host, matches the host reference's argmax on the first token (worst logit delta 0.063 from the
E4M3 K/V round) and decodes at about **45 tok/s** (22.4 ms a token, 5.5x the first arm). The
token's 56 visits are one captured CUDA graph of 1028 launches. The pack is
`sparkflash-model-sensenova`; the served selector is `sensenova-loop-9b-fp8`.

SenseNova-Loop-9B is a 9.4B-parameter multimodal decoder released by SenseTime on 2026-11-17
(Apache-2.0). It is a Qwen3.5 hybrid stack with two structural additions:

* **Looped decoder.** 32 physical layers; layers 0-3 run once, layers 4-27 run twice with shared
  weights, layers 28-31 run once. One forward is 56 layer passes from 32 stored layers.
* **Two-stream manifold-constrained hyper-connections (mHC)** over `[4, 28)`. Two residual
  streams, learned pre/post mixes, and a 2x2 Sinkhorn-normalised combination (20 iterations,
  log-space) refreshed at every attention and FFN sublayer.

The rest is Qwen3.5: 32 layers alternating 3x Gated DeltaNet + 1x full GQA (16 query / 4 KV heads,
head_dim 256, partial RoPE 64, sigmoid output gate), a dense 12,288-wide SwiGLU, 248,320 vocab, a
27-layer vision tower, and 512K context via YaRN factor 2 (native 262,144). There is no MTP stage
in the release.

## What the checkpoint is

| Property | Value |
| :-- | :-- |
| Served id | `sensenova/SenseNova-Loop-9B` |
| Family | `sensenova_loop` |
| Physical layers | 32 (56 visits) |
| Hidden width | 4,096 |
| Full attention | 16 query / 4 KV heads, head 256, rope 64, group 4 |
| Gated DeltaNet | 16 key / 32 value heads, 128 dims, conv 4 |
| MLP | dense 12,288 SwiGLU |
| mHC | 2 streams, 20 Sinkhorn iterations, layers 4-27 |
| Context | 524,288 (YaRN 2.0, original 262,144) |
| Vocabulary | 248,320 |
| Vision | 27 layers, patch 16, merge 2, temporal 2 |
| Text weights, BF16 | 18.8 GB |
| Text weights, FP8 tile | 11.0 GB (772 tensors, 11,002,552,832 bytes) |
| Text weights, AWQ 4-bit | ~5.5 GB (official, named by the model card) |

The FP8 mirror `letechlead/SenseNova-Loop-9B-FP8` stores every dense projection as E4M3 with a
BF16 inverse scale a 128x128 tile. Norms, embeddings, the LM head, the convolution, `A_log`,
`dt_bias` and the fp32 mHC routers stay unquantized.

## Can it run on a 16 GB RTX 5080?

Not in BF16. The text weights alone are 18.8 GB. FP8 fits at a short context:

| Form | Weights | KV/token | KV @ 160K | KV @ 256K |
| :-- | --: | --: | --: | --: |
| BF16 KV | 11.0 GB | 56 KiB | 9.2 GB | 14.7 GB |
| 8-bit KV | 11.0 GB | 28 KiB | 4.6 GB | 7.3 GB |
| FP8 body + BF16 LM head on device, embedding on host | 9.0 GB | 28 KiB | 4.6 GB | 7.3 GB |
| AWQ 4-bit + 8-bit KV | ~5.5 GB | 28 KiB | 4.6 GB | 7.3 GB |

Per-token KV is **56 KiB in BF16, 28 KiB in 8-bit**: 14 full-attention visits x 4 KV heads x
256 x 2 B for K and V. The loop is why there are 14 rather than 8 caches: the six full-attention
layers inside the loop hold two independent caches each. The GDN side is 42 recurrent visits,
~84 MiB of f32 state plus ~4 MiB of convolution history per sequence, also per visit.

A prompt runs in chunks of up to **512 rows** through `q4p_dense_fp8`, the tensor-core dense
projection: a 492-token prompt prefills in 542 ms (~900 tok/s), against ~240 tok/s on the
eight-row decode GEMV and 44 tok/s token at a time. The measurement is ~20 TFLOPS, so the tile is
not yet at the card's MMA rate; the projection grid is one to four row tiles against 32 column
tiles for a 4096-wide matrix, which is the occupancy to attack next. Shapes that are not 128-row
multiples (the GDN gate and decay vectors are 32 rows) still take the decode GEMV a tile at a
time, and a forward of eight rows or fewer does too.

Under 512 rows the prefill runs in chunks of up to eight rows, the decode GEMV's register tile: the
projections read their weights once for the chunk, so a 60-token prompt prefills in ~0.25 s
(~240 tok/s) instead of 1.35 s, and the chunk is bit-identical to the single-token steps it
stands in for (the fixture's chunk test holds all three of 2, 3 and 8 rows to a delta of 0).

The arm defaults to **160K**. With FP8 weights, an 8-bit K/V cache, the BF16 LM head on the device
and the embedding on the host, the device plan is 8.4 GiB of weights plus 4.6 GiB of cache at 160K
(13.3 GiB); the server reported 10.8 GiB resident after a short request, because the caches are only
touched as far as the sequence reaches. 256K needs ~7.3 GiB of cache: with the BF16 head that is
16.6 GiB and just misses on a 16 GB card, which is why 160K is the default; the model card's 256K
assumes 4-bit weights. The 5080 is sm_120, not the
sm_121 GB10 the tree was written for: build with `SPARKFLASH_CUDA_ARCH=sm_120`. The consumer part
has no block-scaled MMA, so the prefill images are skipped by the build script; the token, scan
and attention images this arm uses compile cleanly.

## The exact arithmetic

The reference is the released `modeling_sensenova_loop.py`:

```text
plan = 0..4 + (4..28) * 2 + 28..32          # 56 visits, each its own cache
for layer in plan:                           # weights are stored once per physical layer
    h = enter(h, layer)                      # at layer 4: [d] -> [2, d]
    branch, post, mix = mhc_attn(h)          # RMS over flattened 2d, 8 fp32 logits
    x = norm(branch); y = attn(x); h = mix @ h + post ⊗ y
    branch, post, mix = mhc_ffn(h)
    x = norm(branch); y = mlp(x); h = mix @ h + post ⊗ y
    h = leave(h, layer)                      # at layer 27: mean over streams
```

mHC's router is not GLM's and is not interchangeable with it:

* the RMS epsilon is the router's `1e-6`, over the flattened `2 * hidden` stream;
* `pre = sigmoid(logits[0:2])`, `post = 2 * sigmoid(logits[2:4])`;
* `mix = sinkhorn(logits[4:8], 20)`, where each pass subtracts the row log-sum-exp and then the
  column log-sum-exp in log space, and the final matrix is exponentiated;
* the merge is `mix @ streams + post[:, None] * branch[None, :]`.

The host reference in `src/forward.rs` is the oracle for this. The device arm's
`sensenova_mhc_pre`/`sensenova_mhc_post` kernels implement the same schedule. The router's
8 x 8192 projection is one warp a row, summed as a tree over the lanes: the released loop ran
eight lanes over all 8192 columns in order, which is 8192 dependent FMAs on eight lanes and cost
0.4 ms a router - 0.79 ms a layer, the largest single item in the first arm's 125 ms token. The
tiny fixture holds the device to 5e-3 of the host with a BF16 K/V cache (the exact-arithmetic
path) and to 2e-2 with the 8-bit cache, whose E4M3 round is a measured property of the arm rather
than a kernel difference.

## Serving

```bash
# FP8 release on the 5080 (text only):
SPARKFLASH_CUDA_ARCH=sm_120 cargo run --release -p sparkflash-cli -- serve \
    --checkpoint /path/to/SenseNova-Loop-9B-FP8 --context 16384

# CPU reference (the correctness path; one token at a time):
sparkflash serve --checkpoint /path/to/SenseNova-Loop-9B-FP8 --context 4096
```

The dynamic route recognizes `model_type: sensenova_loop`, picks the tokenizer's Qwen ChatML
codec, plans the 56 visits, and opens the device arm when the images exist; otherwise it opens
`SenseNovaHostBackend`.

## The device arm

Every projection and every block body is a shared kernel:

| stage | kernel |
| :-- | :-- |
| embedding | host row copy, 8 KiB a token |
| dense projection | `q4e_gemv_bf16` or `q4e_gemv_tiled_e4m3_tile` (128x128 tile) |
| RMSNorm | `q4e_rmsnorm` |
| GQA q/gate split | `sensenova_qgate` |
| YaRN RoPE | `sensenova_rope` (host-precomputed inverse frequencies and attention scale) |
| 8-bit KV append | `sensenova_cache_q8` (E4M3, one f32 scale a token and KV head) |
| 8-bit attention | `sensenova_attn_full_q8` + `q4e_attn_full_combine` |
| BF16 KV fallback | `q4e_cache_bf16` + `q4e_attn_full` |
| full attention | `q4e_attn_full` + `q4e_attn_full_combine` |
| Gated DeltaNet | `q4e_gdn_conv` + `q4e_gdn_recur` |
| SwiGLU | `q4e_swiglu` |
| mHC | `sensenova_mhc_pre` / `_post` / `_leave` |
| LM head | `q4e_gemv_bf16` (2 GB BF16 on the device) |

The token's launches are captured once, when the session opens, into a CUDA graph
(`SPARKFLASH_SENSENOVA_GRAPH=0` disables it). A graph bakes in its buffers, so it is per session,
and the position is a device word rather than an argument - the split grid is constant and the
rope, cache and attention kernels read `*position` - which is what makes one graph valid at every
index. The 1028 launches replay as one call; the copies that bracket them (one embedding row in,
the logits out) stay outside. The gain is small (22.5 -> 22.4 ms) because the host launch path was
not the bottleneck: the graph removes its cost, not the deviceside per-kernel overhead.

The weights are uploaded once per **physical** layer; the two loop visits reuse them. All 56
visit caches are allocated at open: 14 8-bit KV caches of `context * 1024` bytes plus one f32
scale a token and KV head, and 42 GDN state pairs. The embedding row is the only host upload and
the 1 MB of logits the only download: the LM head's 2 GB BF16 matvec is one device GEMV, because
a host matvec of that table cost 100-200 ms a token, and the row it replaces is 8 KiB.

### What is deliberately next

The first device arm feeds a prompt one row at a time. That keeps prefill and decode on the same
token GEMV arithmetic (a prompt's logits do not depend on its width) and is the simplest correct
walk, but it reads the weights once a token. The next changes, in order:

0. **Multi-row forward (done).** A chunk now walks the whole visit list over up to 1,024 rows:
   the prefill tile 128 rows at a time, the GDN convolution and scan over the chunk, and
   `sensenova_attn_prefill` eight query rows a block. It is checked against the row walk it
   stands in for — `the_device_chunk_matches_its_single_rows_on_the_release` reports a delta of
   0, 0 and 2.66 with the argmax equal at 2, 3 and 8 rows — which is the property the speculative
   verify needs. The rows>1 history is at the end of this file; the last two faults were the
   standalone `b`/`a` row stride and `attach_fp4` releasing the 32-row projections' E4M3 bytes.
1. **Batch the verify, not the launches.** Fusing kernels was measured and does not pay. Two
   variants were built and served at 160K: the norm folded into both mHC routers (plus an
   `add_norm` for the plain-residual visits) ran 22.3 -> 23.2 ms, and `gate`/`up` concatenated
   into one GEMV ran 22.3 -> 22.6 ms, both with fewer launches (1028 -> 924 and 972). The reason
   is that the FP8 tiled GEMV is decode-issue-bound, not bandwidth-bound: it sustains 596 G
   elements/s where the BF16 head GEMV sustains 435, i.e. 37% more elements at half the bytes.
   Fewer, fatter FP8 GEMVs read the same elements and pay the same issue cost; PDL already
   overlaps the small kernels' tails. What does help is doing more tokens per weight pass, which
   is exactly the speculative verify: at 8 rows the 14.1 GB is read once for 8 positions.
2. **Chunked tensor-core prefill.** The 4,096-wide dense projections fit `q4p_dense_fp8`.
   That image used to build empty on the consumer part: one routed-expert entry
   (`q4p_moe_gate_up_fp8`) declares 61 KiB of static shared, over the 48 KiB a CUDA 12.9 ptxas
   allows an sm_120 entry, and one entry over the limit empties the whole image. It is now left
   out of the sm_120 device pass, so the image builds and carries `q4p_dense_fp8`,
   `q4p_dense_mma` and `q4p_dense_rows`. What remains is the arm's side: a chunked GQA and a
   chunked GDN scan over the same rows (both shared images already carry them), and a host loop
   that walks the chunk. The verify keeps the 8-row GEMV; prefill takes the tensor cores.
3. **AWQ 4-bit decode.** The official 4-bit release is the configuration that reaches 256K on a
   16 GB card. It needs one int4 group-wise GEMV (or a dequantize-on-load path) and a 4-bit
   weight plan; the loop/mHC/cache plan does not change.
4. **Prefix reuse across turns.** The per-visit caches already hold everything a continuation
   needs; the arm reports no reusable prefix yet.
5. **Vision.** The 27-layer tower is 0.55 GB in FP8 and is not loaded. Image spans need mRoPE
   positions and the tower's prefill; text-only is the first serve.

## Tests

```bash
cargo test -p sparkflash-model-sensenova
SPARKFLASH_CUDA_ARCH=sm_120 cargo test -p sparkflash-model-sensenova --test device -- --nocapture
```

The fixture is a seeded 8-layer/56-visit-shaped tiny checkpoint; the tests pin the released plan,
the 772-name FP8 plan and its byte total, the host forward, and device-vs-host logits on the 5080.


## Where the 5080 arm stands

The prefill engine is the NVFP4 block-scaled MMA (`mma.sync.aligned.kind::mxf4nvf4`), which one
m16n8k64 instruction a four SM cycles runs at ~890 TFLOPS peak on this part; measured prefill has
climbed 240 tok/s (the eight-row GEMV) → 917 (E4M3 MMA) → 2,017 (NVFP4) → 2,987 (parallel scan) →
3,982 (fused z/b/a, chunked conv) → 5,077 (register-state scan) → 5,640 (persistent GEMM) → 6,244
tok/s on a 569-token prompt with 1,024-row chunks. The E4M3 fallback (`SPARKFLASH_SENSENOVA_FP8=1`,
the path the bisect below uses) is far slower — measured 1,230 tok/s on 834 prompt tokens and 1,063
tok/s on 3,624 through `serve` — which is why the row-length history above is all NVFP4. Re-measured
on the fixed arm it is 4,169 tok/s on 826 prompt tokens and 2,902 on 3,567, at either an 8K or a
160K context; the 6,244 above is from before the `attach_fp4` fix and the two are not yet reconciled.

**The target is now a 15,000 tok/s floor on the shipped FP8 checkpoint.** That floor is a compute
bound, not a byte one: the block-scaled FP8 MMA peaks at 224 TFLOPS on this part and the E4M3 arm
reaches ~56 of them, so 15,000 tok/s (a 1,024-row chunk in 68 ms) is above what 8-bit operands can
reach at any efficiency. It is reachable on the four-bit tile, whose m16n8k64 form peaks at ~890.
Keeping FP8 for decode as well is the other half of the ask, and it does not fit:

| resident | GiB |
| :-- | :-- |
| text weights as E4M3 | 10.25 |
| NVFP4 codes | 5.12 |
| NVFP4 block scales | 0.16 |
| both | 15.53 |
| the card | 15.92 |

which leaves 0.4 GiB for the KV caches, the activations and the scratch, and it fails on the first
48 MiB request (`cuMemAlloc(50331648) failed: 2`). Dropping the combined z/b/a matrix — about
300 MiB across the 32 physical layers — was tried as the way out and does not close it either.

So on a 16 GB part the 4-bit chunk and an 8-bit decode copy cannot both be resident; a decode that
reads FP8 has to be the *same* copy the chunk reads, which means 8-bit weights everywhere and the
slower ceiling, or the 4-bit tile for both, which is what the arm does today.

What makes the trade sharp rather than academic is that the tile is the *worse* kernel at one row.
On the fixed arm, through `serve` at an 8K context:

| | decode | prefill (833 tokens) |
| :-- | :-- | :-- |
| four-bit tile for everything (shipped) | **30.8 tok/s** (32.5 ms a token) | **4,139 tok/s** |
| E4M3 tiled GEMV decode | **45 tok/s** (22.4 ms) — the figure in the status line above | ~1,200 tok/s |

so the eight-bit decode is 46% faster a token and roughly three and a half times slower a chunk.
A 512-row chunk's stage budget, taken with
`SPARKFLASH_SENSENOVA_TIME=2`, is 89.5 ms in total: dense 54 ms (MLP down 14.1, up 10.5, gate
10.4, GDN qkv 5.4, z 3.5, out 3.5, head 2.3, full q 1.8, kv 1.4, o 1.2) and everything else 35.6 ms
(full attention core 9.1, GDN scan 9.0, norm+mHC 9.6, swiglu 2.7, conv 2.5, merge 1.3). A 12,000
tok/s chunk is 42.6 ms, so the GEMM has to carry ~450 TFLOPS *and* the non-GEMM passes have to
halve; the next levers in order are an MMA attention core, a scan that is not one block a head's
32 columns, and folding the norm/mHC/swiglu passes into the GEMM epilogues.

Four correctness bugs found on the way, all of them silent:

- **The NVFP4 block order.** A 16-value block is two 4-byte halves of a 16-byte pair, and a pair
  carries block 0 above block 2 rather than two adjacent blocks; the quantisers pack block `j` at
  `sensenova_fp4_slot` and the GEMM reads a `uint4` and selects the half. `fp4validate` and a
  sweep over every served projection shape now reproduce the quantised reference exactly.
- **The prefill attention reduction.** Each lane held a *different* key but the logits were summed
  across the warp, so a chunk's rows read the wrong keys. It matches the host exactly now.
- **The GDN scan's two warp reductions** used `=` instead of `+=` on a shuffle, so the recall and
  the output carried one lane's 32 keys instead of all 128 — a per-head scalar error that made
  every visit's GDN output wrong. This one *is* what made the served text boilerplate.
- **The chunked convolution's state fill** wrote the taps back to front, so any row block starting
  from history carried the wrong taps.

Text check, run against the f32 host reference (`SPARKFLASH_SENSENOVA_HOST=1`, which serves the
same plan and weights through independent arithmetic): on the FP8 checkpoint the host answers
"The capital of France is" → " Paris.\nParis is a major city in Europe" and "1, 2, 3, 4, 5, 6, 7," →
" 8, 9, 10,". The device now tracks it on a chunk as well as on a decode step:
`the_device_tracks_the_host_over_a_prompt_and_a_few_steps` in `tests/real_parity.rs` reports the
argmax equal at every step of a two-token prompt and at four of five for a five-token one, with the
worst logit delta 4.2–6.1 against the 16.5 it was before the fix below.

### The rows>1 bug: the 32-row projections lost their weights

The host comparison cannot tell a chunk bug from a precision difference, because the two arms fold
their sums in different orders either way. So there is a second test,
`the_device_chunk_matches_its_single_rows_on_the_release`, that asks the device only about itself:
one session prefills `[1..n]` as a chunk, another walks the same tokens one at a time, and the last
rows are compared. A captured graph makes it a two-second run on the real 11 GB checkpoint.

It localised the fault to the NVFP4 arm — the E4M3 arm agreed at every row count — and ruled out the
GEMM: `/tmp/fp4shapes.cu` reproduces the arm's quantiser and GEMM at the served shapes against a
sampled host decode, and extended with two- and three-row cases on both the wide and the narrow form
it is exact there too (`worst 0.000000` at `tokens` 1, 2, 3, 5, 8, 16 and 128).

Two bugs were behind it, both in the arm's own setup and neither in a kernel:

- **The standalone `b`/`a` row stride.** The scan reads `b[row * b_stride + head]`, and on the path
  without the combined z/b/a matrix that stride was `z.rows` — 4,096, the width of a *combined* row
  — while `scratch.b` is 1,024 by 32. Only row 0 landed where the GEMV wrote it. This one was on the
  E4M3 path, and fixing it took that arm's worst host delta over a two-token prompt and four steps
  from 4.93/1.08/2.13/2.12/0.89 to 0.31/0.12/0.21/0.16/0.07.
- **`attach_fp4` converted the 32-row projections and freed their E4M3 bytes.** It walked every
  E4M3 matrix on the argument that every projection now runs the FP4 GEMM. The GDN gate and decay
  projections are 32 rows, so they fail the prefill tile's 128-row guard in `project_matrix` and are
  read by the **decode GEMV** — which needs exactly the bytes the loop had released. They were
  decoding nibbles as bf16. The loop now skips the shapes the tile cannot take, before the conversion
  and before the free, with the same 128-multiple test `project_matrix` uses.

The second one is what the arm has been chasing for several rounds, and it moves everything:

| | before | after |
| :-- | :-- | :-- |
| chunk vs one-row, rows 2 / 3 / 8 | 18.22 / 16.60 / 19.40, argmax flipped at all three | 0 / 0 / 2.66, argmax equal at all three |
| prompt 2, step 0 | argmax 52028 against the host's 220, worst 19.07 | argmax 220, worst 5.43 |
| prompt 5, steps 1–4 | worst 16.5 | argmax equal, worst 4.6–6.1 |
| `NO_ZCB` arm on NVFP4 | constant argmax 169699, 24 from the host | argmax equal, worst 2.2–5.4 |

A chunk is now literally the row walk it stands in for, which is what the speculative verify needs:
the rows it keeps are the rows those single steps would have produced. What is left is
precision-sized — the 4.2–6.1 of logit delta is the NVFP4 activation round plus the difference
between the chunk kernels (`attn_prefill`, `sensenova_gdn_conv`/`_scan`) and the decode ones
(`sensenova_attn_q8`/`_combine`, `q4e_gdn_conv`/`_recur`).

`SPARKFLASH_SENSENOVA_NO_ZCB=1` takes the combined 4,224-column z/b/a matrix out, which is how the
stride bug above showed itself; it is kept because it is a real second path through the visit.

```bash
SPARKFLASH_CUDA_ARCH=sm_120a cargo test --release -p sparkflash-model-sensenova --test device
SPARKFLASH_CUDA_ARCH=sm_120a SENSENOVA_FP8_DIR=/home/allenh1/models/sensenova-fp8 \
  cargo test --release -p sparkflash-model-sensenova --test real_parity -- --ignored --nocapture \
  --test-threads=1
```

The architecture flag belongs on the test run as well as on the build: `cargo test` re-runs the kernel
build script, and without it the release cubin is rebuilt for the `sm_121a` default and the 5080
refuses it with `cuModuleLoadData failed: 209`.

The arm must be built with `SPARKFLASH_CUDA_ARCH=sm_120a`: the block-scaled MMA does not exist for
plain `sm_120`, and an image without it silently falls back to the GEMV path.
