# SenseNova-Loop-9B (FP8 / BF16)

Status: **host reference on every target, and a TP=1 device arm served on the RTX 5080.** The FP8
mirror boots at a 4,096-token context in 11.03 GB of weights, matches the host reference on the
first token (worst logit delta 4.1e-3, same argmax) and decodes at about **13 tok/s** including a
short prompt. The pack is `sparkflash-model-sensenova`; the served selector is
`sensenova-loop-9b-fp8`.

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

| Form | Weights | KV @ 16K | KV @ 32K | KV @ 256K |
| :-- | --: | --: | --: | --: |
| BF16 | 18.8 GB | 0.47 GB | 0.94 GB | 7.5 GB |
| FP8 | 11.0 GB | 0.47 GB | 0.94 GB | 7.5 GB |
| AWQ 4-bit + 8-bit KV | ~5.5 GB | 0.24 GB | 0.47 GB | 3.75 GB |

Per-token KV is **28 KiB**: 14 full-attention visits x 4 KV heads x 256 x 2 B. The loop is why
there are 14 rather than 8 caches: the six full-attention layers inside the loop hold two
independent caches each. The GDN side is 42 recurrent visits, ~84 MiB of f32 state plus ~4 MiB of
convolution history per sequence, also per visit.

On a 16 GB card, FP8 at 16K-32K is the practical first target. The 512K window needs the 4-bit
weights **and** an 8-bit KV cache (the model card's ~15 GiB at 256K). The 5080 is sm_120, not the
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
`sensenova_mhc_pre`/`sensenova_mhc_post` kernels implement the same schedule, and the tiny
fixture's device-vs-host test holds them to 5e-3 on the logits (the device kernel accumulates the
router's 8 x 8192 projection in fp32, so the remaining difference is launch order).

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
| embedding | `q4e_embed` |
| dense projection | `q4e_gemv_bf16` or `q4e_gemv_tiled_e4m3_tile` (128x128 tile) |
| RMSNorm | `q4e_rmsnorm` |
| GQA q/gate split | `sensenova_qgate` |
| YaRN RoPE | `sensenova_rope` (host-precomputed inverse frequencies and attention scale) |
| KV append | `q4e_cache_bf16` |
| full attention | `q4e_attn_full` + `q4e_attn_full_combine` |
| Gated DeltaNet | `q4e_gdn_conv` + `q4e_gdn_recur` |
| SwiGLU | `q4e_swiglu` |
| mHC | `sensenova_mhc_pre` / `_post` / `_leave` |
| LM head | `q4e_gemv_bf16` |

The weights are uploaded once per **physical** layer; the two loop visits reuse them. All 56
visit caches are allocated at open: 14 BF16 KV caches of `context * 1024 * 2` bytes and 42 GDN
state pairs. The forward has no host synchronization between layers; the only read back is the
logit row the sampler needs.

### What is deliberately next

The first device arm feeds a prompt one row at a time. That keeps prefill and decode on the same
token GEMV arithmetic (a prompt's logits do not depend on its width) and is the simplest correct
walk, but it reads the weights once a token. The next changes, in order:

1. **Chunked tensor-core prefill.** The 4,096-wide dense projections fit the shared
   `glm_prefill_fp8b128_mma` shape; the routed-expert-free stack needs a chunked GQA and a
   chunked GDN scan, both of which the shared images already carry. This is the largest single
   prefill win (the Dory and Qwen notes measure 4-10x).
2. **AWQ 4-bit decode.** The official 4-bit release is the configuration that reaches 256K on a
   16 GB card. It needs one int4 group-wise GEMV (or a dequantize-on-load path) and a 4-bit
   weight plan; the loop/mHC/cache plan does not change.
3. **8-bit KV.** The model card's 256K figure assumes it. The cache is the only per-token growth
   this arm has, so halving it doubles the context a card can hold.
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
