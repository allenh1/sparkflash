# Qwen3.6-35B-A3B (FP8)

Status: **served by the Qwen pack's device arm at TP=1.** The checkpoint is
`Qwen/Qwen3.6-35B-A3B-FP8` (weight selector `qwen3.6-35b-a3b-fp8`), a `qwen3_5_moe` hybrid:
40 decoder layers in 10 blocks of three Gated DeltaNet layers and one full-attention layer,
256 routed experts with eight active, a shared expert behind a sigmoid gate, an MTP stage and a
27-layer vision tower.

The arm is `sparkflash-model-qwen4-exp`, generalised from the Qwen3.8 arm rather than copied.
The same typed config, shared tensor plan (`ArchitectureTensorPlan`), upload walk and kernels
serve both families. The differences are branches on the configuration:

| Qwen3.8 | Qwen3.5/3.6 | Device path |
|---|---|---|
| Four-stream hyper-connections | Plain residual (`hc_count = 1`), zero-centred `1 + w` norms | Hyper blocks are norm-only (no mixer); the injection weights are ones, so the inject and combine kernels are unchanged |
| QSA indexer selection | Full causal gated GQA, `head_dim` 256, partial rotary 0.25 | `q4e_attn_full` + `q4e_attn_full_combine`: 256-key tiles, up to 16 splits a row |
| BF16 dense projections | 128x128-tile E4M3 dense projections and experts | `q4e_gemv_tiled_e4m3_tile` (decode, batch-invariant) and the `q4p_dense_fp8` MMA (prefill chunks) |
| n-gram PLE | none | not built |

FP8 weights are computed W8A16: the f32 activation multiplies the dequantised weight, as the
host oracle does, not the checkpoint's dynamic activation scheme. The arm's architecture
correction (`Architecture::weight_only_fp8`) records that in the execution plan, so a grouped
prefill chunk plans the BF16-activation tile from 64 tokens, as Qwen3.8's does. It does not
plan GLM-5.3's q8 MMA, which the same stored formats would otherwise select. The router, the shared-expert
gate, `in_proj_a`/`in_proj_b`, the norms, the embedding and the head stay BF16, read from each
tensor's stored dtype.

## Serving

```bash
sparkflash serve --model Qwen/Qwen3.6-35B-A3B-FP8 --context 262144
```

The dynamic route reads the checkpoint's `config.json`. When its formats are the ones the
device arm reads, it opens the device arm straight from the safetensors, with no rank-file
conversion. The rule is `sparkflash_ops::parts::dynamic_device_arm`, which the arm's load
(`identity_for`), the CLI route and the web browser share. The runtime profile is
`crates/sparkflash-model-qwen4-exp/config/qwen36_35b_a3b_fp8.toml`.

## What the device arm refuses, and why

Every gap is a refusal at load, never a silent fallback:

- **BF16 routed experts.** The unquantised release stores every expert of a layer fused, as
  `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`. No device kernel reads that layout
  yet. Such a checkpoint is served by the shared CPU host reference, and the browser labels it
  that way.
- **Speculation.** The MTP stage (`mtp.fc` over a plain residual) is planned but not served,
  and no draft head is published for this model. The profile ships `speculation.enabled =
  false`, and the arm refuses `true`. A cached Qwen3.8 DFlash 2 snapshot is never picked up for
  this family; a drafter named in the profile or `SPARKFLASH_QWEN4_DFLASH2` is still loaded, and
  its width is checked against the target's.
- **TP > 1.** The family's profile runs at one rank. The Qwen3.8 TP=2/TP=4 group is not wired
  for the plain residual or full attention yet.
- **Vision.** The tower is stored and planned, but not executed.

## Verification

`crates/sparkflash-model-qwen4-exp/tests/device_parity.rs` runs a tiny `qwen3_5_moe` FP8
fixture (`Qwen4ExpConfig::tiny_qwen35_moe`) through the device arm against the host oracle:

- token decode;
- full attention across several 256-key splits (a 700-token prompt);
- tensor-core prefill;
- the BF16 refusal.

`tests/engine_builder.rs` (`a_qwen35_moe_plan_opens_the_arm`) opens the arm from a
`qwen3_5_moe` execution plan, the path serving takes, with the machine's own hub cache in place.
The parity tests open without a plan, so they never reached the plan's family gate.

`the_released_qwen36_fp8_pack_boots_at_tp1` (sparkflash-cli) runs the serve path's boot gate
on the released `config.json`. The kernel catalogue lists Qwen3.6's routed shape (hidden 2048,
256 experts, top-k 8) next to Qwen3.8's (`QWEN_ROUTED_SHAPES` in `sparkflash-ops/src/parts.rs`),
so its experts resolve to the Qwen images.

The released checkpoint (revision `c653bba`) serves at TP=1 on an RTX Spark N1X (sm_121a,
118 GB unified memory) at the full 262,144-token context. The KV cache is 5.6 GiB, and the
server answers `/health` about 50 s after launch. Greedy answers are coherent. A 5,068-token
retrieval prompt answers correctly, prefilling at about 1.2k tok/s, and a short prompt decodes
at about 57 tok/s. The numerics status is `unregistered`: no reference capture is registered
for this model yet.
