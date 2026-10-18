# Device forward chain: the 40-layer hybrid chain and the 130 + 8 golden comparison

Status: **M3-a chain skeleton (host-only) on `task/device-engine-m3`, rebased on `origin/main`
`5ee8599`; no device chain execution and no token/logits claim.** The host skeleton
(`chain.rs`, `window.rs`, `cubins.rs`, the masked comparator and `device-chain --plan-only`)
is code, not design prose; it opens no driver and launches no kernel. The device path has not run
(2.1, 5), so every number below is plan accounting or a labelled measurement on another harness, not
a measurement of this chain. **Phase 2 exit criterion 1 remains OPEN**
(`phase2-sequence-harness.md` section 7; D25). The document still consumes the main-branch
comparison helpers (1.2) and adopts !68's typed capability vocabulary in M1 only (1.3), and it claims
no device result.

Terminology: **chain** = the code that runs the 40 released backbone layers on the device for one
sequence; **capture** = the reference boundary table (v1 manifest + blob) the chain is diffed
against; **fixture** = the released golden sequence fixture (D25); **stage** = a device stage named
in kernel-dev's Phase 3 coverage matrix; **seam** = a host computation on the chain's path;
**hybrid claim** = a PASS defined on a run with labelled host seams, which never claims a device chain
for a stage whose math runs host (3.2).

## 1. Where the chain lives, and the dependency direction

### 1.1 Verified edges (read from the manifests, not assumed)

| crate | workspace dependencies in `Cargo.toml` |
|:--|:--|
| `sparkflash-engine` | **none** (serde, serde_json, sha2, thiserror only) |
| `sparkflash-kernels` | `sparkflash-trace`, `sparkflash-engine` |
| `sparkflash-api` | trace, engram, kv (+ axum, tokio, tower-http, tokenizers, deepseek-recipe, prometheus) |
| `sparkflash-scheduler` | comm, trace, kv, engram |
| `sparkflash-cli` | trace, api, engram, kv, comm, kernels, scheduler, engine (bin-only, no lib target) |

`kernels -> api` **would be acyclic**: no crate `sparkflash-kernels` depends on (`trace`,
`engine`) depends on `kernels`. It is rejected on weight and layering, not on a cycle the graph
does not have - and not because the graph has no edge in that direction: the last row above is exactly
`sparkflash-cli -> sparkflash-kernels`. The api manifest pulls the HTTP/tokenizer stack and the
kernel crate is deliberately a CPU-only-buildable ABI crate ("nothing links libcuda and a CPU-only
build is unaffected", `sparkflash-kernels/Cargo.toml`). `sparkflash-engine` is a leaf by
construction (same manifest: "the engine crate depends on no other workspace crate"), so hosting
the chain there would make engine -> kernels and kernels -> engine a hard cycle.
`sparkflash-cli` is a `[[bin]]` with no library target, so a test crate cannot import from it.

### 1.2 Decision: a new crate, `sparkflash-device`

~~~
sparkflash-cli -> sparkflash-device -> { sparkflash-api, sparkflash-kernels, sparkflash-engine,
                                         sparkflash-kv, sparkflash-scheduler, sparkflash-trace }
~~~

No cycle: none of those six crates depends on `sparkflash-device`, and `cli` gains the one edge it
needs to select the engine. Each dependency has a named use: `sparkflash-api` = the
`InferenceEngine` trait and `EngineRequest`/`Capabilities` surface; `sparkflash-kernels` =
`cuda_driver` (`KernelId`, `Cuda::load`, `copy_dtod`), `cuda_harness`
(`flash_mla_params`, `launch_flash_mla`, `WindowRingLayout`, `rotate_window_ring`),
`m4_abi`, `sequence_chain` (`Format2AttentionCall`, `SeamLedger`, `BoundaryStage`,
`SequencePlan`); `sparkflash-engine` = reference weights, `forward::{mhc, moe, engram,
boundary}`, `sequence_fixture` comparators; `sparkflash-kv` = `schedule::ced_schedule`,
`DeviceKvStore`, `INDEX_TOPK`; `sparkflash-scheduler` = `StepBoundaryScheduler`;
`sparkflash-trace` = the spans that already bracket engine steps.

**The APIs are on merged main.** The chain APIs this crate consumes landed with !63 (merged as
`1ceeb54`; main `4967d68`): `sequence_chain` (`Format2AttentionCall:86`,
`SeamKind:223`, `SeamRecord:236`, `SeamLedger:266`, `BoundaryStage:517` with
`ORDER:554`, `LayerExtent:415`, `SequencePlan:437`, `reference_tensor:565`,
`compare_reference_f32:582`, `compare_reference_bytes:605`, `compare_boundary:370`),
`Cuda::copy_dtod` (`cuda_driver.rs:381`), and the window-ring helpers `WindowRingLayout`
(`cuda_harness.rs:283`), `ring_view_position` (`:345`), `ring_view_segments` (`:366`),
`rotate_window_ring` (`:413`). The comparison contract's fixture helpers are there too
(`sequence_fixture.rs:55` `LOGITS_ABS_BAR`, `:415` `compared_positions`, `:427`
`exclusion_boundary`, `:699` `compare_steps`, `:753` `compare_logits`). M1 (1.3) does
not need these APIs and can land first; M2/M3 build against the merged symbols.

Rejected, with the reason rather than a preference: the chain inside `kernels` (pulls the
HTTP/tokenizer stack into the ABI crate); inside `engine` (cycle); inside `cli` (no library
target, so no tests); extending the `m8_sequence` example (it is a window harness with its own
plan, and examples cannot be imported).

### 1.3 Iteration-2 boundary and the !68 relationship

The first milestone (M1, section 7) is deliberately narrow: `DeviceEngine` behind a capability flag
with a **fake device session**, implementing the `InferenceEngine` trait that exists on main,
reporting `vision: false` on main (`Capabilities.vision` at `sparkflash-api/src/engine.rs:379`;
`Capabilities::ALL` sets it `true` at `:391`, while the derived `Default` (`:376`) and the trait
default (`Capabilities::default()`, `:414-415`) are `false`), refusing a non-greedy configuration,
and delegating `generate` to the reference engine. After !68 lands the field is gone (its
`Capabilities` has no `vision`: `engine.rs:399` on the !68 branch) and
the same conservative answer
is the constant `multimodal: false` in the `/v1/models` output (`server.rs:1167` on that branch);
M1 adopts whichever vocabulary it is built against. Task-3 iteration 1 (!68) adds the typed
capability vocabulary (`MissingCapability`/`validate_request`); M1 adopts it when merged, but M2
and M3 do not depend on it.

The delegation cannot be a call into a binary: `ReferenceDriver` lives in
`crates/sparkflash-cli/src/serve.rs` and the cli crate has no library target (1.1), so
`sparkflash-device` cannot name it. M1's `DeviceEngine` takes the reference
`Arc<dyn InferenceEngine>` as a constructor parameter and the cli wiring supplies its own
`ReferenceDriver`; the library never depends on the bin. M1 also does not need the !63 chain APIs (1.2).

## 2. The chain graph

### 2.1 Inputs

Prompt 130 token ids, a checkpoint snapshot, the CED schedule (`ced_schedule(130)`,
`kv/src/schedule.rs:141`, `CedSchedule:82`): encoder layers 0..=20 over 130 positions, decoder
layers 21..=39 over positions 2..=129; modes 0-1 SWA-only, 2/8/14/20 Full, 24/28/32/36 Reindex, the
rest Reuse; HSI after layer 20. That **accounts for 5,162 prefill token-layer executions** (21 x 130
+ 19 x 128) and **320 decode executions** (8 x 40), 5,482 in total: this is plan accounting of the
corrected !63 code, not a total the m8 report measured (`phase2-sequence-harness.md:203-217`).
The corrected device path has **not run** (the m8 note says the corrected binary has not run on a
device; section 5), so 5,482 is not a measured chain total.
No prompt text, tokenizer, non-greedy sampling, MTP, vision, batch > 1, or second sequence (2.3).

### 2.2 Per-layer graph

Engine reference: `forward/mod.rs:373` run, `:546` block. Per layer, in order:

1. Engram read/write at its layers (`forward/engram.rs:602`, `:797`); seam `engram_rw`.
2. mHC entry: `hc_mixes + hc_pre` (`forward/mhc.rs:128`, `:310`, `mod.rs:568-600`;
   the previous sublayer's `hc_post` is folded in, `phase1-boundary-table.md:731-734`). The entry
   is followed by the attention pre-norm `rms_norm` (host seam `attn_norm`), then the item-3
   projections and RoPE; the stage order is mHC entry -> norm -> projections -> RoPE.
3. Attention projections: `mxfp8_dense` (wq_a/wq_b/wkv/wo_b path; after the norm) and `wo_a`
   (`attention.rs:1758`), kernels `MXFP8_DENSE` / `MXFP8_WO_A`; RoPE follows the q/kv
   projections, and `wo_a` is the output projection after attention.
4. Window-KV write/rotate: 528 B host/audit row (512 B code + 16 B scale), two device arrays per
   layer, `cuda_harness::rotate_window_ring:413`, `WindowRingLayout:283`,
   `ring_view_position:345` (all on main; 1.2).
5. Attention: format 2 (38 layers) through `sequence_chain::Format2AttentionCall` (`:86`) and
   `launch_flash_mla` (`cuda_harness:178`, params `:71`); layers 0-1 format 1 (D24). The builder
   is on main (`:86`, `params()` at `:139`); its null-indexer validation (`:169-180`) is the
   task-9 fix (section 6).
6. Post-mix mHC: `hc_post` (`mhc.rs:378`), kernel `MEGA_MHC`.
7. MoE: route (`moe.rs:462`); the device router kernel `moe_router_sm121` is measured with one
   launch in the M2 window but not wired into the chain, and is not a host seam. The routed expert
   **gate/up** (`w1`/`w3`) is not dispatched by the chain: its merged typed kernel
   `routed_gate_up_sm121` (task-8; measured in isolation in its own harness `m10_gate_up`,
   `launch_repeated`) is **not wired into the chain yet** and is not a host seam; its stage status
   is "device kernel merged and measured in isolation (m10), not wired into the chain yet". SwiGLU
   and the activation quantiser are still the host seam `moe_swiglu_quant`
   (`swiglu_mxfp8_quant_sm121` is pinned in the baseline but not launched by the chain and has never
   been compared: `roadmap-to-serving.md:438`); applying the router's weights is the host seam
   `moe_routing_weight`; the routed **down/combine** projection is device
   `MXFP4_MOE` (its kernel computes down/combine only and does not duplicate gate/up:
   `kernels/b12x_mxfp4_moe_sm121.cu:33-34`); then residual out (`boundary.rs:140`). The shared
   expert's dense gate/up GEMM is device `MXFP8_DENSE`; its SwiGLU and activation quantiser are
   the host seam `shared_expert`.
8. Tail, once: collapse `hc_pre` (`mod.rs:488`), `rms_norm` (host seam `final_norm`),
   certified head screening `cert_lm_head` (`kernels/src/lm_head_ref.rs:109-151`, kernel
   `CERT_LM_HEAD`), then the block reduction over the per-block maxima and the greedy argmax (host
   seam `lm_head_argmax`). The kernel deliberately does **not** do the grid-wide reduction
   (`kernels/cert_lm_head_sm121.cu:27-30`), so the reduction is part of the token path.

Kernel ABI anchors (main): `KernelId` `cuda_driver.rs:157`; `m4_abi` `MEGA_MHC:854`,
`MXFP8_DENSE:856`, `MXFP4_MOE:858`, `FLASH_MLA_CSA2:860`, `MXFP8_WO_A:862`,
`CERT_LM_HEAD:864`, `KERNELS` at `:867` (a six-entry array, frozen by D26). The ratchet baseline
`crates/sparkflash-kernels/baseline/kernel_baseline.txt` has **nine** entries: the six m4 kernels,
`swiglu_mxfp8_quant_sm121` (host seam, not launched by the chain), `moe_router_sm121` (M2
one-launch proof) and `routed_gate_up_sm121` (task-8, measured in isolation, not wired). The
chain's six-kernel launch set is unchanged: `mega_mhc_sm121`, `b12x_mxfp8_dense_sm121`,
`b12x_mxfp4_moe_sm121`, `flash_mla_csa2_sm121`, `b12x_mxfp8_wo_a_sm121`,
`cert_lm_head_sm121`. Routed gate/up and the shared expert's dense GEMM are device kernels that
are not wired into the chain yet; SwiGLU/quantiser, routing weight, and the head's block reduction
are host seams.

### 2.3 Explicitly outside the chain

MTP (3 stages; a decode-step capture would have 43 rows, `roadmap-to-serving.md:505-510`), batch >
1, multiple sequences, vision, non-greedy sampling, DSpark/MTP draft heads, multi-node. A run that
needs any of them is refused with a typed error, not silently approximated. Consequence for the
comparison: **decode steps get no boundary capture**; their comparison is tokens only (4.2). In the
first hybrid run the FFN stage is still hybrid, because the merged routed gate/up kernel is not
wired into the chain yet (its stage is labelled "merged and measured in isolation (m10), not wired
into the chain yet", not "host seam"); SwiGLU/quantiser and routing-weight application remain host
seams until their device stages exist (3.2a).

## 3. Host seams, the per-run ledger, and the coverage classification

Rule: **every host computation the chain performs is a seam with a stable name, and the run's plan
constructs the expected seam set before the run.** The types are on main (landed with !63): `SeamKind`
(`sequence_chain.rs:223`), `SeamRecord` (`:236`: name, kind, host_bytes_in, host_bytes_out,
transfer_bytes, `phase3_followup`, note), `SeamLedger` (`:266`) with `report_lines()`
(`:302`). The chain records every seam, prints the ledger **before** any comparison line, and fails
its self-check if a planned seam has no record or a record has no plan entry. A host buffer crossing
the boundary without a record is what the ledger cannot see by itself, so seam calls are the only
place allowed to touch host tensors and review enforces that rule.

### 3.1 The seam set

| seam | `SeamKind` | host work | removal |
|:--|:--|:--|:--|
| `embedding_gather` | Other | the 138 prompt+decode rows x 5,120 = 706,560 gathered elements (lookup `forward/mod.rs:401-432`; the `embed.weight` table is 661,913,600 elements, `linalg.rs:81-82`) | embedding kernel (Phase 3) |
| `engram_rw` | Engram | Engram hash/read/write (`forward/engram.rs:602,797`) | device head (Phase 3) |
| `attn_norm` | Other | q/kv RMSNorm before attention (`attention.rs:1758` prologue) | fused norm+RoPE kernel |
| `attn_rope` | Other | RoPE on q and the output side (`attention.rs:250` `rope_tables`) | fused norm+RoPE kernel |
| `window_kv_roundtrip` (matrix S1) | Other | encode window-KV rows and ring rotation (device `copy_dtod`, host plan) | device KV store S1 |
| `compressor_cache` | Other | compressor outputs + compressed-cache FP4 round-trip | compressor kernel |
| `index_key_path` | Other | index-key `wk`, `k_norm`, RoPE, store (8 layers) | index-key kernel |
| `indexer_weights` | Other | `indexer.weights_proj` (8 layers, bf16 32x5120) | indexer kernel |
| `hsi_scoring` | Other | HSI pool population and block scoring candidate list | HSI/indexer kernel |
| `top512_merge` | IndexerSelection | Top-512 merge over the pool (`INDEX_TOPK = 512`, `kv/src/config.rs:35`; engine `attention.rs:1324`) | indexer/top-k kernel |
| `expanded_view` | Other | expanded/compressed store view for the format-2 selected set (D24) | kernel (Phase 3) |
| `moe_routing_weight` | Other | apply the router's top-6 weights to the routed expert outputs | routed weighted-sum kernel (Phase 3) |
| `moe_swiglu_quant` | Other | routed and shared SwiGLU and the activation quantiser (`roadmap-to-serving.md:438`) | fused SwiGLU + activation-quantiser kernel (Phase 3) |
| `shared_expert` | Other | shared-expert SwiGLU + activation quantiser around the dense GEMM | fused shared-expert kernel |
| `dispatch` (matrix S2) | Other | launch scheduling; no arithmetic | Phase 3 CUDA-graph runner |
| `final_norm` | Other | final RMS norm (`linalg::rms_norm`) | final-norm kernel |
| `lm_head_argmax` | Other | the reduced greedy argmax over the four ranks' 32 per-block maxima over the 129,280 screened entries (lowest-index tie-break, `lm_head_ref::best_entry:161`; the kernel's finalisation is separate, `cert_lm_head_sm121.cu:27-30`) | head/sampling kernel |

Sampling is not a separate seam: the chain is greedy-only and refuses a non-greedy configuration
with a typed error, so there is no host sampling fallback to hide. Every seam sets
`phase3_followup = true`; a seam without a removal plan is not allowed.

### 3.2 Which seams may be host-side, and which may not

kernel-dev's Phase 3 coverage matrix (relayed 2026-10-18) is authoritative on kernel status; the
attention-side ops it lists are the chain's real coverage blockers. Three classes:

**(a) Host-side allowed for the first end-to-end chain, labelled and counted, stage not claimed as a
device stage.** This covers every attention prologue/cache op: `attn_norm`, `attn_rope`,
`window_kv_roundtrip` (S1), `compressor_cache`, `index_key_path`, `indexer_weights`,
`hsi_scoring`, `top512_merge`, `expanded_view`; plus `embedding_gather` (input side, not on
the output token path), `engram_rw`, `dispatch` (S2 control, not arithmetic), and
**`moe_swiglu_quant` / `moe_routing_weight` / `shared_expert`**: the SwiGLU/quantiser and the
routing-weight application are host seams, and the shared expert's SwiGLU/quantiser is a host seam
around a device dense GEMM. Routed gate/up is **not** in this class: its kernel is merged on main
b54f11e and measured in isolation (m10), but not wired into the chain, so the stage map says
"device kernel merged and measured in isolation (m10), not wired into the chain yet". A first chain
that runs gate/up on the host anyway is a **HYBRID chain for the FFN stage**; it may run, be
compared and localise a failure, but **no device chain may be claimed for the FFN stage** until the
kernel is wired. The report sentence must say "FFN: host gate/up (device kernel merged, not wired) +
host SwiGLU/quantiser over device down/combine". The same "not a device chain for this stage" rule
applies to every seam above.

**(b) Cannot be host-side for a device-chain claim: `final_norm`, `lm_head_argmax` (the block
reduction plus the greedy selection, and the sampling it implies).** These are the token path
itself. If the final norm or the argmax runs on the host, the produced token is not device math, so
no stage of that run may be called a device chain, and Phase 3 exit criteria 1 and 3 (a tokens/s
number measured on the chain; no unexplained host-seam gap) cannot be claimed. The first **hybrid**
run for Phase 2 criterion 1 may run them on the host only if the report says "hybrid: token produced
by host math over device logits"; that run cannot claim any Phase 3 exit criterion, and it cannot
claim even the Phase 2 token row as device evidence (the token row may pass as a hybrid comparison).

**(c) Already device math.** Dense projections (`MXFP8_DENSE`, `MXFP8_WO_A`), format-1/2
attention through the fixed builder (`SequencePlan` / `Format2AttentionCall`, on main), the routed
**down/combine** projection (`MXFP4_MOE`), and mHC (`MEGA_MHC`); the shared expert's dense GEMM is
covered by `b12x_mxfp8_dense` per the matrix. The certified head is **not** in this class:
`CERT_LM_HEAD` is a screening kernel whose header still records "never executed"
(`kernels/cert_lm_head_sm121.cu:31`, stale: one stage-5 launch per rank is recorded below;
`roadmap-to-serving.md:440`); it writes an upper bound for entries it certifies and exact values
only for uncertified entries (`src/lm_head_ref.rs:109-151`),
and its per-block maxima are finalised outside the kernel. The only recorded device execution of
this kernel is the layer-0 stage-5 harness (`phase1-boundary-table.md:546-559`): one launch per
rank (four launches), over the four 32,320-row shards of `head.weight`, all 129,280 entries exact
and **0 certified**, its argmax equal to that harness's
bf16-matched host reference - a single hidden state from the layer-0 harness, not a chain run, and
the certificate path (pruning) was never exercised. The head is therefore labelled **NOT YET
DEVICE-EXECUTED AS A PRUNING PATH: one stage-5 launch per rank, 0 certified, never compared against
the chain reference**; the first chain report says "first chain execution, unverified", and
its contract is 4.2/4.3, not "already device".

## 4. Comparison contract

### 4.1 Truth, order, and where the boundaries come from

D25: the fixture is the truth. Identity: 2,536,194 B, sha256
`e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308`, node
`/home/user/phase2-evidence/sequence-130-released.json` and dev-host copy of record under
`/home/user/phase2-evidence/devhost-copy-f72738b/` (phase2-golden-fixture.md section 2). Prompt
130 ids; prefill next token **491**; decode tokens **367, 201, 80, 3302, 201, 54, 201, 4341**;
prefill logits 129,280 f32; every step has a sha256 digest, only the prefill step has values.

The boundary reference is one CPU capture from the engine's own forward (the `--boundary-dump` mode
added for this cross-check, branch commit `277f657`, `examples/m8_sequence.rs`): 40 rows x 130
positions x the 18 tensors of `boundary::TENSOR_ORDER` (`forward/boundary.rs:144`), written as
a v1 manifest + blob (expected ~1.76 GB; the manifest comes back, the blob stays on the node). The
device chain compares through `sequence_chain::BoundaryStage` (`:517`, `ORDER` at `:554`),
`reference_tensor` (`:565`), `compare_reference_f32`/`compare_reference_bytes` (`:582`,
`:605`) and `compare_boundary` (`:370`) - all on merged main (1.2) - with the pre-registered Stage-B
per-tensor bars (the same bars the layer-0 fast path used, not tuned for this run).

### 4.2 What is compared, where, and against what

| quantity | positions | bar / rule | gates PASS? |
|:--|:--|:--|:--|
| tokens (prefill next + 8 decode) | 129, 130..=137 | exact equality; in the first run the token is host-reduced (3.2b) | **yes**, as a hybrid comparison |
| head screening (certificate) | 129 | the reduced argmax over the screen's per-block maxima (host `lm_head_argmax` in the first run; lowest-index tie-break `lm_head_ref::best_entry:161`) must equal the token; certified entries are upper bounds and cannot be per-entry compared | **yes**, via the token row (hybrid token) |
| logits | 129 only (the only captured values) | absolute max diff <= 2e-2 (`LOGITS_ABS_BAR`, `sequence_fixture.rs:55`) **over the entries the screen left exact (certified == 0) plus the argmax entry**, by the masked comparator (below) | **yes**, restricted |
| boundary, 18 tensors x 40 layers | **129 only** | Stage-B bars through `compare_boundary` (`sequence_chain.rs:370`); the boundary compare is at 129 only (the prefill element), while `compared_positions` (`:415`) is the token-step set [129, 130..=137], not a boundary set | no (localises; printed) |
| boundary, encoder layers 0..=20 | 2..=128 | same bars, **diagnostic only**, labelled | no |
| boundary, decoder layers | 2..=128 | not comparable: CED extents differ outside the window (D7; `:405-419`) | never |
| boundary, positions 0..=1 | any | never compared (`exclusion_boundary`, `:427`) | never |
| boundary, decode steps | 130..=137 | not captured: 43-row rule needs the 3 MTP stages (`roadmap-to-serving.md:505-510`) | never |

Why only position 129 for logits and boundaries: the engine reference runs all 40 layers over 130
positions, while CED runs decoder layers 21..=39 over positions 2..=129; only at 129 do the attended
sets coincide. The boundary/logits capture is at 129 only; `compare_steps` (`:699`) refuses an older
prefill step (D7), and `compared_positions` (`:415`) names the token-step set [129, 130..=137],
not a wider boundary set. The decode **tokens** at 130..=137 are compared exactly; the exclusion is
the numeric/boundary capture, not the tokens.

**Head contract.** `CERT_LM_HEAD` returns, per block, the block maximum and argmax over an upper
envelope: an entry with `certified == 0` holds its exact logit, an entry with `certified == 1`
holds only the certificate bound (`src/lm_head_ref.rs:109-151`; kernel header
`kernels/cert_lm_head_sm121.cu:9-24`). Because a certified bound is strictly below the running
maximum that pruned it, the winner is always exact; the maximum and argmax are exact by the
certificate, and a comparison of all 129,280 entries at 2e-2 is impossible from this kernel. The
**reduced argmax** is the argmax over the kernel's per-block maxima; in the first run that reduction
runs on the host (`lm_head_argmax`, 3.1) with the lowest-index tie-break of
`lm_head_ref::best_entry:161`, so the token is exact by the certificate but its production is
**hybrid**: host math over device candidates (3.2b). The chain therefore claims (i) the token as a
hybrid token comparison (4.3), and (ii) the per-entry logits only where the device screen returned
exact values plus the argmax entry. The restricted logits comparison cannot use
`sequence_fixture::compare_logits` (`sequence_fixture.rs:753`): it compares every recorded entry
and takes no mask. The chain uses a masked comparator instead - a new helper, e.g.
`compare_logits_exact(fixture, device, certified_mask, bar)`, that skips entries whose certified
bit is 1 and requires the mask's shape to match the device logits. The device report must carry the
certified mask and the certified/exact counts so the compared set is auditable. An all-entry
comparison may be printed only as a **host measurement, not a device output**: position 129's device
hidden state can be fed to a labelled host exact-head recomputation (`lm_head_ref::exact_logit`,
`:99`) and the resulting per-entry deltas reported as diagnostics - the fixture's logits exist only
at prefill 129, so the all-entry comparison is 129-only (decode steps are digest-only and compared as
tokens separately). Those deltas do not gate PASS and cannot be quoted as device logits.

**Mismatch report**, printed and written before any verdict: for a boundary divergence, the first
occurrence in (layer ascending, `TENSOR_ORDER`, index ascending) with layer, tensor, its
`BoundaryStage` name when one exists, position, index, max-abs, bar, measured/bar, and the channel
(`index % 5120` for f32 rows; code byte, 32-element block and E8M0 scale indices for the quantised
pair); then per-stage totals and the aggregate. For a token mismatch: step, position, expected,
actual, and whether the fixture captured logits there (decode steps have only a digest, so no numeric
delta is reported). For the head: the certified/exact counts, the argmax entry, and whether the
compared logits set is the restricted set. Missing compared positions, extra older positions, and a
missing device tensor are errors, not passes.

### 4.3 The exact PASS definition for Phase 2 exit criterion 1

Criterion 1 is "prefill of a prompt over 128 tokens and at least 8 decode steps, every decoded token
reproduced against the reference" (`roadmap-to-serving.md` section 5.5 item 1). The chain claims it
only when **all** hold in one run:

1. exit status 0, and the run actually executed on a device: driver loaded, every cubin sha256
   printed and equal to the nine parsed entries of `baseline/kernel_baseline.txt`; the catalog
   requires eight of them (the six chain kernels plus the router and routed gate/up, 2.2), and the
   ninth, `swiglu_mxfp8_quant_sm121`, is a host seam, not a chain kernel. Device kernel launches
   > 0 (a host-only run cannot pass);
2. tokens exactly 491, 367, 201, 80, 3302, 201, 54, 201, 4341 (`compare_steps`, `:699`);
3. head reduction: the reduced argmax over the screen's per-block maxima (host `lm_head_argmax` in
   the first run; lowest-index tie-break `lm_head_ref::best_entry:161`) equals the expected token,
   and the report carries the certified/exact mask (the exactness of max/argmax is the
   certificate's, not a per-entry comparison). This is the **hybrid token comparison**: the report
   says "hybrid: token produced by host math over device logits", and `CERT_LM_HEAD` is labelled
   **NOT YET DEVICE-EXECUTED AS A PRUNING PATH: one stage-5 launch per rank, 0 certified, never
   compared against the chain reference**; the first chain report says "first chain execution,
   unverified";
4. logits at 129: absolute max diff <= 2e-2 over the entries the screen left exact plus the argmax
   entry, computed by the masked comparator (4.2) - not `sequence_fixture::compare_logits`, which
   takes no mask; the all-entry host recomputation is printed only as "host measurement, not a
   device output" and does not gate PASS;
5. the seam ledger matches the planned set and every seam names its `phase3_followup` removal;
6. complete metadata: binary sha256, cubin sha256s, fixture sha256 + size, checkpoint snapshot, argv,
   UTC start/end, host, MemAvailable before/after, exit 0.

Boundary capture does **not** gate criterion 1; it localises a failure. A run passing 1-6 with a
boundary tensor over its bar is a criterion-1 PASS **with a recorded divergence**, and the claim
sentence must name it - it is never silently dropped. Widening the compared set by one older position
is not a PASS (it fails `compare_steps` by construction, D7); a token or logits failure is a FAIL
regardless of boundaries. This is a **hybrid** claim: host seams are labelled and counted, and the
claim sentence names the stages that are not device (section 3.2) - including the token path (host
final norm and host block reduction over device logits, 3.2b) and the FFN stage while routed gate/up
is merged-but-not-wired. It is not a device-chain claim for those stages, and criteria 2-5 and
anything serving-related remain unclaimed.

## 5. Node window plan

The window is the only place criterion 1 can be claimed. The plan separates **measurement phases**
so no number is contaminated by another phase's clock (blocker B, section 6). Every precondition is
checked before a kernel launch and its value written into the report; a failure aborts nonzero with
the command line recorded, never a retry.

Preconditions, in order: **build identity** (binary sha256 and every cubin sha256 vs
`baseline/kernel_baseline.txt`; the device build goes through `scripts/build-aarch64.sh`, whose
`--example=*` path hardcodes `-p sparkflash-kernels` (`:72`), so this design adds a
`--package=` selector rather than moving the chain into kernels); **fixture** (staged and re-hashed
on the node; a mismatch aborts); **boundary capture** (produced by the same built binary and
checkpoint snapshot as the run: that makes **build and provenance** the intended single variable, and
nothing more - the capture is CPU f32 while the chain is device bf16/FP4 arithmetic, so a numeric gap
is not "a build difference" and is localised by the boundary compare, not explained away);
**memory floor** (`MemAvailable >= 32 GiB` before the run; the measured minimum during the
stage-9 chain was 122,285,404 kB, `phase1-boundary-table.md:736-745`; m8's
`DeviceBuffers::allocate` (`examples/m8_sequence.rs:2553-2605`) is 176 buffers / 9,926,144 B,
of which the window rings are forty rings x 67,584 B = 2,703,360 B - that is m8's attention/KV
plumbing set, a **lower bound**: it holds no weights and no head, and the head alone is ~1.3 GB bf16
unsharded (~330 MB/rank), so the run budget must add weights and head); **configuration** (section 2.3
only).

Phases, each writing `runs/<utc>-<phase>-<sha>.txt`:

| phase | command | claim |
|:--|:--|:--|
| smoke | `sparkflash-cli device-chain --phase smoke --positions 4 --layers 1` | ABI/loader only; it records its own wall and no numerics; there is no measured smoke anchor in the repo, so none is quoted. |
| rotation | `... --phase rotation` | ring rotation alone; the plan is 5,482 executions / 11,764 `copy_dtod` / 376,576 slots / 198,832,128 B (codes 192,806,912 B + scales 6,025,216 B; one full ring 67,584 B, forty rings 2,703,360 B) from the corrected !63 geometry (`phase2-sequence-harness.md:203-217`, on main). The copy-only rate is **measured**: the corrected run of the 2026-10-18T18:32Z window issued 11,764 `copy_dtod` calls and moved 198,832,128 B in 0.025624 s of copy time, i.e. **7,400.000 MiB/s copy-only**; the loop wall 2.588988 s also covers view verification and the attention dispatch (DECISIONS.md D25, [phase2-sequence-harness.md](phase2-sequence-harness.md) section 7). The contaminated first commit's (`8c91fe5`) rotation seconds and MiB/s remain superseded and must not be quoted. |
| dispatch | `... --phase dispatch --positions 130 --layers 40` | launch+readback with rotation excluded. The command's grid is 5,200 launch slots (130 positions x 40 layers), but the chain executes **5,162** prefill token-layer executions because the decoder layers start at position 2 (2.1); 5,200 is the grid, not the execution count. This phase has **no measured anchor** of its own. The only dispatch anchor is the corrected 2026-10-18T18:32Z window: 360 first-launch records (9 positions x 40 layers: 18 window-only format 1 + 342 union format 2), dispatch wall 0.175593 s over those 360 first-launch records = 487.758 us per record, 680 launches in all including 320 determinism repeats, 0 non-finite outputs and 40 device-vs-twin comparisons within the 2.0e-2 bar (DECISIONS.md D25, `phase2-sequence-harness.md` section 7). The earlier 17:16Z window measured 0.174152 s / 483.756 us over its 360 launches; the corrected run above is the run of record for this row. The stage-9 40-row forced-operand wall (298.6 s, two chain runs, `phase1-boundary-table.md:736-746`) is a different harness and is not a dispatch-only clock. |
| full | `... --phase full --prompt 130 --decode 8` | the criterion-1 run: exit 0, tokens, position-129 logits (restricted set, 4.2), seam ledger, metadata; boundary differences printed (4.3). |

Abort rules: baseline mismatch, fixture mismatch, a planned seam missing, MemAvailable below floor on
start or during run, a configuration outside 2.3, or a mid-run device error. The full phase runs in
one process so the ledger belongs to one run - the same run the PASS definition refers to.

## 6. What the chain consumes (blockers, ordered)

* **!63 (resolved; merged as `1ceeb54`; carries the task-9 fix).** The builder is on main now: the
  branch's first commit (`8c91fe5`) passed a literal 0 for the four format-2 indexer pointers
  (`Format2AttentionCall::params()` at `:166-170` on that commit) while refusing only a null
  `selected` pointer; the fix is the refusal of null indexer buffers in `params()`
  (`:169-180` on main), and the merged commits (head `6421ad5`) also carry the plan/rotation
  fixes (1.2). The chain calls that builder (38 of the 40 layers are format 2, D24) and carries no
  private copy.
* **Blocker A - dispatch plan (resolved by design).** m8 anchors rotation via `window_view` /
  `layer_first_position` but its *dispatch* plan uses `(position + 1).min(WINDOW)`
  (`examples/m8_sequence.rs:1301`); the chain uses `sequence_chain::SequencePlan` /
  `LayerExtent` (`:437`, `:415`) instead.
* **Blocker B - measurement interleaving (resolved by the phase split).** `measure_rotation` clocks
  a synchronised copy loop (`examples/m8_sequence.rs:1147`) while `run_dispatch` clocks
  launch+readback (`:2876`); timing both in one loop reports neither truthfully. Rotation and
  dispatch are separate phases, and phases the run did not measure are reported "not measured".
* **Routed gate/up kernel (FFN stage blocker for a device chain).** **Resolved on main b54f11e:**
  the K = 5120 packed-FP4 gate/up GEMM kernel `routed_gate_up_sm121` is merged (task-8) with a
  typed launcher and was measured in isolation in the m10 window (30,720/30,720 and
  499,200/499,200 exact). It is **not wired into the chain yet**, so a first chain still runs
  gate/up and the SwiGLU/quantiser on the host: the FFN stage is a labelled **hybrid** and cannot be
  claimed as a device stage until the kernel is wired. If the position-129 logits bar fails because
  of that host arithmetic, that is a coverage failure: wire the kernel, do not widen the bar.
* **Certified head verification and the block reduction (token-path blocker).** `CERT_LM_HEAD`'s
  header still records "never executed" (`kernels/cert_lm_head_sm121.cu:31`), stale: one stage-5
  launch per rank is recorded (3.2c). It is a screening kernel (4.2), so the head is labelled **NOT
  YET DEVICE-EXECUTED AS A PRUNING PATH: one stage-5 launch per rank, 0 certified, never compared
  against the chain reference** (3.2c); the first chain report says "first chain execution,
  unverified", and the all-entry logits comparison is impossible by design. The block reduction over
  the kernel's per-block maxima is not in the kernel (`:27-30`) and is a host seam (`lm_head_argmax`)
  in the first run: the token is exact by the certificate but produced by host math over device
  candidates, so it is the hybrid token comparison (3.2b, 4.3) and cannot be a device-chain claim.
* **Coverage blockers for a device-chain claim (Phase 3, not Phase 2 criterion 1).** kernel-dev's
  matrix lists the attention-side ops (`attn_norm`, `attn_rope`, `window_kv_roundtrip`/S1,
  `compressor_cache`, `index_key_path`, `indexer_weights`, `hsi_scoring`,
  `top512_merge`, `expanded_view`) and the tail (`final_norm`, `lm_head_argmax`) as
  unkernelised. They do not block the hybrid criterion-1 run (they run host-side, labelled and
  counted, section 3.2a), but they block any device-chain claim, and the tail blocks the token path
  itself (3.2b).
* **task-8 router (not a first-chain blocker).** The MoE router stays the `moe_router` host seam
  until the task-8 kernel lands; the ledger records it as `phase3_followup` and criterion 1 does
  not require it device. This does not make the FFN stage device (gate/up above).
* **task-10 (not a blocker).** Rotation is call-latency bound (11,764 `copy_dtod` calls for
  376,576 slots; the count is derived from the plan, not measured); the copy-only rate is unmeasured
  (section 5), the chain records whatever the rotation phase measures and leaves batching to task-10.
  Criterion 1 has no perf bar.
* **Truth edges that must not drift:** D24 (format 2 for 38 layers, format 1 for layers 0-1), D25
  (fixture is the only truth), D7 (logits and boundary compare only position 129; the decode **tokens**
  130..=137 are compared exactly), 43-row rule (no decode capture).

## 7. Milestones and the iteration-2 mapping

**M1 - iteration 2, first milestone: `DeviceEngine` behind a capability flag with a fake device
session.** Narrow by design and needs no node and none of the !63 chain APIs: the crate exposes
`DeviceEngine<Session: DeviceSession>`, a `FakeSession` that records calls and produces no device
result, and a `--engine reference|device` selection defaulting to `reference`. The engine implements
`InferenceEngine`, reports `vision: false` on main (and the same conservative answer as the
post-!68 `multimodal: false` wire constant, 1.3), refuses non-greedy configurations, and delegates
`generate` to a reference `Arc<dyn InferenceEngine>` supplied by the cli wiring (1.3), because
`ReferenceDriver` is bin-only. Tests run on a CPU-only build. It adopts the typed capability
vocabulary of !68 when that merges (1.3). This is the iteration-2 scope: the serving seam's
capability boundary becomes real without a device claim.

**M2 - re-scoped (2026-10-18): real session and one pinned kernel execution proof.** The original
iteration-2 mapping made M2 the chain skeleton; after M1 the Lead re-scoped it so the device path
is proved incrementally. M2 opens the CUDA driver, loads the pinned `moe_router_sm121` cubin
(37,384 B, sha256 `5992cef9...769b`, checked before the driver) and runs exactly one real router
launch through the same `DeviceSession` seam on a worker thread, reading ids/weights/scores back.
The report then derives `device_execution: true` from the session's evidence, naming the kernel,
cubin sha256/size, launches, tokens and an outputs sha256; `chain_execution` stays `false`. This
is a session/launch proof, not a forward chain: **no token, logits, correctness or criterion-1
claim, and no chain**. It is gated by the `cuda` feature (default off) plus an explicit
`--engine device --device-cubin PATH`, so the default binary and a GPU-less CI host build and test
without a device path; the launch itself is a Lead-run dev-window result, not a test. The wiring
and the fake session stay behind the M1 capability flag.

**M3 - chain skeleton on the device.** The scope formerly called M2. Split into M3-a (first),
M3-b/c/d (serving-readiness 3.3 ordering). **M3-a is the host-side skeleton, the first M3 MR
(this branch):** `chain.rs` derives the 130 + 8 CED schedule (5,162 prefill + 320 decode =
5,482 token-layer executions, cross-checked against `SequencePlan::ced`), fixes the 40-layer
stage map (each line cites its design 2.2 item; device kernels named, gate/up and the router
labelled merged/measured but not wired, the remaining host seams labelled), builds the planned seam
set and the plan-vs-ledger self-check
(fails on missing **and** unplanned seams), and runs through a `ChainOps` backend with
`chain_execution = false`. `window.rs` accounts the 128-slot rings (67,584 B each, 2,703,360 B
for 40) and round-trips the oldest-first view; `cubins.rs` parses the nine baseline pins from
`kernel_baseline.txt`, requires eight of them (the six chain kernels plus router and routed
gate/up) and verifies size and SHA-256 fail-closed; the ninth, `swiglu_mxfp8_quant_sm121`, is a
host seam and not a required chain kernel. The CLI arm
`sparkflash device-chain --plan-only` prints the plan, stage map, seam counts, the plan-only label
verbatim and the three design 4.3 run labels explicitly prefixed "label reserved for a later
stage:", then exits 2 refusing to serve. M3-b/c/d then add the typed builders/catalog, weight
staging and memory admission, and the first host-seam hybrid pass. No token comparison is claimed
in M3-a; the ledger is already required, and no device claim is made until a Lead window.

**M4 - full 130 + 8 run and the criterion-1 claim.** The scope formerly called M3. The full phase
of section 5 with the boundary capture at 129, the contract of section 4, and the report artifacts.
The first point at which this note's subject may be called passing - as a hybrid claim, with
labelled host seams, the token path labelled "hybrid: token produced by host math over device
logits", the FFN stage labelled hybrid while gate/up is merged but not wired, and the head
labelled NOT YET DEVICE-EXECUTED AS A PRUNING PATH.

File layout: `crates/sparkflash-device/src/{lib.rs, engine.rs, session.rs, chain.rs, window.rs,
cubins.rs}`, `crates/sparkflash-device/tests/{engine_offline.rs, chain_contract.rs}`, the
`sparkflash-cli device-chain` subcommand, and the `--package=` selector in
`scripts/build-aarch64.sh` (still to add; the selector is not needed until a node build of the
chain exists). The crate must build without libcuda (same rule as kernels: nothing links libcuda;
the driver is loaded through `Cuda::load`).

## 8. Open questions to the Lead

1. **M1 placement.** M1 is library code, not docs; the Lead asked for the design branch from main.
   Should M1 land on this branch after the doc is approved, or as its own MR after !68 (whose typed
   capability vocabulary it adopts)?
2. **M1 flag surface.** Is `--engine reference|device` on `serve` with default `reference` the right
   shape for M1, or should M1 stay library-only until M2?
3. **Capture evidence.** Manifest + sha256 copied back, blob left on the node with the reproduce
   command and its sha256 in the report: acceptable as review evidence?
4. **Capture vs chain build.** I plan to produce the capture on the node with the same binary and
   snapshot as the chain (a second 613 s-class reference run); the dev-host copy remains fixture
   evidence only. That makes build/provenance the single intended variable; it does **not** make the
   numeric comparison single-variable (CPU f32 vs device bf16/FP4, section 5). Confirm this is the intended
   truth path.
5. **task-10.** Confirm rotation batching is out of scope for criterion 1 and the 11,764-call count
   (derived from the plan, not measured) is recorded, not optimised, in the chain report.
6. **Head and FFN contract.** Confirm the 4.2 head contract (the token is exact by the certificate
   but host-reduced in the first run, i.e. a hybrid token; logits PASS only over exact entries plus
   the argmax; all-entry comparison only as a labelled host diagnostic) and the 3.2a FFN hybrid
   labelling (host gate/up/SwiGLU/quantiser means no device-chain claim for the FFN stage until the
   merged K = 5120 kernel is wired into the chain), or should criterion 1 be blocked on that wiring?

