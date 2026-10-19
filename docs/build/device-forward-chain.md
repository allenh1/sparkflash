# Device forward chain: the 40-layer hybrid chain and the 130 + 8 golden comparison

Status: **M3-c c1/c2/c3/c3b/c4a-1/c4a-2 on `task/device-engine-m3-c3b`, forked from `origin/main`
`5416ccf`; the c1/c2 host pass is tested without a GPU, and the c2 comparison is bit-exact against
the reference boundary on a finite plan. Window b19 (combined) passed the released-checkpoint
one-layer, one-position device-entry arm on device (exit 0, `M3-c LAYER-0 PASS`, hybrid verdict
PASS within the m11 b7 classes); that is the c2 window record. Window b22 passed the 40-layer c3
hybrid-token arm (exit 0, reference and hybrid token id 5, chain-wiring token parity); that is the
c3 window record. c3b adds a device `mhc_post` per layer and defers the FFN epilogue into the next
device entry; window b29 passed the 40-layer c3b arm (exit 0, reference and hybrid token id 5,
chain-wiring token parity). c4a-1 replaces the five host f32 attention projections with the device
B12x MXFP8 GEMMs; window b31 passed the 40-layer c4a-1 arm (exit 0, reference and hybrid token id 5,
chain-wiring token parity). c4a-2 puts the format-2 `flash_mla_csa2_sm121` attention body on the
device at position 0; window b33 passed the 40-layer c4a-2 arm (exit 0, reference and hybrid token id
5, chain-wiring token parity) after the b32 failure record. `MhcEntry`, `MhcPost`,
`wq_a`/`wq_b`/`wkv`, the format-2 body and `wo_a`/`wo_b` are device-executed per layer; the 4 MoE
stages remain host stand-ins, and no token result is a device-token result; Phase 2 exit criterion 1
remains OPEN**
(`phase2-sequence-harness.md` section 7; D25). The chain skeleton
(`chain.rs`, `window.rs`, `cubins.rs`, the masked comparator and `device-chain --plan-only`) is
code, not design prose, and the wiring (`chain_device.rs`) drives its full plan through one
`ChainOps` implementation: host-seam stages go to a host-seam delegate, every device-status stage
goes to an injected launcher, and the wiring runs all 5,482 token-layer executions (108,017
ChainOps calls, of which the fake launcher records 49,347) through `device-chain --dry-run` while
the plan-vs-ledger self-check validates. M3-b now has a real first-stage data path: layer-0 mHC
tensors and declared embed rows are read through `WeightSource`, admitted against the 1 GiB
MemAvailable floor before any device allocation, copied into the sixteen mHC buffers, and passed
to the pinned mHC launch through `ChainDevice` by a cuda-only `--run-stage mhc_entry` arm. That
arm ran on a device in Lead window b9 (three runs, 1/3/130 positions, exit 0); it is a stage path,
not a chain, and it read nothing back. The mHC stage numbers recorded on main by the m11 harness
are **m11's isolated layer-0 measurements, not a chain run**. The default-build `NoCudaLauncher`
still refuses every device stage with `CudaUnavailable`; the plan and dry-run open no driver and
launch nothing. Every number below is plan accounting, a labelled measurement on another harness,
or the b9 stage-run receipt; none is a chain measurement. The document still consumes the
main-branch comparison helpers (1.2) and adopts !68's typed capability vocabulary in M1 only (1.3).

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
`crates/sparkflash-kernels/baseline/kernel_baseline.txt` has **ten** entries: the six m4 kernels,
`swiglu_mxfp8_quant_sm121` (host seam, not launched by the chain), its weighted routed variant
`swiglu_mxfp8_quant_weighted_sm121` (D33; m17a stage 3 loads it, window b33 measured 35 checks
at both token counts, not wired),
`moe_router_sm121` (M2 one-launch proof) and `routed_gate_up_sm121` (task-8, measured in
isolation, not wired). The
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
(`kernels/cert_lm_head_sm121.cu:31`, stale: one stage-5 launch per rank and the m16 window b25
stage PASS are recorded below; `roadmap-to-serving.md:440`); it writes an upper bound for entries
it certifies and exact values only for uncertified entries (`src/lm_head_ref.rs:109-151`),
and its per-block maxima are finalised outside the kernel. The recorded device executions of this
kernel are the layer-0 stage-5 harness (`phase1-boundary-table.md:546-559`) - one launch per rank
over the four 32,320-row shards of `head.weight`, all 129,280 entries exact and **0 certified**,
its argmax equal to that harness's bf16-matched host reference, a single hidden state and not a
chain run - and the m16 window b25 stage measurement (`phase3-lm-head.md`), which recorded the
stage PASS on the re-pinned artifact: real-token control and pruning case both PASS, 4,899
certified, `certified_count == flags` per rank. The certificate path (pruning) is therefore
measured at stage level, not in the chain. The head is labelled **STAGE PASS AT m16 b25 (4,899
certified, counts agree), CHAIN HEAD STILL HOST STAND-IN; criterion 1 OPEN**; the first chain
report says "first chain execution, unverified", and its contract is 4.2/4.3, not "already
device".

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
   printed and equal to the ten parsed entries of `baseline/kernel_baseline.txt`; the catalog
   requires eight of them (the six chain kernels plus the router and routed gate/up, 2.2), and the
   other two - `swiglu_mxfp8_quant_sm121` (the host seam's unweighted epilogue) and its weighted
   routed variant `swiglu_mxfp8_quant_weighted_sm121` (D33, measured in m17a) - are not chain
   kernels. Device kernel launches > 0 (a host-only run cannot pass);
2. tokens exactly 491, 367, 201, 80, 3302, 201, 54, 201, 4341 (`compare_steps`, `:699`);
3. head reduction: the reduced argmax over the screen's per-block maxima (host `lm_head_argmax` in
   the first run; lowest-index tie-break `lm_head_ref::best_entry:161`) equals the expected token,
   and the report carries the certified/exact mask (the exactness of max/argmax is the
   certificate's, not a per-entry comparison). This is the **hybrid token comparison**: the report
   says "hybrid: token produced by host math over device logits", and `CERT_LM_HEAD` is labelled
   **STAGE PASS AT m16 b25 (4,899 certified, counts agree), CHAIN HEAD STILL HOST STAND-IN;
   criterion 1 OPEN**; the first chain report says "first chain execution, unverified";
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
  launch per rank is recorded (3.2c), and the m16 window b25 stage measurement recorded the pruning
  path PASS (4,899 certified, `certified_count == flags` per rank; `phase3-lm-head.md`). It is a
  screening kernel (4.2), so the head is labelled **STAGE PASS AT m16 b25 (4,899 certified, counts
  agree), CHAIN HEAD STILL HOST STAND-IN; criterion 1 OPEN**; the first chain report says "first
  chain execution, unverified", and the all-entry logits comparison is impossible by design. The
  block reduction over the kernel's per-block maxima is not in the kernel (`:27-30`) and is a host
  seam (`lm_head_argmax`)
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
for 40) and round-trips the oldest-first view; `cubins.rs` parses the ten baseline pins from
`kernel_baseline.txt`, requires eight of them (the six chain kernels plus router and routed
gate/up) and verifies size and SHA-256 fail-closed; the other two, `swiglu_mxfp8_quant_sm121` (the
host seam's unweighted epilogue) and its weighted routed variant `swiglu_mxfp8_quant_weighted_sm121`
(D33), are not required chain kernels. The CLI arm
`sparkflash device-chain --plan-only` prints the plan, stage map, seam counts, the plan-only label
verbatim and the three design 4.3 run labels explicitly prefixed "label reserved for a later
stage:", then exits 2 refusing to serve. M3-b adds layer-0 weight staging and budget admission;
M3-c/d add the remaining typed builders/catalog, the first host-seam hybrid pass and the boundary
comparison. No token comparison is claimed in M3-a; the ledger is already required, and no device
claim is made until a Lead window.

**M3 stage wiring (merged on main as `2a1f80b`, on top of `5898e23`).** `chain_device.rs` is the single `ChainOps`
implementation the device run will use. Every stage is classified, never guessed: a
`StageStatus::HostSeam` stage goes to an injected `HostSeamDelegate` (the M3-a seat recorder by
default), and a `DeviceKernel` or `DeviceMeasuredNotWired` stage goes to an injected
`DeviceLauncher`; there is no host fallback for a device stage and no device launch for a host
seam. The default tests and `device-chain --dry-run` use the fake launcher, which records the
49,347 device-launcher calls and, with `SeatHostSeams`, produces the 58,670 host-seam records;
the full plan is 108,017 `ChainOps` calls and the plan-vs-ledger self-check passes. The dry-run
prints the per-stage call counts and host-seam seats, still exits 2 and still prints
`chain_execution=false`, `device_execution=false`, `token_produced=false`. Injecting a missing
cubin or a launcher error fails at the named stage/layer/position: there is no partial success
report.

The real `CudaLauncher` is behind the `cuda` feature. It reads each pinned cubin once and verifies
that same slice with `cubins::verify_bytes` on construction, and has the first typed builder wired
(`mega_mhc_launch::MegaMhcParams`). M3-b now supplies `MhcBuffers` for layer 0 through
`stage_layer_zero`; without that staging, mHC entry/post still refuse `NotStaged`. Router and
routed gate/up are measured but not wired (`NotWired`); the other device stages are `NotStaged`.
The default-build `NoCudaLauncher` refuses every device stage with `CudaUnavailable`. The wiring
MR made **no device execution, no token and no comparison**; it proves the dispatch, launcher
injection, error path and ledger plumbing only.

**M3-b - layer-0 weight staging and budget admission (this branch, on `972c19a`).** `staging.rs`
reads `layers.0.hc_attn_fn` (F32 [24, 4*d], permuted into kernel B/A/C row order without
averaging), `layers.0.hc_attn_base` (F32 [24], rearranged with `bias_into_kernel_rows`),
`layers.0.hc_attn_scale` (F32 [3]), `layers.0.attn_norm.weight` (F32 or BF16 [5120]) and the
declared `embed.weight` rows through `WeightSource`. Every spec is checked before its read; the
read must match the spec's storage length. The stage report labels two SHA-256 sets: a **source**
digest over the checkpoint bytes read, and a **buffer** digest over the exact host vector later
passed to `copy_htod` (the permuted predictor, kernel-order base, f32-widened norm and four-copy
X row), so a buffer digest is never mistaken for source provenance. Embed rows are fetched only
with `read_ranges`: a source that cannot page a row is refused rather than triggering a
whole-table read; `embed.weight` is 1.324 GB BF16. This is **not** an all-resident model and has
no streaming scheduler; it stages layer 0 only. The
layer-0 boundary seed is host-built and declared exactly as m11 declares it (B = identity, C = 0,
A = [1, 0, 0, 0], Y_-1 = 0), not checkpoint data.

`budget.rs` admits a stage only when `MemAvailable - (projected host + projected device)` remains
at least 1 GiB; it reports every number in a refusal and in an admission receipt. This is a
headroom rule, not the harnesses' older absolute 32 GiB gate. `CudaLauncher::stage_layer_zero`
admits first, then requires the caller's projected device bytes to equal
`MhcBufferPlan::LAYER_ZERO` exactly and its projected host bytes to cover the staging plus the
verified cubins; only then does it load the driver, allocate the exact sixteen buffers
(`MhcBufferPlan::LAYER_ZERO`, 2,064,352 B) and copy the staged tensors. A driver or copy failure
frees the allocations and leaves `mhc_buffers` unset. The staging path zeroes `hc_stats` and
`hc_sync` at stage time, and every mHC entry launch re-zeroes both before the launch, matching
m11's per-launch rendezvous reset; without it position 2+ would start with stale tile counters.
`--run-stage mhc_entry` is compiled only with `cuda`; it stages the weights, uses the admitted
numbers, writes each declared position's row into X, and drives one layer-0 mHC entry through
`ChainDevice`. `MhcBuffers` fields are private and the public buffer setter is gone, so only
`stage_layer_zero`'s own allocation set can reach a launch; `MhcPost` and any non-zero layer are
refused `NotStaged` until the coefficient handoff exists. The CLI refuses every other stage by
name (router/gate-up `NotWired`, the rest `NotStaged`). Tests cover staging arithmetic/checks,
buffer digests pinned to the exact host bytes, admission and projection boundaries, buffer layout,
a launch-level test of the sized rendezvous reset through the copy seam before the kernel, and the
refusal-before-driver path without a GPU.

**Window record: b9 (2026-10-18T23:21:21Z).** Lead window b9 (node-only logs:
`window-b9-20261018T232121Z.log`; `runs9/m3b-pos0.log`, `runs9/m3b-pos012.log` and
`runs9/m3b-pos0-129.log`, all at 23:22:35Z-23:22:36Z) ran the cuda-only arm three times on binary
`sparkflash_m3b_1846e72_aarch64` (42,221,656 B, sha256 `1846e729...`; built from `6f79bcc`). The
mHC cubin was `mega_mhc_sm121`, 70,312 B, sha256 `e21293e3...`, verified before any driver or
allocation. All three runs exited 0: positions `0` (1 device stage call), `0,1,2` (3) and
`0..129` (130), each `ADMITTED` before allocation and each
`ChainDevice<CudaLauncher, SeatHostSeams>` with `stage_execution=true`,
`chain_execution=false` and `token_produced=false`. The admission receipts were:

- positions 0: MemAvailable 126,367,248,384 B; projected host 2,067,316 B + device 2,064,352 B =
  4,131,668 B; floor 1,073,741,824 B; remaining 126,363,116,716 B.
- positions 0,1,2: host 2,087,796 B + device 2,064,352 B = 4,152,148 B; remaining
  126,363,002,028 B.
- positions 0..129: host 3,388,276 B + device 2,064,352 B = 5,452,628 B; remaining
  126,341,438,636 B.

The binary was built from `6f79bcc`; the later EmbedRow privacy hardening is not CLI-visible,
so no new binary was built for this record. Staging printed source -> buffer SHA-256 pairs:
predictor 491,520 f32 (`3d384347...` ->
`cbbb089e...`), hc_attn_base 24 f32 (`616e03e5...` -> `deda659f...`), hc_attn_scale 12 B
(`278ed053...`, unchanged) and attn_norm 5,120 f32 from a 10,240 BF16 source (`8da5f030...` ->
`7de8fafc...`, 20,480 buffer bytes); the declared embed rows (3 in the 0,1,2 run and 130 in the
0..129 run) each printed source -> buffer X digests. There was no readback, so **no coefficient
was compared and no per-position output is claimed**.

The honest reading of b9 is the reset evidence, not a numeric result: 130 sequential launches
completing is evidence that the per-launch `hc_sync`/`hc_stats` reset lets each position
rendezvous again (m11's finding was that stale counters never publish), but there is no readback,
so the coefficients are not compared and no per-position output is claimed.

**M3-c c1/c2 one-layer host pass (this branch, rebased onto `159ae8c`; originally cut from
`dfa790b`).** The first MR of M3-c lands the layer-0 arithmetic and its comparison without a GPU:

- **c1, the host pass.** `sparkflash-engine/src/forward/layer_zero.rs` composes the existing
  reference functions (`hc_mixes`/`hc_pre`/`hc_post`, `attention_forward` and `moe_forward`) in
  the reference block's layer-0 order, records one `StageEvent` per active
  `LAYER_ZERO_HOST_STAGES` entry (17 stages), and returns the boundary values.
  `sparkflash-device/src/layer_zero.rs` maps those events onto the chain's stable seam names, builds
  the one-layer `ChainSeamPlan`, records the host ledger, validates it (a missing or unplanned seam
  fails) and prints the execution classification. The device-stage entries are **not** padded with
  fake bytes: they are labelled `host stand-in (device builder not wired)`, and the report says they
  did not execute on a device. This is the declared design decision for this MR: the only device
  stage in the one-layer path is the `mhc_entry` readback below; every other active stage runs the
  reference host arithmetic and says so. In the window hybrid, `attn_norm` is
  `device-fused` rather than host: its result arrives inside the device mHC `xhat_normed` capture,
  so the pass drops that planned seam instead of emitting a zero-byte host seam. `mhc_entry` can be
  marked device only through a private one-shot launch receipt held by
  `run_layer_zero_device_entry`; no host-only run can relabel a stand-in. A `Bar` comparison is a
  failure when it is outside its declared class (see the hybrid bullet below), not a success.
- **c2, the host reference comparison.** The same `LayerZeroTrace` is compared against the
  reference engine's own layer-0 `BoundaryTable` row (`prefill_observed`), quantity by quantity,
  with bit identity as the class. The quantity set is pinned to twelve ordered names (six tensors
  and six mHC mix vectors), and both comparators refuse a dropped quantity. Each quantity now
  carries its first differing index and expected/actual f32 bits, the maximum absolute delta and
  the maximum ULP distance; the report prints those per quantity before the verdict. The unit test
  uses a finite plan-shaped source, reaches `ComparisonClass::Exact` for all twelve, mutates one
  residual value to prove `Bar`, and mutates one ULP to prove the diagnostic fields. This unit-test
  c2 is a finite-plan comparison, not a token; the released-checkpoint window c2 record is b19 in
  the window records below, still one layer/one position and not a token.
- **hybrid class.** The device-entry arm is not a host-vs-reference comparison. The one device
  kernel's outputs are compared with the classes m11 measured in window b7
  (`phase3-mhc-entry.md`), and the host twin uses the kernel's FMA program (kernel-dev's m15
  cross-check: the pinned kernel has 241 `fma.rn.f32`, no `mul.rn.f32`/`add.rn.f32`, and the
  `C Y` term is the single plain `mul.f32`). The step-1 update is
  `kernel_mhc_update` (`acc = C*Y`, then one `mul_add` per B term in stream order), step 2 is
  `kernel_mhc_xhat` (one `mul_add` per A term), and the capture is
  `kernel_mhc_capture` (`xhat * rms_weight * inverse_rms`, left-associative). Unit tests pin
  that these differ from a mul-then-add construction. With the released layer-0 seed
  (X_-1 = four copies of the embed row, Y_-1 = 0, B = I, C = 0, A = [1,0,0,0]) the kernel writes
  X_0 = X_-1 and publishes H(X_0). X_l, the f32 capture, the E4M3 codes and the E8M0 scales are
  **exact** (m11 b7: X_l 20,480/20,480, capture 5,120/5,120, codes/scales exact); the `1/rms`
  scalar carries the `3e-4` bar (b7 maximum `2.746582e-4`), and the 24 B/A/C coefficients carry
  the `1e-4` bar (b7 measured B `1.1920929e-7`, A `2.682209e-6`, C `6.7055225e-7` at position
  129). `compare_entry` gates those five readback quantities (capture, three coefficient groups,
  `1/rms`) and the arm separately requires exact X_l/codes/scales; it does not invent a bar for
  downstream host arithmetic. The twelve
  trace quantities are still printed with their full diagnostics, labelled as host-continuation
  diagnostics: the host trace starts from the host-computed entry, the device trace starts from
  the readback, so every downstream value inherits the entry deltas and is not a device-kernel
  result. The `mega_mhc_layer_with_scale` traffic-accounting return value is **not** used as the
  entry expectation: it applies the coefficients it just formed, while the kernel writes with the
  previous call's B/C.
- **window-only continuation.** `CudaLauncher::read_layer_zero_entry` reads the mHC kernel's f32
  `xhat_normed` and `hc_coeffs` (24 values), transposes B into the host source-major comb order and
  builds `LayerZeroEntry`. `run_layer_zero_device_entry` writes the staged row and reads X back
  **before** the launch (the kernel updates X in place to X_l), launches the pinned
  `mega_mhc_sm121` kernel, reads X_l, the E4M3 codes and the E8M0 scales back, reads the entry and
  the kernel's `hc_rms_out` (1/rms), and continues the host pass from the readback. The arm prints
  the staged boundary seed, the X/X_l/capture/codes/scales comparisons, the device/host `1/rms`
  delta against the `3e-4` bar, the non-zero coefficient count, and the fact that
  `xhat_normed`/`hc_coeffs`/`hc_rms_out` are unconditional writes at
  `kernels/mega_mhc_sm121.cu:291-294,329` while `trace_ring=0` only disables the ring.
  `sparkflash device-chain --run-stage layer_zero --checkpoint DIR --cubin-dir DIR --token 0` is
  the cuda-only arm. It resolves the same checkpoint-aware element cap as `serve`:
  `CHECKPOINT_TENSOR_LIMIT` by default, with `--element-limit N` as the escape hatch, because the
  released `wq_b` alone is 41,943,040 elements and the reference guard would refuse it before any
  device work.
- **window records.** b13 (binary `sparkflash_m3c_8819_aarch64`, sha256 `881918dc...`) failed
  before any allocation in the host reference trace because the arm hard-coded the 32 Mi
  `REFERENCE_TENSOR_LIMIT`; released `attn.wq_b` is 41,943,040 elements. The element-cap delta
  (pre-rebase commit `56a4441`) resolves the same checkpoint-aware default as `serve` and adds
  `--element-limit`; no device work happened in b13, so no window record is claimed.
  b14 (binary sha256
  `497058ed3ff93e107b3709ad6861e3dd24a4b9810f33dc9c9ff4b6bca3d11925`, built from `4722153`)
  ran the same arm in a Lead window: it reached staging and the launch, exited 1 with
  `sparkflash device-chain: layer-0 comparison reported BAR (12 differing); a readback that is not
  bit-identical to the host trace is a failed run`, and printed no per-quantity lines. There was
  no device crash; vLLM was stopped and restored. The log is node-only
  (`window-b14-20261019T012513Z.log`, `runs14/m3c-layer0-token0.log`). b14 is a failed run with
  no c1/c2 record; this delta is its response, and b19 (combined) is the recorded rerun (b16 was
  m15's re-window).
  The old exact comparator started the host trace from the host RMSNorm/capture
  (`value * inv * weight`) while the kernel emits `value * weight * inv`, and the host and
  kernel reduce the `1/rms` sum in different orders; if the entry readback is within the m11
  classes, twelve downstream host values differing is the expected consequence, not evidence of a
  row or coefficient-mapping bug. The new arm gates the entry boundaries and prints the
  per-quantity detail instead of reporting only the count.
  b19 (combined) is the c2 window record. The device ran binary
  `sparkflash_m3c_d9ac_aarch64` (sha256
  `d9ac5a616c289b7089c73b18c84779e69eef1f6f135a80a0d177112228de0d78`, built from commit `b49992f`)
  with cubin `e21293e3...`, exited 0, and printed `M3-c LAYER-0 PASS - one layer, one position, no
  token` and `layer-0 hybrid verdict: PASS - the device MhcEntry readback is within the m11 b7
  classes (hc_coeffs 1.0e-4, 1/rms 3.0e-4); downstream differences are host arithmetic from a
  different entry, not a device claim`. The record is `window-b19-20261019T021326Z.log` and
  `runs19/m3c-layer0-token0.log` on the node; vLLM was stopped and restored. The recorded evidence:
  X readback equals the admitted four-copy BF16 row (true); X_l readback equals the FMA host update
  (true); the FMA host update equals the host trace's four-copy X_0 (true); f32 capture exact
  5,120/5,120; E4M3 codes exact 5,120/5,120 bytes; E8M0 scales exact 160/160 bytes; 24/24 non-zero
  coefficients, source `Device`; staged seed B identity, C zero, A `[1,0,0,0]`, Y_-1 zero; device
  `1/rms` 6.406901360e0 against host 6.406919003e0, max abs delta 1.764297e-5 (bar 3.0e-4). The
  per-quantity classes were `pre_attention_hidden` exact (max abs 0), `hc_attn_pre` bar max abs
  8.493662e-7 / 127 ULP, `hc_attn_post` bar 3.278255e-7 / 55 ULP, `hc_attn_comb` bar 3.129244e-7 /
  238 ULP, and `inv_rms` bar 1.764297e-5 / 37 ULP. BAR is used only where a declared m11 bar
  exists; X/X_l/capture/codes/scales are exact. The eight device-status stages are still host
  stand-ins: the arm accounted 8 host seams and 887,344 B of host work, explicitly not device
  work. This is one layer, one position and no token; c3 is the b22 record below.


### c3: the 40-layer hybrid token (implemented; b22 record below)

The c3 arm is `sparkflash device-chain --run-stage hybrid_token --checkpoint DIR --cubin-dir DIR
--token 0`. It runs the in-process reference prefill first on the same checkpoint and prints the
reference token id and top-5, then runs the 40-layer hybrid chain and prints its token id and
top-5. PASS is exactly token-id equality between those two in-process results; the top-k overlap
and per-value deltas are reported but are not gating. The final comparison is one end-to-end run.

Exactly one stage is device-executed per layer: `MhcEntry`. Every layer is called with the step-1
identity entry `B=I, C=0, Y=0, A=previous FFN pre_mix`; the kernel's previous-layer B/C/Y mixer is
NOT exercised because `mhc_post` and `ffn_epilogue` are host in this stage map. The eight host
stand-ins remain `attention_projections`, `attention`, `attention_output_projection`, `mhc_post`,
`moe_router`, `moe_gate_up`, `moe_down_combine` and `shared_expert_gate`; `lm_head_screen` is the
tail stand-in and `final_norm`/`lm_head_argmax` are tail host seams.

There is a declared BF16 residual staging seam every layer: the host writes the f32 residual as
BF16, the kernel identity-writes it, and the host continues from the device-readback X. The chain
is therefore not bit-identical to the reference even when the host arithmetic is exact. The token
claim is **chain-wiring parity**, not a device-token result; Phase 2 exit criterion 1 remains OPEN.

The report prints per-layer staging/device/host milliseconds and host bytes, the chain totals (40
device launches, device/host milliseconds, host bytes), and progress every 8 layers. The whole
command self-limits: if the projected total exceeds 600 s it aborts cleanly with a log line rather
than waiting for the window timeout. `--plan-only` and `--dry-run` still launch nothing and produce
no token.

### c3b: device post plus the deferred FFN epilogue (implemented; b29 record below)

The c3b arm keeps the c3 command shape, `sparkflash device-chain --run-stage hybrid_token
--checkpoint DIR --cubin-dir DIR --token 0`, and the same in-process reference. It adds one device
`MhcPost` launch after each layer's host attention and moves the FFN epilogue add out of the host
into the next layer's device `MhcEntry` step 1.

Per layer the device does two launches:

1. `MhcEntry`. Layer 0 keeps the identity entry `B=I, C=0, Y=0, A=initial pre_mix`; it is the only
   primary identity entry (the two Engram re-entries below are counted in the 42). Layers 1..39 write
   the real previous-layer FFN `B` (comb), `C` (post), `A`
   (pre) and `Y` (FFN block output) into the staged buffers and launch step 1 as the deferred FFN
   epilogue, followed by step 2 over `A_l` and layer l's own attention `hc_attn` predictor.
2. `MhcPost`. After host attention, the kernel mixes the attention output back onto `H_l` with
   layer l's attention `post`/`comb` coefficients and the previous attention `pre` row, writes
   the post-attention residual `A_l` to X, and publishes the 24 FFN coefficients under layer l's
   `hc_ffn` predictor plus the normalized FFN input capture. The host FFN consumes those device
   values; the host does not recompute the merge or the FFN coefficient set.

The device-status host stand-ins therefore drop from 8 to 7: `attention_projections`,
`attention`, `attention_output_projection`, `moe_router`, `moe_gate_up`, `moe_down_combine`
and `shared_expert_gate`. `mhc_post` is device. The FFN epilogue split is 39 deferred additions
(layers 0..38, executed inside the next `MhcEntry`) and 1 host addition (layer 39, no next entry).
Count it exactly: 40 device `MhcPost` calls and 42 device `MhcEntry` calls per token, 82 launches.
The two extra entries are the Engram split at layers 1 and 14: Engram adds to `H_l` between the
deferred step 1 and the attention entry, and one fused entry call cannot host-insert that addition,
so each of those layers runs the real B/C/Y mixer device, applies Engram to the host readback, and
runs a second identity entry. Engram remains a declared host seam; it is not device-executed.

The host/device crossings that remain: each entry writes `X` as BF16 and reads
back the BF16 `X_l`; each post sends the attention output `Y` as BF16 and reads back BF16 `A_l`;
each launch reads the 24 coefficients, the capture, `1/rms`, the FP8 codes and the E8M0 scales.
The B/C/A rows for layers 1..39 are consumed from the previous device post's readback on the host
and re-uploaded for the next entry; that is the declared coefficient round-trip, not a
device-to-device buffer alias. The chain is still not bit-identical to the f32 reference because of
the declared BF16 residual staging seam.

The report prints per layer `entry_us`, `post_us`, `device_us` (entry+post) and a chain total
in microseconds plus `device_ms` as a fractional millisecond value, so the c3 integer-millisecond
truncation cannot recur. `staging_ms` covers the tensor staging, including the post staging;
the device interval starts at the entry launch and covers the entry X write, the launch and the
readbacks, and at the post launch and covers the post launch and readbacks. At the two Engram
layers, `entry_us` is the sum of the two device intervals (mixer and identity entry) and excludes
the host Engram work; the layer line carries `engram_split=true`. Per-layer ledger lines mark
`mhc_entry` as the layer-0 identity seam or the deferred epilogue+entry, `mhc_post` as device,
and `ffn_epilogue` as deferred or host, so the split is visible in the log rather than only in
prose.

The engine host-control test (`tests/hybrid_chain.rs`) proves the deferred order bit-for-bit
against `prefill` on a finite plan, including that layer 0's post is device in the wiring and layer
39 keeps its host FFN epilogue; a second test perturbs a layer's post `ffn_mixes.pre` coefficient
and requires the chain output to change. The device arm adds its own readback comparisons: the
entry step-1 `X_l`, the post step-2 `A_l`, both captures, both coefficient sets, both `1/rms`
values, and the FP8 codes/scales must stay inside the m11 b7 classes before the chain proceeds.
The claim is unchanged: **CHAIN-WIRING parity, not a device-token result**, and Phase 2 exit
criterion 1 remains OPEN.

### c3b window b29: PASS - CHAIN-WIRING token parity with the deferred-epilogue device path

Window b29 ran the c3b arm on binary `sparkflash_m3c_c3b_dc1b451_aarch64` (sha256
`e7959c4595b6d7fba063faa0a1bf7bb0fb9894401fb2df2b678e241a11e13680`, built from commit
`dc1b451`), exit 0 at 2026-10-19T05:13:49Z. Logs: `window-b29-20261019T051140Z.log` and
`runs29/c3b-hybrid-token0.log` on the node; vLLM was restored by the window trap. The report
printed the identity device entry path, the BF16 residual staging seam, the Engram seam, the
no-cache seam, the 7 host stand-ins and the c3b follow-up before the result lines. **Log caveat:** the
b29 log predates the Batch-18 correction and still prints the retracted "device-to-device" sentence;
the Batch-18 note below is authoritative and the log is kept verbatim as evidence, not as a current
claim.

The verdict was verbatim: `CHAIN-WIRING token parity: PASS - the hybrid chain produced the same
token id as the in-process reference engine; this is chain-wiring parity, not a device-token claim,
and criterion 1 remains OPEN`. The reference half ran first on the same checkpoint:
`reference engine: token_id=5; wall_ms=31796; top5=5: 13.230665207, 201: 12.492528915, 372:
11.897726059, 30: 11.392500877, 1897: 11.015835762`. The hybrid chain printed
`hybrid chain: token_id=5; wall_ms=23973; top5=5: 13.245909691, 201: 12.495220184, 372:
11.903433800, 30: 11.388140678, 1897: 11.013148308`. The comparison was
`token comparison: reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=5`. This is
**chain-wiring parity, not a device-token result**, and Phase 2 exit criterion 1 remains **OPEN**.

Chain totals, verbatim: `layers=40 entry_launches=42 post_launches=40 device_launches=82
engram_split_layers=2 device_us=19656 device_ms=19.656 staging_ms=1050 host_ms=21194
total_chain_ms=23973 host_bytes=61658960 staged_matrix_bytes=237580480 cubin_bytes=70312
device_buffer_bytes=2064352`.

**What the numbers mean.** The 42 entries are 40 layer entries plus the two extra identity entries
at the Engram layers 1 and 14; the 40 posts are one per layer. The 82 device launches are
accounted by `device_launches`, and each one is also in the per-layer ledger and the per-layer
`entry_us`/`post_us` evidence. `device_us=19656` is the sum of the per-launch host wall-clock
intervals around the launches and their readbacks (entry X write, launch, five read calls; post
launch and readbacks), printed in microseconds and as the fractional `device_ms=19.656`. It is not
a CUDA-event GPU duration, but it is not truncated to integer milliseconds either; the c3
`device_ms=0` artefact cannot recur. At the two Engram layers `entry_us` is the sum of the mixer
and identity-entry device intervals and excludes the host Engram work; the per-layer line carries
`engram_split=true`. `staging_ms=1050` is the per-layer tensor-staging total and
`host_ms=21194` is the host-stand-in continuation total; neither is device time.

**Limits stated plainly.** Layer 0 is the only primary identity entry (`B=I, C=0, Y=0, A=initial pre_mix`; the two Engram re-entries of layers 1 and 14 are counted in the 42, not identity-primary entries);
layers 1..39 consume the real previous-layer FFN `B/C/A/Y` from the previous device post. The
`ffn_epilogue` split is 39 deferred additions (layers 0..38) and 1 host addition (layer 39).
`mhc_post` is device on all 40 layers, and the 7 device-status host stand-ins that remain are
`attention_projections`, `attention`, `attention_output_projection`, `moe_router`,
`moe_gate_up`, `moe_down_combine` and `shared_expert_gate`. Engram remains a host seam: layers 1
and 14 run the real deferred B/C/Y mixer on the device, apply Engram on the host readback, then run
a second identity device entry, which is why those two layers have 3 launches each. The 24 B/C/A
coefficients are read back from the device post and re-uploaded for the next entry: a declared
coefficient round-trip, not a device-to-device buffer alias. The residual X/Y crossings remain
BF16 staging, so the chain is not bit-identical to the f32 reference by construction.

**Batch-18 correction.** The b29 report's printed `device entry path` sentence originally said
layers 1..39 consume the previous-layer FFN `B/C/Y` "device-to-device" and that "the host does not
round-trip those coefficients". That was false: the post reads `pre`/`post`/`comb` back to the host
and the next entry re-uploads them (`write_layer_bc`/`write_layer_a`), and `Y` is the host FFN
output, BF16-staged. The sentence now reads `B/C/A read back then re-uploaded: declared coefficient
round-trip; Y is the host FFN output, BF16-staged; the add runs in device step 1`. The b29 measured
numbers and token verdict are unchanged.

**Two pre-window host-twin failures, recorded rather than hidden.** Window b27 failed at the
layer-0 post coefficient bar because the host twin passed `LayerPostStaging::predictor` - the
predictor permuted into the kernel's B/A/C row order for upload - to `hc_mixes`, which consumes
the released pre/post/comb order; the device coefficients were right and the twin was wrong
(`df0412c` retains `released_fn` for the twin and adds a host-only test against the kernel
reference). Window b28 then failed at layer 1 because the Engram mixer twin fed the device's
`X_l` readback back in as if it were `X_{l-1}`; for a non-identity B/C row the update is not a
fixed point. `dc1b451` feeds the staged input bytes to the step-1 twin and records the first
differing element, device value, twin value and input rows in the diagnostic. Both were host-twin
fixes; the device path and the claim label were unchanged.

### c4a-1 window b31: PASS - CHAIN-WIRING token parity with the five device attention projections

Window b31 ran the c4a-1 arm on binary `sparkflash_m3c_c4a1_c40ed50_aarch64` (sha256
`7c4078595638a5d0f0a8734447f68f2fd970ebbd34fc76bc5402525ebdf5ec7e`, built from commit
`c40ed50`), exit 0 at 2026-10-19T05:33:45Z. Logs: `window-b31-20261019T053137Z.log` and
`runs31/c4a1-hybrid-token0.log` on the node; vLLM was restored by the window trap. The report
printed the projection-seam ledger, the declared host seams, the m14 host stand-in, the 5 host
stand-ins and the no-cache seam before the result lines.

The reference half ran first on the same checkpoint: `reference engine: token_id=5; wall_ms=31632;
top5=5: 13.230665207, 201: 12.492528915, 372: 11.897726059, 30: 11.392500877, 1897:
11.015835762`. The hybrid chain printed `hybrid chain: token_id=5; wall_ms=23585; top5=5:
13.245470047, 201: 12.497612000, 372: 11.904157639, 30: 11.388535500, 1897: 11.013092041`. The
verdict was verbatim: `CHAIN-WIRING token parity: PASS - the hybrid chain produced the same
token id as the in-process reference engine; this is chain-wiring parity, not a device-token claim,
and criterion 1 remains OPEN`. The comparison line was `token comparison: reference_id=5 hybrid_id=5
exact_equal=true top5_overlap=5`. This is **chain-wiring parity, not a device-token result**, and
Phase 2 exit criterion 1 remains **OPEN**.

Attention totals, verbatim: `attention totals: launches=200 staging_ms=6326 device_us=536822
staged_matrix_bytes=5069568000 host_bytes=5994560 cubin_bytes=20880 device_buffer_bytes=42083328`.

Chain totals, verbatim: `chain totals: layers=40 entry_launches=42 post_launches=40
attention_launches=200 device_launches=282 engram_split_layers=2 device_us=556036 device_ms=556.036
staging_ms=7033 host_ms=20658 total_chain_ms=23585 host_bytes=67653520
staged_matrix_bytes=5307148480 cubin_bytes=91192 device_buffer_bytes=44147680`.

**What the numbers mean.** The 200 attention launches are exactly the five projection primitives per
layer: 3 launches/layer for `wq_a`+`wq_b`+`wkv` (120) and 2 launches/layer for
`wo_a`+`wo_b` (80). With the 42 MhcEntry and 40 MhcPost launches, the chain's device launches are
282. The per-layer ledger names the three attention stages: `attention_projections` is
`[device: wq_a+wq_b+wkv, 3 launches; host q_norm + D8 quant seams]`; `attention` is
`[HOST STAND-IN: m14 flash-MLA is not wired in c4a-1]`; `attention_output_projection` is
`[device: wo_a+wo_b, 2 launches; host inverse-RoPE + D8 quant seams]`. The m14 attention body is
therefore **not device** in c4a-1, and `wo_a`'s input is exactly the host m14 output, declared rather
than hidden.

The five projection primitives are exact to `mxfp8_ref` on the same packed weight and D8 activation
bytes; the chain comparison is the wiring gate. The engine D8 quantiser was pinned byte-identical to
`m4_quant::act_quant_fp8_row` by `crates/sparkflash-cli/tests/attention_quant.rs` (commit `92cb014`),
so the activation bytes the device harness class consumes are the same bytes this backend produces.

The declared host seams that remain inside the attention stage are q_norm, kv_norm, RoPE, the D8
quantisers (activation operands), the KV cache/window round-trip, the compressor/cache, the indexer
and selected-list assembly, the softmax-adjacent m14 body, and the inverse RoPE. The 5 device-status
host stand-ins after c4a-1 are `attention/m14`, `moe_router`, `moe_gate_up`,
`moe_down_combine` and `shared_expert_gate`.

**Measured cost, stated as a row-1 observation.** `device_us=536822` for the 200 projection
launches is 13.4 ms/layer for five launches (2.68 ms/launch average) at 40 layers, far below the
52 ms/layer whole-matrix budget. That is measured row-1 behaviour on this node, not a general
throughput claim. `staging_ms=7033` is the measured staging total for the 5.07 GB cumulative
matrix bytes; the cold-staging risk did not materialise at this scale, but this is a measured number,
not a residency claim - task-16 is still the place for weight residency. `host_ms=20658` is the
host-stand-in continuation total, not device time, and `total_chain_ms=23585` is the chain wall
including the reference-free chain half.

The residual staging seam, the Engram seam and the layer-0 identity entry are unchanged from c3b;
this window adds device projections only. The binary was built from `c40ed50` plus the quantiser-pin
test `92cb014`; the test does not change the binary.

### c4a-2 window b32: FAILURE RECORD - token parity with only 20 of 40 body launches

Window b32 ran the first c4a-2 binary `sparkflash_m3c_c4a2_ede89a9_aarch64` (sha256
`e9dfe71266defa551ddd216add7ca4b1ad392ef02570d6256217e87f23eedf40`, source `ede89a9`), exit 0
at 2026-10-19T05:47:46Z, log `window-b32-20261019T054537Z.log`, `runs32/c4a2-hybrid-token0.log`;
vLLM was restored. The token comparison passed (`reference_id=5 hybrid_id=5 exact_equal=true
top5_overlap=5`), but the launch accounting did not: `attention totals: projection_launches=200
body_launches=20 ...`, `chain totals: ... attention_body_launches=20 device_launches=302 ...`, and
every one of the 40 `ledger layer NN:` lines still said `attention ... [HOST STAND-IN: m14
flash-MLA is not wired in c4a-1]`. Token parity alone therefore did **not** establish the c4a-2
seam: 20 layers were still on the host sparse loop. This window is kept as a failure record and is
**not** a c4a-2 PASS.

**Root cause.** The released `compress_ratios` are `[0,0, 2x18, 1x20, 0,0,0]`. The 20 ratio-1
layers have `compress_len = end_pos / ratio = 1` at position 0, so `compressed_stride=1` and
`combined=[0,1]`; the first device backend's support guard required `compressed_stride == 0` and
silently returned `false` for those layers, so the engine ran the host `sparse_attention` loop for
exactly those 20. The other 20 (ratio 0 or 2) had no compressed candidate and launched format 2. The
20/302 totals and the stale c4a-1 per-layer label were the only evidence; both are now fixed.

### c4a-2 window b33: PASS - CHAIN-WIRING token parity with the format-2 body on all 40 layers

Window b33 ran the corrected binary `sparkflash_m3c_c4a2_b2_aarch64` (sha256
`73c13ca597eb8d0039091142225ef7b93dc8a23c71aecb5d85d21386d41226de`, source `ee3bdd8`), exit 0
at 2026-10-19T05:59:29Z, log `window-b33-20261019T055723Z.log`, `runs33/c4a2-hybrid-token0.log`; vLLM was
restored by the window trap. The reference half printed `reference engine: token_id=5; wall_ms=31202;
top5=5: 13.230665207, 201: 12.492528915, 372: 11.897726059, 30: 11.392500877, 1897:
11.015835762`; the hybrid chain printed `hybrid chain: token_id=5; wall_ms=22450; top5=5:
13.243947983, 201: 12.495163918, 372: 11.901222229, 30: 11.386974335, 1897: 11.013463020`. The
comparison was `token comparison: reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=5` and
the verdict `CHAIN-WIRING token parity: PASS - the hybrid chain produced the same token id as the
in-process reference engine; this is chain-wiring parity, not a device-token claim, and criterion 1
remains OPEN`. This is **chain-wiring parity, not a device-token result**, and criterion 1 remains
**OPEN**.

Attention totals, verbatim: `attention totals: projection_launches=200 body_launches=40 staging_ms=5876
projection_us=535311 body_us=13980 staged_matrix_bytes=5069568000 host_bytes=11274640
cubin_bytes=81672 device_buffer_bytes=46658816`. Chain totals, verbatim: `chain totals: layers=40
entry_launches=42 post_launches=40 attention_projection_launches=200 attention_body_launches=40
device_launches=322 engram_split_layers=2 device_us=568495 device_ms=568.495 staging_ms=6575
host_ms=19816 total_chain_ms=22450 host_bytes=72933600 staged_matrix_bytes=5307148480
cubin_bytes=151984 device_buffer_bytes=48723168`.

**What is device per layer now.** The format-2 launch is real on all 40 layers: 64 head blocks of
256 threads, one launch per layer, at position 0, in a chain that also has the 42 MhcEntry, 40 MhcPost
and 200 projection launches (322 total). The body costs `body_us=13980` for 40 launches, 349.5
microseconds per layer at row 1, which is a measured row-1 number, not a throughput claim. For the
20 ratio-1 layers the backend now also uploads the single reachable compressed candidate: the host's
FP4-round-tripped row is re-encoded with `quantize_main_kv`, stored at pool position 0, and the
selected list is exactly `[0]`; the format-2 kernel attends the one window row plus that candidate.
The ratio-0 and ratio-2 layers have no compressed candidate at position 0. **The 512-candidate /
640-slot regime is not exercised** by this window; this is one compressed candidate on the ratio-1
layers only.

**Evidence discipline added after b32.** Each layer's attention ledger label is now derived from the
backend's per-layer launch count: `[device: format-2 flash-MLA, 1 launch; host q/window/selected
seams]` when 1, or a `HOST STAND-IN ... body_launches=N` label when not. The report additionally
refuses to print the c4a-2 seam line and totals if any layer's `body_launches != 1`, so the
per-layer ledger, the seam line and the totals cannot disagree silently. The host-control engine test
now builds a 40-layer plan and asserts 40 body calls and 200 projection calls; it fails if a layer
falls back to the host loop.

The attention host seams that remain are q_norm, kv_norm, RoPE, window row construction, the
compressor/cache and selected-list assembly, indexer scoring (zeros plus sentinels, writes ignored),
inverse RoPE and the D8 quantisers. The remaining per-layer device-status host stand-ins are
`moe_router`, `moe_gate_up`, `moe_down_combine` and `shared_expert_gate` (4). The claim stays
**CHAIN-WIRING parity, not a device-token result; criterion 1 remains OPEN**.

### c3 window b22: PASS - CHAIN-WIRING token parity (NOT a device token)

Window b22 ran the c3 arm on binary `sparkflash_m3c_f4446e7_aarch64` (sha256
`ced4a090d41c32bf55f10c99269faca3260181263b9a979446a83fe2167f5d5a`, built from `f4446e7`),
cubin `e21293e3...`, exit 0. Logs: `window-b22-20261019T033707Z.log` and
`runs22/m3c-hybrid-token0.log` on the node; vLLM was stopped and restored. The report printed the
identity device entry path, the BF16 residual staging seam, the eight host stand-ins, the no-cache
seam and the c3b follow-up before the result lines.

The in-process reference ran first on the same checkpoint: `reference engine: token_id=5;
wall_ms=31458; top5=5: 13.230665207, 201: 12.492528915, 372: 11.897726059, 30: 11.392500877,
1897: 11.015835762`. The hybrid chain then printed `hybrid chain: token_id=5; wall_ms=23817;
top5=5: 13.206743240, 201: 12.493413925, 372: 11.897836685, 30: 11.358639717, 1897:
11.019292831`. The comparison line was `token comparison: reference_id=5 hybrid_id=5
exact_equal=true top5_overlap=5`. The verdict was verbatim: `CHAIN-WIRING token parity: PASS - the
hybrid chain produced the same token id as the in-process reference engine; this is chain-wiring
parity, not a device-token claim, and criterion 1 remains OPEN`.

Chain totals, verbatim: `layers=40 device_launches=40 device_ms=0 staging_ms=751 host_ms=21219
total_chain_ms=23817 host_bytes=68358400 cubin_bytes=70312 device_buffer_bytes=2064352`. Progress
lines were emitted at layers 8/16/24/32/40 with chain_elapsed 5017/9566/13944/18339/22692 ms.

**What `device_ms=0` covers.** The timed device launch is accounted in the device status:
`device_ms` is the sum over the 40 layers of the host wall-clock interval that starts after the A
copy and covers the BF16 X write, the MhcEntry launch, and five read calls issuing six device-to-host
copies (entry capture plus 24 coefficients in one call, X, `1/rms`, FP8 codes, E8M0 scales). The A
row copy happens before `device_started` and is not in `staging_ms` either; it is outside both
timing columns. Every per-layer interval was below 1 ms, so the sum truncates to 0. It is not a
CUDA-event GPU duration and it is not a count of device work. The 40 launches did happen and are
accounted by `device_launches=40`, by the per-layer `ledger layer NN:` line (which carries
`mhc_entry` [device identity seam]) and the per-layer `layer NN:` line (capture/codes/scales and
coefficient evidence). c3 builds no `ChainLedger` or `ExecutionEvidence` and prints no separate
launch-status line. `staging_ms=751` is the per-layer tensor-staging total and `host_ms=21219` is
the host stand-in continuation total; neither is device time. A reader must not infer from
`device_ms=0` that no device work occurred.

The token ids match, the top-5 overlap is 5/5, and the top-5 values differ slightly because of the
declared BF16 residual staging seam: the host continues each layer from the device-readback
BF16-rounded X, so the chain is not bit-identical to the f32 reference even though the host
arithmetic is exact. This is **chain-wiring parity**, not a device-token result. Only `MhcEntry` is
device-executed per layer; `mhc_post`, `ffn_epilogue`, attention and the head are host stand-ins
and host seams, exactly as labelled. Phase 2 exit criterion 1 remains **OPEN**.

**c3b follow-up (task-13) - closed by b29.** c3b defers each layer's `ffn_epilogue` into the next
device entry so the kernel's real previous-layer B/C/Y mixer is exercised, reducing the
device-status host stand-ins from 8 to 7. Window b29 passed the chain-wiring token gate and is the
c3b record; the layer-0 identity entry remains the only primary identity entry (the two Engram
re-entries are counted in the 42), and `ffn_epilogue` is 39
deferred additions plus 1 host addition at layer 39. c3 is not a full device mHC path; c3b is the
device mHC path, still with the 7 named host stand-ins and a chain-wiring claim only. The batch-13
/MR description must name the layer-0 identity-entry limitation and link task-13.

Criterion 1 remains **OPEN**. The b19 record is the one-layer window c2 PASS; the 40-layer c3
arm is recorded in b22 above, and M3-d adds the boundary capture.

**M4 - full 130 + 8 run and the criterion-1 claim.** The scope formerly called M3. The full phase
of section 5 with the boundary capture at 129, the contract of section 4, and the report artifacts.
The first point at which this note's subject may be called passing - as a hybrid claim, with
labelled host seams, the token path labelled "hybrid: token produced by host math over device
logits", the FFN stage labelled hybrid while gate/up is merged but not wired, and the head's
pruning path labelled as the m16 stage PASS at b25 (4,899 certified, counts agree) with the
chain's head still a host stand-in and criterion 1 OPEN.

File layout: `crates/sparkflash-device/src/{lib.rs, engine.rs, session.rs, chain.rs,
chain_device.rs, budget.rs, staging.rs, window.rs, cubins.rs}`, `crates/sparkflash-device/tests/{engine_offline.rs,
chain_contract.rs}`, the
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

