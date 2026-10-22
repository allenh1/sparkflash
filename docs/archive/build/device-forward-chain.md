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
`crates/sparkflash-kernels/baseline/kernel_baseline.txt` has **thirteen** entries: the six m4 kernels,
`swiglu_mxfp8_quant_sm121` (host seam, not launched by the chain), its weighted routed variant
`swiglu_mxfp8_quant_weighted_sm121` (D33; m17a stage 3 loads it, window b33 measured 35 checks
at both token counts, not wired),
`moe_router_sm121` (M2 one-launch proof), `routed_gate_up_sm121` (task-8, measured in
isolation, not wired), `rmsnorm_sm121` (D34, wired as the c4c final tail norm in b41) and
`lm_head_reduce_sm121` (D35; window b44 measured m19 PASS, 17/17; not wired until the cut-over
lands). The
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
only when **all** hold in one run.

**Status (2026-10-19): NOT claimed, still OPEN, and measured FAILING end to end.** The first complete
criterion-shaped 130+8 run (b68) finished all 138 positions with flat memory, and at position 129
`128528 of 129280 compared logits` are outside the 0.02 bar (`max |difference| 15.009581`, argmax
8060) with **0 of 9** decoded tokens matching the exactly-reproduced fixture
(`491, 367, 201, 80, 3302, 201, 54, 201, 4341`). Memory is fixed (RSS flat at ~18.9 GB over 138
positions) and the decode path works. The remaining blocker is arithmetic: MXFP8 activations against
a f32 reference, with the attention-path drift flipping the router's expert set from layer 3 on. The
records are the b55-b68 subsections in section 7; the precision-versus-criterion decision is [DECISIONS.md](../../DECISIONS.md) D37.

**Update 2026-10-19 (D39): the token clause below is retired.** The chain runs the checkpoint's native arithmetic (32-element ue8m0-ceil activation quantiser, 32x32 ue8m0 / FP4-32 weights, f32 accumulate) and that is not what the f32 fixture computes; exact token reproduction is therefore not a criterion, and it is not attainable. Acceptance is numeric and distributional at declared tolerances against a native reference (the vLLM-released oracle, D38 actioned) and against the arithmetic twins. The token list and item 2 below are kept as the historical bar only.

The PASS definition is:

1. exit status 0, and the run actually executed on a device: driver loaded, every cubin sha256
   printed and equal to the thirteen parsed entries of `baseline/kernel_baseline.txt`; the catalog
   requires nine of them (the six chain kernels plus the router, routed gate/up and `rmsnorm_sm121`,
   2.2), and the other four - `swiglu_mxfp8_quant_sm121` (the host seam's unweighted epilogue), its
   weighted routed variant `swiglu_mxfp8_quant_weighted_sm121` (D33, measured in m17a) and the certified
   head's device reduce `lm_head_reduce_sm121` (D35, m19 measured b44; required only when the cut-over
   lands) - are not chain kernels. Device kernel launches > 0 (a host-only run cannot pass);
2. **[retired by D39]** tokens exactly 491, 367, 201, 80, 3302, 201, 54, 201, 4341 (`compare_steps`, `:699`);
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
for 40) and round-trips the oldest-first view; `cubins.rs` parses the thirteen baseline pins from
`kernel_baseline.txt`, requires nine of them (the six chain kernels plus router, routed gate/up and
the c4c tail `rmsnorm_sm121`) and verifies size and SHA-256 fail-closed; the other four,
`swiglu_mxfp8_quant_sm121` (the host seam's unweighted epilogue), its weighted routed variant
`swiglu_mxfp8_quant_weighted_sm121` (D33), the certified head's device reduce
`lm_head_reduce_sm121` (D35, m19 measured b44; required only when the cut-over lands) and the
deferred sequence's position-causal CSA2 attention entry `flash_mla_csa2_seq_sm121` (D36, pinned;
window b50 measured m20 PASS 50/50; not wired in), are not required chain kernels.
`rmsnorm_sm121` is no longer pending: m18 measured it at 8 and 130 tokens and b41 wired it. The
CLI arm
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

### c4b window b34: PREFLIGHT PASS, FULL CHAIN FAILURE - the shared-expert seam overwrote the routed row

Window b34 ran the c4b arm on binary `sparkflash_m3c_c4b_12cbb46_aarch64` (sha256
`ac1f766d39addb931651bc5e75c5def92564538b549cc1012ec1b02569f5250e`, built from `12cbb46`,
which is based on `da8d491`). **This is a failure record, not a c4b PASS.** The one-layer
staging/launch preflight passed and its measured number is useful; the full 40-layer chain then
failed token parity.

The preflight line was verbatim: `moe preflight: layer=0 tokens=1 launches=17 one_layer_ms=203.517
staging_ms=188 device_us=183663 staged_expert_bytes=233734840 host_bytes=112896024
staging_gbs=1.148 projected_full_chain_s=8.1`, then `MOE PREFLIGHT PASS: projection under the 500 s
bar`, exit 0 at 06:39:09Z. The staged-expert bytes are the full selected-expert tensors (about 113
MB) plus the per-rank gate/up and w2/scales slices (about 121 MB); they are not an 8.2 GB/layer
staging.

The full chain printed verbatim `reference engine: token_id=5 wall_ms=31202 top5=5,201,372,30,1897;
hybrid chain: token_id=271 wall_ms=19793 top5=271,16,53,50,6328`, then `token comparison:
reference_id=5 hybrid_id=271 exact_equal=false top5_overlap=0`, then `first differing layer (best
effort): layer 0 residual_out max_abs_delta=9.425920e0; this includes the BF16 staging seam and any
device-entry delta`, and `CHAIN-WIRING FAIL` exit=1 at 06:40:00Z. Chain totals verbatim: `layers=40
entry_launches=42 post_launches=40 attention_projection_launches=200 attention_body_launches=40
routed_moe_launches=680 device_launches=1002 engram_split_layers=2 device_us=6758993
device_ms=6758.993 staging_ms=13258 host_ms=17045 total_chain_ms=19793`. Every per-layer line showed
`moe_launches=17` and the attention `capture_exact=true codes_exact=true scales_exact=true`; those
bars cover the attention readbacks, not the routed MoE output, so they could not see this failure.

**Root cause.** `moe::add_shared_expert` (new in c4b) had a single-token path that called
`expert_body(..., row)` with `row = out[token*dim..]`, writing the shared expert directly over the
routed contribution instead of adding to it. The multi-token worker path used `+=` and was correct,
which is why a one-token chain diverged from layer 0 while the 680-call host-control test (it fills
zeros and counts launches) and the attention bars stayed green. The fix splits the host reference
into `routed_forward` + `add_shared_expert`, makes the single-token path accumulate into
`scratch.expert_out` and then add, and adds `add_shared_expert_adds_to_the_routed_row_and_never_
overwrites_it` as the failure-class pin in `12cbb46`'s tree.

**Diagnostic bar added, not another token-only gate.** `c2a205d` first compared the device routed
contribution with the host f32 `routed_forward` reference and failed above a guessed `1e-2`; b35
showed that guess was wrong (see below). `e06e9ff` replaced it with the measured class bar: the host
bf16 chained combine (`mxfp4_ref::moe_combine`) on the SAME device codes/scales and w2/s2 slices,
zero seed, must match every per-rank and final bf16 code exactly. The f32 delta, host magnitude and
relative delta are still printed on the `routed MoE totals` line (`layer0_host_max_abs`,
`layer0_host_max`, `layer0_host_rel`) as calibration, not as an abort; the bar-run cost is inside
`total_chain_ms` but outside the staging/device counters. The fix binary
`sparkflash_m3c_c4b_c2a205d_aarch64` (sha256
`1ba3680172f381c4148428e5431c1e7a66f9bf748a5a83ca8436a85e52990128`) is superseded by
`e06e9ff`; b34 remains a failure record.

### c4b window b35: FAILURE RECORD - the layer-0 bar was a guessed f32 tolerance, not a class bar

Window b35 ran `sparkflash_m3c_c4b_c2a205d_aarch64` (sha256
`1ba3680172f381c4148428e5431c1e7a66f9bf748a5a83ca8436a85e52990128`) with the shared-expert
overwrite fixed and the first layer-0 bar in place. The chain aborted at layer 0 before any token
comparison: `device routed MoE layer-0 bar: the device routed contribution differs from the host
routed reference by max abs 1.9090915e-1 (bar 1e-2); the chain token gate is not the only
detector`, exit 1 at 06:53:37Z; log `window-b35-20261019T065305Z.log`, run
`runs35/c4b-hybrid-token0.log`. **This is a failure record for that bar**, not a token result and
not a c4b PASS. The rest of layer 0 was exact: `partial layer 0 timing: staging_ms=17 entry_us=235
post_us=234 capture_exact=true codes_exact=true scales_exact=true coeff_max_abs=8.493662e-7
inv_rms_delta=1.764297e-5 post_capture_exact=true post_codes_exact=true post_scales_exact=true
post_coeff_max_abs=4.053116e-6 post_inv_rms_delta=1.144409e-5`, and all 17 routed launches ran.

The number was a calibration finding, not a wiring finding. The c4b device path returns the routed
contribution through a four-rank bf16 serial combine with a zero seed (the engine adds residual and
shared afterwards in f32); the host `routed_forward` reference sums the six choices in f32 with no
intermediate bf16 rounding. m17a's stage-4 class is bit-exact only against the **bf16 chain on the
same device operand bytes**, so `1e-2` was never the measured class bar. The bar was replaced by
that bf16-twin identity, with the f32 difference retained as printed calibration rather than an
abort (`c2a205d`'s issue, fixed in `e06e9ff`).

### c4b window b36 attempt 1: INFRASTRUCTURE FAILURE - binary not staged on the window host

The first b36 invocation, log `window-b36-20261019T070444Z.log`, exited 127 because
`sparkflash_m3c_c4b_e06e9ff_aarch64` was present on the build host but had not been copied to
node0. **No kernel ran and no measurement was produced**; this is an infrastructure failure
record, not a c4b result. The binary was copied and the same command was rerun.

### c4b window b36: PASS - CHAIN-WIRING token parity with the routed MoE family on device

Window b36 ran `sparkflash_m3c_c4b_e06e9ff_aarch64` (sha256
`7e1f073e6c2923a5d72a04474d4944953fc7eeafa35b24a5520ab50f85a0bb5a`, built from `e06e9ff`) with
`--moe-layer0-diagnose`; log `window-b36-20261019T070801Z.log`, run
`runs36/c4b-diagnose.log`, exit 0 at `2026-10-19T07:09:58Z`; the log carries
`=== RUN c4b-diagnose exit=0 end 2026-10-19T07:09:58Z` and
`=== all runs complete 2026-10-19T07:09:58Z ===`. It reused the `window-b36.sh` script name; the
first attempt at `07:04:44Z` exited 127 and is the infrastructure failure above. The
verdict line was `CHAIN-WIRING token parity: PASS`; the comparison was `token comparison:
reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=5`. This is **chain-wiring parity, not a
device-token result**, and criterion 1 remains **OPEN**.

**The layer-0 routed block is calibrated, not merely tolerated.** The diagnostic lines were verbatim:
`moe layer0 diag: weighted rank0 codes_mismatched=0 scales_mismatched=0 (width=576, rows=6)` and
`moe layer0 diag: bf16 combine twin exact=true mismatches=0 per_rank=[0, 0, 0, 0]
host_f32_max_abs=1.9090915e-1 host_max=9.479069e0 relative=2.0140074e-2`. The bar is therefore the
measured bf16-twin identity (zero mismatches on every per-rank and final bf16 code, same device
codes/scales, same w2/s2 slices, same local ids); the f32 delta of `1.909e-1` against a host max of
`9.479` (2.014% relative) is the declared f32-vs-bf16 serial-combine class spread and is printed as
calibration (`layer0_host_max_abs`, `layer0_host_rel`, `layer0_host_max`), not used as an abort.
This is a different magnitude from b34's `9.425920e0` residual delta, which was the shared-expert
overwrite bug fixed in `c2a205d`.

The routed MoE totals line reported `layers=40 router_launches=40 gate_up_launches=320
weighted_launches=160 combine_launches=160 moe_launches=680 staging_ms=2982 device_us=2945226
staged_expert_bytes=9349393600 host_bytes=4515856824 layer0_host_max_abs=1.909e-1
layer0_host_rel=2.014e-2 layer0_host_max=9.479e0 layer0_bf16_twin_mismatches=0 ...` (the full line
is in the run log). Chain totals reported `layers=40 entry_launches=42 post_launches=40
attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680
device_launches=1002 engram_split_layers=2 device_us=3514609 device_ms=3514.609 staging_ms=10028
host_ms=14793 total_chain_ms=17631 host_bytes=4588790424 staged_matrix_bytes=14656542080 ...`.

### c4c implementation: shared expert and certified head on device (b40 PASS - wiring, norm pending)

Branch `task/device-engine-m3-c4c` at post-rebase `c573f34`, based on main `acc0189` (batch 19
merged). The c4c implementation commits are `ae27499`/`3d60f60`/`1031891`/`d44202c`/`c573f34`; the c4b
commits below them are the same routed-MoE family rebased onto the merged batch-19 tree. The b40
window candidate was `sparkflash_m3c_c4c_d44202c_aarch64` (sha256
`5e674a2024f89a33f22f0d0bfc31f2bdc07c94d4c09f40d920cae5543cc1fe0b`); `c573f34`
(`sparkflash_m3c_c4c_c573f34_aarch64`, sha256
`7037cebeb33ea07afb335a4cc92e639762b4aa9e6bae8060740537f4f76d40c1`) changes only the shared timing
labels and is the base for the norm-tail change.

**Shared expert, four launches per layer.** The engine's device arm now calls
`MoeDeviceBackend::shared_expert` after the routed FFN. The released shared expert is one expert
(`moe_inter_dim=2304`, `dim=5120`): w1/w3 are FP8 E4M3 `[2304, 5120]` with the 32x32 E8M0 grid
`[72, 160]`, w2 is FP8 E4M3 `[5120, 2304]` with grid `[160, 72]`. The backend repacks each weight
to the m12 B12x micro-tile order (a value-preserving index permutation, no arithmetic), quantises the
layer's activation rows with the D8 rule, then launches, in order, dense w1, dense w3, the 2304-wide
`swiglu_mxfp8_quant_sm121` (unweighted; the shared expert has no routing weight), and dense w2. The
kernels and pins are m17b's: `b12x_mxfp8_dense_sm121` 11,824 B `03e4c422...`,
`swiglu_mxfp8_quant_sm121` 21,776 B `4dbef63b...`. The device shared output is BF16; the engine adds
it to the routed f32 row on the host. The host add has no pinned elementwise kernel and is a declared
seam. The host-twin diagnostic reads the layer prefix `layers.N` (not the shared-expert base) so
`add_shared_expert` composes the released names exactly once; `read_spec` asserts dtype **and**
logical shape before any byte is trusted.

**Certified head, four rank launches.** The deferred tail calls `MoeDeviceBackend::lm_head` with the
host-normalised tail. The backend converts that f32 row to BF16 (the m16 ABI), reads
`head.weight` as BF16 `[129280, 5120]`, and launches `cert_lm_head_sm121` once per 32,320-row rank
shard (pin 14,888 B `7394b129...`). Each launch publishes 32 chunk maxima/argmaxes and a certified
mask/count; the host assembles the four shards' block maxima with the lowest-global-index tie-break.
The head's per-entry exactness is the kernel certificate's; entries the screen pruned carry bounds,
not exact logits, so the chain's `top5` after a device head is ordered by published block values and
is not an exact-logit measurement. The host block assembly is the declared reduction seam.

**What is device and what is not.** c4c removes the last per-layer device-status stand-ins: routed
MoE (17 launches/layer, c4b), shared expert (4 launches/layer), and the head (4 launches once). The
per-layer MoE ledger now requires exactly **21** launches (17 routed + 4 shared) and the chain totals
count 40 x 21 = 840 MoE launches (680 routed + 160 shared) plus 4 head launches; with the merged
batch-19 attention trio (42 entry + 40 post + 200 projection + 40 body) that is **1166 device
launches** in the b40 window. The chain totals print the split explicitly:
`routed_moe_launches=680 shared_moe_launches=160 head_launches=4 moe_launches=840`.

The c4c-norm commit adds the standalone `rmsnorm_sm121` tail norm, so a b41 run also prints
`tail_norm_launches=1` and `device_launches=1167`; that changes the last device-status stand-in
before the head, and it is a separately measured change (the b40 record below is the host-norm
chain). The remaining host seams after c4c-norm are: the shared-add elementwise, the B12x weight
repack, the activation D8 quantiser, the selected-expert staging/dispatch, the head's BF16 conversion
and four-shard block assembly, and the attention/tail seams named in the c4a/c4b records. **The
four-shard block-max/argmax assembly is still host, so c4c/c4c-norm cannot claim a device token.
Criterion 1 remains OPEN until a full 130+8 run meets section 4.3 items 1-6.**

Single-stream decode and the batched-reduce drop are a separate contract: for one position the
head needs the four rank screens plus the exact f32 logit gather, not the batched reduce that the
verify path uses. The harness m27_head_tail, its exactness table, the launch inventory and the
head-stage evidence contract are in [head-tail-contract.md](head-tail-contract.md) (task-16, WS-C).

### c4c window b38: MID-RUN STAGING FAILURE - duplicated `shared_experts` in the host-twin name

Window b38 ran `sparkflash_m3c_c4c_6927573_aarch64` and aborted after the layer-0 attention and
routed-MoE launches (5 projection + 1 format-2 body + 17 routed) but before any token measurement:
`c4a-2 chain abort: the source has no tensor named
layers.0.ffn.shared_experts.ffn.shared_experts.w1.weight`, exit 1 at
`2026-10-19T07:46:35Z`; log `window-b38-20261019T074449Z.log`, run
`runs38/c4c-hybrid-token0.log`. **This is a mid-run staging failure after real device launches,
not a c4c result**; no token measurement is recorded. The released names are `layers.N.ffn.shared_experts.w{1,2,3}.weight` and
`.scale`. The duplicate came from the layer-0 host twin: `diagnose_shared` passed the shared base
`layers.N.ffn.shared_experts` to `add_shared_expert`, whose prefix contract appends
`.ffn.shared_experts` itself. Fixed in `1031891` by passing the layer prefix `layers.N` to the
host twin, plus the dtype/shape check above. b38 is retained as the failure record; the next window
is b39 on the rebased binary.

### c4c window b39: FIRST c4c PASS - canonical numbers, combined routed/shared label

Window b39 ran `sparkflash_m3c_c4c_1031891_aarch64` (sha256
`935cf17a9796789738e365e0faf63678003e7f6f21f4e0380eb28f745231749b`) and **PASSED** the token
comparison (`token comparison: ... exact_equal=true`, `CHAIN-WIRING token parity: PASS`). Log
`window-b39-20261019T075605Z.log`; the run is in the same window's `runs39/c4c-hybrid-token0.log`.
Its chain totals printed `routed_moe_launches=840 device_launches=1166`: 840 was the combined
routed+shared MoE count under a label that said "routed". The total 1166 and the token were correct;
only the reporting label was wrong. b39 is retained as the **first c4c PASS**, and it is honest
evidence that the split fix changed reporting only. It is superseded as canonical by b40.

### c4c window b40: PASS - c4c CHAIN-WIRING parity (NOT criterion 1, final norm still host)

Window b40 ran `sparkflash_m3c_c4c_d44202c_aarch64` (sha256
`5e674a2024f89a33f22f0d0bfc31f2bdc07c94d4c09f40d920cae5543cc1fe0b`) and exited 0 at
`2026-10-19T08:02:19Z`; log `window-b40-20261019T080027Z.log`, run
`runs40/c4c-hybrid-token0.log`. **This is the canonical c4c wiring PASS.** It is not a criterion-1
PASS: the final RMS norm was still host in this window, and the head's four-shard block reduction is
host. The measured lines were:

```text
moe layer0 diag: router ids device=[250, 209, 215, 104, 236, 275] host=[250, 209, 215, 104, 236, 275] equal=true weights_max_abs=1.1920929e-7
moe layer0 diag: staged 6 experts local->global [(0, 250), (1, 209), (2, 215), (3, 104), (4, 236), (5, 275)], expert_bytes=112803840, rows=6
moe layer0 diag: gate_up rank0 device-vs-host bits mismatched=0 up=0 of 3840 padded values
moe layer0 diag: weighted rank0 codes_mismatched=0 scales_mismatched=0 (width=576, rows=6)
moe layer0 diag: bf16 combine twin exact=true mismatches=0 per_rank=[0, 0, 0, 0] host_f32_max_abs=1.9090915e-1 host_max=9.479069e0 relative=2.0140074e-2
moe layer0 diag: shared gate/up bf16 mismatched=0/0 codes=0 scales=0 shared_bf16_mismatches=0
moe layer0 diag: shared host_f32_max_abs=1.3034999e-2 host_max=1.5967474e0 relative=8.16347e-3
lm head diag: certified=0 bf16_twin_logit_mismatches=0 bf16_twin_max_abs=0e0 device_token=5 twin_token=5 token_match=true
shared expert totals: layers=40 shared_launches=160 staging_ms=397 device_us=397895 staged_shared_bytes=1417171200 host_bytes=4516477624 layer0_shared_bf16_mismatches=0 layer0_shared_host_max_abs=1.303e-2 layer0_shared_host_rel=8.163e-3 layer0_shared_host_max=1.597e0
chain totals: layers=40 entry_launches=42 post_launches=40 attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680 shared_moe_launches=160 head_launches=4 moe_launches=840 device_launches=1166 engram_split_layers=2 device_us=5052022 device_ms=5052.022 staging_ms=9662 host_ms=14770 total_chain_ms=17555 host_bytes=4591049624 staged_matrix_bytes=14... (this section abbreviates a long line; the run output itself carried the full number and no truncation marker)
reference engine: token_id=5; wall_ms=23670; top5=5: 13.230665207, 201: 12.492528915, 372: 11.897726059, 30: 11.392500877, 1897: 11.015835762
hybrid chain: token_id=5; wall_ms=17555; top5=5: 13.150072098, 201: 12.476766586, 372: 11.830552101, 30: 11.281064987, 271: 10.979187965
token comparison: reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=4
CHAIN-WIRING token parity: PASS
```

The **class bars** are the bf16-twin identities, not an f32 tolerance: shared
`shared_bf16_mismatches=0` (the f32 delta `1.3034999e-2` against `host_max=1.5967474e0`,
relative `8.16347e-3`, is calibration); routed combine `exact=true mismatches=0` with its own f32
calibration `1.9090915e-1`; head `certified=0`, `bf16_twin_logit_mismatches=0`,
`bf16_twin_max_abs=0`, and both twin and device token 5. The hybrid top5 overlaps the reference in
4 of 5 entries but is **not** an exact-reference-logit claim: the device head consumes a BF16 tail
while the reference head consumes its f32 tail, and the certified screen's entries are only exact
when `certified=0` (as here). Token parity is the gate.

**Timing-label caveat, fixed after b40.** In d44202c the shared line's `staging_ms=397` and
`device_us=397895` are the same total (397,895 us) under two labels: `shared_us` was charged whole
to both. c573f34 splits shared staging from launch+readback and adds the shared staging to
`staging_ms`, so a later run's two numbers are independent; b40's raw line is kept above as the
measured record.

**Cumulative host_bytes caveat.** In the b40-era totals lines the host_bytes field is a cumulative
chain counter on both the routed and shared lines (the b45 section below quotes the same defect); it
is not a routed-only or shared-only byte count and must not be differenced as if it were. The split
accounting starts with the post-fix c4c-norm run.

**The b40 device chain, verbatim counts:** 42 entry + 40 post mHC, 200 attention projection + 40
format-2 body, 680 routed MoE + 160 shared MoE + 4 certified head = **1166 device launches**. The
final RMS norm before the head was host, and the head's block-max/argmax assembly across the four
rank shards was host. c4c is therefore **CHAIN-WIRING parity with a hybrid token path**, not
criterion 1.

### c4c-norm: standalone final RMSNorm wired (b41 PASS)


Branch `task/device-engine-m3-c4c` at `611d21a` (kernel-dev's `e09e08c` + `42beb9e` cherry-picked
onto the b40 tree). The tail now calls `MoeDeviceBackend::final_norm`, which the CLI backend
implements by asserting `norm.weight` is BF16 `[5120]`, converting the collapsed f32 row to BF16,
launching `rmsnorm_sm121` (31,480 B pin `5f953bc0...`) with `eps = config.norm_eps as f32`, and
comparing the BF16 output **byte for byte** against `rmsnorm_ref::final_norm_bf16` (the host
`linalg::rms_norm` over the BF16-decoded input, then `bf16_bits`). A mismatch aborts; the host
`linalg::rms_norm` remains the fallback when the backend returns `false`. The device head consumes
the resulting BF16 tail through its existing BF16 hidden ABI.

`sparkflash-device::cubins::REQUIRED_KERNELS` now has nine entries (rmsnorm added), the chain totals
add `tail_norm_launches` and the 31,480 cubin bytes, and the device-launch total becomes 1167. The
m18 measurement already passed the kernel at 8 and 130 tokens (0 mismatched BF16 bytes of 81,920 and
1,331,200); the b41 window below measures the chain wiring itself.

### c4c-norm window b41: PASS - device tail norm, 1167 launches

Window b41 ran `sparkflash_m3c_c4c_611d21a_aarch64` (sha256
`2f13cbe8875e24e08da7b6bac150dfec6796e885caaefcd73ed5af177eaab0a9`), cubin `rmsnorm_sm121`
`5f953bc0...`, exit 0 at `2026-10-19T08:16:34Z`; log `window-b41-20261019T081430Z.log`, run
`runs41/c4c-norm-hybrid-token0.log`. **This is the c4c-norm PASS**: the final norm is no longer a
host seam, and the chain moves 1167 device launches. Verbatim:

```text
final norm diag: rows=1 cols=5120 bf16_mismatches=0 bytes=10240
final norm totals: final_norm_launches=1 device_us=265 staged_norm_bytes=30720 bf16_mismatches=0 screen=rmsnorm_sm121; host readback into the certified head
lm head diag: certified=0 bf16_twin_logit_mismatches=0 bf16_twin_max_abs=0e0 device_token=5 twin_token=5 token_match=true
reference engine: token_id=5; wall_ms=31859; top5=5: 13.230665207, 201: 12.492528915, 372: 11.897726059, 30: 11.392500877, 1897: 11.015835762
chain totals: layers=40 entry_launches=42 post_launches=40 attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680 shared_moe_launches=160 head_launches=4 tail_norm_launches=1 moe_launches=840 device_launches=1167 engram_split_layers=2 device_us=4895227 device_ms=4895.227 staging_ms=12777 host_ms=15964 total_chain_ms=18799 host_bytes=4591049624 staged_matrix_bytes=16073744000 ...
hybrid chain: token_id=5; wall_ms=18799; top5=5: 13.139515877, 201: 12.485794067, 372: 11.826237679, 30: 11.279764175, 271: 10.986923218

token comparison: reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=4
CHAIN-WIRING token parity: PASS
```

The norm's class bar is the bf16-bit twin, not a tolerance: **0 mismatches of 10,240 bytes** (5,120
values); the 265 us is the launch plus one readback. The b40 host-norm result remains a separate,
unchanged record.

**Accounting note (batch 22, HIGH).** The b41 `staging_ms=12777` above is the c4c **chain
total**, not a routed-only number: it already includes the shared-expert staging that
`moe_device.rs` charged on the same counter. An earlier review text proposed a larger total by
adding the shared staging a second time; that figure is deleted and must not be cited. The b41 line
is retained exactly as measured. b45 and b52 are PRE-FIX runs whose routed/shared totals lines are
the old cumulative ones; the post-fix split record is window b54's token-0 `c4c-norm-token0`
run on binary `e7c0456` (`window-b54-20261019T105232Z.log`; b54's `seq-timing-3` exited 1 on the
known p0/seq divergence and its `seq130x8` was interrupted, so neither is a sequence record, and
b52 `window-b52-20261019T103800Z.log` is superseded PRE-FIX history), which printed
`moe counts: routed_launches=680 shared_launches=160 ... routed_staging_us=3184489 shared_staging_us=1775648
total_staging_us=4960137`, distinct routed/shared host bytes, `head_launches=5` and
`device_launches=1168` (the reduce cut-over), with token 5 == reference 5. Do not cite b41's
12,777 as routed-only.

**The host pieces left before the token are the four-shard block-max/argmax assembly and the
one-time head-weight read/upload** (the head screen and in-kernel block argmax are device); the
shared add, the B12x repack, the D8 quantiser, selected-expert staging and the c4a/c4b attention seams
remain host. The
token is therefore still **hybrid: host math over device logits**. Criterion 1 remains OPEN until a
full 130 + 8 run meets section 4.3 items 1-6. The batch-22 review additionally found that this
sentence omitted the head-weight read/upload, which is host work before the token as well; the
claim text below names it.

**Proposed criterion-1 claim text (not yet claimed by c4c).** c4c is a wiring result only. A
criterion-1 sentence may be issued only after a full 130 + 8 run passes all six items of section 4.3;
if it does, the honest sentence is:

> 130-token prompt prefill plus 8 decode steps reproduced every decoded token (491, 367, 201, 80, 3302,
> 201, 54, 201, 4341). In that run the device executed the mHC entry/post chain, the attention
> projection trio and format-2 body, the routed MoE, the shared expert (dense w1/w3 + 2304-wide
> SwiGLU quantiser + dense w2, four launches per layer), the standalone final RMSNorm
> (`rmsnorm_sm121`, bit-identical to the host norm rounded to BF16), and the certified head screen
> with in-kernel block argmax. **The ONLY host pieces left before the token are the one-time
> head-weight read/upload and the four-shard block-max/argmax assembly**; the shared add, B12x
> repack, D8 activation quantiser, selected-expert staging and the other c4a/c4b attention seams
> remain host inside the layers. The token is therefore
> **hybrid: host math over device logits**, not a device
> token. This is a hybrid criterion-1 PASS with the named host seams, not a full-device-chain claim;
> criteria 2-5 and serving readiness remain unclaimed.

**Lines that falsify the claim.** Any one of these in the same run is a FAIL, not a recorded
divergence:

- `token comparison: reference_id=<a> hybrid_id=<b> exact_equal=false ...` and
  `CHAIN-WIRING token parity: FAIL - the hybrid token id is not the in-process reference token id`;
- `lm head diag: ... token_match=false` or `device head diagnose: the device token ... differs from
  the same-bf16-byte host twin token ...`;
- `final norm diag: ... bf16_mismatches=<nonzero> ...` or `device final norm diagnose: the device BF16
  norm differs from the host bf16 twin ...`;
- `moe layer0 diag: shared ...` with any nonzero gate/up/codes/scales/shared mismatch, or
  `shared expert totals: ... layer0_shared_bf16_mismatches=<nonzero> ...`;
- `c4c seam line withheld: the routed MoE plus shared expert must launch exactly 21 times ...`;
- `c4a-2 chain abort: the source has no tensor named ...`, `declared <dtype> <shape>, expected
  <dtype> <shape>`, or `expected <n> bytes, got <m>` (a staging failure, not a measurement);
- on the full run, `position <p>: fixture token <expected>, device token <found>` or
  `position <p>: max |logit difference| <diff> exceeds the bar 0.02`, or a missing seam-ledger /
  metadata item from section 4.3.

A nonzero boundary delta is a **recorded divergence** and does not rescue a token or logits failure;
the claim sentence must name it if it is present.

### c4c-norm window b45: PRE-FIX accounting run (not the split-counter record)

Window b45 re-ran the review-fix binary before the split-counter fix. Window log
`/home/user/phase2-evidence/window-2026-10-18/window-b45-20261019T090623Z.log`
(sha256 `e3cfa0f079624c83f7365a0ea61754ac10f1d6262501050f025dc89ff378036d`), window start
`2026-10-19T09:06:23Z`, host `node0`. The c4c-norm RUN started `09:07:35Z`, exited 0 at
`09:08:24Z`, binary `sparkflash_m3c_c4c_review_5ea5e2d_aarch64`, sha256
`4c64e1300f66fd3cae59df2441bdf603d653afef2f34290cc30fc9df8befbf63`. The binary predates
`69564a5`, so its routed/shared totals lines are **stale cumulative lines**, not the routed-only
and shared-only split. Quote them only as the stale pre-fix output:

```text
routed MoE totals: layers=40 router_launches=40 gate_up_launches=320 weighted_launches=160 combine_launches=160 moe_launches=680 staging_ms=4937 device_us=4399358 staged_expert_bytes=9349393600 host_bytes=4516477624 layer0_host_max_abs=1.909e-1 layer0_host_rel=2.014e-2 layer0_host_max=9.479e0 layer0_bf16_twin_mismatches=0 layer0_bf16_twin_max_abs=0.000e0
shared expert totals: layers=40 shared_launches=160 staging_ms=1679 device_us=304127 total_us=1983896 staged_shared_bytes=1417171200 host_bytes=4516477624 layer0_shared_bf16_mismatches=0 layer0_shared_host_max_abs=1.303e-2 layer0_shared_host_rel=8.163e-3 layer0_shared_host_max=1.597e0
```

Both lines print the same cumulative `host_bytes` and the routed line's `staging_ms=4937` is the
routed-plus-shared cumulative staging, so neither line is the split counter. The only reproducible
b45 accounting lines are the independently emitted `moe counts` line and the chain totals:

```text
moe counts: routed_launches=680 shared_launches=160 total_launches=840 routed_staging_us=3257768 shared_staging_us=1679769 total_staging_us=4937537
chain totals: layers=40 entry_launches=42 post_launches=40 attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680 shared_moe_launches=160 head_launches=4 tail_norm_launches=1 moe_launches=840 device_launches=1167 engram_split_layers=2 device_us=4968390 device_ms=4968.390 staging_ms=12148 host_ms=15352 total_chain_ms=18182 host_bytes=4591049624 staged_matrix_bytes=16073744000 cubin_bytes=302136 device_buffer_bytes=48723168
```

The c4c token comparison in the same window was
`reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=4`, CHAIN-WIRING parity PASS; it is
not criterion 1. The window's second run, `seq-sanity2`, started `09:08:24Z` and exited 1:
position 0 failed the fixture-token comparison and the CSA2 sequence body launches were absent
(that build had no p>0 sequence entry), so it is a failure record, not a sequence measurement. Logs:
`runs45/c4c-review-hybrid-token0.log` and `runs45/seq-sanity2.log`. The post-fix c4c-norm re-shoot is window b54 (see the accounting note above); b45 is not the split-counter record to cite.

**Embedding-row staging (a29aecd).** hybrid_token still guards --token 0: its in-process
reference is the token-0 slot, so a non-zero prompt id is not accepted there. The a29aecd fix
(stage the requested token's embed.weight row instead of hardcoding row 0) is therefore
**unexercised by this token-0 record**: it is exercised only when a sequence path runs a non-zero
prompt id, and the 3-position sanity run is the first log expected to cite it. The guard is a
deliberate reference-shape limit, not a claim that row 0 is the only embedding row.

### Per-window records retained from the hybrid-sequence branch (b53-b62; superseded by the consolidated record later in this section)

The dated subsections below are the hybrid branch's own window records, kept verbatim so the
merge does not delete a measured window. The consolidated "Hybrid sequence path b55-b62" record
later in this section is the later canon for b55-b62; where the two disagree (for example b53
versus b54 as the split-counter record), the consolidated record and its accounting note govern.

### c4c-norm window b53: split-counter accounting record

Window b53 ran tip 6170230 (binary sparkflash_m3c_c4c_seqreuse_6170230_aarch64, sha256
9853c02b2b7052673473cc851bbded51ecb065d659b867e0e5c55e34d30f3d13), window log
window-b53-20261019T104523Z.log (sha256
8a4e3a571ba1a45987edb9befcef590982dee578762ec87c80019d16be473c63), window start
2026-10-19T10:45:23Z, host node0, MemAvailable 123,991,476 kB. The c4c-norm RUN
started 10:46:38Z and exited 0 at 10:47:30Z; log runs53/c4c-norm-token0.log (sha256
719cc03b517d9ca5c3ea4ab87cca7c37d25fdd6fe13c183c109006c5c73e1f43). This is the first window
whose routed and shared totals lines carry distinct host bytes, so it is the split-counter
accounting record. Verbatim:

```text
routed MoE totals: layers=40 router_launches=40 gate_up_launches=320 weighted_launches=160 combine_launches=160 moe_launches=680 staging_ms=3445 device_us=388431 staged_expert_bytes=9349393600 host_bytes=4515856824 layer0_host_max_abs=1.909e-1 layer0_host_rel=2.014e-2 layer0_host_max=9.479e0 layer0_bf16_twin_mismatches=0 layer0_bf16_twin_max_abs=0.000e0
shared expert totals: layers=40 shared_launches=160 staging_ms=1758 device_us=306274 total_us=2064542 staged_shared_bytes=1417171200 host_bytes=620800 layer0_shared_bf16_mismatches=0 layer0_shared_host_max_abs=1.303e-2 layer0_shared_host_rel=8.163e-3 layer0_shared_host_max=1.597e0
moe counts: routed_launches=680 shared_launches=160 total_launches=840 routed_staging_us=3445105 shared_staging_us=1758268 total_staging_us=5203373
chain totals: layers=40 sequence_positions=1 entry_launches=42 post_launches=40 attention_projection_launches=200 attention_body_launches=40 routed_moe_launches=680 shared_moe_launches=160 head_launches=5 tail_norm_launches=1 moe_launches=840 device_launches=1168 engram_split_layers=2 device_us=5506333 device_ms=5506.333 staging_ms=12793 host_ms=14368 total_chain_ms=17452 host_bytes=4591049624 staged_matrix_bytes=16073744000 cubin_bytes=302136 device_buffer_bytes=48723168 (entry/post/moe/head/norm totals are cumulative over 1 positions; host/bytes/timing are the last position only)
token comparison: reference_id=5 hybrid_id=5 exact_equal=true top5_overlap=4
CHAIN-WIRING token parity: PASS - the hybrid chain produced the same token id as the in-process reference engine; this is chain-wiring parity, not a device-token claim, and criterion 1 remains OPEN
```

The split is observable: routed host bytes 4,515,856,824 and shared host bytes 620,800 are
different counters on the two lines. The chain total staging is 12,793 ms and device_launches=1168
(the lm_head_reduce_sm121 cut-over adds one launch vs b45 1167); the head line reports
head_launches=5 with reduce_used=true, so the token is device-produced. b53 is still CHAIN-WIRING
parity, not criterion 1.

The same window ran the 3-position timing: it printed positions 0/1/2 at wall 18107/8772/8536 ms
and exited 1 at the pre-fix c4a-2 seam because 6170230 predates the sequence seam fix (three
format-2 bodies per layer vs the one-position guard); those timing lines are a measurement, not a
criterion result, and the projection field is superseded by e7c0456. The criterion-shaped 130+8
RUN on 6170230 started 10:49:35Z but was stopped when b54 superseded the binary; it produced no
result and is not a criterion record.

b52 history (superseded). b52 ran the same c4c command on 37b3d45 (sha256
1dc1e2cf18209225b2def2ff06bb6b66db2c87ea8aa36be8a6af170d64b65204). It passed CHAIN-WIRING with
token 5=5 and device_launches=1168, but its routed and shared totals both printed the cumulative
host_bytes=4515856824 because of the shared-host-bytes mischarge, and its 3-position run died at
the same pre-fix c4a-2 seam. b52 is recorded only as history; its host-byte split and its timing
projection must not be cited.

### c4c-norm window b54 and sequence timing b55

Window b54 ran the post-projection binary e7c0456 for c4c-norm and repeated the split accounting:
binary sha256 6b1c15f5e48953918a1fe941ef2dda0a1492f22b7bab33649215d1d7fa19ea39; window
window-b54-20261019T105232Z.log, c4c-norm exit 0 at 10:53:18Z, routed host_bytes=4515856824 and
shared host_bytes=620800 distinct, moe counts routed 3184489 shared 1775648 total 4960137, chain
staging_ms=12620 device_launches=1168, token 5=5 CHAIN-WIRING PASS. The b54 3-position run still
died at the per-layer MoE seam and the b54 130+8 was stopped before a result; b54 is not the
sequence record.

Window b55 ran 641462a (binary sparkflash_m3c_c4c_seqreuse_641462a_aarch64, sha256
a2227d97f7d6adf35823fb751ee863b7752c0f8457edd3f9d05246fd67a5823e), window start
2026-10-19T11:00:46Z, MemAvailable 123,254,016 kB. The 3-position RUN started 11:00:46Z, printed
the three timing lines, passed every seam, and exited 1 only on the known p0 token divergence.
Log runs55/seq-timing-3.log (sha256
223f73bdca0c5561c93850c8c6ac493a132b30a7e1092943aa6829e8621fbe62). Verbatim:

```text
sequence position: position=0 done=1/3 wall_ms=16034 chain_ms=16490 projected_total_s=2348.8 projected_chain_full_s=2275.6 full_positions=138 reference_basis=requested_only budget_s=3600 attention_staging_us=6150565 attention_device_us=541370 moe_staging_us=5922940 moe_device_us=5378222 routed_staging_us=4321254 routed_device_us=436414 shared_staging_us=1601686 shared_device_us=306329 total_staging_us=12073505 total_device_us=5919592 mem_available=105823436800
sequence position: position=1 done=2/3 wall_ms=8257 chain_ms=24748 projected_total_s=1780.8 projected_chain_full_s=1707.6 full_positions=138 reference_basis=requested_only budget_s=3600 attention_staging_us=89497 attention_device_us=536754 moe_staging_us=5557272 moe_device_us=4713179 routed_staging_us=4288779 routed_device_us=384262 shared_staging_us=1268493 shared_device_us=306178 total_staging_us=5646769 total_device_us=5249933 mem_available=102680076288
sequence position: position=2 done=3/3 wall_ms=7735 chain_ms=32484 projected_total_s=1567.5 projected_chain_full_s=1494.2 full_positions=138 reference_basis=requested_only budget_s=3600 attention_staging_us=92278 attention_device_us=546175 moe_staging_us=4964229 moe_device_us=4106543 routed_staging_us=3682850 routed_device_us=387140 shared_staging_us=1281379 shared_device_us=312517 total_staging_us=5056507 total_device_us=4652718 mem_available=99178446848
c4a-2 seam: format-2 body launched exactly 3 times on all 40 layers; 200 projection + 120 body launches plus the 82 mHC launches (42 entry + 40 post)
token comparison: position=0 reference_id=201 hybrid_id=320 exact_equal=false
token comparison: position=1 reference_id=201 hybrid_id=16 exact_equal=false
token comparison: position=2 reference_id=16 hybrid_id=16 exact_equal=true
HYBRID SEQUENCE token parity: FAIL - a position's hybrid token is not the in-process reference token
```

The gate is met: projected_chain_full_s=1494.2 over 138 positions plus b51 measured 614 s full
reference gives about 2108 s, under 3600. The exit 1 is not a seam failure, but it is **not only
the known p0 divergence**: position 0 is device 320 vs reference 201, position 1 is device 16 vs
reference 201, and position 2 is device 16 vs reference 16. The host-fallback b48 run matched
position 1 exactly, so this is a device-body-vs-host-body divergence at p>0 in the sequence entry,
not a state/wrapper issue. The 130+8 full RUN started 11:02:36Z but was killed after this finding;
it produced no result and is not a criterion record. The follow-up is the in-chain host body twin
on c057be2 (below).

### Sequence attention in-chain body twin (fde6b11, b56)

Window b56 ran the short 3-position diagnostic on binary fde6b11 (sha256
eadb442260ee0fbf54c00483d80fed5cedb5d11bd322ca8cffc036888be37101) and produced 120 twin
lines, positions 0-2 x 40 layers, with a per-layer host-body comparison on the same quantised
staged operands. The window log is window-b56-20261019T111735Z.log; the run log is
runs56/twin-3.log, sha256 e679025936189ddd0771756e9fbdfbffbbd862d4341fa2138e0e8113ee2a5d7b.

Result: the device body is faithful on identical operands. Position 0 has 5 mismatched BF16 slots
over 40 x 32768 (single-layer worst 1/32768, exact_heads 63-64/64); position 1 has 38 slots
(single-layer worst 4/32768, exact_heads at least 60/64); position 2 has 47 slots (same bound).
Every per-head-block exact fraction is at least 0.9990 and every max_abs_diff is at most 7.8e-3,
which is the BF16 spacing of these values. The seq entry's indexer, mask, window-row and
selected-list contract is therefore not the token-divergence cause: the device body equals the host
body to BF16 LSB on the operands the device staged. The b55 p1 token mismatch (reference 201,
hybrid 16) is a staged-operand or residual/state divergence, not the csa2 seq kernel.

The earlier first-differing-layer line in b55 was a false comparison: reference residual_out is
token-major and held 3 prompt positions (61440 values) while the hybrid output held one position
(20480), and the old comparison treated the multi-position tensor as the hybrid row. The follow-up
binary slices the reference row before comparing. b56 is a short diagnostic record, not a claim;
the b55 full run remains killed and is not a criterion record.

### Sequence alignment and residual twin (bec6498, diagnostic)

Binary bec6498 (sha256 af6ba20e0bb5eceff420904fa748856c603bff204c3da79f5c1a02073b62c402) adds
the short follow-up: per position 0-2 it prints a fed_token/predicted_top1/logit_max/top5 alignment
line for both hybrid and reference; it records the per-prefix reference boundary and slices the
token-major residual row for the hybrid position; it prints the per-layer residual BF16 mismatch
and max_abs; at the first divergence it prints device-vs-host q, sink, window/selected ids,
compressed count/ratio and first/last window-code checksums. This is the next record; no claim
window is booked until the alignment and operand/residual twin are clean.

### Sequence alignment/residual diagnostic (bec6498, b57)

Window b57 ran bec6498 on the short 3-position command. The alignment table excludes a one-token
offset: hybrid and reference feed the same token at the same position (52479, 56659, 101226) and
compare the same positions; predictions still differ at p0 (320 vs 201) and p1 (16 vs 201) while p2
matches (16 vs 16). The residual twin slices the reference token-major row properly and reports
position 0 layer 0 bf16_mismatches=20335/20480 max_abs_diff=8.587394e-1, i.e. a real value
divergence from the first layer, not a shape artefact.

At that first divergence the operand pair has an identical sink (checksum 031684704ed44895 on both
sides) and identical ids (window [0], selected empty), while q differs (device first 0.164062500 /
checksum c55cad79e647c613 vs host first 0.194739088 / checksum 62266f458e1bc01b) and the staged
window-row code checksums differ (device 6446b4769b567b20 vs host-quantised 9d68aefc560275cc). The
b56 body twin already showed the device body is faithful on identical operands, so b57 localises the
divergence upstream of the body: projection/RoPE/window-latent staging, not the csa2 seq kernel,
not the indexer ids and not teacher-forcing alignment. b57 is a localisation diagnostic, not a
criterion attempt or a pass/fail of the claim.

The follow-up 841566b adds the stage-aligned projection twin: raw wq_a/wq_b/wkv device vs the host
attention_projection_probe on the reference pre-attention row; q compared at the BF16 stage the body
consumed with the expected BF16 delta; device window codes against engine_window_row(host window
row); pre-attention hidden; and a final-norm head-input twin with per-block max abs. The next record
is that run; no claim window is booked until it says whether the device path is faithful to the
FP8-native math or a projection stage is wrong.

### Unreviewed cut-over label (all sequence diagnostics)

Every sequence binary from fde6b11 through 420407e contains c9fa90f as an ancestor, and c9fa90f is
the c4c reduce cut-over (fifth head launch through lm_head_reduce_sm121, device token plus 4-byte
token/status readback, WINNER_CERTIFIED host-assembly fallback, REQUIRED_KERNELS 9->10). It has not
been reviewed or measured in a window; it is planned as batch 25. Therefore b55 and b56-b60 are
sequence diagnostics on an unreviewed cut-over build. b56's body twin still compares only the
attention body on staged operands, but the build it ran on is that cut-over, and the b57-b60
projection/head diagnostics inherit the same label. No claim may be inferred from any of them.

### Projection twin and wq_b kernel check (841566b / f32a9e5 / f228178, b58-b60)

b58 on 841566b: pre-attention hidden device vs host is 4/5120 BF16 mismatches with max_abs
2.831221e-7, so the layer input/MhcEntry agrees. Raw wq_a differs by 2.787181e-3 and raw wkv by
3.555462e-3, both consistent with the MXFP8 activation-scale delta. Raw wq_b q pre-RoPE differs by
2.942886e-1 and q staged BF16 differs by the same amount against an expected BF16 rounding delta of
6.176376e-2. Head-input twin mismatches 5113/5111/5104 of 5120 at p0/p1/p2 with max_abs 6.088e-1 /
6.534e-1 / 5.361e-1 and no single block; the final-norm/head input is not a clean uniform scale.

b59 on f32a9e5: wq_b input latent normed differs by 5.186009e-2; host wq_b on the device latent still
differs from the device wq_b by 2.080269e-1.

b60 on f228178: verbatim projection wq_b kernel vs mxfp8_ref: bf16_mismatches=0/32768
max_abs_diff=0.000000e0. The device wq_b kernel is exactly mxfp8_ref on the exact packed weight and
activation bytes it consumed. The projection path is faithful to the FP8-native math; the gap
against the host f32 reference is the MXFP8/BF16 quantisation gap. The projection investigation is
stopped here under the head-reduce lead; these are localisation records, not a claim.

### Head A/B against the unreviewed reduce (420407e)

The next diagnostic adds one head A/B line per head call: position N, host_top1, host_top5,
device_top1, device_top5, block_max_equal, host_value and device_value. The host side is the
four-shard block assembly from the copied block maxima/argmax (the same path as the WINNER_CERTIFIED
fallback); the device side reads the lm_head_reduce TOP buffers. Setting
SPARKFLASH_HEAD_HOST_ASSEMBLY=1 returns the host-assembly token. If the host assembly reproduces the
reference 201/201/16 while the device reduce gives 320/16/16, the unreviewed c9fa90f cut-over is the
bug; if not, the head-input/upstream path remains the lead. b60 is not a criterion attempt.

### Head A/B result: the reduce is not the bug (420407e, b61)

Window b61 ran 420407e plain and with SPARKFLASH_HEAD_HOST_ASSEMBLY=1. The host four-shard block
assembly and the device reduce agree at every position: p0 host_top1=320 device_top1=320,
p1 host_top1=16 device_top1=16, p2 host_top1=16 device_top1=16, with block_max_equal=true and
identical host_value/device_value at each position. Forcing the host assembly still returns
320/16/16, not the reference 201/201/16. The host and device top5 lists differ only because the host
side ranks one winner per 1024-entry block while the reduce ranks full logits; the winner and block
maximum are identical. The unreviewed c9fa90f reduce is therefore excluded as the cause.

b61's 120 attention body twin lines are zero-mismatch on 35/40 layers at position 0, 18/40 at
position 1 and 13/40 at position 2; the nonzero layers are the same BF16-stage-scale mismatches b56
reported (the body still matches the host body on identical staged operands). Together with b60's
wq_b kernel-vs-mxfp8_ref 0/32768, b58's exact pre-attention row and b56's body twin, the device path
is faithful to its FP8-native definition at every stage measured. The head input and logits
still differ from the in-process f32 reference because q and the window rows are FP8-quantised on the
device; the remaining gap is the criterion-definition question (f32 reference with a 2e-2 logits bar
against an FP8-native device path), not a projection, body, alignment or reduce bug. No claim.

### Criterion-definition evidence (b58-b61, not a claim)

The measured chain is faithful to the device's own FP8-native definition at every stage the
diagnostics can separate, while the criterion compares against an in-process f32 reference:

- pre-attention hidden: 4/5120 BF16 mismatches, max_abs 2.8e-7 (exact for practical purposes);
- raw wq_a: max_abs 2.8e-3; raw wkv: max_abs 3.6e-3 - the MXFP8 activation-scale delta;
- raw wq_b: max_abs 2.9e-1, and the wq_b kernel against mxfp8_ref on the exact launch bytes is
  0/32768 BF16 mismatches with max_abs 0.0 - the kernel is exactly the block-scaled FP8 GEMM;
- attention body on the staged operands: BF16-exact or within the stage LSB (b56/b61);
- head reduce and host four-shard block assembly: identical top1 and block maximum at p0/p1/p2,
  including when the host assembly is forced (320/16/16 on both, against reference 201/201/16);
- final-norm/head input: 5113/5112/5104 of 5120 BF16 mismatch, max_abs ~0.6, no single block.

The logit gap follows from the FP8-quantised q/window rows: the reference computes those layers in
f32, so the device logits are systematically flatter (p0 15.73 vs 19.85, p1 15.43 vs 21.25, p2 14.56
vs 16.97) and the argmax moves at p0/p1. A 2e-2 logits bar against the f32 reference is not reachable
by an FP8-native path; the decision is (a) define the reference with the same FP8 staging/quantisation
so the criterion measures wiring, or (b) add a higher-precision parity path for this comparison. No
mathematics should be 'fixed' before that decision; these are localisation records only.

### b62 criterion attempt: ABORTED by host memory, not a criterion result

b62 (420407e, 130 prompt + 8 decode, live reference vs device chain) was stopped by the operator at
131/138 positions: process RSS reached 121 GB and MemAvailable fell to 1.3 GB, so it was killed
before the host OOM killer could act. It is recorded as an aborted criterion attempt with a named
memory defect, not as a criterion result. The 131 head A/B lines that did complete agreed at every
position (device_top1 == host_top1, block_max_equal=true, identical values), so the head and reduce
remain cleared. The hybrid greedy picks on the synthetic prompt (for example position 120 -> 8060,
128 -> 13763, 130 -> 8060, 131 -> 16) do not look like the fixture
[491, 367, 201, 80, 3302, ...], but the run never reached the live-reference-vs-fixture table, so
the question does the LIVE reference reproduce the FIXTURE tokens is still open.

The growth was ~0.95 GB per position (126 GB available at start, 1.3 GB at position 131). The
attributed cause is the CachedWeightSource unbounded read-through cache of routed-expert tensors:
every position selects different experts, and caching every selected expert w1/w3/w2 raw bytes
retains the whole routed-expert set in RSS instead of letting the mmap page cache stay reclaimable.
The fix reads .ffn.experts.* through the mmap source without inserting it into the cache, strips the
per-layer/head buffers from outputs outside the diagnostic prefix and the two criterion positions,
and prints a per-position sequence memory line with MemAvailable and VmRSS before/after so the
next run attributes any remaining growth. A --reference-check mode prints the nine
reference vs fixture lines and the summary immediately after the reference prefill and returns
before opening any device backend, so the fixture-staleness question needs no 138-position loop.

### Reference identity and vLLM (criterion context)

The deployed vLLM is not serving our checkpoint. Its /v1/models root is
/root/.cache/huggingface/hub/models--pqhaz--DeepSeek-V4.1-Flash-1017-heretic/snapshots/12d58ad0878b5008e80d720251adb969a0069412,
while the fixture and engine use
models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170. Its
completions are also degenerate (the capital-of-France prompt returns filler), so no 'vLLM parity'
statement may be made unless it says 'same weights'; as deployed it is not the same weights and the
f32 fixture reference remains the only truth for our checkpoint.

Separately, a live vLLM query with the fixture's 130 prompt ids (greedy, 9 tokens) returned
[367, 223, 10727, 201, 15, 223, 19, 16, 223] against the fixture's
[491, 367, 201, 80, 3302, 201, 54, 201, 4341]. The fixture prompt is synthetic (seed 0), so this
does not decide the criterion, but 'vLLM parity' and 'f32-reference parity' are two different bars
and must not be conflated. The 130+8 run b62 checks whether the live in-process reference reproduces
the fixture's own reference tokens before the criterion discussion continues.

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

### Hybrid sequence path b55-b62: measured, no claim (criterion 1 still OPEN)

Status (2026-10-19): the device hybrid chain is **stage-faithful** but **not token-parity** with the
in-process f32 reference, and the first criterion-shaped run was **aborted on a memory defect**. No
claim is made here; criterion 1 remains OPEN and the hybrid-sequence branch stays unmerged.

**The builds.** Every sequence binary below (b55 `641462a`, b56 `fde6b11`, b57 `bec6498`, b58
`841566b`, b59 `f32a9e5`, b60 `f228178`, b61 `420407e`, b62 `420407e`) contains the unreviewed,
unmeasured head-reduce cut-over `c9fa90f` (`head_launches=5`, `reduce_used=true`,
`device_launches=1168` only when the reduce is used). Everything in this section is **sequence
diagnostics on an unreviewed cut-over build**, not a reviewed device result.

**b55 (641462a) - position plumbing and timing, 3 positions.** `sequence attention position support:
body_launches=120 expected=120 positions=3 device_position_0_body=40 other_position_bodies=80`. Per
position wall/chain: 16034/16490 ms, 8257/24748 ms, 7735/32484 ms; `projected_chain_full_s=1494.2`,
`full_positions=138`, `budget_s=3600`. Tokens diverge immediately: `token comparison: position=0
reference_id=201 hybrid_id=320 exact_equal=false`, position 1 is 201 vs 16, position 2 is 16 vs 16
(equal). The plumbing is right and the arithmetic is not.

**b56 (fde6b11) - body twin on identical staged operands.** 120 lines (40 layers x 3 positions);
zero-mismatch layers are **35/40 at position 0, 18/40 at position 1, 13/40 at position 2**, every
non-zero line at most 5 slots of 32,768 and max_abs <= 7.8e-3 (bf16 LSB class). The format-2 body is
cleared on equal operands.

**b57 (bec6498) - alignment, residual and operand.** Alignment is excluded: the fed token is
identical at all three positions (52479 / 56659 / 101226), and the pre-attention hidden is exact
(b58 line below). The residual twin still reports `residual first divergence: position=0 layer=0`,
and the operand lines show the divergence is already present: `operand q: device
first=0.164062500 last=0.773437500 checksum=c55cad79e647c613 | host first=0.194739088
last=0.759224892 checksum=62266f458e1bc01b`; the sink is identical, the ids agree on the window row
and differ in the compressed/ratio columns, and `operand window-codes: device ... first_checksum=
6446b4769b567b20 ... | host quantised first_checksum=9d68aefc560275cc`. Note the residual twin
compares the hybrid's deferred-epilogue boundary against the reference's post-FFN residual, i.e.
stages one apart; it is diagnostic only and b63 adds the stage-aligned twin.

**b58 (841566b) - projection and head-input twins.** `operand pre-attention: device vs host
bf16_mismatches=4/5120 max_abs_diff=2.831221e-7` (exact); `projection raw wq_a latent ...
max_abs_diff=2.787181e-3`; `projection raw wkv ... 3.555462e-3`; and `projection raw wq_b q
pre-RoPE: device vs host bf16_mismatches=29209/32768 max_abs_diff=2.942886e-1` against
`expected_bf16_quant_delta=6.176376e-2` (4.8x). The window-row quantiser is not byte-equal
(`byte_equal=false`, checksums above). Head input: `head-input twin: position=0 bf16_mismatches=
5113/5120 max_abs_diff=6.087970e-1` (positions 1 and 2: 6.53e-1, 5.36e-1).

**b59 (f32a9e5) - where the wq_b gap enters.** `projection wq_b input latent normed: device vs host
bf16_mismatches=1147/1280 max_abs_diff=5.186009e-2`, and feeding the device's latent through the
host wq_b still leaves `2.080269e-1` on q, so the input latent already carries most of the gap.

**b60 (f228178) - wq_b kernel vs its reference.** `projection wq_b kernel vs mxfp8_ref:
bf16_mismatches=0/32768 max_abs_diff=0.000000e0`. On the exact launch bytes the block-scaled FP8
GEMM reproduces `mxfp8_ref`, so the projection kernel is faithful to the FP8-native definition; the
gap to the f32 host path is the quantisation definition itself (MXFP8, 32-element blocks,
power-of-two ceil scales), not a kernel defect.

**b61 (420407e) - head A/B, plain and with `SPARKFLASH_HEAD_HOST_ASSEMBLY=1`.** At every position
the device reduce and the host four-shard assembly agree: `head A/B: position=0 host_top1=320 ...
device_top1=320 ... block_max_equal=true host_value=15.734617233 device_value=15.734617233`;
positions 1 and 2 are identical pairs (16/16, values 15.431822777 and 14.563224792). Forcing the
host assembly still returns `returned host assembly token=320` / `16` / `16`. The unreviewed
`c9fa90f` reduce is therefore exonerated; the host/device top-5 tails differ only because the host
path ranks one winner per 1,024-entry block while the reduce ranks full logits. b61 logs:
`runs61/headAB-device.log` sha256 `8f543d1c7f1135d0db9af820fa18139c421aedc01222365f4365898fc67d28cd`
and `runs61/headAB-host.log`.

**b62 (420407e) - criterion-shaped attempt: ABORTED, no criterion result.** One
`--run-stage hybrid_sequence --fixture <fixture> --prompt-tokens 130 --decode-steps 8
--time-budget-s 5400` run. It completed 131 of 138 positions, then the Lead killed it because host
memory was collapsing: `MemAvailable` fell from 102.9 GB at position 1 to 2.29 GB at position 128
and the process RSS reached **121 GB** (about 0.8 GB per position). Window log
`window-b62-20261019T120851Z.log`, run log `runs62/seq130x8.log`, `RUN seq130x8 exit=137`. The run
prints the reference-vs-fixture token table only at the end, so that table was lost; **the memory
defect blocks any 138-position run** and must be fixed before another attempt. The 131 head A/B
lines were identical device/host at every position.

**Where that leaves the criterion.** The separable stages pass: alignment, the pre-attention
hidden, the body, wq_b against `mxfp8_ref`, the head reduce and the host assembly are each at or
inside their measured bars. The token moves at position 0 (320 vs 201), the head input differs by
0.608, and b58-b60 attribute the difference to the FP8-native activation quantisation of the
attention path against a dequantised-weight f32 reference. On that arithmetic, exact 130+8 token
parity is not attainable without a precision-first activation path or a redefined criterion; that is
a user decision, not an implementer one, and no maths was changed. Criterion 1 stays OPEN and the
hybrid-sequence MR stays unmerged until that decision and the memory fix.


**b66 (4d12492) - fixture validity, memory fix and stage-aligned twins.** Three results.
(i) The fixture is **valid**: after the reference prefill the live in-process f32 reference
reproduces it exactly (`reference vs fixture: position=129 reference_token=491 fixture_token=491
equal=true` ... `position=137 ... 4341 ... equal=true`, `reference vs fixture: positions=9
all_equal=true`, reference wall_ms=614382), so the divergence is the device chain, not a stale
fixture.
(ii) The memory defect is **closed**: the routed-expert cache bypass keeps RSS at ~18.87 GB after
position 0 and the per-position deltas are zero or page-cache noise (`rss_delta_kib=0`).
(iii) The **stage-aligned twin** supersedes b57's boundary verdict: layer-0 post-attention is
`9627/20480 max_abs_diff=3.120759e-2` and layer-0 host FFN Y is `4936/5120
max_abs_diff=1.989301e-2`, so **layer 0 is not 0.5 off** and `residual first divergence: position=0
layer=0` in b57 was the deferred-epilogue stage-offset artifact, not a numerical finding. The (d)
layer-1-entry-vs-`residual_out(0)` line is Engram-contaminated (1.289867e0) because the provider
applies Engram before it returns the entry. The router kernel matches its own BF16 host exactly
(`scores_max_abs 5.96e-8`), layer 1 is only a tie-**order** swap (same six experts), and layer 14 is
a real expert-**set** flip (device `[166,98,35,26,304,227]` vs reference `[166,98,35,74,26,205]`,
`scores_max_abs=1.944607e-1`). So a discrete routing mechanism is real, but it starts at or before
layer 14; the first set-flip layer is still owed by the d923906 refinement.

**b67 (4d12492) - the first complete criterion-shaped prefill: measured FAIL, then the decode seam
stops it.** One 130+8 run over 131 of 138 positions: memory stayed flat (`sequence memory:` deltas
within page-cache noise, RSS 18.87 GB, `position=130 ... rss_delta_kib=11684`) and the fixture was
reconfirmed in the same run (`reference vs fixture: positions=9 all_equal=true`). The chain reached
position 130 in 1,177,425 ms (~9 s per position, `projected_chain_full_s=1240.3`); the attention
partial totals are `projection_launches=26200 body_launches=5240 staging_ms=18790
projection_us=69779207 body_us=1523009 staged_matrix_bytes=664113408000 host_bytes=1750529200`.
Two hard results:
- **Criterion 1 fails on this arithmetic at the first compared position.** Position 129 is
  `head A/B: position=129 host_top1=8060 host_top5=[8060:19.415479660, 67:17.912334442,
  16187:17.509645462, 3108:17.090551376, 71484:16.961109...]`, i.e. the device token is **8060**, not
  the fixture's **491**; position 130 agrees with itself (8060 again).
- **The decode seam cannot embed a generated token.** The run then aborts with `c4a-2 chain abort:
  hybrid sequence: sequence embedding: no staged embed row for token 8060`: the embedding staging
  only holds `prompt_ids` plus the reference decode tokens, so a device-generated decode token has no
  staged row and the 8 decode steps cannot run at all. This is a wiring defect, not memory and not
  numerics, and it must be fixed (an on-demand host row read for the fed token, which keeps the
  declared host embedding seam) before positions 130..137 can be measured.

Consequently the b55-b62 evidence is **not** "unmeasured": the prefill path is now measured end to
end, it is stage-faithful, and it diverges from an exactly-reproduced fixture. Remaining work is
ordered: (1) close the decode embedding seam so the 129..137 table exists; (2) the user's precision
versus criterion decision (D37); (3) the router refinement (first set-flip layer, BF16 control).

**b68 (e9e743e) - router-all refinement and the first complete 130+8 criterion run.** This binary adds
the on-demand embed-row fallback (`embed_row_from_source`) that b67 was missing, so the decode steps
can run. Its 3-position refinement first:
- `router twin: first_set_flip_layer=3`: layers 3, 7, 8, 9, 10, 11, 12, 13, 14, ... 39 (**36 of 40**
  at position 0) select a different expert set. `scores_max_abs` grows from 1.472694e-2 at layer 3 to
  1.311069e0 at layer 39, while the same-input host twin is equal at all 40 layers (<=5.96e-8). Layer 0
  keeps the reference set but already drifts by `scores_max_abs=1.649636e-2` - the seed the layer-3
  amplification acts on.
- **The BF16 control is decisive.** On every set-flip line `bf16_control_same_set=true` and
  `bf16_control_matches_device_set=false`: quantising the f32 reference's router input to BF16 alone
  never changes its expert set and never reproduces the device set. The trigger is therefore the
  accumulated attention-path drift in the router input - phrase this as **attention-path drift flips
  router selection at layer 3**, not as an FP8-router or BF16-staging effect. The router kernel itself
  matches its own BF16 host to 5.96e-8.
- The corrected non-Engram `(d2)` line is `3.213581e-2 / 3.433615e-2 / 5.166603e-2` at p0/p1/p2,
  which confirms the old (d) `1.289867e0` was the Engram contamination and that the deferred epilogue
  plus layer-1 Engram is not a 1e0-class error.
- Memory is flat again (position 0 RSS 600,276 -> 15,618,236 KiB, then -34,432 KiB, then 0).

The second run is the first **complete** criterion-shaped 130+8 attempt on the device chain: **all 138
positions finished** (`sequence position: position=137 done=138/138`, `chain_ms=1238987`,
`projected_chain_full_s=1239.0`, 9.1-9.6 s per position) with flat memory (RSS 18,928,152 KiB;
per-position deltas 0 or a few hundred KiB, plus one one-off +577 MB page-cache line at position 136)
and exit 1 only because parity fails. Criterion-1 evidence from that run:
`logits 129: position 129: 128528 of 129280 compared logits outside the 0.02 bar (max |difference|
15.009581 at 5986, argmax 8060)`; the device's nine tokens are **8060, 8060, 1168, 50673, 303, 531,
6525, 5719, 14171** against the fixture's **491, 367, 201, 80, 3302, 201, 54, 201, 4341** - **0 of 9**.
(That table is derived from the head A/B top-1 lines and the logits line because this build prints the
per-position token table only on the shorter run paths; the reporting gap is noted for the next build.)
Window `window-b68-20261019T133112Z.log`, `runs68/router-all-3.log`, `runs68/seq130x8.log`, binary
sha256 `ee916e3d65f154d512e93e3a734aaa878cd1d0e4e9dbe408f3f73ab35adaf59a`.

**Reference note.** The deployed vLLM on the cluster serves a different checkpoint from ours
([DECISIONS.md](../../DECISIONS.md) D38), so its tokens are not a parity target for this fixture.

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

