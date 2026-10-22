# Device chain

The device path is the product. It runs the 40-layer hybrid chain — attention, MoE, mHC seams and
the serving tail — on the GB10 GPUs, with the host loop kept only as the bit-exact reference. The
current plan of record is the **whole-layer device port** on `win/layer-device`: one decoder layer
runs entirely on device, with no steady-state D2H and the cross-rank all-reduce on the GPU.

## Where it lives

| Component | Location | Role |
| :-- | :-- | :-- |
| Device crate | `crates/sparkflash-device/` | Chain, window ring, cubins, paged KV, sessions, device engine. |
| Chain wiring | `sparkflash-device/src/chain.rs`, `window.rs`, `cubins.rs` | 40-layer plan, boundary stages, masked comparator. |
| Device layer | `crates/sparkflash-cli/src/layer_device.rs` | One device slab, 13 spans, the `LayerStages` trait and `run_layer` order. |
| Kernel ABI | `crates/sparkflash-kernels/` | Pinned cubins, launchers, CPU reference twins, `KERNEL_CONTRACTS`. |
| Host lean loop | `crates/sparkflash-cli/src/tp4_forward.rs` | Reference path and the source of the current 179.81 ms/step number. |

Dependency direction: `sparkflash-cli -> sparkflash-device -> {api, kernels, engine, kv, scheduler,
trace}`. No cycle; the chain is not inside `engine` (a leaf) or `kernels` (would pull the HTTP stack
into the ABI crate).

## Two arms

| Arm | Selection | Properties |
| :-- | :-- | :-- |
| Host reference | default | Bit-exact CPU/GPU lean loop; current published number. |
| Device layer | `SPARKFLASH_LAYER_DEVICE=1` | One layer device-resident; opt-in until the whole layer runs. A missing kernel refuses by name with its signature. |

There is no "run what is landed" mode: a half-device layer that silently mixes arms is the failure
mode the contract prevents. A refusal prints the stage, the missing kernel and the contract.

## The 20-stage layer

The stage order is the dependency order in `LayerStages::run_layer`. The contract is **20 stages**
(`ref_select` landed as its own stage at `60bc064`; it was 19 before).

| # | Stage | Kernel / operation | State |
| --: | :-- | :-- | :-- |
| 1 | `engram_write` | `engram_embed/wkv/gate_write_sm121` | layer 0 no-op; later layers unwritten |
| 2 | `mhc_entry` | `mhc_entry_bf16` (20-word ABI) | landed |
| 3 | `quantise_prologue` | `w8a8_prologue_e4m3` | landed |
| 4 | `attention_projections` | `b12x_mxfp8_gemv_m1_sm121` | landed |
| 5 | `query_path` | rmsnorm, `w8a8_prologue_bf16_e4m3`, wq_b GEMV, 16 rope launches | landed; rope launch count is a hoist candidate |
| 6 | `kv_path` | `kv_norm_rope_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16` | landed; declared bf16-latent widening (D54 item, 2 % bound) |
| 7 | `reference_seam` | `b12x_hostw_gemv_m1_sm121` + index writes | refuses: compressor weights, compressed-cache/index writes, `AttnState` residency |
| 8 | `ref_select` | `ref_select_sm121` (18 words) | line landed; kernel pinned; top-k over <= 16,384 candidates, ascending-id tie-break; body still unwired |
| 9 | `stage_attention` | native FP4 expanded store | refuses; expanded store is the chosen cache-native form |
| 10 | `attention_body` | `flash_mla_csa2_seq_sm121` | wiring not started; ring pointer, `window_rows = min(position,128)`, `selected_count` |
| 11 | `attention_output` | inverse rope (new), `b12x_bf16a_mxfp8w_wo_a_sm121`, `w8a8_prologue_bf16_e4m3`, GEMV | not started; inverse rope is the only new kernel |
| 12 | `all_reduce` | `comm_all_reduce_bf16` (NCCL) | refuses until bootstrap; one `RunReduce` per run |
| 13 | `attention_residual` | `mhc_attn_residual` | landed |
| 14 | `ffn_seams` | `mhc_ffn_seams` | landed; `ffn_x_bf16` out row in flight |
| 15 | `moe_router` | `moe_router_sm121`, `moe_router_topk_sm121` | not started; needs the un-normalised bf16 pre-FFN row |
| 16 | `moe_routed` | `b12x_mxfp4_moe_stored_sm121` + 3 | plumbing landed; per-expert loop, f32 delta -> bf16 pre-reduce |
| 17 | `all_reduce_moe` | `comm_all_reduce_bf16` | refuses; same handle/sequence as #12 |
| 18 | `all_reduce_rdma_arm` | GPU-initiated RDMA | typed refusal unless `gpu-rdma-reduce` is built |
| 19 | `moe_shared` | 3 GEMVs + `shared_swiglu_f32` / coded arm | landed, both arms |
| 20 | `ffn_epilogue` | `mhc_ffn_epilogue` | landed; `shared_present=0` until #19's row is declared |

**Frontier: `reference_seam` by stage order.** The first device-arm probe row is a frontier probe,
not a speed row; if it completes a layer it must pass the numerical gate. `ref_select` and
`compressed_append` are pinned but their stage bodies are unwired; the remaining new kernel is the
inverse rope in `attention_output`.

## Memory and ownership invariants

`layer_device.rs` carves one device slab into 13 spans with an ownership table. The steady-state
rules, which are acceptance conditions:

1. **Residual never leaves the device inside the loop.** No stage allocates; no stage returns bytes.
2. **Evidence gates are off in the steady state.** The per-layer `partial_l2`/`reduced_l2` and
   gate/up/codes/scales captures are the source of the old 12 D2H points; a debug flag must not
   silently restore them.
3. **The MoE combine consumes the reduce's device output.** Host-side combine re-crosses PCIe.
4. **Trace/boundary/parity capture is dev-gated.** The steady state records counters.
5. **One legitimate readback:** final logits once per position for the numerical gate, plus per-run
   counters.
6. **Collective-sequence coherence:** every rank enqueues the same collectives in the same order.
   A rank-conditional path that skips a reduce hangs NCCL.
7. **Payload frozen:** bf16, 5,120 elements (10,240 B) per rank for each of the two per-layer
   reduces. f32 partials (20,480 B) are a different arm and must be re-declared.

Key spans are sized in `layer_device.rs` (residual, `hc_rows`, `attn_out_bf16`, `reduced_bf16`,
`attn_mix`, `ffn_x_bf16`, `seam_row`, `moe_ids`, `moe_weights`, `logits`, `moe_reduced_bf16`).
The FFN path reads `ffn_x_bf16` (RNE-narrowed pre-FFN row), not a re-norm of `seam_row`; the
router reads the un-normalised row.

## All-reduce contract

- **Two collectives per layer:** attention delta and MoE routed delta, each 5,120 bf16 elements.
- **One communicator per run**, one `RunReduce` owning the `NcclApi` and `ncclComm_t`.
- **Bootstrap:** rank 0 generates the 128-byte NCCL unique id; the loop broadcasts it over its
  control channel; all ranks pass the same bytes to `open`. A missing libnccl is a typed
  `LibraryAbsent`, never a silent fallback.
- **Warm-up:** 2-3 dummy reduces plus one stream sync at setup, outside the step. Layer 0 is not
  steady state.
- **Async:** enqueue on the caller's stream, no completion query, no per-collective sync. The only
  sync is the step boundary; async failure is checked at the run boundary.
- **D54 tolerance arm:** NCCL bf16 Sum (one f32 accumulation per element, one bf16 round) is a
  declared tolerance arm, measured by differing elements and max ULP; it is not the canonical rank
  order. The row must pass the numerical gate.
- **RDMA arm:** same `DeviceReduce` API, `--reduce-arm nccl|gpu-rdma`, typed refusal if
  `nvidia-peermem` is absent. Default is NCCL until a fabric row says otherwise.

## KV and attention semantics

- The host `kv_path` writes a host-resident cache; the device arm owns its own cache. **No shared
  mutable cache between arms.**
- Device KV is a three-kernel split: `kv_append_bf16` (ownership move), `window_quant_fp8_bf16`
  (round-trip), `rope_rotate_bf16`. The interface is frozen as `DeviceKvCache`; a
  half-finished cache-ownership change is the failure mode the split prevents.
- The attention body consumes the window ring: `window_rows = min(position, 128)`, the selected
  compressed ids, causal order, sink first. Assert `ring_rows >= max_positions` and
  `position < ring_rows`.
- The KV ring has 134 slots while the attention window is 128 (D49); the new symbol carries the
  capacity explicitly and a NaN guard on accept.
- The host `kv_path` contract (fp8 round-trip, append/visibility, window slice arithmetic) is the
  behavior the device port must reproduce; the seam itself does not read the cache.

## Gates and evidence

| Command | Checks |
| :-- | :-- |
| `scripts/device-frontier.py --row-dir <row>` | Stage frontier / open refusal; exit 3 is a valid refusal, not a number. |
| `scripts/numerical-gate.sh --run <row>` | D54 bars against the frozen CPU-twin reference. |
| `scripts/check-collective-count.py` | All four rank records agree: `layers x 2` collectives. |
| `scripts/check-kernel-baseline.sh` | Pinned cubin baseline; required before staging. |
| `scripts/identity-gate.sh --run <row>` | Strict byte identity, only when the slice claims it. |

A device-arm row that fails the numerical gate is a port bug, not a tolerance question. Do not
publish ms/step from `win/layer-device` until one full layer runs device-only and the branch is
measured as a whole.

## Open items

- Wire: `ref_select`, compressed-store append, `moe_router` body, `attention_body` /
  `attention_output`, `reference_seam` and `stage_attention` bodies; write the inverse-rope kernel.
- Bring up NCCL bootstrap and the two per-layer reduces.
- Finish `AttnState` residency and the device KV ownership move.
- Flip the default to the device arm (one commit) after one full device-only layer.
- Then: CUDA graph capture and the GEMV re-tile.

## Archived sources

`device-forward-chain.md`, `device-path-design.md`, `ws-whole-layer-device-port-20261022.md`,
`ws-attn-state-residency-20261022.md`, `ws-kv-path-semantics-20261022.md`,
`attention-sequence-contract-20261022.md`, `layer-device-launch-inventory-20261022.md`,
`layer-device-dtype-audit-20261022.md`, `ws-seam-to-device-plan-20261022.md`,
`ws-device-arm-numerical-gate-20261022.md`, `ws-device-arm-row-readiness-20261022.md`,
`ws-run-pld1-probe1-20261022.md`, `ws-layer-stage-scorecard-20261022.md`,
`head-stage-loop.md`, `head-tail-contract.md`, `moe-stage-contract.md`, `m25-stored-chain-abi.md`,
`engine-tp-seams.md`, `engine-kv-seam-decision.md`, `phase2-device-kv-store.md`,
`gpu-device-roadmap.md`, `roadmap-to-serving.md`, `serving-readiness.md`, `phase1-boundary-table.md`.
