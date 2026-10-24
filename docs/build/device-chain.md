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
| Host reference | `SPARKFLASH_LAYER_DEVICE=0` (or `host`); forced by evidence recording | Bit-exact CPU/GPU lean loop and the numerical gate's host baseline. Needs `SPARK_COMM_REDUCE=tcp` or `roce`. |
| Device layer | **the default since S2** (unset, `1` or `device`) | All 40 layers, 20 stages each, device-only in one decode step since S1 closed ([`s1-result.md`](s1-result.md) §8). Needs `SPARK_COMM_REDUCE=nccl`. A missing kernel still refuses by name with its signature; a switch spelling that names neither arm refuses at startup. |

There is no "run what is landed" mode: a half-device layer that silently mixes arms is the failure
mode the contract prevents. A refusal prints the stage, the missing kernel and the contract.

## The 20-stage layer

The stage order is the dependency order in `LayerStages::run_layer`. The contract is **20 stages**
(`ref_select` landed as its own stage at `60bc064`; it was 19 before).

| # | Stage | Kernel / operation | State |
| --: | :-- | :-- | :-- |
| 1 | `engram_write` | `engram_write_bf16` + the two stream conversions | layer 0 no-op wired and **ran**; later layers still refuse (at-open row arena + 6-launch chain unwired) |
| 2 | `mhc_entry` | `mhc_entry_bf16` (20-word ABI) | landed and **ran** on layer 0 |
| 3 | `quantise_prologue` | `w8a8_prologue_e4m3` | landed and **ran** |
| 4 | `attention_projections` | `b12x_mxfp8_gemv_m1_sm121` | landed and **ran** |
| 5 | `query_path` | rmsnorm, `w8a8_prologue_bf16_e4m3`, wq_b GEMV, 16 rope launches | landed and **ran**; rope launch count is a hoist candidate |
| 6 | `kv_path` | `kv_norm_rope_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16` | landed and **ran**; declared bf16-latent widening (D54 item, 2 % bound) |
| 7 | `reference_seam` | `b12x_hostw_gemv_m1_sm121` + index writes | **ratio 0** wired as window metadata (no launch) and **ran**; ratio>0 refuses (compressor/index host-seam class) |
| 8 | `ref_select` | `ref_select_sm121` (18 words) | **ratio 0** identity no-op wired and **ran**; ratio>0 unwired; kernel pinned, top-k over <= 16,384 candidates, ascending-id tie-break |
| 9 | `stage_attention` | native FP4 expanded store | **ratio 0** validation-only wired and **ran**; ratio>0 refuses (expanded store unwired) |
| 10 | `attention_body` | `flash_mla_csa2_seq_sm121` | **refuses: the frontier.** Launch exists and the query row is device-resident; the per-head sink and indexer spans are not allocated on the arm. 19-word builder + inverse-rope builder landed and tested |
| 11 | `attention_output` | inverse rope (new), `b12x_bf16a_mxfp8w_wo_a_sm121`, `w8a8_prologue_bf16_e4m3`, GEMV | refuses; 8/8/11-word builders landed, wiring pending |
| 12 | `all_reduce` | `comm_all_reduce_bf16` (NCCL) | enqueue wired; not reached in a step. The 2026-10-23 run's 80/step walk is the **TCP reference skeleton**, not this stage; the NCCL arm's only four-rank run is an INVALID unlocked 10 KiB bench |
| 13 | `attention_residual` | `mhc_attn_residual` | landed; not reached in a step |
| 14 | `ffn_seams` | `mhc_ffn_seams` | landed; writes the `ffn_x_bf16` row the router reads; feeds #15's probe guard |
| 15 | `moe_router` | `moe_router_sm121`, `moe_router_topk_sm121` | wired, guarded by the `ffn_seams` probe count; not reached in a step |
| 16 | `moe_routed` | `b12x_mxfp4_moe_stored_sm121` + 3 | device path (staging table + ordered combine, 8 launches/0 copies) landed and pinned; stage body refuses |
| 17 | `all_reduce_moe` | `comm_all_reduce_bf16` | wired; same handle/sequence as #12 |
| 18 | `moe_shared` | 3 GEMVs + `shared_swiglu_f32` / coded arm | landed, both arms; not reached in a step |
| 19 | `all_reduce_rdma_arm` | GPU-initiated RDMA | typed refusal unless `gpu-rdma-reduce` is built; never called by the NCCL arm |
| 20 | `ffn_epilogue` | `mhc_ffn_epilogue` | landed |

**Frontier: `attention_body` (stage 10), measured on hardware on 2026-10-23.** The S1-0 probe ran the
frozen nine-position argv on four ranks with `SPARKFLASH_LAYER_DEVICE=1`: the arm opened on 4/4 ranks,
`mhc_seam` bound `400 expected = 400 bound`, the arena bind was complete (attention 208/208 with 19
declared host seams, census 95,818 resident, per-token weight H2D 0 by construction), the 40-layer
reference walk over TCP executed its 80 reduces/step (the NCCL arm's enqueue is never reached: the
stop is at stage 10, before `all_reduce` at stage 12), and layer 0 position 0 ran stages 1-9 before the
named refusal at stage 10. All four
ranks exited rc=2 (documented refusal). No step completed, so there is no ms/step and no D54 row;
`docs/build/s1-result.md` is the record and the raw evidence is archived at
`docs/archive/build/s1-device-probe-20261023/`.

**Superseded the same day: S1 closed.** All 20 stages now run on all 40 layers, plus the device
head, in one decode step: `d2h=1 / alloc=0 / sync=0`, D56 gate PASS, 202.67 ms/step four-rank mean
at 1500 MHz ([`s1-result.md`](s1-result.md) §8). The State column above and the frontier paragraph
record the S1-0 probe. S2 made the device arm the default. The capture-safety audit of each stage's
host arguments is [`s2-extent-args.md`](s2-extent-args.md).

The first device-arm probe row is a frontier probe, not a speed row; if it completes a layer it must
pass the numerical gate. The remaining new kernel for the attention half is the inverse rope in
`attention_output`; the bodies to wire next are `attention_body`, `attention_output`, the Engram
6-launch chain and `moe_routed`, then the device head (the arm cannot report `d2h = 1` until the host
head stops consuming the residual).

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
| `scripts/s2-verdict.py replicates\|soak --run <row>` | S2 header (device / nccl / build), invariants, registered spans, NCCL faults; replicate agreement or soak tail. |

A device-arm row that fails the numerical gate is a port bug, not a tolerance question. Do not
publish ms/step from `win/layer-device` until one full layer runs device-only and the branch is
measured as a whole.

S1-0 ran on 2026-10-23 with the frozen nine-position argv (four ranks, cross-built `aarch64` binary,
30/30 cubin pins, NCCL 2.31.2); its outputs are in `docs/build/s1-result.md` and the archived evidence
bundle. The run's only numbers are open/bind figures: the forward stops at layer 0 position 0, so it
is not a speed row and carries no tolerance arm.

## Open items

- Wire: `attention_body` (sink/index spans), `attention_output` (inverse rope + wo_a/wo_b), the Engram
  at-open row arena + 6-launch chain, and `moe_routed`; then the device head so a completed step can
  report `d2h = 1`.
- Ratio>0 `reference_seam`/`ref_select`/`stage_attention`: move the 19 `AttentionHostSeam` operands
  into the bind and port the compressor/index path (19 tensors, no arena today).
- An S1-a mixed-arm loop (the arm is chosen once per run today) and the frozen 32×128 D54 fixture.
- The clock-locked 10 KiB collectives gate (S2-a): `scripts/gpu-reduce-bench.sh gate` is implemented
  (1500 MHz lock, GDR evidence from NCCL's INFO log, registered buffers) and has **no locked row**.
  The `s2w01` boost run FAILs it on latency (p50 48.3–48.5 us, p99 85.2–87.3 us) and GDR. GB10
  reports neither GPUDirect RDMA nor DMA-BUF, so the GDR clause needs a Lead decision.
- The default flip landed in S2: the device arm is the default and `SPARK_COMM_REDUCE` is required.
  In window `s2w01`, at boost clocks, the S2-d pair (146.03 / 145.28 ms, numerical gate PASS) and
  the S2-e soak (PASS) ran. The **1500 MHz-locked** S2-d pair is still owed; see
  [`s2-result.md`](s2-result.md).
- Then: CUDA graph capture (S3, starting from [`s2-extent-args.md`](s2-extent-args.md)) and the
  GEMV / MoE re-tile (S4).

## Archived sources

`device-forward-chain.md`, `device-path-design.md`, `ws-whole-layer-device-port-20261022.md`,
`ws-attn-state-residency-20261022.md`, `ws-kv-path-semantics-20261022.md`,
`attention-sequence-contract-20261022.md`, `layer-device-launch-inventory-20261022.md`,
`layer-device-dtype-audit-20261022.md`, `ws-seam-to-device-plan-20261022.md`,
`ws-device-arm-numerical-gate-20261022.md`, `ws-device-arm-row-readiness-20261022.md`,
`ws-run-pld1-probe1-20261022.md`, `ws-layer-stage-scorecard-20261022.md`,
`head-stage-loop.md`, `head-tail-contract.md`, `moe-stage-contract.md`, `m25-stored-chain-abi.md`,
`engine-tp-seams.md`, `engine-kv-seam-decision.md`, `phase2-device-kv-store.md`,
`gpu-device-roadmap.md`, `roadmap-to-serving.md`, `serving-readiness.md`, `phase1-boundary-table.md`,
plus the S1 device-port window records `s1-p1-engram-wiring.md`, `s1-p2-attention-wiring.md`,
`s1-p3-moe-wiring.md`, `s1-p4-collectives.md`, `s1-qa-harness.md` and the 2026-10-23 probe bundle
`s1-device-probe-20261023/`.
