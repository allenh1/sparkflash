# The four-node fabric: transports, topology, and the all-reduce contract

Current-state handbook for the RDMA/collective layer on the four DGX Spark nodes (two
200 Gb/s RoCEv2 rails each, one 200 Gb/s-class switch). Process rules live in
`docs/RULES.md`. Every number carries a label: **MEASURED** (a run exists), **ESTIMATE**
(arithmetic), **DESIGN** (target or contract, never observed).

## 1. Current state

| item | state | label |
| --- | --- | --- |
| host-staged one-shot RDMA all-reduce | implemented, measured on 4 nodes, **not** wired into the forward row | MEASURED |
| canonical host reduce over TCP | implemented; the reference semantics and the fallback | MEASURED |
| NCCL 2.28+ binding (dlopen) | implemented; no fabric run of the NCCL arm exists | DESIGN |
| device-buffer seam (`all_reduce_bf16_device`) | `CommError::Unsupported` stub | DESIGN |
| 25 us p50 decode all-reduce gate | never met on this fabric | MEASURED (missed) |
| MTU 9000 on hosts | applied and persisted 2026-10-21 | MEASURED |
| `nvidia-peermem` / GPUDirect | module present, driver rejects it | MEASURED |
| forward-row collective | TCP by default; device arm opt-in via `SPARKFLASH_LAYER_DEVICE=1` | MEASURED |

## 2. Transports

### 2.1 Host-staged one-shot RDMA

`crates/sparkflash-comm/src/rdma/` (`abi.rs`, `verbs.rs`, `qpinfo.rs`, `dataplane.rs`),
compiled only with the `rdma` feature. The default build gains no dependency: libibverbs is
opened with `libc::dlopen` (`libibverbs.so.1`, then `.so`), 16 entry points resolved by
`dlsym`. `ibv_post_send`/`ibv_post_recv`/`ibv_poll_cq` are `static inline` in rdma-core and
not exported, so they compile into a bundled C shim (`rdma/ibverbs_shim.c`) by `build.rs`
against the real header; a build host without that header gets ENOSYS stubs and
`Verbs::load` refuses with a typed error rather than opening a device it cannot post to.

| contract | value |
| --- | --- |
| ABI | hand-written `repr(C)` mirror of rdma-core 39, pinned by compile-time size/offset asserts (`ibv_send_wr` 128 B, `ibv_qp_attr` 144 B, `ibv_port_attr` 52 B, `ibv_wc` 48 B) |
| QP | one RC QP per (rail, peer); PSN 1 both directions; `min_rnr_timer` 12, `retry_cnt`/`rnr_retry` 7, timeout 14 |
| data plane | one-sided RDMA WRITE_WITH_IMM; no read, no atomics |
| staging | one MR per rail over a `Box<[u8]>`; `2 * world` inbox + `2 * world` send slots |
| slot size | `24 + max_frame_bytes` B (frame header + the whole frame) |
| 4-rank / 2 MiB geometry | 2,097,176 B per slot; 33,554,816 B per rail |
| rail split | by whole bf16 elements, rail 0 takes `elements - elements/2`; no element straddles |
| control plane | one TCP star per collective on `SPARK_COMM_CONTROL_PORT` (18515); 12 B header: magic `0x53515049`, version 1, kind 1, reserved, body length, world, count |
| errors | `InvalidLayout`, `RankOutOfRange`, `FabricUnavailable`, `Unsupported`, `Verbs`, `Protocol`, `Timeout`, `PeerClosed`, `Io` |

The control record carries the geometry (frame cap, slot bytes, rail count) so the decoder
refuses a cluster whose ranks disagree; a record for an unopened rail must be all zeros.
Posted bytes are accounted per rail in `RailLedger`/`CommTelemetry`, reset per collective, so
a counter gate reads exactly one dispatch.

### 2.2 NCCL (Path N, >2 MiB)

`crates/sparkflash-comm/src/nccl.rs`: runtime `dlopen` of `libnccl.so.2` then `libnccl.so`,
no build-time dependency. Version floor `22_800` (2.28.0; decoded
`major*10000 + minor*100 + patch`). The image ships 2.28.3; the host has none. Pinned ABI:
`ncclUniqueId` 128 B, `ncclBfloat16 = 9`, `ncclSum = 0`; `ncclCommInitRankConfig` is called
with a NULL config pointer so no versioned struct layout is depended on. NCCL is **not**
bit-identical to the canonical path and is admissible only for chunked prefill >2 MiB.

### 2.3 Host reduce (reference and fallback)

`exchange::all_reduce` + `reduce::canonical_all_reduce_into` over `TcpTransport`; the RDMA
and loopback paths call the same driver. A four-rank unroll writes the same four additions
in the same order with the pane index hoisted out, and the crate does not enable fast math,
so no reassociation is possible. Frames <= 256 KiB are read sequentially after the writes
and written as one buffer — that (not arithmetic) is what removed the 3-5x handoff cost
(section 5.2). `SPARK_COMM_PROFILE=1` records per-call `prepare/send/collect/reduce/total`
and `peer_wait`; disabled, it costs one relaxed atomic load per call.

### 2.4 Device reduce (the port's arm)

`crates/sparkflash-comm/src/device_reduce.rs` owns `RunReduce` (the `NcclApi` plus the raw
`ncclComm_t`), `DeviceReduce`, `DeviceReduceStats` and `ReduceArm`. Two arms: `nccl`
(default) and `gpu-rdma` (cargo feature `gpu-rdma-reduce`, default off, unbuildable until
peermem loads). Same type and signatures; the transport is chosen once at init, and an
unavailable arm is a **typed refusal**, never a silent fallback.

## 3. Topology and device selection

- The specification names the **f0** functions (`rocep1s0f0`, `roceP2p1s0f0`); they are
  `1: DOWN` on all four nodes. Traffic runs on the **f1** twins (`rocep1s0f1`,
  `roceP2p1s0f1`), `4: ACTIVE` at 200 Gb/s, `active_mtu` 1024 before the MTU change and
  4096 after.
- The resolver enumerates sysfs and requires ACTIVE and >= 200 Gb/s with the two rails on
  distinct cards (D12). A DOWN or 40 Gb/s device is a hard startup error, not a warning.
  `SPARKFLASH_RAIL_DEVICES` pins a different pair and is validated the same way.
- Both rails share `192.0.2.0/24`; each node holds one address per rail (rail 1
  `.80/.84/.59/.64`, rail 2 `.85/.88/.63/.68` for node0/node1/node3/node2). The peer table
  only has to be locally bindable per rank with a reachable rank 0; the data plane uses
  RoCE GIDs, not these addresses. The table that worked on 2026-10-20 is mixed rail-1/rail-2
  in rank order `.80, .88, .59, .68` — reuse it for any run on this cluster.
- `SPARK_COMM_RAIL_COUNT=2` (default) splits every frame across both rails; `=1` puts the
  whole frame on rail 0 and writes the rail-1 record as an all-zero placeholder. In
  single-rail mode the counter gate asserts rail 1 == 0 and prints that the run is not
  dual-rail evidence.

Dual rail is **our** gate requirement (aggregate >= 180 Gbps, QUALIFICATION 3.3), not a
description of the deployed reference: vLLM's recipe sets `NCCL_IB_HCA=rocep1s0f1`,
`NCCL_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_IB_GID_INDEX=3` — one rail (D29). No document may
cite the deployed container as dual-rail evidence.

## 4. Exactness contract

For one element (hard constraint, from the code): every rank sends its whole row to each
peer (one-shot all-to-all); frames are placed by **sender rank**, never by arrival; the
accumulator is f32 seeded `+0.0` and added in **ascending rank order**; one bf16
round-to-nearest-even at the end (`f32_to_bf16_rn`). The `+0.0` seed preserves the
all-negative-zero case.

- **The 2 MiB cap is inclusive**: 2,097,152 B routes one-shot, 2,097,153 B does not
  (`ROCENANTE_MAX_BYTES`); `DECODE_ONESHOT_MAX_BYTES` is 360 KiB for the decode range.
- **Never fall back silently.** `DispatchPolicy::route_with` returns a typed error naming
  the missing backend; a TCP or CPU all-reduce run "instead" is a different, unmeasured,
  differently-rounded collective.
- **f32 <-> bf16 is part of the ABI.** The transport reduces u16 bit patterns; truncating
  with `>> 16` changes the input bits and the canonical reduction cannot repair it. Use RNE.
- **Determinism is a per-route property**: `CollectiveBackend::is_bit_identical()` answers
  false for NCCL on purpose. A >2 MiB seam on NCCL must be declared non-bit-identical (D54
  bars) or chunked into <= 2 MiB one-shot pieces with a pinned chunk order — never mixed
  within one token row.

## 5. Measured numbers

### 5.1 One-shot all-reduce, 10,240 B (5,120 bf16 per rank), 4 ranks

| condition | p50 | other | label |
| --- | --- | --- | --- |
| dual rail, MTU 9000, host buffers, quiet nodes | **35.70 us** (per-rank 35.68-35.80) | p90 37.04 us | MEASURED (2026-10-21) |
| dual rail, MTU 1500, same probe | 36.97 us | p90 1634.09 us (unstable tail) | MEASURED (2026-10-19) |
| single rail, MTU 9000 | **28.70 us** (28.67-28.78) | — | MEASURED (2026-10-21) |
| single rail, MTU 1500 | 29.54 us | p99 2.09 ms | MEASURED (2026-10-19) |
| dual rail, bring-up | 36.0-37.1 us | min 29.2-29.6 us; p99 2.1-4.1 ms | MEASURED (2026-10-20) |
| single rail, bring-up | 29.5 us | min 25.6 us | MEASURED (2026-10-20) |

The 2026-10-21 rows supersede the 2026-10-20 rows for p50 after the MTU change; they differ
by ~3%, and the older row's 25.6 us single-rail minimum is still the lowest time on record.
Gate context: WS-B acceptance is p50 <= **25 us**; the doctor uses 10 KiB with
`MAX_ONESHOT_AR_LATENCY_US = 22.0` (strictly less). Both are missed, and the gap is not
wire-bound.

Tail caveat, from two runs of the same probe: p99 is not a stable fabric property. Runs with
the same p50 (36.73 / 29.18 us) had p99 3.28 ms and 38 us. Quote p50; treat a single p99 as
tail evidence only.

### 5.2 2 MiB, byte accounting, and the host reduce

| item | value | label |
| --- | --- | --- |
| 2 MiB p50 dual rail, MTU 9000 | 6787.03 us (bar 188.7 us = 1.5x the 125.8 us wire floor) | MEASURED (2026-10-21) |
| 2 MiB p50 dual rail, MTU 1500 | 8364.89 us; posted 6.02 Gb/s = 1.5% of 400 Gb/s | MEASURED (2026-10-19) |
| 2 MiB posted bandwidth, MTU 9000 | 7.42 Gb/s = 1.9% of dual-rail capacity | MEASURED (2026-10-21) |
| dual-rail accounting, 1 MiB | rail0 1,572,936 B / rail1 1,572,864 B, total 3,145,800 B as expected | MEASURED (2026-10-20) |
| host reduce per call, local 4-process, after | **113 us** profiled, 120-126 us unprofiled | MEASURED |
| same, before (3 reader threads per call) | 381-566 us (two runs) | MEASURED |
| host reduce async split | 130 us (post 56-78, wait 59-73) | MEASURED |
| row cost, no-sync probe | 0.974 ms/layer | MEASURED |
| row cost, drained probe | 0.295 ms/layer | MEASURED |
| rank skew at 300 us stagger | rank 0 total 1039 us, rank 3 total 146 us | MEASURED |

The 0.974 / 0.295 split is **rank skew, not arithmetic**: the no-sync probe removes the
per-layer device sync, ranks drift, and every rank but the last pays the drift inside
`collect`; the drained probe's sync boundaries align ranks and expose the real work. With
`--stagger-us 300` the per-rank p50 differs by exactly the stagger. A large `peer_wait_us`
on the real nodes is the drift signature, not transport slowness.

Arithmetic from the per-call numbers (ESTIMATE, 40 layers x 2 calls/layer): 77.9 ms/step at
0.974 ms/call, 23.6 at 0.295 ms, 9.0 at 113 us, 2.9 at 35.7 us. Host machinery in the lean
loop is 29.5 ms/step; the published lean-loop row stays 179.81 ms/step.

## 6. Infra state

| item | state | label |
| --- | --- | --- |
| host netdev MTU | 9000 on 16/16 fabric netdevs (including the DOWN f0 twins), persisted in `/etc/netplan/40-cx7.yaml` | MEASURED (2026-10-21) |
| RDMA `active_mtu` | 4096 on 8/8 active f1 devices (`max_mtu` 4096; 9000 is not a valid RoCE MTU) | MEASURED |
| DF-8972 | 24/24 directed same-rail pairs pass | MEASURED (2026-10-21) |
| switch | MikroTik CRS812-8DS-2DQ-2DDQ, RouterOS 7.20.7; four server ports already `mtu 9000`/`l2mtu 9216`; bridge `actual-mtu` still 1500, not a gate | MEASURED |
| `nvidia-peermem` | present (`.../updates/dkms/nvidia-peermem.ko.zst`, 580.178.04), not loaded; `modprobe` fails EINVAL, `MODULE_INIT_IGNORE_VERMAGIC` returns ENOEXEC (stale symbol CRCs) — needs a rebuild against the running kernel as root | MEASURED |
| host `libnccl` | 0 hits in `ldconfig -p`; 2.28.3 only in the image | MEASURED |
| rails | f1 ACTIVE 200 Gb/s both; f0 DOWN on all four | MEASURED |
| driver / kernel | 580.178.04 / 7.0.0-1019-nvidia | MEASURED |

The MTU change is an enabler, not a decode win: the measured row moved 350.39 -> 351.56
ms/step (+1.17 ms) and the 10 KiB p50 moved ~1.0-1.3 us. Do not expect MTU work to close the
25 us gate; GPUDirect would have to come from the dma-buf path on this rig.

## 7. All-reduce contract for the whole-layer port

Both layer collectives have the same frozen shape: **5,120 bf16 elements = 10,240 B per
rank**. Moving the delta row to f32 partials (20,480 B) is a different arm and must be
re-declared and re-measured.

- **Two collectives per layer, not one**: the attention delta and the MoE routed delta. The
  MoE one writes a **separate** span (`workspace.moe_reduced_bf16`, `dim * 2`); reusing
  `reduced_bf16` aliases the residual consumer.
- **Call shape** is a host call that enqueues on a stream:
  `comm_all_reduce_bf16(stream, in, out, count, handle)` — `count` in **elements**, never
  bytes; device addresses; opaque `void*` handle; layer-owned spans; no per-call workspace.
  `device_all_reduce_bf16` names the GPU-RDMA arm's kernel contract only.
- **Bootstrap belongs to the loop**: `open(world, rank, unique_id)` takes the 128 opaque
  bytes as an argument; rank 0 generates them and broadcasts over the control channel the
  loop already owns. `open` loads the library itself, so a missing libnccl is a typed
  `LibraryAbsent` and a null handle is `Config`, never a fallback.
- **Warm-up at setup**: 2-3 dummy reduces plus one sync, outside the step, at the real
  payload count; layer 0 is therefore not steady state. Warm-up collectives are not in
  `DeviceReduceStats`, so counters read exactly layers x collectives.
- **Enqueue-only, no per-collective sync or completion query**: ordering is the stream's job,
  the only sync is the step boundary, and `ncclCommGetAsyncError` is checked once at the run
  boundary (failure detection, not completion).
- **Arena**: `DeviceReduceStats.arena_bytes` recorded at open (default
  `NCCL_ARENA_RESERVE_BYTES` = 16 MiB) and added to the arm's cost, so the memory model and
  the reduce record cannot drift. `workspace.reduce_ws` (40 KiB) sizes the RDMA tree only.
- **The host arm must not acquire the communicator.** Gate the bootstrap on the *full* arm
  decision, not the opt-in predicate: evidence recording forces the host arm, and a
  half-decision gate gives the 179.81 ms/step arm a libnccl dependency and a new setup cost.
- **Collective-sequence coherence**: every rank enqueues the same collectives in the same
  order; no rank-conditional path may skip a reduce, or the communicator hangs. Per-rank MoE
  routing diverges, so the collective itself stays unconditional.
- **Steady state has exactly one named readback**: final logits, once per position, for the
  D54 gate, plus per-run counters. Every other per-layer observation point is a counter; a
  debug flag that restores per-layer D2H silently re-acquires the host path.

The D54 tolerance arm (declared before any number): four-rank bf16 Sum on NCCL, one f32
accumulation and one bf16 round per element, same payload and peers, same collective order
per layer. Deviation is reported as differing-element count and max ULP on the 10,240 B
payload; the row-level proof is greedy tokens identical, top-1 1.00, top-5 >= 0.80, top-50
>= 0.96, max abs dlogprob <= 2.0e-2, mean <= 2.0e-3, KL <= 1.0e-4, all finite. NCCL's ring
order is not the canonical rank order — that is the arm, not a defect.

## 8. Open items

| blocker | effect | owner |
| --- | --- | --- |
| `nvidia-peermem` rebuild (needs root) | GPU-initiated RDMA arm unbuildable; only NCCL over socket/RoCE is available | root |
| no fabric run of either device arm | NCCL arm cost is DESIGN only; `gpu-reduce-bench` has no number | comm-lead + a ~3 idle-node-minute window |
| `IBV_WC_RETRY_EXC_ERR` and ~2 ms stalls | 1/5 dual-rail 10,000-replay runs completed; 2 MiB is ~46x its bar | comm + data plane |
| 25 us p50 gate | missed by 1.4x at 35.70 us even at MTU 9000 | comm / fabric |
| device seam returns `Unsupported` | no zero-copy collective; kernel and data plane remain two protocols (slot geometry, arrival signalling, rail split, caller model) | device-lead |
| host `libnccl` absent (B5) | closed for the port by the vendored library; still open for any host process that dlopens the system one | root |
| GPU window | the nodes serve the session's own model backend; a window needs human authorisation and a restore path | serving owner |

The determinism gate is demonstrated, not repeatable: one dual-rail run passed 10,000/10,000
replays bit-identical (11.7 s) while four failed with retry-exceeded after 19 to 5,781
replays. Failures are transport completions, not silent numerical corruption — a completed run
is bit-identical on every replay.

## Archived sources

rdma-backend.md, rdma-bringup.md, ws-b-collectives-20261021.md,
ws-b-infra-tickets-20261021.md, ws-b-mtu-peermem-apply-20261022.md, ws-b-reduce-20261022.md,
ws-whole-layer-device-port-20261022.md (fabric and all-reduce sections), docs/DECISIONS.md
D12/D29
