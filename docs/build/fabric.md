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
| NCCL 2.28+ binding (dlopen) | implemented and measured across 4 ranks (`libnccl.so.2` `2.31.2+cuda13.0`, `10,000` samples, native boost clocks: `registered_buffers: 2`, `max_ulp: 0`, `p50 = 47.89-49.76 us`, `p90 = 58.51-69.39 us`, `p99 = 78.72-167.20 us` over 40 rank-samples, `gdr_active: false` on GB10 UMA). The S2-a gate **PASSES** under its D57 three-bar set (`p50 <= 75`, `p90 <= 104`, `p99 <= 251`); GDR is recorded as undrivable rather than failed (§5.2, §1) | MEASURED (2026-10-24, 13 runs x 4 ranks) |
| device-buffer seam (`all_reduce_bf16_device`) | `CommError::Unsupported` stub | DESIGN |
| 25 us p50 decode all-reduce gate | never met on this fabric (`47.89-49.76 us` on the NCCL RoCEv2 UMA proxy; `35.70 us` on the host-staged one-shot RDMA). Since D57 this is a **product-target** row, not the S2-a gate bar, whose measured ceilings are `p50 <= 75` / `p90 <= 104` / `p99 <= 251` | MEASURED (missed) |
| MTU 9000 on hosts | applied and persisted 2026-10-21 | MEASURED |
| `nvidia-peermem` / GPUDirect | module present, driver rejects it; `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 0` (attribute 116) and `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0` (attribute 124) on GB10 (`sm_121a`), read by the gate's per-rank capability probe and recorded in `platform-rank<r>.json`. Under D57 the gate requires GDR active only on a platform that reports support, so GB10 records `gdr_required: false` rather than failing | MEASURED |
| forward-row collective | named explicitly by `SPARK_COMM_REDUCE` (no default): `nccl` on the device arm (the default arm since S2), `tcp`/`roce` on the host arm (`SPARKFLASH_LAYER_DEVICE=0`); the run header records the path, the NCCL build and the `NCCL_*` transport env | MEASURED (`s2-d`: `146.03 ms`, `s2-d-rep2`: `145.28 ms`, `s2-e`: `1,000` steps, `0` NCCL faults) |
| collective buffer registration | the four 10,240 B collective spans registered once at open (`ncclCommRegister`, `reduce_registered_buffers = 4`, `40,960 B`), released before the slab is freed | MEASURED (2026-10-23, S2-a / S2-d / S2-e) |

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

### 5.2 NCCL 10 KiB at native boost clocks: the S2-a gate floor (D57)

The S2-a gate's ceilings are not chosen, they are `1.5x` the measured worst of the rows below. The
floor is a **protocol set**, not a selection: a row counts only if the gate script that produced it
is committed (`provenance.git_dirty = false`), the protocol is native boost with no clock lock, and
the summary's schema is one its own `git_head` can emit. Every row taken since the instrument fixes
of 2026-10-24 is listed, including the ones that make the ceilings looser.

| run (UTC) | warm-up | p50 per rank (us) | p90 per rank (us) | p99 per rank (us) | clock median, non-idle (MHz) |
| --- | --- | --- | --- | --- | --- |
| 20261024T025932Z | - | 47.89 48.11 48.06 48.02 | 58.62 58.62 58.91 59.01 | 82.80 81.80 81.46 82.15 | 2411 2418 2411 2411 |
| 20261024T030003Z | - | 48.43 48.50 48.47 48.45 | 58.51 58.85 58.90 58.60 | 81.85 81.45 82.35 82.20 | 2411 2411 2411 2411 |
| 20261024T030033Z | - | 48.16 48.18 48.32 48.20 | 59.15 58.80 58.80 58.78 | 85.50 82.94 84.20 84.60 | 2411 2418 2411 2411 |
| 20261024T031205Z | - | 49.76 49.65 49.20 49.12 | 69.39 69.02 68.67 68.80 | 161.25 164.93 164.30 163.80 | 2554 2554 2502 2541 |
| 20261024T031236Z | - | 48.29 48.35 48.44 48.48 | 65.98 66.20 67.26 66.80 | 166.46 167.20 166.10 166.90 | 2405 2405 2398 2411 |
| 20261024T031342Z | yes | 48.30 48.46 48.19 48.14 | 59.81 60.53 60.19 60.16 | 82.61 82.86 81.49 82.77 | 2411 2411 2411 2411 |
| 20261024T031425Z | yes | 48.37 48.40 48.40 48.26 | 59.46 59.55 59.47 59.42 | 79.47 79.14 79.30 78.72 | 2411 2418 2398 2411 |
| 20261024T031507Z | yes | 48.50 48.48 48.56 48.56 | 60.16 59.76 59.82 59.68 | 86.11 86.90 87.07 87.52 | 2411 2418 2411 2411 |
| 20261024T031608Z | yes | 48.51 48.42 48.50 48.45 | 59.54 59.76 59.90 59.90 | 83.62 83.66 83.36 82.32 | 2411 2418 2411 2411 |
| 20261024T031650Z | yes | 48.21 48.21 48.22 48.26 | 59.68 60.26 59.92 59.78 | 149.58 151.17 154.30 154.98 | 2411 2411 2411 2411 |

Across **40 rank-samples over those 10 runs**: p50 `47.89-49.76 us`, p90 `58.51-69.39 us`,
p99 `78.72-167.20 us`. Over all 16 rows including the six confirmations below p50 is
`47.82-49.76 us` and p90 `58.35-69.39 us` - neither ceiling moves. The D57 ceilings are `p50 <= 75 us`, `p90 <= 104 us`, `p99 <= 251 us`, each
`1.5x` the measured worst.

Six further runs under the same protocol and the committed three-bar gate confirm it, all `PASS`
and all `provenance.git_dirty: false` at `sparkflash-reduce-gate/3`:

| run (UTC) | p50 per rank (us) | p90 per rank (us) | p99 per rank (us) | clock median (MHz) |
| --- | --- | --- | --- | --- |
| 20261024T031917Z | 48.11 48.30 48.05 48.10 | 59.09 59.55 58.98 59.14 | 89.55 87.66 87.14 87.14 | 2411 2418 2411 2411 |
| 20261024T031959Z | 48.35 48.43 48.35 48.34 | 59.74 59.78 59.68 59.84 | 85.52 85.76 85.39 85.36 | 2411 2411 2411 2411 |
| 20261024T032040Z | 48.37 48.29 48.30 48.37 | 59.66 59.42 59.33 59.50 | 83.63 84.27 84.90 84.03 | 2411 2411 2411 2411 |
| 20261024T032944Z | 48.69 48.66 48.70 48.88 | 64.43 64.40 64.29 64.32 | 144.27 147.68 146.37 141.60 | 2411 2418 2489 2411 |
| 20261024T033025Z | 47.84 47.82 48.03 47.84 | 58.48 58.37 58.56 58.50 | 81.81 81.83 83.09 83.10 | 2411 2418 2411 2411 |
| 20261024T033107Z | 48.34 48.45 48.34 48.38 | 58.70 58.50 58.35 58.37 | 122.40 118.94 120.96 126.18 | 2411 2411 2411 2411 |

#### Why p99 is not the primary tail bar here

The p99 on this fabric is **heavy-tailed and not reproducible**, and the tail is not the collective.
Across the 16 rows now on record it takes values at three separate levels - `78.7-89.6 us` (ten
runs), `118.9-126.2 us` (`033107Z`), and `141.6-167.2 us` (`031650Z`, `032944Z`, the ARP-adjacent
runs) - while **p50 stays inside `47.82-49.76` (4.0% spread) and p90 inside `58.35-69.39`, and the
clock medians do not move**. Two adjacent runs of the same committed script on the same warmed
fabric produced p99 `87.5` and then `149.6`. A statistic that moves 1.9x between adjacent runs while
the central tendency moves 1% cannot be a pass/fail bar at that magnitude without making the gate
flaky, which is what happened when the ceiling was set at `131`: `031608Z` passed and `031650Z`
failed, same protocol, consecutive runs. An earlier draft of this section called the tail "bimodal";
`033107Z` (p99 `118.9-126.2`) is the counterexample, so the accurate statement is the one above.

So the gate carries three bars with three jobs:

- **p50** (`<= 75 us`) detects a central shift: a transport fallback, a lost registration, a clock
  change. It is the number the fabric is actually specified by, and it moves 1.1% run to run.
- **p90** (`<= 104 us`) is the tail bar, because p90 is *stable* here: `58.51-69.39 us`, 0.9% spread
  within the warm protocol. It also catches the broken states by a wide margin.
- **p99** (`<= 251 us`) is a catastrophic-tail alarm only. It is documented as non-reproducible
  below that magnitude (`78.7-167.2 us` on healthy fabric), and its job is the ARP-broken fabric of this window
  (`p50 56.7-58.9`, `p90 402-448`, `p99 583-668 us`), which fails all three bars by 4-5x.

The warm-up matters for the same reason. Two runs taken immediately after an ARP flush measured
`p50 49.12-49.76` and p99 `161-167`; with every rank pinging every peer on both rails first, the
tail returns to `78.7-87.5`. Both RoCEv2 rails share `192.0.2.0/24`, so the gate now controls the
neighbour state it measures instead of sampling it (`fabric.warmed` in every summary).

#### Rows that are not part of the floor, and why

| rows (UTC, 2026-10-23/24) | why they are context, not floor |
| --- | --- |
| `20261023T233633Z` `20261024T021535Z` `021603Z` `021630Z` | taken under the retired `p50 <= 30 / p99 <= 60` ceilings |
| `20261023T233633Z` | `locked: true`, `lock_mhz: 2411` - an artificial lock, not native boost |
| `20261024T021803Z` `021905Z` | taken from a working tree whose script was not committed yet |
| `20261024T022148Z` `022219Z` | same, and their recorded `git_head dc3a0342` cannot emit the schema they wrote |
| `20261024T004250Z` `004329Z` | the ARP-broken fabric: p50 `56.7-58.9`, p90 `402-448`, p99 `583-668 us` |
| `20261024T021124Z` `021152Z` | 2411 MHz-locked protocol; p99 `90.8-96.3` and `153.4-156.2` |

They are kept because they are evidence about the fabric, and because dropping the inconvenient rows
is how a "floor" becomes a wish. None of them is needed for a ceiling: the floor above already
contains a p99 outlier of its own (`167.20`).

### 5.3 2 MiB, byte accounting, and the host reduce

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
loop is 29.5 ms/step; the published row is the device arm's `146.03 ms/step` (`s2-d`), with the host
lean loop's `179.81` kept as the reference-path history.

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
  decision (the run header's), not the device switch alone: evidence recording forces the host
  arm, and a half-decision gate gives the host arm a libnccl dependency and a new setup cost. The
  header refuses `SPARK_COMM_REDUCE=nccl` on the host arm for the same reason.
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
| the S2-a gate has no *locked* row | Resolved as a bar decision, not a measurement: D57 retired the 1500 MHz lock (operator decision: the parts hold 2405-2411 MHz) and set the ceilings at 1.5x a measured floor, so the gate's rows are boost rows and **PASS**. No locked row will be taken | closed by D57 |
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
