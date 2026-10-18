# The libibverbs RDMA backend

Status: **real backend, not a skeleton.** `crates/sparkflash-comm/src/rdma.rs` and its four
submodules (`abi.rs`, `verbs.rs`, `qpinfo.rs`, `dataplane.rs`) compile only with the `rdma`
feature and implement the dual-rail RoCEnante transport against libibverbs. Everything below the
hardware line has been built and unit-tested on the x86 development box; **no cluster hardware was
opened by the author of this document**. The three cluster gates and the `rocenante-probe` fabric
path are written to measure, but the numbers they produce must come from a Lead-approved run on
192.0.2.80/.84/.68/.59. The rail, address and device facts in sections 7 and 11 were captured
read-only from the window nodes by the Lead on 2026-10-18 and staged on this dev host; they are not
measurements taken on this dev host, which has no RDMA devices under `/sys/class/infiniband` (the
class directory is present but empty) and no `/dev/nvidia*`.

## 1. What the feature does, and what it does not change

- The `rdma` feature adds **no dependency**: `libc` is already in the crate's dependency list;
  libibverbs is opened with `libc::dlopen` at run time. A default build still links nothing new and
  every non-`rdma` code path is unchanged.
- With the feature, `RdmaTransport::open` / `open_with` is the real bring-up described in
  section 3. `RdmaTransport::is_built()` returns true, and [`crate::exchange`] schedules run over
  the data plane in section 6.
- Without the feature, `rocenante-probe` exits 3 with an empty stdout and names the rebuild
  (this is the existing contract test `tests/probe_cli.rs::without_the_fabric_feature...`), and the
  hardware gates refuse with a rebuild message instead of faking a measurement.

The ABI is hand-written because the backend must compile on hosts where libibverbs is absent. The
static asserts in `abi.rs` are the guard: if rdma-core changes a layout this backend calls through,
the build fails instead of the fabric corrupting memory.

## 2. The ABI subset

16 entry points are resolved from one `dlopen` handle (`libibverbs.so.1`, then
`libibverbs.so`): `ibv_get_device_list`, `ibv_free_device_list`, `ibv_get_device_name`,
`ibv_open_device`, `ibv_close_device`, `ibv_query_port`, `ibv_query_gid`, `ibv_alloc_pd`,
`ibv_dealloc_pd`, `ibv_create_cq`, `ibv_destroy_cq`, `ibv_reg_mr`, `ibv_dereg_mr`,
`ibv_create_qp`, `ibv_destroy_qp`, `ibv_modify_qp`.

The remaining three - `ibv_post_send`, `ibv_post_recv`, `ibv_poll_cq` - are `static inline` in
rdma-core's public header and are **not exported** by `libibverbs.so.1` (`nm -D` against rdma-core
39 resolves 16 of 19). They are compiled into a bundled C shim
(`crates/sparkflash-comm/src/rdma/ibverbs_shim.c`) by `build.rs`, which includes the real
`<infiniband/verbs.h>` and calls the inline wrappers; the shim calls no library symbol, so nothing
links libibverbs and the run-time dlopen still owns the device context. Mirroring
`struct ibv_context_ops` in Rust instead was rejected: that vtable is internal to rdma-core, its
layout changes between releases and no published header promises it. A build host without the
header still compiles the feature with ENOSYS stubs, and `Verbs::load` then refuses with a typed
error naming the missing `libibverbs-dev`/header as the build prerequisite - it never opens a
device that cannot be posted to. On a host with neither the header nor libibverbs the dlopen
`Unsupported` is what a caller sees first; the stub-specific refusal is verified only where
libibverbs is present but the header was missing at build time, and the hardware-free CI host
exercises the stub **compile** path, not that refusal.

Hand-written structures and the offsets the backend actually touches (sizes in bytes, verified by
`const _: () = { ... }` static asserts that run at compile time):

| struct | size | fields used |
|  |  |  |
| `ibv_gid` | 16 | `raw[16]`, 8-byte aligned |
| `ibv_sge` | 16 | `addr`, `length`, `lkey` |
| `ibv_send_wr` | 128 | `wr_id`, `next`, `sg_list`, `num_sge`, `opcode`, `send_flags`, `imm_data` (network order), `wr` @40 (rdma/atomic/ud union), `qp_type` @72, padding @76, trailing anonymous union (bind_mw/tso) @80..128. rdma-core 39 has **no `comp_mask`** field |
| `ibv_recv_wr` | 32 | `wr_id`, `next`, `sg_list`, `num_sge` |
| `ibv_wc` | 48 | `wr_id`, `status`, `opcode`, `byte_len`, `imm_data` |
| `ibv_port_attr` | 52 | `state` @0, `active_mtu` @8, `gid_tbl_len` @12, `link_layer` @46, `flags` @47, `port_cap_flags2` @48. rdma-core 39 has **no `active_speed_ex`** in this struct |
| `ibv_qp_cap` | 20 | `max_send_wr`, `max_recv_wr`, `max_send_sge`, `max_recv_sge`, `max_inline_data` |
| `ibv_global_route` | 24 | `dgid`, `flow_label`, `sgid_index`, `hop_limit`, `traffic_class` |
| `ibv_ah_attr` | 32 | `grh`, `dlid`, `sl`, `src_path_bits`, `static_rate`, `is_global`, `port_num` |
| `ibv_qp_init_attr` | 64 | `qp_context`, `send_cq`, `recv_cq`, `srq`, `cap`, `qp_type`, `sq_sig_all` |
| `ibv_qp_attr` | 144 | `qp_state`, `qp_access_flags`, `pkey_index`, `port_num`, `qkey`, `av`, `path_mtu`, `timeout`, `retry_cnt`, `rnr_retry`, `rq_psn`, `sq_psn`, `max_rd_atomic`, `max_dest_rd_atomic`, `min_rnr_timer`, `dest_qp_num`, `ah_attr` @56, `alt_ah_attr` @88, `rate_limit` @136 |
| `ibv_qp` | 64-byte prefix (C struct is 160) | `qp_num` @52; read-only view, never allocated or copied |
| `ibv_mr` | 48 | `addr`, `length`, `lkey` @36, `rkey` @40 (prefix only) |

Opaque handles (device, context, PD, CQ, SRQ, comp channel, device list) are represented as opaque
pointers; their fields are never read. The struct layouts above were checked field by field against
rdma-core 39's `libibverbs/verbs.h` and are pinned by compile-time `size_of`/`offset_of` asserts, so
an incompatible library is a build failure here; there is no run-time ABI version check, and a
library with different exported symbol names still fails at `dlsym` (typed
[`CommError::Unsupported`]) or on the first refused call ([`CommError::Verbs`]).

## 3. Bring-up (`RdmaTransport::open_with`)

1. `RdmaConfig::validate`: rank/world, rail count 1 or 2, one control address per rank, even frame
   cap not above [`ROCENANTE_MAX_BYTES`], positive timeout.
2. `select_rails_from_live_fabric()`: sysfs enumeration plus [`RailSelection::resolve`], which
   requires ACTIVE and at least 200 Gb/s and puts the two rails on distinct cards. The specification
   names `rocep1s0f0` / `roceP2p1s0f0`; those functions are PORT_DOWN on the live nodes, so the
   resolver picks the ACTIVE f1 twins (section 7).
3. `Verbs::load`: dlopen, `dlsym` of the 16 exported entry points, and the bundled shim's three
   inline wrappers (`ibv_post_send`, `ibv_post_recv`, `ibv_poll_cq`). A build without
   `<infiniband/verbs.h>` compiles ENOSYS stubs and this step refuses with a typed error naming
   the missing header, before any device is opened.
4. Per opened rail: `ibv_open_device` by the measured device name; `ibv_query_port(1)` and refuse
   unless ACTIVE and Ethernet (RoCEv2 only); `ibv_query_gid(1, 3)` and refuse an all-zero GID;
   `ibv_alloc_pd`; `ibv_create_cq(world * 4 + 16)`; allocate one staging buffer of
   `SlotLayout::total_bytes()`; `ibv_reg_mr` with local-write, remote-write, remote-read; create
   one RC QP per rank (send and receive CQ are the rail's CQ, `sq_sig_all = 0`).
5. Publish this rank's `RailEndpoint` record per rail (GID, GID index, negotiated active MTU, staging
   address, remote key, one QP number per rank) in the control plane (section 5), then merge the
   world's records into a `QpTable`.
6. Per rail and peer: `ibv_modify_qp` INIT, then RTR (peer QP number, peer GID, this rank's GID
   index, path MTU = the smaller of the two ports, PSN), then RTS.
7. Hand the rails to `DataPlane` and account the fabric into `CommTelemetry`.

Every failure is a typed [`CommError`] variant and every device object is an RAII wrapper, so a
refusal in the middle of bring-up destroys the objects created before it in reverse order and then
closes libibverbs. The loader field is declared after the data plane on purpose: the library must
outlive every QP, MR, CQ, PD and device context.

## 4. The QP state machine

RC QPs, one per (rail, peer), all transitions from RTR/RTS applied with the exact attribute masks
reported by the transition builders:

| transition | fields the mask announces |
|  |  |
| INIT | `IBV_QP_STATE`, `IBV_QP_PKEY_INDEX` (0), `IBV_QP_PORT` (1), `IBV_QP_ACCESS_FLAGS` (local/remote write, remote read) |
| RTR | `IBV_QP_STATE`, `IBV_QP_AV` (GRH with the peer GID, `is_global = 1`, `sgid_index = 3`), `IBV_QP_PATH_MTU`, `IBV_QP_DEST_QPN`, `IBV_QP_RQ_PSN` (1), `IBV_QP_MAX_DEST_RD_ATOMIC` (1), `IBV_QP_MIN_RNR_TIMER` (12) |
| RTS | `IBV_QP_STATE`, `IBV_QP_SQ_PSN` (1), `IBV_QP_TIMEOUT` (14), `IBV_QP_RETRY_CNT` (7), `IBV_QP_RNR_RETRY` (7), `IBV_QP_MAX_QP_RD_ATOMIC` (1) |

The PSN is the same constant on both queues because each (rail, peer) QP pair carries a fixed,
repeating message schedule. `min_rnr_timer` is 12 and `max_dest_rd_atomic` is 1. The backend does
not use RDMA read or atomics: the data plane is one-sided RDMA WRITE with immediate.

## 5. The control plane

One TCP star per collective. Rank 0 binds the address in `peers[0]` (its own rail address) on
`SPARK_COMM_CONTROL_PORT` (default 18515); every other rank connects to `peers[0]` (retried until
the deadline). Each rank sends one length-prefixed record and receives the merged table. The fixed
header is **12 bytes** (not 16), followed by the body:

| field | bytes | notes |
|  |  |  |
| magic | 4 | `0x5351_5049` (SQI...) |
| version | 2 | 1 |
| kind | 1 | 1 = endpoint table |
| reserved | 1 | 0 |
| body length | 4 | little endian |
| world | 4 | little endian |
| count | 4 | little endian, records in ascending rank order |
| record × count | variable | rank (u32 LE), frame cap (u32 LE), slot bytes (u64 LE), rail count (1 byte + 3 reserved), then per rail: GID (16 raw bytes), GID index (1), active MTU (u16 LE), staging address (u64 LE), remote key (u32 LE), world QP numbers (u32 LE each) |

The geometry is in every record so the decoder can refuse a cluster whose ranks disagree about the
frame cap or slot size - a disagreement that would otherwise put each remote write at a guessed
offset. A record for a rail above its `rail_count` must be all zeros and is validated as such.

`QpTable::decode` refuses a wrong magic/version/world, a truncated or over-long body, a duplicate
or missing rank, an inconsistent staging geometry, a zero GID, a zero QP number, a non-zero record
for an unopened rail, and any table whose record for this rank does not echo what this rank sent.
A non-zero rank that cannot reach rank 0 within the timeout reports [`CommError::Timeout`]; a
rank 0 whose accept or read outlives the deadline does the same (timeouts are not re-labelled as
protocol errors). The control plane carries no measurements: it exists to publish endpoints, and
it closes before the first collective.

In single-rail mode `rail_count` is 1 and the rail-1 record is the all-zero placeholder: the wire
format keeps one shape, the decoder validates that no data was published for an unopened rail, and
the data plane never touches rail 1. Publishing rail 0's endpoint under rail 1 (which an earlier
revision did) is refused by the decoder's zero-placeholder check.


## 6. The data plane

One MR per rail covers a single registered staging buffer (a `Box<[u8]>` of host memory). The
buffer is divided into `2 * world` inbox slots (one per sending peer and schedule phase) followed
by `2 * world` send slots (one per destination and phase). A slot is
`24 + max_frame_bytes` bytes: the 24-byte [`crate::tcp::FrameHeader`] plus the **whole** frame, so
a single-rail send that puts the entire payload on rail 0 fits (an earlier half-sized slot could
not carry a frame above `max_frame_bytes / 2` on one rail and would have cut the receive SGE
short). For the four-rank, 2 MiB configuration that is 2,097,176 bytes per slot and 33,554,816
bytes per rail, and the unit test `slot_layout_keeps_every_slot_inside_the_buffer` asserts every
slot offset plus its full size stays inside the MR.

- prepare: for each expected (peer, phase) frame, post one receive WR per rail whose SGE length is
  the whole slot, tagged with bit 63 set, the rail, the phase and the peer.
- send: write the frame header into the rail-0 outbox, split the bf16 payload into **whole
  elements** (rail 0 takes `elements - elements / 2`, rail 1 the rest, so no element straddles a
  rail), then post one signalled RDMA WRITE_WITH_IMM per rail and peer. The immediate word carries
  sender, phase, kind and the low byte of the sequence number; collect decodes it and refuses a
  frame whose immediate, length, opcode or status disagrees.
- collect: poll both rails' CQs with a short processor yield after 64 empty polls each round, wait
  for every expected frame **and** every signalled send, and fail with
  [`CommError::Timeout`] at the deadline.
- ledger: every posted payload byte is added to [`RailLedger`]
  (`rail0_bytes`, `rail1_bytes`, `transfers`, `element_writes`) at post time, and the delta is
  added to [`CommTelemetry::rdma_bytes_rail0`] / `rdma_bytes_rail1` after the schedule returns.
  The ledger is reset at the start of each collective, so a gate reads the bytes of the one dispatch
  it just ran. The header counts on rail 0.

`DataPlane` is `Send` behind a `parking_lot::Mutex`: the trait hands out `&self` to many
threads, and one collective may be in flight per transport. No QP object is touched concurrently.

## 7. Rails, and the difference between our gate requirement and the deployed reference

The backend is dual-rail capable. That is **our** gate requirement (QUALIFICATION 3.3: aggregate
>= 180 Gbps across both rails), not a measured description of the deployed vLLM configuration.

Evidence for the deployed configuration (Lead-measured from the running container on 2026-10-18,
`docker exec vllm_node env`): `NCCL_IB_HCA=rocep1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1`,
`NCCL_IB_GID_INDEX=3`, `NCCL_IB_DISABLE=0` - **one rail**. The launch script passes a twin pair,
but line 10 of `/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe.sh` overrides it
with the single `rocep1s0f1`. The second twin (`rocep1s0f1` on the other card prefix,
`roceP2p1s0f1`) is ACTIVE at 200000 Mb/s, but the deployed recipe does not use it.

What the resolver sees on the live nodes: the specification names `rocep1s0f0` and
`roceP2p1s0f0`; both f0 functions are PORT_DOWN, so `RailSelection::resolve` refuses them and
selects the ACTIVE f1 twins (`rocep1s0f1` at 200000 Mb/s on card `rocep1s0`, `roceP2p1s0f1` at
200000 Mb/s on card `roceP2p1s0`), which are on distinct cards. `SPARKFLASH_RAIL_DEVICES` can pin
a different pair; the pinned pair is validated the same way.

Address tables, which are easy to confuse:

| table | addresses (rank 0..3) | used by |
| --- | --- | --- |
| rail-1 table | 192.0.2.80 (node0), .84 (node1), .68 (node2), .59 (node3) | `SPARK_COMM_PEERS` in the hardware gates, the probe, and the doctor's new per-rank probe |
| deployed vLLM container | mixed rail-2 addresses (the container's own `NCCL_IB_HCA` selects the device, not the address) | the vLLM reference only |

The gate table is the one the Lead measured for the four nodes' rail-1 interfaces; it is what
`SPARK_COMM_PEERS` carries. It is not a statement about which address family the container prints.

## 8. Modes

| `SPARK_COMM_RAIL_COUNT` | behaviour | gate 3 wording |
| --- | --- | --- |
| 2 (default) | both rails opened, each frame split across them, ledger per rail | asserts both rails carried >= 45% of the measured bytes and the total equals the schedule's posted bytes |
| 1 | rail 0 only; whole frame (header + payload) on rail 0; the rail-1 record is the all-zero placeholder (`rail_count = 1`) and the decoder refuses any non-zero data in it | asserts rail 1 == 0 and rail 0 == every posted byte, and prints that the run is **not** dual-rail evidence |

The counter gate refuses to claim dual-rail evidence from a single-rail run: in single-rail mode the
rail-1 assertion is "zero bytes", never "half the bytes".

## 9. Error taxonomy

| variant | raised for |
| --- | --- |
| `InvalidLayout` | rank/world, rail count, control-address table, frame cap, timeout, an oversized frame |
| `RankOutOfRange` | rank >= world |
| `FabricUnavailable` | no sysfs fabric; a pinned or resolved rail missing/DOWN/too slow; a device that is not ACTIVE, not Ethernet, or whose GID index is zero |
| `Unsupported` | libibverbs cannot be dlopen'd or a symbol is missing; an MR base address mismatch |
| `Verbs` | a libibverbs call returned a non-zero code (call name plus strerror text) |
| `Protocol` | a damaged control record, a peer that published no endpoint/QP number, a completion whose immediate/length/opcode disagrees, an unexpected send completion |
| `Timeout` | the control plane or a collective missed its deadline |
| `PeerClosed` / `Io` | the control-plane socket ending early or a socket error |

No path in the backend panics on a host prerequisite: every refusal is one of the variants above.

## 10. What is verified today, and what is not

Unit-tested on the x86 development box (no RDMA hardware, no libibverbs):

- ABI sizes and offsets, GID classification (unspecified / RoCEv2 IPv4 / other), MTU names and byte
  counts, completion decoding, the `ibv_send_wr` overlay (`abi.rs`);
- the QP-endpoint codec: round-trip, single record, damaged/mismatched table (magic, version, world,
  zero GID, zero QP number, zero frame cap, slot smaller than the cap, rail count 0, unknown MTU,
  null staging address, data published for an unopened rail, ranks disagreeing on geometry), a real
  four-thread 127.0.0.1 star exchange (`qpinfo.rs`);
- slot layout and rail split, immediate and work-request tag round trips, the write plan, the three
  attribute masks, the path-MTU minimum (`dataplane.rs`);
- `RdmaTransport::open(4, 0)` on a host with no configuration/fabric returns a typed error and
  never panics; bad configurations are refused before bring-up; the spec rail names are the ones
  documented; libibverbs absence is a typed `Unsupported` (`rdma.rs`, `verbs.rs`);
- the three hardware gates compile in both feature configurations and refuse (permission, feature)
  without touching the cluster (`tests/hardware_gates.rs`);
- the doctor's check 3 hands every peer to the bandwidth probe and measures **this node's** rank
  view exactly once (`--rank`, with `SPARK_COMM_PEERS` for the full table), failing with the rank
  and reason when it cannot be measured rather than skipping it; other ranks need concurrent probe
  processes (`doctor.rs` unit tests).

Not verified by this document: anything on the cluster. No device was opened, no collective ran,
the three gates and `rocenante-probe --transport rdma` have never printed a number, and the latency
budget in gate 1 is this gate's acceptance bar (wire floor at the negotiated rate, floored at the
specification's 20 us), not a measured cluster result.

## 11. Exact commands

### 11.1 Address tables, and which one the control plane needs

Every node has two rail addresses. They are **control-plane TCP addresses only**: the data plane
uses the RoCE GIDs published in the QP-info table, never these strings.

| table | rank 0..3 (node0, node1, node2, node3) |
| --- | --- |
| rail 1 | 192.0.2.80, 192.0.2.84, 192.0.2.68, 192.0.2.59 |
| rail 2 | 192.0.2.85, 192.0.2.88, 192.0.2.64, 192.0.2.63 |

`SPARK_COMM_PEERS` must be the same table on all four processes and reachable from all four
nodes. Use the rail-1 table by default. Use the rail-2 table instead when the window's management
path is rail 2 - for example when the rail-1 addresses sit on a segment not routable from every
node in the window:

```text
SPARK_COMM_PEERS=192.0.2.85,192.0.2.88,192.0.2.64,192.0.2.63
```

The table selects no device: the rails are resolved by device name from sysfs (section 7). If the
window brings up a different pair than the resolver picks, pin it explicitly with
`SPARKFLASH_RAIL_DEVICES=rocep1s0f1,roceP2p1s0f1`. The control port (18515 by default) must be free
on rank 0 and reachable from the other three nodes, and only one gated process may run per node at
a time.

### 11.2 Build and stage the test binary

```text
cargo test -p sparkflash-comm --features rdma --release --test hardware_gates --no-run
```

The `--no-run` line prints the executable, e.g.
`Executable tests/hardware_gates.rs (target/.../hardware_gates-<hash>)`. Cross-build that binary with
the project's established aarch64 pipeline (docs/build/aarch64-pipeline.md) and stage it, with the
same libibverbs stack the nodes already run, one copy per node. The staged binary needs no cargo and
no source tree.

A safe local check that the binary refuses rather than fakes (run it on the dev box, no cluster, no
`SPARK_COMM_HW_PERMISSION`): it prints three permission panics and opens no device.

```text
cargo test -p sparkflash-comm --features rdma --test hardware_gates -- --ignored --test-threads=1
```

### 11.3 One process invocation per node for all three gates

Run the same staged binary with the same arguments on all four nodes inside the rendezvous window.
libtest executes all three ignored tests sequentially in one process, in the same alphabetical order
on every node (`dual_rail_counters...`, `roce_bits...`, `roce_oneshot...`), and each test opens
its own transport, so every gate is its own four-node rendezvous. `--test-threads=1` is required:
the gates share one control port and one process, and their order must match across nodes. Confirm
the order with `--list --ignored` before the window.

```text
SPARK_COMM_HW_PERMISSION=1 SPARK_COMM_RANK=<0..3> SPARK_COMM_WORLD=4 \
SPARK_COMM_PEERS=<table from 11.1> SPARK_COMM_CONTROL_PORT=18515 SPARK_COMM_TIMEOUT_MS=60000 \
./hardware_gates-<hash> --ignored --test-threads=1 --nocapture
```

`SPARK_COMM_TIMEOUT_MS=60000` is recommended for manual orchestration: the default 5 s deadline is
for a programmed start, and four humans or four terminal windows need slack in the control-plane
rendezvous. The timeout covers setup; it is not part of any latency sample.

Per-gate form (three invocations, same order on every node):

```text
./hardware_gates-<hash> --ignored --exact --nocapture <gate name>
```

Gate names: `roce_oneshot_two_mib_all_reduce_meets_the_latency_budget`,
`roce_bits_match_the_cpu_reference_over_ten_thousand_replays`,
`dual_rail_counters_show_both_interfaces_carrying_half_the_bytes`. Add
`SPARK_COMM_RAIL_COUNT=1` to all four processes for the deployed single-rail configuration; gate 3
then reports single-rail mode and refuses to claim dual-rail evidence.

Doctor check 3 (per-peer ibv_write_bw plus one rocenante-probe for **this node's** rank view):

```text
SPARK_COMM_PEERS=192.0.2.80,192.0.2.84,192.0.2.68,192.0.2.59 \
  sparkflash doctor --rank 0 --nodes 192.0.2.84 --nodes 192.0.2.68 --nodes 192.0.2.59
```

A single doctor process can measure only its own rank's latency: the control plane is a four-node
rendezvous, so the probe must run on all four nodes at the same time (Lead orchestration, the same
window shape as section 11.3). Pass `--rank` for the node the doctor runs on and
`SPARK_COMM_PEERS` as the full rank table including that node's own address, because rank 0 binds
`peers[0]`. An unmeasurable rank fails check 3 instead of being skipped.

### 11.4 What to expect on this hardware (pre-run hypotheses, not measurements)

These are expectations derived from the schedule and the known gaps, written down so that the first
run's numbers are either confirmed or visible as a named gap. They are **not** measurements and none
of them may be quoted as a result.

| gate | expected on 192.0.2.80/.84/.68/.59 | read a surprise as |
| --- | --- | --- |
| 10 KiB one-shot (dual or single rail) | median 10-25 us: host-memory post + RC round trip at active_mtu 1024, with rail splitting not mattering at this size | < 8 us is suspicious (host memory cannot post and complete that fast); > 25 us is the documented host-staging gap, not a regression, and the 20 us spec bar is then unmet |
| 2 MiB one-shot | one_shot sends the whole per-rank buffer to each of 3 peers: posted payload 3 x 2,097,152 + 72 header bytes = 6,291,528 B; wire floor 6,291,528 x 8 / 400 Gb/s = **125.8 us** if both rails contribute 400 Gb/s (251.7 us if the twins share one 200 Gb/s); the gate bar is 1.5x floor = **188.7 us** (377.5 us). Expect 240 us - 1 ms from host staging | expect this gate to report FAIL on the first run and record it as the host-staging gap, not a code regression; below the floor is a timing bug, above 1 ms points at MTU 1024 / host-copy pathology |
| 10,000 replays bit-identical | PASS; per-rank runtime under about 10 s (10,100 x 10-30 us plus pattern generation) | minutes mean RNR/retry timeouts, not correctness. Any bit mismatch is a real bug; note the reference (reduce.rs) accumulates in f32 and rounds to bf16 once, so equality is exact for THIS accumulation order, not a property of bf16 addition in general |
| dual-rail counters (1 MiB, world 4) | one_shot posts the whole 1 MiB per-rank buffer to each of 3 peers: 3 x (1,048,576 + 24) = 3,145,800 B; rail0 1,572,936 B and rail1 1,572,864 B (whole-element split, 72 header bytes), printed as 50.00/50.00% (exact shares 50.0011/49.9989, each rail rounds to 50.00 at two decimals; the 45-55% assertion is the real check); total == 3,145,800 B +/- 3 B | a split outside 45-55% or a total drift is a bug; rail1 == 0 means the run was single-rail or only one rail opened |
| single-rail counters (1 MiB, world 4) | rail0 3,145,800 B, rail1 0 B | any rail-1 byte is a bug |

Control-plane open time (milliseconds, one TCP star) is outside every timed sample. A gate that
fails at open with `Timeout` means the four processes did not start within the window or the peer
table was wrong: fix the window, not the code.

## 12. Honest limits

1. **Host-memory staging.** The MR covers host memory (`Box<[u8]>`). The specification's slot design
   is device memory reached by GPUNetIO / `cudaHostAllocMapped`; this backend posts from registered
   host memory. It is a working RDMA collective, not the zero-copy GPU path, and its latency numbers
   will not be the board's best case.
2. **MTU.** The live nodes negotiate `ibv active_mtu = 1024` (IP MTU 1500), while the doctor's gate
   and the specification want 9000-byte IP MTU (4096 active_mtu). The backend uses the smaller of
   the two ports' active MTUs and still works, but the 10 KiB and 2 MiB numbers measured on 1024-byte
   MTUs are not the qualified configuration.
3. **Driver floor.** All four nodes report driver 580.173.02, below the doctor's floor 580.178.04.
4. **One address family.** Only the RoCEv2 IPv4 GID form (`::ffff:a.b.c.d`) is accepted: a port
   whose configured GID is another IPv6 form is refused at bring-up with a typed error rather than
   guessed at. The deployed vLLM recipe uses GID index 3, the IPv4 form.
5. **The deployed reference is single rail.** Dual rail is our requirement; gate 3 is the only place
   that asserts it, and it runs both modes rather than assuming.
6. **No hardware run by this author.** Every number must come from a Lead-approved four-node run; this
   document deliberately contains none.
7. **The doctor cannot measure a remote rank's view from one node.** It measures exactly the local
   rank named by `--rank`; the other ranks' probe processes must run at the same time (Lead
   orchestration). A single doctor run without the peers therefore FAILS check 3 instead of
   fabricating other ranks' numbers or passing on partial evidence.
8. **Control plane is unauthenticated plain TCP.** It is a bring-up rendezvous on the cluster's
   private network, closes before the first collective, and is not a security boundary.
9. **`ulimit -l unlimited`** is required for the 32.0 MiB-per-rail registration (16 slots x 2,097,176 B = 33,554,816 B; the vLLM container
   already sets `--ulimit memlock=-1`).
10. **The 2 MiB latency budget is this gate's acceptance bar**, derived from the negotiated rail rate
    (1.5x the wire floor, never below the specification's 20 us). It is not a router-level threshold
    and it is not a qualified number.
11. **The three inline verbs need the rdma-core header at build time.** Without
    `<infiniband/verbs.h>` the feature still compiles, with ENOSYS stubs, and `Verbs::load`
    refuses with a typed error naming `libibverbs-dev`; it never opens a device it cannot post to.
    The hardware-free CI host exercises the stub **compile** path; it cannot exercise the
    stub-specific refusal because it has no libibverbs at all, so `load()` there stops at the
    dlopen `Unsupported` first. That refusal and the real inline path were both verified on a host
    that has libibverbs and rdma-core 39 headers (the critic's container).

## 13. The decision record

The decision this document records is `docs/DECISIONS.md` **D29**: the backend is real and dual-rail
capable, the deployed vLLM reference is single rail, and no document may cite that single-rail
reference as dual-rail evidence. `docs/DECISIONS.md` is the normative record; this document stays the
technical record - the ABI subset (2), bring-up (3), the QP machine (4), control and data plane (5-6),
the rail layout and its evidence (7), modes (8), errors (9), verified scope (10), exact commands (11)
and honest limits (12).

## 14. Documentation follow-ups

The stale claims this section listed were fixed when D29 landed: `docs/QUALIFICATION.md` now describes
the backend as real and the three gates as real but never run, the runbook's per-rank doctor probe is
stated with the four-concurrent-processes requirement, and `docs/DECISIONS.md` D17 findings 3-4 and 7
carry the corrected mechanism and the single-rank probe limit. The host-memory staging, active-MTU
and single-rail gaps are recorded in this document's section 12, not in `docs/QUALIFICATION.md` or
`docs/build/roadmap-to-serving.md` (the roadmap never mentions single-rail). Keep this section as the record of that change, not a list of outstanding work.

