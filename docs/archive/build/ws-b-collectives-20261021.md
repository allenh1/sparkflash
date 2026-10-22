# WS-B collectives: interface, dispatch, exactness and the measurement contract (2026-10-21)

**Status:** design + implemented seam for task-2 / WS-B. The dispatch policy, the NCCL 2.28+
dlopen binding and the four-rank microbenchmark are in `crates/sparkflash-comm` and pass
`cargo test -p sparkflash-comm` with **no GPU and no libnccl** on this host. **No WS-B latency
number exists on this branch.** The 25 us p50 gate is *not-yet-measured* and may not be quoted as
met or missed from anything here; the only measured fabric numbers in this document are the
2026-10-20 bring-up row, cited with its evidence path.

This file is the contract the device/forward owner integrates against. It is deliberately explicit
about what is measured, what is an estimate, and what is a hypothesis awaiting a window.

Related: `docs/build/ws-b-infra-tickets-20261021.md` (B1 MTU, B2 peermem, B3 rails, B4 host
doctor, B5 host NCCL), `docs/build/rdma-bringup.md` (measured 2026-10-20 bring-up),
`docs/build/rdma-backend.md` §15 (device-buffer seam), `docs/build/tp4-design.md` §4.3
(Path T vs Path N).

## 1. Shape of the problem

The one-shot RDMA transport reduces a single rank row of bf16 elements with a canonical reduction:
each rank sends its row once, every rank sums the rows in **ascending rank order starting from
0.0f32**, and the result is rounded to bf16 exactly once (RNE). That makes the output
**bit-identical across runs and across ranks** for the same input bits and world size. The
canonical path is capped at `ROCENANTE_MAX_BYTES = 2 MiB` per call; above that the transport
cannot carry the payload and the call is refused, not truncated.

NCCL is the fallback for payloads the one-shot path cannot take. It is not bit-identical (NCCL's
algorithm and reduction order are topology-dependent and version-dependent) and it needs device
buffers plus a host `libnccl.so.2`, which this host does not have (measured 0 hits in
`ldconfig -p`; ticket B5).

**Decode row measured shape:** hidden 5,120 bf16 = 10,240 B per rank. Spec decode all-reduce range
is 10-360 KiB. All of that is inside the 2 MiB one-shot domain, so the decode hot path is the
one-shot path; NCCL is for chunked prefill (>2 MiB), where the row is split or where the contract
is allowed to be non-bit-identical.

## 2. Size-based dispatch (implemented, ungated, pure)

Source: `crates/sparkflash-comm/src/dispatch.rs`. No feature gate, no I/O, no timing: the
policy is a pure function so the routing table can be tested on this host.

| Constant | Value | Meaning |
| --- | --- | --- |
| `DECODE_HIDDEN_ELEMENTS` | 5,120 | one hidden row per rank, bf16 |
| `DECODE_HIDDEN_BYTES` | 10,240 | that row in bytes |
| `DECODE_ONESHOT_MAX_BYTES` | 360 KiB | the top of the spec decode all-reduce range, still one-shot |
| `DispatchPolicy::default().oneshot_max_bytes` | 2,097,152 (ROCENANTE_MAX_BYTES) | inclusive cap |

~~~rust
// crates/sparkflash-comm/src/dispatch.rs (public surface)
pub enum CollectiveBackend { OneShotRdma, NcclRing }
pub struct DispatchPolicy { /* oneshot_max_bytes */ }
impl DispatchPolicy {
    pub fn new(oneshot_max_bytes: usize) -> Result<Self, CommError>;
    pub const fn route(&self, num_bytes: usize) -> CollectiveBackend; // num_bytes <= cap => OneShotRdma
    pub fn route_with(&self, num_bytes: usize, availability: BackendAvailability)
        -> Result<CollectiveBackend, DispatchError>;
    pub const fn is_decode_payload(&self, num_bytes: usize) -> bool;
}
pub struct BackendAvailability { pub oneshot_rdma: bool, pub nccl: bool }
pub enum DispatchError { OneShotUnavailable { num_bytes }, NcclUnavailable { num_bytes, oneshot_max_bytes } }
~~~

Rules the integration must respect:

1. **The 2 MiB cap is inclusive.** 2,097,152 B routes one-shot; 2,097,153 B does not.
2. **Never silently fall back.** `route_with` returns a typed error naming the missing backend
   (and, for NCCL, the string `libnccl.so.2` and ticket B5). A caller may not catch it and run a
   host/TCP all-reduce "instead" in the decode path: that is a different (slower, differently
   rounded) collective and it would be unmeasured. `From<DispatchError> for CommError` maps to
   `CommError::Unsupported` with the same text.
3. **Routing is per call, on the byte count.** There is no hidden state and no learning.
4. `DualPathCommunicator::dispatch_all_reduce_bf16(layer_id, num_bytes, enable_l2_weight_prefetch)
   -> CollectiveRoute` and `dispatch_all_gather_bf16(layer_id, num_bytes) -> CollectiveRoute`
   now branch on this same policy (behaviour unchanged, one source of truth). The route does **not**
   check backend availability; a caller with a transport in hand must call `route_with` first if
   it wants a refusal before encoding.

## 3. NCCL 2.28+ dlopen binding (implemented, ungated on unix)

Source: `crates/sparkflash-comm/src/nccl.rs`. Loaded at runtime with `libc::dlopen` /
`dlsym` / `dlclose`; there is **no build-time NCCL dependency and no new crate** (`libc` was
already a dependency; it is the only way to resolve NCCL without a C toolchain at build time).

- Search order: `libnccl.so.2` then `libnccl.so`; both names are reported in the error.
- Version floor: `NCCL_MIN_VERSION_CODE = 22_800` (2.28.0), decoded as
  `major*10000 + minor*100 + patch`. The image on this cluster has 2.28.3 (measured inside
  `vllm-node:latest`; ticket B5 installs a host copy).
- `ncclUniqueId` is 128 bytes (ABI-pinned by const assertions), hex round-trips, and
  `NcclDataType::ncclBfloat16 = 9` / `NcclRedOp::ncclSum = 0` are pinned with their neighbours.
- `ncclCommInitRankConfig` is called with a **NULL config pointer** (the documented default).
  This avoids depending on a versioned struct layout that differs between minor releases.
- Symbol resolution is injectable (`trait SymbolLoader`) so the missing-library, missing-symbol,
  version-floor and bad-rank paths are all unit-tested on a host with no GPU.

~~~rust
// Public surface (errors elided for brevity)
pub struct NcclApi { /* loader, raw symbols, version */ }
impl NcclApi {
    pub fn load() -> Result<Self, NcclError>;             // dlopen + resolve + version floor
    pub fn version(&self) -> Option<c_int>;               // 22_803 for 2.28.3
    pub fn version_string(&self) -> Option<String>;       // "2.28.3"
    pub fn library_name(&self) -> &str;
    pub fn unique_id(&self) -> Result<NcclUniqueId, NcclError>;
    pub fn comm_init_rank_config(&self, world_size: usize, rank: usize, unique_id: NcclUniqueId)
        -> Result<NcclComm<'_>, NcclError>;
    pub fn group_start(&self) -> Result<(), NcclError>;
    pub fn group_end(&self) -> Result<(), NcclError>;
}
impl NcclComm<'_> {
    pub fn as_raw(&self) -> NcclCommHandle;
    pub unsafe fn all_reduce_bf16(&self, send: *const c_void, recv: *mut c_void,
        count: usize, op: NcclRedOp, stream: CudaStream) -> Result<(), NcclError>;
    pub unsafe fn all_gather_bf16(...) -> Result<(), NcclError>;
}
~~~

`ws-b-allreduce-bench --check-nccl` is the ticket-B5 acceptance probe: it loads the library,
prints path and version, exits 0; today it exits 3 with a message naming B5 and the image.

## 4. Exactness contract (the part that must not be fudged)

| Path | Reduction order | Rounding | Bit-identical across runs/ranks | Allowed for |
| --- | --- | --- | --- | --- |
| one-shot RDMA (host buffers) | ascending rank, f32 accumulator from 0.0 | one bf16 RNE at the end | **yes** (same input bits, same world) | every decode/token row and every seam that requires determinism |
| one-shot RDMA (device buffers) | same canonical kernel | same single RNE | **yes**, once S5.3 lands and the gate proves it | decode hot path (the target) |
| NCCL ring | NCCL's algorithm, topology-dependent | NCCL's own reduction | **no claim** | chunked prefill >2 MiB only, where the spec tolerates fp32-order differences |

Consequences for the device/forward owner:

1. **Never put a >2 MiB engine seam on the NCCL path and call it bit-identical.** If a seam can
   exceed 2 MiB, either chunk it into <=2 MiB one-shot pieces with an explicitly specified combine
   (and a test that pins the chunk order), or declare the seam non-bit-identical in the spec and
   cover it with a tolerance test. Do not mix the two within one token row's reduction.
2. **The engine's f32 <-> bf16 conversion is part of the ABI.** The transport reduces u16 bf16
   bit patterns. Use a round-to-nearest-even f32->bf16 conversion (the engine already has one; do
   not truncate with `>> 16`). A truncating conversion changes the input bits and therefore the
   output; the canonical reduction cannot repair it.
3. **Determinism is a per-route property.** `CollectiveBackend::is_bit_identical()` exists so a
   caller can assert the route it got. The NCCL route answers false on purpose.

## 5. Integration interface for the device/forward owner

The forward owner owns `crates/sparkflash-engine/src/parallel_tp.rs` (the `TpCollective`
trait) and the call site in `crates/sparkflash-engine/src/forward/mod.rs` (around line 1908:
`tp.all_reduce(&mut scratch.ffn_out)`). **This branch does not edit those files.** Two levels,
in order:

### Level 1 - host-f32 adapter over the existing host-slice transport (can land now)

Add a `TpCollective` implementation that owns a `dyn Transport` (the RDMA transport when
`--features rdma` is built, or `TcpTransport`/`LoopbackTransport` elsewhere) and a
`DualPathCommunicator`. Sketch (names are the real ones; error types are the engine's):

~~~rust
struct CommTp {
    transport: Arc<dyn Transport>,
    dispatch: DualPathCommunicator,
    world: usize,
    scratch: Vec<u16>,          // reused; allocated once, never per token
}

impl TpCollective for CommTp {
    fn all_reduce_assign(&mut self, layer_id: u32, values: &mut [f32]) -> Result<(), TpError> {
        let num_bytes = values.len() * 2;
        // 1. Refuse before encoding: which backend must serve this exact byte count?
        let availability = BackendAvailability {
            oneshot_rdma: true,   // there is a transport in hand
            nccl: false,          // no host libnccl (ticket B5) and no device buffers yet
        };
        let backend = DispatchPolicy::default()
            .route_with(num_bytes, availability)
            .map_err(|error| TpError::unsupported(error.to_string()))?;
        if backend != CollectiveBackend::OneShotRdma {
            return Err(TpError::unsupported("NCCL path not wired for token rows"));
        }
        // 2. Encode f32 -> bf16 RNE into the reused scratch buffer.
        self.scratch.clear();
        self.scratch.extend(values.iter().map(|value| f32_to_bf16_rne(*value)));
        // 3. Canonical one-shot reduce. seq is monotonically allocated by the dispatch.
        let route = self.dispatch.dispatch_all_reduce_bf16(layer_id, num_bytes, false);
        let CollectiveRoute::RoCEnanteOneShot { seq, .. } = route else {
            return Err(TpError::unsupported("route changed between check and dispatch"));
        };
        let layout = CollectiveLayout::one_shot(self.world, values.len())
            .map_err(|error| TpError::invalid_layout(error.to_string()))?;
        self.transport
            .all_reduce_bf16(&mut self.scratch, seq, &layout)
            .map_err(TpError::from)?;
        // 4. Decode back to f32 (bf16 is exact in f32).
        for (destination, bits) in values.iter_mut().zip(&self.scratch) {
            *destination = f32::from_bits(u32::from(*bits) << 16);
        }
        Ok(())
    }
    // all_gather_assign: same shape with Transport::all_gather_bf16 and world*local layout.
}
~~~

Interface notes (these are requirements, not suggestions):

- **Preallocate.** `scratch`, `CollectiveLayout` and the `DualPathCommunicator` are built
  once at startup; the decode path may not allocate per call. `CollectiveLayout::one_shot(world,
  elements)` is cheap but keep it out of the loop anyway.
- **seq discipline.** Use the seq that `dispatch_all_reduce_bf16` returns; do not invent a second
  counter. It is per-transport, starts at 1, and the RDMA transport rejects reuse.
- **No host fallback.** On any transport error, return the error (the scheduler can retry the
  step); never run a TCP or CPU all-reduce inside this function to "keep going".
- **Layer id.** The trait today has no layer parameter; pass `0` and file a follow-up to add it
  (telemetry only) or thread it if the trait is being extended anyway. It does not affect bits.
- **Prefetch flag.** Pass `false` until the L2 weight prefetch has a measured win; the flag only
  schedules a 6 MiB prefetch during the RDMA wait.
- **Cost (estimate, not measured):** a bf16 encode+decode of 5,120 elements is a few microseconds
  of scalar work on the host; the decode gate budget is 25 us total, so if profiling shows it,
  encode/decode can move to a fused kernel later. Do not assume it is free.

### Level 2 - device-buffer seam (device-lead owns the launcher; the seam exists)

`Transport::all_reduce_bf16_device(&DeviceBuffer, elements, seq)` already exists and defaults to
`CommError::Unsupported` with a message naming the kernel launcher and the registered proxy. The
device owner wires, in this order:

1. allocate the one-shot slot with `DeviceBuffer::new(address, bytes)` (8-byte aligned, whole bf16
   elements);
2. publish `DispatchAddresses { slot, local_row, peer_rows }` to the peers (control plane);
3. build `Dispatch::new(rank, slot_registration, exchange, release, peer_epoch)` (non-zero
   release/epoch; zero means "not staged" to the kernel);
4. launch the RoCEnante kernel, then hand the same buffer to `all_reduce_bf16_device`.

The forward owner only needs Level 1's `TpCollective` surface; it does not touch slots or the
kernel. The contract between the two is: same bf16 bytes in, same canonical bits out, no host
copy, one `Dispatch` record per collective call.

### Level 3 - NCCL for >2 MiB (after B5 + device buffers; "Path N", not primary)

Path T is the primary path (`docs/build/tp4-design.md` §4.3); Path N is the fallback for
payloads that exceed the one-shot cap. Bootstrap is caller-owned: one rank allocates a unique id
(`NcclApi::unique_id`), the control plane broadcasts the 128-byte id, every rank calls
`comm_init_rank_config(world, rank, id)`. The data call is
`comm.all_reduce_bf16(send_ptr, recv_ptr, count, NcclRedOp::Sum, stream)` inside
`group_start`/`group_end`. Determinism: **none claimed**; the call site must be a code path the
spec says is tolerant, and the route must be recorded (backend string) in telemetry.

## 6. Benchmark harness and runner

### Binary

`crates/sparkflash-comm/src/bin/ws-b-allreduce-bench.rs` (built with
`cargo build --release -p sparkflash-comm --bin ws-b-allreduce-bench`; add `--features rdma`
for the fabric path). Default sweep: `decode_hidden_5120` (10,240 B), `decode_60kib`
(61,440 B), `decode_360kib` (368,640 B), `oneshot_2mib_boundary` (2,097,152 B).

~~~bash
# in-process reference: real rank threads and the real canonical reduction, no fabric
ws-b-allreduce-bench --transport loopback --json

# the fabric run (four ranks; the runner does this for you)
ws-b-allreduce-bench --transport rdma --rank 0 --world 4 \
  --peer 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
  --bytes 10240 --samples 200 --warmup 20 --gate-us 25 --json

# ticket B5 acceptance: is a 2.28+ libnccl loadable?
ws-b-allreduce-bench --check-nccl
~~~

Exit codes: 0 measured, 2 usage, 3 unavailable (no feature/library/transport; names the ticket),
4 exchange failure, 5 measured but the gate failed. **A loopback run is always `fabric=false`,
`gate_applied=false`, and is never a substitute for the gate.**

### What is timed

- **cold**: exactly the first exchange after the transport opened (n=1). It includes connection/QP
  setup and first-touch, so it is a bring-up number, not a steady-state one.
- **hot**: `--samples` exchanges after `--warmup` warmup exchanges; report p50, p90, p99, min,
  max. 200 samples is the default; p99 below ~100 samples is noise and should not be quoted.
- Per size, cold is taken before warmup, so warmup cannot launder the cold sample.
- Gate: p50 of the **decode rows only**, only when `fabric=true` (the RDMA transport actually
  measured). Rows are gated iff their label starts with `decode_` and the run is a fabric run.

### Runner

`scripts/ws-b-allreduce-bench.sh` (subcommands `preflight build selftest stage run collect cleanup
all`). It refuses to run while a GPU process or the `:8000` listener is present (the live TP4
vLLM deployment holds ~111/121 GiB per node), never stops or restarts vLLM, and only ever kills
`ws-b-allreduce-bench` processes. That deployment also serves this session's agent harness, so a
window needs explicit human authorisation and a restore path (see
`docs/build/ws-b-infra-tickets-20261021.md` section 3.1) - the refusal is why no number exists. Control port defaults to 18516 to stay clear of the
probe/gate runner on 18515. Evidence lands in
`/home/user/phase2-evidence/ws-b-20261021/<RUN_ID>/`: `preflight.txt`,
`bench-rank{0..3}.json`, `bench-rank{0..3}.err`. The 2026-10-21 16:36 UTC preflight recorded
MTU 1500/1500 on both rails, `nvidia_peermem` absent, one GPU process per node, `:8000` on rank
0, f1 rails ACTIVE and f0 twins DOWN: a run today would be refused by design.

### Report schema (stable keys)

`schema` (`sparkflash-wsb-allreduce-bench/1`), `transport`, `fabric`, `world`, `rank`,
`samples`, `warmup`, `peers`, `cold_note`, `gate_p50_us`, `gate_applied`,
`gate_passed` (null when not applied), `gate_rows`, `measured`, and `rows[]` with
`label bytes elements backend cold_us cold_samples min_us p50_us p90_us p99_us max_us samples
gated`. The report is the artifact of record; prose summaries copy from it and cite its path.

## 7. The gate, exactly

- **Gate:** p50 <= **25 us** on the decode rows of a **fabric** run (task-2 acceptance).
- **Doctor's separate bar:** `crates/sparkflash-cli/src/doctor.rs` uses 10 KiB and
  `MAX_ONESHOT_AR_LATENCY_US = 22.0` (strictly less than). That is a different, older, stricter
  check; do not conflate the two in a report.
- **Measured today (2026-10-20 bring-up, host buffers, MTU 1500, GPUDirect off):** 10 KiB one-shot
  All-Reduce median 36.0-37.1 us dual rail, 29.5 us single rail (min 25.6 us), 2 MiB 8.76 ms.
  Evidence: `/home/user/phase2-evidence/window-2026-10-18/rdma-t35-2026-10-19/` as cited by
  `docs/build/rdma-bringup.md`. **The gate was not met; the branch inherits that gap, it does
  not fix it.**
- **Status of the 25 us gate on this branch: not-yet-measured.** No code change can claim it. Only
  a fabric run of the bench, in a free window, with `fabric=true`, can.

## 8. Order of experiments once a window exists (one change at a time)

Each row is a single hypothesis with a single change and a before/after capture in the same
evidence directory. Do not batch them: the gap is 11-12 us over a 25.6 us floor and batching makes
attribution impossible.

| # | Change | Expected effect (hypothesis, unmeasured) | Ticket |
| --- | --- | --- | --- |
| 1 | baseline re-run at MTU 1500, host buffers | reproduces 36-37 us, validates the harness | - |
| 2 | MTU 9000 hosts+switch | removes fragmentation, fewer descriptors | B1 |
| 3 | `nvidia-peermem` loaded, device-buffer seam | removes host staging copy | B2 + device seam |
| 4 | single rail vs dual rail at the same size | the 2026-10-20 single-rail 29.5 us suggests rail count matters | B3 |
| 5 | trim control-plane/rendezvous work from the timed path | the floor is 25.6 us; sync overhead is a candidate | comm + engine |

The report for each run states: route backend, fabric flag, cold/hot split, p50/p90/p99, gate
applied or not, and the evidence path. A run next to a serving deployment is labelled "busy slot"
and is not a gate run.

## 9. Risks and blockers (with owners)

| Risk | Impact | Owner / ticket |
| --- | --- | --- |
| MTU still 1500; DF-8972 fails | one-shot decode likely stays above 25 us | B1 (root + switch admin) |
| GPUDirect off | device path impossible; host staging cost stays | B2 (root) |
| host NCCL absent; 2.28.3 only in the image | >2 MiB path unusable from host processes | B5 (root) |
| device seam returns Unsupported today | no device-buffer gate; host path is the only measurable one | device-lead (S5.3) |
| NCCL path is not bit-identical | cannot serve any seam with a determinism requirement | this doc §4; tp4-design §4.3 |
| 2 MiB capture cap vs prefill >2 MiB | must chunk or declare a non-bit-identical contract | device/forward owner + spec |
| live TP4 vLLM occupies the nodes (~111/121 GiB) | no clean slot; busy-slot numbers are invalid | serving owner + Lead window |
| f0 twins DOWN / rail naming drift | code/spec reference a dead rail; dual-rail assumption unproven | B3 (Lead/spec owner or root) |
| SM clock check reads a hardware max (3003) that a lock cannot change | doctor check 4 cannot pass as written | B4a (spec owner) |
| loopback numbers look large (debug build, host exchange) | risk of misquoting them as fabric | this doc; bench labels `fabric=false` |

## 10. What this branch changed, and what it did not

Changed (write scope `crates/sparkflash-comm/**`):

- `src/dispatch.rs` (new): the pure size policy and its refusals, 6 unit tests.
- `src/nccl.rs` (new, unix): dlopen binding, 2.28 floor, ABI pins, injectable loader, tests for
  absent library / missing symbol / old version / bad rank.
- `src/microbench.rs` (new): plan, measurement loop (cold then warmup then samples), gate, JSON
  and human reports, loopback cluster harness, RDMA and NCCL paths, tests.
- `src/bin/ws-b-allreduce-bench.rs` (new binary) + `Cargo.toml` `[[bin]]` entry.
- `src/lib.rs`: modules and re-exports; both dispatch methods now use the shared policy
  (behaviour unchanged).
- `scripts/ws-b-allreduce-bench.sh` (new runner), `docs/build/ws-b-infra-tickets-20261021.md`
  and this file.

Not changed: the forward loop, `TpCollective`, `parallel_tp.rs`, the device seam
implementation, kernel pins/ABI pins, the probe contract (`rocenante-probe` output shape is
untouched), any node configuration, git state (the Lead commits).

## 11. Exact next step for the Lead

1. Read this file and `docs/build/ws-b-infra-tickets-20261021.md`; decide B3-A (f1 standard) and
   B4a/B4b (check definitions) - those are one-line decisions, no root.
2. Hand B1 and B2 to whoever owns root and the switch; B1's switch half gates B2's measurement.
3. Have the device/forward owner land Level 1 (§5) so the engine can call the one-shot path today
   with host buffers, then Level 2 for the device path.
4. When a free window exists, run `scripts/ws-b-allreduce-bench.sh all`; collect
   `bench-rank*.json` and report p50/p90 for `fabric=true` against the 25 us gate, labelled
   measured or missed, with the evidence path.
5. Do not quote the loopback self-test, the 2026-10-20 bring-up row, or any number from this
   document as the WS-B result. The only WS-B result is a fresh fabric run.
