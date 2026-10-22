# Whole-layer device port — the one branch (2026-10-22, user direction)

Status: **the plan of record.** Supersedes the host-seam optimisation track and the device-chain
harness. One branch, one goal: a decoder layer that runs entirely on the GPU, with no copies back to
the CPU, tree reductions on device, and the cross-rank all-reduce on the GPU.

## Status at a glance (updated 2026-10-22)

- **Branch**: `win/layer-device` is the plan of record and the integration branch for the port. The
  published lean-loop number stays **179.81 ms/step**; nothing from this branch is published, and its
  intermediate commits are allowed to be slower.
- **Landed**: the loop cut - the device slab with its spans and ownership rules, the stage order (19
  contract stages on this branch after the 54784974 merge, 20 once the ref_select line lands) with two
  collectives per layer, per-position observation points replaced by per-run counters, the arm
  opt-in through `SPARKFLASH_LAYER_DEVICE=1`, and refusals that name the stage, the missing kernel and
  its signature; the row-norm checked binding and the `KERNEL_CONTRACTS` table. On `task/gpu-reduce`:
  `RunReduce`, NCCL vendored and dlopen'd, and a bench that measures the port's own call path.
- **In flight**: the W8A8 activation prologue and the mHC/FFN seam kernels (driver-fixer); the first
  device-only prefix, `query_path` + `attention_projections` + `attention_body`, with the device
  row-norm's consumers (gemv-f32-impl); the kv-path device cache - host semantics written down, the
  interface and the three kernels behind it (device-lead, gemv-f32-impl, comm-lead).
- **Blocked**: `nvidia-peermem` needs root, so GPU-initiated RDMA stays a flag with a typed refusal; the
  fabric measurement needs ~3 idle node-minutes and the user's go-ahead.
- **Next gates, in order**: the first device-only prefix runs (stage refusals turn into numerics); one
  full device-only layer runs; the default flips to the device arm; then graph capture and the GEMV
  re-tile. Only after that does this branch produce a number worth publishing.
- **Read with**: *Steady-state invariants* and the *All-reduce contract* below. They are acceptance
  conditions for every slice, not commentary.

## Why the track changed (the user's four decisions)

1. **Stop optimising the CPU seams.** The measured host floor of the lean loop is ~175 ms/step and the
   tree sits at 179.81; the remaining host blocks are reductions whose *arithmetic* is trivial and
   whose cost is spawn/join and socket exchange. Three transport arms measured on the fabric (spawn
   186.89 < sequential read 196.00 < parked pool 219.11) said the transport is already at its local
   optimum. Micro-slices in the seams now buy fractions of a millisecond and add review surface.
2. **The correctness gate is tolerance-based** (D54, merged): greedy tokens identical plus declared
   logits bars (top-1, top-5/50 overlap, max/mean |dlogprob|, KL over the top-100) against a frozen
   reference artifact of the bit-exact CPU twin. Byte identity survives only as the strict mode. This
   is what makes a device tree-sum admissible at all.
3. **One branch moves the entire layer**, tree reductions included, no partial-port measurement.
   Regressions on this branch are expected and allowed; it is not a speed-row branch and its
   intermediate commits are never published.
4. **The all-reduce runs on the GPU** - NCCL or GPU-initiated RDMA - and the fabric admin items
   (MTU, peermem) ship now. Graph capture and the GEMV re-tile come *after* the layer is whole.

## The target: one layer, device-resident, no D2H

Per layer, today's lean loop does, in order: attention projections (device kernels + host quantise),
attention body (device), output projection (device GEMV + host reduce machinery), mHC seams
(host SIMD), MoE router (host), gate_up + down (device kernels with host-side combine), shared expert
(device), FFN seams (host SIMD), plus 12 D2H observation points and two cross-rank exchanges whose
machinery is 29.5 ms/step. The port removes, in one cut:

- every host quantise (activation fp8 moves into the kernel's prologue or a device pass);
- every mHC / FFN seam (host SIMD becomes a device kernel; the 80 KiB rows never leave the device);
- every D2H observation point in the steady-state loop (records move to per-run counters, not per
  position);
- the host reduce machinery (spawn/join, socket exchange) - replaced by a device reduce;
- the MoE host combine and router host pass.

The layer's boundary is a device pointer: residual in, residual out. Nothing in the loop crosses PCIe
except the rank's own kernels and the fabric exchange.

## The all-reduce

Two candidate transports, decided by measurement on the fabric, not on loopback:

- **NCCL**, vendored as a shared library and dlopen'd (no system install, no root), one communicator
  per run, one ring reduce per layer. This is the lower-risk path: proven topology handling, and it
  can fall back to its socket transport while the RDMA path is unavailable.
- **GPU-initiated RDMA** with GPUDirect: the kernel writes peer device memory directly over the
  fabric. This is the fastest and the least infrastructure, but it needs **nvidia-peermem loaded**
  (peer-memory registration), which is currently the one blocked item - see the peermem section.

The arm that ships is the one whose measured per-layer cost beats 29.5 ms/step of host machinery by a
margin worth its complexity; the tolerance arm for both is stated in D54 terms: a 4-rank f32 tree sum,
same bf16 output rounding, same exchange bytes.

## Fabric admin status (2026-10-22)

- **MTU: already done and verified.** All four nodes carry `mtu 9000` on both fabric interfaces
  (`enp1s0f1np1` carrier 192.0.2.80/.84/.59/.64 and `enP2p1s0f1np1` 192.0.2.85/.88/.63/.68).
  Nothing to ship; recorded here so it stops being an open item.
- **peermem: present, stale, blocked on root for the rebuild.** The module is in the kernel tree
  (`/lib/modules/7.0.0-1019-nvidia/updates/dkms/nvidia-peermem.ko.zst`) and DKMS has a build for this
  kernel, but the kernel rejects it: `finit_module` returns EINVAL, and with
  `MODULE_INIT_IGNORE_VERMAGIC` it returns ENOEXEC - i.e. the module's symbol CRCs are older than the
  running kernel image (same version string, rebuilt kernel). A control module (`binfmt_misc`) loads
  fine through a privileged container, so module loading itself works; this module specifically needs
  a rebuild against the running kernel's `Module.symvers`. Our account is in the `sudo` group but
  sudo needs a password and root SSH keys are not installed, so the rebuild has to run as root. The
  one-line command is in the night record.

Until peermem loads, the GPU-initiated RDMA arm is not buildable; NCCL over its socket/RoCE fallback
is, so the port starts there and switches arm only if the fabric measurement says it must.

## Sequencing

1. **Fabric and communicator**: NCCL vendored + `dlopen` + one all-reduce per layer, proven on the
   four nodes against the current host reduce (tolerance arm, D54).
2. **Layer port**: the steady-state layer moved whole, seams and quantise included, D2H removed.
   Intermediate commits may be slower than 179.81; nothing is published from this branch.
3. **Then** graph capture of the device-only layer, and the GEMV re-tile against the device kernels'
   measured bandwidth.

## What this branch does not do

No host-seam micro-optimisation, no partial-port rows, no publishable numbers until the layer is whole
and the branch is measured as a whole. The lean-loop number stays 179.81 until then.

## Steady-state invariants (comm-lead's review, accepted 2026-10-22)

Every item below is a way this port could silently re-acquire a host readback. They are acceptance
conditions, not style notes.

1. **Attention output reduce writes a device row.** The reduced delta row and the row the residual add
   consumes both stay device-resident; no `copy_dtoh` of the reduced row in the steady state. The
   wording above is corrected accordingly: the block is not "replaced", its result simply never becomes
   host memory.
2. **Evidence gates are off in the steady state.** `layer_evidence_readbacks()` - per-layer
   `partial_l2`/`reduced_l2` and the gate/up/codes/scales stage captures - is the source of today's 12
   D2H points. A slice that "removes observation points" must also neutralise these gates, or a debug
   flag silently restores the entire readback path.
3. **The MoE combine consumes the reduce's device output.** `moe_combine_rank_sum` /
   `moe_combine_routed_f32` do not stay host-side even temporarily - if the combine is host-side, the
   reduced row crosses PCIe again.
4. **Trace, boundary and parity capture are dev-gated.** `write_boundary_trace_row(..., residual_out:
   &[f32])`, the trace-ring readbacks and the `native_dense_parity`/`parity.capture` path are
   per-layer host-visible; in the port they run only under a dev gate and the steady state records
   counters (see `DeviceReduceStats`).
5. **The one legitimate readback is named.** Final logits, once per position, for the numerical gate
   (D54), plus per-run counters. "Nothing crosses PCIe" is true except that one, and naming it stops
   someone "optimising" the gate away later.
6. **Collective-sequence coherence.** Every rank enqueues the same collectives, in the same order. No
   rank-conditional path (twin, evidence, debug, early return on a recoverable error) may skip a reduce
   on one rank - the communicator hangs. Today's per-rank MoE routing diverges; the collective itself
   stays unconditional and the port keeps it that way.
7. **Payload is frozen.** The declared arm is the **bf16, 10,240 B per-rank** payload (5,120 bf16 = the
   attention delta = the MoE routed delta). If the port moves the delta row to f32 device partials
   (20,480 B), that is a *different arm*: re-declared and re-measured before it gates anything.

## All-reduce contract (agreed with comm-lead, 2026-10-22)

- **One communicator per run**, for both collectives and all payload sizes.
- **One-time warm-up at setup**: 2-3 dummy reduces on the caller's stream plus one stream sync, outside
  the step. Consequence for attribution: **layer 0 is not steady state** and must not be quoted as such
  while NCCL ramps.
- **No completion query and no per-collective sync.** The reduce is enqueued on the caller's stream and
  ordering is the stream's job; the only sync is the step boundary. Exposing a per-layer sync for a timer
  would re-acquire exactly the cost the port exists to remove. NCCL async-failure status
  (`ncclCommGetAsyncError`) is checked once at the run boundary - failure detection, not completion.
- **ARMED shape if peermem is unblocked**: cargo feature `gpu-rdma-reduce` (default off), runtime
  `--reduce-arm nccl|gpu-rdma`, the same `DeviceReduce` type and signatures, the transport an enum
  chosen at init with a **typed refusal** when the selected arm is unavailable (peermem absent, no verbs
  device) - never a silent fallback to the other arm. The tolerance declaration does not change: same
  4-rank f32 tree sum, same bf16 round, same payload; only the wire changes. Default stays `nccl` until
  the fabric measurement says otherwise.

## The D54 tolerance arm, declared before any number exists

> The arm is a four-rank bf16 Sum all-reduce on NCCL - one f32 accumulation per element and one bf16
> round on the result - over the same exchange payload, between the same four peers, in the same order of
> collectives per layer as the host path. The canonical reference is the current rank-order
> f32-accumulate-then-round path (`sparkflash_comm::reduce`), which remains the strict-mode contract.
> The measured deviation of this arm is the benchmark's differing-element count and max ULP on the
> 10,240 B payload. The row-level proof is the numerical gate on a whole-layer row: greedy tokens
> identical, top-1 1.00, top-5 >= 0.80, top-50 >= 0.96, max |dlogprob| <= 2.0e-2, mean <= 2.0e-3,
> KL <= 1.0e-4, all finite - with the fixture's tightest top1-top2 margin, 0.0038 nats at position 4, as
> the position that decides it. NCCL's reduce order inside its ring is not the canonical rank order.
> That is the arm, not a defect: D54 admits it, the benchmark measures its deviation, and any claim
> beyond the declared bars would have to go back through the canonical path.

Recorded verbatim in `crates/sparkflash-comm/src/device_reduce.rs` module docs (branch
`task/gpu-reduce`) so the declaration exists on the record before the measurement.

## What has landed (2026-10-22)

- **Loop half** (`task/layer-host-cut` @ `a1dd55cd`, rebased on `e391dafd`): `crates/sparkflash-cli/src/layer_device.rs`
  is the contract - one device slab carved into 13 spans with an ownership table and six rules
  (residual never host-visible inside the loop, no stage allocates, no stage returns bytes, stage order
  is the dependency, reduce queued not waited, the host arm is not the lean loop) - plus the 16-stage
  order, the `LayerStages` trait (the order lives in `run_layer`) and `STAGE_CONTRACTS`, a
  machine-readable (stage, kernel, signature) table whose refusal a run prints verbatim:
  `layer device stage mhc_entry: kernel mhc_entry_bf16 is not landed on this branch; contract: (...)`.
  The 13 per-layer-position observation points are now per-run counters (no snapshot, no sync, no D2H),
  and the steady state is `LayerStages::run_layer` with one H2D per position (`begin_position`) and two
  D2H at the head boundary (`residual_out`, `attn_pre_mix_out`).
- **The device arm is opt-in** (`SPARKFLASH_LAYER_DEVICE=1`) until the layer runs end to end; the host
  sequence stays the default, because a binary that refuses at layer 0's `mhc_entry` must never be the
  default on a branch something might run for a row. Flipping the default is a single commit made only
  when the last seam kernel is landed and one full layer runs device-only.
- **Seam kernels** (`task/layer-seams`, in flight): `mhc_entry`/`mhc_apply`, `ffn_seams` and
  `quantise_prologue`. The mHC device kernel from the rejected port (`kernels/mhc_hc_proj_sm121.cu`)
  is the starting point, not a retry: it lost under the old design because its readback cost +46.4 ms
  per call, and under this design nothing is read back; the other constraint that made it slow - the
  host summation order - is now a declared D54 tolerance arm rather than a prohibition.
- **all-reduce contract confirmed** (comm-lead): `(stream, in, out, bytes, handle)`, device addresses,
  run-owned handle, layer-owned spans, enqueue-only, no per-call workspace.
- **Load-path diagnosis** recorded in `ws-host-bound-night-20261022.md` with its correction: the rows
  use the serial streaming walk, the FNV-1a digest is ~121 s of the 254 s, and the parked fix measures
  254 -> ~100-110 s. Parked behind this port.

### Contract amendments from the transport cross-check (2026-10-22, comm-lead vs the landed loop cut)

Reading `task/layer-host-cut` against the communicator found five defects in the contract as landed.
The first is a correctness hole, not a shape nit:

1. **Two collectives per layer, not one.** The lean loop makes two cross-rank 5,120-bf16 reduces per
   layer: the attention delta (`attention_output` -> `all_reduce`) and the MoE routed delta
   (`moe_routed` -> `moe_reduce`, which `ffn_epilogue` consumes as `moe_delta`). The contract listed
   only the attention one and gave `moe_routed`/`ffn_epilogue` no stream or handle, so as declared the
   MoE delta would have stayed rank-local and the layer would have been wrong on every rank but one.
   Fix: stage `all_reduce_moe` after `moe_routed`, same call shape, out into a **new** span
   `workspace.moe_reduced_bf16` (`dim * 2`). Not `reduced_bf16`: `attention_residual` consumes that
   first, so reusing it is a latent aliasing bug waiting for a reorder.
2. **The call shape is a host call, not a kernel launch.** `device_all_reduce_bf16(cudaStream_t, const
   void*, void*, int, void*)` is the *ARMED RDMA* arm's shape - a real kernel writing peer memory. NCCL
   is a host API that enqueues onto a stream; there is no cubin to launch, so under that name the stage
   would hit `not_landed` forever, and if the RDMA kernel ever landed the NCCL loop would silently
   select the wrong arm. The stage entry is `comm_all_reduce_bf16(stream, in, out, count, handle)`;
   `device_all_reduce_bf16` stays as the RDMA arm's contract only.
3. **`count`, not `bytes`.** The communicator takes an element count; bytes would force a `/2` in the
   wrapper, which is exactly where a 2x error hides. Both collectives are 5,120 elements.
4. **The handle is an opaque `void*`.** `DeviceReduce<'api>` borrows the loaded `NcclApi`, so it cannot
   be a `void*`-shaped field in the run context. The owned form lands on the communicator branch:
   `RunReduce` owns the `NcclApi` plus the raw `ncclComm_t` (`Drop` destroys it),
   `open(world, rank, id) -> *mut c_void`, `close(handle)`, `reduce(handle, stream, in, out, count)`.
   The contract keeps `void* handle` exactly as declared.
5. **Init needs two lines the contract did not have.** (i) *Unique id*: rank 0 generates 128 opaque bytes
   and every rank must receive the same bytes before `ncclCommInitRank` - at open, rank 0 broadcasts the
   id over the run's control channel, or `open` takes a `const void* unique_id`. (ii) *Warm-up*:
   `comm_warm_up(stream)` after open runs the 2-3 dummy reduces plus one stream sync at setup, so the
   first layer is not the outlier.

Two things already match and stay as they are: the spans (`attn_out_bf16` -> `reduced_bf16`, out of
place) and the queued-not-waited semantics. `workspace.reduce_ws` is unused by the NCCL arm (it
pre-sizes the RDMA tree only); NCCL's internal buffers are allocated at init/first collective and are
charged to the run's arena at open, never to a per-layer span.

### Pinned bootstrap form (decided 2026-10-22 after the communicator landed)

The unique-id question is closed in favour of the loop owning the bootstrap:

- `open(world, rank, unique_id: const void*)` - the 128 opaque bytes are an **argument**, not something
  the communicator fetches. The comm crate exposes `unique_id()` (rank 0) plus its hex helpers and
  nothing else; the loop calls it on rank 0, broadcasts the 128 bytes to ranks 1..3 over the control
  channel it already owns, and all four pass the same bytes to `open`.
- Rationale, recorded because it will be asked again: putting the broadcast inside `open` would make the
  communicator's peer set depend on a private channel nobody else can see. The loop is the only side that
  knows the peer table, so the loop does the broadcast.
- `open` loads the library itself, so a missing libnccl is a typed `LibraryAbsent` and never a fallback.
  A null handle refuses with `Config` rather than dereferencing; `close(null)` is a teardown no-op. The
  loop stores exactly one opaque word.

**Warm-up** is `warm_up(handle, stream, scratch, count)`: it enqueues three in-place reduces over a
caller-provided device span and does **not** synchronise (the comm crate has no CUDA dependency), so the
port's single setup sync completes it - that is the one sync the plan already grants at setup. Pass the
layer's real payload count (5,120 bf16 elements) so the same path is warmed; any device span of that
many elements works as scratch, and `workspace.reduce_ws` (2*dim*4 = 40 KiB) is large enough to serve.
Warm-up collectives are setup, not layer work, and are **not** in `DeviceReduceStats`, so a run's counter
is exactly layers x collectives.

**Arena accounting**: `DeviceReduceStats.arena_bytes` is recorded at open (default
`NCCL_ARENA_RESERVE_BYTES` = 16 MiB, overridable). The run's memory model adds `arena_bytes(handle)` to
the arm's cost and the reduce record states the same number - one struct, so the two accounts cannot
drift. NCCL allocates its channels at init only.

## Frontier and ownership (2026-10-22, after the loop cut merged at `9d5a5bfa`)

The contract is merged into the branch, and the arm is **opt-in** (`SPARKFLASH_LAYER_DEVICE=1`); the host
sequence is the default until the flip commit, which is gated on one full device-only layer. The
refusal prints the missing kernel's signature verbatim **and** the host step the stage still carries
(`STAGE_BLOCKERS`), so the frontier is stated in the contract rather than discovered at run time. No
"run what is landed" mode exists or will: a half-device layer that silently mixes arms is the class of
bug the invariants exist to prevent.

What still blocks the first device-only prefix, and who owns it:

| blocker | unblocks | owner |
| :-- | :-- | :-- |
| **W8A8 prologue** (activation quantise on device) | `attention_projections`, `query_path`, and the quantise half of every projection stage | driver-fixer (`quantise_prologue`, highest priority) |
| **device row-norm** (`rms_norm_qk_gemv`) | `query_path`; second use in `attention_projections` | gemv-f32-impl (kernel + its contract entry) |
| **`kv_path`** - no device kernel exists today (host `rms_norm` + `rope_rotate` + the window fp8 round-trip) | `kv_path`, and the last host seam in the attention half | gemv-f32-impl, contract entry first, then sized before it is scheduled |
| `mhc_entry`/`mhc_apply`, `ffn_seams` | the seam half of the layer | driver-fixer (the rejected port's kernel is the starting point; its two disqualifiers - readback cost and the host summation order - are gone) |

Priority: prologue and row-norm first, because together they make `query_path`,
`attention_projections` and `attention_body` land device-only in one cut - that prefix is the thing to
have in hand when the next node window opens.

## Row-norm landed, kernel table, and the kv_path split (2026-10-22, branch tip `6b77b9d9`)

**Row-norm** (`rmsnorm_sm121` + `rmsnorm_launch`) already existed as a kernel and launcher; the port's work
was a **checked binding**, not a rewrite: the cubin is loaded once at `DeviceLayer::open` from the loop's
kernel dir (a missing dir is a named refusal at open, not a mystery at layer 0), and the launch goes
through named fields with span-size guards that refuse rather than overrun. Its consumers
(`query_path`, `attention_projections`) sit behind the activation prologue, so it is marked dead-code
with that exact reason until the prologue lands. *Not numerically validated*: compile- and syntax-probed
only; the numerics gate is a node row and nobody may quote it as validated before one exists.

**`KERNEL_CONTRACTS`** is now the machine-readable kernel side of the contract - signature, owner and
consuming stage per row - for `rmsnorm_sm121` (landed), `w8a8_prologue_e4m3` (driver-fixer), the three
kv-path kernels (unassigned at the time), and `comm_all_reduce_bf16` (comm-lead, landed). A test pins the
row set, the owners, and that the kv path is exactly three kernels, so a kernel cannot be dropped or
double-declared without a red test.

**`kv_path` is a design piece, and it is split so it cannot land half-done.** It is a sequence of three
kernels - `rope_rotate_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16` - and the line count is not the
cost: the KV cache is **host-resident today**, so the change is kernels + launcher + **the cache moving to
device ownership** with the window-consuming seam following it. Ownership split, decided so there is
exactly one author at the coordination point:

- **gemv-f32-impl writes the interface** into `layer_device.rs`: the device-cache handle type, the three
  signatures, and the ownership rule - the device arm owns its own cache, the host arm keeps the
  host-resident one, and there is **no shared mutable cache between the two arms**. Two arms, two owners,
  one contract.
- **device-lead implements** behind that interface in dependency order (`kv_append_bf16` first - the
  ownership move - then the window round-trip, then rope, which has the most numerical freedom), with the
  same launch/span-guard discipline as row-norm.
- Explicitly rejected: making the host arm read the device cache, or a "temporary" shared cache. A half-
  finished cache-ownership change is the failure mode this split exists to prevent.

**Collective counter as the rank-coherence instrument.** The run record carries
`layer_collectives` = 2/layer, 5,120 elements, expected `layers x 2`. It is a good invariant because it
fails loudly if a rank skips a collective - a skip *hangs* NCCL rather than failing a check, so a
host-visible counter is the only trace that all four ranks enqueued the same sequence. The run boundary
compares it against the expected value and reports a mismatch loudly instead of printing both numbers and
moving on.

## The KV cache interface, frozen (`DeviceKvCache`, branch tip `ee0a0daa`)

`DeviceKvCache { device, capacity_rows, row_values, n_layers }` - one opaque word, layers as offsets
through `layer_span(layer)`, no per-layer table and no host mirror; `bytes()`/`layer_bytes()`/
`append_values()` for allocation and geometry; created by `DeviceLayer::open` **beside** the slab (the
slab is the layer's working set, the cache is the run's history), 256-byte aligned, refusing a zero shape
or a misaligned base by name. Ownership is in the type's doc and repeated as rule 7 in the module header:
device arm owns it for the run, host arm keeps the host-resident cache, no shared mutable cache, and
**no row is ever read back to the host in steady state** - run-boundary numbers only.

**Append visibility, as frozen**: a row appended for position `p` is visible to the attention body only
when the loop's slice-copy into `AttentionBody`'s window buffers is enqueued on the same stream, after the
append, in position order. Stream order *is* the publication - no cross-stream publication, no host-visible
copy, no synchronise. A kernel reading `p` may assume the current window's rows were appended before it;
rows later than `p` are not to be read whatever the ring holds. This is the one sentence in the interface
that cannot be inferred from the code, because it is a property of the loop, not of the type.

The three kernel signatures are unchanged from `KERNEL_CONTRACTS` (`rope_rotate_bf16`,
`window_quant_fp8_bf16`, `kv_append_bf16`); their owner field now names the implementer. A signature that
needs changing comes back to the Lead before it changes, because two people code against this file.

**Process note worth keeping**: a merge cannot drop a line - only a dirty working tree can make a gate
pass locally and a commit lack the fix. When a gate disagrees with a pushed tip, check `git replay`
semantics (or simply `git show <commit>:<file>`) before blaming the integration; the fix is to run the
gate on a clean tree and to keep the repair in its own commit, which is what happened here.

## Build capability: the dev host has nvcc (2026-10-22)

The port's kernels do **not** need a node to be built. `/home/user/cuda-tk/bin/nvcc` is CUDA 13.0
(V13.0.88) extracted under the account, no root required, and it compiles `sm_121a` **PTX and cubin**
from a real repo kernel (`crates/sparkflash-kernels/kernels/rmsnorm_sm121.cu` -> 75 KB PTX, verified).
Use `SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc`; if `build.rs`'s probe trips over this host's
gcc 16 C++ headers, `-std=c++20` fixes it (both `-std=c++17` and `-std=c++20` compile a real kernel
here, but the trivial probe TU fails under c++17 because the host's `type_traits` needs `char8_t`).

**Guardrail on the kernel baseline**: a regen on an **unmodified** tree must reproduce the recorded
hashes. If it does not, the local toolkit is not the one that recorded the baseline and the baseline
must not be regenerated locally - the kernels stay in `pending_kernels/` and the promote happens in a
node window. This is the difference between "we can build our own kernels three times faster" and "we
quietly changed the thing CI checks against".

Consequence for scheduling: the prologue, the seam kernels and the kv kernels can all be built and
probe-checked while the cluster is held, so a node window is needed for *running* the port, not for
building it.

## A defect caught in the frozen interface

`DeviceKvCache::bytes()` aligned the **row count** (`align_up(capacity_rows)`) instead of the per-layer
byte run, so it disagreed with `layer_bytes() = capacity_rows * row_values * 2` and over-allocated up to
~2x whenever `capacity_rows % 256 != 0`. Caught while confirming the kv geometry against the implementer
- the geometry itself was right, the allocation arithmetic was not. Fix: `layer_bytes()` returns
`align_up(capacity_rows * row_values * 2)`, `bytes()` is `n_layers * layer_bytes()`, and a geometry whose
run is not ALIGN-aligned is refused rather than silently rounded. The invariant
(`bytes(n, rows, values) == n * layer_bytes(...)`) is a test. Lesson worth keeping: an aligned-count
formula reads as plausible and only shows up as memory bloat on a window size that is not a multiple of
256 - exactly the sort of thing to check with an invariant rather than by eye.

## First device-arm run: the checklist (written before the window, so it is mechanical)

Nothing here is a speed measurement. The first device-arm run exists to produce either a refusal that
names the next kernel to write, or a numerically-gated result for one layer's worth of stages - and the
"is it faster" question is explicitly *not* asked until the whole layer runs.

**Before touching a node**

1. Branch tip is `win/layer-device` with the kernels the run needs actually in `kernels/` (not
   `pending_kernels/`), and `scripts/check-kernel-baseline.sh` passes locally with
   `SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc`.
2. `cargo clippy -p sparkflash-cli --lib --features cuda -- -D warnings`, `cargo fmt --check`, and
   `cargo test -p sparkflash-cli --lib` green on the exact commit that will be staged (committed tree,
   not working tree).
3. The aarch64 build is staged from that commit, and the staged binary's sha256 is recorded in the row
   name - the staged binary is what runs, never a local rebuild.
4. The arm is selected explicitly: `SPARKFLASH_LAYER_DEVICE=1`. Unset runs the bit-exact host reference;
   a row whose record does not say which arm ran is not a row.

**On the node, in order**

5. Preflight: zero `sparkflash-*` processes and MemFree >= 110 GiB on all four, then the clean-row cache
   drop (`posix_fadvise(fadvise_dontneed)` over the checkpoint shards) before every row - the device arm
   reads the same shards and the band is meaningless without it.
6. One short run first, with the **frozen argv** and `--forward-lean`, and expect either a refusal
   naming a stage + kernel + blocker, or a completed layer. A refusal is a result: record the stage it
   named and stop; do not add a fallback to make it run.
7. If it completes: run `scripts/numerical-gate.sh --run <row>` against the frozen reference and record
   the bars, the tightest top1-top2 margin, and the fixture key. A device-arm row that fails the gate is
   a port bug, never a tolerance question - D54 admits tree reductions, not changed arithmetic.
8. Run `scripts/check-collective-count.py` over the four per-rank records: all four must agree and equal
   `layers x 2`. Exit 2 (a record with no `collective_check`) is a stop, not a retry - a record that
   cannot testify is not a pass.
9. Only then look at the wall clock, and only as a *direction*, not a publishable number: the branch's
   intermediate commits are allowed to be slower, and one device-arm row is not a published result.

**What would make the run invalid**

- a row on the host arm reported as a device row (the record's `layer_device` line must match the env);
- a gate run against a reference frozen from a different fixture key;
- a shortened argv "to get it running" - the frozen argv is the measurement's identity;
- publishing any ms/step from this branch before the whole layer runs, however good the number looks.

**Then, and only then**: flip the default to the device arm (one commit), and the next gate is 150 ms/step
against the published 179.81 - measured as a whole layer, in a clean window, with the same gate.

## The seam kernels are compiled (2026-10-22, driver-fixer, local toolchain)

One image carries all three: `kernels/layer_seams_sm121.cu` with `w8a8_prologue_e4m3` (47 regs, 0
spills), `mhc_ffn_seams` (48 regs, 0 spills) and `mhc_entry_bf16` (46 regs, 0 spills), built with
`/home/user/cuda-tk/bin/nvcc` 13.0 for `sm_121a`. Baseline pin: `layer_seams_sm121`
`cubin_bytes=119312`, `ptx_membar_lines=0`, and - the part that makes the pin trustworthy - the 23
pre-existing pins are byte-identical after regen and the `kernel_baseline` test recompiles all 24 on this
host and reproduces every hash. That is the "local toolkit is the recording toolchain" case the guardrail
required, established with evidence rather than assertion. The CPU twin in `layer_seams_ref.rs` is
bit-identical to the engine's seam functions.

### ABI ruling: the seam kernels are 8 words, and there is one ABI owner

The kernels carry the ABI-uniform trace tail: `(normed, codes, scales, cols, rows, trace_ring, layer_id,
step_seq)` - 8 words, not the 6-word `PrologueLaunch::words()` that `layer_device.rs` emitted. Ruling:

- the **kernel ABI is not changed** (the pin against parsed kernel source exists and is worth more than
  the 6-word shape);
- **`PrologueLaunch::words()` is deleted**, not left unused: an emitter that produces a 6-word argument
  list for an 8-word kernel is a trap that fails only when someone calls it, and "unused" does not
  survive contact with a future reader;
- **one ABI, one owner**: `layer_seams_launch::launch_prologue` owns the argument list and pins it;
  `PrologueLaunch::new` stays as the span check and its doc points at the launcher. The stage body's flip
  is then one call, not an argument-order negotiation.

**Null trace ring means "write nothing"** - and it is pinned at the source level, the way
`ptx_membar_lines=0` is. In the steady state the loop passes no evidence ring; a seam kernel that runs
40+ times per position with an unconditional trace store would re-introduce exactly the host-visible
traffic the port exists to remove, and a null dereference would be worse. Layer id and step seq may carry
anything in the steady state; the ring pointer is the gate.

### Writer serialisation (why the rebase order is not arbitrary)

`tp4_forward.rs` has two writers' work in flight - driver-fixer's `DeviceLayer` kernel fields and
gemv-f32-impl's `PrologueLaunch`/`row_norm` additions - and both are additive, so a rebase resolves
cleanly but a *simultaneous* edit does not. Order fixed by the Lead: driver-fixer rebases onto the current
branch tip, pushes `task/layer-seams` and reports; gemv holds `tp4_forward.rs` until that push, spending
the interval on the weight accessor in `layer_device.rs` (a file only it writes), then rebases and takes
the stage bodies. Two writers on one file is the only way this branch produces a rebase that silently
drops half a change.

## Process: the push gate (2026-10-22)

One red commit reached the branch because a gate's failure was *printed* but not *acted on* - the classic
`cargo clippy ... | grep error && git commit`, which commits when `grep` matches. The fix is
`scripts/branch-gate.sh`: `set -euo pipefail`, **no gate is ever piped** (so its exit code cannot be
swallowed), it refuses to run on a dirty tree (a gate on a dirty tree can pass while the commit is red -
the second time this bit us), and it runs fmt, clippy `--lib` and `--lib --features cuda` at
`-D warnings`, the cli lib tests, and the kernel baseline when `SPARKFLASH_NVCC` is set (`--full` adds
the workspace suite). Usage: `bash scripts/branch-gate.sh && git push`.

The policy that goes with it: **a commit that does not build does not stay in the branch's history.** The
red commit was squashed away by its author before merge - it would otherwise survive into the MR, break
`git bisect`, and mislead anyone reading the log.

## KV: the logical length and the physical ring are two numbers

`kv_append_rule(cache_len, position, row_width)` takes the **logical** length: it refuses unless
`cache_len == position` (the rows already appended, i.e. the logical length *before* this position's
append) and `row_width == MLA_LATENT`. The kernel takes the **physical** ring: it writes slot
`position % capacity_rows` over a ring of `capacity_rows` rows. Both numbers are needed and they are not
interchangeable - conflating them is the wrong-epoch bug, where an append lands on a slot that a later
window read interprets as a different position. The order is **append, then read**: the released window
is `[p + 1 - window_rows ..= p]` with `window_rows = min(p + 1, 128)`, so it includes the current token
(engine `window_indices` decode: `filled = window.min(position + 1)`; m20's `window_count` counts the
current row; the port's reference staging is 1 window row at position 0). Row `p` must therefore be in
the ring before the window read at `p`; a read before the append is refused, not assembled sink-only.

## Kernel baseline: both new images, and the evidence that made the promotion legitimate

Two kernel images are now pinned with the same guardrail evidence, and the numbers are recorded here so a
later reader can re-derive them rather than trust the log.

**`layer_seams_sm121`** (driver-fixer, local build): `w8a8_prologue_e4m3` 47 regs / 0 spills,
`mhc_ffn_seams` 48 / 0, `mhc_entry_bf16` 46 / 0; `cubin_bytes=119312`, `ptx_membar_lines=0`; the 23
pre-existing pins reproduced byte-for-byte on this host.

**`kv_path_sm121`** (comm-lead, local build): `cubin_bytes=36656`,
`cubin_sha256=91e6d62e1f805ef937b5dea30353cb6e9c8f04f4785b81e48d3cc9223cd43224`,
`ptx_membar_lines=0`; one cubin, three symbols, no `KernelId` (`cubin_path` answers `KV_PATH_CUBIN` by
name, like `FLASH_MLA_CUBIN`).

**The promotion evidence** (this is the part that matters): on an **unmodified** tree at the then-tip,
`SPARKFLASH_NVCC=/home/user/cuda-tk/bin/nvcc scripts/regen-kernel-baseline.sh /tmp/base-measured.txt`
produced a file **byte-identical** to `baseline/kernel_baseline.txt` (empty `diff`), so the local toolkit
*is* the one that recorded the existing hashes; only then was the new entry generated. Toolchain recorded
with it: nvcc release V13.0.88, build `cuda_13.0.r13.0/compiler.36424714_0`, `sm_121a`, no flags beyond
the scripts' own. The check then reports `all kernels match the recorded baseline (V13.0.88, ...)`.

**Baseline count collisions are a real scheduling constraint**: each new kernel moves the count in four
counting places (`build.rs PROBED_KERNELS`, the check and regen script lists, the manifest) plus
`baseline.rs`'s pin count, three abi header assertions, `tests/kernel_baseline.rs`, and the
`routed_gate_up_swiglu_splitk_abi` NOW/AFTER_PIN constants. Two kernel branches arriving at once redo that
arithmetic under time pressure, so the rule is: **the branch that adds the second kernel owns the count
bump**, and the busier branch lands first.

**KV ring arithmetic** (for the record, next to `layer_device_memory` when the ring joins it): codes
`512` B/row + scales `16` B/row = `528` B/row; at 128 rows/layer that is `67,584` B/layer - already a
multiple of 256, so no padding is needed. `WindowRingLayout` owns that geometry so the arithmetic exists
once.

## The audit that paid for itself: four nonexistent weights, and a 20x accounting error

gemv audited its own stage body against the bind table and the kv ref before writing it, and the audit
itself produced the value:

**Four of the six mHC "weights" do not exist.** The plan builds only `layers.{layer}.hc_attn_base`
(F32 [4]) and `layers.{layer}.hc_ffn_fn` (F32 [2,2]). There is no `hc_attn_fn`, `hc_attn_scale`,
`hc_ffn_base` or `hc_ffn_scale`. Only the two verified rows go into `WEIGHT_NAMES`; the other four go in
only if driver-fixer names an exact bind key, and if they are derived at launch from base + fn then they
are **not weights** and must not be in the table at all. This is the second time a plausible-but-nonexistent
name was caught before it landed (`post_attention_layernorm.weight` was the first) - the check costs one
grep and saves a silently unbound tensor.

**Rope tables are per table class, so the record was over-reporting by 20x.** Compression is the only
per-layer dependence of the tables (`rope_table_class(config, layer) -> 0|1`), so **two uploads cover
forty layers**, not forty. `rope_table_bytes` is now `2 * 2 * max_positions * 128` and the test matches.
The number in `layer_device_memory` would otherwise have been wrong by a factor of twenty and nobody would
have noticed, because it is a record field rather than a measurement.

**Span audit, proved rather than assumed**: `KvPathRow::span_requirements()` gives `ring_bytes = R * 2048`
which is *exactly* `DeviceKvCache::layer_bytes_for(R, 2*head_dim)`, so the cache base is 256-aligned because
the stride is a multiple of 256 and the base is `ALIGN`ed - alignment by construction, not by hope. The
only genuinely missing span was `kv_latent`, already merged.

**One doc contradiction, fixed by its owner**: `KvPathRow`'s doc placed the stage call after the seam and
slice-copy, contradicting the frozen contract (group enqueued inside `kv_path`, window read before append).
Both are numerically identical because the window is rows before `p`; the doc is being corrected to say
that, so a contradiction becomes a proof.

## Two name checks, opposite errors, one rule

Within an hour the branch produced both directions of the same mistake:

1. **A plausible name that does not exist**: `post_attention_layernorm.weight` was in
   `WEIGHT_NAMES`; this engine never builds it. A suffix that matches nothing is silent.
2. **A fixture that made real names look absent**: an audit grepped `tp4_resident.rs:1693-1694` and found
   `hc_attn_base` as F32 `[4]` and `hc_ffn_fn` as F32 `[2,2]`, and concluded four of the six mHC names
   do not exist. Those two lines are a hand-built tiny tensor list for resident-bind **unit tests**. The
   plan builder at `sparkflash-engine/src/tensor.rs:199-216` builds all six, per `twin in ["attn","ffn"]`:
   `hc_{twin}_fn` F32 `[mix_hc, hc_dim]`, `hc_{twin}_base` F32 `[mix_hc]`, `hc_{twin}_scale` F32
   `[3]`, with `mix_hc = (2+n)*n = 24` and `hc_dim = n*dim = 20480`. The host decode path confirms it at
   run time (`forward/mod.rs:1507-1511` and `1774-1778`).

**The rule**: a name check goes against `TensorPlan::language_model` / the plan builder, **never** against a
unit-test fixture or a hand-written map. Both errors were caught by checking the other direction, which is
why the check is cheap only if it is the *right* check.

**Resolved geometry for the mHC triple** (per sublayer, released config): `fn` 24 x 20480 f32 =
1,966,080 B, `base` 24 f32 = 96 B, `scale` 3 f32 = 12 B. The entry kernel takes `(fn, base, scale)` per
sublayer and the ffn-seam kernel takes the FFN triple - the same three tensors the host path decodes.

**Mix-span spec** (for `workspace.attn_mix` and `workspace.ffn_mix`): f32, `(hc_mult*hc_mult + 2*hc_mult)
* 4` = `(16 + 8) * 4 = 96` bytes each, carved through the shared span helper (256 B alignment, no special
case), additive at the end of `LayerWorkspace` so no offset moves and `total_bytes()` grows by 512 B.
`attn_mix` is written by `mhc_entry` and read by `attention_residual` (post/comb) and `ffn_seams` (pre);
`ffn_mix` is written by `ffn_seams` and read by `ffn_epilogue` and the next layer's `mhc_entry` through
the rotated `residual.pre_mix`.

## The seam kernels are merged (2026-10-22, `6c5a1f06`)

`task/layer-seams` @ `440ee061` is in `win/layer-device`. The tip is the **minimal handoff**: the one
cubin (`kernels/layer_seams_sm121.cu`, three symbols), its baseline pin plus the 24-kernel plumbing in all
ten counting places, `layer_seams_launch` (typed launchers, each ABI pinned against the parsed kernel
source), `layer_seams_ref` (CPU twin), `cuda_driver::load_layer_seams`, and in `tp4_forward.rs` a
`DeviceLayer` that resolves the image at open (digest-checked against the pin) and launches
`quantise_prologue` through `PrologueLaunch::new` (span check) + `launch_prologue` with a null ring.
`mhc_entry`/`ffn_seams` still refuse by name - they need the mix spans and the six weight rows, which are
driver-fixer's follow-up (`seams-full` @ `52948cd7`) rebased onto gemv's commit. `layer_device.rs` was
untouched by this push; `PrologueLaunch::words()` is gone.

**Gate on the merged tip**: `bash scripts/branch-gate.sh` with `SPARKFLASH_NVCC` set - fmt, clippy both
ways, cli lib 376/1, and the 24-kernel baseline check, final line
`all kernels match the recorded baseline (V13.0.88, cuda_13.0.r13.0/compiler.36424714_0)`.

**Recorded hashes** (so a later reader can re-derive rather than trust): kernel source
`9a9d5079...`, cubin `119312 B / ead69ce6...`, baseline file `a26923f7...`; `-Xptxas -v` 47/48/46
registers, 0 spills, 20,720 B smem for the two mHC kernels, 0 smem for the prologue. The `-Xptxas -v`
figures come from a separate diagnostic compile because the flag changes the cubin digest - stated rather
than glossed, because a diagnostic number that silently differs from the pinned artifact is how pins stop
meaning anything.

**The null-ring guard is pinned where the cubin cannot help.** The trace compiles out, so `ptxas` sees no
difference whether the guard exists - the cubin digest is unchanged at `ead69ce6`. So the guard is pinned at
**source level**: `every_seam_kernel_guards_its_trace_ring` parses each kernel body, requires exactly two
`if (trace_ring != nullptr)` with an ENTER and an EXIT inside, and a mutation test deletes one guard and
asserts the checker refuses. Deleting a guard fails CI even though the compiler would not notice. That is
the right place to pin a property the artifact cannot express.

## The gate grew teeth: `--push` (2026-10-22)

The first version of `scripts/branch-gate.sh` was correct and still let a red commit through **twice**,
because the caller piped it: `bash scripts/branch-gate.sh 2>&1 | tail -3 && git push` makes the pipeline's
status `tail`'s, so the `&&` sees success. The lesson is not "be careful with pipes" - it is that any tool
whose correctness depends on how it is invoked is a tool with a defect. So the gate now owns the push:

```
bash scripts/branch-gate.sh --push        # gate first; git push only if the gate's own status is 0
bash scripts/branch-gate.sh --push 2>&1 | tail -20   # safe: the push is inside the script
```

With `--push` the push happens **inside the process**, after the gate's exit status is known, so no external
pipe can mask a failure and no `&&` can reorder it. The chained form is now discouraged in the script's own
doc, and the reason is stated there as a fact about the branch's history (two red commits from one idiom),
not as advice.

**Recorded policy**: a gate that can be defeated by piping is not a gate. When the same mistake happens
twice, the tool was wrong.

## The prefix's kernel is not called w8a8 (2026-10-22)

gemv stopped before writing the first stage body on a real contradiction, and two of the three things it
found were genuine:

**The activation-quantised GEMV exists; the contract named a kernel that does not.** `STAGE_CONTRACTS`
said `attention_projections` launches `wq_b_wo_a_gemv` - no such symbol anywhere. The landed kernel is
`mxfp8_gemv_launch` (`b12x_mxfp8_gemv_m1_sm121`, pinned `112512 B / c33cd129...`) whose **A operand is E4M3
codes plus ue8m0 scales** - exactly the prologue's output. PARAMS (11 words): `(a_fp8, a_scales, w_fp8,
w_scales, out_bf16, rows, cols, inner, trace_ring, layer_id, step_seq)`. Evidence it is the four-rank decode
path's projection kernel: main's `tp4_forward.rs` maps `wq_a`/`wkv`/`wq_b` to the native dense projection
(793-795), documents it as "E4M3 activation codes with one ue8m0 scale per 32" (773), and the query-path note
(2872) says wq_b is "E4M3 activations, ue8m0 block-32 scales, E4M3 B12x weights, f32 accumulate, bf16
output". The host `quantise_row` (1938-1958) is the prologue's host twin.

**Lesson**: the naming in this codebase is `mxfp8` for fp8 activations **and** fp8 weights; there is no
symbol containing `w8a8`. A contract row that names a kernel nobody can load is how a stage body gets written
against a fiction - so the row is corrected in the same commit as the spans, together with
`quantise_prologue`'s 6-word signature (the pre-ruling text; the real ABI is 8 words).

**The span gaps were real and are approved** (numbers from the real run config: dim 5120, n_heads 64,
head_dim 512, rope_head_dim 64, `q_lora_rank` 1280, hc_mult 4, 40 layers):

| span | bytes | writer -> reader |
| --- | --- | --- |
| `workspace.q_lora` | 2,560 (1280 bf16) | `attention_projections` (wq_a) -> `query_path` (q_norm) |
| `workspace.q_normed` | 2,560 (1280 bf16) | `query_path` (rmsnorm) -> latent quantiser, wq_b |
| `workspace.q` | 16,384 (8192 bf16 per rank = (n_heads/world)*head_dim) | `query_path` (wq_b, then rope tail in place) -> `attention_body` |

The latent's activation quantisation **reuses `act_codes`/`act_scales` at `cols = 1280`** (1,280 <= 5,120
codes; 40 <= 160 scales) rather than adding a second code span, and there is **no `q_rotated` span** because
`rope_rotate_bf16`'s contract explicitly allows row and out to alias (`kernels/kv_path_sm121.cu:19`). Total
slab growth: **21,504 B**, additive at the end so no offset moves.

## Both kernel tracks merged: 25 pins, `win/layer-device` = `03e6e211` (2026-10-22)

comm-lead's rebased kv branch and gemv's weight-plumbing recovery are in one branch, and the gate passes on
the merged tree with **25** baseline kernels: `layer_seams_sm121 ead69ce6...` and
`kv_path_sm121 91e6d62e...` both ok, final line `all kernels match the recorded baseline (V13.0.88, ...)`.

**The merge-order rule paid for itself.** comm-lead's baseline diff audit is the check I asked for:
`git diff --numstat` = `2 added, 1 deleted` - the header count sentence plus exactly one new kernel line - and
the regen on the merged tree reproduced driver-fixer's `layer_seams_sm121` hash byte for byte. That is what
distinguishes "two branches merged" from "the manifest was rewritten by a merge"; without it, a green
baseline check would have proved only that the file agrees with itself.

**Ten counting places, and the gate found two of them that the list did not have**: `build.rs`'s stale
"twenty-four" prose sentences and `device_chain`'s "23 baseline pins parsed" assertion (the plan prints its
own parsed count). Recorded conclusion: **a list of counting places is a guess; the gate is a check.**

### The prefix's exact GEMV geometry, from the host path (not from prose)

| launch | cols (output) | inner (reduction) | out | host line |
| --- | --- | --- | --- | --- |
| wq_a | 1,280 | 5,120 | `q_lora` | `attention.rs:2375-2377` |
| wkv | 512 | 5,120 | `kv_latent` | `attention.rs:2413-2418` |
| wq_b | 8,192 per rank (32,768 global) | 1,280 | `q` | `attention.rs` `fn wq_b` |

The host's `dense_view(rows, cols)` is `(output width, reduction width)` and `matmul(out, x, rows, cols,
batch)` takes the same order; `tp4_forward.rs:2355` pins `Mxfp8DenseParams::new(..., rows=1, cols, inner, ...)`
independently. Both are unsigned, so a transposition compiles - hence the assert at **open** (cols and inner
each against config, **and** their product against the resolved weight's real byte length), because a
product-only check passes a transposed pair.

## The first arithmetic is on the device path (2026-10-22, `c3877815`)

`attention_projections` now launches the decode GEMV on the device: wq_a -> `q_lora` then wkv ->
`kv_latent`, reading the prologue's `act_codes`/`act_scales`. The GEMV module loads **at open** beside the
seam image through the pinned resolver, and the three shapes are asserted at open with host provenance in
the comment (wq_a `cols=1,280 inner=dim`, wkv `cols=512 inner=dim`, wq_b `cols=8,192 inner=1,280`).
The weight map now keeps the **pair** `(weight, scale grid)`, because the 11-word GEMV takes `w_fp8` and
`w_scales` separately - a weight-only map could not drive it.

**A check that could not be written, and was said out loud instead of faked**: I asked for the product of
`cols*inner` to be checked against the resolved weight's real byte length. `RankBind.pointers` carries
addresses only, so the length does not exist on this interface. gemv asserted each factor against config
separately and documented why the product check is impossible today, rather than inventing a number to
compare against. That is the right call; the product check becomes possible the day the bind carries lengths.

**The row stops at `query_path`, and says so.** The q rope tail needs rope table spans that nothing
allocated, so the stage refuses with that sentence instead of rotating nothing - an unrotated tail is wrong
only at the last 64 values, which is exactly the class of error the numerical gate would localise late and
expensively.

### Rope tables: two classes, slab-owned, uploaded once

`rope_table_class(config, layer) = compress_ratio(layer) > 0` (`kv_path_ref.rs:342-350`) and
`rope_tables_for_layer` (`attention.rs:300-316`) chooses between exactly two builders on that same test, so
**table content is a function of the class, not the layer**: four arrays, not forty uploads. Layout:
`workspace.rope_tables_plain` and `workspace.rope_tables_compressed`, each
`2 * max_positions * (rope_head_dim/2) * 4 B` = `max_positions * 256 B` at `rope_head_dim` 64 (cos region then
sin region); both classes cost `max_positions * 512 B` (1 MiB at 2,048 positions). Uploaded **at open**, for
positions `0..max_positions`; the per-position pointer is `base + p * pairs * 4`. A run whose position budget
exceeds `max_positions` refuses at open with both numbers rather than growing a span mid-loop, and
`max_positions` goes in the record next to `layer_device_memory`.

Slab rather than an arm-private allocation: the tables are per-run device constants that stages read (rule
2), and one allocation means one capacity accounting and one place where their bytes appear in the record.

**Process, again**: gemv corrected its own earlier report unprompted - `cac9f697` carried only the
`StageRuns` record line, and the weights landed a commit later in `76298fe1`. The recurrence is the same
lesson at one level down from the gate: **the exit code is not the artifact**; a symbol must be grepped, not
assumed from `$?`. The history underclaims rather than overclaims, so it stays as it is.

## The whole prefix is on the device path; the first row is a probe, not a measurement (2026-10-22)

`c3877815`/`006890a0` bring the prefix together: `quantise_prologue` (8-word ABI) ->
`attention_projections` (wq_a -> `q_lora`, wkv -> `kv_latent`, 11-word GEMV on the prologue's codes) ->
`query_path` (rmsnorm `q_lora` -> `q_normed`, latent quantise into the reused `act_codes`/`act_scales` at
`cols=1,280`, wq_b -> `q` with `inner=1,280 cols=8,192`, then the rope tail in place per head). Rope tables
are two slab spans uploaded once at open, `max_positions * 256 B` each, chosen by
`rope_table_class = compress_ratio(layer) > 0` (four arrays, not forty).

**Branch hygiene**: gemv's branch was a cherry-pick of `76298fe1` onto `4a1ff256` while the integration branch
already carried it as a merge, so the merge conflicted in `tp4_forward.rs`; resolved by taking the branch
side, which is the superset, then verified no duplicate `attention_projections`/`query_path` and no conflict
markers. `LayerBuffers::carve` grew to eight positional arguments when the two table spans arrived and now
takes one `LayerGeometry` value (`dim, hc_mult, act_row, topk, vocab_pad, q_cols, max_positions`) - a
mechanical refactor, not a behaviour change.

### The first device row is a frontier probe, and must not be read as a number

Nothing in this prefix is validated against the host reference - there is no free GPU - so what the row can
prove is only: the spans/shapes/ABIs are the ones the launchers declare, the arm reaches a stage it names,
and the record's geometry is coherent (`max_positions`, `rope_table_bytes`, `slab_bytes`). Concretely:

- **The row is expected to abort at `kv_path`** (the body is not written yet). A refusal is the result.
- **It is not a numerical-gate candidate**: the D54 arm stays *unopened* until a row completes a whole layer
  inside the loop boundary; a row that stops early has no fixture output to compare and must never be quoted
  as a pass.
- **It must not produce a ms/step figure.** The loop aborts, so the wall clock is the time to the refusal.
- **Requirement on the record**: a device-stage refusal must still emit the `layer_device` frontier line
  (`stopped at <stage>: <message>`) before the run returns the error. A probe row whose frontier exists only
  in stderr is not machine-readable, and the frontier is the whole point of the row.
- **Arm selection is explicit** (`SPARKFLASH_LAYER_DEVICE=1`) and the record's `layer_device` line must agree
  with the environment - a probe row that ran the host arm while claiming the device arm is the one failure
  mode that would make the frontier meaningless.

## The mHC seam stages run on the device (2026-10-22, `356f00d0`)

With `mhc_entry` and `ffn_seams` landed, the frontier moved past the first stage for the first time:
**layer 0's row now stops at `kv_path`** - the stage order, not a measured row (the gate has no GPU
checkpoint). Layers 1 and 14 still stop at `engram_write`, which refuses by design because those layers
carry engram rows; that is correct behaviour, not a gap.

- `mhc_entry`: `mhc_entry_bf16`, 21-word ABI, one launch. Projects `residual.stream` (hc*dim) into the
  24-row coefficient tail in `workspace.attn_mix` (**pre[0..4], post[4..8], comb[8..24]** f32), applies
  `hc_pre` with the **input** pre set, RMSNorm with `attn_norm.weight` (BF16 branch) -> `residual.normed`,
  copies hc rows, and publishes its own pre set into `attn_mix`'s pre region.
- `ffn_seams`: `mhc_ffn_seams`, 23-word ABI (word 13 is the optional `x_bf16` out: the normed FFN input row, RNE-narrowed, the device router's operand; a null word writes nothing). Projects the post-attention stream into `workspace.ffn_mix`,
  `hc_pre` from `attn_mix`'s pre region, RMSNorm with `ffn_norm.weight` -> `workspace.seam_row`, quantises
  `seam_row` into `act_codes`/`act_scales`, and rotates its own pre set into `residual.pre_mix` for the
  next layer's entry (it writes no hc_rows copy: that output had no device reader). It is landed but
  **unreached**: it sits after `attention_residual`, which has no kernel yet.

**The commit's most valuable change is a semantic fix, not code volume**: `hc_pre` was consuming the seam's
*just-computed* pre set instead of the **input** pre set. In the reference, `hc_pre(pre_mix_in)` at the
attention seam and `hc_pre(attn pre)` at the FFN seam - one buffer apart, and the wrong one produces a small,
silent numerical drift rather than a crash. Both kernels now latch the input set in their own shared buffer;
smem went 20,720 -> 20,736 B (+16 B), regs 42/48, 0 spills. New pin `layer_seams_sm121 119408 B /
4e46149e...`; the regen reproduced the other 24 pins byte-identical and the baseline diff touched exactly one
line.

**Two contract nits in `layer_device.rs` were found by the audit, both assigned to its owner**: the mix-table
layout was documented in the wrong order (it says comb-first; the kernel writes pre, post, comb), and
`workspace.ffn_mix`'s reader list named "next layer's mhc_entry" when the entry reads `residual.pre_mix` and
`ffn_mix`'s only consumer is `ffn_epilogue`. A wrong reader in an ownership table is a six-stages-later bug.

**`main` has moved to `72d09a32` (!165, serve window/boot-log work) and does not overlap a single port file**
(`tp4_forward.rs`, `layer_device.rs`, kernels, `kernel_baseline.txt` all untouched). So the rebase onto main is
cheap and is **deferred until the port is feature-complete**, once, instead of rebasing every branch per
commit.

**Remaining before a whole device layer** (all stage-ordered): `reference_seam`, `stage_attention`,
`attention_body`, `attention_output`, both collectives, `attention_residual`, `moe_router`, `moe_routed`,
`all_reduce_moe`, `ffn_epilogue`. Of those, the two collectives, both moE stages and the attention bodies
have pinned kernels today; `reference_seam`/`stage_attention` and the two remaining mHC stages
(`attention_residual`, `ffn_epilogue`) do not, and driver-fixer has the mHC pair.

## Review: the seam arithmetic is right and one line of it is racy; the KV path was missing its norm (2026-10-22)

Read-only review of `win/layer-device` @ `5c556bc4`, cross-checked against `task/layer-kv-interface` @ `4020f33b`
and `task/layer-seams-full` @ `356f00d0`. No nodes touched.

### Confirmed correct

- **The `hc_pre` input-pre-set correction**: entry consumes `residual.pre_mix` and latches it before the
  coefficient tail writes its own pre set (`layer_seams_sm121.cu:420-421` vs `:429-430`, collapse `:434`);
  FFN consumes `attn_mix`'s pre region and latches it (`:493-494`, collapse `:507`). Reference agrees.
- **Table order** `pre[0..4], post[4..8], comb[8..24]` matches `Mixes` (`mhc.rs:441-462`).
- **eps/iters/norm_eps** reach the kernel from the same config fields as the host; the BF16 norm branch matches
  the plan's dtype (`engine/tensor.rs:193-199`).
- **kv_path's** rope range tables, adjacent-pair tail convention, whole-row block-32 single-amax fp8, and
  `slot = position % capacity` are faithful to the doc.

### Blocking, both silent

1. **A race in both seam kernels.** `spark_seam_coefficient_tail` is thread-0-only for the pre-mix write
   (`:241-296`), and the next statement broadcasts `pre_mix[tid] = pre_mix_smem[tid]` for `tid < 4` with **no
   `__syncthreads()` between them** (entry `:429-431`, FFN `:502-504`; the barrier is after, `:432`/`:505`). In
   the entry the pre-mix **aliases the mix table** (same 96 B span), so a stale lane corrupts exactly the four
   values `ffn_seams` latches; in `ffn_seams` it corrupts `residual.pre_mix`, the next layer's input. The host
   computes all four in one thread, so any stale lane is a wrong number - and **the Rust twin cannot see it**.
   Fix is one line; driver-fixer owns it.
2. **`kv_norm` was missing from the device KV path.** The host norms the latent *before* the rope
   (`tp4_forward.rs:3032-3038`); the device stage went latent -> rope -> quant -> append and never read the
   resolved norm weight. Nothing reads the ring yet, so no row is wrong today - but every window row would be
   the quantisation of an unnormalised latent once `attention_body` lands, a magnitude error that no logits
   bar would attribute to the KV path. The fused `kv_norm_rope_bf16` (norm in f32, rope on the f32 normed
   values, **one** bf16 rounding at the end) is comm-lead's, and the device-latent-is-already-bf16 deviation is
   declared, not hidden.

### Two contract consequences

- **The append guard could never fire**: `kv_append_rule(position, position, row_width)` passes the same value
  twice, so the `cache_len != position` refusal was vacuous while its comment claimed otherwise. A real
  `appended_rows` counter on the layer is required.
- **Append order**: the review called `kv_path`-before-`reference_seam` an inversion because the semantics doc
  says append-after-read. **Decision: keep one stage, and the order is append-then-read - a requirement, not
  an equivalence.** The window includes the current token (the released rule is
  `window_rows = min(p + 1, 128)`, `window_start = p + 1 - window_rows`; engine `window_indices`, m20's
  `window_count` and the port's own reference staging at position 0 all agree), so the row appended at `p`
  **is** read by the step at `p`; appending after the read would drop self-attention. The contract states the
  order and its refusal (`PositionPastRing`: a stage run before its append is refused instead of assembled
  sink-only); multi-position/prefill keeps the same per-position append-then-read rule and a grouped read
  must follow the group's last append. That is the trigger to split the stage, not the order itself.
  `attention_stage::assemble` now derives `window_rows = (p + 1).min(128)` from the body's 0-based `position`
  (device-lead's `42c1515f`), and the stage's `position` argument is that same 0-based p, so the two words the
  body takes cannot disagree.
- **Three ownership rows were still wrong** (`residual.hc_rows` misses `ffn_seams` as writer; `pre_mix` misses
  `begin_position`'s identity seed; `act_codes`/`act_scales` have three producers sharing one span). Assigned
  to the file's owner.

### Blocker decisions (raised by the stage work)

| Blocker | Decision |
| --- | --- |
| `all_reduce`/`all_reduce_moe` API absent (lives on unmerged `task/gpu-reduce`) | comm-lead rebases the real `RunReduce`/`DeviceReduce` onto the tip as `task/comm-reduce-api`; the stages keep refusing in the meantime - **no stub that pretends** |
| `moe_router` contract says one f32-input kernel; the pinned launcher is two kernels (10/8 words, bf16 x, gate+bias) | **Amend the contract to the real ABI**; names and dtypes come from the host path; assert dtype and shape at open. The contract follows the pinned launcher, names follow the host |
| `moe_routed` has no f32 delta span before the reduce | **Approved**: add the f32 routed-delta span (`dim * 4` = 20,480 B) and a bf16 pre-reduce span; record both in `layer_device_memory` and update `carve`/`LayerGeometry` and its tests |
| `attention_body` consumes `staged` from unlanded `stage_attention` | Hold - wiring it would read an unwritten span |

### The planning fact this review surfaced

The pinned 25-kernel baseline contains `flash_mla_csa2*` (body/seq/ring) but **no compressor, indexer or
top-k kernel**: `reference_seam` and `stage_attention` have no symbol to call, so they are new-kernel work,
not wiring work. Combined with the two mHC stages driver-fixer is writing, that is **four new kernels between
the current frontier and a whole device layer** - the honest remaining cost, and the reason the frontier
statement matters more than any estimate.

## A dtype mismatch that a half-width check authorised (2026-10-22)

While wiring `moe_shared`, gemv found that `query_path` hands `workspace.q_normed` - a **BF16** span of
2,560 B for cols=1,280 - to `w8a8_prologue_e4m3`, whose operand is `const float* normed`
(`layer_seams_sm121.cu:529-530`). The kernel therefore read **5,120 B of f32 out of a 2,560 B bf16 span**: the
wrong bytes, and a read past the span into whatever the slab puts next. Every device-arm run would have
quantised garbage for the latent, and nothing would have said so - the GEMV output stays finite-looking and no
numerical check exists on this branch yet.

**Why it was accepted**: `PrologueLaunch::new` checked `row.len >= values * 2` (`layer_device.rs:1083`) - the
*bf16* width. For the f32 variant that is exactly half the bytes the kernel reads. **A half-width check is
worse than no check, because it authorises the launch.**

**Fix**: a `w8a8_prologue_bf16_e4m3` symbol (bf16 input, widen to f32, quantise identically), which is the
reference-equivalent shape - the host's q_norm output is a bf16 tensor (`attention.rs:1318-1341`), and that is
the row the reference quantises. An f32-output rmsnorm would have quantised an unrounded row the reference
never sees, so it was rejected. The launcher check becomes **the input width its own dtype reads** (`values*4`
for f32, `values*2` for bf16) with a test for each refusal; and `open` now asserts, for every span whose dtype
the contract names, that its length equals `width * dtype_bytes`, so this class fails at open instead of
over-reading in a stage.

**The general lesson, recorded because this is the third variant of it on this branch** (wrong `cols`/`inner`
that compiles because both are unsigned; `hc_pre` reading its own output instead of its input; a span passed
at half its kernel's width): the port's contract asserts are the only thing standing between "compiles and
produces finite numbers" and "is the reference". Each of the three was caught by reading the host path or the
kernel's operand types, not by a test - which is why the audit of every launch's operand dtype against its
span dtype is now part of the work, not a suggestion.

## Status after the merge wave: kv_norm lands, dtype audit, moe_shared, routed spans (2026-10-22, `8a5088b1`)

**KV path is correct-forward now.** comm-lead's fused `kv_norm_rope_bf16` lands the host's order on the
device - `sum(x*x)` sequential in f32, `inv = 1/sqrt(mean+eps)` via sqrt-then-divide, `(x*inv)*w` in f32, the
tail's adjacent pairs in f32, **one** bf16 rounding - with the two deviations declared rather than hidden: the
device latent is already bf16 where the host keeps f32 (measured, not asserted: max 2 % relative, bf16-operand
scale), and the kernel uses `sqrt`-then-divide rather than `rsqrt`. New pin
`kv_path_sm121 57976 B / 65a5891b...`; baseline diff exactly one line, the other 24 byte-identical.

**The dtype audit and the open-time width check.** gemv read every kernel parameter declaration in `kernels/*.cu`
against every span the arm passes, and found one real mismatch (the bf16 `q_normed` passed to a `const float*`
prologue - see the half-width section above), one deliberate encoding (the coded arm stores a bf16 row inside
the f32 `seam_row` span under `shared_bf16 = 1`), and no others. `DeviceLayer::open` now runs
`LayerBuffers::assert_contract_widths` over **22** spans and refuses a doctored span by name with its length,
contract width and rounded expectation - the general form of the same defect class.

**`moe_shared` landed with one arm wired and one refusing.** The coded (M3) arm runs four launches (prologue ->
w1/w3 GEMV -> wide SwiGLU+quantise -> w2 GEMV) writing a bf16 row over `seam_row`'s base; the **f32-reference
arm refuses by name** because the device f32 SwiGLU symbol does not exist yet, and a host fallback would
reintroduce the exact D2H/H2D round trip the stage exists to remove. That is the right trade: a refusing stage
is a known gap, a half-running one is a wrong number.

**Routed MoE spans carved ahead of the body**: `workspace.moe_delta` (dim f32) and `workspace.moe_delta_bf16`
(dim bf16) are in the slab, checked at open and covered by tests, with `moe_reduced_bf16` remaining the
collective's output and no aliasing between the three. The body is being split into plumbing-then-loop, because
a 300-line commit that lands its own open-time plumbing hides three bugs at once.

**Critical path to a whole device layer, in order**: `kv_norm` wiring (one call site) -> `reference_seam` and
`stage_attention` kernels (new work, four new kernels total with the two mHC stages) -> `attention_body`/
`attention_output` wiring -> `all_reduce` (waiting on comm's API rebase) -> `attention_residual` (landed) ->
`ffn_seams` (landed) -> `moe_router`, `moe_routed`, `all_reduce_moe` -> `moe_shared` (landed) ->
`ffn_epilogue` (landed). The frontier after the kv_norm wiring is `reference_seam`, stated from the stage
order and not from a run.

## The two remaining mHC stages land, plus three kernels the merge needed (2026-10-22, task-24)

`attention_residual` and `ffn_epilogue` have device bodies, typed launchers and CPU twins, and the seam
image gained the three kernels their consumers were missing: `shared_swiglu_f32` (the default shared arm's
activation between two device GEMVs) and `w8a8_prologue_bf16_e4m3` (the quantiser over the bf16 q_norm row).
The image is still **one** pinned cubin, now with **seven** symbols. Pin: `layer_seams_sm121 156568 B /
5d447239...`, `ptx_membar_lines=0`; the regen reproduced the **other 24 pins byte-identical** and the baseline
diff touched exactly one line, 25 pins total.

`mhc_entry_bf16` is now a **20-word** ABI (21 before): the entry's pre set is the pre region of the one table
it writes, so it has no second pre-mix output. The 21-word form passed the same `attn_mix` span as two
`__restrict__` parameters, which is UB as soon as both are written (the review's finding).

regs / spill / smem, all seven symbols (nvcc V13.0.88, sm_121a): `mhc_entry_bf16` 42 regs, 20,720 B smem,
1 barrier; `mhc_ffn_seams` 48 regs, 20,720 B, 1 barrier; `w8a8_prologue_e4m3` 47 regs, 0 B;
`w8a8_prologue_bf16_e4m3` 46 regs, 0 B; `mhc_attn_residual` 40 regs, 0 B; `mhc_ffn_epilogue` 40 regs, 0 B;
`shared_swiglu_f32` 17 regs, 0 B - 0 spill stores/loads in every one. The two seam kernels are 16 B lighter
than the 356f00d0 record because the pre-mix shared buffer is gone.

**The review's blocking defect is closed structurally.** `spark_seam_coefficient_tail` no longer writes a
shared `float[4]` that lanes 0-3 read back; thread 0 writes the pre set into the mix table (and, in the FFN
seam, the same thread copies it into `residual.pre_mix` afterwards), so there is one writer, no cross-lane
read, and no barrier to forget. The tail's comment names the failure mode so the shape cannot come back.

**Declared arithmetic arms, not hidden constants.** `shared_present` (0 on this tip) says the arm has no
shared-expert row; `shared_bf16` is the arm's word (1 = the coded M3 GEMV's BF16 store, widened in the
epilogue; 0 = the default f32-reference arm's f32 row, added as-is), and `native_bf16` is the mHC arm.
`shared_swiglu_f32` declares the one by-construction difference it carries: the host silu calls Rust's `exp`,
the kernel calls CUDA's `expf`, so that row is f32-reference arithmetic under a D54 tolerance, not a
bit-identical arm. `DeviceLayer::open` refuses a plan whose `attn_norm`/`ffn_norm` dtype is not the seam's
BF16 decoder, naming the tensor and both encodings.

**The prologue's f32 variant was reading a bf16 span.** `query_path` passes `workspace.q_normed` (2,560 B of
bf16, 1,280 values) to `w8a8_prologue_e4m3`, whose input is `const float*`: the kernel read 5,120 B out of a
2,560 B span and encoded wrong bytes, and the launcher's span check (`values * 2`) authorised it because it
was the *bf16* width. The fix is the reference's own shape - the host's q_norm output is a bf16 tensor
(attention.rs:1318-1341), so the quantiser must see the rounded values - hence the new
`w8a8_prologue_bf16_e4m3` (widen, then the same MX-block arithmetic) and a launcher/stage check that asserts
the width the kernel actually reads (`values * 4` for f32, `values * 2` for bf16). The general lesson is on
the record: **a half-width check is worse than no check, because it authorises the launch.**

**Frontier unchanged: `reference_seam`.** Both new stage bodies are callable and twin-pinned, but they sit
behind stages that still refuse on this base (and `moe_shared`, which launches the SwiGLU, is behind
`kv_path`'s missing `kv_norm`), so no run executes them and no row is claimed. `ffn_seams`' copy of
`residual.hc_rows` also has no reader after this change (the epilogue is in-place on `stream`); dropping
that output is a separate ABI trim and is not folded in here.

## The CSA2 chain maps onto the pinned kernels without a repack - but needs two producers (2026-10-22)

Two read-only recons (device-lead on the compressor, driver-fixer on the attention kernels) changed the cost of the
seam from "two new kernels" to "no repack kernel at all, three producers".

**The compressor is not an MXFP8 GEMV.** `compressor.wkv` is bf16 at ratio == 1 and f32 above, and
`compressor.norm.weight` is bf16; a MXFP8 kernel would decode those checkpoint values as E4M3 codes and read
ue8m0 scales that do not exist. The reusable kernel is the pinned **host-weight** GEMV
`b12x_hostw_gemv_m1_sm121` (three symbols, CPU twin, oracle test already green), whose "repack" is a pure
permutation of the checkpoint's own values staged once per layer per run: 10.49 MB f32 on the ratio > 1 layers
(2, 8, 14: `wkv`+`wgate`), 5.24 MB bf16 on layer 20, 1 KB norm. **No new cubin for the compressor.** The
store width is unobservable at ratio == 1 (the host narrows immediately) and **load-bearing above it**
(`pool_group` reads raw f32 rows), so the launch selects its store width from the layer's ratio and the record
says which ran.

**`attention_body` is the sequence entry, and the window needs no gather.** The port's ring is
slot = position % capacity with capacity == `max_positions`, so within a run slot == position and the last 128
rows are contiguous oldest-first - exactly the `flash_mla_csa2_seq_sm121` contract (19 words, grid = per-rank
heads, 256 threads, window codes 512 B/row + 16 scales/row, FP4 store 256 B/row + 32 scales/row, `selected` =
expanded-store row ids). The **ring** entry (21 words) explicitly does not gather and carries no numeric claim -
it is a shape, not evidence. The staged FP4 expanded store docks directly onto `kv_fp4`/`kv_scales`/
`selected`; the 147,456 B per-call zero-fill and the per-selected-row `quantize_main_kv` are the two costs we
now choose to remove by making the expanded store the cache's **native** form rather than repacking into it.

**`attention_output` is wiring except the inverse RoPE.** Order: inverse RoPE of the body's bf16 out (host Rust
today, `tp4_forward.rs:3515-3540`) -> `b12x_bf16a_mxfp8w_wo_a_sm121` (8 words, grid `(groups,4,1)`) ->
`w8a8_prologue_bf16_e4m3` over the bf16 `lo_out` -> `b12x_mxfp8_gemv_m1_sm121` via `Mxfp8DenseParams` ->
`all_reduce` -> `mhc_attn_residual` (landed). The inverse rotation is being added as one rows-parameterised
symbol with a sign word (one table pair, not a second negated class); weights and sink are already resident.

**The bite is producers, not kernels.** The kernels are stateless, so every risk is a span somebody else must
write: (a) the contiguous-window assumption is conditional on ring capacity, so the launch asserts
`ring_rows >= max_positions` and `position < ring_rows` rather than trusting it; (b) `selected` is an
**expanded-store row id, not a token position**, and a device producer that rebuilds that mapping differently
silently drops or admits candidates; (c) unselected store rows must stay zero across positions or the kernel
reads last position's FP4; (d) the indexer pool and `block_best` are zero-filled wiring until the indexer path
is real, and a per-rank head -> index-head offset is needed if it ever goes live.

**`AttnState` residency (ruled, and it is a prerequisite, not a detail).** The arm owns **one** window: the
loop's fp8 kv cache becomes the arm's device buffer; `WindowRing` stays read-only and host-arm only. Per-layer
state is allocated at open, never per position - window 264,192 B/layer (10.6 MB over 40 layers), `compressed`
bounded by `max_positions/ratio` with a refusal past the bound, indexer keys, `kv_state`/`score_state`/
`latent`, and the compressor weights in the slab. All of it charged to the layer-arm memory accounting, and
`reset_shared_attention()`'s process globals unreachable from the arm.

**And the router's missing operand is the same defect class.** `moe_router` reads the **un-normalised** bf16
pre-FFN hidden row, which no device stage produced - `ffn_seams` writes the normed row and the mix outputs, not
that row. Fix: `mhc_ffn_seams` grows one out word (RNE of the row it already reads) into a new
`workspace.ffn_x_bf16` span, and the router refuses if the span was not written rather than casting something
plausible. Third appearance of "a stage that would read a span written by nobody" - which is why the open-time
width asserts and the unwritten-span refusals are now the pattern the remaining stages follow.

### The seam image at eight symbols: the FFN's BF16 hidden out and the sign-word RoPE (task-24 follow-up)

`layer_seams_sm121` is now **eight symbols, one cubin, one pin**: `164944` B,
`5f25cd7598bb93332a7e974c854672228d9ebadab16ba4388be7f2f6c27f2921`, `ptx_membar_lines=0`, from
`156568/5d447239...`. The regen reproduced the **other 24 pins byte-identical** and the baseline diff touched
exactly one line, 25 pins total.

**FFN ABI 23 -> 24 words, one optional out.** `mhc_ffn_seams` gains `__nv_bfloat16 *__restrict__ x_bf16`
after `act_scales`: the FFN block's input **after** its RMS pre-norm, narrowed to BF16 with RNE - the host
arm's `pre_ffn_hidden` and the device router's operand (`docs/build/moe-stage-contract.md` H3b; the host
narrows `pre_ffn_hidden` at the launch). It is the norm's OUTPUT, not the hc_pre collapse the norm consumes:
the router routes on the normalized hidden, and the kernel writes the word inside the norm loop beside
`seam_row`. Under `native_bf16` the released boundary already made the normed value bf16-representable, so
the narrowing is idempotent; under the f32-reference arm the normed row is wider and the router gets its RNE
round, exactly the host's relationship. Pinned by
`the_ffn_x_bf16_word_is_the_normed_row_narrowed_in_both_arms`, which also refuses the collapse satisfying the
assertion. A null pointer writes nothing, so a launcher passes 0 until the span is carved. The ABI bookkeeping
moved in the commit that added the word: `FFN_ARG_WORDS`, `FFN_PARAMS`, the declaration test,
`FfnSeamsParams::words`, the CLI's `FfnSeamsParams` and the doc line.

**Correction (2026-10-22): the word was the collapse, and that was wrong.** The first build of this word
narrowed the pre-norm hc_pre collapse ("the value the norm consumes"). It is not the router's operand: the
host reference computes `pre_ffn_hidden = ffn_in` and then `rms_norm`s it **in place** before the router
launch narrows it (`tp4_forward.rs` 4676-4693 and 5066-5070 on the integration tip), so the router routes on
the normalized row. With the collapse in the word, the device arm's router would have routed on the
un-normalised hidden and disagreed with the host arm at every layer. The fix moved the write into the norm
loop, RNE of the normed value, and re-pinned the image: `170672` B,
`7f5859b762fd02df2ac7be565ef41e2410158020173cebdd8e79fb732c4fb8ea`, `ptx_membar_lines=0` (from
`171016/05adaa68...`); the other 24 pins are byte-identical. The twin test was renamed and now asserts the
normed row and that the collapse does **not** satisfy it.

**`shared_present` is 1.** The shared expert is no longer absent by declaration: `moe_shared` runs a device
arm in stage order before the epilogue and leaves its row at `seam_row`'s base, and `shared_bf16` reads the
same `native_shared` word the arm selection used (0 = the default arm's f32 row, 1 = the coded arm's BF16
store). If the arm cannot run the stage refuses before the epilogue, so a stale row cannot be added. This
supersedes the `shared_present (0 on this tip)` sentence of the task-24 record; the reference's shared add is
no longer a gap of this arm.

**The inverse RoPE is a symbol now, with the sign as a word.** `rope_rotate_rows_bf16` (11 words: four
pointers, `rows`, `head_width`, `rope_head_dim`, `sin_sign`, then the trace trio every seam kernel carries
last), grid = one 256-thread block per head row, adjacent-pair rotation, one BF16 rounding at the end, head
copied unchanged, `out` may alias `src`. `sin_sign` = 1 is the forward rotation and -1 the attention output's
inverse, so one table pair serves both directions; a negated table class would state the same one bit twice.
Declared deviation: **none arithmetic-wise** - the same rotation with the sin negated, the host's
two-roundings-per-pair order and one narrowing; the only difference is that it is a device launch. Declared
ordering deviation: the sign is the last *geometry* word rather than the literal last word, because the trace
trio is the trailer on every seam kernel and the trace-guard pin reads that shape. The kernel records the
`DENSE_MXFP8` scope, the scope the attention output's dense projections record. Twin: `rope_rows_ref` in
`layer_seams_ref.rs`, pinned bit for bit against `kv_path_ref::rope_row_ref` (sign +1) and
`linalg::rope_rotate_inverse` (sign -1).

**The two doc residuals are closed.** The helper's stale `the caller may hand the same buffer as the pre mix
output` sentence is gone (the entry's pre-mix output no longer exists), and `shared_swiglu_f32` now declares
**two** deviations, not one: CUDA's `expf` vs Rust's `exp`, and one true divide vs the host's
reciprocal-rounded-then-multiplied `x * sigmoid(x)` (`tp4_forward.rs:7441-7462`, `linalg::silu` at
`linalg.rs:1171-1173`). Both are inside the D54 f32-reference arm.

**The ring-wrap bite, stated rather than assumed.** The CSA2 **seq** entry (19 words) is the right one for
`attention_body` on this port only while the KV ring satisfies `ring_rows >= max_positions` and the position
satisfies `position < ring_rows`: then `slot_for == position`, the last-128 window is contiguous and no gather
exists to be missed. D49's 134-slot ring breaks both assumptions and needs the gather that the ring entry
(21 words) explicitly does not do; its launch must assert both conditions and refuse otherwise, not rely on
this note. The full mapping (symbols, word layouts, staged-store sizes, missing producers, attention_output's
five steps) is the recon record delivered to the Lead.

**Per-symbol regs / spill / smem** (nvcc V13.0.88, sm_121a): `mhc_entry_bf16` 42 regs, 20,720 B smem,
1 barrier; `mhc_ffn_seams` 48 regs, 20,720 B, 1 barrier; `w8a8_prologue_e4m3` 47 regs, 0 B;
`w8a8_prologue_bf16_e4m3` 46 regs, 0 B; `mhc_attn_residual` 40 regs, 0 B; `mhc_ffn_epilogue` 40 regs, 0 B;
`rope_rotate_rows_bf16` 40 regs, 0 B; `shared_swiglu_f32` 17 regs, 0 B - 0 spill stores/loads in every one.

**Verification.** `cargo test -p sparkflash-kernels --lib` 486 passed; `cargo test -p sparkflash-cli --lib`
378 passed (428 with `--features cuda`); clippy clean for the kernels crate and the CLI in both feature
configurations; the baseline regen reproduces the single changed pin on the regen script's absolute-path nvcc
invocation. Frontier note unchanged: `attention_body`/`attention_output` wiring is gemv's; this record claims
no stage run and no bit identity for a stage row, only the twin pins above.

## Review catch on merge: the bootstrap must not exist on the host arm (2026-10-22, `f666cce5`)

comm-lead's bootstrap landed as a real run object - one `RefCell<RunReduce>` in the arena closure, id exchanged
over the loop transport's own `all_gather_bf16`, `open` on all four ranks, three warm-up collectives, one sync,
drop at scope end - with the two rules in the function doc and the primary-context statement that prevents the
classic "works until it deadlocks" NCCL failure.

**But the block opens unconditionally**: `tp4_chain.rs:933-985` runs whenever a transport exists, with no arm
check. Three consequences, in order of severity: (1) a normal four-rank **host-arm** run now requires libnccl and
fails at setup without it - the arm that produced 179.81 ms/step must not acquire a new runtime dependency;
(2) every host-arm run pays NCCL init plus three setup collectives, changing the setup budget of rows that have
nothing to do with the port; (3) "device arm opt-in, host arm unchanged" stops being true, which is the kind of
drift that makes a later comparison between the two arms misleading rather than invalid.

**Fix**: gate the block on the arm. The arm decision is `layer_device::select_layer_arm`, so `layer_device.rs`
exposes `device_arm_requested()` (the same env predicate, one parser) and `tp4_chain.rs` calls it - not a second
env parser in the chain file, because two parsers that disagree is its own bug class. Host arm: no NCCL open, no
scratch, no warm-up, `reduce: None`. Device arm: the sequence as written.

This is the second time on this branch that a correct-looking slice had a *global* side effect it did not intend
(the first was a launch that read a span written by nobody). The pattern worth keeping: when a slice adds a
resource with a lifetime, ask which arms that resource exists on, not just whether its happy path works.

## Merge wave: bootstrap gated on the arm, stage timing on the ring, 27 pins (2026-10-22, `1ca8e796`)

**The collectives are real and the host arm is untouched.** NCCL bootstrap landed as a run object (one
`RefCell<RunReduce>` in the arena closure, id over the loop transport's `all_gather_bf16`, collective open on
all four ranks, three warm-up collectives, one setup sync, drop last). My review catch - the block opened on the
host arm too, giving the 179.81 ms/step arm a libnccl dependency and a setup cost it never had - is fixed by
gating the whole block on `device_arm_requested()`, which delegates to the loop's own parser, so the two cannot
drift. Host arm: no `Cuda::load`, no scratch, no open, no gather, no warm-up, `reduce: None`.

**Stage timing exists, and it exposed a build decision.** Every seam launch records one stage-coded event on one
host-mapped ring (allocated only under `SPARKFLASH_DEVICE_TRACE=1`, capacity configurable, unset = ring word 0 and
a byte-identical run); the frontier record gains `stage_trace` with per-stage calls/total/min/mean. The finding:
the shipped baseline cubins are **trace-disabled builds**, so adding stage codes alone left the pin
byte-identical - arming a ring would have recorded nothing. The seam image now compiles its hooks in every
build, which is why the pin moved to `171016 / 05adaa68` (+7,464 B, registers up on four symbols, 0 spill, smem
unchanged, other 26 pins byte-identical).

**Two more producers landed.** The compressed-cache device form (native FP4 append + selected fill, twin-equal to
the pinned quantiser, zero-fill and per-row quantise gone, capacity refused past its bound) and the
`stage_attention` operand assembly as a pure function that launches nothing. **Contract decision**: `stage_attention`
stays as the **validation** stage (ring/position/cursor/selected refusals plus `window_start`,
`window_rows = min(position,128)`), not as a copy stage; with capacity == `max_positions` the seq entry reads the
window contiguously from a base pointer, and a D49 134-slot ring is the only thing that would put a real rotation
back in the body.

**27 pins.** `ref_select_sm121 28264/b90bacc`, `compressed_append_sm121 23800/77026c93` (two symbols,
`compressed_append_sm121` + `compressed_selected_fill_sm121`), the trace-enabled seam image, comm's fused
`kv_norm_rope_bf16` pin, and the 23 others unchanged. Three merges needed line-wise baseline unions; a file-level
union silently reverts a pin, which happened twice and was caught by the gate both times - the gate is the only
reason that class does not survive a merge.

**Still refusal stubs**: `attention_body`, `attention_output`, the `ref_select` stage line and `moe_router`. The
frontier statement is unchanged and is **by stage order, not measured**: layer 0 stops at `reference_seam`.

## Adversarial review of the collectives: one blocking gate defect, two medium error-path gaps (2026-10-22)

review-critic audited the merged bootstrap read-only at `54784974`. Headline: **the bodies pass** - exactly one
`reduce_bf16` enqueue per collective, count in elements (5,120) not bytes, stream 0, no sync or readback in the
stage, correct spans and width checks, one `RefCell`, one `Some`-writer, scoped borrow, scratch freed after the run,
communicator destroyed before the Cuda handle - and **the arm gate fails in one reachable configuration**:

- **B1 (blocking).** The gate is `device_arm_requested()`, but the loop's arm is
  `select_layer_arm(layer_device_enabled(), evidence_record_enabled())` and **evidence recording forces the host
  arm**. So `SPARKFLASH_LAYER_DEVICE=1 SPARKFLASH_FORWARD_EVIDENCE=1` runs the host arm *and still* opens NCCL: the
  arm that must have no libnccl dependency acquires one, and a host without libnccl loses that run. Fix is one
  function - `resolve_layer_arm()`, used by both the loop and the chain, so there is no second parser and no drift.
- **M1 (medium).** Pre-gather failures do abort every rank before NCCL, as claimed. Post-gather they are rank-local:
  a rank missing the library errors in `load` while its peers sit in `comm_init_rank_handle`; a warm-up or sync
  failure leaves peers in NCCL init or `cuCtxSynchronize`. They cannot produce a number, but they do not return
  `Err` either - they hang to the outer timeout. Fix: load the library on every rank before the gather and carry each
  rank's status word in the existing world*64 gather, plus a bounded post-open barrier on the same transport, which is
  setup phase and therefore not the in-loop handshake rule 2 forbids.
- **M2 (medium).** The rendezvous carries no per-rank configuration, so a partially-set `SPARKFLASH_LAYER_DEVICE`
  gives a mixed-arm run whose device ranks block in NCCL init forever - a hang, not a refusal - and a per-rank
  `--native-rounding` changes numerics with no failure at all. Same fix: arm bit, native-rounding word and a
  plan digest in the bootstrap gather, refused on mismatch before NCCL.

The pattern in B1 is worth naming: **two correct predicates, one arm**. `device_arm_requested()` is the opt-in half
and `select_layer_arm` is the whole decision; a consumer that takes the half as the whole is right in the default case
and wrong exactly when the other input is set. The rule that prevents it is not "use the parser", it is "consume the
decision, not its inputs".

Low findings: the divergence rule is documented on the recording `run_layer` and the collective trait but not on the
trait's default body implementors follow; `attach_reduce` lacks the double-attach refusal `attach_routed` has; a
world-1 device-arm run refuses at `all_reduce` with a message blaming the bootstrap rather than at open.

**Attention gap, confirmed independently by driver-fixer and gemv**: the seq entry's operands have no device spans
yet - no compressed store, no window codes/scales ring, no `selected`, no indexer/sink/block buffers, and
`attn_out_bf16` cannot be both the pre-inverse-RoPE body out and the wo_b out. The reference arm's `selected` seam
is the host indexer staged per position; everything else must become device-resident producers before a body exists.
