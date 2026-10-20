# Four-rank device loop: dedicated runner (A) vs providerised HybridLine (B)

Status: decision memo, no code, no measurements, no token claimed. Date: 2026-10-20.
Branch: task/task34-arena. Scope: how the four-rank arena-bound arm runs 40 layers with two
reduces per layer without host staging.

## What runs today

- `crates/sparkflash-engine/src/forward/mod.rs`: the block loop is
  `run_hybrid_sequence_step_ranged` (mod.rs:1673) over a `StepLayers` window; the per-layer work is
  `self.block(...)` (call sites mod.rs:788, 1525, 1755, 2052). `block` takes a layer index and
  reads/writes through the engine's own weight source and provider/post hooks — it does not know
  about ranks, arenas or collectives.
- `crates/sparkflash-cli/src/tp4_host.rs`: `HostProviders` (host-side provider/post for the CPU
  twin), `SharedSource` with the B1 rank view (`SharedSource::for_rank(inner, cache, rank, world,
  bound)`) and the arena-bound refusal; `RankWeights` with the sharded expert view.
- `crates/sparkflash-cli/src/device_chain.rs`: the single-node device runner. It stages host
  weights per layer into device buffers (`restage_layer_post(&post_staging)` at the post closure,
  device_chain.rs:1821) and runs every layer through the launcher; it has **no** `HostProviders` or
  `SharedSource` (verified by grep across `src/*.rs`: those symbols exist only in `tp4_host.rs`).
- `crates/sparkflash-cli/src/tp4_chain.rs`: rank setup, 71.376 GiB/rank arena, resident attention
  bind, 80-call collective walk, `SharedSource::for_rank` bound list, posted-reduce arena
  (`64c55d8`) and the real posted/wait seam (`4bc5ff9`). No forward pass.
- Budget facts: 285.504 GiB all-resident set / 4 = 71.376 GiB/rank; attention sharded (wq_b column,
  wo_a column, wo_b row) needs 2 reduces/layer; replicated attention needs 1 but costs ~5.15
  GiB/rank/token vs ~1.29 sharded.

## Option A — dedicated four-rank device runner module

New module (e.g. `crates/sparkflash-cli/src/tp4_device_runner.rs`) built from the pieces that
already exist:

- reuse from `device_chain.rs`: the launcher calls and per-kernel wiring, the MoE backend
  (`moe_device::MoeDeviceBackendImpl`), the attention backend, the token/head code, the
  `--dump-logits`/`--timing-json`/warmup plumbing;
- reuse from `tp4_chain.rs`: `RankSetup`, the arena and `bind_rank_attention`, the geometry table,
  `LayerCollective` / posted-reduce arena;
- new, small: per-layer arena pointers instead of per-layer staging, an explicit
  `Frame { reduce: RankReduceHook }` threaded through the two seams, and a step driver:
  for each token/step, for each layer in `StepLayers::through(...)`: attention from arena-bound
  weights -> `reduce_layer(layer, Attention)` -> MoE -> `reduce_layer(layer, MoE)`; head; logits/tokens.
- drop: all host staging on this arm (`restage_layer_post` must never run; the per-category
  `weight_bytes_per_token == 0` gate enforces it).

Size estimate: ~250–400 new lines in one module plus `--run-stage tp4-device` parsing; no change to
`forward/mod.rs` or `device_chain.rs` behaviour. Risk: duplicates the token/head/evidence glue and
can drift from the single-node runner; but it cannot break the twinned single-node arm, and the
device arm is allowed to diverge because staging and arena binding are mutually exclusive.

## Option B — make engine `HybridLine` accept device weight providers

Give `forward::block` (and its provider/post traits) a device-backed implementation: the engine asks
a provider for per-layer weights by name and receives either a host slice or a device pointer. The
same `run_hybrid_sequence_step_ranged` loop then runs on four ranks with a reduce hook injected
into the provider/post closures.

Required changes:
- a weight-availability abstraction in `forward/mod.rs` covering every `block` weight read
  (attention projections, routed/shared experts, norms, engram, mHC); today `block` assumes host
  data, so this is a pervasive change to the hot loop, not an injection point;
- a device-aware `HybridEntry`/`HybridPost` pair: the existing post closure reports
  `attention_out`/`residual`, but a reduce needs the *sharded partial* buffer at exactly the seam,
  and the host/staging runner would have to be prevented from restaging on the four-rank arm;
- preserving byte-identical single-node behaviour while every call site is re-typed.

Size estimate: ~400–700 lines spread across `sparkflash-engine` and the cli provider layer, touching
the shared hot loop used by the twin and every existing stage. Risk: the twin and the device arm are
the only two implementations of the same rule (D40 acceptance compares them), so a pervasive engine
change enlarges the blast radius of a parity bug and slows the first token; it also makes the
"staging never runs" guarantee an emergent property rather than a structural one.

## Recommendation: A

Take the dedicated four-rank device runner. The device arm's defining property is that weights are
resident in a per-rank arena and never staged; the single-node runner's defining property is that it
stages per layer. Making one runner do both means making the staging path conditional inside the hot
loop, which is both the largest diff and the easiest place to hide an accidental H2D. Option A keeps
the two arms structurally separate, reuses the arena/collective/geometry code that is already tested,
and leaves `forward/mod.rs` untouched, so the twin parity proof stays valid and the four-rank loop
can be reviewed as one new module.

## First concrete commit of option A

Create `crates/sparkflash-cli/src/tp4_device_runner.rs` with: (1) a `Tp4DeviceRunner` struct holding
the rank setup, arena, attention/MoE backends and an optional `RankReduceHook`; (2) a `step` that
walks `StepLayers::through(stop_after_layer, n_layers)` calling the arena-bound attention backend,
the attention reduce seam, the MoE backend, the MoE reduce seam, then head/logits; (3) evidence
fields `execution: "four-rank device"`, `device_tensors_touched: true`, per-layer reduce rows
(`calls_per_step == 80` assertion), and a refusal when any host staging path is reachable. Compile
both feature sets; no token in that commit (MoE shared/routed bindings follow, labelled).
