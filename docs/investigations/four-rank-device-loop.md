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

## Window result 2026-10-20: option A built, bound attention, no token yet

The runner and the rank-collective wiring ran on the real four-rank cluster. Transport, ordering and
payload are proven (80 reduces/step, 819,200 B, canonical bf16 ascending-rank sum verified
byte-for-byte, 14-22 ms/step steady); the resident arena boots at 72.308 GiB/rank with every tensor
verified (see the serving-parity plan addendum for the numbers). Attention binding reached the
coverage check and failed there: the first failure was a real byte-order bug (BF16 `compressor.wgate`
bind demanded an FP8 B12x payload), and after excluding the host seams it became a name-set
disagreement between the expected-shape table and the plan, now reported by name. No forward pass,
no token, therefore no ms/step and no tokens/s claim from this arm.

Next: named coverage difference, MoE shared/routed bindings, first token, then the D48 verify-step
measurement.

## Bind coverage resolved, MoE bindings landed, window kit ready (2026-10-20, later)

Attention coverage: the 2026-10-20 window's "bound 208 of 196" was a denominator bug, not a bind
bug - the check subtracted every plan host seam (19) from the table length (215) although only the
table's own seams (7) were in that set. The bind now compares the two name SETS (table-derived
expected vs plan-derived bound), reports missing and unexpected names in both directions, and a test
fails if the sets differ (208 device-served + 19 plan host seams; `28cc613`).

Arena-resident MoE bindings (`6db8243`): routed w1/w3/w2 plus scale grids, the shared expert, the
router and the tail (embed row shard, certified head row shard, final norm) are bound to the rank
arena with one set check per class against geometry-derived expected sets (61,440 gate/up, 30,720
down, 240 shared, 120 router, 3 tail = 92,523 device names) and one byte check per slice. The bind
has no reader and no source, so per-token weight H2D is zero by construction. A census claims every
resident shard by class (attention 208 + host seams 19 + routed 92,160 + shared 240 + router 120 +
tail 3 + named host-side attention-aux/mhc/engram/mtp classes), so an unclaimed tensor is a named
bug instead of silence. Evidence is written progressively inside the arena epoch (attention, then
MoE, then tail+census) before the collective walk; the window kit is
`scripts/tp4-window-bind.sh` and `docs/investigations/window-kit-m1-bind.md`.

The bind also found the next real blocker by reading the pinned ABI: routed w1/w3 are arena-ABI
compatible (576 output rows x 2560 packed bytes, [576,160] row-scale grid), the shared expert is
compatible (runtime-shaped dense kernel: cols=576/inner=5120 and cols=5120/inner=576), the router is
compatible (row-major BF16/F32) - but the routed DOWN projection is not. `b12x_mxfp4_moe_sm121`
reads the A8 pre-padded 640-K row layout (320 packed bytes + 20 scale bytes per output row) while
this rank's plan stores the checkpoint's 576-K rows (288 packed + 18 scale bytes), because the
one-time padding is a loader step in the single-node path. The bind is bytes-verified but a launch
on that pointer would misindex every row after the first. The evidence names both remedies: a kernel
ABI variant that reads the stored layout (kernel-lead committed the design + pin plan,
`docs/build/moe-stored-layout-variant-plan.md`, `29cd418`), or a boot repack at +25.2 GiB/rank
(rejected on a 121 GiB node with 72.3 GiB arena + KV).

Next-window order (Lead-routed): (1) stored-layout combine kernel + m25_moe_stored harness green on
the released stored layout, (2) wire the stored cubin in moe_device.rs, (3) the four-rank forward
loop wiring (attention from arena -> attention reduce -> MoE from arena -> MoE reduce -> final norm,
head, argmax) and head/logit emission on the rank arena, (4) 8-9 position oracle comparison. The
token milestone depends on all four; nothing in the bind window claims a token.
## Window result 2026-10-20 (run 2, package b69ebb2): full-rank bind coverage PASS, no token

One window, four ranks, one process per node, same stage and argv as the kit (`--run-stage tp4-chain
--upload --loop`, no `--cubin-dir`, transport timeouts 120 s). All four ranks exited 0; launch to
last exit 469 s (Lead's runner measurement; worst-rank arena boot ~380 s, every bind landed before
the walk). Evidence: `/home/user/ev-session-20261020/window/m1-bind/rank{0..3}/`.

Per rank, from `tp4-bind-r<R>.json`: attention 208/208 device-served names (table 215, 19 host seams
excluded, missing/unexpected empty), routed gate/up 61,440, routed down 30,720, shared 240, router
120, tail 3/3 with non-null pointers, every class expected == bound with empty differences;
`summary.complete` true, `device_bound_names` 92,731, per-token weight H2D 0, error null. Plan
identity identical on all four ranks: 95,822 planned shards (95,818 resident / 4 disk), 77,640,198,984
resident bytes = 72.308 GiB, plan hash 0x9a59035fc40776c8 - the same plan the earlier window built.

The census over the REAL plan totals 95,818 with no unclassified name: attention_projection 208,
attention_host_seam 19, attention_aux_host_read 416, routed_expert 92,160, shared_expert 240, router
120, tail 3, special_token 3, mhc 240, engram 8, mtp 2,401. The special_token class (committed as
`b69ebb2`) exists because the released checkpoint carries image_start/image_end/image_newline as
resident tensors even on the text path; without the class the census would report three unclassified
names and `summary.complete` would be false for a reason that is not a bind defect. The vision tower
(259 names) and aligner (4) are already excluded by the plan.

The walk repeated the earlier transport result on the final binary: tcp, 3 steps (2 warmup + 1
decode), 80 calls/step = 240 reduces, every step `calls_per_step_ok`, payload 819,200 B/step,
wall_ms 41/22/20. `forward` is not-run, `tokens` 0, `hybrid_sequence.executed` false, no cubins
were loaded. The names manifest is 185,481 logical lines = 208 attention expected + 208 attention
bound + 19 attention host seams + 2 x 92,520 MoE + 2 x 3 tail, byte-identical on all four ranks; it
has no trailing newline, so Python counts 185,481 and `wc -l` counts 185,480. (An earlier estimate of
185,462 omitted the 19 host-seam lines; the kit and this note now carry the correct breakdown.)

Two defects found by the window itself. (1) The bind log line printed the two routed rows summed as
"gate/up" and their difference as "down", so a correct bind read as "down 0 names"; fixed by
`tp4_chain.rs` printing each coverage row's own bound count. (2) The census completeness rule
initially required `unclassified.is_empty()`, which asked the text-path bind to explain multi-modal
tokens; `b69ebb2` claims them by name instead. Neither affected the stored evidence.

Launch compatibility, restated from the bind JSON: routed gate/up true, routed down FALSE (the
stored-layout ABI blocker: the arena holds the checkpoint's 576-K rows while `b12x_mxfp4_moe_sm121`
reads the 640-K A8 pre-padded layout), shared true, router true. That single false row is why this
window is a pre-token PASS: no forward pass ran and no token is claimed. Next-window order is
unchanged: stored-layout combine kernel + harness on the released stored layout, wire the cubin in
`moe_device.rs`, then the four-rank forward loop + head/logit emission, then the oracle comparison.

