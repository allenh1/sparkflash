# S1 QA harness: steady-state counters, one-command D54 gate, layer diff

Status: **harness landed; no device row run.** Written 2026-10-22 on a host with no GPU, no
`/dev/nvidia*` and no node access. Every command in the "Runnable now" section was executed on this
host; every node-only item is listed in the unrun table and is **not** claimed. The published number
stays **179.81 ms/step** until a D54-passing S1 row exists; no speed-log row is added by this
harness, and none may be fabricated.

This document is the contract for three pieces:

| file | what it is | CPU-verifiable now |
| :-- | :-- | :-- |
| `crates/sparkflash-cli/src/device_counters.rs` + one `pub mod device_counters;` line in `lib.rs` | the per-step steady-state invariant recorder (`DeviceStepCounters`) the device arm carries | `cargo test -p sparkflash-cli --lib device_counters` |
| `scripts/qa-d54.sh` | the one-command 32 prompts x 128 greedy-token D54 gate, wrapping `scripts/numerical-gate.sh` | `--self-test`, `bash -n`, fail-closed path |
| `scripts/qa-layer-diff.sh` | host arm vs device arm per-stage max-abs/ULP bundle with the 10x bisection rule | `--self-test`, `bash -n`, fail-closed path |

The D54 bars are **not restated here**. They live in `scripts/numerical-gate.sh` (`TOL`, `TOP_K`,
`COMPARE_K`) and are used unchanged; `qa-d54.sh` declares no tolerance and re-derives none. The
worst-position line it adds is diagnostic: it never changes the gate's verdict.

---

## 1. Steady-state invariant counters (`DeviceStepCounters`)

### What is enforced

The S1 acceptance shape is a device-only step that crosses PCIe **once**: the position's final
logits row, read back for the D54 gate. The loop counts the three crossings it can make and refuses
a step whose count is not the declared one:

| counter | allowed | why |
| :-- | --: | :-- |
| `d2h_copies` | exactly `1` (the final logits) | steady-state invariant 5 (`ws-whole-layer-device-port-20261022.md`) |
| `allocations` | exactly `0` | rule 2: every span is sized at open; a stage never allocates |
| `sync_waits` | at most `1` | the step-boundary sync; a collective is queued, not waited |

The **declared host-bridge reduce** is the only exception, and only because it is declared: a run
that keeps the host reduce for the two per-layer collectives pays one readback per collective,
`40 layers x 2 collectives = 80` copies, and records that as `HostBridgeReduce::Declared`. The
allowed count is `1` without the declaration and `1 + 80 = 81` with it; `81` without the declaration
fails and `1` with it fails. The summary line prints which declaration produced the allowance, so a
row cannot carry an invisible exception.

Reporting priority when several invariants are broken (deterministic failure text):
`allocations` > `d2h copies` > `sync waits`. Any one fails the step; the order only decides the
sentence. `check(d2h, alloc, sync, bridge)` is the pure function; `DeviceStepCounters::check(bridge)`
is the loop's call; `DeviceStepCounters::summary(bridge)` is the row's line.

Example row line (`PASS`):

```text
device step 7: d2h=1 (allowed 1, logits only) sync=1 (allowed <= 1) alloc=0 (allowed 0) PASS
```

With the declared host bridge:

```text
device step 7: d2h=81 (allowed 81, final logits + declared host-bridge reduce) sync=1 (allowed <= 1) alloc=0 (allowed 0) PASS
```

The module is device-free and clock-free: every boundary is a CPU unit test
(`d2h` 0/1/80/81 with and without the declaration, `alloc` 0/1, `sync` 1/2, reset-per-step, worst
priority, failure text, summary text). It reads no environment and no device.

### Exact wiring instructions (the Lead's edit; this package does not touch the loop)

The counters are only as good as their call sites. Wire them in the **device arm only** - the host
arm keeps its per-layer observation readbacks by design (`LayerArm::Host`), so a host-arm step would
fail these invariants correctly and must not be counted. Concretely, in the device arm's forward
loop (`crates/sparkflash-cli/src/tp4_forward.rs`, the `for position in 0..total_positions` body of
the forward loop, guarded by the selected `LayerArm::Device`):

1. **Carry one recorder per run, outside the position loop.** Declare the run-level bridge
   declaration next to the arm decision (same place `decide_layer_arm` is consumed), from the reduce
   path the run actually selected, and print it in `run.meta`:

   ```rust
   use crate::device_counters::{DeviceStepCounters, HostBridgeReduce};

   // The declaration is a call-site fact, never defaulted: Declared only when the run kept the
   // host reduce for the two per-layer collectives.
   let bridge = if run_uses_host_bridge_reduce { HostBridgeReduce::Declared } else { HostBridgeReduce::Disabled };
   let mut step_counters = DeviceStepCounters::new();
   ```

2. **Reset at the top of each position** (the step), before any layer work:

   ```rust
   step_counters.reset();
   ```

   Setup work (slab allocation, NCCL open and warm-up, the setup sync) happens before the first
   `reset`, so it is not counted - it is setup, not steady state.

3. **Record the one logits readback.** At the single `copy_dtoh` that makes the final logits row
   host-visible for the D54 dump (the head's logits readback), immediately after it:

   ```rust
   step_counters.record_d2h_copy();
   ```

   Exactly one call site. If `LayerStages::residual_out` / `attn_pre_mix_out` still perform head
   boundary copies in the device arm, record those too - the step will then fail with a d2h count of
   2 or 3, which is the forcing function for the device head. **Do not widen the allowance**; the
   S1 invariant is the single final-logits readback.

4. **Record allocations only if the step allocates.** Steady state must not; every `cuda.alloc` /
   `alloc_pooled` reachable inside the position body (not the open-time slab) increments:

   ```rust
   step_counters.record_allocation();
   ```

5. **Record the step-boundary sync.** At the one `cuda.synchronize()` that closes the position's
   timing (the sync behind `step_samples`):

   ```rust
   step_counters.record_sync_wait();
   ```

   No per-stage synchronise may record - a per-stage sync is a port bug, and if it records, `check`
   fails with the count.

6. **Check before the row is recorded; refuse the step on failure.** After the head, before the
   position record/`step_samples.push`:

   ```rust
   if let Err(violation) = step_counters.check(bridge) {
       return Err(format!(
           "S1 steady-state invariant broken at position {position}: {violation}"
       ));
   }
   ```

   `CounterViolation`'s `Display` names the invariant and both numbers, so the refusal is a
   diagnosis, not "a counter was wrong".

7. **Print / record the line for the S1 row.** Add to the position record (and/or `run.meta`) the
   raw counts and the summary, e.g. under `"device_step_invariants"`:

   ```rust
   serde_json::json!({
       "d2h": step_counters.d2h_copies(),
       "allocations": step_counters.allocations(),
       "sync_waits": step_counters.sync_waits(),
       "summary": step_counters.summary(bridge),
   })
   ```

   The S1 row prints the summary verbatim; a row without this line is not an S1 row.

8. **Cross-check once against the probe meter.** The existing `tp4_forward::probe` recorder counts
   the same driver entry points (`d2h_count`, `alloc_count`, `syncs`) in
   `tp4-forward-perf-r{rank}.json`. On one debug (non-timed) run, the counter summary must agree
   with the probe's run-scope counts; a mismatch means a call site bypasses the recorder (a wrapper
   that copies without going through the recorded function). The probe must stay off in any row
   quoted as a number - it is a diagnostic, and it synchronises.

The counters do not enable or disable the device arm. `SPARKFLASH_LAYER_DEVICE=1` stays the opt-in
(`layer_device.rs`); the counters are passive and must not change arithmetic, ordering or timing.

---

## 2. `scripts/qa-d54.sh` - the one-command D54 gate

```sh
scripts/qa-d54.sh --run <run-dir> [--run <run-dir> ...] [--reference DIR] [--manifest FILE]
scripts/qa-d54.sh --self-test
```

What it does, in order:

1. **Freeze check (exit 2 if not frozen).** Parses the fixture manifest, requires
   `schema = s1-d54-fixture-v1`, `32` prompts, `greedy_tokens = 128`, the declared margin prompt
   (`position = 4`, `nats = 0.0038`, tolerance `5e-5`). Then requires the reference directory to hold
   exactly **32 complete fixtures** (`rank0..3.json` each) and at least one reference whose tightest
   top1-top2 margin is the declared `0.0038` nats at position 4. A reference frozen without the
   margin prompt is refused - otherwise the decisive position silently moves.
2. **Run-evidence check.** Each run directory must hold `rank{0..3}/tp4-token-r{rank}.json`
   (`run-fourrank.sh` layout); otherwise exit 2 with the missing path.
3. **The gate.** Invokes `SPARKFLASH_GATE_REFERENCE="$REFERENCE" scripts/numerical-gate.sh --run
   <runs...>` unchanged; its exit code and PASS/FAIL lines are authoritative. Its bars are the D54
   ones: greedy tokens identical, top-1 `1.00`, top-5 `>= 0.80`, top-50 `>= 0.96`,
   max|dlogprob| `<= 2.0e-2`, mean `<= 2.0e-3`, KL `<= 1.0e-4`, all finite.
4. **Worst position (diagnostic).** For each rank, re-reads the run's logits and the frozen
   reference summary, computes `max |ref_logprob - run_logprob|` over the shared top-`COMPARE_K`
   (read from `numerical-gate.sh`'s own `COMPARE_K`, not hardcoded) and prints the per-rank worst
   position plus the overall `WORST:` line, with the reference's top1-top2 margin at that position.

Exit codes: `0` gate PASS + worst position printed; `1` gate FAIL (worst position still printed);
`2` no frozen reference / invalid fixture / incomplete run evidence / usage - **nothing was
compared**. The "no frozen reference" exit is printed with the full freeze procedure, so the
operator does not have to find this document first.

### Fixture freeze state at writing time

**NOT FROZEN.** There is no `docs/build/reference/s1-d54/` directory and no
`docs/build/reference/s1-d54/manifest.json` in this tree, so `qa-d54.sh` exits 2 here. Verified:

```text
$ bash scripts/qa-d54.sh --run /tmp/no-such-run-dir
qa-d54: no frozen reference
  ... declared shape: 32 prompts x 128 greedy tokens, 0.0038-nat margin prompt at position 4
  fixture state: manifest ABSENT, reference dir ABSENT
  ...
exit: 2
```

The only frozen reference in-tree is the **9-position speed-gate row**,
`docs/build/reference/speed-gate/fab52c00a0dc6f6c/{rank0..3}.json`: one 9-token prompt per rank,
fixture key `fab52c00a0dc6f6c`, and (verified by reading `rank0.json`) its tightest top1-top2 margin
is `0.0038127899169921875` nats at position index 4. That is the fixture D54's `0.0038` sentence was
written about, but it is **one prompt x 9 positions**, not the S1 fixture's 32 prompts x 128 greedy
tokens. It must not be used as the S1 reference.

### Manifest schema (`docs/build/reference/s1-d54/manifest.json`, to be frozen)

```json
{
  "schema": "s1-d54-fixture-v1",
  "created": "2026-10-22",
  "description": "S1 device-arm D54 fixture: 32 frozen prompts, 128 greedy tokens each; one prompt carries the 0.0038-nat top1-top2 margin at position 4",
  "prompts": [
    {"id": "p00", "text": "<frozen prompt text>"}
  ],
  "greedy_tokens": 128,
  "margin": {"position": 4, "nats": 0.0038, "tolerance": 5e-5},
  "reference": "docs/build/reference/s1-d54"
}
```

`prompts` must have exactly 32 entries. The manifest is the human-readable declaration; the machine
key is the gate's `sha256(positions, prompt_len, ids)` fixture key computed from each run's
`tp4-token-rN.json`, so the reference directory holds one subdirectory per frozen fixture. A new
prompt needs one more `--freeze`, never a gate change (D54 item 2).

### Freeze procedure (node-only; not run)

On a node, with the four-rank binary and the checkpoint:

1. Produce the 32-prompt token set (the frozen prompt list, one file or argv per prompt) and write
   the manifest above; record its sha256 in the record.
2. For each prompt, run the **host arm** (the bit-exact CPU twin - evidence recording forces the host
   arm, so do not set `SPARKFLASH_LAYER_DEVICE=1`) with the logits dump enabled, one evidence
   directory per prompt:
   `sparkflash ... --dump-logits <run>/logits.jsonl` (the writer is `logits_dump.rs`; the token
   record must carry `tokens.{ids,positions,prompt_len}` and `logits_dump.{bin,rows,vocab}`).
3. Freeze each run's reference into the shared reference directory:

   ```sh
   scripts/numerical-gate.sh --freeze <run-dir> docs/build/reference/s1-d54
   ```

   (It writes one `fixture/rank{0..3}.json` per run; 32 runs -> 32 fixtures x 4 ranks.)
4. Verify the freeze: `scripts/qa-d54.sh --self-test`, then `scripts/qa-d54.sh --run <one-run>` -
   the freeze check must print `fixture OK`, and the gate must PASS (it is comparing the host arm to
   its own frozen row).
5. Only then may a device-arm run be gated with the same command. Keep the output on the record.

---

## 3. `scripts/qa-layer-diff.sh` - host arm vs device arm, per stage

```sh
scripts/qa-layer-diff.sh --layer N --host DIR --device DIR [--position P] [--bounds FILE]
scripts/qa-layer-diff.sh --self-test
```

It diffs one rank's per-boundary f32 rows from the two arms at the same layer/position, prints
per-row `max |host - device|` and max ULP, and applies the bisection rule: a row diverges when its
max-abs exceeds **10x its declared unit-test bound** (and, when declared, its max ULP exceeds 10x
its ULP bound); the **first stage in execution order** with a divergent row is the reported fault.
It does not average, guess or reorder. Exit `0` within bounds, `1` divergent stage found (printed),
`2` fail closed (no GPU / no observation copies / no bounds / usage).

This is the "debug row" of `docs/archive/build/ws-device-arm-numerical-gate-20261022.md` section 3: the
product device arm has no per-stage readback by design, so localisation uses a dev-gated copy of the
boundaries and is never quoted as a number.

### Observation contract

Both arms write one little-endian f32 row per boundary, same names, under
`DIR/pos{P}/row{L}.{name}.f32`. The host arm already writes this set from
`write_boundary_trace_row` when run with:

```sh
SPARKFLASH_BOUNDARY_TRACE=<dir>/host-trace SPARKFLASH_TRACE_POSITIONS=<position> \
  <the four-rank host-arm run>          # rank 0 writes the rows
```

The rows and their producing stage (execution order):

| row(s) | producing stage | note |
| :-- | :-- | :-- |
| `residual_in`, `hc_in_pre` | `layer_input` | written before layer L (L-1's `ffn_epilogue`/`ffn_seams`, or `begin_position`): divergence here is upstream of L |
| `hc_attn_pre`, `hc_attn_post`, `hc_attn_comb`, `pre_attention_hidden` | `mhc_entry` | |
| `attention_out` | `all_reduce` | the reduced attention delta |
| `post_attention_hidden` | `attention_residual` | |
| `hc_ffn_pre`, `hc_ffn_post`, `hc_ffn_comb`, `pre_ffn_hidden` | `ffn_seams` | |
| `ffn_out`, `residual_out` | `ffn_epilogue` | |

**The device arm's copy of this set is not landed.** The product arm deliberately has no per-stage
observation read, so a dev-gated writer must be wired before this tool can run: it must write the
same 14 rows, same names/layout, at the matching boundaries, and be off (or provably cost-free) in
any timed row. Until that writer exists, `qa-layer-diff.sh` fails closed at the artifact check with
exit 2. The host writer is rank-0 only today; the tool therefore localises rank 0 (per-rank logits
divergence is `qa-d54.sh`'s four-rank gate).

### Bounds file (`docs/build/reference/s1-layer-diff-bounds.json`, to be frozen)

**NOT FROZEN** (absent in this tree), so the rule has no declared bounds and the tool exits 2. The
bounds must be copied from each stage's own twin/unit-test bar - not guessed - for example the
declared `<= 1e-6 / 8 ULP` router-scores and `<= 1e-6 / 16 ULP` router-weights bars in
`docs/archive/build/moe-stage-contract.md`, and the other stages' declared bars in their contract rows. All
14 observation rows must be present; a partial table is refused because the first-divergent rule
cannot be applied to an undeclared row.

```json
{
  "schema": "s1-layer-diff-bounds-v1",
  "bounds": {
    "residual_in": {"stage": "layer_input", "max_abs": 1.0e-7, "max_ulp": 4},
    "hc_in_pre": {"stage": "layer_input", "max_abs": 1.0e-7, "max_ulp": 4},
    "hc_attn_pre": {"stage": "mhc_entry", "max_abs": 1.0e-6, "max_ulp": 8},
    "hc_attn_post": {"stage": "mhc_entry", "max_abs": 1.0e-6, "max_ulp": 8},
    "hc_attn_comb": {"stage": "mhc_entry", "max_abs": 1.0e-6, "max_ulp": 8},
    "pre_attention_hidden": {"stage": "mhc_entry", "max_abs": 1.0e-6, "max_ulp": 8},
    "attention_out": {"stage": "all_reduce", "max_abs": 2.0e-2, "max_ulp": null},
    "post_attention_hidden": {"stage": "attention_residual", "max_abs": 1.0e-6, "max_ulp": 8},
    "hc_ffn_pre": {"stage": "ffn_seams", "max_abs": 1.0e-6, "max_ulp": 8},
    "hc_ffn_post": {"stage": "ffn_seams", "max_abs": 1.0e-6, "max_ulp": 8},
    "hc_ffn_comb": {"stage": "ffn_seams", "max_abs": 1.0e-6, "max_ulp": 8},
    "pre_ffn_hidden": {"stage": "ffn_seams", "max_abs": 1.0e-6, "max_ulp": 8},
    "ffn_out": {"stage": "ffn_epilogue", "max_abs": 1.0e-6, "max_ulp": 8},
    "residual_out": {"stage": "ffn_epilogue", "max_abs": 1.0e-6, "max_ulp": 8}
  }
}
```

The values above only show the schema; **the frozen file must cite each row's owning test/contract
bar**, including the declared deviations (D1-D7) rather than replacing them with a wider guess.
`max_ulp: null` means "no ULP bar declared for this row; apply the max-abs rule only".

### Bisection examples (from the self-test)

```text
layer 7 position 4: host .../host vs device .../device
row                      stage              max_abs    max_ulp   abs_bound   ratio  verdict
pre_ffn_hidden           ffn_seams         5.00e-05          0    1.00e-06   50.00  DIVERGENT  (max_abs 5.000e-05 > 10 x 1.000e-06)
...
BISECTION: first divergent stage = ffn_seams (row pre_ffn_hidden: max_abs 5.000e-05 > 10 x 1.000e-06)
```

---

## 4. Runnable now on this host (CPU only)

All commands below were run on 2026-10-22 in `/home/user/deepseek-v4.1-flash-spark`, no GPU.

| command | result |
| :-- | :-- |
| `cargo test -p sparkflash-cli --lib device_counters` | **11 passed, 0 failed** (396 filtered out) |
| `cargo clippy -p sparkflash-cli --lib -- -D warnings` | **clean** (exit 0, full dependency-linting form). An earlier run was transiently blocked by another in-flight package edit, since fixed; `--no-deps` was also clean |
| `RUSTDOCFLAGS="-D missing_docs" cargo doc -p sparkflash-cli --no-deps` | **exit 0** (no missing docs in this module; 34 pre-existing doc warnings elsewhere in the crate are unchanged by this package) |
| `cargo fmt --all -- --check` | **clean** (exit 0) |
| `bash -n scripts/qa-d54.sh scripts/qa-layer-diff.sh` | **OK** |
| `bash scripts/qa-d54.sh --self-test` | **QA-D54 SELF-TEST PASS** (freeze shape rejections, 32 fixtures, worst position, fail-closed path) |
| `bash scripts/qa-d54.sh --run <dir>` | **exit 2**, `no frozen reference` + freeze procedure (correct: no S1 fixture in tree) |
| `bash scripts/qa-layer-diff.sh --self-test` | **QA-LAYER-DIFF SELF-TEST PASS** (bisection at first stage, sub-10x within bounds, non-finite, fail-closed paths) |
| `bash scripts/qa-layer-diff.sh --layer 0 --host H --device D` | **exit 2**, no GPU on this host (correct: fail closed) |
| `bash scripts/numerical-gate.sh --self-test` | **SELF-TEST PASS** (the reused gate, unmodified) |

## 5. Unrun list (node-only; nothing here is claimed)

| item | why it did not run | what unblocks it |
| :-- | :-- | :-- |
| 32 x 128 D54 gate on a real row | no frozen S1 fixture/reference, no GPU, no node | freeze procedure section 2 on a node, then `qa-d54.sh --run` |
| worst-position reporting on real logits | same | same |
| device-only layer execution and counters on a step | no GPU; seam kernels not all landed on this tree | device-lead/comm-lead land the kernels; Lead wires the counters |
| host-vs-device per-stage diff | no GPU; dev-gated device observation writer not landed; bounds file not frozen | land the writer + freeze bounds, then run on a node |
| S1 timed row (d2h/alloc/sync counters in the record) | no device-arm run | a D54-passing device-only layer, then a clean four-rank window |
| speed-log row / published number | none may be added until the gate passes | published number stays **179.81 ms/step**; `docs/build/speed-log.csv` untouched |

## 6. Risks and review notes

- **Counters are call-site counts.** A copy/sync/alloc that bypasses the wired function is not
  counted; the probe-meter cross-check (section 1.8) is the mitigation, and it must be run once on
  a debug row. The counters are not a CUDA API interposer.
- **The 1-d2h invariant is a forcing function, not a description of the current boundary API.**
  `layer_device.rs` documents up to two head-boundary D2H copies (`residual_out`,
  `attn_pre_mix_out`); the S1 row requires exactly one (final logits). Until the device head consumes
  the residual on device, a device step records 2 and correctly fails. Do not raise the allowance.
- **`sync_waits <= 1` admits 0.** If the timing path relies on a sync recorded outside the position
  body, confirm where it is counted; a step with no recorded sync still passes the check.
- **The host-bridge declaration is trust-based.** Recording `Declared` while not actually using the
  host reduce would allow 81 copies. Tie the value to the run's selected reduce path at the same
  place the arm is decided, and print it in `run.meta`.
- **`qa-d54.sh`'s worst position is diagnostic.** It re-uses the gate's transform and reads the
  gate's `COMPARE_K` from the script source; it never re-declares a bar. If the gate's comparison
  changes shape, update this diagnostic, not the bars.
- **The layer-diff bounds are a freeze, not a guess.** A guessed bound makes "10x the unit-test
  bound" an opinion; the tool refuses without a complete bounds file.
- **Shared worktree.** Other packages were being edited concurrently while this harness was
  verified (`sparkflash-comm`); re-run `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --all -- --check` and the workspace tests on the final integration tip before the S1
  row. This document's "runnable now" table records the state at writing time.

### File ownership

This package owns only: `crates/sparkflash-cli/src/device_counters.rs`, the single
`pub mod device_counters;` line in `crates/sparkflash-cli/src/lib.rs`, `scripts/qa-d54.sh`,
`scripts/qa-layer-diff.sh`, and this document. It did **not** edit
`tp4_forward.rs`, `layer_device.rs`, `resident_weights.rs`, `scripts/numerical-gate.sh`,
`docs/build/speed-log.csv` or `docs/build/s1-result.md`.
