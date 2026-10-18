# Review feedback of 2026-10-18: every claim, and what happened to it

Status: written 2026-10-18 on branch `task/review-hygiene` (commits c739f93, 7136ce3, 954d7af and
the docs commit that carries this file). This note goes through the review claims one at a time and
records one of three outcomes: **already true** (no change needed, with the evidence), **fixed here**
(with the commit), or **rejected** (with the reason). Nothing is reported as fixed without a commit
and a command that shows it.

The review text itself is not in this repository; the claims below are the set registered for task-4.
Two of them belong to other owners and are recorded as **open, not fixed here** rather than quietly
dropped.

## 1. `crates/sparkflash-kernels/build.rs` lists six kernels

**Claim as registered:** build.rs lists 6 kernels while the kernel set is larger.

**Verdict: CONFIRMED, OPEN at base `2245a4e` - not fixed here (different owner: kernel-dev).**

Measured on that base commit (and still true on this branch, which does not touch build.rs):

* `crates/sparkflash-kernels/build.rs:147-154`: `const M4_KERNELS: [(&str, &str); 6]` holds
  `mega_mhc_sm121`, `b12x_mxfp8_dense_sm121`, `b12x_mxfp4_moe_sm121`, `flash_mla_csa2_sm121`,
  `b12x_mxfp8_wo_a_sm121`, `cert_lm_head_sm121`. `swiglu_mxfp8_quant_sm121` is **not** in it, and the
  build-script warning says "6 of the six Milestone 4 kernels".
* `crates/sparkflash-kernels/scripts/check-kernel-baseline.sh:19` names **seven** kernels (the six
  above plus `swiglu_mxfp8_quant_sm121`), and
  `crates/sparkflash-kernels/baseline/kernel_baseline.txt:11-17` pins seven cubins.
* `kernels/` holds eight `.cu` files: the six above, `swiglu_mxfp8_quant_sm121.cu`, and
  `rocenante_oneshot_sm121.cu` (which has its own probe). So "six" is spec 6.2's Milestone-4 set; the
  standalone swiglu kernel is compiled and pinned by the baseline script but not by the build-script
  nvcc probe.

Fix belongs in build.rs (kernel-dev's scope): add `swiglu_mxfp8_quant_sm121` to `M4_KERNELS` (or
state why the probe intentionally excludes it) and correct the "6 of the six" wording. No test here
can close it; the baseline already compiles seven, so the drift is in the probe's coverage, not in
the pinned bytes.

## 2. The swiglu kernel takes trace parameters but emits no trace scope

**Claim as registered:** `swiglu_mxfp8_quant_sm121.cu` takes the trace arguments and only casts them
to void.

**Verdict: CONFIRMED, OPEN at base `2245a4e` - not fixed here (same owner).**

`kernels/swiglu_mxfp8_quant_sm121.cu` declares `GpuTraceRing *__restrict__ trace_ring` plus
`layer_id`/`step_seq` in both kernels (lines 85-88 and 123-126) and immediately does
`(void)trace_ring; (void)layer_id; (void)step_seq;`. The in-source comment says it outright: "no scope
is defined for this kernel yet: the trace macros reduce to (void)(ring) when the build does not
define SPARKFLASH_ENABLE_GPU_TRACE". So the parameters exist to keep the ABI uniform across the M4
kernel set, and no trace scope is emitted from this kernel on either build.

If a trace scope is wanted, it is a kernel change (kernel-dev's scope) plus a cubin-baseline
regeneration; adding a source-text pin here would be exactly the kind of scanner D26 stops.

## 3. `docs/DECISIONS.md` contains duplicated D10-D13 blocks

**Claim as registered:** DECISIONS has a verbatim duplicated block.

**Verdict: FIXED in c739f93.**

The second, verbatim copy of D10-D13 (105 lines) was deleted; the first copy is retained and D14
follows it directly. Checked after the commit: `grep -c '^## D' docs/DECISIONS.md` = 26 headings for
D1-D26 with strictly increasing numbers and `grep -c '^## D10' docs/DECISIONS.md` = 1. The index table
still has one row per decision (26 `| D` rows). No decision text was rewritten in that commit.

## 4. Multimodal `TODO(M2)` in `sparkflash-api`

**Claim as registered:** `crates/sparkflash-api/src/multimodal.rs` still carries `TODO(M2)`.

**Verdict: ALREADY TRUE and correctly labelled - no change (kept as the milestone marker).**

* `crates/sparkflash-api/src/multimodal.rs:148-151` is explicit about what is missing: "derive the
  real grid from the decoded image rather than the 32x32 default; see
  reference/inference/image_processor.py for the authoritative preprocessing". `DEFAULT_IMAGE_GRID`
  is 32 (line 33) and `default_image_tokens()` charges `count_image_tokens(32, 32)` (line 160).
* This is the declared M1 behaviour, not a hidden shortcut: `docs/architecture/06-vision.md` records
  the reference preprocessing plan line by line (image_processor.py:46-128) and this crate's path
  does not decode images or fetch URLs yet.
* The `reference/inference/...` path is the same research-reference convention the engram modules
  cite; the reference tree is not vendored in this repository, so the TODO names the upstream file
  rather than a local path.

So the review's observation is accurate and the TODO is the honest marker of an unimplemented M2
capability. Removing it would make the docs less true. No code change.

## 5. `serve.rs` and the `Send + Sync` engine seam

**Claim as registered:** `serve.rs` has a `Send + Sync` problem at the engine seam.

**Verdict: ALREADY HANDLED and documented - no change needed.**

* The API seam requires it: `crates/sparkflash-api/src/engine.rs:399`
  `pub trait InferenceEngine: Send + Sync`, and `state.rs:16-18` stores
  `engine: Arc<dyn InferenceEngine>`, so `AppState` is only `Send + Sync` if the engine is.
* `crates/sparkflash-cli/src/serve.rs:19-31` documents the resolution in the module header: the
  reference engine is not `Send`/`Sync`, so it is built and owned by one dedicated thread
  (`serve.rs:348` spawns it) and the router holds a channel-driving proxy that is `Send + Sync` by
  construction; the engine never leaves its thread and no `unsafe` is added. Requests are therefore
  serialised, which is the reference engine's documented shape.
* The compile-time proof is the call site: `serve.rs:881-882` builds
  `AppState::new(Arc::new(driver))` and passes it to `sparkflash_api::server::router`. Both the trait
  object and the router require `Send + Sync`, so a driver that was not `Send + Sync` would not
  compile. There is no runtime-only claim to pin.

## 6. `m4_abi.rs` is brittle and too large

**Claim as registered:** the Milestone-4 ABI source-text lint keeps growing and every review round
adds scanner features for properties text cannot establish.

**Verdict: FIXED in 7136ce3, pin moved in 954d7af.**

* The file is frozen by a module-level notice: no new source-text scanner/parser/lint for a new
  kernel invariant; new invariants go to compiled artifacts (cubin/PTX baseline via
  `crates/sparkflash-kernels/scripts/check-kernel-baseline.sh`, which the `cuda-abi` job runs) or to
  runtime device assertions in a harness. Existing scanner defects are still fixed.
* Recorded as DECISIONS D26, including the measured size and the reasoning (the file's review history
  reaches round 36: at this branch's pin the round-36 A1 attributed-`try`/`catch` finding is
  `m4_abi.rs:822` and `:3694`, the round-33 A1 `BARE BLOCK` comment is `:4607`, and the round-24
  paragraph is `:203`).
* Enforced by `ci/check-m4-abi-size.sh` (fails if the line count differs from the pin, in either
  direction, since the task-11 hardening) and the `m4-abi-freeze` job in `.gitlab-ci.yml`. Measured
  2026-10-18: 13,854 lines at main `2245a4e`, 13,874 once the freeze notice landed, 13,893 after the
  m7_layer0 split replaced the two `include_str!`s of the old example file with one concatenation
  constant, and 13,906 after the task-11 hardening. The pin moved in the same commit as each
  intentional change, which is D26's deliberate-act rule.

## 7. `m7_layer0` is too large

**Claim as registered:** `crates/sparkflash-kernels/examples/m7_layer0.rs` is 8,100 lines.

**Verdict: FIXED in 954d7af; the review's number was not what the file measured.**

* Measured at base `2245a4e` with `wc -l`: **8,224 lines**, not 8,100.
* Split into `crates/sparkflash-kernels/examples/m7_layer0/` - `main.rs` plus one module per stage
  (stage0..stage9) - as **8,273 lines across 11 files**, no behaviour change. Full record, proof and
  the list of what could not be run on this host: `docs/build/m7-layer0-split.md`.
* Proof in the repository: `python3 ci/verify-m7-layer0-split.py` (verbatim slices from `2245a4e`),
  plus 17 CLI/validation argv paths with byte-identical stdout/stderr/exit against a pre-split
  binary. Device stages were **not** run (no GB10, checkpoint or cubins on the dev host).

## References re-checked, not claimed exhaustive

The D10-D13 dedupe (`c739f93`) removed a duplicate block; it did **not** make every `DECISIONS.md`
citation in the tree resolve. Checked in the task-11 follow-up: `crates/sparkflash-scheduler/src/engine.rs:18`
cited `docs/DECISIONS.md:151-155` for the `X-Request-Id` binding, which was already wrong at base
`2245a4e` - those lines are the Task-6 stage-9 extension inside D4 (Dense 264-byte Engram rows), not
the request-binding decision (D9, `docs/DECISIONS.md:272`). Fixed to `docs/DECISIONS.md D9` there;
this citation predates the dedupe and was not caused by it. No other `DECISIONS.md` line citation was
re-checked, and this note makes no claim that the rest are correct.

## What this note does not do

* It does not re-open the two kernel claims (1 and 2); they are kernel-dev's, and this branch does
  not touch `kernels/` or `build.rs`.
* It makes no claim about the review's wording where the registered task claim is terse; each
  verdict is tied to the source lines it checked.
* No cluster, window or device measurement was involved. Every "measured" number above is a `wc -l`
  or a test/script output from this host.
