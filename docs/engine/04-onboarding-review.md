# 04-onboarding — review of the plan

A review of [04-onboarding.md](04-onboarding.md) at `1e3cb3a9`, before implementation. It records
what was checked against the tree, what holds, what does not, and the order and ownership the
implementation takes. The plan itself is not rewritten here; where this page disagrees, it says so
and the plan is amended in the commit that acts on it.

## What was checked

Every `file:line` claim sampled was checked directly. The numeric claims marked *(survey)* in the
plan are checked only where the check is cheap; the rest are listed under "Open" below for the
audit pass.

| Claim | Checked | Result |
|---|---|---|
| `FusedArm` has 7 required methods and 14 defaulted | `runtime/src/fused.rs:185` | holds |
| `begin_prompt`/`next_part`/`history`/`rewind_prefix`/`give_back`/`ledger` defaults degrade the server | `fused.rs:246,259,287,311,374,398` | holds |
| `PackArm::OPEN_LANES` defaults to `None` | `cli/src/serve/pack.rs:121` | holds |
| `Pack::is_arm` is a negative match | `pack.rs:748` | holds |
| Every `--tp4-rank` group falls back to GLM's context | `cli/src/serve.rs:4161` | holds |
| Every non-Inkling lane group gets `SPARKFLASH_GLM_SPEC` | `web/src/config.rs:1773` | holds |
| Dense's chunk is a fixed 256 | `model-dense/src/lib.rs:110` | holds (`CHUNK_ROWS`) |
| Dense's `walk` already has segments + policy | `lib.rs:2661` | holds: `segments: &mut [(&mut Sequence, &[u32])]`, `policy: Option<WidePolicy>` |
| Policies are model-named constants | `ops/src/projection.rs:95-117` | holds (GLM, DORY, INKLING_TENSOR, DENSE) |
| `KernelSymbol` has no compute class | `ops/src/plan.rs:508` | holds: name, precision mask, phase |
| `CATALOGUE` is a const table | `ops/src/parts.rs:398` | holds: `&'static [PartKernel]` — a `const` assertion is feasible |
| `one_stack` holds the one-stack rules and caps the CLI at 34k | `ops/tests/one_stack.rs:130` | holds; the cap is `34_000` |
| The CLI is over the cap | measured | holds: 35,657 lines under `cli/src` |
| Neither CI file runs `sparkflash-ops`, `-runtime`, `-model-dense`, `-model-inkling2`, `-model-dory`, `-deepseek-forward` | `.gitlab-ci.yml`, `.github/workflows/ci.yml` | holds, both files |
| Qwen has `batch_exactness.rs`, `few_rows.rs`, `checkpoint_roundtrip.rs` | `model-qwen4-exp/tests` | holds; the runtime and dense have no `tests/` |
| Qwen's three generated checks are the source of `conformance!` | those files | holds; they are device tests, not CPU tests |
| `numerics/registry.toml` has Qwen3.8, GLM and DeepSeek, none for Inkling or Dory | `numerics/registry.toml` | holds |
| `03-runtime.md` still says DeepSeek keeps its own `tp4_reduce::DeviceExchange` | `03-runtime.md:266` | holds (row 30) |
| `design.md` says eight sequences share a speculative step | `design.md:118-119` | holds (the plan cites 117, which is blank); `design.md:118` also says 32 run at once |
| The three bit dependences are written down | `design.md:113`, `optimization-strategy.md:497`, `NUMERICS.md:281/284` | holds, all three; the resumed-lane sentence is at 284, the bullet starts at 281 |

## What holds up

* **The diagnosis is the strong part.** The seven causes are each a mechanism in the tree, not a
  preference: hooks with silent defaults (A), copied glue (B), arm-owned dispatch (C), format as a
  runtime value (D), hand registration (E), gates that do not run (F), and the separate program
  (G). Every one is visible in the files cited.
* **The "done" table is the right contract.** It is stated as properties a person cannot forget,
  and each property is mapped to the mechanism that makes forgetting fail. That mapping is what
  the rest of the page is judged against, and it is mostly complete.
* **Starting from the dense arm is right.** Its `walk` already has the target signature, so
  `Walk` is extracted from working code rather than designed in the abstract — the single most
  important de-risking choice in the page.
* **Making the RDMA exchange and packed weights the model for the other rows** is correct: they
  are already "on by default with one switch", which is the shape every other row wants.
* **The regression clause on phase 6** (revert any step that costs DeepSeek speed, record the
  measurement) is the right form of commitment for the only phase that touches the production
  scheduler.

## What does not hold, or is under-specified

1. **`Runtime<W>` is built twice.** Phase 1 puts the strategies inside `Runtime<W>` (the one
   `FusedArm`), and phase 6 replaces `FusedEngine`/`GroupLanes` with DeepSeek's tick, moving
   every arm again. The `Walk` trait survives both, but the glue written in phase 1 (chunk
   policy, lane pool, reuse, rewind, group mirroring) is written *inside the engine that is
   deleted*. Fix: make `StateLayout`, `ChunkPolicy`, `LanePool`, resume/rewind and the group
   mirror standalone runtime services that either engine calls, not methods of `Runtime<W>`. Then
   phase 6 changes which engine calls them, not where they live. This is a change to phase 1, not
   to the plan's thesis.

2. **`Pass` is only specified in mechanism 7.** The sketch in mechanism 1 takes `Pass` but never
   lists its variants; mechanism 7 says what `Walk` needs "beyond the first sketch" (token rows,
   a bulk chunk, paired chunks of several prompts, verify rows under an output constraint) plus a
   hook between layers and head for tick-ahead. If phase 1 ships a two-variant `Pass`, phase 6
   changes the trait and every arm. Fix: define `Pass` completely in phase 1, including the
   verify-under-constraint rows and the mid-walk hook, even though only dense uses them at first.
   The hook is the one item whose *placement* is load-bearing; get it wrong and tick-ahead
   (DeepSeek's "decide the next tick under the running one") has nowhere to live.

3. **`conformance!(Arm)` needs two things the page does not name: a fixture and a device.**
   Qwen's tests boot a device context and read a checkpoint fixture. A generated macro emitted
   into an arm crate therefore (a) needs a fixture the arm supplies — an associated
   `type Fixture` or a path constant on `Arm`, and (b) cannot run in the per-commit CI, which
   the plan itself says is weight-free and GPU-free. Fix: the macro emits `#[ignore]`-d or
   feature-gated tests and a `const` list of the checks the arm claims, and the compile gate is
   "the token exists", while execution is `onboard-gate.sh` on a GPU host. Otherwise phase 1
   makes CI need a GPU and the plan's CI argument inverts.

4. **Phase 0's "GLM and Inkling refuse `decode.tensor_cores=true`" is not bit-neutral at the
   deployment boundary.** A site file that sets `SPARKFLASH_DECODE_MMA=on` (or a profile with
   `decode.tensor_cores = true`) boots today and would refuse after phase 0. That is the intended
   signal, but it is a boot-visible change: it must be called out with the deploy that carries it,
   and the refusal message must name the switch and the arm, as the plan does for settings in
   mechanism 5. The alternative — warn and run — keeps the silent default the page is trying to
   delete.

5. **`ChunkPolicy` "from `Limits` and free memory" breaks bit reproducibility unless the tile is
   boundary-invariant.** The released arms are safe by construction (their present chunk sizes
   become inputs), and new arms are safe *if* the tile gives the same bits at every chunk
   boundary. That is exactly what DeepSeek's first two bit dependences say is not true of every
   path. So the free-memory term must be quantised to a small, documented set of chunk sizes and
   the conformance check "whole prompt = chunked = token by token" must be run at the *ends* of
   that set, not only at the default. Otherwise a busy host changes a new arm's bits.

6. **The ledger cannot be "computed from the types" in full.** Whether an arm drafts or has vision
   is a type (`Drafter`, `Vision`); whether it splits prompts and supports concurrent rows is a
   property of `Walk` the types do not see. Give `Walk` a declared capability value
   (`const PASSES: PassMask`, `const SEGMENTS: Segments`) that the ledger reads, and make
   "unstated" impossible by making the field non-optional. The plan's phrase "computed from the
   types" is achievable for drafting and formats, and should be stated as "from the types and the
   declared capabilities" elsewhere.

7. **`Mat<F>` is sketched without its second half.** "Carries the scales its format needs" is the
   whole problem: NVFP4's per-16 block scales plus a global, MX FP8's per-32 powers of two, FP8
   E4M3's 128x128 block scales, and the packed BF16 form's two-column words are four different
   layouts behind one marker. The plan should name `Mat<F>`'s fields and the scale layout per
   format in phase 3, because phase 5's GLM work depends on the E4M3 form and phase 4's NVFP4
   parts depend on the other. As written, "a weight is `Mat<F>`" hides the largest single
   design item in the page.

8. **The eight serving entry paths are counted but never enumerated.** Mechanism 5 derives the
   registry, console, conversion, scripts and CI from `ARMS`, but the entry-path list (the
   plan's own §"What onboarding costs today") is the thing that actually duplicates per model.
   Phase 2 should state which of the eight become one and which stay, with a test. Otherwise
   "one registration" is true of the table and false of the paths.

9. **Phase 6's gate is weaker than its risk.** A tick trace plus a numerics check plus alternated
   launches is the right *shape*, but the plan does not say what "identical" means for a trace
   that includes wall-clock-driven decisions (the time-based decode share). Fix: the trace records
   decisions and the inputs that produced them (elapsed time, queue), and the comparison is on
   the decision sequence for a replay whose clock is injected. Without an injected clock the trace
   gate is flaky at exactly the boundaries it is meant to police.

10. **Two stale statements are fixed in phase 0, but a third is not mentioned**: `design.md:118`
    ("Up to 32 sequences run at once") versus the claim that the code allows 32 while the doc says
    eight one line earlier — the page cites the eight but should also check the 32 against
    `MAX_LANES`/`DEFAULT_LANES` when it edits the file.

## Measured (see [04-onboarding-audit.md](04-onboarding-audit.md))

The audit pass replaced the *(survey)* counts with measurements. The ones that change a decision:

* **8 compile-stopping `match` sites in 3 files, not 9.** The ninth, `Pack::is_arm`, is a
  `!matches!` that compiles with any new variant — which is the plan's own point about it. The
  wildcard sites are the real list: `profile_toml_for_family`, `from_descriptor`,
  `device_family`, the dynamic family dispatch, `default_convert_world`, `shape_for`, and the
  web's `fit`/`hf` matches.
* **219 DeepSeek-image symbols, 127 named by the forward, 76 in the catalogue, 143 uncatalogued** —
  not ~205 / ~105 / ~100. The catalogue covers a third of the exports, not half.
* **The forward's tests are 7,713 inline lines and 371 tests**, not 11,154; there is no `tests/`
  directory. Scheduling is 5,703–6,321 lines, not ~4,500.
* **9 server-construction sites, not 8**, and mechanism 5 as written collapses none of them: it
  derives the registry, console, conversion, scripts, tests and CI, not the serving dispatch.
* **GLM created exactly 32 files**; "about 52 existing" does not reproduce (55 across the 15
  GLM-titled commits, 42 in the six-commit landing window).
* **The dense line citations drifted**: the logits copy is `lib.rs:2778`, the policy-less walk
  call is `:2799`, not `:2749`/`:2785`.
* **`one_stack`'s cap test is red today**, so phase 0's first commit has to raise it with the CI
  change or CI is red on its first run.

The audit also names three omissions the mechanisms do not cover: `HostDriver`/`ModelSession`
is a second engine the plan never replaces; tokenizer, chat template and response-engine
registration are hardcoded outside `ARMS`; and `Walk`/`PartExec` cannot express mid-walk
drafter taps, Engram state, per-sequence constraints, or request-level sampling. Those are
recorded there, not silently folded into a phase.

## Confirmed during implementation: the drafting gap

Reviewing the shared runtime against the first arm that drafts (Qwen) found a defect the dense move
could not expose, because dense has no drafter. It is recorded here because it changes phase 1's
definition of done.

* `Runtime::step` is `plain_step` (`runtime.rs:457`); nothing in the runtime calls `run_round`.
* The loss is **all drafting**, not only joint rounds: a lone request that used to draft and verify
  its own chain now gets one plain row (`fusion.rs:249-253` was the old path).
* **Three ledger rows were untrue**: Drafting and JointRounds from `VERIFY && Many`, and Runs
  hard-coded `true`; the joint-round row also ignored `SPARKFLASH_JOINT_ROUNDS`. Phase 8's
  `sparkflash onboard` prints the same rows.
* `Runtime::verify` is not yet a verify: it heads only the anchor row, never compares a draft with
  the model's choice, and appends every draft to history (`runtime.rs:202-243`); its own test
  asserts that shape. Wiring `step` to it as it stands would change output tokens.

The decision, taken here: **a released drafting arm stays on its own fusion until the runtime
drafts**, and `Runtime<W>` refuses a walk that has a `Drafter` but no round path, so the lie is
impossible by construction. This follows the plan's regression clause (an arm that measures slower
stays where it is) and keeps phase 1's guarantee honest while the drafting phase is still ahead.

What the runtime must gain before GLM, Inkling or Qwen can leave their fusion — the next phase-1
commit, shared by all three arms rather than designed from one:

1. A real `Drafter` contract (today it has only `PRESENT` and `describe`), lifted from the shape
   the three arms already share (`open_round`, `take_drafts`, `resolve_verify`, `holds`).
2. Verify rows headed in order with a stop at the first mismatch; `HeadInput` has only `Prompt`
   and `Token`, so it needs the verify form.
3. `room` for a step must reserve `verify_rows`, not one row.
4. `Limits.rows` conflates rows a pass and members a batch; a verify pass is
   `members x (1 + depth)`. The two must be separate numbers.
5. A home for the cost model (`JOINT_WALL`, depth) that the old arms had in their own fusion.
6. A seventh conformance check: **drafted = plain**, including a case where the draft is
   rejected. Its absence is why this got through.

A review of GLM's `Walk` — the second arm to declare one, and the first that drafts — added five
things the drafting contract must settle, and exposed a hole in the guard:

* **The guard needs a second half.** `GlmWalk` declared `Absent` while its `open` built drafting
  sequences and its `token_rows` ran the whole round inside a `TokenRows` pass. So an arm whose
  profile ships a draft head passed the `Drafter::PRESENT` refusal vacuously and the ledger
  printed "no drafter" for an arm that drafts. The rule must be: an arm whose profile or backend
  carries a draft head may not name `Absent`.
* **A one-row pass must not append drafts.** `Walk`'s contract says a token pass is one row a
  member; a walk that drafts inside it changes the shape the runtime reasons about.
* **The round cannot be copied.** GLM's walk copy differed from the served one in three places,
  each confirmed against the old code and each a policy the `Drafter` contract must name rather
  than inherit by accident:
  * a lone member truncated the drafter's own depth through `share_rows`; the served arm ran the
    drafter's depth with no row share (`serve.rs:980`). Rule: one member verifies the drafter's
    own depth.
  * with `SPARKFLASH_JOINT_ROUNDS=off` the copy still issued one `verify_batch` over every
    member; the served arm ran each lane's round alone (`serve.rs:273-304`, `rounds.rs:32-34`).
    Rule: off means each lane alone, and on Qwen off means no drafting at all (`fusion.rs:186-189`).
  * more than eight members: `Runtime::max_batch` reports `Limits.rows` (the 4096-row chunk)
    where `GlmFusion::max_batch` reported `MAX_ROWS` (8). The rank does not fail — `run_batch`
    splits rows into `MAX_ROWS`-row steps — so the change is the engine-level member cap, which
    matters only for a drafting walk because `verify_batch` refuses more than `MAX_ROWS` rows in
    one verify. Rule: a drafting walk's member cap is the rank's 8, or the contract splits the
    batch into verify-sized groups and commits each before the next step.
* **The hooks belong on `Drafter`, as required methods.** `Walk::ready` and `Walk::speculation`
  default to false and zero, so a drafting arm that forgets them silently loses multi-token runs
  or reports nothing — the same class of silent default the plan exists to remove. Putting them
  on `Walk` was staging, not the shape.
* **`begin_prompt`'s new position test assumes `position() == history.len()`.** It is right for GLM
  and dense. An arm whose position differs by convention (Dory's point one id short, Qwen's
  respelled last id) would fall to rewind or cold on every turn with the same tokens and nothing
  reporting it, so the conformance set needs a check that a continued turn forwards only the
  suffix.
* A backstop: an arm's `kv_plan` must not swallow an error into an empty layout, because
  `StateKind::None` makes the runtime skip the rewind while still truncating history.

Two smaller findings from the same review: the extra LM-head work per prompt chunk (old shape
`forward_batch(chunk, usize::from(is_last), 0)`, `device.rs:6389`) is restorable because `head`
is a separate call, but in a TP group the head rows are a collective (`tp.rs:545-559`), so the
per-chunk cost can be a collective and not only a GEMM; and `Pass::bulk(rows)` carries no prompt
end, where the old code limited drafter commits to a window before the prompt's end
(`device.rs:6394-6400`), which the drafting contract must carry.

## Order and ownership

The plan's phase order is kept. Two changes are made to how it is executed:

1. Phases 0 and 1 do not wait for each other where they touch different crates: phase 0 owns the
   ledger, the CI shards, the refusals, the leaks and the numerics entries; phase 1 owns
   `sparkflash-runtime`'s new modules and the dense arm. `runtime/src/ledger.rs` is phase 0's
   for the duration, and phase 1 reads it, never writes it.
2. Phase 3's first half (compute class, catalogue assertion, `WAIVERS`, legacy row policies)
   runs in parallel with phase 1, because it is confined to `sparkflash-ops` and changes no
   bits. Phase 3's second half (shared token bodies, `Mat<F>`) follows phase 1/3a.

Concretely, the first window is:

| Agent | Scope | Branch |
|---|---|---|
| phase 0 | CI shards, tensor-core refusals, ledger rows + boot refusal, `sparkflash arms --ledger`, numerics entries for Inkling and Dory, the two environment leaks, the two stale doc statements | `onboarding/p0-gaps` |
| phase 1 | `Walk`, `StateLayout`, `ChunkPolicy`, `LanePool`, `Pass` (complete), `Runtime<W>`, `conformance!` skeleton, dense arm on `Walk` | `onboarding/p1-walk` |
| phase 3a | `ComputeClass` + row band on `KernelSymbol`, the `const` assertion over `CATALOGUE`, the first `WAIVERS` table, row dispatch moved into `ops` with each arm's present thresholds as named legacy policies | `onboarding/p3-kernels` |
| audit | re-measure the *(survey)* counts, enumerate the eight entry paths, verify every `file:line` in the plan at the phase-0 tip | read-only |

The cluster is needed only for `served-gate.py` and `onboard-gate.sh`; phase 0 and phase 3a gate
on unit tests and `cargo check`/`clippy` alone. Phase 1's dense arm move gates on `cargo test -p
sparkflash-model-dense` (CPU reference) first, and on a served gate once a cluster window is
taken.
