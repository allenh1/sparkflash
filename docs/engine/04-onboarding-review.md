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

**Superseded by the contract.** `Drafter<W>` now carries the round (`open_round`, `verify`,
`settle`, `budget`, `ready`, `give_back`, `speculation`, the wall and an arm-owned `Rows`),
`HeadInput::Verify` heads rows in order and stops at the first mismatch, `Pass::prompt_end`
carries what a bulk chunk could not, `Limits` separates rows a pass from members a batch, and the
ledger derives its drafting rows from the drafter and the joint switch. A drafting walk that does
not declare the verify pass is refused at boot. GLM and Qwen now serve on `Runtime<Walk>` with
their own drafters and their fusion code deleted (960 and 561 lines out). Inkling followed, and
with it the last arm on its own shape: its `InklingLanes`, the serial group session and mirror,
the `Prompting` part machine, the resume preamble and the rewind trio are gone.

The device-only differences left for the cluster gate, reported by the two moves rather than
measured:

* GLM with no spec configured now takes a one-row `verify_batch` + `settle` where the frozen path
  used `forward_batch_greedy` + a host choose. The documentation says a one-row verify is the same
  bits; that is the thing to confirm on a GPU.
* GLM's ledger row for drafting is type-derived, so an instance with no drafts configured prints
  "the round path with no drafts configured" rather than off. The path does run the round.
* A profile prompt chunk outside `ChunkPolicy`'s documented set (for example 5000) now runs
  4096-row chunks, which can move block-drafter bits; a chunk below `MIN_PREFILL_ROWS` takes the
  token path per runtime chunk.
* After a give-back the runtime reports a reusable history and rewinds to the point, where the old
  session reported no reusable prefix: same tokens expected, different reuse decision.
* Qwen's joint-off now walks each lane's round alone, as the contract says, where its fusion
  stepped one plain N-row batch over all members. Same tokens expected, one one-row batch a lane.
* `Runtime::room` reserves `1 + budget(1)` = depth + 2 positions a step, not the fusion's
  `verify_rows` = depth + 1, and counts history rather than the arm's position while verified
  drafts are queued. It matters at the context boundary.

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

### The third move: Inkling, and the image path

Inkling moved onto `Runtime<InklingWalk>` in the same shape as GLM and Qwen. Its serving
construction is the runtime on every rank, whether the group steps one request or four: the serial
`GroupSession` + `mirror` path and the fused `InklingLanes` are replaced by one `GroupLanes` over
`Runtime<InklingWalk>`, and a single `ModelSession` is the runtime's one-slot adapter
(`RuntimeSession`). The kernels, the three chunk widths (4,096, 8,192 and 65,535) and the vision
tower are unchanged.

The image path is now the runtime's, and it is a contract rather than a marker:

* `Segment` carries the prompt's fresh images to `Walk::walk`; the runtime keeps them for the
  whole prompt and passes them with every chunk, so a chunk that holds part of an image still
  sees it.
* `Vision<W>` gains `encode` and `clear`. The runtime calls `encode` once, before a prompt's
  first chunk, with the whole prompt and its images, and every chunk after it injects their rows.
  `NoVision` is a text-only implementation with no bits of its own; the other arms keep naming
  it.
* A resumed prefix is accepted only when the images it holds are fed again at the same spans,
  grids and content keys, and no new image lands inside it. A changed image starts the sequence
  over, as the arm's own prompt path did.
* `held_images` reports the runtime slot's spans, so the host driver's prefix reuse sees what the
  arm holds.
* `Walk::rewind_slack` states how far past the rewind point a sliding cache can still go back
  (Inkling: its widest prompt chunk less the drafter's depth). A reply longer than that starts a
  changed turn cold instead of reading overwritten ring rows.

The no-drafter multi-lane step is resolved in the runtime, not deferred. `Drafter::drafts` is the
instance's answer (`PRESENT` remains the type's): a rank whose DSpark or lookup drafter is
configured off takes one plain token pass over every member — the old fused `Rows::Lanes` walk —
rather than a held decode a lane. The ledger, `room` and `max_batch` read the same predicate.

The device-only differences this move leaves for the cluster gate:

* Inkling's no-drafter step now runs the walk's `Rows::Lanes` path (through `Walk::walk` with
  `Pass::token_rows`) where the serial session's `decode` ran one row through `session.step`. The
  arm's docs say a one-lane `Rows::Lanes` walk is the decode step; confirm the logits are
  bit-equal on both the tiny fixture and the released checkpoint.
* With a drafter configured, a lane's round now goes through `Drafter::open_round` /
  `verify` / `settle` and the runtime's row share. A lone request keeps the drafter's own depth
  and a verify's rows are the same `Rows::Segments` walk; confirm the tokens and the speculative
  counters.
* With `SPARKFLASH_JOINT_ROUNDS=off`, each lane runs its round alone, one walk a lane. This is
  the contract's rule, not Inkling's old fused batch; confirm the tokens.
* A rank whose drafter is off reports the runtime's drafting ledger rows (type-derived) while
  `describe` says the switch is off; the boot line's ledger is the thing to read on the device.
* The image path encodes a prompt's fresh images once through `Vision::encode` and frees them at
  the prompt's end (or when the next prompt starts). Rank 0 runs the tower and every rank sums
  the rows exactly as before; confirm an image prompt, a text continuation of one, a changed
  image and a two-chunk image prompt (one whose span crosses the runtime's chunk width).
* The rewind point is the runtime's `point` plus `rewind_slack`, and the resume preamble is the
  runtime's. A continued turn, a changed turn after a long reply, and a turn that rewinds into
  the previous prompt all need the tokens compared with the frozen arm.
* After a drafted run whose last answer was not fed, the runtime reports a reusable history but
  takes the rewind point (the old session gave the drafts back and extended the whole history):
  same tokens expected, one more forward pass of the reply tokens, and only when the reply is
  longer than `rewind_slack` does it start over.

### The sealing pass: an arm supplies a walk, not a strategy

Three commits after Inkling's move close the last way an arm could supply its own strategy
(`6b07a1ff`, `20dd08b3`, `6261935f`; 9 files, +669 / -149, 148 of the deletions the per-arm
lane and pack dispatch). Phase 1's guarantee is now a property of the compiler and of two tests:
a model crate cannot implement a strategy, and `Pack` no longer names a family.

* **`FusedArm` is sealed.** `fused.rs:186` adds a crate-private `sealed` module and `fused.rs:200`
  makes `Sealed` the supertrait, so a crate outside `sparkflash-runtime` cannot implement it. The
  only implementors are `Runtime<W>` and `GroupLanes<A, W>`; every arm is a `Walk`.
  `tests/sealed.rs:53` scans every `.rs` under `crates/` and fails if a model crate implements
  `FusedArm`, and `tests/sealed.rs:93` names the walks as the arms' only strategy-facing trait.
  Cost: one line in the sealed module's doc says why ("the runtime owns every strategy; an arm is
  a `Walk`").
* **`OPEN_LANES` is gone.** `PackArm::open_lanes` (`pack.rs:124`) is required and takes the
  operator's lane count: `--concurrency` when set, otherwise the arm's own `Limits`. It returns
  rank 0's backend and the resolved count, and `serve_group_lanes` passes that count to the host
  driver. The `Option` that remains is rank 0 against a worker whose mirror ends, so the group's
  refusal is "rank 0 opened no lanes", never "this arm has no lanes". Qwen and GLM read
  `Runtime::limits().members` (4 and 8 at their own widths); `runtime.rs:2422` boots a world-2
  `GroupLanes` over a toy arm and asserts the lane width comes from `Limits`.
* **`Pack` is an identity, not a family list.** `Pack` is `{ descriptor, route }` (`pack.rs:769`)
  and the routes are serving shapes: `Reference` (DeepSeek's V4.1 engine), `Typed` (a
  `PackArm` behind the host driver) and `Dynamic` (Dory, Inkling, and a novel checkpoint). One
  table keyed by the registry's weights selector picks the typed arm (`pack.rs:714`), with one
  `PackArm` implementation an arithmetic; Qwen's FP8 release is a selector there, not a variant.
  `Pack::is_arm` is deleted. Two family checks are kept deliberately because their serving paths
  are distinct rather than selected: `is_inkling` (its `--rank` device group, `pack.rs:907`) and
  `is_dory` (its `--recurrence-rank` ring, `pack.rs:912`).

The evidence in the branch: `cargo test -p sparkflash-runtime` (74 + 2 + 1 + 2, 7 pre-existing
conformance stubs ignored), `cargo test -p sparkflash-cli` (exit 0, 275 lib plus the integration
suites), `cargo check --workspace`, clippy with `-D warnings` on both crates, and `cargo fmt
--check`. A new CPU test (`pack.rs:1892`) walks every registry id to a `Pack` and lays out an
`EnginePlan` at world 2, and `pack.rs:1966` boots each Qwen `PackArm`'s group lane width on the
device. The arm the device run could not measure is GLM: its tiny fixture's 32-wide hidden is
refused by the mixer's 256-column threads (a pre-existing fixture limit), so GLM's device lane
boot is covered only by the CPU world-2 plan.

What remains separate, and why it is not a defect:

* DeepSeek, Dory and Inkling still serve on their own group shapes; only the `PackArm` families
  (Qwen, Qwen FP8, Qwen3.6, GLM) take `open_lanes` into `GroupLanes`/`Runtime`. DeepSeek's move
  is phase 6; Dory and Inkling are `Dynamic` by decision, and their device groups
  (`--recurrence-rank`, `--rank`) are the reason their checks stay.
* No real two-process TP2 group was booted: the device host has one GPU. World-2 evidence is
  plan construction (CPU), the CPU `GroupLanes` over a world-2 runtime, and TP1 device runs of
  each `PackArm`.
### The generated checks now run

`sparkflash_runtime::conformance::run_check` was a stub at the moves' tip: every generated
`#[ignore]`d test panicked with "needs the arm's device fixture". It now has a fixture contract
(`WalkFixture::fixture(world)` returning a `Fixture` with the walk, its source, its
bytes-per-token and floor, its formats and its keep set) and the six checks are real: whole
prompt = chunked = token by token, alone = four at once, continued = cold, rewound = cold,
TP 1 = TP 2 = TP 4 within exactness (a named skip without a group), no unrecorded widening + the
bytes-per-token floor, and drafted = plain (a named skip for an arm that declares no drafter,
never a pass). A missing fixture is an error with code `fixture`, not a pass.

Each arm supplies its own: GLM's and Qwen's are their existing synthetic checkpoints opened
through the arm's TP=1 constructor (Qwen's is the 3.5-MoE spelling, because the hub cache's
2,560-wide DFlash-2 drafter is refused by the 128-wide fixture), Inkling's is `TinySource`, and
dense's is the checkpoint `scripts/dense-synthetic-checkpoint.py` writes, named by
`SPARKFLASH_CONFORMANCE_CHECKPOINT`. `scripts/onboard-gate.sh` writes that fixture when absent,
runs the checks, and reports PASS / SKIP / FAIL / NEEDS-FIXTURE, exiting non-zero on the last two.

The `engine_builder` device failure was the fixture, not the check. `hidden 256` and `vocab 64`
are correctly refused by the released-image boot check, so the fix is
`Glm5NextConfig::serving_fixture()` and `Qwen4ExpConfig::serving_fixture()`: one decoder layer at
the catalogue's widths and routed shapes. A separate test pins the tiny refusal, so the check
cannot quietly relax.

All five fixtures pass on the aarch64 device (details in
[04-onboarding-device-gate.md](04-onboarding-device-gate.md)). TP 2 and TP 4 still skip on that
host, which is one checkpoint-backed host's job.

### Phase 7, first arm: dense

The dense arm is the first arm on the typed weight API. `Proj` is
`Bf16(Mat<Bf16>) | Fp8(Mat<Fp8E4m3>) | NvFp4(Mat<NvFp4>)`, embed and head are `Mat<Bf16>`, and
every upload goes through `upload::<F>`, which checks the stored dtype and the scale count.
Vectors stay `Deviceptr` with the reason written at the type: `Mat<F>` is a strided matrix with a
format scale plane and a norm is neither. Each kernel is loaded once through `Bound`, which checks
the image's exported symbols and the catalogue's precision mask, so no launch site looks a symbol
up; `Kernels::load` no longer takes a name. The boot ledger names every part's stored and
arithmetic format, every vector and every packed copy, and refuses an arithmetic format that
differs from the stored one without a `widen` record (dense has one, the BF16 router bias read at
f32).

The ban is a test, not a convention: `crates/sparkflash-ops/tests/kernel_lookup.rs` scans every
`sparkflash-model-*` crate and the DeepSeek forward for `load_named`, `load_named_shared` and
`load_function`, and holds each file to a recorded count with a reason. A converted arm must
tighten its line; an unconverted one records "not converted yet: phase 7 runs by arm".
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
