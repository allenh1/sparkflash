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
`SPARKFLASH_CONFORMANCE_CHECKPOINT`, or written by the fixture itself when the switch is unset
(see [the device window's two gaps](#the-device-windows-two-gaps-a-catalogue-hole-and-a-fixture-checklist)).
`scripts/onboard-gate.sh` also writes that fixture when absent, runs the checks, and reports
PASS / SKIP / FAIL / NEEDS-FIXTURE, exiting non-zero on the last two.

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

The second arm is Qwen, and it is the shape the rest followed: 11 string lookups became 2
(the two loaders inside `Bound`), the prefill and drafter files lost theirs entirely, and the
table went from 15 lines to 13 with no other crate's line touched. GLM (14 to 2), Inkling
(6 to 2, across the device, tower and drafter files) and Dory (4 to 2) then took the same shape,
each with its own local `Bound`, its typed uploads (`Mat<Bf16>`/`Mat<Fp8E4m3>` for GLM,
`Mat<Bf16>`/`Mat<MxFp8>`/`Mat<NvFp4>` for Inkling, `Mat<Bf16>` for Dory) and its own boot
ledger; GLM and Inkling record their widenings (BF16 bias read at f32, the L2-clip gains).
Every model crate is now two loaders plus typed matrices; DeepSeek's forward is the last arm and
phase 6 folds it anyway. The packed copies in these three arms still flow through the shared
`PackedSet` handles rather than as their own ledger rows - a deliberate gap the report states. Six symbols Qwen already
launched were missing from the catalogue and were added to its existing entries, which is the
cost a conversion pays when the catalogue was written from the launches it could see. Weights load
as `Mat<Bf16>`, `Mat<Fp8E4m3>` and `Mat<NvFp4>` through `upload`, vectors stay `Deviceptr` with
the reason at `Uploader::vector`, and the boot builds a `FormatLedger` whose widen log is empty
because this arm reads every weight at its stored format. Device parity is unmeasured on this
host; the report names the thirteen device targets a CUDA host should run rather than implying
they passed.
### The GLM capture, and what a four-rank boot needs

The arm does boot at world 4 on the branch. Three windows on the cluster (head `sg1-oneill`, four
DGX Sparks, GLM-5.3-Flash over 1,048,576 tokens) established it, and the third measured it:

* Window one failed in 8 s with `the bootstrap gather's plan digest differs by rank (rank 0 sent
  0xb067407, rank(s) [1, 2, 3] differ)`. The cause was the window, not the code: the binary had
  been installed on the head node only, and the console's swap began copying it to the three peers
  while the ranks were already starting. Rank 0 ran the branch (whose `Glm5NextConfig` Debug
  string gained fields in the phase-1 merge) and the workers ran the old build, so the agreement
  word differed. The arm never loaded weights; the window cost 4 m 07 s and restored.
* Window two printed the agreement word from all four ranks and booted: every rank reported
  `5b3e9a9f0b067407`, context `1048576`, configuration `bc832624869ede08`, 35808 bytes. The trace
  that made the component nameable is now part of the boot output (`serve.rs`, one line a rank).
* Window three installed the binary on **all four** ranks, verified the four hashes, waited for no
  sync job, and only then swapped: `/readyz` in 21 s, two bit-identical captures, a forced
  capture, a compare, and the served gate's lanes check PASS (41.4 tok/s alone, 54.7/54.8 with
  four at once). Restored on `435179d6` with four ranks in 4 m 30 s.

The measured delta is in [the GLM report](../numerics/glm53/v2-fp8-token-body.md): +1.373e-01
mean NLL per token against version 1, 93.38 % top-1 agreement over 997 positions, and a per-prompt
spread that runs from retrieval at +4.2e-03 to raw continuation at +4.1e-01. It is the tip's
delta, not the decode body's share, because version 1's reference predates the move and the body
both. Qwen's decode move was +7.0e-04 with 99.39 % agreement, so this is a large served change
and the page says so rather than calling it a rounding note.

**The window's separation capture is not evidence, and that is its finding.** The same window
served the build with `SPARKFLASH_DECODE_MMA` unset and with it set to `off` and got two
bit-identical captures. The switch was inert: `DeviceRank::gemv` routed the one-to-eight-row FP8
parts to the tile's few-row body by shape alone and nothing in the decode path read the switch.
The arm's `TOKEN_DECODE` now states what it runs ("the FP8 parts on the tile's few-row body, the
BF16 parts on the tiled GEMV"), the CLI's GLM profile check accepts `decode.tensor_cores = true`
from that same constant, and the switch selects the body or the tiled GEMV. The separation
capture must be retaken on a build after the fix before version 2 declares; until then the delta
above is the tip's, not the body's, and is not decomposed.

Two operational facts the plan's gate should carry:

1. **A four-rank boot needs the same binary on every rank, verified before the swap.** The
   console deploys a changed binary to the peers as part of the launch, so an operator who installs
   on rank 0 alone races that copy. The check is four `sha256sum`s and one wait;
   `scripts/onboard-gate.sh` and the served gate should make it, because the failure mode reads
   like an arithmetic disagreement and is not one.
2. **A failed swap rolls the selection back and leaves the cluster stopped.** The restore is
   `/api/actions/launch` (the selected model is already the production one), not a second swap;
   a swap to the selected model is refused with `already the selected model`.

Both windows' reports and artifacts are under `/home/allenh1/cluster-tests/2026111*` on the head
node. The cluster was over its window for 4 m 07 s, 5 m 23 s and 4 m 30 s respectively, and was
verified serving the production model card, build and four ranks after each.
### Inkling's first reference

The fourth window pinned Inkling-2-Small's version 1 instead of moving it. The arm was launched
by its site's own `scripts/inkling-serve-group.sh` with `BIN` at the branch tip (`5a092d5a`),
`LANES=4`, over 1,048,576 tokens; the group answered `/readyz` in 22 s and reported
`world=4 profile=reference plan=tp arch=sm_121a`, which is the configuration the registry had
registered but left uncaptured. Two captures are bit-identical (`387647bc…`), so the served
arithmetic is deterministic, and `served-gate.py lanes` passed twice on the same boot.

Because there was no earlier reference, the capture declares no change: it fills the entry at
version 1 and pins the arithmetic the decode change lands. The report says exactly that, and keeps
the two things it cannot show - the distance from the row-GEMV decode and the speed - in an
explicit list rather than implying them. `numerics.py ci` now reports Dory as the only uncaptured
configuration.
### Dory, and a registration that could not be served

The fifth window captured Dory1.0-4B's first reference as a single process at world 1: `/readyz`
in 3 s, two bit-identical captures (`99715d00…`), and `check --reference` passing on two separate
boots. Restore took 18 s after a 9 m 56 s window.

It also found that the phase-0 registration was unservable. The release carries **two** recurrent
loops, and `DoryStage::of_rank` refuses a group whose width does not divide them, so the ring's
legal widths are 1 and 2 - the world-4 entry could only have been served by rewriting
`n_recurrent_loops` in a host-side config view, which changes the model rather than measuring it.
The window deliberately did not do that. The registry now holds a world-1 reference and a
`[[retired]]` entry for world 4 that says why.

The ring also reports a **world-1** numerics key by design: `recurrence.rs` serves rank 0
through `serve_backend`, whose comment claims each rank runs the same kernels on the latent rows
so the arithmetic is the TP=1 arm's bit for bit. That comment cites `tests/recurrence.rs`, and
there is no such file - the claim is untested. Until it is, the world-1 reference is the only
measured Dory configuration, and a world-2 capture plus a ring-versus-single-process test are the
named follow-up rather than a second reference invented to match a broken registration.

Phase 5's three arms are now measured: GLM's declared change with a delta, Inkling's and Dory's
first references with their speed gaps stated. `numerics.py ci` reports no uncaptured
configuration.
### The build the early cards could not name

Three of the phase-5 artefacts - Inkling's anchor, its v1 anchor-forced capture and Dory's world-2
ring - were taken on the same binary, `ef3d022e...`, and all three recorded `commit: null` with
"the tree named by --repo is not a git work tree with a HEAD". The captures ran from
`/home/allenh1/rsp-0611`, a staged source tree with no `.git`, so `--repo` could not name a
revision. The source tree was staged from commit `a6ef32e4`: `Cargo.toml`,
`crates/sparkflash-model-glm5-next/src/lib.rs` and `crates/sparkflash-cli/src/serve.rs` in the
staged tree match that commit byte for byte. The registry file was restaged afterwards, so it is
the source tree's identity that is recorded, not a hash of the whole tree.

The three cards now record that inferred commit and say how it was inferred; the fingerprints did
not change, because the server card is outside the fingerprint. The mechanism is closed rather than
noted: the binary embeds its commit at build time, `/v1/models` reports it as `build.id` (with
`-dirty` when the tree was dirty), the serve boot line names it, and `capture` records it as
`server.commit` instead of reading `--repo` first. A new reference or anchor whose card names no
commit is refused; the references committed before the record are grandfathered by the absent
`server.build_source` key.
### The served A/B: what the harness got wrong

`scripts/tick-trace-ab.py` was written against "the server", but it launched a single serve
process. That process takes the reference engine, which never constructs `Batch`, and
`SPARKFLASH_TICK_TRACE` is only read in `Batch` on the four-rank device path - so the committed
harness would have produced empty traces and reported nothing. It now takes `--site-script`,
launches the site's group through `serve-four-node.sh` with `SPARKFLASH_BIN` naming the build,
and the launcher forwards `SPARKFLASH_TICK_TRACE` to the ranks (only rank 0 writes it).

The two builds for the A/B are staged on the head node: the frozen pre-move build with the trace
emission cherry-picked (`512c9910`; the emission commit `de642922` changes `tick.rs`, which the
move `4d144208` introduces, so the literal pre-move tree cannot take it as a cherry-pick) and the
branch head. Their `tick.rs` and `batch.rs` are identical, which is what makes the comparison a
decision comparison rather than a source comparison.
### The first served A/B: three pairs identical, one admission tick apart

The harness finally ran on the four-rank group at `d126c66c` (old build `512c9910`, new
`d126c66c`; both staged on all four ranks, all four boots at 26.5 s). Four pairs, one rep, eight
launches in 4 m 23 s:

| pair | decisions | verdict |
| :-- | --: | :-- |
| 1 stream, 2,048 prompt | 33 | identical |
| 8 streams, 2,048 prompt | 61 | identical |
| 1 stream, 16,384 prompt | 38 | identical |
| 8 streams, 16,384 prompt | 87 | **different** |

The difference is at the admission boundary and nothing else diverges: at tick 9, decision 21,
the old build places seq 1 (`place seq=1 lane=3 shared=0 start=0 target=16414 path=bulk:22`) and
the new one holds (`hold 0:22`). Old admits sequences 0, 1 and 2 before the tick's prefill; new
admits only 0 and defers 1-7 to tick 10, and the remaining 86 lines are the same sequence shifted
by that one tick.

By the plan's rule - "a pair that differs only in an admission tick is reported, not excused" -
the A/B **fails**. What the rule cannot say is whether the deferral is policy or arrival timing:
concurrent admissions depend on when rank 0's stream receives the requests, and one rep cannot
separate the two. So the pair was repeated three times, and the same-build traces were compared
with each other - the comparison the first run had not made.

| same-build pair | verdict |
| :-- | :-- |
| old r0 / r1 | different at decision 21 (r0 places, r1 holds) |
| old r0 / r2 | different at decision 21 |
| old r1 / r2 | different at decision 42 (`hold 1` vs `hold 0`) |
| new r0 / r1 | different at decision 42 |
| new r0 / r2 | different at decision 42 |
| new r1 / r2 | identical |

Every cross-build difference in the three reps falls at decision 21 or 42, which are exactly the
decisions where two launches of the *same* binary also differ. At decision 21 the old r0 launch
places and all other five traces hold (its clock at tick 9 was 1144 against 1373-1660); at
decision 42 the split is by launcher and rep, not by build. The cross-build pairs reproduce the
first run's shape at r0 and then disagree at r1, which is what arrival timing looks like.

A third run staggered the requests (`--stagger 0.25`, one every quarter second) so each tick's
admission set stops being a race. All three cross-build pairs still differ, and the same-build
control still differs in the same place (old r0/r1 and r0/r2, new r0/r1 and r0/r2), at the
admission of the seventh request: tick 10's batch, or deferred one tick. Both builds emit both
variants across their own launches.

The harness can now say what that is. `--diff --unordered` compares the traces as multisets of
decisions, and on this run every cross-build pair is **identical as a multiset: 87 decisions,
none added, none removed** - only the tick at which one admission lands differs, and the same
build differs from itself in exactly that way. `--ignore-tick` (order kept, ticks dropped) still
reports a difference, which is what makes the distinction visible: the two traces are a `place`
and a `hold` swapped, not a different set of decisions.

**Conclusion, stated at the strength the evidence supports:** across four pairs, three staggered
repetitions and same-build controls, the two builds run the same 87 decisions; the only
difference is the tick boundary at which one admission is placed, which is not stable for either
build on its own. Phase 6 step 4's claim is therefore "the moved engine makes the same
decisions as the frozen one, up to tick-boundary admission jitter", not "identical traces". A
strictly identical trace would need admissions that no real-time scheduler decides by arrival,
which is a property of the workload, not of the engine.

Two build facts this run exposed, both now checked before any window stops the cluster: the
binary embeds `numerics/registry.toml`, so a stale build can ship a registry that disagrees with
the tree - here an invalid `[[retired]]` field stopped the binary at boot - and the launcher `cd`s
into the repo tree on every rank, so the tree must exist on all four.
### The tick engine: dense on DeepSeek's scheduler

Commits `ededb08b` and `0807eed5` (the engine and the runtime primitives it drives),
`27328381` (the tick-ahead hook), `ee722ef6` (dense's move) and `5d9a0d01` (its fixes) land
phase 6 step 4's remaining part and step 5's first arm. `crates/sparkflash-runtime/src/engine.rs` is the engine; dense's
single-process serving path is the first arm on it. `FusedEngine` and `GroupLanes` are not
deleted: every other arm still serves through them, and so does dense's tensor-parallel group.

**Shape.** `TickEngine<W: Walk>` owns the scheduler's state:

* the pool's `RuntimeSlot`s, one a lane, and the list of free lanes; a lane given back keeps its
  history for the next prompt;
* each lane's bookkeeping - the prompt tokens it holds, the image spans those rows carry, the
  request it serves, when it was last given up, and where its decoder-layer skip ends;
* the admission queue (prompts waiting for a lane), the decode rows callers have fed, the clock,
  the cost model (`chunk_ms`, `row_ms`, `since_chunk_ms`) and the tick counter;
* the trace sink (`SPARKFLASH_TICK_TRACE`; one flush a tick; nothing recorded when unset) and
  the tick-ahead plan.

`Runtime<W>` stays the executor. The engine drives it with `start_prompt_slot` (resume and
prompt bookkeeping, no rows), `run_part_rows_hooked` (one chunk of exactly the rows the engine
decided), `step_mode_slots` (plain rows or the drafter's round, with the rows it walked),
`give_back_slot` and `release_slot`. `Walk`, `Drafter` and `Pass` are unchanged: no arm has a
new item to supply. `TickServe`/`TickSession` are the serving thread and the handle: the host
driver still opens one session a request, a session posts a prompt or a decode row and blocks,
and the thread turns each wave of jobs into one tick.

**What is shared with `tick.rs`.** Every decision is the function's, called with the same
inputs the old scheduler passed; none is restated in `engine.rs`:

| Decision | Function |
| :-- | :-- |
| which free lane, and its shared prefix | `pick_lane`, `common_prefix`, `reusable_shared` |
| the prompt order and the decoders' share | `prefill_order`, `hold_prefill`, `DecodeShare` |
| the prompt path and the chunk boundary | `bulk_from`, `tick_rows`, `bulk_budget`, `whole_chunk_fits`, `bulk_rows` |
| the decoder-layer skip | `runs_decoder`, `skip_end_after`, `decoder_resume` (DeepSeek's config only; dense has no decoder stack) |
| the step's shape | `decode_plan`, and `share_rows` through `Runtime`'s round |
| the cost model | `StepModel`, `band` |
| the trace | `Decision`, `TickTrace`, `TraceSink`, `Clock` |

`TickConfig` carries the values. `TickConfig::deepseek` is the replay's profile;
`TickConfig::for_walk` narrows it to an arm - the arm's `ChunkPolicy` chunk and
`Limits::rows`, no decoder skip (only DeepSeek has the layers), no row exchange. Dense's chunk
is still 256 rows because `for_walk` reads the same `ChunkPolicy` its runtime read, and
`run_part_rows` is given that number; a CPU test holds a 600-row prompt to the 256/256/88
chunks the old runtime ran.

**Tick-ahead.** The decode plan is decided in a `WalkHook` between the last layer and the head
of the tick's prefill chunk (`Ahead`), then used by that tick's decode. This is the first user
of `Pass::hook`, and the CPU walking test asserts the hook is reached. When no chunk runs in a
tick, the plan is decided at the decode point instead. It is the host decision DeepSeek overlaps
with device work; it is not DeepSeek's device-link `Ahead` (reading ahead of the device link),
which stays in its forward crate.

**What moved for dense.** `DenseDeviceBackend::enable_fusion` starts a `TickServe` over
`DenseWalk` instead of a `FusedEngine` over `Runtime<DenseWalk>`. The walk, its kernels, its
256-row chunk, its profile switches and its ledger rows are unchanged. Its generated checks
drive the tick engine: `WalkFixture::tick_engine` says so for dense, so
`whole = chunked = token`, `alone = four`, `continued = cold`, `rewound = cold` and the TP check
run the engine (and the token-by-token comparison still checks the engine against the arm's
direct chunks).

**Named deviations**, rather than implied:

1. **The grouped path is not on the tick engine.** `enable_group_fusion` and `mirror_group`
   still build `Fusion::group` over `Runtime`, so dense at TP>1 has the old schedule. The
   worker frames carry a prompt and a part, not the rows the engine decided, and
   `mirror_lanes` runs `begin_prompt`'s `ChunkPolicy` chunk on every worker. A frame that
   carries a chunk's rows (or a part that carries a row count) is the next slice. TP=1 is the
   path this branch actually moved.
2. **The resume position is `resume`'s, not `pick_lane`'s arbitrary prefix.** DeepSeek can
   restart a lane at any shared prefix because its state is positional; a walk with a recurrent
   region can only go back to a checkpoint. The engine chooses the *lane* by `pick_lane`'s
   shared prefix and records the *start* `start_prompt_slot` chose (history end, rewind point,
   or zero). The committed engine trace shows this: seq 4 shares 16,384 rows with seq 1's lane
   and still starts at 0, because the lane's checkpoint is at its prompt's end. Making that
   arbitrary start work is a device decision, and it is not claimed.
3. **A decode row runs only when its caller fed it.** `tick::replay` decodes every sequence in
   its decode phase each tick because the device scheduler owns the token. The engine's plan
   counts the rows that can run. This is the one structural difference from the replay, and it
   is why the engine has its own committed trace rather than the replay's: a prompt that
   finishes its prefill is headed in that tick and decoded in the next.
4. **A `Spec` plan on a non-drafting arm runs one plain row and is traced `plain`.** There is no
   speculative row to take; calling it `spec` would name a row the arm never walked.
5. **Adoption, lane growth, paired prompt chunks and preemption are not implemented.** Dense's
   pool is its concurrency, and the runtime has no way to copy another lane's cache. They are
   DeepSeek device decisions in the same list as the unverified items.

**Measured on this host (CPU; it has no GPU).**

* `crates/sparkflash-runtime/tests/tick_engine_trace.rs`: the engine's fixed 30-tick workload
  produces **73 decisions**, pinned in `tests/data/tick_engine_trace.txt`: 4 `place`, 30
  `hold`, 19 `prefill` and 20 `decode`. It exercises the bulk path (30,000- and 25,000-row
  prompts at 2,048/4,096 rows a chunk), the exact path (the 80-row prompt at 64 + 16), the hold
  (9 `hold 1`), prefix reuse across lanes (16,384 shared rows) and the chunk sizes the scripted
  costs select. A second test drives `TickServe` with two sessions: each answers, a parked
  lane's four tokens are resumed by the next prompt, and the walk was asked for the tick-ahead
  plan.
* The generated checks, forced onto the tick engine through a CPU fixture
  (`conformance::tests::the_generated_checks_drive_the_tick_engine`): whole = chunked = token,
  alone = four, continued = cold, rewound = cold and no unrecorded widening **pass**; TP is a
  named skip (the fixture is one rank).
* Dense's chunk test: `the_tick_engine_hands_dense_256_row_chunks` holds a 600-row prompt to
  `256/256/88` and the config's chunk to `CHUNK_ROWS`.
* `cargo test -p sparkflash-runtime -p sparkflash-model-dense`: runtime, 77 library tests (7
  conformance stubs ignored) plus 2 engine-trace, 2 sealed and 1 tick-trace integration tests;
  dense, 24 library tests (7 conformance stubs ignored); 0 failed. `cargo clippy
  -p sparkflash-runtime -p sparkflash-model-dense --all-targets -- -D warnings` and
  `cargo fmt --all -- --check` are clean.

**Unverified without a cluster.**

* The device walk: dense's kernels, its bits, and prompt = chunked = token on the GPU.
* Lane memory on the device: `room_slot`, eviction of an idle lane, and dense's lane mapper.
* Prefix reuse and rewind on a device sequence: `Sequence::rewind`/`checkpoint` and the resumed
  start; only the CPU walk exercised it here.
* The live trace: `WallClock` readings, the per-tick flush and the rank-0 rule; the engine is
  constructed only on the single-process path, so no live trace was produced.
* Speed: no tokens/s, no time to first token, no step time. Nothing here is a measurement of
  whether the tick engine is faster than `FusedEngine`; that is the A/B the regression clause
  asks for.
* The group: no TP2/TP4 boot ran, and the group is not on the engine anyway.

**The gate a window must run.**

1. `scripts/onboard-gate.sh dense` on the GPU host: writes the synthetic checkpoint, runs the
   six generated checks through the tick engine, and reports PASS/SKIP/FAIL/NEEDS-FIXTURE.
2. Dense's served gate at one and four lanes (`served-gate.py lanes`), then a numerics check on
   a freshly booted server as `NUMERICS.md` requires.
3. Alternated launches of the pre-move dense build and this one at 1 and 8 streams and at 2K and
   16K prompts, several reps each, comparing token streams and each request's reported
   `cached_tokens`; a live `SPARKFLASH_TICK_TRACE` from each launch, diffed as decisions. A pair
   that differs only in an admission tick is reported, not excused.
4. Only if the group is to be claimed: a TP2 or TP4 boot. That needs the rows frame first.

**What the next arm (GLM) needs.**

1. The grouped path's rows: a frame (or a `next_part_rows`) that carries the rows rank 0's tick
   decided, so `mirror_lanes` runs the same chunk on every rank; otherwise GLM's group stays on
   `FusedEngine` and only TP=1 moves.
2. `TickConfig` from GLM's profile: its 4,096-row prompt chunks, `rows_max` as its verify
   width, `decoder_skip = false`, `exchange = true` for a group, and `batch_spec` from its MTP
   or DFlash drafter.
3. The drafter plan: the engine already calls `decode_plan` and records the rows
   `Runtime::step_mode_slots` walked; GLM needs a fixture that pins `Spec`/`Shared`/`Plain`
   selection and the held-run drain, and its `Drafter::wall` feeds `share_rows`.
4. A group mirror for the tick engine: rank 0 broadcasts `LaneRequest`, `LanePart`,
   `LaneStep`, `LaneGiveBack` and `LaneRelease` (plus the rows frame) before running; the
   workers already replay through `Runtime`.
5. Its own A/B. Under the regression clause, an arm that measures slower stays on
   `FusedEngine`, and the plan records why.

Items 2 and 3 are done in the section below; item 1's frame and item 4's mirror are still
missing, and item 5 has not run.

### The drafting arm and GLM on the tick engine

Commits `d49dc2ba` (the engine's drafting profile and its CPU gate) and `f92bbe05` (GLM's
single-process move). `TickConfig::for_walk` no longer narrows DeepSeek's profile to a plain
arm: a drafting arm's config carries the profile's chunk and rows, its verify width as the
member cap (`TickConfig::members`, read from `Limits::members`), its drafter's depth (`k`) and
floor, and the `shared_rounds` switch. `for_runtime` sets that switch from
`Runtime::joint_rounds()`, so the engine's plan is the runtime's own rule, not a second one.

**The three shapes.** A drafting arm's decode plan is one of three, and each is pinned by
`crates/sparkflash-runtime/tests/tick_engine_drafter.rs` on a CPU walk with a drafter:

* `Plain`: a batch wider than the verify width, or a rank whose joint rounds are off, steps
  one row a member in groups no wider than `members`. `a_plain_batch_steps_one_row_a_member`
  feeds two rows, sees one `TokenRows(2)` pass and no verify, and the trace says
  `plan=plain`.
* `Shared`: several decoding rows whose anchors fit the verify width run one joint round.
  `a_fitting_batch_verifies_in_one_shared_round` feeds three rows, sees exactly one `Verify`
  pass of three members (not one a lane), and the trace says `plan=shared`.
* `Spec`: one decoding row runs its own round.
  `a_speculative_round_keeps_its_accepted_rows_only` hands the drafter a chain whose second
  draft is deliberately wrong: the round keeps the anchor and the first draft, the request
  receives two steps, the history ends at the accepted draft, and the rejected token never
  reaches the history. The trace says `plan=spec`.
* The tick-ahead plan is not an execution. `a_stale_tick_ahead_plan_does_not_skip_a_round`
  injects a plan for one row that names no round; the engine re-decides and the verify runs.
  The decode path filters `DecodePlan::None` for a nonempty row set, and `with_ahead` is the
  gate seam that makes the case reachable.

**GLM.** `Glm5NextDeviceBackend::enable_fusion` now opens `TickServe` over `GlmWalk` instead of
`Fusion::local` over `Runtime<GlmWalk>`. The executor is the same `Runtime`: a prompt is still
`run_part_rows` in the `ChunkPolicy` chunk, a decode is still `step_mode_slots` with the
runtime's `Round`, and the round is the same `GlmDrafter` verify and settle. `TickConfig` is
GLM's profile: `DEFAULT_PREFILL_ROWS` **4,096** rows a prompt chunk; `MAX_ROWS`
(`GEMV_MAX_BATCH`) **8** as the verify width, anchors plus drafts a step;
`decoder_skip = false` (only DeepSeek has decoder layers); `exchange = false` at TP=1 (no row
exchange); the
MTP/DFlash depth in `k`. The CPU shape tests pin the config and the chunks: a 5,000-row prompt
runs **4,096 then 904** rows, the same split the runtime's `ChunkPolicy` chose before, and
three decoding lanes take **one shared verify of at most 8 rows**. GLM's generated checks now
have `tick_engine() == true`, so `whole = chunked = token`, `alone = four`,
`continued = cold`, `rewound = cold`, `drafted = plain` and `no unrecorded widening` run
through the engine's protocol exactly as dense's do.

**Which path changed, and why the tokens cannot move on paper.** GLM's TP=1 path is the only
released arm's path this branch changed. Dense's `for_walk` config gains `members`, but
`Limits::new` makes members equal rows, so its plans and its committed trace are unchanged;
DeepSeek's `TickConfig::deepseek` sets `members = 64` to match its row width. GLM's tokens
cannot move on paper because the walk and the runtime calls are the same: the engine chooses
*when*, not *what* - the same 4,096-row chunks, the same one-row plain step, the same `Round`
with the drafter's own depth and the same `verdict` settlement. The scheduler is different
(tick boundaries, the hold, the decode share), and a greedy token does not depend on when its
chunk ran. The one arm-level property this leans on is `drafted = plain`, which the generated
check measures on the device; it is **unverified here**, so the paper argument is not a
measurement.

One non-default path is named rather than implied: with `JOINT_ROUNDS=off` the engine plans
`Plain` for a batch of decoding lanes - one plain row a member, no draft - where the fused
runtime ran one round a lane. The switch is off only when an operator sets it; the default
is joint, and the A/B above runs the default. The emitted tokens are still the
`drafted = plain` property's, but the drafting itself is off on that path, so if the off
setting must keep drafting the engine needs a fourth execution shape (a lone round a
member) before it may replace `Fusion::local` for that operator.

**Measured on this host (CPU; it has no GPU).**

* `cargo test -p sparkflash-runtime --test tick_engine_drafter`: 5 passed - the three shapes,
  the rejected draft and the stale ahead plan.
* `cargo test -p sparkflash-runtime`: 87 passed, 7 ignored (77 library tests including the
  existing committed engine trace, 2 sealed, 5 drafter, 2 engine-trace, 1 tick-trace).
* `cargo test -p sparkflash-model-glm5-next`: 93 passed, 8 ignored (71 library tests, 11
  `device_forward`, 4 `engine_builder`, 7 `serve_tiny`). The new CPU tests are
  `the_tick_config_is_the_glm_profile`, `the_tick_engine_hands_glm_4096_row_chunks`,
  `the_tick_engine_keeps_glms_joint_round` and `the_generated_checks_drive_the_tick_engine`.
* `cargo clippy -p sparkflash-runtime -p sparkflash-model-glm5-next --all-targets --
  -D warnings` and `cargo fmt --all -- --check` are clean.

**Unverified without a cluster.**

* GLM's device conformance through the engine: `scripts/onboard-gate.sh GLM` (the seven
  checks), a served gate at one and four lanes, and a numerics check on a freshly booted
  server as `NUMERICS.md` requires.
* The live trace on GLM: `WallClock` readings, the per-tick flush and the rank-0 rule.
* Speed: no step time, tokens/s or time to first token. The regression clause's A/B has not
  run, so nothing here says the tick engine is as fast as the fused one.
* The group: no TP2/TP4 boot ran, and GLM's group is not on the engine (below).

**What a cluster window must measure.**

1. `scripts/onboard-gate.sh GLM`: the seven generated checks through the engine,
   PASS/FAIL/SKIP/NEEDS-FIXTURE.
2. A served gate at one and four lanes, then a numerics check per `NUMERICS.md`; the same
   prompts through the pre-move `Fusion::local` build and this one, comparing token ids for
   greedy and sampled requests.
3. Alternated launches of the old and new builds at 1, 4, 8 and 16 streams and at 2K and 16K
   prompts, several reps each; compare token streams and each request's `cached_tokens`, and
   diff a live `SPARKFLASH_TICK_TRACE` from each launch as decisions. A pair that differs only
   in an admission tick is reported, not excused.
4. Chunk sizes from the trace: every prompt chunk is 4,096 rows except its tail, and no decode
   batch exceeds 8 members.
5. Speed, and this is the one thing the A/B must measure first: TP=1 at one request and at
   four lanes, the old fused path against the tick engine - mean step time and tokens/s with
   the token stream beside them. Under the regression clause, if the tick engine is slower,
   GLM returns to `Fusion::local` and this section records the measurement and why.

**The group did not move.** `enable_group_fusion` and `mirror_group` still build
`Fusion::group` over `Runtime<GlmWalk>`, so GLM at TP>1 keeps the old schedule. The
rows-carrying frame alone is not enough; the tick group needs both halves, and neither is
written here:

* **Frames.** a `LaneRows { lane, rows }` that says "run exactly these rows of the prompt",
  so a worker calls `Runtime::run_part_rows` instead of `next_part`; and a `LaneStepMode` that
  carries the engine's mode (`Plain`, `Round { joint: false }` or `Round { joint: true }`) with
  the `(lane, token)` members, so a worker runs `step_mode_slots` with the same shape. The
  existing `LaneRequest` must also carry the start `start_prompt_slot` chose, or the worker
  must refuse a start that differs, because every row count is a position from that start.
* **Rank 0.** `TickServe`/`TickEngine` needs a group form that broadcasts those frames before
  `start_prompt_slot`, each `run_part_rows_hooked` and each `step_mode_slots`, and before
  `give_back`/`release`. Its `TickConfig` is `for_walk` with `exchange = true`, and only the
  prompt chunks and steps are broadcast: the lane choice, the hold and the tick-ahead plan
  stay rank 0's, as they are for `Fusion::group`.
* **The worker.** a mirror over `Runtime<W>` replaying those frames in order; the existing
  `FusedArm` mirror cannot express the tick's rows because its `LaneRequest` runs a prompt and
  its `LanePart` runs the next `ChunkPolicy` chunk, neither of which is a row count rank 0
  decided.

The frame can be added later without touching another arm: only GLM's group would open the
tick form, and the existing `Fusion::group` path for Qwen, Inkling, Dory and dense's TP>1
stays as it is. It was not added here because it is a new wire protocol and a second mirror,
and its TP>1 bits cannot be checked on this host; moving it without the cluster gate would
make an unverified bit claim for a released arm, which phase 6 forbids.

**Later:** the frames and the rank-0 form are now written, and GLM's group is on them. See
[The tick group: rows and step shape on the wire](#the-tick-group-rows-and-step-shape-on-the-wire)
below; that section is the current state, and this one records why it was not written at this
commit.

### The tick group: rows and step shape on the wire

Commits `65f256ca` (the frames and the request's start), `2fa4b465` (the rank-0 form, the
worker mirror and the CPU gate) and `f2497776` (GLM's group on it).

`TickGroup` and `mirror_tick_lanes` (`crates/sparkflash-runtime/src/group.rs`),
`TickEngine::with_group` and `TickServe::start_group` (`engine.rs`), the request's start and
the two new frames (`crates/sparkflash-comm/src/control.rs`), and GLM's
`enable_group_fusion`/`mirror_group` (`crates/sparkflash-model-glm5-next/src/serve.rs`) put
GLM's TP>1 path on the tick engine. GLM is the first arm whose group is the tick engine; Qwen,
Inkling, Dory and dense's TP>1 still serve through `Fusion::group`/`GroupLanes`, so neither is
deleted.

**The wire, message by message.** A tick group keeps the fused group's lane frames and adds the
two a tick decides. Rank 0 sends these in the order its tick makes the decisions; every worker
replays each one before it reads the next. Only the prompt starts, the chunks, the steps, the
give-backs and the releases travel: the lane choice, the hold, the prompt order, the chunk
boundary, the step shape and the tick-ahead plan stay rank 0's.

| Frame | Fields | When rank 0 sends it | What a worker does |
| :-- | :-- | :-- | :-- |
| `LaneRelease` | lane | before rank 0's own `release_slot`: an idle lane evicted for room, or a lane whose prompt failed before its request frame | `release_slot` |
| `LaneRequest` | lane, prompt ids, image spans, sampler, `start: Some(start)` | after rank 0's `start_prompt_slot` chose the start, before any row of the prompt | `set_sampling`, then `start_prompt_slot`; **refuses a start that differs** |
| `LaneRows` | lane, rows | before each `run_part_rows_hooked` | `run_part_rows(slot, rows)` — exactly these rows, not the arm's `ChunkPolicy` chunk |
| `LaneStepMode` | members `(lane, token)` in row order, mode `Plain` or `Round { joint }` | before each `step_mode_slots` | `step_mode_slots` with that mode, then the held-answer drain |
| `LaneGiveBack` | lane, count | after rank 0's own give-back succeeded | `give_back_slot` |
| `LaneRelease` | lane | before rank 0's own `release_slot` | `release_slot` |
| `Error` | code, message | when rank 0's own half of a frame failed after the frame went out | the mirror returns, and the process ends |

`LanePart` and `LaneStep` remain the fused group's frames; the tick mirror refuses them, and the
fused mirror refuses `LaneRows`/`LaneStepMode`, so a half-updated rank pair is a named error rather
than a wrong row. A prompt's images travel as spans (rank 0 encodes), and the sampler travels whole
because every rank's picks decide what a verify keeps; the request's start is
`Option<u32>` so a fused group's frame (which cannot know its arm's start without changing the
fused arm's prompt path) leaves it absent, and a frame from an older rank 0 reads as absent.

**What rank 0 decides, and what a worker checks.** Rank 0 decides the lane (`pick_lane`), the
prompt order and hold (`prefill_order`, `hold_prefill`), the chunk boundary (`bulk_from`,
`bulk_rows`, `tick_rows`), the step shape (`decode_plan`/`plan_for`), the tick-ahead plan
(`Ahead`) and when to give back or release. A worker checks only what makes its replay the same
execution: the lane is in its pool, the request's start equals the one *its* `start_prompt_slot`
reaches from the history rank 0's frames built, the rows fit the prompt left, and the step's members
are the frame's. It re-derives no decision: the row count comes from `LaneRows` and the shape from
`LaneStepMode`.

**The CPU gate.** `crates/sparkflash-runtime/tests/tick_engine_group.rs` drives a loopback
`Tp4ControlHub` and one worker thread, over a toy `Walk` shaped like GLM's (two-row prompt chunks,
a verify width of four, a drafter with one draft a member). The first test admits two prompts
(three rows and two rows), runs the prompt chunks, one shared decode step over both lanes and a
give-back, and asserts:

* the exact frame order the worker received:
  `request 0 [10, 11, 12] start=0`, `request 1 [20, 21] start=0`, `rows 1 2`, `rows 0 2`,
  `rows 0 1`, `step [(0, 13), (1, 22)] Round { joint: true }`, `give back 0 1`;
* the rows the worker ran are rank 0's chunk decisions, in rank 0's order, `[2, 2, 1]`;
* the worker's passes are rank 0's, in order: `Bulk(2)`, `Bulk(2)`, `Bulk(1)`, then
  `Verify([2, 2])` - one shared verify over both members - not a re-derived shape;
* `with_group` turned `exchange` on and the engine reports itself as a group.

The second test sends a lane a start the worker cannot reach (rank 0 claims 4, the history reaches
3) and asserts the mirror refuses with `"rank 0 starts the prompt at 4, this rank starts it at 3:
the ranks are out of step"` - both positions named, and nothing run.

**Measured on this host (CPU; it has no GPU).**

* `cargo test -p sparkflash-runtime --test tick_engine_group`: 2 passed.
* `cargo test -p sparkflash-runtime`: 89 passed, 7 ignored (77 library tests, 2 sealed, 5
  drafter, 2 group, 2 engine-trace, 1 tick-trace) and 2 doc-tests (1 ignored).
* `cargo test -p sparkflash-model-glm5-next`: 93 passed, 8 ignored (71 library tests, 11
  `device_forward`, 4 `engine_builder`, 7 `serve_tiny`). The group path itself compiles for
  `GlmWalk` (`TickServe::start_group`'s `Send + Sync`/`Seq: Send` bounds), but no GLM rank runs
  on this host.
* `cargo test --workspace`: 4,305 passed, 81 ignored, 0 failed (220 test binaries), exit 0.
* `cargo clippy -p sparkflash-runtime -p sparkflash-model-glm5-next -p sparkflash-model-dense
  --all-targets -- -D warnings` and `cargo fmt --all -- --check`: clean at this tip.

**What the tokens can and cannot be claimed to be.** GLM's token path is unchanged in shape: the
same walk, the same 4,096-row prompt chunks (a group's `exchange = true` marks the bulk path but
`bulk_rows` returns 4,096 then the tail either way), the same 8-row verify width as the member cap
(`Round { joint: true }` for several lanes, `Round { joint: false }` for one), the same
`GlmDrafter` verify and `verdict` settlement. What the group adds is that a worker runs rank 0's
rows and shape instead of its own `ChunkPolicy` chunk, and the start is checked. On paper that is
the same arithmetic on the same rows, so `drafted = plain` and prompt = chunked = token should
hold; **it is unverified**: no GPU boot ran, and the claim that a TP=4 group produces the same
token ids as the pre-move `Fusion::group` build is the first thing the cluster window must
measure. The one GLM path the group does not reproduce is the fused engine's *schedule* (which
lanes step together and when): the tokens are the arm's, the tick order is the engine's.

**What a cluster window must run, and what counts as a pass.**

1. A TP=4 GLM boot on `enable_group_fusion`, and the same prompts through a pre-move build on
   `Fusion::group`. Pass: identical greedy token ids for a fixed prompt set at 2K and 16K, at
   one, four and eight concurrent requests, and `cached_tokens` agreeing per request. A
   difference is a regression, not a scheduling variance; the group changes rows, not sampling.
2. A TP=1 and a TP=4 boot of the same build, same prompts. Pass: the token ids agree (the arm's
   TP check) and the per-tick `SPARKFLASH_TICK_TRACE` from rank 0 shows the chunk rows and the
   plans; a worker writes nothing.
3. Chunk boundaries from the trace: every prompt chunk is 4,096 rows except its tail, and each
   `LaneRows` frame's `rows` equals the chunk the trace recorded. Every chunk a prompt frame names
   is at most the 4,096-row profile chunk; a `LaneRows` frame never carries a step.
4. Verify shapes from the trace and the arm's counters: one `Round { joint: true }` for several
   lanes whose whole verify (anchors plus shared drafts) is at most `MAX_ROWS` = 8 rows,
   `Round { joint: false }` for one lane, and no verify at all while no lane drafts. With
   `JOINT_ROUNDS=off` the engine plans `Plain` (the named deviation below), so that run is a
   separate A/B and not the default.
5. Speed: tokens/s, mean step time and time to first token at TP=4, old build against this one,
   alternated launches, several reps. Under the regression clause a slower GLM group returns to
   `Fusion::group` and this section records the number and why.
6. A numerics check on a freshly booted server per `NUMERICS.md`, and the same prompt twice (cold
   and continued) to show the group's lane reuse keeps the continued turn bit-equal.

**Named deviations, so none is implied.**

* **The request frame follows rank 0's start, not precedes it.** The frame carries the start
  `start_prompt_slot` chose, so rank 0 must run its own start first; the frame is still before
  every row, and the worker's start (the review's "before `start_prompt_slot`") runs after it. A
  text arm's start is a local rewind/truncate with no collective, which is why the order is safe
  here; a vision arm would need its encode after the frame, and that path is not written.
* **The held-answer drain is local, not a frame.** The answers the arm already holds run no device
  work; every rank runs the same `Runtime::step_slots` calls and keeps the same history. If the
  states ever diverged, the next request's start check (or a differing verify shape) ends the
  group rather than running a different row.
* **The resume start still stands** (the earlier section's deviation 2): the frame carries
  `resume`'s checkpoint start, never an arbitrary shared prefix; DeepSeek's arbitrary restart is
  still not claimed.
* **Adoption, lane growth, paired chunks and preemption still stand** (the earlier section's
  deviation 5): the engine has none of them, so the group has none. A paired chunk never travels
  (`Pass::Paired` is unused here); a preempted prompt is still served to its end.
* **`JOINT_ROUNDS=off` plans `Plain` on a drafting arm** (the earlier GLM section's non-default
  path): the engine's own rule, unchanged by the group.
* **The group runs text arms only.** `TextWire` refuses images before the request frame, as it does
  for a fused group; the tick group has no image path of its own.

### The device window's two gaps: a catalogue hole and a fixture checklist

The four-DGX-Spark window at `a6091608` (sm_121a, the branch's ignored tests) passed every arm
except two arms' generated conformance checks, which panicked with `NEEDS-FIXTURE` instead of
serving. Both were real gaps, not flakes:

* **Inkling-2-Small could not resolve six of its own kernels.** The image exports
  `inkling_block_residual_chain_q1_n0`, `_q1_n1`, `_q1_n3`, `_q3_n0`, `_q3_n1` and `_q3_n3`
  (`crates/sparkflash-kernels/src/images.rs`), and the chunk walk loads all six by name through
  the typed `Bound::carried` lookup (`crates/sparkflash-model-inkling2/src/device.rs`), but the
  `Part::BlockAttentionResidual, Image::Inkling` catalogue entry named only the one-block and
  two-launch bodies. The boot refused the first with "the catalogue carries no
  inkling_block_residual_chain_q1_n0 in inkling_sm121 accepting bf16"; `run_generated` maps the
  boot's `fixture` code to `NEEDS-FIXTURE`, so the check panicked rather than ran. The catalogue
  had a hole in a released arm: the kernel, the image export, the loader and the call site were
  all present, and only the entry the typed lookup reads was missing.
  The six are one template (`kernels/inkling_conv_sm121.cu`, `INKLING_CHAIN_ENTRY(NQ, NEXT)`)
  differing only in the compile-time query and next-call counts. The template takes f32
  summaries, partial, queries, out, carry and next queries (`const float*` throughout), so the
  entry is `ComputeClass::Elementwise` at BF16 - the precision the walk loads and the precision
  of the sibling `inkling_block_residual_stats` / `_mix` bodies - in the bulk-prefill pass, over
  the nine-or-more rows the walk chains (`rows > RESIDUAL_SPLIT_PAIRS`, 8). The six are
  catalogued as instantiations of one `chain` helper inside the part's existing entry, the way
  the GDN scan entry names its `q4p_gdn_scan_64_64` / `_128_256` instantiations. The new CPU
  test `every_chained_residual_the_arm_launches_is_in_the_catalogue` runs each
  `RESIDUAL_CHAIN_PAIRS` entry through the same `Bound::carried` lookup the boot runs; it
  prevents another launched symbol from reaching a device uncatalogued. That is the rule dense
  already stated in `every_kernel_the_arm_loads_is_in_the_catalogue`, applied to Inkling.

* **Dense's conformance checks could not run unattended.** `DenseWalk::fixture` required
  `SPARKFLASH_CONFORMANCE_CHECKPOINT`. `scripts/onboard-gate.sh dense` already wrote the
  synthetic checkpoint when the switch was unset, but the window ran the tests directly, so the
  checks panicked `NEEDS-FIXTURE` and the fixture stayed a checklist item - the thing phase 8
  exists to remove. The fixture now writes the checkpoint itself when the switch is unset, into
  the same `target/conformance/dense-synthetic` directory the gate uses, prints the path, and
  stages the write in a sibling directory published by rename so parallel checks never open a
  half-written checkpoint. The fixture errors name the exact command that writes one:
  `python3 scripts/dense-synthetic-checkpoint.py reference target/conformance/dense-synthetic
  --layers 2 --hidden 1024 --intermediate 2816 --context 512`. Writing needs no device:
  `the_dense_fixture_writes_itself_on_the_cpu` writes it and parses the config back on this
  host, and only the checks that boot the walk need the device. `onboard-gate.sh` now rewrites
  a directory missing either file instead of trusting any existing directory; the other arms'
  behaviour is unchanged.

**The device window must rerun these four commands** on the branch tip, with the cubins and
`SPARKFLASH_KERNEL_DIR` the original run used:

```sh
cargo test -p sparkflash-model-dense --lib -- --include-ignored --test-threads=1
cargo test -p sparkflash-model-dense --lib conformance:: -- --ignored --nocapture
cargo test -p sparkflash-model-inkling2 --lib -- --include-ignored --test-threads=1
cargo test -p sparkflash-model-inkling2 --lib conformance:: -- --ignored --nocapture
```

A pass is: dense's generated checks report five PASS and two SKIP (`drafted = plain`, because
dense declares no drafter, and `TP 1 = TP 2 = TP 4`, because the fixture is one rank), with no
`NEEDS-FIXTURE` line; Inkling's report six PASS and one SKIP (TP, one rank), including
`whole_prompt_equals_chunked`, the check that failed first. The two new CPU tests also run in
the `--include-ignored` forms and need no GPU. `scripts/onboard-gate.sh dense` and
`scripts/onboard-gate.sh inkling` are the script forms of the same gate.

### What a container reports in three dimensions: convolutions and expert blocks

Two four-DGX-Spark windows measured this class of refusal at `57cd3c96`. The first passed dense's
generated checks and failed all seven of Inkling's with one error:

```text
conformance: drafted = plain: NEEDS-FIXTURE: model.llm.layers.0.attn_sconv.weight: upload::<bf16> refuses `model.llm.layers.0.attn_sconv.weight`: the tensor is stored as bf16 (BF16 [32, 1, 4]), and the Mat is bf16
```

The tensor is Inkling's per-layer short convolution. The fixture writes it (and `mlp_sconv`, and
the GQA form's `k_sconv`/`v_sconv`) as `[hidden, 1, kernel]` = `[32, 1, 4]` BF16
(`crates/sparkflash-model-inkling2/src/fixture.rs`), and the plan names the same three
dimensions (`src/tensor.rs`), because that is how a container reports a 1-D convolution:
`[out, in/groups, kernel]`. The next window cleared that and stopped on the stacked experts, the
same fact one axis up:

* BF16 `model.llm.layers.0.mlp.experts.w13_weight`, stored `[experts, out, in]` =
  `[2, 32, 32]` in the fixture (the released widths would store `[384, 4608, 4608]`), Mat
  view `[64, 32]`;
* NVFP4 `w13_weight`, stored `U8 [384, 4608, 2304]` (its last axis is bytes), logical
  `Mat<NvFp4>` `[1769472, 4608]`, the released checkpoint.

On the boot path `device::type_slots` flattens the stored leading dimensions with
`flat_matrix` to the matrix the arithmetic sees and still passes the container's spec as
`Tensor::stored`. `upload` rebuilt the logical spec from that 2-D view and required
`TensorSpec::is_stored_as`, which demanded literal dtype and shape equality, so the same tensor
in two honest notations was refused before any weight reached the device. The plan's own bytes
check had already accepted both spellings.

The rule in `TensorSpec::is_stored_as` (`crates/sparkflash-weights/src/spec.rs`) is a contiguous
collapse of the stored leading dimensions into the logical rows: the dtypes agree, the element
counts agree, the stored last dimension is exactly the logical columns, and the product of the
stored leading dimensions is exactly the logical rows. `[2, 32, 32]` is `[64, 32]`,
`[32, 1, 4]` is `[32, 4]` (a unit dimension is the collapse's degenerate case, not a special
one) and `[4, 2, 32]` is `[8, 32]`: the same values in the same row-major order. The FP4 arm
applies the same collapse to the stored leading dimensions and keeps the stored last dimension at
exactly half the logical columns: `U8 [384, 4608, 2304]` is `F4E2M1 [1769472, 4608]`. A wrong
column or a wrong leading product is still refused, as is a reshape that changes the last
dimension (the same bytes read in another order) or a different element count; and the dtype must
still agree, so a `U8` tensor cannot stand in for BF16 and only the FP4 arm relates U8 to another
dtype. The rule lives here, not in `upload`, because it is a statement about a container's
spelling of a tensor versus the plan's, which is what the other caller also asks -
`parts_backend::open` checks a planned spec against the weight source before any kernel is
resolved - and both call sites now mean the same thing.

Tests. `a_stacked_expert_block_is_the_matrix_the_arithmetic_sees` in `spec.rs` accepts the
fixture's `[2, 32, 32]` as `[64, 32]` (and `[32, 1, 4]` as `[32, 4]`, `[4, 2, 32]` as
`[8, 32]`, and a surviving leading unit dimension), and refuses a changed last dimension
(`[128, 16]` as `[2, 32, 32]`), a wrong column, a wrong leading product, a rank-1 shape of
another column, and `U8`/`F32` in place of BF16; the FP4 test adds `U8 [384, 4608, 2304]` as
`F4E2M1 [1769472, 4608]` and refuses a wrong half and a wrong leading product. In
`crates/sparkflash-ops/src/formats.rs`, `the_inkling_fixture_short_convolution_uploads_as_bf16`,
`the_inkling_fixture_stacked_experts_upload_as_bf16` and
`the_released_nvfp4_stacked_experts_upload_as_nvfp4` run those exact specs and the boot's views
and strides through `upload::<Bf16>` and `upload::<NvFp4>`. `upload` never dereferences the
address, so all of them run without a device; the boot's own `type_slots` needs the
CUDA-allocated `Slot` pointers, which is why the upload tests enter at `upload` with an unbacked
`Deviceptr`.

**The device window must rerun these two commands** on the branch tip, with the same cubins and
`SPARKFLASH_KERNEL_DIR` as the runs above, expecting Inkling's six PASS and one SKIP (TP, one
rank) and no `NEEDS-FIXTURE` line:

```sh
cargo test -p sparkflash-model-inkling2 --lib -- --include-ignored --test-threads=1
cargo test -p sparkflash-model-inkling2 --lib conformance:: -- --ignored --nocapture
```

The next Inkling window must launch through `/home/allenh1/inkling-bin/rsp-run.sh`, now on all
four hosts: the group needs the binary copied to ranks 1-3 and the NCCL environment the older
script did not set, and the wrapper does both. `/home/allenh1/rsp-0611` has no `.git`, so it
can be the run directory but never the source of a commit or a `git status` check.

### Qwen on the tick engine: the prefill plan's chunks and the batch's verify width

Commits `659b2dad` (the single process) and `c4d0010d` (the group). Qwen has no `serve.rs`; its
serving wiring is `crates/sparkflash-model-qwen4-exp/src/device.rs`, and the walk, the plan and
the widths are in `device/walk.rs` and `device/prefill.rs`.

`Qwen4ExpDeviceBackend::enable_fusion` now opens a `TickServe` over `QwenWalk` instead of
`Fusion::local` over `Runtime<QwenWalk>`, and `enable_group_fusion`/`mirror_group` start
rank 0 of the tick group and replay its frames, exactly as GLM's do. The walk, the drafter, the
kernels and the layout are untouched: the engine chooses *when* a chunk or a step runs, and the
runtime executes the same rows with the same shape.

**Qwen's chunk was never a Qwen constant, and it does not move.** It is `PrefillPlan`'s chunk -
clamped to 16..4,096 and rounded down to a multiple of `DENSE_ROWS` (512) above 512 - lifted by
`pass_rows` to the smallest documented `ChunkPolicy` size at or above it that it divides, and
handed to the runtime's `ChunkPolicy` over `Limits::new(pass_rows(chunk), context)` with
`members = capacity.clamp(1, MAX_BATCH)`. The engine reads the same two functions
(`TickConfig::for_runtime`, which is `ChunkPolicy::for_limits` over `Walk::limits`), so the part width
is the runtime's by construction in every plan, including the odd ones (a 512-row plan chunk
gives 1,024 rows a part and two 512-row forwards, a 2,560-row chunk gives 4,096 and 2,560 + 1,536,
both before and after):

| | Before (`Fusion` over `Runtime`) | After (`TickServe` over `TickEngine`) |
| :-- | :-- | :-- |
| prompt part, tensor-core plan (the default) | `ChunkPolicy` rows = **4,096** | `exact_tick_rows` = `busy_chunk` = `chunk_min` = `bulk_cap` = **4,096** |
| prompt part, token-kernel plan (`SPARKFLASH_QWEN4_PREFILL_MMA=0`, or a build without the prefill cubin) | **256** (the plan's chunk is 128) | **256** |
| a part's forwards | `PrefillPlan::forwards`: 4,096; or two 128-row forwards a 256-row part | unchanged - the arm still splits its own rows |
| prompt rows | any prompt, one part a pass | the same rows: 5,000 is **4,096 + 904**; 600 at the token plan is **256/256/88** |
| verify width | `Limits::members` = `capacity.clamp(1, MAX_BATCH)` = **4** | `TickConfig::members` = the same **4** |
| a member's verify rows | 1 + depth: `verify_rows` = `MTP_VERIFY_ROWS` (3), or the DFlash 2 head's `block_size` when that is larger, so **3 rows and 2 drafts** on the MTP stage | `k = drafter.budget(1)` = **3 rows, 2 drafts** |
| joint rounds | the runtime's `JOINT_ROUNDS` | `for_runtime` reads the same switch into `shared_rounds` |
| decoder-layer skip | none; only DeepSeek has those layers | `decoder_skip = false` |
| row exchange | none at TP=1 | `exchange = false` at TP=1, `true` in a group |
| drafter | `QwenDrafter`: the DFlash 2 head or the checkpoint's MTP stage | unchanged |
| a group's lane count, when the operator gives none | `lane_limit()` = `limits().members` = 4 | the same 4 |

Neither path that could move the boundary does: `bulk_rows` returns the chunk and then its tail
with `bulk_cap = chunk`, and the exact tick is `exact_tick_rows = chunk`, so a group's bulk
chunks and a single process's exact ones name the same rows. A CPU test pins that on a 5,000-row
prompt: `4,096, 904` on both.

**What the group carries for Qwen.** The tick group's wire (`crates/sparkflash-runtime/src/group.rs`
`TickGroup`, the section above) with Qwen's widths: `LaneRequest` with the start
`start_prompt_slot` chose, `LaneRows` for every prompt chunk rank 0's tick decided,
`LaneStepMode` with the member order and `Plain`/`Round { joint }`, `LaneGiveBack`,
`LaneRelease` and `Error`. Rank 0 keeps every decision - `pick_lane`, `prefill_order`,
`hold_prefill`, the chunk boundary, `decode_plan` and the tick-ahead plan - and a worker checks
only that the lane is in its pool, that the request's start is the one its own
`start_prompt_slot` reaches, that the rows fit the prompt left and that the step's members are the
frame's. A worker runs `Runtime::run_part_rows` with the frame's row count and
`step_mode_slots` with the frame's shape, so lane `n` is the same sequence on every rank and the
collectives inside a verify line up. `TickServe::start_group` turns the row exchange on
(`TickEngine::with_group`), which is what puts a prompt on the bulk path; its rows are the same
split, as above.

**Vision: nothing was left on `Fusion`, because there is no image path on either engine.** `QwenWalk`
declares `type Vision = NoVision`, so the runtime's per-prompt `VisionImage` spans reach the walk
through `Segment::with_images` and the walk has no encode hook to call; the unfused
`DeviceSequence` session does not implement `prefill_images` at all, so it refuses an image. In a
group, `TextWire::admit` refuses a prompt with images before the request frame, as the fused group
already did. So no Qwen configuration was stranded on `Fusion` for images: the tick group carries
exactly what the fused group carried, and the TP=1 path is unchanged. This is the arm's present
state (the vision tower is planned and not loaded while `capabilities()` returns
`Capabilities::ALL`), and this move neither creates nor closes it; loading the tower is a separate
change that would need a real `Vision` impl, not a wire.

**Measured on this host (CPU; it has no GPU).**

* `cargo test -p sparkflash-model-qwen4-exp --lib`: **108 passed, 0 failed, 7 ignored** (the seven
  are the generated checks, which boot the tiny checkpoint on a device). Six are new:
  `the_generated_checks_drive_the_tick_engine` (the fixture now says `tick_engine() == true`, so
  `whole = chunked = token`, `alone = four`, `continued = cold`, `rewound = cold`,
  `drafted = plain` and `no unrecorded widening` drive the engine on the device host),
  `the_tick_config_is_the_qwen_profile` (4,096 pass rows, 4 members, `k` = 3, a drafter present
  and `shared_rounds` following the runtime's own switch, no decoder skip, no exchange at TP=1),
  `the_tick_engine_hands_qwen_the_prefill_plans_chunks` (5,000 rows run 4,096 + 904 on the exact
  path and again with `exchange = true`, the group's bulk path, and the plan's forwards of each
  part are one 4,096-row forward and one 904-row one), `the_token_plans_pass_is_still_256_rows`
  (600 rows run 256/256/88, each part two 128-row forwards), `the_tick_engine_keeps_qwens_joint_round`
  (three lanes, one `Verify` pass of three segments and the trace's `plan=shared`) and
  `a_speculative_round_keeps_its_accepted_rows_only` (one lane, `Round { joint: false }` over the
  anchor and two drafts, the anchor and the accepted draft reach the request, the rejected token
  never does, one verify of three rows, `plan=spec`).
* `cargo test -p sparkflash-runtime`: **91 passed, 0 failed, 8 ignored** (77 library tests, then
  `sealed` 2, `tick_engine_drafter` 5, `tick_engine_group` 2, `tick_engine_trace` 2 and
  `tick_trace` 1, plus 2 doc-tests; the 8 ignored are the 7 conformance stubs and 1 doc-test).
  The group's two tests are
  **reused unchanged**, because the wire did not change: the frame order
  (`request 0 [10, 11, 12] start=0`, `rows 1 2` ... `step [...] Round { joint: true }`,
  `give back 0 1`) and the start agreement
  (`rank 0 starts the prompt at 4, this rank starts it at 3: the ranks are out of step`, nothing
  run). That second test is the start agreement this arm's group relies on; it is the runtime's
  property, not an arm's, which is why it did not need extending.
* `cargo test -p sparkflash-runtime -p sparkflash-model-qwen4-exp`: 0 failed.
* `cargo clippy -p sparkflash-runtime -p sparkflash-model-qwen4-exp --all-targets -- -D warnings`
  and `cargo fmt --all -- --check`: clean at this tip.
* `cargo test --workspace`: **4,313 passed, 81 ignored, 0 failed** over 198 test binaries,
  exit 0, at this tip.

**Unverified without a cluster.**

* Qwen's device walk through the engine: the tiny checkpoint's checks (seven, of which TP is a
  SKIP at one rank) and the served gate.
* The tiny checkpoint's own Qwen drafter through the engine: the CPU tests drive a shape walk, not
  `DeviceSequence`, so the MTP/DFlash round's bits are the arm's existing claim, not a new
  measurement.
* The live trace on Qwen: `WallClock` readings, the per-tick flush and the rank-0 rule.
* TP>1: no Qwen rank pair or group ran here. The claim that a TP=4 group on the tick group gives
  the same token ids as the pre-move `Fusion::group` build is the first thing the window must
  measure.
* Speed: no tokens/s, no step time, no time to first token. Nothing here says the tick engine is
  as fast as the fused one, and the regression clause's A/B has not run.

**What a cluster window must run, and what counts as a pass.**

1. The generated checks through the engine, then the first reference:

   ```sh
   cargo test -p sparkflash-model-qwen4-exp --lib -- --include-ignored --test-threads=1
   cargo test -p sparkflash-model-qwen4-exp --lib conformance:: -- --ignored --nocapture
   scripts/onboard-gate.sh Qwen --url http://127.0.0.1:8000 --port 8000 --world 1 --apply
   scripts/onboard-gate.sh Qwen --url http://127.0.0.1:8000 --port 8000 --world 4 --apply
   ```

   Pass: six PASS and one SKIP - `TP 1 = TP 2 = TP 4`, whose SKIP names the world its fixture
   cannot boot, because the Qwen fixture is one rank - with no `NEEDS-FIXTURE` line. Then a served
   gate at one and four lanes and the first numerics reference at the world that is deployed
   (`--world 4` writes the group's), as `NUMERICS.md` requires.

2. Tokens and `cached_tokens` against the pre-move build at the deployed world. Build the frozen
   tip before this move (`git worktree add /tmp/qwen-pre 659b2dad^`) and run the same fixed
   prompts through both, greedy and sampled, at **TP=1 and TP=4**, at **2K and 16K** prompt
   tokens, at **1, 4 and 8** concurrent requests, several reps each, alternating the launch order.
   Pass: identical token ids and, per request, the same reported `cached_tokens`; a difference is
   a regression, not a scheduling variance, because the move changes when a row runs, not which
   row.

3. The trace's chunk boundaries. Only the new build writes a tick trace - the fused engine has no
   tick to trace - so this is the new build's evidence, not a diff against the old one:

   ```sh
   SPARKFLASH_TICK_TRACE=/tmp/qwen-tp1.trace sparkflash serve \
       --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 --engine device --concurrency 4
   # the group needs the tp4-serve feature and one process a rank; rank 0's trace only
   cargo build --release -p sparkflash-cli --features tp4-serve
   SPARKFLASH_TICK_TRACE=/tmp/qwen-tp4.trace sparkflash serve \
       --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
       --tp4-rank 0 --tp4-world-size 4 --tp4-peers A,B,C,D
   ```

   Pass: every `prefill` line's rows are the prefill plan's chunk (4,096, or 256 at the
   token-kernel plan) except a prompt's tail, each `LaneRows` frame carries exactly the chunk the
   trace recorded, and a `LaneRows` frame never carries a step. `scripts/tick-trace-ab.py` can
   compare two tick builds' decisions (`--checkpoint ... --model ... --streams 1,4,8
   --prompt-tokens 2048,16384`); it cannot compare a fused build, which writes no trace.

4. The verify shapes from the trace and the arm's counters. Pass: no decode step holds more than
   `MAX_BATCH` = 4 members; several lanes take one `Round { joint: true }` whose rows are every
   member's anchor and drafts, at most `budget(4)` = 12 rows; one lane takes
   `Round { joint: false }`; a batch with `JOINT_ROUNDS=off` is `plan=plain` (a separate run,
   the named deviation below); and a member's verify is at most `verify_rows` = 3 rows (or the
   DFlash 2 head's `block_size`).

5. Speed, at TP=1 with one and four lanes and at TP=4: tokens/s, mean step time and time to first
   token, the old fused path against the tick engine, alternated launches, several reps. Under the
   regression clause, a slower Qwen returns to `Fusion::local` (TP=1) and `Fusion::group` (TP>1):
   that is a revert of `659b2dad` and `c4d0010d`, and this section then records the number, the
   workload and why.

**Named deviations, so none is implied.**

* **The tick order is the engine's.** Which lanes step together and when is the scheduler's; the
  rows and the shapes are the arm's. A greedy token does not depend on when its chunk ran.
* **`JOINT_ROUNDS=off` plans `Plain`** (the earlier GLM deviation, unchanged by this move): one
  plain token row a member in groups no wider than 4, where the fused runtime ran one round a
  member. Qwen also turns its own speculation off on that setting (`QwenWalk::open` masks
  `spec_enabled` with the drafter's `joint`, so a held chain is not run and thrown away), so the
  rows are plain token rows and the tokens are still the `drafted = plain` property's. The switch
  is off only when an operator sets it; the A/B runs the default.
* **The engine's pacing values are DeepSeek's**: `share = DecodeShare::new(0.5, 200_000.0)`,
  `model = (23.0, 4.0, 23.0, 4.0, 2.3)` and `prefix_reuse_min_rows`, from `TickConfig::for_walk`.
  They decide when a decoding lane takes a row during a long prompt, not what a row computes; the
  round's own cost stays the arm's `ExpertUnionWall` through `Drafter::wall`.
* **The resume start is `resume`'s checkpoint**, never an arbitrary shared prefix (the earlier
  deviation, unchanged): the engine picks the lane by its shared prefix and records the start
  `start_prompt_slot` chose, and the group's frame carries that start and is checked.
* **Adoption, lane growth, paired prompt chunks and preemption are still absent** (the earlier
  deviation): `Pass::Paired` is unused, and a preempted prompt is served to its end.
* **The group runs text only** (above): `TextWire` refuses images before the request frame.

### The Inkling reference no longer matches, and how far the bisect reached

A window serving inkling-2-Small at world 4 on the branch head ran `scripts/numerics.py check`
and it **failed**: the server served `397907e6…` where the registry names `387647bc…`, with the
first difference at the first position of the first prompt (`logprobs 'dee893bf' != '09f288bf'`).

**The answer is the prefill walk, not the build.** The decisive pair was served on one binary (the
tip at `84f07153`, sha `1558ab2a…`, four ranks one hash), one checkpoint and one converted set:

| launch | verdict |
| :-- | :-- |
| `LANES=4`, `PREFILL=tensor`, `CONTEXT=1048576` (the capture's configuration) | **PASS** `387647bc…` |
| the same binary, `PREFILL=chunk` | **FAIL** `397907e6…` - character for character the earlier failure |

So `397907e6` is the chunk prefill walk of the same code: under `tensor` a chunk's dense
projections run on the wide tile (BF16 head and remainder, f32 accumulate), under `chunk` each
prompt row takes the row kernels' arithmetic and is the decode step's bits. The earlier windows
had compared a chunk-walk, one-slot run - the old `inkling-bin` launcher, which has no `LANES`
and defaults to `PREFILL=chunk` - against a four-lane tensor-walk reference. The lane count turned
out not to matter: `LANES=1` on the tensor walk also gives `387647bc…`. The context does not
matter either. What made the investigation expensive was that neither the capture's card nor the
boot ledger named the walk.

**A real declaration gap, of the same class as GLM's inert decode switch:** the boot ledger prints
"chunked prefill native (8192-row chunks from ChunkPolicy); tensor-core prompt native …" in both
walks - it states the capability, not the selection, and `PrefillWalk::from_env` logs nothing. A
run's own `/proc/<pid>/environ` was the only evidence of which arithmetic ran. The next small
change declares the walk in the ledger and the card.

The conversion `540ae6d0` remains unmeasured as committed: it does not boot without the six
catalogue entries, the unit-dimension and leading-dimension upload rules and the retired-entry
field rename cherry-picked, and with them it serves the chunk walk's fingerprint.

The lesson is worth a mechanism: **a capture's server card has to name the build it came from**.
Without that field, a frozen reference can only say that bits moved, never which commit moved
them, and every bisect pays for the same information twice.

The mechanism now exists. `scripts/numerics.py capture` writes `binary_sha256` (from
`--binary` or `SPARKFLASH_BIN`), `commit` (from `--repo`, default the checkout the tool runs
from) and a `build_note` naming why either is null into the reference's `server` card. The card is
outside the fingerprint, so naming a build changes no bits, and a reference captured before the
fields existed stays valid.

The owner's decision of 2026-11-18 — moving decode to the tensor cores is an approved arithmetic
change, and the output shift it causes is expected — now has a gate path: a `[[config]]` entry
may carry `approved = { decision, date, report }`, which carries one bump while the three global
prerequisites stay false (`docs/NUMERICS.md` §5). It does not excuse the per-model anchor, so
Inkling-2's re-declaration still waits on its own `[[anchor]]` and a teacher-forced capture; a
model that already has one, such as GLM-5.3, no longer waits on `coverage` and `monitors`.

### Inkling on the tick engine: three widths, one verify width, and the group's images

Commits `eddb478f` (the single process) and `d22208e8` (the group). Inkling had no
`enable_fusion`: its device arm opened a session a sequence (`InklingDeviceBackend::open_session`,
a `RuntimeSession` over a `Runtime`) and its group in
`crates/sparkflash-model-inkling2/src/group.rs` (`Fusion::group` over
`GroupLanes<Runtime<InklingWalk>, InklingWire>`). Both now serve on the tick engine. The walk, the
drafter, the kernels and the layout are untouched: the engine chooses *when* a chunk or a step
runs, and the runtime executes the same rows with the same shape.

**Single process.** A session opens on a `TickServe` over `InklingWalk`.
`InklingDeviceBackend::enable_fusion(capacity)` opens one engine for `capacity` lanes; a backend
that does not call it starts a one-lane engine for each session, which is the sequence memory a
`RuntimeSession` held before the move, so the batch runner (`sparkflash inkling-tp4`) and the
device tests keep their shape. No in-tree caller passes `enable_fusion` a concurrency, because
the CLI serves Inkling at TP=1 through its f32 host reference; the device arm's single process is
the batch runner and the tests, and each of them opens a session at a time.

**The chunk, exactly.** The arm's three widths are `DEFAULT_PREFILL_CHUNK` 4,096,
`LONG_PREFILL_CHUNK` 8,192 and `MAX_PREFILL_CHUNK` 65,535. Its widest width
(`Shared::prefill_chunk`) is what the sliding rings and the exchange slots are sized for, and a
*pass* has always been the runtime's `ChunkPolicy` over that width: every one of the three is an
entry of `CHUNK_SIZES`, and a context or a switch that lowers the width is quantised to the
largest documented size at or below it. `InklingWalk::limits` used to state the widest width and
the runtime quantised it; it now states the quantised number, and `TickConfig::for_walk` reads
that as `rows_max`, `chunk_rows`, `exact_tick_rows`, `busy_chunk`, `chunk_min` and
`bulk_cap`. Nothing else moved, so the pass is the number it was:

| configuration | widest width (`Shared::prefill_chunk`) | a prompt pass, before and after |
| :-- | :-- | :-- |
| the switch unset, context >= 8,192 (the released default: 262,144 or 1,048,576) | 8,192 | **8,192** |
| the switch unset, context 4,096..8,191 | the context | the documented size at or below it: **4,096** (5,000 is 4,096) |
| the switch unset, context < 4,096 | the context | the documented size at or below it (1,000 is **256**) |
| `SPARKFLASH_INKLING_PREFILL_CHUNK=4096` | 4,096 | **4,096** |
| `SPARKFLASH_INKLING_PREFILL_CHUNK=65535` (the whole-context pass) | 65,535 | **65,535** |

Inside a pass the arm's own width rule still cuts the rows: `walk_bulk` splits every
`Pass::Bulk` at the positions remaining (`ChunkWidths::next`: 4,096 rows until fewer than
16,384 remain, 8,192 above), so a 20,000-row prompt at the default is the engine's parts
8,192 + 8,192 + 3,616 and the walk's passes 8,192 + 4,096 + 4,096 + 3,616 - the same total split
the arm's prompt loop made before, because the rule depends only on the positions remaining. That
is what `the_runtime_chunks_do_not_move_the_arms_prompt_boundaries` holds, now with the policy's
pass as the part size.

| | Before (`RuntimeSession` / `Fusion::group`) | After (`TickServe` / `TickGroup`) |
| :-- | :-- | :-- |
| prompt part | `ChunkPolicy` rows = **8,192** by default | `exact_tick_rows` = `busy_chunk` = `chunk_min` = `bulk_cap` = `chunk_rows` = **8,192** |
| a 20,000-row prompt | parts 8,192 + 8,192 + 3,616 | the same |
| prompt rows at the whole-context switch | 65,535 + the tail | 65,535 + the tail |
| verify width | `Limits::members` = `BATCH_ROWS` = **8** | `TickConfig::members` = **8** |
| a member's verify rows | 1 + the depth `SPARKFLASH_INKLING_DRAFT_DEPTH` picks, **7** by default | `k = drafter.budget(1)` = min(7, 8 - 1) = **7** |
| a joint round's rows | `share_rows` over `budget(members)` = `(members x depth).min(8 - members)` | the same `budget` and floor, through the engine's `Shared` plan |
| joint rounds | the runtime's `JOINT_ROUNDS` | `for_runtime` reads the same switch into `shared_rounds` |
| drafter | `InklingDrafter`: the DSpark stage or prompt lookup, and `Draft::Held` when the switch is off | unchanged |
| the draft floor | `Drafter::floor()` = 0.0 | `share_floor` = 0.0 |
| decoder-layer skip | none; only DeepSeek has those layers | `decoder_skip = false` |
| row exchange | a fused group's | `exchange = false` at TP=1, `true` in a group (`with_group`) |
| a group's lanes | `GroupOptions::lanes` = `--concurrency` when `--fuse-batch` is given, else 1 | the same |

Neither path that could move the boundary does: `bulk_rows` returns the chunk and then its tail
with `bulk_cap = chunk`, and the exact tick is `exact_tick_rows = chunk`, so a group's bulk
chunks and a single process's exact ones name the same rows. A CPU test pins it on a 20,000-row
prompt (8,192, 8,192, 3,616) and on a 100,000-row one at the 65,535 switch (65,535, 34,465), on
both the exact path and a group's bulk path.

**What the group carries.** The tick group's wire with Inkling's widths: `LaneRequest` with the
start `start_prompt_slot` chose and the prompt's image spans, `LaneRows` for every prompt chunk
rank 0's tick decided, `LaneStepMode` with the member order and `Plain`/`Round { joint }`,
`LaneGiveBack`, `LaneRelease` and `Error`. Rank 0 keeps every decision - `pick_lane`,
`prefill_order`, `hold_prefill`, the chunk boundary, `decode_plan` and the tick-ahead plan - and
a worker checks only that the lane is in its pool, that the request's start is the one its own
`start_prompt_slot` reaches, that the rows fit the prompt left and that the step's members are the
frame's. A worker runs `Runtime::run_part_rows` with the frame's row count and
`step_mode_slots` with the frame's shape, so lane `n` is the same sequence on every rank and the
collectives inside a verify line up.

**Vision: the one configuration that stays on `Fusion`, and why.** The tick group's request frame
does carry a prompt's image spans (`image_frames`), and
`runtime.start_prompt_slot` encodes them through `InklingVision`: rank 0 runs the tower, every
rank receives the rows through one group sum (`exchange_sum`). That sum is a collective. The tick
group orders rank 0's `start_prompt_slot` *before* the request frame, because the frame carries
the start it chose, so rank 0 would enter the sum before any worker had received the request and
the group would deadlock on its first image. The fused group orders the frame first
(`GroupLanes::begin_prompt` broadcasts, then runs rank 0's start), which is exactly why it serves
images today. So:

* a group whose rank 0 loaded the tower (the default when the checkpoint has one and
  `SPARKFLASH_VISION` is not `off`) keeps **`Fusion::group`**, unchanged;
* a **text-only** group (`SPARKFLASH_VISION=off`, or a checkpoint that describes no
  tower) is the tick group, and `InklingWire::serves_images = false` refuses an image request
  *before* any frame - a refusal, not a deadlock, and not a group that dies after the request
  went out.

`SPARKFLASH_VISION` must be the same on every rank, like the rank files; a rank that
chose the other engine refuses rank 0's frames by name rather than running a different row. The
name is neutral because DeepSeek's device path reads the same switch; the Inkling name it
replaced, `SPARKFLASH_INKLING_VISION`, is refused with `SPARKFLASH_VISION` in the message. The
alternative - carrying the encode on the tick group - is a wire change (the request frame would
have to precede rank 0's start and a start frame follow it), and it is not written here: no
cluster was available to check it, and the fused group keeps the image path exactly as it was, so
no image-bearing request runs on an unverified wire. The single process does carry images: at
TP=1 `exchange_sum` returns without a collective (`world == 1`), so `TickSession::prefill_images`
encodes on the engine thread and nothing waits.

**CPU evidence.** `InklingWalk`'s generated checks now say `tick_engine() == true`, so
`whole = chunked = token`, `alone = four`, `continued = cold`, `rewound = cold`,
`drafted = plain`, `no unrecorded widening` and the TP leg drive the engine on the device host.
The CPU tests added here are `the_generated_checks_drive_the_tick_engine`,
`the_tick_config_is_the_inkling_profile` (8,192 pass rows, 8 members, `k` = 7, the arm's 0.0
floor, a drafter present and `shared_rounds` following the runtime's own switch, no decoder skip,
no exchange at TP=1), `the_tick_engine_hands_inkling_chunks_past_one_width` (20,000 rows as
8,192 + 8,192 + 3,616 on the exact and the bulk path; 100,000 at the 65,535 switch),
`the_tick_engine_keeps_inklings_joint_round` (three lanes, one `Verify` pass of three segments
and at most 8 rows, the trace's `plan=shared`), `a_speculative_round_keeps_its_accepted_rows_only`
(one lane, `Round { joint: false }` over the anchor and two drafts, the rejected token never
reaching the history, `plan=spec`), and `a_text_only_group_refuses_an_image_before_any_frame`.
The group's frames did not change, so `crates/sparkflash-runtime/tests/tick_engine_group.rs` is
reused **unchanged**: its frame order and its start agreement test
(`rank 0 starts the prompt at 4, this rank starts it at 3: the ranks are out of step`) are the
runtime's property, which is what Inkling's group relies on.

**Measured on this host (CPU; it has no GPU).**

* `cargo test -p sparkflash-model-inkling2`: **125 passed, 0 failed, 7 ignored** across its
  binaries (73 library tests including the six new CPU ones, then 4 + 4 + 8 + 16 + 2 + 8 + 10 in
  the integration binaries; the 7 ignored are the generated checks, which boot the tiny
  checkpoint on a device).
* `cargo test -p sparkflash-runtime`: **91 passed, 0 failed, 8 ignored** (77 library tests, then
  `sealed` 2, `tick_engine_drafter` 5, `tick_engine_group` 2, `tick_engine_trace` 2,
  `tick_trace` 1, plus 2 doc-tests; the 8 ignored are the 7 conformance stubs and 1 doc-test).
* `cargo test -p sparkflash-runtime -p sparkflash-model-inkling2`: 0 failed.
* `cargo clippy -p sparkflash-runtime -p sparkflash-model-inkling2 --all-targets -- -D warnings`
  and `cargo fmt --all -- --check`: clean at this tip.
* `cargo test --workspace`: **4,326 passed, 0 failed, 81 ignored**, 220 test binaries, exit 0, at
  `76ce0c72` and the same count at `24389eb7` before it. That is the workspace's count and not
  this move's: the other arms' and the CLI fold's tests are in it.

**Unverified without a cluster.**

* Inkling's device walk through the engine: the generated checks on the tiny checkpoint, then the
  served gate.
* The tiny checkpoint's own DSpark round through the engine: the CPU tests drive a shape walk, not
  `InklingDeviceSession`, so the round's bits are the arm's existing claim, not a measurement.
* The live trace on Inkling: `WallClock` readings, the per-tick flush and the rank-0 rule.
* TP>1: no Inkling rank pair or group ran here. The claim that a text-only TP=4 group on the tick
  group gives the same token ids as the pre-move `Fusion::group` build is the first thing the
  window must measure, and so is the image-capable group's unchanged path.
* Speed: no tokens/s, no step time, no time to first token. Nothing here says the tick engine is
  as fast as the fused one, and the regression clause's A/B has not run.

**This move is a schedule change: measure it against the same build's pre-move path.** (Inkling's
registered reference is `387647bc` and the tip reproduces it on the capture's tensor prefill walk;
`397907e6` is the same binary's chunk walk, not a stale reference - see "The Inkling reference
does not match on one prefill walk" above.) A comparison against the reference therefore also
depends on the walk, so the criterion for this move is a **pre-move build of the same tree**:

```sh
# the frozen pre-move path: the commit this move's first commit follows
git worktree add /tmp/inkling-pre eddb478f^
(cd /tmp/inkling-pre && cargo build --release -p sparkflash-cli --features tp4-serve)
cargo build --release -p sparkflash-cli --features tp4-serve   # the moved build, this tip
```

Pass criteria, in order:

1. **The single process, on the device.** The arm's device tests boot the tiny checkpoint through
   the engine and compare it with the host reference:
   ```sh
   cargo test -p sparkflash-model-inkling2 --test device -- --include-ignored --test-threads=1
   ```
   Pass: the same best token per step and the logits within the arm's `RELATIVE_TOLERANCE` of the
   host reference, and `reusable_prefix()` = the prompt plus the tokens fed, as before.
2. **The generated checks through the engine, then a served gate.**
   ```sh
   cargo test -p sparkflash-model-inkling2 --lib -- --include-ignored --test-threads=1
   cargo test -p sparkflash-model-inkling2 --lib conformance:: -- --ignored --nocapture
   scripts/onboard-gate.sh Inkling --url http://127.0.0.1:8000 --port 8000 --world 1 --apply
   scripts/onboard-gate.sh Inkling --url http://127.0.0.1:8000 --port 8000 --world 4 --apply
   ```
   Pass: six PASS and one SKIP - `TP 1 = TP 2 = TP 4`, whose SKIP names the world its fixture
   cannot boot (the Inkling fixture is one rank) - with no `NEEDS-FIXTURE` line.
3. **Tokens against the pre-move build at the deployed world.** Build both, run the same fixed
   prompts through each: greedy and sampled, **TP=1** (`sparkflash inkling-tp4` at world 1, or the
   device test's session path) and **TP=4** (the text-only group, with
   `SPARKFLASH_INKLING_VISION=off` on every rank), at **2K and 16K** prompt tokens, **one and
   four** concurrent requests (the group's `--fuse-batch --concurrency 4`), several reps each,
   alternating the launch order.

   These commands predate the switch's neutral name: they set `SPARKFLASH_INKLING_VISION`,
   which is now refused with `SPARKFLASH_VISION` named. The deployed site file changes in the
   deploy that carries that rename, and that deploy waits for the owner's go-ahead; this review
   does not edit it.

   ```sh
   SPARKFLASH_INKLING_VISION=off CONTEXT=32768 LANES=4 \
     ./scripts/inkling-serve-group.sh launch     # each build in turn, alternating
   ```
   Pass: identical greedy token ids, and per request the same reported `cached_tokens`. A
   difference is a regression, not a scheduling variance, because the move changes when a row
   runs and not which row. Sampled runs need the same seed and temperature, which the group's
   sampler frame carries whole.
4. **The image path is unchanged.** With the tower loaded (no `SPARKFLASH_VISION`), one
   image prompt at TP=4 on both builds: both must take `Fusion::group` and answer the same token
   ids. Pass: identical, and the boot line says "the fused group, images on". A text-only group
   must refuse the same request before any frame ("this group serves text only").
5. **The trace's chunk boundaries.** Only the tick engine writes a tick trace, so this is the new
   build's evidence, not a diff:
   ```sh
   SPARKFLASH_INKLING_VISION=off SPARKFLASH_TICK_TRACE=/tmp/inkling-tp4.trace \
     ./scripts/inkling-serve-group.sh launch   # rank 0's trace only; a worker writes nothing
   ```
   Pass: every `prefill` line's rows are 8,192 (4,096 with a smaller `CONTEXT`, 65,535 with the
   switch) except a prompt's tail, each `LaneRows` frame carries exactly the chunk the trace
   recorded, a `LaneRows` frame never carries a step, and no decode step holds more than 8
   members. Above 8 lanes a decode batch is plain rows in groups of 8 (the named deviation
   below), which the trace shows as `plan=plain`.
6. **Verify shapes.** From the trace and the arm's counters: one `Round { joint: true }` for two
   to eight lanes whose whole verify is at most `BATCH_ROWS` = 8 rows (members x (1 + depth),
   drafted by the arm's own budget), `Round { joint: false }` for one lane, and no verify while
   the drafter is off. With `JOINT_ROUNDS=off` the engine plans `Plain` (the named deviation
   below), so that run is a separate A/B and not the default.
7. **Speed, and this is what the window must measure first.** TP=1 at one request and at four
   lanes, and TP=4 at one and four lanes: tokens/s, mean step time and time to first token, the
   pre-move build against this one, alternated launches, several reps. Under the regression
   clause, a slower Inkling returns to `RuntimeSession` (TP=1) and `Fusion::group` (TP>1): that
   is a revert of `eddb478f` and `d22208e8`, and this section then records the number, the
   workload and why.

**Named deviations, so none is implied.**

* **The tick order is the engine's.** Which lanes step together and when is the scheduler's; the
  rows and the shapes are the arm's. A greedy token does not depend on when its chunk ran.
* **`JOINT_ROUNDS=off` plans `Plain` on a drafting arm** (the earlier GLM deviation, unchanged
  by this move): one plain token row a member in groups no wider than 8, where the fused runtime
  ran one round a member. Inkling's held chain is not opened on that setting - the runtime's
  plain branch runs no `open_round`, so no draft is proposed and thrown away - and the rows are
  plain token rows, so the tokens are still the `drafted = plain` property's.
* **More than eight decoding lanes are plain rows in groups of eight.** `BATCH_ROWS` is 8, so a
  fusion group of 9 or more lanes `--fuse-batch --concurrency 9+` steps plain token rows in
  groups of at most 8, where the fused runtime ran one joint round whose budget was 0 (every
  member's anchor and no draft). The rows per member are one either way; the pass shape is not
  the same, so that configuration is a separate A/B and the window's default group is 1 or 4
  lanes.
* **The vision configuration is the fused group** (above): that is the deviation the move makes
  instead of a wire change, and it is named in the ledger row.
* **The engine's pacing values are DeepSeek's**: `share = DecodeShare::new(0.5, 200_000.0)`,
  `model = (23.0, 4.0, 23.0, 4.0, 2.3)` and `prefix_reuse_min_rows`, from
  `TickConfig::for_walk`. They decide when a decoding lane takes a row during a long prompt, not
  what a row computes; the round's own cost stays the arm's `ExpertUnionWall` through
  `Drafter::wall`.
* **The resume start is `resume`'s checkpoint**, never an arbitrary shared prefix (the earlier
  deviation, unchanged): the engine picks the lane by its shared prefix and records the start
  `start_prompt_slot` chose, and the group's frame carries that start and is checked.
* **Adoption, lane growth, paired prompt chunks and preemption are still absent** (the earlier
  deviation): `Pass::Paired` is unused, and a preempted prompt is served to its end.
* **`enable_fusion` has no in-tree caller yet.** The CLI's TP=1 Inkling path is the f32 host
  reference, so the shared multi-lane engine is the opener a device-arm HTTP serve would call;
  the batch runner and the tests take the per-session one-lane engine, which is the path their
  bits are captured on.

### Phase 6, step 6, first slice: the rank openers leave `serve.rs`

Commits `a593e069` (the fold), `f869952f` (the ratchet) and the cleanup that follows them (the
unused opener alias and the arm-name tie). Phase 2 named this slice and could
not take it: "the rank openers (about 900 lines) leave `serve.rs`", and "the rank openers did
not leave `serve.rs` in this commit; that is phase 6". This is phase 6's step 6, first slice.
Measured instead of estimated, as the phase asks.

**The inventory.** Read after the slice and its cleanup. Every file the CLI serves from, what it
owns, and whether the runtime could own it:

| file | lines | owns | could the runtime own it |
| :-- | --: | :-- | :-- |
| `serve.rs` | 11,449 | the serve entry: option parsing, the engine choice, the reference CPU engine thread, the DeepSeek host group's driver (`Tp4EngineSpec`, `Tp4Driver`), the group transport opener (`open_pack_collective`, `with_group_doorbell`), the three rank step adapters (`HostTp4Rank`, `GlmTp4Rank`, `QwenTp4Rank`), the request loop and the response events | partly: the transport opener is mechanism 7 step 3's, and the step adapters are the response engine's shapes |
| `tp4_host.rs` | 5,245 | DeepSeek's host TP=4 rank: the resident checkpoint, the per-rank sequence, the lockstep host forward, the tcp seams (`tp4_seams`, `tcp_config`) and the prompt-logprob rank contract | yes, when `Batch` moves (step 4); the seams are step 3's |
| `device_chain.rs` | 5,180 | the `device-chain` subcommand: a staged harness (`--plan-only`, `--dry-run`, `--run-stage`), not a served path | no: it is a harness |
| `tp4_chain.rs` | 2,001 | the four-rank device arm's control channel and device epoch (Hello/Ready/Request/Token/End/Error) | yes: `sparkflash_runtime::group` is the same idea for the tick engine |
| `serve/device.rs` | 1,883 | `--tp4-device`: DeepSeek's device forward with DSpark behind the API | yes: the runtime's rounds, after step 4 |
| `serve/pack.rs` | 2,226 | the pack registry, `PackArm`, and the common entry both shapes run (single node and group) | it *is* the common entry |
| `serve/dynamic.rs`, `serve/dynamic_host.rs` | 1,172, 1,592 | the zero-code dynamic HF family's own host and group openers | no: it opens a family no arm names |
| `serve/dense_parity.rs` | 1,382 | a `#[cfg(test)]` module: the dense arm's parity fixtures | — |
| `glm_tp4.rs`, `inkling_tp4.rs` | 782, 173 | per-arm `device-chain --run-stage` harnesses | no: harnesses |
| `serve/recurrence.rs`, `serve/inkling_group.rs` | 421, 76 | Dory's recurrence ring and Inkling's group options | the ring, after DeepSeek (mechanism 7's closing note) |

The plan's 14,545-line figure is exactly `device_chain.rs` + `tp4_host.rs` + `tp4_chain.rs` +
`tp4_device_runner.rs` + `serve/device.rs` (5,180 + 5,245 + 2,001 + 236 + 1,883). This slice
does not touch those five files: they are DeepSeek's serving path, and they are what the later
slices of step 6 fold. What the runtime already owns is the strategies: `sparkflash-runtime` is
13,664 lines (`walk`, `runtime`, `fused`, `group`, `pool`, `resume`, `rounds`, `chunk`, `tick`,
`ledger`, `layout`, `share`), and the rank seam now sits beside them in `sparkflash-serve::tp4`
(`Tp4RankConfig`, `checkpoint_directory`), where both an arm crate and the CLI can read it.

**What moved.** Opening a rank is arm construction, so the arms hold it:

* `sparkflash-model-glm5-next/src/rank.rs` (new, 456 lines): `load_rank` (the checkpoint's
  `config.json`, the CUDA handle, the plan-agreement line, the collective call, the latent
  cache, the converted rank file or the checkpoint's tensors, the block drafter, the drafting
  rule), `open_rank` (the window allocation and the load line), `open_lanes` (the fused lanes,
  rank 0's hub and a worker's mirror), `glm_drafting`/`GlmDrafting` and `CONVERTED_DIR_ENV`.
* `sparkflash-model-qwen4-exp/src/rank.rs` (new, 168 lines): `load_rank` (the checkpoint's
  `config.json`, the collective call, the converted rank file, the load line) and `open_lanes`,
  plus `read_config` and `CONVERTED_DIR_ENV`.
* `sparkflash-serve/src/tp4.rs` (new, 73 lines): the rank seam, `Tp4RankConfig` and
  `checkpoint_directory`, because an arm's crate cannot read a type only the CLI names.
* `sparkflash-weights/src/rank_file.rs` (+27): `converted_rank_file`, the reader both arms'
  converted-rank-file paths and the CLI's single-node path now share.

The CLI keeps what is the group's and not the arm's: the collective opener, called back into the
arm with the same four arguments; the control-channel worker loop; the three step adapters that
map a rank's answer onto `Tp4RankApi`; and the resolution of `attention.latent_cache` and its
deprecated `SPARKFLASH_GLM_KV_CACHE` spelling, because that is the process's reading of the
environment, not a rank's (a model crate reads no environment here, which the arm table's
`ENV_READ_EXCEPTIONS` still holds at GLM 2 / Qwen 3).

**Before and after**, measured with the ratchet's own rule (every `.rs` under
`crates/sparkflash-cli/src`, line counts):

| | before (`a53057f3`) | after (the slice) |
| :-- | --: | --: |
| `crates/sparkflash-cli/src` (the ratchet's measure) | 36,987 | **36,566** |
| `serve.rs` | 11,771 | **11,449** |
| `serve/pack.rs` | 2,325 | **2,226** |
| `sparkflash-model-glm5-next/src/rank.rs` | — | 456 |
| `sparkflash-model-qwen4-exp/src/rank.rs` | — | 168 |
| `sparkflash-serve/src/tp4.rs` | — | 73 |
| `sparkflash-weights/src/rank_file.rs` | 1,046 | 1,073 |

So 421 lines left the CLI and 724 arrived in the crates that own them; the difference is the
documentation and the new seam's doc comments, which a shared type and a public opener carry and
a private one did not have to.

**What the ratchet holds.** `CLI_SOURCE_LINE_LIMIT` is **36,567** in
`crates/sparkflash-ops/tests/one_stack.rs`, one above the measured 36,566, the convention the
previous gate used. It was 36,988; it did not rise. The test still asserts the same three
things: the total, no `tp4_forward.rs` in the CLI, and the DeepSeek device forward in its own
crate.

**No arm's boot path changed.** The moved code is construction around the rank, not the rank's
step: the same `config.json` read with the same refusal wording, the same collective opened with
the same label, hidden width and run string, the same plan-agreement line before the gather, the
same rank-loaded line with the same origin and collective label, the same drafting line on rank 0
only, the same window allocation (`new_block_drafting_sequence` / `new_drafting_sequence` /
`new_sequence`) in the same order, and the same load milliseconds in the first metrics report.
Nothing in this slice touches a kernel, an arithmetic choice or a scheduling decision; the
`glm_drafting` rule, the depth policy and the collective's wire are the same code moved.

The seven `sparkflash serve:` boot lines the moved code prints are byte-identical to the seven
that stood in `serve.rs`'s loader block (`git show a53057f3:crates/sparkflash-cli/src/serve.rs |
sed -n '4600,5250p' | grep 'sparkflash serve:'` against the two `rank.rs` files: same lines, same
order, no reflow). What a window adds is the ranks printing them, and the numbers inside them.

**The window gate.** No boot is possible on this host (no GPU, no checkpoint), so the fold is
owed one window. Build the frozen tip and this one (`git worktree add /tmp/pre-fold a53057f3`),
then boot each arm's group in both builds and compare the boots and one served request:

```sh
cargo build --release -p sparkflash-cli   # default features: cuda + tp4-serve
# GLM, one request at a time and then fused lanes (the two shapes openers serve)
sparkflash serve --model glm-5.3-flash --checkpoint /models/GLM-5.3-Flash \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D
sparkflash serve --model glm-5.3-flash --checkpoint /models/GLM-5.3-Flash \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D --fuse-batch --concurrency 4
# Qwen, the same two shapes
sparkflash serve --model qwen3.8-flash-next --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D
sparkflash serve --model qwen3.8-flash-next --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D --fuse-batch --concurrency 4
```

Pass: every rank's stderr is the same lines in the same order in both builds (the plan
agreement word included, which is why the `fnv1a` that computes it moved to the shared weights
crate rather than a second copy), and one greedy request of the same prompt returns the same
token ids with the same `cached_tokens` on both. One rank that prints a different agreement
word, or one request that returns a different id, is a failed fold. A second boot with
`SPARKFLASH_GLM_KV_CACHE=f32` and one with it unset covers the one setting the CLI still
resolves for the GLM rank.

**What is left of the opener.** The step adapters (`GlmTp4Rank` with its step/speculate bodies,
`QwenTp4Rank` with its session step) are response-engine shapes and stayed in `serve.rs`
(about 490 lines together). They cannot move yet for one concrete reason: the trait they
implement, `Tp4RankApi`, names `sparkflash_model_deepseek_v41::prompt_ranks::MergedPosition` in
`prefill_prompt_logprobs`, and a model crate may not depend on another model crate
(`no_model_crate_depends_on_another_model_crate`). The next slice, in order: `Sampler` (17
lines, today in the Qwen crate and used by the GLM adapter) and `MergedPosition` move to
`sparkflash-model`, the trait moves to `sparkflash-serve::tp4`, and then the two adapters follow
the openers into their arms. That is the remaining ~490 lines of the plan's "about 900".

### Phase 6, step 6, second slice: the rank step adapters follow the openers

Commits `700fa47d` (the shared sampler and merged position), `04584ad4` (the rank step contract
and the adapters), `3ff9c822` (the gate) and the docs commit that follows them (this section). The
first slice named the remaining ~490 lines and the two concrete blockers: `Sampler` (the Qwen
crate's re-export, applied by the GLM adapter) and `MergedPosition` (the DeepSeek pack's type,
named by the trait in the CLI) had no shared home, and `Tp4RankApi` itself lived in `serve.rs`.
This slice removes both blockers and moves the adapters.

**What moved.**

* `crates/sparkflash-model/src/sampler.rs` (604 lines): `Sampler`, `SamplerMark`, `argmax` and the
  penalties/top-k/top-p/draw arithmetic, moved whole from `sparkflash-ops/src/sampler.rs`, which
  now re-exports it (`sparkflash_ops::sampler` keeps every existing path). The packs and the CLI
  read `sparkflash_model::Sampler`; the sampler's own tests moved with the code.
* `crates/sparkflash-model/src/prompt_ranks.rs` (45 lines): `MergedAlternative` and
  `MergedPosition`, moved from the DeepSeek pack and re-exported there
  (`sparkflash_model_deepseek_v41::prompt_ranks` and the `sparkflash-engine` facade keep their
  paths).
* `crates/sparkflash-serve/src/tp4.rs` (73 -> 256 lines): `Tp4RankApi`, `Tp4Step`, `Tp4Sampling`,
  `Tp4RankMetrics` and `PROMPT_LOGPROBS_REFUSED_PREFIX`. The CLI re-exports the four types from
  `serve::` and the constant from `tp4_host::`, so the driver, the pack registry and the loopback
  tests are unchanged.
* `crates/sparkflash-model-glm5-next/src/rank.rs` (456 -> 833 lines): `GlmTp4Rank` (the struct,
  `step`, `speculate` and the trait impl), `reusable_prefix` with its test, and the public
  `open_serve_rank` that boxes the adapter.
* `crates/sparkflash-model-qwen4-exp/src/rank.rs` (168 -> 372 lines): `QwenTp4Rank` (the struct, the
  session `step` and the trait impl), the per-step counter delta as `this_step` with its test, and
  `open_serve_rank`.
* The CLI keeps its entry points (`open_glm_tp4_rank`, `open_qwen_tp4_rank`), now one call each
  into the arm with the collective opener it already owns. `HostTp4Rank` stays in `serve.rs`: it
  adapts DeepSeek's host rank, which is the 14,545-line serving path the later slices of step 6
  fold, not a packed arm.

**Before and after**, measured with the ratchet's own rule (every `.rs` under
`crates/sparkflash-cli/src`, line counts):

| | before (`67146b74`) | after |
| :-- | --: | --: |
| `crates/sparkflash-cli/src` (the ratchet's measure) | 36,566 | **35,972** |
| `serve.rs` | 11,449 | **10,853** |
| `sparkflash-cli/src/tp4_host.rs` | 5,245 | 5,247 |
| `sparkflash-serve/src/tp4.rs` | 73 | 256 |
| `sparkflash-model-glm5-next/src/rank.rs` | 456 | 833 |
| `sparkflash-model-qwen4-exp/src/rank.rs` | 168 | 372 |
| `sparkflash-model/src/sampler.rs` | — | 604 |
| `sparkflash-model/src/prompt_ranks.rs` | — | 45 |
| `sparkflash-ops/src/sampler.rs` | 599 | 42 |
| `sparkflash-model-deepseek-v41/src/prompt_ranks.rs` | 872 | 853 |

So 596 lines left `serve.rs` and 594 left the CLI (two lines arrived in `tp4_host.rs` as the
re-export's doc); the seam and the two arms gained 183 + 377 + 204 lines, and the sampler and the
merged position moved out of the crates that had them (599 -> 42 in `ops`, 872 -> 853 in the
DeepSeek pack).

**What the ratchet holds.** `CLI_SOURCE_LINE_LIMIT` is **35,973** in
`crates/sparkflash-ops/tests/one_stack.rs`, one above the measured 35,972, the convention the
previous gate used. It was 36,567; it did not rise. The test still asserts the same three things:
the total, no `tp4_forward.rs` in the CLI, and the DeepSeek device forward in its own crate.

**No arithmetic, kernel or scheduling decision changed.** The adapter bodies are the first
slice's code, re-read line by line against `67146b74:crates/sparkflash-cli/src/serve.rs`; a
normalised `diff` of the moved blocks is empty except the seam's import paths, the refusal
prefix's `format!` argument, and the extracted `this_step` (the same three `saturating_sub`
calls). The `Sampler` file and its tests are the moved text, and the new
`the_ops_sampler_is_the_models_sampler` test pins the old path and the neutral home to one type,
so the re-export cannot fork into two histories. The one transcription the diff caught
(`self.held.extend_from_slice(&prompt[cached..])` in the GLM prefill) is restored: a window that
forgot the rows it ran is the one behaviour this move could lose without a kernel changing.

**Tests.** The moved inline tests run in their new crates: the whole sampler suite in
`sparkflash-model`, the GLM `reusable_prefix` cases (the prompt rows a request may reuse) and the
new Qwen `this_step` case (a step's share of a session's counters, including a reset or backwards
counter reporting zero rather than a wrap). `cargo test -p sparkflash-cli -p sparkflash-serve -p
sparkflash-model` passes; the CLI's loopback driver tests still drive their fake ranks through the
`serve::` re-exports, so the trait's contract is exercised at its new path.

**The window gate.** No boot is possible on this host (no GPU, no checkpoint), so the fold is owed
one window. Build the frozen tip and this one (`git worktree add /tmp/pre-step6b 67146b74`), then
boot each arm's four-rank group in both builds and compare the boots and one request:

```sh
cargo build --release -p sparkflash-cli   # default features: cuda + tp4-serve
# GLM, four ranks, one greedy request, then four fused lanes
sparkflash serve --model glm-5.3-flash --checkpoint /models/GLM-5.3-Flash \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D
sparkflash serve --model glm-5.3-flash --checkpoint /models/GLM-5.3-Flash \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D --fuse-batch --concurrency 4
# Qwen, the same two shapes
sparkflash serve --model qwen3.8-flash-next --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D
sparkflash serve --model qwen3.8-flash-next --checkpoint /models/Qwen3.8-Flash-Next-NVFP4 \
    --tp4-rank R --tp4-world-size 4 --tp4-peers A,B,C,D --fuse-batch --concurrency 4
```

Pass: every rank's stderr is the same seven load lines in the same order in both builds (the plan
agreement word included, which is why the `fnv1a` that computes it is the shared weights crate's),
and one greedy request of the same prompt per arm returns the same token ids with the same
`cached_tokens` on both. Rank 0's first metrics report must carry the same `load_ms` magnitude.
One different token, `cached_tokens` or agreement word is a failed fold. Because the first
request's `cached_tokens` is zero on both builds, the one counter the adapters' window
bookkeeping feeds is only exercised by a second turn of the same conversation; the window sends
that second greedy turn too, and a build that dropped the bookkeeping answers a different count.

### Inkling's registered v1 is degenerate: the head's few-row body read the row GEMV's packed layout

A cluster window serving Inkling-2-Small at world 4 (`scripts/inkling-serve-group.sh`,
`LANES=4`, `PREFILL=tensor`, `CONTEXT=1048576`, one checkpoint) forced the registered
reference's tokens through two builds and measured:

| build | generated | against `387647bc…` |
| :-- | :-- | :-- |
| the branch tip (`e60f00c5`, three commits past the decode change) | eight prompts decoded to multilingual token soup | **bit-identical**; `numerics.py check` PASS |
| `7850cffc` (the last build before it, head still on the row GEMV) | a clean chat opener | mean NLL **+5.824**, top-1 **29.49%**, max **47.09** |

The divergence is structured, not a precision tail: the worst position of every prompt is position
0. There the reference puts essentially all mass on `200001` (`<|message_model|>`, logprob ~0),
while `7850cffc`'s argmax is content token `125148` at `-1.156` and `200001` sits at `-48.25`.
Equal prompt ids, equal forced tokens, one decode step apart. (The side that assigns `200001`
probability ~1 is the **reference**, the captured tip; the pre-change build is the one that finds
`200001` at `-48.25`. The window's summary sentence is easy to read the other way round.)

**What was wrong.** Commit `84f07db7` moved the head's one-to-eight-row projection to
`q4p_dense_rows` and handed the launch the head's packed copy. That body's `Q4pRowsPacked` stage
decodes the **tiled** packed layout (`q4e_packed_tiled_a/b` in
`kernels/qwen4_prefill_sm121.cu`: plane A at `256 j + 8 l` of a step), while Inkling packs **in
place in the row walk's layout**: `device/packed.rs` builds
`PackedSet::new(form, Placement::InPlace)` and never calls `.tiled()`, and
`q4e_gemv_packed` is that layout's reader. `Shared::dense_rows` filtered the copy only through
`admits` (columns, row count, a 16-byte-aligned activation), which says nothing about layout, so
every decode head row rebuilt its chunks from the wrong offsets: plane A from another chunk, the
plane-B nibble from another chunk, the step's top from a different lane. The head's logits were
garbage at the first decode row and at every one after; `200001` maximised them.

The expert band's change is not implicated by this measurement. The few-row expert bodies take the
same grouped table, plan, split activation planes, outer scales and routing-weight-one convention
as `run_band_mma`'s `q4p_moe_gate_up_capped`/`q4p_moe_down` launches, and the window's signature
is the head's: the first decode row's hidden state is the prompt's own last row, which the head
change does not touch, so the reference putting probability ~1 on `200001` there is the packed
read, not the band. The expert rows are still **unproven on the released checkpoint**, and the
next window compares them (below).

**The fix.** Commit `6dd347c9`: `Shared::dense_rows` routes by the packed copy's layout
(`DenseRowsInput`): no copy → the BF16 rows; a tiled copy → the packed rows the body's stage
reads; the walk layout → refuse, and the head falls back to the row GEMV, which is the arithmetic
`7850cffc` ran and the window measured clean. `SPARKFLASH_DENSE_WEIGHTS=checkpoint` still takes
the body over the checkpoint's BF16 rows, where the body is the wide tile's row. A CPU test pins
the routing
(`sparkflash-model-inkling2`'s `device::tests::the_few_row_head_body_takes_only_the_tiled_packed_copy`).
No kernel changed, so the device test that compares the body with the wide tile is untouched and
keeps passing (`token_bodies_device::the_bf16_rows_body_is_the_wide_tiles_row`).

**The registered reference is degenerate.** `numerics/references/inkling2-tp4-sm121a.json`
(`387647bc…`) reproduces the broken output bit for bit: it pins the defect, not clean arithmetic.
It is **not** edited here and its registry entry is not moved. Inkling's v1 must be
**re-declared** after the fix: a fresh capture from a clean build with the same forced-capture
discipline (`capture --forced-from` the old reference, two captures bit-identical, the served
gate's lanes check, and a server card that names the build), reviewed and registered as the arm's
new version. **No deployment may be proposed on the current v2-less state**: the only declared
reference is known degenerate, so there is no declared arithmetic to serve.

**What the next GPU window runs.**

```sh
# 1. The bodies' device tests (unchanged: no .cu changed, nothing else on the device moved).
cargo test -p sparkflash-kernels --test token_bodies_device -- --ignored --nocapture --test-threads=1
cargo test -p sparkflash-kernels --test expert_rows_device -- --ignored --nocapture --test-threads=1

# 2. The fixed group on the released checkpoint, the capture's configuration.
BIN=<release build of the fix> LANES=4 PREFILL=tensor CONTEXT=1048576 \
    ./scripts/inkling-serve-group.sh launch

# 3. A greedy 32-token completion: the evidence is the text.
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"thinkingmachines/Inkling-2-Small","temperature":0,"max_tokens":32,
         "messages":[{"role":"user","content":"Hello, who are you?"}]}'

# 4. The forced comparison against the degenerate v1 reference, twice, bit-identical.
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
    --forced-from numerics/references/inkling2-tp4-sm121a.json \
    --out /tmp/inkling2-tp4-v2-a.json --binary <release build>
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
    --forced-from numerics/references/inkling2-tp4-sm121a.json \
    --out /tmp/inkling2-tp4-v2-b.json --binary <release build>
sha256sum /tmp/inkling2-tp4-v2-a.json /tmp/inkling2-tp4-v2-b.json
python3 scripts/numerics.py compare --anchor numerics/references/inkling2-tp4-sm121a.json \
    --candidate /tmp/inkling2-tp4-v2-a.json

# 5. The head's body against the row GEMV on the real checkpoint. Boot the same build with
#    SPARKFLASH_DENSE_WEIGHTS=checkpoint: there is then no packed copy, the head takes the
#    few-row body over its BF16 rows, and every other projection takes the BF16 GEMV. The packed
#    form's round-trip is checked at load, so this forced capture must be bit-identical to
#    /tmp/inkling2-tp4-v2-a.json.
SPARKFLASH_DENSE_WEIGHTS=checkpoint BIN=<the same build> LANES=4 PREFILL=tensor \
    CONTEXT=1048576 ./scripts/inkling-serve-group.sh launch
python3 scripts/numerics.py capture --url http://127.0.0.1:8000 \
    --forced-from numerics/references/inkling2-tp4-sm121a.json \
    --out /tmp/inkling2-tp4-checkpoint.json --binary <the same build>
sha256sum /tmp/inkling2-tp4-v2-a.json /tmp/inkling2-tp4-checkpoint.json
```

Pass is: the bodies' device tests still pass; the greedy completion begins with a plausible
assistant opener (not `200001` or a repeated special token); the two forced captures are
byte-identical; the forced comparison against the old reference reports a large **positive** NLL
delta of the pre-change build's order (+5.8 mean) - the forced tokens are the degenerate capture's
own soup, and a clean build assigns them low probability where the reference gave `200001`
probability ~1 - and the checkpoint-form capture matches the packed-form capture bit for bit.
Soup, a delta near zero, or a body/GEMV mismatch is a failed fix.

### The version-1 tail checks: Dory is precision, Inkling was a defect

The owner's rule for a tensor-core decode change is that a shift is expected and only a defect is
a reason to hold. Both arms were checked the same way: the pre-change build's forced capture
against the registered reference, then the 20 worst positions inspected.

**Dory1.0: an unstructured precision tail; version 1 stands.** The pre-change build (`84f07db7`,
Dory's decode still on the row GEMV) against the world-1 reference (`99715d…`): 1024 positions,
mean NLL delta −3.062e-05, mean absolute 4.962e-04, p99 6.610e-03, max **1.006e-02**, top-1
agreement **100%**, mean KL lower bound 2.586e-06. The worst twenty spread over six of the eight
prompts, their positions are scattered (no position-0 or boundary clustering, flat by position
bucket, uniform mod 16), the signs are mixed and the magnitude tracks the anchor's sharpness - an
accumulation-order tail, not a broken part. The reference's own text is coherent ("The user wants
me to solve a word problem step by step…"), which is what the Inkling check taught us to verify.

**Inkling-2-Small: structured, at the first decode row; a defect.** The same comparison gives mean
NLL +5.824, top-1 **29.49%**, max 47.09, and the divergence is the *first decode position of
every prompt*: the pre-change build puts essentially all mass on `<|message_model|>` (200001,
logprob ~0) and puts content token 125148 at −48.25, while the reference's argmax is 125148 at
−1.156. And the reference's eight generations decode to multilingual token soup, while the
pre-change build produces a clean chat opener. So version 1 pinned degenerate output and `check`
passing only proved the tip equals that capture. **Cause, found in `p5id-decode`:** Inkling packs
its weights in place in the *row walk's* layout, but the few-row BF16 body decodes the *tiled*
layout, and the head was routed to that body by shape alone (`Shared::dense_rows` filtered on
columns, rows and alignment, never on the copy's layout). Plane A's walk offsets are
`32·lane + 8·chunk` against the tiled `256·chunk + 8·lane`, so every head row rebuilt its chunks
from another chunk's bytes - only lane 0, chunk 0 coincides - and the logits were garbage. The fix
routes on the copy's layout and refuses the body for a walk-layout copy, so the head returns to
`q4e_gemv_packed`; a same-class guard makes the few-row band's down grid tile the output rows.
**Proven on the GPU (2026-11-19).** The fix's build (`5a1aead8…`, four ranks one hash) serves a
clean model: a greedy 32-token completion answers the question it was asked ("The capital of France
is Paris.", finish `stop`), where the degenerate reference's text was token soup. The forced
capture against the old reference gives mean NLL **+5.823**, top-1 29.49% - the size and shape of
the defect, now on the correct side. Two repeats are byte-identical. The head's body-versus-GEMV
check on the released checkpoint (the head takes the body over BF16 rows when
`SPARKFLASH_DENSE_WEIGHTS=checkpoint`, the GEMV otherwise) is **not bit-identical**, and the
earlier expectation that it would be is corrected here: 986 of 1024 logprobs differ by a last-bit
tail (median 2.0e-5, p90 9.4e-5, max 0.125) with the argmax identical at all 1024 positions -
the accumulation-order tail a body always has against a row GEMV, one order of magnitude smaller
than the layout defect's 47-nat swings. Version 1's re-declaration from a clean build is still
owed: the current reference pins the soup.

**A blocker the GLM coverage sweep found.** The GLM recorder was wired (the `combine_experts`
site) but the sweep wrote no dump at all: `scripts/serve-four-node.sh` starts every rank with an
explicit environment allow-list, and `SPARKFLASH_NUMERICS_*` was not on it, so rank 0's recorder
was never told where to write. The allow-list now carries `NUMERICS` (`885f5edf`), and the sweep
is owed a rerun on a binary built after the recorder landed.

## The 2026-11-19 verification window: version 2, and what production was actually serving

**Production never served the defect.** The deployed site tree's Inkling binary is
`/home/allenh1/inkling-src/target/release/sparkflash`, sha256
`bbe584d5045ef0abfc63b466f930504a4ffd73acd9bb7e92a650a9cc0cb8c600`, identical to
`/home/allenh1/inkling-bin/sparkflash.p18`, mtime 2026-11-17 10:02:52 - one minute after the
`b0a82f5` commit at 10:01:22, and a day before `7850cffc` (2026-11-18 23:46) and two days before
the few-row decode (`84f07db7` / `6dd347c9`, 2026-11-19). The binary was built in a tree
without a git HEAD, so its own card says `unknown`; the attribution is the build timestamp and
the staged tree. No Inkling process was running at the check, so the check is of the file the
site's launcher names, not of a live card.

**The first verification window failed for a build-environment reason worth recording.** The
release build of the fixed tip was made with `cargo` on `PATH` but without `SPARKFLASH_NVCC`
(or `/usr/local/cuda/bin` on `PATH`), and `sparkflash-kernels`' build script silently embeds an
empty cubin when it cannot find `nvcc`: the binary boots and only fails when an arm asks for its
image ("this build carries no Inkling cubin, whose image holds the exchange's stage and collect
kernels"). The build was repeated with `SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc`;
`target/release/build/sparkflash-kernels-*/out/inkling_sm121.cubin` is 2,057,056 bytes and the
binary grew from 188,987,080 to 217,428,688 bytes. Any Inkling build must set it.

**Version 2, and the tail the plan asked for.** The fixed tip `b62e2fcb` (binary `23666b8d`)
served at world 4 on the four DGX Sparks: `/readyz` in 26 s and the card named its commit. Two
free captures are byte-identical (`8270fdc4…`), the anchor captures are identical
(`02578670…`), and the teacher-forced capture of the anchor's own 128 positions scores exactly
zero. The forced comparison against `7850cffc`, the pre-change row-GEMV build (binary
`32b7bd03`), is a **precision tail, not a defect**: 970 positions, top-1 agreement 100.0000%,
mean NLL delta -5.002e-05, mean absolute delta 1.691e-04, max absolute delta 7.307e-02 (the worst
is `translation` position 68, 0.9156 against 0.8425, same top-1), 0 of the worst 20 positions at
position 0, and the worst 20 spread over 7 of the 8 prompts (`code` 6, `translation` 5,
`raw-continuation` 4, `tool-json` 2, one each in `summary`, `arithmetic` and `logic`). With the
head back on its GEMV, the remaining change against `7850cffc` is exactly the expert few-row
band and the MXFP8 tile, so this measurement closes "the expert rows are unproven on the released
checkpoint" end to end.

The known-answer checks decode `The capital of France is` to `" Paris. It is a beautiful city…"`
and `Hello, who are you?` to a coherent English opener. `2+2=` decodes `"5\n\nA\n\nAnonymous…"`
- and the pre-change build decodes the identical text with 970/970 forced tokens equal, so it is
the small model's answer, not a regression. The floor records it as a failed known answer with
that explanation rather than dropping it. The reference's own mean NLL is 0.1806 a token.

The reference is `numerics/references/inkling2-tp4-sm121a-v2.json` (`acd86d46…`), the anchor is
replaced by `inkling2-anchor-v2.json` (`a6f415ff…`, with the old card named in `replaces`), and
version 1 is retired with "pinned degenerate output from the head layout defect (`6dd347c9`)".
The declaration check passes with the owner's approval, because the model's coverage and monitor
rows are still missing.

## The tiled-head window: correct, slower, and kept out

The owner's decision (tensor-core decode on every arm) was only partly met for Inkling after the
layout fix, because the head fell back to the row GEMV. A tiled packed copy for the head (the
head's matrix packed in the wide tile's tiled layout, read by `q4p_dense_rows`) was built on the
layout-type branch, and the first build of it exposed a registry schema defect: the v2
declaration's `[[retired]]` entry carried `reference`/`fingerprint` and its `[[anchor]]` a
`replaces` block, but the Rust `RetiredEntry`/`AnchorEntry` deny unknown fields and declared
neither. `serve` parses the embedded registry at boot, so that release aborted before binding
(`unknown field 'reference'`); the workspace test that parses the embedded text catches it. The
schema now accepts both, validated like a config's reference (`985dcbf6`), and the model's 187
tests pass again.

The corrected build (`985dcbf6`, binary `8f1cdc99`) served at world 4 and was measured against
the GEMV head (`b62e2fcb`, binary `23666b8d`):

- **Correctness: a precision tail.** Forced on the v2 reference's 970 tokens: top-1 agreement
  100.0000%, mean absolute NLL delta 5.742e-06, p99 2.411e-05, max 2.655e-03, 0 of the worst 20
  positions at position 0, and the worst 20 spread over 4 prompts (`raw-continuation` 15, `code`
  3, `tool-json` 1, `arithmetic` 1). The GEMV control reproduced the v2 reference exactly
  (0.000e+00 at every position), so the delta is the tiled head's.
- **The tiled copy is exact.** With `SPARKFLASH_DENSE_WEIGHTS=checkpoint` the same build
  forced-captures the tiled default's own tokens with 0.000e+00 at every position: the body over
  the tiled packed copy and the body over the BF16 rows are the same bits.
- **It is slower, at one request and at four lanes.** Under the written speed rule (served
  concurrency four lanes, drafting as served), two alternated fresh-boot pairs: pair 1, tiled
  31.55 single / 59.56 four-lane against GEMV 33.61 / 62.95; pair 2, GEMV 33.38 / 63.06 against
  tiled 32.07 / 62.69. The ratios are 0.9387 and 0.9461 in pair 1 and 0.9608 and 0.9941 in pair
  2, against a four-lane bar of 1.02 and a single guard of 0.97 in both pairs.

The plan's regression clause keeps the GEMV, so the tiled head's packing and its +355 MB arena
were reverted (`4ee6eb45`). The typed packed layout (`Packed<L>`, `PackedSet<Tiled>`,
`body_copy`) and the route audit (`docs/numerics/v1-few-row-routes.md`) stay: they are what
makes the next tiled attempt a type-checked change, and the audit's Inkling head row still
describes the served path (walk copy → GEMV).

## The speed rule's first results

The rule was written before the runs (`e9ac17da`): a tensor-core route is kept when, on
alternated fresh-boot pairs, the served concurrency's aggregate tok/s is at least +2% better and
single-request tok/s no more than 3% worse, both pairs agreeing; otherwise the arm keeps its row
GEMV, recorded with the numbers. Inkling's tiled head failed both pairs at four lanes (0.9461 and
0.9941) and at one request (0.9387 and 0.9608), so it stays out; its four-lane and single numbers
are in `docs/numerics/inkling2/v2-clean-head.md`.

GLM-5.3's version 2, the FP8 tile body, fails the same rule decisively: on one build
(`18a76075`, binary `102d216d`) the ON route is 42.06 single / 56.22 four-lane against OFF
46.20 / 61.50, and 41.78 / 47.93 against 46.35 / 61.42 tok/s. The clean ratios are ~0.91 on both
axes (~0.78 four-lane on the loaded boot), against a +2% four-lane bar and a -3% single guard.
GLM returns to the tiled row GEMV as version 3, whose reference is version 1's own card and
whose report is `docs/numerics/glm53/v3-gemv.md`; version 2 is retired with that reason and the
monitor pair stays as the body's evidence. The declaration check learned the revert shape: a bump
that names the base's bits again is a change when the version between them is retired with its own
arithmetic, and the self-test pins both sides of that rule.

Dory is the largest loss and the cleanest verdict: the few-row body is 17.77 and 17.55 tok/s
against the row GEMV's 26.19 and 26.15 (ratios 0.679 and 0.671) at world 1, one process. The body
reads the BF16 rows while the GEMV build packs them beside the rows (23.1% fewer bytes a pass), and
a 33% deficit at one lane is past any bar. One-process Dory accepts one lane only
(`--concurrency`/`--fuse-batch` are packed-model knobs; the recurrence group needs two ranks), so
the 16-lane half of the rule degenerates to the single-request number and that is recorded rather
than worked around. Qwen's tensor-core decode is 35.82/35.96 against the token kernels'
36.81/36.77 at its default one lane (ratios 0.973 and 0.978): the single-request guard passes but
the aggregate is 2.45% worse, and the rule asks for a 2% win. Both arms therefore return to the
older route; their version 2 references are captured and stored as evidence
(`numerics/captures/dory10/v2-gemv.json`, `numerics/captures/qwen38/v2-token-kernels.json`), with
bit-identical pairs, precision-tail floors (top-1 100% and 99.39%) and templated known answers,
and the registry declarations follow in the next change.

Two mechanism defects the windows exposed are fixed with it: the declaration gate now parses the
registry with the binary's own parser (the `validate_registry` example) so a schema mismatch
cannot pass Python and abort a release at boot; and both launchers' stop/status patterns now match
a renamed binary (`[s]parkflash[^ ]* serve` and `[s]erve --model`), which had silently left four
ranks serving twice.

## The 2026-11-20 pass: one prompt half for every scheduler, and where onboarding cost sits

No cluster was used. Device runs are on one GB10 host (one rank), CPU runs on a development
machine. What that leaves unverified is listed at the end of each part.

### What the tip verifies on one GB10

At `3f7e62fa`, before any change of this pass:

* The generated checks pass on the toy walk, dense, Qwen, GLM and Inkling: whole prompt = chunked
  = token by token, alone = four at once, continued turn = cold, rewound turn = cold, no
  unrecorded widening; drafted = plain passes on Qwen, GLM and Inkling and is skipped where the
  arm declares no drafter. TP 1 = 2 = 4 is skipped on every arm: one rank.
* `numerics.py check` passes for Qwen world 1 version 3 (`37578a09…872901b`) against the released
  NVFP4 checkpoint and for Dory world 1 version 2 (`4f3af7e0…3955abc`).
* Dory has no `conformance!` invocation, so its resumed = fresh check is the served one:
  `served-gate.py turns --strict` at TP 1. Every continued turn equals its cold run, and every
  one reports `cached 0`: the single-process Dory session reuses nothing
  (`DoryDeviceSession::reusable_prefix` is `None`). The strict verdict is therefore FAIL for want
  of reuse, with no wrong token. That is a missing pattern, not a numerics defect, and it is the
  first row of "Dory" below.
* One observation, not diagnosed: the tip's Qwen serve at a 262,144 context reached about 102 GB
  resident on a 116 GB host.

### The share clock had drifted between the four ticks

The tick existed in four hand-kept copies: `tick::replay::Replay::tick` (the CPU model),
`TickEngine::tick`, DeepSeek's `Batch::tick`, and `tp4_forward/live_trace.rs` (a CPU mirror of
`Batch`). The text-equality gates hold their *decisions* together for one workload; they do not
hold the copies' control flow together. One rule had drifted: `Batch` restarts the decoders'
share clock only on a long chunk (more than `SHARE_MIN_ROWS` rows left when it ran), and the
replay, the engine and the mirror restarted it on any prefill. A short prompt between two long
chunks therefore held the next long chunk back in three of the four and not in the fourth.
`Batch` is the reference, so the other three were corrected and the two committed traces
regenerated; the difference is the long prompt's first chunk running one tick earlier.

### What landed

The decisions and the control flow that were `Batch`'s alone are now in `runtime/src/tick.rs`,
and each scheduler calls them:

| Piece | What it decides | Callers |
|---|---|---|
| `long_chunk` | which prefill restarts the share clock | all four |
| `tick_chunks` | the chunks one bulk tick runs: a chunk a rows state, the decoder's rows split off | replay, engine, mirror; `Batch::prefill_bulk` keeps its own loop (below) |
| `together` over `TogetherHost` | which prompts end together in one tick | all four |
| `ShareClock` | the clock's three values, `hold`, `hold_since`, `settle` | all four |
| `tick_prefill` over `TickHost` | the prompt half of a tick: adopt, hold, together or alone, the long chunk | all four |

The engine gained what the decisions need: `Limits::chunks` (the prompts one pass may hold),
`Runtime::run_parts_paired` (several prompts' rests in one `Paired` pass, each headed from its
own last row), and `TickConfig::ways`. They are inert for the four served arms: none declares a
paired pass it can run (Qwen and Inkling refuse it, GLM runs it as bulk), so `ways` is 1, and a
group's wire carries no paired frame, so a group is 1 as well. CPU tests pin both the three-way
behaviour and the refusal of a config wider than the pass.

`Batch::tick` now runs `tick_prefill` through a host wrapper (`Ticking`): the order, the copy
from another lane, the whole-chunk test, the shared rows, the lane memory and the two prefills
stay `Batch`'s; the rule that chooses among them is the runtime's. Three things about it are
deliberate:

* `Batch::prefill_bulk`'s slot loop was not moved onto `tick_chunks`. Its sizes interleave with
  the Engram reads it submits for the next tick, and a restructure there needs the device gate.
  It is the one remaining copy of the chunk rule.
* `place` and `decode` are still each scheduler's own. `place` differs in what commits a start
  (the runtime's `start_prompt_slot` on the engine, `decoder_resume` and the lane's memory on
  `Batch`); only its candidate scoring is common, and that is already `pick_lane`.
* `live_trace.rs` stays for now. With the prompt half shared it mirrors only `place`, the two
  prefills and `decode`, which is what `tick::replay` also models; it can be deleted once those
  are shared, and the phase 6 text that names it as the gate then names the replay instead.

One defect was found by the audit and fixed: the dense arm's `head` recorded a sampled token
twice (once before the argmax test, once again on the sampling path), so a request with
penalties counted every generated token twice and the sampler's history ran one token a step
ahead of the position a rewind truncates it to. Greedy requests without penalties take the
argmax path and were unaffected, which is why the generated checks did not see it. The second
record is removed and `a_sampled_step_records_its_token_once` (a device test) fails on the old
code and passes on the new. No released model serves on the dense arm.

Verified, CPU: `cargo test -p sparkflash-runtime` (the replay trace, the engine trace, the
drafter, group and sealing tests, and the new unit tests of each piece), clippy clean. Verified,
one GB10: `sparkflash-deepseek-forward`'s library tests (299 pass, the mirror gate among them),
the generated checks on all five crates again after the engine moved onto `tick_prefill`, and the
dense arm's whole test module with its device tests.

Not verified, and owed to the cluster: `Batch` on a device. The prompt half is a restructure
with the same calls in the same order, but DeepSeek's outputs are gated by the tick-trace A/B
and the served gate, and neither has run. The TP checks of every arm are skipped on one rank.

### Where the run path stands

| Model | Scheduler | On `Walk` |
|---|---|---|
| DeepSeek V4.1 | `Batch` (its own; shares the decisions and the prompt half) | no |
| Qwen (three selectors), GLM | `TickEngine` through `TickServe`, alone and in a group | yes |
| Inkling | `TickEngine` alone and in a group, images included; a lone rank still has an f32 host path with no walk | group only |
| dense (any catalogue-only model) | `TickEngine` alone and in a TP group | yes |
| Dory | its own two `ModelSession`s (device, recurrence ring): own lanes, chunking, history, rewind and ledger | no |

So the runtime carries one scheduler (`engine.rs`, with the decisions in `tick.rs`), and two
models are outside it. The fused engine (`fused.rs`, 2,290 lines) is gone.

### What a new family costs today

Measured at this tip, by reading (line ranges are in the two sweeps' notes, not repeated here):

* **The walk.** 391 non-test lines for dense, 829 for Qwen, 864 for GLM, 998 for Inkling. The
  `impl Walk` block itself is 173 to 203 lines in each; the rest is the sequence, the drafter and
  the arm's row plumbing.
* **Test scaffolding.** More than 60% of each walk file was its test module (658 to 808 lines),
  and most of that was the same CPU shape walk, the same tick-engine profile tests and the same
  fixture helpers, copied with the type name changed (Qwen's and Inkling's shape walks differed
  in 61 of about 322 lines). The shape walk is now one type,
  `sparkflash_runtime::testing::ShapeWalk`, and the test modules are 132 to 208 lines for the
  three text arms. `conformance!(Arm, profile: stated())` emits their compute-row, tick-profile
  and round tests from one stated `testing::ArmProfile`, and the fixture helpers are
  `FormatRule::classes` and `testing::token_bytes`.
* **Serving glue.** `enable_fusion`, `enable_group_fusion`, `mirror_group`, `lane_limit`, `wire`,
  `group_rank` and a serving enum in each arm: three to eight lines of body each under ten to
  twenty of comment, near-identical between GLM, Qwen and dense.
* **Outside the crate.** About 25 edits in about 15 files on the cheapest route: the descriptor
  and `MODELS` row, the `Arm` impl and `ARMS` entry, the `ARM_LEDGERS` row, the workspace and
  both CI files (a test enforces those), then the sites `ARMS` does not derive: a `PackArm`
  impl or a family arm in `dynamic.rs`, the codec and `Dialect` match, a response engine, the
  converter's branch, the web console's `deployment_for` row, the launch script's environment
  pattern, and `onboard-gate.sh`'s crate case. The console's world size still repeats a field
  `Arm` already has.
  The first cut of step 1 below took the repeats out of the CLI. `PackArm` is now a family's code
  only (configuration, open, codec, response layer); a release is a `Release` impl on the model
  crate's marker type, about ten lines, whose model id, profile, default context and output cap
  are read from its `ARMS` entry and the registry. `TYPED_PACKS` rows are built from that
  (`TypedPack::of::<R>()`), and both routes read the one table: `--model` by weights, a dynamic
  `--checkpoint` by family (`typed_for_checkpoint`), so `dynamic.rs` names no released pack
  family. The converter's default world and `released` flag are `Arm::conversion()`. A second
  release of a served family (an FP8 or a smaller sibling) is an `ARMS` entry, a `Release` impl
  and one table row; `QwenFp8Arm` and `Qwen36Arm`, 165 lines of delegation, are gone.
* **Two traps.** The CLI line gate has little headroom: the CLI is 35,905 lines and the gate is
  `< 35,973`. And the group default context in `dynamic.rs` is Inkling's constant applied to
  every `--rank` group (no other family reaches it today: `--rank` is refused for them).

Against the plan's test for done (a family supplies its `Architecture` mapping, the arithmetic of
its own parts, and its weight names, tokenizer and template), the gap is these four, in this
order of size: the dispatch outside the crate, the hand-written walk, the copied scaffolding, the
glue.

### Is the unified run path being reached the cheapest way

Three findings, each with the change it implies.

1. **Phase 6 does not lower onboarding cost; mechanisms 2 and 5 do, and neither is finished.**
   Moving DeepSeek onto the runtime's engine removes a program and gives the other arms batched
   prompts, tick-ahead and adoption. A new family writes exactly as much after it as before. The
   two mechanisms that decide what a family writes are the generic walk with part plug-ins
   (mechanism 2, phase 4: `PlanWalk` and `PartExec` do not exist; the dense arm's walk is the
   generic loop with two closed enums, `Mixer` and `LayerFfn`, as its only extension points) and
   the serving dispatch derived from one registration (mechanism 5 explicitly leaves it
   underived). Recommended order from here: the derived dispatch, then the part plug-in, with
   phase 6 continuing underneath as bit-neutral steps.
2. **Phase 6 is cheaper as shared halves under host traits than as a second engine grown to
   parity and a cutover.** The plan's route builds `TickEngine` feature by feature until it can
   replace `Batch`, then moves DeepSeek across in one step with an A/B. This pass took the other
   route for the prompt half: one function, and each scheduler a host of it. It removes the drift
   by construction, needs no cutover, leaves `Batch`'s device work where it is, and each step is
   pinned on the CPU by traces that must not change. The same shape applies to the bulk chunk
   loop (with the read submissions as host calls) and to `decode`'s plan. Whether DeepSeek then
   becomes a `Walk` is a separate and later question; decision 4's condition (no speed loss) is
   easier to meet when its device work has not moved.
3. **The arm contract still makes every arm write what the runtime could own.** Each of these is
   in all four walks:
   * the head's record-then-choose over `HeadInput` (and it is where the dense defect was);
   * a stash between `walk` and `head`, because `Hidden` carries only a row count (dense keeps
     device scratch, GLM three fields, Qwen a queue under a lock, Inkling three fields);
   * `LaneState`'s bounds checks, with the same error strings;
   * the lane memory the runtime hands `open`, discarded by all four (`let _ = memory;`);
   * the draft budget formula and the expert-union wall's construction.
   Moving the first two into the contract changes every arm's `head` and so needs each arm's
   generated checks; one GB10 runs all five suites in about three minutes once built, so this is
   work for a device host, not for the cluster.

### What to do next, in order

1. **One serve entry for any `Walk` arm.** A `WalkArm` item in the runtime (build the walk, name
   the wire, say whether this is a group's rank) with generic `serve`, `serve_group`, `mirror` and
   `lane_limit`, and the three text arms' glue deleted onto it. Then `Arm` gains the crate name,
   the dialect and the lane count, and `TYPED_PACKS`, the codec match, the converter's lists and
   the console's deployment rows read `ARMS`. This is the step that turns "25 edits in 15 files"
   into one declaration, and it lowers the CLI line gate as it goes. The group halves need a
   group to verify.
   Done: `sparkflash_runtime::WalkArm` and the four entry points, with GLM's, Qwen's, the shared
   dense arm's and Inkling's serving glue and enums moved onto it; `Arm` now states the dialect,
   the group defaults, the converter, the console's deployment row and the crate and switches,
   and the CLI's codec, response engine, group defaults, converter dispatch and the console's
   deployment rows read it. The CLI line gate records the gain. Left: Inkling's single-process
   f32 host path still boots in `dynamic.rs` (routing it through `serve_arm_identity` would add
   the kernel catalogue's boot gate, which that path deliberately bypasses). The dense and
   Inkling group halves moved onto the tick group in item 4. The group halves of the
   pack drivers (`serve_group`, `serve_group_lanes`) and `WalkArm`'s own `serve_group` and
   `mirror` compile on the new split and owe one multi-rank run.
2. **The scaffolding.** One parametrised CPU shape walk in the runtime replaces the three copies:
   done, as `sparkflash_runtime::testing` (`ShapeWalk` with the arm's layout, limits, budget rule
   and step wall; `prompt` and `decode` drive a tick engine to a request's answer). An arm states
   a constructor of ten to twenty lines where it carried about 320. GLM's copy took the stricter
   behaviour of the other two (one segment a bulk pass, a paired pass refused, the pass hook
   called); its 91 tests pass unchanged. `conformance!` now emits the rest. An arm states one
   `testing::ArmProfile` (its ledger name, layout, limits, draft depth, `DraftBudget`, wall and
   the chunks its prompts run in) and invokes `conformance!(Arm, profile: stated())`, or with
   `rounds` when its walk drafts. The `profile` module checks the declared compute rows against
   the arm's ledger entry, the ledger line, the tick engine's config, and every stated prompt's
   chunks on the engine's exact path, a group's bulk path and the runtime's prompt loop. The
   `rounds` module checks the budget against the verify width, the engine's joint and
   speculative rounds and the runtime's lone, shared and joint-off rounds. A fixture takes its
   format classes from `FormatRule::classes` and its two byte counts from `token_bytes`. The
   test modules are 132 (GLM), 172 (Qwen), 208 (Inkling) and 436 lines (dense, most of it the
   fixture writer); dense's own recording walk is gone. What an arm still
   writes by hand is what only it can state: its fixture, its layout function, its chunk rule.
3. **The contract** (finding 3). Done for what was a real duplication:
   * `Walk` gains `type Hidden: Residual`; `walk` returns it and `head` takes it by `&mut`, so
     a pass's answers travel in the value. GLM's two sequence fields, Qwen's queue under a lock
     and Inkling's three fields are gone. The runtime reads `Residual::rows` and nothing else;
     a `WalkHook` still sees the plain `Hidden`. Dense names `Hidden` itself: its scratch is
     the device buffer the residual lives in, not a copy carried between the two calls.
   * `HeadInput::record` is the one sampler record a head makes for a row (a prompt resets and
     records the prompt, a token or verify row records its token). Dense, GLM and Inkling call
     it. Qwen's token and verify rows are recorded inside its batch during the walk, and its
     prompt record has never reset the sampler, so its head is unchanged. The choice after the
     record is three different rules (dense's device argmax, GLM's and Qwen's end-of-turn list,
     Inkling's Gumbel draw) and stays with the arms.
   * `check_checkpoint` and `check_rewind` are the bounds checks of every `LaneState`.
   * `Walk::open` takes no argument; the runtime keeps the lane's reservation with the slot as
     before, and the accounting it prints is unchanged.
   * `DraftBudget` (`AnchorsAndDrafts`, `OnePass(rows)`) is the budget rule of Qwen, GLM and
     Inkling, and `ExpertUnionWall::with_experts` builds each arm's wall from its fitted
     constant and the checkpoint's expert counts.
4. **Retire `FusedEngine`.** Done. The tick group's wire already carried a prompt chunk's rows
   (`LaneRows`, run on the worker by `run_part_rows`), so dense's group needed nothing new. A
   prompt whose start encodes images (Inkling's tower) sends its request frame first and the
   start after it (`TickGroup::begin`, `TickGroup::started`, `LaneStart`), so every rank joins
   the encode from the frame instead of waiting for rank 0's start. Dense's group and Inkling's
   image group run on the tick group; `FusedEngine`, `FusedSession`, `FusedArm`, `GroupLanes`,
   `Fusion`, `GroupEngine` and the arms' `group_engine` are deleted. A chunk boundary is the
   arm's `ChunkPolicy` chunk on either engine, so no bits move. Verified on the CPU (the
   runtime's tests, including the image-order and start-check gates) and on one GB10 (the arms'
   library tests and the generated checks); the group halves themselves owe one multi-rank run.
5. **Dory onto `Walk`.** It is the only served model with no reuse at TP 1 and the only one whose
   ledger is written by hand. Done for the single-process device arm: `DoryWalk` (in
   `crates/sparkflash-model-dory/src/device/walk.rs`) serves on the tick engine, so its lanes
   keep a conversation and reuse a prompt's prefix, its layout and rewind point are stated rather
   than hand-kept, and `sparkflash_runtime::conformance!` runs the same eight checks (six pass,
   `TP 1 = 2 = 4` and `drafted = plain` skip). The arm's `DoryDeviceSession` is gone. Left: the
   four-process recurrence group keeps its own ring session, and loop-1 drafting with it.
6. **The part plug-in** (mechanism 2). Done for the dense arm: `sparkflash_runtime::part` holds
   `PartExec`, `PartHost` and `PartRegistry`; the dense walk runs a list of parts built from
   `Architecture::steps()` through `Dense::catalogue()`, and `Mixer` and `LayerFfn` are gone. A
   part kind with no executor is refused by name at open. Left: `dense_device_refusal`, the
   arm's layer shapes and `naming::resolve` still close the set of kinds a checkpoint may name,
   and `PlanWalk` does not exist, so the released arms keep their own walks.
7. **Phase 6, continued as shared halves**: the bulk chunk loop, then `decode`'s plan, each with
   the tick-trace A/B on the cluster.

## The 2026-11-21 sprint: one walk for the catalogue arms, and a command that writes a family

Device runs are on one GB10 host, CPU runs on a development machine, and the served TP 4 checks
of the moved arms on the cluster (W4). The commits are the ones after `d7704d02` on
`onboard/sprint2`; nothing is pushed or deployed. No output bits of a released arm move: the
dense arm moved onto the shared walk first (W2), then GLM, Qwen, Inkling and Dory's
single-process walk (W4), each checked bit-equal against a record of the code before it; DeepSeek
keeps its own walk.

### W1: the served groups on one host

`scripts/one-host-group.sh` starts a dense, GLM, Qwen, Inkling or Dory group on one host (dry run
by default) and runs the served gate against it; `scripts/owed-cluster-window.sh` lists what a
one-GPU host cannot run. On the GB10:

| group | lanes | lanes sampled | interleave | turns |
| --- | --- | --- | --- | --- |
| dense TP 4 (one host) | PASS | PASS | PASS | FAIL |
| Qwen3.6 TP 2 | PASS | PASS | | FAIL |
| Dory | refused at 4 ranks (2 loops: world equals loops); at 2 ranks rank 1 exits before the group is ready (no NCCL ring on one GPU) | | | |
| GLM, Inkling | owed to the cluster (weights and world) | | | |

The kept-lane `turns` failure is in shared serving, not an arm: the 2026-11-17 binary passes it,
the reconciled tip and `d7704d02` fail it. It is not chased here.

### W2: the dense arm on `PlanWalk`

`sparkflash_runtime::plan` holds the walk the dense arm wrote for itself (layer loop, sequence,
record-then-choose head); the arm is its host. Dense `walk.rs` goes from 333 to 248 non-test
lines. `tests/onboard_example.rs` is a whole family with one part of its own in 325 lines and no
walk. The generated checks (TP 1 = 2 = 4), `part_plugin` (bit-equal logits) and
`onboard_example` pass.

### W2.5: what each released arm needs before it can be a `PlanWalk` host

Read from the arms at the tip; sizes are estimates, not measurements.

| arm | layer loop today | what the shared walk lacks | runtime growth | touched |
| --- | --- | --- | --- | --- |
| GLM | `DeviceRank::step`, 8-row steps, prompts under 16 rows on the token path | a residual role (mHC `mix_in`/`mix_out`, which needs the next branch's hyper weights and fuses the rank sum), an indexer role, host-split passes, a drafter hook (MTP and DFlash2), a head that maps EOS to none from host logits | ~300 lines | 1.1-1.3k |
| Qwen | the prefill plan's forwards and the batch's token/verify walk | a widened residual (hyper streams: fold and inject), the side role (PLE), QSA's indexer, the verify/drafter hooks, a prompt head that records through the sampler; its BF16 projections, router and prompt experts are other kernels than dense's, so dense's parts do not give its bits | ~600 | 1.5-2k |
| Inkling | two executors (`chunk.rs` rows, `device.rs` step) | a branch input that is not `norm(x)` (the block residual's carried chain), a post-combine stage (short convs), a summary push, layer-start and final-row taps (DSpark), a host chooser (Gumbel), the vision and drafter hooks, its exchange | 400-600 | 3-3.5k |

Every row needs the same three shared pieces first: a residual role the host executes in place of
the add, a side role landing on the layer input, and an indexer role whose selection a later
layer may read. `plan_roles` refuses all three by name today. Bits hold only where the host keeps
each arm's row splits and fused sums; that is the check each migration owes, and GLM's and
Inkling's served checks are owed to the cluster.

### W3: one registration, and `onboard --emit`

The recount is of the edits outside its own crate that a new family makes. The earlier count
above, about 25 edits in about 15 files, is for a family with an arm of its own. A family with
one part of its own on the dense host now makes 4 edits (workspace member, the CI files, the CLI
dependency and its `DENSE_FAMILIES` row), plus `MODELS` and `ARMS` only to serve under a
registry id. Each is held by cargo or a test except the last:

| edit outside the crate | before this sprint | now |
| --- | --- | --- |
| workspace member | hand edit | hand edit; cargo refuses a non-member |
| the two CI files | hand edit | hand edit; `every_model_crate_is_named_in_both_ci_files` |
| `MODELS` descriptor and `ARMS` entry (only to serve under a registry id) | two hand edits | two hand edits; `every_registered_model_has_an_arm`, `every_model_crate_is_an_arms_crate` |
| the ledger's weights list | hand edit (`weights_ids`) | derived: `ArmLedger::weights` reads `ARMS` |
| the CLI's route for the arm | inferred from the default weights id and `TYPED_PACKS` | `Source::route`; `every_pack_route_arm_has_a_typed_pack_row` holds both directions |
| the CLI's crate table (`--crate`) | hand edit | derived: `sparkflash_model::crate_for` |
| numerics status by weights | CLI copy | `sparkflash_model::weights_numerics_status` |
| the dense structure gate, layer shapes and tensor names, for a new mixer kind | three edits (`dense_device_refusal`, `Shape::of`, `naming::names`) | none: a registered part's mixer is the family's (`dense_device_refusal_hosting`, `AttentionNames::Own`) |
| reaching the family's parts from `sparkflash serve` | no registration point; the open calls themselves | one `DENSE_FAMILIES` row and the CLI dependency (no test holds it yet) |

`sparkflash onboard <id> --emit DIR` writes the crate (config, `architecture()`, `parts()` with
one stub per own part, a conformance test whose fixture refuses by name) and prints these edits.
On the worked example's family and on an MLA-shaped config the emitted crate compiles
clippy-clean and its generated checks refuse by name. `docs/adding-a-model.md` is 128 lines (was
619); its reference material is `docs/arm-reference.md`, and its section 4 names the check
holding each old checklist item.

Two policy changes to confirm: `one_stack`'s model-to-model rule now exempts
`sparkflash-model-dense`, since it is the host an emitted crate builds on; and the CLI source
gate stands at 35,858 of 35,865.

Not done from W3.2: the launch scripts' environment pattern still repeats `Arm` fields (the web
console's world sizes already read `Arm::deployment`).

### The proof of concept: tencent/Hy4-flash-FP8

[models/hy4-flash-onboarding.md](../models/hy4-flash-onboarding.md) records the run. Three
general parser changes read the release (per-head hashed Engram tables, a nested
`quantization_config`, eight named unmodelled switches); `onboard --emit` wrote a 175-line crate
whose one part is its MLA, and the dense host takes that mixer once registered. It is not
registered or served: the shared walk does not run its indexer, hyper-connection or Engram
roles, the same three pieces W2.5 names, and it is refused by those names.

### W4: GLM, Qwen, Inkling and Dory on `PlanWalk`

Each released arm's walk is now `PlanWalk<Host>`: `GlmWalk`, `QwenWalk` (over `QwenHost`),
`InklingWalk` and `DoryWalk` are the runtime's planned walk over the arm's own frames, parts and
windows, and the old `impl Walk` of each is gone. The contract grew only by defaulted hooks, each
added in its own commit before the arm that needed it, so no earlier host changed:

| arm | hooks it added to `PlanHost` | what stays the arm's own |
| --- | --- | --- |
| GLM | `begin_window`, `abandon_window`, `regions`, `PlanWindow`, `PlanSeq::keep` | the rank's stages (mHC mix as the frame, the indexer inside its attention part); the legacy drivers the CLI's TP 4 path and `device_forward` still call |
| Qwen | `walked_layers` (the host's `finish` runs a window's last layers: the overlapped halves), `KEEPS_LOGITS` | `forward_batch` for the unfused session; held answers reach the runtime only as `Draft::Held` |
| Inkling | `type Vision`, `rewind_slack`, `choose` (the Gumbel chooser), `logits_width` | the chunk and row schedules, the session's round/settle/rewind, the DSpark drafter and the vision path |
| Dory | `rewind_point` (its point one id short) | the recurrence group's ring scheduler and loop-1 drafting, which walk the same block schedule |

The check: `sparkflash_runtime::record` writes each arm fixture's generated-check outputs
(`SPARKFLASH_BITS_RECORD`), recorded once on the code before the moves; every move was gated on
every fixture's record being equal, not only its own arm's. A second Qwen fixture,
`Qwen38Fixture` (hyper-connections, PLE, QSA, MTP), was recorded first so the move covered 3.8's
paths. At the tip every record line is equal: dense, GLM and Inkling 12/3/3 (TP 1/2/4), Qwen,
Qwen 3.8 and Dory 12/1/1 (their fixtures have one rank).

The fixtures cannot reach tensor-parallel sums, sequence parallelism, the overlapped halves, the
gathered head or Inkling's exchanges, so the served gate ran on the cluster at release shapes,
TP 4, old build (`a98340c1`) against new (`3acf80e4`), the same flags for both:

| arm | lanes, greedy, equal to the old build's digests | lanes, sampled, equal | turns, old | turns, new |
| --- | --- | --- | --- | --- |
| GLM-5.3-Flash | PASS | PASS | FAIL | FAIL, same pattern |
| Qwen3.8-Flash-Next NVFP4 | PASS | PASS | FAIL | FAIL, same pattern |
| Inkling-2-Small, DSpark depth 7, both rails | PASS | PASS | FAIL | FAIL, same pattern |

`turns` fails on both builds alike: replies equal their cold runs, and prefix reuse is absent
(GLM) or occasional (Qwen, Inkling). The server logs the same reuse events on both builds; which
turns hit depends on the reply text, which differs run to run. Qwen's think-dropped replies differ
from cold on both builds. This is W1's shared-serving `turns` failure, not the move.

Two gate notes. Two source trees must not share one `CARGO_TARGET_DIR`: an archive's files carry
its commit time, so the newer tree's sources read as older than the first tree's artifacts and
cargo relinks nothing (the first attempt built two identical binaries; the gate now builds each
tree in its own target and refuses equal binaries). And `cargo test` without `--no-fail-fast`
stops at the first failing crate, so a battery step with a known failure hides every crate after
it.

The battery at the tip, on one GB10 with `--no-fail-fast`: fmt, clippy and the release CLI build
pass; workspace libraries 3,120 passed; the arms' integration tests 737 passed; the CLI's 338
passed. The 7 failures are the ones the tip had before the moves: the CLI's two plan-aware opens,
GLM's and Qwen's tiny `engine_builder` fixtures, and the laptop-only `qwen_text_alias`.

Code grew: 51 files, +10,011 / -4,835 in `crates/` since `d7704d02`. The moves added each arm's
host beside device code that the legacy entry points still call. The cuts that follow are GLM's
legacy TP 4 drivers (`glm_tp4.rs`, `tp4_host.rs`, `rank.rs`'s `GlmTp4Rank`) once the CLI's TP 4
path serves through the walk, and Qwen's `forward_batch` once the unfused session does.

### What a new family still needs

The walk is shared; the parts are not. A part is typed by its host (`PartHost`), so GLM's mHC
frame and indexer, Qwen's PLE and QSA, and Inkling's exchanges are reachable only from their own
hosts. That is why Hy4 is still refused: its indexer, hyper-connection and Engram roles exist in
GLM's and Qwen's hosts but not in the dense host a new family builds on. The next step is one
host whose frames and parts are the union of these, with the arms' hosts reduced to their
configuration; a family that only recombines existing roles then needs no host of its own.

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
