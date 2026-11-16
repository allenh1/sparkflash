# 03 — The shared runtime: one home for how rows run

This page is the plan for `crates/sparkflash-runtime`, and the record of what has landed. The goal
is that adding a model takes the model's arithmetic and nothing else: every strategy that decides
*which rows run together* (lanes, drafting rounds, prefill chunking, group mirroring) is written
once and every arm gets it.

## Why

The tree already centralizes what a model *is*: `Architecture` parses the released config,
`TpPlan` divides it, `EnginePlan` names one `PartRequirement` per part, the catalogue resolves
each to one kernel, and every `.cu` lives in one stack. It does not centralize how rows *run*:

* Each device arm hand-writes its run loop. Inkling's `device.rs` with `device/chunk.rs` is about
  9,700 lines; Qwen's `device.rs` about 6,000 and GLM's about 6,200. Each has its own prefill
  chunking, decode step, verify, session state and group code.
* The shared plan stops at boot. `sparkflash_ops::launchers` are validated, not executed, and
  `sparkflash_model::compose`'s generic layer walk has no user.
* Strategies exist once per arm, or in only one arm. Joint draft-and-verify over several
  sequences lived only in DeepSeek's `tp4_forward/batch.rs` (and Dory's recurrence group). Qwen's
  and GLM's fused lanes refuse drafting. A pack-path tensor-parallel group serves one request at a
  time; Inkling has its own group lanes.
* The tensor-core tile is a shared header, but each arm wires its own rule for when to use it.

So each optimization had to be asked for once per model: chunked prefill, tensor-core prefill,
lanes, drafting in lanes, parallel drafts.

## One contract

Every strategy above is the same operation: run some rows of some sequences in one pass over the
weights. A **segment** is one sequence's next rows; a walk takes several segments.

| strategy | segments in one walk |
| --- | --- |
| decode step | 1 sequence x 1 row |
| fused lanes | N x 1 |
| verify | 1 x (anchor + drafts), tentative |
| parallel drafts | N x (anchor + drafts), tentative |
| prefill chunk | 1 x chunk, committed, bulk arithmetic allowed |

An arm states which shapes its walk takes; the runtime runs everything else as loops over
smaller shapes. An arm that only takes 1 x 1 still serves correctly, and each larger shape it
accepts turns a strategy on without more model code. Every row of a walk must be the bits its
sequence's row would be alone (the exact path); the conformance test checks it generically.

The arm supplies:

* its sequences (state, position, growth and release, give-back within a round);
* its walk over segments;
* optionally a drafter (chains with per-draft probabilities, for several sequences at once);
* its step cost model (what a walk of `rows` rows over `seqs` sequences costs), for the row share.

## The crate

`sparkflash-runtime` sits between `sparkflash-model` (descriptors, sessions, the acceptance rule)
and the arms:

| module | what it owns | came from |
| --- | --- | --- |
| `fused` | slots, the engine thread, prefix claim, park and least-recently-used reclaim, multi-token runs | `sparkflash_model::fused` |
| `share` | the step wall models and the row share: each verify row to the draft likeliest accepted while it pays | DeepSeek's `share_rows`, `multi_step_ms`, `batch_step_ms` |
| `rounds` | one speculative round for several sequences: draft every chain, share rows, one walk, each sequence accepts | DeepSeek's `spec_multi_step`, generalized |
| `prefill` (next) | chunk sizing, interleaving chunks with running decodes | DeepSeek's scheduler |
| `group` | rank 0 decides and broadcasts, workers replay: lanes, runs, give-backs, releases; an arm supplies only its wire (`GroupWire`: admission, sampler and image frames) | Inkling's `GroupLanes` |
| `ledger` (next) | per arm, which strategies run natively, looped or off; printed at boot | new |

Device memory for lanes stays in `sparkflash_kernels::lane_memory` (it needs the driver); the
runtime only asks an arm's sequence whether it has room.

Below the runtime, `sparkflash-ops` gets one projection entry point that picks row GEMV, batched
GEMV, packed or the wide tile from shape, rows and precision, so tensor-core prefill is a
property of the stack, not of an arm.

## Multi-token runs

A drafting arm's session picks every token itself, so a round's accepted run is known when its
verify ends. The engine used to return one token a member a step; a member still feeding its
cached tokens then waited through the others' rounds and the rounds drifted apart. Now:

* after a step, the engine keeps feeding each member the token it just picked while the arm says
  the answer is already held (`FusedArm::ready`, no device work), and hands the whole run to the
  member's handle;
* the handle answers the following decode calls from the run while the caller feeds the token the
  run expects;
* when the caller stops early (a stop string, the token limit, a forced token), the handle tells
  the arm how many outputs went unused before its next job (`FusedArm::give_back`), and the arm
  rolls back within the round. An arm that cannot give back never holds a run longer than one.

Because every member drains its round in one step, every member needs a new round at the same
step, and one walk serves them all.

## Order

1. Parallel drafts in the runtime, Inkling first. DeepSeek's row share moves to `share` with the
   same decisions (its unit tests move with it unchanged).
2. The generic group mirror, so pack-path groups (Qwen, GLM) get concurrency.
3. Qwen and GLM adopt the contract: lanes with drafting.
4. The prefill scheduler moves into the engine (a prompt no longer stalls every decoder).
5. The projection dispatcher in `sparkflash-ops`.
6. Execute the typed plans directly, so a new arm starts as a schedule plus its unique kernels.
   `docs/adding-a-model.md` then describes the contract instead of a forward.

Steps 1-4 move scheduling, not arithmetic: the existing digests are their gate. DeepSeek keeps
its own driver for now; it donates logic, and porting it onto the contract is optional. What is
unique to a model stays with it: Inkling's convolutions and relative attention, Engram, Dory's
loop pipeline (which uses only the policy pieces).

## Status

| step | state |
| --- | --- |
| 1. parallel drafts, Inkling | done: joint rounds, on by default (`SPARKFLASH_JOINT_ROUNDS=off` opts out) |
| 2. generic group mirror | core done: `GroupLanes`, `mirror_lanes`, `GroupWire`; Inkling is its first user. Qwen and GLM groups need step 3 first |
| 3. Qwen and GLM on the contract | planned |
| 4. prefill scheduler | planned |
| 5. projection dispatcher | planned |
| 6. typed plans executed | planned |

### Step 1: what it measured

Inkling at four lanes, DSpark at depth 7, the served lanes gate on the lead:

| | four at once |
| --- | --- |
| before (each lane drafted on its own) | 43.2 tok/s |
| joint rounds | 54.5-56.3 tok/s greedy, 51.9 sampled |
| joint rounds off | 42.9 tok/s |

Every answer alone is the one the deployed build gave, greedy and sampled, and four at once equal
alone; the batch digests did not move.

The gate also caught second turns going cold on the lanes whose last run ended early. The host
decides whether to keep a finished request's sequence from `reusable_prefix`, before the park
gives the run's unused answers back. The slot had already fed those answers' tokens, so its
history ran past what the handle fed, and the handle named no history at all: the sequence was
dropped, and the next turn started fresh. Some turns still looked continued, because the fresh
session happened to land on the same lane and the arm rewound it on its own. The handle's history
is now the slot's less the run it has not answered (`a_run_left_unused_is_not_in_the_callers_history`).
`scripts/inkling-lanes-gate.py`, given rank 0's log (`LOG=`), counts the host's own
continuations, so a lucky lane no longer passes.

A second window sat beside it: a park with a give-back waits for the engine's current step, and
the sequence is in neither of the pool's lists meanwhile. A next turn arriving then would find the
pool full and no continuation, and take another conversation's kept sequence. Claims now wait
while a sequence is on its way back (`HeldPool` in `sparkflash-serve`), and
`a_turn_sent_while_its_sequence_parks_still_continues_it` fails without the wait.

### Step 2: what is left

The mirror is generic, but only Inkling serves through it. Qwen's fused lanes refuse a
tensor-parallel group (`enable_fusion`), and the pack-path group serves through DeepSeek's
`Tp4Driver` one request at a time. Moving them onto `GroupLanes` means each arm's group session
becomes a `FusedArm` slot (step 3), then a `GroupWire` of a few dozen lines each.
