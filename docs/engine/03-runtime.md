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
runtime only asks an arm's sequence whether it has room (`FusedArm::room`). An arm lays out
what grows with a sequence once (`Layout`), and one `LanePool` a rank backs, refuses and releases
every sequence's span from a fixed budget: Inkling's and GLM's sequences both use it.

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
| 2. generic group mirror | done: `GroupLanes`, `mirror_lanes`, `GroupWire`; Inkling's device group and GLM's `--tp4-rank` group serve through it |
| 3. Qwen and GLM on the contract | done: both arms' fused lanes draft and verify through `RoundArm`; GLM's group steps four lanes by default, on Inkling's lane memory |
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

### Step 2: what was left

The mirror was generic, but only Inkling served through it: Qwen's fused lanes refuse a
tensor-parallel group (`enable_fusion`), and the pack-path groups served through DeepSeek's
`Tp4Driver` one request at a time. Step 3 moved GLM's group onto `GroupLanes`. Qwen's group still
serves one request at a time; its arm sets no `PackArm::OPEN_LANES` yet.

### Step 3: Qwen and GLM on the contract

Both arms' fused slots are `RoundArm`s: a drafting member opens its round with the runtime, and
members whose rounds are open verify together in one walk (`run_round`, the expert-union wall
deciding when splitting is cheaper). Qwen3.8 at TP=1 from the converted rank, four lanes, the
served lanes gate (`lanes-gate.py`: four prompts alone, then four at once twice):

| | alone | four at once |
| --- | --- | --- |
| unfused, one lane | 42.3 tok/s | 42.6 tok/s |
| fused, joint rounds | 40.7 tok/s | 64.3-64.8 tok/s greedy, 41.2 sampled |
| fused, joint rounds off | 18.1 tok/s | 46.5-46.6 tok/s |

Every answer alone, joint rounds on or off, equals the unfused server's, and four at once equal
alone. A lone lane runs at 18 tok/s with joint rounds off, against 42 unfused. This is a speed
gap, not a correctness one, and it is not yet explained.

GLM's group (`--tp4-rank`, `--fuse-batch --concurrency N`) opens every rank as `N` lanes
(`PackArm::OPEN_LANES`, `open_glm_lanes`). Rank 0 serves its backend through the host driver
exactly as one node does (`serve_backend`). Every lane's prompt, step, give-back and release is
broadcast before it runs, and each worker replays them over its own lanes (`mirror_lanes`).

The ranks load and draft through the same loader as the one-request group (`load_glm_tp4_rank`):
the converted rank, the latent cache, and the DFlash 2 drafter with its adaptive depth. The two
shapes therefore serve the same tokens.

The workers' sampler must pick what rank 0 picks, because the picks decide how many drafts a
verify keeps, so the frame now carries the whole sampler (`sampler_frame`, penalties included).
GLM's arm serves text, so `GlmWire` refuses images.

Fused lanes are the GLM group's default: `--tp4-rank` with no `--concurrency` steps
four lanes (`pack::GROUP_LANES`). An explicit `--concurrency 1` is the opt-out. It serves one
request at a time through `Tp4Driver`, which answers a third concurrent request with a 503.

A lane's sparse caches (every DSA layer's latent, scales and indexer rows, and the MTP layer's in
a drafting sequence) are a span of Inkling's lane memory. The KDA state and the drafter's scratch
stay plain allocations, because they do not grow. Each sequence reserves its whole capacity's
span, and a rank backs it from one `LanePool` only as far as the sequence has reached, plus a
1,024-row margin for draft writes and tile reads (`DeviceRank::back`). At 1M positions every
sequence allocated 18.2 GiB eagerly. Now four lanes share one full sequence's worth, so one
request can reach 1M while the others run, and a growth past the pool is refused on every rank
alike (`runtime.lane_memory_mb` sets the budget). A small context's pool holds at least sixteen
lanes' first chunks (`lane_memory::FLOOR_LANES`).

GLM at TP=4, DFlash 2 drafting, the served lanes gate on the cluster:

| | alone, greedy | alone, sampled | four at once, greedy | four at once, sampled |
| --- | --- | --- | --- | --- |
| `--concurrency 1` | 44.3 tok/s | 27.6 tok/s (no drafting) | one request at a time | one request at a time |
| default, four lanes | 43.2 tok/s | 37.3 tok/s | 57.3-57.7 tok/s | 47.1-47.5 tok/s |
| joint rounds off | 43.1 tok/s | | 43.4-44.0 tok/s | |

The one-request group drafts only for a greedy request (`GlmTp4Rank` takes the device's argmax);
a sampled one decodes a token a step with its logits row on the host. The lanes draft both.
With drafting off (`SPARKFLASH_GLM_DRAFTS=0`) the lanes run 27.2 tok/s alone greedy and 25.0
sampled, 53.9-55.2 and 43.8-44.2 four at once.

Every greedy answer alone equals the one-request server's, drafting or not, and four at once
equal alone, greedy and sampled. A sampled answer is the same with drafting on or off.

The one-request path's `Tp4Sampling` used to carry only the temperature and the seed, so it
sampled without the request's `top_p`, `top_k` and penalties, and two of the four gate prompts
answered differently from the lanes. It now carries the whole `Sampling` (`tp4_sampling`), and a
request with a penalty does not draft. DeepSeek's host rank, which has no penalty sampler, refuses
a request that sets one rather than ignoring it.

The group used to ignore `speculation.enabled`. It read only `speculation.drafts`, so
`SPARKFLASH_GLM_SPEC=off` was accepted and changed nothing. `enabled=false` now drafts nothing,
and is refused beside a draft count (`pack::glm_drafting`).

#### How deep the MTP layer drafts

The group's chain used to be a fixed two MTP drafts, a constant in the CLI. It now comes from the
profile, as every other drafting arm's does (`pack::glm_drafting`): `speculation.drafts`
(`SPARKFLASH_GLM_DRAFTS`), else the whole block for a DFlash 2 drafter, else `speculation.depth`
(three in the carried profile). The depth policy is the shared `DepthPolicy`, adaptive unless
`speculation.depth_policy = "fixed"` (`SPARKFLASH_GLM_DRAFT_POLICY`), with `speculation.min_depth`
(`SPARKFLASH_GLM_DRAFT_MIN_DEPTH`) as its floor, weighed against `MTP_STEP_COST`.

The MTP layer drafts one token a pass, so its passes are what an early stop saves. Its head now
writes each draft's probability beside the token (`q4e_argmax_prob_slices`, the same argmax plus
the softmax of the pick). After every pass short of the chain, `DeviceRank::draft_while` asks the
policy whether any row wants another draft and stops when none does. The lanes and the
one-request path both choose a round's depth from that round's own probabilities
(`DeviceRank::draft_chain`). The lanes used to choose from the previous round's, which calibrated
one round's confidence against another round's outcome.

The gate served GLM four ways on the same build. Every greedy answer equals the one-request
server's, and four at once equals alone in each:

| Serve | Alone, four prompts | Four at once | Streamed, prose / code / 4.3k prompt |
| --- | --- | --- | --- |
| one request at a time | 46.2 tok/s | refused by design | 47.9 / 49.1 / 49.7 tok/s |
| lanes, adaptive over 3 (default) | 44.0 | 56.3, 56.7 | 47.8 / 48.4 / 47.7 |
| lanes, fixed 2 (the old constant) | 43.1 | 60.5, 57.6 | 46.5 / 47.7 / 47.8 |
| lanes, adaptive over 4 | 44.7 | 57.2, 56.4 | 47.9 / 46.9 / 49.6 |

The three depths sit within run-to-run noise of each other: adaptive is ahead alone by 1-3%, and
the fixed two is ahead four at once in one run of two. Nothing measured better than the shared
default, so the profile keeps it. Sampled, the lanes run 37.8 tok/s alone and 47.3-47.4 four at
once; the one-request path, which drafts no sampled request, runs 25.5, and its sampled answers
now equal the lanes'.

#### When one rank fails

A lane frame is refused before it is broadcast or runs on every rank. Two failures fall outside
that: a frame that fails on rank 0 after the workers have it, and a rank that dies. Either leaves
the ranks out of step, and a group that is out of step cannot serve another token, so it now
stops as a whole:

- `GroupLanes::in_step` wraps every step, prefill, give-back and release that follows a
  broadcast. On an error it tells the workers (`Error { code: "group_out_of_step" }`) and ends
  rank 0 (`end_group`).
- A worker mirrors until the group ends (`mirror_until_the_group_ends`), and watches rank 0's
  socket (`control::watch_leader`). It exits when rank 0 goes away.
- Rank 0's hub watches the workers' sockets (`watched_hub`, `control::watch_workers`). It exits
  when one closes.

DeepSeek's device ranks already worked this way, and the watchers moved from its `device.rs` into
`sparkflash-comm` unchanged. GLM's and Inkling's lane groups now share them. In the gate, killing
one GLM rank (rank 2) or one Inkling rank (rank 3, launched through the console's
`serve-four-node.sh`) left no rank running a second later.

Two more fixes came out of this gate:

- **A give-back ran after the broadcast.** Rank 0 now gives a run back before it broadcasts the
  next job, so the workers replay the same rollback.
- **A sampled verify drew past its first disagreement.** The sampler's draw depends on the seed
  and the number of draws so far. A verify that chose every row then drew more often than the
  one-token steps it stands in for, so seeded output depended on how many drafts a round
  verified. `Sampler::choose_verify` stops at the first row that disagrees
  (`a_verify_draws_as_the_one_token_steps_it_stands_in_for`).

Two fixes came out of the gate:

- **Qwen's NVFP4 snapshot was refused.** `hf_quant_config.json` is ModelOpt's export, which only
  restates `config.json`'s `quantization_config`. Since Inkling's parameter-group reader, it was
  being read as Inkling's own document. It is now read only when it names parameter groups or
  changes nothing the config set
  (`a_modelopt_export_beside_the_config_keeps_the_configs_formats`).
- **The three penalty fields made `Tp4ControlError` too large.** `UnexpectedFrame` now boxes the
  frame it reports.
