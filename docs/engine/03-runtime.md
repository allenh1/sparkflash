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
| `fused` (prompts) | a long prompt a chunk at a time between the running decoders' steps (`FusedArm::begin_prompt`, `FusedArm::next_part`, `Part`); the arm keeps its own chunk sizing | DeepSeek's scheduler (`PREFILL_DECODE_INTERVAL`, `PREFILL_BYPASS_MAX_TOKENS`) |
| `group` | rank 0 decides and broadcasts, workers replay: lanes, runs, give-backs, releases; an arm supplies only its wire (`GroupWire`: admission, sampler and image frames) | Inkling's `GroupLanes` |
| `ledger` | per arm, which strategies run natively, looped or off (`Ledger`, `Strategy`, `Mode`); the engine states lanes, prompt parts and group, the arm the rest (`FusedArm::ledger`); printed at boot | new |

Device memory for lanes stays in `sparkflash_kernels::lane_memory` (it needs the driver); the
runtime only asks an arm's sequence whether it has room (`FusedArm::room`). An arm lays out
what grows with a sequence once (`SequenceLanes::plan` over a `Layout`), each sequence reserves a
span (`SequenceLane`) and backs it before a forward writes rows (`SequenceLane::back`), and one
`LanePool` a rank backs, refuses and releases every sequence's span from a fixed budget:
Inkling's, GLM's and Qwen's sequences all use it.

Below the runtime, `sparkflash_ops::projection` picks a BF16 projection's kernel (row GEMV,
batched GEMV, the skinny kernel or the wide tensor-core tile) from its shape, its rows and the
arm's `WidePolicy`, and gives the wide tile's launch (`WideTile`), so tensor-core prefill is a
property of the stack, not of an arm. GLM, Dory and Inkling route through it (step 5 below says
what is not routed yet).

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
7. The ledger: per arm, which strategies run natively, looped or off, printed at boot.

Steps 1-4 move scheduling, not arithmetic: the existing digests are their gate. DeepSeek keeps
its own driver for now; it donates logic, and porting it onto the contract is optional. What is
unique to a model stays with it: Inkling's convolutions and relative attention, Engram, Dory's
loop pipeline (which uses only the policy pieces).

### Then: every strategy on every arm

Steps 1-7 gave the strategies one home; the ledger shows which arms still run without one. The
next steps close those gaps, most served benefit first. The bar for each is the next model: a
step is done when a new arm gets the strategy by writing only the hook
[adding-a-model.md](../adding-a-model.md) names for it (a runtime default or a shared helper does
the rest), and when the served check that proves it runs on any model
(`scripts/served-gate.py`: `lanes`, `interleave`, `turns`). A strategy two arms implement alike
belongs in the default instead.

8. Prefix reuse on every lane: GLM's and Qwen's fused lanes keep their sequences between
   requests, as Inkling's do, so a chat turn forwards only what it added. The engine renews a
   slot whose prompt does not extend its history (`FusedArm::continues`, by default), so an arm
   names its history and nothing more. An arm that commits a verify at once keeps no row past
   the reply's end (`sparkflash_model::kept_to_end`). Otherwise a drafted end of turn leaves its
   sequence one token past the history, and an arm that cannot rewind loses the conversation.
9. A rewind point on every recurrent arm: GLM and Qwen keep their recurrent state at the end of
   each prompt, as Inkling does, so a turn whose client re-renders the reply (reasoning dropped)
   goes back to the prompt instead of starting cold.
10. Lane memory for Qwen: its lanes allocate their caches for the whole context at open; on the
    shared `LanePool`, as GLM's and Inkling's are, kept conversations share one budget.
11. Dory on the fused engine: its recurrence group keeps its own lanes, so it has no prompt
    parts, ledger or kept conversations of the runtime's. Done as Dory on the session contract
    with its own ring scheduling: the ring keeps a batch on every rank at once, which one fused
    walk a step cannot, so the arm keeps its scheduler and takes the contract's kept
    conversations, `resume` and ledger.
12. More typed plans executed (step 6): attention, MoE, the indexer and the residual mixers.
    Done for grouped-query attention (`GqaBody`); the rest need descriptor fields first.
13. The dispatcher's remaining rules (step 5): Qwen's split-K tile and the FP8 projections.
    Done: `split_k_slices`/`SplitTile` and `fp8_tile_fits`/`Fp8Tile`.
14. DeepSeek on the contract, optional: its kernels (tensor-core decode, Engram reads, the RDMA
    exchange) stay its own, its outputs must not change, and the contract would give it the
    ledger and the shared prompt parts. Done as the ledger: its batch scheduler states its
    strategies on the runtime's `Ledger` at start; its scheduler, kernels and arithmetic are
    unchanged.

## Status

| step | state |
| --- | --- |
| 1. parallel drafts, Inkling | done: joint rounds, on by default (`SPARKFLASH_JOINT_ROUNDS=off` opts out) |
| 2. generic group mirror | done: `GroupLanes`, `mirror_lanes`, `GroupWire`; Inkling's device group and GLM's and Qwen's `--tp4-rank` groups serve through it |
| 3. Qwen and GLM on the contract | done: both arms' fused lanes draft and verify through `RoundArm`; GLM's and Qwen's groups step four lanes by default, GLM's on Inkling's lane memory |
| 4. prefill scheduler | done: the engine runs a prompt past 256 tokens a chunk at a time, two decode steps between chunks (`SPARKFLASH_PROMPT_PARTS=off` opts out); Inkling, GLM and Qwen split their prompts into parts |
| 5. projection dispatcher | done: BF16 projections (GLM, Dory, Inkling) choose and launch the wide tile through `sparkflash_ops::projection`; since step 13 Qwen's split-K tile and the FP8 tile (Qwen, GLM) too; DeepSeek's MX family stays its own |
| 6. typed plans executed | in part: Qwen's prompt GDN scan and GLM's bulk KDA scan launch the kernel and grid of the rank plan's `ScanPlan`; since step 12 grouped-query attention its `GqaBody`; MLA, MoE, the indexer and the residual mixers still launch from the arms' own load-time geometry |
| 7. ledger | done: every fused engine prints `ledger <arm>: ...` at start; Inkling, GLM and Qwen state their strategies |
| 8. prefix reuse on every lane | done: GLM's and Qwen's lanes name their history; the engine renews a slot whose prompt does not extend it (`FusedArm::continues`); a verify keeps no row past the end of turn (`kept_to_end`); the thinking-off close is the codec's (`PromptCodec::close_thinking`). The host's reply text holds a character a token left unfinished until the next completes it, as tokenizers' `DecodeStream` does, so a reply re-encodes to its ids (a stray U+FFFD had broken a continued turn). Served `turns`: Qwen 10 of 10 continued turns cached past their first prompt; GLM 38 of 40 over four runs (was 1 of 10): one first turn cut at the token limit, whose drafted rows GLM cannot rewind (item 9), and one the host's log did not cover. Lanes greedy and sampled equal the last gates on both, Inkling's equal jr4's |
| 9. a rewind point on every recurrent arm | done (deployed as p14, 2026-11-17): one shared `Checkpoint` (`sparkflash_kernels::checkpoint`) copies an arm's listed regions at each prompt's end and back at a rewind; `sparkflash_runtime::resume` picks extend or rewind; `continues` and the host's claim take the point even when the slot names no history, and the host respells a next turn that merges the point's last id (Qwen's `<think>\n\n` after a held `<think>`, `\n`). Inkling's boundary moved onto it; GLM keeps its KDA layers' convolution and recurrent memory and the MTP layer's hidden rows, Qwen its GDN and PLE state, the MTP stash and the n-gram windows; under a block drafter (Qwen's served DFlash 2, GLM's) the point also keeps the drafter's context rings whole (84 MB a lane for Qwen's), since a ring slot is a position modulo the ring and a reply longer than a block overwrites the window before the point. GLM's `give_back` cuts the history rather than failing, so a cut reply rewinds instead of renewing. One switch for every arm, `SPARKFLASH_REWIND` (`SPARKFLASH_INKLING_REWIND` is refused with the new name). Served gate `turns` (think-dropped every turn): Inkling and GLM (MTP and DFlash 2) REWOUND and equal to cold, GLM's cut first turn REWOUND; Qwen REWOUND+ (cached one short of the first prompt), its answers differ from cold because the model reads its own held ids for the merged newline; lanes gates equal their baselines |
| 10. lane memory for Qwen | done: one helper, `sparkflash_kernels::lane_memory::SequenceLanes` (plan, reserve, back, room, release), lays out and backs every arm's lane caches; GLM's and Inkling's moved onto it, and Qwen's attention caches now back as a sequence grows instead of whole at open. Every arm takes the one budget rule (`LanePool::budget`): one whole sequence's chunks, at least `FLOOR_LANES` lanes' first step, or `SPARKFLASH_LANE_MEMORY_MB`. For Qwen at a 64K context that is a 1152 MiB pool where four lanes held 4 x 960 MiB before, so long prompts on several lanes at once share one context's worth and a growth past the pool is refused with the request's error; set `SPARKFLASH_LANE_MEMORY_MB` for more. Served gate 18: Inkling, GLM and Qwen lanes greedy and sampled equal their baselines (Inkling's equal jr4's), `turns` PASS on all three |
| 11. Dory on the session contract | done: the recurrence group keeps its own ring scheduler (a batch on every rank at once, which one fused walk a step cannot) and takes the contract's parts: kept conversations (a parked lane holds its rows between requests), `sparkflash_runtime::resume` and the ledger (`ledger Dory: ...` at start). Every row is addressed by its position, so a lane goes back anywhere at no cost; its point sits one id before the prompt's end, because Dory's template opens a thinking reply with `<think>\n` and writes a past reply back as `<think></think>...` (and drops earlier turns' reasoning), so the next turn parts from the prompt at its last id. A repeated prompt therefore rewinds onto its own lane too; `scripts/served-gate.py turns` evicts the kept lanes before its cold reference. Served gates 20-21 on p17: lanes greedy and sampled equal p15's; `turns` PASS on every shape, plain turns HIT, thinking and think-dropped turns REWOUND+, every continued answer equal to cold |
| 12. more typed plans executed | in part: grouped-query attention runs the plan's rule. `sparkflash_ops::attention::GqaBody::choose` picks the grouped body (four or six query heads a block, or the per-head `q4e_attn`), its grid and its shared memory from the query groups and the most tokens a query attends (`Indexer::max_selected`, now in the plan's geometry); the boot plan and Qwen's walk take it, the walk adding only which bodies the device granted. Bit-neutral by construction (every body computes each head as `q4e_attn`); Qwen lanes greedy and sampled equal p15's and its kernel test passes on the device. MLA, MoE, the indexer and the residual mixers need descriptor fields first |
| 13. the dispatcher's remaining rules | done: Qwen's BF16 split-K tile (`split_k_slices`, `SplitTile`) and the FP8 tile (`fp8_tile_fits`, `Fp8Tile`, which Qwen's dense FP8 projections and GLM's FP8 parts launch) live in `sparkflash_ops::projection`, each the arm's old rule verbatim (a unit test pins the split against Qwen's old function). GLM and Qwen lanes greedy and sampled equal p15's. Qwen3.8 NVFP4 does not take the dense FP8 tile, so that arm's FP8 path is checked by the shared conditions and unit tests, not served; DeepSeek's MX family stays its own |
| 14. DeepSeek on the contract (optional) | done as the ledger: the batch scheduler prints `ledger DeepSeek: ...` at start (lanes and lane memory, prompt parts, DSpark depth policy, joint rounds and their floor, prefix reuse, the group); its scheduler, kernels and arithmetic are unchanged (a boot line only; the forward crate builds and passes clippy on the device). Its prompt parts stay its own: its scheduler already splits prompts (busy and bulk chunks, a decode share) on the tick rank 0 broadcasts, and moving them onto the engine's would change that tick |

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
`Tp4Driver` one request at a time. Step 3 moved GLM's group onto `GroupLanes`, and then Qwen's
(below, "Qwen's group").

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
GLM's arm serves text, so its wire refuses images (`TextWire`, which Qwen's group shares).

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

#### Qwen's group

Qwen's `--tp4-rank` group now steps fused lanes as GLM's does. The batched walk was already
tensor-parallel: its MLP sums its branch over the group (`inject_branch`) at any row count, so
the change is wiring only. Rank 0 opens its backend as `N` lanes over the control hub
(`Qwen4ExpDeviceBackend::enable_group_fusion`), and each worker replays rank 0's frames over its
own lanes (`mirror_group`). Both arms take the same text wire (`TextWire`). `--tp4-rank` with no
flags steps four lanes (`pack::GROUP_LANES`), and `--concurrency 1` is the opt-out to the
one-request `Tp4Driver`. The vision tower is not loaded at TP, so the lanes serve text.

Qwen3.8 NVFP4 at TP=4, DFlash 2 drafting, the served lanes gate on the cluster:

| | alone, greedy | alone, sampled | four at once, greedy | four at once, sampled |
| --- | --- | --- | --- | --- |
| `--concurrency 1` | 65.5 tok/s | 25.4 tok/s | one request at a time | one request at a time |
| default, four lanes | 62.6 tok/s | 40.0 tok/s | 91.9-97.3 tok/s | 54.0-55.2 tok/s |

Every lane's answer, greedy and sampled, equals the one-request server's, and four at once equals
alone. A lone streamed request runs at the same rate either way: 57.5 / 62.2 / 84.3 tok/s for
prose, code and a 4.7k-token prompt on the lanes, against 58.3 / 62.8 / 84.8 one at a time.
Killing rank 2 left no rank running a second later.

### Step 4: a prompt no longer stalls every decoder

The engine queues a prompt longer than `PREFILL_BYPASS_MAX_TOKENS` (256) and runs it a part at a
time: after the members have taken `PREFILL_DECODE_INTERVAL` (2) steps, or when no member steps,
the front prompt runs its next part. A shorter prompt runs whole as it arrives. Both values are
DeepSeek's (spec 5.6.2); `sparkflash-scheduler` now names the runtime's. A part is one of the
arm's own prefill chunks, so the chunk boundaries and every bit stay as they were; what a chunk
leaves for the next one (caches, the drafter's state, the MTP's stash) is the sequence's own, and
the shared prefill scratch carries nothing between chunks. A group broadcasts each later part
(`LanePart`) so every rank runs it between the same steps. `SPARKFLASH_PROMPT_PARTS=off` runs every
prompt whole; any other value is refused at start.

An arm that does not split its prompts keeps the default `begin_prompt`, which runs the whole
prompt as one part.

The gate: three requests decode (greedy, streamed) when a long prompt arrives, at TP=4, on 4,096-row
chunks. Each decoder's answer equals its answer alone, and every answer is the same with parts on
and off. Inkling's batch digests and every lanes-gate answer equal jr4's, and GLM's and Qwen's equal
the last gate's.

| | long prompt | decoders' longest pause, off | with parts | long prompt's first token, off | with parts |
| --- | --- | --- | --- | --- | --- |
| Inkling (deployed config) | 120k characters | 26.2 s | 6.1 s | 26.6 s | 30.2 s |
| GLM-5.3-Flash | 60k characters | 10.2 s | 2.8 s | 10.1 s | 10.5 s |
| Qwen3.8 NVFP4 | 60k characters | 10.3 s | 2.7 s | 10.1 s | 10.3 s |

The decoders keep streaming while the prompt fills; the prompt's own first token pays for the
decode steps between its chunks.

### Step 5: one rule for the wide tile

GLM's bulk prefill, Dory's prompt walk and Inkling's tensor walk each carried their own copy of
the wide BF16 tile's geometry (128 rows by 256 columns, 256 threads, 96 KiB of shared memory,
whole 32-wide tiles) and their own test for when to take it. They launched the same kernel body
on the same grid, so the copies became one: `WideTile::call` gives the grid, block and arguments,
and `route_bf16` (or `WidePolicy::bulk`, for a walk the arm has already sent down the bulk path)
gives the kernel. What still differs between the arms is data, in `WidePolicy`, because each
arm's digests were recorded under its own rule:

| arm | takes the tile at | a shape the tile does not cover |
| --- | --- | --- |
| GLM | more than `GEMV_MAX_BATCH` rows | the skinny kernel (up to 32 columns over whole 256-wide steps), else the row GEMV |
| Dory | more than `GEMV_MAX_BATCH` rows | refused: the token GEMV would change the chunk's arithmetic |
| Inkling, tensor walk | any row count | the row GEMV |

Each arm still launches the kernel symbol from its own image; only the decision and the launch
shape are shared.

Step 13 moved two more rules into `sparkflash_ops::projection`, each the arm's old rule
verbatim (a unit test pins the shared split against Qwen's old function shape for shape):

- `split_k_slices` and `SplitTile`: Qwen's BF16 tensor-core path, a 128 x 128 tile whose split of
  K decides the sums. The split aims at 96 blocks, keeps at least eight 32-wide stages a slice and
  fits the partial buffer; the arm's walk takes the slices from it and launches through
  `SplitTile::call`. A new arm with the same kernel takes the same split, so its sums match.
- `fp8_tile_fits` and `Fp8Tile`: the 128 x 128 FP8 tile over whole E4M3 scale blocks, which
  Qwen's dense FP8 projections and GLM's FP8 parts both launch.

Still the arms' own, on purpose: DeepSeek's `mxfp8_gemv_launch` family (MX scales, its own
kernels, outputs that must not move) and the arms' skinny and row GEMVs below the tile.

The gate is bit-neutral: Inkling's batch digests and every lanes-gate answer equal jr4's; GLM's
lanes-gate and interleave answers equal step 4's; Dory's single-process server gives the step-4
build's text and logprobs at prompts of 12, 700 and 3,000 words.

### Step 6: the scan runs the plan

The boot check already builds an `EnginePlan` per rank and resolves every part's kernel; until
now the arms then launched their scans from a second copy of the same facts. `ScanPlan::scheduled`
turns a scan requirement into the launch the arms run (heads, head widths, the column split of
`gdn_scan_shape_for`, the resolved kernel), and `ScanPlan::for_architecture` builds it for a
rank from the architecture its pack planned, refusing a plan whose scan layers differ. Qwen's
prompt walk launches its GDN scan from that plan (the plan's symbol, grid and block) and GLM's
bulk prefill its KDA scan; each arm checks the plan against its own loaded shapes and refuses a
mismatch at load, so the plan and the weights cannot drift apart silently. Heads no prompt scan
body takes (a test fixture's narrow ones) keep the token kernel: the boot launcher names the
resolver's kernel, and a prompt walk that reaches the scan refuses with the plan's reason rather
than launch a body built for other widths. The boot launcher
(`LinearScanLauncher`) builds the same plan, so the resolver, the boot path and the serving walk
name one kernel.

The scan was the one part whose plan already held everything its launch needs. Step 12 added
grouped-query attention: the plan now carries the most tokens a query attends
(`Indexer::max_selected`, 2,051 for Qwen3.8's 2,048 budget over pools of four), and
`GqaBody::choose` is the one rule for which grouped body runs (four or six query heads a block,
or the per-head `q4e_attn`) and with what grid and shared memory. The boot plan and Qwen's walk
both take it; the walk adds only whether the device granted the body's shared memory at load.
Every body computes each head as the per-head kernel does, so the choice moves no bits.

Still launched from the arms' own geometry, because the plan lacks what they need: MLA and
sparse MLA (cache layouts, split-KV cuts), the routed experts (work lists; the grouped threshold
is FP8-only and profile overrides are not read by the plan), the indexer and the residual mixers.
Each needs new descriptor fields first, each with a gate.

### The ledger

`sparkflash_runtime::Ledger` says, for one arm, how each runtime strategy runs: lanes, prompt
parts, drafting, joint rounds, multi-token runs, prefix reuse and the group. Each is `native`
(the arm's walk takes the strategy's shape), `looped` (the runtime loops a smaller shape) or
`off`, with a reason. The engine fills what it decides (the lane count; prompt parts off under
`SPARKFLASH_PROMPT_PARTS=off`; the group), the arm the rest through `FusedArm::ledger`; a
strategy nobody states prints as `off (the arm does not state it)`, so a missing statement shows
rather than reads as support. The line is printed once when the engine starts:

```text
sparkflash serve: ledger Inkling: lanes native (4 slots); prompt parts native (chunks of up to
4096 tokens); drafting native (DSpark, up to 7 drafts); joint rounds native; multi-token runs
native; prefix reuse native; group native (inkling: rank 0 decides and broadcasts, the workers
replay)
```

