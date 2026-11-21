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
| `engine` | the tick engine: slots, the scheduler thread, prefix claim, park and least-recently-used reclaim, multi-token runs | the fused engine's, moved onto the runtime |
| `share` | the step wall models and the row share: each verify row to the draft likeliest accepted while it pays | DeepSeek's `share_rows`, `multi_step_ms`, `batch_step_ms` |
| `rounds` | one speculative round for several sequences: draft every chain, share rows, one walk, each sequence accepts | DeepSeek's `spec_multi_step`, generalized |
| `tick` (prompts) | the prompt chunks one tick runs (`tick_chunks`); the runtime executes them a chunk at a time between the running decoders' steps (`Runtime::run_part_rows`, `Part`); the arm keeps its own chunk sizing | DeepSeek's scheduler (`PREFILL_DECODE_INTERVAL`, `PREFILL_BYPASS_MAX_TOKENS`) |
| `group` | rank 0 decides and broadcasts, workers replay: lanes, prompt chunks, steps, give-backs, releases; an arm supplies only its wire (`GroupWire`: admission, sampler and image frames) | Inkling's group |
| `ledger` | per arm, which strategies run natively, looped or off (`Ledger`, `Strategy`, `Mode`); the engine states lanes, prompt parts and group, the runtime the rest from the walk (`Runtime::ledger`); printed at boot | new |

Device memory for lanes stays in `sparkflash_kernels::lane_memory` (it needs the driver); the
runtime only asks an arm's sequence whether it has room (`Runtime::room_slot`). An arm lays out
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
  the answer is already held (`Runtime::ready_slot`, no device work), and hands the whole run to the
  member's handle;
* the handle answers the following decode calls from the run while the caller feeds the token the
  run expects;
* when the caller stops early (a stop string, the token limit, a forced token), the handle tells
  the arm how many outputs went unused before its next job (`Runtime::give_back_slot`), and the arm
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
   slot whose prompt does not extend its history, so an arm
   names its history and nothing more. An arm that commits a verify at once keeps no row past
   the reply's end (`sparkflash_model::kept_to_end`). Otherwise a drafted end of turn leaves its
   sequence one token past the history, and an arm that cannot rewind loses the conversation.
9. A rewind point on every recurrent arm: GLM and Qwen keep their recurrent state at the end of
   each prompt, as Inkling does, so a turn whose client re-renders the reply (reasoning dropped)
   goes back to the prompt instead of starting cold.
10. Lane memory for Qwen: its lanes allocate their caches for the whole context at open; on the
    shared `LanePool`, as GLM's and Inkling's are, kept conversations share one budget.
11. Dory on the fused engine: its recurrence group keeps its own lanes, so it had no prompt
    parts, ledger or kept conversations of the runtime's. Done as Dory on the session contract
    with its own ring scheduling: the ring keeps a batch on every rank at once, which one fused
    walk a step cannot, so the arm keeps its scheduler and takes the contract's kept
    conversations, `resume` and ledger. The single-process device arm is on `Walk` now
    (`sparkflash-model-dory`'s `device/walk.rs`): `DoryWalk` runs a prompt chunk as one bulk pass
    and a step as one row a lane, states its caches as positional regions, and its rewind point
    is one id short of the prompt's end for the template's reason above. The row buffers are
    shared by the lanes, so a pass over several lanes is the arm's own batched body. A prompt
    row's bits must not depend on where the runtime cut the chunks, since reuse forwards a delta
    as its own pass, so the walk keeps two of the arm's kernel rules: a per-loop pass of eight
    rows or fewer runs `PREFILL_Q` rows with its own tokens repeated (the shift the arm's prompt
    loop applied to a narrow last chunk), and a pass that starts off a `PREFILL_Q` boundary
    re-runs the rows of its first query tile from the ids the sequence holds, because a prompt
    attention block's key window is its first row's. Loop-1 drafting stayed on the recurrence
    group: a single process has no second exit to draft from.
12. More typed plans executed (step 6): attention, MoE, the indexer and the residual mixers.
    Done for grouped-query attention (`GqaBody`); the rest need descriptor fields first.
13. The dispatcher's remaining rules (step 5): Qwen's split-K tile and the FP8 projections.
    Done: `split_k_slices`/`SplitTile` and `fp8_tile_fits`/`Fp8Tile`.
14. DeepSeek on the contract, optional: its kernels (tensor-core decode, Engram reads, the RDMA
    exchange) stay its own, its outputs must not change, and the contract would give it the
    ledger and the shared prompt parts. Done as the ledger: its batch scheduler states its
    strategies on the runtime's `Ledger` at start; its scheduler, kernels and arithmetic are
    unchanged.

### Then: the next model starts from the shared parts

Steps 8-14 gave every arm the strategies. Two things still make a new family write its own crate:
its tensor names exist only as probes in the command line's host reference (`cli/src/dynamic.rs`),
which the plan, the converter and the web cannot see, and the launch geometry of routed experts,
the indexer and the residual mixers is still each arm's own. The next steps close both, sharing
first what two arms already do alike; choosing a different body (token vs grouped experts) changes
bits and stays each arm's until a gate says otherwise.

15. Generic tensor naming: `ArchitectureTensorPlan` names any family `Architecture` describes
    (HF-canonical names, shapes, rank splits and roles, in the naming dialect the checkpoint
    uses), so the host reference, the converter and the web read one plan. No device code.
16. The indexer's launch: Qwen and GLM run the same three bodies (pool, score, pick) with the same
    budget rule, and `SparseIndexerLauncher` already reproduces both arms' grids. Both launch
    from it; one `MAX_SELECTED_TOKENS`. A grid move, no arithmetic.
17. The routed experts' shared geometry: the grouped-MMA tile and item count (Qwen's and
    Inkling's are the same constants), the `moe_plan` launch (GLM, Qwen, Inkling) and the
    combine grid (GLM, Qwen) come from `sparkflash_ops`. Which body runs stays the arm's.
18. Typed token-kernel calls: the bodies the epilogue plans (`NormPlan`, `RopePlan`,
    `ActivationPlan`) resolve to, the embed, residual add and KV append get `call`s with the
    argument layouts Qwen and the block drafter (`dflash`) spell today, and both launch through
    them.
19. A shared device arm for new families: dense BF16 grouped-query attention, one rank, on the
    shared plan, uploader and calls, served through `EngineBuilder` when the resolver finds every
    kernel, with the per-layer parity report against the host reference as its gate.
20. That arm on the serving contract (the runtime's lanes, lane memory, a rewind point, the ledger),
    gated by `scripts/served-gate.py`.

### Then: the shared arm takes the parts the released arms run

Steps 19-20 give a dense family the device and the serving strategies with no code of its own.
Most new families route experts, ship an FP8 release beside the BF16 one, or mix linear-attention
layers in, and each of those is already a kernel some arm runs. The next steps put those kernels
behind the shared arm's plan, so the families that need them get them too; each is gated by the
per-layer parity report against the host reference and then `scripts/served-gate.py`.

21. Routed experts on the shared arm: an MoE layer (a softmax, sigmoid or sqrt-softplus score, the
    top-k with or without renormalising, a routed scale, shared experts with or without their
    sigmoid gate) runs the one router body (`q4e_route`), the shared plan's run walk
    (`MoePlanCall::runs`), the token path's expert passes and the combine. The experts are BF16, a
    third format of the one expert body beside NVFP4 and FP8, whose launch rule (`moe::token`) Qwen
    now takes from the same place.
22. FP8 on the shared arm: a release whose projections are 128x128-tile E4M3 takes the shared FP8
    tile (`Fp8Tile`) for its prompt and the FP8 row GEMV for its tokens, and its experts take the
    expert body's FP8 form.
23. GDN layers on the shared arm: a Gated DeltaNet layer runs the shared scan plan (`ScanPlan`)
    and the Qwen image's scan bodies, with its convolution and recurrent state on the lane and in
    the rewind point.

24. The shared arm in a tensor-parallel group: a `--tp4-rank` group of a dynamically resolved
    family opens each rank's share of every tensor as the shared `TpPlan` splits it, sums each
    branch's partial rows in rank order through one shared helper (`sparkflash_ops::tp::RowSums`,
    the host seam or the one-shot and two-shot device legs every arm's reduce takes) and runs rank
    0's lanes on every rank through the runtime's group.
25. GLM's router and SwiGLU on the shared arm: the noaux_tc selection bias (top-k on score plus
    `e_score_correction_bias`, weighted by the plain scores) runs the one router body's bias form
    that GLM's router runs, and a `swiglu_limit` clamps every MLP of the layer (dense, routed and
    shared) in the clamped forms of the SwiGLU and the expert gate/up pass. The host reference
    takes both, so it no longer routes or activates such a layer as if neither were there.

Left after these: the residual mixers (one arm per rule) and MLA (GLM's alone: its kernels are
NoPE and run only over the sparse indexer's picks, so they cannot serve a standard MLA layer).

### Then: one copy of each shared part

Steps 1-25 put each strategy in one place; some parts still had a copy per arm. Each of these
moves a part that two or more arms ran into its shared home, so an arm that lacked it gains it,
and none moves a bit.

26. The packed BF16 form for every BF16 arm: Inkling's lossless packed decode weights (about 77%
    of the BF16 bytes, the same bits as `q4e_gemv_bf16` / `q4e_gemv4_bf16`) live in
    `sparkflash_ops::packed` (`PackedSet`, the typed `gemv` / `gemv4` calls, one switch
    `SPARKFLASH_DENSE_WEIGHTS=packed|checkpoint`). Inkling packs in place; Dory and the shared dense arm keep
    their BF16 rows beside the packed copy for the wide tiles they also run.
27. One fused-engine holder: the arms' local-or-group engine enums become
    `sparkflash_runtime::Fusion` (`local`, `group`, `open`).
28. One row-sum collective: Qwen's collective runs on `sparkflash_ops::tp::RowSums` (the one-shot
    gather its inject folds, the two-shot and host legs), and GLM takes the shared host leg
    (`host_sum`) and two-shot rule (`takes_two_shots`). The kernel tests share one copy of their
    device uploads and read-backs (`sparkflash_kernels::test_io`), and the hashes and mixers their
    shared copies in `sparkflash_weights`.
31. One serve entry for any walk arm: a backend states what the runtime serves it by
    (`sparkflash_runtime::WalkArm`: its walk, its group wire, the place it refuses and the engine
    its group runs on), and the runtime starts it alone, as rank 0 of a group, as a worker's
    mirror, and reads its lane width (`serve`, `serve_group`, `mirror`, `lane_limit`). GLM, Qwen,
    the shared dense arm and Inkling lose their own copies of that glue and their serving enums;
    the dense group and Inkling's image group stay on the fused engine behind the same entry.
    The registration gains what the hand-written serving sites repeated (dialect, group defaults,
    converter, console row, crate and switches), and the CLI's codec, response engine, group
    defaults, converter dispatch and deployment rows read it. The CLI line gate falls with it.
32. The fused engine retired: the tick group's wire already carries a prompt chunk's rows
    (`LaneRows`), and a prompt whose start encodes images now sends its request frame first and
    the start after it (`TickGroup::begin`, `TickGroup::started`, `LaneStart`). The dense
    arm's group and Inkling's image group serve on the tick group; `FusedEngine`,
    `FusedSession`, `FusedArm`, `GroupLanes`, `Fusion` and `GroupEngine` are gone, and the
    arms' `group_engine` with them. No bits move: a chunk boundary is the arm's
    `ChunkPolicy` chunk on either engine and each lane sees the same rows in the same order.
    The group halves owe one multi-rank run.

## Status

| step | state |
| --- | --- |
| 1. parallel drafts, Inkling | done: joint rounds, on by default (`SPARKFLASH_JOINT_ROUNDS=off` opts out) |
| 2. generic group mirror | done: `GroupWire` and the tick group (`TickGroup`, `mirror_tick_lanes`); Inkling's device group and GLM's, Qwen's and dense's `--tp4-rank` groups serve through it |
| 3. Qwen and GLM on the contract | done: both arms' fused lanes draft and verify through `RoundArm`; GLM's and Qwen's groups step four lanes by default, GLM's on Inkling's lane memory |
| 4. prefill scheduler | done: the tick engine decides each prompt chunk's rows (`tick_chunks`) and the runtime runs them between the running decoders' steps; Inkling, GLM, Qwen and dense all chunk their prompts |
| 5. projection dispatcher | done: BF16 projections (GLM, Dory, Inkling) choose and launch the wide tile through `sparkflash_ops::projection`; since step 13 Qwen's split-K tile and the FP8 tile (Qwen, GLM) too; DeepSeek's MX family stays its own |
| 6. typed plans executed | in part: Qwen's prompt GDN scan, the shared dense arm's GDN layers (step 23) and GLM's bulk KDA scan launch the kernel and grid of the rank plan's `ScanPlan`; since step 12 grouped-query attention its `GqaBody`, since step 16 the indexer its `IndexerPass`, since step 17 the routed experts' plan and tile their shared rule; MLA and the residual mixers still launch from the arms' own load-time geometry |
| 7. ledger | done: every engine prints `ledger <arm>: ...` at start; Inkling, GLM and Qwen state their strategies |
| 8. prefix reuse on every lane | done: GLM's and Qwen's lanes name their history; the engine renews a slot whose prompt does not extend it (`Runtime::room_slot`); a verify keeps no row past the end of turn (`kept_to_end`); the thinking-off close is the codec's (`PromptCodec::close_thinking`). The host's reply text holds a character a token left unfinished until the next completes it, as tokenizers' `DecodeStream` does, so a reply re-encodes to its ids (a stray U+FFFD had broken a continued turn). Served `turns`: Qwen 10 of 10 continued turns cached past their first prompt; GLM 38 of 40 over four runs (was 1 of 10): one first turn cut at the token limit, whose drafted rows GLM cannot rewind (item 9), and one the host's log did not cover. Lanes greedy and sampled equal the last gates on both, Inkling's equal jr4's |
| 9. a rewind point on every recurrent arm | done (deployed as p14, 2026-11-17): one shared `Checkpoint` (`sparkflash_kernels::checkpoint`) copies an arm's listed regions at each prompt's end and back at a rewind; `sparkflash_runtime::resume` picks extend or rewind; `continues` and the host's claim take the point even when the slot names no history, and the host respells a next turn that merges the point's last id (Qwen's `<think>\n\n` after a held `<think>`, `\n`). Inkling's boundary moved onto it; GLM keeps its KDA layers' convolution and recurrent memory and the MTP layer's hidden rows, Qwen its GDN and PLE state, the MTP stash and the n-gram windows; under a block drafter (Qwen's served DFlash 2, GLM's) the point also keeps the drafter's context rings whole (84 MB a lane for Qwen's), since a ring slot is a position modulo the ring and a reply longer than a block overwrites the window before the point. GLM's `give_back` cuts the history rather than failing, so a cut reply rewinds instead of renewing. One switch for every arm, `SPARKFLASH_REWIND` (`SPARKFLASH_INKLING_REWIND` is refused with the new name). Served gate `turns` (think-dropped every turn): Inkling and GLM (MTP and DFlash 2) REWOUND and equal to cold, GLM's cut first turn REWOUND; Qwen REWOUND+ (cached one short of the first prompt), its answers differ from cold because the model reads its own held ids for the merged newline; lanes gates equal their baselines |
| 10. lane memory for Qwen | done: one helper, `sparkflash_kernels::lane_memory::SequenceLanes` (plan, reserve, back, room, release), lays out and backs every arm's lane caches; GLM's and Inkling's moved onto it, and Qwen's attention caches now back as a sequence grows instead of whole at open. Every arm takes the one budget rule (`LanePool::budget`): one whole sequence's chunks, at least `FLOOR_LANES` lanes' first step, or `SPARKFLASH_LANE_MEMORY_MB`. For Qwen at a 64K context that is a 1152 MiB pool where four lanes held 4 x 960 MiB before, so long prompts on several lanes at once share one context's worth and a growth past the pool is refused with the request's error; set `SPARKFLASH_LANE_MEMORY_MB` for more. Served gate 18: Inkling, GLM and Qwen lanes greedy and sampled equal their baselines (Inkling's equal jr4's), `turns` PASS on all three |
| 11. Dory on the session contract | done: the recurrence group keeps its own ring scheduler (a batch on every rank at once, which one fused walk a step cannot) and takes the contract's parts: kept conversations (a parked lane holds its rows between requests), `sparkflash_runtime::resume` and the ledger (`ledger Dory: ...` at start). Every row is addressed by its position, so a lane goes back anywhere at no cost; its point sits one id before the prompt's end, because Dory's template opens a thinking reply with `<think>\n` and writes a past reply back as `<think></think>...` (and drops earlier turns' reasoning), so the next turn parts from the prompt at its last id. A repeated prompt therefore rewinds onto its own lane too; `scripts/served-gate.py turns` evicts the kept lanes before its cold reference. Served gates 20-21 on p17: lanes greedy and sampled equal p15's; `turns` PASS on every shape, plain turns HIT, thinking and think-dropped turns REWOUND+, every continued answer equal to cold |
| 12. more typed plans executed | in part: grouped-query attention runs the plan's rule. `sparkflash_ops::attention::GqaBody::choose` picks the grouped body (four or six query heads a block, or the per-head `q4e_attn`), its grid and its shared memory from the query groups and the most tokens a query attends (`Indexer::max_selected`, now in the plan's geometry); the boot plan and Qwen's walk take it, the walk adding only which bodies the device granted. Bit-neutral by construction (every body computes each head as `q4e_attn`); Qwen lanes greedy and sampled equal p15's and its kernel test passes on the device. MLA, MoE, the indexer and the residual mixers need descriptor fields first |
| 13. the dispatcher's remaining rules | done: Qwen's BF16 split-K tile (`split_k_slices`, `SplitTile`) and the FP8 tile (`fp8_tile_fits`, `Fp8Tile`, which Qwen's dense FP8 projections and GLM's FP8 parts launch) live in `sparkflash_ops::projection`, each the arm's old rule verbatim (a unit test pins the split against Qwen's old function). GLM and Qwen lanes greedy and sampled equal p15's. Qwen3.8 NVFP4 does not take the dense FP8 tile, so that arm's FP8 path is checked by the shared conditions and unit tests, not served; DeepSeek's MX family stays its own |
| 14. DeepSeek on the contract (optional) | done as the ledger: the batch scheduler prints `ledger DeepSeek: ...` at start (lanes and lane memory, prompt parts, DSpark depth policy, joint rounds and their floor, prefix reuse, the group); its scheduler, kernels and arithmetic are unchanged (a boot line only; the forward crate builds and passes clippy on the device). Its prompt parts stay its own: its scheduler already splits prompts (busy and bulk chunks, a decode share) on the tick rank 0 broadcasts, and moving them onto the engine's would change that tick |
| 15. generic tensor naming | done: `sparkflash_ops::naming` holds the HF-canonical spelling (`TensorNames`, `NamingDialect`) and probes which of its choices a checkpoint makes (the layer root, the post-attention norm, per-head norms, the router, a gated shared expert, a tied head); `ArchitectureTensorPlan` plans any family `Architecture` describes from it (`named`, or `served` in the canonical dialect), and refuses a family it does not spell (MLA, residual mixers). The host reference (`cli/src/dynamic.rs`, now a thin resolve) and the dynamic converter read the one plan; no device code changed |
| 16. the indexer's launch | done: `sparkflash_ops::indexer::IndexerPass` gives the pool, score, pick and GLM's run-score grids from one pass of rows; Qwen's and GLM's walks and `SparseIndexerLauncher` launch from it, with one `MAX_SELECTED_TOKENS` and `MAX_BUDGET_BLOCKS`. A unit test pins every grid to the arms' old rules; the arguments and bodies are unchanged |
| 17. routed experts' shared geometry | done: `sparkflash_ops::moe` holds the shared plan's call (`MoePlanCall`: the item plan, the token path's run plan, zeroed rows), the grouped W4A16 tile (`grouped`: item rows, stage, columns, `items`, `fits`, `rows_fit`, `grids`) and the combine grid. Qwen's prefill and token path, Inkling's three plan sites and GLM's bulk plan and combine launch from it; GLM keeps its own 128-row items. A unit test pins the argument lists and grids to the arms' old ones; the one argument that moved is the stride beside a null zero-row pointer, which the kernel does not read |
| 18. typed token-kernel calls | done: `sparkflash_ops::calls` builds one launch (`Call`: grid, block, arguments in the kernel's order) for the token image's shared bodies: `RmsNorm` (`q4e_rmsnorm`, and `q4e_rmsnorm_rows` with strides), `Rope`, `add`, `swiglu`, `copy`, `embed`/`embed_rows` and the BF16 cache appends. The block drafter launches through it (`GemvDevice::call`), as do Qwen's norm, rotary, cache appends, residual adds, SwiGLU, copies and embeddings, and GLM's, Inkling's and Dory's norms and embeddings. A unit test pins the lists to the arms' old ones; every argument and grid is the same (an elementwise grid that moved is the same size, and those bodies walk a grid stride) |
| 19. a shared device arm for new families | done: `sparkflash-model-dense` serves a family no pack names when it is a dense BF16 grouped-query decoder (`sparkflash_ops::parts::dense_device_refusal` names everything else), from the `Architecture`, the shared names, `DraftUploader` and the typed calls (new: `gemv_bf16`, `deinterleave`, `FullAttention`, which Qwen and Dory now launch through too); its prompt projections take the wide tile under `WidePolicy::DENSE`. It implements `EngineBuilder` and `serve --model` picks it for such a family (`SPARKFLASH_DENSE_DEVICE`: unset prefers it and names why the host answers instead, `1` requires it, `0` keeps the host). One deviation from the step's text: it gates on its own launch list (`dense_device_images`), not the kernel catalogue, whose `DenseMlp` and GQA rows pin the released presets' hidden widths and query groups (a 2048-wide Llama does not resolve); a catalogue row per shape rule would be a new `Part` identity each. Gate: the per-layer parity report against the host reference (`cli/src/serve/dense_parity.rs`, every layer's residual and the logits, two fixtures: a two-chunk Llama-shaped prompt on the wide tile; a gated, q/k-normed, half-rotated, tied checkpoint whose MLP takes the batched row GEMV), passed on the RTX laptop (sm_121): every residual within 5.2e-3 of the reference (plain prompt 300 rows over both layers, decode within 7.6e-4), logits within 6.1e-4 untied and 4.3e-2 tied (the unit-scale tied head magnifies the BF16 cache rounding; within the 2e-2 + 2e-2·|ref| bound), and the next token agreed at every step, the plain prompt's one near tie (margin 8.0e-2) included |
| 20. that arm on the serving contract | done: `DenseLanes` is the runtime's, so `serve --model` on a dense family starts the shared engine with four lanes by default (`--concurrency`, or the profile's `runtime.concurrency`; a mismatch is refused, the host reference refuses both lane knobs). A fused step walks one row a member through the same launches with the row GEMV (policy `None`), so a member's bits are its lone row's; prompts run in 256-row parts; each lane's keys and values grow in `SequenceLanes` memory (one sequence's worth by default, `runtime.lane_memory_mb` for more); the KV is position-addressed, so the rewind point is the prompt's length and going back is moving the position (`SPARKFLASH_REWIND`); the ledger prints `ledger dense: ...`. Device tests (`dense_parity.rs`): fused lanes at 5 to 300-row prompts give every step's tokens and logit bits as each alone, a kept lane continues (within 7.4e-6 of a cold forward) and goes back to its prompt. Served gate on the RTX laptop on a random-weight checkpoint of a family no pack names (`scripts/dense-synthetic-checkpoint.py`: 8 layers, hidden 1024, GQA 16/4, the Qwen3.6 tokenizer): lanes greedy and sampled equal alone (267 tok/s alone, 161-187 a lane at four), interleave equal alone with a 0.40 s longest pause while a 60K-character prompt filled, `turns` PASS (plain and thinking turns REWOUND and equal to cold; think-dropped REWOUND+, as on Qwen, the model reading its held ids for the merged newline) |
| 21. routed experts on the shared arm | done: an MoE layer of BF16 experts runs on `sparkflash-model-dense` from the parts the released arms run: `q4e_route` (one router kernel over the shared `shared_moe_router_body`, any score with or without renormalising and a routed scale; Qwen's `q4e_router` is the same body at a scale of one, its bits unchanged), `MoePlanCall::runs`, the token path's expert passes in a BF16 form of Qwen's one expert body (`q4e_expert_gate_up_bf16`/`_down_bf16`, rows of eight BF16 a lane; the NVFP4 and FP8 forms untouched) on the launch rule `sparkflash_ops::moe::token` that Qwen now takes from the same place, and `q4e_moe_combine`; the shared expert and its sigmoid gate run as a dense MLP. The router stays on the row GEMV at every row count, so a prompt row picks the experts its decode row would. `dense_device_refusal` names what it does not run (another expert format, a selection bias, a clamp, past 512 experts or 64 picks, widths not a multiple of eight, more than one shared expert, and one pick renormalised under an independent score, which the host reference divides to one where HF keeps it). Gate (`dense_parity.rs`, RTX laptop): three routed fixtures, a softmax router with a sigmoid-gated shared expert over a two-chunk prompt, a sigmoid router after a dense layer (scale 2.5, no renormalising, ungated shared expert, an expert width the staging pads), a sqrt-softplus router with no shared expert; no pick differed from the reference's: residuals within 6.1e-3, logits within 1.1e-2, the next token agreed at every step; fused routed lanes give each conversation's bits alone. Served (`served-gate.py all`, a synthetic checkpoint of 8 routed layers, 32 experts top-4 and a gated shared expert, from `dense-synthetic-checkpoint.py --experts`): four lanes together equal each alone, greedy and sampled; a decoding lane paused at most 0.42 s while a 60k-character prompt filled; every turn shape rewound and matched its cold run. |
| 22. FP8 projections and FP8 experts | done: a release whose projections are 128 x 128 block-scaled E4M3 (`quantization_config` fp8, BF16 `weight_scale_inv`, or F32 scales that narrow to BF16 without loss; others are refused) runs on `sparkflash-model-dense` from the released FP8 parts: the token path's tiled GEMV `q4e_gemv_tiled_e4m3_tile` (`gemv_e4m3_tile`, one scratch of partials and self-resetting counters), the prompt path's `q4p_dense_fp8` (`projection::Fp8Tile`) chosen by `projection::route_fp8`, the BF16 rule over the FP8 tile's own fit, and the FP8 forms of Qwen's expert passes (`q4e_expert_gate_up_fp8`/`_down_fp8`) for routed experts; activations stay f32 as the host reference keeps them. `DraftUploader::mat_e4m3_tile` uploads codes and scales for any arm. `dense_device_refusal` names what it does not run (another block, a hidden, attention or MLP width that is not a multiple of 128, FP8 experts narrower than 128). Gate (`dense_parity.rs`, RTX laptop): an FP8 dense fixture (BF16 and F32-word scales on alternate layers) over a two-chunk prompt, residuals within 2.1e-3, logits within 2.3e-4; decode rows within 4.9e-4; FP8 routed experts, residuals within 3.5e-3, logits within 3.9e-4; the next token agreed at every step; fused FP8 lanes give each conversation's bits alone. Served (`served-gate.py all`, the step 21 checkpoint with every projection and expert FP8, `dense-synthetic-checkpoint.py --experts --fp8`): four lanes together equal each alone, greedy and sampled; a decoding lane paused at most 0.43 s while a 60k-character prompt filled; every turn shape rewound and matched its cold run. |
| 23. Gated DeltaNet layers on the shared arm | done: a hybrid family's `linear_attn` layers run on `sparkflash-model-dense` from the scan image's chunk kernels Qwen's prompts run, now typed in `sparkflash_ops::gdn` (`GdnGeometry` with its prep and scan arguments, `conv_rows`, `conv_history`, `gated_norm`; Qwen's prompt path and GLM's bulk KDA path launch through the same calls, same arguments in the same order, so their bits do not move), with the scan kernel and grid of the rank plan's `ScanPlan`. Every row count takes the chunk path, decode rows included: the projections and the prep and gated norm take every row, the conv, its history and the scan each sequence's rows, so a row's state is its own alone, in a prompt or in a fused step. Each lane holds every GDN layer's conv history and delta-rule state as fixed lane parts, zeroed for a fresh sequence, and its rewind point is the shared `Checkpoint` (an arena in the lane) saved at each prompt's end and restored at a rewind. `dense_device_refusal` takes GDN layers of one geometry at a scan head pair ((64, 64), (128, 128), (128, 256)), 2..=9 taps, value heads a multiple of the key heads and a SiLU or sigmoid gate, and names the rest. Gate (`dense_parity.rs`, RTX laptop): a hybrid fixture (GDN 128/128 between gated attention layers) over a two-chunk prompt, the first GDN layer's residual within 9.3e-6 and every residual within 2.8e-3, logits within 2.6e-4; decode rows within 4.4e-4; a 64/64 sigmoid-gated, three-tap hybrid with FP8 projections and routed experts within 2.7e-3; the next token agreed at every step; fused GDN lanes give each conversation's bits alone; a kept GDN lane continues and rewinds to its prompt's state, each within 1e-5 of cold. Served (`served-gate.py all`, `dense-synthetic-checkpoint.py --experts --fp8 --gdn`: GDN on even layers, routed FP8 experts): four lanes together equal each alone, greedy and sampled; a decoding lane paused at most 0.23 s while a 60k-character prompt filled; every turn shape rewound and matched its cold run. |
| 24. the shared arm in a tensor-parallel group | done: `serve --model` with `--tp4-rank` on a family the shared dense arm takes opens each rank with `DenseDeviceBackend::open_rank`, which holds the rank's share of every tensor as `TpPlan` splits it: query heads, key/value heads (shared when the world exceeds them), both Gated DeltaNet head counts (the fused q/k/v as three row runs, the conv channels with them, `b`, `a`, `A_log`, `dt_bias` and `z` by value head, `out` by value column), every dense and shared-expert width, and the routed experts' width in whole runs of what their passes read (eight BF16 columns, a 128 FP8 tile; the remainder to the first ranks). FP8 shares keep whole scale tiles, and a last share inside one tile (a 64-row key/value head) takes that tile's scale (`DraftUploader::mat_e4m3_tile_parts`, `mat_parts`). The o and out projections and every MLP and expert down projection leave partial rows, summed in rank order from zero before each residual add by one shared helper, `sparkflash_ops::tp::RowSums` (`q4e_sum_slots`): the host seam, or the device legs every arm's reduce takes (one-shot below `SPARKFLASH_TWO_SHOT_ROWS`, two-shot above), all three the same bits on the f32 wire, so every rank holds the same residual, routes the same experts and samples the same token. Rank 0's lanes run on every rank through the tick group (`enable_group_fusion`, `mirror_group`, `TextWire`); workers never detokenize. The pack collective is the presets' (NCCL by default, `SPARK_COMM_REDUCE=tcp` for the host seam); `SPARKFLASH_DENSE_DEVICE=0`, `--tp4-device` and any family the arm refuses are refused at a world above one, since the host reference runs alone. Gate (`dense_parity.rs`, RTX laptop): an in-process group of two on the one GPU, rank 0 against the host reference and rank 1 to the bit as rank 0, on the host leg and on the device legs (met through the host, since NCCL takes one rank a GPU), the two legs to the bit as each other: plain Llama-shaped (residuals within 5.1e-3, logits 6.1e-4), one shared key/value head and a 520-wide MLP (2.7e-3, 3.9e-3), sigmoid-routed BF16 experts split 72 + 64 (6.1e-3, 1.1e-2), Gated DeltaNet (2.8e-3, 3.3e-4), FP8 Gated DeltaNet with FP8 routed experts (2.6e-3, 3.4e-4); the next token agreed at every step. Served (`served-gate.py all`, two `--tp4-rank` processes on the laptop over the host seam, the step 23 checkpoint): four lanes together equal each alone, greedy and sampled (101 tok/s alone, 280 tok/s four at once); a decoding lane paused at most 0.46 s while a 60k-character prompt filled; every turn shape rewound and matched its cold run. Served on the cluster over NCCL at four ranks with step 25's checkpoint (below) |
| 25. noaux_tc router and swiglu_limit | done: a routed layer whose `topk_method` is noaux_tc (one group, or groups that keep every group) takes its top-k on a sigmoid or sqrt-softplus score plus the f32 `e_score_correction_bias`, weighs the picks by the plain score, renormalises above one pick and applies the routed scale: on the device through `q4e_route`'s bias form, the bias instantiation of the one `shared_moe_router_body` GLM's router runs, and on the host through `sparkflash_weights::topk::noaux_top_k`. A `swiglu_limit` clamps every MLP of the layer, dense, routed and shared (`silu(min(gate, L)) * clamp(up, -L, L)`), on the device in `q4e_swiglu_clamped` and the clamped BF16 and FP8 forms of the expert gate/up pass (`q4e_expert_gate_up_bf16_clamped`/`_fp8_clamped`). Before this the host reference ignored both, so it routed and activated such a layer as if neither were there; it now takes them, and refuses by name what it does not run (group-limited selection, a bias over a softmax score, a clamp on another activation). The bias is planned (`FfnNames::Moe::bias`, F32 `[experts]`) and the shared expert is named in either spelling (`shared_expert`, DeepSeek's `shared_experts`). Gate (`dense_parity.rs`, cluster lead): a noaux fixture (dense first layer, sixteen experts top three, scale 1.8, a 0.75 limit so the clamp bites, `shared_experts`) over a two-chunk prompt, residuals within 5.5e-3, logits within 1.2e-3; FP8 experts and projections within 4.4e-3, logits 1.1e-3; two ranks to the bit as each other; fused lanes give each conversation's bits alone; the next token agreed at every step (four or five picks put one 300-row prompt row at a 2e-5 router near tie, so the fixture takes three). Served (`served-gate.py all`, `dense-synthetic-checkpoint.py --experts 32 --shared-intermediate 512 --noaux --fp8 --gdn`, a four-node `--tp4-rank` group over NCCL, which also serves step 24's NCCL legs for the first time): four lanes together equal each alone, greedy and sampled (183 tok/s alone, 485 tok/s four at once); a decoding lane paused at most 0.06 s while a 60k-character prompt filled; every turn shape rewound and matched its cold run. |
| 26. the packed BF16 form, shared | done: `sparkflash_ops::packed` holds Inkling's lossless packed decode weights (`PackedSet`, in place or beside the BF16 rows, typed `gemv`/`gemv4`, one switch `SPARKFLASH_DENSE_WEIGHTS=packed|checkpoint`; the old `SPARKFLASH_INKLING_DENSE` and `SPARKFLASH_BF16_FORM` are refused). Every pack is unpacked and compared word for word on the device before it is used. Dory and the shared dense arm now read packed copies on their row GEMVs (23% fewer bytes a pass) and keep the BF16 rows for the wide tiles. Gate (cluster lead): the Dory device test gives packed and BF16 the same bits at 1-16 rows; Dory served lanes, greedy and sampled, equal the p15 baseline and every turn shape passes, 34.8 vs 28.3 tok/s alone and 116 vs 95 tok/s four at once; dense TP4 lanes packed equal BF16, 219 vs 190 tok/s alone; `dense_parity` 28 pass. Qwen's and GLM's decode run tiled BF16 GEMVs, which the row walk's layout does not feed; step 29 gives them a tiled layout of the same form |
| 27. one fused-engine holder | done: the arms' local-or-group engine enums are `sparkflash_runtime::Fusion` (`local`, `group`, `open`) |
| 28. one row-sum collective and shared test copies | done: Qwen's collective is `sparkflash_ops::tp::RowSums` (its inject folds the one-shot slots that `RowSums::gather` fills; the two-shot and host legs are the shared ones), GLM takes the shared host leg (`host_sum`) and two-shot rule (`takes_two_shots`), the hashes and mixers are `sparkflash_weights`' one copy, and the kernel tests share `sparkflash_kernels::test_io` (`cuda_or_skip`, `upload`, `download`) in place of a copy a file. Gate (the end battery, cluster lead and four nodes): Qwen TP4 lanes greedy and sampled equal the baseline (61 tok/s alone, 94-98 four at once) and every turn shape passes, which runs the one-shot, two-shot and sequence-parallel legs; GLM TP4 lanes equal their baseline (45 tok/s alone) and turns pass; Inkling equals the deployed build and turns pass; Dory equals p15; dense TP4 lanes, interleave and turns pass; the touched crates' unit tests fail only where they failed before; the DeepSeek forward's clippy is clean |
| 29. the packed form on the tiled GEMVs | done: `packed::Layout` (`Walk`, `Tiled`; `PackedSet::tiled`) lays a step's planes A and B chunk by chunk, so a tiled reader's lane loads its own words (`q4e_packed_tiled_a`/`_b` in `bf16_packed_sm121.cuh`); plane C and the escapes are the walk's. GLM's `q4e_tiled_gemv` (`Q4eTgPacked`) and Qwen's `q4p_dense_rows` read it; the row-walk `gemv`/`gemv4` refuse a tiled matrix, and the tiled readers take no walk matrix. The values are the checkpoint's, so the bits do not move. One switch for every arm: `SPARKFLASH_DENSE_WEIGHTS=packed|checkpoint` (`packed` the default). Gate (cluster lead and four nodes): the GLM tiled device test and the Qwen packed rows test give packed and BF16 the same bits; GLM TP4 lanes equal the baseline packed and with the checkpoint's rows (45.2 vs 45.1 tok/s alone, 60.1-60.5 vs 57.3-60.2 four at once); Qwen TP4 lanes equal the baseline packed and with the checkpoint's rows (61.5 vs 53.4 tok/s alone, 94.8-96.0 vs 83.7-97.6 four at once, a noisy pair: step 28 measured 61 alone on the checkpoint's rows). GLM's packed tiled GEMV runs 4-10% under the BF16 one per call (6160 x 4096: 208.4 vs 220.4 us), though at 163-195 GB/s against the BF16 tile's 205-232. Beside the rows, packing costs GLM about 1.6 GB and Qwen about 1.6 GB a rank |
| 30. the direct-RDMA exchange for every group | done: `sparkflash_ops::doorbell::with_doorbell` opens the two-rail exchange for every pack model's TP group of two or more with a device leg (`open_pack_collective`: GLM, Qwen, Inkling and the shared dense arm), from the group's peers, `control_port + 1`, `NCCL_IB_GID_INDEX` (1) and `SPARK_COMM_RAIL_COUNT` (2); `SPARKFLASH_EXCHANGE=nccl` keeps NCCL. IPv6 link-local GIDs are taken, so both rails open. `tp4_reduce::DeviceExchange`'s only users are the pack paths (`serve.rs` and `glm_tp4.rs`); DeepSeek opens the same shared `sparkflash_kernels::write_exchange` through `rdma_exchange`, with its own row cap and cubin. Gate: GLM and Qwen TP4 lanes give the same bits on either exchange; GLM 45.2 vs 45.4 tok/s alone and 60.1-60.5 vs 58.5-59.3 four at once, Qwen 61.5 vs 60.7 alone and 94.8-96.0 vs 93.5-93.8 four at once (RDMA vs NCCL, up to 2%) |
| 31. one serve entry for any walk arm | done: `sparkflash_runtime::WalkArm` states the walk, the group wire and the place the backend refuses, and `serve`, `serve_group`, `mirror` and `lane_limit` are written once over them. GLM, Qwen, the shared dense arm and Inkling serve through it; the dense group and Inkling's image group moved onto the tick group in step 32. `sparkflash_model::Arm` now states the dialect, the group defaults, the converter, the console's deployment row and the crate and switches; the CLI's codec, response engine, converter dispatch and group defaults and the console's deployment rows read it. Gate: the runtime, model and CLI test suites and the generated conformance checks on one rank; a multi-rank group run is still owed for the group halves |

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
`Tp4Driver` one request at a time. Step 3 moved GLM's group onto the runtime's group, and then Qwen's
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
(`PackArm::open_lanes`, `open_glm_lanes`). Rank 0 serves its backend through the host driver
exactly as one node does (`serve_backend`). Every lane's prompt, step, give-back and release is
broadcast before it runs, and each worker replays them over its own lanes (`mirror_tick_lanes`).

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

- `TickGroup::in_step` wraps every step, prefill, give-back and release that follows a
  broadcast. On an error it tells the workers (`Error { code: "group_out_of_step" }`) and ends
  rank 0 (`end_group`).
- A worker mirrors until the group ends (`mirror_ticks_until_the_group_ends`), and watches rank 0's
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
the shared prefill scratch carries nothing between chunks. A group broadcasts each chunk's rows
(`LaneRows`) so every rank runs the same rows between the same steps. This was the fused
engine's schedule; the tick engine runs the same decisions, and `SPARKFLASH_PROMPT_PARTS=off`
is refused at start (its `on` is what this engine runs).

An arm that does not split its prompts runs the whole prompt as one chunk.

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
`off`, with a reason. The engine fills what it decides (the lane count and the group), the
runtime the rest through `Runtime::ledger`; a
strategy nobody states prints as `off (the arm does not state it)`, so a missing statement shows
rather than reads as support. The line is printed once when the engine starts:

```text
sparkflash serve: ledger Inkling: lanes native (4 slots); prompt parts native (chunks of up to
4096 tokens); drafting native (DSpark, up to 7 drafts); joint rounds native; multi-token runs
native; prefix reuse native; group native (inkling: rank 0 decides and broadcasts, the workers
replay)
```

