# Milestone 3: the base at one, two and four ranks, other stored formats, and the part seam

Status: work breakdown, 2026-11-23. It details milestone 3 of
[05-runtime-models.md](05-runtime-models.md) (section 5.2, steps 3 to 5). After
[milestone 2](06-base-reads-description.md) DeepSeek's forward opens from what the scan
resolved. This milestone makes that engine able to run something other than DeepSeek: at a
different rank count, with weights stored another way, and with a part it has no body for. It is
proved on small synthetic checkpoints. No real model moves here.

Line numbers are the tree's at the start of this milestone.

## 1. What the tree says today

### 1.1 Rank count

* **The real DeepSeek cannot run at one or two ranks.** A rank holds 72 GiB of weights at four
  ranks, so it would hold about 145 GiB at two and 290 GiB at one, and a node has 128 GB. One
  and two ranks are proved on the synthetic shape from milestone 2, and the first real model at
  one rank is Qwen.
* **The plan knows the world; the forward does not ask it.** `plan::vocab_per_rank()` and
  `plan::heads_per_rank()` divide by a literal 4 (`plan.rs:805`, `:918`) and feed the tail, the
  rows path, the drafter, token routing and the vision tower's rank. The expert width is divided
  by 4 in four more places (`resident_weights.rs:1484`, `tp4_forward.rs:4972`,
  `tp4_combine.rs:61`, `draft.rs:94`). The inventory test's four-rank list counts three sites
  and misses all of these.
* **Much is already generic.** The weight split is computed from the world (`shard.rs`,
  `ResidentPlan`), the layout says only rows, columns or whole, converted files are named by
  rank and world, the attention output split divides by the world, and the Engram shard, the
  rank sum, the argmax pick and the exchange all take the world at run time.
* **One rank has no path.** The device layer refuses world 1 because it has no communicator
  (`tp4_forward.rs:11674`), the chain refuses a world under 2 (`tp4_chain.rs:282`), the loop
  needs a transport even with nothing to exchange, and `scripts/one-host-group.sh` has no arm
  for this engine. The other arms already have the pieces: `sparkflash_weights::tp::SingleRank`
  is the one-rank collective and `open_pack_collective` opens the group for any world.
* **Three kernel families fix the rank count.** The head sources assert four ranks and the
  reduction takes four pointer pairs (`cert_lm_head_sm121.cu:216`,
  `lm_head_reduce_sm121.cu:209`), with a four-rank table in the Rust launcher. The routed-expert
  kernels pad a rank's columns to a compiled 640, and `b12x_mxfp4_moe`, its stored form and
  `b12x_mxfp8_wo_a` are not on the instance list at all. Every other per-rank shape is already an
  instance argument and needs only more instances.
* **A model fact about ranks is in Rust.** DeepSeek's deployment states `worlds: &[4]`
  (`sparkflash-model/src/arms.rs:340`).

### 1.2 Stored formats

* **Format is three things, not one.** Plan 05 says the decode step is a template argument of
  the same loop. That holds for the decode. Two other things change with the format:
  * *layout*: DeepSeek's dense weights are re-tiled at load into 32 by 32 tiles; every other
    model's are row-major with scales on a 128 grid;
  * *scale kind*: DeepSeek's scales are powers of two and multiply in exactly; an FP8 tile scale
    is not, and has to be applied to a partial sum, which changes the order of accumulation;
  * *activations*: DeepSeek's fastest loops multiply FP8 activations by FP8 or FP4 weights with a
    block-scaled instruction, so its seams end by quantising the row. That loop cannot take BF16
    or NVFP4 weights. The others multiply BF16 activations, or quantise per 128 columns.
* **So an operation's body is chosen by its activations, and the weight decode is the
  argument.** Two bodies in the tree are already written this way: `opt_mma_bf16_body`
  (`opt_mma_bf16_sm121.cuh:76`, a weight mode for tiled FP8, BF16, F32, row-major and row-scaled
  FP8) and `shared_prefill_mma_rows_body` (`shared_prefill_mma_sm121.cuh:512`, the decode as a
  functor, used by GLM, Qwen and Inkling). They are the two candidates for the one copy.
* **The description loses the format on the way to the engine.** `Precision`
  (`sparkflash-ops/src/plan.rs:83`) has no MXFP4 and no block size, and folds MXFP4 into NVFP4.
  The catalogue's key has no format, and DeepSeek's precision masks were filled mechanically and
  are wrong. The instance list has no format field. The forward never reads the stored format:
  it checks dtype strings and tile byte counts (`tp4_operands.rs`, `tp4_forward.rs:619`), and its
  bind has no place for a tensor-wide scale.
* **The decodes the five other models need:** FP8 on 128 tiles (GLM, Qwen3.6, Qwen3.8 FP8),
  BF16 row-major (Dory, most of Inkling and Qwen3.8, GLM's KDA), NVFP4 (Qwen3.8 and Inkling
  experts), packed BF16, and row-scaled MXFP8 (Inkling's last layers).
* **There are two kernel builds.** DeepSeek's cubins are built from the instance list and loaded
  from a directory; the thirteen other images are compiled with no instance arguments and
  embedded.

### 1.3 The part seam

* **The contract has no device in it.** `PartExec` (`sparkflash-runtime/src/part.rs:72`) is
  state, split, a rows body and a token body, over a pass type the host defines. The host owns
  the residual, the streams and the rank sum.
* **Only the dense arm's parts are self-contained.** They take a pass context
  (`sparkflash-model-dense/src/parts.rs:111`, built on `sparkflash-ops/src/roles.rs`): normed
  rows in, a delta out, each lane's rows with pointers to the part's own state regions, and a
  launcher. GLM's, Qwen's, Inkling's and Dory's parts are thin calls into their own arm's rank
  object. Plan 05 says those parts plug into the base unchanged. They cannot: each has to be
  rewritten against the shared pass context when its model moves. That work belongs to
  milestones 4 and 5; this milestone builds what they plug into.
* **The served path has no part boundaries.** Milestone 2 put the part guards on the
  single-position path only (`tp4_forward.rs:14732`). The rows path
  (`tp4_forward/rows.rs:1088`) runs four fixed halves: the attention seam, the rest of
  attention, the FFN seam, the rest of the FFN. The residual is usually not written between
  them: the attention's result is folded into the FFN seam, and the FFN's into the next layer's
  seam.
* **The base has no recurrent state.** Everything a lane keeps is addressed by position, so a
  rejected guess is a counter move, a prefix can be adopted at any boundary, old rows slide
  away, and early prompt rows skip the decoder layers. A recurrence (GLM's KDA, a GDN, a short
  convolution) needs a saved point at the prompt's end, a snapshot for each verify row, a commit
  when rows are kept, and none of those shortcuts.
* **The sum differs.** The contract's summed split is an f32 sum in rank order. The base sums
  on a BF16 wire, off the main stream, and has no path for a part every rank computes whole.
* **The gate exists.** `sparkflash_runtime::conformance!` checks whole against chunked prompts,
  one lane against four, continued and rewound against cold, one, two and four ranks, the bit
  record, and drafted against plain. Five arms pass it. The base does not implement it.

## 2. The rules

1. **Nothing changes for the released DeepSeek.** Every step is held to milestone 2's proof
   (`scripts/open-manifest-ab.py`: manifest, launch trace, tick trace, speed). What is new is
   chosen at open and costs the released shape nothing at run time. A step that costs DeepSeek
   speed and cannot be recovered is reverted and reported.
2. **A rank count is a shape.** One rank's shapes come from the plan at that world. The forward
   contains no rank count, and a world the kernels were not built for is a row the scan shows.
3. **A stored format is a fact of the tensor.** It travels from the layout to the catalogue key
   to the bind without loss, and the body is chosen by it. The forward compares no dtype string.
4. **No third copy.** A format is taken by giving an existing body its decode, or by launching
   the body the tree already has for it. A new body needs a reason written down.
5. **A foreign part is a self-contained part.** It sees the shared pass context and nothing of
   the engine. The base decides at open, block by block, whether a block runs its fused native
   halves or lands the residual and calls parts.

## 3. Proof

* **The test bed is milestone 2's synthetic checkpoint**, opened through the same chain on one
  host. This milestone adds variants of it: other worlds, other stored formats, a foreign part.
  Each variant is compared with the host reference at the same world, on a few prompts at
  temperature 0.
* **The conformance suite runs on the base.** It implements `ArmFixture` over the synthetic
  checkpoint; the milestone closes when every check passes.
* **The released shape**: one comparison window on the four-node group for each of the three
  lines of work, batched at its end.

## 4. The work

### R. Ranks

* **R1. One rank's shapes from the plan (Rust).** `vocab_per_rank`, `heads_per_rank` and the
  expert width read the plan's split at its world. The four-rank list in the inventory test is
  widened to count what section 1.1 names, then drained. The deployment's list of worlds goes:
  a model runs at the worlds its scan resolves.
* **R2. One rank runs (Rust).** At world 1 the engine opens no communicator and no transport,
  and every reduce, gather and exchange is the identity through the shared one-rank collective.
  The chain and the device layer stop refusing it. `scripts/one-host-group.sh` gets this engine
  at one, two and four ranks.
* **R3. Kernels (parallel with R1 and R2).** The head and its reduction take the rank count as
  an instance argument; the routed-expert padding and the three sources not on the list become
  instances; the Rust launch tables follow the instance entry. The released instances keep
  their symbols and are proved by the kernel ladder (hash, then disassembly, then outputs).
* Done when the synthetic checkpoint agrees with the host reference at one, two and four ranks.

**Result (2026-11-23, R1 and R2).** The plan's world is now the split's authority: the vocabulary
shard, the attention heads and the routed-expert width all divide by the world the plan resolved
at, and the deployment no longer states a list of worlds. The forward's transport slot is either
the group's TCP transport or the one-rank identity (the shared `SingleRank` contract, no
communicator opened); the host arm's boot gate no longer demands device kernels for a run that
launches none, so an unbuilt world is a device-scan row rather than a host refusal. The synthetic
checkpoint agrees with the host reference at one, two and four ranks (`scripts/second-shape.py
run --world 1,2,4`). The device arm at one and two ranks still needs R3's rank-count instances;
its scan shows those rows.

### F. Formats

* **F1. The format arrives intact (`sparkflash-model`, `sparkflash-ops`, the instance list).**
  One format type from layout to bind, with block size, scale kind and layout; `Precision` stops
  folding. The catalogue is keyed by part, image and format, and DeepSeek's masks are corrected
  from what each kernel reads. An instance states the format it decodes. The scan's row shows
  the format and says "not built" for a format with no body.
* **F2. The forward chooses by it (forward crate).** The dtype and byte-count checks become the
  plan's format; the bind carries what a format needs (grid scales, row scales, tensor-wide
  scales); a seam ends in the activation form the next body reads, chosen at open. The loader
  arranges a tensor into the layout its resolved body reads, in one place.
* **F3. The recipe, with FP8 tiles as the worked example.** FP8 on 128 tiles, for the dense
  projections, the shared expert and the routed experts, on prompt rows and token rows, with
  both scale types (f32 and BF16) and both activation forms the tree has for it (BF16 rows, and
  rows quantised per 128 columns). Which of the two existing bodies becomes the one copy for each
  operation is decided here and written into this document with the measurement, and the steps
  are written down for the families below.
* **F4. The other decodes, in parallel once F3 has merged.** All of them are in this
  milestone; none waits for its model.
  * *BF16 row-major and packed BF16.* Packed carries its escape list through the bind and the
    loader packs in the one place F2 made.
  * *NVFP4.* Values, a scale per 16 and the tensor-wide scales (two for Qwen; the per-expert
    amax and the outer denominator for Inkling) travel through the bind. Today's NVFP4 body is
    separate because the tensor-wide scale rounds differently when folded in early: either it
    becomes a decode of the shared body with the scale applied where it is applied today, or
    the reason it cannot is written down with the measurement.
  * *Row-scaled MXFP8.* The power-of-two scale per row and 32 columns; the base already decodes
    this for Engram and the drafter head.
* **Format varies by block.** Inkling's experts are NVFP4 in its first forty layers and
  row-scaled MXFP8 in the last eight. The body is chosen for each block's part, not once for
  the stack.
* **One build.** The other images move onto the instance list as their bodies are used by the
  base. Not a goal of its own.
* Done when the synthetic checkpoint agrees with the host reference stored each way: FP8 tiles
  with each scale type, BF16, packed BF16, NVFP4 with each set of tensor-wide scales, row-scaled
  MXFP8, and one variant whose expert format changes part-way up the stack.

### S. The seam

* **S1. Blocks become steps, decided at open (rows path).** For each block the engine builds a
  step list from its parts. A block whose parts are the ones the fused halves cover gets exactly
  today's four halves: same launches, same trace. Any other block lands the residual and runs
  its parts one at a time. An unknown part is refused at open by name.
* **S2. A foreign part runs.** The base implements the host side of the shared pass context:
  land the pending result, norm the rows, hand the part its rows by lane, take its delta, sum it
  across ranks when the part is summed (through the base's own reduce) or not at all when every
  rank computes it whole, and fold it back. Proved with the dense arm's attention and MLP parts
  inside a DeepSeek-shaped synthetic stack. A stack with a foreign part runs prompt chunks one
  at a time until the part has a body for interleaved chunks; the scan says so.
* **S3. Recurrent state in a lane.** A part's regions are carved into the lane's memory at open.
  For a lane that has a recurrent region the engine saves at the prompt's end, snapshots for
  each verify row, commits what is kept, adopts a prefix only at a saved point and does not
  skip decoder layers for early rows. A stack with no recurrent region keeps every shortcut
  untouched. Proved with the dense arm's GDN part.
* **S4. The suite.** The base implements `ArmFixture` and passes `conformance!`.
* Left for later: a frame that carries the pending result across a foreign boundary, so mixed
  stacks keep the fusion. It is a speed step for the models that move, and is measured then.

## 5. Order

1. The close of milestone 2 lands first: everything here is tested on its synthetic checkpoint.
2. R1 with R3 beside it; F1 beside both (other crates).
3. R2 after R1. F2 after F1. S1 can start with R2: it is the rows path, which R1 barely
   touches.
4. F3 after F2, then the F4 families in parallel. S2, then S3, after S1.
5. S4 and the three comparison windows.

The forward crate has one owner at a time for each of R, F2 and S; they touch different files
but the same open path, so they merge often.

## 6. Out of scope

Moving any real model. Rewriting another arm's parts against the pass context. The serving
path and the weight path without model dispatch (plan 05, 5.3 and 5.4). Dory's ring schedule.
The four-serial-slice emulation in `moe_device.rs`: it is not the served path and is left as it
is.

## 7. Stopping

As in milestone 2: a step that cannot be made free for the released DeepSeek is reverted,
measured and brought to the owner.
