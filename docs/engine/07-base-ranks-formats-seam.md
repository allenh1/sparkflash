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
* **The arm that proves a variant is the device arm.** It is the served path: the bind, the
  loader's arrangement, the instance list and the launches. The engine's host arm
  (`SPARKFLASH_LAYER_DEVICE=0`) launches no kernel; a variant that agrees there shows the plan,
  the names and the split are right, and is half the proof. A step is done when the device arm
  agrees with the host reference. On one GPU that is the device arm at one rank, which needs
  R3; two and four ranks run on the four-node group with `scripts/second-shape.py run --device`.

### The reference is taken at the world's own split

The global `host-reference.json` is a single-process sum, and comparing every rank count with
it is valid only when the format's arithmetic is regrouping-invariant. It is not for row-scaled
MXFP8/E4M3: on the synthetic fixture's layer 0 the per-rank partial sums differ from the global
sum by 1.75e-10 absolute and 4.7e-7 relative, 3803 of the 4096 bits of the f32 result, and that
difference flips one greedy tie. World 1 therefore stays exact against the global reference,
which is the same single sum. World W is compared against the host arm's own output at world W
(`host-reference-w{W}.json`), never against the global file, and the device group at W is held
to that file exactly. Drift between a world's reference and the global one is printed with the
prompts that differ and the top-2 logprob gap at the first divergence rather than tolerated,
because a difference that moves a token is the failure the rule exists to attribute; token ids
are never compared with a tolerance.

### 3.1 Where the proof stands (review of `1ef7a6ca`, 2026-11-23)

Every result recorded below under R1, R2, F1, F2 and F3, and the second shape of milestone 2,
was measured on the host arm. No device kernel has run a second shape, a world other than
four, or a format other than the released ones. What that leaves open:

* **R3 is not started** (`cert_lm_head` and `lm_head_reduce` still assert four ranks; the
  instance list has no other rank count). It is the gate for every device proof on one GPU, so
  it goes first.
* **The second shape on the device arm.** Its cubins build; nothing has launched them.
* **F2's seam has no reader.** `operand::arrange` and `ScalePlanes::of` are called only by
  their own tests, and nothing reads an operand's `body` or `activation`. The loader, the bind
  and the launch must take them, or they are a second description of what the old code still
  decides.
* **F3 chose the body and did not launch it.** A tile-format part still resolves to no symbol
  of the base's image. The shared tile header needs rows on the instance list at the
  synthetic shape and a launch from the base's seam, prompt rows and token rows.
* **F4** follows the same bar: a family is done when the device arm reads the synthetic
  checkpoint stored its way, not when the host arm does.

**Result (2026-11-23, the device arm at two ranks).** The synthetic second-shape checkpoint
(2.8 GB, the released `mx-fp4` row-scaled expert format, hidden 4096, 64 experts, top-4,
intermediate 512, vocabulary 32,768) was opened on the device arm at two ranks on one host
(`scripts/second-shape.py run --device --world 2,4`, tip `536a8a5`, 36 second-shape cubins,
release binary `9a60bb02`). The group refused at open, before a launch:

```
cannot execute deepseek_v41 at TP2: moe on sm121 at MXFP4 (row scale + power-of-two grid,
one scale per 32 values, row-major) in the prefill pass: deepseek_ptx: intermediate 256 is
not one of the shapes the kernel was built for ([128, 576])
```

The two-rank split halves the 512-wide expert intermediate to 256, and the instance list has
rows at 128 and 576 only. World 4 was not attempted, because the harness stops at the first
disagreeing world. The host reference for the same checkpoint gave

```
[[6653, 20427, 12813, 422, 9389, 6388, 16395, 6405],
 [25425, 11040, 11559, 23442, 29453, 17940, 5192, 25888],
 [30169, 105, 2323, 11724, 1353, 885, 3013, 30683]]
```

which is a different prompt set from F3's three-prompt reference, so it is not comparable to
those ids. The measurement stands as it is: at two ranks the device arm cannot reach the
routed experts because their shape is not built, and R3's rank-count and padding instances are
what unblock it.

Three more windows, each on the same synthetic checkpoint and the same host, narrowed it:

* **The per-world instances build.** After the world-suffixed rows (134 cubins: 36 `_s2`,
  14 `_s2w2`, 14 `_s2w1`) the harness staged 14 cubins for two ranks and 36 for four, and the
  kernel ladder's hash step passed unchanged (`all kernels match the recorded baseline` over
  51 kernels): relaxing the rank asserts moved no released cubin's bytes.
* **The one-host group is the wrong vehicle.** The same run then refused before a forward with
  `SPARK_COMM_REDUCE=tcp is a host-arm exchange, but this run takes the device arm, whose
  per-layer collectives are NCCL enqueues`. `scripts/one-host-group.sh` pins that variable for
  the deepseek arm, so the device group is the one the deployed launcher starts,
  `scripts/serve-four-node.sh`, and `second-shape.py` now drives it when `SPARK_NODES` names the
  nodes. The launcher takes `SPARK_WORLD` (1, 2 or 4) rather than asserting four, and forwards
  `SPARKFLASH_UNPINNED_SECOND_SHAPE` to every rank, which a staged second-shape cubin needs.
* **The four-node group forms and the workers die.** At world 4 all four ranks started, the run
  header reported `reduce=nccl`, the device opened, and the three workers exited about two
  seconds into the plan builder with no error line - the last line is
  `layers.0.attn_norm.weight: planned as no stated format, the plan builder writes BF16` - after
  which rank 0 reported a closed control channel. At world 1, one rank on the head node, the
  arm refused cleanly at exit 2 with `deepseek_ptx is not loadable under runtime availability`
  for every part, so at one rank the DeepSeek image is judged unloadable and the reason lies
  earlier in the open. Neither is a shape refusal any more.

**Cause (2026-11-24, both refusals traced on the source, no window).** They were two separate
defects, and neither was in the kernels or the formats.

* **The four-rank workers exited in the plan builder.** The chain builds its rank plan
  (`setup_for_chain` → `load_entries`) before `plan::install`, and `load_entries` read each
  tensor's format from `crate::plan::installed()`, which was still empty. Every
  `HeaderEntry.format` was therefore `None` and the `mhc_seam` bind refused
  `layers.0.attn_norm.weight` by name, on all four ranks. The worker did not exit silently: the
  error propagated through the resident chain, the worker printed it and returned exit 2, and
  rank 0, seeing the control socket close, reported only the generic closed-channel line, which
  is what the window recorded. The header loader now opens the checkpoint's plan itself and
  resolves formats from it, so the order of install and load cannot change the answer
  (`tp4_resident.rs`, and a regression test in `tests/checkpoint_formats.rs` that fails with
  `format: None` on the old code).
* **The one-rank run judged the DeepSeek image unloadable.** The direct world-1 command passed
  `--tp4-device cubins-w1` but did not export `SPARKFLASH_CUBINS`, and the pre-boot gate asked
  `Availability::runtime_env()`, which reads only the environment, so every part reported
  `deepseek_ptx is not loadable under runtime availability` (`resolver.rs`, raised through
  `parts.rs`). No cubin was missing: fourteen `_s2w1.cubin` files were built and staged. World 4
  never showed this because `scripts/serve-four-node.sh` does export `SPARKFLASH_CUBINS`. The gate
  now takes its cubin directory from the flag the run was started with, and the launcher and the
  direct command answer alike (`serve/device.rs`, with a test that fails if the gate falls back
  to the environment).

Both fixes are on `r3/device-arm` and none of them has run on a GPU yet: the next window
confirms them. A worker's own log is written on the worker's node, so a group-level read of the
head node sees only rank 0 and hides the cause; the harness now collects every node's log, and a
window that reads only `rank0.log` is not a window.

**Result (2026-11-24, the device arm at one, two and four ranks, two windows).** Both windows
ran the synthetic second-shape checkpoint (2.8 GB; the released `mx-fp4` row-scaled expert
format; hidden 4096; 64 experts, top-4; intermediate 512; vocabulary 32,768) through
`scripts/second-shape.py run --device`, which drives `scripts/serve-four-node.sh`, on the
four-node group: worlds 1, 2 and 4, one at a time. Each world's reference is the host arm's own
output at that world. The host reference was identical to the global file at all three worlds
and gave

```
[[6653, 20427, 12813, 422, 9389, 6388, 16395, 6405],
 [25425, 11040, 11559, 23442, 29453, 17940, 5192, 25888],
 [30169, 105, 2323, 11724, 1353, 885, 3013, 30683]]
```

The first window did not reach a rank. The harness's copy step counted the command list instead
of the hosts it had collected, so the four-node run reported every rank local and skipped the
copy; it then handed `sync-nodes.py` no `--host` and died at world 1. With the distribution in
place, the workers had no work tree to `cd` into - the launcher makes every rank run from the
checkout - so no rank started. Both defects were in the harness and the operator's path, not in
the arm, and both are fixed (`b9583ff`, `335fd7c`); the operator brief now stages the binary to
each worker and checks its sha256 before a rank starts.

The second window built `97544a5c` (binary `7245ee9c`) and staged 72 second-shape cubins (39
`_s2`, 14 `_s2w2`, 14 `_s2w1`, 5 `nvfp4_prefill_sm121_*`), the same binary at the same absolute
path on every node. No rank reached a launch:

* **World 1: no verdict.** Rank 0 printed `TP=4 device rank 0/1: waiting for the workers` and
  then spun at full CPU for over two minutes without binding its port; the thread dumps are in
  `run-r3/w1-hang/`. The harness was in its ready wait, so the window killed it rather than
  wait out the timeout.
* **Worlds 2 and 4: refused at the loader's byte check.** Two ranks at each of those worlds
  printed `tp4-chain: layers.0.attn.wo_a.weight: arena slice is 2097152 bytes but the stored
  F8_E4M3 layout expects 4194304 bytes (no B12x, no scale)` (2 MiB at two ranks, 1 MiB at four)
  and the chain exited.

**Cause (2026-11-24, on the source and in the plan, no window).** The shared-expert refusal the
first window would have hit is one defect and the `wo_a` byte check is another.

* **The shared expert borrowed a format no layer carries.** The gather read
  `plan::operand(Some(block), Part::DenseMlp)` for `ffn.shared_experts.*`, but every layer of
  both the released model and the second shape is MoE: the layout claims those tensors under
  `Part::Moe`, so the description holds no dense-MLP operand at any block and the plan panicked
  `the plan states no format for DenseMlp at Some(0)`. The released four-rank record predates the
  plan-format lookup, which is why the group serves while the tip would not. The plan now states
  the shared expert's format from the tensors whose layout role is `shared_expert`, per block,
  and answers only when the block's MoE geometry splits a shared width; a layer with no shared
  expert refuses by name (`bd23e64`, with a regression test over the released plan and the
  second shape at every rank).
* **`wo_a`'s stored bytes are stated without the world.** At the second shape one layer's `wo_a`
  is (o_groups x o_lora_rank) x dim = 1024 x 4096 E4M3 values, 4 MiB, and the arena holds 2 MiB
  at two ranks and 1 MiB at four: the rows split over the world exactly as the arena does. The
  expectation compared against is the unsharded 4 MiB, so the check is world-blind on one side.
  The instance list has a released row and a `_s2` row for `b12x_mxfp8_wo_a_sm121` and no
  `_s2w1` or `_s2w2` row, so worlds 1 and 2 stage the released cubin, compiled for a geometry
  this shape does not have; world 1 passes the byte check (its arena is the whole tensor) and
  then hangs, which is consistent with that body running against the wrong geometry but is not
  yet proven from the staged cubin's compiled shape. Both halves are open.

So the bar this section sets is still unmet for the second shape at any world: what has run on a
device is the plan builder at four ranks and nothing beyond it. The next window is gated on the
`wo_a` fix.

**Result (2026-11-24, the third window, `wo_a` fixed).** The same checkpoint, arm, worlds and host,
tip `2d4d45de` (binary `cb316937`), 74 cubins staged in 17 s (39 `_s2`, 15 `_s2w1`, 15 `_s2w2`,
5 `nvfp4_prefill_sm121_*`), the same binary at the same path on every node, down 14m33s. The
`wo_a` refusal is gone: no world printed the byte check. Every world then failed at open on one
assertion, before any port bound:

```
thread 'main' panicked at crates/sparkflash-kernels/src/lm_head_ref.rs:128:5:
assertion `left == right` failed: the weight image is a whole number of HIDDEN-wide rows
  left: 2048 (world 1) / 1024 (world 2) / 3072 (world 4), right: 0
```

The host reference was again identical to the global file at all three worlds, and no device id
was produced, so no world reached the compare. The head shard the tail reads is exactly
`vocab_per_rank * hidden * 2` bytes - 32,768 x 4096 x 2 at one rank, halved at two, quartered at
four - and those lengths are the remainders above modulo 5,120. The row width divided by is the
`lm_head_ref::HIDDEN` constant, a released-model fact, while the shard is the plan's. So the
second shape cannot reach a launch until the head's block maxima take their width from the plan,
and that is a defect of the same kind as the last two: a released constant standing where the
plan's answer belongs. World 1 did not hang this time, which is evidence - not proof - that the
hang was the released `wo_a` body running against a geometry this shape does not have.

**Result (2026-11-24, the fourth window, the head width from the plan).** Tip `c4fd7d4d` (binary
`3a6fdf66`), the same checkpoint and group, 74 cubins, down 28m53s. The head assertion is gone and
the host arm passed at all three worlds with the same ids. The device arm then showed what the
previous three failures had hidden:

* **World 1 refused at open, deterministically.** `moe_routed device: the pinned routed launches
  run the released four-rank shape (dim 5120, inter 2304, world 4, 384 experts, top-6); got dim
  4096, inter 512, world 1, 64 experts, top-4`. The routed device path is released-only by
  construction: `MoeRoutedGeometry::new` compares against `released()` and nothing else, and the
  instance plan has no `moe_routed_device` row at any world, so there is no second-shape build to
  resolve.
* **Worlds 2 and 4 did not refuse and did not bind.** They ran at roughly 2.5 to 3 minutes a layer
  (world 2 reached layer 6 of 8 in about 3 minutes, world 4 reached layer 2 in about 6), with
  three workers waiting in an all-reduce. That is consistent with the released routed body running
  against a 64-expert, 512-intermediate shape, which is a hypothesis the next window tests, not a
  measurement this one made.

Two things the window used that are not the shape's: `SPARKFLASH_HEAD_SCREEN=0`, because the rows
head's screen is compiled for the released 5,120-wide row (160 code blocks) and has no 4,096 build,
so running it on this shape would read past its own table rather than fail; and, for the first
time, per-world invocations, because the harness stops a run at its first failing world and waits
600 s per world. The screen is a diagnostic over the head, not the head: what this run measures,
the certified head and its reduction, is unchanged. Giving the screen a 4,096 build remains a
separate item; the harness now declares the screen off and carries on to the next world after a
refusal (#70).

**Result (2026-11-24, the fifth window, the routed builds and the DSpark depth).** Tip
`1646b7c4` (binary `9b7a060b`), the same checkpoint and group, 85 second-shape cubins (40
`_s2`, 20 `_s2w1`, 20 `_s2w2`, 5 `nvfp4_prefill_sm121_*`), the cluster down 35m24s and
production restored to the same sha256 and serving.

* **World 1 opens the routed device path.** Window 4's `moe_routed device` refusal is gone: the
  plan names `moe_routed_device_sm121_s2`, `_s2w1` and `_s2w2`, the 11 new cubin rows are
  exactly the routed set, and the run proceeds. It then refuses 165 s after the device-rank line
  at the DSpark draft depth

      tp4-chain: forward refused on rank 0: SPARKFLASH_DSPARK_K=5: needs 1..=2

  The fixture declares no `speculation.dspark_k`, so the loop fell back to the literal 5 that
  the released plan states, while `draft::draft_block()` reads this plan's `dspark_block_size`
  of 2. Fixed at `ef8643b`: an unset depth is the plan's block, not a released literal. The
  released plan states 5, so the released arm reads the same depth as before.
* **Worlds 2 and 4 did not refuse, did not bind and did not token.** `:8131` never answered
  `/readyz`, no rank wrote an error, and the host was active on one thread; the workers were
  still in the device loop when the window stopped them. The rank log has no line between the
  run header and the serving loop, so this window cannot say which setup phase held them. It
  does retire window 4's hypothesis: the routed body is no longer the suspect, because the
  second-shape routed builds exist and are selected at every world. Window 4's reading of
  "2.5 to 3 minutes a layer" was the host reference arm's collective walk, not the device path,
  and is corrected here.
* **The trace that will answer it** (`ef8643b`, default off): `SPARKFLASH_SETUP_TRACE=1` writes
  cumulative milliseconds for each setup phase (begin, embeddings, attention body, shared
  expert, layer seams, tail, layer device, attachments, setup done) and one start/done pair per
  shared-expert all-gather and per embedding gather, naming the layer and tensor. Stderr only,
  no arithmetic and no launch change, forwarded to every rank by `serve-four-node.sh`.

The arm line is the same text on every rank that reached it,
`arm=device reduce=nccl evidence=false`; the host arm passed at all three worlds and the three
per-world references are identical and equal the global one. Deviations, recorded: the head
screen ran off (`SPARKFLASH_HEAD_SCREEN=0`; the screen's 160-block table has no 4,096 build),
each world ran as its own invocation with caps of 900/780/660 s rather than 15 minutes because
the restore deadline cannot absorb three 15-minute windows and the harness's own 600 s ready
wait ended world 2 first, and the 120 s grace was 107 s because the step-1 recording took 35 s
and the control plane's drain was the path. The driver reports 0% GPU utilisation for every
process including the production server, so the window claims no GPU number. What the next
window owes: world 1 with the DSpark depth from the plan, then worlds 4 and 2 with the trace on,
and the device ids against each world's own reference.

**Result (2026-11-24, the sixth window, the draft stage).** Tip `359233a3` (binary `92485301`), the
same checkpoint and group, the same 85 second-shape cubins, `SPARKFLASH_SETUP_TRACE=1` and
`SPARKFLASH_HEAD_SCREEN=0`, worlds 1, 4 and 2 in that order, each its own invocation. The arm line
is `arm=device reduce=nccl evidence=false`; the host arm passed at every world and the three
per-world references are identical and equal the global reference

```
[[6653, 20427, 12813, 422, 9389, 6388, 16395, 6405],
 [25425, 11040, 11559, 23442, 29453, 17940, 5192, 25888],
 [30169, 105, 2323, 11724, 1353, 885, 3013, 30683]]
```

* **Every world opened and every world died at the same plan check.** Setup is healthy and fast:
  every rank reached `setup done` in 2.0 to 4.3 s (world 1 rank 0: layer seams 203 ms, tail
  3104 ms, layer device open 3890 ms, attachments 3908 ms), the shared-expert gathers are 0 ms
  each, and then rank 0 and every worker panicked at `plan.rs:1224`

      forward open: no MTP stage 2 claims a tensor ending .markov_head.head.weight

  The depth fix moved the failure one layer later: the draft block then opened, and asked the
  plan for the released model's stage two. The synthetic second shape states two DSpark stages
  (`num_nextn_predict_layers: 2`), and the checkpoint writes the Markov head, the head norm and
  the Markov embedding only on the last stage (`tensor.rs`, `stage + 1 == n_mtp_layers`), exactly
  as the host reference derives them from `n_stages - 1` (`forward/mtp.rs`). The device path
  spelled `2` in four places. `draft.rs` now takes the stage from `plan::mtp_last_stage()`;
  `mtp_claim_result` is the reportable form of the name lookup. The released three-stage plan
  still reads stage two. Regressions: `tests/second_shape.rs`
  (`the_draft_blocks_last_stage_is_the_plans`) and the device-open gate below.
* **The silence in window 5 was this panic.** Worlds 2 and 4 printed nothing that window because
  their ranks died at the same check after the last trace line; with the trace on, all three
  worlds print it. No world produced a device id, a first token or a `VERDICT` line.
* **The window is recorded, not repeated.** Per the lead's freeze, no cluster window is dispatched
  until the host gate and the harness change below are merged; the next window is world 1 only,
  and worlds 4 and 2 run only after world 1 produces its ids.

Deviations: the grace period was 31 s, not 120; the harnesses were stopped at the panic, so no
world issued a `VERDICT`; nvcc ran beside the cargo build. The cluster was down 9m14s including
the restore, production came back on the same release binary (`5a299b15`) through the control
plane, and `/readyz` answered 200 25 s after the launch.

**Result (2026-11-24, the ninth window, world 1, the embedding gather's streams).** Tip `4caf7b4`
(binary `880c8e1`), world 1 only. The checkpoint and the world-1 reference were reused from the
seventh window, so the run went straight to the group; 85 cubins, `SPARKFLASH_NCCL_LIB` exported,
trace on. Setup completed for the first time: `setup done` 4,488 ms after the run header (layer
seams 214 ms, tail 3,322 ms, layer device open 4,103 ms, attachments 4,121 ms), and the first
prefill then refused:

```
sparkflash serve: device_failure epoch_ms=... the device loop failed, exiting: rows: the device
embedding gather takes 1..=64 rows of 4096 x 4, not 40 of 40 rows of 4096 x 2
```

The checkpoint states `hc_mult: 2` (the fixture took the model config's default), the row kernels
`OR_S_HC 4u`, and the guard at `rows.rs` compared the two only when the first gather ran. The
stream count is a model fact, not a format: the second shape had taken a count no rows build
states. The shape now states 4, the released model's count and the only one the row kernels are
built for; the guard compares against one named constant (`ROWS_STREAMS`), and the device-open
gate asks the plan for its stream count at every world (`embedding gather streams`, sensitivity
above). The shape's host reference was regenerated on the dev box and now reads

```
[[24319, 14482, 27568, 20871, 10699, 19197, 5245, 24450],
 [2230, 19981, 7902, 23487, 13332, 10093, 23350, 20839],
 [18661, 1217, 25908, 3469, 27591, 27897, 27916, 23008]]
```

The ids in the sixth and seventh window records belong to the two-stream shape and stay as
history. Restore: `readyz` 200 at 14:21:29Z, 4/4 ranks, the production binary unchanged, one
4-token completion; downtime 6m34s (stop 14:14:55Z, launch 14:20:21Z). Deviations: the NCCL
directory holds `libnccl.so.2`, not `libnccl.so` (the loader's first name, present on all four
nodes), so step 0's literal check was satisfied by the `.so.2`; the harness was SIGTERMed at the
refusal, so no world issued a `VERDICT`; no worker ran.

**Result (2026-11-24, the tenth window, world 1, the dedup down block).** Tip `204a256` (binary
`3dcd68a3`), world 1 only, a fresh checkpoint and a fresh world-1 reference (the shape's config
changed), the same 85 cubins. Setup completed at 4,597 ms (layer seams 248, tail 3,398, layer
device open 4,200, attachments 4,222), the head's host arm produced

```
[[24319, 14482, 27568, 20871, 10699, 19197, 5245, 24450],
 [2230, 19981, 7902, 23487, 13332, 10093, 23350, 20839],
 [18661, 1217, 25908, 3469, 27591, 27897, 27916, 23008]]
```

which equals the dev-box reference, and the first prefill then died at the routed dedup down
launch:

```
sparkflash serve: device_failure epoch_ms=... the device loop failed, exiting: moe_routed dedup:
down: cuLaunchKernel failed: 1
```

The gate/up dedup launch before it succeeded, so the plan, the cubin and the readers were all
right up to that point. The down body's thread count is half one rank's intermediate columns
(`OPT_MOE_DD_THREADS = OPT_MOE_INTER / 2`), the instance compiles its launch bounds from the same
number, and the launch passed the released literal 288 - correct for the released 576, refused by
the driver for this shape's 512 (its cubin is built for 256 threads). The block is now the plan's:
`MoeRoutedGeometry::dedup_down_threads()`, half `local_inter()`. The other launches in that path
state constants the instances do not vary (`OPT_MOE_GU_THREADS` 128, `OPT_MOE_D_THREADS` 256), so
this was the one instance-dependent block in the file.

Deviations: the worker check by address failed on the head's stale known-hosts entry and was made
by the nodes' own names, which is what the launcher uses; the harness hung past its 150 s window
after the refusal and was SIGTERMed, so its collect and stop were run by hand; the cluster's
`/v1/cluster/stats` reports an empty node list before and after (a pre-existing fault), so the 4/4
restore is proven by the process list and the launcher's own status. Downtime 7m16s (stop
14:59:59Z, launch 15:06:44Z, `readyz` 200 at 15:07:15Z), production back on the same release
binary (`5a299b15`) and serving.

**Result (2026-11-24, the eleventh window, world 1, the prefill fault).** Tip `e201479` (binary
`1ff96cea`), world 1 only, the checkpoint and the world-1 reference reused from the tenth window
(the global reference was rewritten and matched), the same 85 cubins. The routed dedup down launch
that refused the tenth window with `cuLaunchKernel failed: 1` now runs: the gate/up and down
dedup launches pass and setup completes at 4,177 ms (layer seams 254, tail 3,019, layer device
open 3,805, attachments 3,825). The first prefill then reports

```
sparkflash serve: device_failure epoch_ms=... the device loop failed, exiting: compressor latent:
the forward rope launch: cuLaunchKernel failed: 700
```

700 is `CUDA_ERROR_ILLEGAL_ADDRESS`, not a launch-configuration refusal: it is raised by this
launch whether the fault is its own or an earlier asynchronous kernel's, so the window cannot say
which launch touched the bad address. What the host can already say: the rope's row span, its
norm weight and its two tables are checked before the launch and the checks are satisfied at this
shape (the pair holds two `mla_latent`-wide bf16 rows, the weight is its width, and
`rope_table_span_bytes` is `2 x max_positions x pairs x 4`), and the layer's own refusal for a
position past the tables did not fire. The class is therefore "an asynchronous fault whose first
launch is not named", which no host check of the plan can see; naming it needs the run to make
every launch synchronous. No token was generated and no device id exists; the arm line is the same
`arm=device reduce=nccl evidence=false`.

Restore: stop 15:17:52Z, launch 15:22:54Z, `readyz` 200 at 15:23:24Z, downtime 5m32s (the cap
held), production back on `5a299b15` with four serve processes and `/v1/models` serving. The
harness was stopped 150 s after the rank died, so no `VERDICT` was issued. Deviations: the
one-address stop form the brief suggested is refused by the launcher (it wants four addresses in
rank order), so the stop was the harness's own; `/v1/cluster/stats` again reported an empty node
list beside a serving group.

**The device-open gate (2026-11-24).** `crates/sparkflash-deepseek-forward/tests/device_open_gate.rs`
builds the second shape's plan at worlds 1, 2 and 4 and walks every part the device opens: the
stored format each part states and the kernel symbols the plan names for it, the shared expert's
own format, one rank's `wo_a` rows in the B12x operand the body reads, the routed geometry an
instance was built for, the head's block-maxima buffer at the plan's width, the screen's certified
width and the setting that turns it off, the hidden-stream count the rows kernels are built for, the routed dedup body's block against
the intermediate its cubin was compiled for,
every DSpark stage's tensors including the last stage's Markov head, the depth authority, and the
NVFP4 prefill candidate and its pin. It collects every
refusal and reports them in one run, naming the part. The gate is plan-driven: it does not list
the recorded defects, it asks each part the question they answered wrongly. Sensitivity was checked by
reverting each fix's behaviour in turn on the gate's tip; each revert produced exactly its own
named case at all three worlds, and the passing tree produces none:

| reverted behaviour | gate case |
|---|---|
| head buffer at the released 5,120 width | `head block maxima` |
| E4M3 dense class read from the empty `DenseMlp` aggregate | `wo_a per-rank stored bytes` |
| the plan states no shared-expert format | `shared expert format` |
| the geometry accepts only the released shape | `routed device geometry` |
| an unstated depth falls back to 5 | `DSpark depth` |
| the Markov head reads the released stage two | `drafter stage claims` |
| the plan states 2 hidden streams, the rows kernels are built for 4 | `embedding gather streams` |
| the dedup down block is the released 288 | `routed dedup block` |

The gate cannot be compiled on the pre-fix tree `1ef7a6ca` verbatim: the plan answers it asks for
(`shared_expert_format`, `dense_stored_format`, `blocks_per_row`, `mtp_last_stage`, `draft_depth`)
did not exist there, and a version of the gate that compiled on both trees could only compare
constants, not plan answers. The six reverts are the pre-fix evidence: each is the defect as it
stood, restored in the one place the fix moved, and each is caught by name.

**The harness runs every world in one invocation (2026-11-24).** `scripts/second-shape.py` no
longer stops at the first failing world: it records the world's refusal, tears the group down and
goes on, then prints one PASS or FAIL line per requested world and returns 1 if any failed. Each
world keeps its own `host-reference-w{W}.json` and `.top2.json`, and the comparison is still the
exact id list against that world's own file, with world 1 checked against the global reference
including on a reused file. `host-reference-w{W}.json.plan` hashes the variant config, the world,
the prompt list and the token budget; the host arm is skipped when the reference, its sidecar and
its top-2 file exist and the hash matches. `checkpoint.plan` does the same for the checkpoint.
`SPARKFLASH_HEAD_SCREEN=0` is now part of the device environment, beside
`SPARKFLASH_UNPINNED_SECOND_SHAPE=1`, so the 4,096-wide shape does not run a screen built for
5,120. Unit tests are in `scripts/tests/test_second_shape.py`.

**Result (2026-11-24, the seventh window, world 1, no device execution).** Tip `12ae67b` (binary
`be85d598`), world 1 only, the same checkpoint and the 85 cubins from the sixth window. The host
arm passed in 6 s and the world-1 reference equals the global one. The device group then refused
at the bootstrap before setup began:

```
tp4-chain: forward reduce bootstrap r0: NCCL bootstrap failed before any communicator was
opened: rank(s) [0] could not load the NCCL library before the bootstrap gather; refusing on
every rank before any communicator is opened (a peer that entered NCCL init here would wait
for a rank that never joins); aborting this rank before any layer - a bootstrap failure must
not reach a layer
```

The launch environment carried no `SPARKFLASH_NCCL_LIB`, and the test tree's own
`target/deps/nccl` does not exist; the production install's `target/deps/nccl` does. The plan
gate could not catch this: it walks the plan and never resolves a library, and the brief for the
window did not export the variable. The harness now refuses a device run that cannot name its
NCCL directory before it writes the checkpoint or starts a rank, with tests in
`scripts/tests/test_second_shape.py`; the next brief exports the production path. No device id
and no token were produced, the group was never started, and production was restored in the same
5m13s the group was down (`/readyz` 200, four ranks, the release binary unchanged).

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

**Result (2026-11-24, R3, no device).** The instance list is the one statement of what a kernel is
built for: 730 instances over 32 sources in 111 compile groups, and `scripts/instances.py --check`
holds it together. The head and its reduction take the rank count as an instance argument, the
routed expert's padding is an instance, and every family carries the world-suffixed copies the
launcher stages: for the second shape 39 `_s2`, 14 `_s2w1` and 14 `_s2w2` cubins, built from the
same plan the run reads. The check that the relaxation moved nothing released is the one this
section's ladder already asks for: the world-four lanes reproduce production's cubins byte for
byte, 31 of 31. The routed device path then needed its own second-shape builds (`3db0ce5`):
18 rows in 12 compile groups for the routed expert stage, the combine, the router, the opt
router and both MXFP4 expert bodies, taking the list to 758 instances in 125 groups, of which
the harness selects 85 for the second shape. What R3 still owes is the device run at one rank;
the window above could not give it, because `wo_a`'s stored bytes and world rows are open.

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

**Result (2026-11-23, F1).** `sparkflash_model::format::StoredFormat` is one value — the values,
the block one scale covers, the scale kinds (power-of-two grid, tile scale, row scale,
tensor-wide scale) and the layout (row-major, 32-tile, packed) — converted once from the release's
declaration at the layout boundary and carried by the plan (`PartRequirement::format`) to the
resolver (`ResolvedKernel::format`). `Precision` gains `MxFp4` and no longer folds it into
`NvFP4`. Every catalogue symbol carries the format its build recorded, from the instance list
(all 408 entries state one), and the resolver refuses a symbol of another arithmetic; DeepSeek's
masks were corrected from what each source reads, and a part whose compressor tensors are shared
across blocks takes the format of the tensor rather than the block's class. The scan row prints
the format and names a format with no body as "not built". The forward's own dtype checks are
F2's work and are untouched here.

**Result (2026-11-23, F2).** The forward no longer reads a dtype word to decide what a tensor is.
A stored format now carries the scale plane's own values (`ScaleValues`: `ue8m0`, `e4m3`, `f32`,
`bf16`), so an FP8 128-tile release stored with f32 inverse scales and the same release with bf16
ones are two formats a body cannot be handed for one another; and `mx-fp4` now means the arrangement
every fp4 expert body in the tree reads - one power-of-two scale a row, every 32 values along the
contraction - with the two-dimensional block grid kept as its own `MXFP4-GRID` name that no release
declares. That correction is what makes the bind's row-scale plane the right size for the released
experts: the block grid it used to name is smaller than the stored grid.

The three consequences of a format are derived in one place, `sparkflash_deepseek_forward::operand`,
and `Description` indexes the result per block and per part at open:

* `ScalePlanes` sizes what the bind carries beside the values - a block (grid) plane, a row plane, a
  tensor-wide factor and a packed tensor's escape list - from the format's block and kind, never from
  a byte count of a tensor the caller happened to read;
* `arrange` is the loader's one arrangement: as stored, the 32 x 32 B12x permutation, or the packed
  word form, and it refuses a store that is not the format's byte length;
* `ActivationForm` and `choose` decide the form a seam ends in and the body that reads it, per block
  and per part. A format no claimed catalogue symbol decodes has no body, which the scan shows as
  "not built" rather than launching the nearest one.

The four files the section names ask the format now. `tp4_operands` states each projection's format
from the configuration and checks the scale plane's size against the weight's block instead of the
`F8_E8M0` word; `resident_weights`'s mhc_seam table is a list of formats and the bind compares the
shard's; `moe_device::read_spec` takes the format and compares the file against the dtype that format
implies, its scale-plane reader likewise; and the arena's `b12x_layout` and `tp4_forward`'s
shared-expert gather decide from `reads_b12x(format)`, the shipped bytes' own declaration remaining
the predicate only for a tensor the plan states no format for (a header dump, a fixture). The
production entry builder states every tensor's format from the plan, so the device bind reads it.

Two things are deliberately not in this step. The device bodies still read the released arithmetic
only: a tile-format part resolves to no DeepSeek-image symbol, so its scan row says not built, and
the F3 recipe below names the body that would serve it. And the Engram table keeps its own ue8m0
grid in the writer (model.py:345 is its format, not the linear class's) while the layout's explicit
E4M3 branch still reports the release's linear class for it; that one scan row is F4's to close, and
no body reads it through the plan's format today.

**Result (2026-11-23, F3).** FP8 on 128 x 128 tiles is the worked example: the dense projections,
the shared expert and the routed experts, on prompt rows and token rows, with f32 and bf16 inverse
scales, and with BF16 rows and rows quantised per 128 columns. **The one copy is
`kernels/shared_prefill_mma_sm121.cuh`**: `shared_prefill_mma_body<SPM_W_FP8, ...>` for BF16 rows,
`shared_prefill_q8_body` for rows quantised per 128 columns, and `shared_prefill_mma_rows_body` for
the few-row (token) end of the same tile - one header, one 128-row x 128-column tile and one typed
scale (`ScaleT`: `float` for GLM, `__nv_bfloat16` for Qwen). `opt_mma_bf16_body` was the other
candidate and is refused: its weight modes are the B12x E4M3 tiling with a ue8m0 scale a 32 values,
row-major bf16/f32, and row-scaled FP8. None of them reads a 128 x 128 tile scale, and adding that
mode is the third copy D95 and D96 removed.

The measurements behind the choice are the ones on record. D96 moved Qwen's FP8 experts onto this
tile and measured GLM's FP8 prefill at 594 tok/s against Qwen's 463-467 on the token kernels; D101
measured the same body on GLM's bulk chunk at 979-993 tok/s against 81 tok/s on the token path for a
6.6k-token prompt, with the drafter's acceptance at 24 % against 90 % while the chunk was detached.
D95 records why V4.1's own `opt_mma_moe_*` cannot take this format at all: ue8m0 powers of two
against an arbitrary bf16 tile scale, packed E2M1 weights, and compile-time shapes. On the token end
the same tile body is the few-rows form, so no second kernel family is introduced; the DeepSeek
plan's `opt_mma_bf16a_*` bodies keep serving the released MXFP8 arithmetic.

The steps, for the F4 families to follow:

1. **State the format.** Declare it (values, block, scale values, block width) and add it to
   `WeightFormat`/`StoredFormat` if it is a new arrangement; the layout reader converts it once, and
   `same_arithmetic` must tell apart anything the body's instruction differs on.
2. **Make the plan's tensors agree.** `TensorPlan::scaled` writes the values dtype, the scale dtype
   and the grid's shape from the format, so the store and the plan cannot disagree; a tensor with its
   own model fact (the Engram table's ue8m0 grid) states its format explicitly.
3. **Teach the reference the grid, not a default.** Every read takes the grid the format implies
   (`ModelConfig::dense_grid`/`expert_grid`), so a release stored another way is read on its own
   grid; a scale dtype the fused path does not read falls to the materialised decode, exact for any
   grid.
4. **Let the seam decide once.** `ScalePlanes::of` sizes the bind's planes, `arrange` produces the
   body's layout, `ActivationForm::of` fixes what the seam before the part emits, and `choose` picks
   the body from the plan's claimed symbols by the format each build recorded.
5. **Bind what the format needs.** The grid plane, the row plane, the tensor-wide factor and the
   packed escape list; a body that needs one the bind does not carry is refused by name at open.
6. **Pick the body the tree already has.** The shared tile header for a 128-tile format (BF16 rows on
   the decode end, per-128 quantised rows on the prompt end), or give an existing body the decode. A
   new body needs its reason and its measurement written here.
7. **Prove it on the synthetic checkpoint.** `scripts/second-shape.py run ... --stored-format
   fp8-f32` (or `fp8-bf16`) writes the same fixture in that format, runs the host reference over it
   and compares the group's greedy ids at one, two and four ranks.

Proof of this step, run on this host (`/tmp/sf-f3` for f32 scales, `/tmp/sf-f3b` for bf16): the
second-shape fixture declared as `fp8`, `weight_block_size: [128, 128]`, `scale_fmt: f32` (then
`bf16`) and `expert_dtype: fp8`, written by the crate's own writer, compared against the host
reference at one, two and four ranks: `second-shape: VERDICT PASS` and
`external check PASS` at every world, both scale types. The reference's greedy ids are the same for
both scale types and for all three worlds:

```
[[6653, 28875, 30803, 7797, 30906, 14034, 26554, 23093],
 [14371, 20678, 29812, 23985, 1458, 3459, 20210, 7815],
 [7084, 149, 12756, 20718, 22229, 27765, 24326, 8911]]
```

The released path is unchanged: the same code reads the released 32-block MXFP8 checkpoint through
`Block32`/`PerRow32` exactly as before, and the released proof (`scripts/second-shape.py` with no
`--stored-format`) still passes at one, two and four ranks.

**Result (2026-11-24, F4 NVFP4).** NVFP4 is declared as two formats, because the two releases
multiply the same group dequantisation by a tensor-wide factor in different places. Both are E2M1
values with one E4M3 scale a 16-value group (`ScaleKind::Group`) and two f32 tensor-wide factors
(`NvFp4Wide`). `NVFP4-QWEN` is `weight_scale_2` folded into the group scale plus the static
`input_scale` carried beside it; `NVFP4-INKLING` is the per-expert `weight_amax` over
`weight_outer_scale_denominator`, applied to the expert's output after accumulation.
`same_arithmetic` tells them apart, and `from_name` round-trips `nvfp4`, `nvfp4-qwen` and
`nvfp4-inkling`. A block-part range is a format name with the range attached
(`BlockPartFormat::from_name("nvfp4-qwen@0-39")`); a sibling that changes the expert format
part-way up the stack states the same format with that helper.

The bind carries the group plane and both factors. `ScalePlanes::of` sizes the per-row group plane
(`rows x ceil(cols / 16)` E4M3 bytes) in `group_bytes`, and `tensor_wide` is the convention, not a
bare flag; `total_bytes` adds its two f32. `Operand::tensor_wide` states the Inkling per-expert
amax: it is one f32 per expert tensor, not a block scale, so the routed-expert bind carries one an
expert and the body reads it by the local expert index. `arrange` is the row-major identity for
both conventions and refuses a store that is not the format's byte length; `choose` resolves a
body only from a claimed symbol whose recorded format has the same arithmetic (a symbol with no
recorded format is now refused rather than treated as a wildcard).

The host reference reads the checkpoint's own declaration: `ModelConfig::expert_format` takes the
16-value group from `expert_dtype`, the tensor plan writes the E4M3 group plane and the companions
the convention names, the writer encodes them, and every routed-expert read applies
`PerRow16` and then the convention's factor. `scripts/second-shape.py` gained
`--stored-format nvfp4-qwen` and `nvfp4-inkling`; the dense class stays the fixture's released
32 x 32 power-of-two grid. Both variants pass at one, two and four ranks on the host arm:
`second-shape: VERDICT PASS` and `external check PASS` at every world, with the reference's greedy
ids

```
[[6653, 20427, 12813, 422, 9389, 6388, 16395, 6405],
 [25425, 11040, 11559, 23442, 29453, 17940, 5192, 25888],
 [30169, 105, 2323, 11724, 1353, 885, 3013, 30683]]
```

for Qwen's convention (the same ids the lead's pre-change reference recorded) and the same ids for
Inkling's. The released proof still passes at one, two and four ranks.

**The device path is built and the device arm has not run it here.** The one copy is
`kernels/shared_prefill_mma_sm121.cuh` with an `SPM_W_FP4` weight mode: each E2M1 code is
dequantised by its 16-value E4M3 group scale into the BF16 weight tile, Qwen's weight global joins
the group scale at the load (where its `q4e`/`q4p` body applies it) and Inkling's
`amax / denominator` is applied to the output after accumulation (where `inkling_nvfp4_expert_*`
applies it). The mode takes the group scale without a second copy: the scale is one byte a 16-value
group of the same row, and the tile already stages bf16 weight words, so the decode is a load-time
multiply. The wrappers are `kernels/nvfp4_prefill_sm121.cu`: prompt and few-row entries for the
gate/up and down projections, one pair per convention.

The base's seam reads it. `ScalePlanes::of` sizes the group plane and the two factors;
`ModelConfig::expert_format` selects the convention; `MoeDeviceBackendImpl` branches on
`expert_format().has_group()` before the released MXFP4 path, reads the same `.weight`,
`.scale` and convention companions the bind describes, gathers each local expert's routed slots
into f32 rows, and launches the prompt tile for more than eight rows or the rows form for eight or
fewer, for gate/up and then down. The SwiGLU and routing weight are the reference's f32 arithmetic
between the two launches. A staged NVFP4 cubin is loaded by name; a missing one is refused at open
rather than falling back to an MXFP4 body.

`kernels/instances.toml` carries 40 NVFP4 rows: all eight entries at Qwen3.8's released expert
geometry (hidden 2560, intermediate 640, 512 experts, top-10), at Inkling-2-Small's (hidden 4608,
intermediate 2304, 384 experts, top-8), and at the synthetic second shape (hidden 4096, 512
intermediate, 64 experts, top-4) at intermediate 512 and the per-world 256 (TP2) and 128 (TP4).
`scripts/build-cubins.sh` takes them without a GPU edit; `scripts/second-shape.py` stages the
`nvfp4_prefill_sm121_*.cubin` files under their own names for `--device`.

The verdict is pending the lead's window: no device run, no tok/s and no accuracy gap against the
arms that serve Qwen3.8 and Inkling today exist for this change, and the scan row for an NVFP4
expert part still says `not built` because no catalogue key claims the new symbols. R3's
rank-count instances are still unmerged, so the base device arm at TP2 has its own gate independent
of this body. The command for the window, after `scripts/build-cubins.sh`, is:

```
python3 scripts/second-shape.py run --released <reference> --out /tmp/f4-nvfp4-dev \
  --binary target/release/sparkflash --writer target/release/examples/write-synthetic-checkpoint \
  --reference target/release/examples/host-reference --stored-format nvfp4-qwen \
  --world 2,4 --port <free port> --device --build-cubins --released-cubins <released cubin dir>
```

then the same with `--stored-format nvfp4-inkling` and `--world 4` if TP2 stops at R3.

The single-operation check on real tensors is prepared as two examples, run after the same cubin
build:

```
cargo build --release -p sparkflash-model-qwen4-exp --example nvfp4_expert_qwen
cargo build --release -p sparkflash-model-inkling2 --example nvfp4_expert_inkling
target/release/examples/nvfp4_expert_qwen \
  --checkpoint <Qwen3.8-Flash-Next-NVFP4 snapshot> --cubin-dir target/cubins
target/release/examples/nvfp4_expert_inkling \
  --checkpoint <Inkling-2-Small snapshot> --cubin-dir target/cubins
```

The Qwen check reads one released expert's `gate_proj`/`up_proj`/`down_proj` codes, per-16
E4M3 scales and `weight_scale_2`, and one real `embed_tokens` row. It compares the serving
`q4e_expert_gate_up` (its activated rows) against the shared tile's two gate/up launches plus the
same host `silu(gate) * up`, and the serving `q4e_expert_down` against the tile's down launch on
the serving arm's activated rows. The Inkling check reads one released expert's `w13_weight` and
`w2_weight`, their per-16 scales, the expert amax and the outer denominator, and one real
`model.llm.embed` row; it compares `inkling_nvfp4_expert_matvec` against the tile's rows form on
both projections, with `amax / denominator` applied at the output by both. Each prints max
absolute/relative difference, max ULP distance, bit-equal count, and median launch milliseconds
for the serving arm and the tile over `--repeat` runs. The bar is the accumulation-order
allowance, not a decode tolerance: the arms read the same stored bytes, so only the f32 order of
their sums may differ (about 1e-6 relative); the harness fails above 1e-3 relative, three orders
of magnitude above that noise and three below a two-percent band. A swapped convention, a per-16
scale on the wrong block or an amax/denominator applied twice is a failure. The commands run only
in the lead's window; nothing in this repository changes the catalogue or the scan row until the
device arm has run the synthetic variant.

What a later F4 family must do differently: it inherits the two convention names, the `Group`
plane and the `SPM_W_FP4` mode; it must not re-use one name for both, and it must not fold the
tensor-wide factor into the group scale for Inkling (or apply it to the output for Qwen) merely
because the products are equal. The catalogue claim for the new symbols, the scan row that still
says `not built`, and the real-tensor bit check against the Qwen and Inkling arms are open.

**Result (2026-11-24, F4 BF16 row-major and packed BF16).** BF16 row-major is unscaled:
`StoredFormat::BF16` has no scale plane, so `ScalePlanes::of` sizes nothing and carries no escape
list. The packed word form is that same BF16 arithmetic in the arrangement
`kernels/bf16_packed_sm121.cuh` reads; its one plane is the escape list,
`[offsets: rows + 1][entries: column << 16 | word]`, which the bind states and `arrange` reads.
A checkpoint declares the two by `quantization_config.quant_method`: `bf16` or `packed-bf16`.
**No released layout declares packed BF16** - the layout dtype vocabulary has no spelling for it -
so the packed declaration is exercised only by the synthetic second shape. The writer packs the
BF16 rows it generated and writes `{name}.escape` beside `{name}.weight`; the reference reader
(`forward/weight_view.rs`) and the seam's `arrange` unpack them back to the row-major BF16 the body
already reads, so no kernel is added and the packed form's host ids must equal the row-major
form's. The host-TP path unpacks a packed weight before it shards it (`tp4_host.rs`), because
a packed row cannot be sliced by byte range.

By the recipe: (1) the formats are `StoredFormat::BF16` and `StoredFormat::PACKED_BF16`, already
stated by F1/F2; `from_name("packed-bf16")` round-trips. (2) `TensorPlan::scaled` writes no
`{name}.scale` for an unscaled format and keeps the format per weight, so the writer packs exactly
the dense class (the Engram table keeps its own grid). (3) The reference reads a BF16 weight
through `read_optional_scale` (attention, shared and routed expert, MTP) and a packed one by its
escape list; `dense_grid()` is `None` for BF16. (4) `arrange` unpacks the packed form into
row-major BF16 and refuses a store that is not the escape list's length; `choose` resolves the same
BF16 body for both names. (5) `ScalePlanes` carries the escape plane for a packed tensor and no
plane for a BF16 one. (6) No new body: `opt_mma_hostw_bf16a_bf16w` and `opt_rows_hostw_bf16w`
are the tree's row-major BF16 modes, and their synthetic-shape instances were already on
`kernels/instances.toml` from F1 (`opt_mma_sm121_s2` and `opt_rows_sm121_s2`, hidden 4096; no new
row or cubin edit is needed). The device rows seam now reads a BF16-stored attention part from the
plan's format: wq_a/wkv over the f32 normed rows through `opt_rows_hostw_bf16w`, and
wq_b/wo_a/wo_b over BF16 rows through `opt_mma_hostw_bf16a_bf16w` (wo_a once a group, which has no
group stride). A packed part is refused by name at that seam until the loader arranges it. (7)
Proved on the host arm at worlds 1, 2 and 4 for both declarations: `second-shape: VERDICT PASS`
and `external check PASS` at every world, with the same host-reference ids for `bf16` and
`packed-bf16` (the packed form is lossless):

```
[[6653, 20427, 7012, 12813, 1671, 30705, 17404, 5096],
 [25425, 11040, 2024, 16309, 32049, 1442, 21654, 2395],
 [30169, 2663, 30363, 3368, 22356, 24675, 32436, 8472]]
```

**The device arm is not proved and this family is not done.** What is open, by name: the arena's
upload does not call `arrange`, so a packed weight would bind packed bytes and is refused at the
seam instead; the shared expert, the MTP `main_proj` and any BF16 expert/router projection still
take their coded paths (the synthesised checkpoint's local expert intermediate 256 at two ranks is
the lead's released-format expert row); and no device launch of these bodies has run. The window
commands, to be run by the lead with the cubin build (`scripts/build-cubins.sh` rebuilds
`opt_mma_sm121_s2`/`opt_rows_sm121_s2` from the existing rows):

```
python3 scripts/second-shape.py run --released <released> --out /tmp/f4-bf16-device \
  --binary target/release/sparkflash \
  --writer target/release/examples/write-synthetic-checkpoint \
  --reference target/release/examples/host-reference \
  --stored-format bf16 --world 2,4 --device --released-cubins <released cubins> --port 8131
```

then the same with `--stored-format packed-bf16` (currently expected to refuse by name at the
packed seam), and the one-operation check: the real BF16 dense projections of the model that
stores them (Dory or Inkling), bits first and then tokens/s against the arm that serves that model
today. The released path is unchanged and still passes; the device arm at world 4 was never
attempted by the earlier window.

**Result (2026-11-23, F4 row-scaled MXFP8).** The format is E4M3 codes with one `ue8m0` scale a
row, every 32 columns along the contraction, row-major: `StoredFormat::MXFP8_ROW_32`, carried by
the new `WeightFormat::MxFp8Row` and declared by a checkpoint as `expert_dtype: "mx-fp8-row"`.
The layout reader states it from the layout's own evidence, the sibling scale tensor's shape: one
entry a row rather than a 32 x 32 grid (`sparkflash-ops/src/engine_plan.rs`, `is_row_scale`), so
Inkling's layers 40-47 now report this format instead of a grid. The one copy body is
`opt_mma_fp8_rowscale_f32o` (`kernels/opt_mma_sm121.cu:279`): row-major E4M3 weights, the per-row
`ue8m0` plane and the block-quantised activation seam, at any token count, so it serves prompt rows
and token rows. The base already launches it for the drafter head
(`tp4_forward/draft.rs:290`). The Engram table's embed lookup reads the same row arrangement in
`kernels/engram_embed_sm121.cu:75`; the `engram.wkv` projection beside it is the 32 x 32 grid
(`sparkflash-model-deepseek-v41/src/tensor.rs`, `push_engram`), not this format. No new kernel is
added, so there is no new measurement: the body's cost is the one the drafter head already pays.

The plan states the format per block: the DeepSeek writer's `TensorPlan::scaled` writes the
[out, in/32] scale plane for the routed experts, the MTP experts and every other expert tensor;
`ScalePlanes::of` sizes the bind's row plane (rows x cols/32 bytes), `arrange` passes a row-major
store through, and `moe_device::read_experts` now takes the block's format, sizes both planes from
it, arranges the stored bytes and refuses a tensor-wide factor the routed bind does not carry. A
unit test pins that the instance list builds `opt_mma_fp8_rowscale_f32o` for this format and that
`choose` resolves it for a routed-expert block. `kernels/instances.toml` already carries
`opt_mma_fp8_rowscale_f32o` at the synthetic shape's per-world intermediate 512, 256 and 128
(R3's `_s2w1`, `_s2w2` and `_s2` rows, with the router and gate/up rows the source asserts beside
them); this step adds the `deepseek-moe` catalogue symbol that claims the body for `Part::Moe`.

Proof on the host arm, `/tmp/f4-mxfp8`, this tip: the fixture written with `expert_dtype:
"mx-fp8-row"`, `weight_block_size: [32, 32]`, `scale_fmt: ue8m0` and compared with the host
reference. The reference's greedy ids are

```
[[14394, 19734, 21122, 7155, 13460, 11376, 10110, 6859],
 [23919, 23153, 804, 31899, 25048, 27934, 8126, 23473],
 [25626, 8511, 2658, 12093, 30760, 21890, 17708, 12438]]
```

World 1 is the anchor and agrees exactly (`second-shape: VERDICT PASS`, `external check PASS`).
At two ranks all three prompts part from the third generated token on; at four ranks prompt 1
parts from its third token while prompts 0 and 2 agree. The released path at the same tip passes
at one, two and four ranks with the ids recorded for the device window, so the harness and the
split are sound. The difference is the row-scaled E4M3 expert sum: measured on layer 0 of this
checkpoint, the four rank partials of `routed_forward` differ from the reference's single global
sum by at most 1.75e-10 absolute (4.7e-7 relative; 3,803 of 4,096 f32 values differ in bits). The
released E2M1 arm regroups without changing its bits; E4M3 codes do not, and this fixture's logits
are close enough that the rounding flips a greedy token. The worlds 2 and 4 runs are therefore
**blocked on the reference policy**, not claimed as passes: device(W) must equal host(W) exactly,
and host(W) against host(1) may differ only at a greedy tie inside the measured band. The
property is committed as a test beside the routed forward
(`the_row_scaled_expert_split_rounds_where_the_released_arm_regroups_exactly`): the released
format regroups with zero differing bits at two and four ranks, the row format does not, and the
relative band is bounded. **The family is not done.**

The device launch is wired: when the block's stored format is `MXFP8_ROW_32`, `moe_device::ffn`
takes the plan's arranged expert tensors and calls `ffn_mxfp8_row`, which gathers each expert's
routed slots, launches `opt_mma_fp8_rowscale_f32o` for gate and up over each rank's intermediate
slice, applies the reference's SwiGLU and routing weight on the host, quantises those rows, and
launches the down projection over a gathered `[dim, width]` slice, adding the rank partials in
rank order. The cubin rows use the harness's per-world names (`opt_mma_sm121_s2`, `_s2w2`,
`_s2w1`) so `stage_cubins` picks them up. **No device kernel has read this format**: the verdict
is pending the lead's window with:

```
python3 scripts/second-shape.py run \
  --released /home/allenh1/sparkflash/.agents/worktrees/f4-mxfp8/reference \
  --out /tmp/f4-mxfp8 --device --released-cubins <cubins> \
  --binary target/release/sparkflash \
  --writer target/release/examples/write-synthetic-checkpoint \
  --reference target/release/examples/host-reference \
  --stored-format mx-fp8-row --world 2,4 --port 8133
```

with the cubin build in the window (the per-world `opt_mma_sm121` cubins are part of it). Open
items: the device verdict; the device body's tensor-core accumulation against
the host's per-element sum at the same split (the exact-id bar above); and the joint mid-stack
variant (NVFP4 below a layer boundary, this format above), which is not started and needs this
launch proved first.

**Result (2026-11-24, F4 joint expert format).** A checkpoint states its routed experts one format
a block range: `quantization_config.expert_dtype` is a comma-separated list of
`format@first-last` (also `@first..=last`), parsed by
`sparkflash_model::format::block_part_formats` beside `BlockPartFormat::from_name`. A layer no
range covers is refused by name, never read as the class's format. `ModelConfig::expert_format_at`
is the one per-layer answer; `TensorPlan` states each backbone layer's format and writes that
layer's weights, scales and convention companions in it; the reference's `routed_format` and the
device's per-block `Operand` read the same statement, while an MTP stage keeps the plain class.
The fixture is `expert_dtype: "nvfp4-qwen@0-3,mx-fp8-row@4-7"`: eight backbone layers, NVFP4's
Qwen convention below and row-scaled MXFP8 above, so exactly one variant's expert format changes
part-way up the stack. The host arm agrees at one, two and four ranks (`second-shape: VERDICT PASS`,
`external check PASS` at every world), and each world's own split reproduces the global reference's
greedy ids exactly

```
[[6653, 28875, 12789, 22350, 23742, 29043, 20477, 3895],
 [338, 9686, 24519, 14376, 7470, 29768, 25769, 9869],
 [30169, 2663, 30363, 3368, 1942, 30952, 25487, 12199]]
```

so this stack's answer does not drift with the split. The device arm is the lead's window. The
device launch already chooses per block: the plan's `Operand` carries the block's format and
`moe_device` takes the NVFP4 or the row-scaled path from it. Open: the device verdict for this
stack.

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
