# Models built at load time: the path from here

Status: agreed direction, 2026-11-21 (owner's decisions are in section 8). Written for
implementers. It replaces the goal of `docs/engine/04-onboarding.md` ("an arm supplies one walk")
with a stronger one: there are no arms, and the engine every model runs on is DeepSeek's.

## 1. The target

A model is a checkpoint directory. Nothing in the source tree names it.

1. **Scan.** The web console (and `sparkflash onboard`, the same code) reads the checkpoint's
   config, tokenizer files and tensor index. No weights are read.
2. **Describe.** From those it builds one description of the model: every stack (language,
   vision, audio, draft head), every block in it, every part in every block, each with its shape,
   stored format and tensors.
3. **Resolve.** Each part is looked up in the one kernel library for the target GPU. The result
   is a table: resolved, or one of the named gaps in section 4.
4. **Serve or refuse.** All resolved: convert and serve, with every shared strategy on (chunked
   prefill, lanes, KV reuse, drafting, RDMA exchange, packed weights). Anything unresolved: the
   load fails and prints the same table.

Adding a model is then one of three things, in rising cost:

| Case | Work |
| --- | --- |
| Every part resolves | none: scan, convert, serve |
| A kernel exists but is not built at this shape or target | one line in the kernel build list, rebuild |
| A part no model has had | one new part kind in the shared library: host body, device body, kernel |

No case adds a crate, a `match` arm on a model name, or a serving path.

## 2. What we take from vLLM, and what we do not

Take the layering:

| vLLM | Here |
| --- | --- |
| OpenAI server, chat template, tool and reasoning parsers | `sparkflash-api`, driven by the checkpoint's own template |
| Engine and scheduler | DeepSeek's `Batch` and tick (lanes, chunks, KV reuse, drafting) |
| Model executor | DeepSeek's forward, generalized to walk the model description (5.2) |
| Layers library | `sparkflash-ops` part kinds over `kernels/` |
| Multimodal processors registry | `sparkflash-vision` (and audio), driven by `preprocessor_config.json` |
| Model registry | none: the description is built from the checkpoint |

Do not take: one source file a model family (vLLM has one Python class a family; this plan has
none), graph capture, JIT, CUDA graphs. Kernels stay hand-written CUDA, built ahead of time,
launched from Rust.

## 3. Three corrections to the idea as stated

These are the places where the idea needed a decision. The decisions are in section 8.

### 3.1 A config does not state the wiring

`config.json` gives sizes and switches. It does not say whether a block norms before or after its
branch, where the rotary step applies, or how a merger is ordered. That lives in the producer's
Python. vLLM has the same gap, which is why it keeps a class a family.

So the description has three sources, in this order:

1. **Config keys**, read structurally (never by `model_type` string).
2. **The tensor index**: names and shapes show which components exist (`q_norm.weight`,
   `gate_proj`, a bias or none, a position table or none).
3. **A layout file** for what neither states. Data, not code: a short declarative file that
   names the block order for a stack and maps tensor-name patterns to roles. It ships under
   `models/` for known layouts and can be dropped into `~/.cache/sparkflash/layouts/`.

Most new releases reuse a known layout and need no file. A new layout needs the file and nothing
else. The scan says which case it is.

Rule for implementers: if a fact about a model is in Rust, it is a bug. It belongs in the
checkpoint, or in a layout file.

### 3.2 "The only copy" needs kernels that are not compiled for one model

Today the kernel library is organized by model. The catalogue has 47 entries pinned to one hidden
width (5120, 4096, 2560, 4608), images are named `Deepseek`, `Qwen4*`, `Glm*`, `Inkling`, `Dory`,
and sources carry widths as constants (`OV_DIM 1024`, `OV_INTER 2816`, `OV_ROW 5120`). There are
several MoE, GEMM, attention and norm bodies because each model brought its own.

Without a JIT, one copy means:

* A shape is a template argument, so the compiler still sees constants. Use as many template
  arguments as the kernel needs (widths, heads, experts, top-k, columns a rank, stored format,
  tile sizes). The count does not matter. What matters is that the body is the optimal one and
  that every model instantiates the same body, so a gain found on one model reaches the others.
* The instances are listed in one build file (`kernels/instances.toml`). A new shape is a line
  there and a rebuild. A size becomes a launch argument only where measurement shows the
  constant buys nothing.
* The catalogue's claims are generated from that build file and the kernel sources, so it cannot
  claim a shape that was not built or hide one that was.

"One copy" is one per operation, stored format and pass. NVFP4, FP8-tile, MXFP8 and BF16 experts
are different arithmetic, not duplicates. A prompt-pass tile GEMM and a one-row GEMV are both
kept. The resolver picks among them from a measured speed table, as tensor-core decode does now.

### 3.3 Consolidating kernels moves output bits, once

Two bodies for the same operation sum in different orders. When a model moves from its own body
to the shared one, its outputs change in the last bits. The owner's rule for this effort:

* A one-time bit change is always permitted for a move onto a shared body. It is declared (the
  numerics registry gets the new version), never silent.
* The check for each such change is: the outputs are coherent (the served gate's prompts read
  correctly, greedy and sampled), and the drift against the build before the move is inside a
  stated tolerance (teacher-forced logits over the fixed prompt set). The tolerance is written
  down before the move, from what a reorder of that operation can cost, not chosen afterwards.
* A change that claims the same arithmetic still has to be bit-equal. The bit record and the
  launch trace stay the gate for everything that is not a declared move.
* A move ships only if the model is not slower at any measured row count. If a model's old body
  is faster than the base's, the faster body becomes the shared one and DeepSeek takes the
  one-time change, under the same check.

## 4. The scan report

One function produces it. The console page, `onboard`, and the load-time refusal all print it.
Each row is a part instance (or a group of identical ones) with one status:

| Status | Meaning | Fix |
| --- | --- | --- |
| resolved | kernel chosen for both passes | none |
| not built | a kernel covers the operation, not this shape or format | build-list line |
| no kernel on this target | the part kind exists, the target GPU has no body | port the kernel |
| no part kind | the description names an operation the library lacks | new part kind |
| unknown key | a config key nothing read | teach the reader, or mark it ignorable with a reason |
| unclaimed tensor | a tensor no part took | a missing stack or layout entry |
| unknown layout | wiring not determined | layout file |

Two checks make the report trustworthy and are the general form of the vision-tower complaint:

* **Every tensor is claimed.** A tensor in the index that no planned part takes is a load error.
  A GLM checkpoint's 347 `model.visual.*` tensors could not be dropped without a word.
* **Every config key is consumed** or listed as ignorable with a reason. Today
  `unmodelled_features` checks two keys.

A stack can be turned off by name (`SPARKFLASH_VISION=off`, the profile's `diagnostics.vision`).
The boot line then says the stack is off, and its tensors count as claimed-but-unloaded. Off is
always a stated choice, never the default.

## 5. Workstreams

Sizes are relative: S, M, L, XL.

### 5.1 The description is complete and is data (L)

`sparkflash_model::Architecture` becomes the full description, not a summary an arm completes.

* Stacks. The vision tower, audio tower and draft head are stacks of the same part kinds as the
  language model (norm, attention, MLP, projection) plus a few of their own (patch embedding,
  merger). `Part::Vision` as one opaque part goes away. This is where the GLM tower starts: its
  stack is described, its parts resolve or are listed.
* Tensor roles. The description owns the map from part to tensors. Model crates' naming tables
  move into the layout files.
* State, split and precision per part come from the part kind and the stored format, not from an
  arm.
* The layout-file format, its reader, and layout files that reproduce the six released models.
  Gate: each description equals what its arm builds today.

### 5.2 One engine: DeepSeek's forward is the base (XL)

DeepSeek's forward (`sparkflash-deepseek-forward`) is the best implementation in the tree. It is
the base. The other models move onto it; it does not move onto anything.

What the base contributes, and every model then inherits:

* the scheduler (`Batch`, the tick, tick-ahead, the time-based decode share, preemption);
* lanes and lane memory (prefix adoption, the slide, growth to 32);
* bulk prefill with batched prompts across requests, and the exact row step;
* speculation and verify (`spec`, `draft`);
* the cross-rank reduce and the two-rail exchange;
* resident weights, staging and the kernel cache;
* its kernels, which are the starting "only copy" of each operation they cover.

What has to change in the base before anyone can move onto it. All of it is done in place, with
DeepSeek bit-equal and no slower at every step (tick trace identical, alternated A/B launches;
any step that costs DeepSeek speed is reverted and recorded):

1. **It reads the description.** Today the layer loop is DeepSeek's layer written out
   (`Layer0*` types, 22k lines in `tp4_forward.rs`). It becomes a loop over the description's
   blocks, calling part kinds. DeepSeek's layer is then one description among several, and its
   bodies (MLA with CSA2, the indexer, routed and shared experts, the mHC seams, Engram, the
   head) are the first part kinds in the shared library.
2. **Its shapes become template arguments** (3.2). Today the chain refuses anything but the
   released shape by name (world 4, 384 experts, top-6, width 5120, 576 columns a rank).
3. **It runs at one, two and four ranks.** It is built for four. Qwen serves at TP 1 on the
   laptop, and every other model has TP 1, 2 and 4 records.
4. **Stored format is an axis of each body.** DeepSeek's are MXFP8 dense and MXFP4 experts. GLM
   is FP8 tiles, Qwen NVFP4 and FP8, Dory BF16. The format's decode step is a template argument
   of the same MMA loop, not a second kernel.
5. **A seam for parts DeepSeek lacks.** The part contract built on sprint2 (`PartExec`: state,
   split, prompt body, token body, with enter and leave frames) is that seam. Five models already
   express their layers through it, which is what makes them movable. Their own parts (GDN and
   KDA recurrences, PLE, QSA, relative attention, short convolution, block residual, the nGPT
   norms) plug into the base through it unchanged.

Then the models move, one at a time. A move is: the model's description runs on the base; every
operation the base has uses the base's body (the model's one bit change, 3.3); the model's own
parts come across through the seam; the model's crate is deleted.

* **GLM first.** It is the closest to DeepSeek: MLA, an indexer, routed and shared experts, mHC,
  an MTP head. It exercises the format axis (FP8 tiles) and little else new.
* **Qwen**: recurrent state (GDN), PLE, the hyper mix, TP 1.
* **Inkling**: relative attention, short convolution, its drafter, its vision tower.
* **Dense**: the default for any model with only standard parts. Small once the above are done.
* **Dory last**: its recurrence ring is a second schedule, not a lane walk.

If a step cannot be made free for DeepSeek, stop, record the measurement, and bring it to the
owner. Aborting a step is acceptable; a slower DeepSeek is not.

What happens to `sparkflash-runtime`: its strategies that DeepSeek's engine already has better
are retired as models leave them. What only it has (the part contract, the description walk,
rewind points for recurrent state) moves into the base.

### 5.3 One serving path (M)

* One server construction. Today there are nine sites and `Route::Pack`, `TYPED_PACKS`, `ARMS`
  dispatch on the model. All go.
* Prompt format from the checkpoint. The tree has no template engine; each model hand-writes its
  codec. Evaluate the checkpoint's `chat_template` at run time, and keep a small registry of
  tool-call and reasoning parsers selected by what the template emits.
* Image and audio preprocessing from `preprocessor_config.json`.
* Sampling defaults from `generation_config.json`, then DeepSeek's defaults.
* Draft heads (MTP, DFlash, DSpark) are stacks found by the scan, attached by the runtime.

### 5.4 Weights follow the description (M)

Conversion, TP shards and the packed layout are computed from the description's tensor roles. No
per-family converter. The cache home is already central.

### 5.5 One kernel library (L, mostly a consequence of 5.2)

Consolidation is not a separate campaign. It happens as each model moves: the model takes the
base's body for every shared operation, and its old body is deleted.

* Before each move, measure the model's body against the base's at every row count and format.
  The faster one is kept and becomes the shared body (3.3).
* Reorganize `kernels/` by operation as bodies are shared. No model name in a file, symbol or
  image.
* The build list and the generated catalogue (3.2).
* Operations DeepSeek does not have keep the single body their model brought, moved to the
  shared library and made template-generic.
* Targets: the catalogue is already keyed by target. A new GPU family is a port of the library.
  The scan's "no kernel on this target" rows are the port's work list.

### 5.6 Proof without a model crate (M)

The scan proves every piece has a kernel. It cannot prove the wiring is right. Without this
workstream, automatic pick-up produces models that run and are wrong.

* Reference capture, for a new model: a script that runs the producer's reference (transformers)
  on a fixed prompt set and stores logits. A new model is accepted only when the engine matches
  inside a stated tolerance. It can run on any machine that holds the weights.
* Drift check, for a move (3.3): the same prompt set, teacher-forced, the build before against
  the build after, plus the coherence read.
* Generated conformance: a tiny synthetic checkpoint at the description's shapes, each part's
  device body against its host body, bit records, launch traces.
* The served gate and the speed floors, run from the description.

### 5.7 The console flow (S to M, after 5.1)

Paste a Hub id. The console fetches the config and the tensor index (headers only), shows the
section 4 report per target and per TP, and offers convert and serve when it is clean. When it
is not, each row is a work item an implementer can take as written: operation, shapes, format,
pass, target.

## 6. Order and milestones

1. **The scan tells the truth.** 5.1's stacks and the two coverage checks, 5.7's report. First
   visible result: loading GLM with its vision tower fails with the list of tower parts that have
   no kernel. Bit-equal. Independent of everything below, so it can start now.

   The step is split into three deliveries:

   * **1a. The scan (delivered).** The report, the vision stack, the tensor index and the
     config-key coverage. It exposed the gaps the next two deliveries close.
   * **1b. The layouts (this work).** The layout-file format, its reader, and the worked
     GLM-5.3-Flash layout: GLM's tensor plan, config keys and vision wiring move out of Rust
     into data. Acceptance is the layout plan equalling the retired planner at TP 1, 2 and 4,
     and the real checkpoint scanning with no unknown-layout or unclaimed-tensor row. Only the
     layout changes here; no execution, output bits or kernels move.
   * **1c. The gate (held).** The serve gate and the off switch wait until DeepSeek and Inkling
     scan clean with vision on. Those two checkpoints still describe tower facts no layout
     states and leave `model.visual.*` tensors unclaimed; a gate now would refuse them for
     reasons the scan cannot yet fix. The gate is not merged early.
2. **The base reads a description.** 5.2 steps 1 and 2 on DeepSeek alone. DeepSeek bit-equal,
   same tick trace, same speed. This is the step that decides whether the plan holds; if it
   cannot be done for free, stop here.
3. **The base runs at TP 1 and 2 and takes other formats.** 5.2 steps 3 to 5, proved on tiny
   synthetic checkpoints before a real model moves.
4. **GLM runs on the base and its crate is deleted.** Its one bit change, checked as in 3.3. Its
   vision tower runs as a stack here.
5. **Qwen, Inkling, dense, Dory**, one at a time, each with its own bit change and A/B.
6. **Serving and weights have no model dispatch.** 5.3 and 5.4. Can run beside 4 and 5.
7. **Demonstration:** a model the owner picks, onboarded with zero Rust unless it has a new part
   kind.

Work that can run in parallel from the start: milestone 1, the layout-file format (5.1), the
reference-capture and drift scripts (5.6), and the chat-template engine (5.3). Milestones 2 and 3
are one line of work in one crate and should have one owner at a time.

## 7. Rules that hold the line

Add each as a test in `sparkflash-ops/tests/one_stack.rs` as soon as it can pass, and never
relax one.

* No crate is named for a model. (Starts as a shrinking allow-list.)
* No comparison against a family or `model_type` string outside the layout reader.
* No model name in a kernel file, symbol or image name.
* No model width as a kernel constant.
* Catalogue claims are generated, not written.
* Every refusal names the config key or tensor, the part, the shape, and what would fix it.
* Every option is applied or refused by name.

## 8. The owner's decisions (2026-11-21)

1. **Bits.** A one-time change is always permitted for a move onto a shared body. The check is
   coherent output and drift inside an expected tolerance. DeepSeek is included when another
   body is the faster one.
2. **Layout files.** Yes: a vLLM-style configuration solution for what the config and tensors do
   not state.
3. **Shapes.** A rebuild is fine. Shapes should be template arguments wherever that keeps the
   body optimal; many arguments are fine. The goal is one optimal, shared body, so that
   optimizing it for one model helps the rest.
4. **DeepSeek.** Its forward is the base. Other models move onto it. Abort a step if it shows a
   problem.
5. **Reference runs.** Any machine that holds the weights may run the producer's Python.
