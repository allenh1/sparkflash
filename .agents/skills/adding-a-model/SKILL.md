---
name: adding-a-model
description: Bring a new model up on SparkFlash from its checkpoint, with no model crate. Use when onboarding a Hub checkpoint, reading a scan report, writing or fixing a layout file, adding a kernel instance or a new part kind, porting to a new GPU target, proving a model's numerics, or moving a model onto a shared body.
---

<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Adding a model

This skill is the process once plan 05 ([docs/engine/05-runtime-models.md](../../../docs/engine/05-runtime-models.md))
is complete: no model crates, no arms, one engine (DeepSeek's forward, "the base") that walks a
description built at load from the checkpoint. Section 17 says which stages run in the tree
today. Follow the stages in order. Each one has an entry condition, numbered steps and an exit
gate. Do not start a stage until the previous gate is met, and do not mark a gate met without
the evidence it names.

## 0. What adding a model is

A model is a checkpoint directory. Nothing in the source tree names it. The engine scans the
checkpoint, builds a description (stacks, blocks, parts, shapes, stored formats, tensors),
resolves every part against the one kernel library for the target GPU, and then either serves
with every shared strategy on or refuses with the same table.

The work falls into exactly one of these cases, in rising cost. Most models need more than
one: take every row the scan lists.

| Case | Scan row | The work | Stage |
| --- | --- | --- | --- |
| Every part resolves on a known layout | all `resolved` | none: prove, convert, serve | 6-10 |
| The wiring is not determined | `unknown layout`, `unclaimed tensor`, `unknown key` | a layout file (data) | 2 |
| A kernel covers the operation, not this shape or format | `not built` | lines in `kernels/instances.toml`, rebuild | 3 |
| The part kind exists, this GPU has no body | `no kernel on this target` | port the body | 5 |
| An operation no model has had | `no part kind` | one shared part kind: host body, device body, kernel | 4 |

None of these adds:

* a crate, or a module, named for the model;
* a `match` arm, `if`, or string comparison on a model name or `model_type` (outside a layout's
  `[match]`);
* a serving path, route, pack, codec or converter for one model;
* a model name in a kernel file, symbol, image or cubin;
* a model's width, head count, expert count or any other geometry as a constant in Rust or CUDA.

If you find yourself writing any of these, stop. The fact belongs in the checkpoint, in a layout
file, or in a template argument listed in `kernels/instances.toml`.

## 1. The rules

These hold at every stage. Each one has an enforcer; a rule without one is a test to write
before the work that needs it, not a rule to skip.

1. **A model fact in Rust is a bug.** Shapes, wiring, epsilons, clamps, router rules, head
   folds, casts and splits come from the config, the tensor index, or the layout file. The
   base's value table reads a fact only when a part that reads it is planned. Held by
   `crates/sparkflash-ops/tests/one_stack.rs` (`the_scan_vocabulary_names_no_model_family_or_crate`
   and the rules of plan 05 section 7).
2. **The model computes what its producer computes, in the producer's types** (owner's
   decision D124, 2026-11-25).
   * The function is the model's. A part the model was not trained with is never substituted
     for one of its own. Example: CSA2 compression is DeepSeek V4.1's alone, so no other model
     gets it, and no other model's indexer is reshaped into it. No sink, window, rotation,
     grouped output, router rule or norm the producer does not have.
   * Each value takes the smallest type that is faithful to the producer's calculus. That is
     the type the producer computes it in: the config's `dtype` (usually BF16) for streams and
     module outputs, f32 only where the producer casts to f32 (each cast cited to the
     producer's source line in the layout), and the release's own activation recipe into a
     quantised linear (`quantization_config.activation_scheme`). Never wider "to be safe".
     Never narrower than the producer, except for a serving trade the profile chooses and the
     numerics registry declares (for example an E4M3 latent cache).
   * Where the base's body computes a different function, the model gets its own instance of
     the template, or its own part kind. A model fact is never bent to fit a body.
3. **Nothing changes for the released DeepSeek** unless it is declared under rule 4. Any change
   to a shared body is held to `scripts/open-manifest-ab.py` (manifest, launch trace, tick
   trace, speed) on the four-node group. A step that costs DeepSeek speed and cannot be
   recovered is reverted, measured and brought to the owner.
4. **Bits move only when declared.** A change that claims the same arithmetic is bit-equal (bit
   record, launch trace). A move of an existing model onto a shared body may change bits once,
   declared as a new numerics version with a tolerance written before the move (section 13). A
   new model's first served arithmetic is its version 1.
5. **One copy per operation, stored format and pass.** A new shape is a template instance of the
   existing body, never a second source. When two bodies compute the same function, the faster
   one at every measured row count becomes the shared body. A body faster for one model and
   slower for another stays two instances of one template, chosen by the speed table.
6. **Refuse by name, with a configurable opt-out.** Every option is applied or refused. A
   refusal names the config key or tensor, the part, the shape, and what would fix it. An
   unclaimed tensor or an unread config key is a load error, not a warning. A stack can be
   turned off only by an explicit switch (for example `SPARKFLASH_VISION=off`), and the boot
   line says so. Off is never the default.
7. **Reuse and centralize.** Before adding a helper, a parser, a launcher or a kernel, search
   for the shared one (`sparkflash-ops`, `sparkflash-runtime`, `sparkflash-weights`,
   `sparkflash-kernels`, `kernels/`). Extend it with an argument. A parallel copy is refused in
   review.
8. **The ratchets only go down.** The `one_stack` allow-lists, the CLI line limit, the kernel
   instance count for retired bodies and the list of refusals a model carries never grow to let
   a model in.
9. **No part is deferred because it is hard.** A model's bring-up takes all its stacks at once:
   language, draft head, vision, audio. A stack that cannot run yet is refused by name and
   listed as open work. It is never silently dropped.
10. **Outside-derived code carries attribution.** Any body, constant table or algorithm adapted
    from another project gets a header naming the source and its license, and a `NOTICE` entry,
    with the license verified before the commit.

## 2. Stage 0: intake

Entry: someone names a model.

1. Record the identity in the work log (the model's doc under `docs/models/`, created now):
   * the Hub id `ORG/REPO` and the exact revision hash you will scan;
   * the license, read from the model card, and whether it permits what we do;
   * the producer reference: the modelling source that defines the model's arithmetic. Usually
     `transformers/models/<type>/modeling_<type>.py` at a named transformers commit. If the
     checkpoint ships its own modelling code, that file at that revision. If neither exists,
     say so: the layout's sourcing rule (stage 2) cannot be met for the facts only code
     states, and those facts stay `unknown layout` until the owner decides;
   * the release's stored formats and `quantization_config`;
   * the targets (GPU family, `sm_*`) and the tensor-parallel widths it must serve at.
2. Fetch the config, tokenizer files, `generation_config.json`, `preprocessor_config.json` (if
   any), the chat template, and the safetensors headers only. Do not download weights yet.
   `scripts/dump_safetensors_headers.py` reads headers; the console and `sparkflash onboard` do
   ranged Hub reads.
3. Keep private data out of every request and file: no real names, hostnames or addresses in
   repo text (docs use `node0`-`node3` and TEST-NET addresses).

Exit: the model doc names the revision, license, producer reference and targets.

## 3. Stage 1: the scan

Entry: stage 0's exit.

1. Run the scan for every target and width you recorded:

   ```sh
   sparkflash onboard ORG/REPO
   sparkflash onboard ORG/REPO --config path/to/config.json   # a local document
   ```

   The web console prints the same report when the Hub id is pasted. One function produces the
   report for the console, `onboard` and the load-time refusal, so they cannot disagree.
2. Read every row. Each row is a part instance (or a group of identical ones) with one status:

   | Status | Meaning | Go to |
   | --- | --- | --- |
   | `resolved` | a kernel is chosen for both passes (prompt rows and token rows) | nothing |
   | `not built` | a kernel covers the operation but not this shape or format | stage 3 |
   | `no kernel on this target` | the part kind exists, this GPU has no body | stage 5 |
   | `no part kind` | the description names an operation the library lacks | stage 4 |
   | `unknown key` | a config key nothing read | stage 2 (fact or ignore with reason) |
   | `unclaimed tensor` | a tensor no planned part takes | stage 2 (missing stack or tensor entry) |
   | `unknown layout` | the wiring is not determined | stage 2 |

   Also read the `carried facts` line (facts a layout records that the reader does not apply
   yet), the fit at each width (bytes a rank, KV a position, the window that fits), and
   `tensors not checked` (no header was read; the report is then not clean).
3. Triage. Write the work list into the model doc: one line per non-`resolved` row, with the
   operation, shapes, stored format, pass, target and the stage that closes it. Group rows that
   share a fix. A row is a work item an implementer can take as written.
4. Check the triage against rule 2 before any code. For each `no part kind` row, read the
   producer's code for that operation and confirm it is not an existing part kind with
   different arguments (stage 4, step 1). For each `resolved` row, confirm the resolved part
   computes the producer's function and not a neighbouring one: same norm, same activation,
   same router rule, same casts. A `resolved` row that resolves to the wrong function is a
   layout bug (a fact is missing), not a pass.

Exit: the model doc holds the full scan report for every target and width, and every row has a
stage.

## 4. Stage 2: the layout file

Entry: the scan shows `unknown layout`, `unclaimed tensor` or `unknown key`, or a resolved part
computes the wrong function because a wiring fact is missing.

A layout states what neither the config nor the tensor index states: block order, per-part
facts, tensor name templates, and the config keys the description reads. It is data. It says
nothing about kernels or a shape the config already gives. The format is in
[models/layouts/README.md](../../../models/layouts/README.md); the reader is
`sparkflash-model::layout`.

1. **Look for a layout that already matches.** Most releases reuse a known layout. Check
   `models/layouts/*.toml` `[match]` tables. A new quantisation of a known family takes a file
   narrowed by `match.quantization`, or the same file if the tensors agree.
2. **Read the producer reference end to end** before writing a line. For each module, note:
   the order of operations in a block (pre- or post-norm, where the residual adds, where rotary
   applies); every epsilon, clamp, scale and bias; every `.float()`, `.to(dtype)` and
   `dtype=torch.float32` (these are the f32 casts); the router's scoring, bias, normalisation
   and scale; the head's fold; what the draft head and towers do. Keep the file and line for
   each.
3. **Start the file.** Shipped layouts go under `models/layouts/<layout-id>.toml` (named by the
   layout, not the marketing name). A node-local override goes in
   `~/.cache/sparkflash/layouts/` and must name the shipped `id` it replaces with `replaces`.
   ```toml
   schema = 1
   id = "<layout-id>"
   [match]
   architectures = ["<ProducerClass>"]
   model_types = ["<producer_type>"]
   ```
   `[match]` is the only place in the tree that compares a family string.
4. **Account for every config key.** One `[[keys]]` entry per key the shared reader does not
   already read: `fact = "..."` to read it, or `ignore = "<reason>"`. A reason says why the
   value cannot change what we compute (for example "training only: dropout is off at
   inference"). Only writer metadata (`transformers_version`, `torch_dtype`, `_commit_hash` and
   their kind) is on the shared ignore list. A model or training switch is ignored only in the
   layout that knows why.
5. **Declare the stacks.** `[stacks.language]`, and `[stacks.draft]`, `[stacks.vision]` and
   `[stacks.audio]` when the checkpoint has them, each with an `order` over the part
   vocabulary. A draft head (MTP, a block drafter) and a tower are stacks of ordinary part kinds
   plus their own few (patch embedding, merger), never one opaque part.
6. **State each part's facts**, one `[[stacks.<stack>.parts]]` per part with
   `[[stacks.<stack>.parts.facts]]` entries. Every fact has a comment citing the producer file
   and line, or the config path, or the tensor that proves it. The facts include:
   * the function: the MLA form, the pool and tail rule of an indexer, the router rule, the head
     fold, the collapse a residual mixer uses;
   * the types: each f32 cast as its own fact (`accumulate = "f32"`, `softmax = "f32"`,
     `state = "f32"`), and the stream update in `{ key = "text_config.dtype" }`;
   * constants no config key states (for example a LayerNorm built at `eps 1e-6`);
   * each recurrent block's tensor-parallel split. The split is stated, never inferred from
     what a stack contains.
7. **Map every tensor.** One `[[tensor]]` per name template: `name` (with `{layer}` and
   `{expression}` arithmetic), `part`, `role`, `dtype`, `shape` (expressions over facts, never a
   repeated literal), `split` (`whole`, `rows`, `columns`, `kv_rows`, `vector`, `conv`,
   `expert_rows`, `expert_columns`, `expert_axis`, `tiles`, `tile_rows`), `tile` for tiled
   splits, `repeat`, and `when`. A tensor the engine must not load (a duplicate the producer
   keeps for its converter) is `unloaded = "<reason>"`. Use `dtype = "dense"` or `"expert"` for
   the checkpoint's quantised formats so the scale tensor follows.
8. **Leave out what you cannot source.** A fact with no citation is left out, and the scan keeps
   that row `unknown layout`. Do not guess. Bring the open fact to the owner with what the
   producer code shows and what it leaves undetermined.
9. **Rescan** until the report has no `unknown layout`, `unclaimed tensor` or `unknown key`
   rows. Every tensor in the index is claimed by a planned part or listed `unloaded` with a
   reason; every config key is read or ignored with a reason.
10. **Prove the plan.** The layout's checkpoint plan (names, shapes, dtypes, splits) at TP 1, 2
    and 4 must equal the tensor index and the shard rules the producer implies:
    * every tensor in the index appears once with its stored shape and dtype;
    * each rank's shard shapes divide evenly, or the width is refused by name at that TP;
    * the helper is `crates/sparkflash-ops/tests/layout_equality.rs`. Add the model's real
      config (headers only) as a fixture under `models/` or the test tree, never by
      `include_str!` from a crate.
11. **Self-review against rule 2.** Walk the producer's forward once more with the layout open.
    For each operation: is the function the producer's, and is each value in the producer's
    type? Write the answer into the model doc as a table (part, function source line, types,
    layout fact).

Exit: the scan shows only `resolved`, `not built`, `no kernel on this target` and `no part kind`
rows; the layout equality test passes at TP 1, 2 and 4; the rule-2 table is in the model doc.

## 5. Stage 3: kernel instances

Entry: the scan shows `not built` rows.

A shape is a template argument, so the compiler still sees constants. Instances are listed in
one build file, `kernels/instances.toml`; the catalogue's shape rules are generated from it
(`crates/sparkflash-ops/build.rs`), so a shape cannot be claimed without being built.

1. For each `not built` row, find the body that covers the operation and format: the row names
   the entry symbol and the missing arguments.
2. Add one `[instance]` table per entry symbol and shape a rank sees: `source`, `entry`,
   `cubin` (only when it joins a multi-symbol cubin), `family`, the family's shape fields,
   `format` (required), and `catalogue` when the instance is claimable. Compute the per-rank
   shapes from the description at each width (heads a rank, columns a rank, experts a rank).
3. Validate and rebuild:
   ```sh
   scripts/instances.py --check
   scripts/build-cubins.sh
   ```
   Build on the build host from the pushed branch, not from a copied tree. Builds use a bounded
   job count.
4. Rescan. The rows move to `resolved`, or the scan names the next missing argument.
5. When an operation has more than one body (a prompt tile GEMM and a one-row GEMV, a
   tensor-core and a row body), the resolver picks from the measured speed table. Measure the new
   instance at every row count the model serves (1, 4, 8, prompt chunks) and record it. A new
   instance never changes another model's choice without its own measurement.
6. Check that nothing else moved: the bit records and launch traces of every other model that
   uses the same source are unchanged.

Exit: no `not built` rows; the speed table has the new instances; other models' records are
unchanged.

## 6. Stage 4: a new part kind

Entry: the scan shows `no part kind`, or stage 2's rule-2 review found a resolved part that
computes a different function from the producer's.

1. **Prove it is new.** Read the producer's code for the operation and the shared library's
   nearest part kinds (`sparkflash-ops` roles and `parts`, `docs/parts-catalogue.md`). Decide
   which it is:
   * the same function with different sizes or formats: not a new part kind, an instance
     (stage 3);
   * the same function with a different variant (a bias or none, a different epsilon, a gate,
     a different fold): an argument of the existing body or its template, never a copy;
   * a different function: a new part kind.
   Write the decision and its evidence (producer lines, the existing body's lines) into the
   model doc.
2. **Name it by the operation**, never by the model (`kda_scan`, `pooled_indexer`,
   `manifold_hyper_connection`). Add it to the description's part vocabulary, so a layout can
   name it.
3. **Write the host body first.** It lives in the shared host references in `sparkflash-weights`
   (beside `gqa.rs`, `linear_attention.rs`, `mhc.rs`). It follows the producer line by line, in
   the producer's types (BF16 where the producer is BF16; f32 exactly where it casts). It is the
   reference every device body and every proof compares against.
4. **Write the device body on the shared pass context.** Implement `PartExec`
   (`sparkflash-runtime/src/part.rs`) with no defaults:
   * `state`: the lane regions it adds. Positional regions (addressed by position: K/V,
     latents, index keys) keep every lane shortcut: adoption at any boundary, the slide, the
     early-row decoder skip. Recurrent regions (a conv history, a scan state) make the engine
     save at the prompt's end, snapshot each verify row, commit what is kept, adopt only at a
     saved point and disable the decoder skip. State the stored type of each region (rule 2).
   * `split`: `Summed` when a rank leaves a partial the group sums through the base's reduce;
     `Replicated` when every rank computes it whole. The tensor shard rule comes from the
     layout's `split`, not from code.
   * `rows`: the prompt body. It reads normed rows at `pass.input()` and leaves the branch's
     rows at `pass.output()`. It must take interleaved prompt chunks (a recurrence resumes from
     its region), or the scan says the stack runs one chunk at a time.
   * `token`: the token body (`self.rows(pass)` when they are the same).
   A foreign part sees the pass context and nothing of the engine. The base decides at open,
   block by block, whether a block runs its fused native halves or lands the residual and calls
   parts.
5. **Write the kernel** under `kernels/`, named by operation, with every geometry a template
   argument and every instance in `kernels/instances.toml` (stage 3). Load it through the
   shared image loader and launch it on the walk's stream. A plan fixes shapes, precision,
   layout and stream at open: no allocation and no host synchronization at launch. A missing
   cubin skips the device test loudly; it never falls back to different arithmetic.
6. **Make the catalogue claim it.** The claim is generated from the instance list and the
   source. If the operation is a matrix product, it needs a tensor-core body for both passes or
   a `WAIVERS` line naming the gap and its measurement.
7. **Refusals.** Every input the body does not handle (a world that does not divide its heads, a
   format it does not decode, chunked prompts it cannot resume) is refused at open by name.
8. **Unit-test the part alone**: the device body against the host body on random inputs at the
   model's per-rank shapes and at a small shape; the split at worlds 1, 2 and 4 (partials sum to
   the whole); state save, snapshot, commit and rewind for a recurrent part.
9. **If it overlaps an existing body**, measure both at every row count and position that
   matters (attention: 2k, 13k, 100k, 1M) and keep the faster as the shared body (rule 5,
   section 13).

Exit: the scan resolves the part on every target listed; its unit tests pass; other models'
bit records are unchanged.

## 7. Stage 5: a new target

Entry: `no kernel on this target` rows.

1. The scan's rows for that target are the port's work list. Take the whole list, not the rows
   one model needs.
2. Port each body as the same template with the same arguments. Same arithmetic, same
   accumulation order where the hardware allows; where it cannot, declare the difference as
   that target's numerics version.
3. Add the target's instances to `kernels/instances.toml`, rebuild, rescan.

Exit: no `no kernel on this target` rows for the target.

## 8. Stage 6: the synthetic proof

Entry: the scan is all `resolved` for the model at every width.

The scan proves every piece has a kernel. It cannot prove the wiring is right. A model that
runs and is wrong is the failure this stage exists to catch.

1. **Write the synthetic checkpoint.** Add a variant to the shared writer
   (`scripts/second-shape.py`): a tiny random checkpoint with the description's structure
   (every stack, every block type, every part, the stored formats, the draft head, the towers)
   at small widths. Choose widths so every tensor-parallel split the release uses is exercised
   at worlds 2 and 4: heads divisible by four, intermediates on whole scale tiles (128 for FP8
   tiles). A fixture that plans no split at a world proves nothing at that world.
2. **The host reference** runs the variant on the shared host references at each world's own
   split, and writes the expected ids and each step's alternatives.
3. **Run the device arm** at worlds 1, 2 and 4. The device arm is the bar, not the host arm
   (`SPARKFLASH_LAYER_DEVICE=0` launches no kernel and only proves plan, names and split):
   ```sh
   scripts/second-shape.py run --device --released DIR --out DIR --binary BIN \
       --writer BIN --reference BIN --released-cubins DIR --world 1,2,4
   ```
   Worlds 2 and 4 run on the four-node group. Check free memory on every node first.
4. **The bar is exact ids.** Two relaxations exist, each only with the owner's approval recorded
   in the model doc and in the variant's table in the script, with its reason:
   * the tie band (`TIE_BAND_VARIANTS`, `--tie-band B`): a device kernel the host cannot
     reproduce (for example fast-math `__expf`) flips a near-tie. The band is measured: twice
     the largest gap the worlds show, rounded up. Every forced tie is printed with its gap.
   * the seam gate (`SEAM_GATED`): the fixture's random routers or indexer carry near-ties that
     the stream's rounding flips on a few rows. The host writes one prompt's boundary table, each
     world's device group dumps that prompt's first prefill pass (`SPARKFLASH_ROWS_DUMP`,
     `SPARKFLASH_WARMUP=0`, forwarded to every rank), and each layer's median row distance must
     stay under a bar measured the same way. Ids are still compared and printed.
   A tie gate is a last resort. Try first to make the fixture's routers and selections decisive
   (larger margins) so exact ids hold.
5. **When ids miss, diagnose before relaxing anything.**
   1. Dump both sides' per-layer boundaries on the failing prompt.
   2. Compute each layer's per-row relative L2 distance: the median and the rows past ten times
      the median.
   3. A wiring fault moves every row by a share of one sublayer's effect (a large median, early).
      A near-tie flip moves a few rows, first at a router or selection layer, and spreads
      through attention.
   4. For the flipped rows, recompute the router's top-k margins and the indexer's selections
      from both sides' inputs. A flip at a margin of 1e-4 is a tie; a flip at a margin of 1e-1
      is a bug.
   5. Record the measurements in the model doc whatever the outcome.
6. **The conformance suite** passes on the variant (`sparkflash_runtime::conformance!` over the
   base's `ArmFixture`):
   * whole = chunked = token on a long prompt;
   * continued = cold, through a recurrent save point when the model has one;
   * rewound = cold;
   * drafted = plain, with each draft head the checkpoint has;
   * one lane = four lanes;
   * TP 1 = 2 = 4;
   * the bit record and the launch trace.
7. **A refused forward at world above 1** is written to the rank's evidence file
   (`tp4-token-r<rank>.json`). When a group never becomes ready, read those files first.

Exit: the variant passes at worlds 1, 2 and 4 on the device arm under exact ids or an
owner-approved gate; the conformance suite passes; the measurements are in the model doc.

## 9. Stage 7: the producer capture

Entry: stage 6's exit. The synthetic proof compares us with our own reading of the producer.
This stage checks that reading from outside it.

1. **Write the tolerance first.** Before any run, write in the model doc the distance you
   expect (logit relative L2, top-1 agreement, mean absolute NLL delta) and why: with the types
   aligned, it is what a reorder of the sums can cost.
2. **On the synthetic checkpoint:** run the producer's own classes (transformers at the
   revision the layout cites, in the config's `dtype`) on the tiny checkpoint, on the fixed
   prompt set, and store the logits. The shared host reference, on the producer's type map,
   must match inside the tolerance. This checks part by part: epsilons, pools, routers, folds,
   the activation recipe, where each f32 cast sits. Any machine that holds the weights may run
   the producer's Python.
3. **On the real checkpoint:** teacher-force a producer-side stack that serves it (transformers,
   or vLLM on the cluster) on the anchor prompts once, and record the distance. If no
   producer-side stack can be stood up, write that down; the synthetic capture then stands
   alone.
4. **For a tower:** the same capture on a fixed image (and video) set, through the producer's
   preprocessor and tower, inside a tolerance written first.
5. A miss past the tolerance is a changed function or a wrong type. Find it and fix the layout,
   the part or the body. Never widen the tolerance after the run.

Exit: both captures are inside their written tolerances and recorded.

## 10. Stage 8: the real checkpoint

Entry: stage 7's exit.

1. **Download** the weights to the shared cache. Confirm the size and the disk first; check
   whether the converted ranks already exist on the nodes.
2. **Convert.** Conversion, TP shards and the packed layout are computed from the description's
   tensor roles; there is no per-model converter. Ranks land in
   `~/.cache/sparkflash/converted/<model--hash>/`. Record the load time a rank.
3. **Check memory** on every node (enough headroom for weights, lanes and the window) and
   stop whatever holds the group, noting its connections. Restore it afterwards.
4. **Serve** at each width the fit allows. The boot line names the numerics entry and every
   stack that is on or off.
5. **Register the numerics.** Capture the reference with `scripts/numerics.py capture` at
   temperature 0 over the fixed prompts, write `docs/numerics/<model>/v1-*.md` (what the
   arithmetic is, the producer distance from stage 7, every serving trade), and add the
   `[[config]]` entries to `numerics/registry.toml` with the fingerprint. `scripts/numerics.py
   check` must then reproduce it bit for bit, and `scripts/numerics.py ci` must pass. Add the
   anchor, coverage and monitor rows.
6. **Run the served gate**: `scripts/served-gate.py all PORT --long FILE` (lanes, sampled
   lanes, interleaved prompts, kept turns). A failure every model shares is recorded as such,
   not as passing; a new failure blocks.
7. **Measure the speed table**: alone and four at once, greedy and sampled, prefill at a short
   and a long prompt, the first token at the largest window that fits, draft acceptance. Use
   alternated fresh boots. Record it in the model doc.
8. **Probe the long window**: a needle placed outside any local window (the selection must find
   it) answers correctly.
9. **Towers**: image (and video) chat answers on the served model, greedy and sampled.

Exit: the model serves at every width the fit allows; numerics v1 is registered and checks;
the served gate shows no new failure; the speed table and probes are recorded.

## 11. Stage 9: serving defaults

Done alongside stage 8; nothing here is model code.

1. **Prompt format** comes from the checkpoint's `chat_template`, evaluated at run time. Tool-call
   and reasoning output are parsed by the shared parser registry, selected by what the template
   emits. A template feature the engine does not evaluate is refused by name.
2. **Sampling defaults** come from `generation_config.json`, then DeepSeek's defaults. A profile
   departs from DeepSeek's defaults only with `measured = "docs/..."` pointing at the
   measurement.
3. **Image and audio preprocessing** comes from `preprocessor_config.json`.
4. **Draft heads** (MTP, block drafters) are stacks the scan found; the runtime attaches them.
   The speculator's depth is the shared adaptive one.
5. **Every switch** is a typed profile field with one environment alias, applied or refused by
   name. A switch of a retired path is refused and names its replacement.

## 12. Stage 10: the DeepSeek guard

Entry: any stage touched a shared body, the base, the scheduler or a shared kernel.

1. Run `scripts/open-manifest-ab.py ab` with the build before and after, alternated, on the
   four-node group. The manifest, the launch trace and the tick trace are identical, and the
   speed is no lower at any measured row.
2. Smoke the released model's served prompts.
3. If DeepSeek moved, revert the step, record the measurement, and bring it to the owner. A
   slower DeepSeek is never shipped.
4. Leave the production DeepSeek build serving at the end of every window.

## 13. Moving an existing model onto a shared body

A model that already serves can change its bits once for a move onto a shared body.

1. **Write the bar before the move**, in the model doc and the plan: top-1 agreement and mean
   absolute NLL delta against the previous version's reference and the anchor, set from what a
   reorder of that operation can cost. Measure the share of each operation alone on the
   synthetic first, where the host reference runs both type maps.
2. **Measure both bodies** at every row count and position that model serves. The faster one is
   the shared body. If it is the other model's, DeepSeek (or that model) takes its own declared
   change under the same check, and is never slower.
3. **The check**: coherent output on the served gate's prompts, greedy and sampled; the producer
   comparison no further than the previous version's; teacher-forced drift inside the written
   bar.
4. **Declare it**: a new numerics version with its report, the owner's decision in
   `docs/DECISIONS.md`, the registry entry bumped. The bar is never raised after the run. If
   the synthetic shows the bar cannot be met, the number and its cause go to the owner before
   the move.
5. **Delete the old body.** No model keeps a private copy of an operation the library has.

## 14. Stage 11: landing

1. **Commits**: signed off (`git commit -s`), staged by path, one change each, messages that say
   what changed and what proved it. No co-author trailers for assistants.
2. **Docs**: the model doc (intake, scan, rule-2 table, decisions, measurements, numerics,
   speed), `docs/numerics/<model>/`, `docs/DECISIONS.md` for any owner decision, `NOTICE` for
   any attribution, and `docs/parts-catalogue.md` for a new part kind.
3. **Tests**: run the tests of the components you changed at each stage. Run the full battery
   once, at the end.
4. **Deploys** happen only on the owner's direction.

## 15. Stopping

Stop, record the measurement, and bring it to the owner when:

* a step cannot be made free for the released DeepSeek;
* a step can be made only by changing what the model computes or by taking a value below the
  producer's type;
* a gate can be met only by relaxing it (a tie band, a seam gate, a wider tolerance);
* a fact the wiring needs cannot be sourced;
* the model is slower on a shared body and the loss cannot be recovered.

Aborting a step is acceptable. Shipping a slower or less faithful model under the name of the
move is not.

## 16. The checklist

Copy it into the model doc and tick each line with its evidence (commit, file, measurement).

```text
[ ] 0  revision, license, producer reference, formats, targets, widths recorded
[ ] 1  scan report at every target and width in the model doc; every row has a stage
[ ] 1  every resolved part computes the producer's function (rule 2 check)
[ ] 2  layout: every key read or ignored with a reason; every tensor claimed or unloaded
[ ] 2  every fact cited to a producer line, config path or tensor
[ ] 2  layout equality at TP 1, 2, 4; rule-2 table (function, types) in the model doc
[ ] 3  instances added; scripts/instances.py --check; cubins rebuilt; speed table measured
[ ] 4  new part kinds named by operation; host body; PartExec (state, split, rows, token)
[ ] 4  kernel by operation, template arguments, instance list; refusals by name
[ ] 4  part unit tests; other models' bit records unchanged
[ ] 5  target port complete for the whole work list
[ ] 6  synthetic variant exercises every split at worlds 2 and 4
[ ] 6  device arm exact ids at worlds 1, 2, 4 (or an owner-approved gate, measured)
[ ] 6  conformance suite passes
[ ] 7  tolerances written first; producer capture on synthetic and real; towers
[ ] 8  converted from the description; numerics v1 registered; check and ci pass
[ ] 8  served gate: no new failure; speed table; long-window probe; tower chat
[ ] 9  template, parsers, sampling, preprocessing, drafters from the checkpoint
[ ] 10 open-manifest-ab: DeepSeek identical and no slower; production restored
[ ] 11 commits signed off; docs, DECISIONS, NOTICE; battery once at the end
```

## 17. Where the tree is today (2026-11-25)

The stages above describe the end of plan 05. Today:

| Stage | Today |
| --- | --- |
| 1 scan | runs (`sparkflash onboard`, the console); the load-time refusal gate is held until DeepSeek and Inkling scan clean with vision on (plan 05 step 1c) |
| 2 layouts | runs; shipped files for the released models under `models/layouts/` |
| 3 instances | runs (`kernels/instances.toml`, generated catalogue) |
| 4 part kinds | the seam runs on the base (milestone 3); models' own parts are being moved onto it (GLM: [08-glm-on-the-base.md](../../../docs/engine/08-glm-on-the-base.md)) |
| 6 synthetic | runs (`scripts/second-shape.py`, `conformance!`) |
| 7 capture | not yet a script (plan 05 section 5.6) |
| 8 convert and serve from the description | not yet: converted ranks from the description (5.4) and one serving path (5.3) are milestone 6 |
| 9 template engine and parser registry | not yet (milestone 6) |

Until milestone 6 lands, a released model still serves through its crate, and a new model that
must serve before then is the owner's decision. The crate-era process is in
[docs/adding-a-model.md](../../../docs/adding-a-model.md); do not use it to add a crate without
that decision.
