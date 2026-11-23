# Milestone 2: the base reads a description

Status: work breakdown, 2026-11-22. It details milestone 2 of
[05-runtime-models.md](05-runtime-models.md) (section 5.2, steps 1 and 2): DeepSeek's forward
stops being built from its own constants and is built from the description the scan resolves.
DeepSeek alone, at four ranks, with the same output bits and the same speed. This is the step
that decides whether plan 05 holds.

## 1. What the forward is today

Read from the tree at the start of this milestone (`crates/sparkflash-deepseek-forward`, 67k
lines). Line numbers are that tip's.

* **It does not open from the checkpoint.** The chain calls `ModelConfig::v41_flash()`
  (`sparkflash-cli/src/tp4_chain.rs:905`), which parses a JSON document embedded in the DeepSeek
  model crate. The forward reads 39 of its fields. It never sees `sparkflash_model::Architecture`,
  a layout file or the scan's plan, and it does not use `sparkflash-ops` at all.
* **The layer walk is already one loop.** The `Layer0*` types are a historical name: they are the
  results of `run_layer_*(layer, ..)` for every layer. There are three walks over the same
  bodies: the served one (`spec::serve` to `Batch` to `run_layer_rows` in `tp4_forward/rows.rs`,
  for token rows and for bulk prefill), a single-position loop (`run_layer`,
  `tp4_forward.rs:14589`), and the host reference arm beside it.
* **A layer's parts are fixed in code, its variation is not.** Compress ratio, KV and index
  source layers, the candidate source layer, Engram layers and the drafter's tap layers come from
  config arrays. A few do not: `DSPARK_AUX_LAYERS = [37, 38, 39]`, the encoder split at layer 21
  (from `sparkflash-kv`), a refusal unless the Engram layers equal a constant, `DECODER_LAYERS =
  40`, and 80 reduce calls a step.
* **Model shapes are constants.** `HIDDEN = 5120` has 101 uses, and the literal appears at 77
  more sites. Buffer sizes are computed from literals (heads 64 and 16 a rank, latent 512, window
  128, selected 512, 576 and 640 expert columns, vocabulary 32,320 a rank). The routed path
  refuses anything but four ranks, 384 experts, top-6, width 5120 and 576 columns a rank
  (`tp4_forward.rs:5107`), and top-6 there is a literal, not the config's value.
* **Tensor names are built in the crate.** About a dozen `format!` patterns
  (`layers.{layer}.attn.wq_a.weight` and the like) across `resident_weights.rs`,
  `tp4_forward.rs`, `layer_device.rs`, `engram_device_state.rs` and `draft.rs`.
* **Kernel names are string literals.** About 135 symbol strings spread over `DeviceLayer::open`
  and five other files. There is no table, and the catalogue the scan checks
  (`sparkflash-ops/src/parts.rs`) is a second description the forward never consults. Where the
  two disagree today, the scan is the one that is wrong (the router rule is looser than the
  kernel, the norm and Engram rules are tighter).
* **The kernels fix the same shapes again.** `kernels/m4_common_sm121.cuh` defines 23 model
  constants, and the width 5120 is declared again under eight other names in the `opt_*` sources.
  Most of DeepSeek's kernels take no runtime argument for these. There is no per-shape instance
  mechanism: `scripts/build-cubins.sh` compiles each source once with no defines, the forward
  loads the cubins from a directory at run time, and `baseline/kernel_baseline.txt` pins their
  hashes. Three sources already make instances in-source by template and entry macro
  (`opt_router`: 384 experts top-6 and 128 experts top-3).
* **Four ranks is assumed in places** (the MoE device path, the head and vocabulary split, the
  token envelope). That is milestone 3 and is not touched here.

## 2. The rule

**The engine opens from what the scan resolved, and from nothing else.** One object carries, for
each block, its part instances in order; for each part its geometry and facts, the tensors it
claims (name, shape, stored format, rank split), and the kernel entry the catalogue resolved for
it. `EnginePlan` (`sparkflash-ops/src/engine_plan.rs`) is most of this already. The scan prints
it; the forward opens from it. A row the scan shows resolved is a kernel the forward will launch,
by construction.

Consequences:

* the forward crate stops depending on `sparkflash-model-deepseek-v41` and on the embedded JSON;
* a model value, a tensor name or a kernel symbol written in the forward crate is a bug;
* the catalogue's DeepSeek entries become exactly the symbols the forward loads;
* nothing about a launch changes. This milestone changes where each value comes from, not what
  is launched, in what order, or with what arguments.

What does not change here: the fused seams stay as they are. A body that spans two parts (the
attention post fused with the FFN seam) is chosen when the block's adjacent parts are the ones it
covers, and a block it does not cover is refused at open by name. The seam for other models'
parts (`PartExec`) is milestone 3.

## 3. Proof

Every step is checked the same way, so the harness comes first.

* **Open manifest.** At open the forward records every model value it uses (name and value),
  every tensor it binds (name, shape, format, rank slice) and every kernel it loads (cubin,
  symbol, hash). Recorded once on the starting tip. Equal after every step.
* **Launch trace** (`SPARKFLASH_LAUNCH_TRACE`) and **tick trace** (`SPARKFLASH_TICK_TRACE`) on a
  replayed workload covering one request, several lanes, bulk prefill, drafting and an image.
  Equal after every step. An equal launch trace on equal weights is equal bits.
* **Speed.** Decode step at one and eight lanes and prefill rate, alternated A/B against the
  starting tip. Checked when a step touches code that runs every token; a step that costs speed
  and cannot be recovered is reverted and reported.
* **Kernels.** For a source whose shape moves from a header to an instance argument, the released
  instance's cubin hash is unchanged. Where the hash moves, the disassembly is compared; where
  that moves, the launch outputs are.
* **A second shape runs.** The milestone closes when a small DeepSeek-shaped synthetic checkpoint
  (different width, heads, experts, top-k and layer count, within the tile sizes) opens from its
  own description, runs through the same engine at four ranks on one host, and agrees with the
  host reference arm. This is the proof that no shape is left in code, and it is the first
  DeepSeek test that needs no cluster.

DeepSeek's real proofs need the four-node group with DeepSeek loaded. Batch them: one baseline,
then one comparison for each change that alters what is bound, loaded or launched.

**Result (2026-11-23).** The fixture beside the test is
`crates/sparkflash-deepseek-forward/tests/fixtures/second-shape/config.json`: 8 layers, 4,096
wide, 32 heads, 64 routed experts at top-4, compression ratios `[0,0,2,2,2,2,1,1]`, Engram
layers `[1,5]` and drafter taps `[3,7]` (two stages, top-2). Its tensors are written by
`sparkflash-model-deepseek-v41`'s own writer (`write-synthetic-checkpoint`, the released
formats) and its kernels by the second-shape rows in `kernels/instances.toml` (the same entry
symbols in a separate cubin, so the forward's handles keep their names). `scripts/second-shape.py
run` writes the checkpoint, runs the f32 host reference on three token-id prompts at temperature
0, starts the four ranks through `scripts/one-host-group.sh deepseek 4 ... --check ...` and
compares the greedy ids. All three prompts agree. The same plan opens at four ranks in
`crates/sparkflash-ops/tests/second_shape.rs` and the forward binds only tensors the writer
names in `crates/sparkflash-deepseek-forward/tests/second_shape.rs`.

The four-rank arm is the host arm here because the proof host has one GPU and the device arm's
per-layer collectives are NCCL enqueues, which need four devices; the device boot resolves the
same second shape through the catalogue and loads the second-shape cubins under the released
names, and `scripts/second-shape.py run --device` is the form for a four-GPU host. A second-shape
run is not baseline-pinned (the baseline pins the released bytes); it sets
`SPARKFLASH_UNPINNED_SECOND_SHAPE=1`, and the open manifest records every cubin it loads with its
sha256, which is that run's integrity record.

## 4. The work

### A. Harness and inventory (first, one owner)

* The open manifest (`SPARKFLASH_OPEN_MANIFEST`), the replay workload and the in-repo script
  `scripts/open-manifest-ab.py`, which records and compares manifest, launch trace, tick trace
  and speed.
* A rule test with pinned per-file counts that may only fall: reads of `ModelConfig`, uses of
  model constants and literals, tensor-name strings, kernel-symbol strings. Four-rank sites are
  counted on their own list, which milestone 3 drains.
* Optional: split `tp4_forward.rs` (22k lines) by area in commits that only move code, so the
  later work does not collide.

### B. The resolved plan is complete for DeepSeek (`sparkflash-model`, `sparkflash-ops`)

* Every one of the 39 values the forward reads is a fact or geometry field of a part instance,
  read from the checkpoint's `config.json` and the layout. Per-layer facts sit on that layer's
  part: compress ratio, KV and index source, candidate source, Engram membership and rows, tap
  layers, the encoder split.
* Each part instance carries the tensors it claims with rank splits.
* The catalogue's DeepSeek entries list the symbols the manifest shows, grouped by part.
* Done when the plan built from the real checkpoint reproduces the baseline manifest: same
  values, same tensor set, same symbols.

### C. The forward opens from the plan (`sparkflash-deepseek-forward`, one line of work)

Four changes, in this order, each leaving manifest and traces equal:

1. tensor names come from the part instances;
2. kernel handles come from the resolved entries;
3. model values come from the part instances, and constants that size buffers become sizes
   computed at open;
4. the layer body runs the block's parts: a block without a part skips it, per-layer branches
   read part facts instead of layer numbers, an unknown part is refused at open by name.

Done when the crate no longer depends on the DeepSeek model crate and the pinned lists from A
are empty apart from the four-rank list.

### D. Shapes become instance arguments (`kernels/`, the cubin build, the catalogue)

* `kernels/instances.toml`: one line an instance (source, entry symbol, shape arguments). The
  arguments are the shapes the kernel sees on one rank. The cubin script, the syntax probe and
  the baseline all read it.
* The 23 constants in `m4_common_sm121.cuh` and their copies become arguments with no default.
  The released instance keeps its symbol names.
* The catalogue's shape rules for these kernels are generated from the instance list plus the
  bounds the source states. Hand-written `hidden_equals(5120)` rules go.
* First one family as the worked example (the router and routed experts: one source is already
  templated), then the other families in parallel: attention and selection, seams, head, rows
  and bulk, drafter, vision.
* Done when every DeepSeek kernel takes its model shapes as instance arguments and the second
  shape's instances build.

#### D.1 The worked example: router and routed experts

The router and routed-expert family is the first one moved. It is the recipe the other families
follow; the files below are the whole mechanism, not a special case.

**The list.** `kernels/instances.toml` has one `[[instance]]` table per entry symbol:

```toml
[[instance]]
source = "opt_router_sm121.cu"   # a .cu directly under kernels/
entry = "opt_router_sm121"       # the extern "C" __global__ symbol
cubin = "opt_router_sm121"       # optional; defaults to the source stem
family = "router"                # "router" or "routed": which shape keys are required
catalogue = "deepseek-router"    # optional; the catalogue rule this instance proves
hidden = 5120
experts = 384
top_k = 6
```

A `router` instance needs `hidden`, `experts` and `top_k`; a `routed` instance also needs
`intermediate` and `padded`. The values are what one rank sees. Instances that share a
`cubin` are compiled in one translation unit, so a released multi-symbol cubin keeps its file
name and its full symbol set. An instance with its own `cubin` is a separate compile, which is
how a second shape is built without moving the released bytes.

Both readers validate the list: `scripts/instances.py` for the shell, and the Cargo build
scripts (which parse the same file) so a malformed line fails `cargo build`, not only the
shell build. They check the family's keys, positive values, unique entries, that the entry
appears in its source, and that a cubin is a bare stem.

**The defines.** For each instance the build passes

```
-DSPARK_INSTANCE_<ENTRY_UPPER>_<KEY_UPPER>=<value>u
```

so `opt_router_sm121`/`hidden` becomes `-DSPARK_INSTANCE_OPT_ROUTER_SM121_HIDDEN=5120u`.
The suffix is the source's own: the build does not reinterpret the number. The three readers
that compile a source — `scripts/build-cubins.sh`, the probe in
`crates/sparkflash-kernels/build.rs`, and the baseline generator — all take their flags from
this one list.

**The source.** A converted source has four rules:

1. No default. At the top, a `#error` guarded on the instance macros makes a build with no
   instance fail; a shape constant with a fallback is not converted.
2. The entry block is gated on its own instance, e.g.
   `#if defined(SPARK_INSTANCE_OPT_ROUTER_SM121_HIDDEN)`. A compile that carries only a second
   shape emits only that shape's entry, so a proof build cannot change the released cubin.
3. The body takes the shape as a template parameter where that is natural
   (`opt_router_body<HIDDEN, EXPERTS, TOPK>`), or as a source-level macro defined from its own
   instance where the body is not templated (`opt_moe`, `moe_router`, `routed_gate_up`).
   Derived counts (`K_BLOCKS = HIDDEN / 32`, `PACKED_ROW = HIDDEN / 2`) are arithmetic on the
   argument, not a second copy.
4. Literal `static_assert(HIDDEN == 5120)` checks become the bounds the body really needs
   (`HIDDEN % 256 == 0`, `TOP_K > 0`, `EXPERTS <= PLAN_MAX`). A literal equality would refuse
   the second shape the instance list exists to build.

**The build.** `scripts/build-cubins.sh` asks `scripts/instances.py --cubin-plan` for one
compile per `(source, cubin)` group and compiles every source not in the list exactly as
before. `build.rs` probes each group with the same flags (so the instance is type-checked even
on a host without nvcc, through the clang stub), skips those sources in its no-define loop, and
passes the released groups' flags to the combined probe. The baseline generator asks
`--defines-for-cubin <kernel>` for each pinned kernel, so `baseline/kernel_baseline.txt`
continues to record the compiled form of the same defines.

**The catalogue.** Give the instances that prove a part's shape the same `catalogue` key.
`crates/sparkflash-ops/build.rs` reads the file and emits
`CATALOGUE_<KEY>` constants (`ShapeConstraint::ANY.hidden_equals(...).intermediate_equals(...)
.routed_shapes(&[...])`) into `OUT_DIR/kernel_instances.rs`, which `src/parts.rs` includes.
The rule names exactly the built `(hidden, experts, top_k)` set, so a hand-written
`hidden_equals(5120)` or `max_experts(512)` cannot claim a width or count that was never
compiled. No looser bound is claimed unless the source accepts that dimension at run time; a
body compiled for one shape states its bounds as `static_assert`s and the catalogue claims that
one shape. Keys used by the worked example: `deepseek-moe` (released routed experts),
`deepseek-router` (both built router shapes, 384/6 and 128/3) and `deepseek-mtp` (the
128/3 MTP shape; the k3 expert instances carry `catalogue_intermediate = false` because the
MTP part has no intermediate-width field even though the bodies take one).

**The proof, in order.** For the released instance:

1. the cubin sha256 equals `baseline/kernel_baseline.txt` (the preferred result: the defines
   expand to the same tokens, so the compiled form should not move);
2. if it moves, the disassembly (`cuobjdump -sass`) of the released entry equals the baseline
   cubin's;
3. if that moves, the launch outputs equal the baseline run's.

Then build a second instance with a different shape and check it compiles, contains its own
symbol, and is not in the released cubin or the baseline. The worked example's proof instance is
`opt_router_4096e256k8_sm121` (4096 wide, 256 experts, top-8), built into its own cubin.

**Result for the worked example.** All four released cubins compile to the baseline bytes under
their instance defines (nvcc 13.0.88, sm_121a), so the first proof step holds and neither
disassembly nor launch outputs were needed:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `opt_router_sm121` | 180,264 | `888328156f8699d0ca934375a40443d7f06cd383ab017c5e8b1ff2987a332f4e` |
| `moe_router_sm121` | 27,376 | `3b8823e84158e2956b19509e90f6aed02dbe68235c1a7f5d76ab438c51e53eb0` |
| `routed_gate_up_sm121` | 35,656 | `b108f7fa1a21623e876f17a8e2da5634257a6745e1d2f1233cc7d325c4a360f8` |
| `opt_moe_sm121` | 1,692,848 | `7ebfe83406896c50533f9cbd7b63716bb529d55e1b402fa9424ee17b349cd7c8` |

`scripts/regen-kernel-baseline.sh`, now reading its defines from the instance list, regenerated
the whole 49-kernel baseline and `diff` against `baseline/kernel_baseline.txt` was empty. The
second shape built alone as `opt_router_4096e256k8_sm121.cubin` (126,496 bytes; only that
symbol), not in the released cubin and not in the baseline.

The other families repeat the same four steps: add the instances, convert the source under the
no-default rule, point the build and baseline at `instance_groups()`, and give the instances a
`catalogue` key. Nothing in the mechanism is router-specific.

#### D.2 The seams family

The seams family covers the four sources that carry the mHC layer seams and the kv/tail path:
`kernels/opt_seams_sm121.cu` (two entries), `kernels/layer_seams_sm121.cu` (eight),
`kernels/kv_path_sm121.cu` (four) and `kernels/rmsnorm_sm121.cu` (one). Its shape vocabulary
is `family = "seams"`: `hidden` is required and `hc` is optional. The mHC bodies take both as
compile-time arguments; the rope, quantiser, kv-append and tail-norm entries take their widths as
runtime launch words and carry only `hidden`, so their instance records the released model width
without pretending the body was compiled for it.

The converted sources follow D.1's rules. `opt_seams_sm121.cu` derives `OPT_S_ROWS =
hc*hc + 2*hc`, `OPT_S_COLS = 256/hc` and the two sum-of-squares slots from `OPT_S_ROWS`, replacing
the local 5120/4/24/25/26 literals; `layer_seams_sm121.cu` replaces every `SPARK_M4_HIDDEN`,
`SPARK_M4_HC_MULT` and `SPARK_M4_HC_PREDICTOR_ROWS` read with `SPARK_SEAM_HIDDEN`,
`SPARK_SEAM_HC` and the arithmetic row count. Both sources select the shared body's shape from
whichever mHC entry instance the compile carries and `#error` when none does. Each entry is gated
on its own macros; the runtime-width entries are gated too, so a one-entry proof cubin carries one
symbol. `SPARK_M4_QUANT_BLOCK` stays a format constant: 32 is the MX block, not model geometry.
The mHC host geometry refusal in `sparkflash-kernels::layer_seams_launch::check_seam_geometry`
is unchanged and still names a non-released `(dim, hc_mult, rows)`; the kernel's own guard remains
defence in depth.

The catalogue keys and the generated rules are:

| key | instances | generated rule |
| --- | --- | --- |
| `deepseek-hyperconnection` | the six mHC entries | `hidden_equals(5120).streams_between(4, 4)` |
| `deepseek-rope` | `rope_rotate_bf16`, `kv_norm_rope_bf16`, `rope_rotate_rows_bf16` | `hidden_equals(5120)` |
| `deepseek-quantize` | `w8a8_prologue_e4m3`, `w8a8_prologue_bf16_e4m3`, `window_quant_fp8_bf16` | `ANY` (`catalogue_hidden = false`) |
| `deepseek-rows-norm` | `rmsnorm_sm121` | `hidden_equals(5120)` (the rows family's `opt_rmsnorm_bf16` shares it) |

`catalogue_hidden = false` is the runtime-width escape: the quantiser entries state that their
body has no compiled hidden shape, and `sparkflash-ops/build.rs` then emits an open rule rather
than pinning a width the launch never checked. `scripts/instances.py` and the kernel build script
accept the same field so the three readers stay one list.

**Proof.** The four released cubins compile from the instance defines to the baseline bytes under
nvcc 13.0.88, sm_121a, so no disassembly or launch comparison was needed:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `opt_seams_sm121` | 124,024 | `3e9311bf973538224d2dbe359ed7cd443b509ec720edb63122905c9960eeb18e` |
| `layer_seams_sm121` | 171,808 | `e3e1b4821da9bfafcdefce73d325ab0cf2ef85226020c8d069cdf880c310ad06` |
| `kv_path_sm121` | 57,976 | `65a5891b123d9da0b858823abaf80d11b39dbe07a7ecdfd71e5eb780e4a19487` |
| `rmsnorm_sm121` | 34,512 | `fb559d1bd3666fc249c4f5b5bf46fa236004b3068355963876db43245764e678` |

The second shape is `opt_mhc_seam_4096h2_sm121` (hidden 4096, hc 2), built from
`opt_seams_sm121.cu` alone into its own cubin: 95,200 bytes,
`94830b867f92013a17fffc41c8072e71ded8cb1b68c9cb250a552eda1cbacd19`. `cuobjdump -symbols`
shows only `opt_mhc_seam_4096h2_sm121` (beside the compiler's internal helpers); the symbol is
absent from the released `opt_seams_sm121` cubin and from `kernel_baseline.txt`.

**For the next family.** A seams compile group must carry at least one mHC entry instance because
the shared helpers take `hidden`/`hc` from it; a group of runtime-width entries alone does not
define a body shape. Add second-shape proofs for the runtime-width entries as their own cubins if
a future model changes a launch width, and keep `catalogue_hidden = false` for any entry whose
width the kernel reads at run time.

#### D.3 drafter

The drafter is the third family moved: one source, `kernels/opt_draft_sm121.cu`, and the
fifteen entry symbols the DeepSeek MTP stage loads from `opt_draft_sm121.cubin` — `opt_draft_taps`,
`norm_quant`, `attn`, `logits`, `head_fp8`, `head_fp8t`, `logits_fp8`, `markov`,
`fold_prob`, `pick_prob`, `embed_blocks`, `markov_seqs`, `markov_seqs_mma`, `fold_seqs`
and `pick_seqs`.

**Shape keys.** The `drafter` family takes six one-rank values. Every released entry lists all
six so the compile group has one complete vocabulary; an entry reads only the ones its body uses.

| key | released value | read by |
| --- | --- | --- |
| `hidden` | 5120 | taps, norm/quant, the head logits and quantisers, embed |
| `hc` | 4 | taps, embed (the residual-stream copies) |
| `taps` | 3 | taps |
| `latent` | 512 | attention (the MLA head width; `latent / 32` ue8m0 scales) |
| `window` | 128 | attention (`max keys = window + 32`) |
| `markov_rank` | 256 | markov, markov_seqs, markov_seqs_mma |

**The catalogue.** Every released instance carries `catalogue = "deepseek-mtp"`. The k3 routed
expert instances already prove that key's `(hidden, experts, top-k)` set, so
`sparkflash-ops/build.rs` treats a `drafter` row as a hidden-width contributor only: it adds
`hidden` but does not invent an experts/top-k triple the drafter does not have. The generated
`CATALOGUE_DEEPSEEK_MTP` is unchanged — `hidden_equals(5120)`, `routed_shapes = [(5120, 128, 3)]`,
no intermediate width — and `(Part::Mtp, Image::Deepseek)` keeps using it. A geometry whose hidden
width is not 5120 is refused by name by `ShapeConstraint::accepts`; there is no silent fallback.

**The source.** The no-instance `#error` lists every entry's macros. Each entry defines the
shape macros it reads from its own `SPARK_INSTANCE_*` values and `#undef`s them at the end of its
block, so the released compile group can carry all fifteen entries and a proof compile can carry
one. The local copies are gone: `hidden / 32` and `latent / 32` replace the quantiser widths,
`window + 32` the attention key bound, `hidden / 256` and `hidden / 512` the logit-loop chunk
counts. `markov` reads the rank in `rank / 256` warp spans. `markov_seqs` keeps its original
one-warp dot and asserts `markov_rank == 256`, which is the bound that body really needs rather
than a copy of a model constant.

**The proof.** The released cubin compiled under the instance defines is byte-identical to the
baseline; per-symbol `cuobjdump -sass` comparison shows all fifteen entries identical, so no
launch-output comparison was needed.

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `opt_draft_sm121` | 631,856 | `12743980711ede547cfb32cee2a1ca0ab145228aeb98b73e2fd6b51a5005b161` |

The second shape is `opt_draft_taps_4096h2c2t_sm121` (hidden 4096, hc 2, taps 2; latent 256,
window 64 and Markov rank 128 are named but not read by the tap body). It builds alone into
`opt_draft_taps_4096h2c2t_sm121.cubin` (5,976 bytes,
`4a149c06d19c03ca86477d6b682b642415e1d0539ef7dadf86f898b5caaa854e`), contains only its own
`STT_FUNC`, and appears in neither the released cubin nor `baseline/kernel_baseline.txt`.

**For the next family.** `scripts/instances.py`, `sparkflash-kernels/build.rs` and
`tests/kernel_instances.rs` now know the `drafter` vocabulary. The `deepseek-mtp` key is shared
with the k3 routed rows, so a change to one must be made with the other in view; a new key would
have to be named by `shape_for` in `parts.rs`. The released cubin stayed at the baseline bytes
without a baseline edit, and the second-shape cubin is outside the baseline by construction.

#### D.4 Rows and bulk

The fourth family is the multi-row decode twins, the bulk-prefill MMA bodies and the generic
dense/GEMV projections. Its sources are `opt_rows_sm121.cu`, `opt_mma_sm121.cu`,
`opt_gemv_sm121.cu`, `opt_misc_sm121.cu`, `b12x_mxfp8_dense_sm121.cu`,
`b12x_mxfp8_dense_m1_sm121.cu`, `b12x_mxfp8_gemv_m1_sm121.cu` and
`b12x_hostw_gemv_m1_sm121.cu`. Every released entry symbol is one `[[instance]]` table with
`family = "rows"`; the family's required keys are `hidden`, `intermediate`, `experts` and
`top_k`, and the released values are 5120, 576, 384 and 6.

**What the bodies compile in.** `opt_rows_sm121.cu` had three copies of the released hidden width
(`OR_S_HIDDEN`, `OR_LM_HIDDEN`, `OHS_HIDDEN`); they are now one `OR_ROWS_HIDDEN` taken from the
instance macros, and a mixed-shape line in one cubin fails a `static_assert`. `opt_mma_sm121.cu`
takes `OMR_EXPERTS`/`OMR_TOPK` and `OMM_HIDDEN`/`OMM_INTER`/`OMM_TOPK` the same way, and
`opt_misc_sm121.cu`'s `OPT_LM_HIDDEN` comes from the LM-head instance. The two b12x GEMVs had a
literal `MAX_INNER 5120`; each is now its instance's `hidden`, so the second shape moves the
shared-memory array and the wrapper's refusal bound together. The generic bodies
(`opt_gemv_sm121.cu`, `b12x_mxfp8_dense_sm121.cu` and its M1 sibling) take their widths as
runtime arguments; they are listed because they are released entries, and their macros are unused
rather than defaulted. `b12x_hostw_gemv_m1_sm121.cu` is included by `opt_rows_sm121.cu`; the
include aliases the included entry macros onto the rows entry macros, so the opt_rows cubin keeps
the included symbols and one shape.

**The catalogue.** Five keys are generated from the instance list and the dimensions the sources
actually fix: `deepseek-rows-norm` (RmsNorm, hidden), `deepseek-rows-mlp` (DenseMlp, hidden and
`intermediate_multiple = 16`, the runtime bound the MMA bodies state), `deepseek-rows-gemm`
(DenseGemm, hidden), `deepseek-rows-activation` (Activation, hidden) and
`deepseek-rows-combine` (MoeCombine, hidden, with `max_top_k = 64` — the combine's top-k is a
runtime argument). The generated constants are `CATALOGUE_DEEPSEEK_ROWS_*`; the hand-written
`hidden_equals(5120)`, `intermediate_multiple(16)` and `max_top_k(64)` rules for these parts are
gone. MoePlan keeps `ShapeConstraint::ANY`: its released symbols come from `moe_plan_sm121.cu`
and `opt_moe_sm121.cu`, which are not rows-family sources. The attention bodies inside
`opt_rows_sm121.cu` and `opt_mma_sm121.cu` keep their own `OR_A_*`/`OBA_*` geometry; those
constants are the attention family's, not this family's hidden/intermediate/expert arguments.

**The proof.** All eight released rows cubins compile to the baseline bytes under their instance
defines (nvcc 13.0.88, sm_121a), so neither `cuobjdump -sass` nor a launch comparison was needed:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `b12x_mxfp8_dense_sm121` | 209,424 | `d751f2b65b076659ff0c4cc37542f8ae4899ec61b7f40c3d8223ba3181f9d350` |
| `b12x_mxfp8_dense_m1_sm121` | 14,520 | `8fcd899dad03debc5292106e5f05e6c545f61fcc0d268789d6f4ce64509fb5ed` |
| `b12x_mxfp8_gemv_m1_sm121` | 112,512 | `c33cd129f0003e1a4778bdced1fe788bf1f4a1be84fe8033c7989bc8d4bab469` |
| `b12x_hostw_gemv_m1_sm121` | 73,848 | `ca1d8498830cddfcdb9a3f1a72e777cb0360a80e4ce7be18af514695bda6bc72` |
| `opt_gemv_sm121` | 1,330,752 | `3111ba50858b93449823f23810d7fc75c8e77e76141958697c221123f240c973` |
| `opt_misc_sm121` | 302,128 | `57c8aa68ef3ba12d46a3b3e78755879308373eed4ee1332299e01861122109c2` |
| `opt_rows_sm121` | 1,021,528 | `cc9ee43f59a46321a056ccdd872067ead28d4d4f203b213e0ffa2859f1175a97` |
| `opt_mma_sm121` | 506,560 | `fdb2c5d97d19ce696d98d0420910b3ae20e54df6a9c70347b5297da8aab69587` |

The second shape is `b12x_hostw_gemv_m1_bf16w_4096_sm121` (hidden 4096), built alone into
`b12x_hostw_gemv_m1_4096_sm121.cubin`: 27,440 bytes,
`623c8d3205cacbc7b43cc5e7c552d6f1f3ace9e6cc8973defadcd45100529cc1`, one entry symbol and nothing
else; it is absent from `b12x_hostw_gemv_m1_sm121.cubin` and from the baseline. The released
host-weight cubin's three symbols are unchanged.

**What the next family must know.** A source with an entry name that does not end in `_sm121` gets
the define `SPARK_INSTANCE_<ENTRY>_<KEY>` without the suffix (`opt_rows_lm_head` becomes
`SPARK_INSTANCE_OPT_ROWS_LM_HEAD_HIDDEN`); the work that converts a family must read the exact
entry names from the list, not assume the source stem is the prefix. A source that `#include`s
another instanced source (`opt_rows_sm121.cu` includes `b12x_hostw_gemv_m1_sm121.cu`) needs the
alias block that D.2 uses, or the combined probe sees one family's macros in the other's entries.
Finally, a generic body that reads a width at run time does not have to claim the released width:
the generated rule should carry the source's multiple (DenseMlp's `intermediate_multiple = 16`)
rather than an equality the body does not require.

#### D.5 head: the certified LM head and its reduction

**Sources and shapes.** `kernels/cert_lm_head_sm121.cu` owns the scalar and K8-batched certified
screens; `kernels/lm_head_reduce_sm121.cu` owns the four-shard device reduction. Both take the
head family's three instance arguments, in the vocabulary the list names:

| key | released value | meaning |
| --- | --- | --- |
| `hidden` | 5120 | the hidden width the screen tiles |
| `vocab_per_rank` | 32320 | this rank's vocabulary shard |
| `ranks` | 4 | the shards the reduction folds |

The total vocabulary is `ranks * vocab_per_rank` (129,280), arithmetic on the arguments rather
than a second copy. Each entry is gated on its own `SPARK_INSTANCE_<ENTRY>_*` macros, so the
released cubins keep their two symbols each and the proof shape builds alone. The screen's 128-value
tile, 1,024-entry chunk and six-row verify batch, and the reduction's 256 threads, top-5 and status
bits, stay algorithm constants: they are the body's tiling, not model geometry. The literal
`== 5120`, `== 40`, `== 32` and `== 129280` asserts became `HIDDEN % 128 == 0`, `BLOCKS > 0`,
`ceil(vocab_per_rank / 1024)` and a positive shard; `ranks == 4` remains because the reduction's
four logit and four certified pointers are the ABI, not a negotiable shape.

`kernels/pool_group_sm121.cu` was inventoried with the family and needs no instance: every entry
takes its dimensions at run time (`ratio`, `head_dim`, `cols`, `blocks`, `count`, `hc_mult`,
`dim`). Its launch helpers did have silent zero-shape returns; `launch_k_norm`,
`launch_fp4_round_trip`, `launch_scale_index_weights` and `launch_hc_pre_collapse` now refuse a
zero extent with a message naming the symbol instead of launching a kernel that writes nothing.

**The catalogue.** The instances that prove the released geometry carry
`catalogue = "deepseek-head"`. `crates/sparkflash-ops/build.rs` reads the head family's
`(hidden, ranks * vocab_per_rank)` pair and emits `CATALOGUE_DEEPSEEK_HEAD =
ShapeConstraint::ANY.hidden_equals(5120).vocab_equals(129280)`; `src/parts.rs` uses it for both
the `LmHead` and `Embedding` DeepSeek entries. The old hand-written
`hidden_equals(5120).vocab_multiple(128)` is gone, so a 4096-wide or differently sized head is
refused by name before a weight is read.

**The proof.** The released cubins compile to the baseline bytes under their instance defines
(nvcc 13.0.88, sm_121a), so the first proof step holds and neither disassembly nor launch outputs
were needed:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `cert_lm_head_sm121` | 36,800 | `bbd30c8d6d358bd200b07cffa54d662c9144501b69310864933b68a0e53b6798` |
| `lm_head_reduce_sm121` | 76,176 | `b4ef4ce8141be1678b565cdf838af46fefb4116de3dcaed97d4bca4506a61e30` |
| `pool_group_sm121` | 54,312 | `ed16a3a00d1150ea7b32de59c9dc7f44011faa8a8a45cb3ea518c1a84c6175b7` |

All three hashes equal `baseline/kernel_baseline.txt`. The second shape is
`cert_lm_head_4096v16384_sm121` (hidden 4096, 16,384 entries a rank, four ranks) in its own cubin:
13,216 bytes, `d3c3d6b5074f7eff8820e29bcbfb0aa45f7f031b5f0b0ac7f70bfcd6d6c99a40`, containing only
that symbol; it is absent from the released cubin and from the baseline. The proof was built in
`/tmp/partd/inst-head` on the GPU head with the defines `scripts/instances.py
--defines-for-cubin` prints.

**What the next family should know.** A head-shape instance must carry `hidden` even for
`lm_head_reduce`, which does not read it: the family key list is what the three readers validate.
The generated `vocab_equals` is a new `ShapeConstraint` field; use it rather than
`vocab_multiple` when a body's vocabulary is a compiled count, because a multiple still admits
larger counts. The `m4_abi` static-assert scan resolves the released instance defines through
`crate::instances::<CUBIN>_DEFINES`; a source whose asserts name a second-shape macro needs that
cubin's generated defines too, or the scan sees an unknown identifier.


#### D.6 vision

The vision tower is the second family moved. `kernels/opt_vision_sm121.cu` exports ten entry
symbols that the forward and the Inkling tower load by name from `opt_vision_sm121.cubin`:
`opt_vision_patchify`, `opt_vision_patchify_16`, `opt_vision_gemm`, `opt_vision_rmsnorm`,
`opt_vision_rope`, `opt_vision_attn`, `opt_vision_swiglu`, `opt_vision_unfold`,
`opt_vision_span_rows` and `opt_vision_route`. `kernels/inkling_vision_sm121.cu` includes the
source, so it is the one file outside the family that had to name the same shapes.

**Shape keys.** The `vision` family takes seven values: `hidden` (1024), `heads` (16),
`patch` (14 or 16), `intermediate` (2816), `out_hidden` (5120), `experts` (384) and
`top_k` (6). The per-head width is arithmetic on the arguments (`hidden / heads`), and the body
bounds it with `static_assert(head_dim == 64)` because the rope pair layout and the attention
tiles are written for a 64-wide head; `qkv = 3 * hidden` and `unfold = 9 * hidden` are also
arithmetic. The patchify body was already templated on its patch side, so it kept that form; the
other entries define their shape macros from their own instance inside the entry's `#if` and
`#undef` them after the body. Algorithm constants (the 128x128 GEMM tile, the 64-query/64-key
attention tiles, the 32-value K step, the 1e-6 norm epsilon, the router's softplus threshold,
norm epsilon and 1.5 scale) stay literals.

**Instances.** The ten released entries live in `opt_vision_sm121`; the second-shape proof
`opt_vision_span_rows_4096_sm121` is its own cubin. `opt_vision_gemm` and `opt_vision_route`
prove no catalogue shape (the GEMM takes rows, columns and the reduction extent at run time; the
route's experts/top-k are carried but no part rule claims them).

**The catalogue.** `crates/sparkflash-ops/build.rs` emits one generated rule per key, claiming
only the fields the part's bodies read:

| key | fields claimed |
| --- | --- |
| `deepseek-vision-patch-embed` | hidden, patch sides 14 and 16 |
| `deepseek-vision-position` | hidden, exact heads, head width |
| `deepseek-vision-norm` | hidden |
| `deepseek-vision-attention` | hidden, exact heads, head width |
| `deepseek-vision-mlp` | hidden, intermediate |
| `deepseek-vision-downsample` | hidden |
| `deepseek-vision-merger` | hidden, out_hidden |

Exact heads are `heads_multiple(n).max_heads(n)`; the values come from the instances. The
non-shape facts (norm kind and epsilon, activation, position, downsample code, channels and
temporal patch) stay beside the generated constant in `src/parts.rs`. The merger rule claims
`out_hidden` (the 5120-wide span row the body copies) and no longer claims
`merger_intermediate`: the two projection GEMMs take that width at run time, so no compiled body
proves it. The Qwen refusal test still refuses on the width (1152 against the compiled 1024),
even though the 16-patch body is now claimed.

**The proof.** The released cubin compiles to the baseline bytes under the instance defines, so
neither disassembly nor launch outputs were needed; the Inkling cubin, which composes the same
bodies, also kept its baseline bytes:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `opt_vision_sm121` | 320,072 | `15e75fc99d144ae0d6c8bf1da414321a60988e9fad67d12e1468ad05cac50c6f` |
| `inkling_vision_sm121` (composed) | 345,000 | `5b7687fbd3149c167701d643d63510699283a7ded9d68fe6c9ef131ac294d77d` |

The second shape built alone as `opt_vision_span_rows_4096_sm121.cubin` (6,000 bytes;
`f360dce773125799222428a580b755ae7ef34694fc99d1a09c2c1065ff23d975`), contains only
`opt_vision_span_rows_4096_sm121`, and is in neither the released cubin nor the baseline.

**What the next family needs to know.** A `__device__ __forceinline__` body called from the
entry changed code generation on nvcc 13.0.88 even when every shape folded to the same literal
(the wrapper moved the loop bound and added a first-iteration guard). The released bytes came
back only when each body stayed directly in its entry and the shapes entered as macros; the
already-templated patchify body was the exception and stayed byte-identical. A source that is
`#include`d by another family must also have that family name the shared instance's macros
before the include, and must put its own aliases after it, because the converted source
`#undef`s its per-entry macros. Finally, the vision host's `gemm` wrapper now refuses a
reduction width that is not a positive multiple of 32, an output width that is not a positive
multiple of 8, or strides that are not multiples of 8, by name before the launch instead of
silently computing a truncated tile.


#### D.7 Attention and selection

The attention and candidate-selection family follows D.1 with two new shape vocabularies:
`mla` (the three CSA2 attention entries, the split-KV decode entry and D49's ring entry) and
`select` (the reference and optimized candidate selection). Six sources are listed in
`kernels/instances.toml`; `kernels/flash_mla_csa2_common_sm121.cuh` and
`kernels/select_body_sm121.cuh` are bodies the listed entries include, so they carry no
`[[instance]]` of their own.

**Sources and entries.** `flash_mla_csa2_sm121.cu` (one entry), `flash_mla_csa2_seq_sm121.cu`
(`flash_mla_csa2_seq_sm121` and `flash_mla_csa2_seq_batched_sm121`, one cubin),
`flash_mla_csa2_ring_sm121.cu` (one released entry plus the proof entry),
`opt_attn_sm121.cu` (`opt_flash_mla_csa2_seq`), `ref_select_sm121.cu` (one entry) and
`opt_select_sm121.cu` (ten entries: `opt_ref_select`, `opt_ref_select_pre`,
`opt_ref_select_fast`, `opt_select_rows`, `opt_select_rows_bf16`, `opt_select_rows_tab`,
`opt_select_scores`, `opt_select_scores_rows`, `opt_select_scores_seg`,
`opt_select_scores_segs`). All released cubin names and symbol sets are unchanged.

**Shape keys.** An `mla` instance names `hidden`, `heads`, `head_dim`, `kv_lora`,
`window`, `index_heads`, `index_head_dim`, `index_topk`, `hsi_blocks`,
`hsi_block_tokens`, `ring_capacity` and `positions`. A `select` instance names `hidden`,
`index_heads`, `index_head_dim`, `index_topk` and `positions`. `hidden` and `head_dim`
are the part geometry the catalogue claims; the attention body reads the latent, heads, window
and indexer numbers. Every released instance has `hidden = 5120`, `head_dim = 512`,
`kv_lora = 512`, `heads = 64`, `window = 128`, `index_heads = 32`,
`index_head_dim = 128`, `index_topk = 512`, `hsi_blocks = 2048`,
`hsi_block_tokens = 8`, `ring_capacity = 134` and `positions = 2097152`; a `select`
instance drops the fields it has no geometry for. The window scales, union capacity and the FP4
scale counts are arithmetic on the arguments (`kv_lora / 32`, `index_topk + window`,
`kv_lora / 2`), not second copies.

**The sources.** Each converted source has the D.1 four rules. It fails with `#error` when no
instance macro is defined, and each entry is gated on its own macros so a proof compile emits
only its own symbol. The shared M4 header's geometry names are `#undef`/`#define`d from the
entry's instance before the arithmetic header is included: the header's defaults still belong to
the other families, and this source does not read them. Literal equality assertions became the
bounds the bodies need (`kv_lora == 2 * threads`, `index_head_dim == 4 * warp_lanes`,
`heads % index_heads == 0`, `ring_capacity >= window`); the fast select's `FAST_MAX_POS`,
the segment body's `SR_DIM` and `SG_HEADS` come from the instance. The ring source keeps its
released branch last in the preprocessor chain because the Rust constant readers parse every
`#define` in the file and the last textual definition must be the released shape.

**Generated catalogue keys.** `deepseek-mla` is carried by the five released MLA entries and
generates

```text
CATALOGUE_DEEPSEEK_MLA = ANY.hidden_equals(5120).head_dim_equals(512).positions(0, 2097152)
```

`deepseek-indexer` is carried by the eleven released selection entries and generates

```text
CATALOGUE_DEEPSEEK_INDEXER = ANY.hidden_equals(5120).head_dim_equals(128)
    .top_k_equals(512).max_heads(32).positions(0, 2097152)
```

Those two rules replace the hand-written `(MlaAttention, Deepseek)` and
`(SparseIndexer, Deepseek)` arms of `shape_for`. The old MLA rule's `head_dim_multiple(16)`
and `groups_between(1, MAX)` are gone: the compiled head width is exactly 512 and the group
count is not a shape any listed body reads. The indexer rule keeps `max_heads(32)` because the
selection bodies are runtime-generic over heads and 32 is the widest they address. The
`Csa2Compression` rule is the next family's (D.8): its catalogue entries are served by
`compressed_append_sm121.cu`, `opt_misc_sm121.cu` and `pool_group_sm121.cu`.

**Refusal.** `SparseIndexerLauncher::new` now refuses a DeepSeek selection launch whose rows
reach past `DEEPSEEK_INDEXER_MAX_POSITION` (`1 << 21`), naming the symbol and the bound; the
kernels' in-body bounds remain as the last line of defence. The catalogue rule refuses a model
whose geometry is outside the built shape before a weight is read.

**The proof.** The six released cubins compiled to the baseline bytes under their instance
defines on node0 (nvcc 13.0.88, CUDA 13.0, sm_121a), so neither
disassembly nor launch output was needed:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `flash_mla_csa2_sm121` | 71,256 | `553a8982a06763436a6f2437c6bae8fc2d81f51c4033379c7aee0ea08f894388` |
| `flash_mla_csa2_seq_sm121` | 121,560 | `c278bc97829a79e728c181bc451d331088e27af446c9920e2c00bf2c1b4ad8d9` |
| `flash_mla_csa2_ring_sm121` | 48,800 | `6d346b2b6c3d7955c040a13852c306a979b15bcfcee38388e1e517fb5df27a0e` |
| `opt_attn_sm121` | 78,544 | `274bc2f91f5407efe185f9368cd247ec0ca4ac3f6aa2826ac77df6b6a67465a2` |
| `ref_select_sm121` | 28,264 | `b90baccac9937386c54c5fa69da88e41d759d3218ff2a1ac220dcfbba860f3e8` |
| `opt_select_sm121` | 476,088 | `dd7240901db3beea39f1169567b29ec15e81a85f6d8c92d2412ed23f756b46c5` |

The second shape is `flash_mla_csa2_ring_32h256k_sm121`, the ring body compiled for
`heads = 32`, `index_heads = 16`, `index_topk = 256`, `window = 64`,
`hsi_blocks = 1024`, `ring_capacity = 70` and `positions = 1048576`. It builds into its
own cubin (48,928 bytes,
`ad688dcc5b653d4a9c25829d05feda9438d3909aed75d445a62461e719f4fb66`), whose only kernel
symbol is `flash_mla_csa2_ring_32h256k_sm121`; that symbol appears in neither the released ring
cubin nor `baseline/kernel_baseline.txt`.

**For the next family.** The two build-script readers now accept the `mla` and `select`
families, and `crates/sparkflash-kernels/tests/kernel_instances.rs` checks their released
values and proof. Consts readers of a converted source must extend their table with the generated
`<CUBIN>_DEFINES` (as `m4_abi::consts`, the ring and sequence launch tests now do). Do not
reorder a source's instance chain so that a second shape becomes the last textual definition of a
name a Rust reader resolves.


#### D.8 CSA2 compression

**Sources and shapes.** `compressed_append_sm121.cu`, the compression entry of
`opt_misc_sm121.cu` (`opt_rows_group_append`) and `pool_group_sm121.cu` now carry a
`family = "csa2"` instance. The family's keys are `hidden`, `head_dim`, `ratio` and
`positions`; the released values are 5,120, 512, 1 and 2,097,152 (the source's 2^21 position
bound). The append body compiles its one shape, the latent width, from
`SPARK_INSTANCE_<ENTRY>_HEAD_DIM`; its ratio, row count and store row stay runtime arguments.
`opt_misc_sm121.cu` includes the append source, so its alias block maps the compression entry's
macros onto the included source's, the same pattern D.2 uses.

**The catalogue.** `crates/sparkflash-ops/build.rs` emits

```text
CATALOGUE_DEEPSEEK_CSA2 = ANY.hidden_equals(5120).head_dim_equals(512).min_ratio(1)
    .positions(0, 2097152)
```

and `src/parts.rs` uses it for `(Part::Csa2Compression, Image::Deepseek)`. The hand-written
`ANY.hidden_equals(5120).min_ratio(1).positions(0, 1 << 21)` arm is gone.

**The proof.** The three released cubins compiled to the baseline bytes under their instance
defines on node0 (nvcc 13.0.88, CUDA 13.0, sm_121a), so neither
disassembly nor launch output was needed:

| released cubin | bytes | sha256 |
| --- | --- | --- |
| `compressed_append_sm121` | 23,800 | `77026c93f196ca290203e37b62b56a4e83853d9c5ef1eba9a6b68b72f449f7da` |
| `opt_misc_sm121` | 302,128 | `57c8aa68ef3ba12d46a3b3e78755879308373eed4ee1332299e01861122109c2` |
| `pool_group_sm121` | 54,312 | `ed16a3a00d1150ea7b32de59c9dc7f44011faa8a8a45cb3ea518c1a84c6175b7` |

The second shape is `compressed_append_256_sm121` (`hidden = 4096`, `head_dim = 256`,
`ratio = 2`, `positions = 1048576`), built alone into
`compressed_append_256_sm121.cubin`: 15,520 bytes,
`1d91f6dbb854beab898fc869d64d24f464bd7e157162205f0b2d073666edd496`. Its only entry symbol is
`compressed_append_256_sm121`, which is absent from the released `compressed_append_sm121`
cubin and from `baseline/kernel_baseline.txt`. The full
`crates/sparkflash-kernels/scripts/check-kernel-baseline.sh` run on the head reported every
one of the recorded kernels `ok`.


## 5. Order

1. A, alone.
2. B and the first family of D, in parallel.
3. C once B's plan reproduces the manifest; the remaining families of D beside it.
4. The second shape runs; the speed A/B on the real model; the result goes to the owner.

## 6. Stopping

If a step cannot be made free for DeepSeek, revert it, record the measurement and bring it to the
owner. A slower DeepSeek is not acceptable; an aborted step is.
