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

## 4. The work

### A. Harness and inventory (first, one owner)

* The open manifest, the replay workload and one script that records and compares manifest,
  launch trace, tick trace and speed.
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

#### D.2 Attention and selection

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
`Csa2Compression` rule is deliberately unchanged: its catalogue entries are served by
`compressed_append_sm121.cu`, `opt_misc_sm121.cu` and `pool_group_sm121.cu`, which are not
this family's sources and have no instance here.

**Refusal.** `SparseIndexerLauncher::new` now refuses a DeepSeek selection launch whose rows
reach past `DEEPSEEK_INDEXER_MAX_POSITION` (`1 << 21`), naming the symbol and the bound; the
kernels' in-body bounds remain as the last line of defence. The catalogue rule refuses a model
whose geometry is outside the built shape before a weight is read.

**The proof.** The six released cubins compiled to the baseline bytes under their instance
defines on the GPU head (192.168.1.80, nvcc 13.0.88, CUDA 13.0, sm_121a), so neither
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

## 5. Order

1. A, alone.
2. B and the first family of D, in parallel.
3. C once B's plan reproduces the manifest; the remaining families of D beside it.
4. The second shape runs; the speed A/B on the real model; the result goes to the owner.

## 6. Stopping

If a step cannot be made free for DeepSeek, revert it, record the measurement and bring it to the
owner. A slower DeepSeek is not acceptable; an aborted step is.
