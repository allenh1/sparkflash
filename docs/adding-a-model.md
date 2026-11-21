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

# Adding a model arm

A model arm is configuration, a layer plan and typed op calls. It owns no CUDA and depends on no
other model crate. This guide is the bring-up path the parity harness assumes.

## 0. The config is the model

The released Hugging Face `config.json` is the description of the model, not a hint for a Rust
module. `sparkflash_model::Architecture` (`architecture.rs`) reads it - flat or nested under
`text_config` - and turns its string fields into typed pieces:

- `layer_types` selects the attention of each layer (`linear_attention`,
  `deepseek_sparse_attention`, `full_attention`, ...);
- `mlp_layer_types` and `first_k_dense_replace` select a dense MLP or MoE;
- `indexer_types` says whether a sparse layer runs its own indexer or reuses one;
- `hidden_act`, `scoring_func` and `output_gate_type` select activations;
- `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids` and
  `candidate_source_layer_id` say where CSA2 compression and the HSI pool live;
- `hc_mult`, `hc_count`, `hc_lowrank` and `mhc` select the residual mixer;
- `engram_layer_ids` and `ple_layer_ids` place the side lookups.

The result is an [`sparkflash_model::Architecture`] whose `parts()` is the set of named
`Part`s the run needs. `Part` is the shared vocabulary (`mla_attention`, `csa2_compression`,
`sparse_indexer`, `moe`, `gdn_scan`, `kda_scan`, `hyper_connection`, ...). A part is
implemented once per `Target` (`sm100`, `sm120`, `sm121`); every model that names it uses
the same plan and the same kernel, so the second model on an existing part is configuration only.

`Architecture::missing_on(Target)` is the model-side bring-up list. The per-kernel truth is
`sparkflash_ops::parts` ([catalogue](parts-catalogue.md)): `CATALOGUE` names the target, image
and symbols behind every part, `carried(target)` says what this tree has, and
`missing(&Architecture, target)` is the direct answer for a checkpoint. A checkpoint whose list
is empty has a serving path without new Rust.

`sparkflash serve --checkpoint DIR` uses this document when `--model` is absent: it reads the
checkpoint's `config.json`, matches the family, checks the catalogue for the parts the target
needs, and hands the run to that pack's arm. A missing part is a refusal that names every missing
part, not a silent fallback.

**DeepSeek is canonical.** Where more than one arm implements a part, the DeepSeek implementation
is the preferred one: it is the most exercised in this tree. A GLM or Qwen kernel may be listed
beside it, and may be the only implementation of a part DeepSeek does not have, but it does not
replace the DeepSeek implementation for a model that can use it. `sparkflash_ops::parts` records
this with exactly one `canonical` entry per `(part, target)`, and the walk never chooses a kernel
by model name.

## 1. The one stack

Kernel sources live in `kernels/` and are built by `sparkflash-kernels/build.rs`. Every cubin is
exposed through `sparkflash_kernels::images`:

| image | accessor |
| :-- | :-- |
| Qwen token/norm/dense/hyper, and the shared tiled decode GEMV body | `qwen4_exp_cubin()` |
| Qwen attention/sparse indexer/KV cache | `qwen4_attn_cubin()` |
| Qwen decode/token routed expert (MoE) | `qwen4_moe_cubin()` |
| Qwen tensor-core bulk prefill | `qwen4_prefill_cubin()` |
| Qwen Gated DeltaNet / KDA scan | `qwen4_scan_cubin()` |
| DFlash 2 block drafter | `dflash2_cubin()` |
| GLM block-scaled bulk prefill | `glm_prefill_cubin()` |
| GLM bulk routed expert (MoE) | `glm_moe_cubin()` |
| GLM decode/token (mHC, router, indexer, absorbed MLA, tiled GEMV wrappers) | `glm_decode_cubin()` |
| GLM sparse MLA attention | `glm_attention_cubin()` |
| Dory nGPT tensor-core projections, attention and cache kernels | `dory_cubin()` |

A model crate may not contain a `.cu` or `.cuh` file and may not carry a build script that
compiles one. The workspace test `sparkflash-ops/tests/one_stack.rs` enforces both, together with
the rule that no model crate depends on another model crate.

## 1a. The DeepSeek forward

DeepSeek's device forward is model code and lives in crates/sparkflash-deepseek-forward, not in
sparkflash-cli. The move kept the module names and crate:: paths, so it is numerically inert; the
CLI is a dispatcher over the forward crate and is under the 34k-line gate. The forward crate
depends on the V4.1 model pack and the shared kernel stack, never on the CLI.

## 2. Typed ops

`sparkflash-ops` is the model-facing surface. Every op is a plan built once at load:

- `gemm` — dense, tile-batched, split-K, grouped MoE, routing and combine.
- `attention` — MLA, grouped GQA, sparse/CSA and split-KV decode.
- `scan` — Gated DeltaNet and KDA.
- `epilogue` — norms, RoPE, activations, quantization, embeddings and Engram gathers.
- `collective` — one-shot, two-shot, reduce-scatter, all-gather and overlapped halves.

A plan fixes shapes, precision, layout and stream. It names the kernel it runs and returns a
`Launch`; it owns no buffers, allocates nothing at launch and never synchronizes the host. The
same plan therefore serves every session that reads the same weights.

```rust
use sparkflash_ops::gemm::DenseGemmPlan;
use sparkflash_ops::plan::{Layout, Matrix, OpPlan, Precision, StreamId};

let plan = DenseGemmPlan::new(
    Matrix::new(rows, columns),
    k,
    Precision::MxFp8,
    Layout::BlockScaled { block: 32 },
    StreamId(session_stream),
    1,
)?;
plan.validate()?;
let launch = plan.launch();
```

The shared launch helpers live beside the plans: `gemv` (tiled decode GEMV and its split rules),
`gdn` (scan geometry), `tp` (two-shot all-reduce) and `dflash` (the block drafter and its
loader). An arm implements `GemvDevice` for its stream and scratch; the drafter's
`DraftDevice` adds the profiling mark and the TP reduce.

### The execution plan

`sparkflash_ops::EnginePlan` is the object an arm opens from. It reads the released
`Architecture`, validates the `TpPlan`, and emits one `PartRequirement` per part instance with
the rank-local `PartGeometry` and `Precision`. `KernelResolver` maps each requirement to a
loadable `PartKernel` by target, shape and availability. A backend implements `EngineBuilder`;
it does not parse the configuration again.

The boot sequence is:

```rust
let plan = EnginePlan::from_json(&text, Target::Sm121, world, rank)?;
let resolver = KernelResolver::new(Target::Sm121, Availability::Runtime);
if !resolver.failures(&plan).is_empty() {
    // refuse by part name before opening a weight
}
let backend = MyArm::open(&plan, &weights)?;
```

See [engine/02-execution-plan.md](engine/02-execution-plan.md). The released-config test in
`crates/sparkflash-ops/tests/engine_plan.rs` is the executable contract: all three configs must
plan and resolve at TP=1 and TP=4.

### The one documented direct-launch arm

Dory's nGPT kernels are fused bodies the shared plans do not describe: the looped normalized
residual, the per-head norm folded into the rope-and-cache launch, and the draft exit that runs
the output suffix on an earlier loop's latent. Its device arm therefore launches `dory_*` by
name from load-time geometry rather than composing a `LayerPlan`, and its boot gate is the
catalogue (`sparkflash_ops::parts`) rather than the plan resolver. This is a deliberate,
single-arm exception, not the bring-up path: the arm still fixes its shapes, precision, layout
and stream at load, allocates no device buffers at launch (the serving loop does build small
per-frame and per-batch worklists), and refuses a checkpoint whose shapes the wide tile cannot
run. A new arm composes the typed plans and implements `GemvDevice` as this guide
describes.

## 2b. The contract: a schedule plus its unique kernels

An arm does not write a serving loop. It supplies a schedule (how its walk steps rows) and the
kernels only it has; everything else is shared and named here.

**The schedule** is `sparkflash_runtime::Walk` ([engine/03-runtime.md](engine/03-runtime.md)):
one walk over some rows of some sequences (`Pass` says which shape), a sequence that grows,
gives back and rewinds, a head over the last row, and the layout its state uses. A drafting arm
adds `Drafter` (draft every chain, one verify walk, accept per sequence) and the runtime decides
the rows each round gets (`share_rows`). A tensor-parallel group adds only its wire
(`GroupWire`: admission, sampler and image frames); the tick group (`TickGroup`) runs rank 0's
decisions on every rank. The engine (`TickEngine`, served through `sparkflash_runtime::serve`,
`serve_group` and `mirror`) owns slots, the engine thread, admission, the prompt scheduler and
reclaim.

**What a new arm inherits.** Each strategy below is built once; the arm supplies the hook in the
middle column and gets the rest. The default is what an arm that writes nothing gets, and the
gate is the served check (`scripts/served-gate.py`, any model by port) that proves it on the arm.

| strategy | the arm supplies | default without it | served gate |
| --- | --- | --- | --- |
| fused lanes | `step` over several slots, each row computed as alone | (required) | `lanes`, `lanes --sampled` |
| prompt parts | `begin_prompt` / `next_part` over the chunks its prefill already runs | whole prompts | `interleave` |
| drafting, joint rounds | `RoundArm` (open, verify) and a `StepWall` | no drafts | `lanes` |
| multi-token runs | `ready`, `give_back` | one step a reply | `lanes` |
| prefix reuse | `history` and `cached_tokens`, and a prefill that forwards only past the history; an arm that commits a verify at once keeps no row past the end of turn (`sparkflash_model::kept_to_end`), or its sequence runs past the reply | renewed every request | `turns` |
| a thinking-off turn that the next turn continues | `PromptCodec::close_thinking`, only when the template writes an empty reasoning block other than DeepSeek's `</think>` (GLM appends `</think>` to the open `<think>`) | the trailing `<think>` becomes `</think>` | `turns` (plain) |
| a reply whose text re-encodes to its ids | nothing: the host's decode holds a character a token left unfinished until the next completes it, and releases it at the end | | `turns` |
| renewal for a prompt that does not continue | nothing: `continues` defaults to "the prompt extends the history" | | `turns` (the cold rerun is cached 0) |
| rewind to the last prompt | `rewind_prefix` over a `sparkflash_kernels::checkpoint::Checkpoint` of the regions its caches do not hold (recurrent memory, convolution histories, a drafter's last rows, a block drafter's context rings whole, since a reply longer than a block overwrites the window before the point), saved at each prompt's end and restored when `sparkflash_runtime::resume` says `Rewind`; `continues` and the host's claim take the point by default, even while `history` names none, and the host respells a next turn that merges the point's last id (`<think>\n\n` after a held `<think>`, `\n`); `SPARKFLASH_REWIND=off` keeps none | a re-rendered reply starts cold | `turns` (think-dropped; a cut first turn) |
| a run its caller ended early, on an arm that commits a verify at once | `give_back` that cuts the history and names none while the state is ahead: the rewind point serves the next prompt | the slot is renewed | `turns` (a cut first turn: REWOUND) |
| shared lane memory | where each cache that grows with a sequence lies (`sparkflash_kernels::lane_memory::SequenceLanes::plan` over a `Layout`), a `SequenceLane` a sequence reserves, `back` before a forward writes rows and `room` from `SequenceLane::room`; the pool, its budget (one whole sequence, `SPARKFLASH_LANE_MEMORY_MB`), zeroing, refusals and release are the helper's | each slot owns its whole context | `lanes` |
| a tensor-parallel group | a `GroupWire` | one rank | every gate, on the group |

A hook an arm skips shows in its ledger line as off, with the reason, so the gap is visible at
boot rather than found later.

**An arm that schedules its own lanes** (Dory's recurrence ring keeps a batch on every rank at
once, which one fused walk a step cannot) still serves through the session contract, and keeps
conversations as the fused arms do: `reusable_prefix` names the ids its lane was fed while the
lane holds them, `rewind_prefix` the last prompt's end, `cached_tokens` the hit, and its prefill
takes the prompt as `sparkflash_runtime::resume` says; `park` lets go of whatever runs ahead of
the history (Dory's guessed rows) and takes the lane out of the counts that schedule running
lanes. The host's kept pool does the rest. Such an arm builds its own `Ledger` and prints it at
open, so its line reads as every other arm's.

**The ledger** is how an arm says what it does with that schedule. `WalkCapabilities` states the
shapes the walk takes and what its own kernels decide; `Runtime::ledger` turns that into a
`sparkflash_runtime::Ledger` stating each strategy natively, looped or off, with the reason; the
engine adds what it decides and prints the line at boot. A strategy the arm leaves unstated prints
as off, so an arm cannot appear to support what it does not.

**Shared launch rules.** A BF16 projection's kernel comes from `sparkflash_ops::projection`:
`route_bf16` with the arm's `WidePolicy` chooses the row GEMV, the wide tensor-core tile, the
skinny kernel or a refusal, and `WideTile::call` gives the wide tile's grid, block and arguments.
A new arm picks a policy (or adds one, with a gate) rather than writing its own row-count test.
The decode GEMV's splits are `sparkflash_ops::gemv`'s. Two more tiles have their rules there: the
split-K BF16 tile (`split_k_slices` cuts the inner dimension, the cut that decides a row's sums,
and `SplitTile::call` gives the launch) and the block-scaled FP8 tile (`fp8_tile_fits`,
`Fp8Tile::call`, or `Fp8Tile::grid` for a body with its own arguments), which Qwen's and GLM's FP8
projections both launch. An arm adds only what its own operands need (alignment, a slice start).

**Executed plans.** Where a typed plan carries everything its launch needs, the arm launches the
plan instead of its own copy of the geometry: a linear-attention scan is
`ScanPlan::for_architecture(&architecture, world, rank, kind, rows)`, the same plan the boot
resolver checked, and the arm launches its kernel on its grid and block after checking it against
the loaded weights' shapes. Grouped-query attention is the second: `GqaBody::choose(query_groups,
selected, granted)` names the body (four or six heads a block, or the per-head kernel), its shared
memory and its grid (`GqaBody::grid`), the same rule the rank's `AttentionPlan` names at boot, so
an arm loads the bodies in `GQA_GROUP_KERNELS` and launches what `choose` says over the ones the
device granted. MLA attention, the routed experts, the indexer and the residual mixers still
launch from the arm's load-time geometry, because their plans do not yet carry the cache layouts,
work lists and split cuts those launches take. A new arm should launch the scan and GQA attention
from the plan and the rest through the shared helpers in section 5, and should not add a new
parallel launch rule for a part another arm already runs.

**What stays in the arm**: its weights' layout and loader, the kernels unique to the model
(Inkling's convolutions and relative attention, Engram, Dory's loop pipeline), its sampler wire,
and the order of its layer walk.

## 3. Configuration

Every arm has one typed TOML profile:

| model | profile |
| :-- | :-- |
| Qwen3.8-Flash-Next (NVFP4 and FP8) | `crates/sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml` |
| GLM-5.3-Flash | `crates/sparkflash-model-glm5-next/config/glm53_flash.toml` |
| DeepSeek-V4.1-Flash | `crates/sparkflash-model-deepseek-v41/config/deepseek_v41_flash.toml` |
| NVIDIA Nemotron Labs Dory1.0-4B (BF16) | `crates/sparkflash-model-dory/config/dory1_0_4b_bf16.toml` |
| thinkingmachines/Inkling-2-Small (NVFP4) | `crates/sparkflash-model-inkling2/config/inkling2_small.toml` |

The FP8 release shares that profile: the switches select arithmetic, and the one difference
between the two Qwen checkpoints is the routed experts' storage (`ExpertQuant`), which the
checkpoint's own `quantization_config` declares. A profile opens with the registry model it
configures, beside its family and weights:

```toml
model = "zai-org/GLM-5.3-Flash"
family = "glm5-next"
weights = "glm5.3-flash-fp8"
```

The declaration check reads `model` to know whose references a changed setting needs recaptured
([NUMERICS.md](NUMERICS.md) §5), and a `sparkflash-model` test holds it to the descriptor that
`weights` selects.

`sparkflash_model::profile::ModelProfile` parses with `deny_unknown_fields`, validates ranges and
prints the resolved values at boot. The environment is an override only; the complete override
table is `sparkflash_model::profile::ENV_OVERRIDES`. A value outside a field's domain is refused,
not silently ignored.

A section groups the switches one mechanism owns. `[drafter]` is the speculative drafter's
kernel variants, the ones an arm was qualified on:

```toml
[drafter]
markov_mma = true
moe_dedup = true
head_fp8 = true
seed_split = true
row_table = true
```

Two sections are not per-model facts: `[plan]` holds the shared routed-expert plan's block size
(`threads`, 512 by default, clamped to one block), which is tuning and not layout — the kernel
derives every loop bound from `blockDim.x`, so any value in range computes the same plan.

The applier is table-driven: `BOOL_FIELDS`, `COUNT_FIELDS`, `F32_FIELDS`, `F64_FIELDS` and
`TEXT_FIELDS` map a field name to the struct member it writes, and the closed-set fields
(`prefill.activation`, `attention.latent_cache`, `diagnostics.probe`, `speculation.depth_policy`,
`drafter.logits`) parse their variants in `apply_one`. Two tests hold the table to the aliases:
`every_override_settles_on_one_typed_field` refuses a field with two setters or none, and
`every_override_moves_its_field` applies each alias and refuses one that leaves its field at the
default. A new alias needs a table row, not a new `match` arm.

Issue #20's switches are typed the same way: `prefill.index_pack_append`
(`SPARKFLASH_INDEX_PACK_APPEND`, on unless `0`, which is the re-pack A/B arm),
`prefill.index_pack_dedupe`/`_log`/`_verify` (`SPARKFLASH_INDEX_PACK_DEDUPE|LOG|VERIFY`),
`diagnostics.moe_plan_in_cta` (`SPARKFLASH_MOE_PLAN_INCTA`, the in-block A/B arm of D102),
`diagnostics.moe_dedup` (`SPARKFLASH_MOE_DEDUP`, `0` for the per-row arm),
`diagnostics.moe_plan_stats` (`SPARKFLASH_MOE_PLAN_STATS`) and `diagnostics.moe_ids_dump`
(`SPARKFLASH_MOE_IDS_DUMP`, a layer number). `sparkflash-kernels` does not depend on
`sparkflash-model`, so the forward resolves the two MoE switches into
`MoeRoutedOptions` when it opens the routed device, rather than the launcher reading the
environment once a launch.

Rules for a new switch:

1. add a typed field to the profile;
2. add its environment alias to `ENV_OVERRIDES` (and to a setter table), or do not read it from
   the environment at all;
3. document the field and alias in this guide;
4. never let a behaviour be reachable only through an undocumented variable.

`diagnostics.vision` is the neutral vision switch. Its alias `SPARKFLASH_VISION` is on unless
it is `off` (or the profile's other off spellings, `0` and `false`); the profile's parser
refuses anything else. Inkling's group reads the same variable at open as well as through the
profile, because every rank must choose the same engine: a rank 0 with the tower keeps the fused
group, and a text-only rank 0 takes the tick group and refuses an image before any frame. The
per-model `SPARKFLASH_INKLING_VISION` is refused by `REFUSED_ENV` with `SPARKFLASH_VISION` in
the message. The deployed site file that still sets the old name changes in the deploy that
carries this rename, and that deploy waits for the owner's go-ahead; the repository does not edit
the deployed file.

### What still reads the environment directly

Three classes stay raw, and none of them moves a served response's arithmetic:

- **Lookup seams.** `ModelProfile::apply_process_env` and `parts::missing_loadable_with` take a
  lookup closure; the CLI passes the process environment, and a test passes its own.
- **Operational identity and transport.** `SPARKFLASH_BUILD_ID`, `SPARKFLASH_BOOT_LOG`,
  `SPARKFLASH_SERVE_SCRIPT`, `SPARKFLASH_SERVE_KEEP_ALIVE_MS`, `SPARKFLASH_MODEL_ALIASES`,
  `SPARKFLASH_STALL_SECS`, the allocator kill switch, `SPARKFLASH_CUBIN_DIR`,
  `SPARKFLASH_CONVERTED_DIR`'s sibling resolvers, `SPARK_COMM_PEERS` and
  `SPARKFLASH_CHECKPOINT` name *where* the process runs, not what it computes.
- **Harness-only knobs.** `tp4_host`'s reference-thread, prefill-mode and prompt-logprob probe
  variables, and `tp4_forward`'s sampler (`SPARKFLASH_DSPARK_TEMPERATURE`, `_TOP_P`, `_TOP_K`,
  `_SEED`) belong to a diagnostic subcommand, which the profile does not describe.

## 4. Bring-up parity

`sparkflash_ops::parity` is the layer-by-layer gate. Build a `LayerPlan`, validate it, run the
device path, and compare each named buffer with the reference forward:

```rust
use sparkflash_ops::parity::{LayerPlan, ParityReport, Tolerance};
use sparkflash_ops::plan::OpKind;

let mut report = ParityReport::new("layer 0");
report.record("after_attention", OpKind::MlaAttention, &reference, &device, Tolerance::F32_ROUNDING)?;
assert!(report.passed(), "failed: {:?}", report.failures());
```

A case names the first element outside the tolerance, not just the layer. A new arm runs the same
harness with its host forward as the reference; the harness has no model-specific code.

## 5. Shared homes

Everything an arm needs outside its own forward already has one home. Add to that home, or move
the mechanism there; never copy it into the arm.

| mechanism | home |
| :-- | :-- |
| a routed-expert plan (count by expert, order the slots, cut items) | `kernels/moe_plan_sm121.cu`, `Image::MoePlan`; every arm launches it rather than carrying its own counting pass |
| model id to checkpoint (the Hugging Face cache) | `sparkflash_weights::hub` |
| `--synthetic` prompts without a tokenizer | `sparkflash_ops::codec::ByteCodec` |
| a chat message's role, with OpenAI's `developer` folded onto `system` | `sparkflash_model::prompt::{message_role, template_role}`; a renderer reads every role through it |
| `config.json` errors and a field that is one value or a list | `sparkflash_model::config::{ConfigError, OneOrMany}` |
| an arm that renders every text channel (tools, thinking, reasoning effort) | `Capabilities::ALL`; forced decoding is the driver's, which the host and TP drivers add |
| a tensor's byte length against its slot | `sparkflash_weights::weights::check_length` |
| walk-ordered rank files: placement, index, writer, read-back cursor | `sparkflash_weights::rank_file::{SlotPlan, SlotIndex, SlotFileWriter, SlotCursor, SlotRecords}` |
| `sparkflash convert` for such a pack (ranks, read-back, refusals) | `SlotPack` in `sparkflash-cli/src/convert.rs` |
| `sparkflash serve` at TP=1 and in a tensor-parallel group: the engine around a walk | `sparkflash_runtime::WalkArm` (`serve`, `serve_group`, `mirror`, `lane_limit`) |
| the family's boot on either serving path: fixture, configuration, descriptor, tokenizer, open | `PackArm` in `sparkflash-cli/src/serve/pack.rs` |
| the descriptor a pack reports (window and vocabulary from the checkpoint) | `sparkflash_model::arm::PackIdentity` |
| a session's KV claim and the `--concurrency`/`--kv-budget-gib` admission | `sparkflash_model::arm::KvPlan`, printed and refused by `pack::admit` |
| which drafts a verify keeps, the tokens it queues, the depth, the counters | `sparkflash_model::spec::{verdict, Queue, Speculator}` |
| sessions stepped together as one pass over the weights (`--fuse-batch`), lanes and multi-token runs | `sparkflash_runtime::{Walk, TickEngine}` ([engine/03-runtime.md](engine/03-runtime.md)) |
| several sequences' drafts verified in one walk, and the rows each is given | `sparkflash_runtime::{RoundArm, run_round, share_rows, StepWall}` |
| a walk's CPU profile tests (compute rows, tick config, prompt chunks, joint and speculative rounds) | `sparkflash_runtime::profile_checks!` over the arm's `testing::ShapeWalk` |
| a fixture's stored-format names and token-pass bytes | `sparkflash_runtime::testing::{format_rule, token_bytes}` |
| splitting a streamed answer into text and `<tool_call>` bodies | `sparkflash_api::tool_block::BlockSplitter` (the family parses the body) |
| deterministic fixture weights and hash-table sizes | `sparkflash_weights::rng::{splitmix64, is_prime}` |

A served family implements `PackArm` (its fixture, configuration, descriptor, tokenizer and
open) and states the serving glue its backend supplies as `WalkArm` (its walk, its group wire,
the place it refuses and its group's engine); the runtime starts both paths. Each release of it
implements `Release` on its `sparkflash_model` marker type (the family's `PackArm`, a display
name, a fixture name and a tiny configuration; the model id, profile, default context, output
cap and response dialect come from its `ARMS` entry) and adds one
`TypedPack::of::<R>()` row to `TYPED_PACKS`, which serves both `--model` and a dynamic
`--checkpoint` of that family. Its descriptor is `PackIdentity`'s, its cache claim
is a `KvPlan` derived from its own geometry, and its speculative path is `spec`'s: a decode drafts,
one verify decides what survives, and the queue hands the rest back one token at a time. The shared driver then finds the checkpoint
(`--checkpoint`, the hub cache or the fixture), derives the context on every rank of a group from
the same configuration, and refuses at parse time any option the arm does not apply
(`Pack::unapplied`). A group whose arm opens lanes (`PackArm::open_lanes`: GLM's and Qwen's)
steps the lanes its `ARMS` entry states (`GroupDefaults::lanes`, four today) by default
(`--fuse-batch --concurrency N` for another count, `--concurrency 1` for one request at a
time); a group whose arm opens none serves one request at a time and refuses `--concurrency` and
`--fuse-batch`. `--kv-budget-gib` is refused in any group,
and an arm without a fused batch or a KV budget says so at TP=1. A directory of rank files named by an environment variable resolves through
`converted_rank_file`, which refuses a directory that holds no file for the rank.

## 6. What you supply

Only what this family alone has:

- how its released config maps onto parts (`Architecture`);
- the arithmetic of a part no other model has;
- its weight names, tokenizer and chat template.

Everything else is read from one registration or refused by the compiler:

- **The compiler.** `sparkflash_ops::parts::{CATALOGUE, WAIVERS}` and the const assertion over
  `Part::ALL` make a GEMM-class part without a tensor-core body, or without a `WAIVERS` line
  naming the gap and its measurement, a compile error. `Architecture::parts()` is the only
  description of what the model needs; `parts::missing` is the catalogue's answer.
- **The `ARMS` table.** `sparkflash_model::ARMS` is the one model registration: arm id, model
  id, weights, profile, default context, ledger, tokenizer and prompt dialect, the converter its
  checkpoint goes through, its group's default context and lanes, its row of the web console's
  deployments, and its crate and environment switches. `MODELS` is the descriptor list. `DEFAULTS` holds DeepSeek's measured defaults; an
  arm profile departs from one only with `measured = "docs/…"`.
- **`sparkflash onboard ORG/REPO`.** Before any code is written it prints the parts `CATALOGUE`
  already has, the parts that need a `PartExec` (no tensor-core body in a pass and no `WAIVERS`
  line), the stored format of each weight class, the fit at TP 1, 2 and 4, and the stored
  bytes-per-token floor from `stored_bytes_floor`/`bytes_per_token_floor`. `--config FILE`
  reads a document from disk; otherwise it uses the hub cache and only then the Hub.
- **`scripts/onboard-gate.sh <arm>`.** Runs `scripts/served-gate.py all` and writes the first
  numerics reference through `scripts/numerics.py`, the registry's authority. It prints the exact
  commands (dry run by default) and states that the generated conformance runner is still a stub.

### A family with one part of its own

The shared dense arm (`sparkflash-model-dense`) runs a list of parts. It builds the list from
`Architecture::steps()`: for every layer the token mixer, then the feed-forward. Each part is a
`sparkflash_runtime::PartExec`, made by the constructor a `PartRegistry` holds for the step's part
kind. `Dense::catalogue()` registers the four the arm has: grouped-query attention, the Gated
DeltaNet scan, the dense SwiGLU MLP and the routed experts. The walk matches no enum.

A family whose layer needs a part the arm lacks writes three things.

1. **The part.** `impl PartExec<Dense> for ItsPart`, four required methods and no defaults:
   `state` (the regions it adds to a sequence's lane, positional or recurrent), `split`
   (`RankSplit::Summed` when a rank leaves a partial the group sums, `Replicated` when every rank
   computes the whole rows), `rows` (the prompt body) and `token` (the token body). A body reads
   the normed rows at `pass.input()` and leaves the branch's rows at `pass.output()`. A part whose
   token body is its prompt body says so in one line, `self.rows(pass)`.
2. **The registration line.**

   ```rust
   let mut parts = Dense::catalogue();
   parts.replace(Part::DenseMlp, |open: &mut OpenCx<'_>| ItsPart::open(open))?;
   let backend =
       DenseDeviceBackend::open_with(descriptor, &architecture, &source, context, &parts)?;
   ```

   `register` adds a kind the registry does not hold and refuses one it does; `replace` takes a
   kind over. The constructor uploads this rank's share through `OpenCx` (`source`, `plan`,
   `keep`, `record`, `pack`, `scratch`, and the four catalogue constructors, which a part may
   wrap).
3. **Its kernel.** The constructor loads it through `open.cuda()` and the body launches it with
   `pass.launch`, on the walk's stream.

It no longer writes a walk, a `match` arm in one, the norm before the part, the group's sum and
the residual add after it, the sequence's lane memory, its rewind point, the sampler, the lanes or
anything about serving. The arm lays out each lane from what the parts' `state` declares.

A step whose part kind has no registered executor is refused when the arm opens, by the part's
name, before a kernel image or a tensor is read. The walk never skips a part.
`crates/sparkflash-model-dense/tests/part_plugin.rs` is the worked example: a part defined
outside the crate, registered, and run over a prompt and two token steps.

Two limits remain. The dense walk runs a step's attention and feed-forward roles and refuses the
others by name. And a part kind must be one `Architecture::steps()` names:
`sparkflash_ops::parts::dense_device_refusal`, the arm's layer shapes and
`sparkflash_ops::naming::resolve` still describe the four catalogue kinds only, so a kind with a
new name also needs its `Part` entry and those three to accept its layer.

## 7. The dynamic Hugging Face path (no new arm)

A checkpoint does not need a new `PackArm` to convert or serve. `sparkflash serve --model
ORG/REPO` and `sparkflash serve --checkpoint DIR` resolve the snapshot through
`sparkflash_weights::hub`, parse its `config.json` with the shared `Architecture`, plan it with
`EnginePlan`, and check every requirement with `KernelResolver`. The descriptor is
`ModelDescriptor::from_architecture`, the prompt dialect is the family's `ARMS` entry when
it states one and the `tokenizer.json` markers otherwise (GLM's `[gMASK]`/`<sop>`, ChatML's
`<|im_start|>`/`<|im_end|>`), and the profile is the
canonical preset for a released family or `ModelProfile::default()` plus the declared environment
overrides for a novel one. `MODELS` remains the canonical preset list, not a closed gate.

`sparkflash convert --model ORG/REPO` dispatches from the same plan: the family's `ARMS`
entry names the converter its checkpoint goes through (the Qwen and GLM packs' own, Inkling's
plan-ordered writer, DeepSeek's B12x converter), and a family with no entry goes to the generic
dynamic writer.

**Phase-4 status.** `EngineBuilder`, `ArchitectureTensorPlan` and `SlotPack` are merged. The
released Qwen and GLM arms implement `EngineBuilder` (`PartsDeviceBackend`), and `SlotPack` is
plan-order TP1. `ArchitectureTensorPlan` names every family: a released one in its pack's
spelling, any other in the one shared convention of `sparkflash_ops::naming` (runtime step 15).
`NamingDialect::probe` reads a checkpoint's choices inside the canonical Hugging Face spelling
(its layer root, its post-attention norm, per-head norms, the router's and GDN projections'
names, a shared expert's gate, a tied head) from its headers, `NamingDialect::names` spells every
tensor from the `Architecture`, and `ArchitectureTensorPlan::named` plans them (shapes, rank
splits, roles, FP8 tile scales) in forward order. The converter, the host reference and the
web's fit read that one plan, and:

- a novel family that is a grouped-query decoder (embedding, RMSNorm, GQA with or without
  the per-head output gate and `q_norm`/`k_norm`, plain `rotate_half` RoPE over the leading
  rotated dimensions, Gated DeltaNet layers between them or not, a dense SwiGLU (clamped by a
  `swiglu_limit` or not) or routed experts, a plain residual and the head), BF16 or with its projections and experts in
  128 x 128-tile FP8, is served
  on the device by the shared dense arm, `sparkflash-model-dense` (runtime step 19), with no code
  of its own: it reads the `Architecture`, the shared names, `DraftUploader` and the typed
  `sparkflash_ops::calls`, and its prompt projections take the shared wide tile
  (`WidePolicy::DENSE`). A routed layer (runtime step 21: a softmax, sigmoid or sqrt-softplus
  router, its top-k with or without renormalising, a sigmoid or sqrt-softplus top-k on score
  plus a noaux_tc `e_score_correction_bias` (runtime step 25), a routed scale, one shared expert
  with or without its sigmoid gate) runs the released arms' parts: `q4e_route`, `MoePlanCall::runs`, the
  token path's expert passes in their BF16 form and `q4e_moe_combine`. A release that stores its
  projections and experts as E4M3 codes under one inverse scale a 128 x 128 tile (runtime step
  22, the Qwen3 FP8 form) runs Qwen's and GLM's FP8 kernels: the FP8 tile (`Fp8Tile`, by
  `projection::route_fp8`) for a prompt chunk, the tiled E4M3 GEMV for decode rows, and the
  expert passes' FP8 form; each tensor is uploaded as it is stored, through
  `DraftUploader::mat_e4m3_tile`. A Gated DeltaNet layer (runtime step 23, a hybrid family's
  `linear_attn`) runs the scan image's chunk kernels Qwen's prompts run, through
  `sparkflash_ops::gdn`'s typed calls (the conv over a segment's rows and its history advanced
  past them, the prep, the scan its `ScanPlan` resolves for the head pair, the gated norm) at
  every row count, decode rows included, so a row's state is the same bits alone, in a prompt or
  in a fused step; each lane holds its conv history and delta-rule state, and the rewind point
  copies them (`Checkpoint`) at each prompt's end. Every such layer shares one geometry, at a head
  pair the scan is compiled for ((64, 64), (128, 128), (128, 256)), two to nine conv taps, value
  heads a multiple of the key heads and a SiLU or sigmoid gate.
  `sparkflash_ops::parts::dense_device_refusal` is its gate, and
  it names what it refuses (a window, a rope scaling, another activation, a bias, a projection or
  expert format other than BF16 and the 128 x 128 FP8 tile, a width that splits a scale tile, a
  selection bias, a widened residual...). It gates on its own launch list rather than the kernel
  catalogue, whose rows pin the released presets' shapes. Its check is a per-layer parity report
  against the host reference (`cli/src/serve/dense_parity.rs`), not a recorded digest: it keeps
  BF16 keys and values where the reference keeps f32. It serves at TP1 on the serving contract
  (runtime step 20): `DenseLanes` is the runtime's, so the shared `TickEngine` steps its lanes
  together (four by default; `--concurrency` or the profile's `runtime.concurrency` sets them,
  and a mismatch is refused), feeds long prompts in 256-row chunks between other lanes' steps,
  keeps each conversation for its next turn, and goes back to the prompt when a turn parts from
  the reply (`SPARKFLASH_REWIND=0` keeps only extensions). Each lane's keys and values grow in
  `SequenceLanes` memory, and the boot prints the arm's ledger. A member of a fused step gets the
  bits it would get alone, and `scripts/served-gate.py` gates it served
  (`scripts/dense-synthetic-checkpoint.py` writes a random checkpoint of a family no pack names
  for that); `SPARKFLASH_DENSE_DEVICE=0` keeps such a checkpoint on the host reference and `=1` refuses to
  boot unless the arm opens. Unset, the boot line names which backend answers and, on the host,
  why the arm did not. The same arm serves a `--tp4-rank` group (runtime step 24):
  `DenseDeviceBackend::open_rank` holds the rank's share of every tensor as the shared `TpPlan`
  splits it (query heads, key/value heads shared when the world exceeds them, both Gated DeltaNet
  head counts, every MLP and expert width; FP8 shares on whole scale tiles), every branch's
  partial rows are summed in rank order through `sparkflash_ops::tp::RowSums` (the host seam, or
  NCCL's one-shot and two-shot device legs, which give the same bits), and rank 0's lanes run on
  every rank through the tick group (`enable_group_fusion`, `mirror_group`). A worker never
  detokenizes; the host reference runs alone, so a checkpoint the arm refuses is refused at a
  world above one;
- any other novel family is served by the CPU reference backend in
  `sparkflash-cli/src/serve/dynamic_host.rs`, which executes the parts in
  `sparkflash-cli/src/dynamic.rs::DYNAMIC_PARTS` (embedding, RMSNorm, GQA, Gated DeltaNet,
  dense SwiGLU, routed/shared MoE, LM head, plain residual) with the shared names and f32
  arithmetic. That list is
  `sparkflash_ops::parts::HOST_REFERENCE_PARTS`, and it is the capability gate for both the CLI
  and the web control plane: the CUDA catalogue answers a different question. The GQA arithmetic
  follows Hugging Face: `rotate_half` RoPE pairs dimension `i` with `i + head_dim/2`, an
  `attn_output_gate` query projection interleaves query and gate per head, and optional
  `q_norm`/`k_norm` weights run before RoPE. A declared `rope.scaling` is applied for
  `default`/`linear` and refused by name for any other rule. It is a bring-up path, not a
  throughput claim, and it plans at TP1;
- a novel family's rank file is the same `SlotPlan`/`SlotFileWriter` container holding the raw
  checkpoint tensors in the shared plan's order (`DynamicRecords` in `convert.rs`), not a device
  layout. The generic writer is TP1: a world above one is refused rather than writing several
  whole copies labelled as shards. `convert --inspect` recognises the `dynamic` rank file and
  verifies it, and a conversion with no `--output-dir` writes beside the working directory,
  never inside the Hugging Face cache. A family `ArchitectureTensorPlan` names can use
  `SlotPack` directly; the novel path cannot yet;
- a family this backend does not execute (MLA, KDA, CSA2, an indexer, a hyper-connection, PLE
  or Engram) is refused with the exact host-part list before a kernel is consulted. An MTP stage
  and a vision tower are optional: the host ignores both and runs the backbone, and a quantized
  tensor belonging to either (under `mtp.*`, `model.visual.*`, or an extra decoder layer) does
  not refuse the backbone;
- the host decodes BF16/F32 tensors and 128x128 tile-quantized E4M3 weights. An E4M3 weight
  needs its `<name>_scale_inv` grid beside it, and the decoder multiplies every element by its
  block's inverse scale. NVFP4, MXFP8 and unscaled FP8 storage are still refused;
- `--tokenizer FILE` wins over `<checkpoint>/tokenizer.json`, and a tokenizer carrying neither
  GLM's nor ChatML's markers is refused instead of falling back to `ByteCodec`;
- a single-file `model.safetensors` is accepted without an index: the header synthesizes the
  weight map. A checkpoint whose quantized storage this path cannot decode (ModelOpt `U8` +
  scales, NVFP4, or an E4M3 tensor without its `_scale_inv` grid) is refused at boot with the
  format and tensor name;

Replacing the remaining novel-family host walk with a shared device `EngineBuilder` is runtime
steps 18-20 ([engine/03-runtime.md](engine/03-runtime.md)).

The web control plane evaluates a released family with the same `EnginePlan` +
`KernelResolver` at every width its device arm runs (`crates/sparkflash-web/src/fit.rs`), and
sizes each width from the tensor plan and the KV cache plan against the per-rank memory budget
(`NODE_MEMORY_BUDGET_BYTES` less `MEMORY_FLOOR_BYTES`). `supported` means a device engine serves
the checkpoint at its launch width; `context_tokens` comes from the checkpoint, and `missing_parts`
names what is missing. A novel family with no registry entry is never `supported`: it is
checked against the shared `HOST_REFERENCE_PARTS` set only so an explicit
`engine.kind = "dynamic"` plan can be validated (`host_reference`), and the browser reports it
as missing a device engine. A family registered with a `ModelProfile` whose `engine` is
`dynamic` is judged by the kernel catalogue when a device arm reads its stored formats
(`sparkflash_ops::parts::dynamic_device_arm`, the same rule the CLI's dynamic route and the
arm's load apply), and otherwise by the host-reference gate, reported as running on the CPU host
reference. Qwen3.6-35B-A3B is the first such entry: its FP8 release runs on the Qwen device arm,
a BF16 checkpoint of the family on the host. A cached or explicit
checkpoint is also judged by its safetensors headers, so storage-only quantization is refused
exactly as the CLI refuses it; a Hub-only preview remains declaration-level until the shards are
downloaded. An explicit `model.checkpoint` is authoritative over a cached `model.id` for both the
window and support decision. Selecting a cached model never rewrites the cluster plan. A new
family's device arm makes it appear: add its `ModelProfile` entry with the widths it runs
(`worlds`), and the browser judges it from then on.
