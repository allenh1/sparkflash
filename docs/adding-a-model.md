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

## 3. Configuration

Every arm has one typed TOML profile:

| model | profile |
| :-- | :-- |
| Qwen3.8-Flash-Next (NVFP4 and FP8) | `crates/sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.toml` |
| GLM-5.3-Flash | `crates/sparkflash-model-glm5-next/config/glm53_flash.toml` |
| DeepSeek-V4.1-Flash | `crates/sparkflash-model-deepseek-v41/config/deepseek_v41_flash.toml` |

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

Rules for a new switch:

1. add a typed field to the profile;
2. add its environment alias to `ENV_OVERRIDES` (and to a setter table), or do not read it from
   the environment at all;
3. document the field and alias in this guide;
4. never let a behaviour be reachable only through an undocumented variable.

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
| `config.json` errors and a field that is one value or a list | `sparkflash_model::config::{ConfigError, OneOrMany}` |
| an arm that renders every text channel (tools, thinking, reasoning effort) | `Capabilities::ALL`; forced decoding is the driver's, which the host and TP drivers add |
| a tensor's byte length against its slot | `sparkflash_weights::weights::check_length` |
| walk-ordered rank files: placement, index, writer, read-back cursor | `sparkflash_weights::rank_file::{SlotPlan, SlotIndex, SlotFileWriter, SlotCursor, SlotRecords}` |
| `sparkflash convert` for such a pack (ranks, read-back, refusals) | `SlotPack` in `sparkflash-cli/src/convert.rs` |
| `sparkflash serve` at TP=1 and in a tensor-parallel group | `PackArm` in `sparkflash-cli/src/serve/pack.rs` |
| the descriptor a pack reports (window and vocabulary from the checkpoint) | `sparkflash_model::arm::PackIdentity` |
| a session's KV claim and the `--concurrency`/`--kv-budget-gib` admission | `sparkflash_model::arm::KvPlan`, printed and refused by `pack::admit` |
| which drafts a verify keeps, the tokens it queues, the depth, the counters | `sparkflash_model::spec::{verdict, Queue, Speculator}` |
| sessions stepped together as one pass over the weights (`--fuse-batch`) | `sparkflash_model::fused::{FusedArm, FusedEngine}` |
| splitting a streamed answer into text and `<tool_call>` bodies | `sparkflash_api::tool_block::BlockSplitter` (the family parses the body) |
| deterministic fixture weights and hash-table sizes | `sparkflash_weights::rng::{splitmix64, is_prime}` |

A served arm implements `PackArm` (its ids, fixture, configuration, descriptor, tokenizer, open
and response layer) and adds a `Pack` variant. Its descriptor is `PackIdentity`'s, its cache claim
is a `KvPlan` derived from its own geometry, and its speculative path is `spec`'s: a decode drafts,
one verify decides what survives, and the queue hands the rest back one token at a time. The shared driver then finds the checkpoint
(`--checkpoint`, the hub cache or the fixture), derives the context on every rank of a group from
the same configuration, and refuses at parse time any option the arm does not apply
(`Pack::unapplied`): a group serves one request at a time, so `--concurrency`, `--fuse-batch` and
`--kv-budget-gib` are refused there, and an arm without a fused batch or a KV budget says so at
TP=1. A directory of rank files named by an environment variable resolves through
`converted_rank_file`, which refuses a directory that holds no file for the rank.

## 6. Checklist

- [ ] the released `config.json` parses through `Architecture`, and every string it selects is
      one this build knows;
- [ ] the arm's `Architecture::parts()` is a subset of the build's catalogue, or the missing parts
      are named in the refusal;
- [ ] the config parses, validates, and its boot summary is in the startup log;
- [ ] nothing in section 5 is reimplemented in the arm;
- [ ] no `.cu`/`.cuh` and no kernel build script in the model crate;
- [ ] no dependency on another model crate;
- [ ] every op is a plan, validated before its first launch;
- [ ] the parity harness passes layer by layer against the host reference;
- [ ] new environment aliases are in `ENV_OVERRIDES` and documented here;
- [ ] the arm is registered in the model registry and has a descriptor;
- [ ] every profile override the arm adds is classified in `sparkflash_model::numerics` as
      arithmetic or neutral, and each configuration it serves has a captured reference in
      `numerics/registry.toml` ([NUMERICS.md](NUMERICS.md));
- [ ] the arm's profile names its registry model (`model`, section 3).

## 7. The dynamic Hugging Face path (no new arm)

A checkpoint does not need a new `PackArm` to convert or serve. `sparkflash serve --model
ORG/REPO` and `sparkflash serve --checkpoint DIR` resolve the snapshot through
`sparkflash_weights::hub`, parse its `config.json` with the shared `Architecture`, plan it with
`EnginePlan`, and check every requirement with `KernelResolver`. The descriptor is
`ModelDescriptor::from_architecture`, the prompt codec is detected from `tokenizer.json`
(GLm's `[gMASK]`/`<sop>`, ChatML's `<|im_start|>`/`<|im_end|>`), and the profile is the
canonical preset for a released family or `ModelProfile::default()` plus the declared environment
overrides for a novel one. `MODELS` remains the canonical preset list, not a closed gate.

`sparkflash convert --model ORG/REPO` dispatches from the same plan: a released Qwen/GLM
checkpoint goes to its pack converter, DeepSeek goes to the B12x converter, and any other
resolved architecture goes to the generic dynamic writer.

**Phase-4 status.** `EngineBuilder`, `ArchitectureTensorPlan` and `SlotPack` are merged,
but their reach is deliberately narrow: the released Qwen and GLM arms implement `EngineBuilder`
(`PartsDeviceBackend`), `ArchitectureTensorPlan` names only the `qwen4_exp` and `glm5_next`
families, and `SlotPack` is plan-order TP1. A novel family has neither a plan builder nor a
device arm, so:

- a novel family is served by the CPU reference backend in
  `sparkflash-cli/src/serve/dynamic_host.rs`, which executes the parts in
  `sparkflash-cli/src/dynamic.rs::DYNAMIC_PARTS` (embedding, RMSNorm, GQA, Gated DeltaNet,
  dense SwiGLU, routed/shared MoE, LM head, plain residual) with canonical Hugging Face tensor
  names and f32 arithmetic. That list is
  `sparkflash_ops::parts::HOST_REFERENCE_PARTS`, and it is the capability gate for both the CLI
  and the web control plane: the CUDA catalogue answers a different question. The GQA arithmetic
  follows Hugging Face: `rotate_half` RoPE pairs dimension `i` with `i + head_dim/2`, an
  `attn_output_gate` query projection interleaves query and gate per head, and optional
  `q_norm`/`k_norm` weights run before RoPE. A declared `rope.scaling` is applied for
  `default`/`linear` and refused by name for any other rule. It is a bring-up path, not a
  throughput claim, and it plans at TP1;
- a novel family's rank file is the same `SlotPlan`/`SlotFileWriter` container holding the raw
  checkpoint tensors in architecture-walk order (`DynamicRecords` in `convert.rs`), not a device
  layout. The generic writer is TP1: a world above one is refused rather than writing several
  whole copies labelled as shards. `convert --inspect` recognises the `dynamic` rank file and
  verifies it, and a conversion with no `--output-dir` writes beside the working directory,
  never inside the Hugging Face cache. A family `ArchitectureTensorPlan` names can use
  `SlotPack` directly; the novel path cannot yet;
- a family this backend does not execute (MLA, KDA, CSA2, an indexer, a hyper-connection, PLE,
  Engram or vision) is refused with the exact host-part list before a kernel is consulted. An MTP
  stage is optional: the host ignores it and runs the backbone, and a quantized tensor belonging
  to the stage (under `mtp.*`, or stored as an extra decoder layer) does not refuse the
  backbone;
- `--tokenizer FILE` wins over `<checkpoint>/tokenizer.json`, and a tokenizer carrying neither
  GLM's nor ChatML's markers is refused instead of falling back to `ByteCodec`;
- a single-file `model.safetensors` is accepted without an index: the header synthesizes the
  weight map. A quantized checkpoint (ModelOpt `U8` + scales, or a declared FP8/NVFP4 format)
  is refused at boot with the format and scale names, because this path does not yet apply block
  scales;

Replacing the remaining novel-family host walk with a shared device `EngineBuilder` and adding a
generic `ArchitectureTensorPlan` naming path is the next step; it is a new plan builder, not a
rebase change.

The web control plane evaluates a released family with the same `EnginePlan` +
`KernelResolver` at every width its device arm runs (`crates/sparkflash-web/src/fit.rs`), and
sizes each width from the tensor plan and the KV cache plan against the per-rank memory budget
(`NODE_MEMORY_BUDGET_BYTES` less `MEMORY_FLOOR_BYTES`). `supported` means a device engine serves
the checkpoint at its launch width; `context_tokens` comes from the checkpoint, and `missing_parts`
names what is missing. A novel family is never `supported`: it is checked against the shared
`HOST_REFERENCE_PARTS` set only so an explicit `engine.kind = "dynamic"` plan can be validated
(`host_reference`), and the browser reports it as missing a device engine. A cached or explicit
checkpoint is also judged by its safetensors headers, so storage-only quantization is refused
exactly as the CLI refuses it; a Hub-only preview remains declaration-level until the shards are
downloaded. An explicit `model.checkpoint` is authoritative over a cached `model.id` for both the
window and support decision. Selecting a cached model never rewrites the cluster plan. A new
family's device arm makes it appear: add its `ModelProfile` entry with the widths it runs
(`worlds`), and the browser judges it from then on.
