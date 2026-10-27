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

# Re-organizing SparkFlash for more than one model

Status: proposal, with Phase 1, the host half of Phase 2, and the first Phase 3 slice implemented.
Baseline: `origin/main` at `20599e6c` (fetched 2026-10-26).

Progress on 2026-10-26:

- `sparkflash-model` exists with `ModelDescriptor`, the compile-time registry and `EngineKind`.
- `sparkflash-api` takes model id, context and output ceiling from `AppState`'s descriptor; the
  V4.1 globals remain only as compatibility constants.
- `sparkflash-web` derives its supported weights and engines from the registry; `model.weights`
  and `engine.kind` are separate selectors (`tp4-device` / `host-reference`, plus the legacy
  V4.1 value).
- `sparkflash models` lists the registered models and engines.
- `sparkflash-model-deepseek-v41` now owns the complete V4.1 host model: config, tensor plan,
  layer plan, reference forward (`forward/*`), loader, sequence fixture, report, prompt ranks,
  and the prompt encoding/tokenizer seam moved out of `sparkflash-api`. `sparkflash-engine` is
  a compatibility facade that re-exports the pack, so every existing path and the arithmetic are
  unchanged. The captured native-rounding fixtures moved with the forward tests.
- `sparkflash-weights` now owns the generic storage and host-arithmetic layer: `dtype`, `quant`,
  `parallel`, `TensorSpec`, the weight-source contract, the safetensors container and
  `linalg`. The V4.1 pack re-exports them under the old module paths, so no caller changed. The
  plan-shaped conformance wrapper stays in the pack.
- `sparkflash-model` now defines `PromptCodec` and `PromptError`; `AppState` holds
  `Arc<dyn PromptCodec>` and the V4.1 `TokenizerSeam` implements the trait. The HTTP layer maps
  the codec's parameter/code pair, so the router no longer imports concrete encoding-error types.
- A toy dense model in `sparkflash-model/tests/toy_dense.rs` defines its own descriptor, ASCII
  prompt codec, tensor names and a BF16 block, then runs it through the generic weight and
  arithmetic path. It proves the contracts do not require V4.1 geometry.
- `sparkflash-api` no longer re-exports the V4.1 encoding module; tests and the CLI import the
  pack directly.
- The shared telemetry records (`PrefixMatchPlan`, the KV snapshot and counters, and the Engram
  layer snapshot) now live in `sparkflash-model::telemetry`; `sparkflash-kv` and
  `sparkflash-engram` re-export them so their public paths are unchanged. The KV planner's page
  and window constants are contract defaults that the KV config re-exports.
- `sparkflash-model::vision` defines `VisionCodec`, the prepared-image/span types and an opaque
  `VisionError`; `PromptCodec::vision_codec` returns the model's optional front end, and the
  V4.1 pack adapts `sparkflash-vision` behind it. The API keeps only the wire-level image-part
  handling.
- The reasoning alias table is a protocol-level default in `sparkflash-api`; a test pins it
  against the V4.1 encoding table, which is now a dev-dependency only.
- `cargo tree -p sparkflash-api --edges normal` now contains `sparkflash-model` and generic
  infrastructure, not Engram, KV, vision or the V4.1 pack.
- `sparkflash-model::backend` defines the host execution seam: `ModelBackend` opens a
  `ModelSession` with `prefill`/`decode` step methods, and `StepOutput`, `Sampling` and
  `BackendError` are model-contract types. A backend owns its sampling; the serving layer only
  chooses it per request.
- `sparkflash-serve` now exists with a generic `HostDriver` that implements the API
  `InferenceEngine` over a `ModelBackend`: bounded queue, dedicated engine thread, streaming
  content, stop strings, sample logprobs, and the standard usage/timing envelope. Its toy test
  serves a chat completion through the real router with no V4.1 types.
- The clang CUDA syntax probe now has the stub headers it needs for the `opt_*` kernels on a
  host without a CUDA toolkit.

Still open in Phase 2: the V4.1 loader and synthetic source live in the pack because they consume
its `TensorPlan`; that is a deliberate model-owned boundary rather than an API dependency. The
telemetry schema is still the V4.1 wire shape for every model and is not yet gated on a capability.

## 1. Goal, and what this plan deliberately does not do

The goal is to make "which model is this?" a first-class choice, so that adding a second
checkpoint or a second architecture is a bounded piece of work instead of a fork of the serving
stack. The V4.1-Flash path must remain the default, must remain bit-for-bit compatible where it
is today, and must not lose its performance characteristics.

This is **not** a proposal to build a general-purpose model IR, a graph executor, or a
multi-tenant server. The current code wins by specializing hard for one architecture, one
topology, and one set of kernels. The reorganization should preserve that specialization behind
a boundary rather than dissolve it.

Recommended scope order:

1. Parameterize the *same* architecture (another V4.1-family checkpoint or size).
2. Support a second *dense* architecture on the host/reference path.
3. Add a device backend for the second architecture only when the host path proves the contracts.
4. Consider multiple models in one process only after those work; the recommended deployment is
   still **one model per process**.

## 2. Why the current tree is hard to extend

The workspace is about 360k lines of Rust plus about 20k lines of CUDA. Most of it is excellent
model-specific code; the problem is that the model-specific parts are not separable from the
runtime parts.

| Area | What is true today | Why it blocks a second model |
| :-- | :-- | :-- |
| `sparkflash-engine::config` | One `ModelConfig` with V4.1 fields; the released HF config mapping and validation are here. | Every model is assumed to have MLA, mHC, Engram, DSpark, compress ratios, and the same key names. |
| `sparkflash-engine::{tensor,layer_plan,forward}` | V4.1 tensor names, 43-row layer plan, mHC/MoE/Engram/MTP arithmetic. | These are the model definition, but they live in the crate the runtime calls "engine". |
| `sparkflash-kv::{config,layout}` | Static 40-layer topology, compress ratios, 512-wide MLA pool, 64-token pages, SWA details. | A model with GQA, a different head count, or no compressed KV cannot use the cache layout. |
| `sparkflash-engram` / `scheduler::dspark` | Engram and DSpark are first-class modules of the core. | They are optional features of one family, not obligations of every model. |
| `sparkflash-api` | `MODEL_ID`, `MODEL_CONTEXT_TOKENS`, `MODEL_MAX_OUTPUT_TOKENS`, the V4.1 tokenizer seam, and `deepseek-recipe` are in the protocol crate. | The HTTP layer cannot serve a different vocabulary, context, template, or capability set without code changes. |
| `sparkflash-cli` | About 91k lines; the TP=4 device loop, constants such as `HIDDEN = 5120`, conversion, residency, and serving drivers all live here. | A second model either edits the CLI or copies it. The CLI should be wiring, not a model runtime. |
| `sparkflash-device` | Arena/session/staging are generic; `chain`, `layout_contract`, `converted`, `resident`, and `shard` carry V4.1 geometry and tensor names. | The generic device runtime is entangled with one model's layout. |
| `sparkflash-kernels` | About 107k lines and 46 CUDA files, all pinned to V4.1 shapes and ABI. | This is expected; the problem is that there is no catalog/manifest boundary that says which kernels belong to which model/backend. |
| `sparkflash-web` | `SUPPORTED_WEIGHTS` and `SUPPORTED_ENGINES` are one-element constants. | The control plane cannot enumerate what the installed binary actually supports. |
| Config/selectors | `model.weights` and `engine.kind` currently use the same string, `deepseek-v4.1-flash`. | "Which weights" and "how they run" are conceptually different and must stop sharing a value. |

There are also several independent sources of truth for the same geometry: `ModelConfig`,
`sparkflash_kv::config`, `sparkflash_device::layout_contract`, `sparkflash_cli::tp4_forward`,
and the CUDA headers. The compile-time assertions in `layout_contract.rs` are a good pattern;
the goal is to make the model plan the source and the kernels assert against it, rather than five
places agreeing by hand.

## 3. Design principles

1. **A model is a pack, not a flag.** A model pack owns its config parsing, tensor naming and
   shapes, layer semantics, prompt codec, reference forward, converter, and kernel bindings.
2. **Static specialization, dynamic dispatch only at a boundary.** The hot loop stays generic-free
   and monomorphized or directly dispatched. Trait objects are for model/runtime and
   backend selection, not per-token arithmetic.
3. **One model per process.** The cluster already serves one checkpoint; per-process selection is
   simpler, safer for KV and collectives, and enough for the control plane to launch different
   deployments.
4. **Move, do not rewrite, the V4.1 path.** Extraction should be `git mv` plus re-exports and
   adapter traits. Arithmetic, ABI, and layout bytes stay unchanged.
5. **Optional features are capabilities.** Engram, DSpark, vision, and speculation are optional
   model capabilities, not fields every model must implement with zeroed values.
6. **Artifacts carry identity and version.** A converted weight file must say which model family,
   config digest, layout, TP world, and ABI it was built for, and refuse a mismatch before load.
7. **Prove the abstraction with a second model before polishing it.** A small dense model on the
   host path is the cheapest way to find leaks.

## 4. Target architecture

The target is a layered set of crates. Not every crate needs to exist on day one; the phases below
introduce them in dependency order.

```
HTTP / OpenAI / Anthropic
        │
        ▼
sparkflash-api            protocol, router, SSE, wire types only
        │
        ▼
sparkflash-serve          request lifecycle, drivers, AppState wiring
        │
        ▼
sparkflash-runtime        scheduler/KV/batching coordination (model-agnostic)
        │
        ├──────────────┬─────────────────────────────┐
        ▼              ▼                             ▼
sparkflash-model   sparkflash-model-v41      sparkflash-model-<next>
(contracts)        (V4.1 definition)         (second family)
        │              │
        ▼              ▼
sparkflash-backend-host / sparkflash-backend-cuda
        │
        ▼
sparkflash-kernels (ABI, catalog, loader) + sparkflash-kernels-v41 (current symbols/CUDA)
        │
        ▼
sparkflash-device (arena, staging, session, paged KV) + sparkflash-device-v41 (chain/layout/converted)
        │
        ▼
sparkflash-comm / sparkflash-trace / hardware
```

### 4.1 `sparkflash-model`: the contracts crate

It should contain almost no arithmetic. Candidate contents:

```rust
pub struct ModelDescriptor {
    pub id: String,                 // served model id, e.g. "deepseek-ai/DeepSeek-V4.1-Flash"
    pub family: String,             // "deepseek-v4.1"
    pub weights_id: String,         // control-plane selector
    pub context_tokens: usize,
    pub max_output_tokens: usize,
    pub vocab_size: usize,
    pub tokenizer: TokenizerSpec,   // path/digest/special tokens
    pub capabilities: Capabilities, // vision, tools, thinking, engram, speculation, ...
    pub artifact: ArtifactFormat,   // format version, supported TP worlds, layout id
}

pub trait ModelDefinition: Send + Sync {
    type State: SequenceState;
    fn descriptor(&self) -> &ModelDescriptor;
    fn plan(&self) -> &ModelPlan;                       // tensor + layer + KV geometry
    fn validate_source(&self, weights: &dyn WeightSource) -> Conformance;
    fn new_state(&self) -> Self::State;
    fn prefill(&self, state: &mut Self::State, tokens: &[u32]) -> Result<StepOutput, ModelError>;
    fn decode(&self, state: &mut Self::State, token: u32) -> Result<StepOutput, ModelError>;
    fn prompt_codec(&self) -> &dyn PromptCodec;
}

pub trait PromptCodec: Send + Sync {
    fn encode_chat(&self, request: &ChatInput) -> Result<Vec<u32>, PromptError>;
    fn encode_text(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, PromptError>;
    fn decode_tokens(&self, ids: &[u32]) -> Result<String, PromptError>;
    fn token_str(&self, id: u32) -> Option<String>;
}
```

The exact names are not important; the boundaries are:

- **Model identity** is data. The server asks the descriptor for id, context, max output, vocab,
  and capabilities rather than importing constants.
- **The prompt codec** owns the tokenizer and chat template. `sparkflash-api` should hold
  `Arc<dyn PromptCodec>`, not a concrete `TokenizerSeam`.
- **The plan** (tensor specs, layer specs, KV geometry, required artifacts) is one object owned by
  the model pack.
- **Capabilities and telemetry extensions** are optional data. DSpark/Engram fields can remain in
  the V4.1 wire response, populated only when the model/backend declares them; a generic
  `extensions` map or typed optional section keeps other models from inventing fake zeros.

### 4.2 Model packs

`sparkflash-model-deepseek-v41` is the current model, moved behind the contracts:

- `config`: the current `ModelConfig`, released-HF mapping, validation.
- `tensor`: the current `TensorPlan` and byte-total assertions.
- `layer_plan`: the current `LayerPlan`, modes, sources, DAG validation.
- `forward`: the current CPU reference (attention, mHC, MoE, Engram, MTP, batch, boundary).
- `encoding`: the current V4.1 renderer, tokenizer seam, tool/reasoning parsing, vision prompt
  handling. This moves out of `sparkflash-api`.
- `converter`: the V4.1 tiling rules (B12x, MoE-640 padding, scale adjacency).
- `kernels`: the V4.1 kernel catalog and ABI constants, while the actual CUDA sources can stay in
  a sibling `sparkflash-kernels-v41` crate.

A second pack, for example `sparkflash-model-llama`, implements the same traits with GQA,
RoPE, SwiGLU, a dense FFN, its own tokenizer/template, and no Engram/DSpark/vision. It should
start as a host-only pack.

### 4.3 Backends

Keep the host reference and device execution as separate backends that consume a model pack:

```rust
pub trait ModelBackend: Send + Sync {
    type Session;
    fn open(&self, model: &dyn ModelDefinition, artifact: Option<&ArtifactManifest>)
        -> Result<Self::Session, BackendError>;
    fn step(&self, session: &mut Self::Session, input: StepInput)
        -> Result<StepOutput, BackendError>;
}
```

- `backend-host`: the current reference engine, no GPU, useful for correctness and for small
  models.
- `backend-cuda`: the current TP=4 device path. Its V4.1 implementation moves out of
  `sparkflash-cli`.
- The backend declares which artifact layout and kernel ABI it requires; a model pack declares
  which backends it supports. The server refuses an unsupported pair at startup, not at the first
  token.

### 4.4 Artifact and converter boundary

Replace the implicit "the converter knows V4.1 names" with:

```rust
pub trait ModelConverter {
    fn inspect(&self, checkpoint: &Path) -> Result<CheckpointInfo, ConvertError>;
    fn convert_rank(&self, opts: &ConvertOptions) -> Result<ConvertedRank, ConvertError>;
    fn validate(&self, artifact: &ArtifactManifest) -> Result<(), ConvertError>;
}
```

The generic converted-file container keeps the preamble, alignment, tensor index, and digests.
The model pack supplies the tensor naming, sharding rule, and tiling. Add to the manifest:

- `model_family`, `model_id`
- `config_sha256`
- `format_version`
- `layout_id` / `tp_world_size`
- `kernel_abi_version`

The loader must reject a family/config/ABI mismatch before mapping weights. Existing V4.1 files
remain loadable by keeping their format version and adding the new fields with defaults.

### 4.5 Kernel catalog boundary

Do not rewrite the kernels. Put a manifest in front of them:

```rust
pub struct KernelSpec {
    pub id: KernelId,
    pub cubin: &'static str,
    pub symbol: &'static str,
    pub abi_version: u32,
    pub min_arch: &'static str,
}
```

The generic loader verifies the cubin set, ABI, and architecture from the catalog. The V4.1 pack
owns the current `REQUIRED_KERNELS`, frozen signatures, and layout assertions. A new model adds
symbols to its own catalog; it does not edit the V4.1 list. This is also the natural place to make
the CUDA shape constants assert against the model plan instead of duplicating literals.

## 5. Migration path

The order below is chosen so that every phase can land independently and be reverted.

### Phase 0 — Baseline and guardrails (2-3 days)

- Fast-forward local `main` to `origin/main`; record the exact commit.
- Record a green baseline: `cargo test --workspace`, the Python tests, and the host-only device
  fixtures. Save the current V4.1 golden tokens/logits, converted-file hashes, and HTTP byte
  fixtures.
- Add a decision record to `docs/DECISIONS.md`: model packs, one model per process, artifact
  identity, "move don't rewrite".
- Add a CI matrix entry for a model-feature build, even if only V4.1 exists.

**Acceptance:** no code change; baseline artifacts are reproducible.

### Phase 1 — Model identity and registry (1-2 weeks)

- Introduce `sparkflash-model` with `ModelDescriptor`, `ModelRegistry`, and `PromptCodec`.
- Remove `MODEL_ID`, `MODEL_CONTEXT_TOKENS`, and `MODEL_MAX_OUTPUT_TOKENS` as global defaults
  from the HTTP path. The engine/driver supplies them.
- Make `sparkflash-web` derive `SUPPORTED_WEIGHTS` and `SUPPORTED_ENGINES` from the registry.
  Separate the selectors: `model.weights = "deepseek-v4.1-flash"`,
  `engine.kind = "host-reference" | "tp4-device"`.
- Add `sparkflash models` (or `sparkflash model list`) to print ids, context, capabilities, and
  supported backends.
- Keep V4.1 the only registered model; keep the old CLI defaults.

**Acceptance:** HTTP bytes for existing tests are unchanged; `/v1/models` is descriptor-driven;
an unknown model is refused at startup with a list of supported ids; the web form round-trips.

**Risk:** low. This is mostly plumbing and constants.

### Phase 2 — Extract the V4.1 host model pack (3-5 weeks)

- Create `sparkflash-weights` for the generic container and format code:
  `dtype`, `quant`, `linalg`, `safetensors`, `loader`, `weights`.
- Create `sparkflash-model-deepseek-v41` and move:
  - `engine/config.rs`, `engine/tensor.rs`, `engine/layer_plan.rs`
  - `engine/forward/*`, `engine/sequence_fixture.rs`, `engine/prompt_ranks.rs`
  - `api/encoding/*`, the V4.1 parts of the vision prompt path
  - `kv/config.rs`, `kv/layout.rs`, the V4.1 parts of `kv/swa.rs`
  - `scheduler/dspark.rs` (as a model policy) or a policy trait it implements
  - `engram` stays a crate, but only the V4.1 pack depends on it
- Keep `sparkflash-engine` as a facade re-exporting the old paths for one release, so downstream
  code and tests migrate gradually. Delete the facade when no crate uses it.
- Make `sparkflash-api` depend only on `sparkflash-model` contracts; it must no longer depend on
  `sparkflash-engram`, `sparkflash-kv`, or `deepseek-recipe`.
- Add a **toy dense model** definition in `sparkflash-model` tests to prove the contracts are not
  secretly V4.1-shaped. It needs no real weights.

**Acceptance:** the V4.1 golden sequence fixture, tensor-plan conformance, layer-plan table, HTTP
router tests, and tokenizer goldens are byte-identical before and after. `cargo tree -p
sparkflash-api` no longer contains model/feature crates. The toy dense model runs through the
generic host path.

**Risk:** medium, mostly mechanical. The test suite is the safety net. Do not change arithmetic in
this phase.

### Phase 3 — Make serving generic (1-2 weeks)

- Move the HTTP driver/runtime wiring out of `sparkflash-cli/src/serve.rs` into a
  `sparkflash-serve` library. The CLI keeps flags and dispatch. **Started:** the library and
  generic host adapter exist; the V4.1 reference driver and CLI wiring still need to move onto
  it.
- Implement `InferenceEngine` as a generic adapter over `ModelDefinition` + `ModelBackend`.
  **Host path done:** `sparkflash-serve::HostDriver` drives `ModelBackend`/`ModelSession`.
- Replace `AppState`'s concrete `TokenizerSeam` with `Arc<dyn PromptCodec>`; route
  `/tokenize`, `/detokenize`, chat rendering, and tool/reasoning parsing through it. **Done.**
- Drive context length, max output, vocab, and capabilities from `ModelDescriptor`. **Done for
  the descriptor fields; model-specific telemetry remains V4.1-shaped.**
- Keep model-specific telemetry as an optional extension; V4.1 still emits the same fields.

**Acceptance:** all existing `serve_http` tests pass unchanged; a synthetic toy model serves a
chat completion through the same router on the host path.

**Risk:** medium. The router is large (about 4.4k lines) and has byte-level tests, which is good.

### Phase 4 — Artifact and converter contracts (1-2 weeks)

- Add the manifest fields and versioning described in 4.4.
- Introduce `ModelConverter`; make `sparkflash convert` choose the converter from the detected
  model family, defaulting to V4.1.
- Move B12x and MoE-640 tiling into the V4.1 pack.
- Add a startup check: artifact family/config digest must match the loaded model definition.
- Keep old files readable.

**Acceptance:** converting the same checkpoint produces byte-identical rank files; the new
manifest round-trips; a mismatched family is refused before any large allocation; `inspect` and
`verify` report the model identity.

**Risk:** medium-low. Header compatibility needs care, but the existing `header_hash` and
`file_digest` give a strong baseline.

### Phase 5 — Extract the device path (4-8 weeks)

This is the largest phase. Do it in sub-steps:

1. **Create `sparkflash-backend-cuda`** for the generic device runtime: arena, budget, session,
   staging, paged KV, window, cubin loading, trace.
2. **Create `sparkflash-backend-deepseek-v41`** and move the V4.1 device code out of the CLI:
   `tp4_forward` and its submodules, `tp4_host`, `tp4_chain`, `tp4_combine`,
   `tp4_device_runner`, `tp4_prefill`, `tp4_reduce`, `tp4_resident`, `tp4_tail`,
   `layer_device`, `attention_device`, `moe_device`, `layer_attention_stages`,
   `engram_device_state`, `resident_weights`, `device_chain`.
3. **Move the V4.1 layout/chain/artifact code** out of `sparkflash-device`:
   `chain`, `chain_device`, `layout_contract`, `converted`, `resident`, `shard`,
   `layer_zero`, and the V4.1 parts of `paged_kv`/`paged_read`.
4. **Introduce the kernel catalog** and move the current V4.1 symbols/ABI declarations into
   `sparkflash-kernels-v41`. Keep `sparkflash-kernels` as the loader/ABI/catalog crate.
5. **Shrink `sparkflash-cli`** to `main.rs`, flag parsing, and registry dispatch. Target under
   5k lines.

Do each sub-step as a move-only commit, with the old module re-exported. Run the host device
fixtures after every move; run a GPU window for performance and parity before deleting each
re-export.

**Acceptance:** `sparkflash serve` and `device-chain` behave identically; converted files load;
required-kernel pins and ABI hashes are unchanged; the V4.1 device golden fixture passes; a GPU
window reproduces the same tokens and step timing within noise.

**Risk:** high, because the device path has many implicit couplings. The mitigation is strictly
move-only commits, re-exports, and the existing layout compile-time assertions.

### Phase 6 — Add the second real model, host first (2-4 weeks)

Pick a small dense decoder (for example a 0.5-1B Llama/Qwen-class model). It forces the generic
contracts to handle:

- GQA instead of MLA
- dense SwiGLU FFN instead of routed MoE
- standard RoPE instead of YaRN/compressed KV
- a different tokenizer and chat template
- no Engram, no DSpark, no mHC, no vision initially

Implement the model pack, the converter for its safetensors layout, and the host backend. Serve it
on one node. Only after that works should the device backend be considered. A second model on the
device path is mostly new kernel work, not new abstraction work; budget it separately (4-8 weeks
depending on the architecture).

**Acceptance:** `sparkflash models` lists two models; `sparkflash serve --model <id> --engine
host-reference` serves the second model; V4.1 remains unchanged and default; the HTTP contract
tests run against both.

### Phase 7 — Operational polish (optional, 2-3 weeks)

- Web control plane: model profiles, per-profile checkpoint/converted paths, launch/convert jobs
  selected by descriptor.
- Artifact registry and preflight (`sparkflash doctor --model`).
- CI matrix: V4.1 host, V4.1 device, second model host; feature combinations.
- Documentation split by family under `docs/models/<family>.md`.

## 6. Suggested first pull requests

| # | Change | Size | Risk |
| :-- | :-- | :-- | :-- |
| 1 | This plan + a decision row in `DECISIONS.md` | S | none |
| 2 | `sparkflash-model` descriptor/registry; API and web use it | M | low |
| 3 | Split `model.weights` from `engine.kind`; add `sparkflash models` | S-M | low |
| 4 | `sparkflash-weights` extraction (container/quant/loader) | M | low-medium |
| 5 | Move V4.1 config/tensor/layer plan behind the model pack with re-exports | L | medium |
| 6 | Move V4.1 reference forward + encoding out of `sparkflash-engine`/`sparkflash-api` | L | medium |
| 7 | Artifact manifest + converter trait | M | medium |
| 8 | Toy dense model in tests through the generic host path | M | medium |
| 9 | Move TP=4 device code out of `sparkflash-cli`, move-only | XL | high |
| 10 | Kernel catalog + V4.1 kernel pack | L | high |
| 11 | Second real dense model, host path | L | medium |

PR 9 should be split further (one module family per PR) once the target crate skeleton exists.
Avoid a single "big bang" move of the CLI.

## 7. Risks and mitigations

| Risk | Mitigation |
| :-- | :-- |
| The abstraction leaks through kernel ABIs and layout constants. | Let the model pack own kernels and layout; the generic backend only loads and dispatches. Keep compile-time assertions from plan to kernel defines. |
| Moving 90k lines of CLI code breaks the device path subtly. | Move-only commits; keep re-exports; run host fixtures after each move and a GPU window before deleting the old path. |
| Golden/perf regressions from "cleanup". | Freeze tokens/logits, converted hashes, HTTP bytes, and step timing before Phase 2. No arithmetic edits during extraction. |
| Five copies of the same geometry drift further. | Make `ModelPlan` the source; generate or assert `sparkflash_kv`, device layout, and kernel defines against it. |
| The generic engine becomes slower because of dynamic dispatch. | Dispatch once per request/step at the model boundary, not per tensor or per kernel. Keep the V4.1 hot loop concrete. |
| The second model is chosen too large or too different. | Start with a small dense model on the host path. Treat device support as a separate project. |
| Multi-model serving is assumed too early. | One model per process. The control plane can launch different profiles; it does not need a shared process. |
| Tokenizer/template diversity is underestimated. | `PromptCodec` owns rendering, tokenization, detokenization, tool/reasoning parsing, and special tokens. Test it with golden ids per model. |
| Weight formats differ more than expected. | Artifact manifest + per-model converter; do not force all models through the V4.1 tiling. |

## 8. Decisions to make before Phase 1

1. **Scope of "other models":** same-family checkpoints only, or arbitrary architectures? The
   plan above targets arbitrary families but sequences the work so same-family support arrives
   first.
2. **Hardware scope:** still 4x DGX Spark only? If other GPUs are in scope, the backend/kernel
   boundary needs to be designed earlier.
3. **Concurrency:** confirm one model per process for the foreseeable future.
4. **Second model:** choose a small dense decoder for the pilot, not another 500B MoE.
5. **Packaging:** compile-time feature-selected model packs (recommended initially) vs dynamically
   loaded model plugins. Compile-time keeps ABI checking simple and avoids a plugin SDK before the
   contracts are stable.

## 9. Definition of done

- V4.1-Flash remains the default and passes the existing correctness, HTTP, and device gates
  unchanged.
- A model is selected by descriptor, not by recompiling constants into `sparkflash-api` or
  `sparkflash-web`.
- Adding a model pack does not require editing the generic scheduler, HTTP router, or CLI
  dispatcher.
- Artifacts identify their model family, config, layout, and ABI, and mismatches are refused
  before load.
- A second model runs end-to-end on the host path, and the cost of adding device support is
  measurable in kernels and a backend, not in a fork of the serving stack.