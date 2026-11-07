# 02 — The execution plan and the kernel resolver

This page describes the two objects that sit between a released `config.json` and a model
backend. They are the contract a new arm implements; they are not a new runtime.

## The flow

```text
config.json
  -> sparkflash_model::Architecture        (the released document, typed)
  -> sparkflash_model::TpPlan              (which axes rank r holds)
  -> sparkflash_ops::EnginePlan            (layers, parts, rank-local shapes, precision)
  -> sparkflash_ops::KernelResolver        (each part -> one image + one symbol)
  -> EngineBuilder::open                   (the arm's backend)
```

`sparkflash_ops::EnginePlan::from_json` is the short form a boot path uses. It parses the
architecture, validates the tensor-parallel world, reads the precision policy from the released
quantization block, and walks every layer through the shared `Architecture::steps()`. The
result is one `PartRequirement` per part instance:

```rust
pub struct PartRequirement {
    pub part: Part,
    pub geometry: PartGeometry,   // rank-local, after TpPlan
    pub precision: Precision,
}
```

`PartGeometry` is a flat bag of dimensions (hidden, heads, kv_heads, head_dim, intermediate,
experts, top_k, positions, vocab, ratio, streams, ...). A field an entry's rule names is a proof
obligation: a geometry that leaves it at zero is refused, not skipped. The execution plan fills
the fields its parts need; a typed plan built from the requirement inherits them instead of
inventing a second shape.

## The tensor-parallel plan

`TpPlan` is deliberately small: `world`, `rank`, and `split(total, axis, alignment)`.
The split rules are:

- `Replicated` and `Mtp` axes are held whole on every rank;
- `Heads`, `Experts`, `Intermediate`, `Vocab` and `Hidden` must divide evenly;
- `KvHeads` may be replicated when there are fewer heads than ranks (the Qwen/GLM rule);
- `padded` is the kernel-visible per-rank width and `padding` is the zero fill. The logical
  `total` is never changed.

`TpPlan::for_architecture` validates every layer's heads, key/value heads and feed-forward
intermediate before a weight is read. A world that does not divide the model is a boot refusal,
not a first-request failure.

## The resolver

`KernelResolver` answers one question:

```text
(Part, Target, PartGeometry, Precision, KernelPhase) -> (image, KernelSymbol)
```

It filters `sparkflash_ops::parts::CATALOGUE` by target, availability policy, the entry's
`ShapeConstraint`, and then by the symbol itself: the chosen symbol's own `PrecisionMask`
must contain the requested precision and its own `KernelPhase` must serve the requested pass.
It scores the canonical entry ahead of the others and returns every candidate's refusal reason
when none match. `resolve_all` reports every missing part, not just the first.

`resolve_symbol` takes a priority list of names. The first requested symbol that some entry
exports with a matching precision and pass wins, so a plan that needs a specific arithmetic body
names it and the resolver does not merely accept any symbol in the image.

`boot_check` resolves every requirement instance once for prefill and once for decode. A
model with separate bodies proves both exist; a model with one body proves it serves both.

`ResolvedKernel` carries `part`, `target`, `image`, `symbol`, `precision` and
`geometry`. Its `id()` is the `KernelId` a launch names, and `KernelId` includes the
target:

```rust
pub struct KernelId {
    pub target: Target,
    pub image: Image,
    pub symbol: &'static str,
}
```

A phase-4 `EngineBuilder` can therefore prove that the kernel it is about to load was resolved
for the device it opened.

## Availability

"the catalogue names a symbol" and "this process can load it" are different facts:

| policy | accepts |
| :-- | :-- |
| `Catalogue` | every entry (planning and CPU tests) |
| `Embedded` | only images this binary embedded |
| `Runtime { cubin_dir }` | embedded images, the RoCEnante runtime image, and the DeepSeek PTX set when `SPARKFLASH_CUBINS` names a directory |

`Availability::loadable` and `parts::loadable` call the same `parts::image_loadable`
predicate, so they cannot disagree. `Availability::runtime_env()` reads the cubin-directory
variable for a serving process.

## Typed plans

Every typed plan (`AttentionPlan`, `DenseGemmPlan`, `ScanPlan`, `NormPlan`,
`RopePlan`, `ActivationPlan`, `QuantizePlan`, `GatherPlan`, `GroupedMoeGemmPlan`,
`MoeRoutingPlan`, `ExpertCombinePlan` and `CollectivePlan`) is built from a
`PartRequirement` and a `PlanContext`:

```rust
let context = PlanContext::SM121_CATALOGUE;
let plan = AttentionPlan::new(context, &requirement, kind, rows, context_len, head_dim, stream)?;
```

The requirement is the one canonical geometry source; the plan overrides only the fields the
schedule owns (rows, context, head width) and derives the pass from the row count
(`KernelPhase::for_rows`: one row or fewer is decode, wider is prefill). `PlanContext`
carries the target and availability, and the `KernelId` the plan stores includes the target.
Constructors validate their own shape before they divide or resolve, so a zero head count, an
NVFP4 attention request or a top-k past the expert count is a shape/unsupported refusal, not a
resolution failure.

A plan never names an image directly. The resolver picks it, and the plan keeps the resulting
`KernelId`.

## The backend seam

An arm implements `EngineBuilder`:

```rust
impl EngineBuilder for MyArm {
    type Backend = MyBackend;
    fn open(plan: &EnginePlan, weights: &dyn WeightSource) -> Result<MyBackend, PlanError> {
        // plan.target(), plan.tp(), plan.layers(), plan.globals()
    }
}
```

The plan owns no allocation and no kernel choice. The backend keeps ownership of its device
buffers; the resolver only says which image and symbol can serve a requirement. That is what
lets a new kernel implementation land without changing the architecture walk, and lets a new
architecture use an existing kernel without a new arm.

## How to adopt it

1. Build the plan in the arm's boot path: `EnginePlan::from_json(&config, target, world, rank)`.
2. Resolve it with the narrowest availability that is honest for the build
   (`Availability::runtime_env()` for a run with a cubin directory, `Embedded` for an
   embedded image build).
3. Refuse on `KernelResolver::failures` before opening weights. Use
   `failures_for(plan, phase)` or `boot_check` when both passes must be proven.
4. Pass the plan to the backend; read shapes and splits from `plan`, not from a second parse.
5. Add the arm's `PartKernel` shape rules as they are proven; leave the rest
   `ShapeConstraint::ANY`.

Steps 1-3 are on the boot path for every arm that launches kernels: `serve/pack.rs::check_plan`
for Qwen and GLM, and `serve.rs::check_checkpoint_plan` for the DeepSeek host-TP4 and device
paths. The DeepSeek reference engine runs the host twin and loads no kernel, so it is not gated
by the catalogue's rank-local shape rules: a TP1 reference serve of the released TP4-shaped
checkpoint is legal. The DeepSeek profile is loaded from its carried TOML in the same startup.

Step 4 is on the single-rank serving path: Qwen and GLM `open_host` call their arm's
`EngineBuilder::open` (or `EngineBuilder::open_converted` for a Qwen converted rank file),
which gates on `plan.target()`/`plan.tp()`, runs the phase-aware serving boot check for both
passes through `PartsDeviceRank::open`, and only then opens the arm. The single-rank builder
refuses a plan at another target or at TP>1; the TP4 group keeps its own per-rank
`PackArm::OPEN_RANK` path. The DeepSeek device arm still builds its own shard plan.

## What the released configs resolve at

The executable contract is `sparkflash_ops/tests/engine_plan.rs`:

| arm | worlds | notes |
| :-- | :-- | :-- |
| Qwen3.8 | TP1, TP2, TP4 | the language parts resolve; the 16x16 vision document is refused because the shared tower is 14x14. That is a deliberate product decision until phase 2's templated vision tower lands, not a resolver gap. Its FP8 pack stores only the routed experts as FP8: the arm's architecture bridge corrects the document's global `fp8` flag, so the plan is BF16 dense/attention with FP8 experts |
| GLM-5.3 | TP1, TP2, TP4 | the FP8 release; prefill and decode bodies both present. A BF16 document is an explicit boot-check refusal in both passes: the tree has no BF16 routed-expert body, and an FP8 weight body (including `glm_moe_fp8b128_mma_bf16a`, whose `_bf16a` names its activations) is not a BF16 substitute |
| DeepSeek V4.1 | **TP4 only** on the kernel path | the b12x MXFP4 experts are compiled for 576 rank-local columns; TP1 and TP2 are refused with that width named. The reference engine launches no kernel and is not gated |

A refusal names the part, image, precision and pass, and every candidate's reason, before a
weight is read.
