# 02 — The execution plan and the kernel resolver

This page describes the two objects that sit between a released `config.json` and a model
backend. They are the contract a new arm implements; they are not a new runtime.

## The flow

```text
config.json
  -> sparkflash_model::Architecture        (the released document, typed)
  -> sparkflash_model::TpPlan              (which axes rank r holds)
  -> sparkflash_ops::EnginePlan            (layers, parts, rank-local shapes, precision)
  -> sparkflash_ops::KernelResolver        (each part -> one loadable image + symbols)
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
experts, top_k, positions, vocab, ratio, streams, ...). A zero field is "not applicable". The
fields are the ones a kernel's shape rule checks; a new part adds a field or leaves it zero.

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
(Part, Target, PartGeometry, Precision) -> loadable PartKernel
```

It filters `sparkflash_ops::parts::CATALOGUE` by target, availability policy, the
`ShapeConstraint` on each entry, and the precision mask. It prefers the canonical entry and
returns every candidate's refusal reason when none match. `resolve_all` reports every missing
part, not just the first.

Availability is explicit, because "the catalogue names a symbol" and "this process can load it"
are different facts:

| policy | accepts |
| :-- | :-- |
| `Catalogue` | every entry (planning and CPU tests) |
| `Embedded` | only images this binary embedded |
| `Runtime` | embedded images plus the DeepSeek cubin set and RoCEnante |

`ShapeConstraint` is intentionally conservative. An entry with no proven rule is
`ShapeConstraint::ANY`; an entry that only works on whole 16-wide heads says
`head_dim_multiple(16)`; a GQA entry accepts every group size because the pack has a
generic `q4e_attn` body beside its four- and six-head specialised kernels. A kernel family that
truly lacks a group size would add `gqa_groups(...)` here.
Tightening one entry is local and does not touch the walk.

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
buffers; the resolver only says which image and symbols can serve a requirement. That is what
lets a new kernel implementation land without changing the architecture walk, and lets a new
architecture use an existing kernel without a new arm.

## How to adopt it

1. Build the plan in the arm's boot path: `EnginePlan::from_json(&config, target, world, rank)`.
2. Resolve it with the narrowest availability that is honest for the build
   (`Runtime` for a run with a cubin directory, `Embedded` for an embedded image build).
3. Refuse on `KernelResolver::failures` before opening weights.
4. Pass the plan to the backend; read shapes and splits from `plan`, not from a second parse.
5. Add the arm's `PartKernel` shape rules as they are proven; leave the rest `ANY`.

Steps 1-3 are on the boot path for every arm today: `serve/pack.rs::check_plan` for Qwen and
GLM, and `serve.rs::check_checkpoint_plan` for the DeepSeek reference, host-TP4 and device
paths. The DeepSeek profile is loaded from its carried TOML in the same startup. Step 4 is the
remaining integration: the Qwen and GLM backends still parse their own typed configuration
before opening, and the DeepSeek device arm still builds its own shard plan. The executable
contract is `sparkflash_ops/tests/engine_plan.rs`: all three released configs plan and resolve
at TP=1, TP=2 and TP=4.
