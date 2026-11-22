# Layout files

A layout states the wiring a checkpoint's `config.json` does not: block order, per-part
facts, tensor name templates and the config keys the description reads. The reader is
`sparkflash-model::layout`. A layout says nothing about kernels, tensor-parallel splits or a
shape the config already gives.

## Where a file is read from

* `models/layouts/*.toml` is embedded at build time.
* `~/.cache/sparkflash/layouts/*.toml` is read on this node.

Two files that match one checkpoint are a refusal naming both. A cache file may replace a
shipped layout only by naming its `id`:

```toml
replaces = "glm5-next-fp8"
```

## Sections

```toml
schema = 1
id = "stable-id"
replaces = "shipped-id"          # only in a cache file

[match]
architectures = ["ProducerClass"] # the config's root architectures entries
model_types = ["producer_type"]   # the config's root model_type
```

`match` is the only place in the tree that compares a family string.

```toml
[[keys]]
path = "text_config.qk_head_dim"  # full path from the config root
fact = "language.attention.head_dim"

[[keys]]
path = "producer.metadata"
ignore = "the tensor index states every stored dtype"
```

Every key is either read into a fact or listed ignorable with a written reason. A key that
is absent from the checkpoint is skipped; it is not a fact.

```toml
[stacks.vision]
order = ["vision_patch_embed", "vision_position", "vision_norm"]

[[stacks.vision.parts]]
part = "vision_position"
[[stacks.vision.parts.facts]]
name = "position"
value = "rotary"                 # a literal

[[stacks.vision.parts.facts]]
name = "norm_epsilon"
value = { key = "vision_config.rms_norm_eps", default = 1e-6 } # a config key
```

A stack's `order` is the part vocabulary, not a claim that every block has every part. A fact
the reader does not apply yet is still carried in the description; its scan row stays `unknown
layout` until the reader learns the field, so a source can be recorded without guessing.

```toml
[[tensor]]
name = "model.language_model.layers.{layer}.self_attn.q_proj.weight"
part = "kda_scan"                # a Part name
role = "attention"               # the ops tensor role
dtype = "bf16"                   # bf16, f32, dense or expert
shape = ["kda_width", "hidden"]  # arithmetic over facts, never a repeated literal
split = "rows"                   # whole, rows, columns, kv_rows, vector, conv,
                                 # expert_rows or expert_columns
repeat = "layers"                # once, layers, experts, mtp, mtp_experts or vision_blocks
when = { attention = "kda" }     # attention kind, ffn kind and/or indexer ownership
unloaded = "the producer keeps a duplicate for its converter"
```

`name` may use `{layer}`, `{block}` and `{expert}`. `dtype = "dense"` takes the checkpoint's
dense format and adds its scale tensor; `dtype = "expert"` does the same for a routed expert.

## Sourcing rule

Each wiring fact and key carries a comment naming where it came from: the producer's
reference source and line, a config path, or the tensor evidence. A fact you cannot source is
left out; the scan keeps that row `unknown layout` rather than guessing.

## Proving a file

The layout's checkpoint plan must equal the plan the checkpoint's existing description builds,
name for name and shape for shape, at TP 1, 2 and 4. Use the helper in
`crates/sparkflash-ops/tests/layout_equality.rs`; the worked file's own oracle is
`crates/sparkflash-ops/src/tensor_plan.rs` under `cfg(test)`.
