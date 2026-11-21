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

A new family is its released `config.json`, the arithmetic of the parts no other model has, and
one registration. The walk, the sequence and its lane memory, the rewind point, the sampler, the
head, lanes, prompt chunks, prefix reuse and the tensor-parallel group are the runtime's
(`sparkflash_runtime::PlanWalk` on the shared dense host); a family writes none of them. The
kernel stack, the typed ops, the serving contract and the shared homes are described in
[arm-reference.md](arm-reference.md); a family reads it only for a part of its own.

## 1. Run `onboard` first

```sh
sparkflash onboard ORG/REPO [--config FILE] [--emit DIR]
```

Before any code is written it prints:

- the one `ScanReport` per target and TP: every part the stack names, whether a kernel resolves
  for both passes, and every unread config key or unclaimed tensor with its fix;
- which matrix-product parts have only a row body or no tensor-core body for a pass, and whether
  a `WAIVERS` line covers the gap, so a `PartExec` or a waiver is the next edit;
- the stored format of each weight class, read from `quantization_config` at the top level or
  under `text_config`;
- the config switches the parser does not model yet (`Architecture::unmodelled_features`), which
  must be taught to `Architecture` before a part is written for them;
- the fit at TP 1, 2 and 4, and the stored bytes-per-token floor.

`--config FILE` reads the document from disk; otherwise it uses the hub cache and only then the
Hub. The tensor check reads safetensors headers (local, or ranged Hub reads) without weights; when
no header is available the report says `tensors not checked` and is not clean. `--emit DIR` adds
the last step:

- a family whose every part the dense host registers (`Dense::catalogue()`) needs no crate: it
  prints the `sparkflash serve --checkpoint DIR` command and succeeds;
- any other family gets a crate skeleton in `DIR`: `Cargo.toml`, the config under `config/`,
  `architecture()`, `parts()` registering one `PartExec` stub per part of its own (each body
  refuses by name), and `tests/conformance.rs` whose fixture refuses by name until it is written.
  It then prints the edits outside the crate and what the shared host cannot run yet.

It writes nothing outside `DIR` and refuses a `DIR` that exists.

## 2. What you supply

Only what this family alone has:

1. **Its config mapping.** `sparkflash_model::Architecture` reads `layer_types`,
   `mlp_layer_types`, `first_k_dense_replace`, `indexer_types`, `hidden_act`, `scoring_func`,
   `compress_ratios`, `hc_mult`, `engram_layer_ids` and the rest into typed parts; a field it does
   not model is named by `onboard`. Teach the parser the field, not a module of the family.
2. **The arithmetic of each part of its own.** `impl PartExec<Dense> for ItsPart` with four
   methods and no defaults: `state` (the lane regions it adds, positional or recurrent), `split`
   (`RankSplit::Summed` when a rank leaves a partial the group sums, `Replicated` otherwise),
   `rows` (the prompt body) and `token` (the token body; `self.rows(pass)` when they are the
   same). A body reads the normed rows at `pass.input()` and leaves the branch's rows at
   `pass.output()`.
3. **Its registration.** One line on the catalogue registry:

   ```rust
   let mut parts = Dense::catalogue();
   parts.replace(Part::DenseMlp, |open: &mut OpenCx<'_>| ItsPart::open(open))?;
   ```

   `register` adds a kind the registry does not hold and refuses one it does; `replace` takes a
   kind over. The constructor uploads the rank's share through `OpenCx` (`source`, `plan`, `keep`,
   `record`, `pack`, `scratch`, and the catalogue constructors, which a part may wrap).
4. **Its kernel**, if the part needs one: a source under `kernels/`, loaded through
   `open.cuda()` and launched with `pass.launch` on the walk's stream.
5. **A tiny fixture and a stated profile** for `sparkflash_runtime::conformance!`: a written
   checkpoint at the config's shapes and the prompt and token limits the walk takes.
6. **Weight names, tokenizer and chat template**, only where they leave the shared spelling
   (`sparkflash_ops::naming`) and the `tokenizer.json` markers.

## 3. The worked examples

- `crates/sparkflash-model-dense/tests/onboard_example.rs` is a whole family in one file (325
  lines): a made-up sandwich-normed decoder's config, its one `PartExec`, its registration, a
  written tiny checkpoint and `conformance!` with a stated profile. It writes no walk.
- `crates/sparkflash-model-dense/tests/part_plugin.rs` registers a part defined outside the crate
  and checks its logits against the catalogue part's, bit for bit.
- `scripts/dense-synthetic-checkpoint.py` writes a random checkpoint of a family no pack names,
  and `scripts/onboard-gate.sh <arm>` runs the served gate (`scripts/served-gate.py all`) on it
  and writes the first numerics reference through `scripts/numerics.py`.

## 4. What holds the rest

Each item the old checklist named is now refused by the compiler, a test or the boot:

| rule | held by |
| --- | --- |
| no `.cu`/`.cuh` and no kernel build script in a model crate | `sparkflash-ops/tests/one_stack.rs` `no_model_crate_owns_cuda_or_a_kernel_build_script` |
| no model crate depends on another (the shared host `sparkflash-model-dense` excepted) | `one_stack.rs` `no_model_crate_depends_on_another_model_crate` |
| a GEMM-class part has a tensor-core body or a `WAIVERS` line naming the gap | the const assertion over `Part::ALL` in `sparkflash_ops::parts` |
| a step whose part has no registered executor | refused by the part's name when the arm opens, before a tensor is read |
| a layer role the walk does not run (compression, indexer, residual mixer, side lookup) | `sparkflash_runtime::plan_layer`, by name, for the roles its host does not frame |
| the crate is a workspace member | cargo refuses to build a crate outside the workspace's `members` |
| the crate is named in both CI files | `sparkflash-model/tests/arms.rs` `every_model_crate_is_named_in_both_ci_files` |
| every model crate is some `ARMS` entry's crate | `every_model_crate_is_an_arms_crate` |
| every `MODELS` descriptor has an `ARMS` entry, every weights selector resolves | `every_registered_model_has_an_arm`, `every_weights_selector_resolves_in_the_registry` |
| a ledger states every strategy once; its weights are the `ARMS` entries naming it | `ledger.rs` `every_arm_states_every_strategy_once`, `ArmLedger::weights` (derived, not listed) |
| a `Route::Pack` arm has a `TYPED_PACKS` row, and only those do | `serve/pack.rs` `every_pack_route_arm_has_a_typed_pack_row` |
| a carried profile parses and keeps its registry identity | `every_carried_profile_parses_and_keeps_its_registry_identity` |
| a profile departs from DeepSeek's defaults only with `measured = "docs/…"` | the profile parser (`defaults.rs`, `an_override_without_a_measured_key_is_refused_by_name`) |
| a new switch is one typed field with one environment alias | `profile.rs` `every_override_settles_on_one_typed_field`, `every_override_moves_its_field` |
| the walk's compute rows, tick configuration, prompt chunks and rounds | the checks `conformance!` generates for the crate |
| lanes, sampled lanes, interleaved prompts and kept turns, served | `scripts/served-gate.py`, through `scripts/onboard-gate.sh` |
| an option the arm does not apply | refused at parse time (`Pack::unapplied`) |

## 5. What the shared host does not run yet

The dense host runs a layer's attention and feed-forward roles. A family whose config names a
compression, an indexer, a residual mixer (hyper-connections) or a side lookup (Engram, PLE)
gets its skeleton from `--emit`, and `onboard` names those roles as unhosted; such a family is
served today only by a released arm that owns its own walk (DeepSeek, GLM, Qwen, Inkling, Dory).
Their move onto the shared walk is tracked in
[engine/04-onboarding-review.md](engine/04-onboarding-review.md).
