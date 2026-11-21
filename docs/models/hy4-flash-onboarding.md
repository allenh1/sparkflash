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

# Onboarding tencent/Hy4-flash-FP8: the proof of concept

This page records what [adding-a-model.md](../adding-a-model.md)'s path does with a release no
arm in this tree was written for. It is the test of the onboarding architecture, not a served
model: the checkpoint is not registered in `MODELS` or `ARMS`, because the shared walk does not
yet run three of its layer roles (below), and a registry entry would advertise a model that
refuses at boot. The web console's Hugging Face browser previews it like any other repository.

The release's `config.json` is not in the repository (its license is not this tree's); the
run below read a local copy with `--config`. The unit test
`sparkflash_model_dense::onboard::tests::a_hyper_connected_engram_checkpoint_names_the_roles_left`
holds the same result on a small config of the same structure written for the test.

## The release, as the parser reads it

`hy_v4_flash`, 42 layers at hidden 3072, vocabulary 120,832, a 1M-token window. Attention is a
latent (MLA) attention with a query rank, gated output and per-head query/key norms; 12 layers
are sparse (a DeepSeek-style indexer, preceded by a compressed indexer stage) and 30 slide over
a 512-token window with twice the heads. The residual carries four hyper-connection streams
(`hc_mult` 4, with `enable_ihc`), every sublayer is normed before and after (`sandwich_norm`),
and attention has learned sinks. Four layers carry hashed Engram lookups (n-grams up to three,
eight heads an order, a prime-sized table a head). The feed-forward is one dense layer and then
512 routed experts, 16 a token by a sigmoid score with an expert bias, plus a shared expert. The
projections and experts are FP8 E4M3 on 128 x 128 tiles, declared under `text_config`. A vision
tower and a draft head ship beside the backbone.

Three parser changes were needed to read it at all, and each is general: an Engram declared as
per-layer, per-head table sizes (`engram_layer_head_vocab_sizes`), a `quantization_config` nested
under `text_config`, and naming every switch the shared description does not model, so the
preview lists them instead of computing as if they were absent.

## The run

```sh
sparkflash onboard tencent/Hy4-flash-FP8 --config config.json --emit DIR
```

It reports, in order:

- the stored formats (FP8 dense and expert projections, 128 x 128 tiles, E4M3 scales);
- every part the architecture names (embedding, norm, head, MLA attention, sparse indexer, dense
  MLP, routed experts, hyper-connection, Engram, vision) has a catalogue entry on sm121, and the
  only tensor-core gaps are the waived ones every family shares (the head, the one-row FP8
  experts);
- eight switches to describe first: the per-head Engram tables' arithmetic, the sandwich norm,
  learned sinks, the iHC residual mix, the compressed indexer stage, the MLA output gate, the
  query/key norms and the sliding layers' own head count;
- no stored-bytes floor at TP 1, 2 or 4: the shared naming does not spell a hyper-connection;
- the crate it wrote, `sparkflash-model-hy-v4-flash`: `Cargo.toml` (25 lines), `src/lib.rs` (94:
  the config, `architecture()`, `parts()` registering one `MlaAttentionPart` stub) and
  `tests/conformance.rs` (56: the fixture and the stated profile). As a workspace member it is
  clippy-clean, and its seven generated device checks fail by name ("needs a fixture: write the
  tiny checkpoint and open the arm with `sparkflash_model_hy_v4_flash::parts()`");
- the four edits outside the crate: the workspace member, the two CI files, a `MODELS`
  descriptor with its `ARMS` entry to serve under a registry id, and the CLI dependency with one
  `DENSE_FAMILIES` row (`("hy_v4_flash", sparkflash_model_hy_v4_flash::parts)`).

With the MLA part registered, the dense host's structure gate takes the attention layers: a
family's own mixer is the family's (`dense_device_refusal_hosting`). What it still refuses is
named: `sparse_indexer`, `hyper_connection` and `engram`.

## What is left, and whose it is

| left | whose | why |
| --- | --- | --- |
| the MLA part's bodies (latent projections, rotary, gated output, sinks, the sliding layers) | the family crate | the arithmetic only this release has; the catalogue's MLA kernels are DeepSeek's and GLM's shapes |
| the indexer role (fine and compressed) | the shared walk | `plan_roles` refuses an indexer step; DeepSeek and GLM run theirs inside their own walks |
| the residual role (four hyper-connection streams, iHC) | the shared walk | the walk's residual is one stream; GLM's mHC, Qwen's hyper streams and Inkling's block residual need the same widening |
| the side role (Engram) | the shared walk | DeepSeek's and Qwen's lookups land on the layer input inside their own walks |
| the image tower and the draft head | the shared walk | the walk takes no images and has no drafter hook |

The three shared roles are the same gaps the released arms' migration onto `PlanWalk` closes
([engine/04-onboarding-review.md](../engine/04-onboarding-review.md), "W2.5"). Once the walk runs
them, this family's work is its MLA part and its fixture; until then it is refused by name, not
served wrongly.
