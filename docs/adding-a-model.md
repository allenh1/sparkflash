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

# Adding a model

You don't write a crate to add a model. You point SparkFlash at a checkpoint, and it tells you
what it needs. Usually that is a layout file and a few kernel shapes. Sometimes it is one new
shared part. It is never a serving path, a converter, or a branch on the model's name.

This page describes the process at the end of [plan 05](engine/05-runtime-models.md). Some of
it isn't built yet; [the last section](#what-works-today) says what is. The step-by-step
procedure, with every gate and a checklist, is the skill at
[`.agents/skills/adding-a-model/SKILL.md`](../.agents/skills/adding-a-model/SKILL.md). Read
this page to understand the work, and follow the skill to do it.

## The short version

1. Write down what you're adding: the Hub revision, the license, and the producer's modelling
   code.
2. Scan the checkpoint. Read every row.
3. If the scan doesn't know the wiring, write a layout file.
4. If a kernel isn't built at your shape, add it to the build list.
5. If the model has an operation nobody has had, add one shared part.
6. Prove the wiring on a tiny random checkpoint at one, two and four ranks.
7. Check your reading against the producer's own code.
8. Convert, serve, register the numerics, measure.

Most models stop needing new work at step 4.

## Three things to hold onto

**Serve the model its producer built.** Compute what the producer computes, in the types it
computes in. If the producer runs in BF16 and casts to f32 inside its norms, so do we: BF16
outside, f32 inside, nothing wider "to be safe" and nothing narrower. Don't give a model a part
it wasn't trained with because the base happens to have one. CSA2 belongs to DeepSeek V4.1 and
no one else. When the base's body computes a slightly different function, the model gets its
own instance or its own part. The model's facts never bend to fit a body.

**Keep the model's facts out of the code.** A width, an epsilon, a router rule, a split: each
lives in the checkpoint or in the layout file. If you're typing a model's name or one of its
numbers into Rust or CUDA, stop. It wants to be a layout fact or a template argument.

**Leave DeepSeek alone.** It is the base everything runs on. If you touch a shared body, prove
the released model's bits, traces and speed didn't move
(`scripts/open-manifest-ab.py ab`). If it got slower, revert and say so.

## 1. Scan

```sh
sparkflash onboard ORG/REPO
```

The web console prints the same report when you paste the Hub id. It reads the config, the
tokenizer files and the safetensors headers. No weights are downloaded.

Each row is a part of the model with one status. The status tells you what to do:

| Status | What it means | What you do |
| --- | --- | --- |
| `resolved` | a kernel is chosen for prompts and for tokens | nothing |
| `unknown layout` | the config doesn't say how the blocks are wired | write the layout (step 2) |
| `unknown key` | a config key nothing read | read it or ignore it, in the layout |
| `unclaimed tensor` | a tensor no part took | add the stack or tensor to the layout |
| `not built` | the kernel exists, not at this shape or format | add a build line (step 3) |
| `no kernel on this target` | this GPU has no body for the part | port it |
| `no part kind` | an operation the library doesn't have | add a part (step 4) |

Turn the rows into a work list in the model's page under `docs/models/`. Then do one check
the scan can't do for you: for each `resolved` part, open the producer's code and confirm it
is the same function. Same norm, same activation, same router rule. A part that resolves to a
near neighbour is a missing layout fact, and it will serve wrong answers that look fine.

## 2. Write the layout

A config gives sizes and switches. It doesn't say whether a block norms before or after its
branch, where a cast to f32 sits, or how the head folds its streams. That lives in the
producer's Python, and the layout is where you write it down as data. The format is in
[models/layouts/README.md](../models/layouts/README.md), and the shipped files beside it are
worked examples.

First check whether a shipped layout already matches. Most new releases reuse one.

If not, read the producer's modelling file from top to bottom before you write anything. Note
the order of operations in a block, every epsilon and clamp, every `.float()` and
`.to(dtype)`, and keep the line number for each. Then write the file:

* **`[match]`**: the producer's class and `model_type`. This is the one place in the tree
  that compares a family name.
* **`[[keys]]`**: every config key, either read into a fact or ignored with a reason. "Not
  needed" is not a reason. "Training only: dropout is off at inference" is.
* **`[stacks.*]`**: the language stack, and the draft head and vision or audio towers if the
  checkpoint has them. A tower is a stack of ordinary parts, not one opaque box.
* **Part facts**: the function (which attention form, which pool rule, which fold) and the
  types (each f32 cast is its own fact). State how each recurrent block splits across ranks.
* **`[[tensor]]`**: a name template, dtype, shape and split for every tensor. Shapes are
  expressions over facts, never a repeated number.

Put a comment on every fact saying where it came from: a producer line, a config path, or the
tensor that proves it. If you can't source a fact, leave it out and let the scan keep saying
`unknown layout`. A guess here is the most expensive kind of bug.

You're done when the scan has no `unknown layout`, `unknown key` or `unclaimed tensor` rows,
and the layout's tensor plan matches the checkpoint's index at one, two and four ranks
(`crates/sparkflash-ops/tests/layout_equality.rs`).

A layout you're still working on can live in `~/.cache/sparkflash/layouts/`. A finished one
ships in `models/layouts/`.

## 3. Build the missing shapes

Kernels are templates, and a shape is a template argument. The shapes that get built are
listed in one file, `kernels/instances.toml`, and the catalogue is generated from it, so
nothing can claim a shape that wasn't compiled.

For each `not built` row, add an `[instance]` with the shapes one rank sees, then:

```sh
scripts/instances.py --check
scripts/build-cubins.sh
```

Scan again. When an operation has more than one body (a row body and a tensor-core body, say),
measure your new instance at the row counts the model serves, so the resolver's speed table
has a real number to choose by.

## 4. Add a part (rarely)

First make sure it's really new. Read the producer's code next to the nearest part we have:

* Same function, different sizes: that's an instance. Go back to step 3.
* Same function with a variant (a bias, a different epsilon, a gate): add an argument to the
  existing body.
* A different function: a new part.

For a new part:

1. Name it for what it does (`kda_scan`, not the model).
2. Write the host body first, in `sparkflash-weights`, following the producer line by line
   in the producer's types. Everything else is checked against it.
3. Write the device body as a `PartExec`: the lane state it keeps, how it splits across
   ranks, a prompt body and a token body. It sees normed rows in and leaves its branch's rows
   out. It knows nothing about the engine.
4. Put the kernel under `kernels/`, named by operation, with every size a template argument.
5. Refuse by name whatever it can't handle: a rank count that doesn't divide its heads, a
   format it doesn't decode.
6. Test the device body against the host body, alone, at one, two and four ranks.

If your part overlaps one that exists, measure both. The faster one becomes the shared body
and the other is deleted.

## 5. Prove it on a tiny checkpoint

A clean scan means every piece has a kernel. It doesn't mean the pieces are wired correctly.
So build a tiny random checkpoint with the model's structure (add a variant to
`scripts/second-shape.py`), run it through the host reference, and run it on the device at one,
two and four ranks:

```sh
scripts/second-shape.py run --device ... --world 1,2,4
```

Pick widths that make every split real at two and four ranks. A fixture too narrow to split
proves nothing there.

The bar is the same token ids as the reference. When they differ, find out why before you
relax anything. Dump each layer's output on both sides and look at the per-row distance
(`scripts/seam-diff.py layers HOST DEVICE`):

* Every row drifts, starting early: the wiring is wrong. Fix it.
* A few rows jump at a router or a selection layer, and the rest hold: a random fixture's
  near-tie flipped under rounding. Recompute the margins to confirm (`seam-diff.py router`).

A tie band or a seam gate exists for the second case. Both are last resorts, both are
measured, and both need the owner's sign-off.

Then run the conformance suite on the same checkpoint: a whole prompt equals a chunked one, a
continued turn equals a cold one, drafted equals plain, one lane equals four, and one rank
equals two equals four.

## 6. Check against the producer

Step 5 compares us with our own reading of the producer. This step checks the reading.

Write down the tolerance you expect before you run anything. Then run the producer's own
classes (transformers at the revision your layout cites) on the tiny checkpoint and compare
logits. If a producer-side stack can serve the real checkpoint, teacher-force it on the anchor
prompts and record the distance. Do the same for a vision tower with a fixed set of images.

If you miss the tolerance, something computes a different function or sits in the wrong type.
Find it. Don't widen the number after the fact.

## 7. Serve the real thing

1. Download the weights and convert. The shards are computed from the description; there is
   no converter to write.
2. Check memory on every node, then serve at each rank count that fits.
3. Capture the reference and register it as numerics version 1
   (`scripts/numerics.py capture`, `numerics/registry.toml`, a short report under
   `docs/numerics/<model>/`).
4. Run the served gate (`scripts/served-gate.py all PORT --long FILE`).
5. Measure: alone and four at once, greedy and sampled, short and long prompts, first token
   at the largest window, draft acceptance.

The prompt format comes from the checkpoint's chat template, sampling defaults from its
`generation_config.json`, and image preprocessing from its `preprocessor_config.json`. Any
default that departs from DeepSeek's needs a measurement behind it.

## 8. Land it

Sign off your commits. Update the model's page, its numerics report, `docs/DECISIONS.md` for
anything the owner decided, and `NOTICE` for anything adapted from elsewhere. Test what you
changed as you go, and run the full battery once at the end.

## When to stop and ask

Bring the measurement to the owner when:

* a step can't be made free for the released DeepSeek;
* the only way forward changes what the model computes, or narrows a value below the
  producer's type;
* a gate only passes if you loosen it;
* the wiring needs a fact you can't source;
* the model is slower on a shared body and you can't get it back.

Stopping is fine. Shipping a slower or less faithful model is not.

## What works today

As of 2026-11-25 the scan, the layouts, the instance list, the synthetic proof and the
conformance suite all run. GLM is the first model moving onto the base
([08-glm-on-the-base.md](engine/08-glm-on-the-base.md)).

Not built yet: the producer-capture script (step 6), converting and serving straight from the
description, and the chat-template engine (step 7). Until those land, the released models still
serve through their own crates. `sparkflash onboard --emit DIR` can still write a crate
skeleton, and [arm-reference.md](arm-reference.md) describes those internals, but that route
is being retired. Ask the owner before adding a crate.
