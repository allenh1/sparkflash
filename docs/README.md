# Documentation

| Document | What it covers |
| :-- | :-- |
| [design.md](design.md) | How the engine is built and why: data layout, the decode step, DSpark speculation, prefill, serving, correctness rules |
| [performance.md](performance.md) | Current measured numbers, how they were taken, and how they got here |
| [optimization-strategy.md](optimization-strategy.md) | Making a model fast on GB10: the timing and exactness probes, kernel practice, the hardware's measured limits, strategy, and a record of what each optimisation bought |
| [thermal-fan-control.md](thermal-fan-control.md) | The GB10 fan path that actually works (EC mailbox over FF-A eSPI), the max-RPM setup on all four nodes, measured temperatures and the firmware caveat |
| [serving.md](serving.md) | Converting weights, launching the four ranks, the HTTP API, behaviour and tuning knobs |
| [web.md](web.md) | The `sparkflash --web` lead-node control plane: TOML, mDNS/SSH discovery, launch/convert actions and live graphs |
| [adding-a-model.md](adding-a-model.md) | Adding a model without a crate: the scan, the layout file, kernel shapes, a new part, the proofs, and serving; what works today |
| [arm-reference.md](arm-reference.md) | The released arms' internals: the one stack, typed ops and the execution plan, the serving contract, profiles and switches, parity, shared homes, the dynamic Hugging Face path |
| [agent-setup.md](agent-setup.md) | Step-by-step instructions for an AI agent to build, configure, stage and launch a four-node cluster (or one local Qwen server) |
| [rdma-exchange.md](rdma-exchange.md) | The direct-RDMA exchange: the hardware gate, the four-rank measurements, the two wait bugs, the A/B that returned 2% and not 20%, and why |
| [DECISIONS.md](DECISIONS.md) | The decision register: one row per hard-to-reverse choice, cited from source comments as `Dnn` |
| [models/qwen38-flash-next-nvfp4.md](models/qwen38-flash-next-nvfp4.md) | Qwen3.8-Flash-Next (NVFP4): architecture, quantisation, the PLE n-gram table on the Engram paging path, and the TP=1 device arm |
| [models/glm53-flash.md](models/glm53-flash.md) | GLM-5.3-Flash (FP8, TP=4): the released architecture, its serving frontend (backend, prompt codec, descriptor), the tiny-fixture run, and what the arm does not serve yet |
| [models/dory1-4b-bf16.md](models/dory1-4b-bf16.md) | NVIDIA Nemotron Labs Dory1.0-4B (BF16): looped nGPT arithmetic, the 507-name tensor plan, the TP=1 device arm and the CPU host reference |
| [models/qwen36-35b-a3b-fp8.md](models/qwen36-35b-a3b-fp8.md) | Qwen3.6-35B-A3B (FP8): the Qwen3.5-MoE hybrid on the generalised Qwen device arm at TP=1 (plain residual, full attention, FP8 dense), and what it still refuses |
| [models/inkling2-small-tp4.md](models/inkling2-small-tp4.md) | Inkling 2 Small (NVFP4): the 594B/17B-active hybrid, four-Spark memory feasibility, and the parts, kernels and arm it needs |
| [engine/09-inkling-mini-loop-device.md](engine/09-inkling-mini-loop-device.md) | Inkling 2 Mini on the device walk: applications, per-pass caches, the message ops, and the order of work |
| [models/inkling-2-mini.md](models/inkling-2-mini.md) | Inkling 2 Mini (NVFP4/MXFP8): the looped 24-layer release, its shared core and cross-pass message path, the layout that claims all 1,244 tensors, and the scan's work list |
| [models/hy4-flash-onboarding.md](models/hy4-flash-onboarding.md) | Tencent Hy4-Flash (FP8): the onboarding proof of concept; what `sparkflash onboard --emit` wrote, the four outside edits, and the roles the shared walk still refuses |
| [../crates/sparkflash-model-inkling2/](../crates/sparkflash-model-inkling2/) | Inkling-2-Small: the typed config, exact 390 GB tensor plan and TP4 slices, the f32 host reference, the released prompt template, and the device kernel catalogue |
| [../crates/sparkflash-model-glm5-next/](../crates/sparkflash-model-glm5-next/) | GLM-5.3-Flash: the pack's tensor plan, host reference, device arm and rank-file converter, documented in its crate root |
| [../crates/sparkflash-deepseek-forward/](../crates/sparkflash-deepseek-forward/) | The V4.1 device forward, moved out of sparkflash-cli and composed from the model pack and the shared kernel stack |
| [architecture/](architecture/) | The model: CSA2 attention, mHC, MoE, Engram, DSpark, vision, prompt encoding, KV footprint |
| [engine/01-forward-order.md](engine/01-forward-order.md) | The reference forward pass in execution order, with the tensor map |
| [engine/03-runtime.md](engine/03-runtime.md) | The shared runtime: one home for lanes, drafting rounds, prefill and group mirroring, and the plan to move every arm onto it |
| [engine/05-runtime-models.md](engine/05-runtime-models.md) | The agreed direction: models built at load from the checkpoint, no model crates, DeepSeek's forward as the base engine, one shared kernel body an operation |
| [engine/06-base-reads-description.md](engine/06-base-reads-description.md) | Milestone 2 of plan 05 in detail: what DeepSeek's forward is built from today, the rule that it opens from what the scan resolved, the proofs, and the four pieces of work |
| [engine/07-base-ranks-formats-seam.md](engine/07-base-ranks-formats-seam.md) | Milestone 3 of plan 05 in detail: the base at one, two and four ranks, stored format as a fact the engine chooses by, and the seam for parts it has no body for |
| [../reference/](../reference/) | Upstream model card, configuration, tokenizer and encoding reference |

`build/` holds data files that tests and scripts read (tokenizer parity evidence, frozen
numerical-gate references, the speed log); it is not documentation.

Source comments sometimes cite older design records under `docs/archive/` or `docs/build/*.md`.
Those records were removed when the project was prepared for release; they remain in the git
history.
