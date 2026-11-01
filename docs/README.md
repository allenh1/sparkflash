# Documentation

| Document | What it covers |
| :-- | :-- |
| [design.md](design.md) | How the engine is built and why: data layout, the decode step, DSpark speculation, prefill, serving, correctness rules |
| [performance.md](performance.md) | Current measured numbers, how they were taken, and how they got here |
| [optimization-strategy.md](optimization-strategy.md) | Making a model fast on GB10: the timing and exactness probes, kernel practice, the hardware's measured limits, strategy, and a record of what each optimisation bought |
| [serving.md](serving.md) | Converting weights, launching the four ranks, the HTTP API, behaviour and tuning knobs |
| [web.md](web.md) | The `sparkflash --web` lead-node control plane: TOML, mDNS/SSH discovery, launch/convert actions and live graphs |
| [agent-setup.md](agent-setup.md) | Step-by-step instructions for an AI agent to build, configure, stage and launch a four-node cluster (or one local Qwen server) |
| [DECISIONS.md](DECISIONS.md) | The decision register: one row per hard-to-reverse choice, cited from source comments as `Dnn` |
| [models/qwen38-flash-next-nvfp4.md](models/qwen38-flash-next-nvfp4.md) | Qwen3.8-Flash-Next (NVFP4): architecture, quantisation, the PLE n-gram table on the Engram paging path, and the TP=1 device arm |
| [../crates/sparkflash-model-glm5-next/](../crates/sparkflash-model-glm5-next/) | GLM-5.3-Flash (FP8, TP=4): the pack's architecture, tensor plan, host reference and device arm, documented in its crate root |
| [architecture/](architecture/) | The model: CSA2 attention, mHC, MoE, Engram, DSpark, vision, prompt encoding, KV footprint |
| [engine/01-forward-order.md](engine/01-forward-order.md) | The reference forward pass in execution order, with the tensor map |
| [../reference/](../reference/) | Upstream model card, configuration, tokenizer and encoding reference |

`build/` holds data files that tests and scripts read (tokenizer parity evidence, frozen
numerical-gate references, the speed log); it is not documentation.

Source comments sometimes cite older design records under `docs/archive/` or `docs/build/*.md`.
Those records were removed when the project was prepared for release; they remain in the git
history.
