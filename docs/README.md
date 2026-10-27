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

# Documentation

| Document | What it covers |
| :-- | :-- |
| [design.md](design.md) | How the engine is built and why: data layout, the decode step, DSpark speculation, prefill, serving, correctness rules |
| [performance.md](performance.md) | Current measured numbers, how they were taken, and how they got here |
| [serving.md](serving.md) | Converting weights, launching the four ranks, the HTTP API, behaviour and tuning knobs |
| [web.md](web.md) | The `sparkflash --web` lead-node control plane: TOML, mDNS/SSH discovery, launch/convert actions and live graphs |
| [DECISIONS.md](DECISIONS.md) | The decision register: one row per hard-to-reverse choice, cited from source comments as `Dnn` |
| [multi-model-plan.md](multi-model-plan.md) | The model-pack reorganization plan: current coupling, target contracts, phases and risks |
| [architecture/](architecture/) | The model: CSA2 attention, mHC, MoE, Engram, DSpark, vision, prompt encoding, KV footprint |
| [engine/01-forward-order.md](engine/01-forward-order.md) | The reference forward pass in execution order, with the tensor map |
| [../reference/](../reference/) | Upstream model card, configuration, tokenizer and encoding reference |

`build/` holds data files that tests and scripts read (tokenizer parity evidence, frozen
numerical-gate references, the speed log); it is not documentation.

Source comments sometimes cite older design records under `docs/archive/` or `docs/build/*.md`.
Those records were removed when the project was prepared for release; they remain in the git
history.
