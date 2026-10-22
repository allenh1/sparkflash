# Documentation

The docs are split into three layers. Read only what the task needs.

## 1. Rules and decisions (read first)

| Document | Purpose |
| :-- | :-- |
| [RULES.md](RULES.md) | The single rule set: the one metric, gates, measurement/clean-row protocol, correctness gate, review, docs process, current focus. Supersedes all older process text. |
| [DECISIONS.md](DECISIONS.md) | Stable decision register (D1-D54). One row per decision; full historical rationale is archived. |

If a topic document disagrees with `RULES.md` on process, `RULES.md` wins. If it disagrees on a
technical contract, the code and the cited decision row win.

## 2. Current engineering handbooks

These describe the system as it is today. They are kept short and are updated with the code.

| Document | Covers |
| :-- | :-- |
| [build/device-chain.md](build/device-chain.md) | Device execution path, layer/stage contracts, the whole-layer device port. |
| [build/performance.md](build/performance.md) | The speed number, per-block cost model, measurement facts, limits. |
| [build/kernels.md](build/kernels.md) | CUDA kernel inventory, native arithmetic, pins, harness status. |
| [build/four-rank.md](build/four-rank.md) | TP=4 host path, host twin, token/logprob evidence. |
| [build/serving.md](build/serving.md) | OpenAI/Anthropic surface, scheduler, paged KV, deployment, metrics. |
| [build/fabric.md](build/fabric.md) | RDMA/ibverbs, NCCL, host/device reduce, fabric state. |
| [build/build-deploy.md](build/build-deploy.md) | aarch64 build, boot conversion, container/deployment. |
| [build/operations.md](build/operations.md) | Window runbook, qualification gates, evidence collection, review steps. |
| [build/evidence.md](build/evidence.md) | Evidence artifact formats and the scripts that consume them. |
| [build/README.md](build/README.md) | Index and non-markdown assets. |

## 3. Reference material

| Document | Purpose |
| :-- | :-- |
| [architecture/](architecture/) | Reference-model analysis: overview, CSA2, mHC, MoE, Engram, DSpark, vision, encoding, KV footprint. |
| [engine/01-forward-order.md](engine/01-forward-order.md) | The reference forward in execution order, with the tensor map. |
| [../reference/MODEL_CARD.md](../reference/MODEL_CARD.md) | Upstream model card. |
| [../reference/encoding/README.md](../reference/encoding/README.md) | Upstream text/vision encoding reference. |
| [../deepseek_v41_flash_4x_dgx_spark_rust_spec.md](../deepseek_v41_flash_4x_dgx_spark_rust_spec.md) | Source specification. |
| [archive/investigations/](archive/investigations/) | DeepInfer and `deepseek-recipe` deep dives (archived). |

## History

[archive/](archive/) holds the old build records, investigations and the full decision log. They are
provenance, not current documentation: they contain superseded numbers, rules and plans, and are not
maintained. When a current handbook and an archived record disagree, the handbook wins.

## Conventions

- One topic per document, current truth only. No status pages, no per-window records, no chronology.
- Numbers carry units and the commit/run they came from, and are labelled **MEASURED**,
  **ESTIMATE** or **DESIGN**.
- Decision IDs are stable and are cited from source comments. Never renumber them.
- A new hard-to-reverse choice adds one row to `DECISIONS.md`; a behaviour change updates the
  matching handbook. Historical detail goes to the archive, not into the handbook.
