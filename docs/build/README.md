# Engineering handbooks

Current-state documents. Process rules live in [`../RULES.md`](../RULES.md); decisions live in
[`../DECISIONS.md`](../DECISIONS.md).

| Handbook | Read it for |
| :-- | :-- |
| [device-chain.md](device-chain.md) | Device path architecture, stage contracts, whole-layer port status and invariants. |
| [performance.md](performance.md) | The published ms/step, per-block host/kernel split, measurement constraints, trajectory. |
| [kernels.md](kernels.md) | Kernel inventory, native arithmetic rules, pins, phase-3 harness status. |
| [four-rank.md](four-rank.md) | TP=4 host execution, host twin, token evidence, logprob rank contract. |
| [serving.md](serving.md) | HTTP surfaces, scheduler, paged KV, deployment, metrics, remaining gaps. |
| [fabric.md](fabric.md) | RDMA, NCCL, collectives, exactness, fabric prerequisites. |
| [build-deploy.md](build-deploy.md) | aarch64 toolchain, boot conversion, container and deployment. |
| [operations.md](operations.md) | Window sequence, qualification gates, evidence collection, review procedure. |
| [evidence.md](evidence.md) | Artifact schemas and consumers. |

## Non-markdown assets (stay in this directory)

- [`speed-log.csv`](speed-log.csv) — the authoritative record of published speed rows
  (`scripts/speed-log.sh`).
- [`reference/speed-gate/`](reference/speed-gate/) — frozen CPU-twin reference artifacts for
  `scripts/numerical-gate.sh`.
- [`d38-vllm-baseline-20261021.json`](d38-vllm-baseline-20261021.json) — CI perf-regression
  baseline.
- `task30-evidence/`, `task32-evidence/`, `task53-evidence/` — frozen JSON evidence bundles.
- `d41-*.json` — frozen D41 bars/floors.

Historical build records were moved to [`../archive/build/`](../archive/build/).
