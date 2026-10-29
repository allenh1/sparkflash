# Data files

Machine-readable records that tests and scripts read. None of this is documentation; see
[../README.md](../README.md) for that.

| Path | Read by |
| :-- | :-- |
| `task30-evidence/tokenizer-parity.json` | `sparkflash-api` tests: pinned token ids from the reference tokenizer |
| `task32-evidence/` | `scripts/d41_capture_plan*.json` (pinned digests of early probe rows) |
| `reference/speed-gate/` | `scripts/numerical-gate.sh`, `scripts/device-frontier.py`: frozen per-rank references and bars for the numerical gate |
| `reference/s1-device-bounds.json` | `scripts/s1-first-diff.py`: per-layer divergence bounds |
| `speed-log.csv` | `scripts/speed-log.sh`: the early ms/step history (see [../performance.md](../performance.md) for current numbers) |
