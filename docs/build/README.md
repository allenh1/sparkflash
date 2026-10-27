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
