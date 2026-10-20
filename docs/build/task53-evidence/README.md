# task-53 / task-57 four-node evidence

Raw per-rank evidence JSON for the resident four-node run (task-57, run 3). The numbers are
interpreted in `docs/build/tp4-host-twin.md`, section 6.

Run 3 (resident): binary sha256
`32ffec4492d19630f858d42bf1efeae209d3e760860c8e1b401712d8f3914b04`,
four GB10 nodes at 192.0.2.80/.88/.59/.68, world 4, TCP collectives on 29601-29604,
prompt `Question: What is the capital of France?\nAnswer:` (11 tokens), max-tokens 8,
`--resident-cache`; seam recording and the reference check were off.

| rank | tokens | rchar/token | disk/token | mean step | resident cache | RSS load / peak |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `[11111,201,8197,14,270,3417,344,28]` | 361,700 B | 418,474 B | 13,948 ms | 84.699 GiB | 85.835 / 90.302 GiB |
| 1 | same | 361,700 B | 414,606 B | 14,017 ms | 84.699 GiB | 85.836 / 90.268 GiB |
| 2 | same | 361,700 B | 418,929 B | 15,276 ms | 84.699 GiB | 85.836 / 90.303 GiB |
| 3 | same | 361,700 B | 414,606 B | 14,411 ms | 84.699 GiB | 85.835 / 90.270 GiB |

Completion text on every rank: ` Paris\nNow, the question is:`.

Cold runs 1 and 2 (same code path, no `--resident-cache`) were collected on the dev host at
`/home/user/phase2-evidence/task53-2026-10-19/run1` and `.../run2`; their token sequences are
`[11111,16,455,5185,294,11111,344,223]` (prompt `The capital of France is`) and
`[11111,201,8197,14,270,3417,344,28]` (the run-2 prompt above), with mean steps 16.85 s and
15.81 s and rchar/token 9,649,668,893 B and 9,650,133,557 B.

Each JSON file is the rank's own `tp4-host-rR.json` as written on its node. The launcher logs
(`run-rR.run3.log`) are not committed but are in the same source directory on the dev host.
