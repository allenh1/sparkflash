# O4 record - b70 dump vs released-snapshot oracle (2026-10-20, PRE-FIX / UNBARED)

**Status: recorded 2026-10-20 by qa-lead (task-4); evidence is on the dev host, not in this repo.**
This is the dated O4 record for the b70 device dump. It is a **PRE-FIX diagnostic**: the
native-arithmetic fixes (task-4..task-9) are not in the dump, no D41 bars file exists, and the run
claims no pass/fail. It replaces the "exploratory position-129 comparison" row of
[serving-parity-plan.md](serving-parity-plan.md) with the full 9-position, machine-readable table.

## Evidence (verified sha256 before use)

| artefact | sha256 | note |
| :--- | :--- | :--- |
| device dump `runs70/logits-130x8.jsonl` | `f548d9c3115c0557cc277ed1b32ce2c82985ec38d299869163759f31fc30aea6` | 9 rows, positions 129..137, 11 frozen fields, vocab 129280 |
| sidecar `runs70/logits-130x8.jsonl.bin` | `43b31438c01955c8359457e077322fc39e6c6a76f330bdf958e87d9ddcad07b5` | 4,654,080 B = 9 x 129280 x 4; offsets tile; logfile line `reference vs fixture: positions=9 all_equal=true` |
| oracle used `native-oracle-suite.json` | `efce16c2637aeab70475a43b8dc0c5cbe1f066b6bb9cd671480c7d7e03c4572a` | frozen `sparkflash-native-oracle/1`, `status: ok`, `root_verified: true`, served root ends `.../1bae0e10...` (D38), suite sha `42f66ad2...`, b69 prompt_len 140 covers p129..p137, target tokens equal the dump's `reference_id` on all 9 rows |
| old oracle (not used) `native-oracle-released.json` | `8ecc423d0e6240811421d95823caaeab396209a8cef8d7ef9b6fe54d3521f475` | early O2-era format: no `schema`/`prompts`/`target_token`; the frozen consumer refuses it and a prefix cannot be proven like-for-like |
| pinned suite `scripts/oracle_prompt_suite.json` | `42f66ad2d8d81ff555eeee2ef837e0cd6ca3cb59bde57320223d9c9b9ef19cd7` | matches the oracle's recorded suite sha |
| golden fixture `sequence-130-released.json` | `e6df8290dcc9cfc96af27ad1a4da165e5f3d8d4e9aac0f98819b0030b738b308` | oracle continuation source |

Full record with the copy manifest, shape checks and exact command:
`/home/user/ev-session-20261020/qa/o4/O4-RECORD.md`; comparison JSON/MD in the same directory
(`compare-native-vs-vllm.json` sha256 `fc91dd53...`, `.md` sha256 `ff3090ec...`).

## Command

```sh
python3 scripts/compare_native_vs_vllm.py \
  --dump <qa>/artifacts/runs70/logits-130x8.jsonl \
  --oracle <qa>/artifacts/native-oracle-suite.json --oracle-prompt b69-fixture-130 \
  --positions 129,130,131,132,133,134,135,136,137 --expect-vocab 129280 \
  --out-json <qa>/o4/compare-native-vs-vllm.json --out-md <qa>/o4/compare-native-vs-vllm.md
```

No `--bars` (none exists) -> exit 0, verdict **UNBARED**; `prefix_mismatches = 0` so the 9
positions are like-for-like; generated `2026-10-20T20:12:12Z`.

## Result (measured, 9 positions)

| metric | value |
| :--- | ---: |
| top-1 agreement rate | 0.0 (0/9) |
| top-5 / top-20 / top-100 / top-1000 overlap mean | 0.0222 / 0.0778 / 0.1056 / 0.1364 |
| max abs logprob delta (max / mean-of-max) | 26.0218 / 18.5496 |
| mean abs logprob delta | 8.16175 |
| KL(oracle \|\| device) mean / p95 | 6.42777 / 14.3411 |
| JS divergence mean | 0.539253 |
| oracle top-k retained mass mean | 0.616579 |

This is the pre-fix distance to the released-snapshot oracle: 0/9 top-1 agreement and far outside the
D40 provisional bars, as expected before the native-arithmetic fixes. It is the baseline a post-fix
dump on the same pinned positions/oracle is compared against, and the format a D41 gate file gates.

## Non-claims

Not PASS/FAIL (UNBARED by construction), not token equality (D39 rule 2), not four-rank/TP=4, not a
serving run, not the current tip, and no performance claim of any kind.
