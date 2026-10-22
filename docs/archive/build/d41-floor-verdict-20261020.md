# D41 floor verdict: the B70 idle-only capture cannot produce a gate file (2026-10-20)

**Status: recorded 2026-10-20 by qa-lead (task-4). Verdict: REQUIRES A NEW CAPTURE; no
`sparkflash-d41-bars/1` file was written and no threshold was invented.**
This corrects the state of [d41-capture-attempt-b70.md](d41-capture-attempt-b70.md) (which records
only the 2026-10-20T00:46:45Z refusal against the heretic snapshot): a later attempt in the same
window, `d41-capture-20261020T012706Z.json`, **did** capture against the released snapshot, and this
page is its floor verdict.

## Capture: genuine, released-snapshot (D38 satisfied), but a two-class fallback

| check | result |
| :--- | :--- |
| capture | `/home/user/phase2-evidence/window-2026-10-18/d41-capture-20261020T012706Z.json`, 32,191,965 B, sha256 `0de8d3173860d3de9f6c16ce4857b70a94850ce9cf1c54a633abab572fac5553` |
| schema | `sparkflash-d41-capture/1`; `status ok`; `root_verified true`; `warnings []` |
| time | 2026-10-20T01:27:06Z -> 01:27:15Z (9.1 s wall, 9 requests) |
| served root | `.../snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170` = `expect_root` = released snapshot (D38) |
| plan | `scripts/d41_capture_plan_idle.json` sha256 `4933e49f...`, byte-identical to the repo file; the documented fallback of the primary plan `scripts/d41_capture_plan.json` sha256 `22746d98...` |
| classes | `repeat` (4 requests / 3 pairs) and `idle_distractor` (5 requests / 3 pairs); **`busy_distractor` absent** - the fallback's own `fallback.reason` says "the busy envelope is a named gap, not a gate input" |
| idleness | both blocks `idle_verdict: idle`, `observed_other_traffic: false`, no external peers |

## Floor runs: both refuse (exit 2, nothing written)

1. Strict (D41 defaults, all three required classes):
   `REFUSED: missing required class(es): busy_distractor (captured classes: repeat, idle_distractor)`.
2. Diagnostic two-class run (`--require-classes repeat,idle_distractor`, no `--out-bars`):
   `REFUSED: bar class idle_distractor: 19 rank-1 flip(s) above the flat-position threshold (top-1
   probability >= 0.05); that is not a noise floor`.

So the capture cannot freeze the D40 provisional thresholds even for the class it does have. The
`repeat` control is bit-identical (all pairs KL 0, |delta| 0, top-k overlap 1.0), so the capture
mechanism and the server are deterministic for an identical request - and the `idle_distractor`
result is therefore a real self-consistency signal of the served reference, not capture noise:

| idle pair | shared positions | top-1 agreement | rank-1 flips above flat | KL mean | abs max | top-20 overlap mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `len-130 <-> len-131` | 129 | 0.9302 | 9 | 0.0656 | 5.034 | 0.8833 |
| `len-131 <-> len-132` | 130 | 0.9615 | 5 | 0.0859 | 5.917 | 0.8765 |
| `len-138 <-> len-139` | 137 | 0.9635 | 5 | 0.0669 | 4.172 | 0.8730 |

Full evidence, the exact commands and the copy manifest:
`/home/user/ev-session-20261020/qa/d41/` (`D41-RECORD.md`, `d41-floors-strict.log`,
`d41-capture-inspection.json`) with the floor tool `scripts/d41_floors.py` sha256
`7e27603f82b0182f52c53b4cfc8c143a757e2f5b96fa063d46ed4683739881a2` at base `c62fb0c`.

## What a future D41 capture needs

1. A window that serves the **released snapshot** at TP=4 (a D38 window; not the heretic service).
2. The **primary** plan `scripts/d41_capture_plan.json` (`22746d98...`), all three classes,
   including `busy-A-controlled` (4 probes + a 512-token co-tenant generation; `metrics_during`
   must show `running >= 2`). The capture itself is tens of seconds (9.1 s measured for 9 requests;
   full plan is an estimate); the released model load dominates the window.
3. Investigate the 19 close-margin rank-1 flips before freezing a floor: either a reviewed plan
   change to the probe design or a reviewed decision about close-margin positions. Widening
   `--flat-prob` to pass a capture is not acceptable - propose decision text to the Lead, never edit
   `docs/DECISIONS.md` in this workstream.
4. `python3 scripts/d41_floors.py --capture <capture> --out-json ... --out-md ... --out-bars ...
   --source-revision <git sha>`; only `status: ok` is a gate file, and the Lead reviews the bars
   file before `compare_native_vs_vllm.py --bars` consumes it.

The serving-parity gate remains blocked on a valid D41 bars file. No D41 gate file exists today.
