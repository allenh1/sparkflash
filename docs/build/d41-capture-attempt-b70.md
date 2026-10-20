# D41 capture attempt: B70 idle-only fallback (refused before capture)

Status: **attempted and refused; there is no capture.** On 2026-10-20T00:46:45Z the idle-only
fallback plan was run against the already-running instance on `node0` and `d41_recapture.py`
refused it at the D38 served-root check. This page is the evidence record: the refusal log is not a
capture, and the attempt is not a failed measurement. It is a blocked leg.

## What was attempted

Plan identity was confirmed on the host at main `2de9c07` before any node action:

```sh
python3 scripts/d41_recapture.py --plan scripts/d41_capture_plan_idle.json --dry-run
```

Exit 0. The dry-run shows two blocks and 9 requests, with no background generation:

| block | class | requests | background |
| :-- | :-- | --: | :-- |
| `idle-A-repeat-139` | `repeat` | 4 | null |
| `idle-B-distractor-close-margin` | `idle_distractor` | 5 | null |

The idle plan is blocks 0 and 1 of the primary plan copied verbatim (the plan fallback field says
`do not extend the primary plan for a fallback run`). Primary plan sha256
`22746d98123a4ee67b33edf18c432e5fe067728076a1494b1ebd2514727353a6`.

Pinned identities:

- idle plan file: `scripts/d41_capture_plan_idle.json`, sha256
  `4933e49fcad96e8f3308a1c58a7eaae53d3dbf0668533cc905d18a5c3b4bfee8`
- capture script: `scripts/d41_recapture.py`, sha256
  `e8ddcc017bb6e847927453e6b1be5edc37ffd26833fbbf40e5642cbb765234ee`
- node: `node0` (192.0.2.80); `127.0.0.1:8000` is its own bind
- evidence directory: `/home/user/phase2-evidence/window-2026-10-18/`

Both files were **copied, not built**, to the node evidence directory; the remote sha256s matched the
host values above. The command that ran on the node was:

```sh
cd /home/user/phase2-evidence/window-2026-10-18
python3 d41_recapture.py \
  --base-url http://127.0.0.1:8000 \
  --plan d41_capture_plan_idle.json \
  --out d41-capture-$(date -u +%Y%m%dT%H%M%SZ).json \
  --capture-id d41-b70 --tag d41 \
  --expect-root 1bae0e1088993b9b82de08f59de5be5a34e42170 --compact
```

It exited 2 before any capture request. The `$(date -u ...)` expansion named a candidate output
`d41-capture-20261020T004645Z.json`; that file was never written.

## The refusal

| item | value |
| :-- | :-- |
| attempt time | 2026-10-20T00:46:45Z (2026-10-19 20:46 EDT) |
| node / endpoint | `node0` (192.0.2.80), `http://127.0.0.1:8000` |
| exit code | **2** (refused) |
| served root | `/root/.cache/huggingface/hub/models--pqhaz--DeepSeek-V4.1-Flash-1017-heretic/snapshots/12d58ad0878b5008e80d720251adb969a0069412` |
| expected released root | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| capture JSON | **none** - the script refused before writing `--out` |
| capture requests sent | **zero** |
| metrics snapshots taken | **zero** |
| refusal log (node) | `/home/user/phase2-evidence/window-2026-10-18/d41-capture-20261020T004645Z.json.log` |
| refusal log sha256 | `e292049019052f0fbb287fc07baed4619aa196a4d9e4c1f5227c38b6633c3352` |
| refusal log size | 437 bytes |
| refusal log (host copy) | same path under the host `phase2-evidence/window-2026-10-18/` |

The log records the start line and the refusal. The only HTTP request made was the script's own
root probe (`GET /v1/models`) to read the served root; no completion request was sent, so no D41
capture evidence and no idle metrics exist for these 9 plan requests.

```text
2026-10-20T00:46:45Z d41 re-capture start plan=d41_capture_plan_idle.json out=d41-capture-20261020T004645Z.json dry_run=False
2026-10-20T00:46:45Z REFUSED: served root '/root/.cache/huggingface/hub/models--pqhaz--DeepSeek-V4.1-Flash-1017-heretic/snapshots/12d58ad0878b5008e80d720251adb969a0069412' is not the released snapshot '1bae0e1088993b9b82de08f59de5be5a34e42170'; refusing to capture D41 evidence against another checkpoint (D38)
```

## Conclusion

**Not a measurement leg; blocked on the instance matching the released snapshot.** The deployed
`pqhaz` / `DeepSeek-V4.1-Flash-1017-heretic` snapshot `12d58ad0...` is explicitly not a parity
target under D38, so the guard firing is the designed outcome, not a failure to fix. Do not treat
the refusal log as a capture and do not feed it to `d41_floors.py`; there is no `sparkflash-d41-capture/1`
document in existence.

- No retry was attempted and no flag was used to bypass the guard.
- The instance was left up and untouched: no stop/restart, no `nvidia-smi`, no containers, no GPU
  work, no node configuration.
- Unblock condition: an instance is reachable that serves the released snapshot
  `1bae0e1088993b9b82de08f59de5be5a34e42170`. Re-run the same idle-only plan then, with a fresh
  `--out` filename; never overwrite an existing capture.
- Staged for that re-run and retained on Lead instruction: node EVID holds `d41_recapture.py`
  (`e8ddcc01...`) and `d41_capture_plan_idle.json` (`4933e49f...`); host worktree
  `/home/user/wt/b70-capture` is at main `2de9c07` (the main whose plan and script have those
  sha256s).

This record is branched off main `66bf2000324f66f5e3bf867b18ae0494476547f1` for batch 12.