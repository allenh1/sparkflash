# Device-arm row readiness: what must be true before the first probe row runs

Written while the cluster is hands-off (the user's vLLM is serving). Nothing here has been executed on the
nodes. This is the checklist the first window is spent against, so the window is measurement, not debugging.

## 1. What the first row is, and what it is not

It is a **frontier probe**: the device arm runs the layer until a stage refuses, writes
`tp4-layer-device-r{rank}.json` per rank, and aborts. It is **not** a speed row and **not** a numerical-gate
candidate:

- the **D54 arm stays unopened** - a row that stops early has no fixture output to compare;
- **no ms/step** may be quoted from it (the clock is time-to-refusal);
- every device value in it is **unvalidated against the host reference** and the document says so in `note`.

Expected frontier today: **`kv_path`**, refusing because `kv_norm` is not in the device path yet (the fused
`kv_norm_rope_bf16` is comm-lead's task-26). Once that lands: **`reference_seam`** (no compressor kernel).
The frontier must come from the four documents, never from stderr.

## 2. Preconditions, by owner

| # | Gate | Owner | State |
| --- | --- | --- | --- |
| 1 | Branch merged and gate green on the exact tip | Lead | in progress |
| 2 | `scripts/device-frontier.py` verdict tool green on self-test (16/16) | qa-lead | done on `task/device-frontier` |
| 3 | `kv_norm_rope_bf16` + twin + pin, kv stage wired | comm-lead / gemv | **not landed** |
| 4 | `RunReduce`/`DeviceReduce` merged so the two collectives can be wired | comm-lead | **not landed** |
| 5 | `moe_shared` body (4 launches, f32-reference arm) | gemv | in progress |
| 6 | Pinned cubins on each node match the baseline the binary was built with | Lead (staging) | at window time |
| 7 | Preflight: 0 `sparkflash-*` processes, MemFree >= 110 GiB on all four | Lead | at window time |

Items 3-5 are **not** required for a probe row: the probe is expected to stop before them. They are required
for a row that completes a layer.

## 3. The exact probe sequence (four-rank, frozen argv)

```
# per node, before the row: clean the page cache (the band protocol)
ssh allenh1@<node> "python3 /tmp/dropcache.py"

# the row, from the unversioned tool directory
cd /home/user/ev-session-20261020/tools
EXTRA_ARGS="--forward-reference-ids 491,367,201,80,3302,201,54,201,4341 --forward-lean" \
SPARKFLASH_LAYER_DEVICE=1 \
SPARKFLASH_TOLERANCE_ARM="probe-row: no numerical arm opened (frontier probe, not a gate candidate)" \
  bash run-fourrank.sh <tag> <run-name> token
```

Then, in the run directory:

1. **Verdict**: `python3 scripts/device-frontier.py <run-dir>` -> exit 0 coherent probe / 1 finding / **2 stop**
   (a missing or unreadable document is never a pass).
2. **Agreement with the stage order**: the parsed chain must be a prefix of the 19-stage order and the stopped
   stage must be the next index. A chain shorter than the stopped stage's index is the lost-chain defect the
   tool exists to catch.
3. **Arm agreement**: the `arm` string must be the device arm; a host-arm document in its place is a finding.
4. **Collective count** (only when a row completes a layer): `python3 scripts/check-collective-count.py`
   expects `layers * 2` per rank.

## 4. What would make the row invalid (and must void it, not explain it)

- any rank missing its frontier document, or ranks disagreeing on the stopped stage;
- `arm` naming the host arm while the environment asked for the device arm;
- a refusal whose text does not name a stage in `STAGE_CONTRACTS`;
- a row that stops after `all_reduce` but reports a collective count other than `2 * layers`;
- a run next to serving or with a non-zero `sparkflash-*` process - **void, never quoted**.

## 5. What the probe is allowed to tell us

- which stage the arm actually reaches (the frontier), and whether the four ranks agree;
- that spans/shapes/ABIs are the ones the launchers declare, because they are asserted at open;
- the geometry the row was built with (`max_positions`, `rope_table_bytes`, `slab_bytes`).

It cannot tell us anything about performance, and it cannot tell us whether any bf16 value is right. The first
thing that can is a completed layer plus the D54 gate against the frozen reference.

## 6. Rollback

The device arm is opt-in (`SPARKFLASH_LAYER_DEVICE=1`); unset it and the host arm runs unchanged. No branch
revert is needed to go back to a known number - the published 179.81 ms/step row is the host arm on `main`.
