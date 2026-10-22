# D41 capture protocol v2: one command, all three classes, the 19 flips reviewed (2026-10-20)

**Status: prepared for the next GPU window; not yet executed.** No v2 capture exists and no
`sparkflash-d41-bars/1` file is claimed. This page supersedes the two-class idle fallback procedure
in [d41-capture-attempt-b70.md](d41-capture-attempt-b70.md) and
[d41-floor-verdict-20261020.md](d41-floor-verdict-20261020.md). The floor discipline is unchanged:
bars come only from `scripts/d41_floors.py`, only with `status ok`, and only after Lead review.

## Why v2 exists

The 2026-10-20 capture was genuine (released snapshot, D38 root verified, repeat class bit-identical)
but covered only `repeat` and `idle_distractor`. The strict floor run refused it for two reasons:
missing required class `busy_distractor`, and 19 rank-1 flips above the flat-position threshold in
`idle_distractor` ("that is not a noise floor"). Neither refusal is fixed by changing a threshold:
v2 makes the full primary plan the only capture plan, makes the strict floor run part of the same
command, and makes the flips a reviewed artefact instead of a count in a log.

## The frozen plan (all three classes)

`scripts/d41_capture_plan.json`, schema `sparkflash-d41-capture-plan/1`, sha256
`22746d98123a4ee67b33edf18c432e5fe067728076a1494b1ebd2514727353a6`:

| block | class | requests | controlled co-tenant |
| :-- | :-- | --: | :-- |
| `idle-A-repeat-139` | `repeat` | 4 | none (zero-noise control; must stay bit-identical) |
| `idle-B-distractor-close-margin` | `idle_distractor` | 5 | none (idle; /metrics must show running=0 waiting=0 before and after) |
| `busy-A-controlled` | `busy_distractor` | 4 probes | one 512-token background generation; `metrics_during` must show `num_requests_running >= 2` while a probe is in flight |

The plan is not a knob: the runner defaults to the frozen sha and refuses a changed plan unless a
reviewed new sha is passed explicitly (recorded in the window JSON). The two-class idle fallback
cannot produce D41 bars and is not a v2 option.

## The one command

```sh
python3 scripts/d41_window.py capture \
  --base-url http://127.0.0.1:8000 \
  --expect-root 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --capture-id d41-next \
  --source-revision <git sha of the frozen tree> \
  --out-dir /home/user/ev-session-<date>/qa/d41/<run-name>
```

Order inside the command:

1. **plan check** - frozen-plan sha256 and schema, all three classes, and a `busy_distractor` block
   with a background generation; refuses before any capture request;
2. **plan dry-run** - `d41_recapture.py --dry-run` (no HTTP), logged;
3. **capture** - `d41_recapture.py` with a fresh `--out` name, its log, `--expect-root`, a
   verifiable `--capture-id` and `--compact`; the script itself refuses before any completion
   request if the served root is not the released snapshot or if /metrics cannot prove idleness;
4. **strict floors** - `d41_floors.py` with the default required-class set (all three classes),
   default flat threshold, and `--out-bars`; no class narrowing and no `--flat-prob` change;
5. **flip investigation** - `scripts/d41_flip_investigation.py` on the capture (below);
6. **summary + manifest** - `d41-window.json`/`.md` and `d41-window-sha256.txt`.

Offline re-analysis of an existing capture (no network; used for tests and the post-window record):

```sh
python3 scripts/d41_window.py analyse --capture <capture.json> --out-dir <fresh dir> --source-revision <sha>
```

## Artefacts (one fresh directory per window; never mixed)

| file | content |
| :-- | :-- |
| `d41-capture-<UTC>.json` | the frozen `sparkflash-d41-capture/1` capture; the only floor input |
| `d41-capture-<UTC>.json.log`, `d41-recapture.log` | the capture script's own log (refusals recorded) and the subprocess log |
| `d41-floors.json`/`.md` or `d41-floors.log` | floor tool output or its refusal |
| `d41-bars.json` | **only if** the floor tool returned `status ok`; a proposal for Lead review |
| `d41-flips.json`/`.md` | the close-margin flip investigation |
| `d41-window.json`/`.md` | capture/floor/flip rows, next actions, non-claims |
| `d41-window-sha256.txt` | sha256 of every artefact above |

Exit codes: 0 = bars written (Lead review still required); 3 = capture exists and requires review
(floors refused, flips found, or class missing); 2 = refused or invalid input (no capture written);
1 = tool failure. A refused window is still evidence and is recorded as such.

## The 19 close-margin flips: investigation, not threshold change

`scripts/d41_flip_investigation.py` re-computes the distractor pair metrics through the exact
`d41_floors` code path and writes every flip above the flat threshold with block/pair ids, both
prompt lengths, position, both top-1 token ids/logprobs/probabilities, and both top-1 margins
(top1 - top2 logprob). It adds per-pair/per-class aggregates and a `verified_against_floor_tool`
block whose counts must agree with the refusing tool.

Acceptable decisions, recorded by the tool:

1. a **reviewed probe-design change** to the close-margin prompt pairs (re-capture afterwards);
2. a **reviewed decision about close-margin positions** (exclude or separately classify them, with
   the flip table attached);
3. not acceptable: widening `--flat-prob` to pass the capture.

It never writes bars and never edits `docs/DECISIONS.md`; any D41 decision is proposed to the Lead
as text.

## Sequence in the next GPU window

1. released snapshot served at TP=4 and the D38 root check passes; instance otherwise idle;
2. run the one command above (capture is tens of seconds of serving; the released-model load
   dominates the window);
3. read `d41-window.md`; on `REQUIRES_REVIEW` attach `d41-flips.md` to the Lead decision and freeze
   nothing; on `OK-BARS-WRITTEN` the Lead reviews `d41-bars.json` before any consumer uses it;
4. copy the capture JSON and log off the node and verify their sha256 against the manifest.

## What this protocol does not claim

- no bars file exists until a v2 capture runs and `d41_floors.py` returns `status ok`;
- no device/token-parity claim: a D41 capture is serving self-consistency evidence (D39);
- no tok/s or step-time claim: this leg measures distributions, not throughput;
- no threshold was widened and no `docs/DECISIONS.md` edit is made from this workstream.
