# Window kit: four-rank bind-check run (M1, no token yet)

Status: ready-to-run kit for the 2026-10-20 GPU window. Branch task/m1-four-rank-token.
This kit runs the bind-check arm; it does NOT emit a token. Date: 2026-10-20.

## What the run proves and what it does not

Proves, on real GB10 hardware, per rank:
- the 72.308 GiB rank arena plans and streams from the released checkpoint (the same stage measured
  at 332.4-350.9 s/rank, 95,818 verified resident tensors, plan hash 0x9a59035fc40776c8);
- attention projection binding: 208 device-served names bound, 215 table entries (19 host seams
  excluded structurally), set equality with names in both directions;
- routed-expert binding: 61,440 gate/up + 30,720 down names (weights AND scale grids, 46,080 expert
  slices), shared expert 240 names, router 120 names, tail 3 names, each set-checked against the
  geometry and bytes-checked against the arena slice;
- the resident-plan census claims every resident shard by class (unclassified = []), so nothing is
  silently unaccounted for;
- the four-rank TCP collective with -comm-timeout 120 -comm-connect-timeout 120: 80 reduces per
  step (2 per layer), canonical bf16 ascending-rank sum verified byte for byte, 3 steps
  (2 warmup + 1 steady), every step's calls_per_step_ok true.

Does not prove: any forward pass, any logits, any token, any ms/step or tok/s. Evidence says
forward=not-run, tokens=0, hybrid_sequence.executed=false. Do not quote a serving number from it.

## Exact command

One process per rank, launched detached over ssh from the dev host. The canonical generator is:

    scripts/tp4-window-bind.sh --print-argv <R>          # R in 0..3

which prints (all on one line; REMOTE_BIN must be the Lead's staged binary path):

    <REMOTE_BIN> --run-stage tp4-chain \
      --checkpoint <RELEASED_CHECKPOINT> \
      --tp-rank <R> --tp-world-size 4 \
      --evidence-dir <EVIDENCE_ROOT>/<RUN_NAME>/rank<R> \
      --upload --loop \
      --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
      --comm-base-port 29601 \
      --comm-timeout 120 --comm-connect-timeout 120 \
      --warmup-steps 2 --decode-steps 1

No -cubin-dir: with it, -loop invokes the world=1 hybrid diagnostic, which must not run in
four-rank mode (and prints "hybrid_sequence needs --fixture FILE" because fixture=None). The bind
path needs no cubins at all.

The whole run can also be driven by:

    scripts/tp4-window-bind.sh all        # preflight, scp, launch, wait, collect, verdict
    scripts/tp4-window-bind.sh --rank-local <R>   # one rank (Lead's runner can reuse this)

Env overrides: BIN, REMOTE_BIN, CHECKPOINT, EVIDENCE_ROOT, RUN_NAME, BASE_PORT, WALL_LIMIT,
MEM_FLOOR_MIB, WARMUP_STEPS, DECODE_STEPS.

## Evidence contract (per rank directory)

| file | schema | what it carries |
| --- | --- | --- |
| tp4-bind-r<R>.json | sparkflash-tp4-bind/1 | progressive bind object: attention coverage+names, moe classes (names or digests, launch_compatibility, weight_h2d_bytes_per_token=0), tail, census, summary.complete |
| tp4-bind-names-r<R>.txt | text | one line per checked name: class<TAB>set<TAB>name; 185,481 lines when complete (208 attention expected + 208 attention bound + 19 attention host seams + 2x92,520 MoE + 2x3 tail) |
| tp4-loop-r<R>.json | sparkflash-tp4-loop/1 | plan, arena, bind (embedded), collectives walk with steps[].calls/calls_per_step_ok, forward=not-run, tokens=0 |
| run.log, run.exit | text | stdout+stderr and exit code |

tp4-bind-r<R>.json is written INSIDE the arena epoch at three stage boundaries (attention, then MoE,
then tail+census), so a bind failure at a later stage still leaves the earlier stages and the error
string on disk. The walk evidence is written only after the bind file.

## Expected timing

- header load + plan: ~10-20 s;
- arena allocation + streaming upload: 332.4-350.9 s/rank (measured; managed memory, 72.308 GiB);
- bind: seconds (46,080 byte checks, ~92.5k set memberships);
- transport open + 3-step walk: seconds after the straggler rank binds (connect timeout 120 s covers
  the ~20 s stagger);
- wall from launch to last run.exit: ~6-7 min. WALL_LIMIT=1500 s is a timeout, not an estimate.

## Pass/fail lines

The verdict prints, per rank:

    PASS rank R: bind complete, 208/208 attention, 92420 moe names, census 95818 resident, 80 reduces/step
    VERDICT: PASS - four-rank arena boot + full-rank bind coverage + 80-reduce walk, no token claimed

and on a defect:

    FAIL rank R: <failed checks, named>
    VERDICT: FAIL - <n> rank(s); named names are in tp4-bind-r<R>.json missing/unexpected and run.log

A genuine bind or collective defect (missing/unexpected names, byte mismatch, wrong reduced value)
is a FAIL and must be fixed; absence of a token is NOT a failure of this run (it never claimed one).

## Fallbacks, in order

1. If the binary fails before the arena (host-side panic): run the same argv with --plan-only on one
   node (seconds, no GPU): the plan/evidence path proves the binary and checkpoint wiring, and names
   the panic as host-side.
2. If the MoE bind fails: tp4-bind-r<R>.json still holds the attention bind and the error string; the
   missing/unexpected lists name the plan-vs-geometry disagreement. Do not edit the sets to fit.
3. If the collective walk fails after a complete bind: the bind result stands as the window's
   evidence; the transport error is in run.log (per-rank) and collectives in tp4-loop-r<R>.json.
4. If the staged binary itself is suspect: revert staging to the previous staged sha (Lead's runner
   records it) and rerun the same argv; the CLI flags are the same, only the bind classes change.

## Known risk before launch

The set checks derive their expected side from the model geometry (config), while the bound side
comes from the rank plan built from the REAL checkpoint headers. They agreed in the 2026-10-20
attention run (208/208), but the MoE bind is the first time 92,520 names are checked against the
real headers. If the released checkpoint differs from the TensorPlan-derived geometry anywhere in
the MoE tree (e.g. a router bias_vl missing on a layer), the bind fails with that name in the JSON:
that is a finding, not a flake, and the attention-only object is already on disk.
## Window outcome 2026-10-20 (package b69ebb2): PASS, pre-token

The kit ran as written on all four nodes (same argv, no `--cubin-dir`, 120 s comm timeouts): 4/4 ranks
exit 0, launch to last exit 469 s, every rank binding before the walk. Per rank `tp4-bind-r<R>.json`:
attention 208/208 (19 host seams excluded, no missing/unexpected), routed gate/up 61,440, routed down
30,720, shared 240, router 120, tail 3/3, all classes complete, `summary.complete` true,
`device_bound_names` 92,731, per-token weight H2D 0; census 95,818 with `unclassified == []` and
`special_token 3` (the checkpoint's image_start/image_end/image_newline, claimed by class b69ebb2);
plan 95,818 resident / 4 disk / 72.308 GiB / hash 0x9a59035fc40776c8; walk 3 steps x 80 calls all
`calls_per_step_ok`, 819,200 B/step, wall_ms 41/22/20; `forward` not-run, `tokens` 0. One count
correction from the run: the names manifest carries 185,481 lines, not 185,462 - the 19 attention
host-seam names are written too (208 + 208 + 19 + 2 x 92,520 + 2 x 3). The kit's verdict check and this
doc carry the corrected count. The only launch blocker left is the routed-down stored-layout ABI row
(`launch_compatible: false`), so the row is a pre-token PASS, not a token.

