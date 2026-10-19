# D41 floor computation

Status: task-41, window-free. Script, schema, tests and this record exist; **no D41 floor has been
computed yet**. The numbers can only be frozen after the D41 idle re-capture (task-42) runs on the
Lead's go-ahead. This page records the computation, the rule mapping, and the refusals that stop it
from inventing a floor.

## Why this exists

D40 (docs/DECISIONS.md) froze the serving-parity protocol and left the numeric bars provisional.
D41 freezes them from the measured oracle self-consistency floor, with every bar at least 3x the
measured floor away from perfect. That rule was prose; this task makes it a script with provenance
hashes and tests. The point is to be right before the idle re-capture, so the window produces
numbers that can be frozen rather than numbers nobody can defend.

## Input

A D41 capture, schema sparkflash-d41-capture/1, produced by scripts/d41_recapture.py. The schema is
docs/build/d41-floor-schema.md; it is the one interface between capture and floor computation.
Everything is offline JSON: the floor computation never uses a server, GPU, node or network.

One or more captures may be passed with --capture. Classes may be split across files; the captures
must agree on the served root, the prompt-suite sha256 and expect_root.

## Classes

- **repeat**: the same request body sent twice consecutively while the instance is idle. This is
  the zero-noise control. It must be bit-identical at every shared position: KL exactly 0,
  |delta| exactly 0, top-k overlap exactly 1.0 and rank-1 same. If it is not, the whole run is
  refused (exit 2); a non-zero repeat is not a floor, it is a finding. This class never contributes
  to a bar.
- **idle_distractor**: close-margin prompt pairs at different prompt lengths or batch shapes,
  captured while idle. This is the class the D41 gate bars are derived from.
- **busy_distractor**: the same kind of pair captured while other work is in flight. It records the
  load-state envelope and is not the D41 gate class (task32-root-cause.md section 4.3).

The idleness proof is validated, not trusted: idle blocks need metrics_before and metrics_after at
0 running and 0 waiting, observed_other_traffic false, external_peers empty and idle_verdict idle.
A block with any traffic is tagged busy by the capture script; the floor script refuses a busy
verdict without co-tenancy evidence (external_peers non-empty, or a metrics_during snapshot with
running > 1 or waiting > 0) and refuses idle_verdict unproven outright. Non-zero baselines are
never mixed into an idle class.

## Metrics

For each pair, at every shared position, over the shared support (the token ids present in both
returned top-1000 bands; both sides are renormalised over that support):

- KL: the mean of the two directional KLs (nats); both directions are recorded. Log-space
  computation keeps the result finite when one side's long-tail probability underflows.
- JS divergence: symmetric, bounded.
- |delta|: per-token absolute logprob difference over the shared support; p50/p90/p95/max.
- top-k set overlap: |top-k(A) intersect top-k(B)| / k for k = 20, 100, 1000. Top-k overlap uses
  k >= 20; the old top-5 set bar is dropped (its measured floor is below the D40 provisional bar,
  so it is unmeasurable; task32-root-cause.md section 4.2).
- rank-1: stable when the argmax is the same, or when either top-1 probability is below
  --flat-prob (default 0.05), because a flip at a flat position is not a noise failure. A pair
  with a rank-1 flip above the flat threshold cannot be a floor for the bar class.

## Floor statistics

Pair-level values are summarised per class. Central floors are medians over pair comparisons; tail
floors are p90 over pairs. A single outlier therefore cannot widen a median-based bar: it is
listed in the outliers array and left out of the floor.

| bar | floor statistic |
|:--|:--|
| kl_mean_max | median over pairs of pair KL mean |
| kl_p95_max | p90 over pairs of pair KL p95 |
| max_abs_delta_max | p90 over pairs of pair max abs delta |
| mean_abs_delta_max | median over pairs of pair mean abs delta |
| js_mean_max | median over pairs of pair mean JS |
| top1_agreement_min | aggregate rank-1 stable position rate |
| topK_overlap_mean_min (k=20/100/1000) | median over pairs of pair mean top-k overlap |

An outlier is reported when a central pair value exceeds --outlier-factor (default 3) times the
median (or, for overlap, falls below the equivalent lower threshold). Outliers are evidence to
review; they never change a floor.

## Bars and the 3x rule

Ceilings (KL, JS, abs delta): value = max(D40 provisional, 3 x floor).

Floors (rank-1 agreement, top-k overlap): express the noise as a distance from perfect, so the
floor envelope is 1 - 3 x (1 - floor); value = min(provisional, that envelope). When the envelope is
below the D40 provisional, the provisional is stricter than the measured noise allows and the run
is labelled requires_review rather than silently adopting either number. The top-5 bar is dropped
instead of being "frozen" at a value its own floor fails.

positions_min stays at the D40 provisional (9): the compared-position count is a protocol choice,
not a floor measurement.

Output status is ok or requires_review, and the process exits 3 for requires_review even though the
outputs are written. The D41 record must say why; a bar changed after a gate run is a new decision
(D40). With --out-bars the script also writes a sparkflash-d41-bars/1 file whose plain bars object
scripts/compare_native_vs_vllm.py --bars consumes directly.

## Refusals (exit 2, nothing written)

- schema is not sparkflash-d41-capture/1; producer.script_sha256 missing; timestamps invalid;
- served.id missing; root_verified false; served root or expect_root missing; served.root does not
  match expect_root (trailing slash stripped, exact or suffix match); captures disagree on served
  root, prompt suite or expect_root;
- recipe.prompt_logprobs < 1000 (top-1000 would be unmeasurable);
- a block is structurally invalid: unknown class, duplicate ids, missing/unknown pair id, pair
  shares no position, idle_distractor pair with the same shape, block prompt_len not uniform when
  it is an int, malformed record/position/top list, duplicate token id, non-finite logprob,
  position + 2 > prompt_len, fewer than the requested entries for a top-k check;
- idleness proof missing, unproven, metrics_source not exactly /metrics, an idle block with a
  non-zero before/after baseline or a co-tenant metrics_during snapshot, idle
  observed_other_traffic/external_peers inconsistency, busy verdict without co-tenancy evidence;
- a class tag disagrees with its verdict: idle_distractor/repeat captured busy, or busy_distractor
  captured idle;
- a required class is missing: the error names the class or classes and the captured classes;
- pair or position comparisons below --min-pairs (default 3) or
  --min-position-comparisons (default 12), enforced by recount and never padded;
- repeat is not bit-identical;
- k is below 20, or the k set is not exactly 20/100/1000 (top-5 dropped);
- the bar class has a rank-1 flip above the flat-position threshold.

## Output and provenance

JSON output schema sparkflash-d41-floors/1 contains: capture paths and sha256, capture ids and
block counts; this script's own sha256; prompt-suite sha256 and path; served root and expect_root;
optional source revision; tool/argv; parameters; capture_validation (per block: class, idleness
verdict, before/after/during counters, records, pairs); classes (every pair summary, per-metric
values, floor, outliers); bars (provisional, floor, rule, value, review reason) and dropped_bars.
The markdown is generated from the same report, with the same hashes. Both are written atomically
after validation; on refusal no output path is touched. Files are written only on success, so a
consumer cannot mistake a stale file for this run's output.

## Tests

scripts/tests/test_d41_floors.py, 36 unittest cases, synthetic captures only (no server, GPU, node
or network). Run:

    python3 -m unittest discover -s scripts/tests -v

Covered: repeat zero accepted; repeat non-identical refused; repeat checked per shared position;
response_sha256 differing across a repeat not compared; missing class named by class, with and
without --require-classes narrowing; too few pairs and too few position comparisons refused, and
--min-pairs raised; one outlier does not move the median bar and removing it leaves the floor and
bar unchanged; k < 20 refused; k set must be 20/100/1000; capture without 1000 entries refused;
idle non-zero after baseline, observed traffic and unproven verdict refused; co-tenant traffic in
an idle metrics_during snapshot refused while the block's own request (running=1) is allowed; busy
without co-tenancy refused; class/verdict mismatches refused in both directions; served root vs
expect_root and served.id refusals; metrics_source not /metrics refused; schema/root/pair/
suite-mismatch refusals; block prompt_len null accepted for distractors; extra fields tolerated;
block batch_size above one accepted as metadata; JSON/markdown/bar provenance hashes, dropped
top-5, and requires_review exit 3.

## Limits

- No real capture has run, so no D41 value is claimed here. The existing window-2026-10-18 numbers
  are not the floor: d41-floor.json compares two different prompt shapes (139 vs 140 tokens) and
  is explicitly a shape-sensitivity upper bound; d41-floor-warm.json is the bit-identical zero
  control; the busy table is not the gate floor. Those records carry precomputed metrics, not raw
  distributions, so they cannot be re-derived or fed to this script as a floor: the gate input is
  the new idle repeat/distractor capture in the schema above.
- The capture's top band is capped at prompt_logprobs (1000); bars on the full 129,280-token
  vocabulary are not measurable from this input and are not emitted.
- If window policy forbids the controlled background load, the run can use
  --require-classes repeat,idle_distractor. The omission is recorded in the output parameters and
  the D41 record must state that the busy load-state envelope was not captured.
- D41 also fixes API/performance bars; those are not computed here.
