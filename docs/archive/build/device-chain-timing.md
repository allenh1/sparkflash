# Device-chain timing: the staging-vs-kernel lines from one window

Status: host-side instrumentation only (task-50). No device numbers are recorded here and no
throughput claim is made. This is the mechanism that lets a future window record the lines D42
section 6 asks for without putting a profiler on the node.

## Why

D42's measured prerequisite says no throughput argument is possible until one window records, per
source layer and position, the staging cost separated from the kernel cost. The chain already
prints per-position totals (staging_ms, attention_us, device_us, host_bytes), but those totals mix
the sources and none of them is machine-readable. This flag turns the same run into evidence: the
attention body reports what it actually staged and copied, and the chain reports what each step
cost in wall time.

## How to run

~~~
sparkflash device-chain --run-stage hybrid_sequence \
  --checkpoint DIR --cubin-dir DIR \
  --fixture FIXTURE --prompt-tokens N --decode-steps N \
  --timing-json /path/to/chain-timing.json
~~~

The flag is opt-in and only wired into the hybrid_sequence arm. Without it, no recorder is
constructed, no counter is touched, no extra line is printed and no file is written: the default
run's output and evidence are byte-identical. dry-run refuses it, because a dry run stages nothing
to measure. The run prints one summary line when the document is written and fails the run if the
document cannot be written.

## The document

Schema tag: sparkflash-chain-timing/1. One JSON object, written at the end of the run.

~~~json
{
  "schema": "sparkflash-chain-timing/1",
  "samples": [ { "position": 128, "layer": 0, "window_rows": 128, "window_bytes": 67584,
                 "absolute_rows": 64, "absolute_bytes": 18432, "indexer_pool_bytes": 0,
                 "host_staging_ns": 41230, "h2d_calls": 7, "h2d_bytes": 86432, "launches": 1,
                 "sync_gated_device_ns": 198220 } ],
  "steps": [ { "position": 128, "total_ns": 903112 } ],
  "totals": { "layer_samples": 1, "window_rows": 128, "window_bytes": 67584, "...": 0 }
}
~~~

samples is one entry per sequence attention body call (position, layer). steps is one entry per
sequence step, recorded by the chain around the engine step call. totals are measured sums over
those entries; they are not model estimates.

## Measured versus inferred

Measured directly, from the buffers the body uploads and the wall clock around the calls:

- window_rows, window_bytes: the staged window rows and their codes plus block scales
  (528 B per row: 512 B codes + 16 B scales).
- absolute_rows: the absolute main-KV candidate rows quantised from the host KV.
- absolute_bytes: the pool bytes actually uploaded. This includes the zero-filled ratio-driven
  rows the kernel indexes over, so it can exceed absolute_rows * 288; the row formulas are
  exported and host-tested by chain_timing.
- indexer_pool_bytes: the HSI index key-pool bytes staged by this body. It is zero for the current
  sequence path, which zeroes that pool once at open and does not rebuild it per step. The field
  exists so a paged/HSI path can fill it; a zero here means "not staged per step", not "free".
- host_staging_ns: wall time from the start of window-row/candidate packing to the end of
  activation byte packing, before the first H2D copy. It includes row construction, FP4
  quantisation and BF16 packing, so it is a host-staging cost, not a pure dequantisation cost.
- h2d_calls, h2d_bytes: the copy calls the body issues and the bytes they carry.
- launches: the format-2 attention body launch (projections have their own existing ledger lines).
- sync_gated_device_ns: wall time from the launch call to the end of the readback that already
  synchronises. It is kernel time plus the D2H transfer; it is not a GPU timestamp.
- step_total_ns: total wall time of the engine step, measured by the chain loop.

Inferred and labelled as such: nothing in the document is inferred. Everything outside it is out
of scope here: MoE staging, TTFT/prefill/decode splits, batch timing, launch overhead split from
kernel time, H2D bandwidth, and occupancy are not measured by this flag.

## Host proofs, and what they do not prove

The device body cannot run on a host without a driver, so the host proofs are:

1. Flag off changes nothing: no recorder is constructed, chain_timing's disabled recorder records
   nothing and writes no file, and the two host-only arms (--plan-only, --dry-run) gain no timing
   line. Tests: chain_timing::a_disabled_recorder_records_nothing_and_writes_no_file,
   device_chain::the_default_plan_and_dry_run_print_no_timing_line.
2. The counters account for the bytes the staging code moves: the body builds its sample from the
   lengths of the buffers it uploads and from the shared row-byte helpers, so the formulas and
   their hand-computed values are host-tested (chain_timing::row_byte_formulas_are_the_measured_row_sizes);
   a drift between a staged buffer and its reported bytes would have to be a drift in the buffer
   length itself.
3. A synthetic run totals the hand-computed sum: two synthetic layer samples plus two step samples
   produce exactly the hand-summed totals, and the JSON document carries the schema, samples, steps
   and totals (chain_timing::a_synthetic_run_totals_the_hand_computed_sum,
   chain_timing::the_json_document_carries_the_schema_samples_and_totals).

These prove the accounting and the flag's opt-in behaviour. They do not prove that a real device
run produces sane numbers: only the window can show that, and one run is a diagnostic, not a bar.
