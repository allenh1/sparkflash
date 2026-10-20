# TP=4 token-per-second iteration: fused coded-weight decode

Status: measured on the four-rank host decode path (task 16). This page records the
token-per-second iteration: the fused coded-weight kernels and their bit contract, the MoE
decode guard change, the Engram W8A8 projection, the head/embedding stored-BF16 path, the
bench-mode harness, and the full four-run trajectory from the bench baseline to the first
sub-second step.

All numbers below are from the `tp4-host` stage with the bench harness, not from a serving
endpoint: `--bench-steps 10`, `--resident-cache`, `--profile`,
`SPARKFLASH_REFERENCE_THREADS=20`, aarch64 release binaries, four ranks (world 4) on the
same checkpoint, prompt tokens `11111,201,8197,14,270,3417,344,28`, prefill 8 tokens,
decode positions 8..17. The metric is decode-only tokens per second; prefill and load are
excluded by construction (see §9) and are recorded separately.

## 1. Result

**RUN 4b: 939.7 ms mean decode step (936.5 ms median), 1.0642 tokens/s** — rank 0 of four;
the other ranks are 939.5-939.7 ms and 1.0642-1.0644 tokens/s. That is **-84.2% against the
bench baseline of 5,958.4 ms/step** (0.168 -> 1.064 tokens/s, +533%), with the same generated
tokens on every rank and the same plan hash in every run (`0xcd8adc09d2c71d55`).

## 2. End-to-end endpoint census (serve, not the bench)

The same RUN 4b tree also ran on the serving endpoint with the 4b `tp4-serve` binary (serve-dev
census): a capital prompt with `max_tokens=64` returned **TTFT 39.1 s, decode 0.9376 tokens/s
(1.0666 s/step), wall 106.3 s**; two 8-token repeats measured 0.97-1.00 tokens/s with
byte-identical text. This is the end-to-end datapoint: it includes load, prefill and HTTP, so it is
**not the same metric** as the bench's decode-only 1.0642 tokens/s and must not be compared with
it directly. It does show the serving decode path holding the benchmark's rate (0.94 vs 1.06
tokens/s).

**Known limitation: TTFT 39.13 s on a short chat prompt.** The serve path already gates HTTP
after every worker's post-fill Hello (rank 0: fill -> control hub waits for all Hellos -> ready ->
HTTP bind; serve.rs:2911-2944 worker order 3080-3096, driver 2751-2770, bind 3409-3417), so fill
skew cannot leak into first-request TTFT and no readiness barrier is needed. The census TTFT
39.134645546 s is 37 prompt tokens x one decode-shaped forward per token in the tp4_host.rs
prefill loop at ~0.945 s/token = ~35-37 s, plus ~2 s HTTP/tokenizer overhead; the reported prefill
tok/s 0.945 equals decode 0.938. (The 92 s first-step wait seen in a bench run is a bench-harness
artifact - bench starts the forward as soon as a rank's own fill finishes, with no barrier - not
the serve path.) The named fix is batch 17: wire the TP=4 host runner to the existing engine
chunk-prefill API (InferenceEngine::prefill(&self, state, tokens), forward/mod.rs:275, public via
lib.rs:40) instead of looping one decode-shaped step per prompt token; the remaining work is
batch>1 correctness in parallel_tp (MoE batched F4 choice-groups plus a chunk-vs-eight-one-token
bitwise test on a fake world-4 adapter).

## 3. Trajectory (four runs on the same harness)

Each run is a linear stack on main `0174ed0`; "delta" is against the previous run's rank-0
mean step (the canonical per-run numbers are in `phase2-evidence/task16-bench/evidence-bench-*`).

| run | tree (batch commits) | mean step ms | median ms | tokens/s | delta vs prev | delta vs baseline |
| --- | --- | --: | --: | --: | --: | --: |
| RUN 1 bench baseline | a9f8e07 (bench harness only) | 5,958.4 | 5,953.0 | 0.1678 | — | — |
| RUN 2 fused attention weights | e2b24da + 17246ad + b346662 + 2abe3ed | 4,463.9 | 4,450.5 | 0.2240 | -1,494.5 ms (-25.1%) | -25.1% |
| RUN 3 grouped wo_a + head-parallel sparse attention | 67f97b9 + 6810ec1 | 4,328.3 | 4,320.0 | 0.2310 | -135.6 ms (-3.0%) | -27.3% |
| RUN 4a fused MoE (F4 routed + F8 shared, serial expert loop, batched F4, BF16 router) | c8e409e + ca947b2 | 2,320.2 | 2,303.0 | 0.4310 | -2,008.1 ms (-46.4%) | -61.0% |
| RUN 4b head/embed stored BF16 + Engram W8A8 | f18d139 + 4256616 | **939.7** | **936.5** | **1.0642** | -1,380.5 ms (-59.5%) | **-84.2%** |

All five runs: every rank's `all_ranks_rc` 0, `tokens_identical_all_ranks`
`[344, 436, 3338, 304, 611, 260, 201, 8197, 14, 342, 1970]`, `plan_hash`
`0xcd8adc09d2c71d55`, `reduces_total` 720, teardown verified (0 processes, 0 listeners on
29601-29604/29701-29704).

### RUN 5 (dormant change, no attributable effect)

RUN 5 was measured at 921.2 ms / 1.0855 tokens/s on the 4b tree plus engine commit d18d640
(async routed all-reduce overlap with the shared expert) and its docs-only fix 2fb500f, but
d18d640 is **dormant on the tp4-host runner**: tp4_host.rs:802 constructs InferenceEngine::new
with tp: None, the runner uses run_hybrid_sequence_step with the CLI TpMoeBackend
(tp4_host.rs:1712/1749), and the engine block rejects tp+device backend (forward/mod.rs:839-845),
so d18d640 executes nowhere. The 18.4 ms delta is not attributable: rank 0's RUN 4b reduce
outlier (110.1 ms) reverted to the cluster norm (~49.8-60.9 ms) and provider likewise
(64.0 -> 98.8-100.5 ms), i.e. cross-run skew inside normally-tight per-rank numbers. The 92 s
first-step collective wait that motivated the change is a bench-harness artifact (each rank
starts its forward as soon as its own fill finishes; no barrier), not the serve path. Batch 16
therefore ships the 4b tree and does not book RUN 5 as a win; the headline stays
939.7 ms / 1.0642 tokens/s.


Caveats on this table:

- **7.929 s/step is a non-bench reference.** The batch-15 allocator document's 7.929 s/step was
  measured end-to-end on the older path; it is not the same metric as the bench harness's
  decode-only number. The comparable baseline is RUN 1's 5,958.4 ms/step.
- **RUN 3's 135.6 ms was 64% of projection.** Kernel-dev projected ~150 ms (sparse attention
  body) + ~64 ms (grouped wo_a) = ~214 ms; the measured total improvement was 135.6 ms. Of that,
  -105.8 ms landed in `unattributed` while `ffn_local` stayed flat — the per-class timer moved
  the gain into the engine-internal remainder rather than the named layer components.
- **Rows are whole-run measurements; ~100 ms deltas at multi-second scale are not reliable
  per-kernel attribution.** RUN 3's -135.6 ms whole-run improvement against kernel-dev's 33 ms
  causal split is the example: single-run four-rank step deltas of order 100 ms cannot be
  attributed per kernel. The trajectory rows are whole-run harness measurements; the causal
  splits live in the kernel-level microbenches and bitwise tests.
- **`unattributed_us` is not a per-layer quantity.** It is `step wall - measured layer
  components` (the whole-step remainder), so it must not be divided by layer count or read as a
  per-layer cost. RUN 3's rank-0 `unattributed` was 1,311.5 ms per decode step, RUN 4a's
  1,342.7 ms, RUN 4b's 390.9 ms.
- **RUN 4b binary provenance.** The first RUN 4b build silently reused the RUN 4a binary through
  the shared target dir (identical sha `d8350add`); it was caught before staging and rebuilt from
  RUN 4b's own target dir (sha `85aa078a`). The numbers above are the rebuilt binary's.
- The RUN 1 baseline prefill took 69.1 s at rank 0 (47.6-91.3 s across ranks) for 8 tokens —
  slower than decode, consistent with first-touch/page-cache fill on the resident checkpoint. It
  is excluded from the decode-only metric and is not hidden.

## 4. Per-class breakdown (rank 0, ms per decode step)

| class | RUN 2 | RUN 3 | RUN 4a | RUN 4b |
| --- | --: | --: | --: | --: |
| wall | 4,464.3 | 4,328.8 | 2,320.8 | 940.2 |
| ffn_local | 1,721.8 | 1,731.5 | 244.7 | 237.5 |
| shared | 526.5 | 517.4 | 100.0 | 101.1 |
| provider | 550.1 | 532.8 | 525.9 | 64.0 |
| deferred | 548.1 | 530.6 | 523.8 | 63.0 |
| ffn_reduce | 208.8 | 197.6 | 70.6 | 110.1 |
| post | 39.7 | 38.0 | 36.8 | 36.7 |
| engine_host | 3.0 | 2.8 | 0.8 | 0.8 |
| collective | — | — | 53.4 | 92.2 |
| unattributed | 1,417.3 | 1,311.5 | 1,342.7 | 390.9 |

RUN 4a's win is visible in `ffn_local` (1,731.5 -> 244.7 ms) and `shared` (517.4 -> 100.0 ms).
RUN 4b's win is mostly in `provider`/[deferred] (525.9/523.8 -> 64.0/63.0 ms) plus the
`unattributed` drop (1,342.7 -> 390.9 ms), which is where the removed per-step head/embedding
materialisation lived.

## 5. Fused coded-weight kernels: the bit contract

All decode matmuls that previously decoded stored weights into an f32 matrix now read the stored
codes and E8M0 scales directly (`crates/sparkflash-engine/src/linalg.rs`,
`forward/weight_view.rs`):

- F8-E4M3 and F4-E2M1 decode uses the same 256-entry L1 table built from the same quant
  conversions; BF16 is read as raw BF16 bytes.
- The inner loop walks columns in **ascending order**, with a separate multiply and add per
  column, and keeps **four independent output-row accumulators** that share the `x[col]`
  broadcast. Therefore every output row is computed with the exact per-row arithmetic of the
  materialised path: no f32 weight `Vec`, no decode+scale pass, no per-layer allocation.
- The f32 materialised path remains the fixtures' fallback (fixture sources do not serve coded
  tensors), which is why every fixture stays byte-identical.
- MoE: routed experts are F4-E2M1 / E8M0 PerRow32, the shared expert F8-E4M3 / E8M0 Block32.
  The router gate is scored as a plain BF16 view by `matmul_bf16`; the bias-only selection and
  score functions are unchanged.
- wo_a runs as one block-diagonal `matmul_f8_grouped` call per query (layout
  `[groups][o_lora_rank, group_in]`, group selected per output row) instead of eight serial
  calls; a materialised fallback remains for grids whose row blocks cannot be sliced.
- Routed experts take one batched `matmul_f4_experts` call for the selected experts
  (gate/up) and one for the down projections, instead of six separate calls.

Bitwise evidence (each test compares `f32::to_bits`, 0 mismatches):

| test | scope |
| --- | --- |
| `linalg::tests::fused_f8_scaled_is_bit_identical_to_decode_scale_and_matmul` | F8 Block32 |
| `linalg::tests::fused_f8_scaled_per_row_grid_is_bit_identical_too` | F8 PerRow32 |
| `linalg::tests::fused_f4_scaled_is_bit_identical_even_when_a_row_ends_mid_byte` | F4 packed, partial last byte |
| `linalg::tests::fused_bf16_is_bit_identical_to_decode_and_matmul` | BF16 |
| `linalg::tests::fused_bf16_scaled_is_bit_identical_to_decode_scale_and_matmul` | BF16 + scales |
| `linalg::tests::fused_expert_batch_matches_separate_f4_calls_bitwise` | batched F4 experts vs six calls (FNV) |
| `linalg::tests::fused_grouped_f8_matches_the_group_loop_bitwise` | grouped wo_a vs group loop (FNV) |
| `linalg::tests::fused_w8a8_engram_wkv_matches_the_scalar_tree_bitwise` | Engram W8A8 twin |
| `forward::weight_view::tests::fused_views_are_bit_identical_to_the_materialised_path` | all wired attention/head views, threads 1 and 20, released dense/expert/head shapes (8192x4096, 5120x8192, 32768x1280, 1280x5120, 512x5120, 64x128, 2304x2560 F4, 5120x1152 F4, 1024x5120 BF16) |
| `forward::weight_view::tests::grouped_f8_matmul_is_bit_identical_to_the_per_group_calls` | released wo_a geometry 8192x4096 / 8 groups |
| `forward::attention::tests::head_parallel_sparse_attention_is_bit_identical_to_one_thread` | 1 vs 2/4/20 threads |
| `forward::attention::tests::chunked_projection_is_bit_identical_to_one_projection` | chunked vs one projection |

Microbench table (kernel-level, from the kernel commits; not part of the four-rank run):

| kernel | shape | before | after | speedup |
| --- | --- | --: | --: | --: |
| `matmul_f4_experts` (gate/up, 6 experts) | 2304x5120 | 6.40 ms | 3.79 ms | 1.69x |
| `matmul_f4_experts` (down, 6 experts) | 5120x2304 | 6.69 ms | 3.75 ms | 1.78x |
| `matmul_f8_w8a8` (Engram wkv) | released projection | 131.7 ms serial | 5.8 ms at 20 threads | 22.5x |

The single-call small gains of `matmul_f8_scaled`/`matmul_f8_grouped` are visible in the
per-class timers (§4) rather than in a standalone microbench; the FNV tests above are what make
the speedups safe to claim as bit-identical.

## 6. MoE decode guard change (workers=1 + batched F4 experts)

When the routed expert weights are fused, the decode expert loop runs **serial (one worker)** so
every fused matmul uses the whole reference pool instead of being made serial inside an expert
worker by `enter_worker`. A single-token chunk then takes the batched `matmul_f4_experts` path:
one call for the selected experts' gate/up branches (the token repeated per expert) and one for
the down projections over the swiglu-weighted rows. The combine still adds contributions in the
reference's expert-ascending order, and per-row arithmetic is the kernel's, bit-identical to the
per-expert call. An f32 source (fixtures) keeps the old decode and whole-table path, so fixtures
stay byte-identical.

## 7. Engram W8A8

The released wkv projection now runs `linalg::matmul_f8_w8a8` over the whole output width
instead of a serial row loop: the per-row arithmetic is the same 32-lane shuffle tree, ascending
blocks, activation/weight E8M0 scales and bf16 row cast, parallelised only across independent
output rows. The scalar copy stays as a `cfg(test)` reference and the twin test asserts the
kernel's bytes equal it row for row (FNV equal; 131.7 ms -> 5.8 ms at 20 threads).

## 8. Head and embedding on the stored BF16 bytes

The head is read through the dtype-aware WeightView (plain BF16 on a release source,
`matmul_bf16`; f32 fixtures keep decode + matmul) at all three call sites, including the observed
and TP host fallbacks — no f32 1.3 GB head table per step. `embedding_rows` widens only the rows
the chunk uses from the stored BF16 table (exact) instead of decoding the whole table every step,
and borrows the resident bytes when the source has `read_view`; an f32 source keeps the old
whole-table decode, so fixtures are byte-identical.

## 9. Bench-mode harness semantics

`--bench-steps N` on `--run-stage tp4-host` is the shared timing harness (no arithmetic change):
it opens the rank, calls `begin_request`, prefills the prompt once, then runs N `decode()` calls
with no HTTP server, no token streaming and no reference check.

- World 1 runs with no peers and no transport: `Tp4Rank::open` leaves the transport `None` and
  the canonical reduce is a strict no-op (`reduce_calls` stays 0,
  `world_one_reduce_is_a_strict_bit_no_op`). World > 1 runs the same number of decode steps on
  every rank and records the real per-layer collectives.
- The evidence JSON gains a top-level `bench` object: `steps_requested`, `steps_run`,
  `prefill_tokens`, `prefill_ms`, `prefill_token`, `decode_ms`, `mean_step_ms`,
  `median_step_ms`, `tokens_per_second` (decode-only = `steps_run * 1000 / decode_ms`), the
  per-step wall/rchar/read/collective arrays, `profile_on` and `measure` (the measurement rule).
  The summary gets the same object; stdout prints one bench line.
- **The bench number excludes load and prefill by construction**: `tokens_per_second` divides the
  decode steps only by `decode_ms`; `prefill_ms` and the load/cache header are separate fields,
  and `bench.measure` states the rule so the number is never compared with an end-to-end
  tokens/s. **Per-rank**: every rank writes its own `tp4-host-r<rank>.json` with its own bench
  object; the summary is rank 0's.
- With `--profile`, `profile.steps` holds one entry per forward step in order: the
  `prefill_tokens` prefill entries first, then the `steps_run` decode entries; the bench object
  also records `profile_entries` and `profile_decode_first_index` (= `prefill_tokens`), so
  per-class decode averages are the sum over the last `steps_run` entries divided by
  `steps_run`.
- In bench mode the legacy summary line no longer prints the end-to-end tokens/s (it includes load
  and prefill); non-bench runs keep the old line unchanged.

Reproducing one run (four ranks; the harness is also runnable in world 1 on the dev host with a
fixture):

~~~sh
SPARKFLASH_REFERENCE_THREADS=20 ./sparkflash device-chain --run-stage tp4-host \
  --tp-rank R --tp-world-size 4 \
  --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
  --comm-base-port 29601 --checkpoint <snapshot> \
  --prompt-tokens 11111,201,8197,14,270,3417,344,28 \
  --bench-steps 10 --resident-cache --profile --evidence-dir <dir>
~~~

## 10. Boundaries

- This is a host decode measurement, not a serving endpoint: no HTTP, no tokenizer, no streaming,
  no reference check. Serving TTFT and end-to-end tokens/s are separate metrics.
- The decode-only number deliberately excludes prefill (8 tokens took 47.6-99.4 s per rank across
  the runs, dominated by first-touch/page-cache effects); it is not a latency claim for a request.
- The four-rank numbers are from the same checkpoint, prompt tokens and plan hash
  (`0xcd8adc09d2c71d55`), with identical generated tokens on every rank in every run.
- Kernel-level microbenches are single-call measurements on the dev host; the safe claim is the
  bit-identity tests above plus the per-class timers from the four-rank runs.
- Predictions vs measured: RUN 3 measured 64% of its component projection; RUN 4a beat its
  predictions (ffn_local ~0.5 s predicted, 0.245 measured; step 2.6-3.0 s predicted, 2.32
  measured); RUN 4b beat them too (provider 120-200 ms predicted, 64.0 measured; step 1.5-2.0 s
  predicted, 0.94 measured). RUN 5's prediction is not booked: the change it measured (d18d640)
  is dormant on this runner, so its 18.4 ms sits inside cross-run skew (see section 3).
