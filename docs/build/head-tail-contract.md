# The single-stream head tail: contract, harness and the batched-reduce drop (task-16, WS-C)

Status: harness m27_head_tail is implemented and has now RUN ON DEVICE. The 2026-10-21 window executed
the device path against the released checkpoint and the seam the four-rank loop wrote
(head-input/pre_head_hidden.f32, sha256 f6990803...): device evidence PASS 34/34, twin PASS 6/6,
device-vs-twin + bundle + seams PASS 70 EQUAL / 0 DIFFER, loop-record vs m27-device join PASS 57/0
(four rank logits shas c2c0b9f7.../a9d2eaa3.../13de9364.../f6824f1f... EQUAL, gathered 901fae27...,
argmax token 52945 value 15.939203262329102, normed 1d87882c...). Accepted harness sha256
f7c2a674... (commit c6ec1d45) after two harness defects were fixed: launch_rank() passed the raw
seam row instead of the post-rmsnorm row, and read_eps() now falls back to the released config's
text_config.rms_norm_eps (1e-20) with a top-level fallback; the defective first run (sha 850040c6...)
is preserved as a FAIL record under the artifact. No kernel source or pin changes: the twenty-entry
baseline is untouched. Artifact:
ev-session-20261020/window/m1y7/kernel-m27-adjudication/ (ADJUDICATION.md, provenance, harness-fix/,
fixed-run/, preserved defect run). Contract handoff to device-lead is complete (section 7).

## 1. Subject: the head is 5 launches, single-stream decode needs 4 and no reduce

The certified single-node chain (device_chain) spends five launches per decode position on the head
(kernel-occupancy-inventory.md section 1, rows 15 and 16; device-forward-chain.md, device_launches
1168, head_launches 5):

- 4 x cert_lm_head_sm121, one rank shard of 32,320 vocabulary entries each, 32 blocks of 128 threads;
- 1 x lm_head_reduce_sm121, the cut-over that assembles the four rank shards' per-block maxima and
  claims the token. The inventory note says the reduce is the 1,168th launch.

The six-row batched variants exist and are measured but are not wired into the chain:

- cert_lm_head_batched_sm121: BATCH_ROWS 6, m22 PASS 6/6, 2.853x for six rows (batch19-plan table);
- lm_head_reduce_batched_sm121: BATCH_ROWS 6, m24 PASS byte-equal rows but 1.01x versus six scalar
  launches and explicitly NO WIN (batch19-plan table; window-kernels window record).

Their row count is fixed at six: the batched reduce cannot be called with one row. For single-stream
decode only the final position's logits are consumed, so the batched reduce is pure overhead there,
and a six-row launch would do five rows of work that are thrown away. The single-stream path is:

  final hidden row -> rmsnorm_sm121 -> 4 x cert_lm_head_sm121 rank shards
                   -> exact f32 logit all-gather (host word movement, no device launch)
                   -> argmax with the lowest-index tie break

The verify path (DSpark, six rows) may keep the batched symbols; it records reduce_mode
batched-six-row instead, and nothing in this contract removes a launch it needs.

## 2. The arithmetic and its exactness table

The tail is the certified head's own arithmetic, not a new kernel:

| stage | device | host twin | bar |
|---|---|---|---|
| final norm | rmsnorm_sm121, BF16 in and out, one row per launch, eps from config.json | rmsnorm_ref::final_norm_bf16 (the engine's rms_norm arithmetic, then the kernel's BF16 RNE) | BF16 output bytes bit-identical (m18 bar: 0 mismatched bytes) |
| head screen | cert_lm_head_sm121 per rank shard: 40 block bounds, exact f32 ascending fma chain for surviving entries, certified entries store the bound | lm_head_ref::exact_logit for every entry and lm_head_ref::hidden_block_sums + entry_bound for the bound | exact entries bit-identical; certified bounds within 1e-3 + 1e-5*|host|; per-thread pruning replay (chunk, lane, ascending step, strict bound < running max) with 0 violations; certificate validity eps 1e-4 + 1e-5*|exact| |
| block reductions | 32 published block_max/block_argmax entries per rank | the same per-block scan of the measured logits, lowest entry on ties | bit-identical value and argmax |
| counts | certified_count readback | the count of certified flags | equal |
| determinism | two launches on the same buffers | - | raw outputs byte-identical |
| gather | none (host word movement) | concatenate the four shard vectors | identical bytes, 0 ULP: the gather performs no arithmetic |
| argmax | host best_entry over the gathered vector, strict greater-than | the twin's argmax over its exact logits | device argmax equals twin argmax, same f32 bits; the winning entry must be EXACT (certified==0). A certified winner is a refusal, not a token: the loop must not claim it. |

There is no approximate row to hide in: the only non-exact values are certified bounds, and they are
compared as bounds against the host formula and the certificate eps, exactly as m16 does on the
control and the concentrated cases. The optional engine reference logits (declared artifact, not
recomputed) are used for an argmax equality only, as in m16.

## 3. Launch inventory and the expected per-position saving

| path | norm | head | reduce | exact gather | total |
|---|---|---|---|---|---|
| certified chain (b68) | - | 4 | 1 | 0 | 5 launches, gated by the scalar reduce |
| single-stream (this contract) | 1 per rank | 4 | 0 | 0 device (1 host gather) | 4 launches |

Expected saving, host-side arithmetic from the recorded m24 p50s, never a new measurement:

- launches per position: 1 (the reduce), 5 -> 4;
- if the six-row batched reduce were used for one position, its recorded p50 is 2.4535 ms;
- the recorded six scalar reduce p50 is 2.4709 ms, i.e. 0.4118 ms per row, so dropping the scalar
  reduce saves about 0.4118 ms/position and dropping the batched one saves about 2.0417 ms/position
  (2.4535 - 0.4118);
- the harness prints these numbers with their source and makes no device speedup claim; only a
  window can measure the new path.

## 4. Harness: argv and expected PASS lines

Build the aarch64 binary for the nodes with scripts/build-aarch64.sh --example=m27_head_tail; the
cubin directory must contain the two pinned cubins (cert_lm_head_sm121 38,088 B / 3b71c992..., and
rmsnorm_sm121 31,480 B / 5f953bc0..., both the twenty-entry baseline pins; the harness refuses a
different size or sha256 before any device allocation).

Twin only (dev host or node; needs the released snapshot and a 5120-f32 seam file):

  m27_head_tail --checkpoint <snapshot dir> --input <bundle dir or f32 file> --twin-only       [--evidence out.json] [--dump dir] [--floor-gib 32]

Expected lines:

  twin: final normed BF16 sha256 ...; argmax ... value ...
  twin rank 0..3: logits sha256 ..., bounds sha256 ...
  head inventory: ... 5 launches ... 4 ... 1 reduce dropped ...
  PASS twin: final norm produced the declared BF16 row - ...
  PASS twin rank r: shard screened - ...
  PASS twin: argmax is a real best entry - ...
  m27_head_tail twin-only: PASS (6 checks; ...)

Device run (window, after m1w14 and m26; vLLM stopped, floor 32 GiB):

  m27_head_tail --checkpoint <snapshot dir> --input <evidence dir> --cubin-dir <cubins>       [--reference-logits <f32 129280>] --evidence out.json --dump dir [--repeat 20]

Expected PASS classes (each is one check per rank where marked):

  stage 1: final norm BF16 bit-exact vs twin - 0 mismatched bytes
  stage 1: two norm launches byte-identical
  stage 2 rank r: exact entries bit-identical
  stage 2 rank r: certified bounds within 1e-3 + 1e-5*|host|
  stage 2 rank r: certificate holds
  stage 2 rank r: pruning replay has no violations
  stage 2 rank r: block reductions agree
  stage 2 rank r: certified_count agrees with the flags
  stage 2 rank r: two head launches byte-identical
  stage 3: exact f32 gather is the per-rank concatenation
  stage 4: argmax is on an exact entry (not a certified bound)
  stage 4: device argmax equals the twin argmax
  stage 4: reference engine argmax agrees            (only when --reference-logits is supplied)
  stage 5: single-stream head drops the batched reduce

The harness prints NOT claimed: no device speedup (recorded arithmetic only), no token claim (the
argmax is a measured equality, and a certified winner is a refusal), and no pin change.

Implementation status: cargo build, clippy -D warnings and cargo test -p sparkflash-kernels
--examples clean (397 passed; m27_head_tail 24 tests including the released-config eps regression).
The host twin ran end-to-end on a synthetic released-geometry snapshot and on the real seam. The
device path has been executed: 34/34 PASS on the released checkpoint, device-vs-twin 70/0, and the
four-rank loop's head record (m1y7, seam f6990803...) joins m27 device evidence 57 fields / 0
differences. The accepted harness binary is f7c2a674... ; the buggy first binary 850040c6... is kept
only as the preserved defect record.

## 5. Evidence contract (field names, digests, counts)

Schema string: sparkflash-head-tail-evidence/1. The device record carries both blocks so a diff is
mechanical; the twin-only record carries the same twin block and no device block.

Top level: schema, mode (device or twin), checkpoint, tokens 1, hidden 5120, vocab 129280, ranks 4,
vocab_per_rank 32320, layer, input, norm, head, gather, argmax, reference, reduce, launches, twin,
timing, checks.

- input: path, sha256 (the 20,480 B f32 seam), bytes, bf16_codes_sha256, bf16_codes_bytes. The loop
  writes the seam as <evidence>/head-input/pre_head_hidden.f32; the harness accepts that directory
  or the file itself. This digest is the join key with the loop's head record.
- norm: kernel rmsnorm_sm121, pin cubin_bytes/cubin_sha256, eps (the f32 cast of config
  rms_norm_eps), normed_bf16_sha256, normed_bytes.
- head: kernel cert_lm_head_sm121, pin cubin_bytes/cubin_sha256, grid [32,1,1], block [128,1,1],
  chunks_per_rank 32, deterministic [bool;4], ranks[4] each {rank, logits_sha256 (32,320 f32),
  logits_f32_values, certified_count, certified_sha256 (32,320 u8 flags), block_max_sha256 (32 f32),
  block_argmax_sha256 (32 u32)}.
- gather: route exact-f32-word-gather, words_per_rank 64,640 (each f32 is two u16 halves on the
  wire), gathered_logits_f32_values 129,280, gathered_logits_sha256 (the authoritative full-vocab
  digest), exact.
- argmax: token, value, certified (bool), claimable (!certified), twin_token, twin_value.
- reference: supplied plus, when supplied, the reference path/sha/argmax (the harness prints them).
- reduce: the inventory block: chain_head_launches_per_position 5, chain_composition,
  single_stream_head_launches_per_position 4, single_stream_composition, reduce_launches_dropped 1,
  reduce_mode dropped-single-stream, batched_reduce {kernel, fixed_rows 6, callable_with_one_row
  false, role}, recorded {the m24 p50s}, expected_per_position_saving {vs_batched_six_row_for_one_row_ms,
  vs_scalar_reduce_for_one_row_ms, launches 1, note}, exact_gather.
- launches: loop_norm 4, loop_head 4, loop_reduce 0, harness_norm 1, harness_head 4,
  harness_reduce 0, gather 1 (the harness runs one norm launch and reuses the identical bytes for the
  four shards; the loop count is what the position pays).
- twin: normed_bf16_sha256, normed_bytes, argmax, argmax_value, input_pre_head_sha256, ranks[4]
  {rank, logits_sha256, bounds_sha256}.
- timing: norm_ms, head_ms_per_rank (one launch each; these are harness wall times, not a throughput
  claim).
- checks: [{name, ok, detail}] exactly the PASS lines, so the JSON is the record of the verdicts.

The loop should record the same names under its head stage (or a head block with these keys) and
write the seam bundle; reference_id is unchanged (the head is not a new reference; it uses the
existing exact-logit/reference-logit contract).

Adjudication recipe for the next window:

1. m27 twin-only first (expected record), then the device run on the same seam;
2. compare the device record against the loop's head record: input seam sha, normed sha, per-rank
   logits sha, per-rank certified count and block digests, gathered sha, argmax and claimable flag,
   reduce.inventory. First divergent family names the stage: norm (K-norm), head exact entries
   (K-head), gather (R-head), argmax/reference (token claim);
3. a certified winner is a refusal, not a mismatch: report it as REFUSED with the record, and the
   loop must not claim a token from it;
4. verify the drop arithmetic in the record against this page (5 -> 4 launches, one reduce dropped)
   and state that no device speedup is claimed from the harness.

## 6. Pins, baseline, and what would change them

No kernel source change is needed or made. The twenty-entry baseline
(crates/sparkflash-kernels/baseline/kernel_baseline.txt) is untouched; both cubins the harness uses
are already baseline entries at the pins above. The harness checks the baseline manifest pin before
reading the cubin and the cubin size/sha256 before any device allocation. If a later change ever
needs a new kernel or a different launch, the pin/baseline process (scripts/regen-kernel-baseline.sh,
scripts/check-kernel-baseline.sh, tests/kernel_baseline.rs) applies and the baseline count would
change explicitly rather than silently.

## 7. Handoff to device-lead

The loop side only has to record the head stage and write the seam file; no index-side change is
required and no batched symbol enters the single-stream path. Ask: keep the field names above, emit
the seam as pre_head_hidden.f32 (5120 little-endian f32, the ffn post-hc row the head consumes), and
record reduce_mode dropped-single-stream with launches {norm 4, head 4, reduce 0}. If the loop's head
record uses a different nesting, send one sample and the adjudicator map is extended; the m27 record
itself is a flat table and does not change.

STATUS: COMPLETE. device-lead's loop head record (m1y7) uses exactly the requested nesting - a
literal nested `head` block with input/norm/head.ranks[4]/gather/argmax/reduce/launches, seam
pre_head_hidden.f32 in head-input/ and head-input-r{1,2,3}/ (4/4 byte-identical f6990803...), and
launches {norm 4, head 4, reduce 0}. kernel-lead's m27 adjudicator maps the nested paths, ignores the
extra layer_chain bookkeeping, and passed the strict join (57 matched / 0 differ / 2 skipped, 960
layer_chain leaves ignored). No follow-up ask remains.
