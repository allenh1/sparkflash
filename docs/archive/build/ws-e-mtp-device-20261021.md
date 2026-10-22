# WS-E: on-device DSpark MTP drafting - plan, policy and measurement harness

**Status: 2026-10-21. Plan plus implemented-and-tested policy and harness; NO device execution.**
Nothing on this page measures a device, and no 90 tokens-per-second claim is made. The only measured
acceptance number used is the CPU host-twin result in section 1; the gate arithmetic built on it is
explicitly an estimate of what a device would have to reach. The device plan in sections 3-5 is
design work for the device window, not a result.

## 1. The gate and the only measured number in it

WS-E's serving gate: accepted tokens per verify step, per second, must reach 90 tok/s.

Measured input (host f32, CPU host twin, no device): **1.571 accepted tokens per verify step** from
24 accepted drafts in 42 verify steps, 95% CI 1.333-1.810, p50 1.000, p90 3.000, natural-language
subset 1.710. Evidence: /home/user/dspark-accept-evidence/ (schema
sparkflash-dspark-acceptance/1, device_execution false, binary e3d074b7...), summarised in
STATUS-2026-10-21.md (deleted; see archive/README.md). This is a host-twin measurement: it says what the
draft head accepts on the reference arithmetic, not what a device achieves.

The arithmetic the gate implies (all derived, none measured on device):

| quantity | value | derivation |
| --- | --- | --- |
| accepted tokens per verify step (AL) | 1.571 | 1 + 24/42, measured host twin |
| verify-step time needed for 90 tok/s at that AL | **17.46 ms** | 1.571 x 1000 / 90 |
| plain decode step time that is exactly 90 tok/s | 11.11 ms | 1000 / 90 |
| speedup over plain decode the gate requires | 1.57x the decode step at AL 1.571 | 17.46 / 11.11 |

So at the measured acceptance length the verify step may cost about 57% more than a plain decode step
and still hit 90 tok/s. Accepting more tokens per step is the other lever; the harness in section 4
exists to measure which corpus gives it. The harness reports the plain decode rate alongside the
speculative rate, because a speculative rate without its decode baseline is not evidence.

## 2. What landed and is tested

All of this is host arithmetic over samples a caller supplies; it never runs a forward.

| item | state | location |
| --- | --- | --- |
| draft length k = the spec block size (5), and the graph bucket list [1, 2, 4, 8, 16] | implemented, tested | [MTP_DRAFT_K](../../../crates/sparkflash-scheduler/src/dspark.rs#L485), [DRAFT_BATCH_BUCKETS](../../../crates/sparkflash-scheduler/src/dspark.rs#L493) |
| gate target (90 tok/s) and the measured host-twin AL constants | implemented, tested | [TARGET_TOKENS_PER_SECOND](../../../crates/sparkflash-scheduler/src/dspark.rs#L496), [MEASURED_HOST_TWIN_AL](../../../crates/sparkflash-scheduler/src/dspark.rs#L510) |
| required verify-step time for an AL and target | implemented, tested | [required_verify_step_ms](../../../crates/sparkflash-scheduler/src/dspark.rs#L680) |
| per-corpus acceptance samples (chat / code / natural language), validated | implemented, tested | [AlCorpus](../../../crates/sparkflash-scheduler/src/dspark.rs#L526), [AlSample](../../../crates/sparkflash-scheduler/src/dspark.rs#L552) |
| per-corpus aggregate AL, acceptance rate, tokens/s, required step time, pass/fail | implemented, tested | [AlMeasurement](../../../crates/sparkflash-scheduler/src/dspark.rs#L601), [gate](../../../crates/sparkflash-scheduler/src/dspark.rs#L658) |
| measurement harness with plain-decode baseline reported alongside | implemented, tested | [AlHarness](../../../crates/sparkflash-scheduler/src/dspark.rs#L742), [gate_reports](../../../crates/sparkflash-scheduler/src/dspark.rs#L818) |
| gate report carries device_execution = false so it cannot be quoted as a device run | implemented, tested | [AlGateReport](../../../crates/sparkflash-scheduler/src/dspark.rs#L721) |
| adaptive draft length from a rolling acceptance window | implemented, tested; heuristic, unvalidated | [DraftLengthPolicy](../../../crates/sparkflash-scheduler/src/dspark.rs#L837), [observe](../../../crates/sparkflash-scheduler/src/dspark.rs#L921) |
| one batched draft forward plan: k drafts per request, batches in graph buckets, padded rows | implemented, tested | [DraftBatchPlan](../../../crates/sparkflash-scheduler/src/dspark.rs#L982), [plan](../../../crates/sparkflash-scheduler/src/dspark.rs#L997) |
| the actual device draft forward, verification forward and KV commit | **design only** | sections 3-5 |

Tests that pin the above: the host-twin AL equals the measured ratio; the gate is accepted tokens per
second and its required step time is exact; samples are validated (zero drafts, accepted above
drafted, non-finite or non-positive times refused); the harness reports each corpus separately and
aggregates; the plain decode baseline is reported alongside; the adaptive policy backs off below the
low watermark and recovers above the high one, clamped to 1..=5; one forward carries every request's
drafts in a bucket and 17 requests need two forwards rather than 85; the buckets end at the
CUDA-graph ceiling.

## 3. The device plan (design only)

**Step shape.** One verify step is two forwards.

1. **Draft forward**: for each running request with draft length k_r, run the MTP block over the k_r
   draft positions in one forward per batch, not one forward per draft token. The batch dimension is
   the number of requests; each request's rows are its k_r positions. [DraftBatchPlan](../../../crates/sparkflash-scheduler/src/dspark.rs#L982)
   groups requests into the existing CUDA-graph buckets [1, 2, 4, 8, 16] and pads each batch to
   bucket x max k, so a step with 4 requests at k = 5 is one graph replay with 20 rows.
2. **Verification forward**: the target model runs the anchor plus the drafted positions for every
   request, exactly the rows the dspark confidence policy already defines (anchor + verified prefix).
   The confidence cap decides how many drafted rows are verified; the mask decides which rows reuse
   the anchor's experts. Accepted tokens are appended to the sequence and its KV; rejected tokens are
   dropped. This is one target forward per verify step for the batch, not one per request.

**KV and sequence bookkeeping.** The accepted prefix is the only thing written: the KV of rejected
draft rows must be discarded or overwritten by the next step (the paged cache's slot mapping is the
authority; the scheduler-side draft policy does not model slots). The draft forward reads the same
prefix the verify forward will, so the prefill/prefix-cache contract from
[ws-f-scheduler-20261021.md](ws-f-scheduler-20261021.md) applies unchanged.

**Adaptive draft length.** Start at k = 5. Per verify step record drafted and accepted counts; every
full window of 8 steps, take the mean acceptance. At or above 0.75 grow k by one; below 0.25 shrink
it by one; between, hold; clamp to 1..=5. The rationale is that a low acceptance makes the extra
draft rows pure cost while a high acceptance keeps them nearly free in one graph replay. The
watermarks are a heuristic, not a measured optimum: the harness must A/B fixed k = 5 against the
policy before any claim is made (acceptance criterion in section 5).

**Graph capture.** The buckets exist so the draft forward reuses the serving path's captured graphs.
Variable k changes the row count, which a captured graph cannot absorb per replay, so the plan pads
to the bucket's row shape and writes only the real rows. Whether padding waste is acceptable at
small batches is exactly what the device measurement has to answer; this page does not assume it.

## 4. The measurement harness protocol

The harness is implemented; the protocol it must be run under on device is:

* **Corpora**: chat, code and natural-language prompts, reported separately. The host twin already
  shows the natural-language subset at 1.710 against the 1.571 pooled AL; pooling hides that.
* **Per verify step**: drafted count, accepted count, verify-step wall time, and the corpus. That is
  one [AlSample](../../../crates/sparkflash-scheduler/src/dspark.rs#L552); do not average before storing.
* **Plain decode baseline**: measure decode-only step time on the same device and corpus set, and
  report its tokens/s next to the MTP tokens/s. [PlainDecodeBaseline](../../../crates/sparkflash-scheduler/src/dspark.rs#L693)
  exists so the report cannot omit it.
* **Sizing**: enough verify steps per corpus for the acceptance rate to be stable; the host twin's 42
  verify steps overall is the starting floor to beat, not a target. Report AL, acceptance rate, mean
  verify-step time, tokens/s and the required step time for 90 tok/s, all per corpus.
* **Evidence**: serialise per-step samples plus the binary hash and configuration to an artefact with
  device_execution true (the host-twin artefact says false). A gate report produced by
  [gate_reports](../../../crates/sparkflash-scheduler/src/dspark.rs#L818) is arithmetic over those
  samples; the artefact is what makes it checkable.

## 5. Acceptance criteria for the device window

The device window is done when, for every corpus:

1. the per-corpus gate report passes at 90 tok/s (AL x 1000 / mean verify-step ms >= 90), and the
   plain decode rate is in the same artefact;
2. the draft forward makes one launch per batch and the verify forward one per step - evidenced by a
   launch counter, not by a plan field;
3. accepted tokens are committed to KV exactly once and rejected rows are not visible to the next
   step (a logits/KV comparison against a no-draft run at the accepted positions);
4. the adaptive policy is either demonstrably better than fixed k = 5 or is left off with the
   measurement that justifies it;
5. the non-draft path (plain decode) does not regress against its pre-WS-E census.

If criterion 1 cannot be met at the measured acceptance length, the honest outcome is the measured
shortfall in verify-step milliseconds, not a relaxed gate: at AL 1.571 the gate needs 17.46 ms per
verify step.

## 6. What is not done, and the dependencies

* No device forward, no graph replay, no launch counter: sections 3-5 are design.
* The adaptive draft-length policy is unvalidated; it is a heuristic with no device evidence.
* Bucket padding cost, graph capture of variable k, and the draft forward's actual latency are
  unknown.
* The host twin's AL (1.571) may not predict device acceptance; the harness measures it, it does not
  inherit it.
* Dependencies: a GPU node (this host has none), the device draft/verify kernels, and the paged-KV
  slot mapping from the device window. No kernel pin, ABI pin or device binary was touched by this
  work.

## 7. Provenance

* Measured: the host-twin AL row in section 1, evidence /home/user/dspark-accept-evidence/ and
  STATUS-2026-10-21.md (deleted; see archive/README.md).
* Derived arithmetic: every millisecond and tokens-per-second figure in this page comes from the gate
  formula and the measured AL; none of it is a device measurement.
* Scheduler records this plan builds on: [ws-f-scheduler-20261021.md](ws-f-scheduler-20261021.md),
  [s4-batch-chunked-prefill.md](s4-batch-chunked-prefill.md); the dspark policy itself is in
  [docs/architecture/05-dspark.md](../../architecture/05-dspark.md).
* Branch task/phase1-async-device; no commits made by this work; the live four-node vLLM TP=4
  endpoint was not touched.
