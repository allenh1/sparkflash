# S4 continuous batching host executor (task-22)

**Status: task-22 host-side, 2026-10-19.** Branch task/s4-continuous-batching, based on
a976ca7 (the accepted task-44 paged-serve harness plus its missing-docs fix). This page covers
the host scheduling/execution slice only. Device batching, TTFT and throughput are named window
measurements in section 6 and no number is claimed.

## 1. What task-40 left and what task-22 adds

Task-40 delivered the HTTP batch contract and chunked prefill policy, but its host driver still ran
one engine sequence at a time. This task adds the missing concurrency shape in
crates/sparkflash-cli/src/serve.rs:

- serve::continuous::ContinuousBatch admits several requests against one reference engine.
- Each admitted request owns its own SequenceState; the engine is stateless per sequence, so
  interleaving engine calls is the host equivalent of a device batch step.
- The per-request prefill chunk plan is the scheduler crate's ChunkedPrefillPlan.
- A step record names which request ran which operation, so a caller can replay the schedule
  against the paged KV mirror and check invariants.

The module is crate-internal (pub(crate)), because the HTTP reference driver still serializes
jobs. It is a host executor and a test seam, not a claimed HTTP concurrency switch.

## 2. Scheduling policy

Admission and execution are the two decisions, and both are deliberately simple and stated:

- Admission is FIFO. Pending requests are ordered by arrival step and, for equal arrivals, by
  offer order. A waiting request is admitted only when a slot under max_batch is free, and a later
  offer can never overtake an earlier one. Requests arriving in the future are held until their
  arrival step; the executor does not spin over empty steps before them.
- Execution is round-robin. Every active request advances exactly once per scheduler step, in
  round-robin order. A prefilling request advances one chunk; a decoding request advances one
  token. A long prefill therefore cannot starve another request's decode, and one long decode
  cannot stop a newly admitted request from starting its prefill.
- Progress is bounded. Each request has a finite token cap, each admitted request advances every
  step it is active, and slots free when a request finishes; a queued request therefore reaches
  admission after finitely many earlier requests finish. The tests assert the FIFO admission order
  and the once-per-step active progress.

## 3. Correctness invariants

The host reference path is the engine's own f32 path; the comparisons are bit-level.

1. Alone versus interleaved. For each request, tokens, every generated logits row (compared by
   f32 bits) and the argmax of the first token are identical whether it runs alone
   (max_batch = 1) or interleaved with others.
2. Chunk boundaries and decode windows. The same outputs hold for every tested
   chunk_tokens (1, 2, 3, 4, 8) and max_batch (1, 2, 3). The chunk plan is the scheduler's;
   the test also asserts that changing the chunk size actually changes the schedule, so this is not
   a tautology where the executor ignored the chunk size.
3. No starvation. FIFO admission is asserted with two early requests occupying both slots and
   two later arrivals waiting; the later requests are admitted only after the earlier ones finish,
   in order. Round-robin progress is asserted by the once-per-step operation rule, by the short
   request finishing before a long prefill, and by each request's operation count matching its
   recorded steps.
4. Paged state under the batch. A test drives the executor one step at a time with a paged
   PagedSequenceState and a dense DenseSequenceState mirror. Every appended prompt chunk and
   decoded token is appended to both; PagedKvCache::check_invariants runs after every step, and
   the paged and dense mirrors' attention bits, reference-head logits bits and argmax must agree
   at the new last position. This is the task-44 property "paging is storage, not arithmetic"
   carried through a continuous schedule, not a new arithmetic claim.

The paged mirror is a test-side consumer of the task-44 harness. The production executor keeps its
own dense engine SequenceState; it does not pretend the reference engine's KV is paged.

## 4. Relation to the HTTP path

No HTTP handler, SSE framing, logprob or usage behaviour is changed. ReferenceDriver still runs
one job at a time on its engine thread. This executor is the admission/step-interleave shape the
device serving path will use once it has a batched attention entry; wiring it into the HTTP engine
thread is separate work, because the reference driver's per-request logprob, tokenizer streaming,
sampling and stop handling are not part of this scheduler.

## 5. Verification

    cargo test -p sparkflash-cli --lib continuous
    cargo test -p sparkflash-cli --lib
    cargo clippy -p sparkflash-cli --all-targets -- -D warnings
    cargo fmt --all -- --check
    RUSTDOCFLAGS="-D missing_docs" cargo doc --workspace --no-deps

The rustdoc command is part of the gate set: a crate-internal module must not accidentally expose
undocumented public items, and a green build alone did not catch that on task-44.

## 6. Window measurements, named and unclaimed

None of these were run; no number is claimed.

- Device batched attention through the pinned flash_mla_csa2 entry over rows staged from the paged
  pool, with several sequences in one launch.
- TTFT and throughput versus vLLM on the same prompt suite and batch sizes, including the per-step
  admission, staging and scheduling overhead.
- Eviction and prefix-sharing behaviour under real multi-request load; the host trace asserts
  invariants, not a hit rate or tail latency.
- Full model lm_head logits and next-token agreement for a served paged batch versus the
  device-chain reference run; this is the D40-type numeric comparison that needs the window.

## 7. What is not claimed

- No HTTP concurrency, no device batch, no TTFT or throughput improvement.
- No change to task-44's paged harness, no kernel/ABI change, no kv/device crate edit.
- No engine KV-backend seam: that remains the separately reviewed follow-on named in
  serving-paged-serve-wiring.md.
