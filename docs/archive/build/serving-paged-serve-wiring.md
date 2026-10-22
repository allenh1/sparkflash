# Serving paged KV wiring (task-44, S3)

Status: host-side serving-sequence harness implemented on top of task-21/task-43's landed
primitives. No node, GPU, kernel launch or HTTP performance claim is made here; the window
measurements are named in section 6 and remain unclaimed.

## 1. The seam this wires

Task-21 built the paged allocator (`sparkflash-kv::paged`) and the absolute-position operand
staging (`sparkflash-device::paged_kv`). Task-43 added the absolute indexer and source-row staging.
Neither crate knows what a served sequence is. Task-44 adds that sequence on the serving side,
inside `crates/sparkflash-cli/src/serve.rs`, without touching either crate:

| Piece | What it does |
| --- | --- |
| `PagedSequenceState` | one serving state whose format-2 KV bytes live in the paged physical pool; owns the block-table mapping, not a dense cache |
| `DenseSequenceState` | the reference: the same format-2 bytes held contiguously in logical order |
| `ServedStep` | one query position's observable output: attention output bits, deterministic reference-head logits bits, argmax next token |

The paged state allocates a page on request start, appends prompt tokens in one call and decode
tokens one call at a time, lets `PagedKvCache` share complete 64-token blocks on a repeated prompt,
returns pages and the tail slot on preemption, and recomputes a preempted request from its known
tokens. Every operation is followed by `PagedKvCache::check_invariants` in the tests.

## 2. Paging is storage, not arithmetic

The comparison is deliberately not the engine's `f32` dense `SequenceState`: that path has its own
KV arithmetic and an fp4 format-2 store cannot be bit-identical to it. The reference here is the
**same format-2 KV bytes held contiguously**. Both paths:

1. write the same deterministic format-2 entry bytes for the same compressed token group;
2. stage the task-43 absolute source rows and indexer keys, one through the block table and one
   from the contiguous layout;
3. run the released `mla_ref::attention_with_sink_union` CPU twin over the staged operands; and
4. run the same deterministic host reference head over the attention output.

Bit equality of the attention output, the head's logits and the argmax next token therefore shows
that paging changed storage only. The test names say `paging_is_storage_not_arithmetic` so a
reader cannot mistake this for a claim about the engine's model logits.

The reference head is a fixed host fixture, not the checkpoint `lm_head`. A served paged sequence's
full model logits are the window measurement in section 6, not a host claim.

## 3. What the host tests assert

Seven server-free tests in `crates/sparkflash-cli/src/serve.rs`:

* prefill positions and three decode steps: attention bits, logits bits and next token equal the
  dense reference; the test also covers a ratio-2 source layer (layer 2) as well as ratio-1 layer 20;
* repeated prompt: the second append reports shared complete blocks and both sequences produce the
  dense-reference step bits;
* preemption: a small pool forces a request to be preempted, another request uses the freed pages,
  the first is recomputed from its known tokens, and its step bits match both its pre-preemption
  output and the dense reference;
* staged operands: absolute source rows and the window rows/codes/scales are byte-identical at
  several prefill positions;
* absolute indexer keys: paged and dense staging agree at several positions;
* a randomized multi-request trace (add, append, preempt, recompute, free) calls
  `check_invariants` after every operation and never breaks a paging rule.

The tests are synthetic: the format-2 entry bytes are generated in test code, not read from a
device store. They exercise the real allocator and the real staging functions, not a mock of them.
`PagedSequenceState` owns one source layer's state; a full model would hold one such state per
source layer, which is a device-path concern and not claimed here.

## 4. What is not wired

The HTTP `sparkflash serve` request path still uses the reference engine's dense `SequenceState`.
This task adds the host sequence seam that the device serving path needs; it does not switch a
served HTTP request to a device KV store. That switch needs a device session, the device attention
entry and the window, and is not claimed here.

## 5. Verification command

    cargo test -p sparkflash-cli --lib paged_serve
    cargo clippy -p sparkflash-cli --lib -- -D warnings
    cargo fmt -p sparkflash-cli -- --check

The workspace suites are run before hand-off and their counts are in the task report, not in this
page. The targeted module has seven tests; the landed cli lib suite has 165 tests at this base.

## 6. Window measurements, named and unclaimed

None of these were run; no number is claimed.

* Device attention through the pinned `flash_mla_csa2_sm121` entry over rows staged from the paged
  pool (the CPU twin is bit-equal by construction; the device run is the missing evidence).
* Serving throughput and latency versus the dense reference for prefill and decode, including the
  per-step staging and page-allocation cost.
* Eviction and prefix-sharing behaviour under real multi-request load (the host trace asserts
  invariants, not latency or hit rates under a real scheduler).
* Full model `lm_head` logits and next-token agreement for a served paged sequence versus the
  device-chain reference run; this is the D40-type numeric comparison that needs the window.

## 7. Named follow-on (not authorized here)

An engine KV/attention backend seam (trait boundary, format-2 ownership, scheduler/preemption
interaction) would let `InferenceEngine` consume paged rows directly. That is a separate design and
review task; it cannot be bit-identical across f32 dense and fp4 format-2 and would need D40-type
numeric bars, not token identity. It is named here so this task does not smuggle it in as wiring.
