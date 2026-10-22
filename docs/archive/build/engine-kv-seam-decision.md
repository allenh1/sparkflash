# Engine KV/attention backend seam: decision proposal (task-47)

Status: **decision proposal, 2026-10-19. No code, no branch, no ABI change.** Base is a976ca7
(the accepted task-44 paged-serve harness plus its missing-docs fix). This document exists so the
parity-gate discussion can argue the seam before anyone implements it. It makes no numeric,
throughput or serving-parity claim.

The companion warning lives in serving-paged-kv.md: a paged store that mirrors a request's token
sequence is a diagnostic; it is not evidence that the serving KV is paged. This decision is about
the seam that would make the second statement possible.

## 1. The gap, in facts

- The engine sequence state is dense and private. SequenceState holds the consumed token ids and one
  attention state per backbone layer and MTP stage. Each AttnState holds a f32 window ring, f32
  compressed rows and f32 indexer keys. The fields are private; there is no KV trait, allocator
  handle or backend injection point in the normal prefill/decode path.
- The only existing attention seam is AttentionProjectionBackend. It splits the five attention GEMMs
  and the attention body out of the host reference path, and the device arm implements it. But the
  engine still owns the f32 AttnState, assembles f32 window/compressed rows, and hands the backend
  an already-materialised f32 KV view. The trait does not own storage, paging, prefix sharing or
  capacity.
- The paged store is a host allocator plus a pure staging view. sparkflash-kv::paged::PagedKvCache
  owns pages, block tables, prefix hashes, free queue, SWA tails, preemption and invariants.
  sparkflash-device::paged_kv::PagedKvView maps logical positions to format-2 source entries and
  stages operands for the released kernel; it never allocates or launches. The single-sequence
  device store owns SoA arrays but never allocates or evicts. Nothing connects either to
  InferenceEngine.
- The scheduler plans tokens and chunks, not pages. StepBoundaryScheduler has a background prefill,
  admission kinds, a decode interval and a trace binding. SchedulerConfig has chunk size, TP ranks,
  indexer knobs and cost-model assumptions, but no KV capacity, page quota, eviction or prefix-cache
  state. Admission can promise a first-token step without asking whether the KV pool can hold it.
- The HTTP driver is serial and reports no cache. ReferenceDriver owns one Job channel; the engine
  thread runs one request at a time. Started carries cached_tokens = 0, the usage cache breakdown is
  zero, and stats.kv_cache / kv_cache_hit_rate are the documented zero-state, not a measurement.
  The handler and response schema already have the fields a real cache would fill; the driver has no
  way to fill them.

So the seam has to answer two independent questions: where pixel/format-specific KV storage lives,
and where the attention arithmetic that reads it lives. The three options below draw that line in
different places.

## 2. (a) The three seam boundaries

### Option A - attention backend trait

Keep SequenceState and AttnState exactly as they are. Extend the existing attention backend
contract (or add a sibling attention-body trait) so the engine passes q, the f32 window/compressed
KV rows, candidate ids and metadata to a backend that returns attention output. The page mapping and
staging remain outside the engine, or inside the backend as a private detail.

- Engine: small. The normal prefill/decode path needs an optional backend parameter (the hybrid path
  already has one internally). SequenceState remains the owner of all KV, so no allocator lifetime,
  eviction or preemption enters the engine.
- Scheduler: unchanged. It still plans prompt chunks, decode steps and first-token steps. It has no
  page capacity to check and no page owner to notify.
- HTTP driver: configuration only. Choose host or device attention at startup; no request-time KV
  handle, no cache accounting, no new failure mode. The serving contract stays exactly as it is.
- What it can claim: the attention arithmetic ran on the device for an otherwise engine-owned f32
  sequence. That is a device-attention claim. It cannot claim paged serving KV, cached_tokens,
  kv_cache stats or prefix sharing, because the engine never sees the paged store.
- What it forces later: if the backend is to read paged rows, something still has to hand it a
  PagedKvView and a handle. That handoff is a KV-store boundary by another name, so Option A is a
  first step, not the final seam.

### Option B - KV store trait

Define a small storage contract owned by the engine crate (keeping the current dependency direction:
the engine depends on no other workspace crate). SequenceState owns a store object, defaulting to the
current f32 implementation, and the engine calls store operations for append/read/stage instead of
touching AttnState fields directly. A host dense implementation preserves today's arithmetic; a
paged implementation wraps PagedKvCache and PagedKvView. The attention backend consumes the staged
operands or a checked view, not raw engine-owned vectors.

- Engine: medium and structural. AttnState's KV fields move behind a trait object or a generic store;
  the host implementation is behaviour-preserving but every attention call site changes signature.
  The engine still does not know page hashes, free queues or schedulers; it knows a sequence has a
  store with a capacity/lifetime.
- Scheduler: must gain a capacity and lifetime view if the store is shared. At minimum admission needs
  a page-reservation answer before it promises first-token steps; preemption/recompute has to be
  coordinated with the sequence's store handle. If the trait is per-sequence and the driver owns one
  pool per request, the scheduler can stay token-only at the cost of no prefix sharing and no
  cross-request capacity.
- HTTP driver: owns or borrows the shared pool, creates/destroys store handles per request, maps
  capacity/preemption refusals to the existing error path, and fills the already-existing cache
  fields when the store can report them. No request schema change is required; the values of
  cached_tokens, the prompt-token cache breakdown, stats.kv_cache and kv_cache_hit_rate change from
  the documented zero-state to measured values once the store and prefix lookup exist.
- What it can claim: all of the serving-cache claims, but only after the implementation and a
  window, and only at the store's real accounting: pages in use, prefix blocks shared, evictions,
  preemptions, recomputes, cached prompt tokens and hit rate. It can also keep the host f32
  implementation bit-exact for the default path.
- Cost: this is a real engine refactor, not a flag. It needs a trait that can express both dense f32
  rows and paged format-2 staging without leaking either layout into the engine's arithmetic.

### Option C - SequenceState owns an allocator handle

SequenceState holds an opaque paged-allocator handle (ideally behind a trait, because the engine
currently has no workspace dependency). The engine is responsible for reserving pages as it writes
KV, releasing them when a sequence ends, and asking the allocator to preempt/recompute. The attention
backend reads the paged view through the handle.

- Engine: largest. It now owns page lifetime and capacity failure inside a forward pass, and a
  prefill chunk can partially succeed before a capacity refusal. The engine also becomes coupled to
  paged-store policy unless the handle is so abstract that it is Option B with a narrower interface.
- Scheduler: strongest coupling. It must allocate/admit pages before each step, keep headroom for
  decode growth, decide which request to preempt, and reconcile recompute with the chunk plan.
  SchedulerConfig needs a KV-capacity dimension it does not have; first-token promises become
  capacity-conditional.
- HTTP driver: must surface queueing, capacity refusal and preemption to clients, and must coordinate
  releases on disconnect/cancel. It becomes a pool owner as well as a job owner. The serial engine
  thread is no longer enough if the driver is to keep multiple requests in flight.
- What it can claim: the same cache claims as Option B, with the allocator directly in the engine's
  ownership path. It is the most direct model of a real paged serving engine.
- Cost: highest blast radius and the hardest to keep out of the engine's arithmetic. It also makes
  the engine depend on paging policy, which is exactly the separation task-21 built.

### Comparison

| | Option A attention trait | Option B KV store trait | Option C allocator handle |
| --- | --- | --- | --- |
| Engine owns KV | yes, f32 AttnState | through a store trait, default f32 | through a paged handle |
| Scheduler changes | none | capacity/lifetime view for shared pools | first-class page admission and preemption |
| HTTP driver changes | backend choice only | pool ownership, cache accounting, error mapping | queue/capacity/preempt/release, likely concurrency |
| Paged store is the serving KV | no | yes, in the paged implementation | yes |
| cached_tokens / kv_cache stats / prefix sharing | cannot claim | can claim after implementation and window | can claim after implementation and window |
| Dependency direction | engine unchanged | engine defines trait; kv/device implement it | engine coupled to kv policy unless abstracted |
| Main risk | does not close the gap | engine-wide refactor | engine and scheduler become one design |

## 3. (b) Numerical consequences and the bar each path needs

The engine's KV values are f32 in AttnState. They are produced by a reference round trip: window
through fp8, compressed latent and indexer keys through fp4 with E4M3 scales, then stored as f32.
The paged format is the stored fp4/scale bytes (288 B main entry plus 68 B indexer K, with the 528
B/slot window separate). The released device attention entry reads fp4 codes and scales, takes bf16
q, performs the three-pass f32 softmax over decoded values, and writes bf16 output. The two
representations and the two arithmetic paths are not the same thing.

No path below may claim bit identity to the current engine once it uses the device fp4 kernel. The
only bit-identical comparator is host f32 against host f32 with the same dequantised values and the
same arithmetic order; that is a storage/plumbing test, not a serving-parity test.

- Option A, host path: the host backend is the current reference path. The bar is exact: every
  token, logits row and attention output f32 bit must match the current engine. This is what keeps
  the default path honest while a device path is added.
- Option A, device path: keep the existing published attention bar, ||x - y|| / (1 + ||x|| + ||y||)
  class, 2e-2, plus the D40 metrics on the end-to-end logits: top-1 agreement, top-5/top-20 overlap,
  |delta logprob| and KL after the D41 floor rule freezes them. Attention-output agreement at
  2e-2 does not imply token agreement and must not be reported as parity.
- Option B, host dense store: bit-exact against the current engine, 0 differing f32 bits at every
  position. This is the regression gate for the refactor, not a serving result.
- Option B, paged store with host attention over dequantised values: two separate bars. First, stored
  bytes must match the same quantiser's bytes exactly (entry-level comparison against a dense byte
  sidecar, not against f32 values re-quantised on the fly). Second, attention over those bytes must
  match a host recomputation from the same dequantised values bit for bit. This is the task-44
  storage-is-not-arithmetic property, and it is exact for the host path. Against the current f32
  engine, the comparison is not bit-exact: it is the D40/D41 numeric bar.
- Option B or C, device paged attention: the 2e-2 attention body bar plus the D40/D41 logits bar.
  There is no f32-bit or token-equality bar available. The paged path also needs a same-paged-path
  exactness bar: for one request, paged run with no sharing must produce the same tokens and logits
  bits as paged run with prefix sharing, and paged run with preemption/recompute must produce the
  same tokens and logits bits as paged run without eviction, provided the recompute uses the
  identical retained tokens. That is exact within the paged path even though it is only numeric
  across representations.
- Option C adds one more exact bar: the same request under a capacity-triggered preemption/recompute
  must be bit-identical to the same request when capacity never binds, on the paged path. If that
  fails, capacity is changing the answer, not just the schedule.
- Cache statistics need accounting bars, not numeric tolerances: page refcounts, free queue,
  block hashes, cached token counts and hit rate must equal the model the test constructs; the
  invariants checker must pass after every step. A cache number is not a distribution and must not
  be given a KL-style tolerance.

Two honest limits on the numeric argument. First, the window's oracle is vLLM at TP=4, not this
host f32 engine; D40/D41 compare outputs, they do not bless a storage layout. Second, the stored
dequantised dtype is its own question: the release's cache and AttnState are not assumed to have the
same post-quantisation dtype, and the host stores f32 after its round trip. Any claim that the paged
bytes reproduce "the engine's cache exactly" must prove byte equality against the same quantiser,
not infer it from f32 values.

## 4. (c) ABI cost if the device attention entry must change

The current sequence entry is the released flash_mla_csa2_seq_sm121 with the base entry's 19-word
argument list. FlashMlaParams pins the count at 19 and the order; the m4_abi signature table pins the
declaration; the m4-abi-freeze CI job pins the scanner family line count; the kernel baseline pins
the cubin. Path A in serving-paged-kv.md section 5 is the precedent: appending a 20th positions
argument would change FlashMlaParams 19 -> 20, the m4_abi signature/declaration pin, every launch
site and harness mapping, and the baseline. That is a decision-record-level change with its own
reviewed pin, not a source edit a seam task may slip in.

- Option A: no ABI change is required if the backend keeps staging window and absolute main-KV rows
  through PagedKvView and calls the existing 19-word entry. The cost is the already-named O(context)
  staging per source layer per step plus the HSI key pool. If the backend instead wants compact rows
  or an in-kernel block-table read, that is Path A or a new kernel, with the 19 -> 20-class pin cost.
- Option B: the same. A paged store does not by itself force a kernel change; staging the store's
  rows into the existing operands keeps the ABI. A paged-read kernel or a positions argument would
  be a separate ABI decision. The store trait must not silently grow a device-layout mandate.
- Option C: same device ABI as B. The allocator handle is a host/engine interface; it does not change
  the kernel. The temptation to avoid staging by teaching the kernel to walk pages is exactly the
  ABI change that needs the frozen-surface decision.
- Recommendation: keep the 19-word entry frozen for the first implementation. Do not combine task-47
  with a Path A pin change. If the first measured window shows staging dominates, make Path A its own
  decision with a measured argument, as section 5 of serving-paged-kv.md already says.

## 5. (d) Recommendation and what it would and would not fix

**Recommend Option B, the KV store trait, with the allocator handle hidden inside the paged store
implementation.** Option A is useful as the device-attention arm, but it cannot produce the three
serving claims this task exists for. Option C puts page admission into the engine before the
scheduler has a capacity model, which is the wrong order.

The recommended shape:

1. Engine defines a minimal KV store contract (sequence lifetime, append/write, read/stage, capacity
   error) and keeps a host dense implementation as the default. The engine's arithmetic stays
   layout-agnostic; the host path keeps its current bit-exact tests.
2. The paged implementation wraps PagedKvCache and PagedKvView behind that contract. It owns page
   hashes, sharing, eviction and preemption; it exposes staged operands or a checked view to the
   attention backend. The engine never sees a block table.
3. The scheduler owns the shared pool/quotas and the admission decision. It gains a capacity and
   lifetime view; first-token promises become conditional on a page reservation. This is the task-22
   admission shape applied to KV, not a reimplementation of the chunk planner.
4. The HTTP driver owns request-to-sequence handles and the prefix key derived from prompt tokens. It
   fills cached_tokens, the prompt cache breakdown, stats.kv_cache and kv_cache_hit_rate from store
   counters, and maps capacity/preemption to the existing error path. The handler schema does not
   need to change; the values stop being the documented zero-state.

   The cache report must keep the existing fields' meanings. A first implementation may fill
   cached_tokens and global_kv_cached_tokens from measured complete-block reuse while leaving the
   UMA/NVMe tier fields and the CED/SWA replay fields at their documented zero/not-measured state;
   it must not copy a global paged hit into uma_hit_tokens or nvme_restored_tokens.

What it would let us finally claim, after implementation and a window:

- cached_tokens and the global prompt-cache breakdown are measured complete-block reuse, not zero;
  the other tier and replay fields stay zero unless their own path measures them.
- kv_cache stats and kv_cache_hit_rate are store accounting: pages, shared blocks, evictions,
  preemptions and recomputes.
- Prefix sharing under load: the paged store already shares complete 64-token blocks by the host
  radix key. Serving can claim it only when the driver keys requests consistently and the store
  reports hits; a collision assumption inherited from the host radix key remains part of the claim.
- Device attention over paged rows at the existing 19-word entry, within the 2e-2 attention bar and
  the D40/D41 logits bars, not token equality.

What it would not fix:

The load-bearing boundary, to be quoted in the decision record: Option B does not fix the
f32-engine vs fp4-device numeric gap, the O(context) staging cost, HSI full-pool selection, the
serial HTTP engine, or the host-resident paged pool. Those are the boundary between "the serving KV
is paged" and "serving parity."

- The f32-engine vs fp4-device numeric gap. It gives no bit identity and no token parity; those stay
  D40/D41 facts.
- The O(context) per-source-layer staging cost. Staging is still required by the frozen ABI; the
  throughput question remains the window's to answer.
- HSI full-pool population and the top-512 candidate merge (phase2 open item 2). The seam carries
  staged keys; it does not select candidates.
- The device KV store's one-sequence ownership and the lack of a real device paged-read kernel. The
  current PagedKvCache pool is host-resident and PagedKvView stages bytes for the kernel; a serving
  implementation must either keep that staging or build a device-resident paged store. The seam is a
  host interface and does not create the latter by itself.
- The HTTP driver's serial job execution and the engine's lack of batch state. Task-22's Continuous
  Batch is still host-only and not wired.
- Prefix sharing alone does not make a prefix cache correct under collisions, cancellation, partial
  prefill or eviction; those are store/driver tests, not a trait declaration.
- Performance. It may make staging and cache accounting visible, which is required before throughput
  can be argued; it does not make the path faster.

If the Lead rejects the engine refactor for now, the fallback is not Option A. It is to keep the
crate-internal paged store as diagnostics only and state that the serving KV is not paged until
Option B or C is approved. This proposal also rejects a default-off diagnostic mirror as a
compromise: it would leave the serving KV dense, hide the observer's cost behind a flag, and make
the bookkeeping claim while avoiding the user-visible cost. That is not a seam and not honest
consumption. A diagnostic mirror in the request path would spend CPU to make the bookkeeping look
consumed without changing the served KV.

## 6. (e) Measured prerequisite: the window lines that must exist before throughput

No throughput argument is possible until a window records the following lines at the same device,
driver/cubin set, prompt suite and batch sizes. These are measurement requirements, not code
requests.

### Configuration and identity

- device, driver binary sha256 and full cubin-set identity (name, bytes, sha256), pinned kernel entry
  and ABI word count;
- model/checkpoint root, TP ranks, dtype, clock/frequency/power state, temperature and any co-tenant
  load; the line must say whether the node was idle;
- prompt-suite sha256 and the batch sizes/context lengths in the run.

### Paged store and request accounting

- per request: id, arrival/admitted/first-token steps, prompt length, max tokens, pages reserved,
  pages shared, cached prompt tokens, prefix-block hits, evictions, preemptions, recomputes and the
  free-page/free-tail high-water mark;
- PagedKvCache invariants result after every step; a failed invariant is a failed run, not a warning;
- for each reused block: block key/hash, refcount, and whether the block was complete; partial blocks
  must show as not shared.

### Attention and staging

For each source layer and position (or a stated sample) the line must separate:

- window rows and selected candidate count; absolute main-KV rows staged (count and bytes); indexer
  key pool staged (bytes); host FP4/FP8 dequantisation/copy time; host-to-device copy count and
  bytes; kernel launch count; kernel time; total step time;
- the PagedKvView path used (absolute/main, window rotation, indexer) so the line names which
  staging cost was paid;
- the attention twin line at the published 2e-2 bar: exact-BF16-output count, max abs delta, max ULP,
  non-finite count, and a per-head/per-block mismatch breakdown if the device backend already prints
  it;
- for ratio-2 and ratio-1 layers separately, and at positions 0, 63/64, 127/128 and a post-block
  boundary, so window rotation and page boundaries are visible rather than averaged away.

### Request and batch timing

- TTFT, prefill time/tokens, decode time/tokens, tokens/s, and the same split between engine work,
  staging and launch overhead;
- batch wall time, aggregate tokens/s, queueing time and admission latency for batch widths at least
  1, 4 and 16 (or the stated device limit), with warmup and repeat count;
- cached_tokens and kv_cache_hit_rate alongside the timing, so a speedup cannot be claimed as compute
  if it is only cache reuse.

### Numeric evidence on the same window

- D40/D41 per-position logits comparison against the vLLM oracle (top-1, top-5/20/100/1000 overlap,
  max/mean |delta logprob|, KL, JS) with the D41-frozen bars and the measured floor;
- the attention body's 2e-2 line and the indexer score lines;
- the same-paged-path exactness lines from section 3 (sharing vs no sharing, eviction vs no eviction),
  or an explicit statement that they were not measured.

### What a throughput claim needs that these lines do not give on their own

- a same-window baseline: vLLM TP=4 on the same hardware, prompt suite and batch sizes, with the
  served root and version recorded; a host-f32 run is not a throughput baseline for a device path;
- staging eliminated as the explanation: if the line shows staging time dominates, the result argues
  for Path A or a paged-read kernel, not for the seam;
- variance: repeated runs and a stated spread; one run is a diagnostic, not a bar.

Until those lines exist, task-47's output is a decision, not evidence. The window cannot be replaced
by host tests, and the 2e-2 attention twin cannot be promoted to a serving claim.

## 7. Decision requested

Choose one of:

- A: attention backend trait only; no serving-cache claims, no engine refactor; device attention can
  land first.
- B (recommended): KV store trait with the allocator inside the paged implementation; engine default
  stays exact; scheduler gains capacity/lifetime; driver fills cache fields; the claims in section 5
  become testable.
- C: SequenceState owns the allocator handle; fastest to a "real paged engine" shape, highest risk
  and largest scheduler/driver change.
- Defer: keep serving-paged-kv.md's diagnostic-only status; no serving KV claim.

The recommendation is B, with the 19-word ABI frozen and no Path A change in the first step. That is
the smallest boundary that can eventually make "the serving KV is paged" true rather than a module
name.

## 8. What this proposal does not decide

- It does not approve code, a branch, a kernel, an ABI change or a pin update.
- It does not freeze D41 bars; it uses D40 terminology and defers to the D41 record.
- It does not claim any measured number from task-21, task-22, task-44 or task-43. Those are host
  tests and named-unclaimed windows.
- It does not choose a device paged-read kernel; that is Path A or a successor.
