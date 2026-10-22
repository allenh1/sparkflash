# WS-F: batched/chunked prefill is the scheduler default (continuous batching, preemption, prefix reuse)

**Status: 2026-10-21, host policy plus the reference-executor contract, tested on synthetic
fixtures. NOT validated on a device.** No GPU ran for this page; there is no TTFT, tokens-per-second
or device claim in it from this work. The shipped TP=4 serve path still runs the pre-batch-17
token loop, because that is the path the endpoint measured faster. The batched/chunked policy is
the scheduler default, its reference executor is tested bit-identical to the token loop, and the
one-line runtime flip plus the measurement row it needs are in section 7. Every claim beyond the
three cited measured rows is an estimate, a hypothesis or unvalidated on device, and is labelled
as such below.

## 1. Why the runtime default is not flipped yet

The scheduler default and the runtime default are two different decisions, and the measurements
point in opposite directions (all from [tp4-tokens-per-second.md](tp4-tokens-per-second.md)):

| quantity | value | provenance |
| --- | --- | --- |
| token-loop endpoint TTFT, 37 prompt tokens | **39.134645546 s** | measured, [tp4-tokens-per-second.md#L37](tp4-tokens-per-second.md#L37) |
| batch-17 chunk-in-serve endpoint TTFT, same prompt | **43.624741849 s** | measured, slower, hence opt-in ([#L46](tp4-tokens-per-second.md#L46)) |
| flip-back census (token loop on the batch-17 tree) | **38.964443188 s** | measured, [#L387](tp4-tokens-per-second.md#L387) |
| bench harness, 37 prompt tokens | chunk **24.6 s** vs loop **33.1 s** | measured, bench only ([#L371](tp4-tokens-per-second.md#L371)) |

So the chunked path wins the bench harness and loses the endpoint by 4.5 s. The serve-path overhead
is unattributed (batch-18 work). Making the batched policy the *runtime* default now would trade a
measured regression for an unmeasured hypothesis, so:

* the **scheduler** default is batched ([batching.rs](../../../crates/sparkflash-scheduler/src/batching.rs#L189)),
  which is what new callers and the tests exercise;
* the **shipped CLI** default stays the token loop
  ([tp4_host.rs](../../../crates/sparkflash-cli/src/tp4_host.rs#L816)), and the flip is one constant;
* both are the same code path, selected by
  [PrefillMode](../../../crates/sparkflash-scheduler/src/batching.rs#L129).

## 2. What landed

| item | state | location |
| --- | --- | --- |
| batch configuration: prefill mode, max_num_seqs, token budget, chunk size, KV pool, preemption policy, decode-first, prefix reuse, swap bytes | implemented, validated, defaults tested | [batching.rs](../../../crates/sparkflash-scheduler/src/batching.rs#L189) |
| token-budget chunk planner (contiguous ranges, budgeted) | implemented, tested | [TokenBudgetPlan](../../../crates/sparkflash-scheduler/src/batching.rs#L473) |
| batch step / scheduled ops (prefill chunk, decode, prefix reuse, preempted, swapped in) | implemented, tested | [BatchStep](../../../crates/sparkflash-scheduler/src/batching.rs#L400) |
| continuous batcher: admission, mixed prefill+decode steps, rotation fairness, preemption, swap accounting, prefix reuse | implemented, tested over the real paged KV | [ContinuousBatcher](../../../crates/sparkflash-scheduler/src/batching.rs#L769), [plan_step](../../../crates/sparkflash-scheduler/src/batching.rs#L959) |
| reference batched prefill executor: one forward per chunk, token-loop compatibility, explicit-range validation | implemented, tested bit-identical | [batch.rs](../../../crates/sparkflash-engine/src/forward/batch.rs#L40), [prefill_token_budget](../../../crates/sparkflash-engine/src/forward/batch.rs#L62), [token_loop](../../../crates/sparkflash-engine/src/forward/batch.rs#L190) |
| prefix reuse at the reference level: fork the state that already holds the cached prefix | implemented, tested | [SequenceState::fork](../../../crates/sparkflash-engine/src/forward/mod.rs#L100) |
| CLI prefill-mode seam: shipped constant + new env override + legacy env spelling | implemented, tested; shipped default unchanged | [tp4_host.rs](../../../crates/sparkflash-cli/src/tp4_host.rs#L816), [shipped_prefill_mode](../../../crates/sparkflash-cli/src/tp4_host.rs#L836) |
| one fused device launch per mixed step | **design only** | BatchStep::device_forward_launches is a count, not an execution |
| the swap byte copy | **accounting only** | swap_bytes is recorded; nothing moves KV |
| a serve loop that constructs ContinuousBatcher | **not wired** | integration step, section 8 |

The scheduler crate does not depend on the engine and the engine does not depend on the scheduler;
the batcher plans over the paged cache and the engine executes explicit ranges, so the two stay
independent and neither direction was added.

## 3. The scheduler contract

**Plan then accept.** [ContinuousBatcher::plan_step](../../../crates/sparkflash-scheduler/src/batching.rs#L959)
returns a [BatchStep](../../../crates/sparkflash-scheduler/src/batching.rs#L400); the executor runs every
forward-bearing op in order and calls
[accept_step](../../../crates/sparkflash-scheduler/src/batching.rs#L993) with one token per producing op,
in op order. Preemption has to replay a victim's real history, so the batcher owns the tokens
(prompt plus generated), not just their counts.

**Token budget.** A step plans against max_num_batched_tokens (8192 by default). Decodes consume one
token each; a prefilling sequence contributes at most one chunk per step. A chunk is bounded by the
smaller of prefill_chunk_tokens (4096 default) and the remaining budget, so the plan never exceeds
the budget and every chunk is contiguous.

**One chunk per sequence per step, in rotation.** Both the decode and prefill phases build an order
from the running list and rotate by a per-phase cursor, so under a budget smaller than the demand no
prefilling request starves. This is pinned by
[a_tight_budget_does_not_starve_a_prefilling_request](../../../crates/sparkflash-scheduler/tests/batching.rs).

**Decode first.** With decode_first (the default), a mixed step emits the decode tokens before the
prefill chunks; the test
[a_step_mixes_decode_tokens_with_prefill_chunks](../../../crates/sparkflash-scheduler/tests/batching.rs)
checks the op order.

**Preemption and swap.** Capacity is the real paged cache (64-token pages, hash-block sharing). A
preemption picks the last-admitted running request that does not already have an op in this step,
returns its pages, and queues it as swapped. Two policies:

* Recompute (default): on resume the retained prefix is reused from the paged cache and the rest of
  the consumed history is re-forwarded as prefill chunks.
* Swap: on resume the history is reattached and the request returns to decode without re-forwarding.
  The swap's *bytes* are accounted (history x swap_bytes_per_token + SWA_TAIL_BYTES) but not moved:
  this crate does not touch KV, so the copy belongs to the device/tier swap in the device window.

Both policies are tested end to end at the crate level.

**Prefix caching.** Complete 64-token blocks whose hash the paged cache already holds are reused; a
finished request's data stays in the free list while its hash survives, and is evicted when another
request takes its page. The batcher reports the reused token count per request and plans the prefill
from the end of the reusable prefix, capped one token short of the prompt so the last logits still
come from a forward. prefix_reuse = false is the conservative compatibility path: it forwards the
whole prompt even when blocks are cached. Hit, miss and eviction are all tested against the real
cache. The KV layer's PrefixMatchPlan remains the authority on the SWA-tail replay decision; the
batcher deliberately does not model it.

## 4. The reference executor contract

[BatchedPrefill](../../../crates/sparkflash-engine/src/forward/batch.rs#L40) runs one
InferenceEngine::prefill call per chunk over explicit ranges and counts them. Its rules:

* ranges start at the sequence's consumed position, are contiguous and non-empty, and end at the
  prompt's end; anything else is a typed Shape error, so a cached prefix cannot be silently
  forwarded twice;
* every prompt token is written once (tokens_written equals the covered range length);
* on the batched path forwards equals chunks, never the token count (except for a prompt shorter
  than two chunks); on the compatibility path [token_loop](../../../crates/sparkflash-engine/src/forward/batch.rs#L190)
  forwards equals the token count, one decode-shaped forward per token, which is the branch the
  shipped CLI uses today.

Bit-equality is tested against a whole-prompt call and against the token loop for prompt lengths
1, 2, 3, 5, the tiny window, window + 1, window + 3 and 2 x window + 1, for every chunk size from one
token to the whole prompt. Logits, head input and the sampled next token must be identical bit for
bit. Two fields are excluded by construction, not by tolerance: consumed (a chunk consumes its own
range) and draft/confidence. The reference drafts nothing for a forward that starts a sequence
(forward/mod.rs: the draft anchor is None at position 0), so a whole-prompt prefill has no draft
while a chunked one does. That asymmetry is pinned by
[the_draft_tokens_differ_by_construction_when_the_chunk_starts_at_zero](../../../crates/sparkflash-engine/tests/batched_prefill.rs);
the draft is a proposal the verification step owns, so it is not part of the KV/logits equivalence
claim.

Prefix reuse is tested at the reference level too: compute a 5-token prefix, fork the state, forward
the 4-token suffix in 2 chunks, and compare against the 9-token whole prompt; then decode from both
states with the same token and compare again.

## 5. Tests that pin this work

New tests, all green on this host:

* scheduler unit tests: defaults are batched with the spec budget, mode parser, validation names
  each rejected knob, token-budget plan contiguity and refusals, the labelled TTFT estimate, step
  forward/launch counts, page rounding.
* scheduler scenario tests ([tests/batching.rs](../../../crates/sparkflash-scheduler/tests/batching.rs)):
  default is one chunk per step and zero decodes for a one-token completion; the token-loop
  compatibility mode is one forward per prompt token; a step mixes decodes with chunks and the
  decode-first order holds; chunks are contiguous, budgeted and at most one per request per step;
  a tight budget does not starve; the same submissions produce the same step sequence; recompute
  preemption replays the history and finishes both requests; swap preemption resumes without
  re-forwarding; a cached 64-token block is reused with one forward for the remaining token; a
  different prompt misses; a reused page evicts the cached block and the next attempt misses;
  prefix_reuse = false forwards the whole prompt; malformed submits and accept_step token lists are
  refused; cancel frees KV and leaves the other request alone.
* engine tests ([tests/batched_prefill.rs](../../../crates/sparkflash-engine/tests/batched_prefill.rs)):
  batched and token-loop paths bit-identical to one call across the chunk grid; the batched plan is
  not a per-token loop; a forked prefix forwards only the suffix and decodes identically; a
  non-prefix state and malformed ranges are refused; the range planner refuses impossible shapes;
  the draft asymmetry above.

## 6. Measured, estimated, unvalidated

* **Measured**: the four rows in section 1, and the fact that the suite above passes (host f32
  reference, synthetic tiny model; no device).
* **Estimated**: PrefillCostEstimate converts a cached-token count into a TTFT estimate using the
  measured endpoint chunk cost (1179 ms/prompt-token) or token-loop cost (945 ms/prompt-token). A
  token-budget chunked forward is not the measured whole-prompt chunk, and the endpoint is not the
  bench harness, so the type exists to discuss provenance, not to quote a number as a win. No
  prefix-cache TTFT saving is claimed from this page.
* **Unvalidated on device**: the batched prefill's device cost, the one-launch-per-step fusion, the
  swap copy, the paged pool's behaviour under the device path, and any effect on endpoint TTFT or
  tokens per second. Everything in this paragraph is design work for the device window.

## 7. The one-line flip and the validation row it needs

The flip is [SHIPPED_PREFILL_MODE](../../../crates/sparkflash-cli/src/tp4_host.rs#L816):

    const SHIPPED_PREFILL_MODE: PrefillMode = PrefillMode::TokenLoop;   // -> PrefillMode::Batched

No other change is needed for the runtime default, because
[prefill](../../../crates/sparkflash-cli/src/tp4_host.rs#L1112) branches on it. Overrides:

* SPARKFLASH_TP4_PREFILL_MODE = batched | chunk | chunked | token-loop | token_loop | loop | token.
  An unparseable value warns and falls back to the shipped mode; it never silently changes the
  default.
* SPARKFLASH_TP4_PREFILL_CHUNK, the legacy spelling, keeps working: any truthy value still means
  batched (which is what the serve census used: SPARKFLASH_TP4_PREFILL_CHUNK=1), 0/false/off/no mean
  token loop, and a mode name is parsed like the new variable. A guard test asserts the shipped
  constant is the token loop and that the legacy mapping keeps its meaning.

The flip must land with this row, or with a stated reason it cannot be produced yet:

| field | required |
| --- | --- |
| prompt | the same 37-token prompt as the census rows (and at least one long prompt, e.g. 512 tokens) |
| metric | endpoint TTFT, measured by the serve census harness, at least 3 repeats, median reported |
| bar | beats 39.13 s TTFT at 37 tokens and does not regress decode tokens per second |
| harness | both bench and endpoint, because they disagree (24.6 s vs 43.62 s) |
| evidence | serialised census artefact plus the binary hash and the exact env/config, so the row is reproducible |

Until that row exists, the honest state is: batched policy is the scheduler default and is tested
bit-identical, the runtime default is the measured-better token loop, and the difference is
attributed to the serve-path overhead (batch 18).

## 8. What is not wired yet

No runtime path constructs ContinuousBatcher today. The CLI seam selects between the existing
token-loop and whole-prompt chunk *TL4 methods*; it does not drive the new executor. The integration
step is:

1. a device (or host-twin) prefill forward that executes explicit token-budget ranges per step and
   counts launches - the reference executor in this page is the host contract for it;
2. a serve loop that plans with ContinuousBatcher and accepts the tokens it produced, over the
   per-rank SequenceState / paged KV already in the TP=4 host path;
3. the section 6/7 measurements, then the one-line flip.

Until step 1 exists there is nothing defensible to flip: a batched plan without a batched device
forward would only re-label the token loop.

## 9. Provenance

* Branch task/phase1-async-device; no commits made by this work.
* Host f32 reference engine, synthetic tiny configuration, no GPU, no cluster; the live four-node
  vLLM TP=4 endpoint was not touched.
* Measured rows are quoted from [tp4-tokens-per-second.md](tp4-tokens-per-second.md) with line
  anchors; nothing else on this page is a measurement.
* Prior scheduling records: [s4-host-continuous-batching.md](s4-host-continuous-batching.md),
  [s4-batch-chunked-prefill.md](s4-batch-chunked-prefill.md).
