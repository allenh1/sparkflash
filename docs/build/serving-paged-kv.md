# Serving paged format-2 device KV (task-21, S3)

Status: implemented on branch `task/s3-kv` from main `0639413`. Host code and host tests only; the
device window run is **unclaimed** (the node/GPU freeze is in force, so no store was opened and no
kernel launched for this milestone).

Task-43 (absolute-position staging for the paged serving path) is implemented on the same
branch and documented in Section 5; its device window run is unclaimed like the rest.

## 1. What the milestone adds

The device KV store (`docs/build/phase2-device-kv-store.md`) deliberately owns exactly one
sequence's arrays and never allocates, evicts or infers a position. Serving needs the other half:
many sequences over one device pool. Two new modules split that seam and neither writes KV bytes:

| Where | What |
| --- | --- |
| `crates/sparkflash-kv/src/paged.rs` | the paged allocator: pages, block tables, free queue, prefix sharing, eviction, preemption, SWA tail slots, invariants |
| `crates/sparkflash-device/src/paged_kv.rs` | the device-attention seam: logical token -> source entry through the block table, candidate row staging, absolute-position staging (task-43), oldest-first window rotation gather |
| `crates/sparkflash-kv/tests/paged.rs` | 16 paging/eviction rule tests |
| `crates/sparkflash-device/tests/paged_kv.rs` | 11 addressing, absolute staging and twin tests |

### 1.1 The module name is not a serving claim

**"The paged store mirrors the request token sequence" is NOT the same claim as "the serving KV
is paged."** The first is a diagnostic: the store can reproduce the token sequence it was given,
check its own invariants and stage operands for a comparison. The
second says the engine's KV/attention path actually reads and writes the paged pool as its backing
store, with the serving cache fields and lifetime following from that. Until the engine KV/attention
seam exists, the paged modules are storage tests and a staged device seam, not the KV backing of a
served request. Do not read paged_serve or paged_kv as evidence that serving uses paged KV.

The decision proposal for that seam, including the three possible boundaries and the numeric and ABI
cost of each, is docs/build/engine-kv-seam-decision.md (task-47). This document's device window
remains unclaimed either way.

The unit is the store's own page: [crate::config::KV_PAGE_TOKENS] = 64 logical token positions. A
Full layer at compression ratio r stores 64 / r entries per page, and a block-table page id is the
page index the store's arrays already take. Logical block b covers tokens [64b, 64b + 64); a partial
final block still owns a page. The ratio mapping is the part that must not be flattened, so
`PagedKvView::source_entry` exposes it and the tests assert both ratio-2 (layer 2) and ratio-1
(layer 20) layers.

## 2. Policies

Fixed in the module doc and asserted by the tests, so a reviewer can check intent against evidence:

* **Allocation is FIFO over the free queue.** A page whose last reference dropped is pushed to the
  back, so the never-used pages of a fresh pool are consumed first and cached prefixes survive while
  there is room.
* **Block keys are the host radix tree's key.** A complete block is hashed with
  `radix::block_hash` (the same non-cryptographic u64 FNV-1a key the three-tier host cache uses),
  so the pager inherits that cache's collision assumption; a serving front end that needs a
  collision-proof key changes both together, not this module alone.
* **Prefix sharing is per complete 64-token block.** When an append completes a block whose hash is
  cached, the page the append just built is released and the block table points at the cached page;
  the block's reference count increments. Partial blocks are never shared (their hash would change).
* **Last reference released keeps the data cached.** The page goes to the free queue with its block
  hash; a later sequence with the same block reuses the data. Allocating that page for a different
  block drops the hash first and counts the eviction.
* **SWA tails are a separate hard capacity.** One 128-slot tail per running sequence; a preempted
  sequence returns its tail and a resumed sequence takes a new one.
* **Preemption returns everything** (pages and tail), leaves the handle `Preempted` with no tokens,
  and a later append recomputes it. Preempting twice is refused.
* **Truncation must end on a block boundary**; a partial truncation would need a page-splitting rule
  serving does not have, so it is refused rather than approximated.
* **Fork shares the donor's first complete blocks** (beam/multi-candidate prefix reuse), refusing a
  donor that is preempted, shorter than the prefix, or whose block is not complete.
* **Refusals are named, not silent**: over-capacity (`Capacity`), a 65th tail (`TailCapacity`),
  unknown/duplicate handles, out-of-range positions, bad truncation/fork, and
  `PagingError::Invariant` from the checker.

## 3. Invariants (`PagedKvCache::check_invariants`)

The tests call this after every operation that moves pages; the randomized trace calls it after all
600 steps. It checks:

1. the free queue is unique and in range;
2. every block table is `ceil(tokens/64)` long, has no duplicate page, and mixes no page with the
   free queue; a running sequence has a tail and a preempted one holds nothing;
3. every cached block's reference count equals the number of block tables naming its page, an
   unreferenced cached block's page is on the free queue, and no page is cached under two hashes;
4. every page is exactly one of: free, referenced-and-cached, or a sequence's partial final block
   (referenced without a cached hash);
5. tail slots are unique, disjoint from the free-tail list, and running + free tails cover the pool.

The checker is also tested negatively: two in-module tests corrupt a duplicate free page and a
referenced page with no cached block and assert the checker names an invariant error.

## 4. Device attention seam

`PagedKvView` turns a sequence's block table into the released `flash_mla_csa2_sm121` operands:

* `source_entry(sequence, layer, logical)` -> `(page, entry_in_page, pool_index, byte_offset)`,
  the one mapping every gather uses.
* `absolute_source_rows(...)` (task-43) stages positions `0..=position` as dense
  `kv_fp4`/`kv_scales` rows in absolute token order: row p is token p's expanded row, so a
  ratio-2 entry serves both of its tokens' rows. The paged path passes absolute ids and the absolute
  position, so the kernel's `candidate <= position` mask compares two absolute values and is not
  vacuous, and the indexer stage scores absolute block positions. This is the operand seam the
  serving path uses.
* `candidate_rows(...)` remains for a caller that passes a compact, proved-causal list: it stages
  the main-KV row of each selected compressed entry in candidate order.
* `absolute_indexer_keys(...)` (task-43) gathers the layer's indexer K entries through the
  mapping, dequantises FP4 E2M1 + E8M0 on the host, narrows each value to bf16 and stages the HSI
  pool in absolute block order, zero past the query. The kernel never learns the block table; slots
  past the query are unreachable under its own mask.
* `window_rows(...)` gathers the last min(position, 128) SWA slots **oldest first** from the tail
  ring. That rotation copy is open item 1 of the device-store doc; the per-slot bytes are the
  store's own 528 B (512 code + 16 scale).
* **The mask invariant, in one line:** compacted ids are only safe while every candidate is causal;
  whenever masking is semantically required the device kernel must be given full-length ids (or
  equivalent positions). The chosen path satisfies it by giving the kernel absolute ids: the
  absolute mask rejects a future candidate bit-identically to omitting it, and moving the query one
  token later makes that same candidate attendable and changes the output, so the test shows the
  mask is live rather than vacuous. `compact_candidates` remains the checked helper for callers
  that legitimately pass compact ids, and refuses a candidate past the query position.
* The CPU twins run `mla_ref::attention_with_sink_union` over staged operands and over an
  independently rebuilt contiguous kernel view: absolute rows at ratio 2 reproduce the contiguous
  view bit for bit, and absolute ids over both give the same f32 output. The indexer twin rebuilds
  the HSI pool from the synthetic values without going through the paged pool and compares
  `mla_ref::indexer_block_scores` per block (score bits and best token); a future token that
  would dominate its block if leaked is asserted not to win, and to win once the position passes it.

## 5. Task-43: absolute-position staging (Path B, decided)

The Lead's decision was Path B: no kernel, ABI or pin change. The paged path stages absolute
operands and passes absolute ids plus the absolute query position to the unchanged pinned entry, so
both the mask and the indexer are correct by construction rather than by new kernel logic. Path A
(a 20th `positions` argument that would let stage 2 keep compact rows) is the named follow-on,
not this milestone:

* **Cost paid now:** O(context) main-row staging per source layer per step (288 B per token, four
  source layers) plus the 4 MiB HSI pool and its host FP4->bf16 dequantisation. That is a
  throughput cost, accepted because correctness of the paged path is what blocks serving; the
  device window run that would measure it is unclaimed.
* **Named follow-on (Path A):** append `const uint32_t* positions` to the CSA2 sequence entry, use
  `positions[slot]` for the format-2 mask while stage 2 still reads compact rows, and stage the
  HSI keys absolute (unchanged). That makes staging O(candidates) and is expected to win once a
  paged-read kernel exists and a window can measure the staging cost. It changes `FlashMlaParams`
  19 -> 20, i.e. the frozen M4 ABI surface (`m4_abi` signature table, the `m4-abi-freeze` CI job,
  every launch site, the harness params mapping, the baseline pin), so it is a decision-record-level
  change with its own reviewed pin, not a task-21/43 follow-on. The argument for it must be made
  against a measured throughput number.
* **Scope kept out:** full HSI pool population (the 2,048-block candidate pool and the top-512
  merge) remains phase2 open item 2; this task stages the keys for the twin and the seam and does
  not select candidates.

## 6. Evidence

Host-only, from this worktree with its own `CARGO_TARGET_DIR`:

| Suite | Result |
| --- | --- |
| `cargo test -p sparkflash-kv` | 14 binaries, 166 passed, 0 failed, 2 ignored |
| `cargo test -p sparkflash-device` | 5 binaries, 64 passed, 0 failed, 1 ignored |
| `cargo test --workspace` | 89 binaries, 1671 passed, 0 failed, 12 ignored |
| `cargo clippy -p sparkflash-kv -p sparkflash-device --all-targets -- -D warnings` | clean |
| `cargo fmt --all --check` | clean |

New tests: 16 in `sparkflash-kv/tests/paged.rs` plus 2 in-module invariant-corruption tests
(paged paging rules), 11 in `sparkflash-device/tests/paged_kv.rs` (mapping, staging,
window rotation, refusals, non-vacuous compact and absolute masks, ratio-2 expansion,
indexer block scoring, CPU twins). The workspace suite is run before hand-off and its counts are in the task
report, not this doc.

## 7. What is explicitly unclaimed

* The **device run** of a served sequence attending through `flash_mla_csa2_sm121` over paged rows:
  a window measurement, not run (node/GPU freeze).
* **HSI candidate selection** (full pool population and the top-512 merge): phase2 open item 2.
* **Path A** (positions-array kernel / paged-read kernel): named follow-on in Section 5; its ABI cost
  is stated there and no kernel, ABI or pin was touched.
* A paged store that mirrors a request's token sequence is not a claim that the serving KV is
  paged. The distinction is Section 1.1; the consumer that would make it a serving claim is the
  engine KV seam, not this module.
* Any numeric comparison against the device chain reference run: this milestone's evidence is the
  host test suite and the CPU twin only.
