# Paged-read attention kernel: design and rejection arithmetic (task-49)

**Status.** Design note plus CPU twin. No kernel source, no KernelId, no cubin, no baseline pin, no
ABI change, no device launch, no throughput claim. The twin is
crates/sparkflash-device/src/paged_read.rs and crates/sparkflash-device/tests/paged_read.rs; this
note records the argument set such a kernel would need, the cost of freezing it, and the byte
arithmetic that decides whether it is worth that cost today.

Base: main = 134b444 (batch 8). Path B (task-43) is the landed operand seam: the paged serving path
stages dense absolute-position rows and an absolute HSI key pool through the unchanged 19-word
flash_mla_csa2_sm121 entry. This note is the named follow-on: a kernel that reads the paged pool
directly instead of reading a staging copy.

## 1. What the paged-read kernel would replace

Path B today, per attention step, one sequence at absolute query position p (C = p + 1 held tokens):

* main KV: four source layers (layers 2, 8, 14, 20; ratios 2, 2, 2, 1) each stage C rows of 288 B
  (256 B packed FP4 codes + 32 B E4M3 scales) from the paged global pool into a flat array indexed
  by absolute token position. 1,152 C bytes per step.
* indexer K: paged_kv::absolute_indexer_keys builds the HSI pool [2048 blocks][8 slots][128 bf16
  channels] = 4,194,304 B, writes only tokens 0..=p, and zero-fills the rest. The kernel scores
  block b at absolute positions b*8 + slot. 4,194,304 B of HtoD staging per step, plus the host
  FP4 E2M1 / E8M0 to bf16 dequantisation of the written entries.
* window: window_rows gathers the layer's ring oldest first into 128 rows of 528 B; 40 layers
  attend, 2,703,360 B per step at C >= 128. (The ring rotation copy is device-store open item 1.)

The paged-read kernel keeps the arithmetic of all three stages identical and removes the staging
copies: it receives the layer's source array base pointer, the sequence's block table in device
memory, and the compression ratio, and computes the byte address of each attended entry itself.

## 2. The read path (the kernel's addressing, one line a reviewer can check)

For an absolute compressed position id t (an absolute token position, exactly what
paged_kv::absolute_source_rows produces a row for):

    logical_page = t / 64
    page         = block_table[logical_page]        (0xFFFFFFFF means unmapped)
    entry        = page * (64 / ratio) + (t % 64) / ratio
    codes        = paged_kv[entry * 356 + 0   .. + 256]
    scales       = paged_kv[entry * 356 + 256 .. + 288]

356 is sparkflash_kv::layout::GLOBAL_ENTRY_BYTES (288 main + 68 indexer K, interleaved), 256 is
MAIN_KV_SCALE_OFFSET, 64 is KV_PAGE_TOKENS. The current dense operands use stride 256 / 32; the
paged kernel uses one interleaved stride 356 and two field offsets. That is the entire addressing
difference, and it is exactly what crates/sparkflash-device/src/paged_read.rs implements and the
twin compares against the staging path.

The causal mask does not move: the union format already attends a candidate exactly when
candidate <= position, and the selected ids are absolute token positions. Path A's 20th positions
argument exists only because Path A keeps compact rows; once the kernel derives the absolute id from
page index and offset, the 20th argument is unnecessary. The block table also makes the HSI mask a
page-granularity question: an HSI block is 8 tokens and 8 divides 64, so a block never straddles a
page; block b is scoreable iff its first token is <= position and block_table[b*8/64] is mapped.
Partial tail blocks zero the slots past the query exactly as the staged pool does.

Block-table holes are defensive: paged::BlockTable maps every block the sequence holds, so under
the cache invariants no hole exists below the held token count; holes are free slots of the device
array (or a table copied for a preempted sequence). The kernel must treat 0xFFFFFFFF as unattended
rather than reading physical page 0xFFFFFFFF, and the twin asserts that masking (test
a_block_table_with_holes_is_masked_not_read).

## 3. The argument set versus the frozen 19/20-word params

Today flash_mla_csa2_sm121 takes 19 words (cuda_harness::FLASH_MLA_PARAM_COUNT = 19; the
m4_abi::SIGNATURES pin names each one). Path A adds a 20th, const uint32_t *positions, and keeps the
compact-row staging. A paged-read entry replaces three staged pointers (kv_fp4, kv_scales,
index_key) with four words (the layer's main source array, the indexer K source array, the block
table, and the compression ratio) and keeps the other sixteen unchanged. It is 20 words, the same
count as Path A, but it is a new symbol rather than a change to the frozen one.

Full paged main + HSI variant (20 words; this is the recommended shape if anything is frozen):

| # | word | type | replaces |
|--:|:--|:--|:--|
| 1 | q | const __nv_bfloat16 * | unchanged |
| 2 | main_kv | const uint8_t * | kv_fp4 + kv_scales (one interleaved array, stride 356) |
| 3 | index_k | const uint8_t * | index_key (dense 4 MiB bf16 pool becomes a paged pointer) |
| 4 | block_table | const uint32_t * | new: logical page -> physical page, 0xFFFFFFFF = hole |
| 5 | ratio | uint32_t | new: 1 or 2, the source layer's compression ratio |
| 6 | window_codes | const uint8_t * | unchanged (staged rows) |
| 7 | window_scales | const uint8_t * | unchanged (staged rows) |
| 8 | window_kv_format | uint32_t | unchanged |
| 9 | window_rows | uint32_t | unchanged |
| 10 | selected | const uint32_t * | unchanged (absolute token positions) |
| 11 | index_query | const __nv_bfloat16 * | unchanged |
| 12 | sink | const float * | unchanged |
| 13 | block_scores | float * | unchanged |
| 14 | block_best | uint32_t * | unchanged |
| 15 | out | __nv_bfloat16 * | unchanged |
| 16 | position | uint32_t | unchanged |
| 17 | selected_count | uint32_t | unchanged |
| 18 | trace_ring | GpuTraceRing * | unchanged |
| 19 | layer_id | uint32_t | unchanged |
| 20 | step_seq | uint64_t | unchanged |

Baked constants (not words, but a pin of their own): GLOBAL_ENTRY_BYTES 356, MAIN_KV_DATA_OFFSET 0,
MAIN_KV_SCALE_OFFSET 256, MAIN_KV_DATA_BYTES 256, MAIN_KV_SCALE_BYTES 32,
INDEXER_K_DATA_OFFSET 288, INDEXER_K_SCALE_OFFSET 352, KV_PAGE_TOKENS 64. The indexer K source is
always layer 20 (ratio 1), so its entry index is page * 64 + token % 64 and needs no second ratio
word; the main ratio word selects 32-entry (ratio 2) or 64-entry (ratio 1) pages.

The three partial variants fit the same 20 words or fewer, so no variant needs more than +1 word
over today:

| variant | words | change | what it removes per step |
|:--|--:|:--|:--|
| V0 today (Path B) | 19 | none | nothing (stages main + HSI + window) |
| Path A (task-43 follow-on) | 20 | + const uint32_t *positions | caps main staging at O(selected), not zero |
| V1 main paged only | 20 | kv_fp4/kv_scales -> main_kv + block_table + ratio | 1,152 C bytes |
| V2 HSI paged only | 20 | index_key -> index_k + block_table (+ index ratio baked) | 4,194,304 bytes + host dequant |
| V3 main + HSI paged | 20 | as the table above | 4,194,304 + 1,152 C bytes |
| V4 V3 + window ring read | 20 | window pointers become the ring bases; slot = token % 128 | 2,703,360 bytes more |

V4 adds no word because window_codes/window_scales already point at the layer's ring arrays and the
ring row for absolute token t is slot t % 128. It does change semantics (the kernel must reproduce
the oldest-first rotation the host gather does today) and is therefore a separate correctness
question, not covered by this task's twin; device-store open item 1 remains open.

## 4. ABI cost, named the way Path A's 20th argument was named

Path A's cost as recorded in docs/build/serving-paged-kv.md section 5: it changes FlashMlaParams
19 -> 20, i.e. the frozen M4 ABI surface (m4_abi signature table, the m4-abi-freeze CI job, every
launch site, the harness params mapping, the baseline pin), so it is a decision-record-level change
with its own reviewed pin, and its argument must be made against a measured throughput number.

The paged-read entry's cost, on the same terms:

* It is a new pinned entry, not a change to the existing one. That means a new declaration and
  m4_abi::SIGNATURES row (and the m4-abi-freeze CI job covers the new row), a new KernelId variant
  with its symbol and cubin name and catalog tests, and a new baseline pin: the baseline manifest
  18 -> 19 kernels (crates/sparkflash-kernels/baseline/kernel_baseline.txt), build.rs
  PROBED_KERNELS 18 -> 19, the check/regen scripts, and tests/kernel_baseline.rs. The existing 19
  launch sites are untouched, unlike Path A which recompiles all of them.
* It needs a typed params struct and launch wrapper with its own arity pin (the FlashMlaParams
  pattern), for the new launch site and the harness mapping used by a window run.
* It needs block-table device memory that does not exist today: paged::PagedKvCache is host
  metadata, and the device chain currently receives staged rows instead of a table. That is a new
  allocation per running sequence (capacity * 4 bytes; 4 B per 64 tokens = 0.0625 B/token), an
  upload when pages are allocated (8 bytes per 128 tokens), and invalidation/update on append,
  preempt, free and eviction. This plumbing, not the kernel, is the real implementation cost.
* It needs a layout pin: the kernel's baked 356/256/288/352/64 constants must equal
  sparkflash-kv::layout. A mismatch is not a crash, it is a silent read of the wrong bytes, so the
  freeze has to include a host test that compares the kernel's constants with the store's, in the
  same spirit as the signature pin.
* It needs a measured window before any throughput claim: the kernel's address arithmetic and the
  block-table indirection are new device work, and the copies they remove are only worth removing if
  a window line says so.
* Nothing in this task added any of the above: no KernelId, no source file, no baseline entry, no
  cubin, no ABI edit.

Cost comparison in one line: Path A pays one frozen-ABI mutation and gets a staging cap;
V3/V4 pay one additive pin plus block-table plumbing and get the staging copy removed. The byte
arithmetic in section 5 decides the case.

## 5. What it would buy: staging bytes, not arithmetic

The kernel would buy exactly one thing: the host-side staging copies disappear. It does not reduce
attention arithmetic, softmax work, bf16 rounding or the number of score evaluations; the operand
bytes are the same bytes (section 6), so every released reference produces the same output. It also
adds a little integer arithmetic (page table read, multiply, offset) per candidate and per HSI slot.
There is no FLOP saving and no claim of one.

Per-step staging for one sequence, in bytes, from the operand definitions in section 1. C is the
held token count; window rows are min(C,128) per layer over 40 layers.

| C | V0 Path B total | Path A (positions, 20th arg) | V1 main paged | V2 HSI paged | V3 main+HSI paged | V4 all paged |
|--:|--:|--:|--:|--:|--:|--:|
| 137 | 7,055,488 | 7,057,536 | 6,897,664 | 2,861,184 | 2,703,360 | block table only |
| 512 | 7,487,488 | 7,489,536 | 6,897,664 | 3,293,184 | 2,703,360 | block table only |
| 2,048 | 9,256,960 | 7,489,536 | 6,897,664 | 5,062,656 | 2,703,360 | block table only |
| 8,192 | 16,334,848 | 7,489,536 | 6,897,664 | 12,140,544 | 2,703,360 | block table only |
| 1,048,576 | 1,214,857,216 | 7,489,536 | 6,897,664 | 1,210,662,912 | 2,703,360 | block table only |

Breakdown and the arithmetic behind it:

* Main KV staging is 1,152 C bytes (4 source layers x 288 B x C). Path B pays it in full. Path A
  caps it at 4 x 512 x 288 = 589,824 B plus 2,048 B of positions, so it saves nothing until
  C >= 514 and is 2,048 B worse below that. V1 removes it entirely.
* HSI staging is a fixed 4,194,304 B per step: absolute_indexer_keys allocates the [2048][8][128]
  bf16 pool, writes tokens 0..=p and zero-fills the rest, then uploads it. At C = 137 the indexer
  actually reads 18 reachable blocks x 8 slots x 128 channels x 2 B = 36,864 B, so the staged pool
  is a 113.8x write amplification of the bytes the indexer consumes; the host also dequantises the
  written entries (137 of 16,384 slots at C = 137). V2 removes that.
* Window staging is 40 layers x 128 rows x 528 B = 2,703,360 B per step at C >= 128 (the ring
  rotation copy, device-store open item 1). V4 changes the pointers in place, so it adds no ABI
  word, but it needs its own correctness proof.
* The device reads the kernel performs are not claimed to shrink. For main KV, Path B writes 1,152 C
  bytes and the 40 attention layers then read selected rows from the staged arrays; the paged
  kernel reads the same rows from the pool instead. The saving is the host gather, the HtoD copy and
  the extra zero-fill, i.e. staging bytes and (if measured) staging time. Bytes are arithmetic here;
  time is not measured by this task and is not claimed.

At the window sizes this project has actually run (C <= 137 tokens per sequence), V1 saves
157,824 B/step = 2.2% of the step's staging; V2/V3 save 4,194,304 B/step = 59.4%;
V4 saves the remaining 2,703,360 B. Above C = 512 the main-KV line becomes the growing term
(quarter of the budget at C = 2,048, 58% at C = 8,192); at 1M tokens it is 1.125 GiB per step.

## 6. CPU twin: bit-identity against the dense staging path

crates/sparkflash-device/src/paged_read.rs is the kernel's addressing in host code: paged_entry,
read_main_rows and read_hsi_keys. crates/sparkflash-device/tests/paged_read.rs is the twin, five
tests, no GPU, no launch:

1. main_rows_across_page_boundaries_are_the_staged_rows_bit_for_bit - a 300-token sequence over
   four 64-token pages, layers 2 (ratio 2) and 20 (ratio 1). For every position 0..=299 the paged
   row is compared byte for byte with the row absolute_source_rows stages: codes and scales.
2. selected_candidates_across_a_page_boundary_attend_identically - selected ids at and across page
   boundaries (62, 63, 64, 65, 127, 128, 255, 299). The staged rows and the paged rows are equal
   bytes, and the released mla_ref::attention_with_sink_union output is equal bit for bit on both
   operand sets.
3. a_partially_filled_tail_page_reads_like_the_staging_path - 137 held tokens, so page 2 holds
   nine entries. The tail rows 128..=136 are byte-identical to the staged rows; the HSI key pool
   read through the same table equals absolute_indexer_keys value for value, including the two
   slots of block 16 that are past the query (zero), and mla_ref::indexer_block_scores is equal bit
   for bit on both pools.
4. an_evicted_and_reused_block_is_read_through_the_current_table - a sequence is freed and a new
   one reuses its page; the paged read follows the current table and equals the staged rows for the
   new owner. A two-page fixture then shows a stale table reading the evicted bytes (0x11) while
   the current table reads the owner's bytes (0x22): the block table, not the id order, decides.
5. a_block_table_with_holes_is_masked_not_read - a 0xFFFFFFFF slot is skipped, not read as physical
   page 0; the held rows still equal the staged rows; a hole below the query masks only itself; an
   id past the table fails closed; the HSI pool zeroes hole slots instead of publishing them.

This proves the addressing, not the device. It says nothing about the Tier 2 device FP4 E2M1 /
E8M0 to bf16 dequantisation (the host reference is on both sides here; the kernel would need its
own device twin like the F3 kernel half), nothing about V4's ring rotation, nothing about the
block-table upload and invalidation plumbing, and nothing about time.

## 7. Recommendation, with the arithmetic

Do not freeze a paged-read entry now, and do not freeze Path A either.

* Path A is rejected on bytes: it saves 0 B below C = 514 (it is 2,048 B worse because the
  positions list is extra), 1,767,424 B/step at C = 2,048 and 8,845,312 B at C = 8,192, while
  costing a mutation of the frozen 19-word entry (m4_abi signature, m4-abi-freeze CI, every launch
  site, harness mapping, baseline pin). V3 (the 20-word paged entry) saves 4,352,128 B/step at
  C = 137 and 6,553,600 B at C = 2,048 with no frozen-surface mutation, so at any context where
  staging matters the paged entry dominates Path A.
* A paged entry is not justified by the current windows yet: V1 removes 157,824 B/step (2.2% of
  staging) at C <= 137 and cannot pay for a new pinned entry plus block-table device plumbing; V3
  removes 4,352,128 B/step (61.7%) but needs both new address paths and the same plumbing; V4
  removes the remaining 2,703,360 B but needs the ring-rotation proof that is device-store open
  item 1. And no
  window line exists to turn any of those bytes into time, so there is no throughput claim to make.
* Trigger to revisit, stated in the measurements this note is missing: a window run at C > 512
  (main staging >= 589,824 B/step, growing 1,152 B per token per step) or a profile that shows the
  4,194,304 B HSI zero-fill at current C. The first thing worth freezing then is V3 (the 20-word
  main + HSI entry), because it is additive, it dominates Path A in staging bytes at every C, and
  the block-table plumbing is shared by V4 later. The freeze would need: the new pinned signature
  and m4-abi-freeze row, the KernelId and 18 -> 19 baseline pin, the typed params struct, the
  block-table allocation/upload plumbing, the layout-constant equality pin, and the measured
  window lines for the staging time it claims to save.

## 8. What this task did and did not change

Changed: docs/build/paged-read-attention-kernel-design.md (this note),
crates/sparkflash-device/src/paged_read.rs, crates/sparkflash-device/tests/paged_read.rs,
crates/sparkflash-device/src/lib.rs (module registration only).

Not changed: no kernel source under kernels/, no KernelId, no m4_abi signature, no baseline
manifest, no build.rs probe list, no cubin, no device chain launch, no serve path, no pin. The
paged-read kernel remains a design and a host twin; it is not runnable and no branch of the serving
path selects it.

## 9. Explicitly unclaimed

* No device run, no timing, no throughput, no latency, no memory-transient claim. The byte table in
  section 5 is arithmetic; the time column needs window lines this task does not have.
* No kernel compile, no cubin, no pin, no ABI surface change, no KernelId.
* Tier 2's device dequantisation is not proved equal to the host reference by this twin; it would
  need a device twin (the F3 pattern) before a freeze.
* V4's window ring direct read is not designed or tested here; ring rotation remains open item 1.
* Block-table upload, update on append/preempt/evict/fork, and bounds validation are design notes,
  not implemented or tested.
* Candidate selection and the top-512 merge remain phase2 open item 2; this note starts from
  selected ids and index-query values, as the released entry does.
