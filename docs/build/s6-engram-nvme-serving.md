# S6 host-side: Engram tables on the packed NVMe store (window-free)

**Status: task-24, host-side only, 2026-10-19.** Task-24 (*S6 Engram on NVMe in the serving path*,
[serving-parity-plan.md](serving-parity-plan.md) task board) is split in two: the device/KV
primitives belong to task-21 (kernel-dev), and the measured serving cost belongs to a window run. This page records only the window-free part: the serving path can now
read the two Engram tables from the packed dense store on disk, through the engram crate's
existing windowed reader, and the properties that follow are stated as host tests - which reads
happen, how many, at what granularity, and that the disk path produces the same logits as the
in-memory table for the same fixture.

Nothing on this page is a throughput or latency claim. Section 6 names every measurement a window
must supply and claims none of them.

## 1. What already existed, and what this slice adds

The engine has paged the Engram tables since the weight source gained a range interface:
`sparkflash-engine`'s `engram_forward` builds the rows a chunk actually looks up and asks
`WeightSource::read_ranges` for their bytes (`crates/sparkflash-engine/src/forward/engram.rs`,
`read_paged_table`). The serving path's checkpoint source, `ReferenceWeights`, answers those
requests straight from the safetensors shard (`crates/sparkflash-engine/src/loader.rs`), while a
whole-tensor read of the 98 GB table is refused by name. So "paging" was not missing.

What was missing is the *store*: the tables' serving bytes came from the checkpoint shards, not
from the packed dense rank files that `sparkflash pack-engram` exists to produce, and the packed
store's reader (`sparkflash_engram::store::DirectTable`: `O_DIRECT` windowed `pread`, mmap
fallback, merge gap and I/O cap) was not on the serving path at all. This slice puts it there:

- `sparkflash serve --checkpoint DIR --engram-dir TABLES` wraps the checkpoint source in
  `DiskEngramWeights` (`crates/sparkflash-cli/src/serve.rs`). Every tensor except the two
  Engram embedding tensors is delegated unchanged; a whole-tensor read of those two is still
  delegated and still refused.
- The translation from a range request to packed rows is `sparkflash_engram::paged::PagedTable`
  (`crates/sparkflash-engram/src/paged.rs`): a new, small module (about 200 lines of logic) over
  the existing reader, with no second file-format implementation.
- The packed store directory is the packer's own output: `manifest.json` plus the rank files.
  Every rank named by the manifest is opened, because one reference process has no rank of its
  own and a chunk's hash ids can fall in any rank's row range. A rank-local or device-side
  placement is not claimed here.

## 2. The row translation

The checkpoint stores a layer's table as two tensors: `layers.N.engram.embed.weight` of shape
`[rows, 256]` F8E4M3 and `layers.N.engram.embed.scale` of shape `[rows, 8]` E8M0. The packer
stores one dense row of `264 = 256 + 8` bytes per table row, weight bytes first
(`docs/DECISIONS.md` D4; `crates/sparkflash-engram/src/layout.rs`).

The engine asks for whole rows: range `(id * width, width)` on the weight tensor and
`(id * scale_width, scale_width)` on the scale tensor, with `scale_width = width.div_ceil(32)`
and the *same* ids in both requests because `read_paged_table` calls the two tensors in
sequence for one chunk. The translation is therefore:

| engine request | packed-store read | bytes returned |
| --- | --- | --- |
| weight range `(id*width, width)` | row `id` | `row[0..width]` |
| scale range `(id*scale_width, scale_width)` | row `id` | `row[256..256+scale_width]` |

Three properties are enforced rather than assumed, so a request cannot be silently mis-sliced:

1. A range is exactly one whole row: `len` equals the tensor's row width and `offset` is a
   multiple of it.
2. The row width fits the packed row: `width <= 256`, `scale_width <= 8`, and
   `scale_width == width.div_ceil(32)`. The released configuration uses the equality case
   (256/8); a smaller width is accepted only so a host fixture can use the tiny configuration.
3. A named row is owned by exactly one rank file of its layer. Zero owners and two owners are
   errors, never a guessed file.

Rows owned by different rank files in the same request are grouped per file and read with one
batched `RowTable::read_rows` call per file, preserving request order in the returned bytes.

## 3. What the serving path reads

Per Engram layer per prefill chunk, the engine makes **two range requests**: one for the weight
rows, one for the scale rows (`read_paged_table` in `forward/engram.rs`). The packed source turns
each into **one batched row read per owning rank file**; it does not read row by row and it does
not read the whole table.

The counts are observable without a device: `DiskEngramWeights` carries atomics
(`weight_requests`, `scale_requests`, `rows_requested`, `read_batches`, `bytes_requested`;
`crates/sparkflash-cli/src/serve.rs`), and the tests below assert them. The reader's own
system-call count is `RowTable::read_ops`; because each request is one batched call, that count is
the reader's window-merge result, not the row count.

There is deliberately **no cache between the weight request and the scale request**: the same rows
are read twice (second time normally from the page cache), which is the read amplification section
6 says a window must measure. A cache here would hide the count the host tests and the window are
meant to observe.

## 4. Wiring contract

- `--engram-dir DIR` requires `--checkpoint`. It is refused with `--synthetic`: the synthetic
  plan already carries its tiny table in memory and has no checkpoint for a packed store to be
  rows of.
- The manifest is read at engine build time, on the engine thread, and must agree with the
  compiled geometry (`specs_from_manifest`) and with the served configuration: every
  configured Engram layer must have a packed table with the configured logical row count, and the
  packed store may not hold a layer the configuration does not declare. A mismatch is a start-up
  refusal naming `--engram-dir`, not a mid-request surprise.
- The reader prefers `O_DIRECT` and falls back to the mapping; the fallback reason is printed on
  start-up, as is each opened table's backend and window. Nothing claims `O_DIRECT` engaged on a
  host where the filesystem refuses it.
- The source is `Send + Sync` because it consists of the boxed inner source, `DirectTable` and
  atomics; the engine's thread-safety assertion is unchanged.

## 5. Host-testable properties (the tests)

`crates/sparkflash-engram/src/paged.rs` (unit tests, no files):

- weight ranges come back in request order, one row read per requested row, and nothing outside
  the request is read (`MemoryTable::read_ops`);
- scale bytes come from the scale half of the same row;
- a file's `row_start` is translated to a local row;
- a request spanning two rank files becomes one batch per file and preserves order;
- a row outside every file, two files claiming a row, a non-whole-row range, an over-wide
  width/scale and an undeclared layer are refused;
- the released geometry (256/8) is exactly the prefix-plus-scale slicing.

`crates/sparkflash-cli/src/serve.rs` (`engram_packed_store` tests, temporary files):

- **Same logits.** A 1,024-row packed fixture is built from the tiny configuration's real Engram
  table (obtained from `SyntheticWeights`, then re-packed into 264-byte rows). An engine over
  that fixture with `DiskEngramWeights` and an engine over the in-memory table produce the same
  logits, the same head input and the same next token **bit for bit** for the same prompt. The
  in-memory source is wrapped in a counter that refuses and counts any whole-table Engram read;
  the count is `0`, so the disk result cannot have come from a fallback whole-table read.
- **Which reads and how many.** For that one prefill: one weight range request, one scale range
  request, two batched row reads (one per request; one rank file in the fixture), the same
  distinct row count in both requests, at most `tokens x columns` distinct rows, and
  `bytes_requested` exactly `rows x (width + scale_width)` - the rows' bytes, not a table.
- **Granularity.** Each range request is a single `read_rows` batch; the test asserts
  `read_batches == 2` for two requests. The reader's window size (512 bytes in the fixture,
  4,096 by default in the store) is configuration, and the real system-call count is recorded by
  `read_ops`.
- **Refusal stays honest.** A whole-tensor read of `layers.1.engram.embed.weight` through the
  packed source still reaches the inner source's `Refused` error.
- **Name parsing.** Only `layers.N.engram.embed.weight` and `layers.N.engram.embed.scale` are
  intercepted; every other tensor name is delegated.
- **Wiring.** A recipe with an Engram directory reaches `build_engine`, and a directory without
  `manifest.json` is refused with an error naming `--engram-dir` and `manifest.json`.

Verification for this slice: `cargo fmt --all -- --check`, `cargo check --workspace
--all-targets`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test
--workspace` on the branch tip. No GPU, no window, no released packed store was available; the
tests run on temporary files of a few hundred kilobytes.

## 6. Measurements a window must supply (none claimed here)

The original task-24 acceptance was measured read amplification, bandwidth and per-token overhead
on the serving workload. The host slice cannot measure any of them: there is no NVMe device, no
page-cache state, no concurrent serving and no device clock in this environment, and the reference
engine has no throughput claim of its own. The window must measure, at least:

1. **Cold and warm read latency per chunk and per layer.** Wall time of the two range requests a
   chunk makes, with the packed store on a cold page cache and then warm; separately for the
   weight and scale requests.
2. **Per-token stall.** The share of one decode/prefill step's wall time spent waiting on the
   packed-store reads, including the second (scale) request when it does not hit the cache, and
   its p50/p99 across a serving trace.
3. **Read amplification.** Bytes fetched from the device divided by `bytes_requested` recorded
   by `EngramPagingStats`, as a function of `window_bytes`, `merge_gap_bytes` and
   `max_io_bytes`; and the weight-plus-scale double read, before and after any cache.
4. **Bandwidth and concurrency.** Achieved `O_DIRECT` bandwidth for the row size and window sizes
   of the real workload, including under several in-flight requests; the reference engine
   serialises, so this needs the batched/device serving path, not this host path.
5. **`O_DIRECT` versus the mmap fallback.** The measured cost of the fallback this reader takes
   when a filesystem refuses direct I/O, on the same files.
6. **Rank locality.** The cost when the rows a chunk names live in the local rank file versus a
   remote rank file; this host path opens every rank file, and a real deployment has to decide the
   placement before the number means anything. TP=4 is the serving shape
   ([serving-parity-plan.md](serving-parity-plan.md) addendum), so the window's store is the
   per-node rank file the driver owns, not all four.
7. **End-to-end effect.** TTFT and steady-state tokens/s of the serving workload with the packed
   store versus the checkpoint-shard paging path, with the same model and request trace.

Until those numbers exist, this page claims only the read *shape* (which rows, how many, one batch
each) and the equality of the arithmetic, not their cost.

## 7. Boundaries

- No KV or device code is touched here: task-21's paged format-2 device KV and its primitives are
  kernel-dev's, and the Lead opens the paged serve wiring separately.
- No frozen protocol doc is edited by this slice.
- The 188 GiB packed store is not present in this checkout; a released-shape packed run remains a
  window item (the manifest geometry is the compiled released geometry, so the host tests use the
  tiny configuration's row count over the same 264-byte row layout).
