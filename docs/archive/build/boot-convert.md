# Fast boot: pre-converted per-rank files and parallel direct I/O

**Status: implemented, host-gate green, no node measurement.** This page records the boot-path
change on `tp4_resident` / `tp4_host` / `tp4_chain`: `sparkflash convert` writes one contiguous
per-rank file already in device layout, and the boot loader reads it in parallel `O_DIRECT` chunks
straight into a zero-copy unified-memory slab. Every number below that is not a compile-time
constant is labelled as an expectation, not a measurement: nothing here has run against the
released checkpoint on a GB10 node yet.

## 1. What the old path did

Per rank, `upload_arena_streaming` walked the plan one tensor at a time on one thread:

1. `shard_plan.read(source, name)` seeked to each byte range of the shard and read it into a **fresh
   `Vec<u8>`** — one allocation per tensor, for ~96k tensors.
2. B12x weights were **re-tiled on the CPU on every boot** (`tile_b12x`), and every tensor was
   **re-digested on the CPU** (`fnv1a`) to prove the device copy was the checkpoint slice.
3. Each tensor was then `cuMemcpyHtoD`-ed into the arena. On a GB10 unified-memory node the source
   and destination are the same DRAM, so that copy bought nothing.
4. `MemAvailable` was re-read before **every** tensor.

Two further facts frame the numbers: `/` is f2fs at 94 % full, so a seek-per-range read pattern on
it is dominated by fragmentation, and the routed-expert `w2` shard was bound but not launchable in
the 640-K A8 layout, with a boot repack quoted at **+25.2 GiB/rank**
(`resident_weights.rs`, and `moe-stored-layout-variant-plan.md` §4).

## 2. The six fixes, and which ones this change makes

| | Fix | State |
|---|---|---|
| 1 | Offline `sparkflash convert`: per-rank contiguous files already in device layout | **implemented** |
| 2 | Parallel direct I/O: `O_DIRECT`, 16 MiB chunks, 8 reads in flight, into the final buffer | **implemented** |
| 3 | Unified memory: host-mapped slab / registered `mmap` instead of managed + H2D | **implemented** |
| 4 | Storage: get `/` under ~80 %, XFS/ext4 NVMe, defragment, `fio` baseline | **not code** (operational) |
| 5 | Remove per-tensor overhead: headers parsed once, one MemAvailable check, one allocation | **implemented** |
| 6 | Load all four ranks in parallel, barrier only after every fill finishes | **implemented** |

## 3. The converted file format

`rank-{rank}-of-{world}.bin`, little-endian throughout.

| offset | bytes | content |
|---|---|---|
| 0 | 8 | magic `SPKFRANK` |
| 8 | 4 | version (`1`) |
| 12 | 4 | rank |
| 16 | 4 | world |
| 20 | 4 | reserved |
| 24 | 8 | `index_len` |
| 32 | 8 | `data_offset` (4096-aligned) |
| 40 | 8 | `data_bytes` (logical arena bytes) |
| 48 | 8 | `padded_data_bytes` (4096-aligned) |
| 56 | 8 | `header_hash` = FNV-1a 64 of the index bytes |
| 64 | `index_len` | JSON index |
| `data_offset` | `padded_data_bytes` | arena payload, tensor order = arena order |

The JSON index (`ConvertedRankHeader`) carries `plan_hash`, `stored_bytes`, `payload_bytes`,
`allocated_bytes`, `file_digest`, one `ConvertedGroupEntry` per `ResidentGroup` sub-slab and one
`ConvertedTensorEntry` per tensor: `name`, `group`, `slab_offset`, `group_offset`, `bytes`,
`stored_bytes`, `layout`, `dtype`, `stored_shape`, `shard_shape`, per-tensor FNV-1a `digest`.

Three properties do the work:

- **`slab_offset` is the arena offset.** The payload is the arena, byte for byte, so the loader
  makes one allocation and one sequential read; there is no per-tensor placement at boot.
- **Device layout is baked in.** Dense FP8 weights are written already B12x-tiled
  (`TensorLayout::B12x`); routed `w2` weights and scales are written already padded to 640-wide K
  (`TensorLayout::MoePadded640`, 320 + 20 bytes per row). Boot does no tiling and no padding.
- **Scales sit next to their weights.** `order_scales_next_to_weights` emits `foo.weight` then
  `foo.scale`, so a bound pair is one contiguous run and the scale grid is in the same page window
  as the codes it scales.

`data_offset` and `padded_data_bytes` are 4096 multiples, and every tensor offset is 256-aligned
inside the slab, so `O_DIRECT` and `cuMemHostRegister` are always legal on the payload region.

## 4. The boot loader

`load_converted_rank_arena` does, in order:

1. **Cheap check** — read the 64-byte preamble, compare `st_size` against
   `data_offset + padded_data_bytes`, and verify `fnv1a(index_bytes) == header_hash`. This is a few
   kilobytes of I/O; it catches truncation and a stale/mismatched file index without touching a
   weight byte.
2. **One allocation** — `AllocationMode::UnifiedSingle` (host-mapped slab via `cuMemHostAlloc`
   with `PORTABLE|DEVICEMAP`, falling back to `cuMemAllocManaged`) or
   `AllocationMode::MmapRegistered` (`mmap` + `cuMemHostRegister` + `cuMemHostGetDevicePointer`,
   for warm restarts served from page cache).
3. **Parallel read** — `read_file_parallel_into_slice` opens the file with `O_DIRECT`, splits the
   payload into 16 MiB chunks, and issues 8 concurrent `pread`s from a scoped thread pool, each
   writing directly into the slab at its final offset. No intermediate buffer exists. A chunk whose
   offset, length or destination is not 4096-aligned, or a filesystem that rejects `O_DIRECT`,
   falls back to the buffered descriptor for that chunk only.
4. **Opt-in re-digest** — `--verify-weights` re-reads and verifies every tensor digest plus
   `file_digest`. The default boot trusts the header hash and the conversion-time digests.

## 5. The MoE 640-K problem, moved offline

`sparkflash convert` pads each expert's `[5120, 288]` code slice to `[5120, 320]` and its
`[5120, 18]` scale slice to `[5120, 20]` once, on the conversion host, using `pad_moe_rows_640`
(zero fill, which is what the A8 layout's 64 pad K values are). That converts the +25.2 GiB/rank
from a **boot repack in device memory** into **+25.2 GiB/rank of file bytes**, and makes
`routed_experts_down` launch-compatible against the incumbent `b12x_mxfp4_moe_sm121` without the
stored-layout kernel. `BindCoverage` now reads the layout off the bound slice and reports
`launch_compatible: true` with that reason when the arena came from a converted file, and keeps the
`launch_compatible: false` reason (with `sparkflash convert` as the remedy) when it did not. The
stored-layout kernel (`b12x_mxfp4_moe_stored_sm121`) remains the no-extra-disk-bytes route; this is
the no-kernel-change route.

## 6. Parallel ranks, barrier after fill

The sequential-across-ranks behaviour was a transport-ordering artifact, not an I/O one:

- `Tp4Rank::open` connected the collective **before** `open_input` built the resident cache, so a
  rank sat in `TcpTransport::connect` while its peer was still reading.
- `tp4_chain` opened both forward transports before the arena epoch.

Both now take the transport lazily. `Tp4Rank::open_input_with_transport` builds the plan and fills
the cache first, then calls the closure that connects; `tp4_chain` opens the walk and forward-loop
transports inside the bind/forward closure. All four ranks therefore fill in parallel, and the
collective Hello (and the control hub's Ready) happens only after every rank's fill has finished.

## 7. Per-tensor overhead removed (fix 5)

- `SafeTensorsSource::preload_headers` parses every shard header exactly once; `File` handles are
  cached per shard (`files: Mutex<BTreeMap<String, Arc<File>>>`), so `read_range` uses
  `read_exact_at` on a cached handle instead of `open`/`seek` per range.
- `padded_refusal`-style per-tensor `MemAvailable` syscalls are gone: `check_mem_available_once_for_plan`
  compares one reading against the plan's total resident bytes plus the floor, before the arena is
  allocated.
- `ShardReader::read_shard_into` writes into one reused scratch buffer (sized to the largest stored
  tensor) instead of allocating a `Vec<u8>` per tensor, and `ShardPlan::read_into` validates the
  destination length against the planned geometry before the read.

## 8. How to use it

```
# once, offline, on a machine with the checkpoint and disk headroom
sparkflash convert --checkpoint /ckpt/released --output-dir /nvme/converted --tp-world-size 4
# writes rank-0-of-4.bin .. rank-3-of-4.bin + sparkflash-converted-manifest.json

# cheap boot check on a node
sparkflash convert --inspect /nvme/converted/rank-0-of-4.bin

# boot from converted files
SPARKFLASH_CONVERTED_DIR=/nvme/converted sparkflash device-chain --upload ...
# optional: re-digest every tensor (slow), or mmap+register instead of reading
... --verify-weights
... --mmap
```

`sparkflash convert` also accepts `--tp-rank R` / `--ranks 0,2`, `--no-pad-moe-640`, `--no-verify`.
The loader looks in `SPARKFLASH_CONVERTED_DIR` first, then next to the checkpoint; if neither holds a
`rank-{r}-of-{w}.bin` for the rank, it falls back to the safetensors streaming path with no
behavioural change.

## 9. What is and is not proven

- **Proven on the host (unit tests):** the converted file round-trips through
  `inspect_converted_rank` / `verify_converted_file`; a converted `w2` weight and scale carry the
  640-wide `[rows, 320]` / `[rows, 20]` shapes and lengths; scales land immediately after their
  weights in the index; the arena loaded from a converted file exposes the same slice bytes and
  digests; the parallel reader reassembles a file exactly at 4 KiB and 16 MiB chunk sizes;
  `pad_moe_rows_640` and `order_scales_next_to_weights` are covered.
- **Not measured:** boot wall-clock on a GB10 node, `O_DIRECT` throughput on the node's filesystem,
  page-cache-warm `mmap` restart time, and the end-to-end 10×. Those need a node window; fix 4
  (storage occupancy/`fio`) remains operational and unaddressed here, and it caps whatever the rest
  of this change can deliver on a 94 %-full f2fs `/`.
