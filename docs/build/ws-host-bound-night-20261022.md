# The host-bound step, measured again: the ~490 ms mode is machine state, and the base binary is not main

Date: 2026-10-22 (night session, 02:00-04:00 America/New_York). Author: Lead.
Status: measurement record. No code in this file is a claim of a speed win; the only published
number is still 249.25 ms/step (wsd6-lean, commit 3813ecbe).

## 1. What the night's rows say, in one table

Frozen-argv four-rank lean rows (`run-fourrank.sh <tag> <run> token`, 9 decode steps,
`--forward-lean`). Identity (tokens + logits bin sha `aed6054d...`) PASSED on every row below
except where noted; the rows differ in wall time only.

| row | binary | decode step samples (ms) | mean | clean-step band |
| :-- | :-- | :-- | :-- | :-- |
| wsd6-ctl1 | 3813ecbe (published) | 248,245,244,241,248,247,242,246,239 | 244.4 | 239-248 |
| wsd6-lean5 | 3813ecbe | 249,251,248,247,244,242,251,247,248 | 247.4 | 242-251 |
| wsd6-ctl2 | 3813ecbe | 249,244,244,255,291,287,250,254,248 | 258.0 | 244-255 (+2 at 287-291) |
| main-lean1 | 268f2ab1 | 504,492,493,497,505,521,500,497,496 | 500.6 | none |
| main-lean2 | 268f2ab1 | 261,262,264,491,503,331,266,258,265 | 322.3 | 258-266 |
| wsd13-lean | main + comm | 488,498,497,335,265,262,265,266,259 | 348.3 | 259-266 |
| wsd13-lean2 | main + comm | 493,501,502,512,505,507,397,265,261 | 438.1 | 261-265 |
| wsd16-lean1 | Batch 4, warm on | 491,488,489,487,485,487,499,486,491 | 489.2 | none |
| wsd16-nowarm | Batch 4, warm off | 253,253,426,499,482,321,257,258,256 | 333.9 | 253-258 |
| wsd17-tcp1 | Batch 4 + RoCE slot, tcp | 252,251,436,479,489,480,484,482,499 | 428.0 | 251-252 |
| wsd17-tcp2 | same, tcp | 481,484,485,486,484,485,486,487,321 | 466.6 | none |
| wsd17-roce1 | same, RoCE | 576,549,586,561,562,564,575,559,568 | 566.7 | none |
| wsd17-roce2 | same, RoCE | 392,395,401,393,393,388,392,393,403 | 394.4 | none |

Two facts follow directly and they are not about any branch:

1. **The published binary is not main.** 3813ecbe (the 249.25 row) measures 239-251 clean tonight;
   every binary built from 268f2ab1 measures 258-266 on its clean steps. The base gap is ~+15
   ms/step, and on prefill it is ~+70 ms/position (prefill medians: wsd6 231-234, main-based
   298-318). `!158` is 2892 lines of loader/arena/arena-layout change over 3813ecbe and is the only
   candidate between them.
2. **The ~490-500 ms step is not a code property.** The same binary (4d87d1cd, tag wsd12) measured
   266-286 at 01:37-01:43 and 500.6 at 06:24; the same Batch-4 binary measured 489 (`wsd16-lean1`)
   and 253-258 on clean steps (`wsd16-nowarm`) eight minutes apart with only an env switch between
   them. A row that lands in this mode reports the mode, not the branch.

## 2. The ~490 ms mode: what it is, measured

While a row sat in the mode, on rank 0 of 192.0.2.80:

- Disk: **1.29 GB read per 1.5 s (~860 MB/s) sustained through the decode loop**, `read_bytes` equal
  to `rchar` (every byte a cache miss).
- CPU: **40 ticks / 1.5 s = 0.27 cores**, i.e. the host is waiting, not computing.
- Faults and swap: `pgmajfault = 0`, `pswpin = 0`, process `VmSwap` 28 MB, `VmHWM` 76 GB (the
  arena is `cuMemHostAlloc` page-locked host-mapped memory, so it cannot be the thing paged out).
- Memory: `MemFree` 773 MB, `Cached` 120 GB, `MemAvailable` 44 GB, `shared` 72 GB.

So the step is not slow arithmetic and not swap: the loop is re-reading several hundred MB per step
from the block device. The bytes are the checkpoint pages behind the host seams (the reference
compressor/indexer reads and the paged Engram rows), i.e. exactly the reads the buffered loader of
3813ecbe left in the page cache and the O_DIRECT loader of `!158` does not.

**State is controllable without root.** `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on the
checkpoint shards as the owning user moved `MemFree` from 864 MB to 40 GB
(`/tmp/dropcache.py`, run per node). That is the row-protocol fix this record proposes:
**drop the page cache before every row**, so a row starts from a defined memory state instead of
from whatever the previous row left behind.

## 3. What each branch measured, and the verdict

- **Batch 4** (module cache + evidence gating + indexer wq_b on the pinned GEMV + comm reduce lean
  + warm pass, integration/wsd15, MR !164): clean steps 251-258 against main's 258-266, i.e. ~-5 to
  -8 on the base it is built from, and **not** an improvement on the published 249.25 because the
  base itself is +15. Not publishable yet; the slices' estimates (8-20 + 1.4-2.4 + ~9.5 + 0.8-2.4)
  were made against the host-share table, and the host-share table came from a different base.
- **Warm pass** (576f0170): the host-only probe row with the pass on measured 281.7 ms/position
  against main-era 310.4 (-28.7), and it moved the position-0 cold burst into setup (-46 ms). But
  `wsd16-lean1` (pass on) sat in the 490 mode while `wsd16-nowarm` (pass off, same binary) was
  clean, so the pass is **opt-in only** in the batch until the markers attribute it
  (`SPARKFLASH_SEAM_WARM=1` enables; commit 3022e692).
- **RoCE canonical reduce** (4afa16ec): the slot works and proves itself on every run
  (`forward reduce: requested="roce" path=roce (every rank agreed) ... roce_calls=11040
  tcp_calls=0`; the TCP arm prints `path=tcp ... tcp_calls=11040`), and the bit-identity tests
  pass. But on the fabric through the forward it measured **394-566 ms/step against TCP's 428-466 in
  the same window**, i.e. no win; the 35.70 us per-call figure from the WS-B run does not reproduce
  through this slot. Default stays TCP; the switch stays (it is safe and it falls back).
- **comm reduce lean** (ca9840fa + gating 8a7e811c): the isolated bench win (381-566 us -> 113-126
  us per call on loopback) does not show in the row, because on a LAN the reader threads are kept by
  the gating; clean steps are unchanged against main. Safe, no win, no loss.

## 4. Rules this record adds

1. **A row must start from a defined memory state.** Drop the checkpoint page cache (fadvise
   DONTNEED, no root needed) before each row; record `MemFree`/`Cached` in the row metadata.
2. **Report the clean-step band and the median, not only the mean.** One 490-mode step moves a
   9-step mean by ~25 ms; a row with three of them can be 200 ms off.
3. **Never compare across base commits.** wsd6 (3813ecbe) is not main (268f2ab1); a branch built on
   main must be compared against a main row in the same session, and a published number can only
   come from the same base the MR will land on.
4. The mode is a **deployment defect**, not just a measurement nuisance: a fresh process on a
   pages-cold node will pay 400+ MB/step of disk reads. The fix shape is the warm pass done right
   (cache the seam tensors in process memory, not in page cache) or a loader that leaves the
   seam-touched pages resident; both are open.

## 5. Next levers, in measured order

| lever | class | measured or priced |
| :-- | :-- | :-- |
| fix the 490 mode (seam tensors resident) | (a) removable | up to ~240 ms/step in the mode; ~15 ms/step of base gap always |
| the host-share table itself: moe_reduce 28.2, attention_output 27.4, ffn_seams 21.1, ref_seam 20.8, stage_attention 13.2, mhc_entry 13.0, attention_projections 10.4 ms/step | (a)+(b)+(c) | from the wsd12 host-only probe (host 158 / wait 118 / kernel 89) |
| RoCE reduce re-priced per call inside the forward | (b) | 35.7 us fabric figure did not reproduce; needs a per-call meter in the slot |
| compressor wkv/wgate + indexer wk/weights_proj bf16-weight GEMV instantiation | (b) | needs a new cubin + repin; the seam plan prices ref_seam -11..-15 |

The published number is unchanged at **249.25 ms/step**; tonight added no publishable improvement,
three measured negatives (comm lean neutral, RoCE slower, warm pass unproven) and the two facts
above, which are worth more than the rows they cost.

## 6. The clean row protocol, and the warm pass measured in it

The protocol from rule 1 was applied: `posix_fadvise(DONTNEED)` on the checkpoint shards on all
four nodes, then one frozen-argv lean row. Two effects, both large:

**(a) The rows become reproducible.** Same binary (wsd17 sha `96cea90d...`), same argv, cache
dropped before each row:

| row | decode step samples (ms) | mean |
| :-- | :-- | :-- |
| wsd17-nowarm  | 251,254,254,253,250,254,254,255,250 | 252.78 |
| wsd17-warmclean (warm pass ON) | 224,228,228,232,230,230,231,230,231 | **229.33** |
| wsd17-noarmclean (warm pass OFF, minutes later) | 255,258,257,254,256,254,250,255,257 | 255.11 |
| wsd6 (published binary), same window | 254,247,249,248,249,249,250,248,246 | 248.89 |
| wsd6 (published binary), second row | 251,252,251,245,259,264,253,249,258 | 253.56 |
| wsd17 batch, second row | 252,254,254,259,257,253,256,254,255 | 254.89 |

The whole 490-mode disappeared for every row in the window: 15 consecutive rows, no stalls, bands
within 2-10 ms of each other. **The ~490 ms mode is therefore a page-cache-state artefact of the
row protocol, not a property of any binary.** These rows are the first in this programme that are
comparable across binaries, and they remove the "same binary, 266 and then 500" ambiguity.

**(b) The warm pass is a win, and it is the base-gap fix.** In the same clean window, same binary
`96cea90d`, only `SPARKFLASH_SEAM_WARM` differing: 229.33 with the pass, 255.11 without, i.e.
**-25.8 ms/step**, and 229.33 is below the published 249.25 (commit 3813ecbe) even though the batch
is built on main, which is +15 ms/step slower on its clean steps. Identity gate PASS on the warm
row: 4/4 ranks rc=0, tokens exact, logits bin sha `aed6054d77d6704c...`.

Mechanism, consistent with section 2: the pass reads and holds the seam tensors in the process's
own memory (the `CachedCheckpointSource` map) before the position loop, so the loop does not have
to fault them out of a page cache the O_DIRECT loader never filled. On a pages-cold node this is
the difference between ~400 MB/step of disk reads and none.

**RoCE, measured in the same clean window**: 392,396,392,394,391,398,399,395,398 = 395.22 ms/step
against TCP 252.78 on the same tag and argv. The slot is correct (path=roce, 11040 calls, bit
identity) and costs **+142 ms/step** on this fabric. Rejected; the switch stays, default stays TCP.

### Protocol, final form

1. `python3 - <<'PY' ... os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) ... PY` over the
   checkpoint shards on all four nodes (`/tmp/dropcache.py` in the session tools), before each row.
2. Zero `sparkflash-*` processes and MemFree >= ~110 GiB before launching.
3. Record the clean band and the median, not only the mean.
4. Two rows, interleaved, when the delta under test is < 20 ms; one row is ±5 ms.
5. Never compare across base commits; the published row must be re-run in the same window as the
   candidate when either side changed base.

## 7. Where the 27.8 ms of the warm pass actually is (host-only probes, final tree)

Two host-only probe rows on the final tree (wsd18, 7ecec4c2), decode positions only, cache dropped
before each, per call and per step:

| block | warm off host/c | warm on host/c | calls/step | delta/step |
| :-- | :-- | :-- | :-- | :-- |
| `engram_write` | **12.260** | **1.281** | 2 | **-22.0** |
| `ref_seam` | 0.310 | 0.270 | 40 | -1.6 |
| `attention_output` | 0.515 | 0.467 | 40 | -1.9 |
| everything else | within 0.01 of the warm arm | | | 0 |
| step, host-only wall | 257.5 | **232.2** | | -25.3 |

So the pass is not a general "make the seams faster" effect: **-22 of the -28 ms is the Engram paged
row reads**, which the warm pass moves from the position loop (where each of the two per-position
paged reads cost ~11 ms of host file I/O on a pages-cold node) into setup. The remaining ~-4 is the
seam tensors and the noise floor. That is also the answer to "why is main +15 against the published
binary": the loop's paged Engram read is the same read that 3813ecbe served from a page cache its
loader filled, and `!158` does not fill it.

Next lever, from the same table (host/c x calls/step on the warm arm): ffn_seams 20.8, attention_output
18.7, stage_attention 13.0, mhc_entry 12.8, moe_reduce 12.4, ref_seam 10.8, attention_projections 9.3,
query_path 4.8. Note `moe_reduce` is already 0.309 against the 0.706 of the wsd12 probe (the gated
reader threads) and `ref_seam` 0.270 against 0.521 (warm + the pinned GEMV), so those two levers are
partly spent.
## The mHC device port: rejected by the row (+58.5 ms/step), and why the mechanism could not work

Branch `task/mhc-dev2` @ `0f7c59f6` moved the mHC coefficient projection (`hc_projection`: 24 rows
of 20480 f32, one ascending-k chain each, 637 us of the 904 us `mhc_entry` and of the 1358 us
`ffn_seams` on the dev box) into a new pinned kernel `mhc_hc_proj_sm121` with lane j = row j, so the
step's arithmetic was the host chain's own order and rounding (0 FMA in the PTX, no spills). Host-side
it was clean: engine split into a bit-identical projection + tail pair with a test, a CPU twin oracle at
24x20480 and six smaller widths, all 24 baseline pins reproducing, fmt/clippy/tests green.

The row said no, twice:

| row (tag wsd24 = the merge on the 213.78 tree) | ms/step |
| :-- | :-- |
| wsd24-mhc1 (device arm on) | 272.33 |
| wsd24-mhc2 (device arm on) | 272.22 |
| wsd24-refhost (same binary, `SPARKFLASH_SEAM_REF_HOST=1`, arm on the host) | 225.89 |
| the tree it was merged onto (wsd23-stage1/stage2) | 213.78 / 216.08 |

So the arm costs **+46.4 ms/step** against the same binary with it switched off, ~+1.16 ms per seam
call against the +0.05-0.06 it was priced at.

**The mechanism, and the rule it leaves behind:** lane-as-row is the one layout that removes the
concurrency the host relies on. Every lane's chain is serial - each add waits on the previous add, and
each step needs a load whose latency cannot be hidden behind the next dependent step (the weight rows
are ~1.9 MB per call, L2-resident at best). 20480 dependent loads at ~200 cycles is already
milliseconds, twice per layer. The host wins because 24 rows are 24 *independent* chains, so its
out-of-order window hides the load latency across rows. Generalisation for this track: **a per-call
host block is only a device candidate if its parallel dimension can be mapped to lanes without
splitting any sum; when the natural map is "one row per lane" and each row is a serial reduction, the
GPU is the slower machine.** The same reasoning is why `engram_write` (1.307 ms/call x2) should not be
moved as-is either.

The constructive reading: keep the projection on the host and give it the *host's* own concurrency
explicitly - k outer, rows inner, transposed weights staged off the step timer, 24 independent
accumulators, separate mul/add (no FMA) so the bytes are unchanged, then let the vector unit take the
inner 24. That is the next branch and it is expected to be a 3-4x on 0.225 + 0.244 ms/call
(~12-14 ms/step), not a rewrite.

## Cold start: what the parallel O_DIRECT loader actually bought, and what it still leaves

The loader (`!158`) was measured here as a step-time hazard (its O_DIRECT reads bypass the page cache,
so rows that re-read checkpoint bytes fell into the ~490 ms/step mode until the seam warm pass), and
that framing hid its actual result: **it is a large cold-start win.** The loop evidence records the
upload phase directly, as `arena.upload_ms` over 72.308 GiB/rank and 95,818 tensors:

| tree | upload | throughput | cold start |
| :-- | :-- | :-- | :-- |
| before the loader (window m1y8, perf-model-20261021.md) | 329.3 s | ~236 MB/s | ~5.5 min |
| every row on the current tree (wsd18 -> wsd28) | 240-266 s | ~323 MB/s | ~4 min |

`arena.allocate_ms` is flat at 23.5-24.2 s across all of them, so the entire delta is the upload phase:
**~85-90 s off every cold start**, and launch-to-exit on the pre-loader tag was 370/386 s against
330-347 s now.

**The headroom that is still there:** those same nodes measured `read_bytes` at ~1.29 GB / 1.5 s
(~860 MB/s) during a row - the hardware can feed this phase ~2.7x faster than it is being used, so
72.3 GiB should upload in ~90 s, not 240 s. That is ~150 s still on the table, which at 95,818 tensors
is ~1.5 ms per tensor of overhead that is not disk. A load-path slice is open on it: coalesced
per-shard reads instead of per-tensor work, queue depth, and any per-tensor thread-count query (the
same cgroup-read bug that cost the host seams 27.6 us per call would cost 44 minutes at this tensor
count if the loader has its own copy of it).

### Correction (2026-10-22, from the load-path investigation): the rows never used the converted loader

The table above compares two windows on the *default* path, and one framing in it is wrong, so it is
corrected here rather than left to mislead. The rows run with `converted:false` and a streaming
arena: they never touch the O_DIRECT converted loader that `!158` added. The 240-266 s upload is the
**serial streaming walk**, and its cost is now decomposed rather than attributed:

- per tensor a zeroed whole-tensor bounce buffer + `pread` + copy (safetensors `read_ranges_into`),
  then the optional b12x repack, then the arena memcpy;
- then an FNV-1a digest over the payload **on the same thread**, measured at ~0.64 GB/s: that is
  **~121 s of the ~254 s**;
- the read itself is disk-bound at ~91 s (the ~860 MB/s observed during a row), plus ~25 s of misc.

So the honest statement is: the *default* path is 254 s of which **half is a hash nobody reads in the
steady state**, and a read+hash worker pool with a single-range fast path (no bounce buffer) plus a
per-phase `LoadTrace` (read / tile / digest / upload) measures **254 -> ~100-110 s**, i.e. a cold start
of ~300 -> ~190 s. That work exists as a local commit on `task/loader-fast` (`2f1110af`, not pushed;
its identity test already caught one real pooled-buffer bug) and is **parked behind the whole-layer
device port**, which is where the user has directed the effort. It needs one gating pass and one row
when a window opens.

The earlier claim that the O_DIRECT loader took the upload from 329 s to 240 s was comparing a window
on a different path against today's default; it is withdrawn. The measured comparison that survives is
default-path 254 s against a pool-fast-path build measured on the box at ~100-110 s, with the digest
as the largest single component.
