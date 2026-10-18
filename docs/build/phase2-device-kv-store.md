# Phase 2 milestone 1 — the device KV store (design note for review)

**Status.** Design note for `task-7`, written before the store core. Branch `phase2/device-kv-store`
off `main` = `6d02313` (!53). Scope is **milestone 1 of several**: this note, the store core plus
CPU-side layout/audit tests, and one small allocation/read-back/measurement run on the node. It does
**not** build the sequence harness (prefill > 128 tokens, decode steps), does not claim Phase 2 exit
criteria, and does not run a kernel.

## 1. What the store is, and what it is not

The store is the **backing memory** for the kernel-ABI arrays; `sparkflash-kv` stays the metadata
layer (radix pages, page tables, `plan_request`, SWA replay planning). The store owns exactly the
arrays the 19-parameter `flash_mla_csa2_sm121` kernel at `6d02313` reads (structure-of-arrays, not
array-of-structs), plus the two published handles the reference keeps as process globals. It does
**not** own activations, weights, the sink, scratch or any paging: a device page allocator is a
Phase 4/5 concern (roadmap section 5.2). The host page payload (a 56,960-byte array-of-structs page)
is unpacked into the per-layer SoA arrays on write; it is never uploaded verbatim.

Out of scope, deliberately: sequence execution, kernel launches, multi-sequence batching, eviction,
and any claim about the reference's resident footprint.

## 2. Geometry — live configuration

`max_seq_len = 1,048,576` tokens: the released default (`MODEL_CONTEXT_TOKENS = 2^20`,
`sparkflash-api`, D5). Geometry rounds up to whole 64-token pages: **16,384 pages**. Only
`KV_SOURCE_LAYERS = [2, 8, 14, 20]` own compressed KV and indexer K (config.json, model.py:500,
517-525, 654-679); their ratios are 2, 2, 2, 1, so they store 32, 32, 32 and 64 entries per page.
Total entries: 3×524,288 + 1,048,576 = **2,621,440**.

## 3. Byte layout per array (D1 entry sizes, D16 finding 1)

| buffer | layout | bytes at 2^20 tokens |
|:--|:--|--:|
| `kv_fp4` | per-layer contiguous `[entries][256]` E2M1 packed | 671,088,640 (640 MiB) |
| `kv_scales` | `[entries][32]` E4M3 per-16 scales | 83,886,080 (80 MiB) |
| `index_k_fp4` | `[entries][64]` E2M1 packed | 167,772,160 (160 MiB) |
| `index_k_scales` | `[entries][4]` E8M0 per-32 scales | 10,485,760 (10 MiB) |
| `window_codes` | `[40 layers][128 slots][512]` E4M3 | 2,621,440 (2.5 MiB) |
| `window_scales` | `[40][128][16]` E8M0 | 81,920 (80 KiB) |
| `hsi_pool` | `[2048 blocks][8][128]` bf16 (HSI candidate pool) | 4,194,304 (4 MiB) |
| `selected` | `[512]` u32 current Top-K list | 2,048 |
| **total** | | **940,132,352 B (896.578 MiB)** |

Two numbers, never conflated:
* **890.0 B/token** = main KV + indexer K per token: 933,232,640 B / 1,048,576 tokens exactly. The
  2,703,360-byte window ring is per sequence and constant in context (08-kv-footprint section 3), and
  the two published handles add 4,196,352 B, so they are **not** in the 890.0 figure.
* **940,132,352 B** = the whole store at 2^20 tokens: 933,232,640 (main+index) + 2,703,360 (window) +
  4,194,304 (HSI pool) + 2,048 (Top-K).

The KV arrays are 720 + 170 + 2.578 MiB = 892.578 MiB, which is the section 5.2 arithmetic
(720 MB / 170 MB / 2.7 MB / ~0.9 GB there; those figures are MiB, spelled out here).

Each layer's kernel pointer is `base + layer_entry_base × entry_size`, so a position p is at
`p × 256` (`kv_fp4`) / `p × 32` (`kv_scales`), matching device-path-design section 3. Window ring
slot offset is `(layer × 128 + slot) × 528`, matching `swa_slot_offset`.

### The 890-vs-1050 question
The store measures its own live allocation, not the report's headline: 933,232,640 B / 1,048,576
tokens = **890.0 B/token** for main + index. There is no 64-byte-per-entry region anywhere in the
layout; the spec's 1050 B/token itemisation double-counts `qk_rope_head_dim = 64`, which is already
inside `head_dim = 512` (D16 finding 1, 08-kv-footprint section 2). This settles *which layout the
implementation realises*; it is not a claim that the reference allocates 890 B/token — the reference
allocates bf16 (D1) and the 890 figure is the implemented store's measured live allocation.

## 4. Layer modes and handle resolution

| layers | mode | own main KV | own index-K | window ring |
|:--|:--|:--|:--|:--|
| 0, 1 | SWA only | — | — | 1 each |
| 2, 8, 14 | Full | yes (32/page) | yes (32/page) | 1 each |
| 20 | Full + HSI source | yes (64/page) | yes (64/page) | 1 |
| 24, 28, 32, 36 | Reindex | read layer 20 | read layer 20's index-K (owns_k is only kv_source layers) | 1 each |
| other 30 backbone layers | Reuse | read `kv_source_for(layer)` | read `kv_source_for(layer)` | 1 each |
| 40–42 | MTP, not in store | — | — | — |

The store allocates **40** window rings, one per backbone layer: every layer attends over its own
window (model.py:707 allocates one window cache per layer), and 128 × 528 × 40 = 2,703,360 B is the
same section's own sum. The task brief's "four window rings (one per source layer)" is an arithmetic
slip; roadmap section 5.2's matching "four window rings at 128 × 528 B × 40 layers" sentence is
corrected in this change to say forty. The ring count is not a design choice.

Handles are resolved, never searched: `main_kv_handle(layer)` and `index_k_handle(layer)` return the
nearest preceding Full layer's array; `window_handle(layer)` returns the layer's own ring; the Top-K
list and HSI pool are published buffers with an explicit publication state.

## 5. Ownership and API

One `DeviceKvStore<M: StoreMemory>` owns every buffer (no process globals, no hidden allocation);
constructed once at boot from a `StoreGeometry`, dropped as a whole. `StoreMemory` is the
allocation/copy backend trait (`allocate`, `write`, `read`, `address`, `allocated_bytes`).

The layout/accounting layer (geometry, buffer plan, page unpacking, entry codecs, audit manifest) is
**pure Rust and CUDA-free**, always compiled, and its unit and integration tests run under plain
`cargo test --workspace` with no GPU. The allocation/HtoD/DtoH backend is a separate
`src/device_cuda.rs` module gated by the crate's `cuda` cargo feature (`libloading` + `dlopen`,
same pattern as the project's existing CUDA wrapper; no libcuda link and no cudarc), and the node
example declares `required-features = ["cuda"]`. Host-only users never see a CUDA dependency.

## 6. Plan → store write protocol, and what the store refuses

The host plan decides *where*: page `p` covers tokens `[64p, 64p+64)`; its page payload is unpacked
to entry `p × 16` (ratio 2) or `p × 64` (ratio 1) per source layer. The store validates and refuses
— it never allocates, never evicts, never infers a page position:

* page index ≥ 16,384 → `OutOfBounds` (**the required ratio-derived negative test**); the last
  legal page, 16,383, must succeed and 16,384 must fail;
* main/index position ≥ the source's bound → `OutOfBounds`; the bounds are **524,288 entries for each
  of layers 2, 8, 14** (ratio 2) and **1,048,576 entries for layer 20** (ratio 1). The last legal
  entries (524,287 and 1,048,575) must succeed and 524,288 / 1,048,576 must fail, with the layer and
  position named in the error;
* payload length ≠ 56,960, or window slot bytes ≠ 528 → `InvalidLength`;
* main-KV/index-K write on a non-Full layer → `WrongLayerMode`;
* Top-K length > 512, HSI pool ≠ 4,194,304 bytes → `InvalidLength`;
* a consumer asking for the Top-K list or HSI pool before publication → `NotPublished`.

## 7. Audit manifest and corruption test

Manifest record per **written** entry: `kind: u8` (main/index/window), `layer: u8`, `position: u32`,
`stored_sha256: [u8;32]` over the **raw stored entry bytes** (288 for main KV, 68 for indexer K, 528
for a window slot), and `canonical_sha256: [u8;32]` over the **dequantise round-trip bytes**:

* main KV: the 512 decoded dequantised values, as f32 LE bits;
* indexer K: the 128 decoded dequantised values, as f32 LE bits;
* window slot: the 512 values `code × scale` **rounded to bf16** (the engine's stored dtype), as
  bf16 LE bits.

NaN is not expected from a valid upload; to keep the x86 dev host and the aarch64 node computing the
same digest for the same stored bytes, any NaN is normalised to the quiet NaN `0x7fc00000`
(`0x7fc0` for bf16) before hashing. Records sort by `(kind, layer, position)`. The audit reads the
device back through the store, recomputes both hashes and fails on the first mismatch, naming
`(kind, layer, position)`.

Corruption: flip one byte of a stored entry (same write path as any upload), re-audit; the audit
**must** fail and name that entry. Unit test on the CPU backend, and the node run performs the same
on device memory.

## 8. Node run and evidence plan

One example, `crates/sparkflash-kv/examples/dev_store_probe.rs`: allocates the full live-geometry
store on the node through the CUDA driver (allocation and memcpys only — **no kernel launch, no GPU
compute**), measures `cuMemGetInfo` free-memory delta across alloc/free plus the store's own
accounting, writes one 64-token page for all four source layers plus one window slot per layer and
the two published buffers, reads back, audits (must pass), corrupts one byte, audits (must fail),
frees. The log is copied to this dev host with sha256 and size; the log records the exact cross-build
command, working directory and binary sha256, and the binary hash's path dependence is stated. Number
to be reported: store total and per-token bytes, and the section-5.2 arithmetic they match.

## 9. Measured run (node node0, NVIDIA GB10, 2026-10-18T08:05:33Z)

Binary: cross-built from commit 6548930, cwd `/home/user/wt/phase2-kv`, rust 1.98.1,
`aarch64-unknown-linux-gnu`, cuda feature on. Staged as
`/tmp/phase2-kv-dev_store_probe-18d7a58e`: 3,201,896 B, sha256
`18d7a58ed3f6564fddf6d9c4129cfed0cd5c5f39cddd2bc5e3ad02e19d466524`. The aarch64 hash is
path-dependent; the recorded build cwd and command are the provenance.

Log: `/home/user/phase2-evidence/phase2-task7-dev_store_probe-20261018T080533Z.log`,
1,629 B, sha256 `2bfd70b54409827ddac3696d7e8912e468d1d03e5ed35507e48a079e72e956a5`. The process
exited 0 and printed `RESULT: PASS`; it allocated and copied only — no kernel launch, no compute.

Geometry at 1,048,576 tokens (16,384 pages, 2,621,440 entries) reproduced section 5 exactly:
kv_fp4 671,088,640; kv_scales 83,886,080; index_k_fp4 167,772,160; index_k_scales 10,485,760;
window_codes 2,621,440; window_scales 81,920; hsi_pool 4,194,304; selected 2,048. Global KV
933,232,640 B = **890.000 B/token** (main KV + indexer K only); window rings 2,703,360 B;
published handles 4,196,352 B; store total **940,132,352 B**. The store's own accounting with all 8
allocations live is exactly 940,132,352 B.

Measured driver memory (`cuMemGetInfo_v2`): device total 130,663,170,048 B; free before
107,923,828,736 B; free after allocation 106,965,532,672 B (delta **958,296,064 B**); free after
release 107,926,237,184 B (release delta 960,704,512 B). The allocation delta exceeds the store's
accounting by 18,163,712 B — the primary context and the driver's per-allocation bookkeeping, not
store bytes. Both figures are reported as measured; the live store figure is the accounting number.
No smoothing: the driver delta is not presented as the store's size, and neither is a claim about a
reference bf16 footprint (which stores bf16 and is a different quantity).

Protocol and audit on the device: host page 0 written for all four source layers (320 entries), 40
window slots (one per layer), HSI pool published, 512 Top-K ids published; clean audit passed
360/360 entries; a page entry read back byte-identical; flipping one byte of
`MainKv { layer: 14, position: 3 }` made the audit fail naming that entry; restoring it made the
audit pass again. Layer 21 resolved to layer 20's arrays, and the window/handle addresses printed
in the log.

Probe-text caveat: the log's layer-21 line prints `entries=268435456`, which is the kv_fp4 array's
**byte** count (1,048,576 entries × 256 B), a print label mistake in the example (fixed in a
follow-up commit; no measurement is affected — the byte count itself is correct and is what the
ledger above uses).

## 9. Open items (documented, next milestone)

1. The kernel's `window_rows` input is contiguous **oldest-first**; the store keeps the
   reference's slot-indexed ring (`slot = position % 128`). Producing the kernel's row order needs a
   128 × 528 B rotation copy per layer per step; the sequence milestone must implement it and
   measure it — it is not hidden in this store.
2. HSI pool population (candidate selection and bf16 dequantisation from layer 20's index-K) is
   next milestone; this store allocates and validates the buffer only.
3. Paged/evicting device allocation stays Phase 4/5.
4. One store is one sequence; batching needs a store pool.
