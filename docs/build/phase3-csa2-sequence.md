# The deferred sequence's position-causal CSA2 attention entry (flash_mla_csa2_seq_sm121)

Status: the kernel, its pin, its typed launch module, the causal indexer host reference and the
m20 per-position twin harness are on branch `task/c4c-csa2-seq` (kernel-only, six commits on
`origin/main` @ `4537f33`, batch 23), the branch that carries the reduce kernel measured in b44.
**Window b50 measured m20: PASS, 50 of 50 checks**, at pin 38,080 B / `f6dfc738...` / membar 0
(verbatim in section 8; report 17,879 B sha256 `d95b6292...`). The position-0 ratchet held: the
sequence entry's out[] is **byte-identical** to the base entry's on the same operands, so no
fallback to the old entry is needed. Window b48 is a **contaminated, superseded run** (an
overlapping window started two GPU clients and raced the vLLM stop/restart loop); it is retained as
history in section 8.1 and is **not** the record and not a rescue for anything. The entry is **not wired into the engine** - the c4a-2 call
site is engine-dev's change, and the host loop stays the fallback at positions > 0 until that
wiring is measured.

The base attention kernel `kernels/flash_mla_csa2_sm121.cu` and its pin/artifacts are **untouched**
by this work, so the published b31/b33/b39/b40 evidence continues to refer to the same
60,792 B / `5bfee4c7...` artifact. The sequence entry is a separate source file, a separate cubin
and its own pin.

## 1. Why a sequence entry, and what b45 actually failed

Window b45's sequence sanity run (binary `sparkflash_m3c_c4c_review_seq2`, sha256
`5af95810...`, log `window-b45-20261019T090623Z.log`, `runs45/seq-sanity2.log`, exit 1 at
2026-10-19T09:10:45Z) FAILED and its record is:

```
sequence attention position support: body_launches=40 expected=120 positions=3
  device_position_0_body=40 other_position_bodies=0
token comparison: position=0 reference_id=201 hybrid_id=320 exact_equal=false
```

The first line is the c4a-2 call-site guard, **not** a gap in the attention math: the guard in
`crates/sparkflash-cli/src/attention_device.rs::attention()` admits only
`start_pos == 0 && seqlen == 1 && window_width == 1 && compressed <= 1`. The kernel's format-2
union arithmetic already handles `start_pos > 0`: m8's captured fixture
(`task14/attention-case-released.json`) is position 129 with 128 engine window rows and 65
compressed ids, and its device-vs-host twin passed. So the honest record is: **the host fallback at
positions 1..129+ is a call-site guard, not a math limitation**, and the guard stays fail-closed
until the call site is extended.

The second line (position 0: reference 201, hybrid 320, positions 1 and 2 exact) is a first-step
state difference above/below the attention body (the position-0 device body at b45 was the OLD base
entry), owned by engine-dev, not by this kernel. Nothing in this page claims to explain it.

What the sequence path DOES need beyond the base entry is the **position-causal indexer**:
stage 1 of the base kernel scores every HSI block over all eight of its tokens with no mask, so a
candidate selection built from `block_best` at position p would see future tokens. The sequence
entry adds exactly that mask and leaves the attention arithmetic alone.

## 2. What the entry changes, exactly

`kernels/flash_mla_csa2_seq_sm121.cu` is generated from the base source from
`#include "m4_common_sm121.cuh"` to the end, with every helper renamed `spark_mla_*` ->
`spark_mla_seq_*` and the entry renamed `flash_mla_csa2_seq_sm121`, so the two cubins can be
loaded together without a duplicate symbol. The only semantic change is in indexer stage 1:

- `first_token = block * 8`; a block whose first token is past the query is skipped and publishes
  `block_scores[block] = -FLT_MAX`, `block_best[block] = 0`.
- Within a reachable block the slot scan stops at the first token past the query (tokens are
  ascending inside a block), so only tokens `<= position` contribute.
- The tie rule is unchanged (strict greater-than, first slot wins), and the warp reduction order is
  the base kernel's five-step `shfl_down_sync` tree.

Everything from the format guard through the three-pass softmax is the base kernel's code: format 2
is the only accepted format, the window rows are the caller's oldest-first engine rows, the selected
ids are absolute compressed positions, a candidate is attended only when `candidate <= position`,
the sink enters the denominator after the per-head max, and the output is bf16. The kernel is a
**correctness seam**: it does not implement the top-512 candidate selection (a separate pipeline
step), and it does not add rope (the caller supplies rotated q/k).

## 3. Arithmetic and the rounding class

The attention arithmetic is the base entry's, unchanged: one f32 three-pass softmax over at most
128 window rows plus 512 selected ids, latent values decoded from E2M1 latent + E4M3 scales (window
rows are E4M3 codes + E8M0 scales per 32 channels), output rounded to bf16. The indexer adds a
128-wide f32 dot per token with the base warp reduction order. The harness compares attention
outputs at the published 2B bar (`||x - y|| / (1 + ||x|| + ||y||)` class, 2e-2) and the indexer
scores at the same bar, tokens exactly; the CPU-side reference test asserts the reduction order.

## 4. ABI

`flash_mla_csa2_seq_sm121(q, kv_fp4, kv_scales, window_codes, window_scales, window_kv_format,
window_rows, selected, index_key, index_query, sink, block_scores, block_best, out, position,
selected_count, trace_ring, layer_id, step_seq)` - the base entry's 19-word argument list in the
same order, pinned by `crates/sparkflash-kernels/src/flash_mla_csa2_seq_launch.rs` against the
kernel source (`PARAMS`, `declaration_matches`, with a mutation test proving the pin is not
vacuous). Geometry: grid `(64, 1, 1)` of `256` threads, one block per query head; stage 2 runs only
in `blockIdx.y == 0` exactly like the base entry.

The entry is its own `cuda_driver::KernelId` variant (`FlashMlaCsa2Seq`, symbol
`flash_mla_csa2_seq_sm121`) and is loaded through the shared `Cuda::load_kernel` path. This does
**not** change `REQUIRED_KERNELS` (still 9) or the frozen six-kernel `m4_abi::KERNELS` array
(D26). Engine-dev bumps `REQUIRED_KERNELS` only when he wires the call site.

**Indexer block/parity contract (important for any harness).** Within the launch, the two query
heads of one index head split the HSI pool: even query head -> even blocks, odd head -> odd blocks.
The full grid is 64 heads, i.e. 32 index-head pairs writing the same 2,048 block arrays, which is a
race on `block_scores`/`block_best`; the chain's stage 2 never reads those arrays, and the
candidate-selection step must use a deterministic launch. m20 checks the indexer at grid `(2, 1, 1)`
(head pair 0 only), the same configuration stage 2's indexer self-check uses.

## 5. Host reference and CPU-side evidence (no device)

- `mla_ref::attention_with_sink_union` already existed (m8) and is used unchanged.
- New in `crates/sparkflash-kernels/src/mla_ref.rs`: `indexer_block_scores(index_key, index_query,
  position)`, the position-causal stage-1 reference for ONE index head: per-block skip when
  `first_token > position`, per-slot break past the query, the kernel's per-lane four-product
  expression and five-step warp sum, and the same strict-greater tie rule. Empty blocks return
  `(-f32::MAX, 0)`.
- Unit tests: the mask at positions 0 and 9 (empty blocks publish `-f32::MAX`/0), the lowest-token
  tie rule, and the existing union-attention tests. These ran in the workspace gates and are the
  only evidence that exists before the window.

## 6. Pin (nvcc V13.0.88 / CUDA 13.0.3, sm_121a)

- Source: `kernels/flash_mla_csa2_seq_sm121.cu`.
- Cubin: **38,080 B**, sha256
  `f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3`,
  `ptx_membar_lines=0`.
- PTX inspection: 1,673 lines, 0 `membar`, 61 `fma.rn`, 56 `ld.global`, 40 `shfl.sync`, 0
  `atom`.
- Baseline manifest: `crates/sparkflash-kernels/baseline/kernel_baseline.txt` now records
  **13 entries** (this is the 13th), **12 `KernelId` variants**, and `PROBED_KERNELS` /
  `check-kernel-baseline.sh` / `regen-kernel-baseline.sh` / `tests/kernel_baseline.rs` agree on
  13. The base `flash_mla_csa2_sm121` pin (60,792 B / `5bfee4c7...`) is unchanged.
- Build lines:
  ```
  nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_seq_sm121.cu -o <dir>/flash_mla_csa2_seq_sm121.cubin
  nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_sm121.cu -o <dir>/flash_mla_csa2_sm121.cubin
  ```

## 7. m20 plan, expected check lines and the staged artifacts

`crates/sparkflash-kernels/examples/m20_csa2_sequence.rs` is the per-position twin bar. It reads
NO checkpoint byte: operands are declared synthetic, from a fixed xorshift64* seed, and the report
records that. It uploads the pinned pair of cubins only after checking both against the embedded
baseline manifest, allocates the largest shape each buffer can need, and walks positions
`[0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137]` - b45's prompt positions and the
full-window/compressed regime. Per position it declares the engine-dev regime: ratio-2 store view
(row q = store row q/2), window rows `max(0, p+1-128)..=p` oldest first through
`cuda_harness::engine_window_row`, compressed selected ids at the even absolute positions, one
sink per head, and a 2,048 x 8 x 128 bf16 index pool whose values are small integers (-4..=4), so
every 128-wide block dot is exact in f32 and the indexer twin is a bit-level comparison rather than
a tolerance (the same reason m7's stage-2 indexer self-check uses integer data).

Checks (50 total = 1 geometry + 12 x 4 per position + 1 position-0 identity):

1. geometry (64 heads x 512 latent, window 128, selected bound 512, 256 threads, positions and
   store-row counts inside the bounds).
2. `position p: union attention matches the host twin at the 2B bar` - device out vs
   `mla_ref::attention_with_sink_union` over the exact device bytes, with exact-bf16 count, max
   abs delta, max ULP and non-finite count.
3. `position p: causal indexer matches the host reduction` - every block best token at or behind
   the query, unreachable blocks exactly `(-FLT_MAX, 0)`, and reachable blocks vs
   `mla_ref::indexer_block_scores` at grid `(2, 1, 1)`.
4. `position p: repeated launches are byte-identical` - out[] from two full launches, and
   out[]+indexer arrays from two head-pair launches.
5. `position p: a selected id at position p+2 is excluded` - appending a future id to the selected
   list leaves out[] byte-identical (the kernel's own mask, not the caller's filtering).
6. `position 0: the sequence entry reproduces the base entry byte for byte` - the ratchet that the
   measured position-0 behaviour is preserved. If this ever fails, engine-dev keeps the base entry
   at position 0 and the page records that measured fallback; there is no silent mixing.

Staged artifacts (node `node0`, evidence dir
`/home/user/phase2-evidence/window-2026-10-18`):

- `bin/m20_csa2_sequence`, 3,863,864 B, sha256
  `c2b67d855374bbcc21d8658d0cd396baf302a8707be8d356f848d921fe355b7e`
  (aarch64 release, built by `scripts/build-aarch64.sh --example=m20_csa2_sequence`).
- `cubins/flash_mla_csa2_seq_sm121.cubin`, 38,080 B / `f6dfc738...`.
- `cubins/flash_mla_csa2_sm121.cubin`, 60,792 B / `5bfee4c7...`.

Window b50 ran the binary against the staged cubins and wrote the report to
`runs50/m20-csa2-sequence-report.json` (17,879 B, sha256 `d95b6292...`); the invocation was:

```
m20_csa2_sequence \
  --cubin-dir /home/user/phase2-evidence/window-2026-10-18/cubins \
  --out /home/user/phase2-evidence/window-2026-10-18/runs50/m20-csa2-sequence-report.json
```

Expected PASS line (b50 produced exactly this line):

```
m20: PASS (50 checks; sequence CSA2 entry at positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137]; synthetic operands, one pinned cubin pair, position-0 base-entry identity measured)
```

## 8. Window record

**The record is window b50**, the clean re-measure that Lead ran after b48 was found contaminated.

Window b50 (node `node0`, log
`/home/user/phase2-evidence/window-2026-10-18/window-b50-20261019T094959Z.log`,
`window-b50-20261019T094959Z.log` sha256
`531c6ec7ba903f9d448bc9ae2dff187501b5cfde3053c0588e3d41f06cbeccf7`, started
2026-10-19T09:49:59Z) verified the staged binary `c2b67d85...` and both pinned cubins in its header
before starting the GPU work (`gpu compute apps=0`), then recorded **PASS, 50 of 50 checks**,
exit 0. Report: `runs50/m20-csa2-sequence-report.json`, 17,879 B, sha256
`d95b6292bdfbb245732d22a3b9e2d7e6cc1af8b912ef3a0bfc674a31b258c566`; its log copy
`runs50/m20-csa2-sequence.log` sha256
`ced4a399df9edea77d32d538e0a912c8550f9b5266c06b3f8a135c977b4e5e8d`. Verbatim, from the
window log (no edits, no elisions):

```
=== RUN m20-clean start 2026-10-19T09:52:02Z
baseline manifest agrees: flash_mla_csa2_sm121 60792 B, 5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3, 0 PTX membar lines
baseline manifest agrees: flash_mla_csa2_seq_sm121 38080 B, f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3, 0 PTX membar lines
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/flash_mla_csa2_sm121.cubin (60792 B, sha256 5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3)
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/flash_mla_csa2_seq_sm121.cubin (38080 B, sha256 f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3)
device 0: NVIDIA GB10 of 1 (total 124610 MiB, MemAvailable 120436 MiB)
geometry: CSA2 sequence entry, 64 heads x 512 latent, window 128, selected bound 512, 256 threads; positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137]
  check released geometry and the sequence regime: ok (vocab-free geometry: 64 heads x 512 latent, window 128, selected bound 512, 256 threads; positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137], at most 69 compressed ids)
case position 0: window_start 0, window_rows 1, selected 1 ids (first Some(0), last Some(0)), compressed rows 1
  check position 0: union attention matches the host twin at the 2B bar: ok (2 of 32768 bf16 outputs exactly equal, max abs delta 1.952e-3, max ULP 32766, non-finite 0 (bar 2e-2))
  check position 0: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 0: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 0: a selected id at position 2 is excluded: ok (out[] byte-identical with 2 appended to the selected list (the kernel masks it))
  check position 0: the sequence entry reproduces the base entry byte for byte: ok (65536 out bytes identical to flash_mla_csa2_sm121)
case position 1: window_start 0, window_rows 2, selected 1 ids (first Some(0), last Some(0)), compressed rows 1
  check position 1: union attention matches the host twin at the 2B bar: ok (2 of 32768 bf16 outputs exactly equal, max abs delta 1.953e-3, max ULP 32791, non-finite 0 (bar 2e-2))
  check position 1: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 1: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 1: a selected id at position 3 is excluded: ok (out[] byte-identical with 3 appended to the selected list (the kernel masks it))
case position 2: window_start 0, window_rows 3, selected 2 ids (first Some(0), last Some(2)), compressed rows 2
  check position 2: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 1.947e-3, max ULP 32769, non-finite 0 (bar 2e-2))
  check position 2: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 2: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 2: a selected id at position 4 is excluded: ok (out[] byte-identical with 4 appended to the selected list (the kernel masks it))
case position 129: window_start 2, window_rows 128, selected 65 ids (first Some(0), last Some(128)), compressed rows 65
  check position 129: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.864e-4, max ULP 69368, non-finite 0 (bar 2e-2))
  check position 129: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 129: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 129: a selected id at position 131 is excluded: ok (out[] byte-identical with 131 appended to the selected list (the kernel masks it))
case position 130: window_start 3, window_rows 128, selected 66 ids (first Some(0), last Some(130)), compressed rows 66
  check position 130: union attention matches the host twin at the 2B bar: ok (1 of 32768 bf16 outputs exactly equal, max abs delta 4.826e-4, max ULP 155737, non-finite 0 (bar 2e-2))
  check position 130: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 130: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 130: a selected id at position 132 is excluded: ok (out[] byte-identical with 132 appended to the selected list (the kernel masks it))
case position 131: window_start 4, window_rows 128, selected 66 ids (first Some(0), last Some(130)), compressed rows 66
  check position 131: union attention matches the host twin at the 2B bar: ok (1 of 32768 bf16 outputs exactly equal, max abs delta 4.830e-4, max ULP 91939, non-finite 0 (bar 2e-2))
  check position 131: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 131: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 131: a selected id at position 133 is excluded: ok (out[] byte-identical with 133 appended to the selected list (the kernel masks it))
case position 132: window_start 5, window_rows 128, selected 67 ids (first Some(0), last Some(132)), compressed rows 67
  check position 132: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.856e-4, max ULP 552039, non-finite 0 (bar 2e-2))
  check position 132: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 132: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 132: a selected id at position 134 is excluded: ok (out[] byte-identical with 134 appended to the selected list (the kernel masks it))
case position 133: window_start 6, window_rows 128, selected 67 ids (first Some(0), last Some(132)), compressed rows 67
  check position 133: union attention matches the host twin at the 2B bar: ok (1 of 32768 bf16 outputs exactly equal, max abs delta 4.880e-4, max ULP 90001, non-finite 0 (bar 2e-2))
  check position 133: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 133: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 133: a selected id at position 135 is excluded: ok (out[] byte-identical with 135 appended to the selected list (the kernel masks it))
case position 134: window_start 7, window_rows 128, selected 68 ids (first Some(0), last Some(134)), compressed rows 68
  check position 134: union attention matches the host twin at the 2B bar: ok (2 of 32768 bf16 outputs exactly equal, max abs delta 4.788e-4, max ULP 46630, non-finite 0 (bar 2e-2))
  check position 134: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 134: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 134: a selected id at position 136 is excluded: ok (out[] byte-identical with 136 appended to the selected list (the kernel masks it))
case position 135: window_start 8, window_rows 128, selected 68 ids (first Some(0), last Some(134)), compressed rows 68
  check position 135: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.859e-4, max ULP 34027, non-finite 0 (bar 2e-2))
  check position 135: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 135: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 135: a selected id at position 137 is excluded: ok (out[] byte-identical with 137 appended to the selected list (the kernel masks it))
case position 136: window_start 9, window_rows 128, selected 69 ids (first Some(0), last Some(136)), compressed rows 69
  check position 136: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.862e-4, max ULP 70319, non-finite 0 (bar 2e-2))
  check position 136: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 136: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 136: a selected id at position 138 is excluded: ok (out[] byte-identical with 138 appended to the selected list (the kernel masks it))
case position 137: window_start 10, window_rows 128, selected 69 ids (first Some(0), last Some(136)), compressed rows 69
  check position 137: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.877e-4, max ULP 67807, non-finite 0 (bar 2e-2))
  check position 137: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 137: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 137: a selected id at position 139 is excluded: ok (out[] byte-identical with 139 appended to the selected list (the kernel masks it))
report: /home/user/phase2-evidence/window-2026-10-18/runs50/m20-csa2-sequence-report.json (17879 B, sha256 d95b6292bdfbb245732d22a3b9e2d7e6cc1af8b912ef3a0bfc674a31b258c566)
m20: PASS (50 checks; sequence CSA2 entry at positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137]; synthetic operands, one pinned cubin pair, position-0 base-entry identity measured)
=== RUN m20-clean exit=0 end 2026-10-19T09:52:03Z
```

What the verbatim record says, grouped:

- Pins first: the baseline manifest agreed on both artifacts (base 60,792 B / `5bfee4c7...`,
  sequence 38,080 B / `f6dfc738...`), then the cubin files were sha256-verified before any
  allocation.
- Union attention at every position (0, 1, 2, 129..=137) matched
  `mla_ref::attention_with_sink_union` at the published 2B bar: max abs delta 1.952e-3 / 1.953e-3 /
  1.947e-3 at positions 0/1/2 and 4.788e-4 .. 4.880e-4 across 129..=137, zero non-finite values.
  The exact-bf16 counts are small (0..2 of 32,768) because the bar is absolute and most outputs sit
  a few 1e-4 from the twin; that is the same comparison class m8 uses, not a widened bar.
- Causal indexer at every position: 2,048 blocks causal, **worst score delta 0.0 exactly** and
  every best token exact. That is the position-causal stage-1 mask working against the
  integer-valued bf16 reference (the exactness is by construction: integer index data makes every
  128-wide dot exact in f32).
- Determinism: out[] from two full launches and out[]+indexer arrays (81,920 B) from two head-pair
  launches byte-identical, at every position.
- Over-supply: a selected id at position + 2 appended to the list left out[] byte-identical at every
  position, so the kernel's own `candidate <= position` mask, not the caller's filtering, excludes
  future ids.
- **Position 0 byte identity: ok - 65,536 out bytes identical to `flash_mla_csa2_sm121`.** The
  measured position-0 behaviour is preserved by the sequence entry at format 2; the old entry does
  not have to be kept at position 0 for format-2 calls, while a format-0/1 position-0 call is
  NaN-refused by the sequence entry and still needs the base entry. There is no silent mixing.

The b45 sequence-sanity FAILURE (section 1) is **not** a measurement of this kernel: that window
ran the base entry at position 0 and the host loop at positions > 0, so it could not have exercised
this cubin. The chain/token evidence for the per-position path remains engine-dev's cut-over
window; this page claims only the kernel twin above.

### 8.1 Superseded: window b48 (CONTAMINATED, not the record)

Window b48 (log `window-b48-20261019T093406Z.log`, started 2026-10-19T09:34:06Z) printed the same
50 checks and the same PASS line, and its report has the same 17,879 B / sha256
`d95b6292...` as b50 - but **b48 is contaminated and is not evidence**: engine-dev started a second
window (b49, 09:33:44Z) overlapping b48 (09:34:06Z), so two GPU clients ran at once and the vLLM
stop/restart loops raced. Lead rejected b48's m20 PASS as the record and re-ran cleanly as b50. The
b48 lines are kept below verbatim only as history; do not cite them, and do not treat b50's
agreement with them as a rescue of b48 - the record is b50. b48's own log shas (`window-b48-*.log`
and its report copy) were not retained as record artifacts; only the lines below survive the retired
window, and b50's artifacts (`531c6ec7...` binary, `ced4a399...` cubin set, report `d95b6292...`) are
the ones to cite.

```
=== RUN m20-csa2-seq-with-timeout start 2026-10-19T09:35:57Z
baseline manifest agrees: flash_mla_csa2_sm121 60792 B, 5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3, 0 PTX membar lines
baseline manifest agrees: flash_mla_csa2_seq_sm121 38080 B, f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3, 0 PTX membar lines
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/flash_mla_csa2_sm121.cubin (60792 B, sha256 5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3)
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/flash_mla_csa2_seq_sm121.cubin (38080 B, sha256 f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3)
device 0: NVIDIA GB10 of 1 (total 124610 MiB, MemAvailable 118633 MiB)
geometry: CSA2 sequence entry, 64 heads x 512 latent, window 128, selected bound 512, 256 threads; positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137]
  check released geometry and the sequence regime: ok (vocab-free geometry: 64 heads x 512 latent, window 128, selected bound 512, 256 threads; positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137], at most 69 compressed ids)
case position 0: window_start 0, window_rows 1, selected 1 ids (first Some(0), last Some(0)), compressed rows 1
  check position 0: union attention matches the host twin at the 2B bar: ok (2 of 32768 bf16 outputs exactly equal, max abs delta 1.952e-3, max ULP 32766, non-finite 0 (bar 2e-2))
  check position 0: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 0: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 0: a selected id at position 2 is excluded: ok (out[] byte-identical with 2 appended to the selected list (the kernel masks it))
  check position 0: the sequence entry reproduces the base entry byte for byte: ok (65536 out bytes identical to flash_mla_csa2_sm121)
case position 1: window_start 0, window_rows 2, selected 1 ids (first Some(0), last Some(0)), compressed rows 1
  check position 1: union attention matches the host twin at the 2B bar: ok (2 of 32768 bf16 outputs exactly equal, max abs delta 1.953e-3, max ULP 32791, non-finite 0 (bar 2e-2))
  check position 1: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 1: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 1: a selected id at position 3 is excluded: ok (out[] byte-identical with 3 appended to the selected list (the kernel masks it))
case position 2: window_start 0, window_rows 3, selected 2 ids (first Some(0), last Some(2)), compressed rows 2
  check position 2: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 1.947e-3, max ULP 32769, non-finite 0 (bar 2e-2))
  check position 2: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 2: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 2: a selected id at position 4 is excluded: ok (out[] byte-identical with 4 appended to the selected list (the kernel masks it))
case position 129: window_start 2, window_rows 128, selected 65 ids (first Some(0), last Some(128)), compressed rows 65
  check position 129: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.864e-4, max ULP 69368, non-finite 0 (bar 2e-2))
  check position 129: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 129: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 129: a selected id at position 131 is excluded: ok (out[] byte-identical with 131 appended to the selected list (the kernel masks it))
case position 130: window_start 3, window_rows 128, selected 66 ids (first Some(0), last Some(130)), compressed rows 66
  check position 130: union attention matches the host twin at the 2B bar: ok (1 of 32768 bf16 outputs exactly equal, max abs delta 4.826e-4, max ULP 155737, non-finite 0 (bar 2e-2))
  check position 130: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 130: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 130: a selected id at position 132 is excluded: ok (out[] byte-identical with 132 appended to the selected list (the kernel masks it))
case position 131: window_start 4, window_rows 128, selected 66 ids (first Some(0), last Some(130)), compressed rows 66
  check position 131: union attention matches the host twin at the 2B bar: ok (1 of 32768 bf16 outputs exactly equal, max abs delta 4.830e-4, max ULP 91939, non-finite 0 (bar 2e-2))
  check position 131: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 131: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 131: a selected id at position 133 is excluded: ok (out[] byte-identical with 133 appended to the selected list (the kernel masks it))
case position 132: window_start 5, window_rows 128, selected 67 ids (first Some(0), last Some(132)), compressed rows 67
  check position 132: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.856e-4, max ULP 552039, non-finite 0 (bar 2e-2))
  check position 132: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 132: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 132: a selected id at position 134 is excluded: ok (out[] byte-identical with 134 appended to the selected list (the kernel masks it))
case position 133: window_start 6, window_rows 128, selected 67 ids (first Some(0), last Some(132)), compressed rows 67
  check position 133: union attention matches the host twin at the 2B bar: ok (1 of 32768 bf16 outputs exactly equal, max abs delta 4.880e-4, max ULP 90001, non-finite 0 (bar 2e-2))
  check position 133: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 133: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 133: a selected id at position 135 is excluded: ok (out[] byte-identical with 135 appended to the selected list (the kernel masks it))
case position 134: window_start 7, window_rows 128, selected 68 ids (first Some(0), last Some(134)), compressed rows 68
  check position 134: union attention matches the host twin at the 2B bar: ok (2 of 32768 bf16 outputs exactly equal, max abs delta 4.788e-4, max ULP 46630, non-finite 0 (bar 2e-2))
  check position 134: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 134: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 134: a selected id at position 136 is excluded: ok (out[] byte-identical with 136 appended to the selected list (the kernel masks it))
case position 135: window_start 8, window_rows 128, selected 68 ids (first Some(0), last Some(134)), compressed rows 68
  check position 135: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.859e-4, max ULP 34027, non-finite 0 (bar 2e-2))
  check position 135: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 135: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 135: a selected id at position 137 is excluded: ok (out[] byte-identical with 137 appended to the selected list (the kernel masks it))
case position 136: window_start 9, window_rows 128, selected 69 ids (first Some(0), last Some(136)), compressed rows 69
  check position 136: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.862e-4, max ULP 70319, non-finite 0 (bar 2e-2))
  check position 136: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 136: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 136: a selected id at position 138 is excluded: ok (out[] byte-identical with 138 appended to the selected list (the kernel masks it))
case position 137: window_start 10, window_rows 128, selected 69 ids (first Some(0), last Some(136)), compressed rows 69
  check position 137: union attention matches the host twin at the 2B bar: ok (0 of 32768 bf16 outputs exactly equal, max abs delta 4.877e-4, max ULP 67807, non-finite 0 (bar 2e-2))
  check position 137: causal indexer matches the host reduction: ok (2048 blocks causal, worst score delta 0.000e0, every best token exact (bar 2e-2))
  check position 137: repeated launches are byte-identical: ok (out[] from two full launches and 81920 bytes (out + indexer arrays) from two head-pair launches)
  check position 137: a selected id at position 139 is excluded: ok (out[] byte-identical with 139 appended to the selected list (the kernel masks it))
report: /home/user/phase2-evidence/window-2026-10-18/runs48/m20-csa2-sequence-report.json (17879 B, sha256 d95b6292bdfbb245732d22a3b9e2d7e6cc1af8b912ef3a0bfc674a31b258c566)
m20: PASS (50 checks; sequence CSA2 entry at positions [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137]; synthetic operands, one pinned cubin pair, position-0 base-entry identity measured)
=== RUN m20-csa2-seq-with-timeout exit=0 end 2026-10-19T09:35:58Z
```

## 9. What is not claimed

- Not the served forward pass, not the chain, not a token, not a timing or throughput number.
- Not the base entry's b31/b33/b39/b40 windows; this entry is a separate cubin and does not
  re-measure them.
- Not the c4a-2 call-site wiring, the per-position host-loop removal, or the token mismatch
  reported at b45 position 0 (engine-dev's first-step state issue).
- Not the top-512 candidate selection, rope, or the indexer's full-pipeline semantics; stage 1's
  block arrays are a side channel with a documented head-pair split, not a selection step.
- Not a position-0 behaviour change: the byte-identity check in m20 measures preservation, it does
  not authorize a new p0 path.
