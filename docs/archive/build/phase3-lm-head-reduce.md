# The certified head's device reduction (lm_head_reduce_sm121)

Status: the kernel, its pin, the host reference and the m19 harness are on branch
`task/c4c-lm-head-reduce` (kernel-only, four commits on `origin/main` @ `d95fa65`, batch 22). **Window b44
measured it: m19 PASS (17 of 17 checks)** on the real-token control, the concentrated certificate
and the synthetic tie/offset/bound/refusal cases, at pin 36,120 B / `3fcd7ba5...` / membar 0. Two
earlier windows FAILED and their failure records stay verbatim in section 8: b42 (7 of 14; missing
read base and duplicate top-5 entry) and b43 (5 of 14; the written global entry lacked the block
base, so a certified winner's flag was read at the wrong entry). The kernel is not wired into the
engine yet - the c4c cut-over is engine-dev's change, and the host assembly stays the fallback.

## 1. Why a reduce kernel

Today (b41) the c4c chain's last device launch is `cert_lm_head_sm121` four times, once per rank.
Each launch writes 32,320 stored logits (a certificate bound where its flag is 1), 32,320
certified flags, 32 block maxima and 32 block argmax entries. The host then copies those arrays
back and performs the 129,280-way assembly itself: the 128 block answers are folded with
`consider_head_block` (greater value, then lower global entry), a top-5 is built, and the token is
taken. That is the only remaining host math between the last device launch and the token.

`kernels/lm_head_reduce_sm121.cu` moves that assembly onto the device. It is a **correctness
seam**, not a performance stage: the four shard buffers are already device-resident (the chain
allocates them for the screen), so the launch reads about 2 MiB (4 x 32,320 f32 logits + 4 x 32,320
flags), adds no new allocation, and removes the per-rank logits/flags/block readbacks from the
token path. The host may still read the small report back - the 4-byte token is the declared seam,
with status/count alongside it for the claim/refusal contract - but it does no arithmetic on the
way to the token.

## 2. What it computes, exactly

The reduce reproduces the m16 host assembly step for step:

- **Blocks are the m16 layout, rank-major.** Global block `r * 32 + b` covers rank `r`'s entries
  `[b * 1024, min((b + 1) * 1024, 32,320))`; the last block of each rank has 576 entries. There are
  128 blocks.
- **The four shard ranges are UNIFORM**: rank `r` owns exactly the 32,320 consecutive vocabulary
  entries `[r * 32,320, (r + 1) * 32,320)`. The non-uniform numbers sometimes quoted for the
  concentrated certificate case (1,062 / 1,205 / 1,215 / 1,417) are that case's **certified-flag
  counts per rank**, not shard sizes or ranges; do not re-import them as shard geometry.
- **Block answers** use the screen's rule: strict greater-than, ties to the lowest global entry.
- **The 129,280-way answer** is the lowest-index tie-break over the 128 block answers in ascending
  global block order; the value written is the stored value at that entry.
- **The top-5** is sorted by value descending, then ascending global index, over the stored values.
  It is CLAIMED only when no entry is certified (`certified_count == 0`), because a certified entry
  stores an upper bound rather than its exact value. That is the device-side half of m16's
  `claims_top5`; the host still requires a reference before it may claim the set, which m19 does
  for the real-token control.
- **No entry is silently valued as if it were exact.** If the argmax entry is certified, the status
  word carries `WINNER_CERTIFIED` and the caller must not claim the token (refuse, do not assert);
  the token is still written for diagnostics. On a sound certificate this cannot happen, because a
  certified entry satisfies `bound < running maximum`.

## 3. Arithmetic and the rounding class

**The reduce performs no arithmetic on a logit.** It only compares and moves f32 bit patterns, and
compares/moves u32 indices and flag bytes. It therefore cannot introduce a value delta: the
rounding class is inherited from `cert_lm_head_sm121`, the only arithmetic upstream (the screen's
dot products and BF16 activation ABI). There is no FMA-contraction, division or rsqrt class of its
own to pin; the harness's comparison class is still bit-exact, because the answer must be the same
entry and the same stored bits, not a tolerance.

Determinism comes from fixed scan order - one launch, one block, blocks visited in ascending global
order by fixed warp/lane assignments, thread 0 folding the 128 answers in ascending order. The PTX
has **0** `membar` lines, so the pin records `ptx_membar_lines=0` as expected (the one block is
synchronised with `__syncthreads()`, not with a device-scope fence).

## 4. ABI

```c
extern "C" __global__ void lm_head_reduce_sm121(
    const float *logits_r0, const float *logits_r1,
    const float *logits_r2, const float *logits_r3,   // [32,320] per rank, stored/bound values
    const uint8_t *certified_r0, const uint8_t *certified_r1,
    const uint8_t *certified_r2, const uint8_t *certified_r3, // [32,320] flags, 0 or 1
    unsigned int *token,          // the 4-byte declared seam: global argmax entry
    float *value,                 // stored value at that entry
    float *block_max,             // [128] rank-major
    uint32_t *block_argmax,       // [128] global entries
    float *top5_value,            // [5]
    uint32_t *top5_index,         // [5]
    unsigned int *status,         // 0x1 / 0x2 / 0x4, below
    unsigned int *certified_count,// total flags over the four shards
    GpuTraceRing *trace_ring,     // ABI uniformity only; no event yet
    uint32_t layer_id,
    uint64_t step_seq);
```

Nineteen argument words, declared by `crates/sparkflash-kernels/src/lm_head_reduce_launch.rs`
(`ARG_WORDS = 19`, `BLOCK = (256, 1, 1)`, `GRID = (1, 1, 1)`). The launcher fills the declaration's
parameter order; the declaration pin parses the kernel source and refuses a reordered, retyped or
de-qualified parameter, with a mutation test proving the pin is not vacuous. The four separate
logit pointers and four certified pointers are deliberate: the chain already owns four shard
buffers, and separate pointers cannot be confused with an offset/stride pair.

**Status bits.**

| bit | name | meaning |
| --- | --- | --- |
| `0x1` | `WINNER_CERTIFIED` | the argmax entry is certified, so its value is a bound; the token is diagnostic and must not be claimed |
| `0x2` | `TOP5_CLAIMED` | the top-5 set is exact; set only when `certified_count == 0` (the host still needs a reference) |
| `0x4` | `ANY_CERTIFIED` | at least one entry was certified; informational |

The token is claimable when bit `0x1` is clear even if `0x4` is set (the concentrated case's exact
winner); the top-5 may only be claimed when `0x2` is set. A synthetic winner-is-certified case is
part of the harness and asserts the refusal reason, not just the bit.

**Trace.** The trace words are carried for ABI uniformity but the kernel emits no event: all
sixteen scope bits are assigned (specification 5.8) and none describes the head reduction, so the
arguments mean a future scope revision does not change this ABI.

## 5. Host reference and CPU-side evidence

`crates/sparkflash-kernels/src/lm_head_reduce_ref.rs` is the host assembly the chain performs
today, written as a pure function over the four shard buffers: `assemble` returns the token, stored
value, the 128 block maxima/argmax entries, the top-5 and the status/count; `host_assembly_token`
is the `consider_head_block` fold alone. Neither performs arithmetic on a logit. Its tests cover
the m16 block layout including the 576-entry partial block, a brute-force full-width argmax,
lowest-index ties across ranks, the top-5 order, a certified winner's refusal, the concentrated
case's 4,899 flags with an exact winner (token claimable, top-5 not), the all-exact control, and
validation refusals for wrong rank/length/flag values.

The kernel's build-side checks: the declaration pin (above), the geometry constants parsed from the
source and compared with the Rust constants, the status-bit contract, and the 576-entry partial
block. The kernel is probed on every build with the other eleven sources, and the combined probe
compiles all twelve in one translation unit.

## 6. Pin (nvcc V13.0.88 / CUDA 13.0.3, sm_121a)

```text
kernel lm_head_reduce_sm121 cubin_bytes=36120 cubin_sha256=3fcd7ba5e9871d01804e84825c107bf7741efac7b16982c77aced29b726b2849 ptx_membar_lines=0
```

It is the **twelfth** entry in `crates/sparkflash-kernels/baseline/kernel_baseline.txt`;
`build.rs`'s `PROBED_KERNELS` probes twelve sources; `cuda_driver::KernelId` gained
`LmHeadReduce` (eleven variants); the frozen six-kernel `m4_abi` array is untouched.
`sparkflash-device::cubins::REQUIRED_KERNELS` stays at nine on this branch: engine-dev bumps it to
ten only at the cut-over, after m19 is measured.

PTX inspection: 1,403 lines, 0 `membar`, 0 `fma.rn`, 5 `ld.global`, 10 `shfl.sync`, 0 `atom`.

Three builds, two failures, one artifact: build 1 (37,152 B / `6cff3661...`) failed b42 because it
scanned each block's local prefix instead of `base + local` (every block re-read its rank's first
1,024 entries) and could store a top-5 entry twice; build 2 (36,120 B / `509b933e...`) failed b43
because the block argmax and top-5 entry were written as `rank * 32,320 + local` without the block
base (block 1 reported block 0's entry), which also made a certified winner's flag lookup use the
wrong global entry. Build 3 (this pin) indexes `base + local` for the logits, the flags and the
written global entry, and refuses a duplicate top-5 index. This is the pin b44 measured (PASS, 17/17).

## 7. m19 plan and the expected check lines

`crates/sparkflash-kernels/examples/m19_lm_head_reduce.rs` is the window harness. It pins both
cubins before any device allocation (`cert_lm_head_sm121` 14,888 B / `7394b129...`; the reduce
above), runs the four rank launches and then two reduce launches per case, copies the screen's
published arrays and the reduce report back (the declared seam), and compares them with the host
reference. Cases: the declared real-token control (the phase-1 5,120-f32 head input, size- and
sha256-verified, plus `--reference-logits` for the reference top-5/argmax), the declared
concentrated certificate (synthetic `bf16(1.0)` at index 7), and synthetic full-width tie/refusal
operands.

Expected lines (17 checks; the two pin lines print before the count):

```text
cubin: <cubin-dir>/cert_lm_head_sm121.cubin (14888 B, sha256 7394b129...)
cubin: <cubin-dir>/lm_head_reduce_sm121.cubin (36120 B, sha256 3fcd7ba5...)
baseline manifest agrees: cert_lm_head_sm121 14888 B, 7394b129..., 0 PTX membar lines
baseline manifest agrees: lm_head_reduce_sm121 36120 B, 3fcd7ba5..., 0 PTX membar lines
  check released geometry and the m16 block layout: ok (...)
  check control: reduce token and value are the host assembly's, bit for bit: ok (...)
  check control: all 128 block maxima and argmax entries match the screen's published arrays: ok (...)
  check control: certified count 0, top-5 claimed, no refusal: ok (...)
  check control: top-5 set is the host's and the reference's: ok (...)
  check control: reduce token is the phase-1 reference engine's argmax: ok (...)
  check control: two reduce launches are byte-identical: ok (...)
  check concentrated: the screen certifies 4,899 entries (per rank 1,062/1,205/1,215/1,417): ok (...)
  check concentrated: reduce token is the host assembly's and the declared 33,865, bit for bit: ok (...)
  check concentrated: all 128 block maxima and argmax entries match the screen's published arrays: ok (...)
  check concentrated: exact winner, top-5 NOT claimed (certified entries store bounds): ok (...)
  check concentrated: two reduce launches are byte-identical: ok (...)
  check synthetic tie: the lowest global entry wins: ok (...)
  check synthetic block offsets: a nonzero block's argmax is base + local: ok (...)
  check synthetic certified bound tying the exact max: refuses the token: ok (...)
  check synthetic certified winner in rank 2: WINNER_CERTIFIED is set: ok (...)
  check synthetic certified winner: refuses the token and the top-5 instead of asserting: ok (...)
m19: PASS (17 checks; real-token control, concentrated certificate, synthetic tie/offsets/bound/refusal; one pinned cubin)
```

Window command (from the node, vLLM stopped; `cubins/` holds both pinned cubins):

```text
~/phase2-evidence/window-2026-10-18/bin/m19_lm_head_reduce_aarch64 \
    --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
    --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins \
    --head-input ~/hostrun/phase1-host-token0.head.f32 \
    --reference-logits ~/hostrun/phase1-host-token0.logits.f32
```

The harness refuses to run below 32 GiB MemAvailable (`--floor-gib 32`) at start and end.

**Status: measured by window b44 - PASS (17 of 17 checks).** The b42 and b43 failures, their
fixes and the b44 record (verbatim) are in section 8 of this page. The reduce is not wired into the
engine on this branch: engine-dev's cut-over is the next change, with the host assembly kept as the
declared fallback. A mismatch is a finding against the kernel, not a
tolerance to widen, and the host path of D32/D34 stays the fallback until it is fixed.

## 8. Window record

### b42 — FAILED (7 of 14 checks), 2026-10-19T08:34:22Z–08:35:37Z

Binary `m19_lm_head_reduce_c340b19f_aarch64` (sha256 `c340b19f...`), cubin `lm_head_reduce_sm121`
37,152 B / `6cff3661...`, exit 1. Log: `window-b42-20261019T083422Z.log`,
`runs42/m19-lm-head-reduce.log`. This is a FAILURE record, not a pass; the exact failing lines were:

```text
  check control: all 128 block maxima and argmax entries match the screen's published arrays: FAILED (128 of 128 blocks bit-identical to the screen remapped rank-major, and to the host reference)
  check control: top-5 set is the host's and the reference's: FAILED (top-5 set vs the reference identical (control, all values exact))
  check concentrated: the screen certifies 4,899 entries (per rank 1,062/1,205/1,215/1,417): FAILED (flags [1062, 1205, 1215, 1417], published counts [1062, 1205, 1215, 1417], reduce count 3501)
  check concentrated: reduce token is the host assembly's and the declared 33,865, bit for bit: FAILED (reduce token 65025 value 0.730468750, host assembly token 33865 value 0.800781250, declared 33865)
  check concentrated: all 128 block maxima and argmax entries match the screen's published arrays: FAILED (128 of 128 blocks bit-identical)
  check synthetic tie: the lowest global entry wins: FAILED (token 7 value 5.0, top-5 starts [7, 7], status 0x2)
  check synthetic certified winner: refuses the token and the top-5 instead of asserting: FAILED (token 777 is diagnostic; status 0x2 names WINNER_CERTIFIED, no token/top-5 claim)
```

Root cause: one kernel defect, the missing block base. The scan read `logits[local]` and
`certified[local]` instead of `logits[base + local]` / `certified[base + local]`, so every block of
a rank re-read the rank's first 1,024 entries (the last block re-read the first 576). That explains
all five kernel-side failures: the block arrays were the repeated prefix (not the published arrays);
the count was 31 x the first-chunk flags plus the first-576 flags (3,501); the concentrated winner
was the best entry in the prefixes (entry 65,025 in rank 2), not the real argmax 33,865; and the
synthetic refusal found 777 in the prefix instead of 4,242. The duplicate `[7, 7]` top-5 followed
from the repeated scans inserting the same entry twice. The two control "FAILED" lines printed
success text because the harness built its detail strings unconditionally; the checks now compare
through `compare_blocks` / `compare_top5` / `compare_count_and_status` helpers that return the first
exact mismatch and have unit tests asserting both the equal and the changed case.

### b43 — FAILED (5 of 14 checks), 2026-10-19T08:49:12Z–08:50:27Z

Binary `m19_lm_head_reduce_61a3e8af_aarch64` (sha256 `61a3e8af...`), cubin `lm_head_reduce_sm121`
36,120 B / `509b933e...`, exit 1. Logs: `window-b43-20261019T084912Z.log`, `runs43/m19-reduce.log`.
The b42 base-read and duplicate fixes took effect (counts, block-0 arrays, determinism, tie and
refusal-bits-on-rank-0 cases passed); the exact remaining failure lines were:

```text
  check control: all 128 block maxima and argmax entries match the screen's published arrays: FAILED (block 1 argmax: reduce 873 vs host reference 1897)
  check control: top-5 set is the host's and the reference's: FAILED (top-5 slot 4 entry: reduce 873 vs host reference 1897)
  check concentrated: reduce token is the host assembly's and the declared 33,865, bit for bit: FAILED (reduce token 32841 value 0.800781250, host assembly token 33865 value 0.800781250, declared 33865)
  check concentrated: all 128 block maxima and argmax entries match the screen's published arrays: FAILED (block 1 argmax: reduce 406 vs host reference 1430)
  check synthetic certified winner: refuses the token and the top-5 instead of asserting: FAILED (token 146 is diagnostic; status 0x4 names WINNER_CERTIFIED, no token/top-5 claim)
```

Root cause: the global entry written for a winning local entry was
`rank * 32,320 + local`, missing the block base. Block 0 of a rank happened to be right (base 0),
so all b42-fixed checks that touched block 0 passed; block 1 reported block 0's entry (873 vs 1,897,
and the top-5's slot 4 the same way), the concentrated winner was block 1's local entry reported as
block 0's global (32,841 instead of 33,865, with the same stored value, which is why the value
matched), and the synthetic certified winner's flag lookup used rank 0's index 146 (uncertified)
instead of the global 4,242, so the status was ANY_CERTIFIED (0x4) instead of
WINNER_CERTIFIED|ANY_CERTIFIED (0x5). The fix is `rank * 32,320 + base + local` for the written
entry (the read was already `base + local` after b42), plus the new synthetic checks below.

### b44 — PASS, 2026-10-19T09:02:22Z–09:03:37Z

Binary `m19_lm_head_reduce_55ec92c7_aarch64` (sha256
`55ec92c79ba0d2610457af262e39c27b01d9b151f70e3e4a4b656d9fc227d4f9`), cubin
`lm_head_reduce_sm121` 36,120 B / `3fcd7ba5...`, exit 0. Logs:
`window-b44-20261019T090222Z.log`, `runs44/m19-reduce2.log`. Verbatim:

```text
=== m19: the certified head's device reduction (lm_head_reduce_sm121) against the host assembly ===
geometry: vocabulary 129280 over 4 ranks of 32320; 128 m16 blocks = 4 ranks x 32 chunks of 1024 entries (last chunk 576); reduce block 256 threads, one launch; top-5
comparison class (declared before any number): the reduce's token, stored value, 128 block maxima/argmax entries, top-5 indices/values, status and certified count must equal the host assembly's on the same four device shard buffers BIT FOR BIT; two reduce launches must be byte-identical. The host assembly is lm_head_reduce_ref::assemble, the same block fold the c4c chain performs with consider_head_block. Not compared with a tolerance.
MemAvailable at m19 start: 123740896 kB (118.0 GiB), floor 32 GiB - ok
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/cert_lm_head_sm121.cubin (14888 B, sha256 7394b1290a3210a78ba12e6fc5414b0f4c84d1cdaf8edbcfa63b25dfcdf00493)
cubin: /home/user/phase2-evidence/window-2026-10-18/cubins/lm_head_reduce_sm121.cubin (36120 B, sha256 3fcd7ba5e9871d01804e84825c107bf7741efac7b16982c77aced29b726b2849)
baseline manifest agrees: cert_lm_head_sm121 14888 B, 7394b1290a3210a78ba12e6fc5414b0f4c84d1cdaf8edbcfa63b25dfcdf00493, 0 PTX membar lines
baseline manifest agrees: lm_head_reduce_sm121 36120 B, 3fcd7ba5e9871d01804e84825c107bf7741efac7b16982c77aced29b726b2849, 0 PTX membar lines
  check released geometry and the m16 block layout: ok (vocab 129280 = 4 x 32320, 128 blocks, last chunk 576 entries, top-5)
hidden seam [real-token control]: phase-1 host chain head input from a Lead window, sha256-verified here (not produced by m19): /home/user/hostrun/phase1-host-token0.head.f32 f32, sha256 e6819e83a32c13fa6843ec576ba3959bbb9cb645047d59e445eaa235a7846c04
hidden seam [concentrated certificate]: declared synthetic hidden: bf16(1.0) at index 7, zero elsewhere; not a released hidden state and no token is claimed from it
reference engine logits: 129280 f32 from /home/user/hostrun/phase1-host-token0.logits.f32 (sha256 ed10b840897224e2d6aac65fc3967e02384c83f1754e1bd248fa325b76b4e5e9, argmax 5 value 13.230665207); declared as the phase-1 reference, not recomputed here
released head.weight: BF16 [129280 x 5120] (1323827200 B) read from /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 (1 tensors read, 1323827200 B)
device 0: NVIDIA GB10 of 1 (total 124610 MiB, MemAvailable 119522 MiB)
device bytes admitted: 1324496200
case real-token control:
  check control: reduce token and value are the host assembly's, bit for bit: ok (reduce token 5 value 13.237006187, host assembly token 5 value 13.237006187, reference argmax 5 value 13.230665207)
  check control: all 128 block maxima and argmax entries match the screen's published arrays: ok (128 of 128 blocks bit-identical to the screen remapped rank-major, and to the host reference)
  check control: certified count 0, top-5 claimed, no refusal: ok (0 flagged entries = screen-published 0 = reduce count 0, status 0x2)
  check control: top-5 set is the host's and the reference's: ok (top-5 set vs the reference identical (control, all values exact))
  check control: reduce token is the phase-1 reference engine's argmax: ok (argmax 5 value 13.230665207)
  check control: two reduce launches are byte-identical: ok (1080 report bytes compared)
case concentrated certificate:
  check concentrated: the screen certifies 4,899 entries (per rank 1,062/1,205/1,215/1,417): ok (4899 flagged entries = screen-published 4899 = reduce count 4899, status 0x4)
  check concentrated: reduce token is the host assembly's and the declared 33,865, bit for bit: ok (reduce token 33865 value 0.800781250, host assembly token 33865 value 0.800781250, declared 33865)
  check concentrated: all 128 block maxima and argmax entries match the screen's published arrays: ok (128 of 128 blocks bit-identical to the screen remapped rank-major, and to the host reference)
  check concentrated: exact winner, top-5 NOT claimed (certified entries store bounds): ok (status 0x4; top-5 set NOT claimed (4899 certified entries store bounds))
  check concentrated: two reduce launches are byte-identical: ok (1080 report bytes compared)
  check synthetic tie: the lowest global entry wins: ok (token 7 value 5.0, top-5 starts [7, 32420], status 0x2)
  check synthetic block offsets: a nonzero block's argmax is base + local: ok (block 1 argmax 1124 (want 1124), block 127 argmax 129204 (want 129204), token 1124)
  check synthetic certified bound tying the exact max: refuses the token: ok (token 100 (diagnostic) status 0x5; the certified bound must not be claimed as exact)
  check synthetic certified winner in rank 2: WINNER_CERTIFIED is set: ok (rank-2 token 64740 (diagnostic) status 0x5, want 0x5)
  check synthetic certified winner: refuses the token and the top-5 instead of asserting: ok (token 4242 is diagnostic; status 0x5 names WINNER_CERTIFIED, no token/top-5 claim)
MemAvailable at m19 end: 120820628 kB (115.2 GiB), floor 32 GiB - ok
m19: PASS (17 checks; real-token control, concentrated certificate, synthetic tie/refusal; one pinned cubin)
```

What the PASS means: on the same four device shard buffers the screen wrote, one reduce launch
reproduces the host block assembly bit for bit - token 5 and value 13.237006187 on the real-token
control (host token 5, reference argmax 5), all 128 block answers against both the screen's published
arrays and the host reference, the all-exact top-5 set identical to the reference engine's, and the
concentrated certificate's 4,899 flags with the exact winner 33,865 and the top-5 NOT claimed. The
synthetic cases pin the tie rule, the nonzero-block global index (block 1 -> 1,124, block 127 ->
129,204), the certified-bound-equal-to-exact-max refusal (status 0x5) and a certified winner in
rank 2 (status 0x5). Two launches are byte-identical in both real cases.

What it does not mean: no serving-timing, throughput or host-time-saved claim; no claim about the
served forward pass or the c4c wiring (engine-dev's cut-over, with fallback); the concentrated and
synthetic hidden operands are declared synthetic, and the concentrated certificate's winner is a
preserved argmax, not a released-model token claim.

## 9. What is not claimed

Not the served forward pass, not the c4c engine wiring (engine-dev wires the launch and the host
fallback at the cut-over), not the head screen, not a token/chain/per-layer claim, not host-time
saved, not a performance or bandwidth claim. The real-token control's hidden activation is the
declared phase-1 artifact read by path, not produced by m19; the concentrated and synthetic cases
are declared synthetic operands, and no token is claimed from the concentrated case's certificate
(its exact winner is a preserved argmax, not a released-model token). Other kernels' entries are
untouched.
