# Phase 3 m16: the certified LM head (head screen, cert_lm_head_sm121) in isolation

**Status: m16 PASSED both cases on the fixed artifact (window b25, exit 0): real-token control PASS
(129,280 exact entries bit-identical, 0 certified) and the concentrated certificate case PASS
(4,899 certified, all bounds valid, counts equal to the flags 4,899 = 4,899, argmax 33865 preserved,
determinism byte-identical) - the first recorded device PASS of the certified pruning path. The
path to it: b21/b23 measured a kernel count contract violation (published 34 against 4,899 flags),
fixed and re-pinned under D31 (14,888 B / 7394b129..., ABI unchanged); b24 measured the fixed count
working and exposed a harness gate that compared the synthetic case against the real token's
reference argmax; that gate was fixed and tested, and b25 passed. No token is claimed: the head is a
stage, both hidden inputs are declared seams, and the comparison class was declared before the
windows and is unchanged.**

This is the m16 deliverable: the last kernel between a full layer stack and a token, the
certified LM-head screen, at the real released head geometry. It is a stage measurement, not the
chain and not a token: the chain's final-layer hidden is not a released tensor, so the hidden state
is a declared seam (below) and the output is not claimed as the served token.

## Subject and launch

| item | value |
| :--- | :--- |
| kernel | cert_lm_head_sm121, the pinned cubin 14,888 B, sha256 7394b1290a3210a78ba12e6fc5414b0f4c84d1cdaf8edbcfa63b25dfcdf00493, 0 PTX membar lines (baseline/kernel_baseline.txt:16; D31 re-pinned the count fix; windows b21/b23 measured the pre-fix 14,712 B a833a336... artifact) |
| entry | kernels/cert_lm_head_sm121.cu:60, one extern "C" entry, pinned by crates/sparkflash-kernels/src/m4_abi.rs:1056 |
| release tensor | head.weight, BF16 [129,280 x 5,120] (1,323,827,200 B), read from the released snapshot |
| geometry | 129,280 vocabulary over 4 tensor-parallel ranks of 32,320 rows each; hidden width 5,120; screen tile 128; 40 blocks; grid 32 x 128 threads; one launch per rank |
| host reference | crates/sparkflash-kernels/src/lm_head_ref.rs (the same certificate rule and its mutation proof) |
| launch (device) | one cert_lm_head_sm121 launch per rank over that rank's 32,320 rows; the four shards are all launched and the 129,280-way argmax is assembled from all four |

The cubin is built under the pinned toolkit (nvcc V13.0.88, CUDA 13.0.3, -std=c++17 -arch=sm_121a
-cubin -I kernels) and the harness refuses any other artifact by length and sha256 before the first
device allocation. The pinned cubin reproduces byte-for-byte from the tree source (built here:
14,888 B, 7394b129..., equal to the baseline pin and the D31 fix).

## The two declared hidden seams

The head's input is the chain's final-layer hidden state, which the release does not ship as a
tensor. This harness has two declared cases and no fabricated fallback:

| seam | what it is here | status |
| :--- | :--- | :--- |
| real-token control hidden | the head input the phase-1 host chain produced in a Lead window on the released checkpoint: ~/hostrun/phase1-host-token0.head.f32, f32 5,120 (20,480 B), sha256 e6819e83a32c13fa6843ec576ba3959bbb9cb645047d59e445eaa235a7846c04. Size- and sha-verified here; the harness REFUSES to run if it is missing or differs | declared input, produced by the phase-1 chain in a Lead window, not by m16 |
| concentrated certificate hidden | synthetic and declared: bf16(1.0) at index 7, zero elsewhere. In block 0 the bound is max over the first 128 weights of |w_ji| and the exact value is w_j7, one of those 128, so the certificate actually prunes; not a released hidden state and no token is claimed from it | declared synthetic case, for the pruning path only |
| reference engine logits (optional) | ~/hostrun/phase1-host-token0.logits.f32, 129,280 f32 (517,120 B), the phase-1 reference engine's logits for the same token; sha256 printed, declared as the phase-1 reference and not recomputed here | declared input, used for the argmax/top-5 comparison only |

The head consumes the hidden as the kernel's bf16 ABI, so the twin below is built from the same
bf16 codes: this compares the head's arithmetic, not the f32-to-bf16 conversion (phase 1 measured
that conversion at at most 2.4320781e-2 on a logit). The control is the non-pruning case and the
concentrated case is the pruning case.

## Comparison class (declared before any number)

The pinned PTX has 17 fma.rn.f32 and zero mul.rn/add.rn: the exact dot is one ascending fma chain
per entry. Both operands are bf16, so every product fits f32 exactly and the fused chain is
bit-identical to a mul-then-add chain - the class is not a tolerance. Per entry:

| quantity | class |
| :--- | :--- |
| exact entries (certified == 0), all 129,280 | **bit-identical** to the twin's bf16-matched ascending fma dot |
| certified entries (certified == 1) | the stored value is the certificate bound: (i) it is >= the twin's exact value within f32 eps (1e-4 + 1e-5 * |exact|); (ii) the kernel's own pruning rule is replayed from the measured values in its per-thread order (chunk, lane, ascending steps; bound < running max, running max not updated on a skip) and every certified entry must satisfy it; (iii) it matches the host's same bound formula (absolute block sums times the row's per-block abs-max) within a declared bar (1e-3 + 1e-5 * |host|), because the kernel tree-reduces the block sums and the host accumulates sequentially |
| certified_count | the kernel's atomic count equals the number of certified flags |
| block reductions (32 per rank) | each published block_max/block_argmax equals a per-entry scan of that block's device logits with the lowest-entry tie-break |
| global argmax over 129,280 | equal to the twin's argmax (lowest entry index on ties; assembled from all four rank shards); in the real-token control, when the reference logits are supplied, also equal to the phase-1 reference engine's argmax - a measured equality with the bf16-conversion caveat, not a proof. The reference logits are the real token's, so the synthetic concentrated case reports the reference argmax as not applicable and is not gated on it |
| top-5 set | claimed against the reference only in the real-token control and only if that case certified nothing (then all 129,280 values are exact); the kernel's contract is the argmax and a certified entry stores a bound, so no top-5 set is claimed from the pruning case |
| determinism | two device runs of every case byte-identical across logits, certified flags, certified_count and the block arrays |
| pruning path | the concentrated case must exercise it (predicted from the host twin: 4,899 of 129,280 certified; the control is predicted to certify 0, reproducing phase 1). If the device still certifies 0, the record says so and reports the certified count, how many entries had a bound below their shard maximum, and the measured device-vs-host bound delta instead of claiming the path |

The bound bars are not looseness: the kernel reduces the 40 block sums in a fixed tile order while
the host accumulates sequentially, so bound values are not the same floating-point program; the
certificate replay and the upper-bound checks are what prove the pruning, not the bar.

## Window command

The harness binary for the re-run after the D31 kernel fix and the reference-gate fix is
m16_lm_head_82828285_aarch64 (3,936,984 B, sha256
82828285043cc0fd7c1c743177acebe983048570ab642dab3dbe1cd11d02b360, built here with
scripts/build-aarch64.sh --example=m16_lm_head). Earlier windows keep their own binaries and shas:
b21 used m16_lm_head_21f3f474_aarch64 (3,906,368 B, 21f3f474..., no per-rank count pairs), b23 used
m16_lm_head_6de2e2e2_aarch64 (3,936,056 B, 6de2e2e2...) and b24 used m16_lm_head_e82da196_aarch64
(3,936,088 B, e82da196...). The run is (from the node, with vLLM stopped):

    ~/phase2-evidence/window-2026-10-18/m16_lm_head_82828285_aarch64 \
        --checkpoint /home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
        --cubin-dir ~/phase2-evidence/window-2026-10-18/cubins \
        --head-input ~/hostrun/phase1-host-token0.head.f32 \
        --reference-logits ~/hostrun/phase1-host-token0.logits.f32 \
        --repeat 100

The cubin directory must hold the pinned cert_lm_head_sm121.cubin (14,888 B, 7394b129..., the D31
fix; the pre-fix 14,712 B a833a336... artifact is kept named with its sha for the b21/b23 records)
and the harness verifies it by size and sha256 before any allocation. The
harness refuses to run below 32 GiB MemAvailable (--floor-gib) at start, before the head upload,
after the head upload and at end (and skips the start check only in --twin-only, which touches no
device), printing the MemAvailable reading at each point.

## Host dry run (not a device record)

To validate the twin before spending a window, the harness has --twin-only: no cubin check, no
device allocation, no device claim. On the node it read the released head.weight (1,323,827,200 B),
both seams and the reference logits and printed: real-token control twin argmax 5 value
13.237006187 (the phase-1 device value at the same entry), predicted screen 0 of 129,280 certified,
0 entries with a bound below their shard maximum - reproducing the phase-1 finding that this token's
bound never falls below the running maximum; concentrated case twin argmax 33865 value 0.800781250,
predicted screen 4,899 of 129,280 certified, 128,922 entries with a bound below their shard maximum.
Host twin time was 0.75-0.76 s per case and peak RSS 1,303,264 kB. These are host predictions and
are not quoted as device numbers.

## Window b21 (m16's first device window)

Run by the Lead on 2026-10-19. Node log `window-b21-20261019T033245Z.log` and
`runs21/m16-head.log` under `/home/user/phase2-evidence/window-2026-10-18/`. Binary
`m16_lm_head_21f3f474_aarch64` (3,906,368 B, sha256 21f3f474...); cubin `cert_lm_head_sm121`
14,712 B sha256 a833a336... verified by size and sha256 before the first device allocation;
MemAvailable at start 123,787,692 kB (118.1 GiB), pre-upload 120,886,248 kB (115.3 GiB),
post-upload 120,886,248 kB, end 120,879,336 kB, floor 32 GiB ok at every point; vLLM stopped
for the window and restored afterwards.

Verbatim from `runs21/m16-head.log`:

* control: `case real-token control: PASS - 129280 exact entries bit-identical, 0 certified,
  counts agree, block reductions agree, bound bar (max |device-host| 0.000000e0) holds,
  certificate holds (replay violations 0, validity violations 0), argmax 5 value 13.237006187
  (twin 5 value 13.237006187), reference argmax 5 value 13.230665207, determinism byte-identical`
* control top-5: `case real-token control: top-5 set vs the reference identical (control, all
  values exact)`
* concentrated: `case concentrated certificate case: FAIL - exact mismatches 0, count mismatches
  4, block mismatches 0, bound bar violations 0, certificate violations 0, replay violations 0,
  argmax device 33865 vs twin 33865, reference Some(5), determinism true, top-5 true`
* pruning path: `pruning path: FIRST device execution - the concentrated case certified 4899 of
  129280 entries; every certified bound passed the replay (violations 0), validity (violations 0)
  and the host bound formula (max |device-host| 0.000000e0)`
* timing control: `timing [real-token control, 100 x 4-rank group]: per-token (group) ms min
  32.3482 p50 32.4810 mean 32.5516 max 33.4037; per-launch ms 8.1379`
* timing concentrated: `timing [concentrated certificate case, 100 x 4-rank group]: per-token
  (group) ms min 31.7253 p50 31.8434 mean 31.8641 max 32.7052; per-launch ms 7.9660`
* exit: `summary: 2 cases, 1 failed`, `m16_lm_head: 1 case(s) failed: the head's arithmetic,
  bound, count, block reduction, argmax or determinism class did not hold; no PASS is claimed`

So the real-token control PASSES on the declared class at the real released geometry: all 129,280
exact entries are bit-identical, the certificate certified nothing, every block reduction and the
global 129,280-way argmax agree, the top-5 set matches the phase-1 reference engine, and the
timing line is measured. The concentrated case exercises the pruning path for the first time on
device: 4,899 entries certified, exactly the host twin's prediction, and every certified bound is
valid - exact entries stay bit-identical, the per-thread pruning replay has 0 violations, the
bounds match the host formula bit-for-bit, the argmax is right, and two device runs are
byte-identical. Only the published certified_count disagrees with the flags, on all four ranks.

## Finding (b21): cert_lm_head_sm121 publishes a count that is not the number of flags

This is not a host-side count-assembly bug and not the strict `bound < running max` tie: the host
compares each rank's published count to that rank's own flags independently (no cross-rank sum),
and the replay over the measured values validates every flag with 0 violations; the device's
certified set also equals the host twin's predicted set exactly (4,899). The contract violation is
in the kernel's count reduction.

Evidence (the pre-fix cubin, rebuilt from commit b30820c here, reproduces 14,712 B / a833a336...,
i.e. it is the artifact b21 and b23 verified; it is preserved on the node as
`cubins/cert_lm_head_sm121.a833a336.cubin`):

* `kernels/cert_lm_head_sm121.cu:100` declares `pruned` as a per-thread counter; `:122` increments
  it for a certified entry inside that thread's own entry loop.
* `:148` `if (lane == 0u)` and `:160` `atomicAdd(certified_count, pruned);`: the atomic is executed
  only by lane 0 of each block, and it adds lane 0's own `pruned`, not the block total. The block
  reduction at `:149-157` reduces `running_max`/`running_argmax` only; `pruned` is never reduced
  across the 128 threads.
* The pinned PTX agrees: `mov.u32 %r80, %r74;` (`target/cert_lm_head.ptx:234`) makes %r80 the
  thread-local pruned register, the `tid.x != 0` predicate at `:629`/`:637` skips the
  block-reduction path, and the single `atom.global.add.u32 %r72, [%rd38], %r80;` (`:714`) adds
  only that lane's value.

The contract is explicit: `kernels/cert_lm_head_sm121.cu:65` calls certified_count the "total
pruned entries, atomically accumulated", and the declared class (and `lm_head_ref.rs`'s host twin)
reads it as the number of certified flags. On the pruning path the kernel therefore publishes the
sum over blocks of lane 0's prunes. The host twin predicts that undercount exactly: 34 total
(per rank 12, 6, 3, 13) against 4,899 flags. b21 printed only the aggregate mismatch (4 ranks
disagreeing); the next window's binary prints the per-rank (flagged, published) pairs, which should
read 4,899 flags total against 34 published if this analysis is right.

Phase 1 never saw this because the real-token certificate certified 0 entries, so the atomic added
0: the control's counts agree at 0/0 and the control PASS stands. This is the first device
execution of the pruning path, and it is exactly the path the count reduction had never been
exercised on.

Scope: the exact entries, the certificate bounds, the pruning decisions, the block reductions, the
argmax and determinism are all correct in b21; what is broken is the kernel's published summary
count. That blocks any pruning-path claim (the count is part of the declared class) but does not
change the control's PASS.

The artifact b21/b23 measured is the pre-fix 14,712 B / a833a336... cubin; those records keep its
sha verbatim as the evidence for the fix. The Lead chose the re-baseline option and the fix is
recorded below as D31; window b25 then passed both cases on the new artifact (recorded at the end of
this page), which is where the pruning-path claim comes from.

## Window b23 (diagnostic): the count arithmetic is confirmed on device

Run by the Lead on 2026-10-19, start 03:55:12Z; node log `window-b23-20261019T035512Z.log` and
`runs23/m16-head.log` under `/home/user/phase2-evidence/window-2026-10-18/`. Same command as
b21; binary `m16_lm_head_6de2e2e2_aarch64` (3,936,056 B, sha256 6de2e2e2...) and cubin
`cert_lm_head_sm121` 14,712 B sha256 a833a336... both re-verified by the window log before the
run; node started with MemAvailable 123,448,420 kB (117.7 GiB) after vLLM stopped, 120,558,284 kB
pre/post head upload (115.0 GiB), 120,528,876 kB (114.9 GiB) at end, floor 32 GiB ok everywhere.

Verbatim from `runs23/m16-head.log`:

* control: `case real-token control: PASS - 129280 exact entries bit-identical, 0 certified,
  counts agree, block reductions agree, bound bar (max |device-host| 0.000000e0) holds,
  certificate holds (replay violations 0, validity violations 0), argmax 5 value 13.237006187
  (twin 5 value 13.237006187), reference argmax 5 value 13.230665207, determinism byte-identical`
* control top-5: `case real-token control: top-5 set vs the reference identical (control, all
  values exact)`
* concentrated: `case concentrated certificate case: FAIL - exact mismatches 0, count mismatches
  4, block mismatches 0, bound bar violations 0, certificate violations 0, replay violations 0,
  argmax device 33865 vs twin 33865, reference Some(5), determinism true, top-5 set NOT claimed
  (not the control case)`
* count pairs: `case concentrated certificate case: certified_count vs flags per rank (flagged,
  published) [(1062, 12), (1205, 6), (1215, 3), (1417, 13)]`
* count totals: `case real-token control: certified flags 0 total, published certified_count 0
  total, per rank (flagged, published) [(0, 0), (0, 0), (0, 0), (0, 0)]` and `case concentrated
  certificate case: certified flags 4899 total, published certified_count 34 total, per rank
  (flagged, published) [(1062, 12), (1205, 6), (1215, 3), (1417, 13)]`
* pruning path: `pruning path: FIRST device execution - the concentrated case certified 4899 of
  129280 entries; every certified bound passed the replay (violations 0), validity (violations 0)
  and the host bound formula (max |device-host| 0.000000e0)`
* timing control: `timing [real-token control, 100 x 4-rank group]: per-token (group) ms min
  31.6591 p50 31.7876 mean 31.8780 max 32.6492; per-launch ms 7.9695`
* timing concentrated: `timing [concentrated certificate case, 100 x 4-rank group]: per-token
  (group) ms min 31.1325 p50 31.2469 mean 31.2784 max 32.0638; per-launch ms 7.8196`
* exit: `summary: 2 cases, 1 failed` and the no-PASS line, as in b21

The diagnostic confirms the finding exactly: published certified_count 34 total, per rank
[12, 6, 3, 13] - the host twin's prediction to the number - against 4,899 certified flags. The
control is 0/0 and PASSES again on the declared class with the same binary that records the
failure. The count contract is violated on the pruning path for the pinned artifact a833a336... on
two windows; the certificate bounds, exact entries, pruning decisions, block reductions, argmax
and determinism are intact. The head's control claim (bit-identical exact entries, argmax, top-5
set against the phase-1 reference, block reductions, determinism) is device-measured and stands.
No pruning-path claim is made. b23 ran on the pre-fix a833a336... cubin; the fix and re-pin are
recorded below as D31.

## Window b24 (D31 fix): the count equals the flags; the FAIL exposed a harness gate bug

Run by the Lead on 2026-10-19, start 04:06:30Z, run start 04:07:43Z, end 04:07:53Z; node logs
`window-b24-20261019T040630Z.log` and `runs24/m16-head.log`. Binary
`m16_lm_head_e82da196_aarch64` (3,936,088 B, sha256 e82da196...) and cubin
`cert_lm_head_sm121` 14,888 B sha256 7394b129..., both re-verified by the window log before the
run; node MemAvailable 123,449,528 kB after vLLM stopped, 120,555,608 kB pre/post head upload,
120,548,096 kB at end (floor 32 GiB ok).

The D31 fix works on device. Verbatim:

* `case real-token control: PASS - 129280 exact entries bit-identical, 0 certified, counts agree,
  block reductions agree, bound bar (max |device-host| 0.000000e0) holds, certificate holds (replay
  violations 0, validity violations 0), argmax 5 value 13.237006187 (twin 5 value 13.237006187),
  reference argmax 5 value 13.230665207, determinism byte-identical`
* `case real-token control: top-5 set vs the reference identical (control, all values exact)`
* `case concentrated certificate case: certified_count vs flags per rank (flagged, published)
  [(1062, 1062), (1205, 1205), (1215, 1215), (1417, 1417)]`
* `case concentrated certificate case: certified flags 4899 total, published certified_count 4899
  total, per rank (flagged, published) [(1062, 1062), (1205, 1205), (1215, 1215), (1417, 1417)]`
* `pruning path: FIRST device execution - the concentrated case certified 4899 of 129280 entries;
  every certified bound passed the replay (violations 0), validity (violations 0) and the host
  bound formula (max |device-host| 0.000000e0)`
* timing control: `p50 31.8468 mean 31.9083; per-launch ms 7.9771`; concentrated: `p50 31.2243 mean
  31.2590; per-launch ms 7.8147`
* but: `case concentrated certificate case: FAIL - exact mismatches 0, count mismatches 0, block
  mismatches 0, bound bar violations 0, certificate violations 0, replay violations 0, argmax
  device 33865 vs twin 33865, reference Some(5), determinism true, top-5 set NOT claimed (not the
  control case)`

Every measured check passed and the count disagreement is gone; the only check that could still
fail was the reference argmax: the phase-1 reference logits are the real token's, whose argmax is
5, while the concentrated case's hidden is synthetic and its argmax is 33,865. The harness was
applying the reference comparison to the synthetic case, so a host-side gate - not the kernel and
not the class - produced the FAIL. The kernel and count fix are clean on this window.

Harness fix: `reference_argmax` is now gated on the case being the real-token control (a
`reference_applicable` measurement), and the synthetic case prints `reference not applicable -
synthetic hidden` instead of comparing. The new
`the_synthetic_case_is_not_gated_on_the_reference_argmax` test asserts (i) a synthetic case with a
supplied reference does not fail on it, (ii) gating the synthetic case fails it (mutation A: always
gate -> test fails), and (iii) the control still fails when its reference argmax differs (mutation
B: never gate -> test fails). Both directions were mutation-run; 29 tests pass.

The re-window (b25) with `m16_lm_head_82828285_aarch64` passed both cases, including counts
4,899/4,899; it is recorded at the end of this page as the fixed-artifact verdict.

## Fix (D31): every thread's prunes are counted, and the pin is re-baselined

The Lead chose to fix the kernel and treat it as a deliberate re-baseline: the certified pruning
path is what makes the head viable in serving, and a wrong count would corrupt whatever accounting
uses it. `docs/DECISIONS.md` D31 records the decision.

Source change, minimal and within the existing structure: the atomic moved out of the lane-0 block
and every thread adds its own prunes -
`if (pruned > 0u) atomicAdd(certified_count, pruned);`. A thread with no prunes skips the atomic, so
the all-exact path's work is unchanged; the count is a sum of integers, so the order does not
matter and the value is deterministic across runs (the harness's two-launch byte-identity check
enforces it). The block reduction at `:148-157` still publishes block_max/block_argmax from lane 0
only, which is correct and unchanged.

New artifact, built reproducibly with the pinned 13.0.3 toolkit (nvcc V13.0.88, CUDA 13.0.3,
-std=c++17 -arch=sm_121a -cubin -I kernels): 14,888 B, sha256
7394b1290a3210a78ba12e6fc5414b0f4c84d1cdaf8edbcfa63b25dfcdf00493, 0 PTX membar lines. PTX check:
there is still exactly one `atom.global.add.u32` for the count, now at `target/cert_lm_head.ptx:717`,
guarded only by `setp.eq.s32 %p80, %r81, 0; @%p80 bra $L__BB0_74` (skip when this thread pruned
nothing); the `tid.x != 0` predicate that used to skip the count (`:629`/`:637`) branches to that
same guard, so every lane reaches it and adds its own `%r81` (the thread-local pruned register).
Compare the pre-fix artifact: the atomic sat inside the lane-0 block and added lane 0's value
only.

Re-pin, and what changed: `crates/sparkflash-kernels/baseline/kernel_baseline.txt:16` now pins
14,888 B / 7394b129...; the m16 harness PIN constants, this page's subject/window-command pin and
`docs/build/serving-readiness.md` 2.13 carry the new size and sha; the phase-1 window record's
cubin list (`phase1-boundary-table.md`) keeps a833a336... because it is a historical window record
and is not rewritten. D31 states explicitly that the ABI is unchanged: the entry signature, the
geometry macros and the semantic of certified_count (total pruned entries) are the same; only the
reduction of an internal counter changed, so the artifact now delivers the documented semantic
instead of changing it.

The re-window (b25) ran the same command and cases on this artifact: `certified_count == flags`
held per rank (1062, 1205, 1215, 1417, total 4,899) and every bound/certificate/argmax/determinism
check passed, so the pruning path is recorded PASS in the b25 section below. The b21/b23 undercount
records stay verbatim above as the evidence for D31.

## Window b25 (verdict): both cases PASS on the fixed artifact

Run by the Lead on 2026-10-19, start 04:17:17Z, run start 04:18:30Z, exit 0 at 04:18:39Z; node logs
`window-b25-20261019T041717Z.log` and `runs25/m16-head.log`. Binary
`m16_lm_head_82828285_aarch64` (3,936,984 B, sha256 82828285...) and cubin
`cert_lm_head_sm121` 14,888 B sha256 7394b129..., both re-verified by the window log before the
run; node MemAvailable 123,504,044 kB after vLLM stopped, 123,502,748 kB at harness start, 120,609,788
kB pre-upload, 120,609,564 kB post head upload, 120,602,280 kB (115.0 GiB) at end, floor 32 GiB ok.

Verbatim from `runs25/m16-head.log`:

* `case real-token control: PASS - 129280 exact entries bit-identical, 0 certified, counts agree,
  block reductions agree, bound bar (max |device-host| 0.000000e0) holds, certificate holds (replay
  violations 0, validity violations 0), argmax 5 value 13.237006187 (twin 5 value 13.237006187),
  reference argmax 5 value 13.230665207, determinism byte-identical`
* `case real-token control: top-5 set vs the reference identical (control, all values exact)`
* `case concentrated certificate case: PASS - 124381 exact entries bit-identical, 4899 certified,
  counts agree, block reductions agree, bound bar (max |device-host| 0.000000e0) holds, certificate
  holds (replay violations 0, validity violations 0), argmax 33865 value 0.800781250 (twin 33865
  value 0.800781250), reference not applicable - synthetic hidden, determinism byte-identical`
* `case concentrated certificate case: top-5 set NOT claimed (not the control case)`
* `case concentrated certificate case: certified flags 4899 total, published certified_count 4899
  total, per rank (flagged, published) [(1062, 1062), (1205, 1205), (1215, 1215), (1417, 1417)]`
* `pruning path: FIRST device execution - the concentrated case certified 4899 of 129280 entries;
  every certified bound passed the replay (violations 0), validity (violations 0) and the host
  bound formula (max |device-host| 0.000000e0)`
* timing control: `timing [real-token control, 100 x 4-rank group]: per-token (group) ms min 31.7717
  p50 31.8770 mean 31.9695 max 33.3880; per-launch ms 7.9924`
* timing concentrated: `timing [concentrated certificate case, 100 x 4-rank group]: per-token
  (group) ms min 31.1821 p50 31.2669 mean 31.2993 max 32.0788; per-launch ms 7.8248`
* `summary: 2 cases, 0 failed`
* `m16_lm_head: PASS (both hidden seams at the released head geometry: every exact entry is
  bit-identical to the bf16-matched twin, every certified bound is a valid upper bound whose
  per-thread pruning replay has no violations, certified counts and all 32 block reductions agree,
  the global 129,280-way argmax matches the twin and the phase-1 reference engine, and two runs of
  every case are byte-identical)`

This is the m16 verdict on the fixed artifact: the control (non-pruning) case passes as it did in
b21/b23/b24, and the pruning case passes for the first time - 4,899 certified entries, every
certified bound replayed valid with 0 violations, the published count equal to the flags per rank,
the 129,280-way argmax preserved (33,865 value 0.800781250, device and twin bit-equal), block
reductions agreeing and two runs byte-identical. The reference argmax is reported as not applicable
for the synthetic hidden and the top-5 set is not claimed from the pruning case.

## NOT claimed

- No token is claimed: the head is a stage, not the chain, and the hidden is a declared seam. The
  concentrated case is not a token at all.
- No throughput or latency claim beyond the measured whole-launch timing line (one node, vLLM
  stopped; timed repeats start from the same buffers and their outputs are not compared).
- No statement about the f32-to-bf16 conversion of the hidden beyond the phase-1 measurement quoted
  above; the comparison is the head's arithmetic on the same bf16 codes.
- No claim that the certificate prunes on the real token: the control is the non-pruning case, and
  the page reports the measured count rather than assuming it.
- No top-5 values or set from the pruning case; no cross-layer or chain claim.
- No pruning-path claim from b21/b23 (their published certified_count disagreed with the flags, see
  Finding and Window b23); the pruning-path PASS is only the b25 verdict on the fixed 14,888 B /
  7394b129... artifact, and it is a stage result (the concentrated input is synthetic, not a token).
- No claim that the control's timing generalises beyond the measured one-node, vLLM-stopped run.
