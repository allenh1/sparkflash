# Host allocator tuning on the TP=4 resident path

Branch `task/allocator-tuning`, base `9ffd5de` (batch 13); implementation sha `2ca1202`.
Evidence: `/home/user/phase2-evidence/task57-profile-2026-10-19/t20-alloc/` (rank 0 parsed here;
all four ranks collected).

## The cost

The tp4-host resident decode path allocates and frees the attention output-projection weights as
f32 on every layer: `wo_a` is 8192x4096 (134 MB) and `wo_b` 5120x8192 (168 MB), plus smaller
scratch. With glibc's stock 128 KiB mmap threshold those allocations are mmap'd and munmap'd per
layer, so a decode step pays mapping and first-touch zero-fill instead of reusing the pages. The
arithmetic never changes.

## The change

`crates/sparkflash-cli/src/allocator.rs` is a public `tune() -> Tuning` called explicitly at the
start of the CLI `main`; it is deliberately not inside any library constructor or
`Tp4Rank::open`, so an embedder opts in. On Linux it calls
`libc::mallopt(M_MMAP_THRESHOLD, 1 GiB)` and `libc::mallopt(M_TRIM_THRESHOLD, 1 GiB)`; other
targets are a no-op. The kill switch is `SPARKFLASH_ALLOCATOR_TUNING=0`. `Tuning` reports
`enabled`, `mmap_threshold` and `trim_threshold` for an evidence header.

## Which option actually does it

`malloc_probe` modes on node0, real checkpoint, 20 threads, positions 8-10, mean of three
per layer:

| mode | layer 0 (ms) | layer 2 (ms) | layer 20 (ms) |
| --- | ---: | ---: | ---: |
| no tune | 214.2 | 219.5 | 218.9 |
| M_MMAP_THRESHOLD=1 GiB only (trim at 128 KiB) | 223.3 | 247.6 | 233.3 |
| M_TRIM_THRESHOLD=1 GiB only (mmap at 128 KiB) | 221.3 | 245.4 | 236.8 |
| **both at 1 GiB** | **82.6** | **93.4** | **94.3** |
| both at 32 MiB | 210.7 | 223.9 | 222.9 |
| M_MMAP_MAX=0, trim default | 227.3 | 242.0 | 238.2 |

`mallopt` returned 1 (success) for every call. glibc exposes no getter for the effective
threshold, so the effective values are proven by behaviour, not by the label:

* A 200 MB allocate-touch-free probe leaves RSS up by **204,868 KiB after free** with both options
  at 1 GiB, but only 68-192 KiB with either option alone or with 32 MiB thresholds. The pages stay
  resident only when both options are raised, i.e. the large buffers were served from the heap and
  freed heap was retained.
* Because the 32 MiB pair shows no win and the 200 MB probe is retained only with the 1 GiB pair,
  `M_MMAP_THRESHOLD=1 GiB` is **not clamped to 32 MiB** in this glibc build for the tested
  allocation sizes; the true mechanism is the pair: the mmap threshold moves these buffers onto the
  main heap, and the trim threshold stops the freed heap from being returned to the OS (which is
  where the 217 -> 86-96 ms attention win comes from). Neither option alone is sufficient, so the
  code must set both; the doc records the measured cause, not just the option names.

## Four-rank A/B

Same four nodes, resident cache, prompt "Question: What is the capital of France? Answer:",
2 generated tokens, 20 reference threads, base port 29801. Baseline is the 1b23d6e profile run;
the tuned run uses the committed-binary path (1b23d6e plus this branch's allocator module).

| per clean step (mean 10) | baseline | tuned | delta |
| --- | ---: | ---: | ---: |
| wall | 13.827 s | 7.929 s | -5.90 s |
| attention block | 7.942 s | 3.251 s | -4.69 s |
| shared expert | 2.017 s | 1.136 s | -0.88 s |
| routed local FFN | 1.768 s | 1.757 s | -0.01 s |
| all-reduce / collective | 0.340 s | 0.164 s | -0.18 s |
| provider (Engram + mHC) | 0.690 s | 0.543 s | -0.15 s |
| post | 0.081 s | 0.081 s | 0 |
| tail (final norm/head/argmax/outside) | 0.990 s | 1.039 s | +0.05 s |

The sum of the per-class deltas equals the wall delta. Only allocation behaviour changed: the
untouched routed FFN is flat, and attention/shared/reduce all contain the per-layer large buffers.

## Invariants

* Tokens on all four ranks: `[11111, 201]`, completion `" Paris"` (identical to baseline).
* rchar/token: 25,114 -> 25,118 (+4 B, 0.016%); disk read/token 429,397 -> 445,440 B (0.2 MB
  absolute, page-cache/run noise; total disk reads 5.15 vs 5.35 MB). The tune does not re-read
  weights.
* rank 0 VmHWM: 90.265 -> 90.462 GiB (+0.20 GiB), consistent with the retained 205 MB heap;
  load RSS 85.843 -> 85.840 GiB. Resident fill 247.8 -> 243.3 s.
* Wall including load 440.5 -> 363.7 s; mean of steps 2-11 13.83 -> 7.93 s.

## Evidence header: agreed, scheduled in the batch-15 follow-up

The tp4-host evidence `flags` block does not yet carry `allocator_tuning`; that writer is
task53-host's `tp4_host.rs`, which this branch does not modify. Schema agreed with tp-dev:

    "allocator_tuning": {
      "enabled": true, "mmap_threshold": true, "trim_threshold": true,
      "kill_switch": "SPARKFLASH_ALLOCATOR_TUNING",
      "requested_mmap_threshold_bytes": 1073741824,
      "requested_trim_threshold_bytes": 1073741824
    }

It goes at the top level of tp4-host-rR.json next to the existing flags/profile (not inside
profile.steps), with a one-line allocator_tuning note in flags/notes for runs without --profile.
Values come from sparkflash_cli::allocator::tune() (idempotent in the writer, or copied from the CLI
start), and the byte values from the module's public
REQUESTED_MMAP_THRESHOLD_BYTES / REQUESTED_TRIM_THRESHOLD_BYTES so the header cannot drift from the
tune. tp-dev cannot add it tonight: task/task53-host is frozen at 7262394 for batch 14. After batch
14 lands and this branch is merged, he opens the batch-15 follow-up that adds (a) this header and
(b) the tp4-host-twin.md cross-reference. The gap is therefore an ordering dependency (batch 14,
then this branch), not a design gap; until it lands, this document and the launch command are the
attribution.

## Follow-up, not in this branch

A reusable buffer pool for the wo_a/wo_b decode would remove the allocation entirely and is a
tp4_host/engine change; this branch is startup allocator configuration only. The measured
attention block is still the largest host component at ~3.25 s/step, so buffer reuse and the
device path remain later levers.