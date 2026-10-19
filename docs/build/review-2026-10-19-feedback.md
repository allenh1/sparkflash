# Review feedback of 2026-10-19: every claim, and what the repository says about it

Status: written 2026-10-19 on branch `task/review-2026-10-19` (off `main` at `7edda55`), docs only; no code
changed. The claim set is /home/user/review-2026-10-19-input.md (A1-A11, B1-B15, C1-C6), registered
as shared task-3. Each claim gets one verdict - CONFIRMED, CONFIRMED-BUT-STALE, FALSE, MISFRAMED,
UNVERIFIABLE - a measured evidence line at a named commit, and a disposition. The precedent for this
form is docs/build/review-2026-10-18-feedback.md: nothing below is called measured unless it was run
here, and anything I could not check is labelled with the reason.

The review is not itself in the tree; its base is named only in its own A5 (`main@e39364d`, the
`!110` tip). Main is now `7edda55` (`!112`). I measured at `7edda55` and, where the review's base
matters, also at `e39364d`.

## 0. Measurement base

| Fact | Measured value | How |
|:--|:--|:--|
| Worktree | `task/review-2026-10-19` at `7edda55`, clean | git rev-parse HEAD |
| Main | `7edda55` (`!112`), 97 commits total (96 non-initial) | git rev-list --count main |
| Tracked lines | 481,828 total; 269,791 JSON fixtures + 3,277 Cargo.lock; code-and-docs remainder 208,760 | git ls-files then wc -l |
| Rust | 252 .rs files / 177,982 lines: 118,406 under src/, 751 in two crate-root build.rs; 25,683 tests/, 33,142 examples/ | git ls-files then wc -l |
| CUDA | 19 files (15 .cu + 4 .cuh) / 4,516 lines | git ls-files then wc -l |
| Markdown | 48 files; wc -l sum 20,324; 20,325 counting the spec's unterminated last line | git ls-files then wc -l |
| Decisions | 38, D1-D38 | grep -c of the D headings in docs/DECISIONS.md |
| Kernel pins | 13 kernel lines, nvcc_release=V13.0.88, arch=sm_121a | crates/sparkflash-kernels/baseline/kernel_baseline.txt |
| Windows | 64 window-b*.log files; 62 distinct b numbers b2..b68, missing 18, 37, 63, 64, 65; 63 scripts incl b69 (no b69 log); no b0/b1 on the node | ls on node0, read-only |
| Restart health | 520-590 s across 42 logs; b68 restart 14:05:22Z, healthy after 560 s | grep of the window logs |
| m4_abi.rs | 13,906 lines / 754,981 B | wc -l -c |
| CLI device files | device_chain.rs 2,698; moe_device.rs 2,183; attention_device.rs 823; serve.rs 2,044; doctor.rs 4,142 | wc -l |
| Device crate | 7,341 lines | wc -l over its .rs files |
| Doctor, 2026-10-18 | verdict not_ready; 0 passed / 4 failed / 1 not-applicable | doctor-local-2026-10-18.json |

Host reads used one ssh into allenh1@192.0.2.80 (read-only greps, one scp of the doctor JSON): no
GPU work, no vLLM action, no window.

Verdict counts over the 32 claims: CONFIRMED 20, CONFIRMED-BUT-STALE 4, FALSE 2, MISFRAMED 5,
UNVERIFIABLE 1. The mapping is in section D. (B2's remedy is rejected by addendum I; its CONFIRMED
verdict records that the review accurately reported the repo's D37 option as it stood.)

## A. Development velocity claims

### A1. 206,000+ lines in 4.5 days
Verdict: CONFIRMED (line count); the span is overstated.
207,175 lines of rs+cu+cuh+md+toml+sh+py+txt+yml+hex+h+c excludes the 269,791 JSON fixture lines and
Cargo.lock and still reproduces "206,000+". First commit 2addf30 is 2026-10-15T17:32Z; the review's own
handoff is 2026-10-19T13:31Z, i.e. 3.83 days elapsed, not 4.5 (4.5 is a calendar count).
Disposition: accepted; correct the span.

### A2. 177,982 Rust LoC across 252 .rs files: 119,157 src, 25,683 tests, 33,142 examples
Verdict: CONFIRMED exactly.
252 files, 177,982 lines; tests/ 25,683 and examples/ 33,142 to the line; src/ 118,406 plus the two
crate-root build.rs files (751) = 119,157.
Disposition: accepted.

### A3. 4,516 CUDA LoC across 19 .cu/.cuh files; 13 pinned sm_121a cubins under nvcc V13.0.88
Verdict: CONFIRMED.
19 files, 4,516 lines; baseline/kernel_baseline.txt has 13 kernel entries, nvcc_release=V13.0.88,
arch=sm_121a, and scripts/check-kernel-baseline.sh plus the Rust test
the_baseline_list_matches_the_manifest_and_the_kernels_on_disk pin the same list.
Disposition: accepted.

### A4. 20,325 docs LoC across 48 markdown files; 38 decisions D1-D38
Verdict: CONFIRMED.
48 files; wc -l sums to 20,324 because deepseek_v41_flash_4x_dgx_spark_rust_spec.md has no trailing
newline - counting its final partial line gives the review's 20,325. 38 decisions.
Disposition: accepted.

### A5. 110 merge requests landed (!1 through !110) on main at e39364d
Verdict: FALSE. The highest id was read as the count.
At e39364d: 96 commits total, 95 non-initial, 95 MR-titled commit subjects; the highest id is !110
and eight ids in 1..110 never appear on main (31, 32, 34, 36, 82, 83, 84, 101). 102 distinct !N tokens
appear somewhere in main's messages (doc commits cross-reference MRs), which is probably where "110"
came from. At 7edda55 it is 96 MR commits, highest id !112, and !111 is also absent.
Disposition: rejected as written; the velocity point stands, the number must be quoted as 95/96.

### A6. 68 GPU windows (b0-b68) on head node node0 (192.0.2.80)
Verdict: FALSE. The numbered evidence is b2-b68 with gaps and no b0/b1.
/home/user/phase2-evidence/window-2026-10-18/ (the only window evidence directory on the node) has
64 log files, 62 distinct b numbers (b2..b68; 18, 37, 63, 64, 65 absent), 63 scripts including b69,
which has no log yet. There is no b0 or b1 log or script anywhere under /home/user. The measured
per-day counts are 10 logs dated 2026-10-18 and 54 dated 2026-10-19.
Disposition: rejected as written; quote "62 recorded numbered windows, b2-b68, plus b69 pending".

### A7. ~39,500 LoC/day, ~3 kernels/day, ~8.4 decisions/day, ~24.4 MRs/day, ~15 windows/day
Verdict: CONFIRMED in magnitude; the two errors nearly cancel.
The review's arithmetic is exact: 177,982/4.5 = 39,551, 13/4.5 = 2.89, 38/4.5 = 8.44, 110/4.5 = 24.4,
68/4.5 = 15.1. Against the measured 3.83-day span the real rates are 46,433 LoC/day, 3.39 kernels/day,
9.91 decisions/day, 24.8 MRs/day, 16.2 windows/day - every rate within about 18% of the review's. The
MR and window numerators are wrong (A5, A6) but the long 4.5-day denominator and the high numerators
happen to cancel.
Disposition: accepted in magnitude; restate the counts with A5/A6.

### A8. Chronology, Oct 15-16 through Oct 19
Verdict: CONFIRMED-BUT-STALE; one item is wrong.
Checked item by item: Oct 15-16 foundation with sparkflash-engine as the ninth crate (D18,
docs/DECISIONS.md:706) and the aarch64/doctor-local work - confirmed. Oct 17 first device window
measuring m7_layer0 stages 1-3 (roadmap-to-serving.md section 1.1), D21 determinism and D22 released
mHC Sinkhorn both "Recorded 2026-10-17" - confirmed. But "plus stage 5 LM head" is not Oct 17: the
head's first device window was b21 and it passed in b25 (docs/build/phase3-lm-head.md:3,108), both
2026-10-18/19 - wrong date. Oct 18 DeviceKvStore 896.578 MiB and 890.0 B/token
(phase2-device-kv-store.md:42-45), m8 11,764 copy_dtod and 7,400.000 MiB/s
(phase2-sequence-harness.md:9-10,254-255), moe_router/routed_gate_up in the b-windows (phase3-router.md,
phase3-gate-up.md), RDMA D29 "Recorded 2026-10-18", device M1/M2/M3-a/M3-b scaffolding
(crates/sparkflash-device/src/lib.rs:1-27) - all confirmed. Oct 19 40-layer chain wiring b19-b41 with
1,167 launches (device-forward-chain.md:1303) rising to 1,168 after the head reduce
(STATUS-2026-10-19.md:39), b44 lm_head_reduce (D35), b50 flash_mla_csa2_seq (D36), b66 leak closed,
b68 decode seam and first full 138-position run (STATUS-2026-10-19.md:40-47) - confirmed.
Disposition: accepted with the correction (drop stage 5 from Oct 17).

### A9. STATUS says 55-60% but true distance is ~35-40%, with the axis table
Verdict: MISFRAMED. The measured facts are right; the percentages are an unmeasured re-estimate and
one supporting number is contradicted by the node's own JSON.
Confirmed: 0/9 tokens; 8060 vs 491 at position 129; max logit difference 15.009581 vs the 0.02 bar;
first expert-set flip at layer 3 with 36/40 layers flipping (docs/STATUS-2026-10-19.md:53-80, D37);
b68 ~8.7-9.6 s/position and projected_chain_full_s=1239.0 (runs68/seq130x8.log; STATUS:47); 16.07 GB
staged_matrix_bytes/token (b41, device-forward-chain.md:1303); STATUS says 55-60% at STATUS:117.
Not reproducible: "3 preflight gates fail" - doctor-local-2026-10-18.json is not_ready with 0 passed,
4 failed (driver-kernel, link-mtu, clocks-memory, engram-shards) and 1 not-applicable (rdma-bandwidth
did not run); the 2026-10-15 QUALIFICATION table has 5 FAIL and 3 never-run, so no reading gives 3.
"serve only drives CPU ReferenceDriver/FakeSession" is true by default, but --engine device can also
select DeviceEngine over CudaSession (M2: one real kernel launch, no token) behind the cuda feature
(crates/sparkflash-cli/src/serve.rs:10-15,213,1021,1086; crates/sparkflash-device/src/lib.rs:13-17).
The 35-40% overall and the 75%/25% axis values are a weighting choice; STATUS's own axis table is at
STATUS:105-118.
Disposition: measured facts accepted; the ~35-40% re-estimate rejected as unmeasurable; "3 preflight
gates" must be corrected to 4 fail + 1 not-applicable.

### A10. Timeline projections (Milestone A Oct 28-Nov 1, Milestone B Nov 5-11)
Verdict: UNVERIFIABLE. These are forecasts. No repository artifact encodes them (a grep for the dates
in docs/ returns nothing), and a forecast cannot be confirmed or falsified against a tree.
Disposition: open, no owner (planning input only).

### A11. Every window stops vllm_node on all 4 nodes; ~9-10 min restart+health; 2.5-3.5 h/day
Verdict: CONFIRMED for the per-window cost; the daily total is not additive.
63 of 63 window-b*.sh scripts stop vllm_node on all four nodes with docker stop -t 30 (b68's NODES line
is 192.0.2.80 192.0.2.88 192.0.2.59 192.0.2.68, window-b68.sh:23-27). 42 of 64 logs record a
completed "vLLM healthy after Ns" line in the 520-590 s range (median 560 s = 9.33 min); b68's own log
restarts at 14:05:22Z and reports healthy after 560 s. So "~9-10 minutes" is measured. The daily total
assumes serial windows: the logs show 54 window logs dated 2026-10-19, overlapping window starts
(b47/b48/b49, b59/b60/b61, b66/b67/b68), four windows whose restart ends exit=1 (b59, b60, b66, b67),
and b48 contaminated by b50's overlap. The waiting is a scheduling claim, not a measured sum.
Disposition: accepted for the unit cost; owner Lead for the window protocol.

## B. Plan claims

### B1. Isolate c9fa90f, run a dedicated window to verify head_launches=5 (REQUIRED_KERNELS=10), then merge
Verdict: CONFIRMED-BUT-STALE. The cut-over is real, but the dedicated window already ran and the
constants have moved.
c9fa90f ("c4c reduce cut-over ... REQUIRED_KERNELS 9->10") is an ancestor of b68's binary e9e743e
(git merge-base --is-ancestor c9fa90f e9e743e -> yes). The head A/B window the plan asks for is b61
(420407e, with SPARKFLASH_HEAD_HOST_ASSEMBLY=1): the branch's device-forward-chain.md:1560-1567 records
"Head A/B result: the reduce is not the bug", and b68's runs68/seq130x8.log shows block_max_equal=true
at every printed position. head_launches=15 over 3 positions in runs68/router-all-3.log = 5/position,
so the number to verify is confirmed. On the branch tip REQUIRED_KERNELS is [&str; 11]
(crates/sparkflash-device/src/cubins.rs:41 on task/device-engine-m3-hybrid-sequence; the 11th is
flash_mla_csa2_seq, D36), and on main it is still [&str; 9] (cubins.rs:40). Main's STATUS-2026-10-19.md:23-25
still calls c9fa90f "unreviewed, unmeasured", which the b61 record supersedes.
Disposition: open, owner Lead + engine-dev. Drop (a): b61 already reviewed and measured it. Keep (b):
the merge is gated on the D37 precision decision.

### B2. D37 precision-first fix on the five attention projections, router stays BF16, then re-run b68
Verdict: CONFIRMED. It is the repository's own recommended path and the approved one.
Weight arithmetic: dense attention 5,069,578,240 B / 40 layers = about 127 MB/layer and 5.07 GB total
(roadmap-to-serving.md:682; architecture/00-overview.md:183). Router BF16 agreement 5.96e-8
(STATUS:80; D37). The fixture tokens 491,367,201,80,3302,201,54,201,4341 and the 3-position router-all
diagnostic are the b68 protocol (STATUS:55-80; device-forward-chain.md:1611-1640 on main).
Disposition: **superseded by addendum I (section I.1): the precision-first remedies are rejected.** The
deliverable is the checkpoint's native FP8/FP4 arithmetic, so this is not an accepted plan; the B2
recommendation is recorded as rejected.

### B3. Point vLLM launch recipe at released snapshot 1bae0e10... instead of deployed heretic 12d58ad0...
Verdict: CONFIRMED, and not yet actioned.
D38 (docs/DECISIONS.md:2206-2224) records the mismatch. The live recipe file
/home/user/phase2-evidence/window-2026-10-18/vllm-launch-recipe.sh still serves
models--pqhaz--DeepSeek-V4.1-Flash-1017-heretic/snapshots/12d58ad0... with --tensor-parallel-size 4.
The recipe is on the node, not in the repo (docs/build/window-runbook.md:26 records its path and sha256),
so the fix is a Lead-owned node action plus a D38-conformant recipe.
Disposition: accepted, open, owner Lead (vLLM launch is Lead-owned).

### B4. Wire flash_mla_csa2_seq_sm121 for start_pos>0
Verdict: CONFIRMED-BUT-STALE; already implemented on the unmerged branch.
At main crates/sparkflash-cli/src/attention_device.rs:681 attention() refuses anything but
start_pos == 0. On task/device-engine-m3-hybrid-sequence a separate attention_sequence() handles
start_pos > 0, loads flash_mla_csa2_seq_launch (attention_device.rs:214) and launches it at
attention_device.rs:874; commit ecde183 wires the decode step, and b68 exercised it for 138 positions.
The kernel is f6dfc738 / 38,080 B, the 13th baseline entry and 12th KernelId (D36).
Disposition: accepted; done-on-branch (engine-dev), open until the branch merges.

### B5. Author attention prologue/epilogue kernels
Verdict: CONFIRMED as a plan; none of the named kernels exists.
kernels/ holds 15 .cu files: the 13 baseline kernels plus rocenante_oneshot_sm121 and
flash_mla_csa2_seq_sm121. There is no rmsnorm_rope, no residual-add, and no compressor/index-key/
top-512 kernel. combine_bf16_routed_with_shared exists only in examples and tests/support (m7_layer0
stages, m17b_moe_shared, b2_harness); no src/ launch module consumes it.
Disposition: accepted, open, owner kernel-dev/engine-dev.

### B6. Keep activations device-resident across layers; stop the 24-coefficient and X/X_l/A_l round-trips
Verdict: CONFIRMED as a plan; the round-trip is explicit in the code.
crates/sparkflash-cli/src/device_chain.rs:510 prints the design: layers 1..39 read B/C/A back and
re-upload them (write_layer_bc/write_layer_a calls at 765, 797, 804, 892, 1130, 1137), and Y is the
host FFN output. device-forward-chain.md:884-892 calls it the declared coefficient round-trip, not a
device-to-device buffer alias. Eliminating it is not done.
Disposition: accepted, open, owner engine-dev.

### B7. Stop re-staging/re-packing weights every token; root cause of 9.5 s/token; ~16.07 GB/token, staging_ms=12,777
Verdict: CONFIRMED as a plan; the numbers are the repo's own, the "root cause" is not A/B-proven.
b41 reports staging_ms=12777 and staged_matrix_bytes=16073744000=16.07 GB/token
(device-forward-chain.md:1303, with the accounting note at 1314-1326 saying it is the c4c chain
staging). staging is 68% of b41's total_chain_ms=18799, so it is a plausible dominant cost, but no
controlled run attributes the 9.5 s/position to it; the loader (crates/sparkflash-kernels/src/baseline.rs,
crates/sparkflash-device/src/staging.rs) does not cache the routed repack across tokens.
Disposition: accepted, open, owner engine-dev; label "root cause" as unproven.

### B8. Pack real Engram shards to /var/lib/sparkflash/engram
Verdict: CONFIRMED as a plan; arithmetic and names check; the packing has not run.
202,758,032,400 B / 4 = 50,689,508,100 B = 47.2 GiB/rank (about 48 GiB per node per the spec). Names:
DEFAULT_ENGRAM_DIR=/var/lib/sparkflash/engram (crates/sparkflash-cli/src/doctor.rs:116),
shard_file_name(1,0,4)=engram-l1-r0of4.bin (doctor.rs:2266), 264-byte rows per D4
(docs/DECISIONS.md:127). The packer exists (crates/sparkflash-engram/src/packer.rs) and the CLI has
pack-engram, but QUALIFICATION.md records it has not been run against the checkpoint, so no shard is
written. Not checked: the "shards 47/48" shard-file numbers (the packer selects tensors by name; I did
not open the safetensors index) and the prefetch queue-depth >= 48 figure.
Disposition: accepted, open, owner Lead window + engram/cli dev.

### B9. HSI indexer is 4.27 ms/layer; one warp per (block,slot) with 5-step __shfl_down_sync; target <1.0-1.5 ms/layer
Verdict: CONFIRMED-BUT-STALE. The diagnosis was right; the fix landed on 2026-10-17, two days before
the review, and the 4.27 ms figure is now a historical row.
The review's numbers live in roadmap-to-serving.md section 6 (170.8 ms/token = 40 x 4.27 at :821;
134 MFLOP in 4.27 ms -> about 31 GFLOP/s and 4.2 MiB -> about 1 GB/s at :834-837), all first-window
2026-10-17 measurements of the old kernel. The fix is 826f4f0 (!35, 2026-10-17, "one warp per HSI
block, stage 2 once - 36.2x on the measured launch"), already on main; the 5-step warp sum is
kernels/flash_mla_csa2_sm121.cu:114-119. Current measured numbers (phase3-indexer.md section 2, and
the live row at roadmap:1377ff): window 2026-10-18, stage 2, 200 launches, p50 indexer proxy 0.0930 ms
at one candidate, attention-proper proxy 0.1562 ms (A-B at 128 candidates), 130-token engine total
0.2492 ms, merged-chain equality 163,838/163,840 exact. Under the A-minus-B proxy the indexer is
about 0.6x the attention proper, not 1.4x; the roadmap itself labels the target "measurable and
proxy-met, not established", which is the honest state.
Disposition: rejected as an action (already done); follow-up doc fix to point 6.1-6.3 at 12.1, owner
docs/Lead. The legitimate residue is that the split is a proxy, not a phase-isolated measurement.

### B10. Infra: driver 580.178.04, MTU 9000, lock SM clocks
Verdict: CONFIRMED (one node-list error).
QUALIFICATION.md's 2026-10-15 version table (:84-101) records driver 580.173.02 against the
>= 580.178.04 gate, rail MTU 1500 against 9000, and the clock cap [0, 2200] never measured; the
doctor's --apply --lock-sm-clocks flag exists. The node list in the claim, ".80 .84 .68 .59", is wrong:
the four nodes are 192.0.2.80 .88 .59 .68 (window-b68.sh:23). .84 is not a cluster node. The current
doctor JSON (2026-10-18) still fails driver-kernel and link-mtu.
Disposition: accepted, open, owner Lead/infra (deferred); correct the node list.

### B11. Run the three 4-node RDMA hardware gates
Verdict: CONFIRMED as a plan; the gates exist, none has run.
crates/sparkflash-comm/tests/hardware_gates.rs: roce_oneshot_two_mib_all_reduce_meets_the_latency_budget
(:217), roce_bits_match_the_cpu_reference_over_ten_thousand_replays (:263),
dual_rail_counters_show_both_interfaces_carrying_half_the_bytes (:329); the bar is
ONESHOT_LATENCY_BAR = 20 us on 10 KiB (:112-114). None has run; QUALIFICATION finding 3 records that no
frame has crossed a wire.
Disposition: accepted, open, owner comm-dev + Lead window.

### B12. Wire 4-rank TP sharding and 83 collectives/token
Verdict: CONFIRMED as a plan; every number matches roadmap 7.2 and D30.
roadmap-to-serving.md:950-978: 40 wo_b fp32 collectives of 20,480 B, 40 MoE bf16 of 10,240 B, 2
Engram bf16 of 12,288 B, 1 logits fp32 of 517,120 B = 83 collectives, about 1.77 MB, about 1.7 ms at
20 us; the one-shot limit is 2 MiB, crossed at 204/170 tokens, so the design chunks prefill to 160
tokens. Per-rank resident weights 87.6 GB and about 43 GB free on the 130.66 GB node are D30's
arithmetic (docs/DECISIONS.md:1684-1699). No TP wiring exists yet.
Disposition: accepted, open, owner engine-dev/comm-dev (Phase 4). Note the review's "517 KiB" is
517,120 B = 517 KB decimal (505 KiB).

### B13. Move GPU backends into sparkflash-device and wire DeviceEngine<RealSession> into serve
Verdict: CONFIRMED (the gap is real).
serve --engine device exists but selects DeviceEngine over FakeSession by default and only over
CudaSession with the cuda feature, where CudaSession is M2's single real kernel launch and produces no
token (crates/sparkflash-cli/src/serve.rs:10-15,213,1021,1086; device/src/lib.rs:13-17). The real
multi-launch chain lives in cli (device_chain.rs, moe_device.rs, attention_device.rs), not in
sparkflash-device. This is the same substantive point as C2.
Disposition: accepted, open, owner engine-dev/Lead.

### B14. Connect scheduler + DeviceKvStore; move the D9 binding; Milestone A cut-over on port 8000
Verdict: CONFIRMED as a plan; pieces exist, the connection does not.
The scheduler has prefill.rs, bypass.rs, dspark.rs and indexer.rs; the DeviceKvStore is specified and
its pool measured (phase2-device-kv-store.md); X-Request-Id is D9. Nothing wires the batcher to the
device chain or the KV store, and serve still serialises one CPU request at a time
(QUALIFICATION.md:473-483).
Disposition: accepted, open, owner engine-dev/scheduler-dev (Phase 5).

### B15. CUDA graph capture/replay, wire MTP/DSpark, build the 6 missing gate producers
Verdict: CONFIRMED as a plan on all three.
No cudaGraph, cuStreamBeginCapture or cuGraphLaunch anywhere under crates/ (grep empty); MTP has no
device-chain call site; the six SPARKFLASH_*_RESULTS producers are missing (QUALIFICATION.md:496-501).
Disposition: accepted, open, owner engine-dev/kernel-dev and comm/engine for the producers.

## C. Architectural claims

### C1. Zero todo!/unimplemented!/FIXME across 177,982 lines; every kernel has a pure-Rust twin; D26 type-safe CUDA driver isolation
Verdict: CONFIRMED (one citation wrong).
The marker grep over crates/ returns 0 (todo!, unimplemented!, FIXME; a plain TODO grep is also 0).
Twin coverage: 11 *_ref.rs modules (engram_head_ref, lm_head_reduce_ref, lm_head_ref, mega_mhc_ref,
mla_ref, mxfp4_ref, mxfp8_ref, rmsnorm_ref, routed_gate_up_ref, router_ref, swiglu_ref) cover the 13
pinned kernels (mxfp8_ref covers dense + wo_a, mla_ref covers csa2 + seq, swiglu_ref includes the
weighted quantise), plus rocenante's reference. But D26 is not "type-safe CUDA driver isolation": D26
is "The Milestone-4 ABI source-text lint is frozen" (docs/DECISIONS.md:1292), and grep for
"type-safe" in docs/ returns nothing.
Disposition: accepted; correct the D26 label in citations.

### C2. >5,700 LoC stranded in sparkflash-cli; it is a [[bin]] crate with no [lib] target; sparkflash-device (7,341 lines) is only the dry-run skeleton
Verdict: MISFRAMED. The stranded code and the dependency-direction point are real; the lib-target
premise and the "only dry-run" description are not.
device_chain.rs 2,698 + moe_device.rs 2,183 + attention_device.rs 823 (the review says 824) = 5,704
lines, so ">5,700" is confirmed, and sparkflash-device is exactly 7,341 lines. But cargo metadata
--no-deps lists sparkflash-cli's targets as sparkflash_cli [lib], sparkflash [bin], two examples and
three tests: src/lib.rs is auto-discovered as the library even without an explicit [lib] section, and
main.rs:20 imports sparkflash_cli::{device_chain, doctor, serve}. So "no [lib] target" is false; "no
explicit [lib] section" is true. The substantive point survives: the package graph has
sparkflash-cli -> sparkflash-device and no package depends on sparkflash-cli (cargo metadata), so the
device crate cannot import the chain code without a cycle. The device crate is also not only the dry
run: it has the 108,017-ChainOps dry-run plan with FakeLauncher, an M2 real CudaSession, and M3-b
staging, but its own doc says "none is a chain execution"
(crates/sparkflash-device/src/lib.rs:1-27).
Disposition: the recommendation (move the backends into sparkflash-device) is accepted and open,
owner Lead/engine-dev; the lib-target and "dry-run only" framing is rejected.

### C3. D30 padding 576->640 wastes +8.22 GB/rank because the kernel assumes a 640-wide tile; 576 is 18x32
Verdict: MISFRAMED. Every number is right; the cause and the remedy are not.
Numbers confirmed against docs/DECISIONS.md:1684-1711: routed total 295,997,276,160 B; unpadded /4 =
73,999,319,040 B = 74.0 GB/rank; padded x640/576 = 82,221,465,600 B = 82.2 GB/rank; delta
8,222,146,560 B = +8.22 GB/rank (+11.1%); cluster x4 = 32.9 GB. 576/32 = 18 exactly
(kernels/m4_common_sm121.cuh:96-97). The kernel does assume 640: b12x_mxfp4_moe_sm121.cu:100-116
asserts K_BLOCKS == 20, the scale row is 20 bytes, and the activation row is WIDTH_PADDED bytes; the
launch strides come from the header constant. But the recorded reason is the b12x warp-MMA layout:
spec 2.1 says "Zero-pad K: 576 -> 640 to stay on b12x warp MMA (-0.9 ms)"
(deepseek_v41_flash_4x_dgx_spark_rust_spec.md:45), and D30's first rule is "what has run wins over
what a reference file says should run" (docs/DECISIONS.md:1664). 576 being 18x32 is true but does not
make the layout free: the kernel's stride/scale-row/launch ABI is written to 640, and D30 records a
change as a new kernel ABI, not a docs edit. The cost is memory + HtoD transfer + compute: the padded
region is masked numerically (b12x_mxfp4_moe_sm121.cu:43-45: "the kernel MASKS the padded region"),
but the K loop still walks 20 MX blocks instead of 18 (+11.1% reduction work) and the rows still
store and transfer 11.1% more bytes. "Or adopt EP=4 (D2)" is contradicted by D30, which supersedes D2
(docs/DECISIONS.md:13).
Disposition: the delta arithmetic is accepted as a real cost and an open trade-off (8.22 GB/rank is
what pushes Engram-resident memory out of fit, D30:1695-1699), owner Lead; the "arbitrary tile
assumption / free to unpad" framing is rejected.

### C4. m4_abi.rs is 13,906 lines / 755 KB, 36 rounds; compiling it slows every incremental build; move it to tests/
Verdict: MISFRAMED. Size and freeze facts are exact; the move is not compilable and the build-cost
claim is not reproducible as stated.
Measured: 13,906 lines, 754,981 B (= 755.0 KB decimal, 737.3 KiB); the module doc says the history
"reaches round 36" (crates/sparkflash-kernels/src/m4_abi.rs:5-6). Composition: production-compiled is
about 5,184 lines - module header 848, the public API of eight include_str! sources plus
header_source/consts/header_constant at 849-902 (about 54 lines), and about 4,282 lines of top-level
guard/parser functions at 2966-7247; test-only is about 8,718 lines (the cfg(test) items at 905-2964
plus cfg(test) mod acquire_guard from 7249 to EOF). So about two thirds of the file is not compiled
in a normal build, and only about 54 lines are the API the launch modules use. Moving the whole file
to tests/m4_abi.rs would not compile: production code imports it - mxfp8_dense_launch.rs:23
(pub const SOURCE: &str = m4_abi::MXFP8_DENSE), mxfp8_wo_a_launch.rs:169, mega_mhc_launch.rs:25,
m4_quant.rs:479-492 (header_constant), baseline.rs:160 (m4_abi::KERNELS), plus the
cert/lm-head/flash-MLA launch modules. The viable change is a split: keep the ~54-line const API in
src/, move the ~8.7k test-only parsers and the ~4.3k test-only-used ones under cfg(test) or into
tests/. Build cost on this host: cargo check -p sparkflash-kernels after touching src/m4_abi.rs is
0.50 s; after touching the 108-line src/lib.rs, 0.53 s; cargo build -p sparkflash-kernels after a
touch is 10.5 s; cargo check --tests is 1.2-5.9 s. Cargo's granularity recompiles the whole crate on
any touch, so m4_abi's own contribution is not separable and no m4_abi-specific tax is measurable
here; the review's magnitude is UNVERIFIABLE with this method (an A/B would need a code edit, which a
docs audit does not do).
Disposition: the "36 rounds / 13,906 lines / a 14k-line lint under freeze" observation is accepted;
the "move to tests/" recommendation is rejected as written; propose the split as an open follow-up,
owner kernel-dev, with ci/check-m4-abi-size.sh kept.

### C5. Six Milestone-5 gates only parse env-var strings; parse a signed/hashed JSON artifact like doctor's anti-spoofing check
Verdict: CONFIRMED (and already documented by the repository).
crates/sparkflash-scheduler/src/gates.rs declares six gates and six #[ignore] tests at :291-374, each
reading SPARKFLASH_*_RESULTS and parsing counts; the six env constants are at :32-42; the unignored
tests at :183-245 only pin thresholds and the parser. They run nothing. This is exactly
docs/QUALIFICATION.md section 5 finding 2 (:496-501): "Six run summaries have no producer ... Nothing
in the repository writes them". The anti-spoofing analogy is real: doctor.rs:766-790
(oneshot_allreduce_us_from_probe_json) refuses a number unless the probe JSON says transport == "rdma"
and fabric == "true", with tests including the loopback shim. What does not exist anywhere is a signed
or hashed JSON artifact mechanism: grep for hmac, "signed artifact" or "artifact signature" over crates
and docs is empty, and the doctor check trusts labels the probe emits, not a signature.
Disposition: accepted; already documented (QUALIFICATION finding 2), so it is a known open item,
owner comm/engine-dev for a real producer. The signed/hashed artifact is a new design, not an existing
mechanism to copy.

### C6. Decouple single-node windows from 4-node downtime: TP=2 on .68/.59, or batch stages per window (cuts the ~10-min restart tax 70-80%)
Verdict: MISFRAMED.
The restart cost is measured (A11: 520-590 s per completed health check). But multi-stage windows are
already the practice: b68's script has 4 "=== RUN" markers, b69 has 8, b66 has 4, b9/b10 have 2 each
(grep -c over the scripts). Windows already overlap (b47/b48/b49, b59/b60/b61, b66/b67/b68), which is
what produced the b48 contamination and four restart exit=1 windows, not a saving. TP=2 has no support
in the repository or the deployment: grep for TP=2 in docs/ is empty, the canonical recipe is
--tensor-parallel-size 4 across four nodes (vllm-launch-recipe.sh, window-runbook.md:26-40), and D38
says the deployed server's weights are a different checkpoint, so its tokens are not parity evidence
anyway. The 70-80% figure is not measured.
Disposition: partly accepted - fewer, non-overlapping windows is a real fix, owner Lead; TP=2 as
described is rejected (new deployment, no evidence, invalid weights per D38).

## D. Disposition summary

| Claim | Verdict | Disposition | Owner |
|:--|:--|:--|:--|
| A1-A4 | CONFIRMED | accepted | - |
| A5, A6 | FALSE | rejected as written | Lead/docs |
| A7 | CONFIRMED | accepted in magnitude | docs |
| A8 | CONFIRMED-BUT-STALE | accepted with correction | docs |
| A9 | MISFRAMED | facts accepted, re-estimate rejected | Lead |
| A10 | UNVERIFIABLE | planning input only | - |
| A11 | CONFIRMED | accepted (unit cost); protocol follow-up | Lead |
| B1 | CONFIRMED-BUT-STALE | open: A/B already done; merge gated on D37 | Lead, engine-dev |
| B2 | CONFIRMED (repo language) | REJECTED per addendum I | Lead, engine-dev |
| B3 | CONFIRMED | open | Lead |
| B4 | CONFIRMED-BUT-STALE | done on branch, open until merge | engine-dev |
| B5, B6, B7 | CONFIRMED | open (B7 "root cause" unproven) | kernel-dev, engine-dev |
| B8 | CONFIRMED | open (two sub-items unverified) | Lead, engram-dev |
| B9 | CONFIRMED-BUT-STALE | rejected as action; doc fix in roadmap | docs, Lead |
| B10 | CONFIRMED | open (node list corrected) | Lead/infra |
| B11, B12 | CONFIRMED | open | comm-dev, engine-dev |
| B13 | CONFIRMED | open | engine-dev, Lead |
| B14, B15 | CONFIRMED | open | engine-dev, scheduler-dev, kernel-dev |
| C1 | CONFIRMED | accepted (D26 label fix) | docs |
| C2 | MISFRAMED | move accepted, framing rejected | Lead, engine-dev |
| C3 | MISFRAMED | cost accepted as trade-off, remedy rejected | Lead |
| C4 | MISFRAMED | split proposed, move rejected | kernel-dev |
| C5 | CONFIRMED | accepted, already documented | comm/engine-dev |
| C6 | MISFRAMED | fewer non-overlapping windows accepted; TP=2 rejected | Lead |

## E. What the review gets right

The measured core is sound, and several next steps match the repository's own record: B2's precision-first framing matched D37/STATUS section 4 as written, but addendum I rejects that remedy: the deliverable is the checkpoint's native FP8/FP4 arithmetic; the numerics facts (0/9, 8060 vs 491, 15.01
vs 0.02, layer-3 flip, 36/40) are exact; the missing env-summary producers (C5) are exactly
QUALIFICATION finding 2; the stranded-chain diagnosis (C2), the m4_abi size/freeze (C4), and the
16.07 GB/token restaging (B7) are real and match the docs; the restart cost (A11, 9.33 min measured),
the never-run RDMA gates (B11), the missing cudaGraph executors and MTP wiring (B15), and the
"-0.9 ms padding buys 8.22 GB/rank" tension (C3) are genuine; the code-quality facts (C1: zero
markers, full twin coverage) are accurate.

## F. Where this repository's own docs are stale

1. docs/STATUS-2026-10-19.md:9 says "Main is e39364d" and its table stops at !110; main is now 7edda55
   (!112).
2. STATUS:23-25 calls c9fa90f "unreviewed, unmeasured", but the branch records the dedicated b61 head
   A/B result (device-forward-chain.md:1560-1567 on task/device-engine-m3-hybrid-sequence). One of the
   two needs to move.
3. roadmap-to-serving.md section 6.1-6.3 (:61, :821, :834-837) still carries the pre-!35 4.27 ms / 65% /
   31 GFLOP/s indexer numbers while section 12.1 has the current 0.0930/0.1562/0.2492 ms rows. This is
   why B9's stale numbers still look citable.
4. QUALIFICATION.md's "measured 2026-10-15" version table (:84-101) is presented as current; the node's
   doctor-local-2026-10-18.json has moved to 0 passed / 4 failed / 1 not-applicable.
5. device-forward-chain.md:1314-1326 already corrects the b41 staging accounting that B7 quotes; a
   reader needs the note, not the headline.

## G. Where the repository has already superseded the review

- The HSI warp-dot fix the review asks for landed as !35 / 826f4f0 on 2026-10-17, two days before the
  review; the current rows are phase3-indexer.md section 2 and roadmap 12.1.
- The head-reduce cut-over the review wants isolated was already A/B-tested in b61, and b68 inherits it
  with block_max_equal=true at every printed position.
- The b68 full 138-position run, flat memory and the fixture validity are already the STATUS record the
  review's A9 uses.
- D37 states the MXFP8 root cause; its precision-first options are rejected by the user directives in
  addendum I.

## H. What this note does not do

- It does not re-run any window or GPU work, does not stop or start vLLM, and does not merge anything;
  the node reads are read-only ssh greps and one scp of the doctor JSON.
- It does not verify B8's "shards 47/48" or the prefetch queue depth (the safetensors index was not
  opened), A10's dates (a forecast has no evidence), or the reviewer's 35-40% weighting (no measurement
  defines it).
- It does not attempt the m4_abi split or the C2 move (docs-only constraint); those are recorded as
  open follow-ups with owners.
- It reports 62 windows and 95/96 MRs, not 68 and 110; every count above is a command in section 0.

## I. Addendum 2026-10-19: native arithmetic and acceptance (user directives)

Recorded after the audit, relayed by the Lead from two user directives. This section supersedes the B2
disposition and the "precision-first" framing the audit inherited from the repository.

### I.1 The deliverable is the checkpoint's native arithmetic (FP8/FP4 as trained)

Evidence from the released forward (paths outside this repo, read-only):
- /home/user/dsv41-research/inference/model.py:27-30 sets fp8_block_size = 32 ("one fp8 scale per
  32x32 weight block / 32 activations"), fp4_block_size = 32 ("one fp4 scale per 32 elements along K"),
  scale_fmt = "ue8m0", scale_dtype = float8_e8m0fnu. (The Lead's "29-32" is this four-line block at
  27-30 in the copy on disk.)
- model.py:182-205 linear(): an FP4 weight takes act_quant(x, 32, "ue8m0", e8m0) then fp4_gemm; an FP8
  weight takes the same act_quant then fp8_gemm. Every quantised GEMM consumes E4M3 activations with
  one ue8m0 scale per 32 elements. There is no per-1x128 exact-scale path and no BF16-activation path in
  the checkpoint's forward.
- /home/user/dsv41-research/inference/kernel.py:34-40, 41-96, 98-124: act_quant turns scale_fmt into
  round_scale, and fast_round_scale is 2^ceil(log2(amax/448)) with amax floored at 1e-4 - the
  power-of-two ceil rule.
- D23 (docs/DECISIONS.md:1002-1070) already records exactly this as the released forward's single
  dynamic activation scale rule, with the weight-grid rule kept separate.

Consequences:
1. The review's B2 remedies - "exact per-1x128 FP8 dynamic scales (not ue8m0 pow2 ceil)" and "BF16
   activations against dequantised FP8 weights" - are arithmetically wrong for this checkpoint: they
   compute a different model, not a more faithful one. B2 is reclassified from accepted to REJECTED
   (its section-B disposition is amended above).
2. The f32 golden fixture is a simulation (I.3 proves the source); device-vs-fixture differences are
   expected by construction. They are diagnostics, not device defects, and the device path's native
   arithmetic is the target, not a higher-precision approximation of it.
3. A9's numbers remain valid measurements (0/9, 8060 vs 491, 15.009581, layer-3 flip, 36/40, 8.7-9.6 s,
   1239.0 s, 16.07 GB), but "criterion 1 measured failing" is a statement against a simulated reference,
   not against the model.

### I.2 Token equality is not an acceptance test

Directive: generation is not deterministic, so no acceptance rule may require token-for-token equality;
acceptance is numeric/distributional with an explicit tolerance, and the behavioural oracle is vLLM
("when in doubt, do what vLLM does"). D38 still applies: a vLLM comparison is only meaningful on the
same snapshot and served root.

Consequence: criterion 1 as written is not an acceptance criterion. The user's remaining decision is a
redefined criterion - same-arithmetic reference plus tolerance and distribution, with vLLM as the
behaviour oracle - not the precision-first path D37/STATUS put first.

### I.3 Verified: the f32 reference engine does not implement the native GEMM arithmetic

- Every GEMM is f32 over dequantised weights: linalg::matmul multiplies f32 x by f32 w
  (crates/sparkflash-engine/src/linalg.rs:219-283); weights decode to f32 at linalg.rs:116
  (quant::decode) and their stored E8M0 scale grids are applied in f32 at attention.rs:696-709
  (scaled_or_plain_weight -> linalg::apply_scales).
- No native activation quantiser runs on a GEMM operand in forward/: forward/mod.rs:4 says "no quantised
  kernels", and the only engine call to quant::quantize_activation_fp8 in engine/src is boundary.rs:451,
  which fills the boundary table's captured E4M3 codes, not a compute path.
- Nuance, so the claim stays true: the engine does model the model.py element round-trips written
  explicitly in the attention code - q_index FP4 (attention.rs:1074; model.py:546, 552), window KV FP8
  (attention.rs:1752; model.py:707), index keys FP4 (attention.rs:1841; model.py:760), compressed latent
  FP4 with an E4M3 scale (attention.rs:1896; model.py:760, kernel.py:160-163). What it omits is the
  native network on GEMMs - rows 1-2 of docs/engine/01-forward-order.md:524-546 (model.py:187-205,
  186-195) - which is exactly the rule the device runs. So "the f32 reference engine quantises nothing"
  is literally false; "it quantises nothing on the GEMM path, which is the path that differs" is proven,
  and that is enough to establish the fixture as a simulation.
- Provenance corroboration: phase2-golden-fixture.md:21-23 calls the fixture "the engine's CPU reference
  forward ... over the real released checkpoint", and docs/engine/01-forward-order.md:546 records that
  the reference quantises and dequantises rows 3-6 while keeping bf16 buffers.

### I.4 Repo docs to correct (listed only, per instruction; Lead owns the edits)

| # | doc:line | what it says | why it is wrong under the directives |
|:--|:--|:--|:--|
| 1 | docs/STATUS-2026-10-19.md:89-95 (:90, :92) | option 1: "BF16 activations with dequantised weights"; "the model's native FP8 activation rule (per-1x128 block, exact scale rather than power-of-two ceil)" | the native rule is per-32 ue8m0 power-of-two ceil (model.py:27-30, 182-205; kernel.py:98-124; D23); the wording calls a simulation "native" |
| 2 | docs/DECISIONS.md:48 (D37 index row) | "exact token parity needs either a precision-first activation path or a redefined criterion" | precision-first is not the model arithmetic; token parity is not acceptance (I.1, I.2) |
| 3 | docs/DECISIONS.md:2143-2224 (D37 body; :2147, :2158-2165, :2202) | same two options, with (2) "if the precision-first path is chosen, that is new API-compatible kernel variants" | same; the ordering and the framing both need the directives applied |
| 4 | docs/build/device-forward-chain.md:1563 (context 1520-1568) | "exact 130+8 token parity is not attainable without a precision-first activation path or a redefined criterion" | token parity is not the criterion; precision-first is the wrong remedy |
| 5 | docs/STATUS-2026-10-19.md:49-51 | "fixture is valid. The live in-process f32 reference reproduces all nine fixture tokens exactly ... every divergence below is a device-arithmetic divergence" | token equality as validity test; the f32 reference is a simulation on GEMMs (I.3) |
| 6 | docs/DECISIONS.md:2172-2173 (D37) | "the fixture is valid: the live f32 reference reproduces all nine fixture tokens exactly" | same |
| 7 | docs/build/roadmap-to-serving.md:788-795 (5.5 item 1) | criterion 1: "every decoded token reproduced against the reference (token-for-token, not just the last)" | token equality as the exit criterion (I.2) |
| 8 | docs/build/device-forward-chain.md:364-374 (4.3) | "every decoded token reproduced against the reference", with "0 of 9" in the PASS text | token equality as PASS; needs a same-arithmetic reference and a tolerance/distribution bar |
| 9 | docs/STATUS-2026-10-19.md:53-58 (section 3) | "The criterion is one --phase full --prompt 130 --decode 8 run reproducing the tokens 491, 367, ..." | token equality as criterion (I.2) |
| 10 | docs/DECISIONS.md:36, 1243-1268 (D25) | fixture is "Phase 2 reference truth"; device output "compared token-first" | the fixture is a simulation; token-first comparison is not acceptance |
| 11 | docs/build/phase2-golden-fixture.md:3-11, 141, 156 | "reference truth", "compare_steps token-first", "must not be regenerated" | same; the provenance sections are otherwise accurate |
| 12 | docs/build/phase2-sequence-harness.md:114-116 | comparison contract: "next-token equality, then the logits SHA-256" | token equality used as a bar; the CI fixture comparator is fine as a schema test, the wording is the acceptance language |
| 13 | docs/build/serving-readiness.md:1010-1015 | "fixture is confirmed valid (live f32 reference reproduces all nine tokens exactly)", criterion-1-failing framing | token equality plus a simulated reference as the standard |
| 14 | docs/STATUS-2026-10-19.md:71-72; docs/DECISIONS.md:2158-2163 | "the device attention path quantises activations ... while the fixture reference is f32 over dequantised weights" | arithmetic accurate, framing reversed: the device is native, the fixture is the simulation |
| 15 | docs/build/review-2026-10-19-feedback.md sections B (B2), D (table), E, G (this note) | B2 accepted; precision-first called the repository's recommendation | corrected by this addendum (B2 rejected; sections E/G reworded) |

The many "criterion 1 remains OPEN" lines in device-forward-chain.md section 7 and the phase3 pages
inherit section 4.3's definition and need no separate edit once 4.3 and D37 are corrected. The
architecture and engine docs checked (docs/engine/01-forward-order.md section 6,
docs/architecture/01-csa2-attention.md, docs/architecture/03-moe.md) state the native rule correctly
and are not on this list.

