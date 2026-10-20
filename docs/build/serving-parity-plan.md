# Serving parity with vLLM: plan and task board

**Status: Lead-owned plan, 2026-10-19.** The work items live on the shared team task board (ids
`task-4` .. `task-27`); this page is the human-readable index, the definition of parity, the
critical path and the measurement protocol. Nothing here reopens a decision: the arithmetic
direction is [DECISIONS.md](../DECISIONS.md) D39, the checkpoint/recipe facts are D38, and the
audits that produced the work items are [native-arithmetic-audit.md](native-arithmetic-audit.md)
(engine/deviation register F1-F8) and
[native-kernel-arithmetic-audit.md](native-kernel-arithmetic-audit.md) (kernel mismatches M1-M4).

## 0. What "serving parity" means here

Parity is three gates, not a token comparison:

- **A. Native conformance.** The serving chain computes the released checkpoint's arithmetic
  (D39 rule 1) and its output distributions agree with the vLLM-released oracle within bars that are
  declared before the run (numeric/distributional: top-k overlap, logprob deltas, KL - never token
  equality, D39 rule 2).
- **B. Serving behaviour.** The serving path exposes vLLM-compatible behaviour on the same requests:
  logprobs/top_logprobs/prompt_logprobs semantics, format-2 CSA2 KV with paging, continuous
  batching and chunked prefill, and the same tensor parallelism (TP=4 across the four GB10 nodes) so
  the comparison is like-for-like.
- **C. Measured performance.** Prefill tokens/s, decode tokens/s, TTFT and aggregate throughput are
  measured against the recorded vLLM TP=4 baseline on the same hardware, prompt suite and batch
  sizes, with perf bars declared before the run.

Non-goals: the precision-first family (host-f32 `dd4184e`, and the **dense** BF16-activation
kernels of `6c1d582`) is rejected by D39 and stays unmerged - with one documented exception: the
`b12x_bf16a_mxfp8w_wo_a` variant is not a precision experiment, because native's `wo_a`
activation is bf16 (`model.py:766-788`); it is the native rule for that projection, while the
dense bf16a variants remain rejected and F1's isolation is still unmeasured. Token-for-token
equality is retired; Engram tables stay disk-offloaded (user directive), not staged into HBM.

## 1. Where the work starts (2026-10-19)

| fact | state |
| :--- | :--- |
| Decision D39 (native arithmetic, numeric bars, vLLM as oracle, Engram on disk) | merged (!114) |
| D38 recipe fix (cluster serves the released snapshot) | actioned; window recipe and sha256 recorded in D39 and the runbook |
| Kernel-level audit M1-M4 (wo_a BF16, MoE combine, unweighted SwiGLU, mega_mhc) | merged as docs (b773e87) |
| Stage-level audit F1-F8 (wo_a HIGH, body probs, indexer, Engram, pre-rounds, combine, f32 libm) | merged as docs (15cf26e) |
| Hybrid device chain merge (native path, 19 files, +3571/-266) | resolved on `task/hybrid-merge` (gates green: 1476 tests); S1 lands it |
| Machine-readable logits dump (`--dump-logits`) | in tree: `crates/sparkflash-cli/src/logits_dump.rs` (body from `task/logits-dump` `4f599b7`, unchanged); the one schema is [logits-dump.md](logits-dump.md); O6 done, host-tested, no device run |
| CI check stage | green again (missing doc on `attention_forward_with_backend`, fixed in !113) |
| Exploratory position-129 native-vs-vLLM comparison | b69 window (dump `runs69/logits-130x0.jsonl`, oracle staged) - recorded by O4 |

The audits name the deviations that make A unattainable today, in priority order: **M2** (MoE
combine rounding/order/residual), **M3** (unweighted SwiGLU BF16 round), **M1** (wo_a is BF16 in
native, FP8 in the chain - the only projection where the chain is coarser), **M4/F2** (`mega_mhc`
BF16 residual stream and roundings, body softmax probabilities to BF16 before P*V), then **F3**
(indexer), **F4** (Engram), **F5** (pre-rounds), **F8** (f32 library functions).

## 2. Task board (team tasks `task-4` .. `task-27`)

| id | task | depends on | gate |
| :--- | :--- | :--- | :--- |
| task-4 | N1 native MoE combine (M2/F7) | - | twin-exact + pin + N9 |
| task-5 | N2 native unweighted SwiGLU (M3/F6) | - | twin-exact + pin + N9 |
| task-6 | N3 native wo_a (M1/F1): BF16 grouped einsum | - | bf16 reference delta + pin + N9 |
| task-7 | N4 body softmax probabilities to BF16 (F2) | - | twin + pin + N9 |
| task-8 | N5 indexer native path (F3) | - | topk_idxs comparison + N9 |
| task-9 | N6 Engram native dtype on the disk-offloaded seam (F4) | - | native-rule row comparison |
| task-10 | N7 BF16 pre-rounds at host seams (F5) | - | per-seam verdict + N9 |
| task-11 | N8 f32 library functions (F8) | - | aligned or documented delta |
| task-12 | N9 re-run stage-exactness after every native fix | task-4..10 | window record per fix |
| task-13 | O1 declare the serving-parity bars as D40 | - | D40 in main before any measured claim |
| task-14 | O2 native oracle harness (all positions, prompt suite) | - | oracle JSON + recorded root |
| task-15 | O3 device-vs-oracle comparison harness | task-14 | bars applied + evidence |
| task-16 | O4 record the b69 position-129 comparison | task-15 | dated window subsection |
| task-17 | S1 merge the hybrid device chain into main | - | CI green, ff merge |
| task-18 | O5 pin the prompt suite | task-14 | suite + sha256 |
| task-19 | O6 integrate the logits dump into the merged chain | task-17 | dump available in acceptance runs |
| task-20 | S2 logprobs API parity in serve | task-17 | response/shape/numeric vs vLLM |
| task-21 | S3 paged format-2 device KV in serve | task-20 | numerics + paging invariants |
| task-22 | S4 continuous batching + chunked prefill | task-21 | numerics across batch sizes + perf |
| task-23 | S5 TP=4 (tensor parallel + RDMA) | task-21 | TP=4 numerics + transport tests |
| task-24 | S6 Engram on NVMe in the serving path | task-17 | measured read overhead |
| task-25 | S7 performance parity vs vLLM TP=4 | task-22, task-23, task-13 | perf bars met |
| task-26 | S8 serving-parity gate + decision record | task-15, task-18, task-25 | gate passed and recorded |
| task-27 | Backlog: D30 padding, `m4_abi` split, gate-artifact JSON | - | not blocking |

Owners are assigned by the Lead at pickup; write scopes are recorded on each task. The board is
authoritative for status; this table is a snapshot.

## 3. Critical path

1. **Now, in parallel:** task-13 (declare bars - it must exist before the next measured claim),
   task-17 (merge the device chain), task-14 (oracle harness), task-4 + task-5 (the two cheapest
   native fixes with the widest effect).
2. **Then:** task-7 (body probabilities), task-12 after each fix, task-3 (b69 record) once the
   oracle JSON exists (the b69 dump is already on disk).
3. **Then:** task-6 (wo_a - the largest; plan a dedicated window), task-8, task-9, task-10, task-11.
4. **Serving:** task-19, task-20, task-21, then task-22 and task-23 in parallel, then task-24.
5. **Gate:** task-25, then task-26 (D40/D41 record).

Rationale for the order: M2/M3 are local, high-frequency (every MoE token) and cheap to verify
against the twins; wo_a is the single coarsest deviation but a bigger kernel/host change, so it gets
its own window after the cheap ones land; the oracle harness and bars must precede any claim, and
the merge (S1) unblocks all serving work.

## 4. Measurement protocol (fixed before bars are applied)

- **Oracle:** vLLM serving the released snapshot `1bae0e10...` TP=4; the served root is recorded
  from `/v1/models` in every evidence log; `--max-logprobs 1000`; requests at temperature 0 with
  `prompt_logprobs` over the pinned prompts, so each compared position yields an id-keyed
  distribution (the b69 procedure: fixture prompt + one extra token, read the last prompt position).
- **Device side:** `--dump-logits` writes per-position JSONL (device argmax, logits argmax, vocab,
  byte offset, top-20) plus the full-vocabulary f32 logits sidecar, which is the offline comparison
  input. The schema and its host tests are frozen in [logits-dump.md](logits-dump.md).
- **Comparison:** per position and aggregate over the prompt suite - top-1 agreement, top-k overlap,
  max/mean |logprob delta| over the oracle top-k, and KL of the renormalised top-k distributions.
  Bars are declared in D40 before a run is a gate; the b69 comparison is exploratory and labelled as
  such.
- **No token equality anywhere** (D39): generation is not deterministic across implementations.

## 5. Ops and windows

Windows remain Lead-owned ([window-runbook.md](window-runbook.md)): stop the serving cluster on all
four nodes, run the device evidence, then start the released recipe and verify health before the
next claim. The released snapshot's model load is the long pole in a window (observed >25 min before
the server answers), so measurement windows are planned around one load, and the oracle script waits
for the server rather than being polled.

## 6. Open risks

- **wo_a (task-6)** may need a new BF16 grouped-einsum kernel or a host BF16 GEMM path; it is the
  one place where the chain is coarser than native, so it dominates gate A.
- **TP=4 (task-23)** requires sharding and RDMA all-reduce; vLLM's TP=4 layout must be matched or the
  comparison is not like-for-like.
- **Twin blindness (all N tasks):** device-vs-twin bars are self-consistency (audit class B) and
  cannot see a rule both sides share - only the vLLM oracle (class C) can, so every fix needs a
  measured oracle comparison, not just a green twin.
- **Engram NVMe (task-24)** read amplification under batching is unmeasured; the user's directive
  is that the drive is fast enough, but the number still has to be recorded.
- **Model load time** makes each vLLM window expensive; batch oracle work per window.

## Addendum 2026-10-19T17:30Z - user directives: TP=4 first, and vLLM is itself Engram-disk-served

User, verbatim: *You cannot run this model on a single node. I think I would rather you just go for
TP=4 first, then experiment with EP on top of that once you have TP=4 working.* and *we do patch
vLLM to use the disk served engram as well*.

Consequences for this plan:

- **task-23 (S5, TP=4) is the first parallelism work, not a later stage.** Single-node runs remain a
  numeric/correctness reference only (the device chain computes the same arithmetic while streaming
  weights), while serving and every performance measurement happen at TP=4 across the four nodes,
  because the released checkpoint is 475 GiB and a node has 121 GB.
- **task-31 (S5a, EP=4) is created and blocked by task-23**: expert parallelism is an experiment on
  top of a working TP=4, not a replacement for the parity baseline.
- **Engram-on-disk is not a deviation from the oracle**: the deployed vLLM is patched to serve the
  Engram tables from disk too, so the disk-offloaded table is part of the like-for-like comparison
  (D39 rule 4 stands, and task-9 now includes mirroring the patch's disk read path after inspecting
  it in the container).
- **The per-position weight staging is not a serving configuration**: b68 measured 89.45 ms/position
  of attention staging and 5.066 GB/position H2D for the five projection weights. TP=4 requires
  resident sharded weights per rank - measured from the checkpoint headers as **72.31 GiB**, not the
  ~113-119 GiB that the checkpoint-total-divided-by-four figure implies (that number includes the
  188.83 GiB of Engram tables resident, which is the configuration that wedges a 121.69 GiB node) -
  plus KV per rank (~15 GiB at 3.68M tokens in the reference), RDMA all-reduce, and a four-process
  driver; with Engram on disk the all-in budget is ~90.4 GiB per rank (tp4-design.md section 3). `crates/sparkflash-comm` already carries a dlopen-libibverbs
  RDMA module, an exchange module and a probe binary - tp-dev surveys it before proposing new
  transport work.
- **tp-dev owns the design** (docs/build/tp4-design.md, reviewed by the Lead before implementation)
  and the staged implementation tasks; wo_a (task-6), MoE combine (task-4) and the other native
  fixes stay single-rank work and are revisited for resident weights when TP=4 lands.

## Addendum 2026-10-20 - K8 batched kernels and the exact router measured on device

The batch-19 kernel rewrite was measured on a GB10 (node0) with its own harnesses after two
rounds of fixes. Post-fix state, all from the tree's harnesses against rebuilt pinned cubins:

- **Router (`m9_router`): PASS 7/7.** The K3-final warp-tree reduction had broken the serial
  ascending-k contract the engine host path defines (scores 1.4305115e-6 / 131 ULP, weights
  2.3841858e-7 / 21 ULP against bars 1e-6 / 8 and 1e-6 / 16, ids still 48/48). Restoring one
  active lane per token with ascending k gives ids 48/48, scores 1.1920929e-7 / ULP 1, weights
  5.9604645e-8 / ULP 2, determinism byte-identical - at **0.2279 ms per 8-token launch /
  0.0285 ms per token**, about 2.1x the (wrong) tree version's 0.1084 ms / 0.0136 ms. The exact
  reduction has a real step cost: 40 layers is ~1.14 ms/token of router, counted in D48.
- **Batched six-row kernels, all byte-exact after the fixes:** cert head 6/6 rows identical
  (2.85x one scalar launch, ~2.10x vs six sequential, 11.0% of the traffic roofline), mHC 4.32x,
  sequence entry 6/6 identical (0.597x six sequential), lm-reduce 1.01x (no win - not a lever).
- **The first device attempt failed, and that is the reason the gate exists:** m22/m23/m20 all
  mismatched (m20's published index arrays came from a stage-1 loop that carried per-worker bests
  across blocks and only covered 512 of 2048 blocks), and the router failed two numeric bars. The
  harnesses + rebuilt pins caught it; no speed number from the broken tree is carried anywhere.
- **Still not measured:** a token, a verify-step time, the 134-slot ring numerics, CUDA-graph
  capture, RDMA one-shot. The 767.7 ms/step = 1.3026 tokens/s batch-18 row remains the best
  measured decode until a device run beats it.
