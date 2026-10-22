# Review: "Decoupled Chunk Optimization Plan" (PDF, 4 pp.) - verdict, corrections, what to adopt

Review of `sparkflash_chunk_optimization_plan.pdf` against the tree at `main` = `72d09a32` and the
integration branch `win/layer-device` = `5d5a5ffd`. The PDF's own citations were checked; stale ones are
named with the current value.

## Verdict in one paragraph

The **method** is good and should be adopted: the expensive thing on this cluster is a four-rank window, so
kernel and seam work must be provable **without** one - host-oracle first, then single-GPU correctness, then
single-GPU timing, then a graph-capture check, and only then a batched four-rank row. The **facts and the
gate are stale**, and two of them would actively undo standing decisions: the plan still gates on "byte
identity" via `scripts/identity-gate.sh` and logits SHA `aed6054d...`, which is exactly the gate that was
replaced by the tolerance-based numerical gate; and it plans per-chunk seam optimisation as the path to
90 tok/s, where the plan of record is now to move the **whole layer** to the device and not measure partial
ports. Read as a **test-rig design** it is the most useful document of the three; read as a **roadmap** it is
one revision behind and would send work backwards.

## What is right and should be taken as-is

1. **The tier structure.** Tier 0 (host, no GPU): arithmetic model, ABI static asserts, baseline manifest in
   the hermetic CUDA 13.0 container. Tier 1 (single node, seconds): one correctness launch, then a timed row
   with warmups, emitting `p50_ms`, achieved GB/s and `pct_of_node` against the 273 GB/s floor. Tier 2 (four
   ranks, batched, only for a cumulative win). This is the correct economics and it is what the branch has
   been doing informally; the document makes it a rule.
2. **`.chunk` fixtures** - a self-contained bundle of pre-tiled Rank-0 weights plus the 18 boundary tensors
   for four canonical layers (0 ordinary, 1 Engram, 2 compressor, 14 Engram+compressor) at one decode
   position and a 6-position window. **This is the single most valuable idea in the document for where we
   are now**: the port's stages are written and unvalidated, and a chunk fixture is what lets them be
   validated on one node in seconds instead of in a four-rank window that does not exist yet.
3. **`--floor-gib 1` for chunk harnesses** (`examples/m12_attn_projections.rs` still defaults to 32). A
   <350 MB microbench can run beside a resident server; a 32 GiB floor demands the server be stopped. This is
   an operational unlock, not a convenience.
4. **Roofline discipline**: compute the floor from traffic before coding, target >= 60 % of 273 GB/s, print
   `pct_of_node` in every row. Keep.
5. **The three mandatory `window-kernels.sh` fixes** are real defects: (a) accumulate-in-place kernels
   (`b12x_mxfp4_moe_sm121`, `..._stored_sm121` add into `slot_acc`) must be correctness-run at `--repeat 1`
   and timed separately at `--repeat 200`, or the twin check compares 201 summed launches; (b) the routed-MoE
   roofline traffic must be the 6 experts actually read (`tokens*6*5120*(288+18)` = 9.40 MB stored /
   10.57 MB padded), not 64 staged experts (111.54 MB); (c) memory floor as in (3).

## What is stale, and the current value

| The plan says | The tree says now |
| --- | --- |
| Baseline tip `4f83eee5`, **350.39 ms/step** | Published host-arm figure **179.81 ms/step** (four-rank clean window, D54 numerical PASS); `main` = `72d09a32` |
| Gate = byte identity, logits SHA `aed6054d...` | **Superseded**: greedy tokens identical over N prompts x M tokens, logit max-abs/KL within a declared tolerance, top-k agreement, with the bit-exact CPU twin as the reference. `scripts/numerical-gate.sh` + frozen `fab52c00a0dc6f6c` reference exist **on the port branch only** - see action 1 below |
| "329 s cold-start upload tax per row" | The load path was corrected: rows never used the O_DIRECT loader; the FNV-1a digest alone was ~121 s of the ~254 s, and the parked-pool fix takes the upload to **~100-110 s**. A row is ~2 min of upload, not 6-7 |
| "12 synchronous `cuMemcpyDtoH_v2` per layer-position" | That is the **diagnostic** loop. The lean host loop and the device arm do not do it: the device arm has exactly **two** crossings per position (`begin_position` 81,920 B H2D, `residual_out` D2H), both outside the layer loop |
| Kernels `mega_mhc_sm121`, `m10_gate_up`, `m11_mhc_entry`, `m15_mhc_post`, `m20_csa2`, `m26_moe_stage` | Experiment names, not pinned symbols (`mega_mhc` appears **0** times in `kernels/`). The pinned seam image is `layer_seams_sm121` with `mhc_entry_bf16`, `mhc_ffn_seams`, `w8a8_prologue_e4m3`; the kv image is `kv_path_sm121` |
| Owners: kernel-lead, engine-dev, device-lead | Those roles do not exist in this team; the work is split by branch (gemv / driver-fixer / comm-lead / qa-lead / review-critic). A backlog with dead owner names cannot be executed |
| Per-island wins "-66..-86", "-76..-102", "-140..-180 ms/step" | Derived from the 350.39 baseline and **not additive** (they overlap: island C contains the shared expert and the epilogue island B also touches). They must be re-derived per chunk from the current 179.81 figure, and never summed |

## Two places where the plan must not be followed

1. **The gate.** Adopting `identity-gate.sh` + logits SHA as the acceptance rule for chunk rows would
   re-instate the gate that was deliberately replaced. Chunk rows are **correctness** rows and their
   acceptance is the twin (bit-exact where the kernel is bit-exact) or a **declared tolerance** where
   reduction order changes - the same D51/D54 vocabulary, with the CPU twin as reference. A logits SHA is a
   row-level identity check on the host arm, not a chunk acceptance bar, and vLLM itself is not bit-identical
   to any reference.
2. **"Optimise chunks instead of porting the layer."** Sequencing is the other way round now: the layer goes
   to the device as a unit (plan of record, `docs/build/ws-whole-layer-device-port-20261022.md`), and the
   chunk rig is its **test bench**. Islands A, B and C in the plan are, almost stage for stage, the device
   arm's stage groups; building them as separate harnesses first and wiring the loop later is the detour that
   was explicitly ruled out. Also unchanged: graph capture comes **after** the layer is fully device-side,
   which the plan's Phase C half-anticipates but orders before the seam removals.

The plan's batching rule ("no four-rank window for a single slice; batch until >= 50 ms/step") also needs one
carve-out: **correctness/frontier rows are exempt.** A probe row that stops at a stage name has zero ms/step
of win and is still worth a window, because it is the only thing that tells us whether the port is correct.

## Actions I would take from this document

1. **Land the tolerance gate on `main` as its own MR**, independent of the port branch: `scripts/numerical-gate.sh`
   plus `docs/build/reference/speed-gate/fab52c00a0dc6f6c/rank{0..3}.json`. It exists only on the port branch
   today, which means the tree's own standard still contains the SHA gate the document assumes.
2. **Build `dump-chunk` and capture the four layers in the first window that is granted**, piggybacked on the
   probe row (the run already has the arena resident). That single window investment converts every later
   stage-correctness question from a four-rank window into a seconds-long single-node run.
3. **Fix the three `window-kernels.sh` defects now** - they need no GPU and they make existing evidence rows
   wrong (the accumulate-in-place one mislabels correctness).
4. **Drop `--floor-gib` to 1 in the chunk harnesses** and record in each row that it ran beside a resident
   server, since that changes nothing numerically but does change what "clean row" means for timing.
5. **Re-issue the backlog with live owners and re-derived numbers**, or the tickets stay decorative.

## Method note

The PDF was read with a purpose-written decoder (its text is CID-encoded; no `pdftotext`/Python PDF library on
this host). Table cells that carried glyphs in a symbol font lost digits in extraction, so **every number above
was re-derived from the tree, not from the PDF**, and the PDF's own per-chunk wall times should be treated as
unverified until someone re-extracts them with the font mapping intact.
