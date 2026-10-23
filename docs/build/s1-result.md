# S1 result: the device-only step — state at this workspace, and what still blocks it

**Status: NOT CLOSED.** The S1 deliverable — *one step where all 40 layers run on the GPU with nothing
copied back to the CPU, passing D54, at ≤ 120 ms/step* — has **not** been demonstrated. The published
number stays **179.81 ms/step**.

What did happen: **S1-0 was run on the four GPU nodes on 2026-10-23 and passed** (the device arm opens on
all four ranks, the mHC-seam bind is `400 expected = 400 bound`, the arena bind is complete, and the
40-layer *reference* collective walk (TCP, the canonical skeleton - not the NCCL arm) executes its 80
reduces/step). The device arm then runs its wired prefix — nine
stage bodies on layer 0, position 0 — and refuses deterministically at `attention_body`, which is stage
10 of 20. No step completes, so **S1-a/S1-b/S1-c did not run, no D54 comparison ran, no step was timed,
and no timing row exists**. Section 2 has the probe and its evidence; section 3 the frontier; section 7
what is still required.

This host has no CUDA device (`nvidia-smi` absent, no `/dev/nvidia*`, `/proc/driver/nvidia` absent, no
CUDA libraries). The four DGX Spark nodes (`node0`, `node1`, `node3`, `node2`:
aarch64 GB10, one per rank) are reachable over SSH and hold the released checkpoint, so the cross-built
`aarch64` binary was staged and run there. The local `nvcc` (`/home/user/cuda-tk/bin/nvcc`,
V13.0.88, CUDA 13.0, `sm_121a`) is compile-only; every device number below comes from the nodes and
nothing is inferred.

## 1. What landed on `integration/s1-layer-device`

Branch base: `main` @ `7c0afb78` (!170, the handbook consolidation). Tip: this MR's head. The probe
binary was built at the pre-rebase tip `fe281804`; the rebase changed no executable code
(`git diff fe281804..HEAD -- crates kernels` changes no logic, ABI or geometry - only comment and
message-string doc paths), so the probe describes the tip's code.

| commit | what |
| --- | --- |
| `af8905cf` | **P0/F1b**: the mHC-seam bind is derived from the expected backbone name set (`layers.{l}.{suffix}` exact lookups), never a suffix scan; `bound == expected` stays a hard assertion; the MTP lanes can no longer be bound (400 = 400); unit test proves the 30-tensor MTP seam set is excluded. |
| `9bd1a35b` | **P5**: `device_counters.rs` (d2h / allocation / sync-wait recorder with the declared host-bridge exception, 11 CPU tests), `scripts/qa-d54.sh` (32×128 fixture verifier + `numerical-gate.sh --run`, fails closed), `scripts/qa-layer-diff.sh` (`--layer N` first-stage >10× bound bisection, fails closed), `docs/archive/build/s1-qa-harness.md` (the record moved with the !170 archive split). |
| `06d97c8f` | **P5 wiring**: the device arm counts its own crossings (both boundary readbacks + the head logits gather), fails a head-position step with the invariant named, and carries `device_step_invariants` in every position record. |
| `df50d55d` | **P2 (ratio-0 half)**: `reference_seam` / `ref_select` / `stage_attention` wired for the released ratio-0 layers (0 and 1) — sliding-window metadata, empty candidate list, validation-only staging, zero launches and zero copies; ratio>0 refuses with the truthful blocker; open refuses a window that is not the pinned 128. |
| `1e30a7bc` | **P4**: collectives audit (F1–F8, line-precise), declared `s1-host-bridge` feature (off by default, unreachable through the NCCL API), async-failure API (`ncclCommGetAsyncError` optional symbol, typed `AsyncFailure`, CPU tests), clock-locked 10 KiB gate script + bench flags. |
| `4872d007` | **P4 CLI side**: async-failure query at the run boundary, real `stats.arena_bytes` + collective counters in the record, missing-reduce refusal before position 0, misplaced doc fixed. |
| `c1ea6a0b` | **P1/P2/P3 kernels**: `engram_write_sm121` (fused gate/write), `engram_stream_convert_sm121` (both f32↔bf16 directions, one cubin), `moe_routed_device_sm121` (staging gather from a device expert table + ordered combine), the three typed launchers and the routed-MoE run-scope module; `attention_operands.rs` (window/store/ring/rope/wo_a-wo_b operand geometry, 20 CPU tests); kernel baseline extended 27 → 30 with real local-nvcc pins. |
| `1cf4c74e` | **P2 attention half (builders)**: `layer_attention_stages.rs` — the five attention stages as device-free word/launch builders (19-word body, 11-word inverse rope, wo_a/wo_b output chain) with 27 tests pinning every array and refusal; no `DeviceLayer`, no allocation, no image load. |
| `3b44def0`, `937d0df1` | **S1 status record** (this file) at the pre-window state; superseded by the hardware results below. |
| `7a167ef4` | **S1-0 hardware probe**: the four-rank run, this record, and the archived evidence bundle. |
| this MR's tip | Record corrections from the adversarial review: the walk's transport, no derived per-step number from the invalid gate, exact contract-kernel names, post-rebase provenance. |

The wiring/evidence docs for all five packages are committed under `docs/archive/build/`:
`s1-p1-engram-wiring.md`, `s1-p2-attention-wiring.md`, `s1-p3-moe-wiring.md`,
`s1-p4-collectives.md`, `s1-qa-harness.md`.
What these commits deliberately do **not** include is the `LayerStages` body wiring in
`tp4_forward.rs` (the attention bodies, the Engram stage, the routed-MoE stage, the device head and
the S1-a mixed arm).

## 2. S1-0: the device arm on hardware (2026-10-23)

The probe is the frozen speed-gate argv (nine positions, `--forward-lean`) with
`SPARKFLASH_LAYER_DEVICE=1`. Binary: cross-built `aarch64` `sparkflash` sha256
`c5a657e084c3261f3b6a9fad3208ec20b86c89329fedd2c9cd52d36b756ff91e`
(`scripts/build-aarch64.sh --features=cuda`), 30 runtime cubins whose sha256 match
`crates/sparkflash-kernels/baseline/kernel_baseline.txt` **30/30**, NCCL `libnccl.so.2` sha256
`aeda7a87ab4c3707216d428c6f4dee0f178ed87a6ed76f5bd6be9639e737a51a` (2.31.2), checkpoint snapshot
`1bae0e10…`, peers `192.0.2.80,.88,.59,.68`, TCP base port 29611. The 138-token input is the 130-token
b69 fixture prompt concatenated with the first 8 reference ids (`reference_ids[0..8]`; the list is
committed as `token-ids-138.txt`, sha256 `af1d3635ca32a9aa1f43bf137a2776a55a240ca7f13651d0b07d8cf6f4faf950`); reference ids stay
`491,367,201,80,3302,201,54,201,4341`.

Evidence: `docs/archive/build/s1-device-probe-20261023/` (per-rank `run.log`, `run.exit`, arena-bind
JSON, layer-device frontier JSON, `probe-summary.json`, and the collective diagnostic records in §2.2).

All four ranks exited rc=2 in **303 s wall** (arena boot + bind + 80-reduce walk + the forward attempt;
this is not a step timing). Exit 2 is the documented stage refusal, not a crash and not a gate failure.

| check | rank 0 | rank 1 | rank 2 | rank 3 |
| --- | --- | --- | --- | --- |
| device arm opened | yes | yes | yes | yes |
| `mhc_seam` bound / expected | **400 / 400** | 400 / 400 | 400 / 400 | 400 / 400 |
| `mhc_seam` missing / unexpected | `[]` / `[]` | `[]` / `[]` | `[]` / `[]` | `[]` / `[]` |
| attention bound / expected | 208 / 208 | 208 / 208 | 208 / 208 | 208 / 208 |
| attention host seams excluded | 19 | 19 | 19 | 19 |
| arena bind complete (`device_bound_names`) | yes (93,131) | yes (93,131) | yes (93,131) | yes (93,131) |
| MoE classes bound (gate-up / down / shared / router) | 61,440 / 30,720 / 240 / 120 | same | same | same |
| census resident / unclassified | 95,818 / `[]` | 95,818 / `[]` | 95,818 / `[]` | 95,818 / `[]` |
| per-token weight H2D bytes (structural constant, not a probe counter) | 0 | 0 | 0 | 0 |
| reference collective walk (TCP), calls/step, steps | yes, 80/step, 11 (2 warm + 9 steady), `calls_per_step_ok` every step | same | same | same |
| forward frontier | layer 0, pos 0 | layer 0, pos 0 | layer 0, pos 0 | layer 0, pos 0 |
| stages run before refusal | 9 | 9 | 9 | 9 |
| steps completed / tokens produced | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

**S1-0 verdict: PASS.** The arm opens on 4/4 ranks, the bind is `400 = 400` (the P0/F1b fix holds on
hardware — before it, the arm refused at open with `400 expected, 430 bound` and the 30 MTP lane names
listed as unexpected), every arena class is complete, the census has no unclassified tensor, no weight
is copied per token, and the four-rank *reference* walk (TCP) executes its 80 reduces per step with
`calls_per_step_ok` true on every recorded step. That walk is the canonical exchange skeleton, not
the device arm's NCCL enqueue: the forward stops at stage 10, before `all_reduce` (stage 12) is
reached, so no NCCL collective ran inside a step. The probe also
carries the arm geometry at open: `max_positions` 138, `rope_table_bytes` 70,656, `slab_bytes` 996,864.

The nine stages that ran on layer 0 position 0 are the wired prefix:
`engram_write` (no-op: layer 0 has no Engram rows) → `mhc_entry` → `quantise_prologue` →
`attention_projections` → `query_path` → `kv_path` → `reference_seam` → `ref_select` →
`stage_attention`. The refusal names stage 10:

> `layer device stage attention_body: kernel csa2_sequence_attn is not landed on this branch; …
> the per-head sink and the indexer spans (index_key, index_query, block_scores, block_best) have no
> device spans on this arm`

No value in this row is validated against the host reference (`tolerance_arm: null`), which is why it
is an S1-0 probe and not a numerical row.

### 2.1 What the probe does not show

- **No step, no timing.** The forward stops at layer 0 position 0, so `decode_steps` never completed;
  no ms/step number exists and none is claimed.
- **No D54 row.** No logits were produced on the device arm; `scripts/qa-d54.sh` still has no frozen
  32×128 fixture (`docs/build/reference/speed-gate/fab52c00a0dc6f6c/` is the nine-position row, the
  wrong fixture, and the script correctly exits 2 on it).
- **No counter row.** `d2h = 1` was not observed because no head-position step completed. The wiring's
  present verdict on a hypothetical completed step is `d2h 3` (two boundary readbacks + head logits),
  which would fail; the counters fail the step by name, as designed.
- **Layers 1 and 2–39 were not reached**, so their frontier is still static reading (§3).

### 2.2 The collectives gate: a labelled-invalid diagnostic, not a gate row

The P4 gate (`scripts/gpu-reduce-bench.sh gate`, 10,240 B device-only all-reduce, 10,000 iterations,
clock-locked, p50 ≤ 30 µs and p99 ≤ 60 µs) **could not run validly**: `nvidia-smi -lgc` needs root or
passwordless sudo, neither is available to `allenh1` on the nodes, so no rank locked a clock. The run was
made with `SPARK_REDUCE_ALLOW_UNLOCKED=1` and is **INVALID by the gate's own rule**: its verdict is
recorded (`valid: false`, `clock.locked: false`) and its latency series is deliberately **not
reproduced here**, because an unlocked run is neither a gate result nor a number that may be compared
with the 30/60 µs ceilings or scaled into a per-step estimate. What the run does establish:

- the four ranks' reductions are bit-exact: `max_ulp 0`, 0 differing elements, on all four ranks;
- the clocks were idle/unlocked at 208 of 3003 MHz on all four ranks (`clock.locked: false`);
- the NCCL library in use was 2.31.2 on all four ranks.

Artifacts: `collective-diagnostic-*.json` in the evidence directory (raw bench output, per rank). A
valid gate row needs the clock-lock privilege above.

## 3. The frontier

Layer 0 was **measured** (§2). Layers 1 and 2–39 were **not reached**; their refusal is static reading
of the bodies at this tip and it is labelled as such.

- **Layer 0** (ratio 0, no Engram rows): measured — refuses at **`attention_body`** (stage 10). The
  pinned `flash_mla_csa2_seq_sm121` launch exists and the query row is device-resident, but the per-head
  sink and indexer spans are not allocated on this arm.
- **Layer 1** (ratio 0, **has Engram rows**): not reached; by reading, refuses at `engram_write`
  (stage 1) with rows: the kernel and launcher are landed, the f32↔bf16 conversion kernels are in the
  tree, but the at-open Engram row arena and the 6-launch body wiring are not yet in `tp4_forward`.
- **Layers 2–39** (ratio 2 then 1): not reached; by reading, refuse at `reference_seam` **by design**.
  The compressor (`attn.compressor.wkv/wgate`) and indexer (`indexer.wk/weights_proj`) tensors are
  `AttentionHostSeam` in the plan and are structurally excluded from the arena bind
  (`resident_weights.rs`); the reference seam on those layers needs that class moved into the bind and
  the compressor/index/kernels ported, which is implementation, not wiring.

The S1-a gate (layer 0 device-only while layers 1–39 stay host) also needs a **mixed-arm loop** that does
not exist: the arm is chosen once for the whole run (`select_layer_arm`), and no per-layer device arm
path is wired.

## 4. Per-stage table

The device-run column now says what actually happened on hardware for the layer-0 prefix; no cell in the
GPU-time column can be filled, because no step completed and the arm has no per-stage observation read.
"twin" names the CPU proof that exists; "pin" means the cubin is in `baseline/kernel_baseline.txt`.

| # | stage | contract kernel | body at this tip | device run (2026-10-23) |
| --- | --- | --- | --- | --- |
| 1 | `engram_write` | `engram_write_bf16` | kernel + both conversion kernels landed and pinned; no-op without rows; stage body still refuses with rows (at-open row arena + 6-launch chain not wired) | **RAN** (layer 0, no-op) |
| 2 | `mhc_entry` | `mhc_entry_bf16` | wired (launch) | **RAN** |
| 3 | `quantise_prologue` | `w8a8_prologue_e4m3` | wired | **RAN** |
| 4 | `attention_projections` | `b12x_mxfp8_gemv_m1_sm121` | wired | **RAN** |
| 5 | `query_path` | `rmsnorm_sm121` + GEMV sequence | wired | **RAN** |
| 6 | `kv_path` | `kv_norm_rope_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16` | wired | **RAN** |
| 7 | `reference_seam` | `compressor_select` | **ratio 0 wired (no launch)**; ratio>0 refuses (host-seam class) | **RAN** (ratio 0) |
| 8 | `ref_select` | `ref_select_sm121` | **ratio 0 identity no-op**; ratio>0 not wired (operands above) | **RAN** (no-op) |
| 9 | `stage_attention` | `attention_stage` | **wired (validation)**, ratio 0 | **RAN** |
| 10 | `attention_body` | `csa2_sequence_attn` (`flash_mla_csa2_seq_sm121` cubin) | **refuses**: sink/index spans not allocated on the arm; the 19-word builder + inverse-rope builder are landed and tested | **REFUSED** (frontier) |
| 11 | `attention_output` | `wo_a_wo_b_gemv` + inverse rope | **refuses**: wiring pending (kernels pinned: wo_b = `b12x_mxfp8_gemv_m1_sm121`, rope = `rope_rotate_rows_bf16`); the 8/8/11-word output builders are landed | not reached |
| 12 | `all_reduce` | `comm_all_reduce_bf16` | wired (enqueue-only); not reached in a step. The run's 80/step walk is the TCP reference skeleton (§2), not this stage | not reached in a step |
| 13 | `attention_residual` | `mhc_attn_residual` | wired | not reached |
| 14 | `ffn_seams` | `mhc_ffn_seams` | wired | not reached |
| 15 | `moe_router` | `moe_router_sm121` + top-k | wired | not reached |
| 16 | `moe_routed` | `b12x_mxfp4_moe_stored` + combine | **refuses**: the 8-launch device path (staging table + combine) is landed and pinned; stage body not wired | not reached |
| 17 | `all_reduce_moe` | `comm_all_reduce_bf16` | wired | not reached in a step |
| 18 | `moe_shared` | coded/f32 GEMV + SwiGLU | wired | not reached |
| 19 | `all_reduce_rdma_arm` | `device_all_reduce_bf16` | typed refusal unless `gpu-rdma-reduce` is built; never called by the NCCL arm | — |
| 20 | `ffn_epilogue` | `mhc_ffn_epilogue` | wired | not reached |

Per-stage **GPU time table: NOT RUN** — no step completed, so there is no timing ring to read. The
steady-state counters are wired but their verdict on a completed device step would still be FAIL: two
boundary readbacks (`residual_out`, `attn_pre_mix_out`) plus the head logits gather = `d2h 3`, expected 1.

## 5. Scope findings that change the plan

1. **The ratio-0 class removes the first frontier.** Released ratios are `[0,0, 2×18, 1×20, 0×3 pad]`
   (`cache_plan.rs:4`); `attention_seam_step` runs no compressor, no index keys and no compressed rows
   when the ratio is 0 (`attention.rs:2116-2166`). Layers 0 and 1 therefore need no compressor port for
   the seam; the probe confirms layer 0's seam stages complete on device.
2. **`reference_seam` on ratio>0 is a port, not wiring.** 19 tensors are `AttentionHostSeam` and never
   reach the arena; the seam needs compressor wkv/wgate GEMV, pool/narrow, compressed append, index
   keys, per-query weights and the candidate select on device. This is the block that S1-b waits on.
3. **`d2h = 1` needs a device head.** The host head consumes `stream` (hc_pre + final norm on the CPU),
   so the boundary readbacks are real copies. Nothing in the five work packages moved the head; the
   counters now *fail the step by name* until the head consumes the device residual.
4. **The Engram write needs exact stream-dtype conversions.** The landed fused gate/write kernel is
   bf16-stream/bf16-out and the arm's stream span is f32; two conversion kernels (one new cubin) were
   landed rather than forked into the pinned cubins.
5. **The 32×128 D54 fixture is not frozen.** The only in-tree frozen reference is the 9-position
   speed-gate row; `scripts/qa-d54.sh` rejects it as the wrong fixture and exits 2 with the freeze
   procedure. Freezing needs a host-arm run on a node, which the window made possible but S1-0 did not
   need.
6. **S1-a's mixed arm does not exist.** The arm is a single run-level decision; a per-layer device/host
   split with the one allowed residual copy each way is loop work that is not started.
7. **The collective producers are not live yet.** `attention_output` and `moe_routed` refuse, so no
   per-layer collective can enqueue inside a step; the standalone 10 KiB walk now has hardware evidence
   (§2.2) but it is unlocked and therefore not a gate row. The clock-lock prerequisite (root or
   passwordless sudo for `nvidia-smi -lgc`) is missing on the nodes and must be fixed before S1-c.

## 6. What is genuinely verified

Hardware (2026-10-23, four DGX Spark nodes, evidence in `docs/archive/build/s1-device-probe-20261023/`):

- S1-0: the arm opens on 4/4 ranks, `400 = 400` on `mhc_seam` with no MTP names, 208/208 attention,
  arena bind complete, census 95,818 resident with no unclassified tensor, per-token weight-H2D 0 by
  construction, and the 40-layer TCP reference walk at 80 reduces/step (11 steps, every step
  `calls_per_step_ok`). Nine device stages run on layer 0 before the named refusal.
- The cross-built aarch64 binary runs, and the cubins the launching stages need (drawn from the
  30-kernel pinned baseline) load and launch: the arm reaches real kernel launches (`mhc_entry`
  through `kv_path`; stages 1, 7, 8 and 9 are no-op/no-launch in this configuration), so the cubin
  directory, typed ABIs and open-time width asserts are exercised on hardware, not just compiled. The
  cubins of the unwired stages were not loaded.
- The standalone four-rank NCCL 10 KiB bench reduced bit-exactly (`max_ulp 0`, 0 differing elements),
  but that run is INVALID (§2.2): it is not a gate pass and must not be quoted as one.

CPU / toolchain / scripts (unchanged from the pre-window record):

- `cargo check -p sparkflash-cli --lib` and `--features cuda` clean; `cargo test -p sparkflash-cli
  --lib --features cuda` **487 passed, 1 ignored**; `cargo test -p sparkflash-kernels --lib`
  **584 passed, 2 ignored**; clippy `-D warnings` clean on both crates in both feature configs;
  `cargo fmt --all --check` clean.
- `cargo test -p sparkflash-comm --lib` 139 passed (145 with `--features s1-host-bridge`); clippy
  `-D warnings` clean; aarch64 cross-build succeeds (this is the binary that ran on the nodes).
- Kernel baseline is **30** kernels; the local `nvcc` reproduces the manifest byte for byte
  (`cargo test -p sparkflash-kernels --test kernel_baseline` with `SPARKFLASH_NVCC`, 5/5 tests), and
  the 30 cubins used by the probe matched those pins 30/30 before staging.
- QA scripts: `qa-d54.sh --self-test` PASS, `qa-layer-diff.sh --self-test` PASS,
  `numerical-gate.sh --self-test` PASS; both new scripts fail closed (exit 2) without a frozen
  fixture / GPU / bounds file.
- P0's fix is covered by a CLI unit test (400 expected = 400 bound, 30 MTP seam tensors excluded) and
  now by the hardware bind record.

## 7. What closing S1 requires, in order

1. ~~GPU node window and S1-0~~ **done 2026-10-23**: arm opens on 4 ranks, `400 = 400`, refusal names
   the first unwired body (`attention_body`).
2. Wire `attention_body` + `attention_output` operands/launch (word builders are landed) and
   `moe_routed`; then layers 0–1 run to the FFN half.
3. Engram at-open row arena + conversion + 6-launch body; then layer 1 stops refusing.
4. Device head (hc_pre + final norm + tail consuming the device residual) so a completed step reports
   `d2h = 1`, `alloc = 0`, `sync ≤ 1` — required by the counters and by S1-b's definition.
5. S1-a mixed-arm loop for the layer-0 device-only gate; freeze the 32×128 D54 fixture on a node.
6. Ratio>0 compressor/indexer port (host-seam class into the bind + the five device kernels) for
   layers 2–39, then S1-b.
7. S1-c: two clock-locked replicates ±2 ms, `gpu_mhz`/`power_w` recorded, `d54 PASS`, counters PASS,
   then the `speed-log.csv` row `s1-c,<sha>,≤120.00,d54 PASS,d2h=1,alloc=0,clock-locked` and the
   per-stage GPU-time table in this file. Clock lock needs root or passwordless `nvidia-smi -lgc` on
   the four nodes; without it, no S1-c row can be valid. Only then S1-flip.

Until then the published number is **179.81 ms/step**, `speed-log.csv` is untouched, and no
partial-port timing row may be added.

**Owned follow-up (RULES §5 inventory):** the landed-but-unwired kernel paths (`engram_write_launch`,
`engram_convert_launch`, `moe_routed_device`, `layer_attention_stages`) are the next block's inputs;
each is either wired by the next S1 step list above or deleted, and none is measured or counted as a
product path today.
