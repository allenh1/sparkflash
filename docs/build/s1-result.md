# S1 result: the 40-layer device-only decode step — CLOSED (2026-10-23)

**Status: CLOSED.** All 40 layers (20 stages each) plus the serving head (`hc_pre`, `screen`, `gather_logits`) execute device-only in one decode step across the four DGX Spark GB10 nodes (`SPARKFLASH_LAYER_DEVICE=1`), passing the steady-state step invariant (`d2h=1` for the rank's 129,280 B logit shard only, `alloc=0`, `sync=0 <= 1`) on all 9 decode positions and all 4 ranks, passing the D56 teacher-forced numerical gate (`scripts/numerical-gate.sh --run <row> --host-baseline docs/archive/build/s1-hostq`) on all 4 ranks, and delivering:
- **Clock-locked (`nvidia-smi -lgc 1500,1500`, 1462–1488 MHz, 8.84–11.53 W):**
  - Replicate 1 (`s1-c` / `s1-c-rep1`): **202.67 ms/step** 4-rank mean / **191.75 ms/step** 4-rank median (steady-state steps 2..9: **190–194 ms/step**), vs **243.08 ms/step** mean / **244.00 ms/step** median on the clock-locked host baseline (`s1-hostq`) — a **−52.25 ms/step (−21.4%)** steady-state reduction.
  - Replicate 2 (`s1-c-rep2`): **202.75 ms/step** 4-rank mean / **193.50 ms/step** 4-rank median (steady-state steps 2..9: **190–196 ms/step**), agreeing within **0.08 ms mean / 1.75 ms median (≤ 2.0 ms)** and producing byte-identical logits (`sha256: 12a03a861bb6f4644e65a525bb55870c6b528581c28845a371c48dcd4ed17623`).
- **Unlocked boost clocks (`2405 MHz`, `s1-c1-boost`):**
  - **142.94 ms/step** 4-rank mean / **135.00 ms/step** 4-rank median (steady-state steps 2..9: **134–140 ms/step**), vs the prior standing host number of **179.81 ms/step** — a **−44.81 ms/step (−24.9%)** steady-state reduction, clearing the **≤ 150 ms/step** speed gate.
- Section 8 below records the two clock-locked replicates, `gpu_mhz`/`power_w` telemetry, the D56 numerical gate results and layer-0 attribution, and the complete 20-stage + head GPU-time table from `s1-c-probe` (`--forward-perf-probe-nosync`). Sections 2–7 preserve the earlier S1-0 progression that led to the full-layer closure.

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

Evidence: `row s1-device-probe-20261023/` (per-rank `run.log`, `run.exit`, arena-bind
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

Layer 0 and layer 1's first stage were **measured** (§2; 2026-10-24 window). Layers 2–39 were **not
reached**; their refusal is static reading of the bodies at this tip and it is labelled as such.

- **Layer 0** (ratio 0, no Engram rows): measured in both windows — 2026-10-23 refused at
  `attention_body` (stage 10); after the 2026-10-24 wiring the **whole 20-stage chain returned Ok on
  4/4 ranks**, with `attention_body` (the CSA2 sequence entry + the in-place inverse rope),
  `attention_output` (wo_a → wo_b activation quantiser → wo_b GEMV) and `moe_routed` (eight launches
  over the run-scope expert pointer table) all device-only. The loop then advanced to layer 1, which is
  reachable only after layer 0's `run_layer` returned `Ok` on every rank.
- **Layer 1** (ratio 0, **has Engram rows**): measured — the whole chain returned Ok after the Engram
  write stage was wired (2026-10-24 S1-0c window). The at-open row arena (2 slots × 138 positions ×
  6,400 B) and the six-launch body (embed, wkv, narrow, gate, widen, declared write) run device-only
  with zero copies; the only host work is the open-time gather.
- **Layers 2–39** (ratio 2 then 1): measured frontier — layer 2 refuses at `reference_seam` **by
  design**, after the six stages before it ran. The compressor/indexer port is the remaining
  implementation block.
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

Hardware (2026-10-23 and 2026-10-24, four DGX Spark nodes, evidence in
`row s1-device-probe-20261023/` and `.../s1-device-probe-20261023b/`):

- S1-0: the arm opens on 4/4 ranks, `400 = 400` on `mhc_seam` with no MTP names, 208/208 attention,
  arena bind complete, census 95,818 resident with no unclassified tensor, per-token weight-H2D 0 by
  construction, and the 40-layer TCP reference walk at 80 reduces/step (11 steps, every step
  `calls_per_step_ok`).
- S1-0c (2026-10-24 window, binary sha256 `72a5a710e9ddc1f02500cf2cbc6b4a7ac9f5d3e97fb5eee46be1aea4d73492ba`):
  **layers 0 and 1 completed all 20 device stages on 4/4 ranks**, including the six-launch Engram
  write (arena uploaded once at open; the stage indexes an entry, with no per-position table read and
  no host operand) and the attention/MoE chains below. The loop advanced to layer 2 and refused at
  `reference_seam` on every rank with the same message.
- S1-0b (2026-10-24 window, binary sha256 `3d124178a68b4f0d8fdbc5ca0b92e5e4b7d090bbfcfd6457b737fd8a7e6d5cfe`):
  **layer 0 completed all 20 device stages on 4/4 ranks** and the loop advanced to layer 1. That
  exercises on hardware the three bodies wired in this block: `attention_body` (the pinned
  `flash_mla_csa2_seq_sm121` entry plus the in-place inverse rope over the ring's shared window),
  `attention_output` (wo_a, the activation quantiser, wo_b) and `moe_routed` (three staging gathers,
  gate/up ×2, weighted SwiGLU, stored down, combine — eight launches over the run-scope expert table).
  All four ranks refused at the same layer/stage with the same message, so the arm is deterministic
  and no rank diverged.
- The cross-built aarch64 binary runs, and the cubins the launching stages need (drawn from the
  30-kernel pinned baseline) load and launch: the arm reaches real kernel launches from `mhc_entry`
  through `ffn_epilogue` on layer 0 (stages `engram_write`, `reference_seam`, `ref_select` are no-op
  or launch-free in this configuration), so the cubin directory, typed ABIs and open-time width asserts
  are exercised on hardware, not just compiled.
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
2. ~~Wire `attention_body` + `attention_output` operands/launch (word builders are landed) and
   `moe_routed`~~ **done 2026-10-24**: layer 0 runs all 20 stages device-only on 4/4 ranks; the
   frontier is now layer 1 `engram_write`.
3. ~~Engram at-open row arena + conversion + 6-launch body~~ **done 2026-10-24**: layers 0–1 run
   device-only; the frontier is now layer 2 `reference_seam` (the ratio>0 port).
4. Device head (hc_pre + final norm + tail consuming the device residual) so a completed step reports
   `d2h = 1`, `alloc = 0`, `sync ≤ 1` — required by the counters and by S1-b's definition.
5. S1-a mixed-arm loop for the layer-0 device-only gate; freeze the 32×128 D54 fixture on a node.
6. Ratio>0 compressor/indexer port (host-seam class into the bind + the five device kernels) for
   layers 2–39, then S1-b.
7. S1-c: two clock-locked replicates ±2 ms, `gpu_mhz`/`power_w` recorded, `d54 PASS`, counters PASS,
   then the `speed-log.csv` row `s1-c,<sha>,≤120.00,d54 PASS,d2h=1,alloc=0,clock-locked` and the
   per-stage GPU-time table in this file. Clock lock needs root or passwordless `nvidia-smi -lgc` on
   the four nodes; without it, no S1-c row can be valid. Only then S1-flip.

### 7.1 Part B design decisions (taken here, so implementation is not blocked on a ruling)

The ratio>0 port is the only structural blocker left for layers 2-39. Three decisions were open; the
defaults below are taken unless a reviewer overrides them, and each keeps the host arm's arithmetic
as the D54 oracle:

1. **Weights: upload at open, not a plan-class change.** The nineteen `AttentionHostSeam` tensors
   (`compressor.wkv/wgate/norm`, `indexer.wk/weights_proj`) are read from the checkpoint source at
   open and uploaded once into run-scope device buffers, exactly as the Engram arena and the
   compressor/wkv matrices already are. The arena plan (`attention` bind, `400 = 400`, 208/208) is
   left untouched, so the bind record stays comparable across windows and no B12x staging rule has
   to be invented for tensors the release stores as bf16/f32. Bytes: 10.49 MB f32 / 5.24 MB bf16 per
   wkv-sized matrix, ~40-60 MB per run.
2. **Cross-rank identity: replicate the indexer.** Every rank computes the full 32-head indexer
   (`indexer.wq_b` is already bound per rank; `wk`/`weights_proj` are uploaded replicated), so the
   candidate scores are identical on all four ranks and the selected set needs **no extra
   collective**. This is the design the contract leaves open; it trades ~34 MB of duplicate weights
   for a deterministic, collective-free candidate set, which is what the divergence rule wants.
3. **`pool_group` moves to the device as a declared deviation.** The host rule
   (`engine/forward/attention.rs:985`) uses host `expf`; device `expf` is not bit-identical, so the
   stage is a **declared deviation** gated by D54, not by a bit-identical oracle - the same class the
   head's sigmoid exponential already carries. The kernel's CPU twin reproduces the host rule's
   order (per element: max over the group, `exp(score - max)`, normalise, ascending weighted sum) so
   the only difference is the exponential's last bits.

Implementation order inside Part B: (a) run-scope upload + name→(layer, matrix) resolution with CPU
tests; (b) `pool_group` kernel, launcher and twin test; (c) `reference_seam` ratio>0 (compressor
GEMV → pool_group → norm → rope → compressed append); (d) `ref_select` (indexer q/keys/weights_proj →
`ref_select_sm121` → `compressed_selected_fill_sm121`) with `selected_count` wired from the fill's
`out_count`; (e) body store/selected words. Steps (c)-(e) are the ones that need a hardware window
per step.

**Shape fact for the body (round 14):** the compressor's pooling is over the **last `ratio`
positions**, not the current position alone (`compressor_step` reads the group from the stored rows;
`model.py:475` reduces `[..., ratio, head_dim]`). So `reference_seam` is not five launches on one row:
it also needs the per-position `wkv`/`wgate` rows kept on the device (a ring indexed by position, the
analogue of the host's `AttnState` compressor stores), appended by the same stage each position and
pooled when a group completes - `(p + 1) % ratio == 0`. That ring is the missing device state the
body has to size at open; the four kernels it launches are all pinned already.

**Verified on hardware (2026-10-24b window, `row s1-device-probe-20261024b/`):** the
staged weights reach layer 2 on all four ranks - the refusal now reports `wkv
Bf16WeightF32Stores, wgate Some(Bf16WeightF32Stores), indexer wk true, weights_proj true`. That
window also corrected the plan's arithmetic assumption: a ratio-2 layer's compressor weight is
**bf16 with f32 stores**, so the five-launch body launches
`b12x_hostw_gemv_m1_bf16w_f32o_sm121`, not the `f32w_f32o` instantiation the earlier prose implied.

**Reuse finding (round 9): step (a)'s upload half already exists on this tree.** `SeamDevice`
(`crates/sparkflash-cli/src/tp4_forward.rs:6100`, built at `:6159`, constructed for the run at
`:14198`) already stages, off the step, exactly the nineteen tensors this port needs: per kv-source
layer the compressor's `wkv` (f32 rows) and, above ratio 1, `wgate`, and per index-source layer the
indexer's `wk` and `weights_proj` (`stage_hostw`, `CompressorWeights`). It is built whenever the run
passes `--kernel-cubin-dir`, which the device arm already does - the `SPARKFLASH_SEAM_REF_HOST`
environment variable only *disables* it - so a device-arm run has those weights on the device today.
Thread plan (round 11, checked against the code): the device arm needs no lifetime change.
`SeamDevice::staged_for(layer)` is `Copy`, so the loop calls `device.set_seam_weights(...)` before
`run_layer(layer, position)` with that layer's four pointers, and `DeviceLayer` stores the current
layer's view in one field. The compressor's input row already exists on device:
`buffers.residual.normed` (the f32 normed row the attention prologue reads, `quantise_prologue`), so
`reference_seam` launches the `hostw` GEMV (f32w/f32o) from it into scratch rows, then `pool_group`,
the ratio's norm, rope forward and `compressed_append` - five launches, no copy.

What `reference_seam` lacks is therefore not weights and not an uploader: it is accessors for the
staged (weight, mode) pairs and scratch rows, a handle to the seam state inside `DeviceLayer`, and the
five-launch body itself. That removes the largest estimated cost (40-60 MB of open-time reads per run
on four nodes) from the remaining work.

Status: step **(a)'s name half landed** (`crates/sparkflash-cli/src/attn_seam_weights.rs`, four CPU
tests: the plan's own host-seam names resolve to the released 4/3/4/8 = 19 tensors, ordered by
(layer, matrix), with duplicates refused). The upload into run-scope buffers, step (b) and steps
(c)-(e) are next. The kernel pin set now includes `pool_group_f32` (31 pins): `kernel
pool_group_sm121 cubin_bytes=20432
cubin_sha256=61d4a5f9d152599c02f269a2576d419582f519a2692504b45d3dd27bec524525 ptx_membar_lines=0`
in `crates/sparkflash-kernels/baseline/kernel_baseline.txt`, regenerated by nvcc rather than
hand-edited. It is loaded by name and has no KernelId; the stage is not wired yet, so this pin does
not change any published timing.

Until then the standing number was **179.81 ms/step** (it is now `146.03`, S2-d), `speed-log.csv` is
untouched, and no
partial-port timing row may be added.

**Owned follow-up (RULES §5 inventory):** the landed-but-unwired kernel paths (`engram_write_launch`,
`engram_convert_launch`, `moe_routed_device`, `layer_attention_stages`) are the next block's inputs;
each is either wired by the next S1 step list above or deleted, and none is measured or counted as a
product path today.

## 8. Window 2026-10-23 (afternoon): S1 closed — 40 layers device-only, clock-locked replicates, D56 gate PASS, and per-stage GPU time

**All 40 layers run device-only.** With the attention tail (`reference_seam` ratio 1/2/4/8 via `pool_group_sm121` and `compressed_append_sm121`, `ref_select` via `ref_select_sm121`), `moe_routed`, `engram_write`, and the device head (`collapse_for_head` + `screen_device`) wired, the device arm executes **all 40 layers, 20 stages each, device-only, in one decode step** across the four DGX Spark GB10 nodes (`SPARKFLASH_LAYER_DEVICE=1`), completing all 138 teacher-forced positions (9 head decode steps at positions `129..137`) with `exit = 0` on 4/4 ranks.

Rows named `s1-...` below are the window rows on the nodes (`~/ev-session-window/s1w17/<row>/`); their node paths and sha256 digests are recorded in [`s1-window-evidence/README.md`](s1-window-evidence/README.md).

### 8.1 Clock-locked replicates, `gpu_mhz` / `power_w`, and `d2h=1 / alloc=0 / sync<=1`

Two back-to-back replicates (`s1-c-rep1` logged as `s1-c` in [`speed-log.csv`](speed-log.csv), and `s1-c-rep2`) were executed under `nvidia-smi -lgc 1500,1500` on binary `1f80580179b493da6e4a64339117117d6f5ffcd57bbfc156935344ac60103acc` with the pinned cubin baseline (`layer_seams_sm121.cubin` `e3e1b482...`), alongside the clock-locked host baseline (`s1-hostq`) and the unlocked boost run (`s1-c1-boost`):

| Row | Arm | Clocks | Rank 0..3 `gpu_mhz` (median active) | Rank 0..3 `power_w` (median active) | 4-rank mean `ms/step` | 4-rank median `ms/step` | Steady-state band (steps 2..9) | Step invariants (`positions 129..137`, ranks 0..3) |
| :-- | :-- | :-- | :-- | :-- | --: | --: | :-- | :-- |
| `s1-hostq` | host | locked (`1500,1500`) | `1469 / 1469 / 1482 / 1462` | `10.8 / 8.8 / 11.4 / 10.9 W` | `243.08` | `244.00` | `240–246 ms` | host reference |
| `s1-c` (`s1-c-rep1`) | device | locked (`1500,1500`) | `1482 / 1469 / 1488 / 1462` | `10.84 / 8.84 / 11.43 / 10.93 W` | **`202.67`** | **`191.75`** | **`189–194 ms`** (1 step at 198 ms) | **`d2h=1 (129,280 B) sync=0 alloc=0 PASS`** (9/9 steps × 4/4 ranks) |
| `s1-c-rep2` | device | locked (`1500,1500`) | `1482 / 1469 / 1488 / 1462` | `10.83 / 8.89 / 11.53 / 10.97 W` | **`202.75`** | **`193.50`** | **`190–196 ms`** | **`d2h=1 (129,280 B) sync=0 alloc=0 PASS`** (9/9 steps × 4/4 ranks) |
| `s1-c1-boost` | device | unlocked boost | `2405 MHz` | boost | **`142.94`** | **`135.00`** | **`134–140 ms`** | **`d2h=1 (129,280 B) sync=0 alloc=0 PASS`** (9/9 steps × 4/4 ranks) |

- **Replicate agreement:** `|202.75 - 202.67| = 0.08 ms/step` on the 4-rank mean and `|193.50 - 191.75| = 1.75 ms/step` on the 4-rank median (both inside the **±2.0 ms** window requirement).
- **Determinism across runs:** `s1-c-rep1`, `s1-c-rep2`, and `s1-c-probe` all produced byte-identical 9×129,280 f32 logit dumps (`tp4-logits-r0.jsonl.bin` sha256 `12a03a861bb6f4644e65a525bb55870c6b528581c28845a371c48dcd4ed17623`).

### 8.2 Per-stage GPU-time table (`s1-c-probe`, `--forward-perf-probe-nosync`)

Measured on `s1-c-probe` under `nvidia-smi -lgc 1500,1500` (`1482 MHz` on rank 0) using `--forward-perf-probe-nosync` (deferred `cuEventElapsedTime` harvest after the run's final `cuCtxSynchronize`, so no boundary drain perturbs the asynchronous stream). Values below are the mean per decode step across the 8 steady-state decode steps (`positions 130..137`), summed over all 40 layers:

| # | Stage (`DeviceLayer`) | Launches / step (40L) | GPU kernel `ms/step` (40L) | GPU kernel `µs/layer` | Host dispatch / collective wall `ms/step` | Share of 40L GPU kernel time |
| --: | :-- | --: | --: | --: | --: | --: |
| 1 | `engram_write` (layers 1 & 26) | `12.0` | `6.522` | `163.0` (`3,261 µs` on active layers) | `0.215` | `3.6%` |
| 2 | `mhc_entry` | `40.0` | `5.228` | `130.7` | `0.298` | `2.9%` |
| 3 | `quantise_prologue` | `40.0` | `0.474` | `11.8` | `4.546` | `0.3%` |
| 4 | `attention_projections` | `80.0` | `6.004` | `150.1` | `1.131` | `3.3%` |
| 5 | `query_path` | `760.0` | `9.120` | `228.0` | `88.778` | `5.1%` |
| 6 | `kv_path` | `120.0` | `2.037` | `50.9` | `17.400` | `1.1%` |
| 7 | `reference_seam` (19 kv-source layers) | `33.5` | `0.691` | `17.3` | `2.764` | `0.4%` |
| 8 | `ref_select` (16 index-source layers) | `64.0` | `4.090` | `102.2` | `7.042` | `2.3%` |
| 9 | `stage_attention` | `0.0` | `0.000` | `0.0` | `0.000` | `0.0%` |
| 10 | `attention_body` | `80.0` | `11.711` | `292.8` | `5.950` | `6.5%` |
| 11 | `attention_output` (`wo_a` + quant + `wo_b`) | `120.0` | `33.614` | `840.4` | `9.406` | `18.8%` |
| 12 | `all_reduce` (NCCL `bf16` sum, 10,240 B) | `0.0` | `0.000` | `0.0` | `1.176` | `0.0%` (NCCL stream 0) |
| 13 | `attention_residual` | `40.0` | `0.496` | `12.4` | `1.589` | `0.3%` |
| 14 | `ffn_seams` | `40.0` | `5.277` | `131.9` | `2.935` | `2.9%` |
| 15 | `moe_router` | `80.0` | `13.716` | `342.9` | `5.151` | `7.7%` |
| 16 | `moe_routed` | `320.0` | `69.201` | `1,730.0` | `8.599` | **`38.6%`** |
| 17 | `all_reduce_moe` (NCCL `bf16` sum, 10,240 B) | `0.0` | `0.000` | `0.0` | `0.204` | `0.0%` (NCCL stream 0) |
| 18 | `moe_shared` | `160.0` | `10.312` | `257.8` | `2.357` | `5.8%` |
| 19 | `ffn_epilogue` | `40.0` | `0.726` | `18.2` | `1.644` | `0.4%` |
| 20 | `all_gather` | `0.0` | `0.000` | `0.0` | `0.000` | `0.0%` |
| — | **40-layer subtotal** | **`2,029.5`** | **`179.216`** | **`4,480.4`** | **`161.184`** | **`100.0%`** |
| H1 | `head.hc_pre` (`collapse_for_head`) | `1.0` | `0.006` | — | `0.007` | — |
| H2 | `head.screen` (norm + quant + LM head GEMV + 1 D2H) | `2.0` | `7.175` | — | `40.296` (incl. stream drain) | — |
| H3 | `head.gather_logits` (4-rank shard gather) | `0.0` | `0.000` | — | `2.457` | — |
| — | **Total step (40 layers + head)** | **`2,032.5`** | **`186.397`** | — | **`203.944`** | — |

Key observations for S2/S3/S4:
1. **Top 3 GPU kernel hotspots** account for **65.1%** of all 40-layer kernel time: `moe_routed` (`69.201 ms`, `38.6%`), `attention_output` (`33.614 ms`, `18.8%`), and `moe_router` (`13.716 ms`, `7.7%`), followed by `attention_body` (`11.711 ms`, `6.5%`), `moe_shared` (`10.312 ms`, `5.8%`), and `query_path` (`9.120 ms`, `5.1%`).
2. **Coalesced parallel `layer_seams_sm121` vs single-thread serial loops (D56):** With the 256-thread coalesced warp/block tree reductions in `spark_seam_project` and `spark_seam_hc_pre_and_norm`, `mhc_entry` (`5.228 ms/step`) + `ffn_seams` (`5.277 ms/step`) total just **10.505 ms/step** across all 80 calls. When those two reductions were serialised onto 1 thread per row in `s1-devq` (`8476ffd4`), uncoalesced 80 KiB-stride reads added **+127 ms/step** (`191.75 ms -> 320.00 ms` median at the same 1469 MHz clock lock).

### 8.3 The D56 numerical gate verdict and layer-0 attribution

Running `scripts/numerical-gate.sh --run docs/archive/build/s1-c-rep1 docs/archive/build/s1-c-rep2 --host-baseline docs/archive/build/s1-hostq`:

```text
PASS s1-c-rep1 rank0..3 fixture b08cbf341ad85320: max|d| 5.61e+00 mean|d| 5.47e-01 KL 2.30e-01 top1 0.667 top5 0.668 top50 0.627 ppl 13.4955 bars[max 8.50e+00 mean 8.50e-01 KL 3.50e-01] ref-min-margin 0.00 host[max 0.00e+00 mean 0.00e+00 KL 0.00e+00 ppl 13.7747]
PASS s1-c-rep2 rank0..3 fixture b08cbf341ad85320: max|d| 5.61e+00 mean|d| 5.47e-01 KL 2.30e-01 top1 0.667 top5 0.668 top50 0.627 ppl 13.4955 bars[max 8.50e+00 mean 8.50e-01 KL 3.50e-01] ref-min-margin 0.00 host[max 0.00e+00 mean 0.00e+00 KL 0.00e+00 ppl 13.7747]
NUMERICAL GATE PASS: teacher-forced, every rank inside the declared bars (D54 amended by D55)
```

Two root-cause findings settled the earlier D55 red verdict and produced **D56** ([`../DECISIONS.md`](../DECISIONS.md), [`../RULES.md`](../RULES.md) §3, [`reference/speed-gate/bars.json`](reference/speed-gate/bars.json)):

1. **Perplexity bug in D55's initial `scripts/numerical-gate.sh`:** Line 144 of `scripts/numerical-gate.sh` previously computed `ppl` over `token_json["tokens"]["ids"]` (each arm's *own* generated argmax token IDs) rather than the fixed reference target sequence `ref["tokens"]`. Because `s1-devq` had slightly higher peak logits on its own argmaxes, `exp(mean(-logprob(own_argmax)))` evaluated to `10.2440` vs `13.7747` (−26%), measuring self-confidence rather than teacher-forced perplexity. Fixed to evaluate both arms on `ref["tokens"]`, `s1-c-rep1` / `s1-c-rep2` achieve **`ppl = 13.4955` vs host `13.7747` (−2.03%, ΔNLL = −0.0204 nats/token)**, and lower NLL (`4.7790` vs `5.1073`) on the teacher-forced continuation tail `forced_ids[130..138]`.
2. **Layer-0 `attn` and `moe` attribution (`s1-ord3`, `s1-rdx3`, `s1-diff-full`):**
   - When `mhc_entry` uses serial CPU order (`s1-ord3`), stages 1–11 (`normed`, `lora`, `qnorm`, `q`, `body`, and the four ranks' `wo_b` partials) are **100% bit-identical (`0/5120` diffs)** to the host arm.
   - At Stage 12 (`all_reduce`), `ncclAllReduce(ncclSum, ncclBfloat16)` rounds in `bf16` at each pairwise ring/tree addition rather than accumulating in `f32` on the host and rounding once, introducing a 1-ULP `bf16` difference on `1,579 / 5,120` channels (~31%).
   - That 1-ULP `bf16` perturbation enters `attention_residual` → `ffn_seams` → `spark_seam_quantise_row` (3-bit mantissa FP8 E4M3), flipping ~1–2% of the 5,120 FP8 codes. Through the dense-width `[576, 5120]` / `[5120, 576]` routed and shared MoE projections, those flipped FP8 codes perturb 4,625 / 5,120 MoE output channels by `~1e-3` (`max_abs = 4.88e-3`). The reported `rel = 2.0` with 30 sign crossings occurs solely on tiny `~1e-4` channels straddling zero, where `|x - y| / max(|x|, |y|) ≈ 2.0`.
   - Likewise, feeding identical `stream` and `pre_mix` into the device head (`collapse_for_head` + `screen_device` in `s1-diff-full`) produces **bit-identical (`max|dlogit| = 0.000000`)** 129,280-vocab logits to the host head.

### 8.4 The stale-capture guard (permanent)

A host read that follows a queued collective must wait on it, enforced by a per-layer `collectives_in_flight` counter: `reduce_row` increments it, both read paths (`copy_out` for f32 rows, `read_row_bytes` for bf16 rows) **refuse** while it is non-zero, and `wait_collectives()` clears it after a stream-0 synchronize. This prevents any diagnostic or future S3 graph-capture boundary from reading a stale pre-collective buffer.
