# WS-A: the four-rank decode step host/launch path off the critical path (M2 C1/C2/C3/C5/C10)

Status: **device-validated twice** (windows m2p1 and m2p1b) and **merged after an adversarial
review**. Arithmetic is unchanged: only storage, module, launch and evidence-recording wrappers
moved. Measured: step mean 1201.81 -> 842.69 ms (-29.9 %), byte-identical tokens and logits sha on
all four ranks; the evidence-ON identity row reproduces the pre-M2 chain digest for digest.

Files in this change-set:

* `crates/sparkflash-kernels/src/cuda_driver.rs` — async launch path (C3), cubin/module cache (C1),
  buffer pool (C2).
* `crates/sparkflash-cli/src/tp4_forward.rs` — run-scope switches, cached cubin reads, pooled
  per-layer buffers, skip-zeroed-store upload (C5), evidence gate (C10), `evidence_mode` /
  `evidence_digest_calls` in the embed artifact, tests.

Every number in §2 is labelled **measured** or **estimate**.

## 1 What changed

| item | change | where |
|---|---|---|
| C3 | `Cuda::launch` dispatches to `launch_async` (queue and return) when no bracket consumer is armed and the async run switch is on. Probe meter / `SPARKFLASH_STAGE_TIMING` runs take the old drained path, so probe attribution is unchanged. Env opt-out: `SPARKFLASH_SYNC_LAUNCHES=1`. The switch is a process-global `AtomicBool` (serialized through the driver lock); it is documented as such because an embedder that launches kernels from another thread during a forward run inherits async semantics. | `cuda_driver.rs:967,993,1836,1883` |
| C1 | Per-run cubin cache in the CLI keyed by path, returning shared `Arc<[u8]>` + sha once; driver `ModuleCache` keyed by image bytes with a `(pointer, length)` fast path that **byte-verifies the cached image** and falls through to a byte comparison on mismatch (a reused address with different bytes is a new id); `Cuda::free(module, &[])` is a no-op for a cached module (unloaded at drop). Removes 7 of 8 cubin reads+sha and every repeat module load per layer-position. | `tp4_forward.rs:1346,1367`; `cuda_driver.rs:1007,1037,1778,2089` |
| C2 | `alloc_pooled`/`release_pooled` with an `issued`/`completed` fence: release stamps `needed = issued`, checkout only reuses when `needed <= completed`; a D2H or explicit synchronize is the completion point. Ownership is tracked per pointer (`checked_in`), a double/unknown release is refused instead of duplicating a free-list entry, `free`/`free_buffers` forget a pooled pointer rather than `cuMemFree` it, and `Drop` frees only the authoritative free lists. Removes 34 `cuMemAlloc`/`cuMemFree` per layer-position. | `cuda_driver.rs:1104,1135,1403,1439,2187,2224,2245` |
| C5 | `AttentionBody` zero-fills the 150,272-byte expanded store once in `open`; `run` skips the 147,456-byte H2D when the staged store is all zero (`store_rows == 0`) and the device bytes are already zero, and skips an empty selected upload. | `tp4_forward.rs:2581,2757,2885` |
| C10 | One switch gates the whole evidence chain: every per-layer/per-position data digest goes through `evidence_sha256` (empty string when off, exactly `sha256_hex` when on), and the layer-chain JSON itself is only built when on. `layers_completed` is counted independently so progress and the position record stay correct. The embed artifact always states `evidence_mode` (`recording`/`off`) and `evidence_digest_calls`. | `tp4_forward.rs:1385-1430,7880,8272` |

### Run-scope switches (exactly two, logged at run start)

```text
async launches        on by default                 SPARKFLASH_SYNC_LAUNCHES=1   -> per-launch drain (old path)
evidence chain        off by default                SPARKFLASH_FORWARD_EVIDENCE=1 -> per-layer chain + digests
```

`run_forward_loop` prints `forward loop: launch path <async|synchronous> ... evidence chain
<recording|off>` before the first launch. The two switches are independent (unit test
`the_launch_path_and_the_evidence_chain_are_independent_switches`).

## 2 Measured result (windows m2p1 and m2p1b, four ranks, all rc=0)

| row | env | step mean (4 ranks) | tokens | logits bin sha |
| :-- | :-- | --: | :-- | :-- |
| baseline m1w37-lean | old path, evidence ON | 1201.81 ms | identical | `aed6054d...` |
| B lean (m2p1) | async, evidence OFF | 845.86 ms | identical | `aed6054d...` |
| D sync (m2p1) | `SPARKFLASH_SYNC_LAUNCHES=1`, evidence OFF | 843.94 ms | identical | `aed6054d...` |
| C probe (m2p1) | probe armed (drained path), evidence OFF | 852.06 ms | identical | `aed6054d...` |
| A identity (m2p1b) | async, **evidence ON** | **925.19 ms** | identical | `aed6054d...` |
| **B2 lean (m2p1b)** | **async, evidence OFF (clean argv)** | **842.69 ms** | identical | `aed6054d...` |

Identity row A: `layer_chain` 40 entries, **identical to the pre-M2 `m1w37-lean` chain after
stripping `wall_ms`** (every digest and data-dependent field byte-identical); `evidence_digest_calls
= 271,194` versus **0** in row B2, whose artifact says `"evidence_mode": "off"`. The only other
non-timing differences are staging paths.

Probe row C (attribution; regenerate with `python3 scripts/tp4-probe-attribution.py <run> <baseline>
--markdown`): seam-layers `layer_total` 25.722 -> 19.295 ms/layer (-25.0 %); all-layers 27.341 ->
20.782; probe-scope 27.366 -> 20.803. Blocks (seam basis): attention_projections 5.172, moe_gate_up
4.767, ffn_shared 2.744, attention_output 2.393, moe_reduce 0.969, query_path 0.868, ffn_seams 0.516,
ref_seam 0.420, mhc_entry 0.322, moe_router 0.261, moe_stored_combine 0.220, attention_body 0.188,
stage_attention 0.165, attention_residual 0.069, kv_path 0.056, ffn_hc_post 0.054, moe_weighted
0.012, ffn_combine 0.006, entry_json 0.000 (was 0.201).

Driver meter (`run.loop_total`, over 5160 layer-positions): alloc 187,680 -> 28; free 187,680 -> 0;
module_load 49,680 -> 7; sha256 calls 315,378 -> 30 (bytes 9.52 GB -> 0.334 GB, time 16.82 s ->
0.58 s); launches 66,258 unchanged; d2h 88,374 unchanged (12 per layer-position); h2d count 126,978
-> 126,370 and bytes 2.011 -> 1.988 GB.

The async-vs-sync control (rows B and D) is the isolation: +1.92 ms/step, wrong direction and inside
run noise, so **C3 contributes no measurable step time on this path**. The 356-359 ms is C10 plus
C1/C2. C5's isolated effect is unmeasured: H2D per layer-position fell only 23.00 -> 22.89.

## 3 Exactness argument

**(i) All device->host observations are `cuMemcpyDtoH_v2`.** The layer path has 14 grep hits; the
driver meter shows **12 fire per layer-position** (7 blocks: moe_router 3, attention_output 2,
attention_projections 2, query_path 2, attention_body 1, kv_path 1, moe_stored_combine 1; two hits are
conditional and did not fire in these rows). `cuMemcpyDtoH_v2` is synchronous and stream-ordered on
the legacy default stream, and every launch in this code uses stream 0. A host read therefore observes
exactly the bytes produced by every kernel queued before it — async launches change *when* the host
blocks, never *what* it reads.

**(ii) There is no second observation mechanism.** No host-mapped/direct device pointer read exists in
`crates/sparkflash-cli`: `grep -n 'HostMapped|host_mapped|alloc_host' crates/sparkflash-cli/src` is
empty; `HostMapped` appears only inside `cuda_driver.rs`. So (i) is complete.

**(iii) Nothing is freed, unloaded or overwritten while a queued kernel may read it.** Cached modules
are only unloaded from `Cuda::drop` after a final `synchronize`. Pool release never frees; a checkout
either reuses a buffer only after a completion point strictly later than the release
(`issued`/`completed` fence) or calls `cuMemAlloc` for a fresh region. A pooled pointer is owned by
the pool: `free`/`free_buffers` forget it instead of freeing it, `Drop` frees only checked-in free
lists, and a double release is refused, so no allocation can be handed to two live kernels or freed
twice. `free`/`free_buffers`/`free_managed`/`free_host_mapped` drain first whenever the async path is
on. Buffer identity is numerically irrelevant because every launch rewrites its argument list each
layer and every buffer is fully written by H2D or by its producing kernel before the next observation.

**(iv) The evidence mode is the old code.** `evidence_sha256` is literally `sha256_hex` when the flag
is on, and the 66 gated digest sites plus the layer-chain JSON are the only things gated. Measured:
row A's chain is identical to the pre-M2 run digest for digest (§2).

## 4 C5: what the skip is and what it measured

* **Expanded store (147,456 B).** `stage_attention` hands back `kv_codes`/`kv_scales` views of
  `EXPANDED_ROWS * STORE_CODE_BYTES` / `... * STORE_SCALE_BYTES` initialised to zero, and writes only
  the selected rows; `store_rows == 0` therefore means every staged byte is zero. The device store is
  zero-filled once in `AttentionBody::open`; no kernel writes it (it is a read-only input), and every
  later zero store keeps it zero. Skipping the upload writes the same bytes.
* **Selected list (0 bytes).** `selected_count == 0` means the kernel reads no element; the previous
  upload left exactly the bytes an empty upload would leave.
* **Measured effect:** H2D per layer-position 23.00 -> 22.89 (-0.5 %; attention_body 7.00 -> 6.89), so
  the skip fires on few layer-positions of a prefill-dominated probe. The decode-step effect is not
  isolated and is not claimed.

## 5 Residual syncs / allocations left in the layer loop (window audit)

Layer loop = 40 layers x every position. Counts per layer-position, measured (probe row, 5160
layer-positions).

| site | count | per-layer drain? | why it stays |
|---|---|---|---|
| pooled alloc/release | 17 call sites, 34 buffer ops | no | fence only; a pool miss allocates, never frees (measured 28 allocs, 0 frees for the whole run) |
| module load (`::load` -> `load_function`) | 9 handles | no | `ModuleCache` hit after the first position's first layer (measured 7 loads for the whole run) |
| `cuda.free(module, &[])` | 9 handles | no | cached module: no synchronize, no `cuModuleUnload` |
| `cuMemcpyDtoH` | **12 measured** (14 grep hits) | **yes, by design** (stream-ordered, synchronous) | the host reference arithmetic consumes these bytes: router ids/weights/scores, gate/up outputs, slot accumulator, projection outputs, attention out |
| `read_arena_bytes` | 1 fixed per layer (query norm) + N inside the mHC-entry arena reads | **yes, by design** | the mHC entry is host reference arithmetic by contract |
| explicit `cuda.synchronize()` | 0 in a plain/lean row | no | only `probe::record`, probe rows only |
| `copy_htod` | body q/window/store/selected/sink + mHC/projection inputs | no context drain (pageable H2D) | not a `cuCtxSynchronize`; still host time, priced in the probe blocks |

So the async win is bounded by the 12 design D2H observation points, not by hidden alloc/free/module
drains. Measured: `alloc_count`/`free_count` = 28/0 for the whole run (was 187,680/187,680),
`module_load_count` = 7 (was 49,680), `d2h_count` unchanged at 88,374, `syncs` still 66,258 (the probe
deliberately keeps the drained launch path).

## 6 Validation recipe (Lead-owned windows)

Frozen argv = the P0 lean row command for this checkpoint (4 ranks, `--forward-lean`). Prefixes only:

| row | command | what it proves | result |
|---|---|---|---|
| A identity | `SPARKFLASH_FORWARD_EVIDENCE=1 <frozen argv> --forward-lean` | chain JSON digest-identical to pre-M2; tokens and logits hashes still exact | **measured 925.19 ms; chain identical (wall_ms only)** |
| B step | `<frozen argv> --forward-lean` | the gate/throughput number: async launches on, evidence off (default) | **measured 842.69 ms** |
| C attribution | `<frozen argv> --forward-lean --forward-perf-probe` | per-block host residual after C1/C2/C5/C10; driver meter proves the counters collapsed (**sync launches by design - never compare its step mean to row B**) | **measured 852.06 ms; counters in §5** |
| D rollback | `SPARKFLASH_SYNC_LAUNCHES=1 <frozen argv> --forward-lean` | old path still works and still matches baseline | **measured 843.94 ms, identical tokens/sha** |

Acceptance anchors: tokens `[491, 92928, 201, 15, 4893, 16646, 15, 14265, 86]`; `--dump-logits` bin
sha256 `aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2`; row A also reproduces the
pre-M2 chain (wall_ms excluded); row B vs baseline 1201.81 ms; row D within noise of baseline.

## 7 Unverified / risks

* The async launch path has one four-rank, nine-step validation and the sync control; it is exact for
  this configuration, not proven for every future configuration. The identity row is the evidence for
  the recording path only.
* Probe rows run the drained launch path on purpose; only rows B/B2 measure the async step.
* With evidence off, the per-position `logits_sha256` JSON fields are empty by design; the artifact
  says `evidence_mode: off` and `evidence_digest_calls: 0`. Use the dumped logits bin sha (or row A)
  for hash acceptance, not the lean row JSON.
* The evidence switch is thread-local (like the probe state); the async switch is process-global.
* C5's decode-step effect is unmeasured (not claimed).
