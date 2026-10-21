# M2 execution plan after lean+hoist (2026-10-21)

Inputs: M2 profile doc m2-critical-path-profile-20261021.md (measured decomposition + E record),
Phase 4A Engram analysis (analysis/engram-4a-20261021/ANALYSIS.md d8c190f8...), Phase 4B layer-body map
(analysis/layer4b-20261021/ANALYSIS.md a134e995...), device-lead 2C scope (device-lead/m2-2c-scope-20261021/)
and A/B verification (device-lead/m2-ab-verify-20261021/). All per-block numbers below are ceilings estimated
from code and retained evidence unless marked measured; the probe rows are the settling measurement.

## 1. Measured state (m1y34-lean, four ranks rc=0, logits bin aed6054d...)

Decode step mean 1407.9 ms (default evidence-bearing row 2030.7 ms). Lean decomposition, rank 0:

| component | ms | status |
| :--- | ---: | :--- |
| 38 ordinary layer bodies (mean 32.79 each, range 25-47) | 1246 | measured per-layer wall |
| Engram layers 1 and 14 | 54-56, 79 | measured per-layer wall |
| 40-layer sum | 1379 | measured |
| head residual / outside-layer at a head position | ~26 | measured |
| position overhead | ~3 | measured |

Engram = +68.2 ms/token over ordinary layers (4.8% of the step); it is a secondary lever. The ordinary
layer bodies are 88.5% of the step. No tok/s, TTFT or serving claim.

## 2. Why the ordinary layer is slow: the structural map (4B, code-level, not measured time)

- Cuda::launch calls cuCtxSynchronize after EVERY launch (sparkflash-kernels/src/cuda_driver.rs:1287).
  An ordinary layer issues 12 launches -> 480 launches + 480 full syncs per token. There is no async path.
- Per ordinary layer: 23 H2D + 15 D2H + 36 D2D (30.1 MB MoE weight staging) + 34 cuMemAlloc/Free +
  9 cuModuleLoadData/Unload + 8 fs::read+sha256 of ~110 KB of cubin + 2 blocking TCP collectives.
- Host arithmetic ~48M MAC/layer, dominated by the shared expert on the host ([2304x5120] gate+up and
  [5120x2304] down = 35.4M MAC) - there is no device shared-expert kernel at HEAD.
- ref_seam is a host reference forward whose compressed ids feed the attention kernel, so no CUDA graph can
  span the layer until the candidate select moves to the device.
- Every block is host->H2D->launch(sync)->D2H->host arithmetic: no run-ahead, GPU mostly idle.
- stage_attention zeroes+uploads 147,456 B on every layer even when selected=0.

So the layer body is latency-bound by host serial work and per-call driver overhead, not by device throughput.
That is exactly what the M2 criterion "no host round-trips" targets.

## 3. Candidate levers (ceilings from 4A/4B/2C; acceptance = logits bin sha aed6054d... plus per-layer/head/stream shas)

| id | lever | ceiling / token | effort | arithmetic | notes |
| :--- | :--- | ---: | :--- | :--- | :--- |
| C1 | cache cubin bytes + module handles per run (no fs read/sha/load per layer) | 18-72 ms | S | neutral | 8 reads + 9 module ops/layer |
| C2 | per-run device buffer pool instead of 34 alloc/free per layer | 27-135 ms | M | neutral | pairs with C3 |
| C3 | launch_async + sync only where a result is observed | 2-14 ms + run-ahead | M | neutral | overlapping host work hides latency |
| C5 | reuse/skip the zeroed expanded-store upload for ratio-0 layers | 4-12 ms | S | neutral | 147,456 B/layer |
| C10 | lean-mode entry_json/digest deferral | 3-10 ms | S | neutral | lean only |
| C4 | 2C per-layer host seam cache (8 reads/layer -> 0, cached digests) | ~0.1-0.25 s hypothesis | M (2-2.5 d) | neutral | device-lead scoped; 158.9 MB D2H + 79.5 MB sha per position |
| C6 | device shared expert with exact-order f32 replay | 0.27-0.53 s | L | bit-identical by construction if order matches | biggest single lever; needs D49 pin + a f32-order harness |
| C7 | device ref_seam and/or cached weight views | 76-152 ms | L/M | risk | unblocks graph capture later |
| C11 | MoE D2D staging via pointer indirection | 2-8 ms | M | neutral | 30.1 MB/layer |
| C8 | merge the two gate_up launches | 4-12 ms | S | neutral | same kernels/order |
| Engram C1 | run-scope residency + read_view zero-copy | 45-57 ms | M | neutral | 4A plan, after C1/C2 |
| Engram C2 | device-resident wkv + existing kernels | 62-66 ms | L (3-5 d) | risk | 4A plan |
| C9 | CUDA graph capture of the layer body | <10 ms at HEAD | XL | neutral if all host seams gone | only after C6/C7 + 2C |

## 4. Probe decision rules (rows C/D; probe must split host wall vs kernel_ns per block)

- ffn_epilogue.shared >= 25% of layer wall -> C6 first; 10-25% -> C1-C3 then C6.
- ref_seam >= 15% -> C7; host-overhead (sum block walls - kernel_ns) >= 40% -> C1+C2+C3 first;
  <= 15% -> device-bound, skip graph capture.
- module_load+alloc >= 3 ms/layer -> C1+C2; moe kernel time << moe wall -> C11;
  attention_body kernel >= 50% -> attention kernel first; attention_twin must be 0 in lean rows.
- Expected per ordinary layer: 12 launches / 12 syncs / 34 allocs+frees / 9 module ops / 23 H2D + 15 D2H + 36 D2D
  (counters validate the map before any ceiling is trusted).

## 5. Sequencing

1. Phase P1 (now): probe kit merge -> rows C (--forward-perf-probe) and D (--forward-lean --forward-perf-probe);
   both must keep aed6054d. This produces the first measured host-vs-device split and the attribution needed to
   rank C1..C11 and 4A.
2. Phase P2: one arithmetic-neutral batch kit = C1 + C2 + C3 + C5 + C10 (and C8/C11 if the probe ranks them
   in) + 2C (C4). Default-path byte-identity row A (aed6054d) is the acceptance; expected combined measured
   ceiling 0.1-0.35 s from the cheap items plus C4's 0.1-0.25 s hypothesis. Lean row B measures the step.
3. Phase P3: C6 device shared expert (exact-order f32; bit-identical proof + D49 pin + kernel baseline).
   Expected 0.27-0.53 s if the probe confirms the host shared expert share. If exact bit-identity proves
   impossible, the fallback is a declared-tolerance A/B with the D41 gate re-run, not a silent change.
4. Phase P4: C7 ref_seam device-side + C11/C8 leftovers, then Engram 4A C1 -> C2.
5. Phase P5: CUDA graph capture only after all host seams are gone (C4/C6/C7), with bit-identical logits.
6. Each phase: kit staged by Lead, gates (fmt/clippy/workspace tests/scripts tests/cuda check), row A identity,
   row B lean delta, docs E record, LEAD-NOTES. No bar/oracle/comparator/contract change; no pin change without
   the standing D49 re-pin rule; any speed change moving numerics outside the declared bars is rejected.

## 6. Honest position

Even with every cheap lever ~0.1-0.35 s + 2C 0.1-0.25 s + C6 0.27-0.53 s + C7 0.08-0.15 s, the lean step is
projected at ~0.4-0.75 s, i.e. still 6-11x from 66.7 ms. M2 15 tok/s requires the device loop and kernels
(P5) to land. The value of P1-P4 is that each step is measured, byte-identical and reversible, and every
ceiling becomes a measured number before the next one is attempted.
