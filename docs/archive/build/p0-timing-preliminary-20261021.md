# P0 preliminary per-block timing table (probe rows m1y35, kit m1w35)

Status: PRELIMINARY (block host-wall only; the P0 gate's kernel/staging/host-sync/launch-overhead split
needs the probe-v2 row, in development now). The final table is the device lead's P0 deliverable.

Evidence (kept outside the repo, same as other windows):
- window/m1y35-probe-default and window/m1y35-probe-lean under /home/user/ev-session-20261020 (also mirrored
  on the nodes under /home/user/ev-session-window/m1w35/).
- raw perf files: window/m1y35-probe-*/rank{0..3}/tp4-forward-perf-r{rank}.json, schema
  sparkflash-tp4-forward-perf/1, 2.5 MB/rank; token/loop evidence next to them.
- kit: m1w35, binary sha256 13cc4ebb9c0caf66d70d34fb099ee9089fb20aa8b462b44b839cf45fe767283d on all four nodes,
  built from integration tip 254d64b6.
- rows: m1y35-probe-default = no --forward-lean, --forward-perf-probe; m1y35-probe-lean = both flags.
- both rows: 4/4 ranks rc=0, logits bin sha256
  aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2 (identical to the m1y34 default and
  lean rows), tokens [491,92928,201,15,4893,16646,15,14265,86].

Probe overhead vs the non-probe rows (C-A and D-B): default 2000.1 vs 2030.7 ms (-1.5%, i.e. inside row noise),
lean 1446.2 vs 1407.9 ms (+2.7%). Treat +/-2-3% as the row-to-row band for this table.

No tok/s, TTFT, serving or acceptance claim.
step and row-to-row variance is ~2%). Blocks are host-wall with one cuCtxSynchronize per block, so a block is
device-inclusive and NOT split into kernel vs host yet; the kernel_ns split is the next probe iteration.

## C-default-probe: step mean 2000.1 ms; plain position 1411.6 ms; head position 2001.3 ms

ordinary-layer block means (ms/layer), plain positions | head positions | share of plain layer wall:
| attention_projections | 6.936 | 6.936 | 20.8% |
| attention_twin | 6.058 | 14.365 | 18.2% |
| moe_gate_up | 5.229 | 5.101 | 15.7% |
| mhc_entry | 4.461 | 4.514 | 13.4% |
| attention_output | 3.520 | 3.697 | 10.5% |
| ffn_shared | 3.399 | 4.036 | 10.2% |
| attention_body | 1.607 | 15.182 | 4.8% |
| query_path | 1.532 | 1.530 | 4.6% |
| moe_reduce | 1.246 | 1.475 | 3.7% |
| ffn_seams | 0.849 | 0.895 | 2.5% |
| ref_seam | 0.404 | 0.416 | 1.2% |
| attention_residual | 0.379 | 0.380 | 1.1% |
| moe_stored_combine | 0.346 | 0.347 | 1.0% |
| moe_router | 0.318 | 0.322 | 1.0% |
| entry_json | 0.202 | 0.199 | 0.6% |
| stage_attention | 0.165 | 0.324 | 0.5% |
| kv_path | 0.067 | 0.067 | 0.2% |
| ffn_hc_post | 0.058 | 0.057 | 0.2% |
| moe_weighted | 0.012 | 0.012 | 0.0% |
| ffn_combine | 0.006 | 0.006 | 0.0% |
| **ordinary layer wall (mean)** | **33.37** | **48.26** | 100% |

Engram layers (L1/L14) block means (ms/layer):
| engram_write | 21.60 (n=258) | 22.02 (n=18) |
| attention_output | 15.84 (n=258) | 10.51 (n=18) |
| attention_projections | 6.81 (n=258) | 6.82 (n=18) |
| attention_twin | 4.67 (n=40) | 11.01 (n=18) |
| mhc_entry | 4.37 (n=258) | 4.37 (n=18) |
| ffn_shared | 3.43 (n=258) | 3.57 (n=18) |
| ref_seam | 1.96 (n=258) | 1.96 (n=18) |
| moe_gate_up | 1.75 (n=258) | 1.76 (n=18) |
| query_path | 1.43 (n=258) | 1.43 (n=18) |
| attention_body | 1.35 (n=258) | 11.75 (n=18) |
| ffn_seams | 0.82 (n=258) | 0.86 (n=18) |
| moe_reduce | 0.59 (n=258) | 0.58 (n=18) |
| attention_residual | 0.38 (n=258) | 0.38 (n=18) |
| moe_stored_combine | 0.34 (n=258) | 0.35 (n=18) |
| moe_router | 0.32 (n=258) | 0.32 (n=18) |
| entry_json | 0.20 (n=258) | 0.20 (n=18) |
| stage_attention | 0.07 (n=258) | 0.14 (n=18) |
| kv_path | 0.07 (n=258) | 0.07 (n=18) |
| ffn_hc_post | 0.06 (n=258) | 0.06 (n=18) |
| moe_weighted | 0.01 (n=258) | 0.01 (n=18) |
| ffn_combine | 0.01 (n=258) | 0.01 (n=18) |

covered by blocks: plain 36.79, head 59.86; layer wall mean plain 33.37 head 48.26

head blocks (last head position): {"hc_pre": 0.01, "final_norm": 0.04, "seam_sha_write": 0.14, "screen": 2.97, "read_outputs": 0.04, "gather_logits": 0.93, "side_gather": 0.29, "winner_check": 0.0, "digests": 2.09, "head_record": 0.01, "head_total": 6.57}

run: {"embedding_gathers": 56904.6, "shared_prewarm": 1373.3, "tail_open": 1355.4, "loop_total": 200145.6, "tail_close": 0.3, "evidence_write": 1.5, "consensus": 0.2, "dump": 35.6, "barrier": 0.7, "counters": {"arena_reads": {"count": 55481, "bytes": 22193801600, "ms": 869.505}, "sha256": {"count": 331818, "bytes": 19840583504, "ms": 34901.95}}}

## D-lean-probe: step mean 1446.2 ms; plain position 1422.6 ms; head position 1448.7 ms

ordinary-layer block means (ms/layer), plain positions | head positions | share of plain layer wall:
| attention_projections | 6.965 | 6.952 | 20.8% |
| moe_gate_up | 5.404 | 5.345 | 16.1% |
| mhc_entry | 4.484 | 4.475 | 13.4% |
| ffn_shared | 3.684 | 3.328 | 11.0% |
| attention_output | 3.674 | 4.345 | 11.0% |
| query_path | 1.539 | 1.533 | 4.6% |
| moe_reduce | 1.336 | 1.539 | 4.0% |
| ffn_seams | 0.858 | 0.856 | 2.6% |
| attention_body | 0.670 | 0.821 | 2.0% |
| ref_seam | 0.444 | 0.416 | 1.3% |
| attention_residual | 0.379 | 0.380 | 1.1% |
| moe_stored_combine | 0.350 | 0.345 | 1.0% |
| moe_router | 0.320 | 0.317 | 1.0% |
| entry_json | 0.203 | 0.198 | 0.6% |
| stage_attention | 0.165 | 0.326 | 0.5% |
| kv_path | 0.067 | 0.067 | 0.2% |
| ffn_hc_post | 0.058 | 0.057 | 0.2% |
| moe_weighted | 0.012 | 0.012 | 0.0% |
| ffn_combine | 0.006 | 0.006 | 0.0% |
| **ordinary layer wall (mean)** | **33.55** | **34.13** | 100% |

Engram layers (L1/L14) block means (ms/layer):
| engram_write | 36.95 (n=258) | 36.61 (n=18) |
| attention_projections | 6.85 (n=258) | 6.85 (n=18) |
| mhc_entry | 4.39 (n=258) | 4.39 (n=18) |
| attention_output | 3.35 (n=258) | 4.45 (n=18) |
| ffn_shared | 3.08 (n=258) | 2.96 (n=18) |
| ref_seam | 2.11 (n=258) | 2.02 (n=18) |
| moe_gate_up | 1.78 (n=258) | 1.78 (n=18) |
| query_path | 1.42 (n=258) | 1.44 (n=18) |
| moe_reduce | 0.91 (n=258) | 1.17 (n=18) |
| ffn_seams | 0.80 (n=258) | 0.81 (n=18) |
| attention_body | 0.63 (n=258) | 0.75 (n=18) |
| attention_residual | 0.38 (n=258) | 0.38 (n=18) |
| moe_stored_combine | 0.35 (n=258) | 0.35 (n=18) |
| moe_router | 0.32 (n=258) | 0.32 (n=18) |
| entry_json | 0.20 (n=258) | 0.20 (n=18) |
| stage_attention | 0.07 (n=258) | 0.14 (n=18) |
| kv_path | 0.07 (n=258) | 0.07 (n=18) |
| ffn_hc_post | 0.06 (n=258) | 0.06 (n=18) |
| moe_weighted | 0.01 (n=258) | 0.01 (n=18) |
| ffn_combine | 0.01 (n=258) | 0.01 (n=18) |

covered by blocks: plain 30.62, head 31.32; layer wall mean plain 33.55 head 34.13

head blocks (last head position): {"hc_pre": 0.01, "final_norm": 0.04, "seam_sha_write": 0.12, "screen": 2.94, "read_outputs": 0.04, "gather_logits": 1.39, "side_gather": 0.29, "winner_check": 0.0, "digests": 2.09, "head_record": 0.01, "head_total": 6.97}

run: {"embedding_gathers": 15210.6, "shared_prewarm": 1378.5, "tail_open": 1424.0, "loop_total": 196594.8, "tail_close": 0.3, "evidence_write": 1.6, "consensus": 0.5, "dump": 35.5, "barrier": 1.6, "counters": {"arena_reads": {"count": 55481, "bytes": 22193801600, "ms": 882.093}, "sha256": {"count": 331818, "bytes": 19840583504, "ms": 34926.448}}}

## C -> D (lean) deltas

- plain position: 1411.6 -> 1422.6 ms; head position 2001.3 -> 1448.7 ms
- plain ffn_shared: 3.40 -> 3.68 (delta +0.29)
- plain attention_body: 1.61 -> 0.67 (delta -0.94)

## anomalies to resolve in the next probe iteration

1. engram_write differs between the two rows (plain 21.6 ms in C vs 36.9 ms in D; same lean-untouched code) - likely
   NVMe/page-cache state or CPU scheduling; needs a repeat row before being quoted as the Engram cost.
2. embedding_gathers (run scope) 56.9 s in C vs 15.2 s in D - clearly caches/thermal, not lean; it is pre-loop (TTFT) cost,
   not part of the step mean, but it needs a warm-cache repeat before any TTFT statement.
3. block sum vs layer wall is not additive in a simple mean (Engram blocks appear only in 2/40 layers); the table reports
   ordinary-layer and Engram-layer means separately. Per-position layer wall - block sum = untimed gap (allocations,
   module loads, launch overhead outside blocks) ~3-4 ms/layer in both rows.
4. No kernel_ns split yet: next probe iteration adds per-block kernel time from the driver stage-timing ledger plus
   counters launches/syncs/h2d/d2h/dtod/allocs/frees/module_loads, which the P0 gate requires.

## first decisions (probe rules from the M2 execution plan)

- attention_projections is the largest single block per ordinary layer (~7 ms, ~20% of the layer wall) and cannot be
  device-bound at these sizes: it is the first C1/C2/C3 target (module/cubin cache, buffer pool, async launch).
- mhc_entry (~4.5 ms, ~13%) and ffn_seams (~0.85 ms) are the 2C seam-cache targets; 2C is in implementation now.
- moe_gate_up (~5.2 ms, ~15%) is the only large device kernel block -> split-K/grouped MoE is the kernel target (D51).
- ffn_shared (~3.4-3.7 ms, ~10%) is in the 10-25% band -> C1-C3 first, C6 device shared expert after.
- attention_body is only ~0.7 ms in lean (device kernel is cheap at this shape); attention_twin is the 14.2 ms/head-layer
  diagnostic cost and is already excluded in lean.
- ref_seam ~0.5 ms (<15%) -> not a C7 priority yet.

No tok/s, TTFT or acceptance claim. Both rows keep the byte-identical logits sha above and 4/4 ranks rc=0.