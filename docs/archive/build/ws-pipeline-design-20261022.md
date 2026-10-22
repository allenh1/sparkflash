# WS-P: can the layer loop overlap host and device work, and what is the reachable bound?

Status: design note, 2026-10-22. No code, no device window, no rows. This note answers the
question at the end of docs/build/ws-gb10-host-device-contention-20261022.md (MR !163). Everything
here is either the probe's measurement (that document and docs/build/p0-timing-table-20261021.md,
labelled measured) or arithmetic on those numbers (labelled derived). Nothing here is a new
measurement.

## 0. Frame

- Published step 249.25 ms (268f2ab1), device kernel 71.2 ms/step, probe 6.18 ms/layer wall and
  1.86 ms/layer kernel over 40 layers. 40 x 6.18 = 247.2, so the step's non-kernel time is
  249.25 - 71.2 = 178.05 ms (derived): the host serial, including drains and the two TCP
  all-reduces.
- Every remaining block is host-bound (measured, contention doc lines 114-124): attention_output
  0.837 wall / 0.202 kernel, moe_gate_up 1.894 / 0.637, moe_router 0.423 / 0.352,
  moe_stored_combine 0.316 / 0.278, attention_projections 0.422 / 0.142, ffn_shared 0.324 / 0.238,
  mhc_entry 0.318 host, ffn_seams 0.521 host, ref_seam 0.501 host.
- The ILP window measured host -17 ms against kernel +33.7 ms, net +43 ms/step, with a bimodal
  ~230 ms stall. Host speed is not the lever, and concurrent CPU work is measurably hostile to
  device kernels on this package.

## 1. The loop as coded (crates/sparkflash-cli/src/tp4_forward.rs)

Per position, per layer (loop at :8617). "Host" means CPU code that runs with the device idle
between drains; "device" means launches inside the graph segment. Every copy_dtoh is a hard
host/device alternation.

| seam (anchor) | side | reads -> produces | barrier role |
|---|---|---|---|
| engram_write :8690 | host, layers 1/14 | hc stream f32[4x5120], engram resident source -> updated stream | - |
| mhc_entry :8708 | host | stream, cached attn predictor/norm, attn_pre_mix (prev layer's ffn_pre) -> pre/post/comb mixes, attn_in, normed | - |
| attention_projections :8712 (fn :6274) | device 2 GEMMs + D2H | normed -> H2D E4M3 codes -> wq_a latent bf16[1280], wkv latent bf16[512] as host bytes | - |
| query_path :8723 (fn :3060) | host q_norm/quantise, device wq_b :3206, host rope :3240 + D2H :3208 | wq_a latent, q_norm weight -> q_post_rope bf16[16x512], q_latent host (input side, :3244) | - |
| kv_path :8735 (fn :2506) | host (one 1 KiB weight D2H) | wkv latent, kv_norm weight, rope tables -> window codes 512 B + scales 16 B (host); row pushed to the host KV cache :8747 | - |
| ref_seam :8753 (ReferenceAttention::step) | host | entry.normed, query_path.q_latent, checkpoint compressor/indexer -> window_width, ascending compressed ids, compressed rows | - |
| stage_attention :8770 | host | last window_width host KV rows + seam -> staged window codes/scales, selected list, 147,456 B expanded store (the H2D payload) | - |
| attention_body :8774 | device (one launch) | q, staged window, expanded store, sink, index key -> out buffer bf16, then host inverse-RoPE | - |
| attention_output :8785 (fn :3393) | device wo_a + wo_b + D2H + host B1 :3621 | body activation, pinned wo_a/wo_b -> per-rank wo_b partial bf16[5120] (10,240 B) -> B1 | B1 in, B1 out |
| attention_residual :8798 | host | entry post/comb/hidden, B1 reduced delta (10 KiB) -> updated hc stream f32[4x5120] | consumes B1 |
| ffn_seams :8809 (fn :4019) | host | residual stream, entry.pre, ffn predictor -> ffn pre/post/comb, ffn_in, pre_ffn_hidden f32[5120], act_fp8[5120], act_scales | after B1 |
| moe routed :8812 (fn :4270) | device router/gate_up/weighted/combine + D2H :4544 + host combine + host B2 :4600 | pre_ffn_hidden, act codes, arena expert weights -> slot acc f32[6x5120] -> rank_routed bf16[5120] -> B2 -> delta bf16[5120] | B2 in, B2 out |
| ffn_epilogue :8814 (fn :5986) | device shared GEMVs (4) + host combine | pre_ffn_hidden, shared views (all-gathered once per layer before the position loop, :8404-8421), B2 delta -> ffn_out f32[5120] = routed + shared; hc_post f32[4x5120] = next layer's stream | consumes B2 |
| head :6515 (after all layers) | host collapse/norm, device screens, D2H, host gather | last layer stream, tail_pre_mix -> collapsed row, logits, next token | own head gather, not a layer barrier |

B1 payload is BF16[5120] per rank (10,240 B) and its output is the reduced BF16 attention delta;
B2 payload is the same size and its output is the routed delta. Both are TcpTransport
all_reduce_bf16 calls on the host, so during B1 and B2 the device is idle on all four ranks. The
native-combine arm replaces B2's all-reduce with an all-gather (:4567) but is not the shipped row.
The six shared-expert all-gathers per layer are paid before the position loop, not per position.

## 2. Which seam pairs are genuinely independent

Three pairs, all siblings inside a layer. No pair is "host seam of layer L+1 while device runs
layer L":

1. kv_path (host, 0.067 derived from the m1w37 table) against query_path's wq_b device kernel:
   both read only attention_projections; neither consumes the other.
2. ref_seam (host, 0.501 measured) against query_path's wq_b kernel and q readback: ref_seam needs
   only entry.normed and the input-side q_latent (:3244), which exist before the wq_b launch
   (:3206); it does not read q_post_rope.
3. shared-expert compute (kernel 0.238 + ~0.086 staging, measured) against moe routed + B2: both
   consume ffn_seams; the epilogue only adds them; neither reads the other's result. Today the
   shared kernels are issued inside the epilogue, i.e. after B2 (:6062, :8814).

Everything else is on the chain, and that is the answer to the question as asked:

- mhc_entry and engram_write of layer L+1 need hc_post(L), which needs B2(L). The host seam of
  layer L+1 cannot overlap layer L's device work for a single sequence: layer L's barrier output is
  the next layer's first input.
- attention_residual and ffn_seams need B1; ffn_epilogue needs B2; stage_attention needs the
  current row from kv_path; attention_body needs stage_attention; attention_output needs the body;
  the head needs the last layer's stream.

Overlap is therefore only reachable by hoisting the three siblings, not by pipelining the chain.
Ceiling if all three hide (derived): shared 0.324 always (B2's host reduce is 1.127 ms/layer in the
m1w37 table; the current probe folds it, so this is conservative), ref_seam bounded by the wq_b
kernel, which the probe does not separate (< query_path's 0.621 kernel, likely 0.1-0.2), kv_path
0.067. That is roughly 0.45-0.55 ms/layer, i.e. ~18-22 ms/step.

## 3. What would have to be double-buffered, and what it costs

Only the device scratch a stage writes while the host still reads the other copy: q 16,384 B, window
codes/scales 67,584 B, expanded store 147,456 B, selected 2,048 B, sink 64 B, moe slot acc 122,880 B,
moe stage (gate/up/codes/scales) ~19,000 B, shared rows + pre_ffn_hidden 40,960 B, wo_b partial
10,240 B. About 0.43 MiB per pipeline stage, so a two-deep pipeline costs 0.43-0.86 MiB of device
memory. Host-side double buffering is the hc stream 81,920 B and ~200 KiB of seam vectors per
layer - host RAM.

KV: this loop keeps KV host-side (kv_cache, :8366) and stages only the attended window (67,584 B)
per body launch. If KV is made device-resident for serving, the budget is the kv crate's
890 B/token global: 890 MB per 1M-token context, so a second resident context costs +890 MB
(+114 MB at 128k). That is the only number here that is not noise, and it still fits the
~40 GiB/rank headroom. Memory is not the blocker; the chain is.

## 4. Reachable bound

(a) Arithmetic bound if every independent seam overlaps: per layer max(host_serial, device_serial)
= max(4.32, 1.86) = 4.32 ms, so 40 x 4.32 + 2.05 (head and sampling outside the 40-layer loop) =
~175 ms/step; from the step totals directly, max(249.25 - 71.2, 71.2) = 178 ms/step. The 175/178
difference is the probe's layer_total against the published step (measured, a few ms), not a
correction to chase. The ceiling of the whole pipelining idea is therefore -71 to -74 ms/step
(-28 to -30%), and 178 ms/step is the floor for one token: overlap cannot reach 150 ms/step
(6.7 tok/s) because it only removes device time from the critical path, and there is only 71.2 ms
of device time to remove.

(b) Bound reachable from this graph: only §2's three sibling pairs hide, ~0.45-0.55 ms/layer,
~18-22 ms/step, i.e. ~228-231 ms/step at best. That is the number to weigh against the effort and
the contention risk - not the 175-178 figure.

Corners I cannot quantify from the data we have: (i) the probe's host share is wall minus kernel
and cannot say how much is CPU arithmetic, D2H wait or the TCP all-reduce - if much of the 178 ms
is not CPU, tier (a) is not a CPU floor and the overlap question changes shape; (ii) wq_b's kernel
time is not separated from query_path; (iii) B1/B2 wall is inside attention_output's 0.635 host
share and the moe block, unreported alone, and if the two reduces are >1 ms/layer they are the
largest single host item; (iv) the ILP stall was bimodal, so probe means hide the tail that a
pipelined design would inherit; (v) two-stream capture is untested here - the current segment/graph
path captures one stream.

## 5. The contention risk and the measurement that settles it

The ILP window is a direct measurement that concurrent CPU work slows every kernel on this package
(kernel 10.29 -> 14.94 s, +45%, while the host got faster). Overlap is that experiment by design:
if concurrent host work steals the same bandwidth, the arithmetic win above is reversed. This is
the risk that the pipelining idea is net-negative.

Falsify or confirm it with one four-rank probe window, same binary and staging, three rows:

1. probe-nosync (being built): per seam, split host CPU from device wait, and report the TCP
   reduce separately. Decision: if host CPU over the 40 layers is well below 71.2 ms/step there is
   device time to hide and tier (a) is worth attempting; if it is close to 178 ms, only removing
   host seams can move the step and pipelining cannot reach 150.
2. Pair-3 A/B: shared kernels issued before B2 versus after, nothing else changed. Prediction if
   contention is mild: about -12 ms/step. Falsified if the step does not move or gets worse.
3. Concurrency control: the same kernel chain with a host arithmetic loop running alongside, no
   correctness change, measuring kernel inflation per unit of concurrent host work. Falsified if
   inflation is at least the host time hidden.

Attempt full pipelining only if (1) shows host CPU >= 71 ms/step, (2) shows at least half its
predicted saving, and (3) shows kernel inflation below the hidden host time. Otherwise the honest
lever remains the contention doc's rule 2: remove host work and the bytes it touches, not speed it
up or overlap it.

## 6. Stop point

No code, no window, no rows from this note. If it is picked up, the next artifact is the
probe-nosync row; no pipeline code should be written before it lands.
