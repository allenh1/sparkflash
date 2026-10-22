# DSpark speculative decoding: device-wiring plan

Status: **analysis only**. No engine or kernel change is proposed here that is not gated on a
measurement. Written read-only against `main` (`0174ed0`) plus the design record on
`origin/task/mtp-dspark-design` (`74ad038`, `docs/engine/mtp-dspark-design.md`). No node was
touched, no benchmark was run, and no number below is a new measurement: every figure is either a
computed buffer size (arithmetic, shown) or a citation to an existing document.

Scope: the three MTP layers 40..42, k = 5 draft tokens, the 6-row target verify pass, and what must
exist on the device to run that chain single-stream at TP=4.

## 0. Constraints this plan is written under

- **D39** (`docs/DECISIONS.md:2243`): the chain runs the checkpoint's native arithmetic (E4M3
  activation, 32x32 ue8m0 weights, FP4 routed experts, f32 accumulate). Acceptance is
  **numeric with a declared tolerance against the vLLM oracle**, never token equality. The
  exception is `wo_a`, whose native activation is BF16 (D39 note, task-29).
- **D40** (`docs/DECISIONS.md:2343`): the frozen comparison protocol - vLLM serving the released
  snapshot at TP=4, temperature 0, `prompt_logprobs`, top-1/top-5/top-20 agreement, max/mean
  |delta logprob|, KL - with provisional bars (top-1 >= 95% of positions, median top-20 >= 90%,
  median KL <= 0.02 nat, per-position max <= 1.0 nat). **D41** freezes those thresholds and the
  performance bars; it is recorded as a JSON bars file (`docs/build/d41-floor-computation.md`),
  not as a prose entry, and **has not been computed yet**. So no DSpark serving claim can be made
  before D41 exists.
- **D41 (as designed, not yet frozen)**: no bar may move after a gate run; a changed bar is a new
  decision and a new gate.
- **Bit-exactness against the CPU host twin is retired** as an acceptance criterion for the *device
  chain* (D39 rule 2; D37 superseded). The CPU twin survives in exactly two roles, both used below:
  (a) the **arithmetic oracle for a single kernel or a single stage** where the twin is the same
  arithmetic, and (b) a **stopgap** path that keeps decode correct while the device chain is
  incomplete. `sparkflash-scheduler/src/dspark.rs` and `forward/mtp.rs` are that twin.
- **Weights are resident on the device per rank.** `docs/build/roadmap-to-serving.md:1041-1052`
  gives 138.33 GB/rank with every Engram table resident, against a 130.66 GB device (does not fit
  by 7.67 GB), and **87.6 GB/rank** with the Engram tables on local NVMe. The MTP/DSpark routed
  experts alone are 1.80 GB/rank, 2.01 GB/rank after D30's 576->640 padding, plus 0.21 GB of
  extras (`:1071`), and load **only when speculation is on**.
- **CUDA graphs are planned, with a named fallback list** (section 5, F1..F5). There is no graph
  capture code in the tree today: `cudaGraph`, `cuStreamBeginCapture` and `cuGraphLaunch` do not
  appear under `crates/` (`docs/build/review-2026-10-19-feedback.md:288`).
- **D42** (`docs/DECISIONS.md:2482`): the engine KV seam is **recorded and not implemented**, and
  reviewing it authorises no refactor. DSpark's commit/truncate (design 3.5) needs that seam or a
  host-dense substitute; this plan assumes the host-dense store until D42 lands (open question 7).

## 1. The current CPU DSpark control flow (what actually runs today)

There are **two** implementations, and they disagree about what a draft is. This is the single most
important fact in this document, because the device plan must pick one.

### 1.1 The scheduler's model (`sparkflash-scheduler/src/dspark.rs`)

This is the implementation DECISIONS and the scheduler actually run. It is a **policy model**, not
the model's draft head:

- **Drafts are random tokens, not model output.** `DsparkProposal::draft_tokens`
  (`crates/sparkflash-scheduler/src/dspark.rs:298-312`) calls `SyncFreeDrafter::draft(step, request)`;
  the tokens come from the counter-based philox stream seeded by `(step, request, position)` and no
  rank, so all four ranks draft identically with no broadcast (`:16-18`; test `:471-499`). They are
  not the MTP stages' output - the module says so at `:20-26`.
- **Verification width is chosen by a confidence cap, not by draft-vs-argmax equality.** A position's
  confidence is `sigmoid(z_i)` (D6; `:38-40`, `:245-247`) and `P_survive(j) = prod_{i<=j} p_i`
  (`:250-260`); the verified prefix is the longest `j` in `1..=5` with `P_survive(j) >= 0.10`
  (`DSPARK_CONFIDENCE_CAP`, `crates/sparkflash-scheduler/src/config.rs:53`, read in
  `dspark.rs:265-273`). `verify_rows() = prefix + 1` (`:277-279`).
- **Acceptance rule: there is none.** The prefix is a *scheduling* decision (how many rows to
  forward); nothing here compares a draft to an argmax, and no token is emitted here.
- **Router-live masking:** `RouterMask` (`:56-180`) makes verification row 0 always fresh, rows
  `1..=live_positions` fresh, and every row past the prefix reuse the anchor's `ROUTER_TOP_K = 6`
  experts (`routes_fresh_experts`, `:124-126`). `route()` fails closed: a masked row cannot be given
  fresh experts even if offered (`:146-179`). The mask is exactly the complement of the verified
  prefix (test `:402-424`).
- **Constants:** `DSPARK_BLOCK_SIZE = 5`, `DSPARK_VERIFY_ROWS = 6`, `DSPARK_LAYERS = [37, 38, 39]`
  (`config.rs:46-64`). Note the layers are the **taps**, not the MTP layers 40..42.

### 1.2 The engine's arithmetic (`sparkflash-engine/src/forward/mtp.rs`)

This is the real DSpark forward, in f32 over decoded storage, with **no KV cache**:

- Entry: `InferenceEngine::decode` -> `run` (`crates/sparkflash-engine/src/forward/mod.rs:294-296`,
  `:553`). During the backbone, layers 37/38/39 write a per-position tap into `main_hidden`
  (`:627-649`) as `taps * dim` values per token, taps in tap order.
- After the head, `sample_gumbel` at temperature 0 is `argmax` (`:519-539`); the anchor is that
  token (`:694`) and `mtp::dspark_forward_observed` runs only when `start_pos != 0` (`:690-713`).
  A prefill at position 0 drafts nothing, matching the reference.
- `dspark_forward` (`crates/sparkflash-engine/src/forward/mtp.rs:378-399`) then:
  1. refuses `start_pos == 0` (`:429-434`) and shape-checks stages/taps (`:435-463`);
  2. `main_x = main_norm(main_proj(cat(h37,h38,h39)))` (`:465-490`), `mtp.0.main_proj` being
     5120x15360 = 78,643,200 elements;
  3. builds `ids = [anchor, noise x5]` with `dspark_noise_token_id = 128_799`, embeds and repeats
     each row over `hc_mult = 4` copies (`:492-522`);
  4. runs the 3 stages in order (`:554`), each = mHC mixes, attention that also sees `main_x`
     (`attention::dspark_attention_forward`, `:600-611`), mHC post, MoE (`:656-664`), with no
     allocation inside the position loop (`:528-546`);
  5. head: last stage's `hc_pre` -> `norm` -> LM head over all 5 rows (`:701-729`);
  6. **the Markov chain is a 5-step serial loop** (`:768-796`): for each position, bias =
     `markov_head.head @ embed(ids[position])` added to that row's base logits, then `argmax` ->
     `ids[position+1]`. So the 5-token dependency is injected at sampling time, one cheap GEMV plus
     argmax per position (`:142`, `:788-795`), exactly as the reference does.
  7. confidence = `proj @ [hidden; markov_embed]`, one raw f32 per position (`:797-807`).
- **The result never gates anything.** `StepOutput.draft` / `.confidence`
  (`forward/mod.rs:129-132`) are returned and dropped: the serve loop says "the reference engine
  drafts tokens in its MTP stages and this driver never verifies one, so nothing was proposed,
  accepted or rejected" (`crates/sparkflash-cli/src/serve.rs:1382-1386`, `:1427`), with
  `accepted_prediction_tokens: 0`.

### 1.3 Ring/window sizes today, and the trap in changing them

- Model SWA window: `window_size = 128` in the shipped runtime config
  (`crates/sparkflash-engine/config/deepseek_v41_flash.runtime.json:133`) and
  `SWA_WINDOW_SIZE = 128` in `crates/sparkflash-kv/src/config.rs:26`.
- Engine ring: `WindowRing` holds exactly `config.window_size` rows and addresses slot
  `position % window` (`crates/sparkflash-engine/src/forward/attention.rs:422-449`, `:471`, `:489`).
  The MTP stages get their own rings because `SequenceState` allocates `n_layers + n_mtp_layers`
  states (`forward/mod.rs:61-63`).
- **Do not set `window_size = 134`.** `window_size` is the *attended* window: it sizes the ring, the
  `window_indices` id width (`attention.rs:2538-2543`) and the compressor's view. Raising it
  silently changes the arithmetic to "attend 134 tokens". The ring needs a **capacity** of
  `window + nextn + 1 = 134` while the attended set stays the last 128 rows. That is a new field
  (call it `ring_slots`), not a config edit.
- Why 134 is load-bearing: with 128 slots, writing positions `p..p+5` before the row `p` query
  reads clobbers slots holding `p-127..`, which row `p` still needs. The host multi-row path dodges
  this only because it interleaves write and read per query (`attention.rs:2599-2611`); any
  batched device write-then-attend needs the extra 6 slots.
- Compressor state ring: `ratio + nextn` rows = **7** for ratio 2 and **6** for ratio 1 (design
  1.5). Per the released `compress_ratios`, backbone layers 2..19 are ratio 2, 20..39 are ratio 1,
  and the three MTP layers 40..42 are **ratio 0** - no compressor at all
  (`crates/sparkflash-kv/src/config.rs:81-84`). So the draft layers need an SWA ring and nothing
  else; the 7/6 rings are a verify-pass concern only.

### 1.4 The gap, stated as one sentence

Today the device chain runs **one row per call** through a per-position host loop
(`crates/sparkflash-cli/src/device_chain.rs:2154-2171`), the draft is **random** and never verified
(scheduler) or **real but never verified** (engine), and no path turns a verify result into emitted
tokens. DSpark needs all three to change at once.

## 2. Device buffer inventory

Sizes are arithmetic on the released config (`dim 5120`, `head_dim 512`, `n_heads 64`,
`hc_mult 4` -> `hc_width 20_480`, `o_groups 8`, `o_lora_rank 1024`, `vocab 129_280`,
`index_topk 512`, per-rank vocab 32_320).

### 2.1 Verify pass (6 rows: row 0 anchor, rows 1..5 drafts)

| Buffer | Shape | Bytes | Resident? | Notes |
| --- | --- | ---: | --- | --- |
| hidden stream (`x_hc`) | `[6][hc=4][5120]` bf16 | 245,760 | per step | f32 today (491,520) |
| collapsed / normed row | `[6][5120]` bf16 | 61,440 | per step | head input |
| q rows | `[6][64][512]` bf16 | 393,216 | per step | pre-RoPE wide |
| attention out (wide) | `[6][64][512]` bf16 | 393,216 | per step | before `wo_a` |
| `wo_a` grouped out | `[6][8][1024]` bf16 | 98,304 | per step | `o_groups*o_lora_rank` per row |
| window ring, per layer | `[134][528]` | 70,752 | **persistent** | 512 B codes + 16 B scales per row |
| window ring, 43 layers | 43 x above | 3,042,336 | **persistent** | 2.9 MiB total |
| compressor state ring, ratio 2 | `[7][512]` latent f32 | 14,336 | persistent | 6 rows if ratio 1 |
| compressed pool row | `[n_pos][288]` | - | persistent | FP4 codes + scales |
| selected candidates, per row | `[134+512]` u32 | 2,584 | per step | window ids then absolute positions |
| selected candidates, 6 rows | `[6][646]` u32 | 15,504 | per step | one contiguous list per row |
| per-row positions | `[6]` u32 | 24 | per step | `accepted_pos + i` |
| per-row counts/masks | `[6][3]` u32 | 72 | per step | window_count, selected_count, live flag |
| MoE routing ids/weights | `[6*6]` u32 x2 | 288 | per step | 6 tokens x top-6 |
| routed partials | `[6*6][640]` codes+scales | ~97 KB | per step | 4 rank slices |
| logits, per rank | `[6][32_320]` f32 | 775,680 | per step | **not** read back |
| certified flags, per rank | `[6][32_320]` u8 | 193,920 | per step | 1 = pruned/bound |
| block maxima/argmax | `[6][32]` f32+u32 | 1,536 | per step | head-reduce input |
| accept result | `[8]` u32 | 32 | per step | **the only D2H** |

Draft pass (5 rows, 3 stages, `mtp.{0,1,2}`):

| Buffer | Shape | Bytes | Notes |
| --- | --- | ---: | --- |
| draft stream | `[5][4][5120]` bf16 | 204,800 | slots 1..4 are noise 128,799 |
| draft attention wide | `[5][64][512]` bf16 | 327,680 | |
| draft ring, per stage | `[134][528]` | 70,752 | 3 x above = 212,256 |
| `main_x` per step | `[1][5120]` f32 | 20,480 | one main row per decode step |
| markov bias scratch | `[129_280]` f32 | 517,120 | one per position, reused serially |
| draft ids | `[5]` u32 | 20 | written by the chain kernel |
| markov tables | `[129_280][256]` x2 | 66,191,360 each | weights, resident (132 MB total) |

Weights the draft path needs resident: 1.80 GB/rank of routed experts (2.01 GB after D30 padding)
plus 0.21 GB of extras (`roadmap-to-serving.md:1071`), `main_proj` 78.6 M elements, the two
129,280x256 Markov tables, and the confidence projection **if it is kept** - this plan does not
keep it (design 0).

### 2.2 Launch inventory

Kernels already **row-batched** (the row axis is a grid dimension or a runtime `rows`/`tokens`
argument):

| Kernel | Row mechanism | Source |
| --- | --- | --- |
| `b12x_mxfp8_dense_sm121` (wq_a, wq_b, wkv, wo_b) | grid `(ceil(rows/64), ceil(cols/64), 1)` | `mxfp8_dense_launch.rs:9-12` |
| `moe_router_sm121` | grid `ceil(tokens/8)`, 8 tokens/block | `router_launch.rs:7-9`, `router_abi.rs:41` |
| `routed_gate_up_sm121` | grid `(tokens*top_k, 3)` | `routed_gate_up_launch.rs:9-11` |
| `b12x_mxfp4_moe_sm121` | grid `(tokens, 20, 1)` | `mxfp4_moe_launch.rs:10-13` |
| `rmsnorm_sm121` | grid `ceil(rows/256)` | `rmsnorm_launch.rs:9-12` |
| `swiglu_mxfp8_quant_sm121` / `swiglu_mxfp8_quant_weighted_sm121` | rows and cols from the caller | `moe_device.rs:40-43` |

Kernels that are **1-row-only today** (the call site refuses, or the geometry forbids):

| Kernel | Why | Where |
| --- | --- | --- |
| `flash_mla_csa2_seq_sm121` | grid `(HEADS,1,1)`, one position per launch; the caller requires `seqlen == 1` | `flash_mla_csa2_seq_launch.rs:10-13`, `:46`; `attention_device.rs:913` |
| `b12x_bf16a_mxfp8w_wo_a_sm121` / `b12x_mxfp8_wo_a_sm121` | the *kernel* already has a `groups` grid dim, but the caller hard-refuses `rows != 1` | `attention_device.rs:1789-1794`, `:1875-1880` |
| `cert_lm_head_sm121` | one 1024-entry chunk of one row per block; `hidden` is a single-row pointer | `cert_lm_head_launch.rs:10-13`, `PARAMS[0]` |
| **`mega_mhc_fused_bf16_sm121`** | grid fixed at `(4,1,1)`, `blockIdx.x` = tile, and the 4 tiles rendezvous through `hc_sync`/`hc_stats`; no rows argument | `mega_mhc_launch.rs:78-80`; `kernels/mega_mhc_sm121.cu:235`, `:311` |

`mega_mhc` is the one the brief did not name and it is on the critical path: at 2 mHC passes per
layer it is 80 of the chain's launches per step, and a 6-row verify makes it 480 unless the kernel
takes a row axis (grid `(4, rows, 1)` with `hc_sync` becoming `[rows][2]`, indexed by `blockIdx.y`).

Launches per step, counted from the chain's own accounting (`device_chain.rs:2321` fixes the MoE
block at 21 launches per layer; the projection/attention/mHC counts are the ledger at `:786-788`,
`:2444`):

| Phase | Today (1 row) | 6-row verify, today's kernels | 6-row verify, batched |
| --- | ---: | ---: | ---: |
| per layer: 2 mHC | 2 | 12 | 2 |
| per layer: 3 dense proj | 3 | 3 | 3 |
| per layer: attention | 1 | 6 | 1 |
| per layer: `wo_a` + `wo_b` | 2 | 7 | 2 |
| per layer: MoE block | 21 | 21 | 21 |
| **per layer** | **29** | **49** | **29** |
| **40 layers** | **1,160** | **1,960** | **1,160** |
| draft pass (3 stages x 5 rows) | - | ~105-120 | ~45 |
| head + accept (6 rows) | 5 | 6 x 4 + 6 + 1 = 31 | 4 + 1 + 1 = 6 |
| **per DSpark step** | - | **~2,100** | **~1,210** |

Pure launch overhead is not measured for this chain, so treat the count as a budget, not a
prediction. At 5-10 us per kernel the batched 1,210 launches are 6-12 ms of a 40 ms step (15-30%);
the roadmap's own chained-launch figure is 0.2 ms per four-launch chain step
(`docs/build/roadmap-to-serving.md:862-864`), and at that rate an uncaptured 6-row step would be
~240 ms of launch time alone. Either way graphs are not optional, which is what C5 is for.

Three call-site contracts block a 6-row batch and are **not** just geometry:

1. `attention_device.rs:913` requires `seqlen == 1`, and `:938-942` **infers the compression ratio
   from the candidate count** (`compressed_count > start_pos.div_ceil(2)` -> ratio 1). For 6 rows at
   positions `p..p+5` that heuristic returns the wrong ratio for at least one row. The ratio must
   become an explicit argument.
2. `combined` (the candidate list) is built on the **host**, per layer, by the HSI indexer and
   published per `(start_pos, seqlen)` in a thread-local (`attention.rs:1555-1647`, `:2503-2513`).
   Six rows means 6 x 646 ids per layer, and it means the indexer's top-512 runs for 6 queries.
3. The kernel's union guard refuses `window_rows > 128` **and** `window_rows + selected_count >
   640`, and on refusal it writes **NaN** into `out` and returns
   (`kernels/flash_mla_csa2_seq_sm121.cu:212-221`, with `static_assert(MAX_ATTENDED == 640)` at
   `:48` and `static_assert(SWA_WINDOW == 128)` at `:53`). So a 134-row ring cannot quietly work on
   today's cubin: it produces NaN logits. Widening needs `MAX_ATTENDED >= 646` and a **new symbol**
   (D42's rule: the frozen entry is not mutated; a changed static assert is a decision-record pin
   change, baseline 18 -> 19 for the ring symbol).

## 3. Acceptance counting and the readback floor

### 3.1 The rule

```
a = 0
while a < 5 and draft[a] == argmax(logits row a): a += 1
emit = draft[..a] ++ [argmax(logits row a)]           # a + 1 tokens
commit(state, a + 1)
```

Row `r`'s logits predict the token after `accepted_pos + r`, so row 0 predicts draft 0 and row `a`
predicts the bonus token (design 1.4, 3.5). The argmax must be **exact**. The certified head's
certificate proves exactly that: a pruned entry satisfies `bound < running_max <= final max`, so the
argmax id is always exact even though a pruned entry's *value* is only a bound
(`kernels/cert_lm_head_sm121.cu:1-26`; `certified == 1` means pruned, `:63-65`). Consequence, and it
matters for section 5: **greedy accept needs ids only and is sound**; a later temperature>0 residual
rule needs the *exact* logit at the draft id, which the certified head may have stored as a bound,
so sampling would need one exact re-dot per row.

### 3.2 What must happen on the device

1. **Draft kernel** writes `draft_ids[5]` (and, for debugging only, keeps `draft_logits` on device)
   into a proposal buffer the verify pass reads. No host copy of the proposal is needed: the verify
   rows are written by the same path (`rows[0] = last_emitted`, `rows[1..6] = draft_ids`).
2. **Head, 6 rows**: `cert_lm_head` batched over rows (grid `(ceil(per_rank/1024), rows)`), then
   `lm_head_reduce` batched over rows, producing `row_token[6]`, `row_value[6]`, `row_status[6]`.
   `row_status` is not optional: `STATUS_WINNER_CERTIFIED` must be checked and a set bit must
   refuse the step rather than accept it (`lm_head_reduce_launch.rs:44-56`).
3. **Accept kernel** (new, ~40 lines, one block): reads `draft_ids[5]`, `row_token[6]`,
   `row_status[6]`; computes the prefix `a` (first mismatch); writes `emit[0..a+1]` (`draft[..a]`,
   then `row_token[a]`), `emit_n = a+1`, `next_anchor = emit[a]`, `status = OR(row_status)`. It also
   writes the next step's `rows[0]`, so the chain is self-feeding.
4. **Commit/truncate** for the rings: reset slot `position % 134` for positions
   `len + a + 1 .. len + 6` and rewind the compressor state ring to the committed length. A second
   small kernel (6 threads) or a memset of the 6 slots - and the piece D42's missing KV seam makes
   non-trivial (open question 7).

### 3.3 The minimal readback

One D2H of `[8]` u32 (32 B) after the accept kernel, outside any graph capture: `emit_n`,
`emit[0..6]`, `status`. The host learns how many tokens to detokenize and what they are - exactly
what a serving driver needs. Nothing else in the decode loop reads back.

For scale, today's chain stages **32.7 MB D2H per position** and 67.7 MB H2D per position: the
recorded run reports `host_bytes=4,515,856,824` and `staged_expert_bytes=9,349,393,600` over 138
positions (`docs/build/device-forward-chain.md:1138`, `:1388`). DSpark's 32 B is a 10^6 reduction,
and that is the point: the accept path must not inherit the per-layer readback habit
(`moe_device.rs:612-628` router ids; `attention_device.rs:1838-1848` `wo_a` output;
`moe_device.rs:1612-1713` head logits plus four rank readbacks per row).

### 3.4 TP=4 and the accept decision

The accept count must be identical on all four ranks or the ranks diverge on the next step. The
per-rank head produces only that rank's 32,320-entry argmax, so a rank-local accept is wrong.

- **(a) rank-0 accept + broadcast**: rank 0 runs the reduce and the accept kernel, then broadcasts
  the 32 B result. One small collective per step. Requires the reduce to be rank-0-only (today
  every rank runs the four-shard reduce in one process, `moe_device.rs:1596-1683`).
- **(b) replicate the reduce on every rank**: every rank reduces all four shards' logits, which
  needs all four shards per rank (a 2 MB/step gather). Rejected.

Plan: (a). The broadcast is the one unavoidable collective; it must be a 7-word all-reduce or a
rank-0 broadcast, never a logits gather. This changes the TP seam shape (`Tp4Step` already ignores
worker tokens, `serve.rs:2515-2540`), so it is a driver change, not an engine one.

## 4. Staged plan: 6 commits, one gate each

Ordering principle: the row axis and the rings are prerequisites for *every* later commit and are
also the measurement that can cancel the line (design 6.1), so they come first and are gated before
any draft token is written. The tok/s gate is last and is stated as an outcome, not an assumption.

**Budget arithmetic** (`a = 3.2` accepted of 5, so `1 + a = 4.2` tokens per step):

| target | 6-row step budget | implied 1-row budget at 1.5x | vs today (~939 ms/step, 1.06 tok/s) |
| --- | ---: | ---: | ---: |
| 50 tok/s | 84 ms | 56 ms | 17x |
| 60 tok/s | 70 ms | 47 ms | 20x |
| 90 tok/s (stretch) | 47 ms | 31 ms | 30x |

DSpark's 2.2-2.5x multiplier is necessary and **not sufficient**: at 939 ms/step it yields ~2.4
tok/s. The plan therefore carries the step-cost legs (C2, C5) explicitly; if C2's gate shows the
1-row step is not on track for ~56 ms, the 50 tok/s gate in C6 is declared unreachable instead of
being claimed.

### C1 - Ring capacity and the 6-row engine entry (prerequisite, no draft yet)

- Add `ring_slots = window_size + dspark_block_size + 1 = 134` as a field separate from
  `window_size`; slot = `position % ring_slots`, attended set still the last 128 positions.
  Compressor state ring `ratio + nextn` (7 / 6). Planner reservation `nextn + 1 = 6` rows.
- `decode_rows(&mut SequenceState, rows: &[u32]) -> Vec<StepOutput>` (design 3.4) over the existing
  chain, per-row logits, with a `written_rows` counter beside `tokens` (design 3.5 - `SequenceState`
  has one counter today, `forward/mod.rs:50-53`).
- **Gate G1**: (a) ring wrap fixtures at positions 127/128/133/134 and compressor partial groups
  over 1..6 rows; (b) **row-count invariance** - 1 row at position L vs 6 rows at L..L+5 fed the
  golden tokens, row 0 bit-identical then row 1 and so on, for L in {37, 129, 517, 4097} and three
  prompts; (c) a device-boundary check that the current cubin writes NaN for `window_rows = 134`
  (it must, `kernels/flash_mla_csa2_seq_sm121.cu:212-221`), so nobody believes the ring is wired.

### C2 - Batched kernels and the device 6-row verify

- New symbols, no mutation of frozen entries (D42): `mega_mhc_rows` (grid `(4, rows, 1)`, per-row
  `hc_sync`), `flash_mla_csa2_rows` (`MAX_ATTENDED` >= 646, per-row `position` and candidate list,
  explicit `ratio`), `cert_lm_head_rows` (grid `(chunks, rows)`), `lm_head_reduce_rows`. `wo_a`
  needs only its caller relaxed (`rows` -> `groups = rows`) plus 6-row activation/out buffers.
- Baseline pin 18 -> 19: one new `KernelId` (`FlashMlaCsa2Ring`) with its sha256; the three
  batched row symbols were landed inside the existing `mega_mhc_sm121`, `cert_lm_head_sm121` and
  `lm_head_reduce_sm121` cubins (re-pinned, no new KernelId).
- **Gate G2 (the cancel-the-line gate, design 6.1)**: same weights, same prompt, same TP=4 layout,
  10 warmup + 60 measured steps, trimmed mean `step_ms`; **6-row step <= 1.5 x 1-row step**, both
  shapes in the same Lead-scheduled window. Secondary: the 6-row step's per-kernel twin bars
  (attention 2e-2; head and GEMMs exact-codes) so a "fast because wrong" result cannot pass.

### C3 - Draft-layer KV write and the device draft pass

- Inside the main forward, at layers 37/38/39: write the draft-layer main K at
  `slot = position % 134`. The reference's `main_k_proj` (design 3.2) is the engine's own
  `wkv(main_x)` inside `attention::dspark_attention_forward` (`attention.rs:2877-2885`); that path
  has no KV cache today, so the device version needs the ring plus the slot map.
- 3 stages x 5 rows on device; `dflash_kv_index` built on device (133 valid rows: last 128 main plus
  up to 5 intra-block, `-1` masked, design 1.3); the Markov chain as one kernel with a 5-iteration
  internal loop (the dependency is serial and there is no grid-wide sync per iteration, so one block
  of 256 threads scanning 129,280 entries five times; five launches are the fallback).
- **Gate G3**: 5 draft ids equal the CPU twin `dspark_forward` on the same taps for 200 forced
  steps, 0 mismatches - the one place the CPU twin is a legitimate oracle, because it is the same
  arithmetic (f32 over decoded weights, design 1.1). Native-vs-twin gaps are recorded, not asserted
  (D39).

### C4 - Accept, commit, emit

- `dspark_accept` plus the ring truncate kernel (3.2, 3.4 above).
- The chain becomes self-feeding: verify rows written on device from `draft_ids`, accept count and
  emitted ids written on device, one 32 B D2H.
- **Gate G4**: (a) forced drafts with `a = 0,1,2,3,4,5` each produce exactly the plain-greedy token
  stream over 2,000 tokens on the CPU twin; (b) device emit equals host emit for 200 steps; (c) D2H
  per step <= 32 B measured, not asserted; (d) `STATUS_WINNER_CERTIFIED` forced (by a mutated screen
  in a test build) must refuse the step, not accept it.

### C5 - Graphs and the readback-free loop

- Capture the 6-row frame (verify + accept); the draft pass is optional. Fallbacks F1..F5 in
  section 5.
- **Gate G5**: 10,000 graph replays byte-identical on every boundary buffer; each fallback reachable
  by a flag; the measured `step_ms` enters the C6 budget; and the captured-vs-uncaptured delta is
  recorded, because it is the number that says whether graphs were the lever or the launch count was
  already fine.

### C6 - TP=4 end-to-end, DSpark on, single stream

- Real checkpoint, four ranks, DSpark on: acceptance statistics (accepted-of-5 distribution,
  per-position accept rates, `avg_verified_len`), decode ms, and the D40 comparison against the vLLM
  oracle.
- **Gate G6**: **>= 50 tok/s single-stream** (stretch 90) with `a >= 3.0` of 5, zero divergence from
  the plain-greedy stream when drafts are forced, and the D40 bars (top-1 >= 95%, median KL <=
  0.02 nat) once D41 exists. If D41 does not exist at this point the run is labelled diagnostic and
  the serving claim is withheld (D40, `:2350-2354`).

## 5. Risks and the concrete check for each

**R1 - Row-count invariance (numerical).** The batch-17 class of bug (a reduction's order depends on
the row count) is live in this tree and the design calls it the highest risk. *Check*: G1(b) at four
prefix lengths and three prompts, **plus** a per-kernel twin at 1 vs 6 rows for every batched kernel
in C2 (dense, router, gate_up, mxfp4 MoE, mHC, head). *Fallback*: pad every forward to a fixed 6-row
floor - cheap at these step times - rather than shipping MTP on unproven invariance.

**R2 - Ring state (134 vs 128, eviction order, ratio inference).** Three distinct failure modes:
(a) 128 slots evict rows a multi-row batch still needs (section 1.3); (b) the kernel refuses
`window_rows > 128` by writing **NaN**, loud only if the NaN is detected before it reaches the accept
decision; (c) the compression ratio is *inferred* from the candidate count
(`attention_device.rs:938-942`), which is wrong for any row after the first in a 6-row batch.
*Check*: G1(a) wrap fixtures; a device-vs-host byte compare of the 134 ring rows at positions
126..140; an explicit-ratio unit test that fails on the current inference; and a NaN guard on the
accept kernel's input (`row_value` non-finite -> refuse).

**R3 - Masking (`-1` vs `NO_INDEX`, causality, router-live).** The reference's invalid marker is
`-1`; the engine uses `NO_INDEX = usize::MAX` (`attention.rs:72`); the kernel masks by
`candidate <= position` and writes NaN on a malformed union. A draft row that sees a future row
instead of masking it is silent, not loud. *Check*: (a) a 6-row chunk where every draft is rejected
(`a = 0`) must still emit exactly one token; (b) a no-future-id test at positions 0..5 in one chunk,
compared against the host loop; (c) the draft `kv_index` boundary test at `accepted_pos - 127` (the
oldest valid row) and at 133/134; (d) the router-live mask must be *retired* for the device path
(design 0), so a test asserts it is not applied.

**R4 - Determinism (and which drafter is authoritative).** Today's scheduler drafts are philox
random (1.1) - deterministic, but not model output; the engine's drafts are greedy argmax (1.2).
Both are called deterministic and they disagree. *Check*: one input, two runs, all 6 rows
bit-identical; 10,000 graph replays (G5); accept counts identical on all four ranks for 1,000 steps;
and a tie-break test (`consider_head_top`, `moe_device.rs:2082-2097`), because a tie in the head or
in the top-5 changes the emitted token.

**R5 - Graph capture incompatibility.** Five blockers, each already visible in the code:

1. `mega_mhc` requires a **host H2D zeroing of `hc_sync` and `hc_stats` before every launch**
   (`chain_device.rs:456-478`) - two copies per mHC launch, 160 per step, not capturable as-is.
2. Per-call allocations (`self.cuda.alloc` throughout `moe_device.rs` and `attention_device.rs`) are
   illegal inside capture.
3. `position`, `layer_id`, `step_seq` are **kernel arguments** (`flash_mla_csa2_seq_launch.rs`
   `PARAMS`; `cert_lm_head_launch.rs` `PARAMS[9..11]`), so replay cannot advance them unless they
   move to device memory.
4. D2H readbacks inside the loop would sync a captured graph; the accept readback must stay outside
   (3.3).
5. The mHC rendezvous is a **spin-wait with a deadline** (`mega_mhc_sm121.cu:311`): legal in a
   graph, but it turns a capture bug into a hang rather than an error.

*Check*: capture one 6-row frame, replay three times, byte-compare every boundary buffer against a
non-graph run; then 10,000 replays (G5). *Named fallbacks*: **F1** mHC stays outside the graph
(per-row or per-row-group launches around a captured remainder); **F2** replace the H2D reset with a
device memset kernel (one block, 8 bytes); **F3** device-resident dynamic scalars
(`position`/`layer`/`step` in a pinned buffer, updated by a 16 B H2D before replay); **F4** the draft
pass is not captured (45-120 launches, small); **F5** no graph at all, batched launches only - and if
F5 misses the tok/s gate, the gate is reported as unmet, not redefined.

**R6 - Checkpoint and loader coverage.** The `mtp.{0,1,2}` tensors are in the plan
(`crates/sparkflash-engine/src/tensor.rs`), but `main_proj`, `kv_norm`, the two Markov tables and the
confidence projection must be added under the same f32/bf16 rules, and the f32 reference pass
**cannot** run at released size: it refuses `mtp.0.main_proj.weight` at 78,643,200 elements
(`mtp.rs:1076-1101`; `REFERENCE_TENSOR_LIMIT` 33,554,432). So the "CPU twin oracle" of G3 is only
available at tiny or synthetic scale unless the twin learns to page the tensor - a real cost in C3,
not a footnote.

## 6. Open questions for the Lead

1. **Which step time is the tok/s baseline?** The design doc plans on ~939 ms/step (1.06 tok/s;
   `docs/investigations/deepinfer/architecture-and-runtime.md:284`, `:436`), while the TP=4 resident
   host A/B measures **7.929 s/step** (`docs/build/allocator-tuning.md:58-60`). Both are real, for
   different paths. The 50-60 tok/s target needs a 1-row step of ~56 ms; nothing in the tree is
   within 17x of that. **Is DSpark expected to land after the step-cost work (graphs, device-resident
   rings, readback removal), or is it expected to carry the tok/s number itself?** The plan above
   assumes the former and says so in C2 and C6.
2. **What exactly is K8?** The brief says the batched variants of `wo_a`, `flash_mla_csa2_seq` and
   `cert_lm_head` are being written as K8. Two more kernels are equally 1-row-only and on the
   critical path: `mega_mhc_fused_bf16_sm121` (grid `(4,1,1)`, cross-block rendezvous) and
   `lm_head_reduce_sm121`. **Is mHC row-batching inside K8's scope, and if not, which commit owns
   it?** Without it a 6-row verify costs 480 extra launches per step.
3. **Who signs the pin change for a 134-row ring?** `MAX_ATTENDED == 640` and `SWA_WINDOW == 128`
   are `static_assert`s in the frozen entry, and the kernel writes NaN when `window_rows > 128`.
   Per D42 the frozen symbol is not mutated, so this needs a new symbol, a new `KernelId`, baseline
   18 -> 19, and a decision record. **Is that pre-authorised, or does it need its own D-entry before
   C2 starts?**
4. **Is the philox drafter retired?** `dspark.rs` drafts from philox and gates verification on
   `DSPARK_CONFIDENCE_CAP = 0.10`; the device design has no confidence head, no cap, and drafts from
   the three MTP stages (design 0, 3.1). **Confirm `DsparkProposal` / `RouterMask` are kept only as a
   scheduler-policy simulation and are not the device contract** - otherwise we build two
   incompatible accept paths.
5. **Accept broadcast on TP=4.** May the driver add one 7-word rank-0 broadcast per step (option (a)
   in 3.4)? The alternative - every rank reducing all four shards - is a 2 MB/step gather. This is
   the one new collective DSpark needs.
6. **Residency budget.** `roadmap-to-serving.md:1047` gives 138.33 GB/rank (does not fit 130.66 GB)
   and 87.6 GB/rank with Engram on NVMe. The brief's **71.4 GiB/rank** does not decompose to either
   figure that I can find. **Which budget do the persistent DSpark buffers come from?** The rings are
   only ~3.0 MiB and the draft block ~0.75 MiB, so this is bookkeeping, not risk - but the number in
   the brief should be reconciled before it appears in an evidence file.
7. **Does DSpark wait for D42 (the KV seam)?** Commit/truncate needs a store that can drop unverified
   rows; D42 records Option B and approves no implementation. **Proceed on the host-dense store
   (rings staged per layer, expensive but correct) until D42 lands, or make D42 a hard
   prerequisite?** This is the one dependency that can reorder C1 and C4.

## 7. What this document did not do

- No benchmark, no node, no GPU: every byte count is arithmetic from the released config, and every
  launch count is counted from the chain's own ledger, not measured.
- No claim that the current device chain runs a 6-row step at all: `attention_device.rs:913` and
  `attention.rs:2595-2597` mean rows 1..5 of a 6-row chunk fall back to the host loop today.
- No D41 bars exist yet, so no serving-parity claim is made or implied anywhere above.

## 8. Sources read

`crates/sparkflash-scheduler/src/dspark.rs`, `.../src/config.rs`;
`crates/sparkflash-engine/src/forward/mtp.rs`, `.../forward/mod.rs`, `.../forward/attention.rs`,
`.../forward/moe.rs`, `.../src/config.rs`, `.../config/deepseek_v41_flash.runtime.json`;
`crates/sparkflash-cli/src/device_chain.rs`, `.../src/attention_device.rs`, `.../src/moe_device.rs`,
`.../src/serve.rs`; `crates/sparkflash-kernels/src/*_launch.rs`, `kernels/*.cu`,
`crates/sparkflash-kv/src/config.rs`, `crates/sparkflash-device/src/chain.rs`,
`.../src/chain_device.rs`; `docs/engine/mtp-dspark-design.md` (branch), `docs/DECISIONS.md`
(D37-D43), `docs/build/roadmap-to-serving.md`, `docs/build/device-forward-chain.md`,
`docs/build/serving-readiness.md`, `docs/build/allocator-tuning.md`,
`docs/build/review-2026-10-19-feedback.md`, `docs/investigations/deepinfer/*`.

Note for the record: `docs/DECISIONS.md` has **D37 through D43**; D44-D48 do not exist, and D41 is
recorded as a JSON bars file that has not been computed.

## Review note (2026-10-20, merge-dev, docs/batch-19)

This document was written against `main` `0174ed0`; **D44-D48 have since landed on `main`**
(`24cf032`), so the "D44-D48 do not exist" note above is historical. The open questions are
answered as follows: Q1 (step-time baseline) by the D48 addendum - 767.7 ms/step = 1.3026
tokens/s measured baseline, DSpark after C2/C5; Q3 (pin change) pre-authorised by D49 and landed as the ring symbol (new
symbol, pin 18 -> 19, explicit ratio, NaN guard; the three batched row symbols ride in existing
cubins); Q4 (drafter) by D50 (MTP drafter wins,
`dspark.rs` is scheduler policy only); Q6 (residency budget) by D45/D48 geometry - 285.504 GiB
world=1 / 71.376 GiB/rank world=4, with the ~0.9 GiB difference being replicated classes counted
per rank, and `roadmap-to-serving.md`'s 87.6 GB/rank being the Engram-on-NVMe variant.

**Q2 (mHC row-batching scope): answered - IN SCOPE and already done.** The batched
`mega_mhc_fused_bf16_batched` (grid `(4, 6)`, per-row rendezvous, device-side init kernel so the
pair is CUDA-graph-capturable) and `lm_head_reduce_batched` are implemented and staged (kernel-dev
`68a1fa3`, harnesses m23/m24), **unmeasured**. mHC batching is a **graph-capture enabler**, not
only a launch-count win: the ~480 mHC launches/step figure drops accordingly (batched: 24 blocks
in one launch), and C5's capture risk
F1 loses its main blocker.

**Q7 (does DSpark wait for D42?): answered - NO.** D42 is not a hard prerequisite and C1/C4 must
not be reordered. The verify/commit-truncate path needs to hold unverified rows for one step, which
the **host-dense per-layer rings** already do; the paged store is orthogonal to the attention
arithmetic, and making it a prerequisite would delay the device arm behind an unapproved refactor.
The real prerequisite is **D49** (ring capacity 134 with window 128). Constraint to record: until
D42 lands, **no paged-KV/serving-cache claim may be attached to DSpark**, and the commit path must
be labelled **host-dense rings**.
