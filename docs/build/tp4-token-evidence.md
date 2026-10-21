# The four-rank token evidence (`sparkflash-tp4-token/1`)

Status: the multi-position four-rank loop is wired (commit 893ef1a3) and the prefill body is no
longer a smoke test: every forward position runs all 40 layers with the reference compressor,
indexer and candidate seams and the pinned flash_mla_csa2_seq_sm121 CSA-union attention body, the
inverse RoPE before wo_a, and the certified head tail on every head position. The remaining work is
window time: the smoke window (--prefill-limit N) and then the full acceptance window (no limit).
This document is the schema contract qa-lead's scripts/accept_device_token.py consumes and the argv
the next window calls. It must not be read as a token claim: no token exists until a four-rank
window writes a forward-run envelope.

## Files per rank directory

| file | writer | contents |
| --- | --- | --- |
| `tp4-token-r<rank>.json` | `crates/sparkflash-cli/src/tp4_token.rs` | the envelope below |
| `tp4-logits-r<rank>.jsonl` | rank 0 only, via the frozen D40 writer (`logits_dump.rs`) | one JSON object per position: the 11 frozen fields |
| `tp4-logits-r<rank>.jsonl.bin` | rank 0 only, same writer | full-vocabulary little-endian f32 blocks, `129_280 * 4` bytes per position |

Since 893ef1a3 every rank writes its own dump from the same four-rank consensus logits
(tp4-logits-r{rank}.jsonl). This is accepted by scripts/accept_device_token.py: a replica rank may
omit the dump, and a replica that publishes one is compared by logits_dump.bin_sha256 across ranks
(the sidecars are byte-identical here because they carry the same consensus f32 values in the same
order); only the JSONL bytes may differ, because the logits_bin field names the per-rank sidecar,
and the script treats that as informational. The rank-0 dump remains the primary artefact. The
envelope/dump writer validates its own shape before writing; a malformed envelope fails on the node
that wrote it.

## Envelope fields

```json
{
  "schema": "sparkflash-tp4-token/1",
  "stage": "tp4-chain",
  "loop_step": "C4-token",
  "rank": 0,
  "world": 4,
  "forward": "run | refused",
  "tokens": {
    "ids": [5],
    "positions": [129],
    "start_position": 129,
    "prompt_len": 130,
    "argmax_logits": [12.25],
    "top1_margins": [0.5],
    "logits_sha256": "… or null"
  },
  "positions_run": 1,
  "step_ms": { "samples": [41.0], "mean_ms": 41.0, "worst_ms": 41.0 },
  "wall_ms": [41.0],
  "reduce_calls_per_step": 80,
  "head": {
    "rule": "rank-sharded head rows all-gathered over tcp; every rank argmaxes the full vector; rank0 dumps full logits",
    "vocab": 129280,
    "tie_break": "lowest token id"
  },
  "logits_dump": {
    "path": "/…/tp4-logits-r0.jsonl",
    "bin": "/…/tp4-logits-r0.jsonl.bin",
    "sha256": "…64 hex",
    "bin_sha256": "…64 hex",
    "dtype": "f32-le",
    "vocab": 129280,
    "positions": [129],
    "rows": 1
  },
  "error": null,
  "launch_census": { "routed_experts_down": true, "…": "combine selector evidence row" },
  "hybrid_sequence_executed": false
}
```

Rules the writer enforces (`TokenEnvelope::validate`):

* `forward:"run"`: non-empty `tokens.ids`; `positions_run == len(ids) == len(positions)`;
  positions strictly increasing from `start_position < prompt_len`; one argmax logit and top-1
  margin per id; a non-null `logits_dump` whose rows and positions equal the token block and whose
  `bin` is exactly `path + ".bin"` with 64-hex hashes; `reduce_calls_per_step == 80`
  (2 reduces x 40 layers); `launch_census.routed_experts_down == true`; `error == null`;
  `len(wall_ms) == len(step_ms.samples)`.
* `forward:"refused"`: no ids, no positions, `positions_run == 0`, `start_position` and
  `prompt_len` null, `logits_dump` null, a non-empty `error`, `routed_experts_down == false`.
  A refusal is never a partial token claim.

## Offline consumer dry run (host, synthetic - not a token claim)

On 2026-10-20 the schema was checked against qa-lead's published consumer
(`scripts/accept_device_token.py` at `dd3ca60`) with a synthetic fixture written by the Rust
writer: four rank envelopes with identical consensus ids, rank 0's dump, replicas without a dump.
The consumer reported `shape=PASS` with these checks green: 4 device evidence files, all layouts
recognised, unique ranks 0-3, world 4, token ids present from `tokens.ids`, token range, token
evidence on every rank, all four envelopes consistent, token logits dump present (auto-resolved
from the rank-0 envelope, no `--device-dump`), 4 ranks rank-consistent, 8 step samples, dump/sidecar
tile, all rows vocab 129280, pinned positions present. Its exit code 3 / overall `UNMEASURED`
reflects the absent oracle and absent pre-token evidence, exactly the labelled outcome expected for
a token-only fixture. The fixture lives in `tp4_token.rs`'s `consumer_fixture_tests` and is
re-run by `cargo test`; it makes no token claim.

## The 11 frozen dump fields

`crates/sparkflash-cli/src/logits_dump.rs` is reused as-is (one implementation of the artefact):
`position, device_argmax, logits_argmax, reference_id, hybrid_id, fixture_id, exact_token_equal,
vocab, logits_offset, logits_bin, top20`. The token stage fills `device_argmax = hybrid_id =`
the consensus token, reference_id/fixture_id = null, and logits_offset tiling by 129,280 f32
per position. exact_token_equal is the dump schema's own row verdict - reference_id ==
hybrid_id (docs/build/logits-dump.md) - so a replay row with no reference vector reads false
and a teacher-forced row reads the per-row match. The frozen consumer check
`basename(row.logits_bin) == basename(dump_path) + ".bin"` holds for rank 0's dump.

## Where the preflight runs (and why it is cheap)

The chain runs the token preflight *before* the arena boot. With --forward the order is:

1. parse argv and build the rank plan (host only);
2. build the request from the plan (combine row geometry from the first layer w2.weight/w2.scale
   shard headers, the same headers the bind checks);
3. run the combine layout selector on the host: the interlock flag, the geometry, the staged cubin
   directory and the pinned stored cubin (size 11,544 bytes, sha256 c452a9aa...); a cubin that is not
   the measured artefact is refused by name before any driver sees it;
4. only if that passes: stream the 72 GiB arena, bind the device sets, and run the forward loop
   inside the same arena epoch (the loop's pointers die with it, so bind-and-run is one call);
5. write tp4-loop-rR.json and then the tp4-token-rR.json envelope.

A refusal at step 3 fails the rank in seconds and writes only the token envelope: there is no
arena, no bind file and no loop file for that attempt. Run the pre-token argv (same flags without
--forward) for the bind/walk evidence; the token attempt is purely additive. --forward with
--plan-only is refused with a named reason because the loop has no arena to read.

## When the stage refuses (and why it must)

`--forward` alone never launches. `crates/sparkflash-cli/src/tp4_forward.rs` runs the host-only
preflight and refuses with a named error before any kernel:

1. **Interlock absent** — `--forward` without `--forward-preflight`:
   `forward preflight flag absent: pass --forward-preflight …`.
2. **Combine cubin absent** — the staged kernel directory has no
   `b12x_mxfp4_moe_stored_sm121.cubin`:
   `stored-layout combine kernel unavailable: b12x_mxfp4_moe_stored_sm121.cubin is not staged in
   <dir> (task-6 kernel). Refusing the four-rank combine rather than launching
   b12x_mxfp4_moe_sm121 on stored 576-K rows, which would misindex every row after the first.`
3. **Task-6 ABI not wired** — the cubin is staged but `sparkflash-kernels` has no loader/params for
   it yet: `stored-layout combine ABI not wired: … (task-6 pending)`. This is the state of this
   build; the preflight's expected verdict on the next window is this refusal.
4. **Bind/selector arithmetic disagree** — the w2 arena slice bytes infer to neither the stored
   `288+18` nor the pre-added `320+20` row layout.
5. **No arena** (`--plan-only`), or no `--token-ids`.

Every refusal writes the envelope on every rank and makes the rank exit nonzero: the evidence
survives the failure, and no run envelope is fabricated.

## Next-window argv (per rank)

The canonical generator is scripts/tp4-window-bind.sh. The bind/walk argv is unchanged:

    scripts/tp4-window-bind.sh --print-argv <R>          # R in 0..3

and the token argv is the same line plus the forward-only flags:

    TOKEN_IDS=<prompt ids> KERNEL_CUBIN_DIR=<staged cubins> \
      scripts/tp4-window-bind.sh --print-argv-token <R>

which prints, for rank R (all on one line; REMOTE_BIN is the Lead's staged binary):

    <REMOTE_BIN> --run-stage tp4-chain \
      --checkpoint <released checkpoint> \
      --tp-rank <R> --tp-world-size 4 \
      --evidence-dir <EVIDENCE_ROOT>/<RUN_NAME>/rank<R> \
      --upload --loop \
      --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 \
      --comm-base-port 29601 --comm-timeout 120 --comm-connect-timeout 120 \
      --warmup-steps 2 --decode-steps 9 \
      --forward --forward-preflight \
      --kernel-cubin-dir <staged cubins> \
      --token-ids <prompt ids>

--token-ids is the prompt prefix; the forwarded positions are prompt_len .. prompt_len+decode_steps-1,
so a 129-token prefix with --decode-steps 9 covers positions 129..137 (the pinned positions the
acceptance runner expects). --kernel-cubin-dir is NOT the older --cubin-dir: the latter invokes the
world=1 hybrid diagnostic and must never be passed in four-rank mode.

The teacher-forced reference vector is wired: --forward-reference-ids carries one id per dumped row
(the fixture's nine ids are [491,367,201,80,3302,201,54,201,4341]) and every dump row records it as
reference_id, because QA counts a null reference_id as a prefix mismatch. The chain refuses a vector
whose length differs from --decode-steps, and the generator accepts REFERENCE_IDS to append the flag.
With 138 fed ids and --decode-steps 9 the rows cover positions 129..137 (prompt_len 130 = 138 - 9 + 1;
the 4341 reference is position 137's label, never fed).

The loop writes an attention/loop stage census into the refused-embedding evidence and into the named
error: for each layer-0 projection it reports the plan's own operand layout (the device crate's
ResidentPlan::operand_layout predicate, the same one the arena upload and bind use: F8_E4M3 in a tiled
group means B12x with its tile bytes and scale-grid bytes, anything else means stored), the bind pair
it observes, which kernel family would read it, the B12x tile geometry and byte counts, and whether
that launch is possible. sparkflash_device::resident::check_operand_bind then names a disagreement
(B12x planned but a null or shared pointer, stored planned but a distinct scale pointer) as a refusal
carrying the tensor name instead of launching it. kernel-lead's host gate measured the released
headers (96,085 tensors, plan hash 0x9a59035fc40776c8): all 215 attention projection weights across 43
layers are plan-B12x F8_E4M3 with F8_E8M0 scales, 0 stored, and the m1-bind arena totals reproduce
exactly (attention-replicated 258 tensors/86 tiled/395,066,112 B, column 172/86/812,390,400 B, row
86/43/451,328,000 B), so the five layer-0 projections are B12x-launchable. My earlier note that
wq_a/wkv were stored BF16 was an unverified inference from a plan summary, not measured evidence, and
task-10's repack premise is retracted. The census is evidence, not a token: no kernel is launched by
it and no token can be claimed from it.

The layer-0 mHC entry now runs on the host seam from the live arena: the loop reads the released
24-row hc_attn_fn, hc_attn_base, hc_attn_scale and the attn_norm weight (F32 or BF16 per the plan
dtype) back from the arena, computes hc_mixes over the four residual copies, applies the layer-0 seed
A=[1,0,0,0] through hc_pre and RMSNorm, and records every digest (predictor/base/norm/attn_in/normed,
plus the pre/post/comb coefficient sets) in the refused evidence. That is the released reference
arithmetic, not a device launch: no kernel runs and the evidence says kernel_launched=false. The
increment that followed launched the arena wq_a/wkv B12x projections, then the q_norm/kv_norm/RoPE
seams; the refusal now names the window/selected construction and the format-2 attention body as the
remaining unwired stages.

The layer-0 attention projections are now the first real kernels the four-rank loop launches: the
mHC seam's normed attention input is rounded to BF16 and multiplied against the plan-bound B12x
wq_a [1280x5120] and wkv [512x5120] weight/scale pairs through
b12x_bf16a_mxfp8w_dense_sm121 (the D37 bf16a family: BF16 activations, FP8 E4M3 weights, F8_E8M0
scale grids). No weight is uploaded and nothing is repacked: the kernel reads the arena slices the
plan-dtype census bound. The two latents are read back and their sha256/L2 norms recorded, then the
loop refuses at the next unwired stage (q_norm/kv_norm/RoPE seams and the flash_mla_csa2_seq_sm121
attention body). The launch passes a zeroed 64-byte trace ring because the kernel's trace macro
dereferences the ring before it reads active_scope_mask, so a null pointer is not a legal trace-off.

The layer-0 query path runs next: q_norm (the arena's BF16 q_norm.weight, the engine's RMSNorm over
the whole q_lora_rank-wide wq_a latent, narrowed back to BF16 exactly as the reference's bf16 tensor
does), then the wq_b dense B12x launch on the per-rank 16 head rows (8192x1280: the stored wq_b is
[32768,1280], the Column plan keeps 8192 rows per rank, and the q head width is 512), then the RoPE
tail rotation of every 512-wide head row's last rope_head_dim values with engine
rope_tables_for_layer at the first-step position. The evidence names the per-rank head width and head
count, so a later reader cannot re-derive 128-wide heads from config.head_dim. The evidence records the q_norm digest, the pre-RoPE wq_b output digest and
the post-RoPE q digest, so the next stage can be pinned without a token. The verifier's q/kv norm
weights in the released checkpoint are BF16 (q_norm [1280]), and the arena carries them; no host
stand-in computes them for a device launch: the seam is declared in the evidence as the F5 BF16
round-trip and the RoPE table source.

The layer-0 kv seam runs on the host from the arena's BF16 kv_norm.weight: kv_norm is the engine's
RMSNorm over the 512-wide MLA latent row (config.head_dim is that latent width), narrowed to BF16,
then the last rope_head_dim values are rotated with rope_tables_for_layer at the first-step
position. The seam then runs the window fp8 round-trip on that rotated row through the deployed
sparkflash_kernels::cuda_harness::engine_window_row - the engine's BF16 storage round, the released
engine fp8 quantiser, and a byte-identity check against m4_quant::act_quant_fp8_row - producing the
engine-format row the body reads: 512 E4M3FN code bytes and 16 E8M0 power-of-two scale bytes
(WINDOW_QUANT_BLOCK 32), which the kernel decodes as bf16(code * scale). The evidence records both
digests and their lengths. The refusal names the window ring staging, the selected-candidate
construction and the flash_mla_csa2_seq_sm121 attention body as the remaining stages.

The layer-0 CSA union attention body then launches once on the current row: flash_mla_csa2_seq_sm121
with the query path's rotated BF16 q (16 per-rank heads, head width 512), the kv path's engine-format
window row (512 E4M3 codes, 16 E8M0 scales), the arena's per-rank attn_sink shard (16 f32) and a
zeroed index key/query pair, while the compressed-store pointers are null because selected_count is
0, which the kernel's own guard admits. cuda.launch is called directly with grid (per-rank heads, 1,
1) and the module's 256-thread block, because the deployed launch helper's pinned grid is the
whole-model 64-head grid; the kernel derives head from blockIdx.x and its head-count constant is only
a void use, so one rank's heads are a legal launch. The evidence says window_rows = 1 and
selected_count = 0 and marks the launch WIRING SMOKE ONLY: it is not the released reference attention
over the 130-token prefill window, no oracle comparison can pass on it, and no token can be claimed
from it. It does pin the body's ABI, grid, q/window/sink operands and out bytes (digest and L2), and
the same evidence notes that with a smaller grid the unconsumed index stage's block-score coverage is
partial, which nothing reads while selected_count is 0. The refusal now moves to the
selected-candidate construction and the prefill sequence, then the wo_a/wo_b projections with the
attention all-reduce, the mHC residual, the MoE stage and the certified head.

The attention block's output projection pair then runs and its collective is exercised: the grouped
b12x_bf16a_mxfp8w_wo_a_sm121 launch reads the arena's per-rank B12x wo_a weight and scale grid (the
plan keeps two group buffers of 1024 rows per rank, one grid block per group and 256-output-channel
tile), the row-sharded wo_b runs through the dense b12x_bf16a_mxfp8w_dense_sm121 entry on this
rank's 2048-wide inner slice, and the canonical bf16 all-reduce sums the four 5120-wide partials into
the attention block's delta. The evidence records both cubin digests, the wo_a output and wo_b
partial digests/L2, the reduced digest/L2, the collective transport and the reduce sequence. This is
the first attention-block collective the forward loop issues, and the delta is still computed from
the one-window-row smoke body, so the evidence says no token can be claimed from it; the refusal now
moves to the mHC residual, the MoE stage, the remaining layers and the prefill sequence, and then the
certified head.

The attention block then closes its residual on the host: the reduced BF16 delta (the wo_b dense
output narrowed to BF16 and summed by the canonical all-reduce) is decoded to f32 and mixed back onto
the four hc residual streams with the released engine mhc::hc_post, using the mHC entry seam's post
mixes and sinkhorn comb matrix. The evidence records the delta digest, the residual digest, the
post/comb digests, the updated stream digest/L2 and each stream copy's L2. This is the reference
arithmetic, but the delta still comes from the one-window-row smoke body, so the evidence says it is
not the reference layer-0 attention residual and no token can be claimed from it. The refusal now
moves to the FFN/MoE stage (hc_ffn mixes, router, gate/up, stored down combine, MoE all-reduce, ffn
hc_post), the remaining layers, the prefill sequence and the certified head.

The refused evidence also carries the full-plan attention operand census, computed on the host from
the rank plan alone: every attention tensor by group, the plan's layout predicate deciding B12x
versus stored, and the 215 projection weights across all 43 attention layers (40 backbone + 3 MTP).
Measured on the released headers (plan hash 0x9a59035fc40776c8), the census reproduces the
standalone attn-operands gate line for line: 215/215 projection weights plan-B12x, 0 stored;
attention-replicated 258 tensors (86 tiled, 395,066,112 B), attention-column 172 (86 tiled,
812,390,400 B), attention-row 86 (43 tiled, 451,328,000 B); repacked bytes 0, per-token weight H2D
0 B. The byte totals are the tiled weight plus the B12x scale grid plus the resident non-tiled norm
weights, the same composition the gate prints; the census is plan-only and the per-operand bind
agreement is check_operand_bind in the layer-0 census.

The serving tail is written and host-tested in crates/sparkflash-cli/src/tp4_tail.rs: TailStage
loads rmsnorm_sm121 and cert_lm_head_sm121 from the staged kernel directory, derives the certified
head's per-rank w_block_max side tensor ONCE at loop open from the bound arena's head shard (331 MiB
init-time D2H, zero per-token weight H2D; w_block_max_source="arena head shard, derived once at
loop open"), runs the final RMSNorm against the arena's final-norm weight, screens the rank's own
32,320 vocabulary rows, and gather_logits moves the f32 logits to every rank bit-exactly by
bit-splitting each value into two u16 words through the existing tcp all-gather (no arithmetic, so
no rounding), then argmaxes the full 129,280-entry vector with the lowest-id tie break. It is not
called by the loop yet: the loop stops before the first projection, so no head ever screens a
non-layer hidden state.

The loop body's first real stage is in: with --forward the arena epoch now gathers the last prompt
token's embedding row from its owner rank over the walk's tcp collective (bf16 all-gather of raw u16
words, so the assembled row is bit-identical to the owner's arena bytes) and writes
tp4-forward-embed-rR.json with the token, owner rank, shard row, exchange sequence and the row's
sha256. It then stops before the first attention projection with a named error: the 40-layer arena
sequence (q_a/kv projections, the 16-head-per-rank flash_mla_csa2_seq_sm121 launch, the 129-token
prefill) is the next increment, and screening the embedding row through the head would produce
plausible-looking wrong numbers rather than a token. The named error carries the gathered row's
sha256 and the evidence path.

The combine ABI is now wired: kernel-lead's pinned stored-layout module
(`mxfp4_moe_stored_abi` + `mxfp4_moe_stored_launch`, twenty-entry baseline, cubin 11,544 B
c452a9aa...) is in this build, STORED_COMBINE_ABI_READY is true, and the selector verifies the pin
before the driver loads the bytes. So the preflight now PASSES when the staged cubin directory holds
that exact artefact, and the expected outcome of this command today is the *loop* refusal: every rank
enters its arena epoch, binds, runs the forward preflight, and then refuses at the loop body with a
named error - tp4-token-rR.json says forward="refused", positions_run=0, logits_dump=null, and the
rank exits nonzero. That is still a labelled negative result, not a token run. Without the staged
cubin the refusal is the named missing-stored-cubin error instead.

The bind/loop files are unchanged by --forward (byte-identical evidence for the same argv without
the forward flags); the token stage is purely additive.

## The multi-position loop (893ef1a3)

Loop argv: --forward --forward-preflight --kernel-cubin-dir <cubins> --token-ids <ids>
--checkpoint <ckpt>. The loop runs prompt_len + decode_steps - 1 positions in ascending order, all
40 layers at each. The per-(layer, position) order is the reference forward's:

1. the mHC entry host seam on the current hc stream, then the two dense B12x projection launches;
2. the query path: q_norm, the wq_b dense launch, RoPE at the layer's own tables (compressor rope
   base above ratio 0); its BF16-narrowed q_norm row is the seam's q_latent;
3. the kv path: kv_norm over the wkv latent, RoPE at the position, and the engine-format window row
   (E4M3FN codes + E8M0 scales); the row is cached per layer for every position;
4. the reference seams: attention_seam_step runs the compressor, the index keys, the per-query
   index weights, the two-level candidate select and the fp4 compressed cache on THIS chain's own
   normed layer input and q latent. It reports the attended window width, the ascending compressed
   ids in the engine's shared index space and the compressed rows in row order. Nothing recomputes
   the attention projections;
5. staging (tp4_prefill::stage_attention): the last window_width cached window rows oldest first,
   and the selected ids as kernel values (id - window_rows) * ratio. That product is the compressed
   group's absolute token position, which is what the kernel's per-position expanded store indexes
   (row p decodes as store row p / ratio) and what its causal mask candidate <= position compares;
6. one launch of the pinned flash_mla_csa2_seq_sm121 per rank (grid 16 per-rank heads, 256
   threads), with the layer's sink shard read from the arena once at open. The BF16 out is rotated
   back with the inverse RoPE (f32) at the position and narrowed to the BF16 wo_a activation, the
   reference's order (model.py:772);
7. wo_a grouped + wo_b + the bf16 attention all-reduce, the attention residual host seam, the FFN
   host seams, the five MoE launches, the routed combine + reduce, the shared expert and the ffn
   hc_post. The surviving stream is the next layer's input;
8. every head position (the last decode_steps positions) then runs the certified head tail: hc_pre
   one-hot collapse, final RMSNorm, four rank screens, the exact f32 logits gather, the side-buffer
   gather and the lowest-global-id argmax. Seams are written as
   head-input/pre_head_hidden-pos<P>.f32 per head position (head-input-r<rank>/ on ranks 1-3), with
   pre_head_hidden.f32 as an alias for the last head position so m27's fixed path still exists.

Evidence: tp4-forward-embed-r<rank>.json carries prompt_ids (the exact prompt the run fed) and
forward_ids (the full teacher-forced input, prompt plus decode rows) so a consumer can replay a
claimed envelope without the fixture file, plus position_records (per-position wall, stream
digest, head token/value/logits sha and the reference comparison), forward_positions,
head_positions, the last position's full head record, the layer-0 attention body (staged
q/window/store/selected/sink digests, the inverse-RoPE activation and the sampled host twin) and
per-layer records for the last position. tp4-forward-progress-r<rank>.json is rewritten per
position (and per layer on head positions) with position, total_positions and, on failure,
failed_position / failed_layer / error: read that file first after a refused window.

Token consensus: after the loop the four ranks all-gather the decoded ids in the namespace
TOKEN_CONSENSUS_SEQ = u64::MAX - 0x7_0000 (a whole 0x1_0000 block away from the head gathers) and
the loop refuses unless all four lists are identical; only then do the envelopes become
forward-run with tokens.ids = the consensus ids. A disagreement is a named refusal, never an
averaged token.

Smoke windows use --prefill-limit N: positions 0..N-1 only, head rows at the last
min(decode_steps, N) positions. With the 138-id fixture, decode_steps=1 and --prefill-limit 12 the
smoke envelope is: positions_run=1, tokens.positions=[11], start_position=11, prompt_len=138 (the
prompt_len is derived from the id count and the decode row count), one dump row at position 11. The
row is the causal next-token distribution at position 11, i.e. the prediction for fixture id 12;
QA should compare it shape-only or against a host twin over the same 12-token prefix, not against
fixture id 11. A smoke that wants a reference_id in the dump must pass exactly one
--forward-reference-ids value.

The full acceptance window has no limit: decode_steps=9, the 138-id fixture (prompt_len=130),
start_position=129 and 9 rows at positions 129..137, one row per teacher-forced decode step.

The sampled host twin (mla_ref::attention_with_sink_union over the exact staged bytes) runs on the
head positions, every eighth position and all positions below 4; its BF16 mismatch statistics are
in every layer record, and a smoke window twins every position.

## Window-only gaps (what a GPU window still has to prove)

1. the stored-layout combine launch on real rows (the ABI is pinned and a size/sha256-mismatched
   cubin is refused by name); the window must also check the host twin
   mxfp4_ref::moe_combine_rank_stored;
2. the multi-position loop on device: the smoke window (--prefill-limit 12) proves the real
   attention body, staging, inverse RoPE, head and consensus path in a few minutes; a medium window
   (--prefill-limit 132) proves the saturated 128-row window and compressed ids beyond it;
3. the full acceptance: decode_steps=9 writes a forward-run envelope with 9 rows at positions
   129..137, the four rank envelopes carry identical ids, and the ids equal the host oracle's
   predicted tokens before any run envelope is reported as a token. Only a window can move this
   item; host gates cannot.
