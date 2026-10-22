# The head stage in the four-rank loop: recording of the task-16 contract (WS-A)

Status: recorded, not wired. The loop currently closes at layer 0's FFN hc_post; the head stage runs
after the last layer's ffn hc_post row exists. This page records what the loop owes kernel-lead's
adjudicator (docs/build/head-tail-contract.md, commit d04d03e; harness m27_head_tail, aarch64 sha256
850040c667b53ad547200113eeb89c80f0b0c9ecd9aca7bd385d58df6cd68a68, 5,010,688 B) so the record can be
written once and diffed field by field.

## 1. The path the loop runs

final hidden row (ffn post-hc of the last layer, f32 5120)
  -> rmsnorm_sm121, one launch per rank, BF16 in/out, eps = f32(config rms_norm_eps)
  -> cert_lm_head_sm121, one launch per rank on its 32,320-entry vocabulary shard (grid [32,1,1],
     block [128,1,1], 40 block bounds, exact f32 ascending chain for survivors)
  -> exact f32 logit all-gather over the rank words (host word movement, no device launch)
  -> host argmax with the lowest-index tie break
  -> token iff the winning entry is EXACT (certified == 0); a certified bound is a REFUSED token,
     recorded claimable = false, never a claimed token.

Single-stream launch arithmetic: norm 4, head 4, reduce 0 per position (4 total, one per rank in
each class). lm_head_reduce_sm121 is dropped from this path; the six-row batched symbols
(cert_lm_head_batched_sm121, lm_head_reduce_batched_sm121) are verify-path only and cannot be called
with one row. The loop must not launch either in the single-stream path.

## 2. The seam file the loop writes

<evidence>/head-input/pre_head_hidden.f32 - exactly 5,120 little-endian f32 (20,480 B): the ffn
post-hc row the head consumes, BEFORE the final norm. Rank 0 writes this canonical path; ranks 1-3
write <evidence>/head-input-r{rank}/pre_head_hidden.f32 copies (the row is replicated by
construction: the last layer's MoE reduce makes the delta common, so the bytes must be identical and
a difference is a seam bug). m27 accepts the directory or the file itself.

## 3. The head block the loop records

Schema sparkflash-head-tail-evidence/1, written by rank r as tp4-head-tail-r{rank}.json in the shared
evidence dir. Field names follow the contract exactly:

input {sha256, bytes, bf16_codes_sha256, bf16_codes_bytes}
norm {kernel rmsnorm_sm121, pin cubin_bytes/cubin_sha256, eps, normed_bf16_sha256, normed_bytes}
head {kernel cert_lm_head_sm121, pin, grid [32,1,1], block [128,1,1], chunks_per_rank 32,
      deterministic [bool;4], ranks[4] {rank, logits_sha256 (32,320 f32 LE), logits_f32_values,
      certified_count, certified_sha256 (32,320 u8 flags), block_max_sha256 (32 f32),
      block_argmax_sha256 (32 u32)}}
gather {route exact-f32-word-gather, gathered_logits_f32_values 129280, gathered_logits_sha256}
argmax {token, value, certified, claimable, twin_token, twin_value}
reduce {chain_head_launches_per_position 5, single_stream_head_launches_per_position 4,
        reduce_launches_dropped 1, reduce_mode dropped-single-stream,
        batched_reduce {fixed_rows 6, callable_with_one_row false}}
launches {norm 4, head 4, reduce 0}

After the exact f32 gather every rank holds all 129,280 logits, so every rank can compute all four
per-rank slices (32,320 each) and therefore write the same head.ranks[4] table; the rank field labels
the file, the block is common. twin_token / twin_value are the reference-logit comparison from the
existing exact-logit/reference-logit rule; the loop leaves them null when no reference is supplied and
m27 supplies its twin in the diff.

## 4. Join keys and refusal semantics (kernel-lead's adjudicator)

loop head.input.sha256 == m27 input.sha256; norm.normed_bf16_sha256; per-rank
logits/certified/block digests; gather.gathered_logits_sha256; argmax.token/value/claimable. First
divergent family names norm (K-norm), head (K-head), gather (R-head) or argmax/reference (token
claim). Reduced launches are checked as arithmetic (5 -> 4, one reduce dropped); no device speedup is
claimed from the harness (m24 p50s are host-side arithmetic only).

A certified winner is a refusal, not a mismatch: record it as REFUSED with claimable false and the
refusal names the certified bound. reference_id is unchanged (no new reference).

## 5. Open items before wiring

1. confirm the loop writes tp4-head-tail-r{rank}.json (a separate rank record) rather than embedding
   the head block in tp4-forward-embed-r{rank}.json; the field names are identical either way.
2. confirm twin_token/twin_value may be null in the loop record and filled by m27 in the diff.
3. confirm the seam copies for ranks 1-3 are accepted (canonical rank0 path is the one m27 reads).
4. the head launches need the arena head weight/scale binds and rmsnorm_sm121's in/out layout; the
   bind already covers tail.embed/head/norm, the launch wrapper is cert_lm_head_launch.rs /
   rmsnorm_launch.rs, and the m27 example is the reference sequence.

## 6. Confirmations from kernel-lead (recorded 2026-10-20)

- Record file: a separate <evidence>/tp4-head-tail-r{rank}.json carrying the head block is the
  preferred shape; the adjudicator reads the explicit head.ranks[i] index, so all four ranks writing
  the same common table is ideal. The field names above are kept identical to the contract.
- twin_token / twin_value: null means not-yet-compared, never divergent. Only set them if the loop
  computes the same host twin (rmsnorm_ref::final_norm_bf16 + lm_head_ref::exact_logit per rank, then
  the lowest-index argmax over the gathered exact logits). A reference-logit token is a separate
  field and must not be written into twin_token.
- Seam copies confirmed: rank0 canonical head-input/pre_head_hidden.f32, ranks 1-3
  head-input-r{rank}/pre_head_hidden.f32 byte-identical; the adjudicator cross-checks the copies and
  joins the canonical sha256/bytes to m27 input.sha256/bytes. A cross-rank difference is a seam bug.
- Pins/ABI confirmed: cert_lm_head_sm121 38,088 B sha256 bbd30c8d6d358bd200b07cffa54d662c9144501b69310864933b68a0e53b6798;
  rmsnorm_sm121 31,480 B sha256 5f953bc0fb781afe3f9b42703b19b37f89ef45b74a4ad607938df2e2c3448579;
  both are baseline entries with 0 membar lines, no baseline change. The norm ABI is 9 words
  (hidden BF16 rows x cols, weight BF16 cols, out BF16 rows x cols, rows u32, cols u32, eps f32 bits,
  trace_ring u64, layer_id u32, step_seq u64), grid ceil(rows/256), block 256; feed rows 1, cols 5120,
  hidden = RNE BF16 of the layer-39 ffn post-hc f32 row, eps = f32(config rms_norm_eps), and its
  output digest is norm.normed_bf16_sha256 (10,240 B). The head ABI is 12 words (hidden BF16 [1,5120],
  w BF16 [32320,5120] rank shard, w_block_max BF16 [32320,40] per-block weight maxima derived once
  per rank, logits f32 [32320], certified u8 [32320], certified_count u32, block_max f32 [32],
  block_argmax u32 [32], vocab u32 = 32320, trace_ring u64, layer_id u32, step_seq u64), grid
  (32,1,1), block (128,1,1); no reduce launch in the single-stream path.
- Shard convention and the arena weight: rank r's head shard is the released CONTIGUOUS row block
  [r*32320, (r+1)*32320) (engine TensorPlan declares head.weight BF16 [129280,5120];
  sparkflash-device/src/resident.rs classifies it Group::Embedding / ShardRule::Column and splits stored
  axis 0 into equal row blocks). Gather = r0 || r1 || r2 || r3 = 129,280 f32 words, exact byte
  movement, 0 ULP, 0 launches. The arena head bind has no companion scale tensor, so the head reads
  the released BF16 rows directly - the per-rank logits digest join against m27 is valid. HEAD_GATHER_SEQ
  only needs to be distinct; the join compares bytes and digests, not sequence words.

## 7. The exact derivation rules the loop must reuse (kernel-lead, 2026-10-20)

w_block_max (BF16 [32320, 40] per rank), from lm_head_ref::block_maxima_bf16
(crates/sparkflash-kernels/src/lm_head_ref.rs:111-145) - call that shared function; do not
re-implement:
- rows 32,320 contiguous (rank r = released head rows [r*32320, (r+1)*32320)), HIDDEN 5120, 40 blocks
  of SCREEN_TILE 128 columns;
- per row and block b, widen each BF16 weight exactly to f32, take abs(), keep the strict-greater
  running max (first maximum wins on equal magnitudes);
- store the WINNING WEIGHT'S BF16 BITS WITH THE SIGN CLEARED (bits & 0x7fff) as u16 LE: the abs of the
  original BF16 pattern, never a re-rounding of an f32 magnitude;
- output row-major (entry*40 + block) u16 LE, 2,585,600 B per rank. Kernel pointer for rank r is the
  rank-local base when each rank stages only its shard. Deriving over the full 129,280-row image and
  slicing rows [r*32320,(r+1)*32320) is byte-identical to calling it on the shard.

Norm input conversion: f32 seam row -> BF16 via RNE (__float2bfloat16_rn =
half::bf16::from_f32 bits = m4_quant::bf16_bits, crates/sparkflash-kernels/src/m4_quant.rs:41-45).
Never truncate. The row goes to rmsnorm_sm121 with rows 1, cols 5120 and eps f32 bits; the kernel
rounds its f32 output to BF16 RNE itself, and norm.normed_bf16_sha256 is over those 10,240 B.

Host-recomputed block_max f32 [32] / block_argmax u32 [32] (only if the loop recomputes instead of
reading the kernel readback): per rank block g covers entries [g*1024, min((g+1)*1024, 32320)), so
block 31 is partial (31,744..32,319 = 576 entries); max over logits with strict greater-than, lowest
entry wins ties. The kernel's published arrays are the authority.

Diagnosis order for a first-run mismatch: normed sha mismatch -> BF16 RNE or norm weight/eps; normed
matches but per-rank logits mismatch -> w_block_max derivation or certified/pruning semantics; logits
match but gather/argmax mismatch -> gather order or tie-break; certified winner -> REFUSED, not a
mismatch.

## Buffer initialisation: the certified count (m1y9 finding)

The certified head accumulates its pruned-entry count with atomicAdd and never initialises the
buffer, and cuMemAlloc does not zero the memory it returns. In m1y7 the single tail's fresh
allocation happened to be zero, so the count read back 0 and the run passed; m1y9's multi-position
loop opens a tail stage per head position, the allocator handed back recycled blocks, and the count
read back as per-rank garbage around 2.0e9, which the vocabulary check refused by name. The stage now
writes a zero into certified_count immediately before every certified-head launch, not only at open,
so a stage reused across head positions is correct by construction; the mask, logits, block maxima and
block argmax need no reset because the kernels write every entry or block they read back.

The loop's compact attention-body evidence also carries the position it ran at, so a per-position
record is self-describing without reading the enclosing progress object.
