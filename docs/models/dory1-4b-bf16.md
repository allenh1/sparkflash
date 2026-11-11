# NVIDIA Nemotron Labs Dory1.0-4B (BF16)

Status: **device arm at TP=1 when the CUDA images are built, and a latent-recurrence group of one
rank a node; CPU host reference otherwise.** The released checkpoint runs two recurrent loops, so
its group is two ranks; a three- or four-rank group, and the loop-1 drafting that needs one, runs
a copy configured with that many loops. The checkpoint is
nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (weight selector dory1.0-4b-bf16), a looped
normalized-GPT with 4B parameters and a 256K-token window.

Dory stores 72 sublayers as 36 physical transformer blocks. Each block is an attention sublayer
followed by an MLP sublayer. The first 16 sublayers (8 blocks) are the input prefix, the middle
40 (20 blocks) are a shared recurrent block, and the last 16 (8 blocks) are the output suffix.
The recurrent block runs twice by default; the checkpoint is the same for every loop count.

The arithmetic is NVIDIA's nGPT variant, not a plain residual transformer:

- q and k are rotated first, then L2-normalized per head. q is multiplied by the square of the
  learned sqk scale; k is not scaled.
- the attention output is multiplied by sigmoid(gate * sg), L2-normalized over all heads, and
  projected.
- the MLP scales gate and up by the learned suv vectors, applies SiLU, L2-normalizes the
  intermediate vector, and projects down.
- every residual is a normalized interpolation:
  justnorm(justnorm(x) + alpha * (justnorm(branch) - justnorm(x))).
- the input prefix's output is reinjected at the start of every recurrent pass. In per-loop mode
  each loop has its own KV cache; in last-loop mode one cache is overwritten.

## Serving

    sparkflash serve --model nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --context 8192

The dynamic route reads the checkpoint's config.json, plans the collapsed shared architecture
and opens the device arm when kernels/dory_sm121.cu and the Qwen token image were built. The device
arm uploads the 7.95 GB BF16 checkpoint once, runs the token kernels, and keeps the BF16 KV caches
on the device. Without a cubin, it opens sparkflash_model_dory::DoryHostBackend instead; the host
reference loads the checkpoint in f32 and is a correctness path, not a speed claim. A build with a
cubin but no usable CUDA driver is a refusal, not a silent fallback.

### The latent-recurrence group

Dory's recurrent block is the same weights run once a loop, so its loops can run on different
GPUs in order: the group is a pipeline over the loops, not tensor parallelism. No rank holds a
slice of a layer and nothing is reduced. Rank 0 owns HTTP, embeds the tokens and runs the input
prefix and its loops; each later rank runs its loops on the latent rows the rank before it sent;
the last rank runs the output suffix and the head, samples, and answers rank 0 with the token and
its logits row. Every rank keeps the KV caches of its own blocks and loops only, so a group of
four splits a four-loop cache four ways and each rank holds one loop's recurrent caches (rank 0
also the prefix's, the last rank also the suffix's).

    # node0 (rank 0, HTTP)
    sparkflash serve --checkpoint DORY_4LOOP --context 65536 --bind 0.0.0.0:8000 \
        --recurrence-rank 0 --recurrence-peers 192.0.2.10,192.0.2.11,192.0.2.12,192.0.2.13
    # node1 .. node3: the same, with --recurrence-rank 1, 2 and 3 and --bind 127.0.0.1:0

The world is the peer count. The loops must divide evenly by it (n_recurrent_loops in
config.json; the release ships 2, so its group is one or two ranks, and a three- or four-rank
group serves a copy whose config.json sets 3 or 4, a different model from the two-loop default),
and the caches must be per_loop: in last_loop mode a token's first loop reads rows another rank's
last loop wrote, so the group refuses it. `DoryStage::of_rank` refuses a world that does not
divide the loop count, and the recurrence tests exercise both the synthetic multi-loop fixtures
and the released two-loop document's refusal.

The ring is NCCL point-to-point between the GPUs (sparkflash_comm::device_reduce's send_bf16 and
recv_bf16 on the raw stream); TCP carries only the communicator's rendezvous, rank R on
`--recurrence-port` (default 29801) plus R. The group pins NCCL_NET=IB, which is RoCE on the
Spark rail, so a rail NCCL cannot use fails the open rather than falling back to sockets;
`--recurrence-net any` lets NCCL choose. Every message is a 32-word frame (Connect, Begin, Rows,
Step, Output, Draft, Fault, Shutdown) followed by its payload, so a worker knows what to post
before the payload arrives, and a rank that refuses a chunk sends a Fault in its place, keeping
the ring paired. The open ends with a Connect frame round the ring and back on the answers link: NCCL
connects a pair of ranks on its first transfer, and a first transfer made while requests are in
flight waited on the ring's queued work, which waited on the connect, and hung the group.

Rank 0 writes a whole prompt's frames before its first chunk, and the group cuts a prompt into
384-row chunks below 4,096 tokens and 512-row ones from there (recurrence_chunk) rather than one
process's 4,096, so chunk k+1 runs on rank 0 while chunk k runs on rank 1 and every stage is
busy. A prompt costs about the last rank's work on all its rows plus the first chunk's trip
through the ranks before it, about 2.4 of the last rank's chunk times, so a narrow chunk wins
until its tensor-core tiles leave the GB10's 48 SMs idle. Chunks start on 128-row tiles; a
short last chunk may fold into the previous one and be narrower: at 384 rows q/k/v/g are three
full waves and o_proj and down_proj one wave on 30 SMs, and 512 rows use that wave better once
the fill is a small part of the prompt. A chunk's rows run the
same kernels on every rank as on one process, and those kernels' rows do not depend on the
chunk's width (see Prefill), so the group's logits are the single process's bits:
tests/recurrence.rs checks group against single on the tiny fixture for (loops, ranks) of (3, 3),
(4, 4) and (4, 2), greedy and seeded, including a second session that starts after the first.

A decode step is one row walking the ring, so one stream decodes no faster than one process. The
group's other gains come from holding several sequences: `--concurrency N` (the same N on every
rank, which the communicator's open checks) gives each rank N sequence lanes, each with its own
caches and position, sharing one scratch set because a rank runs one step at a time. A request
takes a free lane at its prefill and keeps it until its session is dropped; a request beyond N
lanes is refused, as the host driver's N workers never send one.

Decode steps travel the ring in batches of up to eight lanes (LANE_BATCH), one Step frame and
one pass on every rank. A stage's pass is its weight reads, so it costs about the same at one row
as at eight: the projections read each weight row once for the whole batch through the token
GEMV, whose sum for a row is the one-row kernel's, while rope, the cache append and attention run
lane by lane on each lane's own cache. A lane's row therefore has the bits of a pass of its own,
whichever lanes share the pass. A session's step joins rank 0's queue and one session at a time
drives: it waits for rank 0 to finish the previous batch, then for fewer batches on the ring than
there are ranks after rank 0 (more when the lanes need more than eight a batch), then up to 2 ms
(RETURN_WINDOW) for the lanes whose answers it has just read, and takes the steps waiting. The
last rank samples every lane of a batch before answering, and the answers come back as one
message of frames and one of logits rows, so a batch's lanes return together and leave together.
Fewer, wider batches are the point: with three batches on the ring the slowest rank always has
one to run, and a fourth would only queue in front of it while its lanes could have ridden in a
wider one (two, three and four batches measured 130.7, 131.6 and 128.8 tok/s at 16 streams).

Answers come back on a second communicator from the last rank to rank 0, on a stream of rank 0's
own, into page-locked memory, so rank 0 waits for one batch's answers while it queues the next
batch's rows. Each rank takes frames in the order they reach it, so answers reach rank 0 in the
order it sent the chunks and batches that wanted them; each takes a ticket and the answers are
read in ticket order. Rank 0 keeps at most two of a prompt's chunks queued ahead of the one
running, so a decode batch waits behind two chunks, not the whole prompt. tests/recurrence.rs
runs three lanes from three threads at once (prompts of 4,200, 1,500 and 37 tokens, greedy and
seeded) and checks each against one process, bit for bit; on the real checkpoint, 16 concurrent
greedy streams (prompts of 38 to 4,347 tokens) match one process's text and top-5 logprobs
exactly. Each lane costs its caches on every rank: at a 65,536 context about 2.3 GB on rank 0 and
less on the others.

### Loop-1 drafting

The output suffix and the head read a latent, and the latent after any loop is one: run on the
latent after loop 1 they are a one-loop model of the same weights, whose greedy token agrees with
the four-loop model's about nine times in ten. A group of three or more ranks drafts with it.
Rank 1 (DRAFTER) holds a draft exit, the output blocks and the head on caches of its own, and
runs it over every row it receives; device.rs's the_draft_exit_is_the_one_loop_model_in_either_order
checks the exit against the one-loop fixture bit for bit. For a row rank 0 marked for a guess the
exit runs first, and its greedy token goes back to rank 0 in a Draft frame on a third
communicator, `{run}:drafts`, before rank 1's own loops run; for any other row it runs once the
row has gone on to rank 2, which then waits only on rank 1's loops (running it first on every row
cost 6.6% at eight streams and 20 ms of first-token time, since the ring is latency-bound there).

On rank 0 a listener thread reads the drafts link and leaves each guess in a mailbox, and a
speculator thread starts the guessed row: the lane's next row, fed the guessed token at the next
position, while the row before it is still on the ring. A lane may run DRAFT_DEPTH (4) rows ahead
of its session this way. The listener only reads: if it started rows itself it would wait on the
head's lock, held by a session waiting on rank 0's stream, which can wait on rank 1's next receive,
which waits behind its send of the next guess to the listener. When an answer comes back the
session checks the oldest row started on a guess. A match means that row is the one the session
would have started, with the same token, position and caches, so its answer is simply the next
step's: the speculation is lossless. A miss, or an end of turn, drops every row started from that
position. The lane's epoch bumps, which makes guesses from the dropped rows stale, and rank 0
rewinds at once. The lane's next row carries REWIND, so every later rank takes the lane's caches
back to that position and the last rank restores its sampler mark (MARKS keeps one for every row a
lane can have in flight); a seeded sampler then draws what it would have drawn had the dropped
rows never run. The response's speculative counts report the guesses checked and kept.

A rank's pass costs about the same at one lane as at eight, so a drafting lane in a pass of its
own costs rank 0 a whole pass. A trace at eight streams showed four chains of lanes taking turns
on rank 0, 26 ms each and 104 ms a round, and drafting gained nothing over the plain group. The
drafting lanes therefore share passes. A session whose lane missed does not start its corrected
row alone while another drafting lane's guess is on its way; the row joins the pass that guess
starts. The speculator holds the guesses that have come, up to GATHER_WAIT (40 ms, a little over a
guess's round trip), while the guesses still due from another pass would fit in the room left in
this one. Joining rows fill the last pass's room, and the rest wait for the next pass unless they
have already waited GATHER_WAIT. A pass is at most LANE_BATCH lanes, and a lane is in a pass at
most once.

`--recurrence-draft-lanes N` (the same on every rank) drafts while the group holds at most N
sequences, and the plain batching above runs beyond that. The default is 16, two passes, in a
group of three or more ranks, and 0 otherwise; asking a two-rank group to draft is refused, since
rank 1 is its last. Past two passes the plain group has full passes to run, and the draft exit and
the rows a wrong guess drops are work it has no idle pass for: the two are even at 24 streams, and
the plain group wins at 32 (see Measured).

### Kernels

The arm reuses the shared Qwen token image for q4e_gemv_bf16, q4e_gemv4_bf16 and q4e_embed, and
adds kernels/dory_sm121.cu for the nGPT-specific arithmetic. The token GEMVs take one weight row
a warp for one row of activations and two for a batch, where each activation load feeds both
weight rows; every (row, token) sum is grouped as the one-row kernel groups it, so the bits do not
depend on the batch. Every dory kernel opens with the
programmatic-launch dependency wait (OPT_PDL_ENTRY), as the q4e kernels between them do, so a
launch overlaps the previous kernel's run rather than leaving a gap after it:

- dory_mma_wide: prompt-chunk projections on the shared wide tensor-core GEMM body
  (shared_prefill_wide_body, GLM's bulk-prefill BF16 GEMM), 128 tokens x 256 columns a block,
  f32 activations split into a BF16 head and remainder against the checkpoint's row-major BF16
  weights.
- dory_mma_wide4: the same tile over up to four matrices of one activation in one launch (q, k,
  v and g; gate and up), each block placed on its matrix's tile by shared_prefill_wide_tile, so
  every output keeps dory_mma_wide's chain. A launch finishes a 48-block wave in about the same
  time whatever its rows, and separate launches left each matrix's last wave part empty: at a
  1,024-row chunk k and v were 32 blocks, a whole wave each. device.rs's
  a_grouped_chunk_projection_is_each_matrix_alone_bit_for_bit checks the grouped launch against
  each matrix's own, bit for bit, with a matrix ending inside a tile and partial row tiles.
- dory_rope_norm / dory_rope_norm_lanes: one block a head. It applies rope through
  q4e_rope_pair (the rotation q4e_rope uses), takes the per-head L2 norm with the sqk scale
  squared on q, and for a kv head appends the BF16 K and V rows to a full-context or wrapping
  sliding-window cache. It replaced two rope launches, a norm and two cache appends with the same
  arithmetic, so the same bits. The first takes a prompt chunk's rows into one cache, the second
  a decode pass's lanes, each at its own position into its own cache; both run one row body.
- dory_attn_split / dory_attn_combine: decode attention split over the keys. Block (kv head,
  split, lane) reads its run of K and V rows once for every query head that shares the kv head,
  keeps an online softmax a query head, and writes a partial; the combine merges a head's splits
  in split order and applies the sigmoid output gate. One launch serves all of a decode pass's
  lanes (DORY_LANES, eight): each lane's position, run length and cache are launch parameters,
  and a lane's blocks do what a one-lane launch's do.
- dory_attn_mma16 / 64 / 128 / 256: prefill attention on tensor cores (mma.sync m16n8k16, f32
  accumulate), one block a 16-query tile and kv head covering the kv head's query heads, with
  cp.async K/V tiles of 32 keys in an XOR-swizzled shared layout. The instance is chosen by
  head_dim.
- dory_mlp_epilogue: suv-scaled SiLU product and intermediate L2 norm.
- dory_slerp / dory_loop_combine / dory_justnorm: normalized residual lerp, recurrent input
  reinjection and state norm.
- dory_logit_scale: the vocabulary logit scale.

### Prefill

A per-loop prompt is processed in chunks of up to 4,096 tokens (prompt_chunks). The arm refuses
a checkpoint whose projection widths are not multiples of 32, because the wide kernel is the
32-column-step MMA and the token GEMV's bits differ; with that shape, projections over a chunk
wider than eight rows run dory_mma_wide. So that no chunk falls to the token GEMV over a
token-sized tail, a last chunk of eight rows or fewer takes 16 rows from the chunk before it.
With that rule every row's projection is the same chain of MMAs whatever the chunk's width,
attention's query tiles start at 16-aligned absolute positions and its key tiles at the window's
first key, and a prompt's logits do not depend on how it was cut: the group's 384- and 512-row
chunks and the 4,096-row chunks of one process give the same bits. Last-loop mode keeps the
token-at-a-time order, because there a later token's cache row has already been overwritten by
its final loop.

Attention writes the whole chunk's K/V rows first. A sliding block never reads a row older than
its window, so its cache is a ring of `window + 4096` rows and its attention starts at the
window's first key; the padding keeps a chunk's later writes from overwriting a row an earlier
query in that chunk still needs. A full-attention block keeps the whole context. This is why
four-loop per-loop mode needs about 30 GiB rather than 96 GiB at 256K, and why its first request
does not pay the full-context allocation.

### Arithmetic changes on this branch

These change bits against the arm's first commit, and each was checked against an f64 reference
and the host oracle:

- **Decode attention reorders its sums.** dory_attn_split sums a row's keys in eight warps and up
  to 256 splits, merged in a fixed order, rather than one block's serial walk. Against an f64
  reference on a 16-head, 4-kv-head, head_dim-256 cache the worst error is 1.7e-6 (bound 1e-5).
- **Prefill attention multiplies 16-bit q and P.** K and V are the cache's BF16 already; q and
  the softmax probabilities P are each split into a BF16 head and the BF16 of the remainder, two
  MMAs a fragment, so each carries about 16 bits rather than f32's 24. Against an f64 reference
  the worst error is 2e-6 on flat scores and 4.1e-5 on sharp ones (bound 1e-4); the f32 tile it
  replaced measured 1.1e-7 and 2.1e-6. On the real checkpoint's 26- and 140-token prompts the
  prefill logits move by KL 5.5e-6 and 4.1e-5 from the f32 tile, less than either is from the
  host f32 oracle (8.3e-5 and 7.0e-5 for the tile, 9.4e-5 and 6.1e-5 for the MMA); the argmax is
  unchanged.
- **Short tails run on MMA.** A prompt whose last chunk had one to eight rows ran those rows on
  the token GEMV; now they run in a 17- to 24-row chunk on dory_mma_wide. This touches only
  prompts of length 1..8 mod 4,096 beyond the first chunk.

Batched prefill and token-at-a-time decode use different attention kernels, so their logits are
close but not identical; tests/device.rs asserts their relative difference stays below 1e-3,
checks the chunked prefill against the host, and checks the sequential decode against the
chunked prefill.

### Measured

The tables below use a four-loop copy of the checkpoint (the released document ships two loops,
which only a two-rank group can split), greedy, client wall time, cold prompt, median of two
rounds; prefill is the one-token request's rate, decode the rate over 64 more tokens. "One GB10"
is one process; "group" is four GB10 nodes, one rank each, over the RoCE rail. "Before" is this
branch's first commit.

| Prompt tokens | Prefill, one GB10 | Group | Decode, one GB10 | Group |
| --: | --: | --: | --: | --: |
| 1,666 | 1,095 tok/s | 2,617 tok/s | 12.60 tok/s | 12.55 tok/s |
| 6,646 | 1,363 tok/s | 3,933 tok/s | 12.20 tok/s | 12.15 tok/s |
| 13,285 | 1,352 tok/s | 4,074 tok/s | 11.69 tok/s | 11.66 tok/s |
| 26,562 | 1,235 tok/s | 3,799 tok/s | 10.85 tok/s | 10.88 tok/s |

With the earlier 1,024-row chunks and a launch a projection the group prefilled the same prompts
at 1,834, 3,296, 3,662 and 3,570 tok/s; the shipped group uses 384 rows below 4,096 tokens and
512 above. Measured chunk widths with the grouped launches (tok/s at 1,666 /
3,326 / 6,646 / 13,285 tokens): 256 rows 2,492 / 3,067 / 3,316 / 3,305, 384 rows 2,606 / 3,390 /
3,812 / 3,966, 448 rows 2,522 / 3,192 / 3,678 / 3,746, 512 rows 2,499 / 3,360 / 3,916 / 4,064,
1,024 rows 1,956 / 2,700 / 3,413 / 3,734, 1,536 rows 1,527 / 2,300 / 3,090 / 3,568.

The branch's first commit prefilled 653 to 359 tok/s and decoded 8.60 to 4.07 tok/s on one GB10
over prompts of 1,202 to 19,590 tokens. One group stream decodes about 5% slower than one process:
its step crosses four hops and comes back.

Concurrent streams on the group, `--concurrency 32` on every rank (which takes rank 0 to 86 of the
GB10's 121 GB at a 65,536 context), distinct 1,666-token prompts, 300 greedy tokens each,
aggregate decode over the window where every stream decodes:

| Streams | Aggregate decode | Per stream |
| --: | --: | --: |
| 1 | 12.6 tok/s | 12.6 tok/s |
| 2 | 24.4 tok/s | 12.2 tok/s |
| 4 | 46.4 tok/s | 11.6 tok/s |
| 8 | 79.3 tok/s | 9.9 tok/s |
| 16 | 152.8 tok/s | 9.5 tok/s |
| 24 | 221.8 tok/s | 9.2 tok/s |
| 32 | 246.5 tok/s | 7.7 tok/s |

Before decode was batched, each lane's step was its own pass over every stage's weights, and four
or more lanes topped out near 40 tok/s, the slowest stage's 25 ms a lane. Batched, a step at 8 and
16 streams takes one traversal of the ring, about 110 ms: three batches of three to six lanes are
on the ring at once. At 32 streams four batches of eight keep the last rank busy, and its pass
sets the pace. On the last rank (28 blocks and the head) a pass takes 26.5 ms at one lane and
32.0 ms at seven; of the 5.5 ms, 3.8 ms is the lanes' split attention, close to their KV rows'
read time, and 1.8 ms the BF16 GEMVs reading seven activation rows a weight row. Two passes
brought the seven-lane pass there from 35.1 ms. Launching rope, the cache append and split
attention once for all the lanes, rather than three launches a lane a block, took it to 33.1 ms.
Then the token GEMVs took two weight rows a warp: eight rows' f32 activations are sixteen times
the BF16 weight bytes, and with one weight row a warp the activations' L1 reads, not DRAM, set
the pace of the 2,560-row o_proj and down_proj, which at seven rows ran 19% slower than at one;
two rows a warp halve those reads, and the gap is 13% (6.8 to 7.7 ms a pass). A microbenchmark of
the kernel alone at eight rows: o_proj 196 to 220 GB/s, down_proj 193 to 223, the head 236 to 240,
and one row 237 to 250 (234 to 243 before). The 16 prompts prefill one after another through
the same pipeline, so the last stream's first token comes about 8 s after its request. Reading the answers into pageable memory serialised the lanes
(a pageable copy held the driver's staging path while it waited): aggregate decode stayed at 11.5
tok/s at every stream count until the answer moved to page-locked memory.

Loop-1 drafting on the same group, `--concurrency 32`: six chat prompts (stream i asks prompt i
mod 6), 256 greedy tokens each, aggregate decode over the window where every stream decodes.
"Off" is `--recurrence-draft-lanes 0`, "1" drafts one sequence at a time (the first build), and
the last two columns share passes as above, at 16 and 32 lanes.

| Streams | Off | 1 | 16 | 32 |
| --: | --: | --: | --: | --: |
| 1 | 12.69 tok/s | 26.27 tok/s | 26.68 tok/s | |
| 2 | 25.03 tok/s | 25.06 tok/s | 50.05 tok/s | |
| 4 | 48.56 tok/s | 48.89 tok/s | 97.70 tok/s | |
| 8 | 93.75 tok/s | 94.24 tok/s | 185.96 tok/s | |
| 16 | 174.36 tok/s | | 227.37 tok/s | 227.84 tok/s |
| 24 | 262.92 tok/s | | | 259.41 tok/s |
| 32 | 283.77 tok/s | | | 201.62 tok/s |

The guesses kept were 89 to 92% of those checked. Every stream's text was byte-identical with
drafting on and off at 1, 8, 16, 24 and 32 streams. Before the guesses shared passes, eight
streams drafting reached 82.7 tok/s with joining alone and 148.8 with gathering; filling a pass's
room with joining rows, and gathering only for a pass that fits, took it to 186.

Where the time went, one process: split decode attention flattened decode across context
(4.07 to 10.5 tok/s at 19.6K, where the old kernel's one-block-a-head walk dominated), tensor-core
prefill attention cut a 9.8K prompt's attention from about 11 s to 1.35 s, and the wide GEMM tile
made projections 23% faster. A decode step is about 80 ms. 74 ms of it is the BF16 GEMVs
streaming the weights at 236 to 250 GB/s, the GB10's DRAM floor (q/k/v/g 209 us, gate/up 295 us,
o_proj 87 us and down_proj 156 us a block, the head 2.7 ms). Of the rest, launching the dory
kernels programmatically removed a 2 us gap before each one, the one rope-norm-append kernel
(3.5 us) replaced five launches a block, and the MLP epilogue loading every operand before its
first product, rather than an L2 round trip an element behind each division's slow-path call,
took it from 16.1 to 5.8 us. Together they lifted decode 3.5 to 3.9% (12.13 to 12.55 tok/s at 1,666 tokens, 11.22
to 11.66 at 13,285) with the same bits.

The device arm's tests/device.rs compares its logits with the host reference on a 4200-token tiny
fixture whose sliding ring wraps, and asserts that the chunked prefill's logits match the same
tokens decoded one at a time within a relative 1e-3. The device and group tests are marked
`#[ignore]`; run them with `cargo test -p sparkflash-model-dory -- --ignored` on a host with a
Dory/Qwen cubin and a CUDA device.

The runtime profile is crates/sparkflash-model-dory/config/dory1_0_4b_bf16.toml.

## What this arm does not have yet

- **Speculation on one process.** No Dory drafter is published, so the profile ships
  speculation.enabled = false. Only the group drafts, from its own first loop (see Loop-1
  drafting). One process could draft the same way. Its recurrent block is one set of weights for
  every loop, so a row guessed after an early loop and the rows still running later loops could
  share the block's weight reads in one batched pass. LoopSpec (arXiv 2609.17184) and WaveFront
  Decoding (arXiv 2609.23033) do this for Ouro and Huginn on one GPU. This arm does not yet.
- **Tensor parallelism.** One process is one GPU; the group splits loops, not layers.
- **Prefix reuse.** Every request prefills its whole prompt.
- **Group fault tolerance.** A worker that dies mid-sequence leaves rank 0 waiting on NCCL; there
  is no communicator timeout, so the group is restarted as a whole.
- **Quantized releases.** Only the BF16 checkpoint is planned. MXFP8, NVFP4 and GGUF are
  different storage formats and are not yet read by this pack.
- **A numerics-registry capture.** The device tests compare the arm with the host oracle and
  reject a non-finite or out-of-band logit, but no released-checkpoint logprob reference has been
  captured on a CUDA host, so there is no `numerics/registry.toml` entry to regression-gate. The
  profile's prefill chunk is fixed at 4,096 rows and the arm refuses a different value rather
  than silently ignoring it.

## Verification

- DoryConfig::from_json parses the released document into 36 blocks, asserts the
  input/recurrent/output split and derives per-block rope and sliding windows.
- DoryTensorPlan describes all 507 tensor names and pins the released 7,950,458,880-byte total.
- tests/host.rs writes a tiny 8-sublayer checkpoint through the real safetensors writer, loads it
  through WeightSource, and runs prefill and decode through DoryHostBackend.
- tests/device.rs (ignored without a CUDA device) runs a 4200-token prompt whose sliding ring
  wraps, asserts the chunked prefill's difference from the same tokens decoded one at a time
  stays under a relative 1e-3, and checks the prefill against the host oracle.
- tests/recurrence.rs runs the group over in-process links and checks its logits and tokens
  against one process, bit for bit, one sequence at a time and three lanes at once, and that a
  sequence beyond the lanes is refused. Groups of three or more ranks run each check with drafting
  off and on. more_drafting_lanes_than_a_pass_are_each_the_single_process_bit_for_bit runs ten
  drafting lanes, greedy and seeded, so full passes, joining rows and holds all occur.
  a_drafting_group_drops_the_rows_a_caller_overrides_and_counts_its_guesses feeds a token
  the group did not choose every third step, so the rows started on guesses are dropped on every
  rank, and checks the result against one process and the guess counts; and drafting_needs_a_rank_between_the_first_and_the_last checks the
  refusal.
- device.rs's a_batched_lane_pass_is_each_lane_alone_bit_for_bit decodes three lanes at
  different positions (one past the sliding ring's wrap) in one batched pass and checks every
  lane's logits against the same sequence run alone, bit for bit, and that a batch wider than
  LANE_BATCH is refused.
- device.rs's unit tests check split decode attention and tensor-core prefill attention against
  f64 references on the real head shape, and pin prompt_chunks' tail rule.
- sparkflash-model parses the same document into the shared 36-layer walk and records the loop in
  LoopSchedule, which the KV plan reads.
