# Version 1: the profile fields that cannot move the bits

This report is the evidence the declaration check asks for when the serving side adds a field to
`sparkflash_model::numerics::NEUTRAL_FIELDS`. It covers the settings that joined the list while
DeepSeek's serving switches moved into the typed profile table and its 40-layer device profile
left its defaults to `DiagnosticsProfile`/`ModelProfile` (the `NEUTRAL_FIELDS` diff from
`536c60ca` to this change). For every field it names the code that reads the setting, why
changing it cannot change a token's arithmetic, and the gate that holds that claim: a named test,
an invariance in [docs/NUMERICS.md](../NUMERICS.md) section 1, or an argument from the code that
a reader can check without a GPU.

The standard is the one in section 5 of docs/NUMERICS.md: a field is neutral only if a profile
may change it without recapturing any reference. "It is a diagnostic" or "it only schedules" is
not enough by itself; the entry below says which statement in the code makes that true. Several
of these settings choose a different launch, a different tiling or a different read path; in
each case the kernels are written expression-for-expression against the path they replace, or
the path only moves bytes that are already fixed. The entries say which.

Not every setting in `NEUTRAL_FIELDS` is arithmetic. The settings whose code does not hold them
to the same bits, and the ones that are harness or filesystem metadata rather than serving
settings, are deliberately absent from this report so the declaration check keeps failing on
them; they are listed in the change's review message, not buried in this evidence file.

## Diagnostics

### `diagnostics.cuda_graph`

Read at `crates/sparkflash-deepseek-forward/src/tp4_graph.rs:131` (the once-per-process
`GraphMode::from_env_value`). `off` launches each kernel directly; `on` captures the launch
list of the first step and replays it; `report` measures without replaying. Capture records the
same kernel arguments and the same stream order and replay re-issues them; the mode has no
branch inside a kernel and no arithmetic of its own. `check_preconditions`
(`tp4_graph.rs:142`) refuses capture when an event bracket (stage timing, the probe meter) is
armed, so a mode cannot silently change what is timed. Gate:
`tp4_graph::tests::the_first_launch_list_is_captured_then_replayed_exactly`
(`crates/sparkflash-deepseek-forward/src/tp4_graph.rs:566`).

### `diagnostics.device_diff`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17623`, `:18348` and `:18670`.
When on, the forward loop takes the host-twin comparison branch instead of stopping after the
reduce dump, records the comparison, and skips the S1 steady-state and collective-count
assertions that exist only to catch a broken device run. The device chain's outputs are still
the ones returned; the host twin's values are written to the evidence record, never fed back
into the device buffers (the head at `:18348` returns the `gathered` device logits regardless).
Gate: code argument above; the switch changes only which comparison and assertions run.

### `diagnostics.device_gather`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:3601` (`device_exchange`).
It decides whether the one-shot embedding/argmax gathers use the device exchange or the host
transport. Both gather the same fixed-width words: the device path packs the f32 argmax value
and id into two u16 words and the rank picks the strict maximum
(`tp4_tail.rs:500-517`), while the host path carries the same values; the gather is a
permutation of words, not a reduction whose order changes. Gate:
`tp4_tail::tests::the_exact_gather_moves_f32_bit_patterns_through_the_u16_collective`
(`crates/sparkflash-deepseek-forward/src/tp4_tail.rs:710`).

### `diagnostics.embed_replica`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:974`
(`replicate_embedding`). On, one AllGather at boot builds a whole-table copy; off, a step reads
its row from the owner rank. The gather is a byte copy of each rank's vocabulary shard; the
embedding row a token selects is the same bytes either way, and the following seam kernels are
unchanged. Gate: code argument in `replicate_embedding` (the AllGather writes exactly the shard
bytes into its rank slot); no arithmetic is performed on the gathered table.

### `diagnostics.engram_ahead`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:526`. It starts the next
step's Engram gather at the end of the current step so the reads overlap the draft. The gather
target and the staging slots are the ones the next step would use anyway; starting it earlier
only changes when the same file bytes land. Gate:
`tp4_forward::spec::tests::pooled_engram_reads_land_every_task_s_bytes`
(`crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:2019`).

### `diagnostics.engram_direct`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:196`
(`EngramReader::open`). It reopens the Engram table with `O_DIRECT` instead of the buffered
file. The read starts at the same aligned offset and returns the same bytes; only the kernel
page-cache path differs. Gate: code argument (`reopen_direct` checks the alignment and keeps
the file region); the bytes are the file's.

### `diagnostics.engram_split`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:2120`. It splits the
Engram `wkv` output width over the world ranks, each rank writing its own contiguous column
slice, then exchanges and permutes the slices back. Output element `(row, col)` is a dot over
that column only, so slicing columns changes which rank computes an output, never the
accumulation order or the operands. Gate:
`tp4_forward::spec::tests::sharded_rows_permute_into_the_unsharded_rows`
(`crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:2081`).

### `diagnostics.engram_staged`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:415`. It selects
`opt_rows_engram_wkv_s` (activations staged in shared memory, two outputs a warp) over
`opt_rows_engram_wkv`. The rows comment at `rows.rs:113` says "Both give the same bits", and
the kernel comment at `kernels/opt_rows_sm121.cu:2239-2246` writes the chain out: the 16
products are taken in order from zero on the mixed-precision FMA, the pair sum is with lane
`l ^ 1`, the scaled add is into the same even lane, and the warp tree is the same, so every
output is the single-row kernel's bits. Gate: that kernel comment, checkable by reading
`or_engram_wkv_staged` against `opt_rows_engram_wkv`.

### `diagnostics.engram_uring`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1026` (`Reads::decode`).
It chooses io_uring or the reader-thread pool; both fill the same staging slots from the same
file regions. Gate:
`tp4_forward::spec::tests::direct_windows_land_the_same_bytes_as_reads_through_the_cache`
(`spec.rs:1934`) and `pooled_engram_reads_land_every_task_s_bytes` (`spec.rs:2019`).

### `diagnostics.head_screen`, `diagnostics.head_screen_check`

The argument for these two fields, and the gates that hold it, are in
[v1-head-screen-neutral.md](v1-head-screen-neutral.md).

### `diagnostics.lean_head`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17128` and used at `:18291`.
For every head position but the last it calls `TailStage::screen_device_argmax`
(`tp4_tail.rs:439`) instead of `screen_device` plus a host gather. Both paths launch the same
`cert_lm_head` kernel with the same arguments (`tp4_tail.rs:459-473` and `:542-556`), then
pick the argmax: the lean path takes the block maxima/argmax with a strict `>` over ascending
chunks, and each block's argmax is "the lowest on a block tie" (`tp4_tail.rs:247`), then the
same strict `>` over ascending ranks; `argmax_of` picks the lowest id on ties
(`tp4_tail.rs:118-140`). The two totals are therefore the same (value, id) pair. The lean path
omits the logits readback, not the arithmetic, and the DeepSeek device path serves greedy text
only (docs/NUMERICS.md section 7: evidence `text`, "the path serves no logprobs"). Gate: the
code equality above, plus `tp4_forward::tests::the_lean_head_record_names_the_written_alias`
(`crates/sparkflash-deepseek-forward/src/tp4_forward.rs:21403`) for the one record it rewrites.
Caveat: this would not be neutral on a path that serves logprobs; it is neutral for the
registered DeepSeek text configuration.

### `diagnostics.prefetch_wqb`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:1694`. It adds the
`wq_b` weight span to the list passed to `prefetch_l2` (`rows.rs:1638`, a side-stream
launch of `opt_rows_prefetch_l2`). That kernel writes nothing: the programmatic-dependent-launch
test lists it among the kernels exempt from a wait precisely because it "writes nothing"
(`crates/sparkflash-kernels/tests/pdl_entry.rs:27`). A cache hint cannot change a value later
read from the same address. Gate: that `WRITES_NOTHING` list and the kernel body
(`kernels/opt_rows_sm121.cu:2038`).

### `diagnostics.prefill_decoder_skip`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:534`, used by
`prefill_layers` (`batch.rs:1265`) and the bulk chunk split (`batch.rs:1650`). Layers 0..=20
are the encoder: they write every cache a later row reads past its window
(`batch.rs:104-108`, `SWA_ENCODER_LAYERS`). Layers 21..=39 keep only their 128-row window, so a
chunk that ends before the prompt's last `DECODER_WARM` rows can skip them; `skip_rows`
(`rows.rs:1007`) counts those rows as appended, and the last `DECODER_WARM` rows rebuild every
decoder window and the drafter's window (`batch.rs:110-123`). The logits, the drafter's seeds and
the windows decode reads are the full run's, as docs/design.md:111 states. Gate:
`tp4_forward::batch::tests::only_the_prompt_s_last_rows_run_the_decoder_layers`
(`batch.rs:2919`) holds the boundary, and
`a_resume_reads_only_the_full_run_s_windows` (`batch.rs:2960`) holds the resume rule.

### `diagnostics.prefill_sp`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17299`, used by
`enable_prefill_sp` (`tp4_forward.rs:13178`). It turns a bulk chunk's mHC seam into a
ReduceScatter over the ranks' row blocks plus an AllGather of the finished rows, instead of every
rank computing every row. Each row's post and seam is still computed once, by one rank, from the
same rank inputs (the ReduceScatter at `rows.rs:1201` and the AllGather at `:1288`); it changes
which rank posts a row, not which row gets posted. docs/optimization-strategy.md:445 records
sequence-parallel hyper-connections as bit-identical. Gate: the per-row split in `sp_rows`
(`rows.rs:1456-1470`) plus the documented invariance.

### `diagnostics.rows_groups`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:456` and used at
`:2484`. It batches the completed compressor groups of every lane into one set of launches
(`rows_groups`, `rows.rs:3024`). The batched kernels run `reference_seam`'s per-group chain for
each group; the group's members and their ring positions are unchanged, so the same pool, key and
append arithmetic runs. Gate: the function's comment and body (the per-group chain is the
single-row one; the switch only decides whether it runs grouped).

### `diagnostics.rows_indexer`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:2834`. It enables the
batched row-local GEMVs of the compressor seam and indexer select; off falls back to
`reference_seam`/`ref_select` per row. The function comment (`rows.rs:2804-2810`) states every
row's chain is the M = 1 kernel's, and the batch is a row-parallel launch, not a reduction across
rows. Gate: the comment plus `rows_seam_prepare`'s launches (each row feeds its own ring slot
and query columns).

### `diagnostics.rows_ring`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:432`. It selects the
ring-form host-weight GEMV kernels (`opt_rows_hostw_*_ring`) for the compressor's projections,
which write into a lane's position ring instead of a contiguous row buffer. The kernel is the
same `spark_hw_gemv_m1_body` as the non-ring form; only the output address is computed from the
row table and `position % ring_rows` (`kernels/opt_rows_sm121.cu:2495-2535`). Gate: the kernel
source; the arithmetic body is textually shared.

### `diagnostics.rows_select`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:2556`. It permits the
batched select over a segment's rows; off uses the per-row select. Both run
`opt_select_rows`' body: `select_rows_body` (`kernels/opt_select_sm121.cu:676-718`), whose
`fast_select` ranks each position by the same total order (score descending, position ascending)
and then takes the top-k. The batched path passes a score stride and a row count, not a
different comparator. Gate: the shared body in `opt_select_sm121.cu` and the comment at
`:663-674` ("its chosen positions land in `selected[y]` ascending...").

### `diagnostics.row_table`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:2176` (and the drafter's
copy at `draft.rs:1072`). It builds the per-row `{lane offset, position}` table and passes it
to kernels that then address a row's ring/keys by its own lane and position. Off, each block
computes the same address from `p0 + y`; the table changes only how the address is supplied.
Gate: `upload_row_table` (`rows.rs:2164-2200`) writes offsets/positions, and the consuming
kernel source uses them solely as addresses.

### `diagnostics.seam_split`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:375`. It selects
`opt_rows_mhc_partials_rb{1,2,4}` + `opt_rows_mhc_tail` over `opt_rows_mhc_seam`. The field's
own doc comment (`rows.rs:91-94`) states the two-launch form is "bit-identical to it, the
partials sharing each `fn_w` read across 1, 2 or 4 rows", and the kernel comment at
`kernels/opt_rows_sm121.cu:472-477` repeats that; the tail sums each row's 80 partials in the
same order as the single-launch tail. Gate: those two comments and the shared
`or_seam_tail`.

### `diagnostics.seam_warm`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:5941`. It runs the mHC seam pass
once before the first request. The pass touches the same buffers the first real step overwrites
and produces no persistent output; the comment at `:5915-5927` records an identity-gate PASS
with the pass on and off. Gate: the call site returns only a bool and the warm pass writes no
served state; the recorded identity run is the measurement.

### `diagnostics.seg_blocks`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:773`
(`seg_blocks()`), used as the CTA cap for `opt_select_scores_seg`. `scores_seg_body`
(`kernels/opt_select_sm121.cu:960-1035`) partitions the reachable positions into disjoint tiles
and each block writes its own tiles with a grid stride; the number of blocks changes which block
computes a tile, never the score's chain. Gate: the kernel's tile loop and the per-position
write.

### `diagnostics.select_together`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:2590`. It batches several
segments' selects through `opt_select_scores_segs` + `opt_select_rows_tab` instead of the
per-segment `opt_select_scores_seg`/`opt_select_rows`. The kernel header says a piece's scores
are `opt_select_scores_seg`'s over its rows with a different slice count
(`kernels/opt_select_sm121.cu:1025-1038`), and `opt_select_rows_tab` calls the same
`select_rows_body`/`fast_select` with the row table supplying each row's position
(`:709-716`). The score chain is the reference's (same per-head dot, bf16 rounds, ascending head
fold; `opt_select_sm121.cu:1-38`), and `fast_select` is a rank over the same total order. Gate:
those kernel comments and the shared bodies; the switch moves where the score is computed, not
the operations.

## Drafter

### `drafter.row_table`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/draft.rs:1072`. It builds the
drafter pass's row table; off, each block computes its own row's address. As with the model
row table, the value is used only as an address (lane offset and position) and the pass's per-row
chain is unchanged. Gate: code argument at `draft.rs:1060-1080` and the consuming
`opt_draft_*` kernels' row-table parameters.

### `drafter.seed_split`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/draft.rs:599`. It splits the
seed's `main_proj` columns over the ranks instead of keeping the whole projection on every rank.
The result is a drafter proposal, and every proposed token is verified against the target's own
logits: `sparkflash_model::spec::verdict` keeps the leading run and always the anchor
(`crates/sparkflash-model/src/spec.rs:17-45`), so a reassociated seed can change which tokens are
proposed, not which tokens are emitted. Gate:
`sparkflash_model::spec::tests::the_verdict_keeps_the_leading_run_and_always_the_anchor`
(`spec.rs:260`).

### `drafter.seqs`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/draft.rs:291-293`. It clamps the
number of sequences one drafter pass serves (`DrafterProfile::seqs`); the pass width changes how
many proposals are produced per launch, not the per-sequence draft chain. Every proposal is
verified as above. Gate: `the_verdict_keeps_the_leading_run_and_always_the_anchor`
(`spec.rs:260`) plus the pass-width clamp at `draft.rs:291`.

## Prefill

### `prefill.deadline_ms`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:177` (inside
`DecodeShare::from_env`). It is the first-token deadline `DecodeShare::allowed` uses to decide
how many decoders' ticks a long prefill chunk may take. It changes when a chunk runs, not what a
row computes. Gate: the scheduler contract at `batch.rs:33-43` (every rows-step row is computed
as if alone, with the sampler's noise a function of seed/position/id), and docs/NUMERICS.md
section 1's batch/concurrency invariance.

### `prefill.decode_share`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:176`. It is the cap on the
decoders' share of the tick while long prompts prefill, used only by `DecodeShare::allowed`.
Same argument as the deadline: it orders the work, and the row arithmetic is independent of the
order. Gate: `batch.rs:33-43` and section 1's batch/concurrency invariance.

### `prefill.read_threads`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:162`
(`prefill_read_threads`). It sizes the `ReadPool` that issues a bulk chunk's Engram reads. Each
read fills its own staging offset; a worker takes reads from a shared task, and neither the
destination nor the bytes depend on how many workers drain the queue. Gate:
`tp4_forward::spec::tests::pooled_engram_reads_land_every_task_s_bytes` (`spec.rs:2019`) and
`direct_windows_land_the_same_bytes_as_reads_through_the_cache` (`spec.rs:1934`).

### `prefill.tick_ahead`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1843`. Rank 0 either
decides the next tick after the current one ends or waits for its frame at every step. It is a
scheduling handshake; the admitted work and the per-row arithmetic are the same. Gate: the
scheduler contract at `batch.rs:33-43` and section 1's batch/concurrency invariance.

### `prefill.warmup`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/spec.rs:1824`. On, it runs
`Batch::warm_up` before rank 0 binds HTTP; off, the first request pays the lazy module loads.
The warm-up writes no served state that a real step does not overwrite. Gate: code argument at
`serve_ticks` (the call is before the tick loop and its result is discarded except for the
timing print).

### `prefill.warmup_steps`

Read at `crates/sparkflash-cli/src/device_chain.rs:2288`. It is used only with
`sparkflash_kernels::cuda_driver::stage_timing_set_warm(step < warmup_steps)`
(`device_chain.rs:2313`) to keep cold steps out of the ledger's steady-state average. It does
not enter any forward call. Gate: the read site and the single use as a timing flag.

## Runtime

### `runtime.arena_hostmapped`

Read at `crates/sparkflash-deepseek-forward/src/tp4_resident.rs:873`. It selects
`AllocationMode::UnifiedSingle` instead of `Device` for the resident arena. The same converted
rank file is loaded, the same kernels run, and the bytes are the same; only where the driver
places them changes. Gate: `mode` at `tp4_resident.rs:874-880` is passed to
`CudaArenaAllocator` and no arithmetic reads it.

### `runtime.arena_mmap`

Read at `crates/sparkflash-deepseek-forward/src/tp4_resident.rs:868`. It selects
`AllocationMode::MmapRegistered`, mapping the converted rank file instead of copying it into a
device allocation. The payload is the file's, and the tensor slices/digests are taken from the
same header. Gate: the branch at `converted.rs:704-722` (mmap vs upload) produces the same
`base` bytes for the same file; the arena evidence records the mode only.

### `runtime.constrained`

Read at `crates/sparkflash-cli/src/serve/device.rs:524`, where it gates the `constrained`
capability; a request that asks for `response_format` is refused when the capability is false
(`crates/sparkflash-api/src/server.rs:2213`, with the refusal text at `:859-882`) rather than
silently served unmasked. It is a capability switch, not an arithmetic setting: it does not
change the unconstrained greedy references, and the constrained path is a request-time mask on
top of the same forward. Gate: the refusal rule at `server.rs:859-882` and `2213`; there is no
silent fallback to unconstrained decoding.

### `runtime.lane_memory_mb`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/lanes.rs:570`. It is the per-lane
device-memory budget; when a step does not fit, the scheduler frees lanes or preempts the newest
sequence, which recomputes its tokens when it resumes. The budget changes admission and
recomputation, not a row's arithmetic. Gate: `batch.rs:33-43` (every rows-step row computed as
if alone) and section 1's batch/concurrency invariance.

### `runtime.mem_floor_gib`

Read at `crates/sparkflash-deepseek-forward/src/tp4_resident.rs:756`
(`mem_available_floor_bytes`). It only sets the `MemAvailable` floor checked before the arena
allocation; below it the process refuses to start instead of risking an OOM kill. No value
reaches a kernel. Gate: `check_mem_available_once_for_plan` at `tp4_resident.rs:738-752`.

### `runtime.verify_weights`

Read at `crates/sparkflash-deepseek-forward/src/tp4_resident.rs:869`, passed as
`ConvertedLoadOptions::verify_weights` and used at
`crates/sparkflash-device/src/converted.rs:724`. On, `verify_converted_file` re-reads every
tensor digest and the file digest after the arena is loaded; a mismatch returns an error, so the
run refuses rather than serving different weights. It does not alter a loaded byte. Gate:
`converted.rs:724-726` and the `weights_verified` field in the load report.

## Speculation

### `speculation.batch_spec`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:149`
(`batch_spec_floor`). Off, several decoding sequences take the plain step; on, they share one
speculative step whose rows are each sequence's own `[anchor, drafts]` segment. Each sequence
then accepts exactly as `spec_step` does (`batch.rs:2525-2530`), through the one
`sparkflash_model::spec::verdict`. The shared step changes which rows are in flight together,
not which token each row picks or which drafts are kept. Gate:
`sparkflash_model::spec::tests::the_verdict_keeps_the_leading_run_and_always_the_anchor`
(`spec.rs:260`) and section 1's "speculation never changes the output".

### `speculation.batch_spec_floor`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:152`. It is the least
calibrated chain probability a draft needs for a shared verify row; a draft below the floor is
simply not verified, and an unverified draft is never emitted (the anchor is always kept and the
kept run stops at the first unverified row). Gate: `share_rows` (`batch.rs:2985` test
`rows_go_to_the_likeliest_drafts_while_they_pay`) and the `verdict` rule above.

### `speculation.depth_policy`

Read at `crates/sparkflash-model/src/speculation.rs:125`. It picks the adaptive or fixed depth
policy, which decides how many of the k drafts a step verifies
(`DepthPolicy::depth`, `speculation.rs:153`). The anchor and the leading matching run are the
only emitted tokens, so a different depth changes cost and which drafts are checked, not the
accepted sequence. Gate:
`sparkflash_model::speculation::tests::a_policy_chooses_the_depth_and_the_room_caps_it`
(`speculation.rs:284`) and `spec::verdict`.

### `speculation.drafts`

Read at `crates/sparkflash-cli/src/serve.rs:4421` by the block-drafter serve arm (the DeepSeek
DSpark arm sizes its chain with `speculation.dspark_k` instead). It is the number of tokens a
greedy request proposes after each token. Every proposal is a draft: the verify step keeps the
leading run the target itself picks and always keeps the anchor (`spec.rs:17-45`). Gate:
`the_verdict_keeps_the_leading_run_and_always_the_anchor` (`spec.rs:260`).

### `speculation.dspark_k`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17214`. It is the number of
DSpark drafts per step (`k`), validated to `1..=draft::BLOCK` and passed to
`spec::run_spec`/`spec::serve`. More or fewer drafts change the proposal chain only; the verify
rows and the anchor come from the target. Gate: the `verdict` rule above; `run_spec` uses the
same acceptance loop as the non-DSpark path (`spec.rs:1701-1730`).

### `speculation.dspark_steps`

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17329`. It caps how many tokens a
generation may produce (`max_new`, `usize::MAX` default) and is passed to `serve`/`run_spec`.
It decides when generation stops, not what any step computes; the per-token result is independent
of the cap. Gate: the value is the `max_new` parameter at `spec.rs:1707`, and `cut_step`
(`spec.rs:1654-1676`) only compares `emitted + tokens.len()` with it to return
`DeviceFinish::Length`.

### `speculation.min_depth`

Read at `crates/sparkflash-model/src/speculation.rs:129` (`with_min_depth`) and applied in
`DepthPolicy::depth` at `speculation.rs:173`. It raises the minimum number of drafts a step
verifies, never lowers it; verifying more drafts cannot change the accepted sequence for the same
reason the depth policy cannot. Gate:
`sparkflash_model::speculation::tests::the_minimum_depth_holds_and_is_capped_at_k`
(`speculation.rs:303`) and `spec::verdict`.

## What these entries do not claim

Every field above is either (a) a launch/read/scheduling choice whose kernels are written against
the path they replace, (b) a check or refusal that runs beside the served path, or (c) a
speculation knob whose proposals are verified by `spec::verdict`. None of them is justified by
the word "diagnostic" alone. The fields that could not be justified this way are reported as
findings in the change review rather than listed here.

## The settings this report does not call neutral

The declaration check's list is a waiver, so the entries below are the ones whose justification did
not survive: three of them change the arithmetic and are now in
`sparkflash_model::numerics::ARITHMETIC_FIELDS`, where setting one withdraws the claim instead of
waiving it. They are recorded here because the review that decided them belongs with this file.

### `diagnostics.layer_device` - arithmetic, moved

Read at `crates/sparkflash-deepseek-forward/src/run_header.rs:167` and passed to
`select_layer_arm`, which chooses the host arm or the device arm. The two are different kernel and
collective paths; D56 records measured device-versus-host logit drift, and no gate holds them to the
same bits. A profile that sets it changes the emitted arithmetic, so a reference captured without it
does not describe the process. It is in `ARITHMETIC_FIELDS` now.

### `diagnostics.forward_evidence` - arithmetic, moved

Read at `crates/sparkflash-deepseek-forward/src/run_header.rs:172`. `select_layer_arm`
(`crates/sparkflash-deepseek-forward/src/layer_device.rs:1684-1697`) forces the host arm whenever
evidence recording is on, regardless of `layer_device`, so this setting reaches the same arm choice
by another name. Same disposition for the same reason.

### `diagnostics.prefix_cache` - arithmetic, moved

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs:522`. [docs/NUMERICS.md](../NUMERICS.md)
section 7 records that DeepSeek's device path breaks the resume invariance: on a warm server seven of
eight suite prompts produce different greedy text than after boot, and with
`SPARKFLASH_PREFIX_CACHE=0` every pass agrees. A setting whose on-state is the documented exception
to an invariance is the opposite of a neutral one. It is in `ARITHMETIC_FIELDS` now.

### `diagnostics.prefix_reuse_min_rows` - arithmetic, moved

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward/batch.rs` in `place()`
(`SPARKFLASH_PREFIX_REUSE_MIN_ROWS`, 256 by default). It decides which requests resume from a
lane's prefix, so it reaches the same resume invariance as `diagnostics.prefix_cache`: on the
device path a resumed request does not always produce the greedy text of a cold prefill
([docs/NUMERICS.md](../NUMERICS.md) section 7). A value of `0` reuses every match (the
pre-`!197` behaviour) and therefore widens that exception rather than removing it.

### `runtime.converted_dir` - not arithmetic, but not a serving setting either

Read at `crates/sparkflash-deepseek-forward/src/tp4_resident.rs:887`. It is a filesystem path, and it
does not change the arithmetic of a fixed input: the same converted payload read from another
directory produces the same bits. What it changes is *which* payload is served, so the reference
describes the checkpoint the path named when it was captured, not this setting. `runtime.verify_weights`
would notice a corrupt file, not a different one. Keeping an input selector in a serving profile is a
design smell the gate cannot see; it stays neutral because the arithmetic for the named input is
invariant, and the report says so rather than implying the setting is inert.

### `runtime.rows`, `runtime.rows_from` - harness knobs, not serving settings

Read at `crates/sparkflash-deepseek-forward/src/tp4_forward.rs:17091` and `:17369`. They gate the
teacher-forced multi-row measurement branch, which prints per-row argmax and reaches
`std::process::exit(0)` at `:17413`: a profile that sets either one stops serving and runs a
measurement instead. The rows-step itself is written as the M=1 exact chain
(`crates/sparkflash-deepseek-forward/src/tp4_forward/rows.rs:15-30`), so the per-position arithmetic is
very likely identical - but a setting that prevents serving at all cannot be covered by a serving
reference, and this is the wrong list for it. It stays neutral for the arithmetic claim and is called
out here as a harness knob that belongs outside the serving profile.
