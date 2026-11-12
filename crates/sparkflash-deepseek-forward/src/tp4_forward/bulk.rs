// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The bulk prefill: a prompt chunk of more than `R_MAX` rows through the rows step, with the
//! GEMVs on the tensor cores (kernels/opt_mma_sm121.cu, through `launch_opt`) and the stages the
//! rows step runs per row replaced by chunk-wide launches: the compressor (every completed group of
//! the chunk at once), the indexer (scores on the FP4 tensor cores, one select launch over the
//! rows), the router and the routed experts (grouped by expert, so each expert's slices are read
//! about once per chunk).
//!
//! The results are not bit-identical to the M = 1 step (the GEMMs accumulate in another order and
//! the routed partials round through bf16), so the prefill driver only takes this path for a long
//! uncached prompt and SPARKFLASH_PREFILL_EXACT=1 keeps every prompt on the exact rows step.

use super::rows::RowsState;
use super::*;
use sparkflash_kernels::cuda_driver::{HostMapped, Kernel, Module};
use sparkflash_kernels::hostw_gemv_launch::Mode;
use sparkflash_kernels::moe_routed_device::{BulkRows, MoeBulk};
use sparkflash_kernels::mxfp8_gemv_launch::{OptArgs, OptKind, launch_opt};

/// Whether [Self::bulk_index] appends the completed key positions to the persistent fragment-order
/// store instead of re-packing the whole history, read from `SPARKFLASH_INDEX_PACK_APPEND`.
///
/// **On by default. `SPARKFLASH_INDEX_PACK_APPEND=0` restores the full re-pack**, which is also the
/// A/B arm. The pack kernel's addressing is on the append's side - its fragment address is a function
/// of the absolute key index from a multiple of eight, which `index_pack_start` guarantees - and the
/// saving is 3.1x at 7,538 tokens, 4.0x at 30,246 and 3.4x at 60,488, with three identical requests
/// an arm returning the re-pack's own sha at every length (`d12c925e`, `4fdfb215`, `f87f6de6`).
///
/// It took three fixes to get there, and each was found by the next prompt length: the packed scratch
/// had to be one slot per kv-source owner (`BulkState::k_codes`), and the watermark and the repeat
/// memory had to live on the chunk state rather than on the layer the several chunk states share
/// (`BulkState::index_packed`). The arm that disagreed looked *better* (6.9x at 30,246) because a
/// shared watermark was skipping packs it should not have.
///
/// **What the default can and cannot touch.** `bulk_index` is reached only through
/// `bulk_seam_select`, whose call site filters on `n > R_MAX`, so a decode or verify step never packs
/// and this switch cannot reach one. A prefill's *first* chunk starts at position zero (or at the
/// resumed prefix), where the scheduler calls `BulkState::reset_index` and `index_pack_have`
/// returns nothing to trust, so it packs every key it has - exactly the bytes the re-pack arm would
/// write. Only the chunks after it in one prefill append, which is what the four measured arms
/// are.
#[must_use]
fn index_pack_append_from(value: Option<&str>) -> bool {
    value != Some("0")
}

/// [index_pack_append_from] over the profile (`prefill.index_pack_append`).
#[must_use]
fn index_pack_append() -> bool {
    index_pack_append_from(
        sparkflash_model::active_profile()
            .prefill
            .index_pack_append
            .as_deref(),
    )
}

/// Whether [Self::bulk_index] prints the bytes it asks the pack kernel for, read from
/// `SPARKFLASH_INDEX_PACK_LOG`.
///
/// This is issue #20 WS2's real gate: the packed bytes a chunk costs must be constant in `p0`, not
/// merely smaller, so the number has to be printed at several prompt lengths rather than inferred
/// from a rate.
#[must_use]
fn index_pack_log_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// [index_pack_log_from] over the profile (`prefill.index_pack_log`).
#[must_use]
fn index_pack_log() -> bool {
    index_pack_log_from(
        sparkflash_model::active_profile()
            .prefill
            .index_pack_log
            .as_deref(),
    )
}

/// Whether this call would pack exactly what the previous call at the same boundary packed.
///
/// The index-source layers do not map one-to-one onto kv-source owners: of the eight in a decoder
/// chunk, five are owner 3 (`owner_list=[0,1,2,3,3,3,3,3]`), so five consecutive calls run the pack
/// kernel over what looks like the same source range into the same destination.
///
/// **The first version of the skip moved the text, and the cause was the memory, not the pack.**
/// With the dedupe on, two consecutive warm requests on one prompt returned `418f6a1e…` then
/// `a6461b79…` (and at 30,246 tokens three different texts in three requests), where the same binary
/// with it off reproduced `418f6a1e…` three times. The checksum probe then showed the five calls
/// packing identical bytes: the boundary `(p0, n_max)` is a property of the prompt, so a repeat
/// memory that outlived its chunk matched the next request's walk and skipped packs it had not run.
/// `pack_scope_is_new` scopes it to one chunk pass; the append's own divergence was the same shape
/// one level up (the store and its watermark outliving the sequence, see `index_pack_have`).
#[must_use]
fn index_pack_repeated(
    last: Option<(u64, usize, usize)>,
    seq: Option<u64>,
    p0: usize,
    n_max: usize,
) -> bool {
    match (last, seq) {
        (Some((packed_for, p0_at, n_at)), Some(seq)) => {
            packed_for == seq && p0_at == p0 && n_at == n_max
        }
        _ => false,
    }
}

/// Whether the chunk now being indexed is a different one from the chunk the repeat memory belongs
/// to, in which case that memory must be dropped before any pack is skipped.
///
/// **This is the half `index_pack_repeated` was missing.** The boundary `(p0, n_max)` is a property
/// of the prompt, not of the call, so a second request over the same prompt walks the same
/// boundaries as the first, and a memory that survives the chunk compares a repeat across requests.
/// Run 20261111T161121Z caught it: on the third of three identical 7,538-token requests, layers 2, 8
/// and 14 (owners 0, 1, 2) reported `packed=false` and the verify probe read owner 3's bytes
/// (`head=d70dceb3511ea7a1`) out of the scratch they had not written - they scored against the
/// previous request's keys, and the text moved where the first two requests had agreed.
///
/// The only genuine repeat is the five index layers that share one owner *inside one chunk pass*:
/// with the scope cleared, the earlier call that set `(p0, n_max)` is necessarily one of them, and a
/// run with the checksum on shows those five packing identical bytes.
#[must_use]
fn pack_scope_is_new(
    scope: Option<(usize, usize, usize)>,
    layer: usize,
    p0: usize,
    n: usize,
) -> bool {
    match scope {
        // The same chunk, and a layer later in the same walk: the only shape a real repeat has.
        Some((p0_at, n_at, last)) => p0_at != p0 || n_at != n || layer <= last,
        None => true,
    }
}

/// Whether [Self::bulk_index] skips what looks like a repeated pack at the same boundary, read from
/// `SPARKFLASH_INDEX_PACK_DEDUPE`.
///
/// **Off by default until its fixed arm has a device A/B.** It was measured to change the text
/// (d14fcf8a), and the cause has since been found and fixed: not the skip - the five owner-3 layers
/// pack identical bytes, measured - but the repeat memory outliving the chunk, so a second request
/// over one prompt skipped packs and scored the previous request's keys. [pack_scope_is_new] now
/// scopes that memory to one chunk pass, and the checksum probe is how the fix is checked: the
/// owner-sharing layers must print `packed=false` with the first one's bytes, and no layer of
/// another owner may ever print `packed=false`.
#[must_use]
fn index_pack_dedupe_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// [index_pack_dedupe_from] over the profile (`prefill.index_pack_dedupe`).
#[must_use]
fn index_pack_dedupe() -> bool {
    index_pack_dedupe_from(
        sparkflash_model::active_profile()
            .prefill
            .index_pack_dedupe
            .as_deref(),
    )
}

/// Whether [Self::bulk_index] hashes the packed store after each pack, read from
/// `SPARKFLASH_INDEX_PACK_VERIFY`. A synchronising copy, so it is not part of the byte log.
#[must_use]
fn index_pack_verify_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// [index_pack_verify_from] over the profile (`prefill.index_pack_verify`).
#[must_use]
fn index_pack_verify() -> bool {
    index_pack_verify_from(
        sparkflash_model::active_profile()
            .prefill
            .index_pack_verify
            .as_deref(),
    )
}

/// The watermark a chunk may trust: zero unless the store was packed by *this* sequence and the
/// chunk continues the pack that sequence left.
///
/// The watermark lives on the chunk state, and one state serves every sequence of a serving epoch
/// (and every resumed sequence, whose first chunk starts at a nonzero `p0`), so a bare position is
/// not enough to identify what the store holds. The checkpoint records the sequence it packed for,
/// and a chunk appends only when that sequence is its own *and* its `p0` is the token end the pack
/// reached (the watermark counts compressed positions, so a contiguous chunk starts at
/// `watermark * ratio`).
///
/// Both halves are load-bearing. Without the position check, a resumed sequence at a boundary its
/// host happened to reach would append onto the host's keys. Without the identity check, two
/// interleaved prefills do it: A packs to 4,096 tokens, B (a 4,096-token prompt) is placed and
/// packs its own keys, then A's next chunk at `p0 = 4,096` passes `p0 == watermark * ratio` and
/// appends only its tail - the head stays B's, and A's queries are scored against B's keys with no
/// error and no refusal. `BulkState::reset_index` clears both on placement, and
/// `BulkState::set_index_seq` clears them when another sequence takes the state.
#[must_use]
fn index_pack_have(
    have: Option<(u64, usize)>,
    seq: Option<u64>,
    p0: usize,
    n_max: usize,
    ratio: usize,
) -> usize {
    match (have, seq) {
        (Some((packed_for, watermark)), Some(seq))
            if packed_for == seq && p0 != 0 && n_max >= watermark && p0 == watermark * ratio =>
        {
            watermark
        }
        _ => 0,
    }
}

/// The key positions an append from `have` must start at: `have` rounded down to the fragment
/// layout's super-block.
///
/// The pack kernel's `frag = 1` address is `(key >> 3) * 512 + (key & 7) * 64 + c * 4`, which is a
/// function of the absolute key index only while the first key of a launch is a multiple of eight —
/// the layout groups eight keys into 512 bytes, and `key >> 3` does not decompose linearly
/// otherwise. So an append re-packs up to seven positions it already packed, which is idempotent and
/// keeps the addressing exact. The chunk boundaries make this a no-op today (a 128-token chunk at
/// ratio 4 is 32 keys and at ratio 1 is 128), and the rounding is what makes that a fact about the
/// code rather than about the scheduler.
#[must_use]
fn index_pack_start(have: usize) -> usize {
    have - have % 8
}

/// What one layer's call at `(p0, n_max)` does to the packed-key store and its bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexPack {
    /// The dedupe arm skips this call: an earlier layer of the same chunk packed the same bytes.
    repeated: bool,
    /// The watermark the call may trust (zero means the call packs the whole history).
    have: usize,
    /// Where the pack starts, rounded to the layout's eight-key super-block.
    start: usize,
    /// Key positions the call asks the kernel for.
    appended: usize,
    /// The watermark the owner holds after the call.
    watermark: Option<(u64, usize)>,
    /// The boundary the owner last packed at after the call.
    at: Option<(u64, usize, usize)>,
}

/// One layer's pack call: the chunk's boundary and the two switches that shape it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackCall {
    /// The chunk's first token position.
    p0: usize,
    /// The compressed key positions the chunk reaches.
    n_max: usize,
    /// The model's compress ratio.
    ratio: usize,
    /// `prefill.index_pack_append`.
    append: bool,
    /// `prefill.index_pack_dedupe`.
    dedupe: bool,
}

/// Decides one layer's pack from the stored watermark, the repeat memory and this call's boundary.
///
/// A skip changes nothing: the earlier owner-sharing layer in the same chunk pass already wrote
/// those bytes and set the watermark, so clearing it here would cost the *next* chunk of a deduped
/// run its append (the two switches must compose).
#[must_use]
fn index_pack_plan(
    stored: Option<(u64, usize)>,
    at: Option<(u64, usize, usize)>,
    seq: Option<u64>,
    call: PackCall,
) -> IndexPack {
    let PackCall {
        p0,
        n_max,
        ratio,
        append,
        dedupe,
    } = call;
    let repeated = dedupe && index_pack_repeated(at, seq, p0, n_max);
    let have = if append && !repeated {
        index_pack_have(stored, seq, p0, n_max, ratio)
    } else {
        0
    };
    let start = index_pack_start(have);
    let appended = if repeated {
        0
    } else {
        n_max.saturating_sub(start)
    };
    let latest = seq.map(|seq| (seq, p0, n_max));
    IndexPack {
        repeated,
        have,
        start,
        appended,
        watermark: if repeated {
            stored
        } else {
            seq.map(|seq| (seq, n_max))
        },
        at: if repeated { at } else { latest },
    }
}

/// The byte offsets an append starting at `have` writes to, from the store's base: the codes and the
/// scales.
///
/// One key is `index_dim` values at four bits, so `index_dim / 2` code bytes, and
/// `index_dim / 32` scale bytes.
#[must_use]
fn index_pack_offsets(have: usize, index_dim: usize) -> (usize, usize) {
    let first_block = have * index_dim / 32;
    (first_block * 16, first_block)
}

/// The indexer scores buffer, in bf16 scores: a chunk whose rows times compressed positions exceed
/// it is scored and selected in row slices.
const SCORE_ENTRIES: usize = 64 << 20;
/// opt_mma_index_scores' rows per block (one per warp) and 128-key tiles per block.
const SCORE_ROWS: usize = 8;
const SCORE_TILES: usize = 8;
/// opt_select_rows' blocks.
const SELECT_BLOCKS: usize = 96;
/// opt_select_sm121.cu's FAST_MAX_POS: the most compressed positions the select serves.
const SELECT_MAX_POSITIONS: usize = 1 << 21;

pub(super) struct BulkKernels {
    _modules: Vec<Module>,
    pack: Kernel,
    scores: Kernel,
    /// opt_mma_hostw_{bf16w, bf16w_f32o, f32w_f32o}, by hostw_gemv_launch::Mode, and their
    /// bf16-activation forms opt_mma_hostw_bf16a_*.
    hostw: [Kernel; 3],
    hostw_bf16a: [Kernel; 3],
    /// opt_mma_bf16a_f32o: a split chunk's shared expert gate/up from its gathered bf16 rows.
    pub(super) bf16a_f32o: Kernel,
    router_dots: Kernel,
    router_topk: Kernel,
    /// opt_mma_fp8_rows_bf16o: the Engram wkv (row-major E4M3 weights).
    engram_wkv: Kernel,
    /// opt_bulk_attn: the sparse attention with the 16 heads on the tensor cores.
    pub(super) attn: Kernel,
    /// opt_bulk_mhc_{partials, tail}: the mHC seam over the chunk; opt_bulk_mhc_post_partials:
    /// the post before a seam fused with that seam's partials.
    pub(super) seam_partials: Kernel,
    pub(super) seam_tail: Kernel,
    pub(super) post_partials: Kernel,
    /// opt_bulk_seam_derive: a split seam's codes and scales (and, given a buffer, f32 rows) from
    /// its gathered bf16 rows.
    pub(super) seam_derive: Kernel,
    /// opt_bulk_engram_{embed, post}: the Engram stage around the wkv over the chunk.
    engram_embed: Kernel,
    engram_post: Kernel,
    select_rows: Kernel,
    pool_rows: Kernel,
    k_norm_rows: Kernel,
    rope: Kernel,
}

/// The routed-expert rows and scratch the interleaved bulk states share, so one pass over the
/// expert weights serves all their chunks (DeviceLayer::rows_moe_merged): way i's FFN codes and
/// scales, expert ids and routing weights are rows i * stride .. of one buffer each, and the
/// grouping, SwiGLU codes and down partials (MoeBulk) span every way. The ways' MoE launches all
/// run on the run's stream, so a way's own pass may use the scratch too.
pub(super) struct MoeShare {
    /// The rows a way holds (its rows state's row buffers).
    pub(super) stride: usize,
    pub(super) ways: usize,
    /// Every way's rows from way 0's.
    pub(super) rows: BulkRows,
    sorted: Deviceptr,
    items: Deviceptr,
    items_count: Deviceptr,
    codes: Deviceptr,
    scales: Deviceptr,
    slot: Deviceptr,
    allocations: Vec<Deviceptr>,
}

/// A bulk state's place in a [MoeShare].
#[derive(Clone, Copy, Debug)]
pub(super) struct MoeWay {
    pub(super) way: usize,
    pub(super) stride: usize,
    /// The share's rows from way 0's.
    pub(super) base: BulkRows,
}

impl MoeShare {
    /// Carves `ways` ways of `stride` rows for `device`'s geometry.
    pub(super) fn open(
        device: &DeviceLayer<'_>,
        stride: usize,
        ways: usize,
    ) -> Result<Self, String> {
        let cuda = device.cuda;
        let dim = device.dim;
        let top_k = device.routed_geometry.top_k;
        let tokens = stride * ways;
        let mut allocations = Vec::new();
        let mut total = 0usize;
        let mut alloc = |bytes: usize, zero: bool| -> Result<Deviceptr, String> {
            let ptr = cuda.alloc(bytes.max(16), &mut total)?;
            if zero {
                cuda.copy_htod(&ptr, &vec![0u8; bytes.max(16)])?;
            }
            allocations.push(ptr);
            Ok(ptr)
        };
        let rows = BulkRows {
            act_codes: alloc(tokens * dim, true)?.0,
            act_scales: alloc(tokens * dim / 32, true)?.0,
            ids: alloc(tokens * top_k * 4, true)?.0,
            weights: alloc(tokens * top_k * 4, true)?.0,
        };
        let [sorted, items, items_count, codes, scales, slot] = MoeBulk::buffer_bytes(tokens);
        Ok(Self {
            stride,
            ways,
            rows,
            sorted: alloc(sorted, false)?,
            items: alloc(items, false)?,
            items_count: alloc(items_count, false)?,
            codes: alloc(codes, false)?,
            scales: alloc(scales, false)?,
            slot: alloc(slot, false)?,
            allocations,
        })
    }

    /// Way `way`'s place, refused past the share.
    pub(super) fn way(&self, way: usize) -> Result<MoeWay, String> {
        if way >= self.ways {
            return Err(format!("bulk share: way {way} of {}", self.ways));
        }
        Ok(MoeWay {
            way,
            stride: self.stride,
            base: self.rows,
        })
    }

    pub(super) fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
    }
}

impl MoeWay {
    /// The way's own rows of the share for `dim` wide rows and `top_k` experts a row.
    pub(super) fn rows(&self, dim: usize, top_k: usize) -> BulkRows {
        let first = self.way * self.stride;
        BulkRows {
            act_codes: self.base.act_codes + (first * dim) as u64,
            act_scales: self.base.act_scales + (first * dim / 32) as u64,
            ids: self.base.ids + (first * top_k * 4) as u64,
            weights: self.base.weights + (first * top_k * 4) as u64,
        }
    }
}

/// The bulk chunk's kernels and its chunk-wide scratch (the row buffers are the [`RowsState`]'s,
/// carved at the chunk capacity).
pub(super) struct BulkState {
    pub(super) k: BulkKernels,
    pub(super) moe: MoeBulk,
    /// The compressor rows of the chunk, `[cap + 16][head_dim]` f32: row i is position
    /// p0 - lead + i, the first `lead` rows the open group's rows copied from the ring.
    wkv: u64,
    wgate: u64,
    /// The chunk's completed groups: pooled f32, narrowed bf16, normed (then roped) bf16, f32.
    pooled: u64,
    latent_bf16: u64,
    latent_normed: u64,
    latent_f32: u64,
    /// The groups' index keys: f32, then normed and roped bf16.
    wk_f32: u64,
    wk_bf16: u64,
    /// The rows' index queries as packed E2M1 codes (two per byte) and ue8m0 scales.
    q_codes: u64,
    q_scales: u64,
    /// Each kv-source owner's index keys as packed E2M1 codes in MMA fragment order
    /// (opt_bulk_fp4_pack) and scales, `max_positions` keys a slot.
    ///
    /// **A slot per owner, because the append arm writes only the tail.** The four owners pack into
    /// this scratch in turn - 0, 1, 2, then 3 five times - and a full re-pack rewrites the whole
    /// buffer every time, so one buffer served them all. The append packs only
    /// `[start, n_max)` and leaves `[0, start)` as it found it, which with one buffer is the
    /// *previous owner's* keys: measured, 102,184 blocks against the re-pack's 705,512 (6.9x fewer)
    /// and three requests returning `ce2136e8`, `8e50040f`, `8e50040f` where the re-pack returns
    /// `d12c925e` three times. **A slot per owner fixed that at 7,538 tokens but was not the whole
    /// story**: at 30,246 tokens the append arm reproduced itself (`b476c13c` three times) and
    /// differed from the re-pack arm (`4fdfb215` three times), because the store outlived its
    /// sequence. The per-sequence reset and the continuity check close that; the append is the
    /// default and reproduces the re-pack's text.
    k_codes: Vec<u64>,
    k_scales: Vec<u64>,
    /// How many compressed key positions each kv-source owner has already packed into this chunk
    /// state's scratch, one per `kv_source_layers` entry.
    ///
    /// **Per chunk state, not per layer.** A prefill runs several bulk states at once (three by
    /// default), each with its own `k_codes`/`k_scales`, and a watermark kept on the layer is shared
    /// between them: the second state is told a position is packed when what was packed is in the
    /// first state's buffer. That is the divergence the 30,246-token append arm measures.
    /// `(sequence, compressed key positions packed for it)` per kv-source owner. The sequence is
    /// part of the record because a position alone does not say *whose* keys the store holds (see
    /// `index_pack_have`).
    pub(super) index_packed: std::cell::RefCell<Vec<Option<(u64, usize)>>>,
    /// The `(sequence, p0, n_max)` each owner last packed at in this state, for the repeat check.
    pub(super) index_packed_at: std::cell::RefCell<Vec<Option<(u64, usize, usize)>>>,
    /// The `(p0, n, layer)` of the walk `index_packed_at` belongs to (see `pack_scope_is_new`).
    pub(super) index_pack_scope: std::cell::Cell<Option<(usize, usize, usize)>>,
    /// The sequence `index_packed` and `index_packed_at` are about; `None` before the first
    /// placement. Set by `set_index_seq`, which clears the three when it changes.
    pub(super) index_seq: std::cell::Cell<Option<u64>>,
    scores: u64,
    /// opt_select_rows' scratch, `select_words` u32 per block: a row's keys and chosen flags
    /// past the shared-memory table (n + n / 4 words for n compressed positions).
    select_scratch: u64,
    select_words: usize,
    /// The router's raw dots, `[cap][experts]` f32.
    pub(super) dots: u64,
    /// The chunk's DSpark taps, `[cap][3][dim]` f32 (the rows state's tap sink).
    pub(super) taps: u64,
    /// A split chunk's last-row exchange (rows::DeviceLayer::sp_finish): per rank a slot of one
    /// row's stream then its pre mix, f32.
    pub(super) sp_last: u64,
    /// With Engram: the two pinned buffers a chunk's table reads land in (a chunk's reads run
    /// under the chunk before it), and the live flags `[slots][positions]` the post kernel reads
    /// through the mapping (host-written as each slot's layer is queued; by position, so a chunk
    /// never rewrites a flag a queued launch reads).
    pub(super) engram_staging: Option<[HostMapped; 2]>,
    /// With Engram: a sharded chunk's exchanged shares, `[world][cap]` rows of a share each, every
    /// column once (spec::EngramShard): (address, bytes).
    pub(super) engram_shares: Option<(u64, usize)>,
    engram_live: Option<HostMapped>,
    allocations: Vec<Deviceptr>,
}

impl BulkState {
    /// Loads the bulk kernels (installing the tensor-core GEMMs `launch_opt` runs above 8 tokens)
    /// and carves the chunk scratch for `cap` rows (the routed experts' from `share` when given).
    pub(super) fn open(
        device: &DeviceLayer<'_>,
        kernel_dir: &Path,
        cap: usize,
        share: Option<&MoeShare>,
    ) -> Result<Self, String> {
        let cuda = device.cuda;
        let mut modules = Vec::new();
        let (mma, _) = cached_cubin(&kernel_dir.join("opt_mma_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        modules.push(sparkflash_kernels::mxfp8_gemv_launch::install_mma(
            cuda, &mma,
        )?);
        let (select, _) = cached_cubin(&kernel_dir.join("opt_select_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        let (misc, _) = cached_cubin(&kernel_dir.join("opt_misc_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        let (rows_cubin, _) = cached_cubin(&kernel_dir.join("opt_rows_sm121.cubin"))
            .map_err(|error| format!("bulk: {error}"))?;
        let mut load = |cubin: &[u8], name: &'static str| -> Result<Kernel, String> {
            let (module, kernel) = cuda.load_named(cubin, name)?;
            modules.push(module);
            Ok(kernel)
        };
        // The routed-expert plan is the shared image (kernels/moe_plan_sm121.cu): Qwen's token
        // path, GLM's bulk prefill and this one count and order their routed slots with one
        // implementation.
        if sparkflash_kernels::images::moe_plan_cubin().is_empty() {
            return Err("bulk: this build embeds no routed-expert plan cubin".to_owned());
        }
        let moe_plan = sparkflash_kernels::images::moe_plan_cubin();
        let moe_kernels = [
            load(moe_plan, "moe_plan")?,
            load(&mma, "opt_mma_moe_gate_up")?,
            load(&mma, "opt_mma_moe_down")?,
            load(&mma, "opt_mma_moe_combine")?,
        ];
        let k = BulkKernels {
            pack: load(&mma, "opt_bulk_fp4_pack")?,
            scores: load(&mma, "opt_mma_index_scores")?,
            hostw: [
                load(&mma, "opt_mma_hostw_bf16w")?,
                load(&mma, "opt_mma_hostw_bf16w_f32o")?,
                load(&mma, "opt_mma_hostw_f32w_f32o")?,
            ],
            hostw_bf16a: [
                load(&mma, "opt_mma_hostw_bf16a_bf16w")?,
                load(&mma, "opt_mma_hostw_bf16a_bf16w_f32o")?,
                load(&mma, "opt_mma_hostw_bf16a_f32w_f32o")?,
            ],
            bf16a_f32o: load(&mma, "opt_mma_bf16a_f32o")?,
            router_dots: load(&mma, "opt_mma_router_dots")?,
            router_topk: load(&mma, "opt_mma_router_topk")?,
            engram_wkv: load(&mma, "opt_mma_fp8_rows_bf16o")?,
            attn: load(&mma, "opt_bulk_attn")?,
            seam_partials: load(&rows_cubin, "opt_bulk_mhc_partials")?,
            seam_tail: load(&rows_cubin, "opt_bulk_mhc_tail")?,
            post_partials: load(&rows_cubin, "opt_bulk_mhc_post_partials")?,
            seam_derive: load(&rows_cubin, "opt_bulk_seam_derive")?,
            engram_embed: load(&misc, "opt_bulk_engram_embed")?,
            engram_post: load(&misc, "opt_bulk_engram_post")?,
            select_rows: load(&select, "opt_select_rows_bf16")?,
            pool_rows: load(&misc, "opt_pool_rows")?,
            k_norm_rows: load(&misc, "opt_k_norm_rows")?,
            rope: load(&misc, "opt_rope_heads_bf16")?,
            _modules: modules,
        };
        let head_dim = device.head_width as usize;
        let index_dim = device.index_head_dim;
        let index_cols = device.index_n_heads * index_dim;
        let mut allocations = Vec::new();
        let mut total = 0usize;
        let mut alloc = |bytes: usize| -> Result<u64, String> {
            let ptr = cuda.alloc(bytes.max(16), &mut total)?;
            allocations.push(ptr);
            Ok(ptr.0)
        };
        let moe = match share {
            Some(share) => MoeBulk {
                plan: moe_kernels[0],
                gate_up: moe_kernels[1],
                down: moe_kernels[2],
                combine: moe_kernels[3],
                sorted: share.sorted,
                items: share.items,
                items_count: share.items_count,
                codes: share.codes,
                scales: share.scales,
                slot: share.slot,
                max_tokens: share.stride * share.ways,
                plan_threads: sparkflash_model::active_profile().plan.block_threads(),
            },
            None => {
                let [sorted, items, items_count, codes, scales, slot] = MoeBulk::buffer_bytes(cap);
                MoeBulk {
                    plan: moe_kernels[0],
                    gate_up: moe_kernels[1],
                    down: moe_kernels[2],
                    combine: moe_kernels[3],
                    sorted: Deviceptr(alloc(sorted)?),
                    items: Deviceptr(alloc(items)?),
                    items_count: Deviceptr(alloc(items_count)?),
                    codes: Deviceptr(alloc(codes)?),
                    scales: Deviceptr(alloc(scales)?),
                    slot: Deviceptr(alloc(slot)?),
                    max_tokens: cap,
                    plan_threads: sparkflash_model::active_profile().plan.block_threads(),
                }
            }
        };
        let ring_rows = cap + 16;
        // One packed-key slot per kv-source owner: the append arm must leave the head its own.
        let owners = device.kv_source_layers.len();
        // A row reaches at most max_positions compressed positions (ratio 1).
        let select_words = device.max_positions + device.max_positions.div_ceil(4);
        let engram_staging = match device.engram_staging(cap)? {
            Some(first) => match device.engram_staging(cap) {
                Ok(Some(second)) => Some([first, second]),
                other => {
                    cuda.free_host_mapped(&first);
                    other?;
                    None
                }
            },
            None => None,
        };
        let engram_shares = match device.engram.as_ref() {
            Some(engram) => {
                let layout = engram.layout;
                let bytes = cap * layout.columns() * (layout.head_dim() + layout.scale_width());
                Some((alloc(bytes)?, bytes))
            }
            None => None,
        };
        let engram_live = match device.engram.as_ref() {
            Some(engram) => {
                Some(cuda.alloc_host_mapped(engram.layer_ids.len() * engram.positions, &mut 0)?)
            }
            None => None,
        };
        let state = Self {
            k,
            moe,
            wkv: alloc(ring_rows * head_dim * 4)?,
            wgate: alloc(ring_rows * head_dim * 4)?,
            pooled: alloc(cap * head_dim * 4)?,
            latent_bf16: alloc(cap * head_dim * 2)?,
            latent_normed: alloc(cap * head_dim * 2)?,
            latent_f32: alloc(cap * head_dim * 4)?,
            wk_f32: alloc(cap * index_dim * 4)?,
            wk_bf16: alloc(cap * index_dim * 2)?,
            q_codes: alloc(cap * index_cols)?,
            q_scales: alloc(cap * index_cols / 32)?,
            k_codes: (0..owners)
                .map(|_| alloc(device.max_positions * index_dim / 2))
                .collect::<Result<Vec<_>, String>>()?,
            k_scales: (0..owners)
                .map(|_| alloc(device.max_positions * index_dim / 32))
                .collect::<Result<Vec<_>, String>>()?,
            index_packed: std::cell::RefCell::new(vec![None; owners]),
            index_packed_at: std::cell::RefCell::new(vec![None; owners]),
            index_pack_scope: std::cell::Cell::new(None),
            index_seq: std::cell::Cell::new(None),
            scores: alloc(SCORE_ENTRIES * 2)?,
            select_scratch: alloc(SELECT_BLOCKS * select_words * 4)?,
            select_words,
            dots: alloc(cap * device.routed_geometry.experts * 4)?,
            taps: alloc(cap * 3 * device.dim * 4)?,
            sp_last: alloc(super::rows::GATHER_WORLD * (device.hc_mult + 1) * device.dim * 4)?,
            engram_staging,
            engram_shares,
            engram_live,
            allocations,
        };
        Ok(state)
    }

    pub(super) fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
        for pinned in self
            .engram_staging
            .iter()
            .flatten()
            .chain(&self.engram_live)
        {
            cuda.free_host_mapped(pinned);
        }
    }

    /// Forgets the packed-key store when a new sequence is placed on this chunk state.
    ///
    /// The store holds only the most recent sequence's keys, and a sequence can start at a
    /// nonzero position (a resumed prefix, or a lane reused by a later request). `p0` alone
    /// cannot tell that first chunk apart from the previous sequence's continuation, so the
    /// scheduler clears both the positions and the identity where it chooses a sequence.
    /// [Self::set_index_seq] then covers the other half: chunks of different sequences that
    /// interleave through the state, including one that resumes exactly at another's boundary.
    pub(super) fn reset_index(&self) {
        for at in self.index_packed.borrow_mut().iter_mut() {
            *at = None;
        }
        for at in self.index_packed_at.borrow_mut().iter_mut() {
            *at = None;
        }
        self.index_pack_scope.set(None);
        self.index_seq.set(None);
    }

    /// Names the sequence the packed-key store now belongs to, forgetting the previous one's when
    /// it changes.
    ///
    /// The scheduler calls this at the start of every bulk chunk: a chunk state is shared by every
    /// sequence of a serving epoch, and interleaved prefills walk through it in turn. The identity
    /// is what lets `index_pack_have` tell "my own earlier chunk" from "another request's keys at
    /// the position I am about to resume at"; without it a shorter prompt's pack at the boundary a
    /// longer prompt had reached is mistaken for the longer prompt's own.
    pub(super) fn set_index_seq(&self, seq: u64) {
        if self.index_seq.get() != Some(seq) {
            self.reset_index();
            self.index_seq.set(Some(seq));
        }
    }

    /// One of the bf16-body tensor-core GEMMs (opt_mma_sm121.cu's OptArgs list) over `args`:
    /// 128 x 128 output tiles, the column tiles fastest (launch_opt's f32 / bf16 form).
    pub(super) fn mma(&self, cuda: &Cuda, kernel: Kernel, args: &OptArgs) -> Result<(), String> {
        cuda.launch(
            kernel,
            (
                args.cols.div_ceil(128),
                args.tokens.div_ceil(128),
                args.groups,
            ),
            (256, 1, 1),
            &[
                args.a,
                args.a_scales,
                args.w,
                args.ws,
                args.out,
                u64::from(args.cols),
                u64::from(args.inner),
                u64::from(args.tokens),
                args.a_tok_stride,
                args.out_tok_stride,
                args.a_grp_stride,
                args.w_grp_stride,
                args.s_grp_stride,
                args.out_grp_stride,
                u64::from(args.k_stride_blocks),
            ],
        )
    }

    /// The hostw kernel of `mode` over f32 rows, or bf16 rows with `bf16a`.
    fn hostw_kernel(&self, mode: Mode, bf16a: bool) -> Kernel {
        let index = match mode {
            Mode::Bf16Weight => 0,
            Mode::Bf16WeightF32Stores => 1,
            Mode::F32WeightF32Stores => 2,
        };
        if bf16a {
            self.k.hostw_bf16a[index]
        } else {
            self.k.hostw[index]
        }
    }
}

impl DeviceLayer<'_> {
    /// The chunk's normed rows for the host-weight projections: a split chunk's gathered bf16 rows
    /// (ffn_x until its FFN seam, [Self::sp_rows]), else the f32 rows; true for the bf16 form.
    fn bulk_normed(&self, rows: &RowsState, n: usize) -> (u64, bool) {
        match self.sp_rows(rows, n) {
            Some(_) => (rows.b.ffn_x, true),
            None => (rows.b.normed, false),
        }
    }

    /// A host-weight projection of `tokens` f32 (or, by the kernel, bf16) rows (`[tokens][inner]`)
    /// on the tensor cores.
    #[allow(clippy::too_many_arguments)]
    fn bulk_hostw(
        &self,
        kernel: Kernel,
        a: u64,
        w: u64,
        out: u64,
        cols: usize,
        inner: usize,
        tokens: usize,
    ) -> Result<(), String> {
        self.cuda.launch(
            kernel,
            (cols.div_ceil(128) as u32, tokens.div_ceil(128) as u32, 1),
            (256, 1, 1),
            &[
                a,
                w,
                out,
                cols as u64,
                inner as u64,
                tokens as u64,
                inner as u64,
                cols as u64,
            ],
        )
    }

    /// opt_rope_heads_bf16 over `tokens` rows of `heads` heads, token t at table position
    /// `first + t * step`.
    #[allow(clippy::too_many_arguments)]
    fn bulk_rope(
        &self,
        bulk: &BulkState,
        rows: u64,
        width: usize,
        heads: usize,
        tokens: usize,
        first: usize,
        step: usize,
    ) -> Result<(), String> {
        let pairs = self.rope_pairs as usize;
        let table_bytes = (self.max_positions * pairs * 4) as u64;
        let tables = self.buffers.workspace.rope_tables_compressed.device;
        let at = (first * pairs * 4) as u64;
        self.cuda.launch(
            bulk.k.rope,
            (heads as u32, tokens as u32, 1),
            (32, 1, 1),
            &[
                rows,
                tables + at,
                tables + table_bytes + at,
                width as u64,
                u64::from(self.rope_head_dim),
                heads as u64,
                (pairs * step) as u64,
                0,
            ],
        )
    }

    /// The compressor and the indexer select of the chunk's rows p0..p0+n (the bulk twin of
    /// `rows_seam_select`).
    pub(super) fn bulk_seam_select(
        &mut self,
        rows: &RowsState,
        bulk: &BulkState,
        layer: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let ratio = self.compress_ratio.get(layer).copied().unwrap_or(0);
        if ratio == 0 {
            return Ok(());
        }
        if self.is_kv_source(layer) {
            self.bulk_compress(rows, bulk, layer, ratio, p0, n)?;
        }
        if self.is_index_source(layer) {
            self.bulk_index(rows, bulk, layer, ratio, p0, n)?;
        }
        Ok(())
    }

    /// wkv / wgate over the rows, then every group the chunk completes: the pool, the latent norm,
    /// the index key (wk, k_norm, rope, FP4 round trip) and the roped latent's store rows. The last
    /// rows go back to the ring, so the decode steps after the chunk pool their open group as the
    /// M = 1 path would have left it.
    fn bulk_compress(
        &mut self,
        rows: &RowsState,
        bulk: &BulkState,
        layer: usize,
        ratio: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        use sparkflash_kernels::engram_convert_launch::{
            self as convert, EngramConvertParams, EngramConvertSpans,
        };
        let refusal = |what: &str| format!("bulk compressor: layer {layer}: {what}");
        let (Some(staged), Some(ring)) = (self.seam_staged, self.seam_ring) else {
            return Err(refusal("no staged compressor"));
        };
        let owner = self
            .owner_index_for(layer)
            .ok_or_else(|| refusal("no kv-source owner"))?;
        let (wkv, wkv_mode) = staged.wkv.ok_or_else(|| refusal("no staged wkv"))?;
        let hd = ring.head_dim;
        let row_bytes = hd * 4;
        let lead = p0 % ratio;
        let owner_base = owner * ring.rows * row_bytes;
        let ring_row = |position: usize| (owner_base + (position % ring.rows) * row_bytes) as u64;
        let cuda = self.cuda;
        // Copies scratch rows [from, from + count) to or from the ring rows of positions
        // p0 - lead + from.., one contiguous run at a time.
        let ring_copy = |scratch: u64, ring_base: u64, from: usize, count: usize, to_ring: bool| {
            let mut i = from;
            while i < from + count {
                let position = p0 - lead + i;
                let slot = position % ring.rows;
                let run = (ring.rows - slot).min(from + count - i);
                let s = Deviceptr(scratch + (i * row_bytes) as u64);
                let r = Deviceptr(ring_base + ring_row(position));
                let (dst, src) = if to_ring { (r, s) } else { (s, r) };
                cuda.copy_dtod_async(&dst, &src, run * row_bytes)?;
                i += run;
            }
            Ok::<(), String>(())
        };
        let mut projections = vec![(wkv, wkv_mode, bulk.wkv, ring.wkv.0)];
        if let Some((wgate, mode)) = staged.wgate {
            projections.push((wgate, mode, bulk.wgate, ring.wgate.0));
        }
        let (normed, bf16a) = self.bulk_normed(rows, n);
        for &(weight, mode, scratch, ring_base) in &projections {
            ring_copy(scratch, ring_base, 0, lead, false)?;
            self.bulk_hostw(
                bulk.hostw_kernel(mode, bf16a),
                normed,
                weight.0,
                scratch + (lead * row_bytes) as u64,
                hd,
                self.dim,
                n,
            )?;
        }
        let g0 = p0 / ratio;
        let groups = (p0 + n) / ratio - g0;
        if groups > 0 {
            let unnormed = if ratio == 1 {
                bulk.wkv
            } else {
                if staged.wgate.is_none() {
                    return Err(refusal("a pooled ratio without wgate"));
                }
                cuda.launch(
                    bulk.k.pool_rows,
                    (hd.div_ceil(128) as u32, groups as u32, 1),
                    (128, 1, 1),
                    &[bulk.wkv, bulk.wgate, bulk.pooled, ratio as u64, hd as u64],
                )?;
                bulk.pooled
            };
            let count = groups * hd;
            let narrow = self.seam_narrow.ok_or_else(|| refusal("no narrow"))?;
            let widen = self.seam_widen.ok_or_else(|| refusal("no widen"))?;
            let norm = self.seam_norm.ok_or_else(|| refusal("no norm"))?;
            let to_bf16 = |src: u64, dst: u64| {
                convert::launch_narrow(
                    cuda,
                    narrow,
                    &EngramConvertParams {
                        src: Deviceptr(src),
                        dst: Deviceptr(dst),
                        count: count as u32,
                    },
                    &EngramConvertSpans {
                        src_bytes: count * 4,
                        dst_bytes: count * 2,
                    },
                )
            };
            let to_f32 = |src: u64, dst: u64| {
                convert::launch_widen(
                    cuda,
                    widen,
                    &EngramConvertParams {
                        src: Deviceptr(src),
                        dst: Deviceptr(dst),
                        count: count as u32,
                    },
                    &EngramConvertSpans {
                        src_bytes: count * 2,
                        dst_bytes: count * 4,
                    },
                )
            };
            to_bf16(unnormed, bulk.latent_bf16)?;
            let (norm_weight, norm_bytes) = staged
                .compressor_norm
                .ok_or_else(|| refusal("no compressor norm"))?;
            if norm_bytes != hd * 2 {
                return Err(refusal("the compressor norm is not one bf16 row"));
            }
            sparkflash_kernels::rmsnorm_launch::launch(
                cuda,
                norm,
                &sparkflash_kernels::rmsnorm_launch::RmsNormLaunchParams {
                    x: Deviceptr(bulk.latent_bf16),
                    weight: norm_weight,
                    out: Deviceptr(bulk.latent_normed),
                    rows: groups as u32,
                    cols: hd as u32,
                    eps: self.eps,
                    trace_ring: 0,
                    layer_id: layer as u32,
                    step_seq: p0 as u64,
                },
            )?;
            // The index keys, from the pre-rope normed latent.
            let wk = staged.index_wk.ok_or_else(|| refusal("no indexer wk"))?;
            let (k_norm, _) = staged
                .index_k_norm
                .ok_or_else(|| refusal("no indexer k_norm"))?;
            let fp4_rt = self
                .seam_fp4_round_trip
                .ok_or_else(|| refusal("no FP4 round trip"))?;
            let index_dim = self.index_head_dim;
            self.bulk_hostw(
                bulk.hostw_kernel(Mode::Bf16WeightF32Stores, true),
                bulk.latent_normed,
                wk.0,
                bulk.wk_f32,
                index_dim,
                hd,
                groups,
            )?;
            cuda.launch(
                bulk.k.k_norm_rows,
                ((groups as u32).div_ceil(128), 1, 1),
                (128, 1, 1),
                &[
                    bulk.wk_f32,
                    k_norm.0,
                    bulk.wk_bf16,
                    index_dim as u64,
                    u64::from(self.eps.to_bits()),
                    groups as u64,
                ],
            )?;
            self.bulk_rope(bulk, bulk.wk_bf16, index_dim, 1, groups, g0 * ratio, ratio)?;
            sparkflash_kernels::pool_group_launch::launch_fp4_round_trip(
                cuda,
                fp4_rt,
                Deviceptr(bulk.wk_bf16),
                Deviceptr(
                    ring.index_keys.0 + ((owner * self.max_positions + g0) * index_dim * 4) as u64,
                ),
                (groups * index_dim / 32) as u32,
            )?;
            // The compressed store rows, from the roped latent.
            self.bulk_rope(bulk, bulk.latent_normed, hd, 1, groups, g0 * ratio, ratio)?;
            to_f32(bulk.latent_normed, bulk.latent_f32)?;
            let append = self.seam_append.ok_or_else(|| refusal("no append"))?;
            let (code_bytes, scale_bytes) = seam_store_owner_bytes(self.max_positions)?;
            let params = sparkflash_kernels::compressed_append_launch::AppendParams {
                latent: bulk.latent_f32,
                codes: ring.store_codes.0 + (owner * code_bytes) as u64,
                scales: ring.store_scales.0 + (owner * scale_bytes) as u64,
                first_row: g0 as u32,
                row_count: groups as u32,
                ratio: ratio as u32,
            };
            cuda.launch(
                append,
                (groups as u32, 1, 1),
                sparkflash_kernels::compressed_append_launch::BLOCK,
                &params.words(),
            )?;
        }
        let keep = n.min(ring.rows);
        for &(_, _, scratch, ring_base) in &projections {
            ring_copy(scratch, ring_base, lead + n - keep, keep, true)?;
        }
        Ok(())
    }

    /// The indexer over the chunk's rows: the head weights, the roped and FP4 round-tripped
    /// queries, then the scores of every row against the owner's keys on the FP4 tensor cores and
    /// one select launch, into the rows' selected lists (and the candidate masks).
    fn bulk_index(
        &mut self,
        rows: &RowsState,
        bulk: &BulkState,
        layer: usize,
        ratio: usize,
        p0: usize,
        n: usize,
    ) -> Result<(), String> {
        let refusal = |what: &str| format!("bulk indexer: layer {layer}: {what}");
        let n_max = (p0 + n) / ratio;
        if n_max == 0 {
            return Ok(());
        }
        if n_max > SELECT_MAX_POSITIONS {
            return Err(refusal("more compressed positions than the select holds"));
        }
        let (Some(staged), Some(ring)) = (self.seam_staged, self.seam_ring) else {
            return Err(refusal("no staged indexer"));
        };
        let owner = self
            .owner_index_for(layer)
            .ok_or_else(|| refusal("no kv-source owner"))?;
        let weights_proj = staged
            .weights_proj
            .ok_or_else(|| refusal("no weights_proj"))?;
        let (wq_b, wq_b_scales) = staged.indexer_wq_b.ok_or_else(|| refusal("no wq_b"))?;
        let scale_weights = self
            .seam_scale_weights
            .ok_or_else(|| refusal("no weight scale"))?;
        let fp4_rt = self
            .seam_fp4_round_trip
            .ok_or_else(|| refusal("no FP4 round trip"))?;
        let cuda = self.cuda;
        let b = rows.b;
        let (heads, index_dim) = (self.index_n_heads, self.index_head_dim);
        let index_cols = heads * index_dim;
        let (normed, bf16a) = self.bulk_normed(rows, n);
        self.bulk_hostw(
            bulk.hostw_kernel(Mode::Bf16Weight, bf16a),
            normed,
            weights_proj.0,
            b.idx_weights_bf16,
            heads,
            self.dim,
            n,
        )?;
        let factor = ((index_dim as f64).powf(-0.5) * (heads as f64).powf(-0.5)) as f32;
        sparkflash_kernels::pool_group_launch::launch_scale_index_weights(
            cuda,
            scale_weights,
            Deviceptr(b.idx_weights_bf16),
            Deviceptr(b.idx_weights_f32),
            factor,
            (n * heads) as u32,
        )?;
        let lora = self.projections[0].cols as usize;
        launch_opt(
            cuda,
            OptKind::Fp8aBf16o,
            &OptArgs {
                a: b.q_codes,
                a_scales: b.q_scales,
                w: wq_b.0,
                ws: wq_b_scales.0,
                out: b.idx_query,
                cols: index_cols as u32,
                inner: lora as u32,
                tokens: n as u32,
                a_tok_stride: lora as u64,
                out_tok_stride: index_cols as u64,
                groups: 1,
                a_grp_stride: 0,
                w_grp_stride: 0,
                s_grp_stride: 0,
                out_grp_stride: 0,
                k_stride_blocks: 0,
            },
        )
        .map_err(|error| refusal(&format!("wq_b: {error}")))?;
        self.bulk_rope(bulk, b.idx_query, index_dim, heads, n, p0, 1)?;
        sparkflash_kernels::pool_group_launch::launch_fp4_round_trip(
            cuda,
            fp4_rt,
            Deviceptr(b.idx_query),
            Deviceptr(b.idx_q_f32),
            (n * index_cols / 32) as u32,
        )?;
        let pack = |src: u64, codes: u64, scales: u64, blocks: usize, frag: u64| {
            cuda.launch(
                bulk.k.pack,
                ((blocks as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                &[src, codes, scales, blocks as u64, frag],
            )
        };
        pack(
            b.idx_q_f32,
            bulk.q_codes,
            bulk.q_scales,
            n * index_cols / 32,
            0,
        )?;
        // The keys, packed from the persistent f32 store into MMA fragment order.
        //
        // The store is keyed by kv-source owner (four slots) and written by the index-source layers
        // (eight), so the five that map to owner 3 write one slot. They write the *same* bytes: the
        // checksum probe (`index_pack_verify`) reads layers 20, 24, 28, 32 and 36 packing identical
        // head and tail at one boundary, because the keys are a function of the owner's compressed
        // latent and the owner's indexer, not of the layer that triggers them. A skip is therefore
        // byte-safe, and the skipped call leaves the scratch holding what the pack would have
        // written.
        //
        // What is *not* safe is a memory of the boundary that outlives the chunk: `(p0, n_max)` is a
        // property of the prompt, so the next request over the same prompt reproduces it, and a
        // surviving entry made layers 2, 8 and 14 skip and score owner 3's keys (run
        // 20261111T161121Z). `pack_scope_is_new` clears the memory whenever the chunk changes.
        // A repeat is only a repeat inside one chunk pass: the boundary is a property of the prompt,
        // so the next request over the same prompt walks the same boundaries and must not inherit
        // this chunk's memory (see `pack_scope_is_new`).
        // The bookkeeping lives on the chunk state, not on the layer: a prefill runs several bulk
        // states at once (three by default), each with its own packed scratch. A watermark shared
        // between them hands the second state a position the first packed into a *different*
        // buffer, so the head it does not rewrite is empty or stale - which is what the
        // 30,246-token arm measures (see `BulkState::index_packed`).
        if pack_scope_is_new(bulk.index_pack_scope.get(), layer, p0, n) {
            bulk.index_pack_scope.set(Some((p0, n, layer)));
            for at in bulk.index_packed_at.borrow_mut().iter_mut() {
                *at = None;
            }
        }
        let seq = bulk.index_seq.get();
        let plan = index_pack_plan(
            bulk.index_packed.borrow()[owner],
            bulk.index_packed_at.borrow()[owner],
            seq,
            PackCall {
                p0,
                n_max,
                ratio,
                append: index_pack_append(),
                dedupe: index_pack_dedupe(),
            },
        );
        let (start, appended) = (plan.start, plan.appended);
        if appended > 0 {
            let (codes_offset, scales_offset) = index_pack_offsets(start, index_dim);
            pack(
                ring.index_keys.0 + ((owner * self.max_positions + start) * index_dim * 4) as u64,
                bulk.k_codes[owner] + codes_offset as u64,
                bulk.k_scales[owner] + scales_offset as u64,
                appended * index_dim / 32,
                1,
            )?;
        }
        // A skip leaves the record the earlier owner-sharing layer set, so the next chunk of a
        // deduped run still appends.
        bulk.index_packed.borrow_mut()[owner] = plan.watermark;
        bulk.index_packed_at.borrow_mut()[owner] = plan.at;
        debug_assert_eq!(start % 8, 0, "an append starts on the layout's super-block");
        if index_pack_verify() {
            // Is a shared owner a repeated pack? A hash of the packed chunk's first kilobyte and of
            // its last, compared against the previous call at the same `(owner, p0, n_max)`, answers
            // it inside one run: `repeat_of=… same` would mean the five owner-3 index layers pack
            // identical bytes (and the skip is sound), `diff` means each layer packs its own
            // projection and the skip / append watermark is not. It costs a synchronising copy, so it
            // is a separate switch from the byte log.
            let words = (n_max * index_dim / 2).min(1024) as u64;
            let mut head = vec![0u8; words as usize];
            self.cuda
                .copy_dtoh(&mut head, &Deviceptr(bulk.k_codes[owner]))
                .map_err(|error| refusal(&format!("verify head: {error}")))?;
            let mut tail = vec![0u8; words as usize];
            let last = (n_max * index_dim / 2).saturating_sub(words as usize) as u64;
            self.cuda
                .copy_dtoh(&mut tail, &Deviceptr(bulk.k_codes[owner] + last))
                .map_err(|error| refusal(&format!("verify tail: {error}")))?;
            let sum = |bytes: &[u8]| -> u64 {
                bytes.as_chunks::<8>().0.iter().fold(0u64, |acc, word| {
                    acc.rotate_left(5) ^ u64::from_le_bytes(*word)
                })
            };
            let (head, tail) = (sum(&head), sum(&tail));
            let repeat = self
                .index_verify_last
                .get()
                .filter(|&(o, p, m, _, _, _)| (o, p, m) == (owner, p0, n_max))
                .map_or_else(
                    || "first".to_owned(),
                    |(_, _, _, prev, head0, tail0)| {
                        let verdict = if (head0, tail0) == (head, tail) {
                            "same"
                        } else {
                            "diff"
                        };
                        format!("layer={prev} {verdict}")
                    },
                );
            self.index_verify_last
                .set(Some((owner, p0, n_max, layer, head, tail)));
            println!(
                "sparkflash serve: index_pack_verify layer={layer} owner={owner} p0={p0} n_max={n_max} packed={} head={head:016x} tail={tail:016x} repeat_of={repeat}",
                appended > 0,
            );
        }
        if index_pack_log() {
            // The issue's real gate for this workstream: the bytes the pack kernel is asked for, per
            // chunk, at increasing prompt lengths. O(history) shows up as a number that grows with
            // p0; an append shows up as a number that does not.
            println!(
                "sparkflash serve: index_pack layer={layer} owner={owner} p0={p0} n={n} keys={} blocks={} appended_keys={appended}",
                n_max,
                appended * index_dim / 32,
            );
        }
        let mask_mode = match layer.cmp(&self.candidate_source_layer) {
            std::cmp::Ordering::Less => 0u64,
            std::cmp::Ordering::Equal => 2,
            std::cmp::Ordering::Greater => 1,
        };
        let mask_bytes = rows.mask_bytes;
        let stride = n_max.next_multiple_of(8);
        let slice = ((SCORE_ENTRIES / stride) / SCORE_ROWS * SCORE_ROWS).max(SCORE_ROWS);
        let mut y0 = 0;
        while y0 < n {
            let m = slice.min(n - y0);
            let keys = (p0 + y0 + m) / ratio;
            if keys > 0 {
                let masks = b.masks + (y0 * mask_bytes) as u64;
                cuda.launch(
                    bulk.k.scores,
                    (
                        m.div_ceil(SCORE_ROWS) as u32,
                        keys.div_ceil(128 * SCORE_TILES) as u32,
                        1,
                    ),
                    (256, 1, 1),
                    &[
                        bulk.q_codes + (y0 * index_cols / 2) as u64,
                        bulk.q_scales + (y0 * index_cols / 32) as u64,
                        bulk.k_codes[owner],
                        bulk.k_scales[owner],
                        b.idx_weights_f32 + (y0 * heads * 4) as u64,
                        bulk.scores,
                        m as u64,
                        (p0 + y0) as u64,
                        ratio as u64,
                        stride as u64,
                        SCORE_TILES as u64,
                        if mask_mode == 1 { masks } else { 0 },
                        (mask_bytes / 4) as u64,
                    ],
                )?;
                cuda.launch(
                    bulk.k.select_rows,
                    (SELECT_BLOCKS as u32, 1, 1),
                    (1024, 1, 1),
                    &[
                        bulk.scores,
                        b.selected + (y0 * SELECTED_STRIDE * 4) as u64,
                        if mask_mode == 1 { masks } else { 0 },
                        if mask_mode == 2 { masks } else { 0 },
                        bulk.select_scratch,
                        m as u64,
                        (p0 + y0) as u64,
                        ratio as u64,
                        self.index_topk as u64,
                        mask_mode,
                        self.candidate_topk_blocks as u64,
                        self.candidate_block_size as u64,
                        stride as u64,
                        SELECTED_STRIDE as u64,
                        (mask_bytes / 4) as u64,
                        bulk.select_words as u64,
                    ],
                )?;
            }
            y0 += m;
        }
        Ok(())
    }

    /// The Engram wkv of the chunk's rows into `engram_kv` (`[rows][out_width]` bf16), one
    /// tensor-core GEMM against the row-major weights.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bulk_engram_wkv(
        &self,
        rows: &RowsState,
        bulk: &BulkState,
        weight: u64,
        scales: u64,
        out_width: usize,
        inner: usize,
        n: usize,
    ) -> Result<(), String> {
        if !out_width.is_multiple_of(32) || !inner.is_multiple_of(32) {
            return Err(format!(
                "bulk engram wkv: {out_width} x {inner} is not in 32 x 32 blocks"
            ));
        }
        let b = rows.b;
        self.cuda.launch(
            bulk.k.engram_wkv,
            (n.div_ceil(128) as u32, out_width.div_ceil(128) as u32, 1),
            (256, 1, 1),
            &[
                b.engram_codes,
                b.engram_scales,
                weight,
                scales,
                b.engram_kv,
                out_width as u64,
                inner as u64,
                n as u64,
                inner as u64,
                out_width as u64,
            ],
        )
    }

    /// The chunk's Engram stage at `slot` over its rows `first ..` (`live.len()` of them, all of
    /// them but a split chunk's): the embed of every row's arena entry, the wkv, then the gates and
    /// the stream write of the rows `live` marks (three launches).
    pub(super) fn bulk_engram(
        &self,
        rows: &RowsState,
        bulk: &BulkState,
        slot: usize,
        live: &[bool],
        p0: usize,
        first: usize,
    ) -> Result<(), String> {
        let n = live.len();
        let engram = self
            .engram
            .as_ref()
            .ok_or("bulk engram: the arena was not attached")?;
        let mask = bulk
            .engram_live
            .as_ref()
            .ok_or("bulk engram: no live flags")?;
        let at = slot * engram.positions + engram.base(p0) + first;
        if at + n > mask.len() {
            return Err(format!(
                "bulk engram: rows {}..{} are outside the flags",
                p0 + first,
                p0 + first + n
            ));
        }
        // SAFETY: inside the mapping (checked), and no queued launch reads these positions' flags.
        let flags =
            unsafe { std::slice::from_raw_parts_mut(mask.host_ptr().cast::<u8>().add(at), n) };
        for (flag, &on) in flags.iter_mut().zip(live) {
            *flag = u8::from(on);
        }
        let layout = engram.layout;
        let align = crate::engram_device_state::ARENA_ALIGN;
        let stride = layout.stride();
        let b = rows.b;
        self.cuda.launch(
            bulk.k.engram_embed,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                engram.arena.0 + (at * stride) as u64,
                stride as u64,
                (layout.codes_len().div_ceil(align) * align) as u64,
                b.engram_codes,
                b.engram_scales,
                engram.inner as u64,
            ],
        )?;
        self.bulk_engram_wkv(
            rows,
            bulk,
            engram.wkv_weight[slot].0,
            engram.wkv_scales[slot].0,
            engram.out_width,
            engram.inner,
            n,
        )?;
        self.cuda.launch(
            bulk.k.engram_post,
            (n as u32, 1, 1),
            (256, 1, 1),
            &[
                b.stream + (first * self.hc_mult * self.dim * 4) as u64,
                b.engram_kv,
                engram.product[slot].0,
                mask.device().0 + at as u64,
                engram.dim as u64,
                engram.hc_mult as u64,
                u64::from(engram.eps.to_bits()),
                engram.out_width as u64,
            ],
        )
    }

    /// The router over the chunk's rows: the raw gate dots on the tensor cores, then the
    /// per-token scoring and top-k into moe_ids / moe_weights.
    pub(super) fn bulk_router(
        &self,
        rows: &RowsState,
        bulk: &BulkState,
        gate: u64,
        bias: u64,
        n: usize,
    ) -> Result<(), String> {
        let b = rows.b;
        let experts = self.routed_geometry.experts;
        if experts != 384 || self.routed_geometry.top_k != 6 {
            return Err("bulk router: the kernels hold 384 experts, top 6".to_owned());
        }
        self.cuda.launch(
            bulk.k.router_dots,
            ((experts / 128) as u32, (n as u32).div_ceil(128), 1),
            (256, 1, 1),
            &[
                b.ffn_x,
                gate,
                bulk.dots,
                experts as u64,
                self.dim as u64,
                n as u64,
            ],
        )?;
        self.cuda.launch(
            bulk.k.router_topk,
            ((n as u32).div_ceil(8), 1, 1),
            (256, 1, 1),
            &[bulk.dots, bias, b.moe_ids, b.moe_weights, n as u64],
        )
    }
}

/// The selected list's row stride in u32 ids (as the rows step's).
const SELECTED_STRIDE: usize = 512;

#[cfg(test)]
mod index_pack_tests {
    use super::*;

    /// The append is the default and `0` is the A/B arm: three prompt lengths return the re-pack's
    /// own sha with it on (issue #20 WS2a), so a run that sets nothing takes the new path.
    #[test]
    fn the_key_append_is_the_default_and_zero_is_the_arm() {
        assert!(index_pack_append_from(None), "no variable keeps the append");
        assert!(
            index_pack_append_from(Some("")),
            "an empty value keeps the append"
        );
        assert!(index_pack_append_from(Some("1")), "1 keeps the append");
        assert!(
            index_pack_append_from(Some("true")),
            "anything but 0 is the append"
        );
        assert!(!index_pack_append_from(Some("0")), "0 is the re-pack arm");
    }

    /// The gate's log is off unless asked for: it prints a line a chunk.
    #[test]
    fn the_pack_log_is_off_unless_asked() {
        assert!(!index_pack_log_from(None));
        assert!(!index_pack_log_from(Some("0")));
        assert!(!index_pack_log_from(Some("true")));
        assert!(index_pack_log_from(Some("1")));
    }

    /// An append's offsets are the re-pack's offsets for the same positions, which is the property
    /// the whole change rests on: `index_dim` values at four bits is `index_dim / 2` code bytes a
    /// key, the pack kernel writes sixteen bytes a block, and there are four blocks a key at the
    /// released 128-wide row.
    /// The dedupe is opt-in: it was measured to remove determinism the unskipped run has, so a run
    /// that sets nothing keeps every pack.
    #[test]
    fn the_repeat_pack_dedupe_is_opt_in() {
        assert!(
            !index_pack_dedupe_from(None),
            "no variable keeps every pack"
        );
        assert!(!index_pack_dedupe_from(Some("0")), "0 keeps every pack");
        assert!(!index_pack_dedupe_from(Some("true")), "only 1 dedupes");
        assert!(index_pack_dedupe_from(Some("1")), "1 dedupes");
        // The check itself: the same boundary repeats, a moved one does not, and an owner that has
        // not packed is not a repeat.
        assert!(index_pack_repeated(
            Some((7, 51514, 2357)),
            Some(7),
            51514,
            2357
        ));
        assert!(index_pack_repeated(Some((7, 0, 1024)), Some(7), 0, 1024));
        assert!(
            !index_pack_repeated(Some((7, 0, 1024)), Some(7), 0, 4096),
            "a longer chunk reruns"
        );
        assert!(!index_pack_repeated(
            Some((7, 4096, 2541)),
            Some(7),
            4200,
            2541
        ));
        assert!(
            !index_pack_repeated(None, Some(7), 0, 1024),
            "a first pack is not a repeat"
        );
        assert!(
            !index_pack_repeated(Some((7, 0, 1024)), Some(8), 0, 1024),
            "another request over the same prompt walks the same boundary and is not a repeat"
        );
        assert!(
            !index_pack_repeated(Some((7, 0, 1024)), None, 0, 1024),
            "a chunk with no sequence named trusts nothing"
        );
        // And the memory only counts inside the chunk that set it: the next request walks the same
        // boundaries, so a boundary match across chunks is not a repeat of this chunk's pack.
        assert!(pack_scope_is_new(None, 2, 0, 512), "the first chunk is new");
        assert!(
            !pack_scope_is_new(Some((0, 1024, 20)), 24, 0, 1024),
            "the owner-sharing layer later in the same walk is the repeat"
        );
        assert!(
            !pack_scope_is_new(Some((0, 1024, 20)), 36, 0, 1024),
            "and so is the last of the five"
        );
        assert!(
            pack_scope_is_new(Some((0, 1024, 36)), 2, 0, 1024),
            "the layer index going backwards is a new walk - the tail pass over one chunk, which is why request 3 moved before this test existed"
        );
        assert!(
            pack_scope_is_new(Some((0, 1024, 20)), 24, 0, 4096),
            "a longer pass over the same prompt is a new chunk, and so is the next request's"
        );
        assert!(
            pack_scope_is_new(Some((4996, 3769, 36)), 2, 4997, 3769),
            "a moved chunk is new"
        );
    }

    /// A watermark is only trusted for the sequence that set it: a chunk at position zero, one
    /// whose key count is below the watermark, or one that does not continue at the previous
    /// chunk's token end starts from nothing. This is the bug the prose arms caught - after the
    /// boot warm-up the watermark was high, the first real request appended nothing and scored the
    /// warm-up's keys - and the interleaved case the block dependency hid.
    #[test]
    fn a_watermark_is_not_trusted_across_sequences() {
        // A sequence's second chunk trusts what its first one packed: 32 keys at ratio 4 end at
        // token 128, which is this chunk's p0.
        assert_eq!(index_pack_have(Some((7, 32)), Some(7), 128, 64, 4), 32);
        // A chunk at position zero never does, whatever the previous one reached.
        assert_eq!(index_pack_have(Some((7, 4096)), Some(7), 0, 1024, 4), 0);
        // A call with fewer keys than the store holds cannot append: re-pack.
        assert_eq!(index_pack_have(Some((7, 4096)), Some(7), 8192, 1024, 2), 0);
        assert_eq!(index_pack_have(Some((7, 8192)), Some(7), 512, 4096, 1), 0);
        // An unstarted store stays unstarted, and one packed for nobody is not trusted.
        assert_eq!(index_pack_have(Some((7, 0)), Some(7), 512, 4096, 1), 0);
        assert_eq!(index_pack_have(None, Some(7), 512, 4096, 1), 0);
        assert_eq!(index_pack_have(Some((7, 128)), None, 512, 128, 4), 0);
        // Two interleaved prefills: A packs 1,024 tokens (256 keys at ratio 4), B's first chunk
        // packs 512 tokens (128 keys), then A's next chunk at 1,024 must not append onto B's
        // head. A contiguous continuation of A's own first chunk does append.
        assert_eq!(index_pack_have(Some((3, 128)), Some(3), 1024, 256, 4), 0);
        assert_eq!(index_pack_have(Some((3, 256)), Some(3), 1024, 256, 4), 256);
        // The case the position check alone cannot see: B's pack ends exactly where A resumes,
        // so the boundary matches and only the sequence identity refuses the append.
        assert_eq!(index_pack_have(Some((4, 256)), Some(3), 1024, 256, 4), 0);
        // A resumed sequence at p0 = 512 can pass the continuity check after its own earlier
        // pack (the scheduler's reset is what makes the host independent of it).
        assert_eq!(index_pack_have(Some((3, 128)), Some(3), 512, 128, 4), 128);
    }

    /// The two switches compose: a dedupe skip keeps the watermark the earlier owner-sharing
    /// layer set, so the sequence's next chunk still appends, and a skip with the dedupe off is a
    /// real pack from the chunk's start.
    #[test]
    fn a_dedupe_skip_keeps_the_watermark_an_append_continues() {
        let call = |p0, n_max, ratio, append, dedupe| PackCall {
            p0,
            n_max,
            ratio,
            append,
            dedupe,
        };
        // Five owner-3 layers at one chunk boundary (p0 = 0, 256 keys): the first appends from
        // nothing, the next four are skips that must not clear what it recorded.
        let first = index_pack_plan(None, None, Some(7), call(0, 256, 4, true, true));
        assert!(!first.repeated);
        assert_eq!((first.have, first.start, first.appended), (0, 0, 256));
        assert_eq!(first.watermark, Some((7, 256)));
        let skip = index_pack_plan(
            first.watermark,
            first.at,
            Some(7),
            call(0, 256, 4, true, true),
        );
        assert!(skip.repeated, "the later owner-sharing layer is the repeat");
        assert_eq!((skip.have, skip.appended), (0, 0));
        assert_eq!(
            skip.watermark,
            Some((7, 256)),
            "a skip must not clear the watermark"
        );
        // The next chunk then appends from it rather than re-packing the history.
        let next = index_pack_plan(
            Some((7, 256)),
            Some((7, 0, 256)),
            Some(7),
            call(1024, 512, 4, true, true),
        );
        assert!(!next.repeated);
        assert_eq!((next.have, next.start, next.appended), (256, 256, 256));
        // The same call with the dedupe off packs the chunk's whole history (p0 = 0, so the
        // continuity check does not trust the watermark) and records it.
        let no_dedupe = index_pack_plan(
            Some((7, 256)),
            Some((7, 0, 256)),
            Some(7),
            call(0, 256, 4, true, false),
        );
        assert!(!no_dedupe.repeated);
        assert_eq!((no_dedupe.have, no_dedupe.appended), (0, 256));
        assert_eq!(no_dedupe.watermark, Some((7, 256)));
        // Another sequence's store is never trusted, whatever boundary it reached.
        let other = index_pack_plan(
            Some((4, 256)),
            Some((4, 0, 256)),
            Some(3),
            call(1024, 512, 4, true, false),
        );
        assert_eq!((other.have, other.appended), (0, 512));
        // The append switched off re-packs everything.
        let repack = index_pack_plan(
            Some((7, 256)),
            Some((7, 0, 256)),
            Some(7),
            call(1024, 512, 4, false, false),
        );
        assert_eq!((repack.have, repack.appended), (0, 512));
    }

    /// An append starts on the fragment layout's eight-key super-block, because `key >> 3` and
    /// `key & 7` in the pack kernel's address only decompose linearly from a multiple of eight. The
    /// rounding may re-pack up to seven positions it already packed, which is idempotent.
    #[test]
    fn an_append_starts_on_the_layouts_eight_key_super_block() {
        for have in 0usize..64 {
            let start = index_pack_start(have);
            assert_eq!(start % 8, 0, "{have} aligned");
            assert!(start <= have, "{have}: never past the watermark");
            assert!(have - start < 8, "{have}: at most seven re-packed");
        }
        // The chunk boundaries a prefill actually produces: 128 tokens at ratio 4 (32 keys) and at
        // ratio 1 (128), and eight of them. None of them need the rounding.
        for have in [0usize, 32, 64, 128, 256, 1024, 4096, 32768] {
            assert_eq!(index_pack_start(have), have, "{have} is already aligned");
        }
    }

    #[test]
    fn an_appends_offsets_are_where_a_repack_would_have_written() {
        let index_dim = 128;
        for have in [0usize, 1, 3, 8, 63, 64, 1000, 4095] {
            let (codes, scales) = index_pack_offsets(have, index_dim);
            assert_eq!(codes, have * index_dim / 2, "codes: {have} keys at 64 B");
            assert_eq!(scales, have * index_dim / 32, "scales: {have} keys at 4");
        }
        // The packed store's per-key width is half the f32 key's, so an append starts at the
        // position the full re-pack would have reached.
        let (codes, _) = index_pack_offsets(4096, index_dim);
        assert_eq!(codes, 4096 * 64);
    }
}
