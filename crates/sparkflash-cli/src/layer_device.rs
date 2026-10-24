//! The whole-layer device port: buffer ownership and the seam kernel contract.
//!
//! The steady-state decode loop in [`crate::tp4_forward`] must not cross PCIe. This module is the
//! contract that makes that true: it names the device buffers the layer owns, the stage order, and
//! the kernel signature behind each stage that used to run as host code (the mHC projection, the
//! FFN epilogue seams, the attention staging, the activation quantiser, the tree reduce, the MoE
//! router and combine, and the cross-rank all-reduce). It contains no arithmetic: it is the
//! interface device-lead and comm-lead implement against, plus the call site the loop uses.
//!
//! # Buffer ownership
//!
//! One allocation at open (`LayerBuffers::bytes` sizes it, `LayerBuffers::carve` splits it), freed
//! at run end. Every stage borrows a [`DeviceSpan`] and writes device memory only:
//!
//! | span | bytes | producer | consumer |
//! | --- | --- | --- | --- |
//! | `residual.stream` | `hc_mult * dim * 4` (the hc stream: 80 KiB at dim 5120) | `begin_position` (H2D, once per position), then `attention_residual` (hc_post, in place), then `ffn_epilogue` | `mhc_entry`, `ffn_seams` of the same layer; `residual_out` (D2H, once per position) |
//! | `residual.hc_rows` | `hc_mult * dim * 4` | `mhc_entry` (its only writer; the FFN seam's dead copy was trimmed from the ABI, driver-fixer `eec3c9ee`) | `attention_residual` (hc_post's residual sources); `ffn_epilogue` reads `residual.stream` in place, not this span |
//! | `residual.normed` | `dim * 4` | `mhc_entry` | `quantise_prologue`, `reference_seam` |
//! | `residual.pre_mix` | `hc_mult * 4` | `begin_position` (the identity seed [1,0,0,0], the first writer of every position and the only writer for layer 0), then `ffn_seams` (the FFN pre mix) | `mhc_entry` of the next layer, `attn_pre_mix_out` |
//! | `workspace.act_codes` | `act_row` (E4M3, one byte each) | `quantise_prologue` (the attention row, up to `act_row` wide), `query_path` (the wq_b latent prologue at `cols = q_lora`), `ffn_seams` (the FFN quantiser at `dim`); three producers share the span, and only stage order makes the reuse safe | `attention_projections` and `query_path`'s wq_b read the attention quantise, `moe_routed` reads the FFN quantise |
//! | `workspace.act_scales` | `ceil(act_row / 32)` (E8M0) | the same three producers as `act_codes`, at the same widths | the same consumers as `act_codes` |
//! | `workspace.attn_out_bf16` | `dim * 2` | `attention_output` | `all_reduce` |
//! | `workspace.reduced_bf16` | `dim * 2` | `all_reduce` | `attention_residual` |
//! | `workspace.moe_delta` | `dim * 4` (20 KiB at dim 5120) | `moe_routed` (the f32 routed delta, single rank) | `moe_routed`'s bf16 cast before the second collective |
//! | `workspace.moe_delta_bf16` | `dim * 2` | `moe_routed` (the pre-reduce row) | `all_reduce_moe` (its input) |
//! | `workspace.ffn_x_bf16` | `dim * 2` (10 KiB at dim 5120, zero aliasing) | `ffn_seams` (the FFN block's RMS-normed row, RNE-narrowed - the host arm's pre_ffn_hidden, moe-stage-contract H3b; the same value seam_row holds, NOT the pre-norm hc_pre collapse. Driver-fixer's FFN ABI bump; the span exists now so its width is asserted and nothing aliases it) | `moe_router` (its x: RNE(pre_ffn_hidden) - the same value the host arm narrows at the router launch) |
//! | `workspace.moe_reduced_bf16` | `dim * 2` | `all_reduce_moe` | `ffn_epilogue` (as its `moe_delta`) |
//! | `workspace.seam_row` | `hc_mult * dim * 4` (80 KiB at dim 5120) | `ffn_seams` (the FFN block norm at the span base), then `moe_shared` (the shared expert row: `layers.{layer}.ffn.shared_experts.w1/w3/w2`, **f32 by default and bf16 only under native-rounding shared**) | `moe_shared` (the f32-activation arm reads the normed row at the base in place) and `ffn_epilogue` (the shared row at the base); `moe_router` reads `workspace.ffn_x_bf16`, not this row |
//! | `workspace.moe_ids` | `topk * 4` | `moe_router` | `moe_routed` |
//! | `workspace.moe_weights` | `topk * 4` | `moe_router` | `moe_routed` |
//! | `workspace.logits` | `vocab_pad * 4` | `moe_router` | `moe_router` |
//! | `workspace.reduce_ws` | `2 * dim * 4` | `all_reduce` | `all_reduce` |
//! | `workspace.kv_rotated` | `1 KiB` | `kv_path` via the fused `kv_norm_rope_bf16` (rms_norm with `kv_norm` in f32, then the rope, one bf16 rounding after the rope; declared deviation: the device latent is bf16 where the host keeps f32, so the kernel widens it) | `kv_path` (the window quantiser and the append) |
//! | `workspace.kv_latent` | `1 KiB` | `attention_projections` (the wkv GEMV, 512 bf16) | `kv_path` (rope input) |
//! | `workspace.attn_mix` | `96 B` | `mhc_entry` (pre[0..4], post[4..8], comb[8..24] f32, the write order of `spark_seam_coefficient_tail`) | `attention_residual` (post, comb), `ffn_seams` (pre, its hc_pre input) |
//! | `workspace.ffn_mix` | `96 B` | `ffn_seams` (same pre/post/comb layout) | `ffn_epilogue` only (the next layer's entry reads `residual.pre_mix`, which `ffn_seams` rotates) |
//! | `workspace.q_lora` | `2,560 B` | `attention_projections` (the wq_a GEMV) | `query_path` (q_norm) |
//! | `workspace.q_normed` | `2,560 B` | `query_path` (rmsnorm) | `query_path` (quantise cols=1280, then wq_b) |
//! | `workspace.q` | `(n_heads / world) * head_dim * 2` | `query_path` (wq_b GEMV, rope tail in place) | `attention_body` |
//! | `workspace.rope_tables_plain` | `max_positions * 256 B` | `DeviceLayer::open` (H2D once, from rope_tables_range class 0) | `query_path`, `kv_path` |
//! | `workspace.rope_tables_compressed` | `max_positions * 256 B` | `DeviceLayer::open` (H2D once, class 1) | `query_path`, `kv_path` |
//!
//! **Decode ordering, stated so it is not a hidden assumption:** \`run_layer\` appends before
//! \`reference_seam\`/\`stage_attention\`/\`attention_body\` read, and that order is a **requirement**,
//! not an equivalence: the window includes the current token, \`[p + 1 - window_rows ..= p]\` with
//! \`window_rows = window_span(p, 128)\`, so the body cannot see position \`p\` unless the append ran
//! first. \`attention_stage::assemble\` refuses a position past the ring (\`PositionPastRing\`), which is
//! the refusal for reading before the append; multi-position/prefill keeps the same per-position
//! rule (append, then read), not a split.
//!
//! Rules the loop and every kernel must hold:
//!
//! 1. **The residual never comes back to the host inside the layer loop.** It is written by
//!    `begin_position` once per position and read by `residual_out` once per position, after the last
//!    layer. Those two copies are the layer loop's only crossings; everything between them is a
//!    device pointer.
//! 2. **No stage allocates and no stage returns bytes.** Workspace is sized at open from the plan's
//!    geometry; a stage that needs more names it in its signature. Evidence the window wants is a
//!    device digest (`*_digest` out-parameters), read at most once per run and only when the
//!    evidence flag is on - never per layer or per position.
//! 3. **Stage order is the dependency.** The stages run in the order of [`LayerStages::run_layer`] on
//!    the default stream; the only synchronisation points are `begin_position`, `residual_out` and
//!    whatever a kernel documents for its own tie-break. The host never waits between stages.
//! 4. **The cross-rank reduce is a device call, queued not waited.** `all_reduce` hands two
//!    device addresses and the stream to the communicator: no synchronise, no readback, no host
//!    spawn or join, and the next stage is queued behind it on the same stream. The spans are the
//!    layer's, the communicator handle is the run's, and the call shape is fixed: device
//!    addresses plus the stream.
//! 5. **The device layer is the default arm (S2).** S1 closed with all 40 layers, 20 stages each,
//!    device-only in one decode step, so the device arm is what an unset `SPARKFLASH_LAYER_DEVICE`
//!    selects. `SPARKFLASH_LAYER_DEVICE=0` (or `host`) keeps the bit-exact host sequence selectable;
//!    a spelling the parser does not know refuses at startup instead of guessing an arm. A stage
//!    whose kernel (or whose device step) is missing still refuses by name instead of falling back.
//! 7. **The KV cache is per-arm.** The device arm owns a device cache (`DeviceKvCache`) created by
//!    `DeviceLayer::open` beside the layer slab; the host arm keeps the host-resident cache it
//!    has always used. Nothing mutable is shared between the arms, a run never mixes them, and the
//!    device arm's cache is never read back to the host in steady state. An appended row becomes visible
//!    to the window seam when its append is enqueued on the same stream the window reads, in
//!    position order: stream ordering is the publication, and a seam reading row p may assume rows
//!    0..=p were appended before it in that stream.
//!
//! 6. **Evidence recording and the device arm are mutually exclusive.** A recording run needs the
//!    per-position host readbacks rule 2 removes, so select_layer_arm forces the host arm when
//!    evidence is on and the run record says so. The device arm never silently drops evidence,
//!    and the host arm never pretends to be the port.
//!
//! # Reduce init, warm-up and accounting (comm-lead's API, pinned here)
//!
//! Two collectives per layer: the attention delta (`all_reduce`) and the routed MoE delta
//! (`all_reduce_moe`), 5,120 bf16 elements each ([COLLECTIVES_PER_LAYER]). The API is comm-lead's
//! (task/gpu-reduce), and this is the shape the loop codes against:
//!
//! * `open(world, rank, unique_id: *const c_void) -> *mut c_void` - the 128 opaque bytes are an
//!   **argument**, not a broadcast the communicator makes. Rank 0 calls `unique_id()` and the loop
//!   broadcasts those 128 bytes to ranks 1..3 over the control channel it already opened, and all
//!   four pass the same bytes to `open`. Library loading happens inside `open`, so an absent NCCL
//!   is a typed `LibraryAbsent`; a null handle refuses with `Config`; `close(null)` is a no-op.
//!   The run context stores exactly one opaque word.
//! * `warm_up(handle, stream, scratch, count)` after open: it enqueues **three** in-place reduces
//!   over the caller's device span and does not synchronise (the comm crate has no CUDA
//!   dependency), so the loop's own single setup sync completes it - the one sync the plan grants
//!   at setup. Pass `count = 5,120` (the real payload, so the same path is warmed) and
//!   `workspace.reduce_ws` as the scratch span (40 KiB, large enough). Warm-up collectives are
//!   setup, not layer work and are **not** in the stats: a run record's collective counter must come
//!   out exactly `layers * COLLECTIVES_PER_LAYER` (80 for a 40-layer position set).
//! * `reduce(handle, stream, input, output, count)` - `count` in bf16 **elements**, out-of-place,
//!   enqueue-only, no sync, no readback. `count` is never bytes: bytes is where a 2x error hides.
//!   The handle is `void*`: opaque and run-owned, so nothing in this module is self-referential.
//! * `stats(handle)` carries `arena_bytes` (a 16 MiB reserve at open by default). The run record
//!   prints it next to the layer slab's own size (`layer_device_memory`) so the arm's memory cost
//!   cannot drift from the account: one struct, one number.
//!
//! `workspace.reduce_ws` therefore has a documented dual purpose: the NCCL arm's warm-up scratch
//! and the RDMA arm's tree scratch. It is never a per-call workspace - the NCCL reduce needs none,
//! and a reduce that grows one per call is a bug in the reduce, not something this span adapts to.
//!
//! # Kernel signatures device-lead lands
//!
//! Each entry of [`STAGE_CONTRACTS`] is the ABI the loop calls and the exact text a refusal carries.
//! Sizes are in elements of the buffer's element type; `dim` is the model width, `hc_mult` the mHC
//! multiplier and `act_row` the widest quantised activation row.
//!
//! The quantiser's place in the order is deliberate: today `run_layer_attention_projections`
//! quantises the normed row on the host before the GEMV; here `quantise_prologue` produces the codes
//! on device as the projection's prologue, so the row is never host-visible.
//!
//! The `all_reduce_rdma_arm` row is not a stage: it pins the RDMA arm's kernel so both transports are
//! stated in one table, and the NCCL arm never launches it.

/// The switch: the device layer is the default arm since S2 (the whole 40-layer, 20-stage step runs
/// device-only; S1 closed 2026-10-23). Unset or empty, `1`, `device`, `true` and `yes` select it;
/// `0`, `host`, `false` and `no` keep the bit-exact host sequence selectable. Any other spelling
/// refuses at startup: a typo must not pick an arm.
pub const LAYER_DEVICE_ENV: &str = "SPARKFLASH_LAYER_DEVICE";

/// The switch's parser, over the raw value so a test can pin every spelling without touching the
/// process environment (the suite runs tests in parallel and must not race a shared variable).
///
/// `Ok(true)` is the device arm, `Ok(false)` the host arm. The value is matched exactly (ASCII case
/// aside): `" 1"` is not `1`, because a padded value usually means a quoting mistake in a launch
/// script, and the arm a row ran is too important to guess.
///
/// # Errors
/// A value that names neither arm, with the accepted spellings.
pub fn parse_device_switch(value: Option<&str>) -> Result<bool, String> {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return Ok(true);
    };
    let is = |names: &[&str]| names.iter().any(|name| value.eq_ignore_ascii_case(name));
    if is(&["1", "device", "true", "yes"]) {
        Ok(true)
    } else if is(&["0", "host", "false", "no"]) {
        Ok(false)
    } else {
        Err(format!(
            "{LAYER_DEVICE_ENV}={value:?} names neither arm: unset, 1, device, true or yes is the device arm (the default); 0, host, false or no is the host arm"
        ))
    }
}

/// Collectives the layer makes: the attention delta and the routed MoE delta. A run record's
/// counter must be exactly `layers * this` (the three warm-up reduces are setup, not layer work).
pub const COLLECTIVES_PER_LAYER: usize = 2;

/// bf16 elements per collective: the attention delta and the routed MoE delta are both `dim` wide.
pub const REDUCE_ELEMENTS: usize = 5_120;

/// The plan tensor names this port uses, as (logical key, plan-name suffix). Every stage body
/// refers to a name through this table, so a name appears once in the codebase and a rename is one
/// edit. The suffix is matched against the end of a bind pointer key, which keeps the per-layer
/// prefix (the released checkpoint's "model.layers.N.") out of the code.
pub const WEIGHT_NAMES: &[(&str, &str)] = &[
    ("q_norm", "attn.q_norm.weight"),
    ("kv_norm", "attn.kv_norm.weight"),
    ("wq_a", "attn.wq_a.weight"),
    ("wq_b", "attn.wq_b.weight"),
    ("wkv", "attn.wkv.weight"),
    ("attn_norm", "attn_norm.weight"),
    ("ffn_norm", "ffn_norm.weight"),
    // The mHC seam tensors, built for each twin by the plan builder (engine/tensor.rs:199-216):
    // fn [mix_hc][hc_dim] f32, base [mix_hc] f32, scale [3] f32. At released geometry that is
    // 1,966,080 B + 96 B + 12 B per sublayer. They are weights the kernels read, not derivations.
    // Name checks go against the plan builder, never against a unit-test fixture.
    ("hc_attn_fn", "hc_attn_fn"),
    ("hc_attn_base", "hc_attn_base"),
    ("hc_attn_scale", "hc_attn_scale"),
    ("hc_ffn_fn", "hc_ffn_fn"),
    ("hc_ffn_base", "hc_ffn_base"),
    ("hc_ffn_scale", "hc_ffn_scale"),
];

// The names are the real plan names the bind table builds ('layers.{layer}.attn.wq_a.weight',
// 'layers.{layer}.attn_norm.weight', ...), so the suffix is anchored at the END of the key. An
// earlier draft carried 'post_attention_layernorm.weight', which no plan in this engine builds: a
// suffix that matches nothing is silent, and a suffix that matches mid-name would bind the wrong
// tensor at all 40 layers. Both are refused by the tests below rather than discovered in a row.

/// The layer index a plan tensor name carries, if any: the digits after "layers.".
#[must_use]
pub fn layer_of(plan_name: &str) -> Option<usize> {
    let rest = plan_name.split_once("layers.")?.1;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Collects the port's weights from a bind's pointer map (address pairs), per layer and per
/// WEIGHT_NAMES entry. Done once at open so no position pays a string lookup.
#[must_use]
pub fn collect_layer_weights(
    pointers: &std::collections::BTreeMap<String, (u64, u64)>,
) -> std::collections::BTreeMap<(usize, usize), u64> {
    let mut by_layer = std::collections::BTreeMap::new();
    for (index, (_logical, suffix)) in WEIGHT_NAMES.iter().enumerate() {
        for (key, (weight, _scales)) in pointers {
            if key.ends_with(suffix)
                && let Some(layer) = layer_of(key)
            {
                by_layer.insert((layer, index), *weight);
            }
        }
    }
    by_layer
}

/// Collects the port's weights with **both** pointer words (weight and scale grid), per layer and
/// per WEIGHT_NAMES entry. The GEMV launchers take the fp8 weight and its scale grid separately,
/// so a map that keeps only the weight word cannot drive them; this is the pair-preserving twin of
/// [`collect_layer_weights`].
#[must_use]
pub fn collect_layer_weight_pairs(
    pointers: &std::collections::BTreeMap<String, (u64, u64)>,
) -> std::collections::BTreeMap<(usize, usize), (u64, u64)> {
    let mut by_layer = std::collections::BTreeMap::new();
    for (index, (_logical, suffix)) in WEIGHT_NAMES.iter().enumerate() {
        for (key, pair) in pointers {
            if key.ends_with(suffix)
                && let Some(layer) = layer_of(key)
            {
                by_layer.insert((layer, index), *pair);
            }
        }
    }
    by_layer
}

/// Resolves one weight by its full plan tensor name.
///
/// The refusal names the candidates that share the requested name's last path component and how
/// many pointers the bind carries: a renamed or typo'd tensor is a sentence, not a hunt.
///
/// # Errors
/// The name is absent, or its weight word is zero (a null device pointer).
pub fn resolve_weight(
    pointers: &std::collections::BTreeMap<String, (u64, u64)>,
    plan_name: &str,
) -> Result<u64, String> {
    if let Some((weight, _scales)) = pointers.get(plan_name) {
        if *weight == 0 {
            return Err(format!(
                "layer device weight: the bind maps {plan_name} to a null pointer"
            ));
        }
        return Ok(*weight);
    }
    let tail = plan_name.rsplit('.').next().unwrap_or(plan_name);
    let mut candidates: Vec<&str> = pointers
        .keys()
        .filter(|key| key.ends_with(tail))
        .map(String::as_str)
        .take(8)
        .collect();
    candidates.sort_unstable();
    let suffix = if candidates.is_empty() {
        String::new()
    } else {
        format!(": {}", candidates.join(", "))
    };
    Err(format!(
        "layer device weight: the bind has no {plan_name}; it carries {} pointers and {} share \
         the name tail {tail}{suffix}",
        pointers.len(),
        candidates.len()
    ))
}

/// The rope output span: 1 KiB, of which the released `rope_head_dim` uses 256 B as f32.
pub const KV_ROTATED_BYTES: usize = 1024;

/// The wkv latent span: 512 bf16 values (1 KiB), the GEMV output `kv_path` rotates.
pub const KV_LATENT_BYTES: usize = 1024;

/// The mHC entry kernel's mix table: **pre[0..4], post[4..8], comb[8..24]** f32, 96 B.
///
/// That is not an arbitrary packing, it is the write order of the only writer of all three regions,
/// `spark_seam_coefficient_tail` in kernels/layer_seams_sm121.cu: it publishes the pre values first
/// (its own `pre_mix` consumers read them immediately), then the post values, then the
/// split-sinkhorn comb row-major. A reader indexes those fixed offsets; a consumer that assumed the
/// comb came first would misread every coefficient of every layer.
pub const ATTN_MIX_BYTES: usize = 96;

/// The FFN-seam kernel's mix table: the same **pre[0..4], post[4..8], comb[8..24]** layout, 96 B
/// (same helper, same write order).
pub const FFN_MIX_BYTES: usize = 96;

/// The wq_a latent: 1,280 bf16 values, written by attention_projections, normed by query_path.
pub const Q_LORA_VALUES: usize = 1280;
/// The raw wq_a latent span, 2,560 B.
pub const Q_LORA_BYTES: usize = 2 * Q_LORA_VALUES;
/// The q_norm output span, 2,560 B. It cannot alias q_lora: rmsnorm_sm121 takes distinct in/out
/// and the raw latent is still wanted for the record digest.
pub const Q_NORMED_BYTES: usize = 2 * Q_LORA_VALUES;

/// The wo_a output row: `groups * o_lora_rank` bf16 values (2,048 at world 4, 4,096 B).
pub const ATTN_LO_VALUES: usize = 2_048;
/// The raw wo_a output span, 4,096 B.
pub const ATTN_LO_BYTES: usize = 2 * ATTN_LO_VALUES;

/// The candidate select's id list: `MAX_SELECTED` u32 ids (512 at the released geometry).
pub const SELECTED_BYTES: usize = sparkflash_kernels::flash_mla_csa2_seq_launch::MAX_SELECTED * 4;

/// The compressed store's FP4 code bytes, one expanded row per position
/// (`EXPANDED_ROWS * CODE_BYTES` = 131,072 B at the released geometry).
pub const STORE_CODE_BYTES: usize = sparkflash_kernels::compressed_cache_ref::EXPANDED_ROWS
    * sparkflash_kernels::compressed_cache_ref::CODE_BYTES;
/// The compressed store's E4M3 group-scale bytes (16,384 B at the released geometry).
pub const STORE_SCALE_BYTES: usize = sparkflash_kernels::compressed_cache_ref::EXPANDED_ROWS
    * sparkflash_kernels::compressed_cache_ref::SCALE_BYTES;

/// The indexer query row the sequence entry reads: index head 0 only, bf16 (8,192 B).
pub const INDEX_QUERY_BYTES: usize =
    sparkflash_kernels::mla_ref::INDEX_HEADS * sparkflash_kernels::mla_ref::INDEX_HEAD_DIM * 2;
/// The sequence entry's per-block scratch: `HSI_BLOCKS` f32 scores and `HSI_BLOCKS` u32 best ids
/// (8,192 B each).
pub const BLOCK_SCRATCH_BYTES: usize = sparkflash_kernels::mla_ref::HSI_BLOCKS * 4;

/// How many rope table classes the device arm carries: `compress_ratio(layer) > 0` is the whole
/// distinction (kv_path_ref::rope_table_class), so the tables are four arrays (two classes x cos,
/// sin), never forty.
pub const ROPE_TABLE_CLASSES: usize = 2;

/// One class's rope span: cos region then sin region, each `max_positions * (rope_head_dim / 2)`
/// f32 values. At rope_head_dim 64 that is `max_positions * 256 B` per class.
#[must_use]
pub fn rope_table_span_bytes(max_positions: usize) -> usize {
    max_positions * 2 * (RELEASED_ROPE_HEAD_DIM / 2) * 4
}

/// The communicator's default init-time arena reserve (`stats.arena_bytes` at open). The run record
/// prints it next to the layer slab's own size so the arm's memory cannot drift from the account.
pub const REDUCE_ARENA_RESERVE_BYTES: usize = 16 << 20;

/// Every span starts on a 256-byte boundary: the widest vector the seam kernels load.
pub const ALIGN: usize = 256;

fn align_up(value: usize) -> usize {
    value.div_ceil(ALIGN) * ALIGN
}

/// One device buffer: the device base pointer and its byte length.
///
/// The owner is the allocation [`LayerBuffers::bytes`] sized and the caller made; a stage borrows
/// the span for the length of its launch and never frees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSpan {
    /// The device pointer a kernel is launched with.
    pub device: u64,
    /// The buffer length in bytes.
    pub len: usize,
}

impl DeviceSpan {
    /// A span at `device` of `len` bytes.
    #[must_use]
    pub fn new(device: u64, len: usize) -> Self {
        Self { device, len }
    }

    /// The same span moved `bytes` into its allocation, for carving one slab into buffers.
    #[must_use]
    pub fn offset(self, bytes: usize) -> Self {
        Self {
            device: self.device + bytes as u64,
            len: self.len.saturating_sub(bytes),
        }
    }
}

/// The residual spans: the layer's input and output, the mHC rows, the normed row and the mix the
/// next layer's entry needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerResidual {
    /// `hc_mult * dim` f32: the hc stream the layer reads at `mhc_entry` and writes at
    /// `ffn_epilogue` (the residual the next layer and the certified head consume).
    pub stream: DeviceSpan,
    /// `hc_mult * dim` f32: the mHC rows the entry builds and the residual/FFN seams read.
    pub hc_rows: DeviceSpan,
    /// `dim` f32: the normed row the quantise prologue and the reference seam read.
    pub normed: DeviceSpan,
    /// `hc_mult` f32: the FFN pre mix, written by `ffn_seams` and read by the next layer's entry.
    pub pre_mix: DeviceSpan,
}

/// The per-stage workspace: activation quantisation, the attention output pair, the 80 KiB seam row
/// and the MoE router's outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerWorkspace {
    /// `act_row` u8: E4M3 activation codes, written by the projection's prologue.
    pub act_codes: DeviceSpan,
    /// `ceil(act_row / 32)` u8: E8M0 block scales for those codes.
    pub act_scales: DeviceSpan,
    /// `dim` bf16: the attention output before the cross-rank reduce.
    pub attn_out_bf16: DeviceSpan,
    /// `dim` bf16: the attention output after the reduce; the residual stage's input.
    pub reduced_bf16: DeviceSpan,
    /// `dim` bf16: the routed MoE delta after its own reduce; `ffn_epilogue` reads it
    /// as `moe_delta`. Separate from `reduced_bf16` on purpose: the attention residual consumes
    /// that span first, so reusing it would alias the two collectives the layer makes.
    pub moe_reduced_bf16: DeviceSpan,
    /// `hc_mult * dim` f32: the seam row the FFN seams hand forward (80 KiB at dim 5120).
    pub seam_row: DeviceSpan,
    /// `dim` f32: the routed MoE delta before the second collective (20 KiB at dim 5120). The
    /// routed stage writes it single-rank, then writes its bf16 pre-reduce row; the f32 row stays
    /// readable for the record, it is not an operand of the collective.
    pub moe_delta: DeviceSpan,
    /// `dim` bf16: the pre-reduce row `all_reduce_moe` takes (out-of-place: in here, out at
    /// `moe_reduced_bf16`). Separate from `moe_reduced_bf16` because the collective is
    /// out-of-place and the two arms must not alias.
    pub moe_delta_bf16: DeviceSpan,
    /// `dim` bf16 (10 KiB at dim 5120): the pre-FFN hidden row the router multiplies. Written by
    /// `ffn_seams` (the RNE narrowing of the f32 stream row it reads), zero-aliased on purpose: the
    /// router's operand is the row before the FFN block's RMSNorm, not the normed row at
    /// `seam_row`'s base.
    pub ffn_x_bf16: DeviceSpan,
    /// `topk` u32: the MoE router's expert ids.
    pub moe_ids: DeviceSpan,
    /// `topk` f32: the MoE router's expert weights.
    pub moe_weights: DeviceSpan,
    /// `vocab_pad` f32: the router's logits scratch.
    pub logits: DeviceSpan,
    /// `2 * dim` f32 (40 KiB at dim 5120): the reduce scratch - the NCCL arm's
    /// `warm_up` span and the RDMA arm's tree scratch. Neither arm needs a per-call
    /// workspace: the NCCL reduce's buffers are init-time and charged to the run arena, and a
    /// reduce that grows one per call is a bug in the reduce, not something this span adapts to.
    pub reduce_ws: DeviceSpan,
    /// 1 KiB: `kv_path`'s rotation output. The rope kernel is out-of-place by its frozen
    /// signature, and the tail is `rope_head_dim` f32 (64 values = 256 B); the span carries slack for a
    /// wider tail. A dedicated span rather than a share of the seam row, so the window seam
    /// cannot alias the rotation output while the append is still queued.
    pub kv_rotated: DeviceSpan,
    /// 1 KiB: the 512-value bf16 wkv latent that `attention_projections` writes and `kv_path` rotates.
    /// It is a slab span because rule 2 forbids a stage allocating: the projection's output needs a
    /// home the next stage can read, and there was none before this.
    pub kv_latent: DeviceSpan,
    /// 96 B: the mHC entry kernel's mix table, **pre[0..4], post[4..8], comb[8..24]** f32 (see
    /// `ATTN_MIX_BYTES` for why that order). Read by the attention residual (post, comb) and by
    /// the FFN seams (pre, as its hc_pre input).
    pub attn_mix: DeviceSpan,
    /// 96 B: the FFN-seam kernel's mix table, same pre/post/comb layout. Its only consumer is the
    /// FFN epilogue; the next layer's entry reads `residual.pre_mix`, not this span.
    pub ffn_mix: DeviceSpan,
    /// 2,560 B: the wq_a latent, written by attention_projections, read by query_path.
    pub q_lora: DeviceSpan,
    /// 2,560 B: the q_norm output, written by query_path, read by the latent quantiser and wq_b.
    pub q_normed: DeviceSpan,
    /// (n_heads / world) * head_dim bf16: the wq_b output, written by query_path, read by
    /// attention_body as its query row. Per rank, because wq_b is sharded over the head axis.
    pub q: DeviceSpan,
    /// (n_heads / world) * head_dim bf16 (16,384 B at world 4): the attention body's per-rank head
    /// rows, written by the sequence entry and rotated in place by the inverse rope; read by
    /// `attention_output` as wo_a's activation. It cannot alias `attn_out_bf16`, which is smaller.
    pub attn_body_bf16: DeviceSpan,
    /// 4,096 B: wo_a's output row, read by the wo_b activation quantiser.
    pub attn_lo: DeviceSpan,
    /// 2,048 B: the candidate select's u32 ids; the body reads `selected_count` of them (0 on a
    /// window-only layer).
    pub selected: DeviceSpan,
    /// 131,072 B: the compressed store's packed FP4 codes, one expanded row per position.
    pub store_codes: DeviceSpan,
    /// 16,384 B: the compressed store's E4M3 group scales.
    pub store_scales: DeviceSpan,
    /// 8,192 B: the indexer query row the sequence entry reads (index head 0 only).
    pub index_query: DeviceSpan,
    /// 8,192 B: the sequence entry's per-block score scratch.
    pub block_scores: DeviceSpan,
    /// 8,192 B: the sequence entry's per-block best-token scratch.
    pub block_best: DeviceSpan,
    /// max_positions * 256 B: the ordinary rope class (cos region then sin region), uploaded once
    /// at open and read by query_path/kv_path at `base + position * pairs * 4`.
    pub rope_tables_plain: DeviceSpan,
    /// max_positions * 256 B: the compressed class (the YaRN/compress theta builder), same layout.
    pub rope_tables_compressed: DeviceSpan,
    /// `3 * dim * 4` B (61,440 B at dim 5,120): fixed contiguous span for the DSpark/Eagle3
    /// drafter taps (`dspark_target_layer_ids = [37, 38, 39]`), carved at open so S3 CUDA Graph
    /// capture and S5 DSpark have fixed device addresses for all three hidden-row exports.
    pub dspark_aux_taps: DeviceSpan,
}

/// The three DSpark/Eagle3 auxiliary target backbone layers (`dspark_target_layer_ids = [37, 38, 39]`).
pub const DSPARK_AUX_LAYERS: [usize; 3] = [37, 38, 39];

/// Number of DSpark/Eagle3 auxiliary target layer taps per position (`cat(main_hiddens)` = `3 * dim`).
pub const DSPARK_AUX_TAP_COUNT: usize = DSPARK_AUX_LAYERS.len();

/// The slab's geometry: everything the carve needs, in one value, so adding a span does not grow
/// a positional argument list that a caller can silently transpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerGeometry {
    /// The model width (5,120 on the released config): the row stride of every activation span.
    pub dim: usize,
    /// The mHC stream multiplier (4 on the released config): `hc_pre` collapses `hc_mult * dim`
    /// into `dim`.
    pub hc_mult: usize,
    /// The routed-expert activation row width the quantise prologue writes (5,120 on the released
    /// config).
    pub act_row: usize,
    /// Routed experts per token (8 on the released config): the router's topk.
    pub topk: usize,
    /// The vocab padded to the head kernel's row multiple: the head span is this times 2 B.
    pub vocab_pad: usize,
    /// The query-projection column count `wq_b` consumes (`n_heads * head_dim / world`, 8,192 at
    /// world 4).
    pub q_cols: usize,
    /// Positions the slab's rope tables cover (`--max-positions`): the largest position the loop may
    /// advance to before the tables run out.
    pub max_positions: usize,
}

/// Everything one layer owns on the device, sized at open and freed at run end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerBuffers {
    /// The residual spans.
    pub residual: LayerResidual,
    /// The workspace spans.
    pub workspace: LayerWorkspace,
}

impl LayerBuffers {
    /// Checks every span whose dtype a stage contract names against that dtype and width.
    ///
    /// A span is exactly its contract width rounded up to ALIGN (and at least one 256-byte
    /// minimum), so a length that is not that number means a stage would hand a kernel a pointer
    /// into a buffer of the wrong dtype or width. This is the open-time half of the dtype audit;
    /// the launch-time half is each kernel's own signature (see
    /// docs/archive/build/layer-device-dtype-audit-20261022.md for the pair-by-pair list).
    ///
    /// # Errors
    /// The first span whose length is not its contract width rounded to the alignment, named with
    /// the three numbers that disagree.
    pub fn assert_contract_widths(&self, geometry: &LayerGeometry) -> Result<(), String> {
        let geometry = *geometry;
        let residual = self.residual;
        let workspace = self.workspace;
        let hc_dim = geometry.hc_mult * geometry.dim;
        let rope_values = geometry.max_positions * (RELEASED_ROPE_HEAD_DIM / 2) * 2;
        let checks: [(&str, DeviceSpan, usize, usize); 34] = [
            ("residual.stream", residual.stream, hc_dim, 4),
            ("residual.hc_rows", residual.hc_rows, hc_dim, 4),
            ("residual.normed", residual.normed, geometry.dim, 4),
            ("residual.pre_mix", residual.pre_mix, geometry.hc_mult, 4),
            (
                "workspace.act_codes",
                workspace.act_codes,
                geometry.act_row,
                1,
            ),
            (
                "workspace.act_scales",
                workspace.act_scales,
                geometry.act_row.div_ceil(32),
                1,
            ),
            (
                "workspace.attn_out_bf16",
                workspace.attn_out_bf16,
                geometry.dim,
                2,
            ),
            (
                "workspace.reduced_bf16",
                workspace.reduced_bf16,
                geometry.dim,
                2,
            ),
            (
                "workspace.moe_reduced_bf16",
                workspace.moe_reduced_bf16,
                geometry.dim,
                2,
            ),
            ("workspace.seam_row", workspace.seam_row, hc_dim, 4),
            ("workspace.moe_delta", workspace.moe_delta, geometry.dim, 4),
            (
                "workspace.moe_delta_bf16",
                workspace.moe_delta_bf16,
                geometry.dim,
                2,
            ),
            (
                "workspace.ffn_x_bf16",
                workspace.ffn_x_bf16,
                geometry.dim,
                2,
            ),
            ("workspace.moe_ids", workspace.moe_ids, geometry.topk, 4),
            (
                "workspace.moe_weights",
                workspace.moe_weights,
                geometry.topk,
                4,
            ),
            ("workspace.logits", workspace.logits, geometry.vocab_pad, 4),
            (
                "workspace.reduce_ws",
                workspace.reduce_ws,
                2 * geometry.dim,
                4,
            ),
            (
                "workspace.kv_rotated",
                workspace.kv_rotated,
                KV_ROTATED_BYTES,
                1,
            ),
            (
                "workspace.kv_latent",
                workspace.kv_latent,
                crate::tp4_forward::MLA_LATENT,
                2,
            ),
            ("workspace.attn_mix", workspace.attn_mix, ATTN_MIX_BYTES, 1),
            ("workspace.ffn_mix", workspace.ffn_mix, FFN_MIX_BYTES, 1),
            ("workspace.q_lora", workspace.q_lora, Q_LORA_VALUES, 2),
            ("workspace.q_normed", workspace.q_normed, Q_LORA_VALUES, 2),
            ("workspace.q", workspace.q, geometry.q_cols, 2),
            (
                "workspace.attn_body_bf16",
                workspace.attn_body_bf16,
                geometry.q_cols,
                2,
            ),
            ("workspace.attn_lo", workspace.attn_lo, ATTN_LO_VALUES, 2),
            (
                "workspace.selected",
                workspace.selected,
                SELECTED_BYTES / 4,
                4,
            ),
            (
                "workspace.store_codes",
                workspace.store_codes,
                STORE_CODE_BYTES,
                1,
            ),
            (
                "workspace.store_scales",
                workspace.store_scales,
                STORE_SCALE_BYTES,
                1,
            ),
            (
                "workspace.index_query",
                workspace.index_query,
                INDEX_QUERY_BYTES,
                1,
            ),
            (
                "workspace.block_scores",
                workspace.block_scores,
                BLOCK_SCRATCH_BYTES,
                1,
            ),
            (
                "workspace.block_best",
                workspace.block_best,
                BLOCK_SCRATCH_BYTES,
                1,
            ),
            (
                "workspace.rope_tables_plain",
                workspace.rope_tables_plain,
                rope_values,
                4,
            ),
            (
                "workspace.dspark_aux_taps",
                workspace.dspark_aux_taps,
                DSPARK_AUX_TAP_COUNT * geometry.dim,
                4,
            ),
        ];
        for (name, span, values, bytes_per_value) in checks {
            let needed = values.checked_mul(bytes_per_value).ok_or_else(|| {
                format!("layer slab: {name} values x {bytes_per_value} B overflow")
            })?;
            let expected = align_up(needed.max(ALIGN));
            if span.len != expected {
                return Err(format!(
                    "layer slab: {name} is {} B but its contract names {values} value(s) x {bytes_per_value} B = {needed} B, which the carve rounds to {expected} B",
                    span.len
                ));
            }
        }
        Ok(())
    }

    /// The byte size of the whole layer allocation for one geometry.
    ///
    /// `act_row` is the widest activation row the layer quantises (the attention projection's input
    /// and the MoE gate/up input differ, so callers pass the larger); `topk` and `vocab_pad` size the
    /// router's outputs. Every span is padded to [`ALIGN`].
    #[must_use]
    pub fn bytes(geometry: &LayerGeometry) -> usize {
        let LayerGeometry {
            dim,
            hc_mult,
            act_row,
            topk,
            vocab_pad,
            q_cols,
            max_positions,
        } = *geometry;
        let spans = [
            hc_mult * dim * 4,
            hc_mult * dim * 4,
            dim * 4,
            hc_mult * 4,
            act_row,
            act_row.div_ceil(32),
            dim * 2,
            dim * 2,
            dim * 2,
            hc_mult * dim * 4,
            dim * 4,
            dim * 2,
            dim * 2,
            topk * 4,
            topk * 4,
            vocab_pad * 4,
            2 * dim * 4,
            KV_ROTATED_BYTES,
            KV_LATENT_BYTES,
            ATTN_MIX_BYTES,
            FFN_MIX_BYTES,
            Q_LORA_BYTES,
            Q_NORMED_BYTES,
            q_cols * 2,
            q_cols * 2,
            ATTN_LO_BYTES,
            SELECTED_BYTES,
            STORE_CODE_BYTES,
            STORE_SCALE_BYTES,
            INDEX_QUERY_BYTES,
            BLOCK_SCRATCH_BYTES,
            BLOCK_SCRATCH_BYTES,
            rope_table_span_bytes(max_positions),
            rope_table_span_bytes(max_positions),
            DSPARK_AUX_TAP_COUNT * dim * 4,
        ];
        spans.iter().map(|&bytes| align_up(bytes.max(ALIGN))).sum()
    }

    /// Returns the fixed `[dim]` f32 device sub-span inside `workspace.dspark_aux_taps` for the
    /// DSpark/Eagle3 auxiliary target layers (`[37, 38, 39]`), or `None` for any other layer.
    #[must_use]
    pub fn dspark_aux_tap_span(&self, layer: usize, dim: usize) -> Option<DeviceSpan> {
        let tap_idx = DSPARK_AUX_LAYERS
            .iter()
            .position(|&target| target == layer)?;
        let row_bytes = dim.checked_mul(4)?;
        let offset = tap_idx.checked_mul(row_bytes)?;
        if offset.checked_add(row_bytes)? > self.workspace.dspark_aux_taps.len {
            return None;
        }
        Some(DeviceSpan::new(
            self.workspace.dspark_aux_taps.device + offset as u64,
            row_bytes,
        ))
    }

    /// The bytes this carve actually spends, span by span: the slab's own number for the record.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        let residual = [
            self.residual.stream,
            self.residual.hc_rows,
            self.residual.normed,
            self.residual.pre_mix,
        ];
        let workspace = [
            self.workspace.act_codes,
            self.workspace.act_scales,
            self.workspace.attn_out_bf16,
            self.workspace.reduced_bf16,
            self.workspace.moe_reduced_bf16,
            self.workspace.seam_row,
            self.workspace.moe_delta,
            self.workspace.moe_delta_bf16,
            self.workspace.ffn_x_bf16,
            self.workspace.moe_ids,
            self.workspace.moe_weights,
            self.workspace.logits,
            self.workspace.reduce_ws,
            self.workspace.kv_rotated,
            self.workspace.kv_latent,
            self.workspace.attn_mix,
            self.workspace.ffn_mix,
            self.workspace.q_lora,
            self.workspace.q_normed,
            self.workspace.q,
            self.workspace.attn_body_bf16,
            self.workspace.attn_lo,
            self.workspace.selected,
            self.workspace.store_codes,
            self.workspace.store_scales,
            self.workspace.index_query,
            self.workspace.block_scores,
            self.workspace.block_best,
            self.workspace.rope_tables_plain,
            self.workspace.rope_tables_compressed,
            self.workspace.dspark_aux_taps,
        ];
        residual
            .iter()
            .chain(workspace.iter())
            .map(|span| span.len)
            .sum()
    }

    /// Splits an allocation of [`LayerBuffers::bytes`] bytes at `base` into its spans, in the order
    /// [`LayerBuffers::bytes`] sums them. Every span is [`ALIGN`]-aligned and no two overlap.
    #[must_use]
    pub fn carve(base: u64, geometry: &LayerGeometry) -> Self {
        let LayerGeometry {
            dim,
            hc_mult,
            act_row,
            topk,
            vocab_pad,
            q_cols,
            max_positions,
        } = *geometry;
        let mut at = 0usize;
        let mut span = |bytes: usize| {
            let len = align_up(bytes.max(ALIGN));
            let span = DeviceSpan::new(base + at as u64, len);
            at += len;
            span
        };
        Self {
            residual: LayerResidual {
                stream: span(hc_mult * dim * 4),
                hc_rows: span(hc_mult * dim * 4),
                normed: span(dim * 4),
                pre_mix: span(hc_mult * 4),
            },
            workspace: LayerWorkspace {
                act_codes: span(act_row),
                act_scales: span(act_row.div_ceil(32)),
                attn_out_bf16: span(dim * 2),
                reduced_bf16: span(dim * 2),
                moe_reduced_bf16: span(dim * 2),
                seam_row: span(hc_mult * dim * 4),
                moe_delta: span(dim * 4),
                moe_delta_bf16: span(dim * 2),
                ffn_x_bf16: span(dim * 2),
                moe_ids: span(topk * 4),
                moe_weights: span(topk * 4),
                logits: span(vocab_pad * 4),
                reduce_ws: span(2 * dim * 4),
                kv_rotated: span(KV_ROTATED_BYTES),
                kv_latent: span(KV_LATENT_BYTES),
                attn_mix: span(ATTN_MIX_BYTES),
                ffn_mix: span(FFN_MIX_BYTES),
                q_lora: span(Q_LORA_BYTES),
                q_normed: span(Q_NORMED_BYTES),
                q: span(q_cols * 2),
                attn_body_bf16: span(q_cols * 2),
                attn_lo: span(ATTN_LO_BYTES),
                selected: span(SELECTED_BYTES),
                store_codes: span(STORE_CODE_BYTES),
                store_scales: span(STORE_SCALE_BYTES),
                index_query: span(INDEX_QUERY_BYTES),
                block_scores: span(BLOCK_SCRATCH_BYTES),
                block_best: span(BLOCK_SCRATCH_BYTES),
                rope_tables_plain: span(rope_table_span_bytes(max_positions)),
                rope_tables_compressed: span(rope_table_span_bytes(max_positions)),
                dspark_aux_taps: span(DSPARK_AUX_TAP_COUNT * dim * 4),
            },
        }
    }
}

/// One stage's kernel contract: the stage name the refusal carries, the kernel symbol, and the ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageContract {
    /// The [`LayerStages`] method (`"mhc_entry"`, `"all_reduce"`, ...).
    pub stage: &'static str,
    /// The CUDA symbol the stage launches.
    pub kernel: &'static str,
    /// The kernel's signature, as the refusal prints it.
    pub signature: &'static str,
}

/// The stage table: order, kernel names and signatures, one entry per [`LayerStages`] method.
pub const STAGE_CONTRACTS: &[StageContract] = &[
    StageContract {
        stage: "engram_write",
        kernel: "engram_embed_sm121 (6-word ABI), engram_wkv_sm121 (10-word ABI), engram_narrow_bf16 (3-word ABI), engram_gate_write_sm121 (11-word ABI), engram_widen_f32 (3-word ABI), engram_write_bf16 (8-word ABI)",
        signature: "(embed: const uint8_t* row_codes, const uint8_t* row_scales, __nv_bfloat16* fetched, uint8_t* act_codes, uint8_t* act_scales, int rows_cols, int head_dim; wkv: const uint8_t* act_codes, const uint8_t* act_scales, const uint8_t* weight, const uint8_t* weight_scales, __nv_bfloat16* out, int out_width, int inner; narrow: const float* src, __nv_bfloat16* dst, int count; gate: const __nv_bfloat16* kv, const __nv_bfloat16* stream, const float* product, __nv_bfloat16* out, float* gates, int dim, int hc, float eps; widen: const __nv_bfloat16* src, float* dst, int count; write: const float* stream, const float* gate, const float* value, float* stream_out, int dim)",
    },
    StageContract {
        stage: "mhc_entry",
        kernel: "mhc_entry_bf16",
        // Twenty words: the declaration carries the entry's own weights (fn_w, hc_base, hc_scale,
        // norm_w), the mix table and the two emitted rows, and the pre_mix output is gone - the
        // entry no longer writes it.
        signature: "(const float* residual, const float* attn_pre_mix, const float* fn_w, const float* hc_base, const float* hc_scale, const uint8_t* norm_w, const float* mix_table, float* hc_rows, float* normed, uint32_t dim, uint32_t hc_mult, uint32_t rows, uint32_t norm_bf16, uint32_t native_bf16, uint32_t sinkhorn_iterations, float norm_eps, float hc_eps, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
    },
    StageContract {
        stage: "quantise_prologue",
        kernel: "w8a8_prologue_e4m3",
        // The row is f32: the f32 prologue reads four bytes per value (the bf16 variant has its own
        // kernel and its own stage path, query_path's latent quantise).
        signature: "(const float* normed, uint8_t* codes, uint8_t* scales, uint32_t cols, uint32_t rows, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
    },
    StageContract {
        stage: "attention_projections",
        kernel: "b12x_mxfp8_gemv_m1_sm121",
        signature: "(const uint8_t* a_fp8, const uint8_t* a_scales, const uint8_t* w_fp8, const uint8_t* w_scales, void* out_bf16, int rows, int cols, int inner, void* trace_ring, int layer_id, int step_seq)",
    },
    StageContract {
        stage: "query_path",
        kernel: "rmsnorm_sm121 then b12x_mxfp8_gemv_m1_sm121 (sequence: q_norm over the wq_a latent, quantise cols=1280, wq_b, rope tail in place)",
        signature: "(const void* x, const void* weight, void* out, unsigned rows, unsigned cols, float eps, void* trace_ring, unsigned layer_id, unsigned long step_seq)",
    },
    // The stage's call: rope, the window quantiser and the append, sequenced on device (the three
    // kernels are pinned in KERNEL_CONTRACTS). No single kernel carries this stage's name, because
    // the stage is a sequence, not a launch.
    StageContract {
        stage: "kv_path",
        kernel: "kv_append_bf16 (sequence: kv_norm_rope_bf16, window_quant_fp8_bf16, kv_append_bf16)",
        signature: "(const void* k_bf16, const void* v_bf16, void* cache, int rows, int head_dim, int position). The fused first launch - (const __nv_bfloat16* row, const __nv_bfloat16* norm, float eps, const float* cos, const float* sin, __nv_bfloat16* out, uint32_t head_width, uint32_t rope_head_dim) - reproduces rms_norm(kv, kv_norm, 512, eps) in f32 then the rope with one bf16 rounding, but its operand is the bf16 latent where the host keeps f32: the one widening the host does not have, and the stage declared deviation.",
    },
    StageContract {
        stage: "reference_seam",
        kernel: "no kernel: a host-decided window width on ratio 0; blocked on the compressor producer above ratio 0",
        signature: "(host-decided: position -> window_width = min(position + 1, 128); no pointer is passed)",
    },
    StageContract {
        stage: "ref_select",
        kernel: "ref_select_sm121 then compressed_selected_fill_sm121 (sequence: select, then the expanded-row map)",
        signature: "(const float* q_index, const float* keys, const float* weights, uint32_t* out_ids, float* out_scores, const uint32_t* mask_in, uint32_t* mask_out, uint32_t n_pos, uint32_t heads, uint32_t head_dim, uint32_t reach, uint32_t topk, uint32_t offset, uint32_t mask_mode, uint32_t topk_blocks, uint32_t block_size, uint32_t no_index, uint32_t flags) then (const uint32_t* ids, uint32_t n_ids, uint32_t window_rows, uint32_t ratio, uint32_t* out, uint32_t* out_count)",
    },
    StageContract {
        stage: "stage_attention",
        kernel: "attention_stage (validation only: no launch)",
        signature: "(no device operand: ring_rows, position, ratio, store_rows, selected_count are host-decided; the refusals are ring/capacity, position-past-ring, cursor, union bound)",
    },
    StageContract {
        stage: "attention_body",
        kernel: "flash_mla_csa2_seq_sm121 then rope_rotate_rows_bf16 (sin_sign = -1)",
        signature: "(const __nv_bfloat16* q, const uint8_t* kv_fp4, const uint8_t* kv_scales, const uint8_t* window_codes, const uint8_t* window_scales, uint32_t window_kv_format, uint32_t window_rows, const uint32_t* selected, const __nv_bfloat16* index_key, const __nv_bfloat16* index_query, const float* sink, float* block_scores, uint32_t* block_best, __nv_bfloat16* out, uint64_t position, uint32_t selected_count, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq) then (const __nv_bfloat16* src, const float* cos, const float* sin, __nv_bfloat16* out, uint32_t rows, uint32_t head_width, uint32_t rope_head_dim, int32_t sin_sign, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
    },
    StageContract {
        stage: "attention_output",
        kernel: "b12x_bf16a_mxfp8w_wo_a_sm121 then w8a8_prologue_bf16_e4m3 then b12x_mxfp8_gemv_m1_sm121",
        signature: "(const __nv_bfloat16* x, const uint8_t* w, const uint8_t* wscales, __nv_bfloat16* out, uint32_t cols, uint32_t inner, uint32_t groups, uint32_t o_lora, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq) then (const __nv_bfloat16* x, uint8_t* codes, uint8_t* scales, uint32_t rows, uint32_t cols) then (const uint8_t* codes, const uint8_t* scales, const uint8_t* w, const uint8_t* wscales, __nv_bfloat16* out, uint32_t rows, uint32_t cols, uint32_t inner, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
    },
    StageContract {
        stage: "all_reduce",
        kernel: "comm_all_reduce_bf16",
        signature: "(cudaStream_t stream, const void* in, void* out, int count, void* handle)",
    },
    StageContract {
        stage: "attention_residual",
        kernel: "mhc_attn_residual",
        signature: "(const float* hc_rows, const float* mix_table, const void* reduced_bf16, float* stream_out, uint32_t dim, uint32_t hc_mult, uint32_t rows, uint32_t native_bf16, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
    },
    StageContract {
        stage: "ffn_seams",
        kernel: "mhc_ffn_seams",
        signature: "(const float* stream, const float* hc_rows, const float* pre_mix, float* seam_row, float* post_mix, int dim, int hc_mult)",
    },
    StageContract {
        stage: "moe_router",
        kernel: "moe_router_sm121 (10-word ABI), then moe_router_topk_sm121 (8-word ABI)",
        signature: "(const __nv_bfloat16* x, const __nv_bfloat16* gate, const float* bias, uint32_t* ids, float* weights, float* scores, int tokens, ...trace tail), then (const float* bias, const float* scores, uint32_t* ids, float* weights, int tokens, ...trace tail)",
    },
    StageContract {
        stage: "moe_routed",
        kernel: "moe_expert_stage_sm121 (10-word ABI) x3, routed_gate_up_sm121 (10-word ABI) x2, swiglu_mxfp8_quant_weighted_sm121 (11-word ABI), b12x_mxfp4_moe_stored_sm121 (10-word ABI), moe_routed_combine_sm121 (6-word ABI)",
        signature: "(three staging gathers: const uint64_t* table, const uint32_t* ids, uint8_t* dst_weight, uint8_t* dst_scales, float* slot_acc, uint32_t rows, uint32_t experts, uint32_t weight_bytes, uint32_t scale_bytes, uint32_t acc_row_floats; two gate/up launches: const uint8_t* act_fp8, const uint8_t* act_scales, const uint8_t* w_fp4, const uint8_t* w_scales, const uint32_t* slot_ids, __nv_bfloat16* out, int tokens; weighted SwiGLU: const __nv_bfloat16* gate, const __nv_bfloat16* up, const float* weights, uint8_t* out_fp8, uint8_t* out_scales, uint32_t rows, uint32_t cols, float limit; stored down: const uint8_t* codes, const uint8_t* scales, const uint8_t* w2_fp4, const uint8_t* w2_scales, const uint32_t* slot_ids, float* slot_acc, int tokens; combine: const float* slot_acc, const uint32_t* ids, float* delta_f32, uint16_t* delta_bf16, uint32_t tokens, uint32_t hidden)",
    },
    StageContract {
        stage: "all_reduce_moe",
        kernel: "comm_all_reduce_bf16",
        signature: "(cudaStream_t stream, const void* in, void* out, int count, void* handle)",
    },
    StageContract {
        stage: "moe_shared",
        kernel: "coded arm: b12x_mxfp8_gemv_m1_nf_sm121 (w1, w3) then swiglu_mxfp8_quant_sm121 then b12x_mxfp8_gemv_m1_nf_sm121 (w2); f32-reference arm: b12x_mxfp8_gemv_m1_f32a_f32o_sm121 (w1, w3, w2) and a device f32 SwiGLU (symbol not yet pinned)",
        signature: "(const float* normed, const uint8_t* w1_fp8, const uint8_t* w1_scales, const uint8_t* w3_fp8, const uint8_t* w3_scales, const uint8_t* w2_fp8, const uint8_t* w2_scales, void* shared_row, int dim, int inter, int shared_bf16). shared_bf16 = 1 is the coded arm's BF16 row (the epilogue widens it); shared_bf16 = 0 is the f32 arm's row. The f32 arm is f32-reference arithmetic on device and is NOT bit-identical to the host row: its silu uses expf where the host uses Rust exp, a declared D54 tolerance item.",
    },
    // The RDMA arm's kernel, pinned in this table so both transports are stated in one place. It
    // is not a stage: the NCCL arm never launches it, and selecting it under the NCCL name would
    // be choosing a transport by accident.
    StageContract {
        stage: "all_reduce_rdma_arm",
        kernel: "device_all_reduce_bf16",
        signature: "(cudaStream_t stream, const void* in, void* out, int bytes, void* handle)",
    },
    StageContract {
        stage: "ffn_epilogue",
        kernel: "mhc_ffn_epilogue",
        signature: "(const float* stream, const void* moe_delta, const void* shared_row, const float* mix_table, float* stream_out, uint32_t dim, uint32_t hc_mult, uint32_t rows, uint32_t shared_present, uint32_t shared_bf16, uint32_t native_bf16, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
    },
];

/// A kernel contract that a stage uses internally, rather than a stage of its own.
///
/// The two consumer stages of the layer's first device prefix (`query_path`,
/// `attention_projections`) both begin with a row-norm, and the projection begins with the
/// activation quantiser after it; putting those two kernels in their own table is what lets the
/// consumers be written against a fixed ABI before the kernels exist, and tells a reader who owns
/// which piece without opening the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelContract {
    /// The entry symbol the launcher loads.
    pub kernel: &'static str,
    /// The kernel's signature, as a refusal prints it.
    pub signature: &'static str,
    /// Who lands it.
    pub owner: &'static str,
    /// The stages that consume it.
    pub used_by: &'static str,
}

/// The kernels the layer's stages call internally. The row-norm is **landed** on this branch
/// through its typed launcher (see the module header); the prologue is driver-fixer's deliverable,
/// and the three kv-path kernels are the design piece that is not started inline.
pub const KERNEL_CONTRACTS: &[KernelContract] = &[
    KernelContract {
        kernel: "rmsnorm_sm121",
        signature: "(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out, unsigned rows, unsigned cols, float eps, unsigned long trace_ring, unsigned layer_id, unsigned long step_seq)",
        owner: "this branch (landed: sparkflash_kernels::rmsnorm_launch, nine named words, one thread per row, grid ceil(rows / 256))",
        used_by: "query_path (the q_norm over the q_lora latent), attention_projections (the row norm before the activation prologue)",
    },
    KernelContract {
        kernel: "w8a8_prologue_e4m3",
        signature: "(const float* normed, uint8_t* codes, uint8_t* scales, uint32_t cols, uint32_t rows, GpuTraceRing* trace_ring, uint32_t layer_id, uint64_t step_seq)",
        owner: "driver-fixer (assigned with mhc_entry/mhc_apply and ffn_seams)",
        used_by: "quantise_prologue; the prologue of attention_projections and query_path",
    },
    KernelContract {
        kernel: "shared_swiglu_f32",
        // Five words, no trace ring: (gate, up, out, limit, inter), grid ceil(inter/256), one row.
        signature: "(const float* gate, const float* up, float* out, float limit, uint32_t inter)",
        // Two declared deviations from the host's Rust SwiGLU, both inside the f32-reference arm and
        // both a D54 tolerance item, not bit-identical rows: the device silu is x/(1+expf(-x)) where
        // the host uses exp, and it divides where the host multiplies by a reciprocal. limit > 0
        // clamps gate to min(limit) and up to [-limit, limit] before the product, as the host does.
        owner: "driver-fixer (the seam image's seventh symbol)",
        used_by: "moe_shared (the default f32-reference arm's activation, in place of the host round trip)",
    },
    KernelContract {
        kernel: "rope_rotate_bf16",
        signature: "(const __nv_bfloat16* row, const float* cos, const float* sin, __nv_bfloat16* out, unsigned head_width, unsigned rope_head_dim, unsigned position)",
        owner: "comm-lead (implements the kv_path design piece behind the DeviceKvCache interface)",
        used_by: "kv_path (the rotated tail of the MLA latent)",
    },
    KernelContract {
        kernel: "window_quant_fp8_bf16",
        signature: "(const __nv_bfloat16* row, unsigned char* codes, unsigned char* scales, unsigned cols, unsigned block)",
        owner: "comm-lead (implements the kv_path design piece behind the DeviceKvCache interface)",
        used_by: "kv_path (the window row's E4M3/E8M0 round-trip)",
    },
    KernelContract {
        kernel: "kv_append_bf16",
        signature: "(const __nv_bfloat16* k_row, const __nv_bfloat16* v_row, __nv_bfloat16* cache, unsigned rows, unsigned head_dim, unsigned position)",
        owner: "comm-lead (implements the kv_path design piece behind the DeviceKvCache interface)",
        used_by: "kv_path (this position's rows appended to the device cache)",
    },
    KernelContract {
        kernel: "comm_all_reduce_bf16",
        signature: "(cudaStream_t stream, const __nv_bfloat16* in, __nv_bfloat16* out, int count, void* handle)",
        owner: "comm-lead (landed: task/gpu-reduce, open/warm_up/reduce/stats)",
        used_by: "all_reduce (attention delta), all_reduce_moe (routed MoE delta)",
    },
];

/// The device KV cache: **the device arm's own cache**, one per run.
///
/// Ownership, stated as the contract it is:
///
/// * `DeviceLayer::open` creates it, beside the layer slab and never inside it: the slab is the
///   layer's working set, the cache is the run's history, and they have different lifetimes.
/// * The device arm owns it for the run. The host arm keeps the host-resident cache it has always
///   used; this type is not a second view of that memory and never aliases it.
/// * No mutable cache is shared between the arms, and a run never mixes them: the arm is chosen once,
///   before the loop ([select_layer_arm]), so there is no path where one arm reads rows the other
///   wrote.
/// * The device arm's cache is never read back to the host in steady state. It is written by
///   `kv_append_bf16`, consumed on device by the loop's slice-copy into the attention
///   body's window buffers, and freed with the run; the only host reads are the run-boundary
///   numbers (sizes and the collective counter), never rows.
///
/// Visibility of an append and the window, stated rather than implied by implementation order.
/// **Corrected 2026-10-22 (device-lead \`42c1515f\`/\`ae095612\`, driver-fixer W1):** the window includes
/// the current token. At position \`p\` the ring holds \`p + 1\` rows; \`attention_stage::window_span(p, 128)\`
/// is the one derivation of \`window_rows = (p + 1).min(128)\` and \`window_start = p + 1 - window_rows\`,
/// so the attended slice is \`[p + 1 - window_rows ..= p]\`. The compressed-position offset used by the
/// id mapping is \`window_rows\` (ids \`window_rows + j\`), **not** \`window_start\`, which slides by up to
/// 127. Append-then-read is therefore a requirement, and \`assemble\` makes it observable:
/// \`ring_rows == p + 1\` is the normal state and a position past the ring is refused as
/// \`PositionPastRing\` before the append.
///
/// An append becomes visible to the attention body only when the read is enqueued on the same
/// stream after the append, in position order; the reference seam does **not** read the cache (it
/// decides \`window_width\`). There is no cross-stream publication, no host-visible copy and no
/// synchronise between them: stream order is the publication. \`kv_path\` appends row \`p\` and the
/// body reads through \`p\`, which is why the append sits before the seam in \`run_layer\`.
/// (\`kv_append_rule\`'s \`cache_len == position\` guard is the length **before** this position's append,
/// so the guard itself is unchanged.)
///
/// The one opaque thing: [`DeviceKvCache::device`] is the only address this type carries. Layers are
/// addressed by offset through [`DeviceKvCache::layer_span`]; there is no per-layer table, no second
/// pointer and no host mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceKvCache {
    /// The cache allocation's device address: the run's one opaque word for the cache.
    pub device: u64,
    /// Rows per layer: the window plus the prefill capacity the host cache is sized for.
    pub capacity_rows: usize,
    /// BF16 values per row: two (k and v) times the head width.
    pub row_values: usize,
    /// Layers the cache carries (the config's layer count).
    pub n_layers: usize,
}

impl DeviceKvCache {
    /// The alignment of the cache base and of every layer's row run: the widest vector the seam
    /// kernels load.
    pub const ALIGN: usize = ALIGN;

    /// The allocation a cache of this shape needs, in bytes.
    #[must_use]
    pub fn bytes(n_layers: usize, capacity_rows: usize, row_values: usize) -> usize {
        n_layers.saturating_mul(Self::layer_bytes_for(capacity_rows, row_values))
    }

    /// The bytes of one layer's run: the raw row bytes rounded up to ALIGN, so every
    /// layer starts aligned. The rounding is per layer and never of the row count: aligning the
    /// *count* is what made an earlier version over-allocate up to 2x when the count was not a
    /// multiple of ALIGN.
    #[must_use]
    pub fn layer_bytes_for(capacity_rows: usize, row_values: usize) -> usize {
        align_up(capacity_rows.saturating_mul(row_values).saturating_mul(2))
    }

    /// Checks a cache's shape against an address, so a wrong carve is refused by name.
    ///
    /// # Errors
    /// A zero layer count, capacity or row width, or a misaligned base.
    pub fn new(
        device: u64,
        n_layers: usize,
        capacity_rows: usize,
        row_values: usize,
    ) -> Result<Self, String> {
        if n_layers == 0 || capacity_rows == 0 || row_values == 0 {
            return Err(format!(
                "device kv cache: a {n_layers}-layer cache of {capacity_rows} rows x {row_values} values is empty"
            ));
        }
        if !device.is_multiple_of(Self::ALIGN as u64) {
            return Err(format!(
                "device kv cache: the base {device:#x} is not {}-byte aligned",
                Self::ALIGN
            ));
        }
        Ok(Self {
            device,
            capacity_rows,
            row_values,
            n_layers,
        })
    }

    /// The bytes of one layer's row run.
    #[must_use]
    pub fn layer_bytes(&self) -> usize {
        Self::layer_bytes_for(self.capacity_rows, self.row_values)
    }

    /// The span one layer's rows occupy: what `kv_append_bf16` writes and the window seam reads.
    ///
    /// # Errors
    /// A layer outside the cache is refused by index, never clamped.
    pub fn layer_span(&self, layer: usize) -> Result<DeviceSpan, String> {
        if layer >= self.n_layers {
            return Err(format!(
                "device kv cache: layer {layer} of {} requested",
                self.n_layers
            ));
        }
        Ok(DeviceSpan {
            device: self.device + (layer * self.layer_bytes()) as u64,
            len: self.layer_bytes(),
        })
    }

    /// The bf16 values one position appends: the k/v pair for one head row.
    #[must_use]
    pub fn append_values(&self) -> usize {
        self.row_values
    }
}

/// Which stages the device arm actually ran, and where it stopped: the run record's frontier.
///
/// The record must name the frontier rather than a binary "device arm" flag, the same way a
/// refusal names the stage and its blocker. A run whose body executed and whose later stage
/// refused is self-describing from the record alone.
#[derive(Debug, Default, Clone)]
pub struct StageRuns {
    ran: Vec<&'static str>,
    refused: Option<(String, String)>,
}

impl StageRuns {
    /// Notes that a stage ran device-only.
    pub fn ran(&mut self, stage: &'static str) {
        self.ran.push(stage);
    }

    /// Notes the stage the arm stopped at, with the refusal text.
    pub fn refused(&mut self, stage: &str, message: &str) {
        self.refused = Some((stage.to_owned(), message.to_owned()));
    }

    /// Records one stage's outcome: `ran` on success, `refused` with the stage and text on failure.
    ///
    /// This is the only way the frontier is built where the layer object exists: `from_refusal` is
    /// for callers that hold nothing but a message, and using it here would report that no stage ran
    /// even when the chain reached the stage before the refusal.
    ///
    /// # Errors
    /// The stage's own error, unchanged.
    pub fn record<T>(
        &mut self,
        stage: &'static str,
        result: Result<T, String>,
    ) -> Result<T, String> {
        match result {
            Ok(value) => {
                self.ran(stage);
                Ok(value)
            }
            Err(message) => {
                self.refused(stage, &message);
                Err(message)
            }
        }
    }

    /// The frontier for a device layer that refused: the stage the message names, and the text.
    #[must_use]
    pub fn from_refusal(message: &str) -> Self {
        let stage = message
            .split_once("layer device stage ")
            .and_then(|(_head, rest)| rest.split_once(':'))
            .map(|(stage, _)| stage.trim().to_owned())
            .unwrap_or_else(|| "unknown stage".to_owned());
        let mut runs = Self::default();
        runs.refused(&stage, message);
        runs
    }

    /// The stages that ran, in order.
    #[must_use]
    pub fn stages(&self) -> &[&'static str] {
        &self.ran
    }

    /// The one line the record carries.
    #[must_use]
    pub fn summary(&self) -> String {
        let ran = if self.ran.is_empty() {
            "no stage ran".to_string()
        } else {
            format!(
                "device-only through {} ({} stage(s))",
                self.ran.join(" -> "),
                self.ran.len()
            )
        };
        match &self.refused {
            Some((stage, message)) => format!("{ran}; stopped at {stage}: {message}"),
            None => ran,
        }
    }
}

/// The arm-private rope tables' bytes for a run: per TABLE CLASS (plain and compressed, at most
/// two), cos and sin, 32 f32 pairs each position (128 B per table). The tables are per class, not
/// per layer: compression is the only per-layer dependence, so two uploads cover forty layers and
/// this count is reported next to the slab rather than left implicit.
///
/// The tables are NOT slab spans: they are built host-side from
/// sparkflash_engine::forward::attention::rope_tables_for_layer (which already applies
/// compress_rope_theta and the YaRN ramp per layer) and uploaded once at open.
#[must_use]
pub fn rope_table_bytes(table_classes: usize, max_positions: usize) -> usize {
    let pairs = RELEASED_ROPE_HEAD_DIM / 2;
    table_classes * 2 * max_positions * pairs * 4
}

/// The released rotary tail: 64 wide, 32 f32 pairs.
pub const RELEASED_ROPE_HEAD_DIM: usize = 64;
/// Checks the run's collective counter against the count the loop's own sequence implies.
///
/// A rank that skips a collective does not fail a check: it hangs the reduce, and the counter is the
/// only host-visible trace that all four ranks enqueued the same sequence. So a mismatch is refused
/// loudly at the run boundary - the run is not correct and its numbers are not to be reported -
/// rather than printing both numbers and moving on.
///
/// # Errors
/// Any mismatch, with both numbers and the per-layer count they imply.
pub fn verify_collective_count(expected: usize, actual: usize) -> Result<(), String> {
    if expected == actual {
        return Ok(());
    }
    Err(format!(
        "collective count mismatch: the layer sequence implies {expected} collectives \
         ({COLLECTIVES_PER_LAYER} per layer) and this rank enqueued {actual}. A rank that skips a \
         collective hangs the reduce instead of failing it, so this counter is the only host-visible \
         trace of the divergence: the run is not correct and its numbers must not be reported. This \
         is a refusal, not a warning."
    ))
}

/// Checks that every rank enqueued the same number of collectives.
///
/// A per-rank mismatch is already fatal ([verify_collective_count]); a *cross-rank* mismatch is the
/// precise signature of a skipped collective - NCCL will hang on the next reduce rather than report
/// it - so the run boundary refuses both, and refuses before it reports a token.
///
/// # Errors
/// Ranks that disagree, or an empty rank list, naming the rank indices and their counts.
pub fn verify_collective_counts_agree(counts: &[usize]) -> Result<(), String> {
    let Some((&first, rest)) = counts.split_first() else {
        return Err("collective counts: no ranks to check".to_owned());
    };
    if let Some((index, _)) = rest.iter().enumerate().find(|(_, count)| **count != first) {
        let diverged: Vec<String> = rest
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != first)
            .map(|(offset, count)| format!("rank {}: {count}", offset + 1))
            .collect();
        return Err(format!(
            "collective counts disagree across ranks: rank 0 enqueued {first} and {} (first at rank \
             {}). The ranks are not running the same sequence, which is what a skipped collective \
             looks like before NCCL hangs on it. The run is not correct and its numbers must not be \
             reported.",
            diverged.join(", "),
            index + 1
        ));
    }
    Ok(())
}

/// The activation prologue's span check, and nothing else: the launch belongs to
/// driver-fixer's typed launcher (`layer_seams_launch::launch_prologue`), whose ABI is **eight
/// words**: `(normed, codes, scales, cols, rows, trace_ring, layer_id, step_seq)` - every seam
/// kernel carries the ABI-uniform trace tail.
///
/// Steady state passes a **null trace ring**, and a null ring means the kernel writes nothing
/// (the source-level `if (trace_ring != nullptr)` guard is pinned the same way as
/// `ptx_membar_lines=0`). Never pass a dummy ring to exercise it: the seam runs forty-plus times
/// per position and a trace store in the steady state is exactly the host-visible traffic this
/// port exists to remove. Layer id and step sequence come from the loop's probe context.
///
/// [PrologueLaunch::new] checks the spans against the geometry before any launch, so a wrong row
/// width is refused by name even while the cubin is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrologueLaunch {
    /// The bf16 row to quantise: `[rows][cols]`.
    pub row: DeviceSpan,
    /// The E4M3 codes: `[rows][cols]`, one byte per value.
    pub codes: DeviceSpan,
    /// The E8M0 scales: `[rows][ceil(cols / 32)]`, one byte per block.
    pub scales: DeviceSpan,
    /// Columns per row.
    pub cols: u32,
    /// Rows in the launch.
    pub rows: u32,
}

impl PrologueLaunch {
    /// Checks the spans of a launch whose row is BF16, against the geometry.
    ///
    /// The dtype is part of the contract because the check must read what the kernel reads: a
    /// BF16 row needs two bytes per value, and accepting a span on the f32 width is how
    /// query_path's latent prologue once authorised a 5,120-byte read out of a 2,560-byte span.
    /// Callers launching the f32 prologue must use [PrologueLaunch::new_f32].
    ///
    /// # Errors
    /// A zero row or column count, a row span shorter than the dtype needs, or a codes/scales
    /// span too small for the geometry: each refused with the numbers that failed.
    pub fn new(
        row: DeviceSpan,
        codes: DeviceSpan,
        scales: DeviceSpan,
        rows: u32,
        cols: u32,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err(format!(
                "activation prologue: {rows} rows x {cols} columns is empty"
            ));
        }
        let (rows_usize, cols_usize) = ((rows as usize), (cols as usize));
        let values = rows_usize
            .checked_mul(cols_usize)
            .ok_or_else(|| format!("activation prologue: {rows} x {cols} values overflow"))?;
        if row.len < values * 2 {
            return Err(format!(
                "activation prologue: {rows} x {cols} bf16 needs {} bytes, the row span is {}",
                values * 2,
                row.len
            ));
        }
        if codes.len < values {
            return Err(format!(
                "activation prologue: {rows} x {cols} E4M3 codes need {values} bytes, the span is {}",
                codes.len
            ));
        }
        let scale_blocks = rows_usize * cols_usize.div_ceil(32);
        if scales.len < scale_blocks {
            return Err(format!(
                "activation prologue: {} scale blocks need {scale_blocks} bytes, the span is {}",
                scale_blocks, scales.len
            ));
        }
        Ok(Self {
            row,
            codes,
            scales,
            cols,
            rows,
        })
    }

    /// Checks a prologue launch whose row is F32, against the geometry.
    ///
    /// The f32 prologue kernel (w8a8_prologue_e4m3) reads four bytes per value; a check on the
    /// bf16 width would authorise it on a half-size span. Callers pass the f32 row (the normed
    /// residual, the seam row), not a bf16 copy.
    ///
    /// # Errors
    /// A zero row or column count, a row span shorter than the f32 width, or a codes/scales span
    /// too small for the geometry.
    pub fn new_f32(
        row: DeviceSpan,
        codes: DeviceSpan,
        scales: DeviceSpan,
        rows: u32,
        cols: u32,
    ) -> Result<Self, String> {
        if rows == 0 || cols == 0 {
            return Err(format!(
                "activation prologue: {rows} rows x {cols} columns is empty"
            ));
        }
        let (rows_usize, cols_usize) = ((rows as usize), (cols as usize));
        let values = rows_usize
            .checked_mul(cols_usize)
            .ok_or_else(|| format!("activation prologue: {rows} x {cols} values overflow"))?;
        if row.len < values * 4 {
            return Err(format!(
                "activation prologue: {rows} x {cols} f32 needs {} bytes, the row span is {}",
                values * 4,
                row.len
            ));
        }
        if codes.len < values {
            return Err(format!(
                "activation prologue: {rows} x {cols} E4M3 codes need {values} bytes, the span is {}",
                codes.len
            ));
        }
        let scale_blocks = rows_usize * cols_usize.div_ceil(32);
        if scales.len < scale_blocks {
            return Err(format!(
                "activation prologue: {} scale blocks need {scale_blocks} bytes, the span is {}",
                scale_blocks, scales.len
            ));
        }
        Ok(Self {
            row,
            codes,
            scales,
            cols,
            rows,
        })
    }
}

/// The refusal a stage returns when its kernel is not on this branch yet: names the stage, the
/// kernel and the ABI, so a run says what device-lead still owes instead of falling back to the host.
#[must_use]
pub fn not_landed(stage: &str, kernel: &str, signature: &str) -> String {
    format!(
        "layer device stage {stage}: kernel {kernel} is not landed on this branch; contract: {signature}"
    )
}

/// The device step a stage still needs before its kernel can run: the host work that has to become
/// device work first, per stage. An empty string means the stage only needs its kernel - the launch
/// is the whole of it.
///
/// This is the frontier, stated where the kernels are listed: on this branch the first two stages of
/// a layer are the ones that block, because the activation quantiser and the mHC/FFN seams are host
/// arithmetic whose device kernels are being landed by device-lead. The rows are also the order the
/// port can honestly land in, so a reader can see that `attention_body` cannot ship before
/// `w8a8_prologue_e4m3`: its input row is quantised by the stage before it.
pub const STAGE_BLOCKERS: &[(&str, &str)] = &[
    (
        "engram_write",
        "none: the six-launch chain is zero copies - the gather and the activation quantiser moved to open, the arena entry is indexed on the device, and the stage's only refusals are setup facts (the arena was not built at open) or a position outside the arena",
    ),
    (
        "mhc_entry",
        "none: the host SIMD projection and the four mixes are the whole stage, and the kernel replaces them",
    ),
    (
        "quantise_prologue",
        "none: this stage is the host activation quantise becoming the device prologue",
    ),
    (
        "attention_projections",
        "the host activation quantise of the normed row (quantise_native_activation) must be the w8a8 prologue; the GEMV itself is a device launch already",
    ),
    (
        "query_path",
        "the host q_norm (engine rms_norm on the wq_a latent) and the host activation quantise must both move on device; the wq_b decode GEMV is a device launch already",
    ),
    (
        "kv_path",
        "a design piece, not a port: the host path is engine rms_norm + rope_rotate + the window fp8 round-trip over a host-resident cache, so it needs three kernels (rope_rotate_bf16, window_quant_fp8_bf16, kv_append_bf16 - see KERNEL_CONTRACTS) and, the real cost, the cache moving to device ownership with the window-consuming seam following it. Estimate ~500 lines across kernel, launcher, engine cache handle and loop wiring: one to two days, not an inline port",
    ),
    (
        "reference_seam",
        "wired for the released ratio-0 layers (0 and 1): the seam is the sliding-window metadata and an empty id list there, with no launch and no copy. Layers 2..39 (ratio 2 then 1) are a port, not wiring: the compressor's wkv/wgate and the indexer's wk/weights_proj are AttentionHostSeam in the plan (not bound into the arena), and the compressed side needs pool_group/narrow, the compressed append and the candidate select on device",
    ),
    (
        "ref_select",
        "wired as the identity no-op for ratio-0 layers: the seam's id list is empty there, so there is nothing to select. Ratio>0 needs the indexer key/weight class in the arena bind (AttentionHostSeam today) and the pinned ref_select_sm121 launch over the window ring; the kernel itself is landed and pinned",
    ),
    (
        "stage_attention",
        "wired as validation for ratio-0 layers: the window rows are already in the device ring and the stage checks the ring row count, the window bounds and the empty candidate list without copying anything. Ratio>0 also stages the selected ids and the compressed store rows, and the compressed-store producer is the reference_seam port above",
    ),
    (
        "attention_body",
        "none for ratio 0: the sequence entry's launch, its operands (the run-wide sink pool, the zero index-key pool and the per-layer indexer spans) and the inverse-rope symbol are all on device, so the stage is two launches and no readback. Ratio > 0 additionally needs the selected ids from ref_select and the compressed store from reference_seam; both are blocked above",
    ),
    (
        "attention_output",
        "none: the wo_a, activation-quantise and wo_b launches are the stage, with no readback and no host quantise",
    ),
    (
        "all_reduce",
        "none: comm-lead's DeviceReduce takes the two spans, the element count and the stream; NCCL enqueues on the stream and keeps nothing per call",
    ),
    (
        "attention_residual",
        "none: the host SIMD residual is the whole stage",
    ),
    (
        "ffn_seams",
        "none: the host SIMD FFN seams are the whole stage",
    ),
    (
        "moe_router",
        "the pinned pair and router_launch exist and the step path calls them; the arm still lacks the router's weight rows (gate bf16 [n_routed][hidden], bias f32 [n_routed]) and a bf16 activation row for x (seam_row is f32), and its scores span is shared with workspace.logits",
    ),
    (
        "moe_routed",
        "none: the routed data path is eight launches and zero copies per layer - three staging gathers from the run-scope expert pointer table, the two landed gate/up launches, the landed weighted SwiGLU, the landed stored down projection and the device combine - with no host combine and no readback. Its only refusals are setup facts (the path was not opened, or the ids are stale because the router did not run)",
    ),
    (
        "all_reduce_moe",
        "none: the second collective, same call shape; in is the routed delta, out is workspace.moe_reduced_bf16, count is dim",
    ),
    (
        "moe_shared",
        "the three dense shared-expert GEMVs are the same 11-word decode GEMV and the swiglu quantiser epilogue exists; what is missing is the gathered w1/w3/w2 weight+scale pairs in the arm (one gather per layer at open, like the host) and the gate/up intermediate spans with their codes and scales",
    ),
    (
        "all_reduce_rdma_arm",
        "none: the RDMA arm's kernel contract; listed here, never called by the NCCL arm",
    ),
    (
        "ffn_epilogue",
        "the host shared-expert/FFN epilogue seams are the step (the shared expert's launches are already on device)",
    ),
];

/// The refusal a stage returns when the device step it still needs is the blocker, not only the
/// kernel: names the stage, the kernel, the ABI, and what has to move on device first.
#[must_use]
pub fn not_landed_with_blocker(
    stage: &str,
    kernel: &str,
    signature: &str,
    blocker: &str,
) -> String {
    if blocker.is_empty() {
        not_landed(stage, kernel, signature)
    } else {
        format!(
            "{}; this stage also still needs: {blocker}",
            not_landed(stage, kernel, signature)
        )
    }
}

/// Which arm the layer loop runs, and why.
///
/// The device arm is the default since S2; the host arm is the bit-exact reference, selectable with
/// `SPARKFLASH_LAYER_DEVICE=0`, and the only arm that can record evidence, because it is the only
/// one that reads back per position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerArm {
    /// The bit-exact host sequence, with the reason it was selected.
    Host(&'static str),
    /// The device-only layer: the default, or `SPARKFLASH_LAYER_DEVICE=1` said explicitly.
    Device,
}

impl LayerArm {
    /// Whether this arm runs a layer on device pointers.
    #[must_use]
    pub fn is_device(self) -> bool {
        matches!(self, Self::Device)
    }

    /// Why the arm was chosen, for the run record.
    #[must_use]
    pub fn why(self) -> &'static str {
        match self {
            Self::Host(reason) => reason,
            Self::Device => {
                "device layer (the default since S2): no per-layer PCIe copy, no per-stage observation read"
            }
        }
    }
}

/// Resolves the arm.
///
/// Evidence recording needs the per-position host readbacks the device arm does not make, so a
/// recording run is forced to the host arm and the record says so: never a silent switch, and never
/// a device arm that claims to record and returns nothing.
#[must_use]
pub fn select_layer_arm(device_switch: bool, evidence_recording: bool) -> LayerArm {
    if evidence_recording {
        return LayerArm::Host(
            "host arm (evidence recording forces it: the device arm has no per-position readback)",
        );
    }
    if device_switch {
        LayerArm::Device
    } else {
        LayerArm::Host(
            "host arm (selected by SPARKFLASH_LAYER_DEVICE=0: the bit-exact host sequence; the device arm is the default)",
        )
    }
}

/// The run's one arm decision, from the two raw switch values.
///
/// The chain reads each environment variable exactly once, passes the raw values here, and carries
/// the result in the request; nothing below (the loop, the bootstrap, the evidence chain) decides the
/// arm again. This exists because two independent decisions can drift: evidence recording forces the
/// host arm, so a bootstrap that keys on the device switch alone would open a communicator for the
/// arm that must never need libnccl.
///
/// # Errors
/// A device switch spelling that names neither arm ([`parse_device_switch`]); the chain refuses the
/// run at startup with it rather than picking an arm.
pub fn decide_layer_arm(
    device: Option<&str>,
    evidence: Option<&str>,
) -> Result<(LayerArm, bool), String> {
    let evidence = crate::tp4_forward::evidence_enabled_from_env(evidence);
    Ok((
        select_layer_arm(parse_device_switch(device)?, evidence),
        evidence,
    ))
}

/// The device layer: one implementor per stage, sequenced by [`LayerStages::run_layer`].
///
/// The boundary methods (`begin_position`, `residual_out`, `attn_pre_mix_out`) are the only copies:
/// one H2D and at most two D2H per position, all outside the per-layer loop. Everything else is a
/// device-pointer hand-off.
pub trait LayerStages {
    /// The spans this implementor owns (see [`LayerBuffers`]).
    fn buffers(&self) -> &LayerBuffers;

    /// Writes position `position`'s residual (`hc_mult` copies of the prompt row) to
    /// `buffers().residual.stream`, and the identity seed mix to `residual.pre_mix`. The layer
    /// loop's first half of the boundary copy.
    fn begin_position(&mut self, position: usize, hidden: &[f32]) -> Result<(), String>;

    /// One whole layer, device-only, in the order of [`STAGE_CONTRACTS`]. Nothing here may read the
    /// residual back to the host.
    ///
    /// **Divergence rule (binding):** every refusal below must be deterministic from the plan and the
    /// kernel set - identical on all four ranks - or happen at open. No stage in the timed region
    /// aborts on a per-stage handshake, and an unanticipated abort (OOM, Xid, fault) is detectable
    /// and void through the outer timeout and the on-node kill, never tolerated.
    fn run_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        self.engram_write(layer, position)?;
        self.mhc_entry(layer, position)?;
        self.quantise_prologue(layer, position)?;
        self.attention_projections(layer, position)?;
        self.query_path(layer, position)?;
        self.kv_path(layer, position)?;
        self.reference_seam(layer, position)?;
        self.ref_select(layer, position)?;
        self.stage_attention(layer, position)?;
        self.attention_body(layer, position)?;
        self.attention_output(layer, position)?;
        self.all_reduce(layer, position)?;
        self.attention_residual(layer, position)?;
        self.ffn_seams(layer, position)?;
        self.moe_router(layer, position)?;
        self.moe_routed(layer, position)?;
        self.all_reduce_moe(layer, position)?;
        self.moe_shared(layer, position)?;
        self.ffn_epilogue(layer, position)
    }

    /// The Engram write for the layers that have rows (1 and 14 today): `stream += gate * value` on
    /// the device. A no-op for a layer without rows.
    fn engram_write(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The mHC entry's projection and 4-way mixes: host SIMD today, `mhc_entry_bf16` here.
    fn mhc_entry(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The activation quantiser, as the projection's device prologue (no host E4M3 conversion).
    fn quantise_prologue(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The attention projections (wq_b / wo_a pair) reading the device codes and scales.
    fn attention_projections(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The query path: the row norm and the q/k/v GEMVs.
    fn query_path(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The kv path: appends this position's rows to the device cache.
    fn kv_path(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The reference seam: the compressor's window selection on device.
    fn reference_seam(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The reference candidate select: the indexer's top-k over the window, on device.
    ///
    /// Its kernel and CPU twin are pinned (ref_select_sm121, 18 words); the stage line is here so
    /// the contract is 20 stages and the consumer's refusal names it, not inside stage_attention.
    fn ref_select(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The attention staging, on device (the 80 KiB rows never come back to the host).
    fn stage_attention(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The attention body.
    fn attention_body(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The attention output projections (wo_a / wo_b GEMV).
    fn attention_output(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The cross-rank all-reduce of `workspace.attn_out_bf16` into `workspace.reduced_bf16`: one
    /// per-layer call to the device communicator, taking device addresses and the stream.
    /// **The call does not synchronise and does not read back**: the host hands the two spans and
    /// the run's communicator handle to the device and queues the next stage behind it on the
    /// same stream, so the layer loop never waits for a collective. The handle is the run's,
    /// the spans are this layer's (see the ownership rules above).
    ///
    /// **Divergence rule for both collectives** (Lead's): a stage refusal must be
    /// deterministic from the plan and the kernel set - identical on all four ranks - or it must
    /// happen at open. The only refusal here is the run-level missing reduce handle, a property of
    /// the run's setup, so all four ranks refuse it together or none does. There is no
    /// per-stage abort handshake in the timed region: a refusal that could fire on one rank alone
    /// in front of a collective with no timeout is a bug, and an unanticipated abort is made
    /// detectable and void by the outer timeout rather than tolerated.
    fn all_reduce(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The mHC attention residual: host SIMD today, a device kernel here.
    fn attention_residual(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The FFN entry seams (the FFN pre/post mixes and the hc_pre), on device.
    fn ffn_seams(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The MoE router: device ids, weights and logits; no host pass over the router's outputs.
    fn moe_router(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The routed experts, including the combine of their outputs on device.
    fn moe_routed(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The second per-layer collective: the routed MoE delta in `workspace.moe_reduced_bf16`. Same call
    /// shape as `all_reduce`: device addresses, `count` = dim elements, queued on the stream, no sync,
    /// no readback. `ffn_epilogue` reads its output as the MoE delta.
    fn all_reduce_moe(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The shared expert: the dense w1/w3/swiglu/w2 MLP on the normed row at `seam_row` base,
    /// overwriting that base with the shared row (f32 by default, bf16 only under native-rounding
    /// shared). The weights are gathered once at open, so this is a single-rank computation and it
    /// adds **no** collective: the two-collective-per-layer contract is unchanged.
    fn moe_shared(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// The FFN epilogue: the shared expert and the hc_post, ending in the next layer's residual.
    fn ffn_epilogue(&mut self, layer: usize, position: usize) -> Result<(), String>;

    /// Reads the residual back once per position, for the certified head's boundary. The layer
    /// loop's second half of the boundary copy.
    fn residual_out(&mut self) -> Result<Vec<f32>, String>;

    /// Reads the mix the head needs (`residual.pre_mix`) once per position.
    fn attn_pre_mix_out(&mut self) -> Result<Vec<f32>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released geometry, in the one value the carve now takes.
    fn geometry() -> LayerGeometry {
        LayerGeometry {
            dim: 5120,
            hc_mult: 4,
            act_row: 8192,
            topk: 8,
            vocab_pad: 129_280,
            q_cols: 8192,
            max_positions: 2048,
        }
    }

    #[test]
    fn the_layer_allocation_is_contiguous_aligned_and_disjoint() {
        let total = LayerBuffers::bytes(&geometry());
        let buffers = LayerBuffers::carve(0x1_0000, &geometry());
        let spans = [
            buffers.residual.stream,
            buffers.residual.hc_rows,
            buffers.residual.normed,
            buffers.residual.pre_mix,
            buffers.workspace.act_codes,
            buffers.workspace.act_scales,
            buffers.workspace.attn_out_bf16,
            buffers.workspace.reduced_bf16,
            buffers.workspace.moe_reduced_bf16,
            buffers.workspace.seam_row,
            buffers.workspace.moe_delta,
            buffers.workspace.moe_delta_bf16,
            buffers.workspace.ffn_x_bf16,
            buffers.workspace.moe_ids,
            buffers.workspace.moe_weights,
            buffers.workspace.logits,
            buffers.workspace.reduce_ws,
            buffers.workspace.kv_rotated,
            buffers.workspace.kv_latent,
            buffers.workspace.attn_mix,
            buffers.workspace.ffn_mix,
            buffers.workspace.q_lora,
            buffers.workspace.q_normed,
            buffers.workspace.q,
            buffers.workspace.attn_body_bf16,
            buffers.workspace.attn_lo,
            buffers.workspace.selected,
            buffers.workspace.store_codes,
            buffers.workspace.store_scales,
            buffers.workspace.index_query,
            buffers.workspace.block_scores,
            buffers.workspace.block_best,
            buffers.workspace.rope_tables_plain,
            buffers.workspace.rope_tables_compressed,
            buffers.workspace.dspark_aux_taps,
        ];
        let mut end = 0u64;
        for span in spans {
            assert_eq!((span.device - 0x1_0000) % ALIGN as u64, 0, "span {span:?}");
            assert_eq!(span.device, 0x1_0000 + end, "spans must be contiguous");
            end += span.len as u64;
        }
        assert_eq!(end as usize, total, "carve must spend exactly bytes()");
    }

    #[test]
    fn the_seam_row_and_the_residual_are_the_documented_sizes() {
        let buffers = LayerBuffers::carve(0, &geometry());
        assert_eq!(buffers.workspace.seam_row.len, align_up(4 * 5120 * 4));
        assert_eq!(buffers.workspace.moe_delta.len, align_up(5120 * 4));
        assert_eq!(buffers.workspace.moe_delta_bf16.len, align_up(5120 * 2));
        assert_eq!(
            buffers.workspace.ffn_x_bf16.len,
            align_up(5120 * 2),
            "the pre-FFN hidden row the router reads"
        );
        assert_eq!(buffers.residual.stream.len, align_up(4 * 5120 * 4));
        assert_eq!(buffers.workspace.reduce_ws.len, align_up(2 * 5120 * 4));
        assert_eq!(buffers.workspace.kv_rotated.len, align_up(KV_ROTATED_BYTES));
        assert_eq!(buffers.workspace.kv_latent.len, align_up(KV_LATENT_BYTES));
        assert_eq!(buffers.workspace.attn_mix.len, align_up(ATTN_MIX_BYTES));
        assert_eq!(buffers.workspace.ffn_mix.len, align_up(FFN_MIX_BYTES));
        assert_eq!(buffers.workspace.q_lora.len, align_up(Q_LORA_BYTES));
        assert_eq!(buffers.workspace.q_normed.len, align_up(Q_NORMED_BYTES));
        assert_eq!(buffers.workspace.q.len, align_up(16_384));
        assert_eq!(
            buffers.workspace.rope_tables_plain.len,
            align_up(2_048 * 256)
        );
        assert_eq!(
            buffers.workspace.rope_tables_compressed.len,
            align_up(2_048 * 256)
        );
        assert_eq!(
            buffers.workspace.moe_ids.len, ALIGN,
            "small spans are padded"
        );
        assert_eq!(buffers.workspace.act_scales.len, ALIGN);
    }

    #[test]
    fn the_arm_selection_never_records_evidence_on_the_device_arm() {
        assert!(!select_layer_arm(false, false).is_device());
        assert!(select_layer_arm(true, false).is_device());
        // Evidence recording forces the host arm, and it says why in the record.
        let forced = select_layer_arm(true, true);
        assert!(!forced.is_device());
        assert!(
            forced.why().contains("evidence recording forces it"),
            "{}",
            forced.why()
        );
        assert!(
            select_layer_arm(false, false)
                .why()
                .contains("SPARKFLASH_LAYER_DEVICE=0")
        );
        assert!(
            select_layer_arm(true, false)
                .why()
                .contains("the default since S2")
        );
    }

    #[test]
    fn the_kernel_table_states_the_first_prefix_and_the_kv_path_pieces() {
        let kernels: Vec<&str> = KERNEL_CONTRACTS.iter().map(|entry| entry.kernel).collect();
        assert_eq!(
            kernels,
            vec![
                "rmsnorm_sm121",
                "w8a8_prologue_e4m3",
                "shared_swiglu_f32",
                "rope_rotate_bf16",
                "window_quant_fp8_bf16",
                "kv_append_bf16",
                "comm_all_reduce_bf16",
            ]
        );
        for entry in KERNEL_CONTRACTS {
            assert!(!entry.signature.is_empty(), "{}", entry.kernel);
            assert!(!entry.owner.is_empty(), "{}", entry.kernel);
            assert!(!entry.used_by.is_empty(), "{}", entry.kernel);
        }
        // The row-norm is the one entry this branch owns and has landed; the prologue names its
        // owner; the kv path is three kernels, not a port. That split is the plan, in the code.
        let row_norm = KERNEL_CONTRACTS
            .iter()
            .find(|entry| entry.kernel == "rmsnorm_sm121")
            .expect("the row-norm row");
        assert!(row_norm.owner.contains("landed"), "{}", row_norm.owner);
        assert!(
            KERNEL_CONTRACTS
                .iter()
                .find(|entry| entry.kernel == "w8a8_prologue_e4m3")
                .expect("the prologue row")
                .owner
                .contains("driver-fixer")
        );
        assert_eq!(
            KERNEL_CONTRACTS
                .iter()
                .filter(|entry| entry.owner.contains("kv_path"))
                .count(),
            3,
            "the kv path is three kernels: rope, the window quantiser and the cache write"
        );
    }

    #[test]
    fn the_reduce_scratch_and_the_layer_slab_are_the_numbers_the_record_prints() {
        let buffers = LayerBuffers::carve(0, &geometry());
        assert_eq!(buffers.total_bytes(), LayerBuffers::bytes(&geometry()));
        assert!(
            buffers.workspace.reduce_ws.len >= REDUCE_ELEMENTS * 2,
            "the warm-up scratch must hold one reduce payload"
        );
        assert_eq!(COLLECTIVES_PER_LAYER, 2);
        assert_eq!(REDUCE_ELEMENTS, 5_120);
    }

    #[test]
    fn the_device_kv_cache_is_one_opaque_word_and_refuses_a_wrong_shape() {
        let cache = DeviceKvCache::new(256, 40, 512, 128).expect("a well-shaped cache");
        assert_eq!(cache.layer_span(0).map(|span| span.device), Ok(256));
        assert_eq!(cache.layer_span(0).map(|span| span.len), Ok(512 * 128 * 2));
        assert_eq!(
            cache.layer_span(39).map(|span| span.device),
            Ok(256 + 39 * 512 * 128 * 2)
        );
        assert_eq!(cache.append_values(), 128);
        assert_eq!(cache.layer_bytes(), 512 * 128 * 2);
        // The one opaque word is the base: no per-layer table, no host mirror.
        assert_eq!(cache.device, 256);
        // Refusals by name, never clamps.
        assert!(cache.layer_span(40).is_err());
        assert!(DeviceKvCache::new(256, 40, 0, 128).is_err());
        assert!(DeviceKvCache::new(256, 0, 512, 128).is_err());
        assert!(DeviceKvCache::new(257, 40, 512, 128).is_err());
        assert!(DeviceKvCache::bytes(40, 512, 128) >= cache.layer_bytes() * 40);
    }

    #[test]
    fn the_prologue_span_check_refuses_a_wrong_geometry() {
        let row = DeviceSpan::new(0x1000, 4 * 5120 * 2);
        let codes = DeviceSpan::new(0x8000, 4 * 5120);
        let scales = DeviceSpan::new(0xc000, 4 * 160);
        PrologueLaunch::new(row, codes, scales, 4, 5120).expect("a shaped launch");
        assert!(
            PrologueLaunch::new(row, codes, DeviceSpan::new(0xc000, 4 * 159), 4, 5120).is_err()
        );
        assert!(
            PrologueLaunch::new(row, DeviceSpan::new(0x8000, 4 * 5120 - 1), scales, 4, 5120)
                .is_err()
        );
        assert!(PrologueLaunch::new(row, codes, scales, 0, 5120).is_err());
    }

    #[test]
    fn the_weight_suffixes_are_anchored_to_the_real_plan_names() {
        let mut pointers = std::collections::BTreeMap::new();
        for layer in [0usize, 2, 39] {
            let prefix = format!("layers.{layer}");
            for (index, (_logical, suffix)) in WEIGHT_NAMES.iter().enumerate() {
                if index % 2 == 0 {
                    pointers.insert(format!("{prefix}.{suffix}"), (0x1000 + layer as u64, 0));
                }
            }
        }
        // A decoy that CONTAINS a suffix but does not end with it must not resolve: that is
        // the wrong-tensor-at-40-layers failure, and endswith is what prevents it.
        pointers.insert("layers.7.attn.q_norm.weight_backup".to_owned(), (0xdead, 0));
        let resolved = collect_layer_weights(&pointers);
        let q_norm = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "q_norm")
            .expect("q_norm");
        assert_eq!(resolved.get(&(2, q_norm)), Some(&0x1002));
        assert_eq!(resolved.get(&(39, q_norm)), Some(&0x1027));
        assert!(
            !resolved.contains_key(&(7, q_norm)),
            "the decoy must not bind"
        );
        // The layer index is an interior component of the real plan name, not a prefix:
        // layer_of walks to it rather than parsing the start of the string.
        assert_eq!(layer_of("layers.2.attn.q_norm.weight"), Some(2));
        assert_eq!(layer_of("layers.39.ffn_norm.weight"), Some(39));
        assert_eq!(layer_of("embed.weight"), None);
    }
    #[test]
    fn weights_resolve_once_into_typed_addresses_and_a_typo_names_the_candidates() {
        let mut pointers = std::collections::BTreeMap::new();
        pointers.insert(
            "model.layers.2.attn.q_norm.weight".to_owned(),
            (0x2000u64, 0u64),
        );
        pointers.insert("model.layers.2.attn.wq_a.weight".to_owned(), (0x4000, 0));
        pointers.insert("model.layers.3.attn.q_norm.weight".to_owned(), (0x6000, 0));
        pointers.insert(
            "model.layers.2.mlp.down_proj.weight".to_owned(),
            (0x8000, 0),
        );
        assert_eq!(
            resolve_weight(&pointers, "model.layers.2.attn.q_norm.weight"),
            Ok(0x2000)
        );
        let error = resolve_weight(&pointers, "model.layers.2.attn.q_norm.weightt")
            .expect_err("a typo must be refused");
        assert!(
            error.contains("no model.layers.2.attn.q_norm.weightt"),
            "{error}"
        );
        assert!(error.contains("weightt"), "{error}");
        assert_eq!(layer_of("model.layers.3.attn.q_norm.weight"), Some(3));
        assert_eq!(layer_of("attn.q_norm.weight"), None);
        let resolved = collect_layer_weights(&pointers);
        let index = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "q_norm")
            .expect("q_norm");
        assert_eq!(resolved.get(&(2, index)), Some(&0x2000));
        assert_eq!(resolved.get(&(3, index)), Some(&0x6000));
        assert_eq!(
            resolved.len(),
            3,
            "only WEIGHT_NAMES suffixes are collected"
        );

        // The two mHC seam tensors: real bind rows, so they resolve per layer like the rest.
        let mut hc = std::collections::BTreeMap::new();
        hc.insert("layers.5.hc_attn_fn".to_owned(), (0x1000u64, 0u64));
        hc.insert("layers.5.hc_attn_base".to_owned(), (0x1040, 0));
        hc.insert("layers.5.hc_attn_scale".to_owned(), (0x1050, 0));
        hc.insert("layers.5.hc_ffn_fn".to_owned(), (0x1080, 0));
        hc.insert("layers.5.hc_ffn_base".to_owned(), (0x10c0, 0));
        hc.insert("layers.5.hc_ffn_scale".to_owned(), (0x10d0, 0));
        let hc_resolved = collect_layer_weights(&hc);
        let attn_base = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "hc_attn_base")
            .expect("hc_attn_base");
        let ffn_fn = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "hc_ffn_fn")
            .expect("hc_ffn_fn");
        assert_eq!(hc_resolved.get(&(5, attn_base)), Some(&0x1040));
        assert_eq!(hc_resolved.get(&(5, ffn_fn)), Some(&0x1080));
        let attn_fn = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "hc_attn_fn")
            .expect("hc_attn_fn");
        let attn_scale = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "hc_attn_scale")
            .expect("hc_attn_scale");
        let ffn_base = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "hc_ffn_base")
            .expect("hc_ffn_base");
        let ffn_scale = WEIGHT_NAMES
            .iter()
            .position(|(name, _)| *name == "hc_ffn_scale")
            .expect("hc_ffn_scale");
        assert_eq!(hc_resolved.get(&(5, attn_fn)), Some(&0x1000));
        assert_eq!(hc_resolved.get(&(5, attn_scale)), Some(&0x1050));
        assert_eq!(hc_resolved.get(&(5, ffn_base)), Some(&0x10c0));
        assert_eq!(hc_resolved.get(&(5, ffn_scale)), Some(&0x10d0));
        assert_eq!(
            hc_resolved.len(),
            6,
            "the six mHC rows resolve on one layer"
        );
    }
    #[test]
    fn a_collective_count_mismatch_is_refused_loudly_on_one_rank_and_across_ranks() {
        assert!(verify_collective_count(80, 80).is_ok());
        let error = verify_collective_count(80, 78).expect_err("78 of 80 must be refused");
        assert!(error.contains("collective count mismatch"), "{error}");
        assert!(error.contains("must not be reported"), "{error}");
        assert_eq!(40 * COLLECTIVES_PER_LAYER, 80);

        assert!(verify_collective_counts_agree(&[80, 80, 80, 80]).is_ok());
        let error = verify_collective_counts_agree(&[80, 80, 78, 80])
            .expect_err("3 of 4 ranks must refuse");
        assert!(error.contains("collective counts disagree"), "{error}");
        assert!(error.contains("rank 2: 78"), "{error}");
        assert!(verify_collective_counts_agree(&[]).is_err());
    }

    #[test]
    fn the_kv_cache_layer_stride_is_aligned_and_bytes_is_layers_times_the_stride() {
        for (rows, values) in [(512usize, 128usize), (500, 128), (1, 64), (7, 3)] {
            let stride = DeviceKvCache::layer_bytes_for(rows, values);
            assert_eq!(stride % ALIGN, 0, "{rows} x {values} stride {stride}");
            assert!(stride >= rows * values * 2);
            for layers in [1usize, 4, 40] {
                assert_eq!(
                    DeviceKvCache::bytes(layers, rows, values),
                    layers * stride,
                    "{layers} layers of {rows} x {values}"
                );
            }
        }
        // And the instance agrees with the per-layer helper it is built on.
        let cache = DeviceKvCache::new(256, 40, 500, 128).expect("a shaped cache");
        assert_eq!(
            cache.layer_bytes(),
            DeviceKvCache::layer_bytes_for(500, 128)
        );
    }

    #[test]
    fn the_run_record_frontier_names_the_stages_that_ran_and_the_one_that_stopped_it() {
        let mut runs = StageRuns::default();
        assert!(runs.summary().contains("no stage ran"));
        runs.ran("mhc_entry");
        runs.ran("quantise_prologue");
        let line = runs.summary();
        assert!(line.contains("mhc_entry -> quantise_prologue"), "{line}");
        assert!(line.contains("(2 stage(s))"), "{line}");
        assert_eq!(runs.stages().len(), 2);
        runs.refused(
            "attention_projections",
            "kernel w8a8_prologue_e4m3 is not landed",
        );
        let line = runs.summary();
        assert!(line.contains("stopped at attention_projections"), "{line}");
        assert!(line.contains("w8a8_prologue_e4m3"), "{line}");
    }
    #[test]
    fn the_rope_table_count_is_two_tables_of_thirty_two_pairs_per_class_and_position() {
        assert_eq!(rope_table_bytes(1, 1), 2 * 128);
        assert_eq!(rope_table_bytes(2, 100), 2 * 2 * 100 * 128);
        assert_eq!(rope_table_bytes(2, 0), 0, "no positions, no tables");
    }
    #[test]
    fn every_stage_has_a_blocker_row_in_the_same_order() {
        let contracts: Vec<&str> = STAGE_CONTRACTS.iter().map(|entry| entry.stage).collect();
        let blockers: Vec<&str> = STAGE_BLOCKERS.iter().map(|(stage, _)| *stage).collect();
        assert_eq!(contracts, blockers);
        assert!(
            STAGE_BLOCKERS
                .iter()
                .all(|(_, blocker)| !blocker.is_empty()),
            "every stage states what still has to move on device, even when the answer is none"
        );
    }

    #[test]
    fn the_refusal_names_the_blocking_device_step_when_there_is_one() {
        let refusal = not_landed_with_blocker(
            "query_path",
            "rms_norm_qk_gemv",
            "(...)",
            "the host q_norm must move on device",
        );
        assert!(refusal.contains("query_path"), "{refusal}");
        assert!(
            refusal.contains("the host q_norm must move on device"),
            "{refusal}"
        );
        let plain = not_landed_with_blocker("all_reduce", "device_all_reduce_bf16", "(...)", "");
        assert!(!plain.contains("also still needs"), "{plain}");
    }

    #[test]
    fn the_arm_decision_matrix_is_the_one_place_the_switches_meet() {
        // The matrix the bootstrap gate depends on: evidence recording forces the host arm even when
        // the device switch is on, so a chain that gated on the device switch alone would open a
        // communicator for the arm that must never need libnccl.
        let decide = |device, evidence| {
            decide_layer_arm(device, evidence).unwrap_or_else(|error| panic!("{error}"))
        };
        for device in [
            Some("0"),
            Some("false"),
            Some("no"),
            Some("host"),
            Some("HOST"),
        ] {
            for evidence in [None, Some("0"), Some("1")] {
                let (arm, recorded) = decide(device, evidence);
                assert!(!arm.is_device(), "{device:?}/{evidence:?} must stay host");
                assert_eq!(recorded, evidence == Some("1"));
            }
        }
        for device in [None, Some(""), Some("1"), Some("device")] {
            for evidence in [None, Some("0")] {
                let (arm, _) = decide(device, evidence);
                assert!(
                    arm.is_device(),
                    "{device:?} with evidence off is the device arm (the S2 default)"
                );
            }
        }
        for device in [None, Some("1"), Some("true"), Some("YES")] {
            for evidence in [Some("1"), Some("true"), Some("yes")] {
                let (arm, recorded) = decide(device, evidence);
                assert!(!arm.is_device(), "evidence recording forces the host arm");
                assert!(recorded);
            }
        }
        // A padded or unknown spelling refuses instead of picking either arm.
        for device in [" 1", "on", "dev", "2"] {
            let error = decide_layer_arm(Some(device), None)
                .expect_err("an unknown spelling must refuse, not pick an arm");
            assert!(error.contains(LAYER_DEVICE_ENV), "{error}");
            assert!(error.contains("names neither arm"), "{error}");
        }
    }

    #[test]
    fn the_device_arm_is_the_default_and_the_host_arm_stays_selectable() {
        // Every spelling the parser accepts, and the ones it refuses, without touching the process
        // environment (the suite runs tests in parallel).
        assert_eq!(
            parse_device_switch(None),
            Ok(true),
            "unset is the device arm"
        );
        assert_eq!(parse_device_switch(Some("")), Ok(true), "empty is unset");
        for value in ["1", "device", "DEVICE", "true", "TRUE", "yes", "Yes"] {
            assert_eq!(parse_device_switch(Some(value)), Ok(true), "{value}");
        }
        for value in ["0", "host", "Host", "false", "no", "NO"] {
            assert_eq!(parse_device_switch(Some(value)), Ok(false), "{value}");
        }
        for value in [" 1", "1 ", "on", "off", "devices", "-1"] {
            assert!(parse_device_switch(Some(value)).is_err(), "{value:?}");
        }
    }

    #[test]
    fn every_stage_of_the_run_has_a_contract_and_the_order_is_the_loop_order() {
        let stages: Vec<&str> = STAGE_CONTRACTS.iter().map(|entry| entry.stage).collect();
        assert_eq!(
            stages,
            vec![
                "engram_write",
                "mhc_entry",
                "quantise_prologue",
                "attention_projections",
                "query_path",
                "kv_path",
                "reference_seam",
                "ref_select",
                "stage_attention",
                "attention_body",
                "attention_output",
                "all_reduce",
                "attention_residual",
                "ffn_seams",
                "moe_router",
                "moe_routed",
                "all_reduce_moe",
                "moe_shared",
                "all_reduce_rdma_arm",
                "ffn_epilogue",
            ]
        );
        for entry in STAGE_CONTRACTS {
            assert!(!entry.kernel.is_empty(), "{}", entry.stage);
            assert!(entry.signature.starts_with('('), "{}", entry.stage);
        }
    }

    #[test]
    fn a_missing_kernel_refusal_names_the_stage_and_the_kernel() {
        let refusal = not_landed("mhc_entry", "mhc_entry_bf16", "(...)");
        assert!(refusal.contains("mhc_entry"), "{refusal}");
        assert!(refusal.contains("mhc_entry_bf16"), "{refusal}");
        assert!(refusal.contains("contract"), "{refusal}");
    }

    #[test]
    fn the_span_offsets_are_contiguous_and_the_env_switch_parses_its_spellings() {
        let span = DeviceSpan::new(0x40, 1024);
        assert_eq!(span.offset(0x40).device, 0x80);
        assert_eq!(span.offset(0x40).len, 960);
        // The switch itself is read from the process environment, so only that whatever value the
        // suite runs under parses is asserted here; the loop's arm is exercised by the forward
        // loop's tests.
        assert!(parse_device_switch(std::env::var(LAYER_DEVICE_ENV).ok().as_deref()).is_ok());
    }

    #[test]
    fn the_prologue_span_check_is_the_row_dtypes_width() {
        let geometry = geometry();
        let buffers = LayerBuffers::carve(0, &geometry);
        // A bf16-width row (the F1 shape) must be refused by the f32 constructor: 5,120 values
        // need 20,480 bytes, not the 10,240 a bf16 span holds.
        let half = DeviceSpan::new(0, geometry.dim * 2);
        let error = PrologueLaunch::new_f32(
            half,
            buffers.workspace.act_codes,
            buffers.workspace.act_scales,
            1,
            geometry.dim as u32,
        )
        .expect_err("a bf16-width row must not authorise the f32 prologue");
        assert!(error.contains("f32 needs 20480"), "{error}");
        // The same span is exactly right for the bf16 constructor.
        PrologueLaunch::new(
            half,
            buffers.workspace.act_codes,
            buffers.workspace.act_scales,
            1,
            geometry.dim as u32,
        )
        .expect("a bf16 row takes the bf16 constructor");
        // And the f32 row the f32 kernel reads passes the f32 constructor.
        PrologueLaunch::new_f32(
            buffers.residual.normed,
            buffers.workspace.act_codes,
            buffers.workspace.act_scales,
            1,
            geometry.dim as u32,
        )
        .expect("the normed f32 row takes the f32 constructor");
    }

    #[test]
    fn every_carved_span_matches_its_contract_width() {
        let geometry = geometry();
        let buffers = LayerBuffers::carve(0, &geometry);
        buffers
            .assert_contract_widths(&geometry)
            .expect("the released geometry carves contract widths");
        // A span carved one dtype short is refused by name: set q_normed to the width a bf16-only
        // reading would give an f32 row (half), which is the shape of the query_path bug.
        let mut broken = buffers;
        broken.workspace.q_normed = DeviceSpan::new(
            broken.workspace.q_normed.device,
            broken.workspace.q_normed.len / 2,
        );
        let error = broken
            .assert_contract_widths(&geometry)
            .expect_err("a half-width span must be refused");
        assert!(error.contains("workspace.q_normed"), "{error}");
        assert!(error.contains("1280"), "{error}");
    }

    #[test]
    fn the_dspark_taps_are_three_fixed_f32_rows_and_only_layers_37_to_39_have_one() {
        let geometry = geometry();
        let buffers = LayerBuffers::carve(0x1_0000, &geometry);
        let taps = buffers.workspace.dspark_aux_taps;
        assert_eq!(taps.len, 3 * geometry.dim * 4);
        let row = geometry.dim * 4;
        for (index, layer) in DSPARK_AUX_LAYERS.into_iter().enumerate() {
            let span = buffers
                .dspark_aux_tap_span(layer, geometry.dim)
                .unwrap_or_else(|| panic!("layer {layer} is a DSpark tap"));
            assert_eq!(span.device, taps.device + (index * row) as u64);
            assert_eq!(span.len, row);
        }
        for layer in [0, 36, 40] {
            assert!(buffers.dspark_aux_tap_span(layer, geometry.dim).is_none());
        }
        // A row wider than the carve does not fit the last tap: refused, never a span past the end.
        assert!(buffers.dspark_aux_tap_span(39, geometry.dim + 1).is_none());
    }

    #[test]
    fn a_layer_that_refuses_at_kv_path_reports_the_stages_that_ran() {
        // A stub chain that reaches kv_path and refuses there: the frontier must name the five
        // stages that ran. `from_refusal` would say "no stage ran" for the same message, which is
        // the lying artifact the probe row must never carry.
        let ran = [
            "engram_write",
            "mhc_entry",
            "quantise_prologue",
            "attention_projections",
            "query_path",
        ];
        let mut runs = StageRuns::default();
        for stage in ran {
            runs.record(stage, Ok::<(), String>(())).unwrap();
        }
        let error = runs
            .record(
                "kv_path",
                Err::<(), String>("layer device stage kv_path: the stub refuses here".to_owned()),
            )
            .unwrap_err();
        assert_eq!(error, "layer device stage kv_path: the stub refuses here");
        assert_eq!(runs.stages(), &ran);
        assert_eq!(
            runs.summary(),
            "device-only through engram_write -> mhc_entry -> quantise_prologue -> attention_projections -> query_path (5 stage(s)); stopped at kv_path: layer device stage kv_path: the stub refuses here"
        );
    }
}
