//! The host arithmetic of one layer-0 pass, decomposed into the chain's stage order.
//!
//! This is the M3-c host half, not a device run: it composes the existing reference functions
//! ([crate::forward::mhc], [crate::forward::attention] and [crate::forward::moe]) in the order
//! the reference block loop runs them, and records one [StageEvent] at every stage the
//! chain's layer-0 graph names. The trace's boundary values are compared against the reference
//! engine's own layer-0 [crate::forward::boundary] row; the reference is the same arithmetic, so
//! the comparison is bit-exact and its class is Exact, not a tolerance bar.
//!
//! Only position 0 is supported in this pass. Layer 0 is SWA-only (compress ratio 0, encoder
//! layer), so the compressor, indexer, HSI and expanded-view stages do not run at this layer; the
//! active stages are [LAYER_ZERO_HOST_STAGES]. The engram stage also does not run because Engram is
//! present only at layers 1 and 14 ([crate::config::ModelConfig::engram_rows]).
//!
//! The one hybrid seam this module anticipates is the mHC entry handoff: [LayerZeroEntry] is the
//! value a device MhcEntry launch can hand back (the normalised block input and its three
//! coefficient sets), so a run can continue on the host with the device's entry values. The default
//! path computes that entry with the reference arithmetic, which keeps the no-GPU tests exact.

use crate::config::ModelConfig;
use crate::forward::attention::{self, AttnState};
use crate::forward::mhc::{self, Mixes};
use crate::forward::moe::{self, MoeScratch};
use crate::linalg::{self, EngineError, REFERENCE_TENSOR_LIMIT};
use crate::weights::WeightSource;

/// The active layer-0 stages, in the chain's order.
///
/// The layer body's Engram, compressor, index-key, indexer, HSI, Top-512 and expanded-view stages
/// do not run at layer 0: Engram is layers 1 and 14, and layer 0 has compress ratio 0
/// (SWA-only). The list is the layer-0 subset of
/// [crate::forward::InferenceEngine]'s reference block in the device chain's stage order.
pub const LAYER_ZERO_HOST_STAGES: [&str; 17] = [
    "dispatch",
    "mhc_entry",
    "attn_norm",
    "attention_projections",
    "attn_rope",
    "window_kv",
    "attention",
    "attention_output_projection",
    "mhc_post",
    "moe_router",
    "moe_routing_weight",
    "moe_gate_up",
    "moe_swiglu_quant",
    "moe_down_combine",
    "shared_expert_gate",
    "shared_expert_swiglu_quant",
    "ffn_epilogue",
];

/// The exact quantities both layer-0 comparators must report, in order.
///
/// Pinned as a value rather than left implicit in the compare calls: deleting one comparison would
/// otherwise silently shrink the c2 evidence set.
pub const LAYER_ZERO_COMPARISON_QUANTITIES: [&str; 12] = [
    "pre_attention_hidden",
    "attention_out",
    "post_attention_hidden",
    "pre_ffn_hidden",
    "ffn_out",
    "residual_out",
    "hc_attn_pre",
    "hc_attn_post",
    "hc_attn_comb",
    "hc_ffn_pre",
    "hc_ffn_post",
    "hc_ffn_comb",
];

/// One host-stage event: the stage name and the logical tensor bytes the host read and wrote.
///
/// The byte counts are the actual slice lengths of the reference tensors at that boundary (four
/// bytes per f32), not estimates copied from the plan. A later seam ledger maps them to the chain's
/// stable seam names; the note says what produced the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageEvent {
    /// The chain stage name, for example `"attn_norm"`.
    pub stage: &'static str,
    /// Bytes the host read for this stage.
    pub host_bytes_in: u64,
    /// Bytes the host wrote for this stage.
    pub host_bytes_out: u64,
    /// One sentence naming the arithmetic the stage performed.
    pub note: &'static str,
}

/// Where a [LayerZeroEntry] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerZeroEntrySource {
    /// The host reference mHC arithmetic computed the entry.
    Host,
    /// A staged device MhcEntry launch and readback produced the entry.
    Device,
}

/// A MhcEntry handoff.
///
/// `pre_attention_hidden` is the normalised attention block input (the kernel's `xhat_normed`
/// capture), and `pre`, `post`, `comb` are the 24 coefficient values in the host
/// [Mixes] orientation. A device path builds this from its readback and passes it to
/// [run_layer_zero]; the host path computes the same values itself. `source` is carried explicitly
/// so no caller has to infer device provenance from an `Option` being `Some`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroEntry {
    /// The normalised attention block input, one [ModelConfig::dim]-long row.
    pub pre_attention_hidden: Vec<f32>,
    /// The attention input mix, `hc_mult` values.
    pub pre: Vec<f32>,
    /// The attention output mix, `hc_mult` values.
    pub post: Vec<f32>,
    /// The attention residual mixer, `hc_mult * hc_mult` values, row `j` is source copy `j`
    /// and column `k` the destination copy.
    pub comb: Vec<f32>,
    /// Who produced the entry: host reference arithmetic or a staged device launch.
    pub source: LayerZeroEntrySource,
}

/// The layer-0 host trace: the stage events plus the boundary values the comparison uses.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroTrace {
    /// The prompt token this pass embedded.
    pub token: u32,
    /// The position, always 0 for this pass.
    pub position: usize,
    /// The embedding row, `dim` values.
    pub embedding: Vec<f32>,
    /// The attention input before its RMS norm, `dim` values.
    pub attn_in: Vec<f32>,
    /// The attention input after its RMS norm, `dim` values.
    pub pre_attention_hidden: Vec<f32>,
    /// The attention input mix.
    pub attn_mixes: Mixes,
    /// The attention output before the mHC post-mix, `dim` values.
    pub attention_out: Vec<f32>,
    /// The residual stream after the attention post-mix, `hc_mult * dim` values.
    pub post_attention_hidden: Vec<f32>,
    /// The FFN input before its RMS norm, `dim` values.
    pub ffn_in: Vec<f32>,
    /// The FFN input after its RMS norm, `dim` values.
    pub pre_ffn_hidden: Vec<f32>,
    /// The FFN input mix.
    pub ffn_mixes: Mixes,
    /// The FFN output before the mHC post-mix, `dim` values.
    pub ffn_out: Vec<f32>,
    /// The residual stream after the FFN post-mix, `hc_mult * dim` values.
    pub residual_out: Vec<f32>,
    /// One event per active stage, in [LAYER_ZERO_HOST_STAGES] order.
    pub stages: Vec<StageEvent>,
}

fn bytes(values: usize) -> u64 {
    (values as u64).saturating_mul(4)
}

fn stage(
    stages: &mut Vec<StageEvent>,
    name: &'static str,
    host_bytes_in: u64,
    host_bytes_out: u64,
    note: &'static str,
) {
    stages.push(StageEvent {
        stage: name,
        host_bytes_in,
        host_bytes_out,
        note,
    });
}

fn read(source: &dyn WeightSource, name: &str, limit: usize) -> Result<Vec<f32>, EngineError> {
    linalg::decode_by_name(source, name, limit)
}

/// Refuses a layer whose schedule is not SWA-only.
///
/// The stage list and the one-layer schedule are the ratio-0 layer-0 decomposition. A released
/// configuration could in principle put a compressor ratio on layer 0, and silently running the
/// SWA-only arithmetic for it would be a wrong layer, not a slower one.
fn require_swa_only_layer_zero(config: &ModelConfig) -> Result<(), EngineError> {
    let ratio = config.compress_ratios.first().copied().unwrap_or(0);
    if ratio != 0 {
        return Err(EngineError::Shape {
            what: "layer-zero host pass".to_owned(),
            detail: format!(
                "layer 0 has compress ratio {ratio}; this pass only supports SWA-only ratio 0"
            ),
        });
    }
    Ok(())
}

/// Runs one layer-0 pass on the host and records every active stage.
///
/// Only position 0 is supported, because the pass has no prefix history and no cache to read. The
/// default entry is computed by the reference mHC arithmetic; when `entry` is supplied, the host
/// continues from that handoff instead. The returned [LayerZeroTrace] has the reference's
/// boundary values for the same token, so a caller can compare bit-for-bit.
///
/// # Errors
/// [EngineError] when the embedding token is outside the table, a weight is missing or over
/// `limit`, or the configuration has a shape the reference functions refuse.
pub fn run_layer_zero(
    config: &ModelConfig,
    source: &dyn WeightSource,
    token: u32,
    limit: usize,
    entry: Option<LayerZeroEntry>,
) -> Result<LayerZeroTrace, EngineError> {
    let dim = config.dim;
    if dim == 0 {
        return Err(EngineError::Shape {
            what: "layer-zero host pass".to_owned(),
            detail: "dim must be positive".to_owned(),
        });
    }
    require_swa_only_layer_zero(config)?;
    let embed = read(source, "embed.weight", limit)?;
    if token as usize >= config.vocab_size {
        return Err(EngineError::Shape {
            what: "embed.weight".to_owned(),
            detail: format!(
                "token {token} is outside the {} row table",
                config.vocab_size
            ),
        });
    }
    let row = token as usize * dim;
    let Some(embedding) = embed.get(row..row + dim) else {
        return Err(EngineError::Shape {
            what: "embed.weight".to_owned(),
            detail: format!(
                "token {token} row {}..{} is outside the decoded table",
                row,
                row + dim
            ),
        });
    };
    run_layer_zero_with_embedding(config, source, token, embedding, limit, entry)
}

/// Runs one layer-0 pass with an embedding row supplied by the caller.
///
/// This is the no-full-table entry point: a device staging path already owns the token's one
/// embedding row, so it must not read the whole `embed.weight` table (the released table is far
/// larger than a host budget). `embedding` must be exactly [ModelConfig::dim] f32 values; the
/// rest of the pass is [run_layer_zero].
///
/// # Errors
/// [EngineError] when the embedding length is wrong or the reference arithmetic refuses a weight
/// or a shape.
pub fn run_layer_zero_with_embedding(
    config: &ModelConfig,
    source: &dyn WeightSource,
    token: u32,
    embedding: &[f32],
    limit: usize,
    entry: Option<LayerZeroEntry>,
) -> Result<LayerZeroTrace, EngineError> {
    let dim = config.dim;
    let hc = config.hc_mult;
    let width = config.hc_width();
    require_swa_only_layer_zero(config)?;
    if dim == 0 || hc == 0 {
        return Err(EngineError::Shape {
            what: "layer-zero host pass".to_owned(),
            detail: "dim and hc_mult must both be positive".to_owned(),
        });
    }
    if embedding.len() != dim {
        return Err(EngineError::Shape {
            what: "embedding row".to_owned(),
            detail: format!("expected {dim} values, got {}", embedding.len()),
        });
    }
    let embedding = embedding.to_vec();
    let prefix = "layers.0";
    let mut hidden = vec![0.0f32; width];
    for copy in 0..hc {
        hidden[copy * dim..(copy + 1) * dim].copy_from_slice(&embedding);
    }
    let mut pre_mix = vec![0.0f32; hc];
    pre_mix[0] = 1.0;

    let mut stages = Vec::with_capacity(LAYER_ZERO_HOST_STAGES.len());
    stage(
        &mut stages,
        "dispatch",
        bytes(dim),
        bytes(width),
        "scatter the embedding row into the hc residual copies",
    );

    let (attn_in, pre_attention_hidden, attn_mixes, entry_source) = if let Some(entry) = entry {
        let source = entry.source;
        let attn_mixes = Mixes {
            pre: entry.pre,
            post: entry.post,
            comb: entry.comb,
        };
        (
            entry.pre_attention_hidden.clone(),
            entry.pre_attention_hidden,
            attn_mixes,
            Some(source),
        )
    } else {
        let fn_w = read(source, &format!("{prefix}.hc_attn_fn"), limit)?;
        let scale = read(source, &format!("{prefix}.hc_attn_scale"), limit)?;
        let base = read(source, &format!("{prefix}.hc_attn_base"), limit)?;
        let mut mixes = Mixes::for_config(config);
        mhc::hc_mixes(config, &hidden, &fn_w, &scale, &base, &mut mixes)?;
        let mut attn_in = vec![0.0f32; dim];
        mhc::hc_pre(config, &pre_mix, &hidden, &mut attn_in)?;
        stage(
            &mut stages,
            "mhc_entry",
            bytes(hidden.len()),
            bytes(attn_in.len() + mixes.pre.len() + mixes.post.len() + mixes.comb.len()),
            "hc_mixes + hc_pre over the hc residual copies",
        );
        let norm = read(source, &format!("{prefix}.attn_norm.weight"), limit)?;
        let mut normed = attn_in.clone();
        linalg::rms_norm(&mut normed, &norm, dim, config.norm_eps as f32)?;
        stage(
            &mut stages,
            "attn_norm",
            bytes(attn_in.len() * 2),
            bytes(normed.len()),
            "RMSNorm of the attention block input with the staged attn_norm weight",
        );
        (attn_in, normed, mixes, None)
    };
    if entry_source.is_some() {
        let mixes = &attn_mixes;
        stage(
            &mut stages,
            "mhc_entry",
            bytes(
                pre_attention_hidden.len() + mixes.pre.len() + mixes.post.len() + mixes.comb.len(),
            ),
            bytes(pre_attention_hidden.len()),
            "caller-supplied MhcEntry handoff; provenance is carried by LayerZeroEntry.source",
        );
        stage(
            &mut stages,
            "attn_norm",
            bytes(pre_attention_hidden.len()),
            0,
            "the caller-supplied entry already contains the normalised attention input; the host does not renormalise",
        );
    }

    let attention_width =
        config
            .n_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::Shape {
                what: "layer-zero host pass".to_owned(),
                detail: "n_heads * head_dim overflows usize".to_owned(),
            })?;
    // The reference's scratch buffer is the wider of the head geometry and one dense row; a
    // head geometry narrower than dim (the tiny fixture) still needs a full dim-long output slice.
    let wide = attention_width.max(dim);
    stage(
        &mut stages,
        "attention_projections",
        bytes(pre_attention_hidden.len()),
        bytes(wide),
        "q/kv projections over the normalised attention input",
    );
    stage(
        &mut stages,
        "attn_rope",
        bytes(wide),
        bytes(wide),
        "RoPE on the projected q/kv rows",
    );
    stage(
        &mut stages,
        "window_kv",
        bytes(wide),
        bytes(wide),
        "write and read the layer-0 sliding-window KV ring",
    );
    stage(
        &mut stages,
        "attention",
        bytes(pre_attention_hidden.len()),
        bytes(wide),
        "flash-MLA arithmetic over the window KV",
    );
    let mut state = AttnState::for_config(config);
    let mut attention_out = vec![0.0f32; wide];
    attention::attention_forward(
        config,
        0,
        prefix,
        &pre_attention_hidden,
        0,
        &mut state,
        source,
        &mut attention_out,
        limit,
    )?;
    attention_out.truncate(dim);
    stage(
        &mut stages,
        "attention_output_projection",
        bytes(wide),
        bytes(dim),
        "wo_a + wo_b projection of the concatenated head rows",
    );

    let residual = hidden.clone();
    let mut post_attention_hidden = vec![0.0f32; width];
    mhc::hc_post(
        config,
        &attention_out,
        &residual,
        &attn_mixes.post,
        &attn_mixes.comb,
        &mut post_attention_hidden,
    )?;
    stage(
        &mut stages,
        "mhc_post",
        bytes(attention_out.len() + residual.len() + attn_mixes.post.len() + attn_mixes.comb.len()),
        bytes(post_attention_hidden.len()),
        "hc_post: attention output mixed back onto the hc residual",
    );

    let fn_w = read(source, &format!("{prefix}.hc_ffn_fn"), limit)?;
    let scale = read(source, &format!("{prefix}.hc_ffn_scale"), limit)?;
    let base = read(source, &format!("{prefix}.hc_ffn_base"), limit)?;
    let mut ffn_mixes = Mixes::for_config(config);
    mhc::hc_mixes(
        config,
        &post_attention_hidden,
        &fn_w,
        &scale,
        &base,
        &mut ffn_mixes,
    )?;
    let mut ffn_in = vec![0.0f32; dim];
    mhc::hc_pre(config, &attn_mixes.pre, &post_attention_hidden, &mut ffn_in)?;
    let ffn_norm = read(source, &format!("{prefix}.ffn_norm.weight"), limit)?;
    let mut pre_ffn_hidden = ffn_in.clone();
    linalg::rms_norm(&mut pre_ffn_hidden, &ffn_norm, dim, config.norm_eps as f32)?;

    let topk = config.n_activated_experts;
    stage(
        &mut stages,
        "moe_router",
        bytes(pre_ffn_hidden.len()),
        bytes(topk * 8),
        "route the normalised FFN input through the reference router",
    );
    stage(
        &mut stages,
        "moe_routing_weight",
        bytes(topk * 8),
        bytes(topk * 12),
        "apply the selected experts' routing weights",
    );
    stage(
        &mut stages,
        "moe_gate_up",
        bytes(pre_ffn_hidden.len()),
        bytes(2 * config.moe_inter_dim),
        "routed gate/up branch for the selected experts",
    );
    stage(
        &mut stages,
        "moe_swiglu_quant",
        bytes(2 * config.moe_inter_dim),
        bytes(config.moe_inter_dim),
        "SwiGLU and activation quantisation of the routed branch",
    );
    stage(
        &mut stages,
        "moe_down_combine",
        bytes(config.moe_inter_dim * topk),
        bytes(dim),
        "routed down projection and combine",
    );
    let shared_inter = config
        .moe_inter_dim
        .saturating_mul(config.n_shared_experts.max(1));
    stage(
        &mut stages,
        "shared_expert_gate",
        bytes(pre_ffn_hidden.len()),
        bytes(2 * shared_inter),
        "dense shared-expert gate/up GEMM",
    );
    stage(
        &mut stages,
        "shared_expert_swiglu_quant",
        bytes(2 * shared_inter),
        bytes(shared_inter),
        "shared-expert SwiGLU and activation quantisation",
    );
    let mut scratch = MoeScratch::for_config(config);
    let mut ffn_out = vec![0.0f32; dim];
    moe::moe_forward(
        config,
        prefix,
        &pre_ffn_hidden,
        source,
        limit,
        &mut scratch,
        &mut ffn_out,
    )?;

    let residual = post_attention_hidden.clone();
    let mut residual_out = vec![0.0f32; width];
    mhc::hc_post(
        config,
        &ffn_out,
        &residual,
        &ffn_mixes.post,
        &ffn_mixes.comb,
        &mut residual_out,
    )?;
    stage(
        &mut stages,
        "ffn_epilogue",
        bytes(ffn_out.len() + residual.len() + ffn_mixes.post.len() + ffn_mixes.comb.len()),
        bytes(residual_out.len()),
        "hc_post of the FFN output and the FFN epilogue add",
    );

    debug_assert_eq!(
        stages.iter().map(|event| event.stage).collect::<Vec<_>>(),
        LAYER_ZERO_HOST_STAGES.to_vec(),
        "the host pass must record the active layer-0 stage order"
    );

    Ok(LayerZeroTrace {
        token,
        position: 0,
        embedding,
        attn_in,
        pre_attention_hidden,
        attn_mixes,
        attention_out,
        post_attention_hidden,
        ffn_in,
        pre_ffn_hidden,
        ffn_mixes,
        ffn_out,
        residual_out,
        stages,
    })
}

/// The default reference limit, re-exported so a caller does not have to import linalg.
#[must_use]
pub const fn default_limit() -> usize {
    REFERENCE_TENSOR_LIMIT
}

/// The measured m11 b7 bar on the 24 mixed mHC coefficients (max absolute delta).
///
/// docs/build/phase3-mhc-entry.md records `hc_coeffs B/A/C` within the declared `1e-4` bar at
/// b7; the kernel's fixed-order tile reduction and the CPU twin's accumulation order are the
/// reason this is a bar rather than bit identity.
pub const M11_HC_COEFF_BAR: f32 = 1e-4;

/// The measured m11 b7 bar on `1/rms(X_hat)` (max absolute delta).
///
/// docs/build/phase3-mhc-entry.md records a maximum of `2.746582e-4` at b7 against the
/// declared `3e-4` bar. The host RMSNorm accumulates the row's sum in a different order from the
/// kernel's tile reduction, so the scalar carries this bar; the `xhat_normed` capture is exact
/// when built with the device's own scalar, as m11 measured.
pub const M11_INV_RMS_BAR: f32 = 3e-4;

/// The five quantities a device MhcEntry readback must reproduce, in order.
///
/// These are the values m11 measured at b7: the normalised block input, the three released
/// coefficient groups and the `1/rms` scalar. The host continuation after the entry is host
/// arithmetic, not a device kernel result, so it is not part of this entry comparison.
pub const LAYER_ZERO_ENTRY_QUANTITIES: [&str; 5] = [
    "pre_attention_hidden",
    "hc_attn_pre",
    "hc_attn_post",
    "hc_attn_comb",
    "inv_rms",
];

/// The first bit-level difference in one compared quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantityDifference {
    /// The element index of the first differing value.
    pub index: usize,
    /// The expected value's raw f32 bits.
    pub expected_bits: u32,
    /// The actual value's raw f32 bits.
    pub actual_bits: u32,
}

/// The released mHC kernel's stream count (the pinned `n = 4` of `mega_mhc_sm121`).
const KERNEL_HC_MULT: usize = 4;

/// The kernel's step-1 update, in its exact FMA order.
///
/// `x_prev` is the four-copy BF16 residual stream widened to f32; `b_prev`, `c_prev` and
/// `y_prev` are the kernel-order boundary coefficients and the previous attention block output.
/// The per-stream width is taken from `y_prev` (the kernel's `dim`), never from `x_prev`, because
/// the pinned kernel has one `Y` value per element and four `X` streams over the same element.
/// The pinned `mega_mhc_sm121` kernel is compiled with FMA contraction: the `C Y` term is one
/// plain multiply and each B term is a fused multiply-add in stream order. A mul-then-add twin is
/// not the device's program and can differ in the last bit.
#[must_use]
pub fn kernel_mhc_update(
    x_prev: &[f32],
    b_prev: &[f32],
    c_prev: &[f32],
    y_prev: &[f32],
) -> Vec<f32> {
    let hc = KERNEL_HC_MULT;
    let width = y_prev.len();
    assert_eq!(
        x_prev.len(),
        hc * width,
        "X must hold {hc} streams of {width} values"
    );
    assert_eq!(b_prev.len(), hc * hc, "B must be {hc} x {hc}");
    assert_eq!(c_prev.len(), hc, "C must hold {hc} values");
    let mut updated = vec![0.0f32; hc * width];
    for index in 0..width {
        let y = y_prev[index];
        for stream in 0..hc {
            let mut acc = c_prev[stream] * y;
            for source in 0..hc {
                acc = b_prev[stream * hc + source].mul_add(x_prev[source * width + index], acc);
            }
            updated[stream * width + index] = acc;
        }
    }
    updated
}

/// The kernel's actual step-1 instruction order, for a non-identity B/C row: the first B product is
/// the one plain multiplication, the C Y term is fused into it, and the remaining B terms are fused
/// in stream order. [kernel_mhc_update] is the source-level order (C Y first, then every B term
/// fused) and matches the kernel only on the identity seed; the c3b deferred entry and post calls
/// consume real previous-layer B/C rows, so their host twin must use this order.
#[must_use]
pub fn kernel_mhc_update_fused(
    x_prev: &[f32],
    b_prev: &[f32],
    c_prev: &[f32],
    y_prev: &[f32],
) -> Vec<f32> {
    let hc = KERNEL_HC_MULT;
    let width = y_prev.len();
    assert_eq!(
        x_prev.len(),
        hc * width,
        "X must hold {hc} streams of {width} values"
    );
    assert_eq!(b_prev.len(), hc * hc, "B must be {hc} x {hc}");
    assert_eq!(c_prev.len(), hc, "C must hold {hc} values");
    let mut updated = vec![0.0f32; hc * width];
    for stream in 0..hc {
        for column in 0..width {
            let mut value = b_prev[stream * hc] * x_prev[column];
            value = c_prev[stream].mul_add(y_prev[column], value);
            for source in 1..hc {
                value =
                    b_prev[stream * hc + source].mul_add(x_prev[source * width + column], value);
            }
            updated[stream * width + column] = value;
        }
    }
    updated
}

/// The kernel's step-2 `X_hat = A .* X_l`, in the kernel's FMA order.
///
/// `x_l` is the unrounded f32 update that the kernel keeps in registers (the BF16 write of X_l
/// is a separate output), and `a_prev` is the layer-boundary A vector. The per-stream width is
/// derived from `x_l`, so a caller cannot pass the four-stream length where the kernel's `dim`
/// belongs.
#[must_use]
pub fn kernel_mhc_xhat(x_l: &[f32], a_prev: &[f32]) -> Vec<f32> {
    let hc = KERNEL_HC_MULT;
    assert!(
        !x_l.is_empty() && x_l.len().is_multiple_of(hc),
        "X_l must hold {hc} equal streams"
    );
    assert_eq!(a_prev.len(), hc, "A must hold {hc} values");
    let width = x_l.len() / hc;
    let mut xhat = vec![0.0f32; width];
    for (index, slot) in xhat.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for stream in 0..hc {
            value = a_prev[stream].mul_add(x_l[stream * width + index], value);
        }
        *slot = value;
    }
    xhat
}

/// The kernel's f32 block-input capture, in its multiplication order.
///
/// `xhat * rms_weight * inverse_rms` is left-associative in the kernel, so a twin must not
/// reorder the two multiplies.
#[must_use]
pub fn kernel_mhc_capture(xhat: &[f32], rms_weight: &[f32], inverse_rms: f32) -> Vec<f32> {
    assert_eq!(xhat.len(), rms_weight.len(), "capture length mismatch");
    xhat.iter()
        .zip(rms_weight)
        .map(|(value, weight)| value * weight * inverse_rms)
        .collect()
}

/// Rounds an f32 to BF16 with round-to-nearest-even, as `__float2bfloat16_rn` does.
#[must_use]
pub fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// The little-endian BF16 bytes of an f32 row, as the kernel writes X_l.
#[must_use]
pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        let bits = (bf16_round(*value).to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// Compares two layer-0 traces quantity by quantity.
///
/// This is the host-reference check: the host pass composes the same reference functions over the
/// same bytes as the reference engine, so the class is Exact only when every compared value is
/// bit-identical. It is not the device-readback comparison; [compare_entry] applies the m11 b7
/// classes to the one device kernel this pass uses.
///
/// # Errors
/// [EngineError::Shape] when the two traces have different lengths for a quantity.
pub fn compare_traces(
    left: &LayerZeroTrace,
    right: &LayerZeroTrace,
) -> Result<LayerZeroComparison, EngineError> {
    use crate::forward::boundary::{
        ATTENTION_OUT, FFN_OUT, HC_ATTN_COMB, HC_ATTN_POST, HC_ATTN_PRE, HC_FFN_COMB, HC_FFN_POST,
        HC_FFN_PRE, POST_ATTENTION_HIDDEN, PRE_ATTENTION_HIDDEN, PRE_FFN_HIDDEN, RESIDUAL_OUT,
    };
    let mut quantities = Vec::with_capacity(LAYER_ZERO_COMPARISON_QUANTITIES.len());
    quantities.push(compare_quantity(
        PRE_ATTENTION_HIDDEN,
        &left.pre_attention_hidden,
        &right.pre_attention_hidden,
        None,
    )?);
    quantities.push(compare_quantity(
        ATTENTION_OUT,
        &left.attention_out,
        &right.attention_out,
        None,
    )?);
    quantities.push(compare_quantity(
        POST_ATTENTION_HIDDEN,
        &left.post_attention_hidden,
        &right.post_attention_hidden,
        None,
    )?);
    quantities.push(compare_quantity(
        PRE_FFN_HIDDEN,
        &left.pre_ffn_hidden,
        &right.pre_ffn_hidden,
        None,
    )?);
    quantities.push(compare_quantity(
        FFN_OUT,
        &left.ffn_out,
        &right.ffn_out,
        None,
    )?);
    quantities.push(compare_quantity(
        RESIDUAL_OUT,
        &left.residual_out,
        &right.residual_out,
        None,
    )?);
    quantities.push(compare_quantity(
        HC_ATTN_PRE,
        &left.attn_mixes.pre,
        &right.attn_mixes.pre,
        None,
    )?);
    quantities.push(compare_quantity(
        HC_ATTN_POST,
        &left.attn_mixes.post,
        &right.attn_mixes.post,
        None,
    )?);
    quantities.push(compare_quantity(
        HC_ATTN_COMB,
        &left.attn_mixes.comb,
        &right.attn_mixes.comb,
        None,
    )?);
    quantities.push(compare_quantity(
        HC_FFN_PRE,
        &left.ffn_mixes.pre,
        &right.ffn_mixes.pre,
        None,
    )?);
    quantities.push(compare_quantity(
        HC_FFN_POST,
        &left.ffn_mixes.post,
        &right.ffn_mixes.post,
        None,
    )?);
    quantities.push(compare_quantity(
        HC_FFN_COMB,
        &left.ffn_mixes.comb,
        &right.ffn_mixes.comb,
        None,
    )?);
    validate_comparison_quantities(&quantities)?;
    Ok(summarize(quantities))
}

/// Compares a device MhcEntry readback against the host reference entry with m11 b7 classes.
///
/// The caller supplies the expected `xhat_normed` capture built in the kernel's order from the
/// host `attn_in`, the staged RMS weight and the **device's own** `1/rms` readback. m11 measured
/// that capture bit-exact, so it is compared exactly here; the `1/rms` scalar itself carries the
/// `3e-4` bar and the coefficient groups carry the `1e-4` bar. No downstream host arithmetic is
/// included, because m11 did not measure it and the window must not invent a bar for it.
///
/// # Errors
/// [EngineError::Shape] when the entry's slices have the wrong lengths.
pub fn compare_entry(
    host: &LayerZeroTrace,
    entry: &LayerZeroEntry,
    expected_capture: &[f32],
    expected_inv_rms: f32,
    actual_inv_rms: f32,
) -> Result<LayerZeroComparison, EngineError> {
    let mut quantities = Vec::with_capacity(LAYER_ZERO_ENTRY_QUANTITIES.len());
    quantities.push(compare_quantity(
        "pre_attention_hidden",
        expected_capture,
        &entry.pre_attention_hidden,
        None,
    )?);
    quantities.push(compare_quantity(
        "hc_attn_pre",
        &host.attn_mixes.pre,
        &entry.pre,
        Some(M11_HC_COEFF_BAR),
    )?);
    quantities.push(compare_quantity(
        "hc_attn_post",
        &host.attn_mixes.post,
        &entry.post,
        Some(M11_HC_COEFF_BAR),
    )?);
    quantities.push(compare_quantity(
        "hc_attn_comb",
        &host.attn_mixes.comb,
        &entry.comb,
        Some(M11_HC_COEFF_BAR),
    )?);
    quantities.push(compare_quantity(
        "inv_rms",
        &[expected_inv_rms],
        &[actual_inv_rms],
        Some(M11_INV_RMS_BAR),
    )?);
    validate_quantity_names(&quantities, LAYER_ZERO_ENTRY_QUANTITIES.as_slice())?;
    Ok(summarize(quantities))
}

/// One quantity's bit-level comparison and optional declared bar.
fn compare_quantity(
    name: &'static str,
    left: &[f32],
    right: &[f32],
    bar: Option<f32>,
) -> Result<LayerZeroQuantity, EngineError> {
    if left.len() != right.len() {
        return Err(EngineError::Shape {
            what: "layer-zero comparison".to_owned(),
            detail: format!(
                "{name}: expected length {} against {}",
                left.len(),
                right.len()
            ),
        });
    }
    let mut exact = true;
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    let mut first_difference = None;
    for (index, (expected, actual)) in left.iter().zip(right).enumerate() {
        let expected_bits = expected.to_bits();
        let actual_bits = actual.to_bits();
        if expected_bits != actual_bits {
            exact = false;
            if first_difference.is_none() {
                first_difference = Some(QuantityDifference {
                    index,
                    expected_bits,
                    actual_bits,
                });
            }
        }
        let delta = (*expected - *actual).abs();
        if !delta.is_finite() || delta > max_abs {
            max_abs = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
        }
        let ulp = ulp_distance(expected_bits, actual_bits);
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    Ok(LayerZeroQuantity {
        name,
        max_abs,
        exact,
        max_ulp,
        first_difference,
        bar,
    })
}

/// The distance between two f32 bit patterns in ULPs, treating `-0.0` as `+0.0`.
fn ulp_distance(left: u32, right: u32) -> u32 {
    fn ordered(bits: u32) -> u32 {
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

/// Builds the overall class from the quantity classes.
fn summarize(quantities: Vec<LayerZeroQuantity>) -> LayerZeroComparison {
    let exact = quantities.iter().all(|quantity| quantity.exact);
    LayerZeroComparison {
        class: if exact {
            ComparisonClass::Exact
        } else {
            ComparisonClass::Bar
        },
        quantities,
    }
}

/// Refuses a comparison whose quantity names or count drifted from the pinned c2 set.
fn validate_comparison_quantities(quantities: &[LayerZeroQuantity]) -> Result<(), EngineError> {
    validate_quantity_names(quantities, LAYER_ZERO_COMPARISON_QUANTITIES.as_slice())
}

/// Refuses a comparison whose names or count drifted from an expected pinned set.
fn validate_quantity_names(
    quantities: &[LayerZeroQuantity],
    expected: &[&'static str],
) -> Result<(), EngineError> {
    let names = quantities
        .iter()
        .map(|quantity| quantity.name)
        .collect::<Vec<_>>();
    if names.as_slice() != expected {
        return Err(EngineError::Shape {
            what: "layer-zero comparison".to_owned(),
            detail: format!("comparison quantities are {names:?}; expected {expected:?}"),
        });
    }
    Ok(())
}

/// The class of a layer-0 comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonClass {
    /// Every compared value is bit-identical.
    Exact,
    /// At least one value differs; each such quantity needs its declared bar to pass.
    Bar,
}

/// One compared boundary quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroQuantity {
    /// The boundary tensor name.
    pub name: &'static str,
    /// The largest absolute difference.
    pub max_abs: f32,
    /// Whether every value is bit-identical.
    pub exact: bool,
    /// The largest ULP distance between the two values.
    pub max_ulp: u32,
    /// The first bit-level difference, if any.
    pub first_difference: Option<QuantityDifference>,
    /// The declared bar for this quantity, if one applies.
    pub bar: Option<f32>,
}

impl LayerZeroQuantity {
    /// Whether this quantity is exact or within its declared bar.
    #[must_use]
    pub fn within_declared_bar(&self) -> bool {
        self.exact
            || match self.bar {
                Some(bar) => self.max_abs <= bar,
                None => false,
            }
    }

    /// The report line for this quantity, with the first differing bits and the ULP delta.
    #[must_use]
    pub fn report_line(&self) -> String {
        let class = if self.exact {
            "exact".to_string()
        } else {
            match self.bar {
                Some(bar) => format!("bar (<= {bar:.1e})"),
                None => "differ (no declared bar)".to_string(),
            }
        };
        let first = match self.first_difference {
            Some(difference) => format!(
                "index={} expected=0x{:08x} actual=0x{:08x}",
                difference.index, difference.expected_bits, difference.actual_bits
            ),
            None => "none".to_string(),
        };
        format!(
            "  {}: class={} exact={} max_abs={:.6e} max_ulp={} first_differing={}",
            self.name, class, self.exact, self.max_abs, self.max_ulp, first
        )
    }
}

/// The c2 result for one layer-0 row.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroComparison {
    /// The overall class.
    pub class: ComparisonClass,
    /// One row per compared boundary quantity.
    pub quantities: Vec<LayerZeroQuantity>,
}

impl LayerZeroComparison {
    /// Whether every differing quantity is within its declared bar.
    #[must_use]
    pub fn within_declared_bars(&self) -> bool {
        self.quantities
            .iter()
            .all(LayerZeroQuantity::within_declared_bar)
    }

    /// The report lines: every per-quantity result first, then the overall class.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines =
            vec!["layer-0 per-quantity comparison (one layer, not a token):".to_string()];
        for quantity in &self.quantities {
            lines.push(quantity.report_line());
        }
        let verdict = if self.class == ComparisonClass::Exact {
            "EXACT".to_string()
        } else if self.within_declared_bars() {
            "BAR (every differing quantity is within its declared bar)".to_string()
        } else {
            "BAR (a differing quantity has no declared bar, or exceeds it)".to_string()
        };
        lines.push(format!("layer-0 comparison class: {verdict}"));
        lines
    }
}

/// Compares a layer-0 host trace against the reference engine's layer-0 boundary row.
///
/// The host pass composes the same reference functions over the same bytes as the reference engine,
/// so the honest class for this comparison is bit identity; the function still reports max_abs for
/// a reader who wants it. Every compared quantity must be present and f32.
///
/// # Errors
/// A missing boundary tensor or a length mismatch.
pub fn compare_to_reference(
    trace: &LayerZeroTrace,
    row: &crate::forward::boundary::LayerBoundary,
) -> Result<LayerZeroComparison, EngineError> {
    use crate::forward::boundary::{
        ATTENTION_OUT, FFN_OUT, HC_ATTN_COMB, HC_ATTN_POST, HC_ATTN_PRE, HC_FFN_COMB, HC_FFN_POST,
        HC_FFN_PRE, POST_ATTENTION_HIDDEN, PRE_ATTENTION_HIDDEN, PRE_FFN_HIDDEN, RESIDUAL_OUT,
    };
    let mut quantities = Vec::with_capacity(LAYER_ZERO_COMPARISON_QUANTITIES.len());
    let mut compare = |name: &'static str, values: &[f32]| -> Result<(), EngineError> {
        let boundary = row.tensor(name).ok_or_else(|| EngineError::Shape {
            what: "layer-zero comparison".to_owned(),
            detail: format!("reference row has no {name}"),
        })?;
        let reference = boundary.data().as_f32().ok_or_else(|| EngineError::Shape {
            what: "layer-zero comparison".to_owned(),
            detail: format!("reference {name} is not f32"),
        })?;
        quantities.push(compare_quantity(name, values, reference, None)?);
        Ok(())
    };
    compare(PRE_ATTENTION_HIDDEN, &trace.pre_attention_hidden)?;
    compare(ATTENTION_OUT, &trace.attention_out)?;
    compare(POST_ATTENTION_HIDDEN, &trace.post_attention_hidden)?;
    compare(PRE_FFN_HIDDEN, &trace.pre_ffn_hidden)?;
    compare(FFN_OUT, &trace.ffn_out)?;
    compare(RESIDUAL_OUT, &trace.residual_out)?;
    compare(HC_ATTN_PRE, &trace.attn_mixes.pre)?;
    compare(HC_ATTN_POST, &trace.attn_mixes.post)?;
    compare(HC_ATTN_COMB, &trace.attn_mixes.comb)?;
    compare(HC_FFN_PRE, &trace.ffn_mixes.pre)?;
    compare(HC_FFN_POST, &trace.ffn_mixes.post)?;
    compare(HC_FFN_COMB, &trace.ffn_mixes.comb)?;
    validate_comparison_quantities(&quantities)?;
    Ok(summarize(quantities))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::forward::boundary::{
        ATTENTION_OUT, FFN_OUT, HC_ATTN_COMB, HC_ATTN_POST, HC_ATTN_PRE, HC_FFN_COMB, HC_FFN_POST,
        HC_FFN_PRE, POST_ATTENTION_HIDDEN, PRE_ATTENTION_HIDDEN, PRE_FFN_HIDDEN, RESIDUAL_OUT,
    };
    use crate::forward::{InferenceEngine, SequenceState};
    use crate::tensor::{TensorPlan, TensorSpec};
    use crate::weights::{WeightError, WeightSource};

    /// A plan-shaped source of finite numbers for the exact comparison. Every tensor keeps the
    /// plan's shape but is served as f32 with a small positive pattern, so nothing here is NaN or
    /// inf; the same construction the reference's one-block tests use.
    struct Hand {
        specs: Vec<TensorSpec>,
    }

    impl Hand {
        fn new(plan: &TensorPlan) -> Self {
            Self {
                specs: plan.specs().to_vec(),
            }
        }
    }

    impl WeightSource for Hand {
        fn names(&self) -> Vec<String> {
            self.specs.iter().map(|spec| spec.name.clone()).collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.specs
                .iter()
                .find(|spec| spec.name == name)
                .map(|spec| TensorSpec {
                    // Every tensor is served as f32 except where the checkpoint's storage format
                    // is the only way to recover the value (the Engram wkv projection, audit F4).
                    dtype: crate::tensor::checkpoint_storage_dtype(name).unwrap_or(DType::F32),
                    ..spec.clone()
                })
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            let spec = self.spec(name)?;
            let elements = usize::try_from(spec.elements()).unwrap_or(usize::MAX);
            let value_at = |index: usize| 0.02 + 0.01 * (index % 5) as f32;
            let mut bytes = Vec::with_capacity(elements.saturating_mul(4));
            match spec.dtype {
                DType::F8E4M3 => {
                    for index in 0..elements {
                        bytes.push(crate::quant::e4m3_encode(value_at(index)));
                    }
                }
                DType::F8E8M0 => bytes.resize(elements, 127u8),
                _ => {
                    for index in 0..elements {
                        bytes.extend_from_slice(&value_at(index).to_le_bytes());
                    }
                }
            }
            Ok(bytes)
        }
    }

    fn config() -> ModelConfig {
        let mut config = ModelConfig::tiny();
        // Make layer 0 SWA-only so the pass has exactly the stages this module names. The released
        // layer 0 has compress ratio 0; tiny's ratio list does not, and the comparison must not
        // depend on the tiny fixture's compressor choice.
        config.compress_ratios = vec![0; config.n_layers + config.n_mtp_layers];
        config
    }

    fn same_bits(left: &[f32], right: &[f32]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }

    #[test]
    fn the_host_layer_zero_matches_the_reference_boundary_exactly() {
        let config = config();
        let plan = TensorPlan::language_model(&config);
        let limit = REFERENCE_TENSOR_LIMIT;
        let source = Hand::new(&plan);
        let token = 7u32;
        let trace = run_layer_zero(&config, &source, token, limit, None).expect("host layer zero");
        let with_row =
            run_layer_zero_with_embedding(&config, &source, token, &trace.embedding, limit, None)
                .expect("the caller-supplied row path");
        assert_eq!(with_row, trace, "the row path must be the same pass");

        let engine = InferenceEngine::new(config.clone(), Box::new(Hand::new(&plan)), limit)
            .expect("reference engine");
        let mut state = SequenceState::new(&config);
        let observed = engine
            .prefill_observed(&mut state, &[token])
            .expect("reference prefill");
        let row = observed.boundary.row(0).expect("layer-0 boundary row");
        let f32_tensor = |name: &str| row.tensor(name).expect(name).data().as_f32().expect(name);

        assert!(same_bits(
            &trace.pre_attention_hidden,
            f32_tensor(PRE_ATTENTION_HIDDEN)
        ));
        assert!(same_bits(&trace.attention_out, f32_tensor(ATTENTION_OUT)));
        assert!(same_bits(
            &trace.post_attention_hidden,
            f32_tensor(POST_ATTENTION_HIDDEN)
        ));
        assert!(same_bits(&trace.pre_ffn_hidden, f32_tensor(PRE_FFN_HIDDEN)));
        assert!(same_bits(&trace.ffn_out, f32_tensor(FFN_OUT)));
        assert!(same_bits(&trace.residual_out, f32_tensor(RESIDUAL_OUT)));
        assert!(same_bits(&trace.attn_mixes.pre, f32_tensor(HC_ATTN_PRE)));
        assert!(same_bits(&trace.attn_mixes.post, f32_tensor(HC_ATTN_POST)));
        assert!(same_bits(&trace.attn_mixes.comb, f32_tensor(HC_ATTN_COMB)));
        assert!(same_bits(&trace.ffn_mixes.pre, f32_tensor(HC_FFN_PRE)));
        assert!(same_bits(&trace.ffn_mixes.post, f32_tensor(HC_FFN_POST)));
        assert!(same_bits(&trace.ffn_mixes.comb, f32_tensor(HC_FFN_COMB)));
        let comparison = compare_to_reference(&trace, row).expect("layer-0 comparison");
        assert_eq!(comparison.class, ComparisonClass::Exact);
        assert_eq!(comparison.quantities.len(), 12);
        assert_eq!(
            comparison
                .quantities
                .iter()
                .map(|quantity| quantity.name)
                .collect::<Vec<_>>(),
            LAYER_ZERO_COMPARISON_QUANTITIES.to_vec(),
            "compare_to_reference must report exactly the pinned c2 quantity set"
        );
        assert!(
            comparison.quantities.iter().all(|quantity| quantity.exact),
            "{comparison:?}"
        );
        let mut altered = trace.clone();
        altered.residual_out[0] += 1.0;
        let mismatch = compare_to_reference(&altered, row).expect("altered comparison");
        assert_eq!(mismatch.class, ComparisonClass::Bar);
        assert!(
            mismatch.quantities.iter().any(|quantity| !quantity.exact),
            "a changed value must not be reported as exact"
        );
        let same = compare_traces(&trace, &trace.clone()).expect("a trace against itself");
        assert_eq!(same.class, ComparisonClass::Exact);
        assert_eq!(same.quantities.len(), 12);
        assert_eq!(
            same.quantities
                .iter()
                .map(|quantity| quantity.name)
                .collect::<Vec<_>>(),
            LAYER_ZERO_COMPARISON_QUANTITIES.to_vec(),
            "compare_traces must report exactly the pinned c2 quantity set"
        );
        let mut dropped_reference = comparison.quantities.clone();
        dropped_reference.remove(0);
        assert!(
            validate_comparison_quantities(&dropped_reference).is_err(),
            "a dropped reference quantity must be refused"
        );
        let mut dropped_traces = same.quantities.clone();
        dropped_traces.remove(0);
        assert!(
            validate_comparison_quantities(&dropped_traces).is_err(),
            "a dropped trace quantity must be refused"
        );
        let changed = compare_traces(&trace, &altered).expect("a changed trace");
        assert_eq!(changed.class, ComparisonClass::Bar);
        assert_eq!(
            trace
                .stages
                .iter()
                .map(|event| event.stage)
                .collect::<Vec<_>>(),
            LAYER_ZERO_HOST_STAGES.to_vec()
        );
        for event in &trace.stages {
            assert!(event.host_bytes_out > 0, "{} has no output", event.stage);
        }
    }

    #[test]
    fn a_nonzero_layer_zero_ratio_is_refused() {
        let config = ModelConfig::tiny();
        assert_eq!(config.compress_ratios.first().copied(), Some(2));
        let plan = TensorPlan::language_model(&config);
        let source = Hand::new(&plan);
        let error = run_layer_zero(&config, &source, 0, REFERENCE_TENSOR_LIMIT, None)
            .expect_err("ratio 2 is not the SWA-only layer-0 schedule");
        let text = error.to_string();
        assert!(text.contains("compress ratio 2"), "{text}");
        let error = run_layer_zero_with_embedding(
            &config,
            &source,
            0,
            &vec![0.0; config.dim],
            REFERENCE_TENSOR_LIMIT,
            None,
        )
        .expect_err("the row path must refuse too");
        assert!(error.to_string().contains("compress ratio 2"), "{error}");
    }

    #[test]
    fn the_device_entry_handoff_is_the_same_arithmetic_as_the_host_entry() {
        let config = config();
        let plan = TensorPlan::language_model(&config);
        let limit = REFERENCE_TENSOR_LIMIT;
        let source = Hand::new(&plan);
        let token = 7u32;
        let host = run_layer_zero(&config, &source, token, limit, None).expect("host layer zero");
        let entry = LayerZeroEntry {
            pre_attention_hidden: host.pre_attention_hidden.clone(),
            pre: host.attn_mixes.pre.clone(),
            post: host.attn_mixes.post.clone(),
            comb: host.attn_mixes.comb.clone(),
            source: LayerZeroEntrySource::Host,
        };
        let device = run_layer_zero(&config, &source, token, limit, Some(entry))
            .expect("device-entry layer zero");
        assert!(same_bits(&host.residual_out, &device.residual_out));
        assert_eq!(
            device
                .stages
                .iter()
                .map(|event| event.stage)
                .collect::<Vec<_>>(),
            host.stages
                .iter()
                .map(|event| event.stage)
                .collect::<Vec<_>>()
        );

        let mut altered = LayerZeroEntry {
            pre_attention_hidden: host.pre_attention_hidden.clone(),
            pre: host.attn_mixes.pre.clone(),
            post: host.attn_mixes.post.clone(),
            comb: host.attn_mixes.comb.clone(),
            source: LayerZeroEntrySource::Host,
        };
        altered.pre_attention_hidden[0] += 1.0;
        let changed =
            run_layer_zero(&config, &source, token, limit, Some(altered)).expect("altered entry");
        assert!(
            !same_bits(&host.residual_out, &changed.residual_out),
            "the handoff must actually feed the downstream arithmetic"
        );
    }

    #[test]
    fn the_entry_comparison_uses_the_m11_b7_classes() {
        let config = config();
        let plan = TensorPlan::language_model(&config);
        let source = Hand::new(&plan);
        let host = run_layer_zero(&config, &source, 7, REFERENCE_TENSOR_LIMIT, None)
            .expect("host layer zero");
        let mut entry = LayerZeroEntry {
            pre_attention_hidden: host.pre_attention_hidden.clone(),
            pre: host.attn_mixes.pre.clone(),
            post: host.attn_mixes.post.clone(),
            comb: host.attn_mixes.comb.clone(),
            source: LayerZeroEntrySource::Device,
        };
        let exact = compare_entry(&host, &entry, &host.pre_attention_hidden, 0.5, 0.5)
            .expect("exact entry");
        assert_eq!(exact.class, ComparisonClass::Exact);
        assert_eq!(
            exact
                .quantities
                .iter()
                .map(|quantity| quantity.name)
                .collect::<Vec<_>>(),
            LAYER_ZERO_ENTRY_QUANTITIES.to_vec()
        );
        assert!(
            exact
                .report_lines()
                .iter()
                .any(|line| line.contains("class=exact"))
        );

        entry.pre[0] += 5e-5;
        entry.post[0] -= 5e-5;
        entry.comb[0] += 5e-5;
        let within = compare_entry(&host, &entry, &host.pre_attention_hidden, 0.5, 0.5)
            .expect("within-bar entry");
        assert_eq!(within.class, ComparisonClass::Bar);
        assert!(
            within.within_declared_bars(),
            "each perturbation is inside its m11 class: {within:?}"
        );
        let text = within.report_lines().join("\n");
        assert!(text.contains("class=bar"), "{text}");
        assert!(text.contains("index=0"), "{text}");
        assert!(text.contains("expected=0x"), "{text}");

        entry.pre[0] += 1.0;
        let outside = compare_entry(&host, &entry, &host.pre_attention_hidden, 0.5, 0.5)
            .expect("outside-bar entry");
        assert!(!outside.within_declared_bars(), "{outside:?}");
        assert!(
            outside
                .quantities
                .iter()
                .any(|quantity| !quantity.within_declared_bar())
        );

        let clean = LayerZeroEntry {
            pre_attention_hidden: host.pre_attention_hidden.clone(),
            pre: host.attn_mixes.pre.clone(),
            post: host.attn_mixes.post.clone(),
            comb: host.attn_mixes.comb.clone(),
            source: LayerZeroEntrySource::Device,
        };
        let outside_scalar =
            compare_entry(&host, &clean, &host.pre_attention_hidden, 0.5, 0.5 + 1e-3)
                .expect("outside-scalar entry");
        assert!(!outside_scalar.within_declared_bars(), "{outside_scalar:?}");
    }

    #[test]
    fn the_diagnostic_reports_the_first_differing_index_bits_and_ulp() {
        let config = config();
        let plan = TensorPlan::language_model(&config);
        let source = Hand::new(&plan);
        let trace = run_layer_zero(&config, &source, 7, REFERENCE_TENSOR_LIMIT, None)
            .expect("host layer zero");
        let mut altered = trace.clone();
        let expected = altered.residual_out[3];
        altered.residual_out[3] = f32::from_bits(expected.to_bits() + 1);
        let comparison = compare_traces(&trace, &altered).expect("one-ULP comparison");
        let quantity = comparison
            .quantities
            .iter()
            .find(|quantity| quantity.name == RESIDUAL_OUT)
            .expect("residual_out quantity");
        assert!(!quantity.exact);
        assert_eq!(quantity.max_ulp, 1);
        let difference = quantity.first_difference.expect("first difference");
        assert_eq!(difference.index, 3);
        assert_eq!(difference.expected_bits, expected.to_bits());
        assert_eq!(difference.actual_bits, expected.to_bits() + 1);
        let text = quantity.report_line();
        assert!(text.contains("max_ulp=1"), "{text}");
        assert!(text.contains("first_differing=index=3"), "{text}");
        assert!(text.contains("expected=0x"), "{text}");
    }

    #[test]
    fn the_kernel_update_uses_fma_in_the_kernel_order() {
        // (1+2^-23)*(1-2^-23) is 1-2^-46 exactly. Added to -1, the fused form keeps the
        // -2^-46 and the mul-then-add twin rounds the product to 1.0 first and gives zero.
        let width = 1;
        let mut b = [0.0f32; 16];
        b[0] = 1.0 + 2.0f32.powi(-23);
        let x = [1.0 - 2.0f32.powi(-23), 0.0, 0.0, 0.0];
        let mut c = [0.0f32; 4];
        c[0] = -1.0;
        let y = [1.0f32];
        let fused = kernel_mhc_update(&x, &b, &c, &y);
        let mut plain = [0.0f32; 4];
        for stream in 0..4 {
            let mut acc = std::hint::black_box(c[stream]) * std::hint::black_box(y[0]);
            for source in 0..4 {
                let product = std::hint::black_box(b[stream * 4 + source])
                    * std::hint::black_box(x[source * width]);
                acc += std::hint::black_box(product);
            }
            plain[stream] = acc;
        }
        assert_ne!(fused[0].to_bits(), plain[0].to_bits());
        assert_eq!(fused[0], -2.0f32.powi(-46));
        assert_eq!(plain[0], 0.0);
    }

    #[test]
    fn the_kernel_update_indexes_y_by_the_per_stream_width() {
        // The kernel has one Y value per element and four X streams over it. A regression that
        // treats the four-stream length as the per-stream width would index Y out of bounds at
        // the released 5,120-element row; this test exercises that shape without a device.
        let width = 5_120;
        let mut x = vec![0.0f32; 4 * width];
        for stream in 0..4 {
            for index in 0..width {
                x[stream * width + index] = (stream as f32) + (index as f32) * 1e-3;
            }
        }
        let y = vec![0.0f32; width];
        let (b, c, _, _) = tests_seed();
        let updated = kernel_mhc_update(&x, &b, &c, &y);
        assert_eq!(updated.len(), 4 * width);
        assert_eq!(updated, x);
        let xhat = kernel_mhc_xhat(&updated, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(xhat.len(), width);
        assert_eq!(xhat, x[..width]);
    }

    #[test]
    fn the_kernel_xhat_uses_fma_in_stream_order() {
        let x_l = [-1.0f32, 1.0 - 2.0f32.powi(-23), 0.0, 0.0];
        let a = [1.0f32, 1.0 + 2.0f32.powi(-23), 0.0, 0.0];
        let fused = kernel_mhc_xhat(&x_l, &a);
        let mut plain = 0.0f32;
        for stream in 0..4 {
            plain += std::hint::black_box(a[stream]) * std::hint::black_box(x_l[stream]);
        }
        assert_ne!(fused[0].to_bits(), plain.to_bits());
        assert_eq!(fused[0], -2.0f32.powi(-46));
        assert_eq!(plain, 0.0);
    }

    #[test]
    fn the_kernel_entry_seed_reproduces_the_admitted_row() {
        let x = [0.25f32, -0.5, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0];
        let (b, c, a, y) = tests_seed();
        let updated = kernel_mhc_update(&x, &b, &c, &y);
        assert_eq!(updated, x);
        let xhat = kernel_mhc_xhat(&updated, &a);
        assert_eq!(xhat, vec![0.25, -0.5]);
        let weight = [2.0f32, 3.0];
        let capture = kernel_mhc_capture(&xhat, &weight, 0.5);
        assert_eq!(capture, vec![0.25, -0.75]);
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(f32::from_bits(0x3f80_8000)), 1.0);
    }

    #[test]
    fn the_fused_kernel_update_rounds_the_first_b_product() {
        // The pinned kernel's step 1 per stream is q = b[s][0]*x[0] (rounded), then
        // fma(c[s], y, q), then the remaining B terms fused. This case separates that order from
        // the source-level C-Y-first order that [kernel_mhc_update] implements.
        let x = [1.0f32, 0.0, 0.0, 0.0];
        let mut b = [0.0f32; 16];
        b[0] = -(1.0f32 + 2f32.powi(-22));
        let mut c = [0.0f32; 4];
        c[0] = 1.0f32 + 2f32.powi(-23);
        let y = [1.0f32 + 2f32.powi(-23)];
        let fused = kernel_mhc_update_fused(&x, &b, &c, &y);
        let source_order = kernel_mhc_update(&x, &b, &c, &y);
        assert_eq!(fused[0], 2f32.powi(-46));
        assert_eq!(source_order[0], 0.0);
        assert_ne!(fused[0].to_bits(), source_order[0].to_bits());
    }

    /// The layer-0 boundary seed, exposed to this module's tests without a device type.
    fn tests_seed() -> ([f32; 16], [f32; 4], [f32; 4], Vec<f32>) {
        let mut b = [0.0f32; 16];
        for (index, value) in b.iter_mut().enumerate() {
            if index / 4 == index % 4 {
                *value = 1.0;
            }
        }
        (b, [0.0f32; 4], [1.0f32, 0.0, 0.0, 0.0], vec![0.0f32; 2])
    }
}
