//! The DSpark MTP stages: the draft blocks and the head that turns them into drafts.
//!
//! Spec: docs/engine/01-forward-order.md sections 2.3 (the MTP tensor list), 3.8 (the DSpark order),
//! 3.2 (the block order the stages reuse) and 7 (what the reference does and does not contain). The
//! reference is Transformer.forward_spec (model.py:1274-1282), DSparkBlock (model.py:1100-1126),
//! DSpark.forward_embed (model.py:1128-1135), DSpark.forward_head (model.py:1137-1156),
//! DSparkMarkovHead (model.py:1077-1086) and DSparkConfidenceHead (model.py:1089-1097).
//!
//! # What this module is
//!
//! One draft pass. main_x = main_norm(main_proj(cat(main_hiddens))) from the backbone taps
//! (model.py:1130); the draft positions start with the anchor token and continue with the configured
//! noise token (model.py:1131-1132), embedded and expanded over the hc copies (model.py:1133-1134);
//! each of the config.n_mtp_layers stages then runs the full block of section 3.2 - the mHC mixes, the
//! attention that also sees main_x, the MoE - and the last stage's norm feeds the LM head. The Markov
//! head biases the logits of one position at a time to produce the next draft (model.py:1149-1153) and
//! the confidence head scores every position (model.py:1155).
//!
//! # What is the reference, and what is this pass's own choice
//!
//! - REFERENCE: the tensor names and shapes (section 2.3), the stage order and the mHC ordering
//!   (section 3.2, model.py:968-994), main_x (model.py:1130), the draft ids (model.py:1131-1132), the
//!   Markov bias added to the base logits with the position's own token feeding its own bias
//!   (model.py:1150-1152), and the confidence head as a projection of [hidden ; markov_embed]
//!   (model.py:1095-1097).
//! - The confidence values are NOT probabilities. The reference returns the raw fp32 projection
//!   (model.py:1095-1097; model.py:1092 says the module is fp32 "for fp32 confidence score") with no
//!   sigmoid and no normalisation. The 0.1 the released sampling policy compares them against is not
//!   part of the reference forward: section 7 records that no acceptance rule exists in it.
//! - Sampling here is GREEDY (linalg::argmax). The reference calls sample(logits, temperature)
//!   (model.py:1153), a Gumbel-max draw (model.py:1285-1292); greedy is its temperature-0 case and is
//!   what makes this pass deterministic and testable. Temperature is a sampling parameter, not a
//!   property of the forward.
//! - The reference returns None at start_pos 0: Transformer.forward_spec returns None
//!   (model.py:1274-1282) because the prefill branch only seeds the window cache and returns x
//!   unchanged (model.py:1123-1125). [dspark_forward] therefore refuses that position with
//!   [EngineError::Shape] instead of inventing a draft the reference does not produce.
//! - Every weight is read through [linalg::decode_by_name] under the caller's element limit, and that
//!   limit is what refuses the released configuration: embed.weight and head.weight are
//!   129,280 * 5,120 = 661,913,600 elements and mtp.0.main_proj is 5,120 * 15,360 = 78,643,200, all far
//!   above the 33,554,432-element reference limit. A real engine pages rows through the packed store
//!   (crates/sparkflash-engram, [crate::loader]); this reference pass decodes whole tensors.
//! - This is an f32 reference over decoded storage: the released kernels quantise activations
//!   (section 6) and the stored fp8/bf16 weights are dequantised on the way in.

use crate::config::ModelConfig;
use crate::forward::attention::{self, AttnState};
use crate::forward::boundary::{self, BoundaryTable, LayerBoundary};
use crate::forward::mhc::{self, Mixes};
use crate::forward::moe::{self, MoeScratch};
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::weights::WeightSource;

/// What one DSpark pass produced (model.py:1156, 1280-1282).
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkOutput {
    /// The anchor followed by config.dspark_block_size sampled drafts: the output_ids of
    /// model.py:1146-1153, so block_size + 1 ids of which the first is the anchor handed in.
    pub draft: Vec<u32>,
    /// One score per draft position (model.py:1155): the raw fp32 projection of
    /// [hidden ; markov_embed]. An unnormalised score, NOT a probability; see the module docs.
    pub confidence: Vec<f32>,
    /// The base head logits with the Markov bias added (model.py:1145-1151), position-major:
    /// config.dspark_block_size * config.vocab_size values.
    pub logits: Vec<f32>,
}

/// A shape refusal naming the operation that could not be applied.
fn shape_error(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}

/// Reads a dense fp8 weight with its 32x32 E8M0 grid - the dense entries of the plan, such as
/// mtp.0.main_proj (tensor.rs:345-349).
///
/// # Errors
///
/// [EngineError::TooLarge] above limit, [EngineError::Missing] when the source lacks the tensor, and
/// [EngineError::Shape] when the grid does not fit the weight.
fn read_dense(
    name: &str,
    src: &dyn WeightSource,
    rows: usize,
    cols: usize,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let scale_name = name.replace("weight", "scale");
    let mut weight = linalg::decode_by_name(src, name, limit)?;
    let scales = linalg::decode_by_name(src, &scale_name, limit)?;
    linalg::apply_scales(&mut weight, rows, cols, &scales, ScaleGrid::Block32)?;
    Ok(weight)
}

/// One position's Markov bias: head @ embed, the logits_bias of model.py:1084-1085.
///
/// head is the [vocab, rank] table and embed the rank-long markov embedding of the position's token,
/// so out[v] = sum_k head[v * rank + k] * embed[k]. The caller owns the output buffer: this runs once
/// per draft position, on the path that must not allocate per position.
///
/// # Errors
///
/// [EngineError::Shape] when the embedding is not rank long, the table is not vocab x rank, or out is
/// not one value per vocabulary row.
fn markov_bias(
    embed: &[f32],
    head: &[f32],
    vocab: usize,
    rank: usize,
    out: &mut [f32],
) -> Result<(), EngineError> {
    if embed.len() != rank {
        return Err(shape_error(
            "markov head",
            format!(
                "one position's embedding is {rank} long, got {}",
                embed.len()
            ),
        ));
    }
    if head.len() != vocab * rank {
        return Err(shape_error(
            "markov head",
            format!(
                "the markov head is {vocab} x {rank} = {}, got {}",
                vocab * rank,
                head.len()
            ),
        ));
    }
    if out.len() != vocab {
        return Err(shape_error(
            "markov head",
            format!(
                "the bias is one value per vocabulary row, {vocab}, got {}",
                out.len()
            ),
        ));
    }
    linalg::matmul(out, embed, head, vocab, rank, 1)
}

/// One position's confidence: proj @ [hidden ; markov_embed] (model.py:1095-1097).
///
/// combined is the caller's [dim + rank] scratch buffer, filled here rather than allocated, because
/// this also runs once per draft position. The result is the raw fp32 projection: no activation is
/// applied, which is why a caller must treat it as a score and not as a probability (module docs).
///
/// # Errors
///
/// [EngineError::Shape] when any length disagrees with the configuration.
fn confidence_score(
    hidden: &[f32],
    markov_embed: &[f32],
    proj: &[f32],
    dim: usize,
    rank: usize,
    combined: &mut [f32],
) -> Result<f32, EngineError> {
    if hidden.len() != dim {
        return Err(shape_error(
            "confidence head",
            format!("the hidden state is {dim} wide, got {}", hidden.len()),
        ));
    }
    if markov_embed.len() != rank {
        return Err(shape_error(
            "confidence head",
            format!(
                "the markov embedding is {rank} wide, got {}",
                markov_embed.len()
            ),
        ));
    }
    if proj.len() != dim + rank {
        return Err(shape_error(
            "confidence head",
            format!(
                "the projection is [1, {}] = {}, got {}",
                dim + rank,
                dim + rank,
                proj.len()
            ),
        ));
    }
    if combined.len() != dim + rank {
        return Err(shape_error(
            "confidence head",
            format!(
                "the concatenation buffer is {} long, got {}",
                dim + rank,
                combined.len()
            ),
        ));
    }
    combined[..dim].copy_from_slice(hidden);
    combined[dim..].copy_from_slice(markov_embed);
    let mut score = [0.0f32; 1];
    linalg::matmul(&mut score, combined, proj, 1, dim + rank, 1)?;
    Ok(score[0])
}

/// The Markov bias of one draft token: DSparkMarkovHead.forward (model.py:1077-1086).
///
/// hidden is the position's markov embedding - the embed(token_ids) of model.py:1084, which is
/// config.dspark_markov_rank long - and the result is the head's full-vocabulary logits, which the
/// caller ADDS to the LM head's base logits (model.py:1151). It is a bias, not a distribution: the
/// reference does not normalise it, and neither does this.
///
/// The table is decoded once per call. A caller that scores a whole block should read it once and use
/// [dspark_forward], which hoists both head tables out of its position loop; this is the
/// single-position entry point of the reference module.
///
/// # Errors
///
/// [EngineError::TooLarge] above limit, [EngineError::Missing] when the source lacks the tensor, and
/// [EngineError::Shape] when the embedding or the table does not fit the configuration.
pub fn markov_head(
    config: &ModelConfig,
    prefix: &str,
    hidden: &[f32],
    src: &dyn WeightSource,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let vocab = config.vocab_size;
    let rank = config.dspark_markov_rank;
    let head = linalg::decode_by_name(src, &format!("{prefix}.markov_head.head.weight"), limit)?;
    let mut out = vec![0.0f32; vocab];
    markov_bias(hidden, &head, vocab, rank, &mut out)?;
    Ok(out)
}

/// The confidence head over one draft position: DSparkConfidenceHead.forward (model.py:1089-1097).
///
/// hidden is that position's collapsed pre-norm stream - the x that model.py:1145 hands to norm,
/// not norm(x) - and markov_embed its markov embedding. The two are concatenated to
/// [dim + dspark_markov_rank] (model.py:1096) and projected to one value.
///
/// The value is an unnormalised fp32 score, NOT a probability: the reference applies no activation
/// (model.py:1097). The released decode policy thresholds it at 0.1, but that threshold and any
/// acceptance rule belong to the speculative-decoding policy, not to this forward (section 7).
///
/// # Errors
///
/// [EngineError::TooLarge] above limit, [EngineError::Missing] when the source lacks the tensor, and
/// [EngineError::Shape] when the two vectors or the projection do not fit the configuration.
pub fn confidence(
    config: &ModelConfig,
    prefix: &str,
    hidden: &[f32],
    markov_embed: &[f32],
    src: &dyn WeightSource,
    limit: usize,
) -> Result<f32, EngineError> {
    let dim = config.dim;
    let rank = config.dspark_markov_rank;
    let proj =
        linalg::decode_by_name(src, &format!("{prefix}.confidence_head.proj.weight"), limit)?;
    let mut combined = vec![0.0f32; dim + rank];
    confidence_score(hidden, markov_embed, &proj, dim, rank, &mut combined)
}

/// The per-stage boundary buffers a capture fills while one MTP stage runs.
///
/// The block of section 3.2 keeps its stream in `x_hc` and its coefficient sets in one
/// `attn_mixes` per position and one `ffn_mixes` reused position by position, and the second half
/// of the loop works one position at a time. A capture has to collect the per-position values into
/// one row, which is what these buffers are for. They exist only when a table was asked for; the
/// default pass allocates nothing here (docs/archive/build/phase1-boundary-table.md section 1).
struct StageBoundary {
    residual_in: Vec<f32>,
    input_mix: Vec<f32>,
    attn_pre: Vec<f32>,
    attn_post: Vec<f32>,
    attn_comb: Vec<f32>,
    post_attention: Vec<f32>,
    ffn_pre: Vec<f32>,
    ffn_post: Vec<f32>,
    ffn_comb: Vec<f32>,
    pre_ffn: Vec<f32>,
    ffn_out: Vec<f32>,
}

impl StageBoundary {
    /// Buffers for one stage of a `block`-position draft pass.
    fn new(config: &ModelConfig, block: usize) -> Self {
        let hc = config.hc_mult;
        let hc_width = config.hc_width();
        Self {
            residual_in: vec![0.0; block * hc_width],
            input_mix: vec![0.0; block * hc],
            attn_pre: vec![0.0; block * hc],
            attn_post: vec![0.0; block * hc],
            attn_comb: vec![0.0; block * hc * hc],
            post_attention: vec![0.0; block * hc_width],
            ffn_pre: vec![0.0; block * hc],
            ffn_post: vec![0.0; block * hc],
            ffn_comb: vec![0.0; block * hc * hc],
            pre_ffn: vec![0.0; block * config.dim],
            ffn_out: vec![0.0; block * config.dim],
        }
    }

    /// The stage's row, in the tensor order of [boundary::TENSOR_ORDER].
    ///
    /// `attn_in` and `attn_out` are already whole-block buffers of `dspark_forward`, and
    /// `residual_out` is the stream after the stage's last position; the rest was collected per
    /// position while the stage ran.
    fn into_row(
        self,
        config: &ModelConfig,
        layer: usize,
        attn_in: &[f32],
        attn_out: &[f32],
        residual_out: &[f32],
    ) -> Result<LayerBoundary, EngineError> {
        let mut row = LayerBoundary::new(layer);
        row.push_stream(config, boundary::RESIDUAL_IN, self.residual_in)?;
        row.push_mixes(config, boundary::HC_IN_PRE, self.input_mix)?;
        row.push_mixes(config, boundary::HC_ATTN_PRE, self.attn_pre)?;
        row.push_mixes(config, boundary::HC_ATTN_POST, self.attn_post)?;
        row.push_comb(config, boundary::HC_ATTN_COMB, self.attn_comb)?;
        row.push_block_input(
            config,
            boundary::PRE_ATTENTION_HIDDEN,
            boundary::PRE_ATTENTION_XHAT,
            boundary::PRE_ATTENTION_SCALES,
            attn_in,
        )?;
        row.push_dense(config, boundary::ATTENTION_OUT, attn_out.to_vec())?;
        row.push_stream(config, boundary::POST_ATTENTION_HIDDEN, self.post_attention)?;
        row.push_mixes(config, boundary::HC_FFN_PRE, self.ffn_pre)?;
        row.push_mixes(config, boundary::HC_FFN_POST, self.ffn_post)?;
        row.push_comb(config, boundary::HC_FFN_COMB, self.ffn_comb)?;
        row.push_block_input(
            config,
            boundary::PRE_FFN_HIDDEN,
            boundary::PRE_FFN_XHAT,
            boundary::PRE_FFN_SCALES,
            &self.pre_ffn,
        )?;
        row.push_dense(config, boundary::FFN_OUT, self.ffn_out)?;
        row.push_stream(config, boundary::RESIDUAL_OUT, residual_out.to_vec())?;
        Ok(row)
    }
}

/// The DSpark stages over one chunk: Transformer.forward_spec for the decode case
/// (model.py:1274-1282) with the stage loop of model.py:1277-1279.
///
/// Inputs:
///
/// - tokens: the token ids of the chunk that produced main_hidden, which size the taps;
/// - start_pos: the position of that chunk's first token. The attention rotates the draft positions
///   at start_pos + block_size (model.py:1055), so it must be the real position, and 0 is refused
///   because the reference returns no draft from a prefill pass;
/// - main_hidden: config.dspark_target_layer_ids.len() * config.dim values per token, one row per
///   token of this chunk, the taps of the target layers (and of the block input, model.py:1264-1266)
///   in tap order;
/// - next_token: the anchor, the token the backbone sampled, which the reference puts at draft
///   position 0 (model.py:1132);
/// - stages: one attention state per MTP stage, in stage order; its length must be
///   config.n_mtp_layers.
///
/// The draft returned is [anchor, draft_1, ..., draft_block_size] (block_size + 1 ids), the
/// confidence has one score per draft position, and the logits are the base head logits with the
/// Markov bias added.
///
/// # Errors
///
/// [EngineError::Shape] when start_pos is 0, when the stage count or the tap buffer does not fit the
/// configuration, when a draft token falls outside the vocabulary, or from any stage;
/// [EngineError::TooLarge] when a weight is above limit, and [EngineError::Missing] when the source
/// lacks one.
#[allow(clippy::too_many_arguments)] // the same reason as the attention entry points: the chunk, its first position, the taps, the anchor, the stage states, the source and the limit are distinct inputs and bundling them would hide which one a call got wrong.
pub fn dspark_forward(
    config: &ModelConfig,
    tokens: &[u32],
    start_pos: usize,
    main_hidden: &[f32],
    next_token: u32,
    stages: &mut [AttnState],
    src: &dyn WeightSource,
    limit: usize,
) -> Result<DsparkOutput, EngineError> {
    dspark_forward_observed(
        config,
        tokens,
        start_pos,
        main_hidden,
        next_token,
        stages,
        src,
        limit,
        None,
    )
}

/// [dspark_forward] with the per-stage boundary rows appended to `capture`, layer id
/// `n_layers + stage` (docs/archive/build/phase1-boundary-table.md section 1).
///
/// `capture` is the table the backbone rows of the same step are in; the stages append their own
/// rows so one observed step is one table. The arithmetic is [dspark_forward]'s.
#[allow(clippy::too_many_arguments)] // dspark_forward's inputs plus the table it may fill.
pub(crate) fn dspark_forward_observed(
    config: &ModelConfig,
    tokens: &[u32],
    start_pos: usize,
    main_hidden: &[f32],
    next_token: u32,
    stages: &mut [AttnState],
    src: &dyn WeightSource,
    limit: usize,
    mut capture: Option<&mut BoundaryTable>,
) -> Result<DsparkOutput, EngineError> {
    let dim = config.dim;
    let hc = config.hc_mult;
    let hc_width = config.hc_width();
    let block = config.dspark_block_size;
    let vocab = config.vocab_size;
    let rank = config.dspark_markov_rank;
    let targets = config.dspark_target_layer_ids.len();
    let n_stages = config.n_mtp_layers;
    let eps = config.norm_eps as f32;

    // The reference has no draft to return from a prefill pass (model.py:1123-1125, 1274-1282).
    if start_pos == 0 {
        return Err(shape_error(
            "dspark draft",
            "the reference returns no draft at start_pos 0 (model.py:1123-1125, 1274-1282): a prefill pass only seeds the window cache".to_owned(),
        ));
    }
    if n_stages == 0 || block == 0 || dim == 0 || hc == 0 || targets == 0 || vocab == 0 || rank == 0
    {
        return Err(shape_error(
            "dspark draft",
            format!(
                "the configuration gives {n_stages} stages, block_size {block}, dim {dim}, hc_mult {hc}, {targets} taps, vocab {vocab} and markov rank {rank}"
            ),
        ));
    }
    if stages.len() != n_stages {
        return Err(shape_error(
            "dspark draft",
            format!(
                "{n_stages} MTP stages need {n_stages} attention states, got {}",
                stages.len()
            ),
        ));
    }
    if main_hidden.len() != tokens.len() * targets * dim {
        return Err(shape_error(
            "dspark draft",
            format!(
                "{} tokens of {targets} taps are {} wide, got {}",
                tokens.len(),
                tokens.len() * targets * dim,
                main_hidden.len()
            ),
        ));
    }

    // forward_embed (model.py:1128-1135): main_x = main_norm(main_proj(cat(main_hiddens))).
    let first = "mtp.0";
    let proj = read_dense(
        &format!("{first}.main_proj.weight"),
        src,
        dim,
        targets * dim,
        limit,
    )?;
    let main_norm = linalg::decode_by_name(src, &format!("{first}.main_norm.weight"), limit)?;
    if main_norm.len() != dim {
        return Err(shape_error(
            "dspark draft",
            format!("main_norm is {dim} long, got {}", main_norm.len()),
        ));
    }
    let mut main_x = vec![0.0f32; tokens.len() * dim];
    linalg::matmul(
        &mut main_x,
        main_hidden,
        &proj,
        dim,
        targets * dim,
        tokens.len(),
    )?;
    linalg::rms_norm(&mut main_x, &main_norm, dim, eps)?;

    // The draft positions: the anchor first, the noise token after it (model.py:1131-1132). ids grows
    // to block + 1 by the end of the head; it is the output_ids of model.py:1146.
    let mut ids = vec![config.dspark_noise_token_id; block + 1];
    ids[0] = next_token;
    let embed_table = linalg::decode_by_name(src, "embed.weight", limit)?;
    if embed_table.len() != vocab * dim {
        return Err(shape_error(
            "dspark embed",
            format!(
                "the embedding table is {vocab} x {dim} = {}, got {}",
                vocab * dim,
                embed_table.len()
            ),
        ));
    }
    let mut x_hc = vec![0.0f32; block * hc_width];
    for (position, &draft) in ids[..block].iter().enumerate() {
        let token = draft as usize;
        if token >= vocab {
            return Err(shape_error(
                "dspark embed",
                format!("draft token {token} is outside the {vocab}-row embedding table"),
            ));
        }
        let row = &embed_table[token * dim..(token + 1) * dim];
        // model.py:1134: unsqueeze over the hc axis and repeat, so every copy starts identical.
        for copy in 0..hc {
            let at = position * hc_width + copy * dim;
            x_hc[at..at + dim].copy_from_slice(row);
        }
    }
    let mut pre_mix = vec![0.0f32; block * hc];
    for position in 0..block {
        mhc::identity_pre_mix(config, &mut pre_mix[position * hc..(position + 1) * hc])?;
    }

    // Scratch, sized once for the pass: nothing here is allocated per token or per draft position.
    let mut attn_mixes: Vec<Mixes> = (0..block).map(|_| Mixes::for_config(config)).collect();
    let mut ffn_mixes = Mixes::for_config(config);
    let mut collapsed = vec![0.0f32; dim];
    let mut stream = vec![0.0f32; hc_width];
    let mut attn_in = vec![0.0f32; block * dim];
    // The stage attention writes one concatenated head vector per draft position before the output
    // projection narrows it to dim, exactly like the backbone's (forward/mod.rs); at the released
    // geometry 64 heads of 512 is 32,768 values against dim 5,120, so a block*dim buffer panics.
    let attention_width = config.n_heads.checked_mul(config.head_dim).ok_or_else(|| {
        shape_error(
            "dspark scratch",
            "the head geometry does not fit a width".to_owned(),
        )
    })?;
    let mut attn_out = vec![0.0f32; block.saturating_mul(dim)];
    let wide_attention = block.saturating_mul(attention_width);
    let mut ffn_out = vec![0.0f32; dim];
    let mut scratch = MoeScratch::for_config(config);

    // The capture buffers exist only when a table was asked for: the default pass allocates nothing
    // here (docs/archive/build/phase1-boundary-table.md section 1).
    let mut captured = capture.is_some().then(|| StageBoundary::new(config, block));

    // The stages in order (model.py:1277-1279), each one the block of section 3.2, whose attention
    // also receives main_x (model.py:1122-1126).
    for (stage, state) in stages.iter_mut().enumerate() {
        let prefix = format!("mtp.{stage}");
        // The stage's input boundaries: the stream and the input mix every position starts from
        // (model.py:982-990; the mix is the previous stage's FFN coefficient, or the identity mix
        // before the first stage, model.py:1260).
        if let Some(buffers) = captured.as_mut() {
            buffers.residual_in.copy_from_slice(&x_hc);
            buffers.input_mix.copy_from_slice(&pre_mix);
        }
        let attn_fn = linalg::decode_by_name(src, &format!("{prefix}.hc_attn_fn"), limit)?;
        let attn_scale = linalg::decode_by_name(src, &format!("{prefix}.hc_attn_scale"), limit)?;
        let attn_base = linalg::decode_by_name(src, &format!("{prefix}.hc_attn_base"), limit)?;
        let attn_norm = linalg::decode_by_name(src, &format!("{prefix}.attn_norm.weight"), limit)?;
        for position in 0..block {
            let at = position * hc_width;
            let current = &x_hc[at..at + hc_width];
            mhc::hc_mixes(
                config,
                current,
                &attn_fn,
                &attn_scale,
                &attn_base,
                &mut attn_mixes[position],
            )?;
            if let Some(buffers) = captured.as_mut() {
                let at = position * hc;
                buffers.attn_pre[at..at + hc].copy_from_slice(&attn_mixes[position].pre);
                buffers.attn_post[at..at + hc].copy_from_slice(&attn_mixes[position].post);
                buffers.attn_comb[at..at + hc * hc].copy_from_slice(&attn_mixes[position].comb);
            }
            mhc::hc_pre(
                config,
                &pre_mix[position * hc..(position + 1) * hc],
                current,
                &mut collapsed,
            )?;
            linalg::rms_norm(&mut collapsed, &attn_norm, dim, eps)?;
            attn_in[position * dim..(position + 1) * dim].copy_from_slice(&collapsed);
        }
        // The projection of the previous stage left one dim-wide row per position, so the head
        // buffer has to grow back to the wide width before every stage's attention.
        if attn_out.len() < wide_attention {
            attn_out.resize(wide_attention, 0.0);
        }
        // The engine's element limit travels with the call, so InferenceEngine::new's cap applies to
        // every tensor this stage reads, not just the ones the engine reads itself.
        attention::dspark_attention_forward(
            config,
            stage,
            &prefix,
            &attn_in,
            &main_x,
            start_pos,
            state,
            src,
            &mut attn_out,
            limit,
        )?;
        // The projection left one dim-wide row per draft position; the head buffer beyond that is
        // not a dense row and must not reach the mHC post or the captured row.
        attn_out.truncate(block * dim);

        let ffn_fn = linalg::decode_by_name(src, &format!("{prefix}.hc_ffn_fn"), limit)?;
        let ffn_scale = linalg::decode_by_name(src, &format!("{prefix}.hc_ffn_scale"), limit)?;
        let ffn_base = linalg::decode_by_name(src, &format!("{prefix}.hc_ffn_base"), limit)?;
        let ffn_norm = linalg::decode_by_name(src, &format!("{prefix}.ffn_norm.weight"), limit)?;
        for position in 0..block {
            let at = position * hc_width;
            // residual = x, then the attention output is mixed back (model.py:986).
            mhc::hc_post(
                config,
                &attn_out[position * dim..(position + 1) * dim],
                &x_hc[at..at + hc_width],
                &attn_mixes[position].post,
                &attn_mixes[position].comb,
                &mut stream,
            )?;
            if let Some(buffers) = captured.as_mut() {
                let at = position * hc_width;
                buffers.post_attention[at..at + hc_width].copy_from_slice(&stream);
            }
            mhc::hc_mixes(
                config,
                &stream,
                &ffn_fn,
                &ffn_scale,
                &ffn_base,
                &mut ffn_mixes,
            )?;
            if let Some(buffers) = captured.as_mut() {
                let at = position * hc;
                buffers.ffn_pre[at..at + hc].copy_from_slice(&ffn_mixes.pre);
                buffers.ffn_post[at..at + hc].copy_from_slice(&ffn_mixes.post);
                buffers.ffn_comb[at..at + hc * hc].copy_from_slice(&ffn_mixes.comb);
            }
            // The ffn reads THIS block's attention coefficient, not its own (model.py:990).
            mhc::hc_pre(config, &attn_mixes[position].pre, &stream, &mut collapsed)?;
            linalg::rms_norm(&mut collapsed, &ffn_norm, dim, eps)?;
            if let Some(buffers) = captured.as_mut() {
                let at = position * dim;
                buffers.pre_ffn[at..at + dim].copy_from_slice(&collapsed);
            }
            moe::moe_forward(
                config,
                &prefix,
                &collapsed,
                src,
                limit,
                &mut scratch,
                &mut ffn_out,
            )?;
            if let Some(buffers) = captured.as_mut() {
                let at = position * dim;
                buffers.ffn_out[at..at + dim].copy_from_slice(&ffn_out);
            }
            mhc::hc_post(
                config,
                &ffn_out,
                &stream,
                &ffn_mixes.post,
                &ffn_mixes.comb,
                &mut x_hc[at..at + hc_width],
            )?;
            // ffn_pre is what the next block consumes (model.py:994, 1267).
            pre_mix[position * hc..(position + 1) * hc].copy_from_slice(&ffn_mixes.pre);
        }
        // The stage's output boundaries and its row: the residual out is the stream after the last
        // position, and the row is appended with layer id n_layers + stage
        // (docs/archive/build/phase1-boundary-table.md section 1).
        if let Some(table) = capture.as_deref_mut() {
            let buffers = captured.take().ok_or_else(|| {
                shape_error(
                    "boundary capture",
                    "an MTP stage ran without its boundary buffers".to_owned(),
                )
            })?;
            table.push_row(buffers.into_row(
                config,
                config.n_layers + stage,
                &attn_in,
                &attn_out,
                &x_hc,
            )?)?;
            captured = Some(StageBoundary::new(config, block));
        }
    }

    // forward_head (model.py:1137-1156): x = hc_pre(x, pre_mix) with the last stage's ffn_pre, the
    // head sees norm(x) while the confidence head sees x itself (model.py:1144-1145, 1155).
    let last = format!("mtp.{}", n_stages - 1);
    let last_norm = linalg::decode_by_name(src, &format!("{last}.norm.weight"), limit)?;
    let head_weight = linalg::decode_by_name(src, "head.weight", limit)?;
    if head_weight.len() != vocab * dim {
        return Err(shape_error(
            "dspark head",
            format!(
                "the LM head is {vocab} x {dim} = {}, got {}",
                vocab * dim,
                head_weight.len()
            ),
        ));
    }
    let mut hidden = vec![0.0f32; block * dim];
    for position in 0..block {
        let at = position * hc_width;
        mhc::hc_pre(
            config,
            &pre_mix[position * hc..(position + 1) * hc],
            &x_hc[at..at + hc_width],
            &mut hidden[position * dim..(position + 1) * dim],
        )?;
    }
    let mut normed = hidden.clone();
    linalg::rms_norm(&mut normed, &last_norm, dim, eps)?;
    let mut logits = vec![0.0f32; block * vocab];
    linalg::matmul(&mut logits, &normed, &head_weight, vocab, dim, block)?;

    // The Markov bias and the drafts (model.py:1148-1154). Both markov tables and the confidence
    // projection are read once here rather than through [markov_head] and [confidence] per position:
    // those decode their own weights, and decoding a [vocab, rank] table inside the position loop is
    // the per-position allocation this engine's rules forbid. The arithmetic is the same projection
    // those two functions apply.
    let markov_embed_table =
        linalg::decode_by_name(src, &format!("{last}.markov_head.embed.weight"), limit)?;
    let markov_head_weight =
        linalg::decode_by_name(src, &format!("{last}.markov_head.head.weight"), limit)?;
    let confidence_weight =
        linalg::decode_by_name(src, &format!("{last}.confidence_head.proj.weight"), limit)?;
    if markov_embed_table.len() != vocab * rank || markov_head_weight.len() != vocab * rank {
        return Err(shape_error(
            "markov head",
            format!(
                "the markov tables are {vocab} x {rank} = {}, got {} and {}",
                vocab * rank,
                markov_embed_table.len(),
                markov_head_weight.len()
            ),
        ));
    }
    if confidence_weight.len() != dim + rank {
        return Err(shape_error(
            "confidence head",
            format!(
                "the projection is [1, {}] = {}, got {}",
                dim + rank,
                dim + rank,
                confidence_weight.len()
            ),
        ));
    }
    let mut bias = vec![0.0f32; vocab];
    let mut combined = vec![0.0f32; dim + rank];
    let mut markov_embeds = vec![0.0f32; block * rank];
    let mut scores = vec![0.0f32; block];
    for position in 0..block {
        let token = ids[position] as usize;
        if token >= vocab {
            return Err(shape_error(
                "markov head",
                format!("draft token {token} is outside the {vocab}-row markov table"),
            ));
        }
        let at = token * rank;
        let embed = &markov_embed_table[at..at + rank];
        markov_bias(embed, &markov_head_weight, vocab, rank, &mut bias)?;
        for (logit, add) in logits[position * vocab..(position + 1) * vocab]
            .iter_mut()
            .zip(bias.iter())
        {
            *logit += add;
        }
        markov_embeds[position * rank..(position + 1) * rank].copy_from_slice(embed);
        // The position's own token feeds its own bias, and the sampled id becomes the next position's
        // token (model.py:1150-1153). Greedy stands in for the Gumbel-max draw (module docs).
        let sampled = linalg::argmax(&logits[position * vocab..(position + 1) * vocab])
            .ok_or_else(|| shape_error("dspark draft", "the vocabulary is empty".to_owned()))?;
        ids[position + 1] = u32::try_from(sampled).map_err(|_| {
            shape_error(
                "dspark draft",
                format!("the sampled id {sampled} does not fit in u32"),
            )
        })?;
    }
    for position in 0..block {
        let at = position * dim;
        scores[position] = confidence_score(
            &hidden[at..at + dim],
            &markov_embeds[position * rank..(position + 1) * rank],
            &confidence_weight,
            dim,
            rank,
            &mut combined,
        )?;
    }
    Ok(DsparkOutput {
        draft: ids,
        confidence: scores,
        logits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::dtype::DType;
    use crate::linalg::REFERENCE_TENSOR_LIMIT;
    use crate::quant;
    use crate::synthetic::SyntheticWeights;
    use crate::tensor::{TensorPlan, TensorSpec};
    use crate::weights::WeightError;

    /// The element limit the reference pass uses by default (linalg::REFERENCE_TENSOR_LIMIT).
    const BIG: usize = REFERENCE_TENSOR_LIMIT;

    /// A weight source whose tensors are written by hand, so every expectation is computed in the
    /// test rather than read back out of the code under test. The tensors here are stored as F32:
    /// the storage dtype of the released file is bf16, and the module's own is fp32, so an fp32 store
    /// keeps the hand-computed values exact.
    struct Hand {
        held: BTreeMap<String, (DType, Vec<usize>, Vec<u8>)>,
    }

    impl Hand {
        fn new() -> Hand {
            Hand {
                held: BTreeMap::new(),
            }
        }

        fn put_f32(&mut self, name: &str, shape: &[usize], values: &[f32]) -> &mut Hand {
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.held
                .insert(name.to_owned(), (DType::F32, shape.to_vec(), bytes));
            self
        }
    }

    impl WeightSource for Hand {
        fn names(&self) -> Vec<String> {
            self.held.keys().cloned().collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.held
                .get(name)
                .map(|(dtype, shape, _)| TensorSpec {
                    name: name.to_owned(),
                    dtype: *dtype,
                    shape: shape.clone(),
                })
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            self.held
                .get(name)
                .map(|(_, _, data)| data.clone())
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }
    }

    /// A plan-wide weight source of finite numbers.
    ///
    /// SyntheticWeights hands back raw bytes, and raw bytes are not numbers: the fp8 formats have NaN
    /// codes and a random exponent decodes to inf or to a 2**-127 scale, so a pass over it is a pass
    /// over NaN. Every tensor here keeps the PLAN's shape - so the element counts and the scale grids
    /// are the planned ones - but is stored as f32 with a small positive pattern, which is what makes
    /// the pipeline test a test of the pipeline rather than of NaN propagation.
    fn finite_hand(plan: &TensorPlan) -> Hand {
        let mut hand = Hand::new();
        for spec in plan.specs() {
            let elements: usize = spec.shape.iter().product();
            let values: Vec<f32> = (0..elements)
                .map(|index| 0.02 + 0.01 * (index % 5) as f32)
                .collect();
            hand.put_f32(&spec.name, &spec.shape, &values);
        }
        hand
    }

    /// The markov bias is head @ embed over the rank axis (model.py:1084-1085).
    #[test]
    fn the_markov_bias_is_the_head_times_the_embedding() {
        let config = ModelConfig {
            dspark_markov_rank: 2,
            vocab_size: 3,
            ..ModelConfig::tiny()
        };
        let mut hand = Hand::new();
        // [vocab, rank] row-major: row 0 = [1, 0], row 1 = [0, 2], row 2 = [3, 4].
        hand.put_f32(
            "mtp.0.markov_head.head.weight",
            &[3, 2],
            &[1.0, 0.0, 0.0, 2.0, 3.0, 4.0],
        );
        let embed = [5.0f32, 7.0];
        let bias = markov_head(&config, "mtp.0", &embed, &hand, BIG).expect("a hand-built bias");
        // 1*5+0*7 = 5; 0*5+2*7 = 14; 3*5+4*7 = 43.
        assert_eq!(bias, vec![5.0, 14.0, 43.0]);

        // A wrong rank or a wrong table is refused by shape, not indexed past its end.
        let error =
            markov_head(&config, "mtp.0", &[1.0], &hand, BIG).expect_err("a short embedding");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("embedding"), "{error}");
        let mut short = Hand::new();
        short.put_f32(
            "mtp.0.markov_head.head.weight",
            &[2, 2],
            &[1.0, 0.0, 0.0, 2.0],
        );
        let error = markov_head(&config, "mtp.0", &embed, &short, BIG).expect_err("a small table");
        assert!(
            error.to_string().contains("markov head is 3 x 2"),
            "{error}"
        );
    }

    /// The confidence head concatenates [hidden ; markov_embed] and projects, with NO activation:
    /// the value is a raw score, so a sum of ones that lands above 1 stays above 1 (model.py:1095-1097).
    #[test]
    fn the_confidence_head_is_a_raw_projection_not_a_probability() {
        let config = ModelConfig::tiny();
        let dim = config.dim;
        let rank = config.dspark_markov_rank;
        let mut proj = vec![1.0f32; dim + rank];
        proj[0] = 2.0;
        proj[dim] = 3.0;
        let mut hand = Hand::new();
        hand.put_f32("mtp.0.confidence_head.proj.weight", &[1, dim + rank], &proj);
        // Distinct values either side of the join, so a swapped concatenation changes the score
        // instead of summing the same numbers in a different order.
        let hidden: Vec<f32> = (0..dim).map(|d| (d + 1) as f32).collect();
        let embed: Vec<f32> = (0..rank).map(|e| 10.0 * (e + 1) as f32).collect();
        let score = confidence(&config, "mtp.0", &hidden, &embed, &hand, BIG)
            .expect("a hand-built confidence");
        // hidden first, then the embedding: 2*1 + sum(d+1 for d in 1..63) + 3*10 + sum(10*(e+1))
        // = 2 + 2079 + 30 + 350 = 2461.
        assert!((score - 2461.0).abs() < 1e-3, "the score is {score}");
        // A sigmoid or a normalisation would put this in [0, 1]; the reference applies neither.
        assert!(
            score > 1.0,
            "the score must not be squashed into a probability"
        );

        let error = confidence(&config, "mtp.0", &hidden[..dim - 1], &embed, &hand, BIG)
            .expect_err("a short hidden state");
        assert!(error.to_string().contains("hidden state"), "{error}");
        let error = confidence(&config, "mtp.0", &hidden, &embed[..rank - 1], &hand, BIG)
            .expect_err("a short embedding");
        assert!(error.to_string().contains("markov embedding"), "{error}");
    }

    /// A prefill position has no draft: the reference returns None there (model.py:1123-1125,
    /// 1274-1282), so the pass refuses before it reads a single weight.
    #[test]
    fn a_prefill_position_has_no_draft() {
        let config = ModelConfig::tiny();
        let empty = Hand::new();
        let tokens = [1u32];
        let taps = vec![0.0f32; tokens.len() * config.dspark_target_layer_ids.len() * config.dim];
        let mut stages = vec![AttnState::for_config(&config)];
        let error = dspark_forward(&config, &tokens, 0, &taps, 9, &mut stages, &empty, BIG)
            .expect_err("position 0 is a prefill");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("start_pos 0"), "{error}");

        // The stage count is checked too, and also before any read: a DSpark pass has one state per
        // MTP stage (model.py:1277-1279).
        let mut none: Vec<AttnState> = Vec::new();
        let error = dspark_forward(&config, &tokens, 5, &taps, 9, &mut none, &empty, BIG)
            .expect_err("no stage states");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("attention states"), "{error}");
    }

    /// The whole draft pass over the tiny configuration: shapes, the anchor, the stage count, the
    /// greedy drafts, finite values, and the same answer twice.
    ///
    /// The weights are [finite_hand] over the plan rather than [SyntheticWeights]: see
    /// [the_byte_source_is_not_a_numeric_weight_source]. attention.rs (task-20) and moe.rs (task-19)
    /// are implemented now, so this test runs the real stages and no longer has a SKIP path.
    #[test]
    fn the_draft_pass_is_a_pipeline_of_the_configured_stages() {
        let config = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&config);
        let source = finite_hand(&plan);
        let tokens = [1u32, 2, 3];
        let taps = vec![0.01f32; tokens.len() * config.dspark_target_layer_ids.len() * config.dim];
        let run = |start_pos: usize| {
            let mut stages: Vec<AttnState> = (0..config.n_mtp_layers)
                .map(|_| AttnState::for_config(&config))
                .collect();
            dspark_forward(
                &config,
                &tokens,
                start_pos,
                &taps,
                9,
                &mut stages,
                &source,
                BIG,
            )
        };
        let first = run(7).expect("the draft pass runs");
        assert_eq!(first.draft.len(), config.dspark_block_size + 1);
        assert_eq!(first.draft[0], 9, "the anchor is the token handed in");
        assert_eq!(first.confidence.len(), config.dspark_block_size);
        assert_eq!(
            first.logits.len(),
            config.dspark_block_size * config.vocab_size
        );
        assert!(
            first
                .draft
                .iter()
                .all(|&id| (id as usize) < config.vocab_size),
            "every drafted id is a vocabulary row: {:?}",
            first.draft
        );
        // Finite weights must give finite numbers: a NaN or an infinity here is a defect in one of
        // the stages, not a property of the input.
        assert!(
            first.confidence.iter().all(|score| score.is_finite()),
            "confidence scores must be finite: {:?}",
            first.confidence
        );
        assert!(
            first.logits.iter().all(|value| value.is_finite()),
            "logits must be finite"
        );
        let second = run(7).expect("the pass runs twice");
        assert_eq!(first, second);
    }

    /// Why the pipeline test above does not use the plan-wide byte source.
    ///
    /// A float dtype's raw bytes are not all numbers: E4M3 has NaN codes and a random exponent decodes
    /// to inf or to a 2**-127 scale. This pins that, so the choice of [finite_hand] is visible in the
    /// code and not only in a report.
    #[test]
    fn the_byte_source_is_not_a_numeric_weight_source() {
        let config = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&config);
        let source = SyntheticWeights::from_plan(&plan, 5);
        let name = "mtp.0.main_proj.weight";
        let spec = source.spec(name).expect("a planned tensor");
        let elements: usize = spec.shape.iter().product();
        let values = quant::decode(spec.dtype, &source.read(name).expect("bytes"), elements)
            .expect("the planned dtype decodes");
        assert!(
            values.iter().any(|value| !value.is_finite()),
            "the raw bytes of a float tensor are not all finite numbers"
        );
    }

    /// The released configuration is refused by the element limit before its 78,643,200-element
    /// main_proj is materialised (section 2.3, section 5).
    #[test]
    fn the_released_draft_pass_is_refused_by_the_element_limit() {
        let config = ModelConfig::v41_flash_cached();
        let plan = TensorPlan::language_model(config);
        let source = SyntheticWeights::from_plan(&plan, 13);
        let tokens = [1u32, 2, 3];
        let taps = vec![0.0f32; tokens.len() * config.dspark_target_layer_ids.len() * config.dim];
        let mut stages: Vec<AttnState> = (0..config.n_mtp_layers)
            .map(|_| AttnState::for_config(config))
            .collect();
        let error = dspark_forward(config, &tokens, 8, &taps, 9, &mut stages, &source, BIG)
            .expect_err("a 78-million-element weight is not materialised");
        match error {
            EngineError::TooLarge {
                name,
                elements,
                limit,
            } => {
                assert_eq!(name, "mtp.0.main_proj.weight");
                assert_eq!(elements, 78_643_200);
                assert_eq!(limit, BIG);
            }
            other => panic!("expected a size refusal, got {other}"),
        }
    }
}
