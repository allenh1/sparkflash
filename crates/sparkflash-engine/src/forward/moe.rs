//! The MoE block: the router, the routed experts and the shared expert.
//!
//! `docs/engine/01-forward-order.md` section 3.6 is the specification. `MoE.forward`
//! (model.py:889-904) flattens the chunk to `[n, d]`, routes every position, accumulates each local
//! expert's contribution in ascending expert index, adds the shared expert, and returns. `Gate.forward`
//! (model.py:809-827) and `Expert.forward` (model.py:841-851) say what those pieces are:
//!
//! - `scores = linear(x, gate.weight) / gate_temp`, then `sqrtsoftplus`, `softmax` or `sigmoid`;
//! - the noaux bias is added **only for selection**: `indices = (scores + bias).topk(topk)`, and the
//!   weights come from the *unbiased* scores (model.py:821-823);
//! - `weights /= (sum + 1e-20)` when `norm_topk_prob` and `topk > 1`, then `weights *= route_scale`
//!   (model.py:824-826);
//! - an expert is `silu(clamp(w1 x)) * clamp(w3 x)` scaled by its routing weight and pushed through
//!   `w2` (model.py:845-847).
//!
//! Two details of that order are load bearing and are what the tests pin. First, the *selection*
//! bias never enters the weights, so an expert can be chosen by the bias and still be weighted by
//! its plain score. Second, the accumulation runs over experts in ascending index order, per token,
//! which is the order the reference's bincount loop accumulates in and therefore the order these
//! fp32 sums round in (model.py:895-900; specification section 3.6).
//!
//! Routing counts come from [crate::config::ModelConfig::moe_config], so `layers.{{i}}` reads the
//! backbone's 384/6 and `mtp.{{N}}` the DSpark stage's 128/3 (model.py:142-149). The image-span bias
//! `gate.bias_vl` is used only when a caller passes it: [route] takes it as an argument and
//! [moe_forward] never supplies one, because this reference path is text-only - the same reason the
//! released `DSparkBlock.forward` passes no image mask at all (model.py:1126).
//!
//! This is an `f32` reference over decoded weights: no quantised kernel runs here, so the FP8
//! activation quantisation that section 6 point 2 gives every expert GEMM is not reproduced. Each
//! weight is read through [crate::linalg::decode_by_name] under the CALLER's element limit - the
//! `limit` argument of [moe_forward], which is [crate::forward::InferenceEngine]'s bound on any
//! single tensor a pass materialises - and dropped when its expert is done, so nothing model-sized is
//! retained. [crate::linalg::REFERENCE_TENSOR_LIMIT] is a caller's choice of that bound, not a value
//! this module may substitute.

use std::fmt::Write as _;

use crate::config::{ConfigError, ModelConfig};
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::parallel::{enter_worker, reference_threads};
use crate::weights::WeightSource;

/// The c4b device capability: replaces the routed MoE block with device launches while the host
/// keeps the declared routing seams; c4c also replaces the shared expert and the certified LM head.
/// The default answers `false`/`None`, which runs the host reference unchanged.
pub trait MoeDeviceBackend {
    /// Computes the routed MoE contribution into `out` (and the host shared addition for c4b).
    /// Returning `false` leaves `out` untouched and the host block runs instead.
    #[allow(clippy::too_many_arguments)]
    fn ffn(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = (config, layer, prefix, x, source, limit, out);
        Ok(false)
    }

    /// Adds the dense shared expert to `out` (already holding the routed contribution). Returning
    /// `false` leaves `out` untouched and the host shared-expert seam runs instead.
    #[allow(clippy::too_many_arguments)]
    fn shared_expert(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = (config, layer, prefix, x, source, limit, out);
        Ok(false)
    }

    /// Runs the final RMSNorm on device over the collapsed tail, writing BF16 `[rows][cols]` bytes.
    /// Returning `false` leaves `out` untouched and the host norm runs instead. A device norm must
    /// be bit-identical to the host norm rounded to BF16; the backend's own twin bar is the check.
    #[allow(clippy::too_many_arguments)]
    fn final_norm(
        &mut self,
        config: &ModelConfig,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool, EngineError> {
        let _ = (config, x, source, limit, out);
        Ok(false)
    }

    /// Runs the certified LM head and its in-kernel argmax over the normalized final hidden
    /// `hidden`, writing the four rank logits into `logits` and returning the assembled token.
    /// `None` leaves `logits` untouched and the host head runs instead.
    fn lm_head(
        &mut self,
        config: &ModelConfig,
        hidden: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        logits: &mut [f32],
    ) -> Result<Option<u32>, EngineError> {
        let _ = (config, hidden, source, limit, logits);
        Ok(None)
    }
}

/// The reduction epsilon of model.py:825: the literal `1e-20`, **not** `norm_eps` and not `hc_eps`.
const ROUTE_NORM_EPSILON: f32 = 1e-20;

/// `F.softplus`'s threshold at the reference's default `beta = 1`: PyTorch returns the input above
/// it instead of `log1p(exp(x))` (model.py:816-817). The two agree far below f32 resolution past the
/// threshold, so this is the reference's own stability rule rather than a value the arithmetic
/// notices.
const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// Room for `layers.39.ffn.shared_experts.w2.scale` and its like, so naming a tensor stops
/// allocating once the buffer has been used once.
const NAME_CAPACITY: usize = 64;

/// The per-call working set of [moe_forward].
///
/// The score and selection buffers are sized from the configuration, so the common decode step
/// allocates nothing for them. [Self::routed] and [Self::choices] carry one entry per chunk token
/// and per selection, so a prefill makes them grow once and every later call reuses that capacity.
pub struct MoeScratch {
    /// The unbiased scores of the token being routed, `[n_routed]`.
    scores: Vec<f32>,
    /// The selection [route] produces for one token, in descending selection-signal order.
    selected: Vec<(usize, f32)>,
    /// Every token's selection as `(token, expert, weight)`, sorted into the accumulation order.
    choices: Vec<(usize, usize, f32)>,
    /// The routed accumulation of the chunk, one `dim`-wide row per token.
    routed: Vec<f32>,
    /// One expert's `w2` output, `[dim]`.
    expert_out: Vec<f32>,
    /// The SwiGLU gate branch `w1 x`, `[inter]`.
    gate_branch: Vec<f32>,
    /// The SwiGLU up branch `w3 x`, `[inter]`.
    up_branch: Vec<f32>,
    /// A reusable expert-name buffer, reused for every weight read.
    base: String,
    /// A reusable tensor-name buffer, reused for every weight read.
    name: String,
}

impl MoeScratch {
    /// Zeroed scratch sized from the configuration.
    ///
    /// One scratch serves a backbone block and a DSpark stage, so the expert buffers take the wider
    /// of the two routed counts (384 against 128 in the released configuration).
    #[must_use]
    pub fn for_config(cfg: &ModelConfig) -> Self {
        let widest = cfg.n_routed_experts.max(cfg.dspark_n_routed_experts);
        let widest_topk = cfg.n_activated_experts.max(cfg.dspark_n_activated_experts);
        let inter = cfg.moe_inter_dim * cfg.n_shared_experts.max(1);
        MoeScratch {
            scores: vec![0.0; widest],
            selected: Vec::with_capacity(widest_topk),
            choices: Vec::new(),
            routed: vec![0.0; cfg.dim],
            expert_out: vec![0.0; cfg.dim],
            gate_branch: vec![0.0; inter],
            up_branch: vec![0.0; inter],
            base: String::with_capacity(NAME_CAPACITY),
            name: String::with_capacity(NAME_CAPACITY),
        }
    }

    /// Reads one expert body's three weights under the caller's element limit, each already
    /// multiplied by its scale grid.
    fn read_expert(
        &mut self,
        cfg: &ModelConfig,
        src: &dyn WeightSource,
        prefix: &str,
        body: ExpertBody,
        limit: usize,
    ) -> Result<ExpertWeights, EngineError> {
        let dim = cfg.dim;
        let (inter, grid) = match body {
            ExpertBody::Routed(expert) => {
                self.base.clear();
                write!(self.base, "{prefix}.ffn.experts.{expert}")
                    .expect("a String cannot fail to be written to");
                (cfg.moe_inter_dim, ScaleGrid::PerRow32)
            }
            ExpertBody::Shared => {
                self.base.clear();
                write!(self.base, "{prefix}.ffn.shared_experts")
                    .expect("a String cannot fail to be written to");
                (
                    cfg.moe_inter_dim * cfg.n_shared_experts.max(1),
                    ScaleGrid::Block32,
                )
            }
        };
        if inter == 0 || inter > self.gate_branch.len() {
            return Err(shape_error(
                &self.base,
                format!(
                    "an intermediate width of {inter} does not fit a scratch of {}",
                    self.gate_branch.len()
                ),
            ));
        }
        let w1 = read_weight(
            src,
            &mut self.name,
            &self.base,
            "w1",
            &WeightGeometry {
                rows: inter,
                cols: dim,
                grid,
            },
            limit,
        )?;
        let w3 = read_weight(
            src,
            &mut self.name,
            &self.base,
            "w3",
            &WeightGeometry {
                rows: inter,
                cols: dim,
                grid,
            },
            limit,
        )?;
        let w2 = read_weight(
            src,
            &mut self.name,
            &self.base,
            "w2",
            &WeightGeometry {
                rows: dim,
                cols: inter,
                grid,
            },
            limit,
        )?;
        Ok(ExpertWeights { w1, w3, w2, inter })
    }
}

/// Which expert body [MoeScratch::read_expert] reads.
enum ExpertBody {
    /// A routed expert of the block: `{{prefix}}.ffn.experts.{{e}}`, FP4 with one scale per 32
    /// elements along K (kernel.py:582).
    Routed(usize),
    /// The shared expert: `{{prefix}}.ffn.shared_experts`, FP8 with 32x32 block scales
    /// (model.py:836, 887).
    Shared,
}

/// The three weights of one expert body, decoded and scaled, with its intermediate width.
struct ExpertWeights {
    /// The gate projection `w1`, `[inter, dim]`.
    w1: Vec<f32>,
    /// The up projection `w3`, `[inter, dim]`.
    w3: Vec<f32>,
    /// The down projection `w2`, `[dim, inter]`.
    w2: Vec<f32>,
    /// The intermediate width `inter` these three share.
    inter: usize,
}

/// Runs one expert body on one token, writing its `dim`-wide output to `out`.
///
/// This is exactly the arithmetic [MoeScratch::add_expert] adds: `gate = w1 x`, `up = w3 x`, the
/// clamps, the routing weight, then `w2`. It is a separate function so the serial and the
/// expert-parallel paths share one definition - the parallel path computes a contribution into a
/// per-choice row and the serial path adds it immediately, and both must be the same bits.
fn expert_body(
    cfg: &ModelConfig,
    row: &[f32],
    expert: &ExpertWeights,
    weight: f32,
    gate_branch: &mut [f32],
    up_branch: &mut [f32],
    out: &mut [f32],
) -> Result<(), EngineError> {
    let dim = cfg.dim;
    let inter = expert.inter;
    if inter > gate_branch.len() || inter > up_branch.len() {
        return Err(shape_error(
            "an expert",
            format!("an intermediate width of {inter} does not fit the scratch"),
        ));
    }
    if out.len() < dim {
        return Err(shape_error(
            "an expert",
            format!(
                "an expert body writes {dim} values into {} slots",
                out.len()
            ),
        ));
    }
    linalg::matmul(&mut gate_branch[..inter], row, &expert.w1, inter, dim, 1)?;
    linalg::matmul(&mut up_branch[..inter], row, &expert.w3, inter, dim, 1)?;
    swiglu(cfg, &mut gate_branch[..inter], &up_branch[..inter])?;
    for value in &mut gate_branch[..inter] {
        *value *= weight;
    }
    linalg::matmul(
        &mut out[..dim],
        &gate_branch[..inter],
        &expert.w2,
        dim,
        inter,
        1,
    )
}

/// The geometry one weight is expected to have, and how its scale grid maps onto it.
struct WeightGeometry {
    /// Logical rows.
    rows: usize,
    /// Logical columns.
    cols: usize,
    /// The grid of its `.scale` tensor.
    grid: ScaleGrid,
}

/// One token's routing work: the unbiased scores and the running selection.
///
/// It owns neither buffer, so a caller with a scratch ([moe_forward]) routes every token of a chunk
/// without allocating. [route] itself allocates exactly the `[n_routed]` score buffer that its
/// signature has nowhere else to put.
struct Routing<'a> {
    /// Unbiased scores, at least `n_routed` long.
    scores: &'a mut [f32],
    /// The selection, in descending selection-signal order.
    selected: &'a mut Vec<(usize, f32)>,
}

impl Routing<'_> {
    /// `Gate.forward` for one token (model.py:809-827), writing this routing's selection.
    ///
    /// `bias_vl` is the image-span bias: the caller passes `Some` for a token in the image span,
    /// where the reference uses `bias_vl` instead of `bias` (model.py:818-820), and `None` for a
    /// text token.
    fn run(
        &mut self,
        cfg: &ModelConfig,
        prefix: &str,
        x: &[f32],
        gate: &[f32],
        bias: &[f32],
        bias_vl: Option<&[f32]>,
    ) -> Result<(), EngineError> {
        let (n_routed, topk) = counts(cfg, prefix)?;
        let dim = cfg.dim;
        if x.len() != dim {
            return Err(shape_error(
                "the router",
                format!("one token of {dim} values, got {}", x.len()),
            ));
        }
        if self.scores.len() < n_routed {
            return Err(shape_error(
                "the router",
                format!(
                    "the scratch holds {} scores, {prefix} routes over {n_routed} experts",
                    self.scores.len()
                ),
            ));
        }
        if gate.len() != n_routed * dim {
            return Err(shape_error(
                &format!("{prefix}.ffn.gate.weight"),
                format!(
                    "the configuration routes {n_routed} experts over {dim} values, so the weight is {} values, got {}",
                    n_routed * dim,
                    gate.len()
                ),
            ));
        }
        if bias.len() != n_routed {
            return Err(shape_error(
                &format!("{prefix}.ffn.gate.bias"),
                format!(
                    "{n_routed} experts need {n_routed} biases, got {}",
                    bias.len()
                ),
            ));
        }
        if let Some(bias_vl) = bias_vl.filter(|bias_vl| bias_vl.len() != n_routed) {
            return Err(shape_error(
                &format!("{prefix}.ffn.gate.bias_vl"),
                format!(
                    "{n_routed} experts need {n_routed} biases, got {}",
                    bias_vl.len()
                ),
            ));
        }
        let temperature = cfg.gate_temp as f32;
        // The reference divides the scores by this, so anything that is not a positive, finite
        // temperature is a configuration the router cannot run: a negative one would turn every
        // sqrtsoftplus into a NaN instead of a score.
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(EngineError::Config(ConfigError::Invalid {
                field: "gate_temp",
                detail: format!(
                    "the router divides the scores by it, and {temperature} is not usable"
                ),
            }));
        }

        // scores = linear(x, gate.weight) / gate_temp, then the configured score function
        // (model.py:811-817).
        let scores = &mut self.scores[..n_routed];
        linalg::matmul(scores, x, gate, n_routed, dim, 1)?;
        for value in scores.iter_mut() {
            *value /= temperature;
        }
        match cfg.score_func.as_str() {
            "sqrtsoftplus" => {
                for value in scores.iter_mut() {
                    *value = softplus(*value).sqrt();
                }
            }
            "sigmoid" => {
                for value in scores.iter_mut() {
                    *value = linalg::sigmoid(*value);
                }
            }
            "softmax" => linalg::softmax(scores),
            other => {
                return Err(EngineError::Config(ConfigError::Invalid {
                    field: "score_func",
                    detail: format!(
                        "the reference scores with sqrtsoftplus, sigmoid or softmax, not {other:?}"
                    ),
                }));
            }
        }

        // The bias only decides *which* experts: the weights come from the unbiased scores
        // (model.py:818-823). Ties go to the lower index, so the selection is a function of the
        // scores rather than of the order the candidates happened to be visited in.
        let selection_bias = bias_vl.unwrap_or(bias);
        self.selected.clear();
        for _ in 0..topk {
            let mut best: Option<usize> = None;
            let mut best_signal = f32::NEG_INFINITY;
            for expert in 0..n_routed {
                if self.selected.iter().any(|(taken, _)| *taken == expert) {
                    continue;
                }
                let signal = scores[expert] + selection_bias[expert];
                if best.is_none() || signal > best_signal {
                    best = Some(expert);
                    best_signal = signal;
                }
            }
            match best {
                Some(expert) => self.selected.push((expert, scores[expert])),
                None => break,
            }
        }

        // weights /= (sum + 1e-20) when norm_topk_prob and topk > 1, then weights *= route_scale
        // (model.py:824-826).
        if cfg.norm_topk_prob && topk > 1 {
            let sum: f32 = self.selected.iter().map(|(_, weight)| *weight).sum();
            let denominator = sum + ROUTE_NORM_EPSILON;
            for (_, weight) in self.selected.iter_mut() {
                *weight /= denominator;
            }
        }
        let route_scale = cfg.route_scale as f32;
        for (_, weight) in self.selected.iter_mut() {
            *weight *= route_scale;
        }
        Ok(())
    }
}

/// `Gate.forward` for one token (model.py:809-827): writes `selected` as `(expert, weight)` pairs in
/// descending selection-signal order, each weight already normalised and scaled by `route_scale`.
///
/// `gate` is the `[n_routed, dim]` gate weight, already decoded: it is BF16 in the checkpoint and
/// unquantised, so it has no `.scale` tensor. `bias` is the noaux selection bias and `bias_vl` the
/// image-span one, used *instead of* `bias` when it is `Some` (model.py:818-820): the caller owns
/// the image mask, so it passes `Some(bias_vl)` exactly for a token in the image span.
///
/// # Errors
///
/// [EngineError::Shape] when the configuration and the slices disagree, and [EngineError::Config]
/// for a `score_func` or `gate_temp` the reference cannot run.
pub fn route(
    cfg: &ModelConfig,
    prefix: &str,
    x: &[f32],
    gate: &[f32],
    bias: &[f32],
    bias_vl: Option<&[f32]>,
    selected: &mut Vec<(usize, f32)>,
) -> Result<(), EngineError> {
    let (n_routed, _) = counts(cfg, prefix)?;
    let mut scores = vec![0.0f32; n_routed];
    let mut routing = Routing {
        scores: &mut scores,
        selected,
    };
    routing.run(cfg, prefix, x, gate, bias, bias_vl)
}

/// `MoE.forward` (model.py:889-904) for a chunk of `x.len() / dim` tokens.
///
/// Every position routes, the selections are then accumulated in ascending expert index - each
/// expert's weights are decoded once and applied to every token that chose it - and the shared
/// expert is added to every token afterwards (model.py:903). Each output row is the routed sum plus
/// the shared expert, in `f32`.
///
/// `bias_vl` is not read here: this path is text-only, so the text bias is the selection bias. A
/// caller with an image mask calls [route] per token itself.
///
/// `limit` is the caller's element bound on any single tensor this pass materialises, exactly as
/// [crate::linalg::decode_by_name] defines it: every weight this block reads - the gate, the
/// selection bias and every expert's `w1`, `w3` and `w2`, with their scale grids - is decoded under
/// it, so an [crate::forward::InferenceEngine] built with a small limit refuses them all.
///
/// # Errors
///
/// [EngineError::Shape] when the configuration, the slices and the scratch disagree,
/// [EngineError::TooLarge] when a tensor is above `limit`, and any other error from a weight read or
/// decode.
pub fn routed_forward(
    cfg: &ModelConfig,
    prefix: &str,
    x: &[f32],
    src: &dyn WeightSource,
    limit: usize,
    scratch: &mut MoeScratch,
    out: &mut [f32],
) -> Result<(), EngineError> {
    let (n_routed, _) = counts(cfg, prefix)?;
    let dim = cfg.dim;
    if dim == 0 || cfg.moe_inter_dim == 0 {
        return Err(shape_error(
            "the MoE block",
            format!(
                "dim {} and moe_inter_dim {} must both be positive",
                cfg.dim, cfg.moe_inter_dim
            ),
        ));
    }
    if !x.len().is_multiple_of(dim) {
        return Err(shape_error(
            "the MoE block",
            format!(
                "{} values are not a whole number of {dim}-wide tokens",
                x.len()
            ),
        ));
    }
    let tokens = x.len() / dim;
    if out.len() != x.len() {
        return Err(shape_error(
            "the MoE block",
            format!(
                "{tokens} tokens need {} outputs, got {}",
                x.len(),
                out.len()
            ),
        ));
    }
    if scratch.scores.len() < n_routed {
        return Err(shape_error(
            "the MoE block",
            format!(
                "the scratch holds {} scores, {prefix} routes over {n_routed} experts",
                scratch.scores.len()
            ),
        ));
    }

    // The gate weight (BF16, unquantised) and the selection bias (F32), read once for the chunk
    // (model.py:891).
    scratch.name.clear();
    write!(scratch.name, "{prefix}.ffn.gate.weight")
        .expect("a String cannot fail to be written to");
    let gate = linalg::decode_by_name(src, &scratch.name, limit)?;
    if gate.len() != n_routed * dim {
        return Err(shape_error(
            &scratch.name,
            format!(
                "the configuration routes {n_routed} experts over {dim} values, so the weight is {} values, got {}",
                n_routed * dim,
                gate.len()
            ),
        ));
    }
    scratch.name.clear();
    write!(scratch.name, "{prefix}.ffn.gate.bias").expect("a String cannot fail to be written to");
    let bias = linalg::decode_by_name(src, &scratch.name, limit)?;
    if bias.len() != n_routed {
        return Err(shape_error(
            &scratch.name,
            format!(
                "{n_routed} experts need {n_routed} biases, got {}",
                bias.len()
            ),
        ));
    }

    // Every position routes, in token order (model.py:891-894).
    scratch.routed.resize(tokens * dim, 0.0);
    scratch.routed.fill(0.0);
    scratch.choices.clear();
    {
        let mut routing = Routing {
            scores: &mut scratch.scores[..n_routed],
            selected: &mut scratch.selected,
        };
        for token in 0..tokens {
            routing.run(
                cfg,
                prefix,
                &x[token * dim..(token + 1) * dim],
                &gate,
                &bias,
                None,
            )?;
            for &(expert, weight) in routing.selected.iter() {
                scratch.choices.push((token, expert, weight));
            }
        }
    }

    // Expert-major: ascending expert index, tokens in order inside an expert. This is the reference's
    // bincount loop (model.py:895-900) and the order the routed fp32 sums round in.
    scratch
        .choices
        .sort_unstable_by_key(|(token, expert, _)| (*expert, *token));
    let groups = expert_groups(&scratch.choices);
    let mut contrib = vec![0.0f32; scratch.choices.len() * dim];
    let pass = RoutedPass {
        cfg,
        prefix,
        x,
        src,
        limit,
        choices: &scratch.choices,
    };
    let workers = reference_threads().min(groups.len()).max(1);
    if workers <= 1 {
        compute_routed(&pass, &groups, &mut contrib)?;
    } else {
        // One worker per contiguous run of experts. Every worker computes its choices' expert
        // outputs into its own rows of `contrib`; the values are the serial values because
        // [expert_body] is the serial body, and the rows are disjoint so scheduling cannot reach
        // them.
        let chunk = groups.len().div_ceil(workers);
        let mut rest = &mut contrib[..];
        let mut blocks: Vec<RoutedBlock<'_>> = Vec::with_capacity(workers);
        for group_chunk in groups.chunks(chunk) {
            let first = group_chunk.first().map_or(0, |group| group.1);
            let last = group_chunk.last().map_or(first, |group| group.2);
            let (head, tail) = rest.split_at_mut((last - first) * dim);
            blocks.push((head, group_chunk));
            rest = tail;
        }
        let pass = &pass;
        std::thread::scope(|scope| -> Result<(), EngineError> {
            let mut handles = Vec::with_capacity(blocks.len());
            for (block, group_chunk) in blocks {
                handles.push(scope.spawn(move || -> Result<(), EngineError> {
                    enter_worker();
                    compute_routed(pass, group_chunk, block)
                }));
            }
            for handle in handles {
                handle.join().expect("a reference MoE worker panicked")?;
            }
            Ok(())
        })?;
    }

    // The order the routed fp32 sums round in: one `+=` per choice, experts ascending. The
    // contributions were computed by the same body either way, so this is the same arithmetic the
    // serial loop does, not a reassociation of it.
    for (&(token, _, _), contribution) in scratch.choices.iter().zip(contrib.chunks(dim)) {
        let row = &mut scratch.routed[token * dim..(token + 1) * dim];
        for (accumulated, value) in row.iter_mut().zip(contribution.iter()) {
            *accumulated += *value;
        }
    }

    out.copy_from_slice(&scratch.routed[..tokens * dim]);
    Ok(())
}

/// The reference MoE block: the routed sum followed by the dense shared expert (model.py:903).
///
/// # Errors
/// The shape and weight-read errors of [routed_forward] and [add_shared_expert].
pub fn moe_forward(
    cfg: &ModelConfig,
    prefix: &str,
    x: &[f32],
    src: &dyn WeightSource,
    limit: usize,
    scratch: &mut MoeScratch,
    out: &mut [f32],
) -> Result<(), EngineError> {
    routed_forward(cfg, prefix, x, src, limit, scratch, out)?;
    add_shared_expert(cfg, prefix, x, src, limit, scratch, out)
}

/// Adds the dense shared expert to an already-computed routed contribution (the c4b device path).
///
/// The runtime order is the reference's: the shared expert runs on the same input after the routed
/// sum, and each token's row gets exactly one `+=` of [expert_body]'s output. The device backend
/// computes the routed contribution on device and calls this host seam for the shared expert.
pub fn add_shared_expert(
    cfg: &ModelConfig,
    prefix: &str,
    x: &[f32],
    src: &dyn WeightSource,
    limit: usize,
    scratch: &mut MoeScratch,
    out: &mut [f32],
) -> Result<(), EngineError> {
    let dim = cfg.dim;
    if dim == 0 || !x.len().is_multiple_of(dim) || out.len() != x.len() {
        return Err(shape_error(
            "the shared-expert seam",
            format!(
                "dim {dim}, {} input values and {} output values do not describe whole tokens",
                x.len(),
                out.len()
            ),
        ));
    }
    let tokens = x.len() / dim;
    let shared = scratch.read_expert(cfg, src, prefix, ExpertBody::Shared, limit)?;
    let shared_workers = reference_threads().min(tokens).max(1);
    if shared_workers <= 1 {
        for token in 0..tokens {
            expert_body(
                cfg,
                &x[token * dim..(token + 1) * dim],
                &shared,
                1.0,
                &mut scratch.gate_branch,
                &mut scratch.up_branch,
                &mut scratch.expert_out,
            )?;
            let row = &mut out[token * dim..(token + 1) * dim];
            for (accumulated, value) in row.iter_mut().zip(scratch.expert_out.iter()) {
                *accumulated += *value;
            }
        }
        return Ok(());
    }
    let rows_per_chunk = tokens.div_ceil(shared_workers);
    let mut rest = out;
    let mut row_blocks: Vec<&mut [f32]> = Vec::with_capacity(shared_workers);
    while !rest.is_empty() {
        let take = (rows_per_chunk * dim).min(rest.len());
        let (head, tail) = rest.split_at_mut(take);
        row_blocks.push(head);
        rest = tail;
    }
    std::thread::scope(|scope| -> Result<(), EngineError> {
        let mut handles = Vec::with_capacity(row_blocks.len());
        for (block_index, rows) in row_blocks.into_iter().enumerate() {
            let first_token = block_index * rows_per_chunk;
            let shared = &shared;
            handles.push(scope.spawn(move || -> Result<(), EngineError> {
                enter_worker();
                let mut local = MoeScratch::for_config(cfg);
                let mut buffer = vec![0.0f32; cfg.dim];
                for (local_token, row) in rows.chunks_mut(dim).enumerate() {
                    let token = first_token + local_token;
                    expert_body(
                        cfg,
                        &x[token * dim..(token + 1) * dim],
                        shared,
                        1.0,
                        &mut local.gate_branch,
                        &mut local.up_branch,
                        &mut buffer,
                    )?;
                    for (accumulated, value) in row.iter_mut().zip(buffer.iter()) {
                        *accumulated += *value;
                    }
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .expect("a reference shared-expert worker panicked")?;
        }
        Ok(())
    })?;
    Ok(())
}

/// One contiguous run of a routed expert in the expert-major choice order:
/// `(expert, first_choice, end_choice)`.
type ChoiceGroup = (usize, usize, usize);

/// One worker's slice of the routed contribution buffer and the groups that fill it.
type RoutedBlock<'a> = (&'a mut [f32], &'a [ChoiceGroup]);

/// The per-chunk inputs [compute_routed] needs that do not change between workers.
struct RoutedPass<'a> {
    cfg: &'a ModelConfig,
    prefix: &'a str,
    x: &'a [f32],
    src: &'a dyn WeightSource,
    limit: usize,
    choices: &'a [(usize, usize, f32)],
}

/// The contiguous runs of one expert in the expert-major choice order.
///
/// Each run is `(expert, first_choice, end_choice)`; runs are ascending by expert and together
/// cover every choice exactly once.
fn expert_groups(choices: &[(usize, usize, f32)]) -> Vec<ChoiceGroup> {
    let mut groups = Vec::new();
    let mut start = 0;
    while start < choices.len() {
        let expert = choices[start].1;
        let mut end = start + 1;
        while end < choices.len() && choices[end].1 == expert {
            end += 1;
        }
        groups.push((expert, start, end));
        start = end;
    }
    groups
}

/// Computes the expert outputs for one run of [expert_groups] into `contrib`.
///
/// `contrib` is one `dim`-wide row per choice in `groups`, in choice order; the first group's
/// choice index is the base. This is the body that fills the routed contribution buffer, whether
/// the run is a whole layer (serial path) or one worker's slice of it (parallel path).
fn compute_routed(
    pass: &RoutedPass<'_>,
    groups: &[ChoiceGroup],
    contrib: &mut [f32],
) -> Result<(), EngineError> {
    let dim = pass.cfg.dim;
    let base = groups.first().map_or(0, |group| group.1);
    let mut local = MoeScratch::for_config(pass.cfg);
    for &(expert, start, end) in groups {
        let weights = local.read_expert(
            pass.cfg,
            pass.src,
            pass.prefix,
            ExpertBody::Routed(expert),
            pass.limit,
        )?;
        for (local_choice, &(token, _, weight)) in pass.choices[start..end].iter().enumerate() {
            expert_body(
                pass.cfg,
                &pass.x[token * dim..(token + 1) * dim],
                &weights,
                weight,
                &mut local.gate_branch,
                &mut local.up_branch,
                &mut local.expert_out,
            )?;
            let offset = (start + local_choice - base) * dim;
            contrib[offset..offset + dim].copy_from_slice(&local.expert_out[..dim]);
        }
    }
    Ok(())
}

/// The routed and activated expert counts of the block a prefix names.
fn counts(cfg: &ModelConfig, prefix: &str) -> Result<(usize, usize), EngineError> {
    let layer = layer_index(cfg, prefix)?;
    Ok(cfg.moe_config(layer))
}

/// The plan index a block prefix names: `layers.{{i}}` is backbone block `i` and `mtp.{{N}}` is
/// DSpark stage `N`, which [crate::config::ModelConfig::moe_config] numbers `n_layers + N`.
fn layer_index(cfg: &ModelConfig, prefix: &str) -> Result<usize, EngineError> {
    if let Some(rest) = prefix.strip_prefix("layers.") {
        return match rest.parse::<usize>() {
            Ok(layer) if layer < cfg.n_layers => Ok(layer),
            _ => Err(shape_error(
                "a block prefix",
                format!(
                    "{prefix:?} is not one of the {} backbone layers",
                    cfg.n_layers
                ),
            )),
        };
    }
    if let Some(rest) = prefix.strip_prefix("mtp.") {
        return match rest.parse::<usize>() {
            Ok(stage) if stage < cfg.n_mtp_layers => Ok(cfg.n_layers + stage),
            _ => Err(shape_error(
                "a block prefix",
                format!(
                    "{prefix:?} is not one of the {} DSpark stages",
                    cfg.n_mtp_layers
                ),
            )),
        };
    }
    Err(shape_error(
        "a block prefix",
        format!("{prefix:?} is neither layers.{{i}} nor mtp.{{N}}"),
    ))
}

/// `F.softplus(x)` with `beta = 1` and PyTorch's threshold of 20 (model.py:816-817).
fn softplus(x: f32) -> f32 {
    if x > SOFTPLUS_THRESHOLD {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `silu(clamp(gate, max = swiglu_limit)) * clamp(up, +-swiglu_limit)`, in place in `gate_branch`.
///
/// The gate branch is clamped on one side only and the up branch on both, which the specification
/// spells out from model.py:845-847 and which the tests pin: a gate of `-448` is *not* clamped up to
/// `-10`.
///
/// Both clamps sit inside the reference's `if self.swiglu_limit > 0` (model.py:845), so a
/// non-positive limit means NO clamp and the branches pass through unchanged: with `0` the module
/// used to clamp the up branch to zero and zero the whole expert, and the reference does not. A NaN
/// limit is not greater than zero either, so it takes the same branch the reference gives it.
///
/// # Errors
///
/// [EngineError::Shape] when the two branches differ in length.
fn swiglu(
    cfg: &ModelConfig,
    gate_branch: &mut [f32],
    up_branch: &[f32],
) -> Result<(), EngineError> {
    if gate_branch.len() != up_branch.len() {
        return Err(shape_error(
            "the SwiGLU",
            format!(
                "{} gate values against {} up values",
                gate_branch.len(),
                up_branch.len()
            ),
        ));
    }
    let limit = cfg.swiglu_limit as f32;
    if limit > 0.0 {
        for (gate, up) in gate_branch.iter_mut().zip(up_branch.iter()) {
            let clamped_gate = gate.min(limit);
            let clamped_up = up.clamp(-limit, limit);
            *gate = linalg::silu(clamped_gate) * clamped_up;
        }
    } else {
        for (gate, up) in gate_branch.iter_mut().zip(up_branch.iter()) {
            *gate = linalg::silu(*gate) * *up;
        }
    }
    Ok(())
}

/// Reads `{{base}}.{{leaf}}.weight` and `{{base}}.{{leaf}}.scale`, decodes both, and multiplies the
/// weight by its grid ([crate::linalg::apply_scales]).
///
/// # Errors
///
/// [EngineError::Shape] when the decoded weight is not the declared geometry, plus whatever
/// [crate::linalg::decode_by_name] and [crate::linalg::apply_scales] report.
fn read_weight(
    src: &dyn WeightSource,
    name: &mut String,
    base: &str,
    leaf: &str,
    geometry: &WeightGeometry,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    name.clear();
    write!(name, "{base}.{leaf}.weight").expect("a String cannot fail to be written to");
    let mut weight = linalg::decode_by_name(src, name, limit)?;
    if weight.len() != geometry.rows * geometry.cols {
        return Err(shape_error(
            name,
            format!(
                "the configuration needs {} values, the source holds {}",
                geometry.rows * geometry.cols,
                weight.len()
            ),
        ));
    }
    name.clear();
    write!(name, "{base}.{leaf}.scale").expect("a String cannot fail to be written to");
    let scales = linalg::decode_by_name(src, name, limit)?;
    linalg::apply_scales(
        &mut weight,
        geometry.rows,
        geometry.cols,
        &scales,
        geometry.grid,
    )?;
    Ok(weight)
}

/// A shape refusal naming the operation that could not be applied.
fn shape_error(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::dtype::{DType, dense_scale_shape, expert_scale_shape};
    use crate::linalg::REFERENCE_TENSOR_LIMIT;
    use crate::quant;
    use crate::synthetic::SyntheticWeights;
    use crate::tensor::{TensorPlan, TensorSpec};
    use crate::weights::WeightError;

    /// The element limit the reference pass uses by default: these tests pass it explicitly because
    /// [moe_forward] now takes the caller's bound, and the reference engine's default is
    /// [crate::linalg::REFERENCE_TENSOR_LIMIT].
    const BIG: usize = REFERENCE_TENSOR_LIMIT;

    /// A weight source a test builds tensor by tensor: the dtype and logical shape it declares and
    /// the bytes it hands back, so a test states a value instead of deriving one.
    #[derive(Default)]
    struct Hand {
        tensors: BTreeMap<String, (DType, Vec<usize>, Vec<u8>)>,
    }

    impl Hand {
        /// One tensor with the bytes the caller computed.
        fn with(mut self, name: &str, dtype: DType, shape: &[usize], bytes: Vec<u8>) -> Self {
            self.tensors
                .insert(name.to_owned(), (dtype, shape.to_vec(), bytes));
            self
        }

        /// An FP8 dense weight `[rows, cols]` with a `1.0` `Block32` grid: the shared expert's
        /// storage, one scale per 32x32 block (section 6 point 2 of the specification).
        fn dense(self, base: &str, rows: usize, cols: usize, values: &[f32]) -> Self {
            let scales = dense_scale_shape(rows, cols);
            self.with(
                &format!("{base}.weight"),
                DType::F8E4M3,
                &[rows, cols],
                fp8_bytes(values),
            )
            .with(
                &format!("{base}.scale"),
                DType::F8E8M0,
                &scales,
                unit_scales(scales),
            )
        }

        /// An FP4 routed expert weight `[rows, cols]` with a `1.0` `PerRow32` grid: one scale per 32
        /// elements along K, per output row (kernel.py:582).
        fn expert(self, base: &str, rows: usize, cols: usize, values: &[f32]) -> Self {
            let scales = expert_scale_shape(rows, cols);
            self.with(
                &format!("{base}.weight"),
                DType::F4E2M1,
                &[rows, cols],
                fp4_bytes(values),
            )
            .with(
                &format!("{base}.scale"),
                DType::F8E8M0,
                &scales,
                unit_scales(scales),
            )
        }

        /// The gate weight, which the checkpoint stores BF16 and unquantised.
        fn gate(self, prefix: &str, n_routed: usize, dim: usize, values: &[f32]) -> Self {
            self.with(
                &format!("{prefix}.ffn.gate.weight"),
                DType::BF16,
                &[n_routed, dim],
                bf16_bytes(values),
            )
        }

        /// The noaux selection bias, which is F32.
        fn bias(self, prefix: &str, n_routed: usize, values: &[f32]) -> Self {
            self.with(
                &format!("{prefix}.ffn.gate.bias"),
                DType::F32,
                &[n_routed],
                f32_bytes(values),
            )
        }
    }

    impl WeightSource for Hand {
        fn names(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            match self.tensors.get(name) {
                Some((dtype, shape, _)) => Ok(TensorSpec {
                    name: name.to_owned(),
                    dtype: *dtype,
                    shape: shape.clone(),
                }),
                None => Err(WeightError::Missing {
                    name: name.to_owned(),
                }),
            }
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            match self.tensors.get(name) {
                Some((_, _, bytes)) => Ok(bytes.clone()),
                None => Err(WeightError::Missing {
                    name: name.to_owned(),
                }),
            }
        }
    }

    /// The bytes of one F32 tensor.
    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// The bytes of one BF16 tensor. Every value these tests use is exactly representable in BF16,
    /// which [crate::quant::decode_bf16] is separately tested to read.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// The bytes of one FP8 E4M3 tensor, through the crate's own encoder.
    fn fp8_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .map(|value| quant::e4m3_encode(*value))
            .collect()
    }

    /// The bytes of one FP4 E2M1 tensor, two values per byte with the even index in the low nibble
    /// ([crate::quant::f4e2m1_pair_to_f32]).
    fn fp4_bytes(values: &[f32]) -> Vec<u8> {
        values
            .chunks(2)
            .map(|pair| {
                let low = quant::f4e2m1_encode(pair[0]);
                let high = pair.get(1).map_or(0, |value| quant::f4e2m1_encode(*value));
                low | (high << 4)
            })
            .collect()
    }

    /// A grid of E8M0 `1.0` scales, checked against the decoder rather than assumed.
    fn unit_scales(shape: [usize; 2]) -> Vec<u8> {
        let code = quant::e8m0_encode_ceil(1.0);
        assert_eq!(quant::decode_f8e8m0(&[code]), vec![1.0], "the unit scale");
        vec![code; shape[0] * shape[1]]
    }

    /// A gate row with `score` at `column` and zero elsewhere.
    fn gate_row(dim: usize, column: usize, score: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; dim];
        row[column] = score;
        row
    }

    /// A block whose routed expert `e` reads the input element `reads[e]` as an identity
    /// (`w1 = w3 = 1` at `[0][column]`, `w2 = 1` at `[0][0]`) and whose other weights are zero. The
    /// gate rows and their raw scores come from `gate_rows`. The shared expert is zeroed, because
    /// the block always reads it; a test that wants a real one calls [with_shared] again and the
    /// names are overwritten.
    fn block_source(
        prefix: &str,
        cfg: &ModelConfig,
        gate_rows: &[Vec<f32>],
        reads: &[Option<usize>],
    ) -> Hand {
        let dim = cfg.dim;
        let inter = cfg.moe_inter_dim;
        let n_routed = cfg.n_routed_experts;
        assert_eq!(gate_rows.len(), n_routed);
        assert_eq!(reads.len(), n_routed);
        let mut gate = vec![0.0f32; n_routed * dim];
        for (expert, row) in gate_rows.iter().enumerate() {
            assert_eq!(row.len(), dim);
            gate[expert * dim..(expert + 1) * dim].copy_from_slice(row);
        }
        let bias = vec![0.0f32; n_routed];
        let mut source = Hand::default()
            .gate(prefix, n_routed, dim, &gate)
            .bias(prefix, n_routed, &bias);
        for (expert, read) in reads.iter().enumerate() {
            let base = format!("{prefix}.ffn.experts.{expert}");
            let mut w1 = vec![0.0f32; inter * dim];
            let mut w3 = vec![0.0f32; inter * dim];
            let mut w2 = vec![0.0f32; dim * inter];
            if let Some(column) = *read {
                w1[column] = 1.0;
                w3[column] = 1.0;
                w2[0] = 1.0;
            }
            source = source
                .expert(&format!("{base}.w1"), inter, dim, &w1)
                .expert(&format!("{base}.w3"), inter, dim, &w3)
                .expert(&format!("{base}.w2"), dim, inter, &w2);
        }
        with_shared(source, prefix, cfg, 0, 0.0, 0.0)
    }

    /// A block whose every routed expert is zero.
    fn zero_block(prefix: &str, cfg: &ModelConfig) -> Hand {
        let n_routed = cfg.n_routed_experts;
        let reads = vec![None; n_routed];
        let rows = vec![vec![0.0f32; cfg.dim]; n_routed];
        block_source(prefix, cfg, &rows, &reads)
    }

    /// The shared expert as an identity on `column` scaled by `gate` and `up`: the element
    /// `column` of the input reaches `silu(clamp(gate * x)) * clamp(up * x)` on output element 0.
    fn with_shared(
        source: Hand,
        prefix: &str,
        cfg: &ModelConfig,
        column: usize,
        gate: f32,
        up: f32,
    ) -> Hand {
        let dim = cfg.dim;
        let inter = cfg.moe_inter_dim * cfg.n_shared_experts.max(1);
        let mut w1 = vec![0.0f32; inter * dim];
        let mut w3 = vec![0.0f32; inter * dim];
        let mut w2 = vec![0.0f32; dim * inter];
        w1[column] = gate;
        w3[column] = up;
        w2[0] = 1.0;
        source
            .dense(&format!("{prefix}.ffn.shared_experts.w1"), inter, dim, &w1)
            .dense(&format!("{prefix}.ffn.shared_experts.w3"), inter, dim, &w3)
            .dense(&format!("{prefix}.ffn.shared_experts.w2"), dim, inter, &w2)
    }

    /// The routed and activated counts come from the block a prefix names: 4 and 2 for a backbone
    /// layer, the DSpark stage's 2 and 1 for `mtp.N` (model.py:142-149).
    #[test]
    fn the_prefixes_name_backbone_layers_and_dspark_stages() {
        let cfg = ModelConfig::tiny();
        assert_eq!(
            counts(&cfg, "layers.0").unwrap(),
            (cfg.n_routed_experts, cfg.n_activated_experts)
        );
        assert_eq!(
            counts(&cfg, "mtp.0").unwrap(),
            (cfg.dspark_n_routed_experts, cfg.dspark_n_activated_experts)
        );
        assert_eq!(counts(&cfg, "layers.0").unwrap(), (4, 2));
        assert_eq!(counts(&cfg, "mtp.0").unwrap(), (2, 1));
        for bad in [
            "layers.2", "layers.x", "layers.", "mtp.1", "mtp.", "blocks.0", "layers",
        ] {
            assert!(
                counts(&cfg, bad).is_err(),
                "{bad:?} names no block of this configuration"
            );
        }
    }

    /// The selection bias decides *which* experts and never *what they weigh*: with `sigmoid`
    /// scoring and four descending scores, the bias alone promotes expert 2, and expert 2 then
    /// carries expert 2's own unbiased score. The mutants are adding the bias to the weights, which
    /// this would catch as a different value, and selecting on the unbiased scores, which could not
    /// pick expert 2 at all.
    #[test]
    fn the_selection_bias_decides_which_experts_but_not_their_weights() {
        let mut cfg = ModelConfig::tiny();
        cfg.score_func = "sigmoid".to_owned();
        let dim = cfg.dim;
        let mut gate = vec![0.0f32; 4 * dim];
        for expert in 0..4 {
            gate[expert * dim] = 4.0 - expert as f32;
        }
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let unbiased = vec![0.0f32; 4];
        let scale = cfg.route_scale as f32;
        let scores: [f32; 4] = [4.0f32, 3.0, 2.0, 1.0].map(linalg::sigmoid);

        let mut selected = Vec::new();
        route(&cfg, "layers.0", &x, &gate, &unbiased, None, &mut selected).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>(),
            vec![0, 1],
            "the two highest scores win"
        );
        let top_two = scores[0] + scores[1];
        assert!((selected[0].1 - scores[0] / top_two * scale).abs() < 1e-6);
        assert!((selected[1].1 - scores[1] / top_two * scale).abs() < 1e-6);

        let bias = [0.0f32, 0.0, 5.0, 0.0];
        let mut selected = Vec::new();
        route(&cfg, "layers.0", &x, &gate, &bias, None, &mut selected).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>(),
            vec![2, 0],
            "the bias promotes expert 2, and the order is by descending signal"
        );
        let dominated = scores[2] + scores[0];
        assert!(
            (selected[0].1 - scores[2] / dominated * scale).abs() < 1e-6,
            "expert 2 weighs {} - the bias leaked into the weights",
            selected[0].1
        );
        assert!((selected[1].1 - scores[0] / dominated * scale).abs() < 1e-6);
    }

    /// The image-span bias is used *instead of* the text bias when the caller passes it, and the
    /// text bias is then not consulted at all (model.py:818-820; specification line 367).
    #[test]
    fn the_image_span_bias_replaces_the_text_bias() {
        let mut cfg = ModelConfig::tiny();
        cfg.score_func = "sigmoid".to_owned();
        let dim = cfg.dim;
        let mut gate = vec![0.0f32; 4 * dim];
        for expert in 0..4 {
            gate[expert * dim] = 4.0 - expert as f32;
        }
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let bias = [0.0f32, 0.0, 5.0, 0.0];
        let bias_vl = [0.0f32; 4];

        let mut selected = Vec::new();
        route(
            &cfg,
            "layers.0",
            &x,
            &gate,
            &bias,
            Some(&bias_vl),
            &mut selected,
        )
        .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>(),
            vec![0, 1],
            "bias_vl is the image span's bias and the text bias is ignored beside it"
        );

        let mut selected = Vec::new();
        route(
            &cfg,
            "layers.0",
            &x,
            &gate,
            &bias_vl,
            Some(&bias),
            &mut selected,
        )
        .unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>(),
            vec![2, 0],
            "and the span's bias is the one that is used"
        );

        assert!(
            route(
                &cfg,
                "layers.0",
                &x,
                &gate,
                &bias,
                Some(&bias[..3]),
                &mut selected
            )
            .is_err(),
            "a bias_vl of the wrong length is refused"
        );
    }

    /// Equal signals select the lower expert index, so the selection is a function of the scores
    /// rather than of the order the candidates happened to be visited in.
    #[test]
    fn equal_signals_select_the_lower_index() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let gate = vec![0.0f32; 4 * dim];
        let x = vec![0.0f32; dim];
        let bias = vec![0.0f32; 4];
        let mut selected = Vec::new();
        route(&cfg, "layers.0", &x, &gate, &bias, None, &mut selected).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|(expert, _)| *expert)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    /// The normalisation divides by the sum *plus the literal `1e-20`* of model.py:825. With every
    /// raw score at `-200`, whose softplus underflows to zero, the sum is exactly zero: the weights
    /// are then `0 / 1e-20 = 0` rather than `0 / 0 = NaN`. The mutant is dropping the epsilon.
    #[test]
    fn the_reduction_epsilon_keeps_a_zero_sum_out_of_nan() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let mut gate = vec![0.0f32; 4 * dim];
        for expert in 0..4 {
            gate[expert * dim] = -1.0;
        }
        let mut x = vec![0.0f32; dim];
        x[0] = 200.0;
        let bias = vec![0.0f32; 4];
        let mut selected = Vec::new();
        route(&cfg, "layers.0", &x, &gate, &bias, None, &mut selected).unwrap();
        assert_eq!(selected.len(), 2);
        for (expert, weight) in &selected {
            assert!(weight.is_finite(), "expert {expert} weighs {weight}");
            assert_eq!(*weight, 0.0, "expert {expert}");
        }
    }

    /// `norm_topk_prob` gates the normalisation and `route_scale` always applies (model.py:824-826):
    /// normalised weights sum to `route_scale`, unnormalised ones are the raw scores times it.
    #[test]
    fn the_weights_are_normalised_only_when_the_configuration_says_so() {
        let mut cfg = ModelConfig::tiny();
        cfg.score_func = "sigmoid".to_owned();
        let dim = cfg.dim;
        let mut gate = vec![0.0f32; 4 * dim];
        for expert in 0..4 {
            gate[expert * dim] = 4.0 - expert as f32;
        }
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let bias = vec![0.0f32; 4];
        let scale = cfg.route_scale as f32;

        let mut selected = Vec::new();
        route(&cfg, "layers.0", &x, &gate, &bias, None, &mut selected).unwrap();
        let total: f32 = selected.iter().map(|(_, weight)| *weight).sum();
        assert!(
            (total - scale).abs() < 1e-6,
            "normalised weights sum to route_scale, got {total}"
        );

        cfg.norm_topk_prob = false;
        let mut selected = Vec::new();
        route(&cfg, "layers.0", &x, &gate, &bias, None, &mut selected).unwrap();
        assert!((selected[0].1 - linalg::sigmoid(4.0) * scale).abs() < 1e-6);
        assert!((selected[1].1 - linalg::sigmoid(3.0) * scale).abs() < 1e-6);
        assert!(
            (selected.iter().map(|(_, w)| *w).sum::<f32>() - scale).abs() > 1e-3,
            "the unnormalised weights must not sum to route_scale"
        );
    }

    /// A DSpark stage routes its own experts and, with one active expert, is not normalised:
    /// `norm_topk_prob` needs `topk > 1` (model.py:824).
    #[test]
    fn a_dspark_stage_routes_its_two_experts_and_normalises_nothing() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let mut gate = vec![0.0f32; 2 * dim];
        gate[0] = 4.0;
        gate[dim] = 1.0;
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let bias = vec![0.0f32; 2];
        let mut selected = Vec::new();
        route(&cfg, "mtp.0", &x, &gate, &bias, None, &mut selected).unwrap();
        assert_eq!(selected.len(), 1, "a DSpark stage activates one expert");
        assert_eq!(selected[0].0, 0);
        let expected = softplus(4.0).sqrt() * cfg.route_scale as f32;
        assert!(
            (selected[0].1 - expected).abs() < 1e-6,
            "the weight is {}, not {expected}",
            selected[0].1
        );
    }

    /// A score function the reference does not define and a temperature the router cannot divide by
    /// are configuration errors rather than silent arithmetic.
    #[test]
    fn the_router_refuses_a_configuration_it_cannot_run() {
        let dim = ModelConfig::tiny().dim;
        let gate = vec![0.0f32; 4 * dim];
        let x = vec![0.0f32; dim];
        let bias = vec![0.0f32; 4];
        let mut selected = Vec::new();

        let mut unknown = ModelConfig::tiny();
        unknown.score_func = "relu".to_owned();
        assert!(matches!(
            route(&unknown, "layers.0", &x, &gate, &bias, None, &mut selected),
            Err(EngineError::Config(_))
        ));

        let mut frozen = ModelConfig::tiny();
        frozen.gate_temp = 0.0;
        assert!(matches!(
            route(&frozen, "layers.0", &x, &gate, &bias, None, &mut selected),
            Err(EngineError::Config(_))
        ));

        let mut negative = ModelConfig::tiny();
        negative.gate_temp = -1.0;
        assert!(matches!(
            route(&negative, "layers.0", &x, &gate, &bias, None, &mut selected),
            Err(EngineError::Config(_))
        ));
    }

    /// Every slice is checked against the configuration before it is read, including the counts,
    /// which come from the prefix rather than from the slice lengths.
    #[test]
    fn the_router_refuses_geometry_that_does_not_fit() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let gate = vec![0.0f32; 4 * dim];
        let bias = vec![0.0f32; 4];
        let full = vec![0.0f32; dim];
        let short = vec![0.0f32; dim - 1];
        let mut selected = Vec::new();
        assert!(route(&cfg, "layers.0", &short, &gate, &bias, None, &mut selected).is_err());
        assert!(
            route(
                &cfg,
                "layers.0",
                &full,
                &gate[1..],
                &bias,
                None,
                &mut selected
            )
            .is_err()
        );
        assert!(
            route(
                &cfg,
                "layers.0",
                &full,
                &gate,
                &bias[1..],
                None,
                &mut selected
            )
            .is_err()
        );
        assert!(
            route(&cfg, "mtp.0", &full, &gate, &bias, None, &mut selected).is_err(),
            "a DSpark stage routes two experts, so a four-row gate weight is refused"
        );
        assert!(route(&cfg, "blocks.0", &full, &gate, &bias, None, &mut selected).is_err());
    }

    /// The scratch is sized from the configuration: the wider routed count, the intermediate width
    /// of the shared expert and one `dim`-wide row per token.
    #[test]
    fn the_scratch_is_sized_from_the_configuration() {
        let cfg = ModelConfig::tiny();
        let scratch = MoeScratch::for_config(&cfg);
        assert_eq!(scratch.scores.len(), cfg.n_routed_experts);
        assert_eq!(scratch.routed.len(), cfg.dim);
        assert_eq!(scratch.gate_branch.len(), cfg.moe_inter_dim);
        assert_eq!(scratch.gate_branch.len(), scratch.up_branch.len());
        assert!(scratch.selected.capacity() >= cfg.n_activated_experts);

        let released = ModelConfig::v41_flash_cached();
        let scratch = MoeScratch::for_config(released);
        assert_eq!(
            scratch.scores.len(),
            released
                .n_routed_experts
                .max(released.dspark_n_routed_experts)
        );
        assert_eq!(scratch.routed.len(), released.dim);
        assert_eq!(
            scratch.gate_branch.len(),
            released.moe_inter_dim * released.n_shared_experts.max(1)
        );
    }

    /// The routed expert's row is `w2 (weight * silu(clamp(w1 x)) * clamp(w3 x))`: with an identity
    /// expert and input element 0 set to 1, the row is the routing weight times `silu(1)`, and the
    /// shared expert adds its own `silu(1)` on top when it is present.
    #[test]
    fn a_routed_expert_is_scaled_by_its_routing_weight() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let reads = [Some(0usize), None, None, None];
        let rows = [
            gate_row(dim, 0, 4.0),
            gate_row(dim, 0, 0.0),
            gate_row(dim, 0, 0.0),
            gate_row(dim, 0, 0.0),
        ];
        let source = block_source("layers.0", &cfg, &rows, &reads);
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let mut scratch = MoeScratch::for_config(&cfg);
        let mut out = vec![0.0f32; dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();

        let mut selected = Vec::new();
        route(
            &cfg,
            "layers.0",
            &x,
            &rows.concat(),
            &[0.0; 4],
            None,
            &mut selected,
        )
        .unwrap();
        let weight = selected
            .iter()
            .find(|(expert, _)| *expert == 0)
            .expect("expert 0 has the highest score")
            .1;
        let expected = weight * linalg::silu(1.0);
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "out[0] is {} but the routing weight gives {expected}",
            out[0]
        );
        for (j, value) in out.iter().enumerate().skip(1) {
            assert_eq!(*value, 0.0, "column {j}");
        }

        // The shared expert is added on top, not instead.
        let source = with_shared(source, "layers.0", &cfg, 0, 1.0, 1.0);
        let mut out = vec![0.0f32; dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        let both = expected + linalg::silu(1.0);
        assert!(
            (out[0] - both).abs() < 1e-5,
            "out[0] is {} but the routed sum plus the shared expert is {both}",
            out[0]
        );
    }

    /// The c4b device path calls `add_shared_expert` on an already-populated routed row; the
    /// single-token path must add the shared expert to it, never overwrite it (the b34 bug).
    #[test]
    fn add_shared_expert_adds_to_the_routed_row_and_never_overwrites_it() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let source = with_shared(Hand::default(), "layers.0", &cfg, 0, 1.0, 1.0);
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let mut scratch = MoeScratch::for_config(&cfg);
        let mut out = vec![2.0f32; dim];
        add_shared_expert(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        assert!(
            (out[0] - (2.0 + linalg::silu(1.0))).abs() < 1e-5,
            "add_shared_expert must add the shared row to the routed row; out[0] is {}",
            out[0]
        );
        for (column, value) in out.iter().enumerate().skip(1) {
            assert_eq!(*value, 2.0, "column {column} must keep its routed value");
        }
    }

    /// The SwiGLU clamps follow model.py:845-847: the gate is clamped from above only and the up
    /// branch on both sides. With 448 - the largest E4M3 magnitude - on both branches and an input
    /// of 1, the unclamped product would be `448 * 448 = 200704` and the clamped one is
    /// `silu(10) * 10`. With `-448` on the gate the row is exactly zero: a symmetric gate clamp
    /// would clamp it to `-10` and give a small non-zero value instead.
    #[test]
    fn the_swiglu_branches_are_clamped_at_the_configured_limit() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        assert_eq!(
            quant::decode_f8e4m3(&[quant::e4m3_encode(448.0)]),
            vec![448.0],
            "448 must survive the test's encoder"
        );
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let reads = [None, None, None, None];
        let rows: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; dim]).collect();
        let source = zero_block("layers.0", &cfg);
        let mut scratch = MoeScratch::for_config(&cfg);

        let source = with_shared(source, "layers.0", &cfg, 0, 448.0, 448.0);
        let mut out = vec![0.0f32; dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        let limit = cfg.swiglu_limit as f32;
        let clamped = linalg::silu(limit) * limit;
        assert!(
            (out[0] - clamped).abs() < 1e-4,
            "out[0] is {} but the clamped product is {clamped}",
            out[0]
        );
        assert!(out[0] < 100.0, "448 * 448 must not reach the output");

        let source = block_source("layers.0", &cfg, &rows, &reads);
        let source = with_shared(source, "layers.0", &cfg, 0, -448.0, 448.0);
        let mut out = vec![0.0f32; dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        assert_eq!(
            out[0], 0.0,
            "silu of a very negative gate is zero, so the up branch is multiplied by zero"
        );
    }

    /// The block mixes nothing across tokens: a chunk of two tokens gives each row exactly what a
    /// single-token call on that token gives, with the two tokens routing to different experts.
    #[test]
    fn a_chunk_gives_every_token_its_own_row() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let rows = [
            gate_row(dim, 0, 4.0),
            gate_row(dim, 0, 0.0),
            gate_row(dim, 0, 0.0),
            gate_row(dim, 1, 4.0),
        ];
        let reads = [Some(0usize), None, None, Some(1)];
        let source = block_source("layers.0", &cfg, &rows, &reads);
        let mut chunk = vec![0.0f32; 2 * dim];
        chunk[0] = 1.0;
        chunk[dim + 1] = 1.0;
        let mut scratch = MoeScratch::for_config(&cfg);
        let mut both = vec![0.0f32; 2 * dim];
        moe_forward(
            &cfg,
            "layers.0",
            &chunk,
            &source,
            BIG,
            &mut scratch,
            &mut both,
        )
        .unwrap();
        assert!(
            both[0] != 0.0 && both[dim] != 0.0,
            "each token routed somewhere"
        );

        let mut alone = vec![0.0f32; 2 * dim];
        for token in 0..2 {
            let mut out = vec![0.0f32; dim];
            moe_forward(
                &cfg,
                "layers.0",
                &chunk[token * dim..(token + 1) * dim],
                &source,
                BIG,
                &mut scratch,
                &mut out,
            )
            .unwrap();
            alone[token * dim..(token + 1) * dim].copy_from_slice(&out);
        }
        assert_eq!(
            both, alone,
            "a token's row must not depend on its neighbours"
        );
    }

    /// The block is a pure function of the source and the input: the same source twice and a fresh
    /// source from the same plan and seed give bit-identical rows. Bit identity rather than a
    /// tolerance, because the synthetic source can hand back an E8M0 NaN scale and NaN is not equal
    /// to itself.
    #[test]
    fn the_block_is_deterministic_in_its_source() {
        let cfg = ModelConfig::tiny();
        let plan = TensorPlan::language_model(&cfg);
        let x: Vec<f32> = (0..cfg.dim)
            .map(|index| (index as f32 * 0.125) - 3.0)
            .collect();
        let run = |seed: u64| -> Vec<u32> {
            let source = SyntheticWeights::from_plan(&plan, seed);
            let mut scratch = MoeScratch::for_config(&cfg);
            let mut out = vec![0.0f32; cfg.dim];
            moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
            out.iter().map(|value| value.to_bits()).collect()
        };
        let bits = run(0x5eed);
        assert_eq!(bits, run(0x5eed), "the same seed must give the same rows");
        assert_ne!(
            bits,
            run(0xfeed),
            "a different seed must give different rows"
        );
    }

    /// A DSpark stage runs from the plan's own tensors, through the same entry point.
    #[test]
    fn a_dspark_stage_runs_against_the_plan() {
        let cfg = ModelConfig::tiny();
        let source = SyntheticWeights::from_plan(&TensorPlan::language_model(&cfg), 7);
        let mut scratch = MoeScratch::for_config(&cfg);
        let x: Vec<f32> = (0..cfg.dim).map(|index| index as f32 * 0.01).collect();
        let mut out = vec![0.0f32; cfg.dim];
        moe_forward(&cfg, "mtp.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        assert!(
            out.iter().any(|value| value.to_bits() != 0),
            "the stage produced something"
        );
    }

    /// `moe_forward` refuses a source whose declared shape disagrees with the configuration, an
    /// absent tensor, an input that is not a whole number of tokens, an output of the wrong length
    /// and a prefix that names no block.
    #[test]
    fn the_moe_block_refuses_geometry_that_does_not_fit() {
        let cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let source = zero_block("layers.0", &cfg);
        let mut scratch = MoeScratch::for_config(&cfg);
        let x = vec![0.0f32; dim];
        let mut out = vec![0.0f32; dim];
        assert!(moe_forward(&cfg, "blocks.0", &x, &source, BIG, &mut scratch, &mut out).is_err());
        assert!(
            moe_forward(
                &cfg,
                "layers.0",
                &x[1..],
                &source,
                BIG,
                &mut scratch,
                &mut out
            )
            .is_err()
        );
        assert!(
            moe_forward(
                &cfg,
                "layers.0",
                &x,
                &source,
                BIG,
                &mut scratch,
                &mut out[1..]
            )
            .is_err()
        );

        let empty = Hand::default();
        assert!(moe_forward(&cfg, "layers.0", &x, &empty, BIG, &mut scratch, &mut out).is_err());

        let narrow = vec![0.0f32; cfg.n_routed_experts * (dim - 1)];
        let source = Hand::default()
            .gate("layers.0", cfg.n_routed_experts, dim - 1, &narrow)
            .bias(
                "layers.0",
                cfg.n_routed_experts,
                &vec![0.0; cfg.n_routed_experts],
            );
        assert!(moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).is_err());
    }

    /// The tensor a [EngineError::TooLarge] refused, or a panic naming what arrived instead.
    fn too_large_name(error: EngineError) -> String {
        match error {
            EngineError::TooLarge { name, .. } => name,
            other => panic!("expected a TooLarge refusal, got {other:?}"),
        }
    }

    /// The element limit is the CALLER's, read by read: each of the four tensors the block decodes is
    /// refused when the limit falls one element below THAT tensor's count, and the refusal names it.
    /// The tiny gate weight is 4 x 64 = 256 values and an expert's `w1` is 32 x 64 = 2048, so the source
    /// every other test builds already refuses those two at 255 and 2047.
    ///
    /// The other two reads cannot be isolated that way: the selection bias is 4 values and a PerRow32
    /// scale is 32, each SMALLER than the tensor decoded before it, and the block decodes the gate
    /// weight first and a weight before its own scale. A limit that refused the small tensor would stop
    /// at the bigger one first, and the name would pin the wrong read. Those two cases therefore hand
    /// the source an oversized tensor - a 512-value bias, which only the gate weight precedes, and a
    /// 4096-value scale, which only that expert's 2048-value weight precedes - so the limit stops the
    /// read under test. Each case is one mutant's counterexample: with the reference limit restored at
    /// any single one of those four reads, only that read's case fails, and it fails on the name.
    #[test]
    fn the_caller_element_limit_bounds_every_weight_the_block_reads() {
        let cfg = ModelConfig::tiny();
        let prefix = "layers.0";
        let x = vec![0.0f32; cfg.dim];
        let mut out = vec![0.0f32; cfg.dim];
        let unit = quant::e8m0_encode_ceil(1.0);
        let plain = zero_block(prefix, &cfg);

        let refuse = |source: &Hand, limit: usize, out: &mut Vec<f32>| -> EngineError {
            let scratch = &mut MoeScratch::for_config(&cfg);
            moe_forward(&cfg, prefix, &x, source, limit, scratch, out)
                .expect_err("a tensor above the caller's limit must be refused")
        };

        // The gate weight, the first read anything else follows: 4 x 64 = 256 values, refused at 255.
        assert_eq!(
            too_large_name(refuse(&plain, 255, &mut out)),
            format!("{prefix}.ffn.gate.weight"),
            "the limit must stop the gate weight, not a later read"
        );

        // The selection bias, behind a 512-value fixture bias so that the gate weight (256) is below
        // the limit and the bias is the first tensor above it.
        let big_bias = zero_block(prefix, &cfg).with(
            &format!("{prefix}.ffn.gate.bias"),
            DType::F32,
            &[512],
            f32_bytes(&[0.0f32; 512]),
        );
        assert_eq!(
            too_large_name(refuse(&big_bias, 511, &mut out)),
            format!("{prefix}.ffn.gate.bias"),
            "the limit must stop the selection bias, not the gate weight before it"
        );

        // The first routed expert's `w1`: 32 x 64 = 2048 values, refused at 2047.
        assert_eq!(
            too_large_name(refuse(&plain, 2047, &mut out)),
            format!("{prefix}.ffn.experts.0.w1.weight"),
            "the limit must stop the first expert's w1"
        );

        // That `w1`'s scale, behind a 4096-value fixture scale so that the weight (2048) is below the
        // limit and the scale is the first tensor above it.
        let big_scale = zero_block(prefix, &cfg).with(
            &format!("{prefix}.ffn.experts.0.w1.scale"),
            DType::F8E8M0,
            &[4096],
            vec![unit; 4096],
        );
        assert_eq!(
            too_large_name(refuse(&big_scale, 4095, &mut out)),
            format!("{prefix}.ffn.experts.0.w1.scale"),
            "the limit must stop the first expert's w1 scale, not the weight before it"
        );

        // The same source and the same input under the reference limit is the straight-line block.
        moe_forward(
            &cfg,
            prefix,
            &x,
            &plain,
            BIG,
            &mut MoeScratch::for_config(&cfg),
            &mut out,
        )
        .unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }

    /// `swiglu_limit` clamps only when it is positive: the reference wraps both clamps in
    /// `if self.swiglu_limit > 0` (model.py:845-847), so a zero limit AND a negative one leave the
    /// branches alone. The shared expert here is a 448 / 448 identity, whose unclamped row is
    /// 448 * 448 = 200704; the module used to clamp the up branch to zero at a limit of 0 and produce 0.
    /// Both non-positive values are here because the two mutants differ: `>= 0.0` clamps at zero only,
    /// and `!= 0.0` clamps at zero and at a negative limit, which the reference never does.
    #[test]
    fn the_swiglu_limit_clamps_only_when_it_is_positive() {
        let mut cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let reads = [None, None, None, None];
        let rows: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; dim]).collect();
        let source = with_shared(
            block_source("layers.0", &cfg, &rows, &reads),
            "layers.0",
            &cfg,
            0,
            448.0,
            448.0,
        );
        let mut x = vec![0.0f32; dim];
        x[0] = 1.0;
        let mut scratch = MoeScratch::for_config(&cfg);
        let mut out = vec![0.0f32; dim];

        cfg.swiglu_limit = 0.0;
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        let unclamped = linalg::silu(448.0) * 448.0;
        assert!(
            (out[0] / unclamped - 1.0).abs() < 1e-3,
            "a zero limit must not clamp: out[0] is {} against {unclamped}",
            out[0]
        );
        assert!(
            out[0] > 100.0,
            "the clamped row is 99.995, so a zero limit produced the old zeroing bug: {}",
            out[0]
        );

        // A negative limit is the model's other no-clamp value (model.py:845 tests `> 0`). The mutant
        // that spells the guard `!= 0.0` clamps here and lands on the clamped row below.
        cfg.swiglu_limit = -10.0;
        let mut out = vec![0.0f32; dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        assert!(
            (out[0] / unclamped - 1.0).abs() < 1e-3,
            "a negative limit must not clamp: out[0] is {} against {unclamped}",
            out[0]
        );

        cfg.swiglu_limit = 10.0;
        let mut out = vec![0.0f32; dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        let clamped = linalg::silu(10.0) * 10.0;
        assert!(
            (out[0] - clamped).abs() < 1e-4,
            "out[0] is {} against the clamped {clamped}",
            out[0]
        );
    }

    /// A chunk's rows are the whole block's output: the block is zero when every expert is zero,
    /// which is what the storage tests above build on.
    #[test]
    fn a_zeroed_block_produces_zero() {
        let cfg = ModelConfig::tiny();
        let source = zero_block("layers.0", &cfg);
        let mut scratch = MoeScratch::for_config(&cfg);
        let x: Vec<f32> = (0..2 * cfg.dim).map(|index| index as f32 * 0.5).collect();
        let mut out = vec![9.0f32; 2 * cfg.dim];
        moe_forward(&cfg, "layers.0", &x, &source, BIG, &mut scratch, &mut out).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }
}
