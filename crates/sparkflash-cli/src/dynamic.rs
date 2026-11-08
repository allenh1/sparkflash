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

//! Checkpoint tensor names for the dynamic (non-preset) path.
//!
//! The released packs each transcribe their own tensor names. A checkpoint this build has never
//! seen has no pack to do that, so this module is the one naming convention the dynamic converter
//! and the dynamic host backend share: the canonical Hugging Face spelling (`model.layers.N...`,
//! `model.language_model.layers.N...`, ChatML/Qwen-style projections, MoE experts and the Gated
//! DeltaNet block). A name is resolved by probing the source, so a checkpoint that uses either
//! prefix or a `layers.N` root works without a second table.
//!
//! The set is deliberately the parts this base can execute on the host. Everything else the
//! architecture names is reported by [unsupported_parts] so the caller can refuse with the exact
//! part list instead of failing at the first missing tensor.

use sparkflash_model::architecture::{AttentionKind, Ffn, Residual};
use sparkflash_model::{Architecture, Part};

use sparkflash_weights::WeightSource;

/// The parts the dynamic host backend executes today.
///
/// The list is [sparkflash_ops::parts::HOST_REFERENCE_PARTS] so the CLI and the web control
/// plane judge a novel checkpoint by the same capability set.
pub(crate) const DYNAMIC_PARTS: &[Part] = sparkflash_ops::parts::HOST_REFERENCE_PARTS;

/// The parts `architecture` names that the dynamic host path does not execute.
///
/// MTP and vision are optional here, exactly as in
/// [sparkflash_ops::parts::host_reference_missing]: the architecture lists a stage or tower when
/// the document declares one, but the host forward never runs either and must still serve the
/// backbone.
#[must_use]
pub(crate) fn unsupported_parts(architecture: &Architecture) -> Vec<Part> {
    architecture
        .parts()
        .iter()
        .filter(|part| !matches!(part, Part::Mtp | Part::Vision) && !DYNAMIC_PARTS.contains(part))
        .collect()
}

/// Everything the dynamic host backend needs from one checkpoint, resolved once.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedNames {
    /// The token embedding table.
    pub embed: String,
    /// The final RMSNorm.
    pub norm: String,
    /// The LM head, or the embedding when the head is tied and the checkpoint stores no head.
    pub head: String,
    /// One entry per decoder layer, in checkpoint order.
    pub layers: Vec<ResolvedLayer>,
}

/// The tensors of one decoder layer the dynamic path reads.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedLayer {
    /// The pre-attention RMSNorm.
    pub input_norm: String,
    /// The attention block.
    pub attention: ResolvedAttention,
    /// The pre-FFN RMSNorm.
    pub post_norm: String,
    /// The feed-forward block.
    pub ffn: ResolvedFfn,
}

/// One attention family's tensors.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedAttention {
    /// Grouped-query attention.
    Gqa {
        /// Query projection.
        q: String,
        /// Key projection.
        k: String,
        /// Value projection.
        v: String,
        /// Output projection.
        o: String,
        /// Per-head query RMSNorm, when the checkpoint carries one.
        q_norm: Option<String>,
        /// Per-head key RMSNorm, when the checkpoint carries one.
        k_norm: Option<String>,
    },
    /// Gated DeltaNet.
    Gdn {
        /// Fused q/k/v projection.
        qkv: String,
        /// Output gate projection.
        z: String,
        /// Beta projection.
        b: String,
        /// Decay projection.
        a: String,
        /// Output projection.
        out: String,
        /// Depthwise causal convolution.
        conv: String,
        /// Log decay rate per value head.
        a_log: String,
        /// Decay bias per value head.
        dt_bias: String,
        /// Gated RMSNorm weight.
        norm: String,
    },
}

/// One FFN family's tensors.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedFfn {
    /// A dense SwiGLU.
    Dense {
        /// Gate projection.
        gate: String,
        /// Up projection.
        up: String,
        /// Down projection.
        down: String,
    },
    /// Routed experts plus an optional shared expert.
    Moe {
        /// Router logits.
        router: String,
        /// Per-expert SwiGLU tensors.
        experts: Vec<ExpertNames>,
        /// The shared expert's SwiGLU tensors, when the checkpoint stores one.
        shared: Option<ExpertNames>,
        /// The learned scalar the shared expert is gated by.
        shared_gate: Option<String>,
    },
}

/// One MoE expert's SwiGLU tensors.
#[derive(Debug, Clone)]
pub(crate) struct ExpertNames {
    /// Gate projection.
    pub gate: String,
    /// Up projection.
    pub up: String,
    /// Down projection.
    pub down: String,
}

/// Resolves every tensor the dynamic path reads for `architecture` from `source`.
///
/// # Errors
///
/// A message naming the architecture part this base does not execute, or the first tensor none
/// of the canonical spellings names.
pub(crate) fn resolve<S: WeightSource + ?Sized>(
    architecture: &Architecture,
    source: &S,
) -> Result<ResolvedNames, String> {
    let unsupported = unsupported_parts(architecture);
    if !unsupported.is_empty() {
        let names: Vec<String> = unsupported.iter().map(ToString::to_string).collect();
        return Err(format!(
            "the checkpoint names parts the dynamic host path does not execute: {}",
            names.join(", ")
        ));
    }
    let embed = resolve_name(source, &embedding_candidates())?;
    let norm = resolve_name(source, &final_norm_candidates())?;
    let head = resolve_name(source, &lm_head_candidates())
        .or_else(|_| {
            if architecture.embedding_tied_to_head {
                Ok(embed.clone())
            } else {
                Err("the checkpoint stores no lm_head.weight and does not tie its head to the embedding".to_owned())
            }
        })?;
    let mut layers = Vec::with_capacity(architecture.layers.len());
    for layer in &architecture.layers {
        if !matches!(layer.residual, Residual::Plain) {
            return Err(format!(
                "layer {} uses a hyper-connection, which the dynamic host path does not execute",
                layer.index
            ));
        }
        if !layer.sides.is_empty() {
            return Err(format!(
                "layer {} carries a side lookup (Engram or PLE), which the dynamic host path does not execute",
                layer.index
            ));
        }
        let prefixes = layer_prefixes(layer.index);
        let input_norm = resolve_from_prefixes(source, &prefixes, &[".input_layernorm.weight"])?;
        let post_norm = resolve_from_prefixes(
            source,
            &prefixes,
            &[
                ".post_attention_layernorm.weight",
                ".ffn_norm.weight",
                ".pre_ffn_layernorm.weight",
            ],
        )?;
        let attention = match &layer.attention.kind {
            AttentionKind::Gqa { .. } => ResolvedAttention::Gqa {
                q: resolve_from_prefixes(source, &prefixes, &[".self_attn.q_proj.weight"])?,
                k: resolve_from_prefixes(source, &prefixes, &[".self_attn.k_proj.weight"])?,
                v: resolve_from_prefixes(source, &prefixes, &[".self_attn.v_proj.weight"])?,
                o: resolve_from_prefixes(source, &prefixes, &[".self_attn.o_proj.weight"])?,
                q_norm: resolve_optional_from_prefixes(
                    source,
                    &prefixes,
                    &[".self_attn.q_norm.weight"],
                ),
                k_norm: resolve_optional_from_prefixes(
                    source,
                    &prefixes,
                    &[".self_attn.k_norm.weight"],
                ),
            },
            AttentionKind::Gdn(_) => ResolvedAttention::Gdn {
                qkv: resolve_from_prefixes(
                    source,
                    &prefixes,
                    &[
                        ".linear_attn.in_proj_qkv.weight",
                        ".linear_attn.in_qkv.weight",
                    ],
                )?,
                z: resolve_from_prefixes(
                    source,
                    &prefixes,
                    &[".linear_attn.in_proj_z.weight", ".linear_attn.in_z.weight"],
                )?,
                b: resolve_from_prefixes(
                    source,
                    &prefixes,
                    &[".linear_attn.in_proj_b.weight", ".linear_attn.in_b.weight"],
                )?,
                a: resolve_from_prefixes(
                    source,
                    &prefixes,
                    &[".linear_attn.in_proj_a.weight", ".linear_attn.in_a.weight"],
                )?,
                out: resolve_from_prefixes(source, &prefixes, &[".linear_attn.out_proj.weight"])?,
                conv: resolve_from_prefixes(source, &prefixes, &[".linear_attn.conv1d.weight"])?,
                a_log: resolve_from_prefixes(source, &prefixes, &[".linear_attn.A_log"])?,
                dt_bias: resolve_from_prefixes(source, &prefixes, &[".linear_attn.dt_bias"])?,
                norm: resolve_from_prefixes(source, &prefixes, &[".linear_attn.norm.weight"])?,
            },
            other => {
                return Err(format!(
                    "layer {} uses {other:?}, which the dynamic host path does not execute",
                    layer.index
                ));
            }
        };
        let ffn = match &layer.ffn {
            Ffn::Dense(dense) => {
                let _ = dense;
                ResolvedFfn::Dense {
                    gate: resolve_from_prefixes(source, &prefixes, &[".mlp.gate_proj.weight"])?,
                    up: resolve_from_prefixes(source, &prefixes, &[".mlp.up_proj.weight"])?,
                    down: resolve_from_prefixes(source, &prefixes, &[".mlp.down_proj.weight"])?,
                }
            }
            Ffn::Moe(moe) => {
                let router = resolve_from_prefixes(
                    source,
                    &prefixes,
                    &[".mlp.gate.weight", ".mlp.router.weight"],
                )?;
                let mut experts = Vec::with_capacity(moe.routed_experts);
                for expert in 0..moe.routed_experts {
                    let base = format!(".mlp.experts.{expert}");
                    experts.push(ExpertNames {
                        gate: resolve_from_prefixes(
                            source,
                            &prefixes,
                            &[&format!("{base}.gate_proj.weight")],
                        )?,
                        up: resolve_from_prefixes(
                            source,
                            &prefixes,
                            &[&format!("{base}.up_proj.weight")],
                        )?,
                        down: resolve_from_prefixes(
                            source,
                            &prefixes,
                            &[&format!("{base}.down_proj.weight")],
                        )?,
                    });
                }
                let shared = if moe.shared_experts == 0 {
                    None
                } else {
                    Some(ExpertNames {
                        gate: resolve_from_prefixes(
                            source,
                            &prefixes,
                            &[".mlp.shared_expert.gate_proj.weight"],
                        )?,
                        up: resolve_from_prefixes(
                            source,
                            &prefixes,
                            &[".mlp.shared_expert.up_proj.weight"],
                        )?,
                        down: resolve_from_prefixes(
                            source,
                            &prefixes,
                            &[".mlp.shared_expert.down_proj.weight"],
                        )?,
                    })
                };
                let shared_gate =
                    resolve_from_prefixes(source, &prefixes, &[".mlp.shared_expert_gate.weight"])
                        .ok();
                ResolvedFfn::Moe {
                    router,
                    experts,
                    shared,
                    shared_gate,
                }
            }
        };
        layers.push(ResolvedLayer {
            input_norm,
            attention,
            post_norm,
            ffn,
        });
    }
    Ok(ResolvedNames {
        embed,
        norm,
        head,
        layers,
    })
}

/// Every tensor name [resolve] would read, in forward order.
#[must_use]
pub(crate) fn walk(names: &ResolvedNames) -> Vec<String> {
    let mut out = vec![names.embed.clone()];
    for layer in &names.layers {
        out.push(layer.input_norm.clone());
        match &layer.attention {
            ResolvedAttention::Gqa {
                q,
                k,
                v,
                o,
                q_norm,
                k_norm,
            } => {
                out.extend([q.clone(), k.clone(), v.clone(), o.clone()]);
                if let Some(q_norm) = q_norm {
                    out.push(q_norm.clone());
                }
                if let Some(k_norm) = k_norm {
                    out.push(k_norm.clone());
                }
            }
            ResolvedAttention::Gdn {
                qkv,
                z,
                b,
                a,
                out: o,
                conv,
                a_log,
                dt_bias,
                norm,
            } => {
                out.extend([
                    qkv.clone(),
                    z.clone(),
                    b.clone(),
                    a.clone(),
                    o.clone(),
                    conv.clone(),
                    a_log.clone(),
                    dt_bias.clone(),
                    norm.clone(),
                ]);
            }
        }
        out.push(layer.post_norm.clone());
        match &layer.ffn {
            ResolvedFfn::Dense { gate, up, down } => {
                out.extend([gate.clone(), up.clone(), down.clone()]);
            }
            ResolvedFfn::Moe {
                router,
                experts,
                shared,
                shared_gate,
            } => {
                out.push(router.clone());
                for expert in experts {
                    out.extend([expert.gate.clone(), expert.up.clone(), expert.down.clone()]);
                }
                if let Some(shared) = shared {
                    out.extend([shared.gate.clone(), shared.up.clone(), shared.down.clone()]);
                }
                if let Some(gate) = shared_gate {
                    out.push(gate.clone());
                }
            }
        }
    }
    out.push(names.norm.clone());
    out.push(names.head.clone());
    out
}

fn resolve_from_prefixes<S: WeightSource + ?Sized>(
    source: &S,
    prefixes: &[String],
    suffixes: &[&str],
) -> Result<String, String> {
    let candidates = candidates_for(prefixes, suffixes);
    resolve_name(source, &candidates)
}

fn resolve_optional_from_prefixes<S: WeightSource + ?Sized>(
    source: &S,
    prefixes: &[String],
    suffixes: &[&str],
) -> Option<String> {
    candidates_for(prefixes, suffixes)
        .into_iter()
        .find(|candidate| source.spec(candidate).is_ok())
}

fn candidates_for(prefixes: &[String], suffixes: &[&str]) -> Vec<String> {
    prefixes
        .iter()
        .flat_map(|prefix| {
            suffixes
                .iter()
                .map(move |suffix| format!("{prefix}{suffix}"))
        })
        .collect()
}

fn resolve_name<S: WeightSource + ?Sized>(
    source: &S,
    candidates: &[String],
) -> Result<String, String> {
    for candidate in candidates {
        if source.spec(candidate).is_ok() {
            return Ok(candidate.clone());
        }
    }
    Err(format!(
        "none of these tensors is in the checkpoint: {}",
        candidates.join(", ")
    ))
}

fn embedding_candidates() -> Vec<String> {
    [
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.embedding.weight",
        "model.language_model.embedding.weight",
        "embed_tokens.weight",
        "model.embed.weight",
        "embed.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn final_norm_candidates() -> Vec<String> {
    [
        "model.norm.weight",
        "model.language_model.norm.weight",
        "model.final_layernorm.weight",
        "norm.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn lm_head_candidates() -> Vec<String> {
    [
        "lm_head.weight",
        "model.lm_head.weight",
        "model.language_model.lm_head.weight",
        "model.head.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn layer_prefixes(index: usize) -> Vec<String> {
    [
        format!("model.language_model.layers.{index}"),
        format!("model.layers.{index}"),
        format!("layers.{index}"),
    ]
    .into()
}
