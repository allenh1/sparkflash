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

//! Tensor names for a family no pack transcribes.
//!
//! The released packs each name their own tensors in [crate::tensor_plan]. A checkpoint this
//! build has never seen has no pack, so this module is the one naming convention every reader of
//! such a checkpoint shares: the converter, the host reference, the web's fit and the shared
//! [ArchitectureTensorPlan](crate::tensor_plan::ArchitectureTensorPlan). It names the canonical
//! Hugging Face spelling (`model.layers.N...`, `model.language_model.layers.N...` or a bare
//! `layers.N` root; Llama/Qwen-style projections, routed experts and the Gated DeltaNet block).
//!
//! A [NamingDialect] is the few choices a checkpoint makes inside that convention (its layer
//! root, its post-attention norm's name, whether its GQA heads carry norms, the router's name).
//! [NamingDialect::probe] reads them from a checkpoint's headers once, from the first layer of
//! each kind; [NamingDialect::canonical] assumes them for a document alone (the web's fit has no
//! weights). [NamingDialect::names] then spells every tensor from the [Architecture], and
//! [resolve] checks that the checkpoint holds each one.

use sparkflash_model::Architecture;
use sparkflash_model::architecture::{AttentionKind, Ffn, Residual};
use sparkflash_weights::WeightSource;

/// Every tensor a novel family's backbone reads, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorNames {
    /// The token embedding table.
    pub embed: String,
    /// The final RMSNorm.
    pub norm: String,
    /// The LM head, or the embedding's name when the head is tied and the checkpoint stores none.
    pub head: String,
    /// One entry per decoder layer, in checkpoint order.
    pub layers: Vec<LayerNames>,
}

impl TensorNames {
    /// Whether the head reads the embedding table rather than a tensor of its own.
    #[must_use]
    pub fn head_tied(&self) -> bool {
        self.head == self.embed
    }

    /// Every distinct tensor name, in forward order (a tied head is the embedding, named once).
    #[must_use]
    pub fn walk(&self) -> Vec<String> {
        let mut out = vec![self.embed.clone()];
        for layer in &self.layers {
            out.push(layer.input_norm.clone());
            match &layer.attention {
                AttentionNames::Gqa {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                } => {
                    out.extend([q.clone(), k.clone(), v.clone(), o.clone()]);
                    out.extend(q_norm.iter().cloned());
                    out.extend(k_norm.iter().cloned());
                }
                AttentionNames::Gdn(gdn) => out.extend(gdn.all().into_iter().map(str::to_owned)),
            }
            out.push(layer.post_norm.clone());
            match &layer.ffn {
                FfnNames::Dense(dense) => {
                    out.extend([dense.gate.clone(), dense.up.clone(), dense.down.clone()]);
                }
                FfnNames::Moe {
                    router,
                    bias,
                    experts,
                    shared,
                    shared_gate,
                } => {
                    out.push(router.clone());
                    out.extend(bias.iter().cloned());
                    for expert in experts.iter().chain(shared) {
                        out.extend([expert.gate.clone(), expert.up.clone(), expert.down.clone()]);
                    }
                    out.extend(shared_gate.iter().cloned());
                }
            }
        }
        out.push(self.norm.clone());
        if !self.head_tied() {
            out.push(self.head.clone());
        }
        out
    }
}

/// The tensors of one decoder layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerNames {
    /// The pre-attention RMSNorm.
    pub input_norm: String,
    /// The attention block.
    pub attention: AttentionNames,
    /// The pre-FFN RMSNorm.
    pub post_norm: String,
    /// The feed-forward block.
    pub ffn: FfnNames,
}

/// One attention family's tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionNames {
    /// Grouped-query attention.
    Gqa {
        /// Query projection (twice the attention width when the output is gated).
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
    Gdn(GdnNames),
}

/// A Gated DeltaNet block's tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdnNames {
    /// Fused q/k/v projection.
    pub qkv: String,
    /// Output gate projection.
    pub z: String,
    /// Beta projection.
    pub b: String,
    /// Decay projection.
    pub a: String,
    /// Output projection.
    pub out: String,
    /// Depthwise causal convolution.
    pub conv: String,
    /// Log decay rate per value head.
    pub a_log: String,
    /// Decay bias per value head.
    pub dt_bias: String,
    /// Gated RMSNorm weight.
    pub norm: String,
}

impl GdnNames {
    /// Every tensor, in the order the forward reads them.
    #[must_use]
    pub fn all(&self) -> [&str; 9] {
        [
            &self.qkv,
            &self.z,
            &self.b,
            &self.a,
            &self.out,
            &self.conv,
            &self.a_log,
            &self.dt_bias,
            &self.norm,
        ]
    }
}

/// One FFN family's tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfnNames {
    /// A dense SwiGLU.
    Dense(ExpertNames),
    /// Routed experts plus an optional shared expert.
    Moe {
        /// Router logits.
        router: String,
        /// The selection's correction bias (`e_score_correction_bias`), when selection adds one.
        bias: Option<String>,
        /// Per-expert SwiGLU tensors.
        experts: Vec<ExpertNames>,
        /// The shared expert's SwiGLU tensors, when the checkpoint stores one.
        shared: Option<ExpertNames>,
        /// The learned scalar the shared expert is gated by.
        shared_gate: Option<String>,
    },
}

/// One SwiGLU's three projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertNames {
    /// Gate projection.
    pub gate: String,
    /// Up projection.
    pub up: String,
    /// Down projection.
    pub down: String,
}

impl ExpertNames {
    fn under(prefix: &str) -> Self {
        Self {
            gate: format!("{prefix}.gate_proj.weight"),
            up: format!("{prefix}.up_proj.weight"),
            down: format!("{prefix}.down_proj.weight"),
        }
    }
}

/// The layer roots the convention accepts, most specific first.
const LAYER_ROOTS: [&str; 3] = ["model.language_model.layers", "model.layers", "layers"];

/// The post-attention norm's spellings.
const POST_NORMS: [&str; 3] = ["post_attention_layernorm", "ffn_norm", "pre_ffn_layernorm"];

/// The router's spellings.
const ROUTERS: [&str; 2] = ["gate", "router"];

/// The shared expert's spellings: one MLP either way.
const SHARED_EXPERTS: [&str; 2] = ["shared_expert", "shared_experts"];

const EMBEDDINGS: [&str; 7] = [
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "model.embedding.weight",
    "model.language_model.embedding.weight",
    "embed_tokens.weight",
    "model.embed.weight",
    "embed.weight",
];

const FINAL_NORMS: [&str; 4] = [
    "model.norm.weight",
    "model.language_model.norm.weight",
    "model.final_layernorm.weight",
    "norm.weight",
];

const HEADS: [&str; 4] = [
    "lm_head.weight",
    "model.lm_head.weight",
    "model.language_model.lm_head.weight",
    "model.head.weight",
];

/// The choices one checkpoint makes inside the canonical spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamingDialect {
    embed: String,
    norm: String,
    /// The head's name, or `None` when it is the embedding.
    head: Option<String>,
    layer_root: &'static str,
    post_norm: &'static str,
    qk_norm: bool,
    /// Whether the GDN projections are spelled `in_qkv` rather than `in_proj_qkv`.
    gdn_short: bool,
    router: &'static str,
    /// The shared expert's spelling: Qwen's `shared_expert` or DeepSeek's `shared_experts`.
    shared: &'static str,
    /// Whether a shared expert carries its scalar gate, or `None` to take the document's word.
    shared_gate: Option<bool>,
}

impl NamingDialect {
    /// The spelling a document alone implies: `model.layers`, Llama's norm names, no per-head
    /// norms, a shared expert's gate when the document declares one, and a head of its own
    /// unless the document ties it.
    #[must_use]
    pub fn canonical(architecture: &Architecture) -> Self {
        Self {
            head: (!architecture.embedding_tied_to_head).then(|| HEADS[0].to_owned()),
            embed: EMBEDDINGS[0].to_owned(),
            norm: FINAL_NORMS[0].to_owned(),
            layer_root: LAYER_ROOTS[1],
            post_norm: POST_NORMS[0],
            qk_norm: false,
            gdn_short: false,
            router: ROUTERS[0],
            shared: SHARED_EXPERTS[0],
            shared_gate: None,
        }
    }

    /// The spelling `source` uses, read from its headers: each choice from the first tensor that
    /// makes it (the first layer for the root and norms, the first layer of each attention and
    /// FFN kind for the rest).
    ///
    /// # Errors
    /// A tensor every spelling misses, named with the spellings tried.
    pub fn probe<S: WeightSource + ?Sized>(
        architecture: &Architecture,
        source: &S,
    ) -> Result<Self, String> {
        let has = |name: &str| source.spec(name).is_ok();
        let first = |candidates: &[&str]| -> Result<String, String> {
            candidates
                .iter()
                .find(|name| has(name))
                .map(|name| (*name).to_owned())
                .ok_or_else(|| {
                    format!(
                        "none of these tensors is in the checkpoint: {}",
                        candidates.join(", ")
                    )
                })
        };
        let embed = first(&EMBEDDINGS)?;
        let norm = first(&FINAL_NORMS)?;
        let head = match first(&HEADS) {
            Ok(head) => Some(head),
            Err(_) if architecture.embedding_tied_to_head => None,
            Err(_) => {
                return Err("the checkpoint stores no lm_head.weight and does not tie its head to the embedding".to_owned());
            }
        };
        let Some(reference) = architecture.layers.first() else {
            return Err("the architecture has no layers".to_owned());
        };
        let at = |root: &str, index: usize, suffix: &str| format!("{root}.{index}.{suffix}");
        let layer_root = LAYER_ROOTS
            .into_iter()
            .find(|root| has(&at(root, reference.index, "input_layernorm.weight")))
            .ok_or_else(|| {
                format!(
                    "none of these tensors is in the checkpoint: {}",
                    LAYER_ROOTS
                        .map(|root| at(root, reference.index, "input_layernorm.weight"))
                        .join(", ")
                )
            })?;
        let pick = |index: usize, choices: &[&'static str], suffix: &str| {
            choices
                .iter()
                .copied()
                .find(|choice| has(&at(layer_root, index, &format!("{choice}{suffix}"))))
                .ok_or_else(|| {
                    format!(
                        "none of these tensors is in the checkpoint: {}",
                        choices
                            .iter()
                            .map(|choice| at(layer_root, index, &format!("{choice}{suffix}")))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
        };
        let post_norm = pick(reference.index, &POST_NORMS, ".weight")?;
        let gqa = architecture
            .layers
            .iter()
            .find(|layer| matches!(layer.attention.kind, AttentionKind::Gqa { .. }));
        let qk_norm =
            gqa.is_some_and(|layer| has(&at(layer_root, layer.index, "self_attn.q_norm.weight")));
        let gdn = architecture
            .layers
            .iter()
            .find(|layer| matches!(layer.attention.kind, AttentionKind::Gdn(_)));
        let gdn_short = gdn.is_some_and(|layer| {
            !has(&at(
                layer_root,
                layer.index,
                "linear_attn.in_proj_qkv.weight",
            )) && has(&at(layer_root, layer.index, "linear_attn.in_qkv.weight"))
        });
        let moe = architecture
            .layers
            .iter()
            .find(|layer| matches!(layer.ffn, Ffn::Moe(_)));
        let router = match moe {
            Some(layer) => {
                let found = pick(layer.index, &["mlp.gate", "mlp.router"], ".weight")?;
                ROUTERS
                    .into_iter()
                    .find(|router| found.ends_with(router))
                    .unwrap_or(ROUTERS[0])
            }
            None => ROUTERS[0],
        };
        // The first layer that holds a shared expert says how it is spelled.
        let shared = architecture
            .layers
            .iter()
            .find(|layer| matches!(&layer.ffn, Ffn::Moe(moe) if moe.shared_experts > 0))
            .map_or(Ok(SHARED_EXPERTS[0]), |layer| {
                let found = pick(
                    layer.index,
                    &["mlp.shared_expert", "mlp.shared_experts"],
                    ".gate_proj.weight",
                )?;
                Ok::<_, String>(found.trim_start_matches("mlp."))
            })?;
        let shared_gate = Some(moe.is_some_and(|layer| {
            has(&at(
                layer_root,
                layer.index,
                "mlp.shared_expert_gate.weight",
            ))
        }));
        Ok(Self {
            embed,
            norm,
            head,
            layer_root,
            post_norm,
            qk_norm,
            gdn_short,
            router,
            shared,
            shared_gate,
        })
    }

    /// Every tensor `architecture` reads, in this dialect.
    ///
    /// # Errors
    /// A layer the convention does not name: a hyper-connection, a side lookup (Engram or PLE),
    /// or an attention family other than GQA and Gated DeltaNet.
    pub fn names(&self, architecture: &Architecture) -> Result<TensorNames, String> {
        let mut layers = Vec::with_capacity(architecture.layers.len());
        for layer in &architecture.layers {
            if !matches!(layer.residual, Residual::Plain) {
                return Err(format!(
                    "layer {} uses a hyper-connection, which the shared naming does not spell",
                    layer.index
                ));
            }
            if !layer.sides.is_empty() {
                return Err(format!(
                    "layer {} carries a side lookup (Engram or PLE), which the shared naming does not spell",
                    layer.index
                ));
            }
            let p = format!("{}.{}", self.layer_root, layer.index);
            let attention = match &layer.attention.kind {
                AttentionKind::Gqa { .. } => {
                    let a = format!("{p}.self_attn");
                    AttentionNames::Gqa {
                        q: format!("{a}.q_proj.weight"),
                        k: format!("{a}.k_proj.weight"),
                        v: format!("{a}.v_proj.weight"),
                        o: format!("{a}.o_proj.weight"),
                        q_norm: self.qk_norm.then(|| format!("{a}.q_norm.weight")),
                        k_norm: self.qk_norm.then(|| format!("{a}.k_norm.weight")),
                    }
                }
                AttentionKind::Gdn(_) => {
                    let a = format!("{p}.linear_attn");
                    let proj = if self.gdn_short { "in" } else { "in_proj" };
                    AttentionNames::Gdn(GdnNames {
                        qkv: format!("{a}.{proj}_qkv.weight"),
                        z: format!("{a}.{proj}_z.weight"),
                        b: format!("{a}.{proj}_b.weight"),
                        a: format!("{a}.{proj}_a.weight"),
                        out: format!("{a}.out_proj.weight"),
                        conv: format!("{a}.conv1d.weight"),
                        a_log: format!("{a}.A_log"),
                        dt_bias: format!("{a}.dt_bias"),
                        norm: format!("{a}.norm.weight"),
                    })
                }
                other => {
                    return Err(format!(
                        "layer {} uses {other:?}, which the shared naming does not spell",
                        layer.index
                    ));
                }
            };
            let ffn = match &layer.ffn {
                Ffn::Dense(_) => FfnNames::Dense(ExpertNames::under(&format!("{p}.mlp"))),
                Ffn::Moe(moe) => FfnNames::Moe {
                    router: format!("{p}.mlp.{}.weight", self.router),
                    bias: moe
                        .selection_bias
                        .then(|| format!("{p}.mlp.{}.e_score_correction_bias", self.router)),
                    experts: (0..moe.routed_experts)
                        .map(|expert| ExpertNames::under(&format!("{p}.mlp.experts.{expert}")))
                        .collect(),
                    shared: (moe.shared_experts > 0)
                        .then(|| ExpertNames::under(&format!("{p}.mlp.{}", self.shared))),
                    shared_gate: (moe.shared_experts > 0
                        && self.shared_gate.unwrap_or(moe.shared_gated))
                    .then(|| format!("{p}.mlp.shared_expert_gate.weight")),
                },
            };
            layers.push(LayerNames {
                input_norm: format!("{p}.input_layernorm.weight"),
                attention,
                post_norm: format!("{p}.{}.weight", self.post_norm),
                ffn,
            });
        }
        Ok(TensorNames {
            head: self.head.clone().unwrap_or_else(|| self.embed.clone()),
            embed: self.embed.clone(),
            norm: self.norm.clone(),
            layers,
        })
    }
}

/// Every tensor `architecture` reads from `source`, in the dialect `source` uses, each checked to
/// be in the checkpoint.
///
/// # Errors
/// As [NamingDialect::probe] and [NamingDialect::names], or the first name the checkpoint does not
/// hold.
pub fn resolve<S: WeightSource + ?Sized>(
    architecture: &Architecture,
    source: &S,
) -> Result<TensorNames, String> {
    let names = NamingDialect::probe(architecture, source)?.names(architecture)?;
    if let Some(missing) = names
        .walk()
        .into_iter()
        .find(|name| source.spec(name).is_err())
    {
        return Err(format!(
            "the checkpoint has no {missing}, which its own spelling names"
        ));
    }
    Ok(names)
}
