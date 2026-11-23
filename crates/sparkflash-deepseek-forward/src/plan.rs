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

//! The DeepSeek device forward's open description.
//!
//! The forward no longer reads a model configuration compiled into a model crate. At open it
//! builds the same [EnginePlan](sparkflash_ops::engine_plan::EnginePlan) the scan and the catalogue
//! build from the checkpoint's `config.json` and the layout, installs it once, and reads every
//! model value, tensor name and kernel symbol from that one object.
//!
//! The loop's historical `ModelConfig` stays the host reference's value object; [values_from_plan]
//! fills every field of it from the plan's facts, so the crate carries no model constant of its
//! own. [Description] carries the plan itself and the by-name indexes the open path needs: a part
//! instance's tensor claims are the only source of a tensor name, and a part's catalogue symbols
//! are the only source of a kernel handle.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub use sparkflash_engine::ModelConfig;
use sparkflash_model::{FactValue, Part, Target};
use sparkflash_ops::engine_plan::EnginePlan;

/// A plan-construction failure, as the forward's callers spell it.
pub type ConfigError = String;

fn int(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<usize, ConfigError> {
    facts
        .get(name)
        .and_then(FactValue::as_integer)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("the plan carries no non-negative integer fact `{name}`"))
}

fn opt_int(facts: &BTreeMap<String, FactValue>, name: &str) -> Option<usize> {
    facts
        .get(name)
        .and_then(FactValue::as_integer)
        .and_then(|value| usize::try_from(value).ok())
}

fn float(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<f64, ConfigError> {
    facts
        .get(name)
        .and_then(FactValue::as_float)
        .ok_or_else(|| format!("the plan carries no numeric fact `{name}`"))
}

fn opt_float(facts: &BTreeMap<String, FactValue>, name: &str) -> Option<f64> {
    facts.get(name).and_then(FactValue::as_float)
}

fn text(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<String, ConfigError> {
    facts
        .get(name)
        .and_then(FactValue::as_text)
        .map(str::to_owned)
        .ok_or_else(|| format!("the plan carries no text fact `{name}`"))
}

fn integers(facts: &BTreeMap<String, FactValue>, name: &str) -> Result<Vec<usize>, ConfigError> {
    match facts.get(name) {
        Some(FactValue::Integers(values)) => values
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .map_err(|_| format!("the fact `{name}` carries a negative value"))
            })
            .collect(),
        _ => Err(format!("the plan carries no integer-list fact `{name}`")),
    }
}

fn boolean(facts: &BTreeMap<String, FactValue>, name: &str, default: bool) -> bool {
    match facts.get(name) {
        Some(FactValue::Bool(value)) => *value,
        _ => default,
    }
}

/// Fill the forward's model-wide value object from the plan's facts.
///
/// Every field is read from [EnginePlan::model_facts]; no literal model value survives here.
///
/// # Errors
/// A fact the plan does not carry, or one of the wrong type. The forward refuses to open rather
/// than inventing a value for a field the description left out.
pub fn values_from_plan(plan: &EnginePlan) -> Result<ModelConfig, ConfigError> {
    let facts = plan.model_facts();
    let compress_ratios = integers(facts, "compress_ratios")?;
    let n_layers = int(facts, "n_layers")?;
    let n_mtp_layers = opt_int(facts, "n_mtp_layers").unwrap_or(0);
    if compress_ratios.len() != n_layers + n_mtp_layers {
        return Err(format!(
            "the plan's compress ratios have {} entries for {n_layers}+{n_mtp_layers} layers",
            compress_ratios.len()
        ));
    }
    Ok(ModelConfig {
        vocab_size: int(facts, "vocab_size")?,
        dim: int(facts, "dim")?,
        moe_inter_dim: int(facts, "moe_inter_dim")?,
        n_layers,
        n_mtp_layers,
        dspark_block_size: opt_int(facts, "dspark_block_size").unwrap_or(0),
        dspark_noise_token_id: u32::try_from(opt_int(facts, "dspark_noise_token_id").unwrap_or(0))
            .map_err(|_| "the plan's noise token does not fit u32".to_owned())?,
        dspark_target_layer_ids: integers(facts, "dspark_target_layer_ids").unwrap_or_default(),
        dspark_markov_rank: int(facts, "dspark_markov_rank")?,
        dspark_n_routed_experts: opt_int(facts, "dspark_n_routed_experts").unwrap_or_default(),
        dspark_n_activated_experts: opt_int(facts, "dspark_n_activated_experts")
            .unwrap_or_default(),
        n_heads: int(facts, "n_heads")?,
        n_routed_experts: int(facts, "n_routed_experts")?,
        n_shared_experts: int(facts, "n_shared_experts")?,
        n_activated_experts: int(facts, "n_activated_experts")?,
        score_func: text(facts, "score_func")?,
        route_scale: float(facts, "route_scale")?,
        swiglu_limit: float(facts, "swiglu_limit")?,
        q_lora_rank: int(facts, "q_lora_rank")?,
        head_dim: int(facts, "head_dim")?,
        rope_head_dim: int(facts, "rope_head_dim")?,
        norm_eps: float(facts, "norm_eps")?,
        o_groups: int(facts, "o_groups")?,
        o_lora_rank: int(facts, "o_lora_rank")?,
        window_size: opt_int(facts, "window_size").unwrap_or(0),
        kv_source_layers: integers(facts, "kv_source_layers")?,
        index_source_layers: integers(facts, "index_source_layers")?,
        original_seq_len: int(facts, "original_seq_len")?,
        rope_theta: float(facts, "rope_theta")?,
        rope_factor: float(facts, "rope_factor")?,
        beta_fast: float(facts, "beta_fast")?,
        beta_slow: float(facts, "beta_slow")?,
        index_n_heads: int(facts, "index_n_heads")?,
        index_head_dim: int(facts, "index_head_dim")?,
        index_topk: int(facts, "index_topk")?,
        candidate_source_layer: int(facts, "candidate_source_layer")?,
        candidate_topk_blocks: int(facts, "candidate_topk_blocks")?,
        candidate_block_size: int(facts, "candidate_block_size")?,
        hc_mult: int(facts, "hc_mult")?,
        hc_sinkhorn_iters: int(facts, "hc_sinkhorn_iters")?,
        hc_eps: float(facts, "hc_eps")?,
        engram_layer_ids: integers(facts, "engram_layer_ids")?,
        engram_vocab_size: opt_int(facts, "engram_vocab_size").unwrap_or(0),
        engram_num_embeddings: integers(facts, "engram_num_embeddings").unwrap_or_default(),
        engram_max_ngram_size: opt_int(facts, "engram_max_ngram_size").unwrap_or(0),
        engram_pad_id: u32::try_from(opt_int(facts, "engram_pad_id").unwrap_or(0))
            .map_err(|_| "the plan's Engram pad id does not fit u32".to_owned())?,
        engram_compressed_vocab_size: opt_int(facts, "engram_compressed_vocab_size").unwrap_or(0),
        engram_n_heads: opt_int(facts, "engram_n_heads").unwrap_or(0),
        engram_head_dim: opt_int(facts, "engram_head_dim").unwrap_or(0),
        dtype: text(facts, "dtype")?,
        expert_dtype: text(facts, "expert_dtype")?,
        compress_rope_theta: float(facts, "compress_rope_theta")?,
        compress_ratios,
        image_token_id: u32::try_from(opt_int(facts, "image_token_id").unwrap_or(0))
            .map_err(|_| "the plan's image token does not fit u32".to_owned())?,
        gate_temp: float(facts, "gate_temp")?,
        norm_topk_prob: boolean(facts, "norm_topk_prob", true),
        vision_n_layers: opt_int(facts, "vision_n_layers"),
        vision_dim: opt_int(facts, "vision_dim"),
        vision_n_heads: opt_int(facts, "vision_n_heads"),
        vision_inter_dim: opt_int(facts, "vision_inter_dim"),
        vision_patch_size: opt_int(facts, "vision_patch_size"),
        vision_downsample_ratio: opt_int(facts, "vision_downsample_ratio"),
        vision_max_n_token: opt_int(facts, "vision_max_n_token"),
        vision_min_pixels: opt_int(facts, "vision_min_pixels"),
        vision_max_wh_ratio: opt_float(facts, "vision_max_wh_ratio"),
        vision_rope_theta: opt_float(facts, "vision_rope_theta"),
    })
}

/// The released checkpoint's directory.
///
/// The chain passes its checkpoint explicitly; the environment and the acceptance cache are the
/// fallbacks the plan test already uses.
#[must_use]
pub fn checkpoint_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SPARKFLASH_CHECKPOINT") {
        let path = PathBuf::from(path);
        if path.join("config.json").is_file() {
            return Some(path);
        }
    }
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".cache/sparkflash/acceptance/hub")
        .join("models--deepseek-ai--DeepSeek-V4.1-Flash-1017")
        .join("snapshots")
        .join("1bae0e1088993b9b82de08f59de5be5a34e42170");
    path.join("config.json").is_file().then_some(path)
}

/// Build the plan from the released checkpoint at `rank` of `world`.
///
/// # Errors
/// No checkpoint on this host, an unreadable `config.json`, or a checkpoint the plan refuses.
pub fn plan_from_checkpoint(world: usize, rank: usize) -> Result<EnginePlan, ConfigError> {
    let dir = checkpoint_dir()
        .ok_or_else(|| "no released DeepSeek checkpoint on this host".to_owned())?;
    plan_from_path(&dir, world, rank)
}

/// Build the plan from a checkpoint directory at `rank` of `world`.
///
/// # Errors
/// An unreadable `config.json`, or a checkpoint the plan refuses.
pub fn plan_from_path(dir: &Path, world: usize, rank: usize) -> Result<EnginePlan, ConfigError> {
    let config = dir.join("config.json");
    EnginePlan::from_path(&config, Target::Sm121, world, rank)
        .map_err(|error| format!("the checkpoint plan {}: {error}", config.display()))
}

/// The released values, built from the checkpoint plan.
///
/// # Errors
/// No checkpoint on this host, or a plan whose facts do not fill the value object.
pub fn released_values() -> Result<ModelConfig, ConfigError> {
    let plan = plan_from_checkpoint(4, 0)?;
    values_from_plan(&plan)
}

/// The installed values, or the released checkpoint's plan when none is installed.
///
/// # Panics
/// No checkpoint is installed or present on this host.
#[must_use]
pub fn cached_values() -> &'static ModelConfig {
    if let Some(description) = installed() {
        return &description.values;
    }
    static CACHED: std::sync::OnceLock<ModelConfig> = std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| released_values().expect("the released checkpoint must plan on this host"))
}

/// A structurally complete tiny configuration for tests that must run without a checkpoint.
#[must_use]
pub fn tiny_values() -> ModelConfig {
    ModelConfig {
        vocab_size: 128,
        dim: 64,
        moe_inter_dim: 32,
        n_layers: 2,
        n_mtp_layers: 1,
        dspark_block_size: 2,
        dspark_noise_token_id: 127,
        dspark_target_layer_ids: vec![1],
        dspark_markov_rank: 8,
        dspark_n_routed_experts: 2,
        dspark_n_activated_experts: 1,
        n_heads: 2,
        n_routed_experts: 4,
        n_shared_experts: 1,
        n_activated_experts: 2,
        score_func: "sqrtsoftplus".to_owned(),
        route_scale: 1.5,
        swiglu_limit: 10.0,
        q_lora_rank: 32,
        head_dim: 16,
        rope_head_dim: 4,
        norm_eps: 1e-20,
        o_groups: 2,
        o_lora_rank: 8,
        window_size: 8,
        kv_source_layers: vec![0],
        index_source_layers: vec![0, 1],
        original_seq_len: 64,
        rope_theta: 10_000.0,
        rope_factor: 2.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        index_n_heads: 2,
        index_head_dim: 8,
        index_topk: 4,
        candidate_source_layer: 0,
        candidate_topk_blocks: 4,
        candidate_block_size: 2,
        hc_mult: 2,
        hc_sinkhorn_iters: 2,
        hc_eps: 1e-6,
        engram_layer_ids: vec![1],
        engram_vocab_size: 100,
        engram_num_embeddings: vec![1024],
        engram_max_ngram_size: 2,
        engram_pad_id: 2,
        engram_compressed_vocab_size: 32,
        engram_n_heads: 2,
        engram_head_dim: 8,
        dtype: "fp8".to_owned(),
        expert_dtype: "fp4".to_owned(),
        compress_rope_theta: 160_000.0,
        compress_ratios: vec![2, 2, 0],
        image_token_id: 127,
        gate_temp: 1.0,
        norm_topk_prob: true,
        vision_n_layers: None,
        vision_dim: None,
        vision_n_heads: None,
        vision_inter_dim: None,
        vision_patch_size: None,
        vision_downsample_ratio: None,
        vision_max_n_token: None,
        vision_min_pixels: None,
        vision_max_wh_ratio: None,
        vision_rope_theta: None,
    }
}

/// One open description: the plan, its values and the by-name indexes.
pub struct Description {
    plan: EnginePlan,
    values: ModelConfig,
    tensors: BTreeMap<String, usize>,
    claims: BTreeMap<(usize, Part), Vec<String>>,
    symbols: BTreeMap<(Option<usize>, Part), Vec<&'static str>>,
}

impl std::fmt::Debug for Description {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Description")
            .field("layers", &self.plan.layers().len())
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl Description {
    /// Index an already-built plan.
    ///
    /// # Errors
    /// A plan whose facts do not fill the value object.
    pub fn new(plan: EnginePlan) -> Result<Self, ConfigError> {
        let values = values_from_plan(&plan)?;
        let mut tensors = BTreeMap::new();
        for (index, tensor) in plan.tensors().iter().enumerate() {
            tensors.insert(tensor.name.clone(), index);
        }
        let mut claims: BTreeMap<(usize, Part), Vec<String>> = BTreeMap::new();
        for stack in plan.stacks() {
            for block in &stack.blocks {
                for instance in &block.parts {
                    let entry = claims.entry((block.index, instance.part)).or_default();
                    for claim in &instance.claims {
                        entry.push(claim.name.clone());
                    }
                }
            }
        }
        let mut symbols: BTreeMap<(Option<usize>, Part), Vec<&'static str>> = BTreeMap::new();
        for entry in plan.symbols() {
            symbols
                .entry((entry.block, entry.part))
                .or_default()
                .extend(entry.symbols.iter().copied());
        }
        Ok(Self {
            plan,
            values,
            tensors,
            claims,
            symbols,
        })
    }

    /// The plan this description was built from.
    #[must_use]
    pub fn plan(&self) -> &EnginePlan {
        &self.plan
    }

    /// The model-wide values.
    #[must_use]
    pub fn values(&self) -> &ModelConfig {
        &self.values
    }

    /// The model-wide values as the loop's `ModelConfig`.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.values
    }

    /// A global tensor's plan name, by the name the plan uses.
    ///
    /// # Errors
    /// The plan claims no such tensor.
    pub fn named_tensor(&self, name: &str) -> Result<&str, ConfigError> {
        self.tensors
            .get_key_value(name)
            .map(|(name, _)| name.as_str())
            .ok_or_else(|| format!("the plan claims no tensor `{name}`"))
    }

    /// One layer part's tensor, by the tail its checkpoint name carries.
    ///
    /// The claim list is the only source: the name returned is the claim's own string.
    ///
    /// # Errors
    /// No part instance of `part` in `block` claims a name with that tail.
    pub fn layer_tensor(&self, block: usize, part: Part, tail: &str) -> Result<&str, ConfigError> {
        let names = self.claims.get(&(block, part)).ok_or_else(|| {
            format!("the plan has no {part:?} part instance in block {block} to name `{tail}`")
        })?;
        let suffix = format!(".{tail}");
        names
            .iter()
            .find(|name| name.ends_with(&suffix))
            .map(String::as_str)
            .ok_or_else(|| {
                format!(
                    "the {part:?} part instance in block {block} claims no tensor ending `{suffix}`"
                )
            })
    }

    /// One layer part's tensor name, with the open-time refusal a missing claim deserves.
    ///
    /// # Panics
    /// The plan does not carry the claim. A name the plan cannot supply is a bug in the open
    /// path, not a user error, so it fails at the point that asked rather than launching on a
    /// guessed string.
    #[must_use]
    pub fn layer_tensor_name(&self, block: usize, part: Part, tail: &str) -> &str {
        self.layer_tensor(block, part, tail)
            .unwrap_or_else(|error| panic!("forward open: {error}"))
    }

    /// One layer part's tensor name as an owned string.
    #[must_use]
    pub fn layer_tensor_owned(&self, block: usize, part: Part, tail: &str) -> String {
        self.layer_tensor_name(block, part, tail).to_owned()
    }

    /// The catalogue symbols of one part instance, in catalogue order.
    #[must_use]
    pub fn symbols(&self, block: Option<usize>, part: Part) -> &[&'static str] {
        self.symbols
            .get(&(block, part))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// The catalogue symbols of one layer's part instance.
    #[must_use]
    pub fn layer_symbols(&self, block: usize, part: Part) -> &[&'static str] {
        self.symbols(Some(block), part)
    }

    /// A part's catalogue symbol, by the priority list the caller names.
    ///
    /// # Errors
    /// The part claims none of the requested symbols.
    pub fn symbol(
        &self,
        block: Option<usize>,
        part: Part,
        priority: &[&str],
    ) -> Result<&'static str, ConfigError> {
        let symbols = self.symbols(block, part);
        for wanted in priority {
            if let Some(found) = symbols.iter().find(|symbol| **symbol == *wanted) {
                return Ok(found);
            }
        }
        Err(format!(
            "the plan's {part:?} part instance at {block:?} claims none of {priority:?} (it claims {symbols:?})"
        ))
    }

    /// A part's symbol that starts with `prefix`.
    ///
    /// # Errors
    /// The part has no symbol with that prefix.
    pub fn symbol_prefix(
        &self,
        block: Option<usize>,
        part: Part,
        prefix: &str,
    ) -> Result<&'static str, ConfigError> {
        self.symbols(block, part)
            .iter()
            .find(|symbol| symbol.starts_with(prefix))
            .copied()
            .ok_or_else(|| {
                format!("the plan's {part:?} part instance claims no `{prefix}*` symbol")
            })
    }
}

/// A part's catalogue symbol at one position, in catalogue order.
///
/// The catalogue order is part of the description; the forward names a role, not a symbol. A
/// position outside the list is a refusal naming the part.
///
/// # Errors
/// The part instance has no symbol at that position.
pub fn symbol_at(
    block: Option<usize>,
    part: Part,
    position: usize,
) -> Result<&'static str, ConfigError> {
    description()
        .symbols(block, part)
        .get(position)
        .copied()
        .ok_or_else(|| {
            format!(
                "the plan's {part:?} part instance at {block:?} has no symbol {position} (it claims {:?})",
                description().symbols(block, part)
            )
        })
}

/// The kernel handles the forward opens, all read from the plan's catalogue entries.
#[derive(Debug, Clone, Copy)]
pub struct KernelHandles {
    /// The MoE router's stage one.
    pub moe_router: &'static str,
    /// The routed gate/up GEMM.
    pub routed_gate_up: &'static str,
    /// The weighted SwiGLU quantiser.
    pub weighted_swiglu: &'static str,
    /// The stored MP4 routed combine.
    pub stored_combine: &'static str,
    /// The SwiGLU quantiser.
    pub swiglu: &'static str,
    /// The certified LM head.
    pub lm_head: &'static str,
    /// The RMS norm.
    pub rmsnorm: &'static str,
    /// The BF16-activation MXFP8 wo_a projection.
    pub wo_a_native: &'static str,
    /// The CSA2 sequence attention entry.
    pub flash_mla_seq: &'static str,
    /// The Engram embed table lookup.
    pub engram_embed: &'static str,
    /// The Engram gate write.
    pub engram_gate_write: &'static str,
    /// The Engram wkv projection.
    pub engram_wkv: &'static str,
    /// The reference-grade candidate select.
    pub ref_select: &'static str,
    /// The MTP router (128 experts, top-3).
    pub mtp_router: &'static str,
    /// The MTP gate/up.
    pub mtp_gate_up: &'static str,
    /// The MTP down combine.
    pub mtp_down_combine: &'static str,
    /// The MTP gate/up, deduplicated.
    pub mtp_gate_up_dedup: &'static str,
    /// The MTP down, deduplicated.
    pub mtp_down_dedup: &'static str,
    /// The MTP combine.
    pub mtp_combine: &'static str,
    /// The MTP plan.
    pub mtp_plan: &'static str,
    /// The non-MTP opt router (384 experts, top-6).
    pub router_opt: &'static str,
    /// The MXFP8 decode GEMV.
    pub gemv: &'static str,
    /// The compressed-cache append.
    pub compressed_append: &'static str,
}

impl KernelHandles {
    /// Read every handle from the installed plan's catalogue entries.
    ///
    /// # Errors
    /// A part whose catalogue entry is missing the position this forward launches.
    pub fn from_plan() -> Result<Self, ConfigError> {
        Ok(Self {
            moe_router: symbol_at(None, Part::MoeRouter, 0)?,
            routed_gate_up: symbol_at(None, Part::Moe, 9)?,
            weighted_swiglu: symbol_at(None, Part::Activation, 1)?,
            stored_combine: symbol_at(None, Part::Moe, 0)?,
            swiglu: symbol_at(None, Part::Activation, 0)?,
            lm_head: symbol_at(None, Part::LmHead, 0)?,
            rmsnorm: symbol_at(None, Part::RmsNorm, 0)?,
            wo_a_native: symbol_at(None, Part::DenseGemm, 0)?,
            flash_mla_seq: symbol_at(None, Part::MlaAttention, 8)?,
            engram_embed: symbol_at(None, Part::Engram, 0)?,
            engram_gate_write: symbol_at(None, Part::Engram, 1)?,
            engram_wkv: symbol_at(None, Part::Engram, 2)?,
            ref_select: symbol_at(None, Part::SparseIndexer, 19)?,
            mtp_router: symbol_at(None, Part::MoeRouter, 3)?,
            mtp_gate_up: symbol_at(None, Part::Moe, 2)?,
            mtp_down_combine: symbol_at(None, Part::Moe, 4)?,
            mtp_gate_up_dedup: symbol_at(None, Part::Moe, 11)?,
            mtp_down_dedup: symbol_at(None, Part::Moe, 8)?,
            mtp_combine: symbol_at(None, Part::MoeCombine, 2)?,
            mtp_plan: symbol_at(None, Part::MoePlan, 1)?,
            router_opt: symbol_at(None, Part::MoeRouter, 2)?,
            gemv: symbol_at(None, Part::DenseGemm, 1)?,
            compressed_append: symbol_at(None, Part::Csa2Compression, 0)?,
        })
    }
}

static HANDLES: Mutex<Option<KernelHandles>> = Mutex::new(None);

/// The installed plan's kernel handles.
///
/// # Panics
/// No plan is installed, or a catalogue entry this forward launches is missing.
#[must_use]
pub fn handles() -> KernelHandles {
    HANDLES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .unwrap_or_else(|| panic!("forward open: no kernel handles installed"))
}

static DESCRIPTION: Mutex<Option<&'static Description>> = Mutex::new(None);

/// Install the process's open description.
///
/// The forward opens one plan for its one rank; tests may install several in sequence.
///
/// # Errors
/// A plan whose facts do not fill the value object.
pub fn install(plan: EnginePlan) -> Result<&'static Description, ConfigError> {
    let description: &'static Description = Box::leak(Box::new(Description::new(plan)?));
    check_language_parts(description)?;
    *DESCRIPTION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(description);
    let handles = KernelHandles::from_plan()?;
    *HANDLES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(handles);
    Ok(description)
}

/// The installed description, when one was installed.
#[must_use]
pub fn installed() -> Option<&'static Description> {
    *DESCRIPTION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The installed description.
///
/// # Panics
/// Nothing was installed. The open path installs before any stage reads a name.
#[must_use]
pub fn description() -> &'static Description {
    if let Some(description) = installed() {
        return description;
    }
    let plan = plan_from_checkpoint(4, 0)
        .expect("forward open: no plan installed and no released checkpoint plans here");
    install(plan).expect("forward open: the released checkpoint plan installs")
}

/// The layer `block` claims a tensor whose name ends with `tail`.
///
/// The name is the claim's own string; a plan that does not claim it is a refusal, never a
/// guessed `format!`.
///
/// # Panics
/// The block claims no such tensor. The open path validates the names it will ask for before the
/// loop starts.
#[must_use]
pub fn layer_claim(block: usize, tail: &str) -> &'static str {
    layer_claim_any(block, tail)
}

fn layer_claim_any(block: usize, tail: &str) -> &'static str {
    let description = description();
    let suffix = format!(".{tail}");
    for ((candidate, _), names) in &description.claims {
        if *candidate != block {
            continue;
        }
        if let Some(name) = names.iter().find(|name| name.ends_with(&suffix)) {
            return name.as_str();
        }
    }
    panic!("forward open: block {block} claims no tensor ending .{tail}");
}

/// The layer prefix a claim under `segment` lives at, e.g. `layers.7.attn`.
///
/// The prefix is cut from a claimed name, never spelled here.
///
/// # Panics
/// The block claims no tensor under that segment.
#[must_use]
pub fn layer_prefix(block: usize, segment: &str) -> &'static str {
    let description = description();
    let needle = format!(".{segment}.");
    for ((candidate, _), names) in &description.claims {
        if *candidate != block {
            continue;
        }
        if let Some(name) = names.iter().find(|name| name.contains(&needle)) {
            let at = name.find(&needle).unwrap_or(0);
            return &name[..at + 1 + segment.len()];
        }
    }
    panic!("forward open: block {block} claims no tensor under .{segment}.");
}

/// The layer root a claim lives under, e.g. `layers.7`.
///
/// The root is cut from a claimed name, never spelled here.
///
/// # Panics
/// The block claims no tensor.
#[must_use]
pub fn layer_root(block: usize) -> &'static str {
    let name = layer_claim_any(block, "attn_norm.weight");
    let at = name
        .match_indices('.')
        .nth(1)
        .map_or(name.len(), |(at, _)| at);
    &name[..at]
}

/// A global tensor's plan name.
#[must_use]
pub fn named(name: &str) -> &'static str {
    description()
        .tensors
        .get_key_value(name)
        .map(|(name, _)| name.as_str())
        .unwrap_or_else(|| panic!("forward open: the plan claims no tensor {name}"))
}

/// One MTP stage's tensor, by the tail its checkpoint name carries.
#[must_use]
pub fn mtp_claim(stage: usize, tail: &str) -> &'static str {
    let description = description();
    let block = description
        .plan
        .stacks()
        .iter()
        .find(|stack| stack.kind == sparkflash_model::StackKind::Draft)
        .and_then(|stack| {
            stack.blocks.iter().find(|block| {
                block.index == stage
                    || block.parts.iter().any(|part| {
                        part.facts.get("stage").and_then(FactValue::as_integer)
                            == Some(stage as i64)
                    })
            })
        })
        .map(|block| block.index)
        .unwrap_or_else(|| panic!("forward open: no draft block for stage {stage}"));
    layer_claim_any(block, tail)
}

/// Install the released checkpoint's plan for `rank` of `world`.
///
/// # Errors
/// No checkpoint, or a plan the forward refuses.
pub fn install_released(world: usize, rank: usize) -> Result<&'static Description, ConfigError> {
    install(plan_from_checkpoint(world, rank)?)
}

/// Whether a part instance is one the forward has a block body for.
///
/// The set is the language stack's parts the four-rank body runs. A named part outside it is
/// refused at open rather than skipped silently.
#[must_use]
pub const fn forward_has_body(part: Part) -> bool {
    matches!(
        part,
        Part::RmsNorm
            | Part::HyperConnection
            | Part::ManifoldHyperConnection
            | Part::MlaAttention
            | Part::Csa2Compression
            | Part::SparseIndexer
            | Part::Moe
            | Part::Engram
            | Part::Mtp
            | Part::Embedding
            | Part::LmHead
            | Part::VisionPatchEmbed
            | Part::VisionPosition
            | Part::VisionNorm
            | Part::VisionAttention
            | Part::VisionMlp
            | Part::VisionDownsample
            | Part::VisionMerger
    )
}

/// Refuse, at open, a language block part the forward has no body for.
///
/// # Errors
/// The part is not in [forward_has_body]; the message names it and its block.
pub fn check_language_parts(description: &Description) -> Result<(), ConfigError> {
    for stack in description.plan().stacks() {
        if stack.kind != sparkflash_model::StackKind::Language {
            continue;
        }
        for block in &stack.blocks {
            for instance in &block.parts {
                if !forward_has_body(instance.part) {
                    return Err(format!(
                        "forward open: block {} carries the part {:?}, which this forward has no body for",
                        block.index, instance.part
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released() -> Option<&'static Description> {
        if let Some(description) = installed() {
            return Some(description);
        }
        let plan = plan_from_checkpoint(4, 0).ok()?;
        install(plan).ok()
    }

    #[test]
    fn every_name_the_open_uses_is_a_claim() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        for layer in 0..description.values.n_layers {
            for tail in [
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "hc_attn_fn",
                "hc_ffn_fn",
                "ffn.gate.weight",
                "ffn.shared_experts.w1.weight",
            ] {
                let name = layer_claim(layer, tail);
                assert!(
                    name.starts_with(&format!("layers.{layer}.")),
                    "{layer} {tail}: {name}"
                );
                assert!(name.ends_with(tail), "{layer} {tail}: {name}");
            }
            assert_eq!(layer_root(layer), format!("layers.{layer}"));
            assert!(layer_prefix(layer, "attn").ends_with(".attn"));
        }
        for expert in [0usize, 383] {
            let name = layer_claim(0, &format!("ffn.experts.{expert}.w1.weight"));
            assert!(
                name.ends_with(&format!("ffn.experts.{expert}.w1.weight")),
                "{name}"
            );
        }
        for stage in 0..description.values.n_mtp_layers {
            let name = mtp_claim(stage, "attn.wq_a.weight");
            assert!(name.ends_with("attn.wq_a.weight"), "{name}");
            assert!(name.starts_with("mtp."), "{name}");
        }
        for name in ["embed.weight", "norm.weight", "head.weight"] {
            assert_eq!(named(name), name);
        }
    }

    #[test]
    fn the_kernel_handles_are_catalogue_entries() {
        let Some(_) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        let handles = handles();
        for (name, symbol) in [
            ("moe_router", handles.moe_router),
            ("routed_gate_up", handles.routed_gate_up),
            ("weighted_swiglu", handles.weighted_swiglu),
            ("stored_combine", handles.stored_combine),
            ("swiglu", handles.swiglu),
            ("lm_head", handles.lm_head),
            ("rmsnorm", handles.rmsnorm),
            ("wo_a_native", handles.wo_a_native),
            ("flash_mla_seq", handles.flash_mla_seq),
            ("engram_embed", handles.engram_embed),
            ("engram_gate_write", handles.engram_gate_write),
            ("engram_wkv", handles.engram_wkv),
            ("ref_select", handles.ref_select),
            ("mtp_router", handles.mtp_router),
            ("mtp_gate_up", handles.mtp_gate_up),
            ("mtp_down_combine", handles.mtp_down_combine),
            ("mtp_gate_up_dedup", handles.mtp_gate_up_dedup),
            ("mtp_down_dedup", handles.mtp_down_dedup),
            ("mtp_combine", handles.mtp_combine),
            ("mtp_plan", handles.mtp_plan),
            ("router_opt", handles.router_opt),
            ("gemv", handles.gemv),
            ("compressed_append", handles.compressed_append),
        ] {
            assert!(symbol.ends_with("_sm121"), "{name}: {symbol}");
            let description = description();
            assert!(
                description
                    .plan()
                    .symbols()
                    .iter()
                    .any(|part| part.symbols.contains(&symbol)),
                "{name}: {symbol} is not a catalogue symbol"
            );
        }
    }

    #[test]
    fn the_language_parts_all_have_a_body() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        check_language_parts(description).expect("every DeepSeek language part has a body");
    }
}
