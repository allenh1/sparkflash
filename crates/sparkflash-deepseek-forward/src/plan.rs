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
use std::path::Path;
#[cfg(any(test, feature = "test-support"))]
use std::path::PathBuf;
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
        dense_ffn_layers: match facts.get("dense_ffn_layers") {
            Some(_) => integers(facts, "dense_ffn_layers")?,
            None => Vec::new(),
        },
        dense_inter_dim: opt_int(facts, "dense_inter_dim").unwrap_or(0),
        gqa_layers: match facts.get("gqa_layers") {
            Some(_) => integers(facts, "gqa_layers")?,
            None => Vec::new(),
        },
        gqa: match facts.get("gqa_layers") {
            Some(_) => Some(sparkflash_engine::config::GqaGeometry {
                heads: int(facts, "gqa_heads")?,
                kv_heads: int(facts, "gqa_kv_heads")?,
                head_dim: int(facts, "gqa_head_dim")?,
                rotary: int(facts, "gqa_rotary")?,
                theta: float(facts, "gqa_rope_theta")?,
            }),
            None => None,
        },
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
        // The block and the scale values are the plan's own statement of its linear format, not
        // facts the released manifest records: read them from the resolved format.
        weight_block: plan
            .stored_format(sparkflash_model::Part::DenseMlp, None)
            .map_or(32, |format| format.block()),
        weight_scale: plan
            .stored_format(sparkflash_model::Part::DenseMlp, None)
            .map_or_else(
                || "ue8m0".to_owned(),
                |format| format.scale_values().name().to_owned(),
            ),
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

/// The released checkpoint's directory, for the unit tests that pin the shipped values.
///
/// The chain passes its checkpoint explicitly. A production read never looks one up: a model
/// value with no installed plan is refused by name in [cached_values]. The path is the
/// environment's, so no repository name or snapshot is written here.
#[cfg(any(test, feature = "test-support"))]
#[must_use]
pub fn checkpoint_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("SPARKFLASH_CHECKPOINT")?);
    path.join("config.json").is_file().then_some(path)
}

/// Build the plan from the released checkpoint at `rank` of `world`.
///
/// # Errors
/// No checkpoint on this host, an unreadable `config.json`, or a checkpoint the plan refuses.
#[cfg(any(test, feature = "test-support"))]
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
#[cfg(any(test, feature = "test-support"))]
pub fn released_values() -> Result<ModelConfig, ConfigError> {
    // The vendored released document is the plan's own input, so a unit test does not need the
    // 510 GB checkpoint on disk to ask what the released values are.
    let plan = EnginePlan::from_json(
        include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"),
        Target::Sm121,
        4,
        0,
    )
    .map_err(|error| format!("the vendored released configuration: {error}"))?;
    values_from_plan(&plan)
}

/// The installed values.
///
/// A read with no installed plan refuses by name. Only a unit test falls back to the released
/// checkpoint's plan, because a production process opens from a checkpoint before it reads a
/// value and a silent released default would hide a missed open.
///
/// # Panics
/// No plan is installed (outside tests).
#[must_use]
pub fn cached_values() -> &'static ModelConfig {
    if let Some(description) = installed() {
        publish_geometry(&description.values);
        return &description.values;
    }
    #[cfg(any(test, feature = "test-support"))]
    {
        static CACHED: std::sync::OnceLock<ModelConfig> = std::sync::OnceLock::new();
        let values = CACHED.get_or_init(|| {
            released_values().expect("the released checkpoint must plan on this host")
        });
        // Publish the released plan like an installed description so a unit test that only asks
        // for a geometry value sees the plan. The vendored document is planned at world 4.
        std::sync::atomic::AtomicUsize::store(
            &WORLD_VALUE,
            4,
            std::sync::atomic::Ordering::Relaxed,
        );
        publish_geometry(values);
        values
    }
    #[cfg(not(any(test, feature = "test-support")))]
    panic!(
        "forward open: no plan installed, so a model value has no source; open the checkpoint's          description before reading one"
    )
}

/// A structurally complete tiny configuration for tests that must run without a checkpoint.
#[must_use]
pub fn tiny_values() -> ModelConfig {
    ModelConfig {
        vocab_size: 128,
        dim: 64,
        moe_inter_dim: 32,
        dense_ffn_layers: Vec::new(),
        dense_inter_dim: 0,
        gqa_layers: Vec::new(),
        gqa: None,
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
        weight_block: 32,
        weight_scale: "ue8m0".to_owned(),
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
    /// Each part instance's operand: its stored format, the activation form its seam ends in and
    /// the body the plan claims for it. Built once, per block and part, at open.
    operands: BTreeMap<(Option<usize>, Part), crate::operand::Operand>,
    /// The draft block's tap layers, read from its part facts once at open.
    tap_layers: Box<[usize]>,
    /// Each language block's steps, decided once at open from its parts ([language_steps]).
    steps: Vec<BlockSteps>,
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
        std::sync::atomic::AtomicUsize::store(
            &WORLD_VALUE,
            plan.tp().world(),
            std::sync::atomic::Ordering::Relaxed,
        );
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
        // The operand of every part instance, per block: the format the plan resolved for it,
        // the activation form that format's seam ends in, and the catalogue body that decodes it.
        // A format no claimed body decodes stays without a body here, which the scan shows.
        let mut operands = BTreeMap::new();
        for stack in plan.stacks() {
            for block in &stack.blocks {
                for instance in &block.parts {
                    let Some(stored) = plan.stored_format(instance.part, Some(block.index)) else {
                        continue;
                    };
                    let claimed = symbols
                        .get(&(Some(block.index), instance.part))
                        .or_else(|| symbols.get(&(None, instance.part)))
                        .map(Vec::as_slice)
                        .unwrap_or(&[]);
                    let operand = crate::operand::Operand {
                        stored,
                        activation: crate::operand::ActivationForm::of(stored)
                            .unwrap_or(crate::operand::ActivationForm::Bf16Rows),
                        tensor_wide: stored.tensor_wide(),
                        body: crate::operand::choose(
                            instance.part,
                            stored,
                            claimed,
                            &sparkflash_ops::parts::built_format,
                        ),
                    };
                    operands.insert((Some(block.index), instance.part), operand);
                }
            }
        }
        let tap_layers = plan
            .stacks()
            .iter()
            .find(|stack| stack.kind == sparkflash_model::StackKind::Draft)
            .and_then(|stack| {
                stack.blocks.iter().find_map(|block| {
                    block
                        .parts
                        .iter()
                        .find_map(|part| match part.facts.get("tap_layers") {
                            Some(FactValue::Integers(values)) => Some(
                                values
                                    .iter()
                                    .filter_map(|value| usize::try_from(*value).ok())
                                    .collect(),
                            ),
                            _ => None,
                        })
                })
            })
            .unwrap_or_default();
        let steps = language_steps(&plan)?;
        Ok(Self {
            plan,
            values,
            tensors,
            claims,
            symbols,
            operands,
            tap_layers,
            steps,
        })
    }

    /// How the rows path runs language block `block`: the fused halves, or its parts one at a
    /// time. A block the plan does not have is the fused one's refusal, not a default.
    ///
    /// # Errors
    /// The plan has no language block `block`.
    pub fn block_steps(&self, block: usize) -> Result<&BlockSteps, ConfigError> {
        self.steps.get(block).ok_or_else(|| {
            format!(
                "forward: block {block} is past the plan's {} language blocks",
                self.steps.len()
            )
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

    /// One part instance's operand, when the plan states its format.
    #[must_use]
    pub fn operand(&self, block: Option<usize>, part: Part) -> Option<&crate::operand::Operand> {
        self.operands.get(&(block, part))
    }

    /// The catalogue symbols of one part instance, in catalogue order.
    ///
    /// The plan groups the image's symbols by part, not by block: one entry serves every block
    /// of that part. A block-keyed list, when the plan states one, answers for its block;
    /// otherwise the part's list does.
    #[must_use]
    pub fn symbols(&self, block: Option<usize>, part: Part) -> &[&'static str] {
        self.symbols
            .get(&(block, part))
            .or_else(|| self.symbols.get(&(None, part)))
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

    /// A part instance's symbol in the role `prefix` names whose build decodes `format`.
    ///
    /// The format key is the one the build recorded for the symbol
    /// ([sparkflash_ops::parts::built_format]), so two entries of one role that read different
    /// arithmetic (NVFP4's two conventions, FP8's two scale dtypes) are told apart by it, never
    /// by their names.
    ///
    /// # Errors
    /// The part instance claims no symbol of that role decoding the format.
    pub fn symbol_decoding(
        &self,
        block: Option<usize>,
        part: Part,
        prefix: &str,
        format: sparkflash_model::format::StoredFormat,
    ) -> Result<&'static str, ConfigError> {
        self.symbols(block, part)
            .iter()
            .copied()
            .find(|symbol| {
                symbol.starts_with(prefix)
                    && sparkflash_ops::parts::built_format(symbol)
                        .is_some_and(|built| built.same_arithmetic(format))
            })
            .ok_or_else(|| {
                format!(
                    "the plan's {part:?} part instance at {block:?} claims no `{prefix}*` symbol \
                     decoding {}",
                    format.describe()
                )
            })
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
    /// The non-split MXFP8 dense fallback body.
    pub dense_fallback: &'static str,
    /// The untiled MXFP8 wo_a fallback body.
    pub wo_a_fallback: &'static str,
    /// The non-sequence CSA2 attention fallback.
    pub flash_mla_fallback: &'static str,
    /// The pre-padded A8 routed-expert combine fallback.
    pub moe_fallback: &'static str,
    /// The device head reduction beside the certified screen.
    pub head_reduce: &'static str,
    /// The compressor's ratio-axis pooling cubin.
    pub pool_group: &'static str,
    /// The stream dtype conversion cubin (narrow and widen entries).
    pub engram_stream_convert: &'static str,
    /// The Engram gate write cubin.
    pub engram_write: &'static str,
    /// The routed-expert staging/combine cubin.
    pub moe_routed_device: &'static str,
}

/// Artifact stems whose entry symbol is not the cubin's name.
///
/// The catalogue names entry symbols, while the pinned loader addresses a cubin by its stem.
/// Four staged cubins export a symbol of a different name; the table is the one place that
/// records the difference, and [artifact_stem] still requires the plan to claim the symbol.
const CUBIN_STEMS: &[(&str, &str)] = &[
    ("pool_group_f32", "pool_group_sm121"),
    ("engram_narrow_bf16", "engram_stream_convert_sm121"),
    ("engram_write_bf16", "engram_write_sm121"),
    ("moe_routed_combine_sm121", "moe_routed_device_sm121"),
];

/// The cubin stem for a symbol the plan's catalogue claims under `part`.
fn artifact_stem(part: Part, name: &'static str) -> Result<&'static str, ConfigError> {
    let symbol = description()
        .symbols(None, part)
        .iter()
        .copied()
        .find(|symbol| *symbol == name)
        .ok_or_else(|| {
            format!(
                "the plan's {part:?} catalogue entry does not claim {name} (it claims {:?})",
                description().symbols(None, part)
            )
        })?;
    Ok(CUBIN_STEMS
        .iter()
        .find_map(|(known, stem)| (*known == symbol).then_some(*stem))
        .unwrap_or(symbol))
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
            dense_fallback: artifact_stem(Part::DenseGemm, "b12x_mxfp8_dense_sm121")?,
            wo_a_fallback: artifact_stem(Part::DenseGemm, "b12x_mxfp8_wo_a_sm121")?,
            flash_mla_fallback: artifact_stem(Part::MlaAttention, "flash_mla_csa2_sm121")?,
            moe_fallback: artifact_stem(Part::Moe, "b12x_mxfp4_moe_sm121")?,
            head_reduce: artifact_stem(Part::LmHead, "lm_head_reduce_sm121")?,
            pool_group: artifact_stem(Part::Csa2Compression, "pool_group_f32")?,
            engram_stream_convert: artifact_stem(Part::Engram, "engram_narrow_bf16")?,
            engram_write: artifact_stem(Part::Engram, "engram_write_bf16")?,
            moe_routed_device: artifact_stem(Part::MoeCombine, "moe_routed_combine_sm121")?,
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
    let installed = *HANDLES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(handles) = installed {
        return handles;
    }
    // A unit test that exercises a refusal or a diagnostic does not open a checkpoint. The
    // released description is still the one whose symbols and catalogue positions it names, so
    // install that plan once rather than reintroducing a symbol literal for the test's sake.
    #[cfg(any(test, feature = "test-support"))]
    {
        if let Ok(plan) = EnginePlan::from_json(
            include_str!(
                "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
            ),
            Target::Sm121,
            4,
            0,
        ) {
            let _ = install(plan);
            if let Some(handles) = *HANDLES
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
            {
                return handles;
            }
        }
    }
    panic!("forward open: no kernel handles installed")
}

// The geometry the per-token code reads on every call. Set once at install, read without a
// lock: a mutex on the crate::plan::hidden()/head/vocabulary path would be a decode-step cost.
static DIM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static VOCAB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static HC_MULT_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static MLA_LATENT_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static N_LAYERS_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static N_HEADS_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static INDEX_TOPK_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static WINDOW_SIZE_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ENCODER_LAYERS_VALUE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static MOE_INTER_DIM_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DSPARK_BLOCK_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static N_MTP_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DSPARK_EXPERTS_VALUE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static DSPARK_TOP_K_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static DSPARK_MARKOV_RANK_VALUE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static WORLD_VALUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn publish_geometry(values: &ModelConfig) {
    use std::sync::atomic::Ordering;
    DIM.store(values.dim, Ordering::Relaxed);
    VOCAB.store(values.vocab_size, Ordering::Relaxed);
    HC_MULT_VALUE.store(values.hc_mult, Ordering::Relaxed);
    MLA_LATENT_VALUE.store(values.head_dim, Ordering::Relaxed);
    N_LAYERS_VALUE.store(values.n_layers, Ordering::Relaxed);
    N_HEADS_VALUE.store(values.n_heads, Ordering::Relaxed);
    INDEX_TOPK_VALUE.store(values.index_topk, Ordering::Relaxed);
    WINDOW_SIZE_VALUE.store(values.window_size, Ordering::Relaxed);
    // The encoder half ends at the last layer that owns compressed KV; the stack's own fact uses
    // the same rule, and the released owners are layers 2, 8, 14 and 20.
    let encoder_layers = values
        .kv_source_layers
        .iter()
        .copied()
        .max()
        .map_or(values.n_layers / 2, |last| last + 1)
        .clamp(1, values.n_layers.max(1));
    ENCODER_LAYERS_VALUE.store(encoder_layers, Ordering::Relaxed);
    MOE_INTER_DIM_VALUE.store(values.moe_inter_dim, Ordering::Relaxed);
    DSPARK_BLOCK_VALUE.store(values.dspark_block_size, Ordering::Relaxed);
    N_MTP_VALUE.store(values.n_mtp_layers, Ordering::Relaxed);
    DSPARK_EXPERTS_VALUE.store(values.dspark_n_routed_experts, Ordering::Relaxed);
    DSPARK_TOP_K_VALUE.store(values.dspark_n_activated_experts, Ordering::Relaxed);
    DSPARK_MARKOV_RANK_VALUE.store(values.dspark_markov_rank, Ordering::Relaxed);
}

/// The world size this process's plan was opened at.
///
/// The plan's split is the one authority for every per-rank width: a model runs at the worlds its
/// scan resolves, and a world the kernels were not built for is a row the scan shows.
#[must_use]
pub fn world() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&WORLD_VALUE, std::sync::atomic::Ordering::Relaxed).max(1)
}

/// The residual stream width, read at open from the plan.
#[must_use]
pub fn hidden() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&DIM, std::sync::atomic::Ordering::Relaxed)
}

/// The vocabulary size, read at open from the plan.
#[must_use]
pub fn vocab() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&VOCAB, std::sync::atomic::Ordering::Relaxed)
}

/// One rank's vocabulary shard, computed at open from the plan's vocabulary and the world the
/// plan resolved at.
#[must_use]
pub fn vocab_per_rank() -> usize {
    vocab() / world()
}

/// The mHC stream count, read at open from the plan.
#[must_use]
pub fn hc_mult() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&HC_MULT_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The mHC coefficient rows, computed at open from the plan's stream count.
#[must_use]
pub fn predictor_rows() -> usize {
    let hc = hc_mult();
    hc * hc + 2 * hc
}

/// The MLA latent width, read at open from the plan's attention head width.
#[must_use]
pub fn mla_latent() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&MLA_LATENT_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The backbone layer count, read at open from the plan.
#[must_use]
pub fn n_layers() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&N_LAYERS_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The attention head count over the whole model, read at open.
#[must_use]
pub fn n_heads() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&N_HEADS_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The compressed-position select bound, read at open from the plan's `index_topk`.
#[must_use]
pub fn index_topk() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&INDEX_TOPK_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The sliding-window width, read at open from the plan.
#[must_use]
pub fn window_size() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&WINDOW_SIZE_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The layers that run the whole prompt (the encoder half), read at open.
#[must_use]
pub fn encoder_layers() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(
        &ENCODER_LAYERS_VALUE,
        std::sync::atomic::Ordering::Relaxed,
    )
}

/// The expert intermediate width before the rank split, read at open.
#[must_use]
pub fn moe_inter_dim() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&MOE_INTER_DIM_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The DSpark draft block size, read at open from the plan.
#[must_use]
pub fn dspark_block_size() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&DSPARK_BLOCK_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The DSpark MTP stage count, read at open from the plan.
#[must_use]
pub fn n_mtp_layers() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&N_MTP_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The DSpark draft block's depth: the profile's stated depth, or the plan's block.
///
/// The block is the plan's own fact (5 on the released model, 2 on the synthetic second shape).
/// A default that fell back to a released literal would ask a second shape for a depth its plan
/// never declared, which is the defect this is the one authority for.
///
/// # Errors
/// The stated depth is zero or deeper than the plan's block.
pub fn draft_depth(stated: Option<usize>) -> Result<usize, String> {
    let block = dspark_block_size();
    let depth = stated.unwrap_or(block);
    if depth == 0 || depth > block {
        return Err(format!("SPARKFLASH_DSPARK_K={depth}: needs 1..={block}"));
    }
    Ok(depth)
}

/// The DSpark drafter's routed expert count, read at open from the plan.
#[must_use]
pub fn dspark_n_routed_experts() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(
        &DSPARK_EXPERTS_VALUE,
        std::sync::atomic::Ordering::Relaxed,
    )
}

/// The DSpark drafter's activated expert count (its top-k), read at open from the plan.
#[must_use]
pub fn dspark_n_activated_experts() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(&DSPARK_TOP_K_VALUE, std::sync::atomic::Ordering::Relaxed)
}

/// The DSpark Markov head's rank, read at open from the plan.
#[must_use]
pub fn dspark_markov_rank() -> usize {
    ensure_geometry();
    std::sync::atomic::AtomicUsize::load(
        &DSPARK_MARKOV_RANK_VALUE,
        std::sync::atomic::Ordering::Relaxed,
    )
}

/// The attention heads this rank runs: the model's heads over the world the plan resolved at.
#[must_use]
pub fn heads_per_rank() -> usize {
    n_heads() / world()
}

/// The most positions one attention row attends: the selected bound plus one full window.
#[must_use]
pub fn max_attended() -> usize {
    index_topk() + window_size()
}

/// The selected list's row stride in ids: one selected entry per compressed position the select
/// may keep, which is the plan's `index_topk`.
#[must_use]
pub fn selected_stride() -> usize {
    index_topk()
}

fn ensure_geometry() {
    if DIM.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        let _ = cached_values();
    }
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
    publish_geometry(&description.values);
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
    #[cfg(any(test, feature = "test-support"))]
    {
        // The vendored released document is the plan's own input, so a dependent crate's unit
        // test resolves the released description without the 510 GB checkpoint on disk.
        let plan = EnginePlan::from_json(
            include_str!(
                "../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
            ),
            Target::Sm121,
            4,
            0,
        )
        .expect("forward open: the vendored released configuration plans");
        install(plan).expect("forward open: the released plan installs")
    }
    #[cfg(not(any(test, feature = "test-support")))]
    panic!(
        "forward open: no plan installed, so no description names the tensors or kernels this          stage reads"
    )
}

/// The installed plan's operand for one part instance.
///
/// The format, activation form and body were decided at open, per block and part. A caller that
/// needs to know what a part's tensors are stored as asks here rather than comparing a dtype word
/// or a byte count of a tensor it read.
///
/// # Panics
/// No description is installed. The open path installs before any stage reads a part.
#[must_use]
pub fn operand(block: Option<usize>, part: Part) -> &'static crate::operand::Operand {
    description().operand(block, part).unwrap_or_else(|| {
        panic!("forward open: the plan states no format for {part:?} at {block:?}")
    })
}

/// The stored format the plan states for one layer's shared expert.
///
/// The shared expert's tensors are claimed by the MoE part (`ffn.shared_experts.w1/w3/w2`), but
/// they are a dense-class MLP, so neither the routed-experts aggregate nor a dense-MLP part the
/// layer does not carry speaks for them. The plan collects the format from the layout's own
/// `shared_expert` role.
///
/// # Errors
/// The layer has no shared expert, or the plan's layout states no format for its tensors. The
/// refusal names the layer; it never falls back to a neighbouring class.
pub fn shared_expert_format(
    block: usize,
) -> Result<sparkflash_model::format::StoredFormat, ConfigError> {
    description()
        .plan()
        .shared_expert_format(block)
        .ok_or_else(|| {
            format!("the plan states no stored format for layer {block}'s shared expert")
        })
}

/// The gated MLP a layer's FFN runs on its dense projections: a MoE block's shared expert beside
/// its routed experts, or a dense-FFN block's one MLP in their place.
#[derive(Debug, Clone, Copy)]
pub struct LayerMlp {
    /// The claim segment its `w1`/`w2`/`w3` live under (`ffn.shared_experts`, or `ffn`).
    pub segment: &'static str,
    /// Its whole intermediate width, every rank's slice together.
    pub inter: usize,
    /// Its stored format.
    pub format: sparkflash_model::format::StoredFormat,
    /// Whether it is a dense-FFN block's MLP (no router and no routed experts beside it).
    pub dense: bool,
}

/// Layer `block`'s gated MLP ([LayerMlp]).
///
/// # Errors
/// The plan states no format for it, by layer.
pub fn layer_mlp(block: usize) -> Result<LayerMlp, ConfigError> {
    let values = description().config();
    if block_has_part(block, Part::DenseMlp) {
        let format = description()
            .plan()
            .stored_format(Part::DenseMlp, Some(block))
            .ok_or_else(|| {
                format!("the plan states no stored format for layer {block}'s dense MLP")
            })?;
        return Ok(LayerMlp {
            segment: "ffn",
            inter: values.dense_inter_dim,
            format,
            dense: true,
        });
    }
    Ok(LayerMlp {
        segment: "ffn.shared_experts",
        inter: values.moe_inter_dim * values.n_shared_experts.max(1),
        format: shared_expert_format(block)?,
        dense: false,
    })
}

/// The widest gated MLP of the stack ([layer_mlp]), whole: what a buffer every layer's MLP shares
/// is sized by.
#[must_use]
pub fn widest_mlp() -> usize {
    let values = description().config();
    (values.moe_inter_dim * values.n_shared_experts.max(1)).max(values.dense_inter_dim)
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

/// A claim under one MTP stage, by the `mtp.<stage>.<segment>` name the plan uses.
///
/// The draft stack's blocks do not carry one stage each (the Markov head is claimed on the
/// first draft block while its name says stage two), so the lookup is by name, not by block
/// index. The name is the identifier the checkpoint and the plan share.
fn mtp_name(stage: usize, tail: &str) -> Result<&'static str, String> {
    let description = description();
    let prefix = format!("mtp.{stage}.");
    let suffix = format!(".{tail}");
    for stack in description.plan.stacks() {
        if stack.kind != sparkflash_model::StackKind::Draft {
            continue;
        }
        for block in &stack.blocks {
            for part in &block.parts {
                for claim in &part.claims {
                    if claim.name.starts_with(&prefix) && claim.name.ends_with(&suffix) {
                        return Ok(claim.name.as_str());
                    }
                }
            }
        }
    }
    Err(format!(
        "forward open: no MTP stage {stage} claims a tensor ending .{tail}"
    ))
}

/// The MTP stage prefix a claim under `segment` lives at, e.g. `mtp.1.attn`.
///
/// A pair of tensors (a weight and its scale) shares a prefix; a caller that needs both derives
/// them from this prefix rather than asking the plan for a name that does not exist.
#[must_use]
pub fn mtp_prefix(stage: usize, segment: &str) -> &'static str {
    let description = description();
    let prefix = format!("mtp.{stage}.");
    let needle = format!(".{segment}.");
    for stack in description.plan.stacks() {
        if stack.kind != sparkflash_model::StackKind::Draft {
            continue;
        }
        for block in &stack.blocks {
            for part in &block.parts {
                for claim in &part.claims {
                    let name = claim.name.as_str();
                    if !name.starts_with(&prefix) {
                        continue;
                    }
                    if let Some(at) = name.find(&needle) {
                        return &name[..at + 1 + segment.len()];
                    }
                }
            }
        }
    }
    panic!("forward open: MTP stage {stage} claims no tensor under .{segment}.");
}

/// One MTP stage's tensor, by the tail its checkpoint name carries, or the refusal.
///
/// This is the form a caller uses while it can still report: the refusal names the stage and the
/// tail instead of panicking at open.
///
/// # Errors
/// The stage claims no tensor whose name ends with `.{tail}`.
pub fn mtp_claim_result(stage: usize, tail: &str) -> Result<&'static str, String> {
    mtp_name(stage, tail)
}

/// The DSpark stage the draft block's last-stage tensors are claimed on.
///
/// The Markov head, the head norm and the Markov embedding belong to the last DSpark stage the
/// plan states, exactly as the host reference derives them from `n_stages - 1`. The stage is a
/// plan fact: the released model has three stages and the synthetic second shape has two, so a
/// released literal would ask for a stage the checkpoint does not carry.
///
/// # Errors
/// The plan states no DSpark stage.
pub fn mtp_last_stage() -> Result<usize, String> {
    n_mtp_layers()
        .checked_sub(1)
        .ok_or_else(|| "forward open: the plan states no DSpark stage".to_owned())
}

/// One MTP stage's tensor, by the tail its checkpoint name carries.
///
/// # Panics
/// The stage claims no such tensor. The open path calls this only after it has validated the
/// names it will ask for; a caller that can report uses [mtp_claim_result].
#[must_use]
pub fn mtp_claim(stage: usize, tail: &str) -> &'static str {
    mtp_name(stage, tail).unwrap_or_else(|error| panic!("{error}"))
}

/// Install the released checkpoint's plan for `rank` of `world`.
///
/// # Errors
/// No checkpoint, or a plan the forward refuses.
#[cfg(any(test, feature = "test-support"))]
pub fn install_released(world: usize, rank: usize) -> Result<&'static Description, ConfigError> {
    install(plan_from_checkpoint(world, rank)?)
}

/// The description's language block for one layer.
#[must_use]
pub fn language_block(layer: usize) -> Option<&'static sparkflash_model::Block> {
    let description = description();
    description
        .plan
        .stacks()
        .iter()
        .find(|stack| stack.kind == sparkflash_model::StackKind::Language)?
        .blocks
        .iter()
        .find(|block| block.index == layer)
}

/// Whether a language block carries a part instance.
#[must_use]
pub fn block_has_part(layer: usize, part: Part) -> bool {
    language_block(layer).is_some_and(|block| block.parts.iter().any(|p| p.part == part))
}

/// One fact a language block's part instances carry.
#[must_use]
pub fn layer_fact(layer: usize, name: &str) -> Option<&'static FactValue> {
    language_block(layer)?
        .parts
        .iter()
        .find_map(|part| part.facts.get(name))
}

/// One boolean fact a language block's part instances carry.
#[must_use]
pub fn layer_fact_bool(layer: usize, name: &str) -> bool {
    matches!(layer_fact(layer, name), Some(FactValue::Bool(true)))
}

/// The compression ratio of one layer, read from that block's part fact.
#[must_use]
pub fn layer_compress_ratio(layer: usize) -> Option<usize> {
    layer_fact(layer, "compress_ratio")
        .and_then(FactValue::as_integer)
        .and_then(|ratio| usize::try_from(ratio).ok())
        .filter(|ratio| *ratio > 0)
}

/// The layers whose block carries a fact switched on, in layer order.
#[must_use]
pub fn layers_with_fact(name: &str, value: bool) -> Vec<usize> {
    let description = description();
    let Some(stack) = description
        .plan
        .stacks()
        .iter()
        .find(|stack| stack.kind == sparkflash_model::StackKind::Language)
    else {
        return Vec::new();
    };
    stack
        .blocks
        .iter()
        .filter(|block| {
            block
                .parts
                .iter()
                .find_map(|part| part.facts.get(name))
                .is_some_and(|fact| matches!(fact, FactValue::Bool(flag) if *flag == value))
        })
        .map(|block| block.index)
        .collect()
}

/// The candidate-pool source layer, read from the blocks' part facts.
#[must_use]
pub fn candidate_source_layer() -> usize {
    let description = description();
    description
        .plan
        .stacks()
        .iter()
        .find(|stack| stack.kind == sparkflash_model::StackKind::Language)
        .and_then(|stack| {
            stack.blocks.iter().find_map(|block| {
                block.parts.iter().find_map(|part| {
                    part.facts
                        .get("candidate_source")
                        .and_then(FactValue::as_integer)
                        .and_then(|layer| usize::try_from(layer).ok())
                })
            })
        })
        .unwrap_or(description.values.candidate_source_layer)
}

/// The tap layers the draft stage reads, read from the draft block's part fact.
#[must_use]
pub fn tap_layers() -> Vec<usize> {
    tap_layers_ref().to_vec()
}

/// The tap layers as the open description holds them, for geometry that outlives one call.
#[must_use]
pub fn tap_layers_ref() -> &'static [usize] {
    &description().tap_layers
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
            | Part::DenseMlp
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

/// How the rows path runs one language block, decided at open from the block's parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockSteps {
    /// The block's parts are the ones the base's four fused halves cover (the attention seam and
    /// rest, the FFN seam and rest): the same launches and the same trace as a released block.
    Fused,
    /// Any other block: the pending residual lands, then each part runs on its own, in plan
    /// order, its delta folded back before the next one reads the rows.
    Parts(Vec<Part>),
}

/// The parts the fused halves cover. A block of only these, carrying both the attention and the
/// MoE part, runs [BlockSteps::Fused].
const FUSED_PARTS: [Part; 8] = [
    Part::RmsNorm,
    Part::HyperConnection,
    Part::ManifoldHyperConnection,
    Part::MlaAttention,
    Part::Csa2Compression,
    Part::SparseIndexer,
    Part::Moe,
    Part::Engram,
];

/// Whether the base runs `part` on its own, outside the fused halves: the MLA attention (its
/// entry seam, its sublayers and its post) and the dense gated MLP (its seam, the shared
/// [sparkflash_ops::roles::GatedMlp] body and its post).
#[must_use]
pub const fn runs_alone(part: Part) -> bool {
    matches!(part, Part::MlaAttention | Part::DenseMlp)
}

/// The sublayers a part that runs alone carries inside its own body: the norms and the stream
/// mixers (every part's seam and post), the compression and the indexer (the attention's), and
/// the Engram write (the attention seam's).
const fn absorbed(part: Part) -> bool {
    matches!(
        part,
        Part::RmsNorm
            | Part::HyperConnection
            | Part::ManifoldHyperConnection
            | Part::Csa2Compression
            | Part::SparseIndexer
            | Part::Engram
    )
}

/// Each language block's steps, in block order.
///
/// # Errors
/// A block carries a part the forward has no body for, or a block that is not fused carries a
/// part that does not run on its own; the message names the block and the part.
pub fn language_steps(plan: &EnginePlan) -> Result<Vec<BlockSteps>, ConfigError> {
    let mut steps = Vec::new();
    for stack in plan.stacks() {
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
            let has = |part| block.parts.iter().any(|instance| instance.part == part);
            let fused = has(Part::MlaAttention)
                && has(Part::Moe)
                && block
                    .parts
                    .iter()
                    .all(|instance| FUSED_PARTS.contains(&instance.part));
            if fused {
                steps.push(BlockSteps::Fused);
                continue;
            }
            // The fused attention-post/FFN seam spans the attention part and the MoE part, and a
            // compression or indexer sublayer belongs to the attention part; outside the halves
            // each part must run alone, and the first one that cannot names the refusal.
            let mut parts = Vec::new();
            for instance in &block.parts {
                let sublayer = matches!(instance.part, Part::Csa2Compression | Part::SparseIndexer);
                if absorbed(instance.part) && (!sublayer || has(Part::MlaAttention)) {
                    continue;
                }
                if !runs_alone(instance.part) {
                    let why = match instance.part {
                        Part::Moe if !has(Part::MlaAttention) => {
                            "the MoE part runs only inside the fused attention/FFN seam, which needs the MlaAttention part beside it"
                        }
                        Part::Csa2Compression | Part::SparseIndexer if !has(Part::MlaAttention) => {
                            "the sublayer belongs to an MlaAttention part the block does not carry"
                        }
                        _ => {
                            "the base runs it only inside the fused halves, and this block is not one they cover"
                        }
                    };
                    return Err(format!(
                        "forward open: block {} carries the {:?} part: {why}",
                        block.index, instance.part
                    ));
                }
                parts.push(instance.part);
            }
            steps.push(BlockSteps::Parts(parts));
        }
    }
    Ok(steps)
}

/// Refuse, at open, a language block part the forward has no body for.
///
/// # Errors
/// As [language_steps].
pub fn check_language_parts(description: &Description) -> Result<(), ConfigError> {
    language_steps(description.plan()).map(|_| ())
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
    fn the_open_prefixes_resolve_to_the_plans_claims() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        let claimed = |block: usize, tail: &str| {
            let suffix = format!(".{tail}");
            description.claims.iter().any(|((candidate, _), names)| {
                *candidate == block && names.iter().any(|name| name.ends_with(&suffix))
            })
        };
        assert_eq!(
            layer_prefix(0, "ffn.shared_experts"),
            "layers.0.ffn.shared_experts"
        );
        for leaf in ["w1", "w3", "w2"] {
            for suffix in ["weight", "scale"] {
                let tail = format!("ffn.shared_experts.{leaf}.{suffix}");
                assert!(claimed(0, &tail), "layer 0 claims no {tail}");
            }
        }
        assert_eq!(layer_prefix(1, "engram"), "layers.1.engram");
        assert!(claimed(1, "engram.q_weight"));
        assert_eq!(
            mtp_claim(2, "markov_head.head.weight"),
            "mtp.2.markov_head.head.weight"
        );
        assert!(mtp_prefix(0, "attn.wq_a").ends_with(".attn.wq_a"));
        for leaf in ["w1", "w3", "w2"] {
            let tail = format!("ffn.experts.0.{leaf}.weight");
            assert_eq!(mtp_claim(0, &tail), format!("mtp.0.{tail}"));
        }
        for tail in [
            "attn.wq_a.weight",
            "attn.wq_a.scale",
            "attn.wq_b.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn.wkv.weight",
            "ffn.shared_experts.w1.weight",
            "ffn.shared_experts.w3.weight",
            "ffn.shared_experts.w2.weight",
        ] {
            assert_eq!(mtp_claim(0, tail), format!("mtp.0.{tail}"));
        }
    }

    #[test]
    fn the_released_plan_names_every_compressor_tensor_it_binds() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        let map = crate::resident_weights::rank_attention_plan_shapes(description.values(), 0, 4)
            .expect("the released plan's attention shapes");
        assert!(
            map.keys()
                .any(|name| name.ends_with("attn.compressor.wkv.weight"))
        );
    }

    #[test]
    fn the_released_plan_owns_kv_on_the_source_layers_only() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        let owners = layers_with_fact("kv_source", true);
        assert_eq!(
            owners,
            vec![2, 8, 14, 20],
            "the kv-source fact must mark the four layers that own the compressed cache"
        );
        for layer in owners {
            description
                .layer_tensor(layer, Part::Csa2Compression, "attn.compressor.wkv.weight")
                .unwrap_or_else(|error| panic!("layer {layer}: {error}"));
        }
        // A layer that shares an owner's cache has no compressor tensor of its own; the seam
        // stages only the owners and must not ask the plan for a name it never claimed.
        let shared = description
            .layer_tensor(3, Part::Csa2Compression, "attn.compressor.wkv.weight")
            .expect_err("layer 3 shares layer 2's compressed cache");
        assert!(shared.contains("claims no tensor"), "{shared}");
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
    fn the_per_layer_branches_read_part_facts() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        let values = description.values();
        assert_eq!(layers_with_fact("kv_source", true), values.kv_source_layers);
        assert_eq!(
            layers_with_fact("index_source", true),
            values.index_source_layers
        );
        assert_eq!(
            layers_with_fact("engram_member", true),
            values.engram_layer_ids
        );
        for layer in 0..values.n_layers {
            assert_eq!(
                layer_compress_ratio(layer),
                values.compress_ratios[layer]
                    .checked_sub(0)
                    .filter(|r| *r > 0),
                "layer {layer}"
            );
            assert!(block_has_part(layer, Part::MlaAttention), "layer {layer}");
            assert!(block_has_part(layer, Part::Moe), "layer {layer}");
            assert!(block_has_part(layer, Part::RmsNorm), "layer {layer}");
        }
        assert_eq!(candidate_source_layer(), values.candidate_source_layer);
        assert!(!tap_layers().is_empty());
    }

    #[test]
    fn the_released_plan_drives_the_forward_geometry() {
        let Some(description) = released() else {
            eprintln!("skipping: the released checkpoint is not on this host");
            return;
        };
        let values = description.values();
        assert_eq!(n_layers(), values.n_layers);
        assert_eq!(hidden(), values.dim);
        assert_eq!(vocab(), values.vocab_size);
        assert_eq!(n_heads(), values.n_heads);
        assert_eq!(heads_per_rank(), values.n_heads / 4);
        assert_eq!(selected_stride(), values.index_topk);
        assert_eq!(max_attended(), values.index_topk + values.window_size);
        assert_eq!(encoder_layers(), 21);
        assert_eq!(moe_inter_dim(), values.moe_inter_dim);
        assert_eq!(moe_inter_dim() / 4, 576);
        assert_eq!(dspark_block_size(), values.dspark_block_size);
        assert_eq!(n_mtp_layers(), values.n_mtp_layers);
        assert_eq!(dspark_n_routed_experts(), values.dspark_n_routed_experts);
        assert_eq!(
            dspark_n_activated_experts(),
            values.dspark_n_activated_experts
        );
        assert_eq!(dspark_markov_rank(), values.dspark_markov_rank);
        // The drafter's compile-time-sized arrays are these open values: the released run
        // allocates exactly the released five-row block, three stages and 128/3 experts.
        assert_eq!(dspark_block_size(), 5);
        assert_eq!(n_mtp_layers(), 3);
        assert_eq!(dspark_n_routed_experts(), 128);
        assert_eq!(dspark_n_activated_experts(), 3);
        assert_eq!(dspark_markov_rank(), 256);
        assert_eq!(tap_layers(), vec![37, 38, 39]);
        assert_eq!(&*description.tap_layers, &[37, 38, 39][..]);
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
