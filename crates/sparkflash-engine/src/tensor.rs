//! The tensor plan: the name, dtype and shape of every tensor a forward pass reads.
//!
//! The plan is derived from the architecture, not transcribed from the checkpoint, and then checked
//! against the checkpoint's index. That order matters: a plan transcribed from the index would be a
//! copy with no opinion, while a plan derived from the config can *disagree* with the index, and the
//! disagreement is the interesting event. The integration test `the_plan_names_are_the_released_index_names`
//! asserts the two name sets are equal over the language-model namespaces; the arithmetic below
//! reproduces the index's own counts exactly (93,415 under `layers.*`, 2,401 under `mtp.*`, and
//! 95,819 language-model tensors out of 96,085, the remaining 266 being the vision tower, the
//! aligner and the three image delimiters, which this crate does not plan yet).
//!
//! Three rules decide a tensor's format, all from docs/architecture/03-moe.md:
//!
//! 1. A dense Linear or the shared expert is FP8 E4M3 with a 2-D UE8M0 scale grid.
//! 2. A routed or MTP expert is FP4 E2M1 with a per-row scale grid.
//! 3. Anything the index stores without a `.scale` companion is not block-scaled. The index is the
//!    evidence: it has no `ffn.gate.scale`, no `compressor.*.scale`, no `indexer.weights_proj.scale`
//!    and no `wo_a`-style exception, so those tensors are BF16 here.
//!
//! The plan is not merely count-compatible with the checkpoint: it is byte-exact. The plan plus the
//! image path the language-model plan leaves out (970,536,960 bytes, derived from the vision and
//! aligner constructors) sums to the released `metadata.total_size` exactly, 510,286,023,000. That
//! identity is what caught the one dtype this plan had wrong - the Engram gate weights, see
//! `push_engram` - and it is why the plan asserts its byte total as an exact number rather than as a
//! tolerance.
//!
//! Three documents disagreed about the gate weight (docs/architecture/03-moe.md called it FP8); the
//! index has no `ffn.gate.scale` and model.py:811 runs the gate through the plain fp32 linear path,
//! so it is not block-quantised. That correction is in docs/architecture/00-overview.md and
//! 03-moe.md, and the byte identity confirms it.

use std::collections::BTreeMap;

use crate::config::ModelConfig;
use crate::dtype::{DType, dense_scale_shape, expert_scale_shape};

/// One tensor's entry in the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// The name exactly as the checkpoint spells it.
    pub name: String,
    /// The element format.
    pub dtype: DType,
    /// The logical shape. For an FP4 weight this is the shape the arithmetic sees, twice the last
    /// dimension of the bytes on disk.
    pub shape: Vec<usize>,
}

impl TensorSpec {
    /// The number of logical elements.
    pub fn elements(&self) -> u128 {
        self.shape.iter().map(|&d| d as u128).product()
    }

    /// The number of bytes this tensor occupies, packing included.
    pub fn storage_bytes(&self) -> u128 {
        self.dtype.storage_bytes(self.elements() as usize) as u128
    }
}

/// The tensor plan for one configuration.
#[derive(Debug, Clone, Default)]
pub struct TensorPlan {
    specs: Vec<TensorSpec>,
    by_name: BTreeMap<String, usize>,
}

impl TensorPlan {
    /// Plan every tensor the language-model forward reads, in a stable order.
    pub fn language_model(config: &ModelConfig) -> Self {
        let mut plan = TensorPlan::default();
        plan.push_globals(config);
        for layer in 0..config.n_layers {
            plan.push_block(&format!("layers.{layer}"), config, false, layer);
            plan.push_compressor(&format!("layers.{layer}"), config, layer);
            plan.push_indexer(&format!("layers.{layer}"), config, layer);
            plan.push_engram(&format!("layers.{layer}"), config, layer);
        }
        for stage in 0..config.n_mtp_layers {
            plan.push_block(&format!("mtp.{stage}"), config, true, stage);
            plan.push_mtp_extras(&format!("mtp.{stage}"), config, stage);
        }
        plan
    }

    fn insert(&mut self, name: String, dtype: DType, shape: Vec<usize>) {
        assert!(
            !self.by_name.contains_key(&name),
            "duplicate tensor name {name}"
        );
        self.by_name.insert(name.clone(), self.specs.len());
        self.specs.push(TensorSpec { name, dtype, shape });
    }

    fn dense(&mut self, name: &str, out: usize, r#in: usize) {
        self.insert(format!("{name}.weight"), DType::F8E4M3, vec![out, r#in]);
        self.insert(
            format!("{name}.scale"),
            DType::F8E8M0,
            dense_scale_shape(out, r#in).to_vec(),
        );
    }

    fn expert(&mut self, name: &str, out: usize, r#in: usize) {
        self.insert(format!("{name}.weight"), DType::F4E2M1, vec![out, r#in]);
        self.insert(
            format!("{name}.scale"),
            DType::F8E8M0,
            expert_scale_shape(out, r#in).to_vec(),
        );
    }

    fn plain(&mut self, name: &str, dtype: DType, shape: Vec<usize>) {
        self.insert(name.to_string(), dtype, shape);
    }

    fn push_globals(&mut self, config: &ModelConfig) {
        self.plain(
            "embed.weight",
            DType::BF16,
            vec![config.vocab_size, config.dim],
        );
        self.plain(
            "head.weight",
            DType::BF16,
            vec![config.vocab_size, config.dim],
        );
        self.plain("norm.weight", DType::BF16, vec![config.dim]);
    }

    fn push_block(&mut self, prefix: &str, config: &ModelConfig, mtp: bool, layer: usize) {
        let dim = config.dim;
        let heads = config.n_heads * config.head_dim;
        let o_wide = config.o_groups * config.o_lora_rank;
        let (n_routed, n_activated) = if mtp {
            (
                config.dspark_n_routed_experts,
                config.dspark_n_activated_experts,
            )
        } else {
            (config.n_routed_experts, config.n_activated_experts)
        };
        let _ = n_activated;

        self.dense(&format!("{prefix}.attn.wq_a"), config.q_lora_rank, dim);
        self.dense(&format!("{prefix}.attn.wq_b"), heads, config.q_lora_rank);
        self.dense(&format!("{prefix}.attn.wkv"), config.head_dim, dim);
        // wo_a is per group: each group projects its own slice of the head dimension, so the input
        // width is heads / o_groups (model.py:785-787). The scale grid is the 32x32 one;
        // convert.py:164 also accepts a 128x128 grid for this tensor, and only a real header says
        // which the release uses.
        self.dense(
            &format!("{prefix}.attn.wo_a"),
            o_wide,
            heads / config.o_groups,
        );
        self.dense(&format!("{prefix}.attn.wo_b"), dim, o_wide);
        self.plain(
            &format!("{prefix}.attn.attn_sink"),
            DType::F32,
            vec![config.n_heads],
        );
        self.plain(
            &format!("{prefix}.attn.q_norm.weight"),
            DType::BF16,
            vec![config.q_lora_rank],
        );
        self.plain(
            &format!("{prefix}.attn.kv_norm.weight"),
            DType::BF16,
            vec![config.head_dim],
        );
        self.plain(
            &format!("{prefix}.attn_norm.weight"),
            DType::BF16,
            vec![dim],
        );
        self.plain(&format!("{prefix}.ffn_norm.weight"), DType::BF16, vec![dim]);

        // mHC: mix_hc = (2 + n) * n coefficients, over a stream of n * d values, plus one bias
        // per coefficient and three scalars (pre, post, comb). docs/architecture/02-mhc.md.
        let n = config.hc_mult;
        let mix_hc = (2 + n) * n;
        let hc_dim = n * dim;
        for twin in ["attn", "ffn"] {
            self.plain(
                &format!("{prefix}.hc_{twin}_fn"),
                DType::F32,
                vec![mix_hc, hc_dim],
            );
            self.plain(
                &format!("{prefix}.hc_{twin}_base"),
                DType::F32,
                vec![mix_hc],
            );
            self.plain(&format!("{prefix}.hc_{twin}_scale"), DType::F32, vec![3]);
        }

        // The router weight carries no scale in the index, so it is not block-scaled (see the
        // module documentation): BF16 here, settled against a real header in a later slice.
        self.plain(
            &format!("{prefix}.ffn.gate.weight"),
            DType::BF16,
            vec![n_routed, dim],
        );
        self.plain(
            &format!("{prefix}.ffn.gate.bias"),
            DType::F32,
            vec![n_routed],
        );
        // The image-span routing bias exists only where there is a vision tower: model.py:807 creates
        // it inside the `vision_enabled` branch. A config without vision has no such tensor in the
        // checkpoint, so planning one unconditionally would make the plan demand a tensor that cannot
        // exist. The released configuration has vision; the test configuration deliberately does not,
        // which is what exercises this condition.
        if config.vision_enabled() {
            self.plain(
                &format!("{prefix}.ffn.gate.bias_vl"),
                DType::F32,
                vec![n_routed],
            );
        }

        // The one shared expert is a dense FP8 Linear set, not an FP4 expert set.
        let inter = config.moe_inter_dim * config.n_shared_experts;
        self.dense(&format!("{prefix}.ffn.shared_experts.w1"), inter, dim);
        self.dense(&format!("{prefix}.ffn.shared_experts.w2"), dim, inter);
        self.dense(&format!("{prefix}.ffn.shared_experts.w3"), inter, dim);

        for expert in 0..n_routed {
            let base = format!("{prefix}.ffn.experts.{expert}");
            self.expert(&format!("{base}.w1"), inter_for_expert(config), dim);
            self.expert(&format!("{base}.w2"), dim, inter_for_expert(config));
            self.expert(&format!("{base}.w3"), inter_for_expert(config), dim);
        }
        let _ = layer;
    }

    fn push_compressor(&mut self, prefix: &str, config: &ModelConfig, layer: usize) {
        if !config.kv_source_layers.contains(&layer) {
            return;
        }
        // No `.scale` companion exists for any compressor tensor, so these are BF16.
        self.plain(
            &format!("{prefix}.attn.compressor.wkv.weight"),
            DType::BF16,
            vec![config.head_dim, config.dim],
        );
        self.plain(
            &format!("{prefix}.attn.compressor.norm.weight"),
            DType::BF16,
            vec![config.head_dim],
        );
        let ratio = config.compress_ratios.get(layer).copied().unwrap_or(0);
        if ratio > 1 {
            self.plain(
                &format!("{prefix}.attn.compressor.wgate.weight"),
                DType::BF16,
                vec![config.head_dim, config.dim],
            );
        }
    }

    fn push_indexer(&mut self, prefix: &str, config: &ModelConfig, layer: usize) {
        if !config.index_source_layers.contains(&layer) {
            return;
        }
        self.dense(
            &format!("{prefix}.attn.indexer.wq_b"),
            config.index_n_heads * config.index_head_dim,
            config.q_lora_rank,
        );
        self.plain(
            &format!("{prefix}.attn.indexer.weights_proj.weight"),
            DType::BF16,
            vec![config.index_n_heads, config.dim],
        );
        // Only a layer that owns the main KV projection publishes index keys.
        if config.kv_source_layers.contains(&layer) {
            self.plain(
                &format!("{prefix}.attn.indexer.wk.weight"),
                DType::BF16,
                vec![config.index_head_dim, config.head_dim],
            );
            self.plain(
                &format!("{prefix}.attn.indexer.k_norm.weight"),
                DType::BF16,
                vec![config.index_head_dim],
            );
        }
    }

    fn push_engram(&mut self, prefix: &str, config: &ModelConfig, layer: usize) {
        let Some(position) = config.engram_layer_ids.iter().position(|&l| l == layer) else {
            return;
        };
        let rows = config.engram_num_embeddings[position];
        let width = config.engram_head_dim;
        self.insert(
            format!("{prefix}.engram.embed.weight"),
            DType::F8E4M3,
            vec![rows, width],
        );
        self.insert(
            format!("{prefix}.engram.embed.scale"),
            DType::F8E8M0,
            vec![rows, width.div_ceil(crate::dtype::SCALE_BLOCK)],
        );
        // One hash column per n-gram order above the unigram: `n_hash_cols = (max_ngram_size - 1)`,
        // and each column contributes one head row of `engram_head_dim`, so the fetched width is
        // `(max_ngram_size - 1) * n_heads * head_dim` (model.py:344-345). Deriving it rather than
        // writing 3 matters: the released configuration has max_ngram_size 4 and so happens to give
        // 3, while the test configuration in this crate has 2 and gives 1, so a hardcoded 3 would be
        // wrong for it.
        let fetched = (config.engram_max_ngram_size - 1) * config.engram_n_heads * width;
        self.dense(
            &format!("{prefix}.engram.wkv"),
            config.dim * (config.hc_mult + 1),
            fetched,
        );
        // BF16, not F32: the two gate weights are created with no dtype (model.py:347-348) and the
        // Engram is built at model.py:932, outside the `set_dtype(torch.float32)` block that opens at
        // model.py:940, so they take the ambient bf16 default. They are cast to fp32 where they are
        // multiplied (model.py:356), which is a use, not a storage format. The byte identity pins it:
        // as F32 they would be 163,840 B - four [4, 5120] tensors - too large by that much for the
        // plan plus the image path to sum to the released total_size.
        self.plain(
            &format!("{prefix}.engram.q_weight"),
            DType::BF16,
            vec![config.hc_mult, config.dim],
        );
        self.plain(
            &format!("{prefix}.engram.k_weight"),
            DType::BF16,
            vec![config.hc_mult, config.dim],
        );
    }

    fn push_mtp_extras(&mut self, prefix: &str, config: &ModelConfig, stage: usize) {
        let dim = config.dim;
        if stage == 0 {
            // One backbone target layer contributes one hidden state, so the projection input is
            // `len(dspark_target_layer_ids) * dim` (model.py:1113), not a hardcoded multiple. The
            // released configuration has three targets and the test configuration has one.
            self.dense(
                &format!("{prefix}.main_proj"),
                dim,
                config.dspark_target_layer_ids.len() * dim,
            );
            self.plain(
                &format!("{prefix}.main_norm.weight"),
                DType::BF16,
                vec![dim],
            );
        }
        if stage + 1 == config.n_mtp_layers {
            self.plain(&format!("{prefix}.norm.weight"), DType::BF16, vec![dim]);
            let rank = config.dspark_markov_rank;
            self.plain(
                &format!("{prefix}.markov_head.embed.weight"),
                DType::BF16,
                vec![config.vocab_size, rank],
            );
            self.plain(
                &format!("{prefix}.markov_head.head.weight"),
                DType::BF16,
                vec![config.vocab_size, rank],
            );
            self.plain(
                &format!("{prefix}.confidence_head.proj.weight"),
                DType::BF16,
                vec![1, dim + rank],
            );
        }
    }

    /// Every entry, in plan order.
    pub fn specs(&self) -> &[TensorSpec] {
        &self.specs
    }

    /// One entry by name.
    pub fn get(&self, name: &str) -> Option<&TensorSpec> {
        self.by_name.get(name).map(|&i| &self.specs[i])
    }

    /// Whether the plan holds a tensor with that name.
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// How many tensors the plan holds.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether the plan is empty. Never true for a real configuration; present because a plan is a
    /// collection and the lint asks for it.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Every name, in plan order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.specs.iter().map(|s| s.name.as_str())
    }

    /// The number of logical elements over every tensor in the plan.
    pub fn total_elements(&self) -> u128 {
        self.specs.iter().map(TensorSpec::elements).sum()
    }

    /// The number of bytes over every tensor in the plan, packing included.
    pub fn total_bytes(&self) -> u128 {
        self.specs.iter().map(TensorSpec::storage_bytes).sum()
    }

    /// The entries whose name starts with `prefix`.
    pub fn under(&self, prefix: &str) -> impl Iterator<Item = &TensorSpec> {
        self.specs
            .iter()
            .filter(move |s| s.name.starts_with(prefix))
    }
}

/// The intermediate width of one routed expert: `moe_inter_dim`, independent of the shared expert.
fn inter_for_expert(config: &ModelConfig) -> usize {
    config.moe_inter_dim
}
