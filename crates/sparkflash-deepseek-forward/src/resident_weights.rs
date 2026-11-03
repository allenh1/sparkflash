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

//! Resident attention projections for the device chain.
//!
//! The single-node device chain reads a projection's packed weight on the host and copies it to the
//! device on every use (attention_device::upload_weight). The five attention projections across all
//! layers are about 6 GB of E4M3 codes, which fits one node's device memory, so this module reads
//! them once, B12x-tiles them once and uploads them once; the launch path then binds the arena
//! pointer and copies no weight per token.
//!
//! Routed experts (about 296 GB) and the Engram tables (about 203 GB) cannot be resident on one
//! node; those stay staged and the four-rank path is what makes them resident. The evidence reports
//! the two H2D classes separately so a run cannot claim zero weight traffic while still staging
//! hundreds of megabytes of experts.
//!
//! The digests: every uploaded weight is re-derived from the read bytes (B12x tiling plus the
//! stored E8M0 scales) and verified against the arena slice before any launch can bind it. bind()
//! then re-checks the byte geometry the kernel is about to read, so a pointer with the wrong shape
//! fails loudly.

use std::collections::{BTreeMap, BTreeSet};

use sparkflash_device::arena::{
    ArenaAllocator, ArenaTensor, ResidentArena, TensorLayout, tile_b12x,
};
use sparkflash_device::resident::{ResidentGroup, ResidentPlan};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes;
/// Host seams are row-major BF16 weights with no scale grid and no device kernel; they are
/// computed on the host (CompressorCache / IndexerWeights) and must never enter the device bind.
fn is_host_seam_weight(name: &str) -> bool {
    name.contains(".compressor.wgate.")
        || name.contains(".compressor.wkv.")
        || name.contains("indexer.weights_proj.")
        || name.contains("indexer.wk.")
}

/// One resident attention projection set: the five projections of every backbone layer.
pub struct ResidentAttention {
    arena: ResidentArena,
    geometry: BTreeMap<String, (usize, usize)>,
}

impl ResidentAttention {
    /// Reads, tiles and uploads every layer's five attention projection weights and their scales.
    ///
    /// # Errors
    /// A missing or short read, a tiling failure, an allocator/copy failure or a slice that does not
    /// verify against the bytes read from the checkpoint.
    pub fn build(
        allocator: &mut dyn ArenaAllocator,
        source: &dyn WeightSource,
        config: &ModelConfig,
    ) -> Result<Self, String> {
        let wide = config.n_heads * config.head_dim;
        let mut names: Vec<String> = Vec::new();
        let mut payloads: Vec<Vec<u8>> = Vec::new();
        let mut geometry: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for layer in 0..config.n_layers {
            let prefix = format!("layers.{layer}.attn");
            let projections = [
                (
                    format!("{prefix}.wq_a.weight"),
                    config.q_lora_rank,
                    config.dim,
                ),
                (format!("{prefix}.wq_b.weight"), wide, config.q_lora_rank),
                (format!("{prefix}.wkv.weight"), config.head_dim, config.dim),
                (
                    format!("{prefix}.wo_a.weight"),
                    config.o_groups * config.o_lora_rank,
                    wide / config.o_groups,
                ),
                (
                    format!("{prefix}.wo_b.weight"),
                    config.dim,
                    config.o_groups * config.o_lora_rank,
                ),
            ];
            for (name, cols, inner) in projections {
                let raw = source
                    .read(&name)
                    .map_err(|error| format!("{name}: {error}"))?;
                let tiled =
                    tile_b12x(&raw, cols, inner).map_err(|error| format!("{name}: {error}"))?;
                let scale_name = format!("{}.scale", name.trim_end_matches(".weight"));
                let scales = source
                    .read(&scale_name)
                    .map_err(|error| format!("{scale_name}: {error}"))?;
                let expected_scales = cols.div_ceil(32) * inner.div_ceil(32);
                if scales.len() != expected_scales {
                    return Err(format!(
                        "{scale_name}: expected {expected_scales} E8M0 bytes, got {}",
                        scales.len()
                    ));
                }
                geometry.insert(name.clone(), (cols, inner));
                names.push(name);
                payloads.push(tiled);
                names.push(scale_name);
                payloads.push(scales);
            }
        }
        let tensors: Vec<ArenaTensor<'_>> = names
            .iter()
            .enumerate()
            .map(|(index, name)| ArenaTensor {
                name,
                group: ResidentGroup::AttentionReplicated,
                bytes: &payloads[index],
                layout: TensorLayout::Stored,
                b12x: None,
            })
            .collect();
        let arena = ResidentArena::build(
            allocator,
            &tensors,
            sparkflash_device::arena::AllocationMode::Unified,
        )
        .map_err(|error| format!("resident attention arena: {error}"))?;
        // The first-use assertion, done at load: every uploaded slice must still be the bytes this
        // build derived from the checkpoint read, or the pointer binding would multiply the wrong
        // tensor.
        for (index, name) in names.iter().enumerate() {
            arena
                .verify_slice(name, &payloads[index])
                .map_err(|error| format!("resident attention verification: {error}"))?;
        }
        Ok(Self { arena, geometry })
    }

    /// The device pointers a launch should bind for one projection weight and its scale grid.
    ///
    /// Returns None for a tensor outside the resident set (the caller keeps its staged path).
    ///
    /// # Errors
    /// The resident slice exists but its byte geometry is not the geometry the kernel was asked to
    /// run, which means the launch and the plan disagree and must not compute.
    pub fn bind(
        &self,
        name: &str,
        cols: usize,
        inner: usize,
    ) -> Result<Option<(Deviceptr, Deviceptr)>, String> {
        let Some(slice) = self.arena.slice(name) else {
            return Ok(None);
        };
        let planned = self
            .geometry
            .get(name)
            .ok_or_else(|| format!("{name}: resident without a planned geometry"))?;
        if *planned != (cols, inner) {
            return Err(format!(
                "{name}: resident geometry {:?} but the launch asks {cols} x {inner}",
                planned
            ));
        }
        let expected = b12x_buffer_bytes(cols, inner);
        if slice.bytes != expected {
            return Err(format!(
                "{name}: resident slice is {} bytes, the kernel reads {expected}",
                slice.bytes
            ));
        }
        let scale_name = format!("{}.scale", name.trim_end_matches(".weight"));
        let expected_scales = cols.div_ceil(32) * inner.div_ceil(32);
        let Some(scale) = self.arena.slice(&scale_name) else {
            return Err(format!("{name}: resident weight without its scale grid"));
        };
        if scale.bytes != expected_scales {
            return Err(format!(
                "{scale_name}: resident scale grid is {} bytes, the kernel reads {expected_scales}",
                scale.bytes
            ));
        }
        Ok(Some((slice.ptr, scale.ptr)))
    }

    /// Resident payload bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.arena.evidence().bytes
    }

    /// Resident tensor count.
    #[must_use]
    pub fn tensors(&self) -> usize {
        self.arena.len()
    }
}

/// The world`world` sharded geometry of the five attention projections for one rank.
///
/// The single-node arena is built from full-width projections. The four-rank plan shards them by
/// the resident rules: Replicated names keep their full shape, Column names split the output rows
/// (cols) across ranks, Row names split the reduction width (inner). This function is the single
/// place that turns the plan's rules into the (cols, inner) a rank's kernels must be launched with,
/// so the rank-aware weight view and the bind-time byte check cannot disagree.
///
/// # Errors
/// A dimension that does not divide by `world`, or `world == 0`.
pub fn rank_attention_geometry(
    config: &ModelConfig,
    rank: usize,
    world: usize,
) -> Result<BTreeMap<String, (usize, usize)>, String> {
    if world == 0 {
        return Err("rank_attention_geometry: world must be non-zero".to_string());
    }
    if rank >= world {
        return Err(format!(
            "rank_attention_geometry: rank {rank} outside world {world}"
        ));
    }
    let wide = config.n_heads * config.head_dim;
    let split = |value: usize, axis: &str, name: &str| -> Result<usize, String> {
        if !value.is_multiple_of(world) {
            return Err(format!(
                "rank_attention_geometry: {name} {axis} {value} does not divide by world {world}"
            ));
        }
        Ok(value / world)
    };
    let mut map = BTreeMap::new();
    for layer in 0..config.n_layers {
        let prefix = format!("layers.{layer}.attn");
        // Replicated: the whole projection on every rank.
        map.insert(
            format!("{prefix}.wq_a.weight"),
            (config.q_lora_rank, config.dim),
        );
        map.insert(
            format!("{prefix}.wkv.weight"),
            (config.head_dim, config.dim),
        );
        // Column: split cols (the kernel's output rows) across ranks.
        map.insert(
            format!("{prefix}.wq_b.weight"),
            (split(wide, "cols", "wq_b")?, config.q_lora_rank),
        );
        map.insert(
            format!("{prefix}.wo_a.weight"),
            (
                split(config.o_groups * config.o_lora_rank, "cols", "wo_a")?,
                wide / config.o_groups,
            ),
        );
        // Row: split inner (the kernel's reduction width) across ranks.
        map.insert(
            format!("{prefix}.wo_b.weight"),
            (
                config.dim,
                split(config.o_groups * config.o_lora_rank, "inner", "wo_b")?,
            ),
        );
    }
    Ok(map)
}

#[cfg(test)]
mod rank_geometry_tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig::v41_flash().expect("the shipped configuration must parse")
    }

    #[test]
    fn the_four_rank_attention_geometry_splits_only_the_sharded_axes() {
        let config = config();
        let rank0 = rank_attention_geometry(&config, 0, 4).expect("rank 0 geometry");
        let wide = config.n_heads * config.head_dim;
        assert_eq!(
            rank0.get("layers.0.attn.wq_b.weight").copied(),
            Some((wide / 4, config.q_lora_rank)),
            "wq_b is Column: cols sharded, inner whole"
        );
        assert_eq!(
            rank0.get("layers.0.attn.wo_b.weight").copied(),
            Some((config.dim, config.o_groups * config.o_lora_rank / 4)),
            "wo_b is Row: cols whole, inner sharded"
        );
        assert_eq!(
            rank0.get("layers.0.attn.wq_a.weight").copied(),
            Some((config.q_lora_rank, config.dim)),
            "wq_a is replicated: both axes whole"
        );
        assert_eq!(
            rank0.get("layers.0.attn.wkv.weight").copied(),
            Some((config.head_dim, config.dim)),
            "wkv is replicated: both axes whole"
        );
        assert_eq!(rank0.len(), config.n_layers * 5);
    }

    #[test]
    fn the_four_rank_geometries_reconstruct_the_full_shapes() {
        let config = config();
        let geometries: Vec<_> = (0..4)
            .map(|rank| rank_attention_geometry(&config, rank, 4).expect("geometry"))
            .collect();
        for rank in 1..4 {
            assert_eq!(
                geometries[rank], geometries[0],
                "every rank runs the same shape"
            );
        }
        let wide = config.n_heads * config.head_dim;
        let wq_b = geometries[0]["layers.39.attn.wq_b.weight"];
        assert_eq!(wq_b.0 * 4, wide);
        let wo_b = geometries[0]["layers.39.attn.wo_b.weight"];
        assert_eq!(wo_b.1 * 4, config.o_groups * config.o_lora_rank);
    }

    #[test]
    fn a_world_that_does_not_divide_the_projection_is_an_error() {
        let config = config();
        let error = rank_attention_geometry(&config, 0, 3).expect_err("3 does not divide 32768");
        assert!(error.contains("does not divide by world"), "{error}");
        assert!(rank_attention_geometry(&config, 0, 0).is_err());
        assert!(rank_attention_geometry(&config, 4, 4).is_err());
    }
}

/// The expected per-rank shapes of every attention-path tensor in the world=`world` plan, keyed by
/// the REAL plan tensor names (compressor and indexer covered, not just the five backbone
/// projections). The values are the kernel's (cols, inner) for a weight.
///
/// This is the table the bind seam iterates plan.shards() against: a plan tensor that is missing
/// from it is named as unexpected instead of failing later as a byte mismatch.
pub fn rank_attention_plan_shapes(
    config: &ModelConfig,
    rank: usize,
    world: usize,
) -> Result<BTreeMap<String, (usize, usize)>, String> {
    let mut map = rank_attention_geometry(config, rank, world)?;
    for &layer in &config.kv_source_layers {
        let prefix = format!("layers.{layer}.attn.compressor");
        map.insert(
            format!("{prefix}.wkv.weight"),
            (config.head_dim, config.dim),
        );
        let ratio = config.compress_ratios.get(layer).copied().unwrap_or(0);
        if ratio > 1 {
            map.insert(
                format!("{prefix}.wgate.weight"),
                (config.head_dim, config.dim),
            );
        }
    }
    for &layer in &config.index_source_layers {
        let prefix = format!("layers.{layer}.attn.indexer");
        map.insert(
            format!("{prefix}.wq_b.weight"),
            (
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
            ),
        );
    }
    Ok(map)
}

/// The ten stored per-layer tensors the whole-layer device seam bodies read and the attention
/// projection bind cannot carry: the six mHC tensors, the two block norms and the q/kv norms. Every
/// one is planned as a stored operand (no B12x payload, no scale grid), which is why it is not in
/// the attention projection table and why this class binds it by stored_bytes instead of a geometry.
///
/// The list is (plan-name suffix, planned dtype), in the order a reader meets them: the two block
/// norms the seam bodies take as norm_w, then the q/kv norms the attention body reads, then the mHC
/// pair. The suffixes are the real plan names engine/tensor.rs:186-216 builds for every layer.
pub const MHC_SEAM_PARTS: [(&str, &str); 10] = [
    ("attn_norm.weight", "BF16"),
    ("ffn_norm.weight", "BF16"),
    ("attn.q_norm.weight", "BF16"),
    ("attn.kv_norm.weight", "BF16"),
    ("hc_attn_fn", "F32"),
    ("hc_attn_base", "F32"),
    ("hc_attn_scale", "F32"),
    ("hc_ffn_fn", "F32"),
    ("hc_ffn_base", "F32"),
    ("hc_ffn_scale", "F32"),
];

/// The class's plan names for one rank, layer-major in plan-name order. The set is the same on
/// every rank (the ten are Resident and replicated: no EP/TP split), which is what makes an absent
/// class's refusal sentence identical on all four ranks.
#[must_use]
pub fn mhc_seam_plan_names(config: &ModelConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(config.n_layers * MHC_SEAM_PARTS.len());
    for layer in 0..config.n_layers {
        for (suffix, _dtype) in MHC_SEAM_PARTS {
            names.push(format!("layers.{layer}.{suffix}"));
        }
    }
    names
}

/// One layer's stored bytes for the class, from the plan builder's shapes: six F32 mHC tensors
/// (fn [mix_hc][hc_dim], base [mix_hc], scale [3] per sublayer) plus the four BF16 norms.
///
/// # Errors
/// A geometry so large the byte count overflows u64 (never at released geometry).
pub fn mhc_seam_layer_bytes(config: &ModelConfig) -> Result<u64, String> {
    let hc_mult =
        u64::try_from(config.hc_mult).map_err(|_| "mhc_seam: hc_mult overflows".to_string())?;
    let dim = u64::try_from(config.dim).map_err(|_| "mhc_seam: dim overflows".to_string())?;
    let mix_hc = (2 + hc_mult)
        .checked_mul(hc_mult)
        .ok_or_else(|| "mhc_seam: mix_hc overflows".to_string())?;
    let hc_dim = hc_mult
        .checked_mul(dim)
        .ok_or_else(|| "mhc_seam: hc_dim overflows".to_string())?;
    let fn_bytes = mix_hc
        .checked_mul(hc_dim)
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| "mhc_seam: the fn matrix overflows".to_string())?;
    let base_bytes = mix_hc
        .checked_mul(4)
        .ok_or_else(|| "mhc_seam: the base row overflows".to_string())?;
    let scale_bytes = 3u64 * 4;
    let sublayer = fn_bytes
        .checked_add(base_bytes)
        .and_then(|bytes| bytes.checked_add(scale_bytes))
        .ok_or_else(|| "mhc_seam: one sublayer overflows".to_string())?;
    let norms = dim
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| {
            bytes.checked_add(u64::try_from(config.q_lora_rank).ok()?.checked_mul(2)?)
        })
        .and_then(|bytes| bytes.checked_add(u64::try_from(config.head_dim).ok()?.checked_mul(2)?))
        .ok_or_else(|| "mhc_seam: the norm rows overflow".to_string())?;
    sublayer
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(norms))
        .ok_or_else(|| "mhc_seam: the layer total overflows".to_string())
}

/// The shards the mhc_seam bind selects: the expected backbone names, looked up in the plan by
/// **exact name**, in config order.
///
/// This is the F1b fix and it is a rule, not an optimisation. The earlier pass scanned
/// `plan.shards()` and matched every shard whose name *ended with* one of the ten suffixes. The 30
/// MTP-lane tensors (`mtp.0/1/2`, ten each) carry the same suffixes, so they bound as well and the
/// coverage equality refused 400 expected / 430 bound at open. The device arm is the 40 backbone
/// layers; the MTP lanes are a different consumer (S5), so the selection is derived from the
/// expected name set and the plan is only ever consulted by exact name. No suffix scan remains in
/// this class, and `bound == expected` is asserted by the caller.
fn mhc_seam_bound_shards<'a>(
    plan: &'a ResidentPlan,
    config: &ModelConfig,
) -> Vec<(&'a sparkflash_device::resident::ResidentShard, &'static str)> {
    let by_name: BTreeMap<&str, &sparkflash_device::resident::ResidentShard> = plan
        .shards()
        .iter()
        .map(|shard| (shard.name.as_str(), shard))
        .collect();
    let mut selected = Vec::with_capacity(config.n_layers * MHC_SEAM_PARTS.len());
    for layer in 0..config.n_layers {
        for (suffix, dtype) in MHC_SEAM_PARTS {
            let name = format!("layers.{layer}.{suffix}");
            if let Some(shard) = by_name.get(name.as_str()) {
                selected.push((*shard, dtype));
            }
        }
    }
    selected
}

/// Binds the mhc_seam class: the ten stored tensors of every layer, from the same already-uploaded
/// plan arena the attention bind reads. Additive by construction - it allocates nothing, reads
/// nothing, copies nothing; it turns plan shard bytes into pointers and reports its own coverage.
///
/// The expected set comes from the config alone, so an absent class reports the same sentence on
/// every rank; the selected names are exactly that set (F1b), so a plan-vs-index disagreement is
/// named in both directions like every other class and an MTP-lane tensor can never bind.
///
/// # Errors
/// A selected shard whose placement, planned dtype or stored byte count is not the class's stored
/// layout, a missing arena slice (the plan and the arena disagree), a null pointer, a bound set that
/// differs from the expected set, or incomplete coverage.
pub fn bind_rank_mhc_seam(
    arena: &ResidentArena,
    plan: &ResidentPlan,
    config: &ModelConfig,
) -> Result<RankBind, String> {
    let expected: BTreeSet<String> = mhc_seam_plan_names(config).into_iter().collect();
    let mut bound: BTreeMap<String, (Deviceptr, Deviceptr)> = BTreeMap::new();
    for (shard, dtype) in mhc_seam_bound_shards(plan, config) {
        if shard.placement != sparkflash_device::resident::Placement::Resident {
            return Err(format!(
                "{}: planned as {:?}, the mhc_seam class binds Resident tensors only",
                shard.name, shard.placement
            ));
        }
        let suffix = MHC_SEAM_PARTS
            .iter()
            .find(|(part, _dtype)| shard.name.ends_with(part))
            .map(|(part, _dtype)| *part)
            .unwrap_or("<unknown>");
        if shard.dtype != *dtype {
            return Err(format!(
                "{}: planned as {}, the plan builder writes {dtype} for {suffix}",
                shard.name, shard.dtype
            ));
        }
        if sparkflash_device::arena::b12x_layout(shard).is_some() {
            return Err(format!(
                "{}: planned with a B12x geometry; the mhc_seam class binds stored tensors only",
                shard.name
            ));
        }
        let slice = arena.slice(&shard.name).ok_or_else(|| {
            format!(
                "{}: no arena slice (the plan and the table disagree)",
                shard.name
            )
        })?;
        let expected_bytes = shard.stored_bytes as usize;
        if slice.bytes != expected_bytes {
            return Err(format!(
                "{}: arena slice is {} bytes but the stored {} layout expects {expected_bytes} bytes (no B12x, no scale)",
                shard.name, slice.bytes, shard.dtype
            ));
        }
        if slice.ptr.0 == 0 {
            return Err(format!(
                "{}: the arena maps it to a null pointer",
                shard.name
            ));
        }
        bound.insert(shard.name.clone(), (slice.ptr, slice.ptr));
    }
    let bound_names: BTreeSet<String> = bound.keys().cloned().collect();
    // The F1b assertion, kept here rather than only inside the coverage report: the bound set is
    // built from the expected set, so a difference means the derivation itself was edited. It is
    // unreachable by construction and is stated anyway, because "400 expected, 430 bound" is the
    // exact failure this class had and the next reader should not have to infer the invariant.
    if bound_names != expected {
        return Err(format!(
            "mhc_seam bind: {} expected, {} bound; the bind must derive from the expected backbone \
             names (layers.0..{}) and never from a suffix scan over the plan",
            expected.len(),
            bound_names.len(),
            config.n_layers - 1
        ));
    }
    let coverage = check_bind_coverage("mhc_seam", &expected, &bound_names, &BTreeSet::new())?;
    Ok(RankBind {
        pointers: bound,
        coverage,
    })
}

#[cfg(test)]
mod rank_plan_shape_tests {
    use super::*;
    use sparkflash_device::resident::{HeaderEntry, Placement, ResidentPlan};
    use sparkflash_engine::tensor::TensorPlan;

    fn released_plan() -> ResidentPlan {
        let config = ModelConfig::v41_flash().expect("config");
        let plan = TensorPlan::language_model(&config);
        let entries: Vec<HeaderEntry> = plan
            .specs()
            .iter()
            .map(|spec| {
                // FP4 experts are stored as I8 with two values per byte in the last dimension.
                let (dtype, shape) = if spec.dtype.name() == "F4_E2M1" {
                    let mut shape = spec.shape.clone();
                    let last = shape.len() - 1;
                    shape[last] /= 2;
                    ("I8".to_string(), shape)
                } else {
                    (spec.dtype.name().to_string(), spec.shape.clone())
                };
                HeaderEntry::new(spec.name.clone(), dtype, shape)
            })
            .collect();
        ResidentPlan::plan(0, 4, entries).expect("world=4 rank 0 plan")
    }

    #[test]
    fn every_attention_shape_key_is_a_real_plan_name_and_the_plan_shape_matches() {
        let config = ModelConfig::v41_flash().expect("config");
        let table = rank_attention_plan_shapes(&config, 0, 4).expect("table");
        let plan = released_plan();
        let mut matched = 0usize;
        for shard in plan.shards() {
            if shard.placement != Placement::Resident {
                continue;
            }
            let Some((cols, inner)) = table.get(&shard.name) else {
                continue;
            };
            matched += 1;
            assert_eq!(
                shard.shard_shape,
                vec![*cols, *inner],
                "{}: plan per-rank shape differs from the expected-shape table",
                shard.name
            );
        }
        assert_eq!(
            matched,
            table.len(),
            "every expected-shape key must exist in the world=4 plan"
        );
    }

    #[test]
    fn the_table_covers_compressor_and_indexer_tensors_the_forward_code_reads() {
        let config = ModelConfig::v41_flash().expect("config");
        let table = rank_attention_plan_shapes(&config, 0, 4).expect("table");
        assert!(table.contains_key("layers.2.attn.compressor.wkv.weight"));
        assert!(table.contains_key("layers.2.attn.indexer.wq_b.weight"));
        assert!(table.contains_key("layers.39.attn.wq_b.weight"));
        assert_eq!(
            table.get("layers.2.attn.compressor.wkv.weight").copied(),
            Some((config.head_dim, config.dim)),
            "compressor wkv is replicated, both axes whole"
        );
    }

    #[test]
    fn the_mhc_seam_bind_is_the_expected_backbone_set_and_never_binds_the_mtp_lanes() {
        // F1b, pinned as a unit test rather than left to the first open on a node. The old pass
        // suffix-matched every resident shard, so the 30 MTP-lane tensors (mtp.0/1/2, ten each)
        // bound too and the coverage check refused 400 expected / 430 bound.
        let config = ModelConfig::v41_flash().expect("config");
        let plan = released_plan();

        // The fixture must actually carry the lanes, or the assertion below proves nothing. The
        // plan carries all 2,401 `mtp.*` tensors; the 30 that matter are the ten seam-shaped ones
        // per lane, because those are what a suffix scan would have bound.
        let mut mtp_seam_names: BTreeSet<String> = BTreeSet::new();
        for stage in 0..config.n_mtp_layers {
            for (suffix, _dtype) in MHC_SEAM_PARTS {
                mtp_seam_names.insert(format!("mtp.{stage}.{suffix}"));
            }
        }
        let present: Vec<&str> = plan
            .shards()
            .iter()
            .filter(|shard| mtp_seam_names.contains(&shard.name))
            .map(|shard| shard.name.as_str())
            .collect();
        assert_eq!(
            present.len(),
            mtp_seam_names.len(),
            "the released plan carries the ten seam-shaped tensors per MTP lane, so a suffix scan \
             would over-bind them"
        );
        assert!(
            mtp_seam_names
                .iter()
                .any(|name| name.ends_with(MHC_SEAM_PARTS[0].0)),
            "the MTP names share the seam suffixes that the old scan matched"
        );

        let selected = mhc_seam_bound_shards(&plan, &config);
        let expected: BTreeSet<String> = mhc_seam_plan_names(&config).into_iter().collect();
        assert_eq!(expected.len(), config.n_layers * MHC_SEAM_PARTS.len());
        assert_eq!(selected.len(), expected.len(), "400 expected, 400 bound");
        let names: BTreeSet<String> = selected
            .iter()
            .map(|(shard, _)| shard.name.clone())
            .collect();
        assert_eq!(
            names, expected,
            "the bound names are exactly the expected names"
        );
        assert!(
            selected
                .iter()
                .all(|(shard, _)| shard.name.starts_with("layers.")),
            "no MTP-lane tensor may enter the backbone bind"
        );
        assert!(
            selected.iter().all(|(shard, dtype)| shard.dtype == *dtype),
            "every selected shard carries the class's planned dtype"
        );
    }
}

/// The set-based coverage report of one bind pass: every expected device-served name, every bound
/// name, and the two differences by name.
///
/// A count mismatch alone is unfalsifiable (`bound 208 of 196` says nothing about which tensor is
/// wrong), so the check compares the two name SETS and reports both directions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BindCoverage {
    /// Device-served names the expected-shape table says this rank's plan must bind, sorted.
    pub expected: Vec<String>,
    /// Names actually bound, sorted.
    pub bound: Vec<String>,
    /// Expected but not bound (`expected - bound`), sorted.
    pub missing: Vec<String>,
    /// Bound but not expected (`bound - expected`), sorted.
    pub unexpected: Vec<String>,
    /// Structurally excluded host seams, sorted. Not a bind set: reported so the exclusion is
    /// visible in the evidence instead of being subtracted as a count.
    pub host_seams: Vec<String>,
}

/// Classes with at most this many names carry every name inline in the evidence row.
pub const SMALL_CLASS_NAMES: usize = 512;

/// FNV-1a over a sorted name list: the compact identity of one side of a bind set check.
#[must_use]
pub fn names_digest(names: &[String]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for name in names {
        for byte in name.as_bytes().iter().chain(std::iter::once(&b'\n')) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

impl BindCoverage {
    /// True when the expected and bound name sets are equal (including the empty case).
    #[must_use]
    pub fn complete(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty()
    }

    /// The machine-readable evidence row for one bind class.
    ///
    /// Small classes (attention) carry every name inline. Large classes (the routed experts number
    /// tens of thousands of names) carry a digest of each side and point at the per-rank name
    /// manifest the same run writes, so the evidence stays readable and every name is still
    /// published. Missing and unexpected names are always inline: a difference is never hidden.
    #[must_use]
    pub fn evidence(&self, class: &str) -> serde_json::Value {
        if self.expected.len() <= SMALL_CLASS_NAMES {
            return serde_json::json!({
                "class": class,
                "expected": self.expected.len(),
                "bound": self.bound.len(),
                "host_seams_excluded": self.host_seams.len(),
                "missing": self.missing,
                "unexpected": self.unexpected,
                "complete": self.complete(),
                "expected_names": self.expected,
                "host_seam_names": self.host_seams,
            });
        }
        serde_json::json!({
            "class": class,
            "expected": self.expected.len(),
            "bound": self.bound.len(),
            "host_seams_excluded": self.host_seams.len(),
            "missing": self.missing,
            "unexpected": self.unexpected,
            "complete": self.complete(),
            "expected_names_digest": names_digest(&self.expected),
            "bound_names_digest": names_digest(&self.bound),
            "names_omitted": "large class: every name is in this rank's bind name manifest, and each side's FNV-1a digest is above",
        })
    }

    /// Every name of this class as manifest lines, one per line: "class<TAB>set<TAB>name".
    #[must_use]
    pub fn manifest_lines(&self, class: &str) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.expected.len() + self.unexpected.len() + 1);
        for set in [
            ("expected", &self.expected),
            ("bound", &self.bound),
            ("missing", &self.missing),
            ("unexpected", &self.unexpected),
            ("host_seam", &self.host_seams),
        ] {
            for name in set.1 {
                lines.push(format!("{class}\t{}\t{name}", set.0));
            }
        }
        lines
    }
}

/// The set check a bind pass must pass: `expected` and `bound` have to be equal as SETS, and any
/// difference is an error naming every tensor in both directions.
///
/// # Errors
/// Any missing or unexpected name; the message lists both directions and the counts.
pub fn check_bind_coverage(
    class: &str,
    expected: &BTreeSet<String>,
    bound: &BTreeSet<String>,
    host_seams: &BTreeSet<String>,
) -> Result<BindCoverage, String> {
    let coverage = BindCoverage {
        expected: expected.iter().cloned().collect(),
        bound: bound.iter().cloned().collect(),
        missing: expected.difference(bound).cloned().collect(),
        unexpected: bound.difference(expected).cloned().collect(),
        host_seams: host_seams.iter().cloned().collect(),
    };
    if !coverage.complete() {
        return Err(format!(
            "{class} bind coverage: {} expected, {} bound, {} host seams excluded; expected-but-not-bound: {:?}; bound-but-not-expected: {:?}",
            coverage.expected.len(),
            coverage.bound.len(),
            coverage.host_seams.len(),
            coverage.missing,
            coverage.unexpected
        ));
    }
    Ok(coverage)
}

/// One bind pass: the pointers a launch binds plus its named coverage report.
pub struct RankBind {
    /// `(weight, scale grid)` device pointers keyed by the real plan tensor name. A stored (non-B12x)
    /// tensor has no scale grid, so its second word repeats the weight pointer.
    pub pointers: BTreeMap<String, (Deviceptr, Deviceptr)>,
    /// The set check this pass satisfied; `complete()` is true or the bind returned an error.
    pub coverage: BindCoverage,
}

/// Binds every attention-path plan shard the table covers to the rank arena and returns the
/// `RankBind`: the (weight, scales) pointers the kernels must be launched with plus the named
/// coverage report of this pass.
///
/// This is the bind-time byte check the seam relies on: the weight slice must be exactly the B12x
/// payload for the (cols, inner) the kernel will read, and the scale slice exactly the 32x32 grid,
/// or the bind fails naming the tensor rather than launching on the wrong bytes. Every table key
/// that is resident in this rank's plan must be bound: a missing entry means the arena and the
/// expected-shape table disagree, which is a bug, not a fallback.
///
/// # Errors
/// A missing arena slice, a payload whose bytes do not match the geometry, or a table key the plan
/// does not carry.
pub fn bind_rank_attention(
    arena: &ResidentArena,
    plan: &ResidentPlan,
    table: &BTreeMap<String, (usize, usize)>,
) -> Result<RankBind, String> {
    let mut bound = BTreeMap::new();
    let mut host_seams: BTreeSet<String> = BTreeSet::new();
    for shard in plan.shards() {
        // Host-seam tensors are row-major BF16 with no scale and no device kernel: they are
        // computed by CompressorCache / IndexerWeights on the host. They are not device weights,
        // so they are excluded from the device bind table rather than bound as B12x or demanding a
        // scale grid.
        if is_host_seam_weight(&shard.name) {
            host_seams.insert(shard.name.clone());
            continue;
        }
        // The table is the bind set: a planned tensor with no (cols, inner) entry is not bound. The
        // shape itself is taken from the arena predicate below, so the tuple's fields are only a filter.
        let Some((_cols, _inner)) = table.get(&shard.name).copied() else {
            continue;
        };
        let weight = arena.slice(&shard.name).ok_or_else(|| {
            format!(
                "{}: no arena slice (the plan and the table disagree)",
                shard.name
            )
        })?;
        match sparkflash_device::arena::b12x_layout(shard) {
            Some((cols, inner)) => {
                let expected_weight = b12x_buffer_bytes(cols, inner);
                if weight.bytes != expected_weight {
                    return Err(format!(
                        "{}: arena slice is {} bytes but the {cols}x{inner} B12x payload is {expected_weight}",
                        shard.name, weight.bytes
                    ));
                }
                let scale_name = format!("{}.scale", shard.name.trim_end_matches(".weight"));
                let expected_scale = cols.div_ceil(32) * inner.div_ceil(32);
                let scales = arena.slice(&scale_name).ok_or_else(|| {
                    format!("{scale_name}: no arena slice for the weight's scale grid")
                })?;
                if scales.bytes != expected_scale {
                    return Err(format!(
                        "{scale_name}: arena slice is {} bytes but the {cols}x{inner} grid is {expected_scale}",
                        scales.bytes
                    ));
                }
                bound.insert(shard.name.clone(), (weight.ptr, scales.ptr));
            }
            None => {
                let expected = shard.stored_bytes as usize;
                if weight.bytes != expected {
                    return Err(format!(
                        "{}: arena slice is {} bytes but the stored {} layout expects {expected} bytes (no B12x, no scale)",
                        shard.name, weight.bytes, shard.dtype
                    ));
                }
                // No scale grid exists for a stored tensor. The weight pointer is repeated so the
                // tuple stays non-null; consumers must select the stored path by dtype and never
                // read this second field (currently only the host CompressorCache reads wgate).
                bound.insert(shard.name.clone(), (weight.ptr, weight.ptr));
            }
        }
    }
    // Set-based coverage, reported with names in both directions: a count mismatch alone is
    // unfalsifiable (208 > 196 says nothing about which tensor is wrong). The expected set is the
    // table filtered by the structural host-seam predicate - the same predicate the loop applies -
    // so the two sets can only disagree for a real reason (a planned tensor the table does not
    // name, or a table tensor this rank's plan does not carry).
    let expected: BTreeSet<String> = table
        .keys()
        .filter(|name| !is_host_seam_weight(name))
        .cloned()
        .collect();
    let bound_names: BTreeSet<String> = bound.keys().cloned().collect();
    let coverage = check_bind_coverage("attention", &expected, &bound_names, &host_seams)?;
    Ok(RankBind {
        pointers: bound,
        coverage,
    })
}

// ---------------------------------------------------------------------------------------------
// MoE and tail arena bindings: the four-rank arm's device-served sets beyond attention.
// ---------------------------------------------------------------------------------------------

/// Which routed/shared matrix a plan name names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Matrix {
    /// Gate projection (w1).
    Gate,
    /// Up projection (w3).
    Up,
    /// Down projection (w2).
    Down,
}

impl Matrix {
    /// The plan's leaf name for this matrix.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gate => "w1",
            Self::Up => "w3",
            Self::Down => "w2",
        }
    }
}

/// The three router tensors per layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RouterPart {
    /// ffn.gate.weight, BF16 [experts, dim].
    Weight,
    /// ffn.gate.bias, F32 [experts].
    Bias,
    /// ffn.gate.bias_vl, F32 [experts]; resident, unused by the text-only path.
    BiasVl,
}

impl RouterPart {
    /// The plan's leaf name for this router part.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Weight => "weight",
            Self::Bias => "bias",
            Self::BiasVl => "bias_vl",
        }
    }
}

/// One matrix's arena pointers: the packed weight and its scale grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixBind {
    /// The packed weight bytes.
    pub weight: Deviceptr,
    /// The E8M0 scale grid (equal to the weight pointer for a stored tensor with no grid).
    pub scales: Deviceptr,
}

/// The arena-resident routed experts, shared experts and routers of one rank.
///
/// Every one of these tensors is bound from the rank arena the streaming upload already verified;
/// no per-token weight H2D exists on this structure at all (there is no reader and no source).
pub struct MoeBind {
    /// (layer, expert, matrix) -> arena pointers.
    pub routed: BTreeMap<(usize, usize, Matrix), MatrixBind>,
    /// (layer, matrix) -> arena pointers.
    pub shared: BTreeMap<(usize, Matrix), MatrixBind>,
    /// (layer, router part) -> arena pointer.
    pub router: BTreeMap<(usize, RouterPart), Deviceptr>,
    /// The set checks, one row per device class with its class name.
    pub coverage: Vec<(&'static str, BindCoverage)>,
    /// Launch-compatibility rows: what the current kernel ABIs read against what is bound.
    pub compat: Vec<serde_json::Value>,
}

impl MoeBind {
    /// The three matrices of one routed expert, if all three bound.
    #[must_use]
    pub fn expert(&self, layer: usize, expert: usize) -> Option<[MatrixBind; 3]> {
        Some([
            *self.routed.get(&(layer, expert, Matrix::Gate))?,
            *self.routed.get(&(layer, expert, Matrix::Up))?,
            *self.routed.get(&(layer, expert, Matrix::Down))?,
        ])
    }

    /// The evidence object: every class's named coverage plus the launch-compatibility rows.
    #[must_use]
    pub fn evidence(&self) -> serde_json::Value {
        serde_json::json!({
            "classes": self.coverage.iter().map(|(class, coverage)| coverage.evidence(class)).collect::<Vec<_>>(),
            "launch_compatibility": self.compat,
            "weight_h2d_bytes_per_token": 0,
            "note": "every pointer is an arena slice the resident stage verified; this bind has no reader and no source, so per-token weight H2D is structurally zero, not a counter that happens to read zero",
        })
    }
}

/// The tail arena pointers one rank serves: its embedding row shard, certified head shard and the
/// final norm. Bound here so a full-rank coverage report names them; the head/token arithmetic is
/// not wired on this arm yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailBind {
    /// embed.weight row shard [vocab / world, dim] BF16.
    pub embed: Deviceptr,
    /// head.weight row shard [vocab / world, dim] BF16.
    pub head: Deviceptr,
    /// norm.weight [dim] BF16 (replicated).
    pub norm: Deviceptr,
}

/// The resident plan's tensors by the class the four-rank arm gives them, so no resident tensor is
/// silently unaccounted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanClass {
    /// Bound by the attention projection bind.
    AttentionProjection,
    /// Structurally excluded host seam (compressor wkv/wgate, indexer wk/weights_proj).
    AttentionHostSeam,
    /// Attention auxiliaries the host reads today (sink, q/kv norms, compressor/indexer norms).
    AttentionAux,
    /// Bound by the routed-expert bind.
    RoutedExpert,
    /// Bound by the shared-expert bind.
    SharedExpert,
    /// Bound by the router bind.
    Router,
    /// Bound by the tail bind (embedding/head/norm).
    Tail,
    /// Multi-modal special-token embeddings carried resident by the plan but not served by the
    /// text path (image_start/image_end/image_newline); claimed so the census stays total.
    SpecialToken,
    /// mHC hyper-connection weights, host f32 path on this arm.
    Mhc,
    /// Engram tensors (the tables themselves are disk placement).
    Engram,
    /// MTP/DSpark stage tensors, not on the M1 backbone path.
    Mtp,
    /// Nothing above claims it: a bug, reported by name.
    Unclassified,
}

impl PlanClass {
    /// The evidence name of this class.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AttentionProjection => "attention_projection",
            Self::AttentionHostSeam => "attention_host_seam",
            Self::AttentionAux => "attention_aux_host_read",
            Self::RoutedExpert => "routed_expert",
            Self::SharedExpert => "shared_expert",
            Self::Router => "router",
            Self::Tail => "tail",
            Self::SpecialToken => "special_token",
            Self::Mhc => "mhc",
            Self::Engram => "engram",
            Self::Mtp => "mtp",
            Self::Unclassified => "unclassified",
        }
    }
}

/// Every resident plan shard's class and the names nothing claims.
pub struct PlanCensus {
    /// Class -> resident shard count.
    pub by_class: BTreeMap<PlanClass, usize>,
    /// Resident shards in total.
    pub total: usize,
    /// Resident shard names no class claims (must be empty).
    pub unclassified: Vec<String>,
}

impl PlanCensus {
    /// The evidence object.
    #[must_use]
    pub fn evidence(&self) -> serde_json::Value {
        let mut by_class = serde_json::Map::new();
        for (class, count) in &self.by_class {
            by_class.insert(class.as_str().to_string(), serde_json::json!(count));
        }
        serde_json::json!({
            "resident_tensors": self.total,
            "by_class": by_class,
            "unclassified": self.unclassified,
            "note": "every resident shard is claimed by exactly one class: device-bound (attention_projection, routed_expert, shared_expert, router, tail), structurally excluded host seam, or a named host/MTP class this arm does not serve yet",
        })
    }
}

/// Parses layers.{L}. from a plan name.
fn plan_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse().ok()
}

/// Parses layers.{L}.ffn.experts.{E}. from a plan name.
fn plan_expert(name: &str) -> Option<usize> {
    let rest = name.split_once(".ffn.experts.")?.1;
    let (expert, _) = rest.split_once('.')?;
    expert.parse().ok()
}

/// The routed/shared matrix and whether the name is the scale grid, from its leaf pair.
fn matrix_and_scale(name: &str) -> Option<(Matrix, bool)> {
    let mut parts = name.rsplit('.');
    let leaf = parts.next()?;
    let parent = parts.next()?;
    let scale = match leaf {
        "weight" => false,
        "scale" => true,
        _ => return None,
    };
    let matrix = match parent {
        "w1" => Matrix::Gate,
        "w3" => Matrix::Up,
        "w2" => Matrix::Down,
        _ => return None,
    };
    Some((matrix, scale))
}

/// Classifies one plan name by the role the four-rank arm gives it.
#[must_use]
pub fn classify_plan_name(
    name: &str,
    attention_table: &BTreeMap<String, (usize, usize)>,
) -> PlanClass {
    // MTP/DSpark first: its stages reuse the backbone leaf names (ffn.experts, ffn.gate,
    // indexer.wk), so classifying by leaf would fold a stage this arm does not serve into a
    // backbone class.
    if name.starts_with("mtp.") {
        return PlanClass::Mtp;
    }
    if is_host_seam_weight(name) {
        return PlanClass::AttentionHostSeam;
    }
    if attention_table.contains_key(name) {
        return PlanClass::AttentionProjection;
    }
    if name.contains(".ffn.experts.") {
        return PlanClass::RoutedExpert;
    }
    if name.contains(".ffn.shared_experts.") {
        return PlanClass::SharedExpert;
    }
    if name.contains(".ffn.gate.") {
        return PlanClass::Router;
    }
    if matches!(name, "embed.weight" | "head.weight" | "norm.weight") {
        return PlanClass::Tail;
    }
    // The released checkpoint carries three multi-modal special-token embeddings resident even on
    // the text path (the vision tower and aligner are excluded from the plan, these three are not).
    // They have no device kernel here, but they are claimed by name so the census stays total.
    if matches!(name, "image_start" | "image_end" | "image_newline") {
        return PlanClass::SpecialToken;
    }
    if name.contains(".engram.") || name.starts_with("engram.") {
        return PlanClass::Engram;
    }
    if name.contains(".hc_") {
        return PlanClass::Mhc;
    }
    if name.contains(".attn.") || name.contains(".attn_") || name.contains(".ffn_norm.") {
        return PlanClass::AttentionAux;
    }
    PlanClass::Unclassified
}

/// Counts every resident plan shard by class.
#[must_use]
pub fn resident_plan_census(
    plan: &ResidentPlan,
    attention_table: &BTreeMap<String, (usize, usize)>,
) -> PlanCensus {
    let mut by_class: BTreeMap<PlanClass, usize> = BTreeMap::new();
    let mut total = 0usize;
    let mut unclassified = Vec::new();
    for shard in plan.shards() {
        if shard.placement != sparkflash_device::resident::Placement::Resident {
            continue;
        }
        total += 1;
        let class = classify_plan_name(&shard.name, attention_table);
        if class == PlanClass::Unclassified {
            unclassified.push(shard.name.clone());
        }
        *by_class.entry(class).or_insert(0) += 1;
    }
    PlanCensus {
        by_class,
        total,
        unclassified,
    }
}

/// Binds one plan weight shard (and its scale grid when the plan carries one) to the arena.
///
/// The byte check is the same rule for every class: a B12x-served FP8 weight must be exactly the
/// tiled payload for its (cols, inner), a stored tensor exactly its rank shard's bytes, and a scale
/// grid exactly its rank shard's bytes.
fn bind_plan_pair(
    arena: &ResidentArena,
    shard: &sparkflash_device::resident::ResidentShard,
    by_name: &BTreeMap<&str, &sparkflash_device::resident::ResidentShard>,
) -> Result<MatrixBind, String> {
    let weight = arena.slice(&shard.name).ok_or_else(|| {
        format!(
            "{}: no arena slice (the plan and the bind disagree)",
            shard.name
        )
    })?;
    let expected_weight = if weight.layout == TensorLayout::MoePadded640
        && let Some((rows, _unpadded, padded_row)) =
            sparkflash_device::arena::moe_padded_640_layout(shard)
    {
        rows * padded_row
    } else {
        match sparkflash_device::arena::b12x_layout(shard) {
            Some((cols, inner)) => b12x_buffer_bytes(cols, inner),
            None => shard.shard_bytes as usize,
        }
    };
    if weight.bytes != expected_weight {
        return Err(format!(
            "{}: arena slice is {} bytes but this rank's planned payload is {expected_weight}",
            shard.name, weight.bytes
        ));
    }
    let scale_name = format!("{}.scale", shard.name.trim_end_matches(".weight"));
    let scales = match by_name.get(scale_name.as_str()) {
        Some(scale_shard) => {
            let slice = arena
                .slice(&scale_name)
                .ok_or_else(|| format!("{scale_name}: no arena slice for the scale grid"))?;
            let expected_scale = if slice.layout == TensorLayout::MoePadded640
                && let Some((rows, _unpadded, padded_row)) =
                    sparkflash_device::arena::moe_padded_640_layout(scale_shard)
            {
                rows * padded_row
            } else {
                scale_shard.shard_bytes as usize
            };
            if slice.bytes != expected_scale {
                return Err(format!(
                    "{scale_name}: arena slice is {} bytes but this rank's planned grid is {expected_scale}",
                    slice.bytes
                ));
            }
            slice.ptr
        }
        None => weight.ptr,
    };
    Ok(MatrixBind {
        weight: weight.ptr,
        scales,
    })
}

/// The MoE classes' expected name sets, derived from the model geometry rather than from the plan.
///
/// The plan can only be right if it carries every expert of every layer; the sets are the check's
/// independent side, so a plan that skipped an expert would be named, not silently accepted.
pub struct MoeExpectedSets {
    /// Routed w1/w3 weights and scales, [layer][expert].
    pub gate_up: BTreeSet<String>,
    /// Routed w2 weights and scales, [layer][expert].
    pub down: BTreeSet<String>,
    /// Shared-expert w1/w3/w2 weights and scales, [layer].
    pub shared: BTreeSet<String>,
    /// Router gate weight, bias and bias_vl, [layer].
    pub router: BTreeSet<String>,
}

/// Builds the four MoE class expected sets from the model geometry.
#[must_use]
pub fn moe_expected_sets(config: &ModelConfig) -> MoeExpectedSets {
    let mut gate_up: BTreeSet<String> = BTreeSet::new();
    let mut down: BTreeSet<String> = BTreeSet::new();
    let mut shared: BTreeSet<String> = BTreeSet::new();
    let mut router: BTreeSet<String> = BTreeSet::new();
    for layer in 0..config.n_layers {
        for expert in 0..config.n_routed_experts {
            for matrix in [Matrix::Gate, Matrix::Up] {
                let base = format!(
                    "layers.{layer}.ffn.experts.{expert}.{}.weight",
                    matrix.as_str()
                );
                gate_up.insert(base.clone());
                gate_up.insert(format!("{}.scale", base.trim_end_matches(".weight")));
            }
            let base = format!("layers.{layer}.ffn.experts.{expert}.w2.weight");
            down.insert(base.clone());
            down.insert(format!("{}.scale", base.trim_end_matches(".weight")));
        }
        for matrix in [Matrix::Gate, Matrix::Up, Matrix::Down] {
            let base = format!(
                "layers.{layer}.ffn.shared_experts.{}.weight",
                matrix.as_str()
            );
            shared.insert(base.clone());
            shared.insert(format!("{}.scale", base.trim_end_matches(".weight")));
        }
        for part in [RouterPart::Weight, RouterPart::Bias, RouterPart::BiasVl] {
            router.insert(format!("layers.{layer}.ffn.gate.{}", part.as_str()));
        }
    }
    MoeExpectedSets {
        gate_up,
        down,
        shared,
        router,
    }
}

/// Binds the routed experts, shared experts and routers of one rank to its arena, with one set
/// check per device class and launch-compatibility rows for the current kernel ABIs.
///
/// # Errors
/// A missing arena slice, a slice whose bytes are not the planned payload, or a class whose
/// expected and bound name sets differ (the message names both directions).
pub fn bind_rank_moe(
    arena: &ResidentArena,
    plan: &ResidentPlan,
    config: &ModelConfig,
) -> Result<MoeBind, String> {
    let by_name: BTreeMap<&str, &sparkflash_device::resident::ResidentShard> = plan
        .shards()
        .iter()
        .map(|shard| (shard.name.as_str(), shard))
        .collect();
    let empty_table: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut routed: BTreeMap<(usize, usize, Matrix), MatrixBind> = BTreeMap::new();
    let mut shared: BTreeMap<(usize, Matrix), MatrixBind> = BTreeMap::new();
    let mut router: BTreeMap<(usize, RouterPart), Deviceptr> = BTreeMap::new();
    let mut routed_gate_up: BTreeSet<String> = BTreeSet::new();
    let mut routed_down: BTreeSet<String> = BTreeSet::new();
    let mut shared_names: BTreeSet<String> = BTreeSet::new();
    let mut router_names: BTreeSet<String> = BTreeSet::new();
    for shard in plan.shards() {
        if shard.placement != sparkflash_device::resident::Placement::Resident {
            continue;
        }
        // Scale shards are bound with their weight; the coverage sets still count their names.
        if shard.name.ends_with(".scale") {
            continue;
        }
        match classify_plan_name(&shard.name, &empty_table) {
            PlanClass::RoutedExpert => {
                let layer = plan_layer(&shard.name)
                    .ok_or_else(|| format!("{}: no layer index", shard.name))?;
                let expert = plan_expert(&shard.name)
                    .ok_or_else(|| format!("{}: no expert index", shard.name))?;
                let Some((matrix, false)) = matrix_and_scale(&shard.name) else {
                    return Err(format!(
                        "{}: a routed expert leaf is not a w1/w3/w2 weight",
                        shard.name
                    ));
                };
                let bind = bind_plan_pair(arena, shard, &by_name)?;
                routed.insert((layer, expert, matrix), bind);
                let scale = format!("{}.scale", shard.name.trim_end_matches(".weight"));
                let set = match matrix {
                    Matrix::Down => &mut routed_down,
                    Matrix::Gate | Matrix::Up => &mut routed_gate_up,
                };
                set.insert(shard.name.clone());
                set.insert(scale);
            }
            PlanClass::SharedExpert => {
                let layer = plan_layer(&shard.name)
                    .ok_or_else(|| format!("{}: no layer index", shard.name))?;
                let Some((matrix, false)) = matrix_and_scale(&shard.name) else {
                    return Err(format!(
                        "{}: a shared-expert leaf is not a w1/w3/w2 weight",
                        shard.name
                    ));
                };
                let bind = bind_plan_pair(arena, shard, &by_name)?;
                shared.insert((layer, matrix), bind);
                shared_names.insert(shard.name.clone());
                shared_names.insert(format!("{}.scale", shard.name.trim_end_matches(".weight")));
            }
            PlanClass::Router => {
                let layer = plan_layer(&shard.name)
                    .ok_or_else(|| format!("{}: no layer index", shard.name))?;
                let part = if shard.name.ends_with(".gate.weight") {
                    RouterPart::Weight
                } else if shard.name.ends_with(".gate.bias") {
                    RouterPart::Bias
                } else if shard.name.ends_with(".gate.bias_vl") {
                    RouterPart::BiasVl
                } else {
                    return Err(format!("{}: an unknown router leaf", shard.name));
                };
                let slice = arena
                    .slice(&shard.name)
                    .ok_or_else(|| format!("{}: no arena slice for the router", shard.name))?;
                if slice.bytes != shard.shard_bytes as usize {
                    return Err(format!(
                        "{}: arena slice is {} bytes but this rank's planned payload is {}",
                        shard.name, slice.bytes, shard.shard_bytes
                    ));
                }
                router.insert((layer, part), slice.ptr);
                router_names.insert(shard.name.clone());
            }
            _ => {}
        }
    }
    // Expected sets come from the model geometry, not from the plan.
    let expected = moe_expected_sets(config);
    let expected_gate_up = expected.gate_up;
    let expected_down = expected.down;
    let expected_shared = expected.shared;
    let expected_router = expected.router;
    let empty: BTreeSet<String> = BTreeSet::new();
    let coverage = vec![
        (
            "routed_experts_gate_up",
            check_bind_coverage(
                "routed_experts_gate_up",
                &expected_gate_up,
                &routed_gate_up,
                &empty,
            )?,
        ),
        (
            "routed_experts_down",
            check_bind_coverage("routed_experts_down", &expected_down, &routed_down, &empty)?,
        ),
        (
            "shared_experts",
            check_bind_coverage("shared_experts", &expected_shared, &shared_names, &empty)?,
        ),
        (
            "router",
            check_bind_coverage("router", &expected_router, &router_names, &empty)?,
        ),
    ];
    let down_is_padded_640 = expected_down.iter().any(|name| {
        arena
            .slice(name)
            .is_some_and(|slice| slice.layout == TensorLayout::MoePadded640)
    });
    let down_compat = if down_is_padded_640 {
        serde_json::json!({
            "class": "routed_experts_down",
            "launch_compatible": true,
            "note": "sparkflash convert pre-padded w2 to the 640-K row layout (320 packed bytes and 20 scale bytes per output row) offline; zero boot repack",
        })
    } else {
        serde_json::json!({
            "class": "routed_experts_down",
            "launch_compatible": false,
            "reason": "b12x_mxfp4_moe_sm121 reads the A8 pre-padded 640-K row layout (320 packed bytes and 20 scale bytes per output row); this rank's unconverted plan stores the checkpoint's 288-byte rows and 18-byte grid",
            "remaining": "either launch the stored-layout kernel variant (b12x_mxfp4_moe_stored_sm121) or run `sparkflash convert` once offline to pre-pad w2 to the 640-wide layout with no boot repack",
        })
    };
    let compat = vec![
        serde_json::json!({
            "class": "routed_experts_gate_up",
            "launch_compatible": true,
            "note": "routed_gate_up_sm121 reads SPARK_M4_MOE_WIDTH=576 output rows of 2560 packed bytes with the checkpoint's [576,160] row-scale grid; the arena stores exactly that per-expert slice",
        }),
        down_compat,
        serde_json::json!({
            "class": "shared_experts",
            "launch_compatible": true,
            "note": "b12x_mxfp8_dense_sm121 is runtime-shaped: w1/w3 launch with cols=576, inner=5120 and w2 with cols=5120, inner=576, which is exactly the B12x arena slice shape",
        }),
        serde_json::json!({
            "class": "router",
            "launch_compatible": true,
            "note": "moe_router_sm121 reads the row-major BF16 [384,5120] gate weight and F32 [384] bias, which the arena stores untransformed",
        }),
    ];
    Ok(MoeBind {
        routed,
        shared,
        router,
        coverage,
        compat,
    })
}

/// Binds the tail tensors (embedding row shard, certified head shard, final norm) to the arena.
///
/// # Errors
/// A missing arena slice or a slice whose bytes are not the rank's planned payload.
pub fn bind_rank_tail(
    arena: &ResidentArena,
    plan: &ResidentPlan,
) -> Result<(TailBind, BindCoverage), String> {
    let by_name: BTreeMap<&str, &sparkflash_device::resident::ResidentShard> = plan
        .shards()
        .iter()
        .map(|shard| (shard.name.as_str(), shard))
        .collect();
    let point = |name: &str| -> Result<Deviceptr, String> {
        let shard = by_name
            .get(name)
            .ok_or_else(|| format!("{name}: the plan does not carry the tail tensor"))?;
        let slice = arena
            .slice(name)
            .ok_or_else(|| format!("{name}: no arena slice for the tail tensor"))?;
        if slice.bytes != shard.shard_bytes as usize {
            return Err(format!(
                "{name}: arena slice is {} bytes but this rank's planned payload is {}",
                slice.bytes, shard.shard_bytes
            ));
        }
        Ok(slice.ptr)
    };
    let tail = TailBind {
        embed: point("embed.weight")?,
        head: point("head.weight")?,
        norm: point("norm.weight")?,
    };
    let expected: BTreeSet<String> = ["embed.weight", "head.weight", "norm.weight"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let coverage = check_bind_coverage("tail", &expected, &expected, &BTreeSet::new())?;
    Ok((tail, coverage))
}

#[cfg(test)]
mod bind_coverage_tests {
    use super::*;
    use sparkflash_device::resident::{HeaderEntry, Placement, ResidentPlan};
    use sparkflash_engine::tensor::TensorPlan;

    fn names(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|name| (*name).to_string()).collect()
    }

    fn released_plan() -> ResidentPlan {
        let config = ModelConfig::v41_flash().expect("config");
        let plan = TensorPlan::language_model(&config);
        let entries: Vec<HeaderEntry> = plan
            .specs()
            .iter()
            .map(|spec| {
                // FP4 experts are stored as I8 with two values per byte in the last dimension.
                let (dtype, shape) = if spec.dtype.name() == "F4_E2M1" {
                    let mut shape = spec.shape.clone();
                    let last = shape.len() - 1;
                    shape[last] /= 2;
                    ("I8".to_string(), shape)
                } else {
                    (spec.dtype.name().to_string(), spec.shape.clone())
                };
                HeaderEntry::new(spec.name.clone(), dtype, shape)
            })
            .collect();
        ResidentPlan::plan(0, 4, entries).expect("world=4 rank 0 plan")
    }

    #[test]
    fn the_set_check_refuses_a_difference_and_names_both_sides() {
        let expected = names(&["layers.0.attn.wq_a.weight", "layers.0.attn.wkv.weight"]);
        let equal = names(&["layers.0.attn.wkv.weight", "layers.0.attn.wq_a.weight"]);
        let ok =
            check_bind_coverage("unit", &expected, &equal, &names(&[])).expect("equal sets bind");
        assert!(ok.complete());
        assert_eq!(ok.expected, ok.bound);

        let missing = names(&["layers.0.attn.wq_a.weight"]);
        let error = check_bind_coverage("unit", &expected, &missing, &names(&[]))
            .expect_err("a missing name must fail");
        assert!(
            error.contains("expected-but-not-bound: [\"layers.0.attn.wkv.weight\"]"),
            "{error}"
        );

        let extra = names(&[
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wkv.weight",
            "layers.0.attn.wq_b.weight",
        ]);
        let error = check_bind_coverage("unit", &expected, &extra, &names(&[]))
            .expect_err("an unexpected name must fail");
        assert!(
            error.contains("bound-but-not-expected: [\"layers.0.attn.wq_b.weight\"]"),
            "{error}"
        );
    }

    /// The plan-side MoE class name sets: every resident shard classified by the bind's own
    /// predicates. Independent of the geometry-derived expected sets, so the comparison below is a
    /// real two-sided check and fails if a plan name is classified into the wrong class.
    fn plan_moe_sets(
        plan: &ResidentPlan,
        table: &BTreeMap<String, (usize, usize)>,
    ) -> MoeExpectedSets {
        let mut gate_up = BTreeSet::new();
        let mut down = BTreeSet::new();
        let mut shared = BTreeSet::new();
        let mut router = BTreeSet::new();
        for shard in plan.shards() {
            if shard.placement != Placement::Resident {
                continue;
            }
            match classify_plan_name(&shard.name, table) {
                PlanClass::RoutedExpert => {
                    if shard.name.contains(".w2.") {
                        down.insert(shard.name.clone());
                    } else {
                        gate_up.insert(shard.name.clone());
                    }
                }
                PlanClass::SharedExpert => {
                    shared.insert(shard.name.clone());
                }
                PlanClass::Router => {
                    router.insert(shard.name.clone());
                }
                _ => {}
            }
        }
        MoeExpectedSets {
            gate_up,
            down,
            shared,
            router,
        }
    }

    #[test]
    fn the_four_rank_moe_expected_and_plan_sets_are_equal_and_named() {
        let config = ModelConfig::v41_flash().expect("config");
        let table = rank_attention_plan_shapes(&config, 0, 4).expect("table");
        let plan = released_plan();
        let expected = moe_expected_sets(&config);
        // Geometry, counted independently of the plan: 40 layers x 384 experts x (w1,w3) x
        // (weight, scale) = 61,440 names; w2 = 30,720; shared 3 x 2 x 40 = 240; router 3 x 40 = 120.
        assert_eq!(expected.gate_up.len(), 61_440);
        assert_eq!(expected.down.len(), 30_720);
        assert_eq!(expected.shared.len(), 240);
        assert_eq!(expected.router.len(), 120);
        let planned = plan_moe_sets(&plan, &table);
        for (class, expected_side, plan_side) in [
            ("routed gate/up", &expected.gate_up, &planned.gate_up),
            ("routed down", &expected.down, &planned.down),
            ("shared", &expected.shared, &planned.shared),
            ("router", &expected.router, &planned.router),
        ] {
            assert_eq!(
                expected_side,
                plan_side,
                "{class}: the geometry and this rank's plan disagree: \
                 expected-but-not-planned: {:?}; planned-but-not-expected: {:?}",
                expected_side.difference(plan_side).collect::<Vec<_>>(),
                plan_side.difference(expected_side).collect::<Vec<_>>()
            );
        }
        // Every expected weight has its scale grid in the plan: the bind pairs them and a missing
        // grid would silently pair the weight pointer as a scale buffer.
        let by_name: BTreeSet<&str> = plan.shards().iter().map(|s| s.name.as_str()).collect();
        for name in &expected.gate_up {
            if let Some(base) = name.strip_suffix(".weight") {
                assert!(
                    by_name.contains(format!("{base}.scale").as_str()),
                    "{name}: the plan carries no scale grid"
                );
            }
        }
    }

    #[test]
    fn the_resident_plan_census_claims_every_shard_by_class() {
        let config = ModelConfig::v41_flash().expect("config");
        let table = rank_attention_plan_shapes(&config, 0, 4).expect("table");
        let plan = released_plan();
        let census = resident_plan_census(&plan, &table);
        assert!(
            census.unclassified.is_empty(),
            "resident shards no class claims: {:?}",
            census.unclassified
        );
        assert_eq!(census.total, plan.resident_tensors());
        let count = |class: PlanClass| *census.by_class.get(&class).unwrap_or(&0);
        assert_eq!(count(PlanClass::AttentionProjection), 208);
        assert_eq!(count(PlanClass::AttentionHostSeam), 19);
        // The mhc_seam class does not move these: the six mHC tensors stay Mhc and the four
        // per-layer norms stay AttentionAux, which is why the bind gets its own second pointer
        // family instead of a wider attention table (the probe row's census read 240 / 416).
        assert_eq!(count(PlanClass::Mhc), 240);
        assert_eq!(count(PlanClass::AttentionAux), 416);
        assert_eq!(count(PlanClass::RoutedExpert), 92_160);
        assert_eq!(count(PlanClass::SharedExpert), 240);
        assert_eq!(count(PlanClass::Router), 120);
        assert_eq!(count(PlanClass::Tail), 3);
        for host_side in [
            PlanClass::AttentionAux,
            PlanClass::Mhc,
            PlanClass::Engram,
            PlanClass::Mtp,
        ] {
            assert!(
                count(host_side) > 0,
                "{}: the census must name this class, not fold it into another",
                host_side.as_str()
            );
        }
        let sum: usize = census.by_class.values().sum();
        assert_eq!(
            sum, census.total,
            "every resident shard in exactly one class"
        );
        let evidence = census.evidence();
        assert_eq!(evidence["resident_tensors"], census.total);
        assert_eq!(evidence["unclassified"], serde_json::json!([]));
    }

    #[test]
    fn the_moe_arena_payload_predicate_matches_the_plan_shapes() {
        let plan = released_plan();
        let by_name: BTreeMap<&str, &sparkflash_device::resident::ResidentShard> = plan
            .shards()
            .iter()
            .map(|shard| (shard.name.as_str(), shard))
            .collect();
        // Routed experts are stored I8 (packed FP4), so b12x_layout does not tile them and the
        // arena slice is exactly the rank shard: 576 x 2560 = 1,474,560 bytes for w1/w3 and
        // 5120 x 288 for w2.
        for (name, bytes) in [
            ("layers.0.ffn.experts.0.w1.weight", 1_474_560usize),
            ("layers.0.ffn.experts.0.w3.weight", 1_474_560usize),
            ("layers.0.ffn.experts.0.w2.weight", 1_474_560usize),
            ("layers.0.ffn.experts.0.w2.scale", 92_160usize),
            ("layers.0.ffn.gate.weight", 3_932_160usize),
        ] {
            let shard = by_name.get(name).expect("plan shard");
            assert!(
                sparkflash_device::arena::b12x_layout(shard).is_none(),
                "{name}: stored, not B12x-tiled"
            );
            assert_eq!(shard.shard_bytes as usize, bytes, "{name}");
        }
        // The shared expert is B12x-served but NOT padded in the plan: 2304 / 4 = 576 output rows
        // for w1/w3 and 576 reduction columns for w2, which the runtime-shaped dense kernel reads.
        for (name, cols, inner) in [
            ("layers.0.ffn.shared_experts.w1.weight", 576usize, 5120usize),
            ("layers.0.ffn.shared_experts.w3.weight", 576, 5120),
            ("layers.0.ffn.shared_experts.w2.weight", 5120, 576),
        ] {
            let shard = by_name.get(name).expect("plan shard");
            assert_eq!(
                sparkflash_device::arena::b12x_layout(shard),
                Some((cols, inner)),
                "{name}: the bind byte rule must tile it"
            );
            assert_eq!(
                b12x_buffer_bytes(cols, inner),
                shard.shard_bytes as usize,
                "{name}: 576 rows are whole 32-row B12x tiles, so tiling does not change the size"
            );
        }
        // The routed down-projection layout deviation the window evidence must name: the arena rows
        // are 288 packed bytes (18 scale bytes) while the A8 pre-padded combine ABI reads 320 (20).
        let w2 = by_name
            .get("layers.0.ffn.experts.0.w2.weight")
            .expect("plan shard");
        let (rows, row_bytes) = (w2.shard_shape[0], w2.shard_shape[1]);
        assert_eq!((rows, row_bytes), (5120, 288));
        assert!(
            row_bytes < 320,
            "the combine kernel's pre-padded row is 320 bytes; the arena stores {row_bytes}"
        );
    }

    #[test]
    fn large_bind_classes_are_digested_and_small_classes_are_named() {
        let config = ModelConfig::v41_flash().expect("config");
        let expected = moe_expected_sets(&config);
        let empty = BTreeSet::new();
        let large = check_bind_coverage(
            "routed_experts_gate_up",
            &expected.gate_up,
            &expected.gate_up,
            &empty,
        )
        .expect("equal sets");
        let evidence = large.evidence("routed_experts_gate_up");
        assert_eq!(evidence["expected"], 61_440);
        assert_eq!(evidence["complete"], true);
        assert!(
            evidence["expected_names"].is_null(),
            "a 61,440-name class must not inline every name in the JSON"
        );
        assert!(evidence["expected_names_digest"].is_u64());
        assert!(evidence["bound_names_digest"].is_u64());
        assert_eq!(
            large.manifest_lines("routed_experts_gate_up").len(),
            2 * 61_440,
            "the manifest carries every expected and bound name"
        );

        let small = check_bind_coverage(
            "attention",
            &names(&["layers.0.attn.wq_a.weight", "layers.0.attn.wkv.weight"]),
            &names(&["layers.0.attn.wq_a.weight", "layers.0.attn.wkv.weight"]),
            &names(&[]),
        )
        .expect("equal sets");
        let evidence = small.evidence("attention");
        assert_eq!(evidence["expected_names"].as_array().map(Vec::len), Some(2));
        assert!(evidence["expected_names_digest"].is_null());
    }

    #[test]
    fn the_four_rank_attention_expected_and_plan_sets_are_equal_and_named() {
        let config = ModelConfig::v41_flash().expect("config");
        let table = rank_attention_plan_shapes(&config, 0, 4).expect("table");
        let plan = released_plan();
        // Side A: the bind check's expected set (the table filtered by the structural host-seam
        // predicate). Side B: the names this rank's plan can bind (a resident plan shard the table
        // names and the predicate does not exclude). The two sets must be EQUAL; a count-only check
        // cannot see a swap, and this test fails if they differ.
        let expected: BTreeSet<String> = table
            .keys()
            .filter(|name| !is_host_seam_weight(name))
            .cloned()
            .collect();
        let plan_derived: BTreeSet<String> = plan
            .shards()
            .iter()
            .filter(|shard| shard.placement == Placement::Resident)
            .filter(|shard| table.contains_key(&shard.name))
            .filter(|shard| !is_host_seam_weight(&shard.name))
            .map(|shard| shard.name.clone())
            .collect();
        assert_eq!(
            expected,
            plan_derived,
            "the expected-shape table and this rank's plan disagree on the attention bind set: \
             expected-but-not-planned: {:?}; planned-but-not-expected: {:?}",
            expected.difference(&plan_derived).collect::<Vec<_>>(),
            plan_derived.difference(&expected).collect::<Vec<_>>()
        );
        assert_eq!(
            expected.len(),
            208,
            "200 backbone projections + 8 indexer wq_b shards are device-served"
        );
        assert_eq!(
            plan.shards()
                .iter()
                .filter(|shard| is_host_seam_weight(&shard.name))
                .count(),
            19,
            "compressor wgate/wkv and indexer wk/weights_proj are host seams, counted by name"
        );
        // The named difference of the 2026-10-20 window: compressor tensors are host seams and must
        // never be in the expected device set; indexer wq_b is the one indexer device weight.
        assert!(
            !expected.contains("layers.2.attn.compressor.wkv.weight"),
            "compressor wkv is a host seam (CompressorCache), not a device weight"
        );
        assert!(
            !expected.contains("layers.2.attn.compressor.wgate.weight"),
            "compressor wgate is BF16 row-major on the host seam, not an FP8 B12x device weight"
        );
        assert!(
            expected.contains("layers.2.attn.indexer.wq_b.weight"),
            "the indexer wq_b is an MXFP8 B12x device weight on the rank plan"
        );
        // Cross-check against the bind check itself: equal sets still pass and report 208/208.
        let coverage = check_bind_coverage("attention", &expected, &plan_derived, &names(&[]))
            .expect("equal sets");
        assert_eq!(coverage.expected.len(), coverage.bound.len());
        assert!(coverage.complete());
        assert_eq!(coverage.evidence("attention")["complete"], true);
    }

    #[test]
    fn the_multi_modal_special_tokens_are_claimed_by_name_not_left_unclassified() {
        // The released checkpoint carries image_start/image_end/image_newline resident; without a
        // class the census would call them unclassified and summary.complete would be false for a
        // reason that is not a bind defect.
        let table: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for name in ["image_start", "image_end", "image_newline"] {
            assert_eq!(
                classify_plan_name(name, &table),
                PlanClass::SpecialToken,
                "{name}"
            );
        }
        assert_eq!(PlanClass::SpecialToken.as_str(), "special_token");
        // The backbone names are unaffected by the new class.
        assert_eq!(classify_plan_name("embed.weight", &table), PlanClass::Tail);
    }
}

#[cfg(test)]
mod mhc_seam_class_tests {
    use super::*;
    use sparkflash_device::resident::{HeaderEntry, Placement, ResidentPlan};
    use sparkflash_engine::tensor::TensorPlan;

    fn released_config() -> ModelConfig {
        ModelConfig::v41_flash().expect("config")
    }

    fn released_plan() -> ResidentPlan {
        let config = released_config();
        let plan = TensorPlan::language_model(&config);
        let entries: Vec<HeaderEntry> = plan
            .specs()
            .iter()
            .map(|spec| {
                let (dtype, shape) = if spec.dtype.name() == "F4_E2M1" {
                    let mut shape = spec.shape.clone();
                    let last = shape.len() - 1;
                    shape[last] /= 2;
                    ("I8".to_string(), shape)
                } else {
                    (spec.dtype.name().to_string(), spec.shape.clone())
                };
                HeaderEntry::new(spec.name.clone(), dtype, shape)
            })
            .collect();
        ResidentPlan::plan(0, 4, entries).expect("world=4 rank 0 plan")
    }

    #[test]
    fn the_class_is_ten_names_per_layer_and_already_in_the_plan() {
        let config = released_config();
        let names = mhc_seam_plan_names(&config);
        assert_eq!(MHC_SEAM_PARTS.len(), 10, "six mHC tensors and four norms");
        assert_eq!(names.len(), 400, "ten per layer over forty layers");
        let plan = released_plan();
        let mut resident: BTreeMap<&str, (u64, &str)> = BTreeMap::new();
        for shard in plan.shards() {
            if shard.placement == Placement::Resident {
                resident.insert(
                    shard.name.as_str(),
                    (shard.stored_bytes, shard.dtype.as_str()),
                );
            }
        }
        let mut total = 0u64;
        for name in &names {
            let (bytes, dtype) = *resident.get(name.as_str()).unwrap_or_else(|| {
                panic!("{name} is not a resident plan shard: the plan already carries this class")
            });
            let planned_dtype = MHC_SEAM_PARTS
                .iter()
                .find(|(suffix, _dtype)| name.ends_with(suffix))
                .map(|(_suffix, dtype)| *dtype)
                .expect("a known suffix");
            assert_eq!(dtype, planned_dtype, "{name} dtype");
            total += bytes;
        }
        let per_layer = mhc_seam_layer_bytes(&config).expect("bytes");
        assert_eq!(per_layer, 3_956_440, "3,932,376 mHC + 24,064 norms");
        assert_eq!(
            total,
            per_layer * 40,
            "the class is already charged in the plan"
        );
        assert_eq!(total, 158_257_600);
    }

    #[test]
    fn the_attention_class_is_not_widened_by_the_seam_class() {
        let config = released_config();
        let names: BTreeSet<String> = mhc_seam_plan_names(&config).into_iter().collect();
        let table = rank_attention_plan_shapes(&config, 0, 4).expect("table");
        for name in &names {
            assert!(
                !table.contains_key(name),
                "{name} must not enter the attention table"
            );
        }
        let mut mhc = 0usize;
        let mut aux = 0usize;
        for name in &names {
            match classify_plan_name(name, &table) {
                PlanClass::Mhc => {
                    assert!(name.contains(".hc_"), "{name}");
                    mhc += 1;
                }
                PlanClass::AttentionAux => {
                    assert!(!name.contains(".hc_"), "{name}");
                    aux += 1;
                }
                other => panic!("{name}: classified {}", other.as_str()),
            }
        }
        assert_eq!(
            (mhc, aux),
            (240, 160),
            "the census keeps naming both classes"
        );
        let attention: BTreeSet<String> = table
            .keys()
            .filter(|name| !is_host_seam_weight(name))
            .cloned()
            .collect();
        assert!(
            names.is_disjoint(&attention),
            "a name bound twice would merge badly"
        );
    }

    #[test]
    fn an_absent_class_refuses_by_name_and_a_short_class_names_the_tensor() {
        let config = released_config();
        let expected: BTreeSet<String> = mhc_seam_plan_names(&config).into_iter().collect();
        let none: BTreeSet<String> = BTreeSet::new();
        let error = check_bind_coverage("mhc_seam", &expected, &none, &none).expect_err("absent");
        assert!(
            error.contains("mhc_seam bind coverage: 400 expected, 0 bound, 0 host seams excluded"),
            "{error}"
        );
        assert!(error.contains("layers.0.hc_attn_fn"), "{error}");
        let mut short = expected.clone();
        assert!(short.remove("layers.7.hc_ffn_scale"));
        let error = check_bind_coverage("mhc_seam", &expected, &short, &none).expect_err("short");
        assert!(error.contains("399 bound"), "{error}");
        assert!(error.contains("layers.7.hc_ffn_scale"), "{error}");
        assert_eq!(expected.len(), mhc_seam_plan_names(&config).len());
    }
}
