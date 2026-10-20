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

use std::collections::BTreeMap;

use sparkflash_device::arena::{
    ArenaAllocator, ArenaTensor, ResidentArena, TensorLayout, tile_b12x,
};
use sparkflash_device::resident::{ResidentGroup, ResidentPlan};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes;

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
}

/// Binds every attention-path plan shard the table covers to the rank arena and returns the
/// (weight, scales) pointers the kernels must be launched with.
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
) -> Result<BTreeMap<String, (Deviceptr, Deviceptr)>, String> {
    let mut bound = BTreeMap::new();
    let mut host_seam = 0usize;
    for shard in plan.shards() {
        // Host-seam tensors are row-major BF16 with no scale and no device kernel: they are
        // computed by CompressorCache / IndexerWeights on the host. They are not device weights,
        // so they are excluded from the device bind table rather than bound as B12x or demanding a
        // scale grid.
        let host_seam_tensor = shard.name.contains(".compressor.wgate.")
            || shard.name.contains(".compressor.wkv.")
            || shard.name.contains("indexer.weights_proj.")
            || shard.name.contains("indexer.wk.");
        if host_seam_tensor {
            host_seam += 1;
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
    let device_served = table.len() - host_seam;
    if bound.len() != device_served {
        return Err(format!(
            "bound {} of {} device-served attention tensors ({host_seam} host seams excluded): the plan is missing a tensor the expected-shape table names",
            bound.len(),
            device_served
        ));
    }
    Ok(bound)
}
