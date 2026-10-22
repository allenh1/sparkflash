//! The per-rank resident weight plan for the full chain: every checkpoint tensor the LLM loads,
//! with the TP=4 sharding rule the reference uses, so a 4-rank run reads the released checkpoint in
//! place and needs no repacked copy.
//!
//! [\`ShardPlan\`](crate::shard::ShardPlan) already verifies one tensor's slice and reads it as byte
//! ranges. This module is the layer above it: it classifies *every* released tensor name (attention,
//! routed and shared experts, router, norms, indexer/compressor, Engram, mHC, MTP/DSPark heads,
//! embedding and head) into a [\`ResidentGroup\`], records the rule and placement, sums the per-rank
//! bytes, and hands the resident subset to a \`ShardPlan\` for the actual read. The Engram tables are
//! the one group that is **not** resident: the released checkpoint keeps 188.83 GiB of them, so
//! they stay on NVMe and this plan only accounts for them.
//!
//! Classification is exhaustive on purpose: a tensor name the released checkpoint grows that this
//! module does not know is a refusal, not a silent skip, because a silently skipped tensor is a
//! missing weight at run time. The only names deliberately not planned are \`vision.*\` and
//! \`aligner.*\`, which the reference's LLM class does not load.
//!
//! The plan is header-only: [\`ResidentPlan::plan\`] needs a name, dtype and shape per tensor and
//! touches no data, so the full 40-layer plan is testable on a host that has only the tensor index.
//! [\`ResidentPlan::read_all\`] is the in-place read and is what a node runs; it delegates to
//! \`ShardPlan\`, so the range arithmetic has one implementation.
//!
//! The design's per-rank budget (docs/archive/build/tp4-design.md section 6.3) is 72.31 GiB. This plan
//! computes the same categories from the released headers; docs/archive/build/tp4-design.md records the
//! comparison instead of implying the estimate was re-measured here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sparkflash_engine::safetensors::SafeTensorsSource;

use crate::shard::{ShardError, ShardPlan, ShardRule};

/// The linear-attention layers the released checkpoint stores (\`layers.0\` through \`layers.39\`).
pub const CHAIN_LAYERS: usize = 40;

/// The MTP / DSPark layers the released checkpoint stores as \`mtp.0\` through \`mtp.2\`.
pub const MTP_LAYERS: usize = 3;

/// The bytes per element for a safetensors dtype word, or \`None\` when this module has not seen it.
///
/// The packed FP4 expert weights are stored as \`I8\` (two FP4 values per byte, already packed in the
/// header shape), so one stored element is one byte there as everywhere their dtype says.
#[must_use]
pub fn dtype_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

/// Whether a planned tensor is read once into device memory or left on NVMe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Placement {
    /// Read once at start-up and kept in device memory.
    Resident,
    /// Never resident: read on demand from the Engram tables on NVMe.
    Disk,
}

impl Placement {
    /// A one-word name for evidence.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::Disk => "disk",
        }
    }
}

/// The weight group a tensor belongs to, as the design's per-rank table groups them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResidentGroup {
    /// Attention projections every rank holds whole (\`wq_a\`, \`wkv\`, q/kv norms).
    AttentionReplicated,
    /// Attention projections split on the output axis (\`wq_b\`, \`wo_a\`).
    AttentionColumn,
    /// The attention output projection split on the input axis (\`wo_b\`).
    AttentionRow,
    /// The per-head attention sink, split by head.
    AttentionSink,
    /// Indexer and compressor tensors, replicated because the reference does not shard them.
    AttentionAux,
    /// The per-layer norms.
    Norms,
    /// The mHC / hyper-connection tensors.
    Mhc,
    /// The router gate weights and biases.
    Router,
    /// Routed-expert \`w1\`/\`w3\` (packed FP4) split on the intermediate axis.
    ExpertsColumn,
    /// Routed-expert \`w2\` split on the input axis.
    ExpertsRow,
    /// Shared-expert \`w1\`/\`w3\` split on the intermediate axis.
    SharedColumn,
    /// Shared-expert \`w2\` split on the input axis.
    SharedRow,
    /// The vocabulary-parallel embedding and LM head.
    Embedding,
    /// The Engram hash tables, which stay on NVMe.
    EngramTable,
    /// The non-table Engram projections, replicated.
    EngramOther,
    /// The MTP/DSPark \`main_proj\`, markov/confidence heads and norms.
    MtpHeads,
    /// The image marker rows and the final norm, replicated.
    SpecialTokens,
}

impl ResidentGroup {
    /// A one-word name for evidence.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AttentionReplicated => "attention-replicated",
            Self::AttentionColumn => "attention-column",
            Self::AttentionRow => "attention-row",
            Self::AttentionSink => "attention-sink",
            Self::AttentionAux => "attention-aux",
            Self::Norms => "norms",
            Self::Mhc => "mhc",
            Self::Router => "router",
            Self::ExpertsColumn => "experts-column",
            Self::ExpertsRow => "experts-row",
            Self::SharedColumn => "shared-column",
            Self::SharedRow => "shared-row",
            Self::Embedding => "embedding",
            Self::EngramTable => "engram-table",
            Self::EngramOther => "engram-other",
            Self::MtpHeads => "mtp-heads",
            Self::SpecialTokens => "special-tokens",
        }
    }
}

/// What one tensor name means for the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    /// The design group.
    pub group: ResidentGroup,
    /// Where the tensor lives.
    pub placement: Placement,
    /// The sharding rule; \`None\` for a disk-only tensor whose on-demand read is the Engram packer's.
    pub rule: Option<ShardRule>,
}

impl Classification {
    const fn resident(group: ResidentGroup, rule: ShardRule) -> Self {
        Self {
            group,
            placement: Placement::Resident,
            rule: Some(rule),
        }
    }

    const fn disk(group: ResidentGroup) -> Self {
        Self {
            group,
            placement: Placement::Disk,
            rule: None,
        }
    }
}

/// One tensor's stored header: the only input the plan needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderEntry {
    /// The checkpoint name.
    pub name: String,
    /// The safetensors dtype word.
    pub dtype: String,
    /// The stored shape.
    pub shape: Vec<usize>,
}

impl HeaderEntry {
    /// Builds one entry.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.into(),
            shape,
        }
    }

    /// The stored byte length implied by the dtype and shape.
    ///
    /// # Errors
    /// [\`ShardError::Plan\`] when the dtype is unknown or a dimension is zero.
    pub fn byte_len(&self) -> Result<u64, ShardError> {
        let width = dtype_bytes(&self.dtype).ok_or_else(|| {
            ShardError::Plan(format!(
                "{}: dtype {} has no element width in this plan",
                self.name, self.dtype
            ))
        })?;
        let mut elements = 1u64;
        for dimension in &self.shape {
            if *dimension == 0 {
                return Err(ShardError::Plan(format!(
                    "{}: a zero dimension in stored shape {:?}",
                    self.name, self.shape
                )));
            }
            elements = elements.saturating_mul(*dimension as u64);
        }
        Ok(elements * width as u64)
    }
}

/// Classifies one checkpoint tensor name.
///
/// # Errors
/// [\`ShardError::Plan\`] when the name is structurally unknown: a \`layers.N\`/\`mtp.N\` tail this
/// module does not describe, or a top-level tensor it has never seen.
///
/// A \`vision.*\` or \`aligner.*\` name returns \`Ok(None)\`: the reference's LLM class does not load
/// them, so planning them would be the error.
pub fn classify(name: &str) -> Result<Option<Classification>, ShardError> {
    use ResidentGroup as Group;

    let unknown = || ShardError::Plan(format!("{name}: not a tensor name the resident plan knows"));
    let classified =
        |group: Group, rule: ShardRule| Ok(Some(Classification::resident(group, rule)));

    if name.starts_with("vision.") || name.starts_with("aligner.") {
        return Ok(None);
    }
    match name {
        "embed.weight" | "head.weight" => return classified(Group::Embedding, ShardRule::Column),
        "norm.weight" | "image_start" | "image_end" | "image_newline" => {
            return classified(Group::SpecialTokens, ShardRule::Replicated);
        }
        _ => {}
    }

    let tail = match name.split_once('.') {
        Some(("layers", rest)) => rest.split_once('.').map(|(_, tail)| tail),
        Some(("mtp", rest)) => rest.split_once('.').map(|(_, tail)| tail),
        _ => None,
    };
    let Some(tail) = tail else {
        return Err(unknown());
    };

    // Attention body. The routed/shared/router/mHC/norm tails are shared with the MTP layers.
    if let Some(rest) = tail.strip_prefix("attn.") {
        return match rest {
            "wq_a.weight" | "wq_a.scale" | "wkv.weight" | "wkv.scale" | "q_norm.weight"
            | "kv_norm.weight" => classified(Group::AttentionReplicated, ShardRule::Replicated),
            "wq_b.weight" | "wq_b.scale" | "wo_a.weight" | "wo_a.scale" => {
                classified(Group::AttentionColumn, ShardRule::Column)
            }
            "wo_b.weight" | "wo_b.scale" => classified(Group::AttentionRow, ShardRule::Row),
            "attn_sink" => classified(Group::AttentionSink, ShardRule::Column),
            other if other.starts_with("indexer.") || other.starts_with("compressor.") => {
                classified(Group::AttentionAux, ShardRule::Replicated)
            }
            _ => Err(unknown()),
        };
    }
    if tail == "attn_norm.weight" || tail == "ffn_norm.weight" || tail == "norm.weight" {
        return classified(
            if tail == "norm.weight" {
                Group::MtpHeads
            } else {
                Group::Norms
            },
            ShardRule::Replicated,
        );
    }
    if tail.starts_with("hc_attn_") || tail.starts_with("hc_ffn_") {
        return classified(Group::Mhc, ShardRule::Replicated);
    }
    if let Some(rest) = tail.strip_prefix("engram.") {
        return match rest {
            "embed.weight" | "embed.scale" => Ok(Some(Classification::disk(Group::EngramTable))),
            "q_weight" | "k_weight" | "wkv.weight" | "wkv.scale" => {
                classified(Group::EngramOther, ShardRule::Replicated)
            }
            _ => Err(unknown()),
        };
    }
    if let Some(rest) = tail.strip_prefix("ffn.experts.") {
        let Some((_, leaf)) = rest.split_once('.') else {
            return Err(unknown());
        };
        return match leaf {
            "w1.weight" | "w1.scale" | "w3.weight" | "w3.scale" => {
                classified(Group::ExpertsColumn, ShardRule::Column)
            }
            "w2.weight" | "w2.scale" => classified(Group::ExpertsRow, ShardRule::Row),
            _ => Err(unknown()),
        };
    }
    if let Some(rest) = tail.strip_prefix("ffn.shared_experts.") {
        return match rest {
            "w1.weight" | "w1.scale" | "w3.weight" | "w3.scale" => {
                classified(Group::SharedColumn, ShardRule::Column)
            }
            "w2.weight" | "w2.scale" => classified(Group::SharedRow, ShardRule::Row),
            _ => Err(unknown()),
        };
    }
    if let Some(rest) = tail.strip_prefix("ffn.gate.") {
        return match rest {
            "weight" | "bias" | "bias_vl" => classified(Group::Router, ShardRule::Replicated),
            _ => Err(unknown()),
        };
    }
    match tail {
        "main_proj.weight"
        | "main_proj.scale"
        | "main_norm.weight"
        | "markov_head.embed.weight"
        | "markov_head.head.weight"
        | "confidence_head.proj.weight" => classified(Group::MtpHeads, ShardRule::Replicated),
        _ => Err(unknown()),
    }
}

/// One planned tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentShard {
    /// The checkpoint name.
    pub name: String,
    /// The design group.
    pub group: ResidentGroup,
    /// Where the tensor lives.
    pub placement: Placement,
    /// The sharding rule, or \`None\` for a disk-only tensor.
    pub rule: Option<ShardRule>,
    /// The safetensors dtype word.
    pub dtype: String,
    /// The stored shape.
    pub stored_shape: Vec<usize>,
    /// The stored byte length.
    pub stored_bytes: u64,
    /// The bytes this rank reads for a resident tensor, or 0 for disk.
    pub shard_bytes: u64,
    /// This rank's slice shape, or empty for disk.
    pub shard_shape: Vec<usize>,
    /// The number of byte ranges the read issues, or 0 for disk.
    pub ranges: usize,
}

/// The device operand layout a planned weight offers a computed kernel.
///
/// Derived from the plan record alone - dtype, group and shard shape - never from a pointer or an
/// arena byte count, so a census that reads it cannot disagree with the plan silently. It is the
/// same predicate the arena upload and the attention bind use (`crate::arena::b12x_layout`),
/// exposed as a named value so a forward loop can state which layout it expects per tensor and
/// refuse a bind that disagrees instead of guessing from pointer arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandLayout {
    /// FP8 E4M3 codes in B12x 32x32 micro-tiles plus the `[ceil(cols/32)][ceil(inner/32)]` ue8m0
    /// scale grid: two distinct buffers, the pair a B12x-weight GEMM binds.
    B12x {
        /// Output rows (the kernel's `cols`; the checkpoint's first stored dimension).
        cols: usize,
        /// Reduction width (the kernel's `inner`; the checkpoint's second stored dimension).
        inner: usize,
        /// The tiled weight bytes, padding included (`mxfp8_ref::b12x_buffer_bytes`).
        weight_bytes: u64,
        /// The scale-grid bytes (`ceil(cols/32) * ceil(inner/32)`).
        scale_bytes: u64,
    },
    /// Bytes read as stored: no B12x tiling and no scale grid (a host seam, an FP4 expert matrix,
    /// or a stored-dtype weight that only a stored kernel may read).
    Stored {
        /// The safetensors dtype word the plan carries.
        dtype: String,
        /// The bytes this rank's slice holds.
        bytes: u64,
    },
}

impl ResidentShard {
    /// The device operand layout this planned tensor offers, from the plan record only.
    #[must_use]
    pub fn operand_layout(&self) -> OperandLayout {
        use crate::arena::b12x_layout;
        match b12x_layout(self) {
            Some((cols, inner)) => OperandLayout::B12x {
                cols,
                inner,
                weight_bytes: sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes(cols, inner) as u64,
                scale_bytes: (cols.div_ceil(32) * inner.div_ceil(32)) as u64,
            },
            None => OperandLayout::Stored {
                dtype: self.dtype.clone(),
                bytes: self.shard_bytes,
            },
        }
    }
}

impl OperandLayout {
    /// Whether this is the B12x pair (`true`) or stored bytes (`false`).
    #[must_use]
    pub fn is_b12x(&self) -> bool {
        matches!(self, Self::B12x { .. })
    }

    /// The layout name, for evidence and log lines.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::B12x { .. } => "b12x",
            Self::Stored { .. } => "stored",
        }
    }
}

/// One group's totals, in evidence order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupTotals {
    /// The group.
    pub group: ResidentGroup,
    /// How many tensors the group holds.
    pub tensors: usize,
    /// The placement every tensor in the group shares.
    pub placement: Placement,
    /// The rule every tensor in the group shares, or \`None\` for disk.
    pub rule: Option<ShardRule>,
    /// This rank's resident bytes in the group (0 for disk).
    pub shard_bytes: u64,
    /// The stored bytes the group accounts for (the disk figure for the Engram tables).
    pub stored_bytes: u64,
    /// The byte ranges the group's read issues.
    pub ranges: usize,
}

/// Compares a planned operand layout with a bind's observed pointer pair and names the tensor
/// when they disagree, before anything launches.
///
/// This is the census half of the four-rank operand path: the arena bind reports the two device
/// pointers it bound per weight, and the forward refuses a tensor whose pair cannot be the planned
/// layout. B12x needs two live and distinct buffers (the tiled codes and the scale grid); a stored
/// tensor repeats its one pointer in the scale slot. A null pointer, a shared pointer for a B12x
/// operand, or a distinct pointer for a stored tensor is a reported refusal carrying the tensor
/// name, never a launch.
///
/// # Errors
/// The refusal string names the tensor, the planned layout and the observed pair.
pub fn check_operand_bind(
    shard: &ResidentShard,
    weight_pointer: u64,
    scale_pointer: u64,
) -> Result<OperandLayout, String> {
    let layout = shard.operand_layout();
    match &layout {
        OperandLayout::B12x { .. } => {
            if weight_pointer == 0 || scale_pointer == 0 {
                return Err(format!(
                    "{}: plan says {} but the bind left a null pointer (weight {weight_pointer:#x}, scale {scale_pointer:#x}); refusing to launch",
                    shard.name,
                    layout.as_str()
                ));
            }
            if weight_pointer == scale_pointer {
                return Err(format!(
                    "{}: plan says {} (two buffers) but weight and scale bound the same pointer {weight_pointer:#x}; refusing to launch",
                    shard.name,
                    layout.as_str()
                ));
            }
        }
        OperandLayout::Stored { .. } => {
            if weight_pointer == 0 {
                return Err(format!(
                    "{}: plan says {} but the bind left a null weight pointer; refusing to launch",
                    shard.name,
                    layout.as_str()
                ));
            }
            if scale_pointer != 0 && scale_pointer != weight_pointer {
                return Err(format!(
                    "{}: plan says {} but the bind returned a distinct scale pointer {scale_pointer:#x} (weight {weight_pointer:#x}); refusing to launch",
                    shard.name,
                    layout.as_str()
                ));
            }
        }
    }
    Ok(layout)
}

/// The full per-rank resident plan.
#[derive(Debug, Clone)]
pub struct ResidentPlan {
    rank: usize,
    world: usize,
    shards: Vec<ResidentShard>,
}

impl ResidentPlan {
    /// Plans one rank from tensor headers.
    ///
    /// # Errors
    /// [\`ShardError::Plan\`] for an impossible rank/world, a duplicate name, an unknown name, an
    /// unknown dtype, or a shape that does not divide by the world size.
    pub fn plan(
        rank: usize,
        world: usize,
        entries: impl IntoIterator<Item = HeaderEntry>,
    ) -> Result<Self, ShardError> {
        if world == 0 {
            return Err(ShardError::Plan(
                "a TP plan needs at least one rank".to_owned(),
            ));
        }
        if rank >= world {
            return Err(ShardError::Plan(format!(
                "rank {rank} is outside a TP world of {world}"
            )));
        }
        let mut shards = Vec::new();
        for entry in entries {
            let Some(classification) = classify(&entry.name)? else {
                continue;
            };
            let stored_bytes = entry.byte_len()?;
            let (shard_bytes, shard_shape, ranges) = match classification.rule {
                None => (0, Vec::new(), 0),
                Some(rule) => {
                    let (bytes, ranges, shape) =
                        slice_geometry(world, &entry.name, rule, &entry.shape, stored_bytes)?;
                    (bytes, shape, ranges)
                }
            };
            shards.push(ResidentShard {
                name: entry.name,
                group: classification.group,
                placement: classification.placement,
                rule: classification.rule,
                dtype: entry.dtype,
                stored_shape: entry.shape,
                stored_bytes,
                shard_bytes,
                shard_shape,
                ranges,
            });
        }
        shards.sort_by(|left, right| left.name.cmp(&right.name));
        for pair in shards.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(ShardError::Plan(format!(
                    "{} was planned twice; the checkpoint index names each tensor once",
                    pair[0].name
                )));
            }
        }
        Ok(Self {
            rank,
            world,
            shards,
        })
    }

    /// This plan's rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// This plan's world size.
    #[must_use]
    pub fn world(&self) -> usize {
        self.world
    }

    /// The planned tensors, in name order.
    #[must_use]
    pub fn shards(&self) -> &[ResidentShard] {
        &self.shards
    }

    /// The device operand layout of one planned tensor, by name, from the plan record alone.
    ///
    /// `None` when the plan does not carry the name (a host seam or a tensor this rank does not
    /// serve); a caller must refuse that, not guess.
    #[must_use]
    pub fn operand_layout(&self, name: &str) -> Option<OperandLayout> {
        self.shards
            .iter()
            .find(|shard| shard.name == name)
            .map(ResidentShard::operand_layout)
    }

    /// The bytes this rank reads into device memory.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.shards
            .iter()
            .filter(|shard| shard.placement == Placement::Resident)
            .map(|shard| shard.shard_bytes)
            .sum()
    }

    /// The bytes the plan keeps on NVMe (the Engram tables).
    #[must_use]
    pub fn disk_bytes(&self) -> u64 {
        self.shards
            .iter()
            .filter(|shard| shard.placement == Placement::Disk)
            .map(|shard| shard.stored_bytes)
            .sum()
    }

    /// The number of resident tensors.
    #[must_use]
    pub fn resident_tensors(&self) -> usize {
        self.shards
            .iter()
            .filter(|shard| shard.placement == Placement::Resident)
            .count()
    }

    /// The number of disk tensors.
    #[must_use]
    pub fn disk_tensors(&self) -> usize {
        self.shards
            .iter()
            .filter(|shard| shard.placement == Placement::Disk)
            .count()
    }

    /// The byte ranges the resident read issues.
    #[must_use]
    pub fn ranges(&self) -> usize {
        self.shards.iter().map(|shard| shard.ranges).sum()
    }

    /// Per-group totals in group-name order.
    #[must_use]
    pub fn groups(&self) -> Vec<GroupTotals> {
        let mut totals: BTreeMap<&'static str, GroupTotals> = BTreeMap::new();
        for shard in &self.shards {
            let entry = totals
                .entry(shard.group.as_str())
                .or_insert_with(|| GroupTotals {
                    group: shard.group,
                    tensors: 0,
                    placement: shard.placement,
                    rule: shard.rule,
                    shard_bytes: 0,
                    stored_bytes: 0,
                    ranges: 0,
                });
            entry.tensors += 1;
            entry.shard_bytes += shard.shard_bytes;
            entry.stored_bytes += shard.stored_bytes;
            entry.ranges += shard.ranges;
        }
        totals.into_values().collect()
    }

    /// A stable hash of the plan's tensor names and rules, for cross-rank evidence comparison.
    ///
    /// Every rank must plan the same set of names with the same rules; only the slice offsets
    /// differ. Two ranks whose hashes differ have drifted apart before a byte was read.
    #[must_use]
    pub fn plan_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for shard in &self.shards {
            for byte in shard.name.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash ^= shard.rule.map_or(0u64, |rule| match rule {
                ShardRule::Replicated => 1,
                ShardRule::Column => 2,
                ShardRule::Row => 3,
            });
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// The verified read-in-place plan for the resident subset.
    ///
    /// # Errors
    /// [\`ShardError::Plan\`] if a resident tensor somehow has no rule.
    pub fn shard_plan(&self) -> Result<ShardPlan, ShardError> {
        let mut plan = ShardPlan::new(self.rank, self.world)?;
        for shard in &self.shards {
            if shard.placement != Placement::Resident {
                continue;
            }
            let rule = shard.rule.ok_or_else(|| {
                ShardError::Plan(format!("{}: resident tensor without a rule", shard.name))
            })?;
            plan.insert(shard.name.clone(), rule);
        }
        Ok(plan)
    }

    /// Reads this rank's resident slices in place and hashes them.
    ///
    /// The read is [\`ShardPlan::read_all\`], so the range arithmetic and the per-tensor header
    /// verification are the ones already checked by the S5.1 stage.
    ///
    /// # Errors
    /// [\`ShardError\`] when the plan cannot be built, a header disagrees, or the checkpoint read
    /// fails.
    pub fn read_all(&self, source: &SafeTensorsSource) -> Result<ReadEvidence, ShardError> {
        let slices = self.shard_plan()?.read_all(source)?;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut bytes = 0u64;
        for slice in &slices {
            bytes += slice.bytes.len() as u64;
            for byte in &slice.bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Ok(ReadEvidence {
            tensors: slices.len(),
            bytes,
            hash,
        })
    }
}

/// What an in-place read measured: the bytes and a content hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadEvidence {
    /// How many tensors were read.
    pub tensors: usize,
    /// How many bytes were read.
    pub bytes: u64,
    /// The FNV-1a 64 hash of the bytes, in name order.
    pub hash: u64,
}

/// Shape and byte arithmetic for one slice, mirroring [\`ShardPlan\`](crate::shard::ShardPlan).
///
/// Kept here because the header-only plan must compute the same numbers without a
/// [\`SafeTensorsSource\`](sparkflash_engine::safetensors::SafeTensorsSource); the cross-check test
/// compares this arithmetic against \`ShardPlan::verify\` on the same synthetic container so the two
/// cannot drift.
fn slice_geometry(
    world: usize,
    name: &str,
    rule: ShardRule,
    stored_shape: &[usize],
    stored_bytes: u64,
) -> Result<(u64, usize, Vec<usize>), ShardError> {
    let rows = stored_shape.first().copied().unwrap_or(0) as u64;
    if rows == 0 || !stored_bytes.is_multiple_of(rows) {
        return Err(ShardError::Plan(format!(
            "{name}: {stored_bytes} stored bytes do not divide into {rows} rows"
        )));
    }
    let row_bytes = stored_bytes / rows;
    let mut shard_shape = stored_shape.to_vec();
    match rule {
        ShardRule::Replicated => Ok((stored_bytes, 1, shard_shape)),
        ShardRule::Column => {
            if !rows.is_multiple_of(world as u64) {
                return Err(ShardError::Plan(format!(
                    "{name}: {rows} stored rows do not divide into {world} ranks"
                )));
            }
            shard_shape[0] /= world;
            Ok((row_bytes * (rows / world as u64), 1, shard_shape))
        }
        ShardRule::Row => {
            if stored_shape.len() != 2 {
                return Err(ShardError::Plan(format!(
                    "{name}: a row-parallel tensor must be stored two-dimensional, stored shape {stored_shape:?}"
                )));
            }
            if !(stored_shape[1] as u64).is_multiple_of(world as u64)
                || !row_bytes.is_multiple_of(world as u64)
            {
                return Err(ShardError::Plan(format!(
                    "{name}: {} stored columns do not divide into {world} ranks",
                    stored_shape[1]
                )));
            }
            shard_shape[1] /= world;
            Ok((
                row_bytes / world as u64 * rows,
                usize::try_from(rows).unwrap_or(usize::MAX),
                shard_shape,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, dtype: &str, shape: &[usize]) -> HeaderEntry {
        HeaderEntry::new(name, dtype, shape.to_vec())
    }

    #[test]
    fn every_released_name_shape_is_classified_with_the_reference_rule() {
        let cases: &[(&str, ResidentGroup, Placement, Option<ShardRule>)] = &[
            (
                "layers.0.attn.wq_a.weight",
                ResidentGroup::AttentionReplicated,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "layers.0.attn.wq_b.scale",
                ResidentGroup::AttentionColumn,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
            (
                "layers.0.attn.wo_b.weight",
                ResidentGroup::AttentionRow,
                Placement::Resident,
                Some(ShardRule::Row),
            ),
            (
                "layers.0.attn.attn_sink",
                ResidentGroup::AttentionSink,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
            (
                "layers.2.attn.indexer.weights_proj.weight",
                ResidentGroup::AttentionAux,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "layers.2.attn.compressor.wkv.weight",
                ResidentGroup::AttentionAux,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "layers.0.ffn.experts.383.w1.scale",
                ResidentGroup::ExpertsColumn,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
            (
                "layers.0.ffn.experts.383.w2.weight",
                ResidentGroup::ExpertsRow,
                Placement::Resident,
                Some(ShardRule::Row),
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight",
                ResidentGroup::SharedColumn,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
            (
                "layers.0.ffn.shared_experts.w2.scale",
                ResidentGroup::SharedRow,
                Placement::Resident,
                Some(ShardRule::Row),
            ),
            (
                "layers.0.ffn.gate.bias_vl",
                ResidentGroup::Router,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "layers.1.engram.embed.weight",
                ResidentGroup::EngramTable,
                Placement::Disk,
                None,
            ),
            (
                "layers.14.engram.embed.scale",
                ResidentGroup::EngramTable,
                Placement::Disk,
                None,
            ),
            (
                "layers.1.engram.wkv.weight",
                ResidentGroup::EngramOther,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "layers.0.hc_attn_scale",
                ResidentGroup::Mhc,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "embed.weight",
                ResidentGroup::Embedding,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
            (
                "head.weight",
                ResidentGroup::Embedding,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
            (
                "norm.weight",
                ResidentGroup::SpecialTokens,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "mtp.2.main_proj.weight",
                ResidentGroup::MtpHeads,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "mtp.2.markov_head.embed.weight",
                ResidentGroup::MtpHeads,
                Placement::Resident,
                Some(ShardRule::Replicated),
            ),
            (
                "mtp.0.attn.wo_a.weight",
                ResidentGroup::AttentionColumn,
                Placement::Resident,
                Some(ShardRule::Column),
            ),
        ];
        for (name, group, placement, rule) in cases {
            let classification = classify(name)
                .unwrap_or_else(|error| panic!("{name} must classify: {error}"))
                .expect("planned");
            assert_eq!(classification.group, *group, "{name}");
            assert_eq!(classification.placement, *placement, "{name}");
            assert_eq!(classification.rule, *rule, "{name}");
        }
    }

    #[test]
    fn the_vision_tower_is_deliberately_not_loaded_and_a_strange_name_is_refused() {
        assert!(
            classify("vision.blocks.0.attn.wqkv.weight")
                .expect("known prefix")
                .is_none()
        );
        assert!(
            classify("aligner.w1.weight")
                .expect("known prefix")
                .is_none()
        );
        assert!(matches!(
            classify("layers.0.attn.wq_z.weight"),
            Err(ShardError::Plan(_))
        ));
        assert!(matches!(
            classify("layers.0.ffn.experts.1.w4.weight"),
            Err(ShardError::Plan(_))
        ));
        assert!(matches!(classify("embed.bias"), Err(ShardError::Plan(_))));
    }

    #[test]
    fn the_plan_sums_one_quarter_of_the_stored_bytes_for_the_split_groups() {
        let entries = vec![
            entry("embed.weight", "BF16", &[8, 2]),
            entry("norm.weight", "BF16", &[8]),
            entry("layers.0.attn.wq_a.weight", "F8_E4M3", &[4, 2]),
            entry("layers.0.attn.wq_b.weight", "F8_E4M3", &[8, 4]),
            entry("layers.0.attn.wo_b.weight", "F8_E4M3", &[4, 8]),
            entry("layers.0.ffn.experts.0.w1.weight", "I8", &[8, 4]),
            entry("layers.0.ffn.experts.0.w2.weight", "I8", &[4, 8]),
            entry("layers.1.engram.embed.weight", "F8_E4M3", &[8, 4]),
        ];
        let plan = ResidentPlan::plan(1, 4, entries).expect("plan");
        // embed 8, norm 16, wq_a 8, wq_b 8, wo_b 8, expert w1 8, expert w2 8.
        assert_eq!(plan.resident_bytes(), 64);
        assert_eq!(plan.disk_bytes(), 32);
        assert_eq!(plan.resident_tensors(), 7);
        assert_eq!(plan.disk_tensors(), 1);
        let groups = plan.groups();
        let embedding = groups
            .iter()
            .find(|group| group.group == ResidentGroup::Embedding)
            .expect("embedding");
        assert_eq!(embedding.shard_bytes, 8);
        assert_eq!(embedding.placement, Placement::Resident);
        let table = groups
            .iter()
            .find(|group| group.group == ResidentGroup::EngramTable)
            .expect("engram table");
        assert_eq!(table.placement, Placement::Disk);
        assert_eq!(table.shard_bytes, 0);
        assert_eq!(table.stored_bytes, 32);
    }

    #[test]
    fn the_plan_hash_is_rank_independent_and_rule_sensitive() {
        let entries = || {
            vec![
                entry("embed.weight", "BF16", &[8, 2]),
                entry("layers.0.attn.wq_b.weight", "F8_E4M3", &[8, 4]),
            ]
        };
        let rank0 = ResidentPlan::plan(0, 4, entries()).expect("rank 0");
        let rank1 = ResidentPlan::plan(1, 4, entries()).expect("rank 1");
        assert_eq!(rank0.plan_hash(), rank1.plan_hash());
        let mut changed = entries();
        changed[0].name = "head.weight".to_owned();
        let other = ResidentPlan::plan(0, 4, changed).expect("other");
        assert_ne!(rank0.plan_hash(), other.plan_hash());
    }

    #[test]
    fn an_impossible_rank_world_duplicate_or_undividing_shape_is_refused() {
        assert!(ResidentPlan::plan(0, 0, Vec::new()).is_err());
        assert!(ResidentPlan::plan(4, 4, Vec::new()).is_err());
        assert!(matches!(
            ResidentPlan::plan(
                0,
                4,
                vec![
                    entry("embed.weight", "BF16", &[8, 2]),
                    entry("embed.weight", "BF16", &[8, 2]),
                ]
            ),
            Err(ShardError::Plan(_))
        ));
        assert!(matches!(
            ResidentPlan::plan(0, 4, vec![entry("embed.weight", "BF16", &[6, 2])]),
            Err(ShardError::Plan(_))
        ));
        assert!(matches!(
            ResidentPlan::plan(
                0,
                4,
                vec![entry("layers.0.attn.wo_b.weight", "F8_E4M3", &[4, 6])]
            ),
            Err(ShardError::Plan(_))
        ));
        assert!(matches!(
            ResidentPlan::plan(0, 4, vec![entry("embed.weight", "F9_FANTASY", &[8, 2])]),
            Err(ShardError::Plan(_))
        ));
    }

    /// Writes a one-shard safetensors container for the entries and opens it, so the header-only
    /// plan can be compared with the verified ShardPlan on identical inputs.
    fn synthetic_source(entries: &[HeaderEntry]) -> (tempfile::TempDir, SafeTensorsSource) {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard_name = "model-00001-of-00001.safetensors";
        let mut header = String::from("{");
        let mut offset = 0u64;
        let mut data = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            let bytes = vec![0u8; entry.byte_len().expect("byte length") as usize];
            if index > 0 {
                header.push(',');
            }
            let begin = offset;
            offset += bytes.len() as u64;
            let shape = entry
                .shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                entry.name, entry.dtype, shape, begin, offset
            ));
            data.extend_from_slice(&bytes);
        }
        header.push('}');
        let mut header = header.into_bytes();
        while (8 + header.len()) % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);
        std::fs::write(dir.path().join(shard_name), file).expect("write the synthetic shard");
        let weight_map = entries
            .iter()
            .map(|entry| format!("\"{}\":\"{shard_name}\"", entry.name))
            .collect::<Vec<_>>()
            .join(",");
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &index_path,
            format!("{{\"metadata\":{{}},\"weight_map\":{{{weight_map}}}}}"),
        )
        .expect("write the synthetic index");
        let source = SafeTensorsSource::open(dir.path(), &index_path).expect("open the source");
        (dir, source)
    }

    #[test]
    fn the_header_only_geometry_matches_the_shard_plans_verified_geometry() {
        let entries = vec![
            entry("embed.weight", "BF16", &[16, 4]),
            entry("norm.weight", "BF16", &[8]),
            entry("layers.0.attn.wq_a.weight", "F8_E4M3", &[4, 2]),
            entry("layers.0.attn.wq_b.scale", "F8_E8M0", &[8, 1]),
            entry("layers.0.attn.wo_b.weight", "F8_E4M3", &[4, 8]),
            entry("layers.0.ffn.experts.0.w1.weight", "I8", &[8, 4]),
            entry("layers.0.ffn.experts.0.w2.scale", "F8_E8M0", &[4, 8]),
            entry("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3", &[4, 8]),
            entry("layers.0.ffn.gate.weight", "BF16", &[4, 4]),
            entry("layers.1.engram.embed.weight", "F8_E4M3", &[8, 4]),
        ];
        let plan = ResidentPlan::plan(2, 4, entries.clone()).expect("plan");
        let (_dir, source) = synthetic_source(&entries);
        let verified = plan
            .shard_plan()
            .expect("shard plan")
            .verify(&source)
            .expect("verify");
        assert_eq!(verified.len(), plan.resident_tensors());
        for (shard, expected) in plan
            .shards()
            .iter()
            .filter(|shard| shard.placement == Placement::Resident)
            .zip(verified.iter())
        {
            assert_eq!(shard.name, expected.name);
            assert_eq!(Some(expected.rule), shard.rule, "{}", shard.name);
            assert_eq!(shard.shard_bytes, expected.shard_bytes, "{}", shard.name);
            assert_eq!(shard.shard_shape, expected.shard_shape, "{}", shard.name);
            assert_eq!(shard.ranges, expected.ranges, "{}", shard.name);
        }
    }

    #[test]
    fn reading_in_place_returns_the_planned_bytes_for_each_rank() {
        let entries = vec![
            entry("embed.weight", "BF16", &[16, 4]),
            entry("layers.0.attn.wo_b.weight", "F8_E4M3", &[4, 8]),
        ];
        let (_dir, source) = synthetic_source(&entries);
        for rank in 0..4 {
            let plan = ResidentPlan::plan(rank, 4, entries.clone()).expect("plan");
            let read = plan.read_all(&source).expect("read");
            assert_eq!(read.tensors, 2);
            assert_eq!(read.bytes, plan.resident_bytes());
        }
    }

    #[test]
    fn the_operand_layout_is_the_b12x_predicate_and_a_stored_tensor_names_its_dtype() {
        // The released layer-0 attention shapes, as the released headers carry them. wq_a/wkv are
        // Replicated and stay whole; wq_b/wo_a split cols and wo_b splits inner on four ranks.
        let entries = vec![
            entry("layers.0.attn.wq_a.weight", "F8_E4M3", &[1280, 5120]),
            entry("layers.0.attn.wq_a.scale", "F8_E8M0", &[40, 160]),
            entry("layers.0.attn.wkv.weight", "F8_E4M3", &[512, 5120]),
            entry("layers.0.attn.wkv.scale", "F8_E8M0", &[16, 160]),
            entry("layers.0.attn.wq_b.weight", "F8_E4M3", &[32768, 1280]),
            entry("layers.0.attn.wq_b.scale", "F8_E8M0", &[1024, 40]),
            entry("layers.0.attn.wo_a.weight", "F8_E4M3", &[8192, 4096]),
            entry("layers.0.attn.wo_a.scale", "F8_E8M0", &[256, 128]),
            entry("layers.0.attn.wo_b.weight", "F8_E4M3", &[5120, 8192]),
            entry("layers.0.attn.wo_b.scale", "F8_E8M0", &[160, 256]),
            entry("layers.0.attn.q_norm.weight", "BF16", &[1280]),
            entry("layers.0.ffn.experts.0.w2.weight", "I8", &[5120, 1152]),
            entry("layers.0.ffn.experts.0.w2.scale", "F8_E8M0", &[5120, 72]),
        ];
        let plan = ResidentPlan::plan(0, 4, entries).expect("released layer-0 plan");
        let wq_a = plan
            .operand_layout("layers.0.attn.wq_a.weight")
            .expect("wq_a is planned");
        assert_eq!(
            wq_a,
            OperandLayout::B12x {
                cols: 1280,
                inner: 5120,
                weight_bytes: 1280 * 5120,
                scale_bytes: 40 * 160,
            },
            "wq_a is F8_E4M3 in the Replicated attention group: B12x, not stored"
        );
        let wkv = plan
            .operand_layout("layers.0.attn.wkv.weight")
            .expect("wkv is planned");
        assert_eq!(
            wkv,
            OperandLayout::B12x {
                cols: 512,
                inner: 5120,
                weight_bytes: 512 * 5120,
                scale_bytes: 16 * 160,
            }
        );
        let wo_b = plan
            .operand_layout("layers.0.attn.wo_b.weight")
            .expect("wo_b is planned");
        assert_eq!(
            wo_b,
            OperandLayout::B12x {
                cols: 5120,
                inner: 2048,
                weight_bytes: 5120 * 2048,
                scale_bytes: 160 * 64,
            },
            "the Row rule splits wo_b's inner axis, not its cols"
        );
        // A BF16 norm is not a weight a B12x kernel can read: stored bytes, named dtype.
        assert_eq!(
            plan.operand_layout("layers.0.attn.q_norm.weight"),
            Some(OperandLayout::Stored {
                dtype: "BF16".to_string(),
                bytes: 1280 * 2,
            })
        );
        // The FP4 expert matrix is I8 (packed two per byte): stored bytes despite being a weight.
        assert_eq!(
            plan.operand_layout("layers.0.ffn.experts.0.w2.weight"),
            Some(OperandLayout::Stored {
                dtype: "I8".to_string(),
                bytes: 5120 * 288,
            })
        );
        // A name the plan does not carry is None, not an invented layout.
        assert_eq!(plan.operand_layout("layers.0.attn.attn_sink"), None);
        assert!(wq_a.is_b12x());
        assert_eq!(wq_a.as_str(), "b12x");
    }

    #[test]
    fn every_released_attention_projection_is_b12x_on_all_four_ranks() {
        // 43 released attention layers (40 chain + 3 MTP), both weight and scale per projection,
        // with the stored shapes the released headers carry. Each rank's slice keeps B12x layout.
        let mut entries = Vec::new();
        for layer in 0..CHAIN_LAYERS {
            let prefix = format!("layers.{layer}.attn");
            for (name, weight, scale) in [
                ("wq_a", [1280usize, 5120], [40usize, 160]),
                ("wq_b", [32768, 1280], [1024, 40]),
                ("wkv", [512, 5120], [16, 160]),
                ("wo_a", [8192, 4096], [256, 128]),
                ("wo_b", [5120, 8192], [160, 256]),
            ] {
                entries.push(entry(
                    &format!("{prefix}.{name}.weight"),
                    "F8_E4M3",
                    &weight,
                ));
                entries.push(entry(&format!("{prefix}.{name}.scale"), "F8_E8M0", &scale));
            }
        }
        for layer in 0..MTP_LAYERS {
            let prefix = format!("mtp.{layer}.attn");
            for (name, weight, scale) in [
                ("wq_a", [1280usize, 5120], [40usize, 160]),
                ("wq_b", [32768, 1280], [1024, 40]),
                ("wkv", [512, 5120], [16, 160]),
                ("wo_a", [8192, 4096], [256, 128]),
                ("wo_b", [5120, 8192], [160, 256]),
            ] {
                entries.push(entry(
                    &format!("{prefix}.{name}.weight"),
                    "F8_E4M3",
                    &weight,
                ));
                entries.push(entry(&format!("{prefix}.{name}.scale"), "F8_E8M0", &scale));
            }
        }
        for rank in 0..4 {
            let plan = ResidentPlan::plan(rank, 4, entries.clone()).expect("plan");
            let mut tiled = 0usize;
            for layer in 0..CHAIN_LAYERS + MTP_LAYERS {
                let prefix = if layer < CHAIN_LAYERS {
                    format!("layers.{layer}.attn")
                } else {
                    format!("mtp.{}.attn", layer - CHAIN_LAYERS)
                };
                for name in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
                    let layout = plan
                        .operand_layout(&format!("{prefix}.{name}.weight"))
                        .expect("every projection is planned");
                    assert!(
                        layout.is_b12x(),
                        "rank {rank} {prefix}.{name}.weight must be B12x, got {layout:?}"
                    );
                    tiled += 1;
                }
            }
            assert_eq!(tiled, (CHAIN_LAYERS + MTP_LAYERS) * 5);
        }
    }

    fn shard_named<'a>(plan: &'a ResidentPlan, name: &str) -> &'a ResidentShard {
        plan.shards()
            .iter()
            .find(|shard| shard.name == name)
            .expect("planned")
    }

    #[test]
    fn the_bind_census_accepts_the_planned_pair_and_names_an_inconsistent_one() {
        let entries = vec![
            entry("layers.0.attn.wq_a.weight", "F8_E4M3", &[1280, 5120]),
            entry("layers.0.attn.wq_a.scale", "F8_E8M0", &[40, 160]),
            entry("layers.0.attn.q_norm.weight", "BF16", &[1280]),
        ];
        let plan = ResidentPlan::plan(0, 4, entries).expect("plan");
        let wq_a = shard_named(&plan, "layers.0.attn.wq_a.weight");
        assert!(
            check_operand_bind(wq_a, 0x1000, 0x2000).is_ok(),
            "two distinct buffers are the B12x pair"
        );
        let same = check_operand_bind(wq_a, 0x1000, 0x1000)
            .expect_err("one buffer cannot be codes and scales");
        assert!(
            same.contains("layers.0.attn.wq_a.weight") && same.contains("refusing to launch"),
            "{same}"
        );
        let null =
            check_operand_bind(wq_a, 0x1000, 0).expect_err("a null scale pointer is refused");
        assert!(
            null.contains("null pointer") && null.contains("wq_a"),
            "{null}"
        );
        let norm = shard_named(&plan, "layers.0.attn.q_norm.weight");
        assert!(
            check_operand_bind(norm, 0x3000, 0x3000).is_ok(),
            "a stored tensor repeats its one pointer in the scale slot"
        );
        let distinct = check_operand_bind(norm, 0x3000, 0x4000)
            .expect_err("a stored tensor has no second buffer");
        assert!(
            distinct.contains("q_norm.weight") && distinct.contains("distinct scale pointer"),
            "{distinct}"
        );
    }

    #[test]
    fn a_fixture_bind_that_disagrees_with_the_plan_is_named_not_launched() {
        // The deliberately inconsistent fixture: the plan carries wq_a as BF16 (a stored operand,
        // so no B12x kernel may read it) while the bind reports the two-pointer pair a B12x kernel
        // would need. The census must name the tensor and refuse, never launch.
        let entries = vec![
            entry("layers.0.attn.wq_a.weight", "BF16", &[1280, 5120]),
            entry("layers.0.attn.wq_a.scale", "F8_E8M0", &[40, 160]),
        ];
        let plan = ResidentPlan::plan(0, 4, entries).expect("plan");
        let wq_a = shard_named(&plan, "layers.0.attn.wq_a.weight");
        assert_eq!(
            wq_a.operand_layout(),
            OperandLayout::Stored {
                dtype: "BF16".to_string(),
                bytes: 1280 * 5120 * 2,
            }
        );
        let refusal = check_operand_bind(wq_a, 0x1000, 0x2000)
            .expect_err("a stored plan operand with two buffers is refused");
        assert!(refusal.contains("layers.0.attn.wq_a.weight"), "{refusal}");
        assert!(refusal.contains("stored"), "{refusal}");
        assert!(refusal.contains("distinct scale pointer"), "{refusal}");
        assert!(refusal.contains("refusing to launch"), "{refusal}");
    }
}
