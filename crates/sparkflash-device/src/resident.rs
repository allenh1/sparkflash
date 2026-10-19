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
//! The design's per-rank budget (docs/build/tp4-design.md section 6.3) is 72.31 GiB. This plan
//! computes the same categories from the released headers; docs/build/tp4-design.md records the
//! comparison instead of implying the estimate was re-measured here.

use std::collections::BTreeMap;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
}
