//! Per-rank sharding of checkpoint tensors: the plan, its verification and the slice reader.
//!
//! A TP rank must read only its own slice of the checkpoint. The file set is 475 GiB, and a rank
//! that materialised a whole row-parallel tensor before slicing it would read four times the
//! bytes it keeps (the w2 expert tensors alone would add roughly 63 GiB per rank). The plan
//! records which stored axis each tensor shards along; [ShardPlan::verify] checks that axis and
//! the per-rank shape against the shard's own header before any read; [ShardPlan::read] turns a
//! tensor into byte ranges - one contiguous range for a row block, one contiguous range per row
//! for a column block - and hands them to the checkpoint reader.
//!
//! Nothing here allocates device memory or does arithmetic, so the whole module is host-testable
//! against a synthetic safetensors container, and the ignored test at the bottom checks the real
//! released checkpoint's layer-0 shapes when a host has it.

use std::collections::BTreeMap;

use sparkflash_engine::WeightError;
use sparkflash_engine::safetensors::SafeTensorsSource;

/// The checkpoint axis a tensor shards along under tensor parallelism.
///
/// The rules mirror the released reference: column-parallel linears split the output axis,
/// row-parallel linears split the input axis, and the replicated tensors are held whole on every
/// rank (docs/build/tp4-design.md section 3 has the per-tensor map).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardRule {
    /// Every rank holds the whole tensor (the reference's disable_tp / ReplicatedLinear tensors).
    Replicated,
    /// Column-parallel: the output axis (stored axis 0) is split into equal row blocks.
    Column,
    /// Row-parallel: the input axis (stored axis 1) is split into equal column blocks, one
    /// contiguous byte segment per row. A row-parallel tensor also needs the all-reduce after its
    /// GEMM; this module only describes the slice.
    Row,
}

impl ShardRule {
    /// A one-word name for evidence files.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Replicated => "replicated",
            Self::Column => "column",
            Self::Row => "row",
        }
    }
}

/// Why a shard plan could not be built, verified or read.
#[derive(Debug, thiserror::Error)]
pub enum ShardError {
    /// The plan does not describe this tensor, or the degree/shape arithmetic is impossible.
    #[error("shard plan: {0}")]
    Plan(String),
    /// The checkpoint reader failed.
    #[error(transparent)]
    Weights(#[from] WeightError),
}

/// One verified tensor slice, as reported before any bytes are read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTensor {
    /// The checkpoint name.
    pub name: String,
    /// The rule the plan records.
    pub rule: ShardRule,
    /// The dtype word the shard header carries.
    pub dtype: String,
    /// The shape stored on disk (a packed FP4 tensor keeps its packed shape here).
    pub stored_shape: Vec<usize>,
    /// This rank's slice shape.
    pub shard_shape: Vec<usize>,
    /// The bytes this rank will read.
    pub shard_bytes: u64,
    /// The bytes one stored row occupies.
    pub row_bytes: u64,
    /// The number of byte ranges the read will issue.
    pub ranges: usize,
}

/// One rank's slice of a tensor, read from the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredShard {
    /// The checkpoint name.
    pub name: String,
    /// The rule the plan records.
    pub rule: ShardRule,
    /// The dtype word the shard header carries.
    pub dtype: String,
    /// The shape stored on disk.
    pub stored_shape: Vec<usize>,
    /// This rank's slice shape.
    pub shard_shape: Vec<usize>,
    /// The slice bytes, in stored row-major order.
    pub bytes: Vec<u8>,
    /// The number of byte ranges the read issued.
    pub ranges: usize,
}

/// The slice geometry of one tensor, computed from its header alone.
#[derive(Debug, Clone, Copy)]
struct SliceGeometry {
    shard_bytes: u64,
    stored_bytes: u64,
    row_bytes: u64,
    ranges: usize,
}

/// A rank's sharding plan: one rule per checkpoint tensor name.
#[derive(Debug, Clone)]
pub struct ShardPlan {
    tp_rank: usize,
    tp_world: usize,
    rules: BTreeMap<String, ShardRule>,
}

impl ShardPlan {
    /// An empty plan for one rank of a TP world.
    ///
    /// # Errors
    /// [ShardError::Plan] when the world is zero or the rank is outside it.
    pub fn new(tp_rank: usize, tp_world: usize) -> Result<Self, ShardError> {
        if tp_world == 0 {
            return Err(ShardError::Plan(
                "a TP plan needs at least one rank".to_owned(),
            ));
        }
        if tp_rank >= tp_world {
            return Err(ShardError::Plan(format!(
                "rank {tp_rank} is outside a TP world of {tp_world}"
            )));
        }
        Ok(Self {
            tp_rank,
            tp_world,
            rules: BTreeMap::new(),
        })
    }

    /// The plan for a layer's five attention projections and their scale grids: the set a
    /// 2-rank attention-projection smoke loads.
    ///
    /// The rules are the reference's: wq_a and wkv are the fused replicated projection, wq_b and
    /// wo_a are column-parallel, and wo_b is row-parallel. A scale tensor shards along the axis
    /// its weight does.
    ///
    /// # Errors
    /// [ShardError::Plan] when the rank/world pair is impossible.
    pub fn layer_attention(
        layer: usize,
        tp_rank: usize,
        tp_world: usize,
    ) -> Result<Self, ShardError> {
        let mut plan = Self::new(tp_rank, tp_world)?;
        for (projection, rule) in [
            ("wq_a", ShardRule::Replicated),
            ("wkv", ShardRule::Replicated),
            ("wq_b", ShardRule::Column),
            ("wo_a", ShardRule::Column),
            ("wo_b", ShardRule::Row),
        ] {
            let base = format!("layers.{layer}.attn.{projection}");
            plan.insert(format!("{base}.weight"), rule);
            plan.insert(format!("{base}.scale"), rule);
        }
        Ok(plan)
    }

    /// Adds one rule; a repeated name replaces the earlier rule and returns it.
    pub fn insert(&mut self, name: impl Into<String>, rule: ShardRule) -> Option<ShardRule> {
        self.rules.insert(name.into(), rule)
    }

    /// The rule for a tensor, if the plan names it.
    pub fn rule(&self, name: &str) -> Option<ShardRule> {
        self.rules.get(name).copied()
    }

    /// The number of tensors the plan describes.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the plan describes no tensors.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// This plan's rank.
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    /// This plan's rank count.
    pub fn tp_world(&self) -> usize {
        self.tp_world
    }

    /// The plan's tensor names and rules, in name order.
    pub fn rules(&self) -> &BTreeMap<String, ShardRule> {
        &self.rules
    }

    /// The verified shape arithmetic for every tensor in the plan.
    ///
    /// This reads headers only: no tensor data is touched, so a wrong plan is refused before a
    /// byte of weights is read and before any device allocation.
    ///
    /// # Errors
    /// [ShardError::Weights] when a tensor or its shard header is missing,
    /// [ShardError::Plan] when the stored shape does not divide into this rank's slice.
    pub fn verify(&self, source: &SafeTensorsSource) -> Result<Vec<VerifiedTensor>, ShardError> {
        let mut verified = Vec::with_capacity(self.rules.len());
        for (name, rule) in &self.rules {
            let header = source.header(name)?;
            let descriptor = header.tensor(name)?;
            let geometry = self.geometry(name, *rule, &descriptor.shape, descriptor.byte_len())?;
            let mut shard_shape = descriptor.shape.clone();
            match rule {
                ShardRule::Replicated => {}
                ShardRule::Column => shard_shape[0] /= self.tp_world,
                ShardRule::Row => shard_shape[1] /= self.tp_world,
            }
            verified.push(VerifiedTensor {
                name: name.clone(),
                rule: *rule,
                dtype: descriptor.dtype.clone(),
                stored_shape: descriptor.shape.clone(),
                shard_shape,
                shard_bytes: geometry.shard_bytes,
                row_bytes: geometry.row_bytes,
                ranges: geometry.ranges,
            });
        }
        Ok(verified)
    }

    /// Reads this rank's slice of every tensor in the plan, in name order.
    ///
    /// # Errors
    /// [ShardError::Weights] or [ShardError::Plan] from the header checks or the range read.
    pub fn read_all(&self, source: &SafeTensorsSource) -> Result<Vec<StoredShard>, ShardError> {
        self.rules
            .keys()
            .map(|name| self.read(source, name))
            .collect()
    }

    /// Reads this rank's slice of one planned tensor.
    ///
    /// # Errors
    /// [ShardError::Plan] when the plan does not name the tensor or its shape cannot be split,
    /// [ShardError::Weights] when the checkpoint read fails.
    pub fn read(&self, source: &SafeTensorsSource, name: &str) -> Result<StoredShard, ShardError> {
        let rule = self
            .rule(name)
            .ok_or_else(|| ShardError::Plan(format!("the plan does not describe {name}")))?;
        let header = source.header(name)?;
        let descriptor = header.tensor(name)?;
        let geometry = self.geometry(name, rule, &descriptor.shape, descriptor.byte_len())?;
        let ranges = self.byte_ranges(rule, &descriptor.shape, geometry);
        let bytes = source.read_ranges(name, &ranges)?;
        let mut shard_shape = descriptor.shape.clone();
        match rule {
            ShardRule::Replicated => {}
            ShardRule::Column => shard_shape[0] /= self.tp_world,
            ShardRule::Row => shard_shape[1] /= self.tp_world,
        }
        Ok(StoredShard {
            name: name.to_owned(),
            rule,
            dtype: descriptor.dtype.clone(),
            stored_shape: descriptor.shape.clone(),
            shard_shape,
            bytes,
            ranges: geometry.ranges,
        })
    }

    /// The shape and byte arithmetic for one slice, from the stored shape and byte length.
    fn geometry(
        &self,
        name: &str,
        rule: ShardRule,
        stored_shape: &[usize],
        stored_bytes: u64,
    ) -> Result<SliceGeometry, ShardError> {
        let tp = self.tp_world as u64;
        let row_bytes = self.row_bytes(name, stored_shape, stored_bytes)?;
        let (shard_bytes, ranges) = match rule {
            ShardRule::Replicated => (stored_bytes, 1),
            ShardRule::Column => {
                let rows = stored_shape.first().copied().unwrap_or(0);
                if rows == 0 || !(rows as u64).is_multiple_of(tp) {
                    return Err(ShardError::Plan(format!(
                        "{name}: {rows} stored rows do not divide into {tp} ranks"
                    )));
                }
                let per_rank = rows / self.tp_world;
                (row_bytes * per_rank as u64, 1)
            }
            ShardRule::Row => {
                if stored_shape.len() != 2 {
                    return Err(ShardError::Plan(format!(
                        "{name}: a row-parallel tensor must be stored two-dimensional, stored shape {stored_shape:?}"
                    )));
                }
                if !(stored_shape[1] as u64).is_multiple_of(tp) {
                    return Err(ShardError::Plan(format!(
                        "{name}: {} stored columns do not divide into {tp} ranks",
                        stored_shape[1]
                    )));
                }
                if row_bytes % tp != 0 {
                    return Err(ShardError::Plan(format!(
                        "{name}: {row_bytes} bytes per row do not divide into {tp} ranks"
                    )));
                }
                (row_bytes / tp * stored_shape[0] as u64, stored_shape[0])
            }
        };
        Ok(SliceGeometry {
            shard_bytes,
            stored_bytes,
            row_bytes,
            ranges,
        })
    }

    /// The bytes one stored row occupies, from the stored shape and the shard's byte length.
    fn row_bytes(
        &self,
        name: &str,
        stored_shape: &[usize],
        stored_bytes: u64,
    ) -> Result<u64, ShardError> {
        let rows = stored_shape.first().copied().unwrap_or(0) as u64;
        if rows == 0 || !stored_bytes.is_multiple_of(rows) {
            return Err(ShardError::Plan(format!(
                "{name}: {stored_bytes} stored bytes do not divide into {} rows",
                stored_shape.first().copied().unwrap_or(0)
            )));
        }
        Ok(stored_bytes / rows)
    }

    /// The byte ranges one slice read issues, in slice order.
    fn byte_ranges(
        &self,
        rule: ShardRule,
        stored_shape: &[usize],
        geometry: SliceGeometry,
    ) -> Vec<(u64, usize)> {
        let rank = self.tp_rank as u64;
        match rule {
            ShardRule::Replicated => vec![(0, geometry.stored_bytes as usize)],
            ShardRule::Column => {
                let at = rank * geometry.shard_bytes;
                vec![(at, geometry.shard_bytes as usize)]
            }
            ShardRule::Row => {
                let per_rank = geometry.row_bytes / self.tp_world as u64;
                let at = rank * per_rank;
                (0..stored_shape[0])
                    .map(|row| (row as u64 * geometry.row_bytes + at, per_rank as usize))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// Writes a one-shard safetensors container plus its index, and opens the source.
    fn source(dir: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> SafeTensorsSource {
        let shard_name = "model-00001-of-00001.safetensors";
        let mut header = String::from("{");
        let mut offset = 0u64;
        for (index, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let begin = offset;
            offset += bytes.len() as u64;
            let shape_text = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_text}],\"data_offsets\":[{begin},{offset}]}}"
            ));
        }
        header.push('}');
        let mut header = header.into_bytes();
        while (8 + header.len()) % 8 != 0 {
            header.push(b' ');
        }
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        for (_, _, _, bytes) in tensors {
            file.extend_from_slice(bytes);
        }
        std::fs::write(dir.join(shard_name), file).expect("write the synthetic shard");
        let mut weight_map = String::new();
        for (index, (name, _, _, _)) in tensors.iter().enumerate() {
            if index > 0 {
                weight_map.push(',');
            }
            weight_map.push_str(&format!("\"{name}\":\"{shard_name}\""));
        }
        let index_text = format!("{{\"metadata\":{{}},\"weight_map\":{{{weight_map}}}}}");
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, index_text).expect("write the synthetic index");
        SafeTensorsSource::open(dir, &index_path).expect("open the synthetic source")
    }

    fn fixture() -> (tempfile::TempDir, SafeTensorsSource) {
        let dir = tempfile::tempdir().expect("tempdir");
        let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
            ("t.col", "F32", vec![4, 3], (0..48u8).collect::<Vec<u8>>()),
            (
                "t.row",
                "F8_E4M3",
                vec![2, 4],
                vec![10, 11, 12, 13, 20, 21, 22, 23],
            ),
            ("t.rep", "BF16", vec![2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]),
        ];
        let source = source(dir.path(), &tensors);
        (dir, source)
    }

    fn plan(rank: usize) -> ShardPlan {
        let mut plan = ShardPlan::new(rank, 2).expect("plan");
        plan.insert("t.col", ShardRule::Column);
        plan.insert("t.row", ShardRule::Row);
        plan.insert("t.rep", ShardRule::Replicated);
        plan
    }

    #[test]
    fn column_slices_are_row_blocks_and_row_slices_are_contiguous_row_segments() {
        let (_dir, source) = fixture();
        let rank0 = plan(0);
        let verified = rank0.verify(&source).expect("verify rank 0");
        let col = verified
            .iter()
            .find(|tensor| tensor.name == "t.col")
            .expect("col");
        assert_eq!(col.shard_shape, vec![2, 3]);
        assert_eq!(col.shard_bytes, 24);
        assert_eq!(col.ranges, 1);
        let row = verified
            .iter()
            .find(|tensor| tensor.name == "t.row")
            .expect("row");
        assert_eq!(row.shard_shape, vec![2, 2]);
        assert_eq!(row.shard_bytes, 4);
        assert_eq!(row.row_bytes, 4);
        assert_eq!(row.ranges, 2);
        let rep = verified
            .iter()
            .find(|tensor| tensor.name == "t.rep")
            .expect("rep");
        assert_eq!(rep.shard_shape, vec![2, 2]);
        assert_eq!(rep.shard_bytes, 8);

        let slices = rank0.read_all(&source).expect("read rank 0");
        let col_bytes = &slices
            .iter()
            .find(|tensor| tensor.name == "t.col")
            .expect("col")
            .bytes;
        assert_eq!(col_bytes, &(0..24u8).collect::<Vec<u8>>()[..]);
        let row_bytes = &slices
            .iter()
            .find(|tensor| tensor.name == "t.row")
            .expect("row")
            .bytes;
        assert_eq!(row_bytes, &[10, 11, 20, 21]);
        let rep_bytes = &slices
            .iter()
            .find(|tensor| tensor.name == "t.rep")
            .expect("rep")
            .bytes;
        assert_eq!(rep_bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn the_second_rank_takes_the_other_half_of_each_axis() {
        let (_dir, source) = fixture();
        let rank1 = plan(1);
        let slices = rank1.read_all(&source).expect("read rank 1");
        let col = &slices
            .iter()
            .find(|tensor| tensor.name == "t.col")
            .expect("col")
            .bytes;
        assert_eq!(col, &(24..48u8).collect::<Vec<u8>>()[..]);
        let row = &slices
            .iter()
            .find(|tensor| tensor.name == "t.row")
            .expect("row")
            .bytes;
        assert_eq!(row, &[12, 13, 22, 23]);
    }

    #[test]
    fn a_shape_that_does_not_divide_is_refused_before_any_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = vec![
            ("odd.rows", "F8_E4M3", vec![3, 4], vec![0u8; 12]),
            ("odd.cols", "F8_E4M3", vec![2, 3], vec![0u8; 6]),
        ];
        let source = source(
            dir.path(),
            &[
                ("odd.rows", "F8_E4M3", vec![3, 4], vec![0u8; 12]),
                ("odd.cols", "F8_E4M3", vec![2, 3], vec![0u8; 6]),
            ],
        );
        let _ = tensors;
        let mut plan = ShardPlan::new(0, 2).expect("plan");
        plan.insert("odd.rows", ShardRule::Column);
        plan.insert("odd.cols", ShardRule::Row);
        let error = plan
            .verify(&source)
            .expect_err("a non-dividing shape must be refused");
        let text = error.to_string();
        assert!(
            text.contains("3 stored rows") || text.contains("3 stored columns"),
            "unexpected refusal: {text}"
        );
    }

    #[test]
    fn an_impossible_rank_or_world_is_refused() {
        assert!(ShardPlan::new(0, 0).is_err());
        assert!(ShardPlan::new(4, 4).is_err());
        assert!(ShardPlan::new(3, 4).is_ok());
    }

    #[test]
    fn the_layer_attention_plan_has_the_reference_rules() {
        let plan = ShardPlan::layer_attention(0, 1, 4).expect("plan");
        assert_eq!(plan.len(), 10);
        assert_eq!(
            plan.rule("layers.0.attn.wq_a.weight"),
            Some(ShardRule::Replicated)
        );
        assert_eq!(
            plan.rule("layers.0.attn.wkv.scale"),
            Some(ShardRule::Replicated)
        );
        assert_eq!(
            plan.rule("layers.0.attn.wq_b.weight"),
            Some(ShardRule::Column)
        );
        assert_eq!(
            plan.rule("layers.0.attn.wo_a.weight"),
            Some(ShardRule::Column)
        );
        assert_eq!(plan.rule("layers.0.attn.wo_b.weight"), Some(ShardRule::Row));
        assert_eq!(plan.rule("layers.0.attn.wo_b.scale"), Some(ShardRule::Row));
    }

    #[test]
    fn a_tensor_the_plan_does_not_name_is_refused() {
        let (_dir, source) = fixture();
        let rank0 = plan(0);
        match rank0.read(&source, "t.missing") {
            Err(ShardError::Plan(_)) => {}
            other => panic!("expected a plan refusal, got {other:?}"),
        }
    }

    /// Reads the released checkpoint's layer-0 attention shapes when the host has it.
    ///
    /// Run with SPARKFLASH_CHECKPOINT pointing at the snapshot directory:
    /// cargo test -p sparkflash-device --lib released_checkpoint -- --ignored --nocapture
    #[test]
    #[ignore = "reads the released checkpoint; set SPARKFLASH_CHECKPOINT to its snapshot directory"]
    fn released_checkpoint_layer_zero_attention_slices_match_the_reference() {
        let Ok(dir) = std::env::var("SPARKFLASH_CHECKPOINT") else {
            panic!("set SPARKFLASH_CHECKPOINT to the snapshot directory");
        };
        let dir = std::path::PathBuf::from(dir);
        let index = dir.join("model.safetensors.index.json");
        let source = SafeTensorsSource::open(&dir, &index).expect("open the released checkpoint");
        for rank in 0..4 {
            let plan = ShardPlan::layer_attention(0, rank, 4).expect("plan");
            let verified = plan
                .verify(&source)
                .expect("verify against the released headers");
            let expected: &[(&str, Vec<usize>)] = &[
                ("layers.0.attn.wq_a.weight", vec![1280, 5120]),
                ("layers.0.attn.wkv.weight", vec![512, 5120]),
                ("layers.0.attn.wq_b.weight", vec![8192, 1280]),
                ("layers.0.attn.wo_a.weight", vec![2048, 4096]),
                ("layers.0.attn.wo_b.weight", vec![5120, 2048]),
                ("layers.0.attn.wq_b.scale", vec![256, 40]),
                ("layers.0.attn.wo_b.scale", vec![160, 64]),
            ];
            for (name, shape) in expected {
                let tensor = verified
                    .iter()
                    .find(|tensor| tensor.name == *name)
                    .unwrap_or_else(|| panic!("{name} missing from the verified plan"));
                assert_eq!(&tensor.shard_shape, shape, "rank {rank} {name}");
            }
        }
    }
}
