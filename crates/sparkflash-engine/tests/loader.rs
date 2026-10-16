//! Integration tests for the plan-verified loader, on shards built here.
//!
//! The 48 released shards are not on this machine, so every test builds its own safetensors
//! container with a layout it chose, and compares the loader against numbers written down by hand
//! instead of against the loader's own output: the E4M3 code 0x38 is 1.0 and the E8M0 code 0x80 is
//! 2.0, so a block of 0x38 under a 0x80 scale is 2.0 everywhere and its bf16 bytes are 0x4000; the
//! packed FP4 pattern 0x21 0x43 0x65 0x87 decodes to 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, low
//! nibble first. The tests that are about a refusal assert which error came back and that its
//! message names the tensor, because is_err() alone is not a test.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::json;
use tempfile::TempDir;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::DType;
use sparkflash_engine::loader::{
    LoadedWeights, Treatment, WO_A_GRIDS, dequantise_wo_a, engram_tables, is_engram_table,
    packed_expert, wo_a_scale_name,
};
use sparkflash_engine::quant;
use sparkflash_engine::safetensors::CheckpointIndex;
use sparkflash_engine::tensor::{TensorPlan, TensorSpec};
use sparkflash_engine::weights::WeightError;

/// The one shard every test writes.
const SHARD: &str = "model-00001-of-00001.safetensors";

/// One tensor in a container we build by hand.
struct Entry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// One entry, with the data length the shape and dtype imply checked here rather than by a reader.
fn entry(name: &str, dtype: &str, shape: &[usize], data: Vec<u8>) -> Entry {
    assert_eq!(
        data.len(),
        stored_bytes(dtype, shape),
        "{name} is {dtype} {shape:?}"
    );
    Entry {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        data,
    }
}

/// How many bytes a stored word of that shape occupies.
fn stored_bytes(word: &str, shape: &[usize]) -> usize {
    let elements: usize = shape.iter().product();
    let per = match word {
        "F32" => 4,
        "BF16" | "F16" => 2,
        _ => 1,
    };
    elements * per
}

/// The stored dtype word and shape the plan's rules require of one planned tensor.
fn stored_form(spec: &TensorSpec) -> (String, Vec<usize>) {
    if Treatment::of(spec, &spec.name) == Treatment::PackedExpert {
        let mut shape = spec.shape.clone();
        let last = shape.len() - 1;
        assert_eq!(shape[last] % 2, 0, "{} is packed two per byte", spec.name);
        shape[last] /= 2;
        ("I8".to_string(), shape)
    } else {
        (spec.dtype.name().to_string(), spec.shape.clone())
    }
}

/// A safetensors container: an 8-byte little-endian header length, the JSON header, then the data.
fn container(entries: &[Entry]) -> Vec<u8> {
    let mut offset = 0u64;
    let mut tensors = serde_json::Map::new();
    let mut data = Vec::new();
    for item in entries {
        let begin = offset;
        offset += item.data.len() as u64;
        tensors.insert(
            item.name.clone(),
            json!({
                "dtype": item.dtype,
                "shape": item.shape,
                "data_offsets": [begin, offset],
            }),
        );
        data.extend_from_slice(&item.data);
    }
    let header = serde_json::Value::Object(tensors).to_string();
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

/// A checkpoint directory holding one shard and the index that names it.
struct Checkpoint {
    dir: TempDir,
    index: PathBuf,
}

impl Checkpoint {
    /// The directory the loader is pointed at.
    fn dir(&self) -> &Path {
        self.dir.path()
    }

    /// Opens it with [LoadedWeights::open].
    fn open(&self, plan: &TensorPlan) -> Result<LoadedWeights, WeightError> {
        LoadedWeights::open(self.dir(), &self.index, plan)
    }

    /// The shard file, for a test that wants to damage it.
    fn shard(&self) -> PathBuf {
        self.dir().join(SHARD)
    }
}

/// Builds a complete checkpoint from a plan, then lets a test damage it.
///
/// Every planned tensor is stored the way the plan's rules say, as zeros, so a test that only wants
/// a wrong shape or a missing name changes that one entry and nothing else.
fn checkpoint(plan: &TensorPlan, damage: impl FnOnce(&mut Vec<Entry>)) -> Checkpoint {
    let mut entries: Vec<Entry> = plan
        .specs()
        .iter()
        .map(|spec| {
            let (word, shape) = stored_form(spec);
            let data = vec![0u8; stored_bytes(&word, &shape)];
            entry(&spec.name, &word, &shape, data)
        })
        .collect();
    damage(&mut entries);

    let dir = TempDir::new().expect("a temporary directory");
    std::fs::write(dir.path().join(SHARD), container(&entries)).expect("the shard");
    let weight_map: BTreeMap<&str, &str> = entries
        .iter()
        .map(|item| (item.name.as_str(), SHARD))
        .collect();
    let total: u64 = entries.iter().map(|item| item.data.len() as u64).sum();
    let index = json!({
        "metadata": { "total_size": total },
        "weight_map": weight_map,
    })
    .to_string();
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, index).expect("the index");
    Checkpoint {
        dir,
        index: index_path,
    }
}

/// One entry by name.
fn find<'a>(entries: &'a mut [Entry], name: &str) -> &'a mut Entry {
    entries
        .iter_mut()
        .find(|item| item.name == name)
        .unwrap_or_else(|| panic!("the shard should hold {name}"))
}

/// Replaces one entry's stored form.
fn replace(entries: &mut [Entry], name: &str, dtype: &str, shape: &[usize], data: Vec<u8>) {
    let replacement = entry(name, dtype, shape, data);
    let item = find(entries, name);
    item.dtype = replacement.dtype;
    item.shape = replacement.shape;
    item.data = replacement.data;
}

/// Removes one entry, so the checkpoint no longer has it anywhere.
fn remove(entries: &mut Vec<Entry>, name: &str) {
    let position = entries
        .iter()
        .position(|item| item.name == name)
        .unwrap_or_else(|| panic!("the shard should hold {name}"));
    entries.remove(position);
}

/// The name inside any of the error variants that carries one.
fn error_name(error: &WeightError) -> &str {
    match error {
        WeightError::Missing { name }
        | WeightError::TooLarge { name, .. }
        | WeightError::Refused { name, .. }
        | WeightError::Io { name, .. }
        | WeightError::Malformed { name, .. } => name,
        WeightError::Plan { detail } => panic!("a plan-level error names no tensor: {detail}"),
    }
}

/// The test configuration: tiny, but with a wo_a of [64, 64] so the released 32x32 grid rule has
/// something to divide. heads is n_heads * head_dim = 8 * 16 = 128, so the in dimension is
/// heads / o_groups = 64, and the out dimension is o_groups * o_lora_rank = 64.
fn test_config() -> ModelConfig {
    ModelConfig {
        o_lora_rank: 32,
        n_heads: 8,
        ..ModelConfig::tiny()
    }
}

/// The plan the reading tests use.
fn test_plan() -> TensorPlan {
    TensorPlan::language_model(&test_config())
}

/// A configuration whose wo_a is [33, 33], which the scale grid cannot divide into whole blocks:
/// dense_scale_shape(33, 33) is [2, 2] and 33 is not a multiple of 2.
fn odd_config() -> ModelConfig {
    ModelConfig {
        o_groups: 3,
        o_lora_rank: 11,
        n_heads: 3,
        head_dim: 33,
        ..ModelConfig::tiny()
    }
}

/// The path of the released index, when this host has it.
fn released_index() -> Option<PathBuf> {
    let path = PathBuf::from("/home/user/dsv41-research/model.safetensors.index.json");
    path.is_file().then_some(path)
}

#[test]
fn a_complete_checkpoint_opens_and_reads_every_planned_tensor() {
    let plan = test_plan();
    let checkpoint = checkpoint(&plan, |_| {});
    let loader = checkpoint.open(&plan).expect("a complete checkpoint opens");

    let report = loader.report();
    assert!(report.missing.is_empty(), "{}", report.summary());
    assert!(report.mismatched.is_empty(), "{}", report.summary());
    assert!(report.unknown.is_empty(), "{}", report.summary());
    assert_eq!(report.matched, plan.len());
    assert!(report.is_conformant());
    assert!(report.is_exact());
    assert_eq!(
        report.summary(),
        format!("{} matched, 0 missing, 0 mismatched, 0 unknown", plan.len())
    );

    let mut read = 0;
    let mut refused = 0;
    for spec in plan.specs() {
        if is_engram_table(&spec.name) {
            let error = loader
                .get(&spec.name)
                .expect_err("an Engram table is never materialised here");
            assert_eq!(error_name(&error), spec.name);
            assert!(matches!(&error, WeightError::Refused { .. }), "{error}");
            refused += 1;
            continue;
        }
        let tensor = loader
            .get(&spec.name)
            .unwrap_or_else(|error| panic!("{}: {error}", spec.name));
        assert_eq!(tensor.name, spec.name);
        assert_eq!(tensor.shape, spec.shape, "{}", spec.name);
        assert_eq!(tensor.elements(), spec.elements(), "{}", spec.name);
        match Treatment::of(spec, &spec.name) {
            Treatment::WoA => {
                assert_eq!(tensor.dtype, DType::BF16, "{}", spec.name);
                assert_eq!(tensor.data.len() as u128, spec.elements() * 2);
            }
            Treatment::PackedExpert => {
                assert_eq!(tensor.dtype, DType::F4E2M1, "{}", spec.name);
                assert_eq!(tensor.data.len() as u128, spec.elements() / 2);
            }
            Treatment::Stored => {
                assert_eq!(tensor.dtype, spec.dtype, "{}", spec.name);
                assert_eq!(tensor.data.len() as u128, spec.storage_bytes());
            }
        }
        read += 1;
    }
    assert_eq!(read + refused, plan.len());
    assert_eq!(refused, 2, "the tiny configuration has one Engram layer");
    assert_eq!(loader.names().count(), plan.len());
    assert_eq!(loader.limit(), 2 * 1024 * 1024 * 1024);
}

#[test]
fn wo_a_dequantises_blockwise_with_hand_computed_numbers() {
    let plan = test_plan();
    let name = "layers.0.attn.wo_a.weight";
    let scale_name = wo_a_scale_name(name);
    assert_eq!(scale_name, "layers.0.attn.wo_a.scale");
    let weight = plan.get(name).expect("the plan holds the layer-0 wo_a");
    let scale = plan.get(&scale_name).expect("and its scale");
    assert_eq!(weight.dtype, DType::F8E4M3);
    assert_eq!(weight.shape, vec![64, 64]);
    assert_eq!(scale.dtype, DType::F8E8M0);
    assert_eq!(scale.shape, vec![2, 2]);

    // Anchor the hand computation on the crate's own tables first: 0x38 is E4M3 for 1.0, 0x80 is
    // E8M0 for 2.0, 0x81 for 4.0, 0x82 for 8.0 and 0x83 for 16.0.
    assert_eq!(quant::f8e4m3_to_f32(0x38), 1.0);
    assert_eq!(quant::f8e8m0_to_f32(0x80), 2.0);
    assert_eq!(quant::f8e8m0_to_f32(0x81), 4.0);
    assert_eq!(quant::f8e8m0_to_f32(0x82), 8.0);
    assert_eq!(quant::f8e8m0_to_f32(0x83), 16.0);

    let checkpoint = checkpoint(&plan, |entries| {
        find(entries, name).data = vec![0x38; 64 * 64];
        // One scale per 32x32 block, in row-major block order: (0,0) 2.0, (0,1) 4.0, (1,0) 8.0,
        // (1,1) 16.0. A blockwise product puts those four values in the four quadrants; a single
        // scale, a transposed grid or a row-major mix-up all produce something else.
        find(entries, &scale_name).data = vec![0x80, 0x81, 0x82, 0x83];
    });
    let loader = checkpoint.open(&plan).expect("a (32, 32) grid opens");

    let tensor = loader.get(name).expect("the wo_a reads");
    assert_eq!(tensor.dtype, DType::BF16);
    assert_eq!(tensor.shape, vec![64, 64]);
    assert_eq!(tensor.data.len(), 64 * 64 * 2);

    // Hand-written bf16: 2.0 is 0x4000, 4.0 is 0x4080, 8.0 is 0x4100, 16.0 is 0x4180.
    let quadrants = [0x4000u16, 0x4080, 0x4100, 0x4180];
    let value = |row: usize, column: usize| -> u16 {
        let offset = (row * 64 + column) * 2;
        u16::from_le_bytes([tensor.data[offset], tensor.data[offset + 1]])
    };
    for row in 0..64 {
        for column in 0..64 {
            let quadrant = (row / 32) * 2 + (column / 32);
            assert_eq!(
                value(row, column),
                quadrants[quadrant],
                "at ({row}, {column})"
            );
        }
    }
    assert_eq!(value(0, 0), 0x4000);
    assert_eq!(value(0, 31), 0x4000);
    assert_eq!(value(0, 32), 0x4080);
    assert_eq!(value(31, 63), 0x4080);
    assert_eq!(value(32, 0), 0x4100);
    assert_eq!(value(63, 63), 0x4180);

    // The scale is still readable: the conversion consumes it, it does not delete it.
    let scale_read_back = loader.get(&scale_name).expect("the scale reads back");
    assert_eq!(scale_read_back.dtype, DType::F8E8M0);
    assert_eq!(scale_read_back.data, vec![0x80, 0x81, 0x82, 0x83]);
}

#[test]
fn an_expert_is_handed_back_packed_with_the_logical_shape() {
    let plan = test_plan();
    let name = "layers.0.ffn.experts.0.w1.weight";
    let spec = plan
        .get(name)
        .expect("the plan holds the first routed expert")
        .clone();
    assert_eq!(spec.dtype, DType::F4E2M1);
    assert_eq!(spec.shape, vec![32, 64]);
    assert_eq!(spec.storage_bytes(), 1024);

    let pattern = [0x21u8, 0x43, 0x65, 0x87];
    let stored: Vec<u8> = (0..1024).map(|index| pattern[index % 4]).collect();
    let checkpoint = checkpoint(&plan, |entries| {
        replace(entries, name, "I8", &[32, 32], stored.clone());
    });
    let loader = checkpoint.open(&plan).expect("a packed expert opens");

    let tensor = loader.get(name).expect("the expert reads");
    assert_eq!(tensor.dtype, DType::F4E2M1);
    assert_eq!(tensor.shape, vec![32, 64]);
    assert_eq!(tensor.data, stored, "no byte of a packed expert may move");
    assert_eq!(tensor.elements(), 32 * 64);

    // Hand-written E2M1: 0x21 is (0.5, 1.0), 0x43 is (1.5, 2.0), 0x65 is (3.0, 4.0), 0x87 is
    // (6.0, -0.0), and the low nibble comes first - torch.float4_e2m1fn_x2's order, which
    // quant.rs pins for all 256 bytes.
    let values = quant::decode_f4e2m1(&tensor.data, tensor.elements() as usize)
        .expect("the packed bytes decode");
    assert_eq!(values.len(), 2048);
    let expected = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0];
    assert_eq!(&values[..8], &expected);
    assert!(values[7].is_sign_negative(), "nibble 8 is negative zero");
    assert_eq!(&values[8..16], &expected);
}

#[test]
fn the_engram_tables_are_refused_by_name_rather_than_read() {
    let plan = test_plan();
    let weight = "layers.1.engram.embed.weight";
    let scale = "layers.1.engram.embed.scale";
    assert_eq!(
        engram_tables(&plan),
        vec![weight.to_string(), scale.to_string()]
    );
    assert!(is_engram_table(weight));
    assert!(is_engram_table(scale));
    assert!(!is_engram_table("embed.weight"));

    let checkpoint = checkpoint(&plan, |_| {});
    let loader = checkpoint
        .open(&plan)
        .expect("the tables are part of the plan, so their headers are checked like any other");
    // refused_names is sorted, because the source keeps its refusal list in a set, while
    // engram_tables follows the plan; both hold the same two names.
    let mut refused = loader.refused_names();
    let mut expected = vec![weight.to_string(), scale.to_string()];
    refused.sort();
    expected.sort();
    assert_eq!(refused, expected);

    for name in [weight, scale] {
        let error = loader
            .get(name)
            .expect_err("an Engram table is never materialised here");
        assert_eq!(error_name(&error), name);
        assert!(matches!(&error, WeightError::Refused { .. }), "{error}");
        assert!(error.to_string().contains(name), "{error}");
    }

    // The refusal is by name: the tensor next to them reads normally.
    let neighbour = loader.get("embed.weight").expect("a normal tensor reads");
    assert_eq!(neighbour.dtype, DType::BF16);
    // And the audit saw the tables as present and conformant, because they are.
    assert!(
        loader.report().missing.is_empty(),
        "{}",
        loader.report().summary()
    );
    assert!(
        loader.report().mismatched.is_empty(),
        "{}",
        loader.report().summary()
    );
}

#[test]
fn a_name_the_plan_does_not_know_is_reported_and_not_refused() {
    let plan = test_plan();
    let extra = "vision.tower.patch_embed.weight";
    let checkpoint = checkpoint(&plan, |entries| {
        entries.push(entry(extra, "BF16", &[4, 4], vec![0; 32]));
    });
    let loader = checkpoint
        .open(&plan)
        .expect("the released checkpoint carries names the plan leaves out");
    assert_eq!(loader.report().unknown, vec![extra.to_string()]);
    assert!(loader.report().is_conformant());
    assert!(!loader.report().is_exact());
    assert_eq!(loader.report().matched, plan.len());
    let error = loader
        .get(extra)
        .expect_err("the loader serves the plan, not the directory");
    assert_eq!(
        error,
        WeightError::Missing {
            name: extra.to_string()
        }
    );
}

#[test]
fn the_limit_is_what_stops_a_model_sized_read() {
    let plan = test_plan();
    let name = "embed.weight";
    let bytes = plan.get(name).unwrap().storage_bytes();
    assert!(bytes > 1024);
    let checkpoint = checkpoint(&plan, |_| {});
    let loader = LoadedWeights::open_with_limit(checkpoint.dir(), &checkpoint.index, &plan, 1024)
        .expect("the audit reads headers, so a 1 KiB limit still opens");
    assert_eq!(loader.limit(), 1024);

    let error = loader
        .get(name)
        .expect_err("the embedding is over a 1 KiB limit");
    assert_eq!(error_name(&error), name);
    assert!(
        matches!(&error, WeightError::TooLarge { bytes: over, limit: 1024, .. } if *over == bytes),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("over this source's limit of 1024"),
        "{error}"
    );

    // A tensor under the limit is unaffected.
    let small = plan
        .specs()
        .iter()
        .find(|spec| spec.storage_bytes() <= 1024 && !is_engram_table(&spec.name))
        .expect("the plan holds tensors smaller than a kilobyte");
    let tensor = loader
        .get(&small.name)
        .unwrap_or_else(|error| panic!("{}: {error}", small.name));
    assert_eq!(tensor.data.len() as u128, small.storage_bytes());
    assert!(tensor.data.len() as u128 <= 1024);
}

#[test]
fn a_shard_that_cannot_be_read_is_reported_as_it_happened() {
    let plan = test_plan();
    let checkpoint = checkpoint(&plan, |_| {});
    std::fs::remove_file(checkpoint.shard()).expect("the shard is removed");
    let error = checkpoint
        .open(&plan)
        .expect_err("a half-uploaded checkpoint is not served");
    assert!(matches!(&error, WeightError::Io { .. }), "{error}");
    let first = plan
        .names()
        .next()
        .expect("the plan is not empty")
        .to_string();
    assert_eq!(error_name(&error), first);
    assert!(
        error.to_string().contains("No such file or directory"),
        "{error}"
    );
}

#[test]
fn a_missing_tensor_is_refused_before_anything_is_read() {
    let plan = test_plan();
    let missing = "embed.weight";
    assert!(
        plan.contains(missing),
        "the test configuration plans {missing}"
    );
    let checkpoint = checkpoint(&plan, |entries| remove(entries, missing));

    let error = checkpoint
        .open(&plan)
        .expect_err("a checkpoint without a planned tensor is not served");
    assert_eq!(
        error,
        WeightError::Missing {
            name: missing.to_string()
        }
    );
    assert_eq!(error.to_string(), format!("no tensor named {missing}"));

    let report = LoadedWeights::audit(checkpoint.dir(), &checkpoint.index, &plan)
        .expect("the audit reports rather than refuses");
    assert_eq!(report.missing, vec![missing.to_string()]);
    assert_eq!(report.matched, plan.len() - 1);
    assert!(!report.is_conformant());
    assert!(!report.is_exact());
    assert_eq!(
        report.summary(),
        format!(
            "{} matched, 1 missing, 0 mismatched, 0 unknown",
            plan.len() - 1
        )
    );
}

#[test]
fn a_wrong_shape_is_refused_with_the_expected_and_the_found_shape() {
    let plan = test_plan();
    let name = "embed.weight";
    let spec = plan
        .get(name)
        .expect("the plan holds the embedding")
        .clone();
    assert_eq!(spec.dtype, DType::BF16);
    assert_eq!(spec.shape, vec![128, 64]);

    let checkpoint = checkpoint(&plan, |entries| {
        replace(entries, name, "BF16", &[128, 32], vec![0; 128 * 32 * 2]);
    });
    let error = checkpoint
        .open(&plan)
        .expect_err("a tensor of another shape is another checkpoint");
    assert_eq!(error_name(&error), name);
    assert!(matches!(&error, WeightError::Malformed { .. }), "{error}");
    let message = error.to_string();
    assert!(message.contains(name), "{message}");
    assert!(message.contains("[128, 64]"), "{message}");
    assert!(message.contains("[128, 32]"), "{message}");
    assert!(message.contains("BF16"), "{message}");
}

#[test]
fn a_wrong_dtype_is_refused_and_the_word_is_named() {
    let plan = test_plan();
    let name = "embed.weight";
    let checkpoint = checkpoint(&plan, |entries| {
        replace(entries, name, "F32", &[128, 64], vec![0; 128 * 64 * 4]);
    });
    let error = checkpoint
        .open(&plan)
        .expect_err("a tensor stored as another format is another checkpoint");
    assert_eq!(error_name(&error), name);
    assert!(matches!(&error, WeightError::Malformed { .. }), "{error}");
    let message = error.to_string();
    assert!(message.contains("the stored dtype word F32"), "{message}");
    assert!(message.contains("BF16"), "{message}");
}

#[test]
fn a_wo_a_whose_grid_is_neither_32_nor_128_is_refused_at_open() {
    // The plain tiny configuration has a wo_a of [16, 16] and so a (16, 16) grid: the released
    // checkpoint is not quantised that way, and this loader refuses it rather than scaling it.
    let plan = TensorPlan::language_model(&ModelConfig::tiny());
    let name = "layers.0.attn.wo_a.weight";
    let scale = wo_a_scale_name(name);
    assert_eq!(plan.get(name).unwrap().shape, vec![16, 16]);
    assert_eq!(plan.get(&scale).unwrap().shape, vec![1, 1]);

    let checkpoint = checkpoint(&plan, |_| {});
    let error = checkpoint
        .open(&plan)
        .expect_err("a (16, 16) grid is not a released grid");
    assert_eq!(error_name(&error), name);
    assert!(matches!(&error, WeightError::Malformed { .. }), "{error}");
    let message = error.to_string();
    assert!(message.contains(name), "{message}");
    assert!(message.contains("(16, 16)"), "{message}");
    assert!(
        message.contains("neither (32, 32) nor (128, 128)"),
        "{message}"
    );

    let report = LoadedWeights::audit(checkpoint.dir(), &checkpoint.index, &plan)
        .expect("the audit reports rather than refuses");
    assert!(!report.is_conformant());
    // Every wo_a weight in the tiny plan has a (16, 16) grid: the two layers and the MTP stage.
    let config = ModelConfig::tiny();
    assert_eq!(
        report.mismatched.len(),
        config.n_layers + config.n_mtp_layers
    );
    for mismatch in &report.mismatched {
        assert!(mismatch.name.ends_with("wo_a.weight"), "{}", mismatch.name);
        assert!(mismatch.found.contains("(16, 16)"), "{}", mismatch.found);
    }
    assert_eq!(report.mismatched[0].name, name);
    assert!(report.missing.is_empty());
}

#[test]
fn a_wo_a_scale_that_does_not_divide_the_weight_is_refused() {
    // 33 x 33 with the plan's own scale grid of [2, 2]: 33 is not a multiple of 2, so a block size
    // does not exist. This is the case that would silently truncate to a 16-wide block elsewhere.
    let plan = TensorPlan::language_model(&odd_config());
    let name = "layers.0.attn.wo_a.weight";
    let scale = wo_a_scale_name(name);
    assert_eq!(plan.get(name).unwrap().shape, vec![33, 33]);
    assert_eq!(plan.get(&scale).unwrap().shape, vec![2, 2]);

    let checkpoint = checkpoint(&plan, |_| {});
    let error = checkpoint
        .open(&plan)
        .expect_err("a scale grid that does not divide the weight is not a grid");
    assert_eq!(error_name(&error), name);
    let message = error.to_string();
    assert!(message.contains("does not divide"), "{message}");
    assert!(message.contains("[33, 33]"), "{message}");
    assert!(message.contains("[2, 2]"), "{message}");
}

#[test]
fn an_expert_that_is_not_packed_is_refused() {
    let plan = test_plan();
    let name = "layers.0.ffn.experts.0.w1.weight";
    // Already at the logical width: the shape an unpacked FP4 or a duplicated file would have.
    let checkpoint = checkpoint(&plan, |entries| {
        replace(entries, name, "I8", &[32, 64], vec![0; 32 * 64]);
    });
    let error = checkpoint
        .open(&plan)
        .expect_err("an expert at the logical width is not the packed view");
    assert_eq!(error_name(&error), name);
    assert!(matches!(&error, WeightError::Malformed { .. }), "{error}");
    let message = error.to_string();
    assert!(message.contains(name), "{message}");
    assert!(message.contains("packed FP4 view"), "{message}");
    assert!(message.contains("halved"), "{message}");

    // The conversion says the relation out loud, for a caller that already has the bytes.
    let error = packed_expert(name, "I8", &[32, 64], &[32, 64], vec![0; 32 * 64])
        .expect_err("the stored last dimension is not half the logical one");
    assert_eq!(error_name(&error), name);
    let message = error.to_string();
    assert!(message.contains("not packed"), "{message}");
    assert!(
        message.contains("the stored last dimension 64"),
        "{message}"
    );
    assert!(message.contains("the logical 64"), "{message}");

    // And the shape rule outside the last dimension is checked too.
    let error = packed_expert(name, "I8", &[16, 32], &[32, 64], vec![0; 16 * 32])
        .expect_err("the stored shape does not match the logical shape");
    assert!(
        error.to_string().contains("outside the last dimension"),
        "{error}"
    );
}

#[test]
fn an_expert_stored_as_fp8_is_refused() {
    let plan = test_plan();
    let name = "layers.0.ffn.experts.0.w1.weight";
    let checkpoint = checkpoint(&plan, |entries| {
        replace(entries, name, "F8_E4M3", &[32, 64], vec![0; 32 * 64]);
    });
    let error = checkpoint
        .open(&plan)
        .expect_err("a non-packed expert is not the packed view");
    assert_eq!(error_name(&error), name);
    let message = error.to_string();
    assert!(
        message.contains("the stored dtype word F8_E4M3"),
        "{message}"
    );
    assert!(message.contains("I8"), "{message}");

    let error = packed_expert(name, "F8_E4M3", &[32, 64], &[32, 64], vec![0; 32 * 64])
        .expect_err("the packed view needs the int8 word");
    assert_eq!(error_name(&error), name);
    assert!(
        error.to_string().contains("needs the int8 word I8"),
        "{error}"
    );
}

#[test]
fn the_conversions_accept_both_released_grids_and_refuse_the_rest() {
    assert_eq!(WO_A_GRIDS, [(32, 32), (128, 128)]);

    // The (32, 32) grid: 64 x 64 with four blocks, all codes 1.0 and one scale each, so the four
    // quadrants hold four different values. A reader that mixed up the row block, the column block or
    // the order of the two would put the wrong one of these four in at least one quadrant; four equal
    // scales could not tell. The code 0x38 is 1.0 and the E8M0 codes 0x80..0x83 are 2, 4, 8 and 16.
    let codes = vec![0x38u8; 64 * 64];
    let small = dequantise_wo_a(
        "x.wo_a.weight",
        &codes,
        &[64, 64],
        &[0x80, 0x81, 0x82, 0x83],
        &[2, 2],
    )
    .expect("a (32, 32) grid is a released grid");
    assert_eq!(small.dtype, DType::BF16);
    assert_eq!(small.shape, vec![64, 64]);
    assert_eq!(small.data.len(), 64 * 64 * 2);
    // The bf16 bits at a position, so each quadrant is checked where it is rather than in a sum.
    let narrow = |row: usize, column: usize| {
        let offset = (row * 64 + column) * 2;
        u16::from_le_bytes([small.data[offset], small.data[offset + 1]])
    };
    assert_eq!(narrow(0, 0), 0x4000, "block (0, 0) holds 2.0");
    assert_eq!(narrow(0, 63), 0x4080, "block (0, 1) holds 4.0");
    assert_eq!(narrow(63, 0), 0x4100, "block (1, 0) holds 8.0");
    assert_eq!(narrow(63, 63), 0x4180, "block (1, 1) holds 16.0");

    // The (128, 128) grid: 256 x 256, one scale per 128-wide block, again four different values.
    let wide = vec![0x38u8; 256 * 256];
    let large = dequantise_wo_a(
        "x.wo_a.weight",
        &wide,
        &[256, 256],
        &[0x80, 0x81, 0x82, 0x83],
        &[2, 2],
    )
    .expect("a (128, 128) grid is the other released grid");
    assert_eq!(large.dtype, DType::BF16);
    assert_eq!(large.shape, vec![256, 256]);
    assert_eq!(large.data.len(), 256 * 256 * 2);
    let broad = |row: usize, column: usize| {
        let offset = (row * 256 + column) * 2;
        u16::from_le_bytes([large.data[offset], large.data[offset + 1]])
    };
    assert_eq!(broad(0, 0), 0x4000, "block (0, 0) holds 2.0");
    assert_eq!(broad(0, 255), 0x4080, "block (0, 1) holds 4.0");
    assert_eq!(broad(255, 0), 0x4100, "block (1, 0) holds 8.0");
    assert_eq!(broad(255, 255), 0x4180, "block (1, 1) holds 16.0");
    // And nothing in between is scaled by a neighbouring block: the seam at 128 is exact.
    assert_eq!(broad(0, 127), 0x4000);
    assert_eq!(broad(0, 128), 0x4080);
    assert_eq!(broad(127, 0), 0x4000);
    assert_eq!(broad(128, 0), 0x4100);

    // The released weight is not square either. wo_a is [8192, 4096] - o_groups * o_lora_rank by
    // heads / o_groups, which is 8 * 1024 by 64 * 512 / 8 - so its 32-block scale grid is [256, 128]:
    // two row blocks per column block, the 2:1 ratio this fixture keeps at a thirty-second of the
    // size. Eight row blocks by four column blocks over [256, 128] means the stride between row
    // blocks is in_blocks = 4 rather than out_blocks = 8, and a reader that used the wrong one would
    // take every row block after the first from the wrong place. A square fixture cannot show that,
    // because the two numbers are equal there. Each row block gets its own scale here: rows 0..32 are
    // 2.0, 32..64 are 4.0, and so on up to 256.0 at row 224.
    let tall_codes = vec![0x38u8; 256 * 128];
    let tall_scales: Vec<u8> = (0..32).map(|block| 0x80 + (block / 4) as u8).collect();
    let tall = dequantise_wo_a(
        "x.wo_a.weight",
        &tall_codes,
        &[256, 128],
        &tall_scales,
        &[8, 4],
    )
    .expect("a non-square (32, 32) grid is the released shape");
    assert_eq!(tall.shape, vec![256, 128]);
    let tall_at = |row: usize, column: usize| {
        let offset = (row * 128 + column) * 2;
        u16::from_le_bytes([tall.data[offset], tall.data[offset + 1]])
    };
    assert_eq!(tall_at(0, 0), 0x4000);
    assert_eq!(tall_at(31, 127), 0x4000);
    assert_eq!(tall_at(32, 0), 0x4080, "the second row block holds 4.0");
    assert_eq!(
        tall_at(32, 127),
        0x4080,
        "the column block does not change the scale"
    );
    assert_eq!(tall_at(224, 0), 0x4380, "the eighth row block holds 256.0");
    assert_eq!(tall_at(255, 127), 0x4380);

    // A grid that is neither.
    let error = dequantise_wo_a(
        "x.wo_a.weight",
        &vec![0u8; 256 * 256],
        &[256, 256],
        &[0u8; 16],
        &[4, 4],
    )
    .expect_err("a (64, 64) grid is neither released pair");
    assert_eq!(error_name(&error), "x.wo_a.weight");
    assert!(error.to_string().contains("(64, 64)"), "{error}");

    // A scale that does not divide the weight.
    let error = dequantise_wo_a(
        "x.wo_a.weight",
        &vec![0u8; 64 * 64],
        &[64, 64],
        &[0u8; 9],
        &[3, 3],
    )
    .expect_err("64 is not a multiple of 3");
    assert_eq!(error_name(&error), "x.wo_a.weight");
    assert!(error.to_string().contains("does not divide"), "{error}");

    // Bytes that are not the length the shapes imply.
    let error = dequantise_wo_a("x.wo_a.weight", &[0u8; 10], &[32, 32], &[0u8; 1], &[1, 1])
        .expect_err("10 bytes is not 1024 codes");
    assert_eq!(error_name(&error), "x.wo_a.weight");
    assert!(error.to_string().contains("1024"), "{error}");

    // A non-two-dimensional shape is refused rather than indexed blindly.
    let error = dequantise_wo_a("x.wo_a.weight", &codes, &[64 * 64], &[0x80], &[1, 1])
        .expect_err("a wo_a weight is two-dimensional");
    assert!(error.to_string().contains("not two-dimensional"), "{error}");
}

#[test]
fn the_released_plan_refuses_exactly_the_two_engram_tables() {
    let config = ModelConfig::v41_flash_cached();
    let plan = TensorPlan::language_model(config);
    assert_eq!(
        engram_tables(&plan),
        vec![
            "layers.1.engram.embed.weight".to_string(),
            "layers.1.engram.embed.scale".to_string(),
            "layers.14.engram.embed.weight".to_string(),
            "layers.14.engram.embed.scale".to_string(),
        ]
    );

    // 384,006,168 x 256 FP8 and 384,016,682 x 256 FP8, each with eight scale bytes per row.
    assert_eq!(
        plan.get("layers.1.engram.embed.weight")
            .unwrap()
            .storage_bytes(),
        98_305_579_008
    );
    assert_eq!(
        plan.get("layers.14.engram.embed.weight")
            .unwrap()
            .storage_bytes(),
        98_308_270_592
    );
    let tables: u128 = engram_tables(&plan)
        .iter()
        .map(|name| plan.get(name).unwrap().storage_bytes())
        .sum();
    assert_eq!(tables, 202_758_032_400);

    // Every released wo_a is on the 32x32 grid: its scale is a thirty-second of each dimension, in
    // both directions, so out / scale_out and in / scale_in are both exactly 32.
    let mut checked = 0;
    for spec in plan.specs() {
        if !spec.name.ends_with("wo_a.weight") {
            continue;
        }
        let scale = plan
            .get(&wo_a_scale_name(&spec.name))
            .expect("every wo_a has a scale");
        let (out_dim, in_dim) = (spec.shape[0], spec.shape[1]);
        assert_eq!(
            scale.shape,
            vec![out_dim / 32, in_dim / 32],
            "{}",
            spec.name
        );
        assert_eq!(out_dim % 32, 0, "{}", spec.name);
        assert_eq!(in_dim % 32, 0, "{}", spec.name);
        checked += 1;
    }
    assert_eq!(checked, config.n_layers + config.n_mtp_layers);
    assert_eq!(
        plan.get("layers.0.attn.wo_a.weight").unwrap().shape,
        vec![8192, 4096]
    );
    assert_eq!(
        plan.get("layers.0.attn.wo_a.scale").unwrap().shape,
        vec![256, 128]
    );
}

#[test]
fn the_released_index_names_the_tables_the_loader_refuses() {
    let Some(path) = released_index() else {
        println!("SKIP: the released checkpoint index is not on this host");
        return;
    };
    let index = CheckpointIndex::load(&path).expect("the released index parses");
    for name in [
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.14.engram.embed.weight",
        "layers.14.engram.embed.scale",
    ] {
        assert!(index.weight_map.contains_key(name), "{name}");
    }
    assert!(is_engram_table("layers.14.engram.embed.weight"));
    assert_eq!(
        index
            .weight_map
            .keys()
            .filter(|name| is_engram_table(name))
            .count(),
        4,
        "the released checkpoint has exactly the two tables and their scales"
    );
}

#[test]
fn an_empty_plan_is_refused_before_the_checkpoint_is_consulted() {
    // The audit is a universal statement, and a universal statement over nothing is true: with an
    // empty plan every directory is conformant, so the check that exists to protect the arithmetic
    // passes while checking nothing. TensorPlan::default() is public, so the state is reachable, and
    // it is refused rather than tolerated.
    let empty = TensorPlan::default();
    // A directory and an index that do not exist prove the order of the checks: a plan-level refusal
    // cannot have come from reading anything.
    let missing_dir = Path::new("/nonexistent-sparkflash-checkpoint");
    let missing_index = Path::new("/nonexistent-sparkflash-index.json");

    match LoadedWeights::open(missing_dir, missing_index, &empty) {
        Err(WeightError::Plan { detail }) => assert!(detail.contains("no tensors"), "{detail}"),
        other => panic!("expected the empty plan to be refused first, got {other:?}"),
    }
    match LoadedWeights::audit(missing_dir, missing_index, &empty) {
        Err(WeightError::Plan { detail }) => assert!(detail.contains("no tensors"), "{detail}"),
        other => panic!("expected the audit to refuse the empty plan too, got {other:?}"),
    }

    // A checkpoint built for a real plan still opens, so the guard is about the plan being empty and
    // not about opening being restricted.
    let plan = test_plan();
    let checkpoint = checkpoint(&plan, |_| {});
    assert!(LoadedWeights::open(checkpoint.dir(), &checkpoint.index, &plan).is_ok());
}
