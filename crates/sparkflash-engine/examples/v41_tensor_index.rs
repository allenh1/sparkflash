//! Writes the v41 language-model tensor index: name -> {dtype, shape}.
//!
//! This is a dev tool for the plan-only dry run. The shapes come from the architecture
//! (TensorPlan::language_model), not from safetensors headers; the plan is byte-exact against the
//! released checkpoint total, so this dump is a synthetic-index arithmetic check, not a measurement.
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::tensor::TensorPlan;

fn main() {
    let config = ModelConfig::v41_flash().expect("the shipped configuration must parse");
    let plan = TensorPlan::language_model(&config);
    let mut map = serde_json::Map::new();
    let mut bytes = 0u128;
    for spec in plan.specs() {
        bytes += spec.storage_bytes();
        // FP4 is not a safetensors dtype: the release stores experts as I8 with two FP4 values per
        // byte in the last dimension. The plan's logical shape halves into that on-disk shape; the
        // byte total is unchanged.
        let (dtype, shape) = if spec.dtype.name() == "F4_E2M1" {
            let mut shape = spec.shape.clone();
            let last = shape.len() - 1;
            shape[last] /= 2;
            ("I8", shape)
        } else {
            (spec.dtype.name(), spec.shape.clone())
        };
        map.insert(
            spec.name.clone(),
            serde_json::json!({ "dtype": dtype, "shape": shape }),
        );
    }
    eprintln!(
        "tensor-index: {} tensors, {} bytes language-model; released total 510286023000 - vision/aligner 970536960 = {}",
        plan.specs().len(),
        bytes,
        510_286_023_000u128 - 970_536_960u128
    );
    let value = serde_json::json!(map);
    println!("{}", serde_json::to_string(&value).expect("serialise"));
}
