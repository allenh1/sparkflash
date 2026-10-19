//! The one writer for the per-position logits dump (O6 / task-19).
//!
//! D40 (docs/DECISIONS.md) freezes the device side of the serving-parity measurement: one JSONL
//! object per compared position plus `FILE.bin`, consecutive little-endian f32 blocks holding the
//! full vocabulary logits in the same position order. This module is the single implementation of
//! that artefact: the device chain calls [write_logits_dump] from `hybrid_sequence`, the field set
//! is documented in docs/build/logits-dump.md, and the frozen consumer is
//! `scripts/compare_native_vs_vllm.py` (its `read_dump` / `read_logits` functions).
//!
//! The dump changes no arithmetic and writes no stdout line: it reads only values the stage already
//! computed, and a parity-failing run still leaves its logits. It was carried on
//! `task/logits-dump` `4f599b7` before O6; the writer body here is that commit's code verbatim,
//! so the merged chain emits the same bytes.

use std::path::{Path, PathBuf};

use sparkflash_engine::forward::HybridTokenOutput;

/// One compared position: the device position, the reference token, the hybrid token, and whether
/// the two token ids were equal. The dump writes the values as-is; `None` becomes JSON `null`.
pub type ComparedRow = (usize, Option<u32>, Option<u32>, bool);

/// The keys every dump object carries, in the order the writer emits them. This is the frozen
/// schema: `scripts/compare_native_vs_vllm.py` requires [CONSUMED_FIELDS] and ignores the rest, so
/// an older dump that predates the parity-only fields still joins.
pub const FIELDS: [&str; 11] = [
    "position",
    "device_argmax",
    "logits_argmax",
    "reference_id",
    "hybrid_id",
    "fixture_id",
    "exact_token_equal",
    "vocab",
    "logits_offset",
    "logits_bin",
    "top20",
];

/// The keys the frozen consumer requires (`scripts/compare_native_vs_vllm.py::read_dump`).
pub const CONSUMED_FIELDS: [&str; 3] = ["position", "vocab", "logits_offset"];

/// The sidecar path for a dump path: D40 freezes it as the dump path plus `.bin`.
#[must_use]
pub fn sidecar_path(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".bin");
    PathBuf::from(sidecar)
}

fn top_logits(logits: &[f32], count: usize) -> Vec<(u32, f32)> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices
        .into_iter()
        .take(count)
        .map(|index| (index as u32, logits[index]))
        .collect()
}

/// Writes the vLLM numeric-comparison dump for the compared sequence positions.
///
/// `<path>` receives one JSON object per line: position, the device head's argmax token id, the
/// argmax of the dumped logits, the reference/fixture token ids where they exist, the vocabulary
/// size, the byte offset of this position's block in the sidecar, and the top-20 (token_id, logit)
/// pairs. `<path>.bin` receives the full vocabulary logits as little-endian f32 blocks in the same
/// position order, so the comparison can rebuild any screened/masked set.
///
/// The rows are the rows the token table already compared: position 129 then the decode
/// positions for a full claim, else positions 0.. for a short run. The dump changes no arithmetic
/// and writes no stdout line; the returned count is the number of positions written.
///
/// # Errors
/// A path that cannot be created/written, or a JSON serialisation failure.
pub fn write_logits_dump(
    path: &Path,
    rows: &[ComparedRow],
    outputs: &[HybridTokenOutput],
    first_output: usize,
    full_claim: bool,
    fixture_steps: &[Option<u32>],
    prompt_len: usize,
) -> Result<usize, String> {
    use std::io::Write;

    let bin_path = sidecar_path(path);
    let text_file = std::fs::File::create(path)
        .map_err(|error| format!("logits dump {}: {error}", path.display()))?;
    let bin_file = std::fs::File::create(&bin_path)
        .map_err(|error| format!("logits dump {}: {error}", bin_path.display()))?;
    let mut text = std::io::BufWriter::new(text_file);
    let mut bin = std::io::BufWriter::new(bin_file);
    let mut written = 0usize;
    let mut offset = 0usize;
    for (row, (position, reference, hybrid, exact)) in rows.iter().enumerate() {
        let Some(output) = outputs.get(first_output + row) else {
            continue;
        };
        let logits = &output.logits;
        if logits.is_empty() {
            continue;
        }
        let mut argmax = 0usize;
        for (index, value) in logits.iter().enumerate() {
            if value.total_cmp(&logits[argmax]).is_gt() {
                argmax = index;
            }
        }
        let mut block = Vec::with_capacity(logits.len() * 4);
        for value in logits {
            block.extend_from_slice(&value.to_le_bytes());
        }
        bin.write_all(&block)
            .map_err(|error| format!("logits dump {}: {error}", bin_path.display()))?;
        let fixture_index = position.saturating_sub(prompt_len.saturating_sub(1));
        let fixture_id = if full_claim {
            fixture_steps.get(fixture_index).copied().flatten()
        } else {
            None
        };
        let top: Vec<serde_json::Value> = top_logits(logits, 20)
            .into_iter()
            .map(|(id, value)| serde_json::json!([id, f64::from(value)]))
            .collect();
        let object = serde_json::json!({
            "position": position,
            "device_argmax": output.token,
            "logits_argmax": u32::try_from(argmax).unwrap_or(u32::MAX),
            "reference_id": reference,
            "hybrid_id": hybrid,
            "fixture_id": fixture_id,
            "exact_token_equal": exact,
            "vocab": logits.len(),
            "logits_offset": offset,
            "logits_bin": bin_path.display().to_string(),
            "top20": top,
        });
        writeln!(text, "{object}")
            .map_err(|error| format!("logits dump {}: {error}", path.display()))?;
        offset += logits.len() * 4;
        written += 1;
    }
    text.flush()
        .map_err(|error| format!("logits dump {}: {error}", path.display()))?;
    bin.flush()
        .map_err(|error| format!("logits dump {}: {error}", bin_path.display()))?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output(logits: Vec<f32>, token: u32) -> HybridTokenOutput {
        HybridTokenOutput {
            layers: Vec::new(),
            tail_stages: Vec::new(),
            head_input: Vec::new(),
            logits,
            token: Some(token),
            certified_mask: None,
        }
    }

    /// The dump written by the D40 protocol demo on 2026-10-19, kept as
    /// `tests/fixtures/logits-dump-demo-scale.jsonl`. It was not produced by this writer: it is the
    /// existing artefact the new writer has to stay compatible with.
    const EXISTING_DUMP: &str = include_str!("../tests/fixtures/logits-dump-demo-scale.jsonl");

    fn objects(text: &str) -> Vec<serde_json::Value> {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("valid JSON line"))
            .collect()
    }

    fn fields_of(object: &serde_json::Value) -> Vec<String> {
        object
            .as_object()
            .expect("a JSON object")
            .keys()
            .cloned()
            .collect()
    }

    #[test]
    fn the_dump_writes_one_jsonl_object_and_one_le_f32_block_per_position() {
        let outputs = vec![
            output(vec![0.5, -1.0, 3.25, 2.0], 2),
            output(vec![9.0, 0.0], 0),
        ];
        let rows: Vec<ComparedRow> = vec![
            (129usize, Some(3), Some(2), false),
            (130, Some(0), Some(0), true),
        ];
        let fixture_steps = vec![Some(7), Some(0)];
        let path = std::env::temp_dir().join(format!(
            "sparkflash-logits-dump-test-{}.jsonl",
            std::process::id()
        ));
        // The full 130-prompt claim starts the token table at position 129 = prompt_len - 1.
        let written = write_logits_dump(&path, &rows, &outputs, 0, true, &fixture_steps, 130)
            .expect("the dump writes");
        assert_eq!(written, 2);
        let text = std::fs::read_to_string(&path).expect("the jsonl reads back");
        let lines = objects(&text);
        assert_eq!(lines.len(), 2);
        let first = &lines[0];
        assert_eq!(first["position"], 129);
        assert_eq!(first["device_argmax"], 2);
        assert_eq!(first["logits_argmax"], 2);
        assert_eq!(first["reference_id"], 3);
        assert_eq!(first["fixture_id"], 7);
        assert_eq!(first["vocab"], 4);
        assert_eq!(first["logits_offset"], 0);
        let top = first["top20"].as_array().expect("top20 is an array");
        assert_eq!(top.len(), 4);
        assert_eq!(top[0], serde_json::json!([2, 3.25]));
        assert_eq!(top[1], serde_json::json!([3, 2.0]));
        let second = &lines[1];
        assert_eq!(second["position"], 130);
        assert_eq!(second["logits_offset"], 16);
        assert_eq!(second["fixture_id"], 0);
        let sidecar = sidecar_path(&path);
        let bin = std::fs::read(&sidecar).expect("the f32 sidecar reads");
        assert_eq!(bin.len(), (4 + 2) * 4);
        assert_eq!(
            f32::from_le_bytes(bin[0..4].try_into().expect("four bytes")),
            0.5
        );
        assert_eq!(
            f32::from_le_bytes(bin[16..20].try_into().expect("four bytes")),
            9.0
        );
        std::fs::remove_file(&path).expect("the jsonl is removable");
        std::fs::remove_file(&sidecar).expect("the sidecar is removable");
    }

    #[test]
    fn every_object_carries_exactly_the_frozen_fields() {
        let outputs = vec![output(vec![0.0, 1.0], 1)];
        let rows: Vec<ComparedRow> = vec![(0usize, None, Some(1), false)];
        let path = std::env::temp_dir().join(format!(
            "sparkflash-logits-schema-test-{}.jsonl",
            std::process::id()
        ));
        write_logits_dump(&path, &rows, &outputs, 0, false, &[], 1).expect("the dump writes");
        let text = std::fs::read_to_string(&path).expect("the jsonl reads back");
        let objects = objects(&text);
        assert_eq!(objects.len(), 1);
        let expected: Vec<String> = FIELDS.iter().map(|field| (*field).to_owned()).collect();
        assert_eq!(
            fields_of(&objects[0]),
            expected,
            "the writer emits exactly the documented fields, in the documented order"
        );
        for field in CONSUMED_FIELDS {
            assert!(
                objects[0].get(field).is_some(),
                "the frozen consumer requires {field}"
            );
        }
        std::fs::remove_file(&path).expect("the jsonl is removable");
        std::fs::remove_file(sidecar_path(&path)).expect("the sidecar is removable");
    }

    #[test]
    fn the_existing_d40_demo_dump_matches_the_same_schema() {
        let objects = objects(EXISTING_DUMP);
        assert_eq!(
            objects.len(),
            9,
            "the demo dump has nine compared positions"
        );
        let mut expected_offset = 0usize;
        for (index, object) in objects.iter().enumerate() {
            let expected_position = 129u64 + index as u64;
            for field in CONSUMED_FIELDS {
                assert!(
                    object.get(field).is_some(),
                    "existing dump position {index} is missing {field}"
                );
            }
            for field in fields_of(object) {
                assert!(
                    FIELDS.contains(&field.as_str()),
                    "existing dump position {index} has a field this writer does not document: {field}"
                );
            }
            assert_eq!(object["position"], expected_position);
            let vocab = object["vocab"].as_u64().expect("vocab is an integer") as usize;
            let offset = object["logits_offset"]
                .as_u64()
                .expect("offset is an integer") as usize;
            assert_eq!(
                offset, expected_offset,
                "the existing dump's offsets tile its sidecar"
            );
            expected_offset += vocab * 4;
            let top = object["top20"].as_array().expect("top20 is an array");
            assert!(!top.is_empty());
            for pair in top {
                let pair = pair.as_array().expect("each top20 entry is a pair");
                assert_eq!(pair.len(), 2);
                assert!(pair[0].as_u64().is_some(), "the token id is an integer");
                assert!(pair[1].as_f64().is_some(), "the logit is a number");
            }
            let bin = object["logits_bin"]
                .as_str()
                .expect("logits_bin is a string");
            let dump = bin
                .strip_suffix(".bin")
                .expect("the sidecar name ends in .bin");
            assert_eq!(
                sidecar_path(Path::new(dump)),
                Path::new(bin),
                "an existing dump names its sidecar the way sidecar_path does"
            );
        }
    }
}
