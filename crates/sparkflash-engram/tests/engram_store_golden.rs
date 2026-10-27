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

//! Integration test: the Engram reader against a golden produced by an independent
//! implementation.
//!
//! `tests/fixtures/store/golden_interleave.py` is a small python program that reimplements
//! the reference conversion rules (rank split with ceil division, `pad_value` 1 for the
//! scale tensor, the 264-byte interleaved row, 512-byte file alignment) without using any
//! Rust code. The bytes it writes are checked in, and this test reads them back through the
//! same `DirectTable` the server uses. A reader that only round-trips its own writer cannot
//! catch a shared misunderstanding of the layout; this one can.
//!
//! Scope note: the fixture models rank 0 of 1 with a padded tail, a shape `convert.py` cannot
//! emit at mp=1 because a single rank needs no padding. It therefore pins the storage
//! convention (ceil split, pad scale byte, interleaved 264-byte row, 512-byte file alignment)
//! against an independent implementation, not agreement with a real weight conversion.

use sparkflash_engram::layout::{ENGRAM_HEAD_DIM, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES};
use sparkflash_engram::packer::PAD_SCALE_BYTE;
use sparkflash_engram::store::{DirectTable, RowTable, TableSpec, TailFill};
use std::path::PathBuf;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/store");

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(format!("{FIXTURE_DIR}/{name}"))
}

fn expectation() -> serde_json::Value {
    let text =
        std::fs::read_to_string(fixture("golden_expect.json")).expect("read golden_expect.json");
    serde_json::from_str(&text).expect("parse golden_expect.json")
}

fn unhex(text: &str) -> Vec<u8> {
    assert!(
        text.len().is_multiple_of(2),
        "odd hex length {}",
        text.len()
    );
    (0..text.len() / 2)
        .map(|index| {
            u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hexadecimal byte")
        })
        .collect()
}

fn u64_field(value: &serde_json::Value, key: &str) -> u64 {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| panic!("golden_expect.json has no integer {key}"))
}

#[test]
fn the_reader_returns_the_independently_generated_rows() {
    let expect = expectation();
    assert_eq!(u64_field(&expect, "row_bytes"), ENGRAM_ROW_BYTES as u64);
    assert_eq!(u64_field(&expect, "weight_bytes"), ENGRAM_HEAD_DIM as u64);
    assert_eq!(u64_field(&expect, "scale_bytes"), ENGRAM_SCALE_BYTES as u64);
    let rows = u64_field(&expect, "rows");
    let source_rows = u64_field(&expect, "source_rows");
    let file_bytes = u64_field(&expect, "file_bytes");
    assert_eq!(rows, source_rows + 1, "the fixture has one padded tail row");

    let packed = fixture("golden_l1_r0of1.bin");
    assert_eq!(
        std::fs::metadata(&packed).expect("stat the golden").len(),
        file_bytes,
        "the checked-in golden is the file the generator described"
    );
    let table = DirectTable::open(vec![TableSpec {
        layer_id: 1,
        layer_index: 0,
        rank: 0,
        world_size: 1,
        path: packed,
        row_start: 0,
        physical_rows: rows,
        logical_rows: rows,
        window_bytes: 512,
        tail_fill: TailFill::Reference,
    }])
    .expect("open the golden shard");

    let samples = expect["samples"].as_object().expect("samples object");
    assert_eq!(
        samples.len(),
        expect["sample_rows"].as_array().unwrap().len()
    );
    for (key, value) in samples {
        let row: u64 = key.parse().expect("sample row index");
        let expected = unhex(value.as_str().expect("hex string"));
        assert_eq!(expected.len(), ENGRAM_ROW_BYTES);
        let got = table.read_row(0, row).expect("read a golden row");
        assert_eq!(
            got.as_slice(),
            expected.as_slice(),
            "row {row} differs from the independent golden"
        );
        if row >= source_rows {
            // The reference pads the short rank tail with pad_value: zero weights and unit
            // scales, which is 0x7F in F8_E8M0.
            assert!(got[..ENGRAM_HEAD_DIM].iter().all(|byte| *byte == 0));
            assert!(
                got[ENGRAM_HEAD_DIM..]
                    .iter()
                    .all(|byte| *byte == PAD_SCALE_BYTE)
            );
        } else {
            assert!(
                got[..ENGRAM_HEAD_DIM].iter().any(|byte| *byte != 0),
                "a source row must not look like padding"
            );
        }
    }

    // Every row of the fixture is served, including the padded tail, and no row repeats.
    let mut seen = std::collections::BTreeSet::new();
    for row in 0..rows {
        let got = table.read_row(0, row).expect("read a golden row");
        assert_eq!(got.len(), ENGRAM_ROW_BYTES);
        assert!(seen.insert(got), "row {row} repeats another row");
    }
}

#[test]
fn the_golden_source_matches_the_packed_file() {
    // Recomputing the interleave here from the checked-in source keeps the reader honest even
    // if the packed golden is ever regenerated with a different script.
    let expect = expectation();
    let source = std::fs::read(fixture("golden_source.bin")).expect("read the golden source");
    let source_rows = u64_field(&expect, "source_rows") as usize;
    assert_eq!(source.len(), source_rows * ENGRAM_ROW_BYTES);
    let packed = std::fs::read(fixture("golden_l1_r0of1.bin")).expect("read the packed golden");
    let (weight, scale) = source.split_at(source_rows * ENGRAM_HEAD_DIM);
    for row in 0..source_rows {
        let mut expected = Vec::with_capacity(ENGRAM_ROW_BYTES);
        expected.extend_from_slice(&weight[row * ENGRAM_HEAD_DIM..(row + 1) * ENGRAM_HEAD_DIM]);
        expected
            .extend_from_slice(&scale[row * ENGRAM_SCALE_BYTES..(row + 1) * ENGRAM_SCALE_BYTES]);
        let start = row * ENGRAM_ROW_BYTES;
        assert_eq!(
            &packed[start..start + ENGRAM_ROW_BYTES],
            expected.as_slice(),
            "row {row} of the packed golden is not the interleaved source row"
        );
    }
    // The tail row is padding, and the file is zero-filled to the alignment.
    let tail = &packed[source_rows * ENGRAM_ROW_BYTES..(source_rows + 1) * ENGRAM_ROW_BYTES];
    assert!(tail[..ENGRAM_HEAD_DIM].iter().all(|byte| *byte == 0));
    assert!(
        tail[ENGRAM_HEAD_DIM..]
            .iter()
            .all(|byte| *byte == PAD_SCALE_BYTE)
    );
    assert!(
        packed[(source_rows + 1) * ENGRAM_ROW_BYTES..]
            .iter()
            .all(|byte| *byte == 0)
    );
    assert_eq!(packed.len(), u64_field(&expect, "file_bytes") as usize);
}
