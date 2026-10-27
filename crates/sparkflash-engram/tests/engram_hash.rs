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

//! Golden-vector tests for the tokenizer compressor and the 24-column n-gram hash.
//!
//! Every expected value here was produced by the released reference implementation
//! (`/home/user/dsv41-research/inference/engram.py`, class `NgramHashState`) running under
//! torch against `reference/tokenizer.json`; see
//! `tests/fixtures/hash/generate_fixtures.py` for the exact command. Nothing in this file
//! needs model weights, NVMe or the cluster - only the tokenizer JSON that is in the repo.

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde_json::Value;
use sha2::{Digest, Sha256};

use sparkflash_engram::compress::{
    EngramError, RAW_VOCAB_SIZE, REFERENCE_TOKENIZER_SHA256, REFERENCE_TOKENIZER_SHA256_HEX,
    TokenizerCompressor, sha256_file,
};
use sparkflash_engram::hash::{HASH_COLUMNS, NgramHasher, hash_history, hash_window};
use sparkflash_engram::primes::{MULTIPLIERS, OFFSETS, PRIMES};

/// The exact tokenizer.json the fixtures were generated from; a change must regenerate them.
const TOKENIZER_SHA256: &str = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";

const TOKEN_MAP_MAGIC: &[u8; 4] = b"EGTM";
const TOKEN_MAP_VERSION: u32 = 1;
const TOKEN_MAP_HEADER_LEN: usize = 48;
const COMPRESSED_VOCAB_SIZE: u32 = 99_092;
const PAD_RAW_ID: u32 = 2;
const MAX_SEQ_LEN: usize = 512;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hash")
}

fn tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../reference/tokenizer.json")
}

/// Builds the compressor once per test binary: the full 129,280-token map takes a few
/// seconds, and every test wants the same one. Uses the digest-verified boot path.
fn shared_compressor() -> &'static TokenizerCompressor {
    static COMPRESSOR: OnceLock<TokenizerCompressor> = OnceLock::new();
    COMPRESSOR.get_or_init(|| {
        TokenizerCompressor::from_reference_tokenizer_json(tokenizer_path())
            .expect("the reference tokenizer.json must build a compressor")
    })
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    to_hex(&hasher.finalize())
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Parses `token_map.bin`: magic, version, count, compressed size, sha256, then u32 LE data.
fn load_token_map_fixture() -> (usize, u32, String, Vec<u32>) {
    let bytes = fs::read(fixture_dir().join("token_map.bin")).expect("token_map.bin must exist");
    assert!(
        bytes.len() > TOKEN_MAP_HEADER_LEN,
        "token_map.bin is truncated"
    );
    assert_eq!(&bytes[..4], TOKEN_MAP_MAGIC, "token_map.bin magic");
    assert_eq!(
        read_u32_le(&bytes, 4),
        TOKEN_MAP_VERSION,
        "token_map.bin version"
    );
    let count = read_u32_le(&bytes, 8) as usize;
    let compressed = read_u32_le(&bytes, 12);
    let sha = bytes[16..48]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let data = bytes[TOKEN_MAP_HEADER_LEN..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect::<Vec<u32>>();
    assert_eq!(data.len(), count, "token_map.bin payload length");
    (count, compressed, sha, data)
}

fn load_hash_cases() -> Value {
    let path = fixture_dir().join("hash_cases.json");
    let bytes = fs::read(&path).expect("hash_cases.json must exist");
    serde_json::from_slice(&bytes).expect("hash_cases.json must be valid JSON")
}

/// Compares one push's columns against the fixture, reporting the first mismatch with its
/// `(position, layer, column)` coordinates.
fn assert_columns_match(context: &str, actual: &[[[u32; HASH_COLUMNS]; 2]], expected: &Value) {
    let positions = expected
        .as_array()
        .unwrap_or_else(|| panic!("{context}: expected positions must be an array"));
    assert_eq!(positions.len(), actual.len(), "{context}: position count");
    for (position, expected_position) in positions.iter().enumerate() {
        let layers = expected_position
            .as_array()
            .unwrap_or_else(|| panic!("{context}: position {position} must be an array"));
        assert_eq!(
            layers.len(),
            2,
            "{context}: position {position} layer count"
        );
        for (layer, expected_layer) in layers.iter().enumerate() {
            let columns = expected_layer
                .as_array()
                .unwrap_or_else(|| panic!("{context}: layer {layer} must be an array"));
            assert_eq!(
                columns.len(),
                HASH_COLUMNS,
                "{context}: layer {layer} column count"
            );
            for (column, expected_column) in columns.iter().enumerate() {
                let want = expected_column
                    .as_u64()
                    .unwrap_or_else(|| panic!("{context}: column {column} must be an integer"))
                    as u32;
                let got = actual[position][layer][column];
                assert_eq!(
                    got, want,
                    "first mismatch in {context}: position {position}, layer {layer}, \
                     column {column}: expected {want}, got {got}"
                );
            }
        }
    }
}

#[test]
fn tokenizer_json_matches_the_pinned_sha256() {
    let bytes = fs::read(tokenizer_path()).expect("reference/tokenizer.json must exist");
    assert_eq!(
        sha256_hex(&bytes),
        TOKENIZER_SHA256,
        "reference/tokenizer.json changed: regenerating the hash fixtures is required"
    );
    let (_, _, header_sha, _) = load_token_map_fixture();
    assert_eq!(
        header_sha, TOKENIZER_SHA256,
        "token_map.bin was generated from a different tokenizer.json"
    );
    let cases = load_hash_cases();
    assert_eq!(
        cases["tokenizer_sha256"]
            .as_str()
            .expect("tokenizer_sha256 field"),
        TOKENIZER_SHA256,
        "hash_cases.json was generated from a different tokenizer.json"
    );
}

#[test]
fn compressed_map_matches_the_golden_fixture() {
    // The checksum comes first: a changed tokenizer must not silently pass.
    let bytes = fs::read(tokenizer_path()).expect("reference/tokenizer.json must exist");
    assert_eq!(sha256_hex(&bytes), TOKENIZER_SHA256);

    let (count, compressed, _, expected) = load_token_map_fixture();
    assert_eq!(count, RAW_VOCAB_SIZE);
    assert_eq!(compressed, COMPRESSED_VOCAB_SIZE);

    let compressor = shared_compressor();
    assert_eq!(compressor.len(), count);
    assert_eq!(compressor.raw_vocab_size(), count);
    assert_eq!(compressor.compressed_vocab_size(), compressed);
    assert_eq!(compressor.pad_id(), compressor.compress(PAD_RAW_ID));
    assert_eq!(compressor.pad_id(), expected[PAD_RAW_ID as usize]);

    let actual = compressor.lookup();
    assert_eq!(actual.len(), expected.len(), "lookup length");
    for (token_id, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, want,
            "first mismatching token id {token_id}: expected {want}, got {got}"
        );
    }
    let classes = expected.iter().copied().max().expect("map is not empty") + 1;
    assert_eq!(classes, COMPRESSED_VOCAB_SIZE);
}

#[test]
fn compressor_rejects_inconsistent_lookups() {
    let error = TokenizerCompressor::from_lookup(vec![0u32; RAW_VOCAB_SIZE], 99_091)
        .expect_err("a 99,091-class map must be rejected");
    assert!(matches!(
        error,
        EngramError::CompressedVocabSize {
            got: 99_091,
            expected: 99_092
        }
    ));

    let mut lookup = vec![0u32; RAW_VOCAB_SIZE];
    lookup[100] = COMPRESSED_VOCAB_SIZE;
    let error = TokenizerCompressor::from_lookup(lookup, COMPRESSED_VOCAB_SIZE)
        .expect_err("an out-of-range entry must be rejected");
    assert!(matches!(
        error,
        EngramError::LookupEntry {
            token_id: 100,
            value: 99_092,
            ..
        }
    ));

    let error = TokenizerCompressor::from_lookup(vec![0u32; 2], COMPRESSED_VOCAB_SIZE)
        .expect_err("a table without the pad id must be rejected");
    assert!(matches!(error, EngramError::MissingToken { token_id: 2 }));
}

#[test]
fn fixture_layout_matches_the_constants() {
    let cases = load_hash_cases();
    assert_eq!(cases["torch_checked"], Value::Bool(true));
    for layer in 0..2 {
        for order in 0..3 {
            for head in 0..8 {
                assert_eq!(
                    cases["primes"][layer][order][head].as_u64().expect("prime") as u32,
                    PRIMES[layer][order][head],
                    "primes[{layer}][{order}][{head}]"
                );
            }
        }
        for column in 0..24 {
            assert_eq!(
                cases["offsets"][layer][column].as_u64().expect("offset"),
                OFFSETS[layer][column],
                "offsets[{layer}][{column}]"
            );
        }
        for shift in 0..4 {
            assert_eq!(
                cases["multipliers"][layer][shift]
                    .as_i64()
                    .expect("multiplier"),
                MULTIPLIERS[layer][shift],
                "multipliers[{layer}][{shift}]"
            );
        }
    }
}

/// Concatenated ids of every step of one fixture scenario.
fn case_ids(cases: &Value, name: &str) -> Vec<u32> {
    let scenario = cases["cases"]
        .as_array()
        .expect("cases must be an array")
        .iter()
        .find(|scenario| scenario["name"] == name)
        .unwrap_or_else(|| panic!("scenario {name} is missing from hash_cases.json"));
    scenario["steps"]
        .as_array()
        .expect("steps must be an array")
        .iter()
        .flat_map(|step| {
            step["ids"]
                .as_array()
                .expect("ids must be an array")
                .iter()
                .map(|value| value.as_u64().expect("id must be an integer") as u32)
        })
        .collect()
}

#[test]
fn golden_hash_cases_match_the_reference() {
    let cases = load_hash_cases();
    let scenarios = cases["cases"].as_array().expect("cases must be an array");
    assert!(scenarios.len() >= 4, "expected at least four scenarios");
    for scenario in scenarios {
        let name = scenario["name"].as_str().expect("scenario name");
        let steps = scenario["steps"]
            .as_array()
            .expect("steps must be an array");
        let expected = scenario["expected"]
            .as_array()
            .expect("expected must be an array");
        assert_eq!(steps.len(), expected.len(), "{name}: step count");
        let mut hasher = NgramHasher::new(shared_compressor().clone(), MAX_SEQ_LEN);
        for (step_index, (step, expected_step)) in steps.iter().zip(expected.iter()).enumerate() {
            let ids: Vec<u32> = step["ids"]
                .as_array()
                .expect("ids must be an array")
                .iter()
                .map(|value| value.as_u64().expect("id must be an integer") as u32)
                .collect();
            let mask: Option<Vec<bool>> = step["mask"].as_array().map(|mask| {
                mask.iter()
                    .map(|value| value.as_bool().expect("mask must be a bool"))
                    .collect()
            });
            let actual = hasher
                .push(&ids, mask.as_deref())
                .expect("the fixture fits the cache");
            assert_columns_match(&format!("{name} step {step_index}"), &actual, expected_step);
        }
    }
}

/// The stateless whole-history form the serving loop uses gives the reference's columns at every
/// live position of every scenario, from any start; a masked position reads as the dead id.
#[test]
fn history_hashes_match_the_reference() {
    const DEAD_ID: u32 = u32::MAX;
    let cases = load_hash_cases();
    for scenario in cases["cases"].as_array().expect("cases must be an array") {
        let name = scenario["name"].as_str().expect("scenario name");
        let mut history = Vec::new();
        let mut expected = Vec::new();
        for (step, expected_step) in scenario["steps"]
            .as_array()
            .expect("steps must be an array")
            .iter()
            .zip(scenario["expected"].as_array().expect("expected array"))
        {
            let ids = step["ids"].as_array().expect("ids must be an array");
            let mask = step["mask"].as_array();
            for (index, id) in ids.iter().enumerate() {
                let live = mask.is_none_or(|mask| mask[index].as_bool().expect("mask bool"));
                history.push(if live {
                    id.as_u64().expect("id must be an integer") as u32
                } else {
                    DEAD_ID
                });
            }
            expected.extend(expected_step.as_array().expect("positions array").iter());
        }
        assert_eq!(history.len(), expected.len(), "{name}: position count");
        for start in [0, history.len() / 2, history.len() - 1] {
            let mut columns = Vec::new();
            hash_history(shared_compressor(), &history, start, DEAD_ID, &mut columns);
            assert_eq!(columns.len(), (history.len() - start) * 2 * HASH_COLUMNS);
            for (position, row) in columns.chunks(2 * HASH_COLUMNS).enumerate() {
                let position = start + position;
                if history[position] == DEAD_ID {
                    assert!(
                        row.iter().all(|&id| id == DEAD_ID),
                        "{name}: dead row {position}"
                    );
                    continue;
                }
                let want: Vec<u32> = expected[position]
                    .as_array()
                    .expect("layers array")
                    .iter()
                    .flat_map(|layer| layer.as_array().expect("columns array"))
                    .map(|id| id.as_u64().expect("column must be an integer") as u32)
                    .collect();
                assert_eq!(row, want, "{name}: position {position} from start {start}");
            }
        }
    }
}

#[test]
fn streaming_split_is_identical_to_one_shot() {
    let cases = load_hash_cases();
    let scenario = cases["cases"]
        .as_array()
        .expect("cases must be an array")
        .iter()
        .find(|scenario| scenario["name"] == "split_8_plus_1")
        .expect("split_8_plus_1 fixture");
    let expected = scenario["expected"].as_array().expect("expected array");
    let ids = case_ids(&cases, "split_8_plus_1");
    assert_eq!(ids.len(), 9);

    let mut split = NgramHasher::new(shared_compressor().clone(), MAX_SEQ_LEN);
    let mut split_columns = split.push(&ids[..8], None).expect("push 8");
    split_columns.extend(split.push(&ids[8..], None).expect("push 1"));
    assert_eq!(split.start_pos(), 9);

    let mut one_shot = NgramHasher::new(shared_compressor().clone(), MAX_SEQ_LEN);
    let one_shot_columns = one_shot.push(&ids, None).expect("push 9");

    assert_eq!(
        split_columns, one_shot_columns,
        "8+1 must be byte-identical to 9 at once"
    );
    assert_columns_match("split_8_plus_1 step 0", &split_columns[..8], &expected[0]);
    assert_columns_match("split_8_plus_1 step 1", &split_columns[8..], &expected[1]);
}

#[test]
fn reset_rewinds_the_position_cache() {
    let cases = load_hash_cases();
    let ids = case_ids(&cases, "single_step_9");
    let mut hasher = NgramHasher::new(shared_compressor().clone(), MAX_SEQ_LEN);
    let first = hasher.push(&ids, None).expect("first push");
    assert_eq!(hasher.start_pos(), ids.len());
    hasher.reset();
    assert_eq!(hasher.start_pos(), 0);
    assert_eq!(hasher.max_seq_len(), MAX_SEQ_LEN);
    let second = hasher.push(&ids, None).expect("second push");
    assert_eq!(first, second, "a reset hasher must replay identically");
}

#[test]
fn windows_clamp_to_pad_at_the_sequence_start() {
    let compressor = shared_compressor();
    let pad = i64::from(compressor.pad_id());
    let mut hasher = NgramHasher::new(compressor.clone(), MAX_SEQ_LEN);
    let ids = [10u32, 20, 30, 40];
    let actual = hasher.push_compressed(&ids, None).expect("push fits");
    let windows = [
        [10, pad, pad, pad],
        [20, 10, pad, pad],
        [30, 20, 10, pad],
        [40, 30, 20, 10],
    ];
    for (position, (columns, window)) in actual.iter().zip(windows.iter()).enumerate() {
        for (layer, &hash) in columns.iter().enumerate() {
            assert_eq!(
                hash,
                hash_window(layer, window),
                "layer {layer} position {position}"
            );
        }
    }
    for (position, columns) in actual.iter().enumerate() {
        assert_ne!(
            columns[0], columns[1],
            "the two layers must hash differently at position {position}"
        );
    }
}

#[test]
fn dead_positions_block_themselves_and_every_later_ngram() {
    let pad = i64::from(shared_compressor().pad_id());
    let ids = [10u32, 20, 30, 40, 50];
    let mask = [true, false, true, true, true];

    let mut masked = NgramHasher::new(shared_compressor().clone(), MAX_SEQ_LEN);
    let actual = masked
        .push_compressed(&ids, Some(&mask))
        .expect("push fits");
    let windows = [
        [10, pad, pad, pad],
        [pad, pad, pad, pad],
        [30, pad, pad, pad],
        [40, 30, pad, pad],
        [50, 40, 30, pad],
    ];
    for (position, (columns, window)) in actual.iter().zip(windows.iter()).enumerate() {
        for (layer, &hash) in columns.iter().enumerate() {
            assert_eq!(
                hash,
                hash_window(layer, window),
                "layer {layer} position {position}"
            );
        }
    }

    // Positions before the dead one are untouched; the dead position and everything within
    // three tokens after it change.
    let mut unmasked = NgramHasher::new(shared_compressor().clone(), MAX_SEQ_LEN);
    let plain = unmasked.push_compressed(&ids, None).expect("push fits");
    assert_eq!(actual[0][0], plain[0][0], "position 0 is unaffected");
    for (position, (masked_columns, plain_columns)) in
        actual.iter().zip(plain.iter()).enumerate().skip(1)
    {
        assert_ne!(
            masked_columns[0], plain_columns[0],
            "position {position} must be blocked by the dead position"
        );
    }
}

#[test]
fn mask_length_and_capacity_errors_are_reported() {
    let mut hasher = NgramHasher::new(shared_compressor().clone(), 4);
    let error = hasher
        .push_compressed(&[1, 2, 3], Some(&[true, true]))
        .expect_err("a short mask must be rejected");
    assert!(matches!(
        error,
        EngramError::TokenMaskLength {
            given: 2,
            expected: 3
        }
    ));

    let error = hasher
        .push_compressed(&[1, 2, 3, 4, 5], None)
        .expect_err("an oversized push must be rejected");
    assert!(matches!(
        error,
        EngramError::CacheCapacity {
            start_pos: 0,
            len: 5,
            max_seq_len: 4
        }
    ));

    hasher
        .push_compressed(&[1, 2, 3], None)
        .expect("first push fits");
    let error = hasher
        .push_compressed(&[4, 5], None)
        .expect_err("a push past the end must be rejected");
    assert!(matches!(
        error,
        EngramError::CacheCapacity {
            start_pos: 3,
            len: 2,
            max_seq_len: 4
        }
    ));
}

#[test]
fn out_of_vocabulary_ids_compress_to_pad() {
    let compressor = shared_compressor();
    assert_eq!(compressor.compress(u32::MAX), compressor.pad_id());
    assert_eq!(
        compressor.compress(RAW_VOCAB_SIZE as u32),
        compressor.pad_id()
    );
    assert_eq!(
        compressor.compress(RAW_VOCAB_SIZE as u32 - 1),
        compressor.lookup()[RAW_VOCAB_SIZE - 1]
    );
}

#[test]
fn raw_out_of_vocabulary_ids_are_padded_not_rejected() {
    // push() must never hit the compressed range error, because compress()
    // has already mapped every id past the vocabulary (including u32::MAX) onto the pad id.
    let compressor = shared_compressor();
    let raw = [10u32, u32::MAX, RAW_VOCAB_SIZE as u32, 20];
    let as_compressed = [
        compressor.compress(raw[0]),
        compressor.pad_id(),
        compressor.pad_id(),
        compressor.compress(raw[3]),
    ];
    let mut from_raw = NgramHasher::new(compressor.clone(), 8);
    let raw_columns = from_raw.push(&raw, None).expect("push() pads OOV ids");
    let mut from_compressed = NgramHasher::new(compressor.clone(), 8);
    let compressed_columns = from_compressed
        .push_compressed(&as_compressed, None)
        .expect("the same sequence, already compressed");
    assert_eq!(
        raw_columns, compressed_columns,
        "push(raw OOV ids) must equal push_compressed(padded ids)"
    );
}

#[test]
fn compressed_ids_out_of_range_are_rejected_without_partial_state() {
    let valid_max = COMPRESSED_VOCAB_SIZE - 1;
    let mut hasher = NgramHasher::new(shared_compressor().clone(), 8);
    hasher
        .push_compressed(&[0, valid_max], None)
        .expect("the largest valid compressed id must be accepted");
    assert_eq!(hasher.start_pos(), 2);

    // One past the boundary.
    let error = hasher
        .push_compressed(&[7, COMPRESSED_VOCAB_SIZE], None)
        .expect_err("an id at the boundary must be rejected");
    assert!(matches!(
        error,
        EngramError::CompressedIdOutOfRange {
            id: 99_092,
            position: 1,
            compressed_vocab_size: 99_092
        }
    ));
    assert_eq!(
        hasher.start_pos(),
        2,
        "a rejected batch must not advance the position"
    );

    // A raw tokenizer id handed to the compressed entry point: 128000 * 76632096046245
    // overflows i64, so this must fail instead of wrapping.
    let error = hasher
        .push_compressed(&[128_000], None)
        .expect_err("a raw tokenizer id must be rejected");
    assert!(matches!(
        error,
        EngramError::CompressedIdOutOfRange {
            id: 128_000,
            position: 0,
            ..
        }
    ));
    let error = hasher
        .push_compressed(&[u32::MAX], None)
        .expect_err("u32::MAX must be rejected");
    assert!(matches!(
        error,
        EngramError::CompressedIdOutOfRange {
            id: u32::MAX,
            position: 0,
            ..
        }
    ));
    assert_eq!(hasher.start_pos(), 2);

    // Nothing from the rejected batches may be cached: the next accepted push has to produce
    // exactly the columns an untouched hasher would produce.
    let mut clean = NgramHasher::new(shared_compressor().clone(), 8);
    clean
        .push_compressed(&[0, valid_max], None)
        .expect("the same prefix");
    let after_rejections = hasher
        .push_compressed(&[5, 6], None)
        .expect("a valid push after the rejections");
    let from_clean = clean
        .push_compressed(&[5, 6], None)
        .expect("the same push on a clean hasher");
    assert_eq!(
        after_rejections, from_clean,
        "a rejected batch must not leave partial cache state"
    );
}

#[test]
fn verified_constructor_requires_the_reference_digest() {
    let reference = tokenizer_path();
    assert_eq!(to_hex(&REFERENCE_TOKENIZER_SHA256), TOKENIZER_SHA256);
    assert_eq!(REFERENCE_TOKENIZER_SHA256_HEX, TOKENIZER_SHA256);

    let digest = sha256_file(&reference).expect("the reference tokenizer.json is readable");
    assert_eq!(to_hex(&digest), TOKENIZER_SHA256);

    // shared_compressor() is built through from_reference_tokenizer_json, so a digest or
    // vocabulary failure there would already have failed every test in this binary.
    let verified = shared_compressor();
    assert_eq!(verified.compressed_vocab_size(), COMPRESSED_VOCAB_SIZE);

    // A single flipped byte is rejected before the file is parsed, so the failure is always the
    // digest mismatch and never a JSON error.
    let directory = tempfile::tempdir().expect("a temporary directory");
    let flipped = directory.path().join("flipped.json");
    let mut bytes = fs::read(&reference).expect("read the reference tokenizer");
    bytes[100] ^= 0x01;
    fs::write(&flipped, &bytes).expect("write the flipped copy");
    let error =
        TokenizerCompressor::from_tokenizer_json_verified(&flipped, &REFERENCE_TOKENIZER_SHA256)
            .expect_err("a modified tokenizer must be rejected");
    match error {
        EngramError::TokenizerDigestMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, TOKENIZER_SHA256);
            assert_ne!(actual, TOKENIZER_SHA256);
            assert_eq!(actual.len(), expected.len());
        }
        other => panic!("unexpected error for a modified tokenizer: {other:?}"),
    }

    // A copy that is still a perfectly good tokenizer must fail the same way: appending a
    // newline changes the digest but not the parse, which is exactly the substituted-tokenizer
    // case the digest exists to stop.
    let appended = directory.path().join("appended.json");
    let byte_newline = b'\n';
    let mut bytes = fs::read(&reference).expect("read the reference tokenizer");
    bytes.push(byte_newline);
    fs::write(&appended, &bytes).expect("write the appended copy");
    let error =
        TokenizerCompressor::from_tokenizer_json_verified(&appended, &REFERENCE_TOKENIZER_SHA256)
            .expect_err("a valid but different tokenizer must still be rejected");
    assert!(matches!(error, EngramError::TokenizerDigestMismatch { .. }));

    // The unverified constructor accepts that same file, which is precisely why the boot path
    // must not use it: the digest is the only thing between a substituted tokenizer and a full
    // table rehash.
    let unverified = TokenizerCompressor::from_tokenizer_json(&appended)
        .expect("the unverified constructor parses the appended copy");
    assert_eq!(unverified.compressed_vocab_size(), COMPRESSED_VOCAB_SIZE);
    assert_eq!(unverified.lookup(), verified.lookup());
}
