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

//! The safetensors reader against files this test writes itself.
//!
//! The 48 shards of the released checkpoint are not on this machine (510 GB), so every test here
//! builds its own tiny safetensors container: eight bytes of header length, a JSON header and a
//! data section is the whole format, and a reader can be exercised against all of it in a few
//! kilobytes. Two tests read the released index when it is present, because the index is a 7.4 MB
//! file of names and needs no weights.

use std::fs::File;
use std::path::{Path, PathBuf};

use sparkflash_engine::dtype::DType;
use sparkflash_engine::safetensors::{
    CheckpointIndex, DEFAULT_LIMIT, GATHER_WHOLE_TENSOR_MAX_BYTES, MAX_HEADER_BYTES,
    SafeTensorsHeader, SafeTensorsSource, gathers_whole_tensor,
};
use sparkflash_engine::weights::{WeightError, WeightSource};

/// One tensor for the writer: name, container dtype word, stored shape and bytes.
struct Entry<'a> {
    name: &'a str,
    dtype: &'a str,
    shape: &'a [u64],
    data: &'a [u8],
}

/// Builds a container: eight bytes of header length, the JSON header, then the data section, then
/// any unclaimed trailing bytes a test wants to prove are never read.
fn container(entries: &[Entry<'_>], metadata: Option<serde_json::Value>, tail: &[u8]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    if let Some(metadata) = metadata {
        header.insert("__metadata__".to_string(), metadata);
    }
    let mut data = Vec::new();
    for entry in entries {
        let begin = data.len() as u64;
        data.extend_from_slice(entry.data);
        let end = data.len() as u64;
        header.insert(
            entry.name.to_string(),
            serde_json::json!({
                "dtype": entry.dtype,
                "shape": entry.shape,
                "data_offsets": [begin, end],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("a JSON header");
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out.extend_from_slice(tail);
    out
}

/// Writes arbitrary bytes to a file under the directory.
fn write_bytes(dir: &Path, file: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(file);
    std::fs::write(&path, bytes).expect("write the container");
    path
}

/// Writes a container with no metadata and no trailing bytes.
fn write_container(dir: &Path, file: &str, entries: &[Entry<'_>]) -> PathBuf {
    write_bytes(dir, file, &container(entries, None, &[]))
}

/// Writes an index JSON file with the given name -> shard pairs.
fn write_index(
    dir: &Path,
    file: &str,
    pairs: &[(&str, &str)],
    total_size: Option<u128>,
) -> PathBuf {
    let mut map = serde_json::Map::new();
    for (name, shard) in pairs {
        map.insert(
            (*name).to_string(),
            serde_json::Value::String((*shard).to_string()),
        );
    }
    let mut root = serde_json::Map::new();
    if let Some(total) = total_size {
        root.insert(
            "metadata".to_string(),
            serde_json::json!({ "total_size": total }),
        );
    }
    root.insert("weight_map".to_string(), serde_json::Value::Object(map));
    let bytes = serde_json::to_vec(&serde_json::Value::Object(root)).expect("index JSON");
    write_bytes(dir, file, &bytes)
}

/// A source over the directory with a fresh index of the given pairs.
fn source(dir: &Path, pairs: &[(&str, &str)]) -> SafeTensorsSource {
    let index = write_index(dir, "model.safetensors.index.json", pairs, None);
    SafeTensorsSource::open(dir, &index).expect("the index must load")
}

/// Asserts the result is a typed Malformed error whose text mentions the needle.
fn expect_malformed<T: std::fmt::Debug>(result: Result<T, WeightError>, needle: &str) {
    match result {
        Err(WeightError::Malformed { name, detail }) => assert!(
            detail.contains(needle) || name.contains(needle),
            "the error must mention {needle:?}, got name {name:?} detail {detail:?}"
        ),
        other => panic!("expected Malformed mentioning {needle:?}, got {other:?}"),
    }
}

#[test]
fn a_valid_shard_reads_exactly_the_tensor_that_was_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let a = [1u8; 8];
    let b = [2u8; 16];
    let c = [3u8; 4];
    // 64 trailing bytes that belong to no tensor: a reader that reads to the end of the file
    // returns them, and a reader that respects data_offsets does not.
    let tail = [0xFFu8; 64];
    let entries = [
        Entry {
            name: "a.weight",
            dtype: "F32",
            shape: &[2],
            data: &a,
        },
        Entry {
            name: "b.weight",
            dtype: "BF16",
            shape: &[2, 4],
            data: &b,
        },
        Entry {
            name: "c.weight",
            dtype: "F8_E4M3",
            shape: &[4],
            data: &c,
        },
    ];
    write_bytes(
        dir.path(),
        "shard-00001.safetensors",
        &container(&entries, None, &tail),
    );
    let source = source(
        dir.path(),
        &[
            ("c.weight", "shard-00001.safetensors"),
            ("a.weight", "shard-00001.safetensors"),
            ("b.weight", "shard-00001.safetensors"),
        ],
    );

    // names() comes from the index, in its sorted order.
    assert_eq!(
        source.names(),
        vec![
            "a.weight".to_string(),
            "b.weight".to_string(),
            "c.weight".to_string()
        ]
    );
    for (name, dtype, shape) in [
        ("a.weight", DType::F32, vec![2usize]),
        ("b.weight", DType::BF16, vec![2, 4]),
        ("c.weight", DType::F8E4M3, vec![4]),
    ] {
        let spec = source.spec(name).expect("the tensor must be described");
        assert_eq!(spec.dtype, dtype, "{name}");
        assert_eq!(spec.shape, shape, "{name}");
    }
    // The middle tensor is the one that proves the seek: a reader that ignored data_offsets
    // would return the first tensor's bytes, and one that read to the end would append the tail.
    let read = source.read("b.weight").expect("the bytes must read");
    assert_eq!(
        read,
        vec![2u8; 16],
        "the read must be the tensor's own bytes and no more"
    );
    assert_eq!(source.read("a.weight").unwrap(), vec![1u8; 8]);
    assert_eq!(source.read("c.weight").unwrap(), vec![3u8; 4]);
    assert!(source.contains("a.weight") && !source.contains("d.weight"));
}

#[test]
fn the_data_section_starts_where_the_header_ends() {
    let dir = tempfile::tempdir().unwrap();
    let data = [7u8; 8];
    let tail = [9u8; 5];
    let entries = [Entry {
        name: "x",
        dtype: "F32",
        shape: &[2],
        data: &data,
    }];
    let path = write_bytes(
        dir.path(),
        "s.safetensors",
        &container(&entries, None, &tail),
    );
    let header = SafeTensorsHeader::read(&path).expect("the header must parse");
    assert_eq!(header.file_len, (8 + header.header_len()) + 8 + 5);
    assert_eq!(header.data_start, 8 + header.header_len());
    assert_eq!(header.data_len(), 13);
    assert_eq!(header.len(), 1);
    assert!(!header.is_empty());
    let descriptor = header.tensor("x").expect("the tensor must be described");
    assert_eq!(descriptor.byte_len(), 8);
    assert_eq!(descriptor.elements(), 2);
    assert_eq!(descriptor.data_offsets, [0, 8]);
    // The raw file agrees: the tensor's bytes sit at data_start + begin.
    let raw = std::fs::read(&path).unwrap();
    let begin = header.data_start as usize + descriptor.data_offsets[0] as usize;
    assert_eq!(&raw[begin..begin + 8], &data);
    assert!(matches!(
        header.tensor("y"),
        Err(WeightError::Missing { .. })
    ));
}

#[test]
fn a_header_that_is_not_json_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut not_json = (8u64).to_le_bytes().to_vec();
    not_json.extend_from_slice(b"not json");
    let path = write_bytes(dir.path(), "bad.safetensors", &not_json);
    expect_malformed(SafeTensorsHeader::read(&path), "not a JSON object");

    // Valid JSON that is not an object is just as malformed.
    let path = write_bytes(
        dir.path(),
        "array.safetensors",
        &container_with_header("[1,2,3]", &[]),
    );
    expect_malformed(SafeTensorsHeader::read(&path), "not a JSON object");

    // A tensor entry that is not a descriptor names the offending key.
    let path = write_bytes(
        dir.path(),
        "entry.safetensors",
        &container_with_header(r#"{"x": 1}"#, &[]),
    );
    expect_malformed(SafeTensorsHeader::read(&path), "x");
}

#[test]
fn a_declared_header_length_outside_the_bound_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    for declared in [0u64, 1, 2, MAX_HEADER_BYTES + 1, u64::MAX] {
        let path = write_bytes(dir.path(), "length.safetensors", &declared.to_le_bytes());
        expect_malformed(SafeTensorsHeader::read(&path), "header length");
    }
    // A believable length that the file cannot hold is caught too.
    let mut short = (64u64).to_le_bytes().to_vec();
    short.extend_from_slice(b"0123456789");
    let path = write_bytes(dir.path(), "short.safetensors", &short);
    expect_malformed(SafeTensorsHeader::read(&path), "the header claims");
    // And a prefix too short to hold the eight-byte length at all.
    let path = write_bytes(dir.path(), "tiny.safetensors", b"abc");
    expect_malformed(SafeTensorsHeader::read(&path), "header length");
}

#[test]
fn a_tensor_whose_offsets_leave_the_file_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    // The header promises 100 bytes of data; the file holds 10.
    let long = container_with_header(
        r#"{"t":{"dtype":"F32","shape":[25],"data_offsets":[0,100]}}"#,
        &[0u8; 10],
    );
    let path = write_bytes(dir.path(), "long.safetensors", &long);
    expect_malformed(
        SafeTensorsHeader::read(&path),
        "the tensor ends at byte 100",
    );

    // Reversed offsets are malformed even when they fit.
    let reversed = container_with_header(
        r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[8,4]}}"#,
        &[0u8; 8],
    );
    let path = write_bytes(dir.path(), "reversed.safetensors", &reversed);
    expect_malformed(SafeTensorsHeader::read(&path), "reversed");
}

#[test]
fn a_shard_missing_from_the_directory_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let source = source(dir.path(), &[("t", "absent.safetensors")]);
    match source.spec("t") {
        Err(WeightError::Io { name, detail }) => {
            assert_eq!(name, "t");
            assert!(detail.contains("absent.safetensors"), "{detail}");
        }
        other => panic!("expected Io for a missing shard, got {other:?}"),
    }
    assert!(matches!(source.read("t"), Err(WeightError::Io { .. })));
    // read uses the same path: it reports the missing shard, not a size problem.
    assert!(matches!(source.header("t"), Err(WeightError::Io { .. })));
}

#[test]
fn a_name_absent_from_the_index_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [Entry {
        name: "known",
        dtype: "F32",
        shape: &[1],
        data: &[0u8; 4],
    }];
    write_container(dir.path(), "s.safetensors", &entries);
    let source = source(dir.path(), &[("known", "s.safetensors")]);
    match source.spec("absent") {
        Err(WeightError::Missing { name }) => assert_eq!(name, "absent"),
        other => panic!("expected Missing, got {other:?}"),
    }
    assert!(matches!(
        source.read("absent"),
        Err(WeightError::Missing { .. })
    ));
    assert!(matches!(
        source.header("absent"),
        Err(WeightError::Missing { .. })
    ));
    assert!(!source.contains("absent"));
    assert_eq!(source.index().len(), 1);
    assert!(!source.index().is_empty());
}

#[test]
fn the_metadata_entry_is_not_a_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [Entry {
        name: "t",
        dtype: "BF16",
        shape: &[2],
        data: &[0u8; 4],
    }];
    // Hugging Face writes an object here, but the specification allows any JSON at all.
    for metadata in [
        serde_json::json!({ "format": "pt" }),
        serde_json::json!("a string"),
        serde_json::json!([1, 2, 3]),
        serde_json::json!(7),
    ] {
        let path = write_bytes(
            dir.path(),
            "meta.safetensors",
            &container(&entries, Some(metadata.clone()), &[]),
        );
        let header = SafeTensorsHeader::read(&path).expect("metadata must not break the header");
        assert_eq!(header.len(), 1, "metadata is not a tensor");
        assert_eq!(header.metadata, Some(metadata));
    }
    // A shard that holds only metadata describes no tensors.
    let path = write_bytes(
        dir.path(),
        "only-meta.safetensors",
        &container(&[], Some(serde_json::json!({ "format": "pt" })), &[]),
    );
    let header = SafeTensorsHeader::read(&path).expect("a metadata-only shard must parse");
    assert!(header.is_empty());
    assert!(header.tensors.is_empty());
    assert!(matches!(
        header.tensor("t"),
        Err(WeightError::Missing { .. })
    ));
}

#[test]
fn an_empty_tensor_reads_as_no_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [Entry {
        name: "empty",
        dtype: "F32",
        shape: &[0],
        data: &[],
    }];
    write_container(dir.path(), "s.safetensors", &entries);
    let source = source(dir.path(), &[("empty", "s.safetensors")]);
    let spec = source
        .spec("empty")
        .expect("an empty tensor is still described");
    assert_eq!(spec.shape, vec![0]);
    assert_eq!(spec.storage_bytes(), 0);
    assert_eq!(source.read("empty").unwrap(), Vec::<u8>::new());
}

/// Writes a container whose header is given verbatim, so a test can declare a length that does not
/// match, an offset past the end, or a dtype that does not exist.
fn container_with_header(header_json: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header_json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn the_limit_refuses_a_tensor_before_anything_is_allocated() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [Entry {
        name: "big",
        dtype: "F32",
        shape: &[4],
        data: &[0u8; 16],
    }];
    write_container(dir.path(), "s.safetensors", &entries);
    let index = write_index(
        dir.path(),
        "model.safetensors.index.json",
        &[("big", "s.safetensors")],
        None,
    );
    let source = SafeTensorsSource::open(dir.path(), &index)
        .unwrap()
        .with_limit(8);
    assert_eq!(source.limit(), 8);
    match source.read("big") {
        Err(WeightError::TooLarge { name, bytes, limit }) => {
            assert_eq!(name, "big");
            assert_eq!(bytes, 16);
            assert_eq!(limit, 8);
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
    // A refusal is about reading, not about describing: the policy can still see the size.
    assert_eq!(source.spec("big").unwrap().storage_bytes(), 16);
    // The boundary is inclusive, so a limit equal to the tensor's size reads it.
    let exact = SafeTensorsSource::open(dir.path(), &index)
        .unwrap()
        .with_limit(16);
    assert_eq!(exact.read("big").unwrap(), vec![0u8; 16]);
    // The default admits the 1.32 GB pinned embedding and refuses the 98 GB Engram tables.
    let default = SafeTensorsSource::open(dir.path(), &index).unwrap();
    assert_eq!(default.limit(), DEFAULT_LIMIT);
    assert_eq!(DEFAULT_LIMIT, 2 * 1024 * 1024 * 1024);
    assert!(default.read("big").is_ok());
}

#[test]
fn a_refused_name_is_refused_by_policy_whatever_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [
        Entry {
            name: "layers.1.engram.embed.weight",
            dtype: "BF16",
            shape: &[2],
            data: &[0u8; 4],
        },
        Entry {
            name: "small.weight",
            dtype: "BF16",
            shape: &[2],
            data: &[1u8; 4],
        },
    ];
    write_container(dir.path(), "s.safetensors", &entries);
    let index = write_index(
        dir.path(),
        "model.safetensors.index.json",
        &[
            ("layers.1.engram.embed.weight", "s.safetensors"),
            ("small.weight", "s.safetensors"),
        ],
        None,
    );
    let source = SafeTensorsSource::open(dir.path(), &index)
        .unwrap()
        .with_limit(u128::MAX)
        .with_refused(["layers.1.engram.embed.weight"]);
    assert!(source.refused().contains("layers.1.engram.embed.weight"));
    match source.read("layers.1.engram.embed.weight") {
        Err(WeightError::Refused { name, detail }) => {
            assert_eq!(name, "layers.1.engram.embed.weight");
            // The refusal is about the name, so it says nothing about a size it never read.
            assert!(!detail.is_empty(), "a refusal must say why");
        }
        other => panic!("expected a policy refusal, got {other:?}"),
    }
    // A refused tensor is still describable, and everything else still reads.
    assert_eq!(
        source.spec("layers.1.engram.embed.weight").unwrap().shape,
        vec![2]
    );
    assert_eq!(source.read("small.weight").unwrap(), vec![1u8; 4]);
    assert_eq!(source.index().shards(), vec!["s.safetensors"]);
}

#[test]
fn a_refused_name_is_refused_without_the_shard_being_opened() {
    // The refusal list exists so the 98 GB Engram tables are never described, let alone read. It is
    // therefore a statement about the NAME, and it must be decided before the shard is touched: a
    // refused tensor whose shard is missing or unreadable must still be refused, because the caller
    // asked for something this source will not serve and the state of the file cannot change that.
    // Reporting Io here would mean the check ran only after a file was opened.
    let dir = tempfile::tempdir().unwrap();
    let index = write_index(
        dir.path(),
        "model.safetensors.index.json",
        &[("layers.1.engram.embed.weight", "never-written.safetensors")],
        None,
    );
    let source = SafeTensorsSource::open(dir.path(), &index)
        .unwrap()
        .with_refused(["layers.1.engram.embed.weight"]);
    // The shard named by the index does not exist at all.
    match source.read("layers.1.engram.embed.weight") {
        Err(WeightError::Refused { name, .. }) => {
            assert_eq!(name, "layers.1.engram.embed.weight");
        }
        other => panic!("a refused name must be refused before its shard is opened, got {other:?}"),
    }
    // And it is still describable, which is what lets a policy see what it is refusing.
    match source.spec("layers.1.engram.embed.weight") {
        Err(WeightError::Io { .. }) => {}
        other => {
            panic!("describing it does need the shard, so this should be an Io error: {other:?}")
        }
    }
}

#[test]
fn dtype_words_map_to_the_plan_dtypes() {
    let dir = tempfile::tempdir().unwrap();
    let f32_bytes = [0u8; 8];
    let bf16_bytes = [0u8; 4];
    let f16_bytes = [0u8; 4];
    let e4m3_bytes = [0u8; 4];
    let e8m0_bytes = [0u8; 4];
    let packed_bytes = [0u8; 8];
    let entries = [
        Entry {
            name: "f32",
            dtype: "F32",
            shape: &[2],
            data: &f32_bytes,
        },
        Entry {
            name: "bf16",
            dtype: "BF16",
            shape: &[2],
            data: &bf16_bytes,
        },
        Entry {
            name: "f16",
            dtype: "F16",
            shape: &[2],
            data: &f16_bytes,
        },
        Entry {
            name: "e4m3",
            dtype: "F8_E4M3",
            shape: &[4],
            data: &e4m3_bytes,
        },
        Entry {
            name: "e8m0",
            dtype: "F8_E8M0",
            shape: &[4],
            data: &e8m0_bytes,
        },
        // The checkpoint stores FP4 E2M1 two values per byte and calls those bytes I8 with the
        // packed shape, so the logical shape doubles the last dimension.
        Entry {
            name: "packed",
            dtype: "I8",
            shape: &[2, 4],
            data: &packed_bytes,
        },
    ];
    let f64_bytes = [0u8; 16];
    let u8_bytes = [0u8; 4];
    let i32_bytes = [0u8; 4];
    let unknown = [
        Entry {
            name: "f64",
            dtype: "F64",
            shape: &[2],
            data: &f64_bytes,
        },
        Entry {
            name: "u8",
            dtype: "U8",
            shape: &[4],
            data: &u8_bytes,
        },
        Entry {
            name: "i32",
            dtype: "I32",
            shape: &[1],
            data: &i32_bytes,
        },
    ];
    write_container(dir.path(), "s.safetensors", &entries);
    write_container(dir.path(), "unknown.safetensors", &unknown);
    let mut pairs: Vec<(&str, &str)> = entries
        .iter()
        .map(|entry| (entry.name, "s.safetensors"))
        .collect();
    pairs.extend(
        unknown
            .iter()
            .map(|entry| (entry.name, "unknown.safetensors")),
    );
    let source = source(dir.path(), &pairs);

    assert_eq!(source.spec("f32").unwrap().dtype, DType::F32);
    assert_eq!(source.spec("bf16").unwrap().dtype, DType::BF16);
    assert_eq!(source.spec("f16").unwrap().dtype, DType::F16);
    assert_eq!(source.spec("e4m3").unwrap().dtype, DType::F8E4M3);
    assert_eq!(source.spec("e8m0").unwrap().dtype, DType::F8E8M0);
    let packed = source.spec("packed").expect("I8 is packed FP4");
    assert_eq!(packed.dtype, DType::F4E2M1);
    assert_eq!(packed.shape, vec![2, 8], "the last dimension doubles");
    assert_eq!(packed.elements(), 16);
    assert_eq!(packed.storage_bytes(), 8);
    // A dtype this checkpoint does not use is malformed, not a panic.
    for (name, word) in [("f64", "F64"), ("u8", "U8"), ("i32", "I32")] {
        expect_malformed(source.spec(name), word);
    }
}

#[test]
fn a_header_whose_byte_length_disagrees_with_its_descriptor_is_malformed() {
    let dir = tempfile::tempdir().unwrap();
    // F8_E4M3 is one byte per element: four elements cannot occupy eight bytes.
    let lying = container_with_header(
        r#"{"t":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    );
    write_bytes(dir.path(), "lying.safetensors", &lying);
    // A container header only carries the dtype word: the source is what maps it to a dtype, and
    // that is where the disagreement is caught.
    let reader = source(dir.path(), &[("t", "lying.safetensors")]);
    expect_malformed(reader.spec("t"), "need 4");
    assert!(matches!(
        reader.read("t"),
        Err(WeightError::Malformed { .. })
    ));

    // A packed I8 tensor halves into FP4: four packed bytes are eight elements and four bytes, so
    // eight stored bytes for a [4] shape is malformed too.
    let lying = container_with_header(
        r#"{"t":{"dtype":"I8","shape":[4],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    );
    write_bytes(dir.path(), "lying-i8.safetensors", &lying);
    let reader = source(dir.path(), &[("t", "lying-i8.safetensors")]);
    expect_malformed(reader.spec("t"), "need 4");

    // The same shape with the right length is accepted, so the check is not vacuous.
    let honest = container(
        &[Entry {
            name: "t",
            dtype: "F8_E4M3",
            shape: &[4],
            data: &[0u8; 4],
        }],
        None,
        &[],
    );
    write_bytes(dir.path(), "honest.safetensors", &honest);
    let path = dir.path().join("honest.safetensors");
    let header = SafeTensorsHeader::read(&path).expect("four bytes for four elements");
    assert_eq!(header.tensor("t").unwrap().byte_len(), 4);
    let reader = source(dir.path(), &[("t", "honest.safetensors")]);
    assert_eq!(reader.spec("t").unwrap().storage_bytes(), 4);
    assert_eq!(reader.read("t").unwrap(), vec![0u8; 4]);
}

#[test]
fn a_shard_name_that_leaves_the_directory_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    for shard in ["../evil.safetensors", "sub/shard.safetensors", ""] {
        let source = source(dir.path(), &[("t", shard)]);
        expect_malformed(source.spec("t"), "outside the shard directory");
        assert!(matches!(
            source.read("t"),
            Err(WeightError::Malformed { .. })
        ));
        assert!(matches!(
            source.header("t"),
            Err(WeightError::Malformed { .. })
        ));
    }
}

#[test]
fn two_shards_are_dispatched_by_the_index_and_their_headers_are_cached() {
    let dir = tempfile::tempdir().unwrap();
    let a1 = [1u8; 4];
    let a2 = [2u8; 4];
    let b1 = [3u8; 4];
    write_container(
        dir.path(),
        "shard-a.safetensors",
        &[
            Entry {
                name: "a1",
                dtype: "F32",
                shape: &[1],
                data: &a1,
            },
            Entry {
                name: "a2",
                dtype: "F32",
                shape: &[1],
                data: &a2,
            },
        ],
    );
    write_container(
        dir.path(),
        "shard-b.safetensors",
        &[Entry {
            name: "b1",
            dtype: "F32",
            shape: &[1],
            data: &b1,
        }],
    );
    let source = source(
        dir.path(),
        &[
            ("a1", "shard-a.safetensors"),
            ("a2", "shard-a.safetensors"),
            ("b1", "shard-b.safetensors"),
        ],
    );
    assert_eq!(
        source.index().shards(),
        vec!["shard-a.safetensors", "shard-b.safetensors"]
    );
    assert_eq!(source.read("a1").unwrap(), vec![1u8; 4]);
    assert_eq!(source.read("a2").unwrap(), vec![2u8; 4]);
    assert_eq!(source.read("b1").unwrap(), vec![3u8; 4]);
    // One header per shard, whatever the number of tensors asked for.
    let first = source.header("a1").unwrap();
    let second = source.header("a2").unwrap();
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "the shard header is cached"
    );
    let other = source.header("b1").unwrap();
    assert!(!std::sync::Arc::ptr_eq(&first, &other));
    assert_eq!(first.len(), 2);
    assert_eq!(other.len(), 1);
    assert_eq!(
        source.shard_path("shard-a.safetensors"),
        dir.path().join("shard-a.safetensors")
    );
    assert_eq!(source.dir(), dir.path());
}

#[test]
fn a_tensor_in_a_shard_but_not_in_the_index_is_not_addressable() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [
        Entry {
            name: "listed",
            dtype: "F32",
            shape: &[1],
            data: &[0u8; 4],
        },
        Entry {
            name: "unlisted",
            dtype: "F32",
            shape: &[1],
            data: &[0u8; 4],
        },
    ];
    write_container(dir.path(), "s.safetensors", &entries);
    let source = source(dir.path(), &[("listed", "s.safetensors")]);
    assert_eq!(source.names(), vec!["listed".to_string()]);
    assert!(matches!(
        source.spec("unlisted"),
        Err(WeightError::Missing { .. })
    ));
    assert!(matches!(
        source.read("unlisted"),
        Err(WeightError::Missing { .. })
    ));
    assert_eq!(
        source.header("listed").unwrap().len(),
        2,
        "the header still holds both, but the index is the authority"
    );
}

#[test]
fn a_large_data_section_is_never_read_to_describe_one_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let entries = [Entry {
        name: "t",
        dtype: "F32",
        shape: &[2],
        data: &[5u8; 8],
    }];
    let path = write_bytes(
        dir.path(),
        "sparse.safetensors",
        &container(&entries, None, &[]),
    );
    // A gigabyte of untouched file after the header: the header reader must not care.
    let gigabyte = 1u64 << 30;
    File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(gigabyte)
        .unwrap();
    let header = SafeTensorsHeader::read(&path).expect("a huge shard still has a small header");
    assert_eq!(header.file_len, gigabyte);
    assert_eq!(header.data_len(), gigabyte - header.data_start);
    let source = source(dir.path(), &[("t", "sparse.safetensors")]);
    assert_eq!(source.spec("t").unwrap().storage_bytes(), 8);
    assert_eq!(source.read("t").unwrap(), vec![5u8; 8]);
}

/// Feeds bytes to both parsers: every input must return, and never unwind.
fn sweep(blob: &[u8]) {
    let _ = SafeTensorsHeader::from_container(blob, blob.len() as u64, "sweep");
    let _ = SafeTensorsHeader::from_container(blob, u64::MAX, "sweep");
    let _ = SafeTensorsHeader::from_container(blob, 0, "sweep");
    let _ = CheckpointIndex::from_slice(blob, "sweep");
}

#[test]
fn malformed_input_never_panics() {
    let padding = format!(
        "{{\"t\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}},\"pad\":\"{}\"}}",
        "x".repeat(4096)
    );
    let deep = format!("{{\"deep\": {}}}", "[".repeat(256));
    let cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        vec![0u8; 7],
        vec![0u8; 8],
        u64::MAX.to_le_bytes().to_vec(),
        (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec(),
        container_with_header("{}", &[]),
        container_with_header("[]", &[]),
        container_with_header("null", &[]),
        container_with_header("{\"__metadata__\": null}", &[]),
        container_with_header(
            r#"{"t":{"dtype":"F32","shape":[18446744073709551615],"data_offsets":[0,0]}}"#,
            &[],
        ),
        container_with_header(
            r#"{"t":{"dtype":"I8","shape":[],"data_offsets":[0,0]}}"#,
            &[],
        ),
        container_with_header(
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#,
            &[],
        ),
        container_with_header(
            r#"{"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
            &[0u8; 4],
        ),
        container_with_header(
            r#"{"t":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}"#,
            &[0u8; 4],
        ),
        container_with_header(
            r#"{"t":{"dtype":"F32","shape":"1","data_offsets":[0,4]}}"#,
            &[0u8; 4],
        ),
        container_with_header(r#"{"t":null}"#, &[]),
        container_with_header(
            r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0u8; 4],
        ),
        container_with_header(&padding, &[0u8; 4]),
        container_with_header(
            r#"{"té":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0u8; 4],
        ),
        container_with_header(&deep, &[]),
        vec![b' '; 64],
    ];
    for case in &cases {
        sweep(case);
    }
    // Mutations of a valid container, plus every truncation of it.
    let valid = container(
        &[
            Entry {
                name: "a",
                dtype: "BF16",
                shape: &[2],
                data: &[0u8; 4],
            },
            Entry {
                name: "b",
                dtype: "I8",
                shape: &[4],
                data: &[0u8; 4],
            },
        ],
        Some(serde_json::json!({ "format": "pt" })),
        &[0u8; 4],
    );
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..512 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let mut copy = valid.clone();
        let at = (state as usize) % copy.len();
        copy[at] = (state >> 32) as u8;
        sweep(&copy);
        let cut = (state as usize) % (valid.len() + 1);
        sweep(&valid[..cut]);
    }
    let intact = SafeTensorsHeader::from_container(&valid, valid.len() as u64, "sweep");
    assert!(
        intact.is_ok(),
        "the unmutated container is valid: {intact:?}"
    );
    // And a container this reader rejects is a typed error, not a panic.
    expect_malformed(
        SafeTensorsHeader::from_container(&[], 0, "sweep"),
        "bytes are present",
    );
}

#[test]
fn an_index_that_is_not_a_safetensors_index_is_a_typed_error() {
    expect_malformed(
        CheckpointIndex::from_slice(b"not json", "index.json"),
        "not a safetensors index",
    );
    expect_malformed(
        CheckpointIndex::from_slice(br#"{"metadata":{}}"#, "index.json"),
        "not a safetensors index",
    );
    expect_malformed(
        CheckpointIndex::from_slice(br#"{"weight_map":[]}"#, "index.json"),
        "not a safetensors index",
    );
    expect_malformed(
        CheckpointIndex::from_slice(br#"{"weight_map":{"t":7}}"#, "index.json"),
        "not a safetensors index",
    );
    // A total_size that is not a non-negative integer is malformed, not silently ignored.
    for bad in [
        r#"{"weight_map":{},"metadata":{"total_size":"510"}}"#,
        r#"{"weight_map":{},"metadata":{"total_size":-1}}"#,
        r#"{"weight_map":{},"metadata":{"total_size":1.5}}"#,
        r#"{"weight_map":{},"metadata":{"total_size":null}}"#,
    ] {
        expect_malformed(
            CheckpointIndex::from_slice(bad.as_bytes(), "index.json"),
            "total_size",
        );
    }
    // A missing index file is Io, not a panic.
    let dir = tempfile::tempdir().unwrap();
    match CheckpointIndex::load(&dir.path().join("absent.json")) {
        Err(WeightError::Io { name, .. }) => assert_eq!(name, "absent.json"),
        other => panic!("expected Io, got {other:?}"),
    }
    match SafeTensorsSource::open(dir.path(), &dir.path().join("absent.json")) {
        Err(WeightError::Io { .. }) => {}
        other => panic!("expected Io, got {other:?}"),
    }
}

#[test]
fn the_index_reports_names_shards_and_the_declared_total() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_index(
        dir.path(),
        "model.safetensors.index.json",
        &[
            ("z.weight", "shard-2.safetensors"),
            ("a.weight", "shard-1.safetensors"),
            ("m.weight", "shard-1.safetensors"),
        ],
        Some(510_286_023_000),
    );
    let index = CheckpointIndex::load(&path).unwrap();
    assert_eq!(index.len(), 3);
    assert_eq!(
        index.names().collect::<Vec<_>>(),
        vec!["a.weight", "m.weight", "z.weight"],
        "names come out sorted"
    );
    assert_eq!(
        index.shards(),
        vec!["shard-1.safetensors", "shard-2.safetensors"]
    );
    assert_eq!(index.shard_of("m.weight").unwrap(), "shard-1.safetensors");
    assert_eq!(index.total_size(), Some(510_286_023_000));
    assert!(matches!(
        index.shard_of("absent"),
        Err(WeightError::Missing { .. })
    ));
    // An index that states no total reports none.
    let path = write_index(dir.path(), "plain.json", &[("a", "s.safetensors")], None);
    let index = CheckpointIndex::load(&path).unwrap();
    assert_eq!(index.total_size(), None);
    assert!(index.metadata.is_empty());
    assert_eq!(index.shards().len(), 1);
    assert_eq!(index.len(), 1);
}

#[test]
fn the_released_index_is_read_when_it_is_present() {
    let path = PathBuf::from("/home/user/dsv41-research/model.safetensors.index.json");
    if !path.is_file() {
        let line = b"SKIP: the released checkpoint index is not on this host\n";
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line);
        println!("SKIP: the released checkpoint index is not on this host");
        return;
    }
    let index = CheckpointIndex::load(&path).expect("the released index must parse");
    assert_eq!(
        index.len(),
        96_085,
        "the released checkpoint has 96,085 tensors"
    );
    assert_eq!(index.shards().len(), 48, "in 48 shards");
    assert_eq!(index.total_size(), Some(510_286_023_000));
    // The two Engram tables are indexed like everything else. That the engine refuses to
    // materialise them is the loader's policy, applied through the limit, not a gap in the index.
    for name in [
        "layers.1.engram.embed.weight",
        "layers.14.engram.embed.weight",
    ] {
        let shard = index.shard_of(name).expect("the Engram tables are indexed");
        assert!(
            shard.starts_with("model-") && shard.ends_with(".safetensors"),
            "{shard}"
        );
    }
    // Describing all 96,085 names needs no shard at all: this directory does not exist.
    let source = SafeTensorsSource::new("/nonexistent-shard-directory", index);
    let names = source.names();
    assert_eq!(names.len(), 96_085);
    assert!(
        names.windows(2).all(|pair| pair[0] <= pair[1]),
        "names are sorted"
    );
    assert!(source.contains("layers.1.engram.embed.weight"));
}

#[test]
fn a_header_that_names_one_tensor_twice_is_refused() {
    // A JSON object with a repeated key is legal JSON and has no map representation: parsing it into
    // one keeps the last entry, so the shard would silently be described by whichever descriptor came
    // second. Two descriptors for one name describe two different tensors, so the file is refused
    // rather than half-believed. The two here differ in dtype, shape and range, which is exactly the
    // kind of duplicate that parsing into a map would have hidden.
    let dir = tempfile::tempdir().unwrap();
    let json = concat!(
        r#"{"t":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"#,
        r#""t":{"dtype":"F32","shape":[2],"data_offsets":[2,10]}}"#
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(&[0u8; 10]);
    let path = write_bytes(dir.path(), "s.safetensors", &bytes);

    match SafeTensorsHeader::read(&path) {
        Err(WeightError::Malformed { name, detail }) => {
            assert_eq!(name, "t");
            assert!(detail.contains("twice"), "{detail}");
        }
        other => panic!("expected Malformed naming the repeated tensor, got {other:?}"),
    }
}

#[test]
fn an_index_that_names_one_tensor_twice_is_refused() {
    // The same defect one level up: the index is the authority on which shard holds a name, and a
    // repeated name would leave one shard unreachable while the reader believed it had them all.
    let json = r#"{"weight_map":{"t":"a.safetensors","t":"b.safetensors"}}"#;
    match CheckpointIndex::from_slice(json.as_bytes(), "index.json") {
        Err(WeightError::Malformed { name, detail }) => {
            assert_eq!(name, "index.json");
            assert!(detail.contains("more than once"), "{detail}");
        }
        other => panic!("expected Malformed naming the repeated tensor, got {other:?}"),
    }
}

#[test]
fn a_header_padded_to_eight_bytes_the_way_safetensors_writes_it_reads() {
    // Every released shard pads the JSON header with spaces to a multiple of eight bytes, because
    // the data section has to start aligned. The containers this test file writes never pad, so
    // without this case the reader would only ever have been exercised against the one layout no
    // real file uses. The padding is inside the header: the length field counts it.
    let dir = tempfile::tempdir().unwrap();
    let mut json = serde_json::to_vec(&serde_json::json!({
        "t": { "dtype": "BF16", "shape": [2u64], "data_offsets": [0u64, 4u64] }
    }))
    .expect("a JSON header");
    let unpadded = json.len();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    assert!(json.len() > unpadded, "the fixture must actually be padded");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&[0x00, 0x3F, 0x00, 0x40]);
    let path = write_bytes(dir.path(), "s.safetensors", &bytes);

    let header = SafeTensorsHeader::read(&path).expect("a padded header is the real layout");
    assert_eq!(header.header_len() as usize, json.len());
    assert_eq!(header.data_start, 8 + json.len() as u64);
    assert_eq!(header.data_len(), 4);
    let descriptor = header.tensor("t").expect("the tensor must be described");
    assert_eq!(descriptor.data_offsets, [0, 4]);
    assert_eq!(header.len(), 1);
    assert!(!header.is_empty());
    // The declared range is relative to the data section, so the padded header did not shift it.
    assert_eq!(
        &bytes[header.data_start as usize..header.data_start as usize + 2],
        &[0x00, 0x3F]
    );
}

#[test]
fn an_index_metadata_block_that_repeats_a_key_is_refused() {
    // The metadata block is the index's other name table, and the same visitor guards it. A repeated
    // total_size would otherwise be read as whichever copy happened to come last - here 1, where the
    // file also says 2 - and the byte total is the number the whole checkpoint-size argument rests on.
    let json = r#"{"weight_map":{},"metadata":{"total_size":1,"total_size":2}}"#;
    match CheckpointIndex::from_slice(json.as_bytes(), "index.json") {
        Err(WeightError::Malformed { name, detail }) => {
            assert_eq!(name, "index.json");
            assert!(detail.contains("more than once"), "{detail}");
            assert!(detail.contains("total_size"), "{detail}");
        }
        other => panic!("expected Malformed naming the repeated metadata key, got {other:?}"),
    }
}

#[test]
fn paged_ranges_return_exactly_the_requested_bytes_and_bypass_the_refusal_list() {
    let dir = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..32).collect();
    write_container(
        dir.path(),
        "a.safetensors",
        &[Entry {
            name: "a.weight",
            dtype: "F32",
            shape: &[8],
            data: &data,
        }],
    );
    let source = source(dir.path(), &[("a.weight", "a.safetensors")]).with_refused(["a.weight"]);
    // A whole read is refused by policy...
    assert!(matches!(
        source.read("a.weight"),
        Err(WeightError::Refused { .. })
    ));
    // ...while paging the same tensor is the alternative to that policy, and the ranges come back
    // in the order asked, byte for byte.
    let paged = source
        .read_ranges("a.weight", &[(8, 4), (0, 8), (28, 4)])
        .expect("the shard can serve the ranges");
    let mut expected = Vec::new();
    expected.extend_from_slice(&data[8..12]);
    expected.extend_from_slice(&data[0..8]);
    expected.extend_from_slice(&data[28..32]);
    assert_eq!(paged, expected);

    // A range that leaves the tensor is a typed refusal, not a short read.
    let error = source.read_ranges("a.weight", &[(30, 4)]).unwrap_err();
    match error {
        WeightError::Malformed { detail, .. } => {
            assert!(detail.contains("outside"), "{detail}");
        }
        other => panic!("expected Malformed for an out-of-range page, got {other:?}"),
    }
}

#[test]
fn paged_ranges_are_bounded_by_the_stored_bytes_not_the_logical_shape() {
    let dir = tempfile::tempdir().unwrap();
    // A packed expert stores two FP4 values per byte, so its logical last dimension (4) is twice
    // the stored one (2): eight stored bytes, not sixteen. Paging must bound itself by the eight
    // the header names, or a range read could run into the next tensor in the shard.
    let data: Vec<u8> = (0..8).collect();
    write_container(
        dir.path(),
        "a.safetensors",
        &[Entry {
            name: "layers.0.ffn.experts.0.w1.weight",
            dtype: "I8",
            shape: &[2, 4],
            data: &data,
        }],
    );
    let source = source(
        dir.path(),
        &[("layers.0.ffn.experts.0.w1.weight", "a.safetensors")],
    );
    let name = "layers.0.ffn.experts.0.w1.weight";
    assert_eq!(
        source
            .read_ranges(name, &[(0, 8)])
            .expect("the stored bytes"),
        data,
        "the whole stored tensor is pageable"
    );
    let error = source.read_ranges(name, &[(8, 1)]).unwrap_err();
    match error {
        WeightError::Malformed { detail, .. } => {
            assert!(detail.contains("outside"), "{detail}");
        }
        other => panic!("expected Malformed past the stored bytes, got {other:?}"),
    }
}

#[test]
fn paged_ranges_resolve_each_tensor_in_its_own_shard() {
    let dir = tempfile::tempdir().unwrap();
    let first: Vec<u8> = (0..8).collect();
    let second: Vec<u8> = (100..116).collect();
    write_container(
        dir.path(),
        "a.safetensors",
        &[Entry {
            name: "a.weight",
            dtype: "F32",
            shape: &[2],
            data: &first,
        }],
    );
    write_container(
        dir.path(),
        "b.safetensors",
        &[Entry {
            name: "b.weight",
            dtype: "F32",
            shape: &[4],
            data: &second,
        }],
    );
    let source = source(
        dir.path(),
        &[("a.weight", "a.safetensors"), ("b.weight", "b.safetensors")],
    );
    // Each page starts from that tensor's own data_offsets plus its own shard's data section,
    // which is what a reader that keeps one data section offset for the whole checkpoint gets
    // wrong on the second shard.
    assert_eq!(
        source.read_ranges("a.weight", &[(4, 4)]).unwrap(),
        first[4..8].to_vec()
    );
    assert_eq!(
        source.read_ranges("b.weight", &[(0, 4), (12, 4)]).unwrap(),
        [second[0..4].to_vec(), second[12..16].to_vec()].concat()
    );
}

#[test]
fn the_whole_tensor_shortcut_is_bounded_by_the_memory_cap() {
    // A paged read of the 98.3 GB Engram table asks for 24 rows: above the range-count threshold
    // and far above the memory cap, so it must take the per-range seek path. The alternative was a
    // 98 GB allocation on a 128 GB node and an OOM kill at the first Engram read (m1y15).
    assert!(!gathers_whole_tensor(
        98_300_000_000,
        24,
        GATHER_WHOLE_TENSOR_MAX_BYTES
    ));
    // The four-rank boot's many-range reads are column slices of ordinary resident tensors:
    // attention-row 20,992,000 bytes and experts-row 3,133,440 bytes per tensor, both under the
    // cap, so the boot keeps its whole-gather path and its measured 320-354 s rather than the
    // 416-521 s seek path.
    assert!(gathers_whole_tensor(
        20_992_000,
        2640,
        GATHER_WHOLE_TENSOR_MAX_BYTES
    ));
    assert!(gathers_whole_tensor(
        3_133_440,
        5120,
        GATHER_WHOLE_TENSOR_MAX_BYTES
    ));
    // The 1.32 GB embedding shards have one range each, and a 16-range read is at the count
    // threshold, so neither reaches the shortcut with or without the cap.
    assert!(!gathers_whole_tensor(
        1_323_827_200,
        1,
        GATHER_WHOLE_TENSOR_MAX_BYTES
    ));
    assert!(!gathers_whole_tensor(
        1_323_827_200,
        16,
        GATHER_WHOLE_TENSOR_MAX_BYTES
    ));
    // Two gigabytes over 24 ranges is the shape the cap exists for: per-range, not whole.
    assert!(!gathers_whole_tensor(
        2 << 30,
        24,
        GATHER_WHOLE_TENSOR_MAX_BYTES
    ));
}

#[test]
fn both_policy_branches_of_a_many_range_read_return_identical_bytes() {
    // Twenty-four ranges of a 64-byte tensor: above the range-count threshold, so the cap alone
    // decides the branch. The bytes must not depend on that decision, and the cap-0 form must read
    // exactly the same bytes one seek at a time.
    let dir = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..64).collect();
    write_container(
        dir.path(),
        "a.safetensors",
        &[Entry {
            name: "a.weight",
            dtype: "F32",
            shape: &[16],
            data: &data,
        }],
    );
    let source = source(dir.path(), &[("a.weight", "a.safetensors")]);
    let ranges: Vec<(u64, usize)> = (0..24u64).map(|i| (i * 2, 2)).collect();
    let mut expected = Vec::new();
    for &(offset, len) in &ranges {
        expected.extend_from_slice(&data[offset as usize..offset as usize + len]);
    }
    // cap u64::MAX takes the whole-tensor branch; cap 0 takes the per-range branch.
    assert!(gathers_whole_tensor(64, ranges.len(), u64::MAX));
    assert!(!gathers_whole_tensor(64, ranges.len(), 0));
    assert_eq!(
        source
            .read_ranges_with_policy("a.weight", &ranges, u64::MAX)
            .unwrap(),
        expected
    );
    assert_eq!(
        source
            .read_ranges_with_policy("a.weight", &ranges, 0)
            .unwrap(),
        expected
    );
    // The named paged entry point is the cap-0 form: never a whole-region buffer, same bytes.
    assert_eq!(
        source.read_ranges_paged("a.weight", &ranges).unwrap(),
        expected
    );
    // The trait form reaches the same override, so a caller holding a dyn WeightSource pages too.
    let trait_paged: &dyn WeightSource = &source;
    assert_eq!(
        trait_paged
            .read_ranges_paged("a.weight", &ranges)
            .expect("the source serves ranges")
            .unwrap(),
        expected
    );
}
