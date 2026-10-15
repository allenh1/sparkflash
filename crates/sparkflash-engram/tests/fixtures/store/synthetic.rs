//! Shared synthetic fixtures for the Engram store integration tests.
//!
//! Included by the `engram_store_*.rs` binaries through
//! `#[path = "fixtures/store/synthetic.rs"] mod synthetic;`. Nothing here touches model
//! weights, a cluster or a real NVMe device: every file is a few kilobytes written into a
//! temporary directory.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use sparkflash_engram::layout::ENGRAM_ROW_BYTES;
use sparkflash_engram::packer::align_up;
use sparkflash_engram::store::{MemoryTable, TableSpec, TailFill};

/// Deterministic row content for a synthetic table.
///
/// The first eight bytes hold the little-endian global row index and the last eight mix in
/// the inverse of that index, so a mis-sliced, shifted or stale read cannot pass by
/// accident.
#[must_use]
pub fn row_for(layer_id: usize, row: u64) -> [u8; ENGRAM_ROW_BYTES] {
    let mut bytes = [0u8; ENGRAM_ROW_BYTES];
    bytes[..8].copy_from_slice(&row.to_le_bytes());
    for (offset, byte) in bytes[8..ENGRAM_ROW_BYTES - 8].iter_mut().enumerate() {
        *byte = (row as u8)
            .wrapping_mul(31)
            .wrapping_add((offset as u8).wrapping_mul(7))
            .wrapping_add(layer_id as u8);
    }
    bytes[ENGRAM_ROW_BYTES - 8..].copy_from_slice(&(row ^ 0x5A5A_5A5A_5A5A_5A5A).to_le_bytes());
    bytes
}

/// Writes a dense rank file exactly like the packer does: `rows * 264` logical bytes and
/// zero padding up to a 512-byte multiple. Returns the file length.
pub fn write_dense_table(path: &Path, layer_id: usize, first_row: u64, rows: u64) -> u64 {
    let mut bytes = Vec::with_capacity(rows as usize * ENGRAM_ROW_BYTES);
    for row in 0..rows {
        bytes.extend_from_slice(&row_for(layer_id, first_row + row));
    }
    let file_bytes = align_up(bytes.len() as u64);
    bytes.resize(file_bytes as usize, 0);
    fs::write(path, &bytes).expect("write synthetic dense table");
    file_bytes
}

/// Writes a file of exactly `logical_bytes` with no alignment padding, to prove the reader
/// refuses a file the packer would never have written.
pub fn write_unaligned_table(path: &Path, layer_id: usize, rows: u64) -> u64 {
    let mut bytes = Vec::with_capacity(rows as usize * ENGRAM_ROW_BYTES);
    for row in 0..rows {
        bytes.extend_from_slice(&row_for(layer_id, row));
    }
    let written = bytes.len() as u64;
    fs::write(path, &bytes).expect("write unaligned table");
    written
}

/// Spec for one synthetic rank file.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn spec(
    layer_id: usize,
    layer_index: usize,
    rank: usize,
    world_size: usize,
    path: &Path,
    row_start: u64,
    rows: u64,
    logical_rows: u64,
    window_bytes: u64,
) -> TableSpec {
    TableSpec {
        layer_id,
        layer_index,
        rank,
        world_size,
        path: PathBuf::from(path),
        row_start,
        physical_rows: rows,
        logical_rows,
        window_bytes,
        tail_fill: TailFill::Reference,
    }
}

/// A safetensors index whose weight and scale tensors all live in `shard`.
#[must_use]
pub fn index_json(shard: &str, layers: &[usize]) -> String {
    let mut entries = Vec::new();
    for layer in layers {
        for kind in ["weight", "scale"] {
            entries.push(format!(
                "\"layers.{layer}.engram.embed.{kind}\":\"{shard}\""
            ));
        }
    }
    format!(
        "{{\"metadata\":{{\"total_size\":0}},\"weight_map\":{{{}}}}}",
        entries.join(",")
    )
}

/// A safetensors index that maps each entry of `map` to its shard.
#[must_use]
pub fn index_json_from(map: &[(&str, &str)]) -> String {
    let entries: Vec<String> = map
        .iter()
        .map(|(name, shard)| format!("\"{name}\":\"{shard}\""))
        .collect();
    format!(
        "{{\"metadata\":{{\"total_size\":0}},\"weight_map\":{{{}}}}}",
        entries.join(",")
    )
}

/// A patterned in-memory table for tests that must not touch the filesystem at all.
#[must_use]
pub fn memory_layer(rows: u64, window_bytes: u64) -> MemoryTable {
    let mut table = MemoryTable::new().with_layer(spec(
        1,
        0,
        0,
        1,
        Path::new("/nonexistent/engram-l1-r0of1.bin"),
        0,
        rows,
        rows,
        window_bytes,
    ));
    table.fill_pattern(0).expect("fill the in-memory table");
    table
}

/// A manifest JSON shaped exactly like `packer::PackManifest`.
///
/// `rows` and `row_start` must match the compiled-in geometry for
/// `store::specs_from_manifest` to accept it; the source tensor shapes carry the full table
/// row count. `sha256` is only metadata here, so a placeholder is fine.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn manifest_json(
    layer_id: usize,
    layer_index: usize,
    rank: usize,
    world_size: usize,
    file: &str,
    row_start: u64,
    rows: u64,
    source_rows: u64,
    table_rows: u64,
    windowless_data_bytes: u64,
    file_bytes: u64,
    source_shard: &str,
) -> String {
    let shard = serde_json::json!({
        "format": "sparkflash-engram",
        "format_version": 1,
        "generated_by": "integration-test",
        "layout": {
            "row_bytes": 264u64,
            "weight_bytes": 256u64,
            "scale_bytes": 8u64,
            "scale_block": 32u64,
            "dtype_weight": "F8_E4M3",
            "dtype_scale": "F8_E8M0",
            "file_alignment": 512u64,
            "dense": true
        },
        "world_size": world_size,
        "layers": [layer_id],
        "index": "model.safetensors.index.json",
        "index_sha256": null,
        "tokenizer": null,
        "tokenizer_sha256": null,
        "total_rows": rows,
        "total_data_bytes": windowless_data_bytes,
        "total_file_bytes": file_bytes,
        "shards": [{
            "file": file,
            "layer": layer_id,
            "rank": rank,
            "world_size": world_size,
            "row_start": row_start,
            "row_end": row_start + source_rows,
            "rows": rows,
            "padded_rows": rows.saturating_sub(source_rows),
            "row_stride": 264u64,
            "dtype": "F8_E4M3",
            "data_bytes": rows * 264,
            "file_bytes": file_bytes,
            "padding_bytes": file_bytes.saturating_sub(rows * 264),
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "source_shard": source_shard
        }],
        "sources": [{
            "layer": layer_id,
            "layer_index": layer_index,
            "shard": source_shard,
            "weight": {
                "name": format!("layers.{layer_id}.engram.embed.weight"),
                "dtype": "F8_E4M3",
                "shape": [table_rows, 256]
            },
            "scale": {
                "name": format!("layers.{layer_id}.engram.embed.scale"),
                "dtype": "F8_E8M0",
                "shape": [table_rows, 8]
            },
            "weight_data_offset": 0u64,
            "scale_data_offset": table_rows * 256,
            "files": [file]
        }]
    });
    shard.to_string()
}
