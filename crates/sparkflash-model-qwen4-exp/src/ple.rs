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

//! The PLE n-gram table as a paged host store: DeepSeek's Engram serving shape.
//!
//! **Attribution:** The 16-head bigram/trigram Per-Layer Embedding (PLE) table architecture
//! originates from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
//! and its 128-shard FP8 E4M3 quantization from
//! [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
//! Out-of-core safetensors PLE shard paging with `POSIX_FADV_RANDOM` and parallel `pread` draws
//! on prior art from [`b12x`](https://github.com/local-inference-lab/b12x)
//! (`b12x/loader/_ple_reader.c`, Apache-2.0; Luke Alonso `@lukealonso`, `@original-el8`, and
//! contributors) and [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0).
//! See the root `NOTICE` file.
//!
//! The released table is 51.2 GiB of FP8 rows. It is an embedding table, not a matrix: a token
//! reads 16 rows of 160 bytes (2.5 KiB) and every other row is irrelevant to that token. Keeping
//! it on the device would cost 51.2 GiB of HBM for 25 KiB/s of unique traffic at 10 tokens/s, so
//! the device arm keeps it in host DRAM (or behind an O_DIRECT file) and pages rows the way
//! sparkflash-engram pages its 189 GiB table: a bounded row cache, whole-row reads batched per
//! shard, and a prefetch entry point the decode loop can call one step ahead.
//!
//! The checkpoint splits the table into 128 equal shards (split_ngram_parts). A global row maps
//! to shard row / shard_rows and local row row % shard_rows; because the shard count divides the
//! padded row count exactly, no row straddles two shards. Each shard is a separate safetensors
//! tensor (shard_N.weight), so a batched read is one call per shard that owns at least one row.
//!
//! The per-tensor FP8 scale is a single BF16 value; the caller passes it as f32, because the
//! table's scale is metadata, not a row.

use std::collections::HashMap;

use crate::config::Qwen4ExpConfig;

/// The largest number of rows the cache keeps by default.
///
/// 16,384 rows x 160 bytes = 2.5 MiB, which is a rounding error beside the 51.2 GiB table and
/// covers the working set of a long decode: the 16 heads of one position are unrelated rows, but
/// a repeated n-gram (the common case in generated code) hits the same 16 rows again.
pub const DEFAULT_CACHE_ROWS: usize = 16_384;

/// A source of whole table rows, batched by shard.
pub trait RowReader: Send + Sync {
    /// Read the byte ranges of one shard, concatenated in request order.
    ///
    /// A range is whole rows: offset = local_row * row_bytes, len = row_bytes.
    ///
    /// # Errors
    /// When the shard does not exist or the read fails.
    fn read_rows(&self, shard: usize, ranges: &[(u64, usize)]) -> Result<Vec<u8>, String>;

    /// Read multiple shards' byte ranges, returning (shard, bytes) in request order.
    ///
    /// Readers backed by NVMe files override this to issue shard preads concurrently across
    /// shards so a 16-head token miss completes in one NVMe round-trip instead of sixteen serial
    /// ones.
    ///
    /// # Errors
    /// When any shard read fails.
    fn read_shards(
        &self,
        requests: &[(usize, Vec<(u64, usize)>)],
    ) -> Result<Vec<(usize, Vec<u8>)>, String> {
        requests
            .iter()
            .map(|(shard, ranges)| Ok((*shard, self.read_rows(*shard, ranges)?)))
            .collect()
    }

    /// Rows in every shard (the shards are equal by construction).
    fn shard_rows(&self) -> u64;
}

/// What the paged table has served, in the shape the serving telemetry uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PleCacheCounters {
    /// Rows answered from the cache.
    pub hits: u64,
    /// Rows fetched from the reader.
    pub misses: u64,
    /// Batched reader calls issued.
    pub reads: u64,
    /// Bytes fetched from the reader.
    pub bytes: u64,
}

/// A paged PLE table with a bounded row cache.
pub struct PleTable<R: RowReader> {
    reader: R,
    shard_rows: u64,
    shards: usize,
    row_bytes: usize,
    heads: usize,
    scale: f32,
    capacity: usize,
    cache: HashMap<u64, Vec<u8>>,
    counters: PleCacheCounters,
}

impl<R: RowReader + std::fmt::Debug> std::fmt::Debug for PleTable<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PleTable")
            .field("shard_rows", &self.shard_rows)
            .field("shards", &self.shards)
            .field("row_bytes", &self.row_bytes)
            .field("capacity", &self.capacity)
            .field("counters", &self.counters)
            .finish()
    }
}

impl<R: RowReader> PleTable<R> {
    /// Build a table over an open reader.
    ///
    /// # Errors
    /// When the configuration's PLE layout is not divisible into its shards.
    pub fn new(config: &Qwen4ExpConfig, reader: R, scale: f32) -> Result<Self, String> {
        let layout = crate::ngram::PleLayout::new(config);
        let shards = config.split_ngram_parts;
        if !layout.padded_rows.is_multiple_of(shards as u64) {
            return Err(format!(
                "the PLE table has {} rows, which does not divide into {shards} shards",
                layout.padded_rows
            ));
        }
        let shard_rows = reader.shard_rows();
        if shard_rows * shards as u64 != layout.padded_rows {
            return Err(format!(
                "the reader holds {shard_rows} rows per shard, but {shards} shards of the released table are {} each",
                layout.padded_rows / shards as u64
            ));
        }
        Ok(Self {
            reader,
            shard_rows,
            shards,
            row_bytes: config.ngram_head_dim(),
            heads: config.ngram_heads(),
            scale,
            capacity: DEFAULT_CACHE_ROWS,
            cache: HashMap::new(),
            counters: PleCacheCounters::default(),
        })
    }

    /// Bound the row cache to this many rows.
    #[must_use]
    pub fn with_capacity(mut self, rows: usize) -> Self {
        self.capacity = rows.max(1);
        self.trim();
        self
    }

    /// The cache counters so far.
    #[must_use]
    pub fn counters(&self) -> PleCacheCounters {
        self.counters
    }

    /// The FP8 per-tensor scale this table multiplies rows by.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The concatenated PLE embedding for one position's head rows, in head order.
    ///
    /// # Errors
    /// When a head count or row is outside the table, or the reader fails.
    pub fn embedding(&mut self, rows: &[u64]) -> Result<Vec<f32>, String> {
        self.check_heads(rows)?;
        let missing = self.missing(rows);
        self.fetch(&missing)?;
        let out = self.decode(rows)?;
        self.trim();
        Ok(out)
    }

    /// [Self::embedding] for many positions at once, handed to `each` in order.
    ///
    /// Every row the positions miss is read in one pass before any is decoded, and the cache is
    /// trimmed once, after the last: a prefill chunk names far more rows than the cache keeps
    /// (4,096 tokens are 65,536), and trimming between the read and the decode would evict most of
    /// them and read each again, one position's sixteen at a time.
    ///
    /// # Errors
    /// As embedding, or what `each` returns.
    pub fn embeddings<P: AsRef<[u64]>>(
        &mut self,
        positions: &[P],
        mut each: impl FnMut(usize, &[f32]) -> Result<(), String>,
    ) -> Result<(), String> {
        for rows in positions {
            self.check_heads(rows.as_ref())?;
        }
        let flat: Vec<u64> = positions
            .iter()
            .flat_map(|rows| rows.as_ref().iter().copied())
            .collect();
        let missing = self.missing(&flat);
        self.fetch(&missing)?;
        for (index, rows) in positions.iter().enumerate() {
            each(index, &self.decode(rows.as_ref())?)?;
        }
        self.trim();
        Ok(())
    }

    fn check_heads(&self, rows: &[u64]) -> Result<(), String> {
        if rows.len() == self.heads {
            Ok(())
        } else {
            Err(format!(
                "the PLE expects {} head rows, got {}",
                self.heads,
                rows.len()
            ))
        }
    }

    /// One position's rows, all already in the cache, decoded in head order.
    fn decode(&self, rows: &[u64]) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(self.heads * self.row_bytes);
        for row in rows {
            let bytes = self
                .cache
                .get(row)
                .ok_or_else(|| format!("the PLE row {row} was not fetched"))?;
            let decoded =
                sparkflash_weights::nvfp4::dequantize_fp8(bytes, self.scale, self.row_bytes)
                    .map_err(|error| error.to_string())?;
            out.extend(decoded);
        }
        Ok(out)
    }

    /// Read rows into the cache without decoding them, so a later embedding call hits.
    ///
    /// # Errors
    /// As embedding.
    pub fn prefetch(&mut self, rows: &[u64]) -> Result<(), String> {
        let missing = self.missing(rows);
        self.fetch(&missing)?;
        self.trim();
        Ok(())
    }

    fn missing(&self, rows: &[u64]) -> Vec<u64> {
        let mut missing: Vec<u64> = rows
            .iter()
            .copied()
            .filter(|row| !self.cache.contains_key(row))
            .collect();
        missing.sort_unstable();
        missing.dedup();
        missing
    }

    fn fetch(&mut self, rows: &[u64]) -> Result<(), String> {
        if rows.is_empty() {
            return Ok(());
        }
        let total_rows = self.shard_rows * self.shards as u64;
        for row in rows {
            if *row >= total_rows {
                return Err(format!(
                    "the PLE row {row} is outside the {total_rows}-row table"
                ));
            }
        }
        let mut by_shard: HashMap<usize, Vec<u64>> = HashMap::new();
        for row in rows {
            let shard = (row / self.shard_rows) as usize;
            by_shard.entry(shard).or_default().push(*row);
        }
        let mut shards: Vec<usize> = by_shard.keys().copied().collect();
        shards.sort_unstable();
        let requests: Vec<(usize, Vec<(u64, usize)>)> = shards
            .iter()
            .map(|&shard| {
                let ranges = by_shard[&shard]
                    .iter()
                    .map(|row| {
                        (
                            (row % self.shard_rows) * self.row_bytes as u64,
                            self.row_bytes,
                        )
                    })
                    .collect();
                (shard, ranges)
            })
            .collect();
        let results = self.reader.read_shards(&requests)?;
        for ((shard, ranges), (_, bytes)) in requests.iter().zip(results) {
            let expected = ranges.len() * self.row_bytes;
            if bytes.len() != expected {
                return Err(format!(
                    "the PLE shard {shard} returned {} bytes for {expected} requested",
                    bytes.len()
                ));
            }
            self.counters.reads += 1;
            self.counters.bytes += bytes.len() as u64;
            self.counters.misses += ranges.len() as u64;
            for (index, row) in by_shard[shard].iter().enumerate() {
                let start = index * self.row_bytes;
                self.cache
                    .insert(*row, bytes[start..start + self.row_bytes].to_vec());
            }
        }
        Ok(())
    }

    fn trim(&mut self) {
        if self.cache.len() <= self.capacity {
            return;
        }
        // A plain bounded cache: drop arbitrary entries rather than pay for an LRU list on a
        // path that reads 2.5 KiB per token. The Engram store's 16-way set-associative cache is
        // the production shape; this keeps the same bound with less machinery.
        let excess = self.cache.len() - self.capacity;
        let victims: Vec<u64> = self.cache.keys().copied().take(excess).collect();
        for victim in victims {
            self.cache.remove(&victim);
        }
    }
}

/// The PLE table name prefix for one layer.
#[must_use]
pub fn shard_tensor_name(layer: usize, shard: usize) -> String {
    format!(
        "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader whose row content encodes (shard, local row) so a wrong mapping is visible.
    struct PatternReader {
        shard_rows: u64,
        row_bytes: usize,
    }

    impl RowReader for PatternReader {
        fn read_rows(&self, shard: usize, ranges: &[(u64, usize)]) -> Result<Vec<u8>, String> {
            let mut out = Vec::new();
            for (offset, len) in ranges {
                let local = offset / self.row_bytes as u64;
                assert_eq!(*len, self.row_bytes);
                for index in 0..*len {
                    out.push(((shard as u64 + local + index as u64) % 251) as u8);
                }
            }
            Ok(out)
        }

        fn shard_rows(&self) -> u64 {
            self.shard_rows
        }
    }

    fn tiny_table(capacity: usize) -> PleTable<PatternReader> {
        let config = Qwen4ExpConfig::tiny();
        let reader = PatternReader {
            shard_rows: config.ple_table_rows() / config.split_ngram_parts as u64,
            row_bytes: config.ngram_head_dim(),
        };
        PleTable::new(&config, reader, 1.0)
            .expect("the tiny table divides into shards")
            .with_capacity(capacity)
    }

    #[test]
    fn a_second_lookup_hits_the_cache() {
        let config = Qwen4ExpConfig::tiny();
        let mut table = tiny_table(64);
        let rows: Vec<u64> = (0..config.ngram_heads() as u64).collect();
        table.embedding(&rows).expect("the rows are in the table");
        let after_first = table.counters();
        assert_eq!(after_first.misses, rows.len() as u64);
        table
            .embedding(&rows)
            .expect("the rows are still in the table");
        let after_second = table.counters();
        assert_eq!(after_second.misses, after_first.misses);
        assert_eq!(after_second.reads, after_first.reads);
    }

    #[test]
    fn a_row_maps_to_the_shard_that_owns_it() {
        let config = Qwen4ExpConfig::tiny();
        let mut table = tiny_table(64);
        let shard_rows = config.ple_table_rows() / config.split_ngram_parts as u64;
        let rows: Vec<u64> = (0..config.ngram_heads() as u64)
            .map(|head| shard_rows + head)
            .collect();
        table.embedding(&rows).expect("shard 1 holds these rows");
        assert!(table.counters().reads >= 1);
    }

    #[test]
    fn an_out_of_range_row_is_refused() {
        let config = Qwen4ExpConfig::tiny();
        let mut table = tiny_table(64);
        let mut rows: Vec<u64> = (0..config.ngram_heads() as u64).collect();
        rows[0] = config.ple_table_rows();
        let error = table.embedding(&rows).expect_err("row is past the table");
        assert!(error.contains("outside"));
    }

    #[test]
    fn the_cache_stays_bounded() {
        let config = Qwen4ExpConfig::tiny();
        let mut table = tiny_table(4);
        let shard_rows = config.ple_table_rows() / config.split_ngram_parts as u64;
        for base in (0..64).step_by(config.ngram_heads()) {
            let rows: Vec<u64> = (0..config.ngram_heads() as u64)
                .map(|head| (base as u64 + head) % shard_rows)
                .collect();
            table.embedding(&rows).expect("rows are in range");
            assert!(table.cache.len() <= 4, "the cache grew past its bound");
        }
    }

    #[test]
    fn a_batch_larger_than_the_cache_reads_each_row_once() {
        let config = Qwen4ExpConfig::tiny();
        let heads = config.ngram_heads();
        let mut table = tiny_table(4);
        let positions: Vec<Vec<u64>> = (0..8u64)
            .map(|position| (0..heads as u64).map(|head| position * 97 + head).collect())
            .collect();
        let mut seen = Vec::new();
        table
            .embeddings(&positions, |index, embedding| {
                seen.push((index, embedding.to_vec()));
                Ok(())
            })
            .expect("rows are in range");
        assert_eq!(table.counters().misses, (positions.len() * heads) as u64);
        assert!(table.cache.len() <= 4, "the cache grew past its bound");
        let mut single = tiny_table(64);
        for (index, rows) in positions.iter().enumerate() {
            assert_eq!(seen[index].0, index);
            assert_eq!(
                seen[index].1,
                single.embedding(rows).expect("rows are in range")
            );
        }
    }

    #[test]
    fn the_embedding_is_the_concatenated_heads() {
        let config = Qwen4ExpConfig::tiny();
        let mut table = tiny_table(64);
        let rows: Vec<u64> = (0..config.ngram_heads() as u64).collect();
        let embedding = table.embedding(&rows).expect("rows are in range");
        assert_eq!(embedding.len(), config.ple_embed_dim);
    }
}
