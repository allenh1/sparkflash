//! Serving a table's tensor byte ranges from its packed rows.
//!
//! The engine asks for an Engram table as two stored tensors: `layers.N.engram.embed.weight`
//! (one `width`-byte F8E4M3 row per table row) and `layers.N.engram.embed.scale` (one
//! `width.div_ceil(32)`-byte E8M0 row). It asks through
//! `sparkflash_engine::weights::WeightSource::read_ranges`: a list of `(offset, len)` byte
//! ranges per tensor, and it only ever asks for whole rows - the offset is `row * width` (or
//! `row * scale_width`) and the length is one row's bytes.
//!
//! The packed store keeps those two byte strings *interleaved* inside each 264-byte row
//! ([crate::layout::ENGRAM_ROW_BYTES]): 256 weight bytes then the 8 scale bytes of the same table
//! row. This module is the translation between the two shapes. It turns a range request into a
//! [crate::store::RowTable] row request, batches the rows the request names into one read per rank
//! file that owns some of them, and copies the requested slice out of every row it got back.
//! Nothing here keeps a table, or any row a later request would need: the only bytes allocated
//! are the caller's requested bytes plus the reader's own bounded scratch, so a serving process
//! can page a 98 GB table a chunk at a time.
//!
//! # What the translation is allowed to assume
//!
//! The engine and this reader must agree on the row geometry or the bytes would be decoded
//! wrongly, so a request that is not exactly whole rows is refused rather than guessed at:
//! `len == width` (or `len == width.div_ceil(32)` for the scale tensor) and
//! `offset % len == 0`. The weight slice has to fit in [crate::layout::ENGRAM_HEAD_DIM] bytes
//! and the scale slice in [crate::layout::ENGRAM_SCALE_BYTES]; the released configuration uses
//! exactly those two bounds, and a smaller width is accepted for a host fixture that does not
//! want to write the released geometry.
//!
//! # Why the scales need their own range request
//!
//! The engine reads the weight ranges and the scale ranges in two calls, so the rows behind them
//! are physically read twice (the second call finds them in the reader's page cache under the
//! released window sizes). This module deliberately does not carry a cache between the two calls:
//! the amortisation it would buy is a window measurement, and a cache here would hide exactly the
//! read count the host tests and the window are meant to observe.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, ensure};

use crate::layout::{ENGRAM_HEAD_DIM, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES, SCALE_BLOCK};
use crate::store::RowTable;

/// Which of a packed row's two stored tensors a range request addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngramTensor {
    /// The F8E4M3 weight half of each row, the first [ENGRAM_HEAD_DIM] bytes.
    Weight,
    /// The E8M0 scale half of each row, the last [ENGRAM_SCALE_BYTES] bytes.
    Scale,
}

impl EngramTensor {
    /// The bytes one row of this tensor occupies when a weight row is `width` bytes.
    #[must_use]
    pub fn row_bytes(self, width: usize) -> usize {
        match self {
            Self::Weight => width,
            Self::Scale => width.div_ceil(SCALE_BLOCK),
        }
    }

    /// The byte offset of this tensor inside a packed row.
    #[must_use]
    pub fn row_offset(self) -> usize {
        match self {
            Self::Weight => 0,
            Self::Scale => ENGRAM_HEAD_DIM,
        }
    }

    /// The name the engine uses for the tensor, for error messages.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Weight => "weight",
            Self::Scale => "scale",
        }
    }
}

/// What one range request returned: the requested bytes, in request order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedRead {
    /// The requested bytes, concatenated in the order the ranges were given.
    pub data: Vec<u8>,
    /// How many table rows the request named (after the whole-row checks).
    pub rows: usize,
    /// How many batched `RowTable` reads served them: one per rank file that owned a row.
    pub read_batches: usize,
}

/// A layer's packed tables, addressed by the engine's two stored tensors.
///
/// The type borrows the [`RowTable`] reader rather than owning it, so one opened store can serve
/// every request without being moved, and it holds no row data itself.
pub struct PagedTable<'a> {
    table: &'a dyn RowTable,
    layer_id: usize,
    width: usize,
    scale_width: usize,
}

impl<'a> PagedTable<'a> {
    /// Binds one Engram layer's rows to the tensor widths the engine requested them with.
    ///
    /// # Errors
    /// When a width is zero, when it does not fit the packed row layout, or when the reader
    /// holds no table for `layer_id`.
    pub fn new(
        table: &'a dyn RowTable,
        layer_id: usize,
        width: usize,
        scale_width: usize,
    ) -> Result<Self> {
        ensure!(width > 0, "the Engram weight width cannot be zero");
        ensure!(scale_width > 0, "the Engram scale width cannot be zero");
        ensure!(
            width <= ENGRAM_HEAD_DIM,
            "the Engram weight width {width} exceeds the packed row's {ENGRAM_HEAD_DIM} weight bytes"
        );
        ensure!(
            scale_width <= ENGRAM_SCALE_BYTES,
            "the Engram scale width {scale_width} exceeds the packed row's {ENGRAM_SCALE_BYTES} scale bytes"
        );
        ensure!(
            scale_width == width.div_ceil(SCALE_BLOCK),
            "the Engram scale width {scale_width} is not what {width} weight values need"
        );
        let mut found = false;
        for index in 0..table.table_count() {
            if let Some(spec) = table.spec(index)
                && spec.layer_id == layer_id
            {
                found = true;
                break;
            }
        }
        ensure!(
            found,
            "the packed store holds no table for Engram layer {layer_id}"
        );
        Ok(Self {
            table,
            layer_id,
            width,
            scale_width,
        })
    }

    /// Reads the whole-row ranges of one tensor the engine asked for.
    ///
    /// The returned bytes are the requested ranges concatenated in the order given, exactly as
    /// `WeightSource::read_ranges` promises, and nothing else was read from the table.
    ///
    /// # Errors
    /// When a range is not one whole row, when a named row is outside every rank file of the
    /// layer, when two rank files claim the same row, or when the underlying read fails.
    pub fn read(&self, tensor: EngramTensor, ranges: &[(u64, usize)]) -> Result<PagedRead> {
        let unit = match tensor {
            EngramTensor::Weight => self.width,
            EngramTensor::Scale => self.scale_width,
        };
        let base = tensor.row_offset();
        ensure!(
            base + unit <= ENGRAM_ROW_BYTES,
            "a {unit}-byte {} slice at row offset {base} does not fit a {ENGRAM_ROW_BYTES}-byte row",
            tensor.name()
        );

        let mut rows = Vec::with_capacity(ranges.len());
        for &(offset, len) in ranges {
            ensure!(
                len == unit,
                "a {} range of {len} bytes is not one {unit}-byte row",
                tensor.name()
            );
            ensure!(
                offset.is_multiple_of(unit as u64),
                "a {} range at byte {offset} does not start on a {unit}-byte row boundary",
                tensor.name()
            );
            rows.push(offset / unit as u64);
        }

        // Which rank file owns each global row: the packed store shards by row, so a request
        // can name rows in more than one file. The batches are per file, not per row.
        let mut by_table: BTreeMap<usize, (Vec<u32>, Vec<usize>)> = BTreeMap::new();
        for (slot, &global) in rows.iter().enumerate() {
            let mut owner = None;
            for index in 0..self.table.table_count() {
                let Some(spec) = self.table.spec(index) else {
                    continue;
                };
                if spec.layer_id != self.layer_id {
                    continue;
                }
                if let Some(local) = spec.local_row(global) {
                    ensure!(
                        owner.is_none(),
                        "global row {global} of layer {} is owned by more than one packed table",
                        self.layer_id
                    );
                    let local = u32::try_from(local).with_context(|| {
                        format!(
                            "the local row {local} of global row {global} does not fit a batch read"
                        )
                    })?;
                    owner = Some((index, local));
                }
            }
            let (index, local) = owner.ok_or_else(|| {
                anyhow::anyhow!(
                    "row {global} is outside every packed table of Engram layer {}",
                    self.layer_id
                )
            })?;
            by_table.entry(index).or_default().0.push(local);
            by_table.entry(index).or_default().1.push(slot);
        }

        let mut data = vec![0u8; ranges.len() * unit];
        let mut read_batches = 0usize;
        for (index, (locals, slots)) in by_table {
            let read = self
                .table
                .read_rows(index, &locals)
                .with_context(|| format!("reading Engram layer {}", self.layer_id))?;
            ensure!(
                read.len() == locals.len(),
                "the packed reader returned {} rows for {} requested",
                read.len(),
                locals.len()
            );
            for (slot, row) in slots.iter().zip(read) {
                data[slot * unit..(slot + 1) * unit].copy_from_slice(&row[base..base + unit]);
            }
            read_batches += 1;
        }
        Ok(PagedRead {
            data,
            rows: ranges.len(),
            read_batches,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::store::{MemoryTable, TableSpec, TailFill};

    use super::*;

    fn memory(row_start: u64, rows: u64) -> MemoryTable {
        let spec = TableSpec {
            layer_id: 1,
            layer_index: 0,
            rank: 0,
            world_size: 1,
            path: Path::new("/nonexistent/engram-l1-r0of1.bin").to_path_buf(),
            row_start,
            physical_rows: rows,
            logical_rows: row_start + rows,
            window_bytes: 512,
            tail_fill: TailFill::Reference,
        };
        let mut table = MemoryTable::new().with_layer(spec);
        table.fill_pattern(0).expect("fill the in-memory table");
        table
    }

    /// The pattern [MemoryTable::fill_pattern] writes into row `global` of layer 1.
    fn pattern(layer_id: usize, global: u64) -> [u8; ENGRAM_ROW_BYTES] {
        let mut bytes = [0u8; ENGRAM_ROW_BYTES];
        bytes[..8].copy_from_slice(&global.to_le_bytes());
        for (offset, byte) in bytes[8..].iter_mut().enumerate() {
            *byte = (global as u8)
                .wrapping_mul(31)
                .wrapping_add((offset as u8).wrapping_mul(7))
                .wrapping_add(layer_id as u8);
        }
        bytes
    }

    #[test]
    fn weight_ranges_come_back_in_request_order_and_nothing_else_is_read() {
        let table = memory(0, 64);
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        let ranges = [(5 * 8, 8), (0, 8), (3 * 8, 8)];
        let read = paged
            .read(EngramTensor::Weight, &ranges)
            .expect("read the rows");
        let mut expected = Vec::new();
        for global in [5u64, 0, 3] {
            expected.extend_from_slice(&pattern(1, global)[..8]);
        }
        assert_eq!(read.data, expected);
        assert_eq!(read.rows, 3);
        assert_eq!(read.read_batches, 1);
        // One row read per requested row, and no row outside the request.
        assert_eq!(table.read_ops(), 3);
    }

    #[test]
    fn scale_ranges_come_from_the_scale_half_of_the_same_row() {
        let table = memory(0, 64);
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        let read = paged
            .read(EngramTensor::Scale, &[(7, 1), (2, 1)])
            .expect("read the scales");
        let seven = pattern(1, 7);
        let two = pattern(1, 2);
        assert_eq!(
            read.data,
            vec![seven[ENGRAM_HEAD_DIM], two[ENGRAM_HEAD_DIM]]
        );
        assert_eq!(read.rows, 2);
        assert_eq!(read.read_batches, 1);
    }

    #[test]
    fn a_row_start_is_translated_to_a_local_row() {
        // The file holds global rows 100..164; global row 105 is local row 5.
        let table = memory(100, 64);
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        let read = paged
            .read(EngramTensor::Weight, &[(105 * 8, 8)])
            .expect("read a row");
        assert_eq!(read.data, pattern(1, 105)[..8].to_vec());
    }

    #[test]
    fn rows_owned_by_two_rank_files_are_read_in_one_batch_each() {
        let mut table = memory(0, 32);
        let second = memory(32, 32);
        table = table.with_layer(second.spec(0).expect("second spec").clone());
        table.fill_pattern(1).expect("fill the second table");
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        let read = paged
            .read(EngramTensor::Weight, &[(40 * 8, 8), (8, 8), (33 * 8, 8)])
            .expect("read across the split");
        let mut expected = Vec::new();
        for global in [40u64, 1, 33] {
            expected.extend_from_slice(&pattern(1, global)[..8]);
        }
        assert_eq!(read.data, expected);
        assert_eq!(read.read_batches, 2, "one batch per owning rank file");
    }

    #[test]
    fn a_global_row_outside_every_rank_file_is_refused() {
        let table = memory(0, 16);
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        let error = paged
            .read(EngramTensor::Weight, &[(16 * 8, 8)])
            .expect_err("row 16 is past the file");
        assert!(error.to_string().contains("outside every packed table"));
    }

    #[test]
    fn two_files_claiming_the_same_row_are_refused() {
        let first = memory(0, 16);
        let second = memory(0, 16);
        let mut table = first.with_layer(second.spec(0).expect("spec").clone());
        table.fill_pattern(1).expect("fill the second table");
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        let error = paged
            .read(EngramTensor::Weight, &[(3 * 8, 8)])
            .expect_err("two owners must not be guessed between");
        assert!(
            error
                .to_string()
                .contains("owned by more than one packed table")
        );
    }

    #[test]
    fn a_range_that_is_not_whole_rows_is_refused() {
        let table = memory(0, 64);
        let paged = PagedTable::new(&table, 1, 8, 1).expect("bind the layer");
        for ranges in [vec![(0u64, 7usize)], vec![(4, 8)], vec![(0, 16)], vec![]] {
            let weight = paged.read(EngramTensor::Weight, &ranges);
            if ranges.is_empty() {
                assert_eq!(weight.expect("an empty request is no request").rows, 0);
                continue;
            }
            assert!(weight.is_err(), "ranges {ranges:?} must be refused");
        }
        let scale = paged
            .read(EngramTensor::Scale, &[(0, 8)])
            .expect_err("a scale range is one byte here");
        assert!(scale.to_string().contains("not one 1-byte row"));
    }

    #[test]
    fn a_width_that_does_not_fit_the_packed_row_is_refused() {
        let table = memory(0, 64);
        assert!(PagedTable::new(&table, 1, ENGRAM_HEAD_DIM + 1, 8).is_err());
        assert!(PagedTable::new(&table, 1, 0, 0).is_err());
        assert!(PagedTable::new(&table, 1, 256, 9).is_err());
        assert!(PagedTable::new(&table, 1, 256, 7).is_err(), "256 needs 8");
    }

    #[test]
    fn a_layer_without_a_packed_table_is_refused() {
        let table = memory(0, 64);
        let error = PagedTable::new(&table, 14, 8, 1)
            .err()
            .expect("there is no layer 14 table");
        assert!(error.to_string().contains("no table for Engram layer 14"));
    }

    #[test]
    fn the_released_row_layout_is_the_prefix_plus_scale_layout_the_engine_names() {
        let table = memory(0, 4);
        let paged = PagedTable::new(&table, 1, ENGRAM_HEAD_DIM, ENGRAM_SCALE_BYTES)
            .expect("the released geometry is accepted");
        let weight = paged
            .read(
                EngramTensor::Weight,
                &[(2 * ENGRAM_HEAD_DIM as u64, ENGRAM_HEAD_DIM)],
            )
            .expect("read a released-width row");
        let scale = paged
            .read(
                EngramTensor::Scale,
                &[(2 * ENGRAM_SCALE_BYTES as u64, ENGRAM_SCALE_BYTES)],
            )
            .expect("read its scale");
        let row = pattern(1, 2);
        assert_eq!(weight.data, row[..ENGRAM_HEAD_DIM].to_vec());
        assert_eq!(scale.data, row[ENGRAM_HEAD_DIM..].to_vec());
    }
}
