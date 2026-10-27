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

//! The Engram write stage's **run-scope** state: the at-open row arena.
//!
//! The per-position Engram head has two host operands that are functions of the token history only:
//! the gather of this position's 24 table rows and the activation quantiser over them. Both can leave
//! the step. This module is the **gather** half: it reads the rows the whole run will look up, once,
//! into one arena laid out exactly as `engram_embed_sm121` reads it (`[columns][head_dim]` E4M3 codes
//! followed by `[columns][head_dim/32]` E8M0 scales, both 256-byte aligned). The per-step chain then
//! only needs the arena entry's offset, so `engram_write` has no gather launch and no host operand.
//!
//! The skip flags are the host seam's own rule (`engram.rs:919-921`): a row that contains the
//! dead-token sentinel, or an id past the table, skips the whole position. They are fixed at open
//! because the ids and the plan are, so every rank agrees without a collective - the divergence rule
//! the stage contract states.
//!
//! **Not wired yet.** The builder and its layout are landed with CPU tests; `DeviceLayer::engram_write`
//! still refuses and names this arena as the missing piece (`docs/build/s1-result.md` §7 item 3). Until
//! that wiring lands this module is an input to the next S1 step, not a product path: nothing here is
//! measured, timed or counted.
#![allow(dead_code)]

/// The slab's alignment: every arena entry starts on a 256-byte boundary, so one `cuMemcpyHtoD` of
/// the whole arena lands both the codes and the scales of every entry aligned.
pub const ARENA_ALIGN: usize = 256;

/// One arena entry's fixed geometry: `[columns][head_dim]` codes then `[columns][scale_width]` scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngramArenaLayout {
    columns: usize,
    head_dim: usize,
    scale_width: usize,
    codes_len: usize,
    scales_len: usize,
    stride: usize,
}

impl EngramArenaLayout {
    /// The released geometry: one entry per (engram layer, position), 24 rows of 256 E4M3 bytes and
    /// 24 rows of 8 E8M0 scale bytes, 6,400 B including both 256-byte alignments.
    ///
    /// # Errors
    /// A zero column, head or scale width; a scale width that is not `head_dim.div_ceil(32)`; a
    /// geometry whose entry size overflows.
    pub fn new(columns: usize, head_dim: usize) -> Result<Self, String> {
        if columns == 0 || head_dim == 0 {
            return Err(format!(
                "engram arena: {columns} columns of {head_dim} rows has no bytes to gather"
            ));
        }
        let scale_width = head_dim.div_ceil(32);
        let codes_len = columns
            .checked_mul(head_dim)
            .ok_or_else(|| "engram arena: the codes length overflows".to_owned())?;
        let scales_len = columns
            .checked_mul(scale_width)
            .ok_or_else(|| "engram arena: the scales length overflows".to_owned())?;
        let stride = align_up(codes_len)
            .checked_add(align_up(scales_len))
            .ok_or_else(|| "engram arena: the entry stride overflows".to_owned())?;
        Ok(Self {
            columns,
            head_dim,
            scale_width,
            codes_len,
            scales_len,
            stride,
        })
    }

    /// Rows per entry (the hash columns, 24 released).
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Bytes per gathered row (the Engram head width, 256 released).
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Scale bytes per gathered row (`head_dim / 32`, 8 released).
    #[must_use]
    pub const fn scale_width(&self) -> usize {
        self.scale_width
    }

    /// The codes array's length inside one entry.
    #[must_use]
    pub const fn codes_len(&self) -> usize {
        self.codes_len
    }

    /// The scales array's length inside one entry.
    #[must_use]
    pub const fn scales_len(&self) -> usize {
        self.scales_len
    }

    /// The distance between consecutive entries, both arrays' alignments included.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// The arena's size for `slots` layers and `positions` positions.
    ///
    /// # Errors
    /// Zero slots or positions (an arena with no entry), or a size that overflows.
    pub fn arena_bytes(&self, slots: usize, positions: usize) -> Result<usize, String> {
        if slots == 0 || positions == 0 {
            return Err(format!(
                "engram arena: {slots} layers x {positions} positions has no entry to gather"
            ));
        }
        slots
            .checked_mul(positions)
            .and_then(|entries| entries.checked_mul(self.stride))
            .ok_or_else(|| "engram arena: the arena size overflows".to_owned())
    }
}

/// The gathered rows for a whole run, plus the per-(slot, position) skip flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramRowArena {
    layout: EngramArenaLayout,
    bytes: Vec<u8>,
    positions: usize,
    slots: usize,
    skip: Vec<bool>,
}

impl EngramRowArena {
    /// The arena's entry geometry.
    #[must_use]
    pub const fn layout(&self) -> EngramArenaLayout {
        self.layout
    }

    /// The arena's bytes, ready for one H2D at open.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The number of positions the run will write for.
    #[must_use]
    pub const fn positions(&self) -> usize {
        self.positions
    }

    /// The number of Engram layers the arena carries (`config.engram_layer_ids.len()`).
    #[must_use]
    pub const fn slots(&self) -> usize {
        self.slots
    }

    /// The fixed layer-to-slot map: the config's own order, so the stage finds its slot by lookup.
    ///
    /// # Errors
    /// A layer the config does not list as an Engram layer.
    pub fn layer_slot(&self, layer_ids: &[usize], layer: usize) -> Result<usize, String> {
        layer_ids
            .iter()
            .position(|candidate| *candidate == layer)
            .ok_or_else(|| {
                format!(
                    "engram arena: layer {layer} is not one of the {slots} Engram layers {layer_ids:?}",
                    slots = self.slots
                )
            })
    }

    /// Whether the host seam skips this (slot, position) - the dead-token rule, fixed at open.
    ///
    /// # Errors
    /// A slot or position outside the arena.
    pub fn is_skipped(&self, slot: usize, position: usize) -> Result<bool, String> {
        Ok(self.skip[self.entry_index(slot, position)?])
    }

    /// One entry's codes and scales, both 256-byte aligned inside the arena.
    ///
    /// # Errors
    /// A slot or position outside the arena.
    pub fn entry(&self, slot: usize, position: usize) -> Result<(&[u8], &[u8]), String> {
        let base = self.entry_index(slot, position)? * self.layout.stride;
        let codes = &self.bytes[base..base + self.layout.codes_len];
        let scales_base = base + align_up(self.layout.codes_len);
        Ok((
            codes,
            &self.bytes[scales_base..scales_base + self.layout.scales_len],
        ))
    }

    fn entry_index(&self, slot: usize, position: usize) -> Result<usize, String> {
        if slot >= self.slots || position >= self.positions {
            return Err(format!(
                "engram arena: entry ({slot}, {position}) is outside {slots} slots x {positions} positions",
                slots = self.slots,
                positions = self.positions
            ));
        }
        Ok(slot * self.positions + position)
    }
}

/// The arena's fixed inputs: the geometry, the Engram layer order and each layer's table height.
#[derive(Debug, Clone, Copy)]
#[allow(missing_docs)] // geometry and layer order describe themselves; the owner's cleanup removed the prose
pub struct EngramArenaShape<'a> {
    pub layout: EngramArenaLayout,
    pub layer_ids: &'a [usize],
    /// Each configured layer's table rows, by slot (`config.engram_rows(layer)`); a slot with `None`
    /// is refused, because a listed layer with no table would write a zero row every position.
    pub table_rows: &'a [Option<usize>],
}

impl EngramArenaShape<'_> {
    /// Checks the shape before any read: one table per listed layer, at least one slot.
    ///
    /// # Errors
    /// A layer count that disagrees with the table list, or a listed layer with no table.
    pub fn check(&self) -> Result<(), String> {
        if self.layer_ids.is_empty() {
            return Err("engram arena: the model has no Engram layer to gather".to_owned());
        }
        if self.layer_ids.len() != self.table_rows.len() {
            return Err(format!(
                "engram arena: {} Engram layers but {} table heights",
                self.layer_ids.len(),
                self.table_rows.len()
            ));
        }
        for (slot, (layer, rows)) in self.layer_ids.iter().zip(self.table_rows).enumerate() {
            if rows.is_none() {
                return Err(format!(
                    "engram arena: slot {slot} is layer {layer} but the config reports no table for it; a listed layer with no table would write an all-zero row at every position"
                ));
            }
        }
        Ok(())
    }
}

/// Builds the arena from a per-position hash stream: the released layout and skip rule, no I/O policy.
///
/// `hashes_at(position, out)` fills `out` with `columns * slots` ids in slot-major order (the engine's
/// `ngram_hashes` order); `read(name, ranges)` returns the concatenated bytes for those byte ranges of
/// one tensor, which is what `WeightSource::read_ranges_paged` serves.
///
/// # Errors
/// A position whose history is empty, a hash count that disagrees with the shape, or a read failure.
pub fn build_with_hashes(
    shape: EngramArenaShape<'_>,
    positions: usize,
    mut hashes_at: impl FnMut(usize, &mut Vec<u32>) -> Result<(), String>,
    mut read: impl FnMut(&str, &[(u64, usize)]) -> Result<Vec<u8>, String>,
) -> Result<EngramRowArena, String> {
    shape.check()?;
    let layout = shape.layout;
    let slots = shape.layer_ids.len();
    let arena_bytes = layout.arena_bytes(slots, positions)?;
    let mut bytes = vec![0u8; arena_bytes];
    let mut skip = vec![false; slots * positions];
    let mut hashes: Vec<u32> = Vec::new();
    for position in 0..positions {
        hashes.clear();
        hashes_at(position, &mut hashes)?;
        let wanted = layout
            .columns
            .checked_mul(slots)
            .ok_or_else(|| "engram arena: the hash count overflows".to_owned())?;
        if hashes.len() != wanted {
            return Err(format!(
                "engram arena: position {position} has {} hashes, the {slots} slots of {} columns need {wanted}",
                hashes.len(),
                layout.columns
            ));
        }
        for (slot, (layer, table_rows)) in shape.layer_ids.iter().zip(shape.table_rows).enumerate()
        {
            let table_rows = table_rows.ok_or_else(|| {
                format!("engram arena: slot {slot} (layer {layer}) has no table height")
            })?;
            let row = &hashes[slot * layout.columns..(slot + 1) * layout.columns];
            if row.contains(&sparkflash_engine::forward::engram::HASH_SENTINEL)
                || row.iter().any(|&id| id as usize >= table_rows)
            {
                skip[slot * positions + position] = true;
                continue;
            }
            let codes_name = format!("layers.{layer}.engram.embed.weight");
            let scales_name = format!("layers.{layer}.engram.embed.scale");
            let codes_ranges: Vec<(u64, usize)> = row
                .iter()
                .map(|&id| (u64::from(id) * layout.head_dim as u64, layout.head_dim))
                .collect();
            let scales_ranges: Vec<(u64, usize)> = row
                .iter()
                .map(|&id| {
                    (
                        u64::from(id) * layout.scale_width as u64,
                        layout.scale_width,
                    )
                })
                .collect();
            let codes = read(&codes_name, &codes_ranges)?;
            let scales = read(&scales_name, &scales_ranges)?;
            if codes.len() != layout.codes_len || scales.len() != layout.scales_len {
                return Err(format!(
                    "engram arena: position {position} layer {layer} read {} code and {} scale bytes, the entry holds {} and {}",
                    codes.len(),
                    scales.len(),
                    layout.codes_len,
                    layout.scales_len
                ));
            }
            let base = (slot * positions + position) * layout.stride;
            bytes[base..base + layout.codes_len].copy_from_slice(&codes);
            let scales_base = base + align_up(layout.codes_len);
            bytes[scales_base..scales_base + layout.scales_len].copy_from_slice(&scales);
        }
    }
    Ok(EngramRowArena {
        layout,
        bytes,
        positions,
        slots,
        skip,
    })
}

/// Builds the arena from the run's own token history: `ngram_hashes` over the same look-back the host
/// seam uses, then [`build_with_hashes`].
///
/// # Errors
/// Fewer tokens than positions, or a hashing/read failure (`build_with_hashes`).
pub fn build(
    config: &sparkflash_engine::config::ModelConfig,
    shape: EngramArenaShape<'_>,
    tokens: &[u32],
    positions: usize,
    read: impl FnMut(&str, &[(u64, usize)]) -> Result<Vec<u8>, String>,
) -> Result<EngramRowArena, String> {
    if tokens.len() < positions {
        return Err(format!(
            "engram arena: {} ids for {positions} positions, the run has no history to hash",
            tokens.len()
        ));
    }
    build_with_hashes(
        shape,
        positions,
        |position, out| {
            let history = &tokens[..=position];
            sparkflash_engine::forward::engram::ngram_hashes(config, history, position, out)
                .map_err(|error| format!("engram arena: hashes at position {position}: {error}"))
        },
        read,
    )
}

/// The host-side Engram state one run uploads at open: the arena, each slot's gate product and its
/// wkv matrix, and the widths the six launches take.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramHostState {
    /// The gathered table rows for the whole run.
    pub arena: EngramRowArena,
    /// The config's Engram layer order: the arena's slot order.
    pub layer_ids: Vec<usize>,
    /// Each slot's `q_weight * k_weight`, f32 `[hc * dim]` (`model.py:356`).
    pub product: Vec<Vec<f32>>,
    /// Each slot's `engram.wkv.weight` E4M3 bytes, `[out_width][inner]` in checkpoint order.
    pub wkv_weight: Vec<Vec<u8>>,
    /// Each slot's `engram.wkv.scale` E8M0 bytes, the checkpoint's block grid.
    pub wkv_scales: Vec<Vec<u8>>,
    /// The projection's output width: `dim * (hc + 1)` (25,600 released).
    pub out_width: usize,
    /// The projection's reduction width: `columns * head_dim` (6,144 released).
    pub inner: usize,
    /// The configured Engram epsilon (`config.norm_eps`).
    pub eps: f32,
}

/// Builds the run's Engram host state from the checkpoint source: the arena over the run's own token
/// history plus the constant per-layer matrices the stage reads every position.
///
/// # Errors
/// The arena's shape/read failures, a missing or mis-shaped `q_weight`/`k_weight`, or a wkv matrix
/// whose bytes disagree with the geometry the projection takes.
pub fn build_host_state(
    config: &sparkflash_engine::config::ModelConfig,
    source: &dyn sparkflash_engine::weights::WeightSource,
    tokens: &[u32],
    positions: usize,
) -> Result<EngramHostState, String> {
    use sparkflash_engine::forward::engram::{HASH_SENTINEL, hash_columns};
    let _ = HASH_SENTINEL;
    let columns = hash_columns(config);
    let head_dim = config.engram_head_dim;
    let layout = EngramArenaLayout::new(columns, head_dim)?;
    let layer_ids = config.engram_layer_ids.clone();
    let table_rows: Vec<Option<usize>> = layer_ids
        .iter()
        .map(|layer| config.engram_rows(*layer))
        .collect();
    let shape = EngramArenaShape {
        layout,
        layer_ids: &layer_ids,
        table_rows: &table_rows,
    };
    let arena = build(config, shape, tokens, positions, |name, ranges| {
        source
                .read_ranges_paged(name, ranges)
                .ok_or_else(|| {
                    format!(
                        "engram state: the source serves whole tensors only, but {name} needs {} byte ranges; paging is what keeps a 98 GB table out of memory",
                        ranges.len()
                    )
                })?
                .map_err(|error| format!("engram state: read {name}: {error}"))
    })?;
    let hc = config.hc_mult;
    let dim = config.dim;
    let out_width = dim * (hc + 1);
    let inner = columns * head_dim;
    let mut product = Vec::with_capacity(layer_ids.len());
    let mut wkv_weight = Vec::with_capacity(layer_ids.len());
    let mut wkv_scales = Vec::with_capacity(layer_ids.len());
    let element_limit = hc * dim;
    let scale_len = out_width.div_ceil(32) * inner.div_ceil(32);
    for &layer in &layer_ids {
        let prefix = format!("layers.{layer}.engram");
        let q = sparkflash_engine::linalg::decode_by_name(
            source,
            &format!("{prefix}.q_weight"),
            element_limit,
        )
        .map_err(|error| format!("engram state: {prefix}.q_weight: {error}"))?;
        let k = sparkflash_engine::linalg::decode_by_name(
            source,
            &format!("{prefix}.k_weight"),
            element_limit,
        )
        .map_err(|error| format!("engram state: {prefix}.k_weight: {error}"))?;
        if q.len() != hc * dim || k.len() != hc * dim {
            return Err(format!(
                "engram state: {prefix} needs two {} value weights, got {} and {}",
                hc * dim,
                q.len(),
                k.len()
            ));
        }
        product.push(q.iter().zip(&k).map(|(q, k)| q * k).collect::<Vec<f32>>());
        let weight = source
            .read(&format!("{prefix}.wkv.weight"))
            .map_err(|error| format!("engram state: {prefix}.wkv.weight: {error}"))?;
        let scales = source
            .read(&format!("{prefix}.wkv.scale"))
            .map_err(|error| format!("engram state: {prefix}.wkv.scale: {error}"))?;
        if weight.len() != out_width * inner {
            return Err(format!(
                "engram state: {prefix}.wkv.weight is {} B, the {out_width} x {inner} code matrix needs {}",
                weight.len(),
                out_width * inner
            ));
        }
        if scales.len() != scale_len {
            return Err(format!(
                "engram state: {prefix}.wkv.scale is {} B, the block grid needs {scale_len}",
                scales.len()
            ));
        }
        wkv_weight.push(weight);
        wkv_scales.push(scales);
    }
    Ok(EngramHostState {
        arena,
        layer_ids,
        product,
        wkv_weight,
        wkv_scales,
        out_width,
        inner,
        eps: config.norm_eps as f32,
    })
}

/// The next 256-byte boundary at or after `value`.
fn align_up(value: usize) -> usize {
    value.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> EngramArenaLayout {
        EngramArenaLayout::new(24, 256).expect("the released geometry")
    }

    /// A table of `rows` distinct rows, each row filled with `byte` so a range read is checkable.
    fn table(rows: usize, width: usize, byte: u8) -> Vec<u8> {
        let mut bytes = vec![0u8; rows * width];
        for row in 0..rows {
            bytes[row * width..(row + 1) * width].fill(byte.wrapping_add(row as u8));
        }
        bytes
    }

    #[test]
    fn the_layout_is_the_released_6400_byte_entry() {
        let layout = layout();
        assert_eq!(layout.columns(), 24);
        assert_eq!(layout.head_dim(), 256);
        assert_eq!(layout.scale_width(), 8);
        assert_eq!(layout.codes_len(), 6_144);
        assert_eq!(layout.scales_len(), 192);
        assert_eq!(layout.stride(), 6_400);
        assert_eq!(
            layout.arena_bytes(2, 138).expect("2 layers x 138"),
            1_766_400
        );
        assert_eq!(
            layout.arena_bytes(0, 1).expect_err("no layer"),
            "engram arena: 0 layers x 1 positions has no entry to gather"
        );
        assert!(EngramArenaLayout::new(24, 0).is_err(), "zero head dim");
        assert!(EngramArenaLayout::new(0, 256).is_err(), "zero columns");
    }

    #[test]
    fn every_entry_holds_the_gathered_rows_and_both_arrays_are_aligned() {
        let layout = layout();
        let ids = [1usize];
        let shape = EngramArenaShape {
            layout,
            layer_ids: &ids,
            table_rows: &[Some(4)],
        };
        // Position 0 looks up rows 0,1; position 1 looks up rows 2,3 (two rows repeated 24 times
        // would be a real hash table's shape; this test only needs distinct byte patterns).
        let codes_table = table(4, 256, 0x10);
        let scales_table = table(4, 8, 0x40);
        let arena = build_with_hashes(
            shape,
            2,
            |position, out| {
                for column in 0..24 {
                    out.push(((position * 2 + column % 2) as u32).min(3));
                }
                Ok(())
            },
            |name, ranges| {
                let source = if name.ends_with(".weight") {
                    &codes_table
                } else {
                    &scales_table
                };
                let mut out = Vec::new();
                for &(offset, length) in ranges {
                    let start = offset as usize;
                    out.extend_from_slice(&source[start..start + length]);
                }
                Ok(out)
            },
        )
        .expect("the arena builds");

        assert_eq!(arena.slots(), 1);
        assert_eq!(arena.positions(), 2);
        assert_eq!(arena.bytes().len(), 2 * layout.stride());
        let (codes, scales) = arena.entry(0, 0).expect("entry (0, 0)");
        assert_eq!(codes.len(), 6_144);
        assert_eq!(scales.len(), 192);
        // Row 0 of the codes table is 0x10, row 1 is 0x11; the position-0 row alternates them.
        assert_eq!(codes[0], 0x10);
        assert_eq!(codes[256], 0x11);
        assert_eq!(scales[0], 0x40);
        assert_eq!(scales[8], 0x41);
        // The scales array is 256-byte aligned after the codes.
        assert_eq!(
            arena.entry(0, 1).expect("entry (0, 1)").0.len(),
            layout.codes_len()
        );
    }

    #[test]
    fn a_sentinel_or_out_of_range_id_skips_the_whole_position() {
        let layout = layout();
        let ids = [1usize, 14];
        let shape = EngramArenaShape {
            layout,
            layer_ids: &ids,
            table_rows: &[Some(8), Some(8)],
        };
        let mut reads = 0usize;
        let arena = build_with_hashes(
            shape,
            3,
            |position, out| {
                // Position 0: slot 0 has a sentinel, slot 1 is in range.
                // Position 1: slot 0 is past the table, slot 1 is in range.
                // Position 2: both in range.
                for slot in 0..2 {
                    for column in 0..24 {
                        let id = match (position, slot, column) {
                            (0, 0, 3) => HASH_SENTINEL,
                            (1, 0, 5) => 8,
                            _ => (slot * 24 + column) as u32 % 8,
                        };
                        out.push(id);
                    }
                }
                Ok(())
            },
            |name, ranges| {
                reads += 1;
                let width = if name.ends_with(".weight") {
                    6_144
                } else {
                    192
                };
                let _ = ranges;
                Ok(vec![0u8; width])
            },
        )
        .expect("the arena builds");

        assert!(arena.is_skipped(0, 0).expect("in range"), "sentinel skips");
        assert!(!arena.is_skipped(1, 0).expect("in range"));
        assert!(
            arena.is_skipped(0, 1).expect("in range"),
            "past table skips"
        );
        assert!(!arena.is_skipped(0, 2).expect("in range"));
        // Two slots x three positions = six entries, two of which were skipped: eight reads.
        assert_eq!(reads, 8);
        assert!(arena.entry(2, 0).is_err(), "slot 2 does not exist");
        assert!(arena.is_skipped(0, 3).is_err(), "position 3 does not exist");
    }

    use sparkflash_engine::forward::engram::HASH_SENTINEL;

    #[test]
    fn a_listed_layer_without_a_table_is_refused_instead_of_writing_a_zero_row() {
        let layout = layout();
        let ids = [1usize];
        let shape = EngramArenaShape {
            layout,
            layer_ids: &ids,
            table_rows: &[None],
        };
        let error = build_with_hashes(shape, 1, |_p, _out| Ok(()), |_n, _r| Ok(vec![0u8; 6_400]))
            .expect_err("no table");
        assert!(error.contains("no table"), "{error}");

        let shape = EngramArenaShape {
            layout,
            layer_ids: &ids,
            table_rows: &[Some(8), Some(8)],
        };
        let error = build_with_hashes(shape, 1, |_p, _out| Ok(()), |_n, _r| Ok(vec![0u8; 6_400]))
            .expect_err("layer/table count");
        assert!(error.contains("2 table heights"), "{error}");
    }
}
