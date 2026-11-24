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

//! The packed BF16 word form, on the host.
//!
//! The device reads this form through `kernels/bf16_packed_sm121.cuh`: a row is plane A (each
//! weight's sign and mantissa byte), plane B (its exponent as a 4-bit offset under the step's
//! largest, two a byte) and plane C (that largest, one byte a step). A word fifteen or more
//! binades under its step's largest reads offset 15, and the whole word travels in the matrix's
//! escape list as `column << 16 | word`; `offsets[row]..offsets[row + 1]` names the row's run.
//!
//! The layout is the row walk's, not the tiled reader's: a step holds 32 lanes' chunks, a lane's
//! chunks together (four in a quad step, two in a pair step, one in a single step), so a lane
//! reads one contiguous run of plane A and one of plane B. `row_bytes` is the row's A + B + C
//! bytes padded to a 16-byte load.
//!
//! This module is the host half: a checkpoint writer packs a BF16 matrix into the form and a
//! loader unpacks it back to the BF16 rows the body reads. The two are inverses, which the tests
//! pin on every bf16 bit pattern they generate.

/// The offset that marks an escape (kernels/bf16_packed_sm121.cuh:Q4E_PACKED_ESCAPE).
pub const ESCAPE: u8 = 15;
/// A packed row's columns are whole rounds of 32 lanes' 16-byte chunks.
const PACKED_MULTIPLE: usize = 256;
/// An escape stores its column in 16 bits.
const MAX_COLUMNS: usize = 1 << 16;
/// A packed row, its planes and the activations it meets are 16-byte loads.
const ALIGN: usize = 16;

/// The steps one lane walks over a row of `columns` (crates/sparkflash-ops/src/packed.rs).
#[must_use]
pub fn walk_steps(columns: usize) -> usize {
    let chunks = columns / 8;
    let (mut chunk, mut steps) = (0, 0);
    while chunk + 96 < chunks {
        chunk += 128;
        steps += 1;
    }
    if chunk + 32 < chunks {
        chunk += 64;
        steps += 1;
    }
    while chunk < chunks {
        chunk += 32;
        steps += 1;
    }
    steps
}

/// One packed row's bytes: plane A, plane B, plane C, padded to a 16-byte load.
#[must_use]
pub fn row_bytes(columns: usize) -> usize {
    (columns + columns / 2 + 32 * walk_steps(columns)).next_multiple_of(ALIGN)
}

/// Whether this form takes a `rows` x `columns` matrix.
#[must_use]
pub fn packable(rows: usize, columns: usize) -> bool {
    rows > 0 && columns.is_multiple_of(PACKED_MULTIPLE) && columns < MAX_COLUMNS
}

/// The column count a packed row stride stands for, if it is one of the form's strides.
///
/// A checkpoint header states the packed rows as `[rows, row_bytes]`; the logical columns are
/// the multiple of 256 whose stride is that many bytes. The search is over the form's own
/// strides, so a header that is not one is refused rather than guessed.
#[must_use]
pub fn columns_of_row_bytes(stride: usize) -> Option<usize> {
    let mut columns = PACKED_MULTIPLE;
    while columns < MAX_COLUMNS {
        match row_bytes(columns).cmp(&stride) {
            core::cmp::Ordering::Equal => return Some(columns),
            core::cmp::Ordering::Greater => return None,
            core::cmp::Ordering::Less => columns += PACKED_MULTIPLE,
        }
    }
    None
}

/// Splits an escape list into its per-row offsets and its entries.
///
/// # Errors
/// Too short for its row count, or an offset that runs past the entries.
pub fn escape_runs(escapes: &[u32], rows: usize) -> Result<(&[u32], &[u32]), String> {
    if escapes.len() < rows + 1 {
        return Err(format!(
            "a packed {rows}-row tensor needs {} escape offsets, the list holds {}",
            rows + 1,
            escapes.len()
        ));
    }
    let (offsets, entries) = escapes.split_at(rows + 1);
    let mut previous = 0u32;
    for (row, offset) in offsets.iter().copied().enumerate() {
        if offset < previous {
            return Err(format!(
                "the escape list's offset for row {row} is {offset}, below the previous row's {previous}"
            ));
        }
        if offset as usize > entries.len() {
            return Err(format!(
                "the escape list's offset for row {row} is {offset}, past its {} entries",
                entries.len()
            ));
        }
        previous = offset;
    }
    Ok((offsets, entries))
}

/// Packs `rows` x `columns` BF16 bytes into the packed word form.
///
/// Returns the packed rows (each [`row_bytes`] long) and the escape list: `rows + 1` offsets
/// followed by the entries, in the order the row walk meets them.
///
/// # Errors
/// A shape the form does not take, or a byte count that is not two per value.
pub fn pack(rows: usize, columns: usize, bf16: &[u8]) -> Result<(Vec<u8>, Vec<u32>), String> {
    if !packable(rows, columns) {
        return Err(format!(
            "the packed form takes whole 256-column rounds and fewer than {MAX_COLUMNS} columns, not {rows} x {columns}"
        ));
    }
    let expected = rows * columns * 2;
    if bf16.len() != expected {
        return Err(format!(
            "a {rows} x {columns} bf16 matrix is {expected} bytes, the store holds {}",
            bf16.len()
        ));
    }
    let stride = row_bytes(columns);
    let steps = walk_steps(columns);
    let mut packed = vec![0u8; rows * stride];
    let mut escapes: Vec<u32> = vec![0; rows + 1];
    let mut entries: Vec<u32> = Vec::new();
    for row in 0..rows {
        let source = &bf16[row * columns * 2..(row + 1) * columns * 2];
        let out = &mut packed[row * stride..(row + 1) * stride];
        let chunks = columns / 8;
        let mut chunk = 0usize; // lane 0; the group offset carries the lane
        let mut step = 0usize;
        let mut a_off = 0usize;
        while chunk + 96 < chunks {
            encode_group(
                source,
                out,
                columns,
                &mut entries,
                row,
                chunk,
                4,
                step,
                a_off,
            )?;
            chunk += 128;
            step += 1;
            a_off += 1024;
        }
        if chunk + 32 < chunks {
            encode_group(
                source,
                out,
                columns,
                &mut entries,
                row,
                chunk,
                2,
                step,
                a_off,
            )?;
            chunk += 64;
            step += 1;
            a_off += 512;
        }
        while chunk < chunks {
            encode_group(
                source,
                out,
                columns,
                &mut entries,
                row,
                chunk,
                1,
                step,
                a_off,
            )?;
            chunk += 32;
            step += 1;
            a_off += 256;
        }
        debug_assert_eq!(step, steps);
        escapes[row + 1] = u32::try_from(entries.len())
            .map_err(|_| "a packed matrix with more than 2^32 escapes".to_owned())?;
    }
    escapes.extend_from_slice(&entries);
    Ok((packed, escapes))
}

/// Encodes one step's group of `lanes` chunks for every lane, at `a_off`.
#[allow(clippy::too_many_arguments)]
fn encode_group(
    source: &[u8],
    out: &mut [u8],
    columns: usize,
    entries: &mut Vec<u32>,
    row: usize,
    base_chunk: usize,
    lanes: usize,
    step: usize,
    a_off: usize,
) -> Result<(), String> {
    let chunks = columns / 8;
    let b_off = columns + a_off / 2;
    let c_off = columns + columns / 2;
    for lane in 0..32usize {
        let first = base_chunk + lane;
        if first + 32 * (lanes - 1) >= chunks {
            return Err(format!(
                "row {row}: step {step} lane {lane} runs past the {chunks} chunks of a {columns}-column row"
            ));
        }
        let mut top = 0u8;
        for j in 0..lanes {
            let chunk = first + 32 * j;
            for k in 0..8 {
                let word = word_at(source, chunk * 8 + k);
                top = top.max(exponent(word));
            }
        }
        out[c_off + step * 32 + lane] = top;
        for j in 0..lanes {
            let chunk = first + 32 * j;
            let a_at = a_off + lane * 8 * lanes + j * 8;
            let b_at = b_off + lane * 4 * lanes + j * 4;
            for k in 0..8usize {
                let word = word_at(source, chunk * 8 + k);
                let offset = top.saturating_sub(exponent(word));
                let a = ((word >> 8) as u8 & 0x80) | (word as u8 & 0x7F);
                out[a_at + k] = a;
                let nib = if offset >= ESCAPE {
                    entries.push(
                        (u32::try_from(chunk * 8 + k)
                            .map_err(|_| "a packed column past 2^32".to_owned())?
                            << 16)
                            | u32::from(word),
                    );
                    ESCAPE
                } else {
                    offset
                };
                if k < 4 {
                    out[b_at + k] = (out[b_at + k] & 0xF0) | nib;
                } else {
                    out[b_at + k - 4] = (out[b_at + k - 4] & 0x0F) | (nib << 4);
                }
            }
        }
    }
    Ok(())
}

/// Unpacks a packed matrix back to its row-major BF16 bytes.
///
/// # Errors
/// A byte count that is not the packed row stride, an escape list that does not describe the
/// rows, or an escape whose column is outside the matrix.
pub fn unpack(
    rows: usize,
    columns: usize,
    packed: &[u8],
    escapes: &[u32],
) -> Result<Vec<u8>, String> {
    if !packable(rows, columns) {
        return Err(format!(
            "the packed form takes whole 256-column rounds and fewer than {MAX_COLUMNS} columns, not {rows} x {columns}"
        ));
    }
    let stride = row_bytes(columns);
    let expected = rows * stride;
    if packed.len() != expected {
        return Err(format!(
            "the packed form of {rows} x {columns} is {expected} bytes, the store holds {}; the escape list describes the same rows",
            packed.len()
        ));
    }
    let (offsets, entries) = escape_runs(escapes, rows)?;
    let mut out = vec![0u8; rows * columns * 2];
    for row in 0..rows {
        let src = &packed[row * stride..(row + 1) * stride];
        let row_entries = &entries[offsets[row] as usize..offsets[row + 1] as usize];
        let chunks = columns / 8;
        let mut chunk = 0usize;
        let mut step = 0usize;
        let mut a_off = 0usize;
        while chunk + 96 < chunks {
            decode_group(
                src,
                &mut out[row * columns * 2..],
                columns,
                row_entries,
                row,
                chunk,
                4,
                step,
                a_off,
            )?;
            chunk += 128;
            step += 1;
            a_off += 1024;
        }
        if chunk + 32 < chunks {
            decode_group(
                src,
                &mut out[row * columns * 2..],
                columns,
                row_entries,
                row,
                chunk,
                2,
                step,
                a_off,
            )?;
            chunk += 64;
            step += 1;
            a_off += 512;
        }
        while chunk < chunks {
            decode_group(
                src,
                &mut out[row * columns * 2..],
                columns,
                row_entries,
                row,
                chunk,
                1,
                step,
                a_off,
            )?;
            chunk += 32;
            step += 1;
            a_off += 256;
        }
    }
    Ok(out)
}

/// Decodes one step's group of chunks for every lane.
#[allow(clippy::too_many_arguments)]
fn decode_group(
    src: &[u8],
    out: &mut [u8],
    columns: usize,
    entries: &[u32],
    row: usize,
    base_chunk: usize,
    lanes: usize,
    step: usize,
    a_off: usize,
) -> Result<(), String> {
    let chunks = columns / 8;
    let b_off = columns + a_off / 2;
    let c_off = columns + columns / 2;
    for lane in 0..32usize {
        let first = base_chunk + lane;
        if first + 32 * (lanes - 1) >= chunks {
            return Err(format!(
                "row {row}: step {step} lane {lane} runs past the {chunks} chunks of a {columns}-column row"
            ));
        }
        let top = src[c_off + step * 32 + lane];
        for j in 0..lanes {
            let chunk = first + 32 * j;
            let a_at = a_off + lane * 8 * lanes + j * 8;
            let b_at = b_off + lane * 4 * lanes + j * 4;
            for k in 0..8usize {
                let column = chunk * 8 + k;
                let nib = if k < 4 {
                    src[b_at + k] & 0x0F
                } else {
                    src[b_at + k - 4] >> 4
                };
                let word = if nib >= ESCAPE {
                    match escape_word(entries, column)? {
                        Some(word) => word,
                        None => {
                            // A value exactly fifteen binades down is an escape too, so a missing
                            // entry is a malformed list rather than a value to reconstruct.
                            return Err(format!(
                                "row {row}: column {column} reads the escape offset but its escape list has no entry"
                            ));
                        }
                    }
                } else {
                    let exponent = top - nib;
                    let a = u16::from(src[a_at + k]);
                    ((a & 0x80) << 8) | (u16::from(exponent) << 7) | (a & 0x7F)
                };
                let at = column * 2;
                out[at] = word as u8;
                out[at + 1] = (word >> 8) as u8;
            }
        }
    }
    Ok(())
}

/// The stored word of the escape at `column`, if the row's run holds one.
fn escape_word(entries: &[u32], column: usize) -> Result<Option<u16>, String> {
    for entry in entries {
        let stored = (entry >> 16) as usize;
        if stored == column {
            return Ok(Some(*entry as u16));
        }
    }
    Ok(None)
}

/// One little-endian BF16 word of the row-major store.
fn word_at(source: &[u8], value: usize) -> u16 {
    u16::from_le_bytes([source[value * 2], source[value * 2 + 1]])
}

/// A BF16 word's exponent byte.
fn exponent(word: u16) -> u8 {
    ((word >> 7) & 0xFF) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        (bits >> 16) as u16
    }

    fn round_trip(rows: usize, columns: usize, words: &[u16]) {
        let mut bytes = Vec::with_capacity(words.len() * 2);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let (packed, escapes) = pack(rows, columns, &bytes).expect("the form takes the shape");
        assert_eq!(packed.len(), rows * row_bytes(columns));
        let back = unpack(rows, columns, &packed, &escapes).expect("the form unpacks");
        assert_eq!(back, bytes, "every bf16 word survives the round trip");
    }

    #[test]
    fn the_row_stride_is_the_one_the_device_header_states() {
        // 4096 columns: 4096 A + 2048 B + 6 * 32 C, already a multiple of 16.
        assert_eq!(walk_steps(4096), 4);
        assert_eq!(row_bytes(4096), 4096 + 2048 + 128);
        assert_eq!(row_bytes(256), 256 + 128 + 32);
        assert!(packable(1, 512));
        assert!(!packable(1, 128), "a row is whole 32-lane chunks");
    }

    #[test]
    fn the_round_trip_holds_on_a_step_top_and_on_a_far_escape() {
        let (rows, columns) = (2usize, 512usize);
        let mut words = vec![0u16; rows * columns];
        // One value at the top exponent of its step, one fifteen binades under it (an escape),
        // and one zero.
        words[0] = bf16_bits(1.0);
        words[1] = bf16_bits(1.0 / 32768.0);
        words[2] = 0;
        for (index, word) in words.iter_mut().enumerate().skip(3) {
            *word = bf16_bits(0.25 + (index % 17) as f32 * 0.0625).wrapping_add((index % 3) as u16);
        }
        round_trip(rows, columns, &words);
    }

    #[test]
    fn the_round_trip_holds_on_every_shape_and_on_random_bits() {
        for (rows, columns) in [(1usize, 256usize), (3, 512), (2, 4096)] {
            let mut state = 0x1234_5678u32;
            let mut words = Vec::with_capacity(rows * columns);
            for _ in 0..rows * columns {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                words.push((state >> 16) as u16);
            }
            round_trip(rows, columns, &words);
        }
    }

    #[test]
    fn an_escape_list_that_does_not_describe_the_rows_is_refused() {
        let bytes = vec![0u8; 256 * 2];
        let (packed, escapes) = pack(1, 256, &bytes).expect("pack");
        assert!(unpack(1, 256, &packed, &[]).is_err());
        let mut bad = escapes.clone();
        bad[1] = u32::MAX;
        assert!(unpack(1, 256, &packed, &bad).is_err());
        assert!(
            unpack(2, 256, &packed, &escapes).is_err(),
            "the bytes are one row"
        );
    }
}
