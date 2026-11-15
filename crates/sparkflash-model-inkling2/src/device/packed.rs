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

//! The dense projections in the packed BF16 form (kernels/bf16_packed_sm121.cuh).
//!
//! A decode step reads every attention projection, the router, the dynamic taps and the head,
//! about 50 MB a layer and 462 MB of head a rank, and those GEMVs run at the DRAM floor. The
//! packed form holds the same BF16 words in about three quarters of the bytes - a sign and
//! mantissa byte a weight, a 4-bit exponent offset under the largest of the 32 weights one lane
//! reads at a step, and that largest - and the packed GEMVs rebuild the words in registers and
//! add them exactly as the BF16 walks do. Nothing about a sum changes: the form is lossless and
//! the arithmetic is the BF16 kernels' expression for expression, so a packed projection is the
//! BF16 projection's bits.
//!
//! The matrices are packed in place at load, a group of rows at a time through one scratch
//! buffer, and every group is unpacked and compared with the BF16 rows it replaces before they
//! are overwritten: a single differing word refuses the rank. A prompt's tensor-core tiles read
//! BF16, so a projection walked by one is unpacked into that scratch first.

use std::collections::{BTreeSet, HashMap};

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_weights::DType;

use super::{Kernels, Shared, TOKEN_BLOCK, Weights, gemv_grid};
use crate::device_plan::{InklingDevicePlan, Op, Projection, Target};

/// The switch between the packed form (`packed`, the default) and the checkpoint's BF16 rows
/// (`bf16`). Anything else is refused.
pub(crate) const DENSE_ENV: &str = "SPARKFLASH_INKLING_DENSE";

/// How the dense projections are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DenseForm {
    /// Packed at load, every matrix whose shape the form takes.
    Packed,
    /// As the checkpoint stores them.
    Bf16,
}

impl DenseForm {
    /// The form [DENSE_ENV] names.
    pub(super) fn from_env() -> Result<Self, String> {
        Self::parse(std::env::var(DENSE_ENV).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("" | "packed") => Ok(Self::Packed),
            Some("bf16") => Ok(Self::Bf16),
            Some(other) => Err(format!(
                "{DENSE_ENV}={other} is not a dense form: use packed (the default: the dense \
                 projections in a lossless 12.25-bit form, the BF16 bits from three quarters of \
                 the bytes) or bf16 (the checkpoint's rows)"
            )),
        }
    }
}

/// A packed row's columns are whole rounds of 32 lanes' 16-byte chunks, so every lane of the
/// walk takes the same steps.
const PACKED_MULTIPLE: usize = 256;
/// An escape stores its column in 16 bits.
const PACKED_COLUMNS: usize = 1 << 16;
/// A packed row, its planes and the activations it meets are 16-byte loads.
const ALIGN: usize = 16;
/// The smallest scratch the load packs through, when the widest packed matrix is smaller.
const MIN_SCRATCH: usize = 1 << 20;

/// The steps q4e_row_dot_bf16 walks a lane through over a row of `columns`: quads, then perhaps
/// a pair, then single chunks.
fn walk_steps(columns: usize) -> usize {
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
fn row_bytes(columns: usize) -> usize {
    (columns + columns / 2 + 32 * walk_steps(columns)).next_multiple_of(ALIGN)
}

/// Whether the form takes a `rows` x `columns` matrix.
fn packable(rows: usize, columns: usize) -> bool {
    rows > 0 && columns.is_multiple_of(PACKED_MULTIPLE) && columns < PACKED_COLUMNS
}

/// One packed matrix: its rows where its BF16 rows were, and its escapes.
#[derive(Debug, Clone, Copy)]
pub(super) struct Packed {
    pointer: Deviceptr,
    offsets: Deviceptr,
    entries: Deviceptr,
    rows: usize,
    columns: usize,
    row_bytes: usize,
}

impl Packed {
    /// Refuses a launch the packed walk cannot take: other columns than the matrix's, more rows
    /// than it has, or activations that are not 16-byte loads (the BF16 walk falls back to a
    /// column walk there; the packed one has none).
    fn admits(&self, rows: usize, columns: usize, input: Deviceptr) -> Result<(), String> {
        if columns != self.columns || rows > self.rows {
            return Err(format!(
                "a {rows} x {columns} projection over a packed {} x {} matrix",
                self.rows, self.columns
            ));
        }
        if !(input.0 as usize).is_multiple_of(ALIGN) {
            return Err(format!(
                "a packed projection reads its activations as 16-byte loads, and {:#x} is not \
                 one ({DENSE_ENV}=bf16 keeps the BF16 walk)",
                input.0
            ));
        }
        Ok(())
    }
}

/// The rank's packed matrices, by the device pointer of the slot each replaced.
pub(super) struct DenseWeights {
    packed: HashMap<u64, Packed>,
    /// Every matrix's offsets and escapes.
    arena: Option<Deviceptr>,
    /// A BF16 matrix's worth, for the prompt tiles: the widest packed matrix but the head,
    /// which only the GEMVs read.
    scratch: Option<(Deviceptr, usize)>,
    /// The BF16 bytes the packed matrices replaced, and their packed bytes.
    bf16_bytes: usize,
    packed_bytes: usize,
}

impl DenseWeights {
    pub(super) fn none() -> Self {
        Self {
            packed: HashMap::new(),
            arena: None,
            scratch: None,
            bf16_bytes: 0,
            packed_bytes: 0,
        }
    }

    /// The packed matrix whose slot starts at `weight`, if it is one.
    pub(super) fn get(&self, weight: Deviceptr) -> Option<Packed> {
        self.packed.get(&weight.0).copied()
    }

    /// The allocations to free.
    pub(super) fn allocations(&self) -> impl Iterator<Item = Deviceptr> + '_ {
        self.arena
            .into_iter()
            .chain(self.scratch.map(|(pointer, _)| pointer))
    }

    /// One line for the boot log.
    pub(super) fn summary(&self) -> String {
        if self.packed.is_empty() {
            return format!("dense projections in BF16 ({DENSE_ENV}=bf16 or none packable)");
        }
        format!(
            "{} dense projections packed: {:.1} -> {:.1} MB ({:.1}% fewer bytes a pass)",
            self.packed.len(),
            self.bf16_bytes as f64 / 1e6,
            self.packed_bytes as f64 / 1e6,
            100.0 * (1.0 - self.packed_bytes as f64 / self.bf16_bytes.max(1) as f64)
        )
    }
}

/// The served-plan slots only the dense GEMV walks read, and the head among them: a slot any
/// other step reads (a norm, an embedding, a band) stays BF16.
fn dense_slots(plans: &[&InklingDevicePlan]) -> (BTreeSet<usize>, Option<usize>) {
    let mut dense = BTreeSet::new();
    let mut other = BTreeSet::new();
    let mut head = None;
    for plan in plans {
        for step in plan.steps() {
            match (step.op, step.target) {
                (Op::Gemv, Target::Project(projection)) => {
                    if let Some((first, rest)) = step.weights.split_first() {
                        dense.insert(*first);
                        if matches!(projection, Projection::Head) {
                            head = Some(*first);
                        }
                        other.extend(rest.iter().copied());
                    }
                }
                // [static taps, the taps' projection]
                (Op::ShortConvDynamic, _) => match step.weights.as_slice() {
                    [taps, projection, rest @ ..] => {
                        other.insert(*taps);
                        dense.insert(*projection);
                        other.extend(rest.iter().copied());
                    }
                    weights => other.extend(weights.iter().copied()),
                },
                _ => other.extend(step.weights.iter().copied()),
            }
        }
    }
    (dense.difference(&other).copied().collect(), head)
}

/// One matrix to pack.
struct Candidate {
    pointer: Deviceptr,
    rows: usize,
    columns: usize,
    offsets: Vec<u32>,
}

/// Packs every dense projection the form takes, in place (see the module note), or none under
/// [DenseForm::Bf16].
///
/// # Errors
/// A driver failure, or a matrix whose packed rows do not unpack to its BF16 rows.
pub(super) fn pack_dense(
    cuda: &Cuda,
    kernels: &Kernels,
    plans: &[&InklingDevicePlan],
    weights: &Weights,
    form: DenseForm,
) -> Result<DenseWeights, String> {
    if form == DenseForm::Bf16 || plans.is_empty() {
        return Ok(DenseWeights::none());
    }
    let served = plans[0].served();
    let (slots, head) = dense_slots(plans);
    let mut candidates = Vec::new();
    for index in slots {
        let spec = served.rank_spec_at(index);
        let slot = weights.get(index);
        let [rows, columns] = spec.shape[..] else {
            continue;
        };
        if spec.dtype != DType::BF16
            || !packable(rows, columns)
            || slot.bytes != rows * columns * 2
            || !(slot.pointer.0 as usize).is_multiple_of(ALIGN)
        {
            continue;
        }
        candidates.push((index, Candidate {
            pointer: slot.pointer,
            rows,
            columns,
            offsets: Vec::new(),
        }));
    }
    if candidates.is_empty() {
        return Ok(DenseWeights::none());
    }
    let mut total = 0usize;
    // The scratch holds the widest matrix but the head as BF16 (what a prompt tile unpacks) and
    // is what the load packs through, a group of rows at a time.
    let scratch_bytes = candidates
        .iter()
        .filter(|(index, _)| Some(*index) != head)
        .map(|(_, matrix)| matrix.rows * matrix.columns * 2)
        .max()
        .unwrap_or(0)
        .max(MIN_SCRATCH);
    let widest_rows = candidates.iter().map(|(_, m)| m.rows).max().unwrap_or(1);
    let counts = cuda.alloc(widest_rows * 4, &mut total)?;
    let differ = cuda.alloc(ALIGN, &mut total)?;
    let scratch = cuda.alloc(scratch_bytes, &mut total)?;
    let mut matrices: Vec<Candidate> = candidates.into_iter().map(|(_, matrix)| matrix).collect();
    let mut arena = None;
    let packed = pack_in_place(
        cuda,
        kernels,
        &mut matrices,
        (counts, differ),
        (scratch, scratch_bytes),
        &mut arena,
    );
    cuda.free_buffers(&[&counts, &differ]);
    match packed {
        Ok((packed, bf16_bytes, packed_bytes)) => Ok(DenseWeights {
            packed,
            arena,
            scratch: Some((scratch, scratch_bytes)),
            bf16_bytes,
            packed_bytes,
        }),
        Err(error) => {
            let mut held = vec![scratch];
            held.extend(arena);
            cuda.free_buffers(&held.iter().collect::<Vec<_>>());
            Err(error)
        }
    }
}

/// [pack_dense]'s passes: count every matrix's escapes, lay out one arena (a matrix's offsets,
/// then its escapes, each a 16-byte run), then pack each matrix a group at a time, checking every
/// group against the BF16 rows it replaces before they are overwritten. Returns the packed
/// matrices, the BF16 bytes they replaced and their packed bytes; `arena` is set as soon as it
/// is allocated, so the caller frees it on an error.
fn pack_in_place(
    cuda: &Cuda,
    kernels: &Kernels,
    matrices: &mut [Candidate],
    (counts, differ): (Deviceptr, Deviceptr),
    (scratch, scratch_bytes): (Deviceptr, usize),
    arena: &mut Option<Deviceptr>,
) -> Result<(HashMap<u64, Packed>, usize, usize), String> {
    let mut arena_bytes = 0usize;
    for matrix in matrices.iter_mut() {
        launch_pack(
            cuda,
            kernels,
            matrix.pointer,
            Deviceptr(0),
            Deviceptr(0),
            Deviceptr(0),
            counts,
            matrix.rows,
            matrix.columns,
        )?;
        let mut bytes = vec![0u8; matrix.rows * 4];
        cuda.copy_dtoh(&mut bytes, &counts)?;
        let mut offsets = Vec::with_capacity(matrix.rows + 1);
        let mut running = 0u32;
        offsets.push(0);
        for word in bytes.as_chunks::<4>().0 {
            running = running
                .checked_add(u32::from_le_bytes(*word))
                .ok_or("a matrix with more than 2^32 escapes")?;
            offsets.push(running);
        }
        arena_bytes += ((matrix.rows + 1) * 4).next_multiple_of(ALIGN)
            + (running as usize * 4).max(4).next_multiple_of(ALIGN);
        matrix.offsets = offsets;
    }
    let mut total = 0usize;
    let base = cuda.alloc(arena_bytes, &mut total)?;
    *arena = Some(base);
    let mut packed = HashMap::new();
    let mut at = 0usize;
    let (mut bf16_bytes, mut packed_bytes) = (0usize, 0usize);
    cuda.memset_d8_async(differ, 0, ALIGN)?;
    for matrix in matrices.iter() {
        let offsets = base.offset(at);
        let offset_bytes: Vec<u8> = matrix
            .offsets
            .iter()
            .flat_map(|offset| offset.to_le_bytes())
            .collect();
        cuda.copy_htod(&offsets, &offset_bytes)?;
        at += offset_bytes.len().next_multiple_of(ALIGN);
        let escapes = matrix.offsets.last().copied().unwrap_or(0) as usize;
        let entries = base.offset(at);
        at += (escapes * 4).max(4).next_multiple_of(ALIGN);
        let row_bytes = row_bytes(matrix.columns);
        // A group's packed rows end before the next group's BF16 rows begin, so copying a group
        // down never overwrites a row still to be read; the check reads the group's BF16 rows
        // before its copy does.
        let group = (scratch_bytes / row_bytes).max(1);
        let mut first = 0;
        while first < matrix.rows {
            let take = group.min(matrix.rows - first);
            let source = matrix.pointer.offset(first * matrix.columns * 2);
            let group_offsets = offsets.offset(first * 4);
            launch_pack(
                cuda,
                kernels,
                source,
                scratch,
                group_offsets,
                entries,
                Deviceptr(0),
                take,
                matrix.columns,
            )?;
            cuda.launch(
                kernels.unpack.kernel,
                (gemv_grid(take), 1, 1),
                (TOKEN_BLOCK, 1, 1),
                &[
                    scratch.0,
                    group_offsets.0,
                    entries.0,
                    0,
                    source.0,
                    differ.0,
                    take as u64,
                    matrix.columns as u64,
                    row_bytes as u64,
                ],
            )?;
            cuda.copy_dtod(
                &matrix.pointer.offset(first * row_bytes),
                &scratch,
                take * row_bytes,
            )?;
            first += take;
        }
        bf16_bytes += matrix.rows * matrix.columns * 2;
        packed_bytes += matrix.rows * row_bytes + matrix.offsets.len() * 4 + escapes * 4;
        packed.insert(
            matrix.pointer.0,
            Packed {
                pointer: matrix.pointer,
                offsets,
                entries,
                rows: matrix.rows,
                columns: matrix.columns,
                row_bytes,
            },
        );
    }
    cuda.synchronize()?;
    let mut word = [0u8; 4];
    cuda.copy_dtoh(&mut word, &differ)?;
    let differ_words = u32::from_le_bytes(word);
    if differ_words != 0 {
        return Err(format!(
            "the packed dense projections unpack to {differ_words} BF16 words other than the \
             checkpoint's; the rank refuses to serve them ({DENSE_ENV}=bf16 keeps the BF16 rows)"
        ));
    }
    Ok((packed, bf16_bytes, packed_bytes))
}

/// One q4e_pack_bf16 launch: counts with `packed` null, else the packed rows and escapes.
#[allow(clippy::too_many_arguments)]
fn launch_pack(
    cuda: &Cuda,
    kernels: &Kernels,
    source: Deviceptr,
    packed: Deviceptr,
    offsets: Deviceptr,
    entries: Deviceptr,
    counts: Deviceptr,
    rows: usize,
    columns: usize,
) -> Result<(), String> {
    cuda.launch(
        kernels.pack.kernel,
        (gemv_grid(rows), 1, 1),
        (TOKEN_BLOCK, 1, 1),
        &[
            source.0,
            packed.0,
            offsets.0,
            entries.0,
            counts.0,
            rows as u64,
            columns as u64,
            row_bytes(columns) as u64,
        ],
    )
}

impl Shared {
    /// A packed matrix's projection of `groups` groups of `batch` rows: q4e_gemv_bf16's walk and
    /// bits ([Shared::gemv_batch]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_packed(
        &self,
        packed: Packed,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
        batch: usize,
        groups: u32,
    ) -> Result<(), String> {
        packed.admits(rows, columns, input)?;
        self.launch(
            self.kernels.gemv_packed,
            (gemv_grid(rows), groups, 1),
            (TOKEN_BLOCK, 1, 1),
            &[
                packed.pointer.0,
                packed.offsets.0,
                packed.entries.0,
                input.0,
                out.0,
                rows as u64,
                columns as u64,
                batch as u64,
                packed.row_bytes as u64,
            ],
        )
    }

    /// Up to four packed matrices over one row, as one launch: q4e_gemv4_bf16's walk and bits.
    /// `parts` holds (matrix, output, rows).
    pub(super) fn gemv4_packed(
        &self,
        parts: &[(Packed, Deviceptr, usize)],
        input: Deviceptr,
        columns: usize,
    ) -> Result<(), String> {
        let fused = super::GEMV_FUSED;
        if parts.is_empty() || parts.len() > fused {
            return Err(format!(
                "a fused projection takes 1 to {fused} matrices, not {}",
                parts.len()
            ));
        }
        // (w0..w3, o0..o3, e0..e3, x, y0..y3, r0..r3, columns, batch, row_bytes)
        let mut args = [0u64; 5 * 4 + 4];
        for (slot, (packed, out, rows)) in parts.iter().enumerate() {
            packed.admits(*rows, columns, input)?;
            args[slot] = packed.pointer.0;
            args[4 + slot] = packed.offsets.0;
            args[8 + slot] = packed.entries.0;
            args[13 + slot] = out.0;
            args[17 + slot] = *rows as u64;
        }
        args[12] = input.0;
        args[21] = columns as u64;
        args[22] = 1;
        args[23] = parts[0].0.row_bytes as u64;
        let total: usize = parts.iter().map(|(_, _, rows)| rows).sum();
        self.launch(
            self.kernels.gemv4_packed,
            (gemv_grid(total), 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &args,
        )
    }

    /// The BF16 rows a prompt tile reads for `weight`: the matrix itself, or a packed matrix's
    /// first `rows` rows unpacked into the scratch. The scratch is one matrix, so the tile that
    /// reads it has to be the next launch on the stream that reads it (every launch here is on
    /// the one stream, and the next unpack waits for it).
    pub(super) fn bf16_rows(
        &self,
        weight: Deviceptr,
        rows: usize,
        columns: usize,
    ) -> Result<Deviceptr, String> {
        let Some(packed) = self.dense.get(weight) else {
            return Ok(weight);
        };
        let (scratch, bytes) = self
            .dense
            .scratch
            .ok_or("a packed matrix with no scratch to unpack it into")?;
        if columns != packed.columns || rows > packed.rows || rows * columns * 2 > bytes {
            return Err(format!(
                "a prompt tile over {rows} x {columns} rows of a packed {} x {} matrix, with a \
                 {bytes}-byte scratch ({DENSE_ENV}=bf16 keeps the BF16 rows)",
                packed.rows, packed.columns
            ));
        }
        self.launch(
            self.kernels.unpack,
            (gemv_grid(rows), 1, 1),
            (TOKEN_BLOCK, 1, 1),
            &[
                packed.pointer.0,
                packed.offsets.0,
                packed.entries.0,
                scratch.0,
                0,
                0,
                rows as u64,
                columns as u64,
                packed.row_bytes as u64,
            ],
        )?;
        Ok(scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_walk_geometry_is_the_row_kernels() {
        // 4,608 columns: four quads and a pair; 1,536: one quad and a pair; 256: one single.
        assert_eq!(walk_steps(4608), 5);
        assert_eq!(row_bytes(4608), 7072);
        assert_eq!(walk_steps(1536), 2);
        assert_eq!(row_bytes(1536), 2368);
        assert_eq!(walk_steps(256), 1);
        assert_eq!(row_bytes(256), 416);
        assert_eq!(walk_steps(768), 2);
        assert_eq!(row_bytes(768), 1216);
    }

    #[test]
    fn only_whole_lane_rounds_pack() {
        assert!(packable(12, 4608));
        assert!(packable(1, 256));
        assert!(!packable(12, 4600));
        assert!(!packable(12, 128));
        assert!(!packable(0, 4608));
        assert!(!packable(4, 1 << 16));
    }

    #[test]
    fn the_switch_names_two_forms_and_refuses_the_rest() {
        assert_eq!(DenseForm::parse(None), Ok(DenseForm::Packed));
        assert_eq!(DenseForm::parse(Some("")), Ok(DenseForm::Packed));
        assert_eq!(DenseForm::parse(Some("packed")), Ok(DenseForm::Packed));
        assert_eq!(DenseForm::parse(Some("bf16")), Ok(DenseForm::Bf16));
        let refused = DenseForm::parse(Some("fp8")).unwrap_err();
        assert!(refused.contains(DENSE_ENV), "{refused}");
    }
}
