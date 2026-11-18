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

//! Dense BF16 projections in the packed form (kernels/bf16_packed_sm121.cuh), for every arm whose
//! decode GEMVs walk `q4e_gemv_bf16` / `q4e_gemv4_bf16` (Inkling, Dory, the dense arm), in the
//! row walk's [Layout], or a tiled reader (GLM's `q4e_gemv_tiled_packed_parts`, Qwen's few-row
//! tensor-core tile), in the tiled one: the same bytes and bits, ordered for the reader.
//!
//! A decode step's dense GEMVs run at the DRAM floor. The packed form holds the same BF16 words
//! in about three quarters of the bytes: a sign and mantissa byte a weight, a 4-bit exponent
//! offset under the largest of the 32 weights one lane reads at a step, and that largest. The
//! packed GEMVs rebuild the words in registers and add them exactly as the BF16 walks do, so a
//! packed projection is the BF16 projection's bits from 77% of the bytes.
//!
//! Every matrix is packed at load and unpacked again and compared with the BF16 rows it stands
//! for; a single differing word refuses the load. [Placement] chooses where the packed rows go:
//! over the BF16 rows ([Placement::InPlace], a prompt tile then unpacks into a scratch first) or
//! beside them ([Placement::Beside], for an arm whose wide or tensor-core walks keep reading the
//! BF16 rows, at the price of the packed bytes on top).

use std::collections::HashMap;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::qwen4_exp_cubin;

use crate::calls::{BLOCK, Call};

/// How every arm that packs stores its dense BF16 projections: `packed` (the default, the
/// lossless packed form) or `checkpoint` (the checkpoint's BF16 rows as they are). Anything else
/// is refused.
pub const FORM_ENV: &str = "SPARKFLASH_DENSE_WEIGHTS";
/// The switch's earlier names (Inkling's own, then the shared `=packed|bf16` one); refused with a
/// pointer to [FORM_ENV] rather than read, so an old script cannot think it chose a form.
const RETIRED_ENVS: [&str; 2] = ["SPARKFLASH_INKLING_DENSE", "SPARKFLASH_BF16_FORM"];

/// How the dense projections are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// Packed at load, every matrix whose shape the form takes.
    Packed,
    /// As the checkpoint stores them.
    Checkpoint,
}

impl Form {
    /// The form [FORM_ENV] names.
    ///
    /// # Errors
    /// An unknown form, or a retired name of the switch set.
    pub fn from_env() -> Result<Self, String> {
        for retired in RETIRED_ENVS {
            if let Ok(value) = std::env::var(retired) {
                return Err(format!(
                    "{retired}={value} is retired: set {FORM_ENV}=packed or \
                     {FORM_ENV}=checkpoint instead"
                ));
            }
        }
        Self::parse(std::env::var(FORM_ENV).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("" | "packed") => Ok(Self::Packed),
            Some("checkpoint") => Ok(Self::Checkpoint),
            Some(other) => Err(format!(
                "{FORM_ENV}={other} is not a dense form: use packed (the default: the dense \
                 projections in a lossless 12.25-bit form, the BF16 bits from three quarters of \
                 the bytes) or checkpoint (the checkpoint's BF16 rows as they are)"
            )),
        }
    }
}

/// Where a packed matrix's rows live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Over the BF16 rows they replace: no extra bytes, and a walk that needs BF16 rows unpacks
    /// them into the set's scratch ([PackedSet::bf16_rows]).
    InPlace,
    /// In an arena of their own, the BF16 rows kept for the walks that read them.
    Beside,
}

/// How a packed matrix orders a step's plane bytes (kernels/bf16_packed_sm121.cuh): for the row
/// walk (`q4e_gemv_packed`, `q4e_gemv4_packed`) or for a tiled reader. The two hold the same bytes
/// and bits; a matrix is in one, and a reader of the other kind refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// A lane's chunks of a step together, as q4e_row_dot_bf16 reads them.
    Walk,
    /// A step's chunks in order, 32 lanes' a round, as the tiled readers take them.
    Tiled,
}

/// The form's four kernels, all in the Qwen4Exp image beside `q4e_gemv_bf16`.
#[derive(Debug, Clone, Copy)]
pub struct Kernels {
    pub pack: Kernel,
    pub unpack: Kernel,
    pub gemv: Kernel,
    pub gemv4: Kernel,
}

impl Kernels {
    /// Loads the four kernels.
    ///
    /// # Errors
    /// A build without the image, or a missing symbol.
    pub fn load(cuda: &Cuda) -> Result<Self, String> {
        let image = qwen4_exp_cubin();
        if image.is_empty() {
            return Err("this build has no Qwen4Exp cubin for the packed BF16 kernels".to_owned());
        }
        let named = |name| cuda.load_named(image, name).map(|(_, kernel)| kernel);
        Ok(Self {
            pack: named("q4e_pack_bf16")?,
            unpack: named("q4e_unpack_bf16")?,
            gemv: named("q4e_gemv_packed")?,
            gemv4: named("q4e_gemv4_packed")?,
        })
    }
}

/// A packed row's columns are whole rounds of 32 lanes' 16-byte chunks, so every lane of the
/// walk takes the same steps.
const PACKED_MULTIPLE: usize = 256;
/// An escape stores its column in 16 bits.
const PACKED_COLUMNS: usize = 1 << 16;
/// A packed row, its planes and the activations it meets are 16-byte loads.
const ALIGN: usize = 16;
/// The smallest scratch an in-place set packs through, when the widest matrix is smaller.
const MIN_SCRATCH: usize = 1 << 20;
/// The matrices one `q4e_gemv4_packed` launch stacks.
pub const FUSED: usize = 4;

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
#[must_use]
pub fn packable(rows: usize, columns: usize) -> bool {
    rows > 0 && columns.is_multiple_of(PACKED_MULTIPLE) && columns < PACKED_COLUMNS
}

/// The grid a packed walk takes: one row a warp of a [BLOCK]-thread block, the kernels striding
/// past 4,096 blocks.
fn grid(rows: usize) -> u32 {
    let warps = BLOCK as usize / 32;
    u32::try_from(rows.div_ceil(warps).max(1))
        .unwrap_or(u32::MAX)
        .min(4_096)
}

/// One packed matrix: its rows, and its escapes.
#[derive(Debug, Clone, Copy)]
pub struct Packed {
    pointer: Deviceptr,
    offsets: Deviceptr,
    entries: Deviceptr,
    rows: usize,
    columns: usize,
    row_bytes: usize,
    layout: Layout,
}

impl Packed {
    /// The order of the matrix's plane bytes, which its reader must take.
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Refuses a reader of the other layout's kind.
    fn read_as(&self, layout: Layout) -> Result<(), String> {
        if self.layout == layout {
            return Ok(());
        }
        Err(format!(
            "a {:?}-layout reader over a packed {} x {} matrix in the {:?} layout",
            layout, self.rows, self.columns, self.layout
        ))
    }

    /// The pointers a packed walk takes for this matrix, in its argument order: the packed rows,
    /// the row offsets into the escapes, and the escapes.
    #[must_use]
    pub fn planes(&self) -> [u64; 3] {
        [self.pointer.0, self.offsets.0, self.entries.0]
    }

    /// The stride from one packed row to the next.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The packed rows' bytes, which a pass over the matrix reads: a prefetch span.
    #[must_use]
    pub fn span(&self) -> (Deviceptr, usize) {
        (self.pointer, self.rows * self.row_bytes)
    }

    /// Refuses a launch the packed walk cannot take: other columns than the matrix's, more rows
    /// than it has, or activations that are not 16-byte loads (the BF16 walk falls back to a
    /// column walk there; the packed one has none).
    ///
    /// # Errors
    /// The launch the matrix does not admit, named.
    pub fn admits(&self, rows: usize, columns: usize, input: Deviceptr) -> Result<(), String> {
        if columns != self.columns || rows > self.rows {
            return Err(format!(
                "a {rows} x {columns} projection over a packed {} x {} matrix",
                self.rows, self.columns
            ));
        }
        if !(input.0 as usize).is_multiple_of(ALIGN) {
            return Err(format!(
                "a packed projection reads its activations as 16-byte loads, and {:#x} is not \
                 one ({FORM_ENV}=checkpoint keeps the BF16 walk)",
                input.0
            ));
        }
        Ok(())
    }
}

/// `q4e_gemv_packed`: `q4e_gemv_bf16`'s walk and bits over a packed matrix, for `groups` groups
/// of `batch` rows along the grid's y.
///
/// # Errors
/// A launch the matrix does not admit.
pub fn gemv(
    packed: Packed,
    input: Deviceptr,
    output: Deviceptr,
    (rows, columns): (usize, usize),
    batch: usize,
    groups: u32,
) -> Result<Call<9>, String> {
    packed.read_as(Layout::Walk)?;
    packed.admits(rows, columns, input)?;
    Ok(Call {
        grid: (grid(rows), groups, 1),
        block: (BLOCK, 1, 1),
        args: [
            packed.pointer.0,
            packed.offsets.0,
            packed.entries.0,
            input.0,
            output.0,
            rows as u64,
            columns as u64,
            batch as u64,
            packed.row_bytes as u64,
        ],
    })
}

/// `q4e_gemv4_packed`: up to [FUSED] packed matrices over one input as one launch,
/// `q4e_gemv4_bf16`'s walk and bits. `parts` holds (matrix, output, rows).
///
/// # Errors
/// No parts or more than [FUSED], or a launch a matrix does not admit.
pub fn gemv4(
    parts: &[(Packed, Deviceptr, usize)],
    input: Deviceptr,
    columns: usize,
    batch: usize,
) -> Result<Call<24>, String> {
    if parts.is_empty() || parts.len() > FUSED {
        return Err(format!(
            "a fused projection takes 1 to {FUSED} matrices, not {}",
            parts.len()
        ));
    }
    // (w0..w3, o0..o3, e0..e3, x, y0..y3, r0..r3, columns, batch, row_bytes)
    let mut args = [0u64; 5 * FUSED + 4];
    for (slot, (packed, output, rows)) in parts.iter().enumerate() {
        packed.read_as(Layout::Walk)?;
        packed.admits(*rows, columns, input)?;
        args[slot] = packed.pointer.0;
        args[FUSED + slot] = packed.offsets.0;
        args[2 * FUSED + slot] = packed.entries.0;
        args[3 * FUSED + 1 + slot] = output.0;
        args[4 * FUSED + 1 + slot] = *rows as u64;
    }
    args[3 * FUSED] = input.0;
    args[5 * FUSED + 1] = columns as u64;
    args[5 * FUSED + 2] = batch as u64;
    args[5 * FUSED + 3] = parts[0].0.row_bytes as u64;
    let total: usize = parts.iter().map(|(_, _, rows)| rows).sum();
    Ok(Call {
        grid: (grid(total), 1, 1),
        block: (BLOCK, 1, 1),
        args,
    })
}

/// `q4e_unpack_bf16` of a packed matrix's first `rows` rows into `output`, or, with `output`
/// null, compared with `source`, every differing word counted at `differ`.
fn unpack(
    packed: &Packed,
    output: Deviceptr,
    (source, differ): (Deviceptr, Deviceptr),
    rows: usize,
) -> Call<10> {
    Call {
        grid: (grid(rows), 1, 1),
        block: (BLOCK, 1, 1),
        args: [
            packed.pointer.0,
            packed.offsets.0,
            packed.entries.0,
            output.0,
            source.0,
            differ.0,
            rows as u64,
            packed.columns as u64,
            packed.row_bytes as u64,
            u64::from(packed.layout == Layout::Tiled),
        ],
    }
}

/// A set of packed matrices, by the device pointer of the BF16 rows each stands for.
pub struct PackedSet {
    packed: HashMap<u64, Packed>,
    /// The form [FORM_ENV] chose, which every later [Self::pack] keeps to.
    form: Form,
    placement: Placement,
    /// The layout every matrix of the set is packed in.
    layout: Layout,
    /// Each [Self::pack] call's offsets, escapes and (beside the rows) packed rows.
    arenas: Vec<Deviceptr>,
    /// In place only: a BF16 matrix's worth, for the walks that read BF16 rows, and what the
    /// load packs through.
    scratch: Option<(Deviceptr, usize)>,
    /// The BF16 bytes the packed matrices stand for, and their packed bytes.
    bf16_bytes: usize,
    packed_bytes: usize,
}

impl PackedSet {
    /// An empty set.
    #[must_use]
    pub fn new(form: Form, placement: Placement) -> Self {
        Self {
            packed: HashMap::new(),
            form,
            placement,
            layout: Layout::Walk,
            arenas: Vec::new(),
            scratch: None,
            bf16_bytes: 0,
            packed_bytes: 0,
        }
    }

    /// The set in the tiled layout, for an arm whose packed reader is tiled. Set before the first
    /// [Self::pack].
    #[must_use]
    pub fn tiled(mut self) -> Self {
        debug_assert!(
            self.packed.is_empty(),
            "a set changes layout before it packs"
        );
        self.layout = Layout::Tiled;
        self
    }

    /// The packed matrix whose BF16 rows start at `weight`, if it is one.
    #[must_use]
    pub fn get(&self, weight: Deviceptr) -> Option<Packed> {
        self.packed.get(&weight.0).copied()
    }

    /// The allocations to free.
    pub fn allocations(&self) -> impl Iterator<Item = Deviceptr> + '_ {
        self.arenas
            .iter()
            .copied()
            .chain(self.scratch.map(|(pointer, _)| pointer))
    }

    /// Packs `matrices`, (rows, row count, columns) BF16 slots, and adds them: the ones the form
    /// takes, none under [Form::Checkpoint]. In place, the slots must be ones only the GEMV walks and
    /// [Self::bf16_rows] read; the first call allocates the scratch at `scratch_bytes` (at least
    /// [MIN_SCRATCH]) and later ones pack through it. Beside the rows, `scratch_bytes` is unused.
    /// On an error this call's allocations are freed and the set is as it was.
    ///
    /// Returns the BF16 bytes the matrices packed here stand for and their packed bytes.
    ///
    /// # Errors
    /// A driver failure, a slot already packed, or a matrix whose packed rows do not unpack to
    /// its BF16 rows.
    pub fn pack(
        &mut self,
        cuda: &Cuda,
        kernels: &Kernels,
        matrices: &[(Deviceptr, usize, usize)],
        scratch_bytes: usize,
    ) -> Result<(usize, usize), String> {
        if self.form == Form::Checkpoint {
            return Ok((0, 0));
        }
        let mut candidates = Vec::new();
        for &(pointer, rows, columns) in matrices {
            if !packable(rows, columns) || !(pointer.0 as usize).is_multiple_of(ALIGN) {
                continue;
            }
            if self.packed.contains_key(&pointer.0) {
                return Err(format!(
                    "the dense slot at {:#x} is packed twice",
                    pointer.0
                ));
            }
            candidates.push(Candidate {
                pointer,
                rows,
                columns,
                offsets: Vec::new(),
            });
        }
        if candidates.is_empty() {
            return Ok((0, 0));
        }
        let mut total = 0usize;
        let fresh = self.placement == Placement::InPlace && self.scratch.is_none();
        let scratch = match (self.placement, self.scratch) {
            (Placement::Beside, _) => None,
            (Placement::InPlace, Some(scratch)) => Some(scratch),
            (Placement::InPlace, None) => {
                let bytes = scratch_bytes.max(MIN_SCRATCH);
                Some((cuda.alloc(bytes, &mut total)?, bytes))
            }
        };
        let release = |held: &[Deviceptr]| cuda.free_buffers(&held.iter().collect::<Vec<_>>());
        let widest_rows = candidates.iter().map(|m| m.rows).max().unwrap_or(1);
        let held = cuda.alloc(widest_rows * 4, &mut total).and_then(|counts| {
            cuda.alloc(ALIGN, &mut total)
                .map(|differ| (counts, differ))
                .inspect_err(|_| release(&[counts]))
        });
        let (counts, differ) = match held {
            Ok(held) => held,
            Err(error) => {
                if fresh && let Some((scratch, _)) = scratch {
                    release(&[scratch]);
                }
                return Err(error);
            }
        };
        let mut arena = None;
        let packed = pack_matrices(
            cuda,
            kernels,
            self.layout,
            &mut candidates,
            (counts, differ),
            scratch,
            &mut arena,
        );
        release(&[counts, differ]);
        match packed {
            Ok((packed, bf16_bytes, packed_bytes)) => {
                self.packed.extend(packed);
                self.arenas.extend(arena);
                if self.placement == Placement::InPlace {
                    self.scratch = scratch;
                }
                self.bf16_bytes += bf16_bytes;
                self.packed_bytes += packed_bytes;
                Ok((bf16_bytes, packed_bytes))
            }
            Err(error) => {
                let mut held: Vec<Deviceptr> = arena.into_iter().collect();
                if fresh && let Some((scratch, _)) = scratch {
                    held.push(scratch);
                }
                release(&held);
                Err(error)
            }
        }
    }

    /// The BF16 rows a wide or tensor-core walk reads for `weight`'s first `rows` rows: the
    /// matrix itself when it is not packed or its BF16 rows stayed, else the launch that unpacks
    /// them into the scratch at `offset` bytes and where they land. The caller launches it on the
    /// stream that then reads the scratch, and two walks that share the scratch at once take
    /// different offsets.
    ///
    /// # Errors
    /// Other columns than the matrix's, more rows than it has, or a scratch too small.
    pub fn bf16_rows(
        &self,
        weight: Deviceptr,
        rows: usize,
        columns: usize,
        offset: usize,
    ) -> Result<(Option<Call<10>>, Deviceptr), String> {
        let Some(packed) = self
            .get(weight)
            .filter(|_| self.placement == Placement::InPlace)
        else {
            return Ok((None, weight));
        };
        let (scratch, bytes) = self
            .scratch
            .ok_or("a packed matrix with no scratch to unpack it into")?;
        if columns != packed.columns
            || rows > packed.rows
            || offset + rows * columns * 2 > bytes
            || !offset.is_multiple_of(ALIGN)
        {
            return Err(format!(
                "a walk over {rows} x {columns} BF16 rows of a packed {} x {} matrix at byte \
                 {offset} of a {bytes}-byte scratch ({FORM_ENV}=checkpoint keeps the BF16 rows)",
                packed.rows, packed.columns
            ));
        }
        let output = scratch.offset(offset);
        Ok((
            Some(unpack(&packed, output, (Deviceptr(0), Deviceptr(0)), rows)),
            output,
        ))
    }

    /// One line for the boot log.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.packed.is_empty() {
            return format!("dense projections in BF16 ({FORM_ENV}=checkpoint or none packable)");
        }
        let place = match self.placement {
            Placement::InPlace => "in place",
            Placement::Beside => "beside the BF16 rows",
        };
        format!(
            "{} dense projections packed {place}: {:.1} -> {:.1} MB ({:.1}% fewer bytes a pass)",
            self.packed.len(),
            self.bf16_bytes as f64 / 1e6,
            self.packed_bytes as f64 / 1e6,
            100.0 * (1.0 - self.packed_bytes as f64 / self.bf16_bytes.max(1) as f64)
        )
    }
}

/// One matrix to pack.
struct Candidate {
    pointer: Deviceptr,
    rows: usize,
    columns: usize,
    offsets: Vec<u32>,
}

/// [PackedSet::pack]'s passes: count every matrix's escapes, lay out one arena (a matrix's
/// offsets, then its escapes, then beside the rows its packed rows, each a 16-byte run), then pack
/// each matrix and check it against the BF16 rows it stands for. In place, `scratch` is set and a
/// matrix packs a group of rows at a time through it, each group checked before its BF16 rows are
/// overwritten. Returns the packed matrices, the BF16 bytes they stand for and their packed bytes;
/// `arena` is set as soon as it is allocated, so the caller frees it on an error.
fn pack_matrices(
    cuda: &Cuda,
    kernels: &Kernels,
    layout: Layout,
    matrices: &mut [Candidate],
    (counts, differ): (Deviceptr, Deviceptr),
    scratch: Option<(Deviceptr, usize)>,
    arena: &mut Option<Deviceptr>,
) -> Result<(HashMap<u64, Packed>, usize, usize), String> {
    let launch = |kernel: Kernel, grid: (u32, u32, u32), args: &[u64]| -> Result<(), String> {
        cuda.launch(kernel, grid, (BLOCK, 1, 1), args)
            .map_err(|error| format!("{}: {error}", kernel.name()))
    };
    // q4e_pack_bf16: counts with `packed` null, else the packed rows and escapes.
    let pack = |source: Deviceptr, packed: &Packed, counts: Deviceptr, rows: usize| {
        let args = [
            source.0,
            packed.pointer.0,
            packed.offsets.0,
            packed.entries.0,
            counts.0,
            rows as u64,
            packed.columns as u64,
            packed.row_bytes as u64,
            u64::from(packed.layout == Layout::Tiled),
        ];
        launch(kernels.pack, (grid(rows), 1, 1), &args)
    };
    let check = |call: Call<10>| launch(kernels.unpack, call.grid, &call.args);
    let mut arena_bytes = 0usize;
    for matrix in matrices.iter_mut() {
        let counting = Packed {
            pointer: Deviceptr(0),
            offsets: Deviceptr(0),
            entries: Deviceptr(0),
            rows: matrix.rows,
            columns: matrix.columns,
            row_bytes: row_bytes(matrix.columns),
            layout,
        };
        pack(matrix.pointer, &counting, counts, matrix.rows)?;
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
        if scratch.is_none() {
            arena_bytes += matrix.rows * row_bytes(matrix.columns);
        }
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
        let mut done = Packed {
            pointer: matrix.pointer,
            offsets,
            entries,
            rows: matrix.rows,
            columns: matrix.columns,
            row_bytes,
            layout,
        };
        match scratch {
            None => {
                done.pointer = base.offset(at);
                at += matrix.rows * row_bytes;
                pack(matrix.pointer, &done, Deviceptr(0), matrix.rows)?;
                check(unpack(
                    &done,
                    Deviceptr(0),
                    (matrix.pointer, differ),
                    matrix.rows,
                ))?;
            }
            Some((scratch, scratch_bytes)) => {
                // A group's packed rows end before the next group's BF16 rows begin, so copying a
                // group down never overwrites a row still to be read; the check reads the
                // group's BF16 rows before its copy does.
                let group = (scratch_bytes / row_bytes).max(1);
                let mut first = 0;
                while first < matrix.rows {
                    let take = group.min(matrix.rows - first);
                    let source = matrix.pointer.offset(first * matrix.columns * 2);
                    let part = Packed {
                        pointer: scratch,
                        offsets: offsets.offset(first * 4),
                        ..done
                    };
                    pack(source, &part, Deviceptr(0), take)?;
                    check(unpack(&part, Deviceptr(0), (source, differ), take))?;
                    cuda.copy_dtod(
                        &matrix.pointer.offset(first * row_bytes),
                        &scratch,
                        take * row_bytes,
                    )?;
                    first += take;
                }
            }
        }
        bf16_bytes += matrix.rows * matrix.columns * 2;
        packed_bytes += matrix.rows * row_bytes + matrix.offsets.len() * 4 + escapes * 4;
        packed.insert(matrix.pointer.0, done);
    }
    cuda.synchronize()?;
    let mut word = [0u8; 4];
    cuda.copy_dtoh(&mut word, &differ)?;
    let differ_words = u32::from_le_bytes(word);
    if differ_words != 0 {
        return Err(format!(
            "the packed dense projections unpack to {differ_words} BF16 words other than the \
             checkpoint's; the rank refuses to serve them ({FORM_ENV}=checkpoint keeps the BF16 rows)"
        ));
    }
    Ok((packed, bf16_bytes, packed_bytes))
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
        // Dory's 2,560: two quads and a pair.
        assert_eq!(walk_steps(2560), 3);
        assert_eq!(row_bytes(2560), 3936);
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
        assert_eq!(Form::parse(None), Ok(Form::Packed));
        assert_eq!(Form::parse(Some("")), Ok(Form::Packed));
        assert_eq!(Form::parse(Some("packed")), Ok(Form::Packed));
        assert_eq!(Form::parse(Some("checkpoint")), Ok(Form::Checkpoint));
        // The old value is not a quiet alias for either form.
        assert!(Form::parse(Some("bf16")).is_err());
        let refused = Form::parse(Some("fp8")).unwrap_err();
        assert!(refused.contains(FORM_ENV), "{refused}");
    }

    #[test]
    fn the_fused_call_lays_out_every_slot() {
        let matrix = |base: u64| Packed {
            pointer: Deviceptr(base),
            offsets: Deviceptr(base + 16),
            entries: Deviceptr(base + 32),
            rows: 64,
            columns: 512,
            row_bytes: row_bytes(512),
            layout: Layout::Walk,
        };
        let call = gemv4(
            &[
                (matrix(0x1000), Deviceptr(0x9000), 64),
                (matrix(0x2000), Deviceptr(0xa000), 32),
            ],
            Deviceptr(0x8000),
            512,
            3,
        )
        .unwrap();
        assert_eq!(&call.args[..2], &[0x1000, 0x2000]);
        assert_eq!(&call.args[4..6], &[0x1010, 0x2010]);
        assert_eq!(&call.args[8..10], &[0x1020, 0x2020]);
        assert_eq!(call.args[12], 0x8000);
        assert_eq!(&call.args[13..15], &[0x9000, 0xa000]);
        assert_eq!(&call.args[17..19], &[64, 32]);
        assert_eq!(&call.args[21..], &[512, 3, row_bytes(512) as u64]);
        assert_eq!(call.grid, (12, 1, 1));
        assert!(gemv4(&[], Deviceptr(0), 512, 1).is_err());
        // Activations off a 16-byte load are refused, not walked.
        assert!(
            gemv(
                matrix(0x1000),
                Deviceptr(0x8004),
                Deviceptr(0),
                (64, 512),
                1,
                1
            )
            .is_err()
        );
        // A tiled matrix is refused by the row walks.
        let tiled = Packed {
            layout: Layout::Tiled,
            ..matrix(0x1000)
        };
        assert!(gemv(tiled, Deviceptr(0x8000), Deviceptr(0), (64, 512), 1, 1).is_err());
        assert!(gemv4(&[(tiled, Deviceptr(0x9000), 64)], Deviceptr(0x8000), 512, 1).is_err());
    }
}
