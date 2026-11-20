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

//! Inkling's dense projections in the shared packed BF16 form ([sparkflash_ops::packed]).
//!
//! A decode step reads every attention projection, the router, the dynamic taps and the head,
//! about 50 MB a layer and 462 MB of head a rank, and those GEMVs run at the DRAM floor; packed,
//! they are the same bits from about three quarters of the bytes. Inkling packs in place (a
//! million-token context leaves no room for both forms), so a prompt's tensor-core tiles unpack a
//! projection into the set's scratch first.

use std::collections::BTreeSet;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_ops::calls::Call;
use sparkflash_ops::packed::{self, Form, Packed, PackedSet, Placement, Tiled, Walk, packable};
use sparkflash_weights::DType;

use super::{Entry, Kernels, Shared, Weights};
use crate::device_plan::{InklingDevicePlan, Op, Projection, Target};

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

/// Packs every dense projection the form takes, in place, or none under [Form::Checkpoint], and
/// the head's tiled packed copy beside the head's walk one.
///
/// The returned pair is (the dense set, the head's tiled set). The head's tiled copy is packed
/// first, while its BF16 rows are still there to verify against, and the in-place walk pack then
/// overwrites them. The head keeps the walk copy because the chunk's wider head walk is the row
/// GEMV's (a shape the few-row body does not cover), and that copy is what the GEMV reads.
///
/// # Errors
/// A driver failure, or a matrix whose packed rows do not unpack to its BF16 rows.
pub(super) fn pack_dense(
    cuda: &Cuda,
    kernels: &Kernels,
    plans: &[&InklingDevicePlan],
    weights: &Weights,
    form: Form,
) -> Result<(PackedSet<Walk>, PackedSet<Tiled>), String> {
    let mut dense = PackedSet::new(form, Placement::InPlace);
    let mut head = PackedSet::new(form, Placement::Beside).tiled();
    if form == Form::Checkpoint || plans.is_empty() {
        return Ok((dense, head));
    }
    let served = plans[0].served();
    let (slots, head_slot) = dense_slots(plans);
    let mut candidates = Vec::new();
    // The head's own entry in [dense_slots]: the epilogue's `(Op::Gemv, Target::Project(
    // Projection::Head))` step's first weight, the unembed/logit matrix
    // (`model.llm.unembed.weight`), never a matrix index or a size guess.
    let mut head_matrix = None;
    // The scratch holds the widest matrix but the head as BF16 (what a prompt tile unpacks) and
    // is what the load packs through, a group of rows at a time.
    let mut scratch_bytes = 0usize;
    for index in slots {
        let spec = served.rank_spec_at(index);
        let slot = weights.get(index);
        let [rows, columns] = spec.shape[..] else {
            continue;
        };
        if spec.dtype != DType::BF16 || slot.bytes != rows * columns * 2 {
            continue;
        }
        if Some(index) == head_slot && packable(rows, columns) {
            head_matrix = Some((slot.pointer, rows, columns));
        }
        if Some(index) != head_slot && packable(rows, columns) {
            scratch_bytes = scratch_bytes.max(rows * columns * 2);
        }
        candidates.push((slot.pointer, rows, columns));
    }
    // Beside the BF16 rows, so read and verify them before the in-place pack replaces them.
    if let Some(matrix) = head_matrix {
        head.pack(cuda, &kernels.packed, &[matrix], 0)?;
    }
    if let Err(error) = dense.pack(cuda, &kernels.packed, &candidates, scratch_bytes) {
        // A set frees a failed pack's own allocations, but not one an earlier call left in it.
        let held: Vec<Deviceptr> = head.allocations().collect();
        let refs: Vec<&Deviceptr> = held.iter().collect();
        cuda.free_buffers(&refs);
        return Err(error);
    }
    Ok((dense, head))
}

impl Shared {
    fn launch_call<const N: usize>(&self, kernel: Entry, call: &Call<N>) -> Result<(), String> {
        self.launch(kernel, call.grid, call.block, &call.args)
    }

    /// A packed matrix's projection of `groups` groups of `batch` rows: q4e_gemv_bf16's walk and
    /// bits ([Shared::gemv_batch]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemv_packed(
        &self,
        matrix: Packed<Walk>,
        input: Deviceptr,
        out: Deviceptr,
        rows: usize,
        columns: usize,
        batch: usize,
        groups: u32,
    ) -> Result<(), String> {
        let call = packed::gemv(matrix, input, out, (rows, columns), batch, groups)?;
        self.launch_call(
            Entry {
                kernel: self.kernels.packed.gemv,
                precision: sparkflash_ops::plan::Precision::Bf16,
            },
            &call,
        )
    }

    /// Up to four packed matrices over `batch` rows, as one launch: q4e_gemv4_bf16's walk and
    /// bits. `parts` holds (matrix, output, rows).
    pub(super) fn gemv4_packed(
        &self,
        parts: &[(Packed<Walk>, Deviceptr, usize)],
        input: Deviceptr,
        columns: usize,
        batch: usize,
    ) -> Result<(), String> {
        let call = packed::gemv4(parts, input, columns, batch)?;
        self.launch_call(
            Entry {
                kernel: self.kernels.packed.gemv4,
                precision: sparkflash_ops::plan::Precision::Bf16,
            },
            &call,
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
        let (unpack, rows_at) = self.dense.bf16_rows(weight, rows, columns, 0)?;
        if let Some(call) = unpack {
            self.launch_call(
                Entry {
                    kernel: self.kernels.packed.unpack,
                    precision: sparkflash_ops::plan::Precision::Bf16,
                },
                &call,
            )?;
        }
        Ok(rows_at)
    }
}
