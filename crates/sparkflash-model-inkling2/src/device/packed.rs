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
use sparkflash_ops::packed::{self, Form, Packed, PackedSet, Placement, packable};
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

/// Packs every dense projection the form takes, in place, or none under [Form::Checkpoint].
///
/// # Errors
/// A driver failure, or a matrix whose packed rows do not unpack to its BF16 rows.
pub(super) fn pack_dense(
    cuda: &Cuda,
    kernels: &Kernels,
    plans: &[&InklingDevicePlan],
    weights: &Weights,
    form: Form,
) -> Result<PackedSet, String> {
    let mut dense = PackedSet::new(form, Placement::InPlace);
    if form == Form::Checkpoint || plans.is_empty() {
        return Ok(dense);
    }
    let served = plans[0].served();
    let (slots, head) = dense_slots(plans);
    let mut candidates = Vec::new();
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
        if Some(index) != head && packable(rows, columns) {
            scratch_bytes = scratch_bytes.max(rows * columns * 2);
        }
        candidates.push((slot.pointer, rows, columns));
    }
    dense.pack(cuda, &kernels.packed, &candidates, scratch_bytes)?;
    Ok(dense)
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
        matrix: Packed,
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
            },
            &call,
        )
    }

    /// Up to four packed matrices over `batch` rows, as one launch: q4e_gemv4_bf16's walk and
    /// bits. `parts` holds (matrix, output, rows).
    pub(super) fn gemv4_packed(
        &self,
        parts: &[(Packed, Deviceptr, usize)],
        input: Deviceptr,
        columns: usize,
        batch: usize,
    ) -> Result<(), String> {
        let call = packed::gemv4(parts, input, columns, batch)?;
        self.launch_call(
            Entry {
                kernel: self.kernels.packed.gemv4,
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
                },
                &call,
            )?;
        }
        Ok(rows_at)
    }
}
