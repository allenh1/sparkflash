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

//! The two-shot all-reduce both tensor-parallel arms run.

use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::BackendError;
use sparkflash_weights::tp::TpRuntime;

/// Whether count floats split over world ranks into whole float4 chunks.
#[must_use]
pub fn splits_two_shot(count: usize, world: usize) -> bool {
    world > 1 && count.is_multiple_of(4 * world)
}

/// The two-shot device all-reduce of count floats at partial, in place.
///
/// Every rank's chunk of each peer's partial lands in slots, sum adds them into this rank's own
/// chunk, and the summed chunks are gathered back in place, half the one-shot's bytes.
///
/// # Errors
/// A collective failure, or whatever sum reports.
pub fn reduce_two_shot(
    tp: &TpRuntime,
    stream: u64,
    partial: Deviceptr,
    slots: Deviceptr,
    count: usize,
    sum: impl FnOnce(Deviceptr, usize) -> Result<(), BackendError>,
) -> Result<(), BackendError> {
    let chunk = count / tp.world();
    let mine = reduce_scatter(tp, stream, partial, slots, count, sum)?;
    gather_in_place(tp, stream, partial, mine, chunk * 4)
}

/// The two-shot all-reduce's first half: this rank's chunk summed over the world in place.
///
/// # Errors
/// A collective failure, or whatever sum reports.
pub fn reduce_scatter(
    tp: &TpRuntime,
    stream: u64,
    partial: Deviceptr,
    slots: Deviceptr,
    count: usize,
    sum: impl FnOnce(Deviceptr, usize) -> Result<(), BackendError>,
) -> Result<Deviceptr, BackendError> {
    let chunk = count / tp.world();
    let mine = partial.offset(tp.rank() * chunk * 4);
    tp.scatter_on_device(stream, partial.0, slots.0, count * 4)
        .map_err(collective_error)?;
    sum(mine, chunk)?;
    Ok(mine)
}

fn gather_in_place(
    tp: &TpRuntime,
    stream: u64,
    base: Deviceptr,
    mine: Deviceptr,
    bytes: usize,
) -> Result<(), BackendError> {
    debug_assert_eq!(mine, base.offset(tp.rank() * bytes));
    tp.exchange_on_device(stream, mine.0, base.0, bytes)
        .map_err(collective_error)
}

fn collective_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::new("tp", error.to_string())
}
