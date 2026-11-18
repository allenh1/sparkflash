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

//! The two-shot all-reduce both tensor-parallel arms run, and the row sums a sharded arm takes
//! its branch partials through ([RowSums]).

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, RunStream};
use sparkflash_model::BackendError;
use sparkflash_weights::tp::{TpRuntime, Wire};

use crate::calls::{self, Call};

/// How one sum of a rank's partial rows over the world runs ([RowSums::leg]). Every leg adds
/// each element's partials in ascending rank order from 0.0, so the three give the same bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumLeg {
    /// Over the host seam: the partials copied back, summed by the collective and copied up.
    Host,
    /// The two-shot device all-reduce ([reduce_two_shot]), half the one-shot's bytes.
    TwoShot,
    /// One device exchange of every rank's partials into the slots, then `q4e_sum_slots`.
    OneShot,
}

/// The collective a sharded arm sums its row partials with: after a row-parallel projection
/// (an attention output, an MLP's down, an expert's) each rank holds a partial of every hidden
/// row, and the sum makes the residual the same bits on every rank, so every replicated decision
/// (the router's top-k, the token) agrees without a second collective.
///
/// The partials travel on the device when the collective has a device leg ([Wire::F32] only:
/// a leg that rounds another way is refused rather than summed as if it did not), two-shot from
/// [sparkflash_weights::tp::two_shot_rows] rows on and one-shot below; else over the host seam.
pub struct RowSums {
    runtime: TpRuntime,
    device_leg: bool,
    two_shot_rows: usize,
    /// Every peer's partial rows of one sum, `[world][rows][hidden]` floats, when the device leg
    /// gathers into them.
    slots: Option<Deviceptr>,
    rows: usize,
    hidden: usize,
}

impl RowSums {
    /// The sums over `runtime` of up to `rows` rows of `hidden` floats; `owner` names the arm in
    /// a refusal. The slots are the caller's to give back ([Self::release]).
    ///
    /// # Errors
    /// A device leg on another wire than [Wire::F32], a row the float4 sum cannot take, a
    /// `SPARKFLASH_TWO_SHOT_ROWS` that is not a count, or the slots allocation.
    pub fn new(
        cuda: &Cuda,
        runtime: TpRuntime,
        rows: usize,
        hidden: usize,
        owner: &'static str,
    ) -> Result<Self, BackendError> {
        let device_leg = match runtime.device_wire() {
            None => false,
            Some(Wire::F32) => true,
            Some(wire) => {
                return Err(BackendError::new(
                    owner,
                    format!(
                        "the collective's device leg sums on the {wire:?} wire; this arm sums only \
                         Wire::F32"
                    ),
                ));
            }
        };
        if device_leg && !hidden.is_multiple_of(4) {
            return Err(BackendError::new(
                owner,
                format!("a {hidden}-wide row is not whole float4s, which the device sum adds"),
            ));
        }
        let two_shot_rows = sparkflash_weights::tp::two_shot_rows()
            .map_err(|error| BackendError::new(owner, error.to_string()))?;
        let slots = if device_leg {
            Some(
                cuda.alloc(runtime.world() * rows * hidden * 4, &mut 0)
                    .map_err(|error| BackendError::new(owner, error))?,
            )
        } else {
            None
        };
        Ok(Self {
            runtime,
            device_leg,
            two_shot_rows,
            slots,
            rows,
            hidden,
        })
    }

    /// The collective.
    #[must_use]
    pub const fn runtime(&self) -> &TpRuntime {
        &self.runtime
    }

    /// Gives the slots back.
    pub fn release(&mut self, cuda: &Cuda) {
        if let Some(slots) = self.slots.take() {
            cuda.free_buffers(&[&slots]);
        }
    }

    /// The leg a sum of `rows` rows takes.
    ///
    /// # Errors
    /// More rows than the slots hold.
    pub fn leg(&self, rows: usize) -> Result<SumLeg, BackendError> {
        if !self.device_leg {
            return Ok(SumLeg::Host);
        }
        if rows > self.rows {
            return Err(BackendError::new(
                "tp",
                format!(
                    "a sum of {rows} rows is wider than the {} the collective's slots hold",
                    self.rows
                ),
            ));
        }
        Ok(
            if self.two_shot_rows > 0
                && rows >= self.two_shot_rows
                && splits_two_shot(rows * self.hidden, self.runtime.world())
            {
                SumLeg::TwoShot
            } else {
                SumLeg::OneShot
            },
        )
    }

    /// Sums `rows` rows of partials at `partial` over the world, in place, on `stream` (`None`:
    /// the context's null stream, which the host leg's copies wait on). `launch` queues the
    /// `q4e_sum_slots` call it is handed on the same stream.
    ///
    /// # Errors
    /// As [Self::leg], a collective or copy failure, or whatever `launch` reports.
    pub fn sum(
        &self,
        cuda: &Cuda,
        stream: Option<&RunStream>,
        partial: Deviceptr,
        rows: usize,
        launch: impl Fn(Call<5>) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        let count = rows * self.hidden;
        let (rank, world) = (self.runtime.rank(), self.runtime.world());
        let raw = stream.map_or(0, RunStream::raw);
        let device = |error: String| BackendError::new("tp", error);
        match (self.leg(rows)?, self.slots) {
            (SumLeg::Host, _) => {
                let mut bytes = vec![0u8; count * 4];
                match stream {
                    Some(stream) => cuda.copy_dtoh_on(stream, &mut bytes, &partial),
                    None => cuda.copy_dtoh(&mut bytes, &partial),
                }
                .map_err(device)?;
                let mut values: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect();
                self.runtime
                    .all_reduce(&mut values)
                    .map_err(collective_error)?;
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                match stream {
                    Some(stream) => cuda.copy_htod_on(stream, &partial, &bytes),
                    None => cuda.copy_htod(&partial, &bytes),
                }
                .map_err(device)
            }
            (SumLeg::TwoShot, Some(slots)) => {
                reduce_two_shot(&self.runtime, raw, partial, slots, count, |mine, chunk| {
                    launch(calls::sum_slots(mine.0, slots.0, chunk, rank, world))
                })
            }
            (SumLeg::OneShot, Some(slots)) => {
                self.runtime
                    .exchange_on_device(raw, partial.0, slots.0, count * 4)
                    .map_err(collective_error)?;
                launch(calls::sum_slots(partial.0, slots.0, count, rank, world))
            }
            (_, None) => Err(device("a device leg without its slots".to_owned())),
        }
    }
}

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
