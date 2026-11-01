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

//! The tensor-parallel seam every model pack shards through: a rank's place in the group, the
//! collective adapter, and the exact bit gather.
//!
//! The adapters live in the CLI over its transport; nothing here names one. A pack only sees f32
//! buffers and a rank/world pair, which is what keeps the pack crates comm-free. Which widths a
//! pack splits and where it reduces is the pack's business: [TpLayout::local_width] checks one
//! split, and [TpRuntime] carries the adapter through the `&self` forward paths.
//!
//! The arithmetic is unchanged from the V4.1 seam (`docs/archive/build/tp4-design.md` section 3):
//! this is a move, not a rewrite.

use std::sync::{Arc, Condvar, Mutex, PoisonError};

use crate::linalg::EngineError;
use crate::quant::bf16_round;

/// Where this process sits in the TP group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpLayout {
    /// This rank's index, `0..world_size`.
    pub rank: usize,
    /// The number of ranks the model is split over.
    pub world_size: usize,
}

impl TpLayout {
    /// Checks the pair.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when `world_size` is zero or `rank` is not below it.
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.world_size == 0 {
            return Err(EngineError::Shape {
                what: "TP layout".to_owned(),
                detail: "world_size is zero".to_owned(),
            });
        }
        if self.rank >= self.world_size {
            return Err(EngineError::Shape {
                what: "TP layout".to_owned(),
                detail: format!(
                    "rank {} is outside a world of {}",
                    self.rank, self.world_size
                ),
            });
        }
        Ok(())
    }

    /// The share of one sharded width this rank holds: `width / world_size`.
    ///
    /// `what` names the width in the error, e.g. `"moe_inter_dim"`.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the pair fails [TpLayout::validate], or `width` is zero or does
    /// not divide by the world.
    pub fn local_width(&self, what: &str, width: usize) -> Result<usize, EngineError> {
        self.validate()?;
        if width == 0 || !width.is_multiple_of(self.world_size) {
            return Err(EngineError::Shape {
                what: "TP layout".to_owned(),
                detail: format!(
                    "{what} {width} does not split over {} ranks",
                    self.world_size
                ),
            });
        }
        Ok(width / self.world_size)
    }
}

/// The engine's view of the TP group's collectives.
///
/// The implementation lives in the CLI (over its TCP transport); the engine never names a
/// transport. Deriving `Send` is required because the engine stores the adapter and its `&self`
/// forward paths run on several threads.
pub trait TpCollective: Send {
    /// The number of ranks.
    fn world(&self) -> usize;
    /// This rank's index.
    fn rank(&self) -> usize;
    /// A short label for evidence ledgers, e.g. `"single-rank identity"` or `"tcp rank 2/4"`.
    fn label(&self) -> &'static str;

    /// Reduces one f32 buffer across the group, in place.
    ///
    /// The wire format is canonical: every rank's values are rounded once to bf16 (ties to even),
    /// the f32 values are accumulated in ascending rank order from `0.0`, the sum is rounded once
    /// to bf16 (ties to even) and widened back to f32. With `world() == 1` this MUST be a strict
    /// no-op: no rounding and no write, because the one-rank engine is the bit-identical reference.
    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError>;

    /// Gathers `local.len()` values from every rank, in ascending rank order, into `out`.
    ///
    /// Each element is bf16-rounded on the wire and widened back to f32. With `world() == 1` this
    /// MUST be a plain copy of `local` into `out` and nothing else.
    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError>;

    /// Sends `values` to `to`.
    ///
    /// The reference seam has no send/recv leg yet; the default body refuses instead of silently
    /// succeeding. A transport adapter overrides it when the design grows one.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] from the default body.
    fn send(&mut self, _to: usize, _values: &[f32]) -> Result<(), EngineError> {
        Err(EngineError::Shape {
            what: "TpCollective::send".to_owned(),
            detail: "the reference seam has no send leg yet".to_owned(),
        })
    }

    /// Receives `out.len()` values from `from`.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] from the default body.
    fn recv(&mut self, _from: usize, _out: &mut [f32]) -> Result<(), EngineError> {
        Err(EngineError::Shape {
            what: "TpCollective::recv".to_owned(),
            detail: "the reference seam has no recv leg yet".to_owned(),
        })
    }

    /// The arithmetic of this collective's device leg ([Self::exchange_on_device]), or `None`
    /// when it has none and every payload goes through the host.
    fn device_wire(&self) -> Option<Wire> {
        None
    }

    /// Enqueues on `stream` the copy of this rank's `bytes` at the device address `send` into
    /// every peer's `slots + rank * bytes`, and every peer's into this rank's
    /// `slots + peer * bytes`. This rank's own slot is not written, and nothing is synchronised:
    /// the caller sums its own payload and the slots in ascending rank order on the same stream,
    /// in [Self::device_wire]'s arithmetic, so every rank gets the same bits without a host round
    /// trip.
    ///
    /// The addresses are device allocations the host never reads, as with the driver's own device
    /// copies: `send` holds `bytes` bytes and `slots` `world() * bytes` on the device current on
    /// this thread, `stream` is a stream there, and every rank enqueues the same exchanges in the
    /// same order. An adapter whose library cannot hold to that takes it as its own unsafe
    /// contract with the caller.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] from the default body, which has no device leg.
    fn exchange_on_device(
        &mut self,
        _stream: u64,
        _send: u64,
        _slots: u64,
        _bytes: usize,
    ) -> Result<(), EngineError> {
        Err(EngineError::Shape {
            what: "TpCollective::exchange_on_device".to_owned(),
            detail: "this collective has no device leg".to_owned(),
        })
    }
}

/// The one-rank adapter: every collective is a strict no-op, so the seam is value-transparent. It
/// is the one-rank [BitGather] too.
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleRank;

impl TpCollective for SingleRank {
    fn world(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        0
    }
    fn label(&self) -> &'static str {
        "single-rank identity"
    }
    fn all_reduce_assign(&mut self, _values: &mut [f32]) -> Result<(), EngineError> {
        Ok(())
    }
    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        if out.len() != local.len() {
            return Err(EngineError::Shape {
                what: "single-rank all_gather".to_owned(),
                detail: format!(
                    "one rank gathers {} values into {} slots",
                    local.len(),
                    out.len()
                ),
            });
        }
        out.copy_from_slice(local);
        Ok(())
    }
}

impl BitGather for SingleRank {
    fn world(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        0
    }
    fn label(&self) -> &'static str {
        "single-rank identity"
    }
    fn gather_bits(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        // One rank's gather is its copy, and so is its rounded gather.
        self.all_gather_assign(local, out)
    }
}

/// The canonical reduce over `world` payloads (including this rank's), for tests and any adapter
/// that wants the engine's exact reference arithmetic: [Wire::Bf16]'s [Wire::reduce_into].
///
/// # Errors
///
/// [EngineError::Shape] when `payloads` is empty or a payload is not `values.len()` long.
pub fn canonical_reduce_into(values: &mut [f32], payloads: &[&[f32]]) -> Result<(), EngineError> {
    Wire::Bf16.reduce_into(values, payloads)
}

/// The arithmetic a reduce applies to its rank payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    /// The transport's: each payload bf16-rounded, summed in f32 in ascending rank order from
    /// `0.0`, then rounded once to bf16 and widened back.
    Bf16,
    /// The same sum with no rounding at all, which a transport reaches by gathering every payload
    /// bit for bit and summing on each rank. It lets a test tell a sharding error, which survives
    /// it, from the bf16 wire's rounding, which does not.
    F32,
}

/// The relative-L2 bar a sharded forward over [Wire::F32] is held to against the one-rank forward,
/// per reduced seam and per token's logits. The exact wire leaves only the f32 reassociation of the
/// split matmuls, near 1e-6 on the packs' tiny fixtures; a wrong slice misses it by orders of
/// magnitude, where the bf16 wire's own rounding is already about 2e-3 at the first seam.
pub const EXACT_WIRE_REL_L2_BAR: f64 = 1e-5;

impl Wire {
    fn round(self, value: f32) -> f32 {
        match self {
            Self::Bf16 => bf16_round(value),
            Self::F32 => value,
        }
    }

    /// Reduce `payloads` into `values`. A single payload is returned unchanged: that is the world-1
    /// no-op the trait promises.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when `payloads` is empty or a payload is not `values.len()` long.
    pub fn reduce_into(self, values: &mut [f32], payloads: &[&[f32]]) -> Result<(), EngineError> {
        if payloads.is_empty() {
            return Err(EngineError::Shape {
                what: "TP reduce".to_owned(),
                detail: "no rank payloads".to_owned(),
            });
        }
        for payload in payloads {
            if payload.len() != values.len() {
                return Err(EngineError::Shape {
                    what: "TP reduce".to_owned(),
                    detail: format!(
                        "{} values in one payload, {} in the output",
                        payload.len(),
                        values.len()
                    ),
                });
            }
        }
        if payloads.len() == 1 {
            values.copy_from_slice(payloads[0]);
            return Ok(());
        }
        let mut accumulator = vec![0.0f32; values.len()];
        for payload in payloads {
            for (sum, value) in accumulator.iter_mut().zip(payload.iter()) {
                *sum += self.round(*value);
            }
        }
        for (out, sum) in values.iter_mut().zip(accumulator.iter()) {
            *out = self.round(*sum);
        }
        Ok(())
    }
}

/// The TP state an engine carries while it runs the sharded reference forward.
pub struct TpRuntime {
    layout: TpLayout,
    collective: Mutex<Box<dyn TpCollective>>,
}

impl TpRuntime {
    /// Validates the layout and the adapter against it.
    ///
    /// The widths a pack shards are its own to check, through [TpLayout::local_width], before or
    /// after this.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the layout is not a valid pair, or when the adapter reports a
    /// different rank/world pair.
    pub fn new(layout: TpLayout, collective: Box<dyn TpCollective>) -> Result<Self, EngineError> {
        layout.validate()?;
        if collective.world() != layout.world_size {
            return Err(EngineError::Shape {
                what: "TP collective".to_owned(),
                detail: format!(
                    "the adapter reports world {} but the layout says {}",
                    collective.world(),
                    layout.world_size
                ),
            });
        }
        if collective.rank() != layout.rank {
            return Err(EngineError::Shape {
                what: "TP collective".to_owned(),
                detail: format!(
                    "the adapter reports rank {} but the layout says {}",
                    collective.rank(),
                    layout.rank
                ),
            });
        }
        Ok(Self {
            layout,
            collective: Mutex::new(collective),
        })
    }

    /// This rank's index.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.layout.rank
    }

    /// The number of ranks.
    #[must_use]
    pub fn world(&self) -> usize {
        self.layout.world_size
    }

    /// The share of one sharded width this rank holds, as [TpLayout::local_width].
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when `width` does not split over the world.
    pub fn local_width(&self, what: &str, width: usize) -> Result<usize, EngineError> {
        self.layout.local_width(what, width)
    }

    /// The adapter's evidence label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .label()
    }

    /// Reduces one seam buffer in place through the adapter.
    ///
    /// # Errors
    ///
    /// Whatever the adapter reports, plus [EngineError::Shape] when the buffer is empty.
    pub fn all_reduce(&self, values: &mut [f32]) -> Result<(), EngineError> {
        if values.is_empty() {
            return Err(EngineError::Shape {
                what: "TP seam".to_owned(),
                detail: "an empty buffer has no seam to reduce".to_owned(),
            });
        }
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .all_reduce_assign(values)
    }

    /// Gathers one seam's local values through the adapter.
    ///
    /// # Errors
    ///
    /// Whatever the adapter reports.
    pub fn all_gather(&self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .all_gather_assign(local, out)
    }

    /// The adapter's [TpCollective::device_wire].
    #[must_use]
    pub fn device_wire(&self) -> Option<Wire> {
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .device_wire()
    }

    /// Enqueues one device exchange through the adapter, as [TpCollective::exchange_on_device].
    ///
    /// # Errors
    ///
    /// Whatever the adapter reports, plus [EngineError::Shape] for an empty payload.
    pub fn exchange_on_device(
        &self,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        if bytes == 0 {
            return Err(EngineError::Shape {
                what: "TP device exchange".to_owned(),
                detail: "an empty payload has nothing to exchange".to_owned(),
            });
        }
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .exchange_on_device(stream, send, slots, bytes)
    }
}

/// The engine's view of the exact f32 all-gather the row-sharded attention path needs.
///
/// This is a BYTE GATHER, not a numeric collective. Rank `r`'s `local` slice is copied into
/// `out[r * local.len() .. (r + 1) * local.len()]` with every bit preserved and no rounding, so
/// the gathered buffer is bit-identical to the replicated computation that produced the pieces.
/// The CLI rides it on the transport's bf16 payload channel by reinterpreting each f32 as its two
/// little-endian u16 halves, which is lossless. Never replace this with
/// [TpCollective::all_gather_assign]: that one rounds to bf16 on the wire by design and would
/// break bit identity, which is why the two gathers are separate traits.
pub trait BitGather: Send {
    /// The number of ranks.
    fn world(&self) -> usize;
    /// This rank's index.
    fn rank(&self) -> usize;
    /// A short label for evidence ledgers.
    fn label(&self) -> &'static str;

    /// Copies every rank's `local` into `out`, in ascending rank order.
    ///
    /// `out.len()` must be exactly `world() * local.len()`. With `world() == 1` this is a plain
    /// copy of `local` into `out` and nothing else.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the lengths disagree or the adapter reports another rank/world
    /// pair; otherwise whatever the adapter reports.
    fn gather_bits(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError>;
}

/// The engine's handle to the [BitGather] adapter: interior mutability so the `&self` forward
/// paths can gather, exactly as [TpRuntime] does for its reduce.
pub struct BitGatherHandle {
    inner: Mutex<Box<dyn BitGather>>,
}

impl BitGatherHandle {
    /// Wraps one adapter.
    pub fn new(inner: Box<dyn BitGather>) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Box<dyn BitGather>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The number of ranks.
    #[must_use]
    pub fn world(&self) -> usize {
        self.lock().world()
    }

    /// This rank's index.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.lock().rank()
    }

    /// The adapter's evidence label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.lock().label()
    }

    /// Gathers `local` from every rank into `out` in ascending rank order, bit-exactly.
    ///
    /// # Errors
    ///
    /// Whatever the adapter reports, plus [EngineError::Shape] when `out` is not `world * local`.
    pub fn gather_bits(&self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        let world = self.lock().world();
        let expected = world
            .checked_mul(local.len())
            .ok_or_else(|| EngineError::Shape {
                what: "bit gather".to_owned(),
                detail: format!("{} segments of {} elements overflow", world, local.len()),
            })?;
        if out.len() != expected {
            return Err(EngineError::Shape {
                what: "bit gather".to_owned(),
                detail: format!(
                    "{} ranks of {} elements gather into {expected} slots, got {}",
                    world,
                    local.len(),
                    out.len()
                ),
            });
        }
        self.lock().gather_bits(local, out)
    }
}

/// An in-process TP group: one [LocalRank] adapter per thread, all over one bus.
///
/// Every collective is a round: each rank contributes one payload and every rank leaves with all
/// of them in ascending rank order. The reduce applies the bus's [Wire] to them and the rounded
/// gather rounds each element the same way, so a local group on [Wire::Bf16], the default, does
/// the transport's arithmetic exactly; the bit gather concatenates them untouched. Every rank must issue the same collectives
/// in the same order, as the ranks of a real group do.
pub struct LocalBus {
    world: usize,
    wire: Wire,
    state: Mutex<Round>,
    ready: Condvar,
    /// Every reduce's output, per rank, when the bus records them.
    seams: Option<Vec<Mutex<Vec<Vec<f32>>>>>,
}

/// One round's slots, and its payloads while the ranks collect them.
struct Round {
    generation: u64,
    slots: Vec<Option<Vec<f32>>>,
    arrived: usize,
    payloads: Option<Arc<Vec<Vec<f32>>>>,
    uncollected: usize,
}

impl LocalBus {
    /// One bus for a world of ranks.
    #[must_use]
    pub fn new(world: usize) -> Self {
        Self {
            world,
            wire: Wire::Bf16,
            state: Mutex::new(Round {
                generation: 0,
                slots: vec![None; world],
                arrived: 0,
                payloads: None,
                uncollected: 0,
            }),
            ready: Condvar::new(),
            seams: None,
        }
    }

    /// A bus that also keeps every reduce's output, per rank, so a test can compare a sharded run
    /// with the one-rank run seam by seam ([LocalBus::seams]).
    #[must_use]
    pub fn recording(world: usize) -> Self {
        Self {
            seams: Some((0..world).map(|_| Mutex::new(Vec::new())).collect()),
            ..Self::new(world)
        }
    }

    /// The same bus over another [Wire].
    #[must_use]
    pub fn with_wire(self, wire: Wire) -> Self {
        Self { wire, ..self }
    }

    /// The number of ranks.
    #[must_use]
    pub fn world(&self) -> usize {
        self.world
    }

    /// Every reduce's output `rank` saw, in order; `None` when the bus does not record or the rank
    /// is outside the world.
    #[must_use]
    pub fn seams(&self, rank: usize) -> Option<Vec<Vec<f32>>> {
        let seams = self.seams.as_ref()?.get(rank)?;
        Some(seams.lock().unwrap_or_else(PoisonError::into_inner).clone())
    }

    fn check(&self, rank: usize) -> Result<(), EngineError> {
        if rank >= self.world {
            return Err(EngineError::Shape {
                what: "in-process bus".to_owned(),
                detail: format!("rank {rank} is outside a world of {}", self.world),
            });
        }
        Ok(())
    }

    /// Contributes `rank`'s payload to the next round and returns every rank's, in rank order.
    fn exchange(&self, rank: usize, local: &[f32]) -> Result<Arc<Vec<Vec<f32>>>, EngineError> {
        self.check(rank)?;
        let wait = |state| {
            self.ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner)
        };
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        // A rank that finished collecting the last round waits for the others to collect it too.
        while state.payloads.is_some() {
            state = wait(state);
        }
        let generation = state.generation;
        state.slots[rank] = Some(local.to_vec());
        state.arrived += 1;
        if state.arrived == self.world {
            let payloads = state
                .slots
                .iter_mut()
                .map(Option::take)
                .collect::<Option<_>>();
            state.payloads = Some(Arc::new(payloads.ok_or_else(|| EngineError::Shape {
                what: "in-process bus".to_owned(),
                detail: "two payloads from one rank in one round".to_owned(),
            })?));
            state.arrived = 0;
            state.uncollected = self.world;
            self.ready.notify_all();
        }
        while state.generation == generation && state.payloads.is_none() {
            state = wait(state);
        }
        let payloads = state
            .payloads
            .clone()
            .expect("a round's payloads stay until every rank collects them");
        state.uncollected -= 1;
        if state.uncollected == 0 {
            state.payloads = None;
            state.generation += 1;
            self.ready.notify_all();
        }
        Ok(payloads)
    }
}

/// One rank's adapter over a [LocalBus]: both its [TpCollective] and its [BitGather].
#[derive(Clone)]
pub struct LocalRank {
    /// This rank's index.
    pub rank: usize,
    /// The group's bus.
    pub bus: Arc<LocalBus>,
}

impl TpCollective for LocalRank {
    fn world(&self) -> usize {
        self.bus.world
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn label(&self) -> &'static str {
        "in-process bus"
    }
    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError> {
        self.bus.check(self.rank)?;
        // One rank's reduce is a strict no-op.
        if self.bus.world > 1 {
            let payloads = self.bus.exchange(self.rank, values)?;
            let payloads: Vec<&[f32]> = payloads.iter().map(Vec::as_slice).collect();
            self.bus.wire.reduce_into(values, &payloads)?;
        }
        if let Some(seams) = &self.bus.seams {
            seams[self.rank]
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(values.to_vec());
        }
        Ok(())
    }
    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.gather_bits(local, out)?;
        if self.bus.world > 1 {
            for value in out.iter_mut() {
                *value = self.bus.wire.round(*value);
            }
        }
        Ok(())
    }
}

impl BitGather for LocalRank {
    fn world(&self) -> usize {
        self.bus.world
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn label(&self) -> &'static str {
        "in-process bus"
    }
    fn gather_bits(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        if out.len() != self.bus.world * local.len() {
            return Err(EngineError::Shape {
                what: "in-process bus gather".to_owned(),
                detail: format!(
                    "{} ranks of {} elements gather into {} slots",
                    self.bus.world,
                    local.len(),
                    out.len()
                ),
            });
        }
        let payloads = self.bus.exchange(self.rank, local)?;
        for (rank, payload) in payloads.iter().enumerate() {
            if payload.len() != local.len() {
                return Err(EngineError::Shape {
                    what: "in-process bus gather".to_owned(),
                    detail: format!(
                        "rank {rank} sent {} elements, this rank {}",
                        payload.len(),
                        local.len()
                    ),
                });
            }
            out[rank * local.len()..(rank + 1) * local.len()].copy_from_slice(payload);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_width_splits_or_refuses() {
        let layout = TpLayout {
            rank: 3,
            world_size: 4,
        };
        assert_eq!(layout.local_width("moe_inter_dim", 2048).unwrap(), 512);
        let err = layout.local_width("moe_inter_dim", 2047).unwrap_err();
        assert!(
            err.to_string()
                .contains("moe_inter_dim 2047 does not split over 4 ranks"),
            "{err}"
        );
        assert!(layout.local_width("vocab", 0).is_err());
        let outside = TpLayout {
            rank: 4,
            world_size: 4,
        };
        assert!(outside.local_width("vocab", 8).is_err());
        assert!(TpRuntime::new(outside, Box::new(SingleRank)).is_err());
        let one = TpLayout {
            rank: 0,
            world_size: 1,
        };
        let runtime = TpRuntime::new(one, Box::new(SingleRank)).unwrap();
        assert_eq!(runtime.local_width("vocab", 7).unwrap(), 7);
    }

    #[test]
    fn a_local_group_reduces_on_its_wire_and_records_what_each_rank_saw() {
        // 2^-10 is below half a bf16 ulp at 1.0: the bf16 wire drops it, the f32 wire keeps it.
        let tiny = 2.0f32.powi(-10);
        for (wire, want) in [(Wire::Bf16, 2.0), (Wire::F32, 2.0 + tiny)] {
            let bus = Arc::new(LocalBus::recording(2).with_wire(wire));
            let sums: Vec<f32> = std::thread::scope(|scope| {
                let handles: Vec<_> = [1.0 + tiny, 1.0]
                    .into_iter()
                    .enumerate()
                    .map(|(rank, value)| {
                        let mut adapter = LocalRank {
                            rank,
                            bus: Arc::clone(&bus),
                        };
                        scope.spawn(move || {
                            let mut values = [value];
                            adapter.all_reduce_assign(&mut values).unwrap();
                            values[0]
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            assert_eq!(sums, [want, want], "{wire:?}");
            assert_eq!(bus.seams(1), Some(vec![vec![want]]), "{wire:?}");
        }
        let mut payload = [1.0f32];
        assert!(Wire::F32.reduce_into(&mut payload, &[]).is_err());
        assert!(Wire::F32.reduce_into(&mut payload, &[&[1.0, 2.0]]).is_err());
    }
}
