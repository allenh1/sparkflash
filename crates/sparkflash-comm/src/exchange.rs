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

//! Schedule driver shared by the loopback and TCP transports.
//!
//! The driver owns every ordering and numerics decision; a transport only has to move frames.
//! That is what makes the in-process loopback and the real-socket TCP backend provably
//! equivalent: both execute the same CollectiveLayout transfer list and the same canonical-order
//! reduction, so a passing loopback test and a passing TCP test are testing the same protocol
//! and the same arithmetic.

use std::ops::Range;

use crate::error::CommError;
use crate::layout::{CollectiveLayout, FabricSchedule, ScheduleTransfer, TransferPhase};
use crate::reduce;

/// Kind of collective an exchange is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExchangeKind {
    /// All-reduce (one-shot or hairpin).
    Reduce,
    /// All-gather of one segment per rank.
    Gather,
}

/// Identifies one phase of one collective on the wire and in a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExchangeTag {
    /// Caller-supplied sequence number, identical on every rank of the collective.
    pub seq: u64,
    /// Which collective is running.
    pub kind: ExchangeKind,
    /// Phase index inside the collective (1 is the hairpin all-gather).
    pub phase: u8,
}

/// One frame received from a peer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Received {
    /// Sending rank.
    pub from: usize,
    /// Elements carried by the frame.
    pub payload: Vec<u16>,
}

/// Transport-specific half of one collective.
pub(crate) trait Exchange {
    /// Announces this rank's participation and prepares to receive the listed frames.
    ///
    /// Each entry is (sending peer, element count).
    fn prepare(&mut self, tag: ExchangeTag, expected: &[(usize, usize)]) -> Result<(), CommError>;

    /// Transmits a slice of the caller's buffer to one peer.
    fn send(&mut self, to: usize, elements: Range<usize>, payload: &[u16])
    -> Result<(), CommError>;

    /// Waits for every expected frame and returns them in unspecified order.
    fn collect(&mut self, expected: &[(usize, usize)]) -> Result<Vec<Received>, CommError>;

    /// Releases per-collective resources, joining any outstanding reader threads.
    fn finish(&mut self) -> Result<(), CommError>;
}

/// Terms of the receive side of one phase, sorted by sender rank.
fn expected_for(
    transfers: &[ScheduleTransfer],
    rank: usize,
    phase: TransferPhase,
) -> Vec<(usize, usize)> {
    let mut expected: Vec<(usize, usize)> = transfers
        .iter()
        .filter(|transfer| transfer.phase == phase && transfer.to == rank)
        .map(|transfer| (transfer.from, transfer.elements.len()))
        .collect();
    expected.sort_unstable();
    expected
}

/// Checks that the frames a transport collected are exactly the ones the schedule asked for.
pub(crate) fn validate_received(
    expected: &[(usize, usize)],
    received: &[Received],
) -> Result<(), CommError> {
    let mut got: Vec<(usize, usize)> = received
        .iter()
        .map(|frame| (frame.from, frame.payload.len()))
        .collect();
    got.sort_unstable();
    let mut want = expected.to_vec();
    want.sort_unstable();
    if got != want {
        return Err(CommError::Protocol(format!(
            "expected frames {want:?} but received {got:?}"
        )));
    }
    Ok(())
}

/// Runs an all-reduce described by a layout.
///
/// # Errors
/// Returns CommError::RankOutOfRange for a rank outside the layout, CommError::InvalidLayout for
/// a buffer that does not match the layout, and any transport error below that.
pub(crate) fn all_reduce(
    exchange: &mut dyn Exchange,
    rank: usize,
    buf: &mut [u16],
    seq: u64,
    layout: &CollectiveLayout,
) -> Result<(), CommError> {
    if rank >= layout.world_size() {
        return Err(CommError::RankOutOfRange {
            rank,
            world_size: layout.world_size(),
        });
    }
    if buf.len() != layout.num_elements() {
        return Err(CommError::InvalidLayout(format!(
            "buffer has {} elements but the layout describes {}",
            buf.len(),
            layout.num_elements()
        )));
    }
    let outcome = match layout.schedule() {
        FabricSchedule::OneShotAllToAll => reduce_one_shot(exchange, rank, buf, seq, layout),
        FabricSchedule::HairpinRingReduceScatterAllGather => {
            reduce_hairpin(exchange, rank, buf, seq, layout)
        }
    };
    let finished = exchange.finish();
    outcome.and(finished)
}

/// Runs an all-gather where every rank contributes one segment of local.len() elements.
///
/// # Errors
/// Returns CommError::RankOutOfRange, CommError::InvalidLayout when out does not have exactly
/// world_size * local.len() elements, and any transport error below that.
pub(crate) fn all_gather(
    exchange: &mut dyn Exchange,
    rank: usize,
    local: &[u16],
    out: &mut [u16],
    seq: u64,
    world_size: usize,
) -> Result<(), CommError> {
    if world_size == 0 {
        return Err(CommError::InvalidLayout(
            "an all-gather needs at least one rank".to_owned(),
        ));
    }
    if rank >= world_size {
        return Err(CommError::RankOutOfRange { rank, world_size });
    }
    let expected_len = world_size.checked_mul(local.len()).ok_or_else(|| {
        CommError::InvalidLayout(format!(
            "{} segments of {} elements overflow the output length",
            world_size,
            local.len()
        ))
    })?;
    if out.len() != expected_len {
        return Err(CommError::InvalidLayout(format!(
            "all-gather output has {} elements but {} ranks of {} need {}",
            out.len(),
            world_size,
            local.len(),
            expected_len
        )));
    }
    let tag = ExchangeTag {
        seq,
        kind: ExchangeKind::Gather,
        phase: 0,
    };
    let expected: Vec<(usize, usize)> = (0..world_size)
        .filter(|peer| *peer != rank)
        .map(|peer| (peer, local.len()))
        .collect();
    let outcome = (|| -> Result<(), CommError> {
        exchange.prepare(tag, &expected)?;
        for peer in 0..world_size {
            if peer != rank {
                exchange.send(peer, 0..local.len(), local)?;
            }
        }
        let received = exchange.collect(&expected)?;
        validate_received(&expected, &received)?;
        let segment = local.len();
        out[rank * segment..(rank + 1) * segment].copy_from_slice(local);
        for frame in &received {
            let start = frame.from * segment;
            out[start..start + segment].copy_from_slice(&frame.payload);
        }
        Ok(())
    })();
    let finished = exchange.finish();
    outcome.and(finished)
}

/// One-shot schedule: all-to-all raw payloads, then a local canonical reduction.
fn reduce_one_shot(
    exchange: &mut dyn Exchange,
    rank: usize,
    buf: &mut [u16],
    seq: u64,
    layout: &CollectiveLayout,
) -> Result<(), CommError> {
    let transfers = layout.transfers();
    let expected = expected_for(&transfers, rank, TransferPhase::OneShot);
    let tag = ExchangeTag {
        seq,
        kind: ExchangeKind::Reduce,
        phase: 0,
    };
    // Phase instrumentation is a no-op unless SPARK_COMM_PROFILE is set; it reads clocks only
    // and never touches the payload or the reduction order.
    let mut timer = crate::profile::CallTimer::start();
    exchange.prepare(tag, &expected)?;
    timer.mark_prepare();
    for transfer in transfers
        .iter()
        .filter(|transfer| transfer.phase == TransferPhase::OneShot && transfer.from == rank)
    {
        let send_clock = timer.send_clock();
        exchange.send(
            transfer.to,
            transfer.elements.clone(),
            &buf[transfer.elements.clone()],
        )?;
        timer.record_send(send_clock);
    }
    timer.mark_send();
    let received = exchange.collect(&expected)?;
    timer.mark_collect();
    validate_received(&expected, &received)?;
    let own: Vec<u16> = buf.to_vec();
    let mut panes: Vec<&[u16]> = vec![&[]; layout.world_size()];
    panes[rank] = own.as_slice();
    for frame in &received {
        panes[frame.from] = frame.payload.as_slice();
    }
    let reduced = reduce::canonical_all_reduce_into(buf, &panes);
    timer.mark_reduce();
    reduced
}

/// Hairpin schedule: reduce-scatter raw contributions to the chunk owners, reduce each chunk in
/// canonical rank order, then all-gather the reduced chunks.
fn reduce_hairpin(
    exchange: &mut dyn Exchange,
    rank: usize,
    buf: &mut [u16],
    seq: u64,
    layout: &CollectiveLayout,
) -> Result<(), CommError> {
    let world = layout.world_size();
    let transfers = layout.transfers();
    let my_chunk = layout.chunk_range(rank);

    // Phase 1: every rank sends its contribution of chunk c to the owner of c.
    let expected_raw = expected_for(&transfers, rank, TransferPhase::ReduceScatter);
    let raw_tag = ExchangeTag {
        seq,
        kind: ExchangeKind::Reduce,
        phase: 0,
    };
    exchange.prepare(raw_tag, &expected_raw)?;
    for transfer in transfers
        .iter()
        .filter(|transfer| transfer.phase == TransferPhase::ReduceScatter && transfer.from == rank)
    {
        exchange.send(
            transfer.to,
            transfer.elements.clone(),
            &buf[transfer.elements.clone()],
        )?;
    }
    let raw = exchange.collect(&expected_raw)?;
    validate_received(&expected_raw, &raw)?;

    // The owner of chunk r accumulates every rank's contribution in ascending rank order.
    let own: Vec<u16> = buf[my_chunk.clone()].to_vec();
    let mut panes: Vec<&[u16]> = vec![&[]; world];
    panes[rank] = own.as_slice();
    for frame in &raw {
        panes[frame.from] = frame.payload.as_slice();
    }
    let mut reduced = vec![0u16; my_chunk.len()];
    reduce::canonical_all_reduce_into(&mut reduced, &panes)?;

    // Phase 2: the owner all-gathers the reduced chunk, still as bf16 bits.
    let expected_reduced = expected_for(&transfers, rank, TransferPhase::AllGather);
    let reduced_tag = ExchangeTag {
        seq,
        kind: ExchangeKind::Reduce,
        phase: 1,
    };
    exchange.prepare(reduced_tag, &expected_reduced)?;
    for transfer in transfers
        .iter()
        .filter(|transfer| transfer.phase == TransferPhase::AllGather && transfer.from == rank)
    {
        let range = transfer.elements.clone();
        if range != my_chunk {
            return Err(CommError::Protocol(format!(
                "all-gather transfer covers {range:?} but this rank owns {my_chunk:?}"
            )));
        }
        exchange.send(transfer.to, range, &reduced)?;
    }
    let gathered = exchange.collect(&expected_reduced)?;
    validate_received(&expected_reduced, &gathered)?;
    buf[my_chunk.clone()].copy_from_slice(&reduced);
    for frame in &gathered {
        let range = layout.chunk_range(frame.from);
        buf[range].copy_from_slice(&frame.payload);
    }
    Ok(())
}
