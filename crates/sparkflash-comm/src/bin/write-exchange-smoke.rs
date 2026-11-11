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

//! A four-rank smoke test of [sparkflash_comm::rdma::WriteExchange] on the real fabric.
//!
//! It stages the row on the host instead of on the GPU, because the point is to exercise the
//! things a unit test cannot reach: the endpoint-exchange control plane, the QP mesh, the pump
//! that posts the writes when the doorbell rings, the arrival flags and the reap. The exchange
//! kernel, which does the staging and the waiting on the device, is a separate problem and is not
//! in this path.
//!
//! Each round fills its whole row with words that name the sender, the round and the position,
//! rings the doorbell, waits for every peer's flag and then checks every word of every peer's row.
//! The row size alternates between --bytes and half of it, because that is what a step boundary
//! does to the real collective and it is the case a row set whose base moved with the row size got
//! wrong. A mismatch on any rank exits non-zero with the rank, the peer, the word and both values,
//! so a wrong stride or a stale flag is named rather than averaged away.
//!
//! Rows are split over --rails rails (2 unless told otherwise), so the same check covers the
//! split: a part that landed at the wrong offset, or a flag that outran its part, is a wrong word.
//!
//! usage: write-exchange-smoke --rank R --world 4 --peers a,b,c,d [--port P] [--bytes N]
//!        [--iters M] [--rails 1|2]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use sparkflash_comm::error::CommError;
use sparkflash_comm::rdma::{RdmaConfig, SlabLayout, WriteExchange};

/// Flattens a transport error into the string this diagnostic exits with.
fn comm<T>(result: Result<T, CommError>) -> Result<T, String> {
    result.map_err(|error| error.to_string())
}

fn main() -> Result<(), String> {
    let mut rank = 0usize;
    let mut world = 4usize;
    let mut peers: Vec<String> = Vec::new();
    let mut port = 18_770u16;
    let mut bytes = 10_240usize;
    let mut iters = 200usize;

    // The last round's index maths sorts an empty vector at zero, so the floor is one round.
    let mut rails = 2usize;
    let args: Vec<String> = std::env::args().collect();
    let mut at = 1;
    while at < args.len() {
        let value = || -> Result<String, String> {
            args.get(at + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[at]))
        };
        match args[at].as_str() {
            "--rank" => {
                rank = value()?
                    .parse()
                    .map_err(|error| format!("--rank: {error}"))?;
                at += 2;
            }
            "--world" => {
                world = value()?
                    .parse()
                    .map_err(|error| format!("--world: {error}"))?;
                at += 2;
            }
            "--peers" => {
                peers = value()?
                    .split(',')
                    .map(|peer| peer.trim().to_owned())
                    .collect();
                at += 2;
            }
            "--port" => {
                port = value()?
                    .parse()
                    .map_err(|error| format!("--port: {error}"))?;
                at += 2;
            }
            "--bytes" => {
                bytes = value()?
                    .parse()
                    .map_err(|error| format!("--bytes: {error}"))?;
                at += 2;
            }
            "--iters" => {
                iters = value()?
                    .parse()
                    .map_err(|error| format!("--iters: {error}"))?;
                at += 2;
            }
            "--rails" => {
                rails = value()?
                    .parse()
                    .map_err(|error| format!("--rails: {error}"))?;
                at += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if peers.len() != world {
        return Err(format!(
            "--peers needs {world} addresses, got {}",
            peers.len()
        ));
    }
    // The last round's summary sorts its samples; zero rounds would index an empty vector.
    iters = iters.max(1);
    let layout = comm(SlabLayout::new(world, bytes, rails))?;
    let total = comm(layout.total_bytes())?;
    // Page-aligned and pinned by registration, which is the memory this fabric accepts.
    let slab = unsafe {
        let layout = std::alloc::Layout::from_size_align(total, 4096)
            .map_err(|error| format!("slab layout: {error}"))?;
        let pointer = std::alloc::alloc(layout);
        if pointer.is_null() {
            return Err(format!("could not allocate {total} bytes"));
        }
        // Filled with a pattern, not zeroed. The open is supposed to zero the slab before it
        // publishes it, because every wait is "the word is at least my sequence number" and one
        // stale flag makes the first round believe a peer has arrived. Allocating zeroed here - as
        // this test used to - is exactly what hid that bug, so the allocation is poisoned and the
        // first round is the proof that the open cleaned it.
        std::ptr::write_bytes(pointer, 0xa5, total);
        pointer
    };
    let config = RdmaConfig {
        world_size: world,
        rank,
        peers,
        control_port: port,
        gid_index: 3,
        port: 1,
        rail_count: rails,
        timeout: Duration::from_millis(5_000),
        max_frame_bytes: bytes,
    };
    let mut exchange = comm(WriteExchange::open(&config, slab.cast::<c_void>(), layout))?;
    println!(
        "[{rank}] exchange up: {world} ranks over {:?}, {bytes} -byte rows, slab {total} bytes at {:#x}, seq {}",
        exchange.rail_names(),
        exchange.address(),
        exchange.seq()
    );
    // The word a sender puts at `index` of its row in round `seq`.
    let word = |sender: usize, seq: u64, index: usize| -> u64 {
        ((sender as u64) << 56) | ((seq & 0xff_ffff) << 32) | index as u64
    };
    let mut last = Vec::with_capacity(iters);
    let mut slowest = (0usize, 0f64);
    let mut words_checked = 0u64;
    for round in 0..iters {
        // Consecutive rounds differ in size as well as in set - and the size schedule is driven
        // from round / 2, not from round, because the set is seq & 1 and seq = round + 1. Keying
        // both on round put the large row in one set and the small row in the other for the whole
        // run, so the pairing that the fixed set base exists to make safe (a small row's second set
        // against a large row's first) was never generated.
        let row = if (round / 2) % 2 == 1 && bytes >= 32 {
            (bytes / 2) & !7
        } else {
            bytes & !7
        };
        let seq = comm(exchange.submit(row))?;
        let parity = WriteExchange::parity(seq);
        let mine = comm(layout.row_offset(rank, row, parity))?;
        // Stage this rank's row on the host, then ring the doorbell the kernel would have rung.
        for index in 0..row / 8 {
            // SAFETY: the offset is inside the slab (row_offset placed a row of `row` bytes).
            unsafe {
                std::ptr::write_volatile(
                    slab.add(mine + index * 8).cast::<u64>(),
                    word(rank, seq, index),
                );
            }
        }
        let started = Instant::now();
        exchange.ring_doorbell(seq);
        comm(exchange.wait_arrival(seq, row))?;
        let took = started.elapsed().as_secs_f64() * 1e6;
        if took > slowest.1 {
            slowest = (round, took);
        }
        last.push(took);
        // Every word of every peer's row must be that peer's word for this round.
        for peer in 0..world {
            if peer == rank {
                continue;
            }
            let flag = exchange.arrival(peer);
            if flag < seq {
                return Err(format!(
                    "rank {rank} round {seq}: peer {peer}'s flag is {flag}, behind the round"
                ));
            }
            let offset = comm(layout.row_offset(peer, row, parity))?;
            for index in 0..row / 8 {
                // SAFETY: as above, for the peer's row of the same size in the same set.
                let seen =
                    unsafe { std::ptr::read_volatile(slab.add(offset + index * 8).cast::<u64>()) };
                let want = word(peer, seq, index);
                if seen != want {
                    return Err(format!(
                        "rank {rank} round {seq}: word {index} of peer {peer}'s {row} -byte row reads {seen:#x}, not {want:#x}; its flag is {flag}"
                    ));
                }
                words_checked += 1;
            }
        }
    }
    let mut sorted = last.clone();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mean: f64 = last.iter().sum::<f64>() / last.len() as f64;
    println!(
        "[{rank}] {iters} rounds alternating {bytes} and {} B, {words_checked} words verified: doorbell to every arrival avg {mean:.2} us, p50 {:.2}, max {:.2}",
        bytes / 2,
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1]
    );
    println!(
        "[{rank}] slowest round was {} at {:.2} us",
        slowest.0, slowest.1
    );
    let stats = exchange.stats();
    println!(
        "[{rank}] pump: {} rounds, {:.2} us a round to post, post to every arrival {:.2} us over {} rounds (max {:.2}), parked {} times",
        stats.rounds,
        stats.post_ns as f64 / 1e3 / stats.rounds.max(1) as f64,
        stats.arrival_ns as f64 / 1e3 / stats.arrivals.max(1) as f64,
        stats.arrivals,
        stats.arrival_max_ns as f64 / 1e3,
        stats.parks
    );
    println!(
        "[{rank}] pump: post to each rank's flag, us: {:?}",
        stats
            .peer_arrival_ns
            .iter()
            .map(|ns| (*ns as f64 / 10.0 / stats.arrivals.max(1) as f64).round() / 100.0)
            .collect::<Vec<_>>()
    );
    // The registration must go before the memory it covers does, and the registration must also
    // outlive every peer's writes. One more round, submitted and waited for by both ranks, is the
    // goodbye: when it returns, each rank knows its peers have finished posting into this slab and
    // are waiting on the next round, so nothing can be in flight over the memory when it goes back.
    let goodbye = comm(exchange.submit(bytes))?;
    let parity = WriteExchange::parity(goodbye);
    let mine = comm(layout.row_offset(rank, bytes, parity))?;
    for index in 0..bytes / 8 {
        // SAFETY: inside the slab, in this rank's row of the round's set.
        unsafe {
            std::ptr::write_volatile(
                slab.add(mine + index * 8).cast::<u64>(),
                word(rank, goodbye, index),
            );
        }
    }
    exchange.ring_doorbell(goodbye);
    comm(exchange.wait_arrival(goodbye, bytes))?;
    // The registration must go before the memory it covers does.
    drop(exchange);
    unsafe {
        let layout = std::alloc::Layout::from_size_align(total, 4096)
            .map_err(|error| format!("slab layout: {error}"))?;
        std::alloc::dealloc(slab, layout);
    }
    Ok(())
}
