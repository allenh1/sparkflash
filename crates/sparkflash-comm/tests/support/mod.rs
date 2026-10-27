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

//! Shared helpers for the sparkflash-comm integration tests.
#![allow(dead_code)]

use std::net::TcpListener;
use std::sync::Mutex;
use std::time::Duration;

/// Ports per block: the four rank listeners plus room for the next test.
const PORT_BLOCK: u16 = 16;

/// First candidate base port of this process, 0 until the first call.
///
/// The lock is held across probing and reservation: two tests that raced between a successful
/// probe and the reservation would otherwise be handed the same block, and one of them would
/// then fail to bind its rank listeners.
static NEXT_PORT: Mutex<u16> = Mutex::new(0);

/// Claims a block of PORT_BLOCK consecutive free loopback ports.
///
/// Two properties keep the socket tests reliable when several test binaries and ordinary network
/// clients share the machine:
///
/// * the base is derived from the process id and kept below the kernel's ephemeral port range,
///   so an unrelated outbound connection can never be handed one of our listener ports (that
///   collision is exactly what a fixed base of 42100 produced: rank 0 failed to bind with
///   "Address already in use" and the other ranks then timed out);
/// * every candidate block is probed by binding all of its ports before it is handed out, and a
///   block that is already taken is skipped, so two processes cannot pick the same block.
pub fn port_block() -> u16 {
    let mut next = NEXT_PORT.lock().unwrap_or_else(|error| error.into_inner());
    if *next == 0 {
        *next = first_candidate();
    }
    let mut base = *next;
    for _ in 0..200 {
        if let Some(free) = probe_block(base) {
            *next = free.saturating_add(PORT_BLOCK);
            return free;
        }
        base = base.saturating_add(PORT_BLOCK);
    }
    panic!("no free loopback port block found for the test transport");
}

/// Starting candidate: 1024 plus a whole-block offset inside the non-ephemeral range.
///
/// The offset is a multiple of PORT_BLOCK and comes from the process id mixed with the process
/// start time, so two test binaries running at the same time pick different, non-overlapping
/// blocks instead of stepping through each other's ports.
fn first_candidate() -> u16 {
    let low = u32::from(ephemeral_low().unwrap_or(32768));
    // Stay comfortably below the ephemeral range; a host configured with a tiny range falls back
    // to a low base and relies on the probe loop instead.
    let span = low.saturating_sub(2048).max(1024);
    let blocks = (span / u32::from(PORT_BLOCK)).max(1);
    let block = (invocation_tag() % u64::from(blocks)) as u32;
    1024 + (block * u32::from(PORT_BLOCK)) as u16
}

/// Per-process tag: the process id mixed with its start time when the host publishes it.
fn invocation_tag() -> u64 {
    let pid = u64::from(std::process::id());
    let start = process_start_ticks().unwrap_or(0);
    pid.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ start.wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Start time of this process in clock ticks, read from the kernel's stat file.
fn process_start_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = stat.rsplit_once(')')?.1;
    rest.split_whitespace().nth(19)?.parse().ok()
}

/// Low end of the kernel's ephemeral port range, when the host publishes it.
fn ephemeral_low() -> Option<u16> {
    let text = std::fs::read_to_string("/proc/sys/net/ipv4/ip_local_port_range").ok()?;
    text.split_whitespace().next()?.parse().ok()
}

/// Binds every port of the block, then releases them, reporting the block as free only if all of
/// them were available.
fn probe_block(base: u16) -> Option<u16> {
    let mut held = Vec::with_capacity(usize::from(PORT_BLOCK));
    for offset in 0..PORT_BLOCK {
        let port = base.checked_add(offset)?;
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => held.push(listener),
            Err(_) => return None,
        }
    }
    drop(held);
    Some(base)
}

/// Runs one closure per rank on its own OS thread and returns the results in rank order.
pub fn run_ranks<T, F>(world: usize, body: F) -> Vec<T>
where
    F: Fn(usize) -> T + Sync,
    T: Send,
{
    run_ranks_order(&(0..world).collect::<Vec<_>>(), body)
        .into_iter()
        .map(|(_rank, value)| value)
        .collect()
}

/// Like run_ranks but starts the rank threads in the given order, which changes which rank
/// stages its payload first.
pub fn run_ranks_order<T, F>(order: &[usize], body: F) -> Vec<(usize, T)>
where
    F: Fn(usize) -> T + Sync,
    T: Send,
{
    std::thread::scope(|scope| {
        let body = &body;
        let handles: Vec<_> = order
            .iter()
            .map(|rank| {
                let rank = *rank;
                (rank, scope.spawn(move || body(rank)))
            })
            .collect();
        handles
            .into_iter()
            .map(|(rank, handle)| (rank, handle.join().expect("a rank thread panicked")))
            .collect()
    })
}

/// Small deterministic PRNG so the shuffle tests do not depend on the rand crate's version.
pub struct Lcg(u64);

impl Lcg {
    /// Seeds the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Next 64-bit value.
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

/// Deterministic finite bf16 bit patterns (no NaN, no infinity) so the independent oracle in
/// reference_all_reduce, which goes through the half crate, agrees on every input.
pub fn finite_pattern(rank: usize, len: usize, seed: u64) -> Vec<u16> {
    let mut state = Lcg::new(seed ^ ((rank as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
    (0..len)
        .map(|_| {
            loop {
                let bits = (state.next() & 0xffff) as u16;
                if (bits & 0x7f80) != 0x7f80 {
                    break bits;
                }
            }
        })
        .collect()
}

/// A hand-picked pattern that exercises negative zero, subnormals, exact powers of two and
/// values whose f32 sums round differently depending on order.
pub fn tricky_pattern(rank: usize, len: usize) -> Vec<u16> {
    const EDGES: [u16; 12] = [
        0x0000,          // +0.0
        0x8000,          // -0.0
        0x3f80,          // 1.0
        0xbf80,          // -1.0
        0x0001,          // smallest bf16 subnormal
        0x8001,          // negative smallest subnormal
        0x7f7f,          // largest finite bf16
        0xff7f,          // most negative finite bf16
        0x3b80,          // 2^-8, half a bf16 ulp at 1.0
        0x3380,          // 2^-24
        0x4048,          // 3.125
        0x7f80 - 0x0100, // large finite value
    ];
    (0..len)
        .map(|index| EDGES[(index + rank * 3) % EDGES.len()])
        .collect()
}

/// Independent CPU reference for the canonical reduction.
///
/// The f32 accumulation order (rank 0 first, seeded with 0.0f) is the definition; the bf16
/// rounding is delegated to the half crate, an independent implementation of round-to-nearest-even,
/// so a bug in our own f32_to_bf16_rn cannot hide behind this oracle.
pub fn reference_all_reduce(payloads: &[Vec<u16>]) -> Vec<u16> {
    assert!(!payloads.is_empty(), "reference needs at least one rank");
    let len = payloads[0].len();
    for (rank, payload) in payloads.iter().enumerate() {
        assert_eq!(payload.len(), len, "rank {rank} has the wrong length");
    }
    (0..len)
        .map(|slot| {
            let mut acc = 0.0f32;
            for payload in payloads {
                let value: f32 = half::bf16::from_bits(payload[slot]).into();
                // Written as an explicit chain of adds: this order is the definition.
                acc += value;
            }
            half::bf16::from_f32(acc).to_bits()
        })
        .collect()
}

/// Independent reference for the all-gather: concatenate the segments in rank order.
pub fn reference_all_gather(locals: &[Vec<u16>]) -> Vec<u16> {
    locals.iter().flatten().copied().collect()
}

/// Compares bit vectors element by element so a failure names the first divergent slot instead
/// of dumping a megabyte of numbers.
pub fn assert_bits_eq(actual: &[u16], expected: &[u16], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, want,
            "{context}: element {index} is 0x{got:04x}, expected 0x{want:04x}"
        );
    }
}

/// Common timeout for loopback clusters: generous enough that a correct run never hits it.
pub const CLUSTER_TIMEOUT: Duration = Duration::from_secs(20);
