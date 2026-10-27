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

//! The NCCL bootstrap rendezvous: the unique id AND the per-run agreement words crossing the run's
//! own TCP transport.
//!
//! The four-rank forward loop already holds a TcpTransport before layer 0 (its own port range,
//! base + 100, so a gather frame can never land in a walk reduce stream). The bootstrap reuses that
//! transport for one all-gather: rank 0's 128 NCCL id bytes as 64 little-endian u16 words, then six
//! agreement words per rank (library flag, id flag, arm, native rounding, behaviour toggles, plan
//! digest). These tests pin the wire format, the rank-0-only id rule, and every refusal - a missing
//! library, a wrong id layout, a mixed arm, a native-rounding difference, a behaviour-toggle
//! difference or a plan-digest difference must refuse deterministically on ALL ranks before NCCL is
//! entered. Four real loopback transports, the same TcpTransport the forward uses.
//!
//! NCCL itself is not entered here (the dev host has no libnccl): the exchange and its refusals are
//! the part that can be proven without a GPU, and RunReduce::open's refusal path is unit-tested.

mod support;

use support::{port_block, run_ranks};

use sparkflash_comm::device_reduce::{
    BOOTSTRAP_ID_WORDS, BOOTSTRAP_SEGMENT_WORDS, BootstrapConfig, bootstrap_agreement,
    bootstrap_exchange, bootstrap_id_from_gathered, bootstrap_segment, setup_barrier,
};
use sparkflash_comm::nccl::{NCCL_UNIQUE_ID_BYTES, NcclUniqueId};
use sparkflash_comm::{TcpConfig, TcpTransport};

/// A deterministic id, so a rank-0 word landing in another rank's slot cannot pass by luck.
fn sample_id(seed: u8) -> NcclUniqueId {
    let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let index = index as u8;
        *byte = index
            .wrapping_mul(31)
            .wrapping_add(seed)
            .wrapping_add(index >> 1);
    }
    NcclUniqueId::from_bytes(bytes)
}

fn agreed_config() -> BootstrapConfig {
    BootstrapConfig {
        native_rounding: 0b101,
        env_toggles: 0,
        plan_digest: 0x1234_5678_9abc_def0,
    }
}

/// One rank's segment for the happy path: rank 0 supplies the id, everyone reports the library.
fn segment_for(rank: usize, id: &NcclUniqueId, config: BootstrapConfig) -> Vec<u16> {
    let local = if rank == 0 { Some(id) } else { None };
    bootstrap_segment(local, true, config).to_vec()
}

#[test]
fn rank_zero_s_unique_id_reaches_every_rank_over_the_run_transport() {
    let base_port = port_block();
    let id = sample_id(11);
    let config = agreed_config();
    let seen = run_ranks(4, |rank| {
        let transport = TcpTransport::connect(TcpConfig::loopback4(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} tcp connect: {error}"));
        let segment = segment_for(rank, &id, config);
        let gathered = bootstrap_exchange(&transport, 4, rank, 0xA_0000, &segment)
            .unwrap_or_else(|error| panic!("rank {rank} gather: {error}"));
        bootstrap_agreement(&gathered, 4)
            .unwrap_or_else(|error| panic!("rank {rank} agreement: {error}"));
        bootstrap_id_from_gathered(&gathered, 4)
            .unwrap_or_else(|error| panic!("rank {rank} id: {error}"))
    });
    for (rank, received) in seen.iter().enumerate() {
        assert_eq!(
            received, &id,
            "rank {rank} must receive rank 0's id byte for byte"
        );
        assert_eq!(received.as_bytes(), id.as_bytes());
    }
}

#[test]
fn a_second_bootstrap_on_a_fresh_transport_sees_the_same_bytes() {
    let base_port = port_block();
    let id = sample_id(5);
    let config = agreed_config();
    let results = run_ranks(4, |rank| {
        let transport = TcpTransport::connect(TcpConfig::loopback4(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} tcp connect: {error}"));
        let segment = segment_for(rank, &id, config);
        let first = bootstrap_exchange(&transport, 4, rank, 0xA_0000, &segment)
            .unwrap_or_else(|error| panic!("rank {rank} first gather: {error}"));
        let second = bootstrap_exchange(&transport, 4, rank, 0xA_0000, &segment)
            .unwrap_or_else(|error| panic!("rank {rank} second gather: {error}"));
        assert_eq!(first, second);
        bootstrap_id_from_gathered(&second, 4)
            .unwrap_or_else(|error| panic!("rank {rank} id: {error}"))
    });
    for (rank, received) in results.iter().enumerate() {
        assert_eq!(received, &id, "rank {rank} second exchange");
    }
}

fn refusal_on_every_rank(mutate: impl Fn(usize, &mut [u16]) + Sync) {
    let base_port = port_block();
    let id = sample_id(7);
    let config = agreed_config();
    let errors = run_ranks(4, |rank| {
        let transport = TcpTransport::connect(TcpConfig::loopback4(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} tcp connect: {error}"));
        let mut segment = segment_for(rank, &id, config);
        mutate(rank, &mut segment);
        let gathered = bootstrap_exchange(&transport, 4, rank, 0xA_0000, &segment)
            .unwrap_or_else(|error| panic!("rank {rank} gather: {error}"));
        let outcome = bootstrap_agreement(&gathered, 4);
        outcome
            .expect_err("the mismatch must refuse on every rank")
            .to_string()
    });
    let first = &errors[0];
    for (rank, error) in errors.iter().enumerate() {
        assert_eq!(error, first, "rank {rank} must refuse for the same reason");
    }
}

#[test]
fn one_rank_without_the_library_refuses_on_every_rank_before_nccl() {
    refusal_on_every_rank(|rank, segment| {
        if rank == 2 {
            // Clear the library flag: rank 2 never loaded libnccl.
            let index = BOOTSTRAP_ID_WORDS;
            segment[index] &= !1;
        }
    });
}

#[test]
fn an_id_supplied_by_a_non_rank_zero_refuses_on_every_rank() {
    refusal_on_every_rank(|rank, segment| {
        if rank == 1 {
            let index = BOOTSTRAP_ID_WORDS;
            segment[index] |= 2;
        }
    });
}

#[test]
fn a_mixed_arm_word_refuses_on_every_rank() {
    refusal_on_every_rank(|rank, segment| {
        if rank == 3 {
            let index = BOOTSTRAP_ID_WORDS + 1;
            segment[index] = 0;
        }
    });
}

#[test]
fn a_native_rounding_word_difference_refuses_on_every_rank() {
    refusal_on_every_rank(|rank, segment| {
        if rank == 2 {
            let index = BOOTSTRAP_ID_WORDS + 2;
            segment[index] = 0b010;
        }
    });
}

#[test]
fn a_behaviour_toggle_word_difference_refuses_on_every_rank() {
    refusal_on_every_rank(|rank, segment| {
        if rank == 0 {
            let index = BOOTSTRAP_ID_WORDS + 3;
            segment[index] = 1;
        }
    });
}

#[test]
fn a_plan_digest_difference_refuses_on_every_rank() {
    refusal_on_every_rank(|rank, segment| {
        if rank == 1 {
            let index = BOOTSTRAP_ID_WORDS + 4;
            segment[index] = segment[index].wrapping_add(1);
        }
    });
}

#[test]
fn the_bootstrap_segment_is_seventy_words() {
    assert_eq!(BOOTSTRAP_ID_WORDS, 64);
    assert_eq!(BOOTSTRAP_SEGMENT_WORDS, 70);
    let id = sample_id(3);
    let segment = segment_for(0, &id, agreed_config());
    assert_eq!(segment.len(), BOOTSTRAP_SEGMENT_WORDS);
}

#[test]
fn the_setup_barrier_completes_on_every_rank() {
    let base_port = port_block();
    run_ranks(4, |rank| {
        let transport = TcpTransport::connect(TcpConfig::loopback4(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} tcp connect: {error}"));
        setup_barrier(&transport, 4, rank)
            .unwrap_or_else(|error| panic!("rank {rank} barrier: {error}"));
    });
}
