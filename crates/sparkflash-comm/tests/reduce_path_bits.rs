//! Bit identity between the two reduce transports on the forward's served shape.
//!
//! The forward's attention-output and routed-MoE collectives are 5,120 bf16 elements = 10,240 B
//! (10 KiB) over four ranks: exactly the payload the specification times. Both transports run the
//! same schedule driver (`sparkflash_comm::exchange::all_reduce`) and the same canonical reduction
//! (`sparkflash_comm::reduce::canonical_all_reduce_into`): f32 accumulation from 0.0, ascending rank
//! order, one round-to-nearest-even bf16 store. These tests pin what can be checked without the
//! live fabric: each transport matches the canonical reference bit for bit, and the RoCEnante
//! handshake model (`arrival_seq` word, doorbell, all-to-all post - the same shape the RDMA
//! dataplane drives) and the real TCP exchange produce identical bytes on that payload.
//!
//! The real RoCE transport cannot be opened on the dev host (no libibverbs and no fabric). Its
//! equality with the CPU reference is the ignored four-node hardware gate
//! `roce_bits_match_the_cpu_reference_over_ten_thousand_replays` (10 KiB x 10,000 replays); its
//! equality with TCP inside a forward run is the row's evidence-digest comparison.

mod support;

use support::{assert_bits_eq, finite_pattern, port_block, reference_all_reduce, run_ranks};

use sparkflash_comm::{
    CollectiveLayout, FabricSchedule, LoopbackCluster, LoopbackConfig, TcpConfig, TcpTransport,
    Transport,
};

/// bf16 elements in one served decode row: hidden 5,120 -> 10,240 B = 10 KiB.
const SERVED_ELEMENTS: usize = 5_120;

#[test]
fn tcp_and_the_rocenante_handshake_model_agree_bit_for_bit_on_the_served_reduce() {
    let world = 4;
    let payloads: Vec<Vec<u16>> = (0..world)
        .map(|rank| finite_pattern(rank, SERVED_ELEMENTS, 0x5eed))
        .collect();
    let expected = reference_all_reduce(&payloads);
    let layout = CollectiveLayout::new(world, SERVED_ELEMENTS, FabricSchedule::OneShotAllToAll)
        .unwrap_or_else(|error| panic!("layout: {error}"));

    let base_port = port_block();
    let over_tcp = run_ranks(world, |rank| {
        let transport = TcpTransport::connect(TcpConfig::loopback4(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} tcp connect: {error}"));
        let mut buf = payloads[rank].clone();
        transport
            .all_reduce_bf16(&mut buf, 1, &layout)
            .unwrap_or_else(|error| panic!("rank {rank} tcp reduce: {error}"));
        buf
    });

    let cluster = LoopbackCluster::new(LoopbackConfig {
        world_size: world,
        timeout: support::CLUSTER_TIMEOUT,
        ..LoopbackConfig::default()
    })
    .unwrap_or_else(|error| panic!("loopback cluster: {error}"));
    let over_model = run_ranks(world, |rank| {
        let transport = cluster
            .handle(rank)
            .unwrap_or_else(|error| panic!("rank {rank} loopback handle: {error}"));
        let mut buf = payloads[rank].clone();
        transport
            .all_reduce_bf16(&mut buf, 1, &layout)
            .unwrap_or_else(|error| panic!("rank {rank} loopback reduce: {error}"));
        buf
    });

    for rank in 0..world {
        assert_bits_eq(
            &over_tcp[rank],
            &expected,
            &format!("tcp rank {rank} vs canonical reference"),
        );
        assert_bits_eq(
            &over_model[rank],
            &over_tcp[rank],
            &format!("rocenante handshake model rank {rank} vs tcp"),
        );
    }
}

/// The same comparison on several sequences, because the exchange tag is part of the protocol and
/// the one-shot handshake must not depend on which sequence number a call carries.
#[test]
fn the_two_transports_agree_on_the_served_reduce_for_every_sequence() {
    let world = 4;
    let payloads: Vec<Vec<u16>> = (0..world)
        .map(|rank| finite_pattern(rank, SERVED_ELEMENTS, 0xfeed))
        .collect();
    let expected = reference_all_reduce(&payloads);
    let layout = CollectiveLayout::new(world, SERVED_ELEMENTS, FabricSchedule::OneShotAllToAll)
        .unwrap_or_else(|error| panic!("layout: {error}"));
    let base_port = port_block();
    for seq in 1..=4 {
        let over_tcp = run_ranks(world, |rank| {
            let transport = TcpTransport::connect(TcpConfig::loopback4(rank, base_port))
                .unwrap_or_else(|error| panic!("rank {rank} tcp connect: {error}"));
            let mut buf = payloads[rank].clone();
            transport
                .all_reduce_bf16(&mut buf, seq, &layout)
                .unwrap_or_else(|error| panic!("rank {rank} tcp reduce seq {seq}: {error}"));
            buf
        });
        let cluster = LoopbackCluster::new(LoopbackConfig {
            world_size: world,
            timeout: support::CLUSTER_TIMEOUT,
            ..LoopbackConfig::default()
        })
        .unwrap_or_else(|error| panic!("loopback cluster: {error}"));
        let over_model = run_ranks(world, |rank| {
            let transport = cluster
                .handle(rank)
                .unwrap_or_else(|error| panic!("rank {rank} loopback handle: {error}"));
            let mut buf = payloads[rank].clone();
            transport
                .all_reduce_bf16(&mut buf, seq, &layout)
                .unwrap_or_else(|error| panic!("rank {rank} loopback reduce seq {seq}: {error}"));
            buf
        });
        for rank in 0..world {
            assert_bits_eq(
                &over_tcp[rank],
                &expected,
                &format!("tcp rank {rank} seq {seq} vs canonical reference"),
            );
            assert_bits_eq(
                &over_model[rank],
                &over_tcp[rank],
                &format!("handshake model rank {rank} seq {seq} vs tcp"),
            );
        }
    }
}
