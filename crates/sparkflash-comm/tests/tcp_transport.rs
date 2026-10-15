//! Real-socket transport tests: framing on a byte stream, partial reads, peer death and
//! deadlines. These run entirely on 127.0.0.1, so they need neither RDMA hardware nor the live
//! four-node cluster, but they exercise the same protocol the RoCEnante kernel would.

mod support;

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use support::{assert_bits_eq, finite_pattern, port_block, reference_all_reduce, run_ranks};

use sparkflash_comm::{
    CollectiveLayout, CommError, FabricSchedule, FrameHeader, FrameKind, TcpConfig, TcpTransport,
    Transport, decode_bf16, encode_bf16, read_frame,
};

fn config(rank: usize, base_port: u16) -> TcpConfig {
    TcpConfig::loopback4(rank, base_port)
}

#[test]
fn four_ranks_all_reduce_over_real_sockets_matches_the_reference() {
    let world = 4;
    let elements = 257;
    let payloads: Vec<Vec<u16>> = (0..world)
        .map(|rank| finite_pattern(rank, elements, 0x7c9e))
        .collect();
    let expected = reference_all_reduce(&payloads);
    for schedule in [
        FabricSchedule::OneShotAllToAll,
        FabricSchedule::HairpinRingReduceScatterAllGather,
    ] {
        let base_port = port_block();
        let layout = CollectiveLayout::new(world, elements, schedule)
            .unwrap_or_else(|error| panic!("layout: {error}"));
        let results = run_ranks(world, |rank| {
            let transport = TcpTransport::connect(config(rank, base_port))
                .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
            let mut buf = payloads[rank].clone();
            transport
                .all_reduce_bf16(&mut buf, 1, &layout)
                .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
            buf
        });
        for (rank, buf) in results.iter().enumerate() {
            assert_bits_eq(buf, &expected, &format!("tcp rank {rank} {schedule:?}"));
        }
    }
}

#[test]
fn all_gather_over_real_sockets_reassembles_every_segment() {
    let world = 4;
    let elements = 1003;
    let base_port = port_block();
    let locals: Vec<Vec<u16>> = (0..world)
        .map(|rank| finite_pattern(rank, elements, 0x9a11))
        .collect();
    let expected = support::reference_all_gather(&locals);
    let results = run_ranks(world, |rank| {
        let transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        let mut out = vec![0u16; world * elements];
        transport
            .all_gather_bf16(&locals[rank], &mut out, 1)
            .unwrap_or_else(|error| panic!("rank {rank}: {error}"));
        out
    });
    for (rank, out) in results.iter().enumerate() {
        assert_bits_eq(out, &expected, &format!("tcp all-gather rank {rank}"));
    }
}

#[test]
fn a_peer_that_dies_is_reported_as_peer_closed() {
    let world = 4;
    let base_port = port_block();
    let barrier = Arc::new(Barrier::new(world));
    let outcomes = run_ranks(world, |rank| {
        let transport = TcpTransport::connect(config(rank, base_port))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        barrier.wait();
        if rank == 3 {
            // Model a rank that crashes: close both directions of every socket.
            transport.shutdown();
            barrier.wait();
            return None;
        }
        let layout = CollectiveLayout::new(world, 8, FabricSchedule::OneShotAllToAll)
            .unwrap_or_else(|error| panic!("layout: {error}"));
        let mut buf = vec![0u16; 8];
        let outcome = transport.all_reduce_bf16(&mut buf, 1, &layout);
        barrier.wait();
        Some(
            outcome
                .err()
                .unwrap_or_else(|| panic!("rank {rank} completed a collective with a dead peer")),
        )
    });
    for (rank, outcome) in outcomes.into_iter().enumerate() {
        if rank == 3 {
            assert!(outcome.is_none());
            continue;
        }
        match outcome {
            Some(CommError::PeerClosed { peer }) => assert!(peer < world),
            Some(CommError::Timeout { .. }) => {
                panic!("rank {rank} should have seen the closed socket, not a timeout")
            }
            other => panic!("rank {rank} expected PeerClosed, got {other:?}"),
        }
    }
}

#[test]
fn a_peer_that_goes_silent_hits_the_deadline() {
    let world = 4;
    let base_port = port_block();
    let timeout = Duration::from_millis(200);
    let barrier = Arc::new(Barrier::new(world));
    let outcomes = run_ranks(world, |rank| {
        let transport = TcpTransport::connect(config(rank, base_port).with_timeout(timeout))
            .unwrap_or_else(|error| panic!("rank {rank} connect: {error}"));
        barrier.wait();
        if rank == 3 {
            // Keep the sockets open but never participate.
            barrier.wait();
            return None;
        }
        let layout = CollectiveLayout::new(world, 8, FabricSchedule::OneShotAllToAll)
            .unwrap_or_else(|error| panic!("layout: {error}"));
        let mut buf = vec![0u16; 8];
        let outcome = transport.all_reduce_bf16(&mut buf, 1, &layout);
        barrier.wait();
        Some(
            outcome
                .err()
                .unwrap_or_else(|| panic!("rank {rank} completed a collective with a silent peer")),
        )
    });
    for (rank, outcome) in outcomes.into_iter().enumerate() {
        if rank == 3 {
            assert!(outcome.is_none());
            continue;
        }
        match outcome {
            Some(CommError::Timeout { timeout_ms, .. }) => assert_eq!(timeout_ms, 200),
            other => panic!("rank {rank} expected Timeout, got {other:?}"),
        }
    }
}

#[test]
fn frames_survive_a_sender_that_dribbles_bytes() {
    let listener =
        TcpListener::bind(("127.0.0.1", 0)).unwrap_or_else(|error| panic!("bind: {error}"));
    let address = listener
        .local_addr()
        .unwrap_or_else(|error| panic!("addr: {error}"));
    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener
            .accept()
            .unwrap_or_else(|error| panic!("accept: {error}"));
        let first =
            read_frame(&mut stream, 4096).unwrap_or_else(|error| panic!("frame 1: {error}"));
        let second =
            read_frame(&mut stream, 4096).unwrap_or_else(|error| panic!("frame 2: {error}"));
        (first, second)
    });

    let values: Vec<u16> = (0..64u16)
        .map(|index| 0x3f80u16.wrapping_add(index))
        .collect();
    let first = FrameHeader {
        kind: FrameKind::ReducePayload,
        phase: 0,
        seq: 11,
        rank: 2,
        payload_bytes: (values.len() * 2) as u32,
    };
    let second = FrameHeader {
        kind: FrameKind::GatherPayload,
        phase: 1,
        seq: 12,
        rank: 2,
        payload_bytes: 0,
    };
    let mut bytes = first.encode().to_vec();
    bytes.extend_from_slice(&encode_bf16(&values));
    bytes.extend_from_slice(&second.encode());

    let mut client = TcpStream::connect(address).unwrap_or_else(|error| panic!("connect: {error}"));
    for (index, byte) in bytes.iter().enumerate() {
        client
            .write_all(&[*byte])
            .unwrap_or_else(|error| panic!("write: {error}"));
        client
            .flush()
            .unwrap_or_else(|error| panic!("flush: {error}"));
        if index % 8 == 0 {
            // Force the reader to observe a frame split across many small TCP segments.
            std::thread::sleep(Duration::from_micros(200));
        }
    }
    drop(client);

    let (first_frame, second_frame) = server
        .join()
        .unwrap_or_else(|_panic| panic!("server panicked"));
    assert_eq!(first_frame.0, first);
    assert_eq!(
        decode_bf16(&first_frame.1).unwrap_or_else(|error| panic!("decode: {error}")),
        values
    );
    assert_eq!(second_frame.0, second);
    assert!(second_frame.1.is_empty());
}

#[test]
fn a_corrupt_magic_is_rejected_instead_of_parsed_as_a_payload() {
    let listener =
        TcpListener::bind(("127.0.0.1", 0)).unwrap_or_else(|error| panic!("bind: {error}"));
    let address = listener
        .local_addr()
        .unwrap_or_else(|error| panic!("addr: {error}"));
    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener
            .accept()
            .unwrap_or_else(|error| panic!("accept: {error}"));
        read_frame(&mut stream, 4096).err()
    });
    let mut client = TcpStream::connect(address).unwrap_or_else(|error| panic!("connect: {error}"));
    let header = FrameHeader {
        kind: FrameKind::ReducePayload,
        phase: 0,
        seq: 1,
        rank: 0,
        payload_bytes: 0,
    };
    let mut bytes = header.encode().to_vec();
    bytes[0] ^= 0xff;
    client
        .write_all(&bytes)
        .unwrap_or_else(|error| panic!("write: {error}"));
    client
        .flush()
        .unwrap_or_else(|error| panic!("flush: {error}"));
    let outcome = server
        .join()
        .unwrap_or_else(|_panic| panic!("server panicked"));
    assert!(
        matches!(outcome, Some(CommError::Protocol(_))),
        "a corrupt magic must be a protocol error, got {outcome:?}"
    );
}

#[test]
fn connect_rejects_a_rank_outside_the_collective() {
    let config = TcpConfig {
        rank: 5,
        world_size: 4,
        ..TcpConfig::loopback4(0, 42190)
    };
    assert!(matches!(
        TcpTransport::connect(config),
        Err(CommError::RankOutOfRange { .. })
    ));
}
