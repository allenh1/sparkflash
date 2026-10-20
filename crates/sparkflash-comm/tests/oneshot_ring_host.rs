//! The ring/chunked host twin for prefill-sized collectives, exercised end to end.
//!
//! This file proves the things a prefill-sized ring path can prove without a device: the payload
//! is split into one-shot exchanges on row boundaries, every chunk and every byte of the whole
//! payload matches the canonical one-rounding twin, the reduction order and the single rounding do
//! not move with the ring depth or the chunk boundaries, and the two-deep pipeline really overlaps
//! (the read-done epoch is load-bearing, demonstrated by corrupting a slot when it is ignored).
//!
//! What none of this claims: prefill latency, per-rail bandwidth, correctness across the four-node
//! mesh, engram gather disk latency, replay determinism or device-side copy absence. Those are
//! named unclaimed in every RingTwinReport.

use sparkflash_comm::oneshot::loopback::{
    KernelDelay, ReadEpochPolicy, RingTwinConfig, run_ring_twin, run_ring_twin_with_payloads,
};
use sparkflash_comm::oneshot::{ONESHOT_RAILS, ONESHOT_ROW_BYTES, ONESHOT_ROW_ELEMENTS};
use sparkflash_comm::{canonical_all_reduce_into, f32_to_bf16_rn};

fn total_above_one_row() -> usize {
    ONESHOT_ROW_ELEMENTS + 1024
}

fn total_with_remainder() -> usize {
    ONESHOT_ROW_ELEMENTS * 2 + 1024
}

#[test]
fn the_ring_twin_reduces_a_prefill_sized_payload_bit_exactly() {
    let total = total_above_one_row();
    let report = run_ring_twin(total, RingTwinConfig::default()).expect("ring twin");
    assert!(
        report.total_bytes > ONESHOT_ROW_BYTES,
        "the payload must be above the 2 MiB one-shot row"
    );
    assert_eq!(report.ring, 2);
    assert_eq!(report.rails, ONESHOT_RAILS);
    assert_eq!(report.chunks, 2);
    assert_eq!(report.chunk_elements, vec![ONESHOT_ROW_ELEMENTS, 1024]);
    assert_eq!(report.chunk_offsets, vec![0, ONESHOT_ROW_BYTES]);
    assert!(
        report.bit_identical,
        "the whole payload must match the twin"
    );
    assert!(report.mismatched_ranks.is_empty());
    assert!(report.device_path_only);
    assert_eq!(report.host_copy_bytes, 0);
    assert_eq!(report.dispatched, 4 * 2);
    assert!(
        report.max_rank_in_flight >= 2,
        "the two-deep pipeline must overlap two chunks, saw {}",
        report.max_rank_in_flight
    );
    assert!(!report.unclaimed.is_empty());
}

#[test]
fn ring_depth_and_chunk_boundaries_do_not_move_a_bit() {
    let total = ONESHOT_ROW_ELEMENTS + 1024;
    let matrix = [
        ("ring2-row", RingTwinConfig::default()),
        (
            "ring2-half",
            RingTwinConfig {
                chunk_elements: Some(ONESHOT_ROW_ELEMENTS / 2),
                ..RingTwinConfig::default()
            },
        ),
        (
            "ring2-quarter",
            RingTwinConfig {
                chunk_elements: Some(ONESHOT_ROW_ELEMENTS / 4),
                ..RingTwinConfig::default()
            },
        ),
        (
            "ring1-row",
            RingTwinConfig {
                ring: 1,
                ..RingTwinConfig::default()
            },
        ),
    ];
    let mut chunk_counts = Vec::new();
    for (name, config) in matrix {
        let ring = config.ring;
        let report = run_ring_twin(total, config).expect(name);
        assert!(
            report.bit_identical,
            "{name}: output must stay bit-identical"
        );
        assert_eq!(report.total_elements, total, "{name}");
        assert!(report.device_path_only, "{name}");
        assert_eq!(report.host_copy_bytes, 0, "{name}");
        if ring == 1 {
            assert_eq!(
                report.max_rank_in_flight, 1,
                "{name}: a one-deep ring cannot overlap two chunks"
            );
        }
        chunk_counts.push(report.chunks);
    }
    assert!(chunk_counts[0] < chunk_counts[1] && chunk_counts[1] < chunk_counts[2]);
}

#[test]
fn one_rounding_survives_the_chunk_boundaries_at_prefill_size() {
    let total = ONESHOT_ROW_ELEMENTS * 2;
    let one = f32_to_bf16_rn(1.0);
    let quarter_ulp = f32_to_bf16_rn(2.0f32.powi(-9));
    let payloads = vec![
        vec![one; total],
        vec![quarter_ulp; total],
        vec![quarter_ulp; total],
        vec![quarter_ulp; total],
    ];
    let report = run_ring_twin_with_payloads(total, RingTwinConfig::default(), payloads.clone())
        .expect("ring twin");
    assert!(
        report.bit_identical,
        "one rounding over the whole prefill payload must hold at every chunk boundary"
    );

    // The same payload through a four-rounding serial chain stays at 1.0; assert the two rules
    // really differ at positions inside and across chunks, so the test above is not vacuous.
    let rows: Vec<&[u16]> = payloads.iter().map(Vec::as_slice).collect();
    let mut canonical = vec![0u16; total];
    canonical_all_reduce_into(&mut canonical, &rows).expect("canonical");
    let boundaries = [0, ONESHOT_ROW_ELEMENTS - 1, ONESHOT_ROW_ELEMENTS, total - 1];
    for position in boundaries {
        assert_eq!(
            canonical[position],
            f32_to_bf16_rn(1.0078125),
            "one rounding rounds the quarter ulps up at {position}"
        );
        let mut serial = rows[0][position];
        for row in &rows[1..] {
            let pair = [serial, row[position]];
            let mut step = vec![0u16; 1];
            canonical_all_reduce_into(&mut step, &[&pair[0..1], &pair[1..2]]).expect("pair");
            serial = step[0];
        }
        assert_eq!(serial, one, "four roundings stay at 1.0 at {position}");
        assert_ne!(canonical[position], serial);
    }
}

#[test]
fn the_two_deep_pipeline_overlaps_and_the_hazard_is_real() {
    let total = total_with_remainder();
    let delay = KernelDelay {
        rank: 0,
        chunk: 0,
        duration: std::time::Duration::from_millis(200),
    };
    let honored = run_ring_twin(
        total,
        RingTwinConfig {
            kernel_delay: Some(delay),
            ..RingTwinConfig::default()
        },
    )
    .expect("honored run");
    assert!(
        honored.bit_identical,
        "the honest run must stay bit-identical while the pipeline overlaps"
    );
    assert!(
        honored.max_rank_in_flight >= 2,
        "two slots must be in flight at once, saw {}",
        honored.max_rank_in_flight
    );
    assert!(
        honored.elapsed >= delay.duration,
        "the delayed kernel must actually have been the long pole"
    );

    // The same run with the read-done epoch ignored: the proxy overwrites slot 0 with chunk 2
    // while rank 0's kernel is still reading it for chunk 0. The output must diverge.
    let hazard = run_ring_twin(
        total,
        RingTwinConfig {
            kernel_delay: Some(delay),
            read_epoch_policy: ReadEpochPolicy::Ignore,
            ..RingTwinConfig::default()
        },
    );
    match hazard {
        Ok(report) => {
            assert!(
                !report.bit_identical,
                "ignoring the read-done epoch must corrupt at least one rank; mismatched={:?} max_in_flight={} elapsed={:?} chunks={:?}",
                report.mismatched_ranks,
                report.max_rank_in_flight,
                report.elapsed,
                report.chunk_elements
            );
            assert!(
                report.mismatched_ranks.contains(&0),
                "the delayed rank must be corrupted, mismatched: {:?}",
                report.mismatched_ranks
            );
        }
        Err(error) => panic!("the hazard run did not complete, so it proved no hazard: {error}"),
    }
}

#[test]
fn a_one_deep_ring_serialises_and_stays_correct() {
    let total = total_above_one_row();
    let report = run_ring_twin(
        total,
        RingTwinConfig {
            ring: 1,
            ..RingTwinConfig::default()
        },
    )
    .expect("one-deep ring");
    assert!(report.bit_identical);
    assert_eq!(report.max_rank_in_flight, 1);
    assert_eq!(report.chunks, 2);
    assert_eq!(report.dispatched, 4 * 2);
    assert_eq!(report.host_copy_bytes, 0);
    assert!(report.device_path_only);
}

#[test]
fn a_payload_that_fits_one_row_is_not_a_ring_plan() {
    let error = run_ring_twin(ONESHOT_ROW_ELEMENTS, RingTwinConfig::default())
        .expect_err("one row is not a ring plan");
    assert!(
        error.to_string().contains("needs more than one chunk"),
        "{error}"
    );
}
