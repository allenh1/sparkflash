//! The four-rank serving-orchestration host twin, exercised end to end.
//!
//! These tests pin the host-provable half of S5.5: every rank must be ready before a request is
//! fanned out, the first-token collective really runs through the loopback device seam, every
//! served token must be the canonical one-rounding token, and a dead, silent or disagreeing rank
//! must become a typed failure instead of a quietly wrong answer. Each failure test also runs the
//! same policy with the guard switched off, and requires that the guarded run refuses while the
//! naive run serves a token that differs from the canonical one - the wrong answer the guard exists
//! to prevent.
//!
//! Nothing here is a node measurement: there are no four processes, no real startup, no real
//! transport failure and no real watchdog deadline. Every report names those as unclaimed.

use sparkflash_comm::oneshot::loopback::{
    FailureKind, FailurePolicy, GuardPolicy, OrchestrationConfig, ServingRequest, StartupStatus,
    canonical_first_token, run_orchestration, shard_first_token,
};

fn requests() -> Vec<ServingRequest> {
    (0..3).map(|id| ServingRequest::new(id, 0x5f)).collect()
}

fn config(guard: GuardPolicy, failure: FailurePolicy) -> OrchestrationConfig {
    OrchestrationConfig {
        requests: requests(),
        guard,
        failure,
        ..OrchestrationConfig::default()
    }
}

#[test]
fn four_ranks_agree_on_the_first_token_behind_the_seam() {
    let config = config(GuardPolicy::guarded(), FailurePolicy::default());
    let expected: Vec<u16> = config.requests.iter().map(canonical_first_token).collect();
    assert!(
        expected.iter().all(|token| *token == 9),
        "the synthetic shards reduce to token 9, got {expected:?}"
    );
    let report = run_orchestration(config).expect("orchestration");
    assert_eq!(report.startup, StartupStatus::Ready);
    assert!(report.all_agreed);
    assert_eq!(report.outcomes.len(), 3);
    for outcome in &report.outcomes {
        assert_eq!(
            outcome.served_token(),
            Some(9),
            "a guarded run must serve the canonical token"
        );
    }
    assert_eq!(report.collective_calls, 3);
    assert_eq!(report.dispatches, 12);
    assert_eq!(report.host_copy_bytes, 0);
    assert!(report.device_path_only);
    assert_eq!(report.fan_out_bytes, 3 * 3 * 16);
    assert!(
        report.rail_ledger.rail0_bytes > 0 && report.rail_ledger.rail1_bytes > 0,
        "the first-token collective must post both rails"
    );
    for (rank, served) in report.per_rank_served.iter().enumerate() {
        assert_eq!(
            served,
            &vec![Some(9); 3],
            "rank {rank} must broadcast-receive every served token"
        );
    }
    assert!(!report.unclaimed.is_empty());
}

#[test]
fn a_rank_that_never_starts_refuses_before_any_request_is_served() {
    let failure = FailurePolicy {
        startup_failure: Some(2),
        ..FailurePolicy::default()
    };
    let report = run_orchestration(config(GuardPolicy::guarded(), failure)).expect("orchestration");
    assert_eq!(report.startup, StartupStatus::Refused { missing: vec![2] });
    assert!(report.startup.refused());
    assert!(report.outcomes.is_empty());
    assert_eq!(report.collective_calls, 0);
    assert_eq!(report.fan_out_bytes, 0);
    assert!(!report.all_agreed);
}

#[test]
fn a_rank_that_dies_mid_request_is_a_typed_failure_with_the_guard() {
    let mut failure = FailurePolicy::default();
    failure
        .set_fate(
            1,
            sparkflash_comm::oneshot::loopback::RankFate::Die { at_request: 1 },
        )
        .expect("fate");
    let guarded =
        run_orchestration(config(GuardPolicy::guarded(), failure.clone())).expect("guarded");
    assert_eq!(guarded.startup, StartupStatus::Ready);
    assert_eq!(guarded.outcomes[0].served_token(), Some(9));
    assert_eq!(
        guarded.outcomes[1].failure(),
        Some(&FailureKind::DeadRank { rank: 1 })
    );
    assert_eq!(
        guarded.outcomes[2].failure(),
        Some(&FailureKind::DeadRank { rank: 1 })
    );
    assert!(!guarded.all_agreed);

    let naive = run_orchestration(config(GuardPolicy::naive(), failure)).expect("naive");
    assert_eq!(naive.outcomes[0].served_token(), Some(9));
    let wrong = naive.outcomes[1]
        .served_token()
        .expect("naive serves anyway");
    assert_ne!(
        wrong,
        canonical_first_token(&requests()[1]),
        "without the liveness guard the coordinator must serve the partial answer"
    );
    assert_eq!(wrong, shard_first_token(0, &requests()[1]));
}

#[test]
fn a_rank_that_hangs_is_a_typed_failure_with_the_guard() {
    let mut failure = FailurePolicy::default();
    failure
        .set_fate(
            2,
            sparkflash_comm::oneshot::loopback::RankFate::Hang { at_request: 1 },
        )
        .expect("fate");
    let guarded =
        run_orchestration(config(GuardPolicy::guarded(), failure.clone())).expect("guarded");
    assert_eq!(
        guarded.outcomes[1].failure(),
        Some(&FailureKind::HungRank { rank: 2 })
    );
    assert_eq!(
        guarded.outcomes[2].failure(),
        Some(&FailureKind::HungRank { rank: 2 })
    );

    let naive = run_orchestration(config(GuardPolicy::naive(), failure)).expect("naive");
    let wrong = naive.outcomes[1]
        .served_token()
        .expect("naive serves anyway");
    assert_ne!(
        wrong,
        canonical_first_token(&requests()[1]),
        "without the liveness guard a silent rank yields the partial answer"
    );
}

#[test]
fn a_rank_that_returns_a_different_token_is_refused_with_the_guard() {
    let mut failure = FailurePolicy::default();
    failure
        .set_fate(
            0,
            sparkflash_comm::oneshot::loopback::RankFate::CorruptToken {
                at_request: 0,
                token: 3,
            },
        )
        .expect("fate");
    let guarded =
        run_orchestration(config(GuardPolicy::guarded(), failure.clone())).expect("guarded");
    assert_eq!(
        guarded.outcomes[0].failure(),
        Some(&FailureKind::Disagreement {
            tokens: vec![3, 9, 9, 9]
        }),
        "the coordinator must refuse a split vote"
    );
    assert_eq!(guarded.outcomes[1].served_token(), Some(9));

    let naive = run_orchestration(config(GuardPolicy::naive(), failure)).expect("naive");
    assert_eq!(
        naive.outcomes[0].served_token(),
        Some(3),
        "without the agreement guard the corrupted vote is served"
    );
    assert_ne!(
        naive.outcomes[0].served_token(),
        Some(canonical_first_token(&requests()[0])),
        "the served token is the wrong answer"
    );
}

#[test]
fn the_same_failure_policy_gives_the_same_evidence_twice() {
    let mut failure = FailurePolicy::default();
    failure
        .set_fate(
            0,
            sparkflash_comm::oneshot::loopback::RankFate::CorruptToken {
                at_request: 0,
                token: 3,
            },
        )
        .expect("fate");
    let first = run_orchestration(config(GuardPolicy::guarded(), failure.clone())).expect("first");
    let second = run_orchestration(config(GuardPolicy::guarded(), failure)).expect("second");
    assert_eq!(first.outcomes, second.outcomes);
    assert_eq!(first.per_rank_served, second.per_rank_served);
    assert_eq!(first.collective_calls, second.collective_calls);
}
