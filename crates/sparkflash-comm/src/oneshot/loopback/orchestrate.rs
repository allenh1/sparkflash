//! The four-rank serving-orchestration host twin (task-52 host portion).
//!
//! S5.5 is the stage where four serving processes must behave as one: a coordinator (rank 0) that
//! starts every worker, fans a request out, drives the first-token collective, checks that every
//! rank derived the same token, broadcasts the agreed token, and turns a dead, hung or disagreeing
//! rank into a typed, loud failure instead of a silently wrong answer. This module is that
//! orchestration on the host, behind the loopback device seam: the orchestration itself is
//! in-process (there are no processes here), but the first-token collective really runs through the
//! seam's device path - four rank threads call all_reduce_bf16_device on their shard logits and read
//! the canonical one-rounding result.
//!
//! Failure injection is a policy, never a race. A rank's fate is a value - it dies before request i,
//! goes silent before request i, or returns a chosen token for request i - so every run is
//! deterministic and can be repeated exactly. A guarded run requires readiness before any request,
//! treats a dead or silent rank as a typed failure, and requires every rank's vote to match before
//! it serves; a naive run (the guard policy switched off) is the control: it serves rank 0's token
//! anyway, which the tests show is the wrong answer.
//!
//! What a host twin cannot be: four processes, real startup timing, a real RDMA or TCP failure, a
//! real watchdog deadline, or four nodes agreeing on a real model's logits. Those are named in every
//! [OrchestrationReport::unclaimed] and none may be cited from a host run.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::bf16::{bf16_to_f32, f32_to_bf16_rn};
use crate::error::CommError;
use crate::oneshot::{DispatchLog, ONESHOT_RAILS, ONESHOT_RANKS};
use crate::rail::RailLedger;
use crate::reduce::canonical_all_reduce_into;
use crate::transport::Transport;

use super::{
    HOST_TWIN_RING, HostDeviceBuffer, LoopbackDeviceCluster, LoopbackDeviceConfig, invalid,
};

/// Ranks a serving orchestration runs: the TP=4 world.
pub const ORCHESTRATION_WORLD: usize = ONESHOT_RANKS;

/// Logit slots the synthetic first-token model emits per rank.
pub const FIRST_TOKEN_VOCAB: usize = 16;

/// Bytes one request record costs on the coordinator-to-worker fan-out (a host accounting unit,
/// not a wire format).
pub const REQUEST_RECORD_BYTES: u64 = 16;

/// A serving request, as the orchestration layer sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingRequest {
    /// Request id, echoed in the evidence.
    pub id: u64,
    /// Opaque prompt payload; the synthetic model only keys its evidence on it.
    pub prompt: u64,
}

impl ServingRequest {
    /// Builds a request.
    #[must_use]
    pub fn new(id: u64, prompt: u64) -> Self {
        Self { id, prompt }
    }
}

/// What a rank does to a request: the deterministic failure-injection knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankFate {
    /// The rank participates normally.
    Alive,
    /// The rank dies before its shard logits for the request are published.
    Die {
        /// Request index it dies on.
        at_request: usize,
    },
    /// The rank goes silent before its shard logits for the request are published (the host
    /// stand-in for a hang; a real watchdog deadline is unclaimed).
    Hang {
        /// Request index it goes silent on.
        at_request: usize,
    },
    /// The rank computes the canonically correct logits but casts a chosen token.
    CorruptToken {
        /// Request index the corrupt vote applies to.
        at_request: usize,
        /// The token the rank votes for instead.
        token: u16,
    },
}

/// The guard a run uses. Guarded is the only policy a serving path may use; naive exists so a test
/// can show what the same failure returns without the guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardPolicy {
    /// Whether a dead or silent rank is a typed failure instead of a quietly partial answer.
    pub liveness: bool,
    /// Whether every rank's first token must agree before a token is served.
    pub agreement: bool,
}

impl GuardPolicy {
    /// The serving policy: readiness is required, failures are typed, votes must agree.
    #[must_use]
    pub fn guarded() -> Self {
        Self {
            liveness: true,
            agreement: true,
        }
    }

    /// The deliberate control: no liveness check and no agreement check.
    #[must_use]
    pub fn naive() -> Self {
        Self {
            liveness: false,
            agreement: false,
        }
    }
}

impl Default for GuardPolicy {
    fn default() -> Self {
        Self::guarded()
    }
}

/// The failure policy: which rank is missing at startup and what each rank does to which request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailurePolicy {
    /// A rank that never becomes ready, or None for a clean startup.
    pub startup_failure: Option<usize>,
    /// One fate per rank, indexed by rank.
    pub fates: [RankFate; ORCHESTRATION_WORLD],
}

impl Default for FailurePolicy {
    fn default() -> Self {
        Self {
            startup_failure: None,
            fates: [RankFate::Alive; ORCHESTRATION_WORLD],
        }
    }
}

impl FailurePolicy {
    /// The fate of one rank.
    #[must_use]
    pub fn fate(&self, rank: usize) -> RankFate {
        self.fates[rank]
    }

    /// Sets one rank's fate.
    ///
    /// # Errors
    /// Refuses a rank outside the world.
    pub fn set_fate(&mut self, rank: usize, fate: RankFate) -> Result<(), CommError> {
        if rank >= ORCHESTRATION_WORLD {
            return Err(invalid(format!(
                "rank {rank} is outside the orchestration world of {ORCHESTRATION_WORLD}"
            )));
        }
        self.fates[rank] = fate;
        Ok(())
    }
}

/// Configuration of one orchestration host-twin run.
#[derive(Debug, Clone)]
pub struct OrchestrationConfig {
    /// The requests the coordinator fans out, in order.
    pub requests: Vec<ServingRequest>,
    /// The guard policy.
    pub guard: GuardPolicy,
    /// The deterministic failure injection.
    pub failure: FailurePolicy,
    /// Rails the first-token collective posts: one or two.
    pub rails: usize,
    /// Slots per rank in the seam's ring: one or two.
    pub ring: usize,
    /// Deadline for the collective underneath (a host parameter, not a node timeout).
    pub timeout: Duration,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            requests: (0..3).map(|id| ServingRequest::new(id, 0x5f)).collect(),
            guard: GuardPolicy::guarded(),
            failure: FailurePolicy::default(),
            rails: ONESHOT_RAILS,
            ring: HOST_TWIN_RING,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Whether the four ranks came up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupStatus {
    /// Every rank became ready before any request was served.
    Ready,
    /// The run refused to serve because a rank never became ready.
    Refused {
        /// The ranks that were missing.
        missing: Vec<usize>,
    },
}

impl StartupStatus {
    /// Whether the startup was refused.
    #[must_use]
    pub fn refused(&self) -> bool {
        matches!(self, Self::Refused { .. })
    }

    /// The missing ranks (empty when ready).
    #[must_use]
    pub fn missing(&self) -> &[usize] {
        match self {
            Self::Ready => &[],
            Self::Refused { missing } => missing,
        }
    }
}

/// Why a request was not served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureKind {
    /// A rank never became ready, so no request was served.
    StartupRefused {
        /// The missing ranks.
        missing: Vec<usize>,
    },
    /// A rank died mid-request.
    DeadRank {
        /// The dead rank.
        rank: usize,
    },
    /// A rank went silent mid-request (the host stand-in for a hung rank).
    HungRank {
        /// The silent rank.
        rank: usize,
    },
    /// The ranks derived different first tokens.
    Disagreement {
        /// Every rank's vote, in rank order.
        tokens: Vec<u16>,
    },
}

impl FailureKind {
    /// A short stable label for evidence files.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::StartupRefused { .. } => "startup_refused",
            Self::DeadRank { .. } => "dead_rank",
            Self::HungRank { .. } => "hung_rank",
            Self::Disagreement { .. } => "disagreement",
        }
    }

    /// The rank this failure names, when it names exactly one.
    #[must_use]
    pub fn rank(&self) -> Option<usize> {
        match self {
            Self::DeadRank { rank } | Self::HungRank { rank } => Some(*rank),
            Self::StartupRefused { .. } | Self::Disagreement { .. } => None,
        }
    }
}

/// The outcome the coordinator reported for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestOutcome {
    /// The request was served with a first token and the votes that backed it.
    Served {
        /// The served first token.
        token: u16,
        /// Every vote the coordinator saw, in rank order.
        votes: Vec<u16>,
    },
    /// The request was refused with a typed failure.
    Failed {
        /// Why it was refused.
        kind: FailureKind,
    },
}

impl RequestOutcome {
    /// The served token, when the request was served.
    #[must_use]
    pub fn served_token(&self) -> Option<u16> {
        match self {
            Self::Served { token, .. } => Some(*token),
            Self::Failed { .. } => None,
        }
    }

    /// The failure, when the request was refused.
    #[must_use]
    pub fn failure(&self) -> Option<&FailureKind> {
        match self {
            Self::Served { .. } => None,
            Self::Failed { kind } => Some(kind),
        }
    }
}

/// The measurements a host orchestration twin cannot make; a window must claim them.
const UNCLAIMED_ORCHESTRATION: [&str; 6] = [
    "real four-process startup and readiness timing, including ssh/launch failures",
    "real rank-death detection and teardown timing through the transport, not an in-process status",
    "real watchdog deadlines and their bounds on a stalled rank",
    "first-token and logit agreement across the four-node mesh on the real model",
    "the per-request D40 artifact (logits, logprobs, timings) a window run must emit",
    "per-request latency, throughput and per-collective cost at serving shape",
];

/// What one orchestration host-twin run did, and what stays unclaimed.
#[derive(Debug, Clone)]
pub struct OrchestrationReport {
    /// Ranks in the world.
    pub world_size: usize,
    /// Requests the coordinator fanned out.
    pub requests: usize,
    /// Whether the ranks came up.
    pub startup: StartupStatus,
    /// The coordinator's outcome per request, in order.
    pub outcomes: Vec<RequestOutcome>,
    /// The canonical reference first token for every request, in order.
    pub expected_tokens: Vec<u16>,
    /// The token each rank recorded from the coordinator's broadcast, per rank and request.
    pub per_rank_served: Vec<Vec<Option<u16>>>,
    /// Every rank served the same token for every served request.
    pub all_agreed: bool,
    /// The guard policy the run used.
    pub guard: GuardPolicy,
    /// The failure policy the run used.
    pub failure: FailurePolicy,
    /// First-token collectives the seam ran.
    pub collective_calls: usize,
    /// Bytes the seam's simulated NIC moved.
    pub rail_ledger: RailLedger,
    /// Bytes the coordinator fanned out (host accounting).
    pub fan_out_bytes: u64,
    /// Dispatch records across the four ranks.
    pub dispatches: usize,
    /// Host-staging bytes across every dispatch record; zero for the device path.
    pub host_copy_bytes: usize,
    /// Whether every recorded dispatch took the device path.
    pub device_path_only: bool,
    /// Wall time of the whole four-rank run (a host number, not a node latency).
    pub elapsed: Duration,
    /// Measurements this run cannot produce; the window run must claim them.
    pub unclaimed: Vec<&'static str>,
}

/// The canonical first token a fully reduced logit vector yields for a request.
///
/// The reference path is the single-rounding rule: the four shard logits are accumulated in rank
/// order in f32 and rounded once, then the argmax is taken (lowest index on a tie).
#[must_use]
pub fn canonical_first_token(request: &ServingRequest) -> u16 {
    let shards: Vec<Vec<u16>> = (0..ORCHESTRATION_WORLD)
        .map(|rank| shard_logits(rank, request))
        .collect();
    let rows: Vec<&[u16]> = shards.iter().map(Vec::as_slice).collect();
    let mut reduced = vec![0u16; FIRST_TOKEN_VOCAB];
    if canonical_all_reduce_into(&mut reduced, &rows).is_err() {
        return 0;
    }
    argmax(&reduced)
}

/// The token one rank's shard alone would yield for a request (the wrong answer a naive
/// coordinator serves when it ignores a missing rank).
#[must_use]
pub fn shard_first_token(rank: usize, request: &ServingRequest) -> u16 {
    argmax(&shard_logits(rank, request))
}

/// The synthetic shard logits: rank 0 prefers slot 5, ranks 1-3 collectively prefer slot 9, so the
/// fully reduced token is 9 while rank 0 alone would say 5. That gap is what makes a partial answer
/// detectably wrong in the tests without any timing dependence.
fn shard_logits(rank: usize, _request: &ServingRequest) -> Vec<u16> {
    let mut values = vec![0u16; FIRST_TOKEN_VOCAB];
    match rank {
        0 => values[5] = f32_to_bf16_rn(1.0),
        1 => values[9] = f32_to_bf16_rn(0.8),
        2 => values[9] = f32_to_bf16_rn(0.7),
        3 => values[9] = f32_to_bf16_rn(0.6),
        _ => {}
    }
    values
}

fn argmax(values: &[u16]) -> u16 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        let converted = bf16_to_f32(*value);
        if converted > best_value {
            best_value = converted;
            best = index;
        }
    }
    u16::try_from(best).unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RankState {
    Running,
    Dead,
    Hung,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogitsWait {
    Ready,
    Dead(usize),
    Hung(usize),
}

struct TwinState {
    ready: Vec<bool>,
    ranks: Vec<RankState>,
    request_epoch: u64,
    collective_epoch: u64,
    abort_epoch: u64,
    broadcast_epoch: u64,
    current: Option<ServingRequest>,
    logits_written: Vec<bool>,
    votes: Vec<Option<u16>>,
    served_tokens: Vec<Option<u16>>,
    outcomes: Vec<RequestOutcome>,
    per_rank_served: Vec<Vec<Option<u16>>>,
    startup: StartupStatus,
    finished: bool,
    error: Option<String>,
}

struct Shared {
    state: Mutex<TwinState>,
    changed: Condvar,
}

impl Shared {
    fn new(world: usize, requests: usize) -> Self {
        Self {
            state: Mutex::new(TwinState {
                ready: vec![false; world],
                ranks: vec![RankState::Running; world],
                request_epoch: 0,
                collective_epoch: 0,
                abort_epoch: 0,
                broadcast_epoch: 0,
                current: None,
                logits_written: vec![false; world],
                votes: vec![None; world],
                served_tokens: vec![None; requests],
                outcomes: Vec::new(),
                per_rank_served: (0..world).map(|_| Vec::new()).collect(),
                startup: StartupStatus::Ready,
                finished: false,
                error: None,
            }),
            changed: Condvar::new(),
        }
    }

    fn set_error(&self, error: String) {
        let mut state = self.state.lock();
        if state.error.is_none() {
            state.error = Some(error);
        }
        state.finished = true;
        self.changed.notify_all();
    }
}

/// Runs one four-rank serving orchestration on the host, behind the loopback device seam.
///
/// # Errors
/// Refuses an illegal configuration, a seam failure while building the cluster, or a participant
/// whose collective failed.
pub fn run_orchestration(config: OrchestrationConfig) -> Result<OrchestrationReport, CommError> {
    validate(&config)?;
    let shared = Arc::new(Shared::new(ORCHESTRATION_WORLD, config.requests.len()));
    let cluster = LoopbackDeviceCluster::new(LoopbackDeviceConfig {
        world_size: ORCHESTRATION_WORLD,
        slot_ring: config.ring,
        rails: config.rails,
        timeout: config.timeout,
        ..LoopbackDeviceConfig::default()
    })?;
    let ledger_before = cluster.rail_ledger();
    let started = Instant::now();
    let (coordinator_log, participant_logs) = std::thread::scope(
        |scope| -> Result<(DispatchLog, Vec<DispatchLog>), CommError> {
            let mut handles = Vec::new();
            for rank in 1..ORCHESTRATION_WORLD {
                let shared = Arc::clone(&shared);
                let cluster_ref = &cluster;
                let config_ref = &config;
                handles.push(
                    scope.spawn(move || run_participant(shared, cluster_ref, rank, config_ref)),
                );
            }
            let coordinator = run_coordinator(&shared, &cluster, &config);
            if coordinator.is_err() {
                shared.set_error("the coordinator aborted the run".to_owned());
            }
            let coordinator_log = coordinator?;
            let mut participant_logs = Vec::with_capacity(ORCHESTRATION_WORLD - 1);
            for handle in handles {
                let log = handle.join().map_err(|_| {
                    CommError::Protocol("an orchestration participant panicked".to_owned())
                })??;
                participant_logs.push(log);
            }
            Ok((coordinator_log, participant_logs))
        },
    )?;
    let elapsed = started.elapsed();
    let ledger_after = cluster.rail_ledger();
    let state = shared.state.lock();
    let mut logs = Vec::with_capacity(ORCHESTRATION_WORLD);
    logs.push(coordinator_log);
    logs.extend(participant_logs);
    let dispatches = logs.iter().map(DispatchLog::len).sum();
    let host_copy_bytes = logs.iter().map(DispatchLog::host_copy_bytes).sum();
    let device_path_only = logs.iter().all(|log| log.device_path_ok().is_ok());
    let mut per_rank_served = state.per_rank_served.clone();
    for served in &mut per_rank_served {
        served.resize(config.requests.len(), None);
    }
    let all_agreed = !state.outcomes.is_empty()
        && state.outcomes.iter().all(|outcome| match outcome {
            RequestOutcome::Served { votes, token } => votes.iter().all(|vote| vote == token),
            RequestOutcome::Failed { .. } => false,
        });
    Ok(OrchestrationReport {
        world_size: ORCHESTRATION_WORLD,
        requests: config.requests.len(),
        startup: state.startup.clone(),
        outcomes: state.outcomes.clone(),
        expected_tokens: config.requests.iter().map(canonical_first_token).collect(),
        per_rank_served,
        all_agreed,
        guard: config.guard,
        failure: config.failure.clone(),
        collective_calls: cluster.collectives() as usize / ORCHESTRATION_WORLD,
        rail_ledger: RailLedger {
            rail0_bytes: ledger_after.rail0_bytes - ledger_before.rail0_bytes,
            rail1_bytes: ledger_after.rail1_bytes - ledger_before.rail1_bytes,
            transfers: ledger_after.transfers - ledger_before.transfers,
            element_writes: ledger_after.element_writes - ledger_before.element_writes,
        },
        fan_out_bytes: if state.startup.refused() {
            0
        } else {
            u64::try_from(config.requests.len()).unwrap_or(u64::MAX)
                * u64::try_from(ORCHESTRATION_WORLD - 1).unwrap_or(u64::MAX)
                * REQUEST_RECORD_BYTES
        },
        dispatches,
        host_copy_bytes,
        device_path_only,
        elapsed,
        unclaimed: UNCLAIMED_ORCHESTRATION.to_vec(),
    })
}

fn validate(config: &OrchestrationConfig) -> Result<(), CommError> {
    if let Some(rank) = config.failure.startup_failure
        && (rank == 0 || rank >= ORCHESTRATION_WORLD)
    {
        return Err(invalid(format!(
            "startup_failure rank {rank} must be a worker rank 1..{ORCHESTRATION_WORLD}"
        )));
    }
    for (rank, fate) in config.failure.fates.iter().enumerate() {
        match fate {
            RankFate::Alive => {}
            RankFate::CorruptToken { at_request, .. } => {
                if *at_request >= config.requests.len() {
                    return Err(invalid(format!(
                        "rank {rank}'s corrupt-token request {at_request} is outside the plan"
                    )));
                }
            }
            RankFate::Die { at_request } | RankFate::Hang { at_request } => {
                if rank == 0 {
                    return Err(invalid(
                        "rank 0 is the coordinator; its death is the launcher's teardown, not this twin's policy"
                            .to_owned(),
                    ));
                }
                if *at_request >= config.requests.len() {
                    return Err(invalid(format!(
                        "rank {rank}'s failure request {at_request} is outside the plan"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn run_coordinator(
    shared: &Arc<Shared>,
    cluster: &LoopbackDeviceCluster,
    config: &OrchestrationConfig,
) -> Result<DispatchLog, CommError> {
    let handle = cluster.handle(0)?;
    let output = HostDeviceBuffer::allocate(cluster, 0, FIRST_TOKEN_VOCAB)?;
    {
        let mut state = shared.state.lock();
        state.ready[0] = true;
        shared.changed.notify_all();
    }
    // Readiness handshake: serve nothing until every rank is up, and refuse loudly if one is gone.
    let startup = {
        let mut state = shared.state.lock();
        loop {
            if state.ranks.contains(&RankState::Dead) {
                let missing: Vec<usize> = state
                    .ranks
                    .iter()
                    .enumerate()
                    .filter(|(_, rank)| **rank == RankState::Dead)
                    .map(|(rank, _)| rank)
                    .collect();
                state.startup = StartupStatus::Refused { missing };
                state.ranks[0] = RankState::Finished;
                state.finished = true;
                shared.changed.notify_all();
                return Ok(handle.dispatch_log());
            }
            if state.ready.iter().all(|ready| *ready) {
                break StartupStatus::Ready;
            }
            shared.changed.wait(&mut state);
        }
    };
    shared.state.lock().startup = startup;
    for (index, request) in config.requests.iter().enumerate() {
        {
            let mut state = shared.state.lock();
            if state.error.is_some() {
                return Err(CommError::Protocol(
                    "a participant failed before the fan-out".to_owned(),
                ));
            }
            state.current = Some(request.clone());
            state.request_epoch = index_u64(index) + 1;
            state
                .logits_written
                .iter_mut()
                .for_each(|flag| *flag = false);
            state.votes.iter_mut().for_each(|vote| *vote = None);
            shared.changed.notify_all();
        }
        let own_logits = shard_logits(0, request);
        output.write(&own_logits)?;
        {
            let mut state = shared.state.lock();
            state.logits_written[0] = true;
            shared.changed.notify_all();
        }
        let wait = {
            let mut state = shared.state.lock();
            loop {
                if let Some(error) = &state.error {
                    return Err(CommError::Protocol(format!("participant failed: {error}")));
                }
                if let Some(rank) = state.ranks.iter().position(|rank| *rank == RankState::Dead) {
                    break LogitsWait::Dead(rank);
                }
                if let Some(rank) = state.ranks.iter().position(|rank| *rank == RankState::Hung) {
                    break LogitsWait::Hung(rank);
                }
                if state.logits_written.iter().all(|written| *written) {
                    break LogitsWait::Ready;
                }
                shared.changed.wait(&mut state);
            }
        };
        let outcome = match wait {
            LogitsWait::Dead(rank) => {
                if config.guard.liveness {
                    RequestOutcome::Failed {
                        kind: FailureKind::DeadRank { rank },
                    }
                } else {
                    naive_outcome(0, request)
                }
            }
            LogitsWait::Hung(rank) => {
                if config.guard.liveness {
                    RequestOutcome::Failed {
                        kind: FailureKind::HungRank { rank },
                    }
                } else {
                    naive_outcome(0, request)
                }
            }
            LogitsWait::Ready => {
                {
                    let mut state = shared.state.lock();
                    state.collective_epoch = index_u64(index) + 1;
                    shared.changed.notify_all();
                }
                handle.all_reduce_bf16_device(
                    &output.device_buffer(),
                    FIRST_TOKEN_VOCAB,
                    index_u64(index),
                )?;
                let reduced = output.read();
                let mut token = argmax(&reduced);
                if let RankFate::CorruptToken {
                    at_request,
                    token: corrupt,
                } = config.failure.fates[0]
                    && at_request == index
                {
                    token = corrupt;
                }
                {
                    let mut state = shared.state.lock();
                    state.votes[0] = Some(token);
                    shared.changed.notify_all();
                }
                let votes = {
                    let mut state = shared.state.lock();
                    loop {
                        if let Some(error) = &state.error {
                            return Err(CommError::Protocol(format!(
                                "participant failed: {error}"
                            )));
                        }
                        if let Some(rank) =
                            state.ranks.iter().position(|rank| *rank == RankState::Dead)
                        {
                            return Err(CommError::Protocol(format!(
                                "rank {rank} died after the first-token collective began"
                            )));
                        }
                        if state.votes.iter().all(Option::is_some) {
                            break state
                                .votes
                                .iter()
                                .map(|vote| vote.unwrap_or_default())
                                .collect::<Vec<u16>>();
                        }
                        shared.changed.wait(&mut state);
                    }
                };
                if config.guard.agreement {
                    let first = votes[0];
                    if votes.iter().all(|vote| *vote == first) {
                        RequestOutcome::Served {
                            token: first,
                            votes,
                        }
                    } else {
                        RequestOutcome::Failed {
                            kind: FailureKind::Disagreement { tokens: votes },
                        }
                    }
                } else {
                    RequestOutcome::Served {
                        token: votes[0],
                        votes,
                    }
                }
            }
        };
        {
            let mut state = shared.state.lock();
            state.served_tokens[index] = outcome.served_token();
            state.per_rank_served[0].push(outcome.served_token());
            state.outcomes.push(outcome);
            state.broadcast_epoch = index_u64(index) + 1;
            state.abort_epoch = index_u64(index) + 1;
            shared.changed.notify_all();
        }
        {
            // If another request follows, the next loop iteration publishes its epoch and clears
            // the per-request flags under one lock, so a rank that wakes early can never have its
            // freshly written logit flag cleared by that reset.
            let mut state = shared.state.lock();
            if index + 1 == config.requests.len() {
                state.ranks[0] = RankState::Finished;
                state.finished = true;
                shared.changed.notify_all();
            }
        }
    }
    Ok(handle.dispatch_log())
}

fn run_participant(
    shared: Arc<Shared>,
    cluster: &LoopbackDeviceCluster,
    rank: usize,
    config: &OrchestrationConfig,
) -> Result<DispatchLog, CommError> {
    let handle = cluster.handle(rank)?;
    let output = HostDeviceBuffer::allocate(cluster, rank, FIRST_TOKEN_VOCAB)?;
    if config.failure.startup_failure == Some(rank) {
        let mut state = shared.state.lock();
        state.ranks[rank] = RankState::Dead;
        shared.changed.notify_all();
        drop(state);
        return Ok(handle.dispatch_log());
    }
    {
        let mut state = shared.state.lock();
        state.ready[rank] = true;
        shared.changed.notify_all();
    }
    for index in 0..config.requests.len() {
        let fanned = {
            let mut state = shared.state.lock();
            loop {
                if state.finished {
                    state.ranks[rank] = RankState::Finished;
                    shared.changed.notify_all();
                    return Ok(handle.dispatch_log());
                }
                if state.request_epoch > index_u64(index) {
                    break state.current.clone();
                }
                shared.changed.wait(&mut state);
            }
        };
        let Some(request) = fanned else {
            continue;
        };
        match config.failure.fates[rank] {
            RankFate::Die { at_request } if at_request == index => {
                let mut state = shared.state.lock();
                state.ranks[rank] = RankState::Dead;
                shared.changed.notify_all();
                drop(state);
                return Ok(handle.dispatch_log());
            }
            RankFate::Hang { at_request } if at_request == index => {
                let mut state = shared.state.lock();
                state.ranks[rank] = RankState::Hung;
                shared.changed.notify_all();
                drop(state);
                return Ok(handle.dispatch_log());
            }
            _ => {}
        }
        let logits = shard_logits(rank, &request);
        output.write(&logits)?;
        {
            let mut state = shared.state.lock();
            state.logits_written[rank] = true;
            shared.changed.notify_all();
        }
        let run_collective = {
            let mut state = shared.state.lock();
            loop {
                if state.finished {
                    state.ranks[rank] = RankState::Finished;
                    shared.changed.notify_all();
                    return Ok(handle.dispatch_log());
                }
                if let Some(error) = &state.error {
                    return Err(CommError::Protocol(format!(
                        "orchestration aborted: {error}"
                    )));
                }
                if state.abort_epoch > index_u64(index) {
                    break false;
                }
                if state.collective_epoch > index_u64(index) {
                    break true;
                }
                shared.changed.wait(&mut state);
            }
        };
        if run_collective {
            handle.all_reduce_bf16_device(
                &output.device_buffer(),
                FIRST_TOKEN_VOCAB,
                index_u64(index),
            )?;
            let reduced = output.read();
            let mut token = argmax(&reduced);
            if let RankFate::CorruptToken {
                at_request,
                token: corrupt,
            } = config.failure.fates[rank]
                && at_request == index
            {
                token = corrupt;
            }
            {
                let mut state = shared.state.lock();
                state.votes[rank] = Some(token);
                shared.changed.notify_all();
            }
        }
        {
            let mut state = shared.state.lock();
            loop {
                if let Some(error) = &state.error {
                    return Err(CommError::Protocol(format!(
                        "orchestration aborted: {error}"
                    )));
                }
                if state.broadcast_epoch > index_u64(index) || state.finished {
                    let served = state.served_tokens[index];
                    state.per_rank_served[rank].push(served);
                    shared.changed.notify_all();
                    if state.finished {
                        state.ranks[rank] = RankState::Finished;
                        shared.changed.notify_all();
                        return Ok(handle.dispatch_log());
                    }
                    break;
                }
                shared.changed.wait(&mut state);
            }
        }
    }
    let mut state = shared.state.lock();
    state.ranks[rank] = RankState::Finished;
    shared.changed.notify_all();
    drop(state);
    Ok(handle.dispatch_log())
}

fn naive_outcome(rank: usize, request: &ServingRequest) -> RequestOutcome {
    let token = shard_first_token(rank, request);
    RequestOutcome::Served {
        token,
        votes: vec![token],
    }
}

fn index_u64(index: usize) -> u64 {
    u64::try_from(index).unwrap_or(u64::MAX)
}
