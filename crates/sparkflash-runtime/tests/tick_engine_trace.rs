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

//! The tick engine's gate: a fixed workload driven through [sparkflash_runtime::TickEngine] on a
//! CPU walk, its decisions compared with the committed text.
//!
//! The engine runs the dense arm's serving path: it places, holds, chunks and decodes by
//! sparkflash_runtime::tick's functions, and this file pins what those decisions are. The
//! workload mirrors tests/tick_trace.rs (three prompts at once, the long one taking the decode
//! share, a turn that reuses its prefix later), with the one structural difference the engine
//! names: a prompt that finishes its prefill is headed in that tick and its decode row is fed by
//! the caller on the next, where the replay decodes it in the same tick.
//!
//! A second workload pins the batched prompts: several prompts' whole rests in one paired pass,
//! each headed from its own last row, and a chunk a rows state of a long prompt.
//!
//! What this cannot see: the device. Lane memory, growth, adoption from another lane, the
//! chunks' interleaving on the device, preemption and the drafter's depth need a cluster; they
//! are named in docs/engine/04-onboarding-review.md.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use sparkflash_model::backend::{BackendError, ModelSession, Sampling, StepOutput};
use sparkflash_runtime::tick::{Clock, QUEUE};
use sparkflash_runtime::{
    Absent, ComputeDeclaration, HeadRow, Hidden, LaneMemory, LaneState, Limits, NoVision, Pass,
    PassKind, Segment, StateKind, StateLayout, TickConfig, TickEngine, TickEvent, Walk,
    WalkCapabilities,
};

/// A clock the test advances before each tick, so the engine's cost model reads the same scripted
/// costs the replay does.
#[derive(Clone, Default)]
struct ScriptedClock(Arc<AtomicU64>);

impl ScriptedClock {
    fn advance(&self, ms: u64) {
        self.0.fetch_add(ms, Ordering::Relaxed);
    }
}

impl Clock for ScriptedClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// A sequence is the tokens it forwarded and still holds.
#[derive(Debug, Default)]
struct Seq {
    rows: Vec<u32>,
}

impl LaneState for Seq {
    fn position(&self) -> usize {
        self.rows.len()
    }

    fn checkpoint(&mut self, _at: usize) -> Result<(), BackendError> {
        Ok(())
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        self.rows.truncate(to);
        Ok(())
    }

    fn room(&self, _positions: usize, _reset: bool) -> bool {
        true
    }
}

/// The CPU walk the gate drives: a pass appends its rows, and the head reads the last one.
struct ReplayWalk {
    layout: StateLayout,
    limits: Limits,
    /// How many walks the engine asked for the tick-ahead hook.
    hooks: Arc<AtomicUsize>,
    /// Every pass and head the engine ran, as text: the pass shape with each segment's rows,
    /// and the residual rows a head read.
    passes: Arc<Mutex<Vec<String>>>,
}

impl ReplayWalk {
    fn new(rows: usize, context: usize) -> Self {
        Self::new_with_hooks(rows, context, Arc::new(AtomicUsize::new(0)))
    }

    fn new_with_hooks(rows: usize, context: usize, hooks: Arc<AtomicUsize>) -> Self {
        Self {
            layout: StateLayout::new(Vec::new()),
            // Three chunks a paired pass: DeepSeek's three bulk rows states.
            limits: Limits::new(rows, context).with_chunks(3),
            hooks,
            passes: Arc::default(),
        }
    }
}

impl Walk for ReplayWalk {
    type Seq = Seq;
    type Drafter = Absent<&'static str>;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: sparkflash_runtime::PassMask::ALL,
        segments: sparkflash_runtime::Segments::Many,
        state: StateKind::Positional,
        compute: ComputeDeclaration::none(
            "the gate walks no tensor cores",
            "the gate walks no tensor cores",
            "the gate stores no weights",
            "the gate keeps every position",
        ),
    };

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self, _memory: LaneMemory) -> Result<Self::Seq, BackendError> {
        Ok(Seq::default())
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        static ABSENT: Absent<&'static str> = Absent("the gate walks no drafter");
        &ABSENT
    }

    fn set_sampling(&self, _seq: &mut Self::Seq, _sampling: Sampling) {}

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        mut pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let mut rows = 0;
        let mut each = Vec::new();
        for segment in segments {
            segment.seq.rows.extend_from_slice(segment.rows);
            rows += segment.rows.len();
            each.push(segment.rows.len().to_string());
        }
        let shape = match pass.kind {
            PassKind::TokenRows => "rows",
            PassKind::Bulk { .. } => "bulk",
            PassKind::Paired { .. } => "paired",
            PassKind::Verify { .. } => "verify",
        };
        self.passes
            .lock()
            .expect("the pass log")
            .push(format!("{shape} {}", each.join("+")));
        let hidden = Hidden::new(rows);
        if let Some(hook) = pass.hook.take() {
            self.hooks.fetch_add(1, Ordering::Relaxed);
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        hidden: &Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        let _ = hidden;
        let read: Vec<String> = rows.iter().map(|row| row.row.to_string()).collect();
        self.passes
            .lock()
            .expect("the pass log")
            .push(format!("head {}", read.join(",")));
        Ok(rows
            .iter_mut()
            .map(|row| {
                let token = row.seq.rows.last().copied().unwrap_or(0);
                let _ = &row.input;
                StepOutput {
                    logits: Vec::new(),
                    next_token: Some(token.wrapping_add(1)),
                    draft: Vec::new(),
                }
            })
            .collect())
    }
}

/// A prompt of len deterministic tokens.
fn tokens(seed: u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| 1 + (seed.wrapping_mul(7919).wrapping_add(i as u32 * 31)) % 90_000)
        .collect()
}

/// A prompt that begins exactly like head and continues on its own.
fn shares(head: &[u32], len: usize) -> Vec<u32> {
    let mut tokens = head[..len.min(head.len())].to_vec();
    tokens.extend((tokens.len()..len).map(|i| 1 + (i as u32 * 131) % 90_000));
    tokens
}

/// The committed decisions.
fn expected() -> &'static str {
    include_str!("data/tick_engine_trace.txt")
}

/// Drives the fixed workload and returns the engine's rendered trace.
fn run() -> String {
    let clock = ScriptedClock::default();
    let walk = ReplayWalk::new(8_192, 200_000);
    let mut engine = TickEngine::new(walk, 3)
        .expect("the engine starts")
        .with_config(TickConfig::deepseek(3))
        .expect("the deepseek config fits the walk")
        .with_clock(Box::new(clock.clone()))
        .with_trace();
    let long = tokens(1, 30_000);
    let first = engine
        .admit(&long, &[], Sampling::default())
        .expect("seq 1");
    let second = engine
        .admit(&shares(&long, 2_000), &[], Sampling::default())
        .expect("seq 2");
    let third = engine
        .admit(&tokens(3, 80), &[], Sampling::default())
        .expect("seq 3");
    // Each request's next token, once its last step has chosen one: the caller feeds it on the
    // next tick, which is the engine's one structural difference from the replay.
    let mut next: Vec<(sparkflash_runtime::RequestId, u32)> =
        vec![(first, 0), (second, 0), (third, 0)];
    let mut live: Vec<sparkflash_runtime::RequestId> = vec![first, second, third];
    let costs = [
        450u64, 40, 40, 40, 40, 40, 40, 40, 40, 40, 450, 40, 40, 40, 40, 40, 40, 40, 40, 40, 450,
        40, 40, 40, 40, 40, 40, 40, 40, 40,
    ];
    let mut fourth = None;
    for (tick, cost) in costs.into_iter().enumerate() {
        if tick == 9 {
            engine.finish(third);
            live.retain(|id| *id != third);
        }
        if tick == 17 {
            engine.finish(first);
            live.retain(|id| *id != first);
            let id = engine
                .admit(&shares(&long, 25_000), &[], Sampling::default())
                .expect("seq 4");
            fourth = Some(id);
            live.push(id);
            next.push((id, 0));
        }
        if tick == 21 {
            engine.finish(second);
            live.retain(|id| *id != second);
        }
        if tick == 27
            && let Some(seq4) = fourth
        {
            engine.finish(seq4);
            live.retain(|id| *id != seq4);
        }
        // Feed every decoding request the token its last answer chose. The caller owns the
        // token, so a request fed this tick runs its decode row this tick.
        let mut feed = Vec::new();
        for (id, token) in &next {
            if live.contains(id) && *token != 0 {
                feed.push((*id, *token));
            }
        }
        for (id, token) in feed {
            engine
                .feed(id, token)
                .unwrap_or_else(|error| panic!("feed {}: {error}", id.0));
            if let Some(entry) = next.iter_mut().find(|(other, _)| *other == id) {
                entry.1 = 0;
            }
        }
        clock.advance(cost);
        let events = engine.tick().expect("the tick");
        for event in events {
            match event {
                TickEvent::Prompt { id, step } => {
                    if let Some(entry) = next.iter_mut().find(|(other, _)| *other == id) {
                        entry.1 = step.next_token.unwrap_or(0);
                    }
                }
                TickEvent::Decode { id, steps } => {
                    let token = steps.last().and_then(|step| step.next_token).unwrap_or(0);
                    if let Some(entry) = next.iter_mut().find(|(other, _)| *other == id) {
                        entry.1 = token;
                    }
                }
                TickEvent::Failed { id, error } => {
                    panic!("request {} failed: {error}", id.0);
                }
            }
        }
    }
    engine.trace().expect("the engine keeps a trace").render()
}

#[test]
fn the_serving_engine_answers_two_sessions_and_keeps_a_lane_s_prefix() {
    let hooks = Arc::new(AtomicUsize::new(0));
    let engine = sparkflash_runtime::TickServe::start(
        ReplayWalk::new_with_hooks(16, 64, Arc::clone(&hooks)),
        2,
    )
    .expect("the serving engine starts");
    let mut alone = engine.open().expect("a handle");
    let mut other = engine.open().expect("a handle");
    let first = alone.prefill(&[1, 2, 3]).expect("the prompt");
    let second = other.prefill(&[4, 5]).expect("the other prompt");
    let next = first.next_token.expect("a token");
    assert_eq!(
        alone.decode(next).expect("the step").next_token,
        Some(next.wrapping_add(1)),
        "a session steps its own walk"
    );
    assert_eq!(
        other
            .decode(second.next_token.expect("a token"))
            .unwrap()
            .next_token,
        Some(7)
    );
    assert_eq!(
        alone.reusable_prefix(),
        Some([1, 2, 3, next].as_slice()),
        "the caller's history"
    );
    assert!(
        hooks.load(Ordering::Relaxed) > 0,
        "the engine asked the walk for the tick-ahead plan"
    );
    alone.park();
    // The lane keeps its history across requests, so a prompt that extends it resumes.
    let longer = [1, 2, 3, next, 9];
    alone.prefill(&longer).expect("the continued prompt");
    assert_eq!(
        alone.cached_tokens(),
        4,
        "the next request resumed the lane's four tokens"
    );
    drop(alone);
    drop(other);
}

/// DeepSeek's queue is the lanes plus [QUEUE] waiting sequences; the next prompt is refused
/// before it is queued, not failed inside a tick.
#[test]
fn the_admission_queue_is_sixty_four_past_the_lanes() {
    let walk = ReplayWalk::new(64, 65_536);
    let mut engine = TickEngine::new(walk, 3).expect("the engine starts");
    for index in 0..3 + QUEUE {
        engine
            .admit(&[index as u32 + 1], &[], Sampling::default())
            .expect("a queued prompt");
    }
    let error = engine
        .admit(&[1], &[], Sampling::default())
        .expect_err("the queue is full");
    assert!(error.message().contains("queue is full"), "{error}");
}

#[test]
fn the_engine_makes_the_committed_decisions() {
    let got = run();
    if std::env::var_os("UPDATE_ENGINE_TRACE").is_some() {
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/tick_engine_trace.txt"
            ),
            &got,
        )
        .expect("write the committed engine trace");
        return;
    }
    assert_eq!(
        got,
        expected(),
        "the tick engine's decisions moved; if that was intended, regenerate          tests/data/tick_engine_trace.txt with UPDATE_ENGINE_TRACE=1"
    );
    assert!(
        got.lines().count() > 40,
        "a trace that records nothing passes vacuously"
    );
    assert!(got.contains("path=bulk"), "a long prompt takes bulk chunks");
    assert!(
        got.contains("path=exact"),
        "a short prompt takes exact rows"
    );
    assert!(
        got.contains("hold 1"),
        "a long chunk is held for the decoders"
    );
    assert!(
        got.contains("plan=plain"),
        "a non-drafting walk steps plain rows"
    );
}

/// Four prompts on four lanes, nothing decoding: three end in one paired pass (a rows state
/// each, in prefill order), each headed from its own last row; the fourth, which shares most of
/// its rest with a picked one, waits and runs alone the tick after, beside the decoders.
#[test]
fn short_prompts_end_in_one_paired_pass_each_headed_from_its_own_row() {
    let clock = ScriptedClock::default();
    let walk = ReplayWalk::new(8_192, 200_000);
    let passes = Arc::clone(&walk.passes);
    let mut engine = TickEngine::new(walk, 4)
        .expect("the engine starts")
        .with_config(TickConfig::deepseek(4))
        .expect("the deepseek config fits the walk")
        .with_clock(Box::new(clock.clone()))
        .with_trace();
    let lead = tokens(5, 1_500);
    let ids = [
        engine
            .admit(&lead, &[], Sampling::default())
            .expect("seq 0"),
        engine
            .admit(&tokens(6, 900), &[], Sampling::default())
            .expect("seq 1"),
        engine
            .admit(&tokens(7, 1_800), &[], Sampling::default())
            .expect("seq 2"),
        engine
            .admit(&shares(&lead[..1_000], 1_200), &[], Sampling::default())
            .expect("seq 3"),
    ];
    clock.advance(100);
    let first = engine.tick().expect("the first tick");
    let ended: Vec<u64> = first
        .iter()
        .map(|event| match event {
            TickEvent::Prompt { id, step } => {
                assert!(step.next_token.is_some());
                id.0
            }
            TickEvent::Decode { id, .. } => panic!("request {} decoded", id.0),
            TickEvent::Failed { id, error } => panic!("request {} failed: {error}", id.0),
        })
        .collect();
    // Fewest rows left first: 900, then the 1,200 that shares with the lead of 1,500 (so that
    // one waits), then 1,800.
    assert_eq!(ended, vec![ids[1].0, ids[3].0, ids[2].0]);
    assert_eq!(
        *passes.lock().expect("the pass log"),
        vec![
            "paired 900+1200+1800".to_string(),
            "head 899,2099,3899".to_string()
        ],
        "one pass, one segment a prompt, each prompt's last row headed"
    );
    clock.advance(40);
    let second = engine.tick().expect("the second tick");
    assert!(
        matches!(second.as_slice(), [TickEvent::Prompt { id, .. }] if *id == ids[0]),
        "the prompt that waited ends alone"
    );
    let trace = engine.trace().expect("the engine keeps a trace").render();
    let want = "\
t=1 clock=100 place seq=0 lane=0 shared=0 start=0 target=1500 path=bulk
t=1 clock=100 place seq=1 lane=1 shared=0 start=0 target=900 path=bulk
t=1 clock=100 place seq=2 lane=2 shared=0 start=0 target=1800 path=bulk
t=1 clock=100 place seq=3 lane=3 shared=0 start=0 target=1200 path=bulk
t=1 clock=100 hold 0
t=1 clock=100 prefill seq=1 from=0 rows=900 to=900 target=900 path=bulk
t=1 clock=100 prefill seq=3 from=0 rows=1200 to=1200 target=1200 path=bulk
t=1 clock=100 prefill seq=2 from=0 rows=1800 to=1800 target=1800 path=bulk
t=2 clock=140 hold 0
t=2 clock=140 prefill seq=0 from=0 rows=1500 to=1500 target=1500 path=bulk
";
    assert_eq!(trace, want);
    assert_eq!(
        passes.lock().expect("the pass log")[2..],
        ["bulk 1500".to_string(), "head 1499".to_string()]
    );
}

/// A config that asks for more chunks a tick than the walk's paired pass holds is refused, not
/// run one at a time.
#[test]
fn a_config_with_more_chunks_than_the_pass_holds_is_refused() {
    let mut walk = ReplayWalk::new(8_192, 200_000);
    walk.limits = walk.limits.with_chunks(2);
    let error = TickEngine::new(walk, 3)
        .expect("the engine starts")
        .with_config(TickConfig::deepseek(3))
        .err()
        .expect("three chunks do not fit a pass of two");
    assert!(
        error.to_string().contains("3 chunks a tick"),
        "the refusal names the chunks: {error}"
    );
}
