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

//! The tick group's wire, on a loopback socket and a CPU walk: rank 0 decides every prompt
//! chunk and decode shape, broadcasts it, and a worker runs exactly those rows with that shape.
//!
//! The walk is shaped like GLM's: a prompt chunk of two rows, one row a member, and a drafter
//! whose single draft is the model's own pick, so a decode's round draft is accepted and its
//! held answer drains. The assertions are the frame order the worker was handed, the passes
//! both ranks walked, and the rows each frame named; nothing is re-derived on the worker.

use std::collections::VecDeque;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use sparkflash_comm::control::{Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker};
use sparkflash_model::backend::{BackendError, Sampling, StepOutput};
use sparkflash_model::verdict;
use sparkflash_runtime::{
    Draft, Drafter, HeadInput, HeadRow, Hidden, LaneState, Limits, NoVision, OutputConstraint,
    Pass, PassKind, Region, Runtime, Segment, StateLayout, StepWall, TickEngine, TickEvent,
    TickGroup, Walk, WalkCapabilities, check_rewind, mirror_tick_lanes, mirror_tick_lanes_seen,
};

/// The model's own pick after `token`: the plain path's next token.
fn pick(token: u32) -> u32 {
    token.wrapping_add(1)
}

/// One pass the walk was handed, for the shape assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Shape {
    /// A prompt chunk of this many rows.
    Bulk(usize),
    /// A verify pass, one entry a member's rows.
    Verify(Vec<usize>),
    /// A plain token pass over this many members.
    TokenRows(usize),
}

/// A sequence is the tokens it forwarded and still holds, plus the answers a round queued.
#[derive(Debug, Default)]
struct Seq {
    rows: Vec<u32>,
    pending: VecDeque<(u32, StepOutput)>,
}

impl LaneState for Seq {
    fn position(&self) -> usize {
        self.rows.len()
    }

    fn checkpoint(&mut self, _at: usize) -> Result<(), BackendError> {
        Ok(())
    }

    fn rewind(&mut self, to: usize) -> Result<(), BackendError> {
        check_rewind("toy", to, self.rows.len())?;
        self.rows.truncate(to);
        self.pending.clear();
        Ok(())
    }

    fn room(&self, _positions: usize, _reset: bool) -> bool {
        true
    }
}

/// The wall of a shared round: flat, so the row share gives the members their budget evenly.
struct Wall;

impl StepWall for Wall {
    fn ms(&self, _seqs: usize, _rows: usize) -> f64 {
        1.0
    }
}

static WALL: Wall = Wall;

/// The CPU walk the gate drives: a two-row prompt chunk, a verify width of four, one draft.
struct GroupWalk {
    layout: StateLayout,
    limits: Limits,
    drafter: GroupDrafter,
    /// Every pass the walk was handed, in order.
    shapes: Arc<Mutex<Vec<Shape>>>,
}

impl GroupWalk {
    fn new(shapes: Arc<Mutex<Vec<Shape>>>) -> Self {
        Self {
            layout: StateLayout::new(vec![Region::Positional {
                name: "toy",
                bytes_per_row: 4,
            }]),
            limits: Limits::new(2, 1_024).with_members(4),
            drafter: GroupDrafter,
            shapes,
        }
    }

    fn shapes(&self) -> Vec<Shape> {
        self.shapes.lock().expect("the pass record").clone()
    }
}

struct GroupDrafter;

impl Walk for GroupWalk {
    type Seq = Seq;
    type Hidden = Hidden;
    type Drafter = GroupDrafter;
    type Vision = NoVision;

    const CAPABILITIES: WalkCapabilities =
        WalkCapabilities::all(sparkflash_runtime::StateKind::Positional);

    fn layout(&self) -> &StateLayout {
        &self.layout
    }

    fn open(&self) -> Result<Self::Seq, BackendError> {
        Ok(Seq::default())
    }

    fn limits(&self) -> Limits {
        self.limits
    }

    fn drafter(&self) -> &Self::Drafter {
        &self.drafter
    }

    fn set_sampling(&self, _seq: &mut Self::Seq, _sampling: Sampling) {}

    fn walk(
        &self,
        segments: &mut [Segment<'_, Self::Seq>],
        mut pass: Pass<'_>,
    ) -> Result<Hidden, BackendError> {
        let shape = match pass.kind {
            PassKind::TokenRows => Shape::TokenRows(segments.len()),
            PassKind::Bulk { .. } | PassKind::Paired { .. } => {
                Shape::Bulk(segments.iter().map(|segment| segment.rows.len()).sum())
            }
            PassKind::Verify { .. } => {
                Shape::Verify(segments.iter().map(|segment| segment.rows.len()).collect())
            }
        };
        self.shapes.lock().expect("the pass record").push(shape);
        let mut rows = 0;
        for segment in segments {
            segment.seq.rows.extend_from_slice(segment.rows);
            rows += segment.rows.len();
        }
        let hidden = Hidden::new(rows);
        if let Some(hook) = pass.hook.take() {
            hook.after_layers(&hidden)?;
        }
        Ok(hidden)
    }

    fn head(
        &self,
        _hidden: &mut Hidden,
        rows: &mut [HeadRow<'_, Self::Seq>],
    ) -> Result<Vec<StepOutput>, BackendError> {
        Ok(rows
            .iter_mut()
            .map(|row| {
                let token = match &row.input {
                    HeadInput::Verify { rows, row: index } => {
                        pick(rows.get(*index).copied().unwrap_or_default())
                    }
                    HeadInput::Prompt(_) | HeadInput::Token(_) => {
                        row.seq.rows.last().copied().map_or(0, pick)
                    }
                };
                StepOutput {
                    logits: Vec::new(),
                    next_token: Some(token),
                    draft: Vec::new(),
                }
            })
            .collect())
    }
}

impl Drafter<GroupWalk> for GroupDrafter {
    const PRESENT: bool = true;
    type Rows = ();

    fn wall(&self) -> &dyn StepWall {
        &WALL
    }

    fn open_round(&self, seq: &mut Seq, token: u32) -> Result<Draft, BackendError> {
        if let Some((front, _)) = seq.pending.front()
            && *front == token
        {
            let (_, step) = seq.pending.pop_front().expect("the front was read");
            return Ok(Draft::Held(step));
        }
        // One draft, and it is the model's own pick after the anchor: the round keeps it.
        Ok(Draft::Chain {
            confidence: vec![0.9],
            drafts: vec![pick(token)],
        })
    }

    fn verify(
        &self,
        walk: &GroupWalk,
        segments: &mut [Segment<'_, Seq>],
        _depths: &[usize],
    ) -> Result<Vec<Self::Rows>, BackendError> {
        let rows: usize = segments.iter().map(|segment| segment.rows.len()).sum();
        walk.walk(
            segments,
            Pass::verify(rows, OutputConstraint::new(&[], rows)),
        )?;
        Ok(vec![(); segments.len()])
    }

    fn settle(
        &self,
        seq: &mut Seq,
        chunk: &[u32],
        _rows: Self::Rows,
    ) -> Result<StepOutput, BackendError> {
        let picks: Vec<u32> = chunk.iter().map(|token| pick(*token)).collect();
        let kept = verdict(chunk, &picks);
        let base = seq.rows.len().saturating_sub(chunk.len());
        seq.rows.truncate(base + kept);
        for (index, &token) in chunk.iter().enumerate().take(kept).skip(1) {
            seq.pending.push_back((
                token,
                StepOutput {
                    logits: Vec::new(),
                    next_token: picks.get(index).copied(),
                    draft: Vec::new(),
                },
            ));
        }
        Ok(StepOutput {
            logits: Vec::new(),
            next_token: picks.first().copied(),
            draft: Vec::new(),
        })
    }

    /// One draft a member, so a shared verify holds the anchors and one draft each.
    fn budget(&self, seqs: usize) -> usize {
        seqs
    }

    fn ready(&self, seq: &Seq, token: u32) -> bool {
        seq.pending
            .front()
            .is_some_and(|(front, _)| *front == token)
    }

    fn give_back(&self, seq: &mut Seq, to: usize) -> Result<(), BackendError> {
        seq.rows.truncate(to.min(seq.rows.len()));
        seq.pending.clear();
        Ok(())
    }

    fn speculation(&self, _seq: &Seq) -> sparkflash_model::backend::Speculation {
        sparkflash_model::backend::Speculation::default()
    }
}

/// A group that never splits during a test: a split is a failure the test wants to see.
fn splits(why: &str) -> ! {
    panic!("the group split: {why}")
}

/// Waits for the worker to record a frame count; the control channel is one-way for rank 0, so a
/// test observes the worker rather than reading an acknowledgement.
fn wait_for(mut ready: impl FnMut() -> bool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if ready() {
            return;
        }
        thread::sleep(Duration::from_millis(2));
    }
    panic!("the worker did not {what} within ten seconds");
}

/// One frame as the test names it: the fields a worker's replay depends on, not the whole type.
fn describe(frame: &Tp4ControlFrame) -> String {
    match frame {
        Tp4ControlFrame::LaneRequest {
            lane,
            prompt_ids,
            start,
            ..
        } => format!(
            "request {lane} {prompt_ids:?} start={}",
            start.map_or_else(|| "none".to_owned(), |start| start.to_string())
        ),
        Tp4ControlFrame::LaneRows { lane, rows } => format!("rows {lane} {rows}"),
        Tp4ControlFrame::LaneStepMode { members, mode } => format!("step {members:?} {mode:?}"),
        Tp4ControlFrame::LaneGiveBack { lane, count } => format!("give back {lane} {count}"),
        Tp4ControlFrame::LaneRelease { lane } => format!("release {lane}"),
        other => other.kind().to_owned(),
    }
}

/// The rows a worker was told to run, in the order it was told.
fn row_frames(seen: &[String]) -> Vec<usize> {
    seen.iter()
        .filter_map(|frame| frame.strip_prefix("rows "))
        .map(|rest| {
            rest.split_once(' ')
                .expect("a rows frame names a lane and a count")
                .1
                .parse()
                .expect("a rows frame's count")
        })
        .collect()
}

#[test]
fn a_worker_runs_rank_zeros_rows_and_step_shape_in_the_order_it_broadcasts_them() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the bound address");
    let worker_shapes = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let worker = {
        let shapes = Arc::clone(&worker_shapes);
        let seen = Arc::clone(&seen);
        thread::spawn(move || {
            let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                .expect("a control connection");
            control.handshake().expect("the handshake");
            let runtime = Runtime::new(GroupWalk::new(shapes), 0).expect("the worker's runtime");
            let wire = sparkflash_runtime::TextWire::new("toy", "toy", 1_024);
            let mut record = |frame: &Tp4ControlFrame| {
                seen.lock().expect("the frames").push(describe(frame));
            };
            mirror_tick_lanes_seen(&mut control, &runtime, &wire, 2, &mut record)
        })
    };
    let hub = Arc::new(Mutex::new(
        Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5)).expect("the hub"),
    ));
    let shapes = Arc::new(Mutex::new(Vec::new()));
    let wire = sparkflash_runtime::TextWire::new("toy", "toy", 1_024);
    let mut engine = TickEngine::new(GroupWalk::new(Arc::clone(&shapes)), 2)
        .expect("the engine starts")
        .with_group(TickGroup::new(Arc::clone(&hub), Box::new(wire)).with_split(splits));
    assert!(engine.grouped(), "the engine is rank 0 of a group");
    assert!(
        engine.config().exchange,
        "a group turns the row exchange on, so the prompt takes bulk chunks"
    );
    // Two prompts: lane 0's three rows run in two chunks, lane 1's two rows in one.
    let first = engine
        .admit(&[10, 11, 12], &[], Sampling::default())
        .expect("the first prompt");
    let second = engine
        .admit(&[20, 21], &[], Sampling::default())
        .expect("the second prompt");
    let mut anchors = [None, None];
    while anchors.iter().any(Option::is_none) {
        for event in engine.tick().expect("the tick") {
            if let TickEvent::Prompt { id, step } = event {
                let token = step.next_token.expect("a prompt's first token");
                if id == first {
                    anchors[0] = Some(token);
                } else if id == second {
                    anchors[1] = Some(token);
                }
            }
        }
    }
    let first_anchor = anchors[0].expect("the first prompt's anchor");
    let second_anchor = anchors[1].expect("the second prompt's anchor");
    // A decode step over both lanes: rank 0's plan is one shared round.
    for (id, token) in [(first, first_anchor), (second, second_anchor)] {
        engine.feed(id, token).expect("the row is fed");
    }
    let mut decoded = 0;
    while decoded < 2 {
        for event in engine.tick().expect("the tick") {
            if matches!(event, TickEvent::Decode { .. }) {
                decoded += 1;
            }
        }
    }
    engine.give_back(first, 1).expect("the give-back");

    let expected = vec![
        "request 0 [10, 11, 12] start=0".to_owned(),
        "request 1 [20, 21] start=0".to_owned(),
        "rows 1 2".to_owned(),
        "rows 0 2".to_owned(),
        "rows 0 1".to_owned(),
        format!("step [(0, {first_anchor}), (1, {second_anchor})] Round {{ joint: true }}"),
        "give back 0 1".to_owned(),
    ];
    wait_for(
        || seen.lock().expect("the frames").len() >= expected.len(),
        "record every frame",
    );
    assert_eq!(
        *seen.lock().expect("the frames"),
        expected,
        "the message order"
    );
    // The rows the worker ran are the rows rank 0's chunk decisions named, in the same order.
    assert_eq!(row_frames(&seen.lock().expect("the frames")), vec![2, 2, 1]);
    // Neither rank chose its own shape: the worker's passes are rank 0's, in order.
    let rank0_shapes = engine.walk().shapes();
    assert_eq!(
        rank0_shapes,
        vec![
            Shape::Bulk(2),
            Shape::Bulk(2),
            Shape::Bulk(1),
            Shape::Verify(vec![2, 2]),
        ],
        "rank 0's prompt chunks and its one shared verify"
    );
    wait_for(
        || *worker_shapes.lock().expect("the worker's passes") == rank0_shapes,
        "walk the same passes",
    );
    // End the worker by closing rank 0's control channel; the mirror can no longer serve. Both
    // handles must go: the engine's group holds one arc, the test the other.
    drop(engine);
    drop(hub);
    let error = worker
        .join()
        .expect("the worker finishes")
        .expect_err("a closed control channel ends the mirror");
    assert!(!error.is_empty(), "the mirror says why it stopped");
}

#[test]
fn a_start_that_differs_from_rank_zeros_is_refused_naming_both_positions() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the bound address");
    let worker = thread::spawn(move || {
        let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
            .expect("a control connection");
        control.handshake().expect("the handshake");
        let runtime = Runtime::new(GroupWalk::new(Arc::new(Mutex::new(Vec::new()))), 0)
            .expect("the worker's runtime");
        let wire = sparkflash_runtime::TextWire::new("toy", "toy", 1_024);
        mirror_tick_lanes(&mut control, &runtime, &wire, 1)
    });
    let mut hub =
        Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5)).expect("the hub");
    hub.broadcast(&Tp4ControlFrame::LaneRequest {
        lane: 0,
        prompt_ids: vec![1, 2, 3],
        images: Vec::new(),
        keep: true,
        sampler: None,
        start: Some(0),
    })
    .expect("the first request");
    hub.broadcast(&Tp4ControlFrame::LaneRows { lane: 0, rows: 3 })
        .expect("the first prompt's rows");
    // The lane now holds [1, 2, 3]; this prompt extends it to its third row, so the worker's own
    // start is 3 while rank 0's frame says 4.
    hub.broadcast(&Tp4ControlFrame::LaneRequest {
        lane: 0,
        prompt_ids: vec![1, 2, 3, 9],
        images: Vec::new(),
        keep: true,
        sampler: None,
        start: Some(4),
    })
    .expect("the mismatched request");
    let error = worker
        .join()
        .expect("the worker finishes")
        .expect_err("the worker refuses a start it did not reach");
    assert!(
        error.contains("rank 0 starts the prompt at 4"),
        "the message names rank 0's start: {error}"
    );
    assert!(
        error.contains("this rank starts it at 3"),
        "the message names the worker's start: {error}"
    );
    assert!(error.contains("out of step"), "{error}");
}
