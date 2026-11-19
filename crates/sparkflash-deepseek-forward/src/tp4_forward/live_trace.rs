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

//! The CPU mirror of `Batch::tick`'s decisions, and the gate that it equals `tick::replay`.
//!
//! `Batch` drives `DeviceLayer`, so its trace needs a GPU, a checkpoint and a cubin. Every
//! decision it makes before the device work, however, is a function of tokens, lanes and the
//! clock; this module models just those, in `Batch::tick`'s order:
//!
//! 1. place every waiting sequence ([tick::pick_lane], [tick::common_prefix],
//!    [tick::decoder_resume], [tick::bulk_from]);
//! 2. record the tick's hold, which `Batch` receives from rank 0 (here `Self::hold_prefill`
//!    answers it before the tick, as rank 0 does);
//! 3. run the first prefill in [tick::prefill_order], unless the hold keeps a long one back;
//! 4. run the decode plan ([tick::decode_plan]).
//!
//! It records through the same [TickTrace] as the device path. The test builds the same fixed
//! workload as `sparkflash-runtime/tests/tick_trace.rs` and requires the two traces to be
//! identical text, so a change to the shared decision functions on one side cannot drift from the
//! other unnoticed. It does **not** exercise the device-only decisions of `Batch` (lane memory,
//! adoption, paired chunks, preemption or the actual step): those need the device gate.

use sparkflash_model::speculation::{DepthPolicy, StepCost};
use sparkflash_runtime::share_rows;
use sparkflash_runtime::tick::replay::{Config, Replay};
use sparkflash_runtime::tick::{
    Clock, Decision, DecodePlan, InjectedClock, LaneCandidate, SHARE_MIN_ROWS, TickRows, TickTrace,
    band, bulk_budget, bulk_from, bulk_rows, common_prefix, decode_plan, decoder_resume,
    hold_prefill, pick_lane, prefill_order, reusable_shared, runs_decoder, skip_end_after,
    tick_rows, whole_chunk_fits,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Waiting,
    Prefill { target: usize, bulk: bool },
    Decode,
}

struct Seq {
    id: u64,
    tokens: Vec<u32>,
    prompt_len: usize,
    lane: Option<usize>,
    pos: usize,
    phase: Phase,
    admitted_ms: u64,
    done: bool,
}

impl Seq {
    fn emitted(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    fn target(&self) -> usize {
        if self.emitted() == 0 {
            self.prompt_len
        } else {
            self.tokens.len() - 1
        }
    }
}

#[derive(Default)]
struct Lane {
    resident: Vec<u32>,
    spans: Vec<sparkflash_model::vision::ImageSpan>,
    seq: Option<u64>,
    used: u64,
    skip_end: usize,
}

/// The CPU mirror of `Batch::tick`: token and lane state, the shared decision functions and an
/// injected clock, with no device.
struct LiveBatch {
    cfg: Config,
    clock: InjectedClock,
    trace: TickTrace,
    ticks: u64,
    lanes: Vec<Lane>,
    seqs: Vec<Seq>,
    pending: std::collections::VecDeque<(u64, u64, Vec<u32>)>,
    chunk_ms: f64,
    row_ms: f64,
    since_chunk_ms: f64,
}

impl LiveBatch {
    fn new(cfg: Config) -> Self {
        Self {
            lanes: (0..cfg.lanes).map(|_| Lane::default()).collect(),
            cfg,
            clock: InjectedClock::new(0),
            trace: TickTrace::new(),
            ticks: 0,
            seqs: Vec::new(),
            pending: std::collections::VecDeque::new(),
            chunk_ms: 0.0,
            row_ms: 0.0,
            since_chunk_ms: 0.0,
        }
    }

    fn admit(&mut self, id: u64, tokens: Vec<u32>, prompt_len: usize) {
        self.pending.push_back((id, prompt_len as u64, tokens));
    }

    fn finish(&mut self, id: u64) {
        if let Some(index) = self.seqs.iter().position(|seq| seq.id == id) {
            self.seqs[index].done = true;
            if let Some(lane) = self.seqs[index].lane.take() {
                self.lanes[lane].seq = None;
                self.lanes[lane].used = self.ticks;
            }
        }
    }

    fn trace(&self) -> &TickTrace {
        &self.trace
    }

    fn tick_rows_config(&self) -> TickRows {
        TickRows {
            busy_chunk: self.cfg.busy_chunk,
            busy_exact_steps: self.cfg.busy_exact_steps,
            exact_tick_rows: self.cfg.exact_tick_rows,
            rows_max: self.cfg.rows_max,
            chunk_min: self.cfg.chunk_min,
        }
    }

    fn decoding(&self) -> bool {
        self.seqs
            .iter()
            .any(|seq| !seq.done && seq.phase == Phase::Decode)
    }

    fn prefill_order(&self) -> Vec<(usize, usize)> {
        prefill_order(
            self.seqs
                .iter()
                .enumerate()
                .filter_map(|(index, seq)| match seq.phase {
                    Phase::Prefill { target, .. } if !seq.done => {
                        Some((target.saturating_sub(seq.pos), index))
                    }
                    _ => None,
                }),
        )
    }

    /// Rank 0's answer for the tick's hold, computed before the tick from the scheduler's state
    /// exactly as `Batch::hold_prefill` is.
    fn hold_prefill(&self) -> bool {
        let order = self.prefill_order();
        hold_prefill(
            &self.cfg.share,
            &order,
            |index| (self.clock.now_ms() - self.seqs[index].admitted_ms) as f64,
            self.decoding(),
            self.row_ms,
            self.chunk_ms,
            self.since_chunk_ms,
        )
    }

    fn record(&mut self, decision: Decision) {
        let clock = self.clock.now_ms();
        self.trace.record(self.ticks, clock, decision);
    }

    fn place(&mut self) {
        loop {
            let Some(index) = self
                .seqs
                .iter()
                .position(|seq| !seq.done && seq.phase == Phase::Waiting)
            else {
                return;
            };
            let target = self.seqs[index].target();
            let wanted = &self.seqs[index].tokens[..target];
            let free = (0..self.lanes.len()).filter(|&lane| self.lanes[lane].seq.is_none());
            let Some(candidate) = pick_lane(free.map(|lane| {
                let held = &self.lanes[lane];
                let shared = if self.cfg.reuse {
                    common_prefix(&held.resident, &held.spans, wanted, &[])
                } else {
                    0
                };
                LaneCandidate {
                    lane,
                    shared: reusable_shared(
                        shared,
                        held.resident.len(),
                        self.cfg.prefix_reuse_min_rows,
                    ),
                    used: held.used,
                }
            })) else {
                return;
            };
            let (lane, shared) = (candidate.lane, candidate.shared);
            let cap = if self.seqs[index].emitted() == 0 {
                target - 1
            } else {
                target
            };
            let mut start = shared.min(cap);
            if self.cfg.decoder_skip {
                start = decoder_resume(
                    self.lanes[lane].skip_end,
                    start,
                    target,
                    self.cfg.decoder_warm,
                );
            }
            let bulk = bulk_from(
                target,
                start,
                self.cfg.bulk_cap.is_some(),
                self.cfg.exchange,
                false,
                self.cfg.chunk_min,
            );
            let seq = &mut self.seqs[index];
            let id = seq.id;
            seq.lane = Some(lane);
            seq.pos = start;
            seq.phase = if start == target {
                Phase::Decode
            } else {
                Phase::Prefill { target, bulk }
            };
            let lane_state = &mut self.lanes[lane];
            lane_state.seq = Some(id);
            lane_state.used = self.ticks;
            lane_state.resident.truncate(start);
            lane_state.spans.retain(|span| span.start < start);
            self.record(Decision::Place {
                seq: id,
                lane,
                shared,
                start,
                target,
                bulk,
            });
        }
    }

    fn prefill(&mut self, index: usize) -> Option<usize> {
        let Phase::Prefill { target, bulk } = self.seqs[index].phase else {
            return None;
        };
        let from = self.seqs[index].pos;
        let left = target - from;
        let lane = self.seqs[index].lane?;
        let id = self.seqs[index].id;
        let mut ran = 0;
        if bulk {
            let cap = self.cfg.bulk_cap;
            let (budget, most) = bulk_budget(&self.tick_rows_config(), cap, self.decoding());
            let fits = whole_chunk_fits(
                left,
                budget,
                self.cfg.decoder_skip,
                self.cfg.decoder_warm,
                self.cfg.rows_max,
            );
            let rows = bulk_rows(
                left,
                if fits { most } else { budget },
                most,
                None,
                self.cfg.rows_max,
            )
            .min(left);
            ran = rows;
            self.record(Decision::Prefill {
                seq: id,
                from,
                rows,
                target,
                bulk: true,
            });
        } else {
            let steps = tick_rows(
                &self.tick_rows_config(),
                self.cfg.bulk_cap,
                false,
                self.decoding(),
            )
            .div_ceil(self.cfg.rows_max);
            let mut p = from;
            for _ in 0..steps {
                if p == target {
                    break;
                }
                let n = self.cfg.rows_max.min(target - p);
                self.record(Decision::Prefill {
                    seq: id,
                    from: p,
                    rows: n,
                    target,
                    bulk: false,
                });
                self.lanes[lane].skip_end = skip_end_after(
                    self.lanes[lane].skip_end,
                    p,
                    n,
                    !self.cfg.decoder_skip || runs_decoder(p, n, target, self.cfg.decoder_warm),
                );
                p += n;
                ran += n;
            }
            self.seqs[index].pos = p;
        }
        if bulk {
            let end = from + ran;
            self.lanes[lane].skip_end = skip_end_after(
                self.lanes[lane].skip_end,
                from,
                ran,
                !self.cfg.decoder_skip || runs_decoder(from, ran, target, self.cfg.decoder_warm),
            );
            self.seqs[index].pos = end;
            self.lanes[lane]
                .resident
                .extend_from_slice(&self.seqs[index].tokens[from..end]);
        }
        if self.seqs[index].pos >= target {
            self.seqs[index].phase = Phase::Decode;
        }
        Some(ran)
    }

    fn decode(&mut self) -> usize {
        let indices: Vec<usize> = self
            .seqs
            .iter()
            .enumerate()
            .filter(|(_, seq)| !seq.done && seq.phase == Phase::Decode)
            .map(|(index, _)| index)
            .collect();
        let plan = decode_plan(
            indices.len(),
            false,
            self.cfg.batch_spec,
            self.cfg.exchange,
            self.cfg.rows_max,
        );
        match plan {
            DecodePlan::None => 0,
            DecodePlan::Spec => {
                let policy = DepthPolicy::adaptive(
                    self.cfg.k,
                    StepCost {
                        step_ms: 27.0,
                        row_ms: 4.0,
                        prior_tokens_per_ms: 0.06,
                    },
                );
                let probs: Vec<f32> = (0..self.cfg.k)
                    .map(|i| [0.9f32, 0.7, 0.5, 0.3, 0.1].get(i).copied().unwrap_or(0.1))
                    .collect();
                let depth = policy.depth(&probs);
                self.record(Decision::Decode {
                    seqs: 1,
                    rows: 1 + depth,
                    plan: "spec",
                });
                1 + depth
            }
            DecodePlan::Plain { groups } => {
                self.record(Decision::Decode {
                    seqs: indices.len(),
                    rows: indices.len(),
                    plan: "plain",
                });
                let _ = groups;
                indices.len()
            }
            DecodePlan::Shared => {
                let policy = DepthPolicy::adaptive(
                    self.cfg.k,
                    StepCost {
                        step_ms: 27.0,
                        row_ms: 4.0,
                        prior_tokens_per_ms: 0.06,
                    },
                );
                let chains: Vec<Vec<f64>> = indices
                    .iter()
                    .enumerate()
                    .map(|(at, _)| {
                        let probs: Vec<f32> = (0..self.cfg.k)
                            .map(|i| {
                                [0.95f32, 0.8, 0.6, 0.4, 0.2]
                                    .get((i + at) % 5)
                                    .copied()
                                    .unwrap_or(0.2)
                            })
                            .collect();
                        policy.chain(&probs)
                    })
                    .collect();
                let budget = self.cfg.rows_max - indices.len();
                let depths =
                    share_rows(&chains, budget, self.cfg.share_floor, &band(self.cfg.model));
                let rows = indices.len() + depths.iter().sum::<usize>();
                self.record(Decision::Decode {
                    seqs: indices.len(),
                    rows,
                    plan: "shared",
                });
                rows
            }
        }
    }

    /// One tick whose whole cost is `cost_ms`, with the hold rank 0 answered before it. The
    /// injected clock advances by the scripted cost, not a measured wall time.
    fn tick(&mut self, hold: bool, cost_ms: u64) {
        self.ticks += 1;
        self.clock.advance(cost_ms);
        while let Some((id, prompt_len, tokens)) = self.pending.pop_front() {
            self.seqs.push(Seq {
                id,
                prompt_len: prompt_len as usize,
                tokens,
                lane: None,
                pos: 0,
                phase: Phase::Waiting,
                admitted_ms: self.clock.now_ms(),
                done: false,
            });
        }
        self.place();
        let decoding = self.decoding();
        self.record(Decision::Hold { hold });
        let order = self.prefill_order();
        let mut long_chunk = None;
        if let Some(&(_, index)) = order.first() {
            let left = match self.seqs[index].phase {
                Phase::Prefill { target, .. } => target.saturating_sub(self.seqs[index].pos),
                Phase::Waiting | Phase::Decode => 0,
            };
            if !(hold && decoding && left > SHARE_MIN_ROWS) {
                long_chunk = self.prefill(index);
            }
        }
        self.decode();
        if let Some(ran) = long_chunk {
            self.chunk_ms = cost_ms as f64;
            self.row_ms = cost_ms as f64 / ran as f64;
            self.since_chunk_ms = 0.0;
        } else {
            self.since_chunk_ms += cost_ms as f64;
        }
        self.seqs.retain(|seq| !seq.done);
    }
}

/// A prompt of `len` deterministic tokens, as `tick_trace.rs` writes it.
fn tokens(seed: u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| 1 + (seed.wrapping_mul(7919).wrapping_add(i as u32 * 31)) % 90_000)
        .collect()
}

/// A prompt that begins exactly like `head` and continues on its own.
fn shares(head: &[u32], len: usize) -> Vec<u32> {
    let mut tokens = head[..len.min(head.len())].to_vec();
    tokens.extend((tokens.len()..len).map(|i| 1 + (i as u32 * 131) % 90_000));
    tokens
}

#[test]
fn the_live_tick_makes_the_replay_s_decisions() {
    let config = Config::deepseek(3);
    let mut live = LiveBatch::new(config);
    let mut replay = Replay::new(config);
    let long = tokens(1, 30_000);
    // The same fixed workload as the committed replay trace: three prompts at once, the short
    // ones first, the long one taking the decode share, then a turn that shares most of it.
    live.admit(1, long.clone(), 30_000);
    live.admit(2, shares(&long, 2_000), 2_000);
    live.admit(3, tokens(3, 80), 80);
    replay.admit(1, long.clone(), 30_000);
    replay.admit(2, shares(&long, 2_000), 2_000);
    replay.admit(3, tokens(3, 80), 80);
    let costs = [
        450u64, 40, 40, 40, 40, 40, 40, 40, 40, 40, 450, 40, 40, 40, 40, 40, 40, 40, 40, 40, 450,
        40, 40, 40, 40, 40, 40, 40, 40, 40,
    ];
    for (tick, cost) in costs.into_iter().enumerate() {
        // Rank 0 answers the hold before the tick, as `serve_ticks` asks `Batch::hold_prefill`.
        let hold = live.hold_prefill();
        live.tick(hold, cost);
        replay.tick(cost);
        match tick {
            9 => {
                live.finish(3);
                replay.finish(3);
            }
            17 => {
                live.finish(1);
                replay.finish(1);
                live.admit(4, shares(&long, 25_000), 25_000);
                replay.admit(4, shares(&long, 25_000), 25_000);
            }
            21 => {
                live.finish(2);
                replay.finish(2);
            }
            27 => {
                live.finish(4);
                replay.finish(4);
            }
            _ => {}
        }
    }
    let live_trace = live.trace().render();
    let replay_trace = replay.trace().render();
    assert!(
        live.trace().entries().len() > 40,
        "{} entries",
        live.trace().entries().len()
    );
    assert_eq!(
        live_trace, replay_trace,
        "the CPU mirror of Batch::tick and tick::replay disagree; the live path's decisions drifted"
    );
}
