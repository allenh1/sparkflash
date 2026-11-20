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

//! The tick-trace gate: a fixed workload replayed through the scheduler's decisions with an
//! injected clock, compared with the committed trace.
//!
//! The trace covers what moved (the queue and lane choice, prompt order, the decode share,
//! chunk boundaries, the decoder skip, the decode plan and the row share). What the device
//! scheduler still decides on its own (memory, lane growth, paired chunks, the actual device
//! step) is not in it; see `sparkflash_runtime::tick`.

use sparkflash_runtime::tick::replay::{Config, Replay};

/// The committed decisions. Regenerate only with a stated reason and a before/after diff: this
/// file is the bit-neutrality gate for the scheduler's move.
fn expected() -> &'static str {
    include_str!("data/tick_trace.txt")
}

/// A prompt of `len` deterministic tokens.
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
fn the_replayed_scheduler_makes_the_committed_decisions() {
    let mut replay = Replay::new(Config::deepseek(3));
    let long = tokens(1, 30_000);
    // Three prompts at once: the short ones prefill first, the long one takes the decode share,
    // and the exact path serves the short prompt.
    replay.admit(1, long.clone(), 30_000);
    replay.admit(2, shares(&long, 2_000), 2_000);
    replay.admit(3, tokens(3, 80), 80);
    // The scripted costs are part of the workload, not wall time: a slower host reads the same
    // clock. The long chunks set the per-row cost, then the cheaper ticks cross the decode
    // share's threshold and come back.
    let costs = [
        450u64, 40, 40, 40, 40, 40, 40, 40, 40, 40, 450, 40, 40, 40, 40, 40, 40, 40, 40, 40, 450,
        40, 40, 40, 40, 40, 40, 40, 40, 40,
    ];
    for (tick, cost) in costs.into_iter().enumerate() {
        replay.tick(cost);
        match tick {
            // The short prompt ends early; the long one ends once its prefill and a few decode
            // steps have run, freeing a lane a later turn reuses most of.
            9 => replay.finish(3),
            17 => {
                replay.finish(1);
                replay.admit(4, shares(&long, 25_000), 25_000);
            }
            21 => replay.finish(2),
            27 => replay.finish(4),
            _ => {}
        }
    }
    let got = replay.trace().render();
    // `UPDATE_TICK_TRACE=1` rewrites the committed file; a reviewer reads the diff.
    if std::env::var_os("UPDATE_TICK_TRACE").is_some() {
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tick_trace.txt"),
            &got,
        )
        .expect("write the committed trace");
    }
    // The trace is compared as text: the decision sequence, lanes, rows and chunk boundaries,
    // with the injected clock that produced them.
    assert_eq!(
        got,
        expected(),
        "the scheduler's decisions moved; if that was intended, regenerate tests/data/tick_trace.txt"
    );
    // And it must actually decide something: a trace that records nothing passes vacuously.
    assert!(
        replay.trace().entries().len() > 40,
        "{} entries",
        replay.trace().entries().len()
    );
    assert!(got.contains("path=bulk"));
    assert!(got.contains("path=exact"));
    assert!(got.contains("hold 1"));
    assert!(got.contains("hold 0"));
    assert!(got.contains("plan=shared"));
    assert!(got.contains("plan=spec"));
}

/// Prompts whose rests are one chunk each end in one tick, a rows state each; one that shares
/// most of its rest with a picked prompt waits for it; and an idle tick runs a chunk a state of a
/// long prompt, the decoder's rows split off at its end.
#[test]
fn short_prompts_end_together_and_a_long_one_takes_a_chunk_a_state() {
    let mut replay = Replay::new(Config::deepseek(4));
    let lead = tokens(5, 1_500);
    replay.admit(1, lead.clone(), 1_500);
    replay.admit(2, tokens(6, 900), 900);
    replay.admit(3, tokens(7, 1_800), 1_800);
    // Its first 1,000 rows of 1,200 are the first prompt's.
    replay.admit(4, shares(&lead[..1_000], 1_200), 1_200);
    for cost in [100u64, 40, 40] {
        replay.tick(cost);
    }
    for id in 1..=4 {
        replay.finish(id);
    }
    replay.admit(5, tokens(8, 15_000), 15_000);
    for cost in [40u64, 300, 300] {
        replay.tick(cost);
    }
    let got = replay.trace().render();
    let want = "\
t=1 clock=100 place seq=1 lane=0 shared=0 start=0 target=1500 path=bulk
t=1 clock=100 place seq=2 lane=1 shared=0 start=0 target=900 path=bulk
t=1 clock=100 place seq=3 lane=2 shared=0 start=0 target=1800 path=bulk
t=1 clock=100 place seq=4 lane=3 shared=0 start=0 target=1200 path=bulk
t=1 clock=100 hold 0
t=1 clock=100 prefill seq=2 from=0 rows=900 to=900 target=900 path=bulk
t=1 clock=100 prefill seq=4 from=0 rows=1200 to=1200 target=1200 path=bulk
t=1 clock=100 prefill seq=3 from=0 rows=1800 to=1800 target=1800 path=bulk
";
    // In prefill order (fewest rows left first) the fourth prompt is picked before the first,
    // which shares 1,000 rows with it; the first then waits for it and the third takes the last
    // rows state.
    assert!(
        got.starts_with(want),
        "the first tick ends three prompts together and keeps the sharing one back:\n{got}"
    );
    // It runs alone the tick after: the replay models no copy, so it computes its rows.
    assert!(
        got.contains("t=2 clock=140 prefill seq=1 from=0 rows=1500 to=1500 target=1500 path=bulk"),
        "{got}"
    );
    // The long prompt, with nothing decoding: three chunks in a tick, then the rows before the
    // decoder's and the decoder's own.
    for line in [
        "t=4 clock=220 prefill seq=5 from=0 rows=4096 to=4096 target=15000 path=bulk",
        "t=4 clock=220 prefill seq=5 from=4096 rows=4096 to=8192 target=15000 path=bulk",
        "t=4 clock=220 prefill seq=5 from=8192 rows=4096 to=12288 target=15000 path=bulk",
        "t=5 clock=520 prefill seq=5 from=12288 rows=171 to=12459 target=15000 path=bulk",
        "t=5 clock=520 prefill seq=5 from=12459 rows=2541 to=15000 target=15000 path=bulk",
    ] {
        assert!(got.contains(line), "missing {line}:\n{got}");
    }
}
