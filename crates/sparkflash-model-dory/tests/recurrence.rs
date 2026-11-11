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

//! The latent-recurrence group against the single-process device arm, bit for bit.
//!
//! Every rank runs in its own thread with its own CUDA context, joined by loopback links that
//! stage each transfer through the host. The group's logits and tokens must be the single
//! process's exactly: the same kernels run on the same rows, only on different contexts, and on
//! a lane of their own whatever the other lanes are running. A drafting group starts rows on
//! rank 1's guesses and drops the wrong ones, and must be the single process all the same.

use std::path::Path;
use std::sync::Barrier;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model::backend::{ModelBackend, Sampling, Speculation, StepOutput};
use sparkflash_model_dory::config::DoryConfig;
use sparkflash_model_dory::fixture;
use sparkflash_model_dory::{
    DoryDeviceBackend, DoryRecurrenceBackend, DoryRecurrenceStage, DoryStage, RecurrenceLink,
    RecurrenceLinks, recurrence_draft_lanes,
};

/// One rank's end of an in-process ring: a channel to and from every other rank.
struct Loopback {
    to: Vec<Option<Sender<Vec<u8>>>>,
    from: Vec<Option<Receiver<Vec<u8>>>>,
}

impl RecurrenceLink for Loopback {
    fn send(
        &mut self,
        cuda: &Cuda,
        peer: usize,
        buffer: Deviceptr,
        bytes: usize,
    ) -> Result<(), String> {
        let mut host = vec![0u8; bytes];
        cuda.make_current()?;
        // Stream-ordered: the copy waits for every launch queued before it.
        cuda.copy_dtoh(&mut host, &buffer)?;
        self.to[peer]
            .as_ref()
            .ok_or_else(|| format!("no loopback channel to rank {peer}"))?
            .send(host)
            .map_err(|_| format!("rank {peer} hung up"))
    }

    fn recv(
        &mut self,
        cuda: &Cuda,
        peer: usize,
        buffer: Deviceptr,
        bytes: usize,
    ) -> Result<(), String> {
        let host = self.from[peer]
            .as_ref()
            .ok_or_else(|| format!("no loopback channel from rank {peer}"))?
            .recv()
            .map_err(|_| format!("rank {peer} hung up"))?;
        if host.len() != bytes {
            return Err(format!(
                "rank {peer} sent {} bytes where {bytes} were expected",
                host.len()
            ));
        }
        cuda.make_current()?;
        cuda.copy_htod(&buffer, &host)?;
        // A copy from pageable memory can return before its DMA lands, and rank 0's answers link
        // is read on a stream of its own that the default stream does not order: wait for it.
        cuda.synchronize()
    }
}

/// `world` loopback ends, one per rank, every pair joined both ways.
fn loopback_ring(world: usize) -> Vec<Loopback> {
    let mut links: Vec<Loopback> = (0..world)
        .map(|_| Loopback {
            to: (0..world).map(|_| None).collect(),
            from: (0..world).map(|_| None).collect(),
        })
        .collect();
    for from in 0..world {
        for to in 0..world {
            if from == to {
                continue;
            }
            let (sender, receiver) = channel();
            links[from].to[to] = Some(sender);
            links[to].from[from] = Some(receiver);
        }
    }
    links
}

/// The fixture's config document with `loops` recurrent loops.
fn fixture_json(loops: usize) -> String {
    let mut value: serde_json::Value =
        serde_json::from_str(&DoryConfig::tiny_ring_json()).expect("the fixture is JSON");
    value["n_recurrent_loops"] = serde_json::json!(loops);
    value.to_string()
}

/// A prompt whose sliding rings wrap (two prefill chunks on one process, five on the group), then
/// `decode` greedy steps.
fn run(backend: &dyn ModelBackend, sampling: Sampling, decode: usize) -> Vec<StepOutput> {
    let prompt: Vec<u32> = (0..4200).map(|index| (index % 63 + 1) as u32).collect();
    let mut session = backend.open_session().expect("a session");
    session.set_sampling(sampling);
    let mut steps = vec![session.prefill(&prompt).expect("the prompt runs")];
    for _ in 0..decode {
        let token = steps
            .last()
            .and_then(|step| step.next_token)
            .expect("every step chooses a token");
        steps.push(session.decode(token).expect("the decode step runs"));
    }
    // A second request on the same group starts a new sequence from position 0. At 1,029 rows
    // the group runs three chunks (384, 384, 261), so it crosses chunk boundaries; the first
    // session goes first, as the host driver drops it, so a one-lane group has its lane free.
    drop(session);
    let mut again = backend.open_session().expect("a second session");
    again.set_sampling(sampling);
    steps.push(
        again
            .prefill(&prompt[..1029])
            .expect("the second prompt runs"),
    );
    steps
}

/// Opens a `world`-rank group of `lanes` lanes, drafting at up to `draft_lanes` held lanes, on
/// threads, runs `body` against rank 0, then closes the group and joins the workers.
fn with_group<T>(
    directory: &Path,
    world: usize,
    lanes: usize,
    draft_lanes: usize,
    body: impl FnOnce(&DoryRecurrenceBackend) -> T,
) -> T {
    let mut rings = loopback_ring(world).into_iter();
    let mut answers = loopback_ring(world).into_iter();
    let mut drafts: Box<dyn Iterator<Item = Option<Loopback>>> = if draft_lanes > 0 {
        Box::new(loopback_ring(world).into_iter().map(Some))
    } else {
        Box::new(std::iter::repeat_with(|| None))
    };
    let (head, head_answers, head_drafts) = (
        rings.next().expect("rank 0's ring"),
        answers.next().expect("rank 0's answers"),
        drafts.next().expect("rank 0's drafts"),
    );
    let links = |ring: Loopback, answers: Loopback, drafts: Option<Loopback>| RecurrenceLinks {
        ring: Box::new(ring),
        answers: Box::new(answers),
        drafts: drafts.map(|link| Box::new(link) as Box<dyn RecurrenceLink>),
    };
    let workers: Vec<_> = rings
        .zip(answers)
        .zip(drafts)
        .enumerate()
        .map(|(index, ((ring, answers), drafts))| {
            let directory = directory.to_path_buf();
            thread::spawn(move || {
                let rank = index + 1;
                let stage = DoryRecurrenceStage::open(
                    &directory,
                    5000,
                    rank,
                    world,
                    lanes,
                    draft_lanes,
                    |_, _| Ok(links(ring, answers, drafts)),
                )
                .expect("the worker stage opens");
                stage.serve().expect("the worker serves until shutdown");
            })
        })
        .collect();
    let backend =
        DoryRecurrenceBackend::open(directory, 5000, world, lanes, draft_lanes, |_, _| {
            Ok(links(head, head_answers, head_drafts))
        })
        .expect("rank 0 opens");
    let result = body(&backend);
    drop(backend);
    for worker in workers {
        worker.join().expect("the worker exits cleanly");
    }
    result
}

fn group(
    directory: &Path,
    world: usize,
    draft_lanes: usize,
    sampling: Sampling,
    decode: usize,
) -> Vec<StepOutput> {
    with_group(directory, world, 1, draft_lanes, |backend| {
        run(backend, sampling, decode)
    })
}

/// The drafting settings a `world`-rank group is checked under: plain, and drafting where it
/// can.
fn draftings(world: usize) -> &'static [usize] {
    if world >= 3 { &[0, 1] } else { &[0] }
}

/// A prompt of `len` tokens that differs from the other requests' from its first token.
fn prompt(len: usize, salt: u32) -> Vec<u32> {
    (0..len)
        .map(|index| (index as u32 * 7 + salt) % 61 + 1)
        .collect()
}

/// One request: the prompt, then `decode` steps fed back.
fn request(
    backend: &dyn ModelBackend,
    tokens: &[u32],
    sampling: Sampling,
    decode: usize,
) -> Vec<StepOutput> {
    let mut session = backend.open_session().expect("a session");
    session.set_sampling(sampling);
    let mut steps = vec![session.prefill(tokens).expect("the prompt runs")];
    for _ in 0..decode {
        let token = steps
            .last()
            .and_then(|step| step.next_token)
            .expect("every step chooses a token");
        steps.push(session.decode(token).expect("the decode step runs"));
    }
    steps
}

fn assert_same(single: &[StepOutput], grouped: &[StepOutput], what: &str) {
    assert_eq!(single.len(), grouped.len());
    for (index, (one, many)) in single.iter().zip(grouped).enumerate() {
        assert_eq!(
            one.next_token, many.next_token,
            "{what}: step {index} token"
        );
        assert_eq!(
            one.logits.len(),
            many.logits.len(),
            "{what}: step {index} width"
        );
        let differing = one
            .logits
            .iter()
            .zip(&many.logits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "{what}: step {index} has {differing} logits off by a bit"
        );
    }
}

fn device_or_skip(what: &str) -> bool {
    if DoryDeviceBackend::available() {
        return true;
    }
    eprintln!(
        "skipping {what}: this build has no Dory/Qwen cubin (rebuild with nvcc, or set \
         SPARKFLASH_CUDA_ARCH to this GPU's architecture)"
    );
    false
}

/// The single process on the fixture in `directory`, or none when this host's CUDA cannot run
/// the arm, said under `what`.
fn single_or_skip(directory: &Path, what: &str) -> Option<DoryDeviceBackend> {
    match DoryDeviceBackend::open(directory, 5000) {
        Ok(backend) => Some(backend),
        Err(error) if error.contains("CUDA:") || error.contains("cuModuleLoadData failed: 209") => {
            eprintln!("skipping {what}: {error}");
            None
        }
        Err(error) => panic!("the device arm refused the fixture: {error}"),
    }
}

#[test]
#[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
fn a_recurrence_group_is_the_single_process_bit_for_bit() {
    if !device_or_skip("the Dory recurrence parity") {
        return;
    }
    let seeded = Sampling {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 20,
        repetition_penalty: 1.05,
        seed: 1234,
        ..Sampling::default()
    };
    // (loops, ranks): one recurrence a rank at three and four loops, and two loops a rank.
    for (loops, world) in [(3, 3), (4, 4), (4, 2)] {
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        fixture::write(directory.path(), &fixture_json(loops)).expect("the fixture writes");
        let Some(single) = single_or_skip(directory.path(), "the Dory recurrence parity") else {
            return;
        };
        for (name, sampling) in [("greedy", Sampling::default()), ("seeded", seeded)] {
            let reference = run(&single, sampling, 6);
            for &draft_lanes in draftings(world) {
                let grouped = group(directory.path(), world, draft_lanes, sampling, 6);
                assert_same(
                    &reference,
                    &grouped,
                    &format!("{loops} loops on {world} ranks, {name}, draft lanes {draft_lanes}"),
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
fn concurrent_lanes_are_each_the_single_process_bit_for_bit() {
    if !device_or_skip("the Dory recurrence lanes") {
        return;
    }
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write(directory.path(), &fixture_json(4)).expect("the fixture writes");
    let Some(single) = single_or_skip(directory.path(), "the Dory recurrence lanes") else {
        return;
    };
    let seeded = Sampling {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 20,
        seed: 99,
        ..Sampling::default()
    };
    // A long prompt over several chunks, a short one, and one a single chunk wide, each with its
    // own sampling, so the lanes interleave prompt chunks with each other's decode steps.
    let requests = [
        (prompt(4200, 3), Sampling::default()),
        (prompt(1500, 11), seeded),
        (prompt(37, 29), Sampling::default()),
    ];
    let reference: Vec<_> = requests
        .iter()
        .map(|(tokens, sampling)| request(&single, tokens, *sampling, 6))
        .collect();
    drop(single);
    let lanes = requests.len();
    // Drafting at one held lane drafts only while the others have not yet come or have ended;
    // at every lane, every lane drafts at once.
    for draft_lanes in [0, 1, lanes] {
        let grouped = lanes_group(directory.path(), &requests, draft_lanes, 6);
        for (index, (one, many)) in reference.iter().zip(&grouped).enumerate() {
            assert_same(
                one,
                many,
                &format!("request {index} on its lane, draft lanes {draft_lanes}"),
            );
        }
    }
}

#[test]
#[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
fn more_drafting_lanes_than_a_pass_are_each_the_single_process_bit_for_bit() {
    if !device_or_skip("the Dory drafting passes") {
        return;
    }
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write(directory.path(), &fixture_json(4)).expect("the fixture writes");
    let Some(single) = single_or_skip(directory.path(), "the Dory drafting passes") else {
        return;
    };
    // Ten lanes drafting at once take two passes; the rows wrong guesses drop rejoin whichever
    // pass has room, and the passes gather again as they come. Half the lanes sample.
    let requests: Vec<_> = (0..10u32)
        .map(|index| {
            let sampling = if index % 2 == 0 {
                Sampling::default()
            } else {
                Sampling {
                    temperature: 0.8,
                    top_k: 20,
                    seed: u64::from(index),
                    ..Sampling::default()
                }
            };
            (prompt(40 + 23 * index as usize, 5 + index), sampling)
        })
        .collect();
    let reference: Vec<_> = requests
        .iter()
        .map(|(tokens, sampling)| request(&single, tokens, *sampling, 24))
        .collect();
    drop(single);
    let grouped = lanes_group(directory.path(), &requests, requests.len(), 24);
    for (index, (one, many)) in reference.iter().zip(&grouped).enumerate() {
        assert_same(one, many, &format!("request {index} of ten drafting lanes"));
    }
}

/// The requests at once, a thread and a lane each, `decode` steps each, on a four-rank group.
fn lanes_group(
    directory: &Path,
    requests: &[(Vec<u32>, Sampling)],
    draft_lanes: usize,
    decode: usize,
) -> Vec<Vec<StepOutput>> {
    let lanes = requests.len();
    with_group(directory, 4, lanes, draft_lanes, |backend| {
        // Every lane is held: one more sequence is refused, not queued behind them.
        let mut held: Vec<_> = (0..lanes)
            .map(|_| {
                let mut session = backend.open_session().expect("a session");
                session.prefill(&[1, 2, 3]).expect("a lane is free");
                session
            })
            .collect();
        let mut extra = backend.open_session().expect("a session");
        let error = extra.prefill(&[1, 2, 3]).expect_err("every lane is held");
        assert!(error.to_string().contains("--concurrency"), "{error}");
        held.pop();
        extra
            .prefill(&[1, 2, 3])
            .expect("a dropped session frees its lane");
        drop((held, extra));
        let start = Barrier::new(lanes);
        thread::scope(|scope| {
            let handles: Vec<_> = requests
                .iter()
                .map(|(tokens, sampling)| {
                    let start = &start;
                    scope.spawn(move || {
                        start.wait();
                        request(backend, tokens, *sampling, decode)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("the request thread"))
                .collect::<Vec<_>>()
        })
    })
}

/// A greedy turn: the prompt, then decode steps fed back until a step ends the turn or `limit`
/// steps have run.
fn turn(backend: &dyn ModelBackend, tokens: &[u32], limit: usize) -> Vec<StepOutput> {
    let mut session = backend.open_session().expect("a session");
    let mut steps = vec![session.prefill(tokens).expect("the prompt runs")];
    while steps.len() <= limit {
        let Some(token) = steps.last().and_then(|step| step.next_token) else {
            break;
        };
        steps.push(session.decode(token).expect("the decode step runs"));
    }
    steps
}

/// The step of `steps` that first chooses a token no earlier step chose, past the prompt's.
fn first_new_token(steps: &[StepOutput]) -> (usize, u32) {
    let tokens: Vec<u32> = steps
        .iter()
        .map(|step| step.next_token.expect("no end-of-turn id is set"))
        .collect();
    (1..tokens.len())
        .find(|index| !tokens[..*index].contains(&tokens[*index]))
        .map(|index| (index, tokens[index]))
        .expect("a decode step chooses a token the turn has not seen")
}

#[test]
#[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
fn an_end_of_turn_id_ends_the_turn_on_one_process_and_on_the_group() {
    if !device_or_skip("the Dory end-of-turn check") {
        return;
    }
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write(directory.path(), &fixture_json(4)).expect("the fixture writes");
    let tokens = prompt(40, 5);
    let Some(single) = single_or_skip(directory.path(), "the Dory end-of-turn check") else {
        return;
    };
    let reference = turn(&single, &tokens, 8);
    drop(single);
    let (end, eos) = first_new_token(&reference);
    // The same checkpoint with that token as its end-of-turn id: every step up to it is the
    // same, and the step that chooses it hands back no token, its logits unchanged.
    let mut value: serde_json::Value =
        serde_json::from_str(&fixture_json(4)).expect("the fixture is JSON");
    value["eos_token_id"] = serde_json::json!(eos);
    fixture::write(directory.path(), &value.to_string()).expect("the fixture writes");
    let mut expected = reference[..=end].to_vec();
    expected[end].next_token = None;
    let single = DoryDeviceBackend::open(directory.path(), 5000).expect("the device arm opens");
    assert_same(&expected, &turn(&single, &tokens, 8), "one process");
    drop(single);
    for draft_lanes in [0, 1] {
        let grouped = with_group(directory.path(), 4, 1, draft_lanes, |backend| {
            turn(backend, &tokens, 8)
        });
        assert_same(
            &expected,
            &grouped,
            &format!("the group, draft lanes {draft_lanes}"),
        );
    }
}

/// A turn fed `fed(step, chosen)` after each step, with the sequence's speculation counters.
fn scripted(
    backend: &dyn ModelBackend,
    tokens: &[u32],
    sampling: Sampling,
    steps: usize,
    fed: impl Fn(usize, u32) -> u32,
) -> (Vec<StepOutput>, Speculation) {
    let mut session = backend.open_session().expect("a session");
    session.set_sampling(sampling);
    let mut outputs = vec![session.prefill(tokens).expect("the prompt runs")];
    for step in 0..steps {
        let chosen = outputs
            .last()
            .and_then(|output| output.next_token)
            .expect("every step chooses a token");
        outputs.push(
            session
                .decode(fed(step, chosen))
                .expect("the decode step runs"),
        );
    }
    let speculation = session.speculation();
    (outputs, speculation)
}

#[test]
#[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
fn a_drafting_group_drops_the_rows_a_caller_overrides_and_counts_its_guesses() {
    if !device_or_skip("the Dory drafting rollback") {
        return;
    }
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write(directory.path(), &fixture_json(4)).expect("the fixture writes");
    let Some(single) = single_or_skip(directory.path(), "the Dory drafting rollback") else {
        return;
    };
    let seeded = Sampling {
        temperature: 0.9,
        top_k: 40,
        seed: 7,
        ..Sampling::default()
    };
    let tokens = prompt(300, 17);
    // Every third step feeds a token the group did not choose: the rows started on the guess
    // after the choice are dropped on every rank, the last rank's sampler with them.
    let fed = |step: usize, chosen: u32| {
        if step % 3 == 2 {
            (chosen + 5) % 61 + 1
        } else {
            chosen
        }
    };
    for (name, sampling) in [("greedy", Sampling::default()), ("seeded", seeded)] {
        let (reference, none) = scripted(&single, &tokens, sampling, 16, fed);
        assert_eq!(none, Speculation::default(), "one process drafts nothing");
        let (grouped, speculation) = with_group(directory.path(), 4, 1, 1, |backend| {
            scripted(backend, &tokens, sampling, 16, fed)
        });
        assert_same(&reference, &grouped, &format!("an overridden turn, {name}"));
        assert!(
            speculation.drafts_proposed > 0,
            "{name}: no row was started on a guess: {speculation:?}"
        );
        assert!(
            speculation.drafts_accepted <= speculation.drafts_proposed
                && speculation.verify_steps == speculation.drafts_proposed,
            "{name}: {speculation:?}"
        );
        // A group that does not draft starts no row on a guess.
        let (_, plain) = with_group(directory.path(), 4, 1, 0, |backend| {
            scripted(backend, &tokens, sampling, 4, fed)
        });
        assert_eq!(plain, Speculation::default(), "{name}: the plain group");
    }
}

#[test]
fn drafting_needs_a_rank_between_the_first_and_the_last() {
    let config = DoryConfig::from_json(&fixture_json(4)).expect("the fixture is valid");
    let drafter = DoryStage::of_rank(&config, 1, 4)
        .expect("rank 1")
        .drafting()
        .expect("rank 1 of 4 drafts");
    assert!(drafter.draft && !drafter.prefix && !drafter.suffix);
    assert!(!DoryStage::of_rank(&config, 1, 4).expect("rank 1").draft);
    let error = DoryStage::of_rank(&config, 0, 4)
        .expect("rank 0")
        .drafting()
        .expect_err("rank 0 receives no latent");
    assert!(error.contains("latent"), "{error}");
    let error = DoryStage::of_rank(&config, 1, 2)
        .expect("rank 1 of 2")
        .drafting()
        .expect_err("the last rank's suffix would share the draft's caches");
    assert!(error.contains("suffix"), "{error}");
    // A two-rank group is refused before any weight loads or link opens.
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    let error = DoryRecurrenceBackend::open(directory.path(), 5000, 2, 1, 1, |_, _| {
        panic!("a refused group opens no link")
    })
    .err()
    .expect("a two-rank group cannot draft");
    assert!(error.contains("at least 3 ranks"), "{error}");
    let error = DoryRecurrenceStage::open(directory.path(), 5000, 1, 2, 1, 1, |_, _| {
        panic!("a refused group opens no link")
    })
    .err()
    .expect("a two-rank group cannot draft");
    assert!(error.contains("at least 3 ranks"), "{error}");
}

#[test]
fn the_released_two_loop_checkpoint_serves_a_two_rank_group() {
    const RELEASED: &str = include_str!("../config/nvidia_nemotron_labs_dory1_0_4b_bf16.hf.json");
    let config = DoryConfig::from_json(RELEASED).expect("the released document is valid");
    assert_eq!(config.n_recurrent_loops, 2);
    assert!(DoryStage::of_rank(&config, 0, 1).is_ok());
    assert!(DoryStage::of_rank(&config, 0, 2).is_ok());
    assert!(DoryStage::of_rank(&config, 1, 2).is_ok());
    for world in [3, 4] {
        let error = DoryStage::of_rank(&config, 0, world)
            .expect_err("two loops do not split a wider group");
        assert!(error.contains("evenly"), "{error}");
    }
    // The released two-loop checkpoint cannot hold the three ranks loop-1 drafting needs.
    assert_eq!(recurrence_draft_lanes(2, None), Ok(0));
    assert_eq!(recurrence_draft_lanes(4, None), Ok(16));
}

#[test]
fn a_stage_holds_only_its_own_loops() {
    let config = DoryConfig::from_json(&fixture_json(4)).expect("the fixture is valid");
    let first = DoryStage::of_rank(&config, 0, 4).expect("rank 0");
    assert_eq!(
        (first.loops, first.prefix, first.suffix),
        (0..1, true, false)
    );
    let middle = DoryStage::of_rank(&config, 2, 4).expect("rank 2");
    assert_eq!(
        (middle.loops, middle.prefix, middle.suffix),
        (2..3, false, false)
    );
    let last = DoryStage::of_rank(&config, 1, 2).expect("rank 1 of 2");
    assert_eq!((last.loops, last.prefix, last.suffix), (2..4, false, true));
    let error = DoryStage::of_rank(&config, 0, 3).expect_err("4 loops do not split 3 ways");
    assert!(error.contains("evenly"), "{error}");
    let error = DoryStage::of_rank(&config, 4, 4).expect_err("rank 4 is outside 4");
    assert!(error.contains("outside"), "{error}");
    let mut last_loop: serde_json::Value =
        serde_json::from_str(&fixture_json(4)).expect("the fixture is JSON");
    last_loop["recurrent_kv_cache_mode"] = serde_json::json!("last_loop");
    let config = DoryConfig::from_json(&last_loop.to_string()).expect("last-loop is valid");
    let error = DoryStage::of_rank(&config, 0, 2).expect_err("last-loop cannot split");
    assert!(error.contains("per_loop"), "{error}");
    assert!(
        DoryStage::of_rank(&config, 0, 1).is_ok(),
        "one rank is the whole model"
    );
}
