// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The device arm and the f32 host reference on the tiny fixture: two walks of one model.
//!
//! The host reference is the oracle, and the two backends share the fixture, the schedule's
//! widths and the sampler. They do not share arithmetic: the device narrows its key/value cache
//! and its attention output to BF16 (the released precision), sums a projection with a warp's
//! partials rather than one accumulator, and reads its exponentials from the fast path. So the
//! comparison is not bit equality; it is that the two walks rank the vocabulary the same way and
//! agree on the logits to a bound far below the gap between two candidates.

use std::sync::Arc;

use sparkflash_model::backend::{ModelBackend, Sampling};
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::device::InklingDeviceBackend;
use sparkflash_model_inkling2::fixture::tiny_source;
use sparkflash_model_inkling2::host::Inkling2HostBackend;

/// The largest logit difference allowed, relative to the row's largest magnitude.
///
/// Two walks that sum the same products in different orders and narrow the key/value cache, the
/// attention output and the logits to BF16 cannot agree to the last bit. On the tiny fixture the
/// measured difference is 0.2-0.3% of the row's largest magnitude, so the bound is an order of
/// magnitude above the observed rounding and still two orders below the gap between candidates.
const RELATIVE_TOLERANCE: f32 = 1e-2;

fn greedy() -> Sampling {
    Sampling {
        temperature: 0.0,
        top_k: 1,
        ..Sampling::default()
    }
}

fn backend_available() -> bool {
    if !InklingDeviceBackend::available() {
        eprintln!("skipped: this build embeds no Inkling or token cubin");
        return false;
    }
    true
}

#[test]
fn the_device_prefill_and_decode_agree_with_the_host() {
    if !backend_available() {
        return;
    }
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let host = Inkling2HostBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config.clone(),
        Arc::clone(&source) as Arc<dyn sparkflash_weights::WeightSource>,
        16,
    )
    .expect("the host reference opens");
    let device = InklingDeviceBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config.clone(),
        Arc::clone(&source) as Arc<dyn sparkflash_weights::WeightSource>,
        16,
    )
    .expect("the device arm opens");
    assert!(device.weight_bytes() > 0, "the arm uploaded no weight");
    let mut host_session = host.open_session().expect("a host session");
    let mut device_session = device.open_session().expect("a device session");
    host_session.set_sampling(greedy());
    device_session.set_sampling(greedy());

    let prompt = [3u32, 7, 11];
    let host_first = host_session.prefill(&prompt).expect("the host prompt");
    let device_first = device_session.prefill(&prompt).expect("the device prompt");
    assert_eq!(device_first.logits.len(), config.unpadded_vocab_size);
    assert!(
        device_first.logits.iter().all(|value| value.is_finite()),
        "a device logit is not finite"
    );
    compare("prefill", &host_first.logits, &device_first.logits);

    // One decode step, from the token the host chose: both walks see the same history.
    let next = host_first.next_token.expect("a next token");
    let host_second = host_session.decode(next).expect("the host decode");
    let device_second = device_session.decode(next).expect("the device decode");
    compare("decode", &host_second.logits, &device_second.logits);

    assert_eq!(
        device_session.reusable_prefix(),
        Some([3, 7, 11, next].as_slice()),
        "the device session's history"
    );
    // The context is 16 tokens and four are used, so the fifth decode must run.
    device_session
        .decode(next)
        .expect("the device accepts a token after the history");
}

/// The device's answer against the host's: the same best token, and logits within the bound.
fn compare(what: &str, host: &[f32], device: &[f32]) {
    assert_eq!(host.len(), device.len(), "{what}: the rows differ in width");
    let best = |values: &[f32]| -> usize {
        let mut index = 0usize;
        for (candidate, value) in values.iter().enumerate() {
            if *value > values[index] {
                index = candidate;
            }
        }
        index
    };
    let host_best = best(host);
    let device_best = best(device);
    let scale = host
        .iter()
        .fold(0f32, |largest, value| largest.max(value.abs()));
    let difference = host
        .iter()
        .zip(device)
        .fold(0f32, |largest, (left, right)| {
            largest.max((left - right).abs())
        });
    eprintln!(
        "{what}: host best {host_best} ({:.4}), device best {device_best} ({:.4}), largest \
         difference {difference:.5} of {scale:.4}",
        host[host_best], device[device_best]
    );
    // The head of the row, on the way to a failure: a scale, a permutation and a rounding are
    // told apart in three numbers.
    if device_best != host_best || difference > RELATIVE_TOLERANCE * scale {
        for index in 0..host.len().min(6) {
            eprintln!(
                "{what}: [{index}] host {:.5} device {:.5}",
                host[index], device[index]
            );
        }
    }
    assert_eq!(
        device_best, host_best,
        "{what}: the device chose {device_best} and the host {host_best}"
    );
    assert!(
        difference <= RELATIVE_TOLERANCE * scale,
        "{what}: the logits differ by {difference}, past the bound {}",
        RELATIVE_TOLERANCE * scale
    );
}

#[test]
fn the_device_arm_refuses_a_context_past_the_window() {
    if !backend_available() {
        return;
    }
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let refused = InklingDeviceBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config.clone(),
        Arc::clone(&source) as Arc<dyn sparkflash_weights::WeightSource>,
        4096,
    );
    assert!(
        refused.is_err(),
        "a context past the checkpoint's window opens"
    );
    let opened = InklingDeviceBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config,
        source as Arc<dyn sparkflash_weights::WeightSource>,
        16,
    )
    .expect("the fixture opens");
    // The schedule the arm executes is the plan's, not a second walk's.
    assert_eq!(
        opened.plan().pass(),
        sparkflash_model_inkling2::device_plan::Pass::Decode
    );
    assert_eq!(opened.plan().layers().len(), 2);
}

/// Lanes stepped together answer what each would decode alone, to the bit.
///
/// Three prompts of different lengths, so the lanes sit at different positions and their caches
/// and convolution histories differ: a lane that read another's state, or a row the walk summed
/// in another order, moves a logit. A step of one member is the decode step itself, so the last
/// steps drop the lanes to one and check that path too.
#[test]
fn fused_lanes_step_as_each_lane_decodes_alone() {
    use sparkflash_runtime::FusedArm;

    if !backend_available() {
        return;
    }
    let config = Inkling2Config::tiny();
    let source = Arc::new(tiny_source(&config));
    let device = InklingDeviceBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config,
        Arc::clone(&source) as Arc<dyn sparkflash_weights::WeightSource>,
        16,
    )
    .expect("the device arm opens");
    let prompts: [&[u32]; 3] = [&[3, 7, 11], &[5], &[2, 9, 4, 8, 1]];

    // Each prompt alone: its first answer and the tokens it decodes after it.
    let steps = 4usize;
    let mut solo: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut fed: Vec<Vec<u32>> = Vec::new();
    for prompt in prompts {
        let mut session = device.open_session().expect("a device session");
        session.set_sampling(greedy());
        let first = session.prefill(prompt).expect("the solo prompt");
        let mut rows = vec![first.logits.clone()];
        let mut tokens = Vec::new();
        let mut next = first.next_token.expect("a next token");
        for _ in 0..steps {
            tokens.push(next);
            let output = session.decode(next).expect("the solo decode");
            next = output.next_token.expect("a next token");
            rows.push(output.logits);
        }
        solo.push(rows);
        fed.push(tokens);
    }

    let lanes = sparkflash_runtime::Runtime::new(device.walk_arm().expect("the walk"), 0)
        .expect("the runtime opens");
    let mut slots = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let mut slot = lanes.open_slot().expect("a lane");
        lanes.set_sampling(&mut slot, greedy());
        let first = lanes.prefill(&mut slot, prompt).expect("the lane's prompt");
        assert_eq!(
            first.logits, solo[index][0],
            "lane {index}: the prompt's answer"
        );
        slots.push(slot);
    }
    for step in 0..steps {
        // The last step runs the first lane alone: the one-member path.
        let width = if step + 1 == steps { 1 } else { slots.len() };
        let mut members: Vec<_> = slots
            .drain(..width)
            .enumerate()
            .map(|(index, slot)| (slot, fed[index][step]))
            .collect();
        let outputs = lanes.step(&mut members).expect("the fused step");
        assert_eq!(outputs.len(), width);
        for (index, output) in outputs.iter().enumerate() {
            let expected = &solo[index][step + 1];
            let differ = output
                .logits
                .iter()
                .zip(expected)
                .filter(|(left, right)| left.to_bits() != right.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "step {step}, lane {index}: {differ} logits differ from the lane decoding alone"
            );
        }
        let mut kept: Vec<_> = members.into_iter().map(|(slot, _)| slot).collect();
        kept.append(&mut slots);
        slots = kept;
    }
    // A lane that served a request is renewed for the next one and answers it from position zero.
    let mut slot = slots.remove(0);
    assert!(lanes.holds_request(&slot));
    lanes.renew(&mut slot).expect("the lane renews");
    lanes.set_sampling(&mut slot, greedy());
    let again = lanes
        .prefill(&mut slot, prompts[1])
        .expect("the renewed prompt");
    assert_eq!(
        again.logits, solo[1][0],
        "a renewed lane answers its prompt afresh"
    );
}

/// Four ranks of one checkpoint: the converted rank files, four backends and the loopback
/// exchange, against the same host reference a single rank is checked on.
///
/// This is the arm's tensor-parallel path end to end without a cluster: the conversion splits
/// the checkpoint exactly once, each rank loads only its file, and the three collectives carry
/// the embedding row, the expert mix and the logits between them.
#[test]
fn four_converted_ranks_answer_like_one_rank() {
    if !backend_available() {
        return;
    }
    // The four-expert fixture: every axis a four-rank plan shards divides four.
    let config = Inkling2Config::tiny_world();
    let source = tiny_source(&config);
    let directory = tempfile::tempdir().expect("a temporary directory");
    let mut paths = Vec::new();
    for rank in 0..4usize {
        let layout = sparkflash_weights::tp::TpLayout {
            rank,
            world_size: 4,
        };
        let written = sparkflash_model_inkling2::convert::convert_rank(
            &source,
            &config,
            layout,
            directory.path(),
            "tiny",
        )
        .expect("the rank file is written");
        paths.push(written.path);
    }

    let reference = {
        let host = Inkling2HostBackend::open(
            sparkflash_model_inkling2::descriptor(),
            config.clone(),
            Arc::new(tiny_source(&config)) as Arc<dyn sparkflash_weights::WeightSource>,
            16,
        )
        .expect("the host reference opens");
        let mut session = host.open_session().expect("a host session");
        session.set_sampling(greedy());
        session.prefill(&[3, 7, 11]).expect("the host prompt")
    };

    let base_port = 29_623u16;
    let prompt = [3u32, 7, 11];
    let mut handles = Vec::new();
    for (rank, path) in paths.into_iter().enumerate() {
        let config = config.clone();
        handles.push(std::thread::spawn(move || -> Result<Vec<f32>, String> {
            let layout = sparkflash_weights::tp::TpLayout {
                rank,
                world_size: 4,
            };
            let source =
                sparkflash_model_inkling2::convert::RankFileSource::open(&path, &config, layout)?;
            let exchange = sparkflash_model_inkling2::collective::TcpExchange::connect(
                &sparkflash_comm::tcp::TcpConfig::loopback4(rank, base_port),
            )?;
            let tp = sparkflash_model::TpPlan::new(4, rank).map_err(|error| error.detail)?;
            // The served ranks' load: the data section whole, in parallel direct reads.
            let backend = InklingDeviceBackend::open_rank_file(
                sparkflash_model_inkling2::descriptor(),
                config,
                &source,
                16,
                tp,
                Box::new(exchange),
            )?;
            let mut session = backend
                .open_session()
                .map_err(|error| error.message().to_owned())?;
            session.set_sampling(greedy());
            let output = session
                .prefill(&prompt)
                .map_err(|error| error.message().to_owned())?;
            Ok(output.logits)
        }));
    }
    let mut achieved = Vec::new();
    let mut failures = Vec::new();
    for (rank, handle) in handles.into_iter().enumerate() {
        match handle.join().expect("the rank thread") {
            Ok(logits) => achieved.push((rank, logits)),
            Err(error) => failures.push(format!("rank {rank}: {error}")),
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
    // Every rank gathered the same row, so all four hold the reference's vocabulary.
    for (rank, logits) in &achieved {
        assert_eq!(logits.len(), reference.logits.len(), "rank {rank} width");
        compare(&format!("tp4/rank{rank}"), &reference.logits, logits);
    }
}
