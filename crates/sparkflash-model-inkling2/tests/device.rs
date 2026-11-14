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

use sparkflash_model::backend::{ModelBackend, ModelSession, Sampling};
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::device::{InklingDeviceBackend, DEVICE};
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
    let device_first = device_session
        .prefill(&prompt)
        .expect("the device prompt");
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
    let scale = host.iter().fold(0f32, |largest, value| largest.max(value.abs()));
    let difference = host
        .iter()
        .zip(device)
        .fold(0f32, |largest, (left, right)| largest.max((left - right).abs()));
    eprintln!(
        "{what}: host best {host_best} ({:.4}), device best {device_best} ({:.4}), largest \
         difference {difference:.5} of {scale:.4}",
        host[host_best], device[device_best]
    );
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
    assert_eq!(opened.plan().pass(), sparkflash_model_inkling2::device_plan::Pass::Decode);
    assert_eq!(opened.plan().layers().len(), 2);
}
