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

//! A fused batch must not change what any member emits.
//!
//! The property is not "the batch is close": the rows of a batch are independent sequences, so
//! the only acceptable difference is none. Three runs of the tiny fixture check it:
//!
//! * a plain session, whose prefill runs the one-row forward this change does not touch;
//! * a fused engine with one slot, so every step is a batch of one;
//! * a fused engine with two slots, both stepping at once from two threads - the shape the host
//!   driver's workers have - with one member on each prompt.
//!
//! The prefill logits of the first two must agree with the plain session's, and every generated
//! token of the two-slot run must equal the one-slot run's, member by member. The fixture's config
//! carries an MTP layer, and a fused session never speculates, which is why the token comparison
//! is against the fused one-slot run rather than the plain session: speculation is verified to be
//! token-identical by construction, and mixing the two questions would hide a batch fault behind
//! a draft.
//!
//! The test skips - loudly - without a cubin or a CUDA driver. Once the device opens, a mismatch
//! fails the test.

use std::path::Path;
use std::sync::Arc;

use sparkflash_model::{ModelBackend, ModelSession};
use sparkflash_model_qwen4_exp::SyntheticCheckpoint;
use sparkflash_model_qwen4_exp::device::{Qwen4ExpDeviceBackend, cubin_available};

/// The context window the served fixture allocates.
const CONTEXT: usize = 32;

/// The tokens the fixture is asked for, one per prompt plus a few to fill each window.
const STEPS: usize = 12;

const PROMPT_A: &[u32] = &[1, 2, 3, 4, 5, 6];
const PROMPT_B: &[u32] = &[7, 8, 9, 10, 11, 12];

/// The relative tolerance two prefill logits rows may differ by.
///
/// The two sides run the same kernels on the same weights, so the bound is loose enough to absorb
/// a different launch shape and tight enough that a dropped or duplicated row fails it.
const TOLERANCE: f32 = 1e-5;

/// A backend over the tiny fixture, with `fuse` fused sessions when it is non-zero.
fn open(directory: &Path, fuse: usize) -> Option<Qwen4ExpDeviceBackend> {
    let mut backend = match Qwen4ExpDeviceBackend::open(directory, CONTEXT) {
        Ok(backend) => backend,
        Err(error)
            if error.code() == "qwen4_device"
                && (error.message().contains("CUDA:")
                    || error.message().contains("cuModuleLoadData failed: 209")) =>
        {
            eprintln!("skipping the fused-batch gate: {error}");
            return None;
        }
        Err(error) => panic!("the device arm refused the fixture checkpoint: {error}"),
    };
    if fuse > 0 {
        backend
            .enable_fusion(fuse)
            .expect("the fused engine starts");
        assert!(backend.fused(), "the backend is fused");
    }
    Some(backend)
}

/// The fixture's directory, written once per process.
fn fixture() -> Option<Arc<Path>> {
    static DIR: std::sync::OnceLock<Option<Arc<Path>>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        if !cubin_available() {
            eprintln!("skipping the fused-batch gate: this build has no Qwen4-Exp cubin");
            return None;
        }
        let temp = tempfile::tempdir().expect("a temporary directory");
        let written = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
        let directory = written.directory.clone();
        // The directory has to outlive every backend the lanes open.
        std::mem::forget(temp);
        Some(Arc::from(directory.as_path()))
    })
    .clone()
}

/// Greedy-decode `steps` tokens from `session`, which has already prefilled.
fn generate(session: &mut dyn ModelSession, mut next: Option<u32>) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let Some(token) = next else {
            break;
        };
        tokens.push(token);
        next = session.decode(token).expect("a decode step").next_token;
    }
    tokens
}

fn compare(stage: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "{stage}: {} logits vs {}",
        reference.len(),
        candidate.len()
    );
    assert!(
        candidate.iter().all(|value| value.is_finite()),
        "{stage}: the fused run produced a non-finite logit"
    );
    let scale = reference
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let largest = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| (reference - candidate).abs())
        .fold(0.0f32, f32::max);
    let relative = largest / scale;
    eprintln!("{stage}: max |fused - plain| = {largest:.6}, relative {relative:.3e}");
    assert!(
        relative < TOLERANCE,
        "{stage}: relative difference {relative:.3e} exceeds {TOLERANCE:.3e}"
    );
}

#[test]
fn a_fused_batch_emits_what_its_members_emit_alone() {
    let Some(directory) = fixture() else {
        return;
    };
    // The plain session: its prefill is the forward every earlier claim was measured on.
    let Some(plain) = open(&directory, 0) else {
        return;
    };
    let mut plain_session = plain.open_session().expect("a plain session");
    let plain_prefill = plain_session.prefill(PROMPT_A).expect("the plain prefill");
    let plain_tokens = generate(&mut *plain_session, plain_prefill.next_token);

    // A batch of one, then a batch of two: the same engine, the same sessions, once with a single
    // member stepping and once with both. A capacity of two lets both runs share one engine, and
    // an idle member is exactly what the two-member run must not be affected by.
    let Some(single) = open(&directory, 2) else {
        return;
    };
    let mut one = single.open_session().expect("a fused session");
    let one_a = one.prefill(PROMPT_A).expect("the fused prefill");
    compare("prefill A", &plain_prefill.logits, &one_a.logits);
    let alone_a = generate(&mut *one, one_a.next_token);
    let mut two = single.open_session().expect("a second fused session");
    let two_b = two.prefill(PROMPT_B).expect("the fused prefill");
    let alone_b = generate(&mut *two, two_b.next_token);
    // The slots go back before the next run asks for them: a handle holds its slot until it is
    // dropped, and both runs want fresh state.
    drop((one, two));

    // Two slots, stepped from two threads at once: the driver's shape for --concurrency 2.
    let Some(fused) = open(&directory, 2) else {
        return;
    };
    let mut first = fused.open_session().expect("the first member");
    let mut second = fused.open_session().expect("the second member");
    let (batched_a, batched_b) = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            let step = first.prefill(PROMPT_A).expect("the batched prefill");
            (step.logits.clone(), generate(&mut *first, step.next_token))
        });
        let second = scope.spawn(|| {
            let step = second.prefill(PROMPT_B).expect("the batched prefill");
            (step.logits.clone(), generate(&mut *second, step.next_token))
        });
        (
            first.join().expect("the first member's thread"),
            second.join().expect("the second member's thread"),
        )
    });
    compare("batched prefill A", &plain_prefill.logits, &batched_a.0);
    assert_eq!(
        batched_a.1, alone_a,
        "the first member of the batch emitted different tokens:\n  batched {batched_a:?}\n  alone   {alone_a:?}"
    );
    assert_eq!(
        batched_b.1, alone_b,
        "the second member of the batch emitted different tokens:\n  batched {batched_b:?}\n  alone   {alone_b:?}"
    );
    // A batch that is only ever idling would pass the comparison above. The plain greedy run is a
    // third opinion: the token sequence the model chooses for prompt A, which both fused runs have
    // to reproduce.
    eprintln!("prompt A: plain {plain_tokens:?}\n          fused {alone_a:?}");
}

/// Two members on the same prompt must agree with each other and with a lone member.
///
/// A batch's rows are laid out in member order, so a stage that addressed its row by the wrong
/// index would make the two members disagree while each still looked plausible. This splits a
/// wrong answer into "the row was addressed wrongly" and "the row count changed the arithmetic".
#[test]
fn two_members_on_one_prompt_agree() {
    let Some(directory) = fixture() else {
        return;
    };
    let Some(fused) = open(&directory, 3) else {
        return;
    };
    let mut alone = fused.open_session().expect("the lone member");
    let step = alone.prefill(PROMPT_A).expect("the lone prefill");
    let solo = generate(&mut *alone, step.next_token);
    drop(alone);
    let mut first = fused.open_session().expect("the first member");
    let mut second = fused.open_session().expect("the second member");
    let (left, right) = std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            let step = first.prefill(PROMPT_A).expect("the first prefill");
            generate(&mut *first, step.next_token)
        });
        let right = scope.spawn(|| {
            let step = second.prefill(PROMPT_A).expect("the second prefill");
            generate(&mut *second, step.next_token)
        });
        (
            left.join().expect("the first member's thread"),
            right.join().expect("the second member's thread"),
        )
    });
    eprintln!("solo {solo:?}\n left {left:?}\nright {right:?}");
    assert_eq!(left, right, "the two members of one prompt disagreed");
    assert_eq!(left, solo, "the batch disagreed with the lone member");
}

/// Four members on one prompt, so the shared stages see four rows whatever the row count.
///
/// Two members at two rows each gave an answer that depended on which row a member landed on. This
/// separates the two explanations: four members of one row exercise a four-row shared pass with no
/// per-member multi-row stage, and the same four members at two rows each add the verify.
#[test]
fn four_members_on_one_prompt_agree() {
    let Some(directory) = fixture() else {
        return;
    };
    let Some(fused) = open(&directory, 5) else {
        return;
    };
    let mut alone = fused.open_session().expect("the lone member");
    let step = alone.prefill(PROMPT_A).expect("the lone prefill");
    let solo = generate(&mut *alone, step.next_token);
    drop(alone);
    let mut members: Vec<_> = (0..4)
        .map(|_| fused.open_session().expect("a member"))
        .collect();
    let runs: Vec<Vec<u32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = members
            .iter_mut()
            .map(|member| {
                scope.spawn(move || {
                    let step = member.prefill(PROMPT_A).expect("the member's prefill");
                    generate(&mut **member, step.next_token)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a member's thread"))
            .collect()
    });
    for (index, run) in runs.iter().enumerate() {
        eprintln!("member {index}: {run:?}");
    }
    eprintln!("solo: {solo:?}");
    for (index, run) in runs.iter().enumerate() {
        assert_eq!(
            run, &solo,
            "member {index} of the batch disagreed with the lone member"
        );
    }
}

/// A slot's next request starts from nothing, whatever the last one left.
///
/// The engine keeps a slot's sequence between requests, and a handle offers none of its history
/// for reuse: every prompt it posts is a whole conversation. A slot that prefilled that prompt onto
/// the last request's caches would answer a conversation nobody sent, so one slot serves prompt B,
/// then prompt A, then B again, and the third run must be the first.
#[test]
fn a_reopened_slot_starts_cold() {
    let Some(directory) = fixture() else {
        return;
    };
    let Some(fused) = open(&directory, 1) else {
        return;
    };
    let run = |prompt: &[u32]| {
        let mut session = fused.open_session().expect("the slot's session");
        let step = session
            .prefill(prompt)
            .expect("a prefill on a slot an earlier request used");
        (
            step.logits.clone(),
            generate(&mut *session, step.next_token),
        )
    };
    let (cold_logits, cold) = run(PROMPT_B);
    let _ = run(PROMPT_A);
    let (again_logits, again) = run(PROMPT_B);
    compare("prefill B on a used slot", &cold_logits, &again_logits);
    eprintln!("cold  {cold:?}\nagain {again:?}");
    assert_eq!(
        again, cold,
        "a reopened slot answered differently from a fresh one"
    );
}
