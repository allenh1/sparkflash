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

//! With joint rounds off, Qwen drafts nothing and each lane's round walks alone.
//!
//! The arm's own policy (kept from its fusion) is that `SPARKFLASH_JOINT_ROUNDS=off` means no
//! drafting at all, not one verify over every lane. The runtime then runs each opened member's
//! round on its own. This binary sets the switch before any backend opens, so the environment is
//! its own process and the parallel tests in `batch_exactness.rs` cannot see it. The run must
//! still emit exactly what the plain session emits, and its counters must show no drafts.
//!
//! The test skips - loudly - without a cubin or a CUDA driver.

use std::path::Path;
use std::sync::Arc;

use sparkflash_model::{ModelBackend, ModelSession};
use sparkflash_model_qwen4_exp::SyntheticCheckpoint;
use sparkflash_model_qwen4_exp::device::{Qwen4ExpDeviceBackend, cubin_available};

const CONTEXT: usize = 32;
const STEPS: usize = 12;
const PROMPT_A: &[u32] = &[1, 2, 3, 4, 5, 6];
const PROMPT_B: &[u32] = &[7, 8, 9, 10, 11, 12];

/// A backend over the tiny fixture, with `fuse` fused sessions when it is non-zero.
fn open(directory: &Path, fuse: usize) -> Option<Qwen4ExpDeviceBackend> {
    let mut backend = match Qwen4ExpDeviceBackend::open(directory, CONTEXT) {
        Ok(backend) => backend,
        Err(error)
            if error.code() == "qwen4_device"
                && (error.message().contains("CUDA:")
                    || error.message().contains("cuModuleLoadData failed: 209")) =>
        {
            eprintln!("skipping the joint-off gate: {error}");
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
            eprintln!("skipping the joint-off gate: this build has no Qwen4-Exp cubin");
            return None;
        }
        let temp = tempfile::tempdir().expect("a temporary directory");
        // The tiny fixture is not the released target the canonical DFlash 2 drafter was trained
        // for, so a resolved cache snapshot would be refused at open. HF_HUB_CACHE wins over the
        // developer's cache variables, and an empty hub directory leaves the MTP stage as the
        // drafter this test needs.
        // SAFETY: only this test binary mutates the environment, and its one test calls
        // fixture() first, so nothing else runs while the variables are set.
        unsafe {
            std::env::set_var("HF_HUB_CACHE", temp.path().join("hub"));
            std::env::set_var("SPARKFLASH_JOINT_ROUNDS", "off");
        }
        let written = SyntheticCheckpoint::write(temp.path()).expect("the fixture writes");
        let directory = written.directory.clone();
        std::mem::forget(temp);
        Some(Arc::from(directory.as_path()))
    })
    .clone()
}

/// Greedy-decode `STEPS` tokens from `session`, which has already prefilled.
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

#[test]
fn joint_off_drafts_nothing_and_still_emits_the_plain_tokens() {
    let Some(directory) = fixture() else {
        return;
    };
    let Some(plain) = open(&directory, 0) else {
        return;
    };
    let mut plain_session = plain.open_session().expect("a plain session");
    let plain_prefill = plain_session.prefill(PROMPT_A).expect("the plain prefill");
    let plain_tokens = generate(&mut *plain_session, plain_prefill.next_token);

    let Some(fused) = open(&directory, 2) else {
        return;
    };
    let mut first = fused.open_session().expect("the first member");
    let mut second = fused.open_session().expect("the second member");
    let (left, right) = std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            let step = first.prefill(PROMPT_A).expect("the first prefill");
            let tokens = generate(&mut *first, step.next_token);
            (tokens, first.speculation())
        });
        let right = scope.spawn(|| {
            let step = second.prefill(PROMPT_B).expect("the second prefill");
            let tokens = generate(&mut *second, step.next_token);
            (tokens, second.speculation())
        });
        (
            left.join().expect("the first member's thread"),
            right.join().expect("the second member's thread"),
        )
    });
    assert_eq!(
        left.0, plain_tokens,
        "a joint-off member emitted other tokens than the plain session"
    );
    for (index, (_, speculation)) in [left, right].iter().enumerate() {
        assert_eq!(
            speculation.verify_steps, 0,
            "member {index} verified a draft with joint rounds off"
        );
        assert_eq!(
            speculation.drafts_proposed, 0,
            "member {index} proposed drafts with joint rounds off"
        );
    }
}
