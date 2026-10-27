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

//! Scenario tests for the continuous batcher: admission, prefill/decode mixing, chunk
//! boundaries, fairness, determinism, preemption and prefix-cache reuse.
//!
//! Everything here runs on synthetic token fixtures: no weights, no device, no cluster.

use std::collections::BTreeSet;

use sparkflash_scheduler::RequestId;
use sparkflash_scheduler::batching::{
    BatchConfig, BatchStep, ContinuousBatcher, PreemptionPolicy, PrefillMode, RequestPhase,
    ScheduledOp,
};

/// A deterministic token id for a position. The scheduler never interprets it, so any stable
/// mapping works; a shared position space makes two prompts with the same ids a prefix hit.
fn token_for(_seq: RequestId, position: usize) -> u32 {
    (position as u32) + 1
}

/// Plans and accepts steps until every submitted request is finished.
fn run(batcher: &mut ContinuousBatcher, max_steps: usize) -> Vec<BatchStep> {
    let mut steps = Vec::new();
    let mut idle = 0;
    for _ in 0..max_steps {
        batcher.check_invariants().expect("batcher invariants hold");
        let step = batcher.plan_step().expect("plan");
        assert!(
            step.prefill_tokens + step.decode_tokens <= step.token_budget,
            "step {} spends {} prefill + {} decode tokens of a {} budget",
            step.step,
            step.prefill_tokens,
            step.decode_tokens,
            step.token_budget
        );
        let produced: Vec<(RequestId, u32)> = step
            .ops
            .iter()
            .filter(|op| op.produces_token())
            .map(|op| {
                let position = match op {
                    ScheduledOp::Decode { position, .. } => *position,
                    ScheduledOp::PrefillChunk { range, .. } => range.end,
                    _ => unreachable!("only decodes and emitting chunks produce tokens"),
                };
                (op.seq(), token_for(op.seq(), position))
            })
            .collect();
        batcher
            .accept_step(&step, &produced)
            .expect("accept the tokens the plan's ops produced");
        let live = batcher.waiting().len() + batcher.running().len() + batcher.swapped().len();
        steps.push(step);
        if live == 0 {
            return steps;
        }
        if steps.last().expect("just pushed").is_empty() {
            idle += 1;
            assert!(idle < 3, "the batcher made no progress for {idle} steps");
        } else {
            idle = 0;
        }
    }
    panic!(
        "the batch did not drain in {max_steps} steps: {:?}",
        batcher.snapshot()
    );
}

fn prompt(len: usize, salt: u32) -> Vec<u32> {
    (0..len).map(|index| (index as u32) + 1 + salt).collect()
}

fn small() -> BatchConfig {
    BatchConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 8,
        prefill_chunk_tokens: 4,
        kv_pages: 16,
        kv_tails: 2,
        ..BatchConfig::default()
    }
}

#[test]
fn the_batched_default_forwards_one_chunk_per_step_not_one_token() {
    let mut batcher = ContinuousBatcher::new(small()).expect("batcher");
    let id = batcher.submit(prompt(10, 0), 1).expect("submit");
    let steps = run(&mut batcher, 20);
    let chunks: Vec<_> = steps
        .iter()
        .flat_map(|step| step.ops_for(id))
        .filter(|op| matches!(op, ScheduledOp::PrefillChunk { .. }))
        .collect();
    assert_eq!(
        chunks.len(),
        3,
        "10 tokens at 4 per chunk is three forwards"
    );
    assert_eq!(
        steps
            .iter()
            .flat_map(|step| step.ops_for(id))
            .filter(|op| matches!(op, ScheduledOp::Decode { .. }))
            .count(),
        0,
        "a one-token completion never decodes: the last prefill chunk produces it"
    );
    assert_eq!(chunks[0].seq(), id);
    let ranges: Vec<(usize, usize)> = chunks
        .iter()
        .map(|op| match op {
            ScheduledOp::PrefillChunk { range, .. } => (range.start, range.end),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(ranges, vec![(0, 4), (4, 8), (8, 10)]);
    let stats = batcher.stats();
    assert_eq!(stats.host_forwards, 3);
    assert_eq!(stats.prefill_tokens, 10);
    assert_eq!(stats.decode_tokens, 0);
    assert_eq!(
        batcher.request(id).expect("known").phase,
        RequestPhase::Finished
    );
}

#[test]
fn the_token_loop_compatibility_mode_is_one_forward_per_prompt_token() {
    let config = small().with_prefill_mode(PrefillMode::TokenLoop);
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let id = batcher.submit(prompt(5, 0), 1).expect("submit");
    run(&mut batcher, 20);
    assert_eq!(batcher.stats().host_forwards, 5);
    assert_eq!(batcher.stats().prefill_tokens, 5);
    assert_eq!(batcher.stats().decode_tokens, 0);
    assert_eq!(
        batcher.request(id).expect("known").phase,
        RequestPhase::Finished
    );
}

#[test]
fn a_step_mixes_decode_tokens_with_prefill_chunks() {
    let mut batcher = ContinuousBatcher::new(small()).expect("batcher");
    let long = batcher.submit(prompt(5, 0), 2).expect("submit");
    let short = batcher.submit(prompt(1, 100), 3).expect("submit");
    let steps = run(&mut batcher, 30);
    let mixed = steps.iter().filter(|step| {
        step.ops
            .iter()
            .any(|op| matches!(op, ScheduledOp::Decode { .. }))
            && step
                .ops
                .iter()
                .any(|op| matches!(op, ScheduledOp::PrefillChunk { .. }))
    });
    assert!(
        mixed.count() >= 1,
        "at least one step must mix a decode with a prefill chunk"
    );
    assert_eq!(batcher.request(long).expect("known").generated_tokens, 2);
    assert_eq!(batcher.request(short).expect("known").generated_tokens, 3);
    // The decode-first policy: in a mixed step the decode ops come before the prefill ops.
    for step in &steps {
        let first_chunk = step
            .ops
            .iter()
            .position(|op| matches!(op, ScheduledOp::PrefillChunk { .. }));
        let last_decode = step
            .ops
            .iter()
            .rposition(|op| matches!(op, ScheduledOp::Decode { .. }));
        if let (Some(chunk), Some(decode)) = (first_chunk, last_decode) {
            assert!(decode < chunk, "decode-first means decodes precede chunks");
        }
    }
}

#[test]
fn chunks_are_contiguous_budgeted_and_at_most_one_per_request_per_step() {
    let mut config = small();
    config.prefill_chunk_tokens = 3;
    config.max_num_batched_tokens = 6;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let left = batcher.submit(prompt(7, 0), 1).expect("submit");
    let right = batcher.submit(prompt(7, 100), 1).expect("submit");
    // Both requests must drain; the cursors below assert they forwarded every token.
    assert_ne!(left, right);
    let steps = run(&mut batcher, 30);
    let mut cursors = [(0usize, 0usize); 2];
    for step in &steps {
        assert!(
            step.prefill_tokens <= 6,
            "step {} forwarded {} prefill tokens over the budget",
            step.step,
            step.prefill_tokens
        );
        for op in &step.ops {
            if let ScheduledOp::PrefillChunk { seq, range, .. } = op {
                let slot = if *seq == left { 0 } else { 1 };
                assert_eq!(
                    range.start, cursors[slot].0,
                    "request {seq:?} chunk is not contiguous"
                );
                cursors[slot] = (range.end, cursors[slot].1 + 1);
            }
        }
        let mut seen = BTreeSet::new();
        for op in &step.ops {
            if matches!(op, ScheduledOp::PrefillChunk { .. }) {
                assert!(
                    seen.insert(op.seq()),
                    "one prefill chunk per request per step"
                );
            }
        }
    }
    assert_eq!(cursors[0].0, 7);
    assert_eq!(cursors[1].0, 7);
    assert!(cursors[0].1 >= 3 && cursors[1].1 >= 3);
}

#[test]
fn a_tight_budget_does_not_starve_a_prefilling_request() {
    let mut config = small();
    config.prefill_chunk_tokens = 2;
    config.max_num_batched_tokens = 2;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let first = batcher.submit(prompt(6, 0), 1).expect("submit");
    let second = batcher.submit(prompt(6, 100), 1).expect("submit");
    let steps = run(&mut batcher, 30);
    let mut served = [0usize; 2];
    let mut order = Vec::new();
    for step in &steps {
        for op in &step.ops {
            if matches!(op, ScheduledOp::PrefillChunk { .. }) {
                let slot = if op.seq() == first { 0 } else { 1 };
                served[slot] += 1;
                order.push(slot);
            }
        }
    }
    assert!(served[0] >= 3 && served[1] >= 3, "{served:?}");
    assert_eq!(batcher.request(first).expect("known").generated_tokens, 1);
    assert_eq!(batcher.request(second).expect("known").generated_tokens, 1);
    assert!(
        order.windows(2).any(|pair| pair[0] != pair[1]),
        "the budget rotates between requests: {order:?}"
    );
}

#[test]
fn the_same_submissions_produce_the_same_step_sequence() {
    let trace = || -> Vec<String> {
        let mut batcher = ContinuousBatcher::new(small()).expect("batcher");
        batcher.submit(prompt(5, 0), 2).expect("submit");
        batcher.submit(prompt(3, 100), 2).expect("submit");
        run(&mut batcher, 30)
            .iter()
            .map(|step| format!("{step:?}"))
            .collect()
    };
    assert_eq!(trace(), trace());
}

#[test]
fn preemption_recomputes_the_consumed_history() {
    let mut config = small();
    config.max_num_seqs = 2;
    config.kv_pages = 4;
    config.prefill_chunk_tokens = 4096;
    config.max_num_batched_tokens = 8192;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let first = batcher.submit(prompt(128, 0), 80).expect("submit");
    let second = batcher.submit(prompt(128, 1000), 80).expect("submit");
    let steps = run(&mut batcher, 400);
    assert!(
        batcher.stats().preemptions >= 1,
        "the 4-page pool cannot hold two 128-token prompts plus their decode pages"
    );
    let preempted: Vec<RequestId> = steps
        .iter()
        .flat_map(|step| step.preempted.iter().copied())
        .collect();
    let victim = *preempted.first().expect("a victim");
    let resume_step = steps
        .iter()
        .position(|step| step.swapped_in.contains(&victim))
        .expect("the victim resumed");
    let resume_ops: Vec<&ScheduledOp> = steps[resume_step..]
        .iter()
        .flat_map(|step| step.ops_for(victim))
        .collect();
    assert!(
        resume_ops
            .iter()
            .any(|op| matches!(op, ScheduledOp::PrefillChunk { .. })),
        "a recompute preemption replays the history with prefill chunks"
    );
    assert!(
        resume_ops
            .iter()
            .any(|op| matches!(op, ScheduledOp::Decode { .. }))
    );
    assert_eq!(batcher.request(first).expect("known").generated_tokens, 80);
    assert_eq!(batcher.request(second).expect("known").generated_tokens, 80);
    assert_eq!(batcher.stats().finished, 2);
    assert!(batcher.stats().swap_ins >= 1);
}

#[test]
fn a_swap_preemption_resumes_without_reforwarding_the_history() {
    let mut config = small();
    config.max_num_seqs = 2;
    config.kv_pages = 4;
    config.prefill_chunk_tokens = 4096;
    config.max_num_batched_tokens = 8192;
    config.preemption = PreemptionPolicy::Swap;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let first = batcher.submit(prompt(128, 0), 80).expect("submit");
    let second = batcher.submit(prompt(128, 1000), 80).expect("submit");
    let steps = run(&mut batcher, 400);
    let victim = *steps
        .iter()
        .flat_map(|step| step.preempted.iter())
        .next()
        .expect("a victim");
    let preempt_index = steps
        .iter()
        .position(|step| step.preempted.contains(&victim))
        .expect("a preemption step");
    assert!(
        steps[preempt_index]
            .ops
            .iter()
            .any(|op| matches!(op, ScheduledOp::Preempted { seq, .. } if *seq == victim))
    );
    let resume_index = steps
        .iter()
        .position(|step| step.swapped_in.contains(&victim))
        .expect("the victim resumed");
    let resumed_without_prefill = !steps[resume_index..].iter().any(|step| {
        step.ops_for(victim)
            .iter()
            .any(|op| matches!(op, ScheduledOp::PrefillChunk { .. }))
    });
    assert!(
        resumed_without_prefill,
        "the swap policy resumes from the reattached history"
    );
    let swap_bytes = batcher
        .snapshot()
        .requests
        .iter()
        .find(|request| request.id == victim)
        .expect("known")
        .swap_bytes;
    assert_eq!(swap_bytes, 0, "a resumed request carries no swap ledger");
    assert_eq!(batcher.request(first).expect("known").generated_tokens, 80);
    assert_eq!(batcher.request(second).expect("known").generated_tokens, 80);
}

#[test]
fn a_cached_prompt_block_is_reused_and_a_forward_is_only_needed_for_the_tail() {
    let mut config = small();
    config.prefill_chunk_tokens = 4096;
    config.max_num_batched_tokens = 8192;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let first = batcher.submit(prompt(64, 0), 1).expect("submit");
    run(&mut batcher, 10);
    assert_eq!(
        batcher.request(first).expect("known").phase,
        RequestPhase::Finished
    );

    let forwards_before = batcher.stats().host_forwards;
    let second = batcher.submit(prompt(64, 0), 1).expect("submit");
    let steps = run(&mut batcher, 10);
    let snapshot = batcher.request(second).expect("known");
    assert_eq!(
        snapshot.prefix_hit_tokens, 64,
        "one complete 64-token block"
    );
    assert_eq!(
        batcher.stats().prefix_reused_tokens,
        64,
        "the step reports the reused tokens"
    );
    let prefill_tokens: usize = steps.iter().map(|step| step.prefill_tokens).sum();
    assert_eq!(
        prefill_tokens, 1,
        "only the last prompt token is forwarded for the logits"
    );
    assert!(
        steps
            .iter()
            .flat_map(|step| &step.ops)
            .any(|op| matches!(op, ScheduledOp::PrefixReuse { tokens: 64, .. }))
    );
    assert_eq!(
        batcher.stats().host_forwards - forwards_before,
        1,
        "the cached block costs no forward; only the last token does"
    );
}

#[test]
fn a_different_prompt_misses_the_prefix_cache() {
    let mut config = small();
    config.prefill_chunk_tokens = 4096;
    config.max_num_batched_tokens = 8192;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    batcher.submit(prompt(64, 0), 1).expect("submit");
    run(&mut batcher, 10);
    let second = batcher.submit(prompt(64, 5000), 1).expect("submit");
    run(&mut batcher, 10);
    assert_eq!(batcher.request(second).expect("known").prefix_hit_tokens, 0);
    assert_eq!(batcher.stats().prefix_reused_tokens, 0);
}

#[test]
fn a_reused_page_evicts_the_cached_block_and_the_next_attempt_misses() {
    let mut config = BatchConfig {
        max_num_seqs: 1,
        max_num_batched_tokens: 8192,
        prefill_chunk_tokens: 4096,
        kv_pages: 2,
        kv_tails: 1,
        ..BatchConfig::default()
    };
    config.prefill_chunk_tokens = 4096;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    batcher.submit(prompt(64, 0), 1).expect("submit");
    run(&mut batcher, 10);
    // A different 64-token prompt takes the free page the cached block was waiting on.
    batcher.submit(prompt(64, 7000), 1).expect("submit");
    run(&mut batcher, 10);
    let third = batcher.submit(prompt(64, 0), 1).expect("submit");
    run(&mut batcher, 10);
    assert!(
        batcher.stats().kv_evictions >= 1,
        "the paged cache evicted the cached block: {:?}",
        batcher.stats()
    );
    assert_eq!(
        batcher.request(third).expect("known").prefix_hit_tokens,
        0,
        "the evicted block is a miss"
    );
}

#[test]
fn prefix_reuse_can_be_disabled_for_the_conservative_path() {
    let mut config = small();
    config.prefill_chunk_tokens = 4096;
    config.max_num_batched_tokens = 8192;
    config.prefix_reuse = false;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    batcher.submit(prompt(64, 0), 1).expect("submit");
    run(&mut batcher, 10);
    let second = batcher.submit(prompt(64, 0), 1).expect("submit");
    let steps = run(&mut batcher, 10);
    assert_eq!(
        batcher.request(second).expect("known").prefix_hit_tokens,
        64
    );
    let forwarded: usize = steps.iter().map(|step| step.prefill_tokens).sum();
    assert_eq!(
        forwarded, 64,
        "the compatibility path forwards the whole prompt"
    );
    assert!(
        !steps
            .iter()
            .flat_map(|step| &step.ops)
            .any(|op| matches!(op, ScheduledOp::PrefixReuse { .. }))
    );
}

#[test]
fn submission_and_accept_step_refuse_mismatched_work() {
    let mut batcher = ContinuousBatcher::new(BatchConfig::default()).expect("batcher");
    assert!(batcher.submit(Vec::new(), 1).is_err());
    assert!(batcher.submit(vec![1], 0).is_err());
    let long = (0..=sparkflash_scheduler::config::MAX_CONTEXT_TOKENS)
        .map(|index| index as u32)
        .collect::<Vec<_>>();
    assert!(batcher.submit(long, 1).is_err());

    let id = batcher.submit(vec![1, 2, 3], 2).expect("submit");
    let step = batcher.plan_step().expect("plan");
    let producers = step.ops.iter().filter(|op| op.produces_token()).count();
    assert_eq!(producers, 1);
    assert!(batcher.accept_step(&step, &[]).is_err());
    assert!(batcher.accept_step(&step, &[(id, 1), (id, 2)]).is_err());
    let wrong = RequestId(id.0 + 99);
    assert!(batcher.accept_step(&step, &[(wrong, 1)]).is_err());
    let finished = batcher.accept_step(&step, &[(id, 1)]).expect("accept");
    assert!(finished.is_empty());
}

#[test]
fn a_cancelled_request_frees_its_kv_and_leaves_the_other_alone() {
    let mut config = small();
    config.prefill_chunk_tokens = 2;
    config.max_num_batched_tokens = 4;
    let mut batcher = ContinuousBatcher::new(config).expect("batcher");
    let keep = batcher.submit(prompt(4, 0), 2).expect("submit");
    let drop = batcher.submit(prompt(4, 100), 2).expect("submit");
    let _ = batcher.plan_step().expect("plan");
    batcher.cancel(drop).expect("cancel a live request");
    assert_eq!(
        batcher.request(drop).expect("known").phase,
        RequestPhase::Finished
    );
    run(&mut batcher, 30);
    assert_eq!(batcher.request(keep).expect("known").generated_tokens, 2);
    assert_eq!(
        batcher.stats().finished,
        1,
        "the cancelled request is not a completion"
    );
}
