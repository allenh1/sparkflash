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

//! The host driver's prefix cache: a kept sequence continues from the tokens it holds, so the
//! next request prefills only what it added, and the hit is reported as such.

use std::sync::{Arc, Mutex};

use futures::StreamExt as _;
use sparkflash_api::engine::{EngineEvent, EngineRequest, InferenceEngine};
use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, PromptCodec, Sampling,
    StepOutput,
};
use sparkflash_serve::{HostDriver, HostOptions};

const MODEL: &str = "toy/prefix";
const VOCAB: usize = 64;

struct Codec;

impl PromptCodec for Codec {
    fn encode_chat(
        &self,
        _request: &serde_json::Value,
    ) -> Result<Vec<u32>, sparkflash_model::PromptError> {
        Ok(vec![1])
    }

    fn encode_text(
        &self,
        _text: &str,
        _add_special_tokens: bool,
    ) -> Result<Vec<u32>, sparkflash_model::PromptError> {
        Ok(vec![1])
    }

    /// The generated ids, dot separated, so a test can read the tokens back out of the stream.
    fn decode(&self, ids: &[u32]) -> Option<String> {
        Some(ids.iter().map(u32::to_string).collect::<Vec<_>>().join("."))
    }

    fn token_str(&self, id: u32) -> Option<String> {
        self.decode(&[id])
    }
}

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL.to_owned(),
        family: "toy-prefix".to_owned(),
        weights_id: "toy-prefix".to_owned(),
        context_tokens: 256,
        max_output_tokens: 8,
        vocab_size: VOCAB,
        capabilities: Capabilities::default(),
    }
}

/// What the backend saw: one entry per forwarded prefill, the tokens it actually received.
#[derive(Default)]
struct Recorder {
    prefills: Mutex<Vec<Vec<u32>>>,
}

impl Recorder {
    fn prefills(&self) -> Vec<Vec<u32>> {
        self.prefills.lock().expect("the recorder lock").clone()
    }
}

struct PrefixBackend {
    descriptor: ModelDescriptor,
    recorder: Arc<Recorder>,
}

impl ModelBackend for PrefixBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(PrefixSession {
            recorder: Arc::clone(&self.recorder),
            consumed: Vec::new(),
            cached: 0,
        }))
    }
}

/// A session whose next token is a function of every token it has forwarded, so a continued
/// sequence and a cold one must agree token for token.
struct PrefixSession {
    recorder: Arc<Recorder>,
    consumed: Vec<u32>,
    cached: usize,
}

impl PrefixSession {
    fn step(&self) -> StepOutput {
        let next = if self.consumed.is_empty() {
            None
        } else {
            let sum: u64 = self.consumed.iter().map(|token| u64::from(*token)).sum();
            Some(u32::try_from(sum % VOCAB as u64).unwrap_or(0))
        };
        let mut logits = vec![f32::NEG_INFINITY; VOCAB];
        if let Some(token) = next {
            logits[token as usize] = 0.0;
        }
        StepOutput {
            logits,
            next_token: next,
            draft: Vec::new(),
        }
    }
}

impl ModelSession for PrefixSession {
    fn set_sampling(&mut self, _sampling: Sampling) {}

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        // The driver only hands a kept sequence a prompt that extends it; anything else is a
        // backend contract violation, as it is on the real arm.
        let reused = self
            .consumed
            .iter()
            .zip(tokens)
            .take_while(|(held, given)| held == given)
            .count();
        if reused != self.consumed.len() || reused >= tokens.len() {
            return Err(BackendError::new(
                "toy",
                "a prefill must extend the tokens this sequence holds",
            ));
        }
        self.recorder
            .prefills
            .lock()
            .expect("the recorder lock")
            .push(tokens[reused..].to_vec());
        self.cached = reused;
        self.consumed.extend_from_slice(&tokens[reused..]);
        Ok(self.step())
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.consumed.push(token);
        Ok(self.step())
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        Some(self.consumed.as_slice())
    }

    fn cached_tokens(&self) -> usize {
        self.cached
    }
}

fn request(id: &str, prompt: Vec<u32>, max_tokens: usize) -> EngineRequest {
    EngineRequest {
        request_id: id.to_owned(),
        model: MODEL.to_owned(),
        prompt_tokens: prompt,
        logprobs: sparkflash_api::engine::LogprobsRequest::default(),
        images: Vec::new(),
        vision: Vec::new(),
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        skip_special_tokens: true,
        thinking: false,
        reasoning_effort: None,
        stop: Vec::new(),
        seed: None,
        trace_scopes: 0,
        trace_steps: 0,
        constraint: None,
        forced_tokens: Vec::new(),
    }
}

struct Answer {
    cached: usize,
    usage_cached: usize,
    usage_miss: usize,
    text: String,
}

async fn answer(driver: &HostDriver, request: EngineRequest) -> Answer {
    let mut events = driver.generate(request);
    let (mut cached, mut usage_cached, mut usage_miss) = (None, None, None);
    let mut text = String::new();
    while let Some(event) = events.next().await {
        match event {
            EngineEvent::Started { cached_tokens, .. } => cached = Some(cached_tokens),
            EngineEvent::ContentDelta(delta) => text.push_str(&delta),
            EngineEvent::Finished { usage, .. } => {
                usage_cached = Some(usage.prompt_tokens_details.cached_tokens);
                usage_miss = Some(usage.prompt_cache_miss_tokens);
            }
            _ => {}
        }
    }
    Answer {
        cached: cached.expect("every request starts"),
        usage_cached: usage_cached.expect("every request finishes"),
        usage_miss: usage_miss.expect("every request finishes"),
        text,
    }
}

fn host_driver(recorder: Arc<Recorder>) -> HostDriver {
    HostDriver::start(
        Box::new(PrefixBackend {
            descriptor: descriptor(),
            recorder,
        }),
        Arc::new(Codec),
        HostOptions {
            max_tokens: 8,
            ..HostOptions::default()
        },
    )
    .expect("the host driver starts")
}

/// The ids a request emitted, from the dot separated text the toy codec produces.
fn emitted(text: &str) -> Vec<u32> {
    text.split('.')
        .filter(|piece| !piece.is_empty())
        .filter_map(|piece| piece.parse().ok())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chat_turn_prefills_only_the_text_it_added() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver(Arc::clone(&recorder));
    let prompt = vec![5u32, 6, 7];
    let first = answer(&driver, request("a", prompt.clone(), 4)).await;
    assert_eq!(first.cached, 0);
    assert_eq!(first.usage_cached, 0);
    assert_eq!(first.usage_miss, prompt.len());
    let reply = emitted(&first.text);
    assert_eq!(reply.len(), 4, "the first request generated four tokens");

    // A chat client resends the prompt and the reply, then the next turn. The sequence decoded
    // every emitted token, so it already holds the whole conversation and only the new turn is
    // forwarded.
    let mut follow = prompt.clone();
    follow.extend_from_slice(&reply);
    follow.push(9);
    let mut next_prompt = follow.clone();
    let held = prompt.len() + reply.len();
    let second = answer(&driver, request("b", next_prompt.clone(), 2)).await;
    assert_eq!(second.cached, held, "the prompt and the whole reply");
    assert_eq!(second.usage_cached, held);
    assert_eq!(second.usage_miss, next_prompt.len() - held);

    assert_eq!(
        recorder.prefills(),
        vec![prompt.clone(), next_prompt.split_off(held)],
        "the second request forwarded only the tokens past the held prefix"
    );

    // The continued sequence answers exactly as a cold one handed the whole prompt does.
    let cold_recorder = Arc::new(Recorder::default());
    let cold = host_driver(Arc::clone(&cold_recorder));
    let cold_answer = answer(&cold, request("cold", follow, 2)).await;
    assert_eq!(cold_answer.cached, 0);
    assert_eq!(cold_answer.text, second.text);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prompt_that_does_not_extend_the_held_tokens_starts_a_fresh_sequence() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver(Arc::clone(&recorder));
    let first = answer(&driver, request("a", vec![5, 6, 7], 2)).await;
    assert_eq!(first.cached, 0);

    // Same length, different content: the held sequence cannot serve it.
    let second = answer(&driver, request("b", vec![5, 6, 8, 9], 2)).await;
    assert_eq!(second.cached, 0);
    assert_eq!(second.usage_miss, 4);
    assert_eq!(
        recorder.prefills(),
        vec![vec![5, 6, 7], vec![5, 6, 8, 9]],
        "a divergent prompt is prefilled whole"
    );
}
