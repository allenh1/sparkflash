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
//! next request prefills only what it added, and the hit is reported as such. A prompt that spells
//! the held tokens' bytes with other ids continues them too, and a request whose client left after
//! its prefill keeps its work for the retry. A sequence that holds images continues with a prompt
//! that carries the same images at the same spans.

use std::sync::{Arc, Mutex};

use futures::StreamExt as _;
use sparkflash_api::engine::{EngineEvent, EngineRequest, InferenceEngine};
use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};
use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, PromptCodec, Sampling,
    StepOutput,
};
use sparkflash_serve::{HostDriver, HostOptions};
use tokio::time::{Duration, timeout};

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

/// A codec whose spellings tile like a byte-level vocabulary's: `10`+`11` and `20`+`21` both spell
/// `abc`, the way a tokenizer may spell a word the model wrote with ids other than the model's.
struct Spelling;

impl PromptCodec for Spelling {
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

    fn decode(&self, ids: &[u32]) -> Option<String> {
        ids.iter().map(|id| self.token_str(*id)).collect()
    }

    fn token_str(&self, id: u32) -> Option<String> {
        Some(match id {
            10 => "ab".to_owned(),
            11 => "c".to_owned(),
            20 => "a".to_owned(),
            21 => "bc".to_owned(),
            other => format!("<{other}>"),
        })
    }
}

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL.to_owned(),
        family: "toy-prefix".to_owned(),
        model_type: "toy-prefix".to_owned(),
        label: "ToyPrefix".to_owned(),
        weights_id: "toy-prefix".to_owned(),
        context_tokens: 256,
        max_output_tokens: 8,
        vocab_size: VOCAB,
        capabilities: Capabilities {
            vision: true,
            ..Capabilities::default()
        },
    }
}

/// What the backend saw: one entry per forwarded prefill, the tokens it actually received and the
/// starts of the images it encoded, and how many sessions were alive at once (live now, and the
/// most ever).
#[derive(Default)]
struct Recorder {
    prefills: Mutex<Vec<Vec<u32>>>,
    encoded: Mutex<Vec<Vec<usize>>>,
    sessions: Mutex<(usize, usize)>,
    /// When set, every prefill waits on it; the test opens it by dropping the sender.
    gate: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    /// Sessions keep the end of their last prompt and rewind to it, as the Inkling arm does.
    rewinds: bool,
}

impl Recorder {
    fn prefills(&self) -> Vec<Vec<u32>> {
        self.prefills.lock().expect("the recorder lock").clone()
    }

    fn encoded(&self) -> Vec<Vec<usize>> {
        self.encoded.lock().expect("the recorder lock").clone()
    }

    fn most_alive(&self) -> usize {
        self.sessions.lock().expect("the recorder lock").1
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
        let mut sessions = self.recorder.sessions.lock().expect("the recorder lock");
        sessions.0 += 1;
        sessions.1 = sessions.1.max(sessions.0);
        drop(sessions);
        Ok(Box::new(PrefixSession {
            recorder: Arc::clone(&self.recorder),
            consumed: Vec::new(),
            images: Vec::new(),
            cached: 0,
            boundary: None,
        }))
    }
}

/// A session whose next token is a function of every token it has forwarded, so a continued
/// sequence and a cold one must agree token for token.
struct PrefixSession {
    recorder: Arc<Recorder>,
    consumed: Vec<u32>,
    /// The last prompt's images, which its end (the rewind point) holds too.
    images: Vec<ImageSpan>,
    cached: usize,
    /// The last prompt's end, when the recorder's sessions rewind.
    boundary: Option<usize>,
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

impl Drop for PrefixSession {
    fn drop(&mut self) {
        self.recorder.sessions.lock().expect("the recorder lock").0 -= 1;
    }
}

impl ModelSession for PrefixSession {
    fn set_sampling(&mut self, _sampling: Sampling) {}

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        self.prefill_images(tokens, &[])
    }

    fn prefill_images(
        &mut self,
        tokens: &[u32],
        images: &[VisionImage],
    ) -> Result<StepOutput, BackendError> {
        if let Some(gate) = self
            .recorder
            .gate
            .lock()
            .expect("the recorder lock")
            .as_ref()
        {
            let _ = gate.recv();
        }
        // The driver only hands a kept sequence a prompt that extends it, or one that goes past
        // its rewind point; anything else is a backend contract violation, as it is on the real
        // arm.
        let mut reused = self
            .consumed
            .iter()
            .zip(tokens)
            .take_while(|(held, given)| held == given)
            .count();
        if reused != self.consumed.len()
            && let Some(point) = self.rewind_prefix().map(<[u32]>::len)
            && reused >= point
            && point < tokens.len()
        {
            self.consumed.truncate(point);
            reused = point;
        }
        if reused != self.consumed.len() || reused >= tokens.len() {
            return Err(BackendError::new(
                "toy",
                "a prefill must extend the tokens this sequence holds",
            ));
        }
        // The held tokens' images are the ones this sequence encoded there, so the prompt must
        // carry those and no other.
        let same = self
            .images
            .iter()
            .all(|span| images.iter().any(|image| image.span == *span))
            && images
                .iter()
                .all(|image| image.span.start >= reused || self.images.contains(&image.span));
        if !same {
            return Err(BackendError::new(
                "toy",
                "a prefill must carry the images this sequence holds",
            ));
        }
        self.recorder
            .prefills
            .lock()
            .expect("the recorder lock")
            .push(tokens[reused..].to_vec());
        self.recorder
            .encoded
            .lock()
            .expect("the recorder lock")
            .push(
                images
                    .iter()
                    .map(|image| image.span.start)
                    .filter(|start| *start >= reused)
                    .collect(),
            );
        self.cached = reused;
        self.consumed.extend_from_slice(&tokens[reused..]);
        self.images = images.iter().map(|image| image.span).collect();
        if self.recorder.rewinds {
            self.boundary = Some(self.consumed.len());
        }
        Ok(self.step())
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.consumed.push(token);
        Ok(self.step())
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        Some(self.consumed.as_slice())
    }

    fn rewind_prefix(&self) -> Option<&[u32]> {
        self.boundary
            .filter(|point| *point < self.consumed.len())
            .map(|point| &self.consumed[..point])
    }

    fn held_images(&self) -> &[ImageSpan] {
        &self.images
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
    host_driver_with(recorder, Arc::new(Codec), true)
}

fn host_driver_with(
    recorder: Arc<Recorder>,
    codec: Arc<dyn PromptCodec>,
    reuse_by_text: bool,
) -> HostDriver {
    HostDriver::start(
        Box::new(PrefixBackend {
            descriptor: descriptor(),
            recorder,
        }),
        codec,
        HostOptions {
            max_tokens: 8,
            reuse_by_text,
            ..HostOptions::default()
        },
    )
    .expect("the host driver starts")
}

/// A first turn that writes `10`,`11` (`abc`) after the prompt `5 6 7`, so the sequence holds
/// `5 6 7 10 11`.
fn first_turn() -> EngineRequest {
    let mut first = request("a", vec![5, 6, 7], 2);
    first.forced_tokens = vec![10, 11];
    first
}

/// An image of three positions from `start` (a start delimiter, one row of one, an end) whose
/// canvas is `key`.
fn image(start: usize, key: u8) -> VisionImage {
    VisionImage {
        span: ImageSpan {
            start,
            grid: ImageGrid {
                n_llm_h: 1,
                n_llm_w: 1,
                best_height: 1,
                best_width: 1,
                line_stride: 1,
            },
            key: [key; 32],
        },
        pixels: Arc::from(vec![key; 3]),
    }
}

/// A first turn whose prompt `5 6 7 8` carries image `1` over `6 7 8` and that writes `10`,`11`,
/// so the sequence holds `5 6 7 8 10 11` and the image.
fn image_turn() -> EngineRequest {
    let mut first = request("a", vec![5, 6, 7, 8], 2);
    first.vision = vec![image(1, 1)];
    first.forced_tokens = vec![10, 11];
    first
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
    // The held sequence is gone before the fresh one opens: a real arm's session owns its whole
    // cache, and at a long context two of them do not fit.
    assert_eq!(recorder.most_alive(), 1, "two sequences were alive at once");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prompt_that_spells_the_held_reply_with_other_ids_continues_it() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver_with(Arc::clone(&recorder), Arc::new(Spelling), true);
    answer(&driver, first_turn()).await;

    // The client sent the reply back as text, and it came back as `20`+`21`: the same bytes.
    let respelled = vec![5u32, 6, 7, 20, 21, 9];
    let second = answer(&driver, request("b", respelled.clone(), 2)).await;
    assert_eq!(
        second.cached, 5,
        "the prompt and the two ids that respell the reply"
    );
    assert_eq!(second.usage_cached, 5);
    assert_eq!(second.usage_miss, 1);
    assert_eq!(
        recorder.prefills(),
        vec![vec![5, 6, 7], vec![9]],
        "only the new turn was forwarded"
    );
    assert_eq!(recorder.most_alive(), 1);

    // The model reads its own ids for the reply: the answer is a cold run of the held spelling.
    let cold = host_driver_with(Arc::new(Recorder::default()), Arc::new(Spelling), true);
    let own = answer(&cold, request("cold", vec![5, 6, 7, 10, 11, 9], 2)).await;
    assert_eq!(second.text, own.text);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reuse_by_text_off_prefills_a_respelled_prompt_whole() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver_with(Arc::clone(&recorder), Arc::new(Spelling), false);
    answer(&driver, first_turn()).await;

    let respelled = vec![5u32, 6, 7, 20, 21, 9];
    let second = answer(&driver, request("b", respelled.clone(), 2)).await;
    assert_eq!(second.cached, 0);
    assert_eq!(second.usage_miss, respelled.len());
    assert_eq!(recorder.prefills(), vec![vec![5, 6, 7], respelled]);
    assert_eq!(recorder.most_alive(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prompt_whose_bytes_differ_from_the_held_reply_starts_a_fresh_sequence() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver_with(Arc::clone(&recorder), Arc::new(Spelling), true);
    answer(&driver, first_turn()).await;

    // `20` alone spells `a`, not `abc`, and `9` does not go on with it.
    let other = vec![5u32, 6, 7, 20, 9];
    let second = answer(&driver, request("b", other.clone(), 2)).await;
    assert_eq!(second.cached, 0);
    assert_eq!(recorder.prefills(), vec![vec![5, 6, 7], other]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prompt_that_keeps_the_held_prompt_and_parts_from_its_reply_rewinds() {
    let recorder = Arc::new(Recorder {
        rewinds: true,
        ..Recorder::default()
    });
    let driver = host_driver(Arc::clone(&recorder));
    answer(&driver, first_turn()).await;

    // The client renders the reply with other ids that spell other bytes, then the next turn:
    // the sequence goes back to the end of the first prompt and forwards the rest.
    let next = vec![5u32, 6, 7, 30, 31, 9];
    let second = answer(&driver, request("b", next.clone(), 2)).await;
    assert_eq!(second.cached, 3, "the held prompt");
    assert_eq!(second.usage_cached, 3);
    assert_eq!(second.usage_miss, 3);
    assert_eq!(
        recorder.prefills(),
        vec![vec![5, 6, 7], vec![30, 31, 9]],
        "only the prompt past the rewind point was forwarded"
    );
    assert_eq!(recorder.most_alive(), 1);

    let cold = host_driver(Arc::new(Recorder::default()));
    let own = answer(&cold, request("cold", next, 2)).await;
    assert_eq!(second.text, own.text);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prompt_that_parts_before_the_rewind_point_or_stops_at_it_starts_fresh() {
    let recorder = Arc::new(Recorder {
        rewinds: true,
        ..Recorder::default()
    });
    let driver = host_driver(Arc::clone(&recorder));
    answer(&driver, first_turn()).await;

    // `5 6 8` parts inside the held prompt.
    let parted = answer(&driver, request("b", vec![5, 6, 8, 9], 2)).await;
    assert_eq!(parted.cached, 0);
    answer(&driver, first_turn()).await;
    // `5 6 7` is the held prompt with nothing past it: a rewind would hold no logits for it.
    let stopped = answer(&driver, request("c", vec![5, 6, 7], 2)).await;
    assert_eq!(stopped.cached, 0);
    assert_eq!(
        recorder.prefills(),
        vec![
            vec![5, 6, 7],
            vec![5, 6, 8, 9],
            vec![5, 6, 7],
            vec![5, 6, 7]
        ],
        "each was prefilled whole"
    );
    assert_eq!(recorder.most_alive(), 1);
}

/// Starts `prompt` on a driver whose prefill waits, reads the request's start, leaves, and lets
/// the prefill finish.
async fn abandon_after_prefill(driver: &HostDriver, recorder: &Recorder, prompt: Vec<u32>) {
    let (open, gate) = std::sync::mpsc::channel();
    *recorder.gate.lock().expect("the recorder lock") = Some(gate);
    let mut events = driver.generate(request("left", prompt, 4));
    let started = timeout(Duration::from_secs(10), events.next())
        .await
        .expect("the request starts");
    assert!(
        matches!(started, Some(EngineEvent::Started { .. })),
        "the first event is the start"
    );
    drop(events);
    drop(open);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_retry_of_a_request_whose_client_left_answers_from_its_prefill() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver(Arc::clone(&recorder));
    let prompt = vec![5u32, 6, 7];
    abandon_after_prefill(&driver, &recorder, prompt.clone()).await;

    let retry = answer(&driver, request("retry", prompt.clone(), 4)).await;
    assert_eq!(retry.cached, prompt.len(), "the whole prompt is held");
    assert_eq!(retry.usage_cached, prompt.len());
    assert_eq!(retry.usage_miss, 0);
    assert_eq!(
        recorder.prefills(),
        vec![prompt.clone()],
        "the retry forwarded nothing"
    );
    assert_eq!(recorder.most_alive(), 1);

    let cold = host_driver(Arc::new(Recorder::default()));
    let own = answer(&cold, request("cold", prompt, 4)).await;
    assert_eq!(retry.text, own.text);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retry_with_another_sampler_prefills_again() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver(Arc::clone(&recorder));
    let prompt = vec![5u32, 6, 7];
    abandon_after_prefill(&driver, &recorder, prompt.clone()).await;

    // The held step was drawn under the first request's sampler, so it cannot answer this one.
    let mut retry = request("retry", prompt.clone(), 4);
    retry.seed = Some(7);
    let retry = answer(&driver, retry).await;
    assert_eq!(retry.cached, 0);
    assert_eq!(recorder.prefills(), vec![prompt.clone(), prompt]);
    assert_eq!(recorder.most_alive(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_that_carries_the_held_image_again_prefills_only_what_it_added() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver(Arc::clone(&recorder));
    answer(&driver, image_turn()).await;

    // An agent resends the conversation with its screenshot, then a new screenshot and a turn.
    let next = vec![5u32, 6, 7, 8, 10, 11, 4, 3, 2, 9];
    let images = vec![image(1, 1), image(6, 2)];
    let mut second = request("b", next.clone(), 2);
    second.vision.clone_from(&images);
    let second = answer(&driver, second).await;
    assert_eq!(second.cached, 6, "the prompt, its image and the reply");
    assert_eq!(second.usage_cached, 6);
    assert_eq!(second.usage_miss, 4);
    assert_eq!(
        recorder.prefills(),
        vec![vec![5, 6, 7, 8], vec![4, 3, 2, 9]],
        "only the new image and turn were forwarded"
    );
    assert_eq!(
        recorder.encoded(),
        vec![vec![1], vec![6]],
        "each image was encoded once"
    );
    assert_eq!(recorder.most_alive(), 1);

    let cold = host_driver(Arc::new(Recorder::default()));
    let mut own = request("cold", next, 2);
    own.vision = images;
    assert_eq!(answer(&cold, own).await.text, second.text);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_whose_held_image_differs_or_is_gone_starts_a_fresh_sequence() {
    let recorder = Arc::new(Recorder::default());
    let driver = host_driver(Arc::clone(&recorder));
    answer(&driver, image_turn()).await;

    // Another canvas over the same placeholders: the ids match, the image does not.
    let next = vec![5u32, 6, 7, 8, 10, 11, 9];
    let mut other = request("b", next.clone(), 2);
    other.vision = vec![image(1, 3)];
    assert_eq!(answer(&driver, other).await.cached, 0);
    answer(&driver, image_turn()).await;
    // The same placeholders with no image at all.
    assert_eq!(
        answer(&driver, request("c", next.clone(), 2)).await.cached,
        0
    );
    assert_eq!(
        recorder.prefills(),
        vec![vec![5, 6, 7, 8], next.clone(), vec![5, 6, 7, 8], next],
        "each was prefilled whole"
    );
    assert_eq!(recorder.most_alive(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_that_parts_from_the_reply_rewinds_past_the_held_image() {
    let recorder = Arc::new(Recorder {
        rewinds: true,
        ..Recorder::default()
    });
    let driver = host_driver(Arc::clone(&recorder));
    answer(&driver, image_turn()).await;

    let next = vec![5u32, 6, 7, 8, 30, 31, 9];
    let mut second = request("b", next.clone(), 2);
    second.vision = vec![image(1, 1)];
    let second = answer(&driver, second).await;
    assert_eq!(second.cached, 4, "the held prompt and its image");
    assert_eq!(second.usage_miss, 3);

    // Another canvas before the rewind point cannot go back there.
    answer(&driver, image_turn()).await;
    let mut other = request("c", next.clone(), 2);
    other.vision = vec![image(1, 3)];
    assert_eq!(answer(&driver, other).await.cached, 0);
    assert_eq!(
        recorder.prefills(),
        vec![vec![5, 6, 7, 8], vec![30, 31, 9], vec![5, 6, 7, 8], next],
        "only the rewound turn forwarded part of its prompt"
    );
    assert_eq!(recorder.most_alive(), 1);
}

/// Two workers keep their sequences in common: each conversation's next turn continues its own,
/// whichever worker takes it, and a third conversation takes the least recently used one's room.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_conversations_each_find_their_own_prefix() {
    let recorder = Arc::new(Recorder::default());
    let driver = HostDriver::start(
        Box::new(PrefixBackend {
            descriptor: descriptor(),
            recorder: Arc::clone(&recorder),
        }),
        Arc::new(Codec),
        HostOptions {
            max_tokens: 8,
            concurrency: 2,
            ..HostOptions::default()
        },
    )
    .expect("the host driver starts");
    let turn = |prompt: &[u32], reply: &str, next: u32| {
        let mut follow = prompt.to_vec();
        follow.extend_from_slice(&emitted(reply));
        follow.push(next);
        follow
    };
    let a1 = vec![5u32, 6, 7];
    let b1 = vec![8u32, 9];
    let a = answer(&driver, request("a1", a1.clone(), 2)).await;
    let b = answer(&driver, request("b1", b1.clone(), 2)).await;
    // The turns alternate, so a worker that kept only its own last sequence would miss half.
    for (id, prompt, reply, next) in [("a2", &a1, &a.text, 3u32), ("b2", &b1, &b.text, 4)] {
        let follow = turn(prompt, reply, next);
        let held = follow.len() - 1;
        let answered = answer(&driver, request(id, follow, 2)).await;
        assert_eq!(answered.cached, held, "{id} continues its own conversation");
    }
    assert_eq!(recorder.prefills()[2..], [vec![3], vec![4]]);
    // A third conversation: two sequences are kept, so it takes the least recently used one's
    // room (a's) and the pool never holds more than two.
    let c = answer(&driver, request("c1", vec![11, 12], 2)).await;
    assert_eq!(c.cached, 0);
    assert!(recorder.most_alive() <= 2, "{} alive", recorder.most_alive());
    let b3 = answer(&driver, request("b3", b1.clone(), 1)).await;
    assert_eq!(b3.cached, 0, "a prompt that only repeats b's first turn parts from its reply");
}
