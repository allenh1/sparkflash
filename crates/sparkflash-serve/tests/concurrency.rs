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

//! The host driver's concurrency knob: N independent sessions really do run at once.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::StreamExt as _;
use sparkflash_api::engine::{EngineEvent, EngineRequest, InferenceEngine};
use sparkflash_model::vision::{ImageGrid, ImageSpan};
use sparkflash_model::{
    BackendError, Capabilities, ModelBackend, ModelDescriptor, ModelSession, PromptCodec, Sampling,
    StepOutput,
};
use sparkflash_serve::{HostDriver, HostOptions};

const MODEL: &str = "toy/concurrency";

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

    fn decode(&self, _ids: &[u32]) -> Option<String> {
        Some(String::new())
    }

    fn token_str(&self, _id: u32) -> Option<String> {
        Some(String::new())
    }
}

fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL.to_owned(),
        family: "toy".to_owned(),
        weights_id: "toy".to_owned(),
        context_tokens: 8,
        max_output_tokens: 1,
        vocab_size: 2,
        capabilities: Capabilities::default(),
    }
}

#[derive(Default)]
struct Probe {
    active: AtomicUsize,
    peak: AtomicUsize,
}

struct BlockingBackend {
    descriptor: ModelDescriptor,
    probe: Arc<Probe>,
}

impl ModelBackend for BlockingBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(BlockingSession {
            probe: Arc::clone(&self.probe),
        }))
    }
}

struct BlockingSession {
    probe: Arc<Probe>,
}

impl ModelSession for BlockingSession {
    fn set_sampling(&mut self, _sampling: Sampling) {}

    fn prefill(&mut self, _tokens: &[u32]) -> Result<StepOutput, BackendError> {
        let active = self.probe.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.probe.peak.fetch_max(active, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(200));
        self.probe.active.fetch_sub(1, Ordering::SeqCst);
        Ok(StepOutput::default())
    }

    fn decode(&mut self, _token: u32) -> Result<StepOutput, BackendError> {
        Ok(StepOutput::default())
    }
}

fn request(id: &str) -> EngineRequest {
    EngineRequest {
        request_id: id.to_owned(),
        model: MODEL.to_owned(),
        prompt_tokens: vec![1],
        logprobs: sparkflash_api::engine::LogprobsRequest::default(),
        images: Vec::new(),
        vision: Vec::new(),
        max_tokens: 1,
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
    }
}

async fn collect(driver: Arc<HostDriver>, request: EngineRequest) -> usize {
    driver.generate(request).collect::<Vec<_>>().await.len()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_requests_overlap_on_two_workers() {
    let probe = Arc::new(Probe::default());
    let driver = Arc::new(
        HostDriver::start(
            Box::new(BlockingBackend {
                descriptor: descriptor(),
                probe: Arc::clone(&probe),
            }),
            Arc::new(Codec),
            HostOptions {
                max_tokens: 1,
                concurrency: 2,
                ..HostOptions::default()
            },
        )
        .expect("the driver starts"),
    );
    let (left, right) = tokio::join!(
        collect(Arc::clone(&driver), request("a")),
        collect(Arc::clone(&driver), request("b")),
    );
    assert!(left >= 2, "the first request emitted its start and finish");
    assert!(
        right >= 2,
        "the second request emitted its start and finish"
    );
    assert_eq!(
        probe.peak.load(Ordering::SeqCst),
        2,
        "both sessions must be inside prefill at the same time"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_default_is_one_session_at_a_time() {
    let probe = Arc::new(Probe::default());
    let driver = Arc::new(
        HostDriver::start(
            Box::new(BlockingBackend {
                descriptor: descriptor(),
                probe: Arc::clone(&probe),
            }),
            Arc::new(Codec),
            HostOptions {
                max_tokens: 1,
                ..HostOptions::default()
            },
        )
        .expect("the driver starts"),
    );
    let (left, right) = tokio::join!(
        collect(Arc::clone(&driver), request("a")),
        collect(Arc::clone(&driver), request("b")),
    );
    assert!(left >= 2 && right >= 2);
    assert_eq!(
        probe.peak.load(Ordering::SeqCst),
        1,
        "the default keeps the serial one-job-at-a-time path"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_prepared_image_is_refused_not_silently_dropped() {
    let driver = HostDriver::start(
        Box::new(BlockingBackend {
            descriptor: descriptor(),
            probe: Arc::new(Probe::default()),
        }),
        Arc::new(Codec),
        HostOptions::default(),
    )
    .expect("the driver starts");
    let mut request = request("image");
    request.vision.push(sparkflash_api::vision::VisionImage {
        span: ImageSpan {
            start: 1,
            grid: ImageGrid {
                n_llm_h: 1,
                n_llm_w: 1,
                best_height: 2,
                best_width: 2,
                line_stride: 1,
            },
            key: [0; 32],
        },
        pixels: Arc::from(vec![0u8; 12]),
    });
    let events = driver.generate(request).collect::<Vec<_>>().await;
    assert!(
        matches!(
            events.first(),
            Some(EngineEvent::Error { code, .. }) if code == sparkflash_serve::host::REFUSED_CODE
        ),
        "a prepared image must be refused, got {events:?}"
    );
}
