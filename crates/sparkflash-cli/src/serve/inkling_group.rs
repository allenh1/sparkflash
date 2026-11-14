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

//! The Inkling-2-Small device arm as a four-rank server.
//!
//! This is the serve face of [sparkflash_model_inkling2::Tp4Run]: the batch walk takes a prompt
//! and a decode count and exits, while a server holds the group open, prefills once, and streams
//! one token at a time with the KV untouched. The arithmetic is the same code either way - the
//! device arm owns the schedule, the kernels and the three collectives - and this module owns
//! only the rank agreement a server adds.
//!
//! # The shape of a request
//!
//! Rank 0 owns HTTP, the tokenizer and the control channel. Every rank, rank 0 included, holds
//! one [InklingDeviceBackend] over its own converted rank file and mirrors every step; the
//! device arm's collectives are what force that. A request therefore runs as:
//!
//! 1. rank 0 broadcasts `Request` with the tokenizer's prompt ids, then prefills its own rank;
//! 2. rank 0 samples a token (the driver's sampler), broadcasts `Token`, then decodes it;
//! 3. the workers receive the same frames in the same order and run the same two steps, so
//!    every rank's sequence state advances identically and the collectives line up;
//! 4. when the request ends, rank 0 keeps its session (the host driver reuses a shared prefix)
//!    and the workers keep theirs; a session *dropped* before its next prefill broadcasts `End`,
//!    after which every rank discards its state and the next `Request` starts clean.
//!
//! # What is not claimed
//!
//! The group serves one request at a time: a second worker would interleave two token sequences
//! across four ranks that are in lockstep, so the parse refuses it. There is no chunked prefill
//! (a prompt is forwarded one position at a time), no drafter, no image path, and a worker that
//! fails ends its process rather than reporting through the control channel - rank 0 then
//! stalls on the next collective until the transport's frame deadline. All of that is stated in
//! docs/models/inkling2-small-tp4.md rather than discovered in production.

use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sparkflash_comm::tcp::TcpConfig;
use sparkflash_model::backend::{BackendError, ModelBackend, ModelSession, Sampling, StepOutput};
use sparkflash_model::{ModelDescriptor, PromptCodec, TpPlan};
use sparkflash_model_inkling2::collective::TcpExchange;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::convert::RankFileSource;
use sparkflash_model_inkling2::device::InklingDeviceBackend;
use sparkflash_weights::tp::TpLayout;

use super::dynamic::{Dialect, DynamicCheckpoint, serve_backend_world};
use super::tp4_control::{Tp4ControlFrame, Tp4ControlHub, Tp4ControlWorker, control_read_is_idle};
use super::{GroupOptions, ServeOptions, TP4_CONTROL_TIMEOUT};

/// How long a collective frame may wait for a peer.
///
/// The device arm's own batch run uses the same deadline: a 594B-parameter rank's step is slow,
/// and the deadline is a liveness guard for a dead peer, not a per-token budget.
const FRAME_TIMEOUT: Duration = Duration::from_secs(900);

/// How long a rank may keep dialing a peer that has not bound its listener yet.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(180);

/// Serves one rank of the Inkling group, or mirrors it when this process is not rank 0.
///
/// # Errors
/// A rank file that does not open, a transport that does not connect, a worker that stops, or a
/// device failure.
pub(super) fn serve(
    options: &ServeOptions,
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
) -> Result<ExitCode, String> {
    let group = options
        .group
        .as_ref()
        .ok_or_else(|| "the device-arm group is dispatched only for --rank".to_owned())?;
    let config = Inkling2Config::from_checkpoint_dir(&checkpoint.directory)
        .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
    let converted = group.rank_file()?;
    let layout = TpLayout {
        rank: group.rank,
        world_size: group.world,
    };
    let transport = TcpConfig::with_addresses(group.rank, group.peers.clone())
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?
        .with_timeout(FRAME_TIMEOUT)
        .with_connect_timeout(CONNECT_TIMEOUT);
    // The collective forms first on every rank: a worker that dialed the control channel before
    // its peers bound their collectors would wait for a Ready that rank 0 cannot send until the
    // collective is whole, and the two waits would meet at the startup deadline.
    let exchange = TcpExchange::connect(&transport)
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    if group.rank != 0 {
        return mirror_rank(group, &descriptor, &config, context, exchange);
    }
    let hub = Tp4ControlHub::start(group.control, group.world, TP4_CONTROL_TIMEOUT)
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let tp = TpPlan::new(group.world, group.rank)
        .map_err(|error| format!("TP{}/rank{}: {}", group.world, group.rank, error.detail))?;
    let source = RankFileSource::open(&converted, &config, layout)
        .map_err(|error| format!("{}: {error}", converted.display()))?;
    let opened = std::time::Instant::now();
    let device = InklingDeviceBackend::open_rank(
        descriptor.clone(),
        config,
        Arc::new(source),
        context,
        tp,
        Box::new(exchange),
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let banner = format!(
        "TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms), context {context}, Inkling device arm",
        group.world,
        group.rank,
        descriptor.id,
        converted.display(),
        device.weight_bytes(),
        opened.elapsed().as_millis()
    );
    let backend = GroupBackend {
        descriptor: descriptor.clone(),
        device,
        hub: Arc::new(Mutex::new(hub)),
        context,
    };
    serve_backend_world(
        Box::new(backend),
        descriptor,
        seam,
        dialect,
        options,
        &banner,
        group.world,
    )
}

/// Opens one worker rank's converted file and mirrors rank 0 until the control channel ends.
fn mirror_rank(
    group: &GroupOptions,
    descriptor: &ModelDescriptor,
    config: &Inkling2Config,
    context: usize,
    exchange: TcpExchange,
) -> Result<ExitCode, String> {
    let converted = group.rank_file()?;
    let layout = TpLayout {
        rank: group.rank,
        world_size: group.world,
    };
    let source = RankFileSource::open(&converted, config, layout)
        .map_err(|error| format!("{}: {error}", converted.display()))?;
    // The control connection is a mirror of rank 0's *admissions*; the lockstep arithmetic is
    // the model collective above it. Connecting after the collective is what makes the two
    // waits on rank 0 (the cluster, then the ranks) never cross.
    let mut control = Tp4ControlWorker::connect(
        group.control,
        u32::try_from(group.rank).unwrap_or(u32::MAX),
        group.world,
        TP4_CONTROL_TIMEOUT,
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    control
        .handshake()
        .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    let tp = TpPlan::new(group.world, group.rank)
        .map_err(|error| format!("TP{}/rank{}: {}", group.world, group.rank, error.detail))?;
    let opened = std::time::Instant::now();
    let backend = InklingDeviceBackend::open_rank(
        descriptor.clone(),
        config.clone(),
        Arc::new(source),
        context,
        tp,
        Box::new(exchange),
    )
    .map_err(|error| format!("TP{}/rank{}: {error}", group.world, group.rank))?;
    println!(
        "sparkflash serve: TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms), context {context}, Inkling device arm worker",
        group.world,
        group.rank,
        descriptor.id,
        converted.display(),
        backend.weight_bytes(),
        opened.elapsed().as_millis()
    );
    mirror(&mut control, &backend)?;
    Ok(ExitCode::SUCCESS)
}

/// Mirrors rank 0's request frames onto this rank's own session.
///
/// The four ranks must run the same steps in the same order, so the loop is the whole protocol:
/// a `Request` prefills, a `Token` decodes the token rank 0 sampled, and an `End` discards the
/// sequence because rank 0 has dropped its own.
fn mirror(control: &mut Tp4ControlWorker, backend: &dyn ModelBackend) -> Result<(), String> {
    let mut session: Option<Box<dyn ModelSession>> = None;
    loop {
        let frame = match control.next_frame() {
            Ok(frame) => frame,
            Err(error) if control_read_is_idle(&error) => continue,
            Err(error) => return Err(error.to_string()),
        };
        match frame {
            Tp4ControlFrame::Request { prompt_ids, .. } => {
                let held = match session.as_mut() {
                    Some(held) => held,
                    None => session.insert(
                        backend
                            .open_session()
                            .map_err(|error| format!("open a sequence: {error}"))?,
                    ),
                };
                held.set_sampling(greedy());
                if trace_on() {
                    eprintln!("[inkling] serve worker prefill {:?}", prompt_ids);
                }
                held.prefill(&prompt_ids)
                    .map_err(|error| format!("prefill: {error}"))?;
            }
            Tp4ControlFrame::Token { token_id } => {
                let held = session.as_mut().ok_or_else(|| {
                    "a Token frame arrived before a Request: the ranks are out of step".to_owned()
                })?;
                if trace_on() {
                    eprintln!("[inkling] serve worker token {token_id}");
                }
                held.decode(token_id)
                    .map_err(|error| format!("decode {token_id}: {error}"))?;
            }
            Tp4ControlFrame::End => {
                // Rank 0 dropped its sequence; this rank must not keep a history the next
                // Request could continue from, or a follow-up prompt would desync the group.
                session = None;
            }
            Tp4ControlFrame::Error { code, message } => {
                return Err(format!("rank 0 ended the group ({code}): {message}"));
            }
            other => {
                return Err(format!(
                    "the worker expected Request, Token, End or Error, not {}",
                    other.kind()
                ));
            }
        }
    }
}

/// The rank-0 backend: one device rank with the control hub every session broadcasts through.
struct GroupBackend {
    descriptor: ModelDescriptor,
    device: InklingDeviceBackend,
    hub: Arc<Mutex<Tp4ControlHub>>,
    /// The admitted context every rank allocated, which a prompt may not exceed.
    context: usize,
}

impl ModelBackend for GroupBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> sparkflash_model::Capabilities {
        self.device.capabilities()
    }

    fn fabric_mode(&self) -> Option<&'static str> {
        Some("tcp tp4 device")
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        Ok(Box::new(GroupSession {
            inner: self.device.open_session()?,
            hub: Arc::clone(&self.hub),
            context: self.context,
            open: false,
            fed: Vec::new(),
        }))
    }
}

/// One request sequence on rank 0, mirrored to the group before every step.
///
/// The broadcast precedes the local step, never follows it: the collectives inside that step
/// block until every rank joins, and a rank that only learned of the step afterwards would never
/// arrive.
struct GroupSession {
    inner: Box<dyn ModelSession>,
    hub: Arc<Mutex<Tp4ControlHub>>,
    /// The admitted context, so an over-long prompt is refused *before* it is broadcast.
    context: usize,
    /// Whether a Request is in flight and not yet Ended; a held session says so between requests.
    open: bool,
    /// The tokens this request fed, for the trace line a gate compares with the batch arm.
    fed: Vec<u32>,
}

impl GroupSession {
    /// Broadcasts one frame to the group.
    fn send(&self, frame: &Tp4ControlFrame) -> Result<(), BackendError> {
        let kind = frame.kind();
        let mut hub = self
            .hub
            .lock()
            .map_err(|_| BackendError::new("group", "the control hub lock is poisoned"))?;
        hub.broadcast(frame)
            .map_err(|error| BackendError::new("group", format!("{kind}: {error}")))
    }
}

impl ModelSession for GroupSession {
    fn set_sampling(&mut self, sampling: Sampling) {
        self.inner.set_sampling(sampling);
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        // Refuse before the broadcast: a prompt the group cannot hold must not start a sequence
        // the workers would then have to unwind from inside a collective.
        if tokens.len() > self.context {
            return Err(BackendError::new(
                "context",
                format!(
                    "the prompt needs {} positions, {} remain",
                    tokens.len(),
                    self.context
                ),
            ));
        }
        if trace_on() {
            eprintln!("[inkling] serve rank 0 prefill {:?}", tokens);
        }
        self.fed.clear();
        self.send(&Tp4ControlFrame::Request {
            prompt_ids: tokens.to_vec(),
            max_tokens: u32::MAX,
            prompt_logprobs: None,
            speculative: false,
            cached_tokens: 0,
        })?;
        self.open = true;
        self.inner.prefill(tokens)
    }

    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.send(&Tp4ControlFrame::Token { token_id: token })?;
        self.fed.push(token);
        if trace_on() {
            eprintln!("[inkling] serve rank 0 token {token}");
        }
        self.inner.decode(token)
    }

    fn reusable_prefix(&self) -> Option<&[u32]> {
        self.inner.reusable_prefix()
    }

    fn cached_tokens(&self) -> usize {
        self.inner.cached_tokens()
    }
}

impl Drop for GroupSession {
    fn drop(&mut self) {
        // A dropped sequence is the one case the workers cannot infer: a *finished* request keeps
        // its state for the driver's prefix reuse, so silence is the signal to wait and End is the
        // signal to forget.
        if !self.open {
            return;
        }
        if let Err(error) = self.send(&Tp4ControlFrame::End) {
            eprintln!("sparkflash serve: rank 0 could not end the group: {error}");
        }
        if trace_on() {
            eprintln!("[inkling] serve rank 0 fed {:?}", self.fed);
        }
    }
}

/// The greedy sampler every mirror rank uses; it never chooses a token the group follows, but a
/// session still wants a deterministic configuration before its first step.
fn greedy() -> Sampling {
    sparkflash_model_inkling2::Tp4Run::greedy()
}

/// Whether the arm's trace is on.
fn trace_on() -> bool {
    std::env::var_os("SPARKFLASH_INKLING_TRACE").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_reads_one_rank_file_per_rank() {
        let e = GroupOptions::from_cli(
            Some(4),
            Some(4),
            Some("127.0.0.1:29600,127.0.0.1:29601,127.0.0.1:29602,127.0.0.1:29603"),
            None,
            Some(std::path::PathBuf::from("/converted")),
        )
        .expect_err("rank 4 is outside the world");
        assert!(e.contains("outside a world of 4"), "{e}");

        let e = GroupOptions::from_cli(
            Some(0),
            Some(4),
            Some("127.0.0.1:29600,127.0.0.1:29601"),
            None,
            Some(std::path::PathBuf::from("/converted")),
        )
        .expect_err("two addresses for a world of four");
        assert!(
            e.contains("2 address(es)") && e.contains("world is 4"),
            "{e}"
        );

        let group = GroupOptions::from_cli(
            Some(0),
            None,
            Some("127.0.0.1:29600,127.0.0.1:29601"),
            None,
            Some(std::path::PathBuf::from("/converted")),
        )
        .expect("the peer list names the world")
        .expect("a group");
        assert_eq!(group.world, 2);
        assert_eq!(
            group.control,
            "127.0.0.1:29700".parse().expect("an address")
        );
        let directory = tempfile::tempdir().expect("a temporary converted directory");
        std::fs::write(directory.path().join("rank-0-of-2.bin"), b"").expect("a rank file");
        let group = GroupOptions::from_cli(
            Some(0),
            None,
            Some("127.0.0.1:29600,127.0.0.1:29601"),
            None,
            Some(directory.path().to_path_buf()),
        )
        .expect("the peer list names the world")
        .expect("a group");
        assert_eq!(
            group
                .rank_file()
                .expect("a rank file")
                .file_name()
                .and_then(std::ffi::OsStr::to_str),
            Some("rank-0-of-2.bin")
        );
    }

    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    /// What the fake mirror recorded, in the order it recorded it.
    #[derive(Default)]
    struct Record {
        prefills: Vec<Vec<u32>>,
        decodes: Vec<u32>,
        sessions: usize,
    }

    struct FakeBackend {
        record: Arc<Mutex<Record>>,
        descriptor: ModelDescriptor,
    }

    struct FakeSession {
        record: Arc<Mutex<Record>>,
        history: Vec<u32>,
    }

    impl ModelBackend for FakeBackend {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
            self.record.lock().expect("the record").sessions += 1;
            Ok(Box::new(FakeSession {
                record: Arc::clone(&self.record),
                history: Vec::new(),
            }))
        }
    }

    impl ModelSession for FakeSession {
        fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
            self.record
                .lock()
                .expect("the record")
                .prefills
                .push(tokens.to_vec());
            self.history.extend_from_slice(tokens);
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(1),
                draft: Vec::new(),
            })
        }

        fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
            self.record.lock().expect("the record").decodes.push(token);
            self.history.push(token);
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(2),
                draft: Vec::new(),
            })
        }

        fn reusable_prefix(&self) -> Option<&[u32]> {
            Some(&self.history)
        }
    }

    /// Waits for the mirror to reach a state; the control channel is one-way for rank 0, so a
    /// test observes the worker rather than reading an acknowledgement.
    fn wait_for(mut ready: impl FnMut() -> bool, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if ready() {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("the mirror did not {what} within five seconds");
    }

    /// Rank 0's own local session in these tests: it answers, and it is deliberately not the
    /// mirror under test, so a recorded decode is one the worker ran.
    struct SilentSession;

    impl ModelSession for SilentSession {
        fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
            let _ = tokens;
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(1),
                draft: Vec::new(),
            })
        }

        fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
            let _ = token;
            Ok(StepOutput {
                logits: vec![0.0; 8],
                next_token: Some(2),
                draft: Vec::new(),
            })
        }

        fn reusable_prefix(&self) -> Option<&[u32]> {
            None
        }
    }

    fn fake(record: &Arc<Mutex<Record>>) -> Arc<FakeBackend> {
        Arc::new(FakeBackend {
            record: Arc::clone(record),
            descriptor: sparkflash_model_inkling2::descriptor(),
        })
    }

    #[test]
    fn the_mirror_follows_a_request_and_discards_it_on_end() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let record = Arc::new(Mutex::new(Record::default()));
        let backend = fake(&record);
        let worker = {
            let backend = Arc::clone(&backend);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                let _ = mirror(&mut control, backend.as_ref());
            })
        };
        let hub = Arc::new(Mutex::new(
            Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
                .expect("the hub"),
        ));
        {
            let mut session = GroupSession {
                inner: Box::new(SilentSession),
                hub: Arc::clone(&hub),
                context: 4096,
                open: false,
                fed: Vec::new(),
            };
            session.prefill(&[1, 2, 3]).expect("the prefill");
            session.decode(4).expect("the first decode");
            session.decode(5).expect("the second decode");
            wait_for(
                || record.lock().expect("the record").decodes == vec![4, 5],
                "mirror both decodes",
            );
        }
        {
            let mut session = GroupSession {
                inner: Box::new(SilentSession),
                hub: Arc::clone(&hub),
                context: 4096,
                open: false,
                fed: Vec::new(),
            };
            session.prefill(&[9]).expect("the second prefill");
            wait_for(
                || record.lock().expect("the record").prefills == vec![vec![1, 2, 3], vec![9]],
                "mirror the second prefill",
            );
            wait_for(
                || record.lock().expect("the record").sessions == 2,
                "discard the ended sequence and open a fresh one",
            );
        }
        drop(hub);
        worker.join().expect("the worker finishes");
        let record = record.lock().expect("the record");
        assert_eq!(
            record.decodes,
            vec![4, 5],
            "End must not be mirrored as a decode"
        );
    }

    #[test]
    fn a_token_before_a_request_is_refused() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
        let address = listener.local_addr().expect("the bound address");
        let record = Arc::new(Mutex::new(Record::default()));
        let backend = fake(&record);
        let worker = {
            let backend = Arc::clone(&backend);
            thread::spawn(move || {
                let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
                    .expect("a control connection");
                control.handshake().expect("the handshake");
                mirror(&mut control, backend.as_ref())
            })
        };
        let mut hub = Tp4ControlHub::start_with_listener(listener, 2, Duration::from_secs(5))
            .expect("the hub");
        hub.broadcast(&Tp4ControlFrame::Token { token_id: 7 })
            .expect("the frame is written");
        let outcome = worker.join().expect("the worker finishes");
        let error = outcome.expect_err("a token with no request is refused");
        assert!(error.contains("before a Request"), "{error}");
    }

    #[test]
    fn the_group_flags_parse_and_refuse_the_knobs_they_cannot_honour() {
        let directory = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(
            directory.path().join(super::super::CHECKPOINT_CONFIG_NAME),
            sparkflash_model_inkling2::carried_config(),
        )
        .expect("the released config is written");
        std::fs::write(
            directory.path().join("hf_quant_config.json"),
            sparkflash_model_inkling2::carried_quant(),
        )
        .expect("the quant groups are written");
        let checkpoint = directory.path().display().to_string();
        let peers = "127.0.0.1:29600,127.0.0.1:29601";
        let group_argv = |extra: &[&str]| {
            let mut argv = vec![
                "--model".to_owned(),
                "inkling2".to_owned(),
                "--checkpoint".to_owned(),
                checkpoint.clone(),
                "--rank".to_owned(),
                "0".to_owned(),
                "--peer".to_owned(),
                peers.to_owned(),
                "--converted".to_owned(),
                checkpoint.clone(),
            ];
            argv.extend(extra.iter().map(|flag| (*flag).to_owned()));
            argv
        };
        let options = ServeOptions::parse(group_argv(&[])).expect("the four-rank group parses");
        let group = options.group.expect("a group");
        assert_eq!(group.rank, 0);
        assert_eq!(group.world, 2);
        assert_eq!(
            group.control.port(),
            super::super::GROUP_DEFAULT_CONTROL_PORT
        );

        let error = ServeOptions::parse(group_argv(&["--concurrency", "2"]))
            .expect_err("a group cannot run two interleaved sequences");
        assert!(error.contains("--concurrency"), "{error}");

        let error = ServeOptions::parse(group_argv(&["--continuous-batching"]))
            .expect_err("a group cannot batch a second request into its lockstep");
        assert!(error.contains("--continuous-batching"), "{error}");

        let error = ServeOptions::parse(group_argv(&["--synthetic"]))
            .expect_err("a group has no synthetic plan to serve");
        assert!(error.contains("--synthetic"), "{error}");

        let other = ServeOptions::parse(vec![
            "--model".to_owned(),
            sparkflash_model::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID.to_owned(),
            "--checkpoint".to_owned(),
            checkpoint.clone(),
            "--rank".to_owned(),
            "0".to_owned(),
            "--peer".to_owned(),
            peers.to_owned(),
            "--converted".to_owned(),
            checkpoint,
        ])
        .expect_err("--rank is the Inkling device arm's group");
        assert!(other.contains("Inkling-2-Small"), "{other}");
    }
}
