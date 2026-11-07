use std::collections::BTreeSet;
use std::net::{TcpListener, TcpStream};
use std::thread;

use futures::StreamExt;

use super::*;

#[test]
fn the_glm_and_qwen_packs_may_take_the_group_flags() {
    for model in [
        sparkflash_model_glm5_next::MODEL_ID,
        sparkflash_model_qwen4_exp::MODEL_ID,
    ] {
        let options = ServeOptions::parse(tp4_args(&["--model", model]))
            .expect("the pack takes the group flags");
        assert_eq!(options.model.as_deref(), Some(model));
        assert!(options.tp4.is_some());
    }
}

#[test]
fn a_packed_model_without_a_group_arm_still_refuses_the_group_flags() {
    let error = ServeOptions::parse(tp4_args(&["--model", "example/no-group-arm"]))
        .expect_err("a pack without a tensor-parallel arm serves TP=1 only");
    assert!(error.contains("--tp4-rank"), "{error}");
}

#[test]
fn a_group_refuses_the_host_drivers_knobs() {
    for model in [
        sparkflash_model_glm5_next::MODEL_ID,
        sparkflash_model_qwen4_exp::MODEL_ID,
    ] {
        for (extra, flags) in [
            (&["--concurrency", "2"][..], &["--concurrency"][..]),
            (
                &["--concurrency", "2", "--fuse-batch"][..],
                &["--concurrency", "--fuse-batch"][..],
            ),
            (&["--kv-budget-gib", "24"][..], &["--kv-budget-gib"][..]),
        ] {
            let mut args = vec!["--model", model];
            args.extend_from_slice(extra);
            let error = ServeOptions::parse(tp4_args(&args))
                .expect_err("a group serves one request at a time");
            for flag in flags {
                assert!(error.contains(flag), "{model}: {error}");
            }
            assert!(error.contains("tensor-parallel group"), "{error}");
        }
    }
}

#[test]
fn every_rank_of_a_group_derives_the_same_window() {
    use pack::{GlmArm, PackArm, QwenArm};
    let dir = tempfile::tempdir().expect("tempdir");
    let config = QwenArm::write_fixture(dir.path()).expect("fixture");
    // A file in the checkpoint names its directory.
    let file = dir.path().join(CHECKPOINT_CONFIG_NAME);
    assert_eq!(pack::checkpoint_directory(&file), dir.path());
    let read = pack::read_config::<QwenArm>(dir.path()).expect("config");
    assert_eq!(read.max_position_embeddings, config.max_position_embeddings);
    assert!(config.max_position_embeddings < QwenArm::DEFAULT_CONTEXT);
    // Without --context the pack's default is capped by the checkpoint's window, the same
    // figure on every rank.
    for rank in 0..2 {
        let mut options =
            ServeOptions::parse(tp4_args(&["--model", sparkflash_model_qwen4_exp::MODEL_ID]))
                .expect("parses");
        options.tp4.as_mut().expect("group").rank = rank;
        assert_eq!(
            pack::pack_context::<QwenArm>(&options, &read),
            Ok(config.max_position_embeddings)
        );
    }
    let wide = ServeOptions::parse(tp4_args(&[
        "--model",
        sparkflash_model_qwen4_exp::MODEL_ID,
        "--context",
        "1000000",
    ]))
    .expect("parses");
    let error = pack::pack_context::<QwenArm>(&wide, &read).expect_err("past the window");
    assert!(error.contains("outside 1..="), "{error}");
    // A checkpoint without a config.json is refused rather than served from a carried release.
    let empty = tempfile::tempdir().expect("tempdir");
    let error = pack::read_config::<GlmArm>(empty.path()).expect_err("no config");
    assert!(error.contains("has no config.json"), "{error}");
    let error = pack::read_config::<QwenArm>(empty.path()).expect_err("no config");
    assert!(
        error.contains("cannot read the Qwen3.8-Flash-Next configuration"),
        "{error}"
    );
}

#[test]
fn a_checkpoint_names_its_pack_from_its_own_config() {
    const GLM: &str =
        include_str!("../../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
    const QWEN: &str =
        include_str!("../../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");
    const DEEPSEEK: &str = include_str!(
        "../../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
    );

    let glm = tempfile::tempdir().expect("tempdir");
    std::fs::write(glm.path().join(CHECKPOINT_CONFIG_NAME), GLM).expect("write");
    assert_eq!(pack::Pack::detect(glm.path()), Some(pack::Pack::Glm));

    let qwen = tempfile::tempdir().expect("tempdir");
    std::fs::write(qwen.path().join(CHECKPOINT_CONFIG_NAME), QWEN).expect("write");
    assert_eq!(pack::Pack::detect(qwen.path()), Some(pack::Pack::Qwen));

    // The V4.1 document is the default arm's model; detection names it, and the serve path
    // routes it to the existing reference/TP4 engine rather than a PackArm.
    let deepseek = tempfile::tempdir().expect("tempdir");
    std::fs::write(deepseek.path().join(CHECKPOINT_CONFIG_NAME), DEEPSEEK).expect("write");
    assert_eq!(
        pack::Pack::detect(deepseek.path()),
        Some(pack::Pack::Deepseek)
    );

    // The same detection makes --checkpoint enough to reach the pack's arm. The parts check is
    // the loadable one: a build with no embedded cubin and no SPARKFLASH_CUBINS refuses the
    // checkpoint by naming the images it cannot load.
    let parsed = ServeOptions::parse(vec![
        "--checkpoint".to_owned(),
        qwen.path().display().to_string(),
    ]);
    // The pack does not stage the vision tower, so the gate's answer is the language parts
    // only; the tower's DeepSeek-only cubin is not a missing pack part.
    let loadable: Vec<sparkflash_model::Part> = sparkflash_ops::parts::missing_loadable(
        &sparkflash_model::Architecture::from_json(QWEN).expect("the Qwen config"),
        sparkflash_model::Target::Sm121,
    )
    .into_iter()
    .filter(|part| *part != sparkflash_model::Part::Vision)
    .collect();
    match parsed {
        Ok(parsed) => {
            assert!(loadable.is_empty(), "this build can load every part");
            assert_eq!(
                parsed.model.as_deref(),
                Some(sparkflash_model_qwen4_exp::MODEL_ID)
            );
        }
        Err(error) => {
            assert!(
                !loadable.is_empty(),
                "a loadable checkpoint must parse: {error}"
            );
            assert!(error.contains("does not carry for sm121"), "{error}");
            for part in loadable {
                assert!(error.contains(&part.to_string()), "{error}");
            }
        }
    }
}

#[test]
fn only_a_read_timeout_counts_as_an_idle_control_wait() {
    let idle = Tp4ControlError::Io(std::io::Error::from(std::io::ErrorKind::WouldBlock));
    assert!(control_read_is_idle(&idle));
    let timed_out = Tp4ControlError::Io(std::io::Error::from(std::io::ErrorKind::TimedOut));
    assert!(control_read_is_idle(&timed_out));
    let reset = Tp4ControlError::Io(std::io::Error::from(std::io::ErrorKind::ConnectionReset));
    assert!(!control_read_is_idle(&reset));
}

#[test]
fn the_glm_cluster_snapshot_names_the_model_the_fabric_and_the_served_context() {
    let snapshot = pack::pack_cluster_snapshot(sparkflash_model_glm5_next::MODEL_ID, 4_096);
    assert_eq!(snapshot.model, sparkflash_model_glm5_next::MODEL_ID);
    assert_eq!(snapshot.fabric_mode, "tp4-tcp");
    assert_eq!(snapshot.max_seq_len, 4_096);
    assert!(snapshot.nodes.is_empty());
}

fn engine_request(prompt: &[u32], max_tokens: usize) -> EngineRequest {
    EngineRequest {
        request_id: "tp4-test".to_owned(),
        model: MODEL_ID.to_owned(),
        prompt_tokens: prompt.to_vec(),
        logprobs: LogprobsRequest::default(),
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
        seed: Some(0),
        trace_scopes: 0,
        trace_steps: 0,
        constraint: None,
        forced_tokens: Vec::new(),
    }
}

fn tp4_args(extra: &[&str]) -> Vec<String> {
    let mut args = vec![
        "--tp4-rank",
        "0",
        "--tp4-world-size",
        "2",
        "--tp4-peers",
        "127.0.0.1,127.0.0.2",
        "--checkpoint",
        "/tmp/tp4-checkpoint",
        "--tokenizer",
        "/tmp/tp4-tokenizer.json",
    ];
    args.extend_from_slice(extra);
    args.into_iter().map(str::to_owned).collect()
}

fn keys(value: &serde_json::Value, prefix: &str, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                let path = format!("{prefix}{key}");
                out.insert(path.clone());
                keys(child, &format!("{path}."), out);
            }
        }
        serde_json::Value::Array(items) => {
            if let Some(first) = items.first() {
                keys(first, &format!("{prefix}[]."), out);
            }
        }
        _ => {}
    }
}

fn shape(value: &serde_json::Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    keys(value, "", &mut out);
    out
}

#[test]
fn tp4_flags_parse_with_defaults() {
    let options = ServeOptions::parse(tp4_args(&[])).expect("the TP=4 flags parse");
    let tp4 = options.tp4.expect("the parse selected TP=4");
    assert_eq!(tp4.rank, 0);
    assert_eq!(tp4.world_size, 2);
    assert_eq!(tp4.base_port, TP4_DEFAULT_BASE_PORT);
    assert_eq!(
        tp4.control_port,
        TP4_DEFAULT_BASE_PORT + TP4_CONTROL_PORT_OFFSET
    );
    assert_eq!(
        tp4.peers,
        vec![
            "127.0.0.1".parse::<IpAddr>().unwrap(),
            "127.0.0.2".parse::<IpAddr>().unwrap()
        ]
    );
}

#[test]
fn tp4_flags_require_checkpoint_and_refuse_synthetic_and_device() {
    let no_checkpoint = ServeOptions::parse(
        tp4_args(&[])
            .into_iter()
            .filter(|arg| arg != "--checkpoint" && arg != "/tmp/tp4-checkpoint")
            .collect::<Vec<_>>(),
    )
    .expect_err("TP=4 without a checkpoint must be refused");
    assert!(no_checkpoint.contains("--checkpoint"), "{no_checkpoint}");

    let synthetic = ServeOptions::parse(tp4_args(&["--synthetic"]))
        .expect_err("TP=4 with the synthetic plan must be refused");
    assert!(synthetic.contains("--synthetic"), "{synthetic}");

    let device = ServeOptions::parse(tp4_args(&["--engine", "device"]))
        .expect_err("TP=4 with the device engine must be refused");
    assert!(device.contains("--engine device"), "{device}");
}

#[test]
fn tp4_flags_name_a_missing_rank_or_wrong_peer_count() {
    let mut no_rank = tp4_args(&[]);
    let position = no_rank.iter().position(|arg| arg == "--tp4-rank").unwrap();
    no_rank.drain(position..position + 2);
    let error = ServeOptions::parse(no_rank).expect_err("the other TP=4 flags need a rank");
    assert!(error.contains("--tp4-rank"), "{error}");

    let mut wrong_world = tp4_args(&[]);
    let world_position = wrong_world
        .iter()
        .position(|arg| arg == "--tp4-world-size")
        .unwrap();
    wrong_world[world_position + 1] = "4".to_owned();
    let error = ServeOptions::parse(wrong_world)
        .expect_err("a two-address peer list cannot describe four ranks");
    assert!(error.contains("--tp4-peers"), "{error}");
}

#[test]
fn control_frames_round_trip_on_a_loopback_socket() {
    let frames = vec![
        Tp4ControlFrame::Hello { rank: 1 },
        Tp4ControlFrame::Ready,
        Tp4ControlFrame::Request {
            prompt_ids: vec![1, 2, 3],
            max_tokens: 4,
            prompt_logprobs: None,
            speculative: true,
            cached_tokens: 0,
        },
        Tp4ControlFrame::Request {
            prompt_ids: vec![9, 8],
            max_tokens: 2,
            prompt_logprobs: Some(3),
            speculative: false,
            cached_tokens: 0,
        },
        Tp4ControlFrame::Token { token_id: 7 },
        Tp4ControlFrame::End,
        Tp4ControlFrame::Error {
            code: "engine_gone".to_owned(),
            message: "rank stopped".to_owned(),
        },
    ];
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the listener address");
    let echoed = frames.clone();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("the test client connects");
        for expected in echoed {
            let got = read_control_frame(&mut stream).expect("read one frame");
            assert_eq!(got, expected);
            write_control_frame(&mut stream, &got).expect("echo one frame");
        }
    });
    let mut client = TcpStream::connect(address).expect("connect to the test server");
    for frame in &frames {
        write_control_frame(&mut client, frame).expect("write one frame");
        let got = read_control_frame(&mut client).expect("read the echo");
        assert_eq!(&got, frame);
    }
    server.join().expect("the test server joins");
}

/// A rank-0 API stub for the control-protocol test: no forward, canned contract values.
struct FakeTp4Rank {
    prompt: Vec<u32>,
    seen_selection: Option<Option<usize>>,
}

impl Tp4RankApi for FakeTp4Rank {
    fn begin_request(&mut self, _sampling: Tp4Sampling) -> Result<(), String> {
        Ok(())
    }

    fn prefill(&mut self, _prompt: &[u32], _cached: usize) -> Result<Tp4Step, String> {
        Ok(Tp4Step {
            speculation: None,
            logits: vec![0.0; 8],
            next_token: Some(500),
            accepted: Vec::new(),
            proposed: 0,
            gated: 0,
            cached_tokens: 0,
        })
    }

    fn prefill_prompt_logprobs(
        &mut self,
        prompt: &[u32],
        selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
        self.seen_selection = Some(selection);
        assert_eq!(prompt, self.prompt.as_slice());
        let alternatives = selection.unwrap_or(0);
        let merged = (0..prompt.len() - 1)
            .map(|position| MergedPosition {
                position,
                own_token_id: prompt[position + 1],
                own_logprob: -1.25 - position as f32,
                own_rank: 1,
                alternatives: (0..alternatives)
                    .map(|slot| sparkflash_engine::prompt_ranks::MergedAlternative {
                        token_id: 600 + slot as u32,
                        logprob: -2.5 - slot as f32,
                    })
                    .collect(),
            })
            .collect();
        Ok((
            Tp4Step {
                speculation: None,
                logits: vec![0.0; 8],
                next_token: Some(500),
                accepted: Vec::new(),
                proposed: 0,
                gated: 0,
                cached_tokens: 0,
            },
            merged,
        ))
    }

    fn decode(&mut self, _token: u32) -> Result<Tp4Step, String> {
        Ok(Tp4Step {
            speculation: None,
            logits: vec![0.0; 8],
            next_token: Some(501),
            accepted: Vec::new(),
            proposed: 0,
            gated: 0,
            cached_tokens: 0,
        })
    }

    fn metrics(&self) -> Tp4RankMetrics {
        Tp4RankMetrics {
            load_ms: 0,
            resident_cache_bytes: 0,
            resident_cache_tensors: 0,
            reduce_calls: 0,
            reduce_elements: 0,
            steps: 0,
            last_step_ms: 0,
        }
    }
}

#[test]
fn the_prompt_logprob_request_frame_carries_the_selection_and_the_events_lead_with_the_payload() {
    let prompt = vec![100u32, 200, 300, 400];
    let tokenizer =
        load_hf_tokenizer(&sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path())
            .expect("tokenizer");
    // Bind a free loopback port, release it, and let the hub take it: the hub does not expose
    // its bound address, and the race window is a test-only detail.
    let probe = TcpListener::bind("127.0.0.1:0").expect("probe listener");
    let address = probe.local_addr().expect("probe address");
    drop(probe);
    let control_address = address;
    // The worker connects before the hub waits for it: connect retries until the hub binds,
    // and the hub cannot return until every worker has said Hello.
    let worker = thread::spawn(move || {
        let mut control = Tp4ControlWorker::connect(control_address, 1, 2, Duration::from_secs(10))
            .expect("worker connect");
        control.handshake().expect("handshake");
        let mut requests = Vec::new();
        while let Ok(frame) = control.next_frame() {
            match frame {
                Tp4ControlFrame::Request {
                    prompt_ids,
                    max_tokens,
                    prompt_logprobs,
                    ..
                } => {
                    requests.push((prompt_ids, max_tokens, prompt_logprobs));
                    loop {
                        match control.next_frame().expect("decode frame") {
                            Tp4ControlFrame::Token { token_id } => {
                                assert_eq!(token_id, 500);
                            }
                            Tp4ControlFrame::End => break,
                            other => panic!("expected Token or End, got {other:?}"),
                        }
                    }
                }
                Tp4ControlFrame::End => break,
                other => panic!("expected Request or End, got {other:?}"),
            }
        }
        requests
    });
    let mut hub = Tp4ControlHub::start(address, 2, Duration::from_secs(10)).expect("hub");
    let (replies, mut answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
    let job = Job {
        prompt_tokens: prompt.clone(),
        forced_tokens: Vec::new(),
        max_tokens: 2,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        skip_special_tokens: true,
        seed: 0,
        logprobs: LogprobsRequest {
            sample: None,
            prompt: Some(LogprobSelection::Top(3)),
            token_ids: Vec::new(),
        },
        stop: Vec::new(),
        enqueued_at: Instant::now(),
        replies,
    };
    let metrics = Metrics::new();
    let mut rank = FakeTp4Rank {
        prompt: prompt.clone(),
        seen_selection: None,
    };
    run_tp4_request(&mut rank, &mut hub, &tokenizer, job, &metrics).expect("request runs");
    assert_eq!(rank.seen_selection, Some(Some(3)));
    // Closing the hub is what ends the worker loop in the test; the real worker exits with
    // the process.
    drop(hub);
    let requests = worker.join().expect("worker joins");
    assert_eq!(
        requests,
        vec![(prompt.clone(), 2, Some(3u32))],
        "the Request frame must carry the prompt-logprob selection"
    );
    // The first reply is the whole-prompt payload, delivered after prefill and before the
    // first generated token, exactly where the serial host path delivers it.
    let first = answers.try_recv().expect("the prompt payload");
    let Reply::PromptLogprobs(payload) = first else {
        panic!("expected PromptLogprobs first, got a different reply");
    };
    assert_eq!(payload.tokens.len(), prompt.len());
    assert!(payload.positions[0].is_none());
    for (index, position) in payload.positions.iter().enumerate().skip(1) {
        let position = position.as_ref().expect("a scored position");
        assert_eq!(position.token_id, prompt[index]);
        assert_eq!(position.top.len(), 4, "own token plus three alternatives");
        assert_eq!(position.top[0].token_id, prompt[index]);
        assert_eq!(position.top[0].logprob, position.logprob);
        assert_eq!(position.top[0].rank, Some(1));
        for (slot, alternative) in position.top.iter().skip(1).enumerate() {
            assert_eq!(alternative.rank, Some(u32::try_from(slot + 1).unwrap()));
        }
    }
    // Then the generated-token deltas and the terminal reply.
    let mut deltas = 0usize;
    let mut done = false;
    while let Ok(reply) = answers.try_recv() {
        match reply {
            Reply::Delta(_) => deltas += 1,
            Reply::Done(_) => done = true,
            _ => panic!("unexpected reply after the payload"),
        }
    }
    assert!(deltas >= 1);
    assert!(done, "the request must end with the Done summary");
}

/// A rank-0 stub whose every step keeps two drafts: decode(t) decodes t + 1 and t + 2 in its
/// verify and picks t + 3.
#[derive(Default)]
struct SpeculativeFakeRank {
    speculative: Option<bool>,
    fed: Vec<u32>,
}

impl Tp4RankApi for SpeculativeFakeRank {
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String> {
        self.speculative = Some(sampling.speculative);
        Ok(())
    }

    fn prefill(&mut self, _prompt: &[u32], _cached: usize) -> Result<Tp4Step, String> {
        Ok(Tp4Step {
            speculation: None,
            next_token: Some(500),
            ..Tp4Step::default()
        })
    }

    fn prefill_prompt_logprobs(
        &mut self,
        _prompt: &[u32],
        _selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
        Err("not asked".to_owned())
    }

    fn decode(&mut self, token: u32) -> Result<Tp4Step, String> {
        self.fed.push(token);
        Ok(Tp4Step {
            speculation: None,
            logits: Vec::new(),
            next_token: Some(token + 3),
            accepted: vec![token + 1, token + 2],
            proposed: 2,
            gated: 0,
            cached_tokens: 0,
        })
    }

    fn metrics(&self) -> Tp4RankMetrics {
        Tp4RankMetrics::default()
    }
}

#[test]
fn a_speculative_steps_kept_drafts_are_emitted_and_only_its_pick_is_fed() {
    let tokenizer =
        load_hf_tokenizer(&sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path())
            .expect("tokenizer");
    let probe = TcpListener::bind("127.0.0.1:0").expect("probe listener");
    let address = probe.local_addr().expect("probe address");
    drop(probe);
    let worker = thread::spawn(move || {
        let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(10))
            .expect("worker connect");
        control.handshake().expect("handshake");
        let (mut speculative, mut tokens) = (None, Vec::new());
        while let Ok(frame) = control.next_frame() {
            match frame {
                Tp4ControlFrame::Request {
                    speculative: flag, ..
                } => speculative = Some(flag),
                Tp4ControlFrame::Token { token_id } => tokens.push(token_id),
                Tp4ControlFrame::End => {}
                other => panic!("unexpected frame {other:?}"),
            }
        }
        (speculative, tokens)
    });
    let mut hub = Tp4ControlHub::start(address, 2, Duration::from_secs(10)).expect("hub");
    let (replies, mut answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
    let job = Job {
        prompt_tokens: vec![100, 200],
        forced_tokens: Vec::new(),
        max_tokens: 6,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        skip_special_tokens: true,
        seed: 0,
        logprobs: LogprobsRequest {
            sample: None,
            prompt: None,
            token_ids: Vec::new(),
        },
        stop: Vec::new(),
        enqueued_at: Instant::now(),
        replies,
    };
    let mut rank = SpeculativeFakeRank::default();
    run_tp4_request(&mut rank, &mut hub, &tokenizer, job, &Metrics::new()).expect("request runs");
    drop(hub);
    // 500, then 501 502 503 from the step fed 500, then 504 505 from the step fed 503: six
    // tokens, two steps, and the workers run the same two.
    assert_eq!(rank.speculative, Some(true));
    assert_eq!(rank.fed, [500, 503]);
    assert_eq!(
        worker.join().expect("worker joins"),
        (Some(true), vec![500, 503])
    );
    let mut completion_tokens = None;
    while let Ok(reply) = answers.try_recv() {
        if let Reply::Done(summary) = reply {
            completion_tokens = Some(summary.completion_tokens);
        }
    }
    assert_eq!(completion_tokens, Some(6));
}

/// A rank-0 stub that always picks 999, drafts only when asked to speculate, and records what
/// it was fed.
#[derive(Default)]
struct StubbornFakeRank {
    speculative: Option<bool>,
    fed: Vec<u32>,
}

impl StubbornFakeRank {
    fn step(&self, token: u32) -> Tp4Step {
        let mut logits = vec![0.0; 1024];
        logits[999] = 4.0;
        Tp4Step {
            speculation: None,
            logits,
            next_token: Some(999),
            accepted: if self.speculative == Some(true) {
                vec![token + 1]
            } else {
                Vec::new()
            },
            proposed: usize::from(self.speculative == Some(true)),
            gated: 0,
            cached_tokens: 0,
        }
    }
}

impl Tp4RankApi for StubbornFakeRank {
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String> {
        self.speculative = Some(sampling.speculative);
        Ok(())
    }

    fn prefill(&mut self, _prompt: &[u32], _cached: usize) -> Result<Tp4Step, String> {
        Ok(self.step(0))
    }

    fn prefill_prompt_logprobs(
        &mut self,
        _prompt: &[u32],
        _selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
        Err("not asked".to_owned())
    }

    fn decode(&mut self, token: u32) -> Result<Tp4Step, String> {
        self.fed.push(token);
        Ok(self.step(token))
    }

    fn metrics(&self) -> Tp4RankMetrics {
        Tp4RankMetrics::default()
    }
}

#[test]
fn a_forced_tp4_request_emits_scores_and_broadcasts_the_forced_tokens() {
    let tokenizer =
        load_hf_tokenizer(&sparkflash_model_deepseek_v41::encoding::reference_tokenizer_path())
            .expect("tokenizer");
    let probe = TcpListener::bind("127.0.0.1:0").expect("probe listener");
    let address = probe.local_addr().expect("probe address");
    drop(probe);
    let worker = thread::spawn(move || {
        let mut control = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(10))
            .expect("worker connect");
        control.handshake().expect("handshake");
        let (mut speculative, mut tokens) = (None, Vec::new());
        while let Ok(frame) = control.next_frame() {
            match frame {
                Tp4ControlFrame::Request {
                    speculative: flag, ..
                } => speculative = Some(flag),
                Tp4ControlFrame::Token { token_id } => tokens.push(token_id),
                Tp4ControlFrame::End => {}
                other => panic!("unexpected frame {other:?}"),
            }
        }
        (speculative, tokens)
    });
    let mut hub = Tp4ControlHub::start(address, 2, Duration::from_secs(10)).expect("hub");
    let (replies, mut answers) = tokio::sync::mpsc::channel(REPLY_BUFFER);
    let forced = vec![300u32, 301, 302];
    let job = Job {
        prompt_tokens: vec![100, 200],
        forced_tokens: forced.clone(),
        max_tokens: forced.len(),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        skip_special_tokens: true,
        seed: 0,
        logprobs: LogprobsRequest {
            sample: Some(LogprobSelection::Top(1)),
            prompt: None,
            token_ids: Vec::new(),
        },
        stop: Vec::new(),
        enqueued_at: Instant::now(),
        replies,
    };
    let mut rank = StubbornFakeRank::default();
    run_tp4_request(&mut rank, &mut hub, &tokenizer, job, &Metrics::new()).expect("request runs");
    drop(hub);
    // The last token is never fed back, so rank 0 and the workers both decode the first two.
    assert_eq!(rank.speculative, Some(false));
    assert_eq!(rank.fed, forced[..2]);
    assert_eq!(
        worker.join().expect("worker joins"),
        (Some(false), forced[..2].to_vec())
    );
    let mut scored = Vec::new();
    let mut completion_tokens = None;
    while let Ok(reply) = answers.try_recv() {
        match reply {
            Reply::TokenLogprobs(entry) => {
                assert!(
                    entry
                        .top
                        .iter()
                        .any(|alternative| alternative.token_id == 999),
                    "rank 0's own choice stays visible in the alternatives"
                );
                scored.push(entry.token_id);
            }
            Reply::Done(summary) => completion_tokens = Some(summary.completion_tokens),
            _ => {}
        }
    }
    assert_eq!(scored, forced, "every forced token is emitted and scored");
    assert_eq!(completion_tokens, Some(forced.len()));
}

#[test]
fn the_tp4_prompt_logprobs_payload_matches_the_serial_assembly() {
    // A fabricated finite log-softmax row and a prompt that walks it: the rank contract's
    // merged values are derived from the same row the serial formula uses, so the payload the
    // TP=4 path builds must equal the serial token_logprob assembly exactly, strings and order
    // included.
    let prompt: Vec<u32> = vec![2, 5, 8, 13, 21];
    let row: Vec<f32> = (0..64).map(|index| index as f32 * 0.7 - 6.0).collect();
    let values = log_softmax(&row);
    let decode_ids = |ids: &[u32]| -> Result<String, String> {
        Ok(ids
            .iter()
            .map(|token| format!("<{token}>"))
            .collect::<String>())
    };
    for requested in [None, Some(0u32), Some(3u32)] {
        let selection = requested.map(LogprobSelection::Top);
        let request = LogprobsRequest {
            sample: None,
            prompt: selection,
            token_ids: Vec::new(),
        };
        let alternatives: usize = requested.unwrap_or(0) as usize;
        let merged: Vec<MergedPosition> = (0..prompt.len() - 1)
            .map(|index| {
                let own_token_id = prompt[index + 1];
                let own_logprob = values[own_token_id as usize];
                MergedPosition {
                    position: index,
                    own_token_id,
                    own_logprob,
                    own_rank: competition_rank(&values, own_logprob),
                    alternatives: top_k_indices(&values, alternatives)
                        .into_iter()
                        .map(|token| sparkflash_engine::prompt_ranks::MergedAlternative {
                            token_id: token as u32,
                            logprob: values[token],
                        })
                        .collect(),
                }
            })
            .collect();
        let mut decoder = LogprobDecoder::new(&decode_ids);
        let payload = tp4_prompt_logprobs_payload(&prompt, &merged, &request, &mut decoder)
            .expect("the TP=4 payload builds");
        // The serial assembly of the same request, with the same decoding sequence.
        let mut serial_decoder = LogprobDecoder::new(&decode_ids);
        let mut tokens = Vec::new();
        let mut positions: Vec<Option<TokenLogprob>> = vec![None; prompt.len()];
        tokens.push(TokenString {
            token_id: prompt[0],
            token: serial_decoder.decode(prompt[0]).expect("decode"),
        });
        serial_decoder.push(prompt[0]);
        for index in 0..prompt.len() - 1 {
            let token_index = index + 1;
            let token = serial_decoder.decode(prompt[token_index]).expect("decode");
            tokens.push(TokenString {
                token_id: prompt[token_index],
                token: token.clone(),
            });
            positions[token_index] = Some(
                token_logprob(
                    &values,
                    prompt[token_index],
                    token,
                    selection,
                    &[],
                    &mut serial_decoder,
                )
                .expect("serial token logprob"),
            );
            serial_decoder.push(prompt[token_index]);
        }
        assert_eq!(
            payload,
            PromptLogprobs { tokens, positions },
            "requested {requested:?} changed the payload"
        );
    }
}

#[test]
fn missing_rank_is_a_typed_startup_timeout() {
    let error = Tp4ControlHub::start(
        SocketAddr::from(([127, 0, 0, 1], 0)),
        2,
        Duration::from_millis(80),
    )
    .expect_err("a missing rank cannot complete the group");
    match error {
        Tp4ControlError::MissingRank {
            world,
            joined,
            waited,
        } => {
            assert_eq!(world, 2);
            assert_eq!(joined, 0);
            assert_eq!(waited, Duration::from_millis(80));
        }
        other => panic!("expected MissingRank, got {other}"),
    }
}

#[test]
fn worker_control_connect_retries_until_the_hub_binds() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the listener address");
    // Release the port so the worker's early connect attempts are refused, exactly the
    // rank-0-loads-slower startup race; the hub binds after a short delay.
    drop(listener);
    let server = thread::spawn(move || {
        thread::sleep(Duration::from_millis(250));
        let listener =
            TcpListener::bind(address).expect("the hub binds after the worker starts trying");
        let (mut stream, _) = listener.accept().expect("the retried worker connects");
        let frame = read_control_frame(&mut stream).expect("read the worker Hello");
        assert!(
            matches!(frame, Tp4ControlFrame::Hello { rank: 1 }),
            "{frame:?}"
        );
        write_control_frame(&mut stream, &Tp4ControlFrame::Ready).expect("send Ready");
    });
    let mut worker = Tp4ControlWorker::connect(address, 1, 2, Duration::from_secs(5))
        .expect("the connect retries until the hub is listening");
    worker
        .handshake()
        .expect("the retried connection handshakes");
    assert!(worker.ready);
    server.join().expect("the test hub joins");
}

#[test]
fn worker_control_connect_names_the_peer_when_the_hub_never_binds() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the listener address");
    drop(listener);
    let waited = Duration::from_millis(150);
    let error = match Tp4ControlWorker::connect(address, 1, 2, waited) {
        Ok(_) => panic!("a hub that never binds must not report a connected worker"),
        Err(error) => error,
    };
    let display = error.to_string();
    match error {
        Tp4ControlError::ConnectTimeout {
            address: tried,
            waited: reported,
        } => {
            assert_eq!(tried, address);
            assert_eq!(reported, waited);
        }
        other => panic!("expected ConnectTimeout, got {other}"),
    }
    assert!(
        display.contains(&address.to_string()),
        "the refusal must name the peer: {display}"
    );
    assert!(
        display.contains("150ms"),
        "the refusal must name the elapsed budget: {display}"
    );
}

#[test]
fn an_idle_control_gap_does_not_take_the_worker_down() {
    // The real timeout is ten minutes; shrink it so the test does not build a minute of wait
    // into CI. The worker is the client, and rank 0 stays silent for several timeouts before
    // it ends the group.
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the listener address");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("the test hub connects");
        thread::sleep(Duration::from_millis(350));
        write_control_frame(&mut stream, &Tp4ControlFrame::End).expect("write End");
    });
    let stream = TcpStream::connect(address).expect("connect to the test hub");
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .expect("a short read timeout");
    let mut worker = Tp4ControlWorker {
        stream,
        rank: 1,
        world: 2,
        ready: true,
    };
    let mut rank = FakeTp4Rank {
        prompt: Vec::new(),
        seen_selection: None,
    };
    let started = Instant::now();
    tp4_worker_loop(&mut rank, &mut worker, 1)
        .expect("a silent rank 0 is an idle wait, not a dead peer");
    assert!(
        started.elapsed() >= Duration::from_millis(300),
        "the loop must survive more than one read timeout"
    );
    server.join().expect("the test hub joins");
}

#[test]
fn request_before_ready_is_refused_by_the_worker() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback listener");
    let address = listener.local_addr().expect("the listener address");
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("the test hub connects");
        let mut worker = Tp4ControlWorker {
            stream,
            rank: 1,
            world: 2,
            ready: false,
        };
        assert!(matches!(
            worker.next_frame(),
            Err(Tp4ControlError::NotReady { .. })
        ));
    });
    let _client = TcpStream::connect(address).expect("connect to the test worker");
    server.join().expect("the test worker joins");
}

#[tokio::test]
async fn one_request_guard_refuses_a_second_concurrent_request() {
    let (jobs, mut queued) = tokio::sync::mpsc::channel(TP4_JOB_QUEUE);
    let driver = Tp4Driver::for_test(jobs, 4);
    assert!(
        !driver.independent_choices(),
        "the router must refuse n > 1 on the lockstep TP=4 group"
    );
    let _first = driver.generate(engine_request(&[1, 2, 3], 2));
    let second = driver.generate(engine_request(&[1, 2, 3], 2));
    let events: Vec<EngineEvent> = second.collect().await;
    assert!(
        matches!(
            events.first(),
            Some(EngineEvent::Error { code, .. }) if code == REFUSED_CODE
        ),
        "the second concurrent request must be the existing typed refusal: {events:?}"
    );
    let _ = queued.try_recv();
}

#[test]
fn tp4_stats_keep_the_reference_shape_and_name_the_fabric() {
    let timing = RequestTiming::default();
    let reference = serde_json::to_value(request_stats_with_fabric(3, timing, "", 0))
        .expect("the reference stats serialize");
    let tp4 = serde_json::to_value(request_stats_with_fabric(3, timing, TP4_FABRIC, 0))
        .expect("the TP=4 stats serialize");
    assert_eq!(
        shape(&reference),
        shape(&tp4),
        "the stats keys must not drift"
    );
    assert_eq!(tp4["transport"]["fabric_mode"], TP4_FABRIC);
    assert_eq!(reference["transport"]["fabric_mode"], "");
    assert_eq!(tp4["kv_cache"]["total_prompt_tokens"], 3);
    assert_eq!(tp4["kv_cache"]["global_kv_cached_tokens"], 0);
}

#[test]
fn the_checkpoint_gate_serves_language_parts_and_deepseek_vision() {
    // Qwen and GLM stage no vision tower; DeepSeek's device arm runs the 14-patch one.
    let cubins = std::env::var_os(sparkflash_ops::parts::CUBIN_DIR_ENV);
    for (name, text) in [
        (
            "Qwen",
            include_str!(
                "../../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_fp8.hf.json"
            ),
        ),
        (
            "GLM",
            include_str!("../../../sparkflash-model-glm5-next/config/glm53_flash.hf.json"),
        ),
    ] {
        let dir = tempfile::tempdir().expect("a temporary checkpoint");
        std::fs::write(dir.path().join(CHECKPOINT_CONFIG_NAME), text).expect("the config");
        // SAFETY: the value is restored before the next document.
        unsafe { std::env::remove_var(sparkflash_ops::parts::CUBIN_DIR_ENV) };
        let parsed = ServeOptions::parse(vec![
            "--checkpoint".to_owned(),
            dir.path().to_string_lossy().to_string(),
        ]);
        if let Some(value) = cubins.as_ref() {
            unsafe { std::env::set_var(sparkflash_ops::parts::CUBIN_DIR_ENV, value) };
        }
        // A CPU-only build embeds no pack image, so the language parts themselves are missing
        // and the gate names them. What must never be missing is the unserved vision tower.
        let missing: Vec<sparkflash_model::Part> = sparkflash_ops::parts::missing_loadable(
            &sparkflash_model::Architecture::from_json(text).expect("the released config"),
            sparkflash_model::Target::Sm121,
        )
        .into_iter()
        .filter(|part| *part != sparkflash_model::Part::Vision)
        .collect();
        if missing.is_empty() {
            let options =
                parsed.unwrap_or_else(|error| panic!("{name}: the checkpoint must parse: {error}"));
            assert!(options.model.is_some(), "{name}: detection selects the arm");
        } else {
            let error = parsed.expect_err("an unembedded pack image still refuses");
            assert!(!error.contains("vision"), "{name}: {error}");
            for part in missing {
                assert!(error.contains(&part.to_string()), "{name}: {error}");
            }
        }
    }
    let released = include_str!(
        "../../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json"
    );
    let dir = tempfile::tempdir().expect("a temporary checkpoint");
    std::fs::write(
        dir.path().join(CHECKPOINT_CONFIG_NAME),
        released.replace("\"patch_size\": 14", "\"patch_size\": 16"),
    )
    .expect("the config");
    let error = check_checkpoint_plan(dir.path(), 4, Availability::Catalogue)
        .expect_err("a 16-patch tower is refused");
    assert!(error.contains("vision patch 16"), "{error}");
}
