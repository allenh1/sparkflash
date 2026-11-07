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

//! A released text-only Qwen checkpoint is a Qwen pack, not a dynamic host checkpoint.
//!
//! The text-only release writes its language configuration at the document root and names
//! itself `qwen4_exp_text`, while the arm, profile and plan are `qwen4_exp`. Detection,
//! conversion and serving must all fold that one alias; otherwise `serve --checkpoint` gates
//! the checkpoint against the dynamic host part set and refuses it before the Qwen arm runs.

use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sparkflash_model_qwen4_exp::{Qwen4ExpConfig, QwenCodec, write_synthetic_checkpoint};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sparkflash"))
}

/// A tiny text-only Qwen checkpoint: the synthetic fixture's tensors, with the configuration
/// flattened to the released text-only schema (`model_type: qwen4_exp_text`, no `text_config`).
fn text_only_checkpoint() -> (tempfile::TempDir, PathBuf) {
    let directory = tempfile::tempdir().expect("tempdir");
    let config = Qwen4ExpConfig::tiny();
    write_synthetic_checkpoint(directory.path(), &config, 7).expect("the tiny Qwen checkpoint");
    let nested: Value = serde_json::from_str(&config.to_json()).expect("the tiny document");
    let mut flat = nested["text_config"].clone();
    flat["model_type"] = json!("qwen4_exp_text");
    flat["architectures"] = json!(["Qwen4ExpTextForCausalLM"]);
    for field in [
        "tie_word_embeddings",
        "image_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ] {
        flat[field] = nested[field].clone();
    }
    std::fs::write(
        directory.path().join("config.json"),
        flat.to_string().into_bytes(),
    )
    .expect("the text-only config");
    std::fs::write(
        directory.path().join("tokenizer.json"),
        serde_json::to_vec(&tokenizer()).expect("tokenizer"),
    )
    .expect("the tokenizer");
    let path = directory.path().to_path_buf();
    (directory, path)
}

fn tokenizer() -> Value {
    // The flat configuration the fixture writes is the tiny Qwen document: it ends at id 2 and
    // names image/vision ids 3, 4 and 5. The tokenizer must agree with every one of them, or the
    // Qwen codec refuses before the arm's own host gate is reached. `tiny()` has one stop id, so
    // the fixture's end-of-text and turn-end spellings share id 2.
    let mut vocab = serde_json::Map::new();
    let specials = [
        ("<|endoftext|>", 2u64),
        ("<|im_end|>", 2),
        ("<|image_pad|>", 3),
        ("<|vision_start|>", 4),
        ("<|vision_end|>", 5),
        ("<|im_start|>", 6),
        ("hello", 7),
        ("world", 8),
        ("user", 9),
        ("assistant", 10),
    ];
    let mut added = Vec::new();
    for (token, id) in specials {
        vocab.insert(token.to_owned(), json!(id));
        if token.starts_with('<') {
            added.push(json!({
                "id": id,
                "content": token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            }));
        }
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>"},
    })
}

#[test]
fn a_text_only_qwen_checkpoint_converts_through_the_qwen_pack() {
    let (_directory, checkpoint) = text_only_checkpoint();
    let output = tempfile::tempdir().expect("output");
    // D89: a fixture process never reads or writes the operator's hub cache.
    let hub_cache = tempfile::tempdir().expect("hub cache");
    let convert = binary()
        .arg("convert")
        .arg("--checkpoint")
        .arg(&checkpoint)
        .arg("--output-dir")
        .arg(output.path())
        .env("HF_HUB_CACHE", hub_cache.path())
        .output()
        .expect("convert runs");
    assert!(
        convert.status.success(),
        "the text-only Qwen convert failed: {}",
        String::from_utf8_lossy(&convert.stderr)
    );
    assert!(
        output.path().join("rank-0-of-1.bin").is_file(),
        "the Qwen converter wrote no rank file: {}",
        String::from_utf8_lossy(&convert.stdout)
    );
}

#[test]
fn a_text_only_qwen_checkpoint_serves_through_the_qwen_arm() {
    let (_directory, checkpoint) = text_only_checkpoint();
    // D89: a fixture process never reads or writes the operator's hub cache. An empty cache also
    // keeps the Qwen arm's optional drafter from resolving a released one on a Spark.
    let hub_cache = tempfile::tempdir().expect("hub cache");
    let mut child = binary()
        .arg("serve")
        .arg("--checkpoint")
        .arg(&checkpoint)
        .arg("--bind")
        .arg("127.0.0.1:0")
        .env("HF_HUB_CACHE", hub_cache.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("serve starts");
    let log = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = mpsc::channel();
    for pipe in [
        child
            .stdout
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        let log = Arc::clone(&log);
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                let _ = tx.send(line.clone());
                log.lock().expect("log").push(line);
            }
        });
    }
    drop(tx);
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut listening = false;
    let mut exited = false;
    while Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                if line.contains("listening on ") {
                    listening = true;
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if child.try_wait().ok().flatten().is_some() {
                    exited = true;
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                exited = true;
                break;
            }
        }
    }
    let status = child.try_wait().ok().flatten();
    let _ = child.kill();
    let _ = child.wait();
    let lines = log.lock().expect("log").join("\n");
    if listening {
        return;
    }
    let code = status.and_then(|status| status.code());
    assert!(
        exited || code.is_some(),
        "the serve process neither bound nor exited; log:\n{lines}"
    );
    // The pre-arm "profile" line used to be enough for this test to pass, which hid a fixture
    // whose tokenizer disagreed with its own configuration. A refusal must now be the Qwen pack's
    // own gate, never the codec/tokenizer that runs before it.
    assert!(
        !lines.to_ascii_lowercase().contains("tokenizer")
            && !lines.contains("ChatML")
            && !lines.contains("special token"),
        "the text-only Qwen fixture was refused by its tokenizer/codec: {lines}"
    );
    // On a host without the Qwen CUDA image the pack's own gate refuses before it dispatches:
    // either the device open names the arm, a Qwen rank reports its loaded step, or the Qwen
    // pack's loadable-image check reports the parts this build does not carry. The pre-arm
    // profile line also contains "Qwen3.8-Flash-Next", so it must not be accepted as proof:
    // match the post-profile refusal/load spellings instead. Either is the Qwen path; the
    // dynamic host fallback (and its HOST_REFERENCE_PARTS gate) must not be the refusal.
    assert!(
        lines.contains("cannot open Qwen3.8-Flash-Next at")
            || lines.contains("Qwen rank ")
            || lines.contains("parts this build does not carry for sm121"),
        "the refusal did not reach the Qwen pack gate: {lines}"
    );
    assert!(
        !lines.contains("parts the dynamic host path does not execute")
            && !lines.contains("dynamic host reference"),
        "the text-only checkpoint fell through to the dynamic host path: {lines}"
    );
}

#[test]
fn the_flat_text_fixture_tokenizer_agrees_with_its_configuration() {
    let (_directory, checkpoint) = text_only_checkpoint();
    let config = Qwen4ExpConfig::from_checkpoint_dir(&checkpoint)
        .expect("the flattened text-only document parses as the Qwen pack");
    QwenCodec::from_path_with_architecture(
        &checkpoint.join("tokenizer.json"),
        &config.architecture,
    )
    .expect("the fixture tokenizer carries the configuration's eos, image and vision ids");
}
