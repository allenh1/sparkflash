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

//! Generates the Phase 2 sequence fixture (milestone 2A).
//!
//! Two invocations matter:
//!
//! The weight-free CI fixture (no checkpoint, no CUDA, deterministic; this is the one committed
//! under `tests/fixtures/` and the one CI regenerates byte-identically):
//!
//! ```text
//! cargo run -p sparkflash-engine --example sequence_fixture -- \
//!     --config tiny --prompt-tokens 130 --decode-steps 8 --seed 7 --logits all \
//!     --out crates/sparkflash-engine/tests/fixtures/sequence-130-tiny.json
//! ```
//!
//! The node golden fixture (real checkpoint, CPU forward, the proof-of-reference for milestone
//! 2B/2C; run on the node, not on the dev host):
//!
//! ```text
//! cargo run --release -p sparkflash-engine --example sequence_fixture -- \
//!     --config released --checkpoint /path/to/snapshot --prompt-tokens 130 \
//!     --decode-steps 8 --seed 7 --logits prefill \
//!     --out /home/user/phase2-evidence/sequence-130-released.json
//! ```
//!
//! The node golden fixture records the shell-runnable command, the host, the generator binary's
//! SHA-256 and free-form notes (each `--note TEXT` is appended verbatim, so a run can record the
//! fix commit, the toolchain and the squash expectation without changing the generated numbers or
//! the compared set - the note changes the fixture's provenance bytes, which is deliberate); the
//! CI fixture records only the canonical command, because it must regenerate byte-identically on
//! any host and binds no evidence to one machine or one binary (the schema says the same). A
//! `--note` on the CI provenance is refused: the CI fixture's recorded command has no note flag,
//! so a noted CI fixture would not regenerate from the command it records. Both print the fixture
//! file's own SHA-256 and size. The example never claims the fixture is a device run: milestone 2B
//! produces that separately.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sha2::{Digest, Sha256};
use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::sequence_fixture::{
    FixtureProvenance, FixtureWeights, LogitsCapture, MILESTONE_DECODE_STEPS,
    MILESTONE_PROMPT_TOKENS, SequenceFixture, ci_provenance_admits_notes, deterministic_prompt,
    generate_fixture,
};

const USAGE: &str = "usage: sequence_fixture --out PATH [--config tiny|released] \
[--checkpoint DIR] [--index PATH] [--prompt-tokens N] [--prompt-ids PATH] [--decode-steps N] \
[--seed N] [--logits all|prefill|none] [--provenance auto|ci|node] [--note TEXT]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigChoice {
    Tiny,
    Released,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProvenanceChoice {
    Auto,
    Ci,
    Node,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sequence_fixture: {error}");
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut config_choice = ConfigChoice::Tiny;
    let mut checkpoint: Option<PathBuf> = None;
    let mut index: Option<PathBuf> = None;
    let mut prompt_tokens = MILESTONE_PROMPT_TOKENS;
    let mut decode_steps = MILESTONE_DECODE_STEPS;
    let mut seed = 7u64;
    let mut logits: Option<LogitsCapture> = None;
    let mut provenance_choice = ProvenanceChoice::Auto;
    let mut out: Option<PathBuf> = None;
    let mut custom_notes: Vec<String> = Vec::new();
    let mut prompt_ids: Option<Vec<u32>> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--config" => {
                config_choice = match value()?.as_str() {
                    "tiny" => ConfigChoice::Tiny,
                    "released" => ConfigChoice::Released,
                    other => return Err(format!("unknown configuration {other}")),
                };
            }
            "--checkpoint" => checkpoint = Some(PathBuf::from(value()?)),
            "--index" => index = Some(PathBuf::from(value()?)),
            "--prompt-tokens" => {
                prompt_tokens = value()?
                    .parse()
                    .map_err(|_| "--prompt-tokens needs a number".to_string())?;
            }
            "--decode-steps" => {
                decode_steps = value()?
                    .parse()
                    .map_err(|_| "--decode-steps needs a number".to_string())?;
            }
            "--seed" => {
                seed = value()?
                    .parse()
                    .map_err(|_| "--seed needs a number".to_string())?;
            }
            // A prompt of real tokenizer ids: the oracle comparison needs the exact prompt the
            // served model saw, which a deterministic pseudo-random prompt cannot supply. The
            // file is a JSON array of integer token ids; the vocabulary bound and the emptiness
            // check stay where they are, in `generate_fixture`.
            "--prompt-ids" => {
                let path = PathBuf::from(value()?);
                let text = std::fs::read_to_string(&path)
                    .map_err(|error| format!("--prompt-ids {}: {error}", path.display()))?;
                let ids: Vec<u32> = serde_json::from_str(text.trim())
                    .map_err(|error| format!("--prompt-ids {}: {error}", path.display()))?;
                if ids.is_empty() {
                    return Err(format!("--prompt-ids {} is an empty array", path.display()));
                }
                prompt_ids = Some(ids);
            }
            "--logits" => {
                logits = Some(match value()?.as_str() {
                    "all" => LogitsCapture::All,
                    "prefill" => LogitsCapture::PrefillOnly,
                    "none" => LogitsCapture::None,
                    other => return Err(format!("unknown --logits value {other}")),
                });
            }
            "--provenance" => {
                provenance_choice = match value()?.as_str() {
                    "auto" => ProvenanceChoice::Auto,
                    "ci" => ProvenanceChoice::Ci,
                    "node" => ProvenanceChoice::Node,
                    other => return Err(format!("unknown --provenance value {other}")),
                };
            }
            "--out" => out = Some(PathBuf::from(value()?)),
            "--note" => custom_notes.push(value()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }

    let out = out.ok_or("--out is required")?;
    let config = match config_choice {
        ConfigChoice::Tiny => ModelConfig::tiny(),
        ConfigChoice::Released => ModelConfig::v41_flash().map_err(|error| error.to_string())?,
    };
    let logits = logits.unwrap_or(match config_choice {
        ConfigChoice::Tiny => LogitsCapture::All,
        ConfigChoice::Released => LogitsCapture::PrefillOnly,
    });
    let provenance_choice = match (provenance_choice, config_choice) {
        (ProvenanceChoice::Auto, ConfigChoice::Tiny) => ProvenanceChoice::Ci,
        (ProvenanceChoice::Auto, ConfigChoice::Released) => ProvenanceChoice::Node,
        (explicit, _) => explicit,
    };
    // A CI provenance on a checkpoint source would mislabel real weights; refuse the mismatch.
    if provenance_choice == ProvenanceChoice::Ci && config_choice == ConfigChoice::Released {
        return Err(
            "--provenance ci cannot describe the released checkpoint; use node or auto".to_string(),
        );
    }
    if provenance_choice == ProvenanceChoice::Node && config_choice == ConfigChoice::Tiny {
        return Err("--provenance node requires --config released".to_string());
    }
    // The CI fixture's canonical command has no note flag, so a note would make the fixture
    // unreproducible from the command it records. Refuse rather than write that.
    if provenance_choice == ProvenanceChoice::Ci {
        ci_provenance_admits_notes(&custom_notes).map_err(|error| error.to_string())?;
        // The CI fixture's recorded command is the deterministic-prompt invocation and must
        // regenerate byte-identically; a token-id file would not.
        if prompt_ids.is_some() {
            return Err("--provenance ci cannot describe a --prompt-ids fixture".to_string());
        }
    }
    // `--prompt-tokens` and `--seed` only describe the deterministic prompt; when real ids are
    // supplied the recorded `prompt_tokens` length comes from them, not from the flag.
    let prompt = match &prompt_ids {
        Some(ids) => ids.clone(),
        None => deterministic_prompt(config.vocab_size, prompt_tokens, seed),
    };
    let index_path = index.clone().or_else(|| {
        checkpoint
            .as_ref()
            .map(|dir| dir.join("model.safetensors.index.json"))
    });
    let weights = match config_choice {
        ConfigChoice::Tiny => FixtureWeights::FiniteFp32 { seed },
        ConfigChoice::Released => {
            let dir = checkpoint
                .as_deref()
                .ok_or("--config released requires --checkpoint")?;
            let index = index_path
                .as_deref()
                .ok_or("--config released needs the checkpoint's safetensors index")?;
            FixtureWeights::Checkpoint { dir, index }
        }
    };
    let provenance = match provenance_choice {
        ProvenanceChoice::Ci => FixtureProvenance {
            command: format!(
                "cargo run -p sparkflash-engine --example sequence_fixture -- --config tiny --prompt-tokens {prompt_tokens} --decode-steps {decode_steps} --seed {seed} --logits {} --out {}",
                logits_flag(logits),
                out.display()
            ),
            host: None,
            binary_sha256: None,
            notes: {
                let mut notes = vec![
                    "weight-free finite-FP32 source: CI fixture, not evidence about the released model"
                        .to_string(),
                    "host and binary sha256 are deliberately absent: the CI fixture must regenerate byte-identically on any host and binds no evidence to one machine or one binary; the node golden fixture records both".to_string(),
                ];
                notes.extend(custom_notes.iter().cloned());
                notes
            },
        },
        ProvenanceChoice::Node => node_provenance(&custom_notes)?,
        ProvenanceChoice::Auto => unreachable!("auto is resolved above"),
    };
    let fixture = generate_fixture(
        &config,
        config_label(config_choice),
        weights,
        &prompt,
        decode_steps,
        logits,
        provenance,
    )
    .map_err(|error| error.to_string())?;
    write_fixture(&out, &fixture)?;
    report(&out, &fixture);
    Ok(())
}

fn config_label(choice: ConfigChoice) -> &'static str {
    match choice {
        ConfigChoice::Tiny => "tiny",
        ConfigChoice::Released => "released",
    }
}

fn logits_flag(capture: LogitsCapture) -> &'static str {
    match capture {
        LogitsCapture::All => "all",
        LogitsCapture::PrefillOnly => "prefill",
        LogitsCapture::None => "none",
    }
}

fn node_provenance(custom_notes: &[String]) -> Result<FixtureProvenance, String> {
    let argv = std::env::args().collect::<Vec<_>>();
    let command = shell_join(&argv);
    let host = std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/proc/sys/kernel/hostname").ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty());
    let binary_sha256 = std::env::current_exe().ok().and_then(|path| {
        std::fs::read(path)
            .ok()
            .map(|bytes| hex(&Sha256::digest(&bytes)))
    });
    let mut notes = vec![
        "real-checkpoint reference forward: node golden fixture for the device comparison"
            .to_string(),
        "the engine does not implement CED; the harness supplies the encoder/decoder split, so only the compared set is comparable".to_string(),
    ];
    if let Ok(cwd) = std::env::current_dir() {
        notes.push(format!("run from {}", cwd.display()));
    }
    notes.extend(custom_notes.iter().cloned());
    Ok(FixtureProvenance {
        command,
        host,
        binary_sha256,
        notes,
    })
}

/// Joins an argv into a shell-runnable command, quoting every word so paths with spaces survive.
fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| format!("'{}'", arg.replace('\'', "'\\''")))
        .collect::<Vec<_>>()
        .join(" ")
}

fn write_fixture(path: &Path, fixture: &SequenceFixture) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut text = serde_json::to_string_pretty(fixture).map_err(|error| error.to_string())?;
    text.push('\n');
    std::fs::write(path, &text).map_err(|error| error.to_string())?;
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    println!(
        "wrote {} ({} B, sha256 {})",
        path.display(),
        bytes.len(),
        hex(&Sha256::digest(&bytes))
    );
    Ok(())
}

fn report(path: &Path, fixture: &SequenceFixture) {
    println!("fixture schema: {}", fixture.schema_version);
    println!(
        "configuration: {} ({} layers)",
        fixture.config_label, fixture.config.n_layers
    );
    println!(
        "source: {:?}{}",
        fixture.source.kind,
        fixture
            .source
            .checkpoint
            .as_deref()
            .map(|dir| format!(" at {dir}"))
            .unwrap_or_default()
    );
    println!(
        "prompt: {} tokens (ids recorded), decode steps: {}",
        fixture.prompt_tokens.len(),
        fixture.decode_steps
    );
    println!("window: {} tokens", fixture.window);
    println!("compared positions: {:?}", fixture.compared_positions);
    println!("steps: {}", fixture.steps.len());
    println!(
        "logits recorded: {} of {} steps, digests for all",
        fixture
            .steps
            .iter()
            .filter(|step| step.logits.is_some())
            .count(),
        fixture.steps.len()
    );
    println!(
        "kv digest rows: {} (2C populates them)",
        fixture.kv_rows.len()
    );
    println!("out: {}", path.display());
    match fixture.source.kind {
        sparkflash_engine::sequence_fixture::FixtureWeightKind::FiniteFp32 => {
            println!("note: weight-free CI fixture; not evidence about the released model");
        }
        sparkflash_engine::sequence_fixture::FixtureWeightKind::ReferenceCheckpoint => {
            println!("note: host reference fixture; the device run is separate evidence");
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
