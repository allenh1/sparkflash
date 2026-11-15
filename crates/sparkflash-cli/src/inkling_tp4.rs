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

//! The `inkling-tp4` subcommand: one rank of an Inkling-2-Small four-host run.
//!
//!   sparkflash inkling-tp4 --model DIR --converted DIR --rank 0 --world 4 \
//!       --peer 192.0.2.10:29600,192.0.2.11:29600,192.0.2.12:29600,192.0.2.13:29600 \
//!       --context 4096 --prompt 1,2,3 --decode 8
//!
//! Every host runs the same command with its own `--rank`. The addresses are given in rank order,
//! so rank 0 accepts and the others dial it. The arguments and the walk are the arm's
//! ([sparkflash_model_inkling2::Tp4Args], [sparkflash_model_inkling2::Tp4Run]) and this file only
//! resolves the prompt and prints, so a launcher, a test and the CLI run the same code.

use std::process::ExitCode;

use sparkflash_model::PromptCodec;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::tp4_run::{TP4_USAGE, Tp4Args};
use sparkflash_model_inkling2::{Tp4Run, rank_file};

/// Runs `sparkflash inkling-tp4` with the arguments that followed the subcommand.
pub fn run(args: impl Iterator<Item = String>) -> ExitCode {
    let args = match Tp4Args::parse(args) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("sparkflash inkling-tp4: {error}");
            eprintln!("{TP4_USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match execute(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sparkflash inkling-tp4: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Opens the rank, walks the prompt, and prints what this rank saw.
fn execute(args: &Tp4Args) -> Result<(), String> {
    let config = Inkling2Config::from_checkpoint_dir(&args.model)
        .map_err(|error| format!("--model {}: {error}", args.model.display()))?;
    let codec = args
        .tokenizer
        .as_ref()
        .map(|path| sparkflash_model_inkling2::InklingCodec::from_path(path, None, Vec::new()))
        .transpose()?;
    // A sentence is tokenized here and only here: the four ranks must forward the same ids, so the
    // prompt is resolved before the rank opens anything.
    let prompt = match (&args.text, &codec) {
        (Some(text), Some(codec)) => {
            // A chat model answers an instruction only through its own template; a bare sentence
            // continues as a document and a greedy decode degenerates, which reads like a kernel
            // bug and is not one.
            let rendered = if args.chat {
                let request = serde_json::json!({
                    "messages": [{"role": "user", "content": text}],
                });
                sparkflash_model_inkling2::InklingCodec::render_chat(&request)
                    .map_err(|error| format!("--chat: {error}"))?
            } else {
                text.clone()
            };
            let ids = codec
                .encode_text(&rendered, true)
                .map_err(|error| format!("--text: {error}"))?;
            println!("inkling-tp4: prompt {:?} from {:?}", ids, rendered);
            ids
        }
        _ => args.prompt.clone(),
    };
    let converted = rank_file(&args.converted, args.rank, args.world)?;
    let run = Tp4Run {
        rank: args.rank,
        world: args.world,
        addresses: args.peers.clone(),
        converted,
        checkpoint: args.model.clone(),
        context: args.context,
        passes: args.passes,
        cuts: args.cuts,
        dump: args.dump.clone(),
    };
    let outcome = run.execute(
        &sparkflash_model_inkling2::descriptor(),
        &config,
        &prompt,
        args.decode,
    )?;
    println!("inkling-tp4: {}", outcome.banner);
    println!(
        "inkling-tp4: rank {} prompt {:?} generated {:?}",
        outcome.rank, outcome.prompt, outcome.generated
    );
    println!(
        "inkling-tp4: rank {} prefill {:?} ms, {} decode step(s) {:?} ms, logits {} best {:?}",
        outcome.rank,
        outcome.prefill_ms,
        outcome.decode_ms.len(),
        outcome.decode_ms,
        outcome.logits,
        outcome.best
    );
    println!(
        "inkling-tp4: rank {} decode logits digest {:016x}",
        outcome.rank, outcome.decode_digest
    );
    for score in &outcome.scores {
        println!(
            "inkling-tp4: rank {} score length {} best {:?} next {:?}",
            outcome.rank, score.length, score.best, score.next
        );
    }
    if let Some(codec) = &codec {
        let text = codec.decode(&outcome.generated).unwrap_or_default();
        println!("inkling-tp4: rank {} text {:?}", outcome.rank, text);
    }
    Ok(())
}
