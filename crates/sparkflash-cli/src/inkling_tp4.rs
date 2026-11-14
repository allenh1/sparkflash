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
//! so rank 0 accepts and the others dial it; the walk is the arm's ([sparkflash_model_inkling2::Tp4Run])
//! and this arm only parses, so a launcher, a test and the CLI run the same code.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use sparkflash_model::PromptCodec;
use sparkflash_model_inkling2::config::Inkling2Config;
use sparkflash_model_inkling2::{Tp4Run, rank_file};

/// One parsed invocation.
#[derive(Debug, Clone)]
struct Args {
    model: PathBuf,
    converted: PathBuf,
    rank: usize,
    world: usize,
    peers: Vec<SocketAddr>,
    context: usize,
    prompt: Vec<u32>,
    decode: usize,
    passes: usize,
    cuts: usize,
    dump: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    text: Option<String>,
    chat: bool,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut converted = None;
        let mut rank = None;
        let mut world = None;
        let mut peers = None;
        let mut context = 4096usize;
        let mut prompt = None;
        let mut tokenizer = None;
        let mut text = None;
        let mut chat = false;
        let mut decode = 8usize;
        let mut passes = 1usize;
        let mut cuts = 0usize;
        let mut dump = None;
        let mut rest = args.peekable();
        while let Some(flag) = rest.next() {
            let mut value = || -> Result<String, String> {
                rest.next().ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value()?)),
                "--converted" => converted = Some(PathBuf::from(value()?)),
                "--rank" => {
                    rank = Some(
                        value()?
                            .parse::<usize>()
                            .map_err(|error| format!("--rank: {error}"))?,
                    )
                }
                "--world" => {
                    world = Some(
                        value()?
                            .parse::<usize>()
                            .map_err(|error| format!("--world: {error}"))?,
                    )
                }
                "--context" => {
                    context = value()?
                        .parse::<usize>()
                        .map_err(|error| format!("--context: {error}"))?;
                }
                "--decode" => {
                    decode = value()?
                        .parse::<usize>()
                        .map_err(|error| format!("--decode: {error}"))?;
                }
                // The prompt is walked this many times on the one open rank, each in a fresh
                // session, so the warm-up the first walk pays shows apart from the steady cost.
                "--passes" => {
                    passes = value()?
                        .parse::<usize>()
                        .map_err(|error| format!("--passes: {error}"))?;
                }
                // After the passes, the prompt's first len * i / cuts tokens are prefilled for i in
                // 1..=cuts, each in a fresh session, and rank 0 writes their logit rows to
                // --dump-logits: two prefill walks compared at the same positions of one text.
                "--cuts" => {
                    cuts = value()?
                        .parse::<usize>()
                        .map_err(|error| format!("--cuts: {error}"))?;
                }
                "--dump-logits" => dump = Some(PathBuf::from(value()?)),
                "--chat" => chat = true,
                "--tokenizer" => tokenizer = Some(PathBuf::from(value()?)),
                "--text" => text = Some(value()?),
                "--prompt" => {
                    let raw = value()?;
                    let mut ids = Vec::new();
                    for token in raw.split(',').filter(|token| !token.is_empty()) {
                        ids.push(
                            token
                                .parse::<u32>()
                                .map_err(|error| format!("--prompt: {token}: {error}"))?,
                        );
                    }
                    prompt = Some(ids);
                }
                "--peer" => {
                    let raw = value()?;
                    let mut list = Vec::new();
                    for address in raw.split(',').filter(|address| !address.is_empty()) {
                        list.push(
                            address
                                .parse::<SocketAddr>()
                                .map_err(|error| format!("--peer: {address}: {error}"))?,
                        );
                    }
                    peers = Some(list);
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        let peers = peers.ok_or_else(|| "--peer is required".to_owned())?;
        let world = world.unwrap_or(peers.len());
        if prompt.is_none() && text.is_none() {
            return Err("--prompt or --text is required".to_owned());
        }
        if text.is_some() && tokenizer.is_none() {
            return Err("--text needs --tokenizer".to_owned());
        }
        if dump.is_some() && cuts == 0 {
            return Err("--dump-logits writes the --cuts rows; give --cuts N".to_owned());
        }
        Ok(Self {
            model: model.ok_or_else(|| "--model is required".to_owned())?,
            converted: converted.ok_or_else(|| "--converted is required".to_owned())?,
            rank: rank.ok_or_else(|| "--rank is required".to_owned())?,
            world,
            peers,
            context,
            prompt: prompt.unwrap_or_default(),
            decode,
            passes,
            cuts,
            dump,
            tokenizer,
            text,
            chat,
        })
    }
}

/// Runs `sparkflash inkling-tp4` with the arguments that followed the subcommand.
pub fn run(args: impl Iterator<Item = String>) -> ExitCode {
    let args = match Args::parse(args) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("sparkflash inkling-tp4: {error}");
            eprintln!(
                "usage: sparkflash inkling-tp4 --model DIR --converted DIR --rank R --world W \
                 --peer ADDR,ADDR --context N --prompt ID,ID --decode N"
            );
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
fn execute(args: &Args) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::Args;

    fn parse(extra: &[&'static str]) -> Result<Args, String> {
        let mut argv = vec![
            "--model",
            "/m",
            "--converted",
            "/c",
            "--rank",
            "1",
            "--peer",
            "10.0.0.1:9,10.0.0.2:9",
        ];
        argv.extend_from_slice(extra);
        Args::parse(argv.into_iter().map(str::to_owned))
    }

    #[test]
    fn a_run_names_its_rank_and_its_peers() {
        let args = parse(&["--prompt", "5,6,7"]).expect("the run parses");
        assert_eq!(args.rank, 1);
        assert_eq!(args.world, 2, "the world defaults to the address list");
        assert_eq!(args.prompt, vec![5, 6, 7]);
        assert_eq!(args.decode, 8);
        assert_eq!(args.peers.len(), 2);
    }

    #[test]
    fn every_required_flag_is_named() {
        let error = parse(&[]).expect_err("no prompt");
        assert!(error.contains("--prompt or --text is required"), "{error}");
        let error = parse(&["--text", "hi"]).expect_err("text without a tokenizer");
        assert!(error.contains("--text needs --tokenizer"), "{error}");
        let argv: Vec<String> = vec![
            "--peer".into(),
            "10.0.0.1:9".into(),
            "--prompt".into(),
            "1".into(),
        ];
        let error = Args::parse(argv.into_iter()).expect_err("no model");
        assert!(error.contains("--model is required"), "{error}");
    }
}
