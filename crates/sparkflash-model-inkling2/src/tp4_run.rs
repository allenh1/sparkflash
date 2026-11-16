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

//! One rank of a four-host run: the device arm, its converted file and its transport.
//!
//! The device arm is written a rank at a time, so a run is four processes that agree on the world,
//! the address list and the prompt. This module is the whole of that agreement: it opens the
//! rank's converted file, connects the collective in rank order, walks the prompt and the requested
//! decode steps, and reports what the rank saw. Nothing here is model arithmetic - the schedule and
//! the kernels are [crate::device]'s - so a launcher, a test and the CLI all run the same code.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sparkflash_comm::tcp::TcpConfig;
use sparkflash_model::backend::{ModelBackend, ModelSession, Sampling, Speculation, StepOutput};
use sparkflash_model::{ModelDescriptor, TpPlan};
use sparkflash_weights::tp::TpLayout;

use crate::collective::connect_rank;
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device::InklingDeviceBackend;

/// What one rank of the run is told.
#[derive(Debug, Clone)]
pub struct Tp4Run {
    /// This process's rank.
    pub rank: usize,
    /// The world, which is the length of [Self::addresses].
    pub world: usize,
    /// Every rank's address, in rank order: element zero is the accepting side.
    pub addresses: Vec<SocketAddr>,
    /// This rank's converted file.
    pub converted: PathBuf,
    /// The released checkpoint's directory, which holds the DSpark drafter's file the rank files
    /// do not; read only when the rank drafts with it (SPARKFLASH_INKLING_DRAFT=dspark).
    pub checkpoint: PathBuf,
    /// The admitted context, in tokens.
    pub context: usize,
    /// How many times the prompt is walked, each in a fresh session on the one open rank: the
    /// first walk pays the warm-up a served request after the first does not.
    pub passes: usize,
    /// Prefix lengths scored after the passes: the prompt's first `len * i / cuts` tokens for i in
    /// 1..=cuts, each prefilled in a fresh session, so two prefill walks can be compared on the
    /// logits they give at the same positions of the same text. Zero scores nothing.
    pub cuts: usize,
    /// Where rank 0 writes the scored logit rows: f32 little-endian, one row a cut, in cut order.
    pub dump: Option<PathBuf>,
    /// After the passes, the last pass's session is handed the prompt and then its own first
    /// `rewind` ids, a second prompt that keeps the first and parts from the reply after it, and
    /// a session that holds the prompt with no reply after it and a fresh session the same: the
    /// prompt-boundary rewind against the held prompt extended and against a cold prefill
    /// ([Tp4Outcome::rewind]). Zero checks nothing.
    pub rewind: usize,
}

/// A second prompt on the last pass's session against the same prompt on a fresh one.
#[derive(Debug, Clone)]
pub struct RewindCheck {
    /// The second prompt's length.
    pub tokens: usize,
    /// The ids the kept session reported it already held: the first prompt's length when it
    /// rewound, none when it started over.
    pub cached: usize,
    /// The ids a third session held when it was handed the second prompt straight after the
    /// first, with no reply between: the first prompt's length.
    pub extended_cached: usize,
    /// The kept session's prefill wall time, and the fresh one's.
    pub rewound_ms: u128,
    pub cold_ms: u128,
    /// The largest difference between the rewound and the cold prefill's logits; zero when they
    /// are the same bits.
    pub max_difference: f32,
    /// Whether the rewound and the cold prefill gave the same bits. A walk whose bits depend on
    /// where a chunk starts gives other bits from a boundary than from the start, as any prefix
    /// reuse does.
    pub logits_equal: bool,
    /// Whether the rewound prefill gave the bits of the extended one, which forwards the same
    /// ids from the same boundary: a rewind that restores the boundary exactly gives them.
    pub extended_equal: bool,
    /// Each session's ids and decode digest from the second prompt on.
    pub rewound_ids: Vec<u32>,
    pub extended_ids: Vec<u32>,
    pub cold_ids: Vec<u32>,
    pub rewound_digest: u64,
    pub extended_digest: u64,
    pub cold_digest: u64,
}

/// One scored prefix.
#[derive(Debug, Clone, Copy)]
pub struct Score {
    /// The prefix's length in tokens.
    pub length: usize,
    /// The logits' best entry and its value.
    pub best: (usize, f32),
    /// The token the text continues with and its logit, when the prefix is not the whole prompt.
    pub next: Option<(u32, f32)>,
}

/// The first greatest entry of a logit row.
fn best_of(logits: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (index, value) in logits.iter().enumerate() {
        if *value > best.1 {
            best = (index, *value);
        }
    }
    best
}

/// FNV-1a over a logit row's little-endian bytes, folded into `digest`.
fn digest_logits(digest: u64, logits: &[f32]) -> u64 {
    logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .fold(digest, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// The FNV-1a offset basis: the digest of no logits.
const DIGEST_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// What one rank of the run saw.
#[derive(Debug, Clone)]
pub struct Tp4Outcome {
    /// This process's rank.
    pub rank: usize,
    /// The boot line's description of the arm.
    pub banner: String,
    /// The prompt the rank forwarded.
    pub prompt: Vec<u32>,
    /// The tokens every rank chose, in order: the prompt's last step and then one a decode.
    pub generated: Vec<u32>,
    /// Each pass's prefill wall time, in milliseconds.
    pub prefill_ms: Vec<u128>,
    /// The last pass's decode step wall times, in milliseconds.
    pub decode_ms: Vec<u128>,
    /// The first logits' width, which is the released unpadded vocabulary.
    pub logits: usize,
    /// The first logits' best entry and its value, the comparison a second rank can make.
    pub best: (usize, f32),
    /// The last pass's decode logits, every step's row in order, as one FNV-1a digest: two builds
    /// that decode the same bits print the same digest, where the generated ids only show that
    /// each step's best entry agreed.
    pub decode_digest: u64,
    /// The last pass's drafted rounds as its session counted them: the rounds settled by the end
    /// of the decode steps, so a round still being fed at the last step is not in it.
    pub speculation: Speculation,
    /// The scored prefixes ([Tp4Run::cuts]), in cut order.
    pub scores: Vec<Score>,
    /// The rewind check ([Tp4Run::rewind]), when one was asked for.
    pub rewind: Option<RewindCheck>,
}

impl Tp4Run {
    /// The greedy sampler: a run's four ranks must choose the same token, and a distribution's draw
    /// would need a shared stream.
    #[must_use]
    pub fn greedy() -> Sampling {
        Sampling {
            temperature: 0.0,
            top_k: 1,
            ..Sampling::default()
        }
    }

    /// The run's identity checks, before a 97 GB file is opened.
    ///
    /// # Errors
    /// A world that is not the address list's length, a rank outside the world, or an empty list.
    pub fn check(&self) -> Result<(), String> {
        if self.world == 0 {
            return Err("the world is zero".to_owned());
        }
        if self.addresses.len() != self.world {
            return Err(format!(
                "TP{}/rank{}: {} address(es) were given for a world of {}",
                self.world,
                self.rank,
                self.addresses.len(),
                self.world
            ));
        }
        if self.rank >= self.world {
            return Err(format!(
                "rank {} is outside a world of {}",
                self.rank, self.world
            ));
        }
        if self.passes == 0 {
            return Err("a run walks the prompt at least once".to_owned());
        }
        Ok(())
    }

    /// Opens the rank, walks the prompt and the decode steps, and reports the run.
    ///
    /// # Errors
    /// A rank file that does not open, a transport that does not connect, or a device failure. Every
    /// rank of the world has to be in this function at the same time: the collectives are inside the
    /// walk.
    pub fn execute(
        &self,
        descriptor: &ModelDescriptor,
        config: &Inkling2Config,
        prompt: &[u32],
        decode: usize,
    ) -> Result<Tp4Outcome, String> {
        self.check()?;
        if prompt.is_empty() {
            return Err("the prompt is empty".to_owned());
        }
        let layout = TpLayout {
            rank: self.rank,
            world_size: self.world,
        };
        let source = RankFileSource::open(&self.converted, config, layout)
            .map_err(|error| format!("{}: {error}", self.converted.display()))?;
        // Every rank loads ~97 GB before its first collective and the load times differ by
        // minutes, so the frame deadline is the run's patience, not a per-token budget: a rank that
        // reads a slow file must not look like a dead peer to the three that are ready.
        let transport = TcpConfig::with_addresses(self.rank, self.addresses.clone())
            .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?
            .with_timeout(Duration::from_secs(900))
            .with_connect_timeout(Duration::from_secs(180));
        let exchange = connect_rank(
            &transport,
            2 * config.hidden_size,
            &format!(
                "{} TP{} context {}",
                descriptor.id, self.world, self.context
            ),
        )
        .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?;
        let tp = TpPlan::new(self.world, self.rank)
            .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?;
        let opened = Instant::now();
        let mut backend = InklingDeviceBackend::open_rank_file(
            descriptor.clone(),
            config.clone(),
            &source,
            self.context,
            tp,
            exchange,
        )
        .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?;
        let drafter = if backend.wants_drafter() {
            let bytes = backend.load_drafter(&self.checkpoint).map_err(|error| {
                format!(
                    "TP{}/rank{}: the DSpark drafter: {error}",
                    self.world, self.rank
                )
            })?;
            format!(", DSpark drafter {bytes} bytes")
        } else {
            String::new()
        };
        let banner = format!(
            "TP{}/rank{} {} on {} ({} weight bytes{drafter}, opened in {} ms)",
            self.world,
            self.rank,
            descriptor.id,
            self.converted.display(),
            backend.weight_bytes(),
            opened.elapsed().as_millis()
        );
        let mut prefill_ms = Vec::with_capacity(self.passes);
        let mut walked = None;
        let mut kept = None;
        for pass in 0..self.passes {
            // The last pass's session is dropped before the next opens: at a long context two
            // sessions' caches do not fit a rank.
            drop(kept.take());
            let mut session = self.open(&backend)?;
            let started = Instant::now();
            let first = self.prefill(session.as_mut(), prompt, "prefill")?;
            prefill_ms.push(started.elapsed().as_millis());
            let best = best_of(&first.logits);
            let (generated, decode_ms, decode_digest) =
                self.decode_steps(session.as_mut(), &first, decode)?;
            // Every pass walks the same ids greedily, so a pass that chose differently is a state
            // leak between sessions, which is a bug and not a report.
            if let Some((earlier, _, _, _, _, _)) = &walked
                && *earlier != generated
            {
                return Err(format!(
                    "TP{}/rank{}: pass {pass} generated {generated:?}, the first pass {earlier:?}",
                    self.world, self.rank
                ));
            }
            walked = Some((
                generated,
                decode_ms,
                first.logits.len(),
                best,
                decode_digest,
                session.speculation(),
            ));
            kept = Some(session);
        }
        let (generated, decode_ms, logits, best, decode_digest, speculation) =
            walked.expect("a run walks at least once");
        let rewind = match kept.take() {
            Some(session) if self.rewind > 0 => {
                Some(self.rewind_check(&backend, session, prompt, decode)?)
            }
            _ => None,
        };
        let mut scores = Vec::with_capacity(self.cuts);
        let mut rows: Vec<u8> = Vec::new();
        for cut in 1..=self.cuts {
            let length = (prompt.len() * cut / self.cuts).max(1);
            let mut session = backend.open_session().map_err(|error| {
                format!("TP{}/rank{}: {}", self.world, self.rank, error.message())
            })?;
            session.set_sampling(Self::greedy());
            let scored = session.prefill(&prompt[..length]).map_err(|error| {
                format!(
                    "TP{}/rank{} score {length}: {}",
                    self.world,
                    self.rank,
                    error.message()
                )
            })?;
            let next = prompt.get(length).map(|&id| {
                (
                    id,
                    scored.logits.get(id as usize).copied().unwrap_or(f32::NAN),
                )
            });
            scores.push(Score {
                length,
                best: best_of(&scored.logits),
                next,
            });
            if self.rank == 0 && self.dump.is_some() {
                rows.extend(scored.logits.iter().flat_map(|value| value.to_le_bytes()));
            }
        }
        if self.rank == 0
            && let Some(path) = &self.dump
        {
            std::fs::write(path, &rows)
                .map_err(|error| format!("--dump-logits {}: {error}", path.display()))?;
        }
        Ok(Tp4Outcome {
            rank: self.rank,
            banner,
            prompt: prompt.to_vec(),
            generated,
            prefill_ms,
            decode_ms,
            logits,
            best,
            decode_digest,
            speculation,
            scores,
            rewind,
        })
    }

    /// A fresh greedy session on `backend`.
    fn open(&self, backend: &InklingDeviceBackend) -> Result<Box<dyn ModelSession>, String> {
        let mut session = backend
            .open_session()
            .map_err(|error| format!("TP{}/rank{}: {}", self.world, self.rank, error.message()))?;
        session.set_sampling(Self::greedy());
        Ok(session)
    }

    /// `tokens` prefilled on `session`, a failure named by `what`.
    fn prefill(
        &self,
        session: &mut dyn ModelSession,
        tokens: &[u32],
        what: &str,
    ) -> Result<StepOutput, String> {
        session.prefill(tokens).map_err(|error| {
            format!(
                "TP{}/rank{} {what}: {}",
                self.world,
                self.rank,
                error.message()
            )
        })
    }

    /// `decode` greedy steps after `first`: the ids chosen (the prompt's and each step's), each
    /// step's wall time, and the steps' logits as one digest.
    fn decode_steps(
        &self,
        session: &mut dyn ModelSession,
        first: &StepOutput,
        decode: usize,
    ) -> Result<(Vec<u32>, Vec<u128>, u64), String> {
        let mut generated = Vec::with_capacity(decode + 1);
        let mut next = first.next_token.ok_or_else(|| {
            format!(
                "TP{}/rank{}: the prefill chose no token",
                self.world, self.rank
            )
        })?;
        generated.push(next);
        let mut decode_ms = Vec::with_capacity(decode);
        let mut digest = DIGEST_BASIS;
        for _ in 0..decode {
            let started = Instant::now();
            let step = session.decode(next).map_err(|error| {
                format!(
                    "TP{}/rank{} decode: {}",
                    self.world,
                    self.rank,
                    error.message()
                )
            })?;
            decode_ms.push(started.elapsed().as_millis());
            digest = digest_logits(digest, &step.logits);
            next = step.next_token.ok_or_else(|| {
                format!(
                    "TP{}/rank{}: a decode chose no token",
                    self.world, self.rank
                )
            })?;
            generated.push(next);
        }
        Ok((generated, decode_ms, digest))
    }

    /// The rewind check ([Self::rewind]): `kept`, which holds `prompt` and its reply, is handed
    /// the prompt and then the prompt's first ids, then dropped; a fresh session is handed the
    /// prompt and then the same second prompt, with no reply between; and a fresh session is
    /// handed the second prompt alone.
    fn rewind_check(
        &self,
        backend: &InklingDeviceBackend,
        mut kept: Box<dyn ModelSession>,
        prompt: &[u32],
        decode: usize,
    ) -> Result<RewindCheck, String> {
        let mut second = prompt.to_vec();
        second.extend_from_slice(&prompt[..self.rewind.min(prompt.len())]);
        kept.set_sampling(Self::greedy());
        let started = Instant::now();
        let rewound = self.prefill(kept.as_mut(), &second, "rewound prefill")?;
        let rewound_ms = started.elapsed().as_millis();
        let cached = kept.cached_tokens();
        let (rewound_ids, _, rewound_digest) =
            self.decode_steps(kept.as_mut(), &rewound, decode)?;
        drop(kept);
        // The second prompt extends the held first, so it forwards the same ids from the same
        // boundary: a rewind that restores the boundary's state exactly gives these bits.
        let mut extended_session = self.open(backend)?;
        extended_session.set_sampling(Self::greedy());
        self.prefill(extended_session.as_mut(), prompt, "extended first prefill")?;
        let extended = self.prefill(extended_session.as_mut(), &second, "extended prefill")?;
        let extended_cached = extended_session.cached_tokens();
        let (extended_ids, _, extended_digest) =
            self.decode_steps(extended_session.as_mut(), &extended, decode)?;
        drop(extended_session);
        let mut cold_session = self.open(backend)?;
        let started = Instant::now();
        let cold = self.prefill(cold_session.as_mut(), &second, "cold prefill")?;
        let cold_ms = started.elapsed().as_millis();
        let (cold_ids, _, cold_digest) = self.decode_steps(cold_session.as_mut(), &cold, decode)?;
        let same_bits = |a: &StepOutput, b: &StepOutput| {
            a.logits.len() == b.logits.len()
                && a.logits
                    .iter()
                    .zip(&b.logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        let max_difference = rewound
            .logits
            .iter()
            .zip(&cold.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        Ok(RewindCheck {
            tokens: second.len(),
            cached,
            extended_cached,
            rewound_ms,
            cold_ms,
            max_difference,
            logits_equal: same_bits(&rewound, &cold),
            extended_equal: same_bits(&rewound, &extended),
            rewound_ids,
            extended_ids,
            cold_ids,
            rewound_digest,
            extended_digest,
            cold_digest,
        })
    }
}

/// The converted file a rank of a world reads, from the directory the release was converted into.
///
/// # Errors
/// A directory that does not hold that rank's file.
pub fn rank_file(directory: &Path, rank: usize, world: usize) -> Result<PathBuf, String> {
    let path = directory.join(format!("rank-{rank}-of-{world}.bin"));
    if path.is_file() {
        return Ok(path);
    }
    Err(format!(
        "{}: no converted file for TP{}/rank{rank} in {}",
        path.display(),
        world,
        directory.display()
    ))
}

/// The subcommand's usage line.
pub const TP4_USAGE: &str = "usage: sparkflash inkling-tp4 --model DIR --converted DIR --rank R \
                               --world W --peer ADDR,ADDR --context N --prompt ID,ID --decode N";

/// One parsed `sparkflash inkling-tp4` invocation: the CLI only prints what [Tp4Run] reports, so
/// the arguments parse here beside the run they configure.
#[derive(Debug, Clone)]
pub struct Tp4Args {
    /// The checkpoint directory (`--model`): its config.
    pub model: PathBuf,
    /// The converted rank files' directory (`--converted`).
    pub converted: PathBuf,
    /// This process's rank (`--rank`).
    pub rank: usize,
    /// The world (`--world`), by default the address list's length.
    pub world: usize,
    /// Every rank's address in rank order (`--peer`).
    pub peers: Vec<SocketAddr>,
    /// The admitted context (`--context`, 4,096 by default).
    pub context: usize,
    /// The prompt's ids (`--prompt`), when no `--text` is given.
    pub prompt: Vec<u32>,
    /// Decode steps after the prompt (`--decode`, 8 by default).
    pub decode: usize,
    /// How many times the prompt is walked ([Tp4Run::passes], `--passes`).
    pub passes: usize,
    /// Prefix lengths scored after the passes ([Tp4Run::cuts], `--cuts`).
    pub cuts: usize,
    /// Where rank 0 writes the scored rows ([Tp4Run::dump], `--dump-logits`).
    pub dump: Option<PathBuf>,
    /// The rewind check's second-prompt ids ([Tp4Run::rewind], `--rewind`).
    pub rewind: usize,
    /// The tokenizer for `--text` (`--tokenizer`).
    pub tokenizer: Option<PathBuf>,
    /// A sentence to tokenize instead of `--prompt` (`--text`).
    pub text: Option<String>,
    /// Whether `--text` is rendered through the chat template (`--chat`).
    pub chat: bool,
}

impl Tp4Args {
    /// Parses the arguments that followed the subcommand.
    ///
    /// # Errors
    /// An unknown flag, a value that does not parse, or a missing required flag, named.
    pub fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
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
        let mut rewind = 0usize;
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
                // After the passes, the last pass's session is handed the prompt and its own first
                // N ids, and a fresh session the same: a rewind to the prompt's end against a cold
                // prefill, bit for bit.
                "--rewind" => {
                    rewind = value()?
                        .parse::<usize>()
                        .map_err(|error| format!("--rewind: {error}"))?;
                }
                "--chat" => chat = true,
                "--tokenizer" => tokenizer = Some(PathBuf::from(value()?)),
                "--text" => text = Some(value()?),
                // The same text from a file at the same path on every rank: a long prompt is past
                // what one argument of a command line may hold (128 KiB on Linux).
                "--text-file" => {
                    let path = value()?;
                    text = Some(
                        std::fs::read_to_string(&path)
                            .map_err(|error| format!("--text-file {path}: {error}"))?,
                    );
                }
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
            return Err("--prompt, --text or --text-file is required".to_owned());
        }
        if text.is_some() && tokenizer.is_none() {
            return Err("--text and --text-file need --tokenizer".to_owned());
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
            rewind,
            tokenizer,
            text,
            chat,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Tp4Args, Tp4Run};

    fn address() -> std::net::SocketAddr {
        "127.0.0.1:29623".parse().expect("an address")
    }

    fn run(world: usize, rank: usize, addresses: usize) -> Tp4Run {
        Tp4Run {
            rank,
            world,
            addresses: vec![address(); addresses],
            converted: std::path::PathBuf::from("/nonexistent"),
            checkpoint: std::path::PathBuf::from("/nonexistent-checkpoint"),
            context: 16,
            passes: 1,
            cuts: 0,
            dump: None,
            rewind: 0,
        }
    }

    #[test]
    fn a_world_must_match_its_address_list() {
        assert!(run(4, 0, 4).check().is_ok());
        let short = run(4, 0, 3)
            .check()
            .expect_err("three addresses for a world of four");
        assert!(short.contains("3 address(es)"), "{short}");
        let outside = run(4, 4, 4).check().expect_err("rank beyond the world");
        assert!(outside.contains("outside a world of 4"), "{outside}");
        let empty = run(0, 0, 0).check().expect_err("a world of nothing");
        assert!(empty.contains("zero"), "{empty}");
        let idle = Tp4Run {
            passes: 0,
            ..run(4, 0, 4)
        }
        .check()
        .expect_err("no pass");
        assert!(idle.contains("at least once"), "{idle}");
    }

    #[test]
    fn the_missing_rank_file_names_itself() {
        let directory = std::path::PathBuf::from("/nonexistent-converted");
        let error = super::rank_file(&directory, 2, 4).expect_err("the file is not there");
        assert!(error.contains("rank-2-of-4.bin"), "{error}");
    }

    fn parse(extra: &[&'static str]) -> Result<Tp4Args, String> {
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
        Tp4Args::parse(argv.into_iter().map(str::to_owned))
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
    fn a_text_file_is_the_text_flag_read_from_a_file() {
        let path = std::env::temp_dir().join(format!("inkling-text-{}.txt", std::process::id()));
        std::fs::write(&path, "a long\nprompt\n").expect("write the text file");
        let argv = [
            "--model",
            "/m",
            "--converted",
            "/c",
            "--rank",
            "1",
            "--peer",
            "10.0.0.1:9",
            "--tokenizer",
            "/t",
            "--text-file",
        ]
        .into_iter()
        .map(str::to_owned)
        .chain([path.display().to_string()]);
        let args = Tp4Args::parse(argv).expect("the run parses");
        std::fs::remove_file(&path).expect("remove the text file");
        assert_eq!(args.text.as_deref(), Some("a long\nprompt\n"));
        let error =
            parse(&["--tokenizer", "/t", "--text-file", "/nonexistent-text"]).expect_err("no file");
        assert!(error.contains("--text-file /nonexistent-text"), "{error}");
    }

    #[test]
    fn every_required_flag_is_named() {
        let error = parse(&[]).expect_err("no prompt");
        assert!(
            error.contains("--prompt, --text or --text-file is required"),
            "{error}"
        );
        let error = parse(&["--text", "hi"]).expect_err("text without a tokenizer");
        assert!(error.contains("--text-file need --tokenizer"), "{error}");
        let argv: Vec<String> = vec![
            "--peer".into(),
            "10.0.0.1:9".into(),
            "--prompt".into(),
            "1".into(),
        ];
        let error = Tp4Args::parse(argv.into_iter()).expect_err("no model");
        assert!(error.contains("--model is required"), "{error}");
    }
}
