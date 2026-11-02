//! `device-chain --run-stage glm-tp4`: one GLM-5.3-Flash tensor-parallel rank on this host's GPU,
//! over the TCP collective tp4-host uses, decoding greedily from a token prompt.
//!
//! Every rank of the world runs the command with the same prompt and stream counts and loads its
//! own slice once, from its converted rank file or from the checkpoint. For each stream count in
//! turn, each stream is a sequence of the same prompt: the rank prefills them together, then
//! decodes them a token each a step, and times every step. A batch row is the same bits as the row
//! alone, so the streams must agree token for token and logit for logit, and stream 0 must decode
//! the same bits at every count. The rank prints its tokens and the digest of its last logits,
//! which every rank of a world must print the same.
//!
//! `--collective nccl` gives the TCP collective the run's NCCL communicator as its device leg
//! ([crate::tp4_reduce::DeviceExchange]): the partials and the logits are exchanged between the
//! devices and summed there in Wire::F32's order, the bits of the TCP run.
//!
//! `--draft N` decodes speculatively: each round the MTP layer drafts N tokens after each
//! stream's next one, one verify step runs them all, and each stream keeps the drafts greedy
//! would have picked and the token after them. Each draft count in the list runs every stream
//! count, and every run must decode the bits of the first, so a list with 0 checks speculative
//! decoding against plain greedy. `--dflash2 DIR` drafts with the DFlash 2 block drafter in DIR
//! instead, sharded over the ranks as the model is, each scoring its slice of the head
//! ([DeviceRank::load_block_drafter]). `--draft-oracle` drafts the tokens the first run (which
//! must draft none) decoded instead, which every verify must keep to the last: the check that
//! each verify row of one sequence, at every depth, is the bits of the step it stands in for.
//! A speculative run reports how many drafts each stream kept a round, from none to all.
//! `--depth adaptive` verifies only as many of the block drafter's drafts as the shared
//! [DepthPolicy] judges worth their rows, from the drafter's confidence in each, calibrated by
//! the stream's acceptances, against [DFLASH2_STEP_COST]; every rank sees the same confidences
//! and so verifies the same rows.

use std::net::IpAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sparkflash_comm::device_reduce::BootstrapConfig;
use sparkflash_comm::{TcpTransport, Transport};
use sparkflash_device::converted::find_converted_rank_file;
use sparkflash_engine::parallel_tp::{BitGatherHandle, TpCollective, TpLayout, TpRuntime, Wire};
use sparkflash_engine::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model::DepthPolicy;
use sparkflash_model_glm5_next::device::CudaHandle;
use sparkflash_model_glm5_next::{
    DFLASH2_STEP_COST, DeviceRank, DeviceSequence, Glm5NextConfig, MAX_ROWS,
};

use crate::tp4_host::{logits_sha256, parse_list, tcp_config, tp4_seams};
use crate::tp4_reduce::{DeviceExchange, open_run_reduce};

/// The stage's arguments.
pub const USAGE: &str = "sparkflash device-chain --run-stage glm-tp4 --tp-rank N --tp-world-size N --comm-peers IP,IP,... --comm-base-port 29701 (--converted DIR | --checkpoint DIR) [--config FILE] (--prompt TEXT --tokenizer PATH | --prompt-tokens 3,7,11) [--tokenizer PATH] [--max-tokens 16] [--streams 1,2,4,8] [--wire f32|bf16] [--collective tcp|nccl] [--draft 0,3] [--dflash2 DIR [--depth fixed|adaptive] | --draft-oracle]; the weights come from DIR's rank-R-of-W.bin or the checkpoint, the configuration from FILE, the checkpoint's config.json, or the released one; each stream count runs in turn on the one load, and a tokenizer also decodes what the rank generates; nccl exchanges the partials on the devices and needs the f32 wire and a world of 2 or more; each draft count past 0 decodes speculatively with the MTP layer, or with DIR's DFlash 2 drafter, or as the first run (of 0 drafts) decoded, which every verify must keep, a stream's drafts and its next token one verify of at most 8 rows a step; adaptive verifies only the block drafts its confidence judges worth their rows";

/// One `flag`'s numeric value.
fn number<T: std::str::FromStr>(flag: &str, text: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    text.parse().map_err(|error| format!("{flag}: {error}"))
}

#[derive(Debug)]
struct GlmArgs {
    layout: TpLayout,
    peers: Vec<IpAddr>,
    base_port: u16,
    converted: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    config: Option<PathBuf>,
    prompt: Vec<u32>,
    prompt_text: Option<String>,
    tokenizer: Option<PathBuf>,
    max_tokens: usize,
    streams: Vec<usize>,
    /// Draft counts, each run in turn; 0 decodes without drafting.
    drafts: Vec<usize>,
    /// The DFlash 2 drafter to draft with in place of the MTP layer.
    dflash2: Option<PathBuf>,
    /// Draft the first run's tokens in place of a drafter, and refuse any the verify rejects.
    oracle: bool,
    /// Verify as many of the block drafter's drafts as [DepthPolicy] judges worth it, not all.
    adaptive: bool,
    wire: Wire,
    /// The NCCL communicator as the collective's device leg.
    nccl: bool,
}

impl GlmArgs {
    fn parse(args: &[&str]) -> Result<Self, String> {
        let (mut rank, mut world) = (0, 1);
        let mut parsed = Self {
            layout: TpLayout {
                rank: 0,
                world_size: 1,
            },
            peers: vec![IpAddr::from([127, 0, 0, 1])],
            base_port: 29701,
            converted: None,
            checkpoint: None,
            config: None,
            prompt: Vec::new(),
            prompt_text: None,
            tokenizer: None,
            max_tokens: 16,
            streams: vec![1],
            drafts: vec![0],
            dflash2: None,
            oracle: false,
            adaptive: false,
            wire: Wire::F32,
            nccl: false,
        };
        let mut index = 0;
        while index < args.len() {
            let flag = args[index];
            if flag == "--draft-oracle" {
                parsed.oracle = true;
                index += 1;
                continue;
            }
            let mut take = || -> Result<&str, String> {
                index += 1;
                args.get(index)
                    .copied()
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag {
                "--tp-rank" => rank = number(flag, take()?)?,
                "--tp-world-size" => world = number(flag, take()?)?,
                "--comm-peers" => parsed.peers = parse_list(flag, take()?)?,
                "--comm-base-port" => parsed.base_port = number(flag, take()?)?,
                "--converted" => parsed.converted = Some(PathBuf::from(take()?)),
                "--checkpoint" => parsed.checkpoint = Some(PathBuf::from(take()?)),
                "--config" => parsed.config = Some(PathBuf::from(take()?)),
                "--prompt-tokens" => parsed.prompt = parse_list(flag, take()?)?,
                "--prompt" => parsed.prompt_text = Some(take()?.to_owned()),
                "--tokenizer" => parsed.tokenizer = Some(PathBuf::from(take()?)),
                "--max-tokens" => parsed.max_tokens = number(flag, take()?)?,
                "--streams" => parsed.streams = parse_list(flag, take()?)?,
                "--draft" => parsed.drafts = parse_list(flag, take()?)?,
                "--dflash2" => parsed.dflash2 = Some(PathBuf::from(take()?)),
                "--depth" => {
                    parsed.adaptive = match take()? {
                        "fixed" => false,
                        "adaptive" => true,
                        other => return Err(format!("--depth {other}: fixed or adaptive")),
                    }
                }
                "--wire" => {
                    parsed.wire = match take()? {
                        "f32" => Wire::F32,
                        "bf16" => Wire::Bf16,
                        other => return Err(format!("--wire {other}: f32 or bf16")),
                    }
                }
                "--collective" => {
                    parsed.nccl = match take()? {
                        "tcp" => false,
                        "nccl" => true,
                        other => return Err(format!("--collective {other}: tcp or nccl")),
                    }
                }
                other => return Err(format!("unknown argument {other}")),
            }
            index += 1;
        }
        if parsed.converted.is_some() == parsed.checkpoint.is_some() {
            return Err("exactly one of --converted and --checkpoint is required".to_owned());
        }
        if parsed.peers.len() != world {
            return Err(format!(
                "--comm-peers names {} addresses for --tp-world-size {world}",
                parsed.peers.len()
            ));
        }
        if rank >= world {
            return Err(format!(
                "--tp-rank {rank} is outside --tp-world-size {world}"
            ));
        }
        if parsed.prompt.is_empty() == parsed.prompt_text.is_none() {
            return Err("exactly one of --prompt and --prompt-tokens is required".to_owned());
        }
        if parsed.prompt_text.is_some() && parsed.tokenizer.is_none() {
            return Err("--prompt needs --tokenizer PATH".to_owned());
        }
        if parsed.streams.contains(&0) || parsed.max_tokens == 0 {
            return Err("--streams and --max-tokens must be at least 1".to_owned());
        }
        // A round's verify is one step: every stream's next token and its drafts.
        for &drafts in parsed.drafts.iter().filter(|&&drafts| drafts > 0) {
            if let Some(&streams) = parsed
                .streams
                .iter()
                .find(|&&streams| streams * (drafts + 1) > MAX_ROWS)
            {
                return Err(format!(
                    "--draft {drafts} with {streams} streams verifies {} rows a step; a verify is \
                     at most {MAX_ROWS}",
                    streams * (drafts + 1)
                ));
            }
        }
        if parsed.dflash2.is_some() && parsed.drafts.iter().all(|&drafts| drafts == 0) {
            return Err("--dflash2 drafts nothing without a --draft count past 0".to_owned());
        }
        // Only the block drafter says how sure it is of each draft.
        if parsed.adaptive && parsed.dflash2.is_none() {
            return Err(
                "--depth adaptive weighs the block drafter's confidences; it needs --dflash2"
                    .to_owned(),
            );
        }
        if parsed.oracle {
            if parsed.dflash2.is_some() {
                return Err("--draft-oracle drafts in place of --dflash2's drafter".to_owned());
            }
            if parsed.drafts.first() != Some(&0) || parsed.drafts.iter().all(|&d| d == 0) {
                return Err(
                    "--draft-oracle drafts the first run's tokens: --draft starts at 0 and has a \
                     count past it"
                        .to_owned(),
                );
            }
        }
        // The device leg sums as Wire::F32 does and has nothing to exchange alone.
        if parsed.nccl && (parsed.wire != Wire::F32 || world < 2) {
            return Err(
                "--collective nccl needs --wire f32 and a --tp-world-size of 2 or more".to_owned(),
            );
        }
        parsed.layout = TpLayout {
            rank,
            world_size: world,
        };
        Ok(parsed)
    }

    /// `--config`, else the checkpoint's `config.json`, else the released configuration.
    fn config(&self) -> Result<Glm5NextConfig, String> {
        let path = self
            .config
            .clone()
            .or_else(|| self.checkpoint.as_ref().map(|dir| dir.join("config.json")));
        match path {
            Some(path) => {
                let text = std::fs::read_to_string(&path)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                Glm5NextConfig::from_json(&text)
                    .map_err(|error| format!("{}: {error}", path.display()))
            }
            None => Glm5NextConfig::glm53_flash().map_err(|error| error.to_string()),
        }
    }
}

/// Entry point for `--run-stage glm-tp4`.
pub fn run(args: &[&str]) -> ExitCode {
    match GlmArgs::parse(args).and_then(|parsed| run_rank(&parsed)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("glm-tp4: {error}");
            eprintln!("glm-tp4: usage: {USAGE}");
            ExitCode::FAILURE
        }
    }
}

/// The run's NCCL communicator over `transport`, opened on this thread's context. Every rank
/// agrees on the run (configuration, prompt, stream and draft counts, drafter and length) before
/// any enters NCCL.
fn open_nccl(
    cuda: &Cuda,
    transport: &TcpTransport,
    args: &GlmArgs,
    config: &Glm5NextConfig,
    prompt: &[u32],
) -> Result<sparkflash_comm::device_reduce::RunReduce, String> {
    let layout = args.layout;
    let run = format!(
        "glm-tp4;{config:?};{prompt:?};{:?};{};{:?};{}",
        args.streams,
        args.max_tokens,
        args.drafts,
        args.dflash2.is_some()
    );
    let agreement = BootstrapConfig {
        native_rounding: 0,
        env_toggles: 0,
        plan_digest: sparkflash_engine::synthetic::fnv1a(run.as_bytes()),
    };
    // The warm-up runs one row's partial through the communicator; its scratch is freed once the
    // open has synchronised it.
    let payload = config.hidden_size * 2;
    let mut allocated = 0usize;
    let scratch = cuda.alloc(payload * 2, &mut allocated)?;
    let opened = open_run_reduce(
        transport,
        layout.world_size,
        layout.rank,
        0,
        scratch.0,
        payload,
        agreement,
    );
    cuda.free_buffers(&[&scratch]);
    opened.map_err(|error| format!("nccl open: {error}"))
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Each stream's next token: its last row's pick.
fn greedy(picks: &[Vec<u32>]) -> Result<Vec<u32>, String> {
    picks
        .iter()
        .map(|rows| {
            rows.last()
                .copied()
                .ok_or_else(|| "a stream returned no rows".to_owned())
        })
        .collect()
}

fn run_rank(args: &GlmArgs) -> Result<(), String> {
    let layout = args.layout;
    let who = format!("rank {} of {}", layout.rank, layout.world_size);
    let config = args.config()?;
    let tokenizer = match &args.tokenizer {
        Some(path) => Some(
            sparkflash_model_deepseek_v41::encoding::load_hf_tokenizer(path)
                .map_err(|error| format!("tokenizer: {error}"))?,
        ),
        None => None,
    };
    let prompt = match (&args.prompt_text, &tokenizer) {
        (Some(text), Some(encoder)) => encoder
            .tokenizer()
            .encode(text.as_str(), false)
            .map_err(|error| format!("tokenize prompt: {error}"))?
            .get_ids()
            .to_vec(),
        _ => args.prompt.clone(),
    };
    if prompt.is_empty() {
        return Err("the prompt is no tokens".to_owned());
    }
    let cuda = Arc::new(CudaHandle::new(Cuda::load()?));

    let started = Instant::now();
    let mut rank = match (&args.converted, &args.checkpoint) {
        (Some(dir), _) => {
            let path = find_converted_rank_file(dir, layout.rank, layout.world_size)
                .ok_or_else(|| format!("{} holds no file for {who}", dir.display()))?;
            DeviceRank::load_converted(Arc::clone(&cuda), &config, &path, layout)
        }
        (None, Some(dir)) => {
            let source = SafeTensorsSource::open(dir, &dir.join(INDEX_NAME))
                .map_err(|error| format!("{}: {error}", dir.display()))?;
            DeviceRank::load(Arc::clone(&cuda), &config, &source, layout)
        }
        (None, None) => unreachable!("the parser requires one source"),
    }
    .map_err(|error| format!("load: {error}"))?;
    println!(
        "glm-tp4: {who}: loaded {:.2} GiB in {:.0} ms",
        rank.device_bytes() as f64 / f64::from(1 << 30),
        ms(started.elapsed())
    );
    if let Some(dir) = &args.dflash2 {
        let started = Instant::now();
        rank.load_block_drafter(dir)
            .map_err(|error| format!("load the drafter: {error}"))?;
        println!(
            "glm-tp4: {who}: loaded the block drafter {} in {:.0} ms",
            dir.display(),
            ms(started.elapsed())
        );
    }

    // Connected after the load, so every rank loads at once and none waits on a peer's load.
    let started = Instant::now();
    let mut reduce = None;
    let transport = if layout.world_size == 1 {
        None
    } else {
        let tcp = tcp_config(layout.rank, &args.peers, args.base_port, true)?;
        let transport =
            TcpTransport::connect(tcp).map_err(|error| format!("tcp connect: {error}"))?;
        if args.nccl {
            reduce = Some(open_nccl(&cuda, &transport, args, &config, &prompt)?);
        }
        Some(Box::new(transport) as Box<dyn Transport>)
    };
    let (mut collective, gather) = tp4_seams(layout.rank, layout.world_size, transport, args.wire);
    if let Some(reduce) = reduce {
        collective = Box::new(DeviceExchange::new(collective, reduce)?) as Box<dyn TpCollective>;
    }
    let tp = TpRuntime::new(layout, collective).map_err(|error| error.to_string())?;
    let gather = BitGatherHandle::new(gather);
    println!(
        "glm-tp4: {who}: {} collective up in {:.0} ms",
        tp.label(),
        ms(started.elapsed())
    );

    let mut first: Option<(String, Vec<u32>, String)> = None;
    let runs = args
        .drafts
        .iter()
        .flat_map(|&drafts| args.streams.iter().map(move |&streams| (drafts, streams)));
    for (drafts, streams) in runs {
        // The parser has the oracle's first run draft nothing, so its tokens are in by now.
        let drafter = match (&first, args.oracle, args.dflash2.is_some()) {
            (Some((_, tokens, _)), true, _) => Drafter::Oracle(tokens),
            (_, _, true) => Drafter::Block {
                adaptive: args.adaptive,
            },
            _ => Drafter::Mtp,
        };
        let (tokens, digest) = decode_streams(
            &mut rank,
            (&tp, &gather),
            &prompt,
            (streams, drafts, drafter),
            args.max_tokens,
            &who,
        )?;
        let run = format!("{streams} streams of {drafts} drafts");
        match &first {
            None => {
                println!("glm-tp4: {who}: prompt {prompt:?}");
                println!("glm-tp4: {who}: tokens {tokens:?}");
                if let Some(encoder) = &tokenizer {
                    let text = encoder
                        .tokenizer()
                        .decode(&tokens, false)
                        .map_err(|error| format!("detokenize: {error}"))?;
                    println!("glm-tp4: {who}: text {text:?}");
                }
                println!("glm-tp4: {who}: last logits sha256 {digest}");
                first = Some((run, tokens, digest));
            }
            Some((alone, expected, expected_digest)) => {
                if (&tokens, &digest) != (expected, expected_digest) {
                    return Err(format!(
                        "stream 0 of {run} decoded {tokens:?} with last logits {digest}, where \
                         stream 0 of {alone} decoded {expected:?} with {expected_digest}"
                    ));
                }
                println!("glm-tp4: {who}: {run} decode the bits of {alone}");
            }
        }
    }
    Ok(())
}

/// What drafts a speculative run's tokens.
#[derive(Debug, Clone, Copy)]
enum Drafter<'a> {
    /// The MTP layer.
    Mtp,
    /// The rank's DFlash 2 block drafter, verifying all its drafts or as many as each stream's
    /// [DepthPolicy] chooses (`adaptive`).
    Block { adaptive: bool },
    /// The tokens a plain greedy run decoded, every one of which a verify must keep.
    Oracle(&'a [u32]),
}

/// `streams` sequences of `prompt`, prefilled in one batch and decoded greedily for
/// `max_tokens`, a token each a step or with `drafts` speculatively from `drafter`, with the
/// times printed.
/// Returns stream 0's tokens and the digest of its last logits, once every stream has decoded
/// the same tokens and the same last logits.
fn decode_streams(
    rank: &mut DeviceRank,
    (tp, gather): (&TpRuntime, &BitGatherHandle),
    prompt: &[u32],
    (streams, drafts, drafter): (usize, usize, Drafter<'_>),
    max_tokens: usize,
    who: &str,
) -> Result<(Vec<u32>, String), String> {
    let capacity = prompt.len() + max_tokens + drafts;
    let mut sequences = (0..streams)
        .map(|_| match (drafts, drafter) {
            (0, _) | (_, Drafter::Oracle(_)) => rank.new_sequence(capacity),
            (_, Drafter::Mtp) => rank.new_drafting_sequence(capacity),
            (_, Drafter::Block { .. }) => rank.new_block_drafting_sequence(capacity),
        })
        .collect::<Result<Vec<DeviceSequence>, _>>()
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut batch: Vec<_> = sequences
        .iter_mut()
        .map(|sequence| (sequence, prompt))
        .collect();
    let picks = rank
        .forward_batch_greedy(tp, gather, &mut batch)
        .map_err(|error| format!("prefill: {error}"))?;
    let prefill = started.elapsed();
    let mut next = greedy(&picks)?;
    let mut generated: Vec<Vec<u32>> = next.iter().map(|&token| vec![token]).collect();
    println!(
        "glm-tp4: {who}: prefilled {streams} streams of {} tokens in {:.1} ms",
        prompt.len(),
        ms(prefill)
    );

    if drafts > 0 {
        speculate(
            rank,
            (tp, gather),
            &mut sequences,
            (&mut next, &mut generated),
            (drafts, drafter),
            max_tokens,
        )
        .map(|report| println!("glm-tp4: {who}: {report}"))?;
    }
    let mut steps = Vec::with_capacity(max_tokens - 1);
    for _ in generated[0].len()..max_tokens {
        let step: Vec<[u32; 1]> = next.iter().map(|&token| [token]).collect();
        let mut batch: Vec<_> = sequences
            .iter_mut()
            .zip(&step)
            .map(|(sequence, token)| (sequence, token.as_slice()))
            .collect();
        let started = Instant::now();
        let picks = rank
            .forward_batch_greedy(tp, gather, &mut batch)
            .map_err(|error| format!("decode: {error}"))?;
        steps.push(started.elapsed());
        next = greedy(&picks)?;
        for (tokens, &token) in generated.iter_mut().zip(&next) {
            tokens.push(token);
        }
    }
    if !steps.is_empty() {
        let mut sorted = steps.clone();
        sorted.sort();
        let total: Duration = steps.iter().sum();
        println!(
            "glm-tp4: {who}: decoded {} steps of {streams} rows: median {:.2} ms, mean {:.2}, min {:.2}, max {:.2}; {:.1} tok/s",
            steps.len(),
            ms(sorted[sorted.len() / 2]),
            ms(total) / steps.len() as f64,
            ms(sorted[0]),
            ms(sorted[sorted.len() - 1]),
            (steps.len() * streams) as f64 / total.as_secs_f64()
        );
    }

    let digests: Vec<String> = sequences
        .iter()
        .map(|sequence| {
            rank.last_logits(sequence)
                .map(|row| logits_sha256(&row))
                .map_err(|error| format!("last logits: {error}"))
        })
        .collect::<Result<_, _>>()?;
    if let Some(stream) = (1..streams)
        .find(|&stream| generated[stream] != generated[0] || digests[stream] != digests[0])
    {
        return Err(format!(
            "stream {stream} of {streams} decoded {:?} with last logits {}, where stream 0 \
             decoded {:?} with {}",
            generated[stream], digests[stream], generated[0], digests[0]
        ));
    }
    let digest = digests.into_iter().next().unwrap_or_default();
    Ok((generated.swap_remove(0), digest))
}

/// Each stream's next token and its tokens so far.
type Streams<'a> = (&'a mut [u32], &'a mut [Vec<u32>]);

/// Speculative greedy decoding until a stream has `max_tokens`: each round drafts `drafts`
/// tokens after each stream's next one (fewer near the end), verifies them all in one step, and
/// keeps each stream's drafts up to the first greedy would not have picked, then the token its
/// last kept row picks. Returns what to print.
///
/// # Errors
/// When a step fails, or an oracle's draft is not kept: the verify row before it picked another
/// token than the one-token steps did.
fn speculate(
    rank: &mut DeviceRank,
    (tp, gather): (&TpRuntime, &BitGatherHandle),
    sequences: &mut [DeviceSequence],
    (next, generated): Streams<'_>,
    (drafts, drafter): (usize, Drafter<'_>),
    max_tokens: usize,
) -> Result<String, String> {
    let (mut rounds, mut drafting) = (Vec::new(), Duration::ZERO);
    let (mut offered, mut accepted, mut proposed) = (0, 0, 0);
    let mut policies: Vec<Option<DepthPolicy>> = (0..sequences.len())
        .map(|_| {
            matches!(drafter, Drafter::Block { adaptive: true })
                .then(|| DepthPolicy::adaptive(drafts, DFLASH2_STEP_COST))
        })
        .collect();
    // How many stream-rounds kept each number of drafts, from none to `drafts`.
    let mut depths = vec![0usize; drafts + 1];
    let made = generated[0].len();
    while generated.iter().all(|tokens| tokens.len() < max_tokens) {
        let count = drafts.min(max_tokens - generated[0].len() - 1);
        let started = Instant::now();
        let mut drafted = match drafter {
            _ if count == 0 => vec![Vec::new(); sequences.len()],
            // The streams decode the oracle's tokens, so each one's drafts are the next of them.
            Drafter::Oracle(tokens) => generated
                .iter()
                .map(|made| tokens[made.len()..made.len() + count].to_vec())
                .collect(),
            Drafter::Mtp | Drafter::Block { .. } => {
                let mut batch: Vec<_> = sequences.iter_mut().zip(next.iter().copied()).collect();
                rank.draft(tp, gather, &mut batch, count)
                    .map_err(|error| format!("draft: {error}"))?
            }
        };
        proposed += drafted.iter().map(Vec::len).sum::<usize>();
        // Each stream verifies the drafts its policy judges worth their rows; every rank's
        // confidences are the same bits, so every rank verifies the same rows.
        let confidences: Vec<Vec<f32>> = sequences
            .iter()
            .map(|sequence| sequence.draft_confidence().to_vec())
            .collect();
        for ((drafts, policy), confidence) in drafted.iter_mut().zip(&policies).zip(&confidences) {
            if let Some(policy) = policy {
                drafts.truncate(policy.depth(confidence));
            }
        }
        drafting += started.elapsed();
        let chunks: Vec<Vec<u32>> = next
            .iter()
            .zip(drafted)
            .map(|(&token, drafted)| std::iter::once(token).chain(drafted).collect())
            .collect();
        let mut batch: Vec<_> = sequences
            .iter_mut()
            .zip(&chunks)
            .map(|(sequence, chunk)| (sequence, chunk.as_slice()))
            .collect();
        let verified = rank
            .verify_batch_greedy(tp, gather, &mut batch)
            .map_err(|error| format!("verify: {error}"))?;
        drop(batch);
        for (stream, (chunk, picks)) in chunks.iter().zip(verified).enumerate() {
            let kept = 1
                + (1..chunk.len())
                    .take_while(|&row| chunk[row] == picks[row - 1])
                    .count();
            if chunk.len() > 1 {
                rank.commit(&mut sequences[stream], kept)
                    .map_err(|error| format!("commit: {error}"))?;
            }
            let token = picks[kept - 1];
            if matches!(drafter, Drafter::Oracle(_)) && kept < chunk.len() {
                return Err(format!(
                    "round {} kept {} of stream {stream}'s {} oracle drafts: verify row {} of {} \
                     picks {token} where the one-token steps decoded {}",
                    rounds.len(),
                    kept - 1,
                    chunk.len() - 1,
                    kept - 1,
                    chunk.len(),
                    chunk[kept]
                ));
            }
            depths[kept - 1] += 1;
            if let Some(policy) = &mut policies[stream] {
                policy.observe(&confidences[stream], chunk.len() - 1, kept - 1);
            }
            generated[stream].extend(&chunk[1..kept]);
            generated[stream].push(token);
            next[stream] = token;
            (offered, accepted) = (offered + chunk.len() - 1, accepted + kept - 1);
        }
        rounds.push(started.elapsed());
    }
    let total: Duration = rounds.iter().sum();
    let mut sorted = rounds.clone();
    sorted.sort();
    let tokens: usize = generated.iter().map(|tokens| tokens.len() - made).sum();
    Ok(format!(
        "speculated {} rounds of {} streams, {drafts} drafts: median round {:.2} ms, drafting \
         {:.2} ms a round; accepted {accepted} of {offered} drafts ({:.0}%) of {proposed} drafted, \
         {:.2} tokens a stream a round; {:.1} tok/s; stream-rounds keeping 0..={drafts} drafts {}",
        rounds.len(),
        sequences.len(),
        ms(sorted.get(sorted.len() / 2).copied().unwrap_or_default()),
        ms(drafting) / rounds.len().max(1) as f64,
        100.0 * accepted as f64 / offered.max(1) as f64,
        tokens as f64 / (rounds.len() * sequences.len()).max(1) as f64,
        tokens as f64 / total.as_secs_f64(),
        depths
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("/")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_rank_and_refuses_what_it_cannot_run() {
        let line = "--tp-rank 2 --tp-world-size 4 --comm-peers 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
                    --converted /w --prompt-tokens 3,7,11 --streams 3,8 --wire bf16";
        let args: Vec<&str> = line.split_whitespace().collect();
        let parsed = GlmArgs::parse(&args).expect("a whole rank");
        assert_eq!((parsed.layout.rank, parsed.layout.world_size), (2, 4));
        assert_eq!(parsed.prompt, [3, 7, 11]);
        assert_eq!(
            (parsed.streams.as_slice(), parsed.max_tokens),
            (&[3, 8][..], 16)
        );
        assert_eq!(parsed.wire, Wire::Bf16);

        for (change, refusal) in [
            ("--checkpoint /c", "exactly one"),
            ("--tp-world-size 2", "addresses"),
            ("--tp-rank 4", "outside"),
            ("--streams 2,0", "at least 1"),
            ("--wire f16", "f32 or bf16"),
            ("--collective nccl", "--wire f32"),
            ("--collective mpi", "tcp or nccl"),
            ("--prompt hi", "exactly one"),
            ("--top-k 5", "unknown argument"),
            ("--draft 0,2", "9 rows"),
            ("--dflash2 /d", "--draft"),
            (
                "--streams 1 --draft 0,2 --draft-oracle --dflash2 /d",
                "in place of",
            ),
            ("--streams 1 --draft 2,0 --draft-oracle", "starts at 0"),
            ("--draft-oracle", "starts at 0"),
            ("--streams 1 --draft 0,2 --depth adaptive", "--dflash2"),
            ("--depth deep", "fixed or adaptive"),
        ] {
            let mut changed = args.clone();
            changed.extend(change.split_whitespace());
            let error = GlmArgs::parse(&changed).expect_err(change);
            assert!(error.contains(refusal), "{change}: {error}");
        }
        let mut oracle = args.clone();
        oracle.extend("--streams 1 --draft 0,3 --draft-oracle".split_whitespace());
        assert!(GlmArgs::parse(&oracle).expect("an oracle run").oracle);
        let mut adaptive = args.clone();
        adaptive.extend("--streams 1 --draft 0,7 --dflash2 /d --depth adaptive".split_whitespace());
        assert!(GlmArgs::parse(&adaptive).expect("an adaptive run").adaptive);
        let unprompted: Vec<&str> = args[..args.len() - 6].to_vec();
        let error = GlmArgs::parse(&unprompted).expect_err("no prompt");
        assert!(error.contains("--prompt-tokens"), "{error}");
        let mut untokenized = unprompted.clone();
        untokenized.extend(["--prompt", "hi"]);
        let error = GlmArgs::parse(&untokenized).expect_err("no tokenizer");
        assert!(error.contains("--tokenizer"), "{error}");
    }
}
