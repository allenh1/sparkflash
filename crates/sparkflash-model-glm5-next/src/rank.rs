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

//! One process's rank of a GLM-5.3-Flash serve group.
//!
//! Opening a rank is the arm's construction: read the checkpoint's own `config.json`, load this
//! rank's share of the weights (from the converted rank file when one is configured), apply the
//! profile's drafting, allocate the request window and print the ranks' boot lines. The process
//! that drives the group supplies the collective's opener, because the transport is the group's
//! and not the arm's; everything a GLM rank is made of is here.
//!
//! Both of the group's serving shapes open their rank here: one request at a time
//! ([open_rank]) and fused lanes ([open_lanes]). The serve seam's step adapter for the one-request
//! shape is here too ([open_serve_rank]): it maps this rank's forward onto [Tp4RankApi] and
//! performs no arithmetic of its own.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use sparkflash_comm::control::{Tp4ControlHub, Tp4ControlWorker};
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model::{MergedPosition, ModelBackend, Sampler};
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_serve::tp4::{
    PROMPT_LOGPROBS_REFUSED_PREFIX, Tp4RankApi, Tp4RankConfig, Tp4RankMetrics, Tp4Sampling,
    Tp4Step, checkpoint_directory,
};
use sparkflash_weights::rank_file::{converted_rank_file, fnv1a};
use sparkflash_weights::tp::{BitGatherHandle, TpLayout, TpRuntime};

use crate::config::Glm5NextConfig;
use crate::device::{DeviceRank, DeviceSequence, LatentCache};
use crate::serve::Glm5NextDeviceBackend;

/// The variable naming a directory of GLM-5.3-Flash rank files (`sparkflash convert`).
pub const CONVERTED_DIR_ENV: &str = "SPARKFLASH_GLM_CONVERTED_DIR";

/// The name this arm's startup messages use.
pub const ARM_NAME: &str = "GLM-5.3-Flash";

/// The model label the collective's messages use.
pub const MODEL_LABEL: &str = "GLM";

/// The configuration file a checkpoint directory carries.
const CHECKPOINT_CONFIG_NAME: &str = "config.json";

/// One GLM-5.3-Flash rank of a group as it comes off the disk: its weights, drafter and
/// collective, and the drafting the profile asks of it.
pub struct LoadedRank {
    /// The rank's resident weights and kernels.
    pub rank: DeviceRank,
    /// The collective's reduce seam.
    pub tp: TpRuntime,
    /// The collective's exact gather.
    pub gather: BitGatherHandle,
    /// The checkpoint's parsed configuration.
    pub config: Glm5NextConfig,
    /// How many drafts a drafting request verifies after each token at most, and the depth rule;
    /// `None` decodes a token a step ([glm_drafting]).
    pub drafting: Option<GlmDrafting>,
    /// Whether a DFlash 2 block drafter is loaded.
    pub block: bool,
    /// Where the weights came from, for the boot line.
    pub origin: String,
    started: Instant,
}

impl LoadedRank {
    /// The most drafts a step verifies; 0 decodes a token a step.
    #[must_use]
    pub fn drafts(&self) -> usize {
        self.drafting.as_ref().map_or(0, |drafting| drafting.drafts)
    }

    /// The shared speculator's depth rule for this rank's drafts.
    #[must_use]
    pub fn depth_policy(&self) -> Option<sparkflash_model::DepthPolicy> {
        self.drafting
            .as_ref()
            .map(|drafting| drafting.policy.clone())
    }

    /// Announces the load and answers how long it took.
    pub fn announce(&self, rank: &Tp4RankConfig) -> u64 {
        let load_ms = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        eprintln!(
            "sparkflash serve: GLM rank {}/{} loaded in {:.1} s from {}; {} collective",
            rank.rank,
            rank.world_size,
            load_ms as f64 / 1000.0,
            self.origin,
            self.tp.label()
        );
        load_ms
    }
}

/// How a GLM rank drafts: the most drafts a step verifies and the shared depth rule over them.
#[derive(Debug, Clone, PartialEq)]
pub struct GlmDrafting {
    /// The most drafts a step proposes and verifies after its token.
    pub drafts: usize,
    /// [sparkflash_model::DepthPolicy::from_speculation] over [Self::drafts].
    pub policy: sparkflash_model::DepthPolicy,
}

impl GlmDrafting {
    /// For the serve log: the chain and its rule.
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "up to {} draft(s) a step, {} depth",
            self.drafts,
            if self.policy.is_adaptive() {
                "adaptive"
            } else {
                "fixed"
            }
        )
    }
}

/// The GLM arm's drafting from the profile's speculation section, for a rank with a block drafter
/// (`block`) or the MTP layer, drafting at most `most`. `None` drafts nothing.
///
/// Either drafter says how sure it is of each draft (the block drafter's selector, the MTP
/// layer's softmax), so the depth is DeepSeek's: adaptive unless `speculation.depth_policy` says
/// fixed (`SPARKFLASH_DRAFT_POLICY`), at least `speculation.min_depth`, against the drafter's
/// own step cost. The chain is `speculation.drafts` (`SPARKFLASH_DRAFT_DRAFTS`), else the whole
/// block for a block drafter, else `speculation.depth` (`SPARKFLASH_DRAFT_DEPTH`; the carried
/// profile says 3), else 2. `speculation.enabled=false` (`SPARKFLASH_DRAFT=off`) drafts
/// nothing.
///
/// # Errors
/// A chain past what the rank drafts, or a draft count asked for with speculation off.
pub fn glm_drafting(
    spec: &sparkflash_model::profile::SpeculationProfile,
    block: bool,
    most: usize,
) -> Result<Option<GlmDrafting>, String> {
    let drafts = match (spec.enabled, spec.drafts) {
        (Some(false), None | Some(0)) => return Ok(None),
        (Some(false), Some(drafts)) => {
            return Err(format!(
                "speculation.enabled=false with speculation.drafts={drafts}: say one or the other"
            ));
        }
        (_, Some(drafts)) => drafts,
        (_, None) if block => most,
        (_, None) => spec.depth.unwrap_or(2),
    };
    if drafts == 0 {
        return Ok(None);
    }
    if drafts > most {
        return Err(format!(
            "{drafts} drafts a step (speculation.drafts or speculation.depth) is past the {most}              this rank's {} drafts",
            if block { "block drafter" } else { "MTP layer" }
        ));
    }
    let cost = if block {
        crate::DFLASH2_STEP_COST
    } else {
        crate::MTP_STEP_COST
    };
    Ok(Some(GlmDrafting {
        drafts,
        policy: sparkflash_model::DepthPolicy::from_speculation(spec, drafts, cost),
    }))
}

/// The checkpoint's `config.json`. A checkpoint without one is refused rather than served from
/// a compiled-in release document.
fn read_config(directory: &Path) -> Result<Glm5NextConfig, String> {
    let path = directory.join(CHECKPOINT_CONFIG_NAME);
    if !path.is_file() {
        return Err(format!(
            "cannot read the {ARM_NAME} configuration: {} has no {CHECKPOINT_CONFIG_NAME}: a \
             GLM-5.3 checkpoint is read from its own configuration, not the carried release",
            directory.display()
        ));
    }
    Glm5NextConfig::from_checkpoint_dir(directory).map_err(|error| {
        format!(
            "cannot read the {ARM_NAME} configuration: {}: {error}",
            path.display()
        )
    })
}

/// Loads one GLM-5.3-Flash rank of the group.
///
/// The weights come from the checkpoint directory (or its index's parent), the configuration from
/// its `config.json`, and the collective from the process's own opener.
///
/// # Errors
/// A missing configuration, a rank the device cannot load, a window outside the checkpoint's, or
/// a drafting setting the rank cannot honour.
pub fn load_rank<F>(
    config: &Tp4RankConfig,
    latent_cache: LatentCache,
    open: F,
) -> Result<LoadedRank, String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    let started = Instant::now();
    let directory = checkpoint_directory(&config.checkpoint);
    let glm_config = read_config(&directory)?;
    let cuda = Arc::new(CudaHandle::new(
        Cuda::load().map_err(|error| format!("CUDA: {error}"))?,
    ));
    let layout = TpLayout {
        rank: config.rank,
        world_size: config.world_size,
    };
    let run = format!("glm-serve;{glm_config:?};{}", config.context);
    // The agreement word, printed before the gather: a rank whose plan differs names the
    // component here (its context, its configuration, or the digest itself) rather than only
    // refusing in the bootstrap.
    eprintln!(
        "sparkflash serve: rank {} plan agreement {:016x}: context {}, configuration {:016x}, \
         {} bytes",
        config.rank,
        fnv1a(run.as_bytes()),
        config.context,
        fnv1a(format!("{glm_config:?}").as_bytes()),
        run.len()
    );
    let (tp, gather) = open(&cuda, MODEL_LABEL, glm_config.hidden_size, &run)?;
    // The converted rank file is one allocation the arm uploads in place, read with the same
    // parallel direct-I/O path the V4.1 ranks use: the checkpoint path re-reads and re-lays-out
    // every tensor, which is the six-minute startup this replaces. A configured directory that
    // does not hold this rank's file is a refusal, not a silent fallback to the slow path.
    let (mut rank, origin) =
        match converted_rank_file(CONVERTED_DIR_ENV, config.rank, config.world_size)? {
            Some(file) => {
                let rank = DeviceRank::load_converted_with_cache(
                    Arc::clone(&cuda),
                    &glm_config,
                    &file,
                    layout,
                    latent_cache,
                )
                .map_err(|error| format!("{}: {error}", file.display()))?;
                (rank, format!("converted {}", file.display()))
            }
            None => {
                let source = sparkflash_weights::safetensors::SafeTensorsSource::open(
                    &directory,
                    &directory.join(sparkflash_weights::safetensors::INDEX_NAME),
                )
                .map_err(|error| format!("{}: {error}", directory.display()))?;
                let rank = DeviceRank::load_with_cache(
                    Arc::clone(&cuda),
                    &glm_config,
                    &source,
                    layout,
                    latent_cache,
                )
                .map_err(|error| error.to_string())?;
                (rank, format!("checkpoint {}", directory.display()))
            }
        };
    if config.context == 0 || config.context > glm_config.max_position_embeddings {
        return Err(format!(
            "context limit {} is outside 1..={}",
            config.context, glm_config.max_position_embeddings
        ));
    }
    // A DFlash 2 block drafter, when the profile names one: it proposes a whole block after each
    // token, which is what takes this arm past the MTP layer's two drafts. Its depth is the
    // shared speculator's, adaptive because the block drafter says how sure it is of each draft.
    // The carried profile under the environment, as the single-GPU arm reads it.
    let profile = sparkflash_model::active_profile();
    if let Some(drafter) = profile.speculation.drafter.as_deref() {
        rank.load_block_drafter(Path::new(drafter))
            .map_err(|error| format!("the DFlash 2 drafter {drafter}: {error}"))?;
        eprintln!(
            "sparkflash serve: GLM rank {}/{} drafts with the DFlash 2 drafter {drafter}",
            config.rank, config.world_size
        );
    }
    // A drafting request's prompt takes the bulk chunk, which captures the rows the drafter's
    // first block reads, unless the profile keeps it on the token path
    // (`prefill.drafting_prompt`).
    rank.set_drafting_prefill(profile.prefill.drafting_prompt.unwrap_or(true));
    let block = rank.has_block_drafter();
    let drafting = glm_drafting(&profile.speculation, block, rank.max_drafts())?;
    if config.rank == 0 {
        eprintln!(
            "sparkflash serve: GLM group drafting: {}",
            drafting
                .as_ref()
                .map_or_else(|| "off".to_owned(), GlmDrafting::label)
        );
    }
    Ok(LoadedRank {
        rank,
        tp,
        gather,
        config: glm_config,
        drafting,
        block,
        origin,
        started,
    })
}

/// Opens one GLM-5.3-Flash rank of the group serving one request at a time.
///
/// The answer is the rank, the window every request resets and reuses, and what its steps report.
/// Allocating the window per request cost every rank about a second of a short prompt's first
/// token at the 1M default; here a window that does not fit beside the weights is refused at
/// startup, not at the first request. A drafting window also keeps the MTP layer's caches, and a
/// block-drafting one the drafter's own.
///
/// # Errors
/// As [load_rank], or a window the rank cannot allocate.
pub fn open_rank<F>(
    config: &Tp4RankConfig,
    latent_cache: LatentCache,
    open: F,
) -> Result<Rank, String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    let loaded = load_rank(config, latent_cache, open)?;
    let sequence = if loaded.block {
        loaded
            .rank
            .new_block_drafting_sequence(config.context)
            .map_err(|error| format!("the GLM request window: {error}"))?
    } else if loaded.drafts() > 0 {
        loaded
            .rank
            .new_drafting_sequence(config.context)
            .map_err(|error| {
                format!("the GLM request window: {error} (speculation.drafts=0 drafts nothing)")
            })?
    } else {
        loaded
            .rank
            .new_sequence(config.context)
            .map_err(|error| format!("the GLM request window: {error}"))?
    };
    let spec = sparkflash_model::Speculator::new(loaded.drafts(), loaded.depth_policy());
    let load_ms = loaded.announce(config);
    Ok(Rank {
        eos: loaded.config.eos_token_ids.clone(),
        rank: loaded.rank,
        sequence,
        tp: loaded.tp,
        gather: loaded.gather,
        spec,
        load_ms,
    })
}

/// One rank of the group opened for a one-request-at-a-time step adapter: the resident rank, the
/// window it serves requests in, and the request's speculation state.
pub struct Rank {
    /// The checkpoint's end-of-sequence ids.
    pub eos: Vec<u32>,
    /// The resident weights and kernels.
    pub rank: DeviceRank,
    /// The request window, allocated once and reused.
    pub sequence: DeviceSequence,
    /// The collective's reduce seam.
    pub tp: TpRuntime,
    /// The collective's exact gather.
    pub gather: BitGatherHandle,
    /// The shared speculator: the depth rule and the drafts it was worth.
    pub spec: sparkflash_model::Speculator,
    /// The load's milliseconds, for the first metrics report.
    pub load_ms: u64,
}

/// Opens one GLM-5.3-Flash rank of the group stepping `lanes` fused lanes
/// ([Glm5NextDeviceBackend::enable_group_fusion]).
///
/// The rank drafts as [open_rank]'s does, so the two shapes serve the same tokens. Rank 0 answers
/// its backend, which broadcasts every lane's frames through the control hub before it runs them;
/// a worker mirrors those frames over its own `lanes` lanes and answers `None` only if the
/// channel's end is clean, which it never is: rank 0's exit ends it with an error.
///
/// # Errors
/// As [load_rank], or a lane width the rank's walk refuses.
pub fn open_lanes<F>(
    config: &Tp4RankConfig,
    lanes: Option<usize>,
    latent_cache: LatentCache,
    open: F,
) -> Result<(Option<Box<dyn ModelBackend>>, usize), String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    let loaded = load_rank(config, latent_cache, open)?;
    let policy = loaded.depth_policy();
    let drafts = loaded.drafts();
    loaded.announce(config);
    let mut backend = Glm5NextDeviceBackend::from_rank(
        loaded.config,
        loaded.rank,
        loaded.tp,
        loaded.gather,
        config.context,
    )
    .map_err(|error| error.to_string())?;
    // The operator's count when given, else the width this rank's walk states.
    let lanes = match lanes {
        Some(lanes) => lanes,
        None => backend
            .lane_limit()
            .map_err(|error| format!("the group's lane width: {error}"))?,
    };
    backend
        .enable_speculation(drafts, policy)
        .map_err(|error| error.to_string())?;
    if config.rank == 0 {
        let hub = Tp4ControlHub::start(config.control_address(), config.world_size, config.timeout)
            .map_err(|error| format!("rank 0 control channel: {error}"))?;
        backend
            .enable_group_fusion(lanes, hub)
            .map_err(|error| format!("cannot fuse the group's lanes: {error}"))?;
        eprintln!(
            "sparkflash serve: GLM rank 0 steps {lanes} fused lanes over {} ranks, {drafts} draft(s)",
            config.world_size
        );
        return Ok((Some(Box::new(backend)), lanes));
    }
    let mut control = control_worker(config)?;
    match backend.mirror_group(&mut control, lanes)? {}
}

/// This rank's control-channel worker, connected and past its handshake.
fn control_worker(config: &Tp4RankConfig) -> Result<Tp4ControlWorker, String> {
    let mut control = Tp4ControlWorker::connect(
        config.control_address(),
        u32::try_from(config.rank).map_err(|_| "rank does not fit in u32".to_owned())?,
        config.world_size,
        config.timeout,
    )
    .map_err(|error| format!("rank {} control connect: {error}", config.rank))?;
    control
        .handshake()
        .map_err(|error| format!("rank {} control handshake: {error}", config.rank))?;
    Ok(control)
}

// ---------------------------------------------------------------------------------------
// The step adapter the serve group drives
// ---------------------------------------------------------------------------------------

/// The tokens a request may keep from the rank's window: the previous request's, when this prompt
/// begins with all of them and continues past them.
///
/// The reuse is all-or-nothing on purpose. A Gated DeltaNet layer's convolution history and
/// recurrent memory are running state, so a window cannot be rewound to an earlier position: a
/// prompt that diverges from what the window holds, or that ends inside it, is served by a full
/// prefill. A prompt equal to the window has no new row to run and no row to take logits from, so
/// it is a full prefill too.
fn reusable_prefix(held: &[u32], prompt: &[u32]) -> usize {
    let shared = held
        .iter()
        .zip(prompt)
        .take_while(|(held, asked)| held == asked)
        .count();
    if shared == held.len() && held.len() < prompt.len() {
        held.len()
    } else {
        0
    }
}

/// One GLM-5.3-Flash rank behind the serve-owned TP=4 control channel.
///
/// The rank mirrors the host rank over the device arm: every rank runs the same prompt and the
/// same sampled token, and the workers ignore their own choice. Rank 0 samples on the host, or
/// takes the device's greedy pick when the host sampler would take the argmax and no logprobs are
/// asked for, which leaves the logits row on the device. Such a request decodes speculatively when
/// rank 0 says so ([Tp4Sampling::speculative]): the MTP layer drafts, one verify step runs the
/// token and its drafts, and the drafts greedy would have picked are kept, so the tokens are the
/// one-token steps'. The sequence is put back at position 0 at the start of every request, so a
/// request's state never leaks into the next one; its window is allocated once, at load, and kept.
struct GlmTp4Rank {
    rank: DeviceRank,
    /// The request window, allocated at load. A request whose prompt extends the tokens the last
    /// one left here keeps them and prefills only its suffix ([Self::prefill]).
    sequence: DeviceSequence,
    /// The ids `sequence` has consumed, in order: the last request's prompt, the tokens its steps
    /// fed, and the drafts its verifies kept. The next request reuses the window only when its
    /// prompt starts with all of them, because a Gated DeltaNet recurrence cannot be rewound to an
    /// earlier position.
    held: Vec<u32>,
    tp: TpRuntime,
    gather: BitGatherHandle,
    sampler: Sampler,
    sampling: sparkflash_model::Sampling,
    /// Whether the request reads each step's logits row ([Tp4Sampling::logits]).
    logits: bool,
    /// How deep a speculative step drafts, and what its drafts were worth: the shared rule's
    /// depth and counters ([sparkflash_model::Speculator]).
    spec: sparkflash_model::Speculator,
    /// Whether the request decodes speculatively ([Tp4Sampling::speculative]) with them.
    speculative: bool,
    eos: Vec<u32>,
    load_ms: u64,
    steps: u64,
    last_step_ms: u64,
}

impl GlmTp4Rank {
    /// One greedy step of `rows` and the sampled token after the last of them.
    fn step(&mut self, rows: &[u32]) -> Result<Tp4Step, String> {
        let started = Instant::now();
        let mut batch = [(&mut self.sequence, rows)];
        let picks = self
            .rank
            .forward_batch_greedy(&self.tp, &self.gather, &mut batch)
            .map_err(|error| format!("the GLM rank forward failed: {error}"))?;
        let (logits, token) = if !self.logits && self.sampler.takes_argmax(&self.sampling) {
            // The device picked the last row's argmax as the host sampler takes it, so the
            // full-vocabulary row stays on the device.
            let token = picks
                .first()
                .and_then(|rows| rows.last())
                .copied()
                .ok_or_else(|| "the GLM rank's step returned no pick".to_owned())?;
            (Vec::new(), token)
        } else {
            let logits = self
                .rank
                .last_logits(batch[0].0)
                .map_err(|error| format!("the GLM rank's last logits: {error}"))?;
            let token = self.sampler.choose(&logits, &self.sampling);
            (logits, token)
        };
        let next_token = if self.eos.contains(&token) {
            None
        } else {
            Some(token)
        };
        self.held.extend_from_slice(rows);
        self.steps += 1;
        self.last_step_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(Tp4Step {
            speculation: None,
            logits,
            next_token,
            accepted: Vec::new(),
            proposed: 0,
            gated: 0,
            cached_tokens: 0,
        })
    }

    /// One speculative step after `token`: the MTP layer drafts up to [Self::drafts] tokens, one
    /// verify runs `token` and them, and the drafts each verify row picks are kept with the token
    /// the last kept row picks: the tokens of one-token greedy steps
    /// ([DeviceRank::verify_batch_greedy]).
    fn speculate(&mut self, token: u32) -> Result<Tp4Step, String> {
        let started = Instant::now();
        // The verify runs the token and its drafts at the next positions, all inside the window.
        let room = self
            .sequence
            .capacity()
            .saturating_sub(self.sequence.position() + 1);
        // The drafter says how sure it is of each draft; an adaptive policy stops the MTP layer
        // and shortens the chain by those against a row's cost, as the fused lanes do.
        let (drafted, confidence) = self
            .rank
            .draft_chain(
                &self.tp,
                &self.gather,
                &mut self.sequence,
                token,
                &self.spec,
                room,
            )
            .map_err(|error| format!("the GLM rank's draft failed: {error}"))?;
        // What the drafter offered, then what the policy thought the rows were worth.
        let proposed = confidence.len();
        let gated = proposed.saturating_sub(drafted.len());
        let chunk: Vec<u32> = std::iter::once(token).chain(drafted).collect();
        let mut batch = [(&mut self.sequence, chunk.as_slice())];
        let picks = self
            .rank
            .verify_batch_greedy(&self.tp, &self.gather, &mut batch)
            .map_err(|error| format!("the GLM rank's verify failed: {error}"))?
            .pop()
            .unwrap_or_default();
        // The shared rule decides how much of the chain survives: the leading drafts the model
        // itself picked, plus the token after them (sparkflash_model::verdict).
        let kept = sparkflash_model::verdict(&chunk, &picks);
        self.spec.observe(&confidence, chunk.len() - 1, kept);
        if chunk.len() > 1 {
            self.rank
                .commit(&mut self.sequence, kept)
                .map_err(|error| format!("the GLM rank's commit failed: {error}"))?;
        }
        let picked = *picks
            .get(kept - 1)
            .ok_or_else(|| "the GLM rank's verify returned a pick a row short".to_owned())?;
        let mut decoded = chunk[1..kept].to_vec();
        decoded.push(picked);
        // The request ends at the first end-of-sequence token, a kept draft or the pick.
        let (accepted, next_token) = match decoded.iter().position(|id| self.eos.contains(id)) {
            Some(end) => (decoded[..end].to_vec(), None),
            None => {
                let next = decoded.pop();
                (decoded, next)
            }
        };
        // The verify consumed its chain up to the row it kept; the drafts past it were written
        // and rewound, so they are not in the window and the next request must not inherit them.
        self.held.extend_from_slice(&chunk[..kept]);
        for &id in &accepted {
            self.sampler.record(id);
        }
        self.steps += 1;
        self.last_step_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(Tp4Step {
            speculation: None,
            logits: Vec::new(),
            next_token,
            accepted,
            proposed,
            gated,
            cached_tokens: 0,
        })
    }
}

impl Tp4RankApi for GlmTp4Rank {
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String> {
        // The window is reset and reused, not rebuilt: a fresh one costs every rank about a
        // second of the request's first token at the 1M default, which rank 0 pays before it
        // starts the prefill clock and the workers pay inside it. A block-drafting window resets
        // in place too: the reset zeroes its drafter's rings, and the prompt refills the rows the
        // first block reads.
        //
        // The reset moved to [Self::prefill]: whether the next request can keep the window at all
        // depends on its prompt, which this call does not see, and a chat turn's prompt begins with
        // everything the previous one left in the window.
        self.sampler.reset();
        self.sampling = sampling.sampling;
        self.logits = sampling.logits;
        self.speculative = sampling.speculative && self.spec.max_depth() > 0;
        if self.speculative && (self.logits || !self.sampler.takes_argmax(&self.sampling)) {
            return Err(
                "a speculative request must take each row's argmax and read no row; this one \
                 samples or asks for logprobs"
                    .to_owned(),
            );
        }
        Ok(())
    }

    fn plan_prefill(&mut self, prompt: &[u32]) -> usize {
        // A chat turn's prompt begins with the previous turn's prompt and reply, which is exactly
        // what the window holds, so the rows below the shared prefix need not be run again. The
        // reuse is all-or-nothing: a Gated DeltaNet layer's convolution history and recurrent
        // memory are running state, so a window cannot be rewound to an earlier position, and a
        // prompt that diverges from it or stops inside it is served by a full prefill. A window
        // whose position and history have drifted apart is not reusable at all.
        if self.sequence.position() != self.held.len() {
            return 0;
        }
        reusable_prefix(&self.held, prompt)
    }

    fn prefill(&mut self, prompt: &[u32], cached: usize) -> Result<Tp4Step, String> {
        if prompt.is_empty() {
            return Err("an empty prompt has no logits".to_owned());
        }
        // The count is rank 0's, taken from the same window this rank holds. A rank that cannot
        // honour it is a different program from its peers, and says so instead of running a
        // prefill the group did not agree to.
        if cached > 0 && reusable_prefix(&self.held, prompt) != cached {
            return Err(format!(
                "the group's {cached}-token prefix is not this rank's: its window holds {} tokens \
                 and the prompt does not extend them",
                self.held.len()
            ));
        }
        if cached == 0 {
            self.rank
                .reset(&mut self.sequence)
                .map_err(|error| format!("resetting the GLM sequence: {error}"))?;
            self.held.clear();
        }
        let started = Instant::now();
        self.sampler.record_prompt(prompt);
        let logits = self
            .rank
            .forward_prefill(
                &self.tp,
                &self.gather,
                &mut self.sequence,
                &prompt[cached..],
            )
            .map_err(|error| format!("the GLM rank's prefill failed: {error}"))?;
        self.held.extend_from_slice(&prompt[cached..]);
        let token = self.sampler.choose(&logits, &self.sampling);
        let next_token = if self.eos.contains(&token) {
            None
        } else {
            Some(token)
        };
        self.steps += 1;
        self.last_step_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(Tp4Step {
            speculation: None,
            logits,
            next_token,
            accepted: Vec::new(),
            proposed: 0,
            gated: 0,
            cached_tokens: cached,
        })
    }

    fn prefill_prompt_logprobs(
        &mut self,
        _prompt: &[u32],
        _selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
        // The rank's logits are one row a step; the prompt-logprob contract needs every scored
        // position's row and a merged candidate order this arm does not produce yet.
        Err(format!(
            "{PROMPT_LOGPROBS_REFUSED_PREFIX} the GLM-5.3-Flash four-rank arm does not score prompt logprobs yet"
        ))
    }

    fn decode(&mut self, token: u32) -> Result<Tp4Step, String> {
        self.sampler.record(token);
        if self.speculative {
            return self.speculate(token);
        }
        self.step(&[token])
    }

    fn metrics(&self) -> Tp4RankMetrics {
        Tp4RankMetrics {
            load_ms: self.load_ms,
            resident_cache_bytes: u64::try_from(self.rank.device_bytes()).unwrap_or(u64::MAX),
            // This arm does not instrument the collective or the tensor count; the fields report
            // zero rather than a number nobody measured.
            resident_cache_tensors: 0,
            reduce_calls: 0,
            reduce_elements: 0,
            steps: self.steps,
            last_step_ms: self.last_step_ms,
        }
    }
}

/// Opens one GLM-5.3-Flash rank of the group serving one request at a time, as the serve seam's
/// step contract.
///
/// The rank and its window are [open_rank]'s; this boxes the step adapter that maps it onto the
/// contract the group driver calls. The collective's opener is the process's, so the CLI passes
/// it in.
///
/// # Errors
/// As [open_rank].
pub fn open_serve_rank<F>(
    config: &Tp4RankConfig,
    latent_cache: LatentCache,
    open: F,
) -> Result<Box<dyn Tp4RankApi>, String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    let rank = open_rank(config, latent_cache, open)?;
    Ok(Box::new(GlmTp4Rank {
        eos: rank.eos,
        rank: rank.rank,
        sequence: rank.sequence,
        held: Vec::new(),
        tp: rank.tp,
        gather: rank.gather,
        sampler: Sampler::new(),
        sampling: sparkflash_model::Sampling::default(),
        logits: false,
        spec: rank.spec,
        speculative: false,
        load_ms: rank.load_ms,
        steps: 0,
        last_step_ms: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::reusable_prefix;

    /// The GLM adapter's rows: a request may keep exactly the window its prompt extends, and
    /// nothing else. This is the decision that says how many prompt rows run, so a move that
    /// changed it would change the step's shape without changing a kernel.
    #[test]
    fn a_window_is_reused_only_by_a_prompt_that_extends_all_of_it() {
        // A chat turn: the previous prompt and reply are the window's contents, and the new turn
        // continues them, so only its own rows are new.
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3, 4, 5]), 3);
        // The same prompt with nothing appended has no row to take logits from.
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3]), 0);
        // A prompt that stops inside the window cannot be served by it: the recurrence is running
        // state and there is no earlier position to rewind to.
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2]), 0);
        // An edited history diverges, at the first mismatching token.
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 9, 3, 4]), 0);
        // A shared prefix that is not the whole window is still no reuse.
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 9, 4]), 0);
        // An empty window, and the first request of a process.
        assert_eq!(reusable_prefix(&[], &[1, 2]), 0);
        assert_eq!(reusable_prefix(&[], &[]), 0);
        assert_eq!(reusable_prefix(&[1], &[1, 2]), 1);
    }
}
