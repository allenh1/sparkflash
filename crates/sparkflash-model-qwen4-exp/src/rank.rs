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

//! One process's rank of a Qwen3.8-Flash-Next serve group.
//!
//! Opening a rank is the arm's construction: read the checkpoint's own `config.json`, load this
//! rank's share of the weights (from the converted rank file when one is configured) and print the
//! boot line. The process that drives the group supplies the collective's opener, because the
//! transport is the group's and not the arm's; everything a Qwen rank is made of is here.
//!
//! Both of the group's serving shapes open their rank here: one request at a time
//! ([load_rank]) and fused lanes ([open_lanes]). The serve seam's step adapter for the one-request
//! shape is here too ([open_serve_rank]): it maps this rank's session steps onto [Tp4RankApi] and
//! performs no arithmetic of its own.

use std::path::Path;
use std::time::Instant;

use sparkflash_comm::control::{Tp4ControlHub, Tp4ControlWorker};
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model::{BackendError, MergedPosition, ModelBackend, ModelSession, StepOutput};
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_serve::tp4::{
    PROMPT_LOGPROBS_REFUSED_PREFIX, Tp4RankApi, Tp4RankConfig, Tp4RankMetrics, Tp4Sampling,
    Tp4Step, checkpoint_directory,
};
use sparkflash_weights::rank_file::converted_rank_file;
use sparkflash_weights::tp::{BitGatherHandle, TpLayout, TpRuntime};

use crate::config::Qwen4ExpConfig;
use crate::device::Qwen4ExpDeviceBackend;

/// The variable naming a directory of Qwen3.8-Flash-Next rank files (`sparkflash convert`).
pub const CONVERTED_DIR_ENV: &str = "SPARKFLASH_QWEN_CONVERTED_DIR";

/// The name this arm's startup messages use.
pub const ARM_NAME: &str = "Qwen3.8-Flash-Next";

/// The configuration file a checkpoint directory carries.
const CHECKPOINT_CONFIG_NAME: &str = "config.json";

/// The checkpoint's `config.json`, with the file named in a refusal.
///
/// # Errors
/// An unreadable or unparsable configuration.
pub fn read_config(directory: &Path) -> Result<Qwen4ExpConfig, String> {
    Qwen4ExpConfig::from_checkpoint_dir(directory).map_err(|error| {
        format!(
            "cannot read the {ARM_NAME} configuration: {}: {error}",
            directory.join(CHECKPOINT_CONFIG_NAME).display()
        )
    })
}

/// Loads one Qwen3.8-Flash-Next rank of the group: its share of the weights (from the converted
/// rank file when one is configured) over the group's collective. Both of the group's serving
/// shapes open their rank here: one request at a time ([load_rank]) and fused lanes
/// ([open_lanes]). Answers the backend, the handle the collective was opened in, and the load's
/// milliseconds.
///
/// # Errors
/// A missing configuration, a rank the device cannot load, or a rank file of another world.
pub fn load_rank<F>(
    config: &Tp4RankConfig,
    open: F,
) -> Result<(Qwen4ExpDeviceBackend, CudaHandle, u64), String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    let started = Instant::now();
    let directory = checkpoint_directory(&config.checkpoint);
    let qwen_config = read_config(&directory)?;
    let cuda = CudaHandle::new(Cuda::load().map_err(|error| format!("CUDA: {error}"))?);
    let (tp, _gather) = open(
        &cuda,
        ARM_NAME,
        qwen_config.hidden_size,
        &format!("qwen-serve;{qwen_config:?};{}", config.context),
    )?;
    let label = tp.label();
    let layout = TpLayout {
        rank: config.rank,
        world_size: config.world_size,
    };
    let converted = converted_rank_file(CONVERTED_DIR_ENV, layout.rank, layout.world_size)?;
    let backend = match &converted {
        Some(path) => {
            Qwen4ExpDeviceBackend::open_rank_converted(&directory, config.context, tp, path)
        }
        None => Qwen4ExpDeviceBackend::open_rank(&directory, config.context, tp),
    }
    .map_err(|error| format!("{}: {error}", directory.display()))?;
    let load_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    eprintln!(
        "sparkflash serve: Qwen rank {}/{} loaded in {:.1} s from {}{}; {label} collective",
        config.rank,
        config.world_size,
        load_ms as f64 / 1000.0,
        directory.display(),
        converted.as_ref().map_or_else(String::new, |path| format!(
            " (converted {})",
            path.display()
        )),
    );
    Ok((backend, cuda, load_ms))
}

/// One Qwen3.8-Flash-Next rank of a group stepping `lanes` fused lanes
/// ([Qwen4ExpDeviceBackend::enable_group_fusion]): rank 0 answers its backend, whose sessions are
/// handles on the group's lanes; a worker replays rank 0's lanes until the group ends
/// ([Qwen4ExpDeviceBackend::mirror_group]). Drafting is the profile's, as it is alone.
///
/// # Errors
/// As [load_rank], or a lane width the rank's walk refuses.
pub fn open_lanes<F>(
    config: &Tp4RankConfig,
    lanes: Option<usize>,
    open: F,
) -> Result<(Option<Box<dyn ModelBackend>>, usize), String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    // The backend's own CUDA handle holds the primary context the collective was opened in
    // (releases are refcounted), so this one may go once the rank is loaded.
    let (mut backend, _cuda, _) = load_rank(config, open)?;
    // The operator's count when given, else the width this rank's walk states.
    let lanes = match lanes {
        Some(lanes) => lanes,
        None => backend
            .lane_limit()
            .map_err(|error| format!("the group's lane width: {error}"))?,
    };
    if config.rank == 0 {
        let hub = Tp4ControlHub::start(config.control_address(), config.world_size, config.timeout)
            .map_err(|error| format!("rank 0 control channel: {error}"))?;
        backend
            .enable_group_fusion(lanes, hub)
            .map_err(|error| format!("cannot fuse the group's lanes: {error}"))?;
        eprintln!(
            "sparkflash serve: Qwen rank 0 steps {lanes} fused lanes over {} ranks",
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

/// The counter delta one step reports from a session's per-request totals.
///
/// The session keeps its own accounting ([ModelSession::speculation]) because it verifies a draft
/// chain inside one call and hands the extra tokens back through later calls. A step reads the
/// totals before and after its call; the difference is what that step did. Saturating because a
/// session that reset a request's counters (or moved one backwards) has nothing to report for this
/// step, and a wrap would be a number nobody measured.
fn this_step(
    before: sparkflash_model::Speculation,
    after: sparkflash_model::Speculation,
) -> sparkflash_model::Speculation {
    sparkflash_model::Speculation {
        verify_steps: after.verify_steps.saturating_sub(before.verify_steps),
        drafts_proposed: after.drafts_proposed.saturating_sub(before.drafts_proposed),
        drafts_accepted: after.drafts_accepted.saturating_sub(before.drafts_accepted),
    }
}

/// One Qwen3.8-Flash-Next rank of a serve-owned tensor-parallel group.
///
/// The backend holds this rank's share of the weights ([Qwen4ExpDeviceBackend::open_rank]) and
/// every branch is summed over the group before it is injected, so each rank computes the same
/// rows, logits and greedy picks. A request opens its own sequence: a speculative one (rank 0's
/// decision, [Tp4Sampling::speculative]) drafts with the DFlash drafter every rank runs whole, and
/// the verified tokens come back one decode call at a time on every rank alike, so the workers see
/// the same token frames a one-token request sends.
///
/// [Qwen4ExpDeviceBackend::open_rank]: sparkflash_model_qwen4_exp::device::Qwen4ExpDeviceBackend::open_rank
struct QwenTp4Rank {
    backend: Qwen4ExpDeviceBackend,
    /// The request's sequence; the last one is dropped before the next is allocated.
    session: Option<Box<dyn ModelSession>>,
    /// Keeps the primary context the collective's communicator was opened in.
    _cuda: CudaHandle,
    load_ms: u64,
    steps: u64,
    last_step_ms: u64,
}

impl QwenTp4Rank {
    fn step<Run>(&mut self, run: Run) -> Result<Tp4Step, String>
    where
        Run: FnOnce(&mut dyn ModelSession) -> Result<StepOutput, BackendError>,
    {
        let session = self
            .session
            .as_deref_mut()
            .ok_or_else(|| "the Qwen rank has no request open".to_owned())?;
        let started = Instant::now();
        // The session verifies its own draft chain inside the call and hands the extra tokens
        // back through later ones, so what this step did is what its speculator counted while it
        // ran. Reading before and after gives this step's share of a per-request total.
        let before = session.speculation();
        let output = run(session).map_err(|error| format!("the Qwen rank's step: {error}"))?;
        let after = session.speculation();
        let counted = this_step(before, after);
        self.steps += 1;
        self.last_step_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(Tp4Step {
            logits: output.logits,
            next_token: output.next_token,
            // The verified chain's later tokens are queued in the session and emitted by the
            // calls that follow, so this step emits one token and reports the chain below.
            accepted: Vec::new(),
            proposed: counted.drafts_proposed,
            // The depth a policy withheld is not in the session's counters; zero says "not
            // reported" rather than inventing a gate count.
            gated: 0,
            // The reference session reports its own reuse through ModelSession::cached_tokens,
            // not through this step.
            cached_tokens: 0,
            speculation: Some(counted),
        })
    }
}

impl Tp4RankApi for QwenTp4Rank {
    fn begin_request(&mut self, sampling: Tp4Sampling) -> Result<(), String> {
        let request = sampling.sampling;
        if sampling.speculative
            && (sampling.logits || !sparkflash_model::Sampler::new().takes_argmax(&request))
        {
            return Err(
                "a speculative request must take each row's argmax and read no row; this one \
                 samples or asks for logprobs"
                    .to_owned(),
            );
        }
        self.session = None;
        let mut session = self
            .backend
            .open_rank_session(sampling.speculative)
            .map_err(|error| format!("the Qwen request window: {error}"))?;
        session.set_sampling(request);
        self.session = Some(session);
        Ok(())
    }

    fn prefill(&mut self, prompt: &[u32], _cached: usize) -> Result<Tp4Step, String> {
        // This arm's session keeps its own window and reports its own reuse
        // (`ModelSession::cached_tokens`), so the group's count is not applied here.
        self.step(|session| session.prefill(prompt))
    }

    fn prefill_prompt_logprobs(
        &mut self,
        _prompt: &[u32],
        _selection: Option<usize>,
    ) -> Result<(Tp4Step, Vec<MergedPosition>), String> {
        // The session reads back one logits row a step; the prompt-logprob contract needs every
        // scored position's row, which the TP=1 arm does not expose either.
        Err(format!(
            "{PROMPT_LOGPROBS_REFUSED_PREFIX} the Qwen3.8-Flash-Next tensor-parallel arm does not score prompt logprobs yet"
        ))
    }

    fn decode(&mut self, token: u32) -> Result<Tp4Step, String> {
        self.step(|session| session.decode(token))
    }

    fn metrics(&self) -> Tp4RankMetrics {
        Tp4RankMetrics {
            load_ms: self.load_ms,
            // Like the GLM arm, the collective and the resident bytes are not instrumented here;
            // the fields report zero rather than a number nobody measured.
            resident_cache_bytes: 0,
            resident_cache_tensors: 0,
            reduce_calls: 0,
            reduce_elements: 0,
            steps: self.steps,
            last_step_ms: self.last_step_ms,
        }
    }
}

/// Opens one Qwen3.8-Flash-Next rank of a serve-owned tensor-parallel group of 2 or 4, as the
/// serve seam's step contract: [load_rank]'s backend and handle, boxed in the adapter that maps
/// its session steps onto [Tp4Step]. The collective's opener is the process's, so the CLI passes
/// it in.
///
/// # Errors
/// As [load_rank].
pub fn open_serve_rank<F>(config: &Tp4RankConfig, open: F) -> Result<Box<dyn Tp4RankApi>, String>
where
    F: FnOnce(&Cuda, &str, usize, &str) -> Result<(TpRuntime, BitGatherHandle), String>,
{
    let (backend, cuda, load_ms) = load_rank(config, open)?;
    Ok(Box::new(QwenTp4Rank {
        backend,
        session: None,
        _cuda: cuda,
        load_ms,
        steps: 0,
        last_step_ms: 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::this_step;
    use sparkflash_model::Speculation;

    /// The Qwen adapter's step shape: a step reports its own delta of the session's per-request
    /// counters, and a reset (or backwards) counter reports nothing rather than wrapping.
    #[test]
    fn a_step_reports_its_own_share_of_the_sessions_counters() {
        let before = Speculation {
            verify_steps: 3,
            drafts_proposed: 9,
            drafts_accepted: 6,
        };
        let after = Speculation {
            verify_steps: 4,
            drafts_proposed: 11,
            drafts_accepted: 7,
        };
        assert_eq!(
            this_step(before, after),
            Speculation {
                verify_steps: 1,
                drafts_proposed: 2,
                drafts_accepted: 1,
            }
        );
        assert_eq!(
            this_step(after, before),
            Speculation {
                verify_steps: 0,
                drafts_proposed: 0,
                drafts_accepted: 0,
            }
        );
    }
}
