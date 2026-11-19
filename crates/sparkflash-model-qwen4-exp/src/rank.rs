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
//! ([load_rank]) and fused lanes ([open_lanes]).

use std::path::Path;
use std::time::Instant;

use sparkflash_comm::control::{Tp4ControlHub, Tp4ControlWorker};
use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model::ModelBackend;
use sparkflash_ops::gemv::CudaHandle;
use sparkflash_serve::tp4::{Tp4RankConfig, checkpoint_directory};
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
