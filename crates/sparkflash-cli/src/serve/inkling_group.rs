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

//! The Inkling device arm as a four-rank server: the CLI's wiring.
//!
//! The rank machinery lives in [sparkflash_model_inkling2::group] - the collective, the control
//! mirror, the session that broadcasts before each step. This module owns only what this binary
//! decides: which rank this process is, what the boot line says, and which backend the HTTP driver
//! wraps. Rank 0 hands its backend to [serve_backend_world]; every other rank mirrors rank 0 and
//! returns when the control channel ends.

use std::process::ExitCode;
use std::sync::Arc;

use sparkflash_model::{ModelDescriptor, PromptCodec};
use sparkflash_model_inkling2::Inkling2Config;

use super::ServeOptions;
use super::dynamic::{Dialect, DynamicCheckpoint, serve_backend_world};

/// Serves one rank of the group, or mirrors it when this process is not rank 0.
///
/// # Errors
/// The checkpoint configuration, a rank file, the transport, or the arm's own device open.
pub(super) fn serve(
    options: &ServeOptions,
    checkpoint: DynamicCheckpoint,
    descriptor: ModelDescriptor,
    seam: Arc<dyn PromptCodec>,
    dialect: Dialect,
    context: usize,
) -> Result<ExitCode, String> {
    let group = options
        .group
        .as_ref()
        .ok_or_else(|| "the device-arm group is dispatched only for --rank".to_owned())?;
    let config = Inkling2Config::from_checkpoint_dir(&checkpoint.directory)
        .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
    if group.rank != 0 {
        sparkflash_model_inkling2::run_worker(&descriptor, &config, group, context)?;
        return Ok(ExitCode::SUCCESS);
    }
    let master = sparkflash_model_inkling2::open_master(
        descriptor.clone(),
        &config,
        group,
        context,
        &checkpoint.directory,
    )?;
    serve_backend_world(
        master.backend,
        descriptor,
        seam,
        dialect,
        options,
        &master.banner,
        group.world,
    )
}
