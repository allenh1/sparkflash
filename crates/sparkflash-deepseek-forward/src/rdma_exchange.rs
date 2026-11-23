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

//! The DeepSeek arm's opening of the direct-RDMA exchange.
//!
//! The exchange itself - the slab, the switch, the open, the stage and collect launches - is
//! [sparkflash_kernels::write_exchange], which the Inkling arm runs too. What is this arm's own is
//! the widest row it sends ([row_cap]), the cubin directory it loads opt_misc_sm121.cubin from,
//! and what its banner says the exchange replaces. Everything else is re-exported, so the forward
//! pass's call sites read as they did when the exchange lived here.

use std::path::Path;
use std::sync::Mutex;

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::write_exchange::Opening;
pub use sparkflash_kernels::write_exchange::{
    EXCHANGE_ENV, ExchangeState, Round, Selection, check, is_open, note_fall_through, refusal_here,
    selection, stats, validate,
};

/// What this arm's open banner says moves to the exchange and what stays on the group.
const REPLACES: &str = "the per-layer one-shot all-reduce and the step's one-shot slot exchanges; bulk prefill, the ring gathers and the overlapped collectives stay on NCCL";

/// The widest row a round carries, in bytes: the plan's `dim` times the most rows a step carries
/// ([crate::tp4_forward::rows::R_MAX]) as bf16. Both are read from the open plan, so raising either
/// grows the cap instead of pushing wider verify rows onto the NCCL fallback without a word.
#[must_use]
pub fn row_cap() -> usize {
    crate::plan::hidden() * crate::tp4_forward::rows::R_MAX * 2
}

/// The shared exchange, opened on the first call with this arm's row cap and the opt_misc image
/// from `cubin_dir`.
///
/// Returns `Ok(None)` when the environment does not ask for it or the world is a single rank
/// (there is nothing to exchange).
///
/// # Errors
/// As [sparkflash_kernels::write_exchange::shared], and a run opened without a cubin directory,
/// whose kernel cannot be loaded.
pub fn shared(
    cuda: &Cuda,
    world: usize,
    rank: usize,
    cubin_dir: Option<&Path>,
) -> Result<Option<&'static Mutex<ExchangeState>>, String> {
    let opening = Opening {
        row_cap: row_cap(),
        replaces: REPLACES,
    };
    sparkflash_kernels::write_exchange::shared(cuda, world, rank, opening, || {
        let directory = cubin_dir.ok_or_else(|| {
            format!(
                "{EXCHANGE_ENV} asked for the direct-RDMA exchange but this run was opened without a cubin directory, so its kernel cannot be loaded"
            )
        })?;
        crate::tp4_forward::cached_cubin(&directory.join("opt_misc_sm121.cubin"))
            .map(|(bytes, _sha)| bytes)
    })
}
