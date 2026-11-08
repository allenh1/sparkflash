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

//! Qwen's side of the shared DFlash 2 drafter: how a session implements the launch contract
//! and how the drafter is resolved from the cache.

use sparkflash_kernels::cuda_driver::{Deviceptr, Kernel};
use sparkflash_model::{BackendError, StepCost};
use sparkflash_ops::dflash::{DraftDevice, GpuDflash2};
use sparkflash_ops::gemv::{GemvDevice, GpuMat, GpuMatE4m3};

use super::DeviceSequence;
use crate::config::Qwen4ExpConfig;

/// The modelled wall of a DFlash 2 step on a GB10-class part.
pub(super) const DFLASH2_STEP_COST: StepCost = StepCost {
    step_ms: 59.0,
    row_ms: 8.4,
    prior_tokens_per_ms: 0.034,
};

/// The drafter cubin the shared stack embeds.
#[must_use]
pub fn embedded_dflash2_cubin() -> &'static [u8] {
    sparkflash_kernels::images::dflash2_cubin()
}

/// The drafter `SPARKFLASH_QWEN4_DFLASH2` names, or the canonical snapshot in the cache.
pub(super) fn load_dflash_from_env(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    target: &Qwen4ExpConfig,
    head: &GpuMat,
) -> Result<Option<GpuDflash2>, BackendError> {
    let Some(directory) = crate::dflash::drafter_directory(target) else {
        return Ok(None);
    };
    let loaded = cuda
        .load_named(super::embedded_cubin(), "q4e_quantize_e4m3_rows")
        .and_then(|(_, quantize)| GpuMatE4m3::quantize(cuda, quantize, head, &mut 0))
        .and_then(|head| {
            GpuDflash2::load(cuda, &directory, target, &head, (0, 1))
                .inspect_err(|_| head.free(cuda))
        });
    loaded.map(Some).map_err(|error| {
        BackendError::new(
            "qwen4_dflash2",
            format!("drafter {}: {error}", directory.display()),
        )
    })
}

impl GemvDevice for DeviceSequence {
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        DeviceSequence::launch(self, kernel, grid, block, args)
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        DeviceSequence::gemv(self, matrix, input, output, rows)
    }

    fn tiled_scratch(&self) -> (Deviceptr, Deviceptr) {
        (self.gemm_partials, self.tiled_counters)
    }
}

impl DraftDevice for DeviceSequence {
    fn mark(&self, name: &'static str) -> Result<(), BackendError> {
        self.prof_mark(name)
    }
}

impl DeviceSequence {
    /// Whether this sequence drafts with DFlash 2 rather than the MTP head.
    pub(super) fn drafting(&self) -> bool {
        self.spec_enabled && self.draft.is_some() && self.backend.dflash.is_some()
    }

    /// Copy target layer `index`'s completed rows of this forward into the drafter's feature
    /// rows, when the drafter reads that layer: the mean of the hyper-connection streams.
    pub(super) fn capture_features(&self, index: usize, batch: usize) -> Result<(), BackendError> {
        let (Some(draft), Some(scratch)) = (self.backend.dflash.as_ref(), self.draft.as_ref())
        else {
            return Ok(());
        };
        if !self.spec_enabled {
            return Ok(());
        }
        let streams = (self.hyper, self.backend.config.hc_count);
        let (first, rows) = self.own_rows.unwrap_or((0, batch));
        draft.capture(self, scratch, index, streams, first, rows)
    }

    /// Every rank's captured rows of a sequence-parallel forward ([Self::own_rows]) gathered into
    /// its peers' rows, so the drafter's context path reads the whole chunk on every rank.
    pub(super) fn gather_features(&self) -> Result<(), BackendError> {
        let (Some(draft), Some(scratch), Some((first, count))) = (
            self.backend.dflash.as_ref(),
            self.draft.as_ref(),
            self.own_rows,
        ) else {
            return Ok(());
        };
        if !self.spec_enabled {
            return Ok(());
        }
        let row = draft.config.feature_width() * 4;
        self.gather_rows(
            scratch.features().offset(self.window * row),
            scratch.features().offset(first * row),
            count * row,
        )
    }

    /// Commit `rows` captured rows of the last forward, from row `first_row` on, which sit at
    /// `position` onward: the drafter's context path, then every layer's ring.
    pub(super) fn dflash_commit(
        &self,
        first_row: usize,
        rows: usize,
        position: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 || !self.drafting() {
            return Ok(());
        }
        let (Some(draft), Some(scratch)) = (self.backend.dflash.as_ref(), self.draft.as_ref())
        else {
            return Ok(());
        };
        // What the host did with the last forward's logits is charged before the drafter's own.
        self.prof_mark("host_answer")?;
        draft.commit(self, scratch, first_row, rows, position)
    }

    /// Draft the block after `anchor`, the token at `base`, into `spec_drafts`: nothing when the
    /// verify that would follow could not fit before the context limit.
    pub(super) fn dflash_propose(&mut self, anchor: u32, base: usize) -> Result<(), BackendError> {
        self.spec_drafts.clear();
        self.spec_confidence.clear();
        if !self.drafting() {
            return Ok(());
        }
        let backend = std::sync::Arc::clone(&self.backend);
        let (Some(draft), Some(scratch)) = (backend.dflash.as_ref(), self.draft) else {
            return Ok(());
        };
        let block = draft.config.block_size;
        if base + block > self.context_limit {
            return Ok(());
        }
        let embed = backend.weights.embed.pointer;
        let candidates = GpuDflash2::candidates(&scratch);
        draft.score(self, &scratch, embed, anchor, base, 0, candidates)?;
        draft.select(self, &scratch, anchor)?;
        let mut bytes = vec![0u8; 2 * block * 4];
        // The readback has to be ordered behind the launches above *on this session's stream*: a
        // plain synchronous copy runs on the default stream, which a session's work no longer
        // shares, and would return the previous step's block (or zeros) while the selector was
        // still queued - drafting from stale tokens, which the target then rejects.
        backend
            .cuda
            .copy_dtoh_on(self.clock(), &mut bytes, &scratch.drafts())
            .map_err(|error| BackendError::new("qwen4_dflash2", error))?;
        self.prof_mark("d_select")?;
        let edges = block - 1;
        let words = bytes.as_chunks::<4>().0;
        // A stop token ends the chain: the model's own answer there is the end of the sequence,
        // which no draft can match, so nothing past it could be accepted.
        for (chunk, confidence) in words[..edges].iter().zip(&words[block..block + edges]) {
            let token = u32::from_le_bytes(*chunk);
            if self.eos.contains(&token) {
                break;
            }
            self.spec_drafts.push(token);
            self.spec_confidence.push(f32::from_le_bytes(*confidence));
        }
        // The shared speculator decides how many of the block's drafts are worth their rows.
        let depth = self
            .spec
            .depth(&self.spec_confidence, self.spec_drafts.len());
        self.spec_drafts.truncate(depth);
        self.spec_confidence.truncate(depth);
        Ok(())
    }
}
