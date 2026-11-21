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

//! The dense walk's parts.
//!
//! The walk runs a list of parts built from the architecture's steps
//! ([sparkflash_model::compose]): a layer's token mixer, then its feed-forward, each a
//! [PartExec] over this arm ([Dense]). The four the catalogue has are registered in
//! [Dense::catalogue]: grouped-query attention, the Gated DeltaNet, the dense SwiGLU MLP and the
//! routed experts. A family with a part of its own writes that part and one registration line,
//! and opens the arm with its registry ([crate::DenseDeviceBackend::open_with]).
//!
//! A part's body reads the normed rows ([PassCx::input]) and leaves the branch's rows in
//! [PassCx::output]; the walk runs the norm before it and the group's sum and the residual add
//! after it. Its constructor uploads this rank's share of its weights through [OpenCx] and says
//! what state and what scratch it needs; the arm lays out the lanes, backs them, rewinds them and
//! frees everything.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_model::backend::BackendError;
use sparkflash_model::{Architecture, Part, TpPlan};
use sparkflash_ops::formats::{Bf16, FormatKind, Mat};
use sparkflash_ops::naming::{LayerNames, TensorNames};
use sparkflash_ops::projection::WidePolicy;
use sparkflash_ops::widen::LedgerRow;
use sparkflash_runtime::{PartExec, PartHost, PartRegistry, RankSplit, Region};
use sparkflash_weights::WeightSource;

use super::{Buffers, DEVICE, DenseMlp, Gdn, Gqa, RoutedExperts, Sequence, Shape, Shared, Uploads};

/// The shared dense device arm, as the host its parts plug into.
#[derive(Debug, Clone, Copy)]
pub struct Dense;

impl PartHost for Dense {
    type Pass<'a> = PassCx<'a>;
    type Open<'a> = OpenCx<'a>;
    type Handle = DensePart;
}

/// One part of the dense walk as the arm holds it. The arm's weights are shared between the
/// serving thread and its sessions, so a part is `Send + Sync`; one that holds a
/// [Kernel] says so itself, as the arm's own kernel table does.
pub type DensePart = Box<dyn PartExec<Dense> + Send + Sync>;

/// The executors a dense backend builds its walk from, by part kind.
pub type DenseParts = PartRegistry<Dense>;

impl Dense {
    /// The catalogue parts' executors: every part the arm runs without a model crate.
    #[must_use]
    pub fn catalogue() -> DenseParts {
        let mut parts = DenseParts::new();
        let registered = [
            parts.register(Part::GqaAttention, |open: &mut OpenCx<'_>| open.gqa()),
            parts.register(Part::GdnScan, |open: &mut OpenCx<'_>| open.gdn()),
            parts.register(Part::DenseMlp, |open: &mut OpenCx<'_>| open.dense_mlp()),
            parts.register(Part::Moe, |open: &mut OpenCx<'_>| open.routed_experts()),
        ];
        debug_assert!(
            registered.iter().all(Result::is_ok),
            "the catalogue names each part once"
        );
        parts
    }
}

/// One pass of the walk over one part: `rows` normed rows in, the branch's rows out.
///
/// The rows of every segment lie one after another, each sequence's at its own position. A
/// row-wise stage takes all of them at once; a stage that reads a sequence's state takes one
/// segment's rows at a time ([Self::segments]). Every launch goes on the walk's one stream, in
/// the order the body makes it.
pub struct PassCx<'a> {
    pub(crate) shared: &'a Shared,
    pub(crate) b: &'a Buffers,
    pub(crate) segments: &'a [(&'a mut Sequence, &'a [u32])],
    /// The layer.
    pub(crate) index: usize,
    /// Which of a sequence's region lists is this part's.
    pub(crate) slot: usize,
    pub(crate) rows: usize,
    pub(crate) policy: Option<WidePolicy>,
}

/// One segment of a pass: a run of rows that belong to one sequence.
#[derive(Debug, Clone, Copy)]
pub struct SegmentRows<'a> {
    /// The segment's first row in the pass.
    pub first: usize,
    /// Its rows.
    pub rows: usize,
    /// The sequence's position before the pass: the segment's first row is this position.
    pub position: usize,
    /// The part's regions in this sequence's lane, in the order [PartExec::state] declares
    /// them. A positional region is backed up to `position + rows` rows.
    pub regions: &'a [Deviceptr],
}

impl PassCx<'_> {
    /// The layer the part runs in.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.index
    }

    /// The rows of the pass, at most [crate::CHUNK_ROWS].
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The residual width.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.shared.hidden
    }

    /// The normed rows the part reads, f32 `[rows][hidden]`.
    #[must_use]
    pub fn input(&self) -> Deviceptr {
        self.b.normed
    }

    /// Where the part leaves the branch's rows, f32 `[rows][hidden]`: a rank's partial when the
    /// part says [RankSplit::Summed].
    #[must_use]
    pub fn output(&self) -> Deviceptr {
        self.b.projected
    }

    /// The wide-tile policy of the pass: `None` keeps every projection on the row GEMV.
    #[must_use]
    pub fn policy(&self) -> Option<WidePolicy> {
        self.policy
    }

    /// The segments, in row order.
    pub fn segments(&self) -> impl Iterator<Item = SegmentRows<'_>> + '_ {
        let mut first = 0usize;
        self.segments.iter().map(move |(sequence, tokens)| {
            let segment = SegmentRows {
                first,
                rows: tokens.len(),
                position: sequence.position(),
                regions: &sequence.regions[self.slot],
            };
            first += tokens.len();
            segment
        })
    }

    /// The scratch the part's constructor reserved as `slot`, f32 `[rows][floats]` for the
    /// [crate::CHUNK_ROWS] rows a pass holds at most. Nothing
    /// in it outlives the pass.
    ///
    /// # Errors
    /// A slot this walk's scratch does not hold.
    pub fn scratch(&self, slot: ScratchSlot) -> Result<Deviceptr, BackendError> {
        self.b.extra.get(slot.0).copied().ok_or_else(|| {
            BackendError::new(
                DEVICE,
                format!(
                    "layer {}'s part asks for scratch {} and the walk reserved {}",
                    self.index,
                    slot.0,
                    self.b.extra.len()
                ),
            )
        })
    }

    /// Launches `kernel` on the walk's stream.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.shared.launch(kernel, (grid, block), args)
    }
}

/// A walk scratch buffer a part reserved when it was opened ([OpenCx::scratch]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchSlot(usize);

/// One part being opened, for one layer of one rank: the checkpoint, the rank's share and the
/// arm's uploader.
///
/// The four catalogue constructors are here ([Self::gqa], [Self::gdn], [Self::dense_mlp],
/// [Self::routed_experts]), so a part of a family's own can hold one and add to it. A part that
/// uploads tensors of its own hands every allocation to [Self::keep] and states each in the boot
/// format ledger ([Self::record]).
pub struct OpenCx<'a> {
    pub(crate) up: Uploads<'a>,
    pub(crate) cuda: &'a Cuda,
    pub(crate) architecture: &'a Architecture,
    pub(crate) names: &'a TensorNames,
    /// Every layer's shape as this rank holds it.
    pub(crate) shapes: &'a [Shape],
    pub(crate) plan: TpPlan,
    /// The layer being opened.
    pub(crate) index: usize,
    /// That layer's shape at one rank.
    pub(crate) whole: Option<Shape>,
    /// The scratch reserved so far, in f32 a row.
    pub(crate) scratch: Vec<usize>,
}

impl<'a> OpenCx<'a> {
    /// The layer being opened.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.index
    }

    /// The checkpoint's architecture.
    #[must_use]
    pub fn architecture(&self) -> &'a Architecture {
        self.architecture
    }

    /// The rank and world the part is opened for: a part uploads its share as this plan splits
    /// it.
    #[must_use]
    pub fn plan(&self) -> &TpPlan {
        &self.plan
    }

    /// The residual width.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.architecture.hidden_size
    }

    /// The device context, for a part that loads a kernel of its own.
    #[must_use]
    pub fn cuda(&self) -> &'a Cuda {
        self.cuda
    }

    /// The checkpoint's tensors.
    #[must_use]
    pub fn source(&self) -> &'a dyn WeightSource {
        self.up.source
    }

    /// Hands a device allocation of `bytes` to the arm, which counts it with the weights and
    /// frees it with them (also when the load fails part way).
    pub fn keep(&mut self, pointer: Deviceptr, bytes: usize) -> Deviceptr {
        self.up.keep(pointer, bytes)
    }

    /// States one uploaded tensor in the boot format ledger: the format it is stored at, the
    /// format its kernel reads it at and the bytes one token reads from it. Boot refuses a row
    /// whose two formats differ, since this context records no widening.
    pub fn record(
        &mut self,
        name: &str,
        stored: FormatKind,
        arithmetic: FormatKind,
        bytes_per_token: u128,
    ) {
        self.up
            .rows
            .push(LedgerRow::new(name, stored, arithmetic, bytes_per_token));
    }

    /// Names a BF16 projection the arm's row GEMV reads for a packed copy
    /// ([sparkflash_ops::packed]). One it is not told of keeps the unpacked rows.
    pub fn pack(&mut self, matrix: &Mat<Bf16>) {
        self.up.packable.push(matrix.clone());
    }

    /// Reserves `floats` f32 a row of walk scratch for the part ([PassCx::scratch]): every
    /// walk's scratch holds [crate::CHUNK_ROWS] rows of it.
    pub fn scratch(&mut self, floats: usize) -> ScratchSlot {
        self.scratch.push(floats);
        ScratchSlot(self.scratch.len() - 1)
    }

    /// The layer's shape as the rank holds it and at one rank, and its tensors' names.
    fn at(&self) -> Result<(Shape, Shape, &'a LayerNames), String> {
        match (
            self.shapes.get(self.index),
            self.whole,
            self.names.layers.get(self.index),
        ) {
            (Some(shape), Some(whole), Some(names)) => Ok((*shape, whole, names)),
            _ => Err(format!("layer {} is not being opened", self.index)),
        }
    }

    /// The catalogue's grouped-query attention for this layer ([Part::GqaAttention]).
    ///
    /// # Errors
    /// A layer that is not one, a tensor at another shape or dtype, or an upload the device
    /// refuses.
    pub fn gqa(&mut self) -> Result<DensePart, String> {
        let (shape, whole, names) = self.at()?;
        let hidden = self.hidden();
        let part = self
            .up
            .gqa(self.index, &names.attention, (&shape, &whole), hidden)?;
        Ok(Box::new(part))
    }

    /// The catalogue's Gated DeltaNet for this layer ([Part::GdnScan]).
    ///
    /// # Errors
    /// As [Self::gqa].
    pub fn gdn(&mut self) -> Result<DensePart, String> {
        let (shape, whole, names) = self.at()?;
        let hidden = self.hidden();
        let part = self
            .up
            .gdn(self.index, &names.attention, (&shape, &whole), hidden)?;
        Ok(Box::new(part))
    }

    /// The catalogue's dense SwiGLU MLP for this layer ([Part::DenseMlp]).
    ///
    /// # Errors
    /// As [Self::gqa].
    pub fn dense_mlp(&mut self) -> Result<DensePart, String> {
        let (shape, whole, names) = self.at()?;
        let hidden = self.hidden();
        let part = self
            .up
            .dense_mlp(self.index, &names.ffn, (&shape, &whole), hidden)?;
        Ok(Box::new(part))
    }

    /// The catalogue's routed experts for this layer ([Part::Moe]).
    ///
    /// # Errors
    /// As [Self::gqa].
    pub fn routed_experts(&mut self) -> Result<DensePart, String> {
        let (shape, whole, names) = self.at()?;
        let hidden = self.hidden();
        let part = self
            .up
            .routed(self.index, &names.ffn, (&shape, &whole), hidden)?;
        Ok(Box::new(part))
    }
}

impl PartExec<Dense> for Gqa {
    fn state(&self) -> &[Region] {
        &self.state
    }

    /// A rank holds its share of the query heads (and of the key/value heads, which a world
    /// wider than them shares), so the output projection leaves a partial.
    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        let (shared, b) = (pass.shared, pass.b);
        shared.attend(self, pass)?;
        shared.project(
            &self.o,
            b,
            b.attended.0,
            b.projected.0,
            pass.rows,
            pass.policy,
        )
    }

    // The token body is the prompt body: the pass's policy and its row count pick each
    // projection's kernel.
    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

impl PartExec<Dense> for Gdn {
    fn state(&self) -> &[Region] {
        &self.state
    }

    /// A rank holds its share of the key heads and of the value heads, so `out_proj` leaves a
    /// partial.
    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        let (shared, b) = (pass.shared, pass.b);
        shared.linear(self, pass)?;
        shared.project(
            &self.o,
            b,
            b.attended.0,
            b.projected.0,
            pass.rows,
            pass.policy,
        )
    }

    // The token body is the prompt body: every row count takes the scan image's chunk kernels.
    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

impl PartExec<Dense> for DenseMlp {
    fn state(&self) -> &[Region] {
        &[]
    }

    /// A rank holds its share of the width (the gate and up rows, the down columns), so the down
    /// projection leaves a partial.
    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        let b = pass.b;
        pass.shared.mlp(
            &self.mlp,
            self.clamp,
            b,
            b.projected,
            pass.rows,
            pass.policy,
        )
    }

    // The token body is the prompt body: the pass's policy and its row count pick each
    // projection's kernel.
    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

impl PartExec<Dense> for RoutedExperts {
    fn state(&self) -> &[Region] {
        &[]
    }

    /// A rank holds its share of every expert's width and of the shared expert's, and the router
    /// whole, so the combine leaves a partial under picks every rank agrees on.
    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Summed
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        pass.shared.routed(self, pass)
    }

    // The token body is the prompt body: the token passes run every row count.
    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}
