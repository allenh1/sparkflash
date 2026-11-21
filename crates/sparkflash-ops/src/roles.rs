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

//! The role bodies a host composes around a layer's token mixer and feed-forward, and the device
//! capabilities they run on.
//!
//! A body takes explicit device buffers and a kernel table resolved when it is opened. It never
//! picks a projection kernel or a row path itself: [DeviceCx::project] is the arm's own GEMV or
//! GEMM dispatch, [DeviceCx::sum] its collective in rank order, and a fused or unfused path is an
//! argument the host passes. An arm calls a body through a small adapter, so the launches, their
//! order and their arguments stay the arm's.
//!
//! The bodies here are the shared shapes of three roles: [StreamMixer] (the residual's
//! hyper-connection streams, a frame around each branch), [PooledIndexer] (the selection a token
//! mixer holds) and [EngramSide] (a lookup that lands on the layer input). Each carries its
//! geometry, the bounds of the kernels it launches and its kernel table. Their passes refuse by
//! name until an arm's body moves in: GLM's manifold mix, GLM's pooled indexer and DeepSeek's
//! Engram gate and write.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::Image;
use sparkflash_model::TpPlan;
use sparkflash_model::architecture::{
    Activation, EngramGeometry, HyperConnection, Indexer, Switch,
};
use sparkflash_model::backend::BackendError;
use sparkflash_weights::mhc::HcGeometry;

use crate::calls::{self, BLOCK, Call};
use crate::indexer::{MAX_BUDGET_BLOCKS, MAX_SELECTED_TOKENS};
use crate::moe::Launch;
use crate::parts::CATALOGUE;
use crate::plan::Precision;

/// The error code of a role body's refusals.
const ROLE: &str = "role";

/// Q4E_MHC_MAX_HC: the most streams `q4e_mhc_in` and `q4e_mhc_post` hold.
pub const MAX_STREAMS: usize = 5;
/// Q4E_MHC_COLUMNS: the columns of every stream copy a thread of `q4e_mhc_in` holds, so the hidden
/// width is a multiple of [BLOCK] of at most this many.
pub const MIX_COLUMNS: usize = 16;
/// Q4E_GLM_MAX_POOL: the most keys `q4e_glm_pool` pools into one.
pub const MAX_POOL: usize = 16;
/// Q4E_GLM_MAX_INDEX_HEADS: the most indexer heads `q4e_glm_index_scores` scores.
pub const MAX_INDEX_HEADS: usize = 32;
/// Q4E_GLM_MAX_INDEX_DIM: the widest indexer head `q4e_glm_index_scores` reads.
pub const MAX_INDEX_DIM: usize = 128;

/// One loaded kernel and the catalogue entry it was resolved from: the image's symbol list and
/// the catalogue's precision mask are checked at load, so a launch site holds a handle rather
/// than a name it looked up.
#[derive(Clone, Copy, Debug)]
pub struct Bound {
    kernel: Kernel,
    precision: Precision,
}

impl Bound {
    /// Whether `image` exports `symbol` and the catalogue carries it at `precision`.
    ///
    /// # Errors
    /// The symbol the image does not export, or the precision the catalogue does not carry it at.
    pub fn carried(image: Image, symbol: &'static str, precision: Precision) -> Result<(), String> {
        if !image.exported_symbols().contains(&symbol) {
            return Err(format!("{} does not export {symbol}", image.name()));
        }
        let carried = CATALOGUE.iter().any(|entry| {
            entry.image == image
                && entry
                    .symbols
                    .iter()
                    .any(|candidate| candidate.name == symbol && candidate.accepts(precision))
        });
        if !carried {
            return Err(format!(
                "the catalogue carries no {symbol} in {} accepting {}",
                image.name(),
                precision.name()
            ));
        }
        Ok(())
    }

    /// Resolve `symbol` in `image` at `precision` and load it from `cubin`.
    ///
    /// # Errors
    /// A symbol the image does not export, one the catalogue does not carry at this precision,
    /// or the driver's load refusal.
    pub fn new(
        cuda: &Cuda,
        image: Image,
        cubin: &'static [u8],
        symbol: &'static str,
        precision: Precision,
    ) -> Result<Self, String> {
        Self::carried(image, symbol, precision)?;
        let (_, kernel) = cuda.load_named(cubin, symbol)?;
        Ok(Self { kernel, precision })
    }

    /// [Self::new] for a body loaded with a dynamic shared-memory reservation.
    ///
    /// # Errors
    /// As [Self::new], or the driver's attribute call.
    pub fn shared(
        cuda: &Cuda,
        image: Image,
        cubin: &'static [u8],
        symbol: &'static str,
        precision: Precision,
        shared_bytes: u32,
    ) -> Result<Self, String> {
        Self::carried(image, symbol, precision)?;
        let (_, kernel) = cuda.load_named_shared(cubin, symbol, shared_bytes)?;
        Ok(Self { kernel, precision })
    }

    /// The loaded driver function.
    #[must_use]
    pub const fn kernel(self) -> Kernel {
        self.kernel
    }

    /// The arithmetic the catalogue entry accepts this body at.
    #[must_use]
    pub const fn precision(self) -> Precision {
        self.precision
    }

    /// The dynamic shared-memory reservation the handle was loaded with, 0 for none.
    #[must_use]
    pub fn shared_bytes(self) -> u32 {
        self.kernel.shared_bytes()
    }
}

// SAFETY: a Kernel is an opaque driver function word, and the handle only names it. The context
// that owns the module outlives every body that holds one.
unsafe impl Send for Bound {}
unsafe impl Sync for Bound {}

impl From<Bound> for Kernel {
    fn from(bound: Bound) -> Self {
        bound.kernel
    }
}

/// A role body's kernels: the image they live in and each symbol at the arithmetic it runs at.
///
/// [Self::check] resolves the table against the catalogue and the image's export list without
/// loading anything, so a host can refuse a checkpoint before it opens a body; [Self::load]
/// loads every symbol from the image this build embeds. A body loads its table when it is opened,
/// so a host that frames no such role loads none of its kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelTable<const N: usize> {
    /// The image every symbol is in.
    pub image: Image,
    /// The symbols and the arithmetic each runs at, in the order [Self::load] returns them.
    pub symbols: [(&'static str, Precision); N],
}

impl<const N: usize> KernelTable<N> {
    /// Whether every symbol is exported by the image and carried by the catalogue at its
    /// precision.
    ///
    /// # Errors
    /// The first symbol [Bound::carried] refuses.
    pub fn check(&self) -> Result<(), String> {
        self.symbols
            .iter()
            .try_for_each(|&(symbol, precision)| Bound::carried(self.image, symbol, precision))
    }

    /// Loads every symbol from the image this build embeds.
    ///
    /// # Errors
    /// An image this build does not embed, or as [Bound::new].
    pub fn load(&self, cuda: &Cuda) -> Result<[Bound; N], String> {
        let cubin = self.image.bytes();
        if cubin.is_empty() {
            return Err(format!(
                "this build embeds no {} image to load {:?} from",
                self.image.name(),
                self.symbols
            ));
        }
        let mut loaded = Vec::with_capacity(N);
        for (symbol, precision) in self.symbols {
            loaded.push(Bound::new(cuda, self.image, cubin, symbol, precision)?);
        }
        loaded
            .try_into()
            .map_err(|_| "a kernel table loaded the wrong count".to_owned())
    }
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
    /// The part's regions in this sequence's lane, in the order the part declares them. A
    /// positional region is backed up to `position + rows` rows.
    pub regions: &'a [Deviceptr],
}

/// The launch half of a body's host: every launch on the host's stream, in the order the body
/// makes it. A body that only launches (the embedding, a norm) asks for this alone, so a host
/// whose passes are not a [DeviceCx] runs it too.
pub trait Launcher {
    /// Launches `kernel` over `launch` with `args`, pointers as device addresses.
    ///
    /// # Errors
    /// The driver's refusal.
    fn launch(
        &self,
        kernel: impl Into<Kernel>,
        launch: Launch,
        args: &[u64],
    ) -> Result<(), BackendError>;

    /// Launches a typed call of the shared token kernels ([crate::calls]).
    ///
    /// # Errors
    /// The driver's refusal.
    fn call<const N: usize>(
        &self,
        kernel: impl Into<Kernel>,
        call: Call<N>,
    ) -> Result<(), BackendError> {
        self.launch(kernel, (call.grid, call.block), &call.args)
    }
}

/// What a role body asks of the host it runs in, for one pass: its launches ([Launcher]), its
/// rows, its share, the arm's projection and its sums.
pub trait DeviceCx: Launcher {
    /// The arm's projection: a weight [Self::project] multiplies by, in the arm's own format.
    type Linear: ?Sized;

    /// The rows of the pass.
    fn rows(&self) -> usize;

    /// The segments, in row order.
    fn segments(&self) -> impl Iterator<Item = SegmentRows<'_>> + '_;

    /// The rank's share of a sharded group, `None` at one rank.
    fn share(&self) -> Option<TpPlan>;

    /// `output[rows][linear's rows] = input[rows][linear's columns] x linearᵀ` on the arm's
    /// own projection dispatch, for `rows` rows.
    ///
    /// # Errors
    /// A format the arm does not project, or the driver's refusal.
    fn project(
        &self,
        linear: &Self::Linear,
        input: Deviceptr,
        output: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError>;

    /// Sums `rows` rows of `width` partial floats at `partial` over the group in rank order, in
    /// place: nothing at one rank.
    ///
    /// # Errors
    /// More floats than the collective holds, or a collective failure.
    fn sum(&self, partial: Deviceptr, rows: usize, width: usize) -> Result<(), BackendError>;

    /// Copies `bytes` to `to`, ahead of the pass's next launch.
    ///
    /// # Errors
    /// The driver's refusal.
    fn upload(&self, to: Deviceptr, bytes: &[u8]) -> Result<(), BackendError>;
}

/// The token embedding: `q4e_embed` widening a token's BF16 row of the table to f32, `streams`
/// copies of it a row (the residual's hyper-connection streams, one for a plain residual).
///
/// Which rows a rank holds (a vocabulary share), where the ids are staged and what a row the
/// rank does not hold gets are the host's: the body launches the rows it is handed.
#[derive(Debug, Clone, Copy)]
pub struct Embed {
    kernel: Bound,
    table: Deviceptr,
    hidden: usize,
    width: usize,
}

impl Embed {
    /// The embedding's kernel.
    pub const TABLE: KernelTable<1> = KernelTable {
        image: Image::Qwen4Exp,
        symbols: [("q4e_embed", Precision::Bf16)],
    };

    /// `kernel` (the loaded `q4e_embed`) over `table`'s `hidden`-wide rows, `streams` copies a
    /// row.
    ///
    /// # Errors
    /// Another kernel, or no row or stream, by name.
    pub fn new(
        kernel: Bound,
        table: Deviceptr,
        hidden: usize,
        streams: usize,
    ) -> Result<Self, BackendError> {
        let [(symbol, _)] = Self::TABLE.symbols;
        runs(kernel, symbol, "the embedding")?;
        if hidden == 0 || streams == 0 {
            return Err(BackendError::new(
                ROLE,
                format!("an embedding of {hidden}-wide rows in {streams} streams"),
            ));
        }
        Ok(Self {
            kernel,
            table,
            hidden,
            width: hidden * streams,
        })
    }

    /// The floats a row takes in `out`.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// `rows` rows into `out`, row `t` the table row of the id at `tokens[t]` (`[rows]` u32 the
    /// device reads), in one launch.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn rows<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        out: Deviceptr,
        tokens: Deviceptr,
        rows: usize,
    ) -> Result<(), BackendError> {
        cx.call(
            self.kernel,
            calls::embed_rows(
                self.table.0,
                out.0,
                tokens.0,
                (self.hidden, self.width),
                rows,
            ),
        )
    }

    /// Table row `row` into the row at `out`, one launch.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn row<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        out: Deviceptr,
        row: u32,
    ) -> Result<(), BackendError> {
        cx.call(
            self.kernel,
            calls::embed(self.table.0, out.0, row, (self.hidden, self.width)),
        )
    }

    /// Each of `rows` into the next row from `out`, a launch a row.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn each<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        out: Deviceptr,
        rows: &[u32],
    ) -> Result<(), BackendError> {
        rows.iter()
            .enumerate()
            .try_for_each(|(at, &row)| self.row(cx, out.offset(at * self.width * 4), row))
    }
}

/// Whether `kernel` is `symbol`, the kernel `role` runs.
fn runs(kernel: Bound, symbol: &str, role: &str) -> Result<(), BackendError> {
    let name = kernel.kernel().name();
    if name == symbol {
        Ok(())
    } else {
        Err(BackendError::new(
            ROLE,
            format!("{role} runs {symbol}, not {name}"),
        ))
    }
}

/// RMSNorm: `q4e_rmsnorm` over contiguous rows, a block a group, and `q4e_rmsnorm_rows` over
/// rows a stride apart, a block a row's group. The plan ([calls::RmsNorm]) carries the buffers,
/// the epsilon and the flags: zero-centred gains, and per-head groups each taking the same gains.
#[derive(Clone, Copy, Debug)]
pub struct Norm {
    contiguous: Bound,
    strided: Option<Bound>,
}

impl Norm {
    /// The contiguous norm and the strided one.
    pub const TABLE: KernelTable<2> = KernelTable {
        image: Image::Qwen4Exp,
        symbols: [
            ("q4e_rmsnorm", Precision::Bf16),
            ("q4e_rmsnorm_rows", Precision::Bf16),
        ],
    };

    /// `contiguous` (the loaded `q4e_rmsnorm`) and, when the arm loads it, `strided`
    /// (`q4e_rmsnorm_rows`).
    ///
    /// # Errors
    /// Another kernel in either place, by name.
    pub fn new(contiguous: Bound, strided: Option<Bound>) -> Result<Self, String> {
        let [(rows, _), (rows_strided, _)] = Self::TABLE.symbols;
        runs(contiguous, rows, "the norm").map_err(|error| error.to_string())?;
        if let Some(strided) = strided {
            runs(strided, rows_strided, "the strided norm").map_err(|error| error.to_string())?;
        }
        Ok(Self {
            contiguous,
            strided,
        })
    }

    /// The contiguous norm's kernel.
    #[must_use]
    pub const fn contiguous(&self) -> Bound {
        self.contiguous
    }

    /// `plan` over `rows` contiguous rows, one launch.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        plan: calls::RmsNorm,
        rows: usize,
    ) -> Result<(), BackendError> {
        cx.call(self.contiguous, plan.call(rows))
    }

    /// `plan` over `rows` rows, `src_stride` floats apart in the input and `dst_stride` in the
    /// output, one launch.
    ///
    /// # Errors
    /// An arm that loaded no strided norm, by name, or the driver's refusal.
    pub fn strided<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        plan: calls::RmsNorm,
        rows: usize,
        (src_stride, dst_stride): (usize, usize),
    ) -> Result<(), BackendError> {
        let Some(strided) = self.strided else {
            return Err(BackendError::new(
                ROLE,
                "this arm loaded no q4e_rmsnorm_rows for a strided norm",
            ));
        };
        cx.call(strided, plan.rows_call(rows, src_stride, dst_stride))
    }
}

/// LayerNorm with gains and biases (`q4e_layer_norm`): a block a row, `width` floats a row.
#[derive(Clone, Copy, Debug)]
pub struct LayerNorm {
    kernel: Bound,
}

impl LayerNorm {
    /// The layer norm's kernel.
    pub const TABLE: KernelTable<1> = KernelTable {
        image: Image::GlmDecode,
        symbols: [("q4e_layer_norm", Precision::Bf16)],
    };

    /// `kernel`, the loaded `q4e_layer_norm`.
    ///
    /// # Errors
    /// Another kernel, by name.
    pub fn new(kernel: Bound) -> Result<Self, String> {
        let [(symbol, _)] = Self::TABLE.symbols;
        runs(kernel, symbol, "the layer norm").map_err(|error| error.to_string())?;
        Ok(Self { kernel })
    }

    /// `rows` rows of `width` floats from `input`, normed by `(gains, biases)` (BF16) at `eps`
    /// into `output`, one launch.
    ///
    /// # Errors
    /// The driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        input: Deviceptr,
        (gains, biases): (Deviceptr, Deviceptr),
        output: Deviceptr,
        width: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), BackendError> {
        cx.launch(
            self.kernel,
            ((rows as u32, 1, 1), (BLOCK, 1, 1)),
            &[
                input.0,
                gains.0,
                biases.0,
                output.0,
                width as u64,
                u64::from(eps.to_bits()),
            ],
        )
    }
}

/// nGPT's norm (`dory_justnorm`): every row of `width` floats scaled to unit length in place, a
/// block a row.
#[derive(Clone, Copy, Debug)]
pub struct NgptNorm {
    kernel: Bound,
}

impl NgptNorm {
    /// The nGPT norm's kernel.
    pub const TABLE: KernelTable<1> = KernelTable {
        image: Image::Dory,
        symbols: [("dory_justnorm", Precision::Bf16)],
    };

    /// `kernel`, the loaded `dory_justnorm`.
    ///
    /// # Errors
    /// Another kernel, by name.
    pub fn new(kernel: Bound) -> Result<Self, String> {
        let [(symbol, _)] = Self::TABLE.symbols;
        runs(kernel, symbol, "the nGPT norm").map_err(|error| error.to_string())?;
        Ok(Self { kernel })
    }

    /// `rows` rows of `width` floats at `values`, each to unit length in place, one launch.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        values: Deviceptr,
        width: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        cx.launch(
            self.kernel,
            ((1, rows as u32, 1), (BLOCK, 1, 1)),
            &[values.0, width as u64],
        )
    }
}

/// How a branch's rows reach the streams when the mix leaves the branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumPath {
    /// The branch's rows are whole: summed over the group before the mix, or one rank's.
    Summed,
    /// Every rank's partial rows sit in `slots`, `[world][rows][hidden]`, and the mix folds
    /// them in rank order from zero.
    FusedSlots {
        /// The gathered partials.
        slots: Deviceptr,
        /// This rank.
        rank: usize,
        /// The ranks.
        world: usize,
    },
}

/// The residual's hyper-connection streams, as frames around each branch.
///
/// The streams are f32 `[rows][streams][hidden]`. A mixer is stateless and replicated: every rank
/// mixes the same streams.
pub trait StreamMixer<C: DeviceCx> {
    /// Mixes `streams` into the branch's normed input at `out`, under the norm gains at `norm`.
    /// With `fused` the previous [Self::leave] already projected the streams for this mix.
    ///
    /// # Errors
    /// The driver's refusal, or a body that does not run yet.
    fn enter(
        &self,
        cx: &C,
        streams: Deviceptr,
        norm: Deviceptr,
        out: Deviceptr,
        fused: bool,
    ) -> Result<(), BackendError>;

    /// Lands the branch's rows at `branch` back on `streams` along `path`, and, when the host
    /// passes the next branch's mixer as `next`, projects the new streams for it. Returns whether
    /// it did, which the next [Self::enter] is told.
    ///
    /// # Errors
    /// The driver's refusal, or a body that does not run yet.
    fn leave(
        &self,
        cx: &C,
        branch: Deviceptr,
        streams: Deviceptr,
        next: Option<&Self>,
        path: SumPath,
    ) -> Result<bool, BackendError>;

    /// Folds the streams to one row each at `out`, f32 `[rows][hidden]`, for the head.
    ///
    /// # Errors
    /// The driver's refusal, or a body that does not run yet.
    fn fold(&self, cx: &C, streams: Deviceptr, out: Deviceptr) -> Result<(), BackendError>;
}

/// The manifold-constrained mix (mHC): `q4e_mhc_in` before a branch, `q4e_mhc_post` after it.
#[derive(Debug, Clone, Copy)]
pub struct ManifoldMixer {
    geometry: HcGeometry,
    kernels: [Bound; 2],
}

impl ManifoldMixer {
    /// The mix's kernels at the host's `precision`.
    #[must_use]
    pub const fn table(precision: Precision) -> KernelTable<2> {
        KernelTable {
            image: Image::DECODE,
            symbols: [("q4e_mhc_in", precision), ("q4e_mhc_post", precision)],
        }
    }

    /// The geometry of `hyper` over a `hidden`-wide residual whose stream norm takes `norm_eps`,
    /// when the mix's kernels take it.
    ///
    /// # Errors
    /// A hyper-connection that is not the manifold mix, a low-rank or iHC variant, a declared
    /// magnitude, or a stream count or width past the kernels' bounds, each by name.
    pub fn geometry(
        hyper: &HyperConnection,
        hidden: usize,
        norm_eps: f32,
    ) -> Result<HcGeometry, String> {
        let HyperConnection {
            streams,
            lowrank,
            manifold,
            sinkhorn_iters,
            epsilon,
            ihc,
            magnitude,
        } = *hyper;
        if !manifold {
            return Err("the hyper-connection is not the manifold mix (mhc)".to_owned());
        }
        if let Some(rank) = lowrank {
            return Err(format!(
                "hc_lowrank {rank}: the manifold mix's projection is full rank"
            ));
        }
        for (on, switch) in [
            (ihc, Switch::Ihc),
            (magnitude.is_some(), Switch::HcMagnitude),
        ] {
            if on {
                return Err(format!("{switch} is not the manifold mix's arithmetic"));
            }
        }
        if !(2..=MAX_STREAMS).contains(&streams) {
            return Err(format!(
                "{streams} streams; the mix's kernels hold 2..={MAX_STREAMS}"
            ));
        }
        let block = BLOCK as usize;
        if hidden == 0 || !hidden.is_multiple_of(block) || hidden / block > MIX_COLUMNS {
            return Err(format!(
                "hidden_size = {hidden} is not a multiple of {block} of at most {MIX_COLUMNS}, the \
                 columns of every stream a thread of the mixer holds"
            ));
        }
        Ok(HcGeometry {
            hc_mult: streams,
            dim: hidden,
            hc_sinkhorn_iters: sinkhorn_iters,
            hc_eps: epsilon as f32,
            norm_eps,
        })
    }

    /// Opens the mix over `geometry`, its kernels loaded at `precision`.
    ///
    /// # Errors
    /// As [KernelTable::load].
    pub fn open(cuda: &Cuda, geometry: HcGeometry, precision: Precision) -> Result<Self, String> {
        Ok(Self {
            geometry,
            kernels: Self::table(precision).load(cuda)?,
        })
    }

    /// The mix's geometry.
    #[must_use]
    pub const fn geometry_of(&self) -> HcGeometry {
        self.geometry
    }

    /// The loaded `q4e_mhc_in` and `q4e_mhc_post`.
    #[must_use]
    pub const fn kernels(&self) -> [Bound; 2] {
        self.kernels
    }
}

/// The refusal of a body whose passes have not moved in from its arm.
fn not_moved(body: &str, from: &str) -> BackendError {
    BackendError::new(
        ROLE,
        format!("the shared {body} does not run yet: its passes are still {from}"),
    )
}

impl<C: DeviceCx> StreamMixer<C> for ManifoldMixer {
    fn enter(
        &self,
        _cx: &C,
        _streams: Deviceptr,
        _norm: Deviceptr,
        _out: Deviceptr,
        _fused: bool,
    ) -> Result<(), BackendError> {
        Err(not_moved("manifold mix", "GLM's mix_in"))
    }

    fn leave(
        &self,
        _cx: &C,
        _branch: Deviceptr,
        _streams: Deviceptr,
        _next: Option<&Self>,
        _path: SumPath,
    ) -> Result<bool, BackendError> {
        Err(not_moved("manifold mix", "GLM's mix_out"))
    }

    fn fold(&self, _cx: &C, _streams: Deviceptr, _out: Deviceptr) -> Result<(), BackendError> {
        Err(not_moved("manifold mix", "GLM's head fold"))
    }
}

/// The pooled indexer a token mixer holds: keys pooled `pool` to one by a learned gate, scored by
/// the indexer heads, and the best positions picked into the selection the mixer attends over.
#[derive(Debug, Clone, Copy)]
pub struct PooledIndexer {
    geometry: PooledGeometry,
    kernels: [Bound; 3],
}

/// The shape of a [PooledIndexer].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PooledGeometry {
    /// The indexer's query heads.
    pub heads: usize,
    /// Their width.
    pub head_dim: usize,
    /// Keys pooled into one scored key.
    pub pool: usize,
    /// Tokens a query keeps.
    pub topk: usize,
    /// Whether the pool holding the newest key is always kept.
    pub always_select_tail: bool,
}

impl PooledIndexer {
    /// The indexer's kernels at the host's `precision`.
    #[must_use]
    pub const fn table(precision: Precision) -> KernelTable<3> {
        KernelTable {
            image: Image::DECODE,
            symbols: [
                ("q4e_glm_pool", precision),
                ("q4e_glm_index_scores", precision),
                ("q4e_glm_pick", precision),
            ],
        }
    }

    /// The geometry of `indexer`, when the pooled indexer's kernels take it.
    ///
    /// # Errors
    /// A compressed stage, another activation, a candidate pool, or a shape past the kernels'
    /// bounds, each by name.
    pub fn geometry(indexer: &Indexer) -> Result<PooledGeometry, String> {
        if indexer.compressed.is_some() {
            return Err(format!(
                "{} is not the pooled indexer",
                Switch::CompressedIndexer
            ));
        }
        if indexer.activation != Activation::Relu {
            return Err(format!(
                "the indexer's {:?} scores are not the pooled indexer's ReLU",
                indexer.activation
            ));
        }
        if indexer.candidate_pool.is_some() {
            return Err("a hierarchical candidate pool is not the pooled indexer's".to_owned());
        }
        let checks = |checks: &[(&str, usize, usize)]| {
            checks.iter().try_for_each(|&(what, value, bound)| {
                if value == 0 || value > bound {
                    Err(format!(
                        "{what} = {value} is outside the kernels' 1..={bound}"
                    ))
                } else {
                    Ok(())
                }
            })
        };
        // The pool comes first: the budget divides by it.
        checks(&[
            ("the indexer's heads", indexer.heads, MAX_INDEX_HEADS),
            ("the indexer's head width", indexer.head_dim, MAX_INDEX_DIM),
            ("the indexer's pool", indexer.pool, MAX_POOL),
        ])?;
        checks(&[
            (
                "the indexer's pool budget",
                indexer.topk / indexer.pool,
                MAX_BUDGET_BLOCKS,
            ),
            (
                "the tokens a query selects",
                indexer.max_selected(),
                MAX_SELECTED_TOKENS,
            ),
        ])?;
        Ok(PooledGeometry {
            heads: indexer.heads,
            head_dim: indexer.head_dim,
            pool: indexer.pool,
            topk: indexer.topk,
            always_select_tail: indexer.always_select_tail,
        })
    }

    /// Opens the indexer over `geometry`, its kernels loaded at `precision`.
    ///
    /// # Errors
    /// As [KernelTable::load].
    pub fn open(
        cuda: &Cuda,
        geometry: PooledGeometry,
        precision: Precision,
    ) -> Result<Self, String> {
        Ok(Self {
            geometry,
            kernels: Self::table(precision).load(cuda)?,
        })
    }

    /// The indexer's geometry.
    #[must_use]
    pub const fn geometry_of(&self) -> PooledGeometry {
        self.geometry
    }

    /// The loaded `q4e_glm_pool`, `q4e_glm_index_scores` and `q4e_glm_pick`.
    #[must_use]
    pub const fn kernels(&self) -> [Bound; 3] {
        self.kernels
    }

    /// Appends the pass's pooled keys to each segment's index cache, its `regions`.
    ///
    /// # Errors
    /// The driver's refusal, or a body that does not run yet.
    pub fn cache<C: DeviceCx>(&self, _cx: &C, _regions: &[Deviceptr]) -> Result<(), BackendError> {
        Err(not_moved("pooled indexer", "GLM's index_cache"))
    }

    /// Scores the cached keys against the queries at `query` and writes each row's picked
    /// positions to `selection`.
    ///
    /// # Errors
    /// The driver's refusal, or a body that does not run yet.
    pub fn pick<C: DeviceCx>(
        &self,
        _cx: &C,
        _query: Deviceptr,
        _selection: Deviceptr,
    ) -> Result<(), BackendError> {
        Err(not_moved("pooled indexer", "GLM's index_pick"))
    }
}

/// An Engram lookup that lands on the layer input: hashed n-gram rows gathered, projected, gated
/// and written into the residual.
#[derive(Debug, Clone)]
pub struct EngramSide {
    geometry: EngramGeometry,
    table: KernelTable<3>,
}

impl EngramSide {
    /// The lookup's kernels: the gate and write at the host's `precision`, and the narrow and
    /// widen between the residual's f32 and the table's BF16 at their own.
    #[must_use]
    pub const fn table(precision: Precision) -> KernelTable<3> {
        KernelTable {
            image: Image::Deepseek,
            symbols: [
                ("engram_gate_write_sm121", precision),
                ("engram_narrow_bf16", Precision::Bf16),
                ("engram_widen_f32", Precision::F32),
            ],
        }
    }

    /// The lookup over `geometry`, its kernel table checked at `precision`. The table is not
    /// loaded: DeepSeek's image is PTX this build does not embed, and the arm loads it.
    ///
    /// # Errors
    /// A geometry with no tables, or a table [KernelTable::check] refuses.
    pub fn new(geometry: EngramGeometry, precision: Precision) -> Result<Self, String> {
        if geometry.tables.is_empty() || geometry.heads == 0 || geometry.head_dim == 0 {
            return Err("an Engram side with no tables, heads or head width".to_owned());
        }
        let table = Self::table(precision);
        table.check()?;
        Ok(Self { geometry, table })
    }

    /// The lookup's geometry.
    #[must_use]
    pub const fn geometry(&self) -> &EngramGeometry {
        &self.geometry
    }

    /// The lookup's kernel table.
    #[must_use]
    pub const fn kernel_table(&self) -> KernelTable<3> {
        self.table
    }

    /// Gathers the rows the pass's n-grams hash to, projects them, and gates and writes them
    /// into the layer input at `residual`.
    ///
    /// # Errors
    /// The driver's refusal, or a body that does not run yet.
    pub fn land<C: DeviceCx>(&self, _cx: &C, _residual: Deviceptr) -> Result<(), BackendError> {
        Err(not_moved("Engram side", "DeepSeek's gate and write"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_model::architecture::{CompressedStage, KeyPooling};

    fn manifold() -> HyperConnection {
        HyperConnection {
            streams: 4,
            lowrank: None,
            manifold: true,
            sinkhorn_iters: 20,
            epsilon: 1e-6,
            ihc: false,
            magnitude: None,
        }
    }

    fn pooled() -> Indexer {
        Indexer {
            heads: 32,
            head_dim: 128,
            kv_heads: 1,
            topk: 2048,
            owns_keys: true,
            reuse: false,
            pool: 4,
            always_select_tail: true,
            activation: Activation::Relu,
            candidate_pool: None,
            compressed: None,
        }
    }

    /// A pass with nothing on a device: the bodies' refusals make no launch.
    struct NoDevice;

    impl Launcher for NoDevice {
        fn launch(&self, _: impl Into<Kernel>, _: Launch, _: &[u64]) -> Result<(), BackendError> {
            unreachable!("no body launches yet")
        }
    }

    impl DeviceCx for NoDevice {
        type Linear = ();

        fn rows(&self) -> usize {
            1
        }

        fn segments(&self) -> impl Iterator<Item = SegmentRows<'_>> + '_ {
            std::iter::empty()
        }

        fn share(&self) -> Option<TpPlan> {
            None
        }

        fn project(
            &self,
            (): &(),
            _: Deviceptr,
            _: Deviceptr,
            _: usize,
        ) -> Result<(), BackendError> {
            unreachable!("no body projects yet")
        }

        fn sum(&self, _: Deviceptr, _: usize, _: usize) -> Result<(), BackendError> {
            unreachable!("no body sums yet")
        }

        fn upload(&self, _: Deviceptr, _: &[u8]) -> Result<(), BackendError> {
            unreachable!("no body uploads yet")
        }
    }

    #[test]
    fn every_body_table_is_carried_at_the_hosts_precisions() {
        for precision in [Precision::Bf16, Precision::Fp8E4M3] {
            ManifoldMixer::table(precision).check().expect("the mix");
            PooledIndexer::table(precision)
                .check()
                .expect("the indexer");
            EngramSide::table(precision).check().expect("the lookup");
        }
        EngramSide::table(Precision::MxFp8)
            .check()
            .expect("DeepSeek's MXFP8 lookup");
        let error = PooledIndexer::table(Precision::NvFp4)
            .check()
            .expect_err("no NVFP4 indexer");
        assert!(error.contains("q4e_glm_pool"), "{error}");
        let error = ManifoldMixer::table(Precision::F32)
            .check()
            .expect_err("no f32 mix");
        assert!(error.contains("q4e_mhc_in"), "{error}");
    }

    #[test]
    fn the_manifold_geometry_is_the_mixes_and_refuses_another_mix_by_name() {
        let geometry = ManifoldMixer::geometry(&manifold(), 4096, 1e-5).expect("GLM's shape");
        assert_eq!(geometry.hc_mix_count(), 24);
        assert_eq!(geometry.hc_width(), 4 * 4096);
        for (hyper, hidden, named) in [
            (
                HyperConnection {
                    ihc: true,
                    ..manifold()
                },
                4096,
                "enable_ihc",
            ),
            (
                HyperConnection {
                    magnitude: Some(2.0),
                    ..manifold()
                },
                4096,
                "hc_magnitude",
            ),
            (
                HyperConnection {
                    manifold: false,
                    ..manifold()
                },
                4096,
                "not the manifold mix",
            ),
            (
                HyperConnection {
                    lowrank: Some(4),
                    ..manifold()
                },
                4096,
                "hc_lowrank",
            ),
            (
                HyperConnection {
                    streams: 6,
                    ..manifold()
                },
                4096,
                "6 streams",
            ),
            (manifold(), 4096 + 128, "hidden_size"),
            (manifold(), 17 * 256, "hidden_size"),
        ] {
            let error = ManifoldMixer::geometry(&hyper, hidden, 1e-5).expect_err(named);
            assert!(error.contains(named), "{error}");
        }
    }

    #[test]
    fn the_pooled_geometry_refuses_a_compressed_stage_and_a_shape_past_its_bounds() {
        let geometry = PooledIndexer::geometry(&pooled()).expect("GLM's shape");
        assert_eq!((geometry.heads, geometry.pool), (32, 4));
        let compressed = Indexer {
            compressed: Some(CompressedStage {
                heads: 16,
                head_dim: 128,
                ratio: 4,
                pooling: KeyPooling::Gated,
                candidates: 2048,
                group_topk: 512,
                reuse_fine: false,
                skip_fine: true,
            }),
            ..pooled()
        };
        let error = PooledIndexer::geometry(&compressed).expect_err("a compressed stage");
        assert!(error.contains("use_compressed_indexer"), "{error}");
        for (indexer, named) in [
            (
                Indexer {
                    heads: 64,
                    ..pooled()
                },
                "heads = 64",
            ),
            (
                Indexer {
                    pool: 32,
                    ..pooled()
                },
                "pool = 32",
            ),
            (
                Indexer {
                    topk: 4096,
                    ..pooled()
                },
                "selects",
            ),
            (
                Indexer {
                    activation: Activation::Silu,
                    ..pooled()
                },
                "Silu",
            ),
        ] {
            let error = PooledIndexer::geometry(&indexer).expect_err(named);
            assert!(error.contains(named), "{error}");
        }
    }

    #[test]
    fn a_body_whose_passes_have_not_moved_refuses_by_name() {
        let geometry = EngramGeometry {
            tables: vec![1024],
            max_ngram: 3,
            heads: 4,
            head_dim: 64,
            vocab_size: 1024,
            compressed_vocab: 1024,
            pad_id: 0,
            head_rows: Vec::new(),
        };
        let side = EngramSide::new(geometry.clone(), Precision::Bf16).expect("the lookup");
        let error = side.land(&NoDevice, Deviceptr(0)).expect_err("not moved");
        assert!(error.to_string().contains("Engram side"), "{error}");
        let empty = EngramGeometry {
            tables: Vec::new(),
            ..geometry
        };
        let error = EngramSide::new(empty, Precision::Bf16).expect_err("no tables");
        assert!(error.contains("no tables"), "{error}");
        let error = not_moved("manifold mix", "GLM's mix_in").to_string();
        assert!(error.contains("GLM's mix_in"), "{error}");
    }
}
