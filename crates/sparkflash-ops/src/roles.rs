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
    /// A verify pass's snapshot slabs, one for each of the part's recurrent regions in their
    /// order: slab r holds `rows` copies of its region a region's bytes apart, copy t the region
    /// as it stood after the segment's row t. Empty when the pass keeps none.
    pub snapshots: &'a [Deviceptr],
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

    /// [Self::project] into rows `stride` floats apart, a row's first `linear's rows` values
    /// written: a body that lays several projections side by side in one row (KDA's q, k and v
    /// before their conv). A host whose dispatch takes no output stride refuses by name.
    ///
    /// # Errors
    /// A host with no strided projection, a format the arm does not project, or the driver's
    /// refusal.
    fn project_strided(
        &self,
        linear: &Self::Linear,
        input: Deviceptr,
        (output, stride): (Deviceptr, usize),
        rows: usize,
    ) -> Result<(), BackendError> {
        let _ = (linear, input, output);
        Err(BackendError::new(
            ROLE,
            format!(
                "a body asked for {rows} projected rows {stride} floats apart, and this host's projection writes packed rows only"
            ),
        ))
    }

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
pub(crate) fn runs(kernel: Bound, symbol: &str, role: &str) -> Result<(), BackendError> {
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

/// The device rows the token expert passes read and write.
#[derive(Clone, Copy, Debug)]
pub struct ExpertRows {
    /// The chosen experts, `[rows][top_k]` i32.
    pub indices: Deviceptr,
    /// The routing weights, `[rows][weight_stride]` f32.
    pub weights: Deviceptr,
    /// The rows the experts read, `[rows][hidden]` f32.
    pub input: Deviceptr,
    /// The plan's run order, `[rows * top_k]` i32.
    pub sorted: Deviceptr,
    /// The plan's run bounds.
    pub runs: Deviceptr,
    /// The activations between gate/up and down, `[rows * top_k][intermediate]` f32.
    pub activated: Deviceptr,
    /// The weighted down rows, `[rows * top_k][hidden]` f32, which the combine sums.
    pub partials: Deviceptr,
}

/// The routed experts' shape the token passes take.
#[derive(Clone, Copy, Debug)]
pub struct ExpertShape {
    /// The experts the table holds.
    pub experts: usize,
    /// The slots a row routes to.
    pub top_k: usize,
    /// A row's floats.
    pub hidden: usize,
    /// An expert's intermediate width.
    pub intermediate: usize,
    /// The floats between one row's routing weights and the next's.
    pub weight_stride: usize,
}

/// The routed experts on the token kernels: the shared plan's runs of one expert's slots
/// ([crate::moe::MoePlanCall::runs]), then a gate/up and a down pass a run in one table format.
/// A format is its named bodies; one arm's pairing of another's never runs.
#[derive(Clone, Copy, Debug)]
pub struct TokenExperts {
    plan: Bound,
    gate_up: Bound,
    gate_up_clamped: Option<Bound>,
    down: Bound,
    scaled: bool,
}

impl TokenExperts {
    /// The formats: gate/up, its clamped form under a `swiglu_limit`, down, and whether the
    /// bodies read the NVFP4 scale table.
    pub const FORMS: [(&'static str, Option<&'static str>, &'static str, bool); 3] = [
        ("q4e_expert_gate_up", None, "q4e_expert_down", true),
        (
            "q4e_expert_gate_up_fp8",
            Some("q4e_expert_gate_up_fp8_clamped"),
            "q4e_expert_down_fp8",
            false,
        ),
        (
            "q4e_expert_gate_up_bf16",
            Some("q4e_expert_gate_up_bf16_clamped"),
            "q4e_expert_down_bf16",
            false,
        ),
    ];

    /// `plan` (the loaded `moe_plan`) and one format's bodies; `gate_up_clamped` is the arm's
    /// clamped body when it loads one.
    ///
    /// # Errors
    /// A plan that is not `moe_plan`, or bodies of no one format, by name.
    pub fn new(
        plan: Bound,
        gate_up: Bound,
        gate_up_clamped: Option<Bound>,
        down: Bound,
    ) -> Result<Self, String> {
        runs(plan, "moe_plan", "the expert plan").map_err(|error| error.to_string())?;
        let name = gate_up.kernel().name();
        let Some(&(_, clamped, down_symbol, scaled)) =
            Self::FORMS.iter().find(|(symbol, ..)| *symbol == name)
        else {
            return Err(format!(
                "no token expert format has the gate/up body {name}"
            ));
        };
        runs(down, down_symbol, "this format's down pass").map_err(|error| error.to_string())?;
        if let Some(kernel) = gate_up_clamped {
            let Some(symbol) = clamped else {
                return Err(format!(
                    "{name} has no clamped form, not {}",
                    kernel.kernel().name()
                ));
            };
            runs(kernel, symbol, "this format's clamped gate/up")
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            plan,
            gate_up,
            gate_up_clamped,
            down,
            scaled,
        })
    }

    /// Whether the bodies read the NVFP4 scale table.
    #[must_use]
    pub const fn scaled(&self) -> bool {
        self.scaled
    }

    /// The plan, gate/up and down over `count` rows of `table` (and, for the scaled format,
    /// `scales`), the gate/up clamped at `limit` when one is given.
    ///
    /// # Errors
    /// A limit on a format with no clamped body, by name, or the driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        (table, scales): (Deviceptr, Deviceptr),
        rows: &ExpertRows,
        shape: ExpertShape,
        count: usize,
        limit: Option<f32>,
    ) -> Result<(), BackendError> {
        let ExpertShape {
            experts,
            top_k,
            hidden,
            intermediate,
            weight_stride,
        } = shape;
        let plan = crate::moe::MoePlanCall::runs(
            rows.indices.0,
            rows.sorted.0,
            rows.runs.0,
            (count * top_k, experts, top_k),
        );
        cx.launch(self.plan, crate::moe::MoePlanCall::launch(), &plan.args())?;
        let mut gate_up = Vec::with_capacity(11);
        gate_up.push(table.0);
        if self.scaled {
            gate_up.push(scales.0);
        }
        gate_up.extend([
            rows.indices.0,
            rows.sorted.0,
            rows.runs.0,
            rows.input.0,
            rows.activated.0,
            top_k as u64,
            hidden as u64,
            intermediate as u64,
        ]);
        let grid = crate::moe::token::gate_up_grid(intermediate, top_k, count);
        match limit {
            None => cx.launch(self.gate_up, grid, &gate_up)?,
            Some(limit) => {
                let Some(clamped) = self.gate_up_clamped else {
                    return Err(BackendError::new(
                        ROLE,
                        format!(
                            "a token expert pass under a swiglu_limit: {} has no clamped form",
                            self.gate_up.kernel().name()
                        ),
                    ));
                };
                gate_up.push(u64::from(limit.to_bits()));
                cx.launch(clamped, grid, &gate_up)?;
            }
        }
        let mut down = Vec::with_capacity(12);
        down.push(table.0);
        if self.scaled {
            down.push(scales.0);
        }
        down.extend([
            rows.indices.0,
            rows.sorted.0,
            rows.runs.0,
            rows.weights.0,
            rows.activated.0,
            rows.partials.0,
            top_k as u64,
            hidden as u64,
            intermediate as u64,
            weight_stride as u64,
        ]);
        cx.launch(
            self.down,
            crate::moe::token::down_grid(hidden, top_k, count),
            &down,
        )
    }
}

/// Where a combined MoE row also lands: the widened residual's streams, each by its injection
/// weight, or nowhere (the combine writes the branch only).
#[derive(Clone, Copy, Debug)]
pub struct Injection {
    /// The widened residual the row is injected into; `None` writes the branch only.
    pub hyper: Option<Deviceptr>,
    /// The streams' injection weights.
    pub weights: Deviceptr,
    /// The streams.
    pub streams: usize,
}

impl Injection {
    /// No streams: the combine writes the branch only.
    pub const NONE: Self = Self {
        hyper: None,
        weights: Deviceptr(0),
        streams: 0,
    };
}

/// The MoE combine (`q4e_moe_combine`): a row's weighted expert partials plus its shared expert's
/// row, the latter under its sigmoid gate when the layer has one.
#[derive(Clone, Copy, Debug)]
pub struct Combine {
    kernel: Bound,
}

impl Combine {
    /// The combine's kernel.
    pub const TABLE: KernelTable<1> = KernelTable {
        image: Image::Qwen4Moe,
        symbols: [("q4e_moe_combine", Precision::Bf16)],
    };

    /// `kernel`, the loaded `q4e_moe_combine`.
    ///
    /// # Errors
    /// Another kernel, by name.
    pub fn new(kernel: Bound) -> Result<Self, String> {
        let [(symbol, _)] = Self::TABLE.symbols;
        runs(kernel, symbol, "the MoE combine").map_err(|error| error.to_string())?;
        Ok(Self { kernel })
    }

    /// `rows` combined rows into `out` from `partials` (`[rows * top_k][hidden]`) and `shared`
    /// (`[rows][hidden]`), gated by `shared_gate`'s logits when given, and injected per
    /// `injection`.
    ///
    /// # Errors
    /// The driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        out: Deviceptr,
        (partials, shared, shared_gate): (Deviceptr, Deviceptr, Option<Deviceptr>),
        injection: Injection,
        (top_k, hidden): (usize, usize),
        rows: usize,
    ) -> Result<(), BackendError> {
        cx.launch(
            self.kernel,
            crate::moe::combine_grid(hidden, rows),
            &[
                out.0,
                partials.0,
                shared.0,
                shared_gate.map_or(0, |gate| gate.0),
                injection.hyper.map_or(0, |hyper| hyper.0),
                injection.weights.0,
                top_k as u64,
                hidden as u64,
                injection.streams as u64,
            ],
        )
    }
}

/// A router's rule: the kernel it names and the arguments that kernel takes. Each router keeps
/// its own scoring, its own pick order and its own weights; a rule runs only on its own kernel, so
/// no arm's routing takes another's arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouterRule {
    /// `q4e_route` (the dense arm): the score [crate::moe::router_score] names, the bias where the
    /// layer has one, the picked weights renormalised when `renormalise` and times `scaling`;
    /// `top_k` weights a row. A 256-thread block a row.
    Route {
        /// The score's code.
        score: u64,
        /// The selection bias.
        bias: Option<Deviceptr>,
        /// The picked weights sum to one.
        renormalise: bool,
        /// The routed scaling factor.
        scaling: f32,
    },
    /// `q4e_router` (Qwen): a row's probabilities over every expert into the weights (which may
    /// be the logits), and its top-k indices. A 256-thread block a row.
    Softmax,
    /// `q4e_noaux_router` (GLM): the biased choice, its weights renormalised when `renormalise`
    /// and times `scaling`. A warp a row.
    NoauxAscending {
        /// The selection bias.
        bias: Deviceptr,
        /// The picked weights sum to one.
        renormalise: bool,
        /// The routed scaling factor.
        scaling: f32,
    },
    /// `inkling_router_topk` (Inkling): the sigmoid router, its shares times `scale`. A
    /// 128-thread block a row shares the walk over the experts and keeps the sequential scan's
    /// choice.
    InklingTopk {
        /// The selection bias.
        bias: Deviceptr,
        /// The route scale.
        scale: f32,
    },
}

impl RouterRule {
    /// The kernel the rule runs on.
    #[must_use]
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Route { .. } => "q4e_route",
            Self::Softmax => "q4e_router",
            Self::NoauxAscending { .. } => "q4e_noaux_router",
            Self::InklingTopk { .. } => "inkling_router_topk",
        }
    }
}

/// The rows a router reads and writes: the logits, the picked experts' indices and their weights.
#[derive(Clone, Copy, Debug)]
pub struct RouterRows {
    /// The router logits, `[rows][experts]` f32.
    pub logits: Deviceptr,
    /// The picked experts, `[rows][top_k]`.
    pub indices: Deviceptr,
    /// The routing weights (a rule's own stride).
    pub weights: Deviceptr,
}

/// A loaded router, one of the [RouterRule]s' kernels.
#[derive(Clone, Copy, Debug)]
pub struct Router {
    kernel: Bound,
}

impl Router {
    /// The routers a rule names.
    pub const SYMBOLS: [&'static str; 4] = [
        "q4e_route",
        "q4e_router",
        "q4e_noaux_router",
        "inkling_router_topk",
    ];

    /// `kernel`, one of [Self::SYMBOLS].
    ///
    /// # Errors
    /// Another kernel, by name.
    pub fn new(kernel: Bound) -> Result<Self, String> {
        let name = kernel.kernel().name();
        if Self::SYMBOLS.contains(&name) {
            Ok(Self { kernel })
        } else {
            Err(format!(
                "a router runs one of {:?}, not {name}",
                Self::SYMBOLS
            ))
        }
    }

    /// `rows` rows of `experts` logits routed to `top_k` experts under `rule`.
    ///
    /// # Errors
    /// A rule for another router, by name, or the driver's refusal.
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        rule: RouterRule,
        rows_in: &RouterRows,
        (experts, top_k): (usize, usize),
        rows: usize,
    ) -> Result<(), BackendError> {
        let name = self.kernel.kernel().name();
        if name != rule.symbol() {
            return Err(BackendError::new(
                ROLE,
                format!("{name} takes its own routing rule, not {}'s", rule.symbol()),
            ));
        }
        let RouterRows {
            logits,
            indices,
            weights,
        } = *rows_in;
        let grid = (rows as u32, 1, 1);
        let (experts, top_k) = (experts as u64, top_k as u64);
        match rule {
            RouterRule::Route {
                score,
                bias,
                renormalise,
                scaling,
            } => cx.launch(
                self.kernel,
                (grid, (256, 1, 1)),
                &[
                    logits.0,
                    bias.map_or(0, |bias| bias.0),
                    indices.0,
                    weights.0,
                    experts,
                    top_k,
                    score,
                    u64::from(renormalise),
                    u64::from(scaling.to_bits()),
                ],
            ),
            RouterRule::Softmax => cx.launch(
                self.kernel,
                (grid, (256, 1, 1)),
                &[logits.0, weights.0, indices.0, experts, top_k],
            ),
            RouterRule::NoauxAscending {
                bias,
                renormalise,
                scaling,
            } => cx.launch(
                self.kernel,
                (grid, (32, 1, 1)),
                &[
                    logits.0,
                    bias.0,
                    indices.0,
                    weights.0,
                    experts,
                    top_k,
                    u64::from(renormalise),
                    u64::from(scaling.to_bits()),
                ],
            ),
            RouterRule::InklingTopk { bias, scale } => cx.launch(
                self.kernel,
                (grid, (128, 1, 1)),
                &[
                    logits.0,
                    bias.0,
                    indices.0,
                    weights.0,
                    experts,
                    top_k,
                    u64::from(scale.to_bits()),
                ],
            ),
        }
    }
}

/// SwiGLU over split gate and up rows: `q4e_swiglu`, and under a release's `swiglu_limit`
/// `q4e_swiglu_clamped` ([calls::swiglu_clamped]) where the arm loads it. A limit on an arm
/// without the clamped body is refused by name.
#[derive(Clone, Copy, Debug)]
pub struct SplitSwiGlu {
    plain: Bound,
    clamped: Option<Bound>,
}

impl SplitSwiGlu {
    /// The plain body and the clamped one.
    pub const SYMBOLS: [&'static str; 2] = ["q4e_swiglu", "q4e_swiglu_clamped"];

    /// The loaded [Self::SYMBOLS], the clamped one where the arm takes a limit.
    ///
    /// # Errors
    /// Another kernel in a place, by name.
    pub fn new(plain: Bound, clamped: Option<Bound>) -> Result<Self, String> {
        let [plain_symbol, clamped_symbol] = Self::SYMBOLS;
        runs(plain, plain_symbol, "the SwiGLU").map_err(|error| error.to_string())?;
        if let Some(clamped) = clamped {
            runs(clamped, clamped_symbol, "the clamped SwiGLU")
                .map_err(|error| error.to_string())?;
        }
        Ok(Self { plain, clamped })
    }

    /// `out = silu(gate) * up` over `count` floats, the gate and up clamped to `limit` first when
    /// given.
    ///
    /// # Errors
    /// A limit without the clamped body, by name, or the driver's refusal.
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        (gate, up, out): (Deviceptr, Deviceptr, Deviceptr),
        count: usize,
        limit: Option<f32>,
    ) -> Result<(), BackendError> {
        match (limit, self.clamped) {
            (None, _) => cx.call(self.plain, calls::swiglu(gate.0, up.0, out.0, count)),
            (Some(limit), Some(clamped)) => cx.call(
                clamped,
                calls::swiglu_clamped(gate.0, up.0, out.0, count, limit),
            ),
            (Some(_), None) => Err(BackendError::new(
                ROLE,
                "a swiglu_limit needs q4e_swiglu_clamped, which this arm does not load",
            )),
        }
    }
}

/// The SwiGLU a [GatedMlp] runs between its projections: `out = silu(gate) * up` over `count`
/// f32 elements, the gate clamped above and the up both ways at `limit` first when it is set.
///
/// The role's own body is [SplitSwiGlu]; a host that loads another kernel of the same
/// arithmetic (DeepSeek's `opt_rows_shared_swiglu`) brings that instead.
pub trait SwiGluBody {
    /// The SwiGLU on `cx`'s stream.
    ///
    /// # Errors
    /// A limit the body has no clamped form for, or the driver's refusal.
    fn swiglu<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        io: (Deviceptr, Deviceptr, Deviceptr),
        count: usize,
        limit: Option<f32>,
    ) -> Result<(), BackendError>;
}

impl SwiGluBody for SplitSwiGlu {
    fn swiglu<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        io: (Deviceptr, Deviceptr, Deviceptr),
        count: usize,
        limit: Option<f32>,
    ) -> Result<(), BackendError> {
        self.run(cx, io, count, limit)
    }
}

/// A gated MLP's three projections in the arm's own format, `width` the gate's (and the up's)
/// rows as the rank holds them: `down(silu(gate(x)) * up(x))`.
///
/// The dense arm's MLPs and shared experts and DeepSeek's dense-MLP blocks and shared experts on
/// its dense format route run this one body: each host brings its projection dispatch
/// ([DeviceCx::project]) and its SwiGLU ([SwiGluBody]).
#[derive(Debug, Clone, Copy)]
pub struct GatedMlp<'a, L: ?Sized> {
    /// The gate projection.
    pub gate: &'a L,
    /// The up projection.
    pub up: &'a L,
    /// The down projection.
    pub down: &'a L,
    /// The gate's rows as the rank holds them.
    pub width: usize,
}

/// Where a [GatedMlp] pass reads and writes: the normed rows, the gate's and the up's scratch
/// (the up's also takes the SwiGLU's product) and the down's output.
#[derive(Debug, Clone, Copy)]
pub struct MlpRows {
    /// The normed rows the gate and the up read.
    pub input: Deviceptr,
    /// The gate's scratch, `[rows][width]` f32.
    pub gate: Deviceptr,
    /// The up's scratch, `[rows][width]` f32, then the SwiGLU's product the down reads.
    pub up: Deviceptr,
    /// The down's output.
    pub output: Deviceptr,
}

impl<L: ?Sized> GatedMlp<'_, L> {
    /// The MLP over `rows` rows on `cx`, the gate and up clamped to `limit` first when given.
    ///
    /// # Errors
    /// A projection the arm does not run, a limit without the clamped body, or the driver's
    /// refusal.
    pub fn run<C: DeviceCx<Linear = L>, S: SwiGluBody + ?Sized>(
        &self,
        cx: &C,
        swiglu: &S,
        at: &MlpRows,
        rows: usize,
        limit: Option<f32>,
    ) -> Result<(), BackendError> {
        cx.project(self.gate, at.input, at.gate, rows)?;
        cx.project(self.up, at.input, at.up, rows)?;
        swiglu.swiglu(cx, (at.gate, at.up, at.up), rows * self.width, limit)?;
        cx.project(self.down, at.up, at.output, rows)
    }
}

/// SwiGLU over fused gate-up rows under a limit, `q4e_swiglu_limited` (GLM's MLPs and experts):
/// each `[2 * width]` row's halves into a `[width]` row, a block a 256-float slice of a row.
#[derive(Clone, Copy, Debug)]
pub struct FusedSwiGlu {
    kernel: Bound,
}

impl FusedSwiGlu {
    /// The kernel.
    pub const SYMBOL: &'static str = "q4e_swiglu_limited";

    /// The loaded [Self::SYMBOL].
    ///
    /// # Errors
    /// Another kernel, by name.
    pub fn new(kernel: Bound) -> Result<Self, String> {
        runs(kernel, Self::SYMBOL, "the fused SwiGLU").map_err(|error| error.to_string())?;
        Ok(Self { kernel })
    }

    /// `rows` fused rows of `gate_up` into `out`, `width` wide, under `limit`.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn run<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        (gate_up, out): (Deviceptr, Deviceptr),
        width: usize,
        limit: f32,
        rows: usize,
    ) -> Result<(), BackendError> {
        cx.launch(
            self.kernel,
            ((calls::flat_blocks(width), rows as u32, 1), (BLOCK, 1, 1)),
            &[gate_up.0, out.0, width as u64, u64::from(limit.to_bits())],
        )
    }
}

/// A grouped-query attention layer's shape on this rank: its query and key/value heads, a head's
/// floats, whether the query projection interleaves a sigmoid output gate with each head, and the
/// rotary's rotated floats and base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GqaShape {
    /// The query heads.
    pub heads: usize,
    /// The key/value heads.
    pub kv_heads: usize,
    /// A head's floats.
    pub head_dim: usize,
    /// The query projection interleaves a gate head with each query head.
    pub gated: bool,
    /// The floats of a head that rotate (none below two).
    pub rotary: usize,
    /// The rotary base.
    pub theta: f32,
}

impl GqaShape {
    /// A query row's floats.
    #[must_use]
    pub const fn query_width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// A key (or value) row's floats.
    #[must_use]
    pub const fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// The f32 rows a grouped-query attention reads and writes, a row a token.
#[derive(Clone, Copy, Debug)]
pub struct GqaRows {
    /// A gated layer's interleaved query-and-gate projection, which [GqaAttention::heads] splits.
    pub interleaved: Deviceptr,
    /// The queries, `[rows][query_width]`.
    pub query: Deviceptr,
    /// A gated layer's gates, `[rows][query_width]`.
    pub gate: Deviceptr,
    /// The keys, `[rows][kv_width]`.
    pub key: Deviceptr,
    /// The values, `[rows][kv_width]`.
    pub value: Deviceptr,
    /// The split partials of one launch's rows ([calls::FullAttention::partial_floats]).
    pub partials: Deviceptr,
    /// The attended heads, `[rows][query_width]`.
    pub attended: Deviceptr,
}

/// Grouped-query attention after its projections: the gate deinterleave and the per-head q/k
/// RMSNorms over every row ([Self::heads]); then a sequence at a time the rotary and the BF16
/// cache append ([Self::append]), and the causal attention over the cache with its combine
/// (`q4e_attn_full`, `q4e_attn_full_combine`, [Self::attend]).
///
/// The attention takes a sequence's rows in one launch, or, configured with a chunk, `chunk` rows
/// a launch: a launch's split count follows its own last row, so the two differ in bits, and
/// each arm keeps the one its records hold.
#[derive(Clone, Copy, Debug)]
pub struct GqaAttention {
    deinterleave: Bound,
    rope: Bound,
    cache: Bound,
    attention: Bound,
    combine: Bound,
    chunk: Option<usize>,
}

impl GqaAttention {
    /// The kernels in the order [Self::new] takes them.
    pub const SYMBOLS: [&'static str; 5] = [
        "q4e_deinterleave",
        "q4e_rope",
        "q4e_cache_bf16",
        "q4e_attn_full",
        "q4e_attn_full_combine",
    ];

    /// The loaded [Self::SYMBOLS], the attention taking `chunk` rows a launch when given.
    ///
    /// # Errors
    /// Another kernel in a place, by name, or a zero chunk.
    pub fn new(kernels: [Bound; 5], chunk: Option<usize>) -> Result<Self, String> {
        for (kernel, symbol) in kernels.iter().zip(Self::SYMBOLS) {
            runs(*kernel, symbol, "grouped-query attention").map_err(|error| error.to_string())?;
        }
        if chunk == Some(0) {
            return Err("grouped-query attention takes no zero-row chunk".to_owned());
        }
        let [deinterleave, rope, cache, attention, combine] = kernels;
        Ok(Self {
            deinterleave,
            rope,
            cache,
            attention,
            combine,
            chunk,
        })
    }

    /// The rotary's kernel, which an arm's other heads (an indexer's) also take.
    #[must_use]
    pub const fn rope(&self) -> Bound {
        self.rope
    }

    /// Over `count` rows: a gated layer's deinterleave of `rows.interleaved` into the queries and
    /// gates, then the query heads' RMSNorm and the key heads' (each where the layer has its
    /// gains) in place, a head a group under the same gains, zero-centred when `zero_centred`.
    ///
    /// # Errors
    /// The driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn heads<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        norm: &Norm,
        shape: GqaShape,
        rows: &GqaRows,
        (q_norm, k_norm): (Option<Deviceptr>, Option<Deviceptr>),
        (eps, zero_centred): (f32, bool),
        count: usize,
    ) -> Result<(), BackendError> {
        if shape.gated {
            cx.call(
                self.deinterleave,
                calls::deinterleave(
                    rows.interleaved.0,
                    (rows.query.0, rows.gate.0),
                    (shape.heads, shape.head_dim),
                    count,
                ),
            )?;
        }
        for (gains, values, width) in [
            (q_norm, rows.query, shape.query_width()),
            (k_norm, rows.key, shape.kv_width()),
        ] {
            if let Some(gains) = gains {
                norm.run(
                    cx,
                    calls::RmsNorm::new(values.0, gains.0, values.0, width, eps)
                        .zero_centred(zero_centred)
                        .groups(shape.head_dim, true),
                    count,
                )?;
            }
        }
        Ok(())
    }

    /// One sequence's `count` rows from row `first`, at `position` on: the queries' and keys'
    /// rotary (when two floats or more rotate), then the keys and values narrowed to BF16 into
    /// `keys` and `values` (`[capacity][kv_width]`) at `position`.
    ///
    /// # Errors
    /// The driver's refusal.
    pub fn append<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        shape: GqaShape,
        rows: &GqaRows,
        (keys, values): (Deviceptr, Deviceptr),
        position: usize,
        (first, count): (usize, usize),
    ) -> Result<(), BackendError> {
        let (query_width, kv_width) = (shape.query_width(), shape.kv_width());
        let row = |base: Deviceptr, width: usize| base.0 + (first * width * 4) as u64;
        if shape.rotary >= 2 {
            for (values, heads, width) in [
                (rows.query, shape.heads, query_width),
                (rows.key, shape.kv_heads, kv_width),
            ] {
                let rope = calls::Rope {
                    rows: row(values, width),
                    heads,
                    head_dim: shape.head_dim,
                    rotary: shape.rotary,
                    position,
                    theta: shape.theta,
                    row_stride: width,
                    tokens: count,
                };
                cx.call(self.rope, rope.call())?;
            }
        }
        let at = (position * kv_width * 2) as u64;
        cx.call(
            self.cache,
            calls::cache_bf16(keys.0 + at, row(rows.key, kv_width), count * kv_width),
        )?;
        cx.call(
            self.cache,
            calls::cache_bf16(values.0 + at, row(rows.value, kv_width), count * kv_width),
        )
    }

    /// One sequence's `count` rows from row `first`, at `position` on, attending over the
    /// `keys` and `values` cache through its own row, combined into `rows.attended` (times the
    /// sigmoid of the gates on a gated layer); `each` runs after every launch's combine. The
    /// partials are a launch's: its combine reads them before the next launch writes them.
    ///
    /// # Errors
    /// The driver's refusal, or `each`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn attend<L: Launcher + ?Sized>(
        &self,
        cx: &L,
        shape: GqaShape,
        rows: &GqaRows,
        (keys, values): (Deviceptr, Deviceptr),
        position: usize,
        (first, count): (usize, usize),
        mut each: impl FnMut() -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        let width = shape.query_width();
        let row = |base: Deviceptr, at: usize| base.0 + ((first + at) * width * 4) as u64;
        let mut piece = |at: usize, count: usize| {
            let attention = calls::FullAttention {
                query: row(rows.query, at),
                keys: keys.0,
                values: values.0,
                partials: rows.partials.0,
                position: position + at,
                rows: count,
                heads: shape.heads,
                kv_heads: shape.kv_heads,
                head_dim: shape.head_dim,
            };
            cx.call(self.attention, attention.call())?;
            let gate = if shape.gated { row(rows.gate, at) } else { 0 };
            cx.call(
                self.combine,
                attention.combine(gate, row(rows.attended, at)),
            )?;
            each()
        };
        match self.chunk {
            None => piece(0, count),
            Some(chunk) => {
                let mut at = 0usize;
                while at < count {
                    let rows = chunk.min(count - at);
                    piece(at, rows)?;
                    at += rows;
                }
                Ok(())
            }
        }
    }
}

/// A grouped-query attention layer's four projections in the arm's own format and its q/k
/// RMSNorm gains (each where the layer has them): the query, key and value projections of the
/// normed rows, [GqaAttention]'s heads over every row, then a segment at a time the rotary, the
/// cache append and the attention over the segment's `[keys, values]` regions, and the output
/// projection of the attended heads.
///
/// The dense arm's grouped-query layers and DeepSeek's grouped-query blocks on its dense format
/// route run this one body: each host brings its projection dispatch ([DeviceCx::project]) and
/// its segments' cache regions ([DeviceCx::segments]).
#[derive(Debug, Clone, Copy)]
pub struct GqaLayer<'a, L: ?Sized> {
    /// The query projection (the interleaved query-and-gate one on a gated layer).
    pub q: &'a L,
    /// The key projection.
    pub k: &'a L,
    /// The value projection.
    pub v: &'a L,
    /// The output projection.
    pub o: &'a L,
    /// The query heads' and the key heads' RMSNorm gains.
    pub norms: (Option<Deviceptr>, Option<Deviceptr>),
}

impl<L: ?Sized> GqaLayer<'_, L> {
    /// The layer over the pass's rows on `cx`: `input`'s normed rows into `output`'s projected
    /// ones, through `rows`' scratch, the norms at `eps` (zero-centred when `zero_centred`).
    ///
    /// # Errors
    /// A segment whose regions are not one key and one value cache, a projection the arm does not
    /// run, or the driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn run<C: DeviceCx<Linear = L>>(
        &self,
        cx: &C,
        (attention, norm): (&GqaAttention, &Norm),
        shape: GqaShape,
        rows: &GqaRows,
        (input, output): (Deviceptr, Deviceptr),
        (eps, zero_centred): (f32, bool),
    ) -> Result<(), BackendError> {
        let count = cx.rows();
        let projected = if shape.gated {
            rows.interleaved
        } else {
            rows.query
        };
        cx.project(self.q, input, projected, count)?;
        cx.project(self.k, input, rows.key, count)?;
        cx.project(self.v, input, rows.value, count)?;
        attention.heads(
            cx,
            norm,
            shape,
            rows,
            self.norms,
            (eps, zero_centred),
            count,
        )?;
        for segment in cx.segments() {
            let &[keys, values] = segment.regions else {
                return Err(BackendError::new(
                    ROLE,
                    format!(
                        "a grouped-query segment holds {} cache regions, not a key and a value one",
                        segment.regions.len()
                    ),
                ));
            };
            let span = (segment.first, segment.rows);
            attention.append(cx, shape, rows, (keys, values), segment.position, span)?;
            // The partials are one segment's: the combine reads them before the next segment's
            // attention writes them, both on the one stream.
            attention.attend(
                cx,
                shape,
                rows,
                (keys, values),
                segment.position,
                span,
                || Ok(()),
            )?;
        }
        cx.project(self.o, rows.attended, output, count)
    }
}

/// The f32 rows a Gated DeltaNet layer works in, a row a token, at the rank's share of its heads.
#[derive(Clone, Copy, Debug)]
pub struct GdnRows {
    /// The fused q/k/v projection, `[rows][conv_width]`.
    pub raw: Deviceptr,
    /// The output gate's projection, `[rows][value_width]`.
    pub z: Deviceptr,
    /// The beta logits, `[rows][heads]`.
    pub b: Deviceptr,
    /// The decay logits, `[rows][heads]`.
    pub a: Deviceptr,
    /// The convolved rows, `[rows][conv_width]`.
    pub mixed: Deviceptr,
    /// The normalised q and k ([crate::gdn::GdnGeometry::qk_floats]).
    pub qk: Deviceptr,
    /// Each row's (decay, beta) per value head.
    pub decay: Deviceptr,
    /// The raw head outputs and, in place, their gated norm, `[rows][value_width]`.
    pub attended: Deviceptr,
}

/// A Gated DeltaNet layer's projections in the arm's own format and its other weights: the
/// fused q/k/v, output gate, beta and decay projections of the normed rows; then a segment at a
/// time the conv over its rows from its history and the history advanced past them; the prep
/// over every row; a segment at a time the scan from its state, in place; the gated norm over
/// every row; and `out_proj` of the normed heads.
///
/// Every row count takes these kernels, so a row's bits do not depend on what it shares a pass
/// with. A segment's regions are its conv history and its delta-rule state; a verify pass's
/// segment hands a snapshot slab for each ([SegmentRows::snapshots]), and the conv and the scan
/// leave the state after each of its rows there.
///
/// The dense arm's Gated DeltaNet layers and DeepSeek's recurrent blocks on its dense format
/// route run this one body: each host brings its projection dispatch ([DeviceCx::project]) and
/// its segments' state regions ([DeviceCx::segments]).
#[derive(Debug, Clone, Copy)]
pub struct GdnLayer<'a, L: ?Sized> {
    /// The fused q/k/v projection.
    pub qkv: &'a L,
    /// The output gate's projection.
    pub z: &'a L,
    /// The beta logits' projection.
    pub b: &'a L,
    /// The decay logits' projection.
    pub a: &'a L,
    /// `out_proj`.
    pub out: &'a L,
    /// The conv weights, BF16 `[conv_width][taps]`.
    pub conv: Deviceptr,
    /// The decay vectors, `A_log` and `dt_bias`.
    pub decay: (Deviceptr, Deviceptr),
    /// The gated norm's gains, BF16 a value head's width.
    pub norm: Deviceptr,
}

impl<L: ?Sized> GdnLayer<'_, L> {
    /// The layer over the pass's rows on `cx`: `input`'s normed rows into `output`'s projected
    /// ones, through `rows`' scratch, the scan on `scan` at its launch, the gated norm at `eps`
    /// with a sigmoid gate when `sigmoid` (else SiLU).
    ///
    /// # Errors
    /// A segment whose regions are not one conv history and one state, or whose snapshots are
    /// neither none nor one for each; a projection the arm does not run; or the driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn run<C: DeviceCx<Linear = L>>(
        &self,
        cx: &C,
        (chunk, scan): (&crate::gdn::GdnChunk, (Bound, Launch)),
        geometry: &crate::gdn::GdnGeometry,
        rows: &GdnRows,
        (input, output): (Deviceptr, Deviceptr),
        (eps, sigmoid): (f32, bool),
    ) -> Result<(), BackendError> {
        let count = cx.rows();
        let width = geometry.conv_width();
        let conv = (width, geometry.taps);
        cx.project(self.qkv, input, rows.raw, count)?;
        cx.project(self.z, input, rows.z, count)?;
        cx.project(self.b, input, rows.b, count)?;
        cx.project(self.a, input, rows.a, count)?;
        let state =
            |segment: &SegmentRows<'_>| recurrent_regions(segment, geometry, "a Gated DeltaNet");
        for segment in cx.segments() {
            let (history, _, (windows, _)) = state(&segment)?;
            let at = segment.first * width * 4;
            let rows_in = crate::gdn::ConvRows {
                raw: rows.raw.offset(at),
                history,
                weight: self.conv,
                out: rows.mixed.offset(at),
            };
            chunk.conv.run(cx, &rows_in, conv, segment.rows, windows)?;
        }
        chunk.prep(
            cx,
            geometry,
            (rows.mixed, rows.a, rows.b),
            self.decay,
            (rows.qk, rows.decay),
            count,
        )?;
        let qk_row = geometry.key_heads * crate::gdn::gdn_qk_row(geometry.key_dim) * 4;
        for segment in cx.segments() {
            let (_, state, (_, states)) = state(&segment)?;
            let first = segment.first;
            chunk.scan(
                cx,
                scan,
                geometry,
                (
                    rows.qk.offset(first * qk_row),
                    rows.decay.offset(first * geometry.heads * 8),
                    rows.mixed.offset(first * width * 4),
                ),
                (state, state),
                rows.attended.offset(first * geometry.value_width() * 4),
                segment.rows,
                states,
            )?;
        }
        chunk.gate.run(
            cx,
            (rows.attended, rows.z, self.norm, rows.attended),
            (geometry.heads, geometry.value_dim),
            (eps, sigmoid),
            count,
        )?;
        cx.project(self.out, rows.attended, output, count)
    }
}

/// A recurrent segment's two regions, its conv history and its state, and the snapshot each
/// keeps on a verify pass: (slab, rows, region stride in floats), or (0, 0, 0) for none. `what`
/// names the layer kind in a refusal.
fn recurrent_regions(
    segment: &SegmentRows<'_>,
    geometry: &crate::gdn::GdnGeometry,
    what: &str,
) -> Result<(Deviceptr, Deviceptr, ((u64, usize, u64), (u64, usize, u64))), BackendError> {
    let &[history, state] = segment.regions else {
        return Err(BackendError::new(
            ROLE,
            format!(
                "{what} segment holds {} state regions, not a conv history and a state",
                segment.regions.len()
            ),
        ));
    };
    let keep = |slab: Deviceptr, bytes: usize| (slab.0, segment.rows, (bytes / 4) as u64);
    let snapshots = match *segment.snapshots {
        [] => ((0, 0, 0), (0, 0, 0)),
        [windows, states] => (
            keep(windows, geometry.conv_bytes()),
            keep(states, geometry.state_bytes()),
        ),
        ref other => {
            return Err(BackendError::new(
                ROLE,
                format!(
                    "{what} segment hands {} snapshot slabs, not none or one for each of its two regions",
                    other.len()
                ),
            ));
        }
    };
    Ok((history, state, snapshots))
}

/// The f32 rows a Kimi Delta Attention layer works in, a row a token, at the rank's share of its
/// heads (`width` = the share's heads times the head width).
#[derive(Clone, Copy, Debug)]
pub struct KdaRows {
    /// q, k and v side by side, `[rows][3 width]`.
    pub raw: Deviceptr,
    /// The forget gate's low-rank half, `[rows][f_a's rows]`.
    pub f_a: Deviceptr,
    /// The output gate's low-rank half, `[rows][g_a's rows]`.
    pub g_a: Deviceptr,
    /// The beta logits, `[rows][heads]`.
    pub b: Deviceptr,
    /// The convolved rows, `[rows][3 width]`.
    pub mixed: Deviceptr,
    /// The forget logits, `[rows][width]`.
    pub forget: Deviceptr,
    /// The output gate's logits, `[rows][width]`.
    pub gate: Deviceptr,
    /// Each channel's decay factor ([crate::gdn::kda_decay]), `[rows][width]`.
    pub decay: Deviceptr,
    /// The normalised q and k ([crate::gdn::GdnGeometry::qk_floats]).
    pub qk: Deviceptr,
    /// Each row's (1, beta) per head.
    pub decay_beta: Deviceptr,
    /// The raw head outputs and, in place, their gated norm, `[rows][width]`.
    pub attended: Deviceptr,
}

/// A Kimi Delta Attention layer (GLM-5.3-Flash's linear-attention blocks) in the arm's own
/// projection format and its other weights: q, k and v of the normed rows side by side, and the
/// low-rank forget and output gates' first halves and the beta logits; a segment at a time the
/// conv over its rows from its history and the history advanced past them; the gates' second
/// halves; each channel's decay; the prep over every row; a segment at a time the scan from its
/// state, in place, each key channel decayed by its own factor; the gated norm with a sigmoid
/// gate over every row; and `o_proj` of the normed heads, this rank's partial when the heads are
/// split.
///
/// As for [GdnLayer], every row count takes these kernels, so a row's bits do not depend on what
/// it shares a pass with, and a verify pass's segment hands a snapshot slab for each region.
/// The arithmetic is the one GLM's own host ran on its bulk chunks (`device.rs` `kda` and
/// `scan`), which matches its token recurrence to f32 rounding.
#[derive(Debug, Clone, Copy)]
pub struct KdaLayer<'a, L: ?Sized> {
    /// q, k and v, each the share's heads, written side by side into a row
    /// ([DeviceCx::project_strided]).
    pub qkv: [&'a L; 3],
    /// The forget gate's low-rank halves, `f_a` whole and `f_b` the share's channels.
    pub forget: (&'a L, &'a L),
    /// The output gate's low-rank halves, `g_a` whole and `g_b` the share's channels.
    pub gate: (&'a L, &'a L),
    /// The beta logits' projection.
    pub b: &'a L,
    /// `o_proj`.
    pub out: &'a L,
    /// The q, k and v convs, one BF16 `[3 width][taps]` buffer.
    pub conv: Deviceptr,
    /// The decay vectors, f32: `A_log` a head's and `dt_bias` a channel's.
    pub decay: (Deviceptr, Deviceptr),
    /// The gated norm's gains, BF16 a head's width.
    pub norm: Deviceptr,
}

impl<L: ?Sized> KdaLayer<'_, L> {
    /// The layer over the pass's rows on `cx`: `input`'s normed rows into `output`'s projected
    /// ones (the rank's partial when the heads are split; the host sums it), through `rows`'
    /// scratch, the scan on `scan` at its launch, the decay's `lower_bound` and the gated norm at
    /// `eps`.
    ///
    /// # Errors
    /// A segment whose regions are not one conv history and one state, or whose snapshots are
    /// neither none nor one for each; a projection the arm does not run; or the driver's refusal.
    #[allow(clippy::too_many_arguments)]
    pub fn run<C: DeviceCx<Linear = L>>(
        &self,
        cx: &C,
        (chunk, scan): (&crate::gdn::KdaChunk, (Bound, Launch)),
        geometry: &crate::gdn::GdnGeometry,
        rows: &KdaRows,
        (input, output): (Deviceptr, Deviceptr),
        (eps, lower_bound): (f32, f32),
    ) -> Result<(), BackendError> {
        let count = cx.rows();
        let width = geometry.conv_width();
        let part = geometry.key_width();
        let conv = (width, geometry.taps);
        for (index, linear) in self.qkv.into_iter().enumerate() {
            cx.project_strided(
                linear,
                input,
                (rows.raw.offset(index * part * 4), width),
                count,
            )?;
        }
        cx.project(self.forget.0, input, rows.f_a, count)?;
        cx.project(self.gate.0, input, rows.g_a, count)?;
        cx.project(self.b, input, rows.b, count)?;
        let state = |segment: &SegmentRows<'_>| recurrent_regions(segment, geometry, "a KDA");
        for segment in cx.segments() {
            let (history, _, (windows, _)) = state(&segment)?;
            let at = segment.first * width * 4;
            let rows_in = crate::gdn::ConvRows {
                raw: rows.raw.offset(at),
                history,
                weight: self.conv,
                out: rows.mixed.offset(at),
            };
            chunk.conv.run(cx, &rows_in, conv, segment.rows, windows)?;
        }
        cx.project(self.forget.1, rows.f_a, rows.forget, count)?;
        cx.project(self.gate.1, rows.g_a, rows.gate, count)?;
        chunk.decay(
            cx,
            geometry,
            (rows.forget, self.decay.0, self.decay.1),
            rows.decay,
            lower_bound,
            count,
        )?;
        chunk.prep(
            cx,
            geometry,
            (rows.mixed, rows.b),
            (rows.qk, rows.decay_beta),
            count,
        )?;
        let qk_row = geometry.key_heads * crate::gdn::gdn_qk_row(geometry.key_dim) * 4;
        for segment in cx.segments() {
            let (_, state, (_, states)) = state(&segment)?;
            let first = segment.first;
            chunk.scan(
                cx,
                scan,
                geometry,
                (
                    rows.qk.offset(first * qk_row),
                    rows.decay_beta.offset(first * geometry.heads * 8),
                    rows.decay.offset(first * part * 4),
                    rows.mixed.offset(first * width * 4),
                ),
                state,
                rows.attended.offset(first * geometry.value_width() * 4),
                segment.rows,
                states,
            )?;
        }
        chunk.gate.run(
            cx,
            (rows.attended, rows.gate, self.norm, rows.attended),
            (geometry.heads, geometry.value_dim),
            (eps, true),
            count,
        )?;
        cx.project(self.out, rows.attended, output, count)
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
