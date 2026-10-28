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

//! The device-only data path of the layer port's `moe_routed` stage.
//!
//! # What this module is for
//!
//! The layer port's twentieth stage (`LayerStages::moe_routed`, see
//! `crates/sparkflash-cli/src/layer_device.rs`) refused with "the five device MoE launches exist;
//! the host combine of their outputs is the step". The five launches
//! (`routed_gate_up_sm121` twice, `swiglu_mxfp8_quant_weighted_sm121`,
//! `b12x_mxfp4_moe_stored_sm121` and the router) all run on the device, but they are only half of
//! the stage's data path:
//!
//! * the routed launches index their weight operand by a *staged slot*
//!   (`weight_row = expert * width + column`), so the selected experts' resident-arena slices have to
//!   be gathered into slot order first, and the host arm does that with a host-issued
//!   `cuMemcpyDtoD` per slot;
//! * after the stored down projection the per-slot f32 accumulators still sit on device, and the host
//!   arm reads them back to run `mxfp4_ref::moe_combine_routed_f32` plus its bf16 narrow.
//!
//! This module supplies the two missing device kernels (kernels/moe_routed_device_sm121.cu) and the
//! run-scope state that owns them, their buffers and the device-resident expert pointer table. It
//! performs **no per-layer PCIe traffic**: the only copies it issues are at open (the pointer table
//! and the constant slot-id vector), and a stage call touches device memory only.
//!
//! # What one stage call does, in order
//!
//! All launches are on one stream, so the order below is the execution order.
//!
//! 1. `moe_expert_stage_sm121` gathers w1 (gate) slice per routed row into one staging buffer;
//! 2. `routed_gate_up_sm121` computes the gate projection of those rows;
//! 3. the same staging launch gathers w3 (up) into the same buffer, then the second
//!    `routed_gate_up_sm121` launch computes the up projection (stream order is what makes reusing
//!    one staging buffer safe: the buffer is written only after its previous reader has run);
//! 4. `swiglu_mxfp8_quant_weighted_sm121` applies the SwiGLU, multiplies the routing weight in and
//!    quantises, exactly as the host arm does;
//! 5. the staging launch gathers w2 (down) and zeroes the f32 slot accumulator;
//! 6. `b12x_mxfp4_moe_stored_sm121` adds the down projection into the slot accumulator;
//! 7. `moe_routed_combine_sm121` rounds each slot to bf16, adds the six rows in ascending global
//!    expert id order in f32, and writes both the f32 total (`workspace.moe_delta`) and its bf16
//!    narrow (`workspace.moe_delta_bf16`, the pre-reduce payload).
//!
//! `all_reduce_moe` then consumes `moe_delta_bf16`; this module does not reduce and does not touch
//! the shared expert.
//!
//! # Arithmetic this module is accountable for
//!
//! None of it is new. The expert math stays in the four pinned launches. The two new kernels
//! transcribe references that already exist in this crate and in the CLI:
//!
//! * staging is a byte copy of the arena slices the host arm copies with `cuMemcpyDtoD`;
//! * the combine is `mxfp4_ref::moe_combine_routed_f32` (per-slot bf16 round, then the f32 adds in
//!   ascending expert-id order) followed by the CLI's `bf16_le_bytes` narrow. Both roundings are
//!   reproduced with the references' own integer bit arithmetic in
//!   [combine_routed_f32]/[round_slot_bf16]/[narrow_payload_bf16], which are the CPU twins of the
//!   kernel and are tested bit-for-bit against `mxfp4_ref` in this module's tests.
//!
//! The combine's slot order comes from the *global* expert ids (`workspace.moe_ids`). The host arm
//! passes the reference its slot indices instead, but it numbers slots in ascending expert-id order
//! (`moe_slots` sorts and dedups before mapping), so the two orders are the same permutation.
//!
//! # Open-time state, and why the pointer table is a device object
//!
//! The routed weights are resident in the arena before any token runs: `DeviceBind::moe` resolves
//! `(layer, expert, matrix)` to a device (weight, scales) address pair. The ids, however, are only
//! known on device during the step, so the host cannot issue the gather. [ExpertPointerTable]
//! therefore records every expert's two addresses in device memory at open, indexed
//! `[layer][matrix][expert]`, and the staging kernel reads the address it needs from device memory.
//! With 40 layers, three matrices and 384 experts the table is 40 * 3 * 384 * 2 * 8 = 737,280 bytes.
//!
//! A note on the package brief's "up to 256 x (w_ptr, s_ptr)" table: 256 cannot be the expert count
//! of the released layer (it has 384 resident experts), so this module reads it as the routed-row
//! **capacity** [MAX_ROUTED_ROWS]: the number of (token, slot) rows one stage call may stage. The
//! released decode is one token, i.e. six rows; a verify step of up to 64 tokens (384 rows) fits.
//!
//! # Out of scope
//!
//! The 640/576 column padding of the routed gate/up/SwiGLU operands is the S4 concern (see
//! docs/build/s1-p3-moe-wiring.md). This module reproduces the incumbent ABI exactly, padding
//! included, so an S4 padding change moves the landed kernels' geometry and not this module.

use crate::c_decl::{self, Consts, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

// ---------------------------------------------------------------------------------------------
// Released geometry
// ---------------------------------------------------------------------------------------------

/// The released hidden size `d` (specification 2.1).
pub const HIDDEN: usize = 5120;
/// The released routed-expert intermediate width.
pub const INTER: usize = 2304;
/// The released tensor-parallel world size.
pub const WORLD: usize = 4;
/// One rank's slice of [INTER]: the gate/up output rows and the stored down rows' K width.
pub const LOCAL_INTER: usize = INTER / WORLD;
/// The routed gate/up/SwiGLU column count: [LOCAL_INTER] real columns padded to a 32-value multiple.
pub const WIDTH_PADDED: usize = 640;
/// The released routed expert count.
pub const EXPERTS: usize = 384;
/// The released routed top-k.
pub const TOP_K: usize = 6;
/// The MX block width.
pub const QUANT_BLOCK: usize = 32;
/// Matrices per routed expert: gate (w1), up (w3), down (w2).
pub const ROUTED_MATRICES: usize = 3;
/// The routed rows (token-slot pairs) one stage call may hold, i.e. the staging capacity.
///
/// The package brief's "up to 256" table capacity, raised to a 64-row verify step's routed rows. It
/// is a capacity, not a model constant, and it is deliberately the routed-row ceiling rather than an
/// expert count (that it equals the released layer's 384 resident experts is a coincidence).
pub const MAX_ROUTED_ROWS: usize = 384;

/// The routed rows the multi-row (dedup) kernels group in one plan (opt_moe_sm121.cu's
/// OPT_MOE_PLAN_MAX): 64 verify rows of top-6 fit.
const DEDUP_MAX_ROWS: usize = 384;

/// One routed expert matrix, in the order the table and the bind use.
///
/// The variant order is the table's matrix axis and the `[Gate, Up, Down]` order
/// `DeviceLayer::attach_routed` resolves, which is also the checkpoint leaf order w1, w3, w2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoutedMatrix {
    /// w1, the SwiGLU gate projection.
    Gate,
    /// w3, the SwiGLU up projection.
    Up,
    /// w2, the down projection.
    Down,
}

impl RoutedMatrix {
    /// Every matrix, in table order.
    pub const ALL: [Self; ROUTED_MATRICES] = [Self::Gate, Self::Up, Self::Down];

    /// The matrix's index on the table's matrix axis.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::Down => 2,
        }
    }

    /// The checkpoint leaf the matrix comes from, for refusal messages and the wiring doc.
    #[must_use]
    pub fn leaf(self) -> &'static str {
        match self {
            Self::Gate => "w1",
            Self::Up => "w3",
            Self::Down => "w2",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/// The released routed-MoE geometry a stage call runs at.
///
/// The four landed launches pin this shape in their kernels (`SPARK_M4_MOE_WIDTH == 576`,
/// `SPARK_M4_MOE_WIDTH_PADDED == 640`, `SPARK_M4_HIDDEN == 5120`, `SPARK_M4_MOE_TOP_K == 6`), so
/// [MoeRoutedGeometry::new] refuses anything else rather than planning a shape the kernels cannot
/// serve. The device layer supplies the values from the model config; a config that is not the
/// released four-rank shape is an open-time refusal, not a per-step surprise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoutedGeometry {
    /// The model's hidden size.
    pub dim: usize,
    /// The routed-expert intermediate width (global, not per rank).
    pub inter: usize,
    /// The tensor-parallel world size.
    pub world: usize,
    /// The routed expert count.
    pub experts: usize,
    /// The routed top-k.
    pub top_k: usize,
}

impl MoeRoutedGeometry {
    /// The released four-rank geometry.
    #[must_use]
    pub fn released() -> Self {
        Self {
            dim: HIDDEN,
            inter: INTER,
            world: WORLD,
            experts: EXPERTS,
            top_k: TOP_K,
        }
    }

    /// The geometry a run's config describes, refused unless it is the released one.
    ///
    /// # Errors
    /// Refuses any shape the pinned routed launches were not compiled for, naming every field.
    pub fn new(
        dim: usize,
        inter: usize,
        world: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<Self, String> {
        let released = Self::released();
        if (dim, inter, world, experts, top_k)
            != (
                released.dim,
                released.inter,
                released.world,
                released.experts,
                released.top_k,
            )
        {
            return Err(format!(
                "moe_routed device: the pinned routed launches run the released four-rank shape \
                 (dim {HIDDEN}, inter {INTER}, world {WORLD}, {EXPERTS} experts, top-{TOP_K}); got \
                 dim {dim}, inter {inter}, world {world}, {experts} experts, top-{top_k}"
            ));
        }
        Ok(Self {
            dim,
            inter,
            world,
            experts,
            top_k,
        })
    }

    /// One rank's gate/up output rows: `inter / world`.
    #[must_use]
    pub fn local_inter(&self) -> usize {
        self.inter / self.world
    }

    /// The packed-FP4 bytes of one gate/up weight row: `dim / 2`.
    #[must_use]
    pub fn packed_row_bytes(&self) -> usize {
        self.dim / 2
    }

    /// The ue8m0 bytes of one gate/up weight row's scale grid: `dim / 32`.
    #[must_use]
    pub fn scale_row_bytes(&self) -> usize {
        self.dim / QUANT_BLOCK
    }

    /// The bytes of one staged gate or up weight slice: `local_inter * dim / 2`.
    ///
    /// At the released geometry this is 1,474,560 bytes, and it is also the size of one staged down
    /// weight slice ([Self::down_weight_bytes]): the two projections are transposes of one another,
    /// which is why one staging buffer can serve all three matrices.
    #[must_use]
    pub fn stage_weight_bytes(&self) -> usize {
        self.local_inter() * self.packed_row_bytes()
    }

    /// The bytes of one staged gate or up scale slice: `local_inter * dim / 32` (92,160 released).
    #[must_use]
    pub fn stage_scale_bytes(&self) -> usize {
        self.local_inter() * self.scale_row_bytes()
    }

    /// The bytes of one staged down weight slice: `dim * inter / 2 / world` (1,474,560 released).
    #[must_use]
    pub fn down_weight_bytes(&self) -> usize {
        self.dim * (self.inter / 2 / self.world)
    }

    /// The bytes of one staged down scale slice: `dim * inter / 32 / world` (92,160 released).
    #[must_use]
    pub fn down_scale_bytes(&self) -> usize {
        self.dim * (self.inter / self.world / QUANT_BLOCK)
    }

    /// The per-e4m3 column row of a stored down projection output row: `inter / 2 / world`.
    #[must_use]
    pub fn down_code_row_bytes(&self) -> usize {
        self.inter / 2 / self.world
    }

    /// The ue8m0 bytes of a stored down projection output row's scale grid: `inter / 32 / world`.
    #[must_use]
    pub fn down_scale_row_bytes(&self) -> usize {
        self.inter / self.world / QUANT_BLOCK
    }
}

// ---------------------------------------------------------------------------------------------
// Buffer plan
// ---------------------------------------------------------------------------------------------

/// The run-scope device buffer sizes of one [MoeRoutedDevice].
///
/// Every size is derived from the geometry and the routed-row capacity; the struct exists so the
/// open-time allocation and the wiring doc can state the same numbers, and so the released byte
/// counts are unit-tested instead of living in a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoutedSizes {
    /// The expert pointer table, in 64-bit words: `layers * 3 * experts * 2`.
    pub table_words: usize,
    /// One staged gate or up weight slice, `rows * stage_weight_bytes`.
    pub stage_weight: usize,
    /// One staged gate or up scale slice, `rows * stage_scale_bytes`.
    pub stage_scales: usize,
    /// The gate projection's bf16 output, `rows * WIDTH_PADDED * 2`.
    pub gate_out: usize,
    /// The up projection's bf16 output, `rows * WIDTH_PADDED * 2`.
    pub up_out: usize,
    /// The weighted SwiGLU's E4M3 codes, `rows * WIDTH_PADDED`.
    pub codes: usize,
    /// The weighted SwiGLU's ue8m0 scales, `rows * WIDTH_PADDED / 32`.
    pub scales: usize,
    /// The f32 slot accumulators, `rows * dim * 4`.
    pub slot_acc: usize,
    /// The constant staged-slot ids, `rows * 4`.
    pub slot_ids: usize,
    /// The total of every buffer above, in bytes.
    pub total: usize,
}

impl MoeRoutedSizes {
    /// The buffer sizes for a geometry and a routed-row capacity.
    ///
    /// # Errors
    /// Refuses a byte count that overflows `usize`.
    pub fn new(
        geometry: &MoeRoutedGeometry,
        layers: usize,
        max_rows: usize,
    ) -> Result<Self, String> {
        let checked = |value: Option<usize>, what: &str| -> Result<usize, String> {
            value.ok_or_else(|| format!("moe_routed device: the {what} byte count overflows usize"))
        };
        let table_words = checked(
            layers
                .checked_mul(ROUTED_MATRICES)
                .and_then(|v| v.checked_mul(geometry.experts))
                .and_then(|v| v.checked_mul(2)),
            "pointer table",
        )?;
        let table_bytes = checked(table_words.checked_mul(8), "pointer table")?;
        let stage_weight = checked(
            max_rows.checked_mul(geometry.stage_weight_bytes()),
            "staged weight",
        )?;
        let stage_scales = checked(
            max_rows.checked_mul(geometry.stage_scale_bytes()),
            "staged scale",
        )?;
        let padded_row_bytes = checked(WIDTH_PADDED.checked_mul(2), "gate/up output row")?;
        let gate_out = checked(
            max_rows.checked_mul(padded_row_bytes),
            "gate projection output",
        )?;
        let up_out = gate_out;
        let codes = checked(max_rows.checked_mul(WIDTH_PADDED), "SwiGLU code row")?;
        let scales = checked(
            max_rows.checked_mul(WIDTH_PADDED / QUANT_BLOCK),
            "SwiGLU scale row",
        )?;
        let slot_acc = checked(
            max_rows
                .checked_mul(geometry.dim)
                .and_then(|v| v.checked_mul(4)),
            "slot accumulator",
        )?;
        let slot_ids = checked(max_rows.checked_mul(4), "slot id vector")?;
        let total = [
            table_bytes,
            stage_weight,
            stage_scales,
            gate_out,
            up_out,
            codes,
            scales,
            slot_acc,
            slot_ids,
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
        .ok_or_else(|| {
            "moe_routed device: the total buffer byte count overflows usize".to_string()
        })?;
        Ok(Self {
            table_words,
            stage_weight,
            stage_scales,
            gate_out,
            up_out,
            codes,
            scales,
            slot_acc,
            slot_ids,
            total,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The plan: everything a stage call validates against, with no device state
// ---------------------------------------------------------------------------------------------

/// The validated plan of one run's routed-MoE device path: geometry, layer span, capacity and the
/// SwiGLU limit. It owns no device memory, so every refusal a stage call can make from the plan is
/// unit-testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeRoutedPlan {
    geometry: MoeRoutedGeometry,
    layers: usize,
    max_rows: usize,
    limit: f32,
}

impl MoeRoutedPlan {
    /// Validates a plan.
    ///
    /// # Errors
    /// Refuses zero layers (nothing to index), a routed-row capacity outside
    /// `[top_k, MAX_ROUTED_ROWS]`, and a SwiGLU limit that is not finite.
    pub fn new(
        geometry: MoeRoutedGeometry,
        layers: usize,
        max_rows: usize,
        limit: f32,
    ) -> Result<Self, String> {
        if layers == 0 {
            return Err("moe_routed device: the plan covers no layers".to_string());
        }
        if max_rows < geometry.top_k {
            return Err(format!(
                "moe_routed device: the staging capacity {max_rows} rows cannot hold one token's \
                 {} routed rows",
                geometry.top_k
            ));
        }
        if max_rows > MAX_ROUTED_ROWS {
            return Err(format!(
                "moe_routed device: the staging capacity {max_rows} rows exceeds the {MAX_ROUTED_ROWS}-row ceiling"
            ));
        }
        if !limit.is_finite() {
            return Err(format!(
                "moe_routed device: the SwiGLU limit {limit} is not a finite value"
            ));
        }
        Ok(Self {
            geometry,
            layers,
            max_rows,
            limit,
        })
    }

    /// The geometry.
    #[must_use]
    pub fn geometry(&self) -> &MoeRoutedGeometry {
        &self.geometry
    }

    /// The number of layers the pointer table covers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The routed-row staging capacity.
    #[must_use]
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// The SwiGLU limit the weighted epilogue clamps with.
    #[must_use]
    pub fn limit(&self) -> f32 {
        self.limit
    }

    /// The buffer sizes this plan needs.
    ///
    /// # Errors
    /// As [MoeRoutedSizes::new].
    pub fn sizes(&self) -> Result<MoeRoutedSizes, String> {
        MoeRoutedSizes::new(&self.geometry, self.layers, self.max_rows)
    }

    /// The routed rows a token count produces: `tokens * top_k`.
    ///
    /// # Errors
    /// Refuses zero tokens and a row count above the plan's capacity.
    pub fn routed_rows(&self, tokens: usize) -> Result<usize, String> {
        if tokens == 0 {
            return Err("moe_routed device: a stage call runs at least one token".to_string());
        }
        let rows = tokens
            .checked_mul(self.geometry.top_k)
            .ok_or_else(|| "moe_routed device: the routed-row count overflows usize".to_string())?;
        if rows > self.max_rows {
            return Err(format!(
                "moe_routed device: {tokens} tokens are {rows} routed rows, above the staged \
                 capacity of {} rows",
                self.max_rows
            ));
        }
        Ok(rows)
    }

    /// Checks one stage call against the plan and the caller's workspace spans, returning the routed
    /// row count on success.
    ///
    /// Every refusal here is a property of the plan and the spans, so it is identical on all four
    /// ranks (the layer port's divergence rule): a short span or a layer outside the table is a
    /// wiring defect, not a race.
    ///
    /// # Errors
    /// Refuses a layer outside the table, a token count the staging cannot hold, and any workspace
    /// span shorter than the stage will write or read.
    pub fn check_operands(
        &self,
        layer: usize,
        tokens: usize,
        operands: &MoeRoutedOperands,
    ) -> Result<usize, String> {
        if layer >= self.layers {
            return Err(format!(
                "moe_routed device: layer {layer} is outside the pointer table's {} layers",
                self.layers
            ));
        }
        let rows = self.routed_rows(tokens)?;
        let dim = self.geometry.dim;
        operands.act_codes.require(
            "workspace.act_codes",
            tokens * dim,
            "one E4M3 byte per activation value",
        )?;
        operands.act_scales.require(
            "workspace.act_scales",
            tokens * (dim / QUANT_BLOCK),
            "one ue8m0 scale per 32-value block",
        )?;
        operands.ids.require(
            "workspace.moe_ids",
            rows * 4,
            "one u32 global expert id per routed row",
        )?;
        operands.weights.require(
            "workspace.moe_weights",
            rows * 4,
            "one f32 routing weight per routed row",
        )?;
        operands.delta_f32.require(
            "workspace.moe_delta",
            tokens * dim * 4,
            "one f32 value per (token, hidden index)",
        )?;
        operands.delta_bf16.require(
            "workspace.moe_delta_bf16",
            tokens * dim * 2,
            "one bf16 value per (token, hidden index)",
        )?;
        Ok(rows)
    }
}

// ---------------------------------------------------------------------------------------------
// Device spans and operands
// ---------------------------------------------------------------------------------------------

/// A device buffer the layer port's workspace owns, with the byte length the stage was given.
///
/// The stage does not allocate from the workspace: it validates the spans the layer's buffer table
/// hands it and writes only inside them, so a wiring mistake is a refusal instead of a stride over
/// another stage's buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceSpan {
    /// The buffer's device address.
    pub device: Deviceptr,
    /// The span's byte length as the layer workspace carved it.
    pub bytes: usize,
}

impl DeviceSpan {
    /// A span over `bytes` bytes at `device`.
    #[must_use]
    pub const fn new(device: Deviceptr, bytes: usize) -> Self {
        Self { device, bytes }
    }

    /// Refuses a span shorter than `needed` bytes.
    fn require(&self, what: &str, needed: usize, stride: &str) -> Result<(), String> {
        if self.bytes < needed {
            return Err(format!(
                "moe_routed device: {what} is {} bytes, the stage needs {needed} ({stride})",
                self.bytes
            ));
        }
        Ok(())
    }
}

/// The workspace spans one stage call reads and writes.
///
/// The producer/consumer relations are the layer port's stage contracts: `act_codes`/`act_scales`
/// come from `ffn_seams`, `ids`/`weights` from `moe_router`, `delta_f32`/`delta_bf16` are this
/// stage's outputs, and `delta_bf16` feeds `all_reduce_moe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRoutedOperands {
    /// `workspace.act_codes`: `[tokens][dim]` E4M3 activation codes (the FFN quantise input).
    pub act_codes: DeviceSpan,
    /// `workspace.act_scales`: `[tokens][dim / 32]` ue8m0 activation scales.
    pub act_scales: DeviceSpan,
    /// `workspace.moe_ids`: `[tokens * top_k]` u32 global expert ids from `moe_router`.
    pub ids: DeviceSpan,
    /// `workspace.moe_weights`: `[tokens * top_k]` f32 routing weights from `moe_router`.
    pub weights: DeviceSpan,
    /// `workspace.moe_delta`: `[tokens][dim]` f32 routed sum, this stage's f32 output.
    pub delta_f32: DeviceSpan,
    /// `workspace.moe_delta_bf16`: `[tokens][dim]` bf16 payload, this stage's pre-reduce output.
    pub delta_bf16: DeviceSpan,
}

// ---------------------------------------------------------------------------------------------
// The expert pointer table
// ---------------------------------------------------------------------------------------------

/// The device-resident pointer table of every routed expert slice, indexed
/// `[layer][matrix][expert]`.
///
/// Each entry is two 64-bit device addresses: the packed-FP4 weight slice and its ue8m0 scale grid,
/// exactly the pair `DeviceBind::moe` resolves for `(layer, expert, matrix)`. The table is built once
/// at open from the resident bind and uploaded once; a stage call only lets the staging kernel read
/// it. Nothing about it changes per step, so no per-layer upload exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPointerTable {
    layers: usize,
    experts: usize,
    words: Vec<u64>,
}

impl ExpertPointerTable {
    /// Builds the table from a per-slice address closure.
    ///
    /// `slice(layer, matrix, expert)` returns the (weight, scales) device addresses of that expert's
    /// matrix slice. The builder walks the table in its storage order and never reorders, so the
    /// upload is the table's own bytes.
    ///
    /// # Errors
    /// Refuses zero layers or zero experts, and refuses a zero address: the arena resolves every
    /// bound slice, so a zero address is a wiring defect that would otherwise become a kernel
    /// reading the null page.
    pub fn build<F>(layers: usize, experts: usize, mut slice: F) -> Result<Self, String>
    where
        F: FnMut(usize, RoutedMatrix, usize) -> (u64, u64),
    {
        if layers == 0 {
            return Err("moe_routed device: the pointer table covers no layers".to_string());
        }
        if experts == 0 {
            return Err("moe_routed device: the pointer table covers no experts".to_string());
        }
        let mut words = Vec::with_capacity(layers * ROUTED_MATRICES * experts * 2);
        for layer in 0..layers {
            for matrix in RoutedMatrix::ALL {
                for expert in 0..experts {
                    let (weight, scales) = slice(layer, matrix, expert);
                    if weight == 0 {
                        return Err(format!(
                            "moe_routed device: layer {layer} expert {expert} {} weight address is null",
                            matrix.leaf()
                        ));
                    }
                    if scales == 0 {
                        return Err(format!(
                            "moe_routed device: layer {layer} expert {expert} {} scale address is null",
                            matrix.leaf()
                        ));
                    }
                    words.push(weight);
                    words.push(scales);
                }
            }
        }
        Ok(Self {
            layers,
            experts,
            words,
        })
    }

    /// The number of layers the table covers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The number of experts the table covers.
    #[must_use]
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// The table's words, in storage order.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// The table's bytes, little-endian, ready for one `copy_htod`.
    #[must_use]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    /// The word index of one (layer, matrix) slice's base, i.e. the address the staging kernel
    /// receives for that launch.
    ///
    /// # Panics
    /// Panics on a layer or matrix outside the table: the caller checks both against the plan before
    /// a launch, and an index outside the table would silently address another slice.
    #[must_use]
    pub fn slice_word_offset(&self, layer: usize, matrix: RoutedMatrix) -> usize {
        assert!(layer < self.layers, "layer {layer} is outside the table");
        (layer * ROUTED_MATRICES + matrix.index()) * self.experts * 2
    }

    /// The word index of one (layer, matrix, expert) entry.
    ///
    /// # Panics
    /// Panics on an out-of-table index, as [Self::slice_word_offset].
    #[must_use]
    pub fn entry_word_offset(&self, layer: usize, matrix: RoutedMatrix, expert: usize) -> usize {
        assert!(
            expert < self.experts,
            "expert {expert} is outside the table"
        );
        self.slice_word_offset(layer, matrix) + expert * 2
    }

    /// The (weight, scales) addresses recorded for one (layer, matrix, expert) entry.
    ///
    /// # Panics
    /// Panics on an out-of-table index, as [Self::slice_word_offset].
    #[must_use]
    pub fn entry(&self, layer: usize, matrix: RoutedMatrix, expert: usize) -> (u64, u64) {
        let index = self.entry_word_offset(layer, matrix, expert);
        (self.words[index], self.words[index + 1])
    }
}

// ---------------------------------------------------------------------------------------------
// ABI pins
// ---------------------------------------------------------------------------------------------

/// The staging kernel's symbol in kernels/moe_routed_device_sm121.cu.
pub const STAGE_KERNEL: &str = "moe_expert_stage_sm121";
/// The combine kernel's symbol in the same translation unit.
pub const COMBINE_KERNEL: &str = "moe_routed_combine_sm121";
/// The weighted SwiGLU symbol this module launches (a landed kernel, pinned here only for the
/// argument order the stage fills).
pub const WEIGHTED_KERNEL: &str = "swiglu_mxfp8_quant_weighted_sm121";
/// The translation unit the two new declarations are parsed from.
pub const SOURCE: &str = include_str!("../../../kernels/moe_routed_device_sm121.cu");
/// The landed weighted-SwiGLU source, for the declaration pin of the launch this module fills.
pub const WEIGHTED_SOURCE: &str =
    include_str!("../../../kernels/swiglu_mxfp8_quant_weighted_sm121.cu");

/// The staging launch's argument words.
pub const STAGE_ARG_WORDS: usize = 10;
/// The combine launch's argument words.
pub const COMBINE_ARG_WORDS: usize = 6;
/// The weighted-SwiGLU launch's argument words.
pub const WEIGHTED_ARG_WORDS: usize = 11;

/// The staging declaration's parameters, in order: (name, base type, pointer, pointee const).
pub const STAGE_PARAMS: [(&str, &str, bool, bool); STAGE_ARG_WORDS] = [
    ("table", "uint64_t", true, true),
    ("expert_ids", "uint32_t", true, true),
    ("dst_weight", "uint8_t", true, false),
    ("dst_scales", "uint8_t", true, false),
    ("slot_acc", "float", true, false),
    ("rows", "uint32_t", false, false),
    ("experts", "uint32_t", false, false),
    ("weight_bytes", "uint32_t", false, false),
    ("scale_bytes", "uint32_t", false, false),
    ("acc_row_floats", "uint32_t", false, false),
];

/// The combine declaration's parameters, in order.
pub const COMBINE_PARAMS: [(&str, &str, bool, bool); COMBINE_ARG_WORDS] = [
    ("slot_acc", "float", true, true),
    ("expert_ids", "uint32_t", true, true),
    ("delta_f32", "float", true, false),
    ("delta_bf16", "uint16_t", true, false),
    ("tokens", "uint32_t", false, false),
    ("hidden", "uint32_t", false, false),
];

/// The weighted-SwiGLU declaration's parameters, in order.
pub const WEIGHTED_PARAMS: [(&str, &str, bool, bool); WEIGHTED_ARG_WORDS] = [
    ("gate", "__nv_bfloat16", true, true),
    ("up", "__nv_bfloat16", true, true),
    ("weights", "float", true, true),
    ("out_fp8", "uint8_t", true, false),
    ("out_scales", "uint8_t", true, false),
    ("rows", "uint32_t", false, false),
    ("cols", "uint32_t", false, false),
    ("limit", "float", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// The staging launch's block: one CTA per routed row, 256 threads.
pub const STAGE_BLOCK: (u32, u32, u32) = (STAGED_COPY_THREADS, 1, 1);
/// The combine launch's block: 256 hidden columns per CTA.
pub const COMBINE_BLOCK: (u32, u32, u32) = (COMBINE_TILE_COLS, 1, 1);
/// The weighted-SwiGLU launch's block: 256 threads, i.e. eight 32-value MX blocks per CTA.
pub const WEIGHTED_BLOCK: (u32, u32, u32) = (256, 1, 1);
/// The weight threads one staged copy row uses; pinned to the kernel's `blockDim.x` assumption.
pub const STAGED_COPY_THREADS: u32 = 256;
/// The combine launch's column tile: one block covers [COMBINE_TILE_COLS] hidden columns.
pub const COMBINE_TILE_COLS: u32 = 256;

/// Parses the staging declaration out of the kernel source.
#[must_use]
pub fn stage_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, STAGE_KERNEL)
}

/// Parses the combine declaration out of the kernel source.
#[must_use]
pub fn combine_declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, COMBINE_KERNEL)
}

/// Parses the weighted-SwiGLU declaration out of the landed kernel source.
#[must_use]
pub fn weighted_declaration() -> KernelDecl {
    c_decl::kernel_declaration(WEIGHTED_SOURCE, WEIGHTED_KERNEL)
}

/// The new kernel's constants with the shared M4 header's names resolvable.
#[must_use]
pub fn constants() -> Consts {
    let mut table = Consts::parse(SOURCE);
    table.extend(crate::m4_abi::COMMON_HEADER);
    table
}

/// Whether a parsed declaration has the pinned shape, order and qualifiers.
#[must_use]
pub fn declaration_matches(
    decl: &KernelDecl,
    name: &str,
    params: &[(&str, &str, bool, bool)],
) -> bool {
    decl.name == name
        && decl.extern_c
        && decl.global
        && decl.params.len() == params.len()
        && decl.params.iter().zip(params).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

// ---------------------------------------------------------------------------------------------
// Launch parameters and launches
// ---------------------------------------------------------------------------------------------

/// One staging launch's arguments, in the declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertStageLaunchParams {
    /// The `(layer, matrix)` slice base of the device pointer table.
    pub table: Deviceptr,
    /// `[rows]` u32 global expert ids.
    pub expert_ids: Deviceptr,
    /// `[rows][weight_bytes]` staged packed weights.
    pub dst_weight: Deviceptr,
    /// `[rows][scale_bytes]` staged ue8m0 grids.
    pub dst_scales: Deviceptr,
    /// `[rows][acc_row_floats]` f32 slot accumulators, or a null operand when `acc_row_floats` is 0.
    pub slot_acc: Deviceptr,
    /// Routed rows to stage.
    pub rows: u32,
    /// The number of experts the table holds, i.e. the bound of a valid id.
    pub experts: u32,
    /// Bytes of one staged weight slice.
    pub weight_bytes: u32,
    /// Bytes of one staged scale slice.
    pub scale_bytes: u32,
    /// Floats per slot-accumulator row to zero; 0 skips the zeroing.
    pub acc_row_floats: u32,
}

impl ExpertStageLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; STAGE_ARG_WORDS] {
        [
            self.table.0,
            self.expert_ids.0,
            self.dst_weight.0,
            self.dst_scales.0,
            if self.acc_row_floats == 0 {
                0
            } else {
                self.slot_acc.0
            },
            u64::from(self.rows),
            u64::from(self.experts),
            u64::from(self.weight_bytes),
            u64::from(self.scale_bytes),
            u64::from(self.acc_row_floats),
        ]
    }

    /// The grid: one CTA per routed row.
    ///
    /// # Errors
    /// Refuses a zero row count instead of launching a zero-sized grid.
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        if self.rows == 0 {
            return Err(
                "moe_routed device: the staging launch needs at least one routed row".to_string(),
            );
        }
        Ok((self.rows, 1, 1))
    }

    /// The launch's own refusals: geometry, slice sizes and the alignment the copy fast path needs.
    ///
    /// # Errors
    /// Refuses zero experts, zero or non-multiple-of-four slice bytes, and a zero row count.
    pub fn validate(&self) -> Result<(), String> {
        if self.rows == 0 {
            return Err(
                "moe_routed device: the staging launch needs at least one routed row".to_string(),
            );
        }
        if self.experts == 0 {
            return Err(
                "moe_routed device: the staging launch needs a non-empty expert table".to_string(),
            );
        }
        for (bytes, what) in [(self.weight_bytes, "weight"), (self.scale_bytes, "scale")] {
            if bytes == 0 {
                return Err(format!(
                    "moe_routed device: a staged {what} slice cannot be empty"
                ));
            }
            if !bytes.is_multiple_of(4) {
                return Err(format!(
                    "moe_routed device: a staged {what} slice of {bytes} bytes is not a whole \
                     number of 32-bit words, so the typed copy path cannot address it"
                ));
            }
        }
        if self.acc_row_floats != 0 && self.slot_acc.0 == 0 {
            return Err(
                "moe_routed device: the staging launch zeroes a slot accumulator but was given a \
                 null address"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Launches the staging kernel once.
///
/// # Errors
/// The parameter refusals, or the driver's launch error.
pub fn launch_stage(
    cuda: &Cuda,
    kernel: Kernel,
    params: &ExpertStageLaunchParams,
) -> Result<(), String> {
    params.validate()?;
    cuda.launch(kernel, params.grid()?, STAGE_BLOCK, &params.words())
}

/// One combine launch's arguments, in the declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CombineLaunchParams {
    /// `[tokens * top_k][hidden]` f32 slot accumulators.
    pub slot_acc: Deviceptr,
    /// `[tokens * top_k]` u32 global expert ids.
    pub expert_ids: Deviceptr,
    /// `[tokens][hidden]` f32 routed-sum output.
    pub delta_f32: Deviceptr,
    /// `[tokens][hidden]` bf16 payload output.
    pub delta_bf16: Deviceptr,
    /// Tokens in the launch.
    pub tokens: u32,
    /// Hidden columns.
    pub hidden: u32,
}

impl CombineLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; COMBINE_ARG_WORDS] {
        [
            self.slot_acc.0,
            self.expert_ids.0,
            self.delta_f32.0,
            self.delta_bf16.0,
            u64::from(self.tokens),
            u64::from(self.hidden),
        ]
    }

    /// The grid: `(tokens, ceil(hidden / COMBINE_TILE_COLS), 1)`.
    ///
    /// # Errors
    /// Refuses a zero token or hidden count instead of launching a zero-sized grid.
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        if self.tokens == 0 {
            return Err(
                "moe_routed device: the combine launch needs at least one token".to_string(),
            );
        }
        if self.hidden == 0 {
            return Err(
                "moe_routed device: the combine launch needs a non-empty hidden row".to_string(),
            );
        }
        Ok((self.tokens, self.hidden.div_ceil(COMBINE_TILE_COLS), 1))
    }
}

/// Launches the combine kernel once.
///
/// # Errors
/// The parameter refusals, or the driver's launch error.
pub fn launch_combine(
    cuda: &Cuda,
    kernel: Kernel,
    params: &CombineLaunchParams,
) -> Result<(), String> {
    cuda.launch(kernel, params.grid()?, COMBINE_BLOCK, &params.words())
}

/// One weighted-SwiGLU launch's arguments, in the landed declaration's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedSwiGLULaunchParams {
    /// `[rows][cols]` bf16 gate output.
    pub gate: Deviceptr,
    /// `[rows][cols]` bf16 up output.
    pub up: Deviceptr,
    /// `[rows]` f32 routing weights.
    pub weights: Deviceptr,
    /// `[rows][cols]` E4M3 output codes.
    pub out_fp8: Deviceptr,
    /// `[rows][cols / 32]` ue8m0 output scales.
    pub out_scales: Deviceptr,
    /// Routed rows.
    pub rows: u32,
    /// Columns, a multiple of 32 (640 on the routed path).
    pub cols: u32,
    /// The SwiGLU clamp limit.
    pub limit: f32,
    /// The trace ring's device address, or 0 for the shipped trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event records.
    pub layer_id: u32,
    /// The step sequence the trace event records.
    pub step_seq: u64,
}

impl WeightedSwiGLULaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; WEIGHTED_ARG_WORDS] {
        [
            self.gate.0,
            self.up.0,
            self.weights.0,
            self.out_fp8.0,
            self.out_scales.0,
            u64::from(self.rows),
            u64::from(self.cols),
            // A 4-byte float argument is passed as a word whose low half is the f32 bit pattern, the
            // same convention the incumbent weighted-SwiGLU launch site uses.
            u64::from(self.limit.to_bits()),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }

    /// The grid: one warp per 32-value MX block, eight warps per CTA.
    ///
    /// # Errors
    /// Refuses zero rows, a zero or non-multiple-of-32 column count, a non-finite limit, and a warp
    /// count the grid cannot address.
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        if self.rows == 0 {
            return Err(
                "moe_routed device: the weighted SwiGLU needs at least one routed row".to_string(),
            );
        }
        if self.cols == 0 || !self.cols.is_multiple_of(32) {
            return Err(format!(
                "moe_routed device: the weighted SwiGLU needs a column count that is a multiple of \
                 32, got {}",
                self.cols
            ));
        }
        if !self.limit.is_finite() {
            return Err(format!(
                "moe_routed device: the SwiGLU limit {} is not a finite value",
                self.limit
            ));
        }
        let warps = u64::from(self.rows) * u64::from(self.cols / 32);
        let blocks = warps.div_ceil(8);
        let blocks = u32::try_from(blocks).map_err(|_| {
            format!(
                "moe_routed device: {warps} MX blocks are more than the launch grid can address"
            )
        })?;
        Ok((blocks, 1, 1))
    }
}

/// Launches the landed weighted-SwiGLU kernel once.
///
/// # Errors
/// The parameter refusals, or the driver's launch error.
pub fn launch_weighted(
    cuda: &Cuda,
    kernel: Kernel,
    params: &WeightedSwiGLULaunchParams,
) -> Result<(), String> {
    cuda.launch(kernel, params.grid()?, WEIGHTED_BLOCK, &params.words())
}

// ---------------------------------------------------------------------------------------------
// The run-scope device path
// ---------------------------------------------------------------------------------------------

/// The five kernel handles a [MoeRoutedDevice] runs: the two this module adds and the three landed
/// launches it drives.
///
/// The handles are passed in already loaded, because loading is the integration step: the two new
/// symbols need their by-name loaders in `cuda_driver` (see the wiring doc), and the three landed
/// kernels load through their existing pinned launch modules. Handing over handles keeps this module
/// free of cubin bytes and of the Lead's pin checks.
#[derive(Debug, Clone, Copy)]
pub struct MoeRoutedKernels {
    /// The staging kernel from kernels/moe_routed_device_sm121.cu.
    pub stage: (Module, Kernel),
    /// The combine kernel from kernels/moe_routed_device_sm121.cu.
    pub combine: (Module, Kernel),
    /// `routed_gate_up_sm121`, loaded through `routed_gate_up_launch::load`.
    pub gate_up: (Module, Kernel),
    /// `swiglu_mxfp8_quant_weighted_sm121`, loaded through its `KernelId` loader.
    pub weighted: (Module, Kernel),
    /// `b12x_mxfp4_moe_stored_sm121`, loaded through `mxfp4_moe_stored_launch::load`.
    pub stored: (Module, Kernel),
}

/// The run-scope device state of the routed-MoE stage: the buffers, the pointer table and the
/// kernels.
///
/// One instance is created at open (the layer port's prewarm, next to the shared expert's own
/// run-scope state) and lives for the run. A stage call allocates nothing, uploads nothing and reads
/// nothing back to the host.
#[derive(Debug)]
pub struct MoeRoutedDevice {
    plan: MoeRoutedPlan,
    table: Deviceptr,
    stage_weight: Deviceptr,
    stage_scales: Deviceptr,
    gate_out: Deviceptr,
    up_out: Deviceptr,
    codes: Deviceptr,
    scales: Deviceptr,
    slot_acc: Deviceptr,
    slot_ids: Deviceptr,
    kernels: MoeRoutedKernels,
    opt: Option<OptMoeKernels>,
    /// An f32 [tokens][5120] row the opt combine adds after the routed slots (the rank's shared
    /// expert partial), or 0 for none.
    addend: std::cell::Cell<u64>,
    device_bytes: usize,
}

/// The fused two-launch routed path (kernels/opt_moe_sm121.cu): gate+up+SwiGLU+quantise reading the
/// expert slices in place, then down+combine. Bit-identical to the staged eight-launch chain.
#[derive(Debug, Clone, Copy)]
pub struct OptMoeKernels {
    /// `opt_moe_gate_up_sm121`.
    pub gate_up: (Module, Kernel),
    /// `opt_moe_down_combine_sm121`.
    pub down: (Module, Kernel),
    /// `opt_moe_gate_up_dedup_sm121`, `opt_moe_down_dedup_sm121`, `opt_moe_combine_sm121`: the
    /// multi-token path that reads each distinct expert once (bit-identical to the two above).
    pub dedup: [(Module, Kernel); 3],
}

impl OptMoeKernels {
    /// Loads both symbols from the opt cubin (unpinned).
    ///
    /// # Errors
    /// The driver's load error.
    pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<Self, String> {
        Ok(Self {
            gate_up: cuda.load_named(cubin, "opt_moe_gate_up_sm121")?,
            down: cuda.load_named(cubin, "opt_moe_down_combine_sm121")?,
            dedup: [
                cuda.load_named(cubin, "opt_moe_gate_up_dedup_sm121")?,
                cuda.load_named(cubin, "opt_moe_down_dedup_sm121")?,
                cuda.load_named(cubin, "opt_moe_combine_sm121")?,
            ],
        })
    }
}

/// The chunked-prefill routed path's kernels (kernels/opt_mma_sm121.cu) and scratch, owned by the
/// caller and sized for `max_tokens`.
#[derive(Debug, Clone, Copy)]
pub struct MoeBulk {
    /// `opt_mma_moe_plan`.
    pub plan: Kernel,
    /// `opt_mma_moe_gate_up`.
    pub gate_up: Kernel,
    /// `opt_mma_moe_down`.
    pub down: Kernel,
    /// `opt_mma_moe_combine`.
    pub combine: Kernel,
    /// `[tokens * top_k]` u32: the routed rows grouped by expert.
    pub sorted: Deviceptr,
    /// `[max_items][4]` u32 work items.
    pub items: Deviceptr,
    /// The work-item count (one u32).
    pub items_count: Deviceptr,
    /// `[tokens * top_k][576]` E4M3 SwiGLU codes.
    pub codes: Deviceptr,
    /// `[tokens * top_k][18]` ue8m0 scales of `codes`.
    pub scales: Deviceptr,
    /// `[tokens * top_k][5120]` bf16 per-row down outputs.
    pub slot: Deviceptr,
    /// The tokens the buffers are sized for.
    pub max_tokens: usize,
}

/// The most chunks one [MoeRoutedDevice::run_bulk_rows] pass groups (opt_mma_moe_plan's
/// OMM_MAX_SEGS).
pub const MAX_BULK_SEGMENTS: usize = 4;

/// A bulk pass's token rows (device addresses): the FP8 activation codes `[tokens][5120]` and their
/// scales `[tokens][160]`, the routed expert ids and routing weights `[tokens][top_k]`.
#[derive(Clone, Copy, Debug)]
pub struct BulkRows {
    /// E4M3 activation codes.
    pub act_codes: u64,
    /// ue8m0 scales of `act_codes`, one per 32.
    pub act_scales: u64,
    /// u32 routed expert ids.
    pub ids: u64,
    /// f32 routing weights.
    pub weights: u64,
}

impl MoeBulk {
    /// The work-item bound for `tokens`: one per expert plus one per 64 routed rows.
    #[must_use]
    pub fn max_items(tokens: usize) -> usize {
        EXPERTS + tokens * TOP_K / 64 + 1
    }

    /// The device bytes of each scratch buffer for `tokens`, in field order (sorted, items,
    /// items_count, codes, scales, slot).
    #[must_use]
    pub fn buffer_bytes(tokens: usize) -> [usize; 6] {
        let rows = tokens * TOP_K;
        [
            rows * 4,
            Self::max_items(tokens) * 16,
            16,
            rows * LOCAL_INTER,
            rows * (LOCAL_INTER / QUANT_BLOCK),
            rows * HIDDEN * 2,
        ]
    }
}

impl MoeRoutedDevice {
    /// Allocates and fills the run-scope device state.
    ///
    /// The only copies this ever issues are here: the pointer table and the constant staged-slot id
    /// vector, both uploaded once.
    ///
    /// # Errors
    /// Refuses a plan the released kernels cannot serve, a pointer table whose layer or expert count
    /// disagrees with the plan, a non-finite SwiGLU limit, and any driver allocation/copy failure.
    pub fn open(
        cuda: &Cuda,
        plan: MoeRoutedPlan,
        table: &ExpertPointerTable,
        kernels: MoeRoutedKernels,
    ) -> Result<Self, String> {
        if table.layers() != plan.layers() {
            return Err(format!(
                "moe_routed device: the pointer table covers {} layers, the plan has {}",
                table.layers(),
                plan.layers()
            ));
        }
        if table.experts() != plan.geometry().experts {
            return Err(format!(
                "moe_routed device: the pointer table holds {} experts, the geometry has {}",
                table.experts(),
                plan.geometry().experts
            ));
        }
        let sizes = plan.sizes()?;
        let table_bytes = table.words().len().checked_mul(8).ok_or_else(|| {
            "moe_routed device: the pointer table byte count overflows usize".to_string()
        })?;
        if table_bytes != sizes.table_words * 8 {
            return Err(format!(
                "moe_routed device: the built pointer table is {table_bytes} bytes, the plan \
                 allocates {}",
                sizes.table_words * 8
            ));
        }
        let slot_ids: Vec<u8> = (0..plan.max_rows())
            .flat_map(|row| {
                u32::try_from(row % plan.geometry().top_k)
                    .expect("a slot below top_k fits u32")
                    .to_le_bytes()
            })
            .collect();

        let mut pointers: Vec<Deviceptr> = Vec::new();
        let mut allocated = 0usize;
        let build = (|| -> Result<Self, String> {
            let fresh = |cuda: &Cuda,
                         pointers: &mut Vec<Deviceptr>,
                         allocated: &mut usize,
                         bytes: usize,
                         what: &str|
             -> Result<Deviceptr, String> {
                let pointer = cuda
                    .alloc(bytes, allocated)
                    .map_err(|error| format!("moe_routed device: {what} allocation: {error}"))?;
                pointers.push(pointer);
                Ok(pointer)
            };
            let table_ptr = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                table_bytes,
                "pointer table",
            )?;
            cuda.copy_htod(&table_ptr, &table.to_le_bytes())
                .map_err(|error| format!("moe_routed device: pointer table upload: {error}"))?;
            let stage_weight = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.stage_weight,
                "staged weight",
            )?;
            let stage_scales = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.stage_scales,
                "staged scales",
            )?;
            let gate_out = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.gate_out,
                "gate out",
            )?;
            let up_out = fresh(cuda, &mut pointers, &mut allocated, sizes.up_out, "up out")?;
            let codes = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.codes,
                "SwiGLU codes",
            )?;
            let scales = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.scales,
                "SwiGLU scales",
            )?;
            let slot_acc = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.slot_acc,
                "slot accumulator",
            )?;
            let slot_ids_ptr = fresh(
                cuda,
                &mut pointers,
                &mut allocated,
                sizes.slot_ids,
                "slot ids",
            )?;
            cuda.copy_htod(&slot_ids_ptr, &slot_ids)
                .map_err(|error| format!("moe_routed device: slot id upload: {error}"))?;
            Ok(Self {
                plan,
                table: table_ptr,
                stage_weight,
                stage_scales,
                gate_out,
                up_out,
                codes,
                scales,
                slot_acc,
                slot_ids: slot_ids_ptr,
                kernels,
                opt: None,
                addend: std::cell::Cell::new(0),
                device_bytes: allocated,
            })
        })();
        match build {
            Ok(device) => Ok(device),
            Err(error) => {
                let borrowed: Vec<&Deviceptr> = pointers.iter().collect();
                cuda.free_buffers(&borrowed);
                Err(error)
            }
        }
    }

    /// Selects the fused two-launch path for every later [Self::run].
    pub fn set_opt(&mut self, opt: Option<OptMoeKernels>) {
        self.opt = opt;
    }

    /// Sets the f32 row the opt combine adds into the delta (0 = none). Only the opt path reads it.
    pub fn set_addend(&self, addend: u64) {
        self.addend.set(addend);
    }

    /// True when the fused opt kernels serve this device (the only path that reads the addend).
    #[must_use]
    pub fn has_opt(&self) -> bool {
        self.opt.is_some()
    }

    /// The fused path: two launches, no staging copy.
    fn run_opt(
        &self,
        cuda: &Cuda,
        opt: &OptMoeKernels,
        layer: usize,
        tokens: usize,
        rows: usize,
        operands: &MoeRoutedOperands,
    ) -> Result<(), String> {
        let experts = self.plan.geometry().experts as u64;
        let gate = self
            .table
            .offset(self.plan.table_word_offset(layer, RoutedMatrix::Gate) * 8);
        let up = self
            .table
            .offset(self.plan.table_word_offset(layer, RoutedMatrix::Up) * 8);
        let down = self
            .table
            .offset(self.plan.table_word_offset(layer, RoutedMatrix::Down) * 8);
        let rows_u32 =
            u32::try_from(rows).map_err(|_| "moe_routed opt: rows overflow".to_owned())?;
        let tokens_u32 =
            u32::try_from(tokens).map_err(|_| "moe_routed opt: tokens overflow".to_owned())?;
        if tokens > 1
            && rows <= DEDUP_MAX_ROWS
            && std::env::var("SPARKFLASH_MOE_DEDUP").as_deref() != Ok("0")
        {
            return self.run_dedup(cuda, opt, (gate, up, down), tokens_u32, rows_u32, operands);
        }
        let gate_up_args = [
            gate.0,
            up.0,
            operands.ids.device.0,
            operands.weights.device.0,
            operands.act_codes.device.0,
            operands.act_scales.device.0,
            self.codes.0,
            self.scales.0,
            u64::from(rows_u32),
            experts,
            u64::from(self.plan.limit().to_bits()),
        ];
        cuda.launch(
            opt.gate_up.1,
            ((INTER / WORLD / QUANT_BLOCK) as u32, rows_u32, 1),
            (128, 1, 1),
            &gate_up_args,
        )
        .map_err(|error| format!("moe_routed opt: gate/up: {error}"))?;
        let down_args = [
            down.0,
            operands.ids.device.0,
            self.codes.0,
            self.scales.0,
            operands.delta_f32.device.0,
            operands.delta_bf16.device.0,
            u64::from(tokens_u32),
            experts,
            self.addend.get(),
        ];
        cuda.launch(
            opt.down.1,
            ((HIDDEN / 32) as u32, tokens_u32, 1),
            (256, 1, 1),
            &down_args,
        )
        .map_err(|error| format!("moe_routed opt: down/combine: {error}"))
    }

    /// The multi-token fused path: gate/up per distinct expert (which also writes the row grouping
    /// into the unused gate-out buffer), down per distinct expert into the slot accumulators, then
    /// the per-token combine.
    fn run_dedup(
        &self,
        cuda: &Cuda,
        opt: &OptMoeKernels,
        (gate, up, down): (Deviceptr, Deviceptr, Deviceptr),
        tokens: u32,
        rows: u32,
        operands: &MoeRoutedOperands,
    ) -> Result<(), String> {
        let experts = self.plan.geometry().experts as u64;
        let plan = self.gate_out.0;
        // A block row per distinct expert; there are at most as many as rows, or as experts.
        let groups = rows.min(self.plan.geometry().experts as u32);
        cuda.launch(
            opt.dedup[0].1,
            ((INTER / WORLD / QUANT_BLOCK) as u32, groups, 1),
            (128, 1, 1),
            &[
                gate.0,
                up.0,
                operands.ids.device.0,
                operands.weights.device.0,
                operands.act_codes.device.0,
                operands.act_scales.device.0,
                self.codes.0,
                self.scales.0,
                u64::from(rows),
                experts,
                u64::from(self.plan.limit().to_bits()),
                plan,
            ],
        )
        .map_err(|error| format!("moe_routed dedup: gate/up: {error}"))?;
        cuda.launch(
            opt.dedup[1].1,
            // Two 32-column tiles per block (OPT_MOE_DD_TILES), two of a tile's 576 chunks a thread.
            ((HIDDEN / 64) as u32, groups, 1),
            (288, 1, 1),
            &[down.0, plan, self.codes.0, self.scales.0, self.slot_acc.0],
        )
        .map_err(|error| format!("moe_routed dedup: down: {error}"))?;
        cuda.launch(
            opt.dedup[2].1,
            ((HIDDEN / 256) as u32, tokens, 1),
            (256, 1, 1),
            &[
                operands.ids.device.0,
                self.slot_acc.0,
                operands.delta_f32.device.0,
                operands.delta_bf16.device.0,
                u64::from(tokens),
                self.addend.get(),
            ],
        )
        .map_err(|error| format!("moe_routed dedup: combine: {error}"))
    }

    /// The chunked-prefill path ([MoeBulk]): groups the routed rows by expert, runs each group's
    /// gate/up and down on the FP4 tensor cores, then the per-token combine (adding the addend)
    /// into `delta_bf16` alone (the bulk prefill reads only the bf16 sums; `delta_f32` is left as
    /// it was). Not bit-identical to [Self::run]: the slot partials round through bf16.
    ///
    /// # Errors
    /// A token count above the bulk buffers' capacity, and any driver launch error.
    pub fn run_bulk(
        &self,
        cuda: &Cuda,
        bulk: &MoeBulk,
        layer: usize,
        tokens: usize,
        operands: &MoeRoutedOperands,
    ) -> Result<(), String> {
        self.run_bulk_rows(
            cuda,
            bulk,
            layer,
            tokens,
            &[tokens],
            &BulkRows {
                act_codes: operands.act_codes.device.0,
                act_scales: operands.act_scales.device.0,
                ids: operands.ids.device.0,
                weights: operands.weights.device.0,
            },
        )?;
        self.combine_bulk(
            cuda,
            bulk,
            0,
            tokens,
            operands.ids.device.0,
            0,
            operands.delta_bf16.device.0,
        )
    }

    /// [Self::run_bulk]'s grouping, gate/up and down over several chunks' tokens at once, so each
    /// expert's weights are read about once for all of them: chunk s is the first `counts[s]` of
    /// the `stride` tokens from token s * stride of `rows` (the rows between are gaps no work item
    /// reads), and its routed rows' down outputs land in the slot rows from s * stride * top-k.
    /// Each row's values are the ones its chunk computes alone; [Self::combine_bulk] then sums each
    /// chunk's tokens.
    ///
    /// # Errors
    /// More than four chunks, tokens past the bulk buffers' capacity, and any driver launch error.
    pub fn run_bulk_rows(
        &self,
        cuda: &Cuda,
        bulk: &MoeBulk,
        layer: usize,
        stride: usize,
        counts: &[usize],
        rows: &BulkRows,
    ) -> Result<(), String> {
        let span = match counts {
            [] => 0,
            [.., last] => (counts.len() - 1) * stride + last,
        };
        if counts.len() > MAX_BULK_SEGMENTS
            || span == 0
            || span > bulk.max_tokens
            || counts.iter().any(|&count| count > stride)
        {
            return Err(format!(
                "moe_routed bulk: chunks of {counts:?} tokens {stride} apart outside the capacity of {}",
                bulk.max_tokens
            ));
        }
        let experts = self.plan.geometry().experts;
        let table = |matrix| {
            self.table
                .offset(self.plan.table_word_offset(layer, matrix) * 8)
                .0
        };
        let routed = (span * TOP_K) as u32;
        let max_items = MoeBulk::max_items(span) as u32;
        // One chunk needs no segments; several are segments of `stride` tokens.
        let segments = if counts.len() > 1 { stride as u64 } else { 0 };
        let count = |s: usize| counts.get(s).map_or(0, |&count| count as u64);
        cuda.launch(
            bulk.plan,
            (1, 1, 1),
            (1024, 1, 1),
            &[
                rows.ids,
                u64::from(routed),
                experts as u64,
                bulk.sorted.0,
                bulk.items.0,
                bulk.items_count.0,
                segments,
                count(0),
                count(1),
                count(2),
                count(3),
            ],
        )
        .map_err(|error| format!("moe_routed bulk: plan: {error}"))?;
        cuda.launch(
            bulk.gate_up,
            ((LOCAL_INTER / 64) as u32, max_items, 1),
            (256, 1, 1),
            &[
                table(RoutedMatrix::Gate),
                table(RoutedMatrix::Up),
                bulk.sorted.0,
                bulk.items.0,
                bulk.items_count.0,
                rows.weights,
                rows.act_codes,
                rows.act_scales,
                bulk.codes.0,
                bulk.scales.0,
                u64::from(self.plan.limit().to_bits()),
            ],
        )
        .map_err(|error| format!("moe_routed bulk: gate/up: {error}"))?;
        cuda.launch(
            bulk.down,
            ((HIDDEN / 128) as u32, max_items, 1),
            (256, 1, 1),
            &[
                table(RoutedMatrix::Down),
                bulk.sorted.0,
                bulk.items.0,
                bulk.items_count.0,
                bulk.codes.0,
                bulk.scales.0,
                bulk.slot.0,
            ],
        )
        .map_err(|error| format!("moe_routed bulk: down: {error}"))
    }

    /// [Self::run_bulk]'s per-token combine for one chunk of [Self::run_bulk_rows]: its `tokens`
    /// tokens from token `first` of the pass (`ids` is the chunk's own routing), adding the addend,
    /// into `delta_bf16` and, unless it is 0, `delta_f32`.
    ///
    /// # Errors
    /// Tokens past the bulk buffers' capacity, and any driver launch error.
    #[allow(clippy::too_many_arguments)]
    pub fn combine_bulk(
        &self,
        cuda: &Cuda,
        bulk: &MoeBulk,
        first: usize,
        tokens: usize,
        ids: u64,
        delta_f32: u64,
        delta_bf16: u64,
    ) -> Result<(), String> {
        if tokens == 0 || first + tokens > bulk.max_tokens {
            return Err(format!(
                "moe_routed bulk: combine of {tokens} tokens from {first} outside the capacity of {}",
                bulk.max_tokens
            ));
        }
        cuda.launch(
            bulk.combine,
            ((HIDDEN / 1024) as u32, tokens as u32, 1),
            (256, 1, 1),
            &[
                ids,
                bulk.slot.0 + (first * TOP_K * HIDDEN * 2) as u64,
                delta_f32,
                delta_bf16,
                tokens as u64,
                self.addend.get(),
            ],
        )
        .map_err(|error| format!("moe_routed bulk: combine: {error}"))
    }

    /// The plan this device was opened with.
    #[must_use]
    pub fn plan(&self) -> &MoeRoutedPlan {
        &self.plan
    }

    /// The device bytes this path owns, for the run's allocation ledger.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    /// The staging buffer addresses, for the doc and for evidence harnesses. `(weight, scales)`.
    #[must_use]
    pub fn staging_buffers(&self) -> (Deviceptr, Deviceptr) {
        (self.stage_weight, self.stage_scales)
    }

    /// Runs one stage call: eight launches (the header's seven numbered steps), device memory
    /// only.
    ///
    /// `layer` selects the pointer-table slice; `position` is the step sequence the landed launches'
    /// trace words carry (it is not read by the shipped trace-disabled cubins).
    ///
    /// # Errors
    /// Every refusal [MoeRoutedPlan::check_operands] makes, a word conversion that does not fit, and
    /// any driver launch error. Each launch error is prefixed with the step it belongs to, so a GPU
    /// fault names its launch.
    pub fn run(
        &self,
        cuda: &Cuda,
        layer: usize,
        position: usize,
        tokens: usize,
        operands: &MoeRoutedOperands,
    ) -> Result<(), String> {
        let rows = self.plan.check_operands(layer, tokens, operands)?;
        if let Some(opt) = self.opt.as_ref() {
            return self.run_opt(cuda, opt, layer, tokens, rows, operands);
        }
        let geometry = *self.plan.geometry();
        let tokens_u32 = u32::try_from(tokens)
            .map_err(|_| format!("moe_routed device: {tokens} tokens do not fit a launch word"))?;
        let rows_u32 = u32::try_from(rows).map_err(|_| {
            format!("moe_routed device: {rows} routed rows do not fit a launch word")
        })?;
        let experts_u32 = u32::try_from(geometry.experts).map_err(|_| {
            format!(
                "moe_routed device: {} experts do not fit a launch word",
                geometry.experts
            )
        })?;
        let weight_bytes = u32::try_from(geometry.stage_weight_bytes()).map_err(|_| {
            format!(
                "moe_routed device: a {} byte staged weight slice does not fit a launch word",
                geometry.stage_weight_bytes()
            )
        })?;
        let scale_bytes = u32::try_from(geometry.stage_scale_bytes()).map_err(|_| {
            format!(
                "moe_routed device: a {} byte staged scale slice does not fit a launch word",
                geometry.stage_scale_bytes()
            )
        })?;
        let hidden = u32::try_from(geometry.dim).map_err(|_| {
            format!(
                "moe_routed device: {} hidden do not fit a launch word",
                geometry.dim
            )
        })?;
        let layer_u32 = u32::try_from(layer)
            .map_err(|_| format!("moe_routed device: layer {layer} does not fit a launch word"))?;
        let step_seq = u64::try_from(position).map_err(|_| {
            format!("moe_routed device: position {position} does not fit a launch word")
        })?;
        // The staged weight and scale slice bytes are equal for gate, up and down at the released
        // geometry; the two projections are transposes of the same slice, so one staging buffer can
        // serve all three. Refuse a geometry where that stops being true instead of staging over a
        // shorter slice than the launch reads.
        for (staged, down, what) in [
            (
                geometry.stage_weight_bytes(),
                geometry.down_weight_bytes(),
                "weight",
            ),
            (
                geometry.stage_scale_bytes(),
                geometry.down_scale_bytes(),
                "scale",
            ),
        ] {
            if staged != down {
                return Err(format!(
                    "moe_routed device: the staged {what} slice ({staged} bytes) is not the down \
                     projection's slice ({down} bytes), so one staging buffer cannot serve all three \
                     matrices"
                ));
            }
        }
        let slot_acc_words = u32::try_from(geometry.dim).map_err(|_| {
            "moe_routed device: the slot accumulator row does not fit a word".to_string()
        })?;

        let stage = |matrix: RoutedMatrix, zero_acc: bool| -> Result<(), String> {
            let params = ExpertStageLaunchParams {
                table: self
                    .table
                    .offset(self.plan.table_word_offset(layer, matrix) * 8),
                expert_ids: operands.ids.device,
                dst_weight: self.stage_weight,
                dst_scales: self.stage_scales,
                slot_acc: self.slot_acc,
                rows: rows_u32,
                experts: experts_u32,
                weight_bytes,
                scale_bytes,
                acc_row_floats: if zero_acc { slot_acc_words } else { 0 },
            };
            launch_stage(cuda, self.kernels.stage.1, &params).map_err(|error| {
                format!(
                    "moe_routed device: staging the {} ({}) slice: {error}",
                    matrix.leaf(),
                    match matrix {
                        RoutedMatrix::Gate => "gate",
                        RoutedMatrix::Up => "up",
                        RoutedMatrix::Down => "down",
                    }
                )
            })
        };
        let gate_up = |out: Deviceptr, what: &str| -> Result<(), String> {
            let params = crate::routed_gate_up_launch::GateUpLaunchParams {
                act_fp8: operands.act_codes.device,
                act_scales: operands.act_scales.device,
                w_fp4: self.stage_weight,
                w_scales: self.stage_scales,
                expert_ids: self.slot_ids,
                out,
                tokens: tokens_u32,
                trace_ring: 0,
                layer_id: layer_u32,
                step_seq,
            };
            crate::routed_gate_up_launch::launch(cuda, self.kernels.gate_up.1, &params)
                .map_err(|error| format!("moe_routed device: the {what} projection: {error}"))
        };

        // 1-2: gate.
        stage(RoutedMatrix::Gate, false)?;
        gate_up(self.gate_out, "gate")?;
        // 3-4: up, reusing the staging buffer after the gate launch has consumed it (stream order).
        stage(RoutedMatrix::Up, false)?;
        gate_up(self.up_out, "up")?;
        // 5: weighted SwiGLU + quantise.
        let weighted = WeightedSwiGLULaunchParams {
            gate: self.gate_out,
            up: self.up_out,
            weights: operands.weights.device,
            out_fp8: self.codes,
            out_scales: self.scales,
            rows: rows_u32,
            cols: u32::try_from(WIDTH_PADDED)
                .expect("the padded routed width is 640, which fits a word"),
            limit: self.plan.limit(),
            trace_ring: 0,
            layer_id: layer_u32,
            step_seq,
        };
        launch_weighted(cuda, self.kernels.weighted.1, &weighted)
            .map_err(|error| format!("moe_routed device: the weighted SwiGLU: {error}"))?;
        // 6-7: down staging (which also zeroes the accumulator), then the stored combine.
        stage(RoutedMatrix::Down, true)?;
        let stored = crate::mxfp4_moe_stored_launch::Mxfp4MoeStoredParams::new(
            self.codes,
            self.scales,
            self.stage_weight,
            self.stage_scales,
            self.slot_ids,
            self.slot_acc,
            tokens_u32,
            0,
            layer_u32,
            step_seq,
        );
        crate::mxfp4_moe_stored_launch::launch(cuda, self.kernels.stored.1, &stored)
            .map_err(|error| format!("moe_routed device: the stored down projection: {error}"))?;
        // 8: the routed combine epilogue.
        let combine = CombineLaunchParams {
            slot_acc: self.slot_acc,
            expert_ids: operands.ids.device,
            delta_f32: operands.delta_f32.device,
            delta_bf16: operands.delta_bf16.device,
            tokens: tokens_u32,
            hidden,
        };
        launch_combine(cuda, self.kernels.combine.1, &combine)
            .map_err(|error| format!("moe_routed device: the routed combine: {error}"))?;
        Ok(())
    }

    /// Frees the buffers and unloads the modules, for the layer port's drop path.
    ///
    /// The driver's module cache owns a module the run-scope cache loaded, so freeing a cached module
    /// handle is a no-op there; the buffers are unpooled allocations and are really freed.
    pub fn release(self, cuda: &Cuda) {
        for (module, _) in [
            self.kernels.stage,
            self.kernels.combine,
            self.kernels.gate_up,
            self.kernels.weighted,
            self.kernels.stored,
        ] {
            cuda.free(module, &[]);
        }
        let pointers = [
            &self.table,
            &self.stage_weight,
            &self.stage_scales,
            &self.gate_out,
            &self.up_out,
            &self.codes,
            &self.scales,
            &self.slot_acc,
            &self.slot_ids,
        ];
        cuda.free_buffers(&pointers);
    }
}

impl MoeRoutedPlan {
    /// The byte offset of one `(layer, matrix)` pointer-table slice inside the uploaded table.
    ///
    /// This is the address arithmetic the staging launch's `table` operand uses; it is a plan method
    /// so the arithmetic has a CPU test independent of a device allocation.
    ///
    /// # Panics
    /// Panics outside the plan's layer span: [Self::check_operands] refuses that first.
    #[must_use]
    pub fn table_word_offset(&self, layer: usize, matrix: RoutedMatrix) -> usize {
        assert!(layer < self.layers, "layer {layer} is outside the plan");
        (layer * ROUTED_MATRICES + matrix.index()) * self.geometry.experts * 2
    }
}

// ---------------------------------------------------------------------------------------------
// CPU twins of the two kernels' arithmetic
// ---------------------------------------------------------------------------------------------

/// The combine's slot order: the reference's selection sort, transcribed.
///
/// Returns, for each ascending position, the slot index whose global expert id is next. Equal ids are
/// resolved to the earlier slot, exactly as `mxfp4_ref::moe_combine_slot_order` resolves them.
///
/// # Panics
/// Panics when `expert_ids` does not hold one token's `top_k` ids.
#[must_use]
pub fn combine_slot_order(expert_ids: &[u32], token: usize) -> [usize; TOP_K] {
    assert!(
        expert_ids.len() >= token * TOP_K + TOP_K,
        "the id list must hold token {token}'s top_k ids"
    );
    let mut order = [0usize; TOP_K];
    let mut used = [false; TOP_K];
    for entry in &mut order {
        let mut best = usize::MAX;
        for slot in 0..TOP_K {
            if used[slot] {
                continue;
            }
            if best == usize::MAX
                || expert_ids[token * TOP_K + slot] < expert_ids[token * TOP_K + best]
            {
                best = slot;
            }
        }
        *entry = best;
        used[best] = true;
    }
    order
}

/// The reference's per-slot bf16 round, as the combine kernel computes it: round to nearest even, and
/// quiet a NaN the way `mxfp4_ref::round_expert_row` does.
#[must_use]
pub fn round_slot_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = if bits & 0x7fff_ffff > 0x7f80_0000 {
        (bits >> 16) | 0x0040
    } else {
        bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
    };
    f32::from_bits(rounded << 16)
}

/// The host arm's bf16 narrow of the fused f32 total, as the combine kernel computes it: the same
/// integer round without the NaN special case, because the payload path spells
/// `(bits + 0x7fff + lsb) & 0xffff_0000`.
#[must_use]
pub fn narrow_payload_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The kernel's combine, as f32: `mxfp4_ref::moe_combine_routed_f32` transcribed.
///
/// # Panics
/// Panics when the accumulator, the id list or the token count disagree.
#[must_use]
pub fn combine_routed_f32(
    slot_acc: &[f32],
    expert_ids: &[u32],
    tokens: usize,
    hidden: usize,
) -> Vec<f32> {
    assert_eq!(
        slot_acc.len(),
        tokens * TOP_K * hidden,
        "the slot accumulator holds one f32 per (routed row, hidden index)"
    );
    let mut delta = vec![0.0f32; tokens * hidden];
    for token in 0..tokens {
        let order = combine_slot_order(expert_ids, token);
        let target = &mut delta[token * hidden..(token + 1) * hidden];
        for slot in order {
            let row = &slot_acc[(token * TOP_K + slot) * hidden..][..hidden];
            for (total, value) in target.iter_mut().zip(row) {
                *total += round_slot_bf16(*value);
            }
        }
    }
    delta
}

/// The kernel's bf16 payload row for a token batch: [combine_routed_f32] followed by the narrow.
///
/// # Panics
/// As [combine_routed_f32].
#[must_use]
pub fn combine_payload_bf16(
    slot_acc: &[f32],
    expert_ids: &[u32],
    tokens: usize,
    hidden: usize,
) -> Vec<u16> {
    combine_routed_f32(slot_acc, expert_ids, tokens, hidden)
        .into_iter()
        .map(narrow_payload_bf16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mxfp4_ref;

    fn released_plan() -> MoeRoutedPlan {
        MoeRoutedPlan::new(MoeRoutedGeometry::released(), 40, TOP_K, 10.0).expect("released plan")
    }

    /// The one-token workspace spans, with every field at exactly the required length.
    fn spans() -> MoeRoutedOperands {
        let rows = TOP_K;
        MoeRoutedOperands {
            act_codes: DeviceSpan::new(Deviceptr(0x1000), HIDDEN),
            act_scales: DeviceSpan::new(Deviceptr(0x2000), HIDDEN / QUANT_BLOCK),
            ids: DeviceSpan::new(Deviceptr(0x3000), rows * 4),
            weights: DeviceSpan::new(Deviceptr(0x4000), rows * 4),
            delta_f32: DeviceSpan::new(Deviceptr(0x5000), HIDDEN * 4),
            delta_bf16: DeviceSpan::new(Deviceptr(0x6000), HIDDEN * 2),
        }
    }

    // ---------------------------------------------------------------------------------------
    // Declarations, constants and static asserts
    // ---------------------------------------------------------------------------------------

    #[test]
    fn the_staging_declaration_is_the_pinned_one() {
        let decl = stage_declaration();
        assert!(
            declaration_matches(&decl, STAGE_KERNEL, &STAGE_PARAMS),
            "the staging declaration drifted: {decl:?}"
        );
    }

    #[test]
    fn the_combine_declaration_is_the_pinned_one() {
        let decl = combine_declaration();
        assert!(
            declaration_matches(&decl, COMBINE_KERNEL, &COMBINE_PARAMS),
            "the combine declaration drifted: {decl:?}"
        );
    }

    #[test]
    fn the_weighted_declaration_is_the_pinned_one() {
        let decl = weighted_declaration();
        assert!(
            declaration_matches(&decl, WEIGHTED_KERNEL, &WEIGHTED_PARAMS),
            "the weighted SwiGLU declaration drifted: {decl:?}"
        );
    }

    #[test]
    fn the_declaration_pins_are_not_vacuous() {
        let mutated = SOURCE.replace(
            "    uint8_t *__restrict__ dst_weight,",
            "    const uint8_t *__restrict__ dst_weight,",
        );
        assert_ne!(mutated, SOURCE, "the mutation must apply");
        let decl = c_decl::kernel_declaration(&mutated, STAGE_KERNEL);
        assert!(
            !declaration_matches(&decl, STAGE_KERNEL, &STAGE_PARAMS),
            "a const staged weight operand must fail the ABI pin"
        );
        let mutated = SOURCE.replace(
            "    const uint32_t *__restrict__ expert_ids,",
            "    uint32_t *__restrict__ expert_ids,",
        );
        assert_ne!(mutated, SOURCE, "the mutation must apply");
        let decl = c_decl::kernel_declaration(&mutated, COMBINE_KERNEL);
        assert!(
            !declaration_matches(&decl, COMBINE_KERNEL, &COMBINE_PARAMS),
            "a non-const id operand must fail the ABI pin"
        );
    }

    #[test]
    fn the_kernel_constants_are_the_module_constants() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_MOE_TOP_K"), TOP_K as i128);
        assert_eq!(table.value("SPARK_M4_HIDDEN"), HIDDEN as i128);
        assert_eq!(
            table.value("SPARK_M4_MOE_WIDTH_PADDED"),
            WIDTH_PADDED as i128
        );
        assert_eq!(table.value("SPARK_M4_QUANT_BLOCK"), QUANT_BLOCK as i128);
        assert_eq!(
            table.value("SPARK_MOE_ROUTED_THREADS"),
            STAGED_COPY_THREADS as i128
        );
        assert_eq!(
            table.value("SPARK_MOE_ROUTED_TILE_COLS"),
            COMBINE_TILE_COLS as i128
        );
        assert_eq!(table.value("SPARK_MOE_ROUTED_TABLE_STRIDE"), 2);
        assert_eq!(
            table.value("SPARK_MOE_ROUTED_MAX_ROWS"),
            MAX_ROUTED_ROWS as i128
        );
        assert_eq!(table.value("SPARK_MOE_ROUTED_COPY_ALIGN"), 16);
        // Local width and padding are the M4 header's, which the landed launches pin too.
        assert_eq!(LOCAL_INTER, 576);
        assert_eq!(WIDTH_PADDED - LOCAL_INTER, 64);
    }

    #[test]
    fn every_static_assert_names_a_project_symbol_and_evaluates_true() {
        let table = constants();
        let asserts = c_decl::static_asserts(SOURCE);
        assert!(
            asserts.len() >= 4,
            "the kernel pins its geometry with asserts: {asserts:?}"
        );
        for assertion in &asserts {
            assert!(
                assertion.condition.contains("SPARK_"),
                "a tautology is not a geometry assert: {}",
                assertion.condition
            );
            assert_eq!(
                table.eval(&assertion.condition),
                1,
                "static_assert must evaluate true: {}",
                assertion.condition
            );
        }
    }

    // ---------------------------------------------------------------------------------------
    // The argument arrays
    // ---------------------------------------------------------------------------------------

    #[test]
    fn the_staging_words_are_the_declaration_order() {
        let params = ExpertStageLaunchParams {
            table: Deviceptr(0x10),
            expert_ids: Deviceptr(0x20),
            dst_weight: Deviceptr(0x30),
            dst_scales: Deviceptr(0x40),
            slot_acc: Deviceptr(0x50),
            rows: 6,
            experts: 384,
            weight_bytes: 1_474_560,
            scale_bytes: 92_160,
            acc_row_floats: HIDDEN as u32,
        };
        assert_eq!(
            params.words(),
            [
                0x10, 0x20, 0x30, 0x40, 0x50, 6, 384, 1_474_560, 92_160, 5120
            ]
        );
        assert_eq!(params.words().len(), STAGE_ARG_WORDS);
        assert_eq!(params.grid().expect("a row grid"), (6, 1, 1));
        // The accumulator operand is null when the launch does not zero it: a stale address must not
        // leak into the launch word.
        let no_zero = ExpertStageLaunchParams {
            acc_row_floats: 0,
            ..params
        };
        assert_eq!(no_zero.words()[4], 0);
        no_zero
            .validate()
            .expect("a null accumulator is valid without zeroing");
    }

    #[test]
    fn the_combine_words_are_the_declaration_order() {
        let params = CombineLaunchParams {
            slot_acc: Deviceptr(0x10),
            expert_ids: Deviceptr(0x20),
            delta_f32: Deviceptr(0x30),
            delta_bf16: Deviceptr(0x40),
            tokens: 1,
            hidden: HIDDEN as u32,
        };
        assert_eq!(params.words(), [0x10, 0x20, 0x30, 0x40, 1, 5120]);
        assert_eq!(params.words().len(), COMBINE_ARG_WORDS);
        assert_eq!(params.grid().expect("a column grid"), (1, 20, 1));
    }

    #[test]
    fn the_weighted_words_are_the_declaration_order() {
        let params = WeightedSwiGLULaunchParams {
            gate: Deviceptr(0x10),
            up: Deviceptr(0x20),
            weights: Deviceptr(0x30),
            out_fp8: Deviceptr(0x40),
            out_scales: Deviceptr(0x50),
            rows: 6,
            cols: WIDTH_PADDED as u32,
            limit: 10.0,
            trace_ring: 0,
            layer_id: 7,
            step_seq: 3,
        };
        assert_eq!(
            params.words(),
            [
                0x10,
                0x20,
                0x30,
                0x40,
                0x50,
                6,
                640,
                u64::from(10.0f32.to_bits()),
                0,
                7,
                3
            ]
        );
        assert_eq!(params.words().len(), WEIGHTED_ARG_WORDS);
        // 6 rows x 20 blocks = 120 warps, eight per CTA -> 15 CTAs.
        assert_eq!(params.grid().expect("a warp grid"), (15, 1, 1));
    }

    // ---------------------------------------------------------------------------------------
    // Geometry, plan and sizes
    // ---------------------------------------------------------------------------------------

    #[test]
    fn the_released_geometry_derives_the_pinned_slice_sizes() {
        let geometry = MoeRoutedGeometry::released();
        assert_eq!(geometry.local_inter(), 576);
        assert_eq!(geometry.packed_row_bytes(), 2560);
        assert_eq!(geometry.scale_row_bytes(), 160);
        assert_eq!(geometry.stage_weight_bytes(), 1_474_560);
        assert_eq!(geometry.stage_scale_bytes(), 92_160);
        assert_eq!(geometry.down_code_row_bytes(), 288);
        assert_eq!(geometry.down_scale_row_bytes(), 18);
        // One staging buffer serves all three matrices only because the two slices are the same size.
        assert_eq!(geometry.stage_weight_bytes(), geometry.down_weight_bytes());
        assert_eq!(geometry.stage_scale_bytes(), geometry.down_scale_bytes());
        // The stored kernel's row sizes are the reference's constants.
        assert_eq!(
            geometry.down_code_row_bytes(),
            mxfp4_ref::MOE_STORED_CODE_ROW
        );
        assert_eq!(
            geometry.down_scale_row_bytes(),
            mxfp4_ref::MOE_STORED_SCALE_ROW
        );
        assert_eq!(WIDTH_PADDED, mxfp4_ref::MOE_WIDTH_PADDED);
        assert_eq!(TOP_K, mxfp4_ref::MOE_TOP_K);
    }

    #[test]
    fn a_non_released_geometry_is_refused() {
        for (dim, inter, world, experts, top_k) in [
            (4096, INTER, WORLD, EXPERTS, TOP_K),
            (HIDDEN, 2048, WORLD, EXPERTS, TOP_K),
            (HIDDEN, INTER, 8, EXPERTS, TOP_K),
            (HIDDEN, INTER, WORLD, 256, TOP_K),
            (HIDDEN, INTER, WORLD, EXPERTS, 8),
        ] {
            let refused = MoeRoutedGeometry::new(dim, inter, world, experts, top_k);
            assert!(
                refused.is_err(),
                "a shape the pinned launches cannot serve must be refused: {refused:?}"
            );
        }
        assert_eq!(
            MoeRoutedGeometry::new(HIDDEN, INTER, WORLD, EXPERTS, TOP_K),
            Ok(MoeRoutedGeometry::released())
        );
    }

    #[test]
    fn a_plan_refuses_a_capacity_it_cannot_stage() {
        let geometry = MoeRoutedGeometry::released();
        assert!(MoeRoutedPlan::new(geometry, 0, TOP_K, 10.0).is_err());
        assert!(MoeRoutedPlan::new(geometry, 40, TOP_K - 1, 10.0).is_err());
        assert!(MoeRoutedPlan::new(geometry, 40, MAX_ROUTED_ROWS + 1, 10.0).is_err());
        assert!(MoeRoutedPlan::new(geometry, 40, TOP_K, f32::NAN).is_err());
        assert!(MoeRoutedPlan::new(geometry, 40, TOP_K, f32::INFINITY).is_err());
        // The 384-row ceiling admits 64 tokens (384 rows) and refuses 65 (390).
        let plan = MoeRoutedPlan::new(geometry, 40, MAX_ROUTED_ROWS, 10.0).expect("the ceiling");
        assert_eq!(plan.routed_rows(1).expect("one token"), TOP_K);
        assert_eq!(plan.routed_rows(64).expect("64 tokens"), 384);
        assert!(plan.routed_rows(65).is_err());
        assert!(plan.routed_rows(0).is_err());
        // The released decode plan stages one token's six rows and refuses a second token.
        let released = released_plan();
        assert_eq!(released.routed_rows(1).expect("one token"), TOP_K);
        assert!(released.routed_rows(2).is_err());
    }

    #[test]
    fn the_released_sizes_are_the_documented_bytes() {
        let plan = released_plan();
        let sizes = plan.sizes().expect("released sizes");
        // 40 layers x 3 matrices x 384 experts x 2 words x 8 bytes.
        assert_eq!(sizes.table_words, 40 * 3 * 384 * 2);
        assert_eq!(sizes.table_words * 8, 737_280);
        // Six routed rows of each staged slice.
        assert_eq!(sizes.stage_weight, 6 * 1_474_560);
        assert_eq!(sizes.stage_scales, 6 * 92_160);
        assert_eq!(sizes.gate_out, 6 * 640 * 2);
        assert_eq!(sizes.up_out, sizes.gate_out);
        assert_eq!(sizes.codes, 6 * 640);
        assert_eq!(sizes.scales, 6 * 20);
        assert_eq!(sizes.slot_acc, 6 * 5120 * 4);
        assert_eq!(sizes.slot_ids, 6 * 4);
        assert_eq!(
            sizes.total,
            737_280 + 6 * (1_474_560 + 92_160) + 2 * 7_680 + 3_840 + 120 + 122_880 + 24
        );
        // The whole run-scope state is one order of magnitude below the 40 layers' routed weights.
        assert!(sizes.total < 11 * 1024 * 1024);
    }

    #[test]
    fn the_size_arithmetic_refuses_an_overflow() {
        let geometry = MoeRoutedGeometry::released();
        assert!(MoeRoutedSizes::new(&geometry, usize::MAX, TOP_K).is_err());
        assert!(MoeRoutedSizes::new(&geometry, 40, usize::MAX).is_err());
        assert!(MoeRoutedSizes::new(&geometry, 40, MAX_ROUTED_ROWS).is_ok());
    }

    #[test]
    fn the_plan_checks_the_workspace_spans() {
        let plan = released_plan();
        let operands = spans();
        assert_eq!(
            plan.check_operands(0, 1, &operands).expect("a clean call"),
            TOP_K
        );
        // A layer outside the pointer table is refused before any launch.
        assert!(plan.check_operands(40, 1, &operands).is_err());
        // Every span is checked, and the refusal names the workspace field.
        for index in 0..6 {
            let mut short = operands;
            let (span, name) = match index {
                0 => (&mut short.act_codes, "workspace.act_codes"),
                1 => (&mut short.act_scales, "workspace.act_scales"),
                2 => (&mut short.ids, "workspace.moe_ids"),
                3 => (&mut short.weights, "workspace.moe_weights"),
                4 => (&mut short.delta_f32, "workspace.moe_delta"),
                _ => (&mut short.delta_bf16, "workspace.moe_delta_bf16"),
            };
            span.bytes -= 1;
            let error = plan.check_operands(0, 1, &short).expect_err("a short span");
            assert!(
                error.contains(name),
                "the refusal must name {name}: {error}"
            );
        }
        // The id and weight spans scale with the routed rows, not with the token count.
        let plan = MoeRoutedPlan::new(MoeRoutedGeometry::released(), 40, 2 * TOP_K, 10.0)
            .expect("a two-token plan");
        let mut batch = operands;
        batch.ids.bytes = 2 * TOP_K * 4;
        batch.weights.bytes = 2 * TOP_K * 4;
        batch.act_codes.bytes = 2 * HIDDEN;
        batch.act_scales.bytes = 2 * (HIDDEN / QUANT_BLOCK);
        batch.delta_f32.bytes = 2 * HIDDEN * 4;
        batch.delta_bf16.bytes = 2 * HIDDEN * 2;
        assert_eq!(
            plan.check_operands(0, 2, &batch).expect("two tokens"),
            2 * TOP_K
        );
    }

    #[test]
    fn the_staging_and_combine_launches_refuse_degenerate_arguments() {
        let mut stage = ExpertStageLaunchParams {
            table: Deviceptr(0x10),
            expert_ids: Deviceptr(0x20),
            dst_weight: Deviceptr(0x30),
            dst_scales: Deviceptr(0x40),
            slot_acc: Deviceptr(0x50),
            rows: 0,
            experts: 384,
            weight_bytes: 1_474_560,
            scale_bytes: 92_160,
            acc_row_floats: 0,
        };
        assert!(stage.validate().is_err());
        stage.rows = 6;
        stage.experts = 0;
        assert!(stage.validate().is_err());
        stage.experts = 384;
        stage.weight_bytes = 1_474_561;
        assert!(stage.validate().is_err());
        stage.weight_bytes = 1_474_560;
        stage.scale_bytes = 0;
        assert!(stage.validate().is_err());
        stage.scale_bytes = 92_160;
        stage.acc_row_floats = HIDDEN as u32;
        stage.slot_acc = Deviceptr(0);
        assert!(stage.validate().is_err());

        let mut combine = CombineLaunchParams {
            slot_acc: Deviceptr(0x10),
            expert_ids: Deviceptr(0x20),
            delta_f32: Deviceptr(0x30),
            delta_bf16: Deviceptr(0x40),
            tokens: 0,
            hidden: HIDDEN as u32,
        };
        assert!(combine.grid().is_err());
        combine.tokens = 1;
        combine.hidden = 0;
        assert!(combine.grid().is_err());
        // A hidden row that is not a whole multiple of the tile still grids: the kernel's column
        // guard covers the tail.
        combine.hidden = HIDDEN as u32 + 1;
        assert_eq!(combine.grid().expect("a column grid"), (1, 21, 1));

        let mut weighted = WeightedSwiGLULaunchParams {
            gate: Deviceptr(0x10),
            up: Deviceptr(0x20),
            weights: Deviceptr(0x30),
            out_fp8: Deviceptr(0x40),
            out_scales: Deviceptr(0x50),
            rows: 6,
            cols: 640,
            limit: 10.0,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        };
        weighted.cols = 641;
        assert!(weighted.grid().is_err());
        weighted.cols = 640;
        weighted.limit = f32::NAN;
        assert!(weighted.grid().is_err());
        weighted.limit = 10.0;
        weighted.rows = 0;
        assert!(weighted.grid().is_err());
    }

    // ---------------------------------------------------------------------------------------
    // The expert pointer table
    // ---------------------------------------------------------------------------------------

    fn synthetic_table(layers: usize, experts: usize) -> ExpertPointerTable {
        ExpertPointerTable::build(layers, experts, |layer, matrix, expert| {
            let base = 0x1_0000_0000u64
                + (layer as u64 * 3 + matrix.index() as u64) * 0x1_0000
                + expert as u64 * 0x10;
            (base, base + 0x8)
        })
        .expect("a synthetic table")
    }

    #[test]
    fn the_pointer_table_orders_layers_then_matrices_then_experts() {
        let table = synthetic_table(2, 3);
        assert_eq!(table.layers(), 2);
        assert_eq!(table.experts(), 3);
        assert_eq!(table.words().len(), 2 * 3 * 3 * 2);
        // The first entry is (layer 0, gate, expert 0) and the matrix axis advances before the expert
        // axis, so the staging launch's slice base is layer-major.
        assert_eq!(
            table.entry(0, RoutedMatrix::Gate, 0),
            (0x1_0000_0000, 0x1_0000_0008)
        );
        assert_eq!(
            table.entry(0, RoutedMatrix::Gate, 1),
            (0x1_0000_0010, 0x1_0000_0018)
        );
        assert_eq!(
            table.entry(0, RoutedMatrix::Up, 0),
            (0x1_0001_0000, 0x1_0001_0008)
        );
        assert_eq!(
            table.entry(0, RoutedMatrix::Down, 2),
            (0x1_0002_0020, 0x1_0002_0028)
        );
        assert_eq!(
            table.entry(1, RoutedMatrix::Gate, 0),
            (0x1_0003_0000, 0x1_0003_0008)
        );
        assert_eq!(table.slice_word_offset(0, RoutedMatrix::Gate), 0);
        assert_eq!(table.slice_word_offset(0, RoutedMatrix::Up), 6);
        assert_eq!(table.slice_word_offset(0, RoutedMatrix::Down), 12);
        assert_eq!(table.slice_word_offset(1, RoutedMatrix::Gate), 18);
        assert_eq!(table.entry_word_offset(1, RoutedMatrix::Down, 2), 34);
        // The plan's offset is the table's: the wiring passes one address computed in one place.
        let plan = released_plan();
        assert_eq!(plan.table_word_offset(0, RoutedMatrix::Gate), 0);
        assert_eq!(plan.table_word_offset(0, RoutedMatrix::Up), 384 * 2);
        assert_eq!(
            plan.table_word_offset(39, RoutedMatrix::Down),
            (39 * 3 + 2) * 384 * 2
        );
    }

    #[test]
    fn the_pointer_table_refuses_a_null_or_empty_slice() {
        let null_weight =
            ExpertPointerTable::build(1, 2, |_, _, _| (0, 0x1000)).expect_err("a null weight");
        assert!(
            null_weight.contains("weight address is null"),
            "{null_weight}"
        );
        let null_scales =
            ExpertPointerTable::build(1, 2, |_, _, _| (0x1000, 0)).expect_err("null scales");
        assert!(
            null_scales.contains("scale address is null"),
            "{null_scales}"
        );
        assert!(ExpertPointerTable::build(0, 2, |_, _, _| (1, 2)).is_err());
        assert!(ExpertPointerTable::build(1, 0, |_, _, _| (1, 2)).is_err());
    }

    #[test]
    fn the_pointer_table_upload_bytes_are_its_words_little_endian() {
        let table = synthetic_table(1, 1);
        let bytes = table.to_le_bytes();
        assert_eq!(bytes.len(), table.words().len() * 8);
        let first = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"));
        assert_eq!(first, table.words()[0]);
    }

    // ---------------------------------------------------------------------------------------
    // The CPU twins against the references
    // ---------------------------------------------------------------------------------------

    #[test]
    fn the_combine_slot_order_matches_the_reference() {
        let cases: [[u32; TOP_K]; 6] = [
            [0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0],
            [383, 0, 192, 7, 128, 64],
            [7, 7, 7, 7, 7, 7],
            [1, 0, 1, 0, 1, 0],
            [12, 300, 12, 5, 300, 0],
        ];
        for ids in cases {
            for token in 0..1 {
                assert_eq!(
                    combine_slot_order(&ids, token),
                    mxfp4_ref::moe_combine_slot_order(&ids, token),
                    "the twin's order must be the reference's for {ids:?}"
                );
            }
        }
    }

    #[test]
    fn the_combine_is_bit_identical_to_the_reference() {
        // A deterministic non-trivial corpus: values that exercise every exponent boundary, both
        // zeros, subnormals, infinities and NaN payloads, because the two roundings differ only
        // there.
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut values = Vec::new();
        for _ in 0..(TOP_K * 8 + TOP_K * 8 + 32) {
            let bits = next();
            values.push(match bits % 8 {
                0 => f32::from_bits(bits & 0x7fff_ffff),
                1 => f32::from_bits(bits | 0x8000_0000),
                2 => f32::from_bits(bits % 0x0080_0000),
                3 => f32::INFINITY.copysign(if bits & 1 == 0 { 1.0 } else { -1.0 }),
                4 => f32::from_bits(0x7fc0_0000 | (bits & 0x003f_ffff)),
                5 => 0.0f32,
                6 => -0.0f32,
                _ => f32::from_bits(bits & 0x7f7f_ffff),
            });
        }
        let hidden = 8;
        let tokens = 2;
        let mut slot_acc = vec![0.0f32; tokens * TOP_K * hidden];
        for (index, slot) in slot_acc.iter_mut().enumerate() {
            *slot = values[index % values.len()];
        }
        let ids: [u32; 2 * TOP_K] = [383, 0, 192, 7, 128, 64, 5, 5, 1, 383, 2, 0];
        let twin = combine_routed_f32(&slot_acc, &ids, tokens, hidden);
        let reference = mxfp4_ref::moe_combine_routed_f32(&slot_acc, &ids, tokens, hidden);
        assert_eq!(twin.len(), reference.len());
        for (index, (left, right)) in twin.iter().zip(&reference).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "value {index}: twin {left:?} ({:#010x}) != reference {right:?} ({:#010x})",
                left.to_bits(),
                right.to_bits()
            );
        }
        // The payload is the CLI's `bf16_le_bytes` of the same f32 row.
        let payload = combine_payload_bf16(&slot_acc, &ids, tokens, hidden);
        let expected: Vec<u16> = reference
            .iter()
            .map(|value| {
                let bits = value.to_bits();
                let lsb = (bits >> 16) & 1;
                (((bits.wrapping_add(0x7fff + lsb)) >> 16) & 0xffff) as u16
            })
            .collect();
        assert_eq!(payload, expected);
    }

    #[test]
    fn the_two_roundings_agree_on_finite_values_and_differ_only_on_a_nan() {
        // Every bf16 value widened to f32 is a fixed point of both roundings.
        for bits in 0u16..=u16::MAX {
            let value = crate::m4_quant::bf16_bits_to_f32(bits);
            if value.is_nan() {
                continue;
            }
            assert_eq!(
                round_slot_bf16(value).to_bits(),
                u32::from(bits) << 16,
                "a bf16 value must be its own round"
            );
            assert_eq!(
                narrow_payload_bf16(value),
                bits,
                "a bf16 value must be its own narrow"
            );
            assert_eq!(
                crate::m4_quant::bf16_bits(value),
                bits,
                "the reference's round agrees on the finite bf16 corpus"
            );
        }
        // A NaN keeps the reference's per-slot quieting (sign and the bits above bit 16) but takes
        // the payload path's wrapping add; the two formulas part company only here.
        let nan = f32::from_bits(0x7fff_ffff);
        assert_eq!(round_slot_bf16(nan).to_bits(), 0x7fff_0000);
        assert_eq!(narrow_payload_bf16(nan), 0x8000);
        assert_eq!(
            round_slot_bf16(f32::from_bits(0x7fc0_1234)).to_bits(),
            0x7fc0_0000
        );
        assert_eq!(narrow_payload_bf16(f32::from_bits(0x7fc0_1234)), 0x7fc0);
        assert_eq!(round_slot_bf16(f32::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(narrow_payload_bf16(f32::NAN), 0x7fc0);
        // Negative zero and the infinities pass through both.
        assert_eq!(round_slot_bf16(-0.0).to_bits(), 0x8000_0000);
        assert_eq!(narrow_payload_bf16(-0.0), 0x8000);
        assert_eq!(round_slot_bf16(f32::INFINITY).to_bits(), 0x7f80_0000);
        assert_eq!(narrow_payload_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(round_slot_bf16(f32::NEG_INFINITY).to_bits(), 0xff80_0000);
        assert_eq!(narrow_payload_bf16(f32::NEG_INFINITY), 0xff80);
    }

    #[test]
    fn the_rounding_formula_is_transcribed_from_the_kernel_source() {
        // The kernel's two helpers are the twin's formulas; the source text is the pin that they
        // were not "simplified" into the CUDA intrinsic, which differs on a NaN.
        assert!(SOURCE.contains("__float_as_uint(value)"));
        assert!(SOURCE.contains("(bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16"));
        assert!(SOURCE.contains("(bits & 0x7FFFFFFFu) > 0x7F800000u"));
        assert!(SOURCE.contains("return (bits >> 16) | 0x0040u;"));
        assert!(SOURCE.contains("__fadd_rn(total, expert_output)"));
        // No multiply in the combine: the routing weight was consumed by the weighted SwiGLU.
        let combine_body = SOURCE
            .rsplit_once(COMBINE_KERNEL)
            .expect("the combine declaration")
            .1;
        assert!(
            !combine_body.contains("weights"),
            "the combine kernel must not re-apply a routing weight"
        );
    }

    #[test]
    fn a_single_token_combine_is_the_native_finalize_routed_part() {
        // The contract's f32 delta plus the shared add is the native finalize: this pins that the
        // combine half is exactly what moe_combine_finalize consumes.
        let ids: [u32; TOP_K] = [3, 1, 4, 3, 5, 9];
        let hidden = 4;
        let mut slot_acc = vec![0.0f32; TOP_K * hidden];
        for (index, value) in slot_acc.iter_mut().enumerate() {
            *value = index as f32 * 0.25 - 1.5;
        }
        let routed = combine_routed_f32(&slot_acc, &ids, 1, hidden);
        assert_eq!(
            routed,
            mxfp4_ref::moe_combine_routed_f32(&slot_acc, &ids, 1, hidden)
        );
        let shared = vec![0u16; hidden];
        let finalize = mxfp4_ref::moe_combine_finalize(&slot_acc, &shared, &ids, 1, hidden);
        let payload = combine_payload_bf16(&slot_acc, &ids, 1, hidden);
        assert_eq!(payload, finalize, "the shared expert is zero here");
    }

    // ---------------------------------------------------------------------------------------
    // GPU tests: not run here (no device in this session) and reported as such
    // ---------------------------------------------------------------------------------------

    #[test]
    #[ignore = "requires a CUDA device: round-trips the pointer-table upload through the driver"]
    fn the_pointer_table_uploads_in_the_kernel_layout_on_a_device() {
        let Ok(cuda) = Cuda::load() else {
            println!("skipping: no CUDA driver in this session");
            return;
        };
        let plan = released_plan();
        let table = synthetic_table(plan.layers(), EXPERTS);
        let mut allocated = 0usize;
        let pointer = cuda
            .alloc(table.to_le_bytes().len(), &mut allocated)
            .expect("a table allocation");
        cuda.copy_htod(&pointer, &table.to_le_bytes())
            .expect("the table upload");
        let mut readback = vec![0u8; table.to_le_bytes().len()];
        cuda.copy_dtoh(&mut readback, &pointer)
            .expect("the table readback");
        assert_eq!(readback, table.to_le_bytes(), "the table must round-trip");
        cuda.free_buffers(&[&pointer]);
    }

    #[test]
    #[ignore = "requires a CUDA device: allocates the released run-scope state and frees it"]
    fn the_released_run_scope_state_is_one_device_allocation_set() {
        let Ok(cuda) = Cuda::load() else {
            println!("skipping: no CUDA driver in this session");
            return;
        };
        let sizes = released_plan().sizes().expect("released sizes");
        let mut allocated = 0usize;
        let pointer = cuda
            .alloc(sizes.total, &mut allocated)
            .expect("the run-scope allocation");
        assert_eq!(allocated, sizes.total);
        cuda.free_buffers(&[&pointer]);
    }
}
