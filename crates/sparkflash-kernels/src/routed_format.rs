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

//! The routed experts of a block whose stored format the released MXFP4 bodies do not read.
//!
//! One served route for every such block, on the kernels the arms already share:
//!
//! 1. `moe_plan` (kernels/moe_plan_sm121.cu) groups the step's slots by expert and cuts each
//!    expert's run into items of [ITEM_ROWS] slots;
//! 2. the format's gate/up entry (kernels/routed_format_sm121.cu, the shared prefill tile's routed
//!    mode) writes each slot's gate and up rows from the f32 FFN rows, one launch a matrix;
//! 3. the base's clamped SwiGLU (`shared_swiglu_f32`) maps the two planes into a third;
//! 4. the format's down entry multiplies each slot's routing weight in and writes the slot's
//!    [hidden] partial;
//! 5. the base's routed combine (`opt_moe_combine_sm121`) sums a token's partials in ascending
//!    expert id with the shared expert's addend into the f32 and bf16 deltas.
//!
//! The expert table is `[layer][expert][9]` words: (weight, scales, wide) for w1, w3 and w2, where
//! `wide` is the f32 bit pattern of an NVFP4 convention's tensor-wide factor (zero elsewhere).
//! Which entries a layer launches is the plan's choice, made at open from the catalogue symbols
//! that decode the layer's format; a layer with no entries is the released path's, and
//! [RoutedFormatDevice::serves] says which.
//!
//! The buffers hold [RoutedFormatDevice::capacity] tokens; a longer step runs in pieces of that
//! many tokens, each piece's slots planned, computed and combined on their own (a token's delta
//! reads only its own slots, so the pieces are independent).

use crate::cuda_driver::{Cuda, Deviceptr, Kernel};

/// The table words a matrix and a matrix's offsets in an expert's entry (RF_TABLE_STRIDE,
/// RF_GATE, RF_UP, RF_DOWN in the source).
pub const TABLE_STRIDE: usize = 9;
/// The words one matrix of an expert takes: weight, scales, wide factor.
pub const MATRIX_WORDS: usize = 3;
/// The slots one plan item holds (GLM_MOE_ROWS): the tile's row count.
pub const ITEM_ROWS: usize = 128;
/// The image every format block's kernels live in.
pub const CUBIN: &str = "routed_format_sm121.cubin";
/// The entry that widens a split chunk's gathered BF16 rows into f32 rows ([SharedKernels::widen]).
pub const WIDEN: &str = "routed_format_widen_sm121";
/// The output columns one tile block computes (GLM_BN).
const TILE_COLS: usize = 128;
/// The block the tile entries run (OM_THREADS).
const TILE_THREADS: u32 = 256;
/// The elementwise launches' block.
const ELEMENT_THREADS: u32 = 256;
/// moe_plan's rule for an id outside the expert range: count it as expert 0 (MOE_PLAN_CLAMP).
const PLAN_CLAMP: u64 = 1;

/// One block's two format entries.
#[derive(Debug, Clone, Copy)]
pub struct FormatEntries {
    /// The gate/up entry (`routed_format_gate_up_<format>_sm121`).
    pub gate_up: Kernel,
    /// The down entry (`routed_format_down_<format>_sm121`).
    pub down: Kernel,
}

/// The kernels every format block shares.
#[derive(Debug, Clone, Copy)]
pub struct SharedKernels {
    /// `moe_plan` (the shared image).
    pub plan: Kernel,
    /// The threads one plan launch runs.
    pub plan_threads: u32,
    /// `shared_swiglu_f32`, the base's clamped SwiGLU.
    pub swiglu: Kernel,
    /// `opt_moe_combine_sm121`, the base's routed combine at the plan's hidden width and top-k.
    pub combine: Kernel,
    /// `routed_format_widen_sm121`: a split chunk's gathered BF16 rows into f32 rows.
    pub widen: Kernel,
}

/// The geometry one rank's routed experts have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedFormatGeometry {
    /// The model width (the gate/up contraction and the down output).
    pub dim: usize,
    /// One rank's slice of the expert intermediate.
    pub local_inter: usize,
    /// The routed experts a layer has.
    pub experts: usize,
    /// The experts a token routes to.
    pub top_k: usize,
}

/// One device step's inputs and outputs (device addresses).
#[derive(Debug, Clone, Copy)]
pub struct RoutedFormatRows {
    /// `[tokens][dim]` f32 FFN rows.
    pub x: u64,
    /// `[tokens][top_k]` u32 expert ids.
    pub ids: u64,
    /// `[tokens][top_k]` f32 routing weights.
    pub weights: u64,
    /// `[tokens][dim]` f32 delta, or 0 to leave it (the combine then writes a scratch row).
    pub delta_f32: u64,
    /// `[tokens][dim]` bf16 delta.
    pub delta_bf16: u64,
    /// `[tokens][dim]` f32 rows the combine adds (the shared expert), or 0 for none.
    pub addend: u64,
}

/// The run-scope state of the format route: the expert table, the per-layer entries and the
/// scratch planes.
pub struct RoutedFormatDevice {
    geometry: RoutedFormatGeometry,
    shared: SharedKernels,
    layers: Vec<Option<FormatEntries>>,
    table: Deviceptr,
    sorted: Deviceptr,
    items: Deviceptr,
    item_count: Deviceptr,
    gate: Deviceptr,
    up: Deviceptr,
    act: Deviceptr,
    slot_acc: Deviceptr,
    /// A `[capacity][dim]` f32 row the combine writes when the caller keeps no f32 delta.
    spare_f32: Deviceptr,
    capacity: usize,
    max_items: usize,
    limit: f32,
    device_bytes: usize,
    allocations: Vec<Deviceptr>,
}

impl RoutedFormatDevice {
    /// Uploads the expert table and allocates the planes for `capacity` tokens.
    ///
    /// `table` is `[layers][experts][9]` words; `layers[l]` names layer `l`'s entries, `None` for
    /// a layer the released path serves.
    ///
    /// # Errors
    /// A table of the wrong length, an empty geometry, a geometry the tiles cannot cut, or the
    /// driver's allocation or copy failure.
    pub fn open(
        cuda: &Cuda,
        geometry: RoutedFormatGeometry,
        shared: SharedKernels,
        layers: Vec<Option<FormatEntries>>,
        table: &[u64],
        capacity: usize,
        limit: f32,
    ) -> Result<Self, String> {
        let RoutedFormatGeometry {
            dim,
            local_inter,
            experts,
            top_k,
        } = geometry;
        if dim == 0 || local_inter == 0 || experts == 0 || top_k == 0 || capacity == 0 {
            return Err(format!(
                "routed format: an empty geometry ({geometry:?}, capacity {capacity})"
            ));
        }
        // The tile reads the contraction 32 columns a stage and writes whole 32-column tiles.
        if dim % 32 != 0 || local_inter % 32 != 0 {
            return Err(format!(
                "routed format: the tile reads 32-column blocks; width {dim} and slice \
                 {local_inter} must both be multiples of 32"
            ));
        }
        // The combine runs one 256-thread block a 256-column group of the hidden width.
        if dim % ELEMENT_THREADS as usize != 0 {
            return Err(format!(
                "routed format: the combine covers the width in {ELEMENT_THREADS}-column \
                 groups; width {dim} is not a multiple"
            ));
        }
        let expected = layers.len() * experts * TABLE_STRIDE;
        if table.len() != expected {
            return Err(format!(
                "routed format: the expert table holds {} words, {} layers x {experts} experts x \
                 {TABLE_STRIDE} need {expected}",
                table.len(),
                layers.len()
            ));
        }
        let slots = capacity * top_k;
        let max_items = experts + slots.div_ceil(ITEM_ROWS);
        let mut total = 0usize;
        let mut allocations = Vec::new();
        let mut alloc = |bytes: usize, what: &str| -> Result<Deviceptr, String> {
            let ptr = cuda
                .alloc(bytes.max(16), &mut total)
                .map_err(|error| format!("routed format: {what}: {error}"))?;
            allocations.push(ptr);
            Ok(ptr)
        };
        let table_ptr = alloc(table.len() * 8, "expert table")?;
        let sorted = alloc(slots * 4, "sorted slots")?;
        let items = alloc(max_items * 16, "plan items")?;
        let item_count = alloc(4, "plan item count")?;
        let gate = alloc(slots * local_inter * 4, "gate plane")?;
        let up = alloc(slots * local_inter * 4, "up plane")?;
        let act = alloc(slots * local_inter * 4, "SwiGLU plane")?;
        let slot_acc = alloc(slots * dim * 4, "slot partials")?;
        let spare_f32 = alloc(capacity * dim * 4, "spare f32 delta")?;
        let words: Vec<u8> = table.iter().flat_map(|word| word.to_le_bytes()).collect();
        cuda.copy_htod(&table_ptr, &words)
            .map_err(|error| format!("routed format: expert table upload: {error}"))?;
        Ok(Self {
            geometry,
            shared,
            layers,
            table: table_ptr,
            sorted,
            items,
            item_count,
            gate,
            up,
            act,
            slot_acc,
            spare_f32,
            capacity,
            max_items,
            limit,
            device_bytes: total,
            allocations,
        })
    }

    /// Whether this route serves `layer` (otherwise the released path does).
    #[must_use]
    pub fn serves(&self, layer: usize) -> bool {
        self.layers.get(layer).is_some_and(Option::is_some)
    }

    /// Whether any layer is served here.
    #[must_use]
    pub fn serves_any(&self) -> bool {
        self.layers.iter().any(Option::is_some)
    }

    /// The tokens one piece holds.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The device bytes the route allocated.
    #[must_use]
    pub const fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    /// Widens `count` BF16 values at `src` into f32 values at `dst` (exact).
    ///
    /// # Errors
    /// The driver's launch failure.
    pub fn widen(&self, cuda: &Cuda, src: u64, dst: u64, count: usize) -> Result<(), String> {
        let blocks = count.div_ceil(ELEMENT_THREADS as usize).clamp(1, 65_535) as u32;
        cuda.launch(
            self.shared.widen,
            (blocks, 1, 1),
            (ELEMENT_THREADS, 1, 1),
            &[src, dst, count as u64],
        )
        .map_err(|error| format!("routed format: widen: {error}"))
    }

    /// The routed experts of `n` tokens of `layer`, into the deltas.
    ///
    /// # Errors
    /// A layer this route does not serve, or the driver's launch failure.
    pub fn run(
        &self,
        cuda: &Cuda,
        layer: usize,
        n: usize,
        rows: &RoutedFormatRows,
    ) -> Result<(), String> {
        let Some(entries) = self.layers.get(layer).copied().flatten() else {
            return Err(format!(
                "routed format: layer {layer} is not a format block (the released path serves it)"
            ));
        };
        let RoutedFormatGeometry {
            dim,
            local_inter,
            experts,
            top_k,
        } = self.geometry;
        let table = self.table.0 + (layer * experts * TABLE_STRIDE * 8) as u64;
        let mut first = 0usize;
        while first < n {
            let m = (n - first).min(self.capacity);
            let at = |base: u64, row_bytes: usize| {
                if base == 0 {
                    0
                } else {
                    base + (first * row_bytes) as u64
                }
            };
            let ids = at(rows.ids, top_k * 4);
            let weights = at(rows.weights, top_k * 4);
            let slots = m * top_k;
            cuda.launch(
                self.shared.plan,
                (1, 1, 1),
                (self.shared.plan_threads, 1, 1),
                &[
                    ids,
                    self.sorted.0,
                    self.items.0,
                    self.item_count.0,
                    0,
                    slots as u64,
                    experts as u64,
                    top_k as u64,
                    ITEM_ROWS as u64,
                    self.max_items as u64,
                    0,
                    PLAN_CLAMP,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            )
            .map_err(|error| format!("routed format layer {layer}: plan: {error}"))?;
            let inter_tiles = local_inter.div_ceil(TILE_COLS) as u32;
            for (matrix, out) in [(0u64, self.gate.0), (1u64, self.up.0)] {
                cuda.launch(
                    entries.gate_up,
                    (inter_tiles, 1, self.max_items as u32),
                    (TILE_THREADS, 1, 1),
                    &[
                        table,
                        self.items.0,
                        at(rows.x, dim * 4),
                        out,
                        local_inter as u64,
                        dim as u64,
                        self.sorted.0,
                        top_k as u64,
                        matrix,
                    ],
                )
                .map_err(|error| format!("routed format layer {layer}: gate/up: {error}"))?;
            }
            let values = slots * local_inter;
            cuda.launch(
                self.shared.swiglu,
                (values.div_ceil(ELEMENT_THREADS as usize) as u32, 1, 1),
                (ELEMENT_THREADS, 1, 1),
                &[
                    self.gate.0,
                    self.up.0,
                    self.act.0,
                    u64::from(self.limit.to_bits()),
                    values as u64,
                ],
            )
            .map_err(|error| format!("routed format layer {layer}: SwiGLU: {error}"))?;
            cuda.launch(
                entries.down,
                (dim.div_ceil(TILE_COLS) as u32, 1, self.max_items as u32),
                (TILE_THREADS, 1, 1),
                &[
                    table,
                    self.items.0,
                    self.act.0,
                    weights,
                    self.slot_acc.0,
                    dim as u64,
                    local_inter as u64,
                    self.sorted.0,
                    top_k as u64,
                ],
            )
            .map_err(|error| format!("routed format layer {layer}: down: {error}"))?;
            let delta_f32 = match at(rows.delta_f32, dim * 4) {
                0 => self.spare_f32.0,
                delta => delta,
            };
            cuda.launch(
                self.shared.combine,
                ((dim / ELEMENT_THREADS as usize) as u32, m as u32, 1),
                (ELEMENT_THREADS, 1, 1),
                &[
                    ids,
                    self.slot_acc.0,
                    delta_f32,
                    at(rows.delta_bf16, dim * 2),
                    m as u64,
                    at(rows.addend, dim * 4),
                ],
            )
            .map_err(|error| format!("routed format layer {layer}: combine: {error}"))?;
            first += m;
        }
        Ok(())
    }

    /// Frees the route's buffers.
    pub fn release(self, cuda: &Cuda) {
        let borrowed: Vec<&Deviceptr> = self.allocations.iter().collect();
        cuda.free_buffers(&borrowed);
    }
}

/// One expert matrix's table words: its weight, its scale plane and its tensor-wide factor.
#[must_use]
pub fn matrix_words(weight: Deviceptr, scales: Deviceptr, wide: f32) -> [u64; MATRIX_WORDS] {
    [weight.0, scales.0, u64::from(wide.to_bits())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_carries_its_factor_as_the_low_word_bits() {
        let words = matrix_words(Deviceptr(0x1000), Deviceptr(0x2000), 0.5);
        assert_eq!(words, [0x1000, 0x2000, u64::from(0.5f32.to_bits())]);
        // The source reads the factor as `(uint32_t)word`, so the high half must stay zero.
        assert_eq!(words[2] >> 32, 0);
    }
}
