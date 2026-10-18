//! The shared CUDA harness pieces the stage harnesses build their launches from.
//!
//! Two families live here, both moved out of examples/m7_layer0/main.rs so the m7 chain harness and the
//! m8 sequence harness cannot drift from each other:
//!
//! * FlashMlaParams and flash_mla_params: the one constructor for flash_mla_csa2_sm121's pinned
//!   19-argument ABI. A launch site that hand-built a stale array is the defect this module exists
//!   to make impossible; the examples' typed launch wrapper only accepts FlashMlaParams.
//! * EngineWindowRow, engine_window_row, assert_engine_window_finite and EngineWindowStats: the one
//!   construction of an engine-format window row (the engine's bf16 storage round, then its
//!   released FP8 activation quantiser, asserted equal to the m4_quant format oracle) and the guard
//!   that refuses a row whose window bytes decode to a non-finite value.
//! * FlashMlaKernel, load_flash_mla, launch_flash_mla and launch_flash_mla_repeated: the one
//!   launch path for flash_mla_csa2_sm121. The handle is opaque (its cuda_driver::Kernel field is
//!   private and there is no conversion), so an example cannot write a raw launch for this kernel:
//!   the only public launch entry points are the typed ones below, and cuda_driver::Cuda::launch
//!   only accepts cuda_driver::Kernel, which the flash_mla loader never returns. The compile_fail
//!   doctests on launch_flash_mla demonstrate the closed path, and a no_run counterpart shows the
//!   typed call compiles.
//!
//! The engine window helper calls sparkflash_engine::quant::quantize_activation_fp8, so the engine
//! crate is a normal dependency of this crate rather than only a dev-dependency of the examples.
//! The engine crate depends on no other workspace crate, so the edge creates no cycle.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
use crate::m4_quant::{act_quant_fp8_row, bf16_bits};
use crate::mla_ref;

/// The flash_mla_csa2_sm121 launch parameters, in the kernel's pinned ABI order. One constructor
/// owns the order and the fixed-size array; no launch site builds its own argument list, because
/// Cuda::launch takes a slice and cannot see a stale arity (a 15-element vector would make
/// cuLaunchKernel read four arguments past the end).
pub const FLASH_MLA_PARAM_COUNT: usize = 19;
/// selected_count is the one field a launch may vary after construction: the out-of-window
/// injection launches keep every other argument and change only the candidate count.
pub const FLASH_MLA_SELECTED_COUNT_INDEX: usize = 15;

/// The flash_mla launch arguments in the kernel's pinned order (see the kernel declaration and
/// m4_abi::SIGNATURES). Pointers are carried as u64 because that is what cuLaunchKernel's
/// argument array holds.
#[derive(Clone, Copy)]
pub struct FlashMlaParams([u64; FLASH_MLA_PARAM_COUNT]);

impl FlashMlaParams {
    /// The parameter words in the kernel's pinned ABI order, the shape cuLaunchKernel takes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }

    /// Returns the same parameter words with only the selected-candidate count replaced.
    #[must_use]
    pub fn with_selected_count(mut self, selected_count: usize) -> Self {
        self.0[FLASH_MLA_SELECTED_COUNT_INDEX] = selected_count as u64;
        self
    }
}

impl Default for FlashMlaParams {
    fn default() -> Self {
        Self([0; FLASH_MLA_PARAM_COUNT])
    }
}

/// Builds the flash_mla launch parameters with the kernel's own parameter names. Every launch in
/// the m7 harness (stage-2 indexer self-check, stage-2 attention, stage-7 K5, stage-9 K5) and every
/// launch in the m8 sequence harness goes through this one constructor; the typed launch wrapper
/// in the example passes as_slice() to cuLaunchKernel, so a future kernel signature change fails to
/// compile at every site instead of leaving one of them on a stale array.
#[allow(clippy::too_many_arguments)]
pub fn flash_mla_params(
    q: u64,
    kv_fp4: u64,
    kv_scales: u64,
    window_codes: u64,
    window_scales: u64,
    window_kv_format: u64,
    window_rows: u64,
    selected: u64,
    index_key: u64,
    index_query: u64,
    sink: u64,
    block_scores: u64,
    block_best: u64,
    out: u64,
    position: u64,
    selected_count: usize,
    trace_ring: u64,
    layer: usize,
    step_seq: u64,
) -> FlashMlaParams {
    FlashMlaParams([
        q,
        kv_fp4,
        kv_scales,
        window_codes,
        window_scales,
        window_kv_format,
        window_rows,
        selected,
        index_key,
        index_query,
        sink,
        block_scores,
        block_best,
        out,
        position,
        selected_count as u64,
        trace_ring,
        layer as u64,
        step_seq,
    ])
}

/// The opaque flash_mla kernel capability.
///
/// A value of this type can only be produced by load_flash_mla: the Kernel inside is private and
/// there is no accessor or conversion, so a launch of the flash_mla kernel cannot be written
/// through cuda_driver::Cuda::launch (which takes cuda_driver::Kernel) or through a raw
/// cuLaunchKernel pointer. The m7 harness and the forthcoming m8 sequence harness call
/// launch_flash_mla and launch_flash_mla_repeated below; that is the one launch path.
#[derive(Clone, Copy, Debug)]
pub struct FlashMlaKernel {
    function: Kernel,
}

/// Loads the flash_mla kernel from a cubin image, returning the module that keeps its code loaded.
///
/// The symbol is chosen here, not by the caller: cuda_driver exposes no way to load this kernel, so
/// the only handle that refers to it is the FlashMlaKernel returned here.
pub fn load_flash_mla(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, FlashMlaKernel), String> {
    let (module, function) = cuda.load_flash_mla(cubin)?;
    Ok((module, FlashMlaKernel { function }))
}

/// Launches flash_mla with the pinned argument order, then synchronises the context.
///
/// The handle is opaque and only launch_flash_mla and launch_flash_mla_repeated accept it. The raw
/// launch a bypass would need is a compile error in both directions:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::Cuda;
/// use sparkflash_kernels::cuda_harness::{FlashMlaParams, load_flash_mla};
///
/// fn bypass(cuda: &Cuda, params: &FlashMlaParams) -> Result<(), String> {
///     let (_module, kernel) = load_flash_mla(cuda, &[])?;
///     // The raw handle inside FlashMlaKernel is private to the harness module.
///     let raw = kernel.function;
///     cuda.launch(raw, (1, 1, 1), (1, 1, 1), params.as_slice())
/// }
/// ```
///
/// Passing the opaque handle to the generic launch is also rejected, because Cuda::launch takes a
/// cuda_driver::Kernel and FlashMlaKernel has no public conversion:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::Cuda;
/// use sparkflash_kernels::cuda_harness::{FlashMlaParams, load_flash_mla};
///
/// fn bypass(cuda: &Cuda, params: &FlashMlaParams) -> Result<(), String> {
///     let (_module, kernel) = load_flash_mla(cuda, &[])?;
///     cuda.launch(kernel, (1, 1, 1), (1, 1, 1), params.as_slice())
/// }
/// ```
///
/// The typed call below does compile, so the compile_fail cases are not vacuous:
///
/// ```no_run
/// use sparkflash_kernels::cuda_driver::Cuda;
/// use sparkflash_kernels::cuda_harness::{FlashMlaParams, launch_flash_mla, load_flash_mla};
///
/// fn typed(cuda: &Cuda, params: &FlashMlaParams) -> Result<(), String> {
///     let (_module, kernel) = load_flash_mla(cuda, &[])?;
///     launch_flash_mla(cuda, kernel, (1, 1, 1), (1, 1, 1), params)
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn launch_flash_mla(
    cuda: &Cuda,
    kernel: FlashMlaKernel,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &FlashMlaParams,
) -> Result<(), String> {
    cuda.launch(kernel.function, grid, block, params.as_slice())
}

/// Launches flash_mla repeatedly for timing with the pinned argument order.
///
/// Same opaque capability as launch_flash_mla; the per-launch and batched timings are the driver's
/// existing measurement.
#[allow(clippy::too_many_arguments)]
pub fn launch_flash_mla_repeated(
    cuda: &Cuda,
    kernel: FlashMlaKernel,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &FlashMlaParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    cuda.launch_repeated(kernel.function, grid, block, params.as_slice(), repeat)
}

/// The engine-format bytes of one window row: E4M3FN codes and one E8M0 scale per 32 channels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineWindowRow {
    /// The E4M3FN codes, one byte per latent channel.
    pub codes: Vec<u8>,
    /// The E8M0 scales, one byte per 32-channel block.
    pub scales: Vec<u8>,
}

/// The one construction of an engine-format window row, shared by stage 2 and stage 9 so the
/// storage/quantiser order cannot drift between the two harnesses. The caller has already applied
/// the layer's rms_norm and rope tail in f32; this function applies the engine's storage-dtype
/// round (bf16) exactly once, quantises with the engine's released
/// sparkflash_engine::quant::quantize_activation_fp8, and asserts the result is byte-identical to
/// m4_quant::act_quant_fp8_row on the same bf16 row. No second bf16 round happens here: the
/// engine quantiser returns the input dtype, so the bf16 row is the stored row.
pub fn engine_window_row(latent: &[f32]) -> Result<EngineWindowRow, String> {
    let bf16_row: Vec<f32> = latent
        .iter()
        .map(|value| half::bf16::from_bits(bf16_bits(*value)).to_f32())
        .collect();
    let engine_row = sparkflash_engine::quant::quantize_activation_fp8(&bf16_row)
        .map_err(|error| format!("engine window quantiser: {error}"))?;
    let (oracle_codes, oracle_scales) = act_quant_fp8_row(&bf16_row);
    if engine_row.codes() != oracle_codes.as_slice()
        || engine_row.scales() != oracle_scales.as_slice()
    {
        return Err(
            "the engine window quantiser and m4_quant::act_quant_fp8_row disagree on the same \
             bf16 row; the format oracle is not the engine's rule"
                .to_string(),
        );
    }
    Ok(EngineWindowRow {
        codes: engine_row.codes().to_vec(),
        scales: engine_row.scales().to_vec(),
    })
}

/// Refuse to hand engine window bytes to the kernel when they decode to a NaN or a non-finite
/// value. The engine quantiser clamps to +/-448 and never emits the E4M3 NaN code 0x7F, so this
/// guard turns a corrupt row into a loud failure instead of a poisoned attention output.
pub fn assert_engine_window_finite(
    codes: &[u8],
    scales: &[u8],
    rows: usize,
    context: &str,
) -> Result<(), String> {
    let nan_codes = codes.iter().filter(|code| **code == 0x7F).count();
    let mut non_finite = 0usize;
    for row in 0..rows {
        for index in 0..mla_ref::MLA_LATENT {
            if !mla_ref::window_kv_value(codes, scales, row, index).is_finite() {
                non_finite += 1;
            }
        }
    }
    if nan_codes != 0 || non_finite != 0 {
        return Err(format!(
            "{context}: engine window decode is not finite: {nan_codes} E4M3 NaN codes (0x7F) and \
             {non_finite} non-finite decoded values in {rows} rows; the engine quantiser cannot \
             emit these, so refuse the launch instead of letting the kernel compute on them"
        ));
    }
    Ok(())
}

/// One device window ring's geometry: the engine window format is 128 slots of one 512-byte E4M3
/// code row plus one 16-byte E8M0 scale row. The 528-byte slot is the rotation unit the Phase 2 2B
/// sequence harness measures; on the device the two arrays are separate, so every copy below moves
/// one array at a time and a "slot" is two rows of two different arrays.
pub const WINDOW_RING_SLOTS: usize = mla_ref::SWA_WINDOW as usize;
/// One E4M3 code row of one engine-format window slot.
pub const WINDOW_RING_CODE_ROW_BYTES: usize = mla_ref::MLA_LATENT;
/// One E8M0 scale row of one engine-format window slot.
pub const WINDOW_RING_SCALE_ROW_BYTES: usize = mla_ref::WINDOW_SCALE_BYTES;

/// The ring a window view is rotated out of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRingLayout {
    /// Slots in the ring; 128 for the released model's sliding window.
    pub slots: usize,
    /// Bytes in one slot's code array row.
    pub code_row_bytes: usize,
    /// Bytes in one slot's scale array row.
    pub scale_row_bytes: usize,
}

impl WindowRingLayout {
    /// The engine window format: 128 slots of 512 B codes + 16 B scales.
    pub const ENGINE: Self = Self {
        slots: WINDOW_RING_SLOTS,
        code_row_bytes: WINDOW_RING_CODE_ROW_BYTES,
        scale_row_bytes: WINDOW_RING_SCALE_ROW_BYTES,
    };

    /// Bytes of one slot across both arrays: 528 for the engine format. This is the host's raw slot
    /// size and the number a byte report quotes; the device never holds the two arrays interleaved.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.code_row_bytes + self.scale_row_bytes
    }

    /// Bytes in one full ring, both arrays.
    #[must_use]
    pub fn ring_bytes(&self) -> usize {
        self.slots * self.slot_bytes()
    }

    /// Refuses a layout that cannot describe a ring.
    pub fn validate(&self) -> Result<(), String> {
        if self.slots == 0 || self.code_row_bytes == 0 || self.scale_row_bytes == 0 {
            return Err(format!(
                "window ring layout needs non-zero slots/rows, got {self:?}"
            ));
        }
        Ok(())
    }
}

/// One contiguous source run of a ring written to a contiguous destination run of the oldest-first
/// view. A view wraps the ring at most once, so a rotation is at most two segments per array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingSegment {
    /// First source slot of the run.
    pub src_slot: usize,
    /// First destination row of the run in the contiguous view.
    pub dst_row: usize,
    /// Slots in the run.
    pub slots: usize,
}

/// The oldest-first window view at one query position: `(window_rows, oldest_slot)`.
///
/// Slot `position % slots` holds the row written at `position`. A layer writes slots for the
/// positions it actually executes, which need not start at 0: the 130-token CED schedule starts
/// encoder layers at 0 and decoder layers at 2, so at decoder position 2 the ring holds one row,
/// not three. The view for a query at `position` is therefore the last
/// `min(position - first_position + 1, slots)` slots, oldest first. Passing 0 for
/// `first_position` is the full-prompt case; any other value would otherwise rotate unwritten
/// slots into the view and corrupt the KV read.
pub fn ring_view_position(
    layout: &WindowRingLayout,
    first_position: usize,
    position: usize,
) -> Result<(usize, usize), String> {
    layout.validate()?;
    if first_position > position {
        return Err(format!(
            "window view at position {position} precedes the layer's first position {first_position}"
        ));
    }
    let window_rows = position
        .saturating_sub(first_position)
        .saturating_add(1)
        .min(layout.slots);
    let newest_slot = position % layout.slots;
    let oldest_slot = (newest_slot + layout.slots - (window_rows - 1)) % layout.slots;
    Ok((window_rows, oldest_slot))
}

/// Splits one oldest-first view into at most two contiguous ring runs, in view order.
pub fn ring_view_segments(
    layout: &WindowRingLayout,
    window_rows: usize,
    oldest_slot: usize,
) -> Result<Vec<RingSegment>, String> {
    layout.validate()?;
    if window_rows == 0 {
        return Err("a window view needs at least one row".to_string());
    }
    if window_rows > layout.slots {
        return Err(format!(
            "window rows {window_rows} exceed the ring's {} slots",
            layout.slots
        ));
    }
    if oldest_slot >= layout.slots {
        return Err(format!(
            "oldest slot {oldest_slot} is outside the ring's {} slots",
            layout.slots
        ));
    }
    let first = (layout.slots - oldest_slot).min(window_rows);
    let mut segments = Vec::with_capacity(2);
    segments.push(RingSegment {
        src_slot: oldest_slot,
        dst_row: 0,
        slots: first,
    });
    let wrapped = window_rows - first;
    if wrapped > 0 {
        segments.push(RingSegment {
            src_slot: 0,
            dst_row: first,
            slots: wrapped,
        });
    }
    Ok(segments)
}

/// Copies one ring into an oldest-first contiguous view with device-to-device copies, returning the
/// number of driver copies issued (two per segment: codes then scales).
///
/// The destination view is one code array and one scale array indexed by oldest-first row; the
/// source is the ring, indexed by slot. Both views have the same per-row sizes, so a segment's
/// byte ranges are the same arithmetic on both sides. This is the measured rotation the 2B harness
/// reports; it moves exactly the rows the kernel reads, never a full ring when the query needs less.
#[allow(clippy::too_many_arguments)] // four pointers, a layout and the view arithmetic are the store's geometry.
pub fn rotate_window_ring(
    cuda: &Cuda,
    dst_codes: &Deviceptr,
    dst_scales: &Deviceptr,
    src_codes: &Deviceptr,
    src_scales: &Deviceptr,
    layout: &WindowRingLayout,
    window_rows: usize,
    oldest_slot: usize,
) -> Result<usize, String> {
    let segments = ring_view_segments(layout, window_rows, oldest_slot)?;
    let mut copies = 0usize;
    for segment in &segments {
        let code_bytes = segment.slots * layout.code_row_bytes;
        let scale_bytes = segment.slots * layout.scale_row_bytes;
        cuda.copy_dtod(
            &dst_codes.offset(segment.dst_row * layout.code_row_bytes),
            &src_codes.offset(segment.src_slot * layout.code_row_bytes),
            code_bytes,
        )?;
        cuda.copy_dtod(
            &dst_scales.offset(segment.dst_row * layout.scale_row_bytes),
            &src_scales.offset(segment.src_slot * layout.scale_row_bytes),
            scale_bytes,
        )?;
        copies += 2;
    }
    Ok(copies)
}

/// Builds the expected oldest-first view on the host from a ring image (both arrays). The device run
/// uses it to check a rotation's order independently of the attention twin: the bytes the kernel
/// will read are compared against the slot order the ring math says they should be.
pub fn host_ring_view(
    codes: &[u8],
    scales: &[u8],
    layout: &WindowRingLayout,
    window_rows: usize,
    oldest_slot: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    layout.validate()?;
    if codes.len() != layout.slots * layout.code_row_bytes
        || scales.len() != layout.slots * layout.scale_row_bytes
    {
        return Err(format!(
            "ring image is {} B codes + {} B scales, expected {} slots x ({}, {})",
            codes.len(),
            scales.len(),
            layout.slots,
            layout.code_row_bytes,
            layout.scale_row_bytes
        ));
    }
    let segments = ring_view_segments(layout, window_rows, oldest_slot)?;
    let mut view_codes = Vec::with_capacity(window_rows * layout.code_row_bytes);
    let mut view_scales = Vec::with_capacity(window_rows * layout.scale_row_bytes);
    for segment in &segments {
        let code_start = segment.src_slot * layout.code_row_bytes;
        view_codes.extend_from_slice(
            &codes[code_start..code_start + segment.slots * layout.code_row_bytes],
        );
        let scale_start = segment.src_slot * layout.scale_row_bytes;
        view_scales.extend_from_slice(
            &scales[scale_start..scale_start + segment.slots * layout.scale_row_bytes],
        );
    }
    Ok((view_codes, view_scales))
}

/// Per-run engine-window counters for stage 9: rows built by the shared helper, oracle agreements,
/// and rows that passed the finite-decode guard.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EngineWindowStats {
    /// Rows built by the shared engine-window helper.
    pub rows: usize,
    /// Rows whose engine-quantiser bytes agreed with the m4_quant oracle.
    pub quantiser_agreements: usize,
    /// Rows that passed the finite-decode guard.
    pub guarded_rows: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_public_builder_is_the_one_flash_mla_parameter_pack() {
        let params = flash_mla_params(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 11, 17, 18, 19,
        );
        assert_eq!(params.as_slice().len(), FLASH_MLA_PARAM_COUNT);
        assert_eq!(params.as_slice()[FLASH_MLA_SELECTED_COUNT_INDEX], 11);
        assert_eq!(params.as_slice()[0], 1);
        assert_eq!(params.as_slice()[FLASH_MLA_PARAM_COUNT - 1], 19);
        let narrowed = params.with_selected_count(7);
        assert_eq!(narrowed.as_slice()[FLASH_MLA_SELECTED_COUNT_INDEX], 7);
        assert_eq!(narrowed.as_slice()[0], 1);
        assert_eq!(
            FlashMlaParams::default().as_slice(),
            &[0u64; FLASH_MLA_PARAM_COUNT]
        );
    }

    #[test]
    fn ring_view_positions_are_oldest_first_and_wrap_at_the_ring() {
        let layout = WindowRingLayout::ENGINE;
        assert_eq!(layout.slot_bytes(), 528);
        assert_eq!(layout.ring_bytes(), 128 * 528);
        // Inside the first pass the view is slots 0..=position, no wrap.
        for position in 0..128 {
            let (rows, oldest) =
                ring_view_position(&layout, 0, position).expect("view inside the ring");
            assert_eq!(rows, position + 1);
            assert_eq!(oldest, 0);
            assert_eq!(
                ring_view_segments(&layout, rows, oldest)
                    .expect("segments")
                    .len(),
                1
            );
        }
        // At and after the ring's first pass the view is the full ring, oldest first, wrapping.
        let cases = [
            (128usize, 128usize, 1usize),
            (129, 128, 2),
            (130, 128, 3),
            (255, 128, 0),
            (256, 128, 1),
        ];
        for (position, rows, oldest) in cases {
            let view = ring_view_position(&layout, 0, position).expect("view past the ring");
            assert_eq!(view, (rows, oldest), "position {position}");
        }
    }

    #[test]
    fn ring_view_position_respects_a_layer_that_starts_mid_ring() {
        let layout = WindowRingLayout::ENGINE;
        // The 130-token CED schedule starts decoder layers at position 2: at decoder position 2
        // the ring holds one row (slot 2), not the three a first_position of 0 would claim.
        assert_eq!(
            ring_view_position(&layout, 2, 2).expect("first decoder row"),
            (1, 2)
        );
        assert_eq!(
            ring_view_position(&layout, 2, 3).expect("second decoder row"),
            (2, 2)
        );
        // Decoder position 128 holds 127 rows, slots 2..=127 then 0; position 129 holds the full
        // ring and starts at slot 2.
        assert_eq!(
            ring_view_position(&layout, 2, 128).expect("decoder at 128"),
            (127, 2)
        );
        assert_eq!(
            ring_view_position(&layout, 2, 129).expect("decoder at 129"),
            (128, 2)
        );
        // Positions are absolute: the oldest slot is the newest slot minus rows-1, modulo the ring.
        for position in 2..200usize {
            let (rows, oldest) = ring_view_position(&layout, 2, position).expect("decoder view");
            assert_eq!(rows, (position - 1).min(128));
            assert_eq!(oldest, position.saturating_sub(rows - 1) % 128);
        }
        assert!(ring_view_position(&layout, 3, 2).is_err());
    }

    #[test]
    fn ring_view_segments_cover_every_row_exactly_once_in_view_order() {
        let layout = WindowRingLayout::ENGINE;
        for position in 0..300usize {
            let (rows, oldest) = ring_view_position(&layout, 0, position).expect("view");
            let segments = ring_view_segments(&layout, rows, oldest).expect("segments");
            assert!(!segments.is_empty() && segments.len() <= 2);
            let mut seen = Vec::new();
            let mut next_dst = 0usize;
            for segment in &segments {
                assert_eq!(segment.dst_row, next_dst);
                assert!(segment.slots > 0);
                for offset in 0..segment.slots {
                    seen.push((segment.src_slot + offset) % layout.slots);
                }
                next_dst += segment.slots;
            }
            assert_eq!(
                next_dst, rows,
                "position {position}: segments do not fill the view"
            );
            let expected: Vec<usize> = (0..rows).map(|row| (oldest + row) % layout.slots).collect();
            assert_eq!(seen, expected, "position {position}");
        }
        assert!(ring_view_segments(&layout, 0, 0).is_err());
        assert!(ring_view_segments(&layout, 129, 0).is_err());
        assert!(ring_view_segments(&layout, 1, 128).is_err());
    }

    #[test]
    fn host_ring_view_is_the_slot_order_the_math_requires() {
        let layout = WindowRingLayout::ENGINE;
        let codes: Vec<u8> = (0..layout.slots)
            .flat_map(|slot| vec![slot as u8; layout.code_row_bytes])
            .collect();
        let scales: Vec<u8> = (0..layout.slots)
            .flat_map(|slot| vec![(slot as u8).wrapping_mul(3); layout.scale_row_bytes])
            .collect();
        // Position 5: rows 0..=5, oldest 0.
        let (rows, oldest) = ring_view_position(&layout, 0, 5).expect("view");
        let (view_codes, view_scales) =
            host_ring_view(&codes, &scales, &layout, rows, oldest).expect("host view");
        assert_eq!(rows, 6);
        assert_eq!(view_codes.len(), 6 * layout.code_row_bytes);
        for row in 0..6 {
            assert_eq!(view_codes[row * layout.code_row_bytes], row as u8);
            assert_eq!(
                view_scales[row * layout.scale_row_bytes],
                (row as u8).wrapping_mul(3)
            );
        }
        // Position 129: oldest slot 2, so rows are slots 2..127 then 0..1.
        let (rows, oldest) = ring_view_position(&layout, 0, 129).expect("view");
        let (view_codes, _) = host_ring_view(&codes, &scales, &layout, rows, oldest).expect("view");
        assert_eq!(rows, 128);
        assert_eq!(view_codes[0], 2);
        assert_eq!(view_codes[125 * layout.code_row_bytes], 127);
        assert_eq!(view_codes[126 * layout.code_row_bytes], 0);
        assert_eq!(view_codes[127 * layout.code_row_bytes], 1);
        assert!(host_ring_view(&codes[..10], &scales, &layout, 1, 0).is_err());
    }

    #[test]
    fn the_public_engine_row_helper_quantises_and_guards() {
        let latent: Vec<f32> = (0..mla_ref::MLA_LATENT)
            .map(|index| (index as f32 * 0.01).sin())
            .collect();
        let row = engine_window_row(&latent).expect("a finite row must quantise");
        assert_eq!(row.codes.len(), mla_ref::MLA_LATENT);
        assert_eq!(row.scales.len(), mla_ref::MLA_LATENT / 32);
        assert_engine_window_finite(&row.codes, &row.scales, 1, "unit test")
            .expect("a finite row must pass the guard");
        let mut poisoned = row.codes.clone();
        poisoned[0] = 0x7F;
        assert!(
            assert_engine_window_finite(&poisoned, &row.scales, 1, "unit test").is_err(),
            "an E4M3 NaN code must be refused"
        );
    }
}
