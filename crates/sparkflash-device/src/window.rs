//! Window-ring accounting for the chain skeleton: the engine window geometry, the per-layer bytes a
//! 40-layer pass needs, and the oldest-first view arithmetic round-tripped against a host ring image.
//!
//! This module is pure host arithmetic over [WindowRingLayout]; it allocates no device memory and
//! opens no driver. The cross-check with the layout is byte-level so a change to the row sizes or the
//! slot count fails a test instead of silently changing the chain's memory story. The M3-a plan
//! quotes these numbers; the first device pass will use the same view arithmetic to rotate rings.

use sparkflash_kernels::cuda_harness::{
    WindowRingLayout, host_ring_view, ring_view_position, ring_view_segments,
};

use crate::chain::{ChainLayer, ChainSchedule};

/// The resident window-ring bytes for one schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowAccounting {
    /// The engine-format layout.
    pub layout: WindowRingLayout,
    /// Backbone layers, each with its own decoder window ring.
    pub layers: usize,
    /// One ring: the layout's slot count x its slot bytes.
    pub bytes_per_layer: usize,
    /// All layer rings resident at once.
    pub total_resident_bytes: usize,
}

impl WindowAccounting {
    /// Builds the accounting for a schedule.
    #[must_use]
    pub fn for_schedule(schedule: &ChainSchedule) -> Self {
        let layout = WindowRingLayout::ENGINE;
        let bytes_per_layer = layout.ring_bytes();
        Self {
            layout,
            layers: schedule.layers.len(),
            bytes_per_layer,
            total_resident_bytes: bytes_per_layer.saturating_mul(schedule.layers.len()),
        }
    }

    /// The accounting lines the plan report prints.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        vec![
            format!(
                "window ring: {} slots x ({} B codes + {} B scales) = {} B per layer",
                self.layout.slots,
                self.layout.code_row_bytes,
                self.layout.scale_row_bytes,
                self.bytes_per_layer
            ),
            format!(
                "window ring resident: {} layers x {} B = {} B (host accounting; no device allocation in M3-a)",
                self.layers, self.bytes_per_layer, self.total_resident_bytes
            ),
        ]
    }
}

/// The oldest-first view one layer reads at one position: rows and oldest slot.
///
/// # Errors
/// The layout or position arithmetic error from [ring_view_position].
pub fn layer_view(
    layout: &WindowRingLayout,
    layer: &ChainLayer,
    position: usize,
) -> Result<(usize, usize), String> {
    ring_view_position(layout, layer.first_position, position)
}

/// The host-side expected view of a ring image at one layer position, oldest first.
///
/// # Errors
/// The layout/row error from [ring_view_position], [ring_view_segments] or [host_ring_view].
pub fn layer_view_bytes(
    codes: &[u8],
    scales: &[u8],
    layer: &ChainLayer,
    position: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let layout = WindowRingLayout::ENGINE;
    let (rows, oldest) = layer_view(&layout, layer, position)?;
    host_ring_view(codes, scales, &layout, rows, oldest)
}

/// The contiguous-copy segment count for one layer position, the number of runs a rotation issues
/// (two driver copies per segment: codes then scales).
///
/// # Errors
/// The layout/row error from [ring_view_position] or [ring_view_segments].
pub fn layer_rotation_segments(
    layout: &WindowRingLayout,
    layer: &ChainLayer,
    position: usize,
) -> Result<usize, String> {
    let (rows, oldest) = layer_view(layout, layer, position)?;
    Ok(ring_view_segments(layout, rows, oldest)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::ChainSchedule;
    use sparkflash_kv::layout::{
        SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOT_DATA_BYTES, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL,
    };

    fn ring_image() -> (Vec<u8>, Vec<u8>) {
        let layout = WindowRingLayout::ENGINE;
        let mut codes = vec![0u8; layout.slots * layout.code_row_bytes];
        let mut scales = vec![0u8; layout.slots * layout.scale_row_bytes];
        for slot in 0..layout.slots {
            codes[slot * layout.code_row_bytes..(slot + 1) * layout.code_row_bytes]
                .fill(u8::try_from(slot).expect("slot fits"));
            scales[slot * layout.scale_row_bytes..(slot + 1) * layout.scale_row_bytes]
                .fill(u8::try_from(slot).expect("slot fits"));
        }
        (codes, scales)
    }

    #[test]
    fn the_engine_window_is_the_kv_layouts_slot() {
        let accounting = WindowAccounting::for_schedule(&ChainSchedule::milestone().expect("plan"));
        assert_eq!(accounting.layout, WindowRingLayout::ENGINE);
        assert_eq!(accounting.layout.slots, SWA_SLOTS_PER_TAIL);
        assert_eq!(accounting.layout.code_row_bytes, SWA_SLOT_DATA_BYTES);
        assert_eq!(accounting.layout.scale_row_bytes, SWA_SLOT_SCALE_BYTES);
        assert_eq!(accounting.layout.slot_bytes(), SWA_SLOT_BYTES);
        assert_eq!(
            accounting.layout.ring_bytes(),
            SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
        );
        assert_eq!(
            accounting.bytes_per_layer,
            SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
        );
        assert_eq!(accounting.layers, SWA_LAYERS);
        assert_eq!(
            accounting.total_resident_bytes,
            SWA_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
        );
        // The numbers the M3-a plan and the serving byte arithmetic quote, held to the layout.
        assert_eq!(accounting.bytes_per_layer, 67_584);
        assert_eq!(accounting.total_resident_bytes, 2_703_360);
    }

    #[test]
    fn a_decoder_layer_view_starts_at_its_first_position_not_at_zero() {
        let schedule = ChainSchedule::milestone().expect("plan");
        let layout = WindowRingLayout::ENGINE;
        // Position 2 for a decoder layer holds one row, not three: the ring has no slot 0/1 writes.
        assert_eq!(
            layer_view(&layout, &schedule.layers[21], 2).expect("view"),
            (1, 2)
        );
        // At position 5 it holds rows 2..=5, oldest 2.
        assert_eq!(
            layer_view(&layout, &schedule.layers[21], 5).expect("view"),
            (4, 2)
        );
        // An encoder layer at position 5 holds rows 0..=5, oldest 0.
        assert_eq!(
            layer_view(&layout, &schedule.layers[0], 5).expect("view"),
            (6, 0)
        );
        // The full window wraps: position 129, 130 rows requested, 128 rows from slot 2.
        assert_eq!(
            layer_view(&layout, &schedule.layers[0], 129).expect("view"),
            (128, 2)
        );
        assert!(layer_view(&layout, &schedule.layers[21], 1).is_err());
    }

    #[test]
    fn a_ring_view_round_trips_through_the_slot_order() {
        let schedule = ChainSchedule::milestone().expect("plan");
        let (codes, scales) = ring_image();
        let (view_codes, view_scales) =
            layer_view_bytes(&codes, &scales, &schedule.layers[21], 5).expect("view");
        assert_eq!(view_codes.len(), 4 * SWA_SLOT_DATA_BYTES);
        assert_eq!(view_scales.len(), 4 * SWA_SLOT_SCALE_BYTES);
        for row in 0..4 {
            let slot = 2 + row;
            assert!(
                view_codes[row * SWA_SLOT_DATA_BYTES..(row + 1) * SWA_SLOT_DATA_BYTES]
                    .iter()
                    .all(|byte| *byte == slot as u8),
                "code row {row} is slot {slot}"
            );
            assert!(
                view_scales[row * SWA_SLOT_SCALE_BYTES..(row + 1) * SWA_SLOT_SCALE_BYTES]
                    .iter()
                    .all(|byte| *byte == slot as u8),
                "scale row {row} is slot {slot}"
            );
        }
        // The wrapped view at 129: rows are slots 2..=127 then 0..=1.
        let (wrapped, _) =
            layer_view_bytes(&codes, &scales, &schedule.layers[0], 129).expect("wrapped view");
        assert_eq!(wrapped.len(), SWA_SLOTS_PER_TAIL * SWA_SLOT_DATA_BYTES);
        assert_eq!(wrapped[0], 2);
        assert_eq!(wrapped[126 * SWA_SLOT_DATA_BYTES], 0);
        assert_eq!(wrapped[127 * SWA_SLOT_DATA_BYTES], 1);
        assert_eq!(
            layer_rotation_segments(&WindowRingLayout::ENGINE, &schedule.layers[0], 129)
                .expect("segments"),
            2
        );
        assert_eq!(
            layer_rotation_segments(&WindowRingLayout::ENGINE, &schedule.layers[21], 5)
                .expect("segments"),
            1
        );
    }
}
