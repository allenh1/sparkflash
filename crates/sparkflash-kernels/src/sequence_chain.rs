//! The m8 sequence chain's shared device surface: the format-2 attention call, the host-seam
//! ledger, and the per-boundary cross-check record.
//!
//! # Boundary and duplication policy
//!
//! This module is the one place the m8 sequence chain names the CSA-union attention launch. It
//! deliberately does not contain kernel launch parameter lists of its own beyond
//! [`Format2AttentionCall::params`]: that method calls
//! [`crate::cuda_harness::flash_mla_params`], the constructor that owns the kernel's pinned
//! 19-argument order, and launches through [`crate::cuda_harness::launch_flash_mla`], the typed
//! opaque-handle launcher. It never calls `cuLaunchKernel` and never builds a raw argument array.
//! The flash_mla handle is opaque, so this module cannot bypass that path even by accident.
//!
//! `examples/m7_layer0/main.rs` is frozen device evidence (stage 2/7/9 pins) and keeps its own stage-9
//! chain. This module does not refactor it and does not import from it. Some launch argument
//! construction therefore looks similar to m7's; the duplication is accepted until 2B is green,
//! at which point a separate unification task can move m7 onto this module. The invariant that
//! matters now is that there is only one *checked* format-2 attention parameter builder, and it is
//! this one.
//!
//! # Host seams
//!
//! Where Phase 2 has no device kernel - the compressor, the indexer/HSI selection, and the Engram
//! writes at layers 1 and 14 - the chain may use a host seam. Every seam is recorded in a
//! [`SeamLedger`] with the bytes that cross the host/device boundary, its input/output sizes, and
//! whether it is scheduled to become a device kernel. A report that omitted the ledger would make
//! host work look like device work; the ledger exists so it cannot.
//!
//! # Incremental cross-checks
//!
//! The chain is built op by op, not as one launch. Each step can capture a
//! [`BoundarySample`] and compare it to the engine reference with [`compare_boundary`], so the
//! first divergence names the op (mHC, projection, attention, FFN, residual) instead of only
//! showing up as a final logit delta.
//!
//! This module contains no `unsafe` code. The one unsafe module in the crate is
//! `cuda_driver`, which this module calls through its safe surface.

use crate::cuda_driver::{Cuda, Deviceptr};
use crate::cuda_harness::{FlashMlaKernel, FlashMlaParams, flash_mla_params, launch_flash_mla};

/// The kernel's window-row bound for the CSA-union format.
pub const CSA_UNION_MAX_WINDOW_ROWS: u32 = 128;
/// The kernel's selected-entry bound.
pub const CSA_UNION_MAX_SELECTED: usize = 512;
/// The kernel's union capacity: selected plus window rows.
pub const CSA_UNION_MAX_ATTENDED: usize =
    CSA_UNION_MAX_SELECTED + CSA_UNION_MAX_WINDOW_ROWS as usize;

const _: () = assert!(CSA_UNION_MAX_ATTENDED == 640);

/// The window_kv_format value of `kernels/flash_mla_csa2_sm121.cu`'s CSA-union mode. The m8
/// example pins the symbolic macro at compile time; this duplicate is a compile-time reference to
/// the same numeric contract, not a second source of truth.
pub const CSA_UNION_WINDOW_KV_FORMAT: u64 = 2;

/// The compressed FP4 code row the kernel's union branch reads through `kv_fp4`: 512 latent values
/// packed two per byte, 256 bytes. This is the store/global-KV layout, not the 512-byte window
/// code row.
pub const CSA_UNION_COMPRESSED_CODE_ROW_BYTES: usize = 256;
/// The compressed scale row the union branch reads through `kv_scales`: one E4M3 scale per 16
/// channels, 32 bytes. This is the store/global-KV layout, not the 16-byte window E8M0 scale row.
pub const CSA_UNION_COMPRESSED_SCALE_ROW_BYTES: usize = 32;
/// The engine window code row: one E4M3 code per latent channel, 512 bytes. Window rows are indexed
/// by slot, not by absolute position.
pub const ENGINE_WINDOW_CODE_ROW_BYTES: usize = 512;
/// The engine window scale row: one E8M0 scale per 32 channels, 16 bytes.
pub const ENGINE_WINDOW_SCALE_ROW_BYTES: usize = 16;

const _: () = assert!(
    CSA_UNION_COMPRESSED_CODE_ROW_BYTES == 512 / 2
        && CSA_UNION_COMPRESSED_SCALE_ROW_BYTES == 512 / 16
        && ENGINE_WINDOW_CODE_ROW_BYTES == 512
        && ENGINE_WINDOW_SCALE_ROW_BYTES == 512 / 32
);

/// One format-2 attention launch: the expanded per-position compressed view, the window ring, the
/// engine-filtered selected ids, and the query position.
///
/// This is the signature task-14 (the isolated CSA-union device check) and the m8 chain share. The
/// selected ids are *values*: absolute token positions `c*ratio`, read by the kernel from
/// `kv_fp4`/`kv_scales` with `row = id`. They are not window rows. The reachability filter is
/// the caller's precondition; this call refuses only the kernel bounds, not a stale candidate list,
/// because it cannot see the engine's `compressed_indices` output.
#[derive(Debug, Clone)]
pub struct Format2AttentionCall<'a> {
    /// The query row, `[heads][latent]` bf16.
    pub q: Deviceptr,
    /// The expanded compressed-view codes, one [CSA_UNION_COMPRESSED_CODE_ROW_BYTES] (256-byte)
    /// row per absolute position: 512 latent values packed two per byte, the store/global-KV
    /// layout. This is not the 512-byte engine window code row.
    pub kv_fp4: Deviceptr,
    /// The expanded compressed-view scales, one [CSA_UNION_COMPRESSED_SCALE_ROW_BYTES] (32-byte)
    /// row per absolute position: one E4M3 scale per 16 channels. This is not the 16-byte engine
    /// window E8M0 scale row.
    pub kv_scales: Deviceptr,
    /// The window ring codes, oldest row first, one [ENGINE_WINDOW_CODE_ROW_BYTES] (512-byte) row
    /// per slot.
    pub window_codes: Deviceptr,
    /// The window ring scales, oldest row first, one [ENGINE_WINDOW_SCALE_ROW_BYTES] (16-byte) row
    /// per slot.
    pub window_scales: Deviceptr,
    /// The sink logit row.
    pub sink: Deviceptr,
    /// The selected-id buffer, `selected_ids.len()` u32 values.
    pub selected: Deviceptr,
    /// The byte offset of the output row.
    pub out: Deviceptr,
    /// The query position, the kernel's causal guard and trace field.
    pub position: u32,
    /// The layer number (trace only).
    pub layer: usize,
    /// The step sequence number (trace only).
    pub step_seq: u64,
    /// Window rows olders-first, at most 128.
    pub window_rows: u32,
    /// The engine-filtered selected absolute ids.
    pub selected_ids: &'a [u32],
    /// Optional trace ring; the production cubin is built without tracing, so this is normally null.
    pub trace_ring: Option<Deviceptr>,
    /// The HSI index-key rows the kernel's stage-1 indexer reads. Stage 1 runs before the union
    /// branch, so this must be a real device buffer even when the caller supplies the selected
    /// list itself; a null here faulted every launch this builder produced.
    pub index_key: Deviceptr,
    /// The index query rows stage 1 reads.
    pub index_query: Deviceptr,
    /// The stage-1 block-score buffer the kernel writes.
    pub block_scores: Deviceptr,
    /// The stage-1 block-best buffer the kernel writes.
    pub block_best: Deviceptr,
}

impl Format2AttentionCall<'_> {
    /// Checks the launch bounds and returns the pinned 19-word parameter array through the shared
    /// constructor. Every pointer word comes from the struct's pointer fields; the order comes from
    /// [`flash_mla_params`], never from this module. The kernel always runs indexer stage 1 before
    /// its union branch, so this method refuses null indexer buffers instead of launching into an
    /// illegal access.
    pub fn params(&self) -> Result<FlashMlaParams, String> {
        if self.window_rows > CSA_UNION_MAX_WINDOW_ROWS {
            return Err(format!(
                "format 2 window_rows {} exceeds {}",
                self.window_rows, CSA_UNION_MAX_WINDOW_ROWS
            ));
        }
        if self.selected_ids.len() > CSA_UNION_MAX_SELECTED {
            return Err(format!(
                "format 2 selected_count {} exceeds {}",
                self.selected_ids.len(),
                CSA_UNION_MAX_SELECTED
            ));
        }
        let union_rows = self.selected_ids.len() + self.window_rows as usize;
        if union_rows > CSA_UNION_MAX_ATTENDED {
            return Err(format!(
                "format 2 union of {union_rows} entries exceeds {}",
                CSA_UNION_MAX_ATTENDED
            ));
        }
        if !self.selected_ids.is_empty() && self.selected.0 == 0 {
            return Err(format!(
                "format 2 has {} selected ids but a null selected pointer",
                self.selected_ids.len()
            ));
        }
        if self.position as usize > i32::MAX as usize {
            return Err("format 2 position does not fit the kernel's i32 parameter".to_string());
        }
        for (name, ptr) in [
            ("index_key", self.index_key),
            ("index_query", self.index_query),
            ("block_scores", self.block_scores),
            ("block_best", self.block_best),
        ] {
            if ptr.0 == 0 {
                return Err(format!(
                    "format 2 {name} pointer is null; the kernel's indexer stage 1 dereferences it"
                ));
            }
        }
        Ok(flash_mla_params(
            self.q.0,
            self.kv_fp4.0,
            self.kv_scales.0,
            self.window_codes.0,
            self.window_scales.0,
            CSA_UNION_WINDOW_KV_FORMAT,
            self.window_rows as u64,
            self.selected.0,
            self.index_key.0,
            self.index_query.0,
            self.sink.0,
            self.block_scores.0,
            self.block_best.0,
            self.out.0,
            self.position as u64,
            self.selected_ids.len(),
            self.trace_ring.map_or(0, |ptr| ptr.0),
            self.layer,
            self.step_seq,
        ))
    }

    /// Launches through the one typed flash_mla path and synchronises the context.
    ///
    /// `grid` is `(heads, grid_y, 1)` and the block is `(threads, 1, 1)`; both are caller
    /// geometry pins rather than constants here, because the isolated task-14 check and the chain
    /// use the same call shape but may pin different grid slices for diagnosis.
    pub fn launch(
        &self,
        cuda: &Cuda,
        kernel: FlashMlaKernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), String> {
        let params = self.params()?;
        launch_flash_mla(cuda, kernel, grid, block, &params)
    }
}

/// What a host seam replaces, and how much work stays on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeamKind {
    /// The ratio compressor that produces compressed KV rows.
    Compressor,
    /// The indexer and HSI pool selection that produces the candidate list.
    IndexerSelection,
    /// The Engram read/write seam at the layers that have an Engram path.
    Engram,
    /// Any other host computation the chain uses temporarily.
    Other,
}

/// One host seam's byte accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeamRecord {
    /// A stable name for the report.
    pub name: &'static str,
    /// What device work the seam stands in for.
    pub kind: SeamKind,
    /// Bytes read from the device by the host.
    pub host_bytes_in: u64,
    /// Bytes written to the device by the host.
    pub host_bytes_out: u64,
    /// Bytes that cross the host/device boundary in either direction, if different from in + out
    /// (for example an overlay that is read and written in place).
    pub transfer_bytes: u64,
    /// Whether a device kernel is planned to remove the seam.
    pub phase3_followup: bool,
    /// One sentence the report quotes next to the number.
    pub note: &'static str,
}

impl SeamRecord {
    /// The seam's total host traffic, in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.host_bytes_in
            .saturating_add(self.host_bytes_out)
            .saturating_add(self.transfer_bytes)
    }
}

/// The seams a run used, in first-use order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeamLedger {
    records: Vec<SeamRecord>,
}

impl SeamLedger {
    /// Records one seam. Two records for the same layer/name are allowed: a seam that runs once
    /// per layer must show up once per layer, not be hidden by a set.
    pub fn record(&mut self, record: SeamRecord) {
        self.records.push(record);
    }

    /// Every recorded seam.
    #[must_use]
    pub fn records(&self) -> &[SeamRecord] {
        &self.records
    }

    /// Total host traffic across all seams, in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.records.iter().map(SeamRecord::total_bytes).sum()
    }

    /// Total host traffic for one seam kind, in bytes.
    #[must_use]
    pub fn bytes_of_kind(&self, kind: SeamKind) -> u64 {
        self.records
            .iter()
            .filter(|record| record.kind == kind)
            .map(SeamRecord::total_bytes)
            .sum()
    }

    /// The report lines, one per seam plus a total. Every line says explicitly that the seam is
    /// host work and whether it is scheduled to become a device kernel.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.records.len() + 1);
        for record in &self.records {
            lines.push(format!(
                "seam {} ({:?}): host in {} B, host out {} B, transfer {} B, total {} B, phase-3 device-kernel follow-up: {} - {}",
                record.name,
                record.kind,
                record.host_bytes_in,
                record.host_bytes_out,
                record.transfer_bytes,
                record.total_bytes(),
                if record.phase3_followup { "yes" } else { "no" },
                record.note
            ));
        }
        lines.push(format!(
            "seam total: {} B across {} seams (host work, NOT device work)",
            self.total_bytes(),
            self.records.len()
        ));
        lines
    }
}

/// One captured boundary tensor: the layer and op name, plus the values in the reference's order.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundarySample {
    /// Layer index.
    pub layer: usize,
    /// The op/stage label, for example "mhc_post_attention" or "attention_out".
    pub stage: &'static str,
    /// The values, in the same order as the reference dump.
    pub values: Vec<f32>,
}

/// The result of comparing one device boundary to the reference sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryComparison {
    /// The largest absolute difference.
    pub max_abs: f32,
    /// The index of the largest difference.
    pub index: usize,
    /// Whether every value is bit-identical.
    pub exact: bool,
    /// Whether max_abs is within the bar.
    pub within_bar: bool,
}

impl BoundaryComparison {
    /// The one-line report; a failure names the layer/stage and the first offending index.
    #[must_use]
    pub fn report(&self, sample: &BoundarySample, abs_bar: f32) -> String {
        format!(
            "boundary layer {} {}: max_abs {:.9} at {} (bar {:.9}), exact {}, within {}",
            sample.layer,
            sample.stage,
            self.max_abs,
            self.index,
            abs_bar,
            self.exact,
            self.within_bar
        )
    }
}

/// Compares a device boundary against the reference, refusing shape and non-finite mismatches
/// instead of turning them into a number. The device chain calls this after each op so the first
/// divergence is named by op, not just by final logits.
pub fn compare_boundary(
    device: &[f32],
    reference: &[f32],
    abs_bar: f32,
) -> Result<BoundaryComparison, String> {
    if device.len() != reference.len() {
        return Err(format!(
            "boundary shape mismatch: device {} values, reference {}",
            device.len(),
            reference.len()
        ));
    }
    if abs_bar < 0.0 || !abs_bar.is_finite() {
        return Err(format!("invalid absolute bar {abs_bar}"));
    }
    let mut max_abs = 0.0f32;
    let mut index = 0usize;
    let mut exact = true;
    for (position, (device_value, reference_value)) in
        device.iter().zip(reference.iter()).enumerate()
    {
        if !device_value.is_finite() || !reference_value.is_finite() {
            return Err(format!(
                "non-finite boundary value at index {position}: device {device_value}, reference {reference_value}"
            ));
        }
        if device_value.to_bits() != reference_value.to_bits() {
            exact = false;
        }
        let difference = (device_value - reference_value).abs();
        if difference > max_abs {
            max_abs = difference;
            index = position;
        }
    }
    Ok(BoundaryComparison {
        max_abs,
        index,
        exact,
        within_bar: max_abs <= abs_bar,
    })
}

/// One layer's work in the sequence chain: the absolute positions it runs, ascending.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerExtent {
    /// The layer id.
    pub layer: usize,
    /// The first absolute position this layer runs.
    pub first_position: usize,
    /// How many positions this layer runs.
    pub tokens: usize,
}

impl LayerExtent {
    /// The absolute position range, ascending.
    #[must_use]
    pub fn positions(&self) -> std::ops::Range<usize> {
        self.first_position..self.first_position + self.tokens
    }
}

/// The CED plan: layers 0..encoder_layers run the whole prompt, the remaining layers run the last
/// \`window\` positions. This is the scheduling fact the engine reference does not implement; the
/// chain follows the store's per-layer extents, and this type is the single host-side statement of
/// them so the device chain and the report cannot disagree about which token-layer executions ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequencePlan {
    /// The prompt length.
    pub prompt_tokens: usize,
    /// The window length, at most the prompt.
    pub window: usize,
    /// One extent per layer, in layer order.
    pub extents: Vec<LayerExtent>,
}

impl SequencePlan {
    /// Builds the CED plan: the first `encoder_layers` layers take all prompt positions, the rest
    /// take the last `window` positions. Refuses a window longer than the prompt, zero layers,
    /// or an encoder split outside the layer list.
    pub fn ced(
        prompt_tokens: usize,
        layers: usize,
        encoder_layers: usize,
        window: usize,
    ) -> Result<Self, String> {
        if prompt_tokens == 0 {
            return Err("the sequence plan needs a non-empty prompt".to_string());
        }
        if layers == 0 {
            return Err("the sequence plan needs at least one layer".to_string());
        }
        if window == 0 || window > prompt_tokens {
            return Err(format!("window {window} must be in 1..={prompt_tokens}"));
        }
        if encoder_layers > layers {
            return Err(format!(
                "encoder split {encoder_layers} exceeds {layers} layers"
            ));
        }
        let decoder_tokens = window.min(prompt_tokens);
        let decoder_first = prompt_tokens - decoder_tokens;
        let mut extents = Vec::with_capacity(layers);
        for layer in 0..layers {
            if layer < encoder_layers {
                extents.push(LayerExtent {
                    layer,
                    first_position: 0,
                    tokens: prompt_tokens,
                });
            } else {
                extents.push(LayerExtent {
                    layer,
                    first_position: decoder_first,
                    tokens: decoder_tokens,
                });
            }
        }
        Ok(Self {
            prompt_tokens,
            window,
            extents,
        })
    }

    /// The token-layer executions the plan runs: the denominator of any per-execution timing claim.
    #[must_use]
    pub fn token_layer_executions(&self) -> usize {
        self.extents.iter().map(|extent| extent.tokens).sum()
    }

    /// The one-line plan description.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "sequence plan: prompt {} tokens, window {}, {} layers, {} token-layer executions",
            self.prompt_tokens,
            self.window,
            self.extents.len(),
            self.token_layer_executions()
        )
    }
}

/// The boundary points the device chain compares after each block section, and the reference tensor
/// each one maps to. The names are the engine boundary format's names, not new labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStage {
    /// The normalised, quantised attention block input.
    MhcAttentionInput,
    /// The attention sublayer output before its post-mix.
    AttentionOutput,
    /// The stream after the attention post-mix.
    PostAttentionHidden,
    /// The normalised, quantised FFN block input.
    MhcFfnInput,
    /// The FFN sublayer output before its post-mix.
    FfnOutput,
    /// The block's residual output and the next block's input.
    ResidualOut,
}

impl BoundaryStage {
    /// The engine boundary tensor name.
    #[must_use]
    pub fn reference_name(self) -> &'static str {
        use sparkflash_engine::forward::boundary;
        match self {
            Self::MhcAttentionInput => boundary::PRE_ATTENTION_XHAT,
            Self::AttentionOutput => boundary::ATTENTION_OUT,
            Self::PostAttentionHidden => boundary::POST_ATTENTION_HIDDEN,
            Self::MhcFfnInput => boundary::PRE_FFN_XHAT,
            Self::FfnOutput => boundary::FFN_OUT,
            Self::ResidualOut => boundary::RESIDUAL_OUT,
        }
    }

    /// Whether the reference stores this boundary as packed code bytes rather than f32 values.
    #[must_use]
    pub fn is_quantised(self) -> bool {
        matches!(self, Self::MhcAttentionInput | Self::MhcFfnInput)
    }

    /// The chain order; the first divergence is reported at the earliest stage in this list.
    pub const ORDER: [Self; 6] = [
        Self::MhcAttentionInput,
        Self::AttentionOutput,
        Self::PostAttentionHidden,
        Self::MhcFfnInput,
        Self::FfnOutput,
        Self::ResidualOut,
    ];
}

/// Finds the engine reference tensor for a layer and boundary stage.
pub fn reference_tensor(
    table: &sparkflash_engine::forward::boundary::BoundaryTable,
    layer: usize,
    stage: BoundaryStage,
) -> Result<&sparkflash_engine::forward::boundary::BoundaryTensor, String> {
    let row = table
        .row(layer)
        .ok_or_else(|| format!("the reference boundary table has no layer {layer} row"))?;
    row.tensor(stage.reference_name()).ok_or_else(|| {
        format!(
            "the reference boundary table has no {} tensor in layer {layer}",
            stage.reference_name()
        )
    })
}

/// Compares a device f32 boundary to the engine reference tensor named by the stage.
pub fn compare_reference_f32(
    device: &[f32],
    table: &sparkflash_engine::forward::boundary::BoundaryTable,
    layer: usize,
    stage: BoundaryStage,
    abs_bar: f32,
) -> Result<BoundaryComparison, String> {
    if stage.is_quantised() {
        return Err(format!(
            "{} is a quantised boundary; use compare_reference_bytes",
            stage.reference_name()
        ));
    }
    let tensor = reference_tensor(table, layer, stage)?;
    let reference = tensor
        .data()
        .as_f32()
        .ok_or_else(|| format!("{} is not an f32 tensor", stage.reference_name()))?;
    compare_boundary(device, reference, abs_bar)
}

/// Compares a device code/scale boundary to the engine reference tensor, returning whether the
/// bytes match exactly and the first differing index (or the device length when shapes differ).
pub fn compare_reference_bytes(
    device: &[u8],
    table: &sparkflash_engine::forward::boundary::BoundaryTable,
    layer: usize,
    stage: BoundaryStage,
) -> Result<(bool, usize), String> {
    if !stage.is_quantised() {
        return Err(format!(
            "{} is an f32 boundary; use compare_reference_f32",
            stage.reference_name()
        ));
    }
    let tensor = reference_tensor(table, layer, stage)?;
    let reference = tensor
        .data()
        .as_u8()
        .ok_or_else(|| format!("{} is not a byte tensor", stage.reference_name()))?;
    if device.len() != reference.len() {
        return Ok((false, device.len().min(reference.len())));
    }
    for (index, (left, right)) in device.iter().zip(reference.iter()).enumerate() {
        if left != right {
            return Ok((false, index));
        }
    }
    Ok((true, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(selected: &[u32], window_rows: u32) -> Format2AttentionCall<'_> {
        Format2AttentionCall {
            q: Deviceptr(1),
            kv_fp4: Deviceptr(2),
            kv_scales: Deviceptr(3),
            window_codes: Deviceptr(4),
            window_scales: Deviceptr(5),
            sink: Deviceptr(6),
            selected: Deviceptr(7),
            out: Deviceptr(8),
            position: 129,
            layer: 20,
            step_seq: 0,
            window_rows,
            selected_ids: selected,
            trace_ring: None,
            index_key: Deviceptr(9),
            index_query: Deviceptr(10),
            block_scores: Deviceptr(11),
            block_best: Deviceptr(12),
        }
    }

    #[test]
    fn format2_params_use_the_pinned_constructor_and_bounds() {
        let ids = [0, 2, 4, 128];
        let params = call(&ids, 128).params().expect("params");
        let words = params.as_slice();
        assert_eq!(words.len(), crate::cuda_harness::FLASH_MLA_PARAM_COUNT);
        assert_eq!(words[5], CSA_UNION_WINDOW_KV_FORMAT);
        assert_eq!(words[6], 128);
        assert_eq!(words[7], 7);
        assert_eq!(words[14], 129);
        assert_eq!(words[15], ids.len() as u64);
        assert!(call(&ids, 129).params().is_err());
        let too_many: Vec<u32> = (0..=CSA_UNION_MAX_SELECTED as u32).collect();
        assert!(call(&too_many, 128).params().is_err());
        let mut null_selected = call(&ids, 128);
        null_selected.selected = Deviceptr(0);
        assert!(null_selected.params().is_err());
        // The four indexer buffers reach their own parameter words (8, 9, 11, 12); word 10 is the
        // sink. A null there is refused because the kernel's stage 1 runs even for format 2.
        assert_eq!(words[8], 9);
        assert_eq!(words[9], 10);
        assert_eq!(words[10], 6);
        assert_eq!(words[11], 11);
        assert_eq!(words[12], 12);
        for field in [
            |call: &mut Format2AttentionCall<'_>| call.index_key = Deviceptr(0),
            |call: &mut Format2AttentionCall<'_>| call.index_query = Deviceptr(0),
            |call: &mut Format2AttentionCall<'_>| call.block_scores = Deviceptr(0),
            |call: &mut Format2AttentionCall<'_>| call.block_best = Deviceptr(0),
        ] {
            let mut incomplete = call(&ids, 128);
            field(&mut incomplete);
            assert!(
                incomplete.params().is_err(),
                "a null indexer pointer must be refused"
            );
        }
    }

    #[test]
    fn compressed_and_window_rows_have_distinct_pinned_strides() {
        // The union branch reads kv_fp4 at 256 B (two latent per byte) and kv_scales at 32 B (one
        // E4M3 per 16 channels). The engine window rows are the separate 512 + 16 E8M0 layout.
        assert_eq!(CSA_UNION_COMPRESSED_CODE_ROW_BYTES, 256);
        assert_eq!(CSA_UNION_COMPRESSED_SCALE_ROW_BYTES, 32);
        assert_eq!(ENGINE_WINDOW_CODE_ROW_BYTES, 512);
        assert_eq!(ENGINE_WINDOW_SCALE_ROW_BYTES, 16);
        assert_ne!(
            CSA_UNION_COMPRESSED_CODE_ROW_BYTES,
            ENGINE_WINDOW_CODE_ROW_BYTES
        );
        assert_ne!(
            CSA_UNION_COMPRESSED_SCALE_ROW_BYTES,
            ENGINE_WINDOW_SCALE_ROW_BYTES
        );
    }

    #[test]
    fn seam_ledger_accounts_host_traffic_per_kind() {
        let mut ledger = SeamLedger::default();
        ledger.record(SeamRecord {
            name: "compressor.L20",
            kind: SeamKind::Compressor,
            host_bytes_in: 2_560,
            host_bytes_out: 512,
            transfer_bytes: 0,
            phase3_followup: true,
            note: "compressed row built on the host",
        });
        ledger.record(SeamRecord {
            name: "engram.L14",
            kind: SeamKind::Engram,
            host_bytes_in: 1_024,
            host_bytes_out: 1_024,
            transfer_bytes: 0,
            phase3_followup: true,
            note: "Engram gate and table read on the host",
        });
        assert_eq!(ledger.total_bytes(), 5_120);
        assert_eq!(ledger.bytes_of_kind(SeamKind::Compressor), 3_072);
        let report = ledger.report_lines();
        assert_eq!(report.len(), 3);
        assert!(report[2].contains("host work, NOT device work"));
    }

    #[test]
    fn boundary_comparison_names_the_first_difference() {
        let reference = [1.0f32, 2.0, 3.0];
        let exact = compare_boundary(&reference, &reference, 0.0).expect("exact");
        assert!(exact.exact && exact.within_bar && exact.max_abs == 0.0);
        let device = [1.0f32, 2.25, 3.0];
        let different = compare_boundary(&device, &reference, 0.2).expect("comparison");
        assert!(!different.exact && !different.within_bar);
        assert_eq!(different.index, 1);
        assert!((different.max_abs - 0.25).abs() < 1e-6);
        assert!(compare_boundary(&device[..2], &reference, 0.2).is_err());
        assert!(compare_boundary(&[f32::NAN], &[0.0], 0.2).is_err());
    }

    #[test]
    fn boundary_sample_report_names_the_op() {
        let sample = BoundarySample {
            layer: 3,
            stage: "attention_out",
            values: vec![0.0],
        };
        let comparison = compare_boundary(&[0.5], &[0.0], 0.25).expect("comparison");
        let line = comparison.report(&sample, 0.25);
        assert!(line.contains("layer 3 attention_out"));
        assert!(line.contains("within false"));
    }

    #[test]
    fn ced_plan_matches_the_21_19_extents_and_execution_count() {
        let plan = SequencePlan::ced(130, 40, 21, 128).expect("plan");
        assert_eq!(plan.extents.len(), 40);
        assert!(
            plan.extents[..21]
                .iter()
                .all(|extent| extent.first_position == 0 && extent.tokens == 130)
        );
        assert!(
            plan.extents[21..]
                .iter()
                .all(|extent| extent.first_position == 2 && extent.tokens == 128)
        );
        assert_eq!(plan.token_layer_executions(), 21 * 130 + 19 * 128);
        assert_eq!(plan.token_layer_executions(), 5_162);
        assert!(plan.describe().contains("5162"));
        assert!(SequencePlan::ced(130, 40, 41, 128).is_err());
        assert!(SequencePlan::ced(130, 40, 21, 131).is_err());
        assert!(SequencePlan::ced(0, 40, 21, 128).is_err());
    }

    #[test]
    fn boundary_stage_names_are_the_engine_names_in_chain_order() {
        use sparkflash_engine::forward::boundary;
        assert_eq!(
            BoundaryStage::MhcAttentionInput.reference_name(),
            boundary::PRE_ATTENTION_XHAT
        );
        assert_eq!(
            BoundaryStage::AttentionOutput.reference_name(),
            boundary::ATTENTION_OUT
        );
        assert_eq!(
            BoundaryStage::PostAttentionHidden.reference_name(),
            boundary::POST_ATTENTION_HIDDEN
        );
        assert_eq!(
            BoundaryStage::MhcFfnInput.reference_name(),
            boundary::PRE_FFN_XHAT
        );
        assert_eq!(BoundaryStage::FfnOutput.reference_name(), boundary::FFN_OUT);
        assert_eq!(
            BoundaryStage::ResidualOut.reference_name(),
            boundary::RESIDUAL_OUT
        );
        assert_eq!(BoundaryStage::ORDER.len(), 6);
        assert!(BoundaryStage::MhcAttentionInput.is_quantised());
        assert!(!BoundaryStage::AttentionOutput.is_quantised());
    }
}
