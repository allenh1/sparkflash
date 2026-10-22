//! The host twin of the three kv-path kernels: rope on the MLA latent tail, the window fp8
//! round-trip, and the cache append rule.
//!
//! This module is the reference the device port is judged against, and it is deliberately a thin
//! composition of the host functions rather than a second hand copy of them:
//!
//! * [rope_row_ref] reproduces the engine's rotation: adjacent pairs, tail only, f32 arithmetic on
//!   the bf16-stored row, one bf16 rounding on the way out.
//! * [window_quant_ref] is the released format: bf16-stored row, one absolute maximum per 32-value
//!   block, 1e-4 floor, round-up-to-a-power-of-two E8M0 scale, +-448 clamp, E4M3 round-to-nearest-even.
//!   It delegates to the m4 format oracle, which is the same code the engine path asserts byte
//!   equality against, so a divergence here is a divergence in the format and not in the arithmetic
//!   around it (see docs/archive/build/ws-kv-path-semantics-20261022.md section 3).
//! * [kv_append_rule] states the visibility rule as an executable check rather than prose: the
//!   append of position p happens after p's window read, so the cache length at the top of position p
//!   must be exactly p, and the row written at p is visible to positions p+1 and later only.
//!
//! The kernels themselves live in kernels/kv_path_sm121.cu (held in pending_kernels/ until a CUDA
//! host regenerates the pinned baseline: the baseline manifest records an nvcc cubin sha256, which a
//! host without nvcc cannot produce). The launchers are kv_path_launch.rs.

use crate::m4_quant;
use crate::mla_ref;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::rope_tables_for_layer;

/// The MLA latent width a rotated kv row has (the engine's MLA_LATENT).
pub const MLA_LATENT: usize = mla_ref::MLA_LATENT;

/// Values per window E8M0 scale (the released window quantiser's block).
pub const WINDOW_QUANT_BLOCK: usize = m4_quant::QUANT_BLOCK;

/// The E4M3FN code bytes of one window row.
pub const WINDOW_CODE_BYTES: usize = MLA_LATENT;

/// The E8M0 scale bytes of one window row.
pub const WINDOW_SCALE_BYTES: usize = MLA_LATENT / WINDOW_QUANT_BLOCK;

/// The number of rows the reference seam can select at most (the window), for span sizing.
pub const WINDOW_ROWS: usize = mla_ref::SWA_WINDOW as usize;

/// Rotates the tail of one bf16 row exactly as the engine's linalg::rope_rotate does.
///
/// The host path is: kv_norm in f32, narrow to bf16, rotate the last rope_head_dim values in f32,
/// narrow to bf16 again. This function takes the already-narrowed row and reproduces the second
/// stage: the tail is read as bf16 (exact in f32), rotated with the adjacent-pair convention
/// (a, b) -> (a cos - b sin, a sin + b cos), and rounded back to bf16 once. The head is copied
/// unchanged - the engine rotates only kv[start..] with start = mla_width - rope_head_dim.
///
/// # Errors
/// Refuses a rope_head_dim that is zero, odd, or wider than the row, and tables that do not carry
/// one cos and one sin per rotated pair.
pub fn rope_row_ref(
    row: &[u16],
    cos: &[f32],
    sin: &[f32],
    head_width: usize,
    rope_head_dim: usize,
) -> Result<Vec<u16>, String> {
    if row.len() != head_width {
        return Err(format!(
            "rope: the row is {} values, one head is {head_width}",
            row.len()
        ));
    }
    if rope_head_dim == 0 || rope_head_dim > head_width || !rope_head_dim.is_multiple_of(2) {
        return Err(format!(
            "rope: a {rope_head_dim}-long tail does not fit a {head_width}-wide row"
        ));
    }
    let pairs = rope_head_dim / 2;
    if cos.len() != pairs || sin.len() != pairs {
        return Err(format!(
            "rope: {rope_head_dim} values need {pairs} cos and {pairs} sin, got {} and {}",
            cos.len(),
            sin.len()
        ));
    }
    let mut out = row.to_vec();
    let start = head_width - rope_head_dim;
    for pair in 0..pairs {
        let a = m4_quant::bf16_value(row[start + 2 * pair]);
        let b = m4_quant::bf16_value(row[start + 2 * pair + 1]);
        let first = a * cos[pair] - b * sin[pair];
        let second = a * sin[pair] + b * cos[pair];
        out[start + 2 * pair] = m4_quant::bf16_bits(first);
        out[start + 2 * pair + 1] = m4_quant::bf16_bits(second);
    }
    Ok(out)
}

/// The host's kv_norm-then-rope sequence on the device's bf16 operands, with ONE bf16 rounding.
///
/// This is the twin of the fused kernel: it widens the bf16 latent and the bf16 weight, computes the
/// row's mean square in ascending f32 order, then (x * inv * w) in f32, rotates the adjacent tail
/// pairs in f32, and rounds each final f32 to bf16 once - the order linalg::rms_norm then
/// linalg::rope_rotate then a single bf16 cast produce on the host.
///
/// The declared deviation is the operand, not the arithmetic: the device latent is bf16 where the
/// host has f32, so the widening here is where that one extra rounding enters. The twin tests pin
/// both facts - equality with the host sequence given the same bf16 operand, and the measured size of
/// the f32-vs-bf16 latent gap.
///
/// # Errors
/// Refuses a row or weight of the wrong width, a tail that does not fit as an even run, and tables
/// that are not one cos and one sin per rotated pair.
pub fn kv_norm_rope_ref(
    latent: &[u16],
    norm: &[u16],
    eps: f32,
    cos: &[f32],
    sin: &[f32],
    head_width: usize,
    rope_head_dim: usize,
) -> Result<Vec<u16>, String> {
    if latent.len() != head_width || norm.len() != head_width {
        return Err(format!(
            "kv norm rope: the latent is {} and the weight {} values, one row is {head_width}",
            latent.len(),
            norm.len()
        ));
    }
    if rope_head_dim == 0 || rope_head_dim > head_width || !rope_head_dim.is_multiple_of(2) {
        return Err(format!(
            "kv norm rope: a {rope_head_dim}-long tail does not fit a {head_width}-wide row as an even run"
        ));
    }
    let pairs = rope_head_dim / 2;
    if cos.len() != pairs || sin.len() != pairs {
        return Err(format!(
            "kv norm rope: {rope_head_dim} values need {pairs} cos and {pairs} sin, got {} and {}",
            cos.len(),
            sin.len()
        ));
    }
    let values: Vec<f32> = latent
        .iter()
        .map(|bits| m4_quant::bf16_value(*bits))
        .collect();
    let mut sum = 0.0_f32;
    for value in &values {
        sum += value * value;
    }
    let mean = sum / head_width as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    let mut normed = Vec::with_capacity(head_width);
    for index in 0..head_width {
        let weight = m4_quant::bf16_value(norm[index]);
        normed.push(values[index] * inv * weight);
    }
    let start = head_width - rope_head_dim;
    for pair in 0..pairs {
        let a = normed[start + 2 * pair];
        let b = normed[start + 2 * pair + 1];
        normed[start + 2 * pair] = a * cos[pair] - b * sin[pair];
        normed[start + 2 * pair + 1] = a * sin[pair] + b * cos[pair];
    }
    Ok(normed
        .iter()
        .map(|value| m4_quant::bf16_bits(*value))
        .collect())
}

/// Quantises one bf16-stored row into the engine window format: E4M3FN codes and E8M0 scales.
///
/// The row arrives already bf16-rounded (the storage-dtype round happened when the rotated row was
/// written), so no second bf16 round is applied: widening a bf16 value to f32 is exact, which is
/// exactly what the engine's quantizer sees because it returns the input dtype.
///
/// # Panics
/// Panics if the row is not a whole number of 32-value blocks, which is the format oracle's own
/// precondition (a partial block has no defined scale).
#[must_use]
pub fn window_quant_ref(row: &[u16]) -> (Vec<u8>, Vec<u8>) {
    let widened: Vec<f32> = row.iter().copied().map(m4_quant::bf16_value).collect();
    let (codes, scales) = m4_quant::act_quant_fp8_row(&widened);
    assert_eq!(codes.len(), row.len(), "one code byte per value");
    assert_eq!(scales.len(), row.len() / WINDOW_QUANT_BLOCK);
    (codes, scales)
}

/// The append rule of one position, as a check a test (or the port's debug path) can run.
///
/// The loop's invariant, verbatim from the served loop: at the top of position p the cache holds
/// exactly p rows (the position loop left the cache out of step otherwise), the window read at p
/// slices the rows visible before p's append, and p's row is pushed after that read. So:
/// [kv_append_rule] accepts cache_len == position and the append of exactly one row, and refuses
/// anything else; a caller that appends before the window read sees position + 1 rows, which is the
/// "window one row too long" bug stated as an error instead of a comment.
///
/// # Errors
/// Refuses a cache length that is not exactly the position being appended, and a row width that is
/// not one MLA latent row.
pub fn kv_append_rule(cache_len: usize, position: usize, row_width: usize) -> Result<(), String> {
    if cache_len != position {
        return Err(format!(
            "kv append: the cache holds {cache_len} rows at position {position}; the append must come after the window read of this position, so the cache is exactly position rows long (a longer cache means the append ran before the read)"
        ));
    }
    if row_width != MLA_LATENT {
        return Err(format!(
            "kv append: the row is {row_width} values, one window row is {MLA_LATENT}"
        ));
    }
    Ok(())
}

/// Appends one position's k/v row pair to the reference cache, exactly as kv_append_bf16 writes it.
///
/// The row for position p goes to slot p % rows: the k row at the slot's first head_dim values, the
/// v row immediately after (the frozen DeviceKvCache geometry, row_values = 2 * head_dim). Only that
/// slot is touched; the kernel never reads the cache, and neither does this reference, so the ring's
/// other rows are byte-identical afterwards.
///
/// This is the cache write of the host loop's cache push, expressed in the device cache's ring
/// coordinates: the host cache grows and the device cache wraps, which is safe because the window
/// seam reads the last window_width positions through the same modulo and rows older than the
/// window are never read.
///
/// # Errors
/// Refuses a zero ring, a zero head width, a cache span shorter than rows rows, and rows shorter than
/// the k/v rows, naming the requirement.
pub fn kv_append_ref(
    cache: &mut [u16],
    k_row: &[u16],
    v_row: &[u16],
    rows: usize,
    head_dim: usize,
    position: usize,
) -> Result<(), String> {
    if rows == 0 {
        return Err("kv append: the ring has zero rows".to_owned());
    }
    if head_dim == 0 {
        return Err("kv append: head_dim is zero".to_owned());
    }
    if k_row.len() < head_dim || v_row.len() < head_dim {
        return Err(format!(
            "kv append: the k row is {} values and the v row {}, a {head_dim}-wide row is needed",
            k_row.len(),
            v_row.len()
        ));
    }
    let row_values = head_dim
        .checked_mul(2)
        .ok_or_else(|| "kv append: 2 * head_dim overflows".to_owned())?;
    let need = rows
        .checked_mul(row_values)
        .ok_or_else(|| "kv append: rows * row_values overflows".to_owned())?;
    if cache.len() < need {
        return Err(format!(
            "kv append: the cache holds {} values, a {rows}-row ring of {row_values}-value rows needs {need}",
            cache.len()
        ));
    }
    let slot = position % rows;
    let base = slot * row_values;
    cache[base..base + head_dim].copy_from_slice(&k_row[..head_dim]);
    cache[base + head_dim..base + row_values].copy_from_slice(&v_row[..head_dim]);
    Ok(())
}

/// The row at "position" in a ring cache, as the window seam would read it (k then v).
///
/// # Errors
/// Refuses a zero ring or a cache span shorter than rows rows.
pub fn kv_row_ref(
    cache: &[u16],
    rows: usize,
    head_dim: usize,
    position: usize,
) -> Result<&[u16], String> {
    if rows == 0 {
        return Err("kv row: the ring has zero rows".to_owned());
    }
    let row_values = head_dim
        .checked_mul(2)
        .ok_or_else(|| "kv row: 2 * head_dim overflows".to_owned())?;
    let need = rows
        .checked_mul(row_values)
        .ok_or_else(|| "kv row: rows * row_values overflows".to_owned())?;
    if cache.len() < need {
        return Err(format!(
            "kv row: the cache holds {} values, a {rows}-row ring of {row_values}-value rows needs {need}",
            cache.len()
        ));
    }
    let slot = position % rows;
    Ok(&cache[slot * row_values..slot * row_values + row_values])
}

/// One layer's rope cos/sin tables for a whole position range, as the device arm uploads them once.
///
/// The tables are the engine's own construction over a range instead of for one position: the range
/// is built by calling the engine's per-layer table builder with every position in
/// [0, max_positions), so the fill arithmetic is the same f32 multiply per position and the same f32
/// trig, the YaRN ramp (f64 band edges, f32 ramp that re-derives 1 - smooth) is the engine's, and a
/// compressed layer's base (compress_rope_theta) is chosen by the same layer argument the host loop
/// passes. A range is allowed by the semantics doc precisely because the arithmetic is not
/// reimplemented here; a cumulative product would be a different table and is not what this builds.
///
/// Two layers can share a table: the base and the ramp depend only on whether the layer compresses
/// its KV, so [rope_table_class] names the class and a caller builds one range per class rather than
/// one per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeRange {
    positions: usize,
    pairs: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeRange {
    /// How many positions the range carries.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// How many f32 values one table has per position (rope_head_dim / 2).
    #[must_use]
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// The index of position's first pair in the cos and sin arrays.
    ///
    /// Refuses a position at or beyond the range: a table that wrapped or clamped would rotate a
    /// later position by an earlier one's angle, which no logits bar could attribute to the rope.
    ///
    /// # Errors
    /// A position outside [0, positions).
    pub fn pair_offset(&self, position: usize) -> Result<usize, String> {
        if position >= self.positions {
            return Err(format!(
                "rope range: position {position} is outside the {}-position table; rebuild it for the run's context length instead of clamping",
                self.positions
            ));
        }
        Ok(position * self.pairs)
    }

    /// Position's cos values (rope_head_dim / 2 f32), in pair order.
    ///
    /// # Errors
    /// A position outside the range.
    pub fn cos(&self, position: usize) -> Result<&[f32], String> {
        let offset = self.pair_offset(position)?;
        Ok(&self.cos[offset..offset + self.pairs])
    }

    /// Position's sin values (rope_head_dim / 2 f32), in pair order.
    ///
    /// # Errors
    /// A position outside the range.
    pub fn sin(&self, position: usize) -> Result<&[f32], String> {
        let offset = self.pair_offset(position)?;
        Ok(&self.sin[offset..offset + self.pairs])
    }

    /// The bytes one upload of both tables costs: positions * pairs * 4, twice.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.positions * self.pairs * 4 * 2
    }
}

/// Builds a layer's rope tables for every position in [0, max_positions).
///
/// # Errors
/// Refuses a zero position count, a layer outside the model's layer count, and a table the engine
/// itself refuses.
pub fn rope_tables_range(
    config: &ModelConfig,
    layer: usize,
    max_positions: usize,
) -> Result<RopeRange, String> {
    if max_positions == 0 {
        return Err("rope range: zero positions would build an empty table".to_owned());
    }
    if layer >= config.n_layers {
        return Err(format!(
            "rope range: layer {layer} of {} requested",
            config.n_layers
        ));
    }
    let positions: Vec<usize> = (0..max_positions).collect();
    let tables = rope_tables_for_layer(config, layer, &positions);
    let pairs = config.rope_head_dim / 2;
    if tables.cos.len() != max_positions * pairs || tables.sin.len() != max_positions * pairs {
        return Err(format!(
            "rope range: the engine built {} cos and {} sin for {max_positions} positions of {pairs} pairs",
            tables.cos.len(),
            tables.sin.len()
        ));
    }
    Ok(RopeRange {
        positions: max_positions,
        pairs,
        cos: tables.cos,
        sin: tables.sin,
    })
}

/// Which rope table class a layer belongs to: 0 for the plain base, 1 for the compressed base with
/// the YaRN ramp.
///
/// The class is the whole of the per-layer dependence: the engine chooses base and ramp from whether
/// the layer compresses its KV (compress_ratio > 0), so layers in one class share a range table.
///
/// # Errors
/// A layer outside the model's layer count.
pub fn rope_table_class(config: &ModelConfig, layer: usize) -> Result<usize, String> {
    if layer >= config.n_layers {
        return Err(format!(
            "rope range: layer {layer} of {} requested",
            config.n_layers
        ));
    }
    Ok(usize::from(config.compress_ratio(layer).unwrap_or(0) > 0))
}

/// The arm-private quantised window ring's layout: one codes array of rows * 512 bytes, then one
/// scales array of rows * 16 bytes, slot indexed by position modulo the row count.
///
/// This is a layout, not an allocation: the arm owns the span (allocated once at open from the run
/// arena, its bytes reported next to layer_device_memory) and asks this type for the byte offset of
/// a slot's codes and scales. One ring type in the repo, so the stage and the kernel cannot disagree
/// about where a row's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRingLayout {
    rows: usize,
}

impl WindowRingLayout {
    /// The layout for a ring of this many rows, refusing a zero ring.
    ///
    /// # Errors
    /// A zero row count.
    pub fn new(rows: usize) -> Result<Self, String> {
        if rows == 0 {
            return Err("window ring: a zero-row ring has no slot to write".to_owned());
        }
        Ok(Self { rows })
    }

    /// Ring capacity in rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The bytes the ring's codes array occupies.
    #[must_use]
    pub fn codes_bytes(&self) -> usize {
        self.rows * WINDOW_CODE_BYTES
    }

    /// The bytes the ring's scales array occupies.
    #[must_use]
    pub fn scales_bytes(&self) -> usize {
        self.rows * WINDOW_SCALE_BYTES
    }

    /// The bytes one allocation for the ring needs, 256-byte aligned (the slab's alignment).
    #[must_use]
    pub fn alloc_bytes(&self) -> usize {
        (self.codes_bytes() + self.scales_bytes()).div_ceil(seam_align()) * seam_align()
    }

    /// One layer's ring as the arm allocates it: the 256-byte aligned per-layer size.
    ///
    /// This is the number that goes next to layer_device_memory, and [Self::total_bytes] multiplies
    /// it once, so the run record and the allocation cannot disagree.
    #[must_use]
    pub fn ring_bytes(&self) -> usize {
        self.alloc_bytes()
    }

    /// The byte offset of layer's ring inside a run allocation of one ring per layer.
    ///
    /// Pure arithmetic: the caller bounds layer by its own layer count, as DeviceKvCache does for
    /// its layer spans, and [Self::total_bytes] is the matching size.
    #[must_use]
    pub fn layer_offset(&self, layer: usize) -> usize {
        layer * self.alloc_bytes()
    }

    /// The whole run's ring allocation for n_layers layers, one ring per layer.
    ///
    /// # Errors
    /// A zero layer count, which would be an allocation with no ring.
    pub fn total_bytes(&self, n_layers: usize) -> Result<usize, String> {
        if n_layers == 0 {
            return Err("window ring: zero layers would allocate no ring".to_owned());
        }
        Ok(n_layers * self.alloc_bytes())
    }

    /// The ring slot a position writes: position modulo the row count, as the append kernel computes
    /// it, so the stage and the kernel cannot disagree about which slot a position owns.
    #[must_use]
    pub fn slot_for(&self, position: usize) -> usize {
        position % self.rows
    }

    /// The byte offset of a slot's code bytes inside the ring span.
    ///
    /// # Errors
    /// A slot at or beyond the row count: a wrapped slot would overwrite a row the window still
    /// reads.
    pub fn codes_offset(&self, slot: usize) -> Result<usize, String> {
        self.check_slot(slot)?;
        Ok(slot * WINDOW_CODE_BYTES)
    }

    /// The byte offset of a slot's scale bytes inside the ring span.
    ///
    /// # Errors
    /// A slot at or beyond the row count.
    pub fn scales_offset(&self, slot: usize) -> Result<usize, String> {
        self.check_slot(slot)?;
        Ok(self.codes_bytes() + slot * WINDOW_SCALE_BYTES)
    }

    fn check_slot(&self, slot: usize) -> Result<(), String> {
        if slot >= self.rows {
            return Err(format!(
                "window ring: slot {slot} of {} requested; the slot is position % rows",
                self.rows
            ));
        }
        Ok(())
    }
}

/// The seam's buffer alignment (the slab's): the ring's allocation must carry it.
fn seam_align() -> usize {
    256
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m4_quant::{act_quant_fp8_row, bf16_bits, bf16_value};
    use sparkflash_engine::linalg;

    fn patterned_row(width: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..width)
            .map(|index| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
                (unit * 2.0 - 1.0) * (1.0 + (index % 7) as f32)
            })
            .collect()
    }

    fn bf16_row(values: &[f32]) -> Vec<u16> {
        values.iter().map(|value| bf16_bits(*value)).collect()
    }

    #[test]
    fn the_rope_reference_matches_the_engine_on_a_bf16_row() {
        use sparkflash_engine::config::ModelConfig;
        use sparkflash_engine::forward::attention::rope_tables_for_layer;
        use sparkflash_engine::linalg;

        let config = ModelConfig::v41_flash_cached();
        let width = config.head_dim;
        let rope = config.rope_head_dim;
        let position = 37usize;
        for layer in [0usize, 2, 14] {
            let tables = rope_tables_for_layer(config, layer, &[position]);
            let values = patterned_row(width, 0x5eed + layer as u64);
            let stored = bf16_row(&values);
            // The host's own sequence: narrow to bf16, rotate the f32 view of the stored row.
            let mut host: Vec<f32> = stored.iter().copied().map(bf16_value).collect();
            let start = width - rope;
            linalg::rope_rotate(&mut host[start..], &tables.cos, &tables.sin)
                .expect("the engine rotates a whole tail");
            let host: Vec<u16> = host.iter().map(|value| bf16_bits(*value)).collect();
            let reference =
                rope_row_ref(&stored, &tables.cos, &tables.sin, width, rope).expect("a valid row");
            assert_eq!(reference, host, "layer {layer}: the rope reference drifted");
        }
    }

    #[test]
    fn the_rope_reference_only_touches_the_tail() {
        use sparkflash_engine::config::ModelConfig;
        let config = ModelConfig::v41_flash_cached();
        let width = config.head_dim;
        let rope = config.rope_head_dim;
        let row = bf16_row(&patterned_row(width, 11));
        let cos = vec![0.5f32; rope / 2];
        let sin = vec![0.5f32; rope / 2];
        let rotated = rope_row_ref(&row, &cos, &sin, width, rope).expect("a valid row");
        assert_eq!(
            &rotated[..width - rope],
            &row[..width - rope],
            "the head must not move"
        );
        assert_eq!(rotated.len(), width);
    }

    #[test]
    fn the_window_reference_is_the_m4_format_oracle() {
        let row = bf16_row(&patterned_row(MLA_LATENT, 9));
        let widened: Vec<f32> = row.iter().copied().map(bf16_value).collect();
        let (codes, scales) = window_quant_ref(&row);
        let (oracle_codes, oracle_scales) = act_quant_fp8_row(&widened);
        assert_eq!(codes, oracle_codes);
        assert_eq!(scales, oracle_scales);
        assert_eq!(codes.len(), WINDOW_CODE_BYTES);
        assert_eq!(scales.len(), WINDOW_SCALE_BYTES);
    }

    #[test]
    fn the_window_reference_rounds_bf16_once_and_never_double_rounds() {
        let mut values = patterned_row(MLA_LATENT, 21);
        // A value that is not representable in bf16: the format sees the bf16 row.
        values[3] = 1.0 + 2.0_f32.powi(-10);
        let stored = bf16_row(&values);
        let widened: Vec<f32> = stored.iter().copied().map(bf16_value).collect();
        let (codes, _) = window_quant_ref(&stored);
        let (oracle_codes, _) = act_quant_fp8_row(&widened);
        assert_eq!(codes, oracle_codes);
        // Quantising the raw f32 row would be a second, different input.
        let (raw_codes, _) = act_quant_fp8_row(&values);
        assert_ne!(
            codes, raw_codes,
            "the stored bf16 row is the format's input"
        );
    }

    #[test]
    fn the_window_reference_is_the_engine_path_row_end_to_end() {
        // The composed host path: engine_window_row rounds f32 to bf16 once and quantises that row
        // (asserting the m4 oracle itself). The twin takes the stored bf16 row and must agree byte
        // for byte, so the port's "quantise the rotated row" is the engine's own construction and
        // not a second copy of the format.
        let values = patterned_row(MLA_LATENT, 77);
        let stored = bf16_row(&values);
        let engine = crate::cuda_harness::engine_window_row(&values).expect("the engine row");
        let (codes, scales) = window_quant_ref(&stored);
        assert_eq!(codes, engine.codes);
        assert_eq!(scales, engine.scales);
    }

    #[test]
    fn the_window_reference_keeps_an_all_zero_block_finite() {
        let codes = window_quant_ref(&vec![0u16; MLA_LATENT]).0;
        assert_eq!(codes, vec![0u8; WINDOW_CODE_BYTES]);
        assert!(
            codes.iter().all(|code| *code != 0x7F),
            "no NaN code is produced"
        );
    }

    #[test]
    fn an_all_infinite_row_takes_the_largest_scale_and_saturates() {
        let mut row = vec![bf16_bits(0.0f32); MLA_LATENT];
        row[0] = bf16_bits(f32::INFINITY);
        let (codes, scales) = window_quant_ref(&row);
        assert_eq!(scales[0], 254, "inf takes the largest E8M0 scale");
        assert_eq!(codes[0], 0x7E, "+-448 is the saturating code");
    }

    #[test]
    fn the_rope_reference_refuses_a_ragged_geometry() {
        let row = bf16_row(&patterned_row(MLA_LATENT, 3));
        assert!(rope_row_ref(&row, &[1.0; 32], &[0.0; 32], MLA_LATENT, 0).is_err());
        assert!(rope_row_ref(&row, &[1.0; 32], &[0.0; 32], MLA_LATENT, 7).is_err());
        assert!(rope_row_ref(&row, &[1.0; 31], &[0.0; 31], MLA_LATENT, 64).is_err());
        assert!(rope_row_ref(&row[..511], &[1.0; 32], &[0.0; 32], MLA_LATENT, 64).is_err());
    }

    #[test]
    fn the_append_reference_writes_only_the_positions_slot() {
        let rows = 8usize;
        let head_dim = 512usize;
        let row_values = 2 * head_dim;
        let mut cache: Vec<u16> = (0..rows * row_values)
            .map(|index| (index as u16) | 0x8000)
            .collect();
        let before = cache.clone();
        let k = bf16_row(&patterned_row(head_dim, 41));
        let v = bf16_row(&patterned_row(head_dim, 42));
        kv_append_ref(&mut cache, &k, &v, rows, head_dim, 3).expect("a valid append");
        let base = (3 % rows) * row_values;
        assert_eq!(&cache[base..base + head_dim], &k[..]);
        assert_eq!(&cache[base + head_dim..base + row_values], &v[..]);
        for (index, value) in before.iter().enumerate() {
            if !(base..base + row_values).contains(&index) {
                assert_eq!(cache[index], *value, "value {index} outside the slot moved");
            }
        }
        // The ring: position rows + 3 addresses the same slot and overwrites only it.
        kv_append_ref(&mut cache, &v, &k, rows, head_dim, rows + 3).expect("a wrapped append");
        assert_eq!(&cache[base..base + head_dim], &v[..]);
        assert_eq!(&cache[base + head_dim..base + row_values], &k[..]);
    }

    #[test]
    fn the_append_becomes_visible_to_the_next_positions_window_only() {
        let rows = 8usize;
        let head_dim = 4usize;
        let row_values = 2 * head_dim;
        let mut cache = vec![0u16; rows * row_values];
        let k = vec![bf16_bits(1.5); head_dim];
        let v = vec![bf16_bits(-2.5); head_dim];
        kv_append_ref(&mut cache, &k, &v, rows, head_dim, 5).expect("a valid append");
        // The window read of position 4 (what the loop does BEFORE position 5 appends) cannot see it.
        let earlier = kv_row_ref(&cache, rows, head_dim, 4).expect("a readable row");
        assert_eq!(earlier, vec![0u16; row_values]);
        // Position 5's own row reads back as the k/v pair the next position's window will consume.
        let written = kv_row_ref(&cache, rows, head_dim, 5).expect("a readable row");
        let mut expected = k.clone();
        expected.extend_from_slice(&v);
        assert_eq!(written, expected);
    }

    #[test]
    fn the_append_reference_refuses_a_bad_ring() {
        let mut cache = vec![0u16; 16];
        let k = [0u16; 4];
        let v = [0u16; 4];
        assert!(
            kv_append_ref(&mut cache, &k, &v, 0, 4, 0).is_err(),
            "zero ring"
        );
        assert!(
            kv_append_ref(&mut cache, &k, &v, 2, 0, 0).is_err(),
            "zero width"
        );
        assert!(
            kv_append_ref(&mut cache, &k[..3], &v, 2, 4, 0).is_err(),
            "short k"
        );
        assert!(
            kv_append_ref(&mut cache, &k, &v[..3], 2, 4, 0).is_err(),
            "short v"
        );
        assert!(
            kv_append_ref(&mut cache, &k, &v, 2, 4, 0).is_ok(),
            "2 rows x 8"
        );
        let mut short = vec![0u16; 15];
        assert!(
            kv_append_ref(&mut short, &k, &v, 2, 4, 0).is_err(),
            "short cache"
        );
        assert!(kv_row_ref(&short, 2, 4, 0).is_err(), "short cache read");
    }

    #[test]
    fn the_rope_range_is_the_engine_tables_for_every_position() {
        let config = ModelConfig::v41_flash_cached();
        let max_positions = 200usize;
        for layer in [0usize, 2] {
            let range = rope_tables_range(config, layer, max_positions).expect("a range");
            assert_eq!(range.positions(), max_positions);
            assert_eq!(range.pairs(), config.rope_head_dim / 2);
            for position in [0usize, 1, 37, 127, max_positions - 1] {
                let single = rope_tables_for_layer(config, layer, &[position]);
                assert_eq!(
                    range.cos(position).expect("in range"),
                    single.cos.as_slice(),
                    "layer {layer} position {position}: cos"
                );
                assert_eq!(
                    range.sin(position).expect("in range"),
                    single.sin.as_slice(),
                    "layer {layer} position {position}: sin"
                );
            }
        }
    }

    #[test]
    fn the_two_rope_table_classes_are_the_plain_and_the_compressed_base() {
        let config = ModelConfig::v41_flash_cached();
        assert_eq!(rope_table_class(config, 0).expect("layer 0"), 0);
        let compressed = (0..config.n_layers)
            .find(|layer| config.compress_ratio(*layer).unwrap_or(0) > 0)
            .expect("a compressed layer");
        assert_eq!(rope_table_class(config, compressed).expect("a layer"), 1);
        let plain = rope_tables_range(config, 0, 8).expect("a range");
        let yarn = rope_tables_range(config, compressed, 8).expect("a range");
        assert_ne!(
            plain.cos(1).expect("in range"),
            yarn.cos(1).expect("in range"),
            "the compressed class takes the other base"
        );
        assert_eq!(plain.bytes(), 8 * 32 * 4 * 2);
        assert!(rope_table_class(config, config.n_layers).is_err());
    }

    #[test]
    fn the_rope_range_refuses_a_position_outside_it_and_a_zero_range() {
        let config = ModelConfig::v41_flash_cached();
        let range = rope_tables_range(config, 0, 4).expect("a range");
        let error = range.cos(4).expect_err("out of range");
        assert!(error.contains("outside"), "{error}");
        assert!(range.pair_offset(3).is_ok());
        assert!(rope_tables_range(config, 0, 0).is_err());
        assert!(rope_tables_range(config, config.n_layers, 4).is_err());
    }

    #[test]
    fn the_window_ring_layout_packs_codes_then_scales_without_overlap() {
        let layout = WindowRingLayout::new(128).expect("a ring");
        assert_eq!(layout.rows(), 128);
        assert_eq!(layout.codes_bytes(), 128 * 512);
        assert_eq!(layout.scales_bytes(), 128 * 16);
        assert!(layout.alloc_bytes() >= layout.codes_bytes() + layout.scales_bytes());
        assert_eq!(layout.alloc_bytes() % 256, 0);
        let mut covered = vec![false; layout.codes_bytes() + layout.scales_bytes()];
        for slot in 0..128usize {
            let codes = layout.codes_offset(slot).expect("in range");
            let scales = layout.scales_offset(slot).expect("in range");
            assert_eq!(codes, slot * 512);
            assert_eq!(scales, layout.codes_bytes() + slot * 16);
            for (index, used) in covered.iter_mut().enumerate().skip(codes).take(512) {
                assert!(!*used, "codes overlap at {index}");
                *used = true;
            }
            for (index, used) in covered.iter_mut().enumerate().skip(scales).take(16) {
                assert!(!*used, "scales overlap at {index}");
                *used = true;
            }
        }
        assert!(
            covered.iter().all(|used| *used),
            "every ring byte belongs to one slot"
        );
        // The per-layer and per-run arithmetic the stage and the run record share.
        assert_eq!(layout.ring_bytes(), 67_584);
        assert_eq!(
            layout.ring_bytes(),
            layout.codes_bytes() + layout.scales_bytes()
        );
        assert_eq!(layout.layer_offset(0), 0);
        assert_eq!(layout.layer_offset(2), 2 * 67_584);
        assert_eq!(layout.total_bytes(40).expect("40 layers"), 40 * 67_584);
        assert!(layout.total_bytes(0).is_err());
        // The slot the stage computes is the slot the append kernel writes.
        assert_eq!(layout.slot_for(0), 0);
        assert_eq!(layout.slot_for(128 + 5), 5);
        assert_eq!(layout.slot_for(128 * 3 + 127), 127);
        assert!(WindowRingLayout::new(0).is_err());
        assert!(layout.codes_offset(128).is_err());
    }

    #[test]
    fn the_append_rule_is_the_window_visibility_invariant() {
        assert!(kv_append_rule(0, 0, MLA_LATENT).is_ok());
        assert!(kv_append_rule(37, 37, MLA_LATENT).is_ok());
        // Appending before the window read leaves one extra row: the window would be one too long.
        assert!(kv_append_rule(38, 37, MLA_LATENT).is_err());
        assert!(kv_append_rule(0, 1, MLA_LATENT).is_err());
        assert!(kv_append_rule(0, 0, 511).is_err());
    }

    /// The host sequence the kernel fuses: f32 latent -> rms_norm -> rope tail -> one bf16 cast.
    fn host_sequence(
        latent: &[f32],
        norm: &[f32],
        eps: f32,
        cos: &[f32],
        sin: &[f32],
        rope_head_dim: usize,
    ) -> Vec<u16> {
        let head_width = latent.len();
        let mut row = latent.to_vec();
        linalg::rms_norm(&mut row, norm, head_width, eps).expect("rms_norm");
        let start = head_width - rope_head_dim;
        linalg::rope_rotate(&mut row[start..], cos, sin).expect("rope_rotate");
        row.iter()
            .map(|value| m4_quant::bf16_bits(*value))
            .collect()
    }

    fn sample_row(seed: u32, width: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..width)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let unit = (state >> 8) as f32 / 16_777_216.0_f32;
                (unit - 0.5) * 4.0
            })
            .collect()
    }

    #[test]
    fn the_fused_norm_rope_twin_is_the_host_sequence_on_the_device_operand() {
        let config = ModelConfig::v41_flash_cached();
        let layer = 0usize;
        let position = 7usize;
        let head_width = 512usize;
        let rope_head_dim = 64usize;
        let eps = 1e-20_f32;
        let latent_f32 = sample_row(11, head_width);
        let norm_f32 = sample_row(13, head_width);
        let latent: Vec<u16> = latent_f32.iter().map(|v| m4_quant::bf16_bits(*v)).collect();
        let norm: Vec<u16> = norm_f32.iter().map(|v| m4_quant::bf16_bits(*v)).collect();
        // The host sequence on the SAME operand the device holds: the widened bf16 values.
        let widened_latent: Vec<f32> = latent
            .iter()
            .map(|bits| m4_quant::bf16_value(*bits))
            .collect();
        let widened_norm: Vec<f32> = norm
            .iter()
            .map(|bits| m4_quant::bf16_value(*bits))
            .collect();
        let tables = rope_tables_for_layer(config, layer, &[position]);
        let host = host_sequence(
            &widened_latent,
            &widened_norm,
            eps,
            &tables.cos,
            &tables.sin,
            rope_head_dim,
        );
        let twin = kv_norm_rope_ref(
            &latent,
            &norm,
            eps,
            &tables.cos,
            &tables.sin,
            head_width,
            rope_head_dim,
        )
        .expect("the twin");
        assert_eq!(twin, host, "the twin must be the host sequence bit for bit");
    }

    #[test]
    fn the_f32_latent_deviation_is_measured_not_assumed() {
        let config = ModelConfig::v41_flash_cached();
        let layer = 0usize;
        let position = 3usize;
        let head_width = 512usize;
        let rope_head_dim = 64usize;
        let eps = 1e-20_f32;
        let latent_f32 = sample_row(17, head_width);
        let norm_f32 = sample_row(19, head_width);
        let latent: Vec<u16> = latent_f32.iter().map(|v| m4_quant::bf16_bits(*v)).collect();
        let norm: Vec<u16> = norm_f32.iter().map(|v| m4_quant::bf16_bits(*v)).collect();
        let tables = rope_tables_for_layer(config, layer, &[position]);
        // The host with its f32 latent, against the device twin reading bf16(latent).
        let host = host_sequence(
            &latent_f32,
            &norm_f32,
            eps,
            &tables.cos,
            &tables.sin,
            rope_head_dim,
        );
        let twin = kv_norm_rope_ref(
            &latent,
            &norm,
            eps,
            &tables.cos,
            &tables.sin,
            head_width,
            rope_head_dim,
        )
        .expect("the twin");
        let mut differing = 0usize;
        let mut max_relative = 0.0_f32;
        for (device, host_value) in twin.iter().zip(&host) {
            if device != host_value {
                differing += 1;
            }
            let a = m4_quant::bf16_value(*device).abs();
            let b = m4_quant::bf16_value(*host_value).abs();
            let scale = a.max(b).max(1e-3);
            max_relative = max_relative.max((a - b).abs() / scale);
        }
        // The gap is the bf16 latent operand: it exists (this is not a no-op deviation) and it is a
        // bf16-ulp-scale difference, far inside the D54 arm's declared budget rather than a
        // sequence error. If this ever grows, the sequence changed, not the operand.
        assert!(
            differing > 0,
            "a bf16 latent must differ from the f32 latent somewhere"
        );
        assert!(
            differing < head_width,
            "the deviation must be a few bf16 roundings, not every value"
        );
        assert!(
            max_relative < 0.02,
            "the f32-vs-bf16 latent gap is {max_relative} relative, larger than a bf16 operand explains"
        );
    }

    #[test]
    fn the_fused_norm_rope_twin_refuses_a_bad_row_and_tail() {
        let cos = vec![1.0_f32; 32];
        let sin = vec![0.0_f32; 32];
        let latent = vec![0u16; 512];
        let norm = vec![0u16; 512];
        assert!(kv_norm_rope_ref(&latent, &norm, 1e-20, &cos, &sin, 512, 64).is_ok());
        assert!(kv_norm_rope_ref(&latent[..511], &norm, 1e-20, &cos, &sin, 512, 64).is_err());
        assert!(kv_norm_rope_ref(&latent, &norm, 1e-20, &cos, &sin, 512, 63).is_err());
        assert!(kv_norm_rope_ref(&latent, &norm, 1e-20, &cos[..31], &sin, 512, 64).is_err());
        assert!(kv_norm_rope_ref(&latent, &norm, 1e-20, &cos, &sin, 512, 0).is_err());
    }
}
