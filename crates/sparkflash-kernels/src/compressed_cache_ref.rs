//! The compressed cache in its native device form: the CPU twin of kernels/compressed_append_sm121.cu.
//!
//! WHAT THE STORE IS. write_compressed_cache (engine forward/attention.rs:2943-2992) publishes the
//! owner layer latent as FP4: one 512-wide row per compressed group, bf16-narrowed, then quantised
//! block 16 / element FP4 E2M1 / scale FP8 E4M3 (the E4M3Six rule, attention.rs:241) with 256 code
//! bytes and 32 scale bytes per row. tp4_prefill::stage_attention calls
//! sparkflash_kv::fp4::quantize_main_kv (crates/sparkflash-kv/src/fp4.rs:411) for that row, and this
//! module transcribes the same statement for the device: amax over 16 channels, the scale is the
//! E4M3 round trip of amax.max(6 * 2^-9) / 6, the codes are E2M1 of value / scale clamped to +-6,
//! low nibble first. tests/compressed_cache_rules.rs compares this transcription to the pinned kv
//! quantiser bit for bit, so the twin and the kernel can be argued about separately from the rule.
//!
//! THE APPEND RULE. A row is appended exactly once, when its group completes (write_compressed_cache
//! runs only when state.latent_rows > 0, count = that many rows, first_row = start_pos / ratio):
//!   (a) who writes it: the owner layer of the ratio group writes rows [first_row, first_row + count)
//!       at store row (first_row + i) * ratio, because the body addresses the store by the expanded
//!       (token) position, not by the compressed row (stage_attention :330-352: row = id -
//!       window_rows, expanded = row * ratio);
//!   (b) when it becomes visible: when the row is written. There is no partial row: a group either
//!       has a latent row or it does not, and the append happens once its position is a multiple of
//!       the ratio;
//!   (c) the cursor: rows_written moves forward by count per append, and a rewrite of a row already
//!       visible is refused, not accepted quietly - the cache is append-only.
//! The zero/skip invariant: the store is allocated zeroed once and only complete rows are ever
//! written, so a slot no group has reached keeps its zeros, which is exactly what the body store_zero
//! skip rule tests. No per-call zero fill is needed, and an append of no rows touches no byte.
//!
//! THE CHANNEL BOUND. Capacity in compressed rows is EXPANDED_ROWS / ratio: an append that would
//! write row first_row + i with (first_row + i) * ratio >= EXPANDED_ROWS is refused with the bound
//! named, never grown (the residency rule for the arm).

use core::fmt;

/// The latent width one compressed row carries (head_dim).
pub const DIM: usize = 512;
/// Channels per FP4 scale group.
pub const SCALE_GROUP: usize = 16;
/// Packed FP4 code bytes per row.
pub const CODE_BYTES: usize = DIM / 2;
/// E4M3 scale bytes per row.
pub const SCALE_BYTES: usize = DIM / SCALE_GROUP;
/// Bytes per expanded store row: codes then scales, the layout the body reads.
pub const ROW_BYTES: usize = CODE_BYTES + SCALE_BYTES;
/// Expanded store rows: the body indexes the store by expanded position.
pub const EXPANDED_ROWS: usize = 512;
/// Bytes of one layer expanded store (stage_attention sizes it the same way).
pub const STORE_BYTES: usize = EXPANDED_ROWS * ROW_BYTES;
/// Largest FP4 E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;
/// Largest finite FP8 E4M3 magnitude (OCP e4m3fn).
pub const E4M3_MAX: f32 = 448.0;
/// The smallest E4M3 scale the rule allows: 6 * 2^-9 (kv/src/fp4.rs:209).
pub const AMAX_FLOOR: f32 = 0.011_718_75;
/// The compressed-position bound the indexer channel is allocated for (the select kernel
/// MAX_POSITIONS): a position past it is refused, never grown.
pub const MAX_POSITIONS: usize = 4096;
/// Sentinel the shared index space uses for a query that selected no compressed position.
pub const NO_INDEX: u32 = u32::MAX;

const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decodes one FP4 E2M1 code (kv/src/fp4.rs:44).
#[must_use]
pub fn e2m1_decode(code: u8) -> f32 {
    let magnitude = E2M1_MAGNITUDES[usize::from(code & 0x7)];
    if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Decodes one FP8 E4M3 code (OCP e4m3fn: no infinities, NaN at 0x7F).
#[must_use]
pub fn e4m3_decode(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((code >> 3) & 0x0F);
    let mant = f32::from(code & 0x07);
    let magnitude = if exp == 0 {
        mant * (1.0 / 512.0)
    } else {
        (1.0 + mant / 8.0) * exp2i(exp - 7)
    };
    if exp == 0x0F && mant == 7.0 {
        f32::NAN
    } else {
        sign * magnitude
    }
}

/// 2^n for a small integer n, exactly.
fn exp2i(n: i32) -> f32 {
    if n >= -126 {
        f32::from_bits(((n + 127) as u32) << 23)
    } else {
        (1.0 / 512.0) * f32::from_bits(((n + 9 + 127) as u32) << 23)
    }
}

/// Encodes one f32 to the nearest FP4 E2M1 code, ties to even code (kv/src/fp4.rs:68).
#[must_use]
pub fn e2m1_encode(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    let sign = if value.is_sign_negative() { 0x8u8 } else { 0 };
    let x = value.abs();
    if x >= E2M1_MAX {
        return sign | 0x7;
    }
    let mut best = 0usize;
    let mut best_error = f32::INFINITY;
    for (index, magnitude) in E2M1_MAGNITUDES.iter().enumerate() {
        let error = (x - magnitude).abs();
        if error < best_error || (error == best_error && index % 2 == 0) {
            best = index;
            best_error = error;
        }
    }
    sign | (best as u8)
}

/// Encodes one f32 to the nearest FP8 E4M3 code, ties to even mantissa (kv/src/fp4.rs:115).
#[must_use]
pub fn e4m3_encode(value: f32) -> u8 {
    const MAX_CODE: u8 = 0x7E;
    if value.is_nan() {
        return 0x7F;
    }
    let sign = if value.is_sign_negative() { 0x80u8 } else { 0 };
    let x = value.abs();
    if x.is_infinite() || x >= E4M3_MAX {
        return sign | MAX_CODE;
    }
    let mut low = 0u8;
    let mut high = MAX_CODE;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if e4m3_decode(mid).abs() <= x {
            low = mid
        } else {
            high = mid - 1
        }
    }
    if low < MAX_CODE {
        let up = low + 1;
        let down_error = x - e4m3_decode(low).abs();
        let up_error = e4m3_decode(up).abs() - x;
        if up_error < down_error || (up_error == down_error && up.is_multiple_of(2)) {
            return sign | up;
        }
    }
    sign | low
}

/// One quantised compressed row: the 256 code bytes and 32 scale bytes the body reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Packed FP4 codes, low nibble first.
    pub codes: [u8; CODE_BYTES],
    /// One E4M3 scale per 16 channels.
    pub scales: [u8; SCALE_BYTES],
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            codes: [0u8; CODE_BYTES],
            scales: [0u8; SCALE_BYTES],
        }
    }
}

/// The E4M3 round-trip scale of one group: decode(encode(amax.max(6 * 2^-9) / 6)).
#[must_use]
pub fn group_scale(amax: f32) -> f32 {
    e4m3_decode(e4m3_encode(amax.max(AMAX_FLOOR) / E2M1_MAX))
}

/// Quantises one 512-wide row exactly as the device append does.
///
/// # Errors
/// [CacheError::Shape] when the row is not [DIM] wide.
pub fn quantise_row(values: &[f32]) -> Result<Entry, CacheError> {
    if values.len() != DIM {
        return Err(CacheError::Shape {
            what: "compressed latent row",
            got: values.len(),
            want: DIM,
        });
    }
    let mut entry = Entry::default();
    let mut codes = [0u8; DIM];
    for group in 0..DIM / SCALE_GROUP {
        let block = &values[group * SCALE_GROUP..(group + 1) * SCALE_GROUP];
        let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = group_scale(amax);
        entry.scales[group] = e4m3_encode(amax.max(AMAX_FLOOR) / E2M1_MAX);
        for (offset, value) in block.iter().enumerate() {
            let scaled = (value / scale).clamp(-E2M1_MAX, E2M1_MAX);
            codes[group * SCALE_GROUP + offset] = e2m1_encode(scaled);
        }
    }
    for (index, code) in codes.iter().enumerate() {
        if index % 2 == 0 {
            entry.codes[index / 2] = code & 0x0F;
        } else {
            entry.codes[index / 2] |= (code & 0x0F) << 4;
        }
    }
    Ok(entry)
}

/// Why an append, a plan or a selected-list mapping was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheError {
    /// A buffer has the wrong length.
    Shape {
        /// What was measured.
        what: &'static str,
        /// The length it had.
        got: usize,
        /// The length it needs.
        want: usize,
    },
    /// A position past the allocation bound, named rather than grown.
    Refused {
        /// What was requested.
        what: &'static str,
        /// The request.
        value: usize,
        /// The bound the allocation stated.
        bound: usize,
    },
    /// A row that is already visible was offered again.
    Rewrite {
        /// The first row of the append.
        first_row: usize,
        /// The cursor: how many rows are already published.
        rows_written: usize,
    },
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { what, got, want } => write!(f, "{what} is {got} long, needs {want}"),
            Self::Refused { what, value, bound } => {
                write!(
                    f,
                    "{what} {value} is past the {bound} the allocation stated"
                )
            }
            Self::Rewrite {
                first_row,
                rows_written,
            } => write!(
                f,
                "append from row {first_row} rewrites a cache with {rows_written} rows published",
            ),
        }
    }
}

/// The compressed channel plan: ratio, capacity in rows, and the byte charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    ratio: usize,
    capacity_rows: usize,
}

impl Plan {
    /// Plans the channel for a layer of the given compress ratio.
    ///
    /// # Errors
    /// [CacheError::Refused] for ratio 0 or a ratio whose first expanded row leaves the store.
    pub fn new(ratio: usize) -> Result<Self, CacheError> {
        if ratio == 0 {
            return Err(CacheError::Refused {
                what: "compress ratio",
                value: ratio,
                bound: 1,
            });
        }
        let capacity_rows = EXPANDED_ROWS / ratio;
        if capacity_rows == 0 {
            return Err(CacheError::Refused {
                what: "compress ratio",
                value: ratio,
                bound: EXPANDED_ROWS,
            });
        }
        Ok(Self {
            ratio,
            capacity_rows,
        })
    }

    /// The layer compress ratio.
    #[must_use]
    pub fn ratio(&self) -> usize {
        self.ratio
    }

    /// Compressed rows this channel holds before it refuses.
    #[must_use]
    pub fn capacity_rows(&self) -> usize {
        self.capacity_rows
    }

    /// The store row a compressed row occupies (the expanded index the body reads).
    #[must_use]
    pub fn store_row(&self, compressed_row: usize) -> usize {
        compressed_row * self.ratio
    }

    /// The per-layer byte charge of the channel: the expanded store and its cursor word.
    #[must_use]
    pub fn bytes(&self) -> usize {
        STORE_BYTES + core::mem::size_of::<u32>()
    }
}

/// The device-form store: zeroed at open, append-only, addressed by expanded row.
#[derive(Clone, Debug)]
pub struct Store {
    codes: Vec<u8>,
    scales: Vec<u8>,
    rows_written: usize,
}

impl Store {
    /// Allocates the expanded store zeroed, which is what the absent rows must read as.
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            codes: vec![0u8; EXPANDED_ROWS * CODE_BYTES],
            scales: vec![0u8; EXPANDED_ROWS * SCALE_BYTES],
            rows_written: 0,
        }
    }

    /// Appends completed compressed rows starting at first_row.
    ///
    /// The rows are the owner layer latent rows for the group that just completed; each lands at
    /// store row (first_row + i) * ratio. An empty append is a no-op that touches no byte, which is
    /// the device skip rule. A rewrite, an overrun past [Plan::capacity_rows], or a row whose
    /// expanded index leaves [EXPANDED_ROWS] is refused.
    ///
    /// # Errors
    /// [CacheError::Rewrite], [CacheError::Refused], [CacheError::Shape].
    pub fn append(
        &mut self,
        plan: &Plan,
        first_row: usize,
        rows: &[f32],
    ) -> Result<usize, CacheError> {
        if !rows.len().is_multiple_of(DIM) {
            return Err(CacheError::Shape {
                what: "compressed latent",
                got: rows.len(),
                want: DIM,
            });
        }
        let count = rows.len() / DIM;
        if count == 0 {
            return Ok(0);
        }
        if first_row < self.rows_written {
            return Err(CacheError::Rewrite {
                first_row,
                rows_written: self.rows_written,
            });
        }
        if first_row != self.rows_written {
            return Err(CacheError::Refused {
                what: "compressed first row",
                value: first_row,
                bound: self.rows_written,
            });
        }
        let last = first_row + count;
        if last > plan.capacity_rows() {
            return Err(CacheError::Refused {
                what: "compressed rows",
                value: last,
                bound: plan.capacity_rows(),
            });
        }
        for row in 0..count {
            let store_row = plan.store_row(first_row + row);
            if store_row >= EXPANDED_ROWS {
                return Err(CacheError::Refused {
                    what: "expanded store row",
                    value: store_row,
                    bound: EXPANDED_ROWS,
                });
            }
            let entry = quantise_row(&rows[row * DIM..(row + 1) * DIM])?;
            self.codes[store_row * CODE_BYTES..(store_row + 1) * CODE_BYTES]
                .copy_from_slice(&entry.codes);
            self.scales[store_row * SCALE_BYTES..(store_row + 1) * SCALE_BYTES]
                .copy_from_slice(&entry.scales);
        }
        self.rows_written = last;
        Ok(count)
    }

    /// Compressed rows published so far.
    #[must_use]
    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    /// True while no row has been published: the body store_zero skip rule.
    #[must_use]
    pub fn zero(&self) -> bool {
        self.rows_written == 0
    }

    /// The code bytes and scale bytes of one expanded store row.
    ///
    /// # Errors
    /// [CacheError::Refused] past [EXPANDED_ROWS].
    pub fn row(&self, store_row: usize) -> Result<(&[u8], &[u8]), CacheError> {
        if store_row >= EXPANDED_ROWS {
            return Err(CacheError::Refused {
                what: "expanded store row",
                value: store_row,
                bound: EXPANDED_ROWS,
            });
        }
        let codes = &self.codes[store_row * CODE_BYTES..(store_row + 1) * CODE_BYTES];
        let scales = &self.scales[store_row * SCALE_BYTES..(store_row + 1) * SCALE_BYTES];
        Ok((codes, scales))
    }

    /// Bytes the store occupies; the cursor is the caller word.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len()
    }
}

/// Maps the select output ids to the expanded store row ids the body reads.
///
/// This is the arithmetic stage_attention performs (tp4_prefill.rs:330-352): skip NO_INDEX,
/// subtract window_rows to get the compressed row, multiply by ratio to get the expanded store row,
/// and hand the body the expanded id - not the token position. A value below the window offset or
/// one that expands past [EXPANDED_ROWS] is refused with the bound named.
///
/// # Errors
/// [CacheError::Refused] with the bound the arithmetic ran past.
pub fn selected_rows(
    ids: &[u32],
    window_rows: usize,
    ratio: usize,
) -> Result<Vec<u32>, CacheError> {
    let mut selected = Vec::with_capacity(ids.len());
    for id in ids.iter().copied() {
        if id == NO_INDEX {
            continue;
        }
        let id = id as usize;
        if id < window_rows {
            return Err(CacheError::Refused {
                what: "compressed id",
                value: id,
                bound: window_rows,
            });
        }
        let row = id - window_rows;
        let expanded = row * ratio;
        if expanded >= EXPANDED_ROWS {
            return Err(CacheError::Refused {
                what: "expanded store row",
                value: expanded,
                bound: EXPANDED_ROWS,
            });
        }
        selected.push(expanded as u32);
    }
    Ok(selected)
}

/// Accounting of one layer compressed and index channels, as the residency doc charges them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accounting {
    /// Bytes of the native FP4 expanded store (codes + scales).
    pub store_bytes: usize,
    /// Bytes of the indexer key channel per position (index head_dim f32).
    pub index_key_bytes_per_position: usize,
    /// Compressed positions the indexer channel is allocated for (MAX_POSITIONS).
    pub positions: usize,
    /// Compressed rows the store holds at this ratio.
    pub compressed_rows: usize,
}

impl Accounting {
    /// The charge for a layer of the given ratio and allocation bound.
    ///
    /// # Errors
    /// [CacheError::Refused] for ratio 0 or a ratio below one expanded row per group.
    pub fn new(ratio: usize) -> Result<Self, CacheError> {
        let plan = Plan::new(ratio)?;
        Ok(Self {
            store_bytes: STORE_BYTES,
            index_key_bytes_per_position: 128 * core::mem::size_of::<f32>(),
            positions: MAX_POSITIONS,
            compressed_rows: plan.capacity_rows(),
        })
    }

    /// Total bytes the layer two channels charge.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.store_bytes + self.index_key_bytes_per_position * self.positions
    }
}
