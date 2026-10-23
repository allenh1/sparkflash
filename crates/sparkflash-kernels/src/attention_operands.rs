//! The attention half's operands, stated once: the window a body reads, the store rows that are
//! published, the selected-id space and its expanded store row, the device addresses a stage
//! launches with, and the per-rank wo_a/wo_b shapes.
//!
//! The five stages this module serves (`reference_seam`, `ref_select`, `stage_attention`,
//! `attention_body`, `attention_output` in `crates/sparkflash-cli/src/tp4_forward.rs`) are being
//! wired from refusal stubs to device launches in one pass. Before that pass each stage body would
//! have re-derived its operands from the host twin next to it, and two stages that disagree about
//! the window start, the store cursor or the selected-id space are exactly the failure mode the
//! sequence contract is written against. So the arithmetic is collected here: pure CPU, no device
//! pointer dereferenced, nothing launched or allocated, every failure a named refusal.
//!
//! The rules are composed from the twins that already own them, never restated:
//!
//! * the window span comes from [`crate::attention_stage::window_span`], whose rows include the
//!   current token and whose start is `position + 1 - window_rows`;
//! * [`selected_expanded_rows`] is [`crate::compressed_cache_ref::selected_rows`] behind a
//!   ratio-zero and count refusal, so the expanded row a stage writes is the host twin's row;
//! * the ring slot byte offsets come from [`crate::kv_path_ref::WindowRingLayout`], the same type
//!   the `kv_path` stage writes with (there is one ring layout in the repo, and this module does not
//!   add a second);
//! * the store bound is [`crate::compressed_cache_ref::EXPANDED_ROWS`], the body's union bound is
//!   [`crate::mla_ref::MAX_ATTENDED`], and the indexer pool bound is
//!   [`crate::mla_ref::HSI_BLOCKS`] x [`crate::mla_ref::HSI_BLOCK_TOKENS`].
//!
//! The one rule this crate cannot import is the CLI's `DeviceKvCache` layout (it lives beside the
//! layer arm, in `crates/sparkflash-cli/src/layer_device.rs`). [`cache_layer_bytes`] is that
//! formula, and the test `the_cache_layer_run_is_the_cli_device_kv_cache_formula` reads the CLI
//! source and refuses a drift, so the helper and the allocation it addresses cannot disagree
//! silently.
//!
//! ## What this module deliberately does not do
//!
//! It does not decide *whether* a stage may run: that is the stage's contract in
//! `crates/sparkflash-cli/src/layer_device.rs` and its refusal text. It does not read device memory:
//! a stage that needs to know whether a store row was published derives it from the position here
//! ([`StoreOperands::new`]) instead of reading a cursor word back, because a decode step's row count
//! is a function of the position, not of a round trip.

use crate::compressed_cache_ref::CacheError;
use crate::kv_path_ref::WindowRingLayout;

/// The window rows a body attends over: `config.sliding_window`, the kernel's `SPARK_M4_SWA_WINDOW`.
pub const WINDOW_ROWS: usize = crate::mla_ref::SWA_WINDOW as usize;
/// The most compressed positions one select may emit: `config.index_topk`, the kernel's bound.
pub const MAX_SELECTED: usize = crate::mla_ref::MAX_SELECTED;
/// The body's union bound: window rows plus selected rows, what the sequence entry guards.
pub const MAX_ATTENDED: usize = crate::mla_ref::MAX_ATTENDED;
/// Expanded store rows: the body indexes the store by expanded position, and a store row past this
/// is refused rather than grown.
pub const EXPANDED_ROWS: usize = crate::compressed_cache_ref::EXPANDED_ROWS;
/// Packed FP4 code bytes in the whole expanded store (`EXPANDED_ROWS` rows of `CODE_BYTES`).
pub const STORE_CODE_BYTES: usize = EXPANDED_ROWS * crate::compressed_cache_ref::CODE_BYTES;
/// E4M3 scale bytes in the whole expanded store (`EXPANDED_ROWS` rows of `SCALE_BYTES`).
pub const STORE_SCALE_BYTES: usize = EXPANDED_ROWS * crate::compressed_cache_ref::SCALE_BYTES;
/// One selected id's width.
pub const SELECTED_WORD_BYTES: usize = 4;
/// The alignment of the run allocations a stage addresses: the CLI's `DeviceKvCache::ALIGN` and the
/// ring's `alloc_bytes` rounding are both this number.
pub const CACHE_ALIGN: usize = 256;
/// Tokens per indexer block: the block mask's granularity.
pub const HSI_BLOCK_TOKENS: usize = crate::mla_ref::HSI_BLOCK_TOKENS;
/// Indexer blocks the key pool holds.
pub const HSI_BLOCKS: usize = crate::mla_ref::HSI_BLOCKS;
/// One index head's width.
pub const INDEX_HEAD_DIM: usize = crate::mla_ref::INDEX_HEAD_DIM;
/// The shared index space's "nothing here" value: a selected id that maps to no store row.
pub const NO_INDEX: u32 = crate::compressed_cache_ref::NO_INDEX;

/// Why an operand was refused.
///
/// Every variant names the numbers that disagree, because a stage refusal is the only record of
/// where the arm stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandError {
    /// A ring with no rows has no slot to write or read.
    ZeroRing,
    /// The position's row was never appended: the ring is shorter than the position.
    PositionPastRing {
        /// The 0-based position asked for.
        position: usize,
        /// The rows the ring holds.
        ring_rows: usize,
    },
    /// The ring reports more rows than the run allocated.
    RingPastCapacity {
        /// The rows the ring reports.
        ring_rows: usize,
        /// The rows the allocation has.
        max_positions: usize,
    },
    /// The window's rows are not inside the ring's rows.
    WindowPastRing {
        /// The window's first row.
        window_start: usize,
        /// The window's row count.
        window_rows: usize,
        /// The rows the ring holds.
        ring_rows: usize,
    },
    /// The window would wrap the ring: contiguous codes/scales need `start + rows <= rows`.
    WindowWrapsRing {
        /// The window's first row.
        window_start: usize,
        /// The window's row count.
        window_rows: usize,
        /// The rows the ring holds.
        ring_rows: usize,
    },
    /// A layer outside the run allocation was asked for.
    LayerOutside {
        /// The layer asked for.
        layer: usize,
        /// The layers the allocation carries.
        n_layers: usize,
    },
    /// A cache or ring shape has a zero dimension.
    ZeroShape {
        /// What was zero.
        what: &'static str,
    },
    /// A byte address or offset overflowed the word arithmetic.
    Overflow {
        /// What was being computed.
        what: &'static str,
    },
    /// A window-only layer (`compress_ratio == 0`) has no compressed store to address.
    RatioZero,
    /// The store row count passed the expanded store's row bound.
    RowsPastStore {
        /// The rows the position has published.
        rows: usize,
        /// The store's row bound.
        bound: usize,
    },
    /// More selected ids than the select's top-k bound.
    SelectedPastTopk {
        /// The ids asked for.
        selected: usize,
        /// The bound.
        topk: usize,
    },
    /// Window plus selected rows exceeds the body's union bound.
    AttendedPastBound {
        /// The window rows.
        window_rows: usize,
        /// The selected rows.
        selected: usize,
        /// The bound.
        bound: usize,
    },
    /// A selected id was below the window offset: it cannot name a compressed position.
    SelectedIdBelowOffset {
        /// The id.
        id: u32,
        /// The window offset the id space starts at.
        offset: u32,
    },
    /// A selected id named a compressed position past the select's bound.
    SelectedIdPastPositions {
        /// The id.
        id: u32,
        /// The window offset.
        offset: u32,
        /// The compressed positions the bound allows.
        positions: u32,
    },
    /// A rope tail is zero-wide, odd, or wider than its row.
    BadRopeTail {
        /// The tail width asked for.
        rope_head_dim: usize,
        /// The row width.
        head_width: usize,
    },
    /// A position has no entry in the uploaded rope table.
    PositionPastTable {
        /// The position.
        position: usize,
        /// The positions the table holds.
        positions: usize,
    },
    /// The config's output groups do not divide across the world.
    GroupsNotDivisible {
        /// `config.o_groups`.
        o_groups: usize,
        /// The world size.
        world: usize,
    },
    /// The per-rank heads do not divide into the per-rank output groups.
    HeadsNotDivisible {
        /// `heads_per_rank`.
        heads: usize,
        /// The per-rank groups.
        groups: usize,
    },
    /// A required shape is zero.
    ZeroDimension {
        /// Which dimension.
        what: &'static str,
    },
    /// The plan's wo_a shard is not the per-rank launch's geometry.
    WoAShape {
        /// The plan's `(cols, inner)`.
        got: (usize, usize),
        /// The per-rank `(cols, inner)`.
        want: (usize, usize),
    },
    /// The plan's wo_b shard is not the per-rank launch's geometry.
    WoBShape {
        /// The plan's `(cols, inner)`.
        got: (usize, usize),
        /// The per-rank `(cols, inner)`.
        want: (usize, usize),
    },
    /// The body's rotated row has the wrong byte length for the per-rank geometry.
    ActivationBytes {
        /// The bytes the producer wrote.
        got: usize,
        /// The bytes the per-rank launch reads.
        want: usize,
    },
    /// The body's indexer would read past the index-key pool.
    IndexPastPool {
        /// The blocks the position needs.
        blocks: usize,
        /// The blocks the pool holds.
        bound: usize,
    },
    /// The selected-id mapping's own refusal, kept as its type rather than flattened.
    Cache(CacheError),
}

impl core::fmt::Display for OperandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroRing => write!(f, "a zero-row ring has no slot to read or write"),
            Self::PositionPastRing {
                position,
                ring_rows,
            } => write!(
                f,
                "position {position} needs its row appended first: the ring holds {ring_rows} rows"
            ),
            Self::RingPastCapacity {
                ring_rows,
                max_positions,
            } => write!(
                f,
                "ring reports {ring_rows} rows, the allocation is {max_positions}"
            ),
            Self::WindowPastRing {
                window_start,
                window_rows,
                ring_rows,
            } => write!(
                f,
                "the {window_rows}-row window at {window_start} is outside the ring's {ring_rows} rows"
            ),
            Self::WindowWrapsRing {
                window_start,
                window_rows,
                ring_rows,
            } => write!(
                f,
                "the {window_rows}-row window at {window_start} wraps a ring of {ring_rows} rows; \
                 a contiguous codes/scales read cannot wrap"
            ),
            Self::LayerOutside { layer, n_layers } => {
                write!(
                    f,
                    "layer {layer} of a {n_layers}-layer allocation requested"
                )
            }
            Self::ZeroShape { what } => write!(f, "{what} is zero, so there is no operand"),
            Self::Overflow { what } => write!(f, "{what} overflows the address word"),
            Self::RatioZero => write!(
                f,
                "a window-only layer (compress_ratio 0) has no compressed store"
            ),
            Self::RowsPastStore { rows, bound } => write!(
                f,
                "{rows} published rows are past the expanded store's {bound} rows"
            ),
            Self::SelectedPastTopk { selected, topk } => {
                write!(f, "{selected} selected ids are past the top-k bound {topk}")
            }
            Self::AttendedPastBound {
                window_rows,
                selected,
                bound,
            } => write!(
                f,
                "{window_rows} window rows plus {selected} selected rows are past the body's bound {bound}"
            ),
            Self::SelectedIdBelowOffset { id, offset } => write!(
                f,
                "selected id {id} is below the window offset {offset}, so it names no compressed row"
            ),
            Self::SelectedIdPastPositions {
                id,
                offset,
                positions,
            } => write!(
                f,
                "selected id {id} is {offset} past the window offset, beyond the {positions} positions the id space carries"
            ),
            Self::BadRopeTail {
                rope_head_dim,
                head_width,
            } => write!(
                f,
                "a {rope_head_dim}-wide tail is not an even, positive tail inside a {head_width}-wide row"
            ),
            Self::PositionPastTable {
                position,
                positions,
            } => write!(
                f,
                "position {position} has no rope pair: the table holds {positions} positions"
            ),
            Self::GroupsNotDivisible { o_groups, world } => write!(
                f,
                "{o_groups} output groups do not divide across {world} ranks"
            ),
            Self::HeadsNotDivisible { heads, groups } => write!(
                f,
                "{heads} per-rank heads do not divide into {groups} output groups"
            ),
            Self::ZeroDimension { what } => write!(f, "{what} is zero, so there is no shape"),
            Self::WoAShape { got, want } => write!(
                f,
                "the plan's wo_a is {got:?}, the per-rank launch expects {want:?}"
            ),
            Self::WoBShape { got, want } => write!(
                f,
                "the plan's wo_b is {got:?}, the per-rank launch expects {want:?}"
            ),
            Self::ActivationBytes { got, want } => write!(
                f,
                "the body's rotated row is {got} bytes, the per-rank geometry reads {want}"
            ),
            Self::IndexPastPool { blocks, bound } => write!(
                f,
                "the indexer needs {blocks} blocks, the pool holds {bound}"
            ),
            Self::Cache(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for OperandError {}

impl From<CacheError> for OperandError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

/// The window a body reads at one 0-based position.
///
/// The rows are the ring slots `window_start .. window_start + window_rows`, oldest first. The
/// current token is included: at `position` 0 the window is one row, not zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowOperands {
    /// The 0-based token position.
    pub position: usize,
    /// The rows the ring holds (the run's capacity).
    pub ring_rows: usize,
    /// `min(position + 1, WINDOW_ROWS)`, the rows the body attends.
    pub window_rows: usize,
    /// `position + 1 - window_rows`, the first slot the body reads.
    pub window_start: usize,
    /// `position % ring_rows`, the slot this position's row was written to.
    pub slot: usize,
    /// The offset the select adds to a compressed position: the window's row count, never its
    /// start. `compressed id = window_rows + compressed_position`.
    pub compressed_offset: usize,
}

impl WindowOperands {
    /// The window operands for one position.
    ///
    /// # Errors
    /// [OperandError::ZeroRing], or [OperandError::PositionPastRing] when the position's row has
    /// not been appended yet (`position >= ring_rows`).
    pub fn new(position: usize, ring_rows: usize) -> Result<Self, OperandError> {
        if ring_rows == 0 {
            return Err(OperandError::ZeroRing);
        }
        if position >= ring_rows {
            return Err(OperandError::PositionPastRing {
                position,
                ring_rows,
            });
        }
        let (window_start, window_rows) =
            crate::attention_stage::window_span(position, WINDOW_ROWS);
        Ok(Self {
            position,
            ring_rows,
            window_rows,
            window_start,
            slot: position % ring_rows,
            compressed_offset: window_rows,
        })
    }

    /// The operands with the ring capacity checked against the allocation first.
    ///
    /// # Errors
    /// [OperandError::RingPastCapacity] when the ring reports more rows than the allocation has,
    /// then [Self::new]'s refusals.
    pub fn checked(
        position: usize,
        ring_rows: usize,
        max_positions: usize,
    ) -> Result<Self, OperandError> {
        if ring_rows > max_positions {
            return Err(OperandError::RingPastCapacity {
                ring_rows,
                max_positions,
            });
        }
        Self::new(position, ring_rows)
    }

    /// The body's attended row count: window rows plus the selected rows it also reads.
    ///
    /// # Errors
    /// [OperandError::SelectedPastTopk] and [OperandError::AttendedPastBound], which are the
    /// sequence entry's two guards.
    pub fn attended(&self, selected: usize) -> Result<usize, OperandError> {
        if selected > MAX_SELECTED {
            return Err(OperandError::SelectedPastTopk {
                selected,
                topk: MAX_SELECTED,
            });
        }
        let attended = self.window_rows + selected;
        if attended > MAX_ATTENDED {
            return Err(OperandError::AttendedPastBound {
                window_rows: self.window_rows,
                selected,
                bound: MAX_ATTENDED,
            });
        }
        Ok(attended)
    }
}

/// How many compressed store rows are published after a position, and whether the store is still
/// all zero.
///
/// The device arm has no host mirror of the store, so the count cannot be a read-back. It is a
/// function of the position: at ratio `r`, the group ending at `position` publishes row
/// `(position + 1) / r - 1`, and the number of published rows is `(position + 1) / r`. That is the
/// same number `write_compressed_cache` writes on the host and the same number
/// `attention_stage::assemble` reads as `Store::rows_written`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreOperands {
    /// The layer's compression ratio (`config.compress_ratio(layer)`), at least 1.
    pub ratio: usize,
    /// Compressed rows published by this position: `(position + 1) / ratio`.
    pub rows_written: usize,
    /// Whether no row is published yet, so the body may skip the store read (`store_zero`).
    pub store_zero: bool,
    /// The compressed rows the expanded store can address: `EXPANDED_ROWS / ratio`.
    pub position_capacity: usize,
}

impl StoreOperands {
    /// The published-row operands for one position.
    ///
    /// # Errors
    /// [OperandError::RatioZero] for a window-only layer, and [OperandError::RowsPastStore] when
    /// the position's last published row would expand past the store.
    pub fn new(ratio: usize, position: usize) -> Result<Self, OperandError> {
        if ratio == 0 {
            return Err(OperandError::RatioZero);
        }
        let rows_written = (position + 1) / ratio;
        // The last row's expanded store row is `(rows_written - 1) * ratio`, and it must be inside
        // the expanded store; that is the append kernel's own bound, checked before any launch.
        let expanded_last = rows_written.saturating_sub(1).saturating_mul(ratio);
        if rows_written > 0 && expanded_last >= EXPANDED_ROWS {
            return Err(OperandError::RowsPastStore {
                rows: rows_written,
                bound: EXPANDED_ROWS,
            });
        }
        Ok(Self {
            ratio,
            rows_written,
            store_zero: rows_written == 0,
            position_capacity: EXPANDED_ROWS / ratio,
        })
    }

    /// The compressed positions (select-space) a decode step can reach: `(position + 1) / ratio`.
    ///
    /// This is the select's `n_pos` and, for a decode step, its `reach`: a single query advances
    /// the reachable boundary to `(start_pos + 1) / ratio`, which is exactly the number of published
    /// rows. The two being equal is why a decode select emits no [NO_INDEX] slot and why the shipped
    /// fill's `out_count` is the shipped `keep`; it does not hold for a prefill chunk, where
    /// `n_pos = (start_pos + seqlen) / ratio` is ahead of the per-query reach.
    #[must_use]
    pub fn decode_n_pos(&self) -> usize {
        self.rows_written
    }
}

/// The row a position appends to the compressed store, or `None` while its group is still filling.
///
/// At ratio `r`, the group that ends at `position` is the one with `position % r == r - 1`, and its
/// store row is `position / r`; the append stage passes that row as `compressed_append_sm121`'s
/// `first_row` with `row_count = 1`, and the kernel writes expanded row `first_row * r`. A position
/// that does not end a group publishes nothing, which is the reference's "no latent while a group
/// is still filling".
///
/// # Errors
/// [OperandError::RatioZero], or [OperandError::RowsPastStore] when the row would expand past the
/// store.
pub fn append_row(ratio: usize, position: usize) -> Result<Option<usize>, OperandError> {
    let store = StoreOperands::new(ratio, position)?;
    if !(position + 1).is_multiple_of(ratio) {
        return Ok(None);
    }
    let first_row = position / ratio;
    if first_row >= store.position_capacity {
        return Err(OperandError::RowsPastStore {
            rows: first_row + 1,
            bound: store.position_capacity,
        });
    }
    Ok(Some(first_row))
}

/// The shared index space: the window rows come first, then the compressed positions.
///
/// A select emits ids in this space (`window_rows + compressed_position`, or [NO_INDEX]); a store
/// fill maps each id back to the expanded store row `(id - window_rows) * ratio`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectedIdSpace {
    /// `window_rows`: what a compressed position 0 is called.
    pub offset: u32,
    /// The layer's compression ratio.
    pub ratio: usize,
    /// The select's top-k bound for this launch: `min(config.index_topk, n_pos)`.
    pub topk: usize,
}

impl SelectedIdSpace {
    /// The id space for one launch.
    ///
    /// # Errors
    /// [OperandError::RatioZero], [OperandError::SelectedPastTopk], or
    /// [OperandError::Overflow] when the offset does not fit u32.
    pub fn new(window_rows: usize, ratio: usize, n_pos: usize) -> Result<Self, OperandError> {
        if ratio == 0 {
            return Err(OperandError::RatioZero);
        }
        let offset = u32::try_from(window_rows).map_err(|_| OperandError::Overflow {
            what: "the selected id space's window offset",
        })?;
        let topk = n_pos.min(MAX_SELECTED);
        Ok(Self {
            offset,
            ratio,
            topk,
        })
    }

    /// The id a compressed position is selected as.
    ///
    /// # Errors
    /// [OperandError::SelectedIdPastPositions] when the position is outside the store, and
    /// [OperandError::Overflow] when the id does not fit u32.
    pub fn id(&self, compressed_position: usize) -> Result<u32, OperandError> {
        let capacity = EXPANDED_ROWS / self.ratio;
        if compressed_position >= capacity {
            return Err(OperandError::SelectedIdPastPositions {
                id: u32::try_from(compressed_position).unwrap_or(u32::MAX),
                offset: self.offset,
                positions: u32::try_from(capacity).unwrap_or(u32::MAX),
            });
        }
        u32::try_from(compressed_position)
            .ok()
            .and_then(|position| position.checked_add(self.offset))
            .ok_or(OperandError::Overflow {
                what: "the selected id",
            })
    }

    /// The expanded store row a selected id names, or `None` for [NO_INDEX].
    ///
    /// # Errors
    /// [OperandError::SelectedIdBelowOffset] for an id inside the window rows, and the store bound
    /// refusal of [`from cache::selected_rows`](crate::compressed_cache_ref::selected_rows).
    pub fn expanded_row(&self, id: u32) -> Result<Option<usize>, OperandError> {
        if id == NO_INDEX {
            return Ok(None);
        }
        if id < self.offset {
            return Err(OperandError::SelectedIdBelowOffset {
                id,
                offset: self.offset,
            });
        }
        let row = usize::try_from(id - self.offset).map_err(|_| OperandError::Overflow {
            what: "the selected id's compressed row",
        })?;
        let expanded = row.checked_mul(self.ratio).ok_or(OperandError::Overflow {
            what: "the selected id's expanded row",
        })?;
        if expanded >= EXPANDED_ROWS {
            return Err(OperandError::RowsPastStore {
                rows: row,
                bound: EXPANDED_ROWS / self.ratio,
            });
        }
        Ok(Some(expanded))
    }
}

/// The expanded rows a selected-id list names, in the list's order, skipping [NO_INDEX].
///
/// This *is* [`crate::compressed_cache_ref::selected_rows`], the fill kernel's CPU twin, behind the
/// ratio-zero refusal that twin cannot make (its `ratio` argument is already a plan's) and the
/// top-k bound. Keeping one mapping matters: the fill kernel's output order and the host staging
/// order are the same list, and a second transcription here would be a second rule to test.
///
/// # Errors
/// [OperandError::RatioZero], [OperandError::SelectedPastTopk], or the twin's own refusals.
pub fn selected_expanded_rows(
    ids: &[u32],
    window_rows: usize,
    ratio: usize,
) -> Result<Vec<u32>, OperandError> {
    if ratio == 0 {
        return Err(OperandError::RatioZero);
    }
    if ids.len() > MAX_SELECTED {
        return Err(OperandError::SelectedPastTopk {
            selected: ids.len(),
            topk: MAX_SELECTED,
        });
    }
    Ok(crate::compressed_cache_ref::selected_rows(
        ids,
        window_rows,
        ratio,
    )?)
}

/// The device byte addresses of one window inside the run's quantised ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRingAddresses {
    /// The address of the window's first row's E4M3 codes (`window_rows * 512` bytes contiguous).
    pub codes: u64,
    /// The address of the window's first row's E8M0 scales (`window_rows * 16` bytes contiguous).
    pub scales: u64,
}

/// The ring addresses the body reads for one window.
///
/// The ring is the run's one [`WindowRingLayout`] allocation: `ring_base` plus the layer's
/// `layer_offset`, plus the slot's `codes_offset`/`scales_offset` for `window_start`. The body reads
/// `window_rows` rows contiguously from each: no gather, no per-row upload.
///
/// # Errors
/// [OperandError::LayerOutside] for a layer the run does not carry,
/// [OperandError::WindowPastRing] or [OperandError::WindowWrapsRing] for a window the ring cannot
/// hold contiguously, or [OperandError::Overflow] for an address that does not fit u64.
pub fn window_ring_addresses(
    ring: &WindowRingLayout,
    ring_base: u64,
    n_layers: usize,
    layer: usize,
    window: &WindowOperands,
) -> Result<WindowRingAddresses, OperandError> {
    if layer >= n_layers {
        return Err(OperandError::LayerOutside { layer, n_layers });
    }
    let ring_rows = ring.rows();
    let end =
        window
            .window_start
            .checked_add(window.window_rows)
            .ok_or(OperandError::Overflow {
                what: "the window's last ring row",
            })?;
    if end > ring_rows {
        return Err(OperandError::WindowPastRing {
            window_start: window.window_start,
            window_rows: window.window_rows,
            ring_rows,
        });
    }
    // A window that does not wrap is contiguous: the ring covers exactly the run's positions, so
    // `window_start + window_rows <= ring_rows` is the condition, already checked above. The slot
    // and the row index coincide here; the check below is the one that would fire if a future ring
    // were shorter than the run.
    let slot = ring.slot_for(window.window_start);
    if slot
        .checked_add(window.window_rows)
        .is_none_or(|end| end > ring_rows)
    {
        return Err(OperandError::WindowWrapsRing {
            window_start: window.window_start,
            window_rows: window.window_rows,
            ring_rows,
        });
    }
    let layer_base = ring_base
        .checked_add(ring.layer_offset(layer) as u64)
        .ok_or(OperandError::Overflow {
            what: "the window ring's layer base",
        })?;
    let codes_offset = ring
        .codes_offset(slot)
        .map_err(|_| OperandError::WindowWrapsRing {
            window_start: window.window_start,
            window_rows: window.window_rows,
            ring_rows,
        })?;
    let scales_offset = ring
        .scales_offset(slot)
        .map_err(|_| OperandError::WindowWrapsRing {
            window_start: window.window_start,
            window_rows: window.window_rows,
            ring_rows,
        })?;
    Ok(WindowRingAddresses {
        codes: layer_base + codes_offset as u64,
        scales: layer_base + scales_offset as u64,
    })
}

/// The bytes one layer's `DeviceKvCache` row run occupies.
///
/// This is `crates/sparkflash-cli/src/layer_device.rs`'s `DeviceKvCache::layer_bytes_for`, which
/// the kernels crate cannot import: `align_up(capacity_rows * row_values * 2, align)`. The test
/// `the_cache_layer_run_is_the_cli_device_kv_cache_formula` reads that source and refuses a drift.
///
/// # Errors
/// [OperandError::ZeroShape] for a zero capacity or row width.
pub fn cache_layer_bytes(
    capacity_rows: usize,
    row_values: usize,
    align: usize,
) -> Result<usize, OperandError> {
    if capacity_rows == 0 {
        return Err(OperandError::ZeroShape {
            what: "the cache's capacity_rows",
        });
    }
    if row_values == 0 {
        return Err(OperandError::ZeroShape {
            what: "the cache's row_values",
        });
    }
    if align == 0 {
        return Err(OperandError::ZeroShape {
            what: "the cache's alignment",
        });
    }
    let raw = capacity_rows
        .checked_mul(row_values)
        .and_then(|values| values.checked_mul(2))
        .ok_or(OperandError::Overflow {
            what: "the cache layer's row bytes",
        })?;
    raw.checked_next_multiple_of(align)
        .ok_or(OperandError::Overflow {
            what: "the cache layer's aligned row bytes",
        })
}

/// The device address of one layer's cache run, and that run's byte length.
///
/// # Errors
/// [OperandError::LayerOutside], or the refusals of [`cache_layer_bytes`].
pub fn cache_layer_base(
    cache_base: u64,
    n_layers: usize,
    layer: usize,
    capacity_rows: usize,
    row_values: usize,
    align: usize,
) -> Result<(u64, usize), OperandError> {
    if layer >= n_layers {
        return Err(OperandError::LayerOutside { layer, n_layers });
    }
    let layer_bytes = cache_layer_bytes(capacity_rows, row_values, align)?;
    let offset = (layer as u64)
        .checked_mul(layer_bytes as u64)
        .ok_or(OperandError::Overflow {
            what: "the cache layer base",
        })?;
    let base = cache_base
        .checked_add(offset)
        .ok_or(OperandError::Overflow {
            what: "the cache layer base",
        })?;
    Ok((base, layer_bytes))
}

/// One position's address inside that layer run: `slot = position % capacity_rows`, and the row is
/// `2 * head_dim` bf16, k then v.
///
/// The attention body does not read this run (it reads the quantised window ring); the `kv_path`
/// stage writes it so the cache stays the run's k/v record. The helper is here because the two
/// stages must agree about which slot a position owns, and `kv_append_rule` states that rule for
/// the append while this states the address the same rule produces.
///
/// # Errors
/// [OperandError::ZeroShape] for a zero capacity, or [OperandError::PositionPastRing] when the
/// position would wrap past the run's capacity.
pub fn cache_position_address(
    layer_base: u64,
    capacity_rows: usize,
    row_values: usize,
    position: usize,
) -> Result<(u64, usize), OperandError> {
    if capacity_rows == 0 {
        return Err(OperandError::ZeroShape {
            what: "the cache's capacity_rows",
        });
    }
    if row_values == 0 {
        return Err(OperandError::ZeroShape {
            what: "the cache's row_values",
        });
    }
    if position >= capacity_rows {
        return Err(OperandError::PositionPastRing {
            position,
            ring_rows: capacity_rows,
        });
    }
    let row_bytes = row_values.checked_mul(2).ok_or(OperandError::Overflow {
        what: "the cache row's bytes",
    })?;
    let offset = (position as u64)
        .checked_mul(row_bytes as u64)
        .ok_or(OperandError::Overflow {
            what: "the cache row's address",
        })?;
    let address = layer_base
        .checked_add(offset)
        .ok_or(OperandError::Overflow {
            what: "the cache row's address",
        })?;
    Ok((address, row_bytes))
}

/// The device addresses of one position's rope pair inside an uploaded table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeTableAddresses {
    /// The position's cos pair: `rope_head_dim / 2` f32 values.
    pub cos: u64,
    /// The position's sin pair: `rope_head_dim / 2` f32 values.
    pub sin: u64,
    /// The bytes one position's pair occupies.
    pub position_bytes: usize,
    /// The bytes one whole table (cos rows then sin rows) occupies.
    pub table_bytes: usize,
}

/// The addresses a rope launch reads for one position.
///
/// The table is the run's one upload per rope class: all positions' cos pairs for
/// `max_positions`, then all positions' sin pairs. The layout is the CLI open's upload
/// (`for position in 0..positions { cos }` then `for position in 0..positions { sin }`), and a stage
/// passes `(cos, sin)` already offset here, which is what [`crate::layer_seams_launch::RopeRowsParams`]
/// documents.
///
/// # Errors
/// [OperandError::BadRopeTail] for a zero/odd tail wider than the row,
/// [OperandError::PositionPastTable], or [OperandError::Overflow].
pub fn rope_table_addresses(
    table_base: u64,
    positions: usize,
    rope_head_dim: usize,
    head_width: usize,
) -> Result<RopeTableAddresses, OperandError> {
    let tail_ok =
        rope_head_dim != 0 && rope_head_dim <= head_width && rope_head_dim.is_multiple_of(2);
    if !tail_ok {
        return Err(OperandError::BadRopeTail {
            rope_head_dim,
            head_width,
        });
    }
    if positions == 0 {
        return Err(OperandError::ZeroDimension {
            what: "the rope table's positions",
        });
    }
    let pairs = rope_head_dim / 2;
    let position_bytes = pairs.checked_mul(4).ok_or(OperandError::Overflow {
        what: "the rope position's bytes",
    })?;
    let table_bytes = positions
        .checked_mul(position_bytes)
        .ok_or(OperandError::Overflow {
            what: "the rope table's bytes",
        })?;
    let sin_base = table_base
        .checked_add(table_bytes as u64)
        .ok_or(OperandError::Overflow {
            what: "the rope table's sin base",
        })?;
    Ok(RopeTableAddresses {
        cos: table_base,
        sin: sin_base,
        position_bytes,
        table_bytes,
    })
}

impl RopeTableAddresses {
    /// The cos and sin addresses for one position, refusing a position the table does not hold.
    ///
    /// # Errors
    /// [OperandError::PositionPastTable], or [OperandError::Overflow].
    pub fn at(&self, position: usize, positions: usize) -> Result<(u64, u64), OperandError> {
        if position >= positions {
            return Err(OperandError::PositionPastTable {
                position,
                positions,
            });
        }
        let offset = (position as u64)
            .checked_mul(self.position_bytes as u64)
            .ok_or(OperandError::Overflow {
                what: "the rope position's address",
            })?;
        let cos = self.cos.checked_add(offset).ok_or(OperandError::Overflow {
            what: "the rope position's cos address",
        })?;
        let sin = self.sin.checked_add(offset).ok_or(OperandError::Overflow {
            what: "the rope position's sin address",
        })?;
        Ok((cos, sin))
    }
}

/// The per-rank geometry of the grouped wo_a output projection and the wo_b combine after it.
///
/// At world `w`, `groups = o_groups / w`; the per-rank heads `heads_per_rank = n_heads / w` split
/// into those `groups` groups, so `group_inner = (heads_per_rank / groups) * head_width`. The plan's
/// wo_a shard is `(groups * o_lora_rank, group_inner)` and wo_b's is
/// `(hidden, groups * o_lora_rank)`. At the released geometry and world 4 that is `groups = 2`,
/// `group_inner = 4,096`, wo_a `(2,048, 4,096)`, wo_b `(5,120, 2,048)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputGeometry {
    /// `config.o_groups`: the global group count.
    pub o_groups: usize,
    /// The world size.
    pub world: usize,
    /// The per-rank group count.
    pub groups: usize,
    /// The per-rank query head count.
    pub heads_per_rank: usize,
    /// One head row's width (`config.head_dim`).
    pub head_width: usize,
    /// The per-rank heads in one group.
    pub heads_per_group: usize,
    /// One group's activation width: `heads_per_group * head_width`.
    pub group_inner: usize,
    /// `config.o_lora_rank`.
    pub o_lora_rank: usize,
    /// `config.dim`: the output row's width.
    pub hidden: usize,
}

impl OutputGeometry {
    /// The per-rank geometry.
    ///
    /// # Errors
    /// A zero world, group count or head count; groups that do not divide across the world; heads
    /// that do not divide into the per-rank groups; a zero width, rank or hidden size.
    pub fn new(
        o_groups: usize,
        world: usize,
        heads_per_rank: usize,
        head_width: usize,
        o_lora_rank: usize,
        hidden: usize,
    ) -> Result<Self, OperandError> {
        if world == 0 {
            return Err(OperandError::ZeroDimension { what: "the world" });
        }
        if o_groups == 0 {
            return Err(OperandError::ZeroDimension {
                what: "config.o_groups",
            });
        }
        if !o_groups.is_multiple_of(world) {
            return Err(OperandError::GroupsNotDivisible { o_groups, world });
        }
        let groups = o_groups / world;
        if heads_per_rank == 0 {
            return Err(OperandError::ZeroDimension {
                what: "heads_per_rank",
            });
        }
        if !heads_per_rank.is_multiple_of(groups) {
            return Err(OperandError::HeadsNotDivisible {
                heads: heads_per_rank,
                groups,
            });
        }
        for (what, value) in [
            ("head_width", head_width),
            ("o_lora_rank", o_lora_rank),
            ("hidden", hidden),
        ] {
            if value == 0 {
                return Err(OperandError::ZeroDimension { what });
            }
        }
        let heads_per_group = heads_per_rank / groups;
        let group_inner =
            heads_per_group
                .checked_mul(head_width)
                .ok_or(OperandError::Overflow {
                    what: "the per-rank group's inner width",
                })?;
        Ok(Self {
            o_groups,
            world,
            groups,
            heads_per_rank,
            head_width,
            heads_per_group,
            group_inner,
            o_lora_rank,
            hidden,
        })
    }

    /// The plan's wo_a shard shape `(cols, inner)` for this rank.
    #[must_use]
    pub fn wo_a_shape(&self) -> (usize, usize) {
        (self.groups * self.o_lora_rank, self.group_inner)
    }

    /// The plan's wo_b shard shape `(cols, inner)` for this rank.
    #[must_use]
    pub fn wo_b_shape(&self) -> (usize, usize) {
        (self.hidden, self.groups * self.o_lora_rank)
    }

    /// The body's rotated row: one head row per per-rank head, bf16.
    #[must_use]
    pub fn activation_bytes(&self) -> usize {
        self.heads_per_rank * self.head_width * 2
    }

    /// The wo_a output row: `groups * o_lora_rank` bf16 values.
    #[must_use]
    pub fn lo_bytes(&self) -> usize {
        self.groups * self.o_lora_rank * 2
    }

    /// The wo_b activation's code bytes: one E4M3 code per `lo` value.
    #[must_use]
    pub fn lo_code_bytes(&self) -> usize {
        self.groups * self.o_lora_rank
    }

    /// The wo_b activation's scale bytes: one E8M0 scale per 32 codes.
    #[must_use]
    pub fn lo_scale_bytes(&self) -> usize {
        (self.groups * self.o_lora_rank).div_ceil(32)
    }

    /// The combined output row: `hidden` bf16 values, what `all_reduce` reduces.
    #[must_use]
    pub fn output_bytes(&self) -> usize {
        self.hidden * 2
    }

    /// The `prologue_bf16` launch's `cols`: the wo_a output width it quantises.
    #[must_use]
    pub fn prologue_cols(&self) -> usize {
        self.groups * self.o_lora_rank
    }

    /// Checks the plan's wo_a shard against the per-rank launch geometry.
    ///
    /// # Errors
    /// [OperandError::WoAShape] naming both shapes.
    pub fn check_wo_a(&self, got: (usize, usize)) -> Result<(), OperandError> {
        let want = self.wo_a_shape();
        if got != want {
            return Err(OperandError::WoAShape { got, want });
        }
        Ok(())
    }

    /// Checks the plan's wo_b shard against the per-rank launch geometry.
    ///
    /// # Errors
    /// [OperandError::WoBShape] naming both shapes.
    pub fn check_wo_b(&self, got: (usize, usize)) -> Result<(), OperandError> {
        let want = self.wo_b_shape();
        if got != want {
            return Err(OperandError::WoBShape { got, want });
        }
        Ok(())
    }

    /// Checks the body's rotated row byte length against the per-rank launch geometry.
    ///
    /// # Errors
    /// [OperandError::ActivationBytes] naming both lengths.
    pub fn check_activation_bytes(&self, got: usize) -> Result<(), OperandError> {
        let want = self.activation_bytes();
        if got != want {
            return Err(OperandError::ActivationBytes { got, want });
        }
        Ok(())
    }
}

/// The index-key bytes the body's indexer stage reads at one position: whole blocks, so the last
/// block is read past the query but never past the pool.
///
/// # Errors
/// [OperandError::IndexPastPool] when the position needs more blocks than the pool holds.
pub fn index_key_block_bytes(position: usize) -> Result<usize, OperandError> {
    let tokens = position.checked_add(1).ok_or(OperandError::Overflow {
        what: "the indexer's token count",
    })?;
    let blocks = tokens.div_ceil(HSI_BLOCK_TOKENS);
    if blocks > HSI_BLOCKS {
        return Err(OperandError::IndexPastPool {
            blocks,
            bound: HSI_BLOCKS,
        });
    }
    blocks
        .checked_mul(HSI_BLOCK_TOKENS)
        .and_then(|tokens| tokens.checked_mul(INDEX_HEAD_DIM))
        .and_then(|values| values.checked_mul(2))
        .ok_or(OperandError::Overflow {
            what: "the indexer's key bytes",
        })
}

/// The index-key pool's whole byte size: what the arm allocates and zero-fills at open while no
/// device producer writes it.
#[must_use]
pub fn index_key_pool_bytes() -> usize {
    HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM * 2
}

/// The bytes one select's id list occupies.
///
/// # Errors
/// [OperandError::SelectedPastTopk] for a count past the select's bound.
pub fn selected_bytes(count: usize) -> Result<usize, OperandError> {
    if count > MAX_SELECTED {
        return Err(OperandError::SelectedPastTopk {
            selected: count,
            topk: MAX_SELECTED,
        });
    }
    count
        .checked_mul(SELECTED_WORD_BYTES)
        .ok_or(OperandError::Overflow {
            what: "the selected id list's bytes",
        })
}

/// The candidate block mask's bytes: one word per 32 positions, `MAX_POSITIONS / 32` words.
#[must_use]
pub fn candidate_mask_bytes() -> usize {
    crate::ref_select_launch::MASK_WORDS * SELECTED_WORD_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released per-rank geometry at world 4, as the config and the plan state it.
    fn geometry() -> OutputGeometry {
        OutputGeometry::new(8, 4, 16, 512, 1024, 5120).expect("released geometry")
    }

    #[test]
    fn the_window_includes_the_current_token_at_every_boundary() {
        // position 0 is one row, not zero: the current token is in its own window.
        let first = WindowOperands::new(0, 2048).expect("position 0");
        assert_eq!((first.window_rows, first.window_start), (1, 0));
        assert_eq!(first.slot, 0);
        // The 128th row fills the window without sliding it.
        let full = WindowOperands::new(127, 2048).expect("position 127");
        assert_eq!((full.window_rows, full.window_start), (128, 0));
        // One past the window, the start slides by one and the count stays at the window size.
        let sliding = WindowOperands::new(128, 2048).expect("position 128");
        assert_eq!((sliding.window_rows, sliding.window_start), (128, 1));
        let late = WindowOperands::new(4095, 4096).expect("last position");
        assert_eq!((late.window_rows, late.window_start), (128, 3968));
        assert_eq!(late.slot, 4095);
    }

    #[test]
    fn the_compressed_offset_is_the_row_count_not_the_start() {
        // The trap the sequence contract records: ids are offset by the window's *width*.
        for position in [1usize, 2, 127, 128, 300] {
            let window = WindowOperands::new(position, 4096).expect("in range");
            assert_eq!(window.compressed_offset, window.window_rows);
            assert_ne!(window.compressed_offset, window.window_start);
        }
        let late = WindowOperands::new(400, 4096).expect("in range");
        assert_eq!((late.window_rows, late.window_start), (128, 273));
        assert_eq!(late.compressed_offset, 128);
    }

    #[test]
    fn a_position_past_the_ring_is_refused_before_any_operand_exists() {
        // A 5-row ring holds positions 0..=4; position 5's row is the first one not appended.
        assert!(WindowOperands::new(4, 5).is_ok());
        let error = WindowOperands::new(5, 5).expect_err("the row was not appended");
        assert_eq!(
            error,
            OperandError::PositionPastRing {
                position: 5,
                ring_rows: 5
            }
        );
        assert_eq!(
            WindowOperands::new(0, 0).expect_err("empty ring"),
            OperandError::ZeroRing
        );
    }

    #[test]
    fn the_ring_capacity_is_checked_against_the_allocation_before_the_position() {
        assert_eq!(
            WindowOperands::checked(0, 8, 4).expect_err("ring past capacity"),
            OperandError::RingPastCapacity {
                ring_rows: 8,
                max_positions: 4
            }
        );
        assert!(WindowOperands::checked(3, 4, 4).is_ok());
    }

    #[test]
    fn the_body_union_bound_is_the_sequence_entry_guard() {
        let window = WindowOperands::new(300, 4096).expect("in range");
        assert_eq!(window.attended(512).expect("full union"), 640);
        assert_eq!(window.attended(0).expect("window only"), 128);
        assert_eq!(
            window.attended(513).expect_err("past top-k"),
            OperandError::SelectedPastTopk {
                selected: 513,
                topk: MAX_SELECTED
            }
        );
        // A short window leaves room under the same bound.
        let first = WindowOperands::new(0, 4096).expect("position 0");
        assert_eq!(first.attended(512).expect("1 + 512"), 513);
    }

    #[test]
    fn the_store_row_count_is_the_host_twins_cursor() {
        // ratio 2: the first row is published by the position that completes the pair.
        for (position, rows, zero) in [
            (0usize, 0usize, true),
            (1, 1, false),
            (2, 1, false),
            (3, 2, false),
            (4, 2, false),
            (5, 3, false),
        ] {
            let store = StoreOperands::new(2, position).expect("in store");
            assert_eq!(store.rows_written, rows, "position {position}");
            assert_eq!(store.store_zero, zero, "position {position}");
        }
        // ratio 1: every position publishes its own row.
        let one = StoreOperands::new(1, 300).expect("in store");
        assert_eq!(one.rows_written, 301);
        assert!(!one.store_zero);
        assert_eq!(one.position_capacity, EXPANDED_ROWS);
    }

    #[test]
    fn a_store_row_past_the_expanded_store_is_refused_rather_than_grown() {
        // ratio 2 has 256 compressed rows, so the last one expands to 510, inside 512.
        assert!(StoreOperands::new(2, 511).is_ok());
        assert_eq!(
            StoreOperands::new(2, 513).expect_err("past the store"),
            OperandError::RowsPastStore {
                rows: 257,
                bound: EXPANDED_ROWS
            }
        );
        // ratio 1 has 512 rows: position 511 is the last one that fits.
        assert!(StoreOperands::new(1, 511).is_ok());
        assert!(StoreOperands::new(1, 512).is_err());
        assert_eq!(
            StoreOperands::new(0, 0).expect_err("window-only layer"),
            OperandError::RatioZero
        );
    }

    #[test]
    fn the_append_row_is_the_group_the_position_completes() {
        // ratio 2: position 1 completes the first pair, position 3 the second; the odd positions
        // behind them publish nothing.
        assert_eq!(append_row(2, 0).expect("position 0"), None);
        assert_eq!(append_row(2, 1).expect("position 1"), Some(0));
        assert_eq!(append_row(2, 2).expect("position 2"), None);
        assert_eq!(append_row(2, 3).expect("position 3"), Some(1));
        // ratio 1: every position is its own group.
        for position in [0usize, 1, 300, 511] {
            assert_eq!(append_row(1, position).expect("in store"), Some(position));
        }
        assert_eq!(
            append_row(0, 0).expect_err("window-only"),
            OperandError::RatioZero
        );
        // The last ratio-1 position is the store's last row; one past it is refused, not grown.
        assert_eq!(
            append_row(1, 512).expect_err("past the store"),
            OperandError::RowsPastStore {
                rows: 513,
                bound: EXPANDED_ROWS
            }
        );
    }

    #[test]
    fn a_decode_select_sees_every_published_row_and_no_more() {
        // n_pos == reach for one query, which is why a decode select emits no NO_INDEX slot.
        for (ratio, position) in [(1usize, 0usize), (1, 300), (2, 0), (2, 1), (2, 511)] {
            let store = StoreOperands::new(ratio, position).expect("in store");
            assert_eq!(store.decode_n_pos(), (position + 1) / ratio);
            assert_eq!(store.decode_n_pos(), store.rows_written);
        }
    }

    #[test]
    fn the_selected_id_space_is_window_rows_plus_the_position() {
        let space = SelectedIdSpace::new(128, 2, 300).expect("ratio 2");
        assert_eq!(space.topk, 300);
        assert_eq!(space.id(0).expect("position 0"), 128);
        assert_eq!(space.id(5).expect("position 5"), 133);
        // A full top-k list caps at the select's bound.
        let full = SelectedIdSpace::new(128, 2, 4096).expect("ratio 2");
        assert_eq!(full.topk, MAX_SELECTED);
        // An id below the offset names no compressed row.
        assert_eq!(
            space.expanded_row(127).expect_err("inside the window"),
            OperandError::SelectedIdBelowOffset {
                id: 127,
                offset: 128
            }
        );
        assert_eq!(space.expanded_row(NO_INDEX).expect("sentinel"), None);
        // The expanded row is (id - window_rows) * ratio.
        assert_eq!(space.expanded_row(128).expect("row 0"), Some(0));
        assert_eq!(space.expanded_row(133).expect("row 5"), Some(10));
        // Past the store: ratio 2 addresses 256 compressed rows, ids 128..=383.
        assert_eq!(
            space.expanded_row(384).expect_err("row 256"),
            OperandError::RowsPastStore {
                rows: 256,
                bound: 256
            }
        );
    }

    #[test]
    fn the_id_space_agrees_with_the_fill_kernels_cpu_twin() {
        // The module does not restate the mapping: selected_expanded_rows *is* the twin. Pin the
        // agreement on the mixed list a real launch can produce (a NO_INDEX slot after a valid one).
        let ids = [128u32, 130, NO_INDEX, 131];
        assert_eq!(
            selected_expanded_rows(&ids, 128, 2).expect("in store"),
            crate::compressed_cache_ref::selected_rows(&ids, 128, 2).expect("twin")
        );
        assert_eq!(
            selected_expanded_rows(&ids, 128, 2).expect("in store"),
            vec![0, 4, 6]
        );
        assert_eq!(
            selected_expanded_rows(&[], 128, 2).expect("empty list"),
            Vec::<u32>::new()
        );
        assert_eq!(
            selected_expanded_rows(&ids, 128, 0).expect_err("window-only layer"),
            OperandError::RatioZero
        );
        let too_many = vec![128u32; MAX_SELECTED + 1];
        assert!(selected_expanded_rows(&too_many, 128, 2).is_err());
    }

    #[test]
    fn the_window_ring_addresses_are_the_slot_offsets_the_kv_path_stage_writes() {
        // The ring the CLI allocates at open: one layout for the run's capacity.
        let ring = WindowRingLayout::new(2048).expect("ring");
        // layer 3, slot 0: the codes array starts at the layer base, scales at codes_bytes().
        let first = WindowOperands::new(0, 2048).expect("position 0");
        let addresses = window_ring_addresses(&ring, 0x10_0000, 40, 3, &first).expect("addresses");
        let layer_base = 0x10_0000 + (3 * ring.alloc_bytes()) as u64;
        assert_eq!(addresses.codes, layer_base);
        assert_eq!(addresses.scales, layer_base + ring.codes_bytes() as u64);
        // slot 128: codes at 128 * 512, scales at codes_bytes + 128 * 16.
        let late = WindowOperands::new(300, 2048).expect("position 300");
        let addresses = window_ring_addresses(&ring, 0x10_0000, 40, 3, &late).expect("addresses");
        assert_eq!(
            addresses.codes,
            layer_base + (late.window_start * crate::kv_path_ref::WINDOW_CODE_BYTES) as u64
        );
        assert_eq!(
            addresses.scales,
            layer_base
                + ring.codes_bytes() as u64
                + (late.window_start * crate::kv_path_ref::WINDOW_SCALE_BYTES) as u64
        );
        // A layer the run does not carry, and a window the ring cannot hold.
        assert_eq!(
            window_ring_addresses(&ring, 0, 40, 40, &first).expect_err("layer 40"),
            OperandError::LayerOutside {
                layer: 40,
                n_layers: 40
            }
        );
    }

    #[test]
    fn the_window_ring_address_arithmetic_is_the_layouts_own() {
        let ring = WindowRingLayout::new(2048).expect("ring");
        for position in [0usize, 1, 127, 128, 500, 2047] {
            let window = WindowOperands::new(position, 2048).expect("in ring");
            let addresses =
                window_ring_addresses(&ring, 0x20_0000, 40, 7, &window).expect("addresses");
            let slot = ring.slot_for(window.window_start);
            assert_eq!(
                addresses.codes,
                0x20_0000 + (ring.layer_offset(7) + ring.codes_offset(slot).expect("slot")) as u64
            );
            assert_eq!(
                addresses.scales,
                0x20_0000 + (ring.layer_offset(7) + ring.scales_offset(slot).expect("slot")) as u64
            );
            // The window is contiguous: each row's codes are 512 bytes apart.
            assert_eq!(
                ring.codes_offset(slot + 1).expect("next slot")
                    - ring.codes_offset(slot).expect("slot"),
                crate::kv_path_ref::WINDOW_CODE_BYTES
            );
        }
    }

    #[test]
    fn the_rope_table_offsets_are_the_open_uploads_layout() {
        let table = rope_table_addresses(0x30_0000, 2048, 64, 512).expect("table");
        assert_eq!(table.position_bytes, 32 * 4);
        assert_eq!(table.table_bytes, 2048 * 32 * 4);
        assert_eq!(
            table.at(0, 2048).expect("position 0"),
            (0x30_0000, 0x30_0000 + table.table_bytes as u64)
        );
        let (cos, sin) = table.at(400, 2048).expect("position 400");
        assert_eq!(cos, 0x30_0000 + 400 * 128);
        assert_eq!(sin, 0x30_0000 + table.table_bytes as u64 + 400 * 128);
        assert_eq!(
            table.at(2048, 2048).expect_err("past the table"),
            OperandError::PositionPastTable {
                position: 2048,
                positions: 2048
            }
        );
        assert!(rope_table_addresses(0, 0, 64, 512).is_err());
        assert_eq!(
            rope_table_addresses(0, 2048, 63, 512).expect_err("odd tail"),
            OperandError::BadRopeTail {
                rope_head_dim: 63,
                head_width: 512
            }
        );
        assert!(rope_table_addresses(0, 2048, 64, 32).is_err());
    }

    #[test]
    fn the_cache_layer_run_is_the_cli_device_kv_cache_formula() {
        // The kernels crate cannot import the CLI's DeviceKvCache, so the formula is restated and
        // the CLI source is checked instead: an edit to either side fails here.
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sparkflash-cli/src/layer_device.rs"
        ))
        .expect("the CLI is in the tree");
        assert!(
            source.contains("pub const ALIGN: usize = 256;"),
            "the cache alignment moved; attention_operands::CACHE_ALIGN is stale"
        );
        assert!(
            source.contains("align_up(capacity_rows.saturating_mul(row_values).saturating_mul(2))"),
            "DeviceKvCache::layer_bytes_for no longer rounds capacity_rows * row_values * 2"
        );
        // The released run: 2048 rows of 1024 bf16 values (k and v for 512 head_dim).
        assert_eq!(
            cache_layer_bytes(2048, 1024, CACHE_ALIGN).expect("layer bytes"),
            2048 * 1024 * 2
        );
        // A row count that is not alignment-divisible rounds up per layer, never the count.
        assert_eq!(
            cache_layer_bytes(100, 3, CACHE_ALIGN).expect("rounded layer"),
            768
        );
        // The formula is DeviceKvCache::layer_bytes_for: the CLI's owns it, this mirrors it.
        let layer = cache_layer_bytes(2048, 1024, CACHE_ALIGN).expect("layer bytes");
        let (base, bytes) =
            cache_layer_base(0x40_0000, 40, 0, 2048, 1024, CACHE_ALIGN).expect("layer 0");
        assert_eq!((base, bytes), (0x40_0000, layer));
        let (base, _) =
            cache_layer_base(0x40_0000, 40, 3, 2048, 1024, CACHE_ALIGN).expect("layer 3");
        assert_eq!(base, 0x40_0000 + 3 * layer as u64);
        assert_eq!(
            cache_layer_base(0, 40, 40, 2048, 1024, CACHE_ALIGN).expect_err("layer 40"),
            OperandError::LayerOutside {
                layer: 40,
                n_layers: 40
            }
        );
        // The row address composes the same slot rule the append kernel uses.
        let (address, row_bytes) = cache_position_address(0x40_0000, 2048, 1024, 5).expect("row 5");
        assert_eq!(address, 0x40_0000 + 5 * 2048);
        assert_eq!(row_bytes, 2048);
        assert!(cache_position_address(0x40_0000, 2048, 1024, 2048).is_err());
    }

    #[test]
    fn the_cache_row_address_matches_the_append_slot_rule() {
        // kv_append_rule's slot is position % capacity_rows; the address helper must compose it,
        // which is why the caller takes the slot from kv_ring.slot_for (the same formula).
        let ring = WindowRingLayout::new(2048).expect("ring");
        for position in [0usize, 1, 1023, 2047] {
            let slot = ring.slot_for(position);
            assert_eq!(slot, position);
            let (address, row_bytes) =
                cache_position_address(0x50_0000, 2048, 1024, slot).expect("row");
            assert_eq!(address, 0x50_0000 + (slot * 2048) as u64);
            assert_eq!(row_bytes, 2048);
        }
    }

    #[test]
    fn the_released_per_rank_output_geometry_is_two_groups_of_four_thousand_96() {
        let geometry = geometry();
        assert_eq!(geometry.groups, 2);
        assert_eq!(geometry.heads_per_group, 8);
        assert_eq!(geometry.group_inner, 4096);
        assert_eq!(geometry.wo_a_shape(), (2048, 4096));
        assert_eq!(geometry.wo_b_shape(), (5120, 2048));
        // The four row widths the five launches use.
        assert_eq!(geometry.activation_bytes(), 16 * 512 * 2);
        assert_eq!(geometry.lo_bytes(), 2048 * 2);
        assert_eq!(geometry.lo_code_bytes(), 2048);
        assert_eq!(geometry.lo_scale_bytes(), 64);
        assert_eq!(geometry.output_bytes(), 5120 * 2);
        assert_eq!(geometry.prologue_cols(), 2048);
        // The released plan shapes are the launched ones.
        geometry.check_wo_a((2048, 4096)).expect("wo_a");
        geometry.check_wo_b((5120, 2048)).expect("wo_b");
        geometry.check_activation_bytes(16 * 512 * 2).expect("row");
        assert_eq!(
            geometry
                .check_wo_a((1024, 8192))
                .expect_err("the world-8 shape"),
            OperandError::WoAShape {
                got: (1024, 8192),
                want: (2048, 4096)
            }
        );
        assert_eq!(
            geometry
                .check_wo_b((5120, 1024))
                .expect_err("the world-8 shape"),
            OperandError::WoBShape {
                got: (5120, 1024),
                want: (5120, 2048)
            }
        );
        assert_eq!(
            geometry
                .check_activation_bytes(8 * 512 * 2)
                .expect_err("world-8 row"),
            OperandError::ActivationBytes {
                got: 8 * 512 * 2,
                want: 16 * 512 * 2
            }
        );
    }

    #[test]
    fn a_geometry_that_does_not_split_is_refused_not_widened() {
        assert_eq!(
            OutputGeometry::new(8, 3, 21, 512, 1024, 5120).expect_err("8/3"),
            OperandError::GroupsNotDivisible {
                o_groups: 8,
                world: 3
            }
        );
        assert_eq!(
            OutputGeometry::new(8, 4, 15, 512, 1024, 5120).expect_err("15/2"),
            OperandError::HeadsNotDivisible {
                heads: 15,
                groups: 2
            }
        );
        assert_eq!(
            OutputGeometry::new(8, 4, 16, 0, 1024, 5120).expect_err("zero width"),
            OperandError::ZeroDimension { what: "head_width" }
        );
        assert_eq!(
            OutputGeometry::new(8, 0, 16, 512, 1024, 5120).expect_err("zero world"),
            OperandError::ZeroDimension { what: "the world" }
        );
    }

    #[test]
    fn the_indexer_reads_whole_blocks_up_to_the_query() {
        // One block covers tokens 0..8, so position 7 is the last of the first block.
        assert_eq!(index_key_block_bytes(0).expect("position 0"), 8 * 128 * 2);
        assert_eq!(index_key_block_bytes(7).expect("position 7"), 8 * 128 * 2);
        assert_eq!(index_key_block_bytes(8).expect("position 8"), 16 * 128 * 2);
        // A full 4096-position run reads half the pool at its last position.
        assert_eq!(
            index_key_block_bytes(4095).expect("last"),
            512 * 8 * 128 * 2
        );
        assert_eq!(
            index_key_pool_bytes(),
            HSI_BLOCKS * HSI_BLOCK_TOKENS * 128 * 2
        );
        let pool = crate::cache_plan::HSI_INDEX_KEY_BYTES;
        assert_eq!(index_key_pool_bytes(), pool);
    }

    #[test]
    fn the_selected_and_mask_spans_are_the_bound_constants() {
        assert_eq!(selected_bytes(0).expect("empty"), 0);
        assert_eq!(selected_bytes(MAX_SELECTED).expect("full"), 2048);
        assert_eq!(
            selected_bytes(MAX_SELECTED + 1).expect_err("past top-k"),
            OperandError::SelectedPastTopk {
                selected: MAX_SELECTED + 1,
                topk: MAX_SELECTED
            }
        );
        assert_eq!(candidate_mask_bytes(), 512);
        assert_eq!(STORE_CODE_BYTES, 131_072);
        assert_eq!(STORE_SCALE_BYTES, 16_384);
        assert_eq!(
            STORE_CODE_BYTES + STORE_SCALE_BYTES,
            crate::compressed_cache_ref::STORE_BYTES
        );
        assert_eq!(MAX_ATTENDED, 640);
        assert_eq!(WINDOW_ROWS, 128);
    }
}
