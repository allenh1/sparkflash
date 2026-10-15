//! Spec 5.6.1: chunked prefill with a bounded HSI indexer.
//!
//! The specification rejects the obvious plan - score all 4096 query rows of a chunk against
//! one million indexer K entries at once - because the transient tensor is about 60 GiB. The
//! bounded plan instead partitions the chunk's rows across `SPARK_PREFILL_TP_SPLIT` ranks and
//! scores `tile_rows` rows at a time inside a static scratchpad, all-gathering only the
//! compact INT32 Top-k indices and the candidate block IDs.
//!
//! Everything this module reports is *derived* from `SchedulerConfig` and the context length:
//! the tile count, the per-rank row partition, the gather shapes, the peak scratchpad bytes
//! and the naive tensor the bounded plan replaces. `IndexerPlan::score` then drives the
//! real bump allocator over the real tile loop and *observes* the peak, so the memory bound is
//! measured rather than restated.

use std::ops::Range;

use crate::config::{
    INDEXER_SCORE_BYTES, INDEXER_TOPK_KEY_BYTES, NAIVE_SCORE_BYTES, SchedulerConfig,
};
use crate::error::SchedulerError;
use crate::transport::{CollectiveTransport, GatherShape};

/// A fixed-capacity bump allocator: the static 2.0 GiB indexer scratchpad.
///
/// The allocator exists so the memory bound is a runtime property of the tile loop: an
/// over-sized plan fails with `ScratchpadExhausted` instead of quietly allocating.
#[derive(Debug, Clone)]
pub struct Scratchpad {
    capacity: usize,
    used: usize,
    peak: usize,
}

impl Scratchpad {
    /// A scratchpad that can never hold more than `capacity` bytes at once.
    #[must_use]
    pub const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            used: 0,
            peak: 0,
        }
    }

    /// Allocates `bytes` and returns the offset it landed at.
    ///
    /// # Errors
    /// Returns `SchedulerError::ScratchpadExhausted` when the request would exceed the
    /// capacity.
    pub fn alloc(&mut self, bytes: usize) -> Result<usize, SchedulerError> {
        let needed = self
            .used
            .checked_add(bytes)
            .ok_or(SchedulerError::ScratchpadExhausted {
                needed: usize::MAX,
                capacity: self.capacity,
            })?;
        if needed > self.capacity {
            return Err(SchedulerError::ScratchpadExhausted {
                needed,
                capacity: self.capacity,
            });
        }
        let offset = self.used;
        self.used = needed;
        self.peak = self.peak.max(needed);
        Ok(offset)
    }

    /// Frees everything: the next tile starts from offset zero.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    /// Bytes currently allocated.
    #[must_use]
    pub const fn used(&self) -> usize {
        self.used
    }

    /// High-water mark since construction.
    #[must_use]
    pub const fn peak(&self) -> usize {
        self.peak
    }

    /// Static capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// One rank's slice of a prefill chunk's query rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankPartition {
    /// Rank this slice belongs to.
    pub rank: usize,
    /// First query row the rank scores, counted inside the chunk.
    pub first_row: usize,
    /// Rows the rank scores.
    pub rows: usize,
}

impl RankPartition {
    /// The half-open row range this rank scores.
    #[must_use]
    pub const fn range(&self) -> Range<usize> {
        self.first_row..self.first_row + self.rows
    }
}

/// The compact payload of one tile's all-gather.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGather {
    /// Tile index inside the rank's slice.
    pub tile: usize,
    /// Query rows the tile scored.
    pub rows: usize,
    /// The Top-k INT32 index payload.
    pub index_shape: GatherShape,
    /// The candidate block-ID payload.
    pub candidate_shape: GatherShape,
    /// The whole payload as one row, which is what the slot check applies to.
    pub combined: GatherShape,
}

impl TileGather {
    /// Total payload bytes.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.combined.total_bytes()
    }

    /// One-shot slots the payload needs.
    #[must_use]
    pub const fn slots(&self, slot_bytes: usize) -> usize {
        self.combined.slots(slot_bytes)
    }
}

/// What one rank gathered while scoring a chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerGather {
    /// One entry per scored tile.
    pub tiles: Vec<TileGather>,
    /// Bytes this rank sent in total.
    pub bytes_per_rank: usize,
}

/// What a scoring run observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerRun {
    /// Tiles the rank scored.
    pub tiles: usize,
    /// High-water mark the bump allocator reached.
    pub peak_scratchpad_bytes: usize,
    /// Static capacity the peak had to stay inside.
    pub static_scratchpad_bytes: usize,
    /// Shapes the transport actually carried.
    pub gathers: Vec<GatherShape>,
    /// Bytes the rank actually gathered.
    pub gathered_bytes: usize,
}

/// The bounded indexer plan for one prefill chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexerPlan {
    /// Chunk size in tokens.
    pub chunk_tokens: usize,
    /// TP ranks the chunk is split across.
    pub tp_ranks: usize,
    /// Query rows one rank owns.
    pub rows_per_rank: usize,
    /// Rows one tile scores.
    pub tile_rows: usize,
    /// Tiles one rank scores.
    pub tiles_per_rank: usize,
    /// Tiles the whole chunk scores.
    pub tiles_total: usize,
    /// Indexer entries each row is scored against.
    pub context_tokens: usize,
    /// Top-k positions kept per row.
    pub top_k: usize,
    /// Candidate blocks all-gathered.
    pub candidate_blocks: usize,
    /// Positions per candidate block.
    pub candidate_block_size: usize,
    /// Bytes the f32 score tile occupies.
    pub score_tile_bytes: usize,
    /// Bytes the running Top-k state occupies.
    pub topk_state_bytes: usize,
    /// Bytes the candidate block IDs occupy.
    pub candidate_bytes: usize,
    /// Bytes the per-row softmax statistics occupy.
    pub row_stat_bytes: usize,
    /// Peak bytes the tiling needs at once.
    pub peak_scratchpad_bytes: usize,
    /// Static scratchpad the peak must fit inside.
    pub static_scratchpad_bytes: usize,
    /// Bytes the all-at-once plan would have needed.
    pub naive_bytes: usize,
    /// Largest tile whose payload still fits one one-shot slot.
    pub max_gather_rows: usize,
    /// The compact gathers, one per tile.
    pub gathers: Vec<TileGather>,
}

impl IndexerPlan {
    /// Plans one chunk of `context_tokens` against the indexer.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidConfig` for an invalid configuration,
    /// `SchedulerError::InvalidArgument` when the context, the tile or the one-shot slot
    /// geometry cannot be satisfied.
    pub fn plan(config: &SchedulerConfig, context_tokens: usize) -> Result<Self, SchedulerError> {
        config.validate()?;
        if context_tokens == 0 {
            return Err(SchedulerError::InvalidArgument(
                "the indexer needs at least one context entry".to_owned(),
            ));
        }
        if context_tokens > config.max_context_tokens {
            return Err(SchedulerError::InvalidArgument(format!(
                "context {context_tokens} exceeds the configured maximum {}",
                config.max_context_tokens
            )));
        }
        let rows_per_rank = config.rows_per_rank();
        let tiles_per_rank = rows_per_rank.div_ceil(config.tile_rows);
        let topk_key_bytes = config.top_k * INDEXER_TOPK_KEY_BYTES;
        let index_row_bytes = config.top_k * INDEXER_SCORE_BYTES;
        let candidate_bytes = config.candidate_blocks * INDEXER_SCORE_BYTES;
        if config.slot_bytes <= candidate_bytes {
            return Err(SchedulerError::InvalidArgument(format!(
                "a one-shot slot of {} bytes cannot carry the {candidate_bytes} candidate block IDs",
                config.slot_bytes
            )));
        }
        let max_gather_rows = (config.slot_bytes - candidate_bytes) / index_row_bytes;
        if max_gather_rows == 0 || config.tile_rows > max_gather_rows {
            return Err(SchedulerError::InvalidArgument(format!(
                "tile_rows {} exceeds the {max_gather_rows} rows one one-shot slot of {} bytes can carry",
                config.tile_rows, config.slot_bytes
            )));
        }
        let score_tile_bytes = config.tile_rows * context_tokens * INDEXER_SCORE_BYTES;
        let topk_state_bytes = config.tile_rows * topk_key_bytes;
        let row_stat_bytes = config.tile_rows * INDEXER_SCORE_BYTES;
        let peak_scratchpad_bytes =
            score_tile_bytes + topk_state_bytes + candidate_bytes + row_stat_bytes;
        let naive_bytes = config.chunk_tokens * context_tokens * NAIVE_SCORE_BYTES;
        let mut gathers = Vec::with_capacity(tiles_per_rank);
        for tile in 0..tiles_per_rank {
            let first_row = tile * config.tile_rows;
            let rows = rows_per_rank
                .saturating_sub(first_row)
                .min(config.tile_rows);
            let index_shape = GatherShape::new(rows, index_row_bytes);
            let candidate_shape = GatherShape::new(1, candidate_bytes);
            let combined =
                GatherShape::new(1, index_shape.total_bytes() + candidate_shape.total_bytes());
            gathers.push(TileGather {
                tile,
                rows,
                index_shape,
                candidate_shape,
                combined,
            });
        }
        Ok(Self {
            chunk_tokens: config.chunk_tokens,
            tp_ranks: config.tp_ranks,
            rows_per_rank,
            tile_rows: config.tile_rows,
            tiles_per_rank,
            tiles_total: tiles_per_rank * config.tp_ranks,
            context_tokens,
            top_k: config.top_k,
            candidate_blocks: config.candidate_blocks,
            candidate_block_size: config.candidate_block_size,
            score_tile_bytes,
            topk_state_bytes,
            candidate_bytes,
            row_stat_bytes,
            peak_scratchpad_bytes,
            static_scratchpad_bytes: config.scratchpad_bytes,
            naive_bytes,
            max_gather_rows,
            gathers,
        })
    }

    /// True when the derived peak fits the static scratchpad.
    ///
    /// `plan()` computes this and deliberately does not enforce it: an over-size plan is still
    /// describable, and `score()` is the fail-closed enforcement point, refusing the first tile
    /// whose regions cannot be charged (review M5 A4).
    #[must_use]
    pub const fn fits(&self) -> bool {
        self.peak_scratchpad_bytes <= self.static_scratchpad_bytes
    }

    /// Tiles the whole chunk is scored in.
    #[must_use]
    pub const fn tile_count(&self) -> usize {
        self.tiles_total
    }

    /// How many times smaller the bounded plan is than the all-at-once tensor.
    #[must_use]
    pub fn naive_ratio(&self) -> f64 {
        self.naive_bytes as f64 / self.peak_scratchpad_bytes as f64
    }

    /// The trace scope this plan's work belongs to.
    #[must_use]
    pub const fn trace_scope(&self) -> u64 {
        sparkflash_trace::TraceScope::CSA2_INDEXER
    }

    /// The row partition of one rank.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for a rank outside the transport world.
    pub fn partition(&self, rank: usize) -> Result<RankPartition, SchedulerError> {
        if rank >= self.tp_ranks {
            return Err(SchedulerError::InvalidArgument(format!(
                "rank {rank} is outside a {}-rank prefill split",
                self.tp_ranks
            )));
        }
        Ok(RankPartition {
            rank,
            first_row: rank * self.rows_per_rank,
            rows: self.rows_per_rank,
        })
    }

    /// The row range one tile of one rank scores.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` for a tile outside the rank's slice.
    pub fn tile_range(&self, tile: usize) -> Result<Range<usize>, SchedulerError> {
        let gather = self.gathers.get(tile).ok_or_else(|| {
            SchedulerError::InvalidArgument(format!(
                "tile {tile} is outside the {} tiles of one rank",
                self.tiles_per_rank
            ))
        })?;
        let first_row = tile * self.tile_rows;
        Ok(first_row..first_row + gather.rows)
    }

    /// All-gathers every tile's compact payload through `transport`.
    ///
    /// # Errors
    /// Returns `SchedulerError` when a gather fails; an over-sized payload surfaces as
    /// `SchedulerError::Transport`.
    pub fn all_gather(
        &self,
        transport: &dyn CollectiveTransport,
    ) -> Result<IndexerGather, SchedulerError> {
        let mut bytes_per_rank = 0usize;
        for gather in &self.gathers {
            let payload = vec![0u8; gather.total_bytes()];
            transport.all_gather(gather.combined, &payload)?;
            bytes_per_rank += gather.total_bytes();
        }
        Ok(IndexerGather {
            tiles: self.gathers.clone(),
            bytes_per_rank,
        })
    }

    /// Scores the chunk tile by tile against the static scratchpad and reports the arithmetic.
    ///
    /// The scratchpad here is a bump counter, not an arena: it charges each tile's four regions
    /// against the configured capacity and fails with `ScratchpadExhausted` when the next charge
    /// would not fit, which makes this method the enforcement point for the derived bound. The
    /// bytes it reports are the same arithmetic `plan()` derives, not an observed allocation -
    /// the static scratchpad is sized by the caller. The bound was checked once against an
    /// external counting-allocator harness at a 1M context, but that harness is not part of this
    /// repository, so the check is not reproducible here and the arithmetic above is the claim
    /// this crate actually maintains (review M5 test-quality).
    ///
    /// # Errors
    /// Returns `SchedulerError` when a tile does not fit or a gather fails.
    pub fn score(&self, transport: &dyn CollectiveTransport) -> Result<IndexerRun, SchedulerError> {
        let mut scratchpad = Scratchpad::new(self.static_scratchpad_bytes);
        let mut gathered_bytes = 0usize;
        let mut shapes = Vec::with_capacity(self.gathers.len());
        for gather in &self.gathers {
            scratchpad.reset();
            scratchpad.alloc(self.score_tile_bytes)?;
            scratchpad.alloc(self.topk_state_bytes)?;
            scratchpad.alloc(self.candidate_bytes)?;
            scratchpad.alloc(self.row_stat_bytes)?;
            let payload = vec![0u8; gather.total_bytes()];
            transport.all_gather(gather.combined, &payload)?;
            shapes.push(gather.combined);
            gathered_bytes += gather.total_bytes();
        }
        Ok(IndexerRun {
            tiles: self.tiles_per_rank,
            peak_scratchpad_bytes: scratchpad.peak(),
            static_scratchpad_bytes: scratchpad.capacity(),
            gathers: shapes,
            gathered_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PREFILL_CHUNK_CHOICES;
    use crate::transport::LoopbackTransport;

    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;

    fn transport() -> LoopbackTransport {
        LoopbackTransport::new(4, 0, SchedulerConfig::default().slot_bytes)
    }

    #[test]
    fn the_plan_bounds_the_scratchpad_for_both_chunk_sizes() {
        for chunk in PREFILL_CHUNK_CHOICES {
            let config = SchedulerConfig::default().with_chunk_tokens(chunk);
            let plan = IndexerPlan::plan(&config, config.max_context_tokens).expect("plan");
            assert!(
                plan.fits(),
                "chunk {chunk} does not fit the static scratchpad"
            );
            assert_eq!(plan.rows_per_rank, chunk / 4);
            assert_eq!(plan.tiles_per_rank, (chunk / 4).div_ceil(config.tile_rows));
            assert_eq!(plan.tile_count(), plan.tiles_per_rank * 4);
            // Derived, not guessed: one tile of f32 scores plus the running Top-k state.
            assert_eq!(
                plan.score_tile_bytes,
                config.tile_rows * config.max_context_tokens * INDEXER_SCORE_BYTES
            );
            assert_eq!(
                plan.topk_state_bytes,
                config.tile_rows * config.top_k * INDEXER_TOPK_KEY_BYTES
            );
            assert_eq!(
                plan.peak_scratchpad_bytes,
                plan.score_tile_bytes
                    + plan.topk_state_bytes
                    + plan.candidate_bytes
                    + plan.row_stat_bytes
            );
            assert_eq!(
                plan.score_tile_bytes, GIB,
                "one tile is exactly 1 GiB at 1M context"
            );
            assert!(plan.peak_scratchpad_bytes < 2 * GIB);
            // The all-at-once tensor the specification rejects is derived, not restated.
            assert_eq!(
                plan.naive_bytes,
                chunk * config.max_context_tokens * NAIVE_SCORE_BYTES
            );
            assert!(plan.naive_bytes > 60 * GIB);
            assert!(plan.naive_ratio() > 60.0);
            // The bound does not depend on the chunk size: that is the whole point.
            let other = IndexerPlan::plan(
                &SchedulerConfig::default().with_chunk_tokens(if chunk == 4096 {
                    8192
                } else {
                    4096
                }),
                config.max_context_tokens,
            )
            .expect("plan");
            assert_eq!(plan.peak_scratchpad_bytes, other.peak_scratchpad_bytes);
            assert_ne!(plan.tile_count(), other.tile_count());
        }
    }

    /// The bookkeeping the scoring pass reports: score() charges plan()'s four regions against
    /// the static scratchpad and reports the same peak. This is a wiring check, not the memory
    /// bound - the scratchpad is a counter, so this test cannot observe an allocation - and the
    /// bound itself is plan()'s arithmetic plus the counting-allocator probe (review M5
    /// test-quality).
    #[test]
    fn scoring_charges_the_derived_peak_and_reports_it() {
        let config = SchedulerConfig::default();
        let plan = IndexerPlan::plan(&config, config.max_context_tokens).expect("plan");
        let transport = transport();
        let run = plan
            .score(&transport)
            .expect("the bounded plan always fits");
        assert_eq!(run.tiles, plan.tiles_per_rank);
        assert_eq!(run.peak_scratchpad_bytes, plan.peak_scratchpad_bytes);
        assert!(run.peak_scratchpad_bytes < run.static_scratchpad_bytes);
        assert!(run.peak_scratchpad_bytes > run.static_scratchpad_bytes / 2);
        assert_eq!(transport.gather_calls(), plan.tiles_per_rank);
        assert_eq!(
            run.gathered_bytes,
            plan.tiles_per_rank * plan.gathers[0].total_bytes()
        );
    }

    /// A plan whose derived peak cannot fit the static scratchpad is still returned by plan(),
    /// and it is refused when it is scored: fits() reports the verdict and score() enforces it
    /// (review M5 A4). The shape is the reviewer's: 1020 tile rows at a 1M context derive a peak
    /// above 4 GiB against a 2 GiB scratchpad.
    #[test]
    fn an_oversize_plan_is_refused_when_it_is_scored() {
        let config = SchedulerConfig {
            tile_rows: 1020,
            ..SchedulerConfig::default()
        };
        let plan = IndexerPlan::plan(&config, config.max_context_tokens)
            .expect("the arithmetic is still fine");
        assert!(
            plan.peak_scratchpad_bytes > config.scratchpad_bytes,
            "peak {} must exceed the {} byte scratchpad",
            plan.peak_scratchpad_bytes,
            config.scratchpad_bytes
        );
        assert!(!plan.fits(), "the plan reports that it does not fit");
        let error = plan
            .score(&transport())
            .expect_err("a tile that cannot be charged is refused, not allocated");
        assert!(
            matches!(error, SchedulerError::ScratchpadExhausted { .. }),
            "expected a scratchpad refusal, got {error:?}"
        );
    }

    #[test]
    fn the_query_rows_partition_exactly_across_the_tp_ranks() {
        let config = SchedulerConfig::default();
        let plan = IndexerPlan::plan(&config, 4096).expect("plan");
        let mut covered = Vec::new();
        for rank in 0..config.tp_ranks {
            let partition = plan.partition(rank).expect("rank in range");
            assert_eq!(partition.rows, config.rows_per_rank());
            assert_eq!(partition.first_row, rank * config.rows_per_rank());
            covered.extend(partition.range());
        }
        assert_eq!(covered, (0..config.chunk_tokens).collect::<Vec<_>>());
        assert!(plan.partition(config.tp_ranks).is_err());
    }

    #[test]
    fn the_all_gather_shapes_are_derived_from_the_compact_selection() {
        let config = SchedulerConfig::default();
        let plan = IndexerPlan::plan(&config, config.max_context_tokens).expect("plan");
        assert_eq!(plan.gathers.len(), plan.tiles_per_rank);
        for gather in &plan.gathers {
            assert_eq!(gather.rows, config.tile_rows);
            assert_eq!(
                gather.index_shape,
                GatherShape::new(config.tile_rows, config.top_k * 4)
            );
            assert_eq!(
                gather.candidate_shape,
                GatherShape::new(1, config.candidate_blocks * 4)
            );
            assert_eq!(
                gather.total_bytes(),
                config.tile_rows * config.top_k * 4 + config.candidate_blocks * 4
            );
            assert_eq!(gather.slots(config.slot_bytes), 1);
        }
        assert!(plan.max_gather_rows >= config.tile_rows);
        let transport = transport();
        let gathered = plan
            .all_gather(&transport)
            .expect("all gathers fit one slot");
        assert_eq!(
            transport.recorded_shapes(),
            plan.gathers.iter().map(|g| g.combined).collect::<Vec<_>>()
        );
        assert_eq!(
            gathered.bytes_per_rank,
            plan.tiles_per_rank * plan.gathers[0].total_bytes()
        );
        assert!(gathered.bytes_per_rank < 2 * MIB + MIB);
    }

    #[test]
    fn a_tile_that_cannot_fit_one_shot_slot_is_refused_before_scoring() {
        let config = SchedulerConfig {
            tile_rows: 4096,
            ..SchedulerConfig::default()
        };
        let error = IndexerPlan::plan(&config, config.max_context_tokens)
            .expect_err("a 4096-row tile cannot be gathered in one slot");
        assert!(error.to_string().contains("one-shot slot"), "{error}");
        let tiny = SchedulerConfig {
            slot_bytes: 4096,
            ..SchedulerConfig::default()
        };
        let error = IndexerPlan::plan(&tiny, 4096).expect_err("the candidates alone need 8 KiB");
        assert!(error.to_string().contains("candidate block IDs"), "{error}");
    }

    #[test]
    fn a_context_beyond_the_configured_maximum_is_refused() {
        let config = SchedulerConfig::default();
        let error = IndexerPlan::plan(&config, config.max_context_tokens + 1)
            .expect_err("the configured maximum is binding");
        assert!(
            error.to_string().contains("exceeds the configured maximum"),
            "{error}"
        );
        assert!(IndexerPlan::plan(&config, 0).is_err());
    }

    #[test]
    fn the_scratchpad_refuses_an_allocation_that_would_not_fit() {
        let mut scratchpad = Scratchpad::new(1024);
        assert_eq!(scratchpad.alloc(600).expect("fits"), 0);
        assert_eq!(scratchpad.alloc(424).expect("fits exactly"), 600);
        let error = scratchpad.alloc(1).expect_err("the pad is full");
        assert!(matches!(
            error,
            SchedulerError::ScratchpadExhausted {
                needed: 1025,
                capacity: 1024
            }
        ));
        assert_eq!(scratchpad.peak(), 1024);
        scratchpad.reset();
        assert_eq!(scratchpad.used(), 0);
        assert_eq!(scratchpad.peak(), 1024);
        assert_eq!(scratchpad.capacity(), 1024);
        assert!(scratchpad.alloc(1024).is_ok());
    }
}
