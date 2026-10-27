//! Static DeepSeek-V4.1-Flash layer topology.
//!
//! Every value here is taken verbatim from the published checkpoint config
//! (text_config in config.json), so the module is a weight-free source of truth for the
//! rest of the engine and for tests.

/// Layers that own the global (compressed) main KV cache and the indexer K.
pub const KV_SOURCE_LAYERS: [usize; 4] = [2, 8, 14, 20];

/// Layers that own an indexer (Full mode or Reindex mode).
pub const INDEX_SOURCE_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

/// Layer that builds the hierarchical sparse indexer candidate pool.
pub const CANDIDATE_SOURCE_LAYER: usize = 20;

/// Number of candidate blocks built by the candidate source layer.
pub const CANDIDATE_TOPK_BLOCKS: usize = 2048;

/// Positions per candidate block.
pub const CANDIDATE_BLOCK_SIZE: usize = 8;

/// Total candidate positions scored by Reindex layers (2048 blocks x 8 positions).
pub const CANDIDATE_POSITIONS: usize = CANDIDATE_TOPK_BLOCKS * CANDIDATE_BLOCK_SIZE;

/// Sliding window size in tokens, active in every layer.
pub const SWA_WINDOW_SIZE: usize = sparkflash_model::telemetry::DEFAULT_SWA_WINDOW_TOKENS;

/// Tokens per global KV page (matches the 64-token FlashMLA page split).
pub const KV_PAGE_TOKENS: usize = sparkflash_model::telemetry::DEFAULT_KV_PAGE_TOKENS;

/// Logical global KV bytes per token across the whole model, as reported in the paper.
pub const GLOBAL_KV_BYTES_PER_TOKEN: usize = 890;

/// Top-k positions selected by the sparse indexer.
pub const INDEX_TOPK: usize = 512;

/// Number of transformer layers in the backbone.
pub const NUM_HIDDEN_LAYERS: usize = 40;

/// Last encoder layer under D7 (inclusive): layers 0..=20 run over the whole prompt.
///
/// D7 fixes the boundary at the model's middle layer, 20, which is also the last Full layer
/// (KV_SOURCE_LAYERS), the Full KV source the decoder layers 21..=39 read, and the HSI
/// candidate source (CANDIDATE_SOURCE_LAYER). This is the single source of the split for the
/// CED schedule and the SWA bounded-replay kinds; the cross-check test pins it against both.
///
/// The value is written out rather than derived as NUM_HIDDEN_LAYERS / 2: the split is 21/19,
/// so the arithmetic half is precisely the coincidence D7 does not follow.
pub const ENCODER_LAST_LAYER: usize = 20;

/// First decoder layer under D7: layers 21..=39 run only over the last window of the prompt.
pub const DECODER_FIRST_LAYER: usize = ENCODER_LAST_LAYER + 1;

/// compress_ratios from config.json, one entry per backbone layer plus the three DSpark
/// prediction layers (indexes 40..=42, which are not backbone attention layers).
pub const COMPRESS_RATIOS: [usize; 43] = [
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
];

/// How a layer participates in compressed sparse attention 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Csa2Mode {
    /// No global branch at all: only the 128-token sliding window.
    SwaOnly,
    /// Computes the compressed main KV, projects indexer K and selects Top-512 itself.
    Full,
    /// Reuses another layer's main KV and indexer K but rescores the hierarchical
    /// candidate pool to pick its own Top-512.
    Reindex,
    /// Reuses both the main KV and the previous layer's Top-512 indices.
    Reuse,
}

impl Csa2Mode {
    /// True when the layer stores its own copy of the compressed KV.
    #[must_use]
    pub fn owns_main_kv(self) -> bool {
        matches!(self, Csa2Mode::Full)
    }

    /// True when the layer runs indexer scoring.
    #[must_use]
    pub fn scores_indices(self) -> bool {
        matches!(self, Csa2Mode::Full | Csa2Mode::Reindex)
    }
}

/// Static mode of a backbone layer.
///
/// Layers 0 and 1 have compress ratio 0 (SWA only). The kv_source_layer_ids are Full. The
/// remaining index_source_layer_ids are Reindex. Everything else reuses.
///
/// # Panics
/// Panics when `layer >= NUM_HIDDEN_LAYERS`.
#[must_use]
pub fn csa2_mode(layer: usize) -> Csa2Mode {
    assert!(
        layer < NUM_HIDDEN_LAYERS,
        "layer {layer} is outside the backbone"
    );
    if compress_ratio(layer) == 0 {
        return Csa2Mode::SwaOnly;
    }
    if KV_SOURCE_LAYERS.contains(&layer) {
        return Csa2Mode::Full;
    }
    if INDEX_SOURCE_LAYERS.contains(&layer) {
        return Csa2Mode::Reindex;
    }
    Csa2Mode::Reuse
}

/// Compression ratio m of a layer: one compressed KV entry per m tokens.
///
/// # Panics
/// Panics when `layer >= NUM_HIDDEN_LAYERS`.
#[must_use]
pub fn compress_ratio(layer: usize) -> usize {
    assert!(
        layer < NUM_HIDDEN_LAYERS,
        "layer {layer} is outside the backbone"
    );
    COMPRESS_RATIOS[layer]
}

/// Layer that owns the main KV an `Reuse`/`Reindex` layer reads from: the nearest
/// preceding Full layer.
///
/// # Panics
/// Panics when `layer` has no preceding Full layer.
#[must_use]
pub fn kv_source_for(layer: usize) -> usize {
    *KV_SOURCE_LAYERS
        .iter()
        .rev()
        .find(|s| **s <= layer)
        .unwrap_or_else(|| panic!("no KV source layer at or before {layer}"))
}

/// Layer that owns the indexer K a Reindex/Reuse layer reads from.
///
/// # Panics
/// Panics when `layer` has no preceding index source layer.
#[must_use]
pub fn index_source_for(layer: usize) -> usize {
    *INDEX_SOURCE_LAYERS
        .iter()
        .rev()
        .find(|s| **s <= layer)
        .unwrap_or_else(|| panic!("no index source layer at or before {layer}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_modes_match_the_published_topology() {
        assert_eq!(csa2_mode(0), Csa2Mode::SwaOnly);
        assert_eq!(csa2_mode(1), Csa2Mode::SwaOnly);
        assert_eq!(csa2_mode(2), Csa2Mode::Full);
        assert_eq!(csa2_mode(3), Csa2Mode::Reuse);
        assert_eq!(csa2_mode(7), Csa2Mode::Reuse);
        assert_eq!(csa2_mode(8), Csa2Mode::Full);
        assert_eq!(csa2_mode(14), Csa2Mode::Full);
        assert_eq!(csa2_mode(19), Csa2Mode::Reuse);
        assert_eq!(csa2_mode(20), Csa2Mode::Full);
        assert_eq!(csa2_mode(24), Csa2Mode::Reindex);
        assert_eq!(csa2_mode(36), Csa2Mode::Reindex);
        assert_eq!(csa2_mode(39), Csa2Mode::Reuse);
    }

    #[test]
    fn exactly_four_layers_own_main_kv() {
        let owners: Vec<usize> = (0..NUM_HIDDEN_LAYERS)
            .filter(|l| csa2_mode(*l).owns_main_kv())
            .collect();
        assert_eq!(owners, KV_SOURCE_LAYERS.to_vec());
    }

    #[test]
    fn compress_ratios_are_encoder_shaped() {
        assert_eq!(compress_ratio(0), 0);
        assert_eq!(compress_ratio(2), 2);
        assert_eq!(compress_ratio(19), 2);
        assert_eq!(compress_ratio(20), 1);
        assert_eq!(compress_ratio(39), 1);
    }

    #[test]
    fn reuse_layers_resolve_to_the_nearest_source() {
        assert_eq!(kv_source_for(3), 2);
        assert_eq!(kv_source_for(13), 8);
        assert_eq!(kv_source_for(19), 14);
        assert_eq!(kv_source_for(39), 20);
        assert_eq!(index_source_for(25), 24);
        assert_eq!(index_source_for(39), 36);
    }

    #[test]
    fn candidate_pool_matches_the_config() {
        assert_eq!(CANDIDATE_POSITIONS, 16_384);
    }
}
