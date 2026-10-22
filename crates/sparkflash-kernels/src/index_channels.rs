//! The two index channels of reference_seam: index-K keys and index weights.
//!
//! INDEX K (host write_index_keys, attention.rs:2122, body :2831). The owner layer projects its
//! latent rows through indexer.wk (index_head_dim x head_dim), runs k_norm, narrows to bf16, applies
//! RoPE at the group positions, then round-trips through FP4 E2M1 with E8M0 power-of-two scales in
//! blocks of 32 (INDEX_QUANT_BLOCK = 32, attention.rs:66; the same grid kv/src/fp4.rs:44 calls
//! INDEXER_K_SCALE_GROUP) and publishes the rows. Three forms matter and none is guessed:
//! (a) the state rows are f32 of width index_head_dim after the round trip: 512 B per position;
//! (b) the packaged cache entry is 68 B per position (64 code bytes + 4 E8M0 scale bytes), the
//!     SparkKV layout for this channel;
//! (c) THE OPERAND the attention body reads is bf16, [hsi_blocks][8][index_head_dim]
//! (kernels/flash_mla_csa2_ring_sm121.cu:190), allocated as one fixed buffer of
//! HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM * 2 = 4,194,304 B (tp4_forward.rs:3295). That
//! is the form the arm writes; the round-tripped values have at most three significant mantissa
//! bits times a power of two, so the bf16 narrow is exact for the finite range the checkpoint
//! uses.
//! The projection itself needs NO new symbol: it is a bf16-weight, bf16-activation Linear, so it is
//! the host-weight GEMV the compressor already established, KERNEL_BF16_WEIGHT_F32_OUT with the
//! weight staged by hostw_gemv_ref::stage_bf16_weight. The kernel returns f32 accumulators; k_norm,
//! the bf16 narrow, RoPE and the fp4 round trip stay the host rule (attention.rs:2884-2892).
//!
//! INDEX WEIGHTS (host attention.rs:2131-2133, body :2776). The producer is indexer.weights_proj
//! (heads=32 x dim=5120) applied to the bf16-rounded layer input, bf16-rounded on the way out, then
//! multiplied by one scalar factor (index_head_dim^-1/2 * heads^-1/2 = 1/64 exactly) and
//! bf16-rounded again; the values are f32 rows of 32 per query position. ONLY the index-source
//! layers have it: the released config names eight of them, [2, 8, 14, 20, 24, 28, 32, 36]
//! (config/deepseek_v41_flash.released.json), while the key owners are the four kv-source layers
//! [2, 8, 14, 20]. The other 32 backbone layers have no index weights and the arm must refuse to
//! write them rather than publish zeros - the engine clears the field there
//! (attention.rs:2140-2142), which is a clear, not a zero tensor.

use crate::hostw_gemv_launch;
use crate::hostw_gemv_ref;

/// Index head width (mla_ref::INDEX_HEAD_DIM).
pub const INDEX_HEAD_DIM: usize = 128;
/// Index heads (mla_ref::INDEX_HEADS).
pub const INDEX_HEADS: usize = 32;
/// HSI blocks in the index operand (mla_ref::HSI_BLOCKS).
pub const HSI_BLOCKS: usize = 2048;
/// Tokens per HSI block (mla_ref::HSI_BLOCK_TOKENS).
pub const HSI_BLOCK_TOKENS: usize = 8;
/// Positions the index channels are allocated for: HSI_BLOCKS * HSI_BLOCK_TOKENS.
pub const KEY_POSITIONS: usize = HSI_BLOCKS * HSI_BLOCK_TOKENS;
/// The device operand size the body dereferences: bf16 keys, 4,194,304 B (tp4_forward.rs:3295).
pub const INDEX_KEY_OPERAND_BYTES: usize = KEY_POSITIONS * INDEX_HEAD_DIM * 2;
/// The cache entry size of the same channel in the packaged layout: 64 code + 4 scale bytes.
pub const INDEX_KEY_ENTRY_BYTES: usize = INDEX_HEAD_DIM / 2 + INDEX_HEAD_DIM / INDEX_QUANT_BLOCK;
/// FP4 scale group of the index channel (INDEX_QUANT_BLOCK, attention.rs:66).
pub const INDEX_QUANT_BLOCK: usize = 32;
/// The f32 state row the engine publishes before any narrowing.
pub const INDEX_KEY_STATE_BYTES: usize = INDEX_HEAD_DIM * 4;
/// The symbol the two projections launch: bf16 weight, f32 out, reusing the host-weight GEMV.
pub const PROJECTION_SYMBOL: &str = hostw_gemv_launch::KERNEL_BF16_WEIGHT_F32_OUT;
/// Whether this slice needs a new kernel symbol: it does not.
pub const NEEDS_NEW_SYMBOL: bool = false;

/// One declared form of an index-channel buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    /// What the buffer is called in the body or the cache.
    pub name: &'static str,
    /// The element dtype and encoding.
    pub dtype: &'static str,
    /// The layout.
    pub layout: &'static str,
    /// Bytes for one instance of the form.
    pub bytes: usize,
    /// Positions the form covers.
    pub positions: usize,
}

/// The operand the attention body reads for index keys.
#[must_use]
pub fn index_key_operand() -> Operand {
    Operand {
        name: "index_key",
        dtype: "bf16",
        layout: "[hsi_blocks][8][index_head_dim]",
        bytes: INDEX_KEY_OPERAND_BYTES,
        positions: KEY_POSITIONS,
    }
}

/// The packaged cache form of the same channel (SparkKV indexer K entry).
#[must_use]
pub fn index_key_entry() -> Operand {
    Operand {
        name: "indexer k entry",
        dtype: "FP4 E2M1 codes, E8M0 scale per 32 channels",
        layout: "64 code bytes then 4 scale bytes",
        bytes: INDEX_KEY_ENTRY_BYTES,
        positions: 1,
    }
}

/// The f32 state row the engine publishes.
#[must_use]
pub fn index_key_state() -> Operand {
    Operand {
        name: "indexer row",
        dtype: "f32 after the FP4 power-of-two round trip",
        layout: "[positions][index_head_dim]",
        bytes: INDEX_KEY_STATE_BYTES,
        positions: 1,
    }
}

/// Why an index channel was refused for a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexChannel {
    /// The index-K keys (only kv-source layers own them).
    Keys,
    /// The index weights (only index-source layers have them).
    Weights,
}

/// A refused index channel, named rather than filled with zeros.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexError {
    /// Which channel.
    pub channel: IndexChannel,
    /// The layer the caller asked about.
    pub layer: usize,
    /// How many layers do have it.
    pub layers_with_channel: usize,
}

impl core::fmt::Display for IndexError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let channel = match self.channel {
            IndexChannel::Keys => "index keys",
            IndexChannel::Weights => "index weights",
        };
        write!(
            f,
            "layer {} has no {channel}; only {} layers do - clear the field, do not publish zeros",
            self.layer, self.layers_with_channel,
        )
    }
}

/// Refuses a layer that does not own the index keys.
///
/// # Errors
/// [IndexError] naming the channel and the layer.
pub fn require_keys(owners: &[usize], layer: usize) -> Result<(), IndexError> {
    if owners.contains(&layer) {
        Ok(())
    } else {
        Err(IndexError {
            channel: IndexChannel::Keys,
            layer,
            layers_with_channel: owners.len(),
        })
    }
}

/// Refuses a layer that has no index weights.
///
/// # Errors
/// [IndexError] naming the channel and the layer.
pub fn require_weights(sources: &[usize], layer: usize) -> Result<(), IndexError> {
    if sources.contains(&layer) {
        Ok(())
    } else {
        Err(IndexError {
            channel: IndexChannel::Weights,
            layer,
            layers_with_channel: sources.len(),
        })
    }
}

/// Rounds an f32 to bf16 and back, round to nearest even.
#[must_use]
pub fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits((bits.wrapping_add(round) >> 16) << 16)
}

/// The scalar index-weights factor: index_head_dim^-1/2 * heads^-1/2, exactly 1/64 here.
#[must_use]
pub fn weights_factor() -> f32 {
    ((INDEX_HEAD_DIM as f64).powf(-0.5) * (INDEX_HEADS as f64).powf(-0.5)) as f32
}

/// Applies the host factor rule to one projection row: round the product to bf16 and widen.
#[must_use]
pub fn weights_row(projected: &[f32]) -> Vec<f32> {
    let factor = weights_factor();
    projected
        .iter()
        .map(|value| bf16_round(value * factor))
        .collect()
}

/// Stages an indexer.wk bf16 weight for the host-weight GEMV.
///
/// # Errors
/// [IndexError] is not used; the shape check is folded into the staging slice length, and a
/// matrix that is not a whole number of 32-value tiles gives back a zero-length image.
#[must_use]
pub fn stage_wk(weight_bf16: &[u16], head_dim: usize) -> Vec<u16> {
    if weight_bf16.len() != INDEX_HEAD_DIM * head_dim || !head_dim.is_multiple_of(32) {
        return Vec::new();
    }
    hostw_gemv_ref::stage_bf16_weight(weight_bf16, INDEX_HEAD_DIM, head_dim)
}

/// Stages an indexer.weights_proj bf16 weight for the host-weight GEMV.
#[must_use]
pub fn stage_weights_proj(weight_bf16: &[u16], dim: usize) -> Vec<u16> {
    if weight_bf16.len() != INDEX_HEADS * dim || !dim.is_multiple_of(32) {
        return Vec::new();
    }
    hostw_gemv_ref::stage_bf16_weight(weight_bf16, INDEX_HEADS, dim)
}

/// Bytes the index weights occupy for one step and one layer: f32 rows of INDEX_HEADS.
#[must_use]
pub fn weights_bytes_per_position() -> usize {
    INDEX_HEADS * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_operand_forms_are_pinned_and_consistent() {
        let operand = index_key_operand();
        assert_eq!(
            operand.bytes, 4_194_304,
            "the cli allocation at tp4_forward.rs:3295"
        );
        assert_eq!(operand.dtype, "bf16");
        assert_eq!(operand.positions, 16_384);
        assert_eq!(operand.bytes, operand.positions * INDEX_HEAD_DIM * 2);
        assert_eq!(index_key_entry().bytes, 68, "64 code + 4 scale bytes");
        assert_eq!(index_key_state().bytes, 512);
        assert_eq!(crate::mla_ref::HSI_BLOCKS, HSI_BLOCKS);
        assert_eq!(crate::mla_ref::HSI_BLOCK_TOKENS, HSI_BLOCK_TOKENS);
        assert_eq!(crate::mla_ref::INDEX_HEAD_DIM, INDEX_HEAD_DIM);
        assert_eq!(crate::mla_ref::INDEX_HEADS, INDEX_HEADS);
        assert_eq!(PROJECTION_SYMBOL, "b12x_hostw_gemv_m1_bf16w_f32o_sm121");
        let needs_new_symbol = std::hint::black_box(NEEDS_NEW_SYMBOL);
        assert!(
            !needs_new_symbol,
            "both projections reuse the host-weight GEMV"
        );
    }

    #[test]
    fn the_body_reads_bf16_index_keys() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/flash_mla_csa2_ring_sm121.cu"
        ))
        .expect("the body kernel source is in the tree");
        assert!(
            source.contains("const __nv_bfloat16 *__restrict__ index_key,     // [hsi_blocks][8][index_head_dim]"),
            "the body signature moved: the operand form has to be re-declared"
        );
    }

    #[test]
    fn the_layers_with_each_channel_are_the_released_configs() {
        let keys = [2usize, 8, 14, 20];
        let weights = [2usize, 8, 14, 20, 24, 28, 32, 36];
        for layer in keys {
            require_keys(&keys, layer).expect("a key owner");
            require_weights(&weights, layer).expect("an index source");
        }
        let refused = require_keys(&keys, 24).expect_err("24 owns no keys");
        assert_eq!(refused.channel, IndexChannel::Keys);
        assert_eq!(refused.layers_with_channel, 4);
        assert!(refused.to_string().contains("do not publish zeros"));
        let refused = require_weights(&weights, 3).expect_err("3 has no index weights");
        assert_eq!(refused.channel, IndexChannel::Weights);
        assert_eq!(refused.layers_with_channel, 8);
        assert!(require_weights(&weights, 36).is_ok());
        assert_eq!(weights_bytes_per_position(), 128);
    }

    #[test]
    fn the_factor_is_one_over_64_and_the_rounding_is_the_engines() {
        let factor = weights_factor();
        assert_eq!(
            factor.to_bits(),
            (1.0f32 / 64.0).to_bits(),
            "1/sqrt(128*32)"
        );
        let mut seed = 99u64;
        for _ in 0..2048 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 40.0;
            assert_eq!(
                bf16_round(value).to_bits(),
                sparkflash_engine::quant::bf16_round(value).to_bits(),
                "bf16 rounding must be the engine rule at {value}"
            );
            let row = weights_row(&[value]);
            assert_eq!(
                row[0].to_bits(),
                sparkflash_engine::quant::bf16_round(value * factor).to_bits()
            );
        }
    }

    #[test]
    fn the_staging_helpers_are_whole_tiles_and_the_right_shape() {
        let wk = vec![1u16; INDEX_HEAD_DIM * 512];
        let staged = stage_wk(&wk, 512);
        assert_eq!(staged.len(), INDEX_HEAD_DIM * 512);
        assert_eq!(
            staged.iter().filter(|word| **word == 1).count(),
            staged.len()
        );
        assert!(stage_wk(&wk, 500).is_empty(), "not a whole tile");
        let wp = vec![2u16; INDEX_HEADS * 5120];
        assert_eq!(stage_weights_proj(&wp, 5120).len(), INDEX_HEADS * 5120);
        assert!(stage_weights_proj(&wp, 5000).is_empty());
        assert!(stage_weights_proj(&wk, 512).is_empty(), "wrong value count");
    }
}
