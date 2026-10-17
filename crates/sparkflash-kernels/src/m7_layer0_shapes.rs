//! The layer-0 attention tensor shapes measured from the checkpoint headers, and the rule that
//! generates them.
//!
//! The shapes here are not design choices: they were read from the safetensors headers of the
//! released checkpoint on the host node (shard `model-00003-of-00048`), and the rule below is what
//! reproduces all five of them. A block-scaled weight stored as `[rows, cols]` carries a ue8m0
//! scale grid of `[rows / 32, cols / 32]` - one scale per 32x32 block, column-major relative to
//! the weight as stored. Two of these were guessed wrong before the headers were read; [scale_grid]
//! and the tests exist so that no future shape can be assumed wrong in silence.
//!
//! Two facts the table settles, both of which a plausible reading gets wrong:
//!
//! * `wo_a` is `[8192, 4096]`: `o_groups` 8 times `o_lora_rank` 1024, so the low-rank
//!   pair runs 4096 -> 8192 -> 5120, not 4096 -> 5120.
//! * `q_norm` is `[1280]` on the q LoRA rank and `kv_norm` is `[512]` on the KV latent, so neither is
//!   normalised per head: the attention normalisation is over the latent, not over `head_dim`.

/// One block-scaled weight of layer 0's attention, as measured: the weight's shape and the ue8m0
/// scale grid that belongs to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaledWeight {
    /// The checkpoint's tensor name beneath `layers.0.attn.`.
    pub name: &'static str,
    /// The weight's first dimension, as stored.
    pub rows: usize,
    /// The weight's second dimension, as stored.
    pub cols: usize,
    /// The scale tensor's first dimension, as stored.
    pub scale_rows: usize,
    /// The scale tensor's second dimension, as stored.
    pub scale_cols: usize,
}

/// The ue8m0 scale grid of a `[rows, cols]` weight: one scale per 32x32 block, so `[rows / 32, cols / 32]`
/// when the dimensions are multiples of 32, and the ceiling otherwise.
#[must_use]
pub const fn scale_grid(rows: usize, cols: usize) -> (usize, usize) {
    (rows.div_ceil(32), cols.div_ceil(32))
}

/// The five block-scaled attention weights of layer 0, measured from the checkpoint headers.
pub const LAYER0_ATTENTION_WEIGHTS: [BlockScaledWeight; 5] = [
    BlockScaledWeight {
        name: "wq_a",
        rows: 1280,
        cols: 5120,
        scale_rows: 40,
        scale_cols: 160,
    },
    BlockScaledWeight {
        name: "wq_b",
        rows: 32768,
        cols: 1280,
        scale_rows: 1024,
        scale_cols: 40,
    },
    BlockScaledWeight {
        name: "wkv",
        rows: 512,
        cols: 5120,
        scale_rows: 16,
        scale_cols: 160,
    },
    BlockScaledWeight {
        name: "wo_a",
        rows: 8192,
        cols: 4096,
        scale_rows: 256,
        scale_cols: 128,
    },
    BlockScaledWeight {
        name: "wo_b",
        rows: 5120,
        cols: 8192,
        scale_rows: 160,
        scale_cols: 256,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_measured_scale_grid_is_the_rule_s_applied_to_its_weight() {
        for weight in &LAYER0_ATTENTION_WEIGHTS {
            let (rows, cols) = scale_grid(weight.rows, weight.cols);
            assert_eq!(
                (rows, cols),
                (weight.scale_rows, weight.scale_cols),
                "{} carries a [{}x{}] scale grid, the rule gives [{rows}x{cols}]",
                weight.name,
                weight.scale_rows,
                weight.scale_cols
            );
        }
    }

    #[test]
    fn the_low_rank_pair_is_eight_groups_of_the_lora_rank() {
        let wo_a = LAYER0_ATTENTION_WEIGHTS[3];
        let wo_b = LAYER0_ATTENTION_WEIGHTS[4];
        assert_eq!(wo_a.name, "wo_a");
        assert_eq!(wo_b.name, "wo_b");
        // 8192 = 8 groups x 1024, and wo_b consumes that widened intermediate to return to 5120.
        assert_eq!(wo_a.rows, 8 * 1024);
        assert_eq!(wo_b.cols, wo_a.rows);
        assert_eq!(wo_b.rows, 5120);
        // The pair's two ends are different widths on purpose: wo_a transforms the 4096 wide grouped
        // attention output into the 8192 wide widened one, and wo_b transforms that back to 5120.
        assert_eq!(wo_a.cols, 4096);
        assert_eq!(wo_b.cols, 8192);
        assert_eq!(wo_b.cols, wo_a.rows);
        assert_ne!(wo_a.cols, wo_b.cols, "the two ends are not the same width");
    }

    #[test]
    fn the_two_norms_are_over_the_latents_not_over_the_head() {
        // q_norm is on the q LoRA rank, kv_norm on the KV latent; head_dim 512 is neither.
        let q_norm = 1280;
        let kv_norm = 512;
        assert_eq!(
            q_norm, LAYER0_ATTENTION_WEIGHTS[0].rows,
            "q_norm is on q_a's rank"
        );
        assert_eq!(
            kv_norm, LAYER0_ATTENTION_WEIGHTS[2].rows,
            "kv_norm is on wkv's latent"
        );
        assert_eq!(
            kv_norm,
            crate::mla_ref::MLA_LATENT,
            "the KV latent is MLA_LATENT wide"
        );
        assert_ne!(q_norm, 512, "q_norm is not the attention head width");
    }

    #[test]
    fn the_attention_projections_multiply_out_to_the_head_stack() {
        // wq_b is heads x head_dim = 64 x 512 = 32768 rows, from a 1280 wide q latent.
        assert_eq!(LAYER0_ATTENTION_WEIGHTS[1].rows, 64 * 512);
        assert_eq!(
            LAYER0_ATTENTION_WEIGHTS[1].cols,
            LAYER0_ATTENTION_WEIGHTS[0].rows
        );
    }
}
