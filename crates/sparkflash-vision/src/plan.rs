//! The pixel size and token grid an image is given, ported from `plan_image_grid`, `safe_resize`,
//! `solve_resize_ratio` and `llm_grid` in the reference `image_processor.py`.
//!
//! The arithmetic follows the Python expression by expression (the same f64 operations in the same
//! order, `x ** 0.5` as `powf(0.5)`, `int()` and `math.floor` as truncation of a positive value),
//! because a size one patch off changes the token count.

/// The vision tower's shape constants that the plan depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// ViT patch side in pixels (`vision_patch_size`).
    pub patch: usize,
    /// Aligner downsample per side (`vision_downsample_ratio`).
    pub downsample: usize,
    /// Largest image span in tokens (`vision_max_n_token`).
    pub max_tokens: usize,
    /// Images under this many pixels are planned as if scaled up to it (`vision_min_pixels`).
    pub min_pixels: usize,
}

impl Geometry {
    /// DeepSeek V4.1 Flash (`reference/inference_config.json`; `vision_max_wh_ratio` is null).
    pub const V41: Self = Self {
        patch: 14,
        downsample: 3,
        max_tokens: 1024,
        min_pixels: 295_936,
    };

    /// Pixels per side of one LLM token.
    #[must_use]
    pub const fn cell(&self) -> usize {
        self.patch * self.downsample
    }

    /// The token grid the aligner makes from a canvas of this pixel size (`llm_grid`).
    #[must_use]
    pub fn grid(&self, best_height: usize, best_width: usize) -> Grid {
        Grid {
            n_llm_h: (best_height / self.patch).div_ceil(self.downsample),
            n_llm_w: (best_width / self.patch).div_ceil(self.downsample),
            best_height,
            best_width,
        }
    }

    /// `solve_resize_ratio`: the largest aspect-preserving size whose grid fits `max_tokens`.
    fn solve(&self, height: usize, width: usize) -> (usize, usize) {
        let max_tokens = self.max_tokens;
        let cell = self.cell();
        let r = height as f64 / width as f64;
        let max_w_float = ((max_tokens - 2) as f64 / r + 0.25).sqrt() - 0.5;
        let max_h_float = max_w_float * r;
        if max_w_float < 1.0 {
            // Very tall: one column.
            return ((max_tokens - 2) / 2 * cell, cell);
        }
        if max_h_float < 1.0 {
            // Very wide: one row.
            return (cell, (max_tokens - 3) * cell);
        }
        let beta_w = (max_w_float.floor() as usize * cell) as f64 / width as f64;
        let beta_h = (max_h_float.floor() as usize * cell) as f64 / height as f64;
        // Python's min() returns its first argument on a tie, as f64::min does for equal values.
        let beta = beta_w.min(beta_h);
        let patch = self.patch;
        (
            (height as f64 * beta / patch as f64).floor() as usize * patch,
            (width as f64 * beta / patch as f64).floor() as usize * patch,
        )
    }
}

/// An image's canvas size and the token grid the aligner makes from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Grid {
    /// Token rows.
    pub n_llm_h: usize,
    /// Token columns.
    pub n_llm_w: usize,
    /// Canvas height in pixels, a multiple of the patch.
    pub best_height: usize,
    /// Canvas width in pixels, a multiple of the patch.
    pub best_width: usize,
}

impl Grid {
    /// Positions in the span: a start, each row's tokens and its newline, and an end
    /// (`num_image_tokens`).
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.n_llm_h * (self.n_llm_w + 1) + 2
    }

    /// Aligner rows: the positions that carry image features.
    #[must_use]
    pub const fn image_rows(&self) -> usize {
        self.n_llm_h * self.n_llm_w
    }

    /// ViT patch rows and columns.
    #[must_use]
    pub const fn vit(&self, geometry: &Geometry) -> (usize, usize) {
        (
            self.best_height / geometry.patch,
            self.best_width / geometry.patch,
        )
    }
}

/// `plan_image_grid`: the canvas and grid for an image of `width` x `height` pixels.
///
/// `width * height` must be nonzero; [`crate::decode()`] rejects empty images.
#[must_use]
pub fn plan_grid(width: usize, height: usize, geometry: &Geometry) -> Grid {
    let (mut width, mut height) = (width, height);
    let p = geometry.patch;
    let pixels = width * height;
    if 0 < pixels && pixels < geometry.min_pixels {
        let ratio = (geometry.min_pixels as f64 / pixels as f64).powf(0.5);
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let best_width = width.div_ceil(p) * p;
    let best_height = height.div_ceil(p) * p;
    // safe_resize
    let grid = geometry.grid(best_height, best_width);
    if grid.tokens() <= geometry.max_tokens {
        return grid;
    }
    let (best_height, best_width) = geometry.solve(height, width);
    let grid = geometry.grid(best_height, best_width);
    debug_assert!(grid.tokens() <= geometry.max_tokens, "{grid:?}");
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// deepseek-recipe-core 0.2.0 `token_spec.rs`: width, height, token count, best width, best
    /// height, n_llm_h, n_llm_w. The recipe iterates its plan to a fixed point; the reference plans
    /// once, and the two must agree.
    const RECIPE: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
        (978, 479, 302, 980, 490, 12, 24),
        (272, 300, 198, 518, 574, 14, 13),
        (516, 517, 184, 546, 546, 13, 13),
        (1091, 397, 272, 1092, 406, 10, 26),
        (743, 870, 401, 756, 882, 21, 18),
        (906, 740, 416, 910, 742, 18, 22),
        (1107, 873, 590, 1120, 882, 21, 27),
        (493, 243, 202, 784, 392, 10, 19),
        (354, 476, 197, 476, 630, 15, 12),
        (766, 642, 322, 770, 644, 16, 19),
        (972, 672, 402, 980, 672, 16, 24),
        (1167, 486, 350, 1176, 490, 12, 28),
        (1134, 389, 282, 1134, 392, 10, 27),
        (275, 346, 197, 490, 616, 15, 12),
        (1033, 141, 187, 1484, 210, 5, 36),
        (125, 751, 226, 224, 1344, 32, 6),
        (560, 1029, 377, 560, 1036, 25, 14),
        (482, 725, 236, 490, 728, 18, 12),
        (782, 879, 422, 784, 882, 21, 19),
        (434, 272, 200, 700, 434, 11, 17),
        (2, 165, 356, 70, 4942, 118, 2),
        (165, 2, 240, 4942, 70, 2, 118),
        (158, 10000, 762, 126, 7966, 190, 3),
        (10000, 158, 962, 10010, 168, 4, 239),
        (40, 50, 197, 490, 616, 15, 12),
        (50, 40, 194, 616, 490, 12, 15),
        (1, 1, 184, 546, 546, 13, 13),
        (100, 100, 184, 546, 546, 13, 13),
        (16809, 11841, 990, 1540, 1092, 26, 37),
        (1920, 330, 378, 1932, 336, 8, 46),
        (8192, 8192, 994, 1302, 1302, 31, 31),
        (4096, 4096, 994, 1302, 1302, 31, 31),
        (1920, 1080, 968, 1708, 966, 23, 41),
        (1080, 1920, 986, 966, 1708, 41, 23),
        (3024, 4032, 1010, 1134, 1512, 36, 27),
        (640, 480, 206, 644, 490, 12, 16),
    ];

    #[test]
    fn the_plan_matches_the_recipe_vectors() {
        let g = Geometry::V41;
        for &(w, h, tokens, best_w, best_h, n_h, n_w) in RECIPE {
            let grid = plan_grid(w, h, &g);
            assert_eq!(
                grid,
                Grid {
                    n_llm_h: n_h,
                    n_llm_w: n_w,
                    best_height: best_h,
                    best_width: best_w,
                },
                "{w}x{h}"
            );
            assert_eq!(grid.tokens(), tokens, "{w}x{h}");
        }
    }

    #[test]
    fn a_planned_canvas_plans_to_itself() {
        let g = Geometry::V41;
        for &(w, h, ..) in RECIPE {
            let grid = plan_grid(w, h, &g);
            assert_eq!(plan_grid(grid.best_width, grid.best_height, &g), grid);
        }
    }

    #[test]
    fn extreme_aspect_ratios_collapse_to_one_row_or_column() {
        let g = Geometry::V41;
        let wide = plan_grid(8192, 1, &g);
        assert_eq!((wide.n_llm_h, wide.n_llm_w), (1, 1021));
        assert_eq!((wide.best_height, wide.best_width), (42, 42882));
        assert_eq!(wide.tokens(), 1024);
        let tall = plan_grid(1, 8192, &g);
        assert_eq!((tall.n_llm_h, tall.n_llm_w), (511, 1));
        assert_eq!(tall.tokens(), 1024);
    }

    #[test]
    fn no_size_up_to_the_limit_exceeds_the_budget() {
        let g = Geometry::V41;
        let sides = [
            1, 2, 3, 13, 14, 15, 41, 42, 43, 100, 543, 544, 545, 1000, 4097, 8192,
        ];
        for &w in &sides {
            for &h in &sides {
                let grid = plan_grid(w, h, &g);
                assert!(grid.tokens() <= g.max_tokens, "{w}x{h}: {grid:?}");
                assert_eq!(grid.best_width % g.patch, 0);
                assert_eq!(grid.best_height % g.patch, 0);
            }
        }
    }
}
