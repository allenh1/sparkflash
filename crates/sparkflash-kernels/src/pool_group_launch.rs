// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `pool_group_f32` (kernels/pool_group_sm121.cu): the compressor's ratio-axis softmax pooling.
//!
//! The kernel is the host rule of `engine/forward/attention.rs:985-1020` moved to the device: per
//! latent element, the maximum over the ratio axis, `exp(score - max)`, the ascending normalised
//! weights, and the ascending weighted sum of the KV rows. The device exponential is a **declared
//! deviation** gated by D54 (`docs/build/s1-result.md` §7.1(3)), not a bit-identical oracle.
//!
//! The launch is one thread per latent element, so the only geometry is `head_dim` (the ratio is a
//! per-thread loop). Spans are the caller's real byte lengths, never a computed pointer.
//!
//! PIN. No cubin bytes or digest live here: registration in `baseline/kernel_baseline.txt` lands with
//! the stage that launches it (the manifest is regenerated, never hand-edited). This module resolves
//! the symbol by name and refuses a launch whose geometry or spans the kernel would read out of
//! bounds on.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// The cubin stem: the staged file's stem and the baseline pin's name.
pub const CUBIN: &str = "pool_group_sm121";

/// The kernel symbol.
pub const KERNEL: &str = "pool_group_f32";

/// The number of argument words one launch fills: kv, score, dest, ratio, head_dim.
pub const ARG_WORDS: usize = 5;

/// Threads per block; the kernel source states the same number.
pub const THREADS: u32 = 128;

/// The largest grid.x a launch forms (the kernel has no grid-stride loop: one thread per element).
pub const MAX_BLOCKS: u32 = 65_535;

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolGroupParams {
    /// `[ratio][head_dim]` f32 rows to pool (the compressor's f32 KV latent).
    pub kv: Deviceptr,
    /// `[ratio][head_dim]` f32 gate scores (the compressor's `wgate` output, f32).
    pub score: Deviceptr,
    /// `[head_dim]` f32 pooled row.
    pub dest: Deviceptr,
    /// Rows in the group (the layer's compress ratio; 1, 2 or 4 released).
    pub ratio: u32,
    /// Columns per row (the MLA latent width, 512 released).
    pub head_dim: u32,
}

impl PoolGroupParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.kv.0,
            self.score.0,
            self.dest.0,
            u64::from(self.ratio),
            u64::from(self.head_dim),
        ]
    }

    /// Refuses a geometry the kernel would read or write out of bounds on, or index with a
    /// non-positive extent.
    ///
    /// # Errors
    /// A zero ratio or zero head width, a width past the declaration's `int`, or a width whose blocks
    /// do not fit the grid.
    pub fn validate(&self) -> Result<(), String> {
        if self.ratio == 0 {
            return Err("pool_group: ratio is zero, so there is no row to pool".to_owned());
        }
        if self.head_dim == 0 {
            return Err("pool_group: head_dim is zero, so there is no element to write".to_owned());
        }
        if self.head_dim > i32::MAX as u32 {
            return Err(format!(
                "pool_group: {} columns do not fit the declaration's int",
                self.head_dim
            ));
        }
        self.grid().map(|_| ())
    }

    /// The launch grid: one thread per latent element.
    ///
    /// # Errors
    /// A block count past [MAX_BLOCKS], which would leave the tail unpooled (the kernel is not
    /// grid-strided).
    pub fn grid(&self) -> Result<(u32, u32, u32), String> {
        let blocks = self.head_dim.div_ceil(THREADS);
        if blocks > MAX_BLOCKS {
            return Err(format!(
                "pool_group: {} blocks do not fit the grid",
                self.head_dim.div_ceil(THREADS)
            ));
        }
        Ok((blocks, 1, 1))
    }

    /// Refuses a span shorter than the launch reads or writes.
    ///
    /// # Errors
    /// Any operand span shorter than `ratio * head_dim` f32 values (the destination: `head_dim`).
    pub fn check_spans(&self, spans: &PoolGroupSpans) -> Result<(), String> {
        let rows = (self.ratio as usize)
            .checked_mul(self.head_dim as usize)
            .ok_or_else(|| "pool_group: the row count overflows".to_owned())?;
        let need_rows = rows
            .checked_mul(4)
            .ok_or_else(|| "pool_group: the row bytes overflow".to_owned())?;
        let need_dest = (self.head_dim as usize) * 4;
        for (name, bytes, need) in [
            ("kv", spans.kv_bytes, need_rows),
            ("score", spans.score_bytes, need_rows),
            ("dest", spans.dest_bytes, need_dest),
        ] {
            if bytes < need {
                return Err(format!(
                    "pool_group: the {name} span is {bytes} B, the {ratio} x {head_dim} launch needs {need} B",
                    ratio = self.ratio,
                    head_dim = self.head_dim
                ));
            }
        }
        Ok(())
    }
}

/// The three spans one launch reads and writes, in bytes: the caller's real lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // the span names carry the geometry; the owner's cleanup removed the prose
pub struct PoolGroupSpans {
    pub kv_bytes: usize,
    pub score_bytes: usize,
    pub dest_bytes: usize,
}

/// Loads the kernel from its cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain `pool_group_f32`.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_pool_group(cubin)
}

/// Launches the pooling once: geometry and span guards, then the driver call.
///
/// # Errors
/// [PoolGroupParams::validate]'s refusal, [PoolGroupParams::check_spans]' refusal, or the driver's
/// launch error.
pub fn launch(
    cuda: &Cuda,
    kernel: Kernel,
    params: &PoolGroupParams,
    spans: &PoolGroupSpans,
) -> Result<(), String> {
    params.validate()?;
    params.check_spans(spans)?;
    let grid = params.grid()?;
    cuda.launch(kernel, grid, (THREADS, 1, 1), &params.words())
}

/// Launches `k_norm_f32_to_bf16(x, weight, out, cols, eps)`.
pub fn launch_k_norm(
    cuda: &Cuda,
    kernel: Kernel,
    x: Deviceptr,
    weight: Deviceptr,
    out: Deviceptr,
    cols: u32,
    eps: f32,
) -> Result<(), String> {
    let args = [
        x.0,
        weight.0,
        out.0,
        u64::from(cols),
        u64::from(eps.to_bits()),
    ];
    cuda.launch(kernel, (1, 1, 1), (1, 1, 1), &args)
}

/// Launches `fp4_round_trip_bf16_to_f32(src, dst, blocks)`.
pub fn launch_fp4_round_trip(
    cuda: &Cuda,
    kernel: Kernel,
    src: Deviceptr,
    dst: Deviceptr,
    blocks: u32,
) -> Result<(), String> {
    let grid_x = blocks.div_ceil(THREADS);
    let args = [src.0, dst.0, u64::from(blocks)];
    cuda.launch(kernel, (grid_x, 1, 1), (THREADS, 1, 1), &args)
}

/// Launches `scale_index_weights_bf16_to_f32(src, dst, factor, count)`.
pub fn launch_scale_index_weights(
    cuda: &Cuda,
    kernel: Kernel,
    src: Deviceptr,
    dst: Deviceptr,
    factor: f32,
    count: u32,
) -> Result<(), String> {
    let grid_x = count.div_ceil(THREADS);
    let args = [src.0, dst.0, u64::from(factor.to_bits()), u64::from(count)];
    cuda.launch(kernel, (grid_x, 1, 1), (THREADS, 1, 1), &args)
}

/// Launches `hc_pre_collapse_bf16(pre_mix, stream, out_bf16, certified_count, hc_mult, dim)`.
///
/// Eight arguments mirror the kernel's own parameter list; grouping them into a struct would only
/// move the names, so the lint is allowed here the way the other multi-operand launches do it.
#[allow(clippy::too_many_arguments)]
pub fn launch_hc_pre_collapse(
    cuda: &Cuda,
    kernel: Kernel,
    pre_mix: Deviceptr,
    stream: Deviceptr,
    out_bf16: Deviceptr,
    certified_count: Deviceptr,
    hc_mult: u32,
    dim: u32,
) -> Result<(), String> {
    let grid_x = dim.div_ceil(THREADS);
    let args = [
        pre_mix.0,
        stream.0,
        out_bf16.0,
        certified_count.0,
        u64::from(hc_mult),
        u64::from(dim),
    ];
    cuda.launch(kernel, (grid_x, 1, 1), (THREADS, 1, 1), &args)
}

/// The CPU twin of `pool_group_f32`: the host rule's four steps in its order, for the tests that pin
/// the reduction order and the empty-group rule. Not used by the device path.
pub fn pool_group_ref(kv: &[f32], score: &[f32], dest: &mut [f32], ratio: usize, head_dim: usize) {
    for element in 0..head_dim {
        let mut max = f32::NEG_INFINITY;
        for row in 0..ratio {
            max = max.max(score[row * head_dim + element]);
        }
        if !max.is_finite() {
            dest[element] = 0.0;
            continue;
        }
        let mut total = 0.0f32;
        for row in 0..ratio {
            total += (score[row * head_dim + element] - max).exp();
        }
        let mut acc = 0.0f32;
        for row in 0..ratio {
            let weight = (score[row * head_dim + element] - max).exp();
            let normalised = if total > 0.0 { weight / total } else { 0.0 };
            acc += normalised * kv[row * head_dim + element];
        }
        dest[element] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_is_one_thread_per_element_and_refuses_a_ratio_of_zero() {
        let params = PoolGroupParams {
            kv: Deviceptr(0x1000),
            score: Deviceptr(0x2000),
            dest: Deviceptr(0x3000),
            ratio: 2,
            head_dim: 512,
        };
        assert_eq!(params.words().len(), ARG_WORDS);
        assert_eq!(params.grid().expect("512 columns"), (4, 1, 1));
        let spans = PoolGroupSpans {
            kv_bytes: 2 * 512 * 4,
            score_bytes: 2 * 512 * 4,
            dest_bytes: 512 * 4,
        };
        params
            .check_spans(&spans)
            .expect("the released ratio-2 group");
        let short = PoolGroupSpans {
            kv_bytes: 512 * 4,
            ..spans
        };
        assert!(params.check_spans(&short).is_err(), "one row is not two");
        let zero = PoolGroupParams { ratio: 0, ..params };
        assert!(zero.validate().is_err());
    }

    #[test]
    fn the_twin_weights_the_rows_by_their_softmax_and_zeroes_an_empty_group() {
        // ratio 2, head_dim 2: element 0 prefers row 1, element 1 prefers row 0.
        let kv = [1.0f32, 10.0, 3.0, 20.0];
        let score = [0.0f32, 2.0, 2.0, 0.0];
        let mut dest = [0.0f32; 2];
        pool_group_ref(&kv, &score, &mut dest, 2, 2);
        // exp(2) dominates exp(0): the weights are exp(0)/Z and exp(2)/Z with Z = 1 + e^2. The
        // tolerance is 1e-4 because the twin sums in f32 while these expectations use f64 exp.
        let z = 1.0f64 + (2.0f64).exp();
        let low = 1.0 / z;
        let high = (2.0f64).exp() / z;
        assert!((dest[0] as f64 - (low * 1.0 + high * 3.0)).abs() < 1e-4);
        // Element 1 sees the scores the other way round (row 0 scores 2, row 1 scores 0), so its
        // high weight is row 0's.
        let z2 = 1.0f64 + (-2.0f64).exp();
        let high2 = 1.0 / z2;
        let low2 = (-2.0f64).exp() / z2;
        assert!((dest[1] as f64 - (high2 * 10.0 + low2 * 20.0)).abs() < 1e-4);

        // An all-`-inf` group writes zero, exactly as the host rule does.
        let mut empty = [7.0f32; 2];
        pool_group_ref(&kv, &[f32::NEG_INFINITY; 4], &mut empty, 2, 2);
        assert_eq!(empty, [0.0, 0.0]);
    }

    #[test]
    fn the_baseline_manifest_pins_pool_group_sm121() {
        let pin = crate::baseline::pin(CUBIN)
            .expect("baseline parses")
            .expect("pool_group_sm121 is pinned");
        assert_eq!(pin.cubin_bytes, 54_312);
        assert_eq!(
            pin.cubin_sha256,
            "ed16a3a00d1150ea7b32de59c9dc7f44011faa8a8a45cb3ea518c1a84c6175b7"
        );
        assert_eq!(pin.ptx_membar_lines, 0);
    }
}
