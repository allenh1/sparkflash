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

//! The typed launch capability of the routed-expert gate/up projection
//! (kernels/routed_gate_up_sm121.cu).
//!
//! The argument array is the `routed_gate_up_sm121` declaration's parameter list in order - six
//! device pointers (the two A buffers, the two B buffers, the expert ids and the output), the token
//! count and the two trace words - so the only way to fill a word wrongly is to fill a field
//! wrongly, and the fields are named.
//!
//! Geometry. One thread block computes one routed (token, slot) row's 32-column output tile with 32
//! reducer columns x 8 parallel K groups, so the grid is `(tokens * top_k, ceil(640 / 32) = 20)`
//! blocks of [crate::routed_gate_up_abi::THREADS] threads: 120 blocks for one token's six slots, and
//! 780 x 20 blocks for a 130-token batch. The twentieth tile covers columns 608..639, of which the
//! kernel computes none (576..639 are padding) and writes all 32 as exactly zero; it guards both the
//! routed-row and the column range, so a full batch needs no padding logic in the caller. The
//! launcher still refuses `tokens == 0` rather than handing the driver a zero grid. Expert ids are a caller contract the launcher cannot validate (the argument list has
//! no expert count): the twin panics on an out-of-range id, and the harness generates ids inside
//! the resident set.
//!
//! Tracing. The kernel takes a `GpuTraceRing *` and records under TRACE_SCOPE_MOE_MXFP4 only when
//! it was compiled with SPARKFLASH_ENABLE_GPU_TRACE; the shipped cubin is the trace-disabled build, so
//! this module passes a null ring ([GateUpLaunchParams::trace_ring] 0) and the kernel's macros are
//! no-ops. Nothing here arms a ring.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::routed_gate_up_abi;

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::RoutedGateUp;

/// The one piece of launch geometry the kernel fixes: 256 threads, one block.
pub const BLOCK: (u32, u32, u32) = (routed_gate_up_abi::THREADS, 1, 1);

/// The number of routed rows a token batch produces: `tokens * top_k`.
#[must_use]
pub fn routed_rows(tokens: u32) -> u32 {
    tokens * routed_gate_up_abi::TOP_K
}

/// The grid for `tokens` tokens: `(tokens * top_k, TILES, 1)`.
///
/// # Errors
/// `tokens == 0` is refused rather than launched as a zero-sized grid.
pub fn grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the gate/up projection runs at least one token, so the grid is not zero-sized"
                .to_owned(),
        );
    }
    Ok((routed_rows(tokens), routed_gate_up_abi::TILES, 1))
}

/// One launch's arguments, in the kernel declaration's order.
///
/// Every device buffer is a [Deviceptr] the caller allocated with [Cuda::alloc]; the scalars are the
/// token count and the two trace words. `trace_ring` is 0 for the shipped, trace-disabled cubin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateUpLaunchParams {
    /// `[tokens][hidden]` E4M3 block-input bytes.
    pub act_fp8: Deviceptr,
    /// `[tokens][hidden / 32]` ue8m0 activation scales.
    pub act_scales: Deviceptr,
    /// `[experts][width][hidden / 2]` packed E2M1 weight bytes for ONE projection (w1 or w3).
    pub w_fp4: Deviceptr,
    /// `[experts][width][hidden / 32]` ue8m0 weight scales.
    pub w_scales: Deviceptr,
    /// `[tokens * top_k]` u32 expert ids from the router.
    pub expert_ids: Deviceptr,
    /// `[tokens * top_k][width_padded]` BF16 output.
    pub out: Deviceptr,
    /// Tokens in the launch.
    pub tokens: u32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event records.
    pub layer_id: u32,
    /// The step sequence the trace event records.
    pub step_seq: u64,
}

impl GateUpLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; routed_gate_up_abi::ARG_WORDS] {
        [
            self.act_fp8.0,
            self.act_scales.0,
            self.w_fp4.0,
            self.w_scales.0,
            self.expert_ids.0,
            self.out.0,
            u64::from(self.tokens),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Loads the gate/up kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain `routed_gate_up_sm121`.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the projection once.
///
/// # Errors
/// Refuses `tokens == 0`, and returns the driver's error for everything else.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &GateUpLaunchParams) -> Result<(), String> {
    let grid = grid(params.tokens)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

/// Launches the projection `repeat` times and returns `(per-launch ms, batched ms/launch)`.
///
/// # Errors
/// As [launch].
pub fn launch_repeated(
    cuda: &Cuda,
    kernel: Kernel,
    params: &GateUpLaunchParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    let grid = grid(params.tokens)?;
    cuda.launch_repeated(kernel, grid, BLOCK, &params.words(), repeat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(tokens: u32) -> GateUpLaunchParams {
        GateUpLaunchParams {
            act_fp8: Deviceptr(0x1000),
            act_scales: Deviceptr(0x2000),
            w_fp4: Deviceptr(0x3000),
            w_scales: Deviceptr(0x4000),
            expert_ids: Deviceptr(0x5000),
            out: Deviceptr(0x6000),
            tokens,
            trace_ring: 0,
            layer_id: 7,
            step_seq: 11,
        }
    }

    #[test]
    fn the_argument_words_are_the_declaration_order() {
        let p = params(8);
        assert_eq!(
            p.words(),
            [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 8, 0, 7, 11],
            "four operands, tokens, trace ring, layer id, step sequence"
        );
        assert_eq!(p.words().len(), routed_gate_up_abi::ARG_WORDS);
    }

    #[test]
    fn the_geometry_is_a_block_per_routed_row_tile() {
        assert_eq!(BLOCK, (256, 1, 1));
        assert_eq!(grid(1), Ok((6, 20, 1)));
        assert_eq!(grid(8), Ok((48, 20, 1)));
        assert_eq!(grid(130), Ok((780, 20, 1)));
        assert_eq!(routed_rows(8), 48);
        assert!(
            grid(0).is_err(),
            "a zero-token launch is refused before the driver"
        );
    }

    #[test]
    fn the_kernel_id_names_the_pinned_symbol() {
        assert_eq!(KERNEL_ID, KernelId::RoutedGateUp);
        assert_eq!(routed_gate_up_abi::GATE_UP_KERNEL, "routed_gate_up_sm121");
    }
}
