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

//! The typed launch capability of the fused split-K routed gate/up kernel
//! (pending_kernels/routed_gate_up_swiglu_splitk_sm121.cu).
//!
//! The argument array is the kernel declaration's parameter list in order - eight device pointers
//! (the activation codes and scales, w1 and w3 packed weights and their scales, the routed ids and
//! the router's weights), the two outputs, the token count, the SwiGLU limit and the three trace
//! words - so the only way to fill a word wrongly is to fill a field wrongly, and every field is
//! named. [ARG_WORDS] is the arity pin: [crate::routed_gate_up_swiglu_splitk_abi] parses the same
//! count out of the declaration, and both must agree with the words array.
//!
//! Geometry. One CTA computes one token's 32-column output tile for ALL six selected experts and
//! both projections, with K split across the CTA's eight warps: grid
//! `(tokens, ceil(640 / 32) = 20)` blocks of [abi::THREADS] threads. A decode token is 20 CTAs -
//! a launch-count fusion (12 launches per layer become 4), not a bandwidth-saturating geometry; the
//! GPU verification plan's first experiment is the occupancy sweep that trades CTAs per token
//! against the fused epilogue's 32-column MX block.
//!
//! Default-off (D51 amendment A, rule 5). [launch] refuses to hand the driver the launch unless the
//! process set `SPARKFLASH_ENABLE_MOE_SPLITK=1`; the check is a pure function of the variable's
//! value ([gate_armed_value]) so it is testable without touching the environment. A performance
//! kernel is wired default-on only after its tolerance row and its D41 oracle-suite row are
//! recorded, which is the Lead's call and not this crate's.
//!
//! Expert ids are the staged-slice indices the incumbent chain passes ([crate::routed_gate_up_launch]
//! documents the same contract): the caller stages the selected experts' per-rank w1/w3 slices in
//! slot order and passes that slot index, which the kernel uses to index the staged buffers.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::routed_gate_up_swiglu_splitk_abi as abi;

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::RoutedGateUpSwigluSplitK;

/// The cubin symbol, which is also the future manifest key.
pub const KERNEL: &str = abi::KERNEL;

/// The one piece of launch geometry the kernel fixes: 256 threads, one block.
pub const BLOCK: (u32, u32, u32) = (abi::THREADS, 1, 1);

/// The argument array a launch fills, in words: the arity pin.
pub const ARG_WORDS: usize = abi::ARG_WORDS;

/// The environment variable that arms the default-off kernel.
pub const GATE_ENV: &str = "SPARKFLASH_ENABLE_MOE_SPLITK";

/// The grid for `tokens` tokens: `(tokens, TILES, 1)`.
///
/// # Errors
/// `tokens == 0` is refused rather than launched as a zero-sized grid.
pub fn grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the fused gate/up kernel runs at least one token, so the grid is not zero-sized"
                .to_owned(),
        );
    }
    Ok((tokens, abi::TILES, 1))
}

/// One launch's arguments, in the kernel declaration's order.
///
/// Every device buffer is a [Deviceptr] the caller allocated with [Cuda::alloc]; `limit` is the
/// released `swiglu_limit` (10.0); `trace_ring` is 0 for the shipped, trace-disabled cubin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateUpSwigluSplitKParams {
    /// `[tokens][hidden]` E4M3 block-input bytes.
    pub act_fp8: Deviceptr,
    /// `[tokens][hidden / 32]` ue8m0 activation scales.
    pub act_scales: Deviceptr,
    /// `[experts][width][hidden / 2]` packed E2M1 gate (w1) weight bytes.
    pub w1_fp4: Deviceptr,
    /// `[experts][width][hidden / 32]` ue8m0 gate scales.
    pub w1_scales: Deviceptr,
    /// `[experts][width][hidden / 2]` packed E2M1 up (w3) weight bytes.
    pub w3_fp4: Deviceptr,
    /// `[experts][width][hidden / 32]` ue8m0 up scales.
    pub w3_scales: Deviceptr,
    /// `[tokens * top_k]` u32 staged expert-slot ids from the router.
    pub expert_ids: Deviceptr,
    /// `[tokens * top_k]` f32 routing weights.
    pub routing_weights: Deviceptr,
    /// `[tokens * top_k][width_padded]` E4M3 activation codes.
    pub out_fp8: Deviceptr,
    /// `[tokens * top_k][width_padded / 32]` ue8m0 activation scales.
    pub out_scales: Deviceptr,
    /// Tokens in the launch.
    pub tokens: u32,
    /// The SwiGLU clamp limit (10.0 in the released config).
    pub limit: f32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event records.
    pub layer_id: u32,
    /// The step sequence the trace event records.
    pub step_seq: u64,
}

impl GateUpSwigluSplitKParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.act_fp8.0,
            self.act_scales.0,
            self.w1_fp4.0,
            self.w1_scales.0,
            self.w3_fp4.0,
            self.w3_scales.0,
            self.expert_ids.0,
            self.routing_weights.0,
            self.out_fp8.0,
            self.out_scales.0,
            u64::from(self.tokens),
            // A 4-byte float argument is passed as a word whose low half is the f32 bit pattern, the
            // same convention the incumbent weighted-SwiGLU launch site uses.
            u64::from(self.limit.to_bits()),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Whether `value` arms the kernel: only the literal `"1"` does.
#[must_use]
pub fn gate_armed_value(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

/// Whether this process armed the default-off kernel (`SPARKFLASH_ENABLE_MOE_SPLITK=1`).
#[must_use]
pub fn gate_armed() -> bool {
    gate_armed_value(std::env::var(GATE_ENV).ok().as_deref())
}

/// Loads the fused kernel from a cubin, resolving the symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain the symbol.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the fused projection once.
///
/// # Errors
/// Refuses `tokens == 0`, refuses while the kernel is not armed ([GATE_ENV]), and returns the
/// driver's error for everything else.
pub fn launch(
    cuda: &Cuda,
    kernel: Kernel,
    params: &GateUpSwigluSplitKParams,
) -> Result<(), String> {
    require_armed()?;
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
    params: &GateUpSwigluSplitKParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    require_armed()?;
    let grid = grid(params.tokens)?;
    cuda.launch_repeated(kernel, grid, BLOCK, &params.words(), repeat)
}

fn require_armed() -> Result<(), String> {
    if gate_armed() {
        Ok(())
    } else {
        Err(format!(
            "{KERNEL} is default-off (D51 amendment A, rule 5): set {GATE_ENV}=1 to arm it, and only after its tolerance row and its D41 oracle-suite row are recorded"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(tokens: u32) -> GateUpSwigluSplitKParams {
        GateUpSwigluSplitKParams {
            act_fp8: Deviceptr(0x1000),
            act_scales: Deviceptr(0x2000),
            w1_fp4: Deviceptr(0x3000),
            w1_scales: Deviceptr(0x4000),
            w3_fp4: Deviceptr(0x5000),
            w3_scales: Deviceptr(0x6000),
            expert_ids: Deviceptr(0x7000),
            routing_weights: Deviceptr(0x8000),
            out_fp8: Deviceptr(0x9000),
            out_scales: Deviceptr(0xA000),
            tokens,
            limit: 10.0,
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
            [
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                0x5000,
                0x6000,
                0x7000,
                0x8000,
                0x9000,
                0xA000,
                8,
                u64::from(10.0f32.to_bits()),
                0,
                7,
                11,
            ],
            "eight operands, two outputs, tokens, limit, trace ring, layer id, step sequence"
        );
        assert_eq!(p.words().len(), ARG_WORDS);
        assert_eq!(p.words().len(), abi::PARAMS.len());
    }

    #[test]
    fn the_geometry_is_a_block_per_token_tile() {
        assert_eq!(BLOCK, (256, 1, 1));
        assert_eq!(grid(1), Ok((1, 20, 1)));
        assert_eq!(grid(8), Ok((8, 20, 1)));
        assert!(grid(0).is_err(), "a zero-token launch is refused");
    }

    #[test]
    fn the_kernel_id_names_the_new_symbol() {
        assert_eq!(KERNEL_ID, KernelId::RoutedGateUpSwigluSplitK);
        assert_eq!(KERNEL, "routed_gate_up_swiglu_splitk_sm121");
        assert_ne!(KERNEL, crate::cuda_driver::FLASH_MLA_CUBIN);
    }

    #[test]
    fn the_default_off_gate_is_only_the_literal_one() {
        assert!(!gate_armed_value(None));
        assert!(!gate_armed_value(Some("")));
        assert!(!gate_armed_value(Some("0")));
        assert!(!gate_armed_value(Some("true")));
        assert!(gate_armed_value(Some("1")));
        assert_eq!(GATE_ENV, "SPARKFLASH_ENABLE_MOE_SPLITK");
    }
}
