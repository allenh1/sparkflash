//! The typed launch capability of the device MoE router (kernels/moe_router_sm121.cu).
//!
//! The argument array is the `moe_router_sm121` declaration's parameter list in order - six device
//! pointers, the token count, the trace ring, the layer id and the step sequence - so the only way
//! to fill a word wrongly is to fill a field wrongly, and the fields are named.
//!
//! Geometry. One thread block routes [crate::router_abi::TOKENS_PER_BLOCK] tokens, so the grid is
//! `ceil(tokens / 8)` blocks of [crate::router_abi::THREADS] threads and a short last block simply
//! has fewer live tokens (the kernel guards them). The launch refuses `tokens == 0` here rather
//! than handing the driver a zero grid: nothing about a zero-token routing is well defined, and a
//! driver error is a worse diagnostic than the caller's own shape check.
//!
//! Tracing. The kernel takes a `GpuTraceRing *` and records under the MOE_MXFP4 scope only when it
//! was compiled with SPARKFLASH_ENABLE_GPU_TRACE; the shipped cubin is the trace-disabled build, so
//! this module passes a null ring ([RouterLaunchParams::trace_ring] 0) and the kernel's macros are
//! no-ops. Nothing here arms a ring; a caller that wants telemetry passes the ring address through
//! [RouterLaunchParams::trace_ring] explicitly and the same argument order carries it.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::router_abi;

/// The kernel id in the shared driver's enum, named here so a caller need not import the driver.
pub const KERNEL_ID: KernelId = KernelId::MoeRouter;

/// The one piece of launch geometry the kernel fixes: 256 threads, one block.
pub const BLOCK: (u32, u32, u32) = (router_abi::THREADS, 1, 1);

/// The grid for `tokens` tokens: `ceil(tokens / TOKENS_PER_BLOCK)` blocks.
///
/// # Errors
/// `tokens == 0` is refused rather than launched as a zero-sized grid.
pub fn grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the router routes at least one token, so the grid is not zero-sized".to_owned(),
        );
    }
    let blocks = tokens.div_ceil(router_abi::TOKENS_PER_BLOCK);
    Ok((blocks, 1, 1))
}

/// One launch's arguments, in the kernel declaration's order.
///
/// Every device buffer is a [Deviceptr] the caller allocated with [Cuda::alloc]; the scalars are
/// the token count and the two trace words. `trace_ring` is 0 for the shipped, trace-disabled
/// cubin and the ring's device address otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterLaunchParams {
    /// `[tokens][hidden]` BF16 activations.
    pub x: Deviceptr,
    /// `[n_routed][hidden]` BF16 gate weight.
    pub gate: Deviceptr,
    /// `[n_routed]` F32 selection bias (the text or the image-span one).
    pub bias: Deviceptr,
    /// `[tokens][top_k]` output u32 expert ids.
    pub ids: Deviceptr,
    /// `[tokens][top_k]` output F32 normalised weights (already scaled by route_scale).
    pub weights: Deviceptr,
    /// `[tokens][n_routed]` output F32 unbiased scores.
    pub scores: Deviceptr,
    /// Tokens in the launch.
    pub tokens: u32,
    /// The trace ring's device address, or 0 for the trace-disabled cubin.
    pub trace_ring: u64,
    /// The layer id the trace event records.
    pub layer_id: u32,
    /// The step sequence the trace event records.
    pub step_seq: u64,
}

impl RouterLaunchParams {
    /// The argument array a launch passes to the driver, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; router_abi::ARG_WORDS] {
        [
            self.x.0,
            self.gate.0,
            self.bias.0,
            self.ids.0,
            self.weights.0,
            self.scores.0,
            u64::from(self.tokens),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Loads the router kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain `moe_router_sm121` (the failure a
/// wrong kernel name or a stale cubin produces).
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the router once.
///
/// # Errors
/// Refuses `tokens == 0`, and returns the driver's error for everything else.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &RouterLaunchParams) -> Result<(), String> {
    let grid = grid(params.tokens)?;
    cuda.launch(kernel, grid, BLOCK, &params.words())
}

/// Launches the router `repeat` times and returns `(per-launch ms, batched ms/launch)`.
///
/// # Errors
/// As [launch].
pub fn launch_repeated(
    cuda: &Cuda,
    kernel: Kernel,
    params: &RouterLaunchParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    let grid = grid(params.tokens)?;
    cuda.launch_repeated(kernel, grid, BLOCK, &params.words(), repeat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(tokens: u32) -> RouterLaunchParams {
        RouterLaunchParams {
            x: Deviceptr(0x1000),
            gate: Deviceptr(0x2000),
            bias: Deviceptr(0x3000),
            ids: Deviceptr(0x4000),
            weights: Deviceptr(0x5000),
            scores: Deviceptr(0x6000),
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
            [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 8, 0, 7, 11,],
            "six pointers, tokens, trace ring, layer id, step sequence"
        );
        assert_eq!(p.words().len(), router_abi::ARG_WORDS);
    }

    #[test]
    fn the_geometry_is_blocks_of_eight_tokens() {
        assert_eq!(BLOCK, (256, 1, 1));
        assert_eq!(grid(1), Ok((1, 1, 1)));
        assert_eq!(grid(8), Ok((1, 1, 1)));
        assert_eq!(grid(9), Ok((2, 1, 1)));
        assert_eq!(grid(130), Ok((17, 1, 1)));
        assert!(
            grid(0).is_err(),
            "a zero-token launch is refused before the driver"
        );
    }

    #[test]
    fn the_kernel_id_names_the_pinned_symbol() {
        // The symbol is private in cuda_driver; this test pins the neighbouring fact that the enum
        // variant exists and is the one this module names, so a rename fails here first.
        assert_eq!(KERNEL_ID, KernelId::MoeRouter);
        assert_eq!(router_abi::ROUTER_KERNEL, "moe_router_sm121");
    }
}
