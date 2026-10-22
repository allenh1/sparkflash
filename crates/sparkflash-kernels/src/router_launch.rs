//! The typed launch capability of the device MoE router (kernels/moe_router_sm121.cu).
//!
//! The router is two launches from one cubin. Stage one (moe_router_sm121) fills
//! scores[tokens][384] with the gate projection and sqrtsoftplus score. Its grid is
//! (ceil(tokens / 8), 12, 1): each of the twelve grid.y blocks owns 32 consecutive experts and
//! writes its own disjoint 32-column slice, so the gate read is spread over twelve SMs and there
//! are no atomics. Stage two (moe_router_topk_sm121) is a second symbol in the same cubin; one
//! thread per token re-scans the 384 scores with the incumbent strict-> selection, normalises the
//! six raw scores and writes ids/weights. A separate launch is the visibility barrier between the
//! two stages.
//!
//! The argument array of stage one is the moe_router_sm121 declaration's parameter list in order -
//! six device pointers, the token count, the trace ring, the layer id and the step sequence. Stage
//! two's array is its own declaration: bias, scores, ids, weights, tokens, trace ring, layer id,
//! step sequence. Both are filled by named fields, so a pointer cannot be passed as a scalar
//! silently.
//!
//! Geometry. Stage one is 256 threads (8 tokens x 32 lanes, one lane per expert in the partition).
//! Stage two is 256 threads with one token per thread; a 130-token batch still fits one block.
//! Both refuse tokens == 0 here rather than handing the driver a zero grid.
//!
//! Tracing. Both kernels take a GpuTraceRing pointer and record under the MOE_MXFP4 scope only
//! when compiled with SPARKFLASH_ENABLE_GPU_TRACE; the shipped cubin is trace-disabled, so this
//! module passes a null ring (0) and the macros are no-ops. Nothing here arms a ring.

use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::router_abi;

/// The stage-one kernel id in the shared driver's enum.
pub const KERNEL_ID: KernelId = KernelId::MoeRouter;
/// The stage-two symbol, loaded through Cuda::load_router_topk.
pub const TOPK_KERNEL: &str = "moe_router_topk_sm121";
/// Stage one: 256 threads, one block per 8 tokens and 32-expert partition.
pub const BLOCK: (u32, u32, u32) = (router_abi::THREADS, 1, 1);
/// Stage two: 256 threads, one token per thread.
pub const TOPK_BLOCK: (u32, u32, u32) = (router_abi::TOPK_THREADS, 1, 1);

/// The stage-one grid for tokens tokens: (N_ROUTED / experts-per-block, ceil(tokens / 8), 1).
///
/// # Errors
/// tokens == 0 is refused rather than launched as a zero-sized grid.
pub fn grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the router routes at least one token, so the grid is not zero-sized".to_owned(),
        );
    }
    let token_blocks = tokens.div_ceil(router_abi::TOKENS_PER_BLOCK);
    Ok((
        router_abi::N_ROUTED / router_abi::EXPERTS_PER_BLOCK,
        token_blocks,
        1,
    ))
}

/// The stage-two grid for tokens tokens: ceil(tokens / 256) one-dimensional blocks.
///
/// # Errors
/// tokens == 0 is refused rather than launched as a zero-sized grid.
pub fn topk_grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the router routes at least one token, so the grid is not zero-sized".to_owned(),
        );
    }
    Ok((tokens.div_ceil(router_abi::TOPK_THREADS), 1, 1))
}

/// One stage-one launch's arguments, in the kernel declaration's order.
///
/// Every device buffer is a Deviceptr the caller allocated with Cuda::alloc; the scalars are the
/// token count and the two trace words. trace_ring is 0 for the shipped, trace-disabled cubin and
/// the ring's device address otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterLaunchParams {
    /// [tokens][hidden] BF16 activations.
    pub x: Deviceptr,
    /// [n_routed][hidden] BF16 gate weight.
    pub gate: Deviceptr,
    /// [n_routed] F32 selection bias (the text or the image-span one).
    pub bias: Deviceptr,
    /// [tokens][top_k] output u32 expert ids (written by stage two).
    pub ids: Deviceptr,
    /// [tokens][top_k] output F32 normalised weights, already scaled by route_scale.
    pub weights: Deviceptr,
    /// [tokens][n_routed] output F32 unbiased scores (written by stage one).
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
    /// The stage-one argument array, in declaration order.
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

    /// The stage-two argument array, in moe_router_topk_sm121 declaration order.
    #[must_use]
    pub fn topk_words(&self) -> [u64; router_abi::TOPK_ARG_WORDS] {
        [
            self.bias.0,
            self.scores.0,
            self.ids.0,
            self.weights.0,
            u64::from(self.tokens),
            self.trace_ring,
            u64::from(self.layer_id),
            self.step_seq,
        ]
    }
}

/// Loads both router kernels from one cubin.
///
/// The two modules are separate handles because the driver's safe loader resolves one symbol per
/// module; both point at the same bytes and both must be freed by the caller. The stage-two symbol
/// is not a KernelId, so it cannot be launched as a router by accident.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain either pinned symbol.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel, Module, Kernel), String> {
    let (router_module, router) = cuda.load_kernel(cubin, KERNEL_ID)?;
    let (topk_module, topk) = cuda.load_router_topk(cubin)?;
    Ok((router_module, router, topk_module, topk))
}

/// Launches stage one, then stage two.
///
/// # Errors
/// Refuses tokens == 0, and returns the driver's error otherwise. The two launches are ordered by
/// the stream they share (stream 0), not by a driver sync: under enable_forward_async_launches the
/// host runs ahead, so stage two reading stage one's scores is a same-stream ordering guarantee.
pub fn launch(
    cuda: &Cuda,
    router: Kernel,
    topk: Kernel,
    params: &RouterLaunchParams,
) -> Result<(), String> {
    cuda.launch(router, grid(params.tokens)?, BLOCK, &params.words())?;
    cuda.launch(
        topk,
        topk_grid(params.tokens)?,
        TOPK_BLOCK,
        &params.topk_words(),
    )
}

/// Runs both stages repeat times and returns (per-iteration ms, batched ms/iteration).
///
/// This is a harness convenience only: the driver syncs each launch, so the numbers include the
/// host launch path between the two stages. Cuda::launch_repeated remains the event-timed
/// single-kernel path.
///
/// # Errors
/// As launch.
pub fn launch_repeated(
    cuda: &Cuda,
    router: Kernel,
    topk: Kernel,
    params: &RouterLaunchParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    let start = std::time::Instant::now();
    for _ in 0..repeat {
        launch(cuda, router, topk, params)?;
    }
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    Ok((
        vec![elapsed / repeat as f64; repeat],
        elapsed / repeat as f64,
    ))
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
        assert_eq!(
            p.topk_words(),
            [0x3000, 0x6000, 0x4000, 0x5000, 8, 0, 7, 11,],
            "stage two: bias, scores, ids, weights, tokens, trace ring, layer id, step"
        );
        assert_eq!(p.topk_words().len(), router_abi::TOPK_ARG_WORDS);
    }

    #[test]
    fn the_geometry_is_one_warp_per_expert() {
        assert_eq!(BLOCK, (256, 1, 1));
        assert_eq!(TOPK_BLOCK, (256, 1, 1));
        assert_eq!(router_abi::EXPERTS_PER_BLOCK, 8);
        assert_eq!(grid(1), Ok((48, 1, 1)));
        assert_eq!(grid(8), Ok((48, 1, 1)));
        assert_eq!(grid(9), Ok((48, 2, 1)));
        assert_eq!(grid(130), Ok((48, 17, 1)));
        assert_eq!(topk_grid(1), Ok((1, 1, 1)));
        assert_eq!(topk_grid(256), Ok((1, 1, 1)));
        assert_eq!(topk_grid(257), Ok((2, 1, 1)));
        assert!(
            grid(0).is_err() && topk_grid(0).is_err(),
            "a zero-token launch is refused before the driver"
        );
    }

    #[test]
    fn the_kernel_id_names_the_pinned_symbol() {
        assert_eq!(KERNEL_ID, KernelId::MoeRouter);
        assert_eq!(router_abi::ROUTER_KERNEL, "moe_router_sm121");
        assert_eq!(TOPK_KERNEL, "moe_router_topk_sm121");
    }
}
