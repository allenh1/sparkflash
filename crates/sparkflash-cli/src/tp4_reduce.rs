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

//! The run's NCCL communicator, opened once for the forward loop and handed to the layer port.
//!
//! This is the bootstrap half of the collectives: the layer port's per-collective call is
//! sparkflash_comm::device_reduce::reduce(handle, stream, in, out, count), and the handle it passes
//! is the RunReduce this module opens. Nothing here is per layer and nothing here is per position.
//!
//! ## The rendezvous
//!
//! Every rank loads the NCCL library FIRST and records whether it succeeded; rank 0 asks for the
//! unique id; every rank then exchanges id and configuration in ONE all-gather over the forward
//! loop's OWN TCP transport - the TcpTransport the loop already holds on its separate port range
//! (--comm-base-port + 100), the one the per-layer reduce uses. The wire is one all-gather of 70
//! little-endian u16 words per rank: 64 opaque id words (128 id bytes), then the agreement words -
//! a flags word (library loaded before the gather, id supplied), the arm word, the native-rounding
//! word, the behaviour-toggle word and a 32-bit plan digest - on sequence BOOTSTRAP_GATHER_SEQ,
//! which sits in its own far-above block so a bootstrap frame cannot be mistaken for a layer reduce
//! frame on a reused transport.
//!
//! The agreement words are checked before RunReduce::open on every rank. A rank whose library load
//! failed, an arm/toggle/rounding/digest difference, or a non-rank-0 id is therefore refused
//! identically on all four ranks BEFORE any rank is inside NCCL - the alternative is a rank that
//! lacks the library failing locally while its peers wait inside ncclCommInitRank, and a per-rank
//! rounding toggle changing numerics with no failure at all.
//!
//! After open, warm-up and the setup sync, one bounded SETUP BARRIER on the same transport makes a
//! rank-local failure there abort everyone at the transport's own deadline: a rank that failed
//! returns Err and never reaches the barrier, so its peers wait in the barrier rather than entering
//! the timed region and hanging to the window's outer timeout. The barrier is setup, once, outside
//! the timed region - it is not the in-loop abort handshake rule 2 forbids.
//!
//! ## Two rules, stated here because they are the function's contract
//!
//! 1. ANY BOOTSTRAP ERROR ON ANY RANK ABORTS THE WHOLE RUN. First error wins: no rank proceeds alone,
//!    there is no host fallback transport and no half-open communicator.
//!    * Gather-phase failures (the id/agreement gather itself) are aborts BEFORE any rank enters
//!      NCCL: the frame never completes, every rank's gather returns an error, and no rank can be
//!      sitting inside a collective while its peers are not. The agreement words extend that to
//!      per-rank configuration: a library/config mismatch is a deterministic refusal on all ranks,
//!      not a rank-local error.
//!    * Post-gather failures - library/init/arm failure at open, a warm-up enqueue failure, a setup
//!      sync failure - are rank-local at the moment they happen, so the SETUP BARRIER after the sync
//!      turns them into an all-rank abort at the transport deadline. The failing rank returns Err
//!      and the surviving ranks block in the barrier; they do not proceed into the timed region.
//! 2. NO IN-LOOP ABORT HANDSHAKE. A stage refusal must be deterministic from the plan and the kernel
//!    set, identical on all four ranks, or it must happen at open. A synchronous abort exchange
//!    before each collective would land in the timed region, and a non-synchronous poll does not
//!    close the A-polls-clean/B-fails/A-enqueues race: it moves a window, it does not close it. The
//!    replacement invariant is that unanticipated aborts are made VOID, not recoverable - the
//!    window's outer timeout and the on-node kill are the backstop, and a row that ends that way is
//!    never quoted.
//!
//! ## Timing
//!
//! Everything this function does is setup: the library load, the id/agreement gather, the
//! communicator's device buffers at init, three warm-up collectives, ONE sync that completes them and
//! the setup barrier. It must be called before layer 0 and outside the timed region; a warm-up inside
//! a timed step would be a fabricated number. The sync is a CUDA-context operation (the comm crate
//! has no CUDA dependency), so this module owns it: it opens the same primary context the run's
//! kernels use and lets cuCtxSynchronize drain the stream the warm-ups were enqueued on.
//!
//! WHICH CONTEXT THE COMMUNICATOR IS INITIALISED AGAINST. RunReduce::open (ncclCommInitRank) binds
//! the communicator to the CUDA context that is CURRENT on the calling thread at init time. This
//! function is called from inside the arena epoch, where tp4_resident holds the device's primary
//! context current, and the sync it then performs is Cuda::load + cuCtxSynchronize - Cuda::load
//! retains that same primary context (it does not create a second one) and cuCtxSynchronize drains
//! it. The stream the loop launches on (id 0, the legacy default stream) belongs to that same
//! primary context, so the communicator's collectives and the kernels that produce their operands
//! are on one context and one stream: the classic "different context, works until it deadlocks"
//! failure cannot occur through this door. A future caller that switches the process to a private
//! context must open the communicator from inside that context, not before it.
//!
//! ## The caller's contract
//!
//! * transport: the run's TCP transport, connected on all four ranks before this call.
//! * world/rank: the run's topology; every rank calls this exactly once with the same world.
//! * stream: the CUDA stream the loop's launches and every later reduce use (0 is the default
//!   stream, which is what the current launch path uses).
//! * scratch: a device address of at least payload bf16 values on the current device, writable; the
//!   warm-up does three in-place reduces over it and never reads it afterwards.
//! * payload: bf16 ELEMENTS of one per-layer collective (the plan's REDUCE_ELEMENTS: 5,120 in the
//!   released config, 10,240 B per call for both per-layer collectives).
//! * config: the per-run agreement words every rank must match; the caller derives them from the
//!   same parsed run options the loop uses (never from a second env parser).
//!
//! The returned RunReduce is owned by the run, used for BOTH per-layer collectives, and closed once
//! at teardown (dropping it; close(handle) must not be called on the value returned here).

use sparkflash_comm::device_reduce::{
    BootstrapConfig, RunReduce, exchange_unique_id, setup_barrier,
};
use sparkflash_comm::nccl::{NcclApi, NcclError};
use sparkflash_comm::tcp::TcpTransport;
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::parallel_tp::{TpCollective, TpLayout, Wire};

/// Opens the run's NCCL communicator: library check, rendezvous, open, warm up, one sync, one barrier.
///
/// The two rules in the module doc are this function's contract, not advice: any error returned here
/// means the caller must abort the run on every rank. A gather-phase error is already an all-rank
/// abort at the transport; a post-gather error is made an all-rank abort by the setup barrier before
/// this returns. The layer loop must never treat a bootstrap error as something to retry or route
/// around.
///
/// # Errors
/// NcclError::Config for a bad world/rank, a zero payload, a null scratch, or an id supplied by the
/// wrong rank; NcclError::Bootstrap when the gather, the agreement words, the setup sync or the setup
/// barrier refuses; the library/init errors of RunReduce::open and RunReduce::warm_up
/// (LibraryAbsent, ArmUnavailable, Call) otherwise.
pub fn open_run_reduce(
    transport: &TcpTransport,
    world: usize,
    rank: usize,
    stream: u64,
    scratch: u64,
    payload: usize,
    config: BootstrapConfig,
) -> Result<RunReduce, NcclError> {
    validate_open_request(world, rank, payload, scratch)?;
    // Load on EVERY rank before the gather, and carry the status: a rank that lacks libnccl then
    // refuses all four ranks here, before any of them is inside NCCL, instead of failing locally
    // while its peers wait in ncclCommInitRank.
    let library = NcclApi::load();
    let local = if rank == 0 {
        match &library {
            Ok(api) => Some(api.unique_id()?),
            Err(_) => None,
        }
    } else {
        None
    };
    let unique = exchange_unique_id(transport, world, rank, local, library.is_ok(), config)?;
    let handle = RunReduce::open(world, rank, unique)?;
    // Turn the run-owned opaque handle back into the owned value. open hands back
    // Box::into_raw(Box::new(RunReduce)); rebuilding the Box and moving the value out is the exact
    // inverse, so this RunReduce is the box's only owner and the raw handle is never used again -
    // in particular close(handle) must NOT be called on it, Drop destroys the communicator. The
    // port stores THIS value (as a reference) for both per-layer collectives.
    // SAFETY: the pointer came from RunReduce::open in this function, it is not null (open is
    // fallible and returned Ok), and Box::from_raw is paired with that into_raw exactly once.
    let mut reduce: RunReduce = unsafe { *Box::from_raw(handle.cast::<RunReduce>()) };
    // SAFETY: the caller contract above - scratch is a writable device address of at least payload
    // bf16 values on the current device, stream is valid there, and every rank passes the same
    // payload - is exactly warm_up's documented requirement.
    unsafe {
        reduce.warm_up(stream, scratch, payload)?;
    }
    synchronize_setup()?;
    // Bounded setup barrier: every rank that completed open + warm-up + sync meets here, so a rank
    // that failed one of them cannot leave its peers to enter the timed region alone.
    setup_barrier(transport, world, rank)?;
    Ok(reduce)
}

/// [`open_run_reduce`] for a [`DeviceExchange`] of `hidden`-wide f32 rows, on `cuda`'s context.
/// Every rank must pass the same `run`: its digest is the agreement word, so ranks that disagree on
/// what they are about to run refuse before any enters NCCL. The warm-up moves one row's partial,
/// and its scratch is freed once the open has synchronised it.
///
/// # Errors
/// The scratch allocation, or any refusal of [`open_run_reduce`].
pub fn open_exchange_reduce(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    transport: &TcpTransport,
    layout: TpLayout,
    hidden: usize,
    run: &str,
) -> Result<RunReduce, String> {
    let agreement = BootstrapConfig {
        native_rounding: 0,
        env_toggles: 0,
        plan_digest: sparkflash_engine::synthetic::fnv1a(run.as_bytes()),
    };
    // One f32 row is two bf16 words an element, the unit the exchange moves.
    let payload = hidden * 2;
    let mut allocated = 0usize;
    let scratch = cuda.alloc(payload * 2, &mut allocated)?;
    let opened = open_run_reduce(
        transport,
        layout.world_size,
        layout.rank,
        0,
        scratch.0,
        payload,
        agreement,
    );
    cuda.free_buffers(&[&scratch]);
    opened.map_err(|error| format!("nccl open: {error}"))
}

/// Refuses an open request whose arguments cannot be honoured, before any transport or NCCL call.
fn validate_open_request(
    world: usize,
    rank: usize,
    payload: usize,
    scratch: u64,
) -> Result<(), NcclError> {
    if world == 0 {
        return Err(NcclError::Config(
            "open_run_reduce: a communicator needs at least one rank".to_owned(),
        ));
    }
    if rank >= world {
        return Err(NcclError::Config(format!(
            "open_run_reduce: rank {rank} is out of range for world {world}"
        )));
    }
    if payload == 0 {
        return Err(NcclError::Config(
            "open_run_reduce: a warm-up over zero bf16 elements is not a collective".to_owned(),
        ));
    }
    if scratch == 0 {
        return Err(NcclError::Config(
            "open_run_reduce: scratch is the null device address; the warm-up needs payload bf16 values"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Completes the three setup collectives with ONE context sync, before the timed region.
fn synchronize_setup() -> Result<(), NcclError> {
    let cuda = sparkflash_kernels::cuda_driver::Cuda::load().map_err(|error| {
        NcclError::Bootstrap(format!(
            "the setup sync cannot open the run's CUDA context: {error}; the setup barrier makes this an all-rank abort"
        ))
    })?;
    cuda.synchronize().map_err(|error| {
        NcclError::Bootstrap(format!(
            "the setup sync after warm-up failed: {error}; the setup barrier makes this an all-rank abort"
        ))
    })
}

/// A host collective with the run's communicator as its device leg: the host seam's collectives
/// go to the seam, and [TpCollective::exchange_on_device] is one [RunReduce::exchange_bf16] of the
/// payload's bytes, which moves bits and sums nothing, so the caller's [Wire::F32] sum sees the
/// peers' values exactly.
pub struct DeviceExchange {
    host: Box<dyn TpCollective>,
    reduce: RunReduce,
}

// SAFETY: RunReduce holds the communicator's raw handles, which NCCL lets any one thread use at a
// time. TpRuntime keeps its collective behind a Mutex, so no two calls overlap, and the run makes
// every call on the thread that opened the communicator, against the same current context.
unsafe impl Send for DeviceExchange {}

impl DeviceExchange {
    /// `host` with `reduce` as its device leg; both are this rank of the same world.
    ///
    /// # Errors
    /// When the two are not the same rank of the same world, or the library has no
    /// point-to-point calls to exchange with.
    pub fn new(host: Box<dyn TpCollective>, reduce: RunReduce) -> Result<Self, String> {
        if !reduce.has_exchange() {
            return Err("the NCCL library has no point-to-point calls to exchange with".to_owned());
        }
        let (world, rank) = reduce.world_rank();
        if (host.world(), host.rank()) != (world, rank) {
            return Err(format!(
                "the host collective is rank {} of {}, the communicator rank {rank} of {world}",
                host.rank(),
                host.world()
            ));
        }
        Ok(Self { host, reduce })
    }
}

impl TpCollective for DeviceExchange {
    fn world(&self) -> usize {
        self.host.world()
    }

    fn rank(&self) -> usize {
        self.host.rank()
    }

    fn label(&self) -> &'static str {
        "nccl-exchange/exact-f32"
    }

    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError> {
        self.host.all_reduce_assign(values)
    }

    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.host.all_gather_assign(local, out)
    }

    fn send(&mut self, to: usize, values: &[f32]) -> Result<(), EngineError> {
        self.host.send(to, values)
    }

    fn recv(&mut self, from: usize, out: &mut [f32]) -> Result<(), EngineError> {
        self.host.recv(from, out)
    }

    fn device_wire(&self) -> Option<Wire> {
        Some(Wire::F32)
    }

    fn exchange_on_device(
        &mut self,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        let refuse = |detail: String| EngineError::Shape {
            what: "NCCL device exchange".to_owned(),
            detail,
        };
        if !bytes.is_multiple_of(2) {
            return Err(refuse(format!(
                "{bytes} bytes are not whole bf16 words, the unit the exchange moves"
            )));
        }
        // SAFETY: TpCollective::exchange_on_device's contract is exchange_bf16's: `send` holds
        // `bytes` and `slots` a world of them on the communicator's device, `stream` is a stream
        // there, and every rank enqueues the same exchanges in the same order.
        unsafe { self.reduce.exchange_bf16(send, slots, bytes / 2, stream) }
            .map_err(|error| refuse(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_payload_is_refused_before_any_collective() {
        let error = validate_open_request(4, 0, 0, 0x1000)
            .expect_err("a zero-element warm-up is not a collective");
        assert!(matches!(error, NcclError::Config(_)));
    }

    #[test]
    fn a_null_scratch_is_refused_before_any_collective() {
        let error = validate_open_request(4, 0, 5_120, 0)
            .expect_err("a null warm-up buffer must be refused, not dereferenced");
        assert!(matches!(error, NcclError::Config(_)));
    }

    #[test]
    fn a_bad_world_or_rank_is_refused_before_any_collective() {
        assert!(matches!(
            validate_open_request(0, 0, 5_120, 0x1000),
            Err(NcclError::Config(_))
        ));
        assert!(matches!(
            validate_open_request(4, 4, 5_120, 0x1000),
            Err(NcclError::Config(_))
        ));
        assert!(validate_open_request(4, 3, 5_120, 0x1000).is_ok());
    }
}
