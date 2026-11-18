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

//! The direct-RDMA exchange (the GB10 doorbell) as the device leg of every TP group's collective.
//!
//! [with_doorbell] puts the exchange in front of a collective whose device leg is the NCCL group:
//! a [TpCollective::exchange_on_device] of at most [DOORBELL_ROW_CAP] bytes - a decode step's row
//! sums ([crate::tp::RowSums]) and logit gather, a narrow prefill chunk's - is one round of the
//! exchange on the caller's stream, and a wider one, the two-shot scatter and every host
//! collective stay on the collective it wraps. Both legs move bits and sum nothing, so the fold
//! behind either gives the same bits: the exchange changes how fast a step's collectives go, never
//! what they carry. The choice is the payload's width, which every rank shares, so every rank
//! takes the same leg.
//!
//! It is the default for every group of two or more ranks with a device leg, on both rails, with
//! the group's own address table when the run names none
//! ([sparkflash_kernels::write_exchange::GroupTable]). `SPARKFLASH_EXCHANGE=nccl` keeps the NCCL
//! group, and a group that cannot open the exchange refuses to start, naming that switch.
//!
//! A round that cannot complete leaves its kernel's wait without the data, so the step that held
//! it produced nothing usable: the open registers the exchange's failure check with every blocking
//! readback ([sparkflash_kernels::cuda_driver::watch_readbacks]), so no arm reads back such a step
//! and no arm has to remember to ask.

use std::sync::Mutex;

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::write_exchange::{
    self, DOORBELL_ROW_CAP, ExchangeState, GroupTable, Opening,
};
use sparkflash_weights::linalg::EngineError;
use sparkflash_weights::tp::{TpCollective, Wire};

/// What the open banner says the exchange replaces in a TP group.
const REPLACES: &str = "every device exchange of a TP group of at most 262144 bytes (a decode \
     step's row sums and logit gather, and a prefill chunk's when it is that narrow); wider \
     exchanges, the two-shot scatter and the host collectives stay on the NCCL group";

/// A collective with the exchange in front of its device leg ([with_doorbell]).
pub struct Doorbell {
    inner: Box<dyn TpCollective>,
    state: &'static Mutex<ExchangeState>,
    /// The context the exchange's kernels launch in, the primary context the arm runs on.
    cuda: Cuda,
}

// SAFETY: `inner` is Send in its own right; the exchange state sits behind its mutex; the context
// handle is the retained primary context, which the driver lets any thread use, and every launch
// through it is made under the exchange's lock. TpRuntime keeps this collective behind its own
// mutex, so no two calls overlap.
unsafe impl Send for Doorbell {}

/// `inner` with the direct-RDMA exchange in front of its device leg, or `inner` itself when the
/// run selected the NCCL group (`SPARKFLASH_EXCHANGE=nccl`), the group is one rank, or `inner`
/// has no device leg (its collectives are the host seam's).
///
/// `table` is the group's own address table and control port, used when `SPARK_COMM_PEERS` is
/// unset. Collective: every rank calls it at the same point, after the NCCL group opens, and the
/// exchange's rendezvous is inside.
///
/// # Errors
/// A switch value this build does not know, a run that asked for the exchange by name on a group
/// with no device leg, or an open that failed (whose message names the opt-out).
pub fn with_doorbell(
    inner: Box<dyn TpCollective>,
    table: GroupTable,
) -> Result<Box<dyn TpCollective>, String> {
    write_exchange::validate()?;
    let (world, rank) = (inner.world(), inner.rank());
    if world < 2 {
        return Ok(inner);
    }
    if inner.device_wire().is_none() {
        // A host-seam group (SPARK_COMM_REDUCE=tcp) has no device exchange to replace.
        if let Some(refusal) = write_exchange::refusal_here() {
            return Err(format!("{refusal}: this TP group has no device leg"));
        }
        return Ok(inner);
    }
    let cuda = Cuda::load()?;
    let opening = Opening {
        row_cap: DOORBELL_ROW_CAP,
        replaces: REPLACES,
    };
    let Some(state) = write_exchange::shared_in(&cuda, world, rank, opening, Some(table), || {
        let image = sparkflash_kernels::images::qwen4_exp_cubin();
        if image.is_empty() {
            return Err(
                "this build carries no Qwen3.8 token image, whose image holds the exchange's \
                 stage and collect kernels"
                    .to_owned(),
            );
        }
        Ok(image)
    })?
    else {
        return Ok(inner);
    };
    sparkflash_kernels::cuda_driver::watch_readbacks(write_exchange::check);
    Ok(Box::new(Doorbell { inner, state, cuda }))
}

/// The refusal a collective reports, in the engine's error type.
fn refused(detail: String) -> EngineError {
    EngineError::Shape {
        what: "direct-RDMA exchange".to_owned(),
        detail,
    }
}

impl TpCollective for Doorbell {
    fn world(&self) -> usize {
        self.inner.world()
    }

    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn label(&self) -> &'static str {
        "rdma-doorbell+nccl/exact-f32"
    }

    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError> {
        self.inner.all_reduce_assign(values)
    }

    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.inner.all_gather_assign(local, out)
    }

    fn send(&mut self, to: usize, values: &[f32]) -> Result<(), EngineError> {
        self.inner.send(to, values)
    }

    fn recv(&mut self, from: usize, out: &mut [f32]) -> Result<(), EngineError> {
        self.inner.recv(from, out)
    }

    fn device_wire(&self) -> Option<Wire> {
        self.inner.device_wire()
    }

    fn exchange_on_device(
        &mut self,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        if bytes > DOORBELL_ROW_CAP || !bytes.is_multiple_of(2) {
            write_exchange::note_fall_through(
                "a TP group's device exchange wider than a round",
                "it is wider than the exchange's 262144 -byte round",
            );
            return self.inner.exchange_on_device(stream, send, slots, bytes);
        }
        let (rank, world) = (self.inner.rank(), self.inner.world());
        let mut state = self.state.lock().map_err(|_| {
            refused("the direct-RDMA exchange is poisoned by an earlier panic".to_owned())
        })?;
        state
            .exchange_on(
                &self.cuda,
                stream,
                send,
                slots,
                bytes,
                rank,
                world,
                &"TP exchange",
            )
            .map_err(refused)
    }

    fn scatter_on_device(
        &mut self,
        stream: u64,
        send: u64,
        slots: u64,
        bytes: usize,
    ) -> Result<(), EngineError> {
        self.inner.scatter_on_device(stream, send, slots, bytes)
    }
}
