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

//! The Dory latent-recurrence group: one process a rank, each running its own recurrent loops.
//!
//! This is not the tensor-parallel group of `--tp4-rank`. No rank holds a slice of a layer and
//! nothing is reduced: rank 0 embeds the tokens and runs the input prefix and its loops, each
//! later rank runs its loops on the latent rows the rank before it sent, and the last rank runs
//! the output suffix and the head and answers rank 0 with the token
//! (sparkflash_model_dory::recurrence). Rank 0 owns HTTP; the workers serve the ring and nothing
//! else.
//!
//! The ring is NCCL point-to-point between the GPUs. TCP carries only the communicator's
//! rendezvous (rank R listens on `--recurrence-port` + R). By default the group pins NCCL to its
//! verbs network (`NCCL_NET=IB`, which is RoCE on the Spark rail), so a rail NCCL cannot use is
//! a failure at the communicator's open rather than a silent fall back to sockets;
//! `--recurrence-net any` lets NCCL choose.

use std::net::IpAddr;

/// Rank R listens on this port plus R for the communicator's rendezvous.
pub const RECURRENCE_DEFAULT_PORT: u16 = 29801;

/// NCCL's network module selector.
const NCCL_NET_ENV: &str = "NCCL_NET";

/// NCCL's name for its verbs (InfiniBand and RoCE) network.
const NCCL_NET_IB: &str = "IB";

/// Which network NCCL may carry the ring on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RecurrenceNet {
    /// The verbs network only (`NCCL_NET=IB`): the open fails rather than use sockets.
    Rdma,
    /// Whatever network NCCL picks, sockets included.
    Any,
}

/// This process's place in a latent-recurrence group (`--recurrence-rank`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrenceServeOptions {
    /// This process's rank; rank 0 owns HTTP and the sequence.
    pub rank: usize,
    /// Every rank's address, in rank order.
    pub peers: Vec<IpAddr>,
    /// Rank R listens on `port + R` for the rendezvous.
    pub port: u16,
    /// The network NCCL may carry the ring on.
    pub net: RecurrenceNet,
    /// A lane drafts while at most this many lanes are held; 0 never drafts.
    pub draft_lanes: usize,
}

impl RecurrenceServeOptions {
    /// Parses the recurrence flags, or returns `None` when none was given.
    ///
    /// # Errors
    /// A flag without `--recurrence-rank`, a missing or short peer list, a rank outside it, a
    /// port range that does not fit, or drafting in a group of fewer than three ranks.
    pub fn from_cli(
        rank: Option<usize>,
        peers: Option<&str>,
        port: Option<u16>,
        net: Option<RecurrenceNet>,
        draft_lanes: Option<usize>,
    ) -> Result<Option<Self>, String> {
        if (rank, peers, port, net, draft_lanes) == (None, None, None, None, None) {
            return Ok(None);
        }
        let rank = rank.ok_or_else(|| {
            "--recurrence-rank is required when any --recurrence-* flag is given; the group is \
             one process a rank"
                .to_owned()
        })?;
        let raw = peers.ok_or_else(|| {
            "--recurrence-peers is required for --recurrence-rank: give every rank's address in \
             rank order, comma separated"
                .to_owned()
        })?;
        let peers = super::parse_peers("--recurrence-peers", raw)?;
        if peers.len() < 2 {
            return Err(format!(
                "--recurrence-peers lists {} address(es): a recurrence group needs at least two \
                 ranks, and one process serves the whole model without the flags",
                peers.len()
            ));
        }
        if rank >= peers.len() {
            return Err(format!(
                "--recurrence-rank {rank} is outside the {} ranks --recurrence-peers lists",
                peers.len()
            ));
        }
        let port = port.unwrap_or(RECURRENCE_DEFAULT_PORT);
        let last = u16::try_from(peers.len() - 1)
            .ok()
            .and_then(|offset| port.checked_add(offset));
        if last.is_none() {
            return Err(format!(
                "--recurrence-port {port} leaves no port for rank {}",
                peers.len() - 1
            ));
        }
        let draft_lanes = sparkflash_model_dory::recurrence_draft_lanes(peers.len(), draft_lanes)
            .map_err(|error| format!("--recurrence-draft-lanes: {error}"))?;
        Ok(Some(Self {
            rank,
            peers,
            port,
            net: net.unwrap_or(RecurrenceNet::Rdma),
            draft_lanes,
        }))
    }

    /// The number of ranks.
    #[must_use]
    pub fn world(&self) -> usize {
        self.peers.len()
    }
}

/// Pins NCCL's network for the ring before anything opens a communicator.
///
/// Under [RecurrenceNet::Rdma] an unset `NCCL_NET` becomes `IB`; one already naming another
/// network is refused rather than overridden. Must run before the process starts a thread.
///
/// # Errors
/// `NCCL_NET` names a network other than the verbs one.
pub(super) fn pin_network(net: RecurrenceNet) -> Result<(), String> {
    if net == RecurrenceNet::Any {
        // "any" means NCCL chooses, including when the parent environment already named a
        // network. Clear the variable rather than leaving an inherited pin in place.
        // SAFETY: run calls this straight after parsing, before the process starts any thread.
        unsafe { std::env::remove_var(NCCL_NET_ENV) };
        return Ok(());
    }
    match std::env::var(NCCL_NET_ENV) {
        Ok(value) if value.eq_ignore_ascii_case(NCCL_NET_IB) => Ok(()),
        Ok(value) => Err(format!(
            "{NCCL_NET_ENV}={value} would carry the recurrence ring off the RDMA rail; unset it \
             (the group sets {NCCL_NET_ENV}={NCCL_NET_IB}), or pass --recurrence-net any to let \
             NCCL choose"
        )),
        Err(_) => {
            // SAFETY: `run` calls this straight after parsing, before the process starts any
            // thread, so nothing reads the environment concurrently. NCCL reads it at the
            // communicator's open.
            unsafe { std::env::set_var(NCCL_NET_ENV, NCCL_NET_IB) };
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) use group::serve;

#[cfg(feature = "cuda")]
mod group {
    use std::process::ExitCode;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use sparkflash_comm::{RunReduce, TcpTransport};
    use sparkflash_engine::parallel_tp::TpLayout;
    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
    use sparkflash_model::{ModelDescriptor, PromptCodec};
    use sparkflash_model_dory::{
        DoryConfig, DoryRecurrenceBackend, DoryRecurrenceStage, DoryStage, LinkStreams,
        RecurrenceLink, RecurrenceLinks,
    };

    use super::{NCCL_NET_ENV, RecurrenceServeOptions};
    use crate::serve::ServeOptions;
    use crate::serve::dynamic::{Dialect, DynamicCheckpoint};

    /// How long a rank waits for its peers' rendezvous listeners: a worker may be started well
    /// before rank 0 finishes loading.
    const RENDEZVOUS_WAIT: Duration = Duration::from_secs(600);

    /// One of a rank's links over an NCCL communicator: one point-to-point transfer a call,
    /// queued on the link's stream (the default stream, behind the stage's kernels, but for rank
    /// 0's answers and drafts links).
    struct NcclLink {
        reduce: RunReduce,
        stream: u64,
        /// The rendezvous sockets both of the rank's communicators were opened over, kept until
        /// they close.
        _transport: Arc<TcpTransport>,
    }

    // SAFETY: RunReduce holds the communicator's raw handles, which NCCL lets any one thread use
    // at a time. The link is owned by one stage: a worker's serve loop, or on rank 0 the ring
    // under the head's lock and the answers and drafts links by a thread each, so no two calls
    // on one communicator overlap; every call first makes the stage's context current.
    unsafe impl Send for NcclLink {}

    /// NCCL moves the frames and rows as bf16 words: every message is a whole number of them.
    fn words(bytes: usize) -> Result<usize, String> {
        if bytes.is_multiple_of(2) {
            Ok(bytes / 2)
        } else {
            Err(format!(
                "a {bytes}-byte ring message is not a whole number of 2-byte words"
            ))
        }
    }

    impl RecurrenceLink for NcclLink {
        fn send(
            &mut self,
            cuda: &Cuda,
            peer: usize,
            buffer: Deviceptr,
            bytes: usize,
        ) -> Result<(), String> {
            let count = words(bytes)?;
            cuda.make_current()?;
            // SAFETY: `buffer` is a live allocation of at least `bytes` on this context, and the
            // stage keeps it until the stream's later work (or the next read) has waited on it;
            // the stream is the default one or one of rank 0's link streams, live while the link
            // is.
            unsafe { self.reduce.send_bf16(buffer.0, count, peer, self.stream) }
                .map_err(|error| format!("nccl send to rank {peer}: {error}"))
        }

        fn recv(
            &mut self,
            cuda: &Cuda,
            peer: usize,
            buffer: Deviceptr,
            bytes: usize,
        ) -> Result<(), String> {
            let count = words(bytes)?;
            cuda.make_current()?;
            // SAFETY: as for `send`; nothing reads `buffer` before the stream reaches the receive.
            unsafe { self.reduce.recv_bf16(buffer.0, count, peer, self.stream) }
                .map_err(|error| format!("nccl receive from rank {peer}: {error}"))
        }
    }

    /// Opens this rank's communicators on `cuda` over one TCP rendezvous: the ring, the answers
    /// link from the last rank to rank 0, and in a drafting group the drafts link from rank 1 to
    /// rank 0; rank 0's ends of the last two queue on `streams`.
    ///
    /// `run` is what every rank must agree on (the model, its loops, the world, the context, the
    /// lanes and the drafting); a rank started with another document, window, `--concurrency` or
    /// `--recurrence-draft-lanes` fails the open instead of the first step.
    fn open_links(
        cuda: &Cuda,
        group: &RecurrenceServeOptions,
        hidden: usize,
        run: &str,
        streams: LinkStreams,
    ) -> Result<RecurrenceLinks, String> {
        let started = Instant::now();
        let config = crate::tp4_host::tcp_config(group.rank, &group.peers, group.port, true)?
            .with_connect_timeout(RENDEZVOUS_WAIT);
        let transport = Arc::new(
            TcpTransport::connect(config).map_err(|error| format!("tcp rendezvous: {error}"))?,
        );
        let layout = TpLayout {
            rank: group.rank,
            world_size: group.world(),
        };
        let open = |run: &str, stream: u64| -> Result<Box<dyn RecurrenceLink>, String> {
            let reduce =
                crate::tp4_reduce::open_exchange_reduce(cuda, &transport, layout, hidden, run)?;
            Ok(Box::new(NcclLink {
                reduce,
                stream,
                _transport: Arc::clone(&transport),
            }))
        };
        let ring = open(run, 0)?;
        let answers = open(&format!("{run}:answers"), streams.answers)?;
        let drafts = (group.draft_lanes > 0)
            .then(|| open(&format!("{run}:drafts"), streams.drafts))
            .transpose()?;
        eprintln!(
            "sparkflash serve: recurrence rank {}/{} communicators open in {:.1} s ({NCCL_NET_ENV}={})",
            group.rank,
            group.world(),
            started.elapsed().as_secs_f64(),
            std::env::var(NCCL_NET_ENV).unwrap_or_else(|_| "unset".to_owned())
        );
        Ok(RecurrenceLinks {
            ring,
            answers,
            drafts,
        })
    }

    fn describe(stage: &DoryStage) -> String {
        let mut text = format!("loops {}..{}", stage.loops.start, stage.loops.end);
        if stage.prefix {
            text.push_str(" + input prefix");
        }
        if stage.suffix {
            text.push_str(" + output suffix and head");
        }
        if stage.draft {
            text.push_str(" + draft exit");
        }
        text
    }

    /// Serves this process's rank of the group: rank 0 behind HTTP, a worker on the ring alone.
    ///
    /// # Errors
    /// The checkpoint's config, the stage's refusal (an uneven loop split, last-loop KV), this
    /// build's missing Dory image, or the communicator's open.
    pub(in crate::serve) fn serve(
        checkpoint: DynamicCheckpoint,
        descriptor: ModelDescriptor,
        seam: Arc<dyn PromptCodec>,
        dialect: Dialect,
        context: usize,
        options: &ServeOptions,
        group: &RecurrenceServeOptions,
    ) -> Result<ExitCode, String> {
        let directory = checkpoint.directory.clone();
        let config = DoryConfig::from_checkpoint_dir(&directory).map_err(|error| {
            format!(
                "--model {}: {}: {error}",
                checkpoint.model_id,
                directory.join("config.json").display()
            )
        })?;
        if !sparkflash_model_dory::DoryDeviceBackend::available() {
            return Err(
                "--recurrence-rank runs the Dory device arm, and this build carries no Dory \
                 cubin; rebuild with nvcc for this GPU"
                    .to_owned(),
            );
        }
        let world = group.world();
        // One sequence lane a concurrent request, on every rank.
        let lanes = options.concurrency;
        let draft_lanes = group.draft_lanes;
        let run = format!(
            "dory-recurrence:{}:loops={}:world={world}:context={context}:lanes={lanes}:\
             draft={draft_lanes}",
            checkpoint.model_id, config.n_recurrent_loops
        );
        let hidden = config.hidden_size;
        let links =
            |cuda: &Cuda, streams: LinkStreams| open_links(cuda, group, hidden, &run, streams);
        if group.rank > 0 {
            let stage = DoryRecurrenceStage::open(
                &directory,
                context,
                group.rank,
                world,
                lanes,
                draft_lanes,
                links,
            )
            .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
            println!(
                "sparkflash serve: {} recurrence rank {}/{world} ({}), context {context}, {lanes} \
                 lane(s), {:.2} GB of weights on the device; serving rank 0's ring",
                descriptor.id,
                group.rank,
                describe(stage.stage()),
                stage.weight_bytes() as f64 / 1e9
            );
            stage.serve()?;
            println!(
                "sparkflash serve: recurrence rank {}/{world}: rank 0 closed the group",
                group.rank
            );
            return Ok(ExitCode::SUCCESS);
        }
        let backend = DoryRecurrenceBackend::open_with_descriptor(
            descriptor.clone(),
            &directory,
            context,
            world,
            lanes,
            draft_lanes,
            links,
        )
        .map_err(|error| format!("--model {}: {error}", checkpoint.model_id))?;
        let banner = format!(
            "{} ({}) as recurrence rank 0/{world} ({}) on {}, context {context}, {lanes} \
             lane(s), drafting at up to {draft_lanes}, {:.2} GB of weights on the device",
            descriptor.id,
            descriptor.weights_id,
            describe(&backend.stage()),
            directory.display(),
            backend.weight_bytes() as f64 / 1e9
        );
        // The group's rows run the single process's kernels on every rank, so its arithmetic is
        // the TP=1 arm's, bit for bit, drafting or not (tests/recurrence.rs): it answers under
        // that numerics key.
        super::super::dynamic::serve_backend(
            Box::new(backend),
            descriptor,
            seam,
            dialect,
            options,
            &banner,
        )
    }
}
