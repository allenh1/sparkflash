//! The seam between `sparkflash serve --tp4-device` and the device forward loop.
//!
//! The device forward runs inside the `device-chain` epoch: the arena, the bound weights, the
//! resident kernels and the NCCL communicator all live on that call's stack, and none of them is
//! `Send`. Serving therefore does not pull the forward out into a server; it hands the forward a
//! [`DeviceLink`] before the epoch starts ([`install`]), and the DSpark branch of the forward loop
//! takes it ([`take`]) and runs its request loop on the epoch's own thread instead of the one-shot
//! harness generation.
//!
//! The loop serves several sequences at once in scheduler ticks. Every rank holds a link. Rank 0's
//! link is fed by the HTTP driver and broadcasts, before each tick, the requests that join and the
//! sequences that end there (a stop string matched, or the client left) over the serve control
//! channel; a worker's link reads the same frames. Every rank runs the same scheduler on the same
//! decisions, so every rank runs the same steps with the same samplers, emits the same tokens (the
//! sampler's noise is a pure function of the seed and the position), and ends every sequence at
//! the same tick.

use std::sync::Mutex;

/// One request as the device loop runs it: identical on every rank.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceJob {
    /// The rendered prompt ids (the chat template is already applied).
    pub prompt: Vec<u32>,
    /// The most completion tokens to emit.
    pub max_tokens: usize,
    /// 0 is greedy.
    pub temperature: f32,
    /// 1 keeps the whole distribution.
    pub top_p: f32,
    /// 0 keeps every token.
    pub top_k: u32,
    /// The sampler seed (rank 0 draws one for an unseeded request).
    pub seed: u64,
    /// Token ids that end the request (emitted, then stop): the end-of-sequence id unless the
    /// request ignores it.
    pub stop_ids: Vec<u32>,
}

/// Why a device request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFinish {
    /// A stop id was emitted.
    Stop,
    /// `max_tokens` or the context ran out.
    Length,
    /// Rank 0 asked every rank to stop (a stop string matched, or the client went away).
    Aborted,
}

/// What one request did, measured on the rank that ran it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceOutcome {
    /// Completion tokens emitted (the first token from the prefill included).
    pub completion_tokens: usize,
    /// Why it ended.
    pub finish: DeviceFinish,
    /// The prefill's wall clock up to the first token, in ms.
    pub prefill_ms: f64,
    /// The decode steps' wall clock after the first token, in ms.
    pub decode_ms: f64,
    /// Verify steps run.
    pub steps: usize,
    /// Drafts verified, over all steps.
    pub drafts_verified: usize,
    /// Drafts accepted, over all steps.
    pub drafts_accepted: usize,
    /// Leading prompt tokens whose state the previous request left on the device (the shared
    /// prefix), so the prefill skipped them.
    pub cached_tokens: usize,
}

/// Rank 0's decisions for one scheduler tick: the same on every rank.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeviceTick {
    /// Requests that join, in arrival order, each with the sequence id rank 0 gave it.
    pub admit: Vec<(u64, DeviceJob)>,
    /// Sequences that end before the tick runs (a stop string matched, or the client left).
    pub cancel: Vec<u64>,
}

/// One rank's end of the serving loop.
pub trait DeviceLink: Send {
    /// The next tick's decisions, the same on every rank; `None` ends the loop. With `idle` (no
    /// sequence is on the device or waiting for it) it blocks until a request arrives; otherwise
    /// it returns at once. It admits at most `room` requests.
    ///
    /// # Errors
    /// The control channel failed; the group cannot continue.
    fn tick(&mut self, idle: bool, room: usize) -> Result<Option<DeviceTick>, String>;

    /// Sequence `seq`'s tokens from this tick (its first carries the prefill's token). `last` is
    /// true when the sequence ends with them. When rank 0's stream must end before that (a stop
    /// string, the client left), its next tick cancels the sequence.
    fn emit(&mut self, seq: u64, tokens: &[u32], last: bool);

    /// Sequence `seq` ended: its outcome, or the error that ended it. A step's error is fatal to
    /// the group (the ranks may have desynchronised): the loop ends every sequence with it and
    /// stops.
    ///
    /// # Errors
    /// The control channel failed.
    fn finish(&mut self, seq: u64, outcome: Result<&DeviceOutcome, &str>) -> Result<(), String>;
}

static LINK: Mutex<Option<Box<dyn DeviceLink>>> = Mutex::new(None);

/// Installs the link the next device forward epoch serves through.
pub fn install(link: Box<dyn DeviceLink>) {
    *LINK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(link);
}

/// Whether a link is installed (the next device epoch serves through it).
pub fn installed() -> bool {
    LINK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
}

/// Takes the installed link, if any: the forward loop serves through it instead of running the
/// one-shot generation.
pub fn take() -> Option<Box<dyn DeviceLink>> {
    LINK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}
