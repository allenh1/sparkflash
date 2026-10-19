//! The transport abstraction and the in-process loopback backend.
//!
//! Transport is deliberately tiny: a rank can all-reduce a bf16 buffer or all-gather one segment.
//! Everything else (which peers talk, in what order, and how the arithmetic is ordered) lives in
//! the schedule driver, so a new backend only has to move frames.
//!
//! LoopbackTransport models the RoCEnante handshake in process memory: each rank writes its
//! payload into the destination slot, publishes an arrival sequence number exactly like the
//! kernel's arrival_seq[] word, rings the doorbell and then waits for all peers. Ranks run as
//! real threads, so arrival order is genuinely racy; the reduction is order independent by
//! construction, and the tests shuffle arrival order on purpose to prove it.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::error::CommError;
use crate::exchange::{self, Exchange, ExchangeTag, Received};
use crate::layout::{CollectiveLayout, MAX_FRAME_ELEMENTS, REFERENCE_WORLD_SIZE};
use crate::rail::RailLedger;

/// A collective transport a scheduler can route work to.
///
/// Implementations must be safe to call from several rank threads at once: each rank owns its
/// own transport handle, and one collective is in flight at a time per handle.
pub trait Transport: Send + Sync {
    /// Number of ranks in the collective.
    fn world_size(&self) -> usize;
    /// Rank this handle belongs to.
    fn rank(&self) -> usize;
    /// All-reduces the local buffer in place, exactly like the RoCEnante kernel.
    ///
    /// The buffer must hold layout.num_elements() elements and is both input (this rank's
    /// contribution) and output.
    ///
    /// # Errors
    /// Returns CommError when the layout does not match the buffer or a peer fails.
    fn all_reduce_bf16(
        &self,
        buf: &mut [u16],
        seq: u64,
        layout: &CollectiveLayout,
    ) -> Result<(), CommError>;
    /// Gathers one segment per rank into out, in ascending rank order.
    ///
    /// out must hold exactly world_size * local.len() elements.
    ///
    /// # Errors
    /// Returns CommError when the lengths are wrong or a peer fails.
    fn all_gather_bf16(&self, local: &[u16], out: &mut [u16], seq: u64) -> Result<(), CommError>;
    /// All-reduces a **device** buffer in place through the one-shot kernel seam.
    ///
    /// This is the device-buffer form of the collective described in
    /// `docs/build/rdma-backend.md` section 15: the payload never crosses the host. The host-slice
    /// `all_reduce_bf16` above stays the only form loopback, TCP and every CI test implement, and
    /// it is deliberately not a default fallback for this one. A backend that has not wired the
    /// kernel and the proxy refuses with [`CommError::Unsupported`] naming the prerequisite, so a
    /// call site cannot believe it ran a copy-elided collective when it did not.
    ///
    /// # Errors
    /// The default implementation always returns [`CommError::Unsupported`]; a real implementation
    /// returns its transport errors as usual.
    fn all_reduce_bf16_device(
        &self,
        buffer: &crate::oneshot::DeviceBuffer,
        elements: usize,
        seq: u64,
    ) -> Result<(), CommError> {
        let _ = (buffer, elements, seq);
        Err(CommError::Unsupported(
            "the device-buffer one-shot seam needs the RoCEnante kernel launcher and a registered \
             slot (S5.3); this backend moves host slices only"
                .to_owned(),
        ))
    }
}

/// Configuration of an in-process loopback cluster.
#[derive(Debug, Clone)]
pub struct LoopbackConfig {
    /// Number of ranks (threads) in the cluster.
    pub world_size: usize,
    /// Largest buffer the modelled slots accept, in bf16 elements.
    pub max_elements: usize,
    /// How long a rank waits for peer arrivals before failing the collective.
    pub timeout: Duration,
    /// Optional artificial per-transfer delay used by the arrival-order tests.
    pub send_delay: Duration,
}

impl Default for LoopbackConfig {
    fn default() -> Self {
        Self {
            world_size: REFERENCE_WORLD_SIZE,
            // The RoCEnante slot carries at most 2 MiB per rank.
            max_elements: crate::ROCENANTE_MAX_BYTES / 2,
            timeout: Duration::from_secs(5),
            send_delay: Duration::ZERO,
        }
    }
}

/// Shared state of one loopback cluster.
pub struct LoopbackCluster {
    inner: Arc<Inner>,
}

struct Inner {
    config: LoopbackConfig,
    state: Mutex<LoopbackState>,
    arrivals: Condvar,
    doorbells: AtomicU64,
    collectives: AtomicU64,
    ledger: Mutex<RailLedger>,
}

struct LoopbackState {
    alive: Vec<bool>,
    slots: Vec<Slot>,
}

/// One rank's modelled RDMA slot: arrival words plus the staged payloads.
#[derive(Default)]
struct Slot {
    arrived: HashMap<ExchangeTag, Vec<bool>>,
    payload: HashMap<(ExchangeTag, usize), Vec<u16>>,
}

impl LoopbackCluster {
    /// Creates a cluster with world_size modelled ranks.
    ///
    /// # Errors
    /// Returns CommError::InvalidLayout for an empty cluster or a slot larger than a frame.
    pub fn new(config: LoopbackConfig) -> Result<Self, CommError> {
        if config.world_size == 0 {
            return Err(CommError::InvalidLayout(
                "a loopback cluster needs at least one rank".to_owned(),
            ));
        }
        if config.max_elements > MAX_FRAME_ELEMENTS {
            return Err(CommError::InvalidLayout(format!(
                "slot of {} elements exceeds the frame limit {MAX_FRAME_ELEMENTS}",
                config.max_elements
            )));
        }
        let world_size = config.world_size;
        Ok(Self {
            inner: Arc::new(Inner {
                config,
                state: Mutex::new(LoopbackState {
                    alive: vec![true; world_size],
                    slots: (0..world_size).map(|_| Slot::default()).collect(),
                }),
                arrivals: Condvar::new(),
                doorbells: AtomicU64::new(0),
                collectives: AtomicU64::new(0),
                ledger: Mutex::new(RailLedger::default()),
            }),
        })
    }

    /// Number of ranks in the cluster.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.inner.config.world_size
    }

    /// Creates the transport handle for one rank.
    ///
    /// # Errors
    /// Returns CommError::RankOutOfRange for a rank outside the cluster.
    pub fn handle(&self, rank: usize) -> Result<LoopbackTransport, CommError> {
        if rank >= self.inner.config.world_size {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: self.inner.config.world_size,
            });
        }
        Ok(LoopbackTransport {
            inner: Arc::clone(&self.inner),
            rank,
        })
    }

    /// Models a peer dying: later sends to it and waits on it fail with PeerClosed.
    ///
    /// # Errors
    /// Returns CommError::RankOutOfRange for a rank outside the cluster.
    pub fn kill(&self, rank: usize) -> Result<(), CommError> {
        self.set_alive(rank, false)
    }

    /// Models a peer coming back.
    ///
    /// # Errors
    /// Returns CommError::RankOutOfRange for a rank outside the cluster.
    pub fn revive(&self, rank: usize) -> Result<(), CommError> {
        self.set_alive(rank, true)
    }

    fn set_alive(&self, rank: usize, alive: bool) -> Result<(), CommError> {
        if rank >= self.inner.config.world_size {
            return Err(CommError::RankOutOfRange {
                rank,
                world_size: self.inner.config.world_size,
            });
        }
        let mut state = self.inner.state.lock();
        state.alive[rank] = alive;
        self.inner.arrivals.notify_all();
        Ok(())
    }

    /// Whether a rank is currently reachable.
    #[must_use]
    pub fn is_alive(&self, rank: usize) -> bool {
        if rank >= self.inner.config.world_size {
            return false;
        }
        self.inner.state.lock().alive[rank]
    }

    /// Actual bytes each rail moved through this cluster so far.
    #[must_use]
    pub fn rail_ledger(&self) -> RailLedger {
        *self.inner.ledger.lock()
    }

    /// Clears the measured rail ledger.
    pub fn reset_rail_ledger(&self) {
        *self.inner.ledger.lock() = RailLedger::default();
    }

    /// Doorbells rung so far: one per exchange phase, like the host proxy posting the WRs.
    #[must_use]
    pub fn doorbells(&self) -> u64 {
        self.inner.doorbells.load(Ordering::Relaxed)
    }

    /// Collectives started so far (one per prepares with phase 0).
    #[must_use]
    pub fn collectives(&self) -> u64 {
        self.inner.collectives.load(Ordering::Relaxed)
    }
}

/// One rank's handle into a loopback cluster.
pub struct LoopbackTransport {
    inner: Arc<Inner>,
    rank: usize,
}

impl Transport for LoopbackTransport {
    fn world_size(&self) -> usize {
        self.inner.config.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_bf16(
        &self,
        buf: &mut [u16],
        seq: u64,
        layout: &CollectiveLayout,
    ) -> Result<(), CommError> {
        self.check_capacity(buf.len())?;
        let mut exchange = LoopbackExchange {
            inner: &self.inner,
            rank: self.rank,
            current: None,
        };
        exchange::all_reduce(&mut exchange, self.rank, buf, seq, layout)
    }

    fn all_gather_bf16(&self, local: &[u16], out: &mut [u16], seq: u64) -> Result<(), CommError> {
        self.check_capacity(out.len())?;
        let mut exchange = LoopbackExchange {
            inner: &self.inner,
            rank: self.rank,
            current: None,
        };
        exchange::all_gather(&mut exchange, self.rank, local, out, seq, self.world_size())
    }
}

impl LoopbackTransport {
    fn check_capacity(&self, elements: usize) -> Result<(), CommError> {
        if elements > self.inner.config.max_elements {
            return Err(CommError::InvalidLayout(format!(
                "{elements} elements exceed the modelled slot capacity of {}",
                self.inner.config.max_elements
            )));
        }
        Ok(())
    }
}

/// The loopback side of the schedule driver.
struct LoopbackExchange<'a> {
    inner: &'a Inner,
    rank: usize,
    current: Option<ExchangeTag>,
}

impl LoopbackExchange<'_> {
    fn record_transfer(&self, bytes: usize) {
        self.inner.ledger.lock().add_transfer(bytes);
    }

    fn world(&self) -> usize {
        self.inner.config.world_size
    }
}

impl Exchange for LoopbackExchange<'_> {
    fn prepare(&mut self, tag: ExchangeTag, expected: &[(usize, usize)]) -> Result<(), CommError> {
        let world = self.world();
        let mut state = self.inner.state.lock();
        if !state.alive[self.rank] {
            return Err(CommError::PeerClosed { peer: self.rank });
        }
        for (peer, _len) in expected {
            if *peer >= world {
                return Err(CommError::RankOutOfRange {
                    rank: *peer,
                    world_size: world,
                });
            }
            if *peer == self.rank {
                return Err(CommError::Protocol(
                    "a rank cannot receive a frame from itself".to_owned(),
                ));
            }
        }
        // Staging the local payload publishes arrival_seq[my_rank] = seq in this rank's slot.
        state.slots[self.rank]
            .arrived
            .entry(tag)
            .or_insert_with(|| vec![false; world])[self.rank] = true;
        drop(state);
        self.current = Some(tag);
        self.inner.doorbells.fetch_add(1, Ordering::Relaxed);
        if tag.phase == 0 {
            self.inner.collectives.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn send(
        &mut self,
        to: usize,
        elements: std::ops::Range<usize>,
        payload: &[u16],
    ) -> Result<(), CommError> {
        let tag = self.current.ok_or_else(|| {
            CommError::Protocol("send() before prepare() on the loopback exchange".to_owned())
        })?;
        let world = self.world();
        if to >= world {
            return Err(CommError::RankOutOfRange {
                rank: to,
                world_size: world,
            });
        }
        if elements.len() != payload.len() {
            return Err(CommError::InvalidLayout(format!(
                "transfer to rank {to} declares {} elements but carries {}",
                elements.len(),
                payload.len()
            )));
        }
        // Optional stagger so the arrival-order tests really do interleave ranks.
        if !self.inner.config.send_delay.is_zero() {
            let factor = 1 + ((self.rank + to) % 3) as u32;
            std::thread::sleep(self.inner.config.send_delay * factor);
        }
        let mut state = self.inner.state.lock();
        if !state.alive[to] {
            return Err(CommError::PeerClosed { peer: to });
        }
        if !state.alive[self.rank] {
            return Err(CommError::PeerClosed { peer: self.rank });
        }
        let slot = &mut state.slots[to];
        slot.payload.insert((tag, self.rank), payload.to_vec());
        slot.arrived
            .entry(tag)
            .or_insert_with(|| vec![false; world])[self.rank] = true;
        self.inner.arrivals.notify_all();
        drop(state);
        self.record_transfer(payload.len() * 2);
        Ok(())
    }

    fn collect(&mut self, expected: &[(usize, usize)]) -> Result<Vec<Received>, CommError> {
        let tag = self.current.ok_or_else(|| {
            CommError::Protocol("collect() before prepare() on the loopback exchange".to_owned())
        })?;
        let world = self.world();
        let timeout = self.inner.config.timeout;
        let deadline = Instant::now() + timeout;
        let mut state = self.inner.state.lock();
        loop {
            let mut missing: Vec<usize> = Vec::new();
            for (peer, _len) in expected {
                if *peer >= world {
                    return Err(CommError::RankOutOfRange {
                        rank: *peer,
                        world_size: world,
                    });
                }
                let arrived = state.slots[self.rank]
                    .arrived
                    .get(&tag)
                    .map(|flags| flags[*peer])
                    .unwrap_or(false);
                if !arrived {
                    if !state.alive[*peer] {
                        return Err(CommError::PeerClosed { peer: *peer });
                    }
                    missing.push(*peer);
                }
            }
            if missing.is_empty() {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(CommError::timeout(
                    timeout,
                    format!("waiting for arrivals from ranks {missing:?}"),
                ));
            }
            self.inner.arrivals.wait_for(&mut state, deadline - now);
        }
        let slot = &mut state.slots[self.rank];
        let mut received = Vec::with_capacity(expected.len());
        for (peer, len) in expected {
            let payload = slot.payload.remove(&(tag, *peer)).unwrap_or_default();
            if payload.len() != *len {
                return Err(CommError::Protocol(format!(
                    "frame from rank {peer} carried {} elements, expected {len}",
                    payload.len()
                )));
            }
            received.push(Received {
                from: *peer,
                payload,
            });
        }
        slot.arrived.remove(&tag);
        Ok(received)
    }

    fn finish(&mut self) -> Result<(), CommError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::CollectiveLayout;
    use crate::reduce::canonical_all_reduce_into;

    fn cluster(world_size: usize, timeout: Duration) -> LoopbackCluster {
        LoopbackCluster::new(LoopbackConfig {
            world_size,
            timeout,
            ..LoopbackConfig::default()
        })
        .unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn rejects_empty_clusters_and_bad_ranks() {
        assert!(matches!(
            LoopbackCluster::new(LoopbackConfig {
                world_size: 0,
                ..LoopbackConfig::default()
            }),
            Err(CommError::InvalidLayout(_))
        ));
        let cluster = cluster(4, Duration::from_secs(1));
        assert!(matches!(
            cluster.handle(4),
            Err(CommError::RankOutOfRange { .. })
        ));
        assert!(matches!(
            cluster.kill(9),
            Err(CommError::RankOutOfRange { .. })
        ));
        assert!(cluster.is_alive(0));
        cluster.kill(0).unwrap_or_else(|error| panic!("{error}"));
        assert!(!cluster.is_alive(0));
    }

    #[test]
    fn single_rank_reduce_matches_the_cpu_reference() {
        let cluster = cluster(1, Duration::from_millis(500));
        let transport = cluster.handle(0).unwrap_or_else(|error| panic!("{error}"));
        let layout = CollectiveLayout::one_shot(1, 4).unwrap_or_else(|error| panic!("{error}"));
        let mut buf = vec![0x3f80u16, 0x8000, 0x0000, 0x4048];
        let input = buf.clone();
        transport
            .all_reduce_bf16(&mut buf, 1, &layout)
            .unwrap_or_else(|error| panic!("{error}"));
        let mut expected = vec![0u16; 4];
        canonical_all_reduce_into(&mut expected, &[&input])
            .unwrap_or_else(|error| panic!("{error}"));
        // Negative zero reduces to positive zero even with a single rank: 0.0f + -0.0f = +0.0f.
        assert_eq!(expected[1], 0x0000);
        assert_eq!(buf, expected);
        assert_eq!(cluster.doorbells(), 1);
        assert_eq!(cluster.collectives(), 1);
    }

    #[test]
    fn a_dead_peer_fails_the_collective_immediately() {
        let cluster = cluster(2, Duration::from_millis(500));
        let transport = cluster.handle(0).unwrap_or_else(|error| panic!("{error}"));
        cluster.kill(1).unwrap_or_else(|error| panic!("{error}"));
        let layout = CollectiveLayout::one_shot(2, 8).unwrap_or_else(|error| panic!("{error}"));
        let mut buf = vec![0u16; 8];
        assert!(matches!(
            transport.all_reduce_bf16(&mut buf, 1, &layout),
            Err(CommError::PeerClosed { peer: 1 })
        ));
    }

    #[test]
    fn a_missing_peer_times_out() {
        let cluster = cluster(2, Duration::from_millis(40));
        let transport = cluster.handle(0).unwrap_or_else(|error| panic!("{error}"));
        let layout = CollectiveLayout::one_shot(2, 8).unwrap_or_else(|error| panic!("{error}"));
        let mut buf = vec![0u16; 8];
        match transport.all_reduce_bf16(&mut buf, 1, &layout) {
            Err(CommError::Timeout { timeout_ms, .. }) => assert_eq!(timeout_ms, 40),
            other => panic!("expected a timeout, got {other:?}"),
        }
    }

    #[test]
    fn slot_capacity_is_enforced() {
        let cluster = LoopbackCluster::new(LoopbackConfig {
            world_size: 2,
            max_elements: 4,
            timeout: Duration::from_millis(50),
            send_delay: Duration::ZERO,
        })
        .unwrap_or_else(|error| panic!("{error}"));
        let transport = cluster.handle(0).unwrap_or_else(|error| panic!("{error}"));
        let layout = CollectiveLayout::one_shot(2, 5).unwrap_or_else(|error| panic!("{error}"));
        let mut buf = vec![0u16; 5];
        assert!(matches!(
            transport.all_reduce_bf16(&mut buf, 1, &layout),
            Err(CommError::InvalidLayout(_))
        ));
    }
}
