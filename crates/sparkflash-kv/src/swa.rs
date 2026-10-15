//! The SWA prefix-tail ring and the bounded replay interface.
//!
//! DeepSeek-V4.1-Flash keeps a 128-token sliding window in every one of its 40 layers. CED
//! prefill never runs the decoder layers over a whole prompt, so resuming a cached prefix
//! either needs that exact 128-token window still resident, or it must be rebuilt by replaying
//! at most 128 tokens through the relevant half of the backbone.
//!
//! A tail is keyed by the FNV-1a hash of its exact 128 token ids and every probe is confirmed
//! against the stored ids, so a hash collision is a miss rather than a wrong window. A tail
//! only counts as a hit once all 40 layers are present, because an encoder replay and a
//! decoder replay each produce half of it; the present-layer mask makes that state explicit
//! instead of pretending a half-built tail is servable.
//!
//! The ring is a separate lock from the page store in crate::cache, which is what makes the
//! "a replay never overwrites cached global KV" rule structural rather than a comment: the
//! replay path can only reach this ring.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::KvError;
use crate::layout::{
    SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOTS_PER_TAIL, SWA_TAIL_BYTES, swa_layer_range,
    swa_slot_offset,
};
use crate::telemetry::KvStoreTelemetry;

/// Identifier of a resident SWA tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TailId(pub u64);

/// Bit mask with every one of the 40 SWA layers present.
pub const ALL_SWA_LAYERS: u64 = (1u64 << SWA_LAYERS) - 1;

/// FNV-1a hash of a 128-token window; the ring key.
#[must_use]
pub fn tail_hash(tokens: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Which half of the backbone a bounded replay runs through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayKind {
    /// Layers 0..19: regenerate SWA KV only, reusing cached global KV.
    Encoder,
    /// Layers 20..39: the decoder's bounded replay of the prompt tail.
    Decoder,
}

impl ReplayKind {
    /// Layers this replay covers.
    #[must_use]
    pub fn layers(self) -> std::ops::Range<usize> {
        match self {
            Self::Encoder => 0..SWA_LAYERS / 2,
            Self::Decoder => SWA_LAYERS / 2..SWA_LAYERS,
        }
    }

    /// Bit mask of the layers this replay produces.
    #[must_use]
    pub fn layer_mask(self) -> u64 {
        let layers = self.layers();
        let mut mask = 0u64;
        for layer in layers {
            mask |= 1 << layer;
        }
        mask
    }

    /// Canonical name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Decoder => "decoder",
        }
    }
}

/// Something that can generate the SWA KV of one layer for a token window.
///
/// The engine implements this over the real attention weights; the tests implement it with a
/// deterministic function of the layer and the token ids, which exercises the whole ring,
/// codec and accounting path without needing a checkpoint.
pub trait SwaProducer {
    /// Fills the SWA slots of one layer for a window of at most SWA_SLOTS_PER_TAIL tokens.
    ///
    /// The output is one layer long (SWA_SLOTS_PER_TAIL slots of SWA_SLOT_BYTES). Slots past
    /// the end of the window are ignored.
    ///
    /// # Re-entrancy
    ///
    /// This runs while the cache holds its SWA ring lock, so the implementation must not call
    /// back into the [KvCache](crate::KvCache) it is producing for. Two locks make that fatal:
    /// a callback into any global-KV method would take the inner lock while SWA is held, the
    /// inverse of the inner-then-SWA order every other multi-lock path uses, and a callback
    /// into any SWA method would take the SWA lock a second time on the same thread.
    /// parking_lot mutexes are not reentrant, so both are deadlocks. [KvCache](crate::KvCache)
    /// detects the attempt and panics with a message naming this method rather than hanging.
    /// The producer is free to use its own state and any weight-side structure.
    ///
    /// # Errors
    /// Propagates a failure from the weight-side implementation.
    fn fill_layer(&mut self, layer: usize, window: &[u32], out: &mut [u8]) -> Result<(), KvError>;
}

/// What one bounded replay actually did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplayOutcome {
    /// Tokens in the replayed window, at most SWA_SLOTS_PER_TAIL.
    pub replayed_tokens: usize,
    /// Layer slots the producer filled.
    pub swa_slots_written: u64,
    /// Global KV pages published while this replay ran, measured from the store's
    /// [replay_global_kv_writes](crate::telemetry::KvStoreTelemetry::replay_global_kv_writes)
    /// counter across the call. Zero for every replay the cache performs, because the store
    /// refuses and counts a global-KV publish attempted from inside a replay scope; a non-zero
    /// value means a replay path reached the global-KV store.
    pub global_kv_pages_written: u64,
    /// Whether the resulting tail is complete enough to be probed as a hit.
    pub tail_completed: bool,
}

impl ReplayOutcome {
    /// True when the replay touched no global KV page.
    #[must_use]
    pub fn global_kv_untouched(&self) -> bool {
        self.global_kv_pages_written == 0
    }
}

/// A resident tail: the exact 128 tokens and the FP8 KV of all 40 layers.
#[derive(Debug)]
pub struct SwaTail {
    id: TailId,
    tokens: Box<[u32]>,
    end_position: usize,
    data: Box<[u8]>,
    present_layers: u64,
    last_used: u64,
}

impl SwaTail {
    /// Ring identifier.
    #[must_use]
    pub const fn id(&self) -> TailId {
        self.id
    }

    /// The exact 128 token ids this tail covers.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Absolute position just past the last token of the window.
    #[must_use]
    pub const fn end_position(&self) -> usize {
        self.end_position
    }

    /// Bit mask of the layers a replay has already produced.
    #[must_use]
    pub const fn present_layers(&self) -> u64 {
        self.present_layers
    }

    /// True when every layer is present, which is what makes the tail servable.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.present_layers == ALL_SWA_LAYERS
    }

    /// Raw SWA bytes for all 40 layers.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
}

/// A bounded ring of SWA prefix tails with LRU eviction.
pub struct SwaTailRing {
    capacity: usize,
    tails: HashMap<u64, SwaTail>,
    next_id: u64,
    clock: u64,
    telemetry: Arc<KvStoreTelemetry>,
}

impl std::fmt::Debug for SwaTailRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwaTailRing")
            .field("capacity", &self.capacity)
            .field("resident", &self.tails.len())
            .field("complete", &self.complete_len())
            .field("bytes", &self.byte_len())
            .finish()
    }
}

impl SwaTailRing {
    /// Creates a ring with a fixed number of tail slots.
    #[must_use]
    pub fn new(capacity: usize, telemetry: Arc<KvStoreTelemetry>) -> Self {
        Self {
            capacity,
            tails: HashMap::new(),
            next_id: 1,
            clock: 0,
            telemetry,
        }
    }

    /// Number of tail slots.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of resident tails, complete or not.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tails.len()
    }

    /// Number of servable tails.
    #[must_use]
    pub fn complete_len(&self) -> usize {
        self.tails
            .values()
            .filter(|tail| tail.is_complete())
            .count()
    }

    /// True when no tail is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tails.is_empty()
    }

    /// Bytes of resident tail payload.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        (self.tails.len() as u64) * (SWA_TAIL_BYTES as u64)
    }

    /// Looks up a *complete* tail by its exact token window.
    ///
    /// Returns None when the window is not exactly SWA_SLOTS_PER_TAIL long, when the hash is
    /// unknown, when the stored ids differ (a hash collision), or when only one half of the
    /// layers has been replayed so far.
    #[must_use]
    pub fn probe(&self, tokens: &[u32]) -> Option<TailId> {
        if tokens.len() != SWA_SLOTS_PER_TAIL {
            return None;
        }
        let tail = self.tails.get(&tail_hash(tokens))?;
        if tail.tokens.as_ref() == tokens && tail.is_complete() {
            Some(tail.id)
        } else {
            None
        }
    }

    /// Looks up a tail by window regardless of completeness.
    #[must_use]
    pub fn probe_any(&self, tokens: &[u32]) -> Option<TailId> {
        if tokens.len() != SWA_SLOTS_PER_TAIL {
            return None;
        }
        let tail = self.tails.get(&tail_hash(tokens))?;
        if tail.tokens.as_ref() == tokens {
            Some(tail.id)
        } else {
            None
        }
    }

    /// Looks up the complete tail ending at the end of a cached prefix.
    ///
    /// This is the plan-time question: is the exact 128-token window that ends at
    /// prefix.len() still resident? A prefix shorter than 128 tokens can never have a full
    /// window, so it always misses.
    #[must_use]
    pub fn probe_prefix(&self, prefix: &[u32]) -> Option<TailId> {
        if prefix.len() < SWA_SLOTS_PER_TAIL {
            return None;
        }
        self.probe(&prefix[prefix.len() - SWA_SLOTS_PER_TAIL..])
    }

    /// Identifies a tail by id.
    #[must_use]
    pub fn get(&self, id: TailId) -> Option<&SwaTail> {
        self.tails.values().find(|tail| tail.id == id)
    }

    /// Creates a zeroed tail for a window if it does not exist yet, evicting the LRU tail
    /// when the ring is full.
    ///
    /// # Errors
    /// Returns KvError::InvalidLength when the window is not exactly SWA_SLOTS_PER_TAIL ids
    /// and KvError::CapacityExceeded when the ring has no slots at all.
    pub fn ensure_tail(&mut self, tokens: &[u32], end_position: usize) -> Result<TailId, KvError> {
        if tokens.len() != SWA_SLOTS_PER_TAIL {
            return Err(KvError::InvalidLength {
                context: "swa tail tokens",
                expected: SWA_SLOTS_PER_TAIL,
                actual: tokens.len(),
            });
        }
        if self.capacity == 0 {
            return Err(KvError::CapacityExceeded {
                requested: 1,
                available: 0,
            });
        }
        let key = tail_hash(tokens);
        if let Some(existing) = self.tails.get(&key)
            && existing.tokens.as_ref() == tokens
        {
            return Ok(existing.id);
        }
        while self.tails.len() >= self.capacity {
            let victim = self
                .tails
                .iter()
                .min_by_key(|(_, tail)| (tail.last_used, tail.id.0))
                .map(|(key, _)| *key);
            match victim {
                Some(key) => {
                    self.tails.remove(&key);
                }
                None => break,
            }
        }
        let id = TailId(self.next_id);
        self.next_id += 1;
        self.clock += 1;
        self.tails.insert(
            key,
            SwaTail {
                id,
                tokens: tokens.to_vec().into_boxed_slice(),
                end_position,
                data: vec![0u8; SWA_TAIL_BYTES].into_boxed_slice(),
                present_layers: 0,
                last_used: self.clock,
            },
        );
        Ok(id)
    }

    /// Copies one layer of a tail.
    ///
    /// # Errors
    /// Returns KvError::TailNotResident when the tail is gone and KvError::InvalidLength when
    /// the payload is not one layer long.
    pub fn read_layer(&self, id: TailId, layer: usize, out: &mut [u8]) -> Result<(), KvError> {
        let range = swa_layer_range(layer);
        if out.len() != range.len() {
            return Err(KvError::InvalidLength {
                context: "swa layer read",
                expected: range.len(),
                actual: out.len(),
            });
        }
        let tail = self.get(id).ok_or(KvError::TailNotResident(id))?;
        out.copy_from_slice(&tail.data[range]);
        Ok(())
    }

    /// Copies one slot of a tail.
    ///
    /// # Errors
    /// Returns KvError::TailNotResident when the tail is gone and KvError::InvalidLength when
    /// the payload is not one slot long.
    pub fn read_slot(
        &self,
        id: TailId,
        layer: usize,
        slot: usize,
        out: &mut [u8],
    ) -> Result<(), KvError> {
        if out.len() != SWA_SLOT_BYTES {
            return Err(KvError::InvalidLength {
                context: "swa slot read",
                expected: SWA_SLOT_BYTES,
                actual: out.len(),
            });
        }
        let offset = swa_slot_offset(layer, slot);
        let tail = self.get(id).ok_or(KvError::TailNotResident(id))?;
        out.copy_from_slice(&tail.data[offset..offset + SWA_SLOT_BYTES]);
        Ok(())
    }

    /// Writes one layer of a resident tail and marks it present.
    ///
    /// # Errors
    /// Returns KvError::TailNotResident when the tail is gone and KvError::InvalidLength when
    /// the payload is not one layer long.
    pub fn write_layer(&mut self, id: TailId, layer: usize, data: &[u8]) -> Result<(), KvError> {
        let range = swa_layer_range(layer);
        if data.len() != range.len() {
            return Err(KvError::InvalidLength {
                context: "swa layer write",
                expected: range.len(),
                actual: data.len(),
            });
        }
        let Some(tail) = self.tails.values_mut().find(|tail| tail.id == id) else {
            return Err(KvError::TailNotResident(id));
        };
        tail.data[range].copy_from_slice(data);
        tail.present_layers |= 1 << layer;
        self.clock += 1;
        tail.last_used = self.clock;
        Ok(())
    }

    /// Marks a whole replay half as present without copying, used when the producer wrote
    /// directly into the ring's buffer.
    ///
    /// # Errors
    /// Returns KvError::TailNotResident when the tail is gone.
    pub fn mark_layers(&mut self, id: TailId, kind: ReplayKind) -> Result<(), KvError> {
        let Some(tail) = self.tails.values_mut().find(|tail| tail.id == id) else {
            return Err(KvError::TailNotResident(id));
        };
        tail.present_layers |= kind.layer_mask();
        self.clock += 1;
        tail.last_used = self.clock;
        Ok(())
    }

    /// Marks a tail as most recently used.
    pub fn touch(&mut self, id: TailId) {
        self.clock += 1;
        let now = self.clock;
        if let Some(tail) = self.tails.values_mut().find(|tail| tail.id == id) {
            tail.last_used = now;
        }
    }

    /// Evicts a tail, returning true when it was present.
    pub fn remove(&mut self, id: TailId) -> bool {
        let key = self
            .tails
            .iter()
            .find(|(_, tail)| tail.id == id)
            .map(|(key, _)| *key);
        match key {
            Some(key) => self.tails.remove(&key).is_some(),
            None => false,
        }
    }

    /// Drops every tail.
    pub fn clear(&mut self) {
        self.tails.clear();
    }

    /// Telemetry sink this ring writes to.
    #[must_use]
    pub fn telemetry(&self) -> &Arc<KvStoreTelemetry> {
        &self.telemetry
    }

    /// Records a completed replay window.
    pub fn note_replay(&self, slots: u64) {
        self.telemetry
            .replay_swa_slots_written
            .fetch_add(slots, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(seed: u32) -> Vec<u32> {
        (0..SWA_SLOTS_PER_TAIL as u32).map(|i| seed + i).collect()
    }

    struct ConstantProducer(u8);

    impl SwaProducer for ConstantProducer {
        fn fill_layer(
            &mut self,
            _layer: usize,
            _window: &[u32],
            out: &mut [u8],
        ) -> Result<(), KvError> {
            out.fill(self.0);
            Ok(())
        }
    }

    fn complete_tail(ring: &mut SwaTailRing, tokens: &[u32]) -> TailId {
        let id = ring.ensure_tail(tokens, 128).expect("tail");
        let mut producer = ConstantProducer(9);
        let mut layer = vec![0u8; SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES];
        for layer_index in 0..SWA_LAYERS {
            producer
                .fill_layer(layer_index, tokens, &mut layer)
                .expect("fill");
            ring.write_layer(id, layer_index, &layer).expect("write");
        }
        id
    }

    #[test]
    fn probe_requires_the_exact_complete_window() {
        let telemetry = Arc::new(KvStoreTelemetry::default());
        let mut ring = SwaTailRing::new(4, telemetry);
        let tokens = window(10);
        let id = complete_tail(&mut ring, &tokens);
        assert_eq!(ring.probe(&tokens), Some(id));
        let mut other = tokens.clone();
        other[0] = 999;
        assert_eq!(ring.probe(&other), None);
        assert_eq!(ring.probe_prefix(&tokens[..64]), None);
        let mut prefix = vec![0u32; 64];
        prefix.extend_from_slice(&tokens);
        assert_eq!(ring.probe_prefix(&prefix), Some(id));
    }

    #[test]
    fn a_half_replayed_tail_is_not_a_hit_until_both_halves_land() {
        let telemetry = Arc::new(KvStoreTelemetry::default());
        let mut ring = SwaTailRing::new(4, telemetry);
        let tokens = window(44);
        let id = ring.ensure_tail(&tokens, 128).expect("tail");
        ring.mark_layers(id, ReplayKind::Encoder).expect("mark");
        assert_eq!(ring.probe(&tokens), None);
        assert_eq!(ring.probe_any(&tokens), Some(id));
        ring.mark_layers(id, ReplayKind::Decoder).expect("mark");
        assert_eq!(ring.probe(&tokens), Some(id));
        assert_eq!(ring.complete_len(), 1);
    }

    #[test]
    fn lru_eviction_keeps_the_ring_bounded() {
        let telemetry = Arc::new(KvStoreTelemetry::default());
        let mut ring = SwaTailRing::new(2, telemetry);
        let a = window(1);
        let b = window(1000);
        let c = window(2000);
        let id_a = complete_tail(&mut ring, &a);
        let _id_b = complete_tail(&mut ring, &b);
        ring.touch(id_a);
        let _id_c = complete_tail(&mut ring, &c);
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.probe(&b), None);
        assert_eq!(ring.probe(&a), Some(id_a));
    }
}
