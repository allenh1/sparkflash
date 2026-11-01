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

//! The tensor-parallel seam: a rank's collective adapter and a resident shard source.
//!
//! This module is the engine half of the Phase-1 TP path described in
//! `docs/archive/build/tp4-design.md` section 3 and `docs/archive/build/engine-tp-seams.md`. The
//! model-agnostic half - [TpCollective], the adapter trait the CLI implements over its transport,
//! [TpLayout], [TpRuntime] and the exact [BitGather] - lives in [sparkflash_weights::tp] and is
//! re-exported here, so every pack shards through the same seam. What stays is V4.1's own:
//! [ResidentSource], a source that reads every non-Engram tensor a rank needs once, at
//! construction, and serves the cached raw bytes for the life of the engine. The reference pass
//! calls it per token and issues no whole-tensor read against the underlying source after the
//! engine exists. The two Engram tables per Engram layer are deliberately NOT cached: they are
//! ~98 GB each and stay in the underlying source, paged through [WeightSource::read_ranges].
//!
//! The Phase-1 sharding map is deliberately narrow: only the routed experts are split. Attention
//! (wq_a/wkv/wq_b/wo_a/wo_b), the embedding, the head and the shared expert are REPLICATED, so the
//! one collective per layer is the routed-FFN all-reduce after the local expert bodies. The
//! deviations from the full section-3 map and the numeric bar they do (not) spend are recorded in
//! `docs/archive/build/engine-tp-seams.md`.

use std::collections::BTreeMap;

use crate::tensor::{TensorPlan, TensorSpec};
use crate::weights::{WeightError, WeightSource};

pub use sparkflash_weights::tp::{
    BitGather, BitGatherHandle, EXACT_WIRE_REL_L2_BAR, LocalBus, LocalRank, SingleRank,
    TpCollective, TpLayout, TpRuntime, Wire, canonical_reduce_into,
};

/// Whether a plan name is one of the two paged Engram table tensors of an Engram layer.
///
/// The released tables are ~98 GB each. [engram_forward](crate::forward::engram) pages the rows a
/// chunk looks up through [WeightSource::read_ranges] rather than materialising the table, so the
/// resident cache leaves exactly these names in the underlying source. Every other planned tensor -
/// the rank's routed-expert slice and every replicated tensor - is resident.
#[must_use]
pub fn is_engram_table(name: &str) -> bool {
    name.ends_with(".engram.embed.weight") || name.ends_with(".engram.embed.scale")
}

/// The names [ResidentSource::load] should consider, so a caller can feed the plan it already has.
///
/// [TensorPlan] implements it with every planned name, which is what [crate::forward::InferenceEngine::new_tp]
/// uses. The CLI's sharded plan is not an engine type; the CLI implements this trait for a thin
/// newtype over its own plan (the orphan rule allows the local newtype), and the resident source then
/// caches the same set: the rank's routed slices plus every replicated tensor, with the Engram tables
/// paged. That is what makes the task-59 swap mechanical.
pub trait ResidentPlan {
    /// Every tensor name to consider, in a stable order.
    ///
    /// Engram tables may be included: [ResidentSource::load] skips them and pages them instead.
    fn resident_plan_names(&self) -> Vec<String>;
}

impl ResidentPlan for TensorPlan {
    fn resident_plan_names(&self) -> Vec<String> {
        self.names().map(str::to_owned).collect()
    }
}

/// A source that holds every non-Engram tensor it serves in memory, read once at construction.
///
/// [ResidentSource::load] reads the plan's tensors through the rank's source exactly once, except
/// the two Engram table tensors of each Engram layer, which stay in the underlying source and are
/// paged per token through [WeightSource::read_ranges]. The engine then reads from the map, so the
/// per-token forward issues no whole-tensor read against the disk/transport source at all. The cache
/// is raw bytes (not decoded f32), decoded on each use: the node target is every non-Engram weight
/// resident (~84.7 GiB of shard plus ~9.8 GiB of replicated tensors per rank), not the ~4x that a
/// decoded-f32 copy would need. A whole-tensor read still clones the cached bytes because
/// [WeightSource::read] returns owned bytes; that is memory bandwidth, not source I/O.
pub struct ResidentSource {
    inner: Box<dyn WeightSource>,
    tensors: BTreeMap<String, (TensorSpec, Vec<u8>)>,
    resident_bytes: usize,
    paged: Vec<String>,
}

impl ResidentSource {
    /// Reads every non-Engram tensor the plan names from `source`, once, into a resident map.
    ///
    /// The served spec (which may be a rank-local shape for a sharded tensor) is kept next to the
    /// bytes, because the forward checks the local geometry against the local spec. Engram tables are
    /// skipped and remembered in [ResidentSource::paged_names]; reads and byte ranges for them pass
    /// through to `source`.
    ///
    /// # Errors
    ///
    /// [WeightError] from the first spec or read the source refuses.
    pub fn load(
        source: Box<dyn WeightSource>,
        plan: &dyn ResidentPlan,
    ) -> Result<Self, WeightError> {
        let mut tensors = BTreeMap::new();
        let mut paged = Vec::new();
        let mut resident_bytes = 0usize;
        for name in plan.resident_plan_names() {
            if is_engram_table(&name) {
                paged.push(name);
                continue;
            }
            let spec = source.spec(&name)?;
            let bytes = source.read(&name)?;
            resident_bytes += bytes.len();
            tensors.insert(name, (spec, bytes));
        }
        Ok(Self {
            inner: source,
            tensors,
            resident_bytes,
            paged,
        })
    }

    /// The number of resident tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The resident bytes across every tensor.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// The planned names that are deliberately not resident and are paged through the source.
    #[must_use]
    pub fn paged_names(&self) -> &[String] {
        &self.paged
    }

    /// Whether one tensor is served from the cache rather than paged from the source.
    #[must_use]
    pub fn is_resident(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// Concatenates the byte ranges of one cached tensor in the order given.
fn slice_ranges(bytes: &[u8], ranges: &[(u64, usize)], name: &str) -> Result<Vec<u8>, WeightError> {
    let total: usize = ranges.iter().map(|&(_, len)| len).sum();
    let mut out = Vec::with_capacity(total);
    for &(offset, len) in ranges {
        let start = usize::try_from(offset).map_err(|_| WeightError::Malformed {
            name: name.to_owned(),
            detail: format!("range offset {offset} does not fit a usize"),
        })?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| WeightError::Malformed {
                name: name.to_owned(),
                detail: format!("range {start}+{len} overflows"),
            })?;
        let slice = bytes
            .get(start..end)
            .ok_or_else(|| WeightError::Malformed {
                name: name.to_owned(),
                detail: format!(
                    "range {start}..{end} is past the resident {} bytes",
                    bytes.len()
                ),
            })?;
        out.extend_from_slice(slice);
    }
    Ok(out)
}

impl WeightSource for ResidentSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.tensors.get(name) {
            Some((spec, _)) => Ok(spec.clone()),
            None => self.inner.spec(name),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        match self.tensors.get(name) {
            Some((_, bytes)) => Ok(bytes.clone()),
            // Paged Engram tables are normally reached by read_ranges; the whole-tensor read stays
            // for a small table in a fixture, and the engine's limit refuses a released-size one.
            None => self.inner.read(name),
        }
    }

    fn read_view(&self, name: &str) -> Result<Option<&[u8]>, WeightError> {
        // The resident cache holds every non-Engram tensor for the process's life, so the forward
        // can borrow the stored bytes instead of copying them on every step.
        Ok(self.tensors.get(name).map(|(_, bytes)| bytes.as_slice()))
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        match self.tensors.get(name) {
            Some((_, bytes)) => Some(slice_ranges(bytes, ranges, name)),
            None => self.inner.read_ranges(name, ranges),
        }
    }
}
