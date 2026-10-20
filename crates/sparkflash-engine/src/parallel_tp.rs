//! The tensor-parallel seam: a rank's collective adapter and a resident shard source.
//!
//! This module is the engine half of the Phase-1 TP path described in
//! `docs/build/tp4-design.md` section 3 and `docs/build/engine-tp-seams.md`. It holds three things:
//!
//! * [TpCollective], the adapter trait the CLI implements over its transport. No transport type
//!   crosses this boundary: the engine only sees f32 buffers and a rank/world pair, which is what
//!   keeps the engine crate comm-free.
//! * [TpLayout], the rank/world pair with the validation the forward must pass before it shards
//!   anything.
//! * [ResidentSource], a source that reads every non-Engram tensor a rank needs once, at
//!   construction, and serves the cached raw bytes for the life of the engine. The reference pass
//!   calls it per token and issues no whole-tensor read against the underlying source after the
//!   engine exists. The two Engram tables per Engram layer are deliberately NOT cached: they are
//!   ~98 GB each and stay in the underlying source, paged through [WeightSource::read_ranges].
//!
//! The Phase-1 sharding map is deliberately narrow: only the routed experts are split. Attention
//! (wq_a/wkv/wq_b/wo_a/wo_b), the embedding, the head and the shared expert are REPLICATED, so the
//! one collective per layer is the routed-FFN all-reduce after the local expert bodies. The
//! deviations from the full section-3 map and the numeric bar they do (not) spend are recorded in
//! `docs/build/engine-tp-seams.md`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::linalg::EngineError;
use crate::quant::bf16_round;
use crate::tensor::{TensorPlan, TensorSpec};
use crate::weights::{WeightError, WeightSource};

/// Where this process sits in the TP group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpLayout {
    /// This rank's index, `0..world_size`.
    pub rank: usize,
    /// The number of ranks the model is split over.
    pub world_size: usize,
}

impl TpLayout {
    /// Checks the pair, and that the world can split the routed intermediate.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when `world_size` is zero, `rank` is not below it, or one of the
    /// sharded widths does not divide by the world.
    pub fn validate(&self, moe_inter_dim: usize) -> Result<(), EngineError> {
        if self.world_size == 0 {
            return Err(EngineError::Shape {
                what: "TP layout".to_owned(),
                detail: "world_size is zero".to_owned(),
            });
        }
        if self.rank >= self.world_size {
            return Err(EngineError::Shape {
                what: "TP layout".to_owned(),
                detail: format!(
                    "rank {} is outside a world of {}",
                    self.rank, self.world_size
                ),
            });
        }
        if moe_inter_dim == 0 || !moe_inter_dim.is_multiple_of(self.world_size) {
            return Err(EngineError::Shape {
                what: "TP layout".to_owned(),
                detail: format!(
                    "moe_inter_dim {moe_inter_dim} does not split over {} ranks",
                    self.world_size
                ),
            });
        }
        Ok(())
    }

    /// The routed intermediate width this rank holds: `moe_inter_dim / world_size`.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] as [TpLayout::validate].
    pub fn local_moe_inter(&self, moe_inter_dim: usize) -> Result<usize, EngineError> {
        self.validate(moe_inter_dim)?;
        Ok(moe_inter_dim / self.world_size)
    }
}

/// The engine's view of the TP group's collectives.
///
/// The implementation lives in the CLI (over its TCP transport); the engine never names a
/// transport. Deriving `Send` is required because the engine stores the adapter and its `&self`
/// forward paths run on several threads.
pub trait TpCollective: Send {
    /// The number of ranks.
    fn world(&self) -> usize;
    /// This rank's index.
    fn rank(&self) -> usize;
    /// A short label for evidence ledgers, e.g. `"single-rank identity"` or `"tcp rank 2/4"`.
    fn label(&self) -> &'static str;

    /// Reduces one f32 buffer across the group, in place.
    ///
    /// The wire format is canonical: every rank's values are rounded once to bf16 (ties to even),
    /// the f32 values are accumulated in ascending rank order from `0.0`, the sum is rounded once
    /// to bf16 (ties to even) and widened back to f32. With `world() == 1` this MUST be a strict
    /// no-op: no rounding and no write, because the one-rank engine is the bit-identical reference.
    fn all_reduce_assign(&mut self, values: &mut [f32]) -> Result<(), EngineError>;

    /// Gathers `local.len()` values from every rank, in ascending rank order, into `out`.
    ///
    /// Each element is bf16-rounded on the wire and widened back to f32. With `world() == 1` this
    /// MUST be a plain copy of `local` into `out` and nothing else.
    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError>;

    /// Sends `values` to `to`.
    ///
    /// The reference seam has no send/recv leg yet; the default body refuses instead of silently
    /// succeeding. A transport adapter overrides it when the design grows one.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] from the default body.
    fn send(&mut self, _to: usize, _values: &[f32]) -> Result<(), EngineError> {
        Err(EngineError::Shape {
            what: "TpCollective::send".to_owned(),
            detail: "the reference seam has no send leg yet".to_owned(),
        })
    }

    /// Receives `out.len()` values from `from`.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] from the default body.
    fn recv(&mut self, _from: usize, _out: &mut [f32]) -> Result<(), EngineError> {
        Err(EngineError::Shape {
            what: "TpCollective::recv".to_owned(),
            detail: "the reference seam has no recv leg yet".to_owned(),
        })
    }
}

/// The one-rank adapter: every collective is a strict no-op, so the seam is value-transparent.
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleRank;

impl TpCollective for SingleRank {
    fn world(&self) -> usize {
        1
    }
    fn rank(&self) -> usize {
        0
    }
    fn label(&self) -> &'static str {
        "single-rank identity"
    }
    fn all_reduce_assign(&mut self, _values: &mut [f32]) -> Result<(), EngineError> {
        Ok(())
    }
    fn all_gather_assign(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        if out.len() != local.len() {
            return Err(EngineError::Shape {
                what: "single-rank all_gather".to_owned(),
                detail: format!(
                    "one rank gathers {} values into {} slots",
                    local.len(),
                    out.len()
                ),
            });
        }
        out.copy_from_slice(local);
        Ok(())
    }
}

/// The canonical reduce over `world` payloads (including this rank's), for tests and any adapter
/// that wants the engine's exact reference arithmetic.
///
/// Each payload is bf16-rounded, summed in f32 in ascending rank order from `0.0`, then rounded
/// once to bf16 and widened back. A single payload is returned unchanged: that is the world-1
/// no-op the trait promises.
///
/// # Errors
///
/// [EngineError::Shape] when `payloads` is empty, a payload is not `values.len()` long, or `out`
/// is not `values.len()` long.
pub fn canonical_reduce_into(values: &mut [f32], payloads: &[&[f32]]) -> Result<(), EngineError> {
    if payloads.is_empty() {
        return Err(EngineError::Shape {
            what: "canonical TP reduce".to_owned(),
            detail: "no rank payloads".to_owned(),
        });
    }
    for payload in payloads {
        if payload.len() != values.len() {
            return Err(EngineError::Shape {
                what: "canonical TP reduce".to_owned(),
                detail: format!(
                    "{} values in one payload, {} in the output",
                    payload.len(),
                    values.len()
                ),
            });
        }
    }
    if payloads.len() == 1 {
        values.copy_from_slice(payloads[0]);
        return Ok(());
    }
    let mut accumulator = vec![0.0f32; values.len()];
    for payload in payloads {
        for (sum, value) in accumulator.iter_mut().zip(payload.iter()) {
            *sum += bf16_round(*value);
        }
    }
    for (out, sum) in values.iter_mut().zip(accumulator.iter()) {
        *out = bf16_round(*sum);
    }
    Ok(())
}

/// The TP state an engine carries while it runs the sharded reference forward.
pub struct TpRuntime {
    layout: TpLayout,
    local_moe_inter: usize,
    collective: Mutex<Box<dyn TpCollective>>,
}

impl TpRuntime {
    /// Validates the layout and the adapter against it.
    ///
    /// # Errors
    ///
    /// [EngineError::Shape] when the layout cannot split the routed intermediate, or when the
    /// adapter reports a different rank/world pair.
    pub fn new(
        layout: TpLayout,
        moe_inter_dim: usize,
        collective: Box<dyn TpCollective>,
    ) -> Result<Self, EngineError> {
        let local_moe_inter = layout.local_moe_inter(moe_inter_dim)?;
        if collective.world() != layout.world_size {
            return Err(EngineError::Shape {
                what: "TP collective".to_owned(),
                detail: format!(
                    "the adapter reports world {} but the layout says {}",
                    collective.world(),
                    layout.world_size
                ),
            });
        }
        if collective.rank() != layout.rank {
            return Err(EngineError::Shape {
                what: "TP collective".to_owned(),
                detail: format!(
                    "the adapter reports rank {} but the layout says {}",
                    collective.rank(),
                    layout.rank
                ),
            });
        }
        Ok(Self {
            layout,
            local_moe_inter,
            collective: Mutex::new(collective),
        })
    }

    /// This rank's index.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.layout.rank
    }

    /// The number of ranks.
    #[must_use]
    pub fn world(&self) -> usize {
        self.layout.world_size
    }

    /// The routed intermediate width this rank holds.
    #[must_use]
    pub fn local_moe_inter(&self) -> usize {
        self.local_moe_inter
    }

    /// The adapter's evidence label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .label()
    }

    /// Reduces one seam buffer in place through the adapter.
    ///
    /// # Errors
    ///
    /// Whatever the adapter reports, plus [EngineError::Shape] when the buffer is empty.
    pub fn all_reduce(&self, values: &mut [f32]) -> Result<(), EngineError> {
        if values.is_empty() {
            return Err(EngineError::Shape {
                what: "TP seam".to_owned(),
                detail: "an empty buffer has no seam to reduce".to_owned(),
            });
        }
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .all_reduce_assign(values)
    }

    /// Gathers one seam's local values through the adapter.
    ///
    /// # Errors
    ///
    /// Whatever the adapter reports.
    pub fn all_gather(&self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        self.collective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .all_gather_assign(local, out)
    }
}

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
