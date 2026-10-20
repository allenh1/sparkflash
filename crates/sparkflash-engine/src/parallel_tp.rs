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
//! * [ResidentSource], a source that reads every tensor a rank needs once, at construction, and
//!   serves the cached bytes for the life of the engine. The reference pass calls it per token; no
//!   read reaches the underlying source after the engine exists.
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

/// A source that holds every tensor it serves in memory, read once at construction.
///
/// [ResidentSource::load] reads the plan's tensors through the rank's source exactly once. The
/// engine then reads from the map, so the per-token forward issues no reads against the disk/transport
/// source at all: that is the resident property the single-node serve path needs. The cache is raw
/// bytes (not decoded f32), so its size is the shard size the source already materialised; the node
/// runner can pass a streaming source instead and keep the CLI-side runner.
#[derive(Debug)]
pub struct ResidentSource {
    tensors: BTreeMap<String, (TensorSpec, Vec<u8>)>,
}

impl ResidentSource {
    /// Reads every tensor the plan names from `source`, once, into a resident map.
    ///
    /// The served spec (which may be a rank-local shape for a sharded tensor) is kept next to the
    /// bytes, because the forward checks the local geometry against the local spec.
    ///
    /// # Errors
    ///
    /// [WeightError] from the first spec or read the source refuses.
    pub fn load(source: &dyn WeightSource, plan: &TensorPlan) -> Result<Self, WeightError> {
        let mut tensors = BTreeMap::new();
        for wanted in plan.specs() {
            let spec = source.spec(&wanted.name)?;
            let bytes = source.read(&wanted.name)?;
            tensors.insert(wanted.name.clone(), (spec, bytes));
        }
        Ok(Self { tensors })
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
        self.tensors.values().map(|(_, bytes)| bytes.len()).sum()
    }
}

impl WeightSource for ResidentSource {
    fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.tensors.get(name) {
            Some((spec, _)) => Ok(spec.clone()),
            None => Err(WeightError::Missing {
                name: name.to_owned(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        match self.tensors.get(name) {
            Some((_, bytes)) => Ok(bytes.clone()),
            None => Err(WeightError::Missing {
                name: name.to_owned(),
            }),
        }
    }
}
