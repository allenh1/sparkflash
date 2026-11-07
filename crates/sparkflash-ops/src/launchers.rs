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

//! The shared launch descriptors for the parts both released arms execute.
//!
//! A [PartRequirement] names a part at one rank's geometry; [ResolvedKernel] names the image
//! and symbol the catalogue chose for it. A launcher binds the two: it fixes the grid, block and
//! shared-memory reservation once (from the same rules the arms already use) and leaves the
//! argument list to the call, because the operands are per-step buffers.
//!
//! These are descriptors validated by [crate::parts_backend::PartsDeviceRank], not the arms'
//! launch path: the Qwen and GLM numeric forwards still spell their own launches (their argument
//! lists are per-step and per-arm). A launcher here proves that the plan's geometry, the
//! catalogue's symbol and the deterministic grid agree before a weight is uploaded.
//!
//! The descriptors are discarded by [crate::parts_backend::PartsDeviceBackend::into_arm], which
//! the packs call after an [crate::engine_plan::EngineBuilder] open: the arms keep launching
//! their own bodies, and the shared rank view is retained only as the boot proof. Wiring a
//! launcher into the numeric forward is a separate change; until then these geometries are
//! validated against the plan and the catalogue, not executed.

use sparkflash_kernels::cuda_driver::{Cuda, Kernel, RunStream};

use crate::attention::{AttentionKind, AttentionPlan};
use crate::engine_plan::{PartRequirement, PlanError};
use crate::gdn::{GDN_VALUE, gdn_scan_shape_for};
use crate::plan::{Launch, OpPlan, StreamId};
use crate::resolver::ResolvedKernel;
use crate::scan::ScanKind;

/// One fixed launch geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLaunch {
    /// The catalogue-selected kernel.
    pub kernel: crate::plan::KernelId,
    /// Grid dimensions.
    pub grid: (u32, u32, u32),
    /// Block dimensions.
    pub block: (u32, u32, u32),
    /// Dynamic shared memory.
    pub shared_bytes: u32,
    /// The stream the launch is ordered on.
    pub stream: StreamId,
}

impl KernelLaunch {
    /// Launch the loaded `kernel` on `stream` with this geometry and `args`.
    ///
    /// The stream and the shared-memory reservation are part of the descriptor, not a second
    /// decision at the call site: the launch goes through [Cuda::launch_on], and a handle loaded
    /// with a different reservation than the descriptor names is refused rather than launched
    /// with the wrong block footprint.
    ///
    /// # Errors
    /// A loaded kernel whose shared reservation differs from the descriptor, or the driver's
    /// launch refusal.
    pub fn launch_on(
        &self,
        cuda: &Cuda,
        stream: &RunStream,
        kernel: Kernel,
        args: &[u64],
    ) -> Result<(), String> {
        if kernel.shared_bytes() != self.shared_bytes {
            return Err(format!(
                "{} was loaded with {} shared bytes; the launcher describes {} on stream {}",
                kernel.name(),
                kernel.shared_bytes(),
                self.shared_bytes,
                self.stream.0
            ));
        }
        cuda.launch_on(stream, kernel, self.grid, self.block, args)
    }
}

impl From<Launch> for KernelLaunch {
    fn from(launch: Launch) -> Self {
        Self {
            kernel: launch.kernel,
            grid: launch.grid,
            block: launch.block,
            shared_bytes: launch.shared_bytes,
            stream: launch.stream,
        }
    }
}

/// The GDN/KDA scan launcher the two hybrid arms share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearScanLauncher {
    /// The recurrence.
    pub kind: ScanKind,
    /// The catalogue-selected kernel.
    pub kernel: crate::plan::KernelId,
    /// The rank-local key-head count, for a GDN scan; the head count for a KDA scan.
    pub key_heads: usize,
    /// The rank-local head count the grid launches: GDN's value heads, KDA's heads.
    ///
    /// GDN stores key heads in [crate::plan::PartGeometry::heads] and value heads in
    /// [crate::plan::PartGeometry::kv_heads]; the scan's recurrence is over the value heads, so
    /// the GDN grid must use `kv_heads`. A KDA head is both, and the grid uses `heads`.
    pub heads: usize,
    /// The value head width.
    pub value_dim: usize,
    /// The scan geometry.
    pub launch: KernelLaunch,
}

impl LinearScanLauncher {
    /// Bind one scan requirement to its resolved kernel.
    ///
    /// # Errors
    /// A requirement of the other recurrence.
    pub fn new(
        requirement: &PartRequirement,
        resolved: &ResolvedKernel,
    ) -> Result<Self, PlanError> {
        let kind = match requirement.part {
            sparkflash_model::Part::GdnScan => ScanKind::Gdn,
            sparkflash_model::Part::KdaScan => ScanKind::Kda,
            other => {
                return Err(PlanError::Invalid(format!(
                    "a {other} requirement is not a linear-attention scan"
                )));
            }
        };
        // GDN stores key heads in `heads` and value heads in `kv_heads`; its recurrence runs
        // one grid row a value head. KDA's head is both, and its grid uses `heads`.
        let (key_heads, heads) = match kind {
            ScanKind::Gdn => (
                requirement.geometry.heads.max(1),
                requirement.geometry.kv_heads.max(1),
            ),
            ScanKind::Kda => (
                requirement.geometry.heads.max(1),
                requirement.geometry.heads.max(1),
            ),
        };
        let value_dim = if requirement.geometry.v_head == 0 {
            GDN_VALUE
        } else {
            requirement.geometry.v_head
        };
        let (grid, block) = gdn_scan_shape_for(heads, value_dim);
        Ok(Self {
            kind,
            kernel: resolved.id(),
            key_heads,
            heads,
            value_dim,
            launch: KernelLaunch {
                kernel: resolved.id(),
                grid,
                block: (block, 1, 1),
                shared_bytes: 0,
                stream: StreamId(0),
            },
        })
    }

    /// The grid and block the scan runs with.
    #[must_use]
    pub const fn shape(&self) -> ((u32, u32, u32), u32) {
        (self.launch.grid, self.launch.block.0)
    }
}

/// The grouped-query and absorbed-MLA attention launcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionLauncher {
    /// The attention arithmetic.
    pub kind: AttentionKind,
    /// The catalogue-selected kernel.
    pub kernel: crate::plan::KernelId,
    /// The fixed launch geometry.
    pub launch: KernelLaunch,
}

impl AttentionLauncher {
    /// Bind an attention requirement to its resolved kernel.
    ///
    /// # Errors
    /// A requirement the attention plans refuse.
    pub fn new(
        requirement: &PartRequirement,
        resolved: &ResolvedKernel,
        rows: usize,
        context_len: usize,
        stream: StreamId,
    ) -> Result<Self, PlanError> {
        let kind = match requirement.part {
            sparkflash_model::Part::GqaAttention => {
                let kv_heads = requirement.geometry.kv_heads.max(1);
                AttentionKind::Gqa {
                    query_heads: requirement.geometry.heads.max(1),
                    kv_heads,
                }
            }
            sparkflash_model::Part::MlaAttention => AttentionKind::Mla,
            sparkflash_model::Part::SparseIndexer => AttentionKind::Sparse {
                selected: requirement.geometry.top_k.max(1),
                block: requirement.geometry.ratio.max(1),
            },
            other => {
                return Err(PlanError::Invalid(format!(
                    "the attention launcher does not serve {other}"
                )));
            }
        };
        // The shared rank already resolved this part instance. Reuse that kernel instead of
        // resolving a second one inside AttentionPlan::new, so the descriptor's kernel and the
        // grid/block the arm launches describe the same symbol.
        let plan = AttentionPlan::with_resolved(
            requirement,
            kind,
            rows.max(1),
            context_len.max(1),
            requirement.geometry.head_dim.max(1),
            stream,
            resolved.id(),
        )
        .map_err(|error| PlanError::Invalid(error.to_string()))?;
        let launch = KernelLaunch::from(plan.launch());
        debug_assert_eq!(launch.kernel, resolved.id());
        Ok(Self {
            kind,
            kernel: resolved.id(),
            launch,
        })
    }
}

/// One symbol of a multi-kernel part, with the launch geometry it runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolLaunch {
    /// The catalogue symbol name.
    pub symbol: &'static str,
    /// The fixed launch geometry.
    pub launch: KernelLaunch,
}

/// Qwen's `q4e_qsa_scores` pools a 64-block tile a grid row (`QSA_TILE` in the arm).
const QSA_SCORE_TILE: usize = 64;
/// `q4e_qsa_pool` runs 128 threads a pooled block (the Qwen arm's pool launch).
const QSA_POOL_BLOCK: u32 = 128;
/// The score and pick bodies of both released families run 256 threads a block
/// (`Q4E_QSA_TILE`/`SCORE_TILE` are 64-pool score tiles inside that block).
const INDEX_SCORE_BLOCK: u32 = 256;
/// GLM's `q4e_glm_index_scores` tiles its pools 64 at a time (`SCORE_TILE` in the arm).
const GLM_SCORE_TILE: usize = 64;

/// The sparse-indexer pool, score and selection launcher.
///
/// The indexer is three bodies in one pass - pool, score and pick - so this launcher carries
/// every symbol the chosen image exports for the requirement in catalogue order, not just the one
/// the resolver selected. The primary [Self::launch] is the resolved symbol; [Self::symbols]
/// is the ordered sequence the numeric forward spells.
///
/// The three bodies do not share one geometry: Qwen's pool runs a 128-thread block a pooled
/// position while its scores run a 256-thread block a (64-pool tile, query row), and its pick
/// runs one 256-thread block a query row. GLM pools a query row a 256-thread block and tiles its
/// scores by `SCORE_TILE`. Each [SymbolLaunch] therefore carries the geometry of its own
/// symbol, computed from the requirement's pool ratio and the rows and absolute position the
/// launch describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseIndexerLauncher {
    /// The catalogue-selected kernel.
    pub kernel: crate::plan::KernelId,
    /// The resolved symbol's fixed launch geometry.
    pub launch: KernelLaunch,
    /// Every symbol of the chosen image that accepts the requirement's precision, in
    /// catalogue order (pool, score, pick for both released families).
    pub symbols: Vec<SymbolLaunch>,
}

impl SparseIndexerLauncher {
    /// Bind an indexer requirement to its resolved kernel.
    ///
    /// `rows` is the number of query rows the pass covers and `position` the absolute position
    /// of its first row. Both feed the per-symbol grids: the pool and score bodies depend on how
    /// many pools the last row can see, and every body's second grid dimension is the row count.
    ///
    /// # Errors
    /// A requirement that names no indexer geometry.
    pub fn new(
        requirement: &PartRequirement,
        resolved: &ResolvedKernel,
        rows: usize,
        position: usize,
    ) -> Result<Self, PlanError> {
        if requirement.part != sparkflash_model::Part::SparseIndexer {
            return Err(PlanError::Invalid(format!(
                "a {} requirement is not a sparse indexer",
                requirement.part
            )));
        }
        let (grid, block) = indexer_geometry(resolved.id().symbol, requirement, rows, position);
        let launch = KernelLaunch {
            kernel: resolved.id(),
            grid,
            block,
            shared_bytes: 0,
            stream: StreamId(0),
        };
        let symbols = crate::parts::CATALOGUE
            .iter()
            .find(|entry| {
                entry.part == requirement.part
                    && entry.target == resolved.target
                    && entry.image == resolved.image
            })
            .map(|entry| {
                entry
                    .symbols
                    .iter()
                    .filter(|symbol| symbol.accepts(requirement.precision))
                    .map(|symbol| {
                        let (grid, block) =
                            indexer_geometry(symbol.name, requirement, rows, position);
                        SymbolLaunch {
                            symbol: symbol.name,
                            launch: KernelLaunch {
                                kernel: crate::plan::KernelId::new(
                                    resolved.target,
                                    resolved.image,
                                    symbol.name,
                                ),
                                grid,
                                block,
                                ..launch
                            },
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            kernel: resolved.id(),
            launch,
            symbols,
        })
    }
}

/// The grid and block one indexer symbol runs at, from its own released-arm rule.
///
/// A symbol this tree does not launch (the DeepSeek reference selection bodies) keeps the one
/// head-block-a-position descriptor it had before, which is the shared validation fallback.
fn indexer_geometry(
    symbol: &str,
    requirement: &PartRequirement,
    rows: usize,
    position: usize,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    let rows = rows.max(1);
    let pool = requirement.geometry.ratio.max(1);
    // The last row's pool count: a pool is visible to the row at or past it.
    let pools = position.saturating_add(rows) / pool;
    let pools = pools.max(1);
    match symbol {
        "q4e_qsa_pool" => ((pools as u32, 1, 1), (QSA_POOL_BLOCK, 1, 1)),
        "q4e_qsa_scores" => (
            (pools.div_ceil(QSA_SCORE_TILE) as u32, rows as u32, 1),
            (INDEX_SCORE_BLOCK, 1, 1),
        ),
        "q4e_qsa_pick" => ((rows as u32, 1, 1), (INDEX_SCORE_BLOCK, 1, 1)),
        "q4e_glm_pool" => ((rows as u32, 1, 1), (INDEX_SCORE_BLOCK, 1, 1)),
        "q4e_glm_index_scores" => (
            (pools.div_ceil(GLM_SCORE_TILE) as u32, rows as u32, 1),
            (INDEX_SCORE_BLOCK, 1, 1),
        ),
        "q4e_glm_pick" => ((rows as u32, 1, 1), (INDEX_SCORE_BLOCK, 1, 1)),
        _ => (
            (requirement.geometry.heads.max(1) as u32, 1, 1),
            (QSA_POOL_BLOCK, 1, 1),
        ),
    }
}

/// The router, expert plan, token GEMV, grouped prefill, shared expert and combine launchers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeLauncher {
    /// The catalogue-selected MoE kernel.
    pub kernel: crate::plan::KernelId,
    /// The routed experts per layer.
    pub experts: usize,
    /// Experts one token activates.
    pub top_k: usize,
    /// The rank-local expert width.
    pub intermediate: usize,
    /// The shared expert's rank-local width.
    pub shared: usize,
}

impl MoeLauncher {
    /// Bind a MoE requirement to its resolved kernel.
    ///
    /// # Errors
    /// A requirement that is not a MoE part.
    pub fn new(
        requirement: &PartRequirement,
        resolved: &ResolvedKernel,
    ) -> Result<Self, PlanError> {
        if requirement.part != sparkflash_model::Part::Moe {
            return Err(PlanError::Invalid(format!(
                "a {} requirement is not a MoE part",
                requirement.part
            )));
        }
        Ok(Self {
            kernel: resolved.id(),
            experts: requirement.geometry.experts,
            top_k: requirement.geometry.top_k,
            intermediate: requirement.geometry.intermediate,
            shared: requirement.geometry.shared,
        })
    }

    /// The tokens the router launches one block a row for.
    #[must_use]
    pub fn router_grid(rows: usize) -> (u32, u32, u32) {
        (rows.max(1) as u32, 1, 1)
    }
}

/// The plain, low-rank hyper-connection and manifold mHC launcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualLauncher {
    /// The catalogue-selected kernel.
    pub kernel: crate::plan::KernelId,
    /// Residual streams the layer carries.
    pub streams: usize,
    /// The low-rank bottleneck, or zero for a plain residual.
    pub lowrank: usize,
    /// Whether the mix is manifold-constrained.
    pub manifold: bool,
}

impl ResidualLauncher {
    /// Bind a residual requirement to its resolved kernel.
    ///
    /// # Errors
    /// A requirement that is not a hyper-connection part.
    pub fn new(
        requirement: &PartRequirement,
        resolved: &ResolvedKernel,
    ) -> Result<Self, PlanError> {
        if !matches!(
            requirement.part,
            sparkflash_model::Part::HyperConnection
                | sparkflash_model::Part::ManifoldHyperConnection
        ) {
            return Err(PlanError::Invalid(format!(
                "a {} requirement is not a residual mixer",
                requirement.part
            )));
        }
        Ok(Self {
            kernel: resolved.id(),
            streams: requirement.geometry.streams,
            lowrank: requirement.geometry.intermediate,
            manifold: requirement.part == sparkflash_model::Part::ManifoldHyperConnection,
        })
    }
}

/// The launchers of one decoder layer, in the shared walk's order.
#[derive(Debug, Clone)]
pub enum LayerLauncher {
    /// A GDN or KDA scan.
    LinearScan(LinearScanLauncher),
    /// GQA or absorbed MLA attention.
    Attention(AttentionLauncher),
    /// The sparse indexer.
    Indexer(SparseIndexerLauncher),
    /// The MoE block.
    Moe(MoeLauncher),
    /// A residual mixer.
    Residual(ResidualLauncher),
    /// A dense projection or activation the typed plans already cover.
    Other(crate::plan::KernelId),
    /// A part this build's catalogue has no kernel for at this geometry.
    Unresolved(sparkflash_model::Part),
}

impl LayerLauncher {
    /// The ordered symbol launches this part records, empty for the single-symbol parts.
    #[must_use]
    pub fn symbols(&self) -> &[SymbolLaunch] {
        match self {
            Self::Indexer(indexer) => &indexer.symbols,
            _ => &[],
        }
    }
}
