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

//! The layer plan: the attention variant of each row and the DAG its KV reads imply.
//!
//! The 43 rows of DeepSeek-V4.1-Flash - forty backbone layers followed by three MTP stages - are not
//! a chain of identical blocks. Three released config lists make the forward a DAG, and two more
//! mark special layers:
//!
//! - `compress_ratios` gives every row its compression ratio: `0` means no compressor and no
//!   indexer (sliding-window attention only), `1` means a compressor that normalises one token into
//!   one latent, and `2` means a compressor that pools two tokens through a softmax gate. The list
//!   has one entry per planned row (`n_layers + n_mtp_layers`); the three stages have ratio 0.
//! - `kv_source_layer_ids = [2, 8, 14, 20]` are the layers that own the main compressed KV. Every
//!   other compressed layer reads the nearest preceding source's cache, so the forward order must
//!   place the source first. A reader's ratio always equals its source's; a ratio-2 layer that read a
//!   ratio-1 cache would be silently wrong.
//! - `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]` are the layers that publish
//!   `topk_idxs`. Layers 24, 28, 32 and 36 are *Reindex* layers: they run their own indexer but do
//!   not own KV and do not publish main KV; they recompute their own Top-K inside the HSI pool. The
//!   Reuse layers after a Reindex layer read *that* layer's indices, not the KV source's.
//! - `candidate_source_layer_id = 20` builds the HSI candidate pool. It must be an index source:
//!   `select_candidate_blocks` runs inside `Indexer.__call__` (model.py:569-571) and
//!   `is_candidate_source` does not consult `owns_k` (model.py:502-503). It need not own KV, though
//!   the released layer 20 does both (docs/architecture/01-csa2-attention.md:56). Layer 20 and the
//!   four Reindex layers after it are the only layers with a `candidate_source` of `Some(20)`.
//! - `engram_layer_ids = [1, 14]` carry an Engram table.
//! - `dspark_target_layer_ids = [37, 38, 39]` are the layers whose *input* hidden state the DSpark
//!   stages tap (model.py:1264-1266). They are ordinary Reuse layers and remain part of the base
//!   forward. What is outside the base forward is `mtp.{0..n_mtp_layers-1}`, the three DSpark
//!   stages at rows 40, 41 and 42 (docs/architecture/05-dspark.md:15-24); they are planned as
//!   [`LayerMode::Mtp`] rows with ratio 0, no KV or index source, no Engram and no DSpark tap.
//!
//! # Ownership is two independent flags
//!
//! Index ownership is independent of KV ownership. The reference computes `is_kv_source` and
//! `is_index_source` separately from the two lists (model.py:654-655), makes the indexer's
//! `owns_k` the KV-source set (model.py:500), lets a KV source without an indexer reuse
//! `shared_attn.topk_idxs` (model.py:725-726) while still publishing its own `compress_kv`
//! (model.py:746-748), and lets a Reindex layer consume the KV source's cache while publishing its
//! own Top-K. [`LayerMode::Full`] names a row that owns KV ([`LayerSpec::owns_kv`]);
//! [`LayerMode::Reindex`] a row that owns only an indexer; [`LayerMode::Reuse`] a row that owns
//! neither; [`LayerMode::SwaOnly`] a ratio-0 backbone row; [`LayerMode::Mtp`] a stage. A Full row
//! whose `owns_index` is false reads the preceding index source's Top-K, exactly as a Reuse row
//! does. The released configuration happens to list every KV source as an index source; the plan
//! does not require it.
//!
//! # The offset
//!
//! The reference shifts every reachable compressed id by the window cache's length before the joint
//! `[window_kv ; compress_kv]` attention: the attention caller computes that length as `kv.size(1)`
//! and passes it into `_compress_kv` at model.py:776, the shift adds it at model.py:580, and the
//! caller concatenates the window and compressed caches and indices at model.py:777-778. That
//! length is the chunk length during prefill (`start_pos == 0`) and `window_size` during decode
//! (`model.py:708-719`). [`LayerSpec::kv_offset`]
//! records the decode value, `window_size`, and [`LayerSpec::kv_offset_at`] applies the phase rule.
//! A ratio-0 row has no compressed KV, so it has no offset.
//!
//! # The execution order and the invariant
//!
//! The residual stream still flows layer to layer, so every topological order must respect
//! `i -> i + 1`; the plan's only possible order is `0, 1, ..., n_rows - 1`, with the three MTP
//! stages after the backbone and no edges of their own. [`LayerPlan::order`] states that order, and
//! [`LayerPlan::is_topologically_ordered`] checks every KV edge and every index edge against it. The
//! invariant is explicit: **every layer's KV source (and index source) precedes the layer that reads
//! it**. The plan is validated at construction, so a plan that violates it cannot exist; a test hands
//! a deliberately wrong edge to [`LayerPlan::from_specs`] to prove the check fires.
//!
//! # What is derived, and from where
//!
//! The plan is a function of [`ModelConfig`]: ratio, sources, Engram and DSpark membership, the
//! window, and the attention geometry. It is not transcribed from the checkpoint or from a table.
//! `tests/layer_plan.rs` builds it from the released HF `config.json` that MR !29 vendored and
//! asserts it against that document; one test additionally pins the whole 43-row table as literals,
//! so a wrong derivation cannot certify itself by agreeing with a reimplementation of its own rule.

use crate::config::ModelConfig;

/// The MLA's key/value head count: one, shared by all 64 query heads.
///
/// This is an architectural constant rather than a runtime-config field. The released reader checks
/// the checkpoint's `num_key_value_heads` is exactly one (`config.rs`), and the flat runtime document
/// has no such key because `ModelArgs` has no such field.
const KV_HEADS: usize = 1;

/// The attention geometry every backbone layer shares.
///
/// The variants of [`LayerMode`] differ in which KV and index tensors they own and read; the head
/// shape is uniform. Keeping it in one place makes the multiplication that sizes the attention
/// tensors (`n_heads * head_dim = 32,768`) a plan-level fact rather than a per-layer repetition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionGeometry {
    /// Query heads per layer: 64 in the released model.
    pub n_heads: usize,
    /// Key/value heads per layer: one in the released MLA.
    pub n_kv_heads: usize,
    /// The MLA latent width, including the rotated slice: 512.
    pub head_dim: usize,
    /// The trailing dimensions of [`Self::head_dim`] that RoPE rotates: 64.
    pub rope_head_dim: usize,
    /// The sliding window: 128 tokens.
    pub window: usize,
    /// Index entries kept per query by the sparse indexer: 512.
    pub index_topk: usize,
}

impl AttentionGeometry {
    /// Read the geometry from a configuration.
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        Self {
            n_heads: config.n_heads,
            n_kv_heads: KV_HEADS,
            head_dim: config.head_dim,
            rope_head_dim: config.rope_head_dim,
            window: config.window_size,
            index_topk: config.index_topk,
        }
    }
}

/// The attention variant a row runs, derived from the released id lists.
///
/// These are the four backbone modes of `docs/architecture/01-csa2-attention.md` section 2 and
/// `docs/engine/01-forward-order.md` section 4, plus the MTP stage row. A device engine dispatches
/// on this value, not on a layer index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerMode {
    /// Ratio 0: sliding-window attention only, with no compressor and no indexer. Backbone layers 0
    /// and 1.
    SwaOnly,
    /// A KV source: owns a compressor and the compressed-KV cache ([`LayerSpec::owns_kv`]). In the
    /// released model layers 2, 8, 14 and 20 also own an indexer and publish Top-K, but index
    /// ownership is independent (model.py:654-655): a Full row with `owns_index == false` reads the
    /// preceding index source's Top-K, exactly as a Reuse row does. Layer 20 is additionally the HSI
    /// pool builder, which [`LayerSpec::candidate_source`] records.
    Full,
    /// An index source that does not own KV: it reads the KV source's main KV and index keys and
    /// publishes its own Top-K. In the released model these are layers 24, 28, 32 and 36, all after
    /// the pool builder, so they also mask their scores with the HSI pool.
    Reindex,
    /// A reader: it consumes the nearest preceding KV source's main KV and the nearest preceding
    /// index source's Top-K without running an indexer. The other 30 compressed layers.
    Reuse,
    /// An MTP stage: rows 40, 41 and 42, which are `mtp.0`, `mtp.1` and `mtp.2`. A stage is a
    /// DSpark block with ratio 0 and window-only attention, but it is not a [`Self::SwaOnly`] layer:
    /// it runs after the backbone, owns no KV or index, reads no backbone cache and is not a DSpark
    /// tap (the taps are backbone layers 37, 38 and 39). The forward selects the DSpark block for
    /// this mode (docs/architecture/05-dspark.md section 1).
    Mtp,
}

/// How a layer compresses KV, derived from its ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionKind {
    /// Ratio 0: no compressor, no compressed KV; the layer attends over the sliding window only.
    Uncompressed,
    /// Ratio 1: the compressor is `norm(wkv(x))`, with no gate and no pooling. Layer 20 and
    /// layers 21 through 39.
    Direct,
    /// Ratio greater than one: the compressor pools `ratio` tokens with a softmax gate
    /// (`wgate`), storing one latent per group. Layers 2 through 19, at ratio 2.
    Pooled,
}

impl CompressionKind {
    /// The compression kind a ratio selects: 0 is [`Self::Uncompressed`], 1 is [`Self::Direct`],
    /// and anything larger is [`Self::Pooled`].
    #[must_use]
    pub fn for_ratio(ratio: usize) -> Self {
        match ratio {
            0 => Self::Uncompressed,
            1 => Self::Direct,
            _ => Self::Pooled,
        }
    }
}

/// One row's entry in the plan: a backbone layer or an MTP stage.
///
/// Each source field is `None` when the row produces that artifact rather than reads it;
/// [`Self::owns_kv`] and [`Self::owns_index`] say which, and an MTP stage owns neither. A Full row
/// that is not an index source publishes KV and reads Top-K, so its `kv_source` is `None` while
/// its `index_source` is `Some`. A row with ratio 0 has no compressed KV and no offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    /// The row index: `0..n_layers` for backbone layers and `n_layers..n_layers + n_mtp_layers`
    /// for the MTP stages.
    pub layer: usize,
    /// The attention variant the row runs.
    pub mode: LayerMode,
    /// The compression kind selected by [`Self::compress_ratio`].
    pub compression: CompressionKind,
    /// The row's entry in `compress_ratios`: 0, 1 or 2 in the released model, and 0 for a stage.
    pub compress_ratio: usize,
    /// Whether this row owns the compressed KV it attends to: the [`LayerMode::Full`] rows and no
    /// other row.
    ///
    /// Independent of [`Self::owns_index`] (model.py:654-655): a Reindex layer owns an indexer but no
    /// KV, and a Full layer may own KV without an indexer. The plan validates both flags against the
    /// configuration's lists, so a mode cannot smuggle ownership in or out.
    pub owns_kv: bool,
    /// Whether this row runs an indexer and publishes `topk_idxs`: the [`LayerMode::Reindex`] rows,
    /// and the [`LayerMode::Full`] rows the configuration also lists as index sources. A Full row
    /// without this flag reuses the preceding index source's Top-K. [`Self::publishes_topk`] is this
    /// flag.
    pub owns_index: bool,
    /// The layer whose compressor published the main compressed KV this row reads, or `None` when
    /// the row owns that KV (a [`LayerMode::Full`] source), has no compressed KV (ratio 0) or is an
    /// MTP stage.
    ///
    /// A reader's indexer reads this row's index-K cache, but the row publishes `index_k` only when
    /// it is both a KV source and an index source: a Full row can own KV without `owns_index` and
    /// publish none, and a Reindex row owns an indexer without KV and reads the cache
    /// (model.py:497-500, 517-525, 537-547).
    pub kv_source: Option<usize>,
    /// The decode-time offset added to compressed ids in the joint `[window ; compressed]` KV:
    /// `Some(window_size)` for a compressed row and `None` for a ratio-0 row. Prefill uses the
    /// chunk length instead; [`Self::kv_offset_at`] applies that rule.
    pub kv_offset: Option<usize>,
    /// The layer whose published `topk_idxs` this row reuses, or `None` when the row runs its own
    /// indexer and publishes (`owns_index`), has no compressed KV, or is an MTP stage. A Full row
    /// with `owns_index == false` names the preceding index source here, just like a Reuse row
    /// (model.py:725-726).
    pub index_source: Option<usize>,
    /// The HSI pool this row is involved with: `Some(20)` on layer 20, which builds it, and on the
    /// four Reindex layers, which mask their scores with it; `None` elsewhere, including the stages
    /// and a configuration with no index source (and therefore no pool).
    pub candidate_source: Option<usize>,
    /// Whether the row carries an Engram table. The released model marks backbone layers 1 and 14;
    /// an MTP stage carries none.
    pub engram: bool,
    /// Whether this row's input hidden state is a DSpark tap. The released model marks backbone
    /// layers 37, 38 and 39; a tap is still an ordinary backbone layer in the base forward, and the
    /// MTP stages are not taps.
    pub dspark_target: bool,
}

impl LayerSpec {
    /// Whether this row runs an indexer and publishes `topk_idxs`. This is [`Self::owns_index`]: a
    /// Full row may own KV without publishing (model.py:654-655, 725-726), and a Reindex row
    /// publishes without owning KV.
    #[must_use]
    pub fn publishes_topk(&self) -> bool {
        self.owns_index
    }

    /// The offset this row adds to compressed ids for a given phase.
    ///
    /// The reference passes the window cache's length (`kv.size(1)`) into `_compress_kv` at
    /// model.py:776, which shifts every reachable id by it (model.py:580); that length is the chunk
    /// length during prefill (`start_pos == 0`) and `window_size` during decode (model.py:708-719).
    /// A ratio-0 row has no compressed KV and therefore no offset.
    #[must_use]
    pub fn kv_offset_at(&self, start_pos: usize, seqlen: usize) -> Option<usize> {
        let decode_offset = self.kv_offset?;
        Some(if start_pos == 0 {
            seqlen
        } else {
            decode_offset
        })
    }
}

/// A validated layer plan for one configuration.
///
/// The plan holds one entry per row: `n_layers` backbone layers followed by `n_mtp_layers` MTP
/// stages. Its [`Self::order`] is the execution order the DAG allows (the ascending row order, which
/// the residual chain forces), and every KV and index edge points forward in that order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    layers: Vec<LayerSpec>,
    order: Vec<usize>,
    attention: AttentionGeometry,
    candidate_source: Option<usize>,
}

impl LayerPlan {
    /// Derive and validate the plan for a configuration: the backbone rows, then the MTP stages.
    ///
    /// # Errors
    ///
    /// Returns a [`PlanError`] when the configuration describes a row that cannot run: a stage or
    /// out-of-range id in one of the id lists, a compressed reader with no preceding source, a
    /// source that does not precede its reader, a reader whose ratio differs from its source's, a
    /// stage with a non-zero ratio, or a released-list entry that contradicts the row's ratio or
    /// role.
    pub fn from_config(config: &ModelConfig) -> Result<Self, PlanError> {
        let n_rows = config.n_layers + config.n_mtp_layers;
        let specs = (0..n_rows)
            .map(|layer| derive_spec(config, layer))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_specs(config, specs)
    }

    /// Validate a plan against the configuration it claims to describe.
    ///
    /// [`Self::from_config`] is the production constructor. This entry point is public because the
    /// validation is the point: a test can hand it a plan with a deliberately wrong edge and watch
    /// the plan refuse to exist, which is what makes the invariant a check rather than a comment.
    ///
    /// # Errors
    ///
    /// Returns a [`PlanError`] for a wrong row count, an unknown row, a non-ascending or duplicated
    /// row, a stage inside the backbone or a backbone row labelled as a stage, an id list naming a
    /// stage or an out-of-range layer, a field that disagrees with the configuration, or an edge
    /// whose source does not precede or does not match the reader.
    pub fn from_specs(config: &ModelConfig, specs: Vec<LayerSpec>) -> Result<Self, PlanError> {
        let n_rows = config.n_layers + config.n_mtp_layers;
        if specs.len() != n_rows {
            return Err(PlanError::LayerCount {
                expected: n_rows,
                found: specs.len(),
            });
        }
        let attention = AttentionGeometry::from_config(config);
        // A plan is only meaningful if every id it names is a backbone row. The configuration
        // constructor checks this too, but `from_config` and `from_specs` are public and must
        // not trust a caller to have run `validate`: a stage id in one of the lists would
        // otherwise reach the plan and be reported as a mismatched row far from its cause.
        validate_config_ids(config)?;
        // The HSI pool is built inside the indexer (`select_candidate_blocks`, model.py:569-571),
        // so the pool builder must be an index source. It need not own KV: `is_candidate_source`
        // does not consult `owns_k` (model.py:502-503), and a Reindex layer is a legal builder.
        // A configuration with no index source has no pool at all, so the plan records `None`
        // rather than refusing a window-only model.
        let candidate_source = if config.index_source_layers.is_empty() {
            None
        } else {
            let source = config.candidate_source_layer;
            if !config.index_source_layers.contains(&source) {
                return Err(PlanError::CandidateSourceNotIndexSource { from_layer: source });
            }
            Some(source)
        };

        for (position, spec) in specs.iter().enumerate() {
            let layer = spec.layer;
            if layer >= n_rows {
                return Err(PlanError::UnknownStage {
                    layer,
                    n_layers: config.n_layers,
                    n_mtp_layers: config.n_mtp_layers,
                });
            }
            if layer != position {
                return Err(PlanError::NotDenseOrAscending {
                    position,
                    layer,
                    rows: n_rows,
                });
            }
            let ratio = config
                .compress_ratio(layer)
                .ok_or(PlanError::MissingRatio { layer })?;
            if spec.compress_ratio != ratio {
                return Err(PlanError::RatioMismatch {
                    layer,
                    found: spec.compress_ratio,
                    expected: ratio,
                });
            }
            let expected_kind = CompressionKind::for_ratio(ratio);
            if spec.compression != expected_kind {
                return Err(PlanError::CompressionMismatch {
                    layer,
                    kind: spec.compression,
                    ratio,
                });
            }

            if layer >= config.n_layers {
                // An MTP stage is a DSpark block, not a backbone layer: window-only (ratio 0),
                // with no KV or index source, no Engram table and no DSpark tap.
                if spec.mode != LayerMode::Mtp {
                    return Err(PlanError::ModeMismatch {
                        layer,
                        found: spec.mode,
                        expected: LayerMode::Mtp,
                    });
                }
                if ratio != 0 {
                    return Err(PlanError::CompressedStage { layer, ratio });
                }
                if spec.owns_kv {
                    return Err(PlanError::OwnsKvMismatch {
                        layer,
                        found: true,
                        expected: false,
                    });
                }
                if spec.owns_index {
                    return Err(PlanError::OwnsIndexMismatch {
                        layer,
                        found: true,
                        expected: false,
                    });
                }
                // Checked here as well as after the row: even a caller that bypassed
                // `validate_config_ids` cannot label a stage with a backbone role.
                if spec.engram {
                    return Err(PlanError::EngramMismatch {
                        layer,
                        found: true,
                        expected: false,
                    });
                }
                if spec.dspark_target {
                    return Err(PlanError::DsparkTargetMismatch {
                        layer,
                        found: true,
                        expected: false,
                    });
                }
            } else {
                if spec.mode == LayerMode::Mtp {
                    return Err(PlanError::StageInBackbone {
                        layer,
                        n_layers: config.n_layers,
                    });
                }
                let owns_kv = config.kv_source_layers.contains(&layer);
                let owns_index = config.index_source_layers.contains(&layer);
                if ratio == 0 && (owns_kv || owns_index) {
                    return Err(PlanError::SourceWithoutCompression { layer });
                }
                if spec.owns_kv != owns_kv {
                    return Err(PlanError::OwnsKvMismatch {
                        layer,
                        found: spec.owns_kv,
                        expected: owns_kv,
                    });
                }
                if spec.owns_index != owns_index {
                    return Err(PlanError::OwnsIndexMismatch {
                        layer,
                        found: spec.owns_index,
                        expected: owns_index,
                    });
                }
                let expected_mode = if ratio == 0 {
                    LayerMode::SwaOnly
                } else if owns_kv {
                    LayerMode::Full
                } else if owns_index {
                    LayerMode::Reindex
                } else {
                    LayerMode::Reuse
                };
                if spec.mode != expected_mode {
                    return Err(PlanError::ModeMismatch {
                        layer,
                        found: spec.mode,
                        expected: expected_mode,
                    });
                }
            }

            // A reader must name a preceding KV source with its own ratio. Check the edge first, so
            // a deliberately wrong source is reported as the edge it is; then check that it is the
            // nearest preceding source, which is the one whose cache the reference publishes.
            if ratio > 0 && !spec.owns_kv {
                let Some(source) = spec.kv_source else {
                    return Err(PlanError::MissingKvSource { layer, ratio });
                };
                let expected_source = preceding_kv_source(config, layer)
                    .ok_or(PlanError::MissingKvSource { layer, ratio })?;
                if source >= layer {
                    return Err(PlanError::KvSourceNotBefore {
                        layer,
                        from_layer: source,
                    });
                }
                if !config.kv_source_layers.contains(&source) {
                    return Err(PlanError::KvSourceNotSource {
                        layer,
                        from_layer: source,
                    });
                }
                let source_ratio = config.compress_ratio(source).unwrap_or(0);
                if source_ratio != ratio {
                    return Err(PlanError::KvRatioMismatch {
                        layer,
                        from_layer: source,
                        ratio,
                        source_ratio,
                    });
                }
                if source != expected_source {
                    return Err(PlanError::KvSourceMismatch {
                        layer,
                        found: source,
                        expected: expected_source,
                    });
                }
            } else if let Some(source) = spec.kv_source {
                return Err(PlanError::KvSourceUnexpected {
                    layer,
                    from_layer: source,
                });
            }

            if ratio > 0 && !spec.owns_index {
                let Some(source) = spec.index_source else {
                    return Err(PlanError::MissingIndexSource { layer, ratio });
                };
                let expected_source = preceding_index_source(config, layer)
                    .ok_or(PlanError::MissingIndexSource { layer, ratio })?;
                if source >= layer {
                    return Err(PlanError::IndexSourceNotBefore {
                        layer,
                        from_layer: source,
                    });
                }
                if !config.index_source_layers.contains(&source) {
                    return Err(PlanError::IndexSourceNotSource {
                        layer,
                        from_layer: source,
                    });
                }
                let source_ratio = config.compress_ratio(source).unwrap_or(0);
                if source_ratio != ratio {
                    return Err(PlanError::IndexRatioMismatch {
                        layer,
                        from_layer: source,
                        ratio,
                        source_ratio,
                    });
                }
                if source != expected_source {
                    return Err(PlanError::IndexSourceMismatch {
                        layer,
                        found: source,
                        expected: expected_source,
                    });
                }
            } else if let Some(source) = spec.index_source {
                return Err(PlanError::IndexSourceUnexpected {
                    layer,
                    from_layer: source,
                });
            }

            let expected_offset = (ratio > 0).then_some(attention.window);
            if spec.kv_offset != expected_offset {
                return Err(PlanError::KvOffsetMismatch {
                    layer,
                    found: spec.kv_offset,
                    expected: expected_offset,
                });
            }
            let expected_candidate =
                candidate_source.filter(|&source| spec.owns_index && layer >= source);
            if spec.candidate_source != expected_candidate {
                return Err(PlanError::CandidateSourceMismatch {
                    layer,
                    found: spec.candidate_source,
                    expected: expected_candidate,
                });
            }
            let expected_engram = config.engram_layer_ids.contains(&layer);
            if spec.engram != expected_engram {
                return Err(PlanError::EngramMismatch {
                    layer,
                    found: spec.engram,
                    expected: expected_engram,
                });
            }
            let expected_target = config.dspark_target_layer_ids.contains(&layer);
            if spec.dspark_target != expected_target {
                return Err(PlanError::DsparkTargetMismatch {
                    layer,
                    found: spec.dspark_target,
                    expected: expected_target,
                });
            }
        }

        Ok(Self {
            layers: specs,
            order: (0..n_rows).collect(),
            attention,
            candidate_source,
        })
    }

    /// The per-row entries, in ascending row order: the backbone, then the MTP stages.
    #[must_use]
    pub fn layers(&self) -> &[LayerSpec] {
        &self.layers
    }

    /// One row's entry, or `None` when the index is outside the plan.
    #[must_use]
    pub fn get(&self, layer: usize) -> Option<&LayerSpec> {
        self.layers.get(layer)
    }

    /// The number of planned rows: exactly `n_layers + n_mtp_layers`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the plan holds no rows. A real configuration always has rows; present because a
    /// plan is a collection and the lint asks for it.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The execution order the DAG implies: the ascending row order. The residual stream's
    /// `i -> i + 1` edges force it, the three MTP stages run after the backbone, and every KV and
    /// index edge is consistent with it.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// The attention geometry shared by every layer.
    #[must_use]
    pub fn attention(&self) -> AttentionGeometry {
        self.attention
    }

    /// The layer that builds the HSI candidate pool, or `None` when the configuration has no
    /// index source and therefore no pool. A window-only configuration still carries a
    /// `candidate_source_layer` field, but no row can build the pool it names.
    #[must_use]
    pub fn candidate_source(&self) -> Option<usize> {
        self.candidate_source
    }

    /// The `(source, reader)` KV edges the plan implies, one per reader.
    pub fn kv_edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.layers
            .iter()
            .filter_map(|spec| spec.kv_source.map(|source| (source, spec.layer)))
    }

    /// The `(source, reader)` index edges the plan implies, one per reader.
    pub fn index_edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.layers
            .iter()
            .filter_map(|spec| spec.index_source.map(|source| (source, spec.layer)))
    }

    /// The DSpark target layers, in ascending order. These are taps in the base forward, not MTP
    /// stages.
    pub fn dspark_targets(&self) -> impl Iterator<Item = &LayerSpec> {
        self.layers.iter().filter(|spec| spec.dspark_target)
    }

    /// Whether every KV and index edge points forward in [`Self::order`].
    ///
    /// [`Self::from_specs`] enforces this at construction, so a plan returned by either constructor
    /// is always topologically ordered. The method exists so the property can be asserted and
    /// printed rather than merely promised.
    #[must_use]
    pub fn is_topologically_ordered(&self) -> bool {
        if self.order.len() != self.layers.len() {
            return false;
        }
        let mut position = vec![usize::MAX; self.layers.len()];
        for (index, &layer) in self.order.iter().enumerate() {
            if layer >= position.len() || position[layer] != usize::MAX {
                return false;
            }
            position[layer] = index;
        }
        if position.contains(&usize::MAX) {
            return false;
        }
        self.kv_edges()
            .chain(self.index_edges())
            .all(|(source, reader)| {
                source < position.len()
                    && reader < position.len()
                    && position[source] < position[reader]
            })
    }
}

/// An error in deriving or validating the layer plan.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// The plan does not hold exactly one entry per row: `n_layers + n_mtp_layers`.
    #[error("the layer plan holds {found} entries, but the model has {expected} rows")]
    LayerCount {
        /// The number of rows the model has.
        expected: usize,
        /// The number of entries supplied.
        found: usize,
    },
    /// A plan row names an index outside the backbone and the MTP stages.
    #[error(
        "layer {layer} is outside the plan: {n_layers} backbone layers and {n_mtp_layers} MTP stages"
    )]
    UnknownStage {
        /// The offending index.
        layer: usize,
        /// The backbone's layer count.
        n_layers: usize,
        /// The number of MTP stages.
        n_mtp_layers: usize,
    },
    /// A backbone row carries the MTP stage mode.
    #[error("layer {layer} is inside the backbone (0..{n_layers}) but is labelled as an MTP stage")]
    StageInBackbone {
        /// The offending layer.
        layer: usize,
        /// The backbone's layer count.
        n_layers: usize,
    },
    /// An MTP stage row has a non-zero compression ratio.
    #[error(
        "layer {layer} is an MTP stage with compression ratio {ratio}; a stage is window-only (ratio 0)"
    )]
    CompressedStage {
        /// The offending stage.
        layer: usize,
        /// The ratio the configuration gives it.
        ratio: usize,
    },
    /// The plan is not the dense ascending row sequence.
    #[error("plan entry {position} names layer {layer}; the plan is dense and ascending 0..{rows}")]
    NotDenseOrAscending {
        /// The entry's position in the plan.
        position: usize,
        /// The layer the entry names.
        layer: usize,
        /// The number of planned rows.
        rows: usize,
    },
    /// The configuration has no ratio for a planned row.
    #[error("the configuration has no compression ratio for layer {layer}")]
    MissingRatio {
        /// The layer with no ratio.
        layer: usize,
    },
    /// A layer's compression kind disagrees with its ratio.
    #[error("layer {layer} is labelled {kind:?} but its ratio is {ratio}")]
    CompressionMismatch {
        /// The layer.
        layer: usize,
        /// The label the plan carries.
        kind: CompressionKind,
        /// The ratio the configuration gives.
        ratio: usize,
    },
    /// A layer's ratio disagrees with the configuration.
    #[error("layer {layer} has ratio {found}, but the configuration gives it {expected}")]
    RatioMismatch {
        /// The layer.
        layer: usize,
        /// The ratio the plan carries.
        found: usize,
        /// The ratio the configuration gives.
        expected: usize,
    },
    /// A ratio-0 layer is named a KV or index source.
    #[error("layer {layer} has ratio 0 but is named as a KV or index source")]
    SourceWithoutCompression {
        /// The offending layer.
        layer: usize,
    },
    /// A row's `owns_kv` flag disagrees with the configuration.
    #[error("layer {layer} has owns_kv {found}, but the configuration gives {expected}")]
    OwnsKvMismatch {
        /// The layer.
        layer: usize,
        /// The flag the plan carries.
        found: bool,
        /// The flag the configuration selects.
        expected: bool,
    },
    /// A row's `owns_index` flag disagrees with the configuration.
    #[error("layer {layer} has owns_index {found}, but the configuration gives {expected}")]
    OwnsIndexMismatch {
        /// The layer.
        layer: usize,
        /// The flag the plan carries.
        found: bool,
        /// The flag the configuration selects.
        expected: bool,
    },
    /// A layer's mode disagrees with the configuration.
    #[error("layer {layer} has mode {found:?}, but the configuration selects {expected:?}")]
    ModeMismatch {
        /// The layer.
        layer: usize,
        /// The mode the plan carries.
        found: LayerMode,
        /// The mode the configuration selects.
        expected: LayerMode,
    },
    /// A compressed reader names a KV source that does not precede it.
    #[error("layer {layer} reads KV from layer {from_layer}, which does not precede it")]
    KvSourceNotBefore {
        /// The reader.
        layer: usize,
        /// The source it names.
        from_layer: usize,
    },
    /// A layer names a KV source that is not in the released list.
    #[error("layer {layer} reads KV from layer {from_layer}, which is not a KV source")]
    KvSourceNotSource {
        /// The reader.
        layer: usize,
        /// The alleged source.
        from_layer: usize,
    },
    /// A reader's ratio differs from its KV source's.
    #[error(
        "layer {layer} has ratio {ratio}, but its KV source {from_layer} has ratio {source_ratio}"
    )]
    KvRatioMismatch {
        /// The reader.
        layer: usize,
        /// Its source.
        from_layer: usize,
        /// The reader's ratio.
        ratio: usize,
        /// The source's ratio.
        source_ratio: usize,
    },
    /// A compressed reader that should name a KV source names none.
    #[error("layer {layer} has ratio {ratio} but no preceding KV source is available")]
    MissingKvSource {
        /// The reader.
        layer: usize,
        /// Its ratio.
        ratio: usize,
    },
    /// A reader names a preceding KV source that is not the nearest one.
    #[error(
        "layer {layer} reads KV from layer {found}, but the nearest preceding source is {expected}"
    )]
    KvSourceMismatch {
        /// The reader.
        layer: usize,
        /// The source it names.
        found: usize,
        /// The source the released list selects.
        expected: usize,
    },
    /// A layer that owns or has no compressed KV names a KV source anyway.
    #[error(
        "layer {layer} names KV source {from_layer}, but it owns its KV or has no compressed KV"
    )]
    KvSourceUnexpected {
        /// The layer.
        layer: usize,
        /// The source it names.
        from_layer: usize,
    },
    /// A reuse reader names an index source that does not precede it.
    #[error("layer {layer} reads indices from layer {from_layer}, which does not precede it")]
    IndexSourceNotBefore {
        /// The reader.
        layer: usize,
        /// The source it names.
        from_layer: usize,
    },
    /// A layer names an index source that is not in the released list.
    #[error("layer {layer} reads indices from layer {from_layer}, which is not an index source")]
    IndexSourceNotSource {
        /// The reader.
        layer: usize,
        /// The alleged source.
        from_layer: usize,
    },
    /// A reader's ratio differs from its index source's.
    #[error(
        "layer {layer} has ratio {ratio}, but its index source {from_layer} has ratio {source_ratio}"
    )]
    IndexRatioMismatch {
        /// The reader.
        layer: usize,
        /// Its source.
        from_layer: usize,
        /// The reader's ratio.
        ratio: usize,
        /// The source's ratio.
        source_ratio: usize,
    },
    /// A reuse layer that should name an index source names none.
    #[error("layer {layer} has ratio {ratio} but no preceding index source is available")]
    MissingIndexSource {
        /// The reader.
        layer: usize,
        /// Its ratio.
        ratio: usize,
    },
    /// A reader names a preceding index source that is not the nearest one.
    #[error(
        "layer {layer} reads indices from layer {found}, but the nearest preceding index source is {expected}"
    )]
    IndexSourceMismatch {
        /// The reader.
        layer: usize,
        /// The source it names.
        found: usize,
        /// The source the released list selects.
        expected: usize,
    },
    /// A layer that runs its own indexer, or has no compressed KV, names an index source anyway.
    #[error(
        "layer {layer} names index source {from_layer}, but it publishes its own Top-K or has no compressed KV"
    )]
    IndexSourceUnexpected {
        /// The layer.
        layer: usize,
        /// The source it names.
        from_layer: usize,
    },
    /// A layer's KV offset disagrees with the window rule.
    #[error("layer {layer} has KV offset {found:?}, but the configuration gives {expected:?}")]
    KvOffsetMismatch {
        /// The layer.
        layer: usize,
        /// The offset the plan carries.
        found: Option<usize>,
        /// The offset the window rule gives.
        expected: Option<usize>,
    },
    /// A layer's candidate-pool source disagrees with the released ids.
    #[error(
        "layer {layer} has candidate source {found:?}, but the released ids select {expected:?}"
    )]
    CandidateSourceMismatch {
        /// The layer.
        layer: usize,
        /// The source the plan carries.
        found: Option<usize>,
        /// The source the released ids select.
        expected: Option<usize>,
    },
    /// A stage or out-of-range id appears in one of the configuration's id lists.
    #[error("the {field} entry {layer} is not a backbone layer of {n_layers}")]
    LayerOutsideBackbone {
        /// The configuration field that names it.
        field: &'static str,
        /// The offending id.
        layer: usize,
        /// The backbone's layer count.
        n_layers: usize,
    },
    /// The HSI pool builder is not an index source.
    #[error("the HSI candidate source {from_layer} is not an index source")]
    CandidateSourceNotIndexSource {
        /// The configured pool builder.
        from_layer: usize,
    },
    /// A layer's Engram flag disagrees with the released ids.
    #[error("layer {layer} has engram {found}, but the released ids say {expected}")]
    EngramMismatch {
        /// The layer.
        layer: usize,
        /// The flag the plan carries.
        found: bool,
        /// The flag the released ids select.
        expected: bool,
    },
    /// A layer's DSpark-target flag disagrees with the released ids.
    #[error("layer {layer} has dspark_target {found}, but the released ids say {expected}")]
    DsparkTargetMismatch {
        /// The layer.
        layer: usize,
        /// The flag the plan carries.
        found: bool,
        /// The flag the released ids select.
        expected: bool,
    },
}

/// Derive one row's entry from the configuration without validating it.
fn derive_spec(config: &ModelConfig, layer: usize) -> Result<LayerSpec, PlanError> {
    let ratio = config
        .compress_ratio(layer)
        .ok_or(PlanError::MissingRatio { layer })?;
    if config.is_mtp(layer) {
        // An MTP stage is a DSpark block: window-only (ratio 0), with no KV or index source, no
        // Engram table and no DSpark tap. `from_specs` rejects a stage whose ratio is not 0.
        return Ok(LayerSpec {
            layer,
            mode: LayerMode::Mtp,
            compression: CompressionKind::for_ratio(ratio),
            compress_ratio: ratio,
            owns_kv: false,
            owns_index: false,
            kv_source: None,
            kv_offset: None,
            index_source: None,
            candidate_source: None,
            engram: false,
            dspark_target: false,
        });
    }
    let owns_kv = config.kv_source_layers.contains(&layer);
    let owns_index = config.index_source_layers.contains(&layer);
    let mode = if ratio == 0 {
        LayerMode::SwaOnly
    } else if owns_kv {
        LayerMode::Full
    } else if owns_index {
        LayerMode::Reindex
    } else {
        LayerMode::Reuse
    };
    let kv_source = if ratio == 0 || owns_kv {
        None
    } else {
        preceding_kv_source(config, layer)
    };
    let index_source = if ratio == 0 || owns_index {
        None
    } else {
        preceding_index_source(config, layer)
    };
    let candidate_source = (!config.index_source_layers.is_empty())
        .then_some(config.candidate_source_layer)
        .filter(|&source| owns_index && layer >= source);
    Ok(LayerSpec {
        layer,
        mode,
        compression: CompressionKind::for_ratio(ratio),
        compress_ratio: ratio,
        owns_kv,
        owns_index,
        kv_source,
        kv_offset: (ratio > 0).then_some(config.window_size),
        index_source,
        candidate_source,
        engram: config.engram_layer_ids.contains(&layer),
        dspark_target: config.dspark_target_layer_ids.contains(&layer),
    })
}

/// Refuse a stage or out-of-range id in any configuration list the plan reads.
///
/// [`ModelConfig::validate`] enforces the same rule, but the plan's constructors are public and
/// must not depend on a caller having run it. A stage id that reached the plan would be compared
/// against a row it does not describe, so the cause would be reported far from the list.
fn validate_config_ids(config: &ModelConfig) -> Result<(), PlanError> {
    let n_layers = config.n_layers;
    for (field, ids) in [
        ("kv_source_layers", &config.kv_source_layers),
        ("index_source_layers", &config.index_source_layers),
        ("engram_layer_ids", &config.engram_layer_ids),
        ("dspark_target_layer_ids", &config.dspark_target_layer_ids),
    ] {
        if let Some(&layer) = ids.iter().find(|&&layer| layer >= n_layers) {
            return Err(PlanError::LayerOutsideBackbone {
                field,
                layer,
                n_layers,
            });
        }
    }
    if config.candidate_source_layer >= n_layers {
        return Err(PlanError::LayerOutsideBackbone {
            field: "candidate_source_layer",
            layer: config.candidate_source_layer,
            n_layers,
        });
    }
    Ok(())
}

/// The nearest KV source strictly before `layer`.
fn preceding_kv_source(config: &ModelConfig, layer: usize) -> Option<usize> {
    config
        .kv_source_layers
        .iter()
        .copied()
        .filter(|&source| source < layer)
        .max()
}

/// The nearest index source strictly before `layer`.
fn preceding_index_source(config: &ModelConfig, layer: usize) -> Option<usize> {
    config
        .index_source_layers
        .iter()
        .copied()
        .filter(|&source| source < layer)
        .max()
}
