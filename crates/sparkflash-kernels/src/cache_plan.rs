//! The device-resident cache plan: per-layer kinds from the model config, not one stride times a
//! count. The mapping back to attention_device.rs is in the residency doc's second section.
//!
//! THE KIND RULE (released text_config): 40 layers; compress_ratios = [0, 0, 2 x 18, 1 x 20, 0 x 3
//! padding]; kv_source_layer_ids [2, 8, 14, 20] own the compressor; index_source_layer_ids
//! [2, 8, 14, 20, 24, 28, 32, 36] have index weights. Every layer does window attention, so every
//! layer owns a window codes/scales ring at capacity max_positions (2,162,688 B at 4,096) plus its
//! selected list and sink. Only a compressor owner owns the pre-RoPE latent ring, the FP4 expanded
//! store and the append cursor. The index operands are per-step workspace (query, block scores,
//! block best, weights); the index KEY is 4 MiB only on a key owner in BlockMask mode and its
//! consumers alias the preceding owner's span, or one declared word in Direct mode when ref_select
//! produces the ids.
//!
//! THE STRIDE IS NOT UNIFORM. Layer offsets are the running sum of the kinds' byte counts, so a
//! 40-layer plan allocates 40 window rings. A single stride multiplied by a count allocated four
//! layers' worth and would have refused - or aliased - from layer five on.

/// The window the seq entry reads: 128 rows, contiguous from a base offset.
pub const WINDOW_ROWS: usize = 128;
/// The expanded FP4 store rows (body view bound).
pub const EXPANDED_ROWS: usize = 512;
/// The compressed-position bound: a plan past it is refused, never grown.
pub const MAX_POSITIONS_BOUND: usize = 4096;
/// One window row: 512 E4M3 code bytes and 16 E8M0 scale bytes.
pub const WINDOW_ROW_BYTES: usize = 512 + 16;
/// One latent ring row: head_dim 512 BF16 - the normed, roped latent the append consumes
/// (compressed_append_sm121 takes bf16 kv_rotated rows; the host fuses norm on f32 then one bf16
/// rounding, tp4_forward.rs:3032-3038). 1,024 B, not the 2,048 B an f32 pre-RoPE ring would cost.
pub const LATENT_ROW_BYTES: usize = 512 * 2;
/// One FP4 store row: 256 code bytes and 32 scale bytes.
pub const STORE_ROW_BYTES: usize = 256 + 32;
/// The store, codes and scales together (EXPANDED_ROWS rows).
pub const STORE_BYTES: usize = EXPANDED_ROWS * STORE_ROW_BYTES;
/// One selected-list word.
pub const SELECTED_WORD_BYTES: usize = 4;
/// The append cursor word.
pub const CURSOR_BYTES: usize = 4;
/// The per-head attention sink, f32, [64].
pub const SINK_BYTES: usize = 64 * 4;
/// The q operand: 64 heads of 512 bf16.
pub const Q_BYTES: usize = 64 * 512 * 2;
/// The pre-inverse-RoPE attention output row.
pub const OUT_PRE_ROPE_BYTES: usize = 64 * 512 * 2;
/// The post-wo_b attention output row.
pub const OUT_POST_WO_B_BYTES: usize = 64 * 512 * 2;
/// The HSI index query operand: 32 indexer heads (mla_ref::INDEX_HEADS; the kernel
/// static_asserts SPARK_M4_INDEX_HEADS == 32), bf16 [32][128] = 8,192 B. attention_device.rs:57
/// allocates 64 heads (16,384 B), which is stale: the pinned kernel and the host producer both
/// use 32.
pub const HSI_INDEX_QUERY_BYTES: usize = 32 * 128 * 2;
/// The HSI block mask scores operand, f32, [2048].
pub const HSI_BLOCK_SCORES_BYTES: usize = 2_048 * 4;
/// The HSI block mask best-token operand, u32, [2048].
pub const HSI_BLOCK_BEST_BYTES: usize = 2_048 * 4;
/// The HSI index key operand, bf16, [2048][8][128], owned by a kv-source layer.
pub const HSI_INDEX_KEY_BYTES: usize = 2_048 * 8 * 128 * 2;
/// The fused norm-on-f32-then-one-bf16-rounding kernel the host uses (tp4_forward.rs:3032-3038).
pub const NORM_KERNEL: &str = "kv_norm_rope_bf16";

/// Whether that kernel is pinned yet: None means the latent ring write must be refused or the
/// deviation recorded, never silently fed the unnormalised latent.
pub const NORM_PIN: Option<&str> = None;

/// Where the latent ring's norm and rope comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormSource {
    /// The fused device kernel is pinned and produces the row the append reads.
    DeviceFused,
    /// Only the host reference path normalises: a device ring write is refused, not approximated.
    HostReferenceOnly,
}

/// Refuses a latent ring write that has no pinned norm source.
///
/// # Errors
/// PlanError::NormUnpinned for NormSource::HostReferenceOnly.
pub fn latent_ring_write_ready(source: NormSource) -> Result<(), PlanError> {
    match source {
        NormSource::DeviceFused => Ok(()),
        NormSource::HostReferenceOnly => Err(PlanError::NormUnpinned),
    }
}

/// The append window at a boundary position: at p with (p + 1) % ratio == 0 the append consumes the
/// ratio latents whose first row is first_row = ((p + 1) / ratio - 1) * ratio, contiguous in slot
/// order (slot = row % max_positions, capacity max_positions so a run never wraps). None at a
/// non-boundary position.
#[must_use]
pub fn append_window(position: usize, ratio: usize) -> Option<(usize, usize)> {
    if ratio == 0 || !(position + 1).is_multiple_of(ratio) {
        return None;
    }
    Some((((position + 1) / ratio - 1) * ratio, ratio))
}

/// One declared index operand word in HsiMode::Direct.
pub const DIRECT_WORD_BYTES: usize = 4;
/// The index weights row: 32 heads f32 per position.
pub const INDEX_WEIGHTS_ROW_BYTES: usize = 32 * 4;

/// Whether the arm runs the HSI block mask or takes the ids from ref_select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HsiMode {
    /// ref_select produces the ids: each HSI operand is one declared word, never a null pointer.
    ///
    /// In milestone A the HSI block arrays are DECLARED ZERO, not populated: the seq entry's indexer
    /// stage strides by SPARK_M4_HEADS = 64 while a per-rank launch covers 16 heads, so the block
    /// pool is not complete at world 4, and index_key/index_query have no device producer among the
    /// 27 pins. Nothing consumes them in the format-2 selected path, where the candidate list comes
    /// from ref_select. A truthful indexer is a later slice with its own pin; until it lands, a
    /// reader must not assume the indexer ran.
    Direct,
    /// The body computes the block mask: a key owner allocates the full operand and its consumers
    /// alias the owner's span.
    BlockMask,
}

/// The channels one layer owns, derived from the config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kind {
    /// The layer compress ratio (0 when it has none).
    pub ratio: usize,
    /// Whether the layer owns a compressor cache (a kv source).
    pub owns_compressor: bool,
    /// Whether the layer has index weights (an index source).
    pub index_source: bool,
    /// Whether the layer owns the HSI index key operand in block-mask mode.
    pub key_owner: bool,
}

impl Kind {
    /// Bytes of this layer's resident spans.
    #[must_use]
    pub fn layer_bytes(&self, max_positions: usize, hsi: HsiMode) -> usize {
        let window = max_positions * WINDOW_ROW_BYTES;
        let compressor = if self.owns_compressor {
            max_positions * LATENT_ROW_BYTES + STORE_BYTES + CURSOR_BYTES
        } else {
            0
        };
        let key = if hsi == HsiMode::BlockMask && self.key_owner {
            HSI_INDEX_KEY_BYTES
        } else {
            0
        };
        window + compressor + EXPANDED_ROWS * SELECTED_WORD_BYTES + SINK_BYTES + key
    }
}

/// The plan geometry and the per-layer kinds input, taken from the model config.
#[derive(Clone, Copy, Debug)]
pub struct PlanConfig<'a> {
    /// The layer count (num_hidden_layers).
    pub layers: usize,
    /// compress_ratios, indexed by layer. Must reach layers; padding beyond is ignored.
    pub ratios: &'a [usize],
    /// kv_source_layer_ids: the compressor owners.
    pub kv_sources: &'a [usize],
    /// index_source_layer_ids: the index-weight layers.
    pub index_sources: &'a [usize],
    /// The compressed-position bound.
    pub max_positions: usize,
    /// The HSI operand mode.
    pub hsi: HsiMode,
}

/// A span inside the slab: an offset from the slab base and a byte length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Byte offset from the slab base.
    pub offset: usize,
    /// Byte length.
    pub bytes: usize,
}

impl Span {
    /// The half-open byte range.
    #[must_use]
    pub fn range(&self) -> core::ops::Range<usize> {
        self.offset..self.offset + self.bytes
    }
}

/// The resident spans of one layer, absolute from the slab base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSpans {
    /// The layer kind the spans were built for.
    pub kind: Kind,
    /// Bytes this layer occupies.
    pub bytes: usize,
    /// Window E4M3 codes ring.
    pub window_codes: Span,
    /// Window E8M0 scales ring.
    pub window_scales: Span,
    /// The selected expanded-id list.
    pub selected: Span,
    /// The per-head sink.
    pub sink: Span,
    /// HSI index key: the owner's full span, a consumer's alias, or one declared word.
    pub index_key: Span,
    /// Pre-RoPE latent ring: zero-length unless the layer owns the compressor.
    pub latent_ring: Span,
    /// FP4 expanded store codes: zero-length unless the layer owns the compressor.
    pub store_codes: Span,
    /// FP4 expanded store scales: zero-length unless the layer owns the compressor.
    pub store_scales: Span,
    /// The append cursor word: zero-length unless the layer owns the compressor.
    pub cursor: Span,
}

/// The per-step working set, one copy for the whole arm: the body runs one layer at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceSpans {
    /// The q operand.
    pub q: Span,
    /// The pre-inverse-RoPE attention output (never aliases the post-wo_b row).
    pub out_pre_rope: Span,
    /// The post-wo_b attention output.
    pub out_post_wo_b: Span,
    /// The HSI index query operand.
    pub index_query: Span,
    /// The HSI block scores operand.
    pub block_scores: Span,
    /// The HSI block best-token operand.
    pub block_best: Span,
    /// The index weights rows (index-source layers share the one copy: sequential steps).
    pub index_weights: Span,
    /// The Direct-mode index key word.
    pub index_key_word: Span,
}

/// The channel totals across all layers, the numbers the doc table prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelTotals {
    /// Window channels over every layer.
    pub window_bytes: usize,
    /// Pre-RoPE latent rings over the compressor owners.
    pub latent_bytes: usize,
    /// FP4 store codes and scales over the compressor owners.
    pub store_bytes: usize,
    /// Append cursors over the compressor owners.
    pub cursor_bytes: usize,
    /// Selected lists over every layer.
    pub selected_bytes: usize,
    /// Sinks over every layer.
    pub sink_bytes: usize,
    /// HSI index keys over the key owners (BlockMask only).
    pub index_key_bytes: usize,
    /// The layer kinds, in layer order.
    pub kinds: Vec<Kind>,
}

/// Why a plan was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The config parse does not expose a ratio for every layer: refuse, do not assume.
    RatiosShort {
        /// Ratios supplied.
        ratios: usize,
        /// Layers the plan is for.
        layers: usize,
    },
    /// A layer named as a kv source has no compression ratio.
    SourceWithoutRatio {
        /// The layer.
        layer: usize,
    },
    /// The ratio does not fit the expanded store.
    RatioPastStore {
        /// The layer.
        layer: usize,
        /// Its ratio.
        ratio: usize,
        /// The bound.
        bound: usize,
    },
    /// A kv or index source names a layer past the plan.
    SourcePastLayers {
        /// The layer.
        layer: usize,
        /// Layers the plan holds.
        layers: usize,
    },
    /// A zero positions bound has no capacity to state.
    ZeroPositions,
    /// The positions bound is past MAX_POSITIONS_BOUND.
    PositionsPastBound {
        /// The request.
        max_positions: usize,
        /// The bound.
        bound: usize,
    },
    /// A plan with no layers has no stride.
    ZeroLayers,
    /// A layer index past the plan.
    LayerPastPlan {
        /// The layer asked for.
        layer: usize,
        /// Layers the plan holds.
        layers: usize,
    },
    /// The latent ring holds fewer rows than one append consumes.
    PositionsShortOfRatio {
        /// The positions bound.
        max_positions: usize,
        /// The ratio it has to cover.
        ratio: usize,
    },
    /// The latent ring has no pinned norm source.
    NormUnpinned,
    /// A consumer layer has no preceding key owner to alias in block-mask mode.
    NoIndexSource {
        /// The layer.
        layer: usize,
    },
    /// The byte arithmetic left usize.
    Overflow {
        /// The size that overflowed.
        bytes: usize,
    },
}

impl core::fmt::Display for PlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RatiosShort { ratios, layers } => write!(
                f,
                "compress_ratios reaches {ratios} layers, the plan is for {layers}: the config parse does not expose a per-layer ratio",
            ),
            Self::SourceWithoutRatio { layer } => {
                write!(
                    f,
                    "layer {layer} is named as a kv source but has no compression ratio"
                )
            }
            Self::RatioPastStore {
                layer,
                ratio,
                bound,
            } => write!(
                f,
                "layer {layer} compresses at ratio {ratio}, past the {bound}-row expanded store",
            ),
            Self::SourcePastLayers { layer, layers } => {
                write!(
                    f,
                    "source layer {layer} is past the {layers} layers the plan holds"
                )
            }
            Self::ZeroPositions => write!(f, "max_positions is 0: no capacity to state"),
            Self::PositionsPastBound {
                max_positions,
                bound,
            } => {
                write!(f, "max_positions {max_positions} is past the {bound} bound")
            }
            Self::ZeroLayers => write!(f, "a plan with no layers has no stride"),
            Self::LayerPastPlan { layer, layers } => {
                write!(
                    f,
                    "layer {layer} is past the {layers} layers the plan holds"
                )
            }
            Self::PositionsShortOfRatio {
                max_positions,
                ratio,
            } => write!(
                f,
                "max_positions {max_positions} is shorter than the {ratio} latents one append consumes",
            ),
            Self::NormUnpinned => write!(
                f,
                "the latent ring has no pinned norm source: {NORM_KERNEL} is not pinned and the host path alone cannot feed a device ring write",
            ),
            Self::NoIndexSource { layer } => {
                write!(
                    f,
                    "layer {layer} has no preceding key owner to alias in block-mask mode"
                )
            }
            Self::Overflow { bytes } => write!(f, "the plan byte arithmetic overflowed at {bytes}"),
        }
    }
}

/// The cache plan: per-layer offsets from the config, the workspace, and the totals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachePlan {
    layers: usize,
    max_positions: usize,
    hsi: HsiMode,
    kinds: Vec<Kind>,
    offsets: Vec<usize>,
    layers_bytes: usize,
    workspace_bytes: usize,
    total_bytes: usize,
}

impl CachePlan {
    /// Plans the arm's slabs from the model config.
    ///
    /// # Errors
    /// PlanError, by name: RatiosShort when the config does not expose a ratio for every layer
    /// (never assumed 4 or 40), SourceWithoutRatio, RatioPastStore, SourcePastLayers,
    /// ZeroPositions, PositionsPastBound, ZeroLayers, Overflow.
    pub fn new(config: &PlanConfig<'_>) -> Result<Self, PlanError> {
        if config.max_positions == 0 {
            return Err(PlanError::ZeroPositions);
        }
        if config.max_positions > MAX_POSITIONS_BOUND {
            return Err(PlanError::PositionsPastBound {
                max_positions: config.max_positions,
                bound: MAX_POSITIONS_BOUND,
            });
        }
        if config.layers == 0 {
            return Err(PlanError::ZeroLayers);
        }
        if config.ratios.len() < config.layers {
            return Err(PlanError::RatiosShort {
                ratios: config.ratios.len(),
                layers: config.layers,
            });
        }
        for source in config.kv_sources.iter().chain(config.index_sources) {
            if *source >= config.layers {
                return Err(PlanError::SourcePastLayers {
                    layer: *source,
                    layers: config.layers,
                });
            }
        }
        let mut kinds = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let ratio = config.ratios[layer];
            let owns_compressor = config.kv_sources.contains(&layer);
            if owns_compressor && ratio == 0 {
                return Err(PlanError::SourceWithoutRatio { layer });
            }
            if ratio > EXPANDED_ROWS {
                return Err(PlanError::RatioPastStore {
                    layer,
                    ratio,
                    bound: EXPANDED_ROWS,
                });
            }
            if owns_compressor && ratio > config.max_positions {
                return Err(PlanError::PositionsShortOfRatio {
                    max_positions: config.max_positions,
                    ratio,
                });
            }
            kinds.push(Kind {
                ratio,
                owns_compressor,
                index_source: config.index_sources.contains(&layer),
                key_owner: owns_compressor,
            });
        }
        let mut offsets = Vec::with_capacity(config.layers + 1);
        let mut layers_bytes = 0usize;
        for kind in &kinds {
            offsets.push(layers_bytes);
            layers_bytes = layers_bytes
                .checked_add(kind.layer_bytes(config.max_positions, config.hsi))
                .ok_or(PlanError::Overflow {
                    bytes: layers_bytes,
                })?;
        }
        offsets.push(layers_bytes);
        let workspace_bytes = workspace_bytes(config.max_positions)?;
        let total_bytes = layers_bytes
            .checked_add(workspace_bytes)
            .ok_or(PlanError::Overflow {
                bytes: layers_bytes,
            })?;
        Ok(Self {
            layers: config.layers,
            max_positions: config.max_positions,
            hsi: config.hsi,
            kinds,
            offsets,
            layers_bytes,
            workspace_bytes,
            total_bytes,
        })
    }

    /// Layers the plan covers.
    #[must_use]
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// The HSI mode the plan allocated for.
    #[must_use]
    pub fn hsi(&self) -> HsiMode {
        self.hsi
    }

    /// The kind of one layer.
    ///
    /// # Errors
    /// PlanError::LayerPastPlan.
    pub fn kind(&self, layer: usize) -> Result<Kind, PlanError> {
        self.kinds
            .get(layer)
            .copied()
            .ok_or(PlanError::LayerPastPlan {
                layer,
                layers: self.layers,
            })
    }

    /// The byte offset of one layer's spans from the slab base.
    ///
    /// # Errors
    /// PlanError::LayerPastPlan.
    pub fn layer_base(&self, layer: usize) -> Result<usize, PlanError> {
        self.kind(layer)?;
        Ok(self.offsets[layer])
    }

    /// Bytes one layer occupies.
    ///
    /// # Errors
    /// PlanError::LayerPastPlan.
    pub fn layer_bytes(&self, layer: usize) -> Result<usize, PlanError> {
        Ok(self.kind(layer)?.layer_bytes(self.max_positions, self.hsi))
    }

    /// The spans of one layer, absolute from the slab base.
    ///
    /// # Errors
    /// PlanError::LayerPastPlan, PlanError::NoIndexSource.
    pub fn layer(&self, layer: usize) -> Result<LayerSpans, PlanError> {
        let kind = self.kind(layer)?;
        let mut offset = self.offsets[layer];
        let mut take = |length: usize| {
            let span = Span {
                offset,
                bytes: length,
            };
            offset += length;
            span
        };
        let window_codes = take(self.max_positions * 512);
        let window_scales = take(self.max_positions * 16);
        let selected = take(EXPANDED_ROWS * SELECTED_WORD_BYTES);
        let sink = take(SINK_BYTES);
        let key_length = if self.hsi == HsiMode::BlockMask && kind.key_owner {
            HSI_INDEX_KEY_BYTES
        } else {
            0
        };
        let index_key = take(key_length);
        let (latent_ring, store_codes, store_scales, cursor) = if kind.owns_compressor {
            (
                take(self.max_positions * LATENT_ROW_BYTES),
                take(EXPANDED_ROWS * 256),
                take(EXPANDED_ROWS * 32),
                take(CURSOR_BYTES),
            )
        } else {
            (
                Span { offset, bytes: 0 },
                Span { offset, bytes: 0 },
                Span { offset, bytes: 0 },
                Span { offset, bytes: 0 },
            )
        };
        debug_assert_eq!(
            offset,
            self.offsets[layer] + self.layer_bytes(layer)?,
            "spans must tile the layer"
        );
        Ok(LayerSpans {
            kind,
            bytes: self.layer_bytes(layer)?,
            window_codes,
            window_scales,
            selected,
            sink,
            index_key,
            latent_ring,
            store_codes,
            store_scales,
            cursor,
        })
    }

    /// The HSI index key span a layer reads in block-mask mode: the owner's own span, the preceding
    /// owner's span for a consumer (an explicit alias, never a fresh 4 MiB copy), or the declared
    /// Direct-mode word.
    ///
    /// # Errors
    /// PlanError::LayerPastPlan, PlanError::NoIndexSource.
    pub fn index_key_span(&self, layer: usize) -> Result<Span, PlanError> {
        let kind = self.kind(layer)?;
        if self.hsi == HsiMode::Direct {
            return Ok(Span {
                offset: self.total_bytes - DIRECT_WORD_BYTES,
                bytes: DIRECT_WORD_BYTES,
            });
        }
        if kind.key_owner {
            return Ok(self.layer(layer)?.index_key);
        }
        let owner = (0..layer)
            .rev()
            .find(|candidate| self.kinds[*candidate].key_owner);
        match owner {
            Some(owner) => Ok(self.layer(owner)?.index_key),
            None => Err(PlanError::NoIndexSource { layer }),
        }
    }

    /// Bytes charged to the per-layer slab.
    #[must_use]
    pub fn layers_bytes(&self) -> usize {
        self.layers_bytes
    }

    /// Bytes charged to the one-copy workspace.
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Bytes the whole arm charges.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// The one-copy workspace, placed after the per-layer slab.
    ///
    /// # Errors
    /// PlanError::Overflow never in practice; kept so the placement is checked arithmetic.
    pub fn workspace(&self) -> Result<WorkspaceSpans, PlanError> {
        let mut offset = self.layers_bytes;
        let mut take = |length: usize| {
            let span = Span {
                offset,
                bytes: length,
            };
            offset += length;
            span
        };
        let q = take(Q_BYTES);
        let out_pre_rope = take(OUT_PRE_ROPE_BYTES);
        let out_post_wo_b = take(OUT_POST_WO_B_BYTES);
        let index_query = take(HSI_INDEX_QUERY_BYTES);
        let block_scores = take(HSI_BLOCK_SCORES_BYTES);
        let block_best = take(HSI_BLOCK_BEST_BYTES);
        let index_weights = take(self.max_positions * INDEX_WEIGHTS_ROW_BYTES);
        let index_key_word = take(DIRECT_WORD_BYTES);
        if offset != self.total_bytes {
            return Err(PlanError::Overflow { bytes: offset });
        }
        Ok(WorkspaceSpans {
            q,
            out_pre_rope,
            out_post_wo_b,
            index_query,
            block_scores,
            block_best,
            index_weights,
            index_key_word,
        })
    }

    /// The channel totals across every layer.
    #[must_use]
    pub fn totals(&self) -> ChannelTotals {
        let window_bytes = self.layers * self.max_positions * WINDOW_ROW_BYTES;
        let owners = self
            .kinds
            .iter()
            .filter(|kind| kind.owns_compressor)
            .count();
        let key_bytes = if self.hsi == HsiMode::BlockMask {
            owners * HSI_INDEX_KEY_BYTES
        } else {
            0
        };
        ChannelTotals {
            window_bytes,
            latent_bytes: owners * self.max_positions * LATENT_ROW_BYTES,
            store_bytes: owners * STORE_BYTES,
            cursor_bytes: owners * CURSOR_BYTES,
            selected_bytes: self.layers * EXPANDED_ROWS * SELECTED_WORD_BYTES,
            sink_bytes: self.layers * SINK_BYTES,
            index_key_bytes: key_bytes,
            kinds: self.kinds.clone(),
        }
    }
}

fn workspace_bytes(max_positions: usize) -> Result<usize, PlanError> {
    Q_BYTES
        .checked_add(OUT_PRE_ROPE_BYTES)
        .and_then(|bytes| bytes.checked_add(OUT_POST_WO_B_BYTES))
        .and_then(|bytes| bytes.checked_add(HSI_INDEX_QUERY_BYTES))
        .and_then(|bytes| bytes.checked_add(HSI_BLOCK_SCORES_BYTES))
        .and_then(|bytes| bytes.checked_add(HSI_BLOCK_BEST_BYTES))
        .and_then(|bytes| bytes.checked_add(max_positions.checked_mul(INDEX_WEIGHTS_ROW_BYTES)?))
        .and_then(|bytes| bytes.checked_add(DIRECT_WORD_BYTES))
        .ok_or(PlanError::Overflow {
            bytes: max_positions,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The released text_config, read from the checkpoint file in the tree rather than restated.
    const RATIOS: [usize; 40] = [
        0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    const KV: [usize; 4] = [2, 8, 14, 20];
    const INDEX: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

    fn plan(hsi: HsiMode) -> CachePlan {
        CachePlan::new(&PlanConfig {
            layers: 40,
            ratios: &RATIOS,
            kv_sources: &KV,
            index_sources: &INDEX,
            max_positions: MAX_POSITIONS_BOUND,
            hsi,
        })
        .expect("the released table plans")
    }

    #[test]
    fn every_layer_has_its_window_ring_and_only_owners_own_the_compressor() {
        let plan = plan(HsiMode::Direct);
        let mut window_only = 0;
        let mut owners = 0;
        for layer in 0..40 {
            let kind = plan.kind(layer).expect("kind");
            let spans = plan.layer(layer).expect("layer");
            assert_eq!(
                spans.window_codes.bytes,
                4096 * 512,
                "layer {layer} owns a window ring"
            );
            assert_eq!(
                spans.window_codes.offset,
                plan.layer_base(layer).expect("base")
            );
            if kind.owns_compressor {
                owners += 1;
                assert_eq!(spans.latent_ring.bytes, 4096 * 1_024);
                assert_eq!(spans.store_codes.bytes, 512 * 256);
                assert_eq!(spans.store_scales.bytes, 512 * 32);
                assert_eq!(spans.cursor.bytes, 4);
            } else {
                window_only += 1;
                assert_eq!(
                    spans.latent_ring.bytes, 0,
                    "layer {layer} must not own the compressor"
                );
                assert_eq!(spans.store_codes.bytes, 0);
            }
            assert_eq!(spans.selected.bytes, 512 * 4);
            assert_eq!(spans.sink.bytes, 256);
        }
        assert_eq!(owners, 4);
        assert_eq!(window_only, 36);
        let totals = plan.totals();
        assert_eq!(totals.kinds.len(), 40);
        assert_eq!(totals.window_bytes, 40 * 4096 * 528);
        assert_eq!(totals.latent_bytes, 4 * 4096 * 1_024);
    }

    #[test]
    fn layer_offsets_are_the_running_sum_of_the_kinds_not_one_stride_times_a_count() {
        let plan = plan(HsiMode::Direct);
        let mut offset = 0usize;
        for layer in 0..40 {
            assert_eq!(
                plan.layer_base(layer).expect("base"),
                offset,
                "layer {layer} base"
            );
            let bytes = plan.layer_bytes(layer).expect("bytes");
            let spans = plan.layer(layer).expect("layer");
            let last = if spans.latent_ring.bytes > 0 {
                spans.cursor
            } else {
                spans.sink
            };
            assert_eq!(
                last.offset + last.bytes,
                offset + bytes,
                "layer {layer} tiles exactly"
            );
            offset += bytes;
        }
        assert_eq!(offset, plan.layers_bytes());
        assert!(plan.layer_base(40).is_err());
        // 36 window-only layers and 4 owners, not 40 of one shape.
        assert_eq!(plan.layer_bytes(0).expect("bytes"), 2_164_992);
        assert_eq!(plan.layer_bytes(2).expect("bytes"), 6_506_756);
        assert_eq!(plan.layers_bytes(), 36 * 2_164_992 + 4 * 6_506_756);
        assert_eq!(plan.workspace_bytes(), 196_608 + 24_576 + 4_096 * 128 + 4);
        assert_eq!(
            plan.total_bytes(),
            plan.layers_bytes() + plan.workspace_bytes()
        );
    }

    #[test]
    fn block_mask_keys_are_owned_once_and_aliased_by_consumers() {
        let direct = plan(HsiMode::Direct);
        let blocked = plan(HsiMode::BlockMask);
        assert_eq!(
            blocked.layer_bytes(2).expect("bytes"),
            6_506_756 + HSI_INDEX_KEY_BYTES
        );
        assert_eq!(
            blocked.layer_bytes(3).expect("bytes"),
            2_164_992,
            "a consumer allocates no key"
        );
        let owner = blocked.index_key_span(2).expect("owner");
        assert_eq!(owner.bytes, HSI_INDEX_KEY_BYTES);
        assert_eq!(
            owner.offset,
            blocked.layer(2).expect("layer").index_key.offset
        );
        assert_eq!(
            blocked.index_key_span(3).expect("consumer"),
            owner,
            "alias, not a copy"
        );
        assert_eq!(blocked.index_key_span(7).expect("consumer"), owner);
        assert_eq!(
            blocked.index_key_span(20).expect("next owner").bytes,
            HSI_INDEX_KEY_BYTES
        );
        assert_ne!(blocked.index_key_span(20).expect("next owner"), owner);
        assert_eq!(
            blocked.index_key_span(0).expect_err("no preceding owner"),
            PlanError::NoIndexSource { layer: 0 }
        );
        let word = direct.index_key_span(2).expect("direct word");
        assert_eq!(word.bytes, 4);
        assert_eq!(
            blocked.total_bytes(),
            direct.total_bytes() + 4 * HSI_INDEX_KEY_BYTES
        );
    }

    #[test]
    fn the_two_attention_rows_are_distinct() {
        let plan = plan(HsiMode::Direct);
        let workspace = plan.workspace().expect("workspace");
        assert_ne!(
            workspace.out_pre_rope.offset,
            workspace.out_post_wo_b.offset
        );
        assert_eq!(
            workspace.out_pre_rope.range().end,
            workspace.out_post_wo_b.range().start
        );
        assert_eq!(
            workspace.index_query.bytes, 8_192,
            "32 indexer heads, not 64"
        );
        assert_eq!(workspace.index_key_word.range().end, plan.total_bytes());
    }

    #[test]
    fn the_index_heads_come_from_the_pinned_kernel_and_the_host_producer() {
        // mla_ref::INDEX_HEADS mirrors the kernel's SPARK_M4_INDEX_HEADS, static_asserted at 32.
        assert_eq!(crate::mla_ref::INDEX_HEADS, 32);
        assert_eq!(
            HSI_INDEX_QUERY_BYTES,
            crate::mla_ref::INDEX_HEADS * crate::mla_ref::INDEX_HEAD_DIM * 2
        );
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/flash_mla_csa2_ring_sm121.cu"
        ))
        .expect("the body kernel source is in the tree");
        assert!(source.contains("static_assert(SPARK_M4_INDEX_HEADS == 32u"));
        let host = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sparkflash-cli/src/tp4_forward.rs"
        ))
        .expect("the cli is in the tree");
        assert!(host.contains("mla_ref::INDEX_HEADS * mla_ref::INDEX_HEAD_DIM * 2"));
    }

    #[test]
    fn the_config_refusals_name_the_layer_rather_than_assuming_a_count() {
        let short = CachePlan::new(&PlanConfig {
            layers: 40,
            ratios: &RATIOS[..39],
            kv_sources: &KV,
            index_sources: &INDEX,
            max_positions: 1024,
            hsi: HsiMode::Direct,
        })
        .expect_err("ratios short");
        assert_eq!(
            short,
            PlanError::RatiosShort {
                ratios: 39,
                layers: 40
            }
        );
        assert!(
            short
                .to_string()
                .contains("does not expose a per-layer ratio")
        );
        let mut bad_ratios = RATIOS;
        bad_ratios[5] = 0;
        let no_ratio = CachePlan::new(&PlanConfig {
            layers: 40,
            ratios: &bad_ratios,
            kv_sources: &[5],
            index_sources: &[],
            max_positions: 1024,
            hsi: HsiMode::Direct,
        })
        .expect_err("source without ratio");
        assert_eq!(no_ratio, PlanError::SourceWithoutRatio { layer: 5 });
        let past_store = CachePlan::new(&PlanConfig {
            layers: 40,
            ratios: &{
                let mut ratios = RATIOS;
                ratios[3] = 513;
                ratios
            },
            kv_sources: &KV,
            index_sources: &INDEX,
            max_positions: 1024,
            hsi: HsiMode::Direct,
        })
        .expect_err("ratio past store");
        assert_eq!(
            past_store,
            PlanError::RatioPastStore {
                layer: 3,
                ratio: 513,
                bound: 512
            }
        );
        let past = CachePlan::new(&PlanConfig {
            layers: 40,
            ratios: &RATIOS,
            kv_sources: &[40],
            index_sources: &[],
            max_positions: 1024,
            hsi: HsiMode::Direct,
        })
        .expect_err("source past layers");
        assert_eq!(
            past,
            PlanError::SourcePastLayers {
                layer: 40,
                layers: 40
            }
        );
        assert_eq!(
            CachePlan::new(&PlanConfig {
                layers: 40,
                ratios: &RATIOS,
                kv_sources: &KV,
                index_sources: &INDEX,
                max_positions: 0,
                hsi: HsiMode::Direct,
            })
            .expect_err("zero"),
            PlanError::ZeroPositions
        );
        assert_eq!(
            CachePlan::new(&PlanConfig {
                layers: 40,
                ratios: &RATIOS,
                kv_sources: &KV,
                index_sources: &INDEX,
                max_positions: 4097,
                hsi: HsiMode::Direct,
            })
            .expect_err("past bound"),
            PlanError::PositionsPastBound {
                max_positions: 4097,
                bound: 4096
            }
        );
    }

    #[test]
    fn the_append_window_is_ratio_contiguous_latents_from_the_first_rows_slot() {
        assert_eq!(
            append_window(7, 4),
            Some((4, 4)),
            "p = 7 completes rows 4..=7"
        );
        assert_eq!(append_window(8, 4), None, "not a boundary");
        assert_eq!(append_window(11, 4), Some((8, 4)));
        assert_eq!(append_window(3, 4), Some((0, 4)));
        assert_eq!(
            append_window(0, 1),
            Some((0, 1)),
            "ratio 1 completes every position"
        );
        assert_eq!(append_window(5, 0), None, "ratio 0 has no append");
        assert_eq!(
            4 * LATENT_ROW_BYTES,
            4_096,
            "one append at ratio 4 reads 4,096 B"
        );
        let (first, rows) = append_window(3, 4).expect("boundary");
        assert_eq!(
            first + rows,
            4,
            "four contiguous rows from slot 0 never wrap"
        );
    }

    #[test]
    fn a_ring_shorter_than_one_append_is_refused() {
        let error = CachePlan::new(&PlanConfig {
            layers: 8,
            ratios: &[2, 2, 2, 2, 2, 2, 2, 2],
            kv_sources: &[2],
            index_sources: &[2],
            max_positions: 1,
            hsi: HsiMode::Direct,
        })
        .expect_err("ring shorter than ratio");
        assert_eq!(
            error,
            PlanError::PositionsShortOfRatio {
                max_positions: 1,
                ratio: 2
            }
        );
    }

    #[test]
    fn the_latent_norm_is_declared_not_assumed() {
        assert_eq!(latent_ring_write_ready(NormSource::DeviceFused), Ok(()));
        let refused = latent_ring_write_ready(NormSource::HostReferenceOnly).expect_err("no pin");
        assert_eq!(refused, PlanError::NormUnpinned);
        assert!(refused.to_string().contains(NORM_KERNEL));
        assert_eq!(NORM_PIN, None, "the fused norm kernel is not pinned yet");
    }

    #[test]
    fn the_selected_id_space_is_guaranteed_by_ref_select_not_rechecked_on_the_host() {
        let window_rows = 128usize;
        let ratio = 4usize;
        let compressed_positions = EXPANDED_ROWS / ratio;
        for position in 0..compressed_positions {
            let id = position + window_rows;
            assert!(id >= window_rows);
            assert!((id - window_rows) * ratio < EXPANDED_ROWS);
        }
        assert_eq!(
            (window_rows + compressed_positions - 1 - window_rows) * ratio,
            EXPANDED_ROWS - ratio
        );
    }
}
