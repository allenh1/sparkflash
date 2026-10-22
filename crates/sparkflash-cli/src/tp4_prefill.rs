//! The multi-position forward's reference host seams and the CSA-union kernel operand staging.
//!
//! The resident plan declares the compressor, the indexer and the two-level candidate select as
//! host seams, and owns the attention projections on device. This module is the seam side of the
//! multi-position loop: it runs the reference's compressed-side block for one (layer, position)
//! through sparkflash_engine::forward::attention::attention_seam_step on the device chain's own
//! per-layer input, then converts the seam's rows plus the chain's per-position window rows into
//! the exact bytes the flash_mla_csa2_seq_sm121 launch reads:
//!
//!   * the window: the engine format-2 E4M3FN rows, oldest first, as the chain's kv path already
//!     encodes them (window_row_bytes of the rotated, narrowed kv row);
//!   * the compressed store: the per-position expanded view the kernel expects, where expanded row
//!     q holds store row q / ratio. The seam gives the selected ids in the shared index space
//!     (window_rows + j); the kernel's selected value is the expanded row j * ratio and the bytes
//!     there are the seam's already-quantised row re-encoded by the pinned main-KV codec;
//!   * the selected list itself, ascending, which is the order the reference's concatenated
//!     attended list uses and therefore the order a twin must sum in.
//!
//! Nothing here is device-specific: it compiles on a host without CUDA and the tests build the
//! bytes for synthetic rows. The loop that owns the device buffers consumes the output.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::{
    AttentionProjectionBackend, AttentionSeamStep, AttnState, attention_seam_step_with_backend,
    reset_shared_attention,
};
use sparkflash_engine::forward::attention_case::NO_INDEX;
use sparkflash_engine::safetensors::SafeTensorsSource;
use sparkflash_engine::tensor::TensorSpec;
use sparkflash_engine::weights::{WeightError, WeightSource};
use sparkflash_kv::fp4::quantize_main_kv;

/// The MLA latent width: one window row, one store row, one head's output.
pub const MLA_LATENT: usize = 512;
/// One engine window row's E4M3FN codes: one byte per latent channel.
pub const WINDOW_CODE_BYTES: usize = MLA_LATENT;
/// One engine window row's E8M0 scales: one byte per 32-channel block.
pub const WINDOW_SCALE_BYTES: usize = MLA_LATENT / 32;
/// One compressed store row's packed FP4 codes: two codes per byte.
pub const STORE_CODE_BYTES: usize = 256;
/// One compressed store row's E4M3 group scales: one byte per 16-channel block.
pub const STORE_SCALE_BYTES: usize = 32;
/// The expanded compressed view's capacity in rows. Position p needs expanded row at most p + 1
/// (ratio 1 stores every position; ratio 2 stores the even positions up to p), so the released
/// 130-position prefill plus nine teacher-forced decode steps stays under 512.
pub const EXPANDED_ROWS: usize = 512;
/// The largest selected list the kernel's ABI admits.
pub const MAX_SELECTED: usize = 512;

/// A weight source over the checkpoint that caches what it reads.
///
/// The seam step reads the same compressor and indexer tensors on every position, so a source that
/// re-read them per call would page the same few megabytes from the page cache 139 times per
/// layer. The cache is keyed by name and guarded, because WeightSource is Send + Sync and the
/// engine may read concurrently.
pub struct CachedCheckpointSource {
    inner: SafeTensorsSource,
    cache: Mutex<HashMap<String, Arc<Vec<u8>>>>,
}

impl CachedCheckpointSource {
    /// Opens the checkpoint directory's index file.
    ///
    /// # Errors
    /// [WeightError::Io] or [WeightError::Malformed] from the index load.
    pub fn open(
        dir: impl Into<std::path::PathBuf>,
        index_path: &Path,
    ) -> Result<Self, WeightError> {
        Ok(Self {
            inner: SafeTensorsSource::open(dir, index_path)?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// The inner source's per-tensor limit.
    #[must_use]
    pub fn limit(&self) -> u128 {
        self.inner.limit()
    }
}

impl WeightSource for CachedCheckpointSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        if let Some(held) = self.cache.lock().expect("seam weight cache").get(name) {
            return Ok(held.as_ref().clone());
        }
        let bytes = self.inner.read(name)?;
        let held = Arc::new(bytes);
        self.cache
            .lock()
            .expect("seam weight cache")
            .insert(name.to_owned(), Arc::clone(&held));
        Ok(held.as_ref().clone())
    }
}

/// The reference's compressed-side seams over one sequence, one (layer, position) call at a time.
///
/// The loop calls [ReferenceAttention::step] in layer order as it walks each position, exactly the
/// order the reference forward runs: that is what lets the first index/kv source of a step publish
/// the channels its consumers read in the same step.
pub struct ReferenceAttention {
    config: ModelConfig,
    source: CachedCheckpointSource,
    states: Vec<AttnState>,
}

impl ReferenceAttention {
    /// Opens the checkpoint and cold-starts one attention state per layer.
    ///
    /// # Errors
    /// The index load error, or a configuration whose layer count and compressor ratios disagree.
    pub fn open(
        config: &ModelConfig,
        checkpoint_dir: impl Into<std::path::PathBuf>,
        index_path: &Path,
    ) -> Result<Self, String> {
        if config.compress_ratios.len() < config.n_layers {
            return Err(format!(
                "reference attention: {} compress ratios for {} layers",
                config.compress_ratios.len(),
                config.n_layers
            ));
        }
        let source = CachedCheckpointSource::open(checkpoint_dir, index_path)
            .map_err(|error| format!("reference attention: checkpoint: {error}"))?;
        let states = (0..config.n_layers)
            .map(|_| AttnState::for_config(config))
            .collect();
        // The compressor/indexer/candidate channels are process-global; a loop that runs one
        // sequence must start them empty so a previous run's cache cannot be read.
        reset_shared_attention();
        Ok(Self {
            config: config.clone(),
            source,
            states,
        })
    }

    /// The model configuration the states were built for.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The checkpoint source the seam reads its tensors from, so a device arm can stage the
    /// weights it will need once, before the position loop starts, from the same bytes the host
    /// rule reads.
    #[must_use]
    pub fn source(&self) -> &dyn WeightSource {
        &self.source
    }

    /// One step's compressed-side seams for layer at position.
    ///
    /// x is the layer's normalised attention input in f32, q_latent the BF16-narrowed q_norm output
    /// the query path produced, both the values the device chain's own stages produced.
    ///
    /// # Errors
    /// The seam step's shape/channel errors, as a string.
    pub fn step(
        &mut self,
        layer: usize,
        position: usize,
        x: &[f32],
        q_latent: &[f32],
    ) -> Result<AttentionSeamStep, String> {
        self.step_with_backend(
            layer,
            position,
            x,
            q_latent,
            &mut sparkflash_engine::forward::attention::HostAttentionProjectionBackend,
        )
    }

    /// [ReferenceAttention::step] with an attention projection backend.
    ///
    /// The backend may run the indexer's own query projection on a device; every other seam
    /// statement (compressor, index keys, per-query weights, candidate select, fp4 cache) is this
    /// module's host code in both arms, and a backend that answers `None` leaves the projection to
    /// the engine's host native rule.
    ///
    /// # Errors
    /// The seam step's shape/channel errors, or the backend's, as a string.
    pub fn step_with_backend(
        &mut self,
        layer: usize,
        position: usize,
        x: &[f32],
        q_latent: &[f32],
        backend: &mut dyn AttentionProjectionBackend,
    ) -> Result<AttentionSeamStep, String> {
        if self.states.get(layer).is_none() {
            return Err(format!(
                "reference attention: layer {layer} of {}",
                self.states.len()
            ));
        }
        attention_seam_step_with_backend(
            &self.config,
            layer,
            &format!("layers.{layer}"),
            x,
            q_latent,
            position,
            &mut self.states[layer],
            &self.source,
            usize::MAX,
            backend,
        )
        .map_err(|error| format!("reference attention: layer {layer} position {position}: {error}"))
    }

    /// Reads every reference-seam tensor once, before the loop's first position.
    ///
    /// The seam step reads the compressor and indexer tensors from the checkpoint on every
    /// (layer, position). [CachedCheckpointSource] makes every read after the first a memory hit,
    /// but the first read of each tensor is a cold page-cache miss: the arena loader reads
    /// O_DIRECT (arena.rs: "CPU file reads go straight into the final GPU-visible arena"), which
    /// bypasses the page cache, so the old buffered loader's side effect of leaving these shards
    /// resident is gone. This pass is that old side effect, made explicit: it reads exactly the
    /// compressor and indexer tensors - the two families the bind census marks host_seam - with
    /// buffered reads, so the loop's first access to each is a cache hit and the cold burst moves
    /// out of position 0, where it stalls a chain no other rank can see.
    ///
    /// It reads nothing the seam does not read, touches no value, and changes no arithmetic: the
    /// loop's reads return the same bytes whether they come from the disk or the cache. Returns the
    /// number of tensors warmed, for the caller's setup accounting.
    ///
    /// # Errors
    ///
    /// The source's read error for a tensor the index advertises.
    pub fn warm_seams(&mut self) -> Result<usize, String> {
        let mut warmed = 0usize;
        for name in self.source.names() {
            if !(name.contains(".attn.compressor.") || name.contains(".attn.indexer.")) {
                continue;
            }
            self.source
                .read(&name)
                .map_err(|error| format!("reference attention: warm-up {name}: {error}"))?;
            warmed += 1;
        }
        Ok(warmed)
    }
}

/// The kernel operands one CSA-union attention launch reads, built from the seam and the chain's
/// per-position window rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedAttention {
    /// The window rows staged, oldest first.
    pub window_rows: usize,
    /// The window codes: window_rows rows of [WINDOW_CODE_BYTES].
    pub window_codes: Vec<u8>,
    /// The window scales: window_rows rows of [WINDOW_SCALE_BYTES].
    pub window_scales: Vec<u8>,
    /// The compressed ids staged, ascending, as the kernel's selected values (expanded rows).
    pub selected: Vec<u32>,
    /// The expanded compressed view's packed FP4 codes, [EXPANDED_ROWS] rows of
    /// [STORE_CODE_BYTES]. Rows no id selects stay zero.
    pub kv_codes: Vec<u8>,
    /// The expanded compressed view's group scales, [EXPANDED_ROWS] rows of [STORE_SCALE_BYTES].
    pub kv_scales: Vec<u8>,
    /// The expanded rows the selected list actually wrote.
    pub store_rows: usize,
    /// The layer's compression ratio the rows were expanded with.
    pub ratio: usize,
}

/// Stages one launch's operands from the chain's window bytes and the seam's compressed rows.
///
/// window holds the chain's engine-format window rows for the positions this query attends, oldest
/// first, as (codes, scales) pairs; it must be exactly seam.window_width long (the evicted rows are
/// dropped before the call, because the kernel reads no id for a row outside the window).
///
/// # Errors
/// A row width or count disagreement, an id below the window offset, a non-finite stored value, or
/// a selected expanded row past [EXPANDED_ROWS].
pub fn stage_attention(
    window: &[(Vec<u8>, Vec<u8>)],
    seam: &AttentionSeamStep,
) -> Result<StagedAttention, String> {
    if window.len() != seam.window_width {
        return Err(format!(
            "attention staging: {} window rows for an attended width of {}",
            window.len(),
            seam.window_width
        ));
    }
    if seam.ratio == 0 {
        if !seam.compressed_ids.is_empty() {
            return Err(format!(
                "attention staging: a window-only layer selected {} compressed ids",
                seam.compressed_ids.len()
            ));
        }
        if !seam.compressed_rows.is_empty() {
            return Err(format!(
                "attention staging: a window-only layer stored {} compressed values",
                seam.compressed_rows.len()
            ));
        }
    }
    let mut window_codes = Vec::with_capacity(window.len() * WINDOW_CODE_BYTES);
    let mut window_scales = Vec::with_capacity(window.len() * WINDOW_SCALE_BYTES);
    for (index, (codes, scales)) in window.iter().enumerate() {
        if codes.len() != WINDOW_CODE_BYTES || scales.len() != WINDOW_SCALE_BYTES {
            return Err(format!(
                "attention staging: window row {index} is {} code bytes and {} scale bytes, the engine row is {WINDOW_CODE_BYTES} and {WINDOW_SCALE_BYTES}",
                codes.len(),
                scales.len()
            ));
        }
        window_codes.extend_from_slice(codes);
        window_scales.extend_from_slice(scales);
    }
    let mut kv_codes = vec![0u8; EXPANDED_ROWS * STORE_CODE_BYTES];
    let mut kv_scales = vec![0u8; EXPANDED_ROWS * STORE_SCALE_BYTES];
    let mut selected: Vec<u32> = Vec::with_capacity(seam.compressed_ids.len());
    let mut store_rows = 0usize;
    for id in seam.compressed_ids.iter().copied() {
        if id == NO_INDEX {
            continue;
        }
        let row = id.checked_sub(seam.window_rows).ok_or_else(|| {
            format!(
                "attention staging: compressed id {id} is below the window offset {}",
                seam.window_rows
            )
        })?;
        let expanded = row
            .checked_mul(seam.ratio)
            .ok_or_else(|| format!("attention staging: compressed row {row} overflowed"))?;
        if expanded >= EXPANDED_ROWS {
            return Err(format!(
                "attention staging: compressed id {id} expands to row {expanded}, past the {EXPANDED_ROWS}-row view"
            ));
        }
        let values = seam
            .compressed_rows
            .get(row * MLA_LATENT..(row + 1) * MLA_LATENT)
            .ok_or_else(|| {
                format!(
                    "attention staging: compressed id {id} needs row {row} of {} stored",
                    seam.compressed_rows.len() / MLA_LATENT
                )
            })?;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "attention staging: compressed row {row} is not finite"
            ));
        }
        let entry = quantize_main_kv(values)
            .map_err(|error| format!("attention staging: compressed row {row} codec: {error}"))?;
        let start = expanded * STORE_CODE_BYTES;
        kv_codes[start..start + STORE_CODE_BYTES].copy_from_slice(&entry.data);
        let start = expanded * STORE_SCALE_BYTES;
        kv_scales[start..start + STORE_SCALE_BYTES].copy_from_slice(&entry.scales);
        selected.push(
            u32::try_from(expanded)
                .map_err(|_| format!("attention staging: row {expanded} does not fit a u32"))?,
        );
        store_rows += 1;
    }
    if selected.len() > MAX_SELECTED {
        return Err(format!(
            "attention staging: {} selected ids, the kernel admits {MAX_SELECTED}",
            selected.len()
        ));
    }
    Ok(StagedAttention {
        window_rows: window.len(),
        window_codes,
        window_scales,
        selected,
        kv_codes,
        kv_scales,
        store_rows,
        ratio: seam.ratio,
    })
}

/// A quantised compressed row as the kernel's store holds it, plus the exact f32 row it was
/// produced from.
///
/// The source row is kept as raw bits, so a cache hit is a proof of equality rather than a
/// fingerprint's probability: [AttentionStaging::stage] compares the bits it was handed with these
/// before reusing the entry, and a mismatch quantises the row again.
#[derive(Debug)]
struct CachedRow {
    source: Box<[u32]>,
    data: [u8; STORE_CODE_BYTES],
    scales: [u8; STORE_SCALE_BYTES],
}

/// How many quantised rows the memo keeps before it is dropped. The memo is an optimisation only:
/// dropping it re-quantises, and never changes a staged byte.
const MAX_CACHED_ROWS: usize = 16_384;

/// Reusable operand state for the loop's per-(layer, position) attention launch.
///
/// [stage_attention] builds every operand from scratch on each call: it allocates and zero-fills
/// the 147,456-byte expanded store, allocates the two window buffers, and re-runs
/// [quantize_main_kv] over every selected compressed row. The forward loop calls it once per layer
/// per position, so all of that is per-call host work.
///
/// This slot keeps the four buffers and the window/selected vectors between calls and memoises the
/// quantised compressed rows under (layer, compressed row):
///
/// * the buffers are reused, and exactly the rows the previous call wrote are cleared, so the
///   store's "unselected rows stay zero" invariant is re-established by writing the same zeros the
///   allocation used to provide;
/// * a selected row's quantised bytes depend only on its stored f32 row, and the engine's
///   compressed cache is append-only per layer (write_compressed_cache stores a completed block at
///   first_row = start_pos / ratio and never rewrites an earlier block), so an entry stays valid.
///   It is checked, not trusted: the bits handed in must equal the bits the entry was produced from
///   or the row is quantised again, so a changed row can never hand back stale bytes.
///
/// The output is exactly [StagedAttention]'s byte set. The caller hands the staged value back with
/// [Self::recycle] after the launch so the next call reuses the same buffers.
#[derive(Debug, Default)]
pub struct AttentionStaging {
    window_codes: Vec<u8>,
    window_scales: Vec<u8>,
    kv_codes: Vec<u8>,
    kv_scales: Vec<u8>,
    selected: Vec<u32>,
    /// The expanded rows the previous call wrote, cleared before the next call places its rows.
    written: Vec<usize>,
    /// Quantised compressed rows, keyed by (layer, compressed row).
    cache: HashMap<(u32, u32), CachedRow>,
}

impl AttentionStaging {
    /// An empty slot: the first call allocates, every later call reuses.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stages one launch's operands into the reusable buffers.
    ///
    /// The checks are [stage_attention]'s, with the same refusals; the staged bytes are the same
    /// bytes. The returned value owns the buffers until [Self::recycle] takes them back.
    ///
    /// # Errors
    ///
    /// A row width or count disagreement, an id below the window offset, a non-finite stored value,
    /// a selected expanded row past [EXPANDED_ROWS], or a selected count past [MAX_SELECTED].
    pub fn stage(
        &mut self,
        layer: usize,
        window: &[(Vec<u8>, Vec<u8>)],
        seam: &AttentionSeamStep,
    ) -> Result<StagedAttention, String> {
        if window.len() != seam.window_width {
            return Err(format!(
                "attention staging: {} window rows for an attended width of {}",
                window.len(),
                seam.window_width
            ));
        }
        if seam.ratio == 0 {
            if !seam.compressed_ids.is_empty() {
                return Err(format!(
                    "attention staging: a window-only layer selected {} compressed ids",
                    seam.compressed_ids.len()
                ));
            }
            if !seam.compressed_rows.is_empty() {
                return Err(format!(
                    "attention staging: a window-only layer stored {} compressed values",
                    seam.compressed_rows.len()
                ));
            }
        }

        let mut window_codes = std::mem::take(&mut self.window_codes);
        let mut window_scales = std::mem::take(&mut self.window_scales);
        window_codes.clear();
        window_scales.clear();
        window_codes.reserve(window.len() * WINDOW_CODE_BYTES);
        window_scales.reserve(window.len() * WINDOW_SCALE_BYTES);
        for (index, (codes, scales)) in window.iter().enumerate() {
            if codes.len() != WINDOW_CODE_BYTES || scales.len() != WINDOW_SCALE_BYTES {
                return Err(format!(
                    "attention staging: window row {index} is {} code bytes and {} scale bytes, the engine row is {WINDOW_CODE_BYTES} and {WINDOW_SCALE_BYTES}",
                    codes.len(),
                    scales.len()
                ));
            }
            window_codes.extend_from_slice(codes);
            window_scales.extend_from_slice(scales);
        }

        let mut kv_codes = std::mem::take(&mut self.kv_codes);
        let mut kv_scales = std::mem::take(&mut self.kv_scales);
        // The full-view size is established first, so a buffer lost to an earlier refusal is
        // re-created zeroed before any previously written row is cleared.
        kv_codes.resize(EXPANDED_ROWS * STORE_CODE_BYTES, 0);
        kv_scales.resize(EXPANDED_ROWS * STORE_SCALE_BYTES, 0);
        for row in std::mem::take(&mut self.written) {
            let start = row * STORE_CODE_BYTES;
            kv_codes[start..start + STORE_CODE_BYTES].fill(0);
            let start = row * STORE_SCALE_BYTES;
            kv_scales[start..start + STORE_SCALE_BYTES].fill(0);
        }

        let mut selected = std::mem::take(&mut self.selected);
        selected.clear();
        selected.reserve(seam.compressed_ids.len());
        let mut store_rows = 0usize;
        for id in seam.compressed_ids.iter().copied() {
            if id == NO_INDEX {
                continue;
            }
            let row = id.checked_sub(seam.window_rows).ok_or_else(|| {
                format!(
                    "attention staging: compressed id {id} is below the window offset {}",
                    seam.window_rows
                )
            })?;
            let expanded = row
                .checked_mul(seam.ratio)
                .ok_or_else(|| format!("attention staging: compressed row {row} overflowed"))?;
            if expanded >= EXPANDED_ROWS {
                return Err(format!(
                    "attention staging: compressed id {id} expands to row {expanded}, past the {EXPANDED_ROWS}-row view"
                ));
            }
            let values = seam
                .compressed_rows
                .get(row * MLA_LATENT..(row + 1) * MLA_LATENT)
                .ok_or_else(|| {
                    format!(
                        "attention staging: compressed id {id} needs row {row} of {} stored",
                        seam.compressed_rows.len() / MLA_LATENT
                    )
                })?;
            if !values.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "attention staging: compressed row {row} is not finite"
                ));
            }
            let layer_key = u32::try_from(layer)
                .map_err(|_| format!("attention staging: layer {layer} does not fit a u32"))?;
            let row_key = u32::try_from(row).map_err(|_| {
                format!("attention staging: compressed row {row} does not fit a u32")
            })?;
            let key = (layer_key, row_key);
            let cached = match self.cache.get(&key) {
                Some(entry)
                    if entry.source.len() == values.len()
                        && entry
                            .source
                            .iter()
                            .zip(values.iter())
                            .all(|(bits, value)| *bits == value.to_bits()) =>
                {
                    Some((entry.data, entry.scales))
                }
                _ => None,
            };
            let (data, scales) = match cached {
                Some(entry) => entry,
                None => {
                    let quantised = quantize_main_kv(values).map_err(|error| {
                        format!("attention staging: compressed row {row} codec: {error}")
                    })?;
                    let entry = CachedRow {
                        source: values.iter().map(|value| value.to_bits()).collect(),
                        data: quantised.data,
                        scales: quantised.scales,
                    };
                    if self.cache.len() >= MAX_CACHED_ROWS {
                        self.cache.clear();
                    }
                    let placed = (entry.data, entry.scales);
                    self.cache.insert(key, entry);
                    placed
                }
            };
            let start = expanded * STORE_CODE_BYTES;
            kv_codes[start..start + STORE_CODE_BYTES].copy_from_slice(&data);
            let start = expanded * STORE_SCALE_BYTES;
            kv_scales[start..start + STORE_SCALE_BYTES].copy_from_slice(&scales);
            selected
                .push(u32::try_from(expanded).map_err(|_| {
                    format!("attention staging: row {expanded} does not fit a u32")
                })?);
            self.written.push(expanded);
            store_rows += 1;
        }
        if selected.len() > MAX_SELECTED {
            return Err(format!(
                "attention staging: {} selected ids, the kernel admits {MAX_SELECTED}",
                selected.len()
            ));
        }
        Ok(StagedAttention {
            window_rows: window.len(),
            window_codes,
            window_scales,
            selected,
            kv_codes,
            kv_scales,
            store_rows,
            ratio: seam.ratio,
        })
    }

    /// Takes the staged value's buffers back for the next call.
    pub fn recycle(&mut self, staged: StagedAttention) {
        let StagedAttention {
            window_codes,
            window_scales,
            selected,
            kv_codes,
            kv_scales,
            ..
        } = staged;
        self.window_codes = window_codes;
        self.window_scales = window_scales;
        self.selected = selected;
        self.kv_codes = kv_codes;
        self.kv_scales = kv_scales;
    }
}

/// One launch's twin statistics: the host reference over the staged bytes against the device out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwinStats {
    /// The query heads compared.
    pub heads: usize,
    /// The window rows staged.
    pub window_rows: usize,
    /// The selected ids staged.
    pub selected: usize,
    /// The BF16 slots that disagree.
    pub mismatches: usize,
    /// The slots compared (heads x head_dim).
    pub slots: usize,
    /// The slots that agree exactly, as a fraction.
    pub exact_fraction: f64,
    /// The greatest absolute difference.
    pub max_abs: f32,
}

/// Runs the reference union attention over the staged bytes and compares it with the device out.
///
/// q is the staged BF16 query as f32 (the values the kernel reads), sink one f32 per head, and
/// device_out the kernel's BF16 out decoded to f32, both in head-major order.
///
/// # Errors
/// A geometry disagreement with the staged bytes.
pub fn twin_out(
    q: &[f32],
    sink: &[f32],
    staged: &StagedAttention,
    position: usize,
    device_out: &[f32],
) -> Result<TwinStats, String> {
    let heads = sink.len();
    let width = heads * MLA_LATENT;
    if q.len() != width || device_out.len() != width {
        return Err(format!(
            "attention twin: q {} and out {} values for {heads} heads of {MLA_LATENT}",
            q.len(),
            device_out.len()
        ));
    }
    if q.iter().chain(device_out).any(|value| !value.is_finite()) {
        return Err("attention twin: the query or the device out is not finite".to_owned());
    }
    let mut mismatches = 0usize;
    let mut max_abs = 0f32;
    for head in 0..heads {
        let head_out = sparkflash_kernels::mla_ref::attention_with_sink_union(
            &q[head * MLA_LATENT..(head + 1) * MLA_LATENT],
            &staged.window_codes,
            &staged.window_scales,
            staged.window_rows,
            &staged.kv_codes,
            &staged.kv_scales,
            &staged.selected,
            u32::try_from(position)
                .map_err(|_| format!("attention twin: position {position} does not fit a u32"))?,
            sink[head],
        );
        if head_out.len() != MLA_LATENT {
            return Err(format!(
                "attention twin: head {head} reference is {} values, one head is {MLA_LATENT}",
                head_out.len()
            ));
        }
        let start = head * MLA_LATENT;
        for slot in 0..MLA_LATENT {
            let diff = (device_out[start + slot] - head_out[slot]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            if sparkflash_kernels::m4_quant::bf16_bits(device_out[start + slot])
                != sparkflash_kernels::m4_quant::bf16_bits(head_out[slot])
            {
                mismatches += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let exact_fraction = 1.0 - mismatches as f64 / width as f64;
    Ok(TwinStats {
        heads,
        window_rows: staged.window_rows,
        selected: staged.selected.len(),
        mismatches,
        slots: width,
        exact_fraction,
        max_abs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_engine::forward::attention::AttentionSeamStep;
    use sparkflash_kernels::cuda_harness::engine_window_row;

    /// Deterministic small values, exactly representable in the codecs the staging uses.
    fn values(count: usize, seed: u32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let value = ((index as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed)
                    >> 9)
                    % 97;
                (value as f32 - 48.0) * 0.03125
            })
            .collect()
    }

    fn encoded_row(seed: u32) -> (Vec<u8>, Vec<u8>) {
        let row = values(MLA_LATENT, seed);
        let encoded = engine_window_row(&row).expect("window row");
        (encoded.codes, encoded.scales)
    }

    /// One stored compressed row per index, as the seam would hand them over.
    fn stored_rows(count: usize, seed: u32) -> Vec<f32> {
        (0..count)
            .flat_map(|index| values(MLA_LATENT, seed + index as u32))
            .collect()
    }

    fn seam(
        ratio: usize,
        window_rows: usize,
        width: usize,
        ids: &[usize],
        rows: &[f32],
    ) -> AttentionSeamStep {
        AttentionSeamStep {
            ratio,
            window_rows,
            window_width: width,
            compressed_ids: ids.to_vec(),
            compressed_rows: rows.to_vec(),
        }
    }

    #[test]
    fn a_window_only_layer_stages_its_rows_and_no_selector() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..3).map(encoded_row).collect();
        let staged = stage_attention(&window, &seam(0, 3, 3, &[], &[])).expect("staged");
        assert_eq!(staged.window_rows, 3);
        assert_eq!(staged.window_codes.len(), 3 * WINDOW_CODE_BYTES);
        assert_eq!(staged.window_scales.len(), 3 * WINDOW_SCALE_BYTES);
        for (row, (codes, scales)) in window.iter().enumerate() {
            let start = row * WINDOW_CODE_BYTES;
            assert_eq!(
                &staged.window_codes[start..start + WINDOW_CODE_BYTES],
                codes
            );
            let start = row * WINDOW_SCALE_BYTES;
            assert_eq!(
                &staged.window_scales[start..start + WINDOW_SCALE_BYTES],
                scales
            );
        }
        assert!(staged.selected.is_empty());
        assert!(staged.kv_codes.iter().all(|byte| *byte == 0));
        assert_eq!(staged.store_rows, 0);
    }

    #[test]
    fn compressed_ids_expand_to_the_ratio_store_rows_the_kernel_reads() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..2).map(encoded_row).collect();
        // Ratio 2, ids 129 and 130 are compressed rows j = 1 and j = 2 (the shared index space is
        // window_rows + j), so three stored rows cover them and j = 0 stays unselected.
        let stored = stored_rows(3, 5);
        let staged =
            stage_attention(&window, &seam(2, 128, 2, &[129, 130], &stored)).expect("staged");
        assert_eq!(staged.selected, vec![2u32, 4u32]);
        assert_eq!(staged.store_rows, 2);
        let expected = quantize_main_kv(&values(MLA_LATENT, 6)).expect("codec");
        assert_eq!(
            &staged.kv_codes[2 * STORE_CODE_BYTES..3 * STORE_CODE_BYTES],
            &expected.data
        );
        assert_eq!(
            &staged.kv_scales[2 * STORE_SCALE_BYTES..3 * STORE_SCALE_BYTES],
            &expected.scales
        );
        let expected = quantize_main_kv(&values(MLA_LATENT, 7)).expect("codec");
        assert_eq!(
            &staged.kv_codes[4 * STORE_CODE_BYTES..5 * STORE_CODE_BYTES],
            &expected.data
        );
        assert!(
            staged.kv_codes[..2 * STORE_CODE_BYTES]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn ratio_one_stores_every_selected_row_at_its_own_expanded_row() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..1).map(encoded_row).collect();
        let stored = stored_rows(11, 3);
        let staged =
            stage_attention(&window, &seam(1, 128, 1, &[137, 138], &stored)).expect("staged");
        assert_eq!(staged.selected, vec![9u32, 10u32]);
        let expected = quantize_main_kv(&values(MLA_LATENT, 12)).expect("codec");
        assert_eq!(
            &staged.kv_codes[9 * STORE_CODE_BYTES..10 * STORE_CODE_BYTES],
            &expected.data
        );
    }

    #[test]
    fn an_unreachable_id_is_skipped_and_keeps_the_selection_order() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..1).map(encoded_row).collect();
        let stored = stored_rows(2, 13);
        let staged =
            stage_attention(&window, &seam(2, 128, 1, &[129, NO_INDEX], &stored)).expect("staged");
        assert_eq!(staged.selected, vec![2u32]);
        assert_eq!(staged.store_rows, 1);
    }

    #[test]
    fn the_twin_agrees_with_itself_over_the_staged_bytes() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..4).map(encoded_row).collect();
        let stored = stored_rows(3, 17);
        let staged =
            stage_attention(&window, &seam(2, 128, 4, &[129, 130], &stored)).expect("staged");
        let q = values(4 * MLA_LATENT, 23);
        let sink = [0.25f32, -0.5, 0.125, 0.75];
        let mut reference = vec![0.0f32; 4 * MLA_LATENT];
        for head in 0..4 {
            let out = sparkflash_kernels::mla_ref::attention_with_sink_union(
                &q[head * MLA_LATENT..(head + 1) * MLA_LATENT],
                &staged.window_codes,
                &staged.window_scales,
                staged.window_rows,
                &staged.kv_codes,
                &staged.kv_scales,
                &staged.selected,
                137,
                sink[head],
            );
            reference[head * MLA_LATENT..(head + 1) * MLA_LATENT].copy_from_slice(&out);
        }
        let stats = twin_out(&q, &sink, &staged, 137, &reference).expect("twin");
        assert_eq!(stats.mismatches, 0);
        assert_eq!(stats.heads, 4);
        assert_eq!(stats.window_rows, staged.window_rows);
        assert_eq!(stats.selected, 2);
        assert_eq!(stats.max_abs, 0.0);
    }

    #[test]
    fn a_row_count_disagreement_is_refused() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..2).map(encoded_row).collect();
        let error = stage_attention(&window, &seam(0, 1, 1, &[], &[])).expect_err("refused");
        assert!(error.contains("window rows"), "{error}");
    }

    #[test]
    fn an_id_below_the_window_offset_is_refused() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..2).map(encoded_row).collect();
        let error = stage_attention(&window, &seam(2, 128, 2, &[127], &[])).expect_err("refused");
        assert!(error.contains("below the window offset"), "{error}");
    }
    /// The shipped decode shape: a 128-row window, the released compress ratios, and a selected set
    /// that grows by one row per position. Every call must stage exactly [stage_attention]'s bytes;
    /// the slot only removes per-call host work, it never changes a byte.
    #[test]
    fn reused_staging_matches_the_reference_bytes_across_a_decode_sequence() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..128).map(encoded_row).collect();
        let rows = stored_rows(300, 31);
        for ratio in [1usize, 2] {
            let mut staging = AttentionStaging::new();
            for position in 1..=64usize {
                let selected = position.min(48);
                let ids: Vec<usize> = (0..selected).map(|row| 128 + row).collect();
                let seam = seam(ratio, 128, 128, &ids, &rows);
                let reference = stage_attention(&window, &seam).expect("reference staged");
                let staged = staging.stage(0, &window, &seam).expect("staged");
                assert_eq!(staged, reference, "ratio {ratio} position {position}");
                staging.recycle(staged);
            }
        }
    }

    /// The window-only layers (ratio 0) stage no selected row and must still reuse the slot.
    #[test]
    fn a_window_only_layer_reuses_its_slot_without_selecting_rows() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..128).map(encoded_row).collect();
        let mut staging = AttentionStaging::new();
        for round in 0..4 {
            let seam = seam(0, 128, 128, &[], &[]);
            let reference = stage_attention(&window, &seam).expect("reference staged");
            let staged = staging.stage(0, &window, &seam).expect("staged");
            assert_eq!(staged, reference, "round {round}");
            staging.recycle(staged);
        }
    }

    /// The memo is checked, not trusted: the same (layer, row) with different stored bits must be
    /// quantised again, and the staged bytes must equal the reference over the new row.
    #[test]
    fn a_changed_compressed_row_is_quantised_again_instead_of_reused() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..4).map(encoded_row).collect();
        let mut rows = stored_rows(8, 41);
        let ids = [128usize, 129];
        let mut staging = AttentionStaging::new();
        let first = staging
            .stage(0, &window, &seam(2, 128, 4, &ids, &rows))
            .expect("staged");
        assert_eq!(
            first,
            stage_attention(&window, &seam(2, 128, 4, &ids, &rows)).expect("reference")
        );
        let first_codes = first.kv_codes.clone();
        staging.recycle(first);
        // Row 1 (id 129) changes between the two calls; a stale memo entry would hand back the old
        // codes, so the reference comparison below is what proves the check ran.
        for value in &mut rows[MLA_LATENT..2 * MLA_LATENT] {
            *value = *value * 4.0 + 1.0;
        }
        let changed = staging
            .stage(0, &window, &seam(2, 128, 4, &ids, &rows))
            .expect("staged");
        assert_eq!(
            changed,
            stage_attention(&window, &seam(2, 128, 4, &ids, &rows)).expect("reference")
        );
        assert_ne!(
            changed.kv_codes, first_codes,
            "the changed row must not keep its old bytes"
        );
        staging.recycle(changed);
    }

    /// A refusal after the slot took its buffers must not poison the next call.
    #[test]
    fn a_refused_stage_leaves_the_slot_usable() {
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..2).map(encoded_row).collect();
        let rows = stored_rows(3, 43);
        let mut staging = AttentionStaging::new();
        assert!(
            staging
                .stage(0, &window, &seam(2, 128, 2, &[127], &rows))
                .is_err()
        );
        let good = seam(2, 128, 2, &[129], &rows);
        let staged = staging.stage(0, &window, &good).expect("staged");
        assert_eq!(staged, stage_attention(&window, &good).expect("reference"));
        staging.recycle(staged);
    }

    /// Prints the per-call host cost of one staging call on the shipped window width, for the
    /// reference builder and the reusable slot in its steady state (the memo warm, one fresh row
    /// per position as a decode step produces). Run with:
    ///   cargo test --release -p sparkflash-cli --lib staging_pricing -- --ignored --nocapture
    /// Prints the per-call host cost of one staging call on the shipped window width, for the
    /// reference builder and the reusable slot, at several selected-row counts and over the
    /// released layer mix (2 window-only, 18 ratio 2, 20 ratio 1). Run with:
    ///   cargo test --release -p sparkflash-cli --lib staging_pricing -- --ignored --nocapture
    #[test]
    #[ignore = "pricing harness: prints per-call staging cost, not a gate"]
    fn staging_pricing() {
        use std::time::Instant;
        let window: Vec<(Vec<u8>, Vec<u8>)> = (0..128).map(encoded_row).collect();
        let rows = stored_rows(600, 47);
        for selected in [1usize, 8, 16, 32, 64, 128, 512] {
            let ids: Vec<usize> = (0..selected).map(|row| 128 + row).collect();
            let seam = seam(1, 128, 128, &ids, &rows);
            let iterations = 200usize;
            let started = Instant::now();
            for _ in 0..iterations {
                let staged = stage_attention(&window, &seam).expect("staged");
                std::hint::black_box(&staged);
            }
            let reference_us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
            let mut staging = AttentionStaging::new();
            let warm = staging.stage(0, &window, &seam).expect("staged");
            staging.recycle(warm);
            let started = Instant::now();
            for _ in 0..iterations {
                let staged = staging.stage(0, &window, &seam).expect("staged");
                std::hint::black_box(&staged);
                staging.recycle(staged);
            }
            let reused_us = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
            println!(
                "stage selected={selected:3} reference={reference_us:8.1}us reused={reused_us:8.1}us"
            );
        }
        // The released layer mix, at several selected counts: the row's measured 0.324 ms/call
        // points at the count whose reference column it lands on, and the reused column is the
        // after.
        for n1 in [16usize, 32, 48, 64] {
            let n2 = n1 / 2;
            let seams: Vec<AttentionSeamStep> = (0..40usize)
                .map(|layer| {
                    let (ratio, count, stored): (usize, usize, &[f32]) = match layer {
                        0 | 1 => (0usize, 0usize, &[]),
                        ..=19 => (2usize, n2, &rows),
                        _ => (1usize, n1, &rows),
                    };
                    let ids: Vec<usize> = (0..count).map(|row| 128 + row).collect();
                    seam(ratio, 128, 128, &ids, stored)
                })
                .collect();
            let mut staging = AttentionStaging::new();
            for (layer, seam) in seams.iter().enumerate() {
                let warm = staging.stage(layer, &window, seam).expect("staged");
                staging.recycle(warm);
            }
            let passes = 20usize;
            let started = Instant::now();
            for _ in 0..passes {
                for seam in &seams {
                    let staged = stage_attention(&window, seam).expect("staged");
                    std::hint::black_box(&staged);
                }
            }
            let reference_ms = started.elapsed().as_secs_f64() * 1e3 / passes as f64;
            let started = Instant::now();
            for _ in 0..passes {
                for (layer, seam) in seams.iter().enumerate() {
                    let staged = staging.stage(layer, &window, seam).expect("staged");
                    std::hint::black_box(&staged);
                    staging.recycle(staged);
                }
            }
            let reused_ms = started.elapsed().as_secs_f64() * 1e3 / passes as f64;
            println!(
                "layer-mix n1={n1:3} reference={reference_ms:7.3}ms/step ({:.1}us/call) reused={reused_ms:6.3}ms/step ({:.1}us/call) saved={:6.3}ms/step",
                reference_ms * 1000.0 / 40.0,
                reused_ms * 1000.0 / 40.0,
                reference_ms - reused_ms
            );
        }
    }
}
