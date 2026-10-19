//! The c4a device attention backend: the five attention projection GEMMs, the indexer's wq_b
//! when a layer has compressed positions, and the format-2 CSA-union attention body on the
//! device, with the host normalisation, RoPE, KV, cache, compressor and selected-list seams left
//! exactly where they are. This module is only compiled with `--features cuda`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::sparse_attention;
use sparkflash_engine::forward::attention::{AttentionProjectionBackend, indexer_wq_b_chunked};
use sparkflash_engine::forward::attention_case::{AttentionCaseRecord, NO_INDEX as CASE_NO_INDEX};
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::quant::quantize_activation_fp8;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::bf16a_mxfp8w_ref::{self, CodeComparison, PathComparison};
use sparkflash_kernels::bf16a_mxfp8w_wo_a_launch::{self, Bf16aMxfp8wWoAParams};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use sparkflash_kernels::cuda_harness::{
    FlashMlaKernel, engine_window_row, flash_mla_params, launch_flash_mla, load_flash_mla,
};
use sparkflash_kernels::flash_mla_csa2_seq_launch::{self, FlashMlaSeqParams};
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mxfp8_dense_launch::{self, Mxfp8DenseParams};
use sparkflash_kernels::mxfp8_ref::b12x_weight_index;
use sparkflash_kernels::mxfp8_wo_a_launch::{self, Mxfp8WoAParams};
use sparkflash_kv::fp4::{dequantize_main_kv, quantize_main_kv};
/// One packed projection weight: the value bytes and their block scales, read once per name.
type PackedWeight = Arc<(Vec<u8>, Vec<u8>)>;

const DENSE_KERNEL: &str = "b12x_mxfp8_dense_sm121";
const WO_A_KERNEL: &str = "b12x_mxfp8_wo_a_sm121";
/// The BF16-activation grouped wo_a cubin: the native path.
const WO_A_NATIVE_KERNEL: &str = "b12x_bf16a_mxfp8w_wo_a_sm121";
const FLASH_MLA_KERNEL: &str = "flash_mla_csa2_sm121";
const FLASH_MLA_SEQ_KERNEL: &str = "flash_mla_csa2_seq_sm121";
/// The CSA-union format value the m14 kernel expects.
const FORMAT_CSA_UNION: u64 = 2;
/// 64 heads x 512 latent x bf16.
const MLA_Q_BYTES: usize = 64 * 512 * 2;
/// The 128-row window ring's code and scale rows.
const MAX_WINDOW_CODES: usize = 128 * 512;
const MAX_WINDOW_SCALES: usize = 128 * 16;
/// The expanded compressed-view capacity; c4a-2 exercises position 0 only.
const MAX_KV_FP4: usize = 512 * 256;
const MAX_KV_SCALES: usize = 512 * 32;
const MAX_SELECTED: usize = 512 * 4;
const HSI_BLOCKS: usize = 2_048;
const HSI_INDEX_KEY_BYTES: usize = HSI_BLOCKS * 8 * 128 * 2;
const HSI_INDEX_QUERY_BYTES: usize = 64 * 128 * 2;
const HSI_BLOCK_BYTES: usize = HSI_BLOCKS * 4;
const MLA_SINK_BYTES: usize = 64 * 4;

/// The maximum packed weight size: wo_b's 5,120 x 8,192 E4M3 bytes.
const MAX_WEIGHT_BYTES: usize = 5_120 * 8_192;
/// The maximum packed weight-scale size: wo_b's [160, 256] grid.
const MAX_WEIGHT_SCALES: usize = 160 * 256;
/// The maximum activation code count: the 64 x 512-wide attention output.
const MAX_ACT_BYTES: usize = 64 * 512;
/// The maximum activation scale count: one per 32 codes.
const MAX_ACT_SCALES: usize = MAX_ACT_BYTES / 32;
/// The maximum BF16 activation bytes: the same 64 x 512-wide output at two bytes each.
const MAX_ACT_BF16_BYTES: usize = MAX_ACT_BYTES * 2;
/// The maximum activation output: wq_b's 32,768 bf16 values.
const MAX_OUT_BYTES: usize = 32_768 * 2;

/// Which wo_a arithmetic the chain runs.
///
/// [WoAMode::Native] is the released rule (native-arithmetic-audit F1/M1): the inverse-RoPE angle
/// is applied to the BF16 body output and no activation quantiser runs, so the grouped GEMM is a
/// BF16 activation times the dequantised FP8 weight. [WoAMode::Mxfp8] is the chain's previous path
/// and stays selectable, so a window log shows which arithmetic it measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WoAMode {
    /// BF16 activation, no activation quantisation.
    #[default]
    Native,
    /// E4M3 activation with per-32 ue8m0 scales.
    Mxfp8,
}

impl WoAMode {
    /// Parses the --wo-a-mode spelling.
    ///
    /// # Errors
    /// Any spelling other than native or mxfp8.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "native" => Ok(Self::Native),
            "mxfp8" => Ok(Self::Mxfp8),
            other => Err(format!(
                "--wo-a-mode: {other} is not a wo_a mode (native or mxfp8)"
            )),
        }
    }

    /// The spelling a run log records.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

/// One layer's wo_a comparison at that layer's first position.
///
/// It records two things on the SAME captured operand: the native launch against the native
/// reference (the kernel-versus-twin check), and the current MXFP8 path against that same
/// reference (the F1 deviation measured on one operand).
#[derive(Debug, Clone)]
pub struct WoAComparison {
    /// Backbone layer index.
    pub layer: usize,
    /// Which wo_a arithmetic the device launch used.
    pub mode: WoAMode,
    /// The native device output against the BF16 reference on the same activation codes.
    pub device_vs_native: CodeComparison,
    /// The current MXFP8 path against the same reference.
    pub mxfp8_vs_native: PathComparison,
    /// XOR-fold of the operand's f32 bits, so an offline dump can be matched to a record.
    pub operand_checksum: u64,
}

/// Positions the in-chain attention twin compares at. The prefill diagnostic positions stay at
/// 0..2, and the ring's first two wrap steps are armed by default: with a window of W the write
/// at position W lands in slot 0 (the first row the ring evicts) and the write at W + 1 rotates
/// the ring's oldest slot. b70's sequence diverges exactly at the 130th token with W = 128, which
/// is the second wrap step, so an unarmed boundary cannot be distinguished from a correct one.
/// A caller can replace the set (for example to diagnose one position); the set selects positions,
/// it never switches the gate off.
#[must_use]
pub fn default_twin_positions(window: usize) -> Vec<usize> {
    let mut positions = vec![0usize, 1, 2];
    for position in [window, window + 1] {
        if !positions.contains(&position) {
            positions.push(position);
        }
    }
    positions
}

/// One launch's bookkeeping.
#[derive(Debug, Default, Clone, Copy)]
pub struct AttentionDeviceStats {
    /// GEMM launches: five per layer when every attention projection runs, plus the indexer wq_b's
    /// row chunks where a layer has compressed positions.
    pub launches: usize,
    /// Host microseconds spent reading, packing, quantising and uploading.
    pub staging_us: u64,
    /// Host microseconds spent inside the launch and its readback.
    pub device_us: u64,
    /// Cumulative packed weight bytes uploaded.
    pub staged_bytes: u64,
    /// Cumulative activation bytes uploaded and read back.
    pub host_bytes: u64,
    /// Format-2 attention launches (one per prefill layer at position 0).
    pub attention_launches: usize,
    /// Host microseconds spent building and launching the format-2 attention body.
    pub attention_us: u64,
    /// Layers seen, for the ledger.
    pub layers: usize,
}

/// One staged attention operand summary for the b56 follow-up: q and sink checksums, the exact
/// window and selected ids the device staged, and the first/last window-code checksums. It is a
/// diagnostic copy only and does not feed the launch.
#[derive(Debug, Clone)]
pub struct AttentionOperandTrace {
    /// Absolute query position.
    pub position: usize,
    /// Backbone layer.
    pub layer: usize,
    /// The q row the body consumed.
    pub q_values: Vec<f32>,
    /// First q value.
    pub q_first: f32,
    /// Last q value.
    pub q_last: f32,
    /// XOR-fold of the q BF16 words (enough to compare device and host staging).
    pub q_checksum: u64,
    /// First sink value.
    pub sink_first: f32,
    /// Last sink value.
    pub sink_last: f32,
    /// XOR-fold of the sink words.
    pub sink_checksum: u64,
    /// The window ids the device staged, oldest first.
    pub window_ids: Vec<usize>,
    /// The selected compressed ids the device staged.
    pub selected: Vec<usize>,
    /// Number of compressed rows staged.
    pub compressed_count: usize,
    /// Inferred compression ratio.
    pub ratio: usize,
    /// Number of staged window codes.
    pub window_codes_len: usize,
    /// XOR-fold of the first window row's codes.
    pub window_first_checksum: u64,
    /// XOR-fold of the last window row's codes.
    pub window_last_checksum: u64,
    /// Number of staged compressed code bytes.
    pub compressed_codes_len: usize,
    /// The staged window row codes, row by row in staged order (512 bytes per row). The operand
    /// audit compares these against the reference's own window rows quantised the same way.
    pub window_codes: Vec<u8>,
    /// The staged window row block scales (16 bytes per row).
    pub window_scales: Vec<u8>,
    /// The staged compressed pool codes, in pool-row order (256 bytes per row, ratio-major).
    pub compressed_codes: Vec<u8>,
    /// The staged compressed pool block scales (32 bytes per row).
    pub compressed_scales: Vec<u8>,
    /// The sink logits the body consumed.
    pub sink_values: Vec<f32>,
}

fn checksum_f32(values: &[f32]) -> u64 {
    values.iter().fold(0u64, |acc, value| {
        acc.rotate_left(5) ^ value.to_bits() as u64
    })
}

fn checksum_bytes(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, value| acc.rotate_left(5) ^ *value as u64)
}

struct Kernels {
    dense: Kernel,
    wo_a: Kernel,
    wo_a_native: Kernel,
    flash: FlashMlaKernel,
    flash_seq: Kernel,
    _dense_module: Module,
    _wo_a_module: Module,
    _wo_a_native_module: Module,
    _flash_module: Module,
    _flash_seq_module: Module,
}

struct Buffers {
    a_codes: Deviceptr,
    a_scales: Deviceptr,
    a_bf16: Deviceptr,
    w: Deviceptr,
    w_scales: Deviceptr,
    out: Deviceptr,
    ptrs: Vec<Deviceptr>,
}

struct MlaBuffers {
    q: Deviceptr,
    kv_fp4: Deviceptr,
    kv_scales: Deviceptr,
    window_codes: Deviceptr,
    window_scales: Deviceptr,
    selected: Deviceptr,
    index_key: Deviceptr,
    index_query: Deviceptr,
    block_scores: Deviceptr,
    block_best: Deviceptr,
    sink: Deviceptr,
    out: Deviceptr,
    ptrs: Vec<Deviceptr>,
}

/// The device implementation of [AttentionProjectionBackend]. It stages one layer's weights at a
/// time into reused buffers (the indexer's wq_b included, when that projection runs); it never
/// allocates the 5.07 GB cumulative matrix set on the device.
pub struct AttentionDeviceBackend {
    cuda: Cuda,
    kernels: Kernels,
    buffers: Buffers,
    mla: MlaBuffers,
    stats: AttentionDeviceStats,
    // Per-layer format-2 launch counts, so the report can name each layer instead of trusting a
    // cumulative total. A layer that silently falls back to the host stays zero here.
    body_by_layer: Vec<usize>,
    // The activation handed from the engine to the next GEMM call. The trait has one primitive per
    // call, so the quantised operand is stashed here between the call that builds it and the one
    // that consumes it; this is a single-threaded chain, and each layer consumes it immediately.
    last_activation: Option<(Vec<u8>, Vec<u8>)>,
    // Packed attention weights, read once per (layer, projection) name and reused across sequence
    // positions. This is the M3-d staging-reuse shape for the projection set; host bytes only.
    weight_cache: HashMap<String, PackedWeight>,
    // Which wo_a arithmetic this chain runs, set once at construction.
    wo_a_mode: WoAMode,
    // Whether the per-layer wo_a comparison (and operand dump) is armed.
    wo_a_diagnose: bool,
    // One slot per layer: whether that layer's comparison already ran.
    wo_a_compared: Vec<bool>,
    wo_a_comparisons: Vec<WoAComparison>,
    // Where to write each layer's first-position operand and device output, when set.
    wo_a_operand_dump: Option<std::path::PathBuf>,
    // Positions 0-2's staged operands for the residual/alignment diagnostic; empty otherwise.
    operand_traces: Vec<AttentionOperandTrace>,
    // The positions the in-chain attention twin compares at (see [default_twin_positions]). The
    // boundary positions are armed by default, so a ring eviction cannot pass unobserved again.
    twin_positions: Vec<usize>,
    // A/B switch: from this position on, the device attention body is skipped and the engine's host
    // attention loop runs on the same operands. None keeps the device body everywhere.
    host_attention_from: Option<usize>,
    // Layer 0's first-position raw projection outputs, for the stage-aligned projection twin.
    diag_latent_raw: Option<Vec<f32>>,
    diag_q_raw: Option<Vec<f32>>,
    diag_wq_b_input: Option<Vec<f32>>,
    diag_wkv_raw: Option<Vec<f32>>,
    diag_window_codes: Option<Vec<u8>>,
    diag_wq_b_kernel: Option<(usize, usize, f32)>,
}

impl AttentionDeviceBackend {
    /// Verifies and loads the two projection cubins, then opens the driver and allocates the reused
    /// activation and weight buffers.
    ///
    /// # Errors
    /// A missing/mis-pinned cubin, a driver failure, or an allocation failure.
    pub fn open(cubin_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = cubin_dir.as_ref();
        let dense_bytes = verified_cubin(dir, DENSE_KERNEL)?;
        let wo_a_bytes = verified_cubin(dir, WO_A_KERNEL)?;
        let wo_a_native_bytes = verified_cubin(dir, WO_A_NATIVE_KERNEL)?;
        let flash_bytes = verified_cubin(dir, FLASH_MLA_KERNEL)?;
        let flash_seq_bytes = verified_cubin(dir, FLASH_MLA_SEQ_KERNEL)?;
        let cuda = Cuda::load()?;
        let (dense_module, dense) = cuda.load_kernel(&dense_bytes, KernelId::Mxfp8Dense)?;
        let (wo_a_module, wo_a) = cuda.load_kernel(&wo_a_bytes, KernelId::Mxfp8WoA)?;
        let (wo_a_native_module, wo_a_native) =
            bf16a_mxfp8w_wo_a_launch::load(&cuda, &wo_a_native_bytes)?;
        let (flash_module, flash) = load_flash_mla(&cuda, &flash_bytes)?;
        let (flash_seq_module, flash_seq) =
            flash_mla_csa2_seq_launch::load(&cuda, &flash_seq_bytes)?;
        let mut total = 0usize;
        let alloc = |bytes: usize, total: &mut usize| cuda.alloc(bytes, total);
        let a_codes = alloc(MAX_ACT_BYTES, &mut total)?;
        let a_scales = alloc(MAX_ACT_SCALES, &mut total)?;
        let a_bf16 = alloc(MAX_ACT_BF16_BYTES, &mut total)?;
        let w = alloc(MAX_WEIGHT_BYTES, &mut total)?;
        let w_scales = alloc(MAX_WEIGHT_SCALES, &mut total)?;
        let out = alloc(MAX_OUT_BYTES, &mut total)?;
        let ptrs = vec![a_codes, a_scales, a_bf16, w, w_scales, out];
        let q = alloc(MLA_Q_BYTES, &mut total)?;
        let kv_fp4 = alloc(MAX_KV_FP4, &mut total)?;
        let kv_scales = alloc(MAX_KV_SCALES, &mut total)?;
        let window_codes = alloc(MAX_WINDOW_CODES, &mut total)?;
        let window_scales = alloc(MAX_WINDOW_SCALES, &mut total)?;
        let selected = alloc(MAX_SELECTED, &mut total)?;
        let index_key = alloc(HSI_INDEX_KEY_BYTES, &mut total)?;
        let index_query = alloc(HSI_INDEX_QUERY_BYTES, &mut total)?;
        let block_scores = alloc(HSI_BLOCK_BYTES, &mut total)?;
        let block_best = alloc(HSI_BLOCK_BYTES, &mut total)?;
        let sink = alloc(MLA_SINK_BYTES, &mut total)?;
        let mla_out = alloc(MLA_Q_BYTES, &mut total)?;
        let mla_ptrs = vec![
            q,
            kv_fp4,
            kv_scales,
            window_codes,
            window_scales,
            selected,
            index_key,
            index_query,
            block_scores,
            block_best,
            sink,
            mla_out,
        ];
        cuda.copy_htod(&kv_fp4, &vec![0u8; MAX_KV_FP4])?;
        cuda.copy_htod(&kv_scales, &vec![0u8; MAX_KV_SCALES])?;
        cuda.copy_htod(&index_key, &vec![0u8; HSI_INDEX_KEY_BYTES])?;
        cuda.copy_htod(&index_query, &vec![0u8; HSI_INDEX_QUERY_BYTES])?;
        cuda.copy_htod(&block_scores, &f32_bytes(&vec![12345.0f32; HSI_BLOCKS]))?;
        cuda.copy_htod(&block_best, &u32_bytes(&vec![u32::MAX; HSI_BLOCKS]))?;
        Ok(Self {
            cuda,
            kernels: Kernels {
                dense,
                wo_a,
                wo_a_native,
                flash,
                flash_seq,
                _dense_module: dense_module,
                _wo_a_module: wo_a_module,
                _wo_a_native_module: wo_a_native_module,
                _flash_module: flash_module,
                _flash_seq_module: flash_seq_module,
            },
            buffers: Buffers {
                a_codes,
                a_scales,
                a_bf16,
                w,
                w_scales,
                out,
                ptrs,
            },
            mla: MlaBuffers {
                q,
                kv_fp4,
                kv_scales,
                window_codes,
                window_scales,
                selected,
                index_key,
                index_query,
                block_scores,
                block_best,
                sink,
                out: mla_out,
                ptrs: mla_ptrs,
            },
            stats: AttentionDeviceStats::default(),
            body_by_layer: vec![0; 64],
            last_activation: None,
            weight_cache: HashMap::new(),
            wo_a_mode: WoAMode::default(),
            wo_a_diagnose: false,
            wo_a_compared: vec![false; 64],
            wo_a_comparisons: Vec::new(),
            wo_a_operand_dump: None,
            operand_traces: Vec::new(),
            twin_positions: default_twin_positions(flash_mla_csa2_seq_launch::SWA_WINDOW),
            host_attention_from: None,
            diag_latent_raw: None,
            diag_q_raw: None,
            diag_wq_b_input: None,
            diag_wkv_raw: None,
            diag_window_codes: None,
            diag_wq_b_kernel: None,
        })
    }

    /// The counters since construction.
    #[must_use]
    pub fn stats(&self) -> AttentionDeviceStats {
        self.stats
    }

    /// Selects the wo_a arithmetic; the chain records its name in the run output.
    #[must_use]
    pub fn with_wo_a_mode(mut self, mode: WoAMode) -> Self {
        self.wo_a_mode = mode;
        self
    }

    /// The wo_a arithmetic this backend runs.
    #[must_use]
    pub fn wo_a_mode(&self) -> WoAMode {
        self.wo_a_mode
    }

    /// Arms the per-layer wo_a comparison at each layer's first position.
    #[must_use]
    pub fn with_wo_a_native_diagnose(mut self) -> Self {
        self.wo_a_diagnose = true;
        self
    }

    /// Writes each layer's first-position wo_a operand and device output under a directory.
    #[must_use]
    pub fn with_wo_a_operand_dump(mut self, dir: std::path::PathBuf) -> Self {
        self.wo_a_operand_dump = Some(dir);
        self
    }

    /// The per-layer comparisons recorded so far, in layer order.
    #[must_use]
    pub fn wo_a_comparisons(&self) -> &[WoAComparison] {
        &self.wo_a_comparisons
    }

    /// The format-2 body launches this backend performed for `layer`.
    #[must_use]
    pub fn body_launches(&self, layer: usize) -> usize {
        self.body_by_layer.get(layer).copied().unwrap_or(0)
    }

    /// The staged operand summaries for positions 0-2, in launch order.
    #[must_use]
    pub fn operand_traces(&self) -> &[AttentionOperandTrace] {
        &self.operand_traces
    }

    /// Layer 0's first-position raw wq_a latent, for the projection twin.
    #[must_use]
    pub fn diag_latent_raw(&self) -> Option<&[f32]> {
        self.diag_latent_raw.as_deref()
    }

    /// Layer 0's first-position raw wq_b q, before RoPE.
    #[must_use]
    pub fn diag_q_raw(&self) -> Option<&[f32]> {
        self.diag_q_raw.as_deref()
    }

    /// Layer 0's first-position wq_b input latent, after q_norm.
    #[must_use]
    pub fn diag_wq_b_input(&self) -> Option<&[f32]> {
        self.diag_wq_b_input.as_deref()
    }

    /// Layer 0's first-position raw wkv output, before kv_norm.
    #[must_use]
    pub fn diag_wkv_raw(&self) -> Option<&[f32]> {
        self.diag_wkv_raw.as_deref()
    }

    /// Layer 0's first-position staged window codes.
    #[must_use]
    pub fn diag_window_codes(&self) -> Option<&[u8]> {
        self.diag_window_codes.as_deref()
    }

    /// The wq_b kernel-vs-reference comparison: (BF16 mismatches, length, max abs diff).
    #[must_use]
    pub fn diag_wq_b_kernel(&self) -> Option<(usize, usize, f32)> {
        self.diag_wq_b_kernel
    }

    /// The persistent device bytes this backend allocated.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        (MAX_ACT_BYTES
            + MAX_ACT_SCALES
            + MAX_ACT_BF16_BYTES
            + MAX_WEIGHT_BYTES
            + MAX_WEIGHT_SCALES
            + MAX_OUT_BYTES
            + MLA_Q_BYTES
            + MAX_KV_FP4
            + MAX_KV_SCALES
            + MAX_WINDOW_CODES
            + MAX_WINDOW_SCALES
            + MAX_SELECTED
            + HSI_INDEX_KEY_BYTES
            + HSI_INDEX_QUERY_BYTES
            + HSI_BLOCK_BYTES * 2
            + MLA_SINK_BYTES
            + MLA_Q_BYTES) as u64
    }

    fn upload_activation(&mut self, codes: &[u8], scales: &[u8]) -> Result<(), String> {
        if codes.len() > MAX_ACT_BYTES || scales.len() > MAX_ACT_SCALES {
            return Err(format!(
                "activation ({} B codes, {} B scales) does not fit the staging buffers",
                codes.len(),
                scales.len()
            ));
        }
        self.cuda.copy_htod(&self.buffers.a_codes, codes)?;
        self.cuda.copy_htod(&self.buffers.a_scales, scales)?;
        Ok(())
    }

    fn upload_activation_bf16(&mut self, values: &[u16]) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        if bytes.len() > MAX_ACT_BF16_BYTES {
            return Err(format!(
                "BF16 activation ({} B) does not fit the {MAX_ACT_BF16_BYTES} B staging buffer",
                bytes.len()
            ));
        }
        self.cuda.copy_htod(&self.buffers.a_bf16, &bytes)?;
        Ok(())
    }

    fn upload_weight(&mut self, packed: &[u8], scales: &[u8]) -> Result<(), String> {
        if packed.len() > MAX_WEIGHT_BYTES || scales.len() > MAX_WEIGHT_SCALES {
            return Err(format!(
                "weight ({} B codes, {} B scales) does not fit the staging buffers",
                packed.len(),
                scales.len()
            ));
        }
        self.cuda.copy_htod(&self.buffers.w, packed)?;
        self.cuda.copy_htod(&self.buffers.w_scales, scales)?;
        self.stats.staged_bytes = self
            .stats
            .staged_bytes
            .saturating_add(packed.len() as u64 + scales.len() as u64);
        Ok(())
    }

    fn packed_weight(
        &mut self,
        src: &dyn WeightSource,
        name: &str,
        cols: usize,
        inner: usize,
    ) -> Result<PackedWeight, EngineError> {
        if let Some(entry) = self.weight_cache.get(name) {
            return Ok(Arc::clone(entry));
        }
        let entry = Arc::new(read_packed(src, name, cols, inner)?);
        self.weight_cache
            .insert(name.to_owned(), Arc::clone(&entry));
        Ok(entry)
    }

    fn read_out(&mut self, bytes: usize) -> Result<Vec<f32>, String> {
        let mut raw = vec![0u8; bytes];
        self.cuda.copy_dtoh(&mut raw, &self.buffers.out)?;
        Ok(decode_bf16(&raw))
    }

    #[allow(clippy::too_many_arguments)]
    fn dense(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        src: &dyn WeightSource,
        weight_name: &str,
        cols: usize,
        inner: usize,
        rows: usize,
    ) -> Result<Vec<f32>, EngineError> {
        if rows * cols * 2 > MAX_OUT_BYTES {
            return Err(EngineError::Shape {
                what: "attention dense output".to_owned(),
                detail: format!(
                    "{rows} x {cols} bf16 values do not fit the {MAX_OUT_BYTES} B staging buffer"
                ),
            });
        }
        let started = std::time::Instant::now();
        let weight =
            self.packed_weight(src, &format!("{prefix}.attn.{weight_name}"), cols, inner)?;
        let codes = self
            .last_activation
            .take()
            .ok_or_else(|| EngineError::Shape {
                what: "attention activation".to_owned(),
                detail: "the projection backend has no quantised activation".to_owned(),
            })?;
        self.upload_weight(&weight.0, &weight.1)
            .map_err(|detail| EngineError::Shape {
                what: "attention weight upload".to_owned(),
                detail,
            })?;
        self.stats.host_bytes = self
            .stats
            .host_bytes
            .saturating_add(codes.0.len() as u64 + codes.1.len() as u64);
        self.upload_activation(&codes.0, &codes.1)
            .map_err(|detail| EngineError::Shape {
                what: "attention activation upload".to_owned(),
                detail,
            })?;
        self.stats.staging_us += started.elapsed().as_micros() as u64;
        let params = Mxfp8DenseParams::new(
            self.buffers.a_codes,
            self.buffers.a_scales,
            self.buffers.w,
            self.buffers.w_scales,
            self.buffers.out,
            u32::try_from(rows).map_err(|_| EngineError::Shape {
                what: "attention rows".to_owned(),
                detail: format!("{rows} does not fit the kernel's u32 rows"),
            })?,
            u32::try_from(cols).map_err(|_| EngineError::Shape {
                what: "attention cols".to_owned(),
                detail: format!("{cols} does not fit the kernel's u32 cols"),
            })?,
            u32::try_from(inner).map_err(|_| EngineError::Shape {
                what: "attention inner".to_owned(),
                detail: format!("{inner} does not fit the kernel's u32 inner"),
            })?,
            0,
            u32::try_from(layer).unwrap_or(0),
            0,
        );
        let launched = std::time::Instant::now();
        mxfp8_dense_launch::launch(&self.cuda, self.kernels.dense, &params).map_err(|detail| {
            EngineError::Shape {
                what: "attention dense launch".to_owned(),
                detail,
            }
        })?;
        let read = self
            .read_out(rows * cols * 2)
            .map_err(|detail| EngineError::Shape {
                what: "attention dense readback".to_owned(),
                detail,
            })?;
        self.stats.host_bytes = self
            .stats
            .host_bytes
            .saturating_add((rows * cols * 2) as u64);
        self.stats.device_us += launched.elapsed().as_micros() as u64;
        self.stats.launches += 1;
        self.stats.layers = layer + 1;
        if layer == 0 && weight_name == "wq_b.weight" && self.diag_wq_b_kernel.is_none() {
            let reference = sparkflash_kernels::mxfp8_ref::dense_gemm(
                &codes.0, &codes.1, &weight.0, &weight.1, rows, cols, inner,
            );
            let reference: Vec<f32> = reference
                .iter()
                .map(|bits| f32::from_bits((*bits as u32) << 16))
                .collect();
            let mut mismatches = 0usize;
            let mut max_abs = 0f32;
            for (device, host) in read.iter().zip(reference.iter()) {
                let diff = (device - host).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                if sparkflash_kernels::m4_quant::bf16_bits(*device)
                    != sparkflash_kernels::m4_quant::bf16_bits(*host)
                {
                    mismatches += 1;
                }
            }
            self.diag_wq_b_kernel = Some((mismatches, reference.len(), max_abs));
        }
        let _ = config;
        Ok(read)
    }
}

fn verified_cubin(dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    let pin = sparkflash_kernels::baseline::pin(name)?
        .ok_or_else(|| format!("kernel {name} is not pinned in the baseline"))?;
    let path = sparkflash_device::cubins::cubin_path(dir, name);
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    sparkflash_device::cubins::verify_bytes(&bytes, &pin)?;
    Ok(bytes)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
        .collect()
}

fn read_packed(
    src: &dyn WeightSource,
    name: &str,
    cols: usize,
    inner: usize,
) -> Result<(Vec<u8>, Vec<u8>), EngineError> {
    let raw = src.read(name).map_err(|error| EngineError::Shape {
        what: name.to_owned(),
        detail: error.to_string(),
    })?;
    if raw.len() != cols * inner {
        return Err(EngineError::Shape {
            what: name.to_owned(),
            detail: format!("expected {} E4M3 bytes, got {}", cols * inner, raw.len()),
        });
    }
    let scale_name = format!("{}.scale", name.trim_end_matches(".weight"));
    let scales = src.read(&scale_name).map_err(|error| EngineError::Shape {
        what: scale_name.clone(),
        detail: error.to_string(),
    })?;
    let expected_scales = cols.div_ceil(32) * inner.div_ceil(32);
    if scales.len() != expected_scales {
        return Err(EngineError::Shape {
            what: scale_name,
            detail: format!(
                "expected {expected_scales} E8M0 bytes, got {}",
                scales.len()
            ),
        });
    }
    let mut tiled = vec![0u8; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            tiled[b12x_weight_index(col, k, inner)] = raw[col * inner + k];
        }
    }
    Ok((tiled, scales))
}

fn quantise(values: &[f32], what: &str) -> Result<(Vec<u8>, Vec<u8>), EngineError> {
    let q = quantize_activation_fp8(values).map_err(|error| EngineError::Shape {
        what: what.to_owned(),
        detail: error.to_string(),
    })?;
    Ok((q.codes().to_vec(), q.scales().to_vec()))
}

/// Quantises a W8A8 activation the way the reference does: the value is a bf16 module output,
/// so the f32 host value is narrowed to bf16 before the quantiser sees it (F5).
///
/// The narrowing is idempotent on values that are already bf16-exact (a device GEMM whose
/// output was written as bf16 and read back as f32), so it is safe to apply at every W8A8
/// activation seam. wo_a's MXFP8 comparison arm is deliberately not narrowed: it is not the
/// native path, and task-6's native wo_a arm already narrows its activation.
fn quantise_native_activation(
    values: &[f32],
    what: &str,
) -> Result<(Vec<u8>, Vec<u8>), EngineError> {
    let narrowed: Vec<f32> = values
        .iter()
        .map(|value| sparkflash_engine::quant::bf16_round(*value))
        .collect();
    quantise(&narrowed, what)
}

fn rows_of(values: &[f32], width: usize, what: &str) -> Result<usize, EngineError> {
    if width == 0 || !values.len().is_multiple_of(width) {
        return Err(EngineError::Shape {
            what: what.to_owned(),
            detail: format!("{} values do not divide into rows of {width}", values.len()),
        });
    }
    Ok(values.len() / width)
}

impl Drop for AttentionDeviceBackend {
    fn drop(&mut self) {
        for ptr in &self.buffers.ptrs {
            self.cuda.free_buffers(&[ptr]);
        }
        for ptr in &self.mla.ptrs {
            self.cuda.free_buffers(&[ptr]);
        }
    }
}

impl AttentionDeviceBackend {
    /// Replaces the positions the in-chain attention twin compares at. The set is sorted and
    /// deduplicated so the armed-position report is stable.
    pub fn set_twin_positions(&mut self, positions: Vec<usize>) {
        let mut positions = positions;
        positions.sort_unstable();
        positions.dedup();
        self.twin_positions = positions;
    }

    /// The positions the in-chain attention twin compares at.
    #[must_use]
    pub fn twin_positions(&self) -> &[usize] {
        &self.twin_positions
    }

    /// Whether the in-chain twin compares at `position`.
    fn twin_armed(&self, position: usize) -> bool {
        self.twin_positions.contains(&position)
    }

    /// Skips the device attention body from `position` on and lets the engine's host loop run.
    /// This is an A/B switch for localising a divergence, not a production path.
    pub fn set_host_attention_from(&mut self, position: usize) {
        self.host_attention_from = Some(position);
    }

    /// One position-causal format-2 attention through the sequence entry. The engine passes the
    /// decode step's ring KV: window rows in combined[..window_width] (ring slots, oldest first,
    /// with NO_INDEX padding when the ring is not full) followed by compressed rows in the shared
    /// index space (window_width + j). The kernel wants the window rows oldest-first and the
    /// compressed candidates as even absolute positions, so the local compressed index j becomes
    /// 2*j here; a stray future id is masked by the kernel.
    #[allow(clippy::too_many_arguments)]
    fn attention_sequence(
        &mut self,
        layer: usize,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        combined: &[usize],
        heads: usize,
        head_dim: usize,
        seqlen: usize,
        window_width: usize,
        scale: f32,
        start_pos: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = scale;
        if seqlen != 1
            || window_width == 0
            || combined.len() < window_width
            || q.len() != heads * head_dim
            || out.len() < heads * head_dim
            || sink.len() != heads
            || !kv.len().is_multiple_of(head_dim)
            || kv.len() / head_dim < window_width
        {
            return Ok(false);
        }
        let kv_rows = kv.len() / head_dim;
        let window_ids: Vec<usize> = combined[..window_width]
            .iter()
            .copied()
            .filter(|id| *id < window_width && *id < kv_rows)
            .collect();
        let window_count = window_ids.len();
        if window_count == 0 || window_count > window_width {
            return Ok(false);
        }
        let compressed_count = kv_rows - window_width;
        // The compressed store groups ratio consecutive token positions per row. The engine mixes
        // ratio-1 and ratio-2 layers, so recover it from the candidate count: at seqlen 1 the
        // ratio-2 count is (start_pos + 1) / 2, and a larger count can only be ratio 1.
        let ratio = if compressed_count == 0 || compressed_count > start_pos.div_ceil(2) {
            1
        } else {
            2
        };
        let pool_rows = if compressed_count == 0 {
            0
        } else {
            (compressed_count - 1) * ratio + 1
        };
        let mut selected: Vec<u32> = Vec::new();
        for id in &combined[window_width..] {
            if *id < window_width || *id >= kv_rows {
                continue;
            }
            let local = id - window_width;
            let absolute = u32::try_from(local * ratio).map_err(|_| EngineError::Shape {
                what: "csa2 sequence selected id".to_owned(),
                detail: format!("compressed index {local} does not fit a u32"),
            })?;
            selected.push(absolute);
        }
        if selected.len() > MAX_SELECTED / 4
            || window_count * 512 > MAX_WINDOW_CODES
            || window_count * 16 > MAX_WINDOW_SCALES
            || pool_rows * 256 > MAX_KV_FP4
            || pool_rows * 32 > MAX_KV_SCALES
        {
            return Ok(false);
        }
        if compressed_count == 0 && !selected.is_empty() {
            return Ok(false);
        }
        let mut window_codes = Vec::with_capacity(window_count * 512);
        let mut window_scales = Vec::with_capacity(window_count * 16);
        for id in &window_ids {
            let row = &kv[*id * head_dim..(*id + 1) * head_dim];
            let window = engine_window_row(row).map_err(|detail| EngineError::Shape {
                what: "csa2 sequence window row".to_owned(),
                detail,
            })?;
            if window.codes.len() != 512 || window.scales.len() != 16 {
                return Ok(false);
            }
            window_codes.extend_from_slice(&window.codes);
            window_scales.extend_from_slice(&window.scales);
        }
        if layer == 0 && start_pos == 0 && self.diag_window_codes.is_none() {
            self.diag_window_codes = Some(window_codes.clone());
        }
        // The kernel indexes the compressed pool by the selected absolute position itself
        // (kv_fp4 + candidate * LATENT/2), so a position j is stored at row 2*j; odd rows stay zero.
        let mut compressed_codes = vec![0u8; pool_rows * 256];
        let mut compressed_scales = vec![0u8; pool_rows * 32];
        for row in 0..compressed_count {
            let values = &kv[(window_width + row) * head_dim..(window_width + row + 1) * head_dim];
            let entry = quantize_main_kv(values).map_err(|error| EngineError::Shape {
                what: "csa2 sequence compressed row".to_owned(),
                detail: error.to_string(),
            })?;
            if entry.data.len() != 256 || entry.scales.len() != 32 {
                return Ok(false);
            }
            let pool_row = row * ratio;
            compressed_codes[pool_row * 256..(pool_row + 1) * 256].copy_from_slice(&entry.data);
            compressed_scales[pool_row * 32..(pool_row + 1) * 32].copy_from_slice(&entry.scales);
        }
        let q_bytes: Vec<u8> = q
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect();
        let sink_bytes = f32_bytes(sink);
        let selected_bytes: Vec<u8> = selected
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        if layer == 0 && self.twin_armed(start_pos) {
            // One line per armed position, from layer 0 only: the window ids in staged order and
            // the first/last row checksums are the eviction contract. A ring-order bug shows up
            // here as a row set that is not the 128 tokens at or behind the query, before the twin
            // body comparison can be blamed.
            let first_window = checksum_bytes(&window_codes[..window_codes.len().min(512)]);
            let last_start = window_codes.len().saturating_sub(512);
            let last_window = checksum_bytes(&window_codes[last_start..]);
            // The ring is full from the first wrap on, so the staged set must be every slot
            // exactly once. A short list or a duplicate is an eviction-order defect and is a
            // refusal to launch, not a warning: a partial window silently changes the softmax.
            let mut sorted_slots = window_ids.clone();
            sorted_slots.sort_unstable();
            sorted_slots.dedup();
            let full_ring = start_pos + 1 >= window_width;
            let eviction_ok =
                !full_ring || (window_count == window_width && sorted_slots.len() == window_width);
            println!(
                "sequence attention eviction check: position={} window_width={} window_count={} unique_slots={} full_ring={} ok={}",
                start_pos,
                window_width,
                window_count,
                sorted_slots.len(),
                full_ring,
                eviction_ok
            );
            println!(
                "sequence attention boundary: position={} window_count={} window_ids_head={:?} window_ids_tail={:?} first_window_codes_checksum={:016x} last_window_codes_checksum={:016x} compressed_count={} ratio={} selected_count={} selected_head={:?} pool_rows={}",
                start_pos,
                window_count,
                &window_ids[..window_ids.len().min(4)],
                &window_ids[window_ids.len().saturating_sub(4)..],
                first_window,
                last_window,
                compressed_count,
                ratio,
                selected.len(),
                &selected[..selected.len().min(4)],
                pool_rows
            );
        }
        if self.twin_armed(start_pos) {
            self.operand_traces.push(AttentionOperandTrace {
                position: start_pos,
                layer,
                q_values: q.to_vec(),
                q_first: q.first().copied().unwrap_or(0.0),
                q_last: q.last().copied().unwrap_or(0.0),
                q_checksum: checksum_f32(q),
                sink_first: sink.first().copied().unwrap_or(0.0),
                sink_last: sink.last().copied().unwrap_or(0.0),
                sink_checksum: checksum_f32(sink),
                window_ids: window_ids.clone(),
                selected: selected.iter().map(|value| *value as usize).collect(),
                compressed_count,
                ratio,
                window_codes_len: window_codes.len(),
                window_first_checksum: checksum_bytes(&window_codes[..window_codes.len().min(512)]),
                window_last_checksum: if window_codes.len() >= 512 {
                    checksum_bytes(&window_codes[window_codes.len() - 512..])
                } else {
                    0
                },
                compressed_codes_len: compressed_codes.len(),
                window_codes: window_codes.clone(),
                window_scales: window_scales.clone(),
                compressed_codes: compressed_codes.clone(),
                compressed_scales: compressed_scales.clone(),
                sink_values: sink.to_vec(),
            });
        }
        let started = std::time::Instant::now();
        self.cuda
            .copy_htod(&self.mla.q, &q_bytes)
            .map_err(|detail| EngineError::Shape {
                what: "csa2 sequence q upload".to_owned(),
                detail,
            })?;
        self.cuda
            .copy_htod(&self.mla.window_codes, &window_codes)
            .map_err(|detail| EngineError::Shape {
                what: "csa2 sequence window codes upload".to_owned(),
                detail,
            })?;
        self.cuda
            .copy_htod(&self.mla.window_scales, &window_scales)
            .map_err(|detail| EngineError::Shape {
                what: "csa2 sequence window scales upload".to_owned(),
                detail,
            })?;
        if !compressed_codes.is_empty() {
            self.cuda
                .copy_htod(&self.mla.kv_fp4, &compressed_codes)
                .map_err(|detail| EngineError::Shape {
                    what: "csa2 sequence compressed codes upload".to_owned(),
                    detail,
                })?;
            self.cuda
                .copy_htod(&self.mla.kv_scales, &compressed_scales)
                .map_err(|detail| EngineError::Shape {
                    what: "csa2 sequence compressed scales upload".to_owned(),
                    detail,
                })?;
        }
        self.cuda
            .copy_htod(&self.mla.selected, &selected_bytes)
            .map_err(|detail| EngineError::Shape {
                what: "csa2 sequence selected upload".to_owned(),
                detail,
            })?;
        self.cuda
            .copy_htod(&self.mla.sink, &sink_bytes)
            .map_err(|detail| EngineError::Shape {
                what: "csa2 sequence sink upload".to_owned(),
                detail,
            })?;
        let params = FlashMlaSeqParams::new(
            self.mla.q,
            self.mla.kv_fp4,
            self.mla.kv_scales,
            self.mla.window_codes,
            self.mla.window_scales,
            FORMAT_CSA_UNION,
            window_count as u64,
            self.mla.selected,
            self.mla.index_key,
            self.mla.index_query,
            self.mla.sink,
            self.mla.block_scores,
            self.mla.block_best,
            self.mla.out,
            start_pos as u64,
            u32::try_from(selected.len()).map_err(|_| EngineError::Shape {
                what: "csa2 sequence selected count".to_owned(),
                detail: "selected list does not fit a u32".to_owned(),
            })?,
            0,
            u32::try_from(layer).map_err(|_| EngineError::Shape {
                what: "csa2 sequence layer".to_owned(),
                detail: "layer does not fit a u32".to_owned(),
            })?,
            0,
        );
        flash_mla_csa2_seq_launch::launch(&self.cuda, self.kernels.flash_seq, &params).map_err(
            |detail| EngineError::Shape {
                what: "csa2 sequence attention launch".to_owned(),
                detail,
            },
        )?;
        let mut raw = vec![0u8; heads * head_dim * 2];
        self.cuda
            .copy_dtoh(&mut raw, &self.mla.out)
            .map_err(|detail| EngineError::Shape {
                what: "csa2 sequence readback".to_owned(),
                detail,
            })?;
        for (slot, value) in decode_bf16(&raw).iter().enumerate() {
            out[slot] = *value;
        }
        self.attention_twin(
            layer,
            q,
            kv,
            sink,
            combined,
            heads,
            head_dim,
            window_width,
            scale,
            start_pos,
            out,
        )?;
        self.stats.host_bytes = self.stats.host_bytes.saturating_add(
            (q_bytes.len()
                + window_codes.len()
                + window_scales.len()
                + sink_bytes.len()
                + selected_bytes.len()
                + compressed_codes.len()
                + compressed_scales.len()
                + raw.len()) as u64,
        );
        self.stats.attention_us += started.elapsed().as_micros() as u64;
        self.stats.attention_launches += 1;
        if let Some(count) = self.body_by_layer.get_mut(layer) {
            *count += 1;
        }
        Ok(true)
    }
}

impl AttentionDeviceBackend {
    /// In-chain body twin: run the host sparse-attention reference on the same q, kv, sink and
    /// combined ids the device body was staged with, and compare the BF16 rounding of both
    /// outputs. It reports a per-head-block mismatch count so a p>0 divergence can be localised
    /// to the indexer/window/compressed-row contract instead of only the final token.
    #[allow(clippy::too_many_arguments)]
    fn attention_twin(
        &self,
        layer: usize,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        combined: &[usize],
        heads: usize,
        head_dim: usize,
        window_width: usize,
        scale: f32,
        start_pos: usize,
        device_out: &[f32],
    ) -> Result<(), EngineError> {
        // The armed set selects the compared positions. The ring's first two wrap steps are
        // armed by default, so the twin cannot be blind where the window first rotates.
        if !self.twin_armed(start_pos) {
            return Ok(());
        }
        let width = heads * head_dim;
        // The host body must see the bytes the device body saw, not the unquantised checkpoint
        // activations: BF16 q, one E4M3+E8M0 window row per ring slot, and the FP4 round-trip of
        // each compressed row. Otherwise a quantisation difference masquerades as a mask/indexer
        // bug.
        let q_host: Vec<f32> = q
            .iter()
            .map(|value| f32::from_bits((bf16_bits(*value) as u32) << 16))
            .collect();
        let kv_rows = kv.len() / head_dim;
        let mut host_kv = vec![0f32; kv.len()];
        for slot in 0..window_width.min(kv_rows) {
            let row = &kv[slot * head_dim..(slot + 1) * head_dim];
            let window = engine_window_row(row).map_err(|detail| EngineError::Shape {
                what: "attention twin window row".to_owned(),
                detail,
            })?;
            for index in 0..head_dim {
                host_kv[slot * head_dim + index] = sparkflash_kernels::mla_ref::window_kv_value(
                    &window.codes,
                    &window.scales,
                    0,
                    index,
                );
            }
        }
        for row in window_width..kv_rows {
            let values = &kv[row * head_dim..(row + 1) * head_dim];
            let entry = quantize_main_kv(values).map_err(|error| EngineError::Shape {
                what: "attention twin compressed row".to_owned(),
                detail: error.to_string(),
            })?;
            let mut staged = vec![0f32; head_dim];
            dequantize_main_kv(&entry, &mut staged).map_err(|error| EngineError::Shape {
                what: "attention twin compressed dequant".to_owned(),
                detail: error.to_string(),
            })?;
            host_kv[row * head_dim..(row + 1) * head_dim].copy_from_slice(&staged);
        }
        let mut host_out = vec![0f32; width];
        sparse_attention(
            &q_host,
            &host_kv,
            sink,
            combined,
            heads,
            head_dim,
            scale,
            &mut host_out,
        )?;
        let blocks = 8.min(heads);
        let mut block_mismatches = vec![0usize; blocks];
        let mut mismatches = 0usize;
        let mut exact_heads = 0usize;
        let mut max_abs_diff = 0f32;
        for head in 0..heads {
            let mut head_mismatches = 0usize;
            for slot in head * head_dim..(head + 1) * head_dim {
                let diff = (device_out[slot] - host_out[slot]).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
                if bf16_bits(device_out[slot]) != bf16_bits(host_out[slot]) {
                    head_mismatches += 1;
                }
            }
            mismatches += head_mismatches;
            if head_mismatches == 0 {
                exact_heads += 1;
            }
            let block = head * blocks / heads;
            block_mismatches[block] += head_mismatches;
        }
        let fractions: Vec<String> = block_mismatches
            .iter()
            .map(|count| {
                let slots = (heads / blocks).max(1) * head_dim;
                format!("{:.6}", 1.0 - *count as f64 / slots as f64)
            })
            .collect();
        println!(
            "sequence attention twin: position={} layer={} heads={} head_dim={} bf16_mismatches={}/{} exact_fraction={:.6} exact_heads={}/{} max_abs_diff={:.6e} per_head_block_mismatches={:?} per_head_block_exact_fraction=[{}]",
            start_pos,
            layer,
            heads,
            head_dim,
            mismatches,
            width,
            1.0 - mismatches as f64 / width as f64,
            exact_heads,
            heads,
            max_abs_diff,
            block_mismatches,
            fractions.join(",")
        );
        Ok(())
    }
}

impl AttentionProjectionBackend for AttentionDeviceBackend {
    fn wq_a(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let _ = limit;
        self.last_activation = Some(quantise_native_activation(x, "wq_a activation")?);
        let out = self.dense(
            config,
            layer,
            prefix,
            src,
            "wq_a.weight",
            config.q_lora_rank,
            config.dim,
            rows_of(x, config.dim, "wq_a input")?,
        )?;
        if layer == 0 && self.diag_latent_raw.is_none() {
            self.diag_latent_raw = Some(out.clone());
        }
        Ok(out)
    }

    fn wq_b(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let _ = limit;
        let rank = config.q_lora_rank;
        if layer == 0 && self.diag_wq_b_input.is_none() {
            self.diag_wq_b_input = Some(latent.to_vec());
        }
        self.last_activation = Some(quantise_native_activation(latent, "wq_b activation")?);
        let out = self.dense(
            config,
            layer,
            prefix,
            src,
            "wq_b.weight",
            config.n_heads * config.head_dim,
            rank,
            rows_of(latent, rank, "wq_b input")?,
        )?;
        if layer == 0 && self.diag_q_raw.is_none() {
            self.diag_q_raw = Some(out.clone());
        }
        Ok(out)
    }

    /// The indexer's own `wq_b`: the same pinned B12x MXFP8 GEMM as the attention projections,
    /// run in row chunks that fit the shared output buffer (the buffer is sized for one projection,
    /// not for prefill rows x 4096). The activation is quantised the native way per row chunk, and
    /// GEMM rows are independent, so the chunking cannot change a value.
    fn indexer_wq_b(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let _ = limit;
        let inner = config.q_lora_rank;
        let cols = config
            .index_n_heads
            .checked_mul(config.index_head_dim)
            .ok_or_else(|| EngineError::Shape {
                what: "indexer wq_b geometry".to_owned(),
                detail: format!(
                    "{} index heads x {} head dim does not fit usize",
                    config.index_n_heads, config.index_head_dim
                ),
            })?;
        let rows = rows_of(latent, inner, "indexer wq_b input")?;
        let out = indexer_wq_b_chunked(rows, cols, MAX_OUT_BYTES, |start, count| {
            let window = &latent[start * inner..(start + count) * inner];
            self.last_activation = Some(quantise_native_activation(
                window,
                "indexer wq_b activation",
            )?);
            self.dense(
                config,
                layer,
                prefix,
                src,
                "indexer.wq_b.weight",
                cols,
                inner,
                count,
            )
        })?;
        Ok(Some(out))
    }

    fn wkv(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let _ = limit;
        self.last_activation = Some(quantise_native_activation(x, "wkv activation")?);
        let out = self.dense(
            config,
            layer,
            prefix,
            src,
            "wkv.weight",
            config.head_dim,
            config.dim,
            rows_of(x, config.dim, "wkv input")?,
        )?;
        if layer == 0 && self.diag_wkv_raw.is_none() {
            self.diag_wkv_raw = Some(out.clone());
        }
        Ok(out)
    }

    fn wo_a(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        wide: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        match self.wo_a_mode {
            WoAMode::Native => self.wo_a_native(config, prefix, wide, src, limit),
            WoAMode::Mxfp8 => self.wo_a_mxfp8(config, prefix, wide, src, limit),
        }
    }

    fn wo_b(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        grouped: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let _ = limit;
        let grouped_width = config.o_groups * config.o_lora_rank;
        let rows = rows_of(grouped, grouped_width, "wo_b input")?;
        self.last_activation = Some(quantise_native_activation(grouped, "wo_b activation")?);
        let layer = self.stats.layers.saturating_sub(1);
        self.dense(
            config,
            layer,
            prefix,
            src,
            "wo_b.weight",
            config.dim,
            grouped_width,
            rows,
        )
    }

    fn attention(
        &mut self,
        layer: usize,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        combined: &[usize],
        heads: usize,
        head_dim: usize,
        seqlen: usize,
        window_width: usize,
        compressed_stride: usize,
        scale: f32,
        start_pos: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        if self
            .host_attention_from
            .is_some_and(|from| start_pos >= from)
        {
            // Deliberate fallback: the engine's host attention loop consumes the same q/kv/ids, so
            // a clean run past `from` attributes the divergence to the device attention path.
            return Ok(false);
        }
        if start_pos > 0 {
            return self.attention_sequence(
                layer,
                q,
                kv,
                sink,
                combined,
                heads,
                head_dim,
                seqlen,
                window_width,
                scale,
                start_pos,
                out,
            );
        }
        // c4a-2 wires the real format-2 launch at position 0 only: one query row, one window row,
        // and either no compressed candidate (ratio > 1) or the single reachable position 0
        // (ratio == 1). Anything else answers false and the host reference loop runs.
        let _ = scale;
        let compressed = compressed_stride;
        let expected_rows = 1 + compressed;
        if start_pos != 0
            || seqlen != 1
            || window_width != 1
            || compressed > 1
            || combined.len() != expected_rows
            || combined.first().copied() != Some(0)
            || q.len() != heads * head_dim
            || kv.len() != expected_rows * head_dim
            || sink.len() != heads
            || out.len() < heads * head_dim
        {
            return Ok(false);
        }
        if compressed == 1 && combined.get(1).copied() != Some(1) {
            // The compressed row must be the first compressed row, stored after the one window row.
            return Ok(false);
        }
        let started = std::time::Instant::now();
        let q_bytes: Vec<u8> = q
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect();
        let window = engine_window_row(&kv[..head_dim]).map_err(|detail| EngineError::Shape {
            what: "m14 window row".to_owned(),
            detail,
        })?;
        if start_pos <= 2 {
            let window_checksum = checksum_bytes(&window.codes);
            self.operand_traces.push(AttentionOperandTrace {
                position: 0,
                layer,
                q_values: q.to_vec(),
                q_first: q.first().copied().unwrap_or(0.0),
                q_last: q.last().copied().unwrap_or(0.0),
                q_checksum: checksum_f32(q),
                sink_first: sink.first().copied().unwrap_or(0.0),
                sink_last: sink.last().copied().unwrap_or(0.0),
                sink_checksum: checksum_f32(sink),
                window_ids: combined[..window_width.min(combined.len())].to_vec(),
                selected: combined[window_width.min(combined.len())..].to_vec(),
                compressed_count: compressed,
                ratio: 1,
                window_codes_len: window.codes.len(),
                window_first_checksum: window_checksum,
                window_last_checksum: window_checksum,
                compressed_codes_len: if compressed == 1 { 256 } else { 0 },
                window_codes: window.codes.clone(),
                window_scales: window.scales.clone(),
                compressed_codes: if compressed == 1 {
                    quantize_main_kv(&kv[head_dim..2 * head_dim])
                        .map_err(|error| EngineError::Shape {
                            what: "m14 compressed row".to_owned(),
                            detail: error.to_string(),
                        })?
                        .data
                        .to_vec()
                } else {
                    Vec::new()
                },
                compressed_scales: if compressed == 1 {
                    quantize_main_kv(&kv[head_dim..2 * head_dim])
                        .map_err(|error| EngineError::Shape {
                            what: "m14 compressed row".to_owned(),
                            detail: error.to_string(),
                        })?
                        .scales
                        .to_vec()
                } else {
                    Vec::new()
                },
                sink_values: sink.to_vec(),
            });
        }
        if layer == 0 && self.diag_window_codes.is_none() {
            self.diag_window_codes = Some(window.codes.clone());
        }
        let sink_bytes = f32_bytes(sink);
        self.cuda
            .copy_htod(&self.mla.q, &q_bytes)
            .map_err(|detail| EngineError::Shape {
                what: "m14 q upload".to_owned(),
                detail,
            })?;
        self.cuda
            .copy_htod(&self.mla.window_codes, &window.codes)
            .map_err(|detail| EngineError::Shape {
                what: "m14 window code upload".to_owned(),
                detail,
            })?;
        self.cuda
            .copy_htod(&self.mla.window_scales, &window.scales)
            .map_err(|detail| EngineError::Shape {
                what: "m14 window scale upload".to_owned(),
                detail,
            })?;
        self.cuda
            .copy_htod(&self.mla.sink, &sink_bytes)
            .map_err(|detail| EngineError::Shape {
                what: "m14 sink upload".to_owned(),
                detail,
            })?;
        if compressed == 1 {
            // The host compressed cache is already FP4 round-tripped, so re-encoding its row is the
            // same bytes the host sparse path dequantised. Position 0 is the only reachable
            // compressed position in this window; the selected list therefore holds exactly {0}.
            let entry = quantize_main_kv(&kv[head_dim..2 * head_dim]).map_err(|error| {
                EngineError::Shape {
                    what: "m14 compressed row".to_owned(),
                    detail: error.to_string(),
                }
            })?;
            self.cuda
                .copy_htod(&self.mla.kv_fp4, &entry.data)
                .map_err(|detail| EngineError::Shape {
                    what: "m14 compressed code upload".to_owned(),
                    detail,
                })?;
            self.cuda
                .copy_htod(&self.mla.kv_scales, &entry.scales)
                .map_err(|detail| EngineError::Shape {
                    what: "m14 compressed scale upload".to_owned(),
                    detail,
                })?;
            self.cuda
                .copy_htod(&self.mla.selected, &0u32.to_le_bytes())
                .map_err(|detail| EngineError::Shape {
                    what: "m14 selected upload".to_owned(),
                    detail,
                })?;
        }
        let params = flash_mla_params(
            self.mla.q.0,
            self.mla.kv_fp4.0,
            self.mla.kv_scales.0,
            self.mla.window_codes.0,
            self.mla.window_scales.0,
            FORMAT_CSA_UNION,
            1,
            self.mla.selected.0,
            self.mla.index_key.0,
            self.mla.index_query.0,
            self.mla.sink.0,
            self.mla.block_scores.0,
            self.mla.block_best.0,
            self.mla.out.0,
            0,
            compressed,
            0,
            layer,
            0,
        );
        let heads_u32 = u32::try_from(heads).map_err(|_| EngineError::Shape {
            what: "m14 heads".to_owned(),
            detail: format!("{heads} does not fit a u32 grid"),
        })?;
        launch_flash_mla(
            &self.cuda,
            self.kernels.flash,
            (heads_u32, 1, 1),
            (256, 1, 1),
            &params,
        )
        .map_err(|detail| EngineError::Shape {
            what: "m14 flash-MLA launch".to_owned(),
            detail,
        })?;
        let mut raw = vec![0u8; heads * head_dim * 2];
        self.cuda
            .copy_dtoh(&mut raw, &self.mla.out)
            .map_err(|detail| EngineError::Shape {
                what: "m14 readback".to_owned(),
                detail,
            })?;
        for (slot, value) in decode_bf16(&raw).iter().enumerate() {
            out[slot] = *value;
        }
        self.attention_twin(
            layer, q, kv, sink, combined, heads, head_dim, 1, scale, start_pos, out,
        )?;
        let compressed_bytes = if compressed == 1 { 256 + 32 + 4 } else { 0 };
        self.stats.host_bytes = self.stats.host_bytes.saturating_add(
            (q_bytes.len()
                + window.codes.len()
                + window.scales.len()
                + sink_bytes.len()
                + raw.len()
                + compressed_bytes) as u64,
        );
        self.stats.attention_us += started.elapsed().as_micros() as u64;
        self.stats.attention_launches += 1;
        if let Some(count) = self.body_by_layer.get_mut(layer) {
            *count += 1;
        }
        Ok(true)
    }
}

/// The wo_a implementations, kept out of the trait impl so each trait method stays one call.
impl AttentionDeviceBackend {
    /// The native wo_a path: the BF16 activation the reference's in-place unit round leaves, the
    /// checkpoint's FP8 weight codes dequantised inside the GEMM, and one BF16 output round.
    fn wo_a_native(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        wide: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let _ = limit;
        let layer = self.stats.layers.saturating_sub(1);
        let wide_row = config.n_heads * config.head_dim;
        let rows = rows_of(wide, wide_row, "wo_a wide")?;
        if rows != 1 {
            return Err(EngineError::Shape {
                what: "wo_a".to_owned(),
                detail: format!("the grouped kernel is one position per launch, got {rows}"),
            });
        }
        let started = std::time::Instant::now();
        let weight = self.packed_weight(
            src,
            &format!("{prefix}.attn.wo_a.weight"),
            config.o_groups * config.o_lora_rank,
            wide_row / config.o_groups,
        )?;
        self.upload_weight(&weight.0, &weight.1)
            .map_err(|detail| EngineError::Shape {
                what: "wo_a weight upload".to_owned(),
                detail,
            })?;
        // The reference applies the inverse rotation into the BF16 body output and reads the 512-wide
        // head slice back from there, so the GEMM operand is that rounded value, not the f32
        // rotation result.
        let activation: Vec<u16> = wide.iter().map(|value| bf16_bits(*value)).collect();
        self.upload_activation_bf16(&activation)
            .map_err(|detail| EngineError::Shape {
                what: "wo_a activation upload".to_owned(),
                detail,
            })?;
        self.stats.host_bytes = self
            .stats
            .host_bytes
            .saturating_add((activation.len() * 2) as u64);
        self.stats.staging_us += started.elapsed().as_micros() as u64;
        let params = Bf16aMxfp8wWoAParams::new(
            self.buffers.a_bf16,
            self.buffers.w,
            self.buffers.w_scales,
            self.buffers.out,
            u32::try_from(config.o_groups).unwrap_or(0),
            0,
            u32::try_from(layer).unwrap_or(0),
            0,
        );
        let launched = std::time::Instant::now();
        bf16a_mxfp8w_wo_a_launch::launch(&self.cuda, self.kernels.wo_a_native, &params).map_err(
            |detail| EngineError::Shape {
                what: "wo_a native launch".to_owned(),
                detail,
            },
        )?;
        let grouped = self
            .read_out(config.o_groups * config.o_lora_rank * 2)
            .map_err(|detail| EngineError::Shape {
                what: "wo_a readback".to_owned(),
                detail,
            })?;
        self.stats.host_bytes = self
            .stats
            .host_bytes
            .saturating_add((config.o_groups * config.o_lora_rank * 2) as u64);
        self.stats.device_us += launched.elapsed().as_micros() as u64;
        self.stats.launches += 1;
        self.record_wo_a_comparison(
            config,
            layer,
            wide,
            &activation,
            &weight.0,
            &weight.1,
            &grouped,
        )?;
        Ok(grouped)
    }

    /// The chain's previous wo_a path: E4M3 activation, per-32 ue8m0 scales, MXFP8 codes.
    fn wo_a_mxfp8(
        &mut self,
        config: &ModelConfig,
        prefix: &str,
        wide: &[f32],
        src: &dyn WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, EngineError> {
        let _ = limit;
        let layer = self.stats.layers.saturating_sub(1);
        let wide_row = config.n_heads * config.head_dim;
        let rows = rows_of(wide, wide_row, "wo_a wide")?;
        if rows != 1 {
            return Err(EngineError::Shape {
                what: "wo_a".to_owned(),
                detail: format!("the grouped kernel is one position per launch, got {rows}"),
            });
        }
        self.last_activation = Some(quantise(wide, "wo_a activation")?);
        let packed_input = self
            .last_activation
            .take()
            .ok_or_else(|| EngineError::Shape {
                what: "wo_a activation".to_owned(),
                detail: "missing quantised activation".to_owned(),
            })?;
        let started = std::time::Instant::now();
        let weight = self.packed_weight(
            src,
            &format!("{prefix}.attn.wo_a.weight"),
            config.o_groups * config.o_lora_rank,
            wide_row / config.o_groups,
        )?;
        self.upload_weight(&weight.0, &weight.1)
            .map_err(|detail| EngineError::Shape {
                what: "wo_a weight upload".to_owned(),
                detail,
            })?;
        self.stats.host_bytes = self
            .stats
            .host_bytes
            .saturating_add(packed_input.0.len() as u64 + packed_input.1.len() as u64);
        self.upload_activation(&packed_input.0, &packed_input.1)
            .map_err(|detail| EngineError::Shape {
                what: "wo_a activation upload".to_owned(),
                detail,
            })?;
        self.stats.staging_us += started.elapsed().as_micros() as u64;
        let params = Mxfp8WoAParams::new(
            self.buffers.a_codes,
            self.buffers.a_scales,
            self.buffers.w,
            self.buffers.w_scales,
            self.buffers.out,
            u32::try_from(config.o_groups).unwrap_or(0),
            0,
            u32::try_from(layer).unwrap_or(0),
            0,
        );
        let launched = std::time::Instant::now();
        mxfp8_wo_a_launch::launch(&self.cuda, self.kernels.wo_a, &params).map_err(|detail| {
            EngineError::Shape {
                what: "wo_a launch".to_owned(),
                detail,
            }
        })?;
        let grouped = self
            .read_out(config.o_groups * config.o_lora_rank * 2)
            .map_err(|detail| EngineError::Shape {
                what: "wo_a readback".to_owned(),
                detail,
            })?;
        self.stats.host_bytes = self
            .stats
            .host_bytes
            .saturating_add((config.o_groups * config.o_lora_rank * 2) as u64);
        self.stats.device_us += launched.elapsed().as_micros() as u64;
        self.stats.launches += 1;
        self.record_wo_a_comparison(config, layer, wide, &[], &weight.0, &weight.1, &grouped)?;
        Ok(grouped)
    }

    /// Records one layer's wo_a comparison the first time that layer runs, and optionally dumps the
    /// captured operand and the device output so the comparison can be re-run offline.
    #[allow(clippy::too_many_arguments)]
    fn record_wo_a_comparison(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        wide: &[f32],
        activation: &[u16],
        w: &[u8],
        w_scales: &[u8],
        grouped: &[f32],
    ) -> Result<(), EngineError> {
        let armed = self.wo_a_diagnose || self.wo_a_operand_dump.is_some();
        if !armed || self.wo_a_compared.get(layer).copied().unwrap_or(false) {
            return Ok(());
        }
        let groups = config.o_groups;
        let inner = config.n_heads * config.head_dim / groups;
        let lo_rank = config.o_lora_rank;
        // The MXFP8 path has no BF16 activation buffer, so it builds the codes here.
        let rounded: Vec<u16> = if activation.is_empty() {
            wide.iter().map(|value| bf16_bits(*value)).collect()
        } else {
            activation.to_vec()
        };
        let native =
            bf16a_mxfp8w_ref::grouped_wo_a_bf16a(&rounded, w, w_scales, groups, inner, lo_rank);
        let device_vs_native = bf16a_mxfp8w_ref::compare_codes(grouped, &native);
        let mxfp8_vs_native = bf16a_mxfp8w_ref::compare_paths(
            wide, w, w_scales, groups, inner, lo_rank,
        )
        .map_err(|detail| EngineError::Shape {
            what: "wo_a comparison".to_owned(),
            detail,
        })?;
        let operand_checksum = checksum_f32(wide);
        if let Some(dir) = self.wo_a_operand_dump.clone() {
            std::fs::create_dir_all(&dir).map_err(|error| EngineError::Shape {
                what: "wo_a operand dump".to_owned(),
                detail: format!("{}: {error}", dir.display()),
            })?;
            let operand_path = dir.join(format!("wo_a_layer_{layer:02}_operand.f32"));
            std::fs::write(&operand_path, f32_bytes(wide)).map_err(|error| EngineError::Shape {
                what: "wo_a operand dump".to_owned(),
                detail: format!("{}: {error}", operand_path.display()),
            })?;
            let mut device_bytes = Vec::with_capacity(grouped.len() * 2);
            for value in grouped {
                device_bytes.extend_from_slice(&bf16_bits(*value).to_le_bytes());
            }
            let device_path = dir.join(format!("wo_a_layer_{layer:02}_device.bf16"));
            std::fs::write(&device_path, device_bytes).map_err(|error| EngineError::Shape {
                what: "wo_a operand dump".to_owned(),
                detail: format!("{}: {error}", device_path.display()),
            })?;
        }
        if layer < self.wo_a_compared.len() {
            self.wo_a_compared[layer] = true;
        }
        self.wo_a_comparisons.push(WoAComparison {
            layer,
            mode: self.wo_a_mode,
            device_vs_native,
            mxfp8_vs_native,
            operand_checksum,
        });
        Ok(())
    }
}

/// The ring slots the device should have staged for one reference record, in staged order.
///
/// A prefill chunk names token positions, and the ring maps a position to `position % window`; a
/// decode step already names ring slots in the reference's own order. The two layouts are why a
/// single ladder can be compared against both kinds of record.
fn audit_expected_slots(ids: &[usize], seqlen: usize, window: usize) -> Vec<usize> {
    if seqlen > 1 {
        ids.iter().map(|id| id % window).collect()
    } else {
        ids.to_vec()
    }
}

fn audit_byte_mismatches(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(a, b)| a != b)
        .count()
        + left.len().abs_diff(right.len())
}

fn audit_bf16_mismatches(expected: &[f32], actual: &[f32]) -> (usize, f32) {
    let mut mismatches = 0usize;
    let mut worst = 0f32;
    for (left, right) in expected.iter().zip(actual.iter()) {
        if bf16_bits(*left) != bf16_bits(*right) {
            mismatches += 1;
        }
        worst = worst.max((left - right).abs());
    }
    mismatches += expected.len().abs_diff(actual.len());
    (mismatches, worst)
}

/// Compares the staged codes of `rows` against `expected`, where each expected row is
/// `(codes, scales)` and the staged buffer stores `code_row`/`scale_row` bytes per row.
///
/// Returns `(rows, rows_exact, bytes_differing, bytes, scales_differing, worst_index,
/// worst_bytes_differing, worst_scales_differing)`: the aggregates for the summary line and the
/// worst row for the detail line that prints the actual bytes.
fn audit_row_codes(
    expected: &[(Vec<u8>, Vec<u8>)],
    codes: &[u8],
    scales: &[u8],
    code_row: usize,
    scale_row: usize,
) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
    let mut rows = 0usize;
    let mut exact = 0usize;
    let mut bytes_differing = 0usize;
    let mut bytes = 0usize;
    let mut scales_differing = 0usize;
    let mut worst_index = usize::MAX;
    let mut worst_bytes_differing = 0usize;
    let mut worst_scales_differing = 0usize;
    for (index, (expected_codes, expected_scales)) in expected.iter().enumerate() {
        let code_start = index * code_row;
        let scale_start = index * scale_row;
        if code_start + code_row > codes.len() || scale_start + scale_row > scales.len() {
            continue;
        }
        rows += 1;
        let staged_codes = &codes[code_start..code_start + code_row];
        let staged_scales = &scales[scale_start..scale_start + scale_row];
        let differing = audit_byte_mismatches(expected_codes, staged_codes);
        bytes_differing += differing;
        bytes += code_row;
        let differing_scales = audit_byte_mismatches(expected_scales, staged_scales);
        scales_differing += differing_scales;
        if differing + differing_scales > worst_bytes_differing + worst_scales_differing {
            worst_index = index;
            worst_bytes_differing = differing;
            worst_scales_differing = differing_scales;
        }
        if differing == 0 && differing_scales == 0 {
            exact += 1;
        }
    }
    (
        rows,
        exact,
        bytes_differing,
        bytes,
        scales_differing,
        worst_index,
        worst_bytes_differing,
        worst_scales_differing,
    )
}

/// The first `limit` byte disagreements between one expected row and the staged row, spelled out so
/// a single log line says which byte moved and by how much.
fn audit_row_disagreements(expected: &[u8], staged: &[u8], limit: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (offset, (left, right)) in expected.iter().zip(staged.iter()).enumerate() {
        if left != right {
            parts.push(format!(
                "offset={offset} expected=0x{left:02x} staged=0x{right:02x}"
            ));
            if parts.len() == limit {
                break;
            }
        }
    }
    if expected.len() != staged.len() {
        parts.push(format!(
            "lengths expected={} staged={}",
            expected.len(),
            staged.len()
        ));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}
/// The per-layer staged-operand audit the next window runs instead of a hand comparison.
///
/// The in-chain twin compares the device body against the host body on the operands the device
/// staged, so it is blind to a staging defect by construction. This audit compares the staged
/// operands against the REFERENCE's own attention inputs, captured by the same process in the
/// reference half of the run: the ring ids (rotation), the window row codes and scales, the
/// compressed candidate selection, its row codes and scales, the query after bf16 staging, and the
/// sink. Row content can differ a little because the reference hidden is f32 and the device hidden
/// is what the chain produced, so the interesting quantity is HOW MUCH of each row differs and
/// whether the ids and the selection agree at all: a wrong rotation or a dropped/duplicated row
/// shows up as ids_exact=false or as most of a row's bytes differing, not as a few.
#[must_use]
pub fn operand_audit_lines(
    cases: &std::collections::BTreeMap<usize, Vec<AttentionCaseRecord>>,
    traces: &[AttentionOperandTrace],
    window: usize,
) -> Vec<String> {
    // Bound the raw detail so one window's log stays readable: the first few mismatching layers
    // per position print the actual slots and bytes, the rest are counted in the summary.
    const DETAIL_LAYERS: usize = 6;
    const DETAIL_IDS: usize = 8;
    const DETAIL_BYTES: usize = 8;
    let mut lines: Vec<String> = Vec::new();
    if cases.is_empty() || window == 0 {
        return lines;
    }
    let mut by_key: HashMap<(usize, usize), &AttentionOperandTrace> = HashMap::new();
    for trace in traces {
        by_key.insert((trace.position, trace.layer), trace);
    }
    for (position, records) in cases {
        let mut layer_lines: Vec<String> = Vec::new();
        let mut layers = 0usize;
        let mut detail_layers = 0usize;
        let mut ids_exact_layers = 0usize;
        let mut first_ids_mismatch = None;
        let mut window_rows = 0usize;
        let mut window_rows_exact = 0usize;
        let mut window_bytes_differing = 0usize;
        let mut window_bytes = 0usize;
        let mut window_scales_differing = 0usize;
        let mut q_mismatches = 0usize;
        let mut q_bits = 0usize;
        let mut q_worst = 0f32;
        let mut sink_exact_layers = 0usize;
        let mut selection_exact_layers = 0usize;
        let mut compressed_rows = 0usize;
        let mut compressed_rows_exact = 0usize;
        let mut compressed_bytes_differing = 0usize;
        let mut compressed_bytes = 0usize;
        if traces.is_empty() {
            lines.push(format!(
                "sequence operand audit: position={position} no device trace: the twin was not armed there or that position ran the host attention fallback, so the captured reference frame has nothing to compare against"
            ));
            continue;
        }
        for record in records {
            let layer = record.layer;
            let Some(trace) = by_key.get(&(*position, layer)) else {
                continue;
            };
            layers += 1;
            let mut details: Vec<String> = Vec::new();
            let expected_slots = audit_expected_slots(&record.window_ids, record.seqlen, window);
            let ids_exact = expected_slots == trace.window_ids;
            if ids_exact {
                ids_exact_layers += 1;
            } else {
                let index = expected_slots
                    .iter()
                    .zip(trace.window_ids.iter())
                    .position(|(expected, staged)| expected != staged)
                    .unwrap_or(expected_slots.len().min(trace.window_ids.len()));
                if first_ids_mismatch.is_none() {
                    first_ids_mismatch = Some((
                        layer,
                        index,
                        expected_slots.get(index).copied().unwrap_or(usize::MAX),
                        trace.window_ids.get(index).copied().unwrap_or(usize::MAX),
                    ));
                }
                details.push(format!(
                    "ids first_mismatch_index={index} expected_slot={} staged_slot={} expected_head={:?} staged_head={:?} expected_tail={:?} staged_tail={:?} expected_count={} staged_count={}",
                    expected_slots.get(index).copied().unwrap_or(usize::MAX),
                    trace.window_ids.get(index).copied().unwrap_or(usize::MAX),
                    &expected_slots[..expected_slots.len().min(DETAIL_IDS)],
                    &trace.window_ids[..trace.window_ids.len().min(DETAIL_IDS)],
                    &expected_slots[expected_slots.len().saturating_sub(DETAIL_IDS)..],
                    &trace.window_ids[trace.window_ids.len().saturating_sub(DETAIL_IDS)..],
                    expected_slots.len(),
                    trace.window_ids.len()
                ));
            }
            let mut expected_window: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for id in &record.window_ids {
                let start = *id * record.head_dim;
                let end = start + record.head_dim;
                if end > record.window_kv.len() {
                    continue;
                }
                if let Ok(row) = engine_window_row(&record.window_kv[start..end]) {
                    expected_window.push((row.codes, row.scales));
                }
            }
            let (
                rows,
                exact,
                differing,
                total,
                scale_differing,
                worst_row,
                worst_bytes,
                worst_scales,
            ) = audit_row_codes(
                &expected_window,
                &trace.window_codes,
                &trace.window_scales,
                512,
                16,
            );
            window_rows += rows;
            window_rows_exact += exact;
            window_bytes_differing += differing;
            window_bytes += total;
            window_scales_differing += scale_differing;
            if exact != rows || scale_differing != 0 {
                let expected_codes: &[u8] = expected_window
                    .get(worst_row)
                    .map_or(&[], |(codes, _)| codes.as_slice());
                let expected_scales: &[u8] = expected_window
                    .get(worst_row)
                    .map_or(&[], |(_, scales)| scales.as_slice());
                let code_start = worst_row * 512;
                let scale_start = worst_row * 16;
                let staged_codes: &[u8] = if code_start + 512 <= trace.window_codes.len() {
                    &trace.window_codes[code_start..code_start + 512]
                } else {
                    &[]
                };
                let staged_scales: &[u8] = if scale_start + 16 <= trace.window_scales.len() {
                    &trace.window_scales[scale_start..scale_start + 16]
                } else {
                    &[]
                };
                details.push(format!(
                    "window_row row_index={worst_row} window_id={} bytes_differing={worst_bytes}/512 scales_differing={worst_scales}/16 disagreement=[{}] scale_disagreement=[{}]",
                    record.window_ids.get(worst_row).copied().unwrap_or(usize::MAX),
                    audit_row_disagreements(expected_codes, staged_codes, DETAIL_BYTES),
                    audit_row_disagreements(
                        expected_scales,
                        staged_scales,
                        DETAIL_BYTES / 2
                    ),
                ));
            }
            let (q_mismatch, q_worst_layer) = audit_bf16_mismatches(&record.q, &trace.q_values);
            q_mismatches += q_mismatch;
            q_bits += record.q.len();
            q_worst = q_worst.max(q_worst_layer);
            if q_mismatch != 0 {
                let index = record
                    .q
                    .iter()
                    .zip(trace.q_values.iter())
                    .position(|(left, right)| bf16_bits(*left) != bf16_bits(*right));
                let index = index.unwrap_or(usize::MAX);
                let reference = record.q.get(index).copied().unwrap_or(f32::NAN);
                let staged = trace.q_values.get(index).copied().unwrap_or(f32::NAN);
                details.push(format!(
                    "q first_mismatch_index={index} reference={reference:.9} staged={staged:.9} mismatches={q_mismatch}/{} max_abs={q_worst_layer:.6}",
                    record.q.len()
                ));
            }
            let sink_exact = record.sink.len() == trace.sink_values.len()
                && record
                    .sink
                    .iter()
                    .zip(trace.sink_values.iter())
                    .all(|(left, right)| left.to_bits() == right.to_bits());
            if sink_exact {
                sink_exact_layers += 1;
            } else {
                details.push(format!(
                    "sink reference={:?} staged={:?}",
                    &record.sink[..record.sink.len().min(DETAIL_IDS)],
                    &trace.sink_values[..trace.sink_values.len().min(DETAIL_IDS)]
                ));
            }
            let expected_rows: Vec<usize> = record
                .compressed_ids
                .iter()
                .filter(|id| **id != CASE_NO_INDEX)
                .filter_map(|id| id.checked_sub(record.window_rows))
                .collect();
            let ratio = trace.ratio.max(1);
            let device_rows: Vec<usize> = trace
                .selected
                .iter()
                .map(|absolute| absolute / ratio)
                .collect();
            let selection_exact = expected_rows == device_rows;
            if selection_exact {
                selection_exact_layers += 1;
            } else {
                details.push(format!(
                    "compressed_selection staged_count={} reference_count={} staged_rows={:?} reference_rows={:?}",
                    device_rows.len(),
                    expected_rows.len(),
                    &device_rows[..device_rows.len().min(DETAIL_IDS)],
                    &expected_rows[..expected_rows.len().min(DETAIL_IDS)]
                ));
            }
            let mut expected_compressed: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            for row_index in &expected_rows {
                let start = *row_index * record.head_dim;
                let end = start + record.head_dim;
                if end > record.compressed_kv.len() {
                    continue;
                }
                if let Ok(entry) = quantize_main_kv(&record.compressed_kv[start..end]) {
                    expected_compressed.push((entry.data.to_vec(), entry.scales.to_vec()));
                }
            }
            let mut compared = 0usize;
            let mut compared_exact = 0usize;
            let mut compared_bytes_differing = 0usize;
            let mut worst_compressed: Option<(usize, usize, usize, usize, usize)> = None;
            for row_index in &device_rows {
                let Ok(reference_index) = expected_rows.binary_search(row_index) else {
                    continue;
                };
                let Some(expected) = expected_compressed.get(reference_index) else {
                    continue;
                };
                let pool_row = row_index * ratio;
                let code_start = pool_row * 256;
                let scale_start = pool_row * 32;
                if code_start + 256 > trace.compressed_codes.len()
                    || scale_start + 32 > trace.compressed_scales.len()
                {
                    continue;
                }
                compared += 1;
                let row_differing = audit_byte_mismatches(
                    &expected.0,
                    &trace.compressed_codes[code_start..code_start + 256],
                );
                let row_scales = audit_byte_mismatches(
                    &expected.1,
                    &trace.compressed_scales[scale_start..scale_start + 32],
                );
                compared_bytes_differing += row_differing;
                if row_differing == 0 && row_scales == 0 {
                    compared_exact += 1;
                }
                let score = row_differing + row_scales;
                let worst_score =
                    worst_compressed.map_or(0, |(_, _, bytes, scales, _)| bytes + scales);
                if score > worst_score {
                    worst_compressed = Some((
                        *row_index,
                        reference_index,
                        pool_row,
                        row_differing,
                        row_scales,
                    ));
                }
            }
            compressed_rows += compared;
            compressed_rows_exact += compared_exact;
            compressed_bytes_differing += compared_bytes_differing;
            compressed_bytes += compared * 256;
            let compressed_detail = worst_compressed
                .filter(|(_, _, _, bytes, scales)| *bytes != 0 || *scales != 0)
                .and_then(|(row_index, reference_index, pool_row, row_differing, row_scales)| {
                    let expected = expected_compressed.get(reference_index)?;
                    let code_start = pool_row * 256;
                    let scale_start = pool_row * 32;
                    Some(format!(
                        "compressed_row row_index={row_index} pool_row={pool_row} bytes_differing={row_differing}/256 scales_differing={row_scales}/32 disagreement=[{}] scale_disagreement=[{}]",
                        audit_row_disagreements(
                            &expected.0,
                            &trace.compressed_codes[code_start..code_start + 256],
                            DETAIL_BYTES
                        ),
                        audit_row_disagreements(
                            &expected.1,
                            &trace.compressed_scales[scale_start..scale_start + 32],
                            DETAIL_BYTES / 2
                        )
                    ))
                });
            if let Some(detail) = compressed_detail {
                details.push(detail);
            }
            layer_lines.push(format!(
                "sequence operand audit: position={position} layer={layer} reference_seqlen={} window_rows={} window_width={} ids_exact={ids_exact} window_rows_compared={rows} window_rows_exact={exact} window_bytes_differing={differing}/{total} window_scales_differing={scale_differing} q_bf16_mismatches={q_mismatch}/{} q_max_abs={q_worst_layer:.6} sink_exact={sink_exact} compressed_selected={} compressed_reference={} selection_exact={selection_exact} compressed_rows_compared={compared} compressed_rows_exact={compared_exact} compressed_bytes_differing={compared_bytes_differing}/{} ratio={ratio}",
                record.seqlen,
                record.window_rows,
                record.window_width,
                record.q.len(),
                trace.selected.len(),
                expected_rows.len(),
                compared * 256,
            ));
            if details.is_empty() {
                continue;
            }
            if detail_layers < DETAIL_LAYERS {
                layer_lines.push(format!(
                    "sequence operand audit: position={position} layer={layer} detail: {}",
                    details.join(" | ")
                ));
                detail_layers += 1;
            } else if detail_layers == DETAIL_LAYERS {
                layer_lines.push(format!(
                    "sequence operand audit: position={position} detail: further mismatching layers suppressed after {DETAIL_LAYERS}"
                ));
                detail_layers += 1;
            }
        }
        if layers == 0 {
            lines.push(format!(
                "sequence operand audit: position={position} no device trace at the captured position: the trace covers other positions only, so the captured reference frame has nothing to compare against"
            ));
            continue;
        }
        let (mismatch_layer, mismatch_index, mismatch_expected, mismatch_staged) =
            first_ids_mismatch.unwrap_or((0, 0, 0, 0));
        lines.push(format!(
            "sequence operand audit: position={position} layers={layers} ids_exact={ids_exact_layers}/{layers} first_ids_mismatch_layer={mismatch_layer} first_ids_mismatch_index={mismatch_index} expected_slot={mismatch_expected} staged_slot={mismatch_staged} window_rows={window_rows} window_rows_exact={window_rows_exact} window_bytes_differing={window_bytes_differing}/{window_bytes} window_scales_differing={window_scales_differing} q_bf16_mismatches={q_mismatches}/{q_bits} q_max_abs={q_worst:.6} sink_exact_layers={sink_exact_layers}/{layers} selection_exact_layers={selection_exact_layers}/{layers} compressed_rows={compressed_rows} compressed_rows_exact={compressed_rows_exact} compressed_bytes_differing={compressed_bytes_differing}/{compressed_bytes}"
        ));
        lines.extend(layer_lines);
    }
    lines
}
#[cfg(test)]
mod tests {
    use super::default_twin_positions;
    use super::operand_audit_lines;
    use super::{audit_bf16_mismatches, audit_byte_mismatches, audit_expected_slots};
    use sparkflash_engine::forward::attention_case::AttentionCaseRecord;
    use std::collections::BTreeMap;

    /// A synthetic prefill record for a 130-token-shaped chunk in miniature: window 4, six chunk
    /// rows, and a query that attends chunk positions 4 and 5 (which the ring stores in slots 0
    /// and 1, i.e. the first two wrap steps), plus one compressed row.
    fn synthetic_record() -> AttentionCaseRecord {
        let head_dim = 512usize;
        let window_rows = 6usize;
        let row = |base: f32| -> Vec<f32> {
            (0..head_dim)
                .map(|index| base + index as f32 * 0.001)
                .collect()
        };
        let mut window_kv = Vec::new();
        for index in 0..window_rows {
            window_kv.extend(row(index as f32));
        }
        AttentionCaseRecord {
            layer: 0,
            position: 5,
            start_pos: 0,
            seqlen: 6,
            ratio: 2,
            heads: 1,
            head_dim,
            window_rows,
            window_width: 2,
            compressed_stride: 2,
            compressed_rows: 1,
            q: vec![0.5; head_dim],
            sink: vec![0.25],
            window_ids: vec![4, 5],
            window_kv,
            compressed_ids: vec![window_rows, super::CASE_NO_INDEX],
            compressed_kv: row(9.0),
            out: vec![0.0; head_dim],
        }
    }

    /// The trace the staging would produce from a matching hidden: same ids, the same quantisers
    /// applied to the same rows, one compressed row at pool row 0, the same q and sink.
    fn matching_trace(record: &AttentionCaseRecord) -> super::AttentionOperandTrace {
        let mut window_codes = Vec::new();
        let mut window_scales = Vec::new();
        for id in &record.window_ids {
            let start = id * record.head_dim;
            let row = super::engine_window_row(&record.window_kv[start..start + record.head_dim])
                .expect("a window row");
            window_codes.extend_from_slice(&row.codes);
            window_scales.extend_from_slice(&row.scales);
        }
        let entry = super::quantize_main_kv(&record.compressed_kv).expect("a compressed row");
        let ratio = 2usize;
        let mut compressed_codes = vec![0u8; 2 * 256];
        let mut compressed_scales = vec![0u8; 2 * 32];
        compressed_codes[..256].copy_from_slice(&entry.data);
        compressed_scales[..32].copy_from_slice(&entry.scales);
        super::AttentionOperandTrace {
            position: record.position,
            layer: record.layer,
            q_values: record.q.clone(),
            q_first: record.q[0],
            q_last: record.q[record.q.len() - 1],
            q_checksum: super::checksum_f32(&record.q),
            sink_first: record.sink[0],
            sink_last: record.sink[0],
            sink_checksum: super::checksum_f32(&record.sink),
            window_ids: vec![0, 1],
            selected: vec![0],
            compressed_count: 1,
            ratio,
            window_codes_len: window_codes.len(),
            window_first_checksum: super::checksum_bytes(&window_codes[..512]),
            window_last_checksum: super::checksum_bytes(&window_codes[512..]),
            compressed_codes_len: 256,
            window_codes,
            window_scales,
            compressed_codes,
            compressed_scales,
            sink_values: record.sink.clone(),
        }
    }

    fn audit_one(trace: &super::AttentionOperandTrace) -> Vec<String> {
        let record = synthetic_record();
        let mut cases = BTreeMap::new();
        cases.insert(record.position, vec![record]);
        operand_audit_lines(&cases, std::slice::from_ref(trace), 4)
    }

    /// A matching hidden audits clean: the prefill record's positions 4 and 5 map to slots 0 and 1,
    /// both window rows and the compressed row match byte for byte, and no detail line is printed.
    #[test]
    fn the_operand_audit_reports_a_matching_hidden_clean() {
        let record = synthetic_record();
        let lines = audit_one(&matching_trace(&record));
        assert_eq!(lines.len(), 2, "one summary and one layer line: {lines:#?}");
        let summary = &lines[0];
        assert!(summary.contains("ids_exact=1/1"), "{summary}");
        assert!(summary.contains("first_ids_mismatch_layer=0"), "{summary}");
        assert!(summary.contains("window_rows_exact=2"), "{summary}");
        assert!(summary.contains("window_bytes_differing=0/"), "{summary}");
        assert!(summary.contains("q_bf16_mismatches=0/512"), "{summary}");
        assert!(summary.contains("sink_exact_layers=1/1"), "{summary}");
        assert!(summary.contains("selection_exact_layers=1/1"), "{summary}");
        assert!(summary.contains("compressed_rows_exact=1"), "{summary}");
        assert!(lines[1].contains("layer=0"), "{}", lines[1]);
    }

    /// A rotated ring prints the raw slots: which index disagreed, what the ring should have held
    /// and what was staged, so one log line places the defect.
    #[test]
    fn the_operand_audit_prints_the_staged_slots_on_an_id_mismatch() {
        let record = synthetic_record();
        let mut trace = matching_trace(&record);
        trace.window_ids = vec![1, 0];
        let lines = audit_one(&trace);
        let all = lines.join("\n");
        assert!(lines[0].contains("ids_exact=0/1"), "{all}");
        assert!(lines[0].contains("expected_slot=0"), "{all}");
        assert!(lines[0].contains("staged_slot=1"), "{all}");
        assert!(all.contains("ids first_mismatch_index=0"), "{all}");
        assert!(all.contains("expected_head=[0, 1]"), "{all}");
        assert!(all.contains("staged_head=[1, 0]"), "{all}");
    }

    /// A single changed code byte is spelled out with its offset and both values, and the row is
    /// named by the window id it belongs to.
    #[test]
    fn the_operand_audit_prints_the_bytes_on_a_row_mismatch() {
        let record = synthetic_record();
        let mut trace = matching_trace(&record);
        trace.window_codes[0] ^= 0x01;
        let lines = audit_one(&trace);
        let all = lines.join("\n");
        assert!(lines[0].contains("window_bytes_differing=1/"), "{all}");
        assert!(all.contains("window_row row_index=0 window_id=4"), "{all}");
        assert!(all.contains("bytes_differing=1/512"), "{all}");
        assert!(all.contains("disagreement=[offset=0"), "{all}");
    }

    /// The query is compared as bf16 words: one changed word is reported with its index and both
    /// values, because a bf16 staging slip is the other half of the same question.
    #[test]
    fn the_operand_audit_prints_the_query_on_a_bf16_mismatch() {
        let record = synthetic_record();
        let mut trace = matching_trace(&record);
        trace.q_values[0] += 0.1;
        let lines = audit_one(&trace);
        let all = lines.join("\n");
        assert!(lines[0].contains("q_bf16_mismatches=1/512"), "{all}");
        assert!(all.contains("q first_mismatch_index=0"), "{all}");
    }

    /// A capture with no matching device trace says so instead of printing a table of zeros: that
    /// happens when the twin was not armed at the captured position or when the A/B switch ran the
    /// host attention loop there, and both cases mean nothing is comparable.
    #[test]
    fn the_operand_audit_says_so_when_there_is_no_device_trace() {
        let record = synthetic_record();
        let mut cases = BTreeMap::new();
        cases.insert(record.position, vec![record]);
        let empty = operand_audit_lines(&cases, &[], 4);
        assert_eq!(empty.len(), 1, "{empty:#?}");
        assert!(empty[0].contains("no device trace"), "{}", empty[0]);
        let mut moved = matching_trace(&synthetic_record());
        moved.position = 99;
        let unmatched = operand_audit_lines(&cases, std::slice::from_ref(&moved), 4);
        assert_eq!(unmatched.len(), 1, "{unmatched:#?}");
        assert!(unmatched[0].contains("no device trace"), "{}", unmatched[0]);
    }

    /// A dropped or renumbered compressed candidate is printed as both row lists, so a selected-id
    /// defect cannot hide behind matching row content.
    #[test]
    fn the_operand_audit_prints_the_candidate_rows_on_a_selection_mismatch() {
        let record = synthetic_record();
        let mut trace = matching_trace(&record);
        trace.selected = vec![2];
        let lines = audit_one(&trace);
        let all = lines.join("\n");
        assert!(lines[0].contains("selection_exact_layers=0/1"), "{all}");
        assert!(
            all.contains("compressed_selection staged_count=1 reference_count=1"),
            "{all}"
        );
        assert!(all.contains("staged_rows=[1]"), "{all}");
        assert!(all.contains("reference_rows=[0]"), "{all}");
    }

    /// A reference prefill record names token positions, and the ring maps a position to its slot.
    /// With window 128 a 130-token prefill's last query attends positions 2..=129, which are ring
    /// slots 2..=127 and then the two rows the first two wraps wrote (0 and 1).
    #[test]
    fn the_operand_audit_maps_a_prefill_record_onto_ring_slots() {
        let ids: Vec<usize> = (2..=129).collect();
        let slots = audit_expected_slots(&ids, 130, 128);
        let mut expected: Vec<usize> = (2..=127).collect();
        expected.push(0);
        expected.push(1);
        assert_eq!(slots, expected);
        // The mapping is a permutation of the ring at the boundary: no slot twice, none missing.
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..128).collect::<Vec<_>>());
        // A prefill chunk shorter than the window keeps its positions as slots, unrotated.
        assert_eq!(audit_expected_slots(&[0, 1, 2], 3, 128), vec![0, 1, 2]);
    }

    /// A reference decode record already names ring slots, so the audit must not rotate them again.
    #[test]
    fn the_operand_audit_leaves_a_decode_record_on_its_own_slots() {
        let slots: Vec<usize> = (1..128).chain([0]).collect();
        assert_eq!(audit_expected_slots(&slots, 1, 128), slots);
    }

    /// The audits report counts, not booleans: a row that differs in one byte is not the same
    /// finding as a row that differs in all of them, and the bf16 check must compare words.
    #[test]
    fn the_operand_audit_counts_mismatches() {
        assert_eq!(audit_byte_mismatches(&[1, 2, 3, 4], &[1, 9, 3, 4]), 1);
        assert_eq!(audit_byte_mismatches(&[1, 2], &[1, 2, 3]), 1);
        assert_eq!(audit_byte_mismatches(&[], &[]), 0);
        let (mismatches, worst) = audit_bf16_mismatches(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert_eq!(mismatches, 0);
        assert_eq!(worst, 0.0);
        // Half a bf16 ulp at 1.0 is 2^-9, so 1.01 is a different word and 1.0 + 0.0005 is not.
        let (mismatches, worst) = audit_bf16_mismatches(&[1.0], &[1.01]);
        assert_eq!(mismatches, 1);
        assert!(worst > 0.0);
        let (mismatches, _) = audit_bf16_mismatches(&[1.0], &[1.0 + 0.0005]);
        assert_eq!(mismatches, 0);
        let (mismatches, _) = audit_bf16_mismatches(&[1.0, 2.0], &[1.0]);
        assert_eq!(mismatches, 1);
    }

    /// The default ladder is the prefill diagnostics plus the ring's first two wrap steps. With a
    /// window of W the write at position W evicts slot 0 (first wrap) and the write at W + 1
    /// rotates the oldest slot (second wrap). b70 diverged at the 130th token with W = 128, which
    /// is position 129, so both boundary positions must be armed without a flag.
    #[test]
    fn the_boundary_ladder_arms_the_first_two_wrap_steps_by_default() {
        assert_eq!(default_twin_positions(128), vec![0, 1, 2, 128, 129]);
        assert_eq!(default_twin_positions(4), vec![0, 1, 2, 4, 5]);
        // A window smaller than the prefill diagnostics cannot add duplicates: the ladder is a set.
        assert_eq!(default_twin_positions(1), vec![0, 1, 2]);
        assert_eq!(default_twin_positions(2), vec![0, 1, 2, 3]);
        assert_eq!(default_twin_positions(3), vec![0, 1, 2, 3, 4]);
        // The ladder is ascending, so the printed armed-position list reads in position order.
        for window in [3usize, 4, 17, 128, 129] {
            let positions = default_twin_positions(window);
            assert!(
                positions.windows(2).all(|pair| pair[0] < pair[1]),
                "{positions:?}"
            );
        }
    }
}

#[cfg(test)]
mod host_seam_tests {
    use super::*;

    /// F5: a W8A8 activation is a bf16 module output in the reference, so the host staging
    /// narrows the f32 value before the quantiser. Rounding first is not cosmetic: there are
    /// blocks whose f32 elements straddle an E4M3 code boundary and pick a code the reference
    /// can never pick. The block is found by a deterministic scan, so the test states the rule
    /// rather than a magic constant.
    #[test]
    fn the_w8a8_activation_quantiser_reads_the_bf16_value() {
        let mut seed = 0x5EEDu32;
        for _ in 0..10_000 {
            let mut block: Vec<f32> = Vec::with_capacity(32);
            for _ in 0..32 {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                block.push((seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0);
            }
            let narrowed: Vec<f32> = block
                .iter()
                .map(|value| sparkflash_engine::quant::bf16_round(*value))
                .collect();
            let (codes, scales) =
                quantise_native_activation(&block, "test activation").expect("32 divides by 32");
            let expected = quantize_activation_fp8(&narrowed).expect("32 divides by 32");
            assert_eq!(codes, expected.codes().to_vec());
            assert_eq!(scales, expected.scales().to_vec());
            let (raw_codes, _) = quantise(&block, "test activation").expect("32 divides by 32");
            if raw_codes != codes {
                return;
            }
        }
        panic!("a bf16 pre-round must be able to change an E4M3 code");
    }
}
