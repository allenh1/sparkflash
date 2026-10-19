//! The c4a device attention backend: the five projection GEMMs plus the format-2 CSA-union
//! attention body on the device, with the host normalisation, RoPE, KV, cache, compressor and
//! selected-list seams left exactly where they are. This module is only compiled with
//! `--features cuda`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::AttentionProjectionBackend;
use sparkflash_engine::forward::attention::sparse_attention;
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

/// One launch's bookkeeping.
#[derive(Debug, Default, Clone, Copy)]
pub struct AttentionDeviceStats {
    /// GEMM launches, five per layer when every method runs.
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

/// The device implementation of [AttentionProjectionBackend]. It stages one layer's five weights at
/// a time into reused buffers; it never allocates the 5.07 GB cumulative matrix set on the device.
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
        if start_pos <= 2 {
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
        if start_pos > 2 {
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
        self.last_activation = Some(quantise(x, "wq_a activation")?);
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
        self.last_activation = Some(quantise(latent, "wq_b activation")?);
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
        self.last_activation = Some(quantise(x, "wkv activation")?);
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
        self.last_activation = Some(quantise(grouped, "wo_b activation")?);
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
