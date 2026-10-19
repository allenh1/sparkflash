//! The c4a device attention backend: the five projection GEMMs plus the format-2 CSA-union
//! attention body on the device, with the host normalisation, RoPE, KV, cache, compressor and
//! selected-list seams left exactly where they are. This module is only compiled with
//! `--features cuda`.

use std::path::Path;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention::AttentionProjectionBackend;
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::quant::quantize_activation_fp8;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use sparkflash_kernels::cuda_harness::{
    FlashMlaKernel, engine_window_row, flash_mla_params, launch_flash_mla, load_flash_mla,
};
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mxfp8_dense_launch::{self, Mxfp8DenseParams};
use sparkflash_kernels::mxfp8_ref::b12x_weight_index;
use sparkflash_kernels::mxfp8_wo_a_launch::{self, Mxfp8WoAParams};
use sparkflash_kv::fp4::quantize_main_kv;

const DENSE_KERNEL: &str = "b12x_mxfp8_dense_sm121";
const WO_A_KERNEL: &str = "b12x_mxfp8_wo_a_sm121";
const FLASH_MLA_KERNEL: &str = "flash_mla_csa2_sm121";
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
/// The maximum activation output: wq_b's 32,768 bf16 values.
const MAX_OUT_BYTES: usize = 32_768 * 2;

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

struct Kernels {
    dense: Kernel,
    wo_a: Kernel,
    flash: FlashMlaKernel,
    _dense_module: Module,
    _wo_a_module: Module,
    _flash_module: Module,
}

struct Buffers {
    a_codes: Deviceptr,
    a_scales: Deviceptr,
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
        let flash_bytes = verified_cubin(dir, FLASH_MLA_KERNEL)?;
        let cuda = Cuda::load()?;
        let (dense_module, dense) = cuda.load_kernel(&dense_bytes, KernelId::Mxfp8Dense)?;
        let (wo_a_module, wo_a) = cuda.load_kernel(&wo_a_bytes, KernelId::Mxfp8WoA)?;
        let (flash_module, flash) = load_flash_mla(&cuda, &flash_bytes)?;
        let mut total = 0usize;
        let alloc = |bytes: usize, total: &mut usize| cuda.alloc(bytes, total);
        let a_codes = alloc(MAX_ACT_BYTES, &mut total)?;
        let a_scales = alloc(MAX_ACT_SCALES, &mut total)?;
        let w = alloc(MAX_WEIGHT_BYTES, &mut total)?;
        let w_scales = alloc(MAX_WEIGHT_SCALES, &mut total)?;
        let out = alloc(MAX_OUT_BYTES, &mut total)?;
        let ptrs = vec![a_codes, a_scales, w, w_scales, out];
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
                flash,
                _dense_module: dense_module,
                _wo_a_module: wo_a_module,
                _flash_module: flash_module,
            },
            buffers: Buffers {
                a_codes,
                a_scales,
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
        })
    }

    /// The counters since construction.
    #[must_use]
    pub fn stats(&self) -> AttentionDeviceStats {
        self.stats
    }

    /// The format-2 body launches this backend performed for `layer`.
    #[must_use]
    pub fn body_launches(&self, layer: usize) -> usize {
        self.body_by_layer.get(layer).copied().unwrap_or(0)
    }

    /// The persistent device bytes this backend allocated.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        (MAX_ACT_BYTES
            + MAX_ACT_SCALES
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
        let (packed, scales) =
            read_packed(src, &format!("{prefix}.attn.{weight_name}"), cols, inner)?;
        let codes = self
            .last_activation
            .take()
            .ok_or_else(|| EngineError::Shape {
                what: "attention activation".to_owned(),
                detail: "the projection backend has no quantised activation".to_owned(),
            })?;
        self.upload_weight(&packed, &scales)
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
        self.dense(
            config,
            layer,
            prefix,
            src,
            "wq_a.weight",
            config.q_lora_rank,
            config.dim,
            rows_of(x, config.dim, "wq_a input")?,
        )
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
        self.last_activation = Some(quantise(latent, "wq_b activation")?);
        self.dense(
            config,
            layer,
            prefix,
            src,
            "wq_b.weight",
            config.n_heads * config.head_dim,
            rank,
            rows_of(latent, rank, "wq_b input")?,
        )
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
        self.dense(
            config,
            layer,
            prefix,
            src,
            "wkv.weight",
            config.head_dim,
            config.dim,
            rows_of(x, config.dim, "wkv input")?,
        )
    }

    fn wo_a(
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
        let (packed, scales) = read_packed(
            src,
            &format!("{prefix}.attn.wo_a.weight"),
            config.o_groups * config.o_lora_rank,
            wide_row / config.o_groups,
        )?;
        self.upload_weight(&packed, &scales)
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
        Ok(grouped)
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
