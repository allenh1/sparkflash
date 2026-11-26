//! Microbenchmark for Milestone 4 measurements on GLM-5.3-Flash TP4 geometry:
//! - A1: single-launch `q4e_glm_mla_attend` vs 4-launch `q4e_glm_mla_scores+softmax+mix+project`
//! - A4: bulk `glm_prefill_mla_attend + glm_prefill_heads_bf16_mma` vs 4-launch at rows=32,64,128
//! - K5: `KdaLayer::run` decode (1 row) and prefill (64 rows) scan + conv + decay time

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::Image;
use sparkflash_model::TpPlan;
use sparkflash_model::backend::BackendError;
use sparkflash_ops::gdn::{ChunkConv, GatedNorm, GdnGeometry, KdaChunk, gdn_scan_shape_for};
use sparkflash_ops::indexer::MAX_SELECTED_TOKENS;
use sparkflash_ops::plan::Precision;
use sparkflash_ops::roles::{
    AbsorbedGeometry, AbsorbedMla, AbsorbedRows, AbsorbedWeights, DeviceCx, KdaLayer, KdaRows,
    KernelTable, Launcher, SegmentRows,
};
use std::time::Instant;

struct NoopCx<'a> {
    cuda: &'a Cuda,
    rows: usize,
    segs: &'a [SegmentRows<'a>],
}

impl Launcher for NoopCx<'_> {
    fn launch(
        &self,
        kernel: impl Into<Kernel>,
        (grid, block): ((u32, u32, u32), (u32, u32, u32)),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.cuda
            .launch(kernel.into(), grid, block, args)
            .map_err(|e| BackendError::new("bench", e))
    }
}

impl DeviceCx for NoopCx<'_> {
    type Linear = ();
    fn rows(&self) -> usize {
        self.rows
    }
    fn segments(&self) -> impl Iterator<Item = SegmentRows<'_>> + '_ {
        self.segs.iter().copied()
    }
    fn share(&self) -> Option<TpPlan> {
        TpPlan::new(4, 0).ok()
    }
    fn project(
        &self,
        _linear: &(),
        _input: Deviceptr,
        _output: Deviceptr,
        _rows: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }
    fn project_strided(
        &self,
        _linear: &(),
        _input: Deviceptr,
        _out: (Deviceptr, usize),
        _rows: usize,
    ) -> Result<(), BackendError> {
        Ok(())
    }
    fn sum(&self, _partial: Deviceptr, _rows: usize, _width: usize) -> Result<(), BackendError> {
        Ok(())
    }
    fn upload(&self, to: Deviceptr, bytes: &[u8]) -> Result<(), BackendError> {
        self.cuda
            .copy_htod(&to, bytes)
            .map_err(|e| BackendError::new("bench", e))
    }
}

fn time_us<F: FnMut() -> Result<(), String>>(
    cuda: &Cuda,
    warmup: usize,
    iters: usize,
    mut f: F,
) -> Result<f64, String> {
    for _ in 0..warmup {
        f()?;
    }
    cuda.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    cuda.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
}

fn main() -> Result<(), String> {
    let cuda = Cuda::load()?;

    let geom = AbsorbedGeometry {
        heads: 16,
        nope: 256,
        value: 256,
        latent: 512,
        q_lora: 1536,
    };
    let mla = AbsorbedMla::open(&cuda, geom, Precision::Bf16).map_err(|e| e.to_string())?;
    let [attend_1l] = KernelTable {
        image: Image::DECODE,
        symbols: [("q4e_glm_mla_attend", Precision::Fp8E4M3)],
    }
    .load(&cuda)?;

    let max_rows = 128usize;
    let max_pos = 2048usize;
    let max_sel = MAX_SELECTED_TOKENS;
    let mut allocs = Vec::new();
    let mut alloc = |bytes: usize| -> Result<Deviceptr, String> {
        let p = cuda.alloc(bytes.max(16), &mut 0)?;
        cuda.memset_d8_async(p, 0, bytes.max(16))?;
        allocs.push(p);
        Ok(p)
    };

    let q_latent = alloc(max_rows * geom.q_lora * 4)?;
    let latent = alloc(max_rows * geom.latent * 4)?;
    let query = alloc(max_rows * geom.heads * geom.nope * 4)?;
    let absorbed = alloc(max_rows * geom.heads * geom.latent * 4)?;
    let logits = alloc(max_rows * geom.heads * max_sel * 4)?;
    let total = alloc(max_rows * geom.heads * 4)?;
    let mixed = alloc(max_rows * geom.heads * geom.latent * 4)?;
    let heads = alloc(max_rows * geom.heads * geom.value * 4)?;
    let out = alloc(max_rows * 4096 * 4)?;
    let codes = alloc(max_pos * geom.latent)?;
    let scales = alloc(max_pos * (geom.latent / 128) * 4)?;
    let kv_b_f32 = alloc(geom.heads * (geom.nope + geom.value) * geom.latent * 4)?;
    let kv_b_bf16 = alloc(geom.heads * (geom.nope + geom.value) * geom.latent * 2)?;
    let selected = alloc(max_rows * max_sel * 4)?;
    let selected_count = alloc(max_rows * 4)?;

    let ar = AbsorbedRows {
        q_latent,
        latent,
        query,
        absorbed,
        logits,
        total,
        mixed,
        heads,
    };
    let dummy = ();
    let w_four = AbsorbedWeights {
        q_a: &dummy,
        kv_a: &dummy,
        q_b: &dummy,
        out: &dummy,
        norms: (Deviceptr(0), Deviceptr(0)),
        kv_b: kv_b_f32,
        kv_b_bf16: None,
    };
    let w_bulk = AbsorbedWeights {
        kv_b_bf16: Some((kv_b_bf16, kv_b_bf16.offset(geom.heads * geom.nope * geom.latent * 2))),
        ..w_four
    };

    println!("=== A1: Absorbed MLA decode (1-launch q4e_glm_mla_attend vs 4-launch split, E4M3 cache, H=16, D=256, L=512) ===");
    for &seq in &[256usize, 512, 2048] {
        let sel_ids: Vec<i32> = (0..seq as i32).collect();
        let mut host_sel = Vec::with_capacity(max_rows * max_sel);
        for _ in 0..max_rows {
            host_sel.extend_from_slice(&sel_ids);
            host_sel.resize(host_sel.len() + (max_sel - seq), 0);
        }
        let sel_bytes: Vec<u8> = host_sel.iter().flat_map(|v| v.to_le_bytes()).collect();
        cuda.copy_htod(&selected, &sel_bytes)?;
        let cnts: Vec<i32> = vec![seq as i32; max_rows];
        let cnt_bytes: Vec<u8> = cnts.iter().flat_map(|v| v.to_le_bytes()).collect();
        cuda.copy_htod(&selected_count, &cnt_bytes)?;

        for &rows in &[1usize, 4] {
            let one_us = time_us(&cuda, 10, 100, || {
                for r in 0..rows {
                    cuda.launch(
                        attend_1l.kernel(),
                        (geom.heads as u32, 1, 1),
                        (256, 1, 1),
                        &[
                            absorbed.offset(r * geom.heads * geom.latent * 4).0,
                            0,
                            codes.0,
                            scales.0,
                            kv_b_f32.0,
                            selected.offset(r * max_sel * 4).0,
                            selected_count.offset(r * 4).0,
                            heads.offset(r * geom.heads * geom.value * 4).0,
                            0,
                            geom.nope as u64,
                            geom.value as u64,
                            geom.latent as u64,
                            max_sel as u64,
                            1,
                        ],
                    )?;
                }
                Ok(())
            })?;
            let regs = [codes, scales];
            let segs = [SegmentRows {
                first: 0,
                rows,
                position: seq - rows,
                regions: &regs,
                snapshots: &[],
            }];
            let cx = NoopCx { cuda: &cuda, rows, segs: &segs };
            let four_us = time_us(&cuda, 10, 100, || {
                mla.attend(&cx, &w_four, &ar, (selected, selected_count), out, (true, false))
                    .map_err(|e| e.to_string())
            })?;
            println!("A1 rows={rows:<3} S={seq:<4}  1-launch={one_us:7.1} us   4-launch={four_us:7.1} us   ratio(4L/1L)={:.2}x", four_us / one_us);
        }

        println!("--- A4: Bulk prefill attention (glm_prefill_mla_attend+BF16 MMA vs 4-launch, S={seq}) ---");
        for &rows in &[32usize, 64, 128] {
            let regs = [codes, scales];
            let segs = [SegmentRows {
                first: 0,
                rows,
                position: seq - rows,
                regions: &regs,
                snapshots: &[],
            }];
            let cx = NoopCx { cuda: &cuda, rows, segs: &segs };
            let four_us = time_us(&cuda, 5, 40, || {
                mla.attend(&cx, &w_four, &ar, (selected, selected_count), out, (true, false))
                    .map_err(|e| e.to_string())
            })?;
            let bulk_us = time_us(&cuda, 5, 40, || {
                mla.attend(&cx, &w_bulk, &ar, (selected, selected_count), out, (true, false))
                    .map_err(|e| e.to_string())
            })?;
            println!("A4 rows={rows:<3} S={seq:<4}  4-launch={four_us:7.1} us   glm-bulk={bulk_us:7.1} us   speedup={:.2}x", four_us / bulk_us);
        }
    }

    let gdn_geom = GdnGeometry {
        key_heads: 16,
        heads: 16,
        key_dim: 128,
        value_dim: 128,
        taps: 4,
    };
    let [conv_rows, conv_hist, kda_prep, kda_scan, gated_norm] = KernelTable {
        image: Image::SCAN,
        symbols: [
            (ChunkConv::SYMBOLS[0], Precision::Bf16),
            (ChunkConv::SYMBOLS[1], Precision::F32),
            (KdaChunk::PREP, Precision::F32),
            ("q4p_kda_scan", Precision::F32),
            ("q4p_gdn_gate", Precision::Bf16),
        ],
    }
    .load(&cuda)?;
    let [kda_decay] = KernelTable {
        image: Image::TOKEN,
        symbols: [(KdaChunk::DECAY, Precision::Bf16)],
    }
    .load(&cuda)?;
    let chunk = KdaChunk::new(
        ChunkConv::new(conv_rows, conv_hist).map_err(|e| e.to_string())?,
        kda_decay,
        kda_prep,
        GatedNorm::new(gated_norm).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let (grid, block) = gdn_scan_shape_for(gdn_geom.heads, gdn_geom.value_dim);
    let scan = (kda_scan, (grid, (block, 1, 1)));
    let w = gdn_geom.key_width();
    let kda_rows = KdaRows {
        raw: alloc(64 * 3 * w * 4)?,
        f_a: alloc(64 * 128 * 4)?,
        g_a: alloc(64 * 128 * 4)?,
        b: alloc(64 * gdn_geom.heads * 4)?,
        mixed: alloc(64 * 3 * w * 4)?,
        forget: alloc(64 * w * 4)?,
        gate: alloc(64 * w * 4)?,
        decay: alloc(64 * w * 4)?,
        qk: alloc(gdn_geom.qk_floats(64) * 4)?,
        decay_beta: alloc(64 * gdn_geom.heads * 8)?,
        attended: alloc(64 * w * 4)?,
    };
    let conv_w = alloc(3 * w * gdn_geom.taps * 4)?;
    let decay_a = alloc(gdn_geom.heads * 4)?;
    let decay_dt = alloc(w * 4)?;
    let norm_w = alloc(gdn_geom.value_dim * 4)?;
    let conv_state = alloc(gdn_geom.conv_bytes())?;
    let rec_state = alloc(gdn_geom.state_bytes())?;
    let kda_layer = KdaLayer {
        qkv: [&dummy, &dummy, &dummy],
        forget: (&dummy, &dummy),
        gate: (&dummy, &dummy),
        b: &dummy,
        out: &dummy,
        conv: conv_w,
        decay: (decay_a, decay_dt),
        norm: norm_w,
        hold: None,
    };
    let inp = alloc(64 * 4096 * 4)?;
    println!("=== K5: KdaLayer (scan+conv+decay, no GEMM) on GLM TP4 geometry (H=16, D=128, K=4) ===");
    for &rows in &[1usize, 64] {
        let regs = [conv_state, rec_state];
        let segs = [SegmentRows {
            first: 0,
            rows,
            position: 0,
            regions: &regs,
            snapshots: &[],
        }];
        let cx = NoopCx { cuda: &cuda, rows, segs: &segs };
        let us = time_us(&cuda, 10, 100, || {
            kda_layer
                .run(&cx, (&chunk, scan), &gdn_geom, &kda_rows, (inp, out), (1e-5, -5.0))
                .map_err(|e| e.to_string())
        })?;
        println!("K5 rows={rows:<3}  KdaLayer={us:7.1} us/layer  (34 layers = {:.2} ms)", us * 34.0 / 1000.0);
    }

    let borrowed: Vec<&Deviceptr> = allocs.iter().collect();
    cuda.free_buffers(&borrowed);
    Ok(())
}
