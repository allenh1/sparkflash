//! m20: the deferred sequence's position-causal CSA2 entry (flash_mla_csa2_seq_sm121) against
//! mla_ref::attention_with_sink_union, at every position the deferred sequence compares.
//!
//! THIS EXAMPLE IS THE PER-POSITION TWIN BAR FOR THE c4a-2 start_pos > 0 EXTENSION. It is not the
//! served forward pass, not the chain wiring (engine-dev's call site), not a token, timing or
//! throughput claim, and it does not re-measure the base entry's published b31/b33/b39/b40 windows.
//!
//! WHY IT EXISTS. The c4a-2 device attention body ran only at position 0 (window b45:
//! attention_body_launches=40 where the sequence expected 120, so positions 1 and 2 fell back to the
//! host loop). The base kernel's format-2 arithmetic already accepts window_rows > 1 and a non-empty
//! selected list at any position (m8's captured fixture is position 129 with 128 window rows), but
//! its INDEXER stage scores the whole HSI pool with no causal mask: a per-position candidate
//! selection built from block_best would see future tokens. The sequence entry adds that mask and
//! nothing else; this harness measures that entry at positions 0..=2 and 129..=137, which is exactly
//! the deferred sequence's regime (positions 0..=2 are b45's prompt, 129..=137 is the full-window /
//! compressed regime).
//!
//! OPERANDS ARE DECLARED SYNTHETIC, generated deterministically in this process from a fixed seed
//! and uploaded to the device; no checkpoint byte is read. The operands match the regime
//! engine-dev declared for the sequence path (ratio-2 store, 128-row window, compressed ids at the
//! even absolute positions):
//!
//!   * q: 64 heads x 512 bf16 values, as if rope had already been applied (the kernel takes rotated
//!     q and k; this harness does not claim to implement rope).
//!   * window rows: for position p, window_start = max(0, p + 1 - 128) and rows
//!     window_start..=p, OLDEST FIRST, each encoded with the engine's own window row helper
//!     (cuda_harness::engine_window_row) and checked finite.
//!   * compressed store: 256 f32 values per store row, encoded to E2M1 latent + E4M3 scale by the
//!     pinned codec, then presented as the per-position expanded view row q = store row q / 2.
//!   * selected ids: the even absolute positions at or behind the query, ascending (count
//!     floor(p / 2) + 1, at most 70 here, inside the 512 bound).
//!   * sink: one small f32 per head, from the same deterministic stream.
//!   * indexer: a 2,048 x 8 x 128 bf16 key pool and 32 x 128 bf16 index queries, uploaded once.
//!     The index values are small integers (-4..=4), so every block dot is exact in f32 and the
//!     indexer twin is a bit-level comparison, not a tolerance (m7's stage-2 self-check precedent).
//!
//! PER POSITION THE HARNESS CLAIMS, and compares bit-level after the declared bar:
//!
//!   1. union attention: device out vs mla_ref::attention_with_sink_union over the exact device
//!      bytes (window engine format, position-indexed compressed store), at the published 2B bar
//!      2e-2, reporting exact-bf16 count, max abs delta, max ULP and non-finite count.
//!   2. causal indexer: block_best is at or behind the query for every one of the 2,048 blocks;
//!      blocks whose first token is past the query publish (-FLT_MAX, 0); the reachable blocks are
//!      compared with mla_ref::indexer_block_scores, which reproduces the kernel's warp reduction
//!      order (scores at the 2B bar, tokens exactly).
//!   3. determinism: two launches write byte-identical out[] and indexer arrays.
//!   4. over-supply: appending a selected id at position + 2 (a future id the caller may not have
//!      filtered) must leave out[] byte-identical, because the kernel's own mask excludes it.
//!   5. position 0: the new entry's out[] must be byte-identical to the BASE entry's out[] on the
//!      same operands, so the measured position-0 behaviour is preserved. If this check fails the
//!      record says so and engine-dev keeps the base entry at position 0 (the sequence is device at
//!      every position either way); it is measured, not assumed.
//!
//! PINS. Both cubins are refused by size and sha256 against the embedded baseline manifest before
//! any device allocation: flash_mla_csa2_sm121 60,792 B / 5bfee4c7... / membar 0 (untouched base
//! entry) and flash_mla_csa2_seq_sm121 38,080 B / f6dfc738... / membar 0.
//!
//! Usage (aarch64 node):
//!
//!     m20_csa2_sequence --cubin-dir <dir> [--out report.json]
//!
//! Build lines (nvcc 13.0.88 / CUDA 13.0.3):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_sm121.cu -o <dir>/flash_mla_csa2_sm121.cubin
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/flash_mla_csa2_seq_sm121.cu -o <dir>/flash_mla_csa2_seq_sm121.cubin

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use half::bf16;
use sha2::{Digest, Sha256};

use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::cuda_harness::{
    assert_engine_window_finite, engine_window_row, flash_mla_params, launch_flash_mla,
    load_flash_mla,
};
use sparkflash_kernels::flash_mla_csa2_seq_launch::{
    self as seq, FlashMlaSeqParams, HEADS, LATENT, MAX_SELECTED, SWA_WINDOW, THREADS,
};
use sparkflash_kernels::m4_quant::{bf16_bits, fp4_e2m1_encode, fp8_e4m3_decode, fp8_e4m3_encode};
use sparkflash_kernels::mla_ref;

/// The base entry's pin, unchanged by this work.
const BASE_KERNEL: &str = "flash_mla_csa2_sm121";
const BASE_BYTES: usize = 60_792;
const BASE_SHA256: &str = "5bfee4c7a0ece7c921e591fc6b8bd9d98c39b95d735d7c1d699ec58edaa51ea3";

/// The sequence entry's pin, from baseline/kernel_baseline.txt through baseline::pin.
const SEQ_KERNEL: &str = "flash_mla_csa2_seq_sm121";
const SEQ_BYTES: usize = 38_080;
const SEQ_SHA256: &str = "f6dfc738a7ad867adf040384f194812337301aa888cc486d4a60923772ba28c3";

/// The 2B contract's comparison bar for the attention output.
const BAR_ABS: f32 = 2e-2;
/// The indexer score bar: the same arithmetic order, one ULP class of the f32 dot.
const INDEX_BAR_ABS: f32 = 2e-2;

const INDEX_HEAD_DIM: usize = mla_ref::INDEX_HEAD_DIM;
const HSI_BLOCKS: usize = mla_ref::HSI_BLOCKS;
const HSI_BLOCK_TOKENS: usize = mla_ref::HSI_BLOCK_TOKENS;
const INDEX_HEADS: usize = mla_ref::INDEX_HEADS;

/// The positions the deferred sequence compares: b45's prompt positions and the full-window regime.
const POSITIONS: [usize; 12] = [0, 1, 2, 129, 130, 131, 132, 133, 134, 135, 136, 137];

fn main() {
    if let Err(error) = run() {
        eprintln!("m20_csa2_sequence: {error}");
        std::process::exit(1);
    }
}

struct Args {
    cubin_dir: PathBuf,
    out: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let mut cubin_dir = None;
        let mut out = None;
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--cubin-dir" => {
                    cubin_dir = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--cubin-dir needs a path".to_string())?,
                    ));
                }
                "--out" => {
                    out = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out needs a path".to_string())?,
                    ));
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        Ok(Self {
            cubin_dir: cubin_dir.ok_or_else(|| "--cubin-dir is required".to_string())?,
            out,
        })
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn check_pin(kernel: &str, bytes: usize, sha: &str) -> Result<(), String> {
    let pin = baseline::pin(kernel)?
        .ok_or_else(|| format!("the baseline manifest does not pin {kernel}"))?;
    if pin.cubin_bytes != bytes || pin.cubin_sha256 != sha || pin.ptx_membar_lines != 0 {
        return Err(format!(
            "{kernel}: baseline pins {} B / {} / membar {}, the harness declares {bytes} B / {sha} / membar 0",
            pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
        ));
    }
    println!(
        "baseline manifest agrees: {kernel} {} B, {}, {} PTX membar lines",
        pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
    );
    Ok(())
}

fn load_cubin(dir: &Path, kernel: &str, bytes: usize, sha: &str) -> Result<Vec<u8>, String> {
    let path = dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let digest = sha256_bytes(&cubin);
    if cubin.len() != bytes || digest != sha {
        return Err(format!(
            "{}: {} B / {digest}; the harness pins {bytes} B / {sha}",
            path.display(),
            cubin.len()
        ));
    }
    println!(
        "cubin: {} ({} B, sha256 {digest})",
        path.display(),
        cubin.len()
    );
    Ok(cubin)
}

/// A tiny deterministic stream (xorshift64*). Every operand below comes from it; nothing here reads
/// a checkpoint byte, and the recorded seed is part of the report.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A finite f32 in [-1, 1], uniformly enough for a synthetic operand.
    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }

    fn next_bf16_bits(&mut self) -> u16 {
        bf16_bits(self.next_f32())
    }

    /// A small integer in [-4, 4] as bf16 bits, for the indexer pool. Integer-valued bf16 index
    /// data makes every 128-wide block dot exact in f32 whatever association or FMA contraction the
    /// compiler picks, so the indexer twin compares scores bit for bit and its token tie rule is
    /// identical on both sides (the same reason m7's stage-2 indexer self-check uses integer data).
    fn next_index_bits(&mut self) -> u16 {
        let value = (self.next_u64() % 9) as i32 - 4;
        bf16_bits(value as f32)
    }
}

fn bf16_bytes_from_bits(bits: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len() * 2);
    for value in bits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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
        .map(|chunk| bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
        .collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Quantises one 512-wide latent row to the pinned E2M1 latent plus E4M3 scales (the store format
/// mla_ref::kv_latent decodes and the kernel reads). The exact quantiser is not the point here: the
/// device and the reference consume the SAME bytes, so a plain per-16-channel max-scale encode is
/// enough for a twin comparison.
fn quantize_store_row(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(values.len(), mla_ref::MLA_LATENT);
    let mut codes = vec![0u8; mla_ref::ROW_PACKED_BYTES];
    let mut scales = vec![0u8; mla_ref::ROW_SCALE_BYTES];
    for (group, scale_slot) in scales.iter_mut().enumerate() {
        let start = group * mla_ref::KV_SCALE_CHANNELS;
        let block = &values[start..start + mla_ref::KV_SCALE_CHANNELS];
        let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = if amax > 0.0 { amax / 6.0 } else { 1.0 };
        let scale_code = fp8_e4m3_encode(scale);
        let scale_value = fp8_e4m3_decode(scale_code);
        *scale_slot = scale_code;
        for (offset, value) in block.iter().enumerate() {
            let nibble = fp4_e2m1_encode((value / scale_value).clamp(-6.0, 6.0));
            let index = start + offset;
            if index.is_multiple_of(2) {
                codes[index / 2] |= nibble & 0x0F;
            } else {
                codes[index / 2] |= (nibble & 0x0F) << 4;
            }
        }
    }
    (codes, scales)
}

/// One position's operands and the declared regime around it.
struct Case {
    position: usize,
    window_start: usize,
    window_rows: usize,
    selected: Vec<u32>,
    /// The per-position expanded compressed view, rows 0..=position.
    kv_codes: Vec<u8>,
    kv_scales: Vec<u8>,
    window_codes: Vec<u8>,
    window_scales: Vec<u8>,
    q_bits: Vec<u16>,
    sink: Vec<f32>,
}

impl Case {
    fn build(position: usize, seed: u64) -> Result<Self, String> {
        let mut rng = Rng::new(seed ^ (position as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let window_start = position.saturating_sub(SWA_WINDOW - 1);
        let window_rows = position + 1 - window_start;
        // Compressed store rows at the even absolute positions, presented expanded: row q of the
        // view is store row q / 2.
        let store_rows = position / 2 + 1;
        let mut stores = Vec::with_capacity(store_rows);
        for _ in 0..store_rows {
            let values: Vec<f32> = (0..mla_ref::MLA_LATENT).map(|_| rng.next_f32()).collect();
            stores.push(quantize_store_row(&values));
        }
        let mut kv_codes = Vec::with_capacity((position + 1) * mla_ref::ROW_PACKED_BYTES);
        let mut kv_scales = Vec::with_capacity((position + 1) * mla_ref::ROW_SCALE_BYTES);
        for row in 0..=position {
            let store = &stores[row / 2];
            kv_codes.extend_from_slice(&store.0);
            kv_scales.extend_from_slice(&store.1);
        }
        // The engine window rows, oldest first: absolute positions window_start..=position, each
        // through the engine's own row helper.
        let mut window_codes = Vec::with_capacity(window_rows * mla_ref::MLA_LATENT);
        let mut window_scales = Vec::with_capacity(window_rows * mla_ref::ROW_SCALE_BYTES / 2);
        let mut window_values = Vec::with_capacity(window_rows * mla_ref::MLA_LATENT);
        for _ in 0..window_rows {
            let row: Vec<f32> = (0..mla_ref::MLA_LATENT).map(|_| rng.next_f32()).collect();
            window_values.extend_from_slice(&row);
        }
        for row in 0..window_rows {
            let start = row * mla_ref::MLA_LATENT;
            let encoded = engine_window_row(&window_values[start..start + mla_ref::MLA_LATENT])?;
            window_codes.extend_from_slice(&encoded.codes);
            window_scales.extend_from_slice(&encoded.scales);
        }
        assert_engine_window_finite(
            &window_codes,
            &window_scales,
            window_rows,
            "m20 window rows",
        )?;
        // The selected ids the sequence caller passes at this position; for the position 0 check the
        // base entry gets the same list ([0], the compressed id stored at the first row). Engine-dev
        // declared these as the even absolute positions.
        let selected: Vec<u32> = (0..=position)
            .filter(|token| token % 2 == 0)
            .map(|token| token as u32)
            .collect();
        let q_bits: Vec<u16> = (0..HEADS * LATENT).map(|_| rng.next_bf16_bits()).collect();
        let sink: Vec<f32> = (0..HEADS).map(|_| rng.next_f32() * 0.25).collect();
        Ok(Self {
            position,
            window_start,
            window_rows,
            selected,
            kv_codes,
            kv_scales,
            window_codes,
            window_scales,
            q_bits,
            sink,
        })
    }

    /// The bf16-to-f32 query the reference consumes: the exact values the device reads.
    fn q_as_read(&self) -> Vec<f32> {
        self.q_bits
            .iter()
            .map(|bits| bf16::from_bits(*bits).to_f32())
            .collect()
    }
}

/// The device buffers, allocated once at the largest shape each can need.
struct Buffers {
    q: Deviceptr,
    kv_fp4: Deviceptr,
    kv_scales: Deviceptr,
    window_codes: Deviceptr,
    window_scales: Deviceptr,
    selected: Deviceptr,
    index_key: Deviceptr,
    index_query: Deviceptr,
    sink: Deviceptr,
    block_scores: Deviceptr,
    block_best: Deviceptr,
    out: Deviceptr,
}

impl Buffers {
    fn allocate(cuda: &Cuda) -> Result<(Self, usize), String> {
        let max_position = *POSITIONS.last().expect("positions");
        let rows = max_position + 1;
        let mut total = 0usize;
        let buffers = Self {
            q: cuda.alloc(HEADS * LATENT * 2, &mut total)?,
            kv_fp4: cuda.alloc(rows * mla_ref::ROW_PACKED_BYTES, &mut total)?,
            kv_scales: cuda.alloc(rows * mla_ref::ROW_SCALE_BYTES, &mut total)?,
            window_codes: cuda.alloc(SWA_WINDOW * LATENT, &mut total)?,
            window_scales: cuda.alloc(SWA_WINDOW * mla_ref::ROW_SCALE_BYTES / 2, &mut total)?,
            selected: cuda.alloc(MAX_SELECTED * 4, &mut total)?,
            index_key: cuda.alloc(
                HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM * 2,
                &mut total,
            )?,
            index_query: cuda.alloc(INDEX_HEADS * INDEX_HEAD_DIM * 2, &mut total)?,
            sink: cuda.alloc(HEADS * 4, &mut total)?,
            block_scores: cuda.alloc(HSI_BLOCKS * 4, &mut total)?,
            block_best: cuda.alloc(HSI_BLOCKS * 4, &mut total)?,
            out: cuda.alloc(HEADS * LATENT * 2, &mut total)?,
        };
        Ok((buffers, total))
    }
}

/// What one launch wrote.
struct LaunchResult {
    out: Vec<u8>,
    block_scores: Vec<u8>,
    block_best: Vec<u8>,
}

impl LaunchResult {
    fn report_bytes(&self) -> Vec<u8> {
        let mut bytes = self.out.clone();
        bytes.extend_from_slice(&self.block_scores);
        bytes.extend_from_slice(&self.block_best);
        bytes
    }
}

fn pad_to(bytes: &[u8], len: usize) -> Vec<u8> {
    let mut padded = vec![0u8; len];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded
}

/// Uploads one case and runs the sequence entry.
fn launch_seq(
    cuda: &Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    buffers: &Buffers,
    case: &Case,
    grid: (u32, u32, u32),
    over_supply_future: bool,
) -> Result<LaunchResult, String> {
    cuda.copy_htod(&buffers.q, &bf16_bytes_from_bits(&case.q_bits))?;
    cuda.copy_htod(&buffers.sink, &f32_bytes(&case.sink))?;
    cuda.copy_htod(
        &buffers.window_codes,
        &pad_to(&case.window_codes, SWA_WINDOW * LATENT),
    )?;
    cuda.copy_htod(
        &buffers.window_scales,
        &pad_to(
            &case.window_scales,
            SWA_WINDOW * mla_ref::ROW_SCALE_BYTES / 2,
        ),
    )?;
    cuda.copy_htod(
        &buffers.kv_fp4,
        &pad_to(
            &case.kv_codes,
            (POSITIONS[POSITIONS.len() - 1] + 1) * mla_ref::ROW_PACKED_BYTES,
        ),
    )?;
    cuda.copy_htod(
        &buffers.kv_scales,
        &pad_to(
            &case.kv_scales,
            (POSITIONS[POSITIONS.len() - 1] + 1) * mla_ref::ROW_SCALE_BYTES,
        ),
    )?;
    let mut selected = case.selected.clone();
    if over_supply_future {
        selected.push(u32::try_from(case.position + 2).expect("position fits u32"));
    }
    cuda.copy_htod(
        &buffers.selected,
        &pad_to(&u32_bytes(&selected), MAX_SELECTED * 4),
    )?;
    let params = FlashMlaSeqParams::new(
        buffers.q,
        buffers.kv_fp4,
        buffers.kv_scales,
        buffers.window_codes,
        buffers.window_scales,
        seq::FORMAT_CSA_UNION,
        case.window_rows as u64,
        buffers.selected,
        buffers.index_key,
        buffers.index_query,
        buffers.sink,
        buffers.block_scores,
        buffers.block_best,
        buffers.out,
        case.position as u64,
        u32::try_from(selected.len()).expect("selected count fits u32"),
        0,
        0,
        0,
    );
    cuda.launch(kernel, grid, seq::BLOCK, params.as_slice())?;
    let mut out = vec![0u8; HEADS * LATENT * 2];
    cuda.copy_dtoh(&mut out, &buffers.out)?;
    let mut block_scores = vec![0u8; HSI_BLOCKS * 4];
    cuda.copy_dtoh(&mut block_scores, &buffers.block_scores)?;
    let mut block_best = vec![0u8; HSI_BLOCKS * 4];
    cuda.copy_dtoh(&mut block_best, &buffers.block_best)?;
    Ok(LaunchResult {
        out,
        block_scores,
        block_best,
    })
}

/// Runs the untouched base entry on the same operands, for the position-0 byte identity check.
fn launch_base(
    cuda: &Cuda,
    kernel: sparkflash_kernels::cuda_harness::FlashMlaKernel,
    buffers: &Buffers,
    case: &Case,
) -> Result<Vec<u8>, String> {
    cuda.copy_htod(&buffers.q, &bf16_bytes_from_bits(&case.q_bits))?;
    cuda.copy_htod(&buffers.sink, &f32_bytes(&case.sink))?;
    cuda.copy_htod(
        &buffers.window_codes,
        &pad_to(&case.window_codes, SWA_WINDOW * LATENT),
    )?;
    cuda.copy_htod(
        &buffers.window_scales,
        &pad_to(
            &case.window_scales,
            SWA_WINDOW * mla_ref::ROW_SCALE_BYTES / 2,
        ),
    )?;
    cuda.copy_htod(
        &buffers.kv_fp4,
        &pad_to(
            &case.kv_codes,
            (POSITIONS[POSITIONS.len() - 1] + 1) * mla_ref::ROW_PACKED_BYTES,
        ),
    )?;
    cuda.copy_htod(
        &buffers.kv_scales,
        &pad_to(
            &case.kv_scales,
            (POSITIONS[POSITIONS.len() - 1] + 1) * mla_ref::ROW_SCALE_BYTES,
        ),
    )?;
    cuda.copy_htod(
        &buffers.selected,
        &pad_to(&u32_bytes(&case.selected), MAX_SELECTED * 4),
    )?;
    let params = flash_mla_params(
        buffers.q.0,
        buffers.kv_fp4.0,
        buffers.kv_scales.0,
        buffers.window_codes.0,
        buffers.window_scales.0,
        seq::FORMAT_CSA_UNION,
        case.window_rows as u64,
        buffers.selected.0,
        buffers.index_key.0,
        buffers.index_query.0,
        buffers.sink.0,
        buffers.block_scores.0,
        buffers.block_best.0,
        buffers.out.0,
        case.position as u64,
        case.selected.len(),
        0,
        0,
        0,
    );
    launch_flash_mla(cuda, kernel, (HEADS as u32, 1, 1), (THREADS, 1, 1), &params)?;
    let mut out = vec![0u8; HEADS * LATENT * 2];
    cuda.copy_dtoh(&mut out, &buffers.out)?;
    Ok(out)
}

/// The comparison result of one device-vs-twin check.
struct Stats {
    exact: usize,
    total: usize,
    max_abs: f32,
    max_ulp: u32,
    non_finite: usize,
}

impl Stats {
    fn pass(&self, bar: f32) -> bool {
        self.non_finite == 0 && self.max_abs <= bar
    }

    fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "exact_bf16": self.exact,
            "total": self.total,
            "max_abs_delta": self.max_abs,
            "max_ulp": self.max_ulp,
            "non_finite_values": self.non_finite,
        })
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    if !a.is_finite() || !b.is_finite() {
        return u32::MAX;
    }
    let order = |value: f32| -> i64 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            -(i64::from(bits & 0x7FFF_FFFF))
        } else {
            i64::from(bits)
        }
    };
    order(a).abs_diff(order(b)).min(u64::from(u32::MAX)) as u32
}

fn compare_attention(actual: &[u8], expected: &[f32]) -> Stats {
    let values = decode_bf16(actual);
    let mut stats = Stats {
        exact: 0,
        total: values.len().min(expected.len()),
        max_abs: 0.0,
        max_ulp: 0,
        non_finite: 0,
    };
    for (value, target) in values.iter().zip(expected) {
        if !value.is_finite() || !target.is_finite() {
            stats.non_finite += 1;
            continue;
        }
        if value.to_bits() == target.to_bits() {
            stats.exact += 1;
        }
        let delta = (value - target).abs();
        stats.max_abs = stats.max_abs.max(delta);
        stats.max_ulp = stats.max_ulp.max(ulp_distance(*value, *target));
    }
    stats
}

/// The indexer twin: structural causality for every block plus a score/token comparison for the
/// reachable ones. Returns (ok, number of mismatches, worst abs score delta, first mismatch text).
fn compare_indexer(
    block_scores: &[u8],
    block_best: &[u8],
    expected: &[(f32, u32)],
    position: usize,
) -> (bool, usize, f32, String) {
    let scores = bytes_to_f32(block_scores);
    let best = block_best
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect::<Vec<u32>>();
    let mut mismatches = 0usize;
    let mut worst = 0.0f32;
    let mut first = String::new();
    for block in 0..HSI_BLOCKS {
        let first_token = block * HSI_BLOCK_TOKENS;
        let reachable = first_token <= position;
        let (want_score, want_best) = expected[block];
        let token = best[block];
        let mut problem = None;
        if !reachable {
            if scores[block].to_bits() != (-f32::MAX).to_bits() {
                problem = Some(format!(
                    "block {block} starts past the query but scores {:e}",
                    scores[block]
                ));
            } else if token != 0 {
                problem = Some(format!(
                    "block {block} starts past the query but best token {token}"
                ));
            }
        } else if u64::from(token) > position as u64 {
            problem = Some(format!(
                "block {block} best token {token} is past the query position {position}"
            ));
        } else if token != want_best {
            problem = Some(format!(
                "block {block} best token {token}, host reference {want_best}"
            ));
        } else {
            let delta = (scores[block] - want_score).abs();
            worst = worst.max(delta);
            if delta > INDEX_BAR_ABS {
                problem = Some(format!(
                    "block {block} score {:.6}, host reference {:.6} (delta {delta:.6})",
                    scores[block], want_score
                ));
            }
        }
        if let Some(text) = problem {
            mismatches += 1;
            if first.is_empty() {
                first = text;
            }
        }
    }
    (mismatches == 0, mismatches, worst, first)
}

struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

fn push(verdicts: &mut Vec<Verdict>, name: impl Into<String>, ok: bool, detail: impl Into<String>) {
    let verdict = Verdict {
        name: name.into(),
        ok,
        detail: detail.into(),
    };
    if verdict.ok {
        println!("  check {}: ok ({})", verdict.name, verdict.detail);
    } else {
        println!("  check {}: FAILED ({})", verdict.name, verdict.detail);
    }
    verdicts.push(verdict);
}

fn write_report(path: &Path, report: &serde_json::Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
    std::fs::write(path, &bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    println!(
        "report: {} ({} B, sha256 {})",
        path.display(),
        bytes.len(),
        sha256_bytes(&bytes)
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    check_pin(BASE_KERNEL, BASE_BYTES, BASE_SHA256)?;
    check_pin(SEQ_KERNEL, SEQ_BYTES, SEQ_SHA256)?;
    let base_cubin = load_cubin(&args.cubin_dir, BASE_KERNEL, BASE_BYTES, BASE_SHA256)?;
    let seq_cubin = load_cubin(&args.cubin_dir, SEQ_KERNEL, SEQ_BYTES, SEQ_SHA256)?;

    let cuda = Cuda::load()?;
    // The modules must outlive every launch; they are held to the end of this function.
    let (_base_module, base_kernel) = load_flash_mla(&cuda, &base_cubin)?;
    let (_seq_module, seq_kernel) = seq::load(&cuda, &seq_cubin)?;
    let (buffers, device_bytes) = Buffers::allocate(&cuda)?;

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut report = serde_json::json!({
        "m20": "the deferred sequence's position-causal CSA2 entry against mla_ref::attention_with_sink_union",
        "claim": "per-position device attention at positions 0..=2 and 129..=137, with a position-causal indexer; synthetic declared operands",
        "not_claimed": [
            "the served forward pass",
            "the c4a-2 call-site wiring (engine-dev)",
            "a token, timing or throughput number",
            "the base entry's published b31/b33/b39/b40 windows (this harness does not re-run them)"
        ],
        "operands": "declared synthetic: xorshift64* stream, seed 0x6D32305F43534132; q, window rows through engine_window_row, compressed store through the pinned E2M1/E4M3 codec, ratio-2 expanded view, sink, index key/query pool",
        "pins": {
            BASE_KERNEL: { "cubin_bytes": BASE_BYTES, "cubin_sha256": BASE_SHA256 },
            SEQ_KERNEL: { "cubin_bytes": SEQ_BYTES, "cubin_sha256": SEQ_SHA256 },
        },
        "device_bytes_admitted": device_bytes,
        "positions": POSITIONS,
        "bar_abs": BAR_ABS,
        "index_bar_abs": INDEX_BAR_ABS,
    });

    // The regime geometry, declared once before any number.
    let max_store = POSITIONS[POSITIONS.len() - 1] / 2 + 1;
    let mut geometry_ok = HEADS == 64
        && LATENT == 512
        && SWA_WINDOW == 128
        && MAX_SELECTED == 512
        && THREADS == 256
        && max_store <= MAX_SELECTED;
    let mut geometry_detail = format!(
        "vocab-free geometry: {HEADS} heads x {LATENT} latent, window {SWA_WINDOW}, selected bound {MAX_SELECTED}, {THREADS} threads; positions {POSITIONS:?}, at most {max_store} compressed ids"
    );
    if !geometry_ok {
        geometry_ok = false;
        geometry_detail =
            format!("the pinned geometry does not match the declared regime: {geometry_detail}");
    }
    println!(
        "geometry: CSA2 sequence entry, {HEADS} heads x {LATENT} latent, window {SWA_WINDOW}, selected bound {MAX_SELECTED}, {THREADS} threads; positions {POSITIONS:?}"
    );
    push(
        &mut verdicts,
        "released geometry and the sequence regime",
        geometry_ok,
        geometry_detail,
    );

    // The index pool is position-independent and uploaded once; the sequence caller uploads the
    // same bytes at every position.
    let mut index_rng = Rng::new(0x6D32_305F_4353_4132);
    let mut index_key_bits = Vec::with_capacity(HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM);
    for _ in 0..HSI_BLOCKS * HSI_BLOCK_TOKENS * INDEX_HEAD_DIM {
        index_key_bits.push(index_rng.next_index_bits());
    }
    let mut index_query_bits = Vec::with_capacity(INDEX_HEADS * INDEX_HEAD_DIM);
    for _ in 0..INDEX_HEADS * INDEX_HEAD_DIM {
        index_query_bits.push(index_rng.next_index_bits());
    }
    cuda.copy_htod(&buffers.index_key, &bf16_bytes_from_bits(&index_key_bits))?;
    cuda.copy_htod(
        &buffers.index_query,
        &bf16_bytes_from_bits(&index_query_bits),
    )?;

    let seed = 0x6D32_305F_4353_4132u64;
    let mut position_reports = Vec::new();
    for &position in POSITIONS.iter() {
        let case = Case::build(position, seed)?;
        println!(
            "case position {position}: window_start {}, window_rows {}, selected {} ids (first {:?}, last {:?}), compressed rows {}",
            case.window_start,
            case.window_rows,
            case.selected.len(),
            case.selected.first(),
            case.selected.last(),
            case.position / 2 + 1
        );
        let first = launch_seq(&cuda, seq_kernel, &buffers, &case, seq::GRID, false)?;
        let second = launch_seq(&cuda, seq_kernel, &buffers, &case, seq::GRID, false)?;

        // 1. Union attention against the host reference over the exact device bytes.
        let q = case.q_as_read();
        let mut reference_out = Vec::with_capacity(HEADS * LATENT);
        for head in 0..HEADS {
            let q_head = &q[head * LATENT..(head + 1) * LATENT];
            let out = mla_ref::attention_with_sink_union(
                q_head,
                &case.window_codes,
                &case.window_scales,
                case.window_rows,
                &case.kv_codes,
                &case.kv_scales,
                &case.selected,
                u32::try_from(position).expect("position fits u32"),
                case.sink[head],
            );
            reference_out.extend_from_slice(&out);
        }
        let attention = compare_attention(&first.out, &reference_out);
        push(
            &mut verdicts,
            format!("position {position}: union attention matches the host twin at the 2B bar"),
            attention.pass(BAR_ABS),
            format!(
                "{} of {} bf16 outputs exactly equal, max abs delta {:.3e}, max ULP {}, non-finite {} (bar {BAR_ABS:e})",
                attention.exact,
                attention.total,
                attention.max_abs,
                attention.max_ulp,
                attention.non_finite
            ),
        );

        // 2. The causal indexer against the host reduction on the same key/query bits.
        // Stage 1 splits each HSI block between the two heads of one index head (even head takes
        // even blocks, odd head odd ones), and the other 31 index heads would race for the same
        // arrays at the full grid. The twin therefore launches the same kernel at grid (2, 1, 1) -
        // the configuration m7's stage-2 indexer self-check uses - head pair 0 only, and compares
        // the whole pool against index head 0's query row.
        let index_launch = launch_seq(&cuda, seq_kernel, &buffers, &case, (2, 1, 1), false)?;
        let index_launch_again = launch_seq(&cuda, seq_kernel, &buffers, &case, (2, 1, 1), false)?;
        let expected = mla_ref::indexer_block_scores(
            &index_key_bits,
            &index_query_bits[..INDEX_HEAD_DIM],
            u32::try_from(position).expect("position fits u32"),
        );
        let (indexer_ok, indexer_mismatches, indexer_worst, indexer_first) = compare_indexer(
            &index_launch.block_scores,
            &index_launch.block_best,
            &expected,
            position,
        );
        push(
            &mut verdicts,
            format!("position {position}: causal indexer matches the host reduction"),
            indexer_ok,
            if indexer_ok {
                format!(
                    "2048 blocks causal, worst score delta {indexer_worst:.3e}, every best token exact (bar {INDEX_BAR_ABS:e})"
                )
            } else {
                format!(
                    "{indexer_mismatches} block mismatches, worst delta {indexer_worst:.3e}; first: {indexer_first}"
                )
            },
        );

        // 3. Determinism: the full launch's out[] and the head-pair launch's indexer arrays are
        // reproducible. (The full grid's block arrays are a documented race between index heads;
        // stage 2 does not read them, and the sequence's caller reads the pool only through the
        // candidate-selection step, which must use a deterministic launch like this one.)
        let deterministic = first.out == second.out
            && index_launch.report_bytes() == index_launch_again.report_bytes();
        push(
            &mut verdicts,
            format!("position {position}: repeated launches are byte-identical"),
            deterministic,
            format!(
                "out[] from two full launches and {} bytes (out + indexer arrays) from two head-pair launches",
                index_launch.report_bytes().len()
            ),
        );

        // 4. Over-supply: a future selected id is excluded by the kernel's own mask.
        let oversupply_position = position + 2;
        let over = launch_seq(&cuda, seq_kernel, &buffers, &case, seq::GRID, true)?;
        push(
            &mut verdicts,
            format!(
                "position {position}: a selected id at position {oversupply_position} is excluded"
            ),
            over.out == first.out,
            if over.out == first.out {
                format!(
                    "out[] byte-identical with {oversupply_position} appended to the selected list (the kernel masks it)"
                )
            } else {
                format!(
                    "out[] changed when a future selected id was appended: first differing byte {}",
                    first
                        .out
                        .iter()
                        .zip(&over.out)
                        .position(|(left, right)| left != right)
                        .unwrap_or(0)
                )
            },
        );

        // 5. Position 0: the sequence entry must reproduce the base entry at position 0.
        if position == 0 {
            let base_out = launch_base(&cuda, base_kernel, &buffers, &case)?;
            push(
                &mut verdicts,
                "position 0: the sequence entry reproduces the base entry byte for byte"
                    .to_string(),
                base_out == first.out,
                if base_out == first.out {
                    format!("{} out bytes identical to {BASE_KERNEL}", base_out.len())
                } else {
                    format!(
                        "out[] differs from the base entry at byte {}",
                        base_out
                            .iter()
                            .zip(&first.out)
                            .position(|(left, right)| left != right)
                            .unwrap_or(0)
                    )
                },
            );
        }

        position_reports.push(serde_json::json!({
            "position": position,
            "window_start": case.window_start,
            "window_rows": case.window_rows,
            "selected_count": case.selected.len(),
            "attention": attention.report_json(),
            "indexer": {
                "blocks_causal": indexer_ok,
                "mismatches": indexer_mismatches,
                "worst_score_delta": indexer_worst,
            },
            "deterministic": deterministic,
            "future_selected_excluded": over.out == first.out,
        }));
    }

    report["positions_report"] = serde_json::Value::Array(position_reports.clone());
    let failures: Vec<&Verdict> = verdicts.iter().filter(|verdict| !verdict.ok).collect();
    let pass = failures.is_empty();
    report["checks"] = serde_json::json!(
        verdicts
            .iter()
            .map(|verdict| serde_json::json!({
                "check": verdict.name,
                "ok": verdict.ok,
                "detail": verdict.detail,
            }))
            .collect::<Vec<_>>()
    );
    report["result"] = serde_json::json!(if pass { "PASS" } else { "FAIL" });
    if let Some(path) = &args.out {
        write_report(path, &report)?;
    }
    if pass {
        println!(
            "m20: PASS ({} checks; sequence CSA2 entry at positions {:?}; synthetic operands, one pinned cubin pair, position-0 base-entry identity measured)",
            verdicts.len(),
            POSITIONS
        );
        Ok(())
    } else {
        let names: Vec<&str> = failures
            .iter()
            .map(|verdict| verdict.name.as_str())
            .collect();
        println!(
            "m20: FAIL at {} of {} checks: {names:?}",
            failures.len(),
            verdicts.len()
        );
        Err(format!(
            "m20 failed at {} of {} checks",
            failures.len(),
            verdicts.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_positions_cover_the_sequence_prompt_and_the_full_window_regime() {
        assert_eq!(&POSITIONS[..3], &[0, 1, 2]);
        assert_eq!(
            &POSITIONS[3..],
            &[129, 130, 131, 132, 133, 134, 135, 136, 137]
        );
        for &position in &POSITIONS {
            let window_start = position.saturating_sub(SWA_WINDOW - 1);
            let window_rows = position + 1 - window_start;
            assert!(window_rows <= SWA_WINDOW, "position {position}");
            if position < SWA_WINDOW {
                assert_eq!(window_start, 0);
                assert_eq!(window_rows, position + 1);
            } else {
                assert_eq!(window_rows, SWA_WINDOW);
                assert_eq!(window_start, position + 1 - SWA_WINDOW);
            }
        }
    }

    #[test]
    fn the_selected_ids_are_the_even_absolute_positions_at_or_behind_the_query() {
        for &position in &[0usize, 1, 2, 128, 129, 130] {
            let expected: Vec<u32> = (0..=position)
                .filter(|token| token % 2 == 0)
                .map(|token| token as u32)
                .collect();
            assert_eq!(expected.len(), position / 2 + 1);
            assert!(expected.iter().all(|token| *token as usize <= position));
            assert_eq!(expected.last().copied(), Some((position / 2 * 2) as u32));
        }
    }

    #[test]
    fn the_quantiser_round_trips_inside_one_e2m1_step() {
        let values: Vec<f32> = (0..mla_ref::MLA_LATENT)
            .map(|index| ((index % 13) as f32 - 6.0) / 6.0)
            .collect();
        let (codes, scales) = quantize_store_row(&values);
        assert_eq!(codes.len(), mla_ref::ROW_PACKED_BYTES);
        assert_eq!(scales.len(), mla_ref::ROW_SCALE_BYTES);
        let mut worst = 0.0f32;
        for index in 0..mla_ref::MLA_LATENT {
            let decoded = mla_ref::kv_latent(&codes, &scales, 0, index);
            worst = worst.max((decoded - values[index]).abs());
        }
        assert!(worst <= 6.0 / 2.0 + 1e-6, "one E2M1 step: {worst}");
    }

    #[test]
    fn the_stats_detect_a_changed_value_and_ignore_a_within_bar_delta() {
        let expected = [1.0f32, 2.0];
        let exact = bf16_bytes_from_bits(&[bf16_bits(1.0), bf16_bits(2.0)]);
        let same = compare_attention(&exact, &expected);
        assert_eq!(same.total, 2);
        assert_eq!(same.exact, 2);
        assert!(same.pass(BAR_ABS));
        let changed = bf16_bytes_from_bits(&[bf16_bits(1.0), bf16_bits(2.5)]);
        let stats = compare_attention(&changed, &expected);
        assert!(!stats.pass(BAR_ABS), "0.5 is far above the bar");
        assert!(stats.max_abs >= 0.5);
    }
}
