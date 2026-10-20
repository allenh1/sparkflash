//! m24_lm_reduce_batched: K8 LM-head reduce row independence and six-row launch cost.
//!
//! Declared before any number. Synthetic operands at the kernel's fixed vocabulary shape: four
//! rank shards of 32,320 stored logits + certified flags per row, six rows. The scalar symbol is
//! launched once per row and its seven outputs are read; the batched symbol then reduces all six
//! rows in one launch; every output of every row is byte-compared for equality. Outputs compared:
//! token, value, block_max[128], block_argmax[128], top5_value[5], top5_index[5], status and
//! certified_count.
//!
//! This proves per-row independence for LM-head reduce. It has no L parameter (vocabulary fixed at
//! 129,280), so the {37,129,517,4097} gate does not apply; the fixed six-row FLOOR is the batched
//! symbol's contract (it reads six batch-major rows and cannot be called with one). The scalar
//! symbol remains the one-row entry.
//!
//! Usage: m24_lm_reduce_batched --cubin-dir out [--repeat 50]

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::Instant;

use sha2::{Digest, Sha256};
use sparkflash_kernels::baseline;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::lm_head_reduce_launch::{
    self, BATCH_ROWS, BLOCKS, LmHeadReduceParams, RANKS, TOP, VOCAB_PER_RANK,
};

const KERNEL: &str = "lm_head_reduce_sm121";
const PIN_BYTES: usize = 73_704;
const PIN_SHA256: &str = "2d18f68d71fbe657ed8e675f3b74c386678d51ff350a82368fff0df50c0032d3";
const PIN_MEMBAR_LINES: u32 = 0;
const ROWS: usize = 6;

fn main() {
    if let Err(error) = run() {
        eprintln!("m24_lm_reduce_batched: {error}");
        std::process::exit(1);
    }
}

struct Args {
    cubin_dir: PathBuf,
    repeat: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut cubin_dir = None;
    let mut repeat = 50usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--cubin-dir" => {
                cubin_dir = Some(PathBuf::from(
                    args.next().ok_or("missing --cubin-dir value")?,
                ))
            }
            "--repeat" => {
                repeat = args
                    .next()
                    .ok_or("missing --repeat value")?
                    .parse()
                    .map_err(|_| "bad --repeat".to_owned())?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        cubin_dir: cubin_dir.ok_or("--cubin-dir is required")?,
        repeat,
    })
}

fn load_checked(cubin_dir: &std::path::Path) -> Result<Vec<u8>, String> {
    let path = cubin_dir.join(format!("{KERNEL}.cubin"));
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let pin = baseline::pin(KERNEL)?
        .ok_or_else(|| format!("{KERNEL} is not in the baseline manifest"))?;
    if bytes.len() != PIN_BYTES || pin.cubin_bytes != PIN_BYTES {
        return Err(format!(
            "{KERNEL}: cubin is {} B but the pin is {PIN_BYTES} B",
            bytes.len()
        ));
    }
    let sha = format!("{:x}", Sha256::digest(&bytes));
    if sha != PIN_SHA256
        || pin.cubin_sha256 != PIN_SHA256
        || pin.ptx_membar_lines != PIN_MEMBAR_LINES
    {
        return Err(format!(
            "{KERNEL}: cubin pin mismatch ({sha}, {} membar lines)",
            pin.ptx_membar_lines
        ));
    }
    Ok(bytes)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f32(&mut self) -> f32 {
        (self.next() as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0
    }
}

fn bytes_of_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read(cuda: &Cuda, pointer: Deviceptr, length: usize) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; length];
    cuda.copy_dtoh(&mut bytes, &pointer)?;
    Ok(bytes)
}

struct Buffers {
    logits: [Deviceptr; RANKS],
    certified: [Deviceptr; RANKS],
    token: Deviceptr,
    value: Deviceptr,
    block_max: Deviceptr,
    block_argmax: Deviceptr,
    top5_value: Deviceptr,
    top5_index: Deviceptr,
    status: Deviceptr,
    certified_count: Deviceptr,
}

impl Buffers {
    fn params_for_row(&self, row: usize) -> LmHeadReduceParams {
        let mut logits = [Deviceptr(0); RANKS];
        let mut certified = [Deviceptr(0); RANKS];
        for rank in 0..RANKS {
            logits[rank] = self.logits[rank].offset(row * VOCAB_PER_RANK * 4);
            certified[rank] = self.certified[rank].offset(row * VOCAB_PER_RANK);
        }
        LmHeadReduceParams::new(
            logits,
            certified,
            self.token.offset(row * 4),
            self.value.offset(row * 4),
            self.block_max.offset(row * BLOCKS * 4),
            self.block_argmax.offset(row * BLOCKS * 4),
            self.top5_value.offset(row * TOP * 4),
            self.top5_index.offset(row * TOP * 4),
            self.status.offset(row * 4),
            self.certified_count.offset(row * 4),
            0,
            0,
            0,
        )
    }
    fn params_for_batch(&self) -> LmHeadReduceParams {
        LmHeadReduceParams::new(
            self.logits,
            self.certified,
            self.token,
            self.value,
            self.block_max,
            self.block_argmax,
            self.top5_value,
            self.top5_index,
            self.status,
            self.certified_count,
            0,
            0,
            0,
        )
    }
    fn read_row(&self, cuda: &Cuda, row: usize) -> Result<Vec<Vec<u8>>, String> {
        Ok(vec![
            read(cuda, self.token.offset(row * 4), 4)?,
            read(cuda, self.value.offset(row * 4), 4)?,
            read(cuda, self.block_max.offset(row * BLOCKS * 4), BLOCKS * 4)?,
            read(cuda, self.block_argmax.offset(row * BLOCKS * 4), BLOCKS * 4)?,
            read(cuda, self.top5_value.offset(row * TOP * 4), TOP * 4)?,
            read(cuda, self.top5_index.offset(row * TOP * 4), TOP * 4)?,
            read(cuda, self.status.offset(row * 4), 4)?,
            read(cuda, self.certified_count.offset(row * 4), 4)?,
        ])
    }
}

fn p50(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let bytes = load_checked(&args.cubin_dir)?;
    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let mut logits = [Deviceptr(0); RANKS];
    let mut certified = [Deviceptr(0); RANKS];
    for rank in 0..RANKS {
        logits[rank] = cuda.alloc(ROWS * VOCAB_PER_RANK * 4, &mut allocated)?;
        certified[rank] = cuda.alloc(ROWS * VOCAB_PER_RANK, &mut allocated)?;
    }
    let buffers = Buffers {
        logits,
        certified,
        token: cuda.alloc(ROWS * 4, &mut allocated)?,
        value: cuda.alloc(ROWS * 4, &mut allocated)?,
        block_max: cuda.alloc(ROWS * BLOCKS * 4, &mut allocated)?,
        block_argmax: cuda.alloc(ROWS * BLOCKS * 4, &mut allocated)?,
        top5_value: cuda.alloc(ROWS * TOP * 4, &mut allocated)?,
        top5_index: cuda.alloc(ROWS * TOP * 4, &mut allocated)?,
        status: cuda.alloc(ROWS * 4, &mut allocated)?,
        certified_count: cuda.alloc(ROWS * 4, &mut allocated)?,
    };
    println!(
        "m24_lm_reduce_batched: synthetic vocab {} rows {} (fixed shape); allocated {allocated} B",
        RANKS * VOCAB_PER_RANK,
        ROWS
    );

    let mut rng = Rng(0x1ce5_5eed_2026_1019);
    for rank in 0..RANKS {
        let mut values = Vec::with_capacity(ROWS * VOCAB_PER_RANK);
        for _ in 0..ROWS * VOCAB_PER_RANK {
            values.push(rng.f32());
        }
        cuda.copy_htod(&logits[rank], &bytes_of_f32(&values))?;
        let flags: Vec<u8> = (0..ROWS * VOCAB_PER_RANK)
            .map(|index| u8::from(index % 97 == 0))
            .collect();
        cuda.copy_htod(&certified[rank], &flags)?;
    }
    println!("operands uploaded: no checkpoint bytes, no H2D inside any timed region");

    let (_module, scalar_kernel): (_, Kernel) = lm_head_reduce_launch::load(&cuda, &bytes)?;
    let (_batch_module, batch_kernel) = lm_head_reduce_launch::load_batched(&cuda, &bytes)?;
    assert_eq!(BATCH_ROWS as usize, ROWS);

    let t = Instant::now();
    for row in 0..ROWS {
        lm_head_reduce_launch::launch(&cuda, scalar_kernel, &buffers.params_for_row(row))?;
    }
    let first_scalar_ms = t.elapsed().as_secs_f64() * 1e3;
    let scalar_rows: Vec<Vec<Vec<u8>>> = (0..ROWS)
        .map(|row| buffers.read_row(&cuda, row))
        .collect::<Result<_, _>>()?;

    let params = buffers.params_for_batch();
    let t = Instant::now();
    lm_head_reduce_launch::launch_batched(&cuda, batch_kernel, &params)?;
    let first_batched_ms = t.elapsed().as_secs_f64() * 1e3;

    let names = [
        "token",
        "value",
        "block_max",
        "block_argmax",
        "top5_value",
        "top5_index",
        "status",
        "certified_count",
    ];
    let mut all_exact = true;
    for (row, scalar_row) in scalar_rows.iter().enumerate() {
        let batched = buffers.read_row(&cuda, row)?;
        for (index, name) in names.iter().enumerate() {
            if scalar_row[index] != batched[index] {
                all_exact = false;
                println!("row {row} {name}: MISMATCH");
            }
        }
    }
    println!(
        "per-row byte equality (scalar row r vs batched row r): {}",
        if all_exact { "EXACT" } else { "FAILED" }
    );

    let mut scalar_samples = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        let t = Instant::now();
        for row in 0..ROWS {
            lm_head_reduce_launch::launch(&cuda, scalar_kernel, &buffers.params_for_row(row))?;
        }
        scalar_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut batched_samples = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        let t = Instant::now();
        lm_head_reduce_launch::launch_batched(&cuda, batch_kernel, &params)?;
        batched_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let scalar_p50 = p50(scalar_samples);
    let batched_p50 = p50(batched_samples);
    println!("six rows: scalar 6 launches p50 {scalar_p50:.4} ms (first {first_scalar_ms:.4} ms)");
    println!("six rows: batched 1 launch p50 {batched_p50:.4} ms (first {first_batched_ms:.4} ms)");
    println!(
        "launch ratio: {:.2}x; fixed six-row floor: the batched symbol reads six rows and cannot be called with one",
        scalar_p50 / batched_p50
    );
    println!(
        "LM reduce has no L parameter (vocab fixed {}): the row-count gate applies to flash_mla_csa2_seq, not this kernel",
        RANKS * VOCAB_PER_RANK
    );

    cuda.free_buffers(&[
        &buffers.logits[0],
        &buffers.logits[1],
        &buffers.logits[2],
        &buffers.logits[3],
        &buffers.certified[0],
        &buffers.certified[1],
        &buffers.certified[2],
        &buffers.certified[3],
        &buffers.token,
        &buffers.value,
        &buffers.block_max,
        &buffers.block_argmax,
        &buffers.top5_value,
        &buffers.top5_index,
        &buffers.status,
        &buffers.certified_count,
    ]);
    if !all_exact {
        return Err("row independence is not byte-exact".to_owned());
    }
    Ok(())
}
