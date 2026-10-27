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

//! The device vision tower against the reference's goldens (DECISIONS.md D59).
//!
//! The goldens come from the checkpoint's own vision.py and image_processor.py run under torch
//! (bf16, CUDA). Per image, `<name>.json` holds the grids and the image's file name,
//! `<name>.patches.bin` the patches (bf16 [patches, 588]), `<name>.block0.bin` the stream after the
//! first block and `<name>.vit.bin` the ViT output (bf16 [patches, 1024]), and `<name>.rows.bin`
//! the aligner rows (bf16 [rows, 5120]); `delims.bin` holds image_start, image_newline and
//! image_end (bf16 [3, 5120]).
//!
//! The canvas comes from sparkflash-vision's `prepare`, so the patch check covers the front end and
//! the patchify kernel together. It must be exact for a lossless file; a JPEG decodes within 4
//! levels of libjpeg-turbo (D59), so its patches are reported, not judged. The delimiter rows must
//! be exact too. The later
//! stages sum in the device's own order, so for them the harness reports the share of equal values,
//! the largest and mean absolute difference, the relative L2 error and the cosine similarity.
//!
//! Usage (on a node with the checkpoint):
//!
//! ```text
//! cargo run --release -p sparkflash-kernels --example vision_golden -- \
//!     --checkpoint DIR --cubin opt_vision_sm121.cubin --gold DIR --images DIR NAME...
//! ```

#![forbid(unsafe_code)]

use sparkflash_engine::safetensors::SafeTensorsSource;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::vision_tower::{
    DIM, Delimiter, PATCH_FEATURES, PATCH_STRIDE, Probe, ROW, VisionShape, VisionTower,
};
use sparkflash_vision::{ImageFormat, PrepOptions, prepare};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    checkpoint: PathBuf,
    cubin: PathBuf,
    gold: PathBuf,
    images: PathBuf,
    names: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vision_golden: {error}");
        std::process::exit(1);
    }
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let (mut checkpoint, mut cubin, mut gold, mut images) = (None, None, None, None);
    let mut names = Vec::new();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value()?)),
            "--cubin" => cubin = Some(PathBuf::from(value()?)),
            "--gold" => gold = Some(PathBuf::from(value()?)),
            "--images" => images = Some(PathBuf::from(value()?)),
            _ if arg.starts_with("--") => return Err(format!("unknown flag {arg}")),
            _ => names.push(arg),
        }
    }
    let need = |v: Option<PathBuf>, flag: &str| v.ok_or_else(|| format!("{flag} is required"));
    Ok(Args {
        checkpoint: need(checkpoint, "--checkpoint")?,
        cubin: need(cubin, "--cubin")?,
        gold: need(gold, "--gold")?,
        images: need(images, "--images")?,
        names,
    })
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let cuda = Cuda::load()?;
    let index = args.checkpoint.join("model.safetensors.index.json");
    let source = SafeTensorsSource::open(&args.checkpoint, &index)
        .map_err(|error| format!("checkpoint source: {error}"))?;
    let cubin = read(&args.cubin)?;
    let started = Instant::now();
    let mut tower = VisionTower::load(&cuda, &cubin, &source)?;
    println!(
        "loaded {:.0} MB of weights in {:.1} s",
        tower.device_bytes() as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );

    let delims = bf16s(&read(&args.gold.join("delims.bin"))?);
    for (i, which) in [Delimiter::Start, Delimiter::NewLine, Delimiter::End]
        .into_iter()
        .enumerate()
    {
        let got = download(&cuda, tower.delimiter(which), ROW)?;
        if got[..] != delims[i * ROW..(i + 1) * ROW] {
            return Err(format!("delimiter {which:?} differs from the reference"));
        }
    }
    println!("delimiters: equal");

    let mut failed = false;
    for name in &args.names {
        let meta_text = std::fs::read_to_string(args.gold.join(format!("{name}.json")))
            .map_err(|error| format!("{name}.json: {error}"))?;
        let meta: serde_json::Value =
            serde_json::from_str(&meta_text).map_err(|error| format!("{name}.json: {error}"))?;
        let field = |key: &str| {
            meta[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("{name}.json: no {key}"))
        };
        let image = meta["image"]
            .as_str()
            .ok_or_else(|| format!("{name}.json: no image"))?;
        let prepared = prepare(&read(&args.images.join(image))?, &PrepOptions::default())
            .map_err(|error| format!("{image}: {error}"))?;
        let (height, width) = (prepared.grid.best_height, prepared.grid.best_width);
        let shape = VisionShape::of(height, width)?;
        let expected = VisionShape {
            n_vit_h: field("n_vit_h")?,
            n_vit_w: field("n_vit_w")?,
            n_llm_h: field("n_llm_h")?,
            n_llm_w: field("n_llm_w")?,
        };
        if shape != expected {
            return Err(format!("{name}: grid {shape:?}, reference {expected:?}"));
        }
        let (n, rows) = (shape.patches(), shape.rows());
        println!(
            "{name}: {} x {} canvas, {n} patches, {rows} rows",
            width, height
        );

        tower.run(&cuda, &prepared.pixels, height, width, 0, None)?;
        let patches = download(&cuda, probe(&tower, Probe::Patches)?, n * PATCH_STRIDE)?;
        let reference = bf16s(&read(&args.gold.join(format!("{name}.patches.bin")))?);
        let mut mismatched = 0usize;
        for (patch, got) in patches.as_chunks::<PATCH_STRIDE>().0.iter().enumerate() {
            let want = &reference[patch * PATCH_FEATURES..(patch + 1) * PATCH_FEATURES];
            mismatched += got[..PATCH_FEATURES]
                .iter()
                .zip(want)
                .filter(|(a, b)| a != b)
                .count();
            mismatched += got[PATCH_FEATURES..].iter().filter(|v| **v != 0).count();
        }
        let lossy = prepared.format == ImageFormat::Jpeg;
        println!(
            "  patches: {mismatched} values differ{}",
            if lossy {
                " (JPEG: reported, not judged)"
            } else {
                ""
            }
        );
        failed |= mismatched != 0 && !lossy;

        tower.run(&cuda, &prepared.pixels, height, width, 1, None)?;
        let block0 = download(&cuda, probe(&tower, Probe::Stream)?, n * DIM)?;
        report(
            "block0",
            &block0,
            &bf16s(&read(&args.gold.join(format!("{name}.block0.bin")))?),
        );

        let mut total = 0usize;
        let out = cuda.alloc(rows * ROW * 2, &mut total)?;
        let mut times = Vec::new();
        for _ in 0..3 {
            cuda.synchronize()?;
            let started = Instant::now();
            tower.encode(&cuda, &prepared.pixels, height, width, out)?;
            cuda.synchronize()?;
            times.push(started.elapsed().as_secs_f64() * 1e3);
        }
        let vit = download(&cuda, probe(&tower, Probe::Vit)?, n * DIM)?;
        let got_rows = download(&cuda, out, rows * ROW)?;
        cuda.free_buffers(&[&out]);
        report(
            "vit",
            &vit,
            &bf16s(&read(&args.gold.join(format!("{name}.vit.bin")))?),
        );
        report(
            "rows",
            &got_rows,
            &bf16s(&read(&args.gold.join(format!("{name}.rows.bin")))?),
        );
        println!(
            "  encode ms: {}",
            times
                .iter()
                .map(|t| format!("{t:.1}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    tower.free(&cuda);
    if failed {
        return Err("the patches are not the reference's".into());
    }
    Ok(())
}

fn probe(tower: &VisionTower, which: Probe) -> Result<Deviceptr, String> {
    tower
        .probe(which)
        .ok_or_else(|| format!("no {which:?} after the run"))
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn bf16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

fn download(cuda: &Cuda, src: Deviceptr, values: usize) -> Result<Vec<u16>, String> {
    let mut bytes = vec![0u8; values * 2];
    cuda.copy_dtoh(&mut bytes, &src)?;
    Ok(bf16s(&bytes))
}

fn widen(v: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(v) << 16))
}

/// One line comparing a device tensor with the reference's.
fn report(stage: &str, got: &[u16], want: &[u16]) {
    if got.len() != want.len() {
        println!("  {stage}: {} values, reference {}", got.len(), want.len());
        return;
    }
    let (mut equal, mut max_abs, mut sum_abs) = (0usize, 0f64, 0f64);
    let (mut dot, mut got_sq, mut want_sq, mut diff_sq) = (0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        equal += usize::from(a == b);
        let (x, y) = (widen(a), widen(b));
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        dot += x * y;
        got_sq += x * x;
        want_sq += y * y;
        diff_sq += d * d;
    }
    let count = got.len() as f64;
    println!(
        "  {stage}: equal {:.4}  max abs {max_abs:.4}  mean abs {:.6}  rel L2 {:.6}  cos {:.8}",
        equal as f64 / count,
        sum_abs / count,
        (diff_sq / want_sq).sqrt(),
        dot / (got_sq.sqrt() * want_sq.sqrt())
    );
}
