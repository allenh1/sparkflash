// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Serve one prompt through the device arm on the real checkpoint, with the tower and the
//! drafter loaded when the checkpoint has them.
//!
//! `cargo run --release -p sparkflash-model-inkling2 --example device-prefill -- DIR IDS...`
//!
//! The device uploads the checkpoint's tensors directly, so no converted rank file is needed.

use std::path::Path;
use std::sync::Arc;

use sparkflash_model::backend::{ModelBackend, Sampling};
use sparkflash_model::vision::{ImageGrid, ImageSpan, VisionImage};
use sparkflash_model_inkling2::Inkling2Config;
use sparkflash_model_inkling2::device::InklingDeviceBackend;
use sparkflash_model_inkling2::vision::IMAGE_TOKEN_ID;
use sparkflash_weights::WeightSource;
use sparkflash_weights::safetensors::{CheckpointIndex, SafeTensorsSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: device-prefill DIR TOKEN_ID [TOKEN_ID ...]")?;
    let tokens: Vec<u32> = args
        .map(|argument| argument.parse::<u32>())
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("give at least one token id".into());
    }
    let directory = Path::new(&dir);
    let config = Inkling2Config::from_checkpoint_dir(directory)?;
    let index = CheckpointIndex::load(&directory.join("model.safetensors.index.json"))?;
    let source: Arc<dyn WeightSource> = Arc::new(SafeTensorsSource::new(directory, index));
    let mut backend = InklingDeviceBackend::open(
        sparkflash_model_inkling2::descriptor(),
        config.clone(),
        Arc::clone(&source),
        16,
    )?;
    if config.vision.is_some() {
        let bytes = backend.load_vision(&*source)?;
        eprintln!("[example] vision tower on rank 0: {bytes} bytes");
    }
    if config.mtp.is_some() && backend.wants_drafter() {
        let bytes = backend.load_drafter(directory)?;
        eprintln!("[example] drafter: {bytes} bytes");
    }
    let mut session = backend.open_session()?;
    session.set_sampling(Sampling {
        temperature: 0.0,
        top_k: 1,
        ..Sampling::default()
    });
    // A synthetic 80x80 canvas takes the tower's four cells: a start delimiter, four
    // placeholders and the end delimiter. The pixels are a gradient, so the image path's work is
    // visible in the logits without depending on an image file.
    let (tokens, images) = if std::env::var_os("SPARKFLASH_EXAMPLE_IMAGE").is_some() {
        let (height, width) = (80usize, 80usize);
        let mut pixels = vec![0u8; height * width * 3];
        for y in 0..height {
            for x in 0..width {
                let at = (y * width + x) * 3;
                pixels[at] = (x * 3) as u8;
                pixels[at + 1] = (y * 3) as u8;
                pixels[at + 2] = ((x + y) * 2) as u8;
            }
        }
        let mut tokens = vec![10u32, 200_005];
        tokens.extend([IMAGE_TOKEN_ID; 4]);
        tokens.extend([200_010, 11]);
        let image = VisionImage {
            span: ImageSpan {
                start: 1,
                grid: ImageGrid {
                    n_llm_h: 2,
                    n_llm_w: 2,
                    best_height: height,
                    best_width: width,
                    line_stride: 2,
                },
                key: [7; 32],
            },
            pixels: Arc::from(pixels),
        };
        (tokens, vec![image])
    } else {
        (tokens, Vec::new())
    };
    let first = session.prefill_images(&tokens, &images)?;
    let mut top: Vec<(usize, f32)> = first.logits.iter().copied().enumerate().collect();
    top.sort_by(|left, right| right.1.total_cmp(&left.1));
    println!("next {:?}", first.next_token);
    for (token, score) in top.iter().take(8) {
        println!("{token}\t{score:.6}");
    }
    // A round's drafting runs in decode, so a few steps show the drafter's path. The counts are
    // what the arm's own ledgers print; this example only proves the rounds complete.
    let decode: usize = std::env::var("SPARKFLASH_EXAMPLE_DECODE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let start = std::time::Instant::now();
    let mut token = first.next_token;
    for _ in 0..decode {
        let Some(next) = token else {
            println!("the model ended");
            break;
        };
        let step = session.decode(next)?;
        println!(
            "decoded {:?} (drafts {})",
            step.next_token,
            step.draft.len()
        );
        token = step.next_token;
    }
    if decode > 0 {
        println!(
            "{decode} decode steps in {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(())
}
