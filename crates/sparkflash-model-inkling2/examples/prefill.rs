// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Run the f32 host reference on a checkpoint directory and print the next-token scores.
//!
//! `cargo run --release -p sparkflash-model-inkling2 --example prefill -- DIR 1 2 3`
//!
//! The reference reads a tensor when a step needs it, so this is the correctness path, not a
//! throughput path: a released checkpoint takes minutes a token on a CPU. It is how the loop
//! schedule and the message path are checked on the real weights before the device walk runs
//! them.

use std::path::Path;
use std::sync::Arc;

use sparkflash_model_inkling2::{Inkling2Config, Inkling2Weights};
use sparkflash_weights::safetensors::{CheckpointIndex, SafeTensorsSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: prefill DIR TOKEN_ID [TOKEN_ID ...]")?;
    let tokens: Vec<u32> = args
        .map(|argument| argument.parse::<u32>())
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("give at least one token id".into());
    }
    let directory = Path::new(&dir);
    let config = Inkling2Config::from_checkpoint_dir(directory)?;
    let index = CheckpointIndex::load(&directory.join("model.safetensors.index.json"))?;
    let source = Arc::new(SafeTensorsSource::new(directory, index));
    let weights = Inkling2Weights::load(config.clone(), source)?;
    let logits = weights.forward_prompt(&tokens)?;
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|left, right| right.1.total_cmp(&left.1));
    for (token, score) in top.iter().take(8) {
        println!("{token}\t{score:.6}");
    }
    Ok(())
}
