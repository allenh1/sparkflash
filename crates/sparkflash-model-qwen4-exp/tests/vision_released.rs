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

//! The released vision tower, loaded from the real checkpoint.
//!
//! Ignored by default because the checkpoint is 132.6 GB and this host may not have it. Run with
//! SPARKFLASH_QWEN4_CHECKPOINT pointing at the checkpoint directory:
//!
//!   SPARKFLASH_QWEN4_CHECKPOINT=~/qwen38 cargo test -p sparkflash-model-qwen4-exp \
//!       --test vision_released -- --ignored --nocapture

use std::path::PathBuf;

use sparkflash_model_qwen4_exp::vision::patchify;
use sparkflash_model_qwen4_exp::{Qwen4ExpConfig, Qwen4Vision, VisionTowerConfig};

#[test]
#[ignore = "needs the released Qwen3.8-Flash-Next checkpoint"]
fn the_released_tower_loads_and_encodes_one_image() {
    let directory = std::env::var_os("SPARKFLASH_QWEN4_CHECKPOINT")
        .map(PathBuf::from)
        .expect("run the ignored test with SPARKFLASH_QWEN4_CHECKPOINT pointing at the checkpoint");
    let config = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
    let tower = VisionTowerConfig::from_config(&config).expect("the config has a vision tower");
    let vision = Qwen4Vision::load(&directory, &config).expect("the released tower loads");
    assert_eq!(vision.config(), &tower);

    // A 256x256 canvas is 16x16 patches, a 2x2 merge grid of 8x8 tokens, and exercises the
    // learned-position resampling away from the 48x48 source grid.
    let width = 256;
    let height = 256;
    let pixels: Vec<u8> = (0..width * height)
        .flat_map(|index| {
            let x = (index % width) as u8;
            let y = (index / width) as u8;
            [x, y, x.wrapping_add(y)]
        })
        .collect();
    let patches = patchify(&pixels, width, height, &tower);
    let rows = vision
        .encode(
            &patches,
            height / tower.patch_size,
            width / tower.patch_size,
        )
        .expect("the released tower encodes");
    assert_eq!(
        rows.len(),
        (height / tower.cell()) * (width / tower.cell()) * tower.out_hidden_size
    );
    assert!(rows.iter().all(|value| value.is_finite()));
    assert!(rows.iter().any(|value| *value != 0.0));
    eprintln!(
        "encoded {} rows of {}",
        rows.len() / tower.out_hidden_size,
        tower.out_hidden_size
    );
}
