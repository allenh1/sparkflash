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

//! The routed-experts few-row bodies against the grouped tiles they must reproduce:
//! `q4p_moe_rows_gate_up_capped`/`q4p_moe_rows_down` (NVFP4, Inkling's interleaved w13 and
//! softcapped SwiGLU) and `q4p_moe_rows_gate_up_capped_mx8`/`q4p_moe_rows_down_mx8` (MXFP8),
//! against `q4p_moe_gate_up_capped`/`q4p_moe_down` and their `_mx8` forms.
//!
//! Both walk the same plan (one item, one expert) over the same weights, activations and routing
//! weights. The gate/up output's two bf16 planes and the down pass's f32 partials must agree
//! word for word at every token count in 1, 2, 4 and 8. Ignored by default (it needs a CUDA
//! device); run on the CUDA host:
//!
//!   cargo test -p sparkflash-kernels --test expert_rows_device -- --ignored --nocapture

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::images::qwen4_prefill_cubin;
use sparkflash_kernels::m4_quant::fp8_e4m3_encode;
use sparkflash_kernels::test_io::{self, download, upload};

const TOKENS: [usize; 4] = [1, 2, 4, 8];
const TOPK: usize = 2;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One stored body: the six table pointers a plan item names and the three outer scales.
struct Body {
    codes: Vec<u8>,
    scales: Vec<u8>,
    down_codes: Vec<u8>,
    down_scales: Vec<u8>,
    outer: [f32; 3],
    code_row: usize,
    scale_row: usize,
}

/// The NVFP4 form: an E2M1 nibble a column, one E4M3 block scale a 16 columns, the w13 rows
/// interleaved (row 2c the gate of column c, 2c + 1 its up).
fn nvfp4_body(hidden: usize, intermediate: usize, mut stream: Lcg) -> Body {
    let code_row = hidden / 2;
    let scale_row = hidden / 16;
    let down_code_row = intermediate / 2;
    let down_scale_row = intermediate / 16;
    let codes: Vec<u8> = (0..2 * intermediate * code_row)
        .map(|at| ((at as u64 * 29 + 7) % 251) as u8)
        .collect();
    let scales: Vec<u8> = (0..2 * intermediate * scale_row)
        .map(|at| fp8_e4m3_encode(0.5 + ((at as u64 * 11 + 3) % 13) as f32 * 0.0625))
        .collect();
    let down_codes: Vec<u8> = (0..hidden * down_code_row)
        .map(|at| ((at as u64 * 17 + 5) % 251) as u8)
        .collect();
    let down_scales: Vec<u8> = (0..hidden * down_scale_row)
        .map(|at| fp8_e4m3_encode(0.5 + ((at as u64 * 7 + 1) % 13) as f32 * 0.0625))
        .collect();
    let _ = &mut stream;
    Body {
        codes,
        scales,
        down_codes,
        down_scales,
        outer: [1.0, 1.0, 1.0],
        code_row,
        scale_row,
    }
}

fn upload_body(cuda: &Cuda, body: &Body) -> (Deviceptr, Deviceptr, [u64; 6]) {
    let codes = upload::<u8>(cuda, &body.codes);
    let scales = upload::<u8>(cuda, &body.scales);
    let down_codes = upload::<u8>(cuda, &body.down_codes);
    let down_scales = upload::<u8>(cuda, &body.down_scales);
    let outer = upload::<u8>(cuda, &f32_bytes(&body.outer));
    let table = [
        codes.0,
        scales.0,
        codes.offset(body.code_row).0,
        scales.offset(body.scale_row).0,
        down_codes.0,
        down_scales.0,
    ];
    (outer, codes, table)
}

/// The tile's and the rows body's gate/up output, and the down output over it, at one token
/// count; every word must agree.
#[allow(clippy::too_many_arguments)]
fn check(
    cuda: &Cuda,
    label: &str,
    tile_gate_up: Kernel,
    rows_gate_up: Kernel,
    tile_down: Kernel,
    rows_down: Kernel,
    body: &Body,
    hidden: usize,
    intermediate: usize,
    gate_cap: f32,
    up_cap: f32,
    count: usize,
    down_columns: (usize, usize),
) {
    let slots = count * TOPK;
    let tokens = count;
    let (outer, _codes, table) = upload_body(cuda, body);
    let table = upload::<u8>(
        cuda,
        &table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    // One item: expert 0 holds every slot, in ascending order.
    let sorted: Vec<i32> = (0..slots as i32).collect();
    let sorted = upload::<u8>(
        cuda,
        &sorted
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let items = [(0i32, 0i32, slots as i32)];
    let items = upload::<u8>(
        cuda,
        &items
            .iter()
            .flat_map(|&(x, y, z)| {
                [
                    x.to_le_bytes(),
                    y.to_le_bytes(),
                    z.to_le_bytes(),
                    0i32.to_le_bytes(),
                ]
                .concat()
            })
            .collect::<Vec<_>>(),
    );
    let item_count = upload::<i32>(cuda, &[1]);
    let routes = upload::<f32>(cuda, &vec![1.0; slots]);
    let activations = upload::<f32>(cuda, &{
        let mut stream = Lcg(0xa11ce);
        (0..tokens * hidden)
            .map(|_| stream.next())
            .collect::<Vec<f32>>()
    });
    let planes = upload::<u8>(cuda, &vec![0u8; 2 * tokens * hidden * 2]);
    let (_split_module, split) = cuda
        .load_named(qwen4_prefill_cubin(), "q4p_split_bf16")
        .expect("q4p_split_bf16");
    cuda.launch(
        split,
        (((tokens * hidden / 2).div_ceil(256)) as u32, 1, 1),
        (256, 1, 1),
        &[activations.0, planes.0, (tokens * hidden) as u64],
    )
    .expect("the split");
    let tile_head = upload::<u8>(cuda, &vec![0u8; 2 * slots * intermediate * 2]);
    let rows_head = upload::<u8>(cuda, &vec![0u8; 2 * slots * intermediate * 2]);
    let tile_down_out = upload::<u8>(cuda, &vec![0u8; slots * hidden * 4]);
    let rows_down_out = upload::<u8>(cuda, &vec![0u8; slots * hidden * 4]);
    cuda.launch(
        tile_gate_up,
        ((intermediate / 64) as u32, 1, 1),
        (256, 1, 1),
        &[
            table.0,
            outer.0,
            sorted.0,
            items.0,
            item_count.0,
            planes.0,
            (tokens * hidden) as u64,
            tile_head.0,
            (slots * intermediate) as u64,
            TOPK as u64,
            hidden as u64,
            intermediate as u64,
            u64::from(gate_cap.to_bits()),
            u64::from(up_cap.to_bits()),
        ],
    )
    .expect("the tile's gate_up");
    cuda.launch(
        rows_gate_up,
        ((intermediate / 32) as u32, 1, 1),
        (128, 1, 1),
        &[
            table.0,
            outer.0,
            sorted.0,
            items.0,
            item_count.0,
            planes.0,
            (tokens * hidden) as u64,
            rows_head.0,
            (slots * intermediate) as u64,
            TOPK as u64,
            hidden as u64,
            intermediate as u64,
            u64::from(gate_cap.to_bits()),
            u64::from(up_cap.to_bits()),
        ],
    )
    .expect("the rows gate_up");
    let tile_words: Vec<u32> = download(cuda, &tile_head, 2 * slots * intermediate / 2);
    let rows_words: Vec<u32> = download(cuda, &rows_head, 2 * slots * intermediate / 2);
    if tile_words != rows_words {
        let at = tile_words
            .iter()
            .zip(&rows_words)
            .position(|(a, b)| a != b)
            .unwrap();
        panic!(
            "{label}: {count} rows: the rows body's gate/up differs at word {at} (plane {}, slot {}, column {}): tile {:#010x} rows {:#010x}",
            at / (slots * intermediate / 2),
            (at % (slots * intermediate / 2)) / (intermediate / 2),
            (at % (intermediate / 2)) * 2,
            tile_words[at],
            rows_words[at]
        );
    }
    for (which, out) in [("tile", tile_down_out), ("rows", rows_down_out)] {
        let kernel = if which == "tile" {
            tile_down
        } else {
            rows_down
        };
        cuda.launch(
            kernel,
            (
                (hidden
                    / if which == "tile" {
                        down_columns.0
                    } else {
                        down_columns.1
                    }) as u32,
                1,
                1,
            ),
            (if which == "tile" { 256 } else { 128 }, 1, 1),
            &[
                table.0,
                outer.0,
                sorted.0,
                items.0,
                item_count.0,
                tile_head.0,
                (slots * intermediate) as u64,
                routes.0,
                out.0,
                TOPK as u64,
                hidden as u64,
                intermediate as u64,
                TOPK as u64,
            ],
        )
        .expect("the down pass");
    }
    let tile_words: Vec<u32> = download(cuda, &tile_down_out, slots * hidden);
    let rows_words: Vec<u32> = download(cuda, &rows_down_out, slots * hidden);
    assert_eq!(
        tile_words, rows_words,
        "{label}: {count} rows: the rows body's down is the tile's, word for word"
    );
}

#[test]
#[ignore = "needs a CUDA device and the shared model images"]
fn the_expert_rows_bodies_are_the_grouped_tiles_rows() {
    let Some(cuda) = test_io::cuda_or_skip(
        qwen4_prefill_cubin()
            .is_empty()
            .then_some("the Qwen prefill image"),
    ) else {
        return;
    };
    let load = |name: &'static str| {
        cuda.load_named(qwen4_prefill_cubin(), name)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .1
    };
    let hidden = 256usize;
    let intermediate = 256usize;
    for (label, body) in [("nvfp4", nvfp4_body(hidden, intermediate, Lcg(0x1234)))] {
        for count in TOKENS {
            let (tile_gate_up, rows_gate_up, tile_down, rows_down) = if label == "nvfp4" {
                (
                    load("q4p_moe_gate_up_capped"),
                    load("q4p_moe_rows_gate_up_capped"),
                    load("q4p_moe_down"),
                    load("q4p_moe_rows_down"),
                )
            } else {
                (
                    load("q4p_moe_gate_up_capped_mx8"),
                    load("q4p_moe_rows_gate_up_capped_mx8"),
                    load("q4p_moe_down_mx8"),
                    load("q4p_moe_rows_down_mx8"),
                )
            };
            let down_columns = if label == "nvfp4" {
                (128, 64)
            } else {
                (128, 32)
            };
            check(
                &cuda,
                label,
                tile_gate_up,
                rows_gate_up,
                tile_down,
                rows_down,
                &body,
                hidden,
                intermediate,
                0.0,
                0.0,
                count,
                down_columns,
            );
        }
    }
}
