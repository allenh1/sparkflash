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

//! What the decode kernels cost at the released model's shapes, one token and a verify's eight.
//!
//! A decode step is weight-bandwidth bound, so each line reports the kernel's time and the rate
//! it streamed its weights at. Launches are queued back to back and timed together, so the
//! driver's per-launch cost stays out of the number, and the weights rotate through enough copies
//! that none is still in L2 when its turn comes round again.
//!
//! ```sh
//! cargo test -p sparkflash-model-qwen4-exp --test decode_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, set_async_launches};
use sparkflash_model_qwen4_exp::device::{
    TILED_GEMV_BLOCKS, TILED_GEMV_PARTIALS, cubin_available, embedded_cubin,
    tiled_gemv_e4m3_splits, tiled_gemv_splits,
};

/// Rows a down-pass warp owns, mirroring `Q4E_DOWN_ROWS_PER_WARP`.
const DOWN_ROWS: usize = 4;

/// Launches timed together.
const LAUNCHES: usize = 40;
/// Weight bytes rotated through, well past the L2.
const ROTATION_BYTES: usize = 512 << 20;

fn open() -> Option<Cuda> {
    if !cubin_available() {
        eprintln!("skipped: this build embeds no Qwen4-Exp cubin");
        return None;
    }
    match Cuda::load() {
        Ok(cuda) => Some(cuda),
        Err(error) => {
            eprintln!("skipped: no CUDA driver ({error})");
            None
        }
    }
}

fn kernel(cuda: &Cuda, name: &'static str) -> Kernel {
    cuda.load_named(embedded_cubin(), name)
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .1
}

/// A buffer of `bytes` filled with small values that are finite read as f32 or as BF16 pairs.
fn filled(cuda: &Cuda, bytes: usize, total: &mut usize) -> Deviceptr {
    let pointer = cuda.alloc(bytes, total).expect("device buffer");
    let pattern: Vec<u8> = (0..bytes.min(1 << 20))
        .map(|index| {
            if index % 2 == 0 {
                0x3d
            } else {
                ((index * 7) % 61) as u8
            }
        })
        .collect();
    let mut offset = 0;
    while offset < bytes {
        let take = pattern.len().min(bytes - offset);
        cuda.copy_htod(&pointer.offset(offset), &pattern[..take])
            .expect("fill");
        offset += take;
    }
    pointer
}

/// Enough copies of a `bytes`-sized weight to rotate through without an L2 hit.
fn copies(cuda: &Cuda, bytes: usize, total: &mut usize) -> Vec<Deviceptr> {
    let count = ROTATION_BYTES.div_ceil(bytes).clamp(1, 64);
    (0..count).map(|_| filled(cuda, bytes, total)).collect()
}

/// Mean microseconds of one `launch(round)` call among `LAUNCHES` queued back to back.
fn time(cuda: &Cuda, mut launch: impl FnMut(usize)) -> f64 {
    let previous = set_async_launches(true);
    launch(0);
    cuda.synchronize().expect("warm-up");
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        for round in 0..LAUNCHES {
            launch(round);
        }
        cuda.synchronize().expect("drain");
        best = best.min(start.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
    }
    set_async_launches(previous);
    best
}

fn report(what: &str, batch: usize, micros: f64, weight_bytes: usize) {
    println!(
        "{what:<34} batch {batch}: {micros:8.1} us  {:6.1} GB/s",
        weight_bytes as f64 / micros / 1e3
    );
}

#[test]
#[ignore = "a benchmark: run with --ignored --nocapture on a GPU"]
fn decode_kernels_at_the_released_shapes() {
    let Some(cuda) = open() else {
        return;
    };
    let gemv = kernel(&cuda, "q4e_gemv_bf16");
    let tiled = kernel(&cuda, "q4e_gemv_tiled_bf16");
    let tiled_e4m3 = kernel(&cuda, "q4e_gemv_tiled_e4m3");
    let tiled_down = kernel(&cuda, "q4e_hyper_down_tiled");
    let tiled_up = kernel(&cuda, "q4e_hyper_up_tiled");
    let gemv4 = kernel(&cuda, "q4e_gemv4_bf16");
    let hyper_down = kernel(&cuda, "q4e_hyper_down");
    let hyper_up = kernel(&cuda, "q4e_hyper_up");
    let (_, recur) = cuda
        .load_named_shared(embedded_cubin(), "q4e_gdn_recur", 128 * 128 * 4)
        .expect("q4e_gdn_recur with its staged state");
    let mut total = 0;
    let x = filled(&cuda, 8 * 16_384 * 4, &mut total);
    let y = cuda.alloc(8 * 248_320 * 4, &mut total).expect("y");
    let block = (256, 1, 1);
    let partials = cuda
        .alloc(TILED_GEMV_PARTIALS * 4, &mut total)
        .expect("partials");
    let counters = cuda
        .alloc(TILED_GEMV_BLOCKS * 4, &mut total)
        .expect("counters");
    cuda.copy_htod(&counters, &vec![0u8; TILED_GEMV_BLOCKS * 4])
        .expect("zero counters");

    // The single projections: the linear layers' output, the sparse layers' output and the head.
    for (what, rows, columns) in [
        ("gemv gdn out_proj 2560x6144", 2560_usize, 6144_usize),
        ("gemv attn o_proj 2560x6144", 2560, 24 * 256),
        ("gemv lm_head 248320x2560", 248_320, 2560),
    ] {
        let bytes = rows * columns * 2;
        let weights = copies(&cuda, bytes, &mut total);
        for batch in [1_usize, 4, 8] {
            let micros = time(&cuda, |round| {
                cuda.launch(
                    gemv,
                    (rows.div_ceil(8) as u32, 1, 1),
                    block,
                    &[
                        weights[round % weights.len()].0,
                        x.0,
                        y.0,
                        rows as u64,
                        columns as u64,
                        batch as u64,
                    ],
                )
                .expect("gemv");
            });
            report(what, batch, micros, bytes);
            let tiles = rows.div_ceil(32);
            let splits = tiled_gemv_splits(tiles, columns);
            let micros = time(&cuda, |round| {
                cuda.launch(
                    tiled,
                    (tiles as u32, splits as u32, 1),
                    block,
                    &[
                        weights[round % weights.len()].0,
                        x.0,
                        y.0,
                        rows as u64,
                        columns as u64,
                        batch as u64,
                        partials.0,
                        counters.0,
                    ],
                )
                .expect("tiled gemv");
            });
            report(&format!("  tiled, {splits} splits"), batch, micros, bytes);
        }
    }

    // The drafter's block projections and its head, in E4M3, at the block's eight rows (the
    // head scores seven).
    let shapes: [(&str, usize, usize, usize); 5] = [
        ("e4m3 draft q_proj 4096x2560", 4096, 2560, 8),
        ("e4m3 draft o_proj 2560x4096", 2560, 4096, 8),
        ("e4m3 draft gate/up 7680x2560", 7680, 2560, 8),
        ("e4m3 draft down 2560x7680", 2560, 7680, 8),
        ("e4m3 draft head 248320x2560", 248_320, 2560, 7),
    ];
    for (what, rows, columns, batch) in shapes {
        let bytes = rows * columns;
        let weights = copies(&cuda, bytes, &mut total);
        let scales = filled(&cuda, rows * 4, &mut total);
        let tiles = rows.div_ceil(32);
        let splits = tiled_gemv_e4m3_splits(tiles, columns);
        let micros = time(&cuda, |round| {
            cuda.launch(
                tiled_e4m3,
                (tiles as u32, splits as u32, 1),
                block,
                &[
                    weights[round % weights.len()].0,
                    scales.0,
                    x.0,
                    y.0,
                    rows as u64,
                    columns as u64,
                    batch as u64,
                    partials.0,
                    counters.0,
                ],
            )
            .expect("e4m3 gemv");
        });
        report(&format!("{what}, {splits} splits"), batch, micros, bytes);
    }

    // The linear layers' fused input projection: qkv, z, b and a against one normed row.
    {
        let (r0, r1, r2, r3, columns) = (10_240_usize, 6144_usize, 48_usize, 48_usize, 2560_usize);
        let rows = r0 + r1 + r2 + r3;
        let bytes = rows * columns * 2;
        let weights = copies(&cuda, bytes, &mut total);
        let outputs = cuda.alloc(8 * rows * 4, &mut total).expect("outputs");
        for batch in [1_usize, 4, 8] {
            let micros = time(&cuda, |round| {
                let w = &weights[round % weights.len()];
                cuda.launch(
                    gemv4,
                    (rows.div_ceil(8) as u32, 1, 1),
                    block,
                    &[
                        w.0,
                        w.offset(r0 * columns * 2).0,
                        w.offset((r0 + r1) * columns * 2).0,
                        w.offset((r0 + r1 + r2) * columns * 2).0,
                        x.0,
                        outputs.0,
                        outputs.offset(8 * r0 * 4).0,
                        outputs.offset(8 * (r0 + r1) * 4).0,
                        outputs.offset(8 * (r0 + r1 + r2) * 4).0,
                        r0 as u64,
                        r1 as u64,
                        r2 as u64,
                        r3 as u64,
                        columns as u64,
                        batch as u64,
                    ],
                )
                .expect("gemv4");
            });
            report("gemv4 gdn in_proj 16480x2560", batch, micros, bytes);
        }
    }

    // One hyper-connection: the low-rank and injection rows down, then the mix back up.
    {
        let (hc, hidden, lowrank) = (4_usize, 2560_usize, 320_usize);
        let width = hc * hidden;
        let down_bytes = (lowrank + hc) * width * 2;
        let up_bytes = width * lowrank * 2;
        let downs = copies(&cuda, down_bytes, &mut total);
        let ups = copies(&cuda, up_bytes, &mut total);
        let low = cuda.alloc(8 * lowrank * 4, &mut total).expect("low");
        let injection = cuda.alloc(8 * hc * 4, &mut total).expect("injection");
        let mixed = cuda.alloc(8 * hidden * 4, &mut total).expect("mixed");
        for batch in [1_usize, 4, 8] {
            let micros = time(&cuda, |round| {
                let w = &downs[round % downs.len()];
                cuda.launch(
                    hyper_down,
                    ((lowrank + hc).div_ceil(8) as u32, 1, 1),
                    block,
                    &[
                        x.0,
                        w.0,
                        w.offset(lowrank * width * 2).0,
                        low.0,
                        injection.0,
                        hc as u64,
                        width as u64,
                        lowrank as u64,
                        batch as u64,
                    ],
                )
                .expect("hyper_down");
            });
            report("hyper_down 324x10240", batch, micros, down_bytes);
            let micros = time(&cuda, |round| {
                cuda.launch(
                    hyper_up,
                    (hidden.div_ceil(8) as u32, 1, 1),
                    block,
                    &[
                        x.0,
                        ups[round % ups.len()].0,
                        low.0,
                        mixed.0,
                        hc as u64,
                        hidden as u64,
                        lowrank as u64,
                        batch as u64,
                    ],
                )
                .expect("hyper_up");
            });
            report("hyper_up 10240x320", batch, micros, up_bytes);
            let tiles = (lowrank + hc).div_ceil(32);
            let splits = tiled_gemv_splits(tiles, width);
            let micros = time(&cuda, |round| {
                let w = &downs[round % downs.len()];
                cuda.launch(
                    tiled_down,
                    (tiles as u32, splits as u32, 1),
                    block,
                    &[
                        x.0,
                        w.0,
                        w.offset(lowrank * width * 2).0,
                        low.0,
                        injection.0,
                        hc as u64,
                        width as u64,
                        lowrank as u64,
                        batch as u64,
                        partials.0,
                        counters.0,
                    ],
                )
                .expect("tiled hyper_down");
            });
            report(
                &format!("  tiled down, {splits} splits"),
                batch,
                micros,
                down_bytes,
            );
            let micros = time(&cuda, |round| {
                cuda.launch(
                    tiled_up,
                    ((hidden * hc).div_ceil(32) as u32, 1, 1),
                    block,
                    &[
                        x.0,
                        ups[round % ups.len()].0,
                        low.0,
                        mixed.0,
                        hc as u64,
                        hidden as u64,
                        lowrank as u64,
                        batch as u64,
                    ],
                )
                .expect("tiled hyper_up");
            });
            report("  tiled up", batch, micros, up_bytes);
        }
    }

    // The gated delta rule: 48 value heads of 128x128 f32 state. A decode step reads and writes
    // the state (a verify too, into the other buffer of the pair), and the replay that keeps
    // fewer of a verify's rows does the same with no output.
    {
        let (heads, key_heads, dim) = (48_usize, 16_usize, 128_usize);
        let key_width = key_heads * dim;
        let conv_width = 2 * key_width + heads * dim;
        let state_bytes = heads * dim * dim * 4;
        let states = copies(&cuda, state_bytes, &mut total);
        let spares = copies(&cuda, state_bytes, &mut total);
        let mixed = filled(&cuda, 8 * conv_width * 4, &mut total);
        let small = filled(&cuda, 8 * heads * dim * 4, &mut total);
        let out = cuda.alloc(8 * heads * dim * 4, &mut total).expect("out");
        for (what, batch, state_bytes_moved) in [
            ("gdn_recur 48 heads, step", 1_usize, 2 * state_bytes),
            ("gdn_recur 48 heads, step", 8, 2 * state_bytes),
            ("gdn_recur 48 heads, replay", 1, 2 * state_bytes),
            ("gdn_recur 48 heads, replay", 4, 2 * state_bytes),
            ("gdn_recur 48 heads, replay", 8, 2 * state_bytes),
        ] {
            let micros = time(&cuda, |round| {
                let state = states[round % states.len()];
                let (state_out, out) = match what {
                    "gdn_recur 48 heads, step" => (state.0, out.0),
                    _ => (spares[round % spares.len()].0, 0),
                };
                cuda.launch(
                    recur,
                    (heads as u32, 1, 1),
                    (128, 1, 1),
                    &[
                        mixed.0,
                        small.0,
                        small.0,
                        small.0,
                        small.0,
                        small.0,
                        small.0,
                        state.0,
                        state_out,
                        out,
                        key_width as u64,
                        heads as u64,
                        dim as u64,
                        dim as u64,
                        (heads / key_heads) as u64,
                        1e-6_f32.to_bits() as u64,
                        1,
                        batch as u64,
                        u64::from(recur.shared_bytes()),
                        // No channel decay: Gated DeltaNet decays a head by one factor.
                        0,
                    ],
                )
                .expect("gdn_recur");
            });
            report(what, batch, micros, state_bytes_moved);
        }
    }
    // The routed experts: a layer's 512 NVFP4 experts, ten a token, each launch's experts cold.
    {
        let (hidden, inter, topk, experts) = (2560_usize, 640_usize, 10_usize, 512_usize);
        let permute = kernel(&cuda, "q4e_moe_permute");
        let gate_up = kernel(&cuda, "q4e_expert_gate_up");
        let down = kernel(&cuda, "q4e_expert_down");
        let (gate_bytes, gate_scale_bytes) = (inter * hidden / 2, inter * hidden / 16);
        let (down_bytes, down_scale_bytes) = (hidden * inter / 2, hidden * inter / 16);
        let expert_bytes = 2 * (gate_bytes + gate_scale_bytes) + down_bytes + down_scale_bytes;
        let backing = filled(&cuda, experts * expert_bytes, &mut total);
        let mut table = Vec::with_capacity(experts * 6);
        for expert in 0..experts {
            let gate = backing.0 + (expert * expert_bytes) as u64;
            let gate_scales = gate + gate_bytes as u64;
            let up = gate_scales + gate_scale_bytes as u64;
            let up_scales = up + gate_bytes as u64;
            let down = up_scales + gate_scale_bytes as u64;
            let down_scales = down + down_bytes as u64;
            table.extend([gate, gate_scales, up, up_scales, down, down_scales]);
        }
        let table_bytes: Vec<u8> = table.iter().flat_map(|word| word.to_le_bytes()).collect();
        let table = cuda.alloc(table_bytes.len(), &mut total).expect("table");
        cuda.copy_htod(&table, &table_bytes).expect("upload table");
        let scale_bytes: Vec<u8> = (0..experts * 3)
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect();
        let scales = cuda.alloc(scale_bytes.len(), &mut total).expect("scales");
        cuda.copy_htod(&scales, &scale_bytes)
            .expect("upload scales");
        let weights = filled(&cuda, 8 * experts * 4, &mut total);
        let perm = cuda.alloc(8 * topk * 4, &mut total).expect("perm");
        let runs = cuda.alloc(8 * topk * 2 * 4, &mut total).expect("runs");
        let activated = cuda
            .alloc(8 * topk * inter * 4, &mut total)
            .expect("activated");
        let partials = cuda
            .alloc(8 * topk * hidden * 4, &mut total)
            .expect("partials");
        // A verify's eight tokens shared about a third of their slots in the profile.
        for (batch, distinct) in [(1_usize, 10_usize), (8, 56)] {
            let pools = experts / distinct;
            let index_sets: Vec<Deviceptr> = (0..pools)
                .map(|pool| {
                    let picks: Vec<u8> = (0..batch * topk)
                        .flat_map(|slot| {
                            let (t, j) = (slot / topk, slot % topk);
                            ((pool * distinct + (t * topk + j) % distinct) as i32).to_le_bytes()
                        })
                        .collect();
                    let buffer = cuda.alloc(picks.len(), &mut total).expect("indices");
                    cuda.copy_htod(&buffer, &picks).expect("upload indices");
                    buffer
                })
                .collect();
            let block = (256, 1, 1);
            let gate_up_bytes = 2 * (gate_bytes + gate_scale_bytes);
            for (stage, passes, bytes) in [
                ("", 3, expert_bytes),
                ("  gate_up", 1, gate_up_bytes),
                ("  down", 2, down_bytes + down_scale_bytes),
            ] {
                let micros = time(&cuda, |round| {
                    let indices = &index_sets[round % pools];
                    cuda.launch(
                        permute,
                        (1, 1, 1),
                        block,
                        &[indices.0, perm.0, runs.0, (batch * topk) as u64],
                    )
                    .expect("permute");
                    if passes & 1 != 0 {
                        cuda.launch(
                            gate_up,
                            (inter.div_ceil(16) as u32, topk as u32, batch as u32),
                            block,
                            &[
                                table.0,
                                scales.0,
                                indices.0,
                                perm.0,
                                runs.0,
                                x.0,
                                activated.0,
                                topk as u64,
                                hidden as u64,
                                inter as u64,
                            ],
                        )
                        .expect("gate_up");
                    }
                    if passes & 2 != 0 {
                        cuda.launch(
                            down,
                            (
                                hidden.div_ceil(DOWN_ROWS * 8) as u32,
                                topk as u32,
                                batch as u32,
                            ),
                            block,
                            &[
                                table.0,
                                scales.0,
                                indices.0,
                                perm.0,
                                runs.0,
                                weights.0,
                                activated.0,
                                partials.0,
                                topk as u64,
                                hidden as u64,
                                inter as u64,
                                experts as u64,
                            ],
                        )
                        .expect("down");
                    }
                });
                let what = if stage.is_empty() {
                    format!("experts nvfp4, {distinct} distinct")
                } else {
                    stage.to_string()
                };
                report(&what, batch, micros, distinct * bytes);
            }
        }
    }
    eprintln!("{:.1} GiB allocated", total as f64 / (1u64 << 30) as f64);
}
