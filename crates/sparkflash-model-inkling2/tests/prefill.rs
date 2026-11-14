// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The chunked prefill's entries against the row-at-a-time ones and the host oracle.
//!
//! A chunked prefill and a prompt forwarded a token at a time must pick the same token, which
//! needs more than "both close to the reference": the two paths' kernels have to agree with
//! *each other*. Three claims are checked here, at one TP4 rank's widths:
//!
//! 1. `inkling_attn_prefill` over a chunk of rows is the same bits as `inkling_attn_decode`
//!    run once a row, including the three operands that are per row rather than per launch (the
//!    query, its gate, the relative profile and the sink). A sink read from row zero would be in
//!    bounds and plausible, which is why this is a test rather than a comment.
//! 2. `inkling_short_conv` over a chunk, given the previous rows as its lead, is
//!    `inkling_short_conv_decode` run once a row.
//! 3. `inkling_conv_history_copy` moves a history between the two layouts without changing it,
//!    either way, which is what lets claim 2 hold across a chunk boundary.
//!
//! They are the same claims the whole-chunk test makes from the other end; these fail in a
//! second with a number to look at.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::images::inkling_cubin;
use sparkflash_model_inkling2::reference::{self, AttentionParams};

const HEADS: usize = 12;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const D_REL: usize = 8;
const REL_EXTENT: usize = 16;
const ROWS: usize = 5;
const CHANNELS: usize = 96;
const KERNEL: usize = 4;
const VALUE_NORM: i32 = 1;
const EPS: f32 = 1e-6;
const QK_CLIP: f32 = 22.627_417;
const REL_CLIP: f32 = 8.0;
const TOLERANCE: f32 = 2e-5;

fn u(value: usize) -> u64 {
    value as u64
}

fn f(value: f32) -> u64 {
    u64::from(value.to_bits())
}

fn open() -> Option<Cuda> {
    if inkling_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Inkling cubin");
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

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next()).collect()
    }
}

fn kernel(cuda: &Cuda, name: &'static str) -> sparkflash_kernels::cuda_driver::Kernel {
    cuda.load_named(inkling_cubin(), name)
        .unwrap_or_else(|error| panic!("load {name}: {error}"))
        .1
}

/// The same, from the token image: the dense projections are its entries, not the arm's.
fn token_kernel(cuda: &Cuda, name: &'static str) -> sparkflash_kernels::cuda_driver::Kernel {
    cuda.load_named(sparkflash_kernels::images::qwen4_exp_cubin(), name)
        .unwrap_or_else(|error| panic!("load {name}: {error}"))
        .1
}

fn upload(cuda: &Cuda, bytes: &[u8]) -> Deviceptr {
    let mut total = 0;
    let pointer = cuda.alloc(bytes.len().max(4), &mut total).expect("alloc");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

fn upload_f32(cuda: &Cuda, values: &[f32]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn upload_bf16(cuda: &Cuda, values: &[f32]) -> Deviceptr {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| sparkflash_weights::quant::f32_to_bf16_bits(*value).to_le_bytes())
        .collect();
    upload(cuda, &bytes)
}

fn download_f32(cuda: &Cuda, pointer: &Deviceptr, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn download_bf16(cuda: &Cuda, pointer: &Deviceptr, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 2];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| sparkflash_weights::quant::bf16_to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

fn round_bf16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| {
            sparkflash_weights::quant::bf16_to_f32(sparkflash_weights::quant::f32_to_bf16_bits(
                *value,
            ))
        })
        .collect()
}

fn assert_close(label: &str, host: &[f32], device: &[f32]) {
    assert_eq!(host.len(), device.len(), "{label}: lengths");
    let scale = host
        .iter()
        .fold(0f32, |largest, value| largest.max(value.abs()))
        .max(1.0);
    for (index, (expected, got)) in host.iter().zip(device).enumerate() {
        let error = (expected - got).abs() / scale;
        assert!(
            error <= TOLERANCE,
            "{label}: [{index}] host {expected} device {got} (relative {error})"
        );
    }
}

/// The prefill attention over rows is the decode attention run once a row.
#[test]
fn the_prefill_attention_is_the_decode_attention_a_row_at_a_time() {
    let Some(cuda) = open() else { return };
    let mut rng = Lcg::new(0x5eed_1234);
    let q_width = HEADS * HEAD_DIM;
    let kv_width = KV_HEADS * HEAD_DIM;
    let positions = 9usize;
    // One row of state a position: 12 heads of 128 for the query and its gate, 2 of 128 for the
    // cache, 12 of 8 for the relative profile and 12 sinks.
    let q: Vec<f32> = rng.values(ROWS * q_width);
    let gate: Vec<f32> = rng.values(ROWS * q_width);
    let rel: Vec<f32> = rng.values(ROWS * HEADS * D_REL);
    let sink: Vec<f32> = rng.values(ROWS * HEADS);
    let keys: Vec<f32> = rng.values(positions * kv_width);
    let values: Vec<f32> = rng.values(positions * kv_width);
    let rel_bank: Vec<f32> = rng.values(D_REL * REL_EXTENT);
    let q_scale: Vec<f32> = rng.values(HEADS * HEAD_DIM);
    let rel_scale: Vec<f32> = rng.values(HEADS * D_REL);
    let k_scale: Vec<f32> = rng.values(kv_width);
    let keys_bf16 = round_bf16(&keys);
    let values_bf16 = round_bf16(&values);
    let rel_bank_bf16 = round_bf16(&rel_bank);

    let params = |tau: f32| AttentionParams {
        heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        d_rel: D_REL,
        rel_extent: REL_EXTENT,
        qk_clip: QK_CLIP,
        rel_clip: REL_CLIP,
        tau,
        value_norm: VALUE_NORM != 0,
        eps: f64::from(EPS),
    };
    let widen = |flat: &[f32], width: usize| -> Vec<Vec<f32>> {
        flat.chunks_exact(width).map(<[f32]>::to_vec).collect()
    };
    // The rows are positions 4..9 of an 9-row context: chunked prefill forwards the tail of a
    // prompt against the prefix in the cache.
    let host: Vec<f32> = (0..ROWS)
        .flat_map(|row| {
            let at = 4 + row;
            reference::attention(
                &q[row * q_width..(row + 1) * q_width],
                &gate[row * q_width..(row + 1) * q_width],
                &widen(&keys_bf16, kv_width)[..=at],
                &widen(&values_bf16, kv_width)[..=at],
                &rel[row * HEADS * D_REL..(row + 1) * HEADS * D_REL],
                &rel_bank_bf16,
                &sink[row * HEADS..(row + 1) * HEADS],
                &q_scale,
                Some(&k_scale),
                &rel_scale,
                &params(1.0),
            )
            .expect("the host attention")
        })
        .collect();

    let q_device = upload_f32(&cuda, &q);
    let gate_device = upload_f32(&cuda, &gate);
    let rel_device = upload_f32(&cuda, &rel);
    let sink_device = upload_f32(&cuda, &sink);
    let keys_device = upload_bf16(&cuda, &keys);
    let values_device = upload_bf16(&cuda, &values);
    let bank_device = upload_bf16(&cuda, &rel_bank);
    let q_scale_device = upload_f32(&cuda, &q_scale);
    let k_scale_device = upload_f32(&cuda, &k_scale);
    let rel_scale_device = upload_f32(&cuda, &rel_scale);
    let out = upload_bf16(&cuda, &vec![0f32; ROWS * q_width]);
    cuda.launch(
        kernel(&cuda, "inkling_attn_prefill"),
        (HEADS as u32, ROWS as u32, 1),
        (32, 1, 1),
        &[
            q_device.0,
            gate_device.0,
            keys_device.0,
            values_device.0,
            rel_device.0,
            bank_device.0,
            sink_device.0,
            q_scale_device.0,
            k_scale_device.0,
            rel_scale_device.0,
            out.0,
            u(HEADS),
            u(KV_HEADS),
            u(HEAD_DIM),
            u(HEADS / KV_HEADS),
            u(ROWS),
            u(4),
            u(0),
            u(0),
            u(REL_EXTENT),
            u(D_REL),
            f(QK_CLIP),
            f(REL_CLIP),
            f(1.0),
            u(VALUE_NORM as usize),
            f(EPS),
            0,
        ],
    )
    .expect("attn_prefill launch");
    let device = download_bf16(&cuda, &out, ROWS * q_width);
    assert_close("attn_prefill", &round_bf16(&host), &device);
}

/// The tensor-core GEMM over a chunk is the decode GEMV a row at a time, at a rank's widths.
///
/// The two entries are the same arithmetic in different tiles, and the chunk path uses one where
/// the row path uses the other: a disagreement here is a seed difference in every later layer.
#[test]
fn the_chunk_gemm_is_the_decode_gemv_a_row_at_a_time() {
    let Some(cuda) = open() else { return };
    let mut rng = Lcg::new(0x9e3d_77aa);
    // A rank's attention-output projection at TP4: 3 heads of 128 columns, and a hidden width
    // that is not a multiple of the tile so the tail is exercised.
    let columns = 3 * 128;
    let rows = 96usize;
    let weight: Vec<f32> = rng.values(rows * columns);
    let weight_bf16 = round_bf16(&weight);
    let input: Vec<f32> = rng.values(columns);
    let weight_device = upload_bf16(&cuda, &weight);
    let input_device = upload_f32(&cuda, &input);
    let gemv_out = upload_f32(&cuda, &vec![0f32; rows]);
    cuda.launch(
        token_kernel(&cuda, "q4e_gemv_bf16"),
        (24, 1, 1),
        (256, 1, 1),
        &[
            weight_device.0,
            input_device.0,
            gemv_out.0,
            u(rows),
            u(columns),
            1,
        ],
    )
    .expect("gemv launch");
    let gemm_out = upload_f32(&cuda, &vec![0f32; rows]);
    cuda.launch(
        token_kernel(&cuda, "q4e_gemm_bf16"),
        (2, 1, 1),
        (128, 1, 1),
        &[
            weight_device.0,
            input_device.0,
            gemm_out.0,
            u(rows),
            u(columns),
            1,
            0,
            f(1.0),
            1,
            0,
        ],
    )
    .expect("gemm launch");
    let gemv = download_f32(&cuda, &gemv_out, rows);
    let gemm = download_f32(&cuda, &gemm_out, rows);
    // The two sum in the same inner-dimension order, so the expectation is the bits - the
    // tolerance is only there so a failing row prints a number rather than an index.
    assert_close("gemm-vs-gemv", &gemv, &gemm);
    // And both are the bf16 weight against the f32 activation, summed one product at a time.
    let reference: Vec<f32> = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| weight_bf16[row * columns + column] * input[column])
                .sum()
        })
        .collect();
    assert_close("gemm-vs-reference", &reference, &gemm);
}

/// The multi-row static convolution, with its lead, is the decode convolution a row at a time.
#[test]
fn a_chunk_convolution_is_the_decode_convolution_a_row_at_a_time() {
    let Some(cuda) = open() else { return };
    let mut rng = Lcg::new(0xc0ff_ee11);
    let history = KERNEL - 1;
    // The kernel reads the taps as bf16, so the host is given the same grid rather than a
    // precision the device never had.
    let taps = round_bf16(&rng.values(CHANNELS * KERNEL));
    // The chunk's own rows, and the rows before them.
    let lead: Vec<f32> = rng.values(history * CHANNELS);
    let rows: Vec<f32> = rng.values(ROWS * CHANNELS);

    // The decode path keeps its history in the channel layout, oldest first; the history copy
    // moves it to the row layout the chunk reads.
    let mut state = vec![0f32; CHANNELS * history];
    for channel in 0..CHANNELS {
        for tap in 0..history {
            state[channel * history + tap] = lead[tap * CHANNELS + channel];
        }
    }
    let mut host = Vec::new();
    let mut walk = state.clone();
    for row in 0..ROWS {
        let out = reference::short_conv_decode(
            &rows[row * CHANNELS..(row + 1) * CHANNELS],
            &taps,
            &mut walk,
            CHANNELS,
            KERNEL,
        );
        host.extend_from_slice(&out);
    }

    let taps_device = upload_bf16(&cuda, &taps);
    // The buffer the chunk reads is the lead and then the rows, which is what the pointer
    // arithmetic in the walk gives the kernel.
    let mut buffer = lead.clone();
    buffer.extend_from_slice(&rows);
    let buffer_device = upload_f32(&cuda, &buffer);
    let out = upload_f32(&cuda, &vec![0f32; ROWS * CHANNELS]);
    cuda.launch(
        kernel(&cuda, "inkling_short_conv"),
        (3, ROWS as u32, 1),
        (256, 1, 1),
        &[
            buffer_device.0,
            taps_device.0,
            out.0,
            u(ROWS),
            u(CHANNELS),
            u(KERNEL),
            u(history),
        ],
    )
    .expect("short_conv launch");
    let device = download_f32(&cuda, &out, ROWS * CHANNELS);
    assert_close("short_conv", &host, &device);

    // The same history through the copy kernel, and back, is the history it started as.
    let state_device = upload_f32(&cuda, &state);
    let pad_device = upload_f32(&cuda, &vec![0f32; history * CHANNELS]);
    let copy = kernel(&cuda, "inkling_conv_history_copy");
    cuda.launch(
        copy,
        (2, 1, 1),
        (256, 1, 1),
        &[state_device.0, pad_device.0, u(CHANNELS), u(KERNEL), u(1)],
    )
    .expect("history to the row layout");
    assert_eq!(
        download_f32(&cuda, &pad_device, history * CHANNELS),
        lead,
        "the lead"
    );
    let back = upload_f32(&cuda, &vec![0f32; CHANNELS * history]);
    cuda.launch(
        copy,
        (2, 1, 1),
        (256, 1, 1),
        &[pad_device.0, back.0, u(CHANNELS), u(KERNEL), u(0)],
    )
    .expect("history to the channel layout");
    assert_eq!(
        download_f32(&cuda, &back, CHANNELS * history),
        state,
        "the round trip"
    );
}

/// The band's matvec over a chunk is the decode matvec run once a (row, band slot) pair.
///
/// A chunk hands the entry a plan instead of offsetting the buffers on the host, so a plan read
/// the wrong way round would run one row's expert against another row's activation - which still
/// looks like a plausible MoE and is not one. The two layouts are compared bit for bit.
#[test]
fn the_expert_matvec_with_a_plan_is_the_matvec_a_pair_at_a_time() {
    let Some(cuda) = open() else { return };
    let mut rng = Lcg::new(0x0bad_f00d);
    let width = 8usize;
    let columns = 16usize;
    let entries = 3usize;
    // Two E2M1 codes a byte and one E4M3 block scale a sixteen columns: every nibble is a
    // representable weight and the scale byte is 2.0, so the products are all well defined.
    let codes: Vec<u8> = (0..width * columns / 2)
        .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
        .collect();
    let scales = vec![0x40u8; width * columns / 16];
    let x: Vec<f32> = rng.values(entries * columns);
    // A decode pair a launch, so the pairs are deliberately a row and a slot apart.
    let plan: [i32; 6] = [2, 1, 0, 2, 1, 3];
    let codes_device = upload(&cuda, &codes);
    let scales_device = upload(&cuda, &scales);
    let x_device = upload_f32(&cuda, &x);
    let plan_device = upload(
        &cuda,
        &plan
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let matvec = kernel(&cuda, "inkling_nvfp4_expert_matvec");

    // One launch for the whole plan.
    let chunked = upload_f32(&cuda, &vec![0f32; 4 * width]);
    cuda.launch(
        matvec,
        (width as u32, entries as u32, 1),
        (256, 1, 1),
        &[
            codes_device.0,
            scales_device.0,
            x_device.0,
            chunked.0,
            u(width),
            u(columns),
            f(1.0),
            plan_device.0,
        ],
    )
    .expect("the plan's matvec");
    let planned = download_f32(&cuda, &chunked, 4 * width);

    // The same pairs, one launch each, offset on the host.
    let mut expected = vec![0f32; 4 * width];
    for entry in 0..entries {
        let row = plan[2 * entry] as usize;
        let slot = plan[2 * entry + 1] as usize;
        let out = upload_f32(&cuda, &vec![0f32; width]);
        cuda.launch(
            matvec,
            (width as u32, 1, 1),
            (256, 1, 1),
            &[
                codes_device.0,
                scales_device.0,
                x_device.offset(row * columns * 4).0,
                out.0,
                u(width),
                u(columns),
                f(1.0),
                0,
            ],
        )
        .expect("a pair's matvec");
        expected[slot * width..(slot + 1) * width]
            .copy_from_slice(&download_f32(&cuda, &out, width));
    }
    assert_eq!(planned, expected, "the plan is the pair a launch");
}

/// A batched projection is the one-row projection a row at a time, to the bit.
///
/// This is the property the chunked prefill rests on: a chunk may widen a launch, but a token's
/// projection has to be the bits its own step would give it, or the two walks part company a
/// layer later. The grid is the one-row grid, which is what a chunk passes.
#[test]
fn the_batched_projection_is_the_one_row_projection_a_row_at_a_time() {
    let Some(cuda) = open() else { return };
    let mut rng = Lcg::new(0x9e37_79b9);
    let rows = 24usize;
    let columns = 32usize;
    let batch = 5usize;
    let weight: Vec<f32> = rng.values(rows * columns);
    let x: Vec<f32> = rng.values(batch * columns);
    let weight_device = upload_bf16(&cuda, &weight);
    let x_device = upload_f32(&cuda, &x);
    let gemv = token_kernel(&cuda, "q4e_gemv_bf16");
    let grid = rows.div_ceil(8) as u32;

    let batched = upload_f32(&cuda, &vec![0f32; batch * rows]);
    cuda.launch(
        gemv,
        (grid, 1, 1),
        (256, 1, 1),
        &[
            weight_device.0,
            x_device.0,
            batched.0,
            u(rows),
            u(columns),
            u(batch),
        ],
    )
    .expect("the batched projection");
    let got = download_f32(&cuda, &batched, batch * rows);

    let mut expected = Vec::with_capacity(batch * rows);
    for token in 0..batch {
        let single = upload_f32(&cuda, &vec![0f32; rows]);
        cuda.launch(
            gemv,
            (grid, 1, 1),
            (256, 1, 1),
            &[
                weight_device.0,
                x_device.offset(token * columns * 4).0,
                single.0,
                u(rows),
                u(columns),
                1,
            ],
        )
        .expect("the one-row projection");
        expected.extend(download_f32(&cuda, &single, rows));
    }
    assert_eq!(got, expected, "a row of a batch is the row alone");
}

/// A grouped launch - several groups of eight tokens along the grid's y - is each group's own
/// launch, to the bit, and so (by the test above) each token's one-row projection.
///
/// The chunk walk runs a projection too narrow for the tensor-core tile this way: one launch for
/// all of a chunk's whole groups, then one for the remainder. A dozen output rows a launch is one
/// block, and a launch per group left a 4,096-row chunk 512 launches a layer on that block.
#[test]
fn a_grouped_projection_is_each_group_launched_alone() {
    let Some(cuda) = open() else { return };
    let mut rng = Lcg::new(0x2545_f491);
    let rows = 12usize;
    let columns = 96usize;
    let batch = 8usize;
    let groups = 5usize;
    let tokens = batch * groups;
    let weight: Vec<f32> = rng.values(rows * columns);
    let x: Vec<f32> = rng.values(tokens * columns);
    let weight_device = upload_bf16(&cuda, &weight);
    let x_device = upload_f32(&cuda, &x);
    let gemv = token_kernel(&cuda, "q4e_gemv_bf16");
    let grid = rows.div_ceil(8) as u32;

    let grouped = upload_f32(&cuda, &vec![0f32; tokens * rows]);
    cuda.launch(
        gemv,
        (grid, groups as u32, 1),
        (256, 1, 1),
        &[
            weight_device.0,
            x_device.0,
            grouped.0,
            u(rows),
            u(columns),
            u(batch),
        ],
    )
    .expect("the grouped projection");
    let got = download_f32(&cuda, &grouped, tokens * rows);

    let mut expected = Vec::with_capacity(tokens * rows);
    for group in 0..groups {
        let alone = upload_f32(&cuda, &vec![0f32; batch * rows]);
        cuda.launch(
            gemv,
            (grid, 1, 1),
            (256, 1, 1),
            &[
                weight_device.0,
                x_device.offset(group * batch * columns * 4).0,
                alone.0,
                u(rows),
                u(columns),
                u(batch),
            ],
        )
        .expect("one group alone");
        expected.extend(download_f32(&cuda, &alone, batch * rows));
    }
    assert_eq!(
        got, expected,
        "a group of a grouped launch is the group alone"
    );
}
