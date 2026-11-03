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

//! GLM-5.3-Flash's kernels in this pack against GLM's host twin, at one TP=4 rank's geometry.
//!
//! Kimi Delta Attention's recurrence runs on the Gated DeltaNet kernels against the shared host
//! delta rule: q4e_kda_decay's factors into the decode kernel's channel form (q4e_gdn_recur), and
//! into the tensor-core prefill's (q4p_kda_prep and q4p_kda_scan, then q4p_gdn_gate).
//!
//! At TP=4 a rank runs 16 of the 64 KDA heads, each 128 x 128 with its own query and key head.
//! The host side is GLM's host twin, head by head: `linear_attention`'s decay factor, L2 norms,
//! delta rule and gated RMSNorm, with `linalg`'s sigmoid. The device kernels sum in their own
//! association and use fast reciprocal square roots and exponentials, so the bound is f32
//! association relative to the largest value, the bound Qwen's device parity holds its logits to.
//!
//! The block-scaled FP8 decode GEMV (q4e_gemv_tiled_e4m3_block) is held to the same bound against
//! GLM's host FP8 linear, and its activation quantisation to the host's bits.
//!
//! It skips - loudly - without an embedded cubin or a CUDA driver.

use half::bf16;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_model_qwen4_exp::device::{
    TILED_GEMV_PARTIALS, cubin_available, embedded_cubin, embedded_prefill_cubin,
    tiled_gemv_e4m3_splits,
};
use sparkflash_weights::linalg::{self, ScaleGrid};
use sparkflash_weights::linear_attention::{self, Decay};
use sparkflash_weights::quant;

/// One rank's KDA heads at TP=4 (64 / 4), their width and the rank's channels.
const HEADS: usize = 16;
const DIM: usize = 128;
const WIDTH: usize = HEADS * DIM;
/// Rows in one launch, as a prefill chunk or a speculative verify runs them: two of the prefill
/// scan's eight-row tiles and part of a third.
const TOKENS: usize = 19;
/// The floats q4p_kda_prep stages a (token, key head) for the scan: q then k, each 128 keys with
/// four floats of padding after every 32.
const QK_ROW: usize = 2 * 144;
/// GLM-5.3-Flash's `linear_lower_bound` and `rms_norm_eps`.
const LOWER_BOUND: f32 = -5.0;
const EPS: f32 = 1e-5;
/// The largest difference allowed, relative to the host's largest magnitude.
const RELATIVE_TOLERANCE: f32 = 1e-5;

fn open(prefill: bool) -> Option<Cuda> {
    if !cubin_available() {
        eprintln!("skipped: this build embeds no Qwen4-Exp cubin");
        return None;
    }
    if prefill && embedded_prefill_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Qwen4-Exp prefill cubin");
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

/// A small deterministic generator, so a failure reproduces.
struct Lcg(u64);

impl Lcg {
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
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn download_f32(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<f32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn assert_close(what: &str, host: &[f32], device: &[f32]) {
    assert_eq!(host.len(), device.len(), "{what}: lengths");
    assert!(
        device.iter().all(|value| value.is_finite()),
        "{what}: the device produced a non-finite value"
    );
    let scale = host
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let worst = host
        .iter()
        .zip(device)
        .map(|(host, device)| (host - device).abs())
        .fold(0.0f32, f32::max);
    let relative = worst / scale;
    eprintln!(
        "{what}: worst |host - device| {worst:.3e} (scale {scale:.3}), relative {relative:.3e}"
    );
    assert!(
        relative < RELATIVE_TOLERANCE,
        "{what}: relative difference {relative:.3e} exceeds {RELATIVE_TOLERANCE:.0e}"
    );
}

/// One rank's inputs for every row, and what the host twin makes of them.
struct Fixture {
    /// Each token's convolved q | k | v.
    mixed: Vec<f32>,
    /// The output gate, the write gate and the forget projection.
    gate: Vec<f32>,
    write: Vec<f32>,
    forget: Vec<f32>,
    /// The per-head and per-channel gate parameters and the norm weight.
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<bf16>,
    /// The recurrent state before the first row.
    start: Vec<f32>,
    /// The host twin's decay factors, gated outputs and final state.
    decay: Vec<f32>,
    out: Vec<f32>,
    state: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let conv_width = 3 * WIDTH;
        let mut rng = Lcg(0x6b64_6131);
        let mixed = rng.values(TOKENS * conv_width);
        let gate = rng.values(TOKENS * WIDTH);
        let write = rng.values(TOKENS * HEADS);
        let forget = rng.values(TOKENS * WIDTH);
        let a_log = rng.values(HEADS);
        let dt_bias = rng.values(WIDTH);
        let norm: Vec<bf16> = (0..DIM)
            .map(|_| bf16::from_f32(1.0 + 0.5 * rng.next()))
            .collect();
        let start = rng.values(HEADS * DIM * DIM);

        // The host twin, one token and one head at a time.
        let norm_f32: Vec<f32> = norm.iter().map(|weight| weight.to_f32()).collect();
        let query_scale = 1.0 / (DIM as f32).sqrt();
        let mut state = start.clone();
        let mut host_decay = Vec::with_capacity(TOKENS * WIDTH);
        let mut host_out = Vec::with_capacity(TOKENS * WIDTH);
        let (mut query, mut key) = (vec![0.0f32; DIM], vec![0.0f32; DIM]);
        for token in 0..TOKENS {
            let row = &mixed[token * conv_width..(token + 1) * conv_width];
            let mut raw = vec![0.0f32; WIDTH];
            for head in 0..HEADS {
                let channels = head * DIM..(head + 1) * DIM;
                linear_attention::l2_normalise(&row[channels.clone()], query_scale, &mut query);
                linear_attention::l2_normalise(&row[WIDTH..][channels.clone()], 1.0, &mut key);
                let rate = a_log[head].exp();
                let decay: Vec<f32> = channels
                    .clone()
                    .map(|channel| {
                        linear_attention::kda_channel_decay(
                            LOWER_BOUND,
                            rate,
                            forget[token * WIDTH + channel],
                            dt_bias[channel],
                            linalg::sigmoid,
                        )
                    })
                    .collect();
                linear_attention::delta_rule_step(
                    &mut state[head * DIM * DIM..(head + 1) * DIM * DIM],
                    Decay::Channel(&decay),
                    &key,
                    &row[2 * WIDTH..][channels.clone()],
                    linalg::sigmoid(write[token * HEADS + head]),
                    &query,
                    &mut raw[channels],
                );
                host_decay.extend(decay);
            }
            for (head, head_gate) in raw
                .chunks(DIM)
                .zip(gate[token * WIDTH..(token + 1) * WIDTH].chunks(DIM))
            {
                host_out.extend(linear_attention::rms_norm_gated(
                    head,
                    &norm_f32,
                    head_gate,
                    EPS,
                    linalg::sigmoid,
                ));
            }
        }
        Self {
            mixed,
            gate,
            write,
            forget,
            a_log,
            dt_bias,
            norm,
            start,
            decay: host_decay,
            out: host_out,
            state,
        }
    }
}

/// The fixture's inputs on the device, with q4e_kda_decay's factors for every row.
struct Uploaded {
    mixed: Deviceptr,
    gate: Deviceptr,
    write: Deviceptr,
    norm: Deviceptr,
    decay: Deviceptr,
}

/// Uploads the fixture and runs q4e_kda_decay over it, checking its factors against the host's.
fn upload_fixture(cuda: &Cuda, fixture: &Fixture) -> Uploaded {
    let forget = upload_f32(cuda, &fixture.forget);
    let a_log = upload_f32(cuda, &fixture.a_log);
    let dt_bias = upload_f32(cuda, &fixture.dt_bias);
    let decay = upload_f32(cuda, &vec![0.0; TOKENS * WIDTH]);
    let (_decay_module, decay_kernel) = cuda
        .load_named(embedded_cubin(), "q4e_kda_decay")
        .expect("q4e_kda_decay");
    cuda.launch(
        decay_kernel,
        ((TOKENS * WIDTH).div_ceil(256) as u32, 1, 1),
        (256, 1, 1),
        &[
            forget.0,
            a_log.0,
            dt_bias.0,
            decay.0,
            HEADS as u64,
            DIM as u64,
            u64::from(LOWER_BOUND.to_bits()),
            TOKENS as u64,
        ],
    )
    .expect("kda_decay launch");
    assert_close(
        "decay factors",
        &fixture.decay,
        &download_f32(cuda, &decay, TOKENS * WIDTH),
    );
    Uploaded {
        mixed: upload_f32(cuda, &fixture.mixed),
        gate: upload_f32(cuda, &fixture.gate),
        write: upload_f32(cuda, &fixture.write),
        norm: upload(
            cuda,
            &fixture
                .norm
                .iter()
                .flat_map(|weight| weight.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        decay,
    }
}

/// Asserts a run left exactly the bits another did, naming the first element that differs.
fn assert_same_bits(what: &str, expected: &[f32], got: &[f32]) {
    assert_eq!(expected.len(), got.len(), "{what}: lengths");
    if let Some(index) = expected
        .iter()
        .zip(got)
        .position(|(expected, got)| expected.to_bits() != got.to_bits())
    {
        panic!(
            "{what}: element {index} is {} where it was {}",
            got[index], expected[index]
        );
    }
}

/// The decode recurrence over the fixture's first `rows` rows from its start state: `kernel` over
/// `parts` blocks a head, q4e_gdn_recur_columns' grid with `split`'s head counters, or
/// q4e_gdn_recur's with one part and none. With `gated` it writes the outputs and updates the
/// state in place; without, it is a replay, from the start state in one buffer into another.
/// Returns the outputs (none for a replay) and the state after the rows.
fn recurrence(
    cuda: &Cuda,
    kernel: Kernel,
    device: &Uploaded,
    fixture: &Fixture,
    (parts, split): (u32, &[u64]),
    rows: usize,
    gated: bool,
) -> (Vec<f32>, Vec<f32>) {
    let from = upload_f32(cuda, &fixture.start);
    let into = if gated {
        from
    } else {
        upload_f32(cuda, &vec![0.0; HEADS * DIM * DIM])
    };
    let out = upload_f32(cuda, &vec![0.0; rows * WIDTH]);
    let mut arguments = vec![
        device.mixed.0,
        device.gate.0,
        device.write.0,
        // The per-head decay's inputs, which the channel form does not read.
        0,
        0,
        0,
        device.norm.0,
        from.0,
        into.0,
        if gated { out.0 } else { 0 },
        WIDTH as u64,
        HEADS as u64,
        DIM as u64,
        DIM as u64,
        // One value head a key head.
        1,
        u64::from(EPS.to_bits()),
        // The sigmoid output gate.
        1,
        rows as u64,
        u64::from(kernel.shared_bytes()),
        device.decay.0,
    ];
    arguments.extend_from_slice(split);
    cuda.launch(kernel, (HEADS as u32, parts, 1), (128, 1, 1), &arguments)
        .expect("gdn_recur launch");
    let outputs = if gated {
        download_f32(cuda, &out, rows * WIDTH)
    } else {
        Vec::new()
    };
    (outputs, download_f32(cuda, &into, HEADS * DIM * DIM))
}

#[test]
fn the_channel_decay_recurrence_matches_the_host_delta_rule_at_one_rank() {
    let Some(cuda) = open(false) else {
        return;
    };
    let fixture = Fixture::new();
    let device = upload_fixture(&cuda, &fixture);
    let staged = cuda
        .load_named_shared(embedded_cubin(), "q4e_gdn_recur", (DIM * DIM * 4) as u32)
        .expect("q4e_gdn_recur with its staged state");
    let global = cuda
        .load_named(embedded_cubin(), "q4e_gdn_recur")
        .expect("q4e_gdn_recur");
    for (label, (_module, kernel)) in [("staged", staged), ("global", global)] {
        let (out, state) = recurrence(&cuda, kernel, &device, &fixture, (1, &[]), TOKENS, true);
        assert_close(&format!("{label} outputs"), &fixture.out, &out);
        assert_close(&format!("{label} state"), &fixture.state, &state);
    }
}

/// GLM's decode splits each head's value columns over several blocks (q4e_gdn_recur_columns).
/// Every column is its own chain, and a head's last block norms its rows with the whole head's
/// reduction, so the outputs and state keep q4e_gdn_recur's bits, staged or not, for a decode row,
/// a verify's rows and a replay; and each head's counter is back at zero for the next launch.
#[test]
fn the_column_split_recurrence_keeps_the_whole_heads_bits() {
    let Some(cuda) = open(false) else {
        return;
    };
    let fixture = Fixture::new();
    let device = upload_fixture(&cuda, &fixture);
    let counters = upload(&cuda, &[0; HEADS * 4]);
    let split = [counters.0];
    let load = |name: &'static str, staged: Option<usize>| {
        match staged {
            Some(bytes) => cuda.load_named_shared(embedded_cubin(), name, bytes as u32),
            None => cuda.load_named(embedded_cubin(), name),
        }
        .expect(name)
    };
    for staged in [true, false] {
        let (_whole_module, whole) = load("q4e_gdn_recur", staged.then_some(DIM * DIM * 4));
        for parts in [2, 4, 8] {
            let slice = DIM * DIM / parts as usize * 4;
            let (_module, columns) = load("q4e_gdn_recur_columns", staged.then_some(slice));
            for (rows, gated) in [(1, true), (TOKENS, true), (TOKENS, false)] {
                let what = format!(
                    "{} state, {parts} parts, {rows} rows{}",
                    if staged { "staged" } else { "global" },
                    if gated { "" } else { " replayed" },
                );
                let expected = recurrence(&cuda, whole, &device, &fixture, (1, &[]), rows, gated);
                let got = recurrence(
                    &cuda,
                    columns,
                    &device,
                    &fixture,
                    (parts, &split),
                    rows,
                    gated,
                );
                assert_same_bits(&format!("{what}: outputs"), &expected.0, &got.0);
                assert_same_bits(&format!("{what}: state"), &expected.1, &got.1);
            }
        }
    }
    assert!(
        download_f32(&cuda, &counters, HEADS)
            .iter()
            .all(|count| count.to_bits() == 0),
        "a head's counter left set"
    );
}

#[test]
fn the_tensor_core_prefill_scan_matches_the_host_delta_rule_at_one_rank() {
    let Some(cuda) = open(true) else {
        return;
    };
    let fixture = Fixture::new();
    let device = upload_fixture(&cuda, &fixture);
    let prefill = embedded_prefill_cubin();
    let (_prep_module, prep) = cuda
        .load_named(prefill, "q4p_kda_prep")
        .expect("q4p_kda_prep");
    let (_scan_module, scan) = cuda
        .load_named(prefill, "q4p_kda_scan")
        .expect("q4p_kda_scan");
    let (_gate_module, gate) = cuda
        .load_named(prefill, "q4p_gdn_gate")
        .expect("q4p_gdn_gate");
    let qk = upload_f32(&cuda, &vec![0.0; TOKENS * HEADS * QK_ROW]);
    let decay_beta = upload_f32(&cuda, &vec![0.0; TOKENS * HEADS * 2]);
    let out = upload_f32(&cuda, &vec![0.0; TOKENS * WIDTH]);
    cuda.launch(
        prep,
        (HEADS as u32, TOKENS as u32, 1),
        (DIM as u32, 1, 1),
        &[
            device.mixed.0,
            device.write.0,
            qk.0,
            decay_beta.0,
            WIDTH as u64,
            HEADS as u64,
            HEADS as u64,
            DIM as u64,
            // One value head a key head.
            1,
        ],
    )
    .expect("kda_prep launch");
    // The scan from the fixture's start state with each head's value columns split `splits` ways.
    let run_scan = |splits: usize| {
        let state = upload_f32(&cuda, &fixture.start);
        let raw = upload_f32(&cuda, &vec![0.0; TOKENS * WIDTH]);
        cuda.launch(
            scan,
            (HEADS as u32, splits as u32, 1),
            ((4 * DIM / splits) as u32, 1, 1),
            &[
                qk.0,
                decay_beta.0,
                device.decay.0,
                device.mixed.0,
                state.0,
                raw.0,
                WIDTH as u64,
                HEADS as u64,
                HEADS as u64,
                DIM as u64,
                1,
                TOKENS as u64,
                // No per-row snapshots.
                0,
                0,
                0,
            ],
        )
        .expect("kda_scan launch");
        (raw, state)
    };
    let (raw, state) = run_scan(1);
    // A value column's recurrence reads no other column, so every split is the same bits.
    let whole = (
        download_f32(&cuda, &raw, TOKENS * WIDTH),
        download_f32(&cuda, &state, HEADS * DIM * DIM),
    );
    for splits in [2, 4, 8, 16] {
        let (split_raw, split_state) = run_scan(splits);
        for (what, expected, got) in [
            (
                "outputs",
                &whole.0,
                download_f32(&cuda, &split_raw, TOKENS * WIDTH),
            ),
            (
                "state",
                &whole.1,
                download_f32(&cuda, &split_state, HEADS * DIM * DIM),
            ),
        ] {
            assert!(
                expected
                    .iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "the scan's {what} split {splits} ways differ from the whole head's"
            );
        }
    }
    cuda.launch(
        gate,
        (HEADS as u32, TOKENS as u32, 1),
        (DIM as u32, 1, 1),
        &[
            raw.0,
            device.gate.0,
            device.norm.0,
            out.0,
            HEADS as u64,
            DIM as u64,
            u64::from(EPS.to_bits()),
            // The sigmoid output gate.
            1,
        ],
    )
    .expect("gdn_gate launch");
    assert_close(
        "prefill outputs",
        &fixture.out,
        &download_f32(&cuda, &out, TOKENS * WIDTH),
    );
    assert_close(
        "prefill state",
        &fixture.state,
        &download_f32(&cuda, &state, HEADS * DIM * DIM),
    );
}

/// GLM-5.3-Flash's FP8 block side: one weight scale a 128 x 128 block, one activation scale a 128
/// columns.
const FP8_BLOCK: usize = 128;
/// The tiled GEMV's rows a block and threads (Q4E_TG_TILE, Q4E_TG_THREADS), and its largest batch.
const GEMV_TILE: usize = 32;
const GEMV_THREADS: u32 = 256;
const GEMV_BATCH: usize = 8;

/// GLM-5.3-Flash's FP8 linear as its host forward runs it (`Linear::apply`), a token at a time:
/// the weight's codes times their block scales, against the token quantised per 128 columns.
fn host_fp8_linear(
    codes: &[u8],
    scales: &[f32],
    x: &[f32],
    rows: usize,
    columns: usize,
) -> Vec<f32> {
    let mut weight: Vec<f32> = codes
        .iter()
        .map(|code| quant::f8e4m3_to_f32(*code))
        .collect();
    linalg::apply_scales(
        &mut weight,
        rows,
        columns,
        scales,
        ScaleGrid::Tile(FP8_BLOCK),
    )
    .expect("block scales");
    let mut out = vec![0.0; x.len() / columns * rows];
    for (token, out) in x.chunks(columns).zip(out.chunks_mut(rows)) {
        let operand =
            quant::dequantized_activation_fp8_tiles(token, FP8_BLOCK).expect("activation tiles");
        linalg::matmul(out, &operand, &weight, rows, columns, 1).expect("matmul");
    }
    out
}

/// q4e_gemv_tiled_e4m3_block over `batch` rows of `x`, split as the host splits it, with its
/// outputs in parts that start at `parts` (every one at `rows` is the plain row order).
fn device_fp8_linear(
    cuda: &Cuda,
    codes: &[u8],
    scales: &[f32],
    x: &[f32],
    rows: usize,
    columns: usize,
    parts: [usize; 3],
) -> Vec<f32> {
    let batch = x.len() / columns;
    let tiles = rows.div_ceil(GEMV_TILE);
    let splits = tiled_gemv_e4m3_splits(tiles, columns);
    let (_module, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_gemv_tiled_e4m3_block")
        .expect("q4e_gemv_tiled_e4m3_block");
    let w = upload(cuda, codes);
    let scales = upload_f32(cuda, scales);
    let x = upload_f32(cuda, x);
    let y = upload_f32(cuda, &vec![f32::NAN; batch * rows]);
    let partials = upload_f32(cuda, &vec![0.0; TILED_GEMV_PARTIALS]);
    let counters = upload(cuda, &vec![0u8; tiles * 4]);
    cuda.launch(
        kernel,
        (tiles as u32, splits as u32, 1),
        (GEMV_THREADS, 1, 1),
        &[
            w.0,
            scales.0,
            x.0,
            y.0,
            rows as u64,
            columns as u64,
            batch as u64,
            partials.0,
            counters.0,
            // The release's dynamic activation scheme.
            1,
            parts[0] as u64,
            parts[1] as u64,
            parts[2] as u64,
        ],
    )
    .expect("gemv_tiled_e4m3_block launch");
    download_f32(cuda, &y, batch * rows)
}

/// An E4M3 code drawn uniformly from the finite ones.
fn finite_code(rng: &mut Lcg) -> u8 {
    let code = ((rng.next() + 1.0) * 128.0) as u8;
    if code & 0x7F == 0x7F { code - 1 } else { code }
}

/// Through an identity weight the GEMV's output is its quantised activations, so they must be the
/// host's bit for bit: runs from 1e-3 to 1e2, a run of zeros, and a run so small its scale
/// underflows to zero, which the host quantises to zeros too.
#[test]
fn the_fp8_gemv_quantises_its_activations_as_glms_host_forward_does() {
    let Some(cuda) = open(false) else {
        return;
    };
    let side = 8 * FP8_BLOCK;
    let mut codes = vec![0u8; side * side];
    for index in 0..side {
        // 1.0: exponent 7 (the bias), mantissa 0.
        codes[index * side + index] = 0x38;
    }
    let scales = vec![1.0f32; (side / FP8_BLOCK).pow(2)];
    let mut rng = Lcg(0x6670_3861);
    let mut x = rng.values(GEMV_BATCH * side);
    for (index, value) in x.iter_mut().enumerate() {
        let run = index / FP8_BLOCK % 8;
        *value *= match run {
            0 => 0.0,
            1 => 1e-44,
            _ => 10f32.powi(run as i32 - 5),
        };
    }
    let host = host_fp8_linear(&codes, &scales, &x, side, side);
    assert!(host.iter().take(2 * FP8_BLOCK).all(|value| *value == 0.0));
    let device = device_fp8_linear(&cuda, &codes, &scales, &x, side, side, [side; 3]);
    let host_bits: Vec<u32> = host.iter().map(|value| value.to_bits()).collect();
    let device_bits: Vec<u32> = device.iter().map(|value| value.to_bits()).collect();
    assert!(
        host_bits == device_bits,
        "the quantised activations differ at {} of {} values",
        host_bits
            .iter()
            .zip(&device_bits)
            .filter(|(host, device)| host != device)
            .count(),
        host.len()
    );
}

/// At one rank's shapes - a KDA projection's 2,048 x 4,096, the dense MLP's down at 4,096 x 3,072,
/// and a shape with part of a band of blocks and part of a round - the GEMV is the host's FP8
/// linear to f32 association, each token's row is the same bits at batch 8 as alone, and a fused
/// projection's parts are the same bits, each part's rows of every token together.
#[test]
fn the_fp8_gemv_matches_glms_host_fp8_linear_at_rank_shapes() {
    let Some(cuda) = open(false) else {
        return;
    };
    let mut rng = Lcg(0x6670_3862);
    for (rows, columns) in [(2048, 4096), (4096, 3072), (200, 1152)] {
        let codes: Vec<u8> = (0..rows * columns).map(|_| finite_code(&mut rng)).collect();
        let scales: Vec<f32> = (0..ScaleGrid::Tile(FP8_BLOCK)
            .shape(rows, columns)
            .iter()
            .product())
            .map(|_| (1.5 + rng.next()) * 1e-3)
            .collect();
        let mut x = rng.values(GEMV_BATCH * columns);
        x[..FP8_BLOCK].fill(0.0);
        let splits = tiled_gemv_e4m3_splits(rows.div_ceil(GEMV_TILE), columns);
        assert!(splits > 1, "{rows} x {columns} should split its rounds");
        let label = format!("{rows} x {columns}, {splits} splits");
        let host = host_fp8_linear(&codes, &scales, &x, rows, columns);
        let device = device_fp8_linear(&cuda, &codes, &scales, &x, rows, columns, [rows; 3]);
        assert_close(&label, &host, &device);
        // Uneven parts, one of them empty, the last running to the end.
        let parts = [rows / 4, rows / 4, rows / 2 + 3];
        let parted = device_fp8_linear(&cuda, &codes, &scales, &x, rows, columns, parts);
        let bounds = [0, parts[0], parts[1], parts[2], rows];
        let mut rearranged = Vec::with_capacity(device.len());
        for part in bounds.windows(2) {
            for token in device.chunks(rows) {
                rearranged.extend_from_slice(&token[part[0]..part[1]]);
            }
        }
        assert!(
            rearranged
                .iter()
                .map(|value| value.to_bits())
                .eq(parted.iter().map(|value| value.to_bits())),
            "{label}: the parts at {parts:?} are not the rows rearranged"
        );
        for (token, batched) in device.chunks(rows).enumerate() {
            let alone = device_fp8_linear(
                &cuda,
                &codes,
                &scales,
                &x[token * columns..(token + 1) * columns],
                rows,
                columns,
                [rows; 3],
            );
            assert!(
                alone
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(batched.iter().map(|value| value.to_bits())),
                "{label}: token {token} at batch 8 differs from its batch-1 run"
            );
        }
    }
}

/// A kernel, its grid, its block width and its arguments.
type Launch = (Kernel, (u32, u32, u32), u32, Vec<u64>);

/// The absorbed MLA at one rank's geometry: GLM-5.3-Flash's 16 heads of 256 key and value
/// columns over a 512-wide latent, and the indexer's 2,048 picks out of q4e_glm_pick's stride.
const MLA_NOPE: usize = 256;
const MLA_VALUE: usize = 256;
const MLA_LATENT: usize = 512;
const MLA_STRIDE: usize = 2304;
const MLA_POSITIONS: usize = 3000;

fn upload_i32(cuda: &Cuda, values: &[i32]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

/// The split attention (q4e_glm_mla_scores, _softmax, _mix and _project) against the one kernel
/// it replaces, bit for bit, over the E4M3 cache and the f32 one: a full selection, one that
/// ends inside a tile, an empty one and a short one, in one run that starts past row 0.
#[test]
fn the_split_latent_attention_keeps_the_attend_kernels_bits() {
    let Some(cuda) = open(false) else {
        return;
    };
    let load = |name: &'static str| cuda.load_named(embedded_cubin(), name).expect(name).1;
    let attend = load("q4e_glm_mla_attend");
    let (scores, softmax, mix, project) = (
        load("q4e_glm_mla_scores"),
        load("q4e_glm_mla_softmax"),
        load("q4e_glm_mla_mix"),
        load("q4e_glm_mla_project"),
    );
    let mut rng = Lcg(0x6d6c_6121);
    let counts = [0, 2048, 1000, 0, 37, 2048];
    let rows = counts.len();
    let (base, run) = (1, rows - 1);
    let q_abs = upload_f32(&cuda, &rng.values(rows * HEADS * MLA_LATENT));
    let codes: Vec<u8> = (0..MLA_POSITIONS * MLA_LATENT)
        .map(|_| finite_code(&mut rng))
        .collect();
    let scales: Vec<f32> = (0..MLA_POSITIONS * MLA_LATENT / FP8_BLOCK)
        .map(|_| (1.5 + rng.next()) / 448.0)
        .collect();
    let latent: Vec<f32> = (0..MLA_POSITIONS * MLA_LATENT)
        .map(|index| quant::f8e4m3_to_f32(codes[index]) * scales[index / FP8_BLOCK])
        .collect();
    let (codes, scales, latent) = (
        upload(&cuda, &codes),
        upload_f32(&cuda, &scales),
        upload_f32(&cuda, &latent),
    );
    let kv_b = upload_f32(
        &cuda,
        &rng.values(HEADS * (MLA_NOPE + MLA_VALUE) * MLA_LATENT),
    );
    let selected: Vec<i32> = (0..rows * MLA_STRIDE)
        .map(|_| ((rng.next() + 1.0) * 0.5 * (MLA_POSITIONS - 1) as f32) as i32)
        .collect();
    let selected = upload_i32(&cuda, &selected);
    let count = upload_i32(&cuda, &counts);
    let unset = vec![f32::from_bits(0x7FC0_1234); rows * HEADS * MLA_VALUE];
    for quantise in [1u64, 0] {
        let expected = upload_f32(&cuda, &unset);
        cuda.launch(
            attend,
            (HEADS as u32, run as u32, 1),
            (256, 1, 1),
            &[
                q_abs.0,
                latent.0,
                codes.0,
                scales.0,
                kv_b.0,
                selected.0,
                count.0,
                expected.0,
                base as u64,
                MLA_NOPE as u64,
                MLA_VALUE as u64,
                MLA_LATENT as u64,
                MLA_STRIDE as u64,
                quantise,
            ],
        )
        .expect("mla_attend launch");
        let got = upload_f32(&cuda, &unset);
        let logits = upload_f32(&cuda, &vec![0.0; rows * HEADS * MLA_STRIDE]);
        let total = upload_f32(&cuda, &vec![0.0; rows * HEADS]);
        let mixed = upload_f32(&cuda, &vec![0.0; rows * HEADS * MLA_LATENT]);
        let launches: [Launch; 4] = [
            (
                scores,
                ((MLA_STRIDE / 64) as u32, run as u32, 1),
                256,
                vec![
                    q_abs.0,
                    latent.0,
                    codes.0,
                    scales.0,
                    selected.0,
                    count.0,
                    logits.0,
                    base as u64,
                    HEADS as u64,
                    MLA_NOPE as u64,
                    MLA_LATENT as u64,
                    MLA_STRIDE as u64,
                    quantise,
                ],
            ),
            (
                softmax,
                (HEADS as u32, run as u32, 1),
                256,
                vec![logits.0, count.0, total.0, base as u64],
            ),
            (
                mix,
                (
                    (MLA_LATENT / 32) as u32,
                    HEADS.div_ceil(8) as u32,
                    run as u32,
                ),
                256,
                vec![
                    latent.0,
                    codes.0,
                    scales.0,
                    selected.0,
                    count.0,
                    logits.0,
                    total.0,
                    mixed.0,
                    base as u64,
                    HEADS as u64,
                    MLA_LATENT as u64,
                    MLA_STRIDE as u64,
                    quantise,
                ],
            ),
            (
                project,
                (HEADS as u32, MLA_VALUE.div_ceil(64) as u32, 1),
                64,
                vec![
                    mixed.0,
                    kv_b.0,
                    count.0,
                    got.0,
                    base as u64,
                    run as u64,
                    MLA_NOPE as u64,
                    MLA_VALUE as u64,
                    MLA_LATENT as u64,
                ],
            ),
        ];
        for (kernel, grid, threads, arguments) in &launches {
            cuda.launch(*kernel, *grid, (*threads, 1, 1), arguments)
                .expect("split attention launch");
        }
        let expected = download_f32(&cuda, &expected, rows * HEADS * MLA_VALUE);
        let got = download_f32(&cuda, &got, rows * HEADS * MLA_VALUE);
        assert!(
            expected[HEADS * MLA_VALUE..2 * HEADS * MLA_VALUE]
                .iter()
                .any(|value| *value != 0.0 && value.is_finite()),
            "the full selection's row came out empty"
        );
        assert_same_bits(
            if quantise == 1 {
                "E4M3 cache"
            } else {
                "f32 cache"
            },
            &expected,
            &got,
        );
    }
}
