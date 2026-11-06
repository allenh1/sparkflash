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

//! The few-row forms of the tensor-core prefill's kernels on a device: q4p_dense_rows for its
//! dense projection, q4p_gemm_rows for the token GEMM it falls back to, and q4p_moe_rows for its
//! routed experts.
//!
//! A decode step, a speculative verify and a short chunk run their projections through them, and
//! a cold prefill of the same tokens runs them through q4p_dense_mma's 128-row tile,
//! q4e_gemm_bf16's 64 x 128 one and the experts' 128-row panel. Two claims for the dense MMA, at
//! the shapes the release's projections take (a partial 64-column tile, a narrow matrix split five
//! ways, a wide one unsplit):
//!
//! 1. batch invariance: every row of an M-row launch, M = 1..=63, is bit-identical to the row the
//!    128-row tile computes, and to that row launched alone. This is what lets a continued
//!    conversation resume on the state a cold prefill of it leaves.
//! 2. accuracy: against an f64 reference, the worst error is the head-and-remainder split's (16
//!    significant bits an activation), well inside the bound a single-BF16 activation breaks.
//!
//! A split few-row launch also adds its slices and applies the epilogue itself, where the tile
//! leaves that to q4e_gemm_reduce: the finished outputs are held to the bits of the tile's partials
//! through that reduce (or, unsplit, through q4p_epilogue), at every row count, on one set of
//! arrival counters the launches must leave zeroed for the next.
//!
//! The token GEMM's twin is held to the first claim, epilogue and reduce included, at the off-grid
//! shapes it serves, and so is each of its CTAs riding a dense launch; its accuracy is the f32 FMA
//! chain the token kernels' own tests bound. The experts' twin is held to both at the release's
//! expert shape (2560 x 640, ten of them a token): every slot of a 1..=63-row pass, SwiGLU planes
//! and routed partials, is the panel's slot in an 80-row chunk, and the panel is within the split's
//! bound of f64.
//!
//! It skips - loudly - without the prefill image or a CUDA driver.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel};
use sparkflash_kernels::m4_quant::{bf16_bits, bf16_bits_to_f32};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_cubin, embedded_prefill_cubin};

/// The worst |error| relative to the largest |reference| the forms may reach. The split carries
/// about 2^-17 of each activation; dropping the remainder (one BF16 activation) lands near 6e-4.
const ACCURACY: f64 = 5e-5;

fn open() -> Option<Cuda> {
    if !cubin_available() || embedded_prefill_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Qwen prefill image");
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
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload(cuda: &Cuda, bytes: &[u8]) -> Deviceptr {
    let mut total = 0;
    let pointer = cuda.alloc(bytes.len().max(4), &mut total).expect("alloc");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

fn download_bits(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<u32> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// Where a q4p_dense_rows launch finishes its outputs: `y`, from the slices on `tickets`, through
/// `epilogue` and `scale`.
#[derive(Clone, Copy)]
struct Finish {
    y: Deviceptr,
    tickets: Deviceptr,
    epilogue: u32,
    scale: f32,
}

/// One q4p_dense_mma (`kernels.0`, `rows` false) or q4p_dense_rows (`kernels.1`) launch over
/// `tokens` rows of `a` into out[segment][token][column]; q4p_dense_rows also finishes them.
#[allow(clippy::too_many_arguments)]
fn launch(
    cuda: &Cuda,
    kernels: (Kernel, Kernel),
    rows: bool,
    w: Deviceptr,
    a: Deviceptr,
    out: Deviceptr,
    cols: usize,
    inner: usize,
    tokens: usize,
    segments: usize,
    finish: Finish,
) {
    if rows {
        launch_rows(
            cuda,
            kernels.1,
            &[(w, out, finish.y, cols)],
            &[],
            a,
            inner,
            tokens,
            segments,
            finish,
        );
        return;
    }
    cuda.launch(
        kernels.0,
        (
            cols.div_ceil(128) as u32,
            tokens.div_ceil(128) as u32,
            segments as u32,
        ),
        (256, 1, 1),
        &[
            w.0,
            a.0,
            out.0,
            cols as u64,
            inner as u64,
            tokens as u64,
            segments as u64,
        ],
    )
    .expect("the launch");
}

/// A q4p_gemm_rows grid riding a q4p_dense_rows launch: `w`'s `rows` over the launch's input,
/// finished into `y` through `epilogue` and `scale`, split `split` ways over `partials` and
/// `tickets`.
#[derive(Clone, Copy)]
struct Rider {
    w: Deviceptr,
    y: Deviceptr,
    rows: usize,
    epilogue: u32,
    scale: f32,
    split: usize,
    partials: Deviceptr,
    tickets: Deviceptr,
}

/// One q4p_dense_rows launch of up to four matrices `(w, out, y, cols)` of one input, each split
/// `segments` ways, their tickets end to end from `finish.tickets` (whose `y` is unused), and up
/// to two `riders`.
#[allow(clippy::too_many_arguments)]
fn launch_rows(
    cuda: &Cuda,
    kernel: Kernel,
    matrices: &[(Deviceptr, Deviceptr, Deviceptr, usize)],
    riders: &[Rider],
    a: Deviceptr,
    inner: usize,
    tokens: usize,
    segments: usize,
    finish: Finish,
) {
    let mut args = vec![
        a.0,
        inner as u64,
        tokens as u64,
        segments as u64,
        finish.tickets.0,
        u64::from(finish.epilogue),
        u64::from(finish.scale.to_bits()),
        matrices.len() as u64,
    ];
    let mut tiles = 0;
    for slot in 0..4 {
        match matrices.get(slot) {
            Some(&(w, out, y, cols)) => {
                args.extend_from_slice(&[w.0, out.0, y.0, cols as u64]);
                tiles += cols.div_ceil(32);
            }
            None => args.extend_from_slice(&[0; 4]),
        }
    }
    args.extend_from_slice(&[tiles as u64, riders.len() as u64]);
    let mut ctas = 0;
    for slot in 0..2 {
        match riders.get(slot) {
            Some(rider) => {
                args.extend_from_slice(&[
                    rider.w.0,
                    rider.y.0,
                    rider.rows as u64,
                    u64::from(rider.epilogue),
                    u64::from(rider.scale.to_bits()),
                    rider.split as u64,
                    rider.partials.0,
                    rider.tickets.0,
                ]);
                ctas += (rider.rows * tokens).div_ceil(128) * rider.split;
            }
            None => args.extend_from_slice(&[0; 8]),
        }
    }
    let planes = tokens.div_ceil(16) * segments;
    cuda.launch(
        kernel,
        (
            (tiles + ctas.div_ceil(planes)) as u32,
            tokens.div_ceil(16) as u32,
            segments as u32,
        ),
        (128, 1, 1),
        &args,
    )
    .expect("the launch");
}

/// One q4p_gemm_rows launch of `rider`'s grid over `count` rows of `a`, `columns` wide.
fn launch_gemm_rows(
    cuda: &Cuda,
    kernel: Kernel,
    rider: Rider,
    a: Deviceptr,
    columns: usize,
    count: usize,
) {
    cuda.launch(
        kernel,
        (
            (rider.rows * count).div_ceil(128) as u32,
            rider.split as u32,
            1,
        ),
        (128, 1, 1),
        &[
            rider.w.0,
            a.0,
            rider.y.0,
            rider.rows as u64,
            columns as u64,
            count as u64,
            u64::from(rider.epilogue),
            u64::from(rider.scale.to_bits()),
            rider.split as u64,
            if rider.split > 1 { rider.partials.0 } else { 0 },
            rider.tickets.0,
        ],
    )
    .expect("the launch");
}

/// The tile's finished outputs: its `segments` slices of `count` outputs added in order by
/// q4e_gemm_reduce and put through the epilogue, or, unsplit, the epilogue alone (q4p_epilogue) -
/// what the tensor-core image's launchers run after the tile. Into `y`.
#[allow(clippy::too_many_arguments)]
fn finish_tile(
    cuda: &Cuda,
    reduce: Kernel,
    epilogue_kernel: Kernel,
    partials: Deviceptr,
    y: Deviceptr,
    count: usize,
    segments: usize,
    epilogue: u32,
    scale: f32,
) {
    if segments > 1 {
        cuda.launch(
            reduce,
            (count.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            &[
                partials.0,
                y.0,
                count as u64,
                segments as u64,
                u64::from(epilogue),
                u64::from(scale.to_bits()),
            ],
        )
        .expect("q4e_gemm_reduce");
    } else {
        cuda.copy_dtod(&y, &partials, count * 4).expect("copy");
        cuda.launch(
            epilogue_kernel,
            (count.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            &[
                y.0,
                count as u64,
                u64::from(epilogue),
                u64::from(scale.to_bits()),
            ],
        )
        .expect("q4p_epilogue");
    }
}

/// Fill a buffer with NaN bits, which no output is: a launch that leaves an output unwritten
/// cannot pass for one that wrote the right bits.
fn poison(cuda: &Cuda, pointer: Deviceptr, elements: usize) {
    cuda.memset_d8_async(pointer, 0xff, elements * 4)
        .expect("memset");
}

#[test]
fn a_few_rows_are_the_bits_the_tile_computes_and_within_the_split_bound() {
    let Some(cuda) = open() else { return };
    let load = |name| {
        cuda.load_named(embedded_prefill_cubin(), name)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    };
    let (_tile_module, tile_kernel) = load("q4p_dense_mma");
    let (_rows_module, rows_kernel) = load("q4p_dense_rows");
    let (_epilogue_module, epilogue_kernel) = load("q4p_epilogue");
    let (_reduce_module, reduce) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_reduce")
        .expect("q4e_gemm_reduce");
    let k = (tile_kernel, rows_kernel);
    const TOKENS: usize = 63;
    let scale = 1.7f32;
    for (cols, inner, segments, epilogue) in [
        (96usize, 2560usize, 5usize, 2u32),
        (512, 2560, 2, 1),
        (2560, 6144, 2, 0),
        (4096, 2560, 1, 1),
    ] {
        let mut rng = Lcg((cols * 31 + inner * 7 + segments) as u64);
        let weights: Vec<u16> = (0..cols * inner)
            .map(|_| bf16_bits(rng.next() * 0.05))
            .collect();
        let activations: Vec<f32> = (0..TOKENS * inner).map(|_| rng.next()).collect();
        let w = upload(
            &cuda,
            &weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let a = upload(
            &cuda,
            &activations
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let partials = segments * TOKENS * cols;
        let tile_out = upload(&cuda, &vec![0u8; partials * 4]);
        let tile_y = upload(&cuda, &vec![0u8; TOKENS * cols * 4]);
        let rows_out = upload(&cuda, &vec![0u8; partials * 4]);
        // One counter a 32-column, 16-row tile, shared by every launch below.
        let tickets = upload(
            &cuda,
            &vec![0u8; cols.div_ceil(32) * TOKENS.div_ceil(16) * 4],
        );
        let rows_y = upload(&cuda, &vec![0u8; TOKENS * cols * 4]);
        let finish = Finish {
            y: rows_y,
            tickets,
            epilogue,
            scale,
        };
        launch(
            &cuda, k, false, w, a, tile_out, cols, inner, TOKENS, segments, finish,
        );
        let tile = download_bits(&cuda, &tile_out, partials);
        finish_tile(
            &cuda,
            reduce,
            epilogue_kernel,
            tile_out,
            tile_y,
            TOKENS * cols,
            segments,
            epilogue,
            scale,
        );
        let finished = download_bits(&cuda, &tile_y, TOKENS * cols);
        let shape = format!("{cols} x {inner} split {segments} epilogue {epilogue}");

        // 1. Every row of an M-row launch is the tile's row, its slices and its finished outputs.
        for count in 1..=TOKENS {
            poison(&cuda, rows_y, count * cols);
            launch(
                &cuda, k, true, w, a, rows_out, cols, inner, count, segments, finish,
            );
            let got = download_bits(&cuda, &rows_out, segments * count * cols);
            for segment in (0..segments).filter(|_| segments > 1) {
                for token in 0..count {
                    let at = (segment * count + token) * cols;
                    let want = (segment * TOKENS + token) * cols;
                    if let Some(column) =
                        (0..cols).find(|column| got[at + column] != tile[want + column])
                    {
                        panic!(
                            "{shape}: row {token} of a {count}-row launch, slice {segment}, column \
                             {column}: {:e} where the tile has {:e}",
                            f32::from_bits(got[at + column]),
                            f32::from_bits(tile[want + column])
                        );
                    }
                }
            }
            let got = if segments > 1 {
                download_bits(&cuda, &rows_y, count * cols)
            } else {
                got
            };
            if let Some(at) = (0..count * cols).find(|at| got[*at] != finished[*at]) {
                panic!(
                    "{shape}: row {} of a {count}-row launch, column {}, finished: {:e} where the \
                     tile's reduce has {:e}",
                    at / cols,
                    at % cols,
                    f32::from_bits(got[at]),
                    f32::from_bits(finished[at])
                );
            }
        }
        // ...and the row launched alone, from its own pointer.
        for token in [0, 17, TOKENS - 1] {
            poison(&cuda, rows_y, cols);
            launch(
                &cuda,
                k,
                true,
                w,
                a.offset(token * inner * 4),
                rows_out,
                cols,
                inner,
                1,
                segments,
                finish,
            );
            let got = download_bits(&cuda, if segments > 1 { &rows_y } else { &rows_out }, cols);
            assert_eq!(
                got,
                finished[token * cols..(token + 1) * cols],
                "{shape}: row {token} alone"
            );
        }

        // 2. The slices summed in order, against f64.
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for token in 0..TOKENS {
            let row = &activations[token * inner..(token + 1) * inner];
            for column in 0..cols {
                let weight = &weights[column * inner..(column + 1) * inner];
                let reference: f64 = row
                    .iter()
                    .zip(weight)
                    .map(|(x, w)| f64::from(*x) * f64::from(bf16_bits_to_f32(*w)))
                    .sum();
                let mut sum = f32::from_bits(tile[token * cols + column]);
                for segment in 1..segments {
                    sum += f32::from_bits(tile[(segment * TOKENS + token) * cols + column]);
                }
                worst = worst.max((f64::from(sum) - reference).abs());
                scale = scale.max(reference.abs());
            }
        }
        eprintln!("{shape}: worst |error| {worst:.3e} against |reference| up to {scale:.3}");
        assert!(
            worst < ACCURACY * scale,
            "{shape}: error {worst:.3e} past {ACCURACY:e} of {scale:.3}"
        );

        // Not a claim: the one-row launch time of each form, for the record.
        for rows in [false, true] {
            launch(
                &cuda, k, rows, w, a, rows_out, cols, inner, 1, segments, finish,
            );
            cuda.synchronize().expect("synchronize");
            let start = std::time::Instant::now();
            for _ in 0..50 {
                launch(
                    &cuda, k, rows, w, a, rows_out, cols, inner, 1, segments, finish,
                );
            }
            cuda.synchronize().expect("synchronize");
            eprintln!(
                "{shape}: one row through {} in {:.1} us",
                if rows {
                    "q4p_dense_rows"
                } else {
                    "q4p_dense_mma"
                },
                start.elapsed().as_secs_f64() * 1e6 / 50.0
            );
        }
    }
}

#[test]
fn a_launch_of_several_matrices_gives_each_the_bits_of_its_own() {
    let Some(cuda) = open() else { return };
    let (_rows_module, rows_kernel) = cuda
        .load_named(embedded_prefill_cubin(), "q4p_dense_rows")
        .expect("q4p_dense_rows");
    const TOKENS: usize = 63;
    const INNER: usize = 2560;
    // A layer's projections of one input: the shared expert's gate and up beside the router, and
    // a wide pair, split as the release splits them and not.
    let shapes = [96usize, 512, 640, 2560];
    let mut rng = Lcg(0x9a0f);
    let a = upload(
        &cuda,
        &(0..TOKENS * INNER)
            .flat_map(|_| rng.next().to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let weights: Vec<Deviceptr> = shapes
        .iter()
        .map(|cols| {
            upload(
                &cuda,
                &(0..cols * INNER)
                    .flat_map(|_| bf16_bits(rng.next() * 0.05).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let tickets = upload(&cuda, &vec![0u8; 256 * 4 * 4]);
    for (segments, epilogue) in [(5usize, 1u32), (2, 0), (1, 2)] {
        let finish = Finish {
            y: Deviceptr(0),
            tickets,
            epilogue,
            scale: 1.3,
        };
        let buffers = |count: usize| -> Vec<(Deviceptr, Deviceptr)> {
            shapes
                .iter()
                .map(|cols| {
                    (
                        upload(&cuda, &vec![0u8; segments * count * cols * 4]),
                        upload(&cuda, &vec![0u8; count * cols * 4]),
                    )
                })
                .collect()
        };
        for count in [1, 7, 16, 63] {
            let (alone, together) = (buffers(count), buffers(count));
            for (i, &(out, y)) in alone.iter().enumerate() {
                launch_rows(
                    &cuda,
                    rows_kernel,
                    &[(weights[i], out, y, shapes[i])],
                    &[],
                    a,
                    INNER,
                    count,
                    segments,
                    finish,
                );
            }
            let group: Vec<_> = together
                .iter()
                .enumerate()
                .map(|(i, &(out, y))| (weights[i], out, y, shapes[i]))
                .collect();
            launch_rows(
                &cuda,
                rows_kernel,
                &group,
                &[],
                a,
                INNER,
                count,
                segments,
                finish,
            );
            for (i, cols) in shapes.iter().enumerate() {
                // An unsplit launch finishes into `out`; a split one into `y`.
                let pick = |(out, y): (Deviceptr, Deviceptr)| if segments > 1 { y } else { out };
                let want = download_bits(&cuda, &pick(alone[i]), count * cols);
                let got = download_bits(&cuda, &pick(together[i]), count * cols);
                assert_eq!(
                    got, want,
                    "matrix {i} ({cols} x {INNER}) of four, {count} rows split {segments}"
                );
            }
        }
    }
    eprintln!("a launch of four matrices gives each the bits of its own launch");
}

#[test]
fn a_rider_is_the_bits_of_its_own_gemm_rows_launch() {
    let Some(cuda) = open() else { return };
    let (_rows_module, rows_kernel) = cuda
        .load_named(embedded_prefill_cubin(), "q4p_dense_rows")
        .expect("q4p_dense_rows");
    let (_gemm_module, gemm_rows) = cuda
        .load_named(embedded_prefill_cubin(), "q4p_gemm_rows")
        .expect("q4p_gemm_rows");
    const TOKENS: usize = 63;
    const INNER: usize = 2560;
    // Matrices off the tile beside the ones on it: the GDN gates (48 rows), the injection (4) and
    // the shared-expert gate (1), at the token GEMM's split and at others, with each epilogue.
    let dense = [512usize, 640];
    let off_tile = [(48usize, 48usize, 0u32), (4, 48, 2), (1, 7, 1), (48, 1, 2)];
    let mut rng = Lcg(0x51de);
    let a = upload(
        &cuda,
        &(0..TOKENS * INNER)
            .flat_map(|_| rng.next().to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let matrix = |rows: usize, rng: &mut Lcg| {
        upload(
            &cuda,
            &(0..rows * INNER)
                .flat_map(|_| bf16_bits(rng.next() * 0.05).to_le_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let dense_weights: Vec<Deviceptr> = dense.iter().map(|&cols| matrix(cols, &mut rng)).collect();
    let rider_weights: Vec<Deviceptr> = off_tile
        .iter()
        .map(|&(rows, ..)| matrix(rows, &mut rng))
        .collect();
    let tickets = upload(&cuda, &vec![0u8; 256 * 4 * 4]);
    let rider = |slot: usize, count: usize| {
        let (rows, split, epilogue) = off_tile[slot];
        Rider {
            w: rider_weights[slot],
            y: upload(&cuda, &vec![0u8; count * rows * 4]),
            rows,
            epilogue,
            scale: 1.7,
            split,
            partials: upload(&cuda, &vec![0u8; split * count * rows * 4]),
            tickets: upload(&cuda, &vec![0u8; (rows * count).div_ceil(128) * 4]),
        }
    };
    for segments in [5usize, 1] {
        let finish = Finish {
            y: Deviceptr(0),
            tickets,
            epilogue: 0,
            scale: 1.0,
        };
        for pair in [[0, 1], [2, 3]] {
            for count in [1, 7, 16, 63] {
                let buffers = || -> Vec<(Deviceptr, Deviceptr)> {
                    dense
                        .iter()
                        .map(|cols| {
                            (
                                upload(&cuda, &vec![0u8; segments * count * cols * 4]),
                                upload(&cuda, &vec![0u8; count * cols * 4]),
                            )
                        })
                        .collect()
                };
                let (alone, together) = (buffers(), buffers());
                let group = |buffers: &[(Deviceptr, Deviceptr)]| -> Vec<_> {
                    buffers
                        .iter()
                        .enumerate()
                        .map(|(i, &(out, y))| (dense_weights[i], out, y, dense[i]))
                        .collect()
                };
                let mine = pair.map(|slot| rider(slot, count));
                let riding = pair.map(|slot| rider(slot, count));
                for rider in mine {
                    poison(&cuda, rider.y, count * rider.rows);
                    launch_gemm_rows(&cuda, gemm_rows, rider, a, INNER, count);
                }
                launch_rows(
                    &cuda,
                    rows_kernel,
                    &group(&alone),
                    &[],
                    a,
                    INNER,
                    count,
                    segments,
                    finish,
                );
                for rider in riding {
                    poison(&cuda, rider.y, count * rider.rows);
                }
                launch_rows(
                    &cuda,
                    rows_kernel,
                    &group(&together),
                    &riding,
                    a,
                    INNER,
                    count,
                    segments,
                    finish,
                );
                for (i, cols) in dense.iter().enumerate() {
                    let pick =
                        |(out, y): (Deviceptr, Deviceptr)| if segments > 1 { y } else { out };
                    assert_eq!(
                        download_bits(&cuda, &pick(together[i]), count * cols),
                        download_bits(&cuda, &pick(alone[i]), count * cols),
                        "dense matrix {i} ({cols} x {INNER}) beside riders {pair:?}, {count} rows \
                         split {segments}"
                    );
                }
                for (alone, beside) in mine.iter().zip(&riding) {
                    let (rows, split, epilogue) = (alone.rows, alone.split, alone.epilogue);
                    assert_eq!(
                        download_bits(&cuda, &beside.y, count * rows),
                        download_bits(&cuda, &alone.y, count * rows),
                        "a {rows} x {INNER} rider split {split} epilogue {epilogue}, {count} rows, \
                         beside a launch split {segments}"
                    );
                }
                // Every counter the riders used is back at zero for the next launch.
                for rider in riding {
                    let counters = (rider.rows * count).div_ceil(128);
                    assert!(
                        download_bits(&cuda, &rider.tickets, counters)
                            .iter()
                            .all(|c| *c == 0),
                        "a rider left its tickets set"
                    );
                }
            }
        }
    }
    eprintln!("a rider's outputs are the bits of its own q4p_gemm_rows launch");
}

#[test]
fn a_few_rows_are_the_bits_the_token_gemm_computes() {
    let Some(cuda) = open() else { return };
    let (_tile_module, tile) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_bf16")
        .expect("q4e_gemm_bf16");
    let (_rows_module, rows_kernel) = cuda
        .load_named(embedded_prefill_cubin(), "q4p_gemm_rows")
        .expect("q4p_gemm_rows");
    let (_reduce_module, reduce) = cuda
        .load_named(embedded_cubin(), "q4e_gemm_reduce")
        .expect("q4e_gemm_reduce");
    const TOKENS: usize = 63;
    // (rows, columns, segments, epilogue): the four-row injection, the GDN gates, the one-row
    // shared-expert gate, a split whose last slice is short, an unaligned inner dimension.
    for (rows, columns, segments, epilogue) in [
        (4usize, 2560usize, 48usize, 2u32),
        (48, 2560, 48, 0),
        (1, 2560, 48, 0),
        (96, 6144, 7, 0),
        (5, 100, 1, 1),
    ] {
        let mut rng = Lcg((rows * 131 + columns + segments) as u64);
        let weights: Vec<u16> = (0..rows * columns)
            .map(|_| bf16_bits(rng.next() * 0.05))
            .collect();
        let activations: Vec<f32> = (0..TOKENS * columns).map(|_| rng.next()).collect();
        let w = upload(
            &cuda,
            &weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let a = upload(
            &cuda,
            &activations
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let outputs = segments * TOKENS * rows;
        let (tile_y, tile_partials) = (
            upload(&cuda, &vec![0u8; outputs * 4]),
            upload(&cuda, &vec![0u8; outputs * 4]),
        );
        let (rows_y, rows_partials) = (
            upload(&cuda, &vec![0u8; outputs * 4]),
            upload(&cuda, &vec![0u8; outputs * 4]),
        );
        let scale = 1.7f32;
        // One counter a 128-output tile, shared by every launch below.
        let tickets = upload(&cuda, &vec![0u8; (rows * TOKENS).div_ceil(128) * 4]);
        let run = |kernel,
                   grid: (usize, usize),
                   y: Deviceptr,
                   partials: Deviceptr,
                   count: usize,
                   arity: usize| {
            let args = [
                w.0,
                a.0,
                y.0,
                rows as u64,
                columns as u64,
                count as u64,
                u64::from(epilogue),
                u64::from(scale.to_bits()),
                segments as u64,
                if segments > 1 { partials.0 } else { 0 },
                tickets.0,
            ];
            cuda.launch(
                kernel,
                (grid.0 as u32, grid.1 as u32, 1),
                (128, 1, 1),
                &args[..arity],
            )
            .expect("the launch");
        };
        run(
            tile,
            (rows.div_ceil(64), segments),
            tile_y,
            tile_partials,
            TOKENS,
            10,
        );
        if segments > 1 {
            cuda.launch(
                reduce,
                ((rows * TOKENS).div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                &[
                    tile_partials.0,
                    tile_y.0,
                    (rows * TOKENS) as u64,
                    segments as u64,
                    u64::from(epilogue),
                    u64::from(scale.to_bits()),
                ],
            )
            .expect("q4e_gemm_reduce");
        }
        let finished = download_bits(&cuda, &tile_y, TOKENS * rows);
        let want = download_bits(&cuda, &tile_partials, outputs);
        let shape = format!("{rows} x {columns} split {segments}");
        for count in 1..=TOKENS {
            poison(&cuda, rows_y, count * rows);
            run(
                rows_kernel,
                ((rows * count).div_ceil(128), segments),
                rows_y,
                rows_partials,
                count,
                11,
            );
            let got = download_bits(&cuda, &rows_y, count * rows);
            if let Some(at) = (0..count * rows).find(|at| got[*at] != finished[*at]) {
                panic!(
                    "{shape}: row {} of a {count}-row launch, output {}, finished: {:e} where the \
                     tile has {:e}",
                    at / rows,
                    at % rows,
                    f32::from_bits(got[at]),
                    f32::from_bits(finished[at])
                );
            }
            if segments == 1 {
                continue;
            }
            let got = download_bits(&cuda, &rows_partials, segments * count * rows);
            for segment in 0..segments {
                for token in 0..count {
                    let at = (segment * count + token) * rows;
                    let reference = (segment * TOKENS + token) * rows;
                    if let Some(row) = (0..rows).find(|row| got[at + row] != want[reference + row])
                    {
                        panic!(
                            "{shape}: row {token} of a {count}-row launch, slice {segment}, output \
                             {row}: {:e} where the tile has {:e}",
                            f32::from_bits(got[at + row]),
                            f32::from_bits(want[reference + row])
                        );
                    }
                }
            }
        }
        eprintln!("{shape}: every row of 1..={TOKENS}-row launches is the tile's");
    }
}

/// One routed-expert pass over the first `tokens` rows: q4p_split_bf16, q4p_moe_plan, then
/// gate_up and down in the 128-row panel form (`rows` false) or q4p_moe_rows'. Returns the SwiGLU
/// rows' two bf16 planes and the routed f32 partials, both [slot][column].
struct Moe {
    split: Kernel,
    plan: Kernel,
    panel: (Kernel, Kernel),
    rows: (Kernel, Kernel),
    table: Deviceptr,
    scales: Deviceptr,
    indices: Deviceptr,
    mixed: Deviceptr,
    probabilities: Deviceptr,
    x: Deviceptr,
    sorted: Deviceptr,
    items: Deviceptr,
    item_count: Deviceptr,
    activated: Deviceptr,
    partials: Deviceptr,
}

const MOE_HIDDEN: usize = 2560;
const MOE_INTERMEDIATE: usize = 640;
const MOE_EXPERTS: usize = 12;
/// The rows one plan item holds, the shared plan's `item_rows` for this arm.
const MOE_ITEM_ROWS: usize = 32;
const MOE_TOPK: usize = 10;

impl Moe {
    fn launch(&self, cuda: &Cuda, rows: bool, tokens: usize) {
        let (hidden, intermediate, topk) = (MOE_HIDDEN, MOE_INTERMEDIATE, MOE_TOPK);
        let (slots, elements) = (tokens * topk, tokens * hidden);
        let launch = |kernel, grid: (usize, usize), block: u32, args: &[u64]| {
            cuda.launch(
                kernel,
                (grid.0 as u32, grid.1 as u32, 1),
                (block, 1, 1),
                args,
            )
            .expect("the launch");
        };
        launch(
            self.split,
            ((elements / 2).div_ceil(256), 1),
            256,
            &[self.mixed.0, self.x.0, elements as u64],
        );
        launch(
            self.plan,
            (1, 1),
            1024,
            &[
                self.indices.0,
                self.sorted.0,
                self.items.0,
                self.item_count.0,
                0,
                slots as u64,
                MOE_EXPERTS as u64,
                MOE_TOPK as u64,
                MOE_ITEM_ROWS as u64,
                (slots.div_ceil(MOE_ITEM_ROWS) + MOE_EXPERTS.min(slots)) as u64,
                0,
                2,
                0,
                0,
                0,
                0,
                self.partials.0,
                hidden as u64,
            ],
        );
        let items = slots.div_ceil(32) + MOE_EXPERTS.min(slots);
        let ((gate_up, down), (gate_up_columns, down_columns), block) = if rows {
            (self.rows, (32, 64), 128)
        } else {
            (self.panel, (64, 128), 256)
        };
        launch(
            gate_up,
            (intermediate / gate_up_columns, items),
            block,
            &[
                self.table.0,
                self.scales.0,
                self.sorted.0,
                self.items.0,
                self.item_count.0,
                self.x.0,
                elements as u64,
                self.activated.0,
                (slots * intermediate) as u64,
                topk as u64,
                hidden as u64,
                intermediate as u64,
            ],
        );
        launch(
            down,
            (hidden / down_columns, items),
            block,
            &[
                self.table.0,
                self.scales.0,
                self.sorted.0,
                self.items.0,
                self.item_count.0,
                self.activated.0,
                (slots * intermediate) as u64,
                self.probabilities.0,
                self.partials.0,
                topk as u64,
                hidden as u64,
                intermediate as u64,
                MOE_EXPERTS as u64,
            ],
        );
    }

    fn run(&self, cuda: &Cuda, rows: bool, tokens: usize) -> (Vec<u16>, Vec<u32>) {
        self.launch(cuda, rows, tokens);
        let slots = tokens * MOE_TOPK;
        let words = download_bits(cuda, &self.activated, slots * MOE_INTERMEDIATE);
        let halves = words
            .iter()
            .flat_map(|word| [*word as u16, (*word >> 16) as u16])
            .collect();
        (
            halves,
            download_bits(cuda, &self.partials, slots * MOE_HIDDEN),
        )
    }
}

#[test]
fn a_few_rows_are_the_bits_the_expert_panel_computes_and_within_the_split_bound() {
    let Some(cuda) = open() else { return };
    use sparkflash_kernels::compressed_cache_ref::{e2m1_decode, e4m3_decode};
    let load = |name| {
        cuda.load_named(embedded_prefill_cubin(), name)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
    };
    let modules: Vec<_> = [
        "q4p_split_bf16",
        "q4p_moe_gate_up",
        "q4p_moe_down",
        "q4p_moe_rows_gate_up",
        "q4p_moe_rows_down",
    ]
    .into_iter()
    .map(load)
    .collect();
    // The plan is the shared image's (kernels/moe_plan_sm121.cu), not the prefill image's.
    let plan = cuda
        .load_named(sparkflash_kernels::images::moe_plan_cubin(), "moe_plan")
        .unwrap_or_else(|error| panic!("moe_plan: {error}"))
        .1;
    let kernel = |i: usize| modules[i].1;
    // A cold prefill's chunk (past GEMM_MIN_BATCH, so the panel) and the decode-sized prefixes.
    const TOKENS: usize = 80;
    const ROWS: usize = 63;
    let (hidden, intermediate, topk) = (MOE_HIDDEN, MOE_INTERMEDIATE, MOE_TOPK);
    let mut rng = Lcg(0x5eed);
    // Per expert: gate, up and down as E2M1 codes with an E4M3 scale every 16 columns, the
    // weight_scale_2 of each, and the table q4p_moe_* reads them through.
    let mut code = |count: usize| -> Vec<u8> {
        (0..count)
            .map(|_| ((rng.next() * 0.5 + 0.5) * 255.99) as u8)
            .collect()
    };
    let matrices: Vec<[Vec<u8>; 6]> = (0..MOE_EXPERTS)
        .map(|_| {
            let scale = |codes: Vec<u8>| codes.into_iter().map(|c| 0x28 + (c & 0x1f)).collect();
            [
                code(intermediate * hidden / 2),
                scale(code(intermediate * hidden / 16)),
                code(intermediate * hidden / 2),
                scale(code(intermediate * hidden / 16)),
                code(hidden * intermediate / 2),
                scale(code(hidden * intermediate / 16)),
            ]
        })
        .collect();
    let pointers: Vec<Deviceptr> = matrices
        .iter()
        .flatten()
        .map(|m| upload(&cuda, m))
        .collect();
    let table: Vec<u8> = pointers.iter().flat_map(|p| p.0.to_le_bytes()).collect();
    let weight_scale_2: Vec<f32> = (0..MOE_EXPERTS * 3)
        .map(|i| [0.004, 0.003, 0.01][i % 3] * (1.0 + 0.1 * rng.next()))
        .collect();
    // Every token routes to topk distinct experts with weights in [0, 0.3).
    let mut indices = Vec::with_capacity(TOKENS * topk);
    for _ in 0..TOKENS {
        let mut experts: Vec<i32> = (0..MOE_EXPERTS as i32).collect();
        for i in 0..topk {
            let j = i + ((rng.next() * 0.5 + 0.5) * (MOE_EXPERTS - i) as f32) as usize;
            experts.swap(i, j.min(MOE_EXPERTS - 1));
        }
        indices.extend_from_slice(&experts[..topk]);
    }
    let probabilities: Vec<f32> = (0..TOKENS * MOE_EXPERTS)
        .map(|_| (rng.next() * 0.5 + 0.5) * 0.3)
        .collect();
    let mixed: Vec<f32> = (0..TOKENS * hidden).map(|_| rng.next()).collect();
    let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
    let slots = TOKENS * topk;
    let moe = Moe {
        split: kernel(0),
        plan,
        panel: (kernel(1), kernel(2)),
        rows: (kernel(3), kernel(4)),
        table: upload(&cuda, &table),
        scales: upload(&cuda, &f32_bytes(&weight_scale_2)),
        indices: upload(
            &cuda,
            &indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        mixed: upload(&cuda, &f32_bytes(&mixed)),
        probabilities: upload(&cuda, &f32_bytes(&probabilities)),
        x: upload(&cuda, &vec![0u8; 2 * TOKENS * hidden * 2]),
        sorted: upload(&cuda, &vec![0u8; slots * 4]),
        items: upload(&cuda, &vec![0u8; (slots.div_ceil(32) + MOE_EXPERTS) * 16]),
        item_count: upload(&cuda, &[0u8; 4]),
        activated: upload(&cuda, &vec![0u8; 2 * slots * intermediate * 2]),
        partials: upload(&cuda, &vec![0u8; slots * hidden * 4]),
    };
    let (panel_activated, panel_partials) = moe.run(&cuda, false, TOKENS);
    let plane = slots * intermediate;

    // 1. Every slot of a 1..=63-row pass is the panel's slot, both planes and the partials.
    for count in 1..=ROWS {
        let (activated, partials) = moe.run(&cuda, true, count);
        let used = count * topk;
        for slot in 0..used {
            for (name, at, want) in [
                ("head", slot * intermediate, slot * intermediate),
                (
                    "remainder",
                    used * intermediate + slot * intermediate,
                    plane + slot * intermediate,
                ),
            ] {
                if let Some(column) = (0..intermediate)
                    .find(|column| activated[at + column] != panel_activated[want + column])
                {
                    panic!(
                        "slot {slot} of a {count}-row pass, SwiGLU {name} column {column}: \
                         {:#06x} where the panel has {:#06x}",
                        activated[at + column],
                        panel_activated[want + column]
                    );
                }
            }
            let at = slot * hidden;
            if let Some(column) =
                (0..hidden).find(|column| partials[at + column] != panel_partials[at + column])
            {
                panic!(
                    "slot {slot} of a {count}-row pass, partial column {column}: {:e} where the \
                     panel has {:e}",
                    f32::from_bits(partials[at + column]),
                    f32::from_bits(panel_partials[at + column])
                );
            }
        }
    }
    eprintln!("routed experts: every slot of 1..={ROWS}-row passes is the panel's");

    // 2. The panel against f64 over the first slots: gate_up from the f32 activations, down from
    // the SwiGLU rows gate_up wrote (head + remainder).
    let e2m1: Vec<f64> = (0..16u8).map(|c| f64::from(e2m1_decode(c))).collect();
    let e4m3: Vec<f64> = (0..=255u8).map(|c| f64::from(e4m3_decode(c))).collect();
    let dot = |packed: &[u8], scales: &[u8], row: usize, inner: usize, x: &dyn Fn(usize) -> f64| {
        (0..inner)
            .map(|k| {
                let byte = packed[row * inner / 2 + k / 2];
                let nibble = if k % 2 == 0 { byte & 15 } else { byte >> 4 };
                e2m1[usize::from(nibble)]
                    * e4m3[usize::from(scales[row * inner / 16 + k / 16])]
                    * x(k)
            })
            .sum::<f64>()
    };
    let half = |bits: u16| f64::from(bf16_bits_to_f32(bits));
    let (mut worst, mut scale) = ([0.0f64; 2], [0.0f64; 2]);
    for slot in 0..40 {
        let (token, expert) = (slot / topk, indices[slot] as usize);
        let m = &matrices[expert];
        let s2 = &weight_scale_2[expert * 3..expert * 3 + 3];
        let row = &mixed[token * hidden..(token + 1) * hidden];
        let x = |k: usize| f64::from(row[k]);
        for column in 0..intermediate {
            let gate = dot(&m[0], &m[1], column, hidden, &x) * f64::from(s2[0]);
            let up = dot(&m[2], &m[3], column, hidden, &x) * f64::from(s2[1]);
            let reference = gate / (1.0 + (-gate).exp()) * up;
            let at = slot * intermediate + column;
            let got = half(panel_activated[at]) + half(panel_activated[plane + at]);
            worst[0] = worst[0].max((got - reference).abs());
            scale[0] = scale[0].max(reference.abs());
        }
        let route = f64::from(probabilities[token * MOE_EXPERTS + slot % topk]);
        let activated = |k: usize| {
            let at = slot * intermediate + k;
            half(panel_activated[at]) + half(panel_activated[plane + at])
        };
        for column in 0..hidden {
            let reference =
                route * f64::from(s2[2]) * dot(&m[4], &m[5], column, intermediate, &activated);
            let got = f64::from(f32::from_bits(panel_partials[slot * hidden + column]));
            worst[1] = worst[1].max((got - reference).abs());
            scale[1] = scale[1].max(reference.abs());
        }
    }
    for (pass, (worst, scale)) in ["gate_up", "down"].iter().zip(worst.iter().zip(scale)) {
        eprintln!("routed {pass}: worst |error| {worst:.3e} against |reference| up to {scale:.3}");
        assert!(
            *worst < ACCURACY * scale,
            "routed {pass}: error {worst:.3e} past {ACCURACY:e} of {scale:.3}"
        );
    }

    // Not a claim: one row through each form, for the record.
    for rows in [false, true] {
        moe.launch(&cuda, rows, 1);
        cuda.synchronize().expect("synchronize");
        let start = std::time::Instant::now();
        for _ in 0..50 {
            moe.launch(&cuda, rows, 1);
        }
        cuda.synchronize().expect("synchronize");
        eprintln!(
            "routed experts: one row through {} in {:.1} us",
            if rows { "q4p_moe_rows" } else { "the panel" },
            start.elapsed().as_secs_f64() * 1e6 / 50.0
        );
    }
}
