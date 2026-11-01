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

//! The DFlash 2 drafter's own kernels (kernels/dflash2_sm121.cu) against host arithmetic: the
//! two-stage top-k at the released vocabulary, the selector's greedy walk, the block
//! convolution, and the ring attention across a wrapped ring.
//!
//! Each test launches the kernel on chosen buffers against the embedded cubin, so a wrong index
//! or a wrong window shows up here in a second rather than as a low acceptance rate behind a
//! 77.8 GiB upload. They skip - loudly - without an embedded cubin or a CUDA driver.

use half::bf16;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_dflash2_cubin};

fn open() -> Option<Cuda> {
    if !cubin_available() || embedded_dflash2_cubin().is_empty() {
        eprintln!("skipped: this build embeds no Qwen4-Exp DFlash 2 cubin");
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

fn upload_i32(cuda: &Cuda, values: &[i32]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn upload_bf16(cuda: &Cuda, values: &[bf16]) -> Deviceptr {
    upload(
        cuda,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn zeros(cuda: &Cuda, elements: usize) -> Deviceptr {
    upload(cuda, &vec![0u8; elements * 4])
}

fn download(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<[u8; 4]> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 4];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes.as_chunks::<4>().0.to_vec()
}

fn download_f32(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<f32> {
    download(cuda, pointer, elements)
        .into_iter()
        .map(f32::from_le_bytes)
        .collect()
}

fn download_i32(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<i32> {
    download(cuda, pointer, elements)
        .into_iter()
        .map(i32::from_le_bytes)
        .collect()
}

fn kernel(cuda: &Cuda, name: &'static str) -> sparkflash_kernels::cuda_driver::Kernel {
    cuda.load_named(embedded_dflash2_cubin(), name)
        .unwrap_or_else(|error| panic!("the cubin carries {name}: {error}"))
        .1
}

/// Each of `rows` rows' top `k` of `vocab` logits at `logits`, ids from `first_id`, through both
/// stages as the drafter runs them: `(values, ids)`, `[rows][k]` each.
fn device_top_k(
    cuda: &Cuda,
    logits: &Deviceptr,
    (vocab, rows, k): (usize, usize, usize),
    first_id: usize,
) -> (Deviceptr, Deviceptr) {
    let chunk = 2048_usize;
    let parts = vocab.div_ceil(chunk);
    let part_values = zeros(cuda, rows * parts * k);
    let part_ids = zeros(cuda, rows * parts * k);
    let values = zeros(cuda, rows * k);
    let ids = zeros(cuda, rows * k);
    cuda.launch(
        kernel(cuda, "q4d_topk_partial"),
        (parts as u32, rows as u32, 1),
        (256, 1, 1),
        &[
            logits.0,
            part_values.0,
            part_ids.0,
            vocab as u64,
            chunk as u64,
            k as u64,
            first_id as u64,
        ],
    )
    .expect("partial launch");
    let threads = ((parts * k).div_ceil(8).next_multiple_of(32)).max(256);
    cuda.launch(
        kernel(cuda, "q4d_topk_final"),
        (rows as u32, 1, 1),
        (threads as u32, 1, 1),
        &[
            part_values.0,
            part_ids.0,
            values.0,
            ids.0,
            parts as u64,
            k as u64,
            (parts * k) as u64,
            k as u64,
        ],
    )
    .expect("final launch");
    (values, ids)
}

/// Coarse logits, so ties are common and the lower id has to win them; one NaN a row, which must
/// never be picked, and a clear winner at the very end of the vocabulary, in the last, partial
/// chunk.
fn tied_logits(vocab: usize, rows: usize) -> Vec<f32> {
    let mut rng = Lcg(7);
    let mut logits: Vec<f32> = (0..vocab * rows)
        .map(|_| (rng.next() * 64.0).round() / 4.0)
        .collect();
    for row in 0..rows {
        logits[row * vocab + 1000 + row] = f32::NAN;
        logits[row * vocab + vocab - 1 - row] = 100.0 + row as f32;
    }
    logits
}

/// Asserts each row's `(values, ids)` are a host sort's top `k`: the larger value, then the lower
/// id, NaNs never.
fn assert_host_top_k(
    logits: &[f32],
    (vocab, rows, k): (usize, usize, usize),
    got_values: &[f32],
    got_ids: &[i32],
) {
    for row in 0..rows {
        let line = &logits[row * vocab..(row + 1) * vocab];
        let mut ranked: Vec<usize> = (0..vocab).filter(|&id| !line[id].is_nan()).collect();
        ranked.sort_by(|&a, &b| line[b].total_cmp(&line[a]).then(a.cmp(&b)));
        let expected: Vec<i32> = ranked[..k].iter().map(|&id| id as i32).collect();
        assert_eq!(
            &got_ids[row * k..(row + 1) * k],
            expected.as_slice(),
            "row {row}"
        );
        for (slot, &id) in expected.iter().enumerate() {
            assert_eq!(
                got_values[row * k + slot],
                line[id as usize],
                "row {row} slot {slot}"
            );
        }
    }
}

#[test]
fn the_top_k_matches_a_host_sort_at_the_released_vocabulary() {
    let Some(cuda) = open() else {
        return;
    };
    let shape @ (vocab, rows, k) = (248_320_usize, 7_usize, 16_usize);
    let logits = tied_logits(vocab, rows);
    let (values, ids) = device_top_k(&cuda, &upload_f32(&cuda, &logits), shape, 0);
    assert_host_top_k(
        &logits,
        shape,
        &download_f32(&cuda, &values, rows * k),
        &download_i32(&cuda, &ids, rows * k),
    );
}

/// At TP each rank scores its slice of the vocabulary: its candidates carry the slice's global
/// ids, and the merge of the ranks' candidates, gathered `[ranks][rows][k]`, is the top k of the
/// whole vocabulary.
#[test]
fn the_ranks_slices_merge_to_the_whole_vocabularys_top_k() {
    let Some(cuda) = open() else {
        return;
    };
    let shape @ (vocab, rows, k) = (154_880_usize, 7_usize, 16_usize);
    let (world, local) = (4_usize, 154_880 / 4);
    let logits = tied_logits(vocab, rows);
    let mut gathered_values = Vec::with_capacity(world * rows * k);
    let mut gathered_ids = Vec::with_capacity(world * rows * k);
    for rank in 0..world {
        let slice: Vec<f32> = (0..rows)
            .flat_map(|row| {
                logits[row * vocab + rank * local..row * vocab + (rank + 1) * local].to_vec()
            })
            .collect();
        let (values, ids) = device_top_k(
            &cuda,
            &upload_f32(&cuda, &slice),
            (local, rows, k),
            rank * local,
        );
        gathered_values.extend(download_f32(&cuda, &values, rows * k));
        gathered_ids.extend(download_i32(&cuda, &ids, rows * k));
    }
    let part_values = upload_f32(&cuda, &gathered_values);
    let part_ids = upload_i32(&cuda, &gathered_ids);
    let values = zeros(&cuda, rows * k);
    let ids = zeros(&cuda, rows * k);
    cuda.launch(
        kernel(&cuda, "q4d_topk_final"),
        (rows as u32, 1, 1),
        (256, 1, 1),
        &[
            part_values.0,
            part_ids.0,
            values.0,
            ids.0,
            world as u64,
            k as u64,
            k as u64,
            (rows * k) as u64,
        ],
    )
    .expect("merge launch");
    assert_host_top_k(
        &logits,
        shape,
        &download_f32(&cuda, &values, rows * k),
        &download_i32(&cuda, &ids, rows * k),
    );
}

#[test]
fn the_selector_walks_the_best_scoring_chain() {
    let Some(cuda) = open() else {
        return;
    };
    let (vocab, rank, k, edges) = (1000_usize, 256_usize, 16_usize, 7_usize);
    let mut rng = Lcg(11);
    let predecessor: Vec<bf16> = (0..vocab * rank)
        .map(|_| bf16::from_f32(rng.next() * 0.5))
        .collect();
    let successor: Vec<bf16> = (0..vocab * rank)
        .map(|_| bf16::from_f32(rng.next() * 0.5))
        .collect();
    let projected: Vec<f32> = (0..edges * rank).map(|_| rng.next()).collect();
    let unary: Vec<f32> = (0..edges * k).map(|_| rng.next() * 2.0).collect();
    let candidates: Vec<i32> = (0..edges * k)
        .map(|index| ((index * 61 + 17) % vocab) as i32)
        .collect();
    let anchor = 123_i32;
    let drafts = zeros(&cuda, edges);
    let confidences = zeros(&cuda, edges);
    cuda.launch(
        kernel(&cuda, "q4d_select"),
        (1, 1, 1),
        (256, 1, 1),
        &[
            upload_i32(&cuda, &candidates).0,
            upload_f32(&cuda, &unary).0,
            upload_f32(&cuda, &projected).0,
            upload_bf16(&cuda, &predecessor).0,
            upload_bf16(&cuda, &successor).0,
            drafts.0,
            confidences.0,
            anchor as u64,
            edges as u64,
            k as u64,
            rank as u64,
        ],
    )
    .expect("select launch");
    let chosen = download_i32(&cuda, &drafts, edges);
    let confidence = download_f32(&cuda, &confidences, edges);
    let mut previous = anchor as usize;
    for edge in 0..edges {
        let score = |c: usize| -> f64 {
            let to = candidates[edge * k + c] as usize;
            f64::from(unary[edge * k + c])
                + (0..rank)
                    .map(|r| {
                        f64::from(predecessor[previous * rank + r].to_f32())
                            * f64::from(projected[edge * rank + r])
                            * f64::from(successor[to * rank + r].to_f32())
                    })
                    .sum::<f64>()
        };
        let best = (0..k).map(score).fold(f64::NEG_INFINITY, f64::max);
        let slot = (0..k)
            .find(|&c| candidates[edge * k + c] == chosen[edge])
            .unwrap_or_else(|| panic!("edge {edge} chose {} outside its candidates", chosen[edge]));
        // Summation order differs from the host's, so a near-tie may go either way; the choice
        // must still be a best one.
        assert!(
            score(slot) >= best - 1e-4,
            "edge {edge}: chose a candidate scoring {} under the best {best}",
            score(slot)
        );
        // The chosen candidate's softmax share over the row, which the best score bounds below
        // by 1 / k.
        let share = 1.0 / (0..k).map(|c| (score(c) - score(slot)).exp()).sum::<f64>();
        assert!(
            (f64::from(confidence[edge]) - share).abs() < 1e-4 && share >= 1.0 / k as f64 - 1e-6,
            "edge {edge}: confidence {} against the host's {share}",
            confidence[edge]
        );
        previous = chosen[edge] as usize;
    }
}

#[test]
fn the_block_convolution_matches_the_host_on_both_sides() {
    let Some(cuda) = open() else {
        return;
    };
    let (hidden, group, taps, block, rows) = (64_usize, 16_usize, 2_usize, 8_usize, 16_usize);
    let groups = hidden / group;
    let stride = 2 * taps * groups;
    let mut rng = Lcg(5);
    let x: Vec<f32> = (0..rows * hidden).map(|_| rng.next()).collect();
    let coefficients: Vec<f32> = (0..rows * stride).map(|_| rng.next() * 0.25).collect();
    let base: Vec<bf16> = (0..2 * taps * hidden)
        .map(|_| bf16::from_f32(rng.next()))
        .collect();
    let device_x = upload_f32(&cuda, &x);
    let device_coefficients = upload_f32(&cuda, &coefficients);
    let device_base = upload_bf16(&cuda, &base);
    for side in 0..2_usize {
        let out = zeros(&cuda, rows * hidden);
        cuda.launch(
            kernel(&cuda, "q4d_conv"),
            (hidden.div_ceil(256) as u32, rows as u32, 1),
            (256, 1, 1),
            &[
                device_x.0,
                device_coefficients.0,
                device_base.0,
                out.0,
                hidden as u64,
                group as u64,
                taps as u64,
                side as u64,
                stride as u64,
                block as u64,
            ],
        )
        .expect("conv launch");
        let got = download_f32(&cuda, &out, rows * hidden);
        for row in 0..rows {
            for column in 0..hidden {
                let mut expected = 0.0_f32;
                for tap in 0..taps.min(row % block + 1) {
                    let kernel = side * taps + tap;
                    let weight = base[kernel * hidden + column].to_f32()
                        + coefficients[row * stride + kernel * groups + column / group];
                    expected += weight * x[(row - tap) * hidden + column];
                }
                let value = got[row * hidden + column];
                assert!(
                    (value - expected).abs() <= 1e-5,
                    "side {side} row {row} column {column}: {value} vs {expected}"
                );
            }
        }
    }
}

/// One block's attention on the host: row r at base + r sees the ring's positions back to
/// base + r - window_left and every block key.
#[allow(clippy::too_many_arguments)]
fn host_attention(
    queries: &[f32],
    ring_keys: &[f32],
    ring_values: &[f32],
    block_keys: &[f32],
    block_values: &[f32],
    heads: usize,
    kv_heads: usize,
    base: usize,
    window_left: usize,
    ring: usize,
    rows: usize,
) -> Vec<f32> {
    const D: usize = 128;
    let (q_width, kv_width) = (heads * D, kv_heads * D);
    let mut out = vec![0.0_f32; rows * q_width];
    let scale = 1.0 / (D as f64).sqrt();
    for row in 0..rows {
        let lowest = (base + row).saturating_sub(window_left);
        for head in 0..heads {
            let key_head = head / (heads / kv_heads);
            let query = &queries[row * q_width + head * D..][..D];
            let mut keys: Vec<(&[f32], &[f32])> = Vec::new();
            for position in lowest..base {
                let slot = position % ring;
                keys.push((
                    &ring_keys[slot * kv_width + key_head * D..][..D],
                    &ring_values[slot * kv_width + key_head * D..][..D],
                ));
            }
            for own in 0..rows {
                keys.push((
                    &block_keys[own * kv_width + key_head * D..][..D],
                    &block_values[own * kv_width + key_head * D..][..D],
                ));
            }
            let scores: Vec<f64> = keys
                .iter()
                .map(|(key, _)| {
                    query
                        .iter()
                        .zip(key.iter())
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..D {
                let value: f64 = keys
                    .iter()
                    .zip(&weights)
                    .map(|((_, v), w)| w * f64::from(v[d]))
                    .sum();
                out[row * q_width + head * D + d] = (value / total) as f32;
            }
        }
    }
    out
}

#[test]
fn the_ring_attention_matches_the_host_across_a_wrapped_ring() {
    let Some(cuda) = open() else {
        return;
    };
    const D: usize = 128;
    let (heads, kv_heads, rows, window_left) = (4_usize, 2_usize, 8_usize, 20_usize);
    let ring = window_left + 1 + rows;
    // A base past the ring's length wraps it; a base under the window starts at position zero.
    for base in [5_usize, 45] {
        let mut rng = Lcg(base as u64);
        let q_width = heads * D;
        let kv_width = kv_heads * D;
        let queries: Vec<f32> = (0..rows * q_width).map(|_| rng.next()).collect();
        let block_keys: Vec<f32> = (0..rows * kv_width).map(|_| rng.next()).collect();
        let block_values: Vec<f32> = (0..rows * kv_width).map(|_| rng.next()).collect();
        // Every committed position written at its slot in order, so later ones overwrite.
        let mut ring_keys = vec![0.0_f32; ring * kv_width];
        let mut ring_values = vec![0.0_f32; ring * kv_width];
        for position in 0..base {
            let slot = position % ring;
            for column in 0..kv_width {
                ring_keys[slot * kv_width + column] = rng.next();
                ring_values[slot * kv_width + column] = rng.next();
            }
        }
        let out = zeros(&cuda, rows * q_width);
        let scale = 1.0_f32 / (D as f32).sqrt();
        cuda.launch(
            kernel(&cuda, "q4d_attn"),
            (heads as u32, rows as u32, 1),
            (256, 1, 1),
            &[
                upload_f32(&cuda, &queries).0,
                upload_f32(&cuda, &ring_keys).0,
                upload_f32(&cuda, &ring_values).0,
                upload_f32(&cuda, &block_keys).0,
                upload_f32(&cuda, &block_values).0,
                out.0,
                heads as u64,
                kv_heads as u64,
                D as u64,
                base as u64,
                window_left as u64,
                ring as u64,
                rows as u64,
                u64::from(scale.to_bits()),
            ],
        )
        .expect("attention launch");
        let got = download_f32(&cuda, &out, rows * q_width);
        let expected = host_attention(
            &queries,
            &ring_keys,
            &ring_values,
            &block_keys,
            &block_values,
            heads,
            kv_heads,
            base,
            window_left,
            ring,
            rows,
        );
        for (index, (value, want)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (value - want).abs() <= 1e-4,
                "base {base} element {index}: {value} vs {want}"
            );
        }
    }
}
