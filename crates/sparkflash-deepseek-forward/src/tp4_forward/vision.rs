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

//! Images on the device path (DECISIONS.md D59): the vision tower, the span rows and the image
//! tokens' routing.
//!
//! Every position of an image span carries the placeholder id, so its embedding row is the
//! placeholder's, which lives in the [`crate::device_link::tower_rank`]'s vocabulary slice. That
//! rank also runs the vision tower: it encodes each image when the first prefill chunk that touches
//! its span runs, and between its embedding copy and the exchange it overwrites its own gather
//! slot's span rows with the delimiters (`image_start`, `image_newline`, `image_end`) and the
//! aligner rows. The exchange then hands them to the other ranks like any token's row, so nothing
//! else in the forward knows about images.
//!
//! The reference's gate picks an image-span token's experts with the layer's `bias_vl` instead of
//! its `bias` (model.py Gate). The router kernels route every row with `bias`; on every rank, the
//! chunk's span rows are then routed again with `bias_vl` (opt_vision_route) before the experts
//! read the choice. The Engram mask needs nothing here: the scheduler holds the span positions as
//! [`sparkflash_engram::HASH_SENTINEL`], which the hashes treat as the reference's masked tokens.

use super::*;
use crate::device_link::tower_rank;
use sparkflash_api::VisionImage;
use sparkflash_kernels::cuda_driver::{Kernel, Module};
use sparkflash_kernels::vision_tower::{Delimiter, ROW, VisionTower};
use sparkflash_model::vision::SpanRow;
use sparkflash_vision::Geometry;
use std::collections::HashMap;

/// The rows one encoded image holds at most: a span is at most `vision_max_n_token` positions.
const SLOT_ROWS: usize = Geometry::V41.max_tokens;

/// The vision path's device state.
pub(super) struct VisionState {
    _module: Module,
    route: Kernel,
    /// The tower, on the tower rank only.
    tower: Option<VisionTower>,
    /// The chunk's span rows (u32 offsets in the chunk) for the routing kernel.
    rows: Deviceptr,
    /// The tower rank's span-row table for the chunk: (offset in the chunk, source row address).
    table: Deviceptr,
    /// Rows a chunk may have.
    cap: usize,
    /// Encoded images' rows (bf16 [SLOT_ROWS][ROW] each), by (sequence, image).
    encoded: HashMap<(u64, usize), Deviceptr>,
    /// Slots no image holds, kept for the next one: freeing would wait for the device.
    free: Vec<Deviceptr>,
    /// Every slot allocated, for the release.
    slots: Vec<Deviceptr>,
    /// The chunk being run, between [`DeviceLayer::vision_begin`] and [`DeviceLayer::vision_end`]:
    /// its span rows in `rows` (every rank) and `table` (the tower rank).
    current: Option<usize>,
}

impl VisionState {
    /// Rows for the tower rank's warm-up canvas: the widest span (one token row) has the most
    /// patches, so the tower's scratch reaches its largest size at boot and never grows in service.
    fn widest() -> (usize, usize) {
        let cell = Geometry::V41.cell();
        (cell, (Geometry::V41.max_tokens - 3) * cell)
    }
}

impl DeviceLayer<'_> {
    /// Opens the vision path: the routing kernel on every rank, and on the tower rank the tower
    /// (its weights from `source`) with its scratch sized by one encode of the largest canvas.
    /// `cap` is the most rows a chunk has.
    ///
    /// # Errors
    /// The cubin, the weights or the driver failed: an epoch that serves images must have them.
    pub(super) fn open_vision(
        &mut self,
        dir: &Path,
        source: Option<&sparkflash_engine::safetensors::SafeTensorsSource>,
        cap: usize,
    ) -> Result<(), String> {
        let cuda = self.cuda;
        let path = dir.join(crate::device_link::VISION_CUBIN);
        let (cubin, _) = cached_cubin(&path).map_err(|error| format!("vision: {error}"))?;
        let (module, route) = cuda.load_named(&cubin, "opt_vision_route")?;
        let tower = if self.tp_rank == tower_rank() {
            let source = source.ok_or("vision: the tower rank has no checkpoint source")?;
            let started = std::time::Instant::now();
            let mut tower = VisionTower::load(cuda, &cubin, source)?;
            let (height, width) = VisionState::widest();
            let canvas = vec![sparkflash_vision::prep::PAD; height * width * 3];
            let mut total = 0usize;
            let out = cuda.alloc(SLOT_ROWS * ROW * 2, &mut total)?;
            let warmed = tower.encode(cuda, &canvas, height, width, out);
            cuda.synchronize()?;
            cuda.free_buffers(&[&out]);
            warmed?;
            eprintln!(
                "vision: the tower holds {} MiB of weights, ready in {:.1} s",
                tower.device_bytes() >> 20,
                started.elapsed().as_secs_f64()
            );
            Some(tower)
        } else {
            None
        };
        let mut total = 0usize;
        let rows = cuda.alloc(cap * 4, &mut total)?;
        let table = cuda.alloc(cap * 16, &mut total)?;
        self.vision = Some(Box::new(VisionState {
            _module: module,
            route,
            tower,
            rows,
            table,
            cap,
            encoded: HashMap::new(),
            free: Vec::new(),
            slots: Vec::new(),
            current: None,
        }));
        Ok(())
    }

    /// Whether the epoch serves images.
    pub(super) fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Prepares the chunk of rows `p..p + n` of sequence `seq`, whose prompt has `images`: on the
    /// tower rank it encodes the images whose spans the chunk reaches first and uploads the
    /// chunk's span-row table for [`Self::vision_inject`]; on every rank it sets the chunk's span
    /// rows for [`Self::vision_route`]. Nothing is set when the chunk has no span row.
    ///
    /// # Errors
    /// An image without its canvas on the tower rank, or the driver.
    pub(super) fn vision_begin(
        &mut self,
        seq: u64,
        images: &[VisionImage],
        p: usize,
        n: usize,
    ) -> Result<(), String> {
        let cuda = self.cuda;
        let Some(vision) = self.vision.as_mut() else {
            return if images.is_empty() {
                Ok(())
            } else {
                Err("this epoch does not run the vision tower".to_owned())
            };
        };
        let vision = &mut **vision;
        vision.current = None;
        let end = p + n;
        let touched: Vec<usize> = (0..images.len())
            .filter(|&index| {
                let span = &images[index].span;
                span.start < end && p < span.end()
            })
            .collect();
        if touched.is_empty() {
            return Ok(());
        }
        if n > vision.cap {
            return Err(format!(
                "vision: a chunk of {n} rows passes the {} the path holds",
                vision.cap
            ));
        }
        let mut offsets: Vec<u32> = Vec::new();
        let mut table: Vec<u64> = Vec::new();
        for &index in &touched {
            let image = &images[index];
            let span = &image.span;
            let source = match vision.tower.as_mut() {
                Some(tower) => {
                    let slot = match vision.encoded.get(&(seq, index)) {
                        Some(&slot) => slot,
                        None => {
                            let grid = span.grid;
                            if image.pixels.is_empty() {
                                return Err(format!(
                                    "vision: sequence {seq} image {index} came without its canvas"
                                ));
                            }
                            let slot = match vision.free.pop() {
                                Some(slot) => slot,
                                None => {
                                    let mut total = 0usize;
                                    let slot = cuda.alloc(SLOT_ROWS * ROW * 2, &mut total)?;
                                    vision.slots.push(slot);
                                    slot
                                }
                            };
                            vision.encoded.insert((seq, index), slot);
                            let shape = tower.encode(
                                cuda,
                                &image.pixels,
                                grid.best_height,
                                grid.best_width,
                                slot,
                            )?;
                            if (shape.n_llm_h, shape.n_llm_w) != (grid.n_llm_h, grid.n_llm_w) {
                                return Err(format!(
                                    "vision: sequence {seq} image {index}: the canvas makes a {} x {} grid, the span says {} x {}",
                                    shape.n_llm_w, shape.n_llm_h, grid.n_llm_w, grid.n_llm_h
                                ));
                            }
                            slot
                        }
                    };
                    Some((tower, slot))
                }
                None => None,
            };
            for pos in span.start.max(p)..span.end().min(end) {
                offsets.push((pos - p) as u32);
                if let Some((tower, slot)) = &source {
                    let row = match span.row(pos) {
                        SpanRow::Start => tower.delimiter(Delimiter::Start).0,
                        SpanRow::NewLine => tower.delimiter(Delimiter::NewLine).0,
                        SpanRow::End => tower.delimiter(Delimiter::End).0,
                        SpanRow::Image(i) => slot.0 + (i * ROW * 2) as u64,
                    };
                    table.extend([(pos - p) as u64, row]);
                }
            }
        }
        // A span done with this chunk gives its slot back; a later encode into it is enqueued
        // after this chunk's copy.
        for &index in &touched {
            if images[index].span.end() <= end
                && let Some(slot) = vision.encoded.remove(&(seq, index))
            {
                vision.free.push(slot);
            }
        }
        let bytes: Vec<u8> = offsets.iter().flat_map(|row| row.to_le_bytes()).collect();
        cuda.copy_htod(&vision.rows, &bytes)?;
        if vision.tower.is_some() {
            let bytes: Vec<u8> = table.iter().flat_map(|word| word.to_le_bytes()).collect();
            cuda.copy_htod(&vision.table, &bytes)?;
        }
        vision.current = Some(offsets.len());
        Ok(())
    }

    /// On the tower rank, between its embedding copy and the exchange: overwrites the chunk's span
    /// rows in its gather slot at `slot` with the delimiters and the aligner rows.
    ///
    /// # Errors
    /// The driver.
    pub(super) fn vision_inject(&self, slot: u64) -> Result<(), String> {
        let Some(vision) = self.vision.as_ref() else {
            return Ok(());
        };
        match (vision.tower.as_ref(), vision.current) {
            (Some(tower), Some(count)) => {
                tower.copy_rows(self.cuda, vision.table, Deviceptr(slot), count)
            }
            _ => Ok(()),
        }
    }

    /// Whether the rows being embedded carry image spans (vision_inject writes them into this
    /// rank's gather slot): the same answer on every rank.
    pub(super) fn vision_rows(&self) -> bool {
        self.vision
            .as_ref()
            .is_some_and(|vision| vision.current.is_some())
    }

    /// The chunk's layers ran: its span rows are no longer routed apart.
    pub(super) fn vision_end(&mut self) {
        if let Some(vision) = self.vision.as_mut() {
            vision.current = None;
        }
    }

    /// Sequence `seq` left the device (ended or preempted): its encoded images' slots go back.
    pub(super) fn vision_release(&mut self, seq: u64) {
        if let Some(vision) = self.vision.as_mut() {
            let held: Vec<(u64, usize)> = vision
                .encoded
                .keys()
                .filter(|(owner, _)| *owner == seq)
                .copied()
                .collect();
            for key in held {
                if let Some(slot) = vision.encoded.remove(&key) {
                    vision.free.push(slot);
                }
            }
        }
    }

    /// Routes the chunk's span rows again with layer `layer`'s `bias_vl`, after the router: `src`
    /// is the bulk router's raw dots (`raw`) or the rows router's scores.
    ///
    /// # Errors
    /// The layer has no `bias_vl`, or the driver.
    pub(super) fn vision_route(
        &self,
        layer: usize,
        src: u64,
        raw: bool,
        ids: u64,
        weights: u64,
    ) -> Result<(), String> {
        let Some(vision) = self.vision.as_ref() else {
            return Ok(());
        };
        let Some(count) = vision.current else {
            return Ok(());
        };
        let bias_vl = self
            .router_vl
            .as_ref()
            .and_then(|biases| biases.get(layer).copied())
            .ok_or_else(|| format!("vision route: layer {layer} has no bias_vl"))?;
        self.cuda.launch(
            vision.route,
            ((count as u32).div_ceil(8), 1, 1),
            (256, 1, 1),
            &[
                src,
                u64::from(raw),
                bias_vl.0,
                vision.rows.0,
                count as u64,
                ids,
                weights,
            ],
        )
    }

    /// Frees the vision path's device memory.
    pub(super) fn release_vision(&mut self) {
        let cuda = self.cuda;
        if let Some(vision) = self.vision.take() {
            let mut buffers = vec![&vision.rows, &vision.table];
            buffers.extend(vision.slots.iter());
            cuda.free_buffers(&buffers);
            if let Some(tower) = vision.tower {
                tower.free(cuda);
            }
        }
    }
}
