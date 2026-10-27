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

//! Off-by-default capture hook for the released reference attention (task-14 / 2B-i(a)).
//!
//! The first device evidence for CSA-union attention (kernel format 2) needs the reference's
//! attention inputs and output for one query position, captured from the real reference forward
//! rather than reconstructed: the rotated query, the window KV the reference stores, the
//! compressed rows the query reaches, the engine selected-id list, the per-head sink, and the
//! reference attention output. This module is the hook that records exactly those values.
//!
//! The hook is a thread-local recorder that is inactive unless [begin] is called. Every value it
//! records is copied from the arrays the reference pass already computed; it introduces no
//! arithmetic and no state into the attention path. [finish] returns the records and disarms the
//! hook. The reference forward's own output is therefore bit-identical whether the hook is active
//! or not; the sequence-fixture regeneration test proves the default (inactive) path is unchanged.
//!
//! The hook records the pre-projection attention output: the per-head concatenated latent that
//! `sparse_attention` writes, before the inverse RoPE and the grouped output projection. That is the
//! value the flash_mla kernel computes, and the fixture comparison is over it.
//!
//! # What a record means
//!
//! For a prefill chunk (`start_pos == 0`), [AttentionCaseRecord::window_ids] are token positions in
//! the chunk, [AttentionCaseRecord::window_kv] holds the chunk's window KV rows (positions in
//! order), and [AttentionCaseRecord::compressed_ids] are the reference's shared index-space ids
//! (window-row offset plus compressed row, with [NO_INDEX] entries). `window_rows` is the number of
//! window KV rows in the buffer the reference built, and `window_width` is how many ids the query
//! actually carries; for a released 130-token prefill they are 130 and 128. The capture is intended
//! for a prefill query; a decode step records the same fields with ring-slot window ids, which a
//! fixture writer must interpret rather than treat as token positions.

use std::cell::RefCell;

/// The sentinel the reference uses for an index that attends nothing (model.py:420, 580).
pub const NO_INDEX: usize = usize::MAX;

/// One captured attention frame: the reference's inputs and pre-projection output for one query.
///
/// Every vector is in the reference's own layout: `q` is `[heads][head_dim]` rotated query values,
/// `sink` is one logit per head, `window_ids` is one id per attended window slot, `window_kv` is
/// `window_rows * head_dim` values in buffer order, `compressed_ids` is one shared-index-space id
/// per compressed slot (including [NO_INDEX]), `compressed_kv` is the compressed buffer's
/// `compressed_rows * head_dim` values in row order, and `out` is `[heads][head_dim]` values written
/// by `sparse_attention`.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionCaseRecord {
    /// Backbone layer index.
    pub layer: usize,
    /// Absolute token position of the captured query.
    pub position: usize,
    /// First position of the chunk the query was in.
    pub start_pos: usize,
    /// Number of queries in that chunk.
    pub seqlen: usize,
    /// The layer's compression ratio (0 for a window-only layer).
    pub ratio: usize,
    /// Attention heads.
    pub heads: usize,
    /// Latent width per head.
    pub head_dim: usize,
    /// Number of window KV rows in the buffer the reference built.
    pub window_rows: usize,
    /// Number of window ids this query carries (the attended window width).
    pub window_width: usize,
    /// Number of compressed ids this query carries.
    pub compressed_stride: usize,
    /// Number of compressed rows in the buffer the reference built.
    pub compressed_rows: usize,
    /// Rotated query, `[heads][head_dim]`.
    pub q: Vec<f32>,
    /// Per-head attention sink logits, `[heads]`.
    pub sink: Vec<f32>,
    /// The window ids the reference attends, in attention order.
    pub window_ids: Vec<usize>,
    /// The window KV buffer the ids index, `window_rows * head_dim`.
    pub window_kv: Vec<f32>,
    /// The compressed ids the reference attends, in attention order.
    pub compressed_ids: Vec<usize>,
    /// The compressed KV buffer the ids index, `compressed_rows * head_dim`.
    pub compressed_kv: Vec<f32>,
    /// The pre-projection attention output, `[heads][head_dim]`.
    pub out: Vec<f32>,
}

/// What [begin] armed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Request {
    layers: Vec<usize>,
    position: usize,
}

thread_local! {
    /// The armed request, or `None` when the hook is inactive.
    static ACTIVE: RefCell<Option<Request>> = const { RefCell::new(None) };
    /// Records captured since the last [begin].
    static RECORDS: RefCell<Vec<AttentionCaseRecord>> = const { RefCell::new(Vec::new()) };
}

/// Arms the hook for the given layers at one absolute query position.
///
/// Calling this again replaces any previous request and clears its records. The hook is
/// thread-local, so it only captures the forward pass that runs on this thread.
///
/// # Errors
///
/// Returns a message when `layers` is empty; a capture with no layers could only be a mistake.
pub fn begin(layers: &[usize], position: usize) -> Result<(), String> {
    if layers.is_empty() {
        return Err("an attention-case capture needs at least one layer".to_string());
    }
    let request = Request {
        layers: layers.to_vec(),
        position,
    };
    ACTIVE.with(|active| *active.borrow_mut() = Some(request));
    RECORDS.with(|records| records.borrow_mut().clear());
    Ok(())
}

/// Disarms the hook and returns the records captured since [begin], in capture order.
pub fn finish() -> Vec<AttentionCaseRecord> {
    ACTIVE.with(|active| *active.borrow_mut() = None);
    RECORDS.with(|records| std::mem::take(&mut *records.borrow_mut()))
}

/// Whether the hook is armed on this thread.
#[must_use]
pub fn is_capturing() -> bool {
    ACTIVE.with(|active| active.borrow().is_some())
}

/// The values `attention_forward` offers to the hook for one query; borrowed so an inactive hook
/// costs one thread-local check and no copies.
pub(crate) struct Frame<'a> {
    /// Backbone layer index.
    pub layer: usize,
    /// Absolute token position of this query.
    pub position: usize,
    /// First position of the chunk.
    pub start_pos: usize,
    /// Queries in the chunk.
    pub seqlen: usize,
    /// The layer's compression ratio.
    pub ratio: usize,
    /// Attention heads.
    pub heads: usize,
    /// Latent width per head.
    pub head_dim: usize,
    /// Window KV rows in the buffer.
    pub window_rows: usize,
    /// Window ids this query carries.
    pub window_width: usize,
    /// Compressed ids this query carries.
    pub compressed_stride: usize,
    /// Rotated query row, `[heads][head_dim]`.
    pub q: &'a [f32],
    /// Per-head sink logits.
    pub sink: &'a [f32],
    /// Window ids for this query.
    pub window_ids: &'a [usize],
    /// Window KV buffer.
    pub window_kv: &'a [f32],
    /// Compressed ids for this query.
    pub compressed_ids: &'a [usize],
    /// Compressed KV buffer.
    pub compressed_kv: &'a [f32],
    /// Pre-projection output row, `[heads][head_dim]`.
    pub out: &'a [f32],
}

/// Records one frame when the hook is armed for its layer and position; a no-op otherwise.
pub(crate) fn record(frame: Frame<'_>) {
    ACTIVE.with(|active| {
        let active = active.borrow();
        let Some(request) = active.as_ref() else {
            return;
        };
        if !request.layers.contains(&frame.layer) || request.position != frame.position {
            return;
        }
        let compressed_rows = frame.compressed_kv.len() / frame.head_dim.max(1);
        RECORDS.with(|records| {
            records.borrow_mut().push(AttentionCaseRecord {
                layer: frame.layer,
                position: frame.position,
                start_pos: frame.start_pos,
                seqlen: frame.seqlen,
                ratio: frame.ratio,
                heads: frame.heads,
                head_dim: frame.head_dim,
                window_rows: frame.window_rows,
                window_width: frame.window_width,
                compressed_stride: frame.compressed_stride,
                compressed_rows,
                q: frame.q.to_vec(),
                sink: frame.sink.to_vec(),
                window_ids: frame.window_ids.to_vec(),
                window_kv: frame.window_kv.to_vec(),
                compressed_ids: frame.compressed_ids.to_vec(),
                compressed_kv: frame.compressed_kv.to_vec(),
                out: frame.out.to_vec(),
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::{AttentionCaseRecord, Frame, NO_INDEX, begin, finish, is_capturing, record};

    fn frame(layer: usize, position: usize) -> Frame<'static> {
        static Q: [f32; 4] = [0.25, -0.5, 0.75, -1.0];
        static SINK: [f32; 2] = [0.5, -0.25];
        static WINDOW_IDS: [usize; 2] = [1, 2];
        static WINDOW_KV: [f32; 8] = [0.0; 8];
        static COMPRESSED_IDS: [usize; 2] = [4, NO_INDEX];
        static COMPRESSED_KV: [f32; 8] = [0.0; 8];
        static OUT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        Frame {
            layer,
            position,
            start_pos: 0,
            seqlen: 4,
            ratio: 2,
            heads: 2,
            head_dim: 2,
            window_rows: 4,
            window_width: 2,
            compressed_stride: 2,
            q: &Q,
            sink: &SINK,
            window_ids: &WINDOW_IDS,
            window_kv: &WINDOW_KV,
            compressed_ids: &COMPRESSED_IDS,
            compressed_kv: &COMPRESSED_KV,
            out: &OUT,
        }
    }

    #[test]
    fn the_hook_is_inactive_until_armed_and_filters_by_layer_and_position() {
        finish();
        assert!(!is_capturing());
        record(frame(7, 5));
        assert!(finish().is_empty());

        begin(&[7], 5).expect("arme");
        assert!(is_capturing());
        record(frame(6, 5)); // wrong layer
        record(frame(7, 4)); // wrong position
        record(frame(7, 5)); // captured
        record(frame(7, 5)); // captured twice (the hook does not deduplicate)
        let records: Vec<AttentionCaseRecord> = finish();
        assert!(!is_capturing());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].layer, 7);
        assert_eq!(records[0].compressed_rows, 4);
        assert_eq!(records[0].compressed_ids, vec![4, NO_INDEX]);
        assert_eq!(records[1].position, 5);

        assert!(begin(&[], 0).is_err(), "an empty layer list is refused");
    }
}
