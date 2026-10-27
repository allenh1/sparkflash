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

//! The launch declaration of the compressed-cache device form.
//!
//! Two symbols, one source, one pin: compressed_append_sm121 (one block per completed compressed
//! row, the native FP4 quantiser of the twin bit for bit) and compressed_selected_fill_sm121 (the
//! expanded-store id mapping, host order kept). The host validates the plan - expanded rows inside
//! EXPANDED_ROWS, no rewrite - before any launch; the kernel assumes a valid plan the way the
//! store does.

/// The append symbol.
pub const KERNEL: &str = "compressed_append_sm121";
/// The selected-list symbol.
pub const SELECT_KERNEL: &str = "compressed_selected_fill_sm121";
/// The CUDA source both symbols are compiled from.
pub const SOURCE: &str = include_str!("../../../kernels/compressed_append_sm121.cu");
/// The append signature, whitespace-normalised for the declaration check.
pub const APPEND_DECLARATION: &str = "extern \"C\" __global__ void compressed_append_sm121(const float* __restrict__ latent, unsigned char* __restrict__ codes, unsigned char* __restrict__ scales, unsigned first_row, unsigned row_count, unsigned ratio)";
/// The selected-fill signature, whitespace-normalised for the declaration check.
pub const SELECT_DECLARATION: &str = "extern \"C\" __global__ void compressed_selected_fill_sm121(const unsigned* __restrict__ ids, unsigned n_ids, unsigned window_rows, unsigned ratio, unsigned* __restrict__ out, unsigned* __restrict__ out_count)";
/// Argument words of one append launch: latent, codes, scales, first_row, row_count, ratio.
pub const ARG_WORDS: usize = 6;
/// Argument words of one selected-fill launch: ids, n_ids, window_rows, ratio, out, out_count.
pub const SELECT_ARG_WORDS: usize = 6;
/// The launch block: 256 threads covers the 256 code bytes of one row.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);
/// Expanded store rows, the body's view bound.
pub const EXPANDED_ROWS: usize = 512;

/// The compiled artifact byte count: nvcc V13.0.88 (CUDA 13.0), sm_121a, the baseline flags.
pub const CUBIN_BYTES: usize = 23800;
/// The compiled artifact SHA-256, from the same build as [CUBIN_BYTES].
pub const CUBIN_SHA256: &str = "77026c93f196ca290203e37b62b56a4e83853d9c5ef1eba9a6b68b72f449f7da";

/// The append launch arguments, in the order the declaration takes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppendParams {
    /// Device pointer of the bf16-narrowed latent rows.
    pub latent: u64,
    /// Device pointer of the code bytes.
    pub codes: u64,
    /// Device pointer of the scale bytes.
    pub scales: u64,
    /// The first compressed row of this append.
    pub first_row: u32,
    /// How many completed rows were appended.
    pub row_count: u32,
    /// The layer compress ratio.
    pub ratio: u32,
}

impl AppendParams {
    /// The words a launcher passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; ARG_WORDS] {
        [
            self.latent,
            self.codes,
            self.scales,
            u64::from(self.first_row),
            u64::from(self.row_count),
            u64::from(self.ratio),
        ]
    }
}

/// The selected-fill launch arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectParams {
    /// Device pointer of the select ids (shared index space).
    pub ids: u64,
    /// How many ids.
    pub n_ids: u32,
    /// The window offset the body subtracts.
    pub window_rows: u32,
    /// The layer compress ratio.
    pub ratio: u32,
    /// Device pointer of the expanded id output.
    pub out: u64,
    /// Device pointer of the written count, or null.
    pub out_count: u64,
}

impl SelectParams {
    /// The words a launcher passes, in declaration order.
    #[must_use]
    pub fn words(&self) -> [u64; SELECT_ARG_WORDS] {
        [
            self.ids,
            u64::from(self.n_ids),
            u64::from(self.window_rows),
            u64::from(self.ratio),
            self.out,
            self.out_count,
        ]
    }
}

fn normalised(text: &str) -> String {
    let joined = text.split_whitespace().collect::<Vec<&str>>().join(" ");
    joined
        .replace("( ", "(")
        .replace(" )", ")")
        .replace(" ,", ",")
}

/// Whether the source still states the append signature this module declares.
#[must_use]
pub fn append_declaration_matches() -> bool {
    normalised(SOURCE).contains(&normalised(APPEND_DECLARATION))
}

/// Whether the source still states the selected-fill signature this module declares.
#[must_use]
pub fn select_declaration_matches() -> bool {
    normalised(SOURCE).contains(&normalised(SELECT_DECLARATION))
}

/// Which side owns the two selected-id refusals on the device-select path: the REFERENCE path, not
/// the device, and by construction rather than by a check. ref_select_sm121 writes ids in the shared
/// index space as position + offset with offset == window_rows and position < reach, so the caller
/// contract is offset == window_rows and reach <= plan.capacity_rows(); that makes id >= window_rows
/// and (id - window_rows) * ratio < 512 true by construction. The two refusals therefore stay in the
/// host reference path and its twin (compressed_cache_ref::selected_rows and the tests); no
/// per-position readback, no device check, no seq ABI change.
///
/// PRESENT vs RULED, because the shipped kernel is still the compacting one: the ruled in-place fill
/// (map every slot, keep NO_INDEX) is the next iteration's first item with its own re-pin; until then
/// this file's fill still skips NO_INDEX and its count is out_count.
///
/// The union-path numbers, quoted from the body where the launch is built: the SEQ entry's own guard
/// is window_rows + keep <= 640 = MAX_ATTENDED (mla_ref::MAX_SELECTED + SWA_WINDOW) and its candidate
/// guard is candidate <= position (spark_mla_seq_union_attends); a candidate past the query is scored
/// with the exact sentinel -3.402823466e38f, so exp(sentinel - s_max) underflows to +0.0. The ring
/// entry carries a separate static_assert RING_MAX_ATTENDED = 646 (512 + 134), quoted for cross-check
/// only - this launch is sized by MAX_ATTENDED = 640, not by 646.
pub const SELECT_VALIDATION: &str = "reference-path: ref_select writes position + offset with offset == window_rows and reach <= capacity_rows, so the two selected-id refusals are reference-path/twin-only; the ruled in-place fill that preserves NO_INDEX is next-iteration work, the shipped fill still skips it. The seq entry is sized by MAX_ATTENDED = 640; the ring entry's 646 static_assert is a cross-check, not this launch's bound";

/// The union path's candidate bound: one window (128) plus the full selected list (512). Quoted by
/// the launch build together with the body's own 646-slot ring bound and the sentinel below.
pub const MAX_ATTENDED: usize = crate::mla_ref::MAX_SELECTED + crate::mla_ref::SWA_WINDOW as usize;
/// The RING entry's candidate-array bound, its own static_assert in
/// flash_mla_csa2_ring_sm121.cu: 512 + 134. Quoted for cross-check only: the seq entry this fill
/// feeds is sized by MAX_ATTENDED = 640, not by this number.
pub const RING_MAX_ATTENDED: usize = crate::mla_ref::MAX_SELECTED + 134;
/// The exact score the body gives a candidate past the query: the body writes -3.402823466e38f,
/// which is f32::MIN. exp(sentinel - s_max) underflows to +0.0, so padding is bit-identical to
/// compaction.
pub const CANDIDATE_SENTINEL: f32 = f32::MIN;
/// The body's candidate guard, quoted: a candidate is attended only when candidate <= position.
pub const CANDIDATE_GUARD: &str = "candidate <= position";

/// The selected count: selected_count = keep = min(topk, n), a host scalar true by construction
/// ONCE THE RULED FILL LANDS. The shipped fill still compacts and skips NO_INDEX, so on today's tree
/// the count it produces is out_count and no caller may pass keep until the in-place map and its
/// re-pin land (next iteration, first item). After that change the seq entry takes selected_count by
/// value as keep and nothing is read back; out_count becomes a debug echo of n_ids.
pub const SELECTED_COUNT_OWNER: &str = "host: selected_count = keep = min(topk, n) once the ruled in-place fill lands (next iteration); the shipped fill still compacts and skips NO_INDEX, so today's count is out_count and no caller may pass keep yet";
