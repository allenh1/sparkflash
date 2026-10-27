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

//! Per-launch stage timing for the device arm's seam launches, on the GPU trace ring.
//!
//! The eight-symbol seam image's kernels carry the in-graph trace hooks of
//! `kernels/trace_globaltimer_sm121.cuh`: one 32-byte `GpuTraceEvent` per launch, written by block
//! 0 thread 0 when the ring's live `active_scope_mask` names the kernel's scope. This module is the
//! Rust half of that record for the device port: it builds the ring header the host writes once,
//! decodes the events after a run, and attributes them to a stage.
//!
//! The record's `payload_meta` word is what makes stage attribution possible. `scope_and_rank` only
//! names a domain, and four stages share `MEGA_MHC` (entry, FFN seams, attention residual, FFN
//! epilogue), so a scope alone cannot separate them. Each seam kernel therefore passes a stage code
//! as `payload_meta`; the codes are macros in `kernels/layer_seams_sm121.cu` and constants here,
//! and the test below pins the two halves against each other.
//!
//! No synchronize and no copy are involved. The ring lives in host-mapped coherent memory
//! (`Cuda::alloc_host_mapped`), so the kernels write the events and the host reads them; the launch
//! path records only an `%globaltimer` pair. The host reads the ring when it next has a boundary it
//! already needed (a stream-ordered copy, a synchronize, or the end of the run).

use sparkflash_trace::TraceScope;

/// Bytes of the ring header: `GpuTraceRing` is 32 bytes, and the event array starts at offset 32.
pub const TRACE_HEADER_BYTES: usize = 32;

/// Bytes of one trace event: `GpuTraceEvent` is `__align__(32)` and 32 bytes wide.
pub const TRACE_EVENT_BYTES: usize = 32;

/// Default ring capacity in events: 16,384 events, 512 KiB of host-mapped memory.
///
/// One device layer position at the released shape launches about seven instrumented kernels per
/// layer, so this covers the first ~58 positions of a 40-layer run before the oldest events are
/// overwritten. The header's `write_idx` keeps counting past the capacity; [`DecodedRing::wrapped`]
/// says whether the reader lost events.
pub const DEFAULT_CAPACITY: usize = 1 << 14;

/// Largest accepted capacity (32 MiB of events). A ring larger than this costs more than the
/// run's other host-mapped slabs.
pub const MAX_CAPACITY: usize = 1 << 20;

/// The scope bits the device arm arms in the ring: the four seam stages that share `MEGA_MHC` and
/// the quantisers that share `DENSE_MXFP8`.
pub const SEAM_SCOPE_MASK: u64 = TraceScope::MEGA_MHC | TraceScope::DENSE_MXFP8;

/// `mhc_entry_bf16`: the mHC entry, `hc_pre` and the attention block's RMSNorm.
pub const STAGE_MHC_ENTRY: u32 = 1;
/// `mhc_ffn_seams`: the FFN's mHC entry, its RMSNorm and its activation quantiser.
pub const STAGE_FFN_SEAMS: u32 = 2;
/// `w8a8_prologue_e4m3`: the attention projection's f32 activation quantiser.
pub const STAGE_PROLOGUE: u32 = 3;
/// `w8a8_prologue_bf16_e4m3`: the same quantiser over a bf16 row (q_norm, the FFN seeds).
pub const STAGE_PROLOGUE_BF16: u32 = 4;
/// `mhc_attn_residual`: the attention sublayer's `hc_post` over the reduced delta.
pub const STAGE_ATTN_RESIDUAL: u32 = 5;
/// `mhc_ffn_epilogue`: the shared-expert add and the FFN's `hc_post`.
pub const STAGE_FFN_EPILOGUE: u32 = 6;
/// `rope_rotate_rows_bf16`: the attention output's inverse RoPE (`sign_sign = -1`).
pub const STAGE_ROPE_ROWS: u32 = 7;

/// The stage codes in launch order, with the kernel symbol each one belongs to.
pub const STAGE_TABLE: [(u32, &str, &str); 7] = [
    (STAGE_MHC_ENTRY, "mhc_entry", "mhc_entry_bf16"),
    (STAGE_FFN_SEAMS, "ffn_seams", "mhc_ffn_seams"),
    (STAGE_PROLOGUE, "prologue", "w8a8_prologue_e4m3"),
    (
        STAGE_PROLOGUE_BF16,
        "prologue_bf16",
        "w8a8_prologue_bf16_e4m3",
    ),
    (STAGE_ATTN_RESIDUAL, "attn_residual", "mhc_attn_residual"),
    (STAGE_FFN_EPILOGUE, "ffn_epilogue", "mhc_ffn_epilogue"),
    (STAGE_ROPE_ROWS, "rope_rows", "rope_rotate_rows_bf16"),
];

/// The canonical stage name for a record's `payload_meta` word, when it is one of ours.
#[must_use]
pub fn stage_name(code: u32) -> Option<&'static str> {
    STAGE_TABLE
        .iter()
        .find(|(stage, _, _)| *stage == code)
        .map(|(_, name, _)| *name)
}

/// The kernel symbol a stage code belongs to, when it is one of ours.
#[must_use]
pub fn stage_kernel(code: u32) -> Option<&'static str> {
    STAGE_TABLE
        .iter()
        .find(|(stage, _, _)| *stage == code)
        .map(|(_, _, kernel)| *kernel)
}

/// Bytes a ring of `capacity` events needs: header plus events, 32-byte aligned.
///
/// # Panics
/// Panics when `capacity` is zero, is not a power of two, or exceeds [`MAX_CAPACITY`].
#[must_use]
pub fn ring_bytes(capacity: usize) -> usize {
    assert!(
        capacity.is_power_of_two(),
        "ring capacity must be a power of two"
    );
    assert!(
        capacity <= MAX_CAPACITY,
        "ring capacity must not exceed MAX_CAPACITY"
    );
    TRACE_HEADER_BYTES + TRACE_EVENT_BYTES * capacity
}

/// The 32-byte header the host writes once, before the first launch.
///
/// Layout, little-endian, matches `GpuTraceRing`: the live scope mask (8 B), the write index (4 B),
/// the capacity minus one (4 B), and 16 padding bytes that must stay zero.
///
/// # Panics
/// Panics for the same reasons as [`ring_bytes`].
#[must_use]
pub fn header_bytes(capacity: usize, scope_mask: u64) -> [u8; TRACE_HEADER_BYTES] {
    assert!(
        capacity.is_power_of_two(),
        "ring capacity must be a power of two"
    );
    assert!(
        capacity <= MAX_CAPACITY,
        "ring capacity must not exceed MAX_CAPACITY"
    );
    let mut header = [0u8; TRACE_HEADER_BYTES];
    header[0..8].copy_from_slice(&scope_mask.to_le_bytes());
    // write_idx starts at zero; the kernel advances it with atomicAdd.
    header[8..12].copy_from_slice(&0u32.to_le_bytes());
    header[12..16].copy_from_slice(&((capacity - 1) as u32).to_le_bytes());
    header
}

/// One decoded trace record, in the field order of `GpuTraceEvent`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Event {
    /// `%globaltimer` nanoseconds at kernel entry.
    pub start_ns: u64,
    /// `%globaltimer` nanoseconds at kernel exit.
    pub end_ns: u64,
    /// In-kernel wait time, when the kernel reports one (the seam hooks pass zero).
    pub sub_wait_ns: u32,
    /// `(scope bit index << 8) | rank`, where the bit index is 1-based, as `__ffsll` produces it.
    pub scope_and_rank: u32,
    /// The `layer_id` argument the launch carried.
    pub layer_id: u32,
    /// The stage code the kernel passed, for the seam kernels; other kernels use it differently.
    pub payload_meta: u32,
}

impl Event {
    /// Decodes the 1-based scope bit index back into a scope mask.
    #[must_use]
    pub fn scope_mask(&self) -> u64 {
        let index = (self.scope_and_rank >> 8) & 0xFF;
        if index == 0 || index > 64 {
            0
        } else {
            1u64 << (index - 1)
        }
    }

    /// The canonical scope name for this record, when the bit is one of the sixteen defined ones.
    #[must_use]
    pub fn scope_name(&self) -> Option<&'static str> {
        TraceScope::name(self.scope_mask())
    }

    /// The rank the record belongs to (grid dimension y of the launch).
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.scope_and_rank & 0xFF
    }

    /// The launch's duration in nanoseconds.
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

/// A decoded ring: capacity, total events written, and the events the reader can still see.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodedRing {
    /// Capacity in events, from the header.
    pub capacity: usize,
    /// Total events ever written, including ones the ring has overwritten.
    pub written: u32,
    /// Whether `written` exceeded `capacity`, i.e. the oldest events were overwritten.
    pub wrapped: bool,
    /// The readable events, oldest first.
    pub events: Vec<Event>,
}

/// Decodes a host-mapped ring buffer (header followed by `capacity` events).
///
/// The buffer must be exactly [`ring_bytes(capacity)`] long and its header's capacity mask must be a
/// power-of-two-minus-one. Only the events a reader can still see are returned: the last `min(written,
/// capacity)` slots in ring order.
///
/// # Errors
/// Returns an error when the buffer is too small, the header's capacity is not a power of two, or
/// the write index is inconsistent with the buffer length.
pub fn decode(bytes: &[u8]) -> Result<DecodedRing, String> {
    if bytes.len() < TRACE_HEADER_BYTES {
        return Err(format!(
            "trace ring is {} B, the header alone is {TRACE_HEADER_BYTES} B",
            bytes.len()
        ));
    }
    let mask = u32::from_le_bytes(bytes[12..16].try_into().expect("four bytes"));
    if mask == 0 || (mask & (mask + 1)) != 0 {
        return Err(format!(
            "trace ring capacity mask {mask:#x} is not a power of two minus one"
        ));
    }
    let capacity = mask as usize + 1;
    let expected = ring_bytes(capacity);
    if bytes.len() != expected {
        return Err(format!(
            "trace ring is {} B, a {capacity}-event ring is {expected} B",
            bytes.len()
        ));
    }
    let written = u32::from_le_bytes(bytes[8..12].try_into().expect("four bytes"));
    let take = (written as usize).min(capacity);
    let first = written as usize - take;
    let mut events = Vec::with_capacity(take);
    for slot in 0..take {
        let index = (first + slot) & mask as usize;
        let start = TRACE_HEADER_BYTES + index * TRACE_EVENT_BYTES;
        let event = &bytes[start..start + TRACE_EVENT_BYTES];
        events.push(Event {
            start_ns: u64::from_le_bytes(event[0..8].try_into().expect("eight bytes")),
            end_ns: u64::from_le_bytes(event[8..16].try_into().expect("eight bytes")),
            sub_wait_ns: u32::from_le_bytes(event[16..20].try_into().expect("four bytes")),
            scope_and_rank: u32::from_le_bytes(event[20..24].try_into().expect("four bytes")),
            layer_id: u32::from_le_bytes(event[24..28].try_into().expect("four bytes")),
            payload_meta: u32::from_le_bytes(event[28..32].try_into().expect("four bytes")),
        });
    }
    Ok(DecodedRing {
        capacity,
        written,
        wrapped: written as usize > capacity,
        events,
    })
}

/// One stage's aggregate over the decoded events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageRow {
    /// The canonical stage name.
    pub stage: &'static str,
    /// Launches attributed to the stage.
    pub calls: u64,
    /// Summed duration, nanoseconds.
    pub total_ns: u64,
    /// Shortest launch, nanoseconds.
    pub min_ns: u64,
    /// Longest launch, nanoseconds.
    pub max_ns: u64,
}

impl StageRow {
    /// Mean duration, nanoseconds (saturating integer division).
    #[must_use]
    pub fn mean_ns(&self) -> u64 {
        self.total_ns.checked_div(self.calls).unwrap_or(0)
    }
}

/// Aggregates decoded events by stage code, in [`STAGE_TABLE`] order.
///
/// Returns the rows and the number of events whose `payload_meta` is not a seam stage code (another
/// kernel family's own meta, or a plain kernel that reports no meta).
#[must_use]
pub fn stages(events: &[Event]) -> (Vec<StageRow>, u64) {
    let mut rows = vec![StageRow::default(); STAGE_TABLE.len()];
    for (row, (_, name, _)) in rows.iter_mut().zip(STAGE_TABLE) {
        row.stage = name;
    }
    let mut unattributed = 0u64;
    for event in events {
        let Some(index) = STAGE_TABLE
            .iter()
            .position(|(code, _, _)| *code == event.payload_meta)
        else {
            unattributed += 1;
            continue;
        };
        let stage = &mut rows[index];
        stage.stage = STAGE_TABLE[index].1;
        let ns = event.duration_ns();
        if stage.calls == 0 {
            stage.min_ns = ns;
            stage.max_ns = ns;
        }
        stage.calls += 1;
        stage.total_ns = stage.total_ns.saturating_add(ns);
        stage.min_ns = stage.min_ns.min(ns);
        stage.max_ns = stage.max_ns.max(ns);
    }
    (rows, unattributed)
}

/// The header/event layout against the trace crate's mirror, so the two cannot drift.
const _: () = {
    assert!(TRACE_EVENT_BYTES == 32);
    assert!(TRACE_HEADER_BYTES == 32);
};

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = include_str!("../../../kernels/layer_seams_sm121.cu");

    fn event_bytes(start: u64, end: u64, scope_bit: u32, layer: u32, meta: u32) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&start.to_le_bytes());
        bytes[8..16].copy_from_slice(&end.to_le_bytes());
        bytes[20..24].copy_from_slice(&(scope_bit << 8).to_le_bytes());
        bytes[24..28].copy_from_slice(&layer.to_le_bytes());
        bytes[28..32].copy_from_slice(&meta.to_le_bytes());
        bytes
    }

    #[test]
    fn the_trace_crate_mirror_has_the_event_and_header_layout_this_module_decodes() {
        assert_eq!(
            std::mem::size_of::<sparkflash_trace::GpuTraceEvent>(),
            TRACE_EVENT_BYTES
        );
        assert_eq!(
            std::mem::align_of::<sparkflash_trace::GpuTraceEvent>(),
            TRACE_EVENT_BYTES
        );
        assert_eq!(
            sparkflash_trace::GpuTraceRingHeader::layout_bytes(4),
            ring_bytes(4)
        );
        let header = header_bytes(4, SEAM_SCOPE_MASK);
        assert_eq!(
            u64::from_le_bytes(header[0..8].try_into().unwrap()),
            SEAM_SCOPE_MASK
        );
        assert_eq!(u32::from_le_bytes(header[8..12].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(header[12..16].try_into().unwrap()), 3);
        assert_eq!(header[16..], [0u8; 16]);
    }

    #[test]
    fn the_seam_scope_mask_is_mega_mhc_and_dense_mxfp8() {
        assert_eq!(SEAM_SCOPE_MASK, (1 << 7) | (1 << 10));
        assert_eq!(TraceScope::name(TraceScope::MEGA_MHC), Some("MEGA_MHC"));
        assert_eq!(
            TraceScope::name(TraceScope::DENSE_MXFP8),
            Some("DENSE_MXFP8")
        );
    }

    #[test]
    fn a_ring_that_has_not_wrapped_decodes_every_event_in_slot_order() {
        let capacity = 4usize;
        let mut bytes = vec![0u8; ring_bytes(capacity)];
        bytes[..TRACE_HEADER_BYTES].copy_from_slice(&header_bytes(capacity, SEAM_SCOPE_MASK));
        let codes = [
            STAGE_MHC_ENTRY,
            STAGE_FFN_SEAMS,
            STAGE_PROLOGUE,
            STAGE_PROLOGUE_BF16,
        ];
        for (slot, code) in codes.iter().enumerate() {
            let start = TRACE_HEADER_BYTES + slot * TRACE_EVENT_BYTES;
            bytes[start..start + TRACE_EVENT_BYTES]
                .copy_from_slice(&event_bytes(10, 20, 8, 3, *code));
        }
        bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(decoded.capacity, capacity);
        assert_eq!(decoded.written, 3);
        assert!(!decoded.wrapped);
        assert_eq!(decoded.events.len(), 3);
        assert_eq!(decoded.events[0].payload_meta, STAGE_MHC_ENTRY);
        assert_eq!(decoded.events[2].payload_meta, STAGE_PROLOGUE);
        assert_eq!(decoded.events[0].duration_ns(), 10);
        assert_eq!(decoded.events[0].scope_mask(), 1 << 7);
        assert_eq!(decoded.events[0].scope_name(), Some("MEGA_MHC"));
    }

    #[test]
    fn a_wrapped_ring_keeps_the_newest_events_and_says_so() {
        let capacity = 4usize;
        let mut bytes = vec![0u8; ring_bytes(capacity)];
        bytes[..TRACE_HEADER_BYTES].copy_from_slice(&header_bytes(capacity, SEAM_SCOPE_MASK));
        // Six events written over a four-slot ring: slots 0 and 1 hold events 4 and 5, slots 2 and
        // 3 hold events 2 and 3. A reader sees 2, 3, 4, 5 in ring order.
        for (slot, meta) in [(0u32, 4u32), (1, 5), (2, 2), (3, 3)] {
            let start = TRACE_HEADER_BYTES + slot as usize * TRACE_EVENT_BYTES;
            bytes[start..start + TRACE_EVENT_BYTES].copy_from_slice(&event_bytes(1, 2, 8, 0, meta));
        }
        bytes[8..12].copy_from_slice(&6u32.to_le_bytes());
        let decoded = decode(&bytes).expect("decode");
        assert!(decoded.wrapped);
        assert_eq!(decoded.written, 6);
        let metas: Vec<u32> = decoded.events.iter().map(|e| e.payload_meta).collect();
        assert_eq!(metas, [STAGE_FFN_SEAMS, STAGE_PROLOGUE, 4, 5]);
    }

    #[test]
    fn the_stage_rows_aggregate_calls_and_durations_in_table_order() {
        let events = [
            Event {
                start_ns: 100,
                end_ns: 110,
                payload_meta: STAGE_MHC_ENTRY,
                ..Event::default()
            },
            Event {
                start_ns: 200,
                end_ns: 230,
                payload_meta: STAGE_MHC_ENTRY,
                ..Event::default()
            },
            Event {
                start_ns: 300,
                end_ns: 305,
                payload_meta: STAGE_FFN_SEAMS,
                ..Event::default()
            },
            Event {
                start_ns: 400,
                end_ns: 404,
                payload_meta: 99,
                ..Event::default()
            },
        ];
        let (rows, unattributed) = stages(&events);
        assert_eq!(rows.len(), STAGE_TABLE.len());
        assert_eq!(unattributed, 1);
        let entry = rows
            .iter()
            .find(|row| row.stage == "mhc_entry")
            .expect("entry row");
        assert_eq!(entry.calls, 2);
        assert_eq!(entry.total_ns, 40);
        assert_eq!(entry.min_ns, 10);
        assert_eq!(entry.max_ns, 30);
        assert_eq!(entry.mean_ns(), 20);
        let ffn = rows
            .iter()
            .find(|row| row.stage == "ffn_seams")
            .expect("ffn row");
        assert_eq!(ffn.calls, 1);
        // A stage with no event stays a zero row rather than disappearing, so the reader's table
        // does not depend on which stages ran.
        let residual = rows
            .iter()
            .find(|row| row.stage == "attn_residual")
            .expect("residual row");
        assert_eq!(residual.calls, 0);
    }

    #[test]
    fn every_seam_kernel_passes_its_own_stage_code_as_the_trace_payload() {
        // The kernel side must agree with STAGE_TABLE: one macro per stage, each EXIT call passing
        // its macro, and each macro equal to the constant this module decodes.
        for (code, name, _kernel) in STAGE_TABLE {
            let macro_name = format!("SPARK_SEAM_STAGE_{}", name.to_uppercase());
            let define = format!("#define {macro_name} {code}U");
            assert!(
                SOURCE.contains(&define),
                "kernels/layer_seams_sm121.cu is missing {define}"
            );
            assert!(
                SOURCE.contains("SPARK_GPU_TRACE_EXIT(trace_ring, "),
                "no trace exit in the kernel source"
            );
        }
        // Every EXIT call site carries a stage macro, not the old literal 0U.
        assert!(!SOURCE.contains("TRACE_EXIT(trace_ring, TRACE_SCOPE_MEGA_MHC, layer_id, 0U,"));
        assert!(!SOURCE.contains("TRACE_EXIT(trace_ring, TRACE_SCOPE_DENSE_MXFP8, layer_id, 0U,"));
        for (_, name, _) in STAGE_TABLE {
            let macro_name = format!("SPARK_SEAM_STAGE_{}", name.to_uppercase());
            // Count with the delimiter: PROLOGUE is a prefix of PROLOGUE_BF16, so a bare
            // substring count would double-count the longer macro.
            assert_eq!(
                SOURCE.matches(&format!("{macro_name} ")).count(),
                1,
                "each stage macro is defined once: {macro_name}"
            );
            assert_eq!(
                SOURCE.matches(&format!("{macro_name},")).count(),
                1,
                "each stage macro is passed by exactly one EXIT call: {macro_name}"
            );
        }
    }
}
