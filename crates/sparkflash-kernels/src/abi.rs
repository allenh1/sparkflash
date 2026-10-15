//! Conformance between the Rust trace ABI and the CUDA header every instrumented kernel
//! includes.
//!
//! The two sides are edited by different people in different languages, so the header is
//! embedded here and checked field by field. The checks are ordinary unit tests, which means
//! they run on a machine without nvcc; the build script additionally compiles a static-assert
//! probe of the same header whenever a toolchain is present.
//!
//! The field checks parse the struct DECLARATION rather than searching the whole file for field
//! names: the trace macros below the structs mention every field too, so a whole-file search
//! would happily accept a reordered struct.

/// The CUDA header, embedded at compile time so the checks cannot pass against a stale copy.
pub const TRACE_ABI_HEADER: &str = include_str!("../../../kernels/trace_globaltimer_sm121.cuh");

#[cfg(test)]
mod tests {
    use sparkflash_trace::{
        GPU_RING_HEADER_BYTES, GPU_RING_HEADER_USED_BYTES, GPU_TRACE_EVENT_ALIGN,
        GPU_TRACE_EVENT_BYTES, GpuTraceEvent, GpuTraceRingHeader, TraceScope,
    };

    fn source() -> &'static str {
        super::TRACE_ABI_HEADER
    }

    /// Strips line comments so a comment containing a semicolon cannot be mistaken for a field.
    fn strip_comments(text: &str) -> String {
        text.lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Field declarations of a struct, in declaration order, as (type, name).
    fn fields(declaration: &str) -> Vec<(String, String)> {
        let start = source()
            .find(declaration)
            .unwrap_or_else(|| panic!("the CUDA header no longer declares {declaration}"));
        let tail = &source()[start..];
        let end = tail
            .find("};")
            .unwrap_or_else(|| panic!("{declaration} has no closing brace"));
        let body = &tail[..end];
        let open = body
            .find('{')
            .unwrap_or_else(|| panic!("{declaration} has no opening brace"));
        strip_comments(&body[open + 1..])
            .split(';')
            .map(|statement| statement.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|statement| !statement.is_empty())
            .map(|statement| {
                statement
                    .rsplit_once(' ')
                    .map(|(ty, name)| (ty.to_owned(), name.to_owned()))
                    .unwrap_or_else(|| panic!("cannot parse the field declaration {statement:?}"))
            })
            .collect()
    }

    fn expected(fields: &[(&str, &str)]) -> Vec<(String, String)> {
        fields
            .iter()
            .map(|(ty, name)| ((*ty).to_owned(), (*name).to_owned()))
            .collect()
    }

    #[test]
    fn every_scope_bit_matches_the_cuda_define() {
        // The C header aligns its macro values with spaces, so compare against a
        // whitespace-collapsed copy of the source.
        let collapsed = source().split_whitespace().collect::<Vec<_>>().join(" ");
        for (name, bit) in TraceScope::TABLE {
            let define = format!(
                "#define TRACE_SCOPE_{name} (1ULL << {})",
                bit.trailing_zeros()
            );
            assert!(
                collapsed.contains(&define),
                "CUDA header is missing: {define}"
            );
        }
        assert_eq!(TraceScope::TABLE.len(), TraceScope::COUNT);
    }

    #[test]
    fn cuda_event_struct_matches_the_rust_type() {
        assert_eq!(
            fields("struct __align__(32) GpuTraceEvent"),
            expected(&[
                ("uint64_t", "start_ns"),
                ("uint64_t", "end_ns"),
                ("uint32_t", "sub_wait_ns"),
                ("uint32_t", "scope_and_rank"),
                ("uint32_t", "layer_id"),
                ("uint32_t", "payload_meta"),
            ]),
            "the CUDA event struct must declare exactly these fields in this order"
        );
        assert_eq!(std::mem::size_of::<GpuTraceEvent>(), GPU_TRACE_EVENT_BYTES);
        assert_eq!(std::mem::align_of::<GpuTraceEvent>(), GPU_TRACE_EVENT_ALIGN);
        assert_eq!(std::mem::size_of::<GpuTraceEvent>(), 6 * 4 + 8);
        // The Rust offsets must match the C ones the build-script probe static-asserts.
        assert_eq!(std::mem::offset_of!(GpuTraceEvent, start_ns), 0);
        assert_eq!(std::mem::offset_of!(GpuTraceEvent, end_ns), 8);
        assert_eq!(std::mem::offset_of!(GpuTraceEvent, sub_wait_ns), 16);
        assert_eq!(std::mem::offset_of!(GpuTraceEvent, scope_and_rank), 20);
        assert_eq!(std::mem::offset_of!(GpuTraceEvent, layer_id), 24);
        assert_eq!(std::mem::offset_of!(GpuTraceEvent, payload_meta), 28);
    }

    #[test]
    fn cuda_ring_header_matches_the_rust_type() {
        assert_eq!(
            fields("struct __align__(32) GpuTraceRing"),
            expected(&[
                ("volatile uint64_t", "active_scope_mask"),
                ("uint32_t", "write_idx"),
                ("uint32_t", "capacity_mask"),
                ("uint64_t", "reserved[2]"),
            ])
        );
        assert_eq!(
            std::mem::size_of::<GpuTraceRingHeader>(),
            GPU_RING_HEADER_BYTES
        );
        assert_eq!(
            std::mem::align_of::<GpuTraceRingHeader>(),
            GPU_TRACE_EVENT_ALIGN
        );
        assert_eq!(
            std::mem::offset_of!(GpuTraceRingHeader, active_scope_mask),
            0
        );
        assert_eq!(std::mem::offset_of!(GpuTraceRingHeader, write_idx), 8);
        assert_eq!(std::mem::offset_of!(GpuTraceRingHeader, capacity_mask), 12);
        assert_eq!(std::mem::offset_of!(GpuTraceRingHeader, reserved), 16);
        // The padding is what keeps the event array 32-byte aligned inside the ring.
        assert_eq!(GPU_RING_HEADER_BYTES, 32);
        assert_eq!(GPU_RING_HEADER_USED_BYTES, 16);
        assert_eq!(GPU_RING_HEADER_BYTES % GPU_TRACE_EVENT_ALIGN, 0);
    }

    /// The parser above is only worth having if it actually catches a reordering: prove it on a
    /// mutated copy of the real header instead of trusting it.
    #[test]
    fn the_struct_parser_detects_a_reordering_and_a_width_change() {
        fn parse(text: &str) -> Vec<String> {
            let tail = &text[text.find("struct __align__(32) GpuTraceEvent").unwrap()..];
            let body = &tail[..tail.find("};").unwrap()];
            let body = &body[body.find('{').unwrap() + 1..];
            strip_comments(body)
                .split(';')
                .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|s| !s.is_empty())
                .collect()
        }

        let reordered = source().replacen(
            "uint32_t layer_id;",
            "uint32_t payload_meta;\n    uint32_t layer_id;",
            1,
        );
        assert_ne!(
            reordered,
            source(),
            "the mutation must change the header text"
        );
        assert_eq!(
            source().find("struct __align__(32) GpuTraceEvent"),
            reordered.find("struct __align__(32) GpuTraceEvent"),
            "the mutation must stay inside the struct body"
        );
        assert_eq!(parse(source()).len(), 6);
        assert_eq!(parse(&reordered).len(), 7);
        assert_ne!(parse(source()), parse(&reordered));

        // A field width change is visible too, because the parser keeps the type.
        let widened = source().replacen("uint32_t payload_meta;", "uint64_t payload_meta;", 1);
        assert_ne!(parse(source()), parse(&widened));
    }

    #[test]
    fn cuda_macro_indexes_events_at_the_header_size() {
        // (GpuTraceEvent *)((char *)(ring) + sizeof(GpuTraceRing)) + idx
        assert!(
            source().contains("(GpuTraceEvent *)((char *)(ring) + sizeof(GpuTraceRing))"),
            "the CUDA macro must keep placing events directly after the header"
        );
        assert!(source().contains("atomicAdd((unsigned int *)&(ring)->write_idx, 1U)"));
        assert!(source().contains("& (ring)->capacity_mask"));
        // The 32-byte header is the reason the +sizeof(GpuTraceRing) offset stays aligned.
        assert_eq!(GPU_RING_HEADER_BYTES % GPU_TRACE_EVENT_ALIGN, 0);
    }

    #[test]
    fn cuda_rank_comes_from_block_idx_y() {
        assert!(source().contains("(uint32_t)(blockIdx.y)"));
    }

    #[test]
    fn trace_macros_guard_on_the_live_scope_mask_not_a_stale_timer() {
        // A kernel entered while armed must not write a record after a disarm, so the EXIT
        // macros test the live mask rather than the captured entry time.
        assert!(!source().contains("(t_enter) != 0"));
        assert!(source().contains("((ring)->active_scope_mask & (scope))"));
    }
}
