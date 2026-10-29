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

//! Conformance between the Rust mirror of the RoCEnante one-shot slot ABI and the CUDA
//! header and kernel that implement it.
//!
//! **Attribution:** The `RoCEnante` name and one-shot dual-rail RoCEv2 collective architecture
//! (pinned unified-memory slot staging with a system fence, CPU proxy doorbell striping across
//! dual ConnectX-7 rails, per-rail sequence arrival flags, canonical rank-order `0 + 1 + 2 + 3`
//! reduction, and the `<= 2 MiB` one-shot RDMA vs `> 2 MiB` dual-rail NCCL dispatch policy)
//! originate from [`b12x`](https://github.com/local-inference-lab/b12x) (Apache-2.0; Luke Alonso
//! `@lukealonso`, `@original-el8`, and contributors),
//! [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0), and
//! [`dgx-spark-networking`](https://github.com/christopherowen/dgx-spark-networking) (Apache-2.0;
//! Christopher Owen). See the root `NOTICE` file for full provenance details.
//!
//! The CUDA side is `kernels/rocenante_oneshot_sm121.cuh` (the slot) and
//! `kernels/rocenante_oneshot_sm121.cu` (the kernel). A GPU writes the slot fields
//! and a ConnectX-7 NIC DMAs into them, so the byte offsets are a wire protocol, not an
//! implementation detail. The two sides are written in different languages, so this module
//! does not trust a comment: it embeds both CUDA sources at compile time, PARSES the
//! declarations out of them, and compares the result field by field with the Rust types
//! below, in declaration order. The build script independently compiles the same sources with
//! a CUDA frontend and static-asserts the same numbers.
//!
//! Three independent statements must agree:
//!
//! 1. the CUDA declaration text (what the GPU and the NIC actually see);
//! 2. the Rust mirror (what the host code passes);
//! 3. the arithmetic re-derived from the parsed array dimensions and C type widths.
//!
//! Nothing in this module needs a GPU, a CUDA toolkit or weights.

use half::bf16;
use sparkflash_trace::GpuTraceEvent;

/// The CUDA slot ABI header, embedded at compile time so the checks cannot pass against a
/// stale copy.
pub const ROCE_ONESHOT_HEADER: &str = include_str!("../../../kernels/rocenante_oneshot_sm121.cuh");

/// The CUDA kernel, embedded at compile time so the checks cannot pass against a stale copy.
pub const ROCE_ONESHOT_KERNEL: &str = include_str!("../../../kernels/rocenante_oneshot_sm121.cu");

/// The build script, embedded so the tests can check that its independent probe still
/// re-declares the kernel signature and repeats the slot layout literals.
pub const ROCE_BUILD_SCRIPT: &str = include_str!("../build.rs");

/// The stub CUDA headers the clang probe compiles against, embedded so the tests can check that
/// the device functions the trace macros need are still declared there. Without them the clang
/// probe would fall back to compiling the kernel with the trace macros disabled.
pub const ROCE_CUDA_STUB: &str = include_str!("../cuda_stub/cuda_stub.h");

/// Ranks in the TP=4 one-shot group.
pub const ROCE_ONESHOT_RANKS: usize = 4;

/// bf16 elements in one rank's payload row (`1024 * 1024`, the 2 MiB RoCEnante payload).
pub const ROCE_ONESHOT_SLOT_ROW_ELEMS: usize = 1024 * 1024;

/// Bytes in one rank's payload row: 2 MiB.
pub const ROCE_ONESHOT_SLOT_ROW_BYTES: usize = ROCE_ONESHOT_SLOT_ROW_ELEMS * 2;

/// Bytes of the arrival sequence array at offset 0: 4 x u64.
pub const ROCE_ONESHOT_SLOT_ARRIVAL_BYTES: usize = ROCE_ONESHOT_RANKS * 8;

/// Bytes of the four payload rows.
pub const ROCE_ONESHOT_SLOT_PAYLOAD_BYTES: usize = ROCE_ONESHOT_RANKS * ROCE_ONESHOT_SLOT_ROW_BYTES;

/// Total bytes of one slot: 8 MiB + 32 bytes, matching `sizeof(RoceOneShotSlot)` in the
/// CUDA header and the build probe's literal static_assert.
pub const ROCE_ONESHOT_SLOT_BYTES: usize =
    ROCE_ONESHOT_SLOT_ARRIVAL_BYTES + ROCE_ONESHOT_SLOT_PAYLOAD_BYTES;

/// Alignment the specification's declaration gives the slot. The spec adds no `__align__`
/// attribute, so this is the natural alignment of its members (u64), not 32.
pub const ROCE_ONESHOT_SLOT_ALIGN: usize = 8;

/// Default bound on the in-kernel peer wait, in nanoseconds. Mirrors
/// `SPARK_ROCENANTE_PEER_WAIT_NS` in the CUDA header, which the tests parse back out.
pub const ROCE_PEER_WAIT_TIMEOUT_NS: u64 = 1_000_000;

/// Polls between `__nanosleep` calls in the peer wait. Mirrors
/// `SPARK_ROCENANTE_SPIN_ITERS`.
pub const ROCE_SPIN_ITERS: u32 = 64;

/// Length of each `__nanosleep` in the peer wait, in nanoseconds. Mirrors
/// `SPARK_ROCENANTE_SPIN_SLEEP_NS`.
pub const ROCE_SPIN_SLEEP_NS: u32 = 200;

/// The slot of specification 5.4, mirrored:
///
/// ```c
/// struct RoceOneShotSlot {
///     volatile uint64_t arrival_seq[4];
///     __nv_bfloat16 payload[4][1024 * 1024];
/// };
/// ```
///
/// `arrival_seq[rank]` is published by that rank and `payload[rank]` holds its
/// staged contribution. The CUDA field is `volatile` because a NIC writes it while the
/// kernel polls; `volatile` has no Rust equivalent and no layout effect, so the mirror
/// is a plain array.
#[repr(C)]
pub struct RoceOneShotSlot {
    /// Per-rank step sequence numbers. In CUDA: `volatile uint64_t arrival_seq[4]`.
    pub arrival_seq: [u64; ROCE_ONESHOT_RANKS],
    /// Per-rank staged payload, one 2 MiB row each. In CUDA:
    /// `__nv_bfloat16 payload[4][1024 * 1024]`.
    pub payload: [[bf16; ROCE_ONESHOT_SLOT_ROW_ELEMS]; ROCE_ONESHOT_RANKS],
}

/// Decoded `payload_meta` of a `TRACE_SCOPE_ROCENANTE_RDMA` trace record.
///
/// The kernel writes the transferred byte count there (at most 2 MiB), so bit 31 is free to
/// mark a diagnostic record. When it is set, the low bits carry the missing-rank mask and the
/// `SPARK_ROCE_DIAG_*` codes, and `sub_wait_ns` carries the wait that timed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoceTracePayload {
    /// A completed one-shot transfer of `bytes` bytes.
    Transfer {
        /// Transferred bytes, always at most [ROCE_ONESHOT_SLOT_ROW_BYTES].
        bytes: u32,
    },
    /// An incomplete collective. The kernel filled the output with bf16 quiet NaN.
    Diagnostic {
        /// Bit `r` is set when rank `r` never published
        /// `arrival_seq[r] >= step_seq` before the deadline.
        missing_ranks: u8,
        /// The request exceeded the 2 MiB slot row and was clamped.
        row_overflow: bool,
    },
}

impl RoceTracePayload {
    /// Bit 31 of `payload_meta`: set on diagnostic records, clear on transfers.
    pub const DIAG_FLAG: u32 = 1 << 31;
    /// Shift of the missing-rank mask in a diagnostic record.
    pub const MISSING_SHIFT: u32 = 8;
    /// Width of the missing-rank mask.
    pub const MISSING_MASK: u8 = 0x0f;
    /// Diagnostic code: the request exceeded the slot row.
    pub const ROW_OVERFLOW: u32 = 0x01;

    /// Decodes the `payload_meta` word written by the kernel.
    #[must_use]
    pub fn decode(payload_meta: u32) -> Self {
        if payload_meta & Self::DIAG_FLAG == 0 {
            Self::Transfer {
                bytes: payload_meta,
            }
        } else {
            Self::Diagnostic {
                missing_ranks: ((payload_meta >> Self::MISSING_SHIFT)
                    & u32::from(Self::MISSING_MASK)) as u8,
                row_overflow: payload_meta & Self::ROW_OVERFLOW != 0,
            }
        }
    }

    /// Encodes a value with the same bit layout the kernel uses.
    #[must_use]
    pub fn encode(self) -> u32 {
        match self {
            Self::Transfer { bytes } => bytes,
            Self::Diagnostic {
                missing_ranks,
                row_overflow,
            } => {
                let mut value = Self::DIAG_FLAG | (u32::from(missing_ranks) << Self::MISSING_SHIFT);
                if row_overflow {
                    value |= Self::ROW_OVERFLOW;
                }
                value
            }
        }
    }

    /// Classifies a trace event, or returns `None` when it belongs to another scope.
    #[must_use]
    pub fn from_event(event: &GpuTraceEvent) -> Option<Self> {
        (event.scope_mask() == sparkflash_trace::TraceScope::ROCENANTE_RDMA)
            .then(|| Self::decode(event.payload_meta))
    }
}

/// Declares the kernel's parameter list once, in a form the compiler checks.
///
/// The macro produces the CUDA spelling table that the tests compare against the parsed kernel
/// source, the byte width of every parameter's Rust type, and a Rust function with the very
/// same parameter names and types. A change to a Rust type that is not mirrored in the CUDA
/// spelling therefore fails either the compiler (the mirror function) or the width test.
macro_rules! kernel_signature {
    ($( ($cuda:literal, $name:ident : $rust:ty) ),* $(,)?) => {
        /// CUDA spelling of every `__global__` parameter, in declaration order.
        pub const ROCE_ONESHOT_KERNEL_PARAMS: &[(&str, &str)] =
            &[ $( ($cuda, stringify!($name)) ),* ];

        /// `size_of` of each parameter's Rust mirror type, in the same order.
        pub const ROCE_ONESHOT_KERNEL_PARAM_BYTES: &[usize] =
            &[ $( ::core::mem::size_of::<$rust>() ),* ];

        /// Compiler-checked mirror of the kernel signature: it is never called, it exists so
        /// that a parameter whose Rust type drifts from the CUDA declaration fails to compile.
        /// Its argument count mirrors the kernel's, which is the whole point of the check.
        #[allow(dead_code, clippy::too_many_arguments)]
        fn roce_oneshot_kernel_signature_mirror($( $name: $rust ),*) {
            $( let _: $rust = $name; )*
        }
    };
}

kernel_signature! {
    ("const __nv_bfloat16* __restrict__", local_in: *const bf16),
    ("__nv_bfloat16* __restrict__", local_out: *mut bf16),
    ("RoceOneShotSlot* __restrict__", slot: *mut RoceOneShotSlot),
    ("volatile uint64_t* __restrict__", host_doorbell_seq: *mut u64),
    ("GpuTraceRing* __restrict__", trace_ring: *mut sparkflash_trace::GpuTraceRingHeader),
    ("const uint32_t", layer_id: u32),
    ("const uint32_t", my_rank: u32),
    ("const uint32_t", num_elements: u32),
    ("const uint64_t", step_seq: u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strips line comments so a comment mentioning a field or a statement cannot satisfy a
    /// check. Every structural test below runs on stripped text for exactly that reason: the
    /// kernel's header comment describes the whole protocol in prose.
    fn strip_comments(text: &str) -> String {
        text.lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Comment-free, whitespace-collapsed source, for substring checks.
    fn normalized(text: &str) -> String {
        strip_comments(text)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The logical body of a `#define`, following backslash continuations.
    fn cuda_define(text: &str, name: &str) -> String {
        let prefix = format!("#define {name} ");
        let start = text
            .find(&prefix)
            .unwrap_or_else(|| panic!("the CUDA header no longer defines {name}"));
        let tail = &text[start + prefix.len()..];
        let mut logical = String::new();
        for line in tail.lines() {
            if let Some(without_continuation) = line.strip_suffix('\\') {
                logical.push_str(without_continuation);
                logical.push(' ');
            } else {
                logical.push_str(line);
                break;
            }
        }
        logical.trim().to_owned()
    }

    /// Evaluates the integer constant expressions the CUDA header uses in its defines and
    /// array dimensions: decimal and hex literals with u/U/l/L suffixes, `+`,
    /// `*` and parentheses. Panics on anything else, so an identifier cannot slip
    /// through unevaluated.
    fn eval_int_expr(expr: &str) -> usize {
        fn tokenize(expr: &str) -> Vec<String> {
            let mut tokens = Vec::new();
            let mut current = String::new();
            for ch in expr.chars() {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    current.push(ch);
                } else {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                    if !ch.is_whitespace() {
                        tokens.push(ch.to_string());
                    }
                }
            }
            if !current.is_empty() {
                tokens.push(current);
            }
            tokens
        }

        fn parse_literal(token: &str) -> Option<usize> {
            let digits = token.trim_end_matches(['u', 'U', 'l', 'L']);
            if digits.is_empty() {
                return None;
            }
            if let Some(hex) = digits
                .strip_prefix("0x")
                .or_else(|| digits.strip_prefix("0X"))
            {
                usize::from_str_radix(hex, 16).ok()
            } else {
                digits.parse::<usize>().ok()
            }
        }

        struct Parser {
            expr: String,
            tokens: Vec<String>,
            position: usize,
        }

        impl Parser {
            fn peek(&self) -> Option<&str> {
                self.tokens.get(self.position).map(String::as_str)
            }

            fn sum(&mut self) -> usize {
                let mut value = self.product();
                while self.peek() == Some("+") {
                    self.position += 1;
                    value += self.product();
                }
                value
            }

            fn product(&mut self) -> usize {
                let mut value = self.atom();
                while self.peek() == Some("*") {
                    self.position += 1;
                    value *= self.atom();
                }
                value
            }

            fn atom(&mut self) -> usize {
                match self.peek() {
                    Some("(") => {
                        self.position += 1;
                        let value = self.sum();
                        assert_eq!(
                            self.peek(),
                            Some(")"),
                            "unbalanced parentheses in {:?}",
                            self.expr
                        );
                        self.position += 1;
                        value
                    }
                    Some(token) => {
                        let token = token.to_owned();
                        self.position += 1;
                        parse_literal(&token).unwrap_or_else(|| {
                            panic!("cannot evaluate {token:?} in {:?}", self.expr)
                        })
                    }
                    None => panic!("unexpected end of the expression {:?}", self.expr),
                }
            }
        }

        let tokens = tokenize(expr);
        let mut parser = Parser {
            expr: expr.to_owned(),
            tokens,
            position: 0,
        };
        let value = parser.sum();
        assert_eq!(
            parser.position,
            parser.tokens.len(),
            "trailing tokens in {expr:?}"
        );
        value
    }

    /// Splits a C declaration into its type and its declared name.
    ///
    /// The name starts after the last whitespace that is not nested in brackets or
    /// parentheses, so "payload[4][1024 * 1024]" stays one name while a pointer parameter keeps
    /// its "* __restrict__" in the type.
    fn split_declaration(declaration: &str) -> (String, String) {
        let mut depth = 0i32;
        let mut split = None;
        for (index, ch) in declaration.char_indices() {
            match ch {
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                ' ' if depth == 0 => split = Some(index),
                _ => {}
            }
        }
        let index = split.unwrap_or_else(|| {
            panic!("cannot split the declaration {declaration:?} into type and name")
        });
        (
            declaration[..index].trim().to_owned(),
            declaration[index + 1..].trim().to_owned(),
        )
    }

    /// Field declarations of a struct, in declaration order, as (type, name).
    ///
    /// The declaration is located in the comment-free text: the CUDA header documents the ABI
    /// with a commented copy of the struct, and a parser that matched that copy would check
    /// nothing at all.
    fn struct_fields(text: &str, declaration: &str) -> Vec<(String, String)> {
        let stripped = strip_comments(text);
        let start = stripped
            .find(declaration)
            .unwrap_or_else(|| panic!("the CUDA source no longer declares {declaration}"));
        let tail = &stripped[start..];
        let end = tail
            .find("};")
            .unwrap_or_else(|| panic!("{declaration} has no closing brace"));
        let body = &tail[..end];
        let open = body
            .find('{')
            .unwrap_or_else(|| panic!("{declaration} has no opening brace"));
        body[open + 1..]
            .split(';')
            .map(|statement| statement.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|statement| !statement.is_empty())
            .map(|statement| split_declaration(&statement))
            .collect()
    }

    /// The `__global__` parameter list of the one-shot kernel, as (type, name) pairs.
    ///
    /// Works on the kernel definition and on the build script's prototype, because both spell
    /// the same `extern "C" __global__ void rocenante_oneshot_allreduce_bf16_tp4(`
    /// prefix. The parameters contain no parentheses, so the first `)` closes the list.
    fn kernel_params(text: &str) -> Vec<(String, String)> {
        const DECLARATION: &str =
            "extern \"C\" __global__ void rocenante_oneshot_allreduce_bf16_tp4(";
        let stripped = strip_comments(text);
        let start = stripped.find(DECLARATION).unwrap_or_else(|| {
            panic!("the CUDA source no longer declares {DECLARATION}");
        });
        let tail = &stripped[start + DECLARATION.len()..];
        let end = tail
            .find(')')
            .unwrap_or_else(|| panic!("the {DECLARATION} parameter list is unterminated"));
        tail[..end]
            .split(',')
            .map(|parameter| parameter.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|parameter| !parameter.is_empty())
            .map(|parameter| split_declaration(&parameter))
            .collect()
    }

    /// Dimensions of the `[a][b * c]` suffix of a field name.
    fn array_dims(name: &str) -> Vec<usize> {
        let mut dims = Vec::new();
        let mut rest = name;
        while let Some(open) = rest.find('[') {
            let close = rest[open..]
                .find(']')
                .unwrap_or_else(|| panic!("unbalanced [ in {name:?}"))
                + open;
            dims.push(eval_int_expr(&rest[open + 1..close]));
            rest = &rest[close + 1..];
        }
        dims
    }

    /// Width in bytes of a CUDA type as it appears in the slot and the kernel signature.
    ///
    /// Device pointers are 64-bit on every target this crate supports (aarch64 Grace and
    /// x86-64), which the tests cross-check against `size_of` on the Rust side.
    fn cuda_type_size(cuda_type: &str) -> usize {
        let cleaned = cuda_type
            .replace("volatile", " ")
            .replace("const", " ")
            .replace("__restrict__", " ");
        let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.ends_with('*') {
            return 8;
        }
        match cleaned.as_str() {
            "uint64_t" => 8,
            "uint32_t" => 4,
            "uint16_t" => 2,
            "__nv_bfloat16" => 2,
            other => panic!("no known width for the CUDA type {other:?}"),
        }
    }

    /// The protocol statements of the kernel, in the order specification 5.4 fixes them, as
    /// byte offsets into the comment-free source. Zero means the statement is missing.
    fn protocol_offsets(text: &str) -> Vec<(&'static str, usize)> {
        let stripped = strip_comments(text);
        [
            "slot->payload[my_rank][i] = local_in[i];",
            "slot->arrival_seq[my_rank] = step_seq;",
            "__threadfence_system();",
            "*host_doorbell_seq = step_seq;",
            // Every block stamps its own wait deadline, after the doorbell and before polling.
            "s_wait_start_ns = spark_read_globaltimer_ns();",
            "while (slot->arrival_seq[peer] < step_seq)",
            "float acc = 0.0f;",
            "__float2bfloat16_rn(acc)",
            "SPARK_GPU_TRACE_EXIT_ROCE(",
        ]
        .iter()
        .map(|needle| {
            (
                *needle,
                stripped
                    .find(needle)
                    .unwrap_or_else(|| panic!("the kernel no longer contains {needle:?}")),
            )
        })
        .collect()
    }

    /// True when every protocol statement appears strictly after the previous one.
    fn protocol_order_is_monotonic(text: &str) -> bool {
        let offsets = protocol_offsets(text);
        offsets.windows(2).all(|pair| pair[0].1 < pair[1].1)
    }

    #[test]
    fn the_c_constant_evaluator_handles_the_forms_the_header_uses() {
        // Guards the guard: if eval_int_expr returned nonsense, the checks below that rely on
        // it would silently stop meaning anything.
        assert_eq!(eval_int_expr("4u"), 4);
        assert_eq!(eval_int_expr("1000000ull"), 1_000_000);
        assert_eq!(eval_int_expr("0x80000000u"), 1 << 31);
        assert_eq!(eval_int_expr("(1024u * 1024u)"), 1_048_576);
        assert_eq!(eval_int_expr("(2u + 3u) * 4u"), 20);
        assert_eq!(array_dims("payload[4][1024 * 1024]"), vec![4, 1_048_576]);
        assert_eq!(array_dims("arrival_seq[4]"), vec![4]);
        assert_eq!(cuda_type_size("volatile uint64_t"), 8);
        assert_eq!(cuda_type_size("const __nv_bfloat16* __restrict__"), 8);
        assert_eq!(cuda_type_size("__nv_bfloat16"), 2);
        assert_eq!(cuda_type_size("const uint32_t"), 4);
    }

    #[test]
    fn slot_declaration_is_byte_for_byte_the_specification() {
        // Specification 5.4, lines 646-649. The members are written as literals on purpose:
        // this is the wire layout, and the name of a macro must not be able to change it.
        assert!(
            normalized(ROCE_ONESHOT_HEADER).contains(
                "struct RoceOneShotSlot { volatile uint64_t arrival_seq[4]; \
                 __nv_bfloat16 payload[4][1024 * 1024]; };"
            ),
            "the CUDA slot declaration must stay byte for byte what specification 5.4 declares"
        );
    }

    #[test]
    fn slot_geometry_matches_the_rust_mirror() {
        let fields = struct_fields(ROCE_ONESHOT_HEADER, "struct RoceOneShotSlot");
        assert_eq!(
            fields,
            vec![
                ("volatile uint64_t".to_owned(), "arrival_seq[4]".to_owned()),
                (
                    "__nv_bfloat16".to_owned(),
                    "payload[4][1024 * 1024]".to_owned()
                ),
            ],
            "arrival_seq must come first, then the payload, with these exact types"
        );

        // Re-derive the C geometry from the parsed declaration instead of trusting the
        // constants: dimensions * element width.
        let arrival_dims = array_dims(&fields[0].1);
        assert_eq!(arrival_dims, vec![ROCE_ONESHOT_RANKS]);
        let arrival_bytes = arrival_dims.iter().product::<usize>() * cuda_type_size(&fields[0].0);
        assert_eq!(arrival_bytes, ROCE_ONESHOT_SLOT_ARRIVAL_BYTES);
        assert_eq!(arrival_bytes, 32);

        let payload_dims = array_dims(&fields[1].1);
        assert_eq!(
            payload_dims,
            vec![ROCE_ONESHOT_RANKS, ROCE_ONESHOT_SLOT_ROW_ELEMS]
        );
        let payload_bytes = payload_dims.iter().product::<usize>() * cuda_type_size(&fields[1].0);
        assert_eq!(payload_bytes, ROCE_ONESHOT_SLOT_PAYLOAD_BYTES);
        assert_eq!(payload_bytes, 4 * 2 * 1024 * 1024);
        assert_eq!(arrival_bytes + payload_bytes, ROCE_ONESHOT_SLOT_BYTES);
        assert_eq!(ROCE_ONESHOT_SLOT_BYTES, 8_388_640);
        assert_eq!(arrival_bytes + payload_bytes, 8 * 1024 * 1024 + 32);

        // And now the Rust side, which is what the host code actually passes.
        assert_eq!(std::mem::size_of::<bf16>(), 2);
        assert_eq!(std::mem::align_of::<bf16>(), 2);
        assert_eq!(
            std::mem::size_of::<RoceOneShotSlot>(),
            ROCE_ONESHOT_SLOT_BYTES
        );
        assert_eq!(
            std::mem::align_of::<RoceOneShotSlot>(),
            ROCE_ONESHOT_SLOT_ALIGN
        );
        assert_eq!(std::mem::offset_of!(RoceOneShotSlot, arrival_seq), 0);
        assert_eq!(std::mem::offset_of!(RoceOneShotSlot, payload), 32);
        assert_eq!(
            std::mem::size_of::<[bf16; ROCE_ONESHOT_SLOT_ROW_ELEMS]>(),
            ROCE_ONESHOT_SLOT_ROW_BYTES
        );
        // The rows are dense: no padding between them, so payload[r] starts at r * 2 MiB.
        assert_eq!(
            std::mem::size_of::<[[bf16; ROCE_ONESHOT_SLOT_ROW_ELEMS]; ROCE_ONESHOT_RANKS]>(),
            ROCE_ONESHOT_SLOT_PAYLOAD_BYTES
        );
        assert_eq!(
            ROCE_ONESHOT_SLOT_BYTES - std::mem::offset_of!(RoceOneShotSlot, payload),
            payload_bytes
        );
    }

    #[test]
    fn cuda_header_static_asserts_cover_the_layout() {
        // The C-side gate must not be deletable without a test noticing. The comparison drops
        // every space so that reformatting a multi-line static_assert cannot break the check.
        let header = normalized(ROCE_ONESHOT_HEADER).replace(' ', "");
        for needle in [
            "static_assert(sizeof(__nv_bfloat16)==2",
            "static_assert(sizeof(RoceOneShotSlot)==SPARK_ROCE_SLOT_BYTES",
            "static_assert(alignof(RoceOneShotSlot)==8",
            "static_assert(offsetof(RoceOneShotSlot,arrival_seq)==0",
            "static_assert(offsetof(RoceOneShotSlot,payload)==32",
        ] {
            assert!(header.contains(needle), "the CUDA header lost: {needle}");
        }
    }

    #[test]
    fn cuda_defines_match_the_rust_constants() {
        let header = ROCE_ONESHOT_HEADER;
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCE_SLOT_RANKS")),
            ROCE_ONESHOT_RANKS
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCE_SLOT_ROW_ELEMS")),
            ROCE_ONESHOT_SLOT_ROW_ELEMS
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCENANTE_PEER_WAIT_NS")),
            ROCE_PEER_WAIT_TIMEOUT_NS as usize
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCENANTE_SPIN_ITERS")),
            ROCE_SPIN_ITERS as usize
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCENANTE_SPIN_SLEEP_NS")),
            ROCE_SPIN_SLEEP_NS as usize
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCE_PAYLOAD_DIAG_FLAG")),
            RoceTracePayload::DIAG_FLAG as usize
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCE_DIAG_MISSING_SHIFT")),
            RoceTracePayload::MISSING_SHIFT as usize
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCE_DIAG_MISSING_MASK")),
            usize::from(RoceTracePayload::MISSING_MASK)
        );
        assert_eq!(
            eval_int_expr(&cuda_define(header, "SPARK_ROCE_DIAG_ROW_OVERFLOW")),
            RoceTracePayload::ROW_OVERFLOW as usize
        );

        // The derived byte defines name their inputs, so they are pinned textually; their
        // numbers are pinned by the static asserts and by the geometry test above.
        assert_eq!(
            cuda_define(header, "SPARK_ROCE_SLOT_ROW_BYTES"),
            "((unsigned long long)(SPARK_ROCE_SLOT_ROW_ELEMS) * 2ull)"
        );
        assert_eq!(
            cuda_define(header, "SPARK_ROCE_SLOT_ARRIVAL_BYTES"),
            "((unsigned long long)(SPARK_ROCE_SLOT_RANKS) * 8ull)"
        );
        let total = cuda_define(header, "SPARK_ROCE_SLOT_BYTES");
        assert!(
            total.contains("SPARK_ROCE_SLOT_RANKS") && total.contains("SPARK_ROCE_SLOT_ROW_BYTES"),
            "the slot byte total must stay derived from the rank count and the row size: {total}"
        );
    }

    #[test]
    fn kernel_signature_matches_the_rust_declarations() {
        let parsed = kernel_params(ROCE_ONESHOT_KERNEL);
        let expected = ROCE_ONESHOT_KERNEL_PARAMS
            .iter()
            .map(|(ty, name)| ((*ty).to_owned(), (*name).to_owned()))
            .collect::<Vec<_>>();
        assert_eq!(
            parsed, expected,
            "the kernel's __global__ signature must match ROCE_ONESHOT_KERNEL_PARAMS"
        );
        assert_eq!(parsed.len(), ROCE_ONESHOT_KERNEL_PARAM_BYTES.len());
        assert_eq!(parsed.len(), 9);

        // The CUDA type spelling and the Rust type must agree on the byte width, which is the
        // part a name-for-name comparison cannot see.
        for ((cuda_type, name), rust_bytes) in parsed.iter().zip(ROCE_ONESHOT_KERNEL_PARAM_BYTES) {
            assert_eq!(
                cuda_type_size(cuda_type),
                *rust_bytes,
                "parameter {name} is declared {cuda_type} in CUDA but {rust_bytes} bytes in Rust"
            );
        }
    }

    #[test]
    fn build_probe_redeclares_the_kernel_signature_and_the_layout_literals() {
        // Without nvcc the probe is never compiled, so check at the source level that it still
        // exists and still says the same thing.
        let probe = ROCE_BUILD_SCRIPT;
        assert_eq!(
            kernel_params(probe),
            kernel_params(ROCE_ONESHOT_KERNEL),
            "the build-script probe must redeclare exactly the kernel's parameter list"
        );
        assert!(
            probe.contains("static_assert(sizeof(RoceOneShotSlot) == 8388640"),
            "the build-script probe must repeat the slot size as a literal"
        );
        assert!(
            probe.contains("static_assert(offsetof(RoceOneShotSlot, payload) == 32"),
            "the build-script probe must repeat the payload offset as a literal"
        );
        assert!(
            probe.contains("#include \"rocenante_oneshot_sm121.cu\""),
            "the build-script probe must compile the kernel together with the prototype"
        );
    }

    #[test]
    fn trace_payload_round_trips_and_keeps_the_flag_above_every_byte_count() {
        assert!(RoceTracePayload::DIAG_FLAG > ROCE_ONESHOT_SLOT_ROW_BYTES as u32);
        for bytes in [0u32, 1, 10 * 1024, ROCE_ONESHOT_SLOT_ROW_BYTES as u32] {
            let payload = RoceTracePayload::Transfer { bytes };
            assert_eq!(payload.encode(), bytes);
            assert_eq!(RoceTracePayload::decode(bytes), payload);
        }
        for missing in 0..=0x0fu8 {
            for row_overflow in [false, true] {
                let payload = RoceTracePayload::Diagnostic {
                    missing_ranks: missing,
                    row_overflow,
                };
                let encoded = payload.encode();
                assert_ne!(encoded & RoceTracePayload::DIAG_FLAG, 0);
                assert_eq!(RoceTracePayload::decode(encoded), payload);
            }
        }
        // A timeout on rank 3 plus a row overflow is the richest diagnostic record.
        let encoded = RoceTracePayload::Diagnostic {
            missing_ranks: 1 << 3,
            row_overflow: true,
        }
        .encode();
        assert_eq!(
            RoceTracePayload::decode(encoded),
            RoceTracePayload::Diagnostic {
                missing_ranks: 8,
                row_overflow: true
            }
        );
    }

    #[test]
    fn trace_payload_classifies_events_by_scope() {
        let event = GpuTraceEvent {
            scope_and_rank: (14u32 << 8) | 2,
            payload_meta: 4096,
            ..GpuTraceEvent::default()
        };
        assert_eq!(
            RoceTracePayload::from_event(&event),
            Some(RoceTracePayload::Transfer { bytes: 4096 })
        );
        let other = GpuTraceEvent {
            scope_and_rank: (15u32 << 8) | 2,
            payload_meta: 4096,
            ..GpuTraceEvent::default()
        };
        assert_eq!(RoceTracePayload::from_event(&other), None);
    }

    /// Byte index just past the `}` that closes the block opened at `open`.
    fn block_end(text: &str, open: usize) -> Option<usize> {
        let mut depth = 0usize;
        for (offset, byte) in text.as_bytes()[open..].iter().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(open + offset + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Whether the kernel's single acquire fence sits AFTER the
    /// `if (slot->arrival_seq[peer] < step_seq)` block and before the next `__syncthreads();`.
    ///
    /// Placement is the whole point. While the fence sat inside that branch, a peer whose
    /// arrival was already visible when the block reached the poll was read without an acquire
    /// paired to the writer's release fence - the defect the M3 review found. Counting fences,
    /// or checking the coarse stage order, cannot see the difference: the review deleted the
    /// fence and moved it back inside the branch, and all 32 tests still passed.
    fn acquire_fence_is_unconditional(stripped: &str) -> bool {
        if stripped.matches("__threadfence();").count() != 1 {
            return false;
        }
        let branch = "if (slot->arrival_seq[peer] < step_seq) {";
        let Some(branch_at) = stripped.find(branch) else {
            return false;
        };
        let open = branch_at + branch.len() - 1;
        let Some(end) = block_end(stripped, open) else {
            return false;
        };
        let Some(fence) = stripped.find("__threadfence();") else {
            return false;
        };
        if fence < end {
            return false;
        }
        let after = &stripped[end..];
        let Some(next) = after.find("__syncthreads();") else {
            return false;
        };
        if fence >= end + next {
            return false;
        }
        // Position alone is not enough: a fence wrapped in a condition that never holds, or
        // preceded by an early return, sits between the same two anchors while never running.
        // The M3 review showed three such variants passing the weaker positional form, so the
        // fence must also be the first statement after the branch.
        after.trim_start().starts_with("__threadfence();")
    }

    #[test]
    fn kernel_preserves_the_specified_protocol_order() {
        assert!(
            protocol_order_is_monotonic(ROCE_ONESHOT_KERNEL),
            "the kernel must stage, publish, fence, ring, wait, then reduce, then trace"
        );
        let stripped = strip_comments(ROCE_ONESHOT_KERNEL);
        // One accumulator start, one ranking loop in ascending rank order, one bf16 rounding,
        // and no second rounding anywhere in the kernel.
        assert_eq!(stripped.matches("float acc = 0.0f;").count(), 1);
        assert_eq!(stripped.matches("__float2bfloat16_rn").count(), 1);
        assert_eq!(stripped.matches("__threadfence_system();").count(), 1);
        assert!(stripped.contains(
            "for (uint32_t r = 0; r < (uint32_t)SPARK_ROCE_SLOT_RANKS; ++r) {\n                \
             acc += __bfloat162float(slot->payload[r][i]);"
        ));
        // The wait must be bounded: the deadline is compared inside the spin.
        assert!(stripped.contains("if (spark_read_globaltimer_ns() >= deadline)"));
        assert!(stripped.contains("__nanosleep((unsigned int)SPARK_ROCENANTE_SPIN_SLEEP_NS);"));
        // The acquire fence must be unconditional on the observing path: after the peer-arrival
        // branch, before the barrier that publishes the observation to the rest of the block.
        assert!(
            acquire_fence_is_unconditional(&stripped),
            "exactly one __threadfence() must sit after the peer-arrival branch and before the \
             following __syncthreads(); a conditional fence leaves an already-visible peer \
             unacquired"
        );
        // Regression guard: only global thread 0 rings the doorbell, but every block polls the
        // slot. s_wait_start_ns is __shared__, so a single global assignment would leave blocks
        // other than block 0 comparing against uninitialised shared memory. The deadline must
        // be stamped per block by that block's thread 0.
        assert_eq!(
            stripped
                .matches("s_wait_start_ns = spark_read_globaltimer_ns();")
                .count(),
            1,
            "the shared wait deadline must be stamped exactly once, inside a per-block guard"
        );
        assert!(stripped.contains(
            "if (threadIdx.x == 0u) {
        s_wait_start_ns = spark_read_globaltimer_ns();
    }"
        ));
        assert!(stripped.contains(
            "const uint64_t deadline = s_wait_start_ns + (uint64_t)SPARK_ROCENANTE_PEER_WAIT_NS;"
        ));
        // Zero-element calls still run the handshake and record zero bytes.
        assert!(stripped.contains("(uint32_t)(n * 2u)"));
        assert!(stripped.contains("const uint32_t n = row_overflow ? row_elems : num_elements;"));
    }

    /// The order check above is only worth having if it catches a reordering: prove it on a
    /// mutated copy of the real kernel instead of trusting it.
    #[test]
    fn the_protocol_order_check_detects_a_reordering_and_a_second_rounding() {
        assert!(protocol_order_is_monotonic(ROCE_ONESHOT_KERNEL));

        // Ring the doorbell before the fence.
        let fence_late = ROCE_ONESHOT_KERNEL.replacen(
            "__threadfence_system();\n        *host_doorbell_seq = step_seq;",
            "*host_doorbell_seq = step_seq;\n        __threadfence_system();",
            1,
        );
        assert_ne!(
            fence_late, ROCE_ONESHOT_KERNEL,
            "the mutation must change the text"
        );
        assert!(!protocol_order_is_monotonic(&fence_late));

        // Round to bf16 before the canonical accumulation.
        let early_rounding = ROCE_ONESHOT_KERNEL.replacen(
            "float acc = 0.0f;",
            "float acc = __bfloat162float(slot->payload[0][i]);",
            1,
        );
        assert_ne!(early_rounding, ROCE_ONESHOT_KERNEL);
        assert_eq!(
            strip_comments(&early_rounding)
                .matches("float acc = 0.0f;")
                .count(),
            0
        );
        assert!(!strip_comments(&early_rounding).contains("float acc = 0.0f;"));

        // The acquire fence's PLACEMENT is load bearing, and the count/order checks above
        // cannot see it: prove that both ways of losing it are caught.
        let no_fence = ROCE_ONESHOT_KERNEL.replacen("        __threadfence();\n", "", 1);
        assert_ne!(no_fence, ROCE_ONESHOT_KERNEL);
        assert!(!acquire_fence_is_unconditional(&strip_comments(&no_fence)));
        let mut inside = ROCE_ONESHOT_KERNEL.replacen(
            "        }\n        // Acquire fence, and deliberately UNCONDITIONAL.",
            "            __threadfence();\n        }\n        // Acquire fence, now inside the branch.",
            1,
        );
        inside = inside.replacen("        __threadfence();\n    }\n", "    }\n", 1);
        assert_ne!(inside, ROCE_ONESHOT_KERNEL);
        let inside_stripped = strip_comments(&inside);
        assert_eq!(inside_stripped.matches("__threadfence();").count(), 1);
        assert!(
            !acquire_fence_is_unconditional(&inside_stripped),
            "a fence moved inside the not-yet-arrived branch must be rejected"
        );

        // A fence that is present but cannot run also satisfies a purely positional check.
        let never_runs = ROCE_ONESHOT_KERNEL.replacen(
            "        __threadfence();",
            "        if (false) { __threadfence(); }",
            1,
        );
        assert_ne!(never_runs, ROCE_ONESHOT_KERNEL);
        assert!(
            !acquire_fence_is_unconditional(&strip_comments(&never_runs)),
            "a fence that can never execute must be rejected"
        );

        let early_return = ROCE_ONESHOT_KERNEL.replacen(
            "        __threadfence();",
            "        if (s_peer_timed_out[peer] != 0u) { return; }\n        __threadfence();",
            1,
        );
        assert_ne!(early_return, ROCE_ONESHOT_KERNEL);
        assert!(
            !acquire_fence_is_unconditional(&strip_comments(&early_return)),
            "an early return before the fence must be rejected"
        );

        // A second rounding is visible in the count.
        let double_round = ROCE_ONESHOT_KERNEL.replacen(
            "local_out[i] = __float2bfloat16_rn(acc);",
            "local_out[i] = __float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(acc)));",
            1,
        );
        assert_ne!(double_round, ROCE_ONESHOT_KERNEL);
        assert_eq!(
            strip_comments(&double_round)
                .matches("__float2bfloat16_rn")
                .count(),
            2
        );
    }

    #[test]
    fn the_slot_row_is_the_router_threshold() {
        // The router sends everything <= 2 MiB here, and 2 MiB is exactly one payload row:
        // that is why the kernel's clamp is a no-op for a conforming caller.
        assert_eq!(ROCE_ONESHOT_SLOT_ROW_BYTES, 2 * 1024 * 1024);
        assert_eq!(ROCE_ONESHOT_SLOT_ROW_BYTES, ROCE_ONESHOT_SLOT_ROW_ELEMS * 2);
    }

    /// The conformance tests above are only worth having if the parsers notice a drift, so
    /// drive them with deliberately broken copies of the real sources.
    #[test]
    fn the_slot_parser_detects_a_reordering_and_a_width_change() {
        let expected = struct_fields(ROCE_ONESHOT_HEADER, "struct RoceOneShotSlot");
        assert_eq!(expected.len(), 2);

        // Swap the two members. The byte total happens to be identical either way, so only the
        // field-by-field comparison can catch this one.
        let swapped = ROCE_ONESHOT_HEADER
            .replace("volatile uint64_t arrival_seq[4]", "@@ARRIVAL@@")
            .replace(
                "__nv_bfloat16 payload[4][1024 * 1024]",
                "volatile uint64_t arrival_seq[4]",
            )
            .replace("@@ARRIVAL@@", "__nv_bfloat16 payload[4][1024 * 1024]");
        assert_ne!(
            swapped, ROCE_ONESHOT_HEADER,
            "the mutation must change the text"
        );
        let mutated = struct_fields(&swapped, "struct RoceOneShotSlot");
        assert_ne!(
            mutated, expected,
            "a member reordering must change the parsed list"
        );
        assert_eq!(mutated[0].0, "__nv_bfloat16");
        assert_eq!(mutated[1].0, "volatile uint64_t");

        // Narrow arrival_seq's element type. The declaration order is untouched, so this is
        // caught by the fields comparison and by the re-derived byte total.
        let narrowed = ROCE_ONESHOT_HEADER.replace(
            "volatile uint64_t arrival_seq",
            "volatile uint32_t arrival_seq",
        );
        assert_ne!(narrowed, ROCE_ONESHOT_HEADER);
        let narrowed_fields = struct_fields(&narrowed, "struct RoceOneShotSlot");
        assert_ne!(narrowed_fields, expected);
        let narrowed_bytes = array_dims(&narrowed_fields[0].1).iter().product::<usize>()
            * cuda_type_size(&narrowed_fields[0].0);
        assert_eq!(narrowed_bytes, 16);
        assert_ne!(narrowed_bytes, ROCE_ONESHOT_SLOT_ARRIVAL_BYTES);
    }

    /// Same idea for the kernel signature: a swapped pointer and a widened scalar must both
    /// change the parsed parameter list.
    #[test]
    fn the_kernel_signature_parser_detects_a_swapped_parameter_and_a_width_change() {
        let expected = kernel_params(ROCE_ONESHOT_KERNEL);
        assert_eq!(expected.len(), 9);

        let swapped = ROCE_ONESHOT_KERNEL
            .replace("const __nv_bfloat16* __restrict__ local_in", "@@IN@@")
            .replace(
                "__nv_bfloat16* __restrict__ local_out",
                "const __nv_bfloat16* __restrict__ local_in",
            )
            .replace("@@IN@@", "__nv_bfloat16* __restrict__ local_out");
        assert_ne!(
            swapped, ROCE_ONESHOT_KERNEL,
            "the mutation must change the text"
        );
        let mutated = kernel_params(&swapped);
        assert_ne!(mutated, expected);
        assert_eq!(mutated[1].1, "local_in");
        assert_eq!(mutated[0].1, "local_out");

        let widened = ROCE_ONESHOT_KERNEL
            .replace("const uint32_t num_elements", "const uint64_t num_elements");
        assert_ne!(widened, ROCE_ONESHOT_KERNEL);
        let widened_params = kernel_params(&widened);
        assert_ne!(widened_params, expected);
        let (cuda_type, _) = widened_params
            .iter()
            .find(|(_, name)| name == "num_elements")
            .expect("num_elements is still a parameter");
        assert_eq!(cuda_type_size(cuda_type), 8);
        assert_ne!(
            cuda_type_size(cuda_type),
            ROCE_ONESHOT_KERNEL_PARAM_BYTES[7]
        );
    }

    /// The clang fallback is the only frontend most hosts have, so pin the wiring that makes it
    /// check the ABI instead of only parsing the kernel: it must run the probe, and the stub
    /// must declare the device functions the trace macros call. Without those declarations the
    /// probe cannot compile with the trace enabled, which is how the kernel's trace call gets
    /// type-checked at all off a CUDA host.
    #[test]
    fn the_clang_fallback_compiles_the_abi_probe_with_the_trace_enabled() {
        assert!(
            ROCE_BUILD_SCRIPT.contains("fn check_roce_probe_syntax"),
            "the build script must keep a clang path that compiles the ABI probe"
        );
        assert!(
            ROCE_BUILD_SCRIPT.contains("report_clang_probe"),
            "the clang path must report what the ABI probe checked"
        );
        assert!(
            ROCE_BUILD_SCRIPT.contains("DSPARKFLASH_CUDA_SYNTAX_STUB=1"),
            "the clang path must define the stub switch"
        );
        for declaration in ["atomicAdd", "__ffsll"] {
            assert!(
                ROCE_CUDA_STUB.contains(declaration),
                "the stub CUDA headers must declare {declaration} so the probe can compile \
                 the trace macros"
            );
        }
        assert!(
            !strip_comments(ROCE_CUDA_STUB).contains("__syncthreads("),
            "the stub must not define __syncthreads: clang provides it as a builtin and \
             redefining it is an error"
        );
    }
}
