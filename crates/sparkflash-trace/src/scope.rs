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

//! Scoped trace domains.
//!
//! TraceScope mirrors the 16-bit scope table in section 5.8 of the SparkFlash-RS spec.
//! Every bit is a *domain*: flipping it in GLOBAL_TRACE_MASK activates exactly the trace
//! points instrumented inside that domain, on both the Grace CPU side and (through the
//! coherent-UMA mirror) inside already-captured CUDA graphs.

use std::fmt;

/// Bit flags identifying instrumented subsystems.
///
/// The bit layout is part of the wire contract with the sm_121a kernels
/// (kernels/trace_globaltimer_sm121.cuh), so never renumber an existing bit.
pub struct TraceScope;

impl TraceScope {
    /// Empty mask: no domain is instrumented.
    pub const NONE: u64 = 0;
    /// Prompt and image encoding: tokenization, NFKC compression, ViT unshuffle.
    pub const API_ENCODE: u64 = 1 << 0;
    /// Request scheduling: queue wait and PREFILL_BYPASS interleaving.
    pub const SCHED_BATCH: u64 = 1 << 1;
    /// KV manager: radix lookup, 64-token page allocation, NVMe KV load.
    pub const KV_RADIX: u64 = 1 << 2;
    /// Sliding-window rebuild: encoder replay versus decoder replay of the tail.
    pub const SWA_REPLAY: u64 = 1 << 3;
    /// Engram CPU tiers: 24-head XOR hash plus Tier-1 DRM and Tier-2 UMA lookups.
    pub const ENGRAM_PREFETCH: u64 = 1 << 4;
    /// Engram NVMe tier: io_uring submission, queue depth and O_DIRECT completion.
    pub const ENGRAM_IO_URING: u64 = 1 << 5;
    /// Engram gate kernel: wkv GEMM, RMSNorm dot-product gate and mean alpha.
    pub const ENGRAM_GPU_GATE: u64 = 1 << 6;
    /// Mega-mHC kernel: residual mix, Sinkhorn-20 and the dynamic FP8 cast.
    pub const MEGA_MHC: u64 = 1 << 7;
    /// FlashMLA CSA2 indexer: Top-512 scoring and the HSI pool build.
    pub const CSA2_INDEXER: u64 = 1 << 8;
    /// FlashMLA CSA2 attention: FP4 dequant, 64-token paged MLA, FP8 SWA combine.
    pub const CSA2_ATTN: u64 = 1 << 9;
    /// Dense MXFP8 projections: wqkv_a, wq_b, wo_a and the shared expert GEMMs.
    pub const DENSE_MXFP8: u64 = 1 << 10;
    /// MXFP4 MoE: route planning, the grouped W4A4 GEMM and its deterministic sum.
    pub const MOE_MXFP4: u64 = 1 << 11;
    /// DSpark drafter: the layers 37..39 draft pass, the confidence cap and anchor-expert reuse.
    /// The rank-256 Markov head is not modelled: its logits are inputs to the survival math.
    pub const DSPARK_DRAFT: u64 = 1 << 12;
    /// RoCEnante RDMA: staging, the peer-ready wait and the canonical-order reduce.
    pub const ROCENANTE_RDMA: u64 = 1 << 13;
    /// NCCL fallback collectives, used during sequence-parallel prefill.
    pub const NCCL_COLLECTIVE: u64 = 1 << 14;
    /// Side-stream L2 warner that overlaps the RoCEnante wait window.
    pub const L2_PREFETCH: u64 = 1 << 15;
    /// Every defined domain set, i.e. all sixteen bits.
    pub const ALL: u64 = 0xFFFF;

    /// Number of defined scopes.
    pub const COUNT: usize = 16;

    /// Table of (NAME, bit) in bit order.
    pub const TABLE: [(&'static str, u64); Self::COUNT] = [
        ("API_ENCODE", Self::API_ENCODE),
        ("SCHED_BATCH", Self::SCHED_BATCH),
        ("KV_RADIX", Self::KV_RADIX),
        ("SWA_REPLAY", Self::SWA_REPLAY),
        ("ENGRAM_PREFETCH", Self::ENGRAM_PREFETCH),
        ("ENGRAM_IO_URING", Self::ENGRAM_IO_URING),
        ("ENGRAM_GPU_GATE", Self::ENGRAM_GPU_GATE),
        ("MEGA_MHC", Self::MEGA_MHC),
        ("CSA2_INDEXER", Self::CSA2_INDEXER),
        ("CSA2_ATTN", Self::CSA2_ATTN),
        ("DENSE_MXFP8", Self::DENSE_MXFP8),
        ("MOE_MXFP4", Self::MOE_MXFP4),
        ("DSPARK_DRAFT", Self::DSPARK_DRAFT),
        ("ROCENANTE_RDMA", Self::ROCENANTE_RDMA),
        ("NCCL_COLLECTIVE", Self::NCCL_COLLECTIVE),
        ("L2_PREFETCH", Self::L2_PREFETCH),
    ];

    /// Returns the canonical upper-case name for a single-bit scope mask.
    #[must_use]
    pub fn name(bit: u64) -> Option<&'static str> {
        Self::TABLE.iter().find(|(_, b)| *b == bit).map(|(n, _)| *n)
    }

    /// Returns the zero-based bit index of a single-bit scope mask.
    #[must_use]
    pub fn index(bit: u64) -> Option<u32> {
        if bit.count_ones() == 1 {
            Some(bit.trailing_zeros())
        } else {
            None
        }
    }

    /// Comma separated rendering of every set bit, in bit order.
    #[must_use]
    pub fn describe(mask: u64) -> String {
        let mut out = Vec::new();
        for (name, bit) in Self::TABLE {
            if mask & bit != 0 {
                out.push(name);
            }
        }
        if mask & !Self::ALL != 0 {
            out.push("UNKNOWN");
        }
        out.join(",")
    }
}

/// Why a scope specification could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScopeParseError {
    /// The specification contained no tokens at all.
    #[error("empty scope specification")]
    Empty,
    /// A token matched no scope name; carries the rejected token.
    #[error("unknown trace scope: {0}")]
    UnknownScope(String),
    /// More entries than the parser accepts; carries the maximum it takes.
    #[error("scope specification has too many entries (max {0})")]
    TooMany(usize),
}

/// Parses one scope token. Accepts the canonical name (ENGRAM_PREFETCH) or its
/// lower-case / dashed form (engram-prefetch, engram_prefetch), plus all and *.
#[must_use]
pub fn parse_scope_token(token: &str) -> Option<u64> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    if token.eq_ignore_ascii_case("all") || token == "*" {
        return Some(TraceScope::ALL);
    }
    let normalized = normalize(token);
    // The CUDA header spells the scopes TRACE_SCOPE_CSA2_ATTN; accept that form too so a
    // name can be copy-pasted straight out of the kernel source.
    let stripped = normalized.strip_prefix("tracescope").unwrap_or(&normalized);
    TraceScope::TABLE
        .iter()
        .find(|(name, _)| {
            let canonical = normalize(name);
            canonical == normalized || canonical == stripped
        })
        .map(|(_, bit)| *bit)
}

fn parse_list(spec: &str, resolve: impl Fn(&str) -> Option<u64>) -> Result<u64, ScopeParseError> {
    let mut mask = 0u64;
    let mut any = false;
    let mut count = 0usize;
    for raw in spec.split([',', ' ', '\t', ';']) {
        let token = raw.trim();
        if token.is_empty() {
            continue;
        }
        count += 1;
        if count > 64 {
            return Err(ScopeParseError::TooMany(64));
        }
        let bit = resolve(token).ok_or_else(|| ScopeParseError::UnknownScope(token.to_owned()))?;
        mask |= bit;
        any = true;
    }
    if any {
        Ok(mask)
    } else {
        Err(ScopeParseError::Empty)
    }
}

/// Parses a list of canonical scope names, e.g. "ENGRAM_PREFETCH,CSA2_ATTN".
///
/// # Errors
/// Returns ScopeParseError when a token is unknown or the list is empty.
pub fn parse_scopes(spec: &str) -> Result<u64, ScopeParseError> {
    parse_list(spec, parse_scope_token)
}

/// Accepts the short aliases used by the X-SparkFlash-Trace HTTP header, where canonical
/// names are abbreviated (engram covers all three Engram scopes, roce covers ROCENANTE_RDMA,
/// and so on).
#[must_use]
pub fn parse_scope_alias(token: &str) -> Option<u64> {
    let n = normalize(token);
    let alias: Option<u64> = match n.as_str() {
        "engram" => Some(
            TraceScope::ENGRAM_PREFETCH | TraceScope::ENGRAM_IO_URING | TraceScope::ENGRAM_GPU_GATE,
        ),
        "roce" | "rocenante" | "rdma" => Some(TraceScope::ROCENANTE_RDMA),
        "nccl" => Some(TraceScope::NCCL_COLLECTIVE),
        "kv" => Some(TraceScope::KV_RADIX | TraceScope::SWA_REPLAY),
        "csa2" => Some(TraceScope::CSA2_INDEXER | TraceScope::CSA2_ATTN),
        "moe" => Some(TraceScope::MOE_MXFP4),
        "dense" => Some(TraceScope::DENSE_MXFP8),
        "mhc" => Some(TraceScope::MEGA_MHC),
        "dspark" | "spec" => Some(TraceScope::DSPARK_DRAFT),
        "sched" => Some(TraceScope::SCHED_BATCH),
        "api" => Some(TraceScope::API_ENCODE),
        "l2" => Some(TraceScope::L2_PREFETCH),
        _ => None,
    };
    alias.or_else(|| parse_scope_token(token))
}

/// Parses a scope list where short aliases are accepted as well.
///
/// # Errors
/// Returns ScopeParseError when a token is unknown or the list is empty.
pub fn parse_scopes_with_aliases(spec: &str) -> Result<u64, ScopeParseError> {
    parse_list(spec, parse_scope_alias)
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '_' && *c != '-' && *c != ' ')
        .flat_map(char::to_lowercase)
        .collect()
}

impl fmt::Display for TraceScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TraceScope")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_covers_every_bit_exactly_once() {
        let mut seen = 0u64;
        for (_, bit) in TraceScope::TABLE {
            assert_eq!(bit.count_ones(), 1);
            assert_eq!(seen & bit, 0, "duplicate scope bit");
            seen |= bit;
        }
        assert_eq!(seen, TraceScope::ALL);
    }

    #[test]
    fn name_and_index_round_trip() {
        for (name, bit) in TraceScope::TABLE {
            assert_eq!(TraceScope::name(bit), Some(name));
            assert_eq!(TraceScope::index(bit), Some(bit.trailing_zeros()));
        }
        assert_eq!(TraceScope::name(TraceScope::ALL), None);
    }

    #[test]
    fn parse_canonical_and_alias_lists() {
        assert_eq!(
            parse_scopes("ENGRAM_PREFETCH,roce").unwrap_err(),
            ScopeParseError::UnknownScope("roce".into())
        );
        assert_eq!(
            parse_scopes("ENGRAM_PREFETCH, CSA2_ATTN").unwrap(),
            TraceScope::ENGRAM_PREFETCH | TraceScope::CSA2_ATTN
        );
        assert_eq!(parse_scopes("all").unwrap(), TraceScope::ALL);
        assert_eq!(
            parse_scopes_with_aliases("engram,roce,nccl").unwrap(),
            TraceScope::ENGRAM_PREFETCH
                | TraceScope::ENGRAM_IO_URING
                | TraceScope::ENGRAM_GPU_GATE
                | TraceScope::ROCENANTE_RDMA
                | TraceScope::NCCL_COLLECTIVE
        );
        assert_eq!(
            parse_scopes_with_aliases("engram-prefetch").unwrap(),
            TraceScope::ENGRAM_PREFETCH
        );
        assert_eq!(parse_scopes("").unwrap_err(), ScopeParseError::Empty);
        assert_eq!(parse_scopes(" , ,").unwrap_err(), ScopeParseError::Empty);
    }

    #[test]
    fn describe_lists_set_bits_in_order() {
        assert_eq!(
            TraceScope::describe(TraceScope::MOE_MXFP4 | TraceScope::API_ENCODE),
            "API_ENCODE,MOE_MXFP4"
        );
        assert_eq!(TraceScope::describe(TraceScope::NONE), "");
        assert_eq!(TraceScope::describe(1 << 40), "UNKNOWN");
    }
}
