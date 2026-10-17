//! Host-side guard for the Mega-mHC grid reduction's determinism.
//!
//! The direct test is the device one: `examples/m7_layer0.rs --stage 4` launches the kernel N times
//! on the same input bytes and requires every compared output byte to match run 0. This file is the
//! cheap half that can run anywhere, including a host with no GPU: it reads the kernel source and
//! fails when the shape that made the reduction nondeterministic comes back.
//!
//! A text check is a shape check, not a proof. It can see an `atomicAdd` into the statistics buffer
//! and the loss of the fixed per-tile slots or the fixed-order combine; it cannot see a compiler
//! turning the fixed loop into something else. That is why the device test exists, and why this file
//! says so rather than implying it covers the same ground.
//!
//! The old reduction was:
//!
//! ```text
//! for (p) atomicAdd(&hc_stats[p], s_stats[p]);
//! ```
//!
//! four blocks, one fp32 atomic per statistic, in whatever order the blocks reached the counter.
//! The fix gives each tile a slot fixed by blockIdx and has the last tile combine the four partial
//! vectors in tile order in binary64, rounding once. Both properties are pinned below.

use sparkflash_kernels::c_decl::strip_comments;
use sparkflash_kernels::m4_abi::{MEGA_MHC, consts};

/// The kernel with comments removed, so a comment that names the old pattern cannot trip the scan.
fn code() -> String {
    strip_comments(MEGA_MHC)
}

/// Collapse every run of whitespace to one space, so formatting cannot decide whether the
/// mechanism is present.
fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every complete `atomicAdd(...)` call in the source, braces and inner parentheses included.
fn atomic_calls(source: &str) -> Vec<String> {
    let mut calls = Vec::new();
    let mut at = 0usize;
    while let Some(relative) = source[at..].find("atomicAdd") {
        let start = at + relative;
        let open = source[start..]
            .find('(')
            .map(|offset| start + offset)
            .expect("an atomicAdd name is followed by a call");
        let mut depth = 0usize;
        let mut end = None;
        for (offset, character) in source[open..].char_indices() {
            match character {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + offset);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.expect("an atomicAdd call must close its parenthesis");
        calls.push(source[start..=end].to_string());
        at = end + 1;
    }
    calls
}

#[test]
fn no_atomic_touches_the_grid_statistics() {
    let source = code();
    let calls = atomic_calls(&source);
    assert_eq!(
        calls.len(),
        1,
        "exactly one atomic is expected in the mHC kernel - the hc_sync rendezvous counter - found: {calls:?}"
    );
    for call in &calls {
        assert!(
            call.contains("hc_sync"),
            "the only atomic in the mHC kernel must be the rendezvous counter on hc_sync, found: {call}"
        );
        assert!(
            !call.contains("hc_stats"),
            "an atomic on the statistics buffer makes the reduction order-dependent again: {call}"
        );
    }
}

#[test]
fn the_statistics_are_published_per_tile_and_combined_in_fixed_tile_order() {
    let source = squash(&code());
    let expected = [
        // The slot is fixed by blockIdx, not chosen by arrival order.
        "hc_stats[(tile + 1u) * SPARK_M4_MHC_STATS + p] = s_stats[p];",
        // The counter only identifies the last tile; it does not order the additions.
        "const uint32_t done = atomicAdd(&hc_sync[0], 1u) + 1u;",
        "if (done == (uint32_t)SPARK_M4_MHC_TILES) {",
        // The combine walks the tiles in ascending order with a compile-time bound...
        "for (uint32_t b = 0u; b < (uint32_t)SPARK_M4_MHC_TILES; ++b) {",
        // ...in binary64, so the four binary32 partials are summed exactly and rounded once...
        "sum += (double)hc_stats[(b + 1u) * SPARK_M4_MHC_STATS + p];",
        "total[p] = (float)sum;",
        // ...and the published output is the combined vector, not a running accumulator.
        "hc_stats[p] = total[p];",
    ];
    for needle in expected {
        assert!(
            source.contains(needle),
            "the deterministic reduction is missing this fixed text: {needle}"
        );
    }
    assert!(
        !source.contains("atomicAdd(&hc_stats"),
        "an atomicAdd into hc_stats is the defect this test exists to catch"
    );
}

#[test]
fn the_grid_statistics_buffer_holds_the_output_and_one_partial_per_tile() {
    let table = consts();
    let stats = table.value("SPARK_M4_MHC_STATS");
    let tiles = table.value("SPARK_M4_MHC_TILES");
    let grid = table.value("SPARK_M4_MHC_GRID_STATS");
    assert_eq!(
        stats, 26,
        "24 predictor dots plus the block RMS and the coefficient RMS statistics"
    );
    assert_eq!(
        tiles, 4,
        "one block per 1280-element tile of the 5120 hidden elements"
    );
    assert_eq!(
        grid,
        stats * (tiles + 1),
        "hc_stats holds the final vector plus one partial vector per tile"
    );
    assert_eq!(grid, 130, "the launcher allocates 130 floats, not 26");
}
