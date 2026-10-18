//! The released per-sublayer row order and the released predictor on the consumers that are easy to
//! get wrong.
//!
//! A source pin is a shape check, not a numeric one, and says so: it exists because these are
//! index-order mistakes that the device-side numbers would only reveal on a machine with a free CUDA
//! context, and the cheap half must not leave them to that.

use sparkflash_kernels::c_decl::strip_comments;
use sparkflash_kernels::m4_abi::MEGA_MHC;

/// Collapse every run of whitespace to one space, so formatting cannot decide whether the
/// mechanism is present.
fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn the_kernel_publishes_each_coefficient_group_from_its_own_statistics_rows() {
    let code = squash(&strip_comments(MEGA_MHC));
    // The pre group is the first n rows after the comb block (n*n .. n*n+n) and the post group the
    // next n (n*n+n .. n*n+2n). Both are read through the same coefficient rstd, so the two
    // multiplication lines are the assertion.
    assert!(
        code.contains("total[n * n + j] * coefficient_rstd"),
        "the A (pre) rows must be the statistics rows n*n + j"
    );
    assert!(
        code.contains("hc_base[n * n + j]"),
        "the A (pre) bias rows must be n*n + j"
    );
    assert!(
        code.contains("total[n * n + n + j] * coefficient_rstd"),
        "the C (post) rows must be the statistics rows n*n + n + j, immediately after the pre group"
    );
    assert!(
        code.contains("hc_base[n * n + n + j]"),
        "the C (post) bias rows must be n*n + n + j"
    );
    assert!(
        !code.contains("2u * n * n") && !code.contains("2 * n * n"),
        "2*n*n is not a group row index for hc_mult = 4: the stats buffer holds 24 predictor statistics, so n*n + 2n = 24 is the end and 2*n*n = 32 is past it"
    );
    assert!(
        code.contains("hc_coeffs[n * n + n + destination] = output[destination]"),
        "C must land in the output's post group, hc_coeffs[n*n+n .. n*n+2n]"
    );
    assert!(
        code.contains("hc_coeffs[destination * n + source] = mixing[source][destination]"),
        "B must be published as [destination][source] from the released [source][destination]"
    );
}

#[test]
fn the_harness_gives_the_reference_checkpoint_order_and_the_kernel_the_kernel_order() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut harness = String::new();
    let mut modules: Vec<std::path::PathBuf> =
        std::fs::read_dir(root.join("crates/sparkflash-kernels/examples/m7_layer0"))
            .expect("the split M7 harness directory must be readable from the crate")
            .map(|entry| entry.expect("a harness module must be readable").path())
            .collect();
    modules.sort();
    for module in modules {
        harness.push_str(
            &std::fs::read_to_string(&module).expect("an M7 harness module must be readable"),
        );
    }
    // Every reference call site gets the checkpoint-order bias. The reference's h_of_with_scale
    // applies the bias in the released [pre, post, comb] group order after the released [24, n*d]
    // projection; handing it the kernel-row-order bias would apply the row permutation twice.
    let mut at = 0usize;
    let mut calls = 0usize;
    while let Some(offset) = harness[at..].find("mega_mhc_layer_with_scale(") {
        let start = at + offset;
        let window = &harness[start..(start + 700).min(harness.len())];
        assert!(
            window.contains("hc_base,"),
            "a mega_mhc_layer_with_scale call does not receive an hc_base (checkpoint order): {window}"
        );
        assert!(
            !window.contains("&hc_base_kernel,") && !window.contains("&mhc.hc_base_kernel"),
            "a mega_mhc_layer_with_scale call receives the kernel-order bias as an argument, so the reference would apply the row permutation twice: {window}"
        );
        calls += 1;
        at = start + "mega_mhc_layer_with_scale(".len();
    }
    // Stage 0's one call site loops over both sublayers (attn, ffn) and stages 1, 2 and 3 have one
    // call site each, so the source has four.
    assert_eq!(
        calls, 4,
        "the reference call-site count moved (stage 0 runs both sublayers through one site); re-derive the count"
    );
    // The kernel-order bias is still produced exactly once from the checkpoint order.
    assert_eq!(
        harness
            .matches("hc_base_kernel: bias_into_kernel_rows(&hc_base)")
            .count(),
        1,
        "the kernel-order bias must be derived from the checkpoint-order hc_base exactly once"
    );
    // The released predictor is [24, n*d] full rows in the release's row order; the kernel's row
    // order is a permutation of those rows, never an average over the four copies (that was the
    // retired fold, and B0 removed it from both sides).
    assert!(
        harness.contains("released_predictor_into_kernel_rows(&raw_fn, &mut predictor)"),
        "the harness must permute the released [24, n*d] rows into kernel order without averaging"
    );
    assert!(
        !harness.contains("fold_predictor_into_kernel_rows("),
        "no pre-B0 predictor fold call may survive in the harness (only comments may name it)"
    );
    // Only stage 1 and stage 4 launch mega_mhc, and both must upload the kernel-order bias.
    assert_eq!(
        harness
            .matches("cuda.copy_htod(&base_d, &f32_to_bytes(&hc_base_kernel))")
            .count(),
        2,
        "only stage 1 and stage 4 launch mega_mhc, and both must upload the kernel-order bias"
    );
}
