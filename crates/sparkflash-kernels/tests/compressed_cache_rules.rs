//! The compressed cache device form against the pinned host rules.
//!
//! The quantiser oracle is sparkflash_kv::fp4::quantize_main_kv, the function
//! tp4_prefill::stage_attention calls per selected row; the selected-list oracle is the arithmetic
//! of the same function (tp4_prefill.rs:330-352), transcribed here so the twin is compared to a
//! second statement of it rather than to itself.

use sparkflash_kernels::compressed_cache_ref as cache;
use sparkflash_kv::fp4;

fn lcg(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        out.push((unit * 2.0 - 1.0) * scale);
    }
    out
}

fn rows(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    lcg(seed, count * cache::DIM, scale)
}

#[test]
fn the_quantiser_is_the_pinned_kv_quantiser_bit_for_bit() {
    let mut samples: Vec<Vec<f32>> = vec![
        vec![0.0f32; cache::DIM],
        vec![6.0f32; cache::DIM],
        vec![-6.0f32; cache::DIM],
        vec![0.0117f32; cache::DIM],
        vec![1e-8f32; cache::DIM],
        rows(1, 7, 0.01),
        rows(1, 11, 1.0),
        rows(1, 13, 6.0),
        rows(1, 17, 64.0),
    ];
    let mut mixed = rows(1, 19, 3.0);
    for (index, value) in mixed.iter_mut().enumerate() {
        if index % 3 == 0 {
            *value = -*value;
        }
    }
    samples.push(mixed);
    for (case, sample) in samples.iter().enumerate() {
        let twin = cache::quantise_row(sample).expect("twin");
        let oracle = fp4::quantize_main_kv(sample).expect("oracle");
        assert_eq!(
            twin.codes.as_slice(),
            oracle.data.as_slice(),
            "case {case}: code bytes"
        );
        assert_eq!(
            twin.scales.as_slice(),
            oracle.scales.as_slice(),
            "case {case}: scale bytes"
        );
    }
}

#[test]
fn the_encoders_are_the_pinned_kv_encoders() {
    let values = lcg(23, 4096, 8.0);
    for value in values.iter().copied() {
        assert_eq!(
            cache::e2m1_encode(value),
            fp4::e2m1_encode(value),
            "e2m1 {value}"
        );
        assert_eq!(
            cache::e4m3_encode(value),
            fp4::e4m3_encode(value),
            "e4m3 {value}"
        );
    }
    for code in 0..=0x7Eu8 {
        assert_eq!(
            cache::e4m3_decode(code).to_bits(),
            fp4::e4m3_decode(code).to_bits(),
            "code {code}"
        );
    }
    for code in 0..16u8 {
        assert_eq!(
            cache::e2m1_decode(code).to_bits(),
            fp4::e2m1_decode(code).to_bits(),
            "code {code}"
        );
    }
}

#[test]
fn the_append_writes_expanded_rows_and_moves_the_cursor() {
    let plan = cache::Plan::new(4).expect("ratio 4");
    assert_eq!(plan.capacity_rows(), cache::EXPANDED_ROWS / 4);
    let mut store = cache::Store::zeroed();
    assert!(store.zero());
    let latent = rows(2, 31, 2.0);
    assert_eq!(store.append(&plan, 0, &latent).expect("append"), 2);
    assert_eq!(store.rows_written(), 2);
    assert!(!store.zero());
    for row in 0..2usize {
        let entry =
            cache::quantise_row(&latent[row * cache::DIM..(row + 1) * cache::DIM]).expect("row");
        let (codes, scales) = store.row(row * 4).expect("expanded row");
        assert_eq!(
            codes,
            entry.codes.as_slice(),
            "row {row} codes at expanded index"
        );
        assert_eq!(
            scales,
            entry.scales.as_slice(),
            "row {row} scales at expanded index"
        );
    }
    let (codes, scales) = store.row(1).expect("untouched slot");
    assert!(codes.iter().all(|byte| *byte == 0) && scales.iter().all(|byte| *byte == 0));
}

#[test]
fn an_empty_append_touches_no_byte() {
    let plan = cache::Plan::new(4).expect("ratio 4");
    let mut store = cache::Store::zeroed();
    assert_eq!(store.append(&plan, 0, &[]).expect("empty append"), 0);
    assert_eq!(store.rows_written(), 0);
    assert!(store.zero());
}

#[test]
fn an_append_past_the_bound_is_refused_not_grown() {
    let plan = cache::Plan::new(4).expect("ratio 4");
    let mut store = cache::Store::zeroed();
    let full = rows(plan.capacity_rows(), 37, 1.0);
    assert_eq!(
        store.append(&plan, 0, &full).expect("fills the channel"),
        plan.capacity_rows()
    );
    let error = store
        .append(&plan, plan.capacity_rows(), &rows(1, 41, 1.0))
        .expect_err("past bound");
    match error {
        cache::CacheError::Refused { bound, .. } => assert_eq!(bound, plan.capacity_rows()),
        other => panic!("refused expected, got {other:?}"),
    }
    assert!(
        cache::Plan::new(0)
            .expect_err("ratio 0")
            .to_string()
            .contains("compress ratio")
    );
    assert_eq!(
        cache::Plan::new(1).expect("ratio 1").capacity_rows(),
        cache::EXPANDED_ROWS
    );
}

#[test]
fn a_rewrite_of_a_visible_row_is_refused() {
    let plan = cache::Plan::new(2).expect("ratio 2");
    let mut store = cache::Store::zeroed();
    store.append(&plan, 0, &rows(2, 43, 1.0)).expect("append");
    let error = store
        .append(&plan, 1, &rows(1, 47, 1.0))
        .expect_err("rewrite");
    assert!(
        matches!(error, cache::CacheError::Rewrite { .. }),
        "{error:?}"
    );
}

#[test]
fn the_selected_list_is_the_expanded_id_arithmetic() {
    let window_rows = 129usize;
    let ratio = 4usize;
    let ids = [129u32, 130, u32::MAX, 131];
    let mut expected: Vec<u32> = Vec::new();
    for id in ids.iter().copied() {
        if id == u32::MAX {
            continue;
        }
        let row = (id as usize) - window_rows;
        expected.push((row * ratio) as u32);
    }
    let got = cache::selected_rows(&ids, window_rows, ratio).expect("mapping");
    assert_eq!(
        got, expected,
        "the expanded store row ids, not the token positions"
    );
    assert_eq!(got, vec![0u32, 4, 8]);
    let error = cache::selected_rows(&[128], window_rows, ratio).expect_err("below the window");
    assert!(
        matches!(error, cache::CacheError::Refused { .. }),
        "{error:?}"
    );
    let error = cache::selected_rows(&[window_rows as u32 + 200], window_rows, ratio)
        .expect_err("past the store");
    assert!(
        matches!(
            error,
            cache::CacheError::Refused {
                bound: cache::EXPANDED_ROWS,
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn the_accounting_is_the_residency_number() {
    assert_eq!(cache::STORE_BYTES, 147_456);
    let plan = cache::Plan::new(4).expect("ratio 4");
    assert_eq!(plan.bytes(), cache::STORE_BYTES + 4);
    let accounting = cache::Accounting::new(4).expect("accounting");
    assert_eq!(accounting.store_bytes, 147_456);
    assert_eq!(accounting.index_key_bytes_per_position, 128 * 4);
    assert_eq!(accounting.positions, cache::MAX_POSITIONS);
    assert_eq!(
        accounting.total_bytes(),
        147_456 + 128 * 4 * cache::MAX_POSITIONS
    );
    assert_eq!(accounting.compressed_rows, 128);
}

#[test]
fn the_store_width_rule_follows_the_ratio() {
    use sparkflash_kernels::compressor_staging::{self, WeightStore};
    assert_eq!(
        compressor_staging::store_width_for_ratio(1),
        WeightStore::Bf16
    );
    for ratio in [2usize, 4, 8, 16, 128] {
        assert_eq!(
            compressor_staging::store_width_for_ratio(ratio),
            WeightStore::F32
        );
    }
    assert_ne!(WeightStore::F32.declared(), WeightStore::Bf16.declared());
    let cols = 32usize;
    let inner = cache::DIM;
    let staged =
        compressor_staging::stage_for_ratio(4, "wkv", &source_of(cols * inner), cols, inner)
            .expect("f32");
    assert_eq!(staged.store, WeightStore::F32);
    assert_eq!(staged.bytes(), cols * inner * 4);
    let staged =
        compressor_staging::stage_for_ratio(1, "wkv", &source_of(cols * inner), cols, inner)
            .expect("bf16");
    assert_eq!(staged.store, WeightStore::Bf16);
    assert_eq!(staged.bytes(), cols * inner * 2);
    let error = compressor_staging::stage_for_ratio(
        1,
        "wkv",
        &source_of(cols * cache::DIM),
        cols,
        cache::DIM + 32,
    )
    .expect_err("shape");
    assert!(error.to_string().contains("wkv"));
    let error = compressor_staging::stage_for_ratio(
        1,
        "wkv",
        &source_of(cols * cache::DIM),
        cols,
        cache::DIM,
    )
    .expect("whole tiles");
    assert_eq!(error.store, WeightStore::Bf16);
}

fn source_of(count: usize) -> Vec<f32> {
    lcg(59, count, 1.0)
}

#[test]
fn the_launch_declaration_is_pinned_and_published() {
    use sparkflash_kernels::compressed_append_launch as launch;
    assert!(
        launch::append_declaration_matches(),
        "the append declaration moved"
    );
    assert!(
        launch::select_declaration_matches(),
        "the selected-fill declaration moved"
    );
    assert_eq!(launch::ARG_WORDS, 6);
    assert_eq!(launch::SELECT_ARG_WORDS, 6);
    assert_eq!(launch::EXPANDED_ROWS, cache::EXPANDED_ROWS);
    let manifest = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/baseline/kernel_baseline.txt"
    ))
    .expect("the baseline manifest is in the crate");
    let line = manifest
        .lines()
        .find(|line| line.starts_with(&format!("kernel {} ", launch::KERNEL)))
        .expect("the manifest pins the compressed append");
    assert!(
        line.contains(&format!("cubin_bytes={}", launch::CUBIN_BYTES)),
        "{line}"
    );
    assert!(
        line.contains(&format!("cubin_sha256={}", launch::CUBIN_SHA256)),
        "{line}"
    );
    let params = launch::AppendParams {
        latent: 0x1000,
        codes: 0x2000,
        scales: 0x3000,
        first_row: 7,
        row_count: 2,
        ratio: 4,
    };
    assert_eq!(params.words(), [0x1000, 0x2000, 0x3000, 7, 2, 4]);
    let select = launch::SelectParams {
        ids: 0x4000,
        n_ids: 3,
        window_rows: 129,
        ratio: 4,
        out: 0x5000,
        out_count: 0x6000,
    };
    assert_eq!(select.words(), [0x4000, 3, 129, 4, 0x5000, 0x6000]);
}

#[test]
fn the_union_guards_quote_the_body_numbers() {
    use sparkflash_kernels::compressed_append_launch::{
        CANDIDATE_GUARD, CANDIDATE_SENTINEL, MAX_ATTENDED, RING_MAX_ATTENDED, SELECT_VALIDATION,
        SELECTED_COUNT_OWNER,
    };
    assert_eq!(MAX_ATTENDED, 640, "128 window rows + 512 selected");
    assert_eq!(
        RING_MAX_ATTENDED, 646,
        "512 + 134, the body's static_assert"
    );
    assert_eq!(RING_MAX_ATTENDED - MAX_ATTENDED, 6, "six slots to spare");
    assert_eq!(CANDIDATE_GUARD, "candidate <= position");
    let keep = sparkflash_kernels::mla_ref::MAX_SELECTED;
    let window = sparkflash_kernels::mla_ref::SWA_WINDOW as usize;
    assert_eq!(window + keep, MAX_ATTENDED);
    assert_eq!(
        CANDIDATE_SENTINEL.to_bits(),
        f32::MIN.to_bits(),
        "the exact sentinel, not an approximation"
    );
    assert_eq!(
        (CANDIDATE_SENTINEL - 5.0f32).exp(),
        0.0,
        "padding contributes zero"
    );
    assert!(SELECTED_COUNT_OWNER.contains("keep = min(topk, n)"));
    assert!(
        SELECTED_COUNT_OWNER.contains("compacts and skips NO_INDEX"),
        "present tense stated"
    );
    assert!(SELECT_VALIDATION.contains("reference-path"));
    assert!(SELECT_VALIDATION.contains("MAX_ATTENDED = 640"));
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/flash_mla_csa2_ring_sm121.cu"
    ))
    .expect("the body source is in the tree");
    assert!(source.contains("static_assert(SPARK_M4_MLA_RING_MAX_ATTENDED == 646u"));
    assert!(source.contains("candidate <= position"));
}
