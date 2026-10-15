//! Milestone 5's decisions, exercised end to end through the crate's public API.
//!
//! Every test here drives the shapes a caller sees - a prefill plan, an admission, a DSpark
//! block, a step boundary - instead of reaching into private fields, so the decisions stay
//! pinned even if the internals move. Nothing needs weights, a GPU or a cluster.

use sparkflash_scheduler::RequestId;
use sparkflash_scheduler::bypass::{AdmissionKind, PrefillTimeline, admit};
use sparkflash_scheduler::config::{
    CED_SPLIT_LAYER, DSPARK_BLOCK_SIZE, DSPARK_CONFIDENCE_CAP, DSPARK_LAYERS,
    DSPARK_NOISE_TOKEN_ID, INDEXER_SCRATCHPAD_BYTES, INDEXER_TILE_ROWS, L2_PREFETCH_WINDOW_BYTES,
    LPDDR5X_BYTES_PER_US, PREFILL_BYPASS_MAX_TOKENS, PREFILL_CHUNK_CHOICES,
    PREFILL_DECODE_INTERVAL, PREFILL_REFERENCE_PROMPT_TOKENS, PREFILL_REFERENCE_SECONDS,
    PREFILL_TP_RANKS, SchedulerConfig,
};
use sparkflash_scheduler::dspark::{DsparkProposal, ExpertSource};
use sparkflash_scheduler::engine::{StepBoundaryScheduler, StepPhase};
use sparkflash_scheduler::error::SchedulerError;
use sparkflash_scheduler::indexer::IndexerPlan;
use sparkflash_scheduler::philox::SyncFreeDrafter;
use sparkflash_scheduler::prefill::{
    CedPrefillPlan, CollectiveBackend, L2PrefetchPlan, SeqParallelPlan,
};
use sparkflash_scheduler::trace_cli::{self, TraceCliPlan};
use sparkflash_trace::TraceScope;

fn config() -> SchedulerConfig {
    SchedulerConfig::default()
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|value| (*value).to_owned()).collect()
}

/// The bounded indexer covers every row once: ranks and tiles both partition the same range.
#[test]
fn the_chunked_prefill_plan_covers_every_rank_and_tile() {
    let config = config();
    let plan = IndexerPlan::plan(&config, PREFILL_REFERENCE_PROMPT_TOKENS).expect("plan");
    assert!(
        plan.fits(),
        "the bounded indexer has to fit the static scratchpad"
    );
    assert!(plan.tile_count() > 0);
    assert_eq!(plan.tile_count(), plan.tiles_per_rank * PREFILL_TP_RANKS);
    assert_eq!(plan.gathers.len(), plan.tiles_per_rank);
    assert_eq!(plan.trace_scope(), TraceScope::CSA2_INDEXER);

    let mut rank_end = 0;
    for rank in 0..PREFILL_TP_RANKS {
        let range = plan.partition(rank).expect("rank").range();
        assert_eq!(
            range.start,
            rank_end,
            "rank {rank} starts where rank {} ended",
            rank - 1
        );
        assert!(range.end > range.start, "rank {rank} owns rows");
        rank_end = range.end;
        // Tile ranges are rank-local, so they must cover exactly this rank's slice, once.
        let mut tile_end = 0;
        for tile in 0..plan.tiles_per_rank {
            let tile_range = plan.tile_range(tile).expect("tile");
            assert_eq!(tile_range.start, tile_end, "tile {tile} is contiguous");
            tile_end = tile_range.end;
        }
        assert_eq!(
            tile_end,
            range.end - range.start,
            "the rank-local tiles cover rank {rank}'s slice once"
        );
        assert!(
            plan.tile_range(plan.tiles_per_rank).is_err(),
            "tiles are rank-local"
        );
    }
    assert!(
        plan.partition(PREFILL_TP_RANKS).is_err(),
        "there are four ranks"
    );
}

/// The CED path bypasses the decoder's upper layers and says how much it saved.
#[test]
fn the_ced_prefill_bypasses_the_decoder_layers() {
    let plan = CedPrefillPlan::plan(PREFILL_REFERENCE_PROMPT_TOKENS).expect("plan");
    assert!(plan.bypasses_the_decoder());
    assert!(plan.saved_layer_tokens() > 0);
    assert!(
        plan.saved_fraction() > 0.0 && plan.saved_fraction() <= 1.0,
        "a saved fraction is a fraction of the decoder"
    );
    assert_ne!(plan.trace_scope(), TraceScope::NONE);
    assert_eq!(CED_SPLIT_LAYER, 20);
}

/// The sequence-parallel split routes a payload to the transport its size calls for.
#[test]
fn the_sequence_parallel_split_picks_a_transport_by_payload() {
    let config = config();
    let small = SeqParallelPlan::plan(&config, 1024, 264).expect("plan");
    assert_eq!(small.trace_scope(), TraceScope::ROCENANTE_RDMA);
    let large = SeqParallelPlan::plan(&config, 4096, 4096).expect("plan");
    assert_eq!(large.trace_scope(), TraceScope::NCCL_COLLECTIVE);
}

/// The L2 prefetch plan never claims more window than the wait window can fill, and NCCL
/// carries its own staging so there is nothing to prefetch.
#[test]
fn the_l2_prefetch_plan_respects_the_wait_window() {
    let config = config();
    let long = L2PrefetchPlan::plan(
        &config,
        CollectiveBackend::RocEnante,
        100.0,
        LPDDR5X_BYTES_PER_US,
    )
    .expect("plan")
    .expect("a plan for a one-shot transport");
    assert!(long.saturates_the_window());
    let short = L2PrefetchPlan::plan(
        &config,
        CollectiveBackend::RocEnante,
        1.0,
        LPDDR5X_BYTES_PER_US,
    )
    .expect("plan")
    .expect("a plan");
    assert!(
        !short.saturates_the_window(),
        "273 KB cannot fill a 6 MiB window"
    );
    assert!(
        L2PrefetchPlan::plan(
            &config,
            CollectiveBackend::Nccl,
            100.0,
            LPDDR5X_BYTES_PER_US
        )
        .expect("plan")
        .is_none(),
        "NCCL stages the payload itself"
    );
}

/// A short request bypasses the cold prefill; a long one waits for it.
#[test]
fn the_short_request_bypasses_the_background_prefill() {
    let config = config();
    let timeline =
        PrefillTimeline::new(PREFILL_REFERENCE_PROMPT_TOKENS, &config).expect("timeline");
    let short = admit(
        RequestId(1),
        PREFILL_BYPASS_MAX_TOKENS,
        0,
        &timeline,
        &config,
    )
    .expect("admitted");
    assert_eq!(short.kind, AdmissionKind::Bypassed);
    assert_eq!(short.admitted_step, 1, "the next decode window serves it");
    let long = admit(
        RequestId(2),
        PREFILL_REFERENCE_PROMPT_TOKENS,
        0,
        &timeline,
        &config,
    )
    .expect("admitted");
    assert_eq!(long.kind, AdmissionKind::QueuedBehindPrefill);
    assert!(long.admitted_step >= timeline.last_chunk_step());
}

/// The DSpark block: sigmoid before the survival product, the cap decides the prefix, the mask
/// is its complement, and the draft needs no communication.
#[test]
fn the_dspark_block_decides_the_prefix_the_mask_and_the_draft() {
    let config = config();
    let confident = DsparkProposal::from_logits(&config, &[4.0; DSPARK_BLOCK_SIZE]).expect("block");
    assert_eq!(confident.verified_prefix(), DSPARK_BLOCK_SIZE);
    let mask = confident.router_mask().expect("mask");
    assert_eq!(mask.live_positions(), DSPARK_BLOCK_SIZE);
    assert_eq!(mask.masked_positions(), 0);
    assert!(mask.routes_fresh_experts(0), "the anchor is always fresh");

    let doomed = DsparkProposal::from_logits(&config, &[-6.0; DSPARK_BLOCK_SIZE]).expect("block");
    assert_eq!(doomed.verified_prefix(), 0, "sigmoid(-6) is below the cap");
    assert_eq!(doomed.verify_rows(), 1, "only the anchor is forwarded");
    let mask = doomed.router_mask().expect("mask");
    assert_eq!(mask.masked_positions(), DSPARK_BLOCK_SIZE);
    let fresh: Vec<u32> = (900..906).collect();
    let anchor: Vec<u32> = (100..106).collect();
    for row in 1..mask.rows() {
        assert_eq!(mask.source(row).expect("row"), ExpertSource::Anchor);
        assert_eq!(
            mask.route(row, &fresh, &anchor).expect("row"),
            anchor,
            "a doomed row cannot select fresh experts"
        );
    }

    let drafter = SyncFreeDrafter::from_config(&config, 128_000).expect("drafter");
    let first = doomed
        .draft_tokens(&drafter, 45, RequestId(3))
        .expect("draft");
    let second = doomed
        .draft_tokens(&drafter, 45, RequestId(3))
        .expect("draft");
    assert_eq!(
        first, second,
        "every rank drafts the same tokens without a broadcast"
    );
    assert_eq!(first.len(), DSPARK_BLOCK_SIZE);
}

/// D9 at the engine: the trace id rides on the request and is re-bound at every step boundary.
#[test]
fn the_engine_rebinds_the_trace_id_at_every_step_boundary() {
    let config = config();
    let mut engine =
        StepBoundaryScheduler::new(&config, PREFILL_REFERENCE_PROMPT_TOKENS).expect("engine");
    let short = engine
        .submit("req-900", PREFILL_BYPASS_MAX_TOKENS)
        .expect("admitted");
    let long = engine
        .submit("req-901", PREFILL_REFERENCE_PROMPT_TOKENS)
        .expect("admitted");
    assert_eq!(engine.step(), 0);
    for step in 0..4u64 {
        let plan = engine.plan_step().expect("plan");
        assert_eq!(plan.step, step);
        assert!(matches!(
            plan.phase,
            StepPhase::PrefillChunk { .. } | StepPhase::InterChunkDecode
        ));
        assert_eq!(plan.scheduled, vec![short, long]);
        for binding in &plan.bindings {
            assert_eq!(binding.epoch, step);
        }
        assert_eq!(
            plan.binding(short).expect("bound").trace_request_id,
            "req-900"
        );
        assert_eq!(
            plan.binding(long).expect("bound").trace_request_id,
            "req-901"
        );
    }
    engine.finish(short).expect("known");
    let plan = engine.plan_step().expect("plan");
    assert!(
        plan.binding(short).is_none(),
        "a finished request is not re-bound"
    );
    let error = engine
        .submit_at("req-902", 128, 0)
        .expect_err("a past arrival");
    assert!(matches!(error, SchedulerError::StepOutOfOrder { .. }));
}

/// The CLI surface the milestone asks for: every scope, 45 steps.
#[test]
fn the_trace_cli_plans_the_45_step_capture() {
    let plan = trace_cli::parse(args(&["--scopes", "all", "--steps", "45"])).expect("plan");
    assert_eq!(plan, TraceCliPlan::gate());
    assert!(plan.captures_every_scope());
    assert_eq!(plan.steps, 45);
    assert_eq!(plan.scope_names().len(), TraceScope::COUNT);
    let text = plan.render();
    assert!(text.contains("45 steps"));
    assert!(text.contains("0xFFFF"));
    assert!(trace_cli::parse(args(&["--steps", "0"])).is_err());
}

/// Every spec number the milestone is graded on, in one place.
#[test]
fn the_spec_numbers_are_the_contract() {
    assert_eq!(PREFILL_CHUNK_CHOICES, [4096, 8192]);
    assert_eq!(PREFILL_TP_RANKS, 4);
    assert_eq!(INDEXER_TILE_ROWS, 256);
    assert_eq!(INDEXER_SCRATCHPAD_BYTES, 2 * 1024 * 1024 * 1024);
    assert_eq!(PREFILL_BYPASS_MAX_TOKENS, 256);
    assert_eq!(PREFILL_DECODE_INTERVAL, 2);
    assert_eq!(DSPARK_BLOCK_SIZE, 5);
    assert!((DSPARK_CONFIDENCE_CAP - 0.10).abs() < 1e-15);
    assert_eq!(DSPARK_NOISE_TOKEN_ID, 128_799);
    assert_eq!(DSPARK_LAYERS, [37, 38, 39]);
    assert_eq!(CED_SPLIT_LAYER, 20);
    assert_eq!(L2_PREFETCH_WINDOW_BYTES, 6 * 1024 * 1024);
    assert_eq!(LPDDR5X_BYTES_PER_US, 273_000);
    assert!((PREFILL_REFERENCE_SECONDS - 78.0).abs() < 1e-12);
    assert_eq!(PREFILL_REFERENCE_PROMPT_TOKENS, 400_000);
}
