//! Cross-checks for the D7 CED boundary between the schedule, the plan and the SWA replay split.

use sparkflash_kv::config::{
    CANDIDATE_SOURCE_LAYER, Csa2Mode, DECODER_FIRST_LAYER, ENCODER_LAST_LAYER, SWA_WINDOW_SIZE,
};
use sparkflash_kv::layout::{
    SWA_DECODER_LAYERS, SWA_ENCODER_LAYERS, SWA_LAYERS, SWA_SLOT_BYTES, SWA_SLOTS_PER_TAIL,
    SWA_TAIL_BYTES, SWA_TAIL_DECODER_BYTES, SWA_TAIL_ENCODER_BYTES,
};
use sparkflash_kv::schedule::{PublishedHandle, ced_schedule};
use sparkflash_kv::swa::{ALL_SWA_LAYERS, ReplayKind};
use sparkflash_kv::{KvCacheTelemetry, replay_layer_count};

#[test]
fn the_replay_split_is_the_d7_split() {
    assert_eq!(SWA_LAYERS, 40);
    assert_eq!(ENCODER_LAST_LAYER, 20);
    assert_eq!(DECODER_FIRST_LAYER, 21);
    assert_eq!(SWA_ENCODER_LAYERS, 21);
    assert_eq!(SWA_DECODER_LAYERS, 19);
    assert_eq!(SWA_ENCODER_LAYERS + SWA_DECODER_LAYERS, SWA_LAYERS);
    // D7's boundary is the candidate-source layer, not an arithmetic half.
    assert_eq!(ENCODER_LAST_LAYER, CANDIDATE_SOURCE_LAYER);
    assert_eq!(ENCODER_LAST_LAYER, 20);
    assert_eq!(replay_layer_count(ReplayKind::Encoder), 21);
    assert_eq!(replay_layer_count(ReplayKind::Decoder), 19);
    assert_eq!(ReplayKind::Encoder.layers(), 0..21);
    assert_eq!(ReplayKind::Decoder.layers(), 21..40);
    assert!(ReplayKind::Encoder.layers().contains(&20));
    assert!(!ReplayKind::Decoder.layers().contains(&20));
    assert_eq!(
        ReplayKind::Encoder.layer_mask() | ReplayKind::Decoder.layer_mask(),
        ALL_SWA_LAYERS
    );
    assert_eq!(
        ReplayKind::Encoder.layer_mask() & ReplayKind::Decoder.layer_mask(),
        0
    );
}

#[test]
fn the_old_arithmetic_half_split_would_fail_the_pins() {
    // The defect this change fixes: the split used to be SWA_LAYERS / 2 on both sides, so the
    // encoder covered 0..19, the decoder 20..39, and replay_layer_count reported 20/20. These
    // assertions are non-vacuous: if any of the three sources drifts back to that arithmetic
    // half, one of them fails.
    let old_encoder = 0..SWA_LAYERS / 2;
    let old_decoder = SWA_LAYERS / 2..SWA_LAYERS;
    assert_ne!(ReplayKind::Encoder.layers(), old_encoder);
    assert_ne!(ReplayKind::Decoder.layers(), old_decoder);
    assert_ne!(replay_layer_count(ReplayKind::Encoder), old_encoder.len());
    assert_ne!(replay_layer_count(ReplayKind::Decoder), old_decoder.len());
    assert_ne!(SWA_ENCODER_LAYERS, old_encoder.len());
    assert_ne!(SWA_DECODER_LAYERS, old_decoder.len());
    let schedule = ced_schedule(130).expect("a 130-token prompt schedules");
    // The old encoder range ended at 19 and the old decoder range started at 20; those two
    // boundaries are the ones D7 moves.
    assert_ne!(*schedule.encoder_range().end(), old_encoder.end - 1);
    assert_ne!(*schedule.decoder_range().start(), old_decoder.start);
    // The old split classified layer 20 as a decoder layer; D7 puts it in the encoder.
    assert!(ReplayKind::Encoder.layers().contains(&20));
    assert!(!old_encoder.contains(&20));
}

#[test]
fn the_schedule_the_plan_and_the_replay_split_agree() {
    let schedule = ced_schedule(130).expect("a 130-token prompt schedules");
    let encoder_start = *schedule.encoder_range().start();
    let encoder_end = *schedule.encoder_range().end();
    let decoder_start = *schedule.decoder_range().start();
    let decoder_end = *schedule.decoder_range().end();
    assert_eq!(encoder_start, 0);
    assert_eq!(encoder_end, ENCODER_LAST_LAYER);
    assert_eq!(decoder_start, DECODER_FIRST_LAYER);
    assert_eq!(decoder_end, SWA_LAYERS - 1);
    // The half-open replay ranges are exactly the schedule's inclusive ranges.
    assert_eq!(ReplayKind::Encoder.layers(), encoder_start..encoder_end + 1);
    assert_eq!(ReplayKind::Decoder.layers(), decoder_start..decoder_end + 1);
    assert_eq!(
        replay_layer_count(ReplayKind::Encoder),
        schedule.encoder_range().count()
    );
    assert_eq!(
        replay_layer_count(ReplayKind::Decoder),
        schedule.decoder_range().count()
    );
    // The plan is plan_request(130, 0, 0, false); its fields are cross-checked here so a change
    // to any of the three sources has to be made in all three.
    let plan = KvCacheTelemetry::default().plan_request(130, 0, 0, false);
    assert_eq!(schedule.plan, plan);
    assert_eq!(schedule.encoder_tokens, 130);
    assert_eq!(schedule.decoder_tokens, 128);
    assert_eq!(plan.uncached_suffix_tokens, 130);
    assert_eq!(plan.encoder_swa_replay_tokens, 0);
    // Re-pinned after the split fix: a cold 130-token prompt still plans 128 decoder tokens.
    // This field is the CED token extent min(prompt, SWA_WINDOW_SIZE), not a layer count, so
    // the 21/19 split does not move it.
    assert_eq!(plan.decoder_swa_replay_tokens, SWA_WINDOW_SIZE);
    assert_eq!(plan.decoder_swa_replay_tokens, schedule.decoder_tokens);
}

#[test]
fn layer_20_is_an_encoder_layer_and_the_full_kv_source_and_hsi_builder() {
    assert_eq!(CANDIDATE_SOURCE_LAYER, 20);
    let schedule = ced_schedule(130).expect("schedules");
    assert!(
        ReplayKind::Encoder.layers().contains(&20),
        "layer 20 must be replayed by the encoder half"
    );
    assert!(!ReplayKind::Decoder.layers().contains(&20));
    let row = schedule.layer(20).expect("one row per layer");
    assert_eq!(row.mode, Csa2Mode::Full);
    assert!(row.owns_main_kv, "layer 20 owns the Full KV it projects");
    assert!(row.owns_index);
    assert_eq!(row.candidate_source, Some(CANDIDATE_SOURCE_LAYER));
    assert!(
        schedule.publications.iter().any(|publication| {
            publication.layer == 20 && publication.handle == PublishedHandle::HsiPool
        }),
        "the HSI pool is published by layer 20"
    );
    // Layer 20 is the Full KV source for every decoder layer that reads compressed KV, which is
    // why moving it into the decoder half would make the decoder's own input unavailable.
    for layer in 21..=39 {
        let row = schedule.layer(layer).expect("one row per layer");
        assert_eq!(row.kv_source, Some(20), "decoder layer {layer}");
    }
}

#[test]
fn the_half_sizes_follow_the_21_19_split() {
    // Claimed numbers that changed with the fix: before it, each half was 20 * 128 * 528 B =
    // 1,351,680 B. After: encoder 21 * 128 * 528 B = 1,419,264 B and decoder 19 * 128 * 528 B =
    // 1,284,096 B. The full tail is unchanged at 40 * 128 * 528 B = 2,703,360 B.
    assert_eq!(SWA_SLOT_BYTES, 528);
    assert_eq!(
        SWA_TAIL_ENCODER_BYTES,
        SWA_ENCODER_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
    );
    assert_eq!(SWA_TAIL_ENCODER_BYTES, 1_419_264);
    assert_eq!(
        SWA_TAIL_DECODER_BYTES,
        SWA_DECODER_LAYERS * SWA_SLOTS_PER_TAIL * SWA_SLOT_BYTES
    );
    assert_eq!(SWA_TAIL_DECODER_BYTES, 1_284_096);
    assert_eq!(
        SWA_TAIL_ENCODER_BYTES + SWA_TAIL_DECODER_BYTES,
        SWA_TAIL_BYTES
    );
    assert_eq!(SWA_TAIL_BYTES, 2_703_360);
    assert_ne!(SWA_TAIL_ENCODER_BYTES, 1_351_680);
    assert_ne!(SWA_TAIL_DECODER_BYTES, 1_351_680);
}
