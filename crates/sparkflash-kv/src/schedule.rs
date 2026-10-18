//! The CED prefill/decode schedule the Phase 2 sequence harness executes.
//!
//! D7 (docs/DECISIONS.md) and roadmap-to-serving section 5.3.2 split a prefill into two halves:
//! the encoder layers 0..=20 run over the whole prompt, and the decoder layers 21..=39 run only
//! over the last min(prompt, 128) tokens. The saving is a scheduling property, not a numerical
//! one: nothing about the weights or the arithmetic changes, and every decoder position's output
//! depends only on caches the encoder half built for the positions the decoder actually needs.
//!
//! This module turns that sentence into data the harness can assert on, so no part of the device
//! path has to re-derive it from prose:
//!
//! * [ced_schedule] returns the exact [PrefixMatchPlan] the cold prompt produces;
//! * the 40-row per-layer table names each layer's [Csa2Mode], its KV and index sources, its HSI
//!   pool relationship, and whether it writes or publishes anything;
//! * the publication order lists every kernel-visible handle in the order the harness must
//!   publish it: main KV, index-K and Top-K at each Full layer, the HSI pool after layer 20's
//!   Top-K, and a republished Top-K at each Reindex layer.
//!
//! Window rings are not publications: every layer owns one ring from construction, and the
//! harness writes one slot per token per layer.
//!
//! The field name caveat: on a cold prompt (nothing cached) `plan_request`'s
//! `decoder_swa_replay_tokens` is not a cache replay at all - it is the CED decoder extent,
//! min(prompt, 128). The pin test asserts that equality, so a future change to `plan_request`
//! cannot silently move the CED boundary.

use crate::config::{
    CANDIDATE_SOURCE_LAYER, Csa2Mode, NUM_HIDDEN_LAYERS, SWA_WINDOW_SIZE, compress_ratio,
    csa2_mode, index_source_for, kv_source_for,
};
use crate::plan::{KvCacheTelemetry, PrefixMatchPlan};

/// Last encoder layer (inclusive): layers 0..=20 run over the whole prompt.
pub const ENCODER_LAST_LAYER: usize = 20;

/// First decoder layer: layers 21..=39 run only over the last window of the prompt.
pub const DECODER_FIRST_LAYER: usize = 21;

/// One kernel-visible handle a layer publishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PublishedHandle {
    /// The compressed main KV arrays of a Full layer.
    MainKv,
    /// The indexer-K arrays of a Full layer that also owns an indexer.
    IndexerK,
    /// The selected Top-K list.
    TopK,
    /// The HSI candidate pool built by the candidate source layer.
    HsiPool,
}

/// One publication event, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Publication {
    /// The layer whose completion makes the handle current.
    pub layer: usize,
    /// The handle published at that point.
    pub handle: PublishedHandle,
}

/// One row of the per-layer schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerSchedule {
    /// Backbone layer id, 0..40.
    pub layer: usize,
    /// Static mode of the layer.
    pub mode: Csa2Mode,
    /// compress_ratios[layer]: one compressed entry per `ratio` tokens.
    pub ratio: usize,
    /// Whether the layer writes its own main-KV arrays (Full only).
    pub owns_main_kv: bool,
    /// Whether the layer runs an indexer and publishes Top-K (Full and Reindex).
    pub owns_index: bool,
    /// The layer whose main KV a reader consumes, or None when the layer owns it / has none.
    pub kv_source: Option<usize>,
    /// The layer whose Top-K a reader consumes, or None when the layer publishes its own.
    pub index_source: Option<usize>,
    /// The HSI pool this layer builds or masks with, or None.
    pub candidate_source: Option<usize>,
}

/// The CED schedule for one prompt, with the plan decision that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CedSchedule {
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Sliding-window size used for the decoder extent.
    pub window: usize,
    /// Tokens the encoder layers consume: the whole prompt.
    pub encoder_tokens: usize,
    /// Tokens the decoder layers consume: min(prompt, window).
    pub decoder_tokens: usize,
    /// The cold-prompt plan the schedule was derived from.
    pub plan: PrefixMatchPlan,
    /// One row per backbone layer, in ascending order.
    pub layers: Vec<LayerSchedule>,
    /// The handle publication order, in execution order.
    pub publications: Vec<Publication>,
}

impl CedSchedule {
    /// Layers that run over the whole prompt.
    #[must_use]
    pub fn encoder_range(&self) -> std::ops::RangeInclusive<usize> {
        0..=ENCODER_LAST_LAYER
    }

    /// Layers that run over the last window of the prompt.
    #[must_use]
    pub fn decoder_range(&self) -> std::ops::RangeInclusive<usize> {
        DECODER_FIRST_LAYER..=(NUM_HIDDEN_LAYERS - 1)
    }

    /// The schedule row for a layer.
    #[must_use]
    pub fn layer(&self, layer: usize) -> Option<&LayerSchedule> {
        self.layers.get(layer)
    }

    /// Positions older than this are outside the window and are never compared (D7): they are
    /// prompt tokens below `prompt_tokens - window`.
    #[must_use]
    pub fn exclusion_boundary(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.window)
    }
}

/// Everything that can reject a sequence schedule request.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScheduleError {
    /// A zero-token prompt has no prefill.
    #[error("a sequence schedule needs at least one prompt token")]
    EmptyPrompt,
}

/// Builds the CED schedule for a cold prompt of `prompt_tokens` tokens.
///
/// Cold means no cached prefix: the plan reports zero cache hits and an uncached suffix equal to
/// the prompt. That is the only case Phase 2's milestone 2 exercises.
///
/// # Errors
/// Returns [ScheduleError::EmptyPrompt] for a zero-token prompt.
pub fn ced_schedule(prompt_tokens: usize) -> Result<CedSchedule, ScheduleError> {
    if prompt_tokens == 0 {
        return Err(ScheduleError::EmptyPrompt);
    }
    let plan = KvCacheTelemetry::default().plan_request(prompt_tokens, 0, 0, false);
    let window = SWA_WINDOW_SIZE;
    let decoder_tokens = prompt_tokens.min(window);
    let mut layers = Vec::with_capacity(NUM_HIDDEN_LAYERS);
    let mut publications = Vec::new();
    for layer in 0..NUM_HIDDEN_LAYERS {
        let mode = csa2_mode(layer);
        let ratio = compress_ratio(layer);
        let owns_main_kv = mode.owns_main_kv();
        let owns_index = mode.scores_indices();
        let kv_source = if ratio > 0 && !owns_main_kv {
            Some(kv_source_for(layer))
        } else {
            None
        };
        let index_source = if ratio > 0 && !owns_index {
            Some(index_source_for(layer))
        } else {
            None
        };
        let candidate_source = if layer == CANDIDATE_SOURCE_LAYER || mode == Csa2Mode::Reindex {
            Some(CANDIDATE_SOURCE_LAYER)
        } else {
            None
        };
        layers.push(LayerSchedule {
            layer,
            mode,
            ratio,
            owns_main_kv,
            owns_index,
            kv_source,
            index_source,
            candidate_source,
        });
        if owns_main_kv {
            publications.push(Publication {
                layer,
                handle: PublishedHandle::MainKv,
            });
        }
        if owns_main_kv && owns_index {
            publications.push(Publication {
                layer,
                handle: PublishedHandle::IndexerK,
            });
        }
        if owns_index {
            publications.push(Publication {
                layer,
                handle: PublishedHandle::TopK,
            });
        }
        if layer == CANDIDATE_SOURCE_LAYER {
            publications.push(Publication {
                layer,
                handle: PublishedHandle::HsiPool,
            });
        }
    }
    Ok(CedSchedule {
        prompt_tokens,
        window,
        encoder_tokens: prompt_tokens,
        decoder_tokens,
        plan,
        layers,
        publications,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_store::StoreGeometry;
    use crate::plan::KvCacheTelemetry;

    /// The expected 40-row mode list, written out rather than derived, so a change to a mode
    /// assignment cannot pass by agreeing with itself.
    const EXPECTED_MODES: [Csa2Mode; 40] = [
        Csa2Mode::SwaOnly,
        Csa2Mode::SwaOnly,
        Csa2Mode::Full,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Full,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Full,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Full,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reindex,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reindex,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reindex,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reindex,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
        Csa2Mode::Reuse,
    ];

    #[test]
    fn cold_prompt_plan_is_pinned_exactly() {
        let schedule = ced_schedule(130).expect("a 130-token prompt schedules");
        assert_eq!(
            schedule.plan,
            PrefixMatchPlan {
                total_prompt_tokens: 130,
                global_kv_cached_tokens: 0,
                uma_hit_tokens: 0,
                nvme_restored_tokens: 0,
                swa_tail_hit: false,
                encoder_swa_replay_tokens: 0,
                uncached_suffix_tokens: 130,
                decoder_swa_replay_tokens: 128,
            }
        );
        assert_eq!(schedule.plan.uncached_suffix_tokens, schedule.prompt_tokens);
        assert_eq!(schedule.plan.global_kv_cached_tokens, 0);
    }

    #[test]
    fn the_plan_and_the_telemetry_agree_on_a_fresh_planner() {
        // The schedule builds its own planner; a fresh planner and a second call must agree, so
        // the schedule's plan is the plan_request contract and not a coincidence of one call.
        let schedule = ced_schedule(130).expect("schedules");
        let plan = KvCacheTelemetry::default().plan_request(130, 0, 0, false);
        assert_eq!(schedule.plan, plan);
    }

    #[test]
    fn the_decoder_extent_is_clamped_to_the_window() {
        for (prompt, decoder) in [
            (1usize, 1usize),
            (64, 64),
            (128, 128),
            (130, 128),
            (4096, 128),
        ] {
            let schedule = ced_schedule(prompt).expect("schedules");
            assert_eq!(schedule.window, 128, "the released window is 128");
            assert_eq!(schedule.encoder_tokens, prompt);
            assert_eq!(schedule.decoder_tokens, decoder, "prompt {prompt}");
            assert_eq!(
                schedule.plan.decoder_swa_replay_tokens, decoder,
                "on a cold prompt this field is the CED decoder extent"
            );
            assert_eq!(schedule.encoder_range(), 0..=20);
            assert_eq!(schedule.decoder_range(), 21..=39);
        }
    }

    #[test]
    fn a_prompt_over_the_window_crosses_the_boundary() {
        let schedule = ced_schedule(130).expect("schedules");
        assert!(schedule.prompt_tokens > schedule.window);
        assert_eq!(schedule.decoder_tokens, schedule.window);
        assert_eq!(schedule.exclusion_boundary(), 2);
        let short = ced_schedule(64).expect("schedules");
        assert!(short.prompt_tokens < short.window);
        assert_eq!(short.decoder_tokens, 64);
        assert_eq!(short.exclusion_boundary(), 0);
    }

    #[test]
    fn the_layer_mode_table_is_pinned() {
        let schedule = ced_schedule(130).expect("schedules");
        assert_eq!(schedule.layers.len(), 40);
        for (layer, expected) in EXPECTED_MODES.iter().enumerate() {
            let row = schedule.layer(layer).expect("one row per layer");
            assert_eq!(row.layer, layer);
            assert_eq!(row.mode, *expected, "layer {layer} mode");
            assert_eq!(row.ratio, compress_ratio(layer));
            assert_eq!(row.owns_main_kv, expected.owns_main_kv(), "layer {layer}");
            assert_eq!(row.owns_index, expected.scores_indices(), "layer {layer}");
            assert_eq!(row.mode, csa2_mode(layer));
        }
        // The sources are the store's own resolution rule for every layer that has compressed KV.
        assert_eq!(schedule.layer(3).expect("row").kv_source, Some(2));
        assert_eq!(schedule.layer(7).expect("row").kv_source, Some(2));
        assert_eq!(schedule.layer(9).expect("row").kv_source, Some(8));
        assert_eq!(schedule.layer(15).expect("row").kv_source, Some(14));
        assert_eq!(schedule.layer(21).expect("row").kv_source, Some(20));
        assert_eq!(schedule.layer(39).expect("row").kv_source, Some(20));
        assert_eq!(schedule.layer(24).expect("row").kv_source, Some(20));
        assert_eq!(schedule.layer(24).expect("row").index_source, None);
        assert_eq!(schedule.layer(25).expect("row").index_source, Some(24));
        assert_eq!(schedule.layer(37).expect("row").index_source, Some(36));
        assert_eq!(schedule.layer(0).expect("row").kv_source, None);
        assert_eq!(schedule.layer(1).expect("row").index_source, None);
        for row in &schedule.layers {
            assert_eq!(row.kv_source, {
                if row.ratio == 0 || row.owns_main_kv {
                    None
                } else {
                    Some(kv_source_for(row.layer))
                }
            });
            assert_eq!(row.index_source, {
                if row.ratio == 0 || row.owns_index {
                    None
                } else {
                    Some(index_source_for(row.layer))
                }
            });
        }
        // The HSI pool is built by layer 20 and masked with by the four Reindex layers.
        let pool_layers: Vec<usize> = schedule
            .layers
            .iter()
            .filter(|row| row.candidate_source == Some(CANDIDATE_SOURCE_LAYER))
            .map(|row| row.layer)
            .collect();
        assert_eq!(pool_layers, vec![20, 24, 28, 32, 36]);
    }

    #[test]
    fn the_store_resolves_the_same_sources() {
        let schedule = ced_schedule(130).expect("schedules");
        let geometry = StoreGeometry::for_max_seq_len(128).expect("a whole-page geometry");
        for row in &schedule.layers {
            // The store reports the owner of the arrays (a Full layer is its own source); the
            // schedule reports None when the layer owns or has nothing to read.
            let expected = if row.ratio == 0 {
                None
            } else {
                Some(row.kv_source.unwrap_or(row.layer))
            };
            assert_eq!(
                geometry.source_layer_for(row.layer),
                expected,
                "layer {}: the store and the schedule must resolve the same source",
                row.layer
            );
        }
        // A device store refuses a SWA-only layer's compressed handles; the schedule's None says
        // the same thing without going through the device path.
        assert_eq!(schedule.layer(0).expect("row").mode, Csa2Mode::SwaOnly);
        assert_eq!(schedule.layer(1).expect("row").mode, Csa2Mode::SwaOnly);
    }

    #[test]
    fn the_publication_order_is_pinned() {
        let schedule = ced_schedule(130).expect("schedules");
        let expected = vec![
            Publication {
                layer: 2,
                handle: PublishedHandle::MainKv,
            },
            Publication {
                layer: 2,
                handle: PublishedHandle::IndexerK,
            },
            Publication {
                layer: 2,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 8,
                handle: PublishedHandle::MainKv,
            },
            Publication {
                layer: 8,
                handle: PublishedHandle::IndexerK,
            },
            Publication {
                layer: 8,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 14,
                handle: PublishedHandle::MainKv,
            },
            Publication {
                layer: 14,
                handle: PublishedHandle::IndexerK,
            },
            Publication {
                layer: 14,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 20,
                handle: PublishedHandle::MainKv,
            },
            Publication {
                layer: 20,
                handle: PublishedHandle::IndexerK,
            },
            Publication {
                layer: 20,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 20,
                handle: PublishedHandle::HsiPool,
            },
            Publication {
                layer: 24,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 28,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 32,
                handle: PublishedHandle::TopK,
            },
            Publication {
                layer: 36,
                handle: PublishedHandle::TopK,
            },
        ];
        assert_eq!(schedule.publications, expected);
        // The pool is published before its first consumer (layer 24).
        let pool_index = schedule
            .publications
            .iter()
            .position(|publication| publication.handle == PublishedHandle::HsiPool)
            .expect("the pool is published");
        let first_pool_consumer = schedule
            .publications
            .iter()
            .position(|publication| publication.layer >= 24)
            .expect("layer 24 follows");
        assert!(pool_index < first_pool_consumer);
        // Window rings are per-layer from construction, so they are not in the list.
        assert!(
            schedule
                .publications
                .iter()
                .all(|publication| publication.handle != PublishedHandle::TopK
                    || matches!(
                        schedule.layers[publication.layer].mode,
                        Csa2Mode::Full | Csa2Mode::Reindex
                    ))
        );
    }

    #[test]
    fn an_empty_prompt_is_refused() {
        assert_eq!(ced_schedule(0), Err(ScheduleError::EmptyPrompt));
    }
}
