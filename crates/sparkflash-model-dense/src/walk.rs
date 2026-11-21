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

//! The dense arm as a host of the shared walk.
//!
//! The runtime drives it: the arm's serving entry opens a [`sparkflash_runtime::Runtime`] over
//! [`DenseWalk`], the runtime's [`PlanWalk`] over this host, and the runtime owns the prompt
//! parts, history, reuse, rewind, lane pool, ledger, the layer loop and the head's
//! record-then-choose. This file supplies only what needs the device: the embed, each branch's
//! frame (the norm before the part; the group's sum of its partial and the residual add after),
//! the head's logits and its device pick, and a sequence's lane memory.
//!
//! Every pass takes the arm's [`WidePolicy::DENSE`]: a prompt chunk takes the wide tile, and a
//! token step takes the tile once it batches more than the row GEMV's eight rows (below that the
//! policy still picks the row GEMV, so a one-to-eight-row step is the kernel it always was).
//!
//! The layout and limits are the arm's: attention layers are positional BF16 key/value caches,
//! Gated DeltaNet layers are recurrent, the context is the served context, and a pass is the
//! chunk [super::chunk_rows] takes from [sparkflash_runtime::ChunkPolicy] (256 rows by default).

use std::sync::Arc;

use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::backend::BackendError;
use sparkflash_model::ledger::State;
use sparkflash_ops::calls::{self, RmsNorm};
use sparkflash_ops::projection::WidePolicy;
use sparkflash_runtime::{
    Absent, ComputeDeclaration, Limits, PassKind, PassMask, PlanBranch, PlanHost, PlanLane,
    PlanPass, PlanSeq, PlanWalk, Segments, StateKind, WalkCapabilities,
};

use super::{
    CHUNK_ROWS, DEVICE, Dense, DenseLane, PassCx, Scratch, Shared, TOKEN_DECODE, floats,
    head as dense_head, head_choices as dense_head_choices,
};

/// The dense arm, exposed to the shared runtime as a Walk: the runtime's walk over [Shared].
pub type DenseWalk = PlanWalk<Shared>;

/// The wide policy a pass shape takes: the arm's policy for every pass.
///
/// [`WidePolicy::DENSE`] keeps the row GEMV at eight rows or fewer, so a one-row or four-lane
/// decode step still runs the kernel its bits were recorded under; a step that batches nine or
/// more rows takes the shared wide tile, as the policy says. Passing `None` here would hide the
/// tile from exactly those steps.
pub(crate) fn policy_for_kind(_kind: PassKind, policy: Option<WidePolicy>) -> Option<WidePolicy> {
    policy
}

/// The dense walk for `shared`'s weights and `capacity` lanes, for the tick engine's thread.
///
/// # Errors
/// A device allocation the walk's scratch cannot make.
pub(crate) fn walk(shared: Arc<Shared>, capacity: usize) -> Result<DenseWalk, BackendError> {
    PlanWalk::new(shared, capacity)
}

impl PlanLane for DenseLane {
    const DEVICE: &'static str = DEVICE;

    fn save(&mut self, at: usize) -> Result<(), BackendError> {
        let live = self.linear_regions();
        if let Some(checkpoint) = self.checkpoint.as_mut() {
            checkpoint
                .save(&self.shared.cuda, None, &live, at)
                .map_err(|error| BackendError::new(DEVICE, format!("the rewind point: {error}")))?;
        }
        Ok(())
    }

    fn restore(&mut self, to: usize) -> Result<(), BackendError> {
        if let Some(checkpoint) = &self.checkpoint {
            let live = self.linear_regions();
            checkpoint
                .restore(&self.shared.cuda, None, &live, to)
                .map_err(|error| BackendError::new(DEVICE, format!("the rewind point: {error}")))?;
        }
        Ok(())
    }

    fn clear(&mut self) -> Result<(), BackendError> {
        self.clear_linear()
    }

    fn room(&self, positions: usize, reset: bool) -> bool {
        self.memory.room(positions, reset)
    }
}

impl PlanHost for Shared {
    type Parts = Dense;
    type Frame = Deviceptr;
    type Lane = DenseLane;
    type Scratch = Scratch;
    type Mode = Option<WidePolicy>;
    type Drafter = Absent<&'static str>;

    const DEVICE: &'static str = DEVICE;
    const CAPABILITIES: WalkCapabilities = WalkCapabilities {
        passes: PassMask::ALL,
        segments: Segments::Many,
        state: StateKind::Both,
        // The arm's own four rows, word for word the declaration in
        // `sparkflash_model::ARM_LEDGERS`; a test below holds them equal. The token row is the
        // same constant the profile check that refuses `decode.tensor_cores = true` reads.
        compute: ComputeDeclaration {
            prompt: State::Native("prompt projections on the shared wide tile"),
            token: TOKEN_DECODE,
            stored: State::Native("BF16, FP8 E4M3 and NVFP4 parts run in their stored formats"),
            rewind: State::Native("a prompt that goes back to the last prompt's end"),
        },
    };

    fn layers(&self) -> usize {
        self.layers.len()
    }

    fn branches(&self, layer: usize) -> &[PlanBranch<Dense, Deviceptr>] {
        &self.layers[layer].branches
    }

    fn vocab(&self) -> usize {
        self.vocab
    }

    fn pass_rows(&self) -> usize {
        CHUNK_ROWS
    }

    fn limits(&self) -> Limits {
        Limits::new(self.chunk, self.context)
    }

    fn scratch(this: &Arc<Self>, rows: usize) -> Result<Scratch, BackendError> {
        Scratch::new(Arc::clone(this), rows).map_err(|error| BackendError::new(DEVICE, error))
    }

    fn lane(this: &Arc<Self>) -> Result<DenseLane, BackendError> {
        DenseLane::open(Arc::clone(this)).map_err(|error| BackendError::new(DEVICE, error))
    }

    fn drafter(_: &Arc<Self>) -> Result<Self::Drafter, BackendError> {
        Ok(Absent("dense has no draft head"))
    }

    fn mode(&self, pass: PlanPass, _rows: usize) -> Option<WidePolicy> {
        policy_for_kind(pass.kind, self.policy)
    }

    fn back(&self, lane: &mut DenseLane, reach: usize) -> Result<(), BackendError> {
        lane.memory
            .back(&self.cuda, reach)
            .map_err(|error| BackendError::new(DEVICE, format!("the lane memory: {error}")))
    }

    fn embed(&self, scratch: &Scratch, ids: &[u32]) -> Result<(), BackendError> {
        let b = &scratch.buffers;
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        self.cuda
            .copy_htod(&b.tokens, &bytes)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        self.call(
            self.kernels.embed,
            calls::embed_rows(
                self.embed.data().0,
                b.x.0,
                b.tokens.0,
                (self.hidden, self.hidden),
                ids.len(),
            ),
        )
    }

    fn enter(&self, scratch: &Scratch, norm: &Deviceptr, rows: usize) -> Result<(), BackendError> {
        let b = &scratch.buffers;
        self.norm(
            RmsNorm::new(b.x.0, norm.0, b.normed.0, self.hidden, self.eps),
            rows,
        )
    }

    fn pass<'a>(
        &'a self,
        scratch: &'a Scratch,
        segments: &'a [(&'a mut PlanSeq<DenseLane>, &'a [u32])],
        layer: usize,
        slot: usize,
        rows: usize,
        mode: Option<WidePolicy>,
    ) -> PassCx<'a> {
        PassCx {
            shared: self,
            b: &scratch.buffers,
            segments,
            index: layer,
            slot,
            rows,
            policy: mode,
        }
    }

    fn leave(
        &self,
        scratch: &Scratch,
        _norm: &Deviceptr,
        _next: Option<&Deviceptr>,
        summed: bool,
        rows: usize,
    ) -> Result<(), BackendError> {
        let b = &scratch.buffers;
        if summed {
            self.sum(b.projected, rows)?;
        }
        self.call(
            self.kernels.add,
            calls::add(b.x.0, b.projected.0, rows * self.hidden),
        )
    }

    fn residual(&self, scratch: &Scratch, rows: usize) -> Result<Vec<f32>, BackendError> {
        let mut bytes = vec![0u8; rows * self.hidden * 4];
        self.cuda
            .copy_dtoh(&mut bytes, &scratch.buffers.x)
            .map_err(|error| BackendError::new(DEVICE, error))?;
        Ok(floats(&bytes))
    }

    fn logits(
        &self,
        scratch: &Scratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<f32>, BackendError> {
        dense_head(self, &scratch.buffers, first, count)
    }

    fn choices(
        &self,
        scratch: &Scratch,
        first: usize,
        count: usize,
    ) -> Result<Vec<u32>, BackendError> {
        dense_head_choices(self, &scratch.buffers, first, count)
    }
}

#[cfg(test)]
mod tests {
    use sparkflash_runtime::testing::ArmProfile;
    use sparkflash_runtime::{ArmFixture, Fixture, FormatRule, Runtime};

    use sparkflash_model::backend::Sampling;
    use sparkflash_runtime::{Region, StateLayout, Walk};

    use super::*;

    /// The workspace root, without the `..` components a path from the crate directory would
    /// print.
    fn dense_workspace_root() -> std::path::PathBuf {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        std::fs::canonicalize(&root).unwrap_or(root)
    }

    /// The directory the dense conformance checkpoint lives in when
    /// `SPARKFLASH_CONFORMANCE_CHECKPOINT` is unset: the one `scripts/onboard-gate.sh dense`
    /// writes, so a direct `cargo test` and the script share one cache.
    fn dense_fixture_directory() -> std::path::PathBuf {
        dense_workspace_root().join("target/conformance/dense-synthetic")
    }

    /// The tokenizer the synthetic checkpoint is written from.
    fn dense_fixture_tokenizer() -> std::path::PathBuf {
        std::env::var_os("SPARKFLASH_TOKENIZER_DIR").map_or_else(
            || dense_workspace_root().join("reference"),
            std::path::PathBuf::from,
        )
    }

    /// The exact command that writes the dense conformance checkpoint at `out`.
    ///
    /// Every fixture error names it, so a host that cannot write the checkpoint is told what to
    /// run instead of what switch it forgot.
    fn dense_fixture_command(tokenizer: &std::path::Path, out: &std::path::Path) -> String {
        format!(
            "python3 {} {} {} --layers 2 --hidden 1024 --intermediate 2816 --context 512",
            dense_workspace_root()
                .join("scripts/dense-synthetic-checkpoint.py")
                .display(),
            tokenizer.display(),
            out.display()
        )
    }

    /// Whether `path` holds the config and weights the fixture opens.
    fn dense_fixture_is_written(path: &std::path::Path) -> bool {
        path.join("config.json").is_file() && path.join("model.safetensors").is_file()
    }

    /// Writes the dense conformance checkpoint with `scripts/dense-synthetic-checkpoint.py`.
    ///
    /// The writer stages into a sibling directory and publishes it by rename, so generated
    /// checks that boot the fixture in parallel never open a half-written checkpoint. Writing it
    /// needs no device; the checks that boot it do.
    fn write_dense_fixture(path: &std::path::Path) -> Result<(), String> {
        // The generated checks run in parallel by default, so the first check through writes the
        // checkpoint and the rest wait here and find it: one Python run, not six.
        static WRITE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = WRITE.lock().unwrap_or_else(|poison| poison.into_inner());
        if dense_fixture_is_written(path) {
            return Ok(());
        }
        let tokenizer = dense_fixture_tokenizer();
        let command = dense_fixture_command(&tokenizer, path);
        if !tokenizer.join("tokenizer.json").is_file() {
            return Err(format!(
                "no tokenizer at {} (set SPARKFLASH_TOKENIZER_DIR); write the checkpoint with `{command}`",
                tokenizer.display()
            ));
        }
        let staged = path.with_file_name(format!(
            "{}-{}.tmp",
            path.file_name().unwrap_or_default().to_string_lossy(),
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&staged);
        let output = std::process::Command::new("python3")
            .arg(dense_workspace_root().join("scripts/dense-synthetic-checkpoint.py"))
            .arg(&tokenizer)
            .arg(&staged)
            .args([
                "--layers",
                "2",
                "--hidden",
                "1024",
                "--intermediate",
                "2816",
                "--context",
                "512",
            ])
            .output()
            .map_err(|error| {
                format!("python3 could not run ({error}); write the checkpoint with `{command}`")
            })?;
        if !output.status.success() {
            let _ = std::fs::remove_dir_all(&staged);
            return Err(format!(
                "`{command}` failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        match std::fs::rename(&staged, path) {
            Ok(()) => Ok(()),
            // Another check, or another test process, published the same fixture first.
            Err(_) if dense_fixture_is_written(path) => {
                let _ = std::fs::remove_dir_all(&staged);
                Ok(())
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(&staged);
                Err(format!(
                    "the checkpoint wrote to {} but could not be published at {} ({error})",
                    staged.display(),
                    path.display()
                ))
            }
        }
    }

    /// The dense fixture at `world` ranks: a [`sparkflash_runtime::local_group`] whose ranks sum
    /// over one in-process bus on the exact f32 wire, the host leg a served group's TCP sums
    /// take. Each rank opens its own share of the checkpoint. The TP check reads its tokens only,
    /// so the fixture states no byte counts.
    fn group_fixture(
        world: usize,
        path: std::path::PathBuf,
        architecture: sparkflash_model::Architecture,
        descriptor: sparkflash_model::ModelDescriptor,
        from: String,
    ) -> Result<Fixture<DenseWalk>, BackendError> {
        use sparkflash_weights::tp::{LocalBus, LocalRank, TpLayout, TpRuntime, Wire};

        let context = architecture.max_position_embeddings;
        let bus = Arc::new(LocalBus::new(world).with_wire(Wire::F32));
        let open = move |rank: usize| {
            let failed =
                |error: String| BackendError::new("fixture", format!("rank {rank}: {error}"));
            let source = sparkflash_weights::safetensors::open_checkpoint(&path)
                .map_err(|error| failed(error.to_string()))?;
            let local = LocalRank {
                rank,
                bus: Arc::clone(&bus),
            };
            let layout = TpLayout {
                rank,
                world_size: world,
            };
            let tp = TpRuntime::new(layout, Box::new(local))
                .map_err(|error| failed(error.to_string()))?;
            crate::DenseDeviceBackend::open_rank(
                descriptor.clone(),
                &architecture,
                &source,
                context,
                tp,
            )
            .map_err(failed)?
            .walk_arm(64)
        };
        let (walk, group, workers) =
            sparkflash_runtime::local_group(world, open, move || Box::new(crate::wire(context)))?;
        Ok(Fixture::new(walk, from, 0, 0, Vec::new(), Some(workers)).in_group(group))
    }

    /// The dense fixture: the checkpoint `scripts/dense-synthetic-checkpoint.py` writes. When
    /// `SPARKFLASH_CONFORMANCE_CHECKPOINT` is unset the fixture writes it into the same cache
    /// directory `scripts/onboard-gate.sh dense` uses and names the path; a named path that is
    /// not a checkpoint is a fixture error that names the exact command which writes one. A
    /// missing fixture is never a skipped check.
    pub struct DenseFixture;

    impl ArmFixture for DenseFixture {
        type Walk = DenseWalk;

        fn fixture(world: usize) -> Result<Fixture<DenseWalk>, BackendError> {
            let path = match std::env::var_os("SPARKFLASH_CONFORMANCE_CHECKPOINT") {
                Some(path) if !path.is_empty() => std::path::PathBuf::from(path),
                _ => {
                    let path = dense_fixture_directory();
                    if !dense_fixture_is_written(&path) {
                        write_dense_fixture(&path)
                            .map_err(|error| BackendError::new("fixture", error))?;
                    }
                    eprintln!(
                        "dense conformance: using the synthetic checkpoint at {} (SPARKFLASH_CONFORMANCE_CHECKPOINT is unset)",
                        path.display()
                    );
                    path
                }
            };
            let command = dense_fixture_command(&dense_fixture_tokenizer(), &path);
            if !dense_fixture_is_written(&path) {
                return Err(BackendError::new(
                    "fixture",
                    format!(
                        "the dense conformance checkpoint at {} is missing its config.json or model.safetensors; write it with `{command}`",
                        path.display()
                    ),
                ));
            }
            let architecture =
                sparkflash_model::Architecture::from_checkpoint_dir(&path).map_err(|error| {
                    BackendError::new(
                        "fixture",
                        format!(
                            "the dense conformance checkpoint at {} does not open: {error}; write it with `{command}`",
                            path.display()
                        ),
                    )
                })?;
            let source = sparkflash_weights::safetensors::open_checkpoint(&path).map_err(
                |error| {
                    BackendError::new(
                        "fixture",
                        format!(
                            "the dense conformance checkpoint at {} does not open: {error}; write it with `{command}`",
                            path.display()
                        ),
                    )
                },
            )?;
            let descriptor = sparkflash_model::ModelDescriptor::from_architecture(
                "dense-conformance",
                &architecture,
                architecture.max_position_embeddings,
                sparkflash_model::Capabilities::ALL,
            );
            let from = format!("the dense synthetic checkpoint at {}", path.display());
            if world > 1 {
                return group_fixture(world, path, architecture, descriptor, from);
            }
            let backend = crate::DenseDeviceBackend::open(
                descriptor,
                &architecture,
                &source,
                architecture.max_position_embeddings,
            )
            .map_err(|error| BackendError::new("fixture", error))?;
            let walk = backend.walk_arm(64)?;
            let bytes = backend.weight_bytes() as u64;
            Ok(Fixture::new(
                walk,
                from,
                bytes,
                bytes,
                FormatRule::classes(&architecture.quantization),
                None,
            ))
        }

        /// The dense arm serves on the tick engine, so its generated checks drive that engine.
        fn tick_engine() -> bool {
            true
        }
    }

    /// The fixture writes its own checkpoint when `SPARKFLASH_CONFORMANCE_CHECKPOINT` is unset,
    /// so the direct `cargo test -p sparkflash-model-dense --lib conformance:: -- --ignored`
    /// works without an operator remembering a fixture first. That is exactly the checklist item
    /// `docs/engine/04-onboarding.md` phase 8 removes, and it is the rule the device window
    /// tripped on: the checks panicked instead of writing the checkpoint the script knows how
    /// to write.
    ///
    /// This half is CPU-only: it writes the checkpoint and reads its architecture back without
    /// opening a device, so it runs on a host with no GPU (and skips where python3 does not).
    #[test]
    fn the_dense_fixture_writes_itself_on_the_cpu() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 is unavailable; skipping the dense fixture writer check");
            return;
        }
        let path = dense_fixture_directory();
        if !dense_fixture_is_written(&path) {
            write_dense_fixture(&path).expect("the dense synthetic checkpoint writes");
        }
        assert!(dense_fixture_is_written(&path), "{}", path.display());
        let architecture = sparkflash_model::Architecture::from_checkpoint_dir(&path)
            .expect("the written checkpoint config parses");
        assert_eq!(architecture.hidden_size, 1024);
        assert_eq!(architecture.layers.len(), 2);
    }
    #[test]
    fn every_pass_routes_to_the_dense_policy() {
        let policy = Some(WidePolicy::DENSE);
        for kind in [
            PassKind::TokenRows,
            PassKind::Verify { rows: 4 },
            PassKind::Bulk { rows: 1 },
            PassKind::Paired { chunks: 2 },
        ] {
            assert_eq!(policy_for_kind(kind, policy), policy, "{kind:?}");
        }
    }

    /// The policy's own threshold is dense's legacy input: eight rows or fewer keep the row
    /// GEMV, so a decode step at the served lane count does not move; nine rows take the wide
    /// tile. That boundary is where a bit delta can first appear.
    #[test]
    fn the_dense_policy_keeps_eight_rows_on_the_row_gemv_and_takes_the_tile_at_nine() {
        use sparkflash_ops::projection::{Route, route_bf16, route_fp8};

        let (columns, inner) = (256, 256);
        for rows in 1..=8 {
            assert_eq!(
                route_bf16(Some(WidePolicy::DENSE), columns, inner, rows),
                Route::RowGemv,
                "{rows} rows stay on the row GEMV"
            );
            assert_eq!(
                route_fp8(Some(WidePolicy::DENSE), columns, inner, rows),
                Route::RowGemv,
                "{rows} FP8 rows stay on the tiled E4M3 GEMV"
            );
        }
        assert_eq!(
            route_bf16(Some(WidePolicy::DENSE), columns, inner, 9),
            Route::Wide,
            "nine BF16 rows take dory_mma_wide"
        );
        assert_eq!(
            route_fp8(Some(WidePolicy::DENSE), columns, inner, 9),
            Route::Wide,
            "nine FP8 rows take q4p_dense_fp8"
        );
        assert_eq!(
            route_bf16(None, columns, inner, 9),
            Route::BatchedGemv,
            "without the policy the same rows stay on the row GEMV"
        );
    }

    /// The walk's Limits come from the arm's chunk rule: the default profile gets 256 rows a
    /// pass, and a profile switch the policy documents gets the smaller pass. The generated
    /// profile tests run a prompt at each.
    #[test]
    fn the_chunk_rule_keeps_256_unless_the_profile_switches_it() {
        use sparkflash_model::profile::ModelProfile;

        let default = crate::chunk_rows(&ModelProfile::default(), 32_768).expect("the default");
        assert_eq!(default, CHUNK_ROWS);
        let mut profile = ModelProfile::default();
        profile.prefill.chunk = Some(SWITCHED_CHUNK);
        let switched = crate::chunk_rows(&profile, 32_768).expect("the switch");
        assert_eq!(switched, SWITCHED_CHUNK);
    }

    /// A sampled step records the token it consumed once, so the sampler's history is the
    /// sequence's tokens: the penalties count each token once, and a rewind truncates both to
    /// the same position.
    #[test]
    #[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh dense"]
    fn a_sampled_step_records_its_token_once() {
        let fixture = DenseFixture::fixture(1).expect("the dense fixture");
        let runtime = Runtime::new(fixture.walk, 0).expect("the runtime starts");
        let mut slot = runtime.open_slot().expect("a slot");
        let sampling = Sampling {
            temperature: 0.8,
            frequency_penalty: 0.5,
            seed: 7,
            ..Sampling::default()
        };
        runtime.walk().set_sampling(slot.seq_mut(), sampling);
        let first = runtime
            .prefill_slot(&mut slot, &[1, 2, 3, 4])
            .expect("the prompt")
            .next_token
            .expect("the prompt's token");
        assert_eq!(slot.seq().sampler.history().len(), 4);
        let mut members = [(slot, first)];
        let steps = runtime.step_slots(&mut members).expect("the sampled step");
        let (mut slot, _) = members.into_iter().next().expect("the member");
        assert_eq!(slot.seq().sampler.history(), &[1, 2, 3, 4, first]);
        let second = steps[0].next_token.expect("the step's token");
        let mut members = [(slot, second)];
        runtime.step_slots(&mut members).expect("the second step");
        (slot, _) = members.into_iter().next().expect("the member");
        assert_eq!(slot.seq().sampler.history(), &[1, 2, 3, 4, first, second]);
        assert_eq!(slot.seq().sampler.history().len(), slot.seq().position());
    }

    #[test]
    fn the_layout_is_positional_attention_and_recurrent_gated_deltanet() {
        let positional = StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 8,
        }]);
        assert_eq!(positional.kind(), StateKind::Positional);
        let both = StateLayout::new(vec![
            Region::Positional {
                name: "kv",
                bytes_per_row: 8,
            },
            Region::Recurrent {
                name: "gated-deltanet",
                bytes: 64,
            },
        ]);
        assert_eq!(both.kind(), StateKind::Both);
        assert!(both.needs_checkpoint());
    }

    /// The repository ships no draft head for the shared dense arm, so the walk names `Absent`
    /// and the boot ledger prints no drafter. The day a head lands this test is the one that says
    /// the type must change with it.
    #[test]
    fn no_draft_head_ships_so_the_walk_names_absent() {
        use sparkflash_runtime::{Drafter, Mode, Strategy, ledger_for};
        const {
            assert!(
                !<<DenseWalk as Walk>::Drafter as Drafter<DenseWalk>>::PRESENT,
                "a draft head landed: give the walk a Drafter, not Absent"
            );
        }
        assert_eq!(
            ledger_for::<DenseWalk>().mode(Strategy::Drafting),
            Mode::off("no drafter configured")
        );
    }

    /// The runtime's boot ledger states the token row the profile refusal reads, so a dense boot
    /// and the `decode.tensor_cores` check cannot disagree.
    #[test]
    fn the_runtime_ledger_states_the_refusal_row() {
        use sparkflash_runtime::{Mode, Strategy, ledger_for};
        let ledger = ledger_for::<DenseWalk>();
        assert_eq!(
            ledger.mode(Strategy::TensorCoreDecode),
            Mode::from(TOKEN_DECODE)
        );
    }

    /// The pass a profile's `prefill.chunk = 64` gives the walk.
    const SWITCHED_CHUNK: usize = 64;

    /// What the arm states for the generated profile tests: [CHUNK_ROWS] rows a pass and as many
    /// members, as [DenseWalk::new] states its limits; a positional cache and the Gated DeltaNet
    /// recurrence as its state; no drafter; and the chunks its prompts run in, at the default
    /// pass and at a switched one.
    fn stated() -> ArmProfile {
        let limits = Limits::new(CHUNK_ROWS, 32_768);
        ArmProfile::new(
            "dense",
            StateLayout::new(vec![
                Region::Positional {
                    name: "kv",
                    bytes_per_row: 4,
                },
                Region::Recurrent {
                    name: "gated-deltanet",
                    bytes: 16,
                },
            ]),
            limits,
        )
        .with_code(DEVICE)
        .prompt(limits, 600, &[256, 256, 88])
        .prompt(limits, 257, &[256, 1])
        .prompt(Limits::new(SWITCHED_CHUNK, 32_768), 150, &[64, 64, 22])
    }

    sparkflash_runtime::conformance!(DenseFixture, profile: stated());
}
