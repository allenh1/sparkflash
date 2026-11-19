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

//! The GLM arm opens through the shared `EngineBuilder`.

use sparkflash_model::{ModelBackend, Target};
use sparkflash_model_glm5_next::checkpoint::write_synthetic_checkpoint;
use sparkflash_model_glm5_next::{
    Glm5NextConfig, Glm5NextDeviceBackend, architecture_from_carried,
};
use sparkflash_ops::tensor_plan::ArchitectureTensorPlan;
use sparkflash_ops::{EngineBuilder, EnginePlan, PartsDeviceRank, PlanContext};
use sparkflash_weights::{TensorSpec, WeightError, WeightSource};

struct Specs {
    names: Vec<String>,
    by_name: std::collections::HashMap<String, TensorSpec>,
}

impl Specs {
    fn new(specs: &[TensorSpec]) -> Self {
        Self {
            names: specs.iter().map(|spec| spec.name.clone()).collect(),
            by_name: specs
                .iter()
                .map(|spec| (spec.name.clone(), spec.clone()))
                .collect(),
        }
    }
}

impl WeightSource for Specs {
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.by_name
            .get(name)
            .cloned()
            .ok_or_else(|| WeightError::Missing {
                name: name.to_owned(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        Ok(vec![0; self.spec(name)?.storage_bytes() as usize])
    }
}

#[test]
fn the_released_plan_resolves_every_part_at_every_world() {
    let architecture = architecture_from_carried().expect("the carried architecture parses");
    for world in [1usize, 2, 4] {
        for rank in 0..world {
            let plan =
                EnginePlan::for_architecture(architecture.clone(), Target::Sm121, world, rank)
                    .expect("the released plan builds");
            let tensors = ArchitectureTensorPlan::served_and_mtp(&architecture, plan.tp())
                .expect("the shared tensor plan builds");
            let source = Specs::new(tensors.plan().specs());
            let shared = PartsDeviceRank::open(&plan, &source, PlanContext::SM121_CATALOGUE)
                .expect("the shared rank opens");
            shared
                .boot_check()
                .unwrap_or_else(|error| panic!("TP{world}/{rank}: {error}"));
            assert_eq!(shared.layers().len(), architecture.layers.len());
            assert!(!shared.boot_summary().is_empty());
        }
    }
}

/// The tiny fixture is below the catalogue's GLM geometry, so the boot check is right to refuse
/// it; the fixture used for a serving boot is [Glm5NextConfig::serving_fixture].
#[test]
fn the_small_fixture_is_refused_by_the_serving_boot_check() {
    let config = Glm5NextConfig::small();
    let plan = EnginePlan::for_architecture(config.architecture.clone(), Target::Sm121, 1, 0)
        .expect("the small plan builds");
    let tensors = ArchitectureTensorPlan::served_and_mtp(&config.architecture, plan.tp())
        .expect("the shared tensor plan builds");
    let source = Specs::new(tensors.plan().specs());
    let error = PartsDeviceRank::open(&plan, &source, PlanContext::SM121_CATALOGUE)
        .expect_err("the 256-wide fixture is not a serving shape");
    let text = error.to_string();
    assert!(
        text.contains("hidden 256 does not equal 4096") && text.contains("glm_prefill_sm121"),
        "the refusal names the fixture's width and the body it is short of: {text}"
    );
}

#[test]
fn a_serving_fixture_passes_the_boot_check_and_opens() {
    let config = Glm5NextConfig::serving_fixture();
    let plan = EnginePlan::for_architecture(config.architecture.clone(), Target::Sm121, 1, 0)
        .expect("the serving plan builds");
    let tensors = ArchitectureTensorPlan::served_and_mtp(&config.architecture, plan.tp())
        .expect("the shared tensor plan builds");
    let source = Specs::new(tensors.plan().specs());
    let rank = PartsDeviceRank::open(&plan, &source, PlanContext::SM121_CATALOGUE)
        .expect("the serving fixture passes the boot check");
    assert_eq!(rank.plan().family(), "glm5_next");
    rank.boot_check().expect("the serving rank has no gap");

    if sparkflash_kernels::cuda_driver::Cuda::load().is_err() {
        eprintln!("skipped: no CUDA driver");
        return;
    }
    let checkpoint = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(checkpoint.path(), &config, 7, 1).expect("the fixture writes");
    let cuda = sparkflash_kernels::cuda_driver::Cuda::load().expect("a CUDA context");
    let mut backend = Glm5NextDeviceBackend::open_single_rank_with_plan(
        std::sync::Arc::new(sparkflash_model_glm5_next::device::CudaHandle::new(cuda)),
        config.clone(),
        checkpoint.path(),
        plan.architecture().max_position_embeddings,
        &plan,
    )
    .expect("the arm opens on the serving plan");
    assert_eq!(backend.descriptor().vocab_size, config.vocab_size);
    backend
        .enable_fusion(1)
        .expect("the arm fuses onto the runtime");
    let mut session = backend.open_session().expect("a session opens");
    let first = session.prefill(&[1, 2, 3, 4]).expect("a prefill runs");
    assert_eq!(first.logits.len(), config.vocab_size);
}

#[test]
fn the_engine_builder_refuses_tp2_and_tp4_plans() {
    let architecture = architecture_from_carried().expect("the carried architecture parses");
    for world in [2usize, 4] {
        let plan = EnginePlan::for_architecture(architecture.clone(), Target::Sm121, world, 0)
            .expect("the released plan builds");
        let tensors = ArchitectureTensorPlan::served_and_mtp(&architecture, plan.tp())
            .expect("the shared tensor plan builds");
        let source = Specs::new(tensors.plan().specs());
        let error = <Glm5NextDeviceBackend as EngineBuilder>::open(
            &plan,
            &source,
            plan.architecture().max_position_embeddings,
        )
        .expect_err("the single-rank builder cannot open a group rank");
        assert!(
            error
                .to_string()
                .contains("single-rank builder opens TP1/rank0")
                && error.to_string().contains(&format!("TP{world}/rank0")),
            "TP{world}: {error}"
        );
    }
}
