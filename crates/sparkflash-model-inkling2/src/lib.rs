// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The thinkingmachines/Inkling-2-Small pack.
//!
//! Inkling is a 594B-parameter / 17B-active hybrid: 48 decoder layers, 40 sliding-window and
//! 8 full-attention, a learned relative bias instead of RoPE, four short convolutions a layer,
//! a block-attention residual stream and 384 sigmoid-routed experts. This crate owns the typed
//! configuration, the shared-architecture bridge, the tensor plan, the prompt seam and the f32
//! reference forward. The device tables live in the shared kernel catalogue.

pub mod architecture;
pub mod collective;
pub mod config;
pub mod convert;
pub mod device;
pub mod device_plan;
pub mod fixture;
pub mod forward;
pub mod group;
pub mod host;
pub mod prompt;
pub mod reference;
pub mod tensor;
pub mod tp4_run;
pub mod vision;

pub use architecture::{architecture_from_config, architecture_from_json};
pub use collective::{
    COMM_REDUCE_ENV, Exchange, NcclExchange, Solo, TcpExchange, connect_rank, nccl_requested,
};
pub use config::{
    Inkling2Config, InklingMtpConfig, InklingQuantConfig, InklingVisionConfig, MODEL_ID,
    MODEL_TYPE, WEIGHTS_ID,
};
pub use convert::{
    InklingRecords, RankFileSource, RankSlot, SLOT_CHUNK_BYTES, convert_rank, rank_slots,
    verify_rank, visit_rank,
};
pub use device::{
    InklingDeviceBackend, InklingDrafter, InklingSequence, InklingVisionTower, InklingWalk,
};
pub use device_plan::{
    BufferPlan, Collective, InklingDevicePlan, LayerSchedule, Op, Pass, Step, TapLayout, TauRule,
    VOCAB_ALIGNMENT,
};
pub use forward::{Inkling2State, Inkling2Weights};
pub use group::{GroupOptions, MasterRank, open_master, run_worker};
pub use host::Inkling2HostBackend;
pub use prompt::InklingCodec;
pub use reference::{AttentionParams, attention, softcapped_swiglu};
pub use tensor::{ExpertSplit, InklingTensorPlan, Role, Slice};
pub use tp4_run::{Tp4Args, Tp4Outcome, Tp4Run, rank_file};
pub use vision::{
    IMAGE_TOKEN_ID, InklingVision, InklingVisionCodec, MAX_MERGED_TOKENS, SpanCheck, patch_values,
    patchify, plan_cells, plan_grid,
};

use sparkflash_model::ModelDescriptor;

/// The pack's carried typed profile.
pub const PROFILE_TOML: &str = include_str!("../config/inkling2_small.toml");

/// The descriptor this pack serves.
#[must_use]
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor::inkling2_small()
}

/// The released configuration carried in this crate.
#[must_use]
pub fn carried_config() -> &'static str {
    include_str!("../config/inkling2_small.hf.json")
}

/// The released external quantization groups carried in this crate.
#[must_use]
pub fn carried_quant() -> &'static str {
    include_str!("../config/inkling2_small.hf_quant.json")
}
