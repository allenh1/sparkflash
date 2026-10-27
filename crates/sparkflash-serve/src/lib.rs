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

//! Model-agnostic serving lifecycle.
//!
//! This crate sits between the HTTP surface in sparkflash-api and a model backend. It owns the
//! request queue, the engine thread, streaming replies, stop-string handling, sample logprobs and
//! the request telemetry envelope. A model pack supplies the arithmetic through
//! sparkflash_model::ModelBackend, and a prompt codec supplies tokenization.
//!
//! The first adapter is the generic host driver. The V4.1 reference driver still lives in the CLI
//! until the next move; it can migrate onto this crate without changing the HTTP contract.

pub mod host;

pub use host::{HostDriver, HostOptions};
