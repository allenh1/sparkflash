// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The bridge from Inkling's typed configuration to the shared architecture.

use sparkflash_model::{Architecture, ConfigError};

use crate::config::Inkling2Config;

/// The shared architecture of an already-parsed configuration.
#[must_use]
pub fn architecture_from_config(config: &Inkling2Config) -> &Architecture {
    config.architecture()
}

/// Parse a released Inkling document through the shared walk.
///
/// # Errors
/// As [Architecture::from_json].
pub fn architecture_from_json(text: &str) -> Result<Architecture, ConfigError> {
    Architecture::from_json(text)
}
