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

//! A stated block on the rows path ([crate::plan::BlockSteps::Stated]): a stack whose layout
//! states its own wiring (GLM-5.3-Flash's), each block walked on the shared role bodies at the
//! producer's types, as the host's whole-model reference walks it.

use super::rows::RowsState;
use super::*;
use crate::plan::StatedBlock;

impl DeviceLayer<'_> {
    /// Layer `layer`'s stated block over rows p0..p0+n.
    pub(super) fn run_block_stated(
        &mut self,
        _rows: &RowsState,
        layer: usize,
        block: StatedBlock,
        _p0: usize,
        _n: usize,
    ) -> Result<(), String> {
        Err(format!(
            "rows layer {layer}: the stated block {block:?} has no device walk yet"
        ))
    }
}
