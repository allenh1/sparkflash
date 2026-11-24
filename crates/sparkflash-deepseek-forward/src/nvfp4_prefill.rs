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

//! The NVFP4 wrappers of the shared bulk-prefill tile.
//!
//! The body is `kernels/shared_prefill_mma_sm121.cuh`'s `SPM_W_FP4` mode; the eight symbols
//! here name the two tensor-wide conventions (Qwen folds `weight_scale_2` into the group scale
//! at the load, Inkling applies `amax / outer_scale_denominator` to the output) and the prompt
//! and few-row forms. A launch takes the packed E2M1 codes, the per-16 E4M3 group plane, the
//! convention's scalar and the f32 activation rows.

use std::path::Path;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, Module};

/// One launch's arguments, in the wrapper declaration's order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nvfp4PrefillParams {
    /// `[cols][inner / 2]` packed E2M1 codes.
    pub w: Deviceptr,
    /// `[cols][inner / 16]` E4M3 group scales.
    pub ws: Deviceptr,
    /// The convention's tensor-wide factor.
    pub wide: f32,
    /// `[tokens][inner]` f32 activation rows.
    pub a: Deviceptr,
    /// `[tokens][cols]` f32 output.
    pub out: Deviceptr,
    /// Output columns.
    pub cols: u32,
    /// Contraction width.
    pub inner: u32,
    /// Token rows.
    pub tokens: u32,
}

impl Nvfp4PrefillParams {
    /// The eight parameter words a launch passes.
    #[must_use]
    pub fn words(&self) -> [u64; 8] {
        [
            self.w.0,
            self.ws.0,
            u64::from(self.wide.to_bits()),
            self.a.0,
            self.out.0,
            u64::from(self.cols),
            u64::from(self.inner),
            u64::from(self.tokens),
        ]
    }
}

/// The eight NVFP4 wrappers, loaded from one cubin.
pub struct Nvfp4Kernels {
    prompt_qwen_gate_up: Kernel,
    prompt_qwen_down: Kernel,
    prompt_inkling_gate_up: Kernel,
    prompt_inkling_down: Kernel,
    rows_qwen_gate_up: Kernel,
    rows_qwen_down: Kernel,
    rows_inkling_gate_up: Kernel,
    rows_inkling_down: Kernel,
    _modules: Vec<Module>,
}

impl Nvfp4Kernels {
    /// Loads the wrappers from the first staged NVFP4 cubin, or `None` when none is staged.
    ///
    /// The synthetic arm stages the shape's cubin under its own name; the released Qwen geometry
    /// stages `nvfp4_prefill_sm121_qwen`. All three carry the same eight symbols because the
    /// wrapper takes its geometry as launch words.
    ///
    /// # Errors
    /// A staged cubin that cannot be read, verified or resolved, or one missing a symbol.
    pub fn load(dir: &Path, cuda: &Cuda) -> Result<Option<Self>, String> {
        let Some((_name, bytes)) = staged(dir)? else {
            return Ok(None);
        };
        let mut modules = Vec::with_capacity(8);
        let (module, prompt_qwen_gate_up) =
            cuda.load_named(&bytes, "nvfp4_prefill_qwen_gate_up_sm121")?;
        modules.push(module);
        let (module, prompt_qwen_down) =
            cuda.load_named(&bytes, "nvfp4_prefill_qwen_down_sm121")?;
        modules.push(module);
        let (module, prompt_inkling_gate_up) =
            cuda.load_named(&bytes, "nvfp4_prefill_inkling_gate_up_sm121")?;
        modules.push(module);
        let (module, prompt_inkling_down) =
            cuda.load_named(&bytes, "nvfp4_prefill_inkling_down_sm121")?;
        modules.push(module);
        let (module, rows_qwen_gate_up) =
            cuda.load_named(&bytes, "nvfp4_rows_qwen_gate_up_sm121")?;
        modules.push(module);
        let (module, rows_qwen_down) = cuda.load_named(&bytes, "nvfp4_rows_qwen_down_sm121")?;
        modules.push(module);
        let (module, rows_inkling_gate_up) =
            cuda.load_named(&bytes, "nvfp4_rows_inkling_gate_up_sm121")?;
        modules.push(module);
        let (module, rows_inkling_down) =
            cuda.load_named(&bytes, "nvfp4_rows_inkling_down_sm121")?;
        modules.push(module);
        Ok(Some(Self {
            prompt_qwen_gate_up,
            prompt_qwen_down,
            prompt_inkling_gate_up,
            prompt_inkling_down,
            rows_qwen_gate_up,
            rows_qwen_down,
            rows_inkling_gate_up,
            rows_inkling_down,
            _modules: modules,
        }))
    }

    /// The prompt tile's gate/up projection for one convention.
    #[must_use]
    pub fn prompt_gate_up(&self, qwen: bool) -> Kernel {
        if qwen {
            self.prompt_qwen_gate_up
        } else {
            self.prompt_inkling_gate_up
        }
    }

    /// The prompt tile's down projection for one convention.
    #[must_use]
    pub fn prompt_down(&self, qwen: bool) -> Kernel {
        if qwen {
            self.prompt_qwen_down
        } else {
            self.prompt_inkling_down
        }
    }

    /// The few-row form's gate/up projection for one convention.
    #[must_use]
    pub fn rows_gate_up(&self, qwen: bool) -> Kernel {
        if qwen {
            self.rows_qwen_gate_up
        } else {
            self.rows_inkling_gate_up
        }
    }

    /// The few-row form's down projection for one convention.
    #[must_use]
    pub fn rows_down(&self, qwen: bool) -> Kernel {
        if qwen {
            self.rows_qwen_down
        } else {
            self.rows_inkling_down
        }
    }
}

/// The cubin name candidates, in the order a staged directory is asked.
const CANDIDATES: [&str; 4] = [
    "nvfp4_prefill_sm121_qwen",
    "nvfp4_prefill_sm121_i512",
    "nvfp4_prefill_sm121_i256",
    "nvfp4_prefill_sm121_i128",
];

fn staged(dir: &Path) -> Result<Option<(&'static str, Vec<u8>)>, String> {
    for name in CANDIDATES {
        let path = sparkflash_device::cubins::cubin_path(dir, name);
        if !path.is_file() {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        if let Some(pin) = sparkflash_kernels::baseline::pin(name)? {
            sparkflash_device::cubins::verify_bytes(&bytes, &pin)?;
        } else if std::env::var_os("SPARKFLASH_UNPINNED_SECOND_SHAPE").is_none() {
            // An unpinned NVFP4 cubin is a second-shape or development build. A released run
            // leaves it unloaded and the format branch refuses by name if the checkpoint needs it.
            continue;
        }
        return Ok(Some((name, bytes)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_launch_words_are_the_wrapper_declaration_in_order() {
        let params = Nvfp4PrefillParams {
            w: Deviceptr(0x10),
            ws: Deviceptr(0x20),
            wide: 0.5,
            a: Deviceptr(0x30),
            out: Deviceptr(0x40),
            cols: 256,
            inner: 4096,
            tokens: 2,
        };
        assert_eq!(
            params.words(),
            [
                0x10,
                0x20,
                u64::from(0.5f32.to_bits()),
                0x30,
                0x40,
                256,
                4096,
                2
            ]
        );
    }
}
