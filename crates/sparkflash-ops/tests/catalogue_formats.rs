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

//! The catalogue is keyed by part, image and format, and every built body states its format.
//!
//! `kernels/instances.toml` records the stored format each entry decodes. The build turns that
//! table into the symbol's catalogue format, so a body cannot claim an arithmetic the build did
//! not record. These checks hold the two ends together: every built symbol carries its format,
//! and the precision mask it is launched under contains that format's arithmetic. A mask that
//! does not is the mechanical fill this workstream corrected (for example an MXFP4 expert body
//! listed under NVFP4 because both were 4-bit).

use sparkflash_ops::engine_plan::{precision_of, precision_of_stored};
use sparkflash_ops::{CATALOGUE, Precision, built_format};

#[test]
fn every_built_symbol_carries_the_format_its_instance_states() {
    let mut checked = 0;
    for entry in CATALOGUE {
        for symbol in entry.symbols {
            // A format the catalogue states by hand (an embedded image's body) must agree with
            // its own mask too.
            if let Some(format) = symbol.format {
                assert!(
                    symbol.accepts(precision_of_stored(format)),
                    "{:?} {} decodes {} but its mask does not accept {}",
                    entry.part,
                    symbol.name,
                    format.describe(),
                    precision_of_stored(format).name()
                );
            }
            let Some(format) = built_format(symbol.name) else {
                continue;
            };
            checked += 1;
            assert_eq!(
                symbol.format,
                Some(format),
                "{:?} {} does not carry the format its instance builds",
                entry.part,
                symbol.name
            );
        }
    }
    // The DeepSeek images are the ones built from the instance list; the embedded arms state
    // their formats by hand where a body has no instance. 205 of the catalogue's symbols are
    // built here, and a drop means the table stopped reaching the catalogue.
    assert!(
        checked >= 200,
        "only {checked} catalogue symbols are built by an instance"
    );
}

#[test]
fn the_mxfp4_experts_are_not_nvfp4_or_mxfp8() {
    // The correction this workstream makes: DeepSeek's routed experts are E2M1 codes with one
    // power-of-two scale a 32 values (kernels/b12x_mxfp4_moe_sm121.cu, opt_moe_sm121.cu and
    // routed_gate_up_sm121.cu all state it). The old catalogue listed the same bodies under
    // NVFP4, MXFP8 and FP8 because the mask was filled by position rather than by source.
    for entry in CATALOGUE.iter().filter(|entry| {
        matches!(
            entry.part,
            sparkflash_model::Part::Moe | sparkflash_model::Part::GroupedMoeGemm
        ) && entry.image == sparkflash_kernels::images::Image::Deepseek
    }) {
        for symbol in entry.symbols {
            let Some(format) = symbol.format else {
                continue;
            };
            // The format route's NVFP4 bodies are E2M1 codes too, with a per-16 scale and a
            // tensor scale: each 4-bit body accepts its own arithmetic and not the other's.
            if format.values() == sparkflash_model::format::FormatValues::Fp4E2M1 {
                let own = precision_of_stored(format);
                assert!(
                    symbol.accepts(own),
                    "{} decodes {} but does not accept {own:?}",
                    symbol.name,
                    format.describe()
                );
                for other in [Precision::MxFp4, Precision::NvFp4] {
                    assert!(
                        other == own || !symbol.accepts(other),
                        "{} decodes {} but also accepts {other:?}: a 4-bit format is its own \
                         arithmetic",
                        symbol.name,
                        format.describe()
                    );
                }
            }
        }
    }
}

#[test]
fn the_arithmetic_of_a_format_is_derived_not_folded() {
    use sparkflash_model::architecture::WeightFormat;

    assert_eq!(
        precision_of(WeightFormat::MxFp4 { block: 32 }),
        Precision::MxFp4
    );
    assert_eq!(
        precision_of(WeightFormat::NvFp4 {
            group_size: 16,
            wide: sparkflash_model::format::NvFp4Wide::Qwen
        }),
        Precision::NvFp4
    );
    assert_eq!(
        precision_of(WeightFormat::MxFp8 { block: 32 }),
        Precision::MxFp8
    );
    assert_eq!(
        precision_of(WeightFormat::Fp8Tile {
            block_m: 128,
            block_n: 128,
            scale_dtype: sparkflash_model::architecture::ScaleDType::F32,
            dynamic_activations: true,
        }),
        Precision::Fp8E4M3
    );
}
