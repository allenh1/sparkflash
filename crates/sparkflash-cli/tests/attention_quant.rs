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

//! The c4a-1 device projection backend quantises every activation once, with the engine's D8
//! quantiser, and hands those bytes to the same B12x kernel the m12/m13 harnesses use. The harness
//! quantises with `m4_quant::act_quant_fp8_row` instead, so this test pins the two implementations
//! to the same bytes over a fixture that covers positives, negatives and zero blocks.

use sparkflash_engine::quant::quantize_activation_fp8;
use sparkflash_kernels::m4_quant::act_quant_fp8_row;

#[test]
fn the_engine_d8_activation_bytes_match_the_b12x_harness_quantiser() {
    let mut values = Vec::new();
    for index in 0..4096 {
        let magnitude = match index % 7 {
            0 => 0.0,
            1 => 1e-4,
            2 => 1e-2,
            3 => 0.25,
            4 => 1.0,
            5 => 3.5,
            _ => 0.75,
        };
        let sign = if index % 3 == 0 { -1.0 } else { 1.0 };
        values.push(sign * magnitude * (1.0 + (index % 11) as f32 * 0.01));
    }
    let engine = quantize_activation_fp8(&values).expect("engine quantiser");
    let (harness_codes, harness_scales) = act_quant_fp8_row(&values);
    assert_eq!(
        engine.codes(),
        harness_codes.as_slice(),
        "E4M3 codes differ"
    );
    assert_eq!(
        engine.scales(),
        harness_scales.as_slice(),
        "E8M0 scale codes differ"
    );
}
