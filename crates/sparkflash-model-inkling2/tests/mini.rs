// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The looped release parses into the arm's typed configuration.

use sparkflash_model_inkling2::Inkling2Config;

const CONFIG: &str = include_str!("../../sparkflash-model/tests/data/inkling2_mini_config.json");
const QUANT: &str = include_str!("../../sparkflash-model/tests/data/inkling2_mini_quant.json");

#[test]
fn the_looped_release_parses_its_geometry() {
    let config =
        Inkling2Config::from_json(CONFIG, Some(QUANT)).expect("the looped document parses");
    assert_eq!(config.hidden_size, 2304);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.num_attention_heads, 24);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.n_routed_experts, 128);
    assert_eq!(config.moe_intermediate_size, 1152);
    assert_eq!(config.local_layer_ids.len(), 20);
    assert_eq!(config.full_layer_ids, vec![5, 11, 17, 23]);
    let mtp = config.mtp.as_ref().expect("the drafter");
    assert_eq!(mtp.num_hidden_layers, 3);
    let vision = config.vision.as_ref().expect("the tower");
    assert_eq!(vision.depth, 23);
    assert_eq!(vision.out_hidden_size, 2304);
}
