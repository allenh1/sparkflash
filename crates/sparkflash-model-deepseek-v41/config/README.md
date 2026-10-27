<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# The configuration documents in this directory

The checkpoint ships two configuration documents, and they are not the same file:

- `deepseek_v41_flash.runtime.json` - the released DeepSeek-V4.1-Flash *runtime* configuration: the
  flat document the reference implementation loads with `ModelArgs(**json.load(...))`
  (`inference/generate.py:121-123`).
- `deepseek_v41_flash.released.json` - the released HF `config.json`, the document the checkpoint
  carries at its root: the same model, nested under `text_config`/`vision_config` and with the HF
  key names (`hidden_size` for `dim`, `num_nextn_predict_layers` for `n_mtp_layers`, ...).
  `ModelConfig::from_released_json` maps it; `ModelConfig::from_json` accepts either document.

Provenance:

- `deepseek_v41_flash.runtime.json`: source the checkpoint distribution's `inference/config.json`,
  sha256 `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`. The same 64 keys with the
  same values, re-serialised with sorted keys and two-space indent, 2,126 bytes.
- `deepseek_v41_flash.released.json`: source the checkpoint's root `config.json`, sha256
  `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`. Byte-for-byte, 3,311 bytes.

The copies exist so the engine can be built and tested on a machine that does not have the
checkpoint. They are configuration files, not weights: nothing in them is model data.
