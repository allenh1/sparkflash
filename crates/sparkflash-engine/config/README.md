# The runtime configuration in this directory

`deepseek_v41_flash.runtime.json` is the released DeepSeek-V4.1-Flash *runtime* configuration: the
flat document the reference implementation loads with `ModelArgs(**json.load(...))`
(`inference/generate.py:121-123`), not the released HF configuration, which nests everything under
`text_config`/`vision_config` and names several keys differently.

Provenance:

- Source: the checkpoint distribution's `inference/config.json`.
- Source sha256: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`.
- Copy: the same 64 keys with the same values, re-serialised with sorted keys and two-space indent,
  2,126 bytes. The value of each key is unchanged; only the JSON formatting differs.

The copy exists so the engine can be built and tested on a machine that does not have the
checkpoint. It is a configuration file, not weights: nothing in it is model data.
